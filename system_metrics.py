"""
Host and GPU metrics shared by the load test and the live monitor.

The GPU backend is detected automatically at start-up:

* ``nvidia`` - NVIDIA GPUs via ``nvidia-smi`` (Linux/Windows, installed with the driver).
  Provides utilisation, memory, temperature and power draw.
* ``apple``  - macOS (Apple Silicon and Intel Macs) via ``ioreg``, which needs no
  administrator rights. Provides utilisation and GPU memory. Temperature and
  power draw require ``sudo powermetrics`` on macOS and are therefore reported
  as ``None``.

Host RAM/CPU values come from ``psutil`` (a dependency of Locust). Any metric
that is unavailable on the current platform is reported as ``None``; callers
keep running and the dashboard leaves the field empty.
"""
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

NVIDIA_FIELDS = ["utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw"]

# ioreg PerformanceStatistics keys differ between Apple Silicon and AMD/Intel GPUs in Macs.
APPLE_UTIL_KEYS = ("Device Utilization %", "GPU Activity(%)", "GPU Core Utilization")
APPLE_MEM_KEYS = ("Alloc system memory", "In use system memory", "vramUsedBytes")
IOREG_PAIR_RE = re.compile(r'"([^"]+)"=(\d+)')
IOREG_MODEL_RE = re.compile(r'"model" = (?:<)?"([^"]+)"')


def _num(value):
    """Convert a value to float, returning None for empty or non-numeric input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run(cmd, timeout=5):
    """Run a command and return its stdout, or None on any failure."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError):
        return None


# --- NVIDIA ---
def _read_nvidia(gpu_index=0):
    exe = shutil.which("nvidia-smi")
    out = _run([exe, f"--id={gpu_index}", f"--query-gpu={','.join(NVIDIA_FIELDS)}",
                "--format=csv,noheader,nounits"]) if exe else None
    if not out or not out.strip():
        return None
    util, used, total, temp, power = (_num(v.strip()) for v in out.strip().splitlines()[0].split(",")[:5])
    vram_pct = round(used / total * 100, 1) if used is not None and total else None
    return {"gpu_util": util, "vram_used_mb": used, "vram_total_mb": total,
            "vram_pct": vram_pct, "gpu_temp": temp, "gpu_power_w": power}


def _nvidia_name(gpu_index=0):
    exe = shutil.which("nvidia-smi")
    out = _run([exe, f"--id={gpu_index}", "--query-gpu=name", "--format=csv,noheader"]) if exe else None
    return out.strip().splitlines()[0] if out and out.strip() else None


# --- Apple (macOS) ---
def _sysctl_int(name):
    out = _run(["sysctl", "-n", name])
    return int(out.strip()) if out and out.strip().isdigit() else None


def _apple_gpu_memory_limit_mb():
    """
    Memory the GPU may use on a Mac with unified memory, in MB.

    Uses the ``iogpu.wired_limit_mb`` override if set; otherwise macOS defaults to
    roughly two thirds of RAM on machines with up to 36 GB and three quarters above.
    """
    override = _sysctl_int("iogpu.wired_limit_mb")
    if override:
        return float(override)
    ram = _sysctl_int("hw.memsize")
    if not ram:
        return None
    ram_mb = ram / 1024 ** 2
    return round(ram_mb * (2 / 3 if ram_mb <= 36 * 1024 else 3 / 4))


def _ioreg_accelerators():
    """Return the raw ioreg output for all GPU accelerators, or None."""
    return _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])


def _read_apple(mem_limit_mb=None):
    out = _ioreg_accelerators()
    if not out:
        return None
    stats_line = next((line for line in out.splitlines() if '"PerformanceStatistics"' in line), None)
    if stats_line is None:
        return None
    stats = {k: int(v) for k, v in IOREG_PAIR_RE.findall(stats_line)}
    util = next((float(stats[k]) for k in APPLE_UTIL_KEYS if k in stats), None)
    used_bytes = next((stats[k] for k in APPLE_MEM_KEYS if stats.get(k)), None)
    used = round(used_bytes / 1024 ** 2) if used_bytes is not None else None
    vram_pct = round(used / mem_limit_mb * 100, 1) if used is not None and mem_limit_mb else None
    return {"gpu_util": util, "vram_used_mb": used, "vram_total_mb": mem_limit_mb,
            "vram_pct": vram_pct, "gpu_temp": None, "gpu_power_w": None}


def _apple_name():
    out = _ioreg_accelerators() or ""
    m = IOREG_MODEL_RE.search(out)
    if m:
        return m.group(1)
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return brand.strip() if brand else None


class GpuReader:
    """Detects the available GPU backend once and reads metrics from it."""

    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.backend, self.name, self.unified_memory = None, None, False
        self._apple_limit = None
        if _read_nvidia(gpu_index) is not None:
            self.backend, self.name = "nvidia", _nvidia_name(gpu_index)
        elif sys.platform == "darwin":
            self._apple_limit = _apple_gpu_memory_limit_mb()
            if _read_apple(self._apple_limit) is not None:
                self.backend, self.name = "apple", _apple_name()
                self.unified_memory = True

    def read(self):
        if self.backend == "nvidia":
            return _read_nvidia(self.gpu_index)
        if self.backend == "apple":
            return _read_apple(self._apple_limit)
        return None

    def info(self):
        """Static description of the detected hardware, for display purposes."""
        return {"gpu_backend": self.backend, "gpu_name": self.name,
                "unified_memory": self.unified_memory, "platform": platform.system()}


def read_host():
    """Return host RAM and CPU usage, or an empty dict if psutil is unavailable."""
    if psutil is None:
        return {}
    mem = psutil.virtual_memory()
    return {"ram_pct": mem.percent, "ram_used_gb": round(mem.used / 1024 ** 3, 1),
            "cpu_pct": psutil.cpu_percent(interval=None)}


class SystemSampler(threading.Thread):
    """Background thread that samples GPU, RAM, and CPU every ``interval`` seconds."""

    def __init__(self, interval=2.0, gpu_index=0):
        super().__init__(daemon=True)
        self.interval = interval
        self.lock = threading.Lock()
        self.samples = []
        self.latest = {}
        self.gpu = GpuReader(gpu_index)
        self.gpu_available = self.gpu.backend is not None
        self._stop = threading.Event()
        if psutil:
            psutil.cpu_percent(interval=None)  # Prime the counter; the first call always returns 0.0.

    def run(self):
        while not self._stop.is_set():
            sample = {"ts": time.time(), **(self.gpu.read() or {}), **read_host()}
            with self.lock:
                self.samples.append(sample)
                self.latest = sample
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def summary(self, reset=True):
        """Return averages and maxima of all samples since the last reset."""
        with self.lock:
            samples = self.samples
            if reset:
                self.samples = []

        def vals(key):
            return [s[key] for s in samples if s.get(key) is not None]

        def avg(key):
            v = vals(key)
            return round(statistics.fmean(v), 1) if v else None

        def mx(key):
            v = vals(key)
            return round(max(v), 1) if v else None

        return {
            "gpu_util_avg": avg("gpu_util"), "gpu_util_max": mx("gpu_util"),
            "vram_used_mb_max": mx("vram_used_mb"), "vram_pct_max": mx("vram_pct"),
            "gpu_temp_max": mx("gpu_temp"), "gpu_power_w_avg": avg("gpu_power_w"),
            "ram_pct_max": mx("ram_pct"), "ram_used_gb_max": mx("ram_used_gb"),
            "cpu_pct_avg": avg("cpu_pct"),
        }
