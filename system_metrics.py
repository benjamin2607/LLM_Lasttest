"""
Host and GPU metrics shared by the load test and the live monitor.

GPU values are read via ``nvidia-smi`` (installed with the NVIDIA driver), host
RAM/CPU values via ``psutil`` (a dependency of Locust). If either source is
unavailable, the corresponding fields are reported as ``None`` and callers keep
running.
"""
import shutil
import statistics
import subprocess
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

GPU_FIELDS = ["utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw"]


def _num(value):
    """Convert a value to float, returning None for empty or non-numeric input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_gpu(gpu_index=0):
    """Return a single GPU reading, or None if nvidia-smi is unavailable or fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, f"--id={gpu_index}", f"--query-gpu={','.join(GPU_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    util, used, total, temp, power = (_num(v.strip()) for v in out.splitlines()[0].split(",")[:5])
    vram_pct = round(used / total * 100, 1) if used is not None and total else None
    return {"gpu_util": util, "vram_used_mb": used, "vram_total_mb": total,
            "vram_pct": vram_pct, "gpu_temp": temp, "gpu_power_w": power}


def read_host():
    """Return host RAM and CPU usage, or an empty dict if psutil is unavailable."""
    if psutil is None:
        return {}
    mem = psutil.virtual_memory()
    return {"ram_pct": mem.percent, "ram_used_gb": round(mem.used / 1024 ** 3, 1),
            "cpu_pct": psutil.cpu_percent(interval=None)}


class SystemSampler(threading.Thread):
    """Background thread that samples GPU, RAM and CPU every ``interval`` seconds."""

    def __init__(self, interval=2.0, gpu_index=0):
        super().__init__(daemon=True)
        self.interval = interval
        self.gpu_index = gpu_index
        self.lock = threading.Lock()
        self.samples = []
        self.latest = {}
        self.gpu_available = read_gpu(gpu_index) is not None
        self._stop = threading.Event()
        if psutil:
            psutil.cpu_percent(interval=None)  # Prime the counter; the first call always returns 0.0.

    def run(self):
        while not self._stop.is_set():
            sample = {"ts": time.time(), **(read_gpu(self.gpu_index) or {}), **read_host()}
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
