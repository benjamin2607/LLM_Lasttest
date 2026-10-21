#!/usr/bin/env python3
"""
Privacy-preserving live monitor for LM Studio. Runs entirely on the local machine.

Data sources:
  1. LM Studio server log, followed like ``tail -f`` (daily log rotation is handled)
  2. Local LM Studio REST API (loaded models, optional synthetic probe request)
  3. GPU and host metrics (NVIDIA via nvidia-smi, macOS via ioreg; detected automatically) and psutil

Only numeric metrics (request counts, latencies, token counts, hardware load) are
kept. LM Studio writes prompt and completion text to its log; this monitor
discards that content while parsing and never displays or stores it. No IP
addresses, user names or chat identifiers are collected.

Outputs:
  * Real-time web dashboard on http://127.0.0.1:8765 (updated every second)
  * One aggregated CSV row per interval in ``monitoring/lmstudio_metrics_<date>.csv``
  * One summary line per interval on stdout

Example:
    python lmstudio_monitor.py --interval 60 --alert-p95 45
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lmstudio_log_parser import LogParser, diagnose, rnd
from system_metrics import SystemSampler

DEFAULT_LOG_DIRS = [
    Path.home() / ".lmstudio" / "server-logs",
    Path.home() / ".cache" / "lm-studio" / "server-logs",
]


class LogFollower(threading.Thread):
    """Continuously follows the newest LM Studio log file and feeds lines to a LogParser."""

    def __init__(self, target, parser, from_start=False, poll=0.2):
        super().__init__(daemon=True)
        self.target = Path(target)
        self.parser = parser
        self.from_start = from_start
        self.poll = poll
        self.current = None
        self._fh = None
        self._stop = threading.Event()

    def _log_files(self):
        if self.target.is_file():
            return [self.target]
        return sorted((p for p in self.target.rglob("*.log") if p.is_file()), key=lambda p: p.stat().st_mtime)

    def newest_file(self):
        logs = self._log_files()
        return logs[-1] if logs else None

    def open_initial(self, replay_window=0, sample=None):
        """
        Open the newest log before following it: learn which models are loaded and, with
        ``replay_window`` > 0, rebuild the last ``replay_window`` seconds from the log
        (including the previous file when the window reaches back past a log rotation).
        """
        logs = self._log_files()
        if not logs:
            return
        newest = logs[-1]
        fh = open(newest, "r", encoding="utf-8", errors="replace")
        self._fh, self.current = fh, newest
        print(f"[monitor] following log: {newest.name}")
        if self.from_start:
            return
        lines = fh.read().splitlines()  # Leaves the file position at the end for following.
        self.parser.prime_engines(lines)
        if replay_window > 0:
            now = time.time()
            if len(logs) > 1 and logs[-2].stat().st_mtime >= now - replay_window:
                lines = logs[-2].read_text(encoding="utf-8", errors="replace").splitlines() + lines
            self.parser.replay(lines, now - replay_window, now, sample)

    def run(self):
        fh, buf, last_check, first = self._fh, "", 0.0, self._fh is None
        while not self._stop.is_set():
            if fh is None or time.time() - last_check > 5:
                last_check = time.time()
                newest = self.newest_file()
                if newest and newest != self.current:
                    if fh:
                        for line in fh:  # Drain the remainder of the previous file before switching.
                            self.parser.feed(line.rstrip("\r\n"))
                        fh.close()
                    fh = open(newest, "r", encoding="utf-8", errors="replace")
                    if first and not self.from_start:
                        self.parser.prime_engines(fh)  # Learn loaded models, then follow new lines only.
                        fh.seek(0, os.SEEK_END)
                    first, buf = False, ""
                    self.current = newest
                    print(f"[monitor] following log: {newest.name}")
            if fh is None:
                self._stop.wait(2)
                continue
            chunk = fh.readline()
            if not chunk:
                try:
                    if fh.tell() > self.current.stat().st_size:  # File was truncated.
                        fh.seek(0)
                except OSError:
                    pass
                self._stop.wait(self.poll)
                continue
            buf += chunk
            if buf.endswith("\n"):
                self.parser.feed(buf.rstrip("\r\n"))
                buf = ""

    def stop(self):
        self._stop.set()


def find_log_target(explicit=None):
    """Resolve the log file or directory: explicit path, LMSTUDIO_LOG_DIR, then default locations."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("LMSTUDIO_LOG_DIR")
    if env:
        return Path(env).expanduser()
    for d in DEFAULT_LOG_DIRS:
        if d.is_dir():
            return d
    return None


class LMStudioWatcher:
    """Convenience wrapper (parser + follower) for use from other scripts such as auto_loadtest.py."""

    def __init__(self, target, from_start=False):
        self.parser = LogParser(use_log_time=from_start)
        self.follower = LogFollower(target, self.parser, from_start=from_start)

    def start(self):
        self.follower.start()
        return self

    def snapshot(self, reset=True):
        return self.parser.snapshot(reset=reset)

    def stop(self):
        self.follower.stop()


# --- LM Studio REST API (local only) ---
def api_request(url, token=None, payload=None, timeout=5):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def loaded_models(base, token=None):
    """Return IDs of loaded LLMs, or None if the API is unreachable."""
    try:
        data = api_request(f"{base}/api/v0/models", token).get("data", [])
        return [m["id"] for m in data if m.get("state") == "loaded" and m.get("type") != "embeddings"]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def probe(base, model, token=None):
    """Send a short synthetic request with a fixed message (no user data) and measure response time."""
    payload = {"model": model, "stream": False, "max_tokens": 16, "temperature": 0,
               "messages": [{"role": "user", "content": "Reply with the single word OK."}]}
    t0 = time.time()
    try:
        res = api_request(f"{base}/api/v0/chat/completions", token, payload, timeout=120)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"probe_ok": False, "probe_error": type(exc).__name__}
    stats = res.get("stats", {})
    return {"probe_ok": True, "probe_latency_s": rnd(time.time() - t0),
            "probe_ttft_s": rnd(stats.get("time_to_first_token")),
            "probe_tokens_per_s": rnd(stats.get("tokens_per_second"))}


# --- Real-time recording (one data point per second) ---
HARDWARE_FIELDS = ("gpu_util", "vram_pct", "vram_used_mb", "vram_total_mb",
                   "gpu_temp", "gpu_power_w", "ram_pct", "ram_used_gb", "cpu_pct")
HISTORY_NUMERIC = ("requests_completed", "requests_started", "max_active", "latency_p95_s", "latency_avg_s",
                   "ttft_p95_s", "gen_tokens_per_s_total", "gpu_util_avg", "vram_pct_max")


class LiveRecorder(threading.Thread):
    """Collects log, hardware and API state once per second and notifies dashboard clients."""

    SAVE_EVERY_S = 15

    def __init__(self, parser, sampler, args, follower=None, keep_s=1800):
        super().__init__(daemon=True)
        self.parser, self.sampler, self.args, self.follower = parser, sampler, args, follower
        self.series = deque(maxlen=keep_s)
        self.history = deque(maxlen=24 * 3600 // max(args.interval, 1))  # Per-interval rows (daily view).
        self.cond = threading.Condition()
        self.seq = 0
        self.state = {}
        self.models, self.api_ok, self._last_api = [], None, 0.0
        self.started = time.time()
        self.keep_s = keep_s
        self.hardware_file = Path(args.out) / ".live_hardware.json"
        self._saved_hardware = self._load_hardware()
        self._last_save = time.time()

    # --- restoring state after a restart ---
    def _load_hardware(self):
        """Hardware points saved by a previous monitor run, keyed by second."""
        try:
            points = json.loads(self.hardware_file.read_text(encoding="utf-8")).get("points", [])
        except (OSError, ValueError):
            return {}
        cutoff = time.time() - self.keep_s
        return {int(p["t"]): p for p in points if p.get("t", 0) >= cutoff}

    def save_hardware(self):
        """Persist the hardware part of the live series (numbers only) for the next start."""
        with self.cond:
            points = [{"t": p["t"], **{k: p.get(k) for k in HARDWARE_FIELDS}} for p in self.series]
        try:
            tmp = self.hardware_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"points": points}), encoding="utf-8")
            os.replace(tmp, self.hardware_file)
        except OSError:
            pass

    def replay_sample(self, second):
        """Called by the log replay once per second: rebuild one data point of the live series."""
        live = self.parser.live()
        hw = self._saved_hardware.get(second, {})
        self.series.append({"t": float(second), "active": live["active"], "waiting": live["waiting"],
                            "tokens_per_s": live["tokens_per_s"], "requests_per_min": live["requests_per_min"],
                            **{k: hw.get(k) for k in HARDWARE_FIELDS}})

    def load_history(self, out_dir):
        """Fill the daily view from the CSV rows of the last 24 hours."""
        cutoff = datetime.now().timestamp() - 24 * 3600
        rows = []
        for path in sorted(Path(out_dir).glob("lmstudio_metrics_*.csv"))[-2:]:
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            if datetime.fromisoformat(row["time"]).timestamp() < cutoff:
                                continue
                        except (KeyError, ValueError):
                            continue
                        for key in HISTORY_NUMERIC:
                            try:
                                row[key] = float(row[key]) if row.get(key) not in (None, "") else None
                            except ValueError:
                                row[key] = None
                        rows.append(row)
            except OSError:
                continue
        with self.cond:
            self.history.extend(rows[-self.history.maxlen:])

    def _poll_api(self):
        # Every poll adds a line to the LM Studio log, so poll sparingly; loaded models rarely change.
        if self.args.no_api or time.time() - self._last_api < 30:
            return
        self._last_api = time.time()
        models = loaded_models(self.args.lmstudio_url, self.args.token)
        self.api_ok, self.models = models is not None, models or []
        # Lets the parser name the model of app chats when it cannot be derived from the log.
        self.parser.default_model = self.models[0] if len(self.models) == 1 else None

    def alerts(self, live, sysm):
        a, args = [], self.args
        if sysm.get("vram_pct") is not None and sysm["vram_pct"] >= args.alert_vram_pct:
            a.append(f"VRAM {sysm['vram_pct']} %")
        if sysm.get("gpu_temp") is not None and sysm["gpu_temp"] >= args.alert_gpu_temp:
            a.append(f"GPU temperature {sysm['gpu_temp']:.0f} °C")
        if live["latency_p95_s"] is not None and live["latency_p95_s"] >= args.alert_p95:
            a.append(f"p95 response time {live['latency_p95_s']} s")
        if args.alert_active and live["active"] >= args.alert_active:
            a.append(f"{live['active']} concurrent requests")
        if self.api_ok is False:
            a.append("LM Studio API unreachable")
        return a

    def run(self):
        while True:
            self._poll_api()
            live, sysm = self.parser.live(), dict(self.sampler.latest)
            point = {"t": round(time.time(), 1), "active": live["active"], "waiting": live["waiting"],
                     "tokens_per_s": live["tokens_per_s"], "requests_per_min": live["requests_per_min"],
                     **{k: sysm.get(k) for k in HARDWARE_FIELDS}}
            state = {**live, "system": sysm, "alerts": self.alerts(live, sysm), "models": self.models,
                     "api_ok": self.api_ok, "gpu_available": self.sampler.gpu_available,
                     **self.sampler.gpu.info(),
                     "log_file": self.follower.current.name if self.follower and self.follower.current else None,
                     "monitor_started": self.started}
            with self.cond:
                self.series.append(point)
                self.state = state
                self.seq += 1
                self.cond.notify_all()
            if time.time() - self._last_save >= self.SAVE_EVERY_S:
                self._last_save = time.time()
                self.save_hardware()
            time.sleep(1)

    def full(self):
        """Complete state for the initial page load."""
        with self.cond:
            return {"series": list(self.series), "history": list(self.history), "state": self.state,
                    "latency_points": self.parser.latency_points(),
                    "config": {"interval": self.args.interval, "alert_p95": self.args.alert_p95,
                               "alert_vram_pct": self.args.alert_vram_pct,
                               "alert_gpu_temp": self.args.alert_gpu_temp, "alert_active": self.args.alert_active,
                               "stall_after": self.parser.stall_after}}


# --- Load test control (dashboard stop button) ---
class LoadTestControl:
    """
    Stops a running auto_loadtest.py run and deletes the AnythingLLM threads it created.

    The load test publishes its state in ``.loadtest_status.json`` and watches for
    ``.loadtest_stop``; the threads it created are listed in ``.loadtest_threads.json``.
    Only threads from that list are deleted, so chats of real users are never touched.
    """

    HEARTBEAT_TIMEOUT_S = 8
    STOP_WAIT_S = 45

    def __init__(self, project_dir, anythingllm_url, api_key):
        project_dir = Path(project_dir)
        self.status_file = project_dir / ".loadtest_status.json"
        self.stop_file = project_dir / ".loadtest_stop"
        self.registry_file = project_dir / ".loadtest_threads.json"
        self.url = anythingllm_url.rstrip("/")
        self.api_key = api_key
        self.lock = threading.Lock()
        self.progress = {"phase": "idle"}

    def loadtest(self):
        try:
            status = json.loads(self.status_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"running": False}
        running = time.time() - status.get("heartbeat", 0) < self.HEARTBEAT_TIMEOUT_S
        return {"running": running, **{k: status.get(k) for k in ("run", "stage", "users")}}

    def threads(self):
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8")).get("threads", [])
        except (OSError, ValueError):
            return []

    def state(self):
        with self.lock:
            progress = dict(self.progress)
        return {"loadtest": self.loadtest(), "threads": len(self.threads()),
                "can_delete": bool(self.api_key), "progress": progress}

    def _set(self, **progress):
        with self.lock:
            self.progress = {"at": time.time(), **progress}
        print(f"[monitor] stop & delete: {progress.get('message', progress.get('phase'))}")

    def start(self):
        """Run the stop-and-delete sequence in the background; returns False if one is already running."""
        with self.lock:
            if self.progress.get("phase") in ("stopping", "deleting"):
                return False
            self.progress = {"phase": "stopping", "at": time.time(), "message": "stopping the load test"}
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        if self.loadtest()["running"]:
            self.stop_file.write_text(str(time.time()), encoding="utf-8")
            deadline = time.time() + self.STOP_WAIT_S
            while self.loadtest()["running"] and time.time() < deadline:
                time.sleep(1)
            if self.loadtest()["running"]:
                self._set(phase="error", message="the load test did not stop within "
                                                 f"{self.STOP_WAIT_S} s; no threads were deleted")
                return
        try:
            self.stop_file.unlink()
        except OSError:
            pass
        if not self.api_key:
            self._set(phase="done", deleted=0, failed=0,
                      message="load test stopped; set ANYTHING_API_KEY for the monitor to delete threads")
            return
        self._set(phase="deleting", message="deleting load-test threads")
        remaining, deleted = [], 0
        for entry in self.threads():
            req = urllib.request.Request(f"{self.url}/api/v1/workspace/{entry['workspace']}/thread/{entry['slug']}",
                                         method="DELETE", headers={"Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=15):
                    deleted += 1
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    deleted += 1  # Already deleted in AnythingLLM.
                else:
                    remaining.append(entry)
            except (urllib.error.URLError, OSError):
                remaining.append(entry)
        tmp = self.registry_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"threads": remaining}, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.registry_file)
        message = f"deleted {deleted} load-test threads" + (f", {len(remaining)} failed" if remaining else "")
        self._set(phase="done" if not remaining else "error", deleted=deleted, failed=len(remaining), message=message)


# --- Local dashboard server (bound to 127.0.0.1 only) ---
DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard.html"


def start_dashboard(port, recorder, control):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(DASHBOARD_FILE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(json.dumps({**recorder.full(), "control": control.state()}).encode(), "application/json")
            elif path == "/api/stream":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self):
            """
            Remove hung requests from tracking. A custom header is required: browsers only send it
            from the dashboard's own origin (other sites would need a CORS preflight, which this server
            never grants), so third-party pages cannot trigger it.
            """
            if self.headers.get("X-Monitor-Action") not in ("dismiss", "stop"):
                self.send_error(403)
                return
            path = self.path.split("?")[0].rstrip("/")
            if path == "/api/system/stop":
                started = control.start()
                self._send(json.dumps({"started": started}).encode(), "application/json")
                return
            if path == "/api/requests/dismiss-stalled":
                removed, what = recorder.parser.dismiss(stalled_only=True), "stalled requests"
            elif path.startswith("/api/requests/") and path.endswith("/dismiss") and path.split("/")[3].isdigit():
                seq = int(path.split("/")[3])
                removed, what = recorder.parser.dismiss(seq=seq), f"request #{seq}"
            else:
                self.send_error(404)
                return
            print(f"[monitor] removed {removed} {what} from tracking (dashboard)")
            self._send(json.dumps({"removed": removed}).encode(), "application/json")

        def _stream(self):
            """Server-Sent Events: push the latest data point and state once per second."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    with recorder.cond:
                        recorder.cond.wait_for(lambda: recorder.seq != seq, timeout=15)
                        seq = recorder.seq
                        msg = {"point": recorder.series[-1] if recorder.series else None,
                               "state": recorder.state, "history_len": len(recorder.history)}
                    msg["control"] = control.state()
                    self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # Client disconnected.

        def log_message(self, *args):
            pass  # No access logging (and therefore no client IPs).

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser(description="Privacy-preserving live monitor for LM Studio")
    ap.add_argument("--log", help="Log file or directory (default: ~/.lmstudio/server-logs)")
    ap.add_argument("--lmstudio-url", default="http://localhost:1234")
    ap.add_argument("--token", default=os.environ.get("LMSTUDIO_API_TOKEN"),
                    help="Only required if 'Require API token' is enabled in LM Studio (or set LMSTUDIO_API_TOKEN)")
    ap.add_argument("--interval", type=int, default=60, help="Aggregation interval for CSV and daily view, in seconds")
    ap.add_argument("--out", default="monitoring", help="Output directory for CSV files")
    ap.add_argument("--dashboard-port", type=int, default=8765, help="Local dashboard port (0 = disabled)")
    ap.add_argument("--no-browser", action="store_true", help="Do not open the dashboard in a browser automatically")
    ap.add_argument("--probe-interval", type=int, default=0,
                    help="Send a synthetic probe request every N seconds (0 = disabled)")
    ap.add_argument("--from-start", action="store_true", help="Parse the log from the beginning instead of new lines only")
    ap.add_argument("--no-api", action="store_true", help="Do not query the LM Studio API")
    ap.add_argument("--anythingllm-url", default="http://localhost:3001",
                    help="AnythingLLM URL for deleting load-test threads (API key from env ANYTHING_API_KEY)")
    ap.add_argument("--project-dir", default=str(Path(__file__).resolve().parent),
                    help="Folder of auto_loadtest.py (status, stop and thread registry files)")
    ap.add_argument("--stall-minutes", type=float, default=5,
                    help="Mark a running request as stalled after N minutes without any log activity")
    ap.add_argument("--replay-minutes", type=int, default=30,
                    help="Rebuild the dashboard from the last N minutes of the log on start (0 = off)")
    ap.add_argument("--diagnose", action="store_true",
                    help="Print which LM Studio log entry types are recognised (no content) and exit")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--alert-vram-pct", type=float, default=95.0)
    ap.add_argument("--alert-gpu-temp", type=float, default=85.0)
    ap.add_argument("--alert-p95", type=float, default=60.0, help="Alert threshold for p95 latency, in seconds")
    ap.add_argument("--alert-active", type=int, default=0, help="Alert at N concurrent requests (0 = disabled)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # Flush every line, also when redirected to a file.

    target = find_log_target(args.log)
    if args.diagnose:
        newest = LogFollower(target, None).newest_file() if target and target.exists() else None
        if newest is None:
            sys.exit("[monitor] No LM Studio log file found. Pass it with --log PATH.")
        print("\n".join(diagnose(newest)))
        return
    if target is None or not target.exists():
        print("[monitor] WARNING: LM Studio log directory not found. Pass it with --log PATH. "
              "Only GPU and API metrics will be collected.")
        target = None

    parser = LogParser(use_log_time=args.from_start, stall_after=args.stall_minutes * 60)
    follower = LogFollower(target, parser, from_start=args.from_start) if target else None
    sampler = SystemSampler(interval=1.0, gpu_index=args.gpu_index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = LiveRecorder(parser, sampler, args, follower)
    recorder.load_history(out_dir)
    if follower:
        # Restore the recent past so the dashboard is filled right after a restart.
        t0 = time.time()
        replay_window = 0 if args.from_start else args.replay_minutes * 60
        follower.open_initial(replay_window=replay_window, sample=recorder.replay_sample)
        if replay_window:
            print(f"[monitor] restored the last {args.replay_minutes} min from the log "
                  f"({len(parser.recent)} completed requests, {time.time() - t0:.1f}s)")
        follower.start()
    sampler.start()
    gpu = sampler.gpu.info()
    if sampler.gpu_available:
        print(f"[monitor] GPU detected: {gpu['gpu_name'] or 'unknown'} ({gpu['gpu_backend']})")
    else:
        print("[monitor] Note: no supported GPU metrics source found, GPU fields stay empty.")
    recorder.start()

    if args.dashboard_port:
        try:
            control = LoadTestControl(args.project_dir, args.anythingllm_url, os.environ.get("ANYTHING_API_KEY"))
            start_dashboard(args.dashboard_port, recorder, control)
        except OSError as exc:
            sys.exit(f"[monitor] ERROR: cannot bind dashboard to 127.0.0.1:{args.dashboard_port} ({exc.strerror}). "
                     "Is another monitor instance running? Choose a different port with --dashboard-port.")
        url = f"http://127.0.0.1:{args.dashboard_port}"
        print(f"[monitor] Dashboard: {url}  (local access only)")
        if not args.no_browser:
            import webbrowser
            webbrowser.open(url)

    print(f"[monitor] running, interval {args.interval}s, CSV output in {out_dir.resolve()}  (Ctrl+C to stop)")
    last_probe = 0.0
    try:
        while True:
            time.sleep(args.interval)
            now = datetime.now()
            row = {"time": now.isoformat(timespec="seconds"), **parser.snapshot(reset=True),
                   **sampler.summary(reset=True)}
            if not args.no_api:
                row["lmstudio_api_ok"] = recorder.api_ok
                row["loaded_models"] = ";".join(recorder.models)
                if args.probe_interval and recorder.models and time.time() - last_probe >= args.probe_interval:
                    last_probe = time.time()
                    row.update(probe(args.lmstudio_url, recorder.models[0], args.token))
            alerts = recorder.state.get("alerts", [])
            row["alerts"] = "WARNING: " + ", ".join(alerts) if alerts else ""
            with recorder.cond:
                recorder.history.append(row)

            csv_path = out_dir / f"lmstudio_metrics_{now:%Y-%m-%d}.csv"
            new_file = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow(row)

            dash = lambda v: "-" if v is None else v
            mem_label = "GPU mem" if gpu["unified_memory"] else "VRAM"
            print(f"{now:%H:%M:%S}  active {row['active_now']:>2} (max {row['max_active']:>2})  "
                  f"done {row['requests_completed']:>3}  p95 {dash(row['latency_p95_s']):>6}s  "
                  f"tok/s {dash(row['gen_tokens_per_s_total']):>6}  GPU {dash(row['gpu_util_avg']):>5}%  "
                  f"{mem_label} {dash(row['vram_pct_max']):>5}%  RAM {dash(row['ram_pct_max']):>5}%  {row['alerts']}")
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")
    finally:
        recorder.save_hardware()
        sampler.stop()
        if follower:
            follower.stop()


if __name__ == "__main__":
    main()
