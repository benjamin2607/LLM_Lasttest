#!/usr/bin/env python3
"""
Privacy-preserving live monitor for LM Studio. Runs entirely on the local machine.

Data sources:
  1. LM Studio server log, followed like ``tail -f`` (daily log rotation is handled)
  2. Local LM Studio REST API (loaded models, optional synthetic probe request)
  3. GPU and host metrics via nvidia-smi and psutil

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
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from system_metrics import SystemSampler

# --- Log patterns (LM Studio 0.3.x). Adjust here if a newer release changes the log format. ---
TS_RE = re.compile(r"^\[?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
LEVEL_RE = re.compile(r"\[(INFO|DEBUG|WARN|WARNING|ERROR)\]")
# Matches both "Received POST request to /v1/..." and "Received request: POST to /v1/...".
REQUEST_RE = re.compile(r"Received (?:request: )?(POST|GET) (?:request )?to (\S+)", re.I)
FIRST_TOKEN_RE = re.compile(r"First token generated", re.I)
STREAM_DONE_RE = re.compile(r"Finished streaming response", re.I)
PREDICTION_RE = re.compile(r"Generated prediction", re.I)
# Only these fields are extracted from JSON bodies; all other content is ignored.
MODEL_RE = re.compile(r'"model"\s*:\s*"([^"]{1,200})"')
PROMPT_TOK_RE = re.compile(r'"prompt_tokens"\s*:\s*(\d+)')
COMPL_TOK_RE = re.compile(r'"completion_tokens"\s*:\s*(\d+)')
TPS_RE = re.compile(r'"tokens_per_second"\s*:\s*([\d.]+)')

DEFAULT_LOG_DIRS = [
    Path.home() / ".lmstudio" / "server-logs",
    Path.home() / ".cache" / "lm-studio" / "server-logs",
]


def classify_endpoint(endpoint):
    """Map an API path to 'embedding', 'generation' or None (not tracked)."""
    if "embedding" in endpoint:
        return "embedding"
    if "completions" in endpoint or "/responses" in endpoint or "/chat" in endpoint:
        return "generation"
    return None


def pct(values, p):
    """Nearest-rank percentile; returns None for an empty sequence."""
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(p / 100 * len(values)) - 1))
    return values[k]


def rnd(value, digits=2):
    return round(value, digits) if value is not None else None


class LogParser:
    """
    Turns LM Studio log lines into request metrics.

    LM Studio does not log a request ID, so concurrent requests are matched
    start-to-finish in FIFO order. Counters and token totals are exact;
    per-request latencies are approximations under high concurrency.
    """

    def __init__(self, use_log_time=False, stale_after=900):
        self.lock = threading.Lock()
        self.use_log_time = use_log_time
        self.stale_after = stale_after
        self.open = deque()              # In-flight generation requests (FIFO).
        self.ctx = None                  # JSON body currently being read: "request", "prediction" or None.
        self.ctx_req = None
        self.just_streamed = False
        self.last_finished = None
        self.last_ts = time.time()
        self.lines_seen = 0
        self.last_line_wall = None       # Wall-clock time of the most recent log line.
        self.recent = deque(maxlen=500)  # Recently completed requests (numeric fields only).
        self.totals = {"started": 0, "completed": 0, "embeddings": 0, "errors": 0,
                       "prompt_tokens": 0, "completion_tokens": 0}
        self._new_window()

    def _new_window(self):
        self.w_start = self.last_ts if self.use_log_time else time.time()
        self.w = {"started": 0, "completed": 0, "embeddings": 0, "errors": 0, "stale": 0,
                  "max_active": len(self.open), "latencies": [], "ttfts": [], "tps": [],
                  "prompt_tokens": 0, "completion_tokens": 0, "models": {}}

    def _inc(self, key, n=1):
        """Increment a counter in both the current window and the running totals."""
        self.w[key] += n
        self.totals[key] += n

    def _now(self, ts_match):
        if self.use_log_time and ts_match:
            try:
                self.last_ts = datetime.strptime(ts_match.group(1).replace("T", " "),
                                                 "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                pass
            return self.last_ts
        return time.time()

    def feed(self, line):
        """Process a single log line."""
        with self.lock:
            self.lines_seen += 1
            self.last_line_wall = time.time()
            ts = TS_RE.match(line)
            if not ts and not LEVEL_RE.search(line[:60]):
                self._scan_body(line)  # Continuation line of a multi-line JSON body.
                return
            now = self._now(ts)
            streamed, self.just_streamed = self.just_streamed, False
            self.ctx = self.ctx_req = None

            rq = REQUEST_RE.search(line)
            if rq:
                method, endpoint = rq.groups()
                kind = classify_endpoint(endpoint) if method.upper() == "POST" else None
                if kind == "embedding":
                    self._inc("embeddings")
                elif kind == "generation":
                    req = {"start": now, "first": None, "model": None}
                    self.open.append(req)
                    self._inc("started")
                    self.w["max_active"] = max(self.w["max_active"], len(self.open))
                    self.ctx, self.ctx_req = "request", req
                    self._scan_body(line)
                return
            if FIRST_TOKEN_RE.search(line):
                for req in self.open:
                    if req["first"] is None:
                        req["first"] = now
                        break
                return
            if STREAM_DONE_RE.search(line):
                self._finish(now)
                self.just_streamed = True
                return
            if PREDICTION_RE.search(line):
                # Directly after "Finished streaming" this belongs to the same request: do not count twice.
                req = self.last_finished if streamed else self._finish(now)
                self.ctx, self.ctx_req = "prediction", req
                self._scan_body(line)
                return
            lvl = LEVEL_RE.search(line[:60])
            if lvl and lvl.group(1) == "ERROR":
                self._inc("errors")

    def _finish(self, now):
        req = self.open.popleft() if self.open else {"start": None, "first": None, "model": None}
        self._inc("completed")
        req["end"] = now
        self.recent.append(req)
        if req["start"] is not None:
            self.w["latencies"].append(now - req["start"])
            if req["first"] is not None:
                self.w["ttfts"].append(req["first"] - req["start"])
        self.last_finished = req
        return req

    def _scan_body(self, line):
        """Extract the model name and token counts from a JSON body; all other content is discarded."""
        if self.ctx is None or self.ctx_req is None:
            return
        req = self.ctx_req
        if req.get("model") is None:
            m = MODEL_RE.search(line)
            if m:
                req["model"] = m.group(1)
        if self.ctx == "prediction":
            for regex, key in ((PROMPT_TOK_RE, "prompt_tokens"), (COMPL_TOK_RE, "completion_tokens")):
                m = regex.search(line)
                if m:
                    self._inc(key, int(m.group(1)))
                    req[key] = int(m.group(1))
            m = TPS_RE.search(line)
            if m:
                self.w["tps"].append(float(m.group(1)))
                req["tps"] = float(m.group(1))
            self._count_model(req)

    def _count_model(self, req):
        if req.get("model") and not req.get("counted"):
            self.w["models"][req["model"]] = self.w["models"].get(req["model"], 0) + 1
            req["counted"] = True

    def active(self):
        with self.lock:
            return len(self.open)

    def latency_points(self, window=1800):
        """Duration and time-to-first-token of completed requests within ``window`` seconds."""
        now = time.time()
        with self.lock:
            return [{"t": r["end"], "duration": rnd(r["end"] - r["start"]),
                     "ttft": rnd(r["first"] - r["start"]) if r["first"] is not None else None}
                    for r in self.recent if r["start"] is not None and now - r["end"] <= window]

    def live(self, tps_window=30, rate_window=60, lat_window=300, recent_n=25):
        """Real-time snapshot for the dashboard, using sliding windows on wall-clock time."""
        now = time.time()
        with self.lock:
            running = [{"age": round(now - r["start"], 1), "generating": r["first"] is not None,
                        "model": r.get("model")} for r in self.open if r["start"] is not None]
            recent = list(self.recent)
            totals = dict(self.totals)
            last_line = self.last_line_wall

        def ended(win):
            return [r for r in recent if now - r["end"] <= win]

        lat = [r["end"] - r["start"] for r in ended(lat_window) if r["start"] is not None]
        ttft = [r["first"] - r["start"] for r in ended(lat_window)
                if r["start"] is not None and r["first"] is not None]

        def row(r):
            dur = r["end"] - r["start"] if r["start"] is not None else None
            out_tok = r.get("completion_tokens")
            return {"end": r["end"], "duration_s": rnd(dur),
                    "ttft_s": rnd(r["first"] - r["start"]) if r["start"] and r["first"] else None,
                    "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": out_tok,
                    "tokens_per_s": rnd(r.get("tps") or (out_tok / dur if out_tok and dur else None), 1),
                    "model": r.get("model")}

        return {
            "active": len(running), "waiting": sum(not r["generating"] for r in running),
            "running": sorted(running, key=lambda r: -r["age"]),
            "tokens_per_s": rnd(sum(r.get("completion_tokens") or 0 for r in ended(tps_window)) / tps_window, 1),
            "requests_per_min": round(len(ended(rate_window)) * 60 / rate_window, 1),
            "latency_p95_s": rnd(pct(lat, 95)), "latency_avg_s": rnd(statistics.fmean(lat)) if lat else None,
            "ttft_p95_s": rnd(pct(ttft, 95)),
            "recent": [row(r) for r in reversed(recent[-recent_n:])],
            "totals": totals, "log_age_s": rnd(now - last_line, 0) if last_line else None,
        }

    def snapshot(self, reset=True):
        """Aggregate the current window into a flat dict (one CSV row) and optionally start a new window."""
        with self.lock:
            now = self.last_ts if self.use_log_time else time.time()
            while self.open and self.open[0]["start"] is not None \
                    and now - self.open[0]["start"] > self.stale_after:
                self.open.popleft()  # E.g. an aborted request that never logged a completion line.
                self.w["stale"] += 1
            w, duration = self.w, max(now - self.w_start, 1e-6)
            lat, ttft = w["latencies"], w["ttfts"]
            snap = {
                "requests_started": w["started"], "requests_completed": w["completed"],
                "embedding_requests": w["embeddings"], "log_errors": w["errors"],
                "stale_requests": w["stale"], "active_now": len(self.open),
                "max_active": max(w["max_active"], len(self.open)),
                "latency_avg_s": rnd(statistics.fmean(lat)) if lat else None,
                "latency_p95_s": rnd(pct(lat, 95)),
                "ttft_avg_s": rnd(statistics.fmean(ttft)) if ttft else None,
                "ttft_p95_s": rnd(pct(ttft, 95)),
                "prompt_tokens": w["prompt_tokens"], "completion_tokens": w["completion_tokens"],
                "gen_tokens_per_s_total": rnd(w["completion_tokens"] / duration),
                "tokens_per_s_per_request_avg": rnd(statistics.fmean(w["tps"])) if w["tps"] else None,
                "models_used": ";".join(f"{k}:{v}" for k, v in sorted(w["models"].items())),
            }
            if reset:
                self._new_window()
            return snap


class LogFollower(threading.Thread):
    """Continuously follows the newest LM Studio log file and feeds lines to a LogParser."""

    def __init__(self, target, parser, from_start=False, poll=0.5):
        super().__init__(daemon=True)
        self.target = Path(target)
        self.parser = parser
        self.from_start = from_start
        self.poll = poll
        self.current = None
        self._stop = threading.Event()

    def newest_file(self):
        if self.target.is_file():
            return self.target
        logs = [p for p in self.target.rglob("*.log") if p.is_file()]
        return max(logs, key=lambda p: p.stat().st_mtime) if logs else None

    def run(self):
        fh, buf, last_check, first = None, "", 0.0, True
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
class LiveRecorder(threading.Thread):
    """Collects log, hardware and API state once per second and notifies dashboard clients."""

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

    def _poll_api(self):
        if self.args.no_api or time.time() - self._last_api < 10:
            return
        self._last_api = time.time()
        models = loaded_models(self.args.lmstudio_url, self.args.token)
        self.api_ok, self.models = models is not None, models or []

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
                     **{k: sysm.get(k) for k in ("gpu_util", "vram_pct", "vram_used_mb", "vram_total_mb",
                                                 "gpu_temp", "gpu_power_w", "ram_pct", "ram_used_gb", "cpu_pct")}}
            state = {**live, "system": sysm, "alerts": self.alerts(live, sysm), "models": self.models,
                     "api_ok": self.api_ok, "gpu_available": self.sampler.gpu_available,
                     "log_file": self.follower.current.name if self.follower and self.follower.current else None,
                     "monitor_started": self.started}
            with self.cond:
                self.series.append(point)
                self.state = state
                self.seq += 1
                self.cond.notify_all()
            time.sleep(1)

    def full(self):
        """Complete state for the initial page load."""
        with self.cond:
            return {"series": list(self.series), "history": list(self.history), "state": self.state,
                    "latency_points": self.parser.latency_points(),
                    "config": {"interval": self.args.interval, "alert_p95": self.args.alert_p95,
                               "alert_vram_pct": self.args.alert_vram_pct,
                               "alert_gpu_temp": self.args.alert_gpu_temp}}


# --- Local dashboard server (bound to 127.0.0.1 only) ---
DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard.html"


def start_dashboard(port, recorder):
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
                self._send(json.dumps(recorder.full()).encode(), "application/json")
            elif path == "/api/stream":
                self._stream()
            else:
                self.send_error(404)

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
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--alert-vram-pct", type=float, default=95.0)
    ap.add_argument("--alert-gpu-temp", type=float, default=85.0)
    ap.add_argument("--alert-p95", type=float, default=60.0, help="Alert threshold for p95 latency, in seconds")
    ap.add_argument("--alert-active", type=int, default=0, help="Alert at N concurrent requests (0 = disabled)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # Flush every line, also when redirected to a file.

    target = find_log_target(args.log)
    if target is None or not target.exists():
        print("[monitor] WARNING: LM Studio log directory not found. Pass it with --log PATH. "
              "Only GPU and API metrics will be collected.")
        target = None

    parser = LogParser(use_log_time=args.from_start)
    follower = LogFollower(target, parser, from_start=args.from_start) if target else None
    if follower:
        follower.start()
    sampler = SystemSampler(interval=1.0, gpu_index=args.gpu_index)
    sampler.start()
    if not sampler.gpu_available:
        print("[monitor] Note: nvidia-smi not found, GPU metrics are unavailable.")
    recorder = LiveRecorder(parser, sampler, args, follower)
    recorder.start()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dashboard_port:
        try:
            start_dashboard(args.dashboard_port, recorder)
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
            print(f"{now:%H:%M:%S}  active {row['active_now']:>2} (max {row['max_active']:>2})  "
                  f"done {row['requests_completed']:>3}  p95 {dash(row['latency_p95_s']):>6}s  "
                  f"tok/s {dash(row['gen_tokens_per_s_total']):>6}  GPU {dash(row['gpu_util_avg']):>5}%  "
                  f"VRAM {dash(row['vram_pct_max']):>5}%  RAM {dash(row['ram_pct_max']):>5}%  {row['alerts']}")
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")
    finally:
        sampler.stop()
        if follower:
            follower.stop()


if __name__ == "__main__":
    main()
