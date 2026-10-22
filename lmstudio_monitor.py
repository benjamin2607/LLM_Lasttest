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
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from live_session import SessionManager, build_from_log, list_sessions
from lmstudio_log_parser import LogParser, diagnose, rnd
from lmstudio_models import ModelControl, request_arguments
from report_charts import with_print_support
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


def csv_header(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)
    except OSError:
        return None


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
    HARDWARE_STEP_S = 5  # Resolution of the saved hardware points (keeps the file small for long series).

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
        self.sessions = None  # SessionManager, set by main().
        self.hardware_file = Path(args.out) / ".live_hardware.json"
        self.cleared_file = Path(args.out) / ".cleared_at"
        self.cleared_at = self._load_cleared_at()
        self._saved_hardware = self._load_hardware()
        self._last_save = time.time()

    # --- dashboard "Clear" ---
    def _load_cleared_at(self):
        try:
            return float(self.cleared_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0.0

    def clear(self):
        """
        Reset everything the dashboard shows and start afresh. Stored CSV data is kept; the time of
        the reset is remembered so a restart does not restore the cleared values from the log.
        """
        now = time.time()
        self.parser.reset()
        with self.cond:
            self.series.clear()
            self.history.clear()
            self.started = self.cleared_at = now
            self._saved_hardware = {}
        try:
            self.cleared_file.write_text(str(now), encoding="utf-8")
            self.hardware_file.unlink(missing_ok=True)
        except OSError:
            pass
        print("[monitor] dashboard values cleared")

    # --- restoring state after a restart ---
    def _load_hardware(self):
        """Hardware points saved by a previous monitor run, keyed by second."""
        try:
            points = json.loads(self.hardware_file.read_text(encoding="utf-8")).get("points", [])
        except (OSError, ValueError):
            return {}
        cutoff = max(time.time() - self.keep_s, self.cleared_at)
        return {int(p["t"]) - int(p["t"]) % self.HARDWARE_STEP_S: p for p in points if p.get("t", 0) >= cutoff}

    def save_hardware(self):
        """Persist the hardware part of the live series (numbers only) for the next start."""
        with self.cond:
            points = [{"t": p["t"], **{k: p.get(k) for k in HARDWARE_FIELDS}} for p in self.series
                      if int(p["t"]) % self.HARDWARE_STEP_S == 0]
        try:
            tmp = self.hardware_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"points": points}), encoding="utf-8")
            os.replace(tmp, self.hardware_file)
        except OSError:
            pass

    def log_files_since(self, start):
        """LM Studio log files that may contain entries from ``start`` onwards."""
        if self.follower is None:
            return []
        return [p for p in self.follower._log_files() if p.stat().st_mtime >= start]

    def hardware_points(self):
        """Hardware points currently kept in memory (for sessions built afterwards)."""
        with self.cond:
            return [{"t": p["t"], **{k: p.get(k) for k in HARDWARE_FIELDS}} for p in self.series
                    if p.get("gpu_util") is not None or p.get("ram_pct") is not None]

    def replay_sample(self, second):
        """Called by the log replay once per second: rebuild one data point of the live series."""
        live = self.parser.live_point()
        hw = self._saved_hardware.get(second - second % self.HARDWARE_STEP_S, {})
        self.series.append({"t": float(second), "active": live["active"], "waiting": live["waiting"],
                            "tokens_per_s": live["tokens_per_s"], "requests_per_min": live["requests_per_min"],
                            **{k: hw.get(k) for k in HARDWARE_FIELDS}})

    def load_history(self, out_dir):
        """Fill the daily view from the CSV rows of the last 24 hours."""
        cutoff = max(datetime.now().timestamp() - 24 * 3600, self.cleared_at)
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
            if self.sessions is not None and self.sessions.current is not None:
                self.sessions.current.record_hardware(sysm)
            if time.time() - self._last_save >= self.SAVE_EVERY_S:
                self._last_save = time.time()
                self.save_hardware()
            time.sleep(1)

    def full(self):
        """Complete state for the initial page load."""
        with self.cond:
            return {"series": list(self.series), "history": list(self.history), "state": self.state,
                    "latency_points": self.parser.latency_points(window=self.keep_s),
                    "config": {"interval": self.args.interval, "alert_p95": self.args.alert_p95,
                               "alert_vram_pct": self.args.alert_vram_pct,
                               "alert_gpu_temp": self.args.alert_gpu_temp, "alert_active": self.args.alert_active,
                               "stall_after": self.parser.stall_after, "keep_s": self.keep_s}}


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

    # Parameters of a custom load test started from the dashboard: (type, min, max) or allowed values.
    LAUNCH_PARAMS = {
        "stage_duration": (int, 10, 7200), "spawn_rate": (float, 0.1, 50), "cooldown": (int, 0, 600),
        "stop_timeout": (int, 0, 600), "max_p95": (float, 1, 3600), "max_fail_rate": (float, 0, 1),
        "student_think_time": (float, 1, 3600), "wait_min": (float, 0, 600), "wait_max": (float, 0, 600),
        "request_timeout": (float, 5, 3600), "max_threads": (int, 0, 10000),
        "context_tokens": (int, 0, 200000),
    }

    def __init__(self, project_dir, anythingllm_url, api_key, lmstudio_url="http://localhost:1234", lmstudio_token=None):
        self.project_dir = project_dir = Path(project_dir)
        self.process = None
        self.launch_info = None
        self.status_file = project_dir / ".loadtest_status.json"
        self.stop_file = project_dir / ".loadtest_stop"
        self.registry_file = project_dir / ".loadtest_threads.json"
        self.url = anythingllm_url.rstrip("/")
        self.api_key = api_key
        self.lmstudio_url = lmstudio_url.rstrip("/")
        self.lmstudio_token = lmstudio_token
        self.models = None  # ModelControl of the dashboard, set by main().
        self._workspace_model = (0.0, None)
        self.lock = threading.Lock()
        self.progress = {"phase": "idle"}

    def loadtest(self):
        try:
            status = json.loads(self.status_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"running": False}
        running = time.time() - status.get("heartbeat", 0) < self.HEARTBEAT_TIMEOUT_S
        return {"running": running, **{k: status.get(k) for k in ("run", "name", "stage", "users", "workspace", "run_started", "target",
                                                       "phase")}}

    def _run_config(self, run):
        """Settings of a load test run from its config.json (values as the run recorded them)."""
        try:
            cfg = json.loads((self.project_dir / "loadtest_results" / run / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            forwarded = ast.literal_eval(cfg.get("passthrough", "[]"))
        except (ValueError, SyntaxError):
            forwarded = []
        opts = {}
        for i, arg in enumerate(forwarded):
            if arg.startswith("--") and "=" in arg:
                key, value = arg[2:].split("=", 1)
                opts[key.replace("-", "_")] = value
            elif arg.startswith("--") and i + 1 < len(forwarded) and not forwarded[i + 1].startswith("--"):
                opts[arg[2:].replace("-", "_")] = forwarded[i + 1]
        cfg["forwarded"] = opts
        return cfg

    def loadtest_details(self):
        """Settings of the running load test, otherwise of the most recent one."""
        lt = self.loadtest()
        run = lt.get("run") if lt.get("running") else None
        if run is None:
            runs = self.list_runs(limit=1)
            run = runs[0]["run"] if runs else None
        cfg = self._run_config(run) if run else None
        return {"running": lt.get("running", False), "run": run, "stage": lt.get("stage"), "users": lt.get("users"),
                "workspace": lt.get("workspace"), "config": cfg}

    def delete_run(self, run):
        """
        Delete a load test result folder from disk. Only plain folder names inside loadtest_results
        are accepted, and a running load test cannot be deleted. Returns (ok, message).
        """
        folder = safe_child(self.project_dir / "loadtest_results", run)
        if folder is None:
            return False, "invalid load test name"
        if not folder.is_dir():
            return False, "load test not found"
        lt = self.loadtest()
        if lt.get("running") and lt.get("run") == run:
            return False, "a running load test cannot be deleted – stop it first"
        shutil.rmtree(folder)
        print(f"[monitor] load test deleted from disk: {run}")
        return True, "load test deleted"

    def list_runs(self, limit=30):
        """Completed and running load tests (newest first) with their basic settings and results."""
        results_dir = self.project_dir / "loadtest_results"
        if not results_dir.is_dir():
            return []
        running = self.loadtest()
        out = []
        for folder in sorted((p for p in results_dir.iterdir() if p.is_dir() and (p / "config.json").exists()),
                             key=lambda p: p.name, reverse=True)[:limit]:
            cfg = self._run_config(folder.name) or {}
            try:
                stages = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stages = []
            passing = [r for r in stages if r.get("passed")]
            best = max(passing, key=lambda r: r.get("chat_rps") or 0, default=None)
            first_fail = next((r for r in sorted(stages, key=lambda r: r["users"]) if not r.get("passed")), None)
            if running.get("running") and running.get("run") == folder.name:
                status = "running"
            elif any(r.get("stopped") for r in stages):
                status = "stopped"
            elif any(r.get("error") for r in stages):
                status = "setup error"
            elif not stages:
                status = "no data"
            else:
                status = "completed"
            target = cfg.get("target") or "anythingllm"  # Runs from before --target existed tested AnythingLLM.
            forwarded = cfg.get("forwarded", {})
            models = sorted({m.rsplit(":", 1)[0] for r in stages for m in (r.get("models") or "").split(";") if m})
            out.append({
                "run": folder.name, "name": cfg.get("name") or "", "status": status, "target": target,
                "model": forwarded.get("model") or ", ".join(models) or None,
                "users": cfg.get("users") if cfg.get("users") not in (None, "", "None")
                else ",".join(str(r["users"]) for r in sorted(stages, key=lambda r: r["users"])),
                "mode": forwarded.get("mode", "query") if target == "anythingllm" else "direct",
                "stage_duration": cfg.get("stage_duration"),
                "stages_run": len(stages), "highest_passing": max((r["users"] for r in passing), default=None),
                "limit_at": first_fail["users"] if first_fail else None,
                "limit_reason": first_fail.get("reasons") if first_fail else None,
                "max_per_min": round(best["chat_rps"] * 60, 1) if best and best.get("chat_rps") else None,
                "p95_at_best": best.get("chat_p95_s") if best else None,
                "report": (folder / "report.html").exists()})
        return out

    def anythingllm_model(self, max_age=30):
        """
        The model AnythingLLM uses for WORKSPACE_SLUG: the workspace's own model, otherwise the system
        model. Returns {"model", "provider", "source"} or None if it cannot be read; cached for 30 s.
        """
        at, cached = self._workspace_model
        if time.time() - at < max_age:
            return cached
        workspace, info = os.environ.get("WORKSPACE_SLUG"), None
        if self.api_key and workspace:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def get(path):
                with urllib.request.urlopen(urllib.request.Request(f"{self.url}{path}", headers=headers), timeout=5) as r:
                    return json.loads(r.read())
            try:
                ws = get(f"/api/v1/workspace/{workspace}").get("workspace")
                ws = (ws[0] if isinstance(ws, list) and ws else ws) or {}
                if ws.get("chatModel"):
                    info = {"model": ws["chatModel"], "provider": ws.get("chatProvider"), "source": "workspace"}
                else:
                    settings = get("/api/v1/system").get("settings") or {}
                    provider = settings.get("LLMProvider")
                    pref = next((v for k, v in settings.items() if k.lower() == f"{(provider or '').lower()}modelpref"), None)
                    if pref:
                        info = {"model": pref, "provider": provider, "source": "system"}
            except (urllib.error.URLError, OSError, ValueError, AttributeError):
                info = None
        self._workspace_model = (time.time(), info)
        return info

    def threads(self):
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8")).get("threads", [])
        except (OSError, ValueError):
            return []

    def state(self):
        with self.lock:
            progress = dict(self.progress)
            launch = dict(self.launch_info) if self.launch_info else None
        if launch and self.process is not None and self.process.poll() is not None:
            launch["exit_code"] = self.process.returncode
            if self.process.returncode:  # Failed early, e.g. preflight: show why.
                launch["log_tail"] = self._log_tail(launch["log"])
        return {"loadtest": self.loadtest(), "threads": len(self.threads()),
                "can_delete": bool(self.api_key), "progress": progress, "launch": launch,
                "latest_report": self.latest_report(),
                "defaults": {"host": self.url, "lmstudio_host": self.lmstudio_url,
                             "workspace": os.environ.get("WORKSPACE_SLUG", ""), "api_key_set": bool(self.api_key)}}

    def latest_report(self):
        results = self.project_dir / "loadtest_results"
        runs = sorted((p for p in results.glob("*/report.html")), key=lambda p: p.stat().st_mtime) if results.is_dir() else []
        return runs[-1].parent.name if runs else None

    @staticmethod
    def _log_tail(path, lines=12):
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        except OSError:
            return []

    def launch(self, params):
        """
        Start auto_loadtest.py with validated parameters from the dashboard. Returns (ok, message).
        The process is started without a shell; every value is checked against a fixed rule.
        """
        if self.loadtest()["running"] or (self.process is not None and self.process.poll() is None):
            return False, "a load test is already running"
        users = str(params.get("users", "")).replace(" ", "")
        if not re.fullmatch(r"\d{1,4}(,\d{1,4}){0,40}", users) or not all(int(u) > 0 for u in users.split(",")):
            return False, "users must be a comma-separated list of positive numbers, e.g. 1,2,4,8"
        name = " ".join(str(params.get("name") or "").split())
        if len(name) > 120 or any(ord(c) < 32 for c in name):
            return False, "the name must be one line of at most 120 characters"
        target = params.get("target") or "anythingllm"
        if target not in ("anythingllm", "lmstudio"):
            return False, "target must be anythingllm or lmstudio"
        host = str(params.get("host") or (self.url if target == "anythingllm" else self.lmstudio_url)).strip()
        if not re.fullmatch(r"https?://[\w.\-\[\]:]+(/[\w./\-]*)?", host):
            return False, "host must be an http(s) URL"
        env = dict(os.environ)
        if target == "anythingllm":
            workspace = str(params.get("workspace") or os.environ.get("WORKSPACE_SLUG", "")).strip()
            if not re.fullmatch(r"[\w-]{1,100}", workspace):
                return False, "workspace slug is missing or invalid"
            api_key = str(params.get("api_key") or self.api_key or "")
            if not api_key:
                return False, "an AnythingLLM API key is required (field or env ANYTHING_API_KEY)"
            mode = params.get("mode", "query")
            if mode not in ("query", "chat"):
                return False, "mode must be query or chat"
            env.update(ANYTHING_API_KEY=api_key, WORKSPACE_SLUG=workspace)
            secret = api_key
        else:
            workspace, mode = None, None
            # The model and its request settings come from the dashboard's LM Studio model card.
            model = self.models.loaded_model() if self.models else None
            if not model:
                return False, "no model is loaded in LM Studio – load one in the LM Studio model card first"
            request_args = request_arguments(self.models.request_settings(model))
            if self.lmstudio_token:
                env["LMSTUDIO_API_TOKEN"] = self.lmstudio_token
            secret = self.lmstudio_token
        values = {}
        for key, (kind, lo, hi) in self.LAUNCH_PARAMS.items():
            if params.get(key) in (None, ""):
                continue
            try:
                value = kind(params[key])
            except (TypeError, ValueError):
                return False, f"{key} must be a number"
            if not lo <= value <= hi:
                return False, f"{key} must be between {lo} and {hi}"
            values[key] = value
        if values.get("wait_min", 0) > values.get("wait_max", values.get("wait_min", 0)):
            return False, "wait_min must not exceed wait_max"

        cmd = [sys.executable, str(self.project_dir / "auto_loadtest.py"), "--target", target, "--host", host,
               "--users", users] + (["--name", name] if name else [])
        for key in ("stage_duration", "spawn_rate", "cooldown", "stop_timeout", "max_p95", "max_fail_rate",
                    "student_think_time"):
            if key in values:
                cmd += [f"--{key.replace('_', '-')}", str(values[key])]
        if params.get("refine"):
            cmd.append("--refine")
        # Options of the Locust user file; each target only gets the ones it understands.
        if target == "anythingllm":
            cmd += ["--", "--mode", mode]
            forwarded = ("wait_min", "wait_max", "request_timeout", "max_threads")
        else:
            cmd += ["--", "--model", model, *request_args]
            forwarded = ("wait_min", "wait_max", "request_timeout", "context_tokens")
        for key in forwarded:
            if key in values:
                cmd += [f"--{key.replace('_', '-')}", str(values[key])]
        log_dir = self.project_dir / "loadtest_results"
        log_dir.mkdir(exist_ok=True)
        log = log_dir / f"dashboard_{datetime.now():%Y%m%d_%H%M%S}.log"
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
        with open(log, "w", encoding="utf-8") as out:
            self.process = subprocess.Popen(cmd, cwd=self.project_dir, env=env, stdout=out, stderr=subprocess.STDOUT,
                                            **flags)
        with self.lock:
            self.launch_info = {"pid": self.process.pid, "started": time.time(), "log": str(log), "target": target,
                                "users": users, "mode": mode, "host": host, "workspace": workspace}
        printable = [c for c in cmd if not secret or c != secret]
        print(f"[monitor] load test started from the dashboard: {' '.join(printable[1:])}")
        return True, f"load test started (log: {log.name})"

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


def safe_child(root, name):
    """Path of folder ``name`` directly inside ``root``, or None if the name could reach anywhere else."""
    if not re.fullmatch(r"[\w][\w.-]*", name or ""):
        return None
    root = Path(root).resolve()
    child = (root / name).resolve()
    return child if child.parent == root else None


def start_dashboard(port, recorder, control, sessions):
    # The server only listens on 127.0.0.1. Checking Host and Origin as well defeats DNS rebinding:
    # a web page whose domain is re-pointed to 127.0.0.1 must not be able to read data or trigger actions.
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{host}" for host in allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        def _trusted(self):
            host = (self.headers.get("Host") or "").lower()
            origin = (self.headers.get("Origin") or "").lower()
            if host not in allowed_hosts or (origin and origin not in allowed_origins):
                self.send_error(403, "Requests are accepted from the local dashboard only")
                return False
            return True

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._trusted():
                return
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(DASHBOARD_FILE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(json.dumps({**recorder.full(), "control": control.state(),
                                       "session": sessions.state()}).encode(), "application/json")
            elif path == "/api/loadtests":
                self._send(json.dumps(control.list_runs()).encode(), "application/json")
            elif path == "/api/models":
                state = control.models.state() if control.models else {"models": [], "error": "not available"}
                self._send(json.dumps({**state, "anythingllm": control.anythingllm_model()}).encode(), "application/json")
            elif path == "/api/loadtest/details":
                self._send(json.dumps(control.loadtest_details()).encode(), "application/json")
            elif path == "/api/sessions":
                self._send(json.dumps(list_sessions(sessions.root)).encode(), "application/json")
            elif path.startswith(("/loadtest/", "/sessions/")) and path.endswith("/report.html") and path.count("/") == 3:
                # Only report.html of a folder directly inside the results or sessions directory is served.
                root = control.project_dir / "loadtest_results" if path.startswith("/loadtest/") else sessions.root
                folder = safe_child(root, path.split("/")[2])
                report = folder / "report.html" if folder else None
                if report is not None and report.is_file():
                    # Reports written before the print styles existed get them here, for the PDF export.
                    page = with_print_support(report.read_text(encoding="utf-8", errors="replace"))
                    self._send(page.encode("utf-8"), "text/html; charset=utf-8")
                else:
                    self.send_error(404)
            elif path == "/api/stream":
                self._stream()
            else:
                self.send_error(404)

        def _json_body(self):
            try:
                length = max(0, min(int(self.headers.get("Content-Length") or 0), 64 * 1024))
            except ValueError:
                return {}
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return {}
            return body if isinstance(body, dict) else {}

        def do_POST(self):
            """
            Actions of the dashboard. Besides the Host/Origin check, a custom header is required: browsers
            only send it from the dashboard's own origin (other sites would need a CORS preflight, which
            this server never grants).
            """
            if not self._trusted():
                return
            if self.headers.get("X-Monitor-Action") not in ("dismiss", "stop", "clear", "session", "loadtest", "delete",
                                                            "model"):
                self.send_error(403)
                return
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/api/models/") and control.models:
                body, models = self._json_body(), control.models
                actions = {"/api/models/apply": lambda: models.apply(body),
                           "/api/models/unload": lambda: models.unload(str(body.get("model") or "")),
                           "/api/models/configs/save": lambda: models.save_config(body),
                           "/api/models/configs/delete": lambda: models.delete_config(str(body.get("name") or ""))}
                if path not in actions:
                    self.send_error(404)
                    return
                try:
                    ok, message = actions[path]()
                except Exception as exc:  # noqa: BLE001 - report any unexpected failure instead of dropping the connection.
                    print(f"[monitor] model action failed: {type(exc).__name__}: {exc}")
                    ok, message = False, "the action failed – see the monitor console"
                self._send(json.dumps({"ok": ok, "message": message}).encode(), "application/json")
                return
            if path == "/api/monitor/clear":
                recorder.clear()
                session = sessions.current
                if session is not None:  # "Clear" ends the running session (with report) and starts a new one.
                    label = session.meta["label"]
                    sessions.end()
                    sessions.start(label, recorder.models)
                self._send(json.dumps({"cleared": True}).encode(), "application/json")
                return
            if path == "/api/session/start":
                label = str(self._json_body().get("label", ""))[:120]
                self._send(json.dumps(sessions.start(label or "Live test", recorder.models)).encode(),
                           "application/json")
                return
            if path == "/api/session/end":
                self._send(json.dumps(sessions.end() or {}).encode(), "application/json")
                return
            if path == "/api/session/report":
                self._send(json.dumps(sessions.report_now() or {}).encode(), "application/json")
                return
            if path == "/api/loadtest/delete":
                ok, message = control.delete_run(str(self._json_body().get("run", "")))
                self._send(json.dumps({"ok": ok, "message": message}).encode(), "application/json")
                return
            if path == "/api/session/delete":
                ok, message = sessions.delete(str(self._json_body().get("folder", "")))
                self._send(json.dumps({"ok": ok, "message": message}).encode(), "application/json")
                return
            if path == "/api/loadtest/start":
                ok, message = control.launch(self._json_body())
                self._send(json.dumps({"ok": ok, "message": message}).encode(), "application/json")
                return
            if path == "/api/session/build":
                body = self._json_body()
                try:
                    start = datetime.fromisoformat(str(body.get("start"))).timestamp()
                    end = datetime.fromisoformat(str(body.get("end"))).timestamp()
                except ValueError:
                    self._send(json.dumps({"ok": False, "message": "invalid start or end"}).encode(), "application/json")
                    return
                if not start < end <= time.time() + 60:
                    self._send(json.dumps({"ok": False, "message": "the range must lie in the past and end after it starts"}
                                          ).encode(), "application/json")
                    return
                info = build_from_log(sessions.root, str(body.get("label") or "Past session")[:120], start, end,
                                      recorder.log_files_since(start), recorder.args.out, sessions.host_info,
                                      recorder.hardware_points())
                print(f"[monitor] session built from the log: {info['folder']} ({info['counts']['completed']} requests)")
                self._send(json.dumps({"ok": True, **info}).encode(), "application/json")
                return
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
                    msg["session"] = sessions.state()
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
    ap.add_argument("--session", metavar="LABEL",
                    help="Start a recording session for a live test right away (or use the dashboard)")
    ap.add_argument("--keep-hours", type=float, default=4,
                    help="Hours of per-second data kept for the dashboard (time range \"Max\")")
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
    recorder = LiveRecorder(parser, sampler, args, follower, keep_s=int(args.keep_hours * 3600))
    recorder.load_history(out_dir)
    # Sessions are set up before the log replay, so requests made while the monitor was not
    # running are added to a resumed session.
    sessions = SessionManager(out_dir, sampler.gpu.info())
    recorder.sessions = sessions
    parser.on_request_done = lambda req, status: sessions.current and sessions.current.record_request(req, status)
    if args.session:
        if sessions.current:
            print(f"[monitor] continuing the active session \"{sessions.current.meta['label']}\" "
                  "(end it in the dashboard to start a new one)")
        else:
            sessions.start(args.session)
    if follower:
        # Restore the recent past so the dashboard is filled right after a restart.
        t0 = time.time()
        replay_window = 0 if args.from_start else args.replay_minutes * 60
        if sessions.current is not None and not args.from_start:
            # "Max" covers the whole running session, so rebuild it completely (within the kept range).
            replay_window = max(replay_window, time.time() - sessions.current.meta["started"])
        replay_window = min(replay_window, recorder.keep_s)
        replay_window = max(0, min(replay_window, time.time() - recorder.cleared_at))  # Not before a "Clear".
        follower.open_initial(replay_window=replay_window, sample=recorder.replay_sample)
        if replay_window:
            print(f"[monitor] restored the last {replay_window / 60:.0f} min from the log "
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
            control = LoadTestControl(args.project_dir, args.anythingllm_url, os.environ.get("ANYTHING_API_KEY"),
                                      args.lmstudio_url, args.token)
            control.models = ModelControl(args.lmstudio_url, args.token, Path(args.project_dir) / "model_configs.json",
                                          busy=lambda: control.loadtest()["running"])
            start_dashboard(args.dashboard_port, recorder, control, sessions)
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
            if sessions.current is not None:
                sessions.current.record_interval(row)

            csv_path = out_dir / f"lmstudio_metrics_{now:%Y-%m-%d}.csv"
            new_file = not csv_path.exists()
            header = list(row.keys()) if new_file else csv_header(csv_path) or list(row.keys())
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                # An existing file keeps its columns, so rows stay aligned after an update adds metrics.
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
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
        if sessions.current is not None:
            sessions.report_now()  # Report so far; the session stays active and resumes on the next start.
            print(f"[monitor] session \"{sessions.current.meta['label']}\" stays active; "
                  f"report so far: {sessions.current.folder / 'report.html'}")
        sampler.stop()
        if follower:
            follower.stop()


if __name__ == "__main__":
    main()
