#!/usr/bin/env python3
"""
Recording sessions for live tests with real users (used by lmstudio_monitor.py).

A session writes into its own folder ``<monitoring>/sessions/<start>_<label>/``:

* ``session.json``   label, start/end time, host and models
* ``requests.csv``   one row per finished request: times, duration, time to first token,
                     tokens in/out, tokens/s, model, source and status
* ``intervals.csv``  the monitor's per-interval aggregates (default: one row per minute)
* ``hardware.csv``   GPU, GPU memory, system memory and CPU every 5 seconds
* ``summary.json`` / ``report.html``  key figures and charts, created from the CSV files

Only numbers, times and model names are stored: no prompts, responses, IP addresses or
user identifiers. The report can be regenerated from the CSV files at any time:

    python live_session.py --report monitoring/sessions/<folder>
"""
import argparse
import bisect
import csv
import html
import json
import os
import re
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

from lmstudio_log_parser import tokens_per_second
from report_charts import TILE_CSS, count_tile, fmt_num, histogram_chart, html_page, stat_tile, tiles, time_chart

REQUEST_FIELDS = ["start", "end", "status", "duration_s", "ttft_s", "prompt_tokens", "completion_tokens",
                  "tokens_estimated", "tokens_per_s", "model", "source", "streamed"]
HARDWARE_FIELDS = ["time", "gpu_util", "vram_pct", "vram_used_mb", "gpu_temp", "gpu_power_w",
                   "ram_pct", "ram_used_gb", "cpu_pct"]
HARDWARE_EVERY_S = 5


def iso(epoch):
    return datetime.fromtimestamp(epoch).isoformat(timespec="seconds") if epoch else ""


def slug(label):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip()).strip("-")[:60] or "session"


class _CsvAppender:
    """Appends rows to a CSV file, writing the header for a new file."""

    def __init__(self, path, fields):
        self.path, self.fields = Path(path), fields
        self.lock = threading.Lock()

    def append(self, row):
        with self.lock:
            new = not self.path.exists()
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
                if new:
                    writer.writeheader()
                writer.writerow(row)


def _existing_header(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)
    except OSError:
        return None


class LiveSession:
    """One recording session; created via :class:`SessionManager`."""

    def __init__(self, folder, meta):
        self.folder = Path(folder)
        self.meta = meta
        self.requests = _CsvAppender(self.folder / "requests.csv", REQUEST_FIELDS)
        self.intervals = None  # Created on the first row, its columns come from the monitor.
        self.hardware = _CsvAppender(self.folder / "hardware.csv", HARDWARE_FIELDS)
        self.counts = {"completed": 0, "failed": 0, "rejected": 0, "dismissed": 0, "stale": 0}
        self._last_hw = 0.0
        # When a session is resumed after a monitor restart, the log replay reports requests again that
        # were written before the restart. Only those (ending up to this boundary) are checked for duplicates.
        self.resume_boundary = meta.get("last_request_end", 0)
        self._written = []
        self._count_existing()

    def _count_existing(self):
        try:
            with open(self.folder / "requests.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.counts[row["status"]] = self.counts.get(row["status"], 0) + 1
                    end = _epoch(row.get("end"))
                    if end is not None and end >= self.resume_boundary - 900:
                        self._written.append((_epoch(row.get("start")), end, row.get("model"), row.get("status")))
        except OSError:
            pass

    def _already_written(self, start, end, model, status):
        # Rows store whole seconds and the replay works in log time (also whole seconds): allow 2 s.
        return any(abs((s or 0) - (start or 0)) <= 2 and abs(e - end) <= 2 and m == (model or "") and st == status
                   for s, e, m, st in self._written)

    def save_meta(self):
        tmp = self.folder / "session.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.folder / "session.json")

    # --- recording ---
    def record_request(self, req, status):
        """Write one finished request. Requests that ended before the session started are skipped."""
        end = req.get("end") or time.time()
        start, first = req.get("start"), req.get("first")
        if end < self.meta["started"]:
            return  # Ended before the session started.
        if self.resume_boundary and end <= self.resume_boundary + 2 and \
                self._already_written(start, end, req.get("model"), status):
            return  # Written before the monitor restarted.
        duration = end - start if start else None
        out_tok = req.get("completion_tokens")
        tps = tokens_per_second({**req, "end": end})
        self.requests.append({
            "start": iso(start), "end": iso(end), "status": status,
            "duration_s": round(duration, 2) if duration is not None and status == "completed" else "",
            "ttft_s": round(max(0.0, first - start), 2) if start and first and status == "completed" else "",
            "prompt_tokens": req.get("prompt_tokens") or "", "completion_tokens": out_tok or "",
            "tokens_estimated": int(bool(req.get("tokens_estimated"))),
            "tokens_per_s": round(tps, 1) if tps and status == "completed" else "",
            "model": req.get("model") or "", "source": "api" if req.get("api") else "app",
            "streamed": int(bool(req.get("streamed")))})
        self.counts[status] = self.counts.get(status, 0) + 1
        self.meta["last_request_end"] = max(end, self.meta.get("last_request_end", 0))
        self.save_meta()  # Lets a restarted monitor continue without writing requests twice.

    def record_interval(self, row):
        if self.intervals is None:
            self.intervals = _CsvAppender(self.folder / "intervals.csv", _existing_header(self.folder / "intervals.csv")
                                          or list(row.keys()))
        self.intervals.append(row)

    def record_hardware(self, sample, now=None):
        now = now or time.time()
        if now - self._last_hw < HARDWARE_EVERY_S:
            return
        self._last_hw = now
        self.hardware.append({"time": iso(now), **{k: sample.get(k) for k in HARDWARE_FIELDS if k != "time"}})

    def info(self):
        return {"active": not self.meta.get("ended"), "label": self.meta["label"], "folder": self.folder.name,
                "started": self.meta["started"], "ended": self.meta.get("ended"), "counts": dict(self.counts),
                "report": (self.folder / "report.html").exists()}


class SessionManager:
    """Starts, resumes and ends sessions below ``<out_dir>/sessions``."""

    def __init__(self, out_dir, host_info=None):
        self.root = Path(out_dir) / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.marker = self.root / ".active"
        self.host_info = host_info or {}
        self.lock = threading.Lock()
        self.current = None
        self.last = None
        self._resume()

    def _resume(self):
        """Continue a session that was active when the monitor stopped."""
        try:
            folder = self.root / self.marker.read_text(encoding="utf-8").strip()
            meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not meta.get("ended"):
            self.current = LiveSession(folder, meta)
            print(f"[monitor] resumed session \"{meta['label']}\" ({folder.name})")

    def start(self, label, models=None):
        with self.lock:
            if self.current:
                self._end_locked()
            now = time.time()
            base = f"{datetime.fromtimestamp(now):%Y-%m-%d_%H%M%S}_{slug(label)}"
            folder, n = self.root / base, 1
            while folder.exists():  # E.g. "Clear" ends and restarts a session within the same second.
                n += 1
                folder = self.root / f"{base}-{n}"
            folder.mkdir(parents=True)
            meta = {"label": label.strip() or "Session", "started": now, "ended": None,
                    "host": self.host_info, "models": models or []}
            self.current = LiveSession(folder, meta)
            self.current.save_meta()
            self.marker.write_text(folder.name, encoding="utf-8")
            print(f"[monitor] session \"{meta['label']}\" started: {folder}")
            return self.current.info()

    def _end_locked(self):
        session, self.current = self.current, None
        session.meta["ended"] = time.time()
        session.save_meta()
        write_report(session.folder)
        try:
            self.marker.unlink()
        except OSError:
            pass
        self.last = session
        print(f"[monitor] session \"{session.meta['label']}\" ended, report: {session.folder / 'report.html'}")
        return session.info()

    def end(self):
        with self.lock:
            return self._end_locked() if self.current else None

    def report_now(self):
        """Write a report for the active session without ending it."""
        session = self.current or self.last
        if session is None:
            return None
        write_report(session.folder)
        return session.info()

    def state(self):
        session = self.current or self.last
        return session.info() if session else {"active": False}


def build_from_log(root, label, start, end, log_files, metrics_dir, host=None, hardware_points=None):
    """
    Create a session for a past time range afterwards and write its report.

    Requests come from the LM Studio log files (replayed for ``start``..``end``), per-interval
    values from the monitor's daily CSV files in ``metrics_dir``. Hardware uses the monitor's
    5-second points where available (``hardware_points``: list of dicts with ``t``), otherwise the
    per-interval values. Only data that the log or the monitor recorded can be included.
    """
    from lmstudio_log_parser import LogParser

    root = Path(root)
    base = f"{datetime.fromtimestamp(start):%Y-%m-%d_%H%M%S}_{slug(label)}_from-log"
    folder, n = root / base, 1
    while folder.exists():
        n += 1
        folder = root / f"{base}-{n}"
    folder.mkdir(parents=True)
    meta = {"label": label.strip() or "Session", "started": start, "ended": end, "host": host or {},
            "models": [], "retroactive": True}
    session = LiveSession(folder, meta)
    session.save_meta()

    # Requests from the LM Studio log, in log time.
    lines = []
    for path in log_files:
        try:
            lines += Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
    parser = LogParser(use_log_time=True)
    parser.on_request_done = session.record_request
    parser.replay(lines, since=start, until=end)

    # Per-interval values and hardware from the monitor's own files.
    interval_rows = []
    for path in sorted(Path(metrics_dir).glob("lmstudio_metrics_*.csv")):
        for row in _read_csv(path):
            t = _epoch(row.get("time"))
            if t is not None and start <= t <= end:
                interval_rows.append(row)
    for row in interval_rows:
        session.record_interval(row)
    # 5-second points where the monitor still holds them, per-interval values for the rest of the range.
    points = [p for p in (hardware_points or []) if start <= p.get("t", 0) <= end]
    points = [p for i, p in enumerate(points) if i == 0 or int(p["t"]) // HARDWARE_EVERY_S
              != int(points[i - 1]["t"]) // HARDWARE_EVERY_S]
    covered = {int(p["t"] // 60) for p in points}
    hw_rows = [(p["t"], {"time": iso(p["t"]), **{k: p.get(k) for k in HARDWARE_FIELDS if k != "time"}}) for p in points]
    mapping = {"gpu_util": "gpu_util_avg", "vram_pct": "vram_pct_max", "vram_used_mb": "vram_used_mb_max",
               "gpu_temp": "gpu_temp_max", "gpu_power_w": "gpu_power_w_avg", "ram_pct": "ram_pct_max",
               "ram_used_gb": "ram_used_gb_max", "cpu_pct": "cpu_pct_avg"}
    for row in interval_rows:
        t = _epoch(row["time"])
        if int(t // 60) not in covered:
            hw_rows.append((t, {"time": row["time"], **{k: row.get(v) for k, v in mapping.items()}}))
    for _, row in sorted(hw_rows, key=lambda item: item[0]):
        session.hardware.append(row)
    session.meta["models"] = sorted({r["model"] for r in _read_csv(folder / "requests.csv") if r.get("model")})
    session.save_meta()
    write_report(folder)
    return session.info()


def list_sessions(root, limit=20):
    """The most recent sessions (newest first) for the dashboard."""
    out = []
    for folder in sorted((p for p in Path(root).iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)[:limit]:
        try:
            meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = _read_csv(folder / "requests.csv")
        out.append({"folder": folder.name, "label": meta.get("label"), "started": meta.get("started"),
                    "ended": meta.get("ended"), "retroactive": bool(meta.get("retroactive")),
                    "completed": sum(1 for r in rows if r.get("status") == "completed"),
                    "report": (folder / "report.html").exists()})
    return out


# --- Report ---
def _read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _epoch(text):
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, min(len(values) - 1, round(p / 100 * len(values)) - 1))]


def _describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": _pct(values, 95), "max": max(values),
            "sd": statistics.pstdev(values) if len(values) > 1 else 0.0}


def _per_minute(rows, value_key=None, agg="count", span=None):
    """
    Group rows by minute of their end time; returns (minute epochs, values). With ``span``
    (start, end), every minute of the session is included: idle minutes count as 0, or as a
    gap (None) for value aggregates, so idle periods are not bridged by the chart lines.
    """
    buckets = {}
    for row in rows:
        t = _epoch(row.get("end") or row.get("time"))
        if t is None:
            continue
        minute = int(t // 60 * 60)
        buckets.setdefault(minute, []).append(_num(row.get(value_key)) if value_key else 1)
    if span:
        first, last = int(span[0] // 60 * 60), int(span[1] // 60 * 60)
        minutes = list(range(first, last + 60, 60))
    else:
        minutes = sorted(buckets)
    if agg == "count":
        return minutes, [len(buckets.get(m, [])) for m in minutes]
    buckets = {m: buckets.get(m, []) for m in minutes}
    out = []
    for m in minutes:
        vals = [v for v in buckets[m] if v is not None]
        out.append(None if not vals else statistics.median(vals) if agg == "median"
                   else _pct(vals, 95) if agg == "p95" else max(vals) if agg == "max" else statistics.fmean(vals))
    return minutes, out


def busy_periods(intervals, span=None):
    """
    Merge request intervals (start, end) into periods with at least one request in flight, clipped
    to ``span``. "Busy" includes waiting for the first token, as users experience it.
    """
    clipped = []
    for start, end in intervals:
        if start is None or end is None:
            continue
        if span:
            start, end = max(start, span[0]), min(end, span[1])
        if end >= start:
            clipped.append((start, end))
    merged = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def busy_stats(periods, span):
    """Busy seconds, share of the span, and the longest idle gap."""
    total = max(span[1] - span[0], 1e-9)
    busy = sum(end - start for start, end in periods)
    edges = [span[0]] + [x for p in periods for x in p] + [span[1]]
    gaps = [edges[i + 1] - edges[i] for i in range(0, len(edges) - 1, 2)]
    return {"busy_s": round(busy, 1), "total_s": round(total, 1), "busy_share": round(busy / total, 4),
            "longest_idle_s": round(max(gaps, default=total), 1)}


def in_periods(t, periods, starts=None):
    """Whether time ``t`` lies in one of the (sorted, merged) periods."""
    starts = starts if starts is not None else [p[0] for p in periods]
    i = bisect.bisect_right(starts, t) - 1
    return i >= 0 and t <= periods[i][1]


def summarise(folder):
    """Key figures of a session, computed from its CSV files."""
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    requests = _read_csv(folder / "requests.csv")
    intervals = _read_csv(folder / "intervals.csv")
    hardware = _read_csv(folder / "hardware.csv")
    done = [r for r in requests if r["status"] == "completed"]
    end = meta.get("ended") or time.time()
    hours = max((end - meta["started"]) / 3600, 1e-9)
    status = {}
    for r in requests:
        status[r["status"]] = status.get(r["status"], 0) + 1
    minutes, per_min = _per_minute(done)
    busiest = max(zip(per_min, minutes)) if per_min else None
    models = {}
    for r in done:
        models.setdefault(r["model"] or "unknown", []).append(r)
    hw = lambda key: _describe([_num(h.get(key)) for h in hardware])  # noqa: E731
    return {
        "label": meta["label"], "started": iso(meta["started"]), "ended": iso(meta.get("ended")) or None,
        "duration_min": round((end - meta["started"]) / 60, 1), "host": meta.get("host", {}),
        "requests": {"total": len(requests), **status, "per_hour": round(len(done) / hours, 1)},
        "response_time_s": _describe([_num(r["duration_s"]) for r in done]),
        "time_to_first_token_s": _describe([_num(r["ttft_s"]) for r in done]),
        "tokens": {"prompt": int(sum(_num(r["prompt_tokens"]) or 0 for r in done)),
                   "completion": int(sum(_num(r["completion_tokens"]) or 0 for r in done)),
                   "per_request_tokens_per_s": _describe([_num(r["tokens_per_s"]) for r in done]),
                   "in_per_request": _describe([_num(r["prompt_tokens"]) for r in done]),
                   "out_per_request": _describe([_num(r["completion_tokens"]) for r in done])},
        "peak_concurrency": max((_num(i.get("max_active")) or 0 for i in intervals), default=0),
        **_busy_summary(meta, requests, done, hardware, end),
        "busiest_minute": {"time": iso(busiest[1]), "requests": busiest[0]} if busiest else None,
        "hardware": {"gpu_util_pct": hw("gpu_util"), "vram_pct": hw("vram_pct"), "ram_pct": hw("ram_pct"),
                     "gpu_temp_c": hw("gpu_temp")},
        "models": {m: {"requests": len(rs), **_describe([_num(r["duration_s"]) for r in rs])}
                   for m, rs in models.items()},
    }


def _busy_summary(meta, requests, done, hardware, end):
    """Figures that refer only to the time with at least one request in flight."""
    span = (meta["started"], end)
    periods = busy_periods(((_epoch(r.get("start")), _epoch(r.get("end"))) for r in requests), span)
    busy = busy_stats(periods, span)
    busy_s = busy["busy_s"]
    out_tokens = sum(_num(r["completion_tokens"]) or 0 for r in done)
    starts = [p[0] for p in periods]
    gpu_busy = [_num(h.get("gpu_util")) for h in hardware
                if _epoch(h.get("time")) is not None and periods and in_periods(_epoch(h["time"]), periods, starts)]
    return {
        "busy": busy,
        "while_busy": {
            "tokens_per_s": round(out_tokens / busy_s, 1) if busy_s >= 1 else None,
            "responses_per_min": round(len(done) / (busy_s / 60), 1) if busy_s >= 1 else None,
            "gpu_util_pct": _describe(gpu_busy),
        },
        "overall": {"tokens_per_s": round(out_tokens / busy["total_s"], 1) if busy["total_s"] >= 1 else None},
    }


def span_text(seconds):
    """Human-readable duration: seconds below a minute, minutes below two hours, hours above."""
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def write_report(folder):
    """Write summary.json and report.html for a session folder."""
    folder = Path(folder)
    s = summarise(folder)
    (folder / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    requests = _read_csv(folder / "requests.csv")
    done = [r for r in requests if r["status"] == "completed"]
    intervals = _read_csv(folder / "intervals.csv")
    hardware = _read_csv(folder / "hardware.csv")

    def sec(d, key):
        return f"{fmt_num(d[key])} s" if d.get(key) is not None else "–"

    rt, tt, hwd = s["response_time_s"], s["time_to_first_token_s"], s["hardware"]
    # Count tiles: (label, value, note). Distribution tiles: (label, statistics, unit).
    count_tiles = [
        ("Completed requests", fmt_num(s["requests"].get("completed", 0)), f"{fmt_num(s['requests']['per_hour'])} per hour"),
        ("Not answered", fmt_num(sum(s["requests"].get(k, 0) for k in ("failed", "rejected", "stale", "dismissed"))),
         ", ".join(f"{s['requests'][k]} {k}" for k in ("failed", "rejected", "stale", "dismissed")
                   if s["requests"].get(k)) or "none"),
        ("Busy time", f"{s['busy']['busy_share'] * 100:.0f} %",
         f"{span_text(s['busy']['busy_s'])} of {span_text(s['busy']['total_s'])} with requests · "
         f"longest idle {span_text(s['busy']['longest_idle_s'])}"),
        ("Throughput while busy", f"{fmt_num(s['while_busy']['tokens_per_s'])} tokens/s"
         if s["while_busy"]["tokens_per_s"] is not None else "–",
         f"{fmt_num(s['while_busy']['responses_per_min'])} responses/min while busy · overall "
         f"{fmt_num(s['overall']['tokens_per_s'])} tokens/s" if s["while_busy"]["tokens_per_s"] is not None else ""),
        ("Peak concurrency", fmt_num(s["peak_concurrency"]), "parallel requests (max per interval)"),
        ("Busiest minute", fmt_num(s["busiest_minute"]["requests"]) if s["busiest_minute"] else "–",
         f"requests at {s['busiest_minute']['time'][11:16]}" if s["busiest_minute"] else ""),
        ("Generated tokens", fmt_num(s["tokens"]["completion"]), f"{fmt_num(s['tokens']['prompt'])} prompt tokens"),
    ]
    # GPU utilisation only means something while requests run; the overall average is given as a note.
    gpu_busy = s["while_busy"]["gpu_util_pct"]
    gpu_tile = (("GPU utilisation while busy", gpu_busy, " %",
                 f"overall Ø {fmt_num(hwd['gpu_util_pct'].get('mean'))} % · n = {gpu_busy.get('n')}")
                if gpu_busy else ("GPU utilisation", hwd["gpu_util_pct"], " %"))
    stat_tiles = [("Response time", rt, " s"), ("First token", tt, " s"),
                  ("Tokens in per request", s["tokens"]["in_per_request"], ""),
                  ("Tokens out per request", s["tokens"]["out_per_request"], ""),
                  ("Tokens/s per request", s["tokens"]["per_request_tokens_per_s"], ""),
                  gpu_tile, ("GPU memory", hwd["vram_pct"], " %"),
                  ("System memory", hwd["ram_pct"], " %"), ("GPU temperature", hwd["gpu_temp_c"], " °C")]

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    span = (meta["started"], meta.get("ended") or time.time())
    charts = []
    m, v = _per_minute(done, span=span)
    if done:
        charts.append(time_chart("Completed requests per minute", m, [("requests", v)]))
        m_med, v_med = _per_minute(done, "duration_s", "median", span=span)
        _, v_p95 = _per_minute(done, "duration_s", "p95", span=span)
        charts.append(time_chart("Response time per minute", m_med, [("median", v_med), ("p95", v_p95)], unit=" s"))
    # Per-minute series over the whole session; minutes without data count as 0 (e.g. while the
    # monitor was not running). Response times are the exception: an idle minute has no response time.
    def zero(values):
        return [0 if v is None else v for v in values]

    if intervals:
        m_iv, active = _per_minute(intervals, "max_active", "max", span=span)
        _, tps = _per_minute(intervals, "gen_tokens_per_s_total", "mean", span=span)
        charts.append(time_chart("Concurrent requests (max per minute)", m_iv, [("parallel", zero(active))]))
        charts.append(time_chart("Generated tokens per second", m_iv, [("tokens/s", zero(tps))]))
    hw_min, gpu = _per_minute(hardware, "gpu_util", "mean", span=span)
    _, vram = _per_minute(hardware, "vram_pct", "max", span=span)
    _, ram = _per_minute(hardware, "ram_pct", "max", span=span)
    has_gpu, has_ram = any(x is not None for x in gpu + vram), any(x is not None for x in ram)
    gpu, vram, ram = zero(gpu), zero(vram), zero(ram)
    if hardware and has_gpu:
        charts.append(time_chart("GPU utilisation and memory", hw_min,
                                 [("GPU utilisation (avg)", gpu), ("GPU memory (max)", vram)], unit=" %", y_max=100))
    if hardware and has_ram:
        charts.append(time_chart("System memory (max)", hw_min, [("system memory", ram)], unit=" %", y_max=100))
    durations = [_num(r["duration_s"]) for r in done if _num(r["duration_s"]) is not None]
    if durations:
        charts.append(histogram_chart("Distribution of response times", durations, unit=" s"))

    model_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{d['requests']}</td><td>{sec(d, 'median')}</td><td>{sec(d, 'p95')}</td>"
        f"<td>{sec(d, 'mean')}</td></tr>" for name, d in sorted(s["models"].items(), key=lambda kv: -kv[1]["requests"]))
    host = s["host"]
    body = [f"<h1>{html.escape(s['label'])}</h1>",
            f"<p class='meta'>{html.escape(s['started'].replace('T', ' '))} – "
            f"{html.escape((s['ended'] or 'running').replace('T', ' '))} · {fmt_num(s['duration_min'])} min"
            + (f" · {html.escape(host.get('gpu_name') or '')}" if host.get("gpu_name") else "") + "</p>",
            tiles(count_tile(*t) for t in count_tiles),
            tiles(stat_tile(*t) for t in stat_tiles if t[1]),
            "<h2>Charts</h2><div class='charts'>" + "".join(f"<figure>{c}</figure>" for c in charts) + "</div>"
            if charts else "<p>No completed requests recorded yet.</p>",
            "<h2>Models</h2><div class='table'><table><thead><tr><th>Model</th><th>Requests</th><th>Median</th>"
            "<th>p95</th><th>Average</th></tr></thead><tbody>" + model_rows + "</tbody></table></div>" if model_rows else "",
            "<h2>Data</h2><p>Raw data in this folder: <code>requests.csv</code> (one row per request), "
            "<code>intervals.csv</code> (per interval), <code>hardware.csv</code> (every 5 s), "
            "<code>summary.json</code> (key figures). Only numbers, times and model names are stored – "
            "no prompts, responses, IP addresses or user identifiers.</p>"]
    extra_css = TILE_CSS
    page = html_page(f"Live session: {s['label']}", extra_css + "\n".join(body))
    (folder / "report.html").write_text(page, encoding="utf-8")
    return folder / "report.html"


def main():
    ap = argparse.ArgumentParser(description="Regenerate the report of a live session from its CSV files")
    ap.add_argument("--report", required=True, help="Session folder")
    args = ap.parse_args()
    print(write_report(args.report))


if __name__ == "__main__":
    main()
