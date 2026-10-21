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

from report_charts import fmt_num, histogram_chart, html_page, time_chart

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
        self._count_existing()

    def _count_existing(self):
        try:
            with open(self.folder / "requests.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.counts[row["status"]] = self.counts.get(row["status"], 0) + 1
        except OSError:
            pass

    def save_meta(self):
        tmp = self.folder / "session.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.folder / "session.json")

    # --- recording ---
    def record_request(self, req, status):
        """Write one finished request. Requests that ended before the session started are skipped."""
        end = req.get("end") or time.time()
        if end < self.meta["started"] or end <= self.meta.get("last_request_end", 0):
            return  # Before the session, or already written before a monitor restart.
        start, first = req.get("start"), req.get("first")
        duration = end - start if start else None
        out_tok = req.get("completion_tokens")
        tps = req.get("tps") or (out_tok / duration if out_tok and duration else None)
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
        self.meta["last_request_end"] = end
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
                   "per_request_tokens_per_s": _describe([_num(r["tokens_per_s"]) for r in done])},
        "peak_concurrency": max((_num(i.get("max_active")) or 0 for i in intervals), default=0),
        "busiest_minute": {"time": iso(busiest[1]), "requests": busiest[0]} if busiest else None,
        "hardware": {"gpu_util_pct": hw("gpu_util"), "vram_pct": hw("vram_pct"), "ram_pct": hw("ram_pct"),
                     "gpu_temp_c": hw("gpu_temp")},
        "models": {m: {"requests": len(rs), **_describe([_num(r["duration_s"]) for r in rs])}
                   for m, rs in models.items()},
    }


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
    tiles = [
        ("Completed requests", fmt_num(s["requests"].get("completed", 0)), f"{fmt_num(s['requests']['per_hour'])} per hour"),
        ("Not answered", fmt_num(sum(s["requests"].get(k, 0) for k in ("failed", "rejected", "stale", "dismissed"))),
         ", ".join(f"{s['requests'][k]} {k}" for k in ("failed", "rejected", "stale", "dismissed")
                   if s["requests"].get(k)) or "none"),
        ("Response time", sec(rt, "median"), f"median · Ø {sec(rt, 'mean')} · p95 {sec(rt, 'p95')} · max {sec(rt, 'max')}"),
        ("First token", sec(tt, "median"), f"median · Ø {sec(tt, 'mean')} · p95 {sec(tt, 'p95')}"),
        ("Peak concurrency", fmt_num(s["peak_concurrency"]), "parallel requests (max per interval)"),
        ("Busiest minute", fmt_num(s["busiest_minute"]["requests"]) if s["busiest_minute"] else "–",
         f"requests at {s['busiest_minute']['time'][11:16]}" if s["busiest_minute"] else ""),
        ("Generated tokens", fmt_num(s["tokens"]["completion"]), f"{fmt_num(s['tokens']['prompt'])} prompt tokens"),
        ("GPU", f"{fmt_num(hwd['gpu_util_pct'].get('mean'))} %" if hwd["gpu_util_pct"] else "–",
         f"Ø utilisation · max {fmt_num(hwd['gpu_util_pct'].get('max'))} %" if hwd["gpu_util_pct"] else "no GPU data"),
        ("GPU memory", f"{fmt_num(hwd['vram_pct'].get('max'))} %" if hwd["vram_pct"] else "–", "max"),
        ("System memory", f"{fmt_num(hwd['ram_pct'].get('max'))} %" if hwd["ram_pct"] else "–", "max"),
    ]

    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    span = (meta["started"], meta.get("ended") or time.time())
    charts = []
    m, v = _per_minute(done, span=span)
    if done:
        charts.append(time_chart("Completed requests per minute", m, [("requests", v)]))
        m_med, v_med = _per_minute(done, "duration_s", "median", span=span)
        _, v_p95 = _per_minute(done, "duration_s", "p95", span=span)
        charts.append(time_chart("Response time per minute", m_med, [("median", v_med), ("p95", v_p95)], unit=" s"))
    it = [(_epoch(i.get("time")), i) for i in intervals]
    it = [(t, i) for t, i in it if t is not None]
    if it:
        charts.append(time_chart("Concurrent requests (max per interval)", [t for t, _ in it],
                                 [("parallel", [_num(i.get("max_active")) for _, i in it])]))
        charts.append(time_chart("Generated tokens per second", [t for t, _ in it],
                                 [("tokens/s", [_num(i.get("gen_tokens_per_s_total")) for _, i in it])]))
    hw_min, gpu = _per_minute(hardware, "gpu_util", "mean")
    _, vram = _per_minute(hardware, "vram_pct", "max")
    _, ram = _per_minute(hardware, "ram_pct", "max")
    if hw_min and any(x is not None for x in gpu + vram):
        charts.append(time_chart("GPU utilisation and memory", hw_min,
                                 [("GPU utilisation (avg)", gpu), ("GPU memory (max)", vram)], unit=" %", y_max=100))
    if hw_min and any(x is not None for x in ram):
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
            "<div class='tiles'>" + "".join(
                f"<div class='tile'><div class='label'>{html.escape(a)}</div><div class='big'>{html.escape(b)}</div>"
                f"<div class='sub'>{html.escape(c)}</div></div>" for a, b, c in tiles) + "</div>",
            "<h2>Charts</h2><div class='charts'>" + "".join(f"<figure>{c}</figure>" for c in charts) + "</div>"
            if charts else "<p>No completed requests recorded yet.</p>",
            "<h2>Models</h2><div class='table'><table><thead><tr><th>Model</th><th>Requests</th><th>Median</th>"
            "<th>p95</th><th>Average</th></tr></thead><tbody>" + model_rows + "</tbody></table></div>" if model_rows else "",
            "<h2>Data</h2><p>Raw data in this folder: <code>requests.csv</code> (one row per request), "
            "<code>intervals.csv</code> (per interval), <code>hardware.csv</code> (every 5 s), "
            "<code>summary.json</code> (key figures). Only numbers, times and model names are stored – "
            "no prompts, responses, IP addresses or user identifiers.</p>"]
    extra_css = """<style>
.meta { color: var(--text-2); margin-top: -6px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 16px 0; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
.tile .label { color: var(--text-2); font-size: 12px; } .tile .big { font-size: 24px; font-weight: 650; margin-top: 2px; }
.tile .sub { color: var(--text-3); font-size: 12px; }
</style>"""
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
