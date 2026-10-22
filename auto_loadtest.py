#!/usr/bin/env python3
"""
Automated capacity test for AnythingLLM backed by LM Studio, or for LM Studio directly.

Runs a Locust user file headless several times with an increasing number of
concurrent users: main.py for AnythingLLM (--target anythingllm, the default) or
lmstudio_user.py for LM Studio's OpenAI-compatible API (--target lmstudio). During each stage it samples GPU/host metrics and, if the
LM Studio log is available, token throughput. As soon as a stage violates a
limit (failure rate, p95 latency, VRAM, GPU temperature) the run stops; with
--refine the exact limit is then located by bisection. Results are written as
CSV/JSON plus a Markdown report to the results directory.

Any argument not recognised by this script is forwarded to the Locust user file
(e.g. --mode chat --wait-min 2 --wait-max 5 for AnythingLLM, --model ID --context-tokens 1500
for LM Studio).

Examples:
    python auto_loadtest.py --users 1,2,4,8,16,32,48,64 --stage-duration 300 --refine -- --mode query
    python auto_loadtest.py --target lmstudio --users 1,2,4,8,16,32 -- --context-tokens 1500
"""
import argparse
import csv
import html
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from live_session import busy_periods, busy_stats
from lmstudio_monitor import LMStudioWatcher, find_log_target
from report_charts import TILE_CSS, count_tile, html_page, line_chart, markdown_to_html, stat_tile, tiles, time_chart
from system_metrics import SystemSampler

BASE_DIR = Path(__file__).resolve().parent

# What differs between the systems under test; everything else (stages, limits, report) is shared.
TARGETS = {
    "anythingllm": {"label": "AnythingLLM", "host": "http://localhost:3001", "locustfile": "main.py",
                    "chat": "/thread/chat"},
    "lmstudio": {"label": "LM Studio (direct)", "host": "http://localhost:1234", "locustfile": "lmstudio_user.py",
                 "chat": "/v1/chat/completions"},
}
CHAT_PREFIXES = tuple(t["chat"] for t in TARGETS.values())  # Names of the chat entries in the Locust statistics.


def num(value):
    """Convert a Locust CSV cell to float, returning None for 'N/A' or empty cells."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


REFUSAL_TYPES = ("REFUSAL", "REJECTED")  # Entry types reported by main.py next to the HTTP requests.


def final(path):
    """The complete statistics written when Locust quits (see locust_final_stats.py), if present."""
    complete = path.with_name(path.stem + "_final" + path.suffix)
    return complete if complete.exists() else path


def read_locust_stats(path):
    """
    Parse a Locust *_stats.csv file. Returns (totals, chat row, refusal counts), where totals
    sums the HTTP requests only (refusal entries are excluded) and refusal counts maps
    e.g. "REJECTED http-503" or "REFUSAL no-context" to the number of occurrences.
    """
    path = final(path)
    if not path.exists():
        return None, None, {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    http_rows = [r for r in rows if r.get("Name") != "Aggregated" and r.get("Type") not in REFUSAL_TYPES]
    totals = {"Request Count": sum(int(num(r["Request Count"]) or 0) for r in http_rows),
              "Failure Count": sum(int(num(r["Failure Count"]) or 0) for r in http_rows)}
    chat = next((r for r in http_rows if r.get("Name", "").startswith(CHAT_PREFIXES)), None)
    refusals = {f"{r['Type']} {r['Name']}": int(num(r["Request Count"]) or 0)
                for r in rows if r.get("Type") in REFUSAL_TYPES}
    return totals, chat, refusals


PERCENTILE_COLUMNS = ("50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%", "99.9%", "100%")


def chat_percentiles(chat):
    """Response time percentiles of the chat requests (seconds) from a Locust stats row."""
    if not chat:
        return None
    values = {col: num(chat.get(col)) for col in PERCENTILE_COLUMNS}
    return {col: round(v / 1000, 2) for col, v in values.items() if v is not None} or None


def read_history(path):
    """Per-second history of the chat requests from Locust (users, responses/s, median and p95 in seconds)."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Name", "").startswith(CHAT_PREFIXES)]
    out = []
    for r in rows:
        t = num(r.get("Timestamp"))
        if t is None:
            continue
        ms = lambda key: round(num(r.get(key)) / 1000, 2) if num(r.get(key)) is not None else None  # noqa: E731
        out.append({"t": t, "users": num(r.get("User Count")), "rps": num(r.get("Requests/s")),
                    "median": ms("50%"), "p95": ms("95%")})
    return out


def read_refusal_examples(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def color(text, code):
    """Wrap text in an ANSI color if the console supports it."""
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if USE_COLOR and os.name == "nt":
    os.system("")  # Enables ANSI escape sequences in the Windows console.


SETUP_HINTS = {
    "anythingllm": ("Nearly all requests failed at the lowest load. This usually indicates a configuration problem, "
                    "not a capacity limit: check the LLM provider in AnythingLLM. When AnythingLLM runs in Docker, "
                    "LM Studio must be configured as http://host.docker.internal:1234/v1 and "
                    "'Serve on Local Network' must be enabled in LM Studio."),
    "lmstudio": ("Nearly all requests failed at the lowest load. This usually indicates a configuration problem, "
                 "not a capacity limit: check that the LM Studio server is running, that the model can be loaded "
                 "(--model) and, if 'Require API token' is enabled, that LMSTUDIO_API_TOKEN is set."),
}
SETUP_HINT = SETUP_HINTS["anythingllm"]  # Replaced by the hint of the selected target in main().


def looks_like_setup_problem(result):
    chat_total = result["chat_requests"]
    return result["stage"] == 1 and chat_total > 0 and result["chat_failures"] / chat_total >= 0.9


def check_locust():
    """Exit with a clear message if Locust cannot be run by this Python interpreter."""
    try:
        out = subprocess.run([sys.executable, "-m", "locust", "--version"], capture_output=True, text=True, timeout=120)
        ok = out.returncode == 0
    except (subprocess.SubprocessError, OSError):
        ok = False
    if not ok:
        sys.exit(f"ERROR: Locust cannot be run with this Python interpreter ({sys.executable}).\n"
                 f"Install it for this interpreter with: \"{sys.executable}\" -m pip install locust")


def read_failures(path, limit=5):
    """Return the most frequent (error, occurrences) pairs from a Locust *_failures.csv file."""
    path = final(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: -int(num(r.get("Occurrences")) or 0))
    return [(r.get("Error", "")[:300], int(num(r.get("Occurrences")) or 0)) for r in rows[:limit]]


def log_tail(path, lines=25):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def preflight_lmstudio(host, model=None):
    """Verify that LM Studio is reachable and offers a language model (or the requested one)."""
    token = os.environ.get("LMSTUDIO_API_TOKEN")
    req = urllib.request.Request(f"{host}/v1/models", headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ids = [m.get("id", "") for m in json.loads(resp.read()).get("data", [])]
    except urllib.error.HTTPError as exc:
        sys.exit(f"ERROR: LM Studio refused the model list (HTTP {exc.code}). "
                 "If 'Require API token' is enabled, set LMSTUDIO_API_TOKEN.")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        sys.exit(f"ERROR: LM Studio is not reachable at {host} ({exc}). Start the server in LM Studio.")
    llms = [i for i in ids if "embed" not in i.lower()]
    if model and model not in ids:
        sys.exit(f"ERROR: LM Studio does not offer the model '{model}'. Available: {', '.join(llms) or 'none'}")
    if not llms:
        sys.exit("ERROR: LM Studio offers no language model. Download or load one first.")
    print(f"[preflight] LM Studio reachable, model {model or 'auto (first loaded model)'}; available: {', '.join(llms)}.")


def preflight(host):
    """Verify that AnythingLLM is reachable and the API key is valid before starting."""
    key, ws = os.environ.get("ANYTHING_API_KEY"), os.environ.get("WORKSPACE_SLUG")
    if not key or not ws:
        sys.exit("ERROR: the environment variables ANYTHING_API_KEY and WORKSPACE_SLUG must be set.")
    req = urllib.request.Request(f"{host}/api/v1/auth", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if not json.loads(resp.read()).get("authenticated"):
                sys.exit("ERROR: AnythingLLM rejected the API key.")
    except urllib.error.HTTPError as exc:
        sys.exit(f"ERROR: AnythingLLM authentication failed (HTTP {exc.code}). Check the API key.")
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"ERROR: AnythingLLM is not reachable at {host} ({exc}).")
    print(f"[preflight] AnythingLLM reachable, API key valid, workspace '{ws}'.")


def build_levels(args):
    """Return the ordered list of user counts to test."""
    if args.users:
        return sorted({int(u) for u in args.users.split(",") if u.strip()})
    levels, u = [], args.start_users
    while u <= args.max_users:
        levels.append(u)
        u = max(u + 1, math.ceil(u * args.step_factor)) if args.step_factor > 1 else u + args.step
    return levels


# --- Control by the live monitor's dashboard ---
STATUS_FILE = BASE_DIR / ".loadtest_status.json"  # Written while a run is active (heartbeat, stage).
STOP_FILE = BASE_DIR / ".loadtest_stop"            # Created by the dashboard's stop button.


def write_status(**fields):
    """Publish the run's state for the dashboard; written atomically."""
    try:
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "heartbeat": time.time(), **fields}), encoding="utf-8")
        os.replace(tmp, STATUS_FILE)
    except OSError:
        pass


def clear_status():
    for path in (STATUS_FILE, STATUS_FILE.with_suffix(".tmp")):
        try:
            path.unlink()
        except OSError:
            pass


def stop_requested():
    return STOP_FILE.exists()


class StatusPublisher(threading.Thread):
    """Keeps the status file fresh (every 2 s) for as long as the run is active."""

    def __init__(self, run, target="anythingllm", name=""):
        super().__init__(daemon=True)
        self.fields = {"run": run, "name": name, "stage": 0, "users": 0, "stage_started": time.time(), "target": target,
                       "workspace": os.environ.get("WORKSPACE_SLUG") if target == "anythingllm" else None,
                       "run_started": time.time()}
        self._stop = threading.Event()

    def update(self, **fields):
        self.fields.update(fields)
        write_status(**self.fields)

    def run(self):
        while not self._stop.is_set():
            write_status(**self.fields)
            self._stop.wait(2)

    def close(self):
        self._stop.set()
        clear_status()


STATUS = None  # StatusPublisher of the current run.


def pause(seconds):
    """Sleep between stages; returns True as soon as a stop is requested from the dashboard."""
    end = time.time() + seconds
    while time.time() < end:
        if stop_requested():
            return True
        time.sleep(min(1.0, max(0.0, end - time.time())))
    return stop_requested()


def popen_flags():
    # A separate process group lets us send Ctrl+Break to Locust on Windows without hitting ourselves.
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}


def stop_locust(proc, grace=15):
    """Ask Locust to stop (it then writes its statistics); kill it if it does not exit in time."""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        proc.wait(timeout=grace)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        proc.kill()
        proc.wait()


def token_stats(requests, span=None):
    """
    Average, median, p95 and maximum of the input and output tokens per request (from the LM Studio
    log), plus the share of the stage with a request in flight and the throughput during that time.
    """
    out = {}
    if span and requests:
        periods = busy_periods(((r["start"], r["end"]) for r in requests), span)
        busy = busy_stats(periods, span)
        out_tokens = sum(r.get("tokens_out") or 0 for r in requests)
        out.update({"busy_share": busy["busy_share"],
                    "tokens_per_s_busy": round(out_tokens / busy["busy_s"], 1) if busy["busy_s"] >= 1 else None})
    for key in ("tokens_in", "tokens_out", "ttft", "tps"):
        values = sorted(r[key] for r in requests if r.get(key) is not None)
        if not values:
            continue
        out.update({f"{key}_mean": round(sum(values) / len(values), 2), f"{key}_median": round(statistics.median(values), 2),
                    f"{key}_p95": round(values[max(0, min(len(values) - 1, round(0.95 * len(values)) - 1))], 2),
                    f"{key}_max": round(values[-1], 2), f"{key}_n": len(values)})
    return out


def run_stage(idx, users, args, passthrough, out_dir, sampler, watcher):
    """Run one Locust stage with ``users`` concurrent users and evaluate it against the limits."""
    prefix = out_dir / f"stage_{idx:02d}_u{users:03d}"
    ramp = math.ceil(users / args.spawn_rate)
    cmd = [sys.executable, "-m", "locust", "-f", str(args.locustfile), "--headless",
           "--host", args.host, "-u", str(users), "-r", str(args.spawn_rate),
           "-t", f"{ramp + args.stage_duration}s", "--reset-stats",
           "--stop-timeout", str(args.stop_timeout), "--only-summary",
           "--csv", str(prefix), "--csv-full-history", "--html", f"{prefix}.html",
           "--refusal-log", f"{prefix}_refusals.jsonl",
           "--run-id", f"{out_dir.name}-s{idx:02d}",  # Thread names identify run and stage.
           *passthrough]
    print(f"\n=== Stage {idx}: {users} users  (ramp-up {ramp}s + {args.stage_duration}s measurement) ===")

    sampler.summary(reset=True)
    if watcher:
        watcher.snapshot(reset=True)
    t0 = time.time()
    stopped = None
    STATUS.update(stage=idx, users=users, stage_started=t0)
    with open(f"{prefix}.log", "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=args.locustfile.parent, stdout=logf, stderr=subprocess.STDOUT,
                                **popen_flags())
        try:
            while proc.poll() is None:
                if stop_requested():
                    stopped = "stopped from the dashboard"
                    print(color("  !!! STOPPED FROM THE DASHBOARD – stopping Locust now", "1;37;41"))
                    stop_locust(proc)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            stop_locust(proc)
            raise
    hardware_timeline = sampler.samples_since(t0)
    sysm = sampler.summary(reset=True)
    lm = watcher.snapshot(reset=True) if watcher else {}
    tokens = token_stats(watcher.parser.completed_since(t0), (t0, time.time())) if watcher else {}

    agg, chat, refusals = read_locust_stats(Path(f"{prefix}_stats.csv"))
    requests = int(num(agg["Request Count"]) or 0) if agg else 0
    failures = int(num(agg["Failure Count"]) or 0) if agg else 0

    def seconds(key):
        value = num(chat.get(key)) if chat else None
        return round(value / 1000, 2) if value is not None else None

    result = {
        "stage": idx, "users": users, "duration_s": round(time.time() - t0),
        "locust_exit": proc.returncode, "requests": requests, "failures": failures,
        "fail_rate": round(failures / requests, 4) if requests else None,
        "chat_requests": int(num(chat["Request Count"]) or 0) if chat else 0,
        "chat_failures": int(num(chat["Failure Count"]) or 0) if chat else 0,
        "rejected": sum(n for k, n in refusals.items() if k.startswith("REJECTED")),
        "refused_no_context": refusals.get("REFUSAL no-context", 0),
        "chat_rps": round(num(chat["Requests/s"]) or 0, 4) if chat else 0,
        "chat_median_s": seconds("Median Response Time"), "chat_avg_s": seconds("Average Response Time"),
        "chat_p95_s": seconds("95%"), "chat_p99_s": seconds("99%"), "chat_max_s": seconds("Max Response Time"),
        **sysm,
        "lm_gen_tokens_per_s": lm.get("gen_tokens_per_s_total"),
        "lm_tokens_per_s_per_request": lm.get("tokens_per_s_per_request_avg"),
        "lm_ttft_p95_s": lm.get("ttft_p95_s"), "lm_max_active": lm.get("max_active"),
        "lm_completion_tokens": lm.get("completion_tokens"), "lm_prompt_tokens": lm.get("prompt_tokens"),
        "models": lm.get("models_used") or None,
        "chat_percentiles": chat_percentiles(chat),
        "timeline": read_history(Path(f"{prefix}_stats_history.csv")),
        "hardware_timeline": hardware_timeline,
        **tokens,
    }

    result["refusal_rate"] = (round((result["rejected"] + result["refused_no_context"]) / result["chat_requests"], 4)
                              if result["chat_requests"] else None)
    result["refusal_breakdown"] = refusals
    result["refusal_examples"] = read_refusal_examples(f"{prefix}_refusals.jsonl")
    result["top_errors"] = read_failures(Path(f"{prefix}_failures.csv"))
    result["stopped"] = stopped
    # Without statistics Locust did not run at all: a setup problem, not a capacity limit.
    result["error"] = None if agg is not None or stopped else "\n".join(log_tail(f"{prefix}.log"))

    reasons = [stopped] if stopped else []
    if agg is None and not stopped:
        reasons.append("Locust failed to run (see .log)")
    elif result["chat_requests"] == 0:
        reasons.append("no chat request completed")
    if result["fail_rate"] is not None and result["fail_rate"] > args.max_fail_rate:
        reasons.append(f"failure rate {result['fail_rate']:.1%} > {args.max_fail_rate:.0%}")
    if result["chat_p95_s"] is not None and result["chat_p95_s"] > args.max_p95:
        reasons.append(f"p95 {result['chat_p95_s']}s > {args.max_p95}s")
    if sysm.get("vram_pct_max") is not None and sysm["vram_pct_max"] > args.max_vram_pct:
        reasons.append(f"VRAM {sysm['vram_pct_max']}% > {args.max_vram_pct}%")
    if sysm.get("gpu_temp_max") is not None and sysm["gpu_temp_max"] > args.max_gpu_temp:
        reasons.append(f"GPU {sysm['gpu_temp_max']}°C > {args.max_gpu_temp}°C")
    result["passed"] = not reasons
    result["reasons"] = "; ".join(reasons)

    print(f"  requests {requests}, failures {failures}, chat p95 {result['chat_p95_s']}s, "
          f"{result['chat_rps']} chat req/s, GPU avg {sysm.get('gpu_util_avg')}%, VRAM max {sysm.get('vram_pct_max')}%"
          + (f", {result['lm_gen_tokens_per_s']} tok/s" if watcher else ""))
    if result["error"] is not None:
        print("  -> LOCUST ERROR (setup problem, not a capacity limit). Last lines of the Locust log:")
        print("\n".join("     " + line for line in result["error"].splitlines()))
        return result
    if refusals:
        rate = result["refusal_rate"] or 0
        detail = ", ".join(f"{n}x {k}" for k, n in sorted(refusals.items(), key=lambda kv: -kv[1]))
        line = f"  !! REFUSALS: {rate:.1%} of chat requests ({detail})"
        print(color(line, "1;31" if rate >= args.refusal_warn_rate else "1;33"))
    print("  -> " + ("PASSED" if result["passed"] else "STOPPED" if stopped else "LIMIT REACHED: " + result["reasons"]))
    for error, count in result["top_errors"][:3]:
        print(f"     {count}x {error}")
    if looks_like_setup_problem(result):
        print(f"     Note: {SETUP_HINT}")
    return result


def write_results(out_dir, results, args, last_ok, first_bad, aborted, user_stop=False):
    """Write results.csv, results.json and the Markdown report."""
    if results:
        # Union of all keys in stage order: later stages can have fields the first one lacks (e.g. token
        # statistics when no request completed in stage 1). Nested data stays in results.json only.
        nested = ("top_errors", "error", "refusal_breakdown", "refusal_examples", "chat_percentiles", "timeline",
                  "hardware_timeline")
        fields = list(dict.fromkeys(k for r in results for k in r if k not in nested))
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(results)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    ordered = sorted(results, key=lambda r: r["users"])
    passing = [r for r in ordered if r["passed"]]
    # Saturation: throughput grows by less than 5 % despite more users.
    knee = None
    for prev, cur in zip(ordered, ordered[1:]):
        if prev["chat_rps"] and cur["chat_rps"] < prev["chat_rps"] * 1.05:
            knee = prev["users"]
            break
    best = max(passing, key=lambda r: r["chat_rps"], default=None)

    lines = [f"# {report_title(args, out_dir)}", "",
             *([f"- Run: `{out_dir.name}`"] if args.name else []),
             f"- Target: {TARGETS[args.target]['label']}, host `{args.host}`, stage duration {args.stage_duration}s, "
             f"forwarded to {args.locustfile.name}: `{' '.join(args.passthrough) or '-'}`",
             f"- Limits: failure rate ≤ {args.max_fail_rate:.0%}, chat p95 ≤ {args.max_p95}s, "
             f"VRAM ≤ {args.max_vram_pct}%, GPU ≤ {args.max_gpu_temp}°C", ""]
    if aborted:
        lines += ["**The test was aborted manually; results are incomplete.**", ""]
    if user_stop and not any(r.get("stopped") for r in results):
        lines += ["**The test was stopped from the dashboard between stages; later stages were not run.**", ""]
    lines += ["## Result", ""]
    stopped = next((r for r in results if r.get("stopped")), None)
    if stopped is not None:
        lines += [f"### ⛔ Stopped from the dashboard at {stopped['users']} users (stage {stopped['stage']})", "",
                  "Locust was stopped immediately; no further stages were run. Results of this stage cover "
                  "only the time until the stop.", ""]
    failed_run = next((r for r in results if r.get("error") is not None), None)
    if failed_run is not None:
        lines += [f"**The run stopped at stage {failed_run['stage']} because Locust failed to run. "
                  "This is a setup problem, not a capacity limit.**", "",
                  "Last lines of the Locust log:", "", "```", failed_run["error"], "```", ""]
    if failed_run is None and last_ok is None and first_bad is not None:
        lines.append(f"- A limit was already exceeded at **{first_bad} Locust users**.")
    elif failed_run is None and last_ok is not None:
        lines.append(f"- Highest passing stage: **{last_ok} concurrent Locust users**"
                     + (f" (limit at {first_bad})" if first_bad else " (limit not reached; increase --max-users)"))
    if knee:
        lines.append(f"- Throughput saturates at about **{knee} users** (more users only increase waiting time).")
    if best and best["chat_rps"]:
        cycle = args.student_think_time + (best["chat_avg_s"] or 0)
        est = best["chat_rps"] * cycle
        lines.append(f"- Maximum stable throughput: **{best['chat_rps']:.3f} responses/s** "
                     f"({best['chat_rps'] * 60:.1f}/min) at {best['users']} users.")
        lines.append(f"- Estimated real users (Little's law, {args.student_think_time}s think/read time "
                     f"between questions): about **{est:.0f} concurrently active students**.")
    first_fail = next((r for r in ordered if not r["passed"] and r.get("top_errors")), None)
    if first_fail is not None:
        lines += ["", f"Most frequent errors at {first_fail['users']} users:", ""]
        lines += [f"- {count}× `{error}`" for error, count in first_fail["top_errors"]]
        if looks_like_setup_problem(first_fail):
            lines += ["", f"> {SETUP_HINT}"]
    refusal_stages = [r for r in ordered if r.get("refusal_breakdown")]
    if refusal_stages:
        lines += ["", "## ⚠ Refusals", "",
                  "Requests that were answered with a refusal or rejected by the server, per stage. "
                  "Technical rejections also count as failures; \"no relevant documents\" refusals (AnythingLLM "
                  "query mode only) count as answered.", "",
                  "| Users | Chat requests | Rejected (technical) | No relevant documents | Refusal rate |",
                  "|---:|---:|---:|---:|---:|"]
        for r in ordered:
            rate = "-" if r.get("refusal_rate") is None else f"**{r['refusal_rate'] * 100:.1f} %**" \
                if r["refusal_rate"] >= args.refusal_warn_rate else f"{r['refusal_rate'] * 100:.1f} %"
            lines.append(f"| {r['users']} | {r['chat_requests']} | {r['rejected']} | {r['refused_no_context']} | {rate} |")
        rates = [r["refusal_rate"] or 0 for r in ordered]
        if len(rates) > 1 and rates[-1] > rates[0] + 0.05:
            lines += ["", f"Refusals increase with load: {rates[0]:.0%} at {ordered[0]['users']} users, "
                          f"{rates[-1]:.0%} at {ordered[-1]['users']} users."]
        totals = {}
        for r in ordered:
            for key, n in r["refusal_breakdown"].items():
                totals[key] = totals.get(key, 0) + n
        lines += ["", "By reason: " + ", ".join(f"`{k}` {n}×" for k, n in sorted(totals.items(), key=lambda kv: -kv[1]))]
        examples, seen = [], {}
        for r in ordered:
            for ex in r.get("refusal_examples", []):
                if seen.get(ex["category"], 0) < 3:
                    seen[ex["category"]] = seen.get(ex["category"], 0) + 1
                    examples.append((r["users"], ex))
        if examples:
            lines += ["", "Examples (synthetic prompt → start of the response):", ""]
            for users, ex in examples:
                response = " ".join(ex["response"].split()).replace("|", "\\|") or "(empty)"
                thread = f", thread `{ex['thread']}`" if ex.get("thread") else ""
                lines.append(f"- `{ex['category']}` at {users} users{thread}: „{ex['prompt']}“ → {response}")

    charts_at = len(lines)  # The charts section is inserted here (before the stage table).
    lines += ["", "## Stages", "",
              "| Users | Requests | Failures % | Refusals % | Chat avg s | Chat p95 s | Chat req/s | GPU avg % | VRAM max % | RAM max % | tok/s | Status |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]

    def cell(v):
        return "-" if v is None else v

    for r in ordered:
        fail = "-" if r["fail_rate"] is None else f"{r['fail_rate'] * 100:.1f}"
        refused = "-" if r.get("refusal_rate") is None else f"{r['refusal_rate'] * 100:.1f}"
        lines.append(f"| {r['users']} | {r['requests']} | {fail} | {refused} | {cell(r['chat_avg_s'])} | {cell(r['chat_p95_s'])} | "
                     f"{r['chat_rps']} | {cell(r['gpu_util_avg'])} | {cell(r['vram_pct_max'])} | {cell(r['ram_pct_max'])} | "
                     f"{cell(r['lm_gen_tokens_per_s'])} | {'OK' if r['passed'] else r['reasons']} |")
    lines += ["", "Note: Locust users only wait for the configured think time (--wait-min/--wait-max). "
              "Real students send requests less frequently, hence the Little's law estimate above.",
              "Per-stage details: `stage_XX_uYYY.html` (Locust report) and `.log`."]
    charts = build_charts(ordered, args)
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    md_charts = ["", "## Charts", ""]
    for name, title, standalone, _ in charts:
        (chart_dir / f"{name}.svg").write_text(standalone, encoding="utf-8")
        md_charts.append(f"![{title}](charts/{name}.svg)")
    md = lines[:charts_at] + (md_charts if charts else []) + lines[charts_at:]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # report.html is a single self-contained file in the same layout as the live session reports.
    summary = {"last_ok": last_ok, "first_bad": first_bad, "knee": knee, "best": best, "aborted": aborted,
               "user_stop": user_stop}
    (out_dir / "report.html").write_text(html_report(out_dir, ordered, results, args, summary, lines, charts),
                                         encoding="utf-8")


def report_title(args, out_dir):
    """The load test's name if one was given, otherwise "Load test <folder>" (the folder stays the unique id)."""
    return args.name or f"Load test {out_dir.name}"


def passthrough_value(args, flag):
    """Value of ``flag`` among the arguments forwarded to main.py, or None."""
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def html_report(out_dir, ordered, results, args, summary, lines, charts):
    """Load test report with key figure tiles, statistics tiles, charts and the detailed sections."""
    last_ok, first_bad, knee, best = summary["last_ok"], summary["first_bad"], summary["knee"], summary["best"]
    stopped = next((r for r in results if r.get("stopped")), None)
    failed = next((r for r in results if r.get("error") is not None), None)
    chat_total = sum(r["chat_requests"] for r in ordered)
    fail_total = sum(r["chat_failures"] for r in ordered)
    refused = sum(r["rejected"] + r["refused_no_context"] for r in ordered)
    # Statistics refer to the operating point that matters in practice: the stage with the highest
    # stable throughput (otherwise the highest passing stage, otherwise the last stage run).
    ref = best or next((r for r in reversed(ordered) if r["passed"]), None) or (ordered[-1] if ordered else None)

    if last_ok is not None:
        limit_note = f"limit at {first_bad} users" if first_bad else "limit not reached – increase the users"
    else:
        limit_note = "a limit was exceeded at the first stage" if first_bad else "no stage passed"
    not_answered = {"failed": fail_total, "rejected": sum(r["rejected"] for r in ordered),
                    "no relevant documents": sum(r["refused_no_context"] for r in ordered)}

    # --- Result: the capacity of the system ---
    result_tiles = [
        count_tile("Highest passing stage", f"{last_ok} users" if last_ok is not None else "–", limit_note),
        count_tile("Throughput saturation", f"{knee} users" if knee else "not reached",
                   "more users only add waiting time" if knee else "throughput still grows with users"),
        count_tile("Max stable throughput", f"{best['chat_rps'] * 60:.1f}/min" if best else "–",
                   f"{best['chat_rps']:.2f} responses/s at {best['users']} users" if best else ""),
        count_tile("Estimated students", f"≈ {best['chat_rps'] * (args.student_think_time + (best['chat_avg_s'] or 0)):.0f}"
                   if best else "–", f"Little's law, {args.student_think_time:g} s between questions"),
    ]

    at = f" at {ref['users']} users" if ref else ""

    def pick(prefix_or_keys, n=None):
        """Statistics of the reference stage: either a key prefix (prefix_mean, …) or explicit keys."""
        keys = ({"mean": f"{prefix_or_keys}_mean", "median": f"{prefix_or_keys}_median", "p95": f"{prefix_or_keys}_p95",
                 "max": f"{prefix_or_keys}_max"} if isinstance(prefix_or_keys, str) else prefix_or_keys)
        stats = {k: ref.get(v) for k, v in keys.items()} if ref else {}
        stats["n"] = n
        return stats if any(stats.get(k) is not None for k in ("mean", "median", "p95", "max")) else None

    def stat(label, stats, unit):
        return stat_tile(label + at, stats, unit, note=None if stats.get("n") is not None else "") if stats else ""

    # --- Requests ---
    request_tiles = [
        count_tile("Chat requests", f"{chat_total:,}", f"over {len(ordered)} stages"),
        count_tile("Not answered", f"{sum(not_answered.values()):,}",
                   ", ".join(f"{n} {k}" for k, n in not_answered.items() if n) or "none"),
        count_tile("Refusals", f"{refused / chat_total:.1%}" if chat_total else "–",
                   (f"{refused} of {chat_total} (rejected or no relevant documents)" if args.target == "anythingllm"
                    else f"{refused} of {chat_total} rejected") if chat_total else ""),
    ]
    if ref and ref.get("lm_max_active") is not None:
        request_tiles.append(count_tile("Peak concurrency in LM Studio", f"{ref['lm_max_active']:g}",
                                        f"parallel requests{at}"))
    if ref and ref.get("busy_share") is not None:
        request_tiles.append(count_tile("Busy time", f"{ref['busy_share'] * 100:.0f} %",
                                        f"share of the stage with a request in flight{at}"))
    if ref and ref.get("models"):
        names = [m.rsplit(":", 1)[0] for m in ref["models"].split(";")]
        request_tiles.append(count_tile("Models", str(len(names)), ", ".join(names) + at))

    # --- Response times ---
    response_tiles = [
        stat("Response time", pick({"mean": "chat_avg_s", "median": "chat_median_s", "p95": "chat_p95_s",
                                    "max": "chat_max_s"}, ref["chat_requests"] if ref else None), " s"),
        stat("First token (LM Studio)", pick("ttft", ref.get("ttft_n") if ref else None), " s"),
    ]

    # --- Tokens ---
    token_tiles = []
    if ref and ref.get("tokens_per_s_busy") is not None:
        token_tiles.append(count_tile("Throughput while busy", f"{ref['tokens_per_s_busy']:g} tokens/s",
                                      f"overall {ref.get('lm_gen_tokens_per_s') or '–'} tokens/s{at}"))
    elif ref and ref.get("lm_gen_tokens_per_s") is not None:
        token_tiles.append(count_tile("Generated tokens/s", f"{ref['lm_gen_tokens_per_s']:g}", at.strip()))
    if any(r.get("lm_completion_tokens") for r in ordered):
        token_tiles.append(count_tile("Generated tokens", f"{sum(r.get('lm_completion_tokens') or 0 for r in ordered):,}",
                                      f"{sum(r.get('lm_prompt_tokens') or 0 for r in ordered):,} prompt tokens, all stages"))
    token_tiles += [stat("Tokens in per request", pick("tokens_in", ref.get("tokens_in_n") if ref else None), ""),
                    stat("Tokens out per request", pick("tokens_out", ref.get("tokens_out_n") if ref else None), ""),
                    stat("Tokens/s per request", pick("tps", ref.get("tps_n") if ref else None), "")]

    # --- Hardware ---
    hardware_tiles = [
        stat("GPU utilisation", pick({"mean": "gpu_util_avg", "median": "gpu_util_median", "p95": "gpu_util_p95",
                                      "max": "gpu_util_max"}), " %"),
        stat("GPU memory", pick({"mean": "vram_pct_avg", "median": "vram_pct_median", "p95": "vram_pct_p95",
                                 "max": "vram_pct_max"}), " %"),
        stat("System memory", pick({"mean": "ram_pct_avg", "median": "ram_pct_median", "p95": "ram_pct_p95",
                                    "max": "ram_pct_max"}), " %"),
        stat("GPU temperature", pick({"mean": "gpu_temp_avg", "median": "gpu_temp_median", "p95": "gpu_temp_p95",
                                      "max": "gpu_temp_max"}), " °C"),
    ]

    banners = []
    if stopped:
        banners.append(f"Stopped from the dashboard at {stopped['users']} users (stage {stopped['stage']}); "
                       "no further stages were run.")
    elif summary["user_stop"]:
        banners.append("Stopped from the dashboard between stages; later stages were not run.")
    if summary["aborted"]:
        banners.append("The test was aborted manually; results are incomplete.")
    if failed:
        banners.append(f"Locust failed to run at stage {failed['stage']} – a setup problem, not a capacity limit.")
    errors = next((r for r in ordered if not r["passed"] and r.get("top_errors")), None)

    gpu = getattr(args, "gpu_name", None)
    started = datetime.strptime(out_dir.name, "%Y%m%d_%H%M%S") if re.fullmatch(r"\d{8}_\d{6}", out_dir.name) else None
    if args.target == "lmstudio":
        model = passthrough_value(args.passthrough, "--model")
        context = passthrough_value(args.passthrough, "--context-tokens")
        details = [f"model {model}" if model else None,
                   f"≈ {context} context tokens per request" if context and context != "0" else None]
    else:
        details = [f"mode {passthrough_value(args.passthrough, '--mode') or 'query'}"]
    body = [f"<h1>{html.escape(report_title(args, out_dir))} – {html.escape(TARGETS[args.target]['label'])}</h1>",
            "<p class='meta'>" + " · ".join(html.escape(x) for x in filter(None, [
                started.strftime("%Y-%m-%d %H:%M") if started else None, f"run {out_dir.name}" if args.name else None,
                args.host, *details,
                f"{args.stage_duration} s per stage", gpu])) + "</p>"]
    body += [f"<div class='banner'>{html.escape(b)}</div>" for b in banners]
    if failed:
        body.append(f"<pre>{html.escape(failed['error'])}</pre>")

    def section(title, items, note=None):
        items = [i for i in items if i]
        if items:
            body.append(f"<h2>{html.escape(title)}</h2>" + (f"<p class='meta'>{html.escape(note)}</p>" if note else "")
                        + tiles(items))

    reference_note = (f"Statistics refer to the stage with the highest stable throughput ({ref['users']} users)."
                      if ref and ref is best else
                      f"Statistics refer to stage {ref['stage']} ({ref['users']} users)." if ref else None)
    section("Result", result_tiles)
    section("Requests", request_tiles)
    section("Response times", response_tiles, reference_note)
    section("Tokens", token_tiles, "Token figures come from the LM Studio log." if token_tiles else None)
    section("Hardware", hardware_tiles)
    if errors:
        body.append(f"<h2>Most frequent errors at {errors['users']} users</h2><ul>"
                    + "".join(f"<li>{n}× <code>{html.escape(e)}</code></li>" for e, n in errors["top_errors"]) + "</ul>")
        if looks_like_setup_problem(errors):
            body.append(f"<blockquote>{html.escape(SETUP_HINT)}</blockquote>")
    if charts:
        body.append("<h2>Charts by load</h2><div class='charts'>" + "".join(f"<figure>{svg}</figure>" for *_, svg in charts)
                    + "</div>")
    over_time = time_charts(ordered, ref)
    if over_time:
        body.append("<h2>Charts over time</h2><div class='charts'>" + "".join(f"<figure>{svg}</figure>" for svg in over_time)
                    + "</div>")
    # Detailed sections from the Markdown report (refusals, stage table, notes); the result summary is in the tiles.
    rest = next((i for i, line in enumerate(lines) if line.startswith("## ") and not line.startswith("## Result")), len(lines))
    body.append(markdown_to_html("\n".join(lines[rest:])))
    return html_page(report_title(args, out_dir), TILE_CSS + "\n".join(body))


def time_charts(ordered, ref):
    """Charts over the whole run: users and responses/s, response times, hardware; percentiles of the reference stage."""
    # Stages in run order; a gap point between stages keeps the pause from being drawn as a line.
    runs = sorted(ordered, key=lambda r: r["stage"])
    timeline, hardware = [], []
    for r in runs:
        if timeline and r.get("timeline"):
            timeline.append({"t": timeline[-1]["t"] + 1, "users": None, "rps": None, "median": None, "p95": None})
        timeline += r.get("timeline") or []
        if hardware and r.get("hardware_timeline"):
            hardware.append({"ts": hardware[-1]["ts"] + 1})
        hardware += r.get("hardware_timeline") or []
    charts = []
    if timeline:
        t = [p["t"] for p in timeline]
        charts.append(time_chart("Simulated users over time", t, [("users", [p["users"] for p in timeline])]))
        charts.append(time_chart("Chat responses per second over time", t, [("responses/s", [p["rps"] for p in timeline])],
                                 unit="/s"))
        charts.append(time_chart("Chat response time over time", t, [("median", [p["median"] for p in timeline]),
                                                                     ("p95", [p["p95"] for p in timeline])], unit=" s"))
    if hardware and any(p.get("gpu_util") is not None for p in hardware):
        charts.append(time_chart("GPU utilisation and memory over time", [p["ts"] for p in hardware],
                                 [("GPU utilisation", [p.get("gpu_util") for p in hardware]),
                                  ("GPU memory", [p.get("vram_pct") for p in hardware])], unit=" %", y_max=100))
    if ref and ref.get("chat_percentiles"):
        pct = ref["chat_percentiles"]
        charts.append(line_chart(f"Response time percentiles at {ref['users']} users", list(pct.keys()),
                                 [("response time", list(pct.values()))], unit=" s", x_title="share of requests faster than",
                                 standalone=False))
    return charts


def build_charts(ordered, args):
    """Charts over the stages: (file name, title, standalone SVG, SVG for inline use)."""
    if not ordered:
        return []
    labels = [r["users"] for r in ordered]
    col = lambda key, factor=1: [None if r.get(key) is None else r[key] * factor for r in ordered]  # noqa: E731
    specs = [
        ("response_time", "Response time per chat request", [("average", col("chat_avg_s")), ("p95", col("chat_p95_s"))],
         " s", None, (args.max_p95, "p95 limit")),
        ("throughput", "Throughput: chat responses per second", [("responses/s", col("chat_rps"))], "/s", None, None),
        ("tokens", "Generated tokens per second (LM Studio)", [("tokens/s", col("lm_gen_tokens_per_s"))], "", None, None),
        ("failures", "Failure and refusal rate", [("failures", col("fail_rate", 100)), ("refusals", col("refusal_rate", 100))],
         " %", None, (args.max_fail_rate * 100, "failure limit")),
        ("gpu", "GPU utilisation and memory", [("GPU utilisation (avg)", col("gpu_util_avg")),
                                               ("GPU memory (max)", col("vram_pct_max"))], " %", 100, None),
        ("system_memory", "System memory (max)", [("system memory", col("ram_pct_max"))], " %", 100, None),
    ]
    charts = []
    for name, title, series, unit, y_max, limit in specs:
        series = [(label, vals) for label, vals in series if any(v is not None for v in vals)]
        if not series:
            continue  # E.g. no GPU metrics or no LM Studio log on this machine.
        kwargs = dict(unit=unit, x_title="concurrent users", y_max=y_max, limit=limit)
        charts.append((name, title, line_chart(title, labels, series, **kwargs),
                       line_chart(title, labels, series, standalone=False, **kwargs)))
    return charts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="", help="Name of the load test, shown as the report title (optional)")
    ap.add_argument("--target", choices=sorted(TARGETS), default="anythingllm",
                    help="System under test: AnythingLLM (via its developer API) or LM Studio directly")
    ap.add_argument("--host", help="Base URL (default: http://localhost:3001 for AnythingLLM, "
                                   "http://localhost:1234 for LM Studio)")
    ap.add_argument("--locustfile", type=Path, help="Locust user file (default: main.py or lmstudio_user.py)")
    ap.add_argument("--users", help="Explicit stages, e.g. 1,2,4,8,16,32 (overrides start/step/max)")
    ap.add_argument("--start-users", type=int, default=1)
    ap.add_argument("--step", type=int, default=2, help="Additive step size")
    ap.add_argument("--step-factor", type=float, default=1.0, help=">1 = multiplicative steps (e.g. 2 -> 1,2,4,8...)")
    ap.add_argument("--max-users", type=int, default=64)
    ap.add_argument("--spawn-rate", type=float, default=1.0, help="New users per second")
    ap.add_argument("--stage-duration", type=int, default=300, help="Measurement time per stage in seconds (after ramp-up)")
    ap.add_argument("--cooldown", type=int, default=30, help="Pause between stages in seconds")
    ap.add_argument("--stop-timeout", type=int, default=120,
                    help="Grace period for in-flight requests at the end of a stage, in seconds")
    ap.add_argument("--max-fail-rate", type=float, default=0.05)
    ap.add_argument("--max-p95", type=float, default=120.0, help="Maximum p95 chat response time in seconds")
    ap.add_argument("--max-vram-pct", type=float, default=98.0)
    ap.add_argument("--max-gpu-temp", type=float, default=88.0)
    ap.add_argument("--refusal-warn-rate", type=float, default=0.10,
                    help="Refusal rate from which refusals are highlighted in red (they never fail a stage)")
    ap.add_argument("--refine", action="store_true",
                    help="Bisect between the last passing and the first failing stage to find the exact limit")
    ap.add_argument("--student-think-time", type=float, default=90.0,
                    help="Assumed real time between two questions of one student (used for the estimate)")
    ap.add_argument("--lmstudio-log", default="auto", help="'auto', a path, or 'off'")
    ap.add_argument("--out", default="loadtest_results")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Only print the planned stages and duration")
    args, passthrough = ap.parse_known_args()
    sys.stdout.reconfigure(line_buffering=True)  # Flush every line, also when redirected to a file.
    args.passthrough = [p for p in passthrough if p != "--"]
    args.name = " ".join(args.name.split())[:120]  # One line; the result folder stays the unique identifier.
    target = TARGETS[args.target]
    args.host = (args.host or target["host"]).rstrip("/")
    args.locustfile = (args.locustfile or BASE_DIR / target["locustfile"]).resolve()
    global SETUP_HINT
    SETUP_HINT = SETUP_HINTS[args.target]

    levels = build_levels(args)
    est = sum(math.ceil(u / args.spawn_rate) + args.stage_duration + args.cooldown for u in levels)
    print(f"Planned stages: {levels}  (up to ~{est / 60:.0f} min, excluding --refine)")
    if args.dry_run:
        return
    check_locust()
    if not args.skip_preflight:
        if args.target == "lmstudio":
            preflight_lmstudio(args.host, passthrough_value(args.passthrough, "--model"))
        else:
            preflight(args.host)

    out_dir = (BASE_DIR / args.out if not Path(args.out).is_absolute() else Path(args.out)) \
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    config = {k: str(v) for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results: {out_dir}")

    sampler = SystemSampler()
    sampler.start()
    args.gpu_name = sampler.gpu.info().get("gpu_name")
    global STATUS
    if stop_requested():
        STOP_FILE.unlink()  # A stop request from an earlier run must not end this one.
    STATUS = StatusPublisher(out_dir.name, args.target, args.name)
    STATUS.start()
    print(f"Target: {target['label']} at {args.host}")
    print("Stop at any time with the \"Stop & delete threads\" button in the live monitor's dashboard.")
    if not sampler.gpu_available:
        print("Note: no supported GPU metrics source found, GPU limits will not be checked.")
    elif sampler.gpu.backend == "apple":
        print("Note: GPU temperature is not available on macOS without sudo, the temperature limit is skipped.")
    watcher = None
    if args.lmstudio_log != "off":
        target = find_log_target(None if args.lmstudio_log == "auto" else args.lmstudio_log)
        if target and target.exists():
            watcher = LMStudioWatcher(target).start()
        else:
            print("Note: LM Studio log not found, no tokens/s metrics (use --lmstudio-log PATH).")

    results, last_ok, first_bad, aborted, user_stop = [], None, None, False, False
    try:
        for users in levels:
            r = run_stage(len(results) + 1, users, args, args.passthrough, out_dir, sampler, watcher)
            results.append(r)
            write_results(out_dir, results, args, last_ok, first_bad, False)
            if r["error"] is not None:
                break  # Setup problem: further stages would fail the same way.
            if r["stopped"]:
                user_stop = True
                break  # Stopped from the dashboard: no further stages, no refinement.
            if not r["passed"]:
                first_bad = users
                break
            last_ok = users
            if pause(args.cooldown):
                user_stop = True
                break

        if args.refine and first_bad is not None and not user_stop:
            lo, hi = (last_ok or 0), first_bad
            while hi - lo > 1:
                if pause(args.cooldown):
                    user_stop = True
                    break
                mid = (lo + hi) // 2
                r = run_stage(len(results) + 1, mid, args, args.passthrough, out_dir, sampler, watcher)
                results.append(r)
                if r["stopped"]:
                    user_stop = True
                    break
                lo, hi = (mid, hi) if r["passed"] else (lo, mid)
                write_results(out_dir, results, args, lo or None, hi, False)
            last_ok, first_bad = (lo or None), hi
    except KeyboardInterrupt:
        aborted = True
        print("\nAborted, writing partial results ...")
    finally:
        STATUS.close()
        sampler.stop()
        if watcher:
            watcher.stop()
        write_results(out_dir, results, args, last_ok, first_bad, aborted, user_stop)
        print(f"\nReport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
