#!/usr/bin/env python3
"""
Automated capacity test for AnythingLLM backed by LM Studio.

Runs main.py (Locust) headless several times with an increasing number of
concurrent users. During each stage it samples GPU/host metrics and, if the
LM Studio log is available, token throughput. As soon as a stage violates a
limit (failure rate, p95 latency, VRAM, GPU temperature) the run stops; with
--refine the exact limit is then located by bisection. Results are written as
CSV/JSON plus a Markdown report to the results directory.

Any argument not recognised by this script is forwarded to main.py
(e.g. --mode chat --wait-min 2 --wait-max 5).

Example:
    python auto_loadtest.py --users 1,2,4,8,16,32,48,64 --stage-duration 300 --refine -- --mode query
"""
import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from lmstudio_monitor import LMStudioWatcher, find_log_target
from system_metrics import SystemSampler

BASE_DIR = Path(__file__).resolve().parent


def num(value):
    """Convert a Locust CSV cell to float, returning None for 'N/A' or empty cells."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


REFUSAL_TYPES = ("REFUSAL", "REJECTED")  # Entry types reported by main.py next to the HTTP requests.


def read_locust_stats(path):
    """
    Parse a Locust *_stats.csv file. Returns (totals, chat row, refusal counts), where totals
    sums the HTTP requests only (refusal entries are excluded) and refusal counts maps
    e.g. "REJECTED http-503" or "REFUSAL no-context" to the number of occurrences.
    """
    if not path.exists():
        return None, None, {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    http_rows = [r for r in rows if r.get("Name") != "Aggregated" and r.get("Type") not in REFUSAL_TYPES]
    totals = {"Request Count": sum(int(num(r["Request Count"]) or 0) for r in http_rows),
              "Failure Count": sum(int(num(r["Failure Count"]) or 0) for r in http_rows)}
    chat = next((r for r in http_rows if r.get("Name", "").startswith("/thread/chat")), None)
    refusals = {f"{r['Type']} {r['Name']}": int(num(r["Request Count"]) or 0)
                for r in rows if r.get("Type") in REFUSAL_TYPES}
    return totals, chat, refusals


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


SETUP_HINT = ("Nearly all requests failed at the lowest load. This usually indicates a configuration problem, "
              "not a capacity limit: check the LLM provider in AnythingLLM. When AnythingLLM runs in Docker, "
              "LM Studio must be configured as http://host.docker.internal:1234/v1 and "
              "'Serve on Local Network' must be enabled in LM Studio.")


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


class EmergencyWatchdog:
    """
    Checks the hardware once per second while a stage runs. It trips when a reading stays at or
    above its emergency limit for ``--stop-after`` seconds, so the stage can be stopped at once
    instead of being evaluated only when it ends.
    """

    def __init__(self, sampler, args):
        self.sampler = sampler
        self.hold = args.stop_after
        self.limits = [(key, label, unit, limit) for key, label, unit, limit in (
            ("vram_pct", "VRAM", "%", args.stop_vram_pct),
            ("ram_pct", "System memory", "%", args.stop_ram_pct),
            ("gpu_temp", "GPU temperature", "°C", args.stop_gpu_temp)) if limit > 0]
        self.above_since = {}

    def describe(self):
        return ", ".join(f"{label} ≥ {limit:g}{unit}" for _, label, unit, limit in self.limits) or "disabled"

    def check(self, now):
        """Return the reason for an emergency stop, or None."""
        latest = self.sampler.latest
        for key, label, unit, limit in self.limits:
            value = latest.get(key)
            if value is None or value < limit:
                self.above_since.pop(key, None)
                continue
            since = self.above_since.setdefault(key, now)
            if now - since >= self.hold:
                return f"{label} {value:g}{unit} ≥ {limit:g}{unit} for {self.hold:g} s"
        return None


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


def run_stage(idx, users, args, passthrough, out_dir, sampler, watcher):
    """Run one Locust stage with ``users`` concurrent users and evaluate it against the limits."""
    prefix = out_dir / f"stage_{idx:02d}_u{users:03d}"
    ramp = math.ceil(users / args.spawn_rate)
    cmd = [sys.executable, "-m", "locust", "-f", str(args.locustfile), "--headless",
           "--host", args.host, "-u", str(users), "-r", str(args.spawn_rate),
           "-t", f"{ramp + args.stage_duration}s", "--reset-stats",
           "--stop-timeout", str(args.stop_timeout), "--only-summary",
           "--csv", str(prefix), "--html", f"{prefix}.html",
           "--refusal-log", f"{prefix}_refusals.jsonl",
           "--run-id", f"{out_dir.name}-s{idx:02d}",  # Thread names identify run and stage.
           *passthrough]
    print(f"\n=== Stage {idx}: {users} users  (ramp-up {ramp}s + {args.stage_duration}s measurement) ===")

    sampler.summary(reset=True)
    if watcher:
        watcher.snapshot(reset=True)
    t0 = time.time()
    dog, emergency = EmergencyWatchdog(sampler, args), None
    with open(f"{prefix}.log", "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=args.locustfile.parent, stdout=logf, stderr=subprocess.STDOUT,
                                **popen_flags())
        try:
            while proc.poll() is None:
                emergency = dog.check(time.time())
                if emergency:
                    print(color(f"  !!! EMERGENCY STOP: {emergency} – stopping Locust now", "1;37;41"))
                    stop_locust(proc)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            stop_locust(proc)
            raise
    sysm = sampler.summary(reset=True)
    lm = watcher.snapshot(reset=True) if watcher else {}

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
        "lm_completion_tokens": lm.get("completion_tokens"),
    }

    result["refusal_rate"] = (round((result["rejected"] + result["refused_no_context"]) / result["chat_requests"], 4)
                              if result["chat_requests"] else None)
    result["refusal_breakdown"] = refusals
    result["refusal_examples"] = read_refusal_examples(f"{prefix}_refusals.jsonl")
    result["top_errors"] = read_failures(Path(f"{prefix}_failures.csv"))
    result["emergency"] = emergency
    # Without statistics Locust did not run at all: a setup problem, not a capacity limit.
    result["error"] = None if agg is not None or emergency else "\n".join(log_tail(f"{prefix}.log"))

    reasons = [f"EMERGENCY STOP: {emergency}"] if emergency else []
    if agg is None and not emergency:
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
    print("  -> " + ("PASSED" if result["passed"] else "LIMIT REACHED: " + result["reasons"]))
    for error, count in result["top_errors"][:3]:
        print(f"     {count}x {error}")
    if looks_like_setup_problem(result):
        print(f"     Note: {SETUP_HINT}")
    return result


def write_results(out_dir, results, args, last_ok, first_bad, aborted):
    """Write results.csv, results.json and the Markdown report."""
    if results:
        fields = [k for k in results[0] if k not in ("top_errors", "error", "refusal_breakdown", "refusal_examples")]
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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

    lines = [f"# Load test report {out_dir.name}", "",
             f"- Host: `{args.host}`, stage duration {args.stage_duration}s, forwarded to main.py: `{' '.join(args.passthrough) or '-'}`",
             f"- Limits: failure rate ≤ {args.max_fail_rate:.0%}, chat p95 ≤ {args.max_p95}s, "
             f"VRAM ≤ {args.max_vram_pct}%, GPU ≤ {args.max_gpu_temp}°C", ""]
    if aborted:
        lines += ["**The test was aborted manually; results are incomplete.**", ""]
    lines += ["## Result", ""]
    stopped = next((r for r in results if r.get("emergency")), None)
    if stopped is not None:
        lines += [f"### ⛔ Emergency stop at {stopped['users']} users (stage {stopped['stage']})", "",
                  f"**{stopped['emergency']}.** Locust was stopped immediately; no further stages were run. "
                  "The highest stage that passed before is the safe operating limit.", ""]
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
                  "Technical rejections also count as failures; \"no relevant documents\" refusals count as answered.", "",
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
            lines += ["", "Examples (prompt from prompts.json → start of the response):", ""]
            for users, ex in examples:
                response = " ".join(ex["response"].split()).replace("|", "\\|") or "(empty)"
                thread = f", thread `{ex['thread']}`" if ex.get("thread") else ""
                lines.append(f"- `{ex['category']}` at {users} users{thread}: „{ex['prompt']}“ → {response}")

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
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://localhost:3001", help="AnythingLLM base URL")
    ap.add_argument("--locustfile", type=Path, default=BASE_DIR / "main.py")
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
    ap.add_argument("--stop-vram-pct", type=float, default=99.0,
                    help="Emergency stop when VRAM stays at or above this value during a stage (0 = off)")
    ap.add_argument("--stop-ram-pct", type=float, default=99.0,
                    help="Emergency stop when system memory stays at or above this value (0 = off)")
    ap.add_argument("--stop-gpu-temp", type=float, default=90.0,
                    help="Emergency stop when the GPU temperature stays at or above this value (0 = off)")
    ap.add_argument("--stop-after", type=float, default=10.0,
                    help="Seconds a reading must stay above its emergency limit before the stop")
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
    args.locustfile = args.locustfile.resolve()

    levels = build_levels(args)
    est = sum(math.ceil(u / args.spawn_rate) + args.stage_duration + args.cooldown for u in levels)
    print(f"Planned stages: {levels}  (up to ~{est / 60:.0f} min, excluding --refine)")
    if args.dry_run:
        return
    check_locust()
    if not args.skip_preflight:
        preflight(args.host)

    out_dir = (BASE_DIR / args.out if not Path(args.out).is_absolute() else Path(args.out)) \
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    config = {k: str(v) for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results: {out_dir}")

    sampler = SystemSampler(interval=1.0)
    sampler.start()
    print(f"Emergency stop: {EmergencyWatchdog(sampler, args).describe()} for {args.stop_after:g} s")
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

    results, last_ok, first_bad, aborted = [], None, None, False
    try:
        for users in levels:
            r = run_stage(len(results) + 1, users, args, args.passthrough, out_dir, sampler, watcher)
            results.append(r)
            write_results(out_dir, results, args, last_ok, first_bad, False)
            if r["error"] is not None:
                break  # Setup problem: further stages would fail the same way.
            if r["emergency"]:
                break  # Hardware at its limit: no further stages, no refinement.
            if not r["passed"]:
                first_bad = users
                break
            last_ok = users
            time.sleep(args.cooldown)

        if args.refine and first_bad is not None:
            lo, hi = (last_ok or 0), first_bad
            while hi - lo > 1:
                time.sleep(args.cooldown)
                mid = (lo + hi) // 2
                r = run_stage(len(results) + 1, mid, args, args.passthrough, out_dir, sampler, watcher)
                results.append(r)
                lo, hi = (mid, hi) if r["passed"] else (lo, mid)
                write_results(out_dir, results, args, lo or None, hi, False)
            last_ok, first_bad = (lo or None), hi
    except KeyboardInterrupt:
        aborted = True
        print("\nAborted, writing partial results ...")
    finally:
        sampler.stop()
        if watcher:
            watcher.stop()
        write_results(out_dir, results, args, last_ok, first_bad, aborted)
        print(f"\nReport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
