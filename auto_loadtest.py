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


def read_locust_stats(path):
    """Return the 'Aggregated' row and the chat endpoint row from a Locust *_stats.csv file."""
    if not path.exists():
        return None, None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    agg = next((r for r in rows if r.get("Name") == "Aggregated"), None)
    chat = next((r for r in rows if r.get("Name", "").startswith("/thread/chat")), None)
    return agg, chat


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


def run_stage(idx, users, args, passthrough, out_dir, sampler, watcher):
    """Run one Locust stage with ``users`` concurrent users and evaluate it against the limits."""
    prefix = out_dir / f"stage_{idx:02d}_u{users:03d}"
    ramp = math.ceil(users / args.spawn_rate)
    cmd = [sys.executable, "-m", "locust", "-f", str(args.locustfile), "--headless",
           "--host", args.host, "-u", str(users), "-r", str(args.spawn_rate),
           "-t", f"{ramp + args.stage_duration}s", "--reset-stats",
           "--stop-timeout", str(args.stop_timeout), "--only-summary",
           "--csv", str(prefix), "--html", f"{prefix}.html", *passthrough]
    print(f"\n=== Stage {idx}: {users} users  (ramp-up {ramp}s + {args.stage_duration}s measurement) ===")

    sampler.summary(reset=True)
    if watcher:
        watcher.snapshot(reset=True)
    t0 = time.time()
    with open(f"{prefix}.log", "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=args.locustfile.parent, stdout=logf, stderr=subprocess.STDOUT)
    sysm = sampler.summary(reset=True)
    lm = watcher.snapshot(reset=True) if watcher else {}

    agg, chat = read_locust_stats(Path(f"{prefix}_stats.csv"))
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
        "chat_rps": round(num(chat["Requests/s"]) or 0, 4) if chat else 0,
        "chat_median_s": seconds("Median Response Time"), "chat_avg_s": seconds("Average Response Time"),
        "chat_p95_s": seconds("95%"), "chat_p99_s": seconds("99%"), "chat_max_s": seconds("Max Response Time"),
        **sysm,
        "lm_gen_tokens_per_s": lm.get("gen_tokens_per_s_total"),
        "lm_tokens_per_s_per_request": lm.get("tokens_per_s_per_request_avg"),
        "lm_ttft_p95_s": lm.get("ttft_p95_s"), "lm_max_active": lm.get("max_active"),
        "lm_completion_tokens": lm.get("completion_tokens"),
    }

    reasons = []
    if agg is None:
        reasons.append("Locust produced no statistics (see .log)")
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
    print("  -> " + ("PASSED" if result["passed"] else "LIMIT REACHED: " + result["reasons"]))
    return result


def write_results(out_dir, results, args, last_ok, first_bad, aborted):
    """Write results.csv, results.json and the Markdown report."""
    if results:
        fields = list(results[0].keys())
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
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
    if last_ok is None and first_bad is not None:
        lines.append(f"- A limit was already exceeded at **{first_bad} Locust users**.")
    elif last_ok is not None:
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
    lines += ["", "## Stages", "",
              "| Users | Requests | Failures % | Chat avg s | Chat p95 s | Chat req/s | GPU avg % | VRAM max % | RAM max % | tok/s | Status |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]

    def cell(v):
        return "-" if v is None else v

    for r in ordered:
        fail = "-" if r["fail_rate"] is None else f"{r['fail_rate'] * 100:.1f}"
        lines.append(f"| {r['users']} | {r['requests']} | {fail} | {cell(r['chat_avg_s'])} | {cell(r['chat_p95_s'])} | "
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
    if not args.skip_preflight:
        preflight(args.host)

    out_dir = (BASE_DIR / args.out if not Path(args.out).is_absolute() else Path(args.out)) \
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    config = {k: str(v) for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results: {out_dir}")

    sampler = SystemSampler()
    sampler.start()
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
