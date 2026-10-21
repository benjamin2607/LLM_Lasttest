# Load Testing & Monitoring – AnythingLLM + LM Studio

Tools to determine the capacity limits of a self-hosted AnythingLLM + LM Studio
setup and to monitor its load in production. Everything runs locally; no data
leaves the machine.

| File | Purpose |
|---|---|
| `main.py` | Locust user behaviour against the AnythingLLM developer API (see `start_command.md`) |
| `auto_loadtest.py` | Automated capacity test: increases concurrent users stage by stage until a limit is exceeded |
| `lmstudio_monitor.py` | Live monitor for production use (LM Studio log + API + GPU) with a real-time dashboard |
| `lmstudio_log_parser.py` | Turns LM Studio log entries into request metrics (used by the monitor and the load test) |
| `dashboard.html` | Dashboard front end, served by the monitor (no external libraries) |
| `system_metrics.py` | Shared GPU/host metrics sampler (NVIDIA via nvidia-smi, macOS via ioreg, psutil) |
| `prompts.json` | Questions sent by the simulated users |
| `tests/` | Unit tests for the log parser |

No additional packages are required beyond `locust` (which ships `psutil`) and
the Python standard library.

## Prerequisites

1. **Python virtual environment with Locust**: `pip install locust`
2. **AnythingLLM API key** (AnythingLLM → Settings → Developer API) and the workspace slug as environment variables:
   - Linux/macOS: `export ANYTHING_API_KEY=...` and `export WORKSPACE_SLUG=...`
   - Windows PowerShell: `$env:ANYTHING_API_KEY="..."` and `$env:WORKSPACE_SLUG="..."`
3. **GPU metrics** are detected automatically (no configuration needed):
   - **NVIDIA (Linux/Windows):** `nvidia-smi` on the `PATH` (installed with the driver; no administrator rights needed).
   - **macOS (Apple Silicon and Intel Macs):** read via `ioreg`, no administrator rights needed. See [macOS notes](#macos-notes).
4. **Read access to the LM Studio log directory** `~/.lmstudio/server-logs`
   (Windows: `C:\Users\<name>\.lmstudio\server-logs`). The simplest setup is to run
   the scripts under the same user account as LM Studio; otherwise pass `--log PATH`.
5. **LM Studio** server running (default port 1234). If "Require API token" is enabled,
   pass the token via `--token` or `LMSTUDIO_API_TOKEN`.
6. **Recommended: enable Verbose Logging** in LM Studio (Developer → Logging → Verbose Logging). See [What the monitor can see](#what-the-monitor-can-see).

## 1. Automated capacity test

Preview the planned stages and the expected duration:

```bash
python auto_loadtest.py --dry-run --users 1,2,4,8,16,24,32,48,64
```

Run the test:

```bash
python auto_loadtest.py --users 1,2,4,8,16,24,32,48,64 --stage-duration 300 --refine -- --mode query --wait-min 5 --wait-max 15
```

Each stage starts Locust headless, ramps up, measures for `--stage-duration`
seconds while sampling GPU/host/token metrics, and evaluates the result. The
first stage that violates a limit ends the run; `--refine` then bisects to find
the exact limit.

Limits (configurable): `--max-fail-rate 0.05`, `--max-p95 120` (seconds, chat
request), `--max-vram-pct 98`, `--max-gpu-temp 88`. Everything after `--` is
forwarded to `main.py` (e.g. `--mode chat`).

Results are written to `loadtest_results/<timestamp>/`: `report.md` (summary and
recommendation), `results.csv`/`results.json`, and a Locust HTML report plus log
per stage. The report also identifies throughput saturation and estimates how
many *real* students the system can serve, using Little's law with
`--student-think-time` (default: 90 s between two questions).

Tip: run once with `--mode query` and once with `--mode chat`; for overnight runs, choose a generous `--users` list.

## 2. Live monitoring in production

```bash
python lmstudio_monitor.py
```

Opens the dashboard at `http://127.0.0.1:8765` (reachable from the local machine
only). It updates **every second** via a live stream and shows:

- **Status bar:** connection, LM Studio online/offline, whether the log is being written, loaded models, and an alert banner when thresholds are exceeded
- **Key metrics:** active requests (and how many are waiting), responses per minute, tokens per second, p95 response time and time to first token (last 5 minutes), totals since start
- **Hardware:** GPU utilisation, VRAM (GB and %), temperature, power draw, system memory, CPU – with alert thresholds
- **Live charts** (5/15/30 minutes, tooltips on hover): concurrent requests, tokens per second, GPU/VRAM, response time of every individual request
- **Running requests** as bars that grow in real time (phase: queued/processing prompt or generating)
- **Recently completed requests** (duration, first token, tokens in/out, tokens per second, model – no content)
- **Daily view** of completed requests per interval

Options: `--dashboard-port 0` (no dashboard), `--no-browser`, `--interval 60`
(CSV and daily view interval), alert thresholds `--alert-vram-pct 95`,
`--alert-gpu-temp 85`, `--alert-p95 60`, `--alert-active N`. In addition, the
monitor writes one line per interval to stdout and to
`monitoring/lmstudio_metrics_YYYY-MM-DD.csv`.

- `--probe-interval 300` sends a fixed synthetic request every 5 minutes and measures user-perceived response time (it counts as one request in the log).
- `--from-start` parses an existing log file from the beginning (useful for CSV output; the live view is meaningful for new lines only).

### What the monitor can see

The monitor reads the LM Studio server log. What LM Studio writes there depends on
where a request comes from and on the Verbose Logging setting:

| Request source | Verbose Logging off (default) | Verbose Logging on |
|---|---|---|
| HTTP API (AnythingLLM, Locust, scripts) | ✓ tracked | ✓ tracked, plus engine timings |
| Chat window of the LM Studio app | – not logged by LM Studio | ✓ tracked via llama.cpp engine entries |

A request that appears in several sources (an API request served by the llama.cpp
engine) is merged into one record, so nothing is counted twice. The dashboard
shows a notice when Verbose Logging is off and marks each completed request as
"API" or "LM Studio app".

Accuracy:

- **Streaming API responses** carry a completion ID in every chunk and are tracked exactly:
  duration, time to first token and live tokens/s. Completion tokens are counted per chunk
  (≈ one token each, marked "≈" in the dashboard) unless LM Studio logs a usage block.
- **Non-streaming API responses** report exact token counts; with several in flight at once,
  they are matched in arrival order, so individual latencies are approximations.
- **llama.cpp engine tasks** carry a task ID and an engine uptime clock, so durations are exact.
  Completion tokens come from the engine's final timing entry or are extrapolated from its
  progress reports (≈). Several loaded models are told apart by their engine process.
- A request that LM Studio announces but never starts (for example, rejected while a model
  loads on demand) is dropped after 120 s.

### Troubleshooting

If the dashboard shows no requests while the GPU is busy, run the diagnosis. It lists only
which entry types were recognised – no content – and gives hints:

```bash
python lmstudio_monitor.py --diagnose
```

If a newer LM Studio release changes the log format, share that report; the patterns are
defined at the top of `lmstudio_log_parser.py`.

### Tests

```bash
python -m unittest discover -s tests
```

## macOS notes

The monitor and the load test run unchanged on a MacBook; the platform is detected at start-up.

| Metric | NVIDIA | macOS |
|---|---|---|
| GPU utilisation | ✓ | ✓ |
| GPU memory | VRAM used / total | GPU memory allocated / GPU memory limit (unified memory) |
| GPU temperature | ✓ | empty (requires `sudo powermetrics`) |
| GPU power draw | ✓ | empty (requires `sudo powermetrics`) |
| System memory, CPU | ✓ | ✓ |

On Macs the GPU shares system memory. The limit is taken from `sysctl iogpu.wired_limit_mb` if set,
otherwise the macOS default (about two thirds of RAM up to 36 GB, three quarters above). The dashboard
labels the metric "GPU memory" instead of "VRAM". In the load test, the GPU temperature limit is skipped on macOS.

## Privacy

- All scripts run locally and send no data to third parties. The dashboard binds to `127.0.0.1` only and writes no access logs.
- Only **aggregated numeric metrics** are stored (counts, durations, tokens, model name, GPU values).
  Prompt and response text, IP addresses, user names and chat IDs are neither stored nor displayed.
- **Important:** LM Studio itself writes full prompts and responses to its server log, and AnythingLLM stores chat histories in its database.
  This is outside the scope of these scripts but should be covered by the data protection concept
  (define a retention period for `~/.lmstudio/server-logs`, enable verbose logging in production only when needed,
  and limit chat history retention in AnythingLLM where appropriate).
