# LLM Load Test & Monitoring – AnythingLLM + LM Studio

Tools to determine the capacity limits of a self-hosted AnythingLLM + LM Studio
setup and to monitor its load in production and in live tests with students.
Everything runs locally; no data leaves the machine.

**Quick start**

1. `pip install locust` (in a virtual environment)
2. `python lmstudio_monitor.py` – opens the live dashboard at `http://127.0.0.1:8765`
3. Set `ANYTHING_API_KEY` and `WORKSPACE_SLUG`, then `python auto_loadtest.py --users 1,2,4,8,16` – or start a load test from the dashboard

Step-by-step guide for the first run: **[QUICKSTART.md](QUICKSTART.md)**. All options with descriptions and guideline values: **[PARAMETERS.md](PARAMETERS.md)**.

| File | Purpose |
|---|---|
| `main.py` | Locust user behaviour against the AnythingLLM developer API (options for manual Locust runs: `PARAMETERS.md`) |
| `auto_loadtest.py` | Automated capacity test: increases concurrent users stage by stage until a limit is exceeded |
| `lmstudio_monitor.py` | Live monitor for production use (LM Studio log + API + GPU) with a real-time dashboard |
| `lmstudio_log_parser.py` | Turns LM Studio log entries into request metrics (used by the monitor and the load test) |
| `dashboard.html` | Dashboard front end, served by the monitor (no external libraries) |
| `report_charts.py` | Self-contained SVG charts and HTML rendering for the reports |
| `live_session.py` | Recording sessions for live tests: per-request data, per-minute values, hardware and report |
| `system_metrics.py` | Shared GPU/host metrics sampler (NVIDIA via nvidia-smi, macOS via ioreg, psutil) |
| `prompts.json` | Questions sent by the simulated users |
| `tests/` | Unit tests for the log parser and the live sessions |
| `QUICKSTART.md` | Step-by-step guide: install, monitor, first load test, live session |
| `PARAMETERS.md` | Reference of all options and environment variables with guideline values |

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

**Stopping a run.** The stage limits above are evaluated when a stage ends. To stop a run at any
time, use the **"Stop & delete threads"** button at the bottom of the live monitor's dashboard: it
stops Locust immediately (also during the pause between stages), runs no further stages, and
then deletes all threads the load test created in AnythingLLM. The report notes the stop.
The load test publishes its state in `.loadtest_status.json` and watches for `.loadtest_stop`;
the monitor needs `ANYTHING_API_KEY` (and `--anythingllm-url` if AnythingLLM is not on
`http://localhost:3001`) to delete the threads. Without a running load test, the button only
deletes the load-test threads.

Results are written to `loadtest_results/<timestamp>/`:

| File | Content |
|---|---|
| `report.html` | The report, in the same layout as the live session reports. Sections: **Result** (stable limit, throughput, estimated number of students, models), **Requests** (answered, failed, refusals), **Response times**, **Tokens** and **Hardware** – each as tiles with average, median, p95 and maximum at the stage with the highest stable throughput – then the most frequent errors, **charts by load** (response time, throughput, tokens/s, failures and refusals, GPU and memory per number of users), **charts over time** (every request and the hardware over the whole run), refusals and the stage table |
| `report.md` | The same as Markdown, charts as SVG files in `charts/` |
| `results.csv` / `results.json` | All figures per stage |
| `stage_XX_*` | Locust HTML report, CSV statistics and log per stage |

The report also identifies throughput saturation and estimates how many *real* students the
system can serve, using Little's law with `--student-think-time` (default: 90 s between two questions).

Tip: run once with `--mode query` and once with `--mode chat`; for overnight runs, choose a generous `--users` list.

### Threads

Every question is asked in its own AnythingLLM thread (`--messages-per-thread 1`), so
requests share no chat history and run in parallel. Threads are named traceably as
`Loadtest <run>-s<stage> u<user> #<seq>`, e.g. `Loadtest 20261021-160853-s02 u003 #0007`
(stage 2, simulated user 3, seventh question). The thread identifier also appears in the
refusal examples of the report.

Threads are kept so the chats can be inspected in AnythingLLM, but at most 30 per workspace
(`-- --max-threads N`, `0` = keep all). Beyond the limit, the oldest finished load-test threads
are deleted first (FIFO). The load test records the threads it created in
`.loadtest_threads.json`, so the limit holds across stages and runs; threads it did not
create are never deleted, and neither are threads that are still waiting for an answer.

### Refusals

Refusals are counted separately and highlighted in the console (yellow, red from
`--refusal-warn-rate`, default 10 %) and in a dedicated "⚠ Refusals" section of the report
(per stage, trend under load, examples with prompt, thread and start of the response):

| Category | Detection | Counts as |
|---|---|---|
| `REJECTED http-<status>` / `timeout` / `connection-error` | HTTP status ≠ 200 or no response | failure |
| `REJECTED error-response` | `error` field in the AnythingLLM response | failure |
| `REFUSAL no-context` | Query mode: AnythingLLM's refusal response with no sources (no relevant documents found) | answered |

If the workspace uses a custom query refusal response, pass it with `-- --refusal-text "..."`.
Refusals never fail a stage on their own.

## 2. Live monitoring in production

```bash
python lmstudio_monitor.py
```

Opens the dashboard at `http://127.0.0.1:8765` (reachable from the local machine
only). It updates **every second** via a live stream and shows:

- **Status bar:** a compact row of status LEDs (connection, LM Studio, log, model, GPU) that expands into the details on click, the load test status, and **Clear**, which resets all values shown and starts afresh (stored CSV data is kept; a restart does not restore cleared values). Warnings appear only in the tile they belong to
- **Load test chip** at the top: shows stage, users and elapsed time; click it to see the settings of the running (or last) load test; technical details are under "Advanced"
- **Load test card** at the bottom: status, link to the latest report, "Stop & delete threads", a table of all previous load tests (settings, result, throughput, p95, status, report, and a trash button that deletes the run's folder from disk after confirmation), and **New load test** – a form with all main parameters (stages, durations, think time, mode, limits, threads kept) and their defaults that starts `auto_loadtest.py`. The monitor needs `ANYTHING_API_KEY` and `WORKSPACE_SLUG` (or enter them in the form; the key is passed to the process only and never stored)
- **Sessions card** at the bottom: **Report for a past time range** (see below) and the list of recorded sessions with links to their reports and a trash button (deletes the session folder from disk after confirmation; the recording session cannot be deleted)
- Long ranges (30 min, Max) are drawn with at most 600 points per chart, using fixed time buckets aligned to the clock (maximum per bucket), so the drawn data stays stable while the window moves
- **Key metrics:** active requests (and how many are waiting), responses per minute, tokens per second, p95 response time and time to first token (last 5 minutes), totals since start. Below a divider, each tile also shows the average over the selected time range (5/15/30 min); the response time and first token tiles add the p95 for that range
- **While busy vs. overall:** averages over time (active requests, responses per minute, generated tokens/s, GPU utilisation) are shown "while busy" – only the seconds with at least one request in flight, including waiting for the first token – and "overall", which includes idle time. The busy share tells how much of the range the system was working. Use "while busy" to judge performance and "overall" for usage and planning. Figures per request are never affected by idle time
- **Per request:** response time, time to first token, tokens in, tokens out and tokens/s per request – each with average, median, p95 and maximum for the selected time range. "Tokens in" is everything the model processes: the workspace system prompt, RAG excerpts from documents, the thread's chat history and the question; "tokens out" includes reasoning tokens of thinking models
- **Statistics:** the tokens-per-second and response-time charts show mean (Ø), median and standard deviation (σ) for the selected range next to the current value
- **Hardware:** GPU utilisation, VRAM (GB and %), temperature, power draw, system memory, CPU – with alert thresholds
- **Live charts** (5/15/30 minutes or **Max** – since the running session started, otherwise all kept data, up to `--keep-hours`, default 4 h; tooltips on hover): concurrent requests, tokens per second, GPU/VRAM, response time of every individual request
- **Running requests** as bars that grow in real time (phase: queued/processing prompt or generating).
  A request without any log activity for 5 minutes (`--stall-minutes`) is marked **stalled** in red.
  Its "⋯" menu removes it from the monitor, and "Remove N stalled" removes all of them; each removal is
  printed to the console. This only affects the monitor: LM Studio offers no way to cancel another
  client's request, so a generation that is really still running continues there. Stalled entries are
  typically requests whose completion LM Studio never logged (e.g. the client disconnected); they no
  longer capture newer streams and are dropped automatically after 15 minutes without activity.
- **Recently completed requests** (duration, first token, tokens in/out, tokens per second, model – no content)
- **Daily view** of completed requests per interval

Options: `--dashboard-port 0` (no dashboard), `--no-browser`, `--interval 60`
(CSV and daily view interval), alert thresholds `--alert-vram-pct 95`,
`--alert-gpu-temp 85`, `--alert-p95 60`, `--alert-active N`. In addition, the
monitor writes one line per interval to stdout and to
`monitoring/lmstudio_metrics_YYYY-MM-DD.csv`.

- `--probe-interval 300` sends a fixed synthetic request every 5 minutes and measures user-perceived response time (it counts as one request in the log).
- `--from-start` parses an existing log file from the beginning (useful for CSV output; the live view is meaningful for new lines only).

**After a restart** the dashboard is filled right away: the monitor rebuilds the last 30 minutes
(the longest selectable range, `--replay-minutes`) from the LM Studio log, including periods
when the monitor was not running, and reloads the daily view from the CSV files. GPU and host
values are not in the log; the monitor saves them every 15 s to `monitoring/.live_hardware.json`
(numbers only) and restores them, so they are available for the time the monitor was running.
"Requests since start" counts new requests only.

### Live tests with real users: recording sessions

For a live test (e.g. a seminar where students use AnythingLLM), start a **session** in the bar
below the dashboard header, or start the monitor with `--session "Seminar KI, 22.10."`. Each
session gets its own folder `monitoring/sessions/<date_time>_<name>/`:

| File | Content |
|---|---|
| `requests.csv` | One row per request: start, end, status (`completed`, `failed`, `rejected`, `stale`, `dismissed`), duration, time to first token, tokens in/out, tokens/s, model, source (API / LM Studio app), streamed |
| `intervals.csv` | The monitor's per-interval values (default one row per minute): requests, concurrency, p95, tokens/s, GPU, memory |
| `hardware.csv` | GPU utilisation, GPU memory, temperature, power, system memory, CPU every 5 s |
| `session.json` | Name, start/end, host (GPU) and models |
| `summary.json` / `report.html` | Key figures (including busy time, throughput and GPU utilisation while busy, longest idle period) and charts: requests and response time per minute (median, p95), concurrency, tokens/s, GPU, memory, distribution of response times, per-model statistics |

"End session" writes the report; "Create report" writes it at any time without ending the
session, and the dashboard links to it. A session survives a monitor restart: it is resumed,
and requests made while the monitor was not running are added from the LM Studio log without
duplicates. When the monitor stops, it writes the report so far. "Clear" ends a running session
(with report) and starts a new one with the same name. Reports can be regenerated from the CSV
files at any time:

```bash
python live_session.py --report monitoring/sessions/<folder>
```

**Report for a past time range.** A session can also be created afterwards for any time range, of any length
(Sessions card): requests are reconstructed from the LM Studio log, per-minute values come from
the monitor's daily CSV files, and hardware values from the monitor's 5-second data where it still
holds them, otherwise from the per-minute values. Minutes without data count as 0 in the charts
(response times are left empty instead, since an idle minute has no response time). Without a
running monitor in that time, only the request data is available.

Sessions store only numbers, times and model names – no prompts, responses, IP addresses or
user identifiers – so the data cannot be linked to individual students. Without a session,
the monitor still writes the daily per-minute CSV (`monitoring/lmstudio_metrics_<date>.csv`).

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
  loads on demand) is dropped after 120 s and shown as "rejected" in the dashboard. A request
  ended by an ERROR entry for its model (e.g. the model was unloaded) is closed as failed and
  excluded from response times. The monitor
  does not analyse response content, so refusals in the answers of real users are not evaluated.

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
