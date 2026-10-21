# Parameter reference

All settings of the load test and the live monitor in one place. **Required** settings are marked
as such; everything else is optional and has a working default. The guideline values assume the
lab setup (AnythingLLM + LM Studio on one workstation with a large GPU, e.g. RTX 6000 Pro, 128 GB
RAM) and should be adjusted to your own hardware and model.

Contents:

1. [Environment variables](#1-environment-variables)
2. [Automated load test – `auto_loadtest.py`](#2-automated-load-test--auto_loadtestpy)
3. [Simulated users – `main.py`](#3-simulated-users--mainpy)
4. [Live monitor – `lmstudio_monitor.py`](#4-live-monitor--lmstudio_monitorpy)
5. [Load test form in the dashboard](#5-load-test-form-in-the-dashboard)
6. [Session reports – `live_session.py`](#6-session-reports--live_sessionpy)
7. [Typical commands](#7-typical-commands)

---

## 1. Environment variables

| Variable | Needed by | Required | Description |
|---|---|---|---|
| `ANYTHING_API_KEY` | load test; monitor (only to delete load-test threads and to start load tests from the dashboard) | **yes** for the load test | AnythingLLM developer API key (AnythingLLM → Settings → Developer API). Never written to disk by these scripts. |
| `WORKSPACE_SLUG` | load test; monitor (default of the dashboard form) | **yes** for the load test | Slug of the workspace to test, as shown in the workspace URL (e.g. `my-workspace`). Use a dedicated test workspace if possible. |
| `LMSTUDIO_API_TOKEN` | monitor | only if "Require API token" is enabled in LM Studio | Same as `--token`. |
| `LMSTUDIO_LOG_DIR` | monitor, load test | no | LM Studio log folder if it is not in the default location (`~/.lmstudio/server-logs`). `--log` takes precedence. |
| `NO_COLOR` | load test | no | Any value disables coloured console output. |

Setting them:

```bash
export ANYTHING_API_KEY="..."        # Linux/macOS
export WORKSPACE_SLUG="my-workspace"
```

```powershell
$env:ANYTHING_API_KEY="..."          # Windows PowerShell (current window only)
$env:WORKSPACE_SLUG="my-workspace"
```

---

## 2. Automated load test – `auto_loadtest.py`

```bash
python auto_loadtest.py [options] [-- options for main.py]
```

The test runs one *stage* per number of concurrent users, measures each stage and stops at the
first stage that breaks a limit. Everything after `--` is forwarded to `main.py`
(see [section 3](#3-simulated-users--mainpy)).

### Stages

| Option | Default | Description | Guideline |
|---|---|---|---|
| `--users LIST` | – | Explicit stages, e.g. `1,2,4,8,16,32`. Overrides the three options below. | Roughly doubling steps, e.g. `1,2,4,8,16,24,32,48,64`. Add steps beyond the expected limit. |
| `--start-users N` | `1` | First stage (if `--users` is not given). | `1` |
| `--step N` | `2` | Additive step between stages. | `2`–`8` |
| `--step-factor F` | `1.0` | Values `> 1` give multiplicative steps (`2` → 1, 2, 4, 8 …). | `2` for a first rough search |
| `--max-users N` | `64` | Last stage. | Above the expected limit |
| `--refine` | off | After the first failing stage, bisects between the last passing and the first failing stage to find the exact limit. Adds a few stages. | Use for the final measurement |

### Timing

| Option | Default | Description | Guideline |
|---|---|---|---|
| `--stage-duration S` | `300` | Measurement time per stage, in seconds (after ramp-up). | `300` (5 min). Short tests: `120`. Below `60`, a stage has too few answers to be meaningful. |
| `--spawn-rate R` | `1.0` | New simulated users per second during ramp-up. | `1`–`2`; higher values produce an unrealistic burst |
| `--cooldown S` | `30` | Pause between stages, so the queue drains. | `30`–`60` |
| `--stop-timeout S` | `120` | Grace period for requests still running at the end of a stage. | At least the expected p95 response time |

### Limits (a stage fails if one is exceeded)

| Option | Default | Description | Guideline |
|---|---|---|---|
| `--max-fail-rate F` | `0.05` | Maximum share of failed requests (HTTP errors, timeouts, error responses). `0.05` = 5 %. | `0.01`–`0.05` |
| `--max-p95 S` | `120` | Maximum 95th percentile of the response time of a chat request, in seconds. | What students will tolerate: `30`–`60` for an interactive chat, `120` as an upper bound |
| `--max-vram-pct P` | `98` | Maximum GPU memory use, in percent. | `95`–`98` |
| `--max-gpu-temp C` | `88` | Maximum GPU temperature, in °C (skipped on macOS). | `85`–`88`; check your GPU's specification |

### Reporting

| Option | Default | Description | Guideline |
|---|---|---|---|
| `--refusal-warn-rate F` | `0.10` | Refusal rate from which refusals are highlighted in red. Refusals never fail a stage. | `0.10` |
| `--student-think-time S` | `90` | Assumed time between two questions of one real student; used to estimate how many students the system can serve (Little's law). | `60`–`180`; measure it in a live session if possible |
| `--out DIR` | `loadtest_results` | Results folder (one subfolder per run). | – |

### Connection and setup

| Option | Default | Description |
|---|---|---|
| `--host URL` | `http://localhost:3001` | AnythingLLM base URL. |
| `--locustfile PATH` | `main.py` next to the script | Locust file with the user behaviour. |
| `--lmstudio-log VALUE` | `auto` | LM Studio log for exact token and timing figures: `auto` (default location), a path, or `off`. |
| `--skip-preflight` | off | Skip the initial check of the API key, the workspace and a test question. |
| `--dry-run` | off | Only print the planned stages and the expected duration, then exit. |

---

## 3. Simulated users – `main.py`

Options for the simulated users. With `auto_loadtest.py` they go after `--`; with a manual Locust
run they are passed to Locust directly (`python -m locust -f main.py --host … --mode chat`).

| Option | Default | Description | Guideline |
|---|---|---|---|
| `--mode` | `query` | `query`: answers only from the workspace documents (RAG); without matching documents AnythingLLM sends its refusal response. `chat`: regular chat; documents are used when relevant, otherwise the model answers freely. | Test the mode the students use; ideally run both |
| `--wait-min S` | `5` | Minimum think time of a simulated user between two questions, in seconds. | `5` |
| `--wait-max S` | `15` | Maximum think time, in seconds. | `15`. Short think times put a lot of load on few users; the report converts the result to real students with `--student-think-time`. |
| `--request-timeout S` | `300` | Timeout per request; a request without an answer by then counts as failed. | Above `--max-p95` |
| `--messages-per-thread N` | `1` | Questions per AnythingLLM thread before a new thread is started. `1` = a new thread for every question (no shared chat history), `0` = one thread per user. | `1` |
| `--max-threads N` | `30` | Load-test threads kept per workspace for inspection; beyond that, the oldest finished ones are deleted (FIFO). `0` = keep all. | `30` |
| `--refusal-text TEXT` | AnythingLLM default | The workspace's query mode refusal response, if it was changed in the workspace settings. | Only if customised |
| `--run-id ID` | start time | Identifier in every thread name. Set automatically by `auto_loadtest.py`. | – |
| `--thread-registry PATH` | `.loadtest_threads.json` | File that records the threads created by the load test (for the FIFO limit). | – |
| `--refusal-log PATH` | – | JSON Lines file for refusal examples. Set automatically by `auto_loadtest.py`. | – |

The questions come from `prompts.json` (a JSON list of strings); edit it to match the documents in
the test workspace.

---

## 4. Live monitor – `lmstudio_monitor.py`

```bash
python lmstudio_monitor.py [options]
```

None of the options is required: the monitor finds the LM Studio log, the API and the GPU on its own.

### Sources

| Option | Default | Description |
|---|---|---|
| `--log PATH` | `~/.lmstudio/server-logs` | LM Studio log file or folder (or `LMSTUDIO_LOG_DIR`). |
| `--lmstudio-url URL` | `http://localhost:1234` | LM Studio server, used for the loaded models and the status. |
| `--token TOKEN` | `LMSTUDIO_API_TOKEN` | LM Studio API token, only if "Require API token" is enabled. |
| `--no-api` | off | Do not query the LM Studio API (log and GPU only). |
| `--gpu-index N` | `0` | GPU to monitor in multi-GPU systems (NVIDIA). |
| `--from-start` | off | Read the whole log from the beginning instead of new lines only. Rarely needed; see `--replay-minutes`. |
| `--replay-minutes N` | `30` | On start, rebuild the dashboard from the last N minutes of the log, so a restart does not start empty. `0` = off. |

### Dashboard

| Option | Default | Description | Guideline |
|---|---|---|---|
| `--dashboard-port N` | `8765` | Port of the dashboard on `127.0.0.1` (local access only). `0` = no dashboard. | Change only if the port is in use |
| `--no-browser` | off | Do not open the dashboard in the browser on start. | – |
| `--keep-hours H` | `4` | Hours of per-second data kept in memory for the dashboard (time range "Max") and for range reports at 5-second resolution. | `4`–`12`; more hours use more memory |
| `--stall-minutes M` | `5` | A running request without any log activity for M minutes is marked as stalled and can be removed in the dashboard. | Above the longest expected answer |

### Alert thresholds (shown in the matching tile)

| Option | Default | Description | Guideline |
|---|---|---|---|
| `--alert-vram-pct P` | `95` | GPU memory, in percent. | `90`–`95` |
| `--alert-gpu-temp C` | `85` | GPU temperature, in °C. | `80`–`85` |
| `--alert-p95 S` | `60` | p95 response time over the last 5 minutes, in seconds. | What students will tolerate, e.g. `30`–`60` |
| `--alert-active N` | `0` | Number of concurrent requests; `0` = off. | The stable limit found by the load test |

### Recording and output

| Option | Default | Description |
|---|---|---|
| `--out DIR` | `monitoring` | Output folder (relative to the current folder): daily per-minute CSV files and `sessions/`. |
| `--interval S` | `60` | Aggregation interval of the CSV files, in seconds. Keep `60`, as the session reports work per minute. |
| `--session LABEL` | – | Start a recording session right away (same as **Start session** in the dashboard). |
| `--probe-interval S` | `0` | Send a short synthetic test request to LM Studio every S seconds to measure the response time even without users. `0` = off. Each probe adds a little load. |
| `--anythingllm-url URL` | `http://localhost:3001` | AnythingLLM, used to delete load-test threads ("Stop & delete threads"). Needs `ANYTHING_API_KEY`. |
| `--project-dir DIR` | folder of the script | Folder of `auto_loadtest.py`, its status and stop files and `loadtest_results/`. |
| `--diagnose` | off | Print which LM Studio log entry types are recognised (no content) and exit. |

---

## 5. Load test form in the dashboard

**New load test …** in the Load test card starts `auto_loadtest.py` with these fields. The monitor
checks every value against the range below before starting; the process is started without a shell.

| Field | Option | Allowed range | Default |
|---|---|---|---|
| AnythingLLM URL | `--host` | http(s) URL | `--anythingllm-url` of the monitor |
| Workspace slug | `WORKSPACE_SLUG` | letters, digits, `-`, `_` | `WORKSPACE_SLUG` of the monitor |
| API key | `ANYTHING_API_KEY` | – | `ANYTHING_API_KEY` of the monitor; passed to the process only, never stored |
| Mode | `-- --mode` | `query`, `chat` | `query` |
| Stages: concurrent users | `--users` | comma-separated, 1–9999 each | `1,2,4,8,16` |
| Measurement per stage (s) | `--stage-duration` | 10–7200 | `300` |
| New users per second | `--spawn-rate` | 0.1–50 | `1` |
| Pause between stages (s) | `--cooldown` | 0–600 | `30` |
| Grace for running requests (s) | `--stop-timeout` | 0–600 | `120` |
| Think time min / max (s) | `-- --wait-min` / `--wait-max` | 0–600, min ≤ max | `5` / `15` |
| Request timeout (s) | `-- --request-timeout` | 5–3600 | `300` |
| Limit: p95 response time (s) | `--max-p95` | 1–3600 | `120` |
| Limit: failure rate (%) | `--max-fail-rate` (as a fraction) | 0–100 % | `5` |
| Student think time for estimate (s) | `--student-think-time` | 1–3600 | `90` |
| Threads kept in AnythingLLM | `-- --max-threads` | 0–10000 | `30` |
| Find the exact limit | `--refine` | on/off | off |

The remaining options (GPU limits, refusal highlight, LM Studio log) keep their defaults; use the
command line to change them.

---

## 6. Session reports – `live_session.py`

| Option | Required | Description |
|---|---|---|
| `--report DIR` | **yes** | Rebuild `summary.json` and `report.html` of a session folder from its CSV files, e.g. `python live_session.py --report monitoring/sessions/<folder>`. |

Sessions are normally started, ended and created for past time ranges in the dashboard.

---

## 7. Typical commands

```bash
# Monitor for a live test with students (dashboard opens automatically)
python lmstudio_monitor.py

# Preview a load test
python auto_loadtest.py --dry-run --users 1,2,4,8,16,24,32,48,64

# Full capacity test in query mode with exact limit
python auto_loadtest.py --users 1,2,4,8,16,24,32,48,64 --refine -- --mode query

# Quick check (about 15 minutes)
python auto_loadtest.py --users 1,4,8 --stage-duration 120 --cooldown 15

# Strict interactive limits for a chat workspace
python auto_loadtest.py --users 2,4,8,12,16,24 --max-p95 45 --max-fail-rate 0.01 --refine -- --mode chat
```
