# Quickstart

From zero to a running dashboard, a first load test and a recorded live session in about
15 minutes. Details are in the [README](README.md); every option is listed in
[PARAMETERS.md](PARAMETERS.md).

## 1. Requirements

- Python 3.10 or newer
- LM Studio with the local server running (default port 1234)
- AnythingLLM using LM Studio as its LLM provider (default port 3001) – only for AnythingLLM load tests
- Recommended: in LM Studio, **Developer → Logging → Verbose Logging** on, so that requests from
  the LM Studio chat window are tracked as well

Run the scripts under the same user account as LM Studio, so the monitor can read its log.

## 2. Install

In the project folder:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install locust
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install locust
```

Locust is the only package needed; everything else is the Python standard library.

## 3. Set the AnythingLLM access

Create an API key in AnythingLLM (**Settings → Developer API**) and note the slug of the workspace
to test (the last part of the workspace URL). Preferably use a separate test workspace with the
same documents and settings as the real one.

**macOS / Linux**

```bash
export ANYTHING_API_KEY="your-key"
export WORKSPACE_SLUG="your-workspace"
```

**Windows (PowerShell)**

```powershell
$env:ANYTHING_API_KEY="your-key"
$env:WORKSPACE_SLUG="your-workspace"
```

These variables apply to the current terminal window only. Set them before starting the monitor
if you want to start load tests from the dashboard.

## 4. Start the live monitor

```bash
python lmstudio_monitor.py
```

The dashboard opens at <http://127.0.0.1:8765>. The status LEDs at the top show whether the LM
Studio log, the API, the model and the GPU were found. Send a question in AnythingLLM: it appears
under **Running requests** and then under **Recently completed**.

If no requests appear, run `python lmstudio_monitor.py --diagnose` (see the README, Troubleshooting).

Keep the monitor running during load tests and live tests.

### Choose the model

The **LM Studio model** card (above the load test) shows all models with their load state. Pick a
model, then either keep it as loaded, **Adjust settings** (context length, parallel requests,
system prompt, temperature, top P, thinking; more under *Advanced*) and reload it, or reload it
with LM Studio's defaults. Click the save icon to keep the settings as a named configuration and
pick it again later from *Saved configuration*. **Unload model** (bottom left) frees the memory.

Tip: set *Parallel requests* at least to the number of users you want to test at the same time –
with fewer slots, LM Studio queues the requests.

## 5. Run a load test

**From the dashboard:** open **New load test …** in the Load test card at the bottom, give it a
name (it becomes the report title), choose **AnythingLLM** or **LM Studio native**, check the
stages (e.g. `1,2,4,8,16`) and start. The form shows which model will be tested. The chip at the
top shows stage, users and elapsed time; **Stop & delete threads** ends the test at any time.

Before the first stage, the test waits until the model is loaded and answers ("waiting for the
model"), so loading time is never measured.

**From a second terminal** (with the variables from step 3 set):

```bash
python auto_loadtest.py --dry-run --users 1,2,4,8,16
```

```bash
python auto_loadtest.py --users 1,2,4,8,16
```

The dry run only prints the plan and the expected duration. With the defaults, each stage takes
about 6 minutes. The test stops at the first stage that breaks a limit (p95 response time
above 120 s, more than 5 % failures, GPU memory or temperature too high).

**Result:** `loadtest_results/<date_time>/report.html`, also linked in the dashboard's load test
table. The *Result* section at the top shows the highest stable number of concurrent users and an
estimate of how many real students the system can serve.

### Testing LM Studio natively

To measure the model alone, without AnythingLLM, choose the model and its settings in the
**LM Studio model** card (keep it as loaded, adjust and reload, or load a saved configuration),
then choose **LM Studio native** in the load test form. On the command line:

```bash
python auto_loadtest.py --target lmstudio --users 1,2,4,8,16
```

No API key or workspace is needed; on the command line the first loaded model is used
(`-- --model ID` picks another).
The questions come from `prompts_lmstudio.json` and need no documents. Add
`-- --context-tokens 2000` to put a RAG-like input load on the model. The report has the same
layout; comparing it with an AnythingLLM report shows how much time AnythingLLM itself adds.

## 6. Record a live test with students

1. Click **Start session** below the dashboard header and enter a name, e.g. `Seminar 22.10.`
2. Let the students work as usual.
3. Click **End session**. The report is written to `monitoring/sessions/<date_time>_<name>/report.html`
   and linked in the Sessions card.

**PDF:** the **PDF** button next to each load test and session report opens it with a print layout
and the print dialog; choose "Save as PDF". Give load tests a name in the form, it becomes the
report title.

Forgot to start a session? As long as the monitor was running, use **Create a report for a past time range**
in the Sessions card and enter the start and end time.

Sessions store only numbers, times and model names, no prompts, answers or user data.

## 7. Next steps

- Compare `--mode query` and `--mode chat`; see [PARAMETERS.md](PARAMETERS.md#3-simulated-users-for-anythingllm--mainpy).
- Adjust the limits to what students will tolerate, e.g. `--max-p95 45`.
- Use `--refine` for the final measurement to find the exact limit.
- Edit `prompts.json` so the questions match the documents in the workspace.
- Save model configurations for the setups you compare (e.g. context length or parallel requests)
  and name the load tests accordingly; export the reports as PDF for your documentation.
