"""
Parser that turns LM Studio server log lines into request metrics.

LM Studio writes three kinds of relevant entries, depending on how a request
arrives and whether "Verbose Logging" (Developer settings) is enabled:

=========================  ==========================================================
Source                     Entries (level)
=========================  ==========================================================
HTTP API                   ``Received request: POST to /v1/...`` (DEBUG),
                           ``[model] Running chat completion ...`` (INFO),
                           ``[model] Generated packet: {...}`` per streamed chunk (INFO),
                           ``Finished streaming response`` / ``Generated prediction`` (INFO)
llama.cpp engine           ``<uptime> I slot launch_slot_: id S | task T | ...`` (DEBUG),
                           ``... print_timing ...`` progress and timings (DEBUG),
                           ``... release: ... stop processing`` (DEBUG)
=========================  ==========================================================

Chats in the LM Studio app itself only produce engine entries, and those are
written only with verbose logging enabled. API requests are visible at INFO level
in any case. A request that appears in several sources (an API request served by
the llama.cpp engine) is merged into a single record, so nothing is counted twice.

Only timestamps, identifiers, model names and token counts are extracted. Prompt
and response text in JSON bodies is never stored.
"""
import re
import statistics
import threading
import time
from collections import Counter, OrderedDict, deque
from datetime import datetime
from pathlib import Path

# --- Common entry structure ---
TS_RE = re.compile(r"^\[?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
LEVEL_RE = re.compile(r"\[(INFO|DEBUG|WARN|WARNING|ERROR)\]")
# "[2026-10-21 15:03:37][INFO][google/gemma-4-e4b] ..." -> "google/gemma-4-e4b"
MODEL_TAG_RE = re.compile(r"^\[[^\]]+\]\[INFO\]\[([^\]]{1,200})\] ")
NON_MODEL_TAGS = ("Plugin(", "LM STUDIO", "LMSAuthenticator", "Client=")

# --- HTTP API layer ---
# Matches both "Received POST request to /v1/..." and "Received request: POST to /v1/...".
REQUEST_RE = re.compile(r"Received (?:request: )?(POST|GET) (?:request )?to (\S+)", re.I)
RUNNING_RE = re.compile(r"Running (?:chat )?completion on conversation", re.I)
STREAMING_RE = re.compile(r"\] Streaming response\.\.\.", re.I)
PROMPT_PROGRESS_RE = re.compile(r"\] Prompt processing progress:", re.I)
STREAM_FLAG_RE = re.compile(r'"stream"\s*:\s*(true|false)')
FIRST_TOKEN_RE = re.compile(r"First token generated", re.I)
STREAM_DONE_RE = re.compile(r"Finished streaming response", re.I)
PREDICTION_RE = re.compile(r"Generated prediction", re.I)
PACKET_RE = re.compile(r"Generated packet", re.I)
# Non-streaming responses: one entry per generated token. Only the count is read; the text after it is ignored.
ACCUMULATED_RE = re.compile(r"\] Accumulated \d+ tokens in (?:response|reasoning content)")
ACCUMULATED_RECENT_S = 60   # While such entries appear, non-streaming tokens are counted live from them.
IDLE_CLOSE_S = 2.0          # A multi-line entry counts as complete after this long without a new line.
# Only these fields are read from JSON bodies; all other content is ignored.
MODEL_RE = re.compile(r'"model"\s*:\s*"([^"]{1,200})"')
PROMPT_TOK_RE = re.compile(r'"prompt_tokens"\s*:\s*(\d+)')
COMPL_TOK_RE = re.compile(r'"completion_tokens"\s*:\s*(\d+)')
TPS_RE = re.compile(r'"tokens_per_second"\s*:\s*([\d.]+)')
COMPLETION_ID_RE = re.compile(r'"id"\s*:\s*"(chatcmpl-[^"]{1,100})"')
FINISH_REASON_RE = re.compile(r'"finish_reason"\s*:\s*"([^"]+)"')

# --- llama.cpp engine layer ---
# "0.31.546.459 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0"
# The prefix is the engine uptime as minutes.seconds.milliseconds.microseconds.
ENGINE_RE = re.compile(r"(?:^|\]\s?)(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) [IWED] (slot|srv) +(\w+): (.*)$")
SLOT_RE = re.compile(r"id +(\d+) \| task +(-?\d+) \| (.*)$")
LOAD_MODEL_RE = re.compile(r"loading model '([^']+)'")
IS_CHILD_RE = re.compile(r"is_child = (\d+)")
NGEN_RE = re.compile(r"\bn_(?:gen|decoded) = +(\d+), tg = +([\d.]+) t/s")
PROMPT_EVAL_RE = re.compile(r"prompt eval time = +([\d.]+) ms / +(\d+) tokens")
EVAL_RE = re.compile(r"(?<!prompt )\beval time = +([\d.]+) ms / +(\d+) tokens \([^,]*, +([\d.]+) tokens per second")

ENGINE_BASE_TOLERANCE_S = 3.0  # Max. clock offset to attribute uptime readings to the same engine process.


def classify_endpoint(endpoint):
    """Map an API path to 'embedding', 'generation' or None (not tracked)."""
    if "embedding" in endpoint:
        return "embedding"
    if "completions" in endpoint or "/responses" in endpoint or "/chat" in endpoint:
        return "generation"
    return None


def pct(values, p):
    """Nearest-rank percentile; returns None for an empty sequence."""
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(p / 100 * len(values)) - 1))
    return values[k]


def rnd(value, digits=2):
    return round(value, digits) if value is not None else None


MIN_GENERATION_S = 0.5  # Shorter generation times give no meaningful tokens/s (log entries arrive in bursts).
BACKLOG_S = 1.5         # Entries read this much later than their timestamp get their log time.


def tokens_per_second(req):
    """
    Generation speed of one request: the engine's or LM Studio's own figure if available, otherwise
    completion tokens over the time from the first token (or the start) to the end. Returns None
    when that time is too short to be meaningful.
    """
    if req.get("tps"):
        return req["tps"]
    out_tok, start, first, end = req.get("completion_tokens"), req.get("start"), req.get("first"), req.get("end")
    if not out_tok or start is None or end is None:
        return None
    begin = first if first is not None and start <= first <= end else start
    span = end - begin
    return out_tok / span if span >= MIN_GENERATION_S else None


def model_tag(line):
    """Return the model name from an INFO entry tagged with a model, or None."""
    m = MODEL_TAG_RE.match(line)
    if m and not m.group(1).startswith(NON_MODEL_TAGS):
        return m.group(1)
    return None


class LogParser:
    """
    Stateful log parser. Feed it lines in order with :meth:`feed`; read metrics
    with :meth:`live` (sliding windows for the dashboard) and :meth:`snapshot`
    (one aggregation window, e.g. one CSV row).

    Matching rules:
      * Streamed API responses carry a completion ID in every chunk and are tracked exactly.
      * llama.cpp engine tasks carry a task ID and are tracked exactly per engine process.
      * An engine task is merged with the oldest API request that is not yet bound to a task.
      * API completions without an ID (non-streaming) are matched in arrival order.
    """

    def __init__(self, use_log_time=False, stale_after=900, dispatch_timeout=120, stall_after=300, clock=time.time):
        self.lock = threading.Lock()
        self.use_log_time = use_log_time
        self.stale_after = stale_after            # Inactivity after which an in-flight request is given up.
        self.dispatch_timeout = dispatch_timeout  # Max. wait for "Running chat completion" after "Received request".
        self.stall_after = stall_after            # Without any log activity for this long, a request is "stalled".
        self.clock = clock
        self.default_model = None         # Set by the caller when exactly one model is loaded.
        # Called as on_request_done(request, status) when a request ends: "completed", "failed",
        # "rejected", "stale" or "dismissed". Runs while the parser lock is held; keep it short.
        self.on_request_done = None
        self.open = []                    # In-flight requests, oldest first.
        self.api_pending = deque()        # API requests awaiting their API-level completion entry.
        self.by_cid = {}                  # Streamed requests keyed by completion ID.
        self.done_cids = OrderedDict()    # Recently finished streams: late chunks (e.g. usage) belong to them.
        self._entry_now = None            # Time assigned to the current log entry (continuation lines inherit it).
        self._last_accumulated = None     # Time of the last per-token entry of a non-streaming response.
        self.by_task = {}                 # Engine tasks keyed by (engine id, task id).
        self.engines = []                 # Known engine processes: {"id", "base", "model", "file"}.
        self.pending_stream_done = 0      # Streams finished by their final chunk, awaiting "Finished streaming".
        self.last_running = None          # Most recent API request that logged "Running chat completion".
        self.last_api_done = None
        self.just_streamed = False
        self.ctx = None                   # JSON body being read: "request", "packet", "prediction" or None.
        self.ctx_req = None
        self.last_ts = clock()
        self.last_line_wall = None
        self.lines_seen = 0
        self.signals = Counter()          # Recognised entry types (for diagnostics; no content).
        self.token_events = deque(maxlen=200_000)  # (time, tokens) as tokens are generated.
        self.recent = deque(maxlen=5000)  # Completed requests (numeric fields only).
        self.totals = {"started": 0, "completed": 0, "embeddings": 0, "errors": 0, "rejected": 0, "failed": 0,
                       "dismissed": 0,
                       "prompt_tokens": 0, "completion_tokens": 0}
        self._seq = 0
        self._new_window()

    # ------------------------------------------------------------------ helpers
    def _new_window(self):
        self.w_start = self.last_ts if self.use_log_time else self.clock()
        self.w = {"started": 0, "completed": 0, "embeddings": 0, "errors": 0, "stale": 0, "failed": 0,
                  "max_active": len(self.open), "latencies": [], "ttfts": [], "tps": [],
                  "prompt_tokens": 0, "completion_tokens": 0, "models": {}}

    def _inc(self, key, n=1):
        """Increment a counter in both the current window and the running totals."""
        self.w[key] += n
        self.totals[key] += n

    def _now(self, ts_match):
        if not self.use_log_time:
            # Live: the time a line is read, unless it is read clearly later than it was written (LM Studio
            # writes in bursts under load, or the monitor catches up): then the entry's own timestamp counts.
            wall = self.clock()
            if ts_match is None:
                return self._entry_now if self._entry_now is not None else wall
            try:
                logged = datetime.strptime(ts_match.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                logged = wall
            self._entry_now = logged + 0.5 if wall - logged > BACKLOG_S else wall
            return self._entry_now
        if ts_match:
            try:
                self.last_ts = datetime.strptime(ts_match.group(1).replace("T", " "),
                                                 "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                pass
        return self.last_ts  # Continuation lines inherit the time of their entry.

    def _new_request(self, now, source, model=None):
        self._seq += 1
        req = {"seq": self._seq, "start": now, "first": None, "end": None, "model": model,
               "sources": {source}, "api": source == "api", "active_at": now}
        self.open.append(req)
        if req["api"]:
            self.api_pending.append(req)
        self._inc("started")
        self.w["max_active"] = max(self.w["max_active"], len(self.open))
        return req

    # --------------------------------------------------------------- main entry
    def feed(self, line):
        """Process a single log line."""
        with self.lock:
            self.lines_seen += 1
            self.last_line_wall = self.clock()
            ts = TS_RE.match(line)
            engine = ENGINE_RE.search(line)
            if ts:
                self._close_ctx()
                if "[DEBUG]" in line[:40]:
                    self.signals["debug_entries"] += 1
            if engine:
                self._engine(engine, self._now(ts))
                return
            if not ts and not LEVEL_RE.search(line[:60]):
                self._scan_body(line)  # Continuation line of a multi-line JSON body.
                return
            self._api(line, ts)

    # ---------------------------------------------------------------- API layer
    def _api(self, line, ts):
        now = self._now(ts)
        streamed, self.just_streamed = self.just_streamed, False

        rq = REQUEST_RE.search(line)
        if rq:
            method, endpoint = rq.groups()
            kind = classify_endpoint(endpoint) if method.upper() == "POST" else None
            if kind == "embedding":
                self.signals["api_embedding"] += 1
                self._inc("embeddings")
            elif kind == "generation":
                self.signals["api_received"] += 1
                req = self._new_request(now, "api")
                self.ctx, self.ctx_req = "request", req
                self._scan_body(line)
            return
        if RUNNING_RE.search(line):
            self.signals["api_running"] += 1
            tag = model_tag(line)
            # With verbose logging, the request was already announced by "Received request". The gap can be
            # long (e.g. while LM Studio loads the model on demand), so match by order and model, not by time.
            req = next((r for r in self.open if r["api"] and not r.get("running") and not r.get("task")
                        and r["end"] is None and r.get("model") in (None, tag)), None)
            if req is None:
                req = self._new_request(now, "api", tag)  # Verbose logging off: this is the first sign.
            req["running"] = True
            req["active_at"] = now
            req["model"] = req["model"] or tag
            self.last_running = req
            return
        if PROMPT_PROGRESS_RE.search(line):
            # Prompt evaluation is progressing for the oldest dispatched request of this model.
            tag = model_tag(line)
            req = next((r for r in self.open if r.get("running") and r["first"] is None
                        and r.get("model") in (None, tag)), None)
            if req is not None:
                req["active_at"] = now
            return
        if STREAMING_RE.search(line):
            if self.last_running is not None:
                self.last_running["stream_flag"] = True
            return
        if ACCUMULATED_RE.search(line):
            self.token_events.append((now, 1))
            self._last_accumulated = now
            return
        if PACKET_RE.search(line):
            self.signals["api_packet"] += 1
            self.ctx, self.ctx_req = "packet", {"now": now}
            self._scan_body(line)
            return
        if FIRST_TOKEN_RE.search(line):
            req = next((r for r in self.open if r["api"] and r["first"] is None), None)
            if req is not None:
                req["first"] = now
            return
        if STREAM_DONE_RE.search(line):
            self.signals["api_stream_done"] += 1
            if self.pending_stream_done:
                self.pending_stream_done -= 1  # Already completed by its final chunk.
            else:
                req = self._pop_api_pending(prefer_streamed=True)
                if req is not None:
                    self._api_complete(req, now)
                    self._book(req)
            self.just_streamed = True
            return
        if PREDICTION_RE.search(line):
            self.signals["api_prediction"] += 1
            # Directly after "Finished streaming" this belongs to the same request: do not complete it twice.
            if streamed and self.last_api_done is not None:
                req = self.last_api_done
            else:
                req = self._pop_api_pending(prefer_streamed=False)
                if req is not None:
                    self._api_complete(req, now)
            if req is not None:
                self.ctx, self.ctx_req = "prediction", req
                self._scan_body(line)
            return
        lvl = LEVEL_RE.search(line[:60])
        if lvl and lvl.group(1) == "ERROR":
            self.signals["errors"] += 1
            self._inc("errors")
            tag = model_tag(line.replace("[ERROR]", "[INFO]", 1))
            if tag:
                self._abort_oldest(tag, now)

    def _notify(self, req, status):
        if self.on_request_done is not None:
            try:
                self.on_request_done(req, status)
            except Exception as exc:  # A failing consumer must never break log parsing.
                print(f"[monitor] could not record request: {exc}")

    def _abort_oldest(self, model, now):
        """
        An ERROR entry tagged with a model (e.g. "Model unloaded by user or API request") ends one
        of its requests without an answer: close the oldest open one as failed. Failed requests do
        not count as completed and are excluded from latency statistics.
        """
        req = next((r for r in self.open if r.get("model") == model), None)
        if req is None:
            return
        self._drop(req)
        req["failed"] = True
        req["end"] = now
        self.w["failed"] += 1
        self.totals["failed"] += 1
        self._notify(req, "failed")

    def _is_stalled(self, req, now):
        return now - req["active_at"] > self.stall_after

    def _drop(self, req):
        """Stop tracking an in-flight request without recording it as completed."""
        if req in self.open:
            self.open.remove(req)
        if req in self.api_pending:
            self.api_pending.remove(req)
        self.by_task.pop(req.get("task"), None)
        self.by_cid.pop(req.get("cid"), None)

    def dismiss(self, seq=None, stalled_only=False):
        """
        Remove in-flight requests from tracking, e.g. entries whose completion LM Studio never
        logged. ``seq`` selects one request; ``stalled_only`` selects all stalled ones. This only
        affects the monitor: LM Studio offers no way to cancel another client's request.
        Returns the number of removed requests.
        """
        with self.lock:
            now = self.clock()
            victims = [r for r in self.open
                       if (seq is not None and r["seq"] == seq)
                       or (stalled_only and self._is_stalled(r, now))]
            for req in victims:
                self._drop(req)
                req["dismissed"] = True
                req["end"] = now
                self._notify(req, "dismissed")
            self.totals["dismissed"] += len(victims)
            return len(victims)

    def _pop_api_pending(self, prefer_streamed):
        """
        Pick the request an unidentified API completion entry belongs to: same streaming mode
        first, and among those one whose engine task has already ended, else the oldest.
        """
        if not self.api_pending:
            return None

        def is_stream(r):
            return bool(r.get("streamed") or r.get("stream_flag"))

        same_mode = [r for r in self.api_pending if is_stream(r) == prefer_streamed] or list(self.api_pending)
        req = next((r for r in same_mode if r["end"] is not None), same_mode[0])
        self.api_pending.remove(req)
        return req

    def _api_complete(self, req, now):
        """Mark the API-level completion of ``req``; finishes it unless the engine already did."""
        req["api_done"] = True
        self.last_api_done = req
        if req in self.api_pending:
            self.api_pending.remove(req)
        if req["end"] is None:
            self._finish(req, now)

    def _scan_body(self, line):
        """Extract identifiers and token counts from a JSON body; all other content is discarded."""
        if self.ctx is None or self.ctx_req is None:
            return
        if self.ctx == "packet":
            self._scan_packet(line)
            return
        req = self.ctx_req
        if req.get("model") is None:
            m = MODEL_RE.search(line)
            if m:
                req["model"] = m.group(1)
        if self.ctx == "request":
            m = STREAM_FLAG_RE.search(line)
            if m:
                req["stream_flag"] = m.group(1) == "true"
        if self.ctx == "prediction":
            for regex, key in ((PROMPT_TOK_RE, "prompt_tokens"), (COMPL_TOK_RE, "completion_tokens")):
                m = regex.search(line)
                if m:
                    req[key] = int(m.group(1))
                    req["usage_seen"] = True
            m = TPS_RE.search(line)
            if m:
                req["tps"] = float(m.group(1))

    def _scan_packet(self, line):
        """Track a streamed chunk: bind it to its request by completion ID; content is ignored."""
        pkt = self.ctx_req
        m = COMPLETION_ID_RE.search(line)
        if m and "req" not in pkt:
            cid, now = m.group(1), pkt["now"]
            req = self.by_cid.get(cid)
            if req is None and cid in self.done_cids:
                # A chunk after the final one (e.g. the usage block of stream_options.include_usage):
                # it belongs to the finished request and must not start a new one.
                pkt["req"] = self.done_cids[cid]
                return
            if req is None:
                # First chunk of a new stream: bind to the oldest open streaming API request without an ID.
                unbound = [r for r in self.open if r["api"] and not r.get("cid") and r.get("stream_flag") is not False]
                # Hung entries (no activity for a while) must not capture a new stream, otherwise the
                # real request would look queued forever; prefer live candidates, oldest first.
                unbound = sorted(unbound, key=lambda r: (self._is_stalled(r, now), not r.get("stream_flag")))
                req = unbound[0] if unbound else None
                if req is None:  # Monitor started mid-stream.
                    req = self._new_request(now, "api")
                req["cid"], req["streamed"] = cid, True
                self.by_cid[cid] = req
            if req["first"] is None:
                req["first"] = now
            req["chunks"] = req.get("chunks", 0) + 1
            req["active_at"] = now
            if not req.get("task"):
                self.token_events.append((now, 1))
                req["live_tokens"] = True
            pkt["req"] = req
            return
        req = pkt.get("req")
        if req is None:
            return
        if req.get("model") is None:
            m = MODEL_RE.search(line)
            if m:
                req["model"] = m.group(1)
        for regex, key in ((PROMPT_TOK_RE, "prompt_tokens"), (COMPL_TOK_RE, "completion_tokens")):
            m = regex.search(line)
            if m:  # Optional usage block (stream_options.include_usage), possibly after the request was booked.
                value = int(m.group(1))
                if req.get("booked"):
                    self._inc(key, value - (req.get(key) or 0))  # Replace the chunk-count estimate.
                    req["tokens_estimated"] = False
                req[key] = value
                req["usage_seen"] = True
        m = FINISH_REASON_RE.search(line)
        if m and not req.get("api_done"):
            req["chunks"] -= 1  # The final chunk carries no token.
            if not req.get("task") and self.token_events:
                self.token_events.pop()
            self._api_complete(req, pkt["now"])
            self.pending_stream_done += 1
            pkt["final"] = True

    def _close_ctx(self):
        """Called when a new log entry starts: settle accounting that depended on the finished body."""
        if self.ctx == "packet" and self.ctx_req.get("final"):
            self._book(self.ctx_req["req"])
        elif self.ctx == "prediction":
            self._book(self.ctx_req)
        self.ctx = self.ctx_req = None

    # ------------------------------------------------------------- engine layer
    def _engine_for(self, base):
        """Return the engine process whose start time matches ``base`` (wall time minus uptime)."""
        for eng in self.engines:
            if abs(eng["base"] - base) <= ENGINE_BASE_TOLERANCE_S:
                return eng
        eng = {"id": len(self.engines) + 1, "base": base, "model": None, "file": None}
        self.engines.append(eng)
        return eng

    def _engine(self, m, now):
        minutes, seconds, ms, us, component, func, rest = m.groups()
        uptime = int(minutes) * 60 + int(seconds) + int(ms) / 1000 + int(us) / 1e6
        eng = self._engine_for(now - uptime)
        self.signals["engine_entries"] += 1

        if component == "srv":
            lm = LOAD_MODEL_RE.search(rest)
            if func == "load_model" and lm:
                eng["file"] = Path(lm.group(1).replace("\\", "/")).stem  # File name only, never the path.
            return
        sm = SLOT_RE.search(rest)
        if not sm:
            return
        task, detail = int(sm.group(2)), sm.group(3)
        if task < 0:
            return
        key = (eng["id"], task)

        if func == "launch_slot_":
            child = IS_CHILD_RE.search(detail)
            if child and child.group(1) != "0":
                return  # Parallel sampling child of another task.
            self.signals["engine_launch"] += 1
            self._engine_launch(key, eng, uptime, now)
            return
        req = self.by_task.get(key)
        if req is None:
            return
        if func == "print_timing":
            self._engine_timing(req, detail, uptime, now)
        elif func == "release":
            self.signals["engine_release"] += 1
            self._engine_release(req, uptime, now)

    def _engine_launch(self, key, eng, uptime, now):
        candidates = [r for r in self.open if r["api"] and not r.get("task") and r["end"] is None]
        candidates.sort(key=lambda r: self._is_stalled(r, now))  # Stable: oldest live candidate first.
        if eng["model"]:
            same_model = [r for r in candidates if r.get("model") == eng["model"]]
            candidates = same_model or [r for r in candidates if r.get("model") is None] or candidates
        req = candidates[0] if candidates else None
        if req is None:
            # Not announced by the API layer, e.g. a chat in the LM Studio app.
            req = self._new_request(now, "engine", eng["model"] or eng["file"] or self.default_model)
        else:
            req["sources"].add("engine")
            if req.get("model") and not eng["model"]:
                eng["model"] = req["model"]  # Learn which model this engine process serves.
        req.update(task=key, engine_id=eng["id"], et_launch=uptime, launched=now, ngen=0, ngen_et=uptime,
                   active_at=now)
        self.by_task[key] = req

    def _engine_timing(self, req, detail, uptime, now):
        req["active_at"] = now
        ng = NGEN_RE.search(detail)
        if ng:
            n, tg = int(ng.group(1)), float(ng.group(2))
            if req["first"] is None:
                req["first"] = now
            delta = n - req["ngen"]
            # Never count more than the reported speed allows in the elapsed engine time (guards against
            # mismatched or restarted counters, which would show as huge spikes).
            plausible = max(tg, req.get("tg") or 0, 1.0) * max(uptime - req["ngen_et"], 1.0) * 1.5 + 32
            counted_elsewhere = req.get("streamed") or (req["api"] and self._accumulated_recent(now))
            if not counted_elsewhere and 0 < delta <= plausible:
                self.token_events.append((now, delta))
                req["live_tokens"] = True
            req.update(ngen=n, ngen_et=uptime, tg=tg)
            return
        pe = PROMPT_EVAL_RE.search(detail)
        if pe:
            req["prompt_ms"] = float(pe.group(1))
            req.setdefault("prompt_tokens", int(pe.group(2)))
            if not req.get("streamed"):  # Streamed requests already have an exact first-chunk time.
                # Time to first token = queueing before the engine started + prompt evaluation.
                req["first"] = req["launched"] + req["prompt_ms"] / 1000
            return
        ev = EVAL_RE.search(detail)
        if ev:
            req["engine_completion_tokens"] = int(ev.group(2))
            req["engine_tps"] = float(ev.group(3))

    def _engine_release(self, req, uptime, now):
        if req.get("streamed") and not req.get("api_done"):
            # Engine tasks are bound in launch order, streams by completion ID. If this task's stream is
            # still sending chunks while another engine-bound stream has just finished, the order differed.
            other = next((r for r in self.by_task.values()
                          if r is not req and r.get("streamed") and r.get("api_done")), None)
            if other is not None:
                self._swap_engine_fields(req, other)
                req = other
        self.by_task.pop(req["task"], None)
        if "engine_completion_tokens" not in req and req.get("ngen"):
            # Progress is printed every few seconds; extrapolate to the release time.
            req["engine_completion_tokens"] = round(req["ngen"] + req.get("tg", 0) * max(0.0, uptime - req["ngen_et"]))
            req["tokens_estimated"] = True
        extra = req.get("engine_completion_tokens", 0) - req.get("ngen", 0)
        if extra > 0 and not req.get("streamed"):
            self.token_events.append((now, extra))
        if req["end"] is None:
            if not req["api"]:
                req["start"] = now - (uptime - req["et_launch"])  # Exact engine-side duration.
            self._finish(req, now)
        self._book(req)

    ENGINE_FIELDS = ("task", "engine_id", "et_launch", "launched", "ngen", "ngen_et", "tg",
                     "prompt_ms", "engine_completion_tokens", "engine_tps")

    def _swap_engine_fields(self, a, b):
        """Exchange the engine task bindings of two requests."""
        for field in self.ENGINE_FIELDS:
            va, vb = a.pop(field, None), b.pop(field, None)
            if vb is not None:
                a[field] = vb
            if va is not None:
                b[field] = va
        self.by_task[a["task"]] = a
        self.by_task[b["task"]] = b

    # ------------------------------------------------------------- completion
    def _accumulated_recent(self, now):
        return self._last_accumulated is not None and now - self._last_accumulated <= ACCUMULATED_RECENT_S

    def _finish(self, req, now):
        """Mark ``req`` as finished and record latency metrics."""
        if req in self.open:
            self.open.remove(req)
        if req["api"] and not req.get("streamed") and self._accumulated_recent(now):
            req["live_tokens"] = True  # Its tokens were already counted from the per-token entries.
        req["end"] = now
        self._inc("completed")
        self.recent.append(req)
        self.w["latencies"].append(now - req["start"])
        if req["first"] is not None:
            self.w["ttfts"].append(max(0.0, req["first"] - req["start"]))

    def _book(self, req):
        """Account tokens once all sources that will report on ``req`` have done so."""
        if req.get("booked") or req["end"] is None:
            return
        if req["api"] and not req.get("api_done"):
            return  # API-level completion (with a possible usage block) still to come.
        if req.get("task") and req["task"] in self.by_task:
            return  # Engine still running.
        if req.get("cid"):
            self.by_cid.pop(req["cid"], None)
            self.done_cids[req["cid"]] = req
            while len(self.done_cids) > 2000:
                self.done_cids.popitem(last=False)
        req["booked"] = True
        if not req.get("usage_seen"):
            if "engine_completion_tokens" in req:
                req["completion_tokens"] = req["engine_completion_tokens"]
            elif req.get("chunks"):
                req["completion_tokens"] = req["chunks"]
                req["tokens_estimated"] = True
        req.setdefault("tps", req.get("engine_tps"))
        self._inc("completion_tokens", req.get("completion_tokens") or 0)
        self._inc("prompt_tokens", req.get("prompt_tokens") or 0)
        if req.get("tps"):
            self.w["tps"].append(req["tps"])
        if req.get("model"):
            self.w["models"][req["model"]] = self.w["models"].get(req["model"], 0) + 1
        self._notify(req, "completed")

    def _close_idle_ctx(self, now):
        """Book a multi-line entry (e.g. the usage block of the last response) when no further line follows."""
        if self.ctx is not None and self.last_line_wall is not None and now - self.last_line_wall > IDLE_CLOSE_S:
            self._close_ctx()

    def _expire_stale(self, now):
        def never_started(r):
            # Announced but never dispatched to a model: LM Studio rejected it without a log entry.
            return r["api"] and not r.get("running") and not r.get("task") and not r.get("cid")

        for req in list(self.open):
            if never_started(req) and now - req["start"] > self.dispatch_timeout:
                self.totals["rejected"] += 1
                status = "rejected"
            elif now - req["active_at"] <= self.stale_after:
                continue  # Still showing activity (or not idle long enough to give up on it).
            else:
                status = "stale"
            self._drop(req)  # E.g. an aborted request that never logged a completion entry.
            self.w["stale"] += 1
            req["end"] = now
            self._notify(req, status)

    def replay(self, lines, since, until, sample=None):
        """
        Rebuild the recent state from earlier log lines, e.g. after the monitor restarts.

        Entries from ``since`` (epoch seconds) onwards are processed in log time. ``sample``
        is called once per second from ``since`` to ``until`` with that second, while the
        parser's clock is set to it, so the caller can record a data point via :meth:`live`.
        Afterwards the parser continues in wall-clock time; the totals and the aggregation
        window start fresh, so CSV rows and "since start" counters cover new data only.
        """
        live_clock, cursor = self.clock, int(since)
        self.use_log_time, self.clock = True, (lambda: self._replay_now)
        self._replay_now = float(cursor)

        def advance(to):
            nonlocal cursor
            while cursor < to:
                self._replay_now = float(cursor)
                if sample is not None:
                    sample(cursor)
                cursor += 1

        started = False
        for line in lines:
            ts = TS_RE.match(line)
            if ts:
                try:
                    t = datetime.strptime(ts.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    t = None
                if t is not None:
                    if t < since:
                        started = False
                        continue
                    if t > until:
                        break  # Entries after the window are not part of this replay.
                    started = True
                    advance(int(t))
                    self._replay_now = t
            if started:
                self.feed(line)
        with self.lock:
            self._close_ctx()  # The last entry of the range has no following line that would complete it.
        advance(int(until) + 1)

        self.clock, self.use_log_time = live_clock, False
        with self.lock:
            self.last_line_wall = None
            self.totals = {k: 0 for k in self.totals}
            self._new_window()

    def reset(self):
        """Start the statistics afresh (dashboard "Clear"); requests still in flight keep being tracked."""
        with self.lock:
            self.recent.clear()
            self.token_events.clear()
            self.totals = {k: 0 for k in self.totals}
            self._new_window()

    def prime_engines(self, lines):
        """
        Learn which model each running engine process serves from earlier "loading model" entries,
        without counting any requests. Used when following a log from its current end.
        """
        with self.lock:
            last_ts = None
            for line in lines:
                ts = TS_RE.match(line)
                if ts:
                    try:
                        last_ts = datetime.strptime(ts.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S").timestamp()
                    except ValueError:
                        continue
                m = ENGINE_RE.search(line)
                if m and last_ts is not None and m.group(6) == "load_model" and LOAD_MODEL_RE.search(m.group(7)):
                    minutes, seconds, ms, us = (int(g) for g in m.groups()[:4])
                    eng = self._engine_for(last_ts - (minutes * 60 + seconds + ms / 1000 + us / 1e6))
                    eng["file"] = Path(LOAD_MODEL_RE.search(m.group(7)).group(1).replace("\\", "/")).stem

    # ---------------------------------------------------------------- readers
    def active(self):
        with self.lock:
            return len(self.open)

    def live_point(self, stream_window=10, tps_window=30, rate_window=60):
        """
        The request part of one live data point (active, waiting, tokens/s, requests/min).
        Cheaper than :meth:`live` because it only looks at the most recent entries, which matters
        when the replay after a restart rebuilds hours of per-second points.
        """
        now = self.clock()
        with self.lock:
            self._close_idle_ctx(now)
            self._expire_stale(now)
            active = len(self.open)
            waiting = sum(1 for r in self.open if r["first"] is None or r["first"] > now)
            chunks = 0
            for t, n in reversed(self.token_events):
                if now - t > stream_window:
                    break
                chunks += n
            done_rate = done_tokens = 0
            for r in reversed(self.recent):
                age = now - r["end"]
                if age > max(rate_window, tps_window):
                    break
                if age <= rate_window:
                    done_rate += 1
                if age <= tps_window and not r.get("live_tokens"):
                    done_tokens += r.get("completion_tokens") or 0
        return {"active": active, "waiting": waiting,
                "tokens_per_s": rnd(chunks / stream_window + done_tokens / tps_window, 1),
                "requests_per_min": round(done_rate * 60 / rate_window, 1)}

    def completed_since(self, since):
        """Numeric fields of the requests completed since ``since`` (epoch seconds)."""
        with self.lock:
            return [{"start": r["start"], "end": r["end"], "duration": r["end"] - r["start"],
                     "ttft": max(0.0, r["first"] - r["start"]) if r["first"] is not None else None,
                     "tokens_in": r.get("prompt_tokens"), "tokens_out": r.get("completion_tokens"),
                     "tps": tokens_per_second(r), "model": r.get("model")}
                    for r in self.recent if r["end"] >= since]

    def latency_points(self, window=1800):
        """Duration and time-to-first-token of completed requests within ``window`` seconds."""
        now = self.clock()
        with self.lock:
            return [{"t": r["end"], "duration": rnd(r["end"] - r["start"]),
                     "ttft": rnd(max(0.0, r["first"] - r["start"])) if r["first"] is not None else None,
                     "tokens_in": r.get("prompt_tokens"), "tokens_out": r.get("completion_tokens"),
                     "tps": rnd(tokens_per_second(r), 1)}
                    for r in self.recent if now - r["end"] <= window]

    def live(self, tps_window=30, stream_window=10, rate_window=60, lat_window=300, recent_n=25):
        """Real-time snapshot for the dashboard, using sliding windows on wall-clock time."""
        now = self.clock()
        with self.lock:
            self._close_idle_ctx(now)
            self._expire_stale(now)
            running = [{"id": r["seq"], "age": round(now - r["start"], 1),
                        "generating": r["first"] is not None and r["first"] <= now,
                        "idle_s": round(now - r["active_at"]), "stalled": self._is_stalled(r, now),
                        "model": r.get("model"), "source": "api" if r["api"] else "app"} for r in self.open]
            recent = list(self.recent)
            totals = dict(self.totals)
            last_line = self.last_line_wall
            live_tokens = 0
            for t, n in reversed(self.token_events):
                if now - t > stream_window:
                    break
                live_tokens += n
            signals = {"verbose": self.signals["debug_entries"] > 0,
                       "api": any(self.signals[k] for k in ("api_received", "api_running", "api_packet")),
                       "engine": self.signals["engine_launch"] > 0}

        def ended(win):
            return [r for r in recent if now - r["end"] <= win]

        lat = [r["end"] - r["start"] for r in ended(lat_window)]
        ttft = [max(0.0, r["first"] - r["start"]) for r in ended(lat_window) if r["first"] is not None]

        def row(r):
            dur = r["end"] - r["start"]
            out_tok = r.get("completion_tokens")
            return {"end": r["end"], "duration_s": rnd(dur),
                    "ttft_s": rnd(max(0.0, r["first"] - r["start"])) if r["first"] is not None else None,
                    "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": out_tok,
                    "tokens_estimated": bool(r.get("tokens_estimated")),
                    "tokens_per_s": rnd(tokens_per_second(r), 1),
                    "model": r.get("model"), "source": "app" if not r["api"] else "api"}

        # Tokens are counted as they are generated where possible; otherwise once the request completes.
        completed_tokens = sum(r.get("completion_tokens") or 0 for r in ended(tps_window) if not r.get("live_tokens"))
        return {
            "active": len(running), "waiting": sum(not r["generating"] for r in running),
            "running": sorted(running, key=lambda r: -r["age"]),
            "tokens_per_s": rnd(live_tokens / stream_window + completed_tokens / tps_window, 1),
            "requests_per_min": round(len(ended(rate_window)) * 60 / rate_window, 1),
            "latency_p95_s": rnd(pct(lat, 95)), "latency_avg_s": rnd(statistics.fmean(lat)) if lat else None,
            "ttft_p95_s": rnd(pct(ttft, 95)),
            "recent": [row(r) for r in reversed(recent[-recent_n:])],
            "totals": totals, "log_age_s": rnd(now - last_line, 0) if last_line else None,
            "log_signals": signals,
        }

    def snapshot(self, reset=True):
        """Aggregate the current window into a flat dict (one CSV row) and optionally start a new window."""
        with self.lock:
            now = self.last_ts if self.use_log_time else self.clock()
            if not self.use_log_time:
                self._close_idle_ctx(now)
            self._expire_stale(now)
            w, duration = self.w, max(now - self.w_start, 1e-6)
            lat, ttft = w["latencies"], w["ttfts"]
            snap = {
                "requests_started": w["started"], "requests_completed": w["completed"],
                "embedding_requests": w["embeddings"], "log_errors": w["errors"],
                "stale_requests": w["stale"], "failed_requests": w["failed"], "active_now": len(self.open),
                "max_active": max(w["max_active"], len(self.open)),
                "latency_avg_s": rnd(statistics.fmean(lat)) if lat else None,
                "latency_p95_s": rnd(pct(lat, 95)),
                "ttft_avg_s": rnd(statistics.fmean(ttft)) if ttft else None,
                "ttft_p95_s": rnd(pct(ttft, 95)),
                "prompt_tokens": w["prompt_tokens"], "completion_tokens": w["completion_tokens"],
                "gen_tokens_per_s_total": rnd(w["completion_tokens"] / duration),
                "tokens_per_s_per_request_avg": rnd(statistics.fmean(w["tps"])) if w["tps"] else None,
                "models_used": ";".join(f"{k}:{v}" for k, v in sorted(w["models"].items())),
            }
            if reset:
                self._new_window()
            return snap


def diagnose(path, max_lines=200_000):
    """
    Summarise which entry types a log file contains, without printing any content.
    Returns a list of report lines.
    """
    parser = LogParser(use_log_time=True)
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    for line in lines:
        parser.feed(line)
    s, t = parser.signals, parser.totals
    report = [f"Log file: {Path(path).name} ({len(lines)} lines analysed)", "",
              "Recognised entries:"]
    labels = [("debug_entries", "DEBUG entries (verbose logging)"), ("api_received", "API: request received"),
              ("api_running", "API: chat completion started"), ("api_packet", "API: streamed chunks"),
              ("api_stream_done", "API: stream finished"), ("api_prediction", "API: prediction (non-streaming)"),
              ("api_embedding", "API: embedding requests"), ("engine_launch", "Engine: task started"),
              ("engine_release", "Engine: task released"), ("errors", "ERROR entries")]
    report += [f"  {label:<36} {s[key]:>8}" for key, label in labels]
    report += ["", f"Requests reconstructed: {t['completed']} completed, {len(parser.open)} still open; "
               f"{t['completion_tokens']} completion tokens, {t['prompt_tokens']} prompt tokens."]
    hints = []
    if not s["debug_entries"]:
        hints.append("Verbose logging appears to be off. API requests are still recognised, but chats in the "
                     "LM Studio app are not. Enable it in LM Studio: Developer > Logging > Verbose Logging.")
    if not (s["api_running"] or s["api_received"] or s["engine_launch"]):
        hints.append("No generation requests found. Send a request and run the diagnosis again. If requests were "
                     "sent, the log format may differ: please share this report (it contains no content).")
    if hints:
        report += ["", "Hints:"] + [f"  - {h}" for h in hints]
    return report
