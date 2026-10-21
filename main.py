"""
Locust user definition for load testing the AnythingLLM developer API.

Each simulated user sends questions from prompts.json. By default every question
is asked in a new workspace thread, so requests are independent of each other
(no shared chat history) and can be processed in parallel. With
--messages-per-thread N, a thread is reused for N questions instead.

Threads are named traceably as ``Loadtest <run> u<user> #<seq>`` (slug
``lt-<run>-u<user>-<seq>``) and kept, but at most --max-threads (default 30) per
workspace: when the limit is exceeded, the oldest finished load-test threads are
deleted first (FIFO). The list of load-test threads is stored in
--thread-registry, so the limit holds across stages and runs. Threads that were
not created by the load test are never deleted.

Refusals are reported as separate entries in the Locust statistics:

* ``REJECTED <reason>`` - the request was rejected technically (HTTP error status,
  timeout, connection error or an ``error`` field in the response). These requests
  also count as failures.
* ``REFUSAL no-context`` - query mode found no relevant documents and AnythingLLM
  returned its refusal response. The request counts as answered.

With --refusal-log PATH, up to a few examples per category (prompt and the start
of the response) are written to a JSON Lines file for the report. Prompts are the
synthetic questions from prompts.json.

Required environment variables:
    ANYTHING_API_KEY  AnythingLLM developer API key
    WORKSPACE_SLUG    Slug of the workspace under test
"""
import itertools
import json
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from locust import HttpUser, task, events

API_KEY = os.environ["ANYTHING_API_KEY"]
WORKSPACE = os.environ["WORKSPACE_SLUG"]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"), "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

# AnythingLLM's default "query mode refusal response" (Workspace settings > Chat settings).
DEFAULT_REFUSAL_TEXT = "There is no relevant information in this workspace to answer your query."
EXAMPLES_PER_CATEGORY = 3
EXAMPLE_CHARS = 150

_examples_lock = threading.Lock()
_examples_written = {}
_user_ids = itertools.count(1)  # Sequential number per simulated user within one Locust run.
REGISTRY = None                 # ThreadRegistry, created when Locust initialises.


# ---------- Custom options (also shown in the Locust web UI) ----------
@events.init_command_line_parser.add_listener
def add_custom_args(parser):
    g = parser.add_argument_group("LLM test")
    g.add_argument("--wait-min", type=float, default=5.0,
                   help="Minimum think time in seconds")
    g.add_argument("--wait-max", type=float, default=15.0,
                   help="Maximum think time in seconds")
    g.add_argument("--mode", choices=["query", "chat"], default="query",
                   help="query = answer from workspace documents only (RAG), chat = regular chat")
    g.add_argument("--request-timeout", type=float, default=300.0,
                   help="Timeout per request in seconds")
    g.add_argument("--messages-per-thread", type=int, default=1,
                   help="Questions per thread before a new one is started (1 = new thread per question, 0 = never)")
    g.add_argument("--refusal-text", default=DEFAULT_REFUSAL_TEXT,
                   help="Query mode refusal response configured in the workspace")
    g.add_argument("--refusal-log", default="",
                   help="JSON Lines file for refusal examples (empty = disabled)")
    g.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"),
                   help="Identifier included in every thread name (default: start time)")
    g.add_argument("--max-threads", type=int, default=30,
                   help="Load-test threads kept per workspace; the oldest are deleted first (0 = keep all)")
    g.add_argument("--thread-registry", default=str(Path(__file__).resolve().parent / ".loadtest_threads.json"),
                   help="File that records the threads created by the load test")


class ThreadRegistry:
    """
    FIFO list of the threads created by the load test, persisted so that the limit holds
    across stages and runs. Only threads listed here are ever deleted, and never one that
    is still in use.
    """

    def __init__(self, path, limit):
        self.path = Path(path)
        self.limit = limit
        self.lock = threading.Lock()
        self.entries = []  # Oldest first: {"workspace", "slug", "name", "created", "done"}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            # Threads from earlier runs are no longer in use.
            self.entries = [{**e, "done": True} for e in data.get("threads", [])]
        except (OSError, ValueError):
            pass

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"threads": self.entries}, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, workspace, slug, name):
        with self.lock:
            self.entries.append({"workspace": workspace, "slug": slug, "name": name,
                                 "created": round(time.time(), 1), "done": False})
            self._save()

    def finish(self, slug):
        with self.lock:
            for entry in self.entries:
                if entry["slug"] == slug:
                    entry["done"] = True
            self._save()

    def take_excess(self, workspace):
        """Remove and return the oldest finished threads of ``workspace`` beyond the limit."""
        if self.limit <= 0:
            return []
        with self.lock:
            own = [e for e in self.entries if e["workspace"] == workspace]
            excess = len(own) - self.limit
            victims = [e for e in own if e["done"]][:max(0, excess)]
            if victims:
                self.entries = [e for e in self.entries if e not in victims]
                self._save()
            return victims


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    global REGISTRY
    opts = environment.parsed_options
    REGISTRY = ThreadRegistry(opts.thread_registry, opts.max_threads)


def rejection_reason(response):
    """Return a short reason if the chat request was rejected technically, else None."""
    if response.status_code == 0:
        error = type(response.error).__name__ if response.error else "connection"
        return "timeout" if "Timeout" in error else "connection-error"
    if response.status_code != 200:
        return f"http-{response.status_code}"
    return None


def run_label(run_id):
    """Make a run identifier safe for use in a thread slug."""
    return re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-") or "run"


def record_example(environment, category, prompt, status, text, thread=None):
    """Append one refusal example (synthetic prompt + start of the response) to the refusal log."""
    path = environment.parsed_options.refusal_log
    if not path:
        return
    with _examples_lock:
        if _examples_written.get(category, 0) >= EXAMPLES_PER_CATEGORY:
            return
        _examples_written[category] = _examples_written.get(category, 0) + 1
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"category": category, "status": status, "prompt": prompt, "thread": thread,
                                "response": (text or "")[:EXAMPLE_CHARS]}, ensure_ascii=False) + "\n")


class AnythingLLMUser(HttpUser):

    # Defined as a method so the values can come from the command-line options.
    def wait_time(self):
        opts = self.environment.parsed_options
        return random.uniform(opts.wait_min, opts.wait_max)

    def on_start(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        self.thread_slug = None
        self.msg_count = 0
        self.user_no = next(_user_ids)
        self.thread_seq = 0
        self.create_thread()

    def on_stop(self):
        self.finish_thread()

    def create_thread(self):
        """Open a new thread with a traceable name: run, simulated user and sequence number."""
        opts = self.environment.parsed_options
        self.thread_seq += 1
        run = run_label(opts.run_id)
        requested_slug = f"lt-{run}-u{self.user_no:03d}-{self.thread_seq:04d}"
        thread_name = f"Loadtest {run} u{self.user_no:03d} #{self.thread_seq:04d}"
        with self.client.post(
            f"/api/v1/workspace/{WORKSPACE}/thread/new",
            headers=self.headers,
            json={"name": thread_name, "slug": requested_slug},
            name="/thread/new",
            timeout=30,
            catch_response=True,
        ) as response:
            if response.status_code not in (200, 201):
                self.thread_slug = None
                response.failure(f"Thread: {response.status_code} {(response.text or '')[:200]}")
                return
            try:
                data = response.json()
                self.thread_slug = data.get("thread", {}).get("slug", requested_slug)
                self.msg_count = 0
                response.success()
                REGISTRY.add(WORKSPACE, self.thread_slug, thread_name)
            except ValueError:
                self.thread_slug = None
                response.failure("Invalid response while creating thread")

    def finish_thread(self):
        """Mark the current thread as done and delete the oldest threads beyond the limit (FIFO)."""
        if self.thread_slug:
            REGISTRY.finish(self.thread_slug)
            self.thread_slug = None
        for entry in REGISTRY.take_excess(WORKSPACE):
            with self.client.delete(
                f"/api/v1/workspace/{WORKSPACE}/thread/{entry['slug']}",
                headers=self.headers,
                name="/thread/delete",
                timeout=30,
                catch_response=True,
            ) as response:
                if response.status_code in (200, 404):  # 404: already deleted in AnythingLLM.
                    response.success()
                else:
                    response.failure(f"Delete thread: {response.status_code} {(response.text or '')[:200]}")

    def report(self, request_type, name, response_time):
        """Add a refusal entry to the Locust statistics (shown separately from the chat requests)."""
        self.environment.events.request.fire(
            request_type=request_type, name=name, response_time=response_time,
            response_length=0, exception=None, context={})

    @task
    def ask(self):
        opts = self.environment.parsed_options
        if not self.thread_slug:
            self.create_thread()
            if not self.thread_slug:
                return

        self.send_chat(opts)  # Returns once the complete answer has been received and recorded.

        # A thread is done after N questions (default: every question).
        self.msg_count += 1
        if opts.messages_per_thread and self.msg_count >= opts.messages_per_thread:
            self.finish_thread()

    def send_chat(self, opts):
        endpoint = f"/api/v1/workspace/{WORKSPACE}/thread/{self.thread_slug}/chat"
        prompt = random.choice(QUESTIONS)
        payload = {"message": prompt, "mode": opts.mode}

        with self.client.post(
            endpoint,
            headers=self.headers,
            json=payload,
            name=f"/thread/chat [{opts.mode}]",   # Keeps statistics separate per mode.
            timeout=opts.request_timeout,
            catch_response=True,
        ) as response:
            elapsed_ms = response.elapsed.total_seconds() * 1000 if response.elapsed else 0
            reason = rejection_reason(response)
            if reason:
                self.report("REJECTED", reason, elapsed_ms)
                record_example(self.environment, f"REJECTED {reason}", prompt, response.status_code, response.text,
                               self.thread_slug)
                response.failure(f"HTTP {response.status_code}: {(response.text or '')[:200]}")
                return
            try:
                result = response.json()
            except ValueError:
                response.failure("Response is not valid JSON")
                return
            if result.get("error"):
                self.report("REJECTED", "error-response", elapsed_ms)
                record_example(self.environment, "REJECTED error-response", prompt, 200, str(result["error"]),
                               self.thread_slug)
                response.failure(f"Error response: {str(result['error'])[:200]}")
                return
            text = result.get("textResponse")
            if not text:
                response.failure("No textResponse received")
                return
            response.success()
            if opts.mode == "query" and text.strip() == opts.refusal_text.strip() and not result.get("sources"):
                self.report("REFUSAL", "no-context", elapsed_ms)
                record_example(self.environment, "REFUSAL no-context", prompt, 200, text, self.thread_slug)
