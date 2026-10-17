"""
Locust user definition for load testing the AnythingLLM developer API.

Each simulated user opens a workspace thread, sends questions from prompts.json
and starts a new thread after a configurable number of messages to bound
context growth.

Required environment variables:
    ANYTHING_API_KEY  AnythingLLM developer API key
    WORKSPACE_SLUG    Slug of the workspace under test
"""
import os
import random
import uuid
import json

from locust import HttpUser, task, events

API_KEY = os.environ["ANYTHING_API_KEY"]
WORKSPACE = os.environ["WORKSPACE_SLUG"]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"), "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)


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
    g.add_argument("--messages-per-thread", type=int, default=5,
                   help="Start a new thread after this many messages (0 = never)")


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
        self.create_thread()

    def create_thread(self):
        requested_slug = f"locust-{uuid.uuid4().hex}"
        with self.client.post(
            f"/api/v1/workspace/{WORKSPACE}/thread/new",
            headers=self.headers,
            json={"slug": requested_slug},
            name="/thread/new",
            timeout=30,
            catch_response=True,
        ) as response:
            if response.status_code not in (200, 201):
                self.thread_slug = None
                response.failure(f"Thread: {response.status_code} {response.text[:200]}")
                return
            try:
                data = response.json()
                self.thread_slug = data.get("thread", {}).get("slug", requested_slug)
                self.msg_count = 0
                response.success()
            except ValueError:
                self.thread_slug = None
                response.failure("Invalid response while creating thread")

    @task
    def ask(self):
        opts = self.environment.parsed_options

        # Rotate to a new thread after N messages to bound context growth.
        if opts.messages_per_thread and self.msg_count >= opts.messages_per_thread:
            self.create_thread()

        if not self.thread_slug:
            self.create_thread()
            if not self.thread_slug:
                return

        endpoint = f"/api/v1/workspace/{WORKSPACE}/thread/{self.thread_slug}/chat"
        payload = {"message": random.choice(QUESTIONS), "mode": opts.mode}

        with self.client.post(
            endpoint,
            headers=self.headers,
            json=payload,
            name=f"/thread/chat [{opts.mode}]",   # Keeps statistics separate per mode.
            timeout=opts.request_timeout,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                result = response.json()
                if not result.get("textResponse"):
                    response.failure("No textResponse received")
                else:
                    self.msg_count += 1
                    response.success()
            except ValueError:
                response.failure("Response is not valid JSON")
