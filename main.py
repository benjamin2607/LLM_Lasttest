import os
import random
import uuid
import json

from locust import HttpUser, task, events

API_KEY = os.environ["ANYTHING_API_KEY"]
WORKSPACE = os.environ["WORKSPACE_SLUG"]

with open("prompts.json", "r") as f:
    QUESTIONS = json.load(f)


# ---------- Eigene Parameter (erscheinen auch in der Web-UI) ----------
@events.init_command_line_parser.add_listener
def add_custom_args(parser):
    g = parser.add_argument_group("LLM-Test")
    g.add_argument("--wait-min", type=float, default=5.0,
                   help="Minimale Denkpause in Sekunden")
    g.add_argument("--wait-max", type=float, default=15.0,
                   help="Maximale Denkpause in Sekunden")
    g.add_argument("--mode", choices=["query", "chat"], default="query",
                   help="query = nur RAG/Dokumente, chat = normaler Chat")
    g.add_argument("--request-timeout", type=float, default=300.0,
                   help="Timeout pro Anfrage in Sekunden")
    g.add_argument("--messages-per-thread", type=int, default=5,
                   help="Nach so vielen Nachrichten startet ein neuer Thread (0 = nie)")


class AnythingLLMUser(HttpUser):

    # wait_time als Methode, damit die Werte aus den Parametern kommen
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
                response.failure("Ungültige Antwort beim Erstellen des Threads")

    @task
    def ask(self):
        opts = self.environment.parsed_options

        # Neuer Thread nach N Nachrichten (begrenzt das Kontextwachstum)
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
            name=f"/thread/chat [{opts.mode}]",   # trennt die Statistik nach Modus
            timeout=opts.request_timeout,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                result = response.json()
                if not result.get("textResponse"):
                    response.failure("Keine textResponse erhalten")
                else:
                    self.msg_count += 1
                    response.success()
            except ValueError:
                response.failure("Antwort ist kein gültiges JSON")