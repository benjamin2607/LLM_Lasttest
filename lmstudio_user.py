"""
Locust user definition for load testing LM Studio directly, without AnythingLLM.

Each simulated user sends questions from prompts_lmstudio.json (general questions that
need no workspace documents) to LM Studio's OpenAI-compatible endpoint
``/v1/chat/completions``. Every question is a new conversation without chat history,
like a new AnythingLLM thread.

Responses are streamed by default, as in a chat window. The response time recorded in
the Locust statistics is the time until the last chunk has arrived, not only until the
response headers, so it is comparable with the AnythingLLM load test.

With --context-tokens N, a block of about N tokens of lecture-style text is added to
every request, to put a similar input load on the model as the document excerpts that
AnythingLLM adds in RAG mode. The sentences are shuffled per request, so LM Studio
cannot simply reuse a cached prompt.

Technical rejections are reported as separate ``REJECTED <reason>`` entries in the
Locust statistics (HTTP error status, timeout, connection error, an error in the
response stream or an empty response); they also count as failures. With
--refusal-log PATH, a few examples per category are written to a JSON Lines file for
the report.

Optional environment variable:
    LMSTUDIO_API_TOKEN  Only needed if "Require API token" is enabled in LM Studio
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from locust import HttpUser, events, task

import locust_final_stats  # noqa: F401 - writes complete statistics when Locust quits.

API_TOKEN = os.environ.get("LMSTUDIO_API_TOKEN")
CHAT_ENDPOINT = "/v1/chat/completions"  # Also the name of the chat entry in the statistics.
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts_lmstudio.json"
DEFAULT_SYSTEM_PROMPT = ("Du bist ein hilfreicher Assistent für Studierende einer Hochschule. "
                         "Antworte sachlich, verständlich und auf Deutsch.")
EXAMPLES_PER_CATEGORY = 3
EXAMPLE_CHARS = 150
CHARS_PER_TOKEN = 4  # Rough average for German and English text; the report shows the exact count from the log.

# Lecture-style sentences for --context-tokens; the content does not matter, only the length.
CONTEXT_SENTENCES = [
    "Ein Algorithmus ist eine eindeutige Handlungsvorschrift zur Lösung eines Problems oder einer Klasse von Problemen.",
    "Die Laufzeit eines Algorithmus wird meist in Abhängigkeit von der Größe der Eingabe angegeben.",
    "Datenstrukturen wie Listen, Bäume und Hashtabellen bestimmen, wie effizient auf Daten zugegriffen werden kann.",
    "In relationalen Datenbanken werden Daten in Tabellen mit Zeilen und Spalten gespeichert.",
    "Die Normalisierung reduziert Redundanzen und verhindert Anomalien beim Einfügen, Ändern und Löschen.",
    "Ein Betriebssystem verwaltet die Hardware und stellt Programmen eine einheitliche Schnittstelle bereit.",
    "Prozesse besitzen einen eigenen Adressraum, während sich Threads eines Prozesses den Speicher teilen.",
    "Das Internetprotokoll sorgt für die Adressierung und Weiterleitung von Paketen zwischen Netzwerken.",
    "Verschlüsselung schützt die Vertraulichkeit von Daten bei der Übertragung und bei der Speicherung.",
    "Beim maschinellen Lernen werden Modelle anhand von Beispieldaten trainiert, statt explizit programmiert zu werden.",
    "Ein neuronales Netz besteht aus Schichten von Neuronen, deren Gewichte beim Training angepasst werden.",
    "Versionsverwaltungssysteme speichern die Änderungsgeschichte von Dateien und erleichtern die Zusammenarbeit.",
    "Tests stellen sicher, dass Software sich wie spezifiziert verhält und Änderungen keine Fehler einführen.",
    "Agile Methoden setzen auf kurze Iterationen, enge Zusammenarbeit und regelmäßiges Feedback.",
    "Die Datenschutz-Grundverordnung regelt die Verarbeitung personenbezogener Daten in der Europäischen Union.",
    "Cloud-Dienste stellen Rechenleistung und Speicher bedarfsgerecht über das Netzwerk bereit.",
]

_examples_lock = threading.Lock()
_examples_written = {}
QUESTIONS = []
MODEL = None  # Model id, resolved when Locust initialises.


@events.init_command_line_parser.add_listener
def add_custom_args(parser):
    g = parser.add_argument_group("LM Studio test")
    g.add_argument("--model", default="",
                   help="Model id as listed by LM Studio (default: the first loaded model, else the first LLM)")
    g.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="JSON file with a list of questions")
    g.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt of every request (empty = none)")
    g.add_argument("--context-tokens", type=int, default=0,
                   help="Add about N tokens of text to every request to simulate RAG excerpts (0 = off)")
    g.add_argument("--max-tokens", type=int, default=0,
                   help="Maximum tokens per answer (0 = no limit, as in AnythingLLM)")
    g.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    g.add_argument("--no-stream", action="store_true", help="Request the complete answer at once instead of streaming")
    g.add_argument("--wait-min", type=float, default=5.0, help="Minimum think time in seconds")
    g.add_argument("--wait-max", type=float, default=15.0, help="Maximum think time in seconds")
    g.add_argument("--request-timeout", type=float, default=300.0, help="Timeout per request in seconds")
    g.add_argument("--refusal-log", default="", help="JSON Lines file for rejection examples (empty = disabled)")
    g.add_argument("--run-id", default="", help="Identifier of the run (set by auto_loadtest.py; informational)")


def _get_json(url):
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as resp:
        return json.loads(resp.read())


def pick_model(host):
    """
    The first loaded language model, otherwise the first language model LM Studio offers (it then
    loads it on demand). Uses LM Studio's model list with load state and falls back to /v1/models.
    """
    try:
        models = [m for m in _get_json(f"{host}/api/v0/models").get("data", []) if m.get("type") in ("llm", "vlm")]
        loaded = [m for m in models if m.get("state") == "loaded"]
        if loaded or models:
            return (loaded or models)[0]["id"]
    except (urllib.error.URLError, OSError, ValueError):
        pass
    try:
        ids = [m.get("id", "") for m in _get_json(f"{host}/v1/models").get("data", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return next((i for i in ids if "embed" not in i.lower()), None)


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    global MODEL, QUESTIONS
    opts = environment.parsed_options
    with open(opts.prompts, encoding="utf-8") as f:
        QUESTIONS = json.load(f)
    MODEL = opts.model or pick_model((environment.host or "").rstrip("/"))
    if not MODEL:
        raise SystemExit(f"No model found at {environment.host}. Load a model in LM Studio or pass --model.")
    print(f"[lmstudio] model {MODEL}, {len(QUESTIONS)} questions")


def context_block(tokens):
    """About ``tokens`` tokens of shuffled lecture-style text."""
    sentences, text = CONTEXT_SENTENCES[:], []
    target = tokens * CHARS_PER_TOKEN
    while sum(len(s) + 1 for s in text) < target:
        random.shuffle(sentences)
        text += sentences
    out, length = [], 0
    for sentence in text:
        if length >= target:
            break
        out.append(sentence)
        length += len(sentence) + 1
    return " ".join(out)


def rejection_reason(response):
    """Return a short reason if the request was rejected technically, else None."""
    if response.status_code == 0:
        error = type(response.error).__name__ if response.error else "connection"
        return "timeout" if "Timeout" in error else "connection-error"
    if response.status_code != 200:
        return f"http-{response.status_code}"
    return None


def record_example(environment, category, prompt, status, text):
    """Append one rejection example (synthetic prompt + start of the response) to the refusal log."""
    path = environment.parsed_options.refusal_log
    if not path:
        return
    with _examples_lock:
        if _examples_written.get(category, 0) >= EXAMPLES_PER_CATEGORY:
            return
        _examples_written[category] = _examples_written.get(category, 0) + 1
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"category": category, "status": status, "prompt": prompt,
                                "response": (text or "")[:EXAMPLE_CHARS]}, ensure_ascii=False) + "\n")


def read_stream(lines):
    """
    Collect an OpenAI-style server-sent event stream. Returns (text, error): the generated text
    (answer and reasoning) and the error message LM Studio sent in the stream, if any.
    """
    parts, error = [], None
    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if chunk.get("error"):
            err = chunk["error"]
            error = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            break
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            parts.append((delta.get("content") or "") + (delta.get("reasoning_content") or ""))
    return "".join(parts), error


def read_message(result):
    """Text and error of a complete (non-streamed) chat completion response."""
    if result.get("error"):
        err = result["error"]
        return "", err.get("message", str(err)) if isinstance(err, dict) else str(err)
    message = ((result.get("choices") or [{}])[0].get("message") or {})
    return (message.get("content") or "") + (message.get("reasoning_content") or ""), None


class LMStudioUser(HttpUser):

    # Defined as a method so the values can come from the command-line options.
    def wait_time(self):
        opts = self.environment.parsed_options
        return random.uniform(opts.wait_min, opts.wait_max)

    def on_start(self):
        self.headers = {"Content-Type": "application/json"}
        if API_TOKEN:
            self.headers["Authorization"] = f"Bearer {API_TOKEN}"

    def report(self, name, response_time):
        """Add a rejection entry to the Locust statistics (shown separately from the chat requests)."""
        self.environment.events.request.fire(request_type="REJECTED", name=name, response_time=response_time,
                                             response_length=0, exception=None, context={})

    def build_payload(self, opts, prompt):
        system = opts.system_prompt or ""
        if opts.context_tokens > 0:
            system = (system + "\n\n" if system else "") + "Kontext aus den Unterlagen:\n" + context_block(opts.context_tokens)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": MODEL, "messages": messages, "temperature": opts.temperature, "stream": not opts.no_stream}
        if opts.max_tokens > 0:
            payload["max_tokens"] = opts.max_tokens
        if payload["stream"]:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @task
    def ask(self):
        opts = self.environment.parsed_options
        prompt = random.choice(QUESTIONS)
        stream = not opts.no_stream
        started = time.perf_counter()
        with self.client.post(CHAT_ENDPOINT, headers=self.headers, json=self.build_payload(opts, prompt),
                              name=CHAT_ENDPOINT, timeout=opts.request_timeout, stream=stream,
                              catch_response=True) as response:
            reason = rejection_reason(response)
            text, error = "", None
            if reason is None:
                try:
                    if stream:
                        text, error = read_stream(response.iter_lines())
                    else:
                        text, error = read_message(response.json())
                except ValueError:
                    error = "response is not valid JSON"
                except Exception as exc:  # noqa: BLE001 - e.g. the connection broke or timed out mid-stream.
                    reason = "timeout" if "Timeout" in type(exc).__name__ else "connection-error"
                    error = type(exc).__name__
            # Locust measures a streamed request until the headers arrive; record the time to the last chunk.
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.request_meta["response_time"] = elapsed_ms
            response.request_meta["response_length"] = len(text.encode("utf-8"))
            if reason is None and error is not None:
                reason = "error-response"
            elif reason is None and not text.strip():
                reason, error = "empty-response", "no content received"
            if reason:
                self.report(reason, elapsed_ms)
                body = error if error is not None else (response.text or "")  # HTTP errors have a short body.
                record_example(self.environment, f"REJECTED {reason}", prompt, response.status_code, body)
                response.failure(f"{reason}: {(body or '')[:200]}" if body else reason)
                return
            response.success()
