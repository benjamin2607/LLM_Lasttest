"""
Unit tests for lmstudio_log_parser.

The log lines reproduce the structure of real LM Studio 0.3.x / llama.cpp entries;
message content is replaced by placeholders. Run with:

    python -m unittest discover -s tests
"""
import json
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmstudio_log_parser import LogParser  # noqa: E402

SECRET = "CONFIDENTIAL-STUDENT-TEXT"
MODEL = "google/gemma-4-e4b"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def ts(level="INFO", tag=None):
    return "[2026-10-21 15:03:37][" + level + "]" + (f"[{tag}]" if tag else "")


def uptime(seconds):
    """Format engine uptime as minutes.seconds.milliseconds.microseconds."""
    minutes, rest = divmod(seconds, 60)
    whole = int(rest)
    frac = round((rest - whole) * 1_000_000)
    return f"{int(minutes)}.{whole:02d}.{frac // 1000:03d}.{frac % 1000:03d}"


class Harness:
    """Feeds lines while advancing a fake clock, and generates entries in LM Studio's format."""

    def __init__(self, engine_start=None):
        self.clock = Clock()
        self.parser = LogParser(clock=self.clock)
        self.engine_start = self.clock.t - 100 if engine_start is None else engine_start

    def feed(self, *lines, dt=0.0):
        for line in lines:
            self.parser.feed(line)
        self.clock.advance(dt)

    def engine(self, func, task, detail, slot=0, engine_start=None):
        base = self.engine_start if engine_start is None else engine_start
        return (f"{ts('DEBUG')} {uptime(self.clock.t - base)} I slot {func}: "
                f"id  {slot} | task {task} | {detail}")

    # --- API layer entries ---
    @staticmethod
    def received(stream=True):
        return [f"{ts('DEBUG')} Received request: POST to /v1/chat/completions with body {{",
                f'  "model": "{MODEL}",',
                f'  "messages": [{{"role": "user", "content": "{SECRET}"}}],',
                f'  "stream": {"true" if stream else "false"}',
                "}"]

    @staticmethod
    def running():
        return f"{ts('INFO', MODEL)} Running chat completion on conversation with 1 messages."

    @staticmethod
    def streaming():
        return f"{ts('INFO', MODEL)} Streaming response..."

    @staticmethod
    def packet(cid, final=False):
        return [f"{ts('INFO', MODEL)} Generated packet: {{",
                f'  "id": "{cid}",',
                '  "object": "chat.completion.chunk",',
                f'  "model": "{MODEL}",',
                '  "choices": [{',
                f'      "delta": {{"content": "{"" if final else SECRET}"}},',
                f'      "finish_reason": {json.dumps("stop") if final else "null"}',
                "  }]",
                "}"]

    @staticmethod
    def stream_done():
        return f"{ts('INFO', MODEL)} Finished streaming response"

    @staticmethod
    def prediction(prompt_tokens, completion_tokens):
        return [f"{ts('INFO', MODEL)} Generated prediction: {{",
                f'  "choices": [{{"message": {{"content": "{SECRET}"}}}}],',
                '  "usage": {',
                f'    "prompt_tokens": {prompt_tokens},',
                f'    "completion_tokens": {completion_tokens}',
                "  }",
                "}"]


def completed(parser):
    return parser.live()["recent"]


class StreamingApiTests(unittest.TestCase):
    def test_verbose_stream_served_by_engine_is_counted_once(self):
        h = Harness()
        h.feed(*h.received(stream=True), h.running(), h.streaming(), dt=0.1)
        h.feed(h.engine("launch_slot_", 7, "processing task, is_child = 0"), dt=0.4)
        for _ in range(5):
            h.feed(*h.packet("chatcmpl-a"), dt=0.1)
        h.feed(h.engine("print_timing", 7, "n_gen =      5, tg =  10.00 t/s, tg_3s =  10.00 t/s"))
        h.feed(*h.packet("chatcmpl-a", final=True), dt=0.05)
        h.feed(h.engine("release", 7, "stop processing: n_tokens = 40, truncated = 0"), h.stream_done())

        rows = completed(h.parser)
        self.assertEqual(len(rows), 1)
        self.assertEqual(h.parser.totals["started"], 1)
        self.assertEqual(rows[0]["completion_tokens"], 5)  # One token per streamed chunk.
        self.assertAlmostEqual(rows[0]["ttft_s"], 0.5, places=2)
        self.assertAlmostEqual(rows[0]["duration_s"], 1.0, places=2)
        self.assertEqual(rows[0]["source"], "api")
        self.assertEqual(h.parser.active(), 0)

    def test_stream_without_verbose_logging(self):
        h = Harness()
        h.feed(h.running(), h.streaming(), dt=0.3)
        for _ in range(3):
            h.feed(*h.packet("chatcmpl-b"), dt=0.1)
        h.feed(*h.packet("chatcmpl-b", final=True))
        h.feed(h.stream_done())

        rows = completed(h.parser)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["completion_tokens"], 3)
        self.assertEqual(rows[0]["model"], MODEL)
        self.assertAlmostEqual(rows[0]["ttft_s"], 0.3, places=2)

    def test_interleaved_streams_are_tracked_by_completion_id(self):
        h = Harness()
        h.feed(*h.received(), *h.received(), h.running(), h.running(), dt=0.1)
        h.feed(h.engine("launch_slot_", 1, "processing task, is_child = 0", slot=0),
               h.engine("launch_slot_", 2, "processing task, is_child = 0", slot=1), dt=0.2)
        for i in range(6):
            h.feed(*h.packet("chatcmpl-x" if i % 3 else "chatcmpl-y"), dt=0.05)
        h.feed(*h.packet("chatcmpl-y", final=True), h.engine("release", 2, "stop processing: n_tokens = 9", slot=1),
               h.stream_done())
        self.assertEqual(h.parser.active(), 1)
        h.feed(*h.packet("chatcmpl-x", final=True), h.engine("release", 1, "stop processing: n_tokens = 9"),
               h.stream_done())

        self.assertEqual(h.parser.totals["started"], 2)
        self.assertEqual(sorted(r["completion_tokens"] for r in completed(h.parser)), [2, 4])
        self.assertEqual(h.parser.active(), 0)


class ApiEdgeCaseTests(unittest.TestCase):
    def test_on_demand_model_load_delays_dispatch(self):
        h = Harness()
        h.feed(*h.received(), *h.received(), dt=10.0)  # Model is being loaded just in time.
        h.feed(h.running(), h.streaming(), h.running(), h.streaming())
        self.assertEqual(h.parser.totals["started"], 2)
        self.assertEqual(h.parser.active(), 2)

    def test_model_unloaded_aborts_open_requests(self):
        h = Harness()
        h.feed(*h.received(), *h.received(), h.running(), h.streaming(), h.running(), h.streaming(), dt=5.0)
        error = f"{ts('ERROR', MODEL)} Model unloaded by user or API request.. Error Data: n/a, Additional Data: n/a"
        h.feed(error, error)
        self.assertEqual(h.parser.active(), 0)
        self.assertEqual(h.parser.totals["failed"], 2)
        self.assertEqual(h.parser.totals["completed"], 0)
        self.assertEqual(h.parser.live()["latency_p95_s"], None)

    def test_request_rejected_without_log_entry_expires(self):
        h = Harness()
        h.feed(*h.received(stream=False), dt=1.0)
        self.assertEqual(h.parser.live()["active"], 1)
        h.clock.advance(h.parser.dispatch_timeout)
        self.assertEqual(h.parser.live()["active"], 0)
        self.assertEqual(h.parser.snapshot()["stale_requests"], 1)
        self.assertEqual(h.parser.totals["rejected"], 1)


class NonStreamingApiTests(unittest.TestCase):
    def test_usage_block_without_verbose_logging(self):
        h = Harness()
        h.feed(h.running(), dt=2.0)
        h.feed(*h.prediction(prompt_tokens=28, completion_tokens=150))
        h.feed(h.running())  # Next entry closes the prediction body.

        row = completed(h.parser)[0]
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"]), (28, 150))
        self.assertFalse(row["tokens_estimated"])
        self.assertAlmostEqual(row["duration_s"], 2.0, places=2)

    def test_usage_block_overrides_engine_estimate(self):
        h = Harness()
        h.feed(*h.received(stream=False), h.running(), h.engine("launch_slot_", 3, "processing task, is_child = 0"),
               dt=3.0)
        h.feed(h.engine("print_timing", 3, "n_gen =    100, tg =  33.00 t/s, tg_3s =  33.00 t/s"), dt=1.0)
        h.feed(h.engine("release", 3, "stop processing: n_tokens = 200, truncated = 0"))
        h.feed(*h.prediction(prompt_tokens=60, completion_tokens=131), h.running())

        row = completed(h.parser)[0]
        self.assertEqual(h.parser.totals["completed"], 1)
        self.assertEqual(row["completion_tokens"], 131)
        self.assertEqual(h.parser.totals["completion_tokens"], 131)


class EngineOnlyTests(unittest.TestCase):
    def test_chat_in_lm_studio_app(self):
        h = Harness()
        h.feed(h.engine("launch_slot_", 0, "processing task, is_child = 0"), dt=3.0)
        h.feed(h.engine("print_timing", 0, "n_gen =    100, tg =  31.55 t/s, tg_3s =  31.87 t/s"), dt=3.0)
        h.feed(h.engine("print_timing", 0, "n_gen =    207, tg =  33.52 t/s, tg_3s =  35.57 t/s"), dt=1.0)
        h.feed(h.engine("print_timing", 0, "prompt eval time =     453.73 ms /    28 tokens "
                                           "(   16.20 ms per token,    61.71 tokens per second)"))
        h.feed(h.engine("release", 0, "stop processing: n_tokens = 270, truncated = 0"))

        row = completed(h.parser)[0]
        self.assertEqual(row["source"], "app")
        self.assertEqual(row["prompt_tokens"], 28)
        self.assertEqual(row["completion_tokens"], 207 + round(33.52 * 1.0))  # Extrapolated to release.
        self.assertTrue(row["tokens_estimated"])
        self.assertAlmostEqual(row["duration_s"], 7.0, places=2)
        self.assertAlmostEqual(row["ttft_s"], 0.45, places=2)

    def test_exact_eval_line_from_older_engine(self):
        h = Harness()
        h.feed(h.engine("launch_slot_", 0, "processing task, is_child = 0"), dt=5.0)
        uptime_now = uptime(h.clock.t - h.engine_start)
        h.feed(h.engine("print_timing", 0, "prompt eval time =    4373.86 ms /   564 tokens "
                                           "(    7.76 ms per token,   128.95 tokens per second)"),
               f"{uptime_now} I slot print_timing: id  0 | task 0 |        eval time =   46404.26 ms /  1603 tokens "
               "(   28.97 ms per token,    34.52 tokens per second)",
               h.engine("release", 0, "stop processing: n_tokens = 2166, truncated = 0"))

        row = completed(h.parser)[0]
        self.assertEqual(row["completion_tokens"], 1603)
        self.assertFalse(row["tokens_estimated"])
        self.assertEqual(row["tokens_per_s"], 34.5)

    def test_two_engine_processes_with_same_task_id(self):
        h = Harness()
        other_engine = h.clock.t - 1000  # Second model loaded much earlier.
        h.feed(h.engine("launch_slot_", 0, "processing task, is_child = 0"),
               h.engine("launch_slot_", 0, "processing task, is_child = 0", engine_start=other_engine), dt=1.0)
        self.assertEqual(h.parser.active(), 2)
        h.feed(h.engine("release", 0, "stop processing: n_tokens = 10"), dt=1.0)
        h.feed(h.engine("release", 0, "stop processing: n_tokens = 10", engine_start=other_engine))
        self.assertEqual(h.parser.totals["completed"], 2)
        self.assertEqual(sorted(r["duration_s"] for r in completed(h.parser)), [1.0, 2.0])

    def test_model_name_primed_from_earlier_load_entry(self):
        h = Harness()
        # The model was loaded 5 s after the engine started, before the monitor began following the log.
        load_time = datetime.fromtimestamp(h.engine_start + 5).strftime("%Y-%m-%d %H:%M:%S")
        h.parser.prime_engines([f"[{load_time}][DEBUG] {uptime(5)} I srv    load_model: loading model "
                                "'/home/lab/.lmstudio/models/org/Model-GGUF/Model-Q4_K_M.gguf'"])
        h.feed(h.engine("launch_slot_", 0, "processing task, is_child = 0"), dt=1.0)
        h.feed(h.engine("release", 0, "stop processing: n_tokens = 10"))
        self.assertEqual(completed(h.parser)[0]["model"], "Model-Q4_K_M")  # File name only, no path.


class ReplayTests(unittest.TestCase):
    @staticmethod
    def at(epoch, level="INFO", tag=MODEL):
        return "[" + datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S") + f"][{level}][{tag}]"

    def test_restores_requests_within_window_only(self):
        now = float(int(time.time()))
        lines = []
        for start in (now - 3000, now - 600, now - 120):  # The first one is older than the window.
            lines += [f"{self.at(start)} Running chat completion on conversation with 1 messages.",
                      *(line.replace("[2026-10-21 15:03:37]", self.at(start + 4)[:21]) for line in Harness.prediction(10, 50)),
                      f"{self.at(start + 4)} Model generated tool calls: []"]
        parser = LogParser()
        samples = []
        parser.replay(lines, since=now - 1800, until=now, sample=lambda second: samples.append(parser.live()))

        rows = parser.live()["recent"]
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["completion_tokens"] for r in rows], [50, 50])
        self.assertAlmostEqual(rows[0]["duration_s"], 4.0, places=1)
        self.assertEqual(len(samples), 1801)  # One data point per second of the window.
        self.assertEqual(max(s["active"] for s in samples), 1)
        self.assertEqual(parser.totals["completed"], 0)  # Totals restart with live data.
        self.assertFalse(parser.use_log_time)

    def test_request_running_at_restart_continues_live(self):
        now = float(int(time.time()))
        clock = Clock(now)
        parser = LogParser(clock=clock)
        parser.replay([f"{self.at(now - 30)} Running chat completion on conversation with 1 messages.",
                       f"{self.at(now - 30)} Streaming response..."], since=now - 1800, until=now)
        self.assertEqual(parser.active(), 1)
        clock.advance(2)
        h = Harness()
        for line in [*h.packet("chatcmpl-r"), *h.packet("chatcmpl-r", final=True), h.stream_done()]:
            parser.feed(line)
        row = parser.live()["recent"][0]
        self.assertAlmostEqual(row["duration_s"], 32.0, places=1)
        self.assertEqual(parser.active(), 0)


class PrivacyTests(unittest.TestCase):
    def test_no_content_in_any_output(self):
        h = Harness()
        h.feed(*h.received(), h.running(), h.streaming(), h.engine("launch_slot_", 1, "processing task, is_child = 0"))
        h.feed(*h.packet("chatcmpl-p"), *h.packet("chatcmpl-p", final=True), h.stream_done())
        h.feed(h.running(), *h.prediction(1, 2), h.running())
        output = json.dumps([h.parser.live(), h.parser.snapshot(), h.parser.latency_points()], default=str)
        self.assertNotIn(SECRET, output)


if __name__ == "__main__":
    unittest.main()
