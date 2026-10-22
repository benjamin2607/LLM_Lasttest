"""Unit tests for the direct LM Studio load test (lmstudio_user.py and --target lmstudio)."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_loadtest  # noqa: E402
import lmstudio_user  # noqa: E402
from lmstudio_monitor import LoadTestControl  # noqa: E402


def sse(*chunks):
    return [f"data: {json.dumps(c)}".encode() for c in chunks] + [b"", b"data: [DONE]"]


class StreamTests(unittest.TestCase):
    def test_collects_answer_and_reasoning(self):
        lines = sse({"choices": [{"delta": {"reasoning_content": "Hmm. "}}]},
                    {"choices": [{"delta": {"content": "Hallo"}}]},
                    {"choices": [{"delta": {"content": " Welt"}}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
        self.assertEqual(lmstudio_user.read_stream(lines), ("Hmm. Hallo Welt", None))

    def test_error_in_stream(self):
        lines = sse({"choices": [{"delta": {"content": "Teil"}}]}, {"error": {"message": "Model unloaded"}})
        self.assertEqual(lmstudio_user.read_stream(lines), ("Teil", "Model unloaded"))

    def test_ignores_comments_and_broken_lines(self):
        lines = [b": keep-alive", b"data: {not json", *sse({"choices": [{"delta": {"content": "ok"}}]})]
        self.assertEqual(lmstudio_user.read_stream(lines), ("ok", None))

    def test_non_streamed_message(self):
        result = {"choices": [{"message": {"content": "Antwort"}}]}
        self.assertEqual(lmstudio_user.read_message(result), ("Antwort", None))
        self.assertEqual(lmstudio_user.read_message({"error": "busy"}), ("", "busy"))


class ContextTests(unittest.TestCase):
    def test_context_block_has_about_the_requested_size(self):
        text = lmstudio_user.context_block(1000)
        self.assertAlmostEqual(len(text) / lmstudio_user.CHARS_PER_TOKEN, 1000, delta=60)

    def test_prompts_need_no_documents(self):
        prompts = json.loads(lmstudio_user.DEFAULT_PROMPTS.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(prompts), 10)
        self.assertFalse([p for p in prompts if "Dokument" in p or "Unterlagen" in p])


class TargetTests(unittest.TestCase):
    def test_locust_statistics_of_both_targets_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("/thread/chat [query]", "/v1/chat/completions"):
                path = Path(tmp) / "stats.csv"
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["Type", "Name", "Request Count", "Failure Count", "Requests/s"])
                    writer.writeheader()
                    writer.writerow({"Type": "POST", "Name": name, "Request Count": 7, "Failure Count": 1, "Requests/s": 0.5})
                    writer.writerow({"Type": "REJECTED", "Name": "http-503", "Request Count": 1, "Failure Count": 0})
                totals, chat, refusals = auto_loadtest.read_locust_stats(path)
                self.assertEqual(chat["Name"], name)
                self.assertEqual(totals["Request Count"], 7)  # Rejection entries are not counted twice.
                self.assertEqual(refusals, {"REJECTED http-503": 1})

    def test_dashboard_launch_validates_lmstudio_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = LoadTestControl(tmp, "http://localhost:3001", None)
            self.assertEqual(control.launch({"target": "other", "users": "1"})[0], False)
            ok, message = control.launch({"target": "lmstudio", "users": "1,2", "model": "bad model; rm -rf"})
            self.assertFalse(ok)
            self.assertIn("model", message)
            ok, message = control.launch({"target": "lmstudio", "users": "1", "context_tokens": "-5"})
            self.assertFalse(ok)
            self.assertIn("context_tokens", message)
            self.assertIsNone(control.process)  # Nothing was started.


if __name__ == "__main__":
    unittest.main()
