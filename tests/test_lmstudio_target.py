"""Unit tests for the direct LM Studio load test (lmstudio_user.py and --target lmstudio)."""
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_loadtest  # noqa: E402
import lmstudio_user  # noqa: E402
from lmstudio_models import ModelControl, clean_request_settings, request_arguments  # noqa: E402
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
        class Models:
            def loaded_model(self):
                return "google/gemma-4-e4b"

            def request_settings(self, model):
                return {}

        with tempfile.TemporaryDirectory() as tmp:
            control = LoadTestControl(tmp, "http://localhost:3001", None)
            self.assertEqual(control.launch({"target": "other", "users": "1"})[0], False)
            ok, message = control.launch({"target": "lmstudio", "users": "1"})
            self.assertFalse(ok)
            self.assertIn("no model is loaded", message)  # Without the model card nothing is started.
            control.models = Models()
            ok, message = control.launch({"target": "lmstudio", "users": "1", "context_tokens": "-5"})
            self.assertFalse(ok)
            self.assertIn("context_tokens", message)
            ok, message = control.launch({"target": "lmstudio", "users": "1", "name": "x" * 121})
            self.assertFalse(ok)
            self.assertIn("name", message)
            self.assertIsNone(control.process)  # Nothing was started.


class ModelControlTests(unittest.TestCase):
    MODELS = {"models": [
        {"type": "llm", "key": "google/gemma-4-e4b", "display_name": "Gemma 4 E4B", "max_context_length": 131072,
         "loaded_instances": [{"id": "google/gemma-4-e4b", "config": {"context_length": 8192, "parallel": 4}}],
         "capabilities": {"reasoning": {"allowed_options": ["off", "on"], "default": "on"}}},
        {"type": "llm", "key": "google/gemma-4-e4b", "display_name": "Gemma 4 E4B", "loaded_instances": []},
        {"type": "embedding", "key": "nomic-embed", "loaded_instances": []}]}

    def control(self, tmp):
        control = ModelControl("http://localhost:1234", None, Path(tmp) / "model_configs.json")
        control.calls = []

        def api(method, path, body=None, timeout=10):
            control.calls.append((method, path, body))
            return self.MODELS if method == "GET" else {"load_config": body}
        control._api = api
        return control

    def test_models_are_merged_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = self.control(tmp).models()
            self.assertEqual([m["key"] for m in models], ["google/gemma-4-e4b"])  # Embeddings left out, variants merged.
            self.assertEqual(len(models[0]["instances"]), 1)

    def test_settings_are_validated(self):
        self.assertEqual(clean_request_settings({"temperature": "0.5", "reasoning": "off", "top_k": ""})[0],
                         {"temperature": 0.5, "reasoning": "off"})
        self.assertIn("temperature", clean_request_settings({"temperature": "5"})[1])
        self.assertIn("thinking", clean_request_settings({"reasoning": "maybe"})[1])
        self.assertEqual(request_arguments({"presence_penalty": -0.5, "system_prompt": "Hi there"}),
                         ["--presence-penalty=-0.5", "--system-prompt=Hi there"])

    def test_reload_unloads_then_loads_and_becomes_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.control(tmp)
            ok, _ = control.apply({"action": "adjust", "model": "google/gemma-4-e4b",
                                   "load": {"context_length": "16384", "flash_attention": "true"},
                                   "request": {"temperature": "0.3"}})
            self.assertTrue(ok)
            for _ in range(100):  # Loading runs in a background thread.
                if control.progress.get("phase") != "working":
                    break
                time.sleep(0.02)
            posts = [(p, b) for m, p, b in control.calls if m == "POST"]
            self.assertEqual(posts[0], ("/api/v1/models/unload", {"instance_id": "google/gemma-4-e4b"}))
            self.assertEqual(posts[1][1]["context_length"], 16384)
            self.assertIs(posts[1][1]["flash_attention"], True)
            self.assertEqual(control.request_settings("google/gemma-4-e4b"), {"temperature": 0.3})
            stored = json.loads((Path(tmp) / "model_configs.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["active"]["model"], "google/gemma-4-e4b")

    def test_context_length_above_the_maximum_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, message = self.control(tmp).apply({"action": "adjust", "model": "google/gemma-4-e4b",
                                                   "load": {"context_length": 200000}})
            self.assertFalse(ok)
            self.assertIn("maximum", message)

    def test_no_changes_while_a_load_test_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.control(tmp)
            control.busy = lambda: True
            self.assertFalse(control.apply({"action": "defaults", "model": "google/gemma-4-e4b"})[0])
            self.assertFalse(control.unload("google/gemma-4-e4b")[0])

    def test_configurations_are_saved_and_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = self.control(tmp)
            self.assertTrue(control.save_config({"name": " Gemma  8k ", "model": "google/gemma-4-e4b",
                                                 "load": {"context_length": 8192}, "request": {"reasoning": "off"}})[0])
            self.assertFalse(control.save_config({"name": "", "load": {}})[0])
            reopened = self.control(tmp)
            self.assertEqual(reopened.store["configs"][0]["name"], "Gemma 8k")
            self.assertTrue(reopened.delete_config("Gemma 8k")[0])
            self.assertFalse(reopened.delete_config("Gemma 8k")[0])


class WarmupTests(unittest.TestCase):
    class Args:
        target, host, passthrough = "lmstudio", "http://localhost:1234", ["--model", "m"]

    def test_waits_for_the_answer_and_survives_errors(self):
        original = auto_loadtest.warmup_lmstudio
        try:
            auto_loadtest.warmup_lmstudio = lambda host, model: time.sleep(0.2) or model
            took, stopped = auto_loadtest.warm_up(self.Args, "run")
            self.assertGreaterEqual(took, 0.2)
            self.assertFalse(stopped)

            def fail(host, model):
                raise RuntimeError("no model")
            auto_loadtest.warmup_lmstudio = fail
            self.assertFalse(auto_loadtest.warm_up(self.Args, "run")[1])  # The test goes on; the console warns.
        finally:
            auto_loadtest.warmup_lmstudio = original

    def test_stop_from_the_dashboard_ends_the_warm_up(self):
        original, original_stop = auto_loadtest.warmup_lmstudio, auto_loadtest.stop_requested
        try:
            auto_loadtest.warmup_lmstudio = lambda host, model: time.sleep(5)
            auto_loadtest.stop_requested = lambda: True
            took, stopped = auto_loadtest.warm_up(self.Args, "run")
            self.assertTrue(stopped)
            self.assertLess(took, 2)
        finally:
            auto_loadtest.warmup_lmstudio, auto_loadtest.stop_requested = original, original_stop


class ReportTests(unittest.TestCase):
    def test_print_support_is_added_once_to_old_reports(self):
        from report_charts import PRINT_MARKER, html_page, with_print_support
        old = "<html><head><title>x</title><style></style></head><body></body></html>"
        page = with_print_support(old)
        self.assertEqual(page.count(f'id="{PRINT_MARKER}"'), 1)
        self.assertEqual(with_print_support(page), page)
        self.assertEqual(html_page("T", "<p>x</p>").count(f'id="{PRINT_MARKER}"'), 1)

    def test_name_replaces_the_folder_in_the_title(self):
        class Args:
            name = "Gemma 12B"
        self.assertEqual(auto_loadtest.report_title(Args, Path("20261022_120000")), "Gemma 12B")
        Args.name = ""
        self.assertEqual(auto_loadtest.report_title(Args, Path("20261022_120000")), "Load test 20261022_120000")


if __name__ == "__main__":
    unittest.main()
