"""Unit tests for live_session (recording sessions of live tests)."""
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_session import SessionManager, summarise  # noqa: E402


def request(end, duration=4.0, ttft=1.0, tokens=120, model="google/gemma-4-e4b"):
    return {"start": end - duration, "first": end - duration + ttft, "end": end, "completion_tokens": tokens,
            "prompt_tokens": 30, "model": model, "api": True, "streamed": True}


def rows(folder, name):
    with open(Path(folder) / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_only_requests_of_the_session(self):
        manager = SessionManager(self.out)
        info = manager.start("Seminar KI")
        session = manager.current
        session.record_request(request(info["started"] - 10), "completed")  # Before the session.
        session.record_request(request(time.time() + 1), "completed")
        session.record_request({"start": time.time(), "first": None, "end": time.time() + 2, "api": True}, "failed")
        data = rows(session.folder, "requests.csv")
        self.assertEqual([r["status"] for r in data], ["completed", "failed"])
        self.assertEqual(data[0]["duration_s"], "4.0")
        self.assertEqual(data[1]["duration_s"], "")  # No response time for unanswered requests.

    def test_resume_after_restart_writes_no_duplicates(self):
        manager = SessionManager(self.out)
        manager.start("Live test")
        first = request(time.time() + 1)
        manager.current.record_request(first, "completed")
        folder = manager.current.folder

        resumed = SessionManager(self.out)  # Monitor restarted: the log replay reports the request again.
        self.assertIsNotNone(resumed.current)
        resumed.current.record_request(first, "completed")
        resumed.current.record_request(request(time.time() + 5), "completed")
        self.assertEqual(len(rows(folder, "requests.csv")), 2)
        self.assertEqual(resumed.current.counts["completed"], 2)

    def test_end_writes_report_and_summary(self):
        manager = SessionManager(self.out, {"gpu_name": "Test GPU"})
        manager.start("Report test")
        session = manager.current
        now = time.time()
        for i, duration in enumerate([2.0, 4.0, 6.0, 8.0, 30.0]):
            session.record_request(request(now + 1 + i, duration=duration), "completed")
        session.record_interval({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "max_active": 3,
                                 "gen_tokens_per_s_total": 42.0})
        session.record_hardware({"gpu_util": 80, "vram_pct": 60, "ram_pct": 50}, now=now + 10)
        folder = session.folder
        info = manager.end()

        self.assertFalse(info["active"])
        self.assertTrue((folder / "report.html").is_file())
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["requests"]["completed"], 5)
        self.assertEqual(summary["response_time_s"]["median"], 6.0)
        self.assertEqual(summary["peak_concurrency"], 3)
        self.assertEqual(summary["hardware"]["gpu_util_pct"]["max"], 80)
        page = (folder / "report.html").read_text(encoding="utf-8")
        self.assertIn("Report test", page)
        self.assertIn("<svg", page)
        self.assertIsNone(SessionManager(self.out).current)  # Ended sessions are not resumed.

    def test_no_content_is_stored(self):
        manager = SessionManager(self.out)
        manager.start("Privacy")
        req = request(time.time() + 1)
        req["prompt"] = "CONFIDENTIAL-STUDENT-TEXT"  # Must never reach the files.
        manager.current.record_request(req, "completed")
        folder = manager.current.folder
        manager.end()
        for path in folder.iterdir():
            self.assertNotIn("CONFIDENTIAL", path.read_text(encoding="utf-8"), path.name)

    def test_restart_within_the_same_second_gets_its_own_folder(self):
        manager = SessionManager(self.out)
        first = manager.start("Same")["folder"]
        second = manager.start("Same")["folder"]  # Ends the first one (like "Clear").
        self.assertNotEqual(first, second)
        self.assertTrue((self.out / "sessions" / first / "report.html").is_file())

    def test_summary_of_empty_session(self):
        manager = SessionManager(self.out)
        manager.start("Empty")
        folder = manager.current.folder
        manager.end()
        self.assertEqual(summarise(folder)["requests"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
