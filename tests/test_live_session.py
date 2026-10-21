"""Unit tests for live_session (recording sessions of live tests)."""
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime  # noqa: E402

from live_session import SessionManager, build_from_log, list_sessions, summarise  # noqa: E402


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

    def test_requests_finishing_out_of_order_or_in_the_same_second_are_all_kept(self):
        manager = SessionManager(self.out)
        manager.start("Order")
        now = float(int(time.time())) + 5
        for end in (now, now, now - 0.5, now + 1):  # Same second twice, one booked after a later one.
            manager.current.record_request(request(end), "completed")
        self.assertEqual(len(rows(manager.current.folder, "requests.csv")), 4)

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

    def test_build_session_for_a_past_time_range(self):
        base = float(int(time.time()) - 3600)

        def at(epoch, level="INFO"):
            return "[" + datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S") + f"][{level}][google/gemma-4-e4b]"

        lines = []
        for offset in (60, 600, 1200, 3000):  # The last one lies outside the chosen range.
            t = base + offset
            lines += [f"{at(t)} Running chat completion on conversation with 1 messages.",
                      f"{at(t + 3)} Generated prediction: {{", '  "usage": {', '    "prompt_tokens": 10,',
                      '    "completion_tokens": 50', "  }", "}", f"{at(t + 3)} Model generated tool calls: []"]
        log = self.out / "server.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        metrics = self.out / "monitoring"
        metrics.mkdir()
        with open(metrics / "lmstudio_metrics_test.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "max_active", "gpu_util_avg", "vram_pct_max", "ram_pct_max"])
            writer.writeheader()
            for offset in (0, 600, 3000):
                writer.writerow({"time": datetime.fromtimestamp(base + offset).isoformat(timespec="seconds"),
                                 "max_active": 1, "gpu_util_avg": 70, "vram_pct_max": 55, "ram_pct_max": 60})

        info = build_from_log(metrics / "sessions", "Seminar afterwards", base, base + 1800, [log], metrics)
        folder = metrics / "sessions" / info["folder"]
        self.assertEqual(len(rows(folder, "requests.csv")), 3)
        self.assertEqual(len(rows(folder, "intervals.csv")), 2)
        self.assertEqual(rows(folder, "hardware.csv")[0]["gpu_util"], "70")
        self.assertTrue((folder / "report.html").is_file())
        listed = list_sessions(metrics / "sessions")
        self.assertEqual(listed[0]["completed"], 3)
        self.assertTrue(listed[0]["retroactive"])

    def test_summary_of_empty_session(self):
        manager = SessionManager(self.out)
        manager.start("Empty")
        folder = manager.current.folder
        manager.end()
        self.assertEqual(summarise(folder)["requests"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
