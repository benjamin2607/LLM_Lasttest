"""
Complete final statistics for headless Locust runs, imported by the Locust user files.

Locust rewrites its ``<prefix>_stats.csv`` and ``<prefix>_failures.csv`` once per second
and does not write them again at shutdown. Requests that finish during the stop grace
period (--stop-timeout) - typically the last request of every simulated user - are
therefore missing from those files. This module writes the same tables once more when
Locust quits, after all users have finished, to ``<prefix>_stats_final.csv`` and
``<prefix>_failures_final.csv``. auto_loadtest.py reads these files when they exist.
"""
import csv

from locust import events
from locust.stats import PERCENTILES_TO_REPORT, StatsCSV


@events.quitting.add_listener
def write_final_stats(environment, **_kwargs):
    prefix = getattr(environment.parsed_options, "csv_prefix", None) if environment.parsed_options else None
    if not prefix:
        return
    tables = StatsCSV(environment, PERCENTILES_TO_REPORT)
    for name, write in (("stats", tables.requests_csv), ("failures", tables.failures_csv)):
        with open(f"{prefix}_{name}_final.csv", "w", newline="", encoding="utf-8") as f:
            write(csv.writer(f))
