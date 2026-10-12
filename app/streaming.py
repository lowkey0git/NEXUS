"""In-memory fan-out of live text for a run. Only a *view* of what the model is producing:
the authoritative result is always the stored message, so losing this on restart loses nothing durable."""
from __future__ import annotations

import threading
import time

RETAIN_AFTER_DONE = 120  # seconds a finished run's events stay available to late subscribers


class StreamHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, dict] = {}

    def open(self, run_id: str):
        with self._lock:
            self._runs.setdefault(run_id, {"events": [], "done": False, "ts": time.time()})
            self._cleanup()

    def publish(self, run_id: str, event: dict):
        with self._lock:
            r = self._runs.setdefault(run_id, {"events": [], "done": False, "ts": time.time()})
            r["events"].append(event)

    def finish(self, run_id: str):
        with self._lock:
            r = self._runs.setdefault(run_id, {"events": [], "done": False, "ts": time.time()})
            r["done"], r["ts"] = True, time.time()

    def read(self, run_id: str, after: int) -> tuple[list[dict], bool, bool]:
        """(new events, finished, known)"""
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                return [], False, False
            return r["events"][after:], r["done"], True

    def _cleanup(self):
        now = time.time()
        for k in [k for k, r in self._runs.items() if r["done"] and now - r["ts"] > RETAIN_AFTER_DONE]:
            del self._runs[k]
