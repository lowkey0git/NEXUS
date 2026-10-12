"""SQLite persistence. One short-lived connection per operation; WAL mode for concurrent readers.

SQLite is a deliberate choice for the first release (zero-config, durable, single file).
All access goes through this module, so moving to PostgreSQL later means replacing it.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw_salt BLOB NOT NULL, pw_hash BLOB NOT NULL,
  preferences TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL, run_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS run_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  kind TEXT NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, user_id TEXT NOT NULL,
  tool_use_id TEXT NOT NULL, tool_name TEXT NOT NULL, args TEXT NOT NULL, args_hash TEXT NOT NULL,
  summary TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL, decided_at REAL);
CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  conversation_id TEXT, name TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
  origin TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_evt_run ON run_events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_runs_conv ON runs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_files_user ON files(user_id);
"""


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=15, isolation_level=None)  # autocommit
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
        finally:
            c.close()

    def one(self, sql: str, args: tuple = ()):
        with self.conn() as c:
            return c.execute(sql, args).fetchone()

    def all(self, sql: str, args: tuple = ()):
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    def run(self, sql: str, args: tuple = ()) -> int:
        with self.conn() as c:
            return c.execute(sql, args).rowcount

    def add_message(self, conversation_id: str, user_id: str, role: str, content: list[dict], run_id: str | None = None):
        now = time.time()
        with self.conn() as c:
            c.execute(
                "INSERT INTO messages(conversation_id,user_id,run_id,role,content,created_at) VALUES(?,?,?,?,?,?)",
                (conversation_id, user_id, run_id, role, json.dumps(content), now))
            c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))

    def add_event(self, run_id: str, kind: str, data: dict):
        self.run("INSERT INTO run_events(run_id,kind,data,created_at) VALUES(?,?,?,?)",
                 (run_id, kind, json.dumps(data), time.time()))

    def set_run_status(self, run_id: str, status: str, error: str | None = None):
        self.run("UPDATE runs SET status=?, error=?, updated_at=? WHERE id=?", (status, error, time.time(), run_id))
