"""Password hashing (scrypt), opaque session tokens (stored hashed), and a small login throttle."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from collections import defaultdict, deque

from .db import Database, new_id

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_attempts: dict[str, deque] = defaultdict(deque)
MAX_FAILS, WINDOW = 8, 300


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def _throttle_check(key: str):
    q = _attempts[key]
    now = time.time()
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= MAX_FAILS:
        raise PermissionError("Too many failed attempts. Try again in a few minutes.")


def register(db: Database, email: str, password: str) -> str:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if db.one("SELECT 1 FROM users WHERE email=?", (email,)):
        raise ValueError("An account with this email already exists.")
    salt = secrets.token_bytes(16)
    uid = new_id()
    db.run("INSERT INTO users(id,email,pw_salt,pw_hash,created_at) VALUES(?,?,?,?,?)",
           (uid, email, salt, _scrypt(password, salt), time.time()))
    return uid


def login(db: Database, email: str, password: str) -> str:
    email = email.strip().lower()
    _throttle_check(email)
    row = db.one("SELECT id,pw_salt,pw_hash FROM users WHERE email=?", (email,))
    ok = bool(row) and hmac.compare_digest(_scrypt(password, row["pw_salt"]), row["pw_hash"])
    if not ok:
        _attempts[email].append(time.time())
        raise PermissionError("Incorrect email or password.")
    _attempts.pop(email, None)
    return row["id"]


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db: Database, user_id: str, days: int) -> str:
    token = secrets.token_urlsafe(32)
    db.run("INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)",
           (_h(token), user_id, time.time() + days * 86400))
    return token


def user_for_token(db: Database, token: str | None):
    if not token:
        return None
    row = db.one("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?",
                 (_h(token), time.time()))
    return row


def end_session(db: Database, token: str):
    db.run("DELETE FROM sessions WHERE token_hash=?", (_h(token),))
