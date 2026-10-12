"""Per-user file storage. Files live under data/files/<user_id>/<file_id><ext>; names are metadata only,
so user-supplied names can never influence a filesystem path."""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from .db import Database, new_id

UPLOAD_EXTS = {".txt", ".md", ".csv", ".json", ".log", ".pdf", ".docx", ".xlsx"}
GENERATED_EXTS = {".txt", ".md", ".csv", ".json", ".xlsx"}


def safe_name(name: str) -> str:
    name = Path(name or "file").name  # drop any directory components
    name = re.sub(r"[^\w.\- ]+", "_", name).strip(" .") or "file"
    return name[:120]


class FileStore:
    def __init__(self, db: Database, root: Path):
        self.db, self.root = db, Path(root)

    def save(self, user_id: str, name: str, data: bytes, origin: str, conversation_id: str | None = None,
             allowed: set[str] | None = None) -> dict:
        name = safe_name(name)
        ext = Path(name).suffix.lower()
        if allowed is not None and ext not in allowed:
            raise ValueError(f"Unsupported file type '{ext or 'none'}'. Supported: {', '.join(sorted(allowed))}")
        fid = new_id()
        d = self.root / user_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{fid}{ext}"
        p.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        self.db.run("INSERT INTO files(id,user_id,conversation_id,name,path,size,sha256,origin,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (fid, user_id, conversation_id, name, str(p), len(data), sha, origin, time.time()))
        return self.get(user_id, fid)

    def get(self, user_id: str, file_id: str) -> dict | None:
        r = self.db.one("SELECT * FROM files WHERE id=? AND user_id=?", (file_id, user_id))
        return dict(r) if r else None

    def find(self, user_id: str, ref: str) -> dict | None:
        """Resolve by id first, then by exact name (newest wins)."""
        f = self.get(user_id, ref)
        if f:
            return f
        r = self.db.one("SELECT * FROM files WHERE user_id=? AND name=? ORDER BY created_at DESC LIMIT 1", (user_id, ref))
        return dict(r) if r else None

    def list(self, user_id: str) -> list[dict]:
        return [dict(r) for r in self.db.all(
            "SELECT id,name,size,origin,conversation_id,created_at FROM files WHERE user_id=? ORDER BY created_at DESC", (user_id,))]

    def read_bytes(self, f: dict) -> bytes:
        return Path(f["path"]).read_bytes()

    def delete(self, user_id: str, file_id: str) -> bool:
        f = self.get(user_id, file_id)
        if not f:
            return False
        Path(f["path"]).unlink(missing_ok=True)
        self.db.run("DELETE FROM files WHERE id=? AND user_id=?", (file_id, user_id))
        return True

    def verify(self, f: dict) -> tuple[bool, str]:
        """Re-read from disk and compare against the recorded hash: evidence the file really exists."""
        p = Path(f["path"])
        if not p.exists():
            return False, "file missing on disk"
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != f["sha256"] or len(data) != f["size"]:
            return False, "content does not match recorded hash"
        return True, f"{len(data)} bytes, sha256 {f['sha256'][:12]}"
