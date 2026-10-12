import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.helpers import ScriptedProvider  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path, approval_timeout=8, tool_timeout=5, max_steps=6)
    s.extra["resolver"] = lambda host: ["93.184.216.34"]  # pretend every host is public
    return s


@pytest.fixture
def make_client(settings):
    def _make(script, **overrides):
        for k, v in overrides.items():
            setattr(settings, k, v)
        prov = ScriptedProvider(script)
        client = TestClient(create_app(settings, prov))
        tok = client.post("/api/auth/register", json={"email": "a@example.com", "password": "password123"}).json()["token"]
        return client, {"Authorization": f"Bearer {tok}"}, prov
    return _make


def ctx_for(settings, mode="execute"):
    """Build a ToolContext backed by a real db + file store for unit-testing tools directly."""
    from app.db import Database
    from app.filestore import FileStore
    from app.tools.registry import ToolContext
    from app import auth
    db = Database(settings.data_dir / f"unit-{__import__('uuid').uuid4().hex[:8]}.db")
    uid = auth.register(db, f"u{id(db)}@example.com", "password123")
    now = __import__("time").time()
    db.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES('c1',?,?,?,?)", (uid, "t", now, now))
    db.run("INSERT INTO runs(id,conversation_id,user_id,mode,status,created_at,updated_at) VALUES('r1','c1',?,?,?,?,?)", (uid, mode, "running", now, now))
    return ToolContext(user_id=uid, run_id="r1", conversation_id="c1", mode=mode, db=db, settings=settings,
                       files=FileStore(db, settings.data_dir / "files"))
