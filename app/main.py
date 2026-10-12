"""FastAPI application: auth, conversations, runs, approvals, files, and the static UI."""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth
from .agent import load_history, run_agent
from .config import Settings
from .db import Database, new_id
from .filestore import UPLOAD_EXTS, FileStore
from .providers import Provider, build_provider
from .streaming import StreamHub
from .tools.builtin import build_registry
from .tools.registry import MODES, decide_approval

STATIC = Path(__file__).parent / "static"
ACTIVE = ("queued", "running", "awaiting_approval")


class Creds(BaseModel):
    email: str
    password: str = Field(max_length=200)


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    file_ids: list[str] = Field(default_factory=list, max_length=10)
    mode: str = "auto"


class ContinueIn(BaseModel):
    mode: str = "auto"


class PrefsIn(BaseModel):
    preferences: str = Field(max_length=4000)


class DecisionIn(BaseModel):
    approve: bool


def create_app(settings: Settings | None = None, provider: Provider | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.data_dir / "assistant.db")
    files = FileStore(db, settings.data_dir / "files")
    provider = provider or build_provider(settings)
    registry = build_registry()
    workers = ThreadPoolExecutor(max_workers=8, thread_name_prefix="run")
    hub = StreamHub()

    # Runs and pending approvals cannot survive a restart (the loop lives in memory): mark them honestly.
    db.run("UPDATE approvals SET status='expired' WHERE status='pending'")
    for r in db.all("SELECT id FROM runs WHERE status IN ('queued','running','awaiting_approval')"):
        db.add_event(r["id"], "interrupted", {"error": "The server restarted while this was running."})
        db.set_run_status(r["id"], "interrupted", "The server restarted while this was running. Send 'continue' to resume.")

    app = FastAPI(title="Personal Assistant", docs_url=None, redoc_url=None)
    app.state.db, app.state.settings, app.state.files, app.state.provider = db, settings, files, provider

    @app.middleware("http")
    async def headers(request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"
        return resp

    def current_user(authorization: str | None = Header(default=None)):
        token = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None
        u = auth.user_for_token(db, token)
        if not u:
            raise HTTPException(401, "Please sign in.")
        return u

    def own_conversation(cid: str, u) -> dict:
        c = db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (cid, u["id"]))
        if not c:
            raise HTTPException(404, "Conversation not found.")
        return dict(c)

    def own_run(rid: str, u) -> dict:
        r = db.one("SELECT * FROM runs WHERE id=? AND user_id=?", (rid, u["id"]))
        if not r:
            raise HTTPException(404, "Run not found.")
        return dict(r)

    def start_run(cid: str, u, mode: str) -> str:
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
        if db.one("SELECT 1 FROM runs WHERE conversation_id=? AND status IN ('queued','running','awaiting_approval')", (cid,)):
            raise HTTPException(409, "This conversation is still working on your previous request.")
        rid, now = new_id(), time.time()
        db.run("INSERT INTO runs(id,conversation_id,user_id,mode,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
               (rid, cid, u["id"], mode, "queued", now, now))
        hub.open(rid)
        workers.submit(run_agent, db=db, settings=settings, provider=provider, registry=registry, files=files, run_id=rid, hub=hub)
        return rid

    # ---- auth -------------------------------------------------------------------------------------
    @app.post("/api/auth/register")
    def register(c: Creds):
        if not settings.allow_registration:
            raise HTTPException(403, "Registration is disabled.")
        try:
            uid = auth.register(db, c.email, c.password)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"token": auth.create_session(db, uid, settings.session_days)}

    @app.post("/api/auth/login")
    def login(c: Creds):
        try:
            uid = auth.login(db, c.email, c.password)
        except PermissionError as e:
            raise HTTPException(401, str(e))
        return {"token": auth.create_session(db, uid, settings.session_days)}

    @app.post("/api/auth/logout")
    def logout(authorization: str | None = Header(default=None), u=Depends(current_user)):
        auth.end_session(db, authorization[7:])
        return {"ok": True}

    @app.get("/api/me")
    def me(u=Depends(current_user)):
        return {"email": u["email"], "preferences": u["preferences"]}

    @app.put("/api/me/preferences")
    def set_prefs(p: PrefsIn, u=Depends(current_user)):
        db.run("UPDATE users SET preferences=? WHERE id=?", (p.preferences, u["id"]))
        return {"ok": True}

    @app.get("/api/capabilities")
    def capabilities(u=Depends(current_user)):
        ok, why = provider.configured()
        return {"provider": provider.name, "model": getattr(provider, "model", ""), "provider_ready": ok, "provider_note": why,
                "native_web_search": bool(settings.native_web_search and provider.capabilities.native_web_search),
                "token_streaming": provider.capabilities.streaming,
                "tools": registry.describe(settings), "modes": list(MODES),
                "not_implemented": ["browser automation (Playwright)", "scheduled/recurring workflows", "calendar and Gmail integrations", "image/screenshot input"]}

    # ---- conversations ------------------------------------------------------------------------------
    @app.get("/api/conversations")
    def list_conversations(u=Depends(current_user)):
        return [dict(r) for r in db.all("SELECT id,title,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 200", (u["id"],))]

    @app.post("/api/conversations")
    def new_conversation(u=Depends(current_user)):
        cid, now = new_id(), time.time()
        db.run("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", (cid, u["id"], "New conversation", now, now))
        return {"id": cid}

    @app.delete("/api/conversations/{cid}")
    def delete_conversation(cid: str, u=Depends(current_user)):
        own_conversation(cid, u)
        if db.one("SELECT 1 FROM runs WHERE conversation_id=? AND status IN ('queued','running','awaiting_approval')", (cid,)):
            raise HTTPException(409, "Stop the running task first.")
        db.run("DELETE FROM conversations WHERE id=?", (cid,))
        return {"ok": True}

    @app.get("/api/conversations/{cid}")
    def get_conversation(cid: str, u=Depends(current_user)):
        conv = own_conversation(cid, u)
        msgs = [{"id": r["id"], "role": r["role"], "run_id": r["run_id"], "content": json.loads(r["content"])}
                for r in db.all("SELECT id,role,run_id,content FROM messages WHERE conversation_id=? ORDER BY id", (cid,))]
        runs = []
        for r in db.all("SELECT id,mode,status,error FROM runs WHERE conversation_id=? ORDER BY created_at", (cid,)):
            ev = [{"id": e["id"], "kind": e["kind"], **json.loads(e["data"])} for e in db.all("SELECT id,kind,data FROM run_events WHERE run_id=? ORDER BY id", (r["id"],))]
            runs.append({**dict(r), "events": ev})
        return {"id": cid, "title": conv["title"], "messages": msgs, "runs": runs}

    @app.post("/api/conversations/{cid}/messages")
    def send_message(cid: str, m: MessageIn, u=Depends(current_user)):
        conv = own_conversation(cid, u)
        text = m.text.strip()
        attached = []
        for fid in m.file_ids:
            f = files.get(u["id"], fid)
            if not f:
                raise HTTPException(400, f"Attached file {fid} was not found.")
            attached.append(f)
        if m.mode not in MODES:
            raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
        if db.one("SELECT 1 FROM runs WHERE conversation_id=? AND status IN ('queued','running','awaiting_approval')", (cid,)):
            raise HTTPException(409, "This conversation is still working on your previous request.")
        body = text + ("".join(f"\n\n[Attached file: {f['name']} (id={f['id']})]" for f in attached) if attached else "")
        db.add_message(cid, u["id"], "user", [{"type": "text", "text": body}])
        if conv["title"] == "New conversation":
            db.run("UPDATE conversations SET title=? WHERE id=?", (" ".join(text.split())[:60], cid))
        return {"run_id": start_run(cid, u, m.mode)}

    @app.post("/api/conversations/{cid}/continue")
    def continue_run(cid: str, body: ContinueIn, u=Depends(current_user)):
        """Retry/resume from the current history (after a failure, interruption or cancellation)."""
        own_conversation(cid, u)
        hist = load_history(db, cid)
        if not hist:
            raise HTTPException(400, "Nothing to continue.")
        last = hist[-1]
        if last["role"] == "assistant" and not any(b.get("type") == "tool_use" for b in last["content"]):
            raise HTTPException(400, "The last reply is already complete. Send a new message instead.")
        return {"run_id": start_run(cid, u, body.mode)}

    # ---- runs & approvals ----------------------------------------------------------------------------
    @app.get("/api/runs/{rid}")
    def get_run(rid: str, after: int = 0, u=Depends(current_user)):
        r = own_run(rid, u)
        ev = [{"id": e["id"], "kind": e["kind"], **json.loads(e["data"])} for e in db.all("SELECT id,kind,data FROM run_events WHERE run_id=? AND id>? ORDER BY id", (rid, after))]
        appr = [{"id": a["id"], "tool": a["tool_name"], "summary": a["summary"], "status": a["status"]} for a in db.all("SELECT id,tool_name,summary,status FROM approvals WHERE run_id=? AND status='pending'", (rid,))]
        return {"id": rid, "status": r["status"], "error": r["error"], "events": ev, "pending_approvals": appr}

    @app.get("/api/runs/{rid}/stream")
    async def stream_run(rid: str, after: int = 0, u=Depends(current_user)):
        """Server-sent events with live text for a run. Purely a view: the stored messages are the record,
        and the polling endpoint keeps working if this connection drops."""
        own_run(rid, u)

        async def gen():
            idx, last = max(after, 0), time.time()
            while True:
                evs, done, known = hub.read(rid, idx)
                for e in evs:
                    idx += 1
                    last = time.time()
                    yield f"data: {json.dumps(e)}\n\n"
                if done or not known:  # finished, or unknown to this process (e.g. after a restart)
                    yield 'data: {"type": "done"}\n\n'
                    return
                if time.time() - last > 15:
                    last = time.time()
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.05)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/runs/{rid}/cancel")
    def cancel_run(rid: str, u=Depends(current_user)):
        r = own_run(rid, u)
        if r["status"] not in ACTIVE:
            return {"ok": True, "status": r["status"]}
        db.run("UPDATE runs SET cancel_requested=1 WHERE id=?", (rid,))
        return {"ok": True, "status": "cancelling"}

    @app.post("/api/approvals/{aid}/decision")
    def decision(aid: str, d: DecisionIn, u=Depends(current_user)):
        if not decide_approval(db, u["id"], aid, d.approve):
            raise HTTPException(409, "This approval is no longer pending.")
        return {"ok": True, "status": "approved" if d.approve else "denied"}

    # ---- files ------------------------------------------------------------------------------------------
    @app.get("/api/files")
    def list_files(u=Depends(current_user)):
        return files.list(u["id"])

    @app.post("/api/files")
    async def upload(file: UploadFile = File(...), u=Depends(current_user)):
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
        if not data:
            raise HTTPException(400, "The file is empty.")
        try:
            f = files.save(u["id"], file.filename or "file", data, "upload", allowed=UPLOAD_EXTS)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {k: f[k] for k in ("id", "name", "size", "origin")}

    @app.get("/api/files/{fid}/download")
    def download(fid: str, u=Depends(current_user)):
        f = files.get(u["id"], fid)
        if not f or not Path(f["path"]).exists():
            raise HTTPException(404, "File not found.")
        return FileResponse(f["path"], filename=f["name"], media_type="application/octet-stream")

    @app.delete("/api/files/{fid}")
    def remove_file(fid: str, u=Depends(current_user)):
        if not files.delete(u["id"], fid):
            raise HTTPException(404, "File not found.")
        return {"ok": True}

    @app.get("/api/health")
    def health():
        return {"ok": True}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def app_factory() -> FastAPI:  # for `uvicorn app.main:app_factory --factory`
    return create_app()
