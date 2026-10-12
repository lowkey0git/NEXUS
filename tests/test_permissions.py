import time
from app.tools.builtin import build_registry
from app.tools.registry import execute, request_approval, decide_approval
from tests.conftest import ctx_for

R = build_registry()


def test_advise_mode_blocks_write_tools(settings):
    ctx = ctx_for(settings, mode="advise")
    res = execute(R, ctx, "write_text_file", "t", {"name": "a.md", "content": "x"}, 5)
    assert res.is_error and "advice-only" in res.content
    assert ctx.files.find(ctx.user_id, "a.md") is None
    assert "write_text_file" not in [t.name for t in R.for_mode("advise", settings)]
    assert "calculator" in [t.name for t in R.for_mode("advise", settings)]


def test_invalid_args_and_unknown_tool(settings):
    ctx = ctx_for(settings)
    assert "Invalid arguments" in execute(R, ctx, "calculator", "t", {"expr": "1"}, 5).content
    assert execute(R, ctx, "rm_rf", "t", {}, 5).is_error
    assert execute(R, ctx, "calculator", "t", "1+1", 5).is_error


def test_consequential_tool_refuses_without_approval(settings):
    ctx = ctx_for(settings)
    f = ctx.files.save(ctx.user_id, "keep.txt", b"important", "upload")
    res = execute(R, ctx, "delete_file", "tu1", {"file_id": f["id"]}, 5)
    assert res.is_error and "NOT executed" in res.content
    assert ctx.files.get(ctx.user_id, f["id"]) is not None


def test_approval_is_single_use_and_bound_to_exact_args(settings):
    ctx = ctx_for(settings)
    a = ctx.files.save(ctx.user_id, "a.txt", b"a", "upload")
    b = ctx.files.save(ctx.user_id, "b.txt", b"b", "upload")
    tool = R.tools["delete_file"]
    aid = request_approval(ctx.db, ctx, tool, "tu1", {"file_id": a["id"]})
    assert decide_approval(ctx.db, ctx.user_id, aid, True)
    # approval for file a must not authorise deleting file b
    assert execute(R, ctx, "delete_file", "tu1", {"file_id": b["id"]}, 5).is_error
    assert ctx.files.get(ctx.user_id, b["id"])
    # and not under a different tool_use id
    assert execute(R, ctx, "delete_file", "tu2", {"file_id": a["id"]}, 5).is_error
    # exact match works once...
    ok = execute(R, ctx, "delete_file", "tu1", {"file_id": a["id"]}, 5)
    assert not ok.is_error and "Verified" in ok.content and ctx.files.get(ctx.user_id, a["id"]) is None
    # ...and never again
    c = ctx.files.save(ctx.user_id, "c.txt", b"c", "upload")
    assert execute(R, ctx, "delete_file", "tu1", {"file_id": a["id"]}, 5).is_error


def test_denied_approval_cannot_be_used_and_other_user_cannot_decide(settings):
    ctx = ctx_for(settings)
    f = ctx.files.save(ctx.user_id, "a.txt", b"a", "upload")
    aid = request_approval(ctx.db, ctx, R.tools["delete_file"], "tu1", {"file_id": f["id"]})
    assert not decide_approval(ctx.db, "someone-else", aid, True)
    assert decide_approval(ctx.db, ctx.user_id, aid, False)
    assert execute(R, ctx, "delete_file", "tu1", {"file_id": f["id"]}, 5).is_error
    assert not decide_approval(ctx.db, ctx.user_id, aid, True)  # already decided


def test_email_unavailable_without_smtp_and_sent_after_approval(settings, monkeypatch):
    ctx = ctx_for(settings)
    args = {"to": ["bob@example.com"], "subject": "Hi", "body": "Hello"}
    assert "send_email" not in [t.name for t in R.for_mode("execute", settings)]
    assert "unavailable" in execute(R, ctx, "send_email", "tu", args, 5).content
    settings.smtp_host, settings.smtp_from = "smtp.test", "me@example.com"
    sent = []

    class FakeSMTP:  # stands in for a real mail server; we only verify our call pattern
        def __init__(self, host, port, timeout): sent.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def starttls(self): sent.append("starttls")
        def login(self, u, p): sent.append("login")
        def send_message(self, msg): sent.append(("msg", msg["To"], msg["Subject"])); return {}
    monkeypatch.setattr("app.tools.builtin.smtplib.SMTP", FakeSMTP)
    assert execute(R, ctx, "send_email", "tu", args, 5).is_error and not sent  # no approval -> nothing sent
    aid = request_approval(ctx.db, ctx, R.tools["send_email"], "tu", args)
    decide_approval(ctx.db, ctx.user_id, aid, True)
    res = execute(R, ctx, "send_email", "tu", args, 5)
    assert not res.is_error and ("msg", "bob@example.com", "Hi") in sent and "does not prove" in res.content
    bad = {"to": ["not-an-email"], "subject": "x", "body": "y"}
    aid = request_approval(ctx.db, ctx, R.tools["send_email"], "tu9", bad)
    decide_approval(ctx.db, ctx.user_id, aid, True)
    assert execute(R, ctx, "send_email", "tu9", bad, 5).is_error


def test_tool_timeout_reported(settings, monkeypatch):
    from app.tools.registry import Tool
    ctx = ctx_for(settings)
    R2 = build_registry()
    R2.register(Tool("slow", "slow", {"type": "object", "properties": {}}, "read", lambda c, a: (time.sleep(2), "late")[1]))
    res = execute(R2, ctx, "slow", "t", {}, 0.3)
    assert res.is_error and "timed out" in res.content
