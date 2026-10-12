"""Streaming tests. The wire format used here is the documented OpenAI-style SSE that Groq also emits;
these are simulated streams, not recordings from the live service."""
import json

import httpx
import pytest

from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider
from tests.helpers import text, use, wait_for
from tests.test_api import DONE, start

MSGS = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
TOOLS = [{"name": "calculator", "description": "d", "input_schema": {"type": "object"}}]


def sse(*chunks, done=True):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def delta(content=None, tool_calls=None, finish=None):
    d = {}
    if content is not None:
        d["content"] = content
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def prov(handler, **kw):
    return OpenAICompatProvider("k", "m", base_url="https://api.groq.com/openai/v1", name="groq", key_env="GROQ_API_KEY",
                                transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


def ok(body):
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def test_text_deltas_stream_in_order_and_final_matches():
    body = sse(delta("Hel"), delta("lo, "), delta("world"), delta(None, finish="stop"))
    got = []
    r = prov(lambda req: ok(body)).complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=got.append)
    assert got == ["Hel", "lo, ", "world"]
    assert r.content == [{"type": "text", "text": "Hello, world"}] and r.stop_reason == "end_turn"


def test_request_asks_for_stream_only_when_callback_given():
    seen = []
    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]})
    p = prov(handler)
    p.complete(system="s", messages=MSGS, tools=[], max_tokens=5)
    assert "stream" not in seen[0]
    seen.clear()
    p2 = prov(lambda req: (seen.append(json.loads(req.content)), ok(sse(delta("x", finish="stop"))))[1])
    p2.complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=lambda t: None)
    assert seen[0]["stream"] is True


def test_tool_call_arguments_split_across_chunks_are_reassembled():
    body = sse(
        delta("Let me compute. "),
        delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "calculator", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": "{\"expres"}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": "sion\": \"2*"}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": "21\"}"}}]),
        delta(None, finish="tool_calls"))
    got = []
    r = prov(lambda req: ok(body)).complete(system="s", messages=MSGS, tools=TOOLS, max_tokens=5, on_text=got.append)
    assert "".join(got) == "Let me compute. "
    assert r.stop_reason == "tool_use"
    assert r.content[1] == {"type": "tool_use", "id": "call_1", "name": "calculator", "input": {"expression": "2*21"}}


def test_parallel_tool_calls_interleaved_and_missing_ids():
    body = sse(
        delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "calculator", "arguments": "{\"expression\":"}},
                          {"index": 1, "function": {"name": "list_files", "arguments": "{"}}]),
        delta(tool_calls=[{"index": 1, "function": {"arguments": "}"}}, {"index": 0, "function": {"arguments": " \"1+1\"}"}}]),
        delta(None, finish="tool_calls"))
    r = prov(lambda req: ok(body)).complete(system="s", messages=MSGS, tools=TOOLS, max_tokens=5, on_text=lambda t: None)
    names = [b["name"] for b in r.content]
    assert names == ["calculator", "list_files"]
    assert r.content[0]["input"] == {"expression": "1+1"} and r.content[1]["input"] == {}
    assert r.content[0]["id"] == "a" and r.content[1]["id"].startswith("call_")  # generated when the server sent none


def test_garbled_tool_arguments_do_not_crash():
    body = sse(delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "calculator", "arguments": "{\"expr"}}]), delta(None, finish="tool_calls"))
    r = prov(lambda req: ok(body)).complete(system="s", messages=MSGS, tools=TOOLS, max_tokens=5, on_text=lambda t: None)
    assert "_unparseable_arguments" in r.content[0]["input"]  # schema validation will reject it with a clear error


def test_ignores_comments_and_malformed_frames_and_reads_groq_usage():
    raw = b": keepalive\n\ndata: {not json}\n\n" + sse(delta("ok"), {"choices": [], "x_groq": {"usage": {"total_tokens": 7}}}, delta(None, finish="stop"))
    r = prov(lambda req: ok(raw)).complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=lambda t: None)
    assert r.content[0]["text"] == "ok" and r.usage == {"total_tokens": 7}


class Dropping(httpx.SyncByteStream):
    def __iter__(self):
        yield sse(delta("partial "), done=False)
        raise httpx.ReadError("connection reset")


def test_drop_after_text_shown_is_not_silently_replayed():
    calls = []
    def handler(req):
        calls.append(1)
        return httpx.Response(200, stream=Dropping(), headers={"content-type": "text/event-stream"})
    got = []
    with pytest.raises(ProviderError) as e:
        prov(handler).complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=got.append)
    assert got == ["partial "] and len(calls) == 1 and not e.value.retryable and "mid-reply" in str(e.value)


def test_drop_before_any_text_is_retried():
    n = []
    class DropFirst(httpx.SyncByteStream):
        def __iter__(self):
            raise httpx.ReadError("reset")
            yield b""
    def handler(req):
        n.append(1)
        return httpx.Response(200, stream=DropFirst()) if len(n) == 1 else ok(sse(delta("fine", finish="stop")))
    r = prov(handler).complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=lambda t: None)
    assert r.content[0]["text"] == "fine" and len(n) == 2


def test_rate_limit_then_success_and_hints():
    n = []
    def handler(req):
        n.append(1)
        if len(n) == 1:
            return httpx.Response(429, json={"error": {"message": "Rate limit reached for tokens per minute"}}, headers={"retry-after": "0"})
        return ok(sse(delta("ok", finish="stop")))
    assert prov(handler).complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=lambda t: None).content[0]["text"] == "ok"
    always = prov(lambda r: httpx.Response(429, json={"error": {"message": "Rate limit reached for tokens per minute"}}), max_attempts=2)
    with pytest.raises(ProviderError) as e:
        always.complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=lambda t: None)
    assert "TOOL_OUTPUT_CHARS" in str(e.value) and e.value.retryable


def test_malformed_tool_call_400_is_retried_once_then_explained():
    n = []
    def handler(req):
        n.append(1)
        return httpx.Response(400, json={"error": {"message": "Failed to call a function", "code": "tool_use_failed"}})
    with pytest.raises(ProviderError) as e:
        prov(handler, max_attempts=3).complete(system="s", messages=MSGS, tools=TOOLS, max_tokens=5, on_text=lambda t: None)
    assert len(n) == 3 and "malformed tool call" in str(e.value)
    m = []
    def handler2(req):
        m.append(1)
        return httpx.Response(400, json={"error": {"code": "tool_use_failed"}}) if len(m) == 1 else ok(sse(delta("ok", finish="stop")))
    assert prov(handler2).complete(system="s", messages=MSGS, tools=TOOLS, max_tokens=5, on_text=lambda t: None).content[0]["text"] == "ok"


def test_missing_key_names_the_right_variable_and_no_secret_leak():
    p = OpenAICompatProvider(None, "m", name="groq", key_env="GROQ_API_KEY")
    with pytest.raises(ProviderError) as e:
        p.complete(system="s", messages=MSGS, tools=[], max_tokens=1)
    assert "GROQ_API_KEY" in str(e.value)
    bad = OpenAICompatProvider("gsk_secret", "m", transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "invalid"}})))
    with pytest.raises(ProviderError) as e2:
        bad.complete(system="s", messages=MSGS, tools=[], max_tokens=1)
    assert "gsk_secret" not in str(e2.value)


def test_anthropic_adapter_emits_whole_text_once():
    from app.providers.anthropic import AnthropicProvider
    p = AnthropicProvider("k", "m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": "whole"}], "stop_reason": "end_turn"})))
    got = []
    p.complete(system="s", messages=MSGS, tools=[], max_tokens=5, on_text=got.append)
    assert got == ["whole"] and p.capabilities.streaming is False


# ---- end to end through the HTTP stream endpoint -----------------------------------------------------
def read_sse(client, h, rid, timeout=15):
    events = []
    with client.stream("GET", f"/api/runs/{rid}/stream", headers=h) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
                if events[-1]["type"] == "done":
                    break
    return events


def test_stream_endpoint_delivers_live_text_then_step_end_then_done(make_client):
    answer = "First, bring a valid ID. Then your account number and the transaction reference."
    client, h, prov = make_client([text(answer)])
    cid, rid = start(client, h, "bank prep?")
    ev = read_sse(client, h, rid)
    types = [e["type"] for e in ev]
    assert types[-1] == "done" and "step_end" in types
    assert types.count("delta") > 5  # arrived in pieces, not as one blob
    assert "".join(e["text"] for e in ev if e["type"] == "delta") == answer
    stored = client.get(f"/api/conversations/{cid}", headers=h).json()["messages"][-1]["content"][0]["text"]
    assert stored == answer  # the live text and the stored record agree


def test_stream_with_tool_step_has_a_step_end_per_model_call(make_client):
    client, h, prov = make_client([[text("Checking. "), use("calculator", expression="2+2")], text("It is 4.")])
    _, rid = start(client, h, "2+2?", mode="execute")
    ev = read_sse(client, h, rid)
    types = [e["type"] for e in ev]
    assert types.count("step_end") == 2 and types[-1] == "done"
    assert "".join(e["text"] for e in ev if e["type"] == "delta") == "Checking. It is 4."


def test_late_subscriber_and_failed_run_still_terminate(make_client):
    from app.providers.base import ProviderError
    client, h, prov = make_client([ProviderError("boom")])
    _, rid = start(client, h, "x")
    wait_for(client, h, rid, DONE)
    assert read_sse(client, h, rid)[-1] == {"type": "done"}  # connecting after the end does not hang


def test_stream_requires_auth_and_ownership(make_client):
    client, h, _ = make_client([text("hi")])
    _, rid = start(client, h, "x")
    wait_for(client, h, rid, DONE)
    assert client.get(f"/api/runs/{rid}/stream").status_code == 401
    h2 = {"Authorization": "Bearer " + client.post("/api/auth/register", json={"email": "z@example.com", "password": "password123"}).json()["token"]}
    assert client.get(f"/api/runs/{rid}/stream", headers=h2).status_code == 404


def test_unknown_run_after_restart_ends_cleanly(make_client, settings):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from tests.helpers import ScriptedProvider
    client, h, _ = make_client([text("hi")])
    _, rid = start(client, h, "x")
    wait_for(client, h, rid, DONE)
    fresh = TestClient(create_app(settings, ScriptedProvider([])))  # new process: empty hub
    assert read_sse(fresh, h, rid) == [{"type": "done"}]


def test_capabilities_report_streaming(make_client):
    client, h, _ = make_client([])
    assert client.get("/api/capabilities", headers=h).json()["token_streaming"] is True
