import json

import httpx
import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider, from_openai_message, to_openai_messages

MSGS = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_anthropic_retries_rate_limit_then_succeeds():
    seen = []
    def handler(req):
        seen.append(json.loads(req.content))
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "0"})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}})
    p = AnthropicProvider("k", "m", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    r = p.complete(system="s", messages=MSGS, tools=[], max_tokens=10)
    assert r.content[0]["text"] == "ok" and len(seen) == 2


def test_anthropic_sends_native_search_tool_and_hides_key_in_errors():
    captured = {}
    def handler(req):
        captured["body"] = json.loads(req.content); captured["key"] = req.headers["x-api-key"]
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})
    p = AnthropicProvider("sk-secret", "m", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(ProviderError) as e:
        p.complete(system="s", messages=MSGS, tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}], max_tokens=10, use_native_search=True)
    assert "sk-secret" not in str(e.value) and "ANTHROPIC_API_KEY" in str(e.value)
    assert captured["key"] == "sk-secret"
    assert [t["name"] for t in captured["body"]["tools"]] == ["t", "web_search"]


def test_anthropic_gives_up_after_max_attempts():
    p = AnthropicProvider("k", "m", max_attempts=3, transport=httpx.MockTransport(lambda r: httpx.Response(529, json={})), sleep=lambda s: None)
    with pytest.raises(ProviderError) as e:
        p.complete(system="s", messages=MSGS, tools=[], max_tokens=10)
    assert e.value.retryable


def test_anthropic_requires_key():
    with pytest.raises(ProviderError):
        AnthropicProvider(None, "m").complete(system="s", messages=MSGS, tools=[], max_tokens=1)


def test_openai_message_conversion_roundtrip():
    hist = [
        {"role": "user", "content": [{"type": "text", "text": "do it"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "c1", "name": "calculator", "input": {"expression": "1+1"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "1+1 = 2"}]},
    ]
    out = to_openai_messages("sys", hist)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "1+1 = 2"}
    blocks = from_openai_message(out[2])
    assert blocks[1] == {"type": "tool_use", "id": "c1", "name": "calculator", "input": {"expression": "1+1"}}


def test_openai_provider_parses_tool_call_response():
    body = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
        {"id": "c9", "type": "function", "function": {"name": "calculator", "arguments": "{\"expression\": \"2*3\"}"}}]}}]}
    p = OpenAICompatProvider("k", "m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    r = p.complete(system="s", messages=MSGS, tools=[{"name": "calculator", "description": "d", "input_schema": {"type": "object"}}], max_tokens=5)
    assert r.stop_reason == "tool_use" and r.content[0]["input"] == {"expression": "2*3"}
