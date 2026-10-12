"""Adapter for OpenAI-compatible /chat/completions endpoints (Groq, OpenAI, OpenRouter, vLLM, Ollama, ...).

Streams over server-sent events when the caller passes on_text. Tool-call fragments are reassembled
by index; the returned ProviderResponse is identical to the non-streaming path.
No native web search: the assistant falls back to its own web_fetch tool."""
from __future__ import annotations

import json
import time
import uuid

import httpx

from .base import Capabilities, Provider, ProviderError, ProviderResponse

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
FINISH = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            results = [b for b in m["content"] if b.get("type") == "tool_result"]
            texts = [b["text"] for b in m["content"] if b.get("type") == "text" and b.get("text", "").strip()]
            for r in results:
                c = r["content"] if isinstance(r["content"], str) else json.dumps(r["content"])
                out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": c})
            if texts:
                out.append({"role": "user", "content": "\n\n".join(texts)})
        else:
            text = "\n".join(b["text"] for b in m["content"] if b.get("type") == "text" and b.get("text"))
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                     for b in m["content"] if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            if text or calls:
                out.append(msg)
    return out


def _parse_args(raw: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"_unparseable_arguments": raw}
    except json.JSONDecodeError:
        return {"_unparseable_arguments": raw}


def from_openai_message(msg: dict) -> list[dict]:
    blocks = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                       "input": _parse_args(tc["function"].get("arguments"))})
    return blocks


class OpenAICompatProvider(Provider):
    name = "openai_compat"
    capabilities = Capabilities(tool_calling=True, native_web_search=False, vision=False, streaming=True)

    def __init__(self, api_key, model, base_url="https://api.openai.com/v1", max_attempts=4, transport=None,
                 sleep=time.sleep, name="openai_compat", key_env="OPENAI_API_KEY"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.max_attempts, self.transport, self.sleep = max_attempts, transport, sleep
        self.name, self.key_env = name, key_env

    def configured(self):
        return (True, "") if self.api_key else (False, f"{self.key_env} is not set")

    # ---- one HTTP attempt ------------------------------------------------------------------------
    def _read_stream(self, resp: httpx.Response, on_text) -> ProviderResponse:
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        finish, usage = None, {}
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue  # ignore a malformed keep-alive/comment frame
            if chunk.get("error"):
                err = chunk["error"]
                raise ProviderError(f"Model provider error mid-reply: {(err.get('message') if isinstance(err, dict) else err)}"[:300])
            usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    text_parts.append(d["content"])
                    on_text(d["content"])
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name") and not slot["name"]:
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        content = []
        text = "".join(text_parts)
        if text:
            content.append({"type": "text", "text": text})
        for i in sorted(calls):
            c = calls[i]
            if c["name"]:
                content.append({"type": "tool_use", "id": c["id"] or "call_" + uuid.uuid4().hex[:16],
                                "name": c["name"], "input": _parse_args(c["args"])})
        if any(b["type"] == "tool_use" for b in content):
            finish = "tool_calls"
        if finish is None and not content:
            raise ProviderError("The model stream ended without a reply.", retryable=True)
        return ProviderResponse(content=content, stop_reason=FINISH.get(finish, "other"), usage=usage)

    def complete(self, *, system, messages, tools, max_tokens, use_native_search=False, on_text=None) -> ProviderResponse:
        ok, why = self.configured()
        if not ok:
            raise ProviderError(why)
        body = {"model": self.model, "messages": to_openai_messages(system, messages), "max_tokens": max_tokens}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["input_schema"]}} for t in tools]
        streaming = on_text is not None
        if streaming:
            body["stream"] = True
        headers = {"Authorization": f"Bearer {self.api_key}"}
        emitted = {"any": False}

        def emit(t):
            emitted["any"] = True
            on_text(t)

        for attempt in range(1, self.max_attempts + 1):
            try:
                with httpx.Client(timeout=httpx.Timeout(120, connect=15), transport=self.transport) as c:
                    with c.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as r:
                        if r.status_code == 200:
                            if streaming:
                                return self._read_stream(r, emit)
                            r.read()
                            d = r.json()
                            choice = d["choices"][0]
                            return ProviderResponse(content=from_openai_message(choice["message"]),
                                                    stop_reason=FINISH.get(choice.get("finish_reason"), "other"),
                                                    usage=d.get("usage", {}))
                        r.read()
                        status, rtext, retry_after = r.status_code, r.text, r.headers.get("retry-after")
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if emitted["any"]:  # part of a reply was already shown: do not silently replay it
                    raise ProviderError(f"The connection dropped mid-reply ({type(e).__name__}). Try again.", retryable=False)
                if attempt < self.max_attempts:
                    self.sleep(min(2 ** attempt, 20))
                    continue
                raise ProviderError(f"Could not reach the model provider ({type(e).__name__}).", retryable=True)
            malformed_tool_call = status == 400 and "tool_use_failed" in rtext  # Groq: model emitted a bad tool call
            if (status in RETRY_STATUS or malformed_tool_call) and attempt < self.max_attempts:
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 2 ** attempt
                self.sleep(min(wait, 1 if malformed_tool_call else 30))
                continue
            hint = ""
            if malformed_tool_call:
                hint = " The model produced a malformed tool call; try again or choose a larger model."
            elif status == 401:
                hint = f" Check {self.key_env}."
            elif status == 404:
                hint = " Check ASSISTANT_MODEL."
            elif status == 413 or (status == 429 and "tokens" in rtext.lower()):
                hint = " The request may exceed the provider's token limits; lower ASSISTANT_TOOL_OUTPUT_CHARS or wait a minute."
            raise ProviderError(f"Model provider error (HTTP {status}): {rtext[:200]}.{hint}",
                                retryable=status in RETRY_STATUS)
        raise ProviderError("Model provider failed.", retryable=True)
