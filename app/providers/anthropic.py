from __future__ import annotations

import time

import httpx

from .base import Capabilities, Provider, ProviderError, ProviderResponse

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class AnthropicProvider(Provider):
    name = "anthropic"
    capabilities = Capabilities(tool_calling=True, native_web_search=True, vision=True)

    def __init__(self, api_key: str | None, model: str, base_url: str = "https://api.anthropic.com",
                 max_attempts: int = 4, transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.max_attempts, self.transport, self.sleep = max_attempts, transport, sleep

    def configured(self):
        return (True, "") if self.api_key else (False, "ANTHROPIC_API_KEY is not set")

    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            blocks = [b for b in m["content"] if not (b.get("type") == "text" and not b.get("text", "").strip())]
            if blocks:
                out.append({"role": m["role"], "content": blocks})
        return out

    def complete(self, *, system, messages, tools, max_tokens, use_native_search=False, on_text=None) -> ProviderResponse:
        ok, why = self.configured()
        if not ok:
            raise ProviderError(why)
        tool_defs = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in tools]
        if use_native_search:
            tool_defs.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})
        body = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": self._clean(messages)}
        if tool_defs:
            body["tools"] = tool_defs
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        last = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                with httpx.Client(timeout=httpx.Timeout(120, connect=15), transport=self.transport) as c:
                    r = c.post(f"{self.base_url}/v1/messages", json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"network error: {type(e).__name__}"
                if attempt < self.max_attempts:
                    self.sleep(min(2 ** attempt, 20))
                    continue
                raise ProviderError(f"Could not reach the model provider ({last}).", retryable=True)
            if r.status_code == 200:
                d = r.json()
                if on_text:  # not streamed on the wire: the whole reply arrives at once
                    for b in d.get("content", []):
                        if b.get("type") == "text" and b.get("text"):
                            on_text(b["text"])
                return ProviderResponse(content=d.get("content", []), stop_reason=d.get("stop_reason") or "other",
                                        usage=d.get("usage", {}))
            try:
                detail = r.json().get("error", {}).get("message", r.text[:200])
            except Exception:
                detail = r.text[:200]
            last = f"HTTP {r.status_code}: {detail}"
            if r.status_code in RETRY_STATUS and attempt < self.max_attempts:
                ra = r.headers.get("retry-after")
                self.sleep(min(float(ra) if ra and ra.replace('.', '').isdigit() else 2 ** attempt, 30))
                continue
            hint = {401: " Check ANTHROPIC_API_KEY.", 404: " Check ASSISTANT_MODEL."}.get(r.status_code, "")
            raise ProviderError(f"Model provider error ({last}).{hint}", retryable=r.status_code in RETRY_STATUS)
        raise ProviderError(f"Model provider failed: {last}", retryable=True)
