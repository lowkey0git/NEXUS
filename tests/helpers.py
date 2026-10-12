"""Test doubles. ScriptedProvider replays a fixed list of model responses; it exists ONLY in tests and is
never importable from the app, so no production code path can return fabricated 'successful' results."""
from __future__ import annotations

import time
import uuid

from app.providers.base import Capabilities, Provider, ProviderError, ProviderResponse


def text(t):
    return {"type": "text", "text": t}


def use(tool, /, **inp):
    return {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:12], "name": tool, "input": inp}


class ScriptedProvider(Provider):
    name = "scripted"
    capabilities = Capabilities(native_web_search=False, streaming=True)

    def __init__(self, script, chunk=4, delay=0.002):
        self.script, self.calls, self.chunk, self.delay = list(script), [], chunk, delay

    def complete(self, *, system, messages, tools, max_tokens, use_native_search=False, on_text=None):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": [t["name"] for t in tools]})
        if not self.script:
            raise ProviderError("script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        content = item if isinstance(item, list) else [item] if isinstance(item, dict) else [text(item)]
        if on_text:
            for b in content:
                if b["type"] == "text":
                    for i in range(0, len(b["text"]), self.chunk):
                        on_text(b["text"][i:i + self.chunk])
                        time.sleep(self.delay)
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return ProviderResponse(content=content, stop_reason=stop)


def wait_for(client, headers, run_id, statuses, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/runs/{run_id}", headers=headers).json()
        if r["status"] in statuses:
            return r
        time.sleep(0.05)
    raise AssertionError(f"run never reached {statuses}; last={r['status']} events={r['events'][-3:]}")
