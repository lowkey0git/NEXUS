"""Provider-agnostic model interface.

Internal message format = Anthropic-style content blocks:
  user:      [{"type":"text","text":...}] or [{"type":"tool_result","tool_use_id":...,"content":...,"is_error":bool}]
  assistant: [{"type":"text",...}, {"type":"tool_use","id":...,"name":...,"input":{...}}, (provider-native blocks)]
Adapters translate to/from their wire format. Provider-native blocks (e.g. server-side web search) are
stored verbatim and only replayed to the provider that produced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ProviderResponse:
    content: list[dict]
    stop_reason: str  # end_turn | tool_use | max_tokens | pause_turn | other
    usage: dict = field(default_factory=dict)


@dataclass
class Capabilities:
    tool_calling: bool = True
    native_web_search: bool = False
    vision: bool = False  # image inputs are not wired into the UI in this release
    streaming: bool = False  # delivers text incrementally through on_text


class Provider:
    name = "base"
    capabilities = Capabilities()

    def configured(self) -> tuple[bool, str]:
        return True, ""

    def complete(self, *, system: str, messages: list[dict], tools: list[dict],
                 max_tokens: int, use_native_search: bool = False, on_text=None) -> ProviderResponse:
        """on_text(str) is called with text as it is produced (streaming providers) or once with the
        whole text (non-streaming providers). The returned ProviderResponse is always the complete turn."""
        raise NotImplementedError
