from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolUseStart:
    id: str
    name: str


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Stop:
    reason: str
    usage: dict[str, Any] = field(default_factory=dict)


Event = TextDelta | ToolUseStart | ToolUse | Stop


class Provider(Protocol):
    name: str
    model: str
    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str) -> Iterator[Event]: ...


_CURRENT_PROVIDER: Provider | None = None


def set_current_provider(p: Provider | None) -> None:
    global _CURRENT_PROVIDER
    _CURRENT_PROVIDER = p


def get_current_provider() -> Provider:
    if _CURRENT_PROVIDER is None:
        raise RuntimeError(
            "No active LLM provider. Agent tools require an AgentRunner "
            "turn to be in progress (the runner registers its provider "
            "before tool dispatch)."
        )
    return _CURRENT_PROVIDER
