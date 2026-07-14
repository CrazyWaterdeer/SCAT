from __future__ import annotations
import json
from collections.abc import Iterator
from typing import Any
from scat.agent.providers.base import Event, Stop, TextDelta, ToolUse, ToolUseStart


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-opus-4-8",
                 max_tokens: int = 4096, max_retries: int = 3) -> None:
        from anthropic import Anthropic
        self.model = model
        self.max_tokens = max_tokens
        # The SDK already does exponential backoff (+jitter) on 408/409/429/>=500 (529 overloaded
        # included) at request establishment, honoring x-should-retry — so we do NOT hand-roll retries,
        # only widen the count. Falsy api_key -> let the SDK resolve env/profile credentials.
        self._client = (Anthropic(api_key=api_key, max_retries=max_retries) if api_key
                        else Anthropic(max_retries=max_retries))

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str) -> Iterator[Event]:
        cached_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        cached_tools: list[dict[str, Any]] = []
        for i, t in enumerate(tools):
            entry = dict(t)
            if i == len(tools) - 1:
                entry["cache_control"] = {"type": "ephemeral"}
            cached_tools.append(entry)
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                  "system": cached_system, "messages": messages}
        if cached_tools:
            kwargs["tools"] = cached_tools
        with self._client.messages.stream(**kwargs) as stream:
            tool_inputs: dict[int, dict[str, Any]] = {}
            tool_meta: dict[int, dict[str, str]] = {}
            for event in stream:
                et = event.type
                if et == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_meta[event.index] = {"id": block.id, "name": block.name}
                        tool_inputs[event.index] = {"_buf": ""}
                        yield ToolUseStart(id=block.id, name=block.name)
                elif et == "content_block_delta":
                    dt = getattr(event.delta, "type", None)
                    if dt == "text_delta":
                        yield TextDelta(text=event.delta.text)
                    elif dt == "input_json_delta":
                        tool_inputs.setdefault(event.index, {"_buf": ""})
                        tool_inputs[event.index]["_buf"] += event.delta.partial_json
                elif et == "content_block_stop":
                    idx = event.index
                    if idx in tool_meta:
                        buf = tool_inputs[idx]["_buf"]
                        try:
                            parsed = json.loads(buf) if buf else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolUse(id=tool_meta[idx]["id"], name=tool_meta[idx]["name"], input=parsed)
            final = stream.get_final_message()
            u = final.usage
            usage = {k: getattr(u, k, 0) for k in
                     ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
            yield Stop(reason=final.stop_reason or "end_turn", usage=usage)
