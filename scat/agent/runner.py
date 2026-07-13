from __future__ import annotations
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from scat.agent.providers.base import Provider, Stop, TextDelta, ToolUse, ToolUseStart


@dataclass
class ToolResult:
    tool_use_id: str
    name: str
    output: Any
    is_error: bool = False


@dataclass
class TurnDone:
    stop_reason: str
    total_usage: dict[str, int] = field(default_factory=dict)


RunEvent = TextDelta | ToolUseStart | ToolUse | ToolResult | TurnDone

_MAX_TOOL_RESULT_CHARS = 6000
_MAX_STRING_CHARS = 1200
_MAX_LIST_ITEMS = 8
_MAX_DICT_ITEMS = 40


def _stringify_output(o: Any) -> str:
    try:
        return json.dumps(o, default=str)
    except Exception:
        return str(o)


def _truncate_text(t: str, n: int = _MAX_STRING_CHARS) -> str:
    return t if len(t) <= n else t[:n] + f"... [truncated {len(t) - n} chars]"


def _compact_value(v: Any, depth: int = 0) -> Any:
    if isinstance(v, str):
        return _truncate_text(v)
    if isinstance(v, dict):
        items = list(v.items())[:_MAX_DICT_ITEMS]
        return {k: _compact_value(val, depth + 1) for k, val in items}
    if isinstance(v, (list, tuple)):
        return [_compact_value(x, depth + 1) for x in list(v)[:_MAX_LIST_ITEMS]]
    return v


def _compact_tool_result(tool_name: str, output: Any) -> str:
    text = _stringify_output(_compact_value(output))
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text
    fb = {"tool": tool_name, "result_summary": _truncate_text(text, _MAX_TOOL_RESULT_CHARS),
          "note": "Tool result was compacted to keep the conversation within context."}
    if isinstance(output, dict):
        for k in ("path", "output_dir", "csv_path", "report_path", "n_deposits", "n_images",
                  "n_normal", "n_rod", "n_artifact", "warnings"):
            if k in output:
                fb[k] = _compact_value(output[k])
    return _stringify_output(fb)


class AgentRunner:
    def __init__(self, provider: Provider, system_prompt: str, max_loops: int = 40, tool_caller: Any | None = None):
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_loops = max_loops
        self.messages: list[dict[str, Any]] = []
        self._cancelled = False
        self._tool_caller = tool_caller

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self.messages = []
        self._cancelled = False

    def turn(self, user_text: str) -> Iterator[RunEvent]:
        from scat.tools import call_tool, tools_for_anthropic
        from scat.agent.providers.base import set_current_provider
        self.messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        tools_spec = tools_for_anthropic()
        total_usage: dict[str, int] = {}
        set_current_provider(self.provider)
        try:
            for _ in range(self.max_loops):
                if self._cancelled:
                    yield TurnDone("cancelled", total_usage); self._cancelled = False; return
                assistant_blocks: list[dict[str, Any]] = []
                current_text = ""
                stop_reason = "end_turn"
                try:
                    for event in self.provider.stream(self.messages, tools_spec, self.system_prompt):
                        if self._cancelled:
                            break
                        if isinstance(event, TextDelta):
                            current_text += event.text; yield event
                        elif isinstance(event, ToolUseStart):
                            if current_text:
                                assistant_blocks.append({"type": "text", "text": current_text}); current_text = ""
                            yield event
                        elif isinstance(event, ToolUse):
                            assistant_blocks.append({"type": "tool_use", "id": event.id,
                                                     "name": event.name, "input": event.input})
                            yield event
                        elif isinstance(event, Stop):
                            stop_reason = event.reason
                            for k, v in (event.usage or {}).items():
                                total_usage[k] = total_usage.get(k, 0) + int(v)
                except Exception as exc:
                    # Keep history clean: drop the user turn we just appended.
                    if self.messages and self.messages[-1]["role"] == "user":
                        self.messages.pop()
                    yield TextDelta(f"\n[error: {exc}]\n")
                    yield TurnDone("error", total_usage); return
                if self._cancelled:
                    yield TurnDone("cancelled", total_usage); self._cancelled = False; return
                if current_text:
                    assistant_blocks.append({"type": "text", "text": current_text})
                if assistant_blocks:
                    self.messages.append({"role": "assistant", "content": assistant_blocks})
                if stop_reason != "tool_use":
                    yield TurnDone(stop_reason, total_usage); return
                tool_result_blocks: list[dict[str, Any]] = []
                for block in assistant_blocks:
                    if block.get("type") != "tool_use":
                        continue
                    if self._cancelled:
                        break
                    caller = self._tool_caller or call_tool
                    try:
                        result = caller(block["name"], **block.get("input", {}))
                        tool_result_blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                                                   "content": _compact_tool_result(block["name"], result)})
                        yield ToolResult(block["id"], block["name"], result)
                    except Exception as e:
                        tool_result_blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                                                   "content": f"ERROR: {e}", "is_error": True})
                        yield ToolResult(block["id"], block["name"], str(e), is_error=True)
                done = {b["tool_use_id"] for b in tool_result_blocks}
                for block in assistant_blocks:
                    if block.get("type") != "tool_use" or block["id"] in done:
                        continue
                    tool_result_blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                                               "content": "ERROR: cancelled before execution", "is_error": True})
                    yield ToolResult(block["id"], block["name"], "cancelled before execution", is_error=True)
                if tool_result_blocks:
                    self.messages.append({"role": "user", "content": tool_result_blocks})
            yield TurnDone("max_loops", total_usage)
        finally:
            set_current_provider(None)
