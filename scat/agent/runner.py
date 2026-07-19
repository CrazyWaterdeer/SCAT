from __future__ import annotations
import json
import re
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
# High enough that a folder's filename list (which the agent needs to infer groups)
# survives the primary compaction pass; overall size is still bounded by
# _MAX_TOOL_RESULT_CHARS + the fallback below.
_MAX_LIST_ITEMS = 500
_MAX_DICT_ITEMS = 60


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


_KEEP_RECENT = 6
_STUB_OVER = 800
# Match only genuine context-overflow errors (NOT any 400 — a bad schema / tool_use error is a 400 too,
# and compaction can't fix those). Deliberately does not match a bare "context".
_CTX_RE = re.compile(
    r"prompt is too long|input length and .*exceed|exceed\w*\s+context|"
    r"context (?:window|length|limit)|too many tokens|maximum context", re.I)


def _is_context_limit_error(exc: Exception) -> bool:
    """Provider-agnostic: a context-window overflow (so the runner needs no `anthropic` import)."""
    status = getattr(exc, "status_code", None)
    if status is not None and status != 400:      # if the exc exposes a status, a non-400 is not this
        return False
    return bool(_CTX_RE.search(str(exc)))


def _block_ids(blocks: list, key: str, btype: str) -> list[str] | None:
    """Ordered ids of blocks of `btype` (via `key`), or None if a duplicate id is present."""
    ids = [b.get(key) for b in blocks if isinstance(b, dict) and b.get("type") == btype]
    return ids if len(ids) == len(set(ids)) else None


def _soft_rewrite(msg: dict) -> dict:
    """Return a COPY with only lossy-but-structure-safe shrinks: stub oversized tool_result content,
    truncate text blocks. tool_use.input and every id/name/type/is_error field are preserved so the
    tool_use<->tool_result pairing the API requires is never disturbed. Never mutates the input."""
    content = msg.get("content")
    if not isinstance(content, list):
        return dict(msg)
    new_blocks = []
    for b in content:
        if not isinstance(b, dict):
            new_blocks.append(b); continue
        bt = b.get("type")
        if bt == "tool_result" and isinstance(b.get("content"), str) and len(b["content"]) > _STUB_OVER:
            nb = dict(b); nb["content"] = "[earlier result elided to save context]"; new_blocks.append(nb)
        elif bt == "text" and isinstance(b.get("text"), str):
            nb = dict(b); nb["text"] = _truncate_text(b["text"]); new_blocks.append(nb)
        else:
            new_blocks.append(b)   # tool_use (input kept as-is) and anything else: unchanged
    out = dict(msg); out["content"] = new_blocks
    return out


def _compact_history(messages: list[dict]) -> list[dict]:
    """Aggressive one-shot compaction used to recover from a context-limit overflow. Keeps messages[0]
    (the task) and the last _KEEP_RECENT messages (the live work) verbatim; soft-rewrites other
    survivors; and drops whole matched adjacent (assistant tool_use, user tool_result) round-pairs from
    the unprotected middle. Pairing-safe: drops BOTH sides of a pair, only on exact non-empty dup-free
    id-set equality, never touching the protected boundary."""
    n = len(messages)
    protected = {0, *range(max(0, n - _KEEP_RECENT), n)}
    drop: set[int] = set()
    i = 1
    while i < n - 1:
        a, u = messages[i], messages[i + 1]
        if (i not in protected and (i + 1) not in protected
                and isinstance(a, dict) and a.get("role") == "assistant"
                and isinstance(u, dict) and u.get("role") == "user"
                and isinstance(a.get("content"), list) and isinstance(u.get("content"), list)
                and all(isinstance(b, dict) and b.get("type") in ("text", "tool_use") for b in a["content"])
                and u["content"] and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in u["content"])):
            use_ids = _block_ids(a["content"], "id", "tool_use")
            res_ids = _block_ids(u["content"], "tool_use_id", "tool_result")
            if use_ids and res_ids and set(use_ids) == set(res_ids):
                drop.add(i); drop.add(i + 1); i += 2; continue
        i += 1
    out = []
    for idx, m in enumerate(messages):
        if idx in drop:
            continue
        out.append(m if idx in protected else _soft_rewrite(m))
    return out


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
        from scat.progress import AnalysisCancelled
        # Snapshot BEFORE appending: on a fatal error we restore this exact list, which drops the whole
        # failed turn (and any in-turn compaction) and is always a valid history — unlike popping the
        # "last user" message, which orphans a preceding tool_use after a tool round.
        pre_turn = list(self.messages)
        self.messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        tools_spec = tools_for_anthropic()
        total_usage: dict[str, int] = {}
        set_current_provider(self.provider)
        try:
            for _ in range(self.max_loops):
                if self._cancelled:
                    yield TurnDone("cancelled", total_usage); self._cancelled = False; return
                retried = False
                while True:  # one retry, only for a pre-event context-limit overflow (compact + re-stream)
                    assistant_blocks: list[dict[str, Any]] = []
                    current_text = ""
                    stop_reason = "end_turn"
                    saw_event = False
                    try:
                        for event in self.provider.stream(self.messages, tools_spec, self.system_prompt):
                            saw_event = True   # set for EVERY event incl. Stop -> no retry (nor usage dbl-count)
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
                        break  # stream consumed cleanly
                    except Exception as exc:
                        if self._cancelled:
                            self.messages = pre_turn
                            yield TurnDone("cancelled", total_usage); self._cancelled = False; return
                        if not saw_event and not retried and _is_context_limit_error(exc):
                            retried = True
                            self.messages = _compact_history(self.messages)  # recover from overflow
                            continue                                         # re-stream once, compacted
                        self.messages = pre_turn   # restore the exact pre-turn history (always valid)
                        yield TextDelta(f"\n[error: {exc}]\n")
                        yield TurnDone("error", total_usage); return
                if self._cancelled:
                    yield TurnDone("cancelled", total_usage); self._cancelled = False; return
                if current_text:
                    assistant_blocks.append({"type": "text", "text": current_text})
                if assistant_blocks:
                    self.messages.append({"role": "assistant", "content": assistant_blocks})
                if stop_reason != "tool_use":
                    if stop_reason == "max_tokens":
                        # The reply hit the per-request output cap and was cut off mid-text. Surface it
                        # so it is never a silent truncation; raising agent.max_tokens allows longer replies.
                        yield TextDelta("\n\n_[Reply cut off at the output-token limit — raise "
                                        "`agent.max_tokens` in ~/.scat/config.json for longer replies.]_\n")
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
                    except AnalysisCancelled:
                        # User pressed Stop mid-analysis: record a valid tool_result and end the
                        # turn (don't feed a generic error back and invite a retry).
                        self._cancelled = True
                        tool_result_blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                                                   "content": "Analysis cancelled by the user.", "is_error": True})
                        yield ToolResult(block["id"], block["name"], "cancelled by user", is_error=True)
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
