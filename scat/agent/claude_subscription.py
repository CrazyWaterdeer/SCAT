"""Subscription-backed agent that runs on the local Claude Code login.

Unlike ``AnthropicProvider`` (a ``Provider`` implementing a single-model-turn
``stream()`` behind ``AgentRunner``), the Claude Agent SDK owns its *own* agentic
loop (it drives the ``claude`` CLI, which executes tools itself), so it cannot sit
behind ``stream()``. This class fuses provider + runner: it presents the same
surface the chat/CLI layer drives on ``AgentRunner`` — ``turn()`` yielding the
shared ``RunEvent`` types, plus ``reset()`` / ``cancel()`` / ``close()`` — while
internally delegating the loop to the SDK. SCAT's tools are bridged in as an
in-process MCP server; the SDK's message stream is translated back into RunEvents.

Auth: the ``claude`` CLI resolves its own credentials, so **no API key is needed**
— it uses whatever the user logged into Claude Code with (a Pro/Max subscription).
We never read or forward the OAuth token ourselves; the CLI holds it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import queue
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scat.agent.providers.base import TextDelta, ToolUse, ToolUseStart
from scat.agent.runner import ToolResult, TurnDone

_MCP_SERVER = "scat"
_TOOL_PREFIX = f"mcp__{_MCP_SERVER}__"

_AUTH_ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_ENV_LOCK = threading.Lock()

# Built-in Claude Code tools the analysis agent never needs. permission_mode
# "dontAsk" already denies anything not in allowed_tools; this stops wasted turns.
_DISALLOWED_BUILTINS = ["Bash", "Read", "Write", "Edit", "NotebookEdit", "WebFetch", "WebSearch"]


def _sdk():
    import claude_agent_sdk
    return claude_agent_sdk


def subscription_available() -> tuple[bool, str | None]:
    """Cheap probe (no SDK import / CLI spawn) of whether the subscription path works."""
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return False, "SDK missing"
    if shutil.which("claude") is None:
        return False, "claude not found"
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return True, None
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    creds = (Path(config_dir) if config_dir else Path.home() / ".claude") / ".credentials.json"
    if creds.exists():
        return True, None
    return False, "not logged in"


def _strip_ns(name: str) -> str:
    return name[len(_TOOL_PREFIX):] if name.startswith(_TOOL_PREFIX) else name


def _flatten_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _map_usage(raw: Any) -> dict[str, int]:
    usage: dict[str, int] = {}
    if isinstance(raw, dict):
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = raw.get(key)
            if isinstance(value, int):
                usage[key] = value
    return usage


def _translate_message(message: Any, id_to_name: dict[str, str]) -> list[Any]:
    """Translate one SDK message into RunEvent objects (duck-typed; blocks lack a `type`)."""
    events: list[Any] = []
    if hasattr(message, "num_turns") and hasattr(message, "session_id"):
        subtype = getattr(message, "subtype", None)
        stop = getattr(message, "stop_reason", None) or ("end_turn" if subtype == "success" else (subtype or "end_turn"))
        events.append(TurnDone(stop_reason=str(stop), total_usage=_map_usage(getattr(message, "usage", None))))
        return events
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return events
    for block in content:
        if hasattr(block, "tool_use_id"):
            tool_use_id = block.tool_use_id
            events.append(ToolResult(tool_use_id=tool_use_id, name=id_to_name.get(tool_use_id, "tool"),
                                     output=_flatten_tool_result(getattr(block, "content", "")),
                                     is_error=bool(getattr(block, "is_error", False))))
        elif hasattr(block, "input") and hasattr(block, "name") and hasattr(block, "id"):
            name = _strip_ns(block.name)
            id_to_name[block.id] = name
            events.append(ToolUseStart(id=block.id, name=name))
            events.append(ToolUse(id=block.id, name=name, input=dict(block.input or {})))
        elif hasattr(block, "text"):
            if block.text:
                events.append(TextDelta(text=block.text))
    return events


@contextmanager
def _force_subscription_env():
    """Pop API-key env vars around the turn so the CLI uses subscription OAuth."""
    with _ENV_LOCK:
        saved = {k: os.environ.pop(k) for k in _AUTH_ENV_KEYS if k in os.environ}
        try:
            yield
        finally:
            os.environ.update(saved)


_SENTINEL = object()


class ClaudeSubscriptionRunner:
    """Runner backed by the local Claude Code subscription via the Claude Agent SDK."""

    name = "subscription"

    def __init__(self, model: str, system_prompt: str, tool_caller: Any | None = None, max_turns: int = 40) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._tool_caller = tool_caller
        self._cancelled = False
        self._session_id: str | None = None
        self._server: Any | None = None
        self._allowed: list[str] = []
        self._cwd = tempfile.gettempdir()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any | None = None

    def cancel(self) -> None:
        self._cancelled = True
        client, loop = self._client, self._loop
        if client is not None and loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(client.interrupt(), loop)
            except Exception:
                pass

    def reset(self) -> None:
        self._cancelled = False
        self._session_id = None
        self._disconnect()

    def close(self) -> None:
        self._disconnect()
        loop, thread = self._loop, self._thread
        self._loop, self._thread = None, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        if loop is not None:
            try:
                loop.close()
            except Exception:
                pass

    def _bridged_entries(self) -> list[Any]:
        import scat.tools  # noqa: F401 — ensure @tool registration side-effects ran
        from scat.agent.registry import iter_tools
        return [e for e in iter_tools() if e.subagent is None and e.llm]

    def _make_handler(self, tool_name: str):
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            from scat.agent.runner import _compact_tool_result
            from scat.tools import call_tool
            caller = self._tool_caller or call_tool
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, lambda: caller(tool_name, **args))
                return {"content": [{"type": "text", "text": _compact_tool_result(tool_name, result)}]}
            except Exception as exc:
                return {"content": [{"type": "text", "text": f"ERROR: {exc}"}], "is_error": True}
        return handler

    def _build_server(self) -> tuple[Any, list[str]]:
        if self._server is not None:
            return self._server, self._allowed
        sdk = _sdk()
        sdk_tools = []
        allowed: list[str] = []
        for entry in self._bridged_entries():
            sdk_tools.append(sdk.tool(entry.name, entry.description, entry.json_schema)(self._make_handler(entry.name)))
            allowed.append(f"{_TOOL_PREFIX}{entry.name}")
        self._server = sdk.create_sdk_mcp_server(name=_MCP_SERVER, version="0.1.0", tools=sdk_tools)
        self._allowed = allowed
        return self._server, self._allowed

    def _build_options(self, server: Any, allowed: list[str]) -> Any:
        return _sdk().ClaudeAgentOptions(
            model=self.model, system_prompt=self.system_prompt,
            mcp_servers={_MCP_SERVER: server}, allowed_tools=allowed,
            disallowed_tools=list(_DISALLOWED_BUILTINS), permission_mode="dontAsk",
            setting_sources=[], max_turns=self.max_turns, cwd=self._cwd, resume=self._session_id)

    def _ensure_loop(self) -> None:
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name="claude-subscription-loop", daemon=True)
        thread.start()
        self._loop, self._thread = loop, thread

    def _call(self, coro: Any, timeout: float | None = None) -> Any:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _ensure_connected(self) -> None:
        self._ensure_loop()
        if self._client is not None:
            return
        sdk = _sdk()
        server, allowed = self._build_server()
        client = sdk.ClaudeSDKClient(options=self._build_options(server, allowed))
        with _force_subscription_env():
            self._call(client.connect())
        self._client = client

    def _disconnect(self) -> None:
        client = self._client
        self._client = None
        if client is None or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(client.disconnect(), self._loop).result(timeout=10)
        except Exception:
            pass

    async def _adrive_turn(self, user_text: str, q: queue.Queue) -> None:
        id_to_name: dict[str, str] = {}
        try:
            await self._client.query(user_text)
            async for message in self._client.receive_response():
                if self._cancelled:
                    break
                if hasattr(message, "num_turns") and hasattr(message, "session_id"):
                    self._session_id = getattr(message, "session_id", None) or self._session_id
                for event in _translate_message(message, id_to_name):
                    q.put(event)
        except Exception as exc:
            q.put(TextDelta(text=f"\n[subscription agent error] {type(exc).__name__}: {exc}"))
            q.put(TurnDone(stop_reason="error", total_usage={}))
            client, self._client, self._session_id = self._client, None, None
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    pass
        finally:
            q.put(_SENTINEL)

    def turn(self, user_text: str) -> Iterator[Any]:
        self._cancelled = False
        try:
            self._ensure_connected()
        except Exception as exc:
            self._disconnect()
            yield TextDelta(text=f"\n[subscription agent error] {type(exc).__name__}: {exc}")
            yield TurnDone(stop_reason="error", total_usage={})
            return
        q: queue.Queue = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(self._adrive_turn(user_text, q), self._loop)
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            try:
                future.result(timeout=5)
            except Exception:
                pass
