import sys
import types
import pytest
from scat.agent.claude_subscription import (
    subscription_available, ClaudeSubscriptionRunner, _translate_message, _strip_ns,
)


def test_subscription_probe_returns_tuple():
    ok, reason = subscription_available()
    assert isinstance(ok, bool)


def test_strip_ns():
    assert _strip_ns("mcp__scat__scan_folder") == "scan_folder"
    assert _strip_ns("plain") == "plain"


def test_translate_tool_use_strips_prefix():
    from scat.agent.runner import ToolResult
    from scat.agent.providers.base import ToolUse, ToolUseStart

    class Blk:
        pass

    tu = Blk(); tu.id = "1"; tu.name = "mcp__scat__scan_folder"; tu.input = {"path": "/x"}
    msg = Blk(); msg.content = [tu]
    id_to_name: dict = {}
    evs = _translate_message(msg, id_to_name)
    kinds = [type(e).__name__ for e in evs]
    assert kinds == ["ToolUseStart", "ToolUse"]
    assert id_to_name["1"] == "scan_folder"


def test_build_server_with_fake_sdk(monkeypatch):
    # C12: cover _build_server / allowed-tool naming without a login, via a fake SDK.
    import scat.tools  # noqa: F401 — register tools
    captured = {}

    def fake_tool(name, desc, schema):
        def deco(handler):
            return {"name": name}
        return deco

    def fake_server(name, version, tools):
        captured["name"] = name
        captured["tools"] = tools
        return object()

    fake = types.SimpleNamespace(tool=fake_tool, create_sdk_mcp_server=fake_server)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)

    r = ClaudeSubscriptionRunner(model="m", system_prompt="s")
    _server, allowed = r._build_server()
    assert captured["name"] == "scat"
    assert "mcp__scat__scan_folder" in allowed
    assert "mcp__scat__analyze_folder" in allowed
    assert len(captured["tools"]) == len(allowed) >= 5


@pytest.mark.skipif(not subscription_available()[0], reason="no claude subscription login")
def test_bridge_round_trips_a_tool():
    import scat.tools  # noqa: F401
    runner = ClaudeSubscriptionRunner(
        model="claude-opus-4-8",
        system_prompt="You are a test. Call scan_folder on /nonexistent-xyz, then stop.")
    kinds = [type(ev).__name__ for ev in runner.turn("scan the folder /nonexistent-xyz")]
    runner.close()
    assert "TurnDone" in kinds
