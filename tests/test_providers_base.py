import pytest
from scat.agent.providers import base


def test_events_and_provider_globals():
    base.set_current_provider(None)
    with pytest.raises(RuntimeError):
        base.get_current_provider()
    t = base.ToolUse(id="x", name="scan", input={"path": "/a"})
    assert t.name == "scan" and t.input["path"] == "/a"
    assert base.Stop(reason="end_turn").usage == {}
