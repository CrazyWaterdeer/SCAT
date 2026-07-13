import pytest
from scat.agent import backend


def test_api_requires_key(monkeypatch):
    monkeypatch.setattr("scat.agent.claude_subscription.subscription_available", lambda: (False, "no"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        backend.build_runner(backend="auto")


def test_api_path_builds_with_key(monkeypatch):
    monkeypatch.setattr("scat.agent.claude_subscription.subscription_available", lambda: (False, "no"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    runner, desc = backend.build_runner(backend="api")
    assert "billed" in desc and runner.__class__.__name__ == "AgentRunner"
