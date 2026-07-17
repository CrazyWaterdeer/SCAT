import pytest
from scat.agent import backend


def _no_subscription(monkeypatch):
    monkeypatch.setattr("scat.agent.claude_subscription.subscription_available", lambda: (False, "no"))


def _config_api_key(monkeypatch, value):
    """Pin config.get('agent.api_key') to `value` without touching the user's real config."""
    from scat.config import config
    orig = config.get
    monkeypatch.setattr(config, "get", lambda k, d=None: value if k == "agent.api_key" else orig(k, d))


def test_api_requires_key(monkeypatch):
    _no_subscription(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _config_api_key(monkeypatch, "")            # no env key AND no saved key
    with pytest.raises(RuntimeError):
        backend.build_runner(backend="auto")


def test_api_path_builds_with_key(monkeypatch):
    _no_subscription(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    runner, desc = backend.build_runner(backend="api")
    assert "billed" in desc and runner.__class__.__name__ == "AgentRunner"


def test_api_path_falls_back_to_config_key(monkeypatch):
    """With no env var, the API backend uses the key saved in Settings (config.agent.api_key)."""
    _no_subscription(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _config_api_key(monkeypatch, "sk-cfg")
    runner, desc = backend.build_runner(backend="api")
    assert "billed" in desc and runner.__class__.__name__ == "AgentRunner"
