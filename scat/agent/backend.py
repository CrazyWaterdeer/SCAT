from __future__ import annotations
import os
from scat.agent.prompts import SYSTEM_PROMPT

# The current shipping Claude models offered in the GUI model picker, newest/most-capable
# first. SINGLE SOURCE OF TRUTH — update this when Anthropic ships newer models. IDs are
# exact and version-pinned (there are no "-latest" aliases). Both backends take the same
# bare id: the API path passes it to the Messages API; the subscription path passes it to
# ClaudeAgentOptions and the `claude` CLI resolves it. Default stays claude-opus-4-8.
LATEST_MODELS = [
    ("Opus 4.8", "claude-opus-4-8"),
    ("Fable 5", "claude-fable-5"),
    ("Sonnet 5", "claude-sonnet-5"),
    ("Haiku 4.5", "claude-haiku-4-5"),
]


def build_runner(backend: str = "auto", model: str = "claude-opus-4-8", max_loops: int = 40):
    """Return (runner, description). Prefers subscription unless overridden.

    description is loud about billing so a user never unknowingly bills the API.
    """
    import scat.tools  # noqa: F401 — ensure tools are registered
    from scat.agent.claude_subscription import subscription_available, ClaudeSubscriptionRunner
    from scat.agent.runner import AgentRunner
    from scat.agent.providers.anthropic_api import AnthropicProvider

    sub_ok, sub_reason = subscription_available()
    want = backend
    auto_fell_back = False
    if want == "auto":
        want = "subscription" if sub_ok else "api"
        auto_fell_back = want == "api"   # subscription wasn't available; Auto is billing the API

    if want == "subscription":
        if not sub_ok:
            raise RuntimeError(f"subscription backend requested but unavailable: {sub_reason}")
        runner = ClaudeSubscriptionRunner(model=model, system_prompt=SYSTEM_PROMPT, max_turns=max_loops)
        return runner, f"Claude subscription (no API charges), model={model}"

    if want == "api":
        from scat.config import config
        # A non-empty ANTHROPIC_API_KEY wins; otherwise use the key saved in Settings. Both are
        # stripped so stray whitespace never reaches the SDK (which would send a broken header).
        key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip() or (config.get("agent.api_key") or "").strip()
        if not key:
            raise RuntimeError(
                "No backend available. Either log in to Claude (`claude` CLI) for the "
                "subscription path, or set an API key (Settings › Assistant, or ANTHROPIC_API_KEY).")
        provider = AnthropicProvider(api_key=key, model=model,
                                     max_tokens=config.get("agent.max_tokens", 16384),
                                     max_retries=config.get("agent.max_retries", 3))
        runner = AgentRunner(provider, SYSTEM_PROMPT, max_loops=max_loops)
        if auto_fell_back:
            # Auto silently landed on the billed API because the subscription isn't connected —
            # warn loudly so a transient Claude-login failure never bills the user by surprise.
            return runner, (f"⚠ Subscription not connected ({sub_reason}) — using the billed "
                            f"API (requests cost money), model={model}")
        return runner, f"API key set — requests are billed, model={model}"

    raise ValueError(f"unknown backend: {backend}")
