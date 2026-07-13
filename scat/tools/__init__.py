"""Import every tool module so @tool registers it, and re-export registry helpers."""
from scat.agent.registry import call_tool, tools_for_anthropic, iter_tools, get_tool  # noqa: F401
from . import scan, grouping, pipeline_tools  # noqa: F401  (registration side-effects)

__all__ = ["call_tool", "tools_for_anthropic", "iter_tools", "get_tool"]
