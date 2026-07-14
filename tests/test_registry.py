from scat.agent import registry


def test_tool_schema_from_hints():
    # C3: do NOT clear the global registry (corrupts full-suite runs). Register and
    # assert the demo tool is present among the specs.
    @registry.tool(description="demo")
    def demo(path: str, min_area: int = 20) -> dict:
        return {"path": path}

    specs = {s["name"]: s for s in registry.tools_for_anthropic()}
    s = specs["demo"]
    assert s["description"] == "demo"
    props = s["input_schema"]["properties"]
    assert props["path"]["type"] == "string" and props["min_area"]["type"] == "integer"
    assert "title" not in s["input_schema"]


def test_call_tool_validates_and_dispatches():
    @registry.tool()
    def add(a: int, b: int) -> int:
        return a + b

    assert registry.call_tool("add", a=2, b=3) == 5
