from scat.agent import runner
from scat.agent.runner import AgentRunner, ToolResult, TurnDone
from scat.agent.providers.base import TextDelta, ToolUse, Stop


def test_compact_tool_result_short_passthrough():
    out = runner._compact_tool_result("scan_folder", {"n_images": 3, "path": "/x"})
    assert '"n_images": 3' in out


def test_compact_tool_result_truncates_large():
    # C1: a many-key payload survives per-item compaction to >6000 chars, triggering the fallback.
    big = {f"k{i}": "y" * 300 for i in range(80)}
    out = runner._compact_tool_result("analyze_folder", big)
    assert len(out) <= 7000 and "compacted" in out


def test_scan_filename_list_survives_compaction():
    # The agent infers groups from filenames; a folder's file list (here 20 > old cap of 8)
    # must NOT be truncated by compaction.
    files = [{"filename": f"cond_{i}.tif", "subfolder": None} for i in range(20)]
    out = runner._compact_tool_result("scan_folder", {"n_images": 20, "files": files})
    assert out.count("cond_") >= 20


class FakeProvider:
    """Round 1: call scan_folder. Round 2: finish."""
    name = "fake"; model = "fake"

    def __init__(self):
        self._round = 0

    def stream(self, messages, tools, system):
        self._round += 1
        if self._round == 1:
            yield ToolUse(id="tu1", name="scan_folder", input={"path": messages[0]["content"][0]["text"]})
            yield Stop(reason="tool_use", usage={"input_tokens": 10, "output_tokens": 5})
        else:
            yield TextDelta(text="Found the images.")
            yield Stop(reason="end_turn", usage={"input_tokens": 8, "output_tokens": 4})


def test_fakeprovider_drives_scan(synth_dir):
    import scat.tools  # noqa: F401 — register tools
    r = AgentRunner(FakeProvider(), "sys", max_loops=5)
    events = list(r.turn(str(synth_dir)))
    kinds = [type(e).__name__ for e in events]
    assert "ToolResult" in kinds and kinds[-1] == "TurnDone"
    tr = next(e for e in events if isinstance(e, ToolResult))
    assert tr.name == "scan_folder" and tr.output["n_images"] > 0
    assert events[-1].stop_reason == "end_turn"


class TwoToolProvider:
    """Round 1: two tool_use blocks, one of which will raise. Round 2: finish."""
    name = "fake2"; model = "fake2"

    def __init__(self):
        self._round = 0

    def stream(self, messages, tools, system):
        self._round += 1
        if self._round == 1:
            yield ToolUse(id="a", name="scan_folder", input={"path": "/tmp"})
            yield ToolUse(id="b", name="boom", input={})
            yield Stop(reason="tool_use", usage={})
        else:
            yield TextDelta(text="done")
            yield Stop(reason="end_turn", usage={})


def test_multi_tool_error_backfill(tmp_path):
    # C6: two tool_use blocks (one raises) -> both get a ToolResult; the error one is flagged;
    # the appended user message contains only tool_result blocks (Anthropic requirement).
    def caller(name, **kw):
        if name == "boom":
            raise ValueError("kaboom")
        return {"n_images": 0}

    r = AgentRunner(TwoToolProvider(), "sys", max_loops=5, tool_caller=caller)
    events = list(r.turn("go"))
    results = [e for e in events if isinstance(e, ToolResult)]
    assert {r_.tool_use_id for r_ in results} == {"a", "b"}
    assert any(r_.is_error and r_.tool_use_id == "b" for r_ in results)
    # the user message with tool results is second-to-... find it in history
    tr_msg = [m for m in r.messages if m["role"] == "user" and
              all(b.get("type") == "tool_result" for b in m["content"])]
    assert tr_msg and {b["tool_use_id"] for b in tr_msg[0]["content"]} == {"a", "b"}


def test_stream_error_cleans_history():
    class BoomProvider:
        name = "boom"; model = "boom"
        def stream(self, messages, tools, system):
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover

    r = AgentRunner(BoomProvider(), "sys", max_loops=3)
    events = list(r.turn("hi"))
    assert isinstance(events[-1], TurnDone) and events[-1].stop_reason == "error"
    # user message was popped so history is clean for the next turn
    assert r.messages == []
