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


def test_analysis_cancelled_ends_turn_without_retry():
    """A tool raising AnalysisCancelled ends the turn with a valid tool_result and no retry
    (Codex F2) — the runner must not feed a generic error back and loop."""
    from scat.progress import AnalysisCancelled

    class OneToolProvider:
        name = "fake"; model = "fake"
        def __init__(self):
            self.rounds = 0
        def stream(self, messages, tools, system):
            self.rounds += 1
            yield ToolUse(id="tu1", name="analyze_folder", input={})
            yield Stop(reason="tool_use", usage={})

    p = OneToolProvider()
    def caller(name, **kw):
        raise AnalysisCancelled("stop")
    r = AgentRunner(p, "sys", max_loops=5, tool_caller=caller)
    events = list(r.turn("go"))
    assert type(events[-1]).__name__ == "TurnDone" and events[-1].stop_reason == "cancelled"
    assert p.rounds == 1, "must not re-query the model after a cancel"
    # every tool_use has a recorded tool_result (valid message history)
    blocks = [b for m in r.messages if isinstance(m.get("content"), list) for b in m["content"]]
    assert any(b.get("type") == "tool_result" for b in blocks)


# ------------------------- T3.2/T3.3: compaction + retry hardening -------------------------
from scat.agent.runner import _KEEP_RECENT, _is_context_limit_error


def _u_text(t="hi"):
    return {"role": "user", "content": [{"type": "text", "text": t}]}


def _a_tool(*ids, text=None):
    blocks = ([{"type": "text", "text": text}] if text else [])
    blocks += [{"type": "tool_use", "id": i, "name": "scan_folder", "input": {"path": "/x"}} for i in ids]
    return {"role": "assistant", "content": blocks}


def _u_result(*ids, content="ok"):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": content} for i in ids]}


def _a_text(t="done"):
    return {"role": "assistant", "content": [{"type": "text", "text": t}]}


def _assert_paired(messages):
    """Every assistant tool_use must be answered by the IMMEDIATELY-following user tool_result with an
    exact, dup-free id-set match (the Anthropic invariant). Adjacency-based, not a global id map."""
    for idx, m in enumerate(messages):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        use_ids = [b["id"] for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not use_ids:
            continue
        assert len(use_ids) == len(set(use_ids)), f"duplicate tool_use ids at {idx}"
        nxt = messages[idx + 1] if idx + 1 < len(messages) else None
        assert nxt and nxt.get("role") == "user" and isinstance(nxt.get("content"), list), f"no result after {idx}"
        res_ids = [b["tool_use_id"] for b in nxt["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
        assert set(use_ids) == set(res_ids), f"tool_use/result id mismatch at {idx}: {use_ids} vs {res_ids}"


def _valid_history(n_pairs):
    msgs = [_u_text("task")]
    for k in range(n_pairs):
        msgs += [_a_tool(f"t{k}"), _u_result(f"t{k}", content="R" * 1500)]
    return msgs


def test_soft_rewrite_stubs_and_preserves():
    m = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "X" * 2000},
        {"type": "tool_result", "tool_use_id": "b", "content": "short"}]}
    out = runner._soft_rewrite(m)
    assert out["content"][0]["content"] == "[earlier result elided to save context]"
    assert out["content"][0]["tool_use_id"] == "a"          # id preserved
    assert out["content"][1]["content"] == "short"          # short one untouched
    assert m["content"][0]["content"] == "X" * 2000         # input NOT mutated
    a = {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "n", "input": {"p": "/x", "k": 1}}]}
    assert runner._soft_rewrite(a)["content"][0]["input"] == {"p": "/x", "k": 1}   # tool_use.input intact object


def test_compact_history_small_unchanged():
    msgs = [_u_text("t"), _a_tool("a"), _u_result("a")]     # n=3 <= _KEEP_RECENT+1 -> all protected
    assert runner._compact_history(msgs) == msgs


def test_compact_history_always_valid_and_preserves_ends():
    for n_pairs in range(0, 9):
        msgs = _valid_history(n_pairs)
        out = runner._compact_history(msgs)
        _assert_paired(out)                                 # never orphans a tool_use / result
        assert out[0] == msgs[0]                            # task kept verbatim
        k = min(_KEEP_RECENT, len(msgs))
        assert out[-k:] == msgs[-k:]                        # last _KEEP_RECENT kept verbatim


def _ids_present(messages):
    return {b["id"] for m in messages if m.get("role") == "assistant"
            for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use"}


def _pad3():   # 3 valid pairs (6 messages) to fill the protected last-6 window
    out = []
    for k in range(3):
        out += [_a_tool(f"p{k}"), _u_result(f"p{k}")]
    return out


def test_compact_history_pair_predicate_guards():
    pad = _pad3()
    # partial (assistant a,b -> result a only): must NOT be dropped
    out = runner._compact_history([_u_text("t"), _a_tool("a", "b"), _u_result("a")] + pad)
    assert {"a", "b"} <= _ids_present(out)
    # extra (assistant a -> result a,x): must NOT be dropped
    out = runner._compact_history([_u_text("t"), _a_tool("a"), _u_result("a", "x")] + pad)
    assert "a" in _ids_present(out)
    # text-only assistant then a result: not a pair -> assistant kept, not treated as droppable
    out = runner._compact_history([_u_text("t"), _a_text("thinking"), _u_result("a")] + pad)
    assert any(m.get("role") == "assistant" and m["content"][0].get("type") == "text" for m in out)
    # valid (text + tool_use a -> result a): IS dropped
    out = runner._compact_history([_u_text("t"), _a_tool("a", text="hmm"), _u_result("a")] + pad)
    assert "a" not in _ids_present(out) and {"p0", "p1", "p2"} <= _ids_present(out)
    _assert_paired(out)


def test_compact_history_duplicate_ids_not_dropped():
    pad = _pad3()
    dup_assistant = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "a", "name": "n", "input": {}},
        {"type": "tool_use", "id": "a", "name": "n", "input": {}}]}   # duplicate id 'a'
    out = runner._compact_history([_u_text("t"), dup_assistant, _u_result("a")] + pad)
    assert "a" in _ids_present(out)   # dup ids -> not droppable (guarded), left in place


def test_is_context_limit_error():
    class E400(Exception):
        status_code = 400

    class E429(Exception):
        status_code = 429

    class Estatusless(Exception):
        pass

    assert _is_context_limit_error(E400("prompt is too long: 999 tokens > 200 maximum"))
    assert _is_context_limit_error(Estatusless("the context window was exceeded"))
    assert _is_context_limit_error(Estatusless("input length and max_tokens exceed context limit"))
    assert not _is_context_limit_error(ValueError("bad tool schema"))
    assert not _is_context_limit_error(E400("invalid request: unknown field in context of tool"))  # incidental
    assert not _is_context_limit_error(E429("too many tokens"))   # non-400 status -> not a context overflow


class _FakeCtxError(Exception):
    status_code = 400

    def __str__(self):
        return "prompt is too long: 1000000 tokens > 200000 maximum"


def test_reactive_retry_on_context_limit():
    class CtxThenOK:
        name = "x"; model = "x"
        def __init__(self): self.calls = 0
        def stream(self, messages, tools, system):
            self.calls += 1
            if self.calls == 1:
                raise _FakeCtxError()
            yield TextDelta(text="recovered")
            yield Stop(reason="end_turn", usage={"input_tokens": 5})

    p = CtxThenOK()
    r = AgentRunner(p, "sys", max_loops=3)
    events = list(r.turn("go"))
    assert p.calls == 2                                     # compacted + retried once
    assert isinstance(events[-1], TurnDone) and events[-1].stop_reason == "end_turn"
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "recovered" in text and "[error" not in text


def test_no_retry_on_non_context_error():
    class BadReq(Exception):
        status_code = 400

    class BadThenOK:
        name = "x"; model = "x"
        def __init__(self): self.calls = 0
        def stream(self, m, t, s):
            self.calls += 1
            raise BadReq("invalid tool_use block: schema mismatch")
            yield  # pragma: no cover

    p = BadThenOK()
    r = AgentRunner(p, "sys", max_loops=3)
    events = list(r.turn("go"))
    assert p.calls == 1 and events[-1].stop_reason == "error"
    assert r.messages == []                                 # restored to the (empty) pre-turn snapshot


def test_no_retry_after_visible_event():
    class YieldThenCtx:
        name = "x"; model = "x"
        def __init__(self): self.calls = 0
        def stream(self, m, t, s):
            self.calls += 1
            yield TextDelta(text="partial")
            raise _FakeCtxError()

    p = YieldThenCtx()
    r = AgentRunner(p, "sys", max_loops=3)
    events = list(r.turn("go"))
    assert p.calls == 1 and events[-1].stop_reason == "error"
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text.count("partial") == 1                       # no duplicate UI output


def test_stop_then_error_no_retry_no_double_count():
    class StopThenBoom:
        name = "x"; model = "x"
        def __init__(self): self.calls = 0
        def stream(self, m, t, s):
            self.calls += 1
            yield Stop(reason="end_turn", usage={"input_tokens": 10})
            raise _FakeCtxError()

    p = StopThenBoom()
    r = AgentRunner(p, "sys", max_loops=3)
    events = list(r.turn("go"))
    assert p.calls == 1                                     # saw Stop -> no retry
    td = [e for e in events if isinstance(e, TurnDone)][-1]
    assert td.stop_reason == "error" and td.total_usage.get("input_tokens", 0) == 10   # counted once


def test_fatal_error_after_tool_round_leaves_no_orphan():
    """The pre-existing cleanup bug: a stream error on the 2nd iteration (after a real tool round) must
    not orphan the first round's tool_use. Restoring the pre-turn snapshot keeps history valid."""
    class ToolThenBoom:
        name = "x"; model = "x"
        def __init__(self): self.round = 0
        def stream(self, m, t, s):
            self.round += 1
            if self.round == 1:
                yield ToolUse(id="tu1", name="scan_folder", input={"path": "/x"})
                yield Stop(reason="tool_use", usage={})
            else:
                raise RuntimeError("boom on second call")
                yield  # pragma: no cover

    def caller(name, **kw):
        return {"n_images": 0}

    r = AgentRunner(ToolThenBoom(), "sys", max_loops=5, tool_caller=caller)
    # a prior completed turn so pre_turn is non-empty
    r.messages = [_u_text("earlier"), _a_text("ok")]
    prior = list(r.messages)
    events = list(r.turn("go"))
    assert events[-1].stop_reason == "error"
    assert r.messages == prior                              # restored -> no orphaned tool_use
    _assert_paired(r.messages)


def test_anthropic_provider_forwards_retries_and_tokens(monkeypatch):
    import scat.agent.providers.anthropic_api as ap
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    prov = ap.AnthropicProvider(api_key="sk-x", model="m", max_tokens=99, max_retries=7)
    assert captured.get("max_retries") == 7 and prov.max_tokens == 99
