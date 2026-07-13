# SCAT AI-Agent Layer (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conversational AI-agent layer to SCAT so `python -m scat.cli chat` → "analyze /data/exp1" runs the full pipeline (scan → detect → classify → auto-group → stats → report) autonomously, driven by an LLM over a shared `@tool` registry.

**Architecture:** Port Imajin's napari-free agent core (Provider Protocol + AgentRunner tool-loop + `@tool` registry + provenance + Claude-subscription bridge). SCAT's pipeline becomes **plain Python services** in `scat/pipeline.py`; `@tool` wrappers in `scat/tools/` are thin adapters over them. CLI `analyze` is reimplemented on the same services. Backend: Claude subscription (Agent SDK, no key) preferred, `ANTHROPIC_API_KEY` fallback.

**Tech Stack:** Python 3.14 (uv venv), `anthropic` SDK, `pydantic` v2, `claude-agent-sdk` (subscription, optional), existing SCAT (`Analyzer`, `ReportGenerator`, `statistics`, `report`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-14-scat-ai-agent-design.md`. **Branch:** `feat/ai-agent`.

**Source to port from:** `/home/lab/Imajin/src/imajin/` — `tools/registry.py`, `agent/providers/base.py`, `agent/providers/anthropic.py`, `agent/runner.py`, `agent/provenance.py`, `agent/providers/claude_agent.py`.

---

## File structure (created by this plan)

```
scat/
  pipeline.py          # NEW plain services: scan_folder_service, analyze_folder_service, run_statistics_service, generate_report_service  (no pydantic/anthropic)
  agent/
    __init__.py
    provenance.py      # port of Imajin provenance.py
    registry.py        # port of Imajin tools/registry.py (@tool machinery)
    providers/
      __init__.py
      base.py          # port: Event dataclasses + Provider Protocol + current-provider globals
      anthropic_api.py # port: AnthropicProvider
    runner.py          # port (stripped): AgentRunner + ToolResult/TurnDone/RunEvent + _compact_tool_result
    claude_subscription.py  # port of Imajin claude_agent.py (subscription bridge)
    prompts.py         # NEW SCAT system prompt
    context.py         # NEW durable ledger from on-disk results
    backend.py         # NEW provider/runner selection + observability
  tools/
    __init__.py        # barrel: imports all tool modules (registration) + re-exports call_tool/tools_for_anthropic/iter_tools
    scan.py            # @tool scan_folder
    grouping.py        # @tool infer_groups + _build_group_metadata helper
    pipeline_tools.py  # @tool analyze_folder / run_statistics / generate_report (thin adapters over scat.pipeline)
  cli.py               # MODIFY: add `chat` subcommand; reimplement `analyze` on scat.pipeline services
  config.py            # MODIFY: add "agent" section defaults; delete 4 dead keys
tests/
  test_registry.py test_infer_groups.py test_group_metadata.py
  test_agent_runner.py (FakeProvider e2e) test_backend.py
  test_pipeline_parity.py (anti-drift gate) test_subscription_bridge.py (guarded)
  test_core_imports_without_agent.py
pyproject.toml         # MODIFY: add pydantic + anthropic deps; [agent] extra with claude-agent-sdk
```

**Import-cycle rule:** `registry.py` lazy-imports `provenance` inside the wrapper (not at module top). `runner.py` and `claude_subscription.py` import tools via `from scat.tools import ...` (the barrel), which imports registry — so the barrel must not import runner/claude_subscription at module load.

---

## Task 0: Dependencies + package skeleton

**Files:**
- Modify: `pyproject.toml` (dependencies + optional-dependencies)
- Create: `scat/agent/__init__.py`, `scat/agent/providers/__init__.py`, `scat/tools/__init__.py` (temporary empty), `scat/pipeline.py` (empty stub)

- [ ] **Step 1: Add deps to `pyproject.toml`.** In `[project].dependencies` add `"pydantic>=2.0",` and `"anthropic>=0.40",`. Add:
```toml
[project.optional-dependencies]
agent = ["claude-agent-sdk>=0.1.0"]
```
(keep existing `dev`, `deep`, `pdf` extras).

- [ ] **Step 2: Install.**
Run: `cd /home/lab/SCAT && uv sync` (or `.venv/bin/python -m pip install pydantic anthropic`)
Expected: `pydantic` and `anthropic` importable: `.venv/bin/python -c "import pydantic, anthropic; print('ok')"` → `ok`.

- [ ] **Step 3: Create empty package files.**
```bash
mkdir -p scat/agent/providers scat/tools
: > scat/agent/__init__.py
: > scat/agent/providers/__init__.py
: > scat/tools/__init__.py
```
Create `scat/pipeline.py` with only `"""SCAT pipeline services (plain Python, no agent deps)."""`.

- [ ] **Step 4: Commit.**
```bash
git add pyproject.toml scat/agent scat/tools scat/pipeline.py
git commit -m "chore(agent): add pydantic/anthropic deps + agent package skeleton"
```

---

## Task 1: Provenance (`scat/agent/provenance.py`)

**Files:**
- Create: `scat/agent/provenance.py`
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_provenance.py
from scat.agent import provenance

def test_record_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_sessions_dir", lambda: tmp_path)
    sid = provenance.start_session(driver="test")
    provenance.record_call("scan_folder", {"path": "/x"}, {"n_images": 3}, 0.01, ok=True)
    rows = provenance.read_session(sid)
    assert len(rows) == 1 and rows[0]["tool"] == "scan_folder" and rows[0]["ok"] is True

def test_summarize_collapses_array():
    import numpy as np
    out = provenance._summarize({"img": np.zeros((4, 4)), "items": list(range(50))})
    assert out["img"]["_kind"] == "array" and out["img"]["shape"] == [4, 4]
    assert len(out["items"]) == 20  # truncated
```

- [ ] **Step 2: Run test to verify it fails.**
Run: `.venv/bin/python -m pytest tests/test_provenance.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `scat/agent/provenance.py`.** Type this verbatim (already SCAT-adapted):
```python
from __future__ import annotations
import json, uuid, tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from scat.config import get_config_dir


@dataclass
class CallRecord:
    timestamp: str
    tool: str
    inputs: dict[str, Any]
    output_summary: Any
    duration_s: float
    ok: bool
    driver: str


_CURRENT_SESSION_ID: str | None = None
_CURRENT_DRIVER: str = "direct"
_LOG_PATH: Path | None = None


def _sessions_dir() -> Path:
    d = get_config_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_session(driver: str = "direct") -> str:
    global _CURRENT_SESSION_ID, _CURRENT_DRIVER, _LOG_PATH
    _CURRENT_SESSION_ID = uuid.uuid4().hex[:12]
    _CURRENT_DRIVER = driver
    _LOG_PATH = _sessions_dir() / f"{_CURRENT_SESSION_ID}.jsonl"
    return _CURRENT_SESSION_ID


def set_driver(driver: str) -> None:
    global _CURRENT_DRIVER
    _CURRENT_DRIVER = driver


def current_session_id() -> str | None:
    return _CURRENT_SESSION_ID


def _summarize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _summarize(v) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_summarize(v) for v in list(value)[:20]]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"_kind": "array", "shape": list(shape), "dtype": str(getattr(value, "dtype", "?"))}
    return f"<{type(value).__name__}>"


def record_call(tool: str, inputs: dict[str, Any], output: Any, duration_s: float, ok: bool) -> None:
    global _LOG_PATH
    if _LOG_PATH is None:
        start_session(driver=_CURRENT_DRIVER)
    rec = CallRecord(datetime.now(UTC).isoformat(), tool, _summarize(inputs),
                     _summarize(output), duration_s, ok, _CURRENT_DRIVER)
    line = json.dumps(asdict(rec), default=str) + "\n"
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        fb = Path(tempfile.gettempdir()) / "scat" / "sessions"
        fb.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = fb / _LOG_PATH.name
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)


def read_session(session_id: str | None = None) -> list[dict[str, Any]]:
    path = _LOG_PATH if session_id is None else _sessions_dir() / f"{session_id}.jsonl"
    if path is None or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out
```

- [ ] **Step 4: Run test to verify it passes.**
Run: `.venv/bin/python -m pytest tests/test_provenance.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit.**
```bash
git add scat/agent/provenance.py tests/test_provenance.py
git commit -m "feat(agent): provenance session log (ported from Imajin)"
```

---

## Task 2: Tool registry (`scat/agent/registry.py`)

**Files:**
- Create: `scat/agent/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_registry.py
from scat.agent import registry

def test_tool_schema_from_hints():
    registry._REGISTRY.clear()

    @registry.tool(description="demo")
    def demo(path: str, min_area: int = 20) -> dict:
        return {"path": path, "min_area": min_area}

    specs = registry.tools_for_anthropic()
    assert len(specs) == 1
    s = specs[0]
    assert s["name"] == "demo" and s["description"] == "demo"
    props = s["input_schema"]["properties"]
    assert props["path"]["type"] == "string" and props["min_area"]["type"] == "integer"
    assert "title" not in s["input_schema"]  # _compact_json_schema strips titles

def test_call_tool_validates_and_dispatches():
    registry._REGISTRY.clear()

    @registry.tool()
    def add(a: int, b: int) -> int:
        return a + b

    assert registry.call_tool("add", a=2, b=3) == 5
```

- [ ] **Step 2: Run to verify fail.**
Run: `.venv/bin/python -m pytest tests/test_registry.py -q` → FAIL (no module).

- [ ] **Step 3: Write `scat/agent/registry.py`.** Type verbatim (vision_hint/phase/worker fields dropped):
```python
from __future__ import annotations
import functools, inspect, time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints
from pydantic import BaseModel, create_model


@dataclass
class ToolEntry:
    name: str
    description: str
    func: Callable[..., Any]
    input_model: type[BaseModel]
    subagent: str | None = None
    manual: bool = True
    llm: bool = True

    @property
    def json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


_REGISTRY: dict[str, ToolEntry] = {}


def _build_input_model(name: str, func: Callable[..., Any]) -> type[BaseModel]:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    fields: dict[str, Any] = {}
    for pname, p in sig.parameters.items():
        if pname in {"self", "cls"}:
            continue
        ann = hints.get(pname, p.annotation if p.annotation is not inspect.Parameter.empty else Any)
        default = p.default if p.default is not inspect.Parameter.empty else ...
        fields[pname] = (ann, default)
    return create_model(f"{name}Input", **fields)


def tool(*, name=None, description="", subagent=None, manual=None, llm=True, input_model=None):
    def decorator(func):
        tname = name or func.__name__
        model = input_model or _build_input_model(tname, func)
        desc = description or (func.__doc__ or "").strip().split("\n")[0]
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            from scat.agent.provenance import record_call
            try:
                bound = sig.bind(*args, **kwargs); bound.apply_defaults()
                inputs = dict(bound.arguments)
            except TypeError:
                inputs = {"args": args, "kwargs": kwargs}
            t0 = time.perf_counter()
            try:
                r = func(*args, **kwargs)
                record_call(tname, inputs, r, time.perf_counter() - t0, ok=True)
                return r
            except Exception as e:
                record_call(tname, inputs, str(e), time.perf_counter() - t0, ok=False)
                raise

        entry = ToolEntry(tname, desc, wrapped, model, subagent=subagent,
                          manual=(subagent is None) if manual is None else manual, llm=llm)
        _REGISTRY[tname] = entry
        wrapped.__tool_entry__ = entry
        return wrapped
    return decorator


def get_tool(name): return _REGISTRY[name]
def iter_tools(): return list(_REGISTRY.values())


def call_tool(tool_name: str, **kwargs: Any) -> Any:
    entry = _REGISTRY[tool_name]
    return entry.func(**entry.input_model(**kwargs).model_dump())


def _entries_for(subagent):
    return [e for e in _REGISTRY.values() if e.subagent == subagent and e.llm]


def _compact_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _compact_json_schema(v) for k, v in value.items() if k != "title"}
    if isinstance(value, list):
        return [_compact_json_schema(v) for v in value]
    return value


def tools_for_anthropic(subagent=None):
    return [{"name": e.name, "description": e.description,
             "input_schema": _compact_json_schema(e.json_schema)} for e in _entries_for(subagent)]
```

- [ ] **Step 4: Run to verify pass.**
Run: `.venv/bin/python -m pytest tests/test_registry.py -q` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add scat/agent/registry.py tests/test_registry.py
git commit -m "feat(agent): @tool registry with pydantic schema generation (ported)"
```

---

## Task 3: Provider base (`scat/agent/providers/base.py`)

**Files:**
- Create: `scat/agent/providers/base.py`
- Test: `tests/test_providers_base.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_providers_base.py
import pytest
from scat.agent.providers import base

def test_events_and_provider_globals():
    base.set_current_provider(None)
    with pytest.raises(RuntimeError):
        base.get_current_provider()
    t = base.ToolUse(id="x", name="scan", input={"path": "/a"})
    assert t.name == "scan" and t.input["path"] == "/a"
    assert base.Stop(reason="end_turn").usage == {}
```

- [ ] **Step 2: Run to verify fail.** → FAIL.

- [ ] **Step 3: Write `scat/agent/providers/base.py`.** Type verbatim:
```python
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolUseStart:
    id: str
    name: str


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Stop:
    reason: str
    usage: dict[str, Any] = field(default_factory=dict)


Event = TextDelta | ToolUseStart | ToolUse | Stop


class Provider(Protocol):
    name: str
    model: str
    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str) -> Iterator[Event]: ...


_CURRENT_PROVIDER: Provider | None = None


def set_current_provider(p: Provider | None) -> None:
    global _CURRENT_PROVIDER
    _CURRENT_PROVIDER = p


def get_current_provider() -> Provider:
    if _CURRENT_PROVIDER is None:
        raise RuntimeError(
            "No active LLM provider. Agent tools require an AgentRunner "
            "turn to be in progress (the runner registers its provider "
            "before tool dispatch)."
        )
    return _CURRENT_PROVIDER
```

- [ ] **Step 4: Run to verify pass.** → PASS.

- [ ] **Step 5: Commit.**
```bash
git add scat/agent/providers/base.py tests/test_providers_base.py
git commit -m "feat(agent): provider Event protocol (ported)"
```

---

## Task 4: Anthropic API provider (`scat/agent/providers/anthropic_api.py`)

**Files:**
- Create: `scat/agent/providers/anthropic_api.py`
- Test: covered indirectly by Task 6 (FakeProvider); no network test here.

- [ ] **Step 1: Write `scat/agent/providers/anthropic_api.py`.** Type verbatim (model_catalog resolution removed, default model `claude-opus-4-8`):
```python
from __future__ import annotations
import json, os
from collections.abc import Iterator
from typing import Any
from scat.agent.providers.base import Event, Stop, TextDelta, ToolUse, ToolUseStart


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-opus-4-8", max_tokens: int = 4096) -> None:
        from anthropic import Anthropic
        self.model = model
        self.max_tokens = max_tokens
        # Falsy api_key -> let the SDK resolve env/profile credentials.
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str) -> Iterator[Event]:
        cached_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        cached_tools: list[dict[str, Any]] = []
        for i, t in enumerate(tools):
            entry = dict(t)
            if i == len(tools) - 1:
                entry["cache_control"] = {"type": "ephemeral"}
            cached_tools.append(entry)
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                  "system": cached_system, "messages": messages}
        if cached_tools:
            kwargs["tools"] = cached_tools
        with self._client.messages.stream(**kwargs) as stream:
            tool_inputs: dict[int, dict[str, Any]] = {}
            tool_meta: dict[int, dict[str, str]] = {}
            for event in stream:
                et = event.type
                if et == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_meta[event.index] = {"id": block.id, "name": block.name}
                        tool_inputs[event.index] = {"_buf": ""}
                        yield ToolUseStart(id=block.id, name=block.name)
                elif et == "content_block_delta":
                    dt = getattr(event.delta, "type", None)
                    if dt == "text_delta":
                        yield TextDelta(text=event.delta.text)
                    elif dt == "input_json_delta":
                        tool_inputs.setdefault(event.index, {"_buf": ""})
                        tool_inputs[event.index]["_buf"] += event.delta.partial_json
                elif et == "content_block_stop":
                    idx = event.index
                    if idx in tool_meta:
                        buf = tool_inputs[idx]["_buf"]
                        try:
                            parsed = json.loads(buf) if buf else {}
                        except json.JSONDecodeError:
                            parsed = {}
                        yield ToolUse(id=tool_meta[idx]["id"], name=tool_meta[idx]["name"], input=parsed)
            final = stream.get_final_message()
            u = final.usage
            usage = {k: getattr(u, k, 0) for k in
                     ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
            yield Stop(reason=final.stop_reason or "end_turn", usage=usage)
```

- [ ] **Step 2: Import smoke check.**
Run: `.venv/bin/python -c "from scat.agent.providers.anthropic_api import AnthropicProvider; print('ok')"` → `ok`.

- [ ] **Step 3: Commit.**
```bash
git add scat/agent/providers/anthropic_api.py
git commit -m "feat(agent): Anthropic API streaming provider (ported)"
```

---

## Task 5: Agent runner (`scat/agent/runner.py`)

**Files:**
- Create: `scat/agent/runner.py`
- Test: `tests/test_agent_runner.py` (added in Task 11 with the FakeProvider; here just an import + `_compact_tool_result` unit test).

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_agent_runner.py  (part 1)
from scat.agent import runner

def test_compact_tool_result_short_passthrough():
    out = runner._compact_tool_result("scan_folder", {"n_images": 3, "path": "/x"})
    assert '"n_images": 3' in out

def test_compact_tool_result_truncates_large():
    big = {"blob": "x" * 20000}
    out = runner._compact_tool_result("analyze_folder", big)
    assert len(out) <= 7000 and "compacted" in out
```

- [ ] **Step 2: Run to verify fail.** → FAIL.

- [ ] **Step 3: Write `scat/agent/runner.py`.** Type verbatim (napari/vision helpers omitted; SCAT compaction keys):
```python
from __future__ import annotations
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from scat.agent.providers.base import Provider, Stop, TextDelta, ToolUse, ToolUseStart


@dataclass
class ToolResult:
    tool_use_id: str
    name: str
    output: Any
    is_error: bool = False


@dataclass
class TurnDone:
    stop_reason: str
    total_usage: dict[str, int] = field(default_factory=dict)


RunEvent = TextDelta | ToolUseStart | ToolUse | ToolResult | TurnDone

_MAX_TOOL_RESULT_CHARS = 6000
_MAX_STRING_CHARS = 1200
_MAX_LIST_ITEMS = 8
_MAX_DICT_ITEMS = 40


def _stringify_output(o: Any) -> str:
    try:
        return json.dumps(o, default=str)
    except Exception:
        return str(o)


def _truncate_text(t: str, n: int = _MAX_STRING_CHARS) -> str:
    return t if len(t) <= n else t[:n] + f"... [truncated {len(t) - n} chars]"


def _compact_value(v: Any, depth: int = 0) -> Any:
    if isinstance(v, str):
        return _truncate_text(v)
    if isinstance(v, dict):
        items = list(v.items())[:_MAX_DICT_ITEMS]
        return {k: _compact_value(val, depth + 1) for k, val in items}
    if isinstance(v, (list, tuple)):
        return [_compact_value(x, depth + 1) for x in list(v)[:_MAX_LIST_ITEMS]]
    return v


def _compact_tool_result(tool_name: str, output: Any) -> str:
    text = _stringify_output(_compact_value(output))
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text
    fb = {"tool": tool_name, "result_summary": _truncate_text(text, _MAX_TOOL_RESULT_CHARS),
          "note": "Tool result was compacted to keep the conversation within context."}
    if isinstance(output, dict):
        for k in ("path", "output_dir", "csv_path", "report_path", "n_deposits", "n_images",
                  "n_normal", "n_rod", "n_artifact", "warnings"):
            if k in output:
                fb[k] = _compact_value(output[k])
    return _stringify_output(fb)


def _context_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(w in msg for w in ("rate limit", "quota", "429", "overloaded")):
        return False
    return any(w in msg for w in ("context", "too many tokens", "maximum context", "token limit"))


class AgentRunner:
    def __init__(self, provider: Provider, system_prompt: str, max_loops: int = 40, tool_caller: Any | None = None):
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_loops = max_loops
        self.messages: list[dict[str, Any]] = []
        self._cancelled = False
        self._tool_caller = tool_caller

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self.messages = []
        self._cancelled = False

    def turn(self, user_text: str) -> Iterator[RunEvent]:
        from scat.tools import call_tool, tools_for_anthropic
        from scat.agent.providers.base import set_current_provider
        self.messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        tools_spec = tools_for_anthropic()
        total_usage: dict[str, int] = {}
        set_current_provider(self.provider)
        try:
            for _ in range(self.max_loops):
                if self._cancelled:
                    yield TurnDone("cancelled", total_usage); self._cancelled = False; return
                assistant_blocks: list[dict[str, Any]] = []
                current_text = ""
                stop_reason = "end_turn"
                attempt = 0
                while True:
                    try:
                        for event in self.provider.stream(self.messages, tools_spec, self.system_prompt):
                            if self._cancelled:
                                break
                            if isinstance(event, TextDelta):
                                current_text += event.text; yield event
                            elif isinstance(event, ToolUseStart):
                                if current_text:
                                    assistant_blocks.append({"type": "text", "text": current_text}); current_text = ""
                                yield event
                            elif isinstance(event, ToolUse):
                                assistant_blocks.append({"type": "tool_use", "id": event.id,
                                                         "name": event.name, "input": event.input})
                                yield event
                            elif isinstance(event, Stop):
                                stop_reason = event.reason
                                for k, v in (event.usage or {}).items():
                                    total_usage[k] = total_usage.get(k, 0) + int(v)
                        break
                    except Exception as exc:
                        if _context_limit_error(exc) and attempt == 0:
                            # v1: no message compaction implemented; surface and stop cleanly.
                            attempt += 1
                        yield TextDelta("\n[context/stream error: retrying is not available; please start a new chat]\n")
                        yield TurnDone("error", total_usage); return
                if self._cancelled:
                    yield TurnDone("cancelled", total_usage); self._cancelled = False; return
                if current_text:
                    assistant_blocks.append({"type": "text", "text": current_text})
                if assistant_blocks:
                    self.messages.append({"role": "assistant", "content": assistant_blocks})
                if stop_reason != "tool_use":
                    yield TurnDone(stop_reason, total_usage); return
                tool_result_blocks: list[dict[str, Any]] = []
                for block in assistant_blocks:
                    if block.get("type") != "tool_use":
                        continue
                    if self._cancelled:
                        break
                    caller = self._tool_caller or call_tool
                    try:
                        result = caller(block["name"], **block.get("input", {}))
                        tool_result_blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                                                   "content": _compact_tool_result(block["name"], result)})
                        yield ToolResult(block["id"], block["name"], result)
                    except Exception as e:
                        tool_result_blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                                                   "content": f"ERROR: {e}", "is_error": True})
                        yield ToolResult(block["id"], block["name"], str(e), is_error=True)
                done = {b["tool_use_id"] for b in tool_result_blocks}
                for block in assistant_blocks:
                    if block.get("type") != "tool_use" or block["id"] in done:
                        continue
                    tool_result_blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                                               "content": "ERROR: cancelled before execution", "is_error": True})
                    yield ToolResult(block["id"], block["name"], "cancelled before execution", is_error=True)
                if tool_result_blocks:
                    self.messages.append({"role": "user", "content": tool_result_blocks})
            yield TurnDone("max_loops", total_usage)
        finally:
            set_current_provider(None)
```

> **Note vs Imajin:** message compaction (`_compact_messages`) is intentionally deferred in v1 — SCAT folder runs are ~5 turns, well under context. The context-limit branch surfaces an error and stops cleanly instead. If long multi-folder sessions appear, port Imajin's `_compact_messages` (runner.py:415-446) in a follow-up.

- [ ] **Step 4: Run to verify pass.** → PASS (2 tests).

- [ ] **Step 5: Commit.**
```bash
git add scat/agent/runner.py tests/test_agent_runner.py
git commit -m "feat(agent): AgentRunner tool loop (ported, vision-stripped)"
```

---

## Task 6: Plain pipeline services (`scat/pipeline.py`)

**Files:**
- Create/replace: `scat/pipeline.py`
- Test: `tests/test_pipeline_parity.py` (Task 12); import test here.

These are **plain Python** (no pydantic/anthropic). They wrap existing SCAT exactly like `cli.analyze_command`.

- [ ] **Step 1: Write `scat/pipeline.py`.** Type verbatim:
```python
"""SCAT pipeline services — plain Python, no agent/LLM deps.

The single canonical implementation the CLI, the @tool adapters, and (phase 2)
the GUI all call. Nothing here imports pydantic/anthropic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd

from .detector import DepositDetector
from .classifier import ClassifierConfig
from .analyzer import Analyzer, ReportGenerator
from .config import get_timestamped_output_dir

IMAGE_GLOBS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def list_images(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    imgs: list[Path] = []
    for g in IMAGE_GLOBS:
        imgs += sorted(p.rglob(g))
    return imgs


def resolve_model_type(model_type: Optional[str], model_path: Optional[str]) -> tuple[str, Optional[str]]:
    """Canonical default: rf if a model file is available, else threshold."""
    if model_type:
        return model_type, model_path
    default_model = Path(__file__).parent.parent / "models" / "model_rf.pkl"
    if model_path:
        return "rf", model_path
    if default_model.exists():
        return "rf", str(default_model)
    return "threshold", None


@dataclass
class AnalyzeResult:
    output_dir: str
    n_images: int
    n_normal: int
    n_rod: int
    n_artifact: int
    n_failed: int
    groups: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def scan_folder_service(path: str) -> dict:
    imgs = list_images(path)
    exts = sorted({p.suffix.lower() for p in imgs})
    subdirs = sorted({p.parent.name for p in imgs if p.parent != Path(path)})
    return {"path": str(path), "n_images": len(imgs), "extensions": exts,
            "subfolders": subdirs, "sample_names": [p.name for p in imgs[:8]]}


def analyze_folder_service(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                           model_path: Optional[str] = None, min_area: int = 20, max_area: int = 10000,
                           circularity: float = 0.6, annotate: bool = True,
                           output_dir: Optional[str] = None) -> AnalyzeResult:
    from .grouping_util import build_group_metadata  # defined in Task 7
    images = list_images(path)
    if not images:
        raise ValueError(f"No images found in {path}")
    mtype, mpath = resolve_model_type(model_type, model_path)
    detector = DepositDetector(min_area=min_area, max_area=max_area)
    cfg = ClassifierConfig(model_type=mtype, circularity_threshold=circularity, model_path=mpath)
    analyzer = Analyzer(detector=detector, classifier_config=cfg)

    metadata = None
    group_by = None
    group_names: list[str] = []
    warnings: list[str] = []
    if groups:
        metadata, group_by = build_group_metadata(groups)
        group_names = sorted(set(metadata["group"]) - {"ungrouped"})
        if len(group_names) < 2:
            warnings.append(f"stats skipped: {len(group_names)} group(s) — need >=2 to compare")

    results = analyzer.analyze_batch(images, metadata=metadata)
    out = Path(output_dir) if output_dir else get_timestamped_output_dir(Path(path).parent, "results")
    reporter = ReportGenerator(out)
    reports = reporter.save_all(results, metadata, group_by)
    n_failed = sum(1 for r in results if r.n_total == 0)

    if annotate:
        from PIL import Image
        import numpy as np
        ann_dir = out / "annotated"; ann_dir.mkdir(exist_ok=True)
        for img_path, res in zip(images, results):
            if res.n_total == 0:
                continue
            arr = np.array(Image.open(img_path))
            annotated = analyzer.generate_annotated_image(arr, res.deposits, show_labels=True, skip_artifacts=True)
            Image.fromarray(annotated).save(ann_dir / f"{img_path.stem}_annotated.png")

    summary = reports["film_summary"]
    return AnalyzeResult(
        output_dir=str(out), n_images=len(results),
        n_normal=int(summary["n_normal"].sum()), n_rod=int(summary["n_rod"].sum()),
        n_artifact=int(summary["n_artifact"].sum()), n_failed=n_failed,
        groups=group_names, warnings=warnings)


def run_statistics_service(results_dir: str, group_col: str = "group") -> dict:
    from .statistics import run_comprehensive_analysis
    rd = Path(results_dir)
    film = pd.read_csv(rd / "image_summary.csv")
    deposits = pd.read_csv(rd / "all_deposits.csv") if (rd / "all_deposits.csv").exists() else None
    if group_col not in film.columns or film[group_col].dropna().nunique() < 2:
        return {"skipped": True, "reason": f"<2 groups in column '{group_col}'"}
    return run_comprehensive_analysis(film, deposits_df=deposits, group_column=group_col)


def generate_report_service(results_dir: str, statistical_results: Optional[dict] = None,
                            group_by: Optional[str] = None) -> str:
    from .report import generate_report
    rd = Path(results_dir)
    film = pd.read_csv(rd / "image_summary.csv")
    deposits = pd.read_csv(rd / "all_deposits.csv") if (rd / "all_deposits.csv").exists() else None
    return generate_report(film, output_dir=rd, deposit_data=deposits,
                           statistical_results=statistical_results, group_by=group_by, format="html")
```

- [ ] **Step 2: Import smoke check.**
Run: `.venv/bin/python -c "import scat.pipeline; print('ok')"` → `ok` (note: `grouping_util` import is lazy inside the function, so this passes before Task 7).

- [ ] **Step 3: Commit.**
```bash
git add scat/pipeline.py
git commit -m "feat: plain-Python pipeline services (canonical orchestration)"
```

---

## Task 7: Group inference (`scat/grouping_util.py` + tool in Task 9)

**Files:**
- Create: `scat/grouping_util.py` (plain helpers: `infer_groups_from_folder`, `build_group_metadata`)
- Test: `tests/test_infer_groups.py`, `tests/test_group_metadata.py`

> Plain helpers live in `scat/grouping_util.py` (no pydantic) so `scat.pipeline` and the CLI can use them; the `@tool infer_groups` in Task 9 is a thin wrapper.

- [ ] **Step 1: Write the failing tests.**
```python
# tests/test_group_metadata.py
from scat.grouping_util import build_group_metadata

def test_build_metadata_columns():
    df, group_by = build_group_metadata({"a.tif": "control", "b.tif": "treated", "c.tif": None})
    assert group_by == ["group"]
    assert set(df.columns) == {"filename", "group"}
    row = df[df.filename == "c.tif"].iloc[0]
    assert row["group"] == "ungrouped"
```
```python
# tests/test_infer_groups.py
from scat.grouping_util import infer_groups_from_folder

def _touch(d, names):
    for n in names:
        p = d / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")

def test_vocab_grouping(tmp_path):
    _touch(tmp_path, ["control_01.tif", "control_02.tif", "treated_01.tif", "treated_02.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "filename_vocab"
    assert set(r["groups"]) == {"control", "treated"}
    assert r["mapping"]["control_01.tif"] == "control"

def test_subfolder_grouping(tmp_path):
    _touch(tmp_path, ["Control/a.tif", "Control/b.tif", "Treatment/c.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "subfolder"
    assert r["mapping"]["a.tif"] == "Control"

def test_single_cohort_fallback(tmp_path):
    _touch(tmp_path, ["img_1.tif", "img_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "single_cohort" and r["groups"] == ["all"]

def test_duplicate_basename_across_subfolders_flagged(tmp_path):
    _touch(tmp_path, ["Control/x.tif", "Treatment/x.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["confidence"] == "low" and any("duplicate" in w for w in r["warnings"])
```

- [ ] **Step 2: Run to verify fail.** → FAIL.

- [ ] **Step 3: Write `scat/grouping_util.py`.** Type verbatim:
```python
"""Deterministic experimental-group inference from filenames / subfolders."""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import Optional
import pandas as pd

from .pipeline import list_images

# token (lowercased) -> canonical group label
CONDITION_VOCAB: dict[str, str] = {
    "control": "control", "ctrl": "control", "ctl": "control", "wt": "control",
    "wildtype": "control", "vehicle": "control", "veh": "control", "mock": "control",
    "treated": "treated", "treatment": "treated", "treat": "treated",
    "mutant": "mutant", "mut": "mutant", "ko": "ko", "knockout": "ko",
    "rnai": "rnai", "drug": "drug", "exp": "experimental", "test": "experimental",
}
_DELIMS = re.compile(r"[_\-\s.]+")


def _tokens(stem: str) -> list[str]:
    return [t for t in _DELIMS.split(stem.lower()) if t]


def infer_groups_from_folder(path: str) -> dict:
    """Return {mapping:{basename:group}, basis, groups, confidence, unmatched, warnings, matched_tokens}."""
    images = list_images(path)
    root = Path(path)
    names = [p.name for p in images]
    warnings: list[str] = []

    # 1) subfolder grouping (highest priority)
    sub = {p.name: p.parent.name for p in images if p.parent != root}
    if sub and len({v for v in sub.values()}) >= 2:
        dup = [n for n, c in Counter(names).items() if c > 1]
        conf = "high"
        if dup:
            conf = "low"
            warnings.append(f"duplicate basenames across subfolders: {dup[:5]} — grouping may mis-join on the 'filename' key")
        return {"mapping": sub, "basis": "subfolder", "groups": sorted(set(sub.values())),
                "confidence": conf, "unmatched": [], "warnings": warnings, "matched_tokens": []}

    # 2) filename vocabulary grouping
    mapping: dict[str, str] = {}
    matched: set[str] = set()
    unmatched: list[str] = []
    for p in images:
        g = None
        for t in _tokens(p.stem):
            if t in CONDITION_VOCAB:
                g = CONDITION_VOCAB[t]; matched.add(t); break
        if g:
            mapping[p.name] = g
        else:
            mapping[p.name] = "ungrouped"; unmatched.append(p.name)
    groups = sorted(set(mapping.values()) - {"ungrouped"})
    if len(groups) >= 2:
        conf = "high" if not unmatched else "medium"
        if unmatched:
            warnings.append(f"{len(unmatched)} file(s) matched no condition token -> 'ungrouped'")
        return {"mapping": mapping, "basis": "filename_vocab", "groups": groups,
                "confidence": conf, "unmatched": unmatched, "warnings": warnings,
                "matched_tokens": sorted(matched)}

    # 3) fallback: single cohort
    return {"mapping": {n: "all" for n in names}, "basis": "single_cohort", "groups": ["all"],
            "confidence": "low", "unmatched": [], "warnings": ["no group structure detected; single cohort (no comparison)"],
            "matched_tokens": []}


def build_group_metadata(mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    """{basename: group|None} -> (DataFrame[filename, group], ['group']). None/'' -> 'ungrouped'."""
    rows = [{"filename": f, "group": (g if g else "ungrouped")} for f, g in mapping.items()]
    return pd.DataFrame(rows), ["group"]
```

- [ ] **Step 4: Run to verify pass.**
Run: `.venv/bin/python -m pytest tests/test_infer_groups.py tests/test_group_metadata.py -q` → PASS (5 tests).

- [ ] **Step 5: Commit.**
```bash
git add scat/grouping_util.py tests/test_infer_groups.py tests/test_group_metadata.py
git commit -m "feat: deterministic infer_groups + build_group_metadata"
```

---

## Task 8: SCAT tools + barrel (`scat/tools/*.py`)

**Files:**
- Create: `scat/tools/scan.py`, `scat/tools/grouping.py`, `scat/tools/pipeline_tools.py`, `scat/tools/__init__.py`
- Test: `tests/test_tools_registered.py`

- [ ] **Step 1: Write `scat/tools/scan.py`.**
```python
from scat.agent.registry import tool
from scat.pipeline import scan_folder_service


@tool(description="List images in a folder and summarize filename structure. Call this first.")
def scan_folder(path: str) -> dict:
    """List images in a folder and summarize extensions/subfolders/sample names."""
    return scan_folder_service(path)
```

- [ ] **Step 2: Write `scat/tools/grouping.py`.**
```python
from scat.agent.registry import tool
from scat.grouping_util import infer_groups_from_folder


@tool(description="Infer experimental groups from filenames/subfolders. Returns {basename: group} plus confidence. State the mapping to the user before analyzing; if confidence is 'low', recommend confirmation or a metadata CSV.")
def infer_groups(path: str) -> dict:
    """Infer experimental groups from filenames or subfolders (deterministic)."""
    return infer_groups_from_folder(path)
```

- [ ] **Step 3: Write `scat/tools/pipeline_tools.py`.**
```python
from typing import Optional
from dataclasses import asdict
from scat.agent.registry import tool
from scat.pipeline import analyze_folder_service, run_statistics_service, generate_report_service


@tool(description="Detect + classify deposits across a folder and write CSVs/JSON/annotations to a timestamped results dir. Pass groups={basename: group} from infer_groups to enable group comparison.")
def analyze_folder(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                   min_area: int = 20, max_area: int = 10000, circularity: float = 0.6,
                   annotate: bool = True) -> dict:
    """Run detection + classification over a folder; returns counts + output dir."""
    return asdict(analyze_folder_service(path, groups=groups, model_type=model_type,
                                         min_area=min_area, max_area=max_area,
                                         circularity=circularity, annotate=annotate))


@tool(description="Run group statistics on a completed results dir. No-ops (with a reason) if <2 groups.")
def run_statistics(results_dir: str, group_col: str = "group") -> dict:
    """Group comparison statistics over a results directory."""
    return run_statistics_service(results_dir, group_col=group_col)


@tool(description="Generate the HTML report for a results dir. Pass the statistics dict if available.")
def generate_report(results_dir: str, statistical_results: Optional[dict] = None,
                    group_by: Optional[str] = None) -> dict:
    """Build the self-contained HTML report; returns its path."""
    return {"report_path": generate_report_service(results_dir, statistical_results, group_by)}
```

- [ ] **Step 4: Write `scat/tools/__init__.py`** (barrel — registration side-effect + re-exports):
```python
"""Import every tool module so @tool registers it, and re-export registry helpers."""
from scat.agent.registry import call_tool, tools_for_anthropic, iter_tools, get_tool  # noqa: F401
from . import scan, grouping, pipeline_tools  # noqa: F401  (registration side-effects)

__all__ = ["call_tool", "tools_for_anthropic", "iter_tools", "get_tool"]
```

- [ ] **Step 5: Write the test.**
```python
# tests/test_tools_registered.py
import scat.tools as tools

def test_all_tools_registered():
    names = {e.name for e in tools.iter_tools()}
    assert {"scan_folder", "infer_groups", "analyze_folder", "run_statistics", "generate_report"} <= names

def test_tool_specs_have_schemas():
    specs = {s["name"]: s for s in tools.tools_for_anthropic()}
    assert specs["analyze_folder"]["input_schema"]["properties"]["path"]["type"] == "string"
```

- [ ] **Step 6: Run to verify pass.**
Run: `.venv/bin/python -m pytest tests/test_tools_registered.py -q` → PASS.

- [ ] **Step 7: Commit.**
```bash
git add scat/tools/ tests/test_tools_registered.py
git commit -m "feat(agent): SCAT @tool wrappers + registration barrel"
```

---

## Task 9: System prompt (`scat/agent/prompts.py`)

**Files:**
- Create: `scat/agent/prompts.py`

- [ ] **Step 1: Write `scat/agent/prompts.py`.**
```python
SYSTEM_PROMPT = """\
You are SCAT's analysis agent. SCAT detects and classifies Drosophila excreta \
deposits (Normal / ROD / Artifact) in images and produces statistics and an HTML report.

Bias to action: when the user names a folder, run the whole pipeline to completion. \
Do NOT ask clarifying questions unless something is genuinely ambiguous.

Pipeline recipe for "analyze this folder":
1. scan_folder(path) — confirm images exist and see the filename structure.
2. infer_groups(path) — infer experimental groups. STATE the inferred {file: group} \
   mapping to the user in plain language before analyzing. If the result's confidence \
   is "low", say so and recommend the user confirm or supply a metadata CSV — this is \
   the ONE case where you may pause for confirmation.
3. analyze_folder(path, groups=<the mapping from step 2>) — detect + classify + write results.
4. If the analysis reports >=2 groups, run_statistics(results_dir, group_col="group").
5. generate_report(results_dir, statistical_results=<from step 4 if any>, group_by="group").
6. Report to the user: total deposits, Normal/ROD/Artifact counts, the groups used, and \
   the paths to the results dir and report.html.

Statistics guidance: you assert the design. State whether groups are independent or paired \
(default independent). For 3+ groups rely on the omnibus test plus a multiplicity-corrected \
post-hoc — never uncorrected pairwise. Relay any warnings (small n, non-normal, stats skipped) \
rather than a bare p-value.

Never invent group names beyond what the filename/subfolder structure supports. Treat any \
injected session/progress context as authoritative — do not re-analyze images already done.
"""
```

- [ ] **Step 2: Import smoke check.**
Run: `.venv/bin/python -c "from scat.agent.prompts import SYSTEM_PROMPT; print(len(SYSTEM_PROMPT) > 200)"` → `True`.

- [ ] **Step 3: Commit.**
```bash
git add scat/agent/prompts.py
git commit -m "feat(agent): SCAT system prompt"
```

---

## Task 10: Durable context ledger (`scat/agent/context.py`)

**Files:**
- Create: `scat/agent/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_context.py
import pandas as pd
from scat.agent.context import summarize_results_dir

def test_summary_reports_counts(tmp_path):
    pd.DataFrame({"filename": ["a.tif", "b.tif"], "n_total": [3, 0],
                  "n_normal": [2, 0], "n_rod": [1, 0], "n_artifact": [0, 0],
                  "group": ["control", "treated"]}).to_csv(tmp_path / "image_summary.csv", index=False)
    s = summarize_results_dir(str(tmp_path))
    assert "2 image" in s and "control" in s and "treated" in s
```

- [ ] **Step 2: Run to verify fail.** → FAIL.

- [ ] **Step 3: Write `scat/agent/context.py`.**
```python
from __future__ import annotations
from pathlib import Path
import pandas as pd


def summarize_results_dir(results_dir: str) -> str:
    """One-paragraph ledger of a results dir for injection into the system prompt."""
    rd = Path(results_dir)
    csv = rd / "image_summary.csv"
    if not csv.exists():
        return f"No completed results in {results_dir} yet."
    df = pd.read_csv(csv)
    n = len(df)
    groups = sorted(set(df["group"].dropna())) if "group" in df.columns else []
    tot = int(df["n_total"].sum()) if "n_total" in df.columns else 0
    parts = [f"Results dir {rd.name}: {n} image(s) analysed, {tot} deposits total."]
    if groups:
        parts.append(f"Groups present: {', '.join(groups)}.")
    return " ".join(parts)
```

> The chat driver (Task 12) may append `summarize_results_dir(dir)` to the system prompt on later turns once a results dir exists. Kept minimal and disk-only (no Qt), per spec §7.

- [ ] **Step 4: Run to verify pass.** → PASS.

- [ ] **Step 5: Commit.**
```bash
git add scat/agent/context.py tests/test_context.py
git commit -m "feat(agent): durable results-dir ledger"
```

---

## Task 11: Subscription bridge (`scat/agent/claude_subscription.py`)

**Files:**
- Create: `scat/agent/claude_subscription.py`
- Test: `tests/test_subscription_bridge.py` (guarded/skipped without SDK+login)

- [ ] **Step 1: Copy + adapt from Imajin.**
Copy `/home/lab/Imajin/src/imajin/agent/providers/claude_agent.py` to `scat/agent/claude_subscription.py`, then apply exactly:
- Imports: `from scat.agent.providers.base import TextDelta, ToolUseStart, ToolUse`; `from scat.agent.runner import ToolResult, TurnDone, _compact_tool_result`; `from scat.tools import iter_tools, call_tool`.
- `_MCP_SERVER = "scat"`; reword the `_DISALLOWED_BUILTINS` comment to "the analysis agent".
- Keep verbatim (they have no imajin deps): `_map_usage`, `_flatten_tool_result`, `_strip_ns`, `subscription_available`, `_force_subscription_env`, `_translate_message`, `_make_handler`, `_build_server`, `_build_options`, `_ensure_loop`, `_ensure_connected`, `_adrive_turn`, `turn`, `cancel`, `reset`, `close`, `_disconnect`, `__init__`.
- In `_build_server`, drop any `.subagent` filter — SCAT has no subagents; keep `if not e.llm: continue`.
The key mechanisms (verbatim from the recipe) are:
```python
_MCP_SERVER = "scat"; _TOOL_PREFIX = f"mcp__{_MCP_SERVER}__"
_AUTH_ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"); _ENV_LOCK = threading.Lock()
_DISALLOWED_BUILTINS = ["Bash","Read","Write","Edit","NotebookEdit","WebFetch","WebSearch"]

def subscription_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("claude_agent_sdk") is None: return False, "SDK missing"
    if shutil.which("claude") is None: return False, "claude not found"
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"): return True, None
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    creds = (Path(cfg) if cfg else Path.home()/".claude")/".credentials.json"
    return (True, None) if creds.exists() else (False, "not logged in")

@contextmanager
def _force_subscription_env():
    with _ENV_LOCK:
        saved = {k: os.environ.pop(k) for k in _AUTH_ENV_KEYS if k in os.environ}
        try: yield
        finally: os.environ.update(saved)
```
`ClaudeAgentRunner` (rename to `ClaudeSubscriptionRunner`) `_build_server()` builds one `sdk.tool(e.name, e.description, e.json_schema)(handler)` per registered tool, `sdk.create_sdk_mcp_server(name="scat", ...)`, `permission_mode="dontAsk"`, and bridges the SDK async loop → sync `turn()` via a `queue.Queue` + sentinel (full bodies in `/tmp/claude-1000/.../scratchpad/porting.txt` §claude_agent, or read the Imajin source). **Provenance parity:** the handler wraps `call_tool` which already records provenance (Task 2) — no extra work.

- [ ] **Step 2: Rename the class** to `ClaudeSubscriptionRunner` throughout the file (and keep `name = "claude-agent"` attr or set `"subscription"`).

- [ ] **Step 3: Write the guarded test.**
```python
# tests/test_subscription_bridge.py
import pytest
from scat.agent.claude_subscription import subscription_available, ClaudeSubscriptionRunner

def test_subscription_probe_returns_tuple():
    ok, reason = subscription_available()
    assert isinstance(ok, bool)

@pytest.mark.skipif(not subscription_available()[0], reason="no claude subscription login")
def test_bridge_round_trips_a_tool():
    import scat.tools  # register tools
    runner = ClaudeSubscriptionRunner(model="claude-opus-4-8",
                                      system_prompt="You are a test. Call scan_folder on the path you are given, then stop.")
    got = []
    for ev in runner.turn("scan the folder /nonexistent-xyz and report the image count"):
        got.append(type(ev).__name__)
    runner.close()
    assert "TurnDone" in got  # bridge completed a turn without crashing
```

- [ ] **Step 4: Run.**
Run: `.venv/bin/python -m pytest tests/test_subscription_bridge.py -q`
Expected: 1 pass (probe) + 1 skip (or pass if logged in). Also import check:
`.venv/bin/python -c "from scat.agent.claude_subscription import subscription_available; print(subscription_available())"`

- [ ] **Step 5: Commit.**
```bash
git add scat/agent/claude_subscription.py tests/test_subscription_bridge.py
git commit -m "feat(agent): Claude subscription bridge (ported)"
```

---

## Task 12: Backend selection + FakeProvider e2e test

**Files:**
- Create: `scat/agent/backend.py`
- Test: `tests/test_backend.py`, extend `tests/test_agent_runner.py` with FakeProvider e2e

- [ ] **Step 1: Write `scat/agent/backend.py`.**
```python
from __future__ import annotations
import os
from scat.agent.prompts import SYSTEM_PROMPT


def build_runner(backend: str = "auto", model: str = "claude-opus-4-8", max_loops: int = 40):
    """Return (runner, description). Prefers subscription unless overridden."""
    import scat.tools  # ensure tools are registered
    from scat.agent.claude_subscription import subscription_available, ClaudeSubscriptionRunner
    from scat.agent.runner import AgentRunner
    from scat.agent.providers.anthropic_api import AnthropicProvider

    sub_ok, sub_reason = subscription_available()
    want = backend
    if want == "auto":
        want = "subscription" if sub_ok else "api"

    if want == "subscription":
        if not sub_ok:
            raise RuntimeError(f"subscription backend requested but unavailable: {sub_reason}")
        runner = ClaudeSubscriptionRunner(model=model, system_prompt=SYSTEM_PROMPT, max_turns=max_loops)
        return runner, f"Claude subscription (no API charges), model={model}"

    if want == "api":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No backend available. Either log in to Claude (`claude` CLI) for the "
                "subscription path, or set ANTHROPIC_API_KEY for the API path.")
        runner = AgentRunner(AnthropicProvider(api_key=key, model=model), SYSTEM_PROMPT, max_loops=max_loops)
        return runner, f"ANTHROPIC_API_KEY (requests are billed), model={model}"

    raise ValueError(f"unknown backend: {backend}")
```

- [ ] **Step 2: Write `tests/test_backend.py`.**
```python
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
```

- [ ] **Step 3: Extend `tests/test_agent_runner.py` with the FakeProvider e2e.**
```python
# tests/test_agent_runner.py  (part 2 — appended)
from scat.agent.providers.base import TextDelta, ToolUse, Stop
from scat.agent.runner import AgentRunner, ToolResult, TurnDone


class FakeProvider:
    """Scripts a two-round conversation: call scan_folder, then finish. No network."""
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
    import scat.tools  # register tools
    runner = AgentRunner(FakeProvider(), "sys", max_loops=5)
    events = list(runner.turn(str(synth_dir)))
    kinds = [type(e).__name__ for e in events]
    assert "ToolResult" in kinds and kinds[-1] == "TurnDone"
    tr = next(e for e in events if isinstance(e, ToolResult))
    assert tr.name == "scan_folder" and tr.output["n_images"] > 0
    assert events[-1].stop_reason == "end_turn"
```
(`synth_dir` fixture already exists in `tests/conftest.py` from the bugfix suite.)

- [ ] **Step 4: Run.**
Run: `.venv/bin/python -m pytest tests/test_backend.py tests/test_agent_runner.py -q` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add scat/agent/backend.py tests/test_backend.py tests/test_agent_runner.py
git commit -m "feat(agent): backend selection + FakeProvider e2e runner test"
```

---

## Task 13: CLI `chat` + reimplement `analyze` on services

**Files:**
- Modify: `scat/cli.py`
- Test: `tests/test_pipeline_parity.py` (anti-drift gate)

- [ ] **Step 1: Add `chat` + rewrite `analyze_command` in `scat/cli.py`.**
Replace the body of `analyze_command` so it calls the services (collapsing the duplicate). New functions:
```python
def analyze_command(args):
    from .pipeline import analyze_folder_service, run_statistics_service, generate_report_service
    from .grouping_util import build_group_metadata
    import pandas as pd
    groups = None
    if args.metadata:
        meta = pd.read_csv(args.metadata)
        col = args.group_by or (meta.columns[1] if len(meta.columns) > 1 else None)
        if col:
            groups = dict(zip(meta["filename"], meta[col]))
    res = analyze_folder_service(args.input, groups=groups, model_type=args.model_type,
                                 model_path=args.model_path, min_area=args.min_area,
                                 max_area=args.max_area, circularity=args.threshold,
                                 annotate=args.annotate, output_dir=args.output)
    print(f"Analyzed {res.n_images} images -> {res.output_dir}")
    print(f"  Normal={res.n_normal} ROD={res.n_rod} Artifact={res.n_artifact} failed={res.n_failed}")
    stats = None
    if args.stats and len(res.groups) >= 2:
        stats = run_statistics_service(res.output_dir, group_col="group")
    if args.stats and len(res.groups) < 2:
        print(f"  stats skipped: {len(res.groups)} group(s)")
    if args.report:
        path = generate_report_service(res.output_dir, statistical_results=stats, group_by="group")
        print(f"  report: {path}")


def chat_command(args):
    from .agent.backend import build_runner
    from .agent.provenance import start_session, set_driver
    from .agent.runner import TextDelta, ToolUse, ToolResult, TurnDone
    start_session(driver="cli-chat"); set_driver("cli-chat")
    runner, desc = build_runner(backend=args.backend, model=args.model)
    print(f"[backend] {desc}\nType a request (Ctrl-D to exit).")
    try:
        while True:
            try:
                text = input("\n> ")
            except EOFError:
                break
            for ev in runner.turn(text):
                if isinstance(ev, TextDelta):
                    print(ev.text, end="", flush=True)
                elif isinstance(ev, ToolUse):
                    print(f"\n  [tool] {ev.name}({ev.input})", flush=True)
                elif isinstance(ev, ToolResult):
                    tag = "ERR" if ev.is_error else "ok"
                    print(f"\n  [result:{tag}] {ev.name}", flush=True)
                elif isinstance(ev, TurnDone):
                    print(f"\n[turn done: {ev.stop_reason}]", flush=True)
    finally:
        close = getattr(runner, "close", None)
        if close:
            close()
```
Add args (in `main()`): give `analyze` a `--report` flag (default from config), and register the `chat` subparser:
```python
    ap.add_argument('--report', action='store_true', help='Generate HTML report')
    ...
    cp = subparsers.add_parser('chat', help='Conversational agent (analyze a folder by asking)')
    cp.add_argument('--backend', default='auto', choices=['auto', 'subscription', 'api'])
    cp.add_argument('--model', default='claude-opus-4-8')
    cp.set_defaults(func=chat_command)
```

- [ ] **Step 2: Write the parity/anti-drift test.**
```python
# tests/test_pipeline_parity.py
import pandas as pd
from scat.pipeline import analyze_folder_service

def test_service_matches_direct_pipeline(synth_dir, tmp_path):
    """The new service must produce the same per-image summary as the raw Analyzer path."""
    from scat.detector import DepositDetector
    from scat.classifier import ClassifierConfig
    from scat.analyzer import Analyzer, ReportGenerator
    from scat.pipeline import list_images, resolve_model_type
    imgs = list_images(str(synth_dir))
    mtype, mpath = resolve_model_type(None, None)
    az = Analyzer(detector=DepositDetector(), classifier_config=ClassifierConfig(model_type=mtype, model_path=mpath))
    direct = ReportGenerator(tmp_path / "direct")
    direct.save_all(az.analyze_batch(imgs))
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "svc"), annotate=False)
    a = pd.read_csv(tmp_path / "direct" / "all_deposits.csv")
    b = pd.read_csv(res.output_dir + "/all_deposits.csv")
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 3: Run.**
Run: `.venv/bin/python -m pytest tests/test_pipeline_parity.py -q` → PASS.
Then smoke the CLI: `.venv/bin/python -m scat.cli analyze <synth-dir> -o /tmp/x --report` runs without error.

- [ ] **Step 4: Commit.**
```bash
git add scat/cli.py tests/test_pipeline_parity.py
git commit -m "feat(cli): add chat subcommand; reimplement analyze on pipeline services"
```

---

## Task 14: Config — `agent` section + dead-key cleanup + packaging guard

**Files:**
- Modify: `scat/config.py`
- Test: `tests/test_core_imports_without_agent.py`

- [ ] **Step 1: Edit `DEFAULT_CONFIG` in `scat/config.py`.**
Delete these dead keys: `last_metadata_path` (top level), `detection.sensitive_mode`, `detection.edge_margin`, `analysis.group_by`. Add:
```python
    "agent": {
        "backend": "auto",   # auto | subscription | api
        "model": "claude-opus-4-8",
        "max_loops": 40,
    },
```
(Secrets are NOT stored here — `ANTHROPIC_API_KEY` comes from the environment.)

- [ ] **Step 2: Verify the deleted keys are truly unused.**
Run: `grep -rn "sensitive_mode\|last_metadata_path\|analysis.group_by\|detection.edge_margin" scat/ | grep -v "def \|# "`
Expected: no *config-read* references remain (detector still takes `sensitive_mode`/`edge_margin` as constructor args — those stay; only the config keys go). If a `config.get("detection.sensitive_mode")` exists, leave that key. (Per the spec sweep these are unread, but verify before deleting.)

- [ ] **Step 3: Write the packaging-guard test.**
```python
# tests/test_core_imports_without_agent.py
import importlib, subprocess, sys

def test_core_imports_do_not_require_agent_extras():
    # scat, analyzer, pipeline, cli must import without pydantic/anthropic present at import time.
    code = "import scat, scat.analyzer, scat.pipeline, scat.cli, scat.grouping_util; print('ok')"
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "ok"

def test_pipeline_module_has_no_top_level_agent_import():
    src = open("scat/pipeline.py").read()
    assert "import pydantic" not in src and "import anthropic" not in src
    assert "from scat.agent" not in src and "scat.tools" not in src  # services never import the agent layer
```

- [ ] **Step 4: Run.**
Run: `.venv/bin/python -m pytest tests/test_core_imports_without_agent.py -q` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add scat/config.py tests/test_core_imports_without_agent.py
git commit -m "chore(config): add agent section; drop 4 dead keys; packaging guard"
```

---

## Task 15: Full-suite verification + end-to-end run

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite.**
Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (the 21 pre-existing + the new agent tests; subscription-bridge test may skip).

- [ ] **Step 2: End-to-end CLI analyze (no LLM) on synthetic images.**
Generate synth images (reuse `tests/conftest.py`'s generator or the bugfix `make_synth.py`), then:
Run: `.venv/bin/python -m scat.cli analyze <synth-dir> -m <groups.csv> --group-by group --stats --report`
Expected: prints counts, writes a timestamped results dir with `image_summary.csv` (group column), `report.html`.

- [ ] **Step 3: End-to-end chat (if a backend is available).**
Run: `.venv/bin/python -m scat.cli chat` then type `analyze <synth-dir>`.
Expected: `[backend] ...` line; the agent calls scan_folder → infer_groups → analyze_folder → (stats) → generate_report; prints the mapping and final paths. If no backend, `build_runner` prints the clear both-options error — that is correct behavior, not a failure.

- [ ] **Step 4: Final commit (if any stragglers).**
```bash
git add -A && git commit -m "test(agent): full v1 verification green" || echo "nothing to commit"
```

---

## Self-review notes (author)

- **Spec coverage:** providers/base+anthropic (T3,4), runner (T5), registry (T2), provenance (T1), subscription bridge (T11), backend selection+observability (T12), infer_groups+seam (T7), plain services + @tool thin adapters (T6,8), system prompt (T9), context ledger (T10), CLI chat + analyze collapse (T13), agent config + dead keys + packaging guard (T14), parity gate + FakeProvider + bridge contract test (T7,12,11,13). Cost/cancel controls: `max_loops` backstop + cooperative `_cancelled` checks are in the runner; per-image progress + wall-clock timeout are noted as follow-ups (not blocking v1 — folder = 1 tool call).
- **Deferred from spec (documented):** message compaction (`_compact_messages`) — v1 surfaces context-limit and stops; GUI dock, result bundles, Ollama — phase 2/later.
- **Type consistency:** `build_group_metadata` (grouping_util) is the single metadata builder; services take `groups: dict`; `AnalyzeResult` fields match the `asdict()` returned by the `analyze_folder` tool.
