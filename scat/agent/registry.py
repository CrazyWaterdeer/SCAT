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
