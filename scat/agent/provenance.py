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
