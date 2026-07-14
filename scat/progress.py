"""Process-global progress + cooperative-cancel channel for long analysis turns.

Why a module global rather than a contextvar: the analysis tool runs on a *different* thread
from the harness that wants progress/cancel — the ChatWorker QThread (API backend) or the SDK's
default ThreadPoolExecutor (subscription backend) — and contextvars do not propagate across those
boundaries. A plain module global is visible from any thread. It is safe because exactly one agent
turn is ever active at a time (the chat dock enforces the single-active-turn contract; CLI chat is
a sequential input loop). It is NOT safe for concurrent agent sessions in one process.

The GUI "Run" button never opens a run (`run_progress`), so `report_progress` / `raise_if_cancelled`
are fully inert on that path — it keeps using its own explicit `progress_callback` and Qt bar.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

__all__ = [
    "AnalysisCancelled", "report_progress", "raise_if_cancelled",
    "request_cancel", "run_progress", "is_active",
]


class AnalysisCancelled(Exception):
    """Raised by raise_if_cancelled() when the active run has been asked to stop."""


_lock = threading.Lock()      # guards installing/clearing the sink
_sink = None                  # callable(current:int, total:int, note:str) | None
_cancelled = False            # set by request_cancel(); reset when a run starts/ends


def report_progress(current: int, total: int, note: str = "") -> None:
    """Report per-item progress to the active run's sink. No-op when no run is active."""
    s = _sink
    if s is not None:
        try:
            s(current, total, note)
        except Exception:
            pass  # never let a UI sink break the analysis


def raise_if_cancelled() -> None:
    """Raise AnalysisCancelled if the active run has been cancelled. Inert with no active run,
    so the non-agent GUI Run path (which opens no run) is never affected."""
    if _sink is not None and _cancelled:
        raise AnalysisCancelled("analysis cancelled")


def request_cancel() -> None:
    """Ask the active run to stop at the next item (called by the GUI Stop button)."""
    global _cancelled
    _cancelled = True


def is_active() -> bool:
    return _sink is not None


@contextmanager
def run_progress(sink):
    """Install `sink` as the active run's progress sink and arm cancellation for its duration.
    Resets the cancel flag on entry (so a stray earlier request_cancel can't leak in) and clears
    both on exit."""
    global _sink, _cancelled
    with _lock:
        _sink, _cancelled = sink, False
    try:
        yield
    finally:
        with _lock:
            _sink, _cancelled = None, False
