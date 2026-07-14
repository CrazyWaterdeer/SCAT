# T1.1 — Chat-dock per-image progress + cooperative cancel

**Branch:** `feat/hardening` · **Roadmap:** T1.1 · **Spec:** §10.1.
Fixes the most visible defect in the shipped chat dock: a real folder analysis is ONE coarse
`analyze_folder` tool call, so the dock shows one line then looks frozen until the whole batch
finishes, and **Stop does not halt the batch** (cancel is only checked at tool/loop boundaries,
never inside `analyze_batch`).

## Execution model (verified)
- `Analyzer.analyze_batch` calls `progress_callback(current, total)` on the **caller's thread**
  in both modes — sequential (before each `analyze_image`) and parallel (in the `as_completed`
  collector loop; the pool only runs per-image compute). analyzer.py:171-233.
- The tool runs on the caller thread of `runner.turn()`: the **ChatWorker QThread** (GUI) or,
  for the subscription backend, the **default ThreadPoolExecutor** thread
  (`loop.run_in_executor(None, caller)`, claude_subscription.py). Either way it is NOT the GUI
  main thread.
- One agent turn is active at a time (dock enforces the single-active-turn contract; CLI chat is
  a sequential input loop).

## Decision — a process-global channel, not a contextvar
The spec suggests a `report_progress` contextvar, but contextvars do **not** propagate across the
`threading.Thread` / executor boundaries the tool crosses, so a contextvar set by the harness
would be invisible inside `analyze_batch`. Given the single-active-turn invariant, a small
**process-global, thread-safe channel** is simpler and correct (and matches how `provenance`
already uses module globals). Gate everything on an active run so the non-agent GUI Run button
(which never opens the channel) is completely unaffected.

## 1. New core module `scat/progress.py` (no agent/GUI deps)
```python
class AnalysisCancelled(Exception): ...
_sink = None            # callable(current:int, total:int, note:str) | None
_cancelled = False
def report_progress(current, total, note=""):     # no-op unless a run is active
    s = _sink
    if s:
        try: s(current, total, note)
        except Exception: pass
def raise_if_cancelled():                          # only fires inside an active run
    if _sink is not None and _cancelled:
        raise AnalysisCancelled("analysis cancelled")
def request_cancel():                              # called by the GUI Stop
    global _cancelled; _cancelled = True
@contextmanager
def run_progress(sink):                            # harness wraps the turn with this
    global _sink, _cancelled
    _sink, _cancelled = sink, False
    try: yield
    finally: _sink, _cancelled = None, False
```
`_sink is not None` is the "active run" gate: `raise_if_cancelled` is inert for the GUI Run path
(no sink) and a stray `request_cancel()` is reset to `False` by the next `run_progress`.

## 2. `analyze_folder_service` (pipeline.py) — compose the callback
Wrap the `progress_callback` handed to `analyze_batch` so it also drives the ambient channel:
```python
from . import progress as _pg
def _cb(c, t):
    if progress_callback: progress_callback(c, t)   # GUI Run button's Qt bar (unchanged)
    _pg.report_progress(c, t, "analyzing images")    # ambient (agent turn) progress
    _pg.raise_if_cancelled()                         # cooperative cancel at the next image
results = analyzer.analyze_batch(images, metadata=metadata, progress_callback=_cb,
                                 parallel=parallel, max_workers=max_workers)
```
`AnalysisCancelled` propagates out of `analyze_batch` (parallel: after in-flight images finish via
`executor.__exit__`'s `shutdown(wait=True)` — cooperative, not preemptive) → out of the service →
the tool → the runner records a tool error and the turn ends (runner `_cancelled` is also set by
Stop). No output dir is written on cancel (save_all runs after analyze_batch) — acceptable.

## 3. ChatWorker + dock (chat_widget.py)
- `ChatWorker` gains `progress = Signal(int, int, str)`. In `run()`, wrap the turn:
  `with run_progress(lambda c, t, note="": self.progress.emit(c, t, note)): for ev in runner.turn(...)`.
  The sink emits a Qt signal (thread-safe from any thread) → queued to the GUI thread.
- Dock connects `worker.progress` → a slot that shows "analyzing 142/500" in `self.status`
  (cleared on `finished_turn`).
- `_stop()` additionally calls `scat.progress.request_cancel()` (lazy import) alongside
  `runner.cancel()`, so Stop halts the in-progress batch at the next image.

## 4. CLI chat (cli.py) — render progress (cancel stays Ctrl-C)
Wrap the per-turn `for ev in runner.turn(text)` with `run_progress(lambda c,t,note="": print(...))`
so a CLI chat shows "142/500" during the compute turn. Cancel in the CLI is out of scope here
(no Stop affordance); the GUI is where Stop lives.

## 5. Verification
- `tests/test_progress.py`: (a) `report_progress`/`raise_if_cancelled` are inert with no active
  run; (b) inside `run_progress(sink)`, sink receives calls and `request_cancel()` makes
  `raise_if_cancelled()` raise `AnalysisCancelled`; (c) the flag resets on context exit.
- Service integration: `analyze_folder_service` under `run_progress(sink)` calls the sink with
  ascending (c, total); with `request_cancel()` pre-set it raises `AnalysisCancelled` and writes
  no output dir. Parity gate stays green (defaults: no sink, `_cb` still forwards the explicit
  callback and the ambient calls are no-ops).
- Dock: extend a chat-widget test to assert `ChatWorker.progress` fires (drive an AgentRunner +
  FakeProvider whose tool calls the service under a sink), and that `_stop()` calls request_cancel.
- Full suite green; a real subscription turn still round-trips.

## Codex review — incorporated
A Codex read-only pass (threading verdict: the process-global IS visible from both tool-exec
sites; the contextvar argument is sound) flagged 6 real gaps, all folded in:
- **F1 (parallel cancel drained the whole batch):** `analyze_batch` submitted every image up
  front and `with ThreadPoolExecutor` exited via `shutdown(wait=True)`, so Stop waited for the
  full batch. Rewrote the parallel branch to `shutdown(wait=False, cancel_futures=True)` on a
  callback exception → queued images are dropped, in-flight ones finish (cooperative).
- **F2 (cancel not terminal):** the API runner treated `AnalysisCancelled` as a generic tool
  error and could loop. It now special-cases it: records a valid `tool_result`, sets
  `_cancelled`, and the turn ends — no retry.
- **F4/F5 (cross-feature leak / no ownership):** made ambient participation **opt-in** via
  `analyze_folder_service(ambient_progress=…)`. Only the agent `@tool analyze_folder` passes
  `True`; the GUI Run button never does, so a chat turn's open sink can't report into or cancel
  a concurrent GUI Run analysis. (Single-active-turn still holds for the one ambient user.)
- **F6:** `ChatDockWidget.shutdown()` now also `request_cancel()`s so app-close halts a blocking
  batch.
- **F7:** added a parallel cancel test (1 worker, cancel armed → only the first image runs) and
  a GUI-run-isolation test.
- **F3 (subscription handler swallows the exception):** left as-is — the subscription Stop path
  works via `client.interrupt()` (ends the response) + `request_cancel()` (stops the batch);
  documented as the backend divergence.

## Risks
- **R1 concurrent runs** — the channel is process-global; safe only under single-active-turn
  (dock-enforced, CLI sequential). Documented; not for concurrent agent sessions.
- **R2 parallel cancel latency** — cancel halts at the next completion, and `shutdown(wait=True)`
  lets in-flight images finish. Cooperative by design (spec §10.1). Bounded by worker count.
- **R3 GUI Run button** — must stay unaffected: it passes an explicit callback and never opens
  `run_progress`, so ambient report/cancel are inert. Covered by the parity + gui_slimdown tests.
