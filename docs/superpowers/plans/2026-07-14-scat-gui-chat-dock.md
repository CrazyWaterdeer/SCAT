# SCAT GUI Chat Dock — Implementation Plan

**Branch:** `feat/ai-agent` · **Scope:** the remaining phase-2 GUI item (spec §13) — embed the
conversational agent as a PySide6 dock. The slimdown half is done
([[../plans/2026-07-14-scat-gui-phase2-slimdown]]). **Process:** [[feature-dev-workflow]].

## 0. Goal
A dockable "Assistant" panel in the MainWindow that drives the same `AgentRunner` /
`ClaudeSubscriptionRunner` the CLI `chat` uses, streaming its events into a conversation view.
Non-goals: no new tool marshalling (SCAT tools are pure services), no result-bundle/resume.

## 1. Runner contract (verified in code)
Both backends expose the SAME sync interface — `build_runner(backend, model, max_loops) ->
(runner, desc)`:
- `runner.turn(text) -> Iterator[RunEvent]` — **synchronous generator** (the subscription
  runner bridges its own asyncio loop via an internal queue + daemon thread; `turn()` blocks
  the *caller* draining that queue). Events: `TextDelta(.text)`, `ToolUseStart(.id,.name)`,
  `ToolUse(.id,.name,.input)`, `ToolResult(.tool_use_id,.name,.output,.is_error)`,
  `TurnDone(.stop_reason,.total_usage)`.
- `runner.cancel()` — cooperative (subscription also `client.interrupt()`).
- `runner.reset()` — clears conversation state.
- `runner.close()` — teardown (subscription joins its loop thread). Present on subscription;
  optional on API (`getattr(runner,'close',None)`, as the CLI does).

**Threading model:** run `for ev in runner.turn(text)` on a `QThread` (ChatWorker); emit a Qt
signal per event; render on the main thread (auto queued connection). Works for both backends
because both `turn()`s are sync generators. One turn at a time (disable input while running).
Tools are pure-compute services → **no Qt-main-thread marshalling needed** (only rendering is
marshalled, via the signal). provenance uses **process-global** mutable module state (not
contextvars); with a single dock running one turn at a time this is fine, but it is NOT safe for
concurrent agent sessions (they would stomp each other's session id/log) — out of scope here.
[Codex-corrected: the runner's `cancel/reset/close/turn` share unsynchronised fields too, so the
GUI must enforce a strict single-active-turn contract — which `_send`/`_new` guards do.]

## 2. Packaging guard (critical)
Core `import scat.main_gui` and constructing `MainWindow` must work **without** the `[agent]`
extra; only actually chatting needs it.
- New module **`scat/agent/chat_widget.py`** imports **only PySide6** at top. It renders events
  by `type(ev).__name__` + duck-typed attributes (no `from scat.agent...` import of the event
  classes), and imports `build_runner` + `start_session` **lazily inside `_ensure_runner()`**
  (first send). So importing/constructing the widget never pulls pydantic/anthropic/sdk.
- `main_gui` imports the widget lazily inside `MainWindow._build_chat_dock()` (runtime, not
  module top) wrapped in try/except → if PySide-only import somehow fails, a placeholder dock.
- `build_runner()` failure (no extra / no backend) is caught in `_ensure_runner()` → the dock
  shows the reason + an "install `scat[agent]` or log in to Claude" hint and disables input.
- **Guard tests:** (a) extend `test_core_imports_without_agent_extras` to also
  `import scat.main_gui` under blocked pydantic/anthropic/claude_agent_sdk → must succeed;
  (b) relax `test_gui_has_no_agent_imports` to forbid only **top-level** agent imports
  (indented lazy imports allowed) — the runtime test is the real guard.

## 3. `scat/agent/chat_widget.py`
- `ChatWorker(QThread)`: fields `runner`, `text`. Signals `event = Signal(object)`,
  `finished_turn = Signal()`. `run()`: `for ev in self.runner.turn(self.text): self.event.emit(ev)`
  in a try/except (emit a synthetic error TextDelta+TurnDone on exception), then
  `finished_turn.emit()`.
- `ChatDockWidget(QWidget)`:
  - UI: a status `QLabel` (backend/desc or error), a read-only `QTextBrowser` conversation
    view, a one-line input (`QLineEdit`, Enter=send), **Send** / **Stop** / **New** buttons.
  - `self.runner=None; self.desc=None; self.worker=None`. Config: `agent.backend` (default
    "auto"), `agent.model` (default "claude-opus-4-8"), `agent.max_loops` (default 40).
  - `_ensure_runner() -> bool`: if `self.runner` set, True. Else lazily
    `from scat.agent.backend import build_runner` + `from scat.agent.provenance import
    start_session, set_driver`; `start_session("gui-chat"); set_driver("gui-chat")`;
    `self.runner, self.desc = build_runner(...)`; set status; return True. On
    Exception → status shows the message + "install `scat[agent]` / log in to Claude", disable
    input, return False.
  - `_send()`: text = input; if empty or a turn is running, ignore; if not `_ensure_runner()`,
    return. Append the user line; clear input; disable Send/input, enable Stop; start
    `ChatWorker(runner, text)`, connect `event`→`_on_event`, `finished_turn`→`_on_turn_finished`.
  - `_on_event(ev)`: dispatch by `type(ev).__name__` —
    `TextDelta`→append `ev.text` to the live assistant paragraph (streaming);
    `ToolUse`→a dim "🔧 {name}({compact input})" line; `ToolResult`→"✓/✗ {name}" (dim, red on
    `is_error`); `ToolUseStart`→ignore (ToolUse carries the input); `TurnDone`→status back to
    idle + a subtle "· {stop_reason}" if not end_turn. Keep the view scrolled to bottom.
  - `_on_turn_finished()`: re-enable Send/input, disable Stop.
  - `_stop()`: `self.runner and self.runner.cancel()`.
  - `_new()`: `self.runner and self.runner.reset()`; clear the view.
  - `shutdown()`: cancel a running worker + `wait()` (bounded), then `close = getattr(runner,
    'close',None); close and close()`. Idempotent.
- Rendering helper: escape HTML, compact tool input dicts to a short repr (reuse a small
  truncation; do not import the runner's private compaction).

## 4. `scat/main_gui.py` wiring (minimal, guard-safe)
- `MainWindow._setup_ui`: after the tabs, call `self._build_chat_dock()`.
- `_build_chat_dock()`: `from .agent.chat_widget import ChatDockWidget` (lazy, try/except);
  `self.chat_dock = QDockWidget("Assistant", self)`; `self.chat_widget = ChatDockWidget()`;
  `self.chat_dock.setWidget(self.chat_widget)`;
  `self.addDockWidget(Qt.RightDockWidgetArea, self.chat_dock)`; remember initial visibility from
  `config.get("window.chat_visible", True)`. On import failure, build a placeholder QLabel dock.
- A **View toggle**: `self.chat_dock.toggleViewAction()` added to a small "View" menu (or a
  toolbar button) so the user can hide/show it; persist visibility in `closeEvent`.
- `MainWindow.closeEvent`: `getattr(self,'chat_widget',None) and self.chat_widget.shutdown()`
  BEFORE `self._save_window_state(); event.accept()` (join the subscription loop thread cleanly).
- `QDockWidget` is already importable? add to the QtWidgets import list if missing.

## 5. Config
Reuse the existing `agent` config section (`agent.backend/model/max_loops`). Add
`window.chat_visible` (bool) for dock visibility persistence.

## 6. Verification
1. `test_chat_widget.py` (offscreen, no network): build `ChatDockWidget`; inject a runner built
   from the **FakeProvider** pattern (reuse `tests/test_agent_runner.py`) via a small
   `_set_runner_for_test`; drive `_send()` synchronously (run the worker inline, or call
   `_on_event` over `list(runner.turn(text))`) → assert the conversation view contains the
   streamed text + a tool line + no crash. Assert `shutdown()` is safe with no runner.
2. Graceful-degradation test: monkeypatch `build_runner` to raise → `_ensure_runner()` returns
   False, status shows the hint, input disabled — no crash.
3. Packaging guards (§2) green; full suite green; headless `MainWindow` constructs/shows/closes
   clean WITH the dock (agent extra IS installed in this env, so the real widget builds).
4. If subscription login is present, one real end-to-end smoke via the GUI is a manual step
   (not in CI); the CLI `chat` already proves the runner path.

## 7. Risks
- **R1 thread teardown:** the subscription runner owns a **daemon** asyncio loop thread, so a
  missing `close()` leaks client/session/subprocess state rather than hanging the process
  [Codex-corrected]. Still call `close()`. `shutdown()` in `closeEvent` does cancel → bounded
  `wait()` → `close()`, never blocking indefinitely (so app-close can't hang even on a stuck turn).
- **R2 double turn:** starting a turn while one runs corrupts runner state. → disable Send/input
  until `finished_turn`.
- **R3 packaging:** an accidental top-level agent import in chat_widget/main_gui breaks core.
  → render-by-classname + lazy `build_runner`; guard tests (§2).
- **R4 cancel races:** `cancel()` sets a flag mid-turn; the worker may emit a few more events
  before stopping (cooperative, documented). Acceptable.
- **R5 event object identity across threads:** RunEvents are plain dataclasses passed via a
  queued signal (`Signal(object)`) — copied by reference, read-only in the slot. Fine.

## 8. Sequencing
chat_widget.py (self-contained) → main_gui wiring → guard-test updates → tests → verify.
Do NOT merge to `main` (user reviews).
