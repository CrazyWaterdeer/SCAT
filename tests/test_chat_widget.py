"""Chat dock — offline (no network / no LLM) tests via the FakeProvider pattern.

Drives the real ChatDockWidget + ChatWorker with an AgentRunner backed by a scripted provider,
so the rendering, the QThread event bridge, and the graceful-degradation path are all exercised
without the [agent] extra's runtime backends.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from scat.agent.runner import AgentRunner
from scat.agent.providers.base import TextDelta, ToolUse, Stop


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


class _ScriptedProvider:
    """Round 1: call scan_folder. Round 2: reply with text."""
    name = "fake"; model = "fake"

    def __init__(self):
        self._round = 0

    def stream(self, messages, tools, system):
        self._round += 1
        if self._round == 1:
            yield ToolUse(id="tu1", name="scan_folder",
                          input={"path": messages[0]["content"][0]["text"]})
            yield Stop(reason="tool_use", usage={"input_tokens": 10, "output_tokens": 5})
        else:
            yield TextDelta(text="Found 6 images across two groups.")
            yield Stop(reason="end_turn", usage={"input_tokens": 8, "output_tokens": 4})


def _runner():
    import scat.tools  # noqa: F401 — register tools so scan_folder resolves
    return AgentRunner(_ScriptedProvider(), "sys", max_loops=5)


def test_renders_user_tool_and_assistant(app, synth_dir):
    from scat.agent.chat_widget import ChatDockWidget
    w = ChatDockWidget()
    w._set_runner_for_test(_runner())
    # Drive the rendering directly with the real event stream (deterministic, no thread).
    w._append_html("<p><b>You:</b> analyze it</p>")
    for ev in w.runner.turn(str(synth_dir)):
        w._on_event(ev)
    text = w.view.toPlainText()
    assert "You:" in text and "scan_folder" in text
    assert "Found 6 images" in text          # streamed assistant text rendered
    assert "✓ scan_folder" in text           # successful tool result marker


def test_worker_bridges_events_to_signals(app, synth_dir):
    from scat.agent.chat_widget import ChatWorker
    collected, done = [], []
    worker = ChatWorker(_runner(), str(synth_dir))
    worker.event.connect(collected.append)
    worker.finished_turn.connect(lambda: done.append(True))
    worker.start()
    for _ in range(600):                      # spin the loop so queued signals deliver
        if done:
            break
        app.processEvents()
        worker.wait(20)
    assert done, "worker never signalled finished"
    kinds = [type(e).__name__ for e in collected]
    assert "ToolUse" in kinds and "ToolResult" in kinds and kinds[-1] == "TurnDone"
    worker.wait(2000)


def test_graceful_degradation_when_backend_missing(app, monkeypatch):
    from scat.agent.chat_widget import ChatDockWidget
    import scat.agent.backend as backend
    monkeypatch.setattr(backend, "build_runner",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("No backend available")))
    w = ChatDockWidget()
    assert w._ensure_runner() is False
    assert not w.input.isEnabled() and not w.send_btn.isEnabled()
    assert "unavailable" in w.status.text().lower()


def test_backend_error_shows_in_conversation(app, monkeypatch):
    """A backend/connection failure is rendered inline in the transcript (like Claude), with the
    user's message and the underlying reason — not just the small gray status line."""
    from scat.agent.chat_widget import ChatDockWidget
    import scat.agent.backend as backend
    monkeypatch.setattr(backend, "build_runner", lambda **k: (_ for _ in ()).throw(
        RuntimeError("subscription backend requested but unavailable: not logged in")))
    w = ChatDockWidget()
    w.input.setPlainText("analyze it")
    w._send()
    text = w.view.toPlainText()
    assert "analyze it" in text                 # the user's message is shown
    assert "Assistant unavailable" in text      # the error is in the conversation, not just status
    assert "not logged in" in text              # with the underlying reason


def test_shutdown_is_safe_without_runner(app):
    from scat.agent.chat_widget import ChatDockWidget
    ChatDockWidget().shutdown()  # must not raise


def test_model_and_provider_pickers(app):
    from scat.agent.chat_widget import ChatDockWidget, _PROVIDERS
    from scat.agent.backend import LATEST_MODELS
    from scat.config import config
    w = ChatDockWidget()
    # Model picker lists the single-source latest models and selects the configured one
    assert [w.model_combo.itemData(i) for i in range(w.model_combo.count())] == [m for _, m in LATEST_MODELS]
    assert w.model_combo.currentData() == config.get("agent.model", "claude-opus-4-8")
    assert [w.provider_combo.itemData(i) for i in range(w.provider_combo.count())] == [v for _, v in _PROVIDERS]
    assert w.provider_combo.currentData() == config.get("agent.backend", "auto")


def test_provider_picker_marks_subscription_not_connected(app, monkeypatch):
    monkeypatch.setattr("scat.agent.claude_subscription.subscription_available",
                        lambda: (False, "not logged in"))
    from scat.agent.chat_widget import ChatDockWidget
    w = ChatDockWidget()
    assert w.provider_combo.itemText(1) == "Subscription — not connected"
    assert w.provider_combo.itemData(1) == "subscription"   # backend value (UserRole) untouched


def test_provider_picker_marks_subscription_connected(app, monkeypatch):
    monkeypatch.setattr("scat.agent.claude_subscription.subscription_available",
                        lambda: (True, None))
    from scat.agent.chat_widget import ChatDockWidget
    w = ChatDockWidget()
    assert w.provider_combo.itemText(1) == "Subscription"
    assert w.provider_combo.itemData(1) == "subscription"


def test_model_change_persists_and_invalidates_runner(app, monkeypatch):
    from scat.agent.chat_widget import ChatDockWidget
    from scat.config import config
    saved = {}
    monkeypatch.setattr(config, "set", lambda k, v, **kw: saved.__setitem__(k, v))
    w = ChatDockWidget()
    w._set_runner_for_test(_runner())
    assert w.runner is not None
    w.model_combo.setCurrentIndex(2)  # -> triggers _on_model_changed
    assert saved.get("agent.model") == w.model_combo.itemData(2)
    assert w.runner is None  # invalidated so the next send rebuilds with the new model


def test_last_model_and_provider_persist_across_restart(app, monkeypatch):
    """Picking a model/provider persists (config auto-save); a fresh dock — as after a restart —
    restores both. Locks in the 'remember my last model + provider' behavior."""
    from scat.agent.chat_widget import ChatDockWidget
    from scat.config import config
    store = {}
    monkeypatch.setattr(config, "set", lambda k, v, **kw: store.__setitem__(k, v))
    monkeypatch.setattr(config, "get", lambda k, d=None: store.get(k, d))
    w = ChatDockWidget()
    w.model_combo.setCurrentIndex(2)      # -> _on_model_changed   -> config.set(agent.model, …)
    w.provider_combo.setCurrentIndex(2)   # -> _on_provider_changed -> config.set(agent.backend, …)
    picked_model, picked_provider = w.model_combo.itemData(2), w.provider_combo.itemData(2)
    assert store.get("agent.model") == picked_model
    assert store.get("agent.backend") == picked_provider
    # a brand-new dock (as after quit + relaunch) reads the persisted values back
    w2 = ChatDockWidget()
    assert w2.model_combo.currentData() == picked_model
    assert w2.provider_combo.currentData() == picked_provider


def test_slash_clear_command(app):
    from scat.agent.chat_widget import ChatDockWidget
    w = ChatDockWidget()
    w._set_runner_for_test(_runner())
    w._append_html("<p>some prior text</p>")
    assert "some prior text" in w.view.toPlainText()
    w.input.setPlainText("/clear")
    w._send()
    assert "some prior text" not in w.view.toPlainText()
    assert "cleared" in w.view.toPlainText()


def test_slash_unknown_command(app):
    from scat.agent.chat_widget import ChatDockWidget
    w = ChatDockWidget()
    w.input.setPlainText("/bogus")
    w._send()  # no runner needed — commands short-circuit before _ensure_runner
    assert "Unknown command" in w.view.toPlainText()


class _AnalyzeProvider:
    """Round 1: call analyze_folder on the synth folder. Round 2: finish."""
    name = "fake"; model = "fake"

    def __init__(self, folder):
        self._folder = folder
        self._round = 0

    def stream(self, messages, tools, system):
        self._round += 1
        if self._round == 1:
            yield ToolUse(id="tu1", name="analyze_folder",
                          input={"path": self._folder, "annotate": False})
            yield Stop(reason="tool_use", usage={})
        else:
            yield TextDelta(text="done.")
            yield Stop(reason="end_turn", usage={})


def test_chatworker_emits_per_image_progress(app, synth_dir):
    """The ChatWorker.progress signal fires per image while the analyze_folder tool runs under
    the ambient run_progress sink — the fix for the 'frozen dock' defect (T1.1)."""
    import scat.tools  # noqa: F401 — register analyze_folder
    from scat.agent.runner import AgentRunner
    from scat.agent.chat_widget import ChatWorker
    runner = AgentRunner(_AnalyzeProvider(str(synth_dir)), "sys", max_loops=5)
    prog, done = [], []
    w = ChatWorker(runner, "analyze it")
    w.progress.connect(lambda c, t, note: prog.append((c, t)))
    w.finished_turn.connect(lambda: done.append(True))
    w.start()
    for _ in range(2000):
        if done:
            break
        app.processEvents()
        w.wait(20)
    assert done, "turn never finished"
    assert prog, "no per-image progress was emitted during analyze_folder"
    assert prog[-1][1] == 6 and max(c for c, _ in prog) == 6   # 6 synth images, ends at total
    w.wait(2000)


def test_stop_requests_cancel(app, monkeypatch):
    from scat.agent.chat_widget import ChatDockWidget
    import scat.progress as prog
    called = []
    monkeypatch.setattr(prog, "request_cancel", lambda: called.append(True))
    w = ChatDockWidget()
    w._set_runner_for_test(_runner())
    w._stop()
    assert called, "_stop must request cancellation of the in-progress batch"
