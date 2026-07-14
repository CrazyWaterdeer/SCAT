"""PySide6 chat dock — embeds SCAT's conversational agent in the GUI.

Design constraints (see docs/superpowers/plans/2026-07-14-scat-gui-chat-dock.md):
- Top level imports **nothing that requires the ``[agent]`` extra** — only PySide6, the core
  ``Theme``, the plain ``LATEST_MODELS`` list, and ``config`` (none pull pydantic/anthropic).
  The agent runner (``build_runner``) is imported **lazily on first send**, so the dock
  constructs fine without the extra — chatting is what needs it, and its absence is surfaced as
  a friendly message rather than an import crash. Events are rendered by class NAME + duck-typed
  attributes so we never import the RunEvent classes here either.
- ``runner.turn(text)`` is a synchronous generator (for the subscription backend it internally
  owns an asyncio loop on its own daemon thread and bridges via a queue). We drive it on a
  ``ChatWorker(QThread)`` and marshal each event to the GUI thread through a queued signal.
"""
from __future__ import annotations

import html

from PySide6.QtCore import Qt, QThread, Signal, QPoint, QSize
from PySide6.QtGui import QTextCursor, QIcon, QPixmap, QPainter, QPolygon, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser, QTextEdit, QPushButton, QComboBox,
)

from scat.ui_common import Theme                 # core GUI theme (no agent deps)
from scat.config import config                   # core config (no agent deps)
from scat.agent.backend import LATEST_MODELS     # plain model list (backend top-level = os + prompts)

_PROVIDERS = [("Auto", "auto"), ("Subscription", "subscription"), ("API", "api")]


def _shape_icon(kind: str, size: int = 18, color: str = "#FFFFFF") -> QIcon:
    """Draw the send/stop glyph with QPainter so it renders identically in any font
    environment (Unicode arrows/shapes fall back to .notdef in minimal fonts)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    s = size
    if kind == "send":                       # upward arrow: triangle head + stem
        p.drawPolygon(QPolygon([
            QPoint(int(s * 0.50), int(s * 0.18)),
            QPoint(int(s * 0.20), int(s * 0.52)),
            QPoint(int(s * 0.80), int(s * 0.52)),
        ]))
        p.drawRect(int(s * 0.42), int(s * 0.46), int(s * 0.16), int(s * 0.34))
    else:                                     # stop: rounded square
        p.drawRoundedRect(int(s * 0.28), int(s * 0.28), int(s * 0.44), int(s * 0.44), 2, 2)
    p.end()
    return QIcon(pm)


def _compact_input(value, limit: int = 200) -> str:
    """Short, safe repr of a tool-call input dict for the transcript."""
    try:
        import json
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > limit:
        text = text[:limit] + f"… (+{len(text) - limit})"
    return text


class _Composer(QTextEdit):
    """A Claude-app-style multi-line input: Enter sends, Shift+Enter inserts a newline.
    Hosts a floating send/stop button pinned to its bottom-right corner."""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.corner_button = None   # set by ChatDockWidget; repositioned on resize

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_button()

    def _reposition_button(self):
        b = self.corner_button
        if b is not None:
            margin = 10
            b.move(self.width() - b.width() - margin, self.height() - b.height() - margin)


class ChatWorker(QThread):
    """Runs one agent turn off the GUI thread, emitting each RunEvent as it arrives."""

    event = Signal(object)
    finished_turn = Signal()

    def __init__(self, runner, text: str):
        super().__init__()
        self._runner = runner
        self._text = text

    def run(self):
        try:
            for ev in self._runner.turn(self._text):
                self.event.emit(ev)
        except Exception as exc:  # never let the worker die silently
            self.event.emit(_SyntheticError(f"{type(exc).__name__}: {exc}"))
        finally:
            self.finished_turn.emit()


class _SyntheticError:
    """Rendered like a TextDelta; used when the worker itself raises."""
    def __init__(self, text):
        self.text = "\n[chat error] " + text


class ChatDockWidget(QWidget):
    """The dock contents: status line, transcript, a composer with an in-corner send/stop
    button (Send ↑ toggles to Stop ■ while a turn runs), and model + provider pickers below."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner = None
        self.desc = None
        self.worker = None
        self._assistant_open = False   # has the current turn printed an "Assistant:" header?
        self._build_ui()

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(440, 620)   # a roomy default so the dock opens spacious

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.status = QLabel("Assistant ready. Type a request (e.g. “analyze this folder”).")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.status)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        layout.addWidget(self.view, 1)

        # Composer (multi-line) — Claude-app style: a spacious rounded input with the send/stop
        # button pinned INSIDE its bottom-right corner. Enter sends, Shift+Enter = newline.
        self.input = _Composer()
        self.input.setPlaceholderText("Ask the assistant…    Type / for commands")
        self.input.setMinimumHeight(100)
        self.input.setMaximumHeight(200)
        self.input.setAcceptRichText(False)
        self.input.submitted.connect(self._send)
        self.input.setStyleSheet(f"""
            QTextEdit {{
                background: {Theme.BG_MEDIUM};
                border: 1px solid {Theme.BORDER};
                border-radius: 14px;
                padding: 12px 14px 46px 14px;   /* extra bottom room for the corner button */
                color: {Theme.TEXT_PRIMARY};
                font-size: 13px;
            }}
            QTextEdit:focus {{ border: 1px solid {Theme.BORDER_FOCUS}; }}
        """)
        # Floating send/stop button inside the composer's bottom-right corner
        self.send_btn = QPushButton(self.input)
        self.send_btn.setFixedSize(34, 34)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send_or_stop)
        self.input.corner_button = self.send_btn
        self._set_send_mode(True)
        self.input._reposition_button()
        layout.addWidget(self.input)

        # Below the composer: model + provider pickers (select both at once), Claude-app style.
        picker_row = QHBoxLayout()
        picker_row.setSpacing(6)

        self.model_combo = QComboBox()
        for _name, _mid in LATEST_MODELS:
            self.model_combo.addItem(_name, _mid)
        _cur_model = config.get("agent.model", "claude-opus-4-8")
        self.model_combo.setCurrentIndex(
            next((i for i, (_n, m) in enumerate(LATEST_MODELS) if m == _cur_model), 0))
        self.model_combo.setToolTip("Model — always the latest Claude versions")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)  # after setCurrentIndex
        picker_row.addWidget(self.model_combo)

        self.provider_combo = QComboBox()
        for _name, _val in _PROVIDERS:
            self.provider_combo.addItem(_name, _val)
        _cur_backend = config.get("agent.backend", "auto")
        self.provider_combo.setCurrentIndex(
            next((i for i, (_n, v) in enumerate(_PROVIDERS) if v == _cur_backend), 0))
        self.provider_combo.setToolTip(
            "Provider — Auto uses your Claude subscription if logged in (no API charges), else the billed API")
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        picker_row.addWidget(self.provider_combo)

        picker_row.addStretch(1)
        layout.addLayout(picker_row)

    def _set_send_mode(self, sending: bool):
        """Toggle the corner button between Send (▲ arrow, accent) and Stop (■) — same button.
        Uses painter-drawn icons so the glyph is never a missing-font box."""
        b = self.send_btn
        b.setIconSize(QSize(18, 18))
        base = "border:none; border-radius:17px;"
        if sending:
            b.setIcon(_shape_icon("send"))
            b.setToolTip("Send  (Enter)")
            b.setStyleSheet(
                f"QPushButton{{{base} background:{Theme.PRIMARY};}}"
                f"QPushButton:hover{{background:{Theme.PRIMARY_LIGHT};}}"
                f"QPushButton:disabled{{background:{Theme.BG_LIGHTER};}}")
        else:
            b.setIcon(_shape_icon("stop"))
            b.setToolTip("Stop")
            b.setStyleSheet(
                f"QPushButton{{{base} background:{Theme.BG_LIGHTER};}}"
                f"QPushButton:hover{{background:{Theme.SECONDARY};}}")

    def _set_running(self, running: bool):
        """Enter/leave the 'turn in progress' UI state. The composer goes read-only (not
        disabled) so the in-corner Stop button stays clickable; the pickers are frozen."""
        self.input.setReadOnly(running)
        self._set_send_mode(not running)
        self.model_combo.setEnabled(not running)
        self.provider_combo.setEnabled(not running)

    def _append_html(self, fragment: str):
        self.view.moveCursor(QTextCursor.MoveOperation.End)
        self.view.insertHtml(fragment)
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---- runner lifecycle -------------------------------------------------
    def _ensure_runner(self) -> bool:
        """Lazily build the agent runner (needs the [agent] extra + a backend). Returns
        True when a runner is available; otherwise surfaces why and disables input."""
        if self.runner is not None:
            return True
        try:
            from scat.agent.backend import build_runner
            from scat.agent.provenance import start_session, set_driver
            start_session(driver="gui-chat")
            set_driver("gui-chat")
            self.runner, self.desc = build_runner(
                backend=config.get("agent.backend", "auto"),
                model=config.get("agent.model", "claude-opus-4-8"),
                max_loops=config.get("agent.max_loops", 40),
            )
            self.status.setText(self.desc)
            return True
        except Exception as exc:
            self.status.setText(
                f"Assistant unavailable: {exc}\n"
                "Install the agent extra (pip install 'scat[agent]') and log in to Claude "
                "(`claude` CLI) or set ANTHROPIC_API_KEY."
            )
            self.input.setEnabled(False)
            self.send_btn.setEnabled(False)
            return False

    def _set_runner_for_test(self, runner, desc="test runner"):
        """Inject a runner (e.g. AgentRunner+FakeProvider) so the dock is testable offline."""
        self.runner = runner
        self.desc = desc
        self.status.setText(desc)

    # ---- interactions -----------------------------------------------------
    def _send(self):
        if self.worker is not None and self.worker.isRunning():
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        if text.startswith("/"):
            self.input.clear()
            self._handle_command(text)
            return
        if not self._ensure_runner():
            return
        self.input.clear()
        self._append_html(f"<p><b>You:</b> {html.escape(text)}</p>")
        self._assistant_open = False
        self.worker = ChatWorker(self.runner, text)
        self.worker.event.connect(self._on_event)
        self.worker.finished_turn.connect(self._on_turn_finished)
        self.worker.start()
        self._set_running(True)   # composer read-only; send button becomes Stop

    def _on_send_or_stop(self):
        """The one corner button: Stop while a turn runs, Send otherwise."""
        if self.worker is not None and self.worker.isRunning():
            self._stop()
        else:
            self._send()

    def _ensure_assistant_header(self):
        if not self._assistant_open:
            self._append_html("<p><b>Assistant:</b> ")
            self._assistant_open = True

    def _on_event(self, ev):
        kind = type(ev).__name__
        if kind in ("TextDelta", "_SyntheticError"):
            self._ensure_assistant_header()
            self._append_html(html.escape(getattr(ev, "text", "")).replace("\n", "<br>"))
        elif kind == "ToolUse":
            name = html.escape(str(getattr(ev, "name", "tool")))
            args = html.escape(_compact_input(getattr(ev, "input", {})))
            self._append_html(
                f"<div style='color:gray'>\U0001f527 {name}(<span style='color:#888'>{args}</span>)</div>")
        elif kind == "ToolResult":
            name = html.escape(str(getattr(ev, "name", "tool")))
            if getattr(ev, "is_error", False):
                self._append_html(f"<div style='color:#c0392b'>✗ {name}</div>")
            else:
                self._append_html(f"<div style='color:gray'>✓ {name}</div>")
        elif kind == "TurnDone":
            stop = str(getattr(ev, "stop_reason", ""))
            if stop and stop not in ("end_turn", "tool_use"):
                self._append_html(f"<div style='color:gray; font-size:11px'>· {html.escape(stop)}</div>")
            self._assistant_open = False
        # ToolUseStart: ignored (ToolUse carries the same id + the input)

    def _on_turn_finished(self):
        self._set_running(False)   # composer editable again; button back to Send
        self.input.setFocus()

    def _stop(self):
        if self.runner is not None:
            try:
                self.runner.cancel()
            except Exception:
                pass

    # ---- slash commands + model/provider selection ------------------------
    def _handle_command(self, text):
        """Handle a /command typed in the composer (Claude-app-style)."""
        cmd = text.split()[0].lower()
        if cmd in ("/clear", "/new", "/reset"):
            if self.worker is not None and self.worker.isRunning():
                return
            if self.runner is not None:
                try:
                    self.runner.reset()
                except Exception:
                    pass
            self.view.clear()
            self._assistant_open = False
            self._append_html("<div style='color:gray; font-size:11px'>· conversation cleared</div>")
        elif cmd == "/help":
            self._append_html(
                "<div style='color:gray; font-size:12px'>Commands: "
                "<b>/clear</b> — start a new conversation · <b>/help</b> — this list<br>"
                "Enter sends · Shift+Enter for a new line · Stop cancels a running turn</div>")
        else:
            self._append_html(
                f"<div style='color:#c0392b'>Unknown command: {html.escape(cmd)} — try /help</div>")

    def _on_model_changed(self, idx):
        mid = self.model_combo.itemData(idx)
        if mid:
            config.set("agent.model", mid)
            self._invalidate_runner(f"Model set to {self.model_combo.itemText(idx)}")

    def _on_provider_changed(self, idx):
        val = self.provider_combo.itemData(idx)
        if val:
            config.set("agent.backend", val)
            self._invalidate_runner(f"Provider set to {self.provider_combo.itemText(idx)}")

    def _invalidate_runner(self, note):
        """Drop the current runner so the next send rebuilds with the new model/provider.
        Ignored mid-turn (the pickers are disabled then, so this shouldn't fire)."""
        if self.worker is not None and self.worker.isRunning():
            return
        runner, self.runner = self.runner, None
        if runner is not None:
            close = getattr(runner, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
        self.input.setEnabled(True)   # re-enable if a prior build had disabled input
        self.send_btn.setEnabled(True)
        self.status.setText(f"{note} — starts a new conversation on your next message.")

    def shutdown(self):
        """Cancel any in-flight turn, then tear the runner down. Bounded so app-close never
        hangs: cancel() wakes the turn (API loop checks the flag; subscription interrupts and
        the async response ends → its queue sentinel unblocks the worker), we wait for that,
        then close() the runner (subscription owns a daemon asyncio loop worth closing cleanly;
        it is a daemon thread, so a leftover only leaks session state, it does not block exit).
        We never block indefinitely on the worker. Idempotent."""
        if self.worker is not None and self.worker.isRunning():
            try:
                self.runner and self.runner.cancel()
            except Exception:
                pass
            self.worker.wait(5000)  # give cancel/interrupt time to end the turn cleanly
        runner, self.runner = self.runner, None
        if runner is not None:
            close = getattr(runner, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
