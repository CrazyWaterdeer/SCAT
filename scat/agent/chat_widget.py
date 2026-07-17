"""PySide6 chat dock — embeds SCAT's conversational agent in the GUI.

The transcript is modelled on the Claude app: each turn is its own widget, not a line appended
to one text log. User messages are right-aligned bubbles; the assistant answers full-width under
a spark avatar with real markdown; tool calls render as subtle pills. Every text surface sets an
explicit light colour, so nothing ever renders black-on-black.

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
- The transcript keeps a **synchronous plain-text mirror** (each block reports ``plain_text()``),
  so ``view.toPlainText()`` is correct the instant an event is handled — independent of when the
  widgets actually repaint. Rendering can therefore be deferred/coalesced without breaking tests
  or the "copy the conversation" affordance.
"""
from __future__ import annotations

import html
import math

from PySide6.QtCore import Qt, QThread, Signal, QPoint, QSize, QTimer
from PySide6.QtGui import (
    QIcon, QPixmap, QPainter, QPolygon, QColor, QPalette, QTextDocument,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser, QTextEdit, QPushButton, QComboBox,
    QFrame, QScrollArea, QSizePolicy,
)

from scat.ui_common import Theme, icon           # core GUI theme + bundled Material Symbol loader
from scat.config import config                   # core config (no agent deps)
from scat.agent.backend import LATEST_MODELS     # plain model list (backend top-level = os + prompts)

_PROVIDERS = [("Auto", "auto"), ("Subscription", "subscription"), ("API", "api")]

_EXAMPLE_PROMPTS = [
    "Analyze this folder and compare the groups",
    "Summarize the ROD vs Normal differences",
    "Build an HTML report of the results",
]


# --------------------------------------------------------------------------- painter glyphs
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


def _spark_pixmap(size: int = 26, fg: str = "#FFFFFF", bg: str = Theme.PRIMARY) -> QPixmap:
    """The assistant avatar: a coral disc with a four-point spark. Painter-drawn so it never
    depends on an emoji/symbol font being present."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(bg))
    p.drawEllipse(0, 0, size, size)
    c = size / 2.0
    outer, inner = size * 0.34, size * 0.13
    pts = []
    for k in range(8):
        ang = -math.pi / 2 + k * math.pi / 4
        rr = outer if k % 2 == 0 else inner
        pts.append(QPoint(int(c + rr * math.cos(ang)), int(c + rr * math.sin(ang))))
    p.setBrush(QColor(fg))
    p.drawPolygon(QPolygon(pts))
    p.end()
    return pm


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


def _html_to_plain(fragment: str) -> str:
    """Plain-text of an HTML fragment, for the synchronous transcript mirror."""
    d = QTextDocument()
    d.setHtml(fragment)
    return d.toPlainText()


# --------------------------------------------------------------------------- composer
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


# --------------------------------------------------------------------------- transcript blocks
class _GrowingBrowser(QTextBrowser):
    """One assistant text run, rendered as inline markdown that flows like plain text: no frame,
    no scrollbars, height tracked to the document so many of them stack in a scrolling column.
    Explicit light text colour (QSS + palette) so it can never render black-on-black."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantText")
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(0)               # never force the transcript wider than its viewport
        self.setLineWrapMode(QTextBrowser.WidgetWidth)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        self.document().setDocumentMargin(0)
        self.document().setDefaultStyleSheet(
            f"a{{color:{Theme.PRIMARY_LIGHT};text-decoration:none;}}"
            f"code{{color:{Theme.TEXT_PRIMARY};}}"
            f"pre{{color:{Theme.TEXT_PRIMARY};}}")
        self.setStyleSheet(
            "QTextBrowser#assistantText{background:transparent;border:none;"
            f"color:{Theme.TEXT_PRIMARY};font-size:13px;}}")
        pal = self.palette()
        pal.setColor(QPalette.Text, QColor(Theme.TEXT_PRIMARY))
        pal.setColor(QPalette.Base, QColor(0, 0, 0, 0))
        self.setPalette(pal)
        self._raw = ""
        self._last_w = -1

    def set_markdown(self, md: str):
        self._raw = md
        self.setMarkdown(md)
        self._sync_height()

    def _sync_height(self):
        doc = self.document()
        w = self.viewport().width()
        if w <= 0:
            w = self.width()
        doc.setTextWidth(w)
        self.setFixedHeight(int(math.ceil(doc.size().height())) + 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.viewport().width()
        if w != self._last_w:                 # width-only guard: setFixedHeight must not recurse
            self._last_w = w
            self._sync_height()

    def wheelEvent(self, event):
        event.ignore()                        # let the transcript scroll, not this inline view


class _TypingDots(QWidget):
    """Three painter-drawn dots that cycle while the assistant is thinking (before its first
    token). Font-free, so it never shows a missing-glyph box."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 16)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self._phase = (self._phase + 1) % 3
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        cy = self.height() // 2
        for i in range(3):
            p.setBrush(QColor(Theme.TEXT_PRIMARY if i == self._phase else Theme.TEXT_MUTED))
            cx = 5 + i * 10
            p.drawEllipse(cx - 3, cy - 3, 6, 6)
        p.end()


class _UserBubble(QFrame):
    """A right-aligned message bubble for what the user said."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("userBubble")
        self._text = text
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(0)
        lbl = QLabel()
        lbl.setObjectName("userText")
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setText(html.escape(text).replace("\n", "<br>"))
        v.addWidget(lbl)

    def plain_text(self) -> str:
        return self._text


class _ToolChip(QFrame):
    """A subtle pill for a tool call or its result: a bundled (tofu-proof) icon + the tool name,
    tinted by state. The full tool input lives in the tooltip so the pill can't force overflow."""

    def __init__(self, text: str, color: str, glyph: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("toolChip")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        h = QHBoxLayout(self)
        h.setContentsMargins(9, 4, 11, 4)
        h.setSpacing(6)
        pm = icon(glyph, color, 13).pixmap(13, 13)
        if not pm.isNull():
            ic = QLabel()
            ic.setPixmap(pm)
            ic.setFixedSize(13, 13)
            h.addWidget(ic, 0, Qt.AlignVCenter)
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{color}; background:transparent; font-size:12px;")
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        h.addWidget(lbl)
        if tooltip:
            self.setToolTip(tooltip)


class _SystemLine(QFrame):
    """A centred, muted status line (stop reasons, /help, command notes) — renders whatever HTML
    fragment it is given, with its own inline colours honoured."""

    def __init__(self, fragment: str, parent=None):
        super().__init__(parent)
        self.setObjectName("systemLine")
        self._plain = _html_to_plain(fragment)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        lbl = QLabel()
        lbl.setObjectName("systemText")
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setText(fragment)
        h.addWidget(lbl)

    def plain_text(self) -> str:
        return self._plain


class _AssistantBlock(QFrame):
    """One assistant turn: a spark avatar beside a column that interleaves markdown text runs and
    tool-call pills in arrival order, with a typing indicator until the first content lands."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantBlock")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        avatar = QLabel()
        avatar.setFixedSize(26, 26)
        avatar.setPixmap(_spark_pixmap(26))
        row.addWidget(avatar, 0, Qt.AlignTop)

        colw = QWidget()
        colw.setObjectName("assistantCol")
        self.col = QVBoxLayout(colw)
        self.col.setContentsMargins(0, 2, 0, 0)
        self.col.setSpacing(8)
        row.addWidget(colw, 1)

        self._typing = _TypingDots()
        self.col.addWidget(self._typing, 0, Qt.AlignLeft)

        self._cur_browser = None      # current text run (None after a tool pill breaks the run)
        self._cur_part = None         # its plain-text mirror dict
        self._parts = []              # ordered [{kind:'text',raw} | {kind:'plain',text}]

    def stop_typing(self):
        if self._typing is not None:
            self._typing.setParent(None)   # unparent now (removeWidget alone leaves it painted)
            self._typing.deleteLater()
            self._typing = None

    def append_markdown(self, delta: str):
        self.stop_typing()
        if self._cur_browser is None:
            self._cur_browser = _GrowingBrowser()
            self.col.addWidget(self._cur_browser)
            self._cur_part = {"kind": "text", "raw": ""}
            self._parts.append(self._cur_part)
        self._cur_part["raw"] += delta
        self._cur_browser.set_markdown(self._cur_part["raw"])

    def add_tool_use(self, name: str, args: str):
        self.stop_typing()
        self._cur_browser = None
        self._cur_part = None
        # Collapsed, Claude-style: the pill shows a gear + the tool name (always fits the column);
        # the full input lives in the tooltip so it can't force horizontal overflow.
        chip = _ToolChip(name, Theme.TEXT_SECONDARY, "settings", tooltip=f"{name}({args})")
        self.col.addWidget(chip, 0, Qt.AlignLeft)
        self._parts.append({"kind": "plain", "text": f"\U0001f527 {name}({args})"})

    def add_tool_result(self, name: str, ok: bool):
        self.stop_typing()
        self._cur_browser = None
        self._cur_part = None
        color = Theme.NORMAL if ok else "#C0392B"
        chip = _ToolChip(name, color, "check_circle" if ok else "close")
        self.col.addWidget(chip, 0, Qt.AlignLeft)
        self._parts.append({"kind": "plain", "text": f"{'✓' if ok else '✗'} {name}"})

    def plain_text(self) -> str:
        return "\n".join(p["raw"] if p["kind"] == "text" else p["text"] for p in self._parts)


class _Welcome(QFrame):
    """Empty-state hero: avatar, one-line pitch, and clickable example prompts."""

    def __init__(self, on_example=None, parent=None):
        super().__init__(parent)
        self.setObjectName("welcome")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 28, 8, 8)
        v.setSpacing(10)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        avatar = QLabel()
        avatar.setFixedSize(40, 40)
        avatar.setPixmap(_spark_pixmap(40))
        v.addWidget(avatar, 0, Qt.AlignHCenter)

        title = QLabel("SCAT Assistant")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignCenter)
        v.addWidget(title)

        sub = QLabel("Ask me to analyze a folder, compare experimental groups, or build a report.")
        sub.setObjectName("welcomeSub")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        v.addWidget(sub)

        v.addSpacing(4)
        for ex in _EXAMPLE_PROMPTS:
            chip = QPushButton(ex)
            chip.setObjectName("exampleChip")
            chip.setCursor(Qt.PointingHandCursor)
            if on_example is not None:
                chip.clicked.connect(lambda _=False, t=ex: on_example(t))
            v.addWidget(chip, 0, Qt.AlignHCenter)

    def plain_text(self) -> str:
        return ""


_TRANSCRIPT_QSS = f"""
    QScrollArea {{ background: transparent; border: none; }}
    #transcriptBody {{ background: transparent; }}
    #userBubble {{
        background: {Theme.BG_HOVER};
        border: 1px solid {Theme.BORDER};
        border-radius: 16px;
    }}
    #userText {{ color: {Theme.TEXT_PRIMARY}; font-size: 13px; background: transparent; }}
    #assistantCol {{ background: transparent; }}
    #toolChip {{
        background: {Theme.BG_SURFACE};
        border: 1px solid {Theme.BORDER};
        border-radius: 10px;
    }}
    #systemText {{ color: {Theme.TEXT_MUTED}; font-size: 11px; background: transparent; }}
    #welcomeTitle {{ color: {Theme.TEXT_PRIMARY}; font-size: 15px; font-weight: 600; background: transparent; }}
    #welcomeSub {{ color: {Theme.TEXT_SECONDARY}; font-size: 12px; background: transparent; }}
    #exampleChip {{
        color: {Theme.TEXT_SECONDARY};
        background: {Theme.BG_SURFACE};
        border: 1px solid {Theme.BORDER};
        border-radius: 14px;
        padding: 7px 14px;
        font-size: 12px;
        text-align: center;
    }}
    #exampleChip:hover {{ color: {Theme.TEXT_PRIMARY}; background: {Theme.BG_HOVER}; border-color: {Theme.SECONDARY}; }}
"""


class _Transcript(QScrollArea):
    """The conversation surface: a scrolling column of message blocks. Keeps a synchronous
    ``plain_text()`` mirror (via each block) so ``toPlainText()`` never depends on repaint timing.
    Exposes the small surface the dock and the test-suite drive: ``add_user``, ``ensure_assistant``,
    ``append_assistant``, ``add_tool_use``, ``add_tool_result``, ``append_html``, ``clear``."""

    def __init__(self, on_example=None, parent=None):
        super().__init__(parent)
        self._on_example = on_example
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)

        self._body = QWidget()
        self._body.setObjectName("transcriptBody")
        self._lay = QVBoxLayout(self._body)
        self._lay.setContentsMargins(16, 16, 16, 16)
        self._lay.setSpacing(16)
        self._lay.addStretch(1)               # trailing stretch: blocks stack top-down
        self.setWidget(self._body)

        self._blocks = []                     # anything with plain_text() (not the welcome)
        self._userbubbles = []                # width-capped on resize
        self._current = None                  # open assistant block, or None
        self._welcome = None
        self._stick = True                    # auto-scroll unless the user scrolled up

        self.setStyleSheet(_TRANSCRIPT_QSS)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self._show_welcome()

    # ---- layout helpers ----
    def _insert(self, w):
        self._lay.insertWidget(self._lay.count() - 1, w)   # before the trailing stretch

    def _left_row(self, w):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w, 1)
        return row

    def _right_row(self, w):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addStretch(1)
        h.addWidget(w)
        return row

    def _show_welcome(self):
        if self._welcome is None:
            self._welcome = _Welcome(self._on_example)
            self._insert(self._welcome)

    def _clear_welcome(self):
        if self._welcome is not None:
            self._welcome.setParent(None)   # unparent now (removeWidget alone leaves it painted)
            self._welcome.deleteLater()
            self._welcome = None
            self._body.update()

    # ---- content API ----
    def add_user(self, text: str):
        self.end_assistant()
        self._clear_welcome()
        bubble = _UserBubble(text)
        self._insert(self._right_row(bubble))
        self._blocks.append(bubble)
        self._userbubbles.append(bubble)
        self._apply_bubble_width(bubble)
        self._defer_scroll()

    def ensure_assistant(self) -> "_AssistantBlock":
        if self._current is None:
            self._clear_welcome()
            self._current = _AssistantBlock()
            self._insert(self._left_row(self._current))
            self._blocks.append(self._current)
            self._defer_scroll()
        return self._current

    def end_assistant(self):
        if self._current is not None:
            self._current.stop_typing()
            self._current = None

    def append_assistant(self, delta: str):
        self.ensure_assistant().append_markdown(delta)
        self._defer_scroll()

    def add_tool_use(self, name: str, args: str):
        self.ensure_assistant().add_tool_use(name, args)
        self._defer_scroll()

    def add_tool_result(self, name: str, ok: bool):
        self.ensure_assistant().add_tool_result(name, ok)
        self._defer_scroll()

    def append_html(self, fragment: str):
        self._clear_welcome()
        line = _SystemLine(fragment)
        self._insert(line)
        self._blocks.append(line)
        self._defer_scroll()

    def clear(self):
        self.end_assistant()
        while self._lay.count() > 1:          # keep the trailing stretch (always last)
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)             # unparent now so it stops painting immediately
                w.deleteLater()
        self._blocks = []
        self._userbubbles = []
        self._current = None
        self._welcome = None

    def toPlainText(self) -> str:
        return "\n".join(b.plain_text() for b in self._blocks)

    # ---- scrolling + sizing ----
    def _on_scroll(self, value):
        sb = self.verticalScrollBar()
        self._stick = value >= sb.maximum() - 4

    def _defer_scroll(self):
        QTimer.singleShot(0, self._scroll_bottom)

    def _scroll_bottom(self):
        if self._stick:
            sb = self.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _apply_bubble_width(self, bubble):
        w = self.viewport().width()
        if w > 0:
            bubble.setMaximumWidth(max(140, int(w * 0.82)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for b in self._userbubbles:
            self._apply_bubble_width(b)


# --------------------------------------------------------------------------- worker
class ChatWorker(QThread):
    """Runs one agent turn off the GUI thread, emitting each RunEvent as it arrives."""

    event = Signal(object)
    finished_turn = Signal()
    progress = Signal(int, int, str)   # current, total, note — per-image during analyze_folder

    def __init__(self, runner, text: str):
        super().__init__()
        self._runner = runner
        self._text = text

    def run(self):
        from scat.progress import run_progress
        try:
            # Ambient progress/cancel sink for the pure-compute analyze turn; emits a queued
            # Qt signal (thread-safe from this worker or the SDK executor thread).
            with run_progress(lambda c, t, note="": self.progress.emit(c, t, note)):
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


# --------------------------------------------------------------------------- dock
class ChatDockWidget(QWidget):
    """The dock contents: status line, a Claude-style conversation transcript, a composer with an
    in-corner send/stop button (Send ↑ toggles to Stop ■ while a turn runs), and model + provider
    pickers below."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runner = None
        self.desc = None
        self.worker = None
        self._build_ui()

    def sizeHint(self):
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

        self.view = _Transcript(on_example=self._use_example)
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
        self.model_combo.setObjectName("ghostPicker")   # Claude-style: text until hover
        for _name, _mid in LATEST_MODELS:
            self.model_combo.addItem(_name, _mid)
        _cur_model = config.get("agent.model", "claude-opus-4-8")
        self.model_combo.setCurrentIndex(
            next((i for i, (_n, m) in enumerate(LATEST_MODELS) if m == _cur_model), 0))
        self.model_combo.setToolTip("Model — always the latest Claude versions")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)  # after setCurrentIndex
        picker_row.addWidget(self.model_combo)

        self.provider_combo = QComboBox()
        self.provider_combo.setObjectName("ghostPicker")   # Claude-style: text until hover
        for _name, _val in _PROVIDERS:
            self.provider_combo.addItem(_name, _val)
        _cur_backend = config.get("agent.backend", "auto")
        self.provider_combo.setCurrentIndex(
            next((i for i, (_n, v) in enumerate(_PROVIDERS) if v == _cur_backend), 0))
        self.provider_combo.setToolTip(
            "Provider — Auto uses your Claude subscription if logged in (no API charges), else the billed API")
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self._refresh_subscription_state()   # mark "not connected" if no Claude login is detected
        picker_row.addWidget(self.provider_combo)

        picker_row.addStretch(1)
        layout.addLayout(picker_row)

    def _use_example(self, text: str):
        """Clicking a welcome example prompt drops it into the composer, ready to send/edit."""
        self.input.setPlainText(text)
        self.input.setFocus()
        cursor = self.input.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.input.setTextCursor(cursor)

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
        """Escape hatch / test seam: drop a raw HTML fragment into the transcript as a system
        line. Production text goes through the typed block API (add_user / append_assistant / …)."""
        self.view.append_html(fragment)

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
        self.view.add_user(text)
        self.view.ensure_assistant()          # show the typing indicator immediately
        self.worker = ChatWorker(self.runner, text)
        self.worker.event.connect(self._on_event)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_turn.connect(self._on_turn_finished)
        self.worker.start()
        self._set_running(True)   # composer read-only; send button becomes Stop

    def _on_send_or_stop(self):
        """The one corner button: Stop while a turn runs, Send otherwise."""
        if self.worker is not None and self.worker.isRunning():
            self._stop()
        else:
            self._send()

    def _on_event(self, ev):
        kind = type(ev).__name__
        if kind in ("TextDelta", "_SyntheticError"):
            self.view.append_assistant(getattr(ev, "text", ""))
        elif kind == "ToolUse":
            name = str(getattr(ev, "name", "tool"))
            args = _compact_input(getattr(ev, "input", {}))
            self.view.add_tool_use(name, args)
        elif kind == "ToolResult":
            name = str(getattr(ev, "name", "tool"))
            self.view.add_tool_result(name, not getattr(ev, "is_error", False))
        elif kind == "TurnDone":
            stop = str(getattr(ev, "stop_reason", ""))
            if stop and stop not in ("end_turn", "tool_use"):
                self.view.append_html(
                    f"<span style='color:{Theme.TEXT_MUTED};font-size:11px'>· {html.escape(stop)}</span>")
            self.view.end_assistant()
        # ToolUseStart: ignored (ToolUse carries the same id + the input)

    def _on_progress(self, current, total, note):
        self.status.setText(f"{note} — {current}/{total}" if note else f"{current}/{total}")

    def _on_turn_finished(self):
        self.view.end_assistant()
        self._set_running(False)   # composer editable again; button back to Send
        self.status.setText(self.desc or "Assistant ready.")
        self.input.setFocus()

    def _stop(self):
        # Halt the in-progress analyze batch at the next image, and end the turn.
        try:
            from scat.progress import request_cancel
            request_cancel()
        except Exception:
            pass
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

    def _refresh_subscription_state(self):
        """Annotate the provider picker's Subscription entry so an un-logged-in subscription reads
        'not connected' without the user having to send a message first. Cheap probe (no CLI spawn),
        and never fatal — if the check can't run, the entry keeps its plain label."""
        try:
            from scat.agent.claude_subscription import subscription_available
            ok, reason = subscription_available()
        except Exception:
            return
        # index 1 is the Subscription provider (see _PROVIDERS); UserRole (the backend value) is
        # untouched — setItemText/ToolTipRole only change what is shown.
        self.provider_combo.setItemText(1, "Subscription" if ok else "Subscription — not connected")
        self.provider_combo.setItemData(
            1, ("Claude subscription (no API charges)" if ok
                else f"Claude subscription not connected ({reason}). Log in with the `claude` CLI, "
                     "or pick the API provider with a key."),
            Qt.ToolTipRole)

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
        self._refresh_subscription_state()   # a login may have changed since the dock opened
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
                from scat.progress import request_cancel
                request_cancel()                       # halt a blocking analyze_batch
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
