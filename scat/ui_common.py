"""
Common UI components shared between main_gui and labeling_gui.
Consolidates Theme, custom widgets, and utility functions.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QSpinBox, QDoubleSpinBox, QComboBox, QTableWidgetItem,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel,
)
from PySide6.QtGui import QWheelEvent, QFontDatabase, QPainter, QColor
from PySide6.QtCore import Qt, QRectF, QSize, QPropertyAnimation, QEasingCurve, Property


# =============================================================================
# Theme Colors - Dark Theme with Coral Accent (DIC2497 #DA4E42)
# =============================================================================
class Theme:
    """SCAT Application Color Theme — dark, with a single coral accent (DIC2497).

    Token discipline (docs/superpowers/specs/2026-07-15-scat-design-elevation.md):
      * Surface tiers are role-named and monotonic in lightness
        (BG_BASE < BG_INSET < BG_SURFACE < BG_HOVER < BG_ACTIVE).
      * PRIMARY / SECONDARY are CHROME tokens — selection, focus, state — never a data marker.
      * NORMAL / ROD / ARTIFACT are SEMANTIC data colors, byte-identical to the report tokens.
    """

    # ---- Chrome accents (UI state only — never a data marker) ----
    PRIMARY = "#DA4E42"        # Coral (DIC2497) — selected tab, focus ring, primary action
    PRIMARY_DARK = "#C44539"
    PRIMARY_LIGHT = "#E8695E"

    SECONDARY = "#636867"      # Gray-green (DIC540) — neutral hover / secondary
    SECONDARY_DARK = "#4E5251" # pressed feedback
    SECONDARY_LIGHT = "#7A7F7E"

    # ---- Semantic deposit colors (IDENTICAL to report --normal/--rod/--artifact) ----
    NORMAL = "#1F9E77"         # Teal — color-blind-safe, ties the pH-axis mid, separates from ROD
    NORMAL_DARK = "#177A5C"
    ROD = "#DA4E42"            # Coral — RODs (matches report --rod)
    ROD_DARK = "#C44539"
    ARTIFACT = "#636867"       # Gray-green — artifacts (matches report --artifact)
    ARTIFACT_DARK = "#4E5251"

    # ---- Surface tiers (role-named, monotonic in lightness) ----
    BG_BASE    = "#0A0A0A"     # window
    BG_INSET   = "#0E0E0E"     # input fields, table alternate rows
    BG_SURFACE = "#171717"     # group boxes / cards — a visible step above BASE
    BG_HOVER   = "#242424"     # hover
    BG_ACTIVE  = "#2E2E2E"     # active / pressed
    # Backwards-compatible aliases (old names → new tiers). Prefer the role names above.
    BG_DARKEST = BG_BASE
    BG_MEDIUM  = BG_INSET
    BG_DARK    = BG_SURFACE
    BG_LIGHT   = BG_HOVER
    BG_LIGHTER = BG_ACTIVE

    # ---- Text (contrast-checked; see spec §6) ----
    TEXT_PRIMARY   = "#EDEDED"  # near-white — softer than pure #FFF
    TEXT_SECONDARY = "#9A9A9A"  # the one muted/secondary token (6.66:1 on BASE)
    TEXT_MUTED     = "#8A8A8A"  # was #5A5A5A (FAIL) → 5.43:1 on SURFACE
    TEXT_DISABLED  = "#6E6E6E"  # disabled labels — 3.27:1 on #1E1E1E (was #404040, 1.61:1)

    # ---- Borders / focus ----
    BORDER = "#2A2A2A"
    BORDER_LIT = "#3A3A3A"     # a lit top edge for dark-theme card depth (drop shadows vanish on near-black)
    BORDER_FOCUS = "#DA4E42"   # focus ring (rendered 2px — 4.67:1 on the inset field)
    FOCUS_RING = "#DA4E42"

    # ---- Scales (spacing 4/8 · radius · weight · font-size) ----
    RADIUS_CONTROL = 6         # button, input, combo, checkbox, menu item, tooltip
    RADIUS_CONTAINER = 8       # group box, tab, table, list, tree, textedit
    RADIUS_PILL = 17           # chat send button (34px control → full round)
    SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5, SPACE_6 = 4, 8, 12, 16, 20, 24
    WEIGHT_BODY, WEIGHT_LABEL, WEIGHT_TITLE = 400, 500, 600
    FS_XS, FS_SM, FS_BODY, FS_TITLE, FS_DISPLAY = 11, 12, 13, 15, 24
    # Letter-spacing tokens (Qt QSS uses px): tighten the largest display numbers, one caps tracking.
    TRACK_DISPLAY = "-0.5px"   # size-specific tightening for hero/display numbers
    TRACK_CAPS = "0.6px"       # uppercase captions (hero kicker, section labels, tile labels)

    # Cached stylesheet
    _cached_app_stylesheet = None
    _cached_labeling_stylesheet = None
    
    @staticmethod
    def button_style(bg_color: str, text_color: str = "#FFFFFF",
                     hover_color: str = None, pressed_color: str = None) -> str:
        """Generate a filled action-button stylesheet with specific colors (run button, the
        semantic labeling buttons, …). A 2px focus ring is reserved as a transparent border so
        keyboard focus never reflows the button. Pass ``pressed_color`` (usually the color's
        own dark shade) so a coloured button darkens on press instead of flashing neutral gray."""
        if hover_color is None:
            hover_color = Theme.BG_ACTIVE
        if pressed_color is None:
            pressed_color = Theme.BG_HOVER
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: 2px solid transparent;
                padding: 8px 16px;
                border-radius: {Theme.RADIUS_CONTROL}px;
                font-weight: bold;
                font-size: {Theme.FS_BODY}px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
            QPushButton:focus {{
                border: 2px solid {Theme.FOCUS_RING};
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: {Theme.TEXT_DISABLED};
            }}
        """
    
    @classmethod
    def get_app_stylesheet(cls) -> str:
        """Return the complete application stylesheet for main_gui."""
        from . import ui_styles  # deferred: avoids a ui_common <-> ui_styles import cycle
        # Memoize (Theme tokens are not mutated at runtime) — mirrors get_labeling_stylesheet;
        # avoids rebuilding the ~430-line stylesheet on every call.
        if cls._cached_app_stylesheet is None:
            cls._cached_app_stylesheet = ui_styles.build_app_stylesheet(cls)
        return cls._cached_app_stylesheet

    @classmethod
    def get_labeling_stylesheet(cls) -> str:
        """Return the stylesheet for labeling_gui (simplified version)."""
        from . import ui_styles  # deferred: avoids a ui_common <-> ui_styles import cycle
        if cls._cached_labeling_stylesheet is None:
            cls._cached_labeling_stylesheet = ui_styles.build_labeling_stylesheet(cls)
        return cls._cached_labeling_stylesheet

    @classmethod
    def primary_button_style(cls) -> str:
        """The app's canonical primary/CTA button skin (coral fill, white text)."""
        return cls.button_style(cls.PRIMARY, "#FFFFFF", cls.PRIMARY_LIGHT, cls.PRIMARY_DARK)


# =============================================================================
# Custom Widgets - SpinBox/ComboBox without scroll wheel
# =============================================================================
class NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores mouse wheel events to prevent accidental changes."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()  # Pass to parent for scrolling


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores mouse wheel events."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores mouse wheel events and always drops down below."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Force popup to always appear below the combobox
        self.view().window().setWindowFlags(
            Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint
        )
    
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()
    
    def showPopup(self):
        """Override to ensure dropdown always appears below."""
        # Call parent showPopup first
        super().showPopup()
        
        # Get the popup and move it below the combobox
        popup = self.view().window()
        pos = self.mapToGlobal(self.rect().bottomLeft())
        popup.move(pos)


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically."""
    
    def __init__(self, value, display_format: str = None):
        if display_format:
            super().__init__(display_format.format(value))
        else:
            super().__init__(str(value))
        self._value = value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._value < other._value
        return super().__lt__(other)


class ToggleSwitch(QCheckBox):
    """An iOS/Apple-style toggle switch. A QCheckBox subclass, so the existing isChecked/
    setChecked/toggled API is unchanged — only the look. No text (pair it with a QLabel in the
    row). The knob slides with a short eased animation, reduced-motion aware."""

    _W, _H = 42, 24            # track size
    _PAD = 3                   # knob inset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self._offset = 1.0 if self.isChecked() else 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        from scat import ui_motion
        target = 1.0 if checked else 0.0
        if ui_motion.motion_reduced():
            self._offset = target
            self.update()
        else:
            self._anim.stop()
            self._anim.setStartValue(self._offset)
            self._anim.setEndValue(target)
            self._anim.start()

    def _get_offset(self):
        return self._offset

    def _set_offset(self, v):
        self._offset = v
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def sizeHint(self):
        return QSize(self._W + 4, self._H + 4)

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        x = (self.width() - self._W) / 2
        y = (self.height() - self._H) / 2
        off = QColor(Theme.BG_ACTIVE)
        on = QColor(Theme.PRIMARY)
        t = self._offset
        track = QColor(int(off.red() + (on.red() - off.red()) * t),
                       int(off.green() + (on.green() - off.green()) * t),
                       int(off.blue() + (on.blue() - off.blue()) * t))
        if not self.isEnabled():
            track = QColor("#1E1E1E")
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(x, y, self._W, self._H), self._H / 2, self._H / 2)
        d = self._H - 2 * self._PAD
        kx = x + self._PAD + t * (self._W - d - 2 * self._PAD)
        p.setBrush(QColor("#FFFFFF") if self.isEnabled() else QColor("#6E6E6E"))
        p.drawEllipse(QRectF(kx, y + self._PAD, d, d))
        p.end()


def setting_row(title: str, control, description: str = None) -> QWidget:
    """An Apple/VS-Code-style settings row: a bold title (+ optional muted description) on the
    left, and the control (toggle, dropdown, …) aligned right. Returns the row widget."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 6, 0, 6)
    h.setSpacing(12)
    text = QVBoxLayout()
    text.setContentsMargins(0, 0, 0, 0)
    text.setSpacing(2)
    lbl = QLabel(title)
    lbl.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-weight: {Theme.WEIGHT_LABEL}; background: transparent;")
    text.addWidget(lbl)
    if description:
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_XS}px; background: transparent;")
        text.addWidget(desc)
    h.addLayout(text, 1)
    h.addWidget(control, 0, Qt.AlignRight | Qt.AlignVCenter)
    return row


class CenteredCap(QWidget):
    """A widget that horizontally centers a width-capped content column. Add content to
    ``.content_layout`` (a QVBoxLayout). Used so single-column forms don't stretch across a
    16:9 / 16:10 display — the column caps at ``max_width`` and is centered with side gutters."""

    def __init__(self, max_width: int = 1080, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addStretch(1)
        self._inner = QWidget()
        self._inner.setMaximumWidth(max_width)
        self.content_layout = QVBoxLayout(self._inner)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        row.addWidget(self._inner, 0)
        row.addStretch(1)

    def add_widget(self, w):
        self.content_layout.addWidget(w)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class CollapsibleSection(QWidget):
    """A titled section whose body shows/hides via a header toggle — progressive disclosure
    for advanced settings. The header is a full-width bar with a rotating chevron; the body
    animates open/closed (height), honoring reduced motion. Add content with ``add_widget`` /
    ``add_layout`` or by laying out on ``.body``."""

    _MAX = 16777215  # Qt's QWIDGETSIZE_MAX

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = expanded
        self._anim = None   # running height animation, stopped on re-toggle

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = QPushButton()
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 9px 12px;
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_CONTROL}px;
                background-color: {Theme.BG_INSET};
                color: {Theme.TEXT_SECONDARY};
                font-weight: {Theme.WEIGHT_TITLE};
            }}
            QPushButton:hover {{ background-color: {Theme.BG_HOVER}; color: {Theme.TEXT_PRIMARY}; }}
            QPushButton:checked {{ color: {Theme.TEXT_PRIMARY}; }}
            QPushButton:focus {{ border: 2px solid {Theme.FOCUS_RING}; padding: 8px 11px; }}
        """)
        self.header.clicked.connect(self._toggle)
        root.addWidget(self.header)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 8, 4, 4)
        self.body_layout.setSpacing(6)
        root.addWidget(self.body)
        self.body.setVisible(expanded)
        self._update_header()

    # -- content API --
    def add_widget(self, w):
        self.body_layout.addWidget(w)

    def add_layout(self, layout):
        self.body_layout.addLayout(layout)

    # -- behaviour --
    def _update_header(self):
        chevron = "▾" if self._expanded else "▸"   # ▾ / ▸
        self.header.setText(f"{chevron}   {self._title}")

    def _toggle(self):
        from scat import ui_motion
        # Stop any in-flight animation first: ui_motion.animate does NOT retarget a prior
        # animation, so a fast collapse-then-expand could otherwise let the collapse's
        # finished handler hide the body AFTER the expand ran (expanded-but-hidden).
        if self._anim is not None:
            try:
                self._anim.stop()   # a manual stop does NOT emit finished, so no stale handler
            except RuntimeError:
                pass                # DeleteWhenStopped may have already freed the C++ object
            self._anim = None
        self._expanded = self.header.isChecked()
        self._update_header()
        if self._expanded:
            self.body.setVisible(True)
            target = self.body.sizeHint().height()
            self._anim = ui_motion.animate(self.body, b"maximumHeight", target,
                                           ui_motion.DUR_TAB, ui_motion.CURVE_OUT, start=0)
            # After opening, lift the cap so the body can grow with its content.
            self._anim.finished.connect(lambda: self.body.setMaximumHeight(self._MAX))
        else:
            start = self.body.height()
            self._anim = ui_motion.animate(self.body, b"maximumHeight", 0,
                                           ui_motion.DUR_TAB, ui_motion.CURVE_OUT, start=start)
            self._anim.finished.connect(lambda: self.body.setVisible(False))


# =============================================================================
# Utility Functions
# =============================================================================
_icon_cache = {}


def icon(name: str, color: str = None, size: int = 20):
    """Load a bundled Material Symbol (``scat/resources/icons/ms_<name>.svg``) as a recolored
    ``QIcon`` — replaces emoji glyphs that render as tofu in the bundled font. Cached. These are
    Google Material Symbols (Apache-2.0). Returns an empty QIcon if the name is missing."""
    from PySide6.QtGui import QIcon, QPixmap, QPainter
    color = color or Theme.TEXT_PRIMARY
    key = (name, color, size)
    if key in _icon_cache:
        return _icon_cache[key]
    ic = QIcon()
    path = Path(__file__).parent / "resources" / "icons" / f"ms_{name}.svg"
    if path.exists():
        try:
            from PySide6.QtSvg import QSvgRenderer
            from PySide6.QtCore import Qt, QByteArray
            svg = path.read_text().replace("<svg ", f'<svg fill="{color}" ', 1)
            r = QSvgRenderer(QByteArray(svg.encode()))
            pm = QPixmap(size, size)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            r.render(p)
            p.end()
            ic = QIcon(pm)
        except Exception:
            ic = QIcon()
    _icon_cache[key] = ic
    return ic


def qss_icon(name: str) -> str:
    """Absolute POSIX path to a bundled QSS icon (``scat/resources/icons/<name>``), for use in
    a QSS ``image: url(...)``. Forward slashes so Qt accepts it on every platform. Qt's QSS
    engine does not support the CSS border-triangle caret trick or ``data:`` URIs (verified),
    so combobox carets and the checkbox glyph are shipped as real SVG files referenced here."""
    return (Path(__file__).parent / "resources" / "icons" / name).as_posix()


def load_custom_fonts():
    """Load custom fonts bundled under scat/resources/fonts/."""
    fonts_dir = Path(__file__).parent / "resources" / "fonts"
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id < 0:
                print(f"Warning: Failed to load font {font_file.name}")


def get_icon_path() -> str:
    """Get the path to the application icon (scat/resources/icon.ico)."""
    for path in (
        Path(__file__).parent / "resources" / "icon.ico",
        Path(__file__).parent.parent / "scat" / "resources" / "icon.ico",
        Path(__file__).parent.parent / "resources" / "icon.ico",
    ):
        if path.exists():
            return str(path)
    return ""
