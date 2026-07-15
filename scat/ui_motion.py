"""Centralized Qt motion for SCAT — the single home for every animation in the GUI.

Why this module exists
----------------------
Qt Style Sheets (QSS) support **no** transitions or animations. All motion in a Qt app is
imperative: ``QPropertyAnimation`` driving a widget/effect property, timed by a
``QEasingCurve``. To keep motion cohesive and reduced-motion-honoring, **no widget site
instantiates ``QPropertyAnimation`` directly** — every animation goes through the helpers
here, which read one reduced-motion gate and use one set of curve/duration tokens.

Design constraints baked in (all confirmed against PySide6 6.10.1)
------------------------------------------------------------------
* **One QGraphicsEffect per widget.** A ``QWidget`` accepts only a single ``QGraphicsEffect``
  at a time, so a button's *press* and *hover* feedback share ONE drop-shadow effect
  (``attach_button_motion``), and a widget that fades must not also carry a shadow. The fade
  helpers refuse to clobber a foreign effect (they bail rather than delete someone's shadow).
* **Animations are parented, not just referenced.** An unparented ``QPropertyAnimation`` with
  no Python reference can be garbage-collected the instant ``start()`` returns, silently
  killing the animation. Every animation here is parented to its target so the C++ side owns
  it; ``DeleteWhenStopped`` frees it when done (and it dies with the target — safe if the
  widget is destroyed mid-flight). Interruptible animations (progress) are stopped-and-
  retargeted, and their Python ref is cleared on ``finished`` so it can never dangle.
* **Reduced motion = no movement, not no feedback.** When reduced, animations run with
  duration 0 (land instantly on the end value); static depth/color still apply.

Curves & durations follow Emil Kowalski's catalog: strong custom ease-out for entrances and
press-release, ease-in-out for on-screen movement, an iOS-style drawer curve for the dock,
linear only for the progress quantity. All durations stay < 300ms.

DO NOT ANIMATE (kept instant — high-frequency / keyboard-initiated):
    * Enter-to-send in the chat composer, and the Send<->Stop mode swap.
    * QShortcut paths (run_analysis, quit) and the labeling 1/2/3 & R/G/F label shortcuts.
    * Transcript ``insertHtml`` appends (streamed tokens — motion would fight the stream).
"""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QPointF, QObject, QEvent
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect
from PySide6.QtGui import QColor

from scat.config import config


# --------------------------------------------------------------------------- gate
def _os_prefers_reduced() -> bool:
    """Best-effort OS reduced-motion probe. Qt exposes no cross-platform signal, so this is
    only wired up on native Windows (SPI_GETCLIENTAREAANIMATION). Elsewhere — including WSLg,
    where SCAT usually runs — it returns False, i.e. ``auto`` means full motion. Documented in
    the config comment; users force reduction with ``ui.reduced_motion:"on"`` or the env var."""
    if sys.platform == "win32":
        try:
            import ctypes
            SPI_GETCLIENTAREAANIMATION = 0x1042
            enabled = ctypes.c_int(1)
            ok = ctypes.windll.user32.SystemParametersInfoW(
                SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(enabled), 0)
            return bool(ok) and enabled.value == 0   # animations disabled → prefers reduced
        except Exception:
            return False
    return False


def motion_reduced() -> bool:
    """True when motion should be reduced. Read live so a Settings change takes effect without
    a restart. Precedence: env var ``SCAT_REDUCED_MOTION`` (truthy) forces reduction; then the
    ``ui.reduced_motion`` config — ``"on"`` reduces, ``"off"`` forces full motion, ``"auto"``
    defers to the OS probe (Windows only; full motion elsewhere)."""
    if os.environ.get("SCAT_REDUCED_MOTION", "") not in ("", "0", "false", "False"):
        return True
    pref = config.get("ui.reduced_motion", "auto")
    if pref == "on":
        return True
    if pref == "off":
        return False
    return _os_prefers_reduced()


# Convenience snapshot for call sites that only need the value once at build time.
REDUCED = motion_reduced()


# ------------------------------------------------------------------------- curves
def _bezier(x1: float, y1: float, x2: float, y2: float) -> QEasingCurve:
    """A CSS-style ``cubic-bezier(x1,y1,x2,y2)`` from (0,0) to (1,1)."""
    c = QEasingCurve(QEasingCurve.BezierSpline)
    c.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return c


CURVE_OUT = _bezier(0.23, 1.0, 0.32, 1.0)      # entering/exiting, hover, press-release
CURVE_INOUT = _bezier(0.77, 0.0, 0.175, 1.0)   # morphing / on-screen movement
CURVE_DRAWER = _bezier(0.32, 0.72, 0.0, 1.0)   # dock open/close
CURVE_LINEAR = QEasingCurve(QEasingCurve.Linear)  # progress quantity ONLY
# Never QEasingCurve.InQuad/InCubic (ease-in — sluggish; delays the frame the user watches).

# Durations (ms) — all < 300.
DUR_PRESS = 120
DUR_HOVER = 140
DUR_FADE = 160
DUR_TAB = 200
DUR_PROGRESS = 200
DUR_DOCK = 240


# ---------------------------------------------------------------------- core anim
def animate(target: QObject, prop: bytes, end, dur: int,
            curve: QEasingCurve = CURVE_OUT, start=None) -> QPropertyAnimation:
    """Animate ``target``'s ``prop`` to ``end`` over ``dur`` ms. Parented to ``target`` so it
    survives GC and dies with the target. When motion is reduced, duration is 0 (instant land).
    For a property that is updated repeatedly (progress), use ``animate_value`` instead — it
    retargets safely."""
    anim = QPropertyAnimation(target, prop, target)   # parent=target → C++ owns it
    anim.setDuration(0 if motion_reduced() else max(0, dur))
    anim.setEasingCurve(curve)
    if start is not None:
        anim.setStartValue(start)
    anim.setEndValue(end)
    anim.start(QPropertyAnimation.DeleteWhenStopped)
    return anim


def animate_value(bar, value: int) -> QPropertyAnimation:
    """Ease a ``QProgressBar`` from its current value to ``value`` (linear — a quantity).
    Fully interruptible: stops and retargets any prior value animation from the *current*
    value, and clears the stored ref on ``finished`` so it can never dangle to a freed object."""
    prev = getattr(bar, "_scat_value_anim", None)
    if prev is not None:
        try:
            prev.stop()          # emits finished → _clear drops the ref
        except RuntimeError:
            pass                 # C++ side already gone — nothing to stop
    anim = QPropertyAnimation(bar, b"value", bar)
    anim.setDuration(0 if motion_reduced() else DUR_PROGRESS)
    anim.setEasingCurve(CURVE_LINEAR)
    anim.setStartValue(bar.value())
    anim.setEndValue(int(value))

    def _clear():
        if getattr(bar, "_scat_value_anim", None) is anim:
            bar._scat_value_anim = None

    anim.finished.connect(_clear)
    anim.start(QPropertyAnimation.DeleteWhenStopped)
    bar._scat_value_anim = anim
    return anim


# ------------------------------------------------------------------- opacity fades
def _foreign_effect(widget) -> bool:
    """True if the widget carries a graphics effect that is NOT our managed opacity effect —
    fading would delete it, which we refuse to do (one-effect-per-widget)."""
    eff = widget.graphicsEffect()
    return eff is not None and eff is not getattr(widget, "_scat_opacity_effect", None)


def _managed_opacity_effect(widget) -> QGraphicsOpacityEffect:
    eff = getattr(widget, "_scat_opacity_effect", None)
    if eff is None:
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
        widget._scat_opacity_effect = eff
    return eff


def _run_opacity(widget, end: float, dur: int, curve: QEasingCurve, on_finished=None):
    """Drive the managed opacity effect to ``end``; on settle-to-opaque, remove the effect so
    the widget paints natively again (a lingering opacity effect renders to a pixmap on every
    paint). One effect + one anim per widget, both stashed on the widget."""
    prev = getattr(widget, "_scat_opacity_anim", None)
    if prev is not None:
        try:
            prev.stop()
        except RuntimeError:
            pass
    eff = _managed_opacity_effect(widget)
    eff.setEnabled(True)

    def _done():
        if on_finished is not None:
            on_finished()
        if abs(eff.opacity() - 1.0) < 1e-3 and widget.graphicsEffect() is eff:
            widget.setGraphicsEffect(None)
            widget._scat_opacity_effect = None
        if getattr(widget, "_scat_opacity_anim", None) is anim:
            widget._scat_opacity_anim = None

    anim = QPropertyAnimation(eff, b"opacity", eff)
    anim.setDuration(0 if motion_reduced() else max(0, dur))
    anim.setEasingCurve(curve)
    anim.setStartValue(eff.opacity())
    anim.setEndValue(float(end))
    anim.finished.connect(_done)
    anim.start(QPropertyAnimation.DeleteWhenStopped)
    widget._scat_opacity_anim = anim
    return anim


def fade_in(widget, dur: int = DUR_TAB, curve: QEasingCurve = CURVE_OUT):
    """Show ``widget`` and fade it to opaque. Used for tab-pane crossfade and dock reveal.
    No-op fade (just show) if motion is reduced or the widget carries a foreign effect."""
    if not widget.isVisible():
        widget.setVisible(True)
    if motion_reduced() or _foreign_effect(widget):
        if not _foreign_effect(widget):     # tidy any leftover managed effect
            widget.setGraphicsEffect(None)
            widget._scat_opacity_effect = None
        return None
    _managed_opacity_effect(widget).setOpacity(0.0)   # seed transparent so entry is visible
    return _run_opacity(widget, 1.0, dur, curve)


def fade_out(widget, on_finished=None, dur: int = DUR_DOCK, curve: QEasingCurve = CURVE_DRAWER):
    """Fade ``widget`` to transparent, then call ``on_finished`` (e.g. to hide it). If motion
    is reduced or a foreign effect is present, skip straight to ``on_finished``."""
    if motion_reduced() or _foreign_effect(widget):
        if on_finished is not None:
            on_finished()
        return None
    return _run_opacity(widget, 0.0, dur, curve, on_finished=on_finished)


# ------------------------------------------------------------- button press/hover
class _ButtonMotion(QObject):
    """Owns ONE ``QGraphicsDropShadowEffect`` on a button and animates it between three depth
    states — rest, hover (lift), pressed (depress) — from pointer events. A single effect
    (Qt allows only one per widget) drives both hover and press feedback."""

    def __init__(self, btn, *, rest, hover, press, color):
        super().__init__(btn)                 # parented → not garbage-collected
        self._rest, self._hover, self._press = rest, hover, press   # each (blur, y_offset)
        self._hovered = False
        self._banim = None
        self._yanim = None

        eff = QGraphicsDropShadowEffect(btn)
        eff.setColor(QColor(*color))
        eff.setXOffset(0)
        eff.setBlurRadius(rest[0])
        eff.setYOffset(rest[1])
        btn.setGraphicsEffect(eff)
        self._eff = eff
        btn.installEventFilter(self)

    def _to(self, state, dur):
        blur, y = state
        for a in (self._banim, self._yanim):
            if a is not None:
                try:
                    a.stop()
                except RuntimeError:
                    pass
        if motion_reduced():
            self._eff.setBlurRadius(float(blur))
            self._eff.setYOffset(float(y))
            self._banim = self._yanim = None
            return
        self._banim = QPropertyAnimation(self._eff, b"blurRadius", self._eff)
        self._banim.setDuration(dur)
        self._banim.setEasingCurve(CURVE_OUT)
        self._banim.setEndValue(float(blur))
        self._banim.start(QPropertyAnimation.DeleteWhenStopped)
        self._yanim = QPropertyAnimation(self._eff, b"yOffset", self._eff)
        self._yanim.setDuration(dur)
        self._yanim.setEasingCurve(CURVE_OUT)
        self._yanim.setEndValue(float(y))
        self._yanim.start(QPropertyAnimation.DeleteWhenStopped)

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.Enter:
            self._hovered = True
            self._to(self._hover, DUR_HOVER)
        elif t == QEvent.Leave:
            self._hovered = False
            self._to(self._rest, DUR_HOVER)
        elif t == QEvent.MouseButtonPress:
            self._to(self._press, DUR_PRESS)
        elif t == QEvent.MouseButtonRelease:
            self._to(self._hover if self._hovered else self._rest, DUR_HOVER)
        return False   # never consume — the button still gets the event


def attach_button_motion(btn, *, primary: bool = False):
    """Give ``btn`` responsive depth: it lifts on hover and depresses on press, through a
    single shared drop-shadow effect. ``primary=True`` uses a coral-tinted shadow (the accent
    action button); otherwise a neutral dark shadow. Reserve for primary / card surfaces —
    not every gray secondary button. Returns the controller (parented to ``btn``)."""
    if primary:
        color = (218, 78, 66, 110)          # coral, DIC2497
        rest, hover, press = (16, 3), (22, 5), (6, 1)
    else:
        color = (0, 0, 0, 150)
        rest, hover, press = (12, 2), (20, 4), (6, 1)
    return _ButtonMotion(btn, rest=rest, hover=hover, press=press, color=color)


def apply_ui_polish(root, *, cursors: bool = True, elevate: bool = False):
    """Post-build polish that QSS can't express: a pointing-hand cursor on interactive widgets
    (QSS has no ``cursor`` property). Call once after a window's UI is constructed.

    ``elevate`` (drop-shadow on top-level cards) defaults OFF: on SCAT's near-black theme a
    black drop shadow is invisible against a near-black background (and costs a blurred pixmap
    per card). Card depth instead comes from the surface-tier lift (``BG_SURFACE`` above
    ``BG_BASE``) plus a lit top-edge border in the stylesheet. Elevation is only applied to
    group boxes NOT nested inside another group box, for the rare lighter-surface context."""
    from PySide6.QtWidgets import (QPushButton, QComboBox, QCheckBox, QRadioButton,
                                   QToolButton, QGroupBox, QTabBar)
    from PySide6.QtCore import Qt
    if cursors:
        for cls_ in (QPushButton, QComboBox, QCheckBox, QRadioButton, QToolButton):
            for w in root.findChildren(cls_):
                w.setCursor(Qt.PointingHandCursor)
        for tb in root.findChildren(QTabBar):
            tb.setCursor(Qt.PointingHandCursor)
    if elevate:
        for gb in root.findChildren(QGroupBox):
            p = gb.parentWidget()
            while p is not None and not isinstance(p, QGroupBox):
                p = p.parentWidget()
            if p is None:              # not nested in a group box → a top-level card
                attach_elevation(gb)


def attach_elevation(widget, *, blur: int = 18, y: int = 3, alpha: int = 120):
    """Attach a static soft drop shadow so a container (e.g. a QGroupBox) reads as a raised
    surface — QSS cannot do ``box-shadow``. No animation; pure depth. Skipped if the widget
    already has a graphics effect (one-effect-per-widget). The caller must leave layout margin
    around the widget (>= the blur radius/2) so the shadow is not clipped by the parent."""
    if widget.graphicsEffect() is not None:
        return None
    eff = QGraphicsDropShadowEffect(widget)
    eff.setColor(QColor(0, 0, 0, alpha))
    eff.setBlurRadius(blur)
    eff.setXOffset(0)
    eff.setYOffset(y)
    widget.setGraphicsEffect(eff)
    return eff
