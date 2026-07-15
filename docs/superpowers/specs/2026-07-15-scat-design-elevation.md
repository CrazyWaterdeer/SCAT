<!-- Source: parallel 6-auditor design workflow (wf_d60fd054-071) synthesized by lead agent.
     Commit at authoring: 47d42ad. Scope locked by Jin: Qt GUI + HTML report; refine current
     identity (dark+coral Qt, light-paper Imajin report); subtle/functional motion; reduced-motion honored.
     Qt animation primitives (BezierSpline easing, duration=0 reduced-motion, graphics-effect anims)
     verified working in PySide6 6.10.1 before authoring. -->

# SCAT Design Elevation — Master Implementation Spec

Single source of truth. Merged from six audits (2× Qt visual/motion/component, 2× HTML report, 1× cross-cut). Conflicts resolved inline with a stated decision. Locked decisions (dark+coral Qt, light-paper Imajin report, subtle/functional motion, reduced-motion honored) are treated as fixed constraints, not re-opened.

---

## 1. Design intent

**Qt app — a crisp scientific instrument, not a prototype.** Today the dark+coral identity is real but undisciplined: flat `#121212`-on-`#0A0A0A` with no elevation, every label bold, five ad-hoc radii, coral leaking into press states and data colors, and near-invisible keyboard focus. "Refined" here means a genuine token layer — surface tiers with soft depth, a 400/500/600 weight ladder, a 6/8px radius scale on a 4/8px spacing grid, one 2px coral focus ring — plus subtle Qt-native micro-interactions (eased progress, dock fade, press elevation) that all pass through one reduced-motion-gated helper.

**HTML report — a print-grade paper document.** The `_REPORT_CSS` token system is well-conceived but systematically bypassed by inline Material-indigo/Material-green hexes and off-token grays, and it has zero print protection and no motion guardrails. "Elevated" means promoting the appendix/comparison inline styles into token-driven classes (`.data-table`, `.result-box`, `.callout`, `.section h3`), landing `break-inside: avoid` so figures and stat cards never orphan, and adding a single reduced-motion-safe, print-safe scroll reveal.

**One story across both.** Semantic NORMAL/ROD/ARTIFACT must be byte-identical on both surfaces; muted text must funnel through one token per surface; motion must be centralized and reduced-motion-honored before any transition ships.

---

## 2. Shared design tokens

### (a) Qt — refined `Theme` token set (`scat/ui_common.py`)

Replace the current luminance-inconsistent background ladder and add scales. All values exact.

```python
# ---- Surface tiers (role-named, monotonic in lightness) ----
BG_BASE     = "#0A0A0A"   # window (was BG_DARKEST)
BG_INSET    = "#0E0E0E"   # input fields, table alt-rows (was BG_MEDIUM #101010)
BG_SURFACE  = "#171717"   # group boxes / cards  (was BG_DARK #121212 — raised for a visible step)
BG_HOVER    = "#242424"   # hover                (was BG_LIGHT)
BG_ACTIVE   = "#2E2E2E"   # active/pressed        (was BG_LIGHTER)
# NOTE: "MEDIUM darker than DARK" naming trap is removed. Update every BG_DARKEST/
# BG_DARK/BG_MEDIUM/BG_LIGHT/BG_LIGHTER reference in both stylesheets to the new names.

# ---- Chrome accents (UI state only — never a data marker) ----
PRIMARY        = "#DA4E42"   # coral, selection/active/focus
PRIMARY_LIGHT  = "#E8695E"
SECONDARY      = "#636867"
SECONDARY_DARK = "#4E5251"   # (existing) pressed feedback
SECONDARY_LIGHT= "#7A7F7E"

# ---- Semantic deposit colors (IDENTICAL to report tokens — see decision) ----
NORMAL      = "#1F9E77"   # was #4CAF50  → teal wins (color-blind-safe, ties pH axis)
NORMAL_DARK = "#177A5C"   # was #388E3C
ROD         = "#DA4E42"   # unchanged (matches report --rod)
ARTIFACT    = "#636867"   # unchanged (matches report --artifact)

# ---- Text ----
TEXT_PRIMARY   = "#EDEDED"   # (existing)
TEXT_SECONDARY = "#9A9A9A"   # 6.66:1 on BASE — the ONE muted/secondary token for code
TEXT_MUTED     = "#8A8A8A"   # was #5A5A5A (2.72:1 FAIL) → 5.43:1 on #171717
TEXT_DISABLED  = "#6E6E6E"   # disabled labels: 3.27:1 on #1E1E1E (was #404040 = 1.61:1)

# ---- Weight scale (Noto Sans) ----
WEIGHT_BODY  = 400
WEIGHT_LABEL = 500
WEIGHT_TITLE = 600
# Bundle medium (500) + semibold (600) faces into resources/fonts to make 500/600 real.

# ---- Radius scale ----
RADIUS_CONTROL   = 6    # button, input, combo, checkbox, menu item, tooltip
RADIUS_CONTAINER = 8    # group box, tab pane/tab, table, list, tree, textedit
RADIUS_PILL      = 17   # chat send button (34px control → full round)
# Radio indicator stays circular: radius = half its indicator width (9px for an 18px box).

# ---- Spacing scale (4/8 rhythm) ----
SPACE_1=4; SPACE_2=8; SPACE_3=12; SPACE_4=16; SPACE_5=20; SPACE_6=24

# ---- Font-size scale ----
FS_XS=11; FS_SM=12; FS_BODY=13; FS_TITLE=15; FS_DISPLAY=24

# ---- Focus ----
FOCUS_RING = "#DA4E42"   # 2px, 4.67:1 on #101010 field
```

**Depth (QSS can't do box-shadow):** attach a `QGraphicsDropShadowEffect(blurRadius=24, color=QColor(0,0,0,140), xOffset=0, yOffset=2)` to each `QGroupBox`.

### (b) Report CSS token additions/renames (`scat/report.py`, `_REPORT_CSS`)

Add to `:root`, then remap every stray hex to the token:

```css
--fill:      #F1F0ED;   /* warm inset panel (replaces #f5f5f5) */
--ok-bg:     #E7F2EC;   /* significance callout bg (replaces Material #e8f5e9) */
--ok-line:   #1F9E77;   /* = --normal (replaces #4caf50) */
--ok-ink:    #176B4E;   /* callout heading (replaces #2e7d32) */
--track-caps: 0.06em;   /* single uppercase-caption tracking */
```

| Stray hex | Where | → Token |
|---|---|---|
| `#3949ab` (h3) | 975, 1178 | `--accent` `#2F6B9E` |
| `#e8eaf6` (thead) | 1194, 1283 | `--accent-soft` `#EAF1F6` |
| `#ddd` (~25 table borders) | 1195–1337 | `--hair` `#E4E3E0` |
| `#666` (muted text ×5) | 1044, 1092, 1105, 1152, 1249 | `--muted` `#5A5A5A` |
| `#f5f5f5` (inset panels ×3) | 1090, 1103, 1328 | `--fill` `#F1F0ED` |
| `#e8f5e9 / #4caf50 / #2e7d32` | 1124, 1234, 1250 | `--ok-bg / --ok-line / --ok-ink` |
| `#e0e0e0` (separator) | 1071 | `--hair` `#E4E3E0` |
| `#fff3e0` (caution row) | 1232, 1251 | `--warn-bg` `#FBF1DC` |
| `#fff` (row) | 1236 | `--surface` `#FFFFFF` |

### Cross-surface semantic decision (RESOLVED)

- **NORMAL → `#1F9E77` on both surfaces.** Qt was `#4CAF50`; report is `#1F9E77`. **Teal wins** (change Qt, not report): it equals `--ph-mid`, so it ties into the pH story; its blue channel separates it from red ROD far better for deutan/protan viewers; and it still clears text contrast on dark (5.54:1 on `#171717`). Set `NORMAL="#1F9E77"`, `NORMAL_DARK="#177A5C"`.
- **ROD `#DA4E42` and ARTIFACT `#636867` stay byte-identical across surfaces** (already match). **Auditor-1's proposal to split the deposit ROD to `#E8695E` and ARTIFACT to `#6E7A78` is DECLINED.** Reason: the locked direction is "keep the palette / one story," and the cross-cut audit explicitly requires the three semantics identical on both surfaces. Introducing a 4th coral to disambiguate data-vs-chrome costs more than the marginal ambiguity it removes. Mitigation is contextual, not chromatic: PRIMARY/SECONDARY are documented as **chrome-only** tokens and coral is pulled OUT of transient press states (see Qt #6) so the remaining coral in chrome reads as "selected," while coral-as-data appears only on labeling markers.

---

## 3. Motion system (foundation — build first)

### Qt: new module `scat/ui_motion.py` (sibling of `ui_common.py`)

The single home for all Qt motion. Every animation site calls `animate()`; no site instantiates `QPropertyAnimation` directly. QSS supports no transitions — all motion is `QPropertyAnimation` on widget/effect properties.

```python
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QPointF
from PySide6.QtWidgets import QGraphicsDropShadowEffect
from PySide6.QtGui import QColor
from scat.config import config
import os

# Reduced-motion gate (Qt has no prefers-reduced-motion signal — config + env, one place)
REDUCED = (config.get("ui.reduced_motion", "auto") == "on") or \
          (os.environ.get("SCAT_REDUCED_MOTION", "") not in ("", "0", "false"))

def _bezier(x1, y1, x2, y2):
    c = QEasingCurve(QEasingCurve.BezierSpline)
    c.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return c

CURVE_OUT    = _bezier(0.23, 1.0, 0.32, 1.0)     # entering/exiting, hover, press-release (≈OutQuint)
CURVE_INOUT  = _bezier(0.77, 0.0, 0.175, 1.0)    # morphing/moving
CURVE_DRAWER = _bezier(0.32, 0.72, 0.0, 1.0)     # dock open/close
CURVE_LINEAR = QEasingCurve(QEasingCurve.Linear) # progress ONLY
# Never QEasingCurve.InQuad/InCubic (ease-in = sluggish).

DUR_PRESS=120; DUR_HOVER=140; DUR_FADE=160; DUR_TAB=200; DUR_PROGRESS=200; DUR_DOCK=240  # ms, all <300

def animate(target, prop: bytes, end, dur: int, curve=CURVE_OUT, start=None):
    a = QPropertyAnimation(target, prop)
    a.setDuration(0 if REDUCED else dur)     # REDUCED → duration 0: lands on end value, no movement
    a.setEasingCurve(curve)
    if start is not None:
        a.setStartValue(start)
    a.setEndValue(end)
    a.start(QPropertyAnimation.DeleteWhenStopped)
    return a

def attach_press_feedback(btn, accent=QColor(218,78,66,110)):
    eff = QGraphicsDropShadowEffect(btn); eff.setBlurRadius(16); eff.setOffset(0,3); eff.setColor(accent)
    btn.setGraphicsEffect(eff)   # eventFilter on MouseButtonPress→blur6/y1, Release→blur16/y3

def attach_hover_elevation(w, rest_blur=12, hover_blur=22, rest_y=2, hover_y=5):
    ...  # eventFilter on QEvent.Enter→hover_blur/hover_y, QEvent.Leave→rest_blur/rest_y (DUR_HOVER, CURVE_OUT)

# DO-NOT-ANIMATE (comment block): Enter-to-send submit; Send↔Stop mode swap;
# QShortcut run_analysis/quit; labeling 1/2/3 & R/G/F label shortcuts. Keep instant.
```

Add to `scat/config.py` `DEFAULT_CONFIG` after the `"window"` block: `"ui": {"reduced_motion": "auto"}` (auto|on|off).

### Report: CSS motion tokens + guardrails (`scat/report.py`, `_REPORT_CSS`)

Add to `:root`:

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1);
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
--dur-hover: 150ms;
--dur-reveal: 360ms;
--reveal-stagger: 60ms;
```

Append after `.footer`:

```css
/* Scroll reveal — hidden state gated behind JS-added .js-reveal; no-JS shows all */
.js-reveal .section, .js-reveal .plot-container {
  opacity: 0; transform: translateY(8px);
  transition: opacity var(--dur-reveal) var(--ease-out), transform var(--dur-reveal) var(--ease-out);
  will-change: transform, opacity;
}
.js-reveal .section.is-visible, .js-reveal .plot-container.is-visible { opacity: 1; transform: none; }

@media (prefers-reduced-motion: reduce) {
  .js-reveal .section, .js-reveal .plot-container { opacity: 1; transform: none; transition: none; }
  .stat-card, .plot-container img, tbody tr { transition: none; }
  .stat-card:hover, .plot-container img:hover { transform: none; }
}
@media print {
  * { animation: none !important; transition: none !important; }
  .js-reveal .section, .js-reveal .plot-container { opacity: 1 !important; transform: none !important; }
  .section, .stat-card, .plot-container, table, .highlight { page-break-inside: avoid; break-inside: avoid; }
  tbody tr:hover { background: transparent; }
}
```

IntersectionObserver script before `</body>` (early-returns on reduced-motion / missing IO; adds `.js-reveal`; 60ms per-batch stagger capped at 4; `unobserve` once; `rootMargin:'0px 0px -8% 0px'`, `threshold:0.05`). See report backlog #R7.

---

## 4. Qt GUI backlog

Ordered by leverage (impact ÷ effort) within each group.

### Visual

| # | Sev | Track | Location | Change (exact) | Effort |
|---|---|---|---|---|---|
| Q1 | MED | qt-visual | `ui_common.py:41-45` | Rename+revalue BG ladder → `BG_BASE #0A0A0A`, `BG_INSET #0E0E0E`, `BG_SURFACE #171717`, `BG_HOVER #242424`, `BG_ACTIVE #2E2E2E`. Update all refs in both stylesheets. | S |
| Q2 | MED | qt-visual | `labeling_gui.py:503, 601` | `QPushButton:pressed`/`QToolButton:pressed { background-color: {cls.SECONDARY_DARK}; }` (was `PRIMARY`) — match app sheet line 176; pull coral out of transient press. | S |
| Q3 | HIGH | qt-visual | `ui_common.py:245`; labeling `:573` | Base `QLabel { font-weight: 400; }`. Add weight tokens 400/500/600. Apply 600 to `GroupBox::title` (141) + section headers; 500 to form labels via `setProperty('role','formLabel')` + `QLabel[role="formLabel"]{font-weight:500;}`. | M |
| Q4 | HIGH | qt-visual | `ui_common.py:41` | Raise card surface to `#171717`; attach `QGraphicsDropShadowEffect(blurRadius=24, color=QColor(0,0,0,140), xOffset=0, yOffset=2)` to each `QGroupBox`. | M |
| Q5 | MED | qt-visual | `ui_common.py:169, 188…`; labeling `:494…` | Apply radius scale: controls `RADIUS_CONTROL=6`, containers `RADIUS_CONTAINER=8`, send-btn `17`. Unify button padding to `8px 16px` in both sheets. **Radius conflict resolved:** controls=6 (not 4) — see note. | M |
| Q6 | MED | qt-visual | `ui_common.py:117,138,441,70,168` | Snap off-grid values: tab padding `12px 24px` (was `12px 28px`), menu item `8px 24px` (was `10px 24px`), `button_style` padding `8px 16px` (was `10px 20px`). Replace scattered `11/13/24px` literals with `FS_*` tokens. | M |
| Q7 | MED | qt-visual | `ui_common.py:50` | `TEXT_MUTED = "#8A8A8A"` (was `#5A5A5A`; see a11y X2). | S |

### Components

| # | Sev | Track | Location | Change (exact) | Effort |
|---|---|---|---|---|---|
| Q8 | HIGH | qt-component | `ui_common.py:220`; labeling `:624-628` | Draw caret (both sheets): `QComboBox::down-arrow { width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent; border-top:6px solid #9A9A9A; margin-right:2px; }` + `:hover { border-top-color:#FFFFFF; }`. Remove `image:none`. | S |
| Q9 | HIGH | qt-component | `labeling_gui.py:788…` | Route the three semantic buttons through `Theme.button_style(...)` so `:hover/:pressed` are defined (adds hover/pressed shades) and the global gray hover no longer repaints the green Normal button. e.g. `button_style(Theme.NORMAL, "#FFFFFF", Theme.NORMAL_DARK)`. | S |
| Q10 | MED | qt-component | `ui_common.py:325,341`; labeling `:657,674` | Scrollbar handle `background-color: #484848` (was `#2E2E2E`, ~1.4:1). Keep hover `#636867`. | S |
| Q11 | MED | qt-component | `ui_common.py:495` | Labeling `QPushButton { min-height: 28px; }` (was 24). Add `:disabled { background:#1E1E1E; color:#6E6E6E; border-color:#2A2A2A; }` and `:focus { border:2px solid #DA4E42; }` (see X1 for padding comp). | S |
| Q12 | MED | qt-component | `ui_common.py:148`; labeling `:483-487` | Extract `_groupbox_title_qss()`; derive gradient stops from tokens: `stop:0 {BG_BASE}, stop:0.55 {BG_BASE}, stop:0.65 {BG_SURFACE}, stop:1 {BG_SURFACE}`. Remove magic `rgba(10,10,10)/rgba(18,18,18)`. | S |
| Q13 | MED | qt-component | `ui_common.py:249-268` | Add `QTableWidget { alternate-background-color:#0E0E0E; }` + `::item:hover { background:#1A1A1A; }`; call `table.setAlternatingRowColors(True)` where tables are built. | M |
| Q14 | HIGH | qt-component | `main_gui.py`, `labeling_gui.py` (post-build) | Pointer cursor app-wide: `for w in self.findChildren((QPushButton,QComboBox,QCheckBox,QRadioButton,QToolButton)): w.setCursor(Qt.PointingHandCursor)` and `tabBar().setCursor(Qt.PointingHandCursor)`. (QSS has no `cursor`.) | M |
| Q15 | LOW | qt-component | `ui_common.py:389`; labeling `:645-648` | Checked checkbox glyph via data-URI SVG check (`M2 6.5 L5 9.5 L10 3`, white, stroke-width 2). Fallback: ship `resources/check.svg` if data-URI won't render in PySide6. | S |
| Q16 | LOW | qt-component | `ui_common.py:709` | Restore popup elevation without OS shadow: in `NoScrollComboBox.__init__`, `QGraphicsDropShadowEffect(blurRadius=24, xOffset=0, yOffset=6, color=QColor(0,0,0,160))` on `self.view()`. Keep frameless flags. | M |
| Q17 | MED | qt-component | `ui_common.py:605+` (refactor) | Extract shared component rules (button/input/combo/scrollbar/table/checkbox) into classmethods returning QSS from tokens; both `get_app_stylesheet`/`get_labeling_stylesheet` compose from them, overriding only deliberate diffs. **Do LAST** (absorbs Q5/Q8/Q11 divergences). | L |

### Motion (depends on `scat/ui_motion.py`)

| # | Sev | Track | Location | Change (exact) | Effort |
|---|---|---|---|---|---|
| Q18 | HIGH | qt-motion | **new** `scat/ui_motion.py` | Build the module in §3 (REDUCED flag, curves, durations, `animate()`, `attach_press_feedback`, `attach_hover_elevation`, do-not-animate comment). Everything below is a 2-line call into it. | M |
| Q19 | HIGH | qt-motion | `main_gui.py:1289` (+1899,1917,1941,1952,1960) | Replace bare `setValue` with `self._progress_anim = animate(self.progress, b'value', current, DUR_PROGRESS, CURVE_LINEAR, start=self.progress.value())`. Keep the ref for interruptibility. LINEAR (constant quantity). REDUCED snaps. | S |
| Q20 | HIGH | qt-motion | `main_gui.py:946` | `attach_press_feedback(self.run_btn)`: resting shadow blur16/y3/accent `rgba(218,78,66,110)`; press→blur6/y1 (`DUR_PRESS`,`CURVE_OUT`); release→blur16/y3 (`DUR_HOVER`,`CURVE_OUT`). | M |
| Q21 | MED | qt-motion | `main_gui.py:2152` | Fade dock: add `self._dock_fx=QGraphicsOpacityEffect(self.chat_widget)` once; toggle animates `b'opacity'` 0→1 / 1→0 over `DUR_DOCK` `CURVE_DRAWER`; hide after finish. | M |
| Q22 | MED | qt-motion | `main_gui.py:2070` | `tabs.currentChanged.connect(self._fade_current_tab)`: put `QGraphicsOpacityEffect` on new pane, `animate(fx,b'opacity',1.0,DUR_TAB,CURVE_OUT,start=0.0)`, `finished→setGraphicsEffect(None)`. Covers programmatic `setCurrentWidget`. | M |
| Q23 | MED | qt-motion | `main_gui.py:944, 1490` | `attach_hover_elevation(run_btn)` and thumbnail buttons (rest blur12/y2 → hover blur22/y5, `DUR_HOVER`,`CURVE_OUT`). Primary/card surfaces only, not every gray secondary. | M |
| Q24 | LOW | qt-motion | `agent/chat_widget.py:264,280,341,345,412` | Route status writes through `set_status()`; add `self._status_fx=QGraphicsOpacityEffect(self.status)`; fade `0.35→1.0` over `DUR_FADE` `CURVE_OUT`. Do NOT animate transcript `insertHtml`. | S |
| Q25 | MED | qt-motion | `chat_widget.py:76,214,234`; `main_gui.py:2158`; labeling shortcuts | Enforce the do-not-animate list: Enter-to-send, Send↔Stop mode swap (instant; pointer-only hover elevation OK), `QShortcut` paths, labeling 1/2/3 & R/G/F. Document in `ui_motion.py`. | S |

**Radius conflict (Q5/Q17) resolved:** Auditor-1 proposed controls=4px; the component auditor proposed controls=6px. **Decision: controls=6, containers=8, pill=17.** A 6px control radius is a softer, more current read and gives a cleaner two-step scale; the radio indicator stays circular (radius = half its box). Both stylesheets converge on this.

---

## 5. HTML report backlog

Ordered by leverage. All `scat/report.py`.

| # | Sev | Track | Location | Change (exact) | Effort |
|---|---|---|---|---|---|
| R1 | HIGH | html-visual | `report.py:975, 1178` | Delete inline `color:#3949ab`; emit plain `<h3>`. Add `.section h3 { font-family:var(--serif); font-size:1.05rem; font-weight:600; letter-spacing:-0.005em; color:var(--accent); margin-top:28px; margin-bottom:12px; }`. `#3949ab → --accent #2F6B9E`. | S |
| R2 | HIGH | html-visual | `report.py:1194, 1283` | `.data-table thead tr { background: var(--accent-soft); }` — `#e8eaf6 → --accent-soft #EAF1F6`. | S |
| R3 | HIGH | html-visual | `report.py:196` | Append print block: `.section,.stat-card,.plot-container,.highlight,tr { break-inside:avoid; }` + `.section h2,.section h3 { break-after:avoid; }` + `@media print { body{max-width:none;padding:0;} .section{border:none;box-shadow:none;break-inside:avoid;} tbody tr:hover{background:none;} } .plot-container img{break-inside:avoid;}`. (Consolidate with §3 print block.) | M |
| R4 | MED | html-visual | `report.py:1044,1092,1105,1152,1249` | `#666 → var(--muted) #5A5A5A`. Add `.section-intro { color:var(--muted); margin-bottom:20px; }` and `.appendix-ref { color:var(--muted); font-size:0.82rem; margin-left:10px; }` (0.9em→0.82rem). | S |
| R5 | MED | html-visual | `report.py:1090,1103,1328` | `<div class="result-box">` + `.result-box { background:var(--fill); border:1px solid var(--hair); border-radius:8px; padding:12px 14px; margin-top:12px; }`. `#f5f5f5 → --fill #F1F0ED`; 5px→8px radius; adds missing hairline. | M |
| R6 | MED | html-visual | `report.py:1124,1234,1250` (+overload 1370) | Add base `.callout { padding:15px 18px; border-radius:8px; border-left:4px solid; margin:16px 0; }` + `.callout--warn` (warn tokens) + `.callout--ok { background:var(--ok-bg); border-left-color:var(--ok-line); } .callout--ok h4 { color:var(--ok-ink); margin-top:0; }`. Retag `.highlight` uses → `.callout callout--warn`; significance box → `.callout callout--ok`, no inline color. `#4caf50→#1F9E77`, `#2e7d32→#176B4E`, `#e8f5e9→#E7F2EC`. | M |
| R7 | HIGH | html-motion | `report.py:52, 196, 204` | Add motion tokens (§3) to `:root`; add `.js-reveal` reveal CSS after `.footer`; inject IntersectionObserver script before `</body>` (bails on reduced-motion/no-IO, `.js-reveal` on `<html>`, 60ms stagger cap 4, unobserve-once). | M |
| R8 | HIGH | html-visual | `report.py:1195-1337` (~25×) | Strip inline `border/padding/text-align/border-collapse/font-size`; emit `<table class="data-table">` + `.data-table { width:100%; border-collapse:collapse; font-size:0.86rem; font-variant-numeric:tabular-nums; margin:12px 0; } .data-table th,.data-table td { border:1px solid var(--hair); padding:8px 10px; } .data-table td.num,.data-table th.num { text-align:center; }`. `#ddd → --hair`. | L |
| R9 | MED | html-visual | `report.py:1071` | `.plot-container--sep { margin-bottom:44px; padding-bottom:28px; border-bottom:1px solid var(--hair); }`; emit `<div class="plot-container plot-container--sep">`. `#e0e0e0→--hair`; 50/30→44/28. | S |
| R10 | LOW | html-visual | `report.py:1232,1234,1236,1251` | Set `row_bg='var(--warn-bg)'` / `'var(--ok-bg)'` / `'var(--surface)'` (was `#fff3e0/#e8f5e9/#fff`). | S |
| R11 | MED | html-motion | `report.py:152` | Move row hover into `@media (hover:hover) and (pointer:fine)`: `tbody tr { transition:background-color var(--dur-hover) ease; }`; add `.stat-card:hover { transform:translateY(-2px); box-shadow:0 6px 16px rgba(26,26,26,0.08); }` + `.plot-container img:hover { box-shadow:0 4px 14px rgba(26,26,26,0.10); }` (both with `--dur-hover var(--ease-out)` transitions). Remove the standalone line-152 rule. | M |
| R12 | LOW | html-visual | `report.py:86,128,149` | Unify uppercase-caption tracking to `letter-spacing: 0.06em` (was 0.08/0.05/0.04em); reference `--track-caps`. | S |
| R13 | LOW | html-visual | `report.py:95` | `.section h2 { letter-spacing: -0.008em; }` (matches H1's negative tracking). | S |
| R14 | LOW | html-visual | `report.py:63` | Add `-moz-osx-font-smoothing: grayscale;` after `-webkit-font-smoothing`. | S |
| R15 | LOW | html-visual | `report.py:130` | Add a code comment documenting that `.value` semantic colors (`--normal` 3.24:1, `--rod` 3.90:1) pass only via the large-text (1.9rem/700) exemption; for any small-text semantic use, gate on `--rod-text #C4453B` (4.6:1) / `--normal-text #17805F` (4.5:1). | S |

---

## 6. Cross-cut accessibility fixes

WCAG ratios verified; corrected hexes exact.

| # | Sev | Track | Location | Change (exact) | Effort |
|---|---|---|---|---|---|
| X1 | HIGH | a11y | `ui_common.py:193,207` + `QPushButton`/`QCheckBox`/`QRadioButton`; labeling `:490-515,613` | **2px coral focus ring, padding-compensated (no reflow).** Inputs: `QLineEdit/QSpinBox/QDoubleSpinBox:focus { border:2px solid #DA4E42; padding:7px 9px; }` (base 8px 10px). Combo: `QComboBox:focus { border:2px solid #DA4E42; padding:7px 27px 7px 9px; }` (base 8px 28px 8px 10px). Buttons: `QPushButton:focus { border:2px solid #DA4E42; padding:7px 15px; }` (base 1px/8px 16px). Checkbox/radio: `QCheckBox:focus::indicator, QRadioButton:focus::indicator { border:2px solid #DA4E42; }`. Mirror all into labeling. 2px `#DA4E42` = 4.67:1 on `#101010` (was 1px, 3.52:1). | M |
| X2 | HIGH | a11y | `ui_common.py:50` | `TEXT_MUTED = "#8A8A8A"` (was `#5A5A5A` = 2.72:1 on `#121212`, FAIL). `#8A8A8A` = 5.43:1 on `#171717`, 5.73:1 on `#0A0A0A`. **Conflict resolved:** auditor-1 proposed `#808080` (4.74:1); cross-cut proposed `#8A8A8A` — **choose `#8A8A8A`** for the wider AA margin. | S |
| X3 | HIGH | a11y | `ui_common.py:83,180` | Disabled text `color: #6E6E6E` (was `#404040` = 1.61:1 on `#1E1E1E` → 3.27:1). Both `button_style()` and app `QPushButton:disabled`. Add `TEXT_DISABLED` token. | S |
| X4 | HIGH | a11y | `main_gui.py:2067` | Subtitle `color: {Theme.TEXT_SECONDARY}` `#9A9A9A` (6.66:1) — was `#666` (3.45:1 on `#0A0A0A`, FAIL). | S |
| X5 | HIGH | cross-cut | `ui_common.py:31-32` | `NORMAL="#1F9E77"`, `NORMAL_DARK="#177A5C"` (was `#4CAF50/#388E3C`). Teal: 5.54:1 on `#171717`; color-blind-safe hue separation from ROD; ties `--ph-mid`. | S |
| X6 | MED | cross-cut | `chat_widget.py:150,326`; `main_gui.py:1658,1707,1729,2141` | Replace six hand-typed grays (`gray`,`#888`,`#B0B0B0`,`#666`) with tokens: secondary → `Theme.TEXT_SECONDARY #9A9A9A`; dimmer arg spans → `Theme.TEXT_MUTED #8A8A8A`. In report, standardize inline `#666` → `var(--muted)`. | M |
| X7 | MED | a11y | `report.py:196` | Add `:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }` + `@media (prefers-contrast:more) { :root{--hair:#9A9A97; --muted:#3A3A3A;} .section,.stat-card,.plot-container img{border-color:var(--hair);} tbody tr:hover{background:var(--accent-soft); outline:1px solid var(--accent);} }`. `--accent #2F6B9E` on paper = 5.6:1. | S |
| X8 | MED | cross-cut | `config.py:101` + `report.py:196` | Reduced-motion mechanism on BOTH surfaces. Qt: `"ui":{"reduced_motion":"auto"}` (read by `ui_motion.REDUCED`). Report: covered by §3 `@media (prefers-reduced-motion: reduce)` block (keeps opacity/color, drops movement — never zero-state). | M |

---

## 7. Implementation order

Strict dependency order. **Nothing visual/motion ships before its token layer and the motion helper exist.**

**Phase 0 — Foundations (do first, blocks everything).**
1. **Qt tokens** — X2, X3, X5, Q1 (BG rename+revalue), plus the radius/spacing/weight/font-size/focus token constants (§2a). One commit; it is the substrate for every Qt finding.
2. **`scat/ui_motion.py`** (Q18) + `config.py` `ui.reduced_motion` (X8). No call sites yet — just the module + REDUCED gate.
3. **Report tokens** — add `--fill/--ok-*/--track-caps` + motion tokens (R7 `:root` part), and the reduced-motion + print guardrail blocks (§3, R3, X8). Land guardrails before any reveal so motion is provably invisible on paper.

**Phase 1 — Qt visual/component (QSS-only, no motion).** Q7, Q2, Q3, Q4, Q5, Q6 → then components Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16 → then X1 (focus ring), X4, X6. Finish with **Q17** (shared-stylesheet refactor) so it absorbs the now-final metrics.

**Phase 2 — Qt motion (each is a 2-line call into `ui_motion.py`).** Q19 (progress) → Q20 (run-button press) → Q21 (dock) → Q22 (tabs) → Q23 (hover elevation) → Q24 (status fade). Apply Q25 (do-not-animate enforcement) alongside — verify Enter-to-send, mode swap, and label shortcuts stay instant.

**Phase 3 — Report visual.** R1, R2 (brand hex remaps) → R4, R5, R6, R9, R10 (token remaps + classes) → R8 (`.data-table` — largest) → R11 (hover, gated) → R12, R13, R14, R15, X7 (focus-visible/prefers-contrast). R7 reveal CSS + IntersectionObserver script last, on top of the already-landed guardrails.

### Verify (must pass before merge)

- **Qt before/after screenshots** at each surface: (a) Analysis tab, Results tab, Setup tab — confirm elevation step (`#171717` cards lift off `#0A0A0A`), weight ladder (only titles/section headers bold), 6/8px radii consistent app vs labeling. (b) Keyboard-Tab through buttons/inputs/combos/checkboxes — 2px coral ring visible, **zero layout jitter**. (c) Combobox shows the caret; hover a color-coded labeling button — it keeps its color (Q9). (d) Run button press = shadow depress; hover = lift. (e) Dock toggle fades; tab switch crossfades; progress eases, not hops.
- **Reduced-motion:** set `SCAT_REDUCED_MOTION=1` (or `ui.reduced_motion:"on"`) → every animation lands on its end value with no movement; app still fully functional.
- **Report in browser:** confirm no Material indigo/green remains (grep `report.py` for `#3949ab|#e8eaf6|#4caf50|#2e7d32|#e8f5e9|#f5f5f5|#666|#ddd|#e0e0e0|#fff3e0` → 0 hits outside comments). Scroll reveal fires once per section; disable JS → everything visible.
- **Print preview / weasyprint PDF:** no section, stat card, figure, or `.plot-container` splits across a page; captions stay with figures; reveal targets render at full opacity; row-hover tint absent.
- **Contrast re-check** (WCAG): `TEXT_MUTED #8A8A8A` ≥4.5:1, disabled `#6E6E6E` ≥3:1, subtitle `#9A9A9A` ≥4.5:1, focus ring 2px `#DA4E42` ≥3:1 non-text, report `--accent` focus 5.6:1.
- **Tests green:** run the existing suite; confirm no regression in `report.py` HTML generation (snapshot/string tests) and GUI smoke tests. Semantic-color change (`NORMAL`) must propagate identically to Qt swatches and report stat cards.
---

## 8. Post-review adjustments (codex second opinion — incorporated)

Codex (gpt-5.5, xhigh) reviewed this spec and flagged the Qt motion/effect model. Verified and folded in:

- **One-effect-per-widget (codex #1,#3,#4,#5):** `scat/ui_motion.py` centralizes all motion. A button's press+hover share ONE `_ButtonMotion` drop-shadow (never two attachers). Effect-property anims animate `b'blurRadius'` + scalar `b'yOffset'` (never `b'offset'`). Controllers/anims are parented (GC-safe). Fade helpers **refuse to clobber a foreign effect** (`_foreign_effect` guard) so a fade never deletes a shadow. So Q20+Q23 on the run button = ONE `attach_button_motion(run_btn, primary=True)`.
- **Interruptible progress (codex #2):** `animate_value()` stops+retargets from the current value and clears its ref on `finished` (no dangling C++ pointer). Q19 calls `animate_value(self.progress, current)`.
- **Group-box shadow (codex #6):** spiked — blur 18 / y 3 / α120 with ≥ container margins reads as clean elevation, no clipping. Applied to top-level group boxes only.
- **QSS caret + checkbox glyph (codex #7,#8):** spiked — the CSS border-triangle caret renders as a gray rectangle in Qt, and `data:` URIs don't render. **Both ship as real resource SVGs** (`scat/resources/icons/caret-down.svg`, `caret-down-hover.svg`, `check.svg`), referenced via `qss_icon()`.
- **`auto` reduced-motion (codex #14):** honest semantics — `motion_reduced()` probes the OS only on native Windows (`SPI_GETCLIENTAREAANIMATION`); elsewhere (incl. WSLg) `auto` = full motion. Users force with `ui.reduced_motion:"on"` or `SCAT_REDUCED_MOTION=1`.
- **Q17 shared-stylesheet refactor: DEFERRED** (codex #12). High regression risk, zero user-visible benefit. The visual/motion work does not need it.
- **Report reveal (codex #9,#10):** add `.js-reveal` to `<html>` only *after* the IntersectionObserver is constructed and observing (try/catch removes it on failure, plus a safety timeout), and **drop `will-change`** (avoid pinning a layer per section).
- **Contrast (codex #11):** recomputed every pair against the ACTUAL backgrounds — all pass their WCAG target. The spec's ratio *claims* were slightly overstated (e.g. `#8A8A8A` on `#171717` is 5.19:1 not 5.43; `#17805F` on the real `--paper #FAFAF9` is 4.69:1, not the 4.49 codex got from a wrong paper hex). No value changes.

Phase 0 (tokens + `ui_motion.py` + `config.ui.reduced_motion` + report tokens/guardrails) is implemented and the test suite is green.

### Implementation-diff review (codex, second pass)

Codex (gpt-5.5, xhigh) reviewed the actual diff — verdict: **"No blocking runtime regressions found."** It independently confirmed: the one-QGraphicsEffect-per-widget rule holds at every call site (run button press+hover share ONE shadow; tab/dock fades use opacity on effect-free widgets; group-box elevation opt-in, default off); animations are parented; `_ButtonMotion` is owned; `animate_value()` has no crash path; focus padding keeps box size stable (no reflow); no QSS the parser rejects; the report reveal can't leave content stuck hidden (class-after-setup + catch + 3s safety net + no-JS/reduced-motion visible). One LOW finding fixed: quoted the `url("...")` icon paths so caret/check survive install paths containing spaces. Motion interruption paths additionally stress-tested (rapid progress retarget, button events, fade churn, foreign-effect guard) — no crashes/leaks. Full suite: 211 passed.
