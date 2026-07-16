# SCAT redesign — Plan 3: result-window recomposition (Phase 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the result window a triage surface, not a dashboard: collapse the six KPI tiles into
one thin **composition strip**, and **exile the report-grade charts/stats** (Visualizations gallery,
descriptive stats, group comparisons, spatial) from the working view — they live in the report — with
a quiet pointer to the report in their place.

**Architecture:** Two focused edits to `ResultsTab` in `scat/main_gui.py`. The verdict header (hero +
trust line + state-driven actions) and the Review(N) worklist already exist (Plans 1–2). This plan
removes the tiles band + the `_load_statistics_tab` content. No new modules. The "Review next"
navigation action stays out of scope (it depends on the deferred drill-in worklist, Plan 2c); the
primary action remains the existing state-driven **Open report**.

**Tech Stack:** Python 3.10+, PySide6, pytest. No new deps.

**Spec:** `docs/.../2026-07-17-scat-results-report-redesign-design.md` (§4). **Depends on Plans 1–2**
(hero reads the primary metric; Review column + trust line present). **No tests reference**
`tiles_layout` / `stats_host` / `_load_statistics_tab` / `_kpi_tile` / the viz helpers (verified), so
this restructures freely.

---

## File structure

- **Modify `scat/main_gui.py`** `ResultsTab`:
  - `_setup_ui`: replace `tiles_host`/`tiles_layout` with a single `composition_line` QLabel; replace
    the `stats_host` block's role (keep the widget as the pointer host, or add a `report_pointer` label).
  - `load_results`: fill the composition line; stop building KPI tiles.
  - `_load_statistics_tab`: gut its body to a single report-pointer line (the charts/stats are the
    report's job). The now-unused helpers (`_kpi_tile`, `_viz_cell`, `_viz_grid`, `_format_viz_name`,
    `_generate_descriptive_stats`, `_generate_comparison_stats`, `_generate_spatial_stats`) may be left
    as dead code or removed in a follow-up cleanup — do NOT remove them in this plan (keeps the diff
    small and avoids touching their imports/tests).
- **Test:** `tests/test_results_recomposition.py`.

---

## Task 1: Composition strip replaces the six KPI tiles

**Files:** Modify `scat/main_gui.py` (`_setup_ui` ~1823-1827 + `load_results` ~1968-1985);
Test `tests/test_results_recomposition.py`

- [ ] **Step 1: Read** — `sed -n '1820,1830p' scat/main_gui.py` (the `tiles_host` block) and
  `sed -n '1966,1986p' scat/main_gui.py` (the KPI fill in `load_results`, incl. the local vars
  `total_normal/total_rod/total_artifact/mean_rod_frac` and the `total_iod` sum).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_results_recomposition.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def _tab(tmp_path, synth_dir):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    return tab


def test_composition_strip_replaces_kpi_tiles(synth_dir, tmp_path):
    tab = _tab(tmp_path, synth_dir)
    assert not hasattr(tab, "tiles_layout")           # KPI tile band gone
    txt = tab.composition_line.text()
    for token in ("Normal", "ROD", "Artifact"):        # one semantic composition line
        assert token in txt
```

- [ ] **Step 3: Run** — `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest tests/test_results_recomposition.py -q` → FAIL (`tiles_layout` still exists / no `composition_line`).

- [ ] **Step 4: Implement**

In `_setup_ui`, replace the `tiles_host`/`tiles_layout` block with:

```python
# Composition strip (one thin semantic line — replaces the six KPI tiles).
self.composition_line = QLabel("")
self.composition_line.setStyleSheet(
    f"color: {Theme.TEXT_SECONDARY}; font-size: {Theme.FS_BODY}px; padding-top: 4px;")
self.composition_line.setTextFormat(Qt.RichText)
hv.addWidget(self.composition_line)
```

In `load_results`, delete the tiles loop (the `while self.tiles_layout...` clear + the six
`self.tiles_layout.addWidget(self._kpi_tile(...))` lines) and set the composition line instead. Reuse
the already-computed `total_normal/total_rod/total_artifact`; compute the rest inline:

```python
_iod = film_summary['total_iod'].sum() if 'total_iod' in film_summary.columns else 0
self.composition_line.setText(
    f"Normal <b style='color:{Theme.NORMAL}'>{total_normal:.0f}</b> &nbsp;·&nbsp; "
    f"ROD <b style='color:{Theme.ROD}'>{total_rod:.0f}</b> &nbsp;·&nbsp; "
    f"ROD fraction <b>{mean_rod_frac*100:.1f}%</b> &nbsp;·&nbsp; "
    f"Artifact <span style='color:{Theme.TEXT_MUTED}'>{total_artifact:.0f}</span> &nbsp;·&nbsp; "
    f"Total IOD <b>{_iod:.0f}</b>")
```

(Leave `_kpi_tile` defined but unused — a follow-up cleanup removes it.)

- [ ] **Step 5: Run + full suite** — PASS; `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest -q` all pass.
- [ ] **Step 6: Verify** — render ResultsTab; the six tiles are replaced by one composition line under the hero.
- [ ] **Step 7: Commit**

```bash
git add scat/main_gui.py tests/test_results_recomposition.py
git commit -m "feat(gui): composition strip replaces the six KPI tiles (triage surface, not dashboard)"
```

---

## Task 2: Exile report-grade charts/stats → a report pointer

**Files:** Modify `scat/main_gui.py` (`_load_statistics_tab` ~2022); Test `tests/test_results_recomposition.py`

The working view stops carrying the report's job (the ~20-chart gallery + descriptive/group/spatial
stats). `_load_statistics_tab` becomes a single quiet pointer to the report. QC stays via the drill-in
annotated overlay (unchanged).

- [ ] **Step 1: Read** — `sed -n '2022,2075p' scat/main_gui.py` (the full `_load_statistics_tab` body).

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_results_recomposition.py
def test_report_grade_content_is_exiled_to_a_pointer(synth_dir, tmp_path):
    tab = _tab(tmp_path, synth_dir)
    # No in-app Visualizations / Descriptive-stats / Group-comparison section labels remain.
    from PySide6.QtWidgets import QLabel
    labels = [w.text() for w in tab.findChildren(QLabel)]
    assert not any("VISUALIZATIONS" in t or "DESCRIPTIVE STATISTICS" in t or "GROUP COMPARISONS" in t
                   for t in labels)
    # A pointer to the report is present instead.
    assert any("report" in t.lower() and ("distribution" in t.lower() or "statistic" in t.lower())
               for t in labels)
```

- [ ] **Step 3: Run** → FAIL (the VISUALIZATIONS/DESCRIPTIVE labels still exist).

- [ ] **Step 4: Implement** — replace the whole body of `_load_statistics_tab` with a clear + a single
  pointer (keep the method + its signature + the `stats_layout` clear so re-loads work):

```python
def _load_statistics_tab(self, results: dict):
    """The working view is for triage; the report carries the full distributions, group comparisons
    and statistics. Show a quiet pointer to it instead of duplicating (and out-dumping) the report."""
    while self.stats_layout.count():
        item = self.stats_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
    pointer = QLabel("Full distributions, group comparisons and statistics are in the report.")
    pointer.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_SM}px; padding-top: 8px;")
    pointer.setWordWrap(True)
    self.stats_layout.addWidget(pointer)
```

- [ ] **Step 5: Run + full suite** — PASS; `pytest -q` all pass. (No tests referenced the removed
  sections, verified — but re-run the full suite to be sure.)
- [ ] **Step 6: Verify** — render; the working view ends at the per-image table + a one-line report
  pointer; no chart gallery / stats tables. The page is markedly shorter.
- [ ] **Step 7: Commit**

```bash
git add scat/main_gui.py tests/test_results_recomposition.py
git commit -m "feat(gui): exile report-grade charts/stats from the working view (pointer to the report)"
```

---

## Self-review checklist

- **Spec coverage (§4):** composition strip replaces KPI tiles (T1); report-grade charts/stats exiled
  to a pointer (T2). The verdict header (hero + trust line) and Review(N) worklist already shipped
  (Plans 1–2). Out of scope by design: the "Review next" nav action (needs the deferred drill-in
  worklist, Plan 2c); the threshold slider + All/Needs-review toggle (Plan 2b/3-follow-up).
- **Low risk:** no tests reference the removed widgets/methods (verified via grep). Dead helpers
  (`_kpi_tile`, viz/stat generators) are LEFT in place to keep the diff small; a follow-up cleanup can
  delete them.
- **Placeholder scan:** none. **Type consistency:** `composition_line` (QLabel, RichText) and the gutted
  `_load_statistics_tab` are self-consistent; `stats_layout`/`stats_host` retained so re-loads (edit →
  `_reload_results` → `load_results` → `_load_statistics_tab`) still work.
- **QC preserved:** the drill-in annotated overlay is untouched, so image-quality problems remain
  visible (codex #10).
