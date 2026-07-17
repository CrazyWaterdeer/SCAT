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
from PySide6.QtWidgets import QApplication, QWidget
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def _tab(tmp_path, synth_dir, **kw):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False, **kw)
    d = _results_dict_from_output(Path(res.output_dir))
    tab = ResultsTab(); tab.load_results(d)
    return tab, d


def test_composition_strip_replaces_kpi_tiles(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    assert not hasattr(tab, "tiles_layout")                 # KPI tile band gone
    # no residual KPI tile widgets
    assert not [w for w in tab.findChildren(QWidget) if w.objectName() == "kpiTile"]
    fs = d["film_summary"]
    n_normal, n_rod = int(fs['n_normal'].sum()), int(fs['n_rod'].sum())
    txt = tab.composition_line.text()
    assert f"{n_normal + n_rod}" in txt          # explicit Deposits count preserved
    assert str(n_normal) in txt and str(n_rod) in txt
    for token in ("Deposits", "Normal", "ROD", "Artifact"):
        assert token in txt


def test_total_iod_omitted_when_column_absent(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    d2 = dict(d); d2["film_summary"] = d["film_summary"].drop(columns=["total_iod"], errors="ignore")
    tab.load_results(d2)                          # must not raise, must not show "Total IOD 0"
    assert "Total IOD 0" not in tab.composition_line.text()
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
sep = " &nbsp;·&nbsp; "
parts = [
    f"Deposits <b>{total_normal + total_rod:.0f}</b>",
    f"Normal <b style='color:{Theme.NORMAL}'>{total_normal:.0f}</b>",
    f"ROD <b style='color:{Theme.ROD}'>{total_rod:.0f}</b>",
    f"ROD fraction <b>{mean_rod_frac*100:.1f}%</b>",
    f"Artifact <span style='color:{Theme.TEXT_MUTED}'>{total_artifact:.0f}</span>",
]
if 'total_iod' in film_summary.columns:              # omit when absent — never show a fake "0"
    parts.append(f"Total IOD <b>{film_summary['total_iod'].sum():.0f}</b>")
self.composition_line.setText(sep.join(parts))
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
from scat.ui_common import CollapsibleSection


def test_report_grade_content_is_exiled_to_one_pointer(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    # The stats area holds exactly ONE widget (the pointer) — no gallery, tables, or sections.
    assert tab.stats_layout.count() == 1
    assert not [w for w in tab.findChildren(QWidget) if w.objectName() == "vizCell"]
    assert not tab.findChildren(CollapsibleSection)
    from PySide6.QtWidgets import QLabel
    labels = [w.text() for w in tab.findChildren(QLabel)]
    assert not any("VISUALIZATIONS" in t or "DESCRIPTIVE STATISTICS" in t or "GROUP COMPARISONS" in t
                   for t in labels)


def test_pointer_is_report_state_aware(synth_dir, tmp_path):
    # analyze with annotate=False does NOT write report.html -> pointer must not claim it exists.
    tab, d = _tab(tmp_path, synth_dir)
    ptxt = tab.stats_layout.itemAt(0).widget().text().lower()
    assert "report" in ptxt
    assert "generate" in ptxt          # no report yet -> "Generate a report…", not "in the report"
    # now pretend a report exists and re-load -> pointer flips to the "in the report" wording
    (Path(d["output_dir"]) / "report.html").write_text("<html></html>")
    tab.load_results(_results_dict_from_output(Path(d["output_dir"])))
    assert "in the report" in tab.stats_layout.itemAt(0).widget().text().lower()
    assert tab.stats_layout.count() == 1     # reload still leaves exactly one pointer


def test_reload_updates_composition_and_clears_stats(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    tab.load_results(d)                      # second load (as _reload_results does after an edit)
    assert tab.stats_layout.count() == 1
    assert "Deposits" in tab.composition_line.text()
```

- [ ] **Step 3: Run** → FAIL (the VISUALIZATIONS/DESCRIPTIVE labels still exist).

- [ ] **Step 4: Implement** — replace the whole body of `_load_statistics_tab` with a clear + a single
  pointer (keep the method + its signature + the `stats_layout` clear so re-loads work):

```python
def _load_statistics_tab(self, results: dict):
    """The working view is for triage; the report carries the full distributions, group comparisons
    and statistics. Show ONE quiet, report-state-aware pointer instead of duplicating (and out-dumping)
    the report. stats_layout is retained so re-loads (edit -> _reload_results -> load_results) work."""
    while self.stats_layout.count():
        item = self.stats_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
    out = results.get("output_dir", "")
    report_exists = bool(out) and (Path(out) / "report.html").exists()
    text = ("Full distributions, group comparisons and statistics are in the report."
            if report_exists else
            "Generate a report to see the full distributions, group comparisons and statistics.")
    pointer = QLabel(text)
    pointer.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_SM}px; padding-top: 8px;")
    pointer.setWordWrap(True)
    self.stats_layout.addWidget(pointer)
```

(`Path` is already imported in `main_gui.py`.)

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
