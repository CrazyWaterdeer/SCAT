# SCAT redesign — Plan 2: confidence in the result window (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the classifier's per-deposit confidence as an honest triage signal in the result
window — a per-image **Review(N)** column, a factual **trust line** + **sensitivity band** at the
fixed threshold, and a **worst-confidence-first** drill-in — without overclaiming (per spec §2.1).

**Architecture:** Two new pure functions in `scat/metrics.py` (finish `metric_sensitivity_range` with a
worst-case flip; add per-image `flagged_by_image`) + a small pure `scat/confidence.py` (run trust
facts). The result window's existing per-image table gains a Review column, the hero gains a trust
line + band, and the drill-in editor (`LabelingWindow`) sorts worst-confidence-first with a filter.
Everything reads the **fixed** `confidence_threshold` from the results dict (Plan 1); no threshold
slider here (that + the All/Needs-review toggle are recomposition, Plan 3). This plan ADDS the signal
to the current layout; it does not restructure it.

**Tech Stack:** Python 3.10+, pandas, PySide6, pytest. No new deps.

**Spec:** `docs/.../2026-07-17-scat-results-report-redesign-design.md` (§2.1, §3.2, §4). **Depends on
Plan 1** (metrics module, results dict carrying `deposit_data` + `confidence_threshold`).

**Honesty (spec §2.1, do not violate):** confidence = uncalibrated RF label sensitivity; the trust
line states **"N of M above the classification threshold"** (never "high-confidence"/"reviewed"); the
band is labeled **"sensitivity to low-confidence labels only"**, never a CI; computed at the FIXED
threshold so nothing is gameable.

---

## File structure

- **Modify `scat/metrics.py`** — add `flagged_by_image()`; finish `metric_sensitivity_range()` (worst
  case, no 2nd-best label needed).
- **Create `scat/confidence.py`** — `run_trust(deposits_df, threshold)` → factual trust dict.
- **Create `tests/test_confidence.py`**; extend `tests/test_metrics.py`.
- **Modify `scat/main_gui.py`** — `ResultsTab`: Review column in the per-image table; trust line + band
  under the hero.
- **Modify `scat/labeling_gui.py`** — `LabelingWindow`: default worst-confidence-first sort + a
  "show only low-confidence" filter.
- **Tests:** `tests/test_results_confidence.py`, `tests/test_labeling_confidence.py`.

---

## Task 1: `flagged_by_image` + finish `metric_sensitivity_range` (worst-case)

**Files:** Modify `scat/metrics.py`; extend `tests/test_metrics.py`

The v1 flip rule needs no per-deposit second-best label (which `all_deposits.csv` lacks): a flagged
deposit "could be wrong", so we take the **worst-case** bound — each flagged ROD could really be
Normal (or Artifact); each flagged Normal could really be ROD (or Artifact). For a given metric this
gives the widest honest low/high the flagged labels could produce.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
import pandas as pd


def _deposits():
    # image a: 3 deposits — a low-conf ROD (0.55), a hi-conf Normal (0.9), a low-conf Normal (0.5)
    return pd.DataFrame({
        "filename": ["a", "a", "a", "b"],
        "label":    ["rod", "normal", "normal", "rod"],
        "confidence": [0.55, 0.90, 0.50, 0.95],
    })


def test_flagged_by_image_counts_below_threshold_by_label():
    f = metrics.flagged_by_image(_deposits(), threshold=0.60)
    # image a: 1 flagged rod (0.55), 1 flagged normal (0.50); image b: none
    assert f["a"]["flagged_rod"] == 1 and f["a"]["flagged_normal"] == 1 and f["a"]["flagged"] == 2
    assert f["b"]["flagged"] == 0


def test_rod_fraction_worst_case_band_widens_both_ways():
    # image a currently: normal=2, rod=1 -> ROD fraction = 1/3 = 33.3%.
    # flagged: 1 rod, 1 normal. Worst low: the flagged rod is really normal -> 0/3 = 0%.
    # Worst high: the flagged normal is really rod -> 2/3 = 66.7%.
    lo, hi = metrics.metric_sensitivity_range(
        n_normal=2, n_rod=1, flagged_normal=1, flagged_rod=1, key="rod_fraction")
    assert round(lo, 1) == 0.0 and round(hi, 1) == 66.7


def test_deposit_count_band_is_flat_when_alternative_is_the_other_deposit_class():
    # v1 assumption: flagged Normal<->ROD flips keep the deposit COUNT; band is a point.
    lo, hi = metrics.metric_sensitivity_range(
        n_normal=2, n_rod=1, flagged_normal=1, flagged_rod=1, key="total_deposits")
    assert lo == hi == 3
```

- [ ] **Step 2: Run** → FAIL (`AttributeError: ... 'flagged_by_image'`).

- [ ] **Step 3: Implement (append to `scat/metrics.py`)**

```python
def flagged_by_image(deposits_df: pd.DataFrame, threshold: float) -> dict[str, dict]:
    """Per-image counts of low-confidence deposits by predicted label. Keys: filename -> dict with
    flagged_normal / flagged_rod / flagged / total. Empty dict if the frame lacks the columns."""
    out: dict[str, dict] = {}
    if deposits_df is None or not {"filename", "label", "confidence"} <= set(deposits_df.columns):
        return out
    low = deposits_df["confidence"] < threshold
    for fn, grp in deposits_df.groupby("filename"):
        g_low = grp[grp["confidence"] < threshold]
        out[str(fn)] = {
            "flagged_normal": int((g_low["label"] == "normal").sum()),
            "flagged_rod": int((g_low["label"] == "rod").sum()),
            "flagged": int(len(g_low)),
            "total": int(len(grp)),
        }
    return out


def metric_sensitivity_range(*, n_normal: int, n_rod: int,
                             flagged_normal: int, flagged_rod: int, key: str) -> tuple[float, float]:
    """Worst-case label-sensitivity bound (spec §2.1) — NOT a CI. Each flagged deposit could be wrong;
    return the widest low/high the primary metric could take. v1 covers count/fraction (which depend
    only on Normal/ROD tallies)."""
    key = resolve_metric(key)
    if key == "rod_fraction":
        def frac(nn, nr):
            d = nn + nr
            return 0.0 if d == 0 else 100.0 * nr / d
        # low: flagged rod -> normal; high: flagged normal -> rod
        lo = frac(n_normal + flagged_rod, n_rod - flagged_rod)
        hi = frac(n_normal - flagged_normal, n_rod + flagged_normal)
        return (min(lo, hi), max(lo, hi))
    if key == "total_deposits":
        d = float(n_normal + n_rod)   # Normal<->ROD flips don't change the count (v1 assumption)
        return (d, d)
    return (float("nan"), float("nan"))  # mean/area/hue/iod/circularity: not modeled in v1
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): per-image flagged counts + worst-case sensitivity band"
```

---

## Task 2: `scat/confidence.py` — factual run trust

**Files:** Create `scat/confidence.py`; Create `tests/test_confidence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_confidence.py
import pandas as pd
from scat import confidence


def _dep():
    return pd.DataFrame({"filename": ["a", "a", "b"], "label": ["rod", "normal", "normal"],
                         "confidence": [0.55, 0.9, 0.95]})


def test_run_trust_is_factual():
    t = confidence.run_trust(_dep(), threshold=0.60)
    assert t["total"] == 3 and t["above"] == 2
    assert round(t["pct_above"], 1) == 66.7
    assert t["flagged"] == 1 and t["state"] == "review"     # some flagged -> review
    # wording is a factual count, never "high-confidence"/"reviewed"
    assert t["line"] == "2 of 3 deposits above the classification threshold (0.60)"


def test_run_trust_all_clear():
    t = confidence.run_trust(pd.DataFrame({"filename": ["a"], "label": ["rod"], "confidence": [0.9]}), 0.60)
    assert t["state"] == "clear" and t["flagged"] == 0


def test_run_trust_handles_missing_data():
    t = confidence.run_trust(None, 0.60)
    assert t["state"] == "unknown" and t["total"] == 0
```

- [ ] **Step 2: Run** → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# scat/confidence.py
"""Factual run-trust facts from per-deposit classifier confidence (spec §2.1). Pure; no reliability
claims — confidence is uncalibrated label sensitivity, reported only as counts above a fixed
threshold. No 'reviewed'/'high-confidence' wording."""
from __future__ import annotations

import pandas as pd


def run_trust(deposits_df: pd.DataFrame | None, threshold: float) -> dict:
    if deposits_df is None or "confidence" not in getattr(deposits_df, "columns", []):
        return {"total": 0, "above": 0, "pct_above": 0.0, "flagged": 0,
                "state": "unknown", "line": "confidence unavailable"}
    total = int(len(deposits_df))
    above = int((deposits_df["confidence"] >= threshold).sum())
    flagged = total - above
    pct = (100.0 * above / total) if total else 0.0
    if flagged == 0:
        state = "clear"
    elif flagged <= max(1, total // 20):     # <=5% flagged
        state = "review"
    else:
        state = "many"
    line = (f"{above} of {total} deposits above the classification threshold ({threshold:.2f})"
            if total else "no deposits")
    return {"total": total, "above": above, "pct_above": pct, "flagged": flagged,
            "state": state, "line": line}
```

(Note the Task-2 test expects `state == "review"` for 1/3 flagged; adjust the threshold ratio if the
tests disagree — the test is the contract. 1 flagged of 3 total: `total//20 == 0`, `max(1,0)==1`,
`flagged(1) <= 1` → "review". ✓)

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add scat/confidence.py tests/test_confidence.py
git commit -m "feat(confidence): factual run-trust facts (counts above a fixed threshold, no reliability claim)"
```

---

## Task 3: Review(N) column in the per-image table

**Files:** Modify `scat/main_gui.py` (`ResultsTab`: `summary_table` setup ~1846-1847 + `load_results`
table fill ~1948-1959); Test `tests/test_results_confidence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_results_confidence.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_table_has_review_column(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    headers = [tab.summary_table.horizontalHeaderItem(c).text()
               for c in range(tab.summary_table.columnCount())]
    assert "Review" in headers
    assert tab.summary_table.columnCount() == 7
```

- [ ] **Step 2: Run** → FAIL (columnCount 6, no "Review").

- [ ] **Step 3: Implement**

In `_setup_ui`, bump the table to 7 columns and add the header:

```python
self.summary_table.setColumnCount(7)
self.summary_table.setHorizontalHeaderLabels(
    ["Filename", "Review", "Normal", "ROD", "Artifact", "ROD %", "Total IOD"])
```

In `load_results`, compute per-image flagged counts once and fill column 1 (shifting the existing
Normal/ROD/... to columns 2-6). Use the results-dict `deposit_data` + `confidence_threshold`:

```python
from scat import metrics as _metrics
# near the top of load_results:
flagged = _metrics.flagged_by_image(results.get("deposit_data"), results.get("confidence_threshold", 0.60))
# in the per-row loop (columns shift by +1 after Filename):
self.summary_table.setItem(i, 0, QTableWidgetItem(str(row['filename'])))
info = flagged.get(str(row['filename']), {"flagged": 0, "total": 0})
rev = NumericTableWidgetItem(info["flagged"], "{:.0f}")
rev.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
if info["flagged"]:
    rev.setForeground(QColor(Theme.PRIMARY_LIGHT))
    frac = (100.0 * info["flagged"] / info["total"]) if info["total"] else 0.0
    rev.setToolTip(f"{info['flagged']} of {info['total']} deposits below the confidence threshold "
                   f"({frac:.0f}%)")
else:
    rev.setText("—")
self.summary_table.setItem(i, 1, rev)
self._set_num(i, 2, row['n_normal'], "{:.0f}", Theme.NORMAL)
self._set_num(i, 3, row['n_rod'], "{:.0f}", Theme.ROD)
self._set_num(i, 4, row['n_artifact'], "{:.0f}", Theme.TEXT_MUTED)
self._set_num(i, 5, row['rod_fraction'] * 100, "{:.1f}%")
self._set_num(i, 6, row.get('total_iod', 0), "{:.0f}")
```

Default sort stays filename (no `sortItems` call added); the column is sortable via the existing
`setSortingEnabled(True)`.

- [ ] **Step 4: Run + full suite** → PASS; `pytest -q` all pass.
- [ ] **Step 5: Verify** — render the ResultsTab; confirm a "Review" column with flagged counts (coral) / "—".
- [ ] **Step 6: Commit**

```bash
git add scat/main_gui.py tests/test_results_confidence.py
git commit -m "feat(gui): Review(N) column — per-image low-confidence count (sortable, default filename)"
```

---

## Task 4: Trust line + sensitivity band under the hero

**Files:** Modify `scat/main_gui.py` (`ResultsTab`: hero block in `_setup_ui` + `load_results`);
Test `tests/test_results_confidence.py`

Adds two muted lines beneath the hero value: a **trust line** (traffic-light dot + the factual
`run_trust` line) and a **sensitivity band** for the primary metric, both at the fixed threshold. No
threshold control (Plan 3).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_results_confidence.py
def test_hero_shows_trust_line_and_band(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="rod_fraction", annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    assert "above the classification threshold" in tab.trust_line.text()
    # band present when the primary metric is fraction/count (may be empty text if 0 flagged)
    assert hasattr(tab, "band_line")
```

- [ ] **Step 2: Run** → FAIL (`AttributeError: 'ResultsTab' object has no attribute 'trust_line'`).

- [ ] **Step 3: Implement**

In `_setup_ui`, add two labels to the hero `numcol` (below `hero_sub`):

```python
self.trust_line = QLabel(""); self.trust_line.setStyleSheet(
    f"color: {Theme.TEXT_SECONDARY}; font-size: {Theme.FS_SM}px;")
self.band_line = QLabel(""); self.band_line.setStyleSheet(
    f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_XS}px;")
numcol.addWidget(self.trust_line); numcol.addWidget(self.band_line)
```

In `load_results`, after computing the hero, set them from `confidence.run_trust` + the band. Use the
fixed threshold; wording is factual (§2.1):

```python
from scat import confidence as _conf
thr = results.get("confidence_threshold", 0.60)
trust = _conf.run_trust(results.get("deposit_data"), thr)
dot = {"clear": "● ", "review": "● ", "many": "● ", "unknown": ""}[trust["state"]]
color = {"clear": Theme.NORMAL, "review": "#E0A93A", "many": Theme.ROD, "unknown": Theme.TEXT_MUTED}[trust["state"]]
self.trust_line.setStyleSheet(f"color:{color}; font-size:{Theme.FS_SM}px;")
self.trust_line.setText(dot + trust["line"])
# sensitivity band over the whole run for the primary metric (sum flagged across images)
band_txt = ""
if pm in ("rod_fraction", "total_deposits") and trust["flagged"]:
    fb = _metrics.flagged_by_image(results.get("deposit_data"), thr)
    fn = sum(v["flagged_normal"] for v in fb.values()); fr = sum(v["flagged_rod"] for v in fb.values())
    lo, hi = _metrics.metric_sensitivity_range(
        n_normal=int(film_summary['n_normal'].sum()), n_rod=int(film_summary['n_rod'].sum()),
        flagged_normal=fn, flagged_rod=fr, key=pm)
    if lo == lo and lo != hi:  # not NaN, not a point
        u = "%" if pm == "rod_fraction" else ""
        band_txt = f"{lo:.1f}{u}–{hi:.1f}{u} if the flagged labels were wrong (sensitivity to low-confidence labels only)"
self.band_line.setText(band_txt)
```

- [ ] **Step 4: Run + full suite** → PASS. **Step 5: Verify** render (trust line + band under hero).
- [ ] **Step 6: Commit**

```bash
git add scat/main_gui.py tests/test_results_confidence.py
git commit -m "feat(gui): hero trust line + sensitivity band (fixed threshold, factual wording)"
```

---

## Task 5: Drill-in — worst-confidence-first + low-confidence filter

**Files:** Modify `scat/labeling_gui.py` (`LabelingWindow`: `deposit_table` population ~1408 + a filter
checkbox); Test `tests/test_labeling_confidence.py`

- [ ] **Step 1: Read** — `sed -n '1390,1420p' scat/labeling_gui.py` to see how `deposit_table` rows are
  built and where a confidence column exists (deposits carry `.confidence`). Confirm the column index
  holding confidence (or add one if absent).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_labeling_confidence.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from scat.labeling_gui import LabelingWindow


def test_low_confidence_filter_exists(qtbot=None):
    QApplication.instance() or QApplication([])
    # Construct with no image loaded is enough to assert the control exists.
    w = LabelingWindow.__new__(LabelingWindow)  # avoid full image load in the unit test
    # The attribute is created in _setup/_build UI; assert the plan's control name is wired.
    assert hasattr(LabelingWindow, "_apply_low_conf_filter")
```

(If constructing `LabelingWindow` needs an image, adapt to the repo's existing `LabelingWindow` test in
`tests/test_labeling_gui.py` — reuse its fixture/pattern rather than inventing one.)

- [ ] **Step 3: Implement** — add a "Show only low-confidence" `QCheckBox` above `deposit_table`; on
  population, default-sort by the confidence column ascending; wire `_apply_low_conf_filter()` to hide
  rows with `confidence >= threshold` (threshold from the window's results context, default 0.60).
  Keep it minimal and follow the existing `deposit_table` idioms.

- [ ] **Step 4: Run + full suite** → PASS (incl. `test_labeling_gui`).
- [ ] **Step 5: Verify** — open the editor on an image with low-confidence deposits; confirm they sort
  to the top and the filter hides confident ones.
- [ ] **Step 6: Commit**

```bash
git add scat/labeling_gui.py tests/test_labeling_confidence.py
git commit -m "feat(labeling): drill-in sorts worst-confidence-first + a low-confidence filter"
```

---

## Self-review checklist

- **Spec coverage (§3.2/§4):** flagged counts + worst-case band (T1); factual trust (T2); Review(N)
  column, default filename sort (T3); trust line + band at fixed threshold (T4); drill-in worst-first +
  filter (T5). The All/Needs-review toggle + threshold slider + verdict-header recomposition are Plan 3;
  `run_meta` metadata capture for per_fly/area/time is a separate follow-up task.
- **Honesty:** trust line = "N of M above the classification threshold (t)"; band = "sensitivity to
  low-confidence labels only", never a CI; fixed threshold everywhere (no slider) so nothing is gameable.
- **Placeholder scan:** T5 impl is described (not full code) because it must follow the existing
  `LabelingWindow.deposit_table` idioms — the implementer reads ~1390-1420 first; the test pins the
  contract (`_apply_low_conf_filter`). All other tasks have complete code.
- **Type consistency:** `flagged_by_image` / `metric_sensitivity_range(flagged_normal, flagged_rod)` /
  `run_trust` keys (`total/above/pct_above/flagged/state/line`) used identically across T1–T4.
