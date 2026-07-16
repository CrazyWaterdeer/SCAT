# SCAT redesign — Plan 2: confidence in the result window, honest core (Phase 2a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the classifier's per-deposit confidence as an HONEST triage signal in the result
window — a per-image **Review(N)** column and a purely factual **trust line** at the fixed threshold —
with no reliability claim, no color-coded verdict, and wording that holds for both the RF and the
rule-based classifier.

**Architecture:** One tiny pure helper in `scat/metrics.py` (`flagged_by_image`) + a small pure
`scat/confidence.py` (`run_trust`). The result window's existing per-image table gains a Review column
(count of low-confidence deposits, from `all_deposits.csv`, which carries `confidence`), and the hero
gains one factual trust line. Everything reads the **fixed** `confidence_threshold` from the results
dict (Plan 1).

**Rescoped after a codex review (2026-07-17):** the **sensitivity band** (needs a per-deposit
second-best label `all_deposits.csv` lacks, and an estimand that matches the hero's per-image
normalization) and the **worst-confidence-first drill-in** (needs confidence merged into the JSON edit
path + ID-based selection in `LabelingWindow`, which today uses index-based `selectRow`) are each their
own later plan. This plan ships only what is honest and low-risk today.

**Tech Stack:** Python 3.10+, pandas, PySide6, pytest. No new deps.

**Spec:** `docs/.../2026-07-17-scat-results-report-redesign-design.md` (§2.1, §3.2). **Depends on Plan 1**
(results dict carries `deposit_data` + `confidence_threshold`).

**Honesty (spec §2.1 + codex):** the trust line states a **factual count** — "M of N deposits below the
confidence-score threshold (t)" — never "high-confidence", "reviewed", or "reliable"; the marker is
**neutral** (no green/amber/red verdict that implies reliability); wording is **"confidence score"**
(generic — the RF gives a class probability, the rule-based classifier a circularity-derived score,
and neither is a calibrated probability of correctness). Empty/absent confidence → an explicit
"unavailable" state, never a clean/green one.

---

## File structure

- **Modify `scat/metrics.py`** — add `flagged_by_image(deposits_df, threshold)`.
- **Create `scat/confidence.py`** + **`tests/test_confidence.py`** — `run_trust(deposits_df, threshold)`.
- **Modify `scat/main_gui.py`** `ResultsTab` — Review column (table 6→7); one trust line under the hero.
- **Tests:** extend `tests/test_metrics.py`; create `tests/test_results_confidence.py`.

---

## Task 1: `flagged_by_image` (per-image low-confidence count)

**Files:** Modify `scat/metrics.py`; extend `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
import pandas as pd


def _deps():
    return pd.DataFrame({
        "filename": ["a", "a", "a", "b"],
        "label":    ["rod", "normal", "artifact", "rod"],
        "confidence": [0.55, 0.90, 0.50, 0.95],
    })


def test_flagged_by_image_counts_all_below_threshold():
    f = metrics.flagged_by_image(_deps(), threshold=0.60)
    # image a: two below 0.60 (rod 0.55, artifact 0.50); image b: none
    assert f["a"] == {"flagged": 2, "total": 3}
    assert f["b"] == {"flagged": 0, "total": 1}


def test_flagged_by_image_threshold_is_strict_less_than():
    f = metrics.flagged_by_image(
        pd.DataFrame({"filename": ["a"], "label": ["rod"], "confidence": [0.60]}), threshold=0.60)
    assert f["a"]["flagged"] == 0          # exactly at threshold is NOT flagged


def test_flagged_by_image_missing_columns_returns_empty():
    assert metrics.flagged_by_image(None, 0.6) == {}
    assert metrics.flagged_by_image(pd.DataFrame({"filename": ["a"]}), 0.6) == {}
```

- [ ] **Step 2: Run** — `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest tests/test_metrics.py -q` → FAIL (`AttributeError: ... 'flagged_by_image'`).

- [ ] **Step 3: Implement (append to `scat/metrics.py`)**

```python
def flagged_by_image(deposits_df, threshold: float) -> dict[str, dict]:
    """Per-image count of low-confidence deposits (confidence < threshold), any label. Returns
    {filename: {"flagged": int, "total": int}}. Empty dict if the frame lacks the needed columns.
    Confidence is the classifier's score (RF class probability OR the rule-based circularity score);
    it is NOT a calibrated probability of correctness — this is only a workload/triage signal."""
    import pandas as pd
    if deposits_df is None or not {"filename", "confidence"} <= set(getattr(deposits_df, "columns", [])):
        return {}
    out: dict[str, dict] = {}
    for fn, grp in deposits_df.groupby("filename"):
        out[str(fn)] = {"flagged": int((grp["confidence"] < threshold).sum()), "total": int(len(grp))}
    return out
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): flagged_by_image — per-image low-confidence deposit count"
```

---

## Task 2: `scat/confidence.py` — factual run trust (no reliability claim)

**Files:** Create `scat/confidence.py`; Create `tests/test_confidence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_confidence.py
import pandas as pd
from scat import confidence


def test_flagged_run_is_factual_and_worklike():
    t = confidence.run_trust(
        pd.DataFrame({"filename": ["a", "a", "b"], "confidence": [0.55, 0.9, 0.95]}), threshold=0.60)
    assert t["total"] == 3 and t["below"] == 1 and t["state"] == "review"
    assert t["line"] == "1 of 3 deposits below the confidence-score threshold (0.60) — review recommended"
    # never a reliability word
    for bad in ("high-confidence", "reviewed", "reliable", "trustworthy"):
        assert bad not in t["line"].lower()


def test_all_above_threshold_is_not_green_reliability():
    t = confidence.run_trust(pd.DataFrame({"filename": ["a"], "confidence": [0.9]}), 0.60)
    assert t["state"] == "none_flagged" and t["below"] == 0
    assert t["line"] == "all 1 deposits at or above the confidence-score threshold (0.60)"


def test_empty_or_missing_is_unavailable_not_clean():
    assert confidence.run_trust(None, 0.60)["state"] == "unavailable"
    assert confidence.run_trust(pd.DataFrame({"filename": []}), 0.60)["state"] == "unavailable"
    assert confidence.run_trust(pd.DataFrame({"confidence": []}), 0.60)["state"] == "unavailable"
```

- [ ] **Step 2: Run** → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# scat/confidence.py
"""Factual run-trust facts from per-deposit classifier confidence (spec §2.1). PURE; no reliability
claims — the classifier score (RF probability or rule-based heuristic) is uncalibrated and covers
classification only (not detection/segmentation/missed deposits). Reported as counts vs a fixed
threshold. States are workload cues, never a reliability verdict, and carry no color here."""
from __future__ import annotations


def run_trust(deposits_df, threshold: float) -> dict:
    cols = set(getattr(deposits_df, "columns", []))
    if deposits_df is None or "confidence" not in cols or len(deposits_df) == 0:
        return {"total": 0, "below": 0, "state": "unavailable", "line": "confidence unavailable"}
    total = int(len(deposits_df))
    below = int((deposits_df["confidence"] < threshold).sum())
    if below == 0:
        return {"total": total, "below": 0, "state": "none_flagged",
                "line": f"all {total} deposits at or above the confidence-score threshold ({threshold:.2f})"}
    return {"total": total, "below": below, "state": "review",
            "line": f"{below} of {total} deposits below the confidence-score threshold ({threshold:.2f}) — review recommended"}
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add scat/confidence.py tests/test_confidence.py
git commit -m "feat(confidence): factual run-trust line (counts vs a fixed threshold, no reliability claim)"
```

---

## Task 3: Review(N) column in the per-image table (values, not just header)

**Files:** Modify `scat/main_gui.py` (`ResultsTab`: `summary_table` setup ~1846-1847 + `load_results`
fill ~1948-1959); Test `tests/test_results_confidence.py`

- [ ] **Step 1: Read** — `sed -n '1846,1860p' scat/main_gui.py` and `sed -n '1948,1962p' scat/main_gui.py`
  to see the exact `setColumnCount`, headers, and the per-row `_set_num`/`setItem` fill (so the +1
  column shift is applied to every existing column).

- [ ] **Step 2: Write the failing test** (asserts VALUES + placement, per codex — not header-only)

```python
# tests/test_results_confidence.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_review_column_present_with_values(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    t = tab.summary_table
    headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert headers == ["Filename", "Review", "Normal", "ROD", "Artifact", "ROD %", "Total IOD"]
    # Column 0 is still Filename (double-click path unchanged); Normal shifted to column 2.
    assert t.item(0, 0) is not None and t.item(0, 2) is not None
    # Review cell exists for every row and reads a count or an em dash.
    for r in range(t.rowCount()):
        txt = t.item(r, 1).text()
        assert txt == "—" or txt.strip().isdigit()


def test_review_column_survives_missing_deposit_data(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    d = _results_dict_from_output(Path(res.output_dir)); d["deposit_data"] = None
    tab = ResultsTab(); tab.load_results(d)  # must not raise
    assert tab.summary_table.item(0, 1).text() == "—"
```

- [ ] **Step 3: Run** → FAIL (columnCount 6 / header mismatch).

- [ ] **Step 4: Implement** — 7 columns + shift every existing fill by +1, insert Review at col 1:

```python
# _setup_ui:
self.summary_table.setColumnCount(7)
self.summary_table.setHorizontalHeaderLabels(
    ["Filename", "Review", "Normal", "ROD", "Artifact", "ROD %", "Total IOD"])

# load_results — near the top:
from scat import metrics as _metrics
flagged = _metrics.flagged_by_image(results.get("deposit_data"), results.get("confidence_threshold", 0.60))

# in the per-row loop — Filename stays col 0, Review is col 1, the rest shift +1:
self.summary_table.setItem(i, 0, QTableWidgetItem(str(row['filename'])))
info = flagged.get(str(row['filename']))
rev = NumericTableWidgetItem(info["flagged"] if info else 0, "{:.0f}")
rev.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
if info and info["flagged"]:
    rev.setForeground(QColor(Theme.PRIMARY_LIGHT))
    frac = 100.0 * info["flagged"] / info["total"] if info["total"] else 0.0
    rev.setToolTip(f"{info['flagged']} of {info['total']} deposits below the confidence-score threshold ({frac:.0f}%)")
else:
    rev.setText("—")
self.summary_table.setItem(i, 1, rev)
self._set_num(i, 2, row['n_normal'], "{:.0f}", Theme.NORMAL)
self._set_num(i, 3, row['n_rod'], "{:.0f}", Theme.ROD)
self._set_num(i, 4, row['n_artifact'], "{:.0f}", Theme.TEXT_MUTED)
self._set_num(i, 5, row['rod_fraction'] * 100, "{:.1f}%")
self._set_num(i, 6, row.get('total_iod', 0), "{:.0f}")
```

Also update `_on_table_double_click`: confirm it reads the filename from **column 0** (unchanged) — if
it reads any other column index for data, shift it. (`grep -n "_on_table_double_click" -A15 scat/main_gui.py`.)

- [ ] **Step 5: Run + full suite** → PASS; `QT_QPA_PLATFORM=offscreen /home/lab/SCAT/.venv/bin/python -m pytest -q` all pass. Fix any test that hard-codes the old 6-column layout.
- [ ] **Step 6: Verify** — render ResultsTab; a "Review" column shows flagged counts (coral) / "—".
- [ ] **Step 7: Commit**

```bash
git add scat/main_gui.py tests/test_results_confidence.py
git commit -m "feat(gui): Review(N) column — per-image low-confidence count (from all_deposits)"
```

---

## Task 4: One factual trust line under the hero

**Files:** Modify `scat/main_gui.py` (`ResultsTab`: hero `numcol` in `_setup_ui` + `load_results`);
Test `tests/test_results_confidence.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_results_confidence.py
def test_hero_has_factual_trust_line(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    txt = tab.trust_line.text().lower()
    assert "confidence-score threshold" in txt
    for bad in ("high-confidence", "reviewed", "reliable", "trustworthy"):
        assert bad not in txt
```

- [ ] **Step 2: Run** → FAIL (`AttributeError: ... 'trust_line'`).

- [ ] **Step 3: Implement** — add ONE muted label under `hero_sub` (neutral color, no verdict tint):

```python
# _setup_ui, after hero_sub:
self.trust_line = QLabel("")
self.trust_line.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_SM}px;")
numcol.addWidget(self.trust_line)

# load_results, after the hero is set:
from scat import confidence as _conf
trust = _conf.run_trust(results.get("deposit_data"), results.get("confidence_threshold", 0.60))
self.trust_line.setText(trust["line"])
```

Do NOT add a colored traffic-light dot or a "trustworthy/clear" green state (codex: implies
reliability). The line is a neutral, factual count.

- [ ] **Step 4: Run + full suite** → PASS.
- [ ] **Step 5: Verify** — render; a muted trust line sits under the hero subtitle.
- [ ] **Step 6: Commit**

```bash
git add scat/main_gui.py tests/test_results_confidence.py
git commit -m "feat(gui): factual trust line under the hero (neutral, no reliability claim)"
```

---

## Self-review checklist

- **Scope (rescoped per codex + Jin):** ships Review(N) column + factual trust line ONLY. The
  sensitivity band and the drill-in worst-first/filter are **separate later plans** (each needs data-
  model work codex surfaced: a per-deposit second-best label; confidence in the edit path + ID-based
  `LabelingWindow` selection). Nothing here claims a band, a decrement-on-edit, or a reliability verdict.
- **Codex fixes folded:** flagged count is "any label below threshold" (no artifact-vs-normal band
  math); `run_trust` returns "unavailable" for empty/missing (never green), simple honest states, and
  generic "confidence-score" wording (RF prob OR rule heuristic); tests assert VALUES + the missing-
  `deposit_data` path + strict `<` threshold, not header-only.
- **Integration:** table 6→7 shifts every existing column fill by +1 with Filename kept at column 0;
  `_on_table_double_click` reads column 0, verified unchanged. No `LabelingWindow` changes here.
- **Placeholder scan:** none. **Type consistency:** `flagged_by_image` → `{"flagged","total"}`;
  `run_trust` → `{"total","below","state","line"}`, used identically in T3/T4.
