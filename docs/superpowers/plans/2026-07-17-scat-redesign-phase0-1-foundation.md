# SCAT redesign — Plan 1: statistical foundation + primary-metric mechanism (Phases 0–1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared foundation both redesigned surfaces render from — a single "statistical
contract" module (`scat/metrics.py`: the primary-metric registry, normalization, and rate/headline
formatting) plus the plumbing that makes the primary metric a predeclared, persisted, confirmable
endpoint (default = deposits per image).

**Architecture:** `scat/metrics.py` is the single source of truth: each selectable metric is a
value-extractor over `film_summary`. The "deposits" metric uses key **`total_deposits`** and computes
**Normal + ROD** (NOT the existing artifact-inclusive `film_summary["n_total"]` column, which many
consumers still read — we do not reuse that key). Config's existing `analysis` block gains three keys;
the pipeline threads them to the results dict + `run_manifest.json`; the agent's `analyze_folder` tool
gains a confirmed `primary_metric`. **The only surface change here** is a plumbing-proof: the hero reads
the metric. The confidence UI + sensitivity band (needs a richer per-deposit data model) are **Plan 2**;
recomposition is Plan 3; the report is Plan 4.

**Tech Stack:** Python 3.10+, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-17-scat-results-report-redesign-design.md` (§2.1, §3.1, §9 Phase 0–1).

**Incorporates codex plan-review (2026-07-17):** distinct `total_deposits` key (not `n_total`); merge
into the existing config `analysis` block; real `_results_dict_from_output` shape (`out`, `_json`,
`deposit_data`); `AnalyzeResult.output_dir` is a `str`; degrade-note surfaced; agent prompt updated +
forwarding test; shared threshold constant + guarded parse; per-image divisor = image count;
backward-compat fallback; sensitivity band deferred to Plan 2; earlier full-suite runs.

---

## File structure

- **Create `scat/metrics.py`** — `Metric` dataclass, `METRICS` registry, `DEFAULT_METRIC="total_deposits"`,
  `DEFAULT_THRESHOLD=0.60`, `NORMALIZATIONS`, `DEFAULT_NORMALIZATION`, `resolve_metric()`,
  `metric_values()`, `effective_normalization()`, `format_headline()`. Pure; no I/O.
- **Create `tests/test_metrics.py`**.
- **Modify `scat/config.py`** — add three keys to the *existing* `DEFAULT_CONFIG["analysis"]` block.
- **Modify `scat/manifest.py`** — `write_run_manifest` persists an `analysis` block (guarded).
- **Modify `scat/pipeline.py`** — `analyze_folder_service` accepts + forwards the three settings.
- **Modify `scat/tools/pipeline_tools.py`** + **`scat/agent/prompts.py`** — tool param + confirm instruction.
- **Modify `scat/main_gui.py`** `_results_dict_from_output` (~709) + `ResultsTab.load_results` (~1923).

---

## Task 1: The metric registry (`total_deposits` = Normal+ROD, with fallback)

**Files:** Create `scat/metrics.py`; Test `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import pandas as pd
from scat import metrics


def _film():
    return pd.DataFrame({
        "filename": ["a.tif", "b.tif"],
        "n_normal": [10, 20], "n_rod": [2, 0], "n_artifact": [3, 5],
        "n_total": [15, 25],  # artifact-inclusive existing column — must NOT be the deposits metric
        "rod_fraction": [2 / 12, 0.0],
        "mean_area": [80.0, 90.0], "mean_hue": [160.0, 170.0],
        "total_iod": [1000.0, 2000.0], "mean_circularity": [0.8, 0.9],
    })


def test_default_metric_is_total_deposits():
    assert metrics.DEFAULT_METRIC == "total_deposits"
    m = metrics.METRICS["total_deposits"]
    assert m.label == "Total deposits" and m.is_rate is True


def test_deposit_values_are_normal_plus_rod_not_n_total_column():
    assert list(metrics.metric_values(_film(), "total_deposits")) == [12, 20]


def test_deposit_values_fall_back_to_n_total_when_split_absent():
    legacy = pd.DataFrame({"n_total": [7, 9]})  # old dir without n_normal/n_rod
    assert list(metrics.metric_values(legacy, "total_deposits")) == [7, 9]


def test_fraction_metric_is_percent_scaled():
    assert round(metrics.metric_values(_film(), "rod_fraction").iloc[0], 2) == round(100 * 2 / 12, 2)


def test_resolve_metric_falls_back():
    assert metrics.resolve_metric("bogus") == "total_deposits"
    assert metrics.resolve_metric("mean_area") == "mean_area"
    assert metrics.DEFAULT_THRESHOLD == 0.60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'scat.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scat/metrics.py
"""The statistical contract as code (spec §2.1/§3.1): the primary-metric registry, normalization,
and headline formatting. Pure functions over a film_summary DataFrame — the single source of truth
both the result window and the report render from. No I/O."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

DEFAULT_THRESHOLD = 0.60          # fixed classification threshold (trust state/report; spec §2.1)


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    values: Callable[[pd.DataFrame], pd.Series]
    is_rate: bool                 # True = a count normalization divides (deposits, IOD)
    unit: str = ""
    fmt: str = "{:.1f}"


def _deposits(film: pd.DataFrame) -> pd.Series:
    # Deposits = Normal + ROD (artifacts are the reject class). Fall back to the legacy n_total
    # column only when the split columns are absent (old/synthetic result dirs).
    if "n_normal" in film.columns and "n_rod" in film.columns:
        return film["n_normal"].astype(float) + film["n_rod"].astype(float)
    return film["n_total"].astype(float)


def _col(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda film: film[name].astype(float)


METRICS: dict[str, Metric] = {
    "total_deposits":   Metric("total_deposits", "Total deposits", _deposits, True, "", "{:.1f}"),
    "rod_fraction":     Metric("rod_fraction", "ROD fraction", lambda f: f["rod_fraction"].astype(float) * 100, False, "%", "{:.1f}"),
    "mean_area":        Metric("mean_area", "Mean deposit area", _col("mean_area"), False, " px²", "{:.1f}"),
    "mean_hue":         Metric("mean_hue", "pH indicator (hue)", _col("mean_hue"), False, "°", "{:.1f}"),
    "total_iod":        Metric("total_iod", "Total pigment (IOD)", _col("total_iod"), True, "", "{:.0f}"),
    "mean_circularity": Metric("mean_circularity", "Mean circularity", _col("mean_circularity"), False, "", "{:.3f}"),
}

DEFAULT_METRIC = "total_deposits"


def resolve_metric(key: str | None) -> str:
    return key if key in METRICS else DEFAULT_METRIC


def metric_values(film: pd.DataFrame, key: str) -> pd.Series:
    return METRICS[resolve_metric(key)].values(film)
```

- [ ] **Step 4: Run test to verify it passes** — `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q` → PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): primary-metric registry (total_deposits = Normal+ROD, distinct from n_total)"
```

---

## Task 2: Normalization modes + rate headline (degrade note surfaced)

**Files:** Modify `scat/metrics.py`; Test `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
def test_normalizations_and_default():
    assert metrics.DEFAULT_NORMALIZATION == "per_image"
    assert metrics.NORMALIZATIONS[0] == "per_image"
    assert set(metrics.NORMALIZATIONS) == {"per_image", "per_fly", "per_area", "per_time"}


def test_per_image_divisor_is_image_count_not_non_nan_count():
    # total_iod present on both images; deposits rate over 2 images = (12+20)/2 = 16.0
    assert metrics.format_headline(_film(), "total_deposits", "per_image", meta={}) == "16.0 deposits / image"


def test_fraction_headline_is_mean_percent():
    assert metrics.format_headline(_film(), "rod_fraction", "per_image", meta={}).startswith("8.3%")


def test_per_fly_without_metadata_degrades_and_is_flagged():
    text, mode, note = metrics.effective_normalization("per_fly", meta={})
    assert mode == "per_image" and note  # a non-empty degrade note
    # and the headline reflects the effective (degraded) mode
    assert "/ image" in metrics.format_headline(_film(), "total_deposits", "per_fly", meta={})


def test_per_fly_with_metadata_normalizes():
    text, mode, note = metrics.effective_normalization("per_fly", meta={"n_flies": 8})
    assert mode == "per_fly" and note == ""
    # (12+20)/8 = 4.0 deposits / fly
    assert metrics.format_headline(_film(), "total_deposits", "per_fly", meta={"n_flies": 8}) == "4.0 deposits / fly"
```

- [ ] **Step 2: Run** → FAIL (`AttributeError: ... 'DEFAULT_NORMALIZATION'`).

- [ ] **Step 3: Implement**

```python
# append to scat/metrics.py
NORMALIZATIONS = ("per_image", "per_fly", "per_area", "per_time")
DEFAULT_NORMALIZATION = "per_image"

# Each non-default mode needs run metadata (captured in a LATER task — see Plan 1 scope note); until
# then it degrades to per_image with a note. Keys are the run_meta names the pipeline will provide.
_NORM_META_KEY = {"per_fly": "n_flies", "per_area": "roi_area", "per_time": "duration"}
_NORM_UNIT = {"per_image": "image", "per_fly": "fly", "per_area": "mm²", "per_time": "h"}


def effective_normalization(mode: str, meta: dict) -> tuple[str, str, str]:
    """Resolve a requested normalization against available run metadata.
    Returns (unit_label, effective_mode, degrade_note). degrade_note is "" when not degraded."""
    mode = mode if mode in NORMALIZATIONS else DEFAULT_NORMALIZATION
    if mode == "per_image":
        return (_NORM_UNIT["per_image"], "per_image", "")
    if meta.get(_NORM_META_KEY[mode]):
        return (_NORM_UNIT[mode], mode, "")
    return (_NORM_UNIT["per_image"], "per_image",
            f"no {_NORM_META_KEY[mode]} metadata — showing per image")


def format_headline(film: pd.DataFrame, key: str, normalization: str, meta: dict) -> str:
    """Primary-metric headline. Rate metrics show a rate (divisor = image count for per_image, or the
    metadata value), never a pooled total (spec §2.1). Fraction/mean metrics show a per-image mean."""
    m = METRICS[resolve_metric(key)]
    vals = m.values(film).dropna()
    if len(vals) == 0:
        return "—"
    if m.is_rate:
        unit, eff, _note = effective_normalization(normalization, meta)
        divisor = {"per_image": len(film), "per_fly": meta.get("n_flies"),
                   "per_area": meta.get("roi_area"), "per_time": meta.get("duration")}[eff] or len(film)
        rate = float(vals.sum()) / float(divisor)
        noun = m.label.lower().replace("total ", "")  # "Total deposits" -> "deposits"
        return f"{m.fmt.format(rate)} {noun} / {unit}"
    return f"{m.fmt.format(float(vals.mean()))}{m.unit}"
```

- [ ] **Step 4: Run** → PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): normalization modes + rate headline with surfaced degrade note"
```

> **Deferred (own task, later in this plan's follow-ups):** capturing `n_flies` / `roi_area` /
> `duration` into `run_meta`. Until then `per_fly/area/time` are *selectable but degrade to per_image
> with a note* — the honest behavior codex asked for. The sensitivity band (spec §2.1 flip rule) needs
> a per-deposit second-best label that `all_deposits.csv` does not carry, so it is **Plan 2**, designed
> alongside the confidence data model.

---

## Task 3: Add the three keys to the EXISTING config `analysis` block

**Files:** Modify `scat/config.py`; Test `tests/test_config_analysis.py`

- [ ] **Step 1: Read the block** — `sed -n '/"analysis"/,/},/p' scat/config.py`. Confirm it already holds
  `model_type/annotate/visualize/spatial/stats/report`. We **merge** into it (never redefine it).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config_analysis.py
from scat.config import DEFAULT_CONFIG


def test_analysis_gains_contract_keys_without_losing_existing():
    a = DEFAULT_CONFIG["analysis"]
    assert a["model_type"] == "rf" and "annotate" in a and "visualize" in a  # existing keys intact
    assert a["primary_metric"] == "total_deposits"
    assert a["normalization"] == "per_image"
    assert a["confidence_threshold"] == 0.60
```

- [ ] **Step 3: Run** → FAIL `KeyError: 'primary_metric'`.

- [ ] **Step 4: Implement** — add three lines *inside* the existing `"analysis": { ... }` literal:

```python
        "primary_metric": "total_deposits",   # predeclared endpoint (metrics.DEFAULT_METRIC)
        "normalization": "per_image",          # per_image | per_fly | per_area | per_time
        "confidence_threshold": 0.60,          # fixed classification threshold (metrics.DEFAULT_THRESHOLD)
```

- [ ] **Step 5: Run + full suite** → `pytest tests/test_config_analysis.py -q` PASS; then
  `QT_QPA_PLATFORM=offscreen python -m pytest -q` → all pass (config merge didn't drop keys).

- [ ] **Step 6: Commit**

```bash
git add scat/config.py tests/test_config_analysis.py
git commit -m "feat(config): analysis block gains primary_metric/normalization/confidence_threshold"
```

---

## Task 4: Persist the analysis contract in the run manifest (guarded)

**Files:** Modify `scat/manifest.py` (`write_run_manifest`, ~112); Test `tests/test_manifest.py`

- [ ] **Step 1: Read** — `sed -n '112,135p' scat/manifest.py` (kwargs + the dict it builds/writes).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_manifest.py (add)
import json
from scat.manifest import write_run_manifest


def test_manifest_records_analysis(tmp_path):
    write_run_manifest(tmp_path, image_paths=[], model_type="rf",
                       primary_metric="rod_fraction", normalization="per_fly", confidence_threshold=0.6)
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert m["analysis"] == {"primary_metric": "rod_fraction", "normalization": "per_fly",
                             "confidence_threshold": 0.6}


def test_manifest_analysis_defaults_and_survives_bad_threshold(tmp_path):
    write_run_manifest(tmp_path, image_paths=[], model_type="rf", confidence_threshold="oops")
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert m["analysis"]["primary_metric"] == "total_deposits"
    assert m["analysis"]["confidence_threshold"] == 0.60  # bad value → default, never raises
```

- [ ] **Step 3: Run** → FAIL (`TypeError: unexpected keyword 'primary_metric'`).

- [ ] **Step 4: Implement** — add kwargs + a guarded `analysis` block:

```python
from scat import metrics as _metrics   # top of scat/manifest.py

def write_run_manifest(output_dir, *, path=None, image_paths, model_type=None, model_path=None,
                       # ...existing kwargs...
                       primary_metric=None, normalization=None, confidence_threshold=None,
                       # ...existing kwargs...):
    def _thr(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return _metrics.DEFAULT_THRESHOLD
    # ...existing body building the manifest dict...
    manifest["analysis"] = {
        "primary_metric": _metrics.resolve_metric(primary_metric),
        "normalization": normalization if normalization in _metrics.NORMALIZATIONS else _metrics.DEFAULT_NORMALIZATION,
        "confidence_threshold": _thr(confidence_threshold),
    }
```

- [ ] **Step 5: Run + full suite** → `pytest tests/test_manifest.py -q` PASS; `pytest -q` all pass.

- [ ] **Step 6: Commit**

```bash
git add scat/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): persist the analysis contract (guarded threshold parse)"
```

---

## Task 5: Thread the settings through `analyze_folder_service`

**Files:** Modify `scat/pipeline.py` (`analyze_folder_service`, ~80); Test `tests/test_pipeline_primary_metric.py`

- [ ] **Step 1: Read** — `sed -n '80,180p' scat/pipeline.py`; find its `write_run_manifest(...)` call.

- [ ] **Step 2: Write the failing test** (note: `output_dir` is a **str** — wrap in `Path`)

```python
# tests/test_pipeline_primary_metric.py
import json
from pathlib import Path
from scat.pipeline import analyze_folder_service


def test_service_persists_primary_metric(synth_dir, tmp_path):
    out = tmp_path / "out"
    res = analyze_folder_service(str(synth_dir), output_dir=str(out),
                                 primary_metric="rod_fraction", normalization="per_image",
                                 confidence_threshold=0.6, annotate=False)
    m = json.loads((Path(res.output_dir) / "run_manifest.json").read_text())
    assert m["analysis"]["primary_metric"] == "rod_fraction"
```

- [ ] **Step 3: Run** → FAIL (`TypeError: unexpected keyword 'primary_metric'`).

- [ ] **Step 4: Implement** — add the three kwargs (default `None`) and forward them to the
  `write_run_manifest(...)` call inside the service.

- [ ] **Step 5: Run + full suite** → PASS; `pytest -q` all pass.

- [ ] **Step 6: Commit**

```bash
git add scat/pipeline.py tests/test_pipeline_primary_metric.py
git commit -m "feat(pipeline): thread primary_metric/normalization/threshold into the manifest"
```

---

## Task 6: Tool param + agent confirmation instruction

**Files:** Modify `scat/tools/pipeline_tools.py` (`analyze_folder`, ~10) + `scat/agent/prompts.py`;
Test `tests/test_tools_primary_metric.py`

- [ ] **Step 1: Write the failing test** (forwarding, not just signature — monkeypatch the service)

```python
# tests/test_tools_primary_metric.py
import scat.tools.pipeline_tools as pt


def test_tool_forwards_primary_metric(monkeypatch, synth_dir):
    seen = {}
    def fake(path, **kw):
        seen.update(kw)
        class R: output_dir = "/tmp/x"
        return R()
    monkeypatch.setattr(pt, "analyze_folder_service", fake)
    pt.analyze_folder(str(synth_dir), primary_metric="mean_area")
    assert seen.get("primary_metric") == "mean_area"
```

- [ ] **Step 2: Run** → FAIL (`TypeError: unexpected keyword 'primary_metric'`).

- [ ] **Step 3: Implement** — add the param + forward it, and extend the `@tool` description with one
  sentence; add a confirmation instruction to the agent prompt in `scat/agent/prompts.py`:

```python
# pipeline_tools.py
@tool(description="... existing ... Pass primary_metric (total_deposits [DEFAULT], rod_fraction, "
      "mean_area, mean_hue, total_iod, mean_circularity) — the predeclared endpoint this experiment "
      "measures; CONFIRM it with the user before running, never silently guess.")
def analyze_folder(path: str, groups=None, model_type=None, primary_metric=None, ...):
    return analyze_folder_service(path, groups=groups, model_type=model_type,
                                  primary_metric=primary_metric, ...)
```
In `scat/agent/prompts.py`, add to the system prompt: *"Before analyze_folder, confirm the primary
metric (what the experiment measures) with the user; default total_deposits if they don't specify."*

- [ ] **Step 4: Run + full suite** → PASS; `pytest -q` all pass (incl. `test_tools_registered`).

- [ ] **Step 5: Commit**

```bash
git add scat/tools/pipeline_tools.py scat/agent/prompts.py tests/test_tools_primary_metric.py
git commit -m "feat(agent): analyze_folder confirms + forwards the primary metric endpoint"
```

---

## Task 7: Surface the contract in the results dict (real shape)

**Files:** Modify `scat/main_gui.py` `_results_dict_from_output` (~709, uses `out` + `_json`, returns
`deposit_data`); Test `tests/test_results_dict_analysis.py`

- [ ] **Step 1: Read** — `sed -n '709,745p' scat/main_gui.py`. Confirm: `out = Path(output_dir)`,
  local `import json as _json` (or module `_json`), returns a dict with `film_summary` + `deposit_data`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_results_dict_analysis.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from scat.main_gui import _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_results_dict_carries_analysis_and_keeps_deposit_data(synth_dir, tmp_path):
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="rod_fraction", annotate=False)
    d = _results_dict_from_output(Path(res.output_dir))
    assert d["primary_metric"] == "rod_fraction"
    assert d["normalization"] == "per_image"
    assert d["confidence_threshold"] == 0.60
    assert d["run_meta"] == {}                 # empty until metadata capture (later task)
    assert "deposit_data" in d                 # existing key preserved (GUI edit paths read it)
```

- [ ] **Step 3: Run** → FAIL (`KeyError: 'primary_metric'`).

- [ ] **Step 4: Implement** — using the REAL locals (`out`, `_json`), read the manifest `analysis` and
  add keys to the existing return dict (keep `deposit_data`; add `run_meta` empty for now):

```python
from scat import metrics as _metrics   # top of main_gui.py

# inside _results_dict_from_output, before the return:
    analysis = {}
    mpath = out / "run_manifest.json"
    if mpath.exists():
        try:
            analysis = (_json.loads(mpath.read_text()) or {}).get("analysis", {}) or {}
        except Exception:
            analysis = {}
    # ... in the returned dict literal, add:
        "primary_metric": _metrics.resolve_metric(analysis.get("primary_metric")),
        "normalization": analysis.get("normalization") or _metrics.DEFAULT_NORMALIZATION,
        "confidence_threshold": float(analysis.get("confidence_threshold", _metrics.DEFAULT_THRESHOLD)),
        "run_meta": {},   # n_flies/roi_area/duration land here in the metadata-capture task
```

- [ ] **Step 5: Run + full suite** → PASS; `pytest -q` all pass.

- [ ] **Step 6: Commit**

```bash
git add scat/main_gui.py tests/test_results_dict_analysis.py
git commit -m "feat(gui): results dict carries the analysis contract (deposit_data + run_meta preserved)"
```

---

## Task 8: The one surface change — hero reads the primary metric

**Files:** Modify `scat/main_gui.py` `ResultsTab.load_results` (hero block ~1923-1925); Test
`tests/test_results_primary_metric.py`. **This is the single allowed surface change in Plan 1** (a
plumbing proof; full recomposition is Plan 3), so it carries full GUI regression.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_results_primary_metric.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_hero_reflects_primary_metric(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="total_deposits", annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    assert "/ image" in tab.hero_value.text()               # deposits as a rate
    assert "DEPOSIT" in tab.hero_kicker.text().upper()      # not "ROD FRACTION"
```

- [ ] **Step 2: Run** → FAIL (hero still "MEAN ROD FRACTION" / "8.3%").

- [ ] **Step 3: Implement** — replace the hard-coded ROD hero in `load_results`:

```python
from scat import metrics as _metrics

# where hero_kicker/hero_value/hero_sub are set (~1923-1925):
pm = _metrics.resolve_metric(results.get("primary_metric"))
norm = results.get("normalization", _metrics.DEFAULT_NORMALIZATION)
meta = results.get("run_meta", {})
m = _metrics.METRICS[pm]
self.hero_kicker.setText(m.label.upper())
self.hero_value.setText(_metrics.format_headline(film_summary, pm, norm, meta))
vals = _metrics.metric_values(film_summary, pm).dropna()
sd = vals.std() if len(vals) > 1 else 0.0
self.hero_sub.setText(f"±{sd:.1f}{m.unit}   ·   across {n} image{'s' if n != 1 else ''}")
```

- [ ] **Step 4: Run + full suite** → PASS. Then `QT_QPA_PLATFORM=offscreen python -m pytest -q` → all
  pass. **If `test_gui_slimdown` asserts the literal "MEAN ROD FRACTION" hero, update that assertion**
  to the metric-driven text (it's an intended change).

- [ ] **Step 5: Verify visually** — run the standing ResultsTab render harness; confirm the hero reads
  e.g. "TOTAL DEPOSITS / 16.0 deposits / image" for the default metric, and "MEAN ROD FRACTION / 8.3%"
  when `primary_metric="rod_fraction"`.

- [ ] **Step 6: Commit**

```bash
git add scat/main_gui.py tests/test_results_primary_metric.py
git commit -m "feat(gui): hero reads the configurable primary metric (default deposits/image)"
```

---

## Self-review checklist (run before handoff)

- **Spec coverage (Phase 0–1):** registry + deposits=Normal+ROD w/ fallback (T1); normalization + rate
  headline + degrade note (T2); config merge (T3); manifest persistence (T4); pipeline threading (T5);
  agent confirmation + forwarding (T6); results dict real-shape (T7); hero reads metric (T8). Sensitivity
  band + confidence UI = Plan 2; recomposition = Plan 3; report = Plan 4.
- **Codex fixes folded:** distinct `total_deposits` key; config merged not clobbered; `out`/`_json`/
  `deposit_data` real shape; `Path(res.output_dir)`; degrade note surfaced; prompt + forwarding test;
  guarded threshold + shared constant; per-image divisor = image count; backward-compat fallback; full
  suite after T3/T4/T5/T7/T8; T8 flagged as the one surface change.
- **Placeholder scan:** none. **Type consistency:** `resolve_metric`/`metric_values`/`format_headline`/
  `effective_normalization`/`Metric.is_rate`/`Metric.unit` used identically across tasks.
- **No confidence UI here** — nothing in this plan renders a "reviewed" or "high-confidence" claim.
