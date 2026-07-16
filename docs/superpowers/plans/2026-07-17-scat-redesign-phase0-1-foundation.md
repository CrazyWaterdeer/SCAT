# SCAT redesign — Plan 1: statistical foundation + primary-metric mechanism (Phases 0–1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared foundation both redesigned surfaces render from — a single "statistical
contract" module (the primary-metric registry, normalization, headline formatting, and the sensitivity
flip rules) plus the plumbing that makes the primary metric a predeclared, persisted, confirmable
endpoint (default = deposits per image).

**Architecture:** A new pure module `scat/metrics.py` is the single source of truth: it defines each
selectable metric as a value-extractor over `film_summary` (so "deposits" = Normal+ROD, not the
artifact-inclusive `n_total` column), the normalization modes, the headline/rate formatting, and the
metric-specific flip rule used later for the sensitivity band. Config gains defaults; the pipeline
threads `primary_metric` / `normalization` / `confidence_threshold` through to the results dict and
`run_manifest.json`; the agent's `analyze_folder` tool accepts a confirmed `primary_metric`. No UI
recomposition here (that is Plans 2–4) — only the data contract + a minimal wiring the renderers read.

**Tech Stack:** Python 3.10+, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-17-scat-results-report-redesign-design.md` (§2.1, §3.1, §9 Phase 0–1).

**Scope note:** This is Plan 1 of a phased effort. Plan 2 = confidence in the result window; Plan 3 =
result-window recomposition; Plan 4 = report "Findings Note". Each is its own plan/commit stream and
ships green independently. This plan produces working, tested software on its own (the metrics contract
+ persisted primary metric), even before any surface is recomposed.

---

## File structure

- **Create `scat/metrics.py`** — the statistical contract: `Metric` dataclass, `METRICS` registry,
  `DEFAULT_METRIC`, `NORMALIZATIONS`, `DEFAULT_METRIC`, `metric_values()`, `normalize_label()`,
  `format_headline()`, `flip_rule()`. Pure functions over a `film_summary` DataFrame; no I/O.
- **Create `tests/test_metrics.py`** — unit tests for the module.
- **Modify `scat/config.py`** — add an `analysis` block to `DEFAULT_CONFIG` (primary_metric,
  normalization, confidence_threshold).
- **Modify `scat/manifest.py`** — `write_run_manifest` persists an `analysis` block.
- **Modify `scat/pipeline.py`** — `analyze_folder_service` accepts + forwards the three settings;
  writes them to the manifest.
- **Modify `scat/tools/pipeline_tools.py`** — the `analyze_folder` `@tool` accepts `primary_metric`.
- **Modify `scat/main_gui.py`** `_results_dict_from_output` — read `analysis` from the manifest (with
  defaults) and include `primary_metric` / `normalization` / `confidence_threshold` in the results dict.
- **Modify `tests/test_manifest*.py` / `tests/test_pipeline*.py`** — extend for the new fields.

---

## Task 1: The metric registry (`Metric` + `METRICS` + values)

**Files:**
- Create: `scat/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import pandas as pd
from scat import metrics


def _film():
    # 2 images: (normal, rod, artifact) = (10,2,3) and (20,0,5)
    return pd.DataFrame({
        "filename": ["a.tif", "b.tif"],
        "n_normal": [10, 20], "n_rod": [2, 0], "n_artifact": [3, 5],
        "n_total": [15, 25],  # artifact-inclusive column — must NOT be used for "deposits"
        "rod_fraction": [2 / 12, 0.0],
        "mean_area": [80.0, 90.0], "mean_hue": [160.0, 170.0],
        "total_iod": [1000.0, 2000.0], "mean_circularity": [0.8, 0.9],
    })


def test_default_metric_is_deposit_count():
    assert metrics.DEFAULT_METRIC == "n_total"          # config key
    m = metrics.METRICS["n_total"]
    assert m.label == "Total deposits" and m.is_rate is True


def test_deposit_values_exclude_artifacts():
    # "deposits" per image = Normal + ROD (12 and 20), NOT the n_total column (15, 25)
    vals = metrics.metric_values(_film(), "n_total")
    assert list(vals) == [12, 20]


def test_fraction_metric_reads_its_column():
    vals = metrics.metric_values(_film(), "rod_fraction")
    assert round(vals.iloc[0], 4) == round(2 / 12, 4)


def test_unknown_metric_falls_back_to_default():
    assert metrics.resolve_metric("bogus") == "n_total"
    assert metrics.resolve_metric("rod_fraction") == "rod_fraction"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scat.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scat/metrics.py
"""The statistical contract as code (spec §2.1/§3.1): the primary-metric registry, normalization,
headline formatting, and the sensitivity flip rules. Pure functions over a film_summary DataFrame —
the single source of truth both the result window and the report render from. No I/O."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Metric:
    key: str                        # config value + manifest key, e.g. "n_total"
    label: str                      # human label, e.g. "Total deposits"
    values: Callable[[pd.DataFrame], pd.Series]  # per-image values from film_summary
    is_rate: bool                   # True = a count that normalization divides (deposits, IOD)
    unit: str = ""                  # display unit, e.g. "%", "px²", "°"
    fmt: str = "{:.1f}"             # value format


def _deposits(film: pd.DataFrame) -> pd.Series:
    # "Deposits" = Normal + ROD; artifacts are the reject class (spec, commit 9400e0f).
    return film["n_normal"].astype(float) + film["n_rod"].astype(float)


def _col(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda film: film[name].astype(float)


METRICS: dict[str, Metric] = {
    "n_total":          Metric("n_total", "Total deposits", _deposits, True, "", "{:.1f}"),
    "rod_fraction":     Metric("rod_fraction", "ROD fraction", lambda f: _col("rod_fraction")(f) * 100, False, "%", "{:.1f}"),
    "mean_area":        Metric("mean_area", "Mean deposit area", _col("mean_area"), False, "px²", "{:.1f}"),
    "mean_hue":         Metric("mean_hue", "pH indicator (hue)", _col("mean_hue"), False, "°", "{:.1f}"),
    "total_iod":        Metric("total_iod", "Total pigment (IOD)", _col("total_iod"), True, "", "{:.0f}"),
    "mean_circularity": Metric("mean_circularity", "Mean circularity", _col("mean_circularity"), False, "", "{:.3f}"),
}

DEFAULT_METRIC = "n_total"


def resolve_metric(key: str | None) -> str:
    """Return a valid metric key, falling back to the default for unknown/None."""
    return key if key in METRICS else DEFAULT_METRIC


def metric_values(film: pd.DataFrame, key: str) -> pd.Series:
    """Per-image values for a metric (pre-normalization), NaNs dropped by the caller as needed."""
    return METRICS[resolve_metric(key)].values(film)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): primary-metric registry (deposits = Normal+ROD, not n_total column)"
```

---

## Task 2: Normalization modes + headline formatting

**Files:**
- Modify: `scat/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
def test_normalizations_listed_with_default():
    assert metrics.DEFAULT_NORMALIZATION == "per_image"
    assert metrics.NORMALIZATIONS[0] == "per_image"
    assert set(metrics.NORMALIZATIONS) == {"per_image", "per_fly", "per_area", "per_time"}


def test_rate_metric_headline_is_a_rate_not_a_pooled_total():
    # deposits per image over 2 images: (12 + 20) / 2 = 16.0 per image
    text = metrics.format_headline(_film(), "n_total", "per_image", meta={})
    assert text == "16.0 deposits / image"


def test_fraction_metric_headline_is_a_mean_percent():
    text = metrics.format_headline(_film(), "rod_fraction", "per_image", meta={})
    # mean of (16.667%, 0%) = 8.3%
    assert text.startswith("8.3%")


def test_normalization_degrades_to_per_image_without_metadata():
    # per_fly requested but no fly count in meta -> falls back, flagged
    text, mode = metrics.effective_normalization("per_fly", meta={})
    assert mode == "per_image" and "per_image" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q`
Expected: FAIL (`AttributeError: module 'scat.metrics' has no attribute 'DEFAULT_NORMALIZATION'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to scat/metrics.py
NORMALIZATIONS = ("per_image", "per_fly", "per_area", "per_time")
DEFAULT_NORMALIZATION = "per_image"

# meta key each non-default normalization needs (run metadata; absent => degrade to per_image)
_NORM_META_KEY = {"per_fly": "n_flies", "per_area": "roi_area", "per_time": "duration"}
_NORM_UNIT = {"per_image": "image", "per_fly": "fly", "per_area": "mm²", "per_time": "h"}


def effective_normalization(mode: str, meta: dict) -> tuple[str, str]:
    """Resolve the requested normalization against available run metadata.
    Returns (human_note, effective_mode). Degrades to per_image with a note when metadata is missing."""
    mode = mode if mode in NORMALIZATIONS else DEFAULT_NORMALIZATION
    if mode == "per_image":
        return ("per image", "per_image")
    key = _NORM_META_KEY[mode]
    if meta.get(key):
        return (f"per {_NORM_UNIT[mode]}", mode)
    return (f"per image (no {key} metadata; per {_NORM_UNIT[mode]} unavailable)", "per_image")


def format_headline(film: pd.DataFrame, key: str, normalization: str, meta: dict) -> str:
    """The primary-metric headline string. Rate metrics (counts) are shown as a rate, never a
    pooled total (spec §2.1); fraction/mean metrics show their per-image mean."""
    m = METRICS[resolve_metric(key)]
    vals = m.values(film).dropna()
    if len(vals) == 0:
        return "—"
    if m.is_rate:
        _note, eff = effective_normalization(normalization, meta)
        divisor = {"per_image": len(vals), "per_fly": meta.get("n_flies"),
                   "per_area": meta.get("roi_area"), "per_time": meta.get("duration")}[eff] or len(vals)
        rate = float(vals.sum()) / float(divisor)
        noun = m.label.lower()  # "total deposits" -> we want "deposits / image"
        noun = noun.replace("total ", "")
        return f"{m.fmt.format(rate)} {noun} / {_NORM_UNIT[eff]}"
    mean = float(vals.mean())
    return f"{m.fmt.format(mean)}{m.unit}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): normalization modes + rate-not-total headline formatting"
```

---

## Task 3: Metric-specific flip rule (sensitivity-band foundation)

**Files:**
- Modify: `scat/metrics.py`
- Test: `tests/test_metrics.py`

Per spec §2.1, the flip rule is defined here (before any UI copy). v1 rule: the sensitivity band
recomputes the primary metric at the two extremes of relabeling the flagged (low-confidence) deposits
to their two most-plausible alternatives. This task ships the pure `metric_sensitivity_range()` used
later by Plan 2's UI; it does not draw anything.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
def test_rod_fraction_sensitivity_range_widens_with_flagged_rod_calls():
    # image with 8 normal, 2 rod, of which 1 rod call is flagged (low-confidence).
    # ROD fraction now = 2/10 = 20%. If that flagged ROD were actually Normal -> 1/10 = 10%.
    # If a flagged Normal were actually ROD (none flagged here) -> unchanged upper.
    lo, hi = metrics.metric_sensitivity_range(
        n_normal=8, n_rod=2, flagged_normal_to_rod=0, flagged_rod_to_normal=1, key="rod_fraction")
    assert round(lo, 1) == 10.0 and round(hi, 1) == 20.0


def test_deposit_count_range_is_flat_under_normal_rod_flips():
    # Normal<->ROD flips don't change the deposit COUNT (both are deposits); range is a point.
    lo, hi = metrics.metric_sensitivity_range(
        n_normal=8, n_rod=2, flagged_normal_to_rod=1, flagged_rod_to_normal=1, key="n_total")
    assert lo == hi == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q`
Expected: FAIL (`AttributeError: ... 'metric_sensitivity_range'`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to scat/metrics.py
def metric_sensitivity_range(*, n_normal: int, n_rod: int,
                             flagged_normal_to_rod: int, flagged_rod_to_normal: int,
                             key: str) -> tuple[float, float]:
    """Sensitivity of a per-image metric to reclassifying the flagged (low-confidence) deposits, per
    the metric-specific flip rule (spec §2.1). Returns (low, high). Not a CI — a label-sensitivity
    bound only. v1 covers the count/fraction metrics that depend only on Normal/ROD tallies."""
    key = resolve_metric(key)
    # extreme A: every flagged ROD is actually Normal; extreme B: every flagged Normal is actually ROD
    normal_a, rod_a = n_normal + flagged_rod_to_normal, n_rod - flagged_rod_to_normal
    normal_b, rod_b = n_normal - flagged_normal_to_rod, n_rod + flagged_normal_to_rod
    if key == "rod_fraction":
        def frac(nn, nr):
            d = nn + nr
            return 0.0 if d == 0 else 100.0 * nr / d
        vals = [frac(normal_a, rod_a), frac(normal_b, rod_b)]
        return (min(vals), max(vals))
    if key == "n_total":
        # deposit count = Normal + ROD; Normal<->ROD flips leave it unchanged
        d = n_normal + n_rod
        return (float(d), float(d))
    # mean/area/hue/circularity/iod: v1 does not model per-deposit flips (needs deposit measurements);
    # report a degenerate (point) range until Plan 2 extends it with deposit-level values.
    return (float("nan"), float("nan"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_metrics.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add scat/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): metric-specific sensitivity flip rule (label-sensitivity bound)"
```

---

## Task 4: Config defaults for the analysis contract

**Files:**
- Modify: `scat/config.py` (the `DEFAULT_CONFIG` dict)
- Test: `tests/test_config_analysis.py` (create)

- [ ] **Step 1: Read the current defaults**

Run: `grep -n "DEFAULT_CONFIG" scat/config.py`
Then read the `DEFAULT_CONFIG = { ... }` literal so the new block is inserted consistently (same
indentation / trailing comma style).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config_analysis.py
from scat.config import DEFAULT_CONFIG


def test_analysis_defaults_present():
    a = DEFAULT_CONFIG["analysis"]
    assert a["primary_metric"] == "n_total"
    assert a["normalization"] == "per_image"
    assert a["confidence_threshold"] == 0.60
```

- [ ] **Step 3: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_config_analysis.py -q`
Expected: FAIL (`KeyError: 'analysis'`).

- [ ] **Step 4: Add the block**

In `scat/config.py`, add to the `DEFAULT_CONFIG` dict (top level, beside the existing groups):

```python
    "analysis": {
        "primary_metric": "n_total",      # predeclared endpoint (metrics.DEFAULT_METRIC)
        "normalization": "per_image",     # per_image | per_fly | per_area | per_time
        "confidence_threshold": 0.60,     # fixed classification threshold for trust state/report
    },
```

- [ ] **Step 5: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_config_analysis.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scat/config.py tests/test_config_analysis.py
git commit -m "feat(config): analysis defaults (primary_metric, normalization, confidence_threshold)"
```

---

## Task 5: Persist the analysis contract in the run manifest

**Files:**
- Modify: `scat/manifest.py` (`write_run_manifest`, ~line 112)
- Test: `tests/test_manifest.py` (extend; create if absent)

- [ ] **Step 1: Read `write_run_manifest`**

Run: `sed -n '112,140p' scat/manifest.py` — note its keyword args and the returned/written dict shape.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_manifest.py (add)
import json
from pathlib import Path
from scat.manifest import write_run_manifest


def test_manifest_records_analysis_block(tmp_path):
    write_run_manifest(tmp_path, image_paths=[], model_type="rf",
                       primary_metric="rod_fraction", normalization="per_fly",
                       confidence_threshold=0.6)
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert m["analysis"] == {"primary_metric": "rod_fraction",
                             "normalization": "per_fly", "confidence_threshold": 0.6}


def test_manifest_analysis_defaults_when_omitted(tmp_path):
    write_run_manifest(tmp_path, image_paths=[], model_type="rf")
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert m["analysis"]["primary_metric"] == "n_total"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_manifest.py -k analysis -q`
Expected: FAIL (`TypeError: unexpected keyword 'primary_metric'` or `KeyError: 'analysis'`).

- [ ] **Step 4: Implement**

In `write_run_manifest`, add keyword args (defaulting from `scat.metrics`) and an `analysis` block in
the written dict:

```python
from scat import metrics as _metrics   # top of scat/manifest.py

def write_run_manifest(output_dir, *, path=None, image_paths, model_type=None, model_path=None,
                       # ... existing kwargs ...
                       primary_metric=None, normalization=None, confidence_threshold=None,
                       # ... existing kwargs ...
                       ):
    # ... existing body building the manifest dict ...
    manifest["analysis"] = {
        "primary_metric": _metrics.resolve_metric(primary_metric),
        "normalization": normalization if normalization in _metrics.NORMALIZATIONS else _metrics.DEFAULT_NORMALIZATION,
        "confidence_threshold": float(confidence_threshold) if confidence_threshold is not None else 0.60,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_manifest.py -k analysis -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add scat/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): persist the analysis contract (primary_metric/normalization/threshold)"
```

---

## Task 6: Thread the settings through `analyze_folder_service`

**Files:**
- Modify: `scat/pipeline.py` (`analyze_folder_service`, ~line 80; its `write_run_manifest` call)
- Test: `tests/test_pipeline_primary_metric.py` (create)

- [ ] **Step 1: Read the service + its manifest call**

Run: `sed -n '80,180p' scat/pipeline.py` — find where it calls `write_run_manifest` and where it
returns/collects the results scalars.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_pipeline_primary_metric.py
import json
from scat.pipeline import analyze_folder_service


def test_service_persists_primary_metric_to_manifest(synth_dir):
    res = analyze_folder_service(str(synth_dir), primary_metric="rod_fraction",
                                 normalization="per_image", confidence_threshold=0.6)
    m = json.loads((res.output_dir / "run_manifest.json").read_text())
    assert m["analysis"]["primary_metric"] == "rod_fraction"
```

(`synth_dir` is the existing fixture in `tests/conftest.py`.)

- [ ] **Step 3: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_pipeline_primary_metric.py -q`
Expected: FAIL (`TypeError: unexpected keyword 'primary_metric'`).

- [ ] **Step 4: Implement**

Add the three keyword args to `analyze_folder_service` (defaulting to `None`) and pass them into its
`write_run_manifest(...)` call:

```python
def analyze_folder_service(path, groups=None, model_type=None, ...,
                           primary_metric=None, normalization=None, confidence_threshold=None, ...):
    # ... unchanged detection/classification ...
    write_run_manifest(output_dir, path=path, image_paths=image_paths, model_type=model_type,
                       # ... existing args ...
                       primary_metric=primary_metric, normalization=normalization,
                       confidence_threshold=confidence_threshold)
    # ... unchanged ...
```

- [ ] **Step 5: Run test to verify it passes + full suite green**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_pipeline_primary_metric.py -q`
Expected: PASS.
Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: all pass (no regression; existing callers use the new defaults).

- [ ] **Step 6: Commit**

```bash
git add scat/pipeline.py tests/test_pipeline_primary_metric.py
git commit -m "feat(pipeline): thread primary_metric/normalization/threshold into the run manifest"
```

---

## Task 7: `analyze_folder` @tool accepts a confirmed primary metric

**Files:**
- Modify: `scat/tools/pipeline_tools.py` (`analyze_folder`, ~line 10)
- Test: `tests/test_tools_primary_metric.py` (create)

Spec §2.1/§3.1: the metric is predeclared and confirmed, not silently inferred. The tool exposes a
`primary_metric` parameter; the agent's job (its prompt, updated here) is to CONFIRM it with the user
before calling. This task adds the parameter + forwards it; the prompt wording is a one-line doc change.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_primary_metric.py
import inspect
from scat.tools.pipeline_tools import analyze_folder


def test_tool_exposes_primary_metric_param():
    assert "primary_metric" in inspect.signature(analyze_folder).parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_tools_primary_metric.py -q`
Expected: FAIL (`AssertionError`).

- [ ] **Step 3: Implement**

Add the parameter to `analyze_folder` and forward it to `analyze_folder_service`; extend the `@tool`
description with one sentence:

```python
@tool(description="... existing ... Pass primary_metric (one of: n_total [deposits, DEFAULT], "
      "rod_fraction, mean_area, mean_hue, total_iod, mean_circularity) — the predeclared endpoint "
      "this experiment measures; CONFIRM it with the user before running, do not silently guess.")
def analyze_folder(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                   primary_metric: Optional[str] = None, ...):
    return analyze_folder_service(path, groups=groups, model_type=model_type,
                                  primary_metric=primary_metric, ...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_tools_primary_metric.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scat/tools/pipeline_tools.py tests/test_tools_primary_metric.py
git commit -m "feat(agent): analyze_folder accepts a confirmed primary_metric endpoint"
```

---

## Task 8: Surface the contract in the results dict + load deposit data

**Files:**
- Modify: `scat/main_gui.py` (`_results_dict_from_output`, ~line 699)
- Test: `tests/test_results_dict_analysis.py` (create)

The result window + report both read the results dict. Add the analysis block (from the manifest, with
metrics defaults) and confirm the per-deposit data is present (needed by Plan 2's Review(N)).

- [ ] **Step 1: Read `_results_dict_from_output`**

Run: `sed -n '699,745p' scat/main_gui.py` — note it already reads `deposit_data` and returns a dict
with `film_summary`; add the analysis fields + ensure `deposits` is keyed in the returned dict.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_results_dict_analysis.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from pathlib import Path
from scat.main_gui import _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_results_dict_carries_analysis_and_deposits(synth_dir):
    res = analyze_folder_service(str(synth_dir), primary_metric="rod_fraction")
    d = _results_dict_from_output(Path(res.output_dir))
    assert d["primary_metric"] == "rod_fraction"
    assert d["normalization"] == "per_image"
    assert d["confidence_threshold"] == 0.60
    assert d.get("deposits") is not None      # per-deposit data available for Review(N)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_results_dict_analysis.py -q`
Expected: FAIL (`KeyError: 'primary_metric'`).

- [ ] **Step 4: Implement**

In `_results_dict_from_output`, read the manifest's `analysis` block (defaulting via `scat.metrics`),
and add the fields + the deposits data to the returned dict:

```python
from scat import metrics as _metrics

def _results_dict_from_output(output_dir, group_by=None, image_paths=None, stats=None):
    # ... existing reads: film_summary, deposit_data ...
    analysis = {}
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            analysis = json.loads(manifest_path.read_text()).get("analysis", {}) or {}
        except Exception:
            analysis = {}
    # ... existing return dict, with these added keys:
    return {
        # ... existing keys (film_summary, output_dir, group_by, ...) ...
        "deposits": deposit_data,
        "primary_metric": _metrics.resolve_metric(analysis.get("primary_metric")),
        "normalization": analysis.get("normalization") or _metrics.DEFAULT_NORMALIZATION,
        "confidence_threshold": float(analysis.get("confidence_threshold", 0.60)),
    }
```

- [ ] **Step 5: Run test to verify it passes + full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_results_dict_analysis.py -q`
Expected: PASS.
Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scat/main_gui.py tests/test_results_dict_analysis.py
git commit -m "feat(gui): results dict carries the analysis contract + per-deposit data"
```

---

## Task 9: Minimal wiring — the hero reads the primary metric

**Files:**
- Modify: `scat/main_gui.py` (`ResultsTab.load_results`, the hero kicker/value; ~line 1912)
- Test: extend `tests/test_gui_slimdown.py` or a new `tests/test_results_primary_metric.py`

No recomposition yet (that is Plan 3) — just prove the plumbing reaches the surface: the hero shows the
chosen primary metric via `metrics.format_headline`, not a hard-coded ROD fraction.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_results_primary_metric.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from scat.main_gui import ResultsTab
from scat.pipeline import analyze_folder_service
from scat.main_gui import _results_dict_from_output
from pathlib import Path


def test_hero_reflects_primary_metric(synth_dir):
    app = QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), primary_metric="n_total")
    d = _results_dict_from_output(Path(res.output_dir))
    tab = ResultsTab()
    tab.load_results(d)
    assert "/ image" in tab.hero_value.text()          # deposits shown as a rate
    assert "DEPOSIT" in tab.hero_kicker.text().upper()  # kicker names the metric, not "ROD FRACTION"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_results_primary_metric.py -q`
Expected: FAIL (hero still shows "MEAN ROD FRACTION" / "8.3%").

- [ ] **Step 3: Implement**

In `ResultsTab.load_results`, replace the hard-coded ROD hero with the metric-driven one:

```python
from scat import metrics as _metrics

# inside load_results, where hero_kicker/hero_value/hero_sub are set:
pm = results.get("primary_metric", _metrics.DEFAULT_METRIC)
norm = results.get("normalization", _metrics.DEFAULT_NORMALIZATION)
meta = results.get("run_meta", {})   # {} until normalization metadata exists
m = _metrics.METRICS[_metrics.resolve_metric(pm)]
self.hero_kicker.setText(m.label.upper())
self.hero_value.setText(_metrics.format_headline(film_summary, pm, norm, meta))
# hero_sub keeps ±SD + "across N images"; SD is now of the primary metric's per-image values:
vals = _metrics.metric_values(film_summary, pm).dropna()
self.hero_sub.setText(f"±{vals.std():.1f}{m.unit}   ·   across {n} image{'s' if n != 1 else ''}")
```

- [ ] **Step 4: Run test to verify it passes + full suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_results_primary_metric.py -q`
Expected: PASS.
Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: all pass. (Update `test_gui_slimdown` if it asserted the literal "MEAN ROD FRACTION" hero.)

- [ ] **Step 5: Verify visually**

```bash
python /tmp/.../render_current.py   # or the standing ResultsTab render harness
```
Confirm the hero reads "TOTAL DEPOSITS / 16.0 deposits / image" for the default metric.

- [ ] **Step 6: Commit**

```bash
git add scat/main_gui.py tests/test_results_primary_metric.py
git commit -m "feat(gui): result-window hero reads the configurable primary metric (default deposits/image)"
```

---

## Self-review checklist (run before handoff)

- **Spec coverage (Phase 0–1):** metric registry + deposits=Normal+ROD (Task 1); normalization + rate
  headline (Task 2); metric-specific flip rule (Task 3); config defaults (Task 4); manifest persistence
  (Task 5); pipeline threading (Task 6); agent confirmation param (Task 7); results dict + deposits
  (Task 8); hero reads the metric (Task 9). Phases 2–4 (confidence UI, recomposition, report) are
  separate plans — out of scope here by design.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `resolve_metric` / `metric_values` / `format_headline` / `effective_normalization`
  / `metric_sensitivity_range` / `Metric.values` / `Metric.is_rate` / `Metric.unit` used identically in
  tasks 1–9 and in the consuming code.
- **Contract honesty:** no trust/confidence UI here (Plan 2); this plan only lands the *data* contract
  and the metric headline. Nothing in this plan renders a "reviewed" or "high-confidence" claim.
