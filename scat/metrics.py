"""The statistical contract as code (spec §2.1/§3.1): the primary-metric registry, normalization,
and headline formatting. Pure functions over a film_summary DataFrame — the single source of truth
both the result window and the report render from. No I/O."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

DEFAULT_THRESHOLD = 0.60          # review/triage threshold on per-deposit confidence (display only; not a classification input) — spec §2.1


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    values: Callable[[pd.DataFrame], pd.Series]
    is_rate: bool                 # True = a count normalization divides (deposits, IOD)
    unit: str = ""
    fmt: str = "{:.1f}"
    is_circular: bool = False      # True = an angular metric (hue); a min/max range is misleading


def _deposits(film: pd.DataFrame) -> pd.Series:
    # Deposits = Normal + ROD (artifacts are the reject class). Fall back to the legacy n_total
    # column only when the split columns are absent (old/synthetic result dirs).
    if "n_normal" in film.columns and "n_rod" in film.columns:
        return film["n_normal"].astype(float) + film["n_rod"].astype(float)
    return film["n_total"].astype(float)


def _col(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda film: film[name].astype(float)


def _col_or_normal(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """A per-image value column, falling back to the ``normal_<name>`` variant. image_summary.csv has
    no combined ``mean_hue``/``mean_circularity`` column — only ``normal_*``/``rod_*`` — so these
    metrics use the normal-deposit column, matching the report's group-comparison boxplots
    (_generate_all_group_comparisons falls back the same way)."""
    def f(film: pd.DataFrame) -> pd.Series:
        col = name if name in film.columns else f"normal_{name}"
        return film[col].astype(float)
    return f


METRICS: dict[str, Metric] = {
    "total_deposits":   Metric("total_deposits", "Total deposits", _deposits, True, "", "{:.1f}"),
    "rod_fraction":     Metric("rod_fraction", "ROD fraction", lambda f: f["rod_fraction"].astype(float) * 100, False, "%", "{:.1f}"),
    "mean_area":        Metric("mean_area", "Mean deposit area", _col_or_normal("mean_area"), False, " px²", "{:.1f}"),
    "mean_hue":         Metric("mean_hue", "pH indicator (hue)", _col_or_normal("mean_hue"), False, "°", "{:.1f}", is_circular=True),
    "total_iod":        Metric("total_iod", "Total pigment (IOD)", _col("total_iod"), True, "", "{:.0f}"),
    "mean_circularity": Metric("mean_circularity", "Mean circularity", _col_or_normal("mean_circularity"), False, "", "{:.3f}"),
}

DEFAULT_METRIC = "total_deposits"

# primary-metric (registry key) -> statistics results key. Single source of truth for the
# report + findings (which import these). The stats module compares split-by-class metrics, so
# area/hue/circularity have no COMBINED comparison key — fall back to the normal_* comparison
# (matching the metric value column and the group-comparison boxplots).
# total_deposits → n_deposits (Normal+ROD, artifact-exclusive), matching the metric VALUE
# (_deposits = n_normal+n_rod) and the report's artifact-exclusive Deposit Count. The stats
# module derives the in-memory n_deposits column (comprehensive.run_comprehensive_analysis).
STATS_KEY = {
    "total_deposits": "n_deposits", "rod_fraction": "rod_fraction", "mean_area": "mean_area",
    "mean_hue": "mean_hue", "total_iod": "total_iod", "mean_circularity": "mean_circularity",
}
STATS_KEY_FALLBACK = {
    "mean_area": "normal_mean_area", "mean_hue": "normal_mean_hue",
    "mean_circularity": "normal_mean_circularity",
}


def resolve_metric(key: str | None) -> str:
    return key if key in METRICS else DEFAULT_METRIC


def metric_values(film: pd.DataFrame, key: str) -> pd.Series:
    return METRICS[resolve_metric(key)].values(film)


NORMALIZATIONS = ("per_image", "per_fly", "per_area", "per_time")
DEFAULT_NORMALIZATION = "per_image"

# The count/abundance columns that scale with the number of flies in a vial, so a raw total is
# meaningless to compare across vials with different fly counts — these are divided by n_flies for
# per-fly normalization. Fractions (rod_fraction) and per-deposit means (area/hue/circularity/
# mean_iod) are intensive quantities and are NEVER divided. n_deposits is the derived Normal+ROD
# count. (This is the same set run_all_tests compares; keeping them consistent keeps the whole report
# per-fly or all-total, never mixed.)
COUNT_SUM_COLUMNS = ("n_deposits", "n_total", "n_normal", "n_rod",
                     "total_iod", "normal_total_iod", "rod_total_iod")

_PREPARED_ATTR = "scat_per_fly"   # DataFrame.attrs marker so fly_normalize() is idempotent


def has_fly_counts(film: pd.DataFrame) -> bool:
    """True only when EVERY image has a valid (>0) n_flies — a partial column can't silently mix
    per-fly and total values in one comparison, so partial coverage falls back to totals."""
    if "n_flies" not in getattr(film, "columns", []) or len(film) == 0:
        return False
    import numpy as np
    nf = pd.to_numeric(film["n_flies"], errors="coerce")
    return bool((np.isfinite(nf) & (nf > 0)).all())   # every image needs a finite, positive count


def fly_normalize(film: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Return (film2, per_fly). Derives the shared artifact-exclusive n_deposits (Normal+ROD), and —
    when every image carries a valid n_flies — divides the count/sum columns (COUNT_SUM_COLUMNS) by
    n_flies so all downstream stats/plots/headline are PER FLY. Otherwise returns totals (per_fly=
    False, the fallback). Idempotent: a film already normalized here is returned unchanged.

    The on-disk image_summary.csv keeps RAW totals + an n_flies column; per-fly is computed in memory
    here at every entry point, so re-report/re-render read raw and re-derive consistently."""
    if getattr(film, "attrs", {}).get(_PREPARED_ATTR) is not None:
        return film, film.attrs[_PREPARED_ATTR]
    f = film
    if {"n_normal", "n_rod"} <= set(f.columns) and "n_deposits" not in f.columns:
        f = f.copy()
        f["n_deposits"] = f["n_normal"].astype(float) + f["n_rod"].astype(float)
    per_fly = has_fly_counts(f)
    if per_fly:
        if f is film:
            f = f.copy()
        nf = pd.to_numeric(f["n_flies"], errors="coerce")
        for col in COUNT_SUM_COLUMNS:
            if col in f.columns:
                f[col] = f[col].astype(float) / nf
    if f is film:            # nothing was copied (no derivation, no per-fly) — copy so attrs is ours
        f = f.copy()
    f.attrs[_PREPARED_ATTR] = per_fly
    return f, per_fly


def format_headline(film: pd.DataFrame, key: str, per_fly: bool = False) -> str:
    """Primary-metric headline. Rate/count metrics show a per-unit rate (per fly when per_fly else per
    image), never a pooled total (spec §2.1). Fraction/mean metrics show a per-image mean. Pass a film
    already run through fly_normalize() so count columns are in the right (per-fly or total) units."""
    m = METRICS[resolve_metric(key)]
    vals = m.values(film).dropna()
    if len(vals) == 0:
        return "—"
    if m.is_rate:
        n = len(film)
        rate = float(vals.sum()) / n if n else 0.0
        unit = "fly" if per_fly else "image"
        noun = m.label.lower().replace("total ", "")  # "Total deposits" -> "deposits"
        return f"{m.fmt.format(rate)} {noun} / {unit}"
    return f"{m.fmt.format(float(vals.mean()))}{m.unit}"


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
