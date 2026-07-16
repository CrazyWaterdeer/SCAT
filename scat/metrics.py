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
