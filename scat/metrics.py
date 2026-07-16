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
