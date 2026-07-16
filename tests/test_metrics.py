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
