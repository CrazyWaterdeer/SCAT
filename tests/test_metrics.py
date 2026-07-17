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
