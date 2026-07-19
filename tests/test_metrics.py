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
    assert metrics.format_headline(_film(), "total_deposits", per_fly=False) == "16.0 deposits / image"


def test_fraction_headline_is_mean_percent():
    assert metrics.format_headline(_film(), "rod_fraction", per_fly=False).startswith("8.3%")


def test_fly_normalize_no_counts_falls_back_to_totals():
    f2, per_fly = metrics.fly_normalize(_film())
    assert per_fly is False
    # counts unchanged (still totals); headline stays per image
    assert list(metrics.metric_values(f2, "total_deposits")) == [12, 20]
    assert "/ image" in metrics.format_headline(f2, "total_deposits", per_fly=per_fly)


def test_fly_normalize_divides_count_and_sum_columns_per_fly():
    film = _film().assign(n_flies=[3, 2],
                          normal_total_iod=[900.0, 1800.0], rod_total_iod=[100.0, 200.0])
    f2, per_fly = metrics.fly_normalize(film)
    assert per_fly is True
    # deposits per fly: image a 12/3=4, image b 20/2=10  -> headline mean = (4+10)/2 = 7.0
    assert list(metrics.metric_values(f2, "total_deposits")) == [4.0, 10.0]
    assert metrics.format_headline(f2, "total_deposits", per_fly=per_fly) == "7.0 deposits / fly"
    # total_iod per fly: 1000/3, 2000/2 ; sum-columns divided, fractions/means untouched
    assert f2["total_iod"].tolist() == [1000.0 / 3, 2000.0 / 2]
    assert f2["rod_fraction"].tolist() == _film()["rod_fraction"].tolist()   # fraction NOT divided
    assert f2["mean_area"].tolist() == [80.0, 90.0]                          # mean NOT divided


def test_fly_normalize_partial_counts_falls_back():
    film = _film().assign(n_flies=[3, 0])   # one image missing a valid count
    f2, per_fly = metrics.fly_normalize(film)
    assert per_fly is False                 # ALL images must have a count, else totals


def test_fly_normalize_is_idempotent():
    film = _film().assign(n_flies=[3, 2])
    f2, per_fly = metrics.fly_normalize(film)
    f3, per_fly2 = metrics.fly_normalize(f2)   # second pass must NOT divide again
    assert per_fly2 is True
    assert f3["total_iod"].tolist() == f2["total_iod"].tolist()


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
