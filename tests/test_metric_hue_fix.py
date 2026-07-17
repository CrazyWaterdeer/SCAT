"""The mean_hue/mean_circularity (and mean_area) primary metrics must work on REAL image_summary
data, whose columns are split by class (normal_mean_hue/rod_mean_hue) with no combined column, and
whose stats compare the split-by-class keys. Regression for the latent Plan-1 registry bug."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pandas as pd
from scat import metrics, findings


def test_metric_value_falls_back_to_normal_column():
    film = pd.DataFrame({"normal_mean_hue": [160.0, 170.0],
                         "normal_mean_circularity": [0.8, 0.9]})   # no combined mean_hue/circularity
    assert list(metrics.metric_values(film, "mean_hue")) == [160.0, 170.0]
    assert list(metrics.metric_values(film, "mean_circularity")) == [0.8, 0.9]


def test_combined_column_preferred_when_present():
    film = pd.DataFrame({"mean_hue": [1.0, 2.0], "normal_mean_hue": [9.0, 9.0]})
    assert list(metrics.metric_values(film, "mean_hue")) == [1.0, 2.0]


def test_finding_falls_back_to_normal_stats_key():
    # real stats expose normal_mean_hue (not a combined mean_hue) — the finding must still get a verdict
    stats = {"normal_mean_hue": {"overall_test": "Kruskal-Wallis", "overall_p_value": 0.01,
                                 "overall_significant": True}}
    f = findings.compose_finding(stats=stats, primary_metric="mean_hue", headline="167°",
                                 n_images=30, n_groups=6, group_label="condition")
    assert "differ" in f["sentence"].lower() and "Kruskal-Wallis" in f["sentence"]
    assert f["test"] != "no group comparison"
