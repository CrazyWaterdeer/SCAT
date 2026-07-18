"""Regression tests for the statistics bugs that were fixed."""
import numpy as np
import pandas as pd

from scat.statistics import StatisticalAnalyzer, PigmentationAnalyzer


def test_holm_correction_known_case():
    """Holm-Bonferroni step-down on a known counterexample.

    The old implementation used an inverse permutation + reversed cumulative
    minimum and returned [0.025, 0.025, 0.025, 0.4, 0.4] (wrongly flagging the
    p=0.2 test as significant). Correct output maps back to original order.
    """
    sa = StatisticalAnalyzer()
    p = [0.2, 0.01, 0.5, 0.04, 0.005]
    expected = [0.4, 0.04, 0.5, 0.12, 0.025]
    got = sa._correct_pvalues(p, "holm")
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, expected)), got


def test_holm_is_bounded_and_ge_raw():
    sa = StatisticalAnalyzer()
    p = [0.001, 0.002, 0.03, 0.9]
    got = sa._correct_pvalues(p, "holm")
    assert all(0.0 <= c <= 1.0 for c in got)
    # correction can only increase (or hold) each p-value
    assert all(c >= raw - 1e-12 for c, raw in zip(got, p))


def test_bonferroni():
    sa = StatisticalAnalyzer()
    assert sa._correct_pvalues([0.1, 0.2], "bonferroni") == [0.2, 0.4]


def test_pigment_density_handles_misaligned_nans():
    """iod and area NaNs on different rows must not crash or mis-pair.

    The old code called dropna() on each column independently and divided the
    two arrays element-wise, raising IndexError (or silently pairing the wrong
    deposits). The fix drops NaNs jointly so rows stay aligned.
    """
    df = pd.DataFrame({
        "label": ["normal", "normal", "rod", "rod"],
        "iod":    [10.0, np.nan, 30.0, 40.0],   # NaN on row 1
        "area_px": [5.0, 20.0, np.nan, 8.0],    # NaN on row 2 (different row)
    })
    res = PigmentationAnalyzer().analyze_deposit_pigmentation(df)
    assert "error" not in res
    # Valid aligned pairs: 10/5=2.0 and 40/8=5.0 -> mean 3.5
    assert abs(res["mean_pigment_density"] - 3.5) < 1e-9
    # normal label: 10/5=2.0 (row1 dropped) -> 2.0
    assert abs(res["normal_mean_pigment_density"] - 2.0) < 1e-9


def test_coefficient_of_variation_canonical():
    import numpy as np
    from scat.statistics import coefficient_of_variation as cv
    assert np.isnan(cv([5.0]))                    # CV of one sample is undefined (was 0)
    assert not np.isnan(cv([1.0, 2.0, np.nan]))   # NaN skipped -> computed over the rest
    assert abs(cv([10.0, 12.0, 8.0]) - float(np.std([10, 12, 8]) / 10 * 100)) < 1e-9
    assert np.isnan(cv([-1.0, 1.0]))              # mean 0 -> NaN
    assert np.isnan(cv([]))                       # empty -> NaN


def test_compare_group_values_dispatch():
    import numpy as np
    from scat.statistics import compare_group_values
    two = compare_group_values({"a": np.array([1., 2, 3, 4]), "b": np.array([3., 4, 5, 6])})
    assert two.get("group1_name") == "a" and two.get("group2_name") == "b"
    multi = compare_group_values({"a": np.array([1., 2, 3]), "b": np.array([2., 3, 4]),
                                  "c": np.array([5., 6, 7])})
    assert "overall_test" in multi or "pairwise_comparisons" in multi


def test_correct_pvalues_holm_restores_input_order():
    from scat.statistics import correct_pvalues
    # unsorted input; Holm must return values mapped back to the ORIGINAL positions
    out = correct_pvalues([0.04, 0.01, 0.03], "holm")
    # sort asc [0.01@1,0.03@2,0.04@0] * [3,2,1] = [0.03,0.06,0.04]; cummax=[0.03,0.06,0.06]
    assert abs(out[1] - 0.03) < 1e-9 and abs(out[0] - 0.06) < 1e-9 and abs(out[2] - 0.06) < 1e-9


def test_correct_pvalues_bonferroni_and_passthrough():
    from scat.statistics import correct_pvalues
    assert correct_pvalues([0.02, 0.5], "bonferroni") == [0.04, 1.0]
    assert correct_pvalues([0.02, 0.5], "none") == [0.02, 0.5]


def test_analyzer_delegates_to_module_correct_pvalues():
    from scat.statistics import correct_pvalues, StatisticalAnalyzer
    p = [0.04, 0.01, 0.03, 0.2, 0.005]
    assert StatisticalAnalyzer()._correct_pvalues(p, "holm") == correct_pvalues(p, "holm")


def test_visualization_uses_canonical_correct_pvalues():
    # the drifted Visualizer._correct_pvalues copy is gone; viz delegates to statistics
    from scat import visualization
    assert not hasattr(visualization.Visualizer, "_correct_pvalues")


def test_compare_metric_between_groups_preserves_output_shapes():
    """The shared body (used by the pigmentation/size/density/morphology wrappers on 4 different
    classes) must reproduce each metric's exact output — median only when include_median=True."""
    import pandas as pd
    from scat.statistics import _compare_metric_between_groups as cmp
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B", "A", "B"],
        "total_iod": [1.0, 2, 3, 4, 1.5, 3.5],
        "normal_mean_area": [10., 11, 20, 21, 12, 22],
    })
    with_med = cmp(df, "group", "total_iod", 0.05, include_median=True)         # pigmentation/density
    no_med = cmp(df, "group", "normal_mean_area", 0.05, include_median=False)   # size/morphology
    for r, metric in [(with_med, "total_iod"), (no_med, "normal_mean_area")]:
        assert set(r) == {"metric", "group_statistics", "n_groups", "comparison"}
        assert r["metric"] == metric and r["n_groups"] == 2
    assert list(with_med["group_statistics"]["A"].keys()) == ["n", "mean", "std", "median", "cv"]
    assert list(no_med["group_statistics"]["A"].keys()) == ["n", "mean", "std", "cv"]


def test_compare_metric_error_paths_preserved():
    import pandas as pd
    from scat.statistics import _compare_metric_between_groups as cmp
    df = pd.DataFrame({"group": ["A", "B"], "total_iod": [1.0, 2.0]})
    assert "column not found" in cmp(df, "group", "nope", 0.05, include_median=False)["error"]
    assert "column not found" in cmp(df, "nogroup", "total_iod", 0.05, include_median=True)["error"]
    one = pd.DataFrame({"group": ["A", "A"], "total_iod": [1.0, 2.0]})
    assert "at least 2 groups" in cmp(one, "group", "total_iod", 0.05, include_median=True)["error"]


def test_domain_analyzer_wrappers_delegate():
    """Each public compare_* wrapper (on its own analyzer class) still returns the right shape."""
    import pandas as pd
    from scat.statistics import (PigmentationAnalyzer, SizeDistributionAnalyzer,
                                  DensityAnalyzer, MorphologyAnalyzer)
    # n>=3 per group so the significance test participates (2.0 raised the gate from n>=2)
    df = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"], "total_iod": [1., 2, 3, 4, 5, 6],
                       "normal_mean_area": [10., 11, 12, 20, 21, 22], "n_total": [5, 6, 7, 8, 9, 10],
                       "normal_mean_circularity": [0.8, 0.82, 0.79, 0.6, 0.62, 0.61]})
    assert PigmentationAnalyzer().compare_pigmentation_between_groups(df, "group")["n_groups"] == 2
    assert SizeDistributionAnalyzer().compare_size_between_groups(df, "group")["n_groups"] == 2
    assert DensityAnalyzer().compare_density_between_groups(df, "group")["n_groups"] == 2
    assert MorphologyAnalyzer().compare_morphology_between_groups(df, "group")["n_groups"] == 2
