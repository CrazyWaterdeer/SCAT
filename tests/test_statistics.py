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
