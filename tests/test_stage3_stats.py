"""Numeric regression tests for the SCAT 2.0 Stage-3 statistics changes
(Welch's t-test, df-weighted Cohen's d, n>=3 significance gate, artifact-exclusive
n_deposits comparison). These deliberately pin the new reported numbers."""
import numpy as np
import pandas as pd
import pytest

from scat.statistics.common import StatisticalAnalyzer, _compare_metric_between_groups


def test_cohens_d_uses_df_weighted_pooled_sample_sd():
    """[12] Cohen's d denominator is the df-weighted pooled SAMPLE sd (ddof=1), correct
    for unequal n — not the old equal-n mean of population variances."""
    a = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])   # n=8
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])                    # n=5 (unequal n)
    r = StatisticalAnalyzer().compare_two_groups(a, b)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    expected = (np.mean(a) - np.mean(b)) / pooled
    assert abs(r["cohens_d"] - expected) < 1e-9
    # the OLD equal-n formula would give a different value here (n1 != n2)
    old_pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
    assert abs(r["cohens_d"] - (np.mean(a) - np.mean(b)) / old_pooled) > 1e-6


def test_two_group_parametric_branch_is_welch_not_student_or_paired():
    """[30]/[31] The parametric two-group test is Welch's; Student's/paired labels are gone."""
    rng = np.random.RandomState(0)
    a = rng.normal(10.0, 1.0, 40)
    b = rng.normal(12.0, 3.0, 40)          # unequal variance
    r = StatisticalAnalyzer().compare_two_groups(a, b)
    assert r["test_name"] in ("Welch's t-test", "Mann-Whitney U")
    assert r["test_name"] not in ("Independent t-test", "Paired t-test", "Wilcoxon signed-rank")
    if r["test_name"] == "Welch's t-test":
        from scipy import stats as ss
        _, p = ss.ttest_ind(a, b, equal_var=False)
        assert abs(r["p_value"] - p) < 1e-9


def test_paired_parameter_removed():
    """[31] The dead, latent-buggy `paired` parameter is gone."""
    with pytest.raises(TypeError):
        StatisticalAnalyzer().compare_two_groups(np.arange(5.0), np.arange(5.0), paired=True)


def test_n2_group_is_descriptive_but_excluded_from_significance():
    """[28] A group with n==2 still appears in the descriptive group_statistics, but the
    significance test requires n>=3 per group (omnibus and pairwise see the same set)."""
    film = pd.DataFrame({
        "group": ["a", "a", "a", "b", "b"],            # a: n=3 (testable), b: n=2 (descriptive only)
        "total_iod": [10.0, 11.0, 12.0, 20.0, 21.0],
    })
    r = _compare_metric_between_groups(film, "group", "total_iod", 0.05, include_median=True)
    assert "a" in r["group_statistics"] and "b" in r["group_statistics"]   # both described
    # only 'a' qualifies for a test -> fewer than 2 testable groups
    assert "error" in r


def test_n_deposits_gets_an_artifact_exclusive_significance_test():
    """[8] run_comprehensive_analysis derives an in-memory n_deposits (Normal+ROD) column and
    tests it, so the Deposit Count comparison is artifact-exclusive."""
    from scat.statistics import run_comprehensive_analysis
    film = pd.DataFrame({
        "filename": [f"i{i}" for i in range(12)],
        "group": ["c"] * 6 + ["t"] * 6,
        "n_normal": [8, 9, 10, 7, 8, 9, 3, 4, 2, 3, 4, 3],
        "n_rod":    [1, 0, 1, 2, 1, 0, 5, 6, 5, 6, 5, 6],
        "n_artifact": [2] * 12,
        "n_total":  [11, 11, 13, 11, 11, 11, 10, 12, 9, 11, 11, 11],
        "rod_fraction": [0.1] * 6 + [0.6] * 6,
        "total_iod": [100.0 + i for i in range(12)],
    })
    res = run_comprehensive_analysis(film, group_column="group")
    assert "n_deposits" in res["basic"]["metrics"]
