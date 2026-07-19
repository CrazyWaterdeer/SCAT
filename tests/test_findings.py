# tests/test_findings.py
from scat import findings


def _stats(key, *, test, p, sig):
    return {key: {"overall_test": test, "overall_p_value": p, "overall_significant": sig}}


def test_default_metric_maps_to_n_deposits_stats_key():
    # primary_metric total_deposits must look up stats key n_deposits (Normal+ROD,
    # artifact-EXCLUSIVE) — matching the metric value and the rest of the report.
    s = _stats("n_deposits", test="Kruskal-Wallis", p=0.008, sig=True)
    f = findings.compose_finding(stats=s, primary_metric="total_deposits", headline="40.8 deposits / image",
                                 n_images=30, n_groups=6, group_label="condition")
    assert "Total deposits showed a difference across condition groups" in f["sentence"]
    assert "Kruskal-Wallis" in f["sentence"] and "p = 0.008" in f["sentence"]
    assert f["test"].startswith("Kruskal-Wallis") and f["scope"] == "30 images · 6 groups"


def test_omnibus_significant_says_across_not_between():
    s = _stats("rod_fraction", test="Kruskal-Wallis", p=0.0004, sig=True)
    f = findings.compose_finding(stats=s, primary_metric="rod_fraction", headline="8.3%",
                                 n_images=30, n_groups=6, group_label="condition")
    assert "across condition groups" in f["sentence"] and "between" not in f["sentence"]
    assert "p < 0.001" in f["sentence"]              # tiny p formatting


def test_two_group_significant_uses_between_and_names_the_test():
    s = {"rod_fraction": {"test_name": "Mann-Whitney U", "p_value": 0.03, "significant": True}}
    f = findings.compose_finding(stats=s, primary_metric="rod_fraction", headline="8.3%",
                                 n_images=10, n_groups=2, group_label="group")
    assert "differed between the group groups" in f["sentence"] or "differed between" in f["sentence"]
    assert "Mann-Whitney U" in f["sentence"]


def test_nonsignificant_never_claims_equivalence():
    s = _stats("rod_fraction", test="Kruskal-Wallis", p=0.29, sig=False)
    f = findings.compose_finding(stats=s, primary_metric="rod_fraction", headline="8.3%",
                                 n_images=30, n_groups=6, group_label="condition")
    assert "no statistically detected difference" in f["sentence"].lower()
    assert "did not differ" not in f["sentence"].lower()


def test_ungrouped_is_descriptive():
    f = findings.compose_finding(stats=None, primary_metric="total_deposits",
                                 headline="40.8 deposits / image", n_images=30, n_groups=0, group_label=None)
    assert f["sentence"] == "Across 30 images, Total deposits averaged 40.8 deposits / image."
    assert f["test"] == "no group comparison" and f["scope"] == "30 images"


def test_missing_stats_key_degrades_to_descriptive():
    # grouped run but the primary metric's stats are absent -> descriptive, never a crash
    f = findings.compose_finding(stats={"mean_area": {"overall_test": "x", "overall_p_value": 0.1,
                                 "overall_significant": False}}, primary_metric="rod_fraction",
                                 headline="8.3%", n_images=10, n_groups=2, group_label="group")
    assert "averaged 8.3%" in f["sentence"] and f["test"] == "no group comparison"
