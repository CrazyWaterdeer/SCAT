"""Regression tests for the Stage-5 group-comparison correctness fixes (2026-07-18).

Two clustered defect classes:
  1. Combined mean_area/mean_hue/mean_circularity stat keys never matched the stats dict (which
     keys them per-class as normal_*), so those omnibus p-values were silently dropped from the
     boxplot captions AND the statistics appendix — including whenever one was the primary endpoint.
  2. The 'ungrouped' sentinel leaked into the group comparison: it slipped past the stats skip-gate
     (fabricating p=1.0/0.0 verdicts on a 1-real-group run), rendered as an extra boxplot box, and
     inflated the finding lede's group count.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scat.report import ReportGenerator, generate_report
from scat.statistics import run_comprehensive_analysis


def _real_film(groups):
    """A film shaped like image_summary.csv — per-class normal_*/rod_* columns, one row per label
    in `groups`. Row values are a deterministic function of the row index, so two films that share a
    prefix of indices share those rows' values exactly."""
    n = len(groups)
    rng = np.arange(n)
    return pd.DataFrame({
        "filename": [f"img_{i:02d}.tif" for i in rng],
        "group": list(groups),
        "n_total": (10 + rng % 5).astype(int),
        "n_normal": (8 + rng % 3).astype(int),
        "n_rod": (2 + rng % 4).astype(int),
        "n_artifact": np.zeros(n, dtype=int),
        "rod_fraction": 0.2 + (rng % 4) * 0.05,
        "total_iod": 100.0 + (rng % 7) * 5.0,
        "normal_mean_area": 50.0 + (rng % 4) * 2.0,
        "rod_mean_area": 80.0 + (rng % 3) * 2.0,
        "normal_mean_hue": 210.0 + (rng % 5),
        "rod_mean_hue": 220.0 + (rng % 5),
        "normal_mean_lightness": 0.50 - (rng % 3) * 0.02,
        "rod_mean_lightness": 0.45 - (rng % 3) * 0.02,
        "normal_mean_circularity": 0.80 - (rng % 3) * 0.01,
        "rod_mean_circularity": 0.50 - (rng % 3) * 0.01,
    })


# ---------------------------------------------------------------------------
# Bug 1 — combined mean_area/hue/circularity fall back to the normal_* stats key
# ---------------------------------------------------------------------------
def test_resolve_stat_key_falls_back_to_normal(tmp_path):
    rg = ReportGenerator(str(tmp_path))
    stats = {"n_total": {}, "normal_mean_hue": {}, "normal_mean_area": {}}
    assert rg._resolve_stat_key("n_total", stats) == "n_total"           # present -> unchanged
    assert rg._resolve_stat_key("mean_hue", stats) == "normal_mean_hue"  # combined absent -> fallback
    assert rg._resolve_stat_key("mean_area", stats) == "normal_mean_area"
    assert rg._resolve_stat_key("mean_circularity", stats) == "mean_circularity"  # neither -> original


def test_report_renders_area_hue_circularity_verdicts(tmp_path):
    """On a real-shaped (normal_*) film, the area/hue/circularity omnibus verdicts must render in
    both the boxplot captions and the appendix — they were silently dropped before the fix."""
    film = _real_film(["ctrl"] * 6 + ["treat"] * 6)
    stats = run_comprehensive_analysis(film, group_column="group")
    metrics = stats.get("basic", {}).get("metrics") or stats
    # the stats really are keyed per-class (normal_*), not by the combined name
    assert "normal_mean_hue" in metrics and "mean_hue" not in metrics
    assert "normal_mean_area" in metrics and "normal_mean_circularity" in metrics

    html = Path(generate_report(film, output_dir=str(tmp_path), statistical_results=metrics,
                                group_by="group", format="html")).read_text(encoding="utf-8")
    # appendix <h3> entries for the three metrics now appear (numbered "N. <title>")
    assert "Mean Deposit Area</h3>" in html
    assert "pH Indicator (Hue)</h3>" in html
    assert "Mean Circularity</h3>" in html
    # every one of the six group metrics now carries an omnibus caption (was 3 before);
    # count the caption markup only, not the single CSS rule that also mentions the class
    assert html.count('<p class="omnibus-line">') == 6


# ---------------------------------------------------------------------------
# Bug 2 — stats skip-gate must exclude the 'ungrouped' sentinel
# ---------------------------------------------------------------------------
def test_stats_gate_skips_one_real_group_plus_ungrouped(tmp_path):
    from scat.pipeline import run_statistics_service
    from scat.artifacts import IMAGE_SUMMARY
    _real_film(["ctrl"] * 6 + ["ungrouped"] * 4).to_csv(tmp_path / IMAGE_SUMMARY, index=False)
    res = run_statistics_service(str(tmp_path), group_col="group")
    assert res.get("skipped") is True   # 1 real group -> no comparison, no fabricated p


def test_stats_gate_runs_two_real_groups_plus_ungrouped(tmp_path):
    from scat.pipeline import run_statistics_service
    from scat.artifacts import IMAGE_SUMMARY
    _real_film(["ctrl"] * 4 + ["treat"] * 4 + ["ungrouped"] * 3).to_csv(tmp_path / IMAGE_SUMMARY, index=False)
    res = run_statistics_service(str(tmp_path), group_col="group")
    assert not res.get("skipped")       # 2 real groups -> the comparison still runs


# ---------------------------------------------------------------------------
# Bug 3 — the group-comparison boxplot must not draw an 'ungrouped' box
# ---------------------------------------------------------------------------
def test_boxplot_ignores_ungrouped_rows(tmp_path):
    rg = ReportGenerator(str(tmp_path))
    clean = _real_film(["ctrl"] * 6 + ["treat"] * 6)
    mixed = _real_film(["ctrl"] * 6 + ["treat"] * 6 + ["ungrouped"] * 3)   # shares rows 0..11
    # identical real-group data -> identical boxplot; the ungrouped rows must contribute nothing
    assert rg._generate_metric_boxplot(mixed, "n_total", "group", "Count") == \
           rg._generate_metric_boxplot(clean, "n_total", "group", "Count")


def test_boxplot_empty_when_all_ungrouped(tmp_path):
    rg = ReportGenerator(str(tmp_path))
    film = _real_film(["ungrouped"] * 6)
    assert rg._generate_metric_boxplot(film, "n_total", "group", "Count") == ""


# ---------------------------------------------------------------------------
# Bug 4 — the finding lede's group count must exclude the sentinel
# ---------------------------------------------------------------------------
def test_finding_lede_group_count_excludes_ungrouped(tmp_path):
    film = _real_film(["ctrl"] * 6 + ["treat"] * 6 + ["ungrouped"] * 3)
    stats = run_comprehensive_analysis(film, group_column="group")
    metrics = stats.get("basic", {}).get("metrics") or stats
    html = Path(generate_report(film, output_dir=str(tmp_path), statistical_results=metrics,
                                group_by="group", format="html")).read_text(encoding="utf-8")
    assert "2 groups" in html         # ctrl + treat only
    assert "3 groups" not in html     # not counting 'ungrouped'
