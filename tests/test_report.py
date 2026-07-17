"""HTML report generation test.

Regression guard: report.py failed to import on Python < 3.14 (missing Union
import) and carried a large dead ``_build_comprehensive_stats_html`` path.
This exercises the real HTML build end-to-end.
"""
from pathlib import Path

import pandas as pd

from scat.report import generate_report


def test_generate_html_report(tmp_path):
    film = pd.DataFrame({
        "filename": ["a.tif", "b.tif", "c.tif", "d.tif"],
        "group": ["Control", "Control", "Treatment", "Treatment"],
        "n_total": [10, 12, 8, 9],
        "n_normal": [8, 10, 3, 4],
        "n_rod": [2, 2, 5, 5],
        "n_artifact": [0, 0, 0, 0],
        "rod_fraction": [0.20, 0.167, 0.625, 0.556],
        "total_iod": [100.0, 110.0, 90.0, 95.0],
        "normal_mean_area": [50.0, 52.0, 55.0, 54.0],
        "rod_mean_area": [80.0, 82.0, 85.0, 83.0],
        "normal_mean_hue": [210.0, 211.0, 215.0, 214.0],
        "rod_mean_hue": [220.0, 221.0, 225.0, 224.0],
        "normal_mean_lightness": [0.50, 0.49, 0.40, 0.41],
        "rod_mean_lightness": [0.45, 0.44, 0.40, 0.39],
        "normal_mean_circularity": [0.80, 0.79, 0.75, 0.76],
        "rod_mean_circularity": [0.50, 0.49, 0.45, 0.46],
    })
    out = generate_report(film, output_dir=str(tmp_path), group_by="group", format="html")
    out = Path(out)
    assert out.exists() and out.name == "report.html"
    assert out.stat().st_size > 1000
    assert "SCAT" in out.read_text(encoding="utf-8")


def test_generate_html_report_with_deposits_and_groups(tmp_path):
    """Exercise the deposit-level histogram branches (area/IOD/pH/circularity) plus the
    group-comparison and statistical-appendix sections — the paths reshaped by the
    Stage-6 report refactor that the plain report test does not reach."""
    import numpy as np
    from scat.statistics import run_comprehensive_analysis

    rng = np.arange(12)
    film = pd.DataFrame({
        "filename": [f"img_{i:02d}.tif" for i in rng],
        "group": ["ctrl"] * 6 + ["treat"] * 6,
        "n_total": (10 + rng % 5).astype(int),
        "n_normal": (8 + rng % 3).astype(int),
        "n_rod": (2 + rng % 4).astype(int),
        "n_artifact": np.zeros(12, dtype=int),
        "rod_fraction": np.array([0.2] * 6 + [0.6] * 6) + (rng % 3) * 0.01,
        "total_iod": 100.0 + (rng % 7) * 5.0,
        "mean_area": 50.0 + (rng % 4) * 2.0,
        "mean_hue": 215.0 + (rng % 5),
        "mean_circularity": 0.70 - (rng % 3) * 0.01,
    })
    # deposit-level data drives the area/IOD/pH/circularity histograms
    dep_rows = []
    for i in range(12):
        for j in range(8):
            label = "rod" if (i + j) % 3 == 0 else ("artifact" if (i + j) % 7 == 0 else "normal")
            dep_rows.append({
                "filename": f"img_{i:02d}.tif", "label": label,
                "area_px": 40.0 + (i * 3 + j * 5) % 300,
                "iod": 20.0 + (i * 2 + j * 4) % 150,
                "mean_hue": (30.0 + i * 13 + j * 7) % 360,
                "circularity": ((i * 5 + j * 3) % 100) / 100.0,
            })
    deposits = pd.DataFrame(dep_rows)
    stats = run_comprehensive_analysis(film, deposits_df=deposits, group_column="group")
    metrics = stats.get("basic", {}).get("metrics") or stats

    out = Path(generate_report(film, output_dir=str(tmp_path), deposit_data=deposits,
                               statistical_results=metrics, group_by="group", format="html"))
    html = out.read_text(encoding="utf-8")
    assert out.exists() and out.stat().st_size > 5000
    # grouped run: the pooled distribution histograms are demoted to a per-group means
    # table (the pooled grand mean describes no single condition), so "Distributions" is
    # gone from the overview and the per-group table renders instead.
    assert "Distributions" not in html
    assert "Deposits / img" in html          # per-group means table header (unique to it)
    assert "Group Comparison" in html
    assert "Appendix" in html
    # grouped mode embeds the group-comparison boxplots (the pooled distribution histograms
    # are demoted to the per-group table) — sanity lower bound that figures still render
    assert html.count("data:image/png;base64,") >= 2
