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
