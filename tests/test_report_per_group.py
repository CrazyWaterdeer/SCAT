# tests/test_report_per_group.py
import pandas as pd
from scat.report import ReportGenerator


def _film():
    return pd.DataFrame({
        "filename": list("abcd"),
        "group": ["Ctrl", "Ctrl", "Treat", "ungrouped"],   # a stray ungrouped row
        "n_normal": [8, 12, 4, 5], "n_rod": [2, 0, 6, 1], "n_artifact": [1, 1, 1, 1],
        "rod_fraction": [0.2, 0.0, 0.6, 0.1], "mean_area": [80.0, 90.0, 70.0, 60.0],
        "total_iod": [1000.0, 2000.0, 500.0, 700.0],
        "normal_mean_hue": [160.0, 170.0, 150.0, 155.0],
        "normal_mean_circularity": [0.8, 0.9, 0.7, 0.75],
    })


def test_effective_groups_excludes_ungrouped():
    g = ReportGenerator._effective_groups(_film(), "group")
    assert g == ["Ctrl", "Treat"]        # 'ungrouped' + order preserved (no sort)


def test_per_group_table_has_real_columns_and_values(tmp_path):
    rg = ReportGenerator(tmp_path)
    html = rg._html_per_group_table(_film(), "group")
    assert "Ctrl" in html and "Treat" in html and "ungrouped" not in html
    assert "ROD" in html and "Deposits" in html          # metric column headers
    # Ctrl deposits/img mean = mean(10, 12) = 11.0
    assert "11.0" in html
    # a pH column exists (from normal_mean_hue) — pH is present, not silently dropped
    assert "pH" in html
