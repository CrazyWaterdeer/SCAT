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


from pathlib import Path


def _report(tmp_path, synth_dir, grouped=True):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    kw = {}
    if grouped:
        gm = {f"ctrl_{i}.tif": "Control" for i in range(3)}
        gm.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
        kw["groups"] = gm
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False, **kw)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd)) if grouped else None
    generate_report_service(str(rd), statistical_results=stats, group_by=("group" if grouped else None))
    return (rd / "report.html").read_text()


def test_grouped_overview_is_per_group_table_no_pooled_biology(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, grouped=True)
    i = html.lower().find("population overview")
    pop = html[i:html.find("</div>\n    </div>", i) + 20] if i != -1 else ""
    assert "Control" in pop and "Treatment" in pop            # per-group rows
    assert "Total Images" in pop and "Total Deposits" in pop  # scope counts kept
    assert "stat-card rod" not in pop                          # no pooled ROD hero card
    # pooled histograms gone from the grouped overview. (The plan's "Distribution of
    # Deposit Counts" is a matplotlib title baked into a base64 PNG — never literal HTML;
    # the real marker for the pooled histogram body is the "Distributions" <h3>.)
    assert "Distributions" not in pop


def test_ungrouped_overview_keeps_pooled(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, grouped=False)
    i = html.lower().find("population overview")
    pop = html[i:i + 12000]
    assert "stat-card rod" in pop and "Distributions" in pop
