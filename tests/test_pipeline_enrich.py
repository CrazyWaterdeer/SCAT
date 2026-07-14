"""Phase-2 slimdown: analyze_folder_service superset enrichment.

Guards that the params added so the GUI can call the canonical service
(image_paths / spatial / sensitive_mode / parallel / save_json / progress_callback)
do NOT change the core CSV output on their defaults, and behave as specified.
"""
import json
from pathlib import Path

import pandas as pd

from scat.pipeline import analyze_folder_service


def _read_csvs(d):
    film = pd.read_csv(Path(d) / "image_summary.csv")
    dep = Path(d) / "all_deposits.csv"
    return film, (pd.read_csv(dep) if dep.exists() else None)


def test_enrich_defaults_are_csv_identical(synth_dir, tmp_path):
    """spatial/visualize/progress params must not perturb image_summary/all_deposits."""
    base = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "base"), annotate=False)
    rich = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "rich"),
                                  annotate=True, visualize=True, spatial=True,
                                  parallel=False, max_workers=1, save_json=True)
    bf, bd = _read_csvs(base.output_dir)
    rf, rd = _read_csvs(rich.output_dir)
    pd.testing.assert_frame_equal(bf, rf)
    if bd is not None and rd is not None:
        pd.testing.assert_frame_equal(bd, rd)
    assert base.n_normal == rich.n_normal and base.n_rod == rich.n_rod


def test_spatial_writes_sidecar(synth_dir, tmp_path):
    """spatial=True writes spatial_stats.json (a valid JSON object) as a Results-tab sidecar."""
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "sp"), spatial=True)
    sidecar = Path(res.output_dir) / "spatial_stats.json"
    assert sidecar.exists(), res.warnings
    assert isinstance(json.loads(sidecar.read_text()), dict)


def test_image_paths_subset(synth_dir, tmp_path):
    """An explicit image_paths list (GUI multi-file picker) analyses exactly that list."""
    one = sorted(Path(synth_dir).glob("ctrl_*.tif"))[:1]
    res = analyze_folder_service(str(synth_dir), image_paths=[str(one[0])],
                                 output_dir=str(tmp_path / "one"), annotate=False)
    assert res.n_images == 1
    film, _ = _read_csvs(res.output_dir)
    assert len(film) == 1 and film.iloc[0]["filename"] == one[0].name


def test_report_forwards_spatial_stats(synth_dir, tmp_path):
    """generate_report_service must load spatial_stats.json and render the Spatial Analysis
    section (regression: the GUI rewire otherwise drops spatial from the HTML report)."""
    from scat.pipeline import generate_report_service
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "rep"), annotate=False)
    # Write a deterministic sidecar (synthetic spatial can be degenerate/empty)
    (Path(res.output_dir) / "spatial_stats.json").write_text(json.dumps(
        {"mean_nnd": 12.3, "mean_clark_evans": 1.1, "mean_edge_fraction": 0.2,
         "n_clustered": 2, "n_random": 3, "n_dispersed": 1, "n_images": 6}))
    html_path = generate_report_service(res.output_dir, statistical_results=None, group_by=None)
    assert "Spatial Analysis" in Path(html_path).read_text()


def test_progress_callback_fires(synth_dir, tmp_path):
    """progress_callback(current, total) is threaded into analyze_batch for the GUI bar."""
    seen = []
    analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "pg"), annotate=False,
                           parallel=False, progress_callback=lambda c, t: seen.append((c, t)))
    assert seen, "progress_callback was never invoked"
    assert seen[-1][0] == seen[-1][1] == 6  # 6 synth images, ends at total
