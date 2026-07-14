"""Combining compatible results dirs (scat/combine.py) — strict merge with refuse-on-mismatch."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scat.combine import combine_results_service
from scat.pipeline import analyze_folder_service, run_statistics_service


def _write_dir(d, *, dataset_path, summary_rows, model=None, detection=None,
               grouping_col="group", deposits=None):
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df.to_csv(d / "image_summary.csv", index=False)
    if deposits is not None:
        pd.DataFrame(deposits).to_csv(d / "all_deposits.csv", index=False)
    mapping = (dict(zip(df["filename"].astype(str), df[grouping_col].astype(str)))
               if grouping_col and grouping_col in df.columns else None)
    m = {"schema": "scat.run_manifest/1", "created_at": "2026-07-15T10:00:00+00:00",
         "dataset": {"path": str(dataset_path), "sha256": "x", "n_images": len(df)},
         "model": model or {"type": "rf", "path": None, "circularity": 0.6},
         "detection": detection or {"min_area": 20, "max_area": 10000},
         "grouping": {"column": grouping_col, "mapping": mapping} if grouping_col else None,
         "warnings": []}
    (d / "run_manifest.json").write_text(json.dumps(m))


def test_combine_disjoint_ok_and_discoverable(tmp_path):
    exp = tmp_path / "exp"
    a, b = tmp_path / "resA", tmp_path / "resB"
    _write_dir(a, dataset_path=exp, summary_rows=[{"filename": "a.tif", "group": "C", "n_total": 3}])
    _write_dir(b, dataset_path=exp, summary_rows=[{"filename": "b.tif", "group": "T", "n_total": 5}])
    res = combine_results_service([str(a), str(b)], output_dir=str(tmp_path / "merged"))
    assert res["n_images"] == 2 and set(res["groups"]) == {"C", "T"}
    merged = pd.read_csv(tmp_path / "merged" / "image_summary.csv")
    assert set(merged["filename"]) == {"a.tif", "b.tif"}
    man = json.loads((tmp_path / "merged" / "run_manifest.json").read_text())
    assert set(man["combined_from"]) == {str(a.resolve()), str(b.resolve())}
    from scat.results_index import find_analyses
    assert "merged" in [Path(r.results_dir).name for r in find_analyses([tmp_path])]


def test_combine_refuses_different_dataset(tmp_path):
    a, b = tmp_path / "resA", tmp_path / "resB"
    _write_dir(a, dataset_path=tmp_path / "exp1", summary_rows=[{"filename": "a.tif", "group": "C"}])
    _write_dir(b, dataset_path=tmp_path / "exp2", summary_rows=[{"filename": "b.tif", "group": "T"}])
    with pytest.raises(ValueError, match="different datasets"):
        combine_results_service([str(a), str(b)])


def test_combine_refuses_different_model(tmp_path):
    exp = tmp_path / "exp"
    a, b = tmp_path / "resA", tmp_path / "resB"
    _write_dir(a, dataset_path=exp, summary_rows=[{"filename": "a.tif", "group": "C"}], model={"type": "rf"})
    _write_dir(b, dataset_path=exp, summary_rows=[{"filename": "b.tif", "group": "T"}], model={"type": "threshold"})
    with pytest.raises(ValueError, match="model"):
        combine_results_service([str(a), str(b)])


def test_combine_refuses_different_detection(tmp_path):
    exp = tmp_path / "exp"
    a, b = tmp_path / "resA", tmp_path / "resB"
    _write_dir(a, dataset_path=exp, summary_rows=[{"filename": "a.tif", "group": "C"}], detection={"min_area": 20})
    _write_dir(b, dataset_path=exp, summary_rows=[{"filename": "b.tif", "group": "T"}], detection={"min_area": 99})
    with pytest.raises(ValueError, match="detection"):
        combine_results_service([str(a), str(b)])


def test_combine_refuses_overlap_differs(tmp_path):
    exp = tmp_path / "exp"
    a, b = tmp_path / "resA", tmp_path / "resB"
    _write_dir(a, dataset_path=exp, summary_rows=[{"filename": "x.tif", "group": "C", "n_total": 3}])
    _write_dir(b, dataset_path=exp, summary_rows=[{"filename": "x.tif", "group": "C", "n_total": 9}])
    with pytest.raises(ValueError, match="differing image_summary"):
        combine_results_service([str(a), str(b)])


def test_combine_overlap_identical_kept_once(tmp_path):
    exp = tmp_path / "exp"
    a, b = tmp_path / "resA", tmp_path / "resB"
    row = {"filename": "x.tif", "group": "C", "n_total": 3}
    _write_dir(a, dataset_path=exp, summary_rows=[row, {"filename": "y.tif", "group": "C", "n_total": 4}])
    _write_dir(b, dataset_path=exp, summary_rows=[dict(row), {"filename": "z.tif", "group": "T", "n_total": 5}])
    res = combine_results_service([str(a), str(b)], output_dir=str(tmp_path / "merged"))
    merged = pd.read_csv(tmp_path / "merged" / "image_summary.csv")
    assert list(merged["filename"]).count("x.tif") == 1 and res["n_images"] == 3


def test_analyze_folder_resolves_relative_image_paths(synth_dir, tmp_path, monkeypatch):
    """A resume passing folder-relative image paths from a different cwd must still open them
    (they get resolved against `path`), not silently fail (Codex P2)."""
    exp = tmp_path / "exp"; exp.mkdir()
    f = sorted(synth_dir.glob("*.tif"))[0]
    shutil.copy(f, exp / f.name)
    monkeypatch.chdir(tmp_path)                                 # cwd != the dataset folder
    res = analyze_folder_service(str(exp), image_paths=[f.name],   # bare basename
                                 output_dir=str(tmp_path / "r"), annotate=False)
    assert res.n_images == 1 and res.n_failed == 0             # opened & analyzed, not a failed placeholder


def test_combine_recomputes_fingerprint_for_whole_folder(synth_dir, tmp_path):
    """When the merge reconstitutes the whole source folder, the combined dir gets a real
    fingerprint so a later scan recognizes it as fingerprint-verified complete (Codex P2)."""
    exp = tmp_path / "exp"; exp.mkdir()
    ctrl = sorted(synth_dir.glob("ctrl_*.tif"))[:2]
    treat = sorted(synth_dir.glob("treat_*.tif"))[:2]
    for f in ctrl + treat:
        shutil.copy(f, exp / f.name)                       # exactly the 4 we analyze == whole folder
    a = analyze_folder_service(str(exp), groups={ctrl[0].name: "C", treat[0].name: "T"},
                               image_paths=[str(exp / ctrl[0].name), str(exp / treat[0].name)],
                               output_dir=str(tmp_path / "resA"), annotate=False)
    b = analyze_folder_service(str(exp), groups={ctrl[1].name: "C", treat[1].name: "T"},
                               image_paths=[str(exp / ctrl[1].name), str(exp / treat[1].name)],
                               output_dir=str(tmp_path / "resB"), annotate=False)
    combine_results_service([a.output_dir, b.output_dir], output_dir=str(tmp_path / "merged"))
    man = json.loads((tmp_path / "merged" / "run_manifest.json").read_text())
    assert man["dataset"]["sha256"] is not None
    from scat.results_index import analysis_status
    assert analysis_status(str(exp))["status"] == "complete"   # verified-complete via the merged dir


def test_combine_grouped_stats_integration(synth_dir, tmp_path):
    exp = tmp_path / "exp"; exp.mkdir()
    for f in sorted(synth_dir.glob("*.tif")):
        shutil.copy(f, exp / f.name)
    ctrl = sorted(n.name for n in exp.glob("ctrl_*.tif"))
    treat = sorted(n.name for n in exp.glob("treat_*.tif"))
    ra = analyze_folder_service(str(exp), groups={ctrl[0]: "Control", treat[0]: "Treatment"},
                                image_paths=[str(exp / ctrl[0]), str(exp / treat[0])],
                                output_dir=str(tmp_path / "resA"), annotate=False)
    rb = analyze_folder_service(str(exp), groups={ctrl[1]: "Control", treat[1]: "Treatment"},
                                image_paths=[str(exp / ctrl[1]), str(exp / treat[1])],
                                output_dir=str(tmp_path / "resB"), annotate=False)
    merged = combine_results_service([ra.output_dir, rb.output_dir], output_dir=str(tmp_path / "merged"))
    assert merged["n_images"] == 4 and set(merged["groups"]) == {"Control", "Treatment"}
    stats = run_statistics_service(str(tmp_path / "merged"), group_col="group")
    assert not stats.get("skipped")
