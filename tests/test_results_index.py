"""On-disk results ledger (scat/results_index.py) — discovery + two-tier analysed-vs-pending."""
import json
import shutil
from pathlib import Path

import pandas as pd

from scat import manifest
from scat import results_index as ri
from scat.pipeline import analyze_folder_service, scan_folder_service


def _imgs(folder, names):
    out = []
    for i, n in enumerate(names):
        p = folder / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * (10 + i))   # distinct sizes -> meaningful fingerprint
        out.append(p)
    return out


def _fp(paths):
    return manifest.dataset_fingerprint([str(p) for p in paths])["sha256"]


def _write_run(results_dir, *, dataset_path, basenames, sha256,
               created_at="2026-07-15T10:00:00+00:00", grouping=None, model=None,
               detection=None, write_manifest=True, dup_rows=False, n_images=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = list(basenames)
    if dup_rows and rows:
        rows = rows + [rows[0]]
    pd.DataFrame({"filename": rows, "n_normal": [1] * len(rows)}).to_csv(
        results_dir / "image_summary.csv", index=False)
    if write_manifest:
        m = {"schema": "scat.run_manifest/1", "created_at": created_at,
             "dataset": {"path": str(dataset_path), "sha256": sha256,
                         "n_images": n_images if n_images is not None else len(basenames)},
             "model": model or {"type": "rf", "path": None, "circularity": 0.6},
             "detection": detection or {"min_area": 20}, "grouping": grouping, "warnings": []}
        (results_dir / "run_manifest.json").write_text(json.dumps(m))


def test_find_analyses_records_skips_and_order(tmp_path):
    exp = tmp_path / "exp"
    imgs = _imgs(exp, ["a.tif", "b.tif"])
    _write_run(tmp_path / "results_1", dataset_path=exp, basenames=["a.tif", "b.tif"],
               sha256=_fp(imgs), created_at="2026-07-15T10:00:00+00:00")
    _write_run(tmp_path / "results_2", dataset_path=exp, basenames=["a.tif"],
               sha256="deadbeef", created_at="2026-07-15T11:00:00+00:00")
    _write_run(tmp_path / "results_partial", dataset_path=exp, basenames=["a.tif"],
               sha256=None, write_manifest=False)                       # csv, no manifest -> partial
    rb = tmp_path / "results_bad"; rb.mkdir()
    (rb / "run_manifest.json").write_text("{not json")                  # broken manifest, no csv -> skip

    records, skipped = ri.find_analyses_with_skips([tmp_path])
    names = [Path(r.results_dir).name for r in records]
    assert {"results_1", "results_2", "results_partial"} <= set(names)
    assert any(Path(s["dir"]).name == "results_bad" for s in skipped)
    partial = next(r for r in records if Path(r.results_dir).name == "results_partial")
    assert partial.status == "partial" and partial.dataset_path is None
    idx = {Path(r.results_dir).name: i for i, r in enumerate(records)}
    assert idx["results_1"] < idx["results_2"]                          # ascending by created_at


def test_analysis_status_complete(tmp_path):
    exp = tmp_path / "exp"
    imgs = _imgs(exp, ["a.tif", "b.tif", "c.tif"])
    _write_run(tmp_path / "results_full", dataset_path=exp,
               basenames=["a.tif", "b.tif", "c.tif"], sha256=_fp(imgs))
    st = ri.analysis_status(str(exp))
    assert st["status"] == "complete" and st["verified"] and st["n_pending"] == 0
    assert Path(st["results_dir"]).name == "results_full"


def test_analysis_status_partial(tmp_path):
    exp = tmp_path / "exp"
    imgs = _imgs(exp, ["a.tif", "b.tif", "c.tif"])
    _write_run(tmp_path / "results_a", dataset_path=exp, basenames=["a.tif"], sha256=_fp([imgs[0]]))
    st = ri.analysis_status(str(exp))
    assert st["status"] == "partial" and st["verified"] is False
    assert st["n_analyzed"] == 1 and st["n_pending"] == 2
    assert set(st["pending"]) == {"b.tif", "c.tif"}


def test_analysis_status_ambiguous_duplicate_current_basenames(tmp_path):
    exp = tmp_path / "exp"
    _imgs(exp / "s1", ["x.tif"])
    _imgs(exp / "s2", ["x.tif"])                                        # same basename, two subfolders
    st = ri.analysis_status(str(exp))
    assert st["status"] == "ambiguous"


def test_analysis_status_ambiguous_duplicate_csv_rows(tmp_path):
    exp = tmp_path / "exp"
    _imgs(exp, ["a.tif", "b.tif"])
    _write_run(tmp_path / "results_dup", dataset_path=exp, basenames=["a.tif", "b.tif"],
               sha256="nomatch", dup_rows=True)
    st = ri.analysis_status(str(exp))
    assert st["status"] == "ambiguous"


def test_analysis_status_none(tmp_path):
    exp = tmp_path / "exp"
    _imgs(exp, ["a.tif", "b.tif", "c.tif"])
    st = ri.analysis_status(str(exp))
    assert st["status"] == "none" and st["n_pending"] == 3


def test_analysis_status_image_paths_honored(tmp_path):
    exp = tmp_path / "exp"
    imgs = _imgs(exp, ["a.tif", "b.tif", "c.tif"])
    _write_run(tmp_path / "results_a", dataset_path=exp, basenames=["a.tif"], sha256="x")
    st = ri.analysis_status(str(exp), image_paths=[str(imgs[0]), str(imgs[1])])
    assert st["n_current"] == 2 and st["n_pending"] == 1 and set(st["pending"]) == {"b.tif"}


def test_analysis_status_resolved_path_equality(tmp_path):
    a_exp = tmp_path / "a" / "exp"
    b_exp = tmp_path / "b" / "exp"
    ia = _imgs(a_exp, ["a.tif", "b.tif"])
    _imgs(b_exp, ["a.tif", "b.tif"])                                    # different folder, same basenames
    _write_run(tmp_path / "a" / "results_full", dataset_path=a_exp,
               basenames=["a.tif", "b.tif"], sha256=_fp(ia))
    assert ri.analysis_status(str(a_exp))["status"] == "complete"
    assert ri.analysis_status(str(b_exp))["status"] == "none"          # must NOT match a/'s run


def test_analysis_status_single_file(tmp_path):
    folder = tmp_path / "data"
    imgs = _imgs(folder, ["only.tif"])
    _write_run(folder / "results_one", dataset_path=imgs[0], basenames=["only.tif"],
               sha256=_fp([imgs[0]]))
    st = ri.analysis_status(str(imgs[0]))                              # single-file input
    assert st["status"] == "complete"


def test_norm_roots(tmp_path):
    roots = ri._norm_roots(["", "does_not_exist_xyz", str(tmp_path), str(tmp_path)])
    assert roots == [tmp_path.resolve()]


def test_scan_folder_already_analyzed_integration(synth_dir, tmp_path):
    exp = tmp_path / "exp"; exp.mkdir()
    for f in sorted(synth_dir.glob("*.tif"))[:3]:
        shutil.copy(f, exp / f.name)
    analyze_folder_service(str(exp), output_dir=str(tmp_path / "results_run"), annotate=False)
    aa = scan_folder_service(str(exp)).get("already_analyzed")
    assert aa and aa["status"] == "complete" and aa["n_pending"] == 0
