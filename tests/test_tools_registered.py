import json
import shutil
from pathlib import Path

import pandas as pd

import scat.tools as tools
from scat.agent.registry import call_tool


def test_expected_tools_registered():
    names = {e.name for e in tools.iter_tools()}
    assert {"scan_folder", "analyze_folder", "run_statistics", "generate_report",
            "list_analyses"} <= names
    # grouping is the LLM's job now — there is no deterministic infer_groups tool.
    assert "infer_groups" not in names


def test_tool_specs_have_schemas():
    specs = {s["name"]: s for s in tools.tools_for_anthropic()}
    assert specs["analyze_folder"]["input_schema"]["properties"]["path"]["type"] == "string"


def test_new_optional_params_are_nullable_not_required():
    specs = {s["name"]: s for s in tools.tools_for_anthropic()}
    af = specs["analyze_folder"]["input_schema"]
    assert "image_paths" in af["properties"] and "image_paths" not in af.get("required", [])
    assert "path" in af.get("required", [])
    la = specs["list_analyses"]["input_schema"]
    assert "folder" in la["properties"] and "folder" not in la.get("required", [])


def _fake_run(results_dir, dataset_path, basenames):
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filename": basenames, "n_normal": [1] * len(basenames)}).to_csv(
        results_dir / "image_summary.csv", index=False)
    (results_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "scat.run_manifest/1", "created_at": "2026-07-15T10:00:00+00:00",
        "dataset": {"path": str(dataset_path), "sha256": "x", "n_images": len(basenames)},
        "model": {"type": "rf"}, "detection": {"min_area": 20},
        "grouping": {"column": "group", "mapping": {b: "g" for b in basenames}}, "warnings": []}))


def test_list_analyses_finds_sibling_results(tmp_path):
    exp = tmp_path / "exp"; exp.mkdir()
    (exp / "a.tif").write_bytes(b"x")
    _fake_run(tmp_path / "results_x", exp, ["a.tif"])          # sibling of exp (under exp.parent)
    out = call_tool("list_analyses", folder=str(exp))
    dirs = [Path(a["results_dir"]).name for a in out["analyses"]]
    assert "results_x" in dirs                                 # guards _search_roots incl. folder.parent


def test_analyze_folder_tool_accepts_image_paths(synth_dir, tmp_path):
    exp = tmp_path / "exp"; exp.mkdir()
    src = sorted(synth_dir.glob("*.tif"))[:2]
    for f in src:
        shutil.copy(f, exp / f.name)
    res = call_tool("analyze_folder", path=str(exp),
                    image_paths=[str(exp / src[0].name)], annotate=False)
    assert res["n_images"] == 1                                # analyzed only the one image, not both
