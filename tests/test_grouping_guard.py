import pytest
from scat.grouping_util import duplicate_basenames, build_group_metadata
from scat.pipeline import scan_folder_service, analyze_folder_service


def test_build_group_metadata():
    df, group_by = build_group_metadata({"a.tif": "wt", "b.tif": "dnc1", "c.tif": None})
    assert group_by == ["group"]
    assert set(df.columns) == {"filename", "group"}
    assert df[df.filename == "c.tif"].iloc[0]["group"] == "ungrouped"


def test_duplicate_basenames_detected():
    assert duplicate_basenames(["A/x.tif", "B/x.tif", "C/y.tif"]) == ["x.tif"]
    assert duplicate_basenames(["A/x.tif", "B/X.TIF"]) == ["x.tif"]  # case-insensitive
    assert duplicate_basenames(["a.tif", "b.tif"]) == []


def test_scan_folder_returns_all_filenames(synth_dir):
    r = scan_folder_service(str(synth_dir))
    assert r["n_images"] == len(r["files"]) > 0
    assert all("filename" in f and "subfolder" in f for f in r["files"])


def test_analyze_folder_refuses_grouping_on_duplicate_basenames(tmp_path):
    (tmp_path / "Ctrl").mkdir(); (tmp_path / "Treat").mkdir()
    from PIL import Image
    import numpy as np
    for sub in ("Ctrl", "Treat"):
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(tmp_path / sub / "x.tif")
    with pytest.raises(ValueError, match="duplicate basenames"):
        analyze_folder_service(str(tmp_path), groups={"x.tif": "Ctrl"}, annotate=False,
                               output_dir=str(tmp_path / "out"))
