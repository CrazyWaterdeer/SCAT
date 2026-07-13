from scat.grouping_util import infer_groups_from_folder


def _touch(d, names):
    for n in names:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_vocab_grouping(tmp_path):
    _touch(tmp_path, ["control_01.tif", "control_02.tif", "treated_01.tif", "treated_02.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "filename_vocab"
    assert set(r["groups"]) == {"control", "treated"}
    assert r["mapping"]["control_01.tif"] == "control"


def test_subfolder_grouping(tmp_path):
    _touch(tmp_path, ["Control/a.tif", "Control/b.tif", "Treatment/c.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "subfolder"
    assert r["mapping"]["a.tif"] == "Control"


def test_single_cohort_fallback(tmp_path):
    _touch(tmp_path, ["img_1.tif", "img_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "single_cohort" and r["groups"] == ["all"]


def test_duplicate_basename_across_subfolders_flagged(tmp_path):
    # C2/C10: duplicate basenames across subfolders -> refuse subfolder grouping.
    _touch(tmp_path, ["Control/x.tif", "Treatment/x.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "single_cohort" and r["confidence"] == "low"
    assert any("duplicate" in w for w in r["warnings"])
