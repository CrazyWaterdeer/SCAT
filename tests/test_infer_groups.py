from scat.grouping_util import infer_groups_from_folder


def _touch(d, names):
    for n in names:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_arbitrary_conditions_via_replicate_prefix(tmp_path):
    # Three genotypes named arbitrarily, <condition>_<replicate>.
    _touch(tmp_path, ["geno1_1.tif", "geno1_2.tif", "geno2_1.tif", "geno2_2.tif",
                      "dnc_1.tif", "dnc_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "filename_prefix" and r["confidence"] == "high"
    assert set(r["groups"]) == {"geno1", "geno2", "dnc"}
    assert r["mapping"]["dnc_1.tif"] == "dnc"


def test_dose_conditions(tmp_path):
    _touch(tmp_path, ["10uM_1.tif", "10uM_2.tif", "20uM_1.tif", "20uM_2.tif", "50uM_1.tif", "50uM_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"10uM", "20uM", "50uM"}


def test_timepoints_with_uneven_replicates(tmp_path):
    _touch(tmp_path, ["0h_1.tif", "0h_2.tif", "6h_1.tif", "24h_1.tif", "24h_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"0h", "6h", "24h"}
    assert r["confidence"] == "medium"  # 6h has a single replicate


def test_control_treated_still_works(tmp_path):
    _touch(tmp_path, ["control_01.tif", "control_02.tif", "treated_01.tif", "treated_02.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"control", "treated"}


def test_subfolder_arbitrary_names(tmp_path):
    _touch(tmp_path, ["rut2080/a.tif", "rut2080/b.tif", "dnc1/c.tif", "wt/d.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "subfolder"
    assert set(r["groups"]) == {"rut2080", "dnc1", "wt"}


def test_positional_factor(tmp_path):
    # condition token is not last, no trailing replicate to strip cleanly
    _touch(tmp_path, ["exp_wt_a.tif", "exp_wt_b.tif", "exp_mut_a.tif", "exp_mut_b.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"wt", "mut"}


def test_single_cohort_fallback(tmp_path):
    _touch(tmp_path, ["sampleA.tif", "sampleB.tif", "sampleC.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "single_cohort" and r["groups"] == ["all"]


def test_duplicate_basename_across_subfolders_flagged(tmp_path):
    _touch(tmp_path, ["Control/x.tif", "Treatment/x.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "single_cohort" and r["confidence"] == "low"
    assert any("duplicate" in w for w in r["warnings"])
