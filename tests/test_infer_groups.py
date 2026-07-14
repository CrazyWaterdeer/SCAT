from scat.grouping_util import infer_groups_from_folder, _strip_replicate


def _touch(d, names):
    for n in names:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_strip_replicate_unit():
    assert _strip_replicate("rut2080_1") == "rut2080"      # not over-stripped to 'rut'
    assert _strip_replicate("rut2080") == "rut2080"        # no replicate tail -> unchanged
    assert _strip_replicate("wt_rep1") == "wt"
    assert _strip_replicate("wt_r01") == "wt"
    assert _strip_replicate("wt_n3") == "wt"
    assert _strip_replicate("0.5uM_1") == "0.5uM"          # decimal preserved
    assert _strip_replicate("control_01") == "control"
    assert _strip_replicate("neuron_1") == "neuron"        # 'n' of neuron not consumed


def test_arbitrary_genotypes_three_groups(tmp_path):
    _touch(tmp_path, ["geno1_1.tif", "geno1_2.tif", "geno2_1.tif", "geno2_2.tif", "dnc_1.tif", "dnc_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "filename_prefix" and r["confidence"] == "high"
    assert set(r["groups"]) == {"geno1", "geno2", "dnc"}


def test_digit_ending_condition_not_overstripped(tmp_path):
    _touch(tmp_path, ["rut2080_1.tif", "rut2080_2.tif", "dnc1_1.tif", "dnc1_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"rut2080", "dnc1"}  # NOT {'rut', 'dnc'}


def test_decimal_dose_preserved(tmp_path):
    _touch(tmp_path, ["0.5uM_1.tif", "0.5uM_2.tif", "1uM_1.tif", "1uM_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"0.5uM", "1uM"}


def test_replicate_word_forms(tmp_path):
    _touch(tmp_path, ["wt_rep1.tif", "wt_rep2.tif", "dnc_rep1.tif", "dnc_rep2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"wt", "dnc"}


def test_control_treated_via_prefix(tmp_path):
    _touch(tmp_path, ["control_01.tif", "control_02.tif", "treated_01.tif", "treated_02.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"control", "treated"}


def test_single_replicate_groups_warn_not_dropped(tmp_path):
    _touch(tmp_path, ["geno1_1.tif", "geno1_2.tif", "geno2_1.tif", "geno3_1.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert set(r["groups"]) == {"geno1", "geno2", "geno3"}  # labels kept
    assert r["confidence"] == "medium"
    assert any("single-replicate" in w for w in r["warnings"])


def test_subfolder_arbitrary_names(tmp_path):
    _touch(tmp_path, ["rut2080/a.tif", "rut2080/b.tif", "dnc1/c.tif", "wt/d.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "subfolder"
    assert set(r["groups"]) == {"rut2080", "dnc1", "wt"}


def test_mixed_root_and_subfolder_not_subfolder_mode(tmp_path):
    # root-level images present -> subfolder mode is skipped so no file is lost.
    _touch(tmp_path, ["Ctrl/a_1.tif", "Ctrl/a_2.tif", "Treat/b_1.tif", "c_1.tif", "c_2.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] != "subfolder"
    assert "c_1.tif" in r["mapping"]  # root file is grouped, not dropped


def test_multi_factor_low_confidence(tmp_path):
    _touch(tmp_path, ["wt_10uM_1.tif", "wt_20uM_1.tif", "dnc_10uM_1.tif", "dnc_20uM_1.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "filename_factor" and r["confidence"] == "low"
    assert any("factorial" in w or "factors" in w for w in r["warnings"])


def test_duplicate_basename_refused_globally(tmp_path):
    _touch(tmp_path, ["Control/x.tif", "Treatment/x.tif", "Control/y.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "single_cohort" and r["confidence"] == "low"
    assert any("duplicate" in w for w in r["warnings"])


def test_no_structure_single_cohort(tmp_path):
    _touch(tmp_path, ["sampleA.tif", "sampleB.tif", "sampleC.tif"])
    r = infer_groups_from_folder(str(tmp_path))
    assert r["basis"] == "single_cohort" and r["groups"] == ["all"]
