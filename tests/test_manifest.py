"""Reproducibility manifest primitives (scat/manifest.py) + the analyze_folder_service sidecar."""
import hashlib
import json
from pathlib import Path

from scat import manifest
from scat.pipeline import analyze_folder_service


def test_run_context_has_keys():
    ctx = manifest.run_context()
    for k in ("scat_version", "git_commit", "python", "platform"):
        assert k in ctx
    assert ctx["scat_version"]  # non-empty (module __version__)


def test_dataset_fingerprint_order_independent_and_size_sensitive(tmp_path):
    a = tmp_path / "a.png"; a.write_bytes(b"12345")
    b = tmp_path / "b.png"; b.write_bytes(b"67")
    fp1 = manifest.dataset_fingerprint([str(a), str(b)])
    fp2 = manifest.dataset_fingerprint([str(b), str(a)])   # reversed order -> same fingerprint
    assert fp1 == fp2 and fp1["n_images"] == 2
    a.write_bytes(b"123456")                                # a content/size change flips the hash
    assert manifest.dataset_fingerprint([str(a), str(b)])["sha256"] != fp1["sha256"]


def test_sha256_file(tmp_path):
    f = tmp_path / "m.bin"; f.write_bytes(b"hello")
    assert manifest.sha256_file(str(f)) == hashlib.sha256(b"hello").hexdigest()
    assert manifest.sha256_file(str(tmp_path / "nope")) is None


def test_write_run_manifest(tmp_path):
    img = tmp_path / "x.tif"; img.write_bytes(b"i")
    out = tmp_path / "out"; out.mkdir()
    manifest.write_run_manifest(
        out, path=str(tmp_path), image_paths=[str(img)], model_type="rf", model_path=None,
        circularity=0.6, groups={"x.tif": "ctrl"}, group_column="group",
        detection={"min_area": 20}, warnings=["w"])
    disk = json.loads((out / "run_manifest.json").read_text())
    assert disk["schema"] == manifest.SCHEMA and disk["created_at"]
    assert disk["dataset"]["n_images"] == 1
    assert disk["model"]["type"] == "rf" and disk["model"]["sha256"] is None
    assert disk["model"]["circularity"] == 0.6           # classifier param under model, not detection
    assert "circularity" not in disk["detection"]
    assert disk["grouping"]["mapping"] == {"x.tif": "ctrl"} and disk["grouping"]["column"] == "group"
    assert disk["detection"]["min_area"] == 20 and disk["warnings"] == ["w"]


def test_run_manifest_stores_absolute_dataset_path(tmp_path):
    """A relative dataset path would later resolve against a different cwd and miss the run on resume."""
    img = tmp_path / "x.tif"; img.write_bytes(b"i")
    out = tmp_path / "out"; out.mkdir()
    manifest.write_run_manifest(out, path="some/relative/dir", image_paths=[str(img)],
                                model_type="rf", model_path=None)
    disk = json.loads((out / "run_manifest.json").read_text())
    assert Path(disk["dataset"]["path"]).is_absolute()


def test_dataset_fingerprint_disambiguates_subfolders(tmp_path):
    """Same basename+size in different subfolders must NOT collide (Codex F1)."""
    (tmp_path / "ctrl").mkdir(); (tmp_path / "treated").mkdir()
    a = tmp_path / "ctrl" / "x.png"; a.write_bytes(b"same")
    b = tmp_path / "treated" / "x.png"; b.write_bytes(b"same")   # identical name + size
    flat = tmp_path / "x.png"; flat.write_bytes(b"same")
    fp_tree = manifest.dataset_fingerprint([str(a), str(b)])
    fp_flat = manifest.dataset_fingerprint([str(flat), str(flat)])
    assert fp_tree["sha256"] != fp_flat["sha256"]


def test_service_writes_run_manifest(synth_dir, tmp_path):
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "r"), annotate=False)
    m = json.loads((Path(res.output_dir) / "run_manifest.json").read_text())
    assert m["dataset"]["n_images"] == 6
    assert "scat_version" in m and "git_commit" in m
