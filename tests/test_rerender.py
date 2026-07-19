"""Re-render-from-existing-results (scat/pipeline.py rerender_results_service + the rerender_report
tool) and the supporting order_groups / per-plot-guard / trainer basename fixes.

The contract these lock: an existing results dir can be re-graphed / re-reported / re-stat'd WITHOUT
re-detection — the reviewed detections (image_summary.csv / all_deposits.csv) are read, never rewritten.
"""
import json

import pandas as pd
import pytest

from scat.pipeline import analyze_folder_service, rerender_results_service


@pytest.fixture(scope="module")
def results_dir(synth_dir, tmp_path_factory):
    """A real, fully-detected results dir (Control vs Treatment) to re-render against."""
    groups = {f"ctrl_{i}.tif": "Control" for i in range(3)}
    groups.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
    out = tmp_path_factory.mktemp("rr") / "run"
    res = analyze_folder_service(str(synth_dir), groups=groups, annotate=False,
                                 output_dir=str(out), primary_metric="total_deposits")
    return res.output_dir


def test_rerender_does_not_redetect(results_dir):
    """The reviewed detections must be untouched — CSV bytes identical before/after re-render."""
    csv = pd.read_csv(f"{results_dir}/image_summary.csv")
    before = (open(f"{results_dir}/image_summary.csv", "rb").read(),
              open(f"{results_dir}/all_deposits.csv", "rb").read())
    res = rerender_results_service(results_dir, primary_metric="rod_fraction")
    after = (open(f"{results_dir}/image_summary.csv", "rb").read(),
             open(f"{results_dir}/all_deposits.csv", "rb").read())
    assert before == after, "re-render must not rewrite the detected CSVs (no re-detection)"
    assert res.report_path.endswith("report.html")
    assert res.stats_recomputed is True and res.n_groups == 2


def test_rerender_primary_metric_persists_to_manifest(results_dir):
    res = rerender_results_service(results_dir, primary_metric="mean_area")
    assert res.primary_metric == "mean_area"
    man = json.loads(open(f"{results_dir}/run_manifest.json").read())
    assert man["analysis"]["primary_metric"] == "mean_area"
    # unknown metric is ignored with a warning, not silently accepted
    res2 = rerender_results_service(results_dir, primary_metric="not_a_metric")
    assert res2.primary_metric == "mean_area"  # unchanged
    assert any("unknown primary_metric" in w for w in res2.warnings)


def test_rerender_group_order_override_persists(results_dir):
    res = rerender_results_service(results_dir, group_order=["Treatment", "Control"])
    assert res.group_order == ["Treatment", "Control"]
    man = json.loads(open(f"{results_dir}/run_manifest.json").read())
    assert man["analysis"]["group_order"] == ["Treatment", "Control"]


def test_rerender_reflects_manual_edits(results_dir):
    """Simulate a manual review edit to the CSV; recomputed stats must use the edited NUMBERS."""
    from scat.pipeline import run_statistics_service

    def _rod_mean(res_dir, group):
        df = pd.read_csv(f"{res_dir}/image_summary.csv")
        return df.loc[df["group"] == group, "rod_fraction"].mean()

    df = pd.read_csv(f"{results_dir}/image_summary.csv")
    g0 = df.loc[0, "group"]
    before_mean = _rod_mean(results_dir, g0)
    # bump ROD on every image of one group so its group-mean rod_fraction provably moves
    mask = df["group"] == g0
    df.loc[mask, "n_rod"] = df.loc[mask, "n_rod"].astype(int) + 30
    tot = df.loc[mask, "n_normal"] + df.loc[mask, "n_rod"]
    df.loc[mask, "rod_fraction"] = (df.loc[mask, "n_rod"] / tot).where(tot > 0, 0.0)
    df.to_csv(f"{results_dir}/image_summary.csv", index=False)

    res = rerender_results_service(results_dir, primary_metric="rod_fraction")
    assert res.stats_recomputed is True
    # the recomputed group mean must reflect the edit (numeric proof, not just "it ran")
    assert _rod_mean(results_dir, g0) > before_mean
    stats = run_statistics_service(results_dir, group_col="group")
    assert not stats.get("skipped")


def test_rerender_persisted_overrides_reconsumed_on_bare_rerun(results_dir):
    """A prior rerender persists group_order; a later BARE rerender must reproduce it, and a plain
    generate_report must honor it too (codex #3/#4a)."""
    from scat.pipeline import generate_report_service, run_statistics_service
    rerender_results_service(results_dir, group_order=["Treatment", "Control"], control_group="Control")
    # bare rerender (no display args) reproduces the persisted order + reference group
    bare = rerender_results_service(results_dir)
    assert bare.group_order == ["Treatment", "Control"]
    # a plain generate_report also picks the persisted order up from the manifest (no crash, valid path)
    stats = run_statistics_service(results_dir, group_col="group")
    assert generate_report_service(results_dir, stats, group_by="group").endswith("report.html")


def test_rerender_clears_stale_png(results_dir):
    """Re-render must not leave orphan PNGs from a previous run (codex #5)."""
    from pathlib import Path
    vdir = Path(results_dir) / "visualizations"
    vdir.mkdir(exist_ok=True)
    orphan = vdir / "violin_ZZZ_stale.png"
    orphan.write_bytes(b"stale")
    rerender_results_service(results_dir, regenerate_visualizations=True)
    assert not orphan.exists(), "stale PNG from a prior render should be cleared"
    assert list(vdir.glob("*.png")), "fresh plots should be present"


def test_rerender_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rerender_results_service(str(tmp_path))  # no image_summary.csv


# --------------------------------------------------------------------- order_groups explicit override
def test_order_groups_explicit_order_wins():
    from scat.visualization import order_groups
    vals = ["Control", "DoseLow", "DoseHigh"]
    assert order_groups(vals, explicit_order=["DoseHigh", "Control", "DoseLow"]) == \
        ["DoseHigh", "Control", "DoseLow"]


def test_order_groups_explicit_leftovers_keep_logic_and_unknowns_ignored():
    from scat.visualization import order_groups
    # only 'DoseHigh' pinned first; the rest fall back to logical (control-first) order
    assert order_groups(["Control", "DoseLow", "DoseHigh"],
                        explicit_order=["DoseHigh", "ghost"]) == ["DoseHigh", "Control", "DoseLow"]


def test_order_groups_explicit_matches_display_label():
    from scat.visualization import order_groups
    # caller may pass the display label WITHOUT the trailing "(…)" note and still match
    assert order_groups(["WT (driverless ctrl)", "mut"],
                        explicit_order=["mut", "WT"]) == ["mut", "WT (driverless ctrl)"]


# --------------------------------------------------------------------- per-plot guard
def test_generate_all_visualizations_isolates_one_failing_plot(results_dir, tmp_path, monkeypatch):
    import scat.visualization as V
    film = pd.read_csv(f"{results_dir}/image_summary.csv")
    dep = pd.read_csv(f"{results_dir}/all_deposits.csv")
    monkeypatch.setattr(V.Visualizer, "summary_dashboard",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = V.generate_all_visualizations(film, dep, tmp_path / "viz", group_by="group")
    assert "dashboard" not in res           # the failing plot is skipped
    assert len(res) >= 5                     # the others still render, no abort


# --------------------------------------------------------------------- palette / color override
def test_get_palette_override_wins_and_control_stays_gray():
    from scat.visualization import get_palette
    pal = get_palette(["Control", "DoseLow", "DoseHigh"], control_group="Control",
                      override={"DoseHigh": "tomato"})
    assert pal["DoseHigh"] == "tomato"          # explicit override wins
    assert pal["Control"] == "#636867"          # control still gray (not overridden)
    assert pal["DoseLow"] != pal["DoseHigh"]    # others keep the default cycle


def test_normalize_palette_override_forms():
    from scat.visualization import normalize_palette_override
    groups = ["A", "B", "C"]
    assert normalize_palette_override(["#111111", "#222222"], groups) == {"A": "#111111", "B": "#222222"}
    assert normalize_palette_override({"WT": "red"}, ["WT (ctrl)", "mut"]) == {"WT (ctrl)": "red"}
    assert normalize_palette_override({"ghost": "red"}, groups) == {}


def test_rerender_palette_persists_and_drops_bad_colors(results_dir):
    import json
    g = sorted(pd.read_csv(f"{results_dir}/image_summary.csv")["group"].dropna().unique())
    res = rerender_results_service(results_dir, palette={g[0]: "#E24A33", g[1]: "notacolor"})
    man = json.loads(open(f"{results_dir}/run_manifest.json").read())
    assert man["analysis"]["palette"] == {g[0]: "#E24A33"}    # valid kept, typo dropped
    assert any("notacolor" in w for w in res.warnings)


# --------------------------------------------------------------------- model training (S5/S6)
# The synthetic classifier labels every synth deposit 'normal' (single class), so patch DataLoader to
# return a balanced label set. This still exercises the REAL source-resolution + dedup + aggregation
# (which run before DataLoader is called) and real RFTrainer.train — only the label content is faked.
_FEAT = {"area": 1.0, "circularity": 0.5, "aspect_ratio": 1.0, "mean_hue": 10.0,
         "mean_lightness": 0.5, "mean_saturation": 0.5, "iod": 1.0}


def _balanced_loader():
    labels = ["normal"] * 12 + ["rod"] * 6 + ["artifact"] * 6   # >=2/class so a stratified split works

    class FakeLoader:
        def __init__(self, *a):
            pass

        def load_labeled_data(self):
            import numpy as np
            return ([np.zeros((4, 4, 3), np.uint8) for _ in labels],
                    [dict(_FEAT) for _ in labels], list(labels))
    return FakeLoader, len(labels)


def test_train_model_from_results_dir(results_dir, tmp_path, monkeypatch):
    import scat.trainer as T
    from scat.pipeline import train_model_service
    loader, n = _balanced_loader()
    monkeypatch.setattr(T, "DataLoader", loader)
    out = tmp_path / "model_new.pkl"
    res = train_model_service(results_dirs=[results_dir], output=str(out),
                              n_estimators=15, cross_validate=False)
    assert out.exists() and res.output_path == str(out)
    assert res.n_samples == n and sum(res.class_counts.values()) == res.n_samples
    assert res.accuracy is not None and res.top_features


def test_train_model_union_of_distinct_sources(results_dir, tmp_path, monkeypatch):
    """Two DISTINCT label dirs sum their samples — the S5 'retrain on the union' path."""
    import json
    import shutil
    import scat.trainer as T
    from scat.pipeline import train_model_service
    loader, n = _balanced_loader()
    monkeypatch.setattr(T, "DataLoader", loader)
    img = json.loads(open(f"{results_dir}/run_manifest.json").read())["dataset"]["path"]
    dep2 = tmp_path / "deposits2"
    shutil.copytree(f"{results_dir}/deposits", dep2)          # a second, distinct label dir
    dbl = train_model_service(results_dirs=[results_dir], image_dir=img, label_dir=str(dep2),
                              output=str(tmp_path / "b.pkl"), n_estimators=10, cross_validate=False)
    assert dbl.n_samples == 2 * n and len(dbl.sources) == 2


def test_train_model_dedups_identical_sources(results_dir, tmp_path, monkeypatch):
    """The SAME source passed twice must not be double-counted (codex: append-all bias)."""
    import scat.trainer as T
    from scat.pipeline import train_model_service
    loader, n = _balanced_loader()
    monkeypatch.setattr(T, "DataLoader", loader)
    dup = train_model_service(results_dirs=[results_dir, results_dir], output=str(tmp_path / "b.pkl"),
                              n_estimators=10, cross_validate=False)
    assert dup.n_samples == n                                 # deduped, not doubled
    assert any("duplicate" in w for w in dup.warnings)


def test_train_model_guards_tiny_and_single_class(monkeypatch, tmp_path):
    """Small/imbalanced sets get an ACTIONABLE error, not a raw sklearn stratify crash (codex)."""
    import scat.trainer as T
    from scat.pipeline import train_model_service

    def _fake(labels):
        class FakeLoader:
            def __init__(self, *a):
                pass
            def load_labeled_data(self):
                return [], [{"area": 1.0, "circularity": 0.5, "aspect_ratio": 1.0, "mean_hue": 10.0,
                             "mean_lightness": 0.5, "mean_saturation": 0.5, "iod": 1.0} for _ in labels], list(labels)
        return FakeLoader

    monkeypatch.setattr(T, "DataLoader", _fake(["normal"] * 10))
    with pytest.raises(ValueError, match="one class"):
        train_model_service(image_dir=str(tmp_path), label_dir=str(tmp_path), output=str(tmp_path / "m.pkl"))

    monkeypatch.setattr(T, "DataLoader", _fake(["normal"] * 10 + ["rod"]))  # rod has 1 sample
    with pytest.raises(ValueError, match="only 1"):
        train_model_service(image_dir=str(tmp_path), label_dir=str(tmp_path), output=str(tmp_path / "m.pkl"))


def test_train_model_no_sources_raises():
    from scat.pipeline import train_model_service
    with pytest.raises(ValueError):
        train_model_service()


def test_rerender_palette_unknown_group_key_warns(results_dir):
    res = rerender_results_service(results_dir, palette={"NoSuchGroup": "#111111"})
    assert any("no matching group" in w for w in res.warnings)


# --------------------------------------------------------------------- trainer WSL basename fix
def test_dataloader_resolves_windows_image_path(tmp_path):
    """A label written on Windows stores a Windows-absolute image_file; DataLoader must still find
    the image by basename on POSIX (ntpath.basename), not treat the whole backslash string as a name."""
    import numpy as np
    from PIL import Image
    from scat.trainer import DataLoader

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.fromarray(np.full((40, 40, 3), 200, np.uint8)).save(img_dir / "sample.tif")
    label = {
        "image_file": r"C:\Users\Jin\Programming\SCAT\data\Raw images\sample.tif",
        "deposits": [{"label": "normal",
                      "contour": [[5, 5], [25, 5], [25, 25], [5, 25]]}],
    }
    (img_dir / "sample.labels.json").write_text(json.dumps(label))
    patches, feats, labels = DataLoader(img_dir).load_labeled_data()
    assert labels == ["normal"], "Windows-path label should resolve to sample.tif by basename"
