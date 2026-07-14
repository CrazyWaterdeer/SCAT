from pathlib import Path
import pandas as pd
from scat.pipeline import analyze_folder_service, list_images, resolve_model_type


def test_service_matches_direct_pipeline(synth_dir, tmp_path):
    """Anti-drift gate: the service must produce output identical to the raw Analyzer path."""
    from scat.detector import DepositDetector
    from scat.classifier import ClassifierConfig
    from scat.analyzer import Analyzer, ReportGenerator
    imgs = list_images(str(synth_dir))
    mtype, mpath = resolve_model_type(None, None)
    az = Analyzer(detector=DepositDetector(), classifier_config=ClassifierConfig(model_type=mtype, model_path=mpath))
    ReportGenerator(tmp_path / "direct").save_all(az.analyze_batch(imgs))
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "svc"), annotate=False)
    for name in ("all_deposits.csv", "image_summary.csv"):
        a = pd.read_csv(tmp_path / "direct" / name)
        b = pd.read_csv(Path(res.output_dir) / name)
        pd.testing.assert_frame_equal(a, b)


def test_grouped_service_lights_up_stats(synth_dir, tmp_path):
    imgs = list_images(str(synth_dir))
    groups = {p.name: ("control" if i % 2 == 0 else "treated") for i, p in enumerate(imgs)}
    res = analyze_folder_service(str(synth_dir), groups=groups, output_dir=str(tmp_path / "g"), annotate=False)
    assert set(res.groups) == {"control", "treated"}
    film = pd.read_csv(Path(res.output_dir) / "image_summary.csv")
    assert "group" in film.columns
    assert (Path(res.output_dir) / "condition_summary.csv").exists()
