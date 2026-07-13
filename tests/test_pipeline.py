"""End-to-end analysis pipeline tests (detection -> classify -> report CSVs)."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scat.detector import DepositDetector
from scat.classifier import ClassifierConfig
from scat.analyzer import Analyzer, ReportGenerator

MODEL = Path(__file__).resolve().parents[1] / "models" / "model_rf.pkl"


def _config():
    if MODEL.exists():
        return ClassifierConfig(model_type="rf", model_path=str(MODEL))
    return ClassifierConfig(model_type="threshold")


def test_detection_nonempty(synth_dir):
    img = np.array(Image.open(sorted(synth_dir.glob("*.tif"))[0]))
    deposits = DepositDetector().detect(img)
    assert len(deposits) > 0
    d = deposits[0]
    assert d.area > 0 and d.id >= 1


def test_rf_pipeline_writes_reports(synth_dir, tmp_path):
    analyzer = Analyzer(detector=DepositDetector(), classifier_config=_config())
    paths = sorted(synth_dir.glob("*.tif"))
    results = analyzer.analyze_batch(paths)
    assert len(results) == len(paths)
    assert sum(r.n_total for r in results) > 0

    meta = pd.read_csv(synth_dir / "groups.csv")
    reports = ReportGenerator(tmp_path).save_all(results, meta, group_by=["group"])

    assert (tmp_path / "image_summary.csv").exists()
    assert (tmp_path / "all_deposits.csv").exists()
    for col in ["filename", "n_total", "n_normal", "n_rod", "n_artifact", "rod_fraction"]:
        assert col in reports["film_summary"].columns
    # per-image label JSON is written for re-editing / retraining
    assert list((tmp_path / "deposits").glob("*.labels.json"))


def test_parallel_matches_sequential(synth_dir):
    """Guards the analyze_batch thread-safety fix: concurrent per-image
    extraction must yield the same per-file counts as sequential."""
    analyzer = Analyzer(detector=DepositDetector(), classifier_config=_config())
    paths = sorted(synth_dir.glob("*.tif"))
    seq = analyzer.analyze_batch(paths, parallel=False)
    par = analyzer.analyze_batch(paths, parallel=True, max_workers=4)
    assert [r.filename for r in seq] == [r.filename for r in par]  # order preserved
    for s, p in zip(seq, par):
        assert (s.n_total, s.n_normal, s.n_rod, s.n_artifact) == \
               (p.n_total, p.n_normal, p.n_rod, p.n_artifact)
