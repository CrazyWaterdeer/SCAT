import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from scat.main_gui import _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_results_dict_carries_analysis_and_keeps_deposit_data(synth_dir, tmp_path):
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="rod_fraction", annotate=False)
    d = _results_dict_from_output(Path(res.output_dir))
    assert d["primary_metric"] == "rod_fraction"
    assert d["normalization"] == "per_image"
    assert d["confidence_threshold"] == 0.60
    assert d["run_meta"] == {}                 # empty until metadata capture (later task)
    assert "deposit_data" in d                 # existing key preserved (GUI edit paths read it)
