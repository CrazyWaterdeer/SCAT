import json
from pathlib import Path
from scat.pipeline import analyze_folder_service


def test_service_persists_primary_metric(synth_dir, tmp_path):
    out = tmp_path / "out"
    res = analyze_folder_service(str(synth_dir), output_dir=str(out),
                                 primary_metric="rod_fraction", normalization="per_image",
                                 confidence_threshold=0.6, annotate=False)
    m = json.loads((Path(res.output_dir) / "run_manifest.json").read_text())
    assert m["analysis"]["primary_metric"] == "rod_fraction"
