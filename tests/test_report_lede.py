import json
from pathlib import Path
import scat.report as report_mod


def test_analysis_block_reaches_build_html(monkeypatch, synth_dir, tmp_path):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="rod_fraction", annotate=False)
    rd = Path(res.output_dir)
    seen = {}
    orig = report_mod.ReportGenerator._build_html
    def spy(self, *a, **kw):
        seen.update(kw)
        return orig(self, *a, **kw)
    monkeypatch.setattr(report_mod.ReportGenerator, "_build_html", spy)
    stats = run_statistics_service(str(rd))
    generate_report_service(str(rd), statistical_results=stats, group_by="group")
    assert seen.get("analysis", {}).get("primary_metric") == "rod_fraction"
