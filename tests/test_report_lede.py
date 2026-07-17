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


def _grouped_report(tmp_path, synth_dir, primary):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    # synth_dir ships ctrl_0..2 / treat_0..2 — wire the 2-group map so stats actually run.
    groups = {f"ctrl_{i}.tif": "Control" for i in range(3)}
    groups.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric=primary, groups=groups, annotate=False)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd))     # 2 groups -> real comparison
    generate_report_service(str(rd), statistical_results=stats, group_by="group")
    return (rd / "report.html").read_text()


def test_lede_leads_with_finding_for_default_metric(synth_dir, tmp_path):
    html = _grouped_report(tmp_path, synth_dir, "total_deposits")
    assert 'class="finding"' in html
    assert "Total deposits" in html
    # grouped run -> a real verdict sentence (differed / no detected difference), not the fallback.
    assert "differ" in html.lower() and "group" in html.lower()
    assert "averaged" not in html.lower()               # NOT the descriptive fallback
    assert "confidence-score threshold" in html          # factual trust line
    for bad in ("did not differ", "high-confidence", "reviewed in the app", "caused"):
        assert bad not in html.lower()


def test_lede_escapes_composed_fields(monkeypatch, synth_dir, tmp_path):
    # markup coming from the composed fields must be escaped in the lede (the report footer legitimately
    # contains its own <script>, so we check the specific injected string, not the whole document).
    import scat.findings as _findings
    monkeypatch.setattr(_findings, "compose_finding", lambda **kw: {
        "sentence": "<script>alert('x')</script>", "metric": "m", "test": "t", "scope": "s"})
    html = _grouped_report(tmp_path, synth_dir, "rod_fraction")
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html   # escaped in the lede
    assert "<script>alert('x')</script>" not in html                    # never rendered raw


def test_methods_appendix_is_honest(synth_dir, tmp_path):
    html = _grouped_report(tmp_path, synth_dir, "total_deposits").lower()
    assert "methods" in html
    assert "image-level" in html                                     # experimental unit
    assert ("uncalibrated" in html) or ("not a calibrated probability" in html)  # confidence honesty
    assert "reject class" in html                                    # artifacts = reject class
    # conditional test wording — must NOT hard-claim one test always applies
    assert "depending on" in html or "normality" in html
