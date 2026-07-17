import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def _tab(tmp_path, synth_dir, **kw):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False, **kw)
    d = _results_dict_from_output(Path(res.output_dir))
    tab = ResultsTab(); tab.load_results(d)
    return tab, d


def test_composition_strip_replaces_kpi_tiles(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    assert not hasattr(tab, "tiles_layout")                 # KPI tile band gone
    # no residual KPI tile widgets
    assert not [w for w in tab.findChildren(QWidget) if w.objectName() == "kpiTile"]
    fs = d["film_summary"]
    n_normal, n_rod = int(fs['n_normal'].sum()), int(fs['n_rod'].sum())
    txt = tab.composition_line.text()
    assert f"{n_normal + n_rod}" in txt          # explicit Deposits count preserved
    assert str(n_normal) in txt and str(n_rod) in txt
    for token in ("Deposits", "Normal", "ROD", "Artifact"):
        assert token in txt


def test_total_iod_omitted_when_column_absent(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    d2 = dict(d); d2["film_summary"] = d["film_summary"].drop(columns=["total_iod"], errors="ignore")
    tab.load_results(d2)                          # must not raise, must not show "Total IOD 0"
    assert "Total IOD 0" not in tab.composition_line.text()


from scat.ui_common import CollapsibleSection


def test_report_grade_content_is_exiled_to_one_pointer(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    # The stats area holds exactly ONE widget (the pointer) — no gallery, tables, or sections.
    assert tab.stats_layout.count() == 1
    assert not [w for w in tab.findChildren(QWidget) if w.objectName() == "vizCell"]
    assert not tab.findChildren(CollapsibleSection)
    from PySide6.QtWidgets import QLabel
    labels = [w.text() for w in tab.findChildren(QLabel)]
    assert not any("VISUALIZATIONS" in t or "DESCRIPTIVE STATISTICS" in t or "GROUP COMPARISONS" in t
                   for t in labels)


def test_pointer_is_report_state_aware(synth_dir, tmp_path):
    # analyze with annotate=False does NOT write report.html -> pointer must not claim it exists.
    tab, d = _tab(tmp_path, synth_dir)
    ptxt = tab.stats_layout.itemAt(0).widget().text().lower()
    assert "report" in ptxt
    assert "generate" in ptxt          # no report yet -> "Generate a report…", not "in the report"
    # now pretend a report exists and re-load -> pointer flips to the "in the report" wording
    (Path(d["output_dir"]) / "report.html").write_text("<html></html>")
    tab.load_results(_results_dict_from_output(Path(d["output_dir"])))
    assert "in the report" in tab.stats_layout.itemAt(0).widget().text().lower()
    assert tab.stats_layout.count() == 1     # reload still leaves exactly one pointer


def test_reload_updates_composition_and_clears_stats(synth_dir, tmp_path):
    tab, d = _tab(tmp_path, synth_dir)
    tab.load_results(d)                      # second load (as _reload_results does after an edit)
    assert tab.stats_layout.count() == 1
    assert "Deposits" in tab.composition_line.text()
