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
