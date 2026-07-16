# tests/test_results_confidence.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_review_column_present_with_values(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    t = tab.summary_table
    headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    assert headers == ["Filename", "Review", "Normal", "ROD", "Artifact", "ROD %", "Total IOD"]
    # Column 0 is still Filename (double-click path unchanged); Normal shifted to column 2.
    assert t.item(0, 0) is not None and t.item(0, 2) is not None
    # Review cell exists for every row and reads a count or an em dash.
    for r in range(t.rowCount()):
        txt = t.item(r, 1).text()
        assert txt == "—" or txt.strip().isdigit()


def test_review_column_survives_missing_deposit_data(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    d = _results_dict_from_output(Path(res.output_dir)); d["deposit_data"] = None
    tab = ResultsTab(); tab.load_results(d)  # must not raise
    assert tab.summary_table.item(0, 1).text() == "—"
