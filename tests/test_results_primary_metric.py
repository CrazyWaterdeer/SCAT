import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from scat.main_gui import ResultsTab, _results_dict_from_output
from scat.pipeline import analyze_folder_service


def test_hero_reflects_primary_metric(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="total_deposits", annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    assert "/ image" in tab.hero_value.text()               # deposits as a rate
    assert "DEPOSIT" in tab.hero_kicker.text().upper()      # not "ROD FRACTION"
