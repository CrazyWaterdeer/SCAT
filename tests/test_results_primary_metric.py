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


def test_hero_group_aware_note(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    gm = {f"ctrl_{i}.tif": "Control" for i in range(3)}
    gm.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric="total_deposits", groups=gm, annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    sub = tab.hero_sub.text().lower()
    assert "pooled across 2 groups" in sub and "group image-means" in sub


def test_hero_no_group_note_when_ungrouped(synth_dir, tmp_path):
    QApplication.instance() or QApplication([])
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"), annotate=False)
    tab = ResultsTab(); tab.load_results(_results_dict_from_output(Path(res.output_dir)))
    assert "pooled across" not in tab.hero_sub.text().lower()
