"""Phase-2 GUI slimdown — driven offscreen smoke test.

Exercises the real rewired flow end to end (not just imports): subfolder auto-grouping →
the Run closure calling the pipeline services → the Results tab rendering the (nested→flat)
stats without crashing → regenerate-after-edit. Requires PySide6 with an offscreen platform.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


@pytest.fixture
def no_modal(monkeypatch):
    """Neutralise blocking dialogs so the driven flow runs headless."""
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))


@pytest.fixture
def grouped_images(tmp_path):
    """Blue-blob images split across two per-condition subfolders (ctrl/, treated/)."""
    def make(path, n):
        w = h = 256
        img = Image.new("RGB", (w, h), (245, 244, 240))
        rng = np.random.RandomState(abs(hash(path.name)) % 2**32)
        arr = np.array(img)
        for _ in range(n):
            cx, cy = rng.randint(30, w - 30), rng.randint(30, h - 30)
            r = rng.randint(8, 14)
            y, x = np.ogrid[:h, :w]
            arr[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = (55, 60, 185)
        Image.fromarray(arr).save(path)

    files = []
    for cond, n in (("ctrl", 9), ("treated", 5)):
        d = tmp_path / cond
        d.mkdir()
        for i in range(3):
            p = d / f"{cond}_{i}.png"
            make(p, n)
            files.append(str(p))
    return tmp_path, files


def _drive_run(tab, files, out_dir):
    """Populate the tab and call the Run closure synchronously (no QThread)."""
    from scat.main_gui import WorkerThread
    tab._selected_files = files
    tab._image_files_for_analysis = files
    tab.model_type.setCurrentIndex(0)   # threshold — no model file needed
    tab.use_groups.setChecked(True)
    tab.spatial.setChecked(True)
    tab.report.setChecked(True)
    tab.stats.setChecked(True)
    tab._autogroup_by_subfolder(announce=False)
    tab.worker = WorkerThread(lambda: None)  # gives progress_callback a live signal
    return tab._do_analysis(str(out_dir))


def test_autogroup_by_subfolder(app, grouped_images):
    from scat.main_gui import AnalysisTab
    _, files = grouped_images
    tab = AnalysisTab()
    tab._selected_files = files
    tab._image_files_for_analysis = files
    tab._autogroup_by_subfolder(announce=False)
    assert set(tab._group_data.keys()) == {"ctrl", "treated"}
    assert all(len(v) == 3 for v in tab._group_data.values())


def test_run_flow_produces_outputs_and_renders(app, no_modal, grouped_images, tmp_path):
    from scat.main_gui import AnalysisTab, ResultsTab
    root, files = grouped_images
    out = tmp_path / "results"

    tab = AnalysisTab()
    result = _drive_run(tab, files, out)

    # Outputs written by the canonical services
    assert (out / "image_summary.csv").exists()
    assert (out / "annotated").exists()
    assert (out / "visualizations").exists()
    assert (out / "report.html").exists()
    assert (out / "spatial_stats.json").exists()

    # Rebuilt Results-tab dict: flat stats mapping (F5), grouping detected, paths carried
    assert result["group_by"] == "group"
    assert isinstance(result["stats_results"], dict)
    assert result["image_paths"] == files
    assert "is_quick_mode" not in result

    # Results tab renders the nested→flat stats WITHOUT KeyError (the F5 crash guard)
    rtab = ResultsTab()
    rtab.load_results(result)  # would raise on the old nested-shape mismatch


def test_regenerate_after_edit(app, no_modal, grouped_images, tmp_path, monkeypatch):
    from scat.main_gui import AnalysisTab, ResultsTab
    from scat import config as cfg_mod
    root, files = grouped_images
    out = tmp_path / "results2"

    tab = AnalysisTab()
    result = _drive_run(tab, files, out)

    # Simulate the edit flow's config state (originals live in subfolders under root)
    cfg_mod.config.set("last_input_dir", str(root))

    rtab = ResultsTab()
    rtab.load_results(result)
    # _find_original_image must resolve a subfolder original (Codex F2)
    assert rtab._find_original_image(Path(files[0]).name) is not None
    # regenerate: no inline cv2, services drive stats+report; must not raise
    rtab._generate_report()
    assert (out / "report.html").exists()
