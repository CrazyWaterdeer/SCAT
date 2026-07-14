"""LabelingWindow deposit-from-contour helper (scat/labeling_gui.py) — the dedup'd shared tail."""
import types

import numpy as np

from scat.labeling_gui import LabelingWindow
from scat.detector import Deposit


def _fake_window():
    ns = types.SimpleNamespace(
        next_id=1, deposits=[], image=np.zeros((60, 60, 3), np.uint8),
        extractor=types.SimpleNamespace(extract_features=lambda img, deps: None),
        viewer=types.SimpleNamespace(add_deposit=lambda d: None), _msgs=[])
    ns._update_table = lambda: None
    ns._update_stats = lambda: None
    ns.statusBar = lambda: types.SimpleNamespace(showMessage=lambda m: ns._msgs.append(m))
    return ns


def test_finalize_builds_and_registers_deposit():
    w = _fake_window()
    contour = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])       # 20x20 square
    dep = LabelingWindow._finalize_deposit_from_contour(w, contour)
    assert isinstance(dep, Deposit) and dep.id == 1 and w.next_id == 2
    assert len(w.deposits) == 1 and abs(dep.area - 400) < 1 and dep.width >= 20
    assert w._msgs and "deposit 1" in w._msgs[-1]


def test_finalize_min_area_guard_rejects_tiny():
    w = _fake_window()
    tiny = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    dep = LabelingWindow._finalize_deposit_from_contour(w, tiny, min_area=5)
    assert dep is None and w.deposits == [] and w.next_id == 1


def test_finalize_fallback_centroid_used_for_degenerate_contour():
    w = _fake_window()
    line = np.array([[5, 5], [15, 5]])                                 # zero-area -> moments m00==0
    dep = LabelingWindow._finalize_deposit_from_contour(w, line, fallback_centroid=(7, 8))
    assert dep is not None and dep.centroid == (7, 8)


def test_finalize_manual_status_message():
    w = _fake_window()
    contour = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
    LabelingWindow._finalize_deposit_from_contour(w, contour, manual=True)
    assert "manual deposit" in w._msgs[-1]
