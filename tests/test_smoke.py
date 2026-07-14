"""Import smoke tests.

Regression guard: report.py once used ``Union`` without importing it, which
broke ``import scat`` on Python 3.10-3.13 (masked on 3.14 by PEP 649). These
tests fail loudly if any module can no longer be imported.
"""
import importlib

import pytest

MODULES = [
    "scat",
    "scat.config",
    "scat.detector",
    "scat.features",
    "scat.spatial",
    "scat.classifier",
    "scat.trainer",       # was broken: used PIL.Image without importing it
    "scat.segmentation",
    "scat.analyzer",
    "scat.statistics",
    "scat.visualization",
    "scat.report",        # was broken: Union used but not imported
    "scat.cli",           # scriptable entry point (analyze/chat/train/label/gui)
    "scat.grouping_util",
]

# GUI modules need PySide6 (a core dep) but no display; guarded so a headless/agent-less env
# still runs the core smoke set. These are the largest untested modules — a bad import here
# (the class of bug this file exists to catch) would otherwise surface only at GUI launch.
GUI_MODULES = ["scat.ui_common", "scat.main_gui", "scat.labeling_gui"]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", GUI_MODULES)
def test_gui_import(module):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    importlib.import_module(module)


def test_trainer_imports_pil_image():
    import scat.trainer as t
    assert hasattr(t, "Image"), "trainer.py must import PIL.Image (used by DataLoader)"
