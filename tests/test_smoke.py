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
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)


def test_trainer_imports_pil_image():
    import scat.trainer as t
    assert hasattr(t, "Image"), "trainer.py must import PIL.Image (used by DataLoader)"
