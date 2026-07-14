"""U-Net segmentation module (scat/segmentation.py) — torch is the optional [deep] extra."""
import importlib.util

import numpy as np
import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_segmentation_imports_without_torch():
    import scat.segmentation as seg   # torch is lazy-loaded, so the module must import regardless
    assert hasattr(seg, "UNetDetector") and hasattr(seg, "SegmentationTrainer")


def test_unet_detector_constructs_without_model():
    from scat.segmentation import UNetDetector
    det = UNetDetector(model_path=None)     # no model -> no torch needed
    assert det.model is None


@pytest.mark.skipif(_HAS_TORCH, reason="torch is installed")
def test_load_torch_raises_helpful_error_without_torch():
    from scat.segmentation import _load_torch
    with pytest.raises(ImportError):
        _load_torch()


@pytest.mark.skipif(_HAS_TORCH, reason="torch is installed")
def test_predict_without_torch_raises():
    from scat.segmentation import UNetDetector
    det = UNetDetector(model_path=None)
    with pytest.raises(Exception):
        det.predict(np.zeros((32, 32, 3), np.uint8))
