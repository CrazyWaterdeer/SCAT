"""Regression tests for the SCAT 2.0 Stage-2 correctness fixes."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_grayscale_input_does_not_crash(tmp_path):
    """[5] A single-channel (mode 'L') image must be normalized to RGB, not crash
    feature extraction at cv2.cvtColor(RGB2HLS)."""
    from scat.analyzer import Analyzer

    p = tmp_path / "gray.tif"
    arr = np.full((256, 256), 240, dtype=np.uint8)
    arr[40:60, 40:120] = 60          # a dark blob so detection has something to find
    Image.fromarray(arr, mode="L").save(p, dpi=(600, 600))

    res = Analyzer().analyze_image(p)          # must not raise
    assert res.failed is False
    assert res.n_total >= 0


def test_rgba_input_does_not_crash(tmp_path):
    """[5] RGBA should also be coerced to 3-channel RGB."""
    from scat.analyzer import Analyzer

    p = tmp_path / "rgba.png"
    arr = np.full((256, 256, 4), 255, dtype=np.uint8)
    arr[40:60, 40:120, :3] = 60
    Image.fromarray(arr, mode="RGBA").save(p)
    res = Analyzer().analyze_image(p)
    assert res.failed is False


def test_clean_image_is_not_a_failure(tmp_path):
    """[26] A blank (0-deposit) image is analyzed fine and must NOT be tagged failed;
    only the parallel placeholder (a real failure) carries failed=True."""
    from scat.analyzer import Analyzer
    from scat.parallel import _placeholder

    p = tmp_path / "blank.tif"
    Image.fromarray(np.full((128, 128, 3), 245, dtype=np.uint8), mode="RGB").save(p, dpi=(600, 600))
    res = Analyzer().analyze_image(p)
    assert res.failed is False                 # clean, not a failure

    ph = _placeholder(Analyzer(), str(p))
    assert ph.failed is True and ph.n_total == 0


def test_guess_control_group_matches_is_control(tmp_path):
    """[13] guess_control_group now delegates to _is_control, so 'sham' (and any other
    _CONTROL_SUBSTR term) is detected consistently with group ordering."""
    from scat.visualization import guess_control_group, order_groups, _is_control

    assert guess_control_group(["sham", "treated"]) == "sham"
    assert _is_control("sham")
    # ordering leads with the same control the bracket logic will pick
    assert order_groups(["treated", "sham"])[0] == "sham"
    # regression: the originally-covered cases still hold
    assert guess_control_group(["Control", "Drug"]) == "Control"
    assert guess_control_group(["WT", "mutant"]) == "WT"
    assert guess_control_group(["A", "B"]) is None


