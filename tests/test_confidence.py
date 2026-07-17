# tests/test_confidence.py
import pandas as pd
from scat import confidence


def test_flagged_run_is_factual_and_worklike():
    t = confidence.run_trust(
        pd.DataFrame({"filename": ["a", "a", "b"], "confidence": [0.55, 0.9, 0.95]}), threshold=0.60)
    assert t["total"] == 3 and t["below"] == 1 and t["state"] == "review"
    assert t["line"] == "1 of 3 deposits below the confidence-score threshold (0.60) — review recommended"
    # never a reliability word
    for bad in ("high-confidence", "reviewed", "reliable", "trustworthy"):
        assert bad not in t["line"].lower()


def test_all_above_threshold_is_not_green_reliability():
    t = confidence.run_trust(pd.DataFrame({"filename": ["a"], "confidence": [0.9]}), 0.60)
    assert t["state"] == "none_flagged" and t["below"] == 0
    assert t["line"] == "all 1 deposits at or above the confidence-score threshold (0.60)"


def test_empty_or_missing_is_unavailable_not_clean():
    assert confidence.run_trust(None, 0.60)["state"] == "unavailable"
    assert confidence.run_trust(pd.DataFrame({"filename": []}), 0.60)["state"] == "unavailable"
    assert confidence.run_trust(pd.DataFrame({"confidence": []}), 0.60)["state"] == "unavailable"
