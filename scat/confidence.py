# scat/confidence.py
"""Factual run-trust facts from per-deposit classifier confidence (spec §2.1). PURE; no reliability
claims — the classifier score (RF probability or rule-based heuristic) is uncalibrated and covers
classification only (not detection/segmentation/missed deposits). Reported as counts vs a fixed
threshold. States are workload cues, never a reliability verdict, and carry no color here."""
from __future__ import annotations


def run_trust(deposits_df, threshold: float) -> dict:
    cols = set(getattr(deposits_df, "columns", []))
    if deposits_df is None or "confidence" not in cols or len(deposits_df) == 0:
        return {"total": 0, "below": 0, "state": "unavailable", "line": "confidence unavailable"}
    total = int(len(deposits_df))
    below = int((deposits_df["confidence"] < threshold).sum())
    if below == 0:
        return {"total": total, "below": 0, "state": "none_flagged",
                "line": f"all {total} deposits at or above the confidence-score threshold ({threshold:.2f})"}
    return {"total": total, "below": below, "state": "review",
            "line": f"{below} of {total} deposits below the confidence-score threshold ({threshold:.2f}) — review recommended"}
