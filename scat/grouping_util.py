"""Deterministic experimental-group inference from filenames / subfolders."""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
import pandas as pd

from .pipeline import list_images

# token (lowercased) -> canonical group label
CONDITION_VOCAB: dict[str, str] = {
    "control": "control", "ctrl": "control", "ctl": "control", "wt": "control",
    "wildtype": "control", "vehicle": "control", "veh": "control", "mock": "control",
    "treated": "treated", "treatment": "treated", "treat": "treated",
    "mutant": "mutant", "mut": "mutant", "ko": "ko", "knockout": "ko",
    "rnai": "rnai", "drug": "drug", "exp": "experimental", "test": "experimental",
}
_DELIMS = re.compile(r"[_\-\s.]+")


def _tokens(stem: str) -> list[str]:
    return [t for t in _DELIMS.split(stem.lower()) if t]


def _single_cohort(names: list[str], warnings: list[str]) -> dict:
    return {"mapping": {n: "all" for n in names}, "basis": "single_cohort", "groups": ["all"],
            "confidence": "low", "unmatched": [], "warnings": warnings, "matched_tokens": []}


def infer_groups_from_folder(path: str) -> dict:
    """Return {mapping:{basename:group}, basis, groups, confidence, unmatched, warnings, matched_tokens}."""
    images = list_images(path)
    root = Path(path)
    names = [p.name for p in images]
    dups = [n for n, c in Counter(names).items() if c > 1]

    # 1) subfolder grouping (highest priority) — but refuse if duplicate basenames
    #    (SCAT merges metadata on basename 'filename', so duplicates would mis-join).
    #    Count subfolders from the full image list (a dict would collapse duplicates).
    sub_names = {p.parent.name for p in images if p.parent != root}
    if len(sub_names) >= 2:
        if dups:
            return _single_cohort(names, [
                f"duplicate basenames across subfolders {dups[:5]} — SCAT keys on basename, so "
                "subfolder grouping is unsafe here; flatten or rename files before grouping."])
        sub = {p.name: p.parent.name for p in images if p.parent != root}
        return {"mapping": sub, "basis": "subfolder", "groups": sorted(sub_names),
                "confidence": "high", "unmatched": [], "warnings": [], "matched_tokens": []}

    # 2) filename vocabulary grouping
    mapping: dict[str, str] = {}
    matched: set[str] = set()
    unmatched: list[str] = []
    for p in images:
        g = None
        for t in _tokens(p.stem):
            if t in CONDITION_VOCAB:
                g = CONDITION_VOCAB[t]; matched.add(t); break
        if g:
            mapping[p.name] = g
        else:
            mapping[p.name] = "ungrouped"; unmatched.append(p.name)
    groups = sorted(set(mapping.values()) - {"ungrouped"})
    if len(groups) >= 2:
        conf = "high" if not unmatched else "medium"
        warnings = [f"{len(unmatched)} file(s) matched no condition token -> 'ungrouped'"] if unmatched else []
        return {"mapping": mapping, "basis": "filename_vocab", "groups": groups,
                "confidence": conf, "unmatched": unmatched, "warnings": warnings,
                "matched_tokens": sorted(matched)}

    # 3) fallback: single cohort
    return _single_cohort(names, ["no group structure detected; single cohort (no comparison)"])


def build_group_metadata(mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    """{basename: group|None} -> (DataFrame[filename, group], ['group']). None/'' -> 'ungrouped'."""
    rows = [{"filename": f, "group": (g if g else "ungrouped")} for f, g in mapping.items()]
    return pd.DataFrame(rows), ["group"]
