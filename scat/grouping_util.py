"""Deterministic experimental-group inference from filenames / subfolders.

Groups are discovered *structurally* so ARBITRARY condition names and ANY number
of groups work — genotypes (rut2080, dnc1, wt), doses (10uM, 0.5uM), timepoints
(0h, 6h, 24h), etc. — not a fixed control/treated vocabulary.

Safety notes (SCAT joins group metadata on the image BASENAME, and group
statistics need >=2 images per group):
- Any duplicate basename (case-insensitive) makes the basename join ambiguous, so
  we REFUSE grouping entirely in that case.
- Label inference is separated from statistical usability: correct labels with
  single-replicate groups are still returned, with a warning — not silently
  dropped or mis-grouped.
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
import pandas as pd

from .pipeline import list_images

# A trailing replicate/index: a separator (NOT '.', which is decimal/extension),
# an optional replicate word (rep/r/n), optional leading zeros, then digits.
_TRAILING_REPLICATE = re.compile(r"[ _\-]+(?:rep|r|n)?0*\d+$", re.IGNORECASE)
# Delimiters for the positional-factor fallback (keep '.' out so decimals stay whole).
_DELIMS = re.compile(r"[ _\-]+")


def _tokens(stem: str) -> list[str]:
    return [t for t in _DELIMS.split(stem) if t]


def _strip_replicate(stem: str) -> str:
    """Remove a trailing replicate token, preserving the rest of the stem verbatim.

    'rut2080_1' -> 'rut2080'; 'wt_rep1' -> 'wt'; '0.5uM_1' -> '0.5uM';
    'rut2080' -> 'rut2080' (no separator+digit tail, so nothing is stripped).
    """
    m = _TRAILING_REPLICATE.search(stem)
    if m and m.start() > 0:
        return stem[:m.start()]
    return stem


def _single_cohort(names: list[str], warnings: list[str]) -> dict:
    return {"mapping": {n: "all" for n in names}, "basis": "single_cohort", "groups": ["all"],
            "confidence": "low", "unmatched": [], "warnings": warnings, "matched_tokens": []}


def _result(mapping: dict, basis: str, confidence: str, warnings: list[str]) -> dict:
    groups = sorted({g for g in mapping.values() if g != "ungrouped"})
    unmatched = [f for f, g in mapping.items() if g == "ungrouped"]
    counts = Counter(g for g in mapping.values() if g != "ungrouped")
    singletons = sorted(g for g, c in counts.items() if c < 2)
    if singletons and confidence == "high":
        confidence = "medium"
    if singletons:
        warnings = warnings + [f"single-replicate group(s) {singletons[:5]} — statistics need >=2 images per group"]
    return {"mapping": mapping, "basis": basis, "groups": groups, "confidence": confidence,
            "unmatched": unmatched, "warnings": warnings, "matched_tokens": sorted(set(mapping.values()) - {"ungrouped"})}


def _is_partition(mapping: dict, n_files: int) -> bool:
    """2..(n-1) distinct labels (labels may be singletons; stats-readiness is separate)."""
    return 2 <= len(set(mapping.values())) < n_files


def infer_groups_from_folder(path: str) -> dict:
    """Return {mapping:{basename:group}, basis, groups, confidence, unmatched, warnings, matched_tokens}."""
    images = list_images(path)
    root = Path(path)
    names = [p.name for p in images]
    n = len(names)

    # 0) Global duplicate-basename guard (case-insensitive) — the join is by basename.
    lower_counts = Counter(nm.lower() for nm in names)
    dups = sorted({nm.lower() for nm in names if lower_counts[nm.lower()] > 1})
    if dups:
        return _single_cohort(names, [
            f"duplicate basenames {dups[:5]} — SCAT joins on basename, so grouping is unsafe; "
            "rename files to unique names or supply a metadata CSV."])

    # 1) Subfolders — only when EVERY image is in a subfolder (no root-level images),
    #    so no file is silently left ungrouped. Arbitrary group names.
    root_imgs = [p for p in images if p.parent == root]
    sub_names = {p.parent.name for p in images if p.parent != root}
    if not root_imgs and len(sub_names) >= 2:
        sub = {p.name: p.parent.name for p in images if p.parent != root}
        return _result(sub, "subfolder", "high", [])

    stems = {p.name: p.stem for p in images}

    # 2) Strip a trailing replicate/index -> the remaining prefix is the condition,
    #    whatever it is. Handles <condition>_<replicate> for arbitrary condition names.
    prefix_map = {name: _strip_replicate(st) for name, st in stems.items()}
    if _is_partition(prefix_map, n):
        return _result(prefix_map, "filename_prefix", "high", [])

    # 3) Positional factor: filenames with the same token count; group by the varying,
    #    non-index position. Multiple varying factors => factorial design => low confidence.
    token_lists = {name: _tokens(st) for name, st in stems.items()}
    if len({len(t) for t in token_lists.values()}) == 1 and n >= 2:
        length = len(next(iter(token_lists.values())))
        candidates: list[tuple[int, int]] = []
        for i in range(length):
            vals = [token_lists[name][i] for name in names]
            distinct = set(vals)
            if 2 <= len(distinct) < n:  # varying, not a per-file id
                candidates.append((len(distinct), i))
        if candidates:
            candidates.sort()
            i = candidates[0][1]
            factor_map = {name: token_lists[name][i] for name in names}
            if len(candidates) == 1:
                return _result(factor_map, "filename_factor", "high", [])
            return _result(factor_map, "filename_factor", "low", [
                f"{len(candidates)} varying factors in the filenames (factorial design) — grouped by the "
                "one with fewest levels; supply a metadata CSV to model the full design."])

    # 4) Nothing usable -> single cohort (label inference failed).
    return _single_cohort(names, [
        "no group structure detected from filenames; single cohort (no comparison). "
        "Use one subfolder per condition, a <condition>_<replicate> naming, or a metadata CSV."])


def build_group_metadata(mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    """{basename: group|None} -> (DataFrame[filename, group], ['group']). None/'' -> 'ungrouped'."""
    rows = [{"filename": f, "group": (g if g else "ungrouped")} for f, g in mapping.items()]
    return pd.DataFrame(rows), ["group"]
