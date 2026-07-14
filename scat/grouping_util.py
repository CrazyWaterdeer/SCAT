"""Deterministic experimental-group inference from filenames / subfolders.

Groups are discovered *structurally* from the filename pattern, so ARBITRARY
condition names and ANY number of groups work — genotypes (wt, dnc, rut2080),
doses (10uM, 20uM), timepoints (0h, 6h, 24h), drug names, etc. — not just a
fixed control/treated vocabulary. The dominant biology convention
``<condition>_<replicate>`` is handled by stripping the trailing replicate/index
token; the remaining prefix is the condition label whatever it is.
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
import pandas as pd

from .pipeline import list_images

# Optional canonicalization hint used only as a LAST resort (structural detection
# comes first). Maps a known token -> a tidy label.
CONDITION_VOCAB: dict[str, str] = {
    "control": "control", "ctrl": "control", "ctl": "control", "wt": "wt",
    "wildtype": "control", "vehicle": "vehicle", "veh": "vehicle", "mock": "mock",
    "treated": "treated", "treatment": "treated", "treat": "treated",
    "mutant": "mutant", "mut": "mutant", "ko": "ko", "knockout": "ko", "rnai": "rnai",
}
_DELIMS = re.compile(r"[_\-\s.]+")
# a trailing replicate / image index: optional separator then digits at end of stem
_TRAILING_REPLICATE = re.compile(r"[ _\-.]*\d+$")


def _tokens(stem: str) -> list[str]:
    return [t for t in _DELIMS.split(stem) if t]


def _strip_replicate(stem: str) -> str:
    stripped = _TRAILING_REPLICATE.sub("", stem).strip(" _-.")
    return stripped or stem


def _single_cohort(names: list[str], warnings: list[str]) -> dict:
    return {"mapping": {n: "all" for n in names}, "basis": "single_cohort", "groups": ["all"],
            "confidence": "low", "unmatched": [], "warnings": warnings, "matched_tokens": []}


def _result(mapping: dict, basis: str, confidence: str,
            warnings: list[str], matched_tokens: list[str]) -> dict:
    groups = sorted({g for g in mapping.values() if g != "ungrouped"})
    unmatched = [f for f, g in mapping.items() if g == "ungrouped"]
    return {"mapping": mapping, "basis": basis, "groups": groups, "confidence": confidence,
            "unmatched": unmatched, "warnings": warnings, "matched_tokens": matched_tokens}


def _valid_partition(mapping: dict, n_files: int) -> bool:
    """A usable grouping: 2..(n-1) distinct labels and at least one has replicates."""
    counts = Counter(mapping.values())
    if not (2 <= len(counts) < n_files):
        return False
    return max(counts.values()) >= 2


def infer_groups_from_folder(path: str) -> dict:
    """Return {mapping:{basename:group}, basis, groups, confidence, unmatched, warnings, matched_tokens}."""
    images = list_images(path)
    root = Path(path)
    names = [p.name for p in images]
    n = len(names)
    dups = [x for x, c in Counter(names).items() if c > 1]

    # 1) Subfolders (highest priority) — arbitrary group names. Refuse on duplicate
    #    basenames (SCAT merges metadata on basename 'filename').
    sub_names = {p.parent.name for p in images if p.parent != root}
    if len(sub_names) >= 2:
        if dups:
            return _single_cohort(names, [
                f"duplicate basenames across subfolders {dups[:5]} — SCAT keys on basename, so "
                "subfolder grouping is unsafe here; flatten or rename files before grouping."])
        sub = {p.name: p.parent.name for p in images if p.parent != root}
        return _result(sub, "subfolder", "high", [], [])

    stems = {p.name: p.stem for p in images}

    # 2) Strip trailing replicate/index -> the remaining prefix is the condition label.
    #    Handles <condition>_<rep> with ANY condition name and ANY number of groups.
    prefix_map = {name: _strip_replicate(st) for name, st in stems.items()}
    if _valid_partition(prefix_map, n):
        balanced = all(c >= 2 for c in Counter(prefix_map.values()).values())
        conf = "high" if balanced else "medium"
        warn = [] if balanced else ["some groups have a single replicate — verify the grouping"]
        return _result(prefix_map, "filename_prefix", conf, warn, sorted(set(prefix_map.values())))

    # 3) Positional factor: filenames with the same token count; group by the varying,
    #    non-index position with the fewest levels (a two-factor design surfaces >1 factor).
    token_lists = {name: _tokens(st) for name, st in stems.items()}
    if len({len(t) for t in token_lists.values()}) == 1 and n >= 2:
        length = len(next(iter(token_lists.values())))
        candidates: list[tuple[int, int]] = []
        for i in range(length):
            vals = [token_lists[name][i] for name in names]
            counts = Counter(vals)
            if 2 <= len(counts) < n and max(counts.values()) >= 2:
                candidates.append((len(counts), i))
        if candidates:
            candidates.sort()
            i = candidates[0][1]
            factor_map = {name: token_lists[name][i] for name in names}
            conf = "high" if len(candidates) == 1 else "medium"
            warn = ([] if len(candidates) == 1
                    else [f"{len(candidates)} varying factors detected; grouped by the primary one — "
                          "supply a metadata CSV for a multi-factor design"])
            return _result(factor_map, "filename_factor", conf, warn, sorted(set(factor_map.values())))

    # 4) Vocabulary match (canonical, last resort).
    vocab_map: dict[str, str] = {}
    matched: set[str] = set()
    for name, st in stems.items():
        g = None
        for t in _tokens(st.lower()):
            if t in CONDITION_VOCAB:
                g = CONDITION_VOCAB[t]; matched.add(t); break
        vocab_map[name] = g or "ungrouped"
    if len({g for g in vocab_map.values() if g != "ungrouped"}) >= 2:
        unmatched_n = sum(1 for g in vocab_map.values() if g == "ungrouped")
        conf = "high" if unmatched_n == 0 else "medium"
        warn = [f"{unmatched_n} file(s) matched no condition token -> 'ungrouped'"] if unmatched_n else []
        return _result(vocab_map, "filename_vocab", conf, warn, sorted(matched))

    # 5) Nothing usable -> single cohort.
    return _single_cohort(names, [
        "no group structure detected from filenames; single cohort (no comparison). "
        "Use subfolders per condition or a <condition>_<replicate> naming, or supply a metadata CSV."])


def build_group_metadata(mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    """{basename: group|None} -> (DataFrame[filename, group], ['group']). None/'' -> 'ungrouped'."""
    rows = [{"filename": f, "group": (g if g else "ungrouped")} for f, g in mapping.items()]
    return pd.DataFrame(rows), ["group"]
