"""Group-metadata plumbing shared by the CLI and the agent tools.

Group *inference* is intentionally NOT done here — the LLM agent reads the
filenames (from scan_folder) and decides the grouping itself (arbitrary condition
names, domain knowledge, ambiguity handling). The only thing kept deterministic is
the data-safety invariant below (SCAT joins group metadata on the image basename,
so duplicate basenames are unsafe) and the mapping -> DataFrame conversion.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd


def duplicate_basenames(paths) -> list[str]:
    """Basenames that collide (case-insensitive) across the given paths."""
    lc = Counter(Path(p).name.lower() for p in paths)
    return sorted({Path(p).name.lower() for p in paths if lc[Path(p).name.lower()] > 1})


def build_group_metadata(mapping: dict, n_flies: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """{basename: group|None} -> (DataFrame[filename, group(, n_flies)], ['group']). None/'' -> 'ungrouped'.

    n_flies (optional) is a {basename: count} map; its column rides into image_summary.csv via the
    metadata merge (generate_film_summary), enabling per-fly normalization downstream."""
    rows = [{"filename": f, "group": (g if g else "ungrouped")} for f, g in mapping.items()]
    df = pd.DataFrame(rows)
    if n_flies:
        df["n_flies"] = df["filename"].map(lambda f: n_flies.get(f))
    return df, ["group"]
