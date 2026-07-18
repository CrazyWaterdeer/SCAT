"""Combine compatible SCAT results dirs into one (core — no agent/LLM deps).

Enables correct whole-experiment stats after a partial/resumed analysis: a full run + a pending-only
run of the SAME folder produce two results dirs; ``combine_results_service`` merges their CSVs into one
dir (with a ``run_manifest.json`` superset) that ``run_statistics_service`` / ``generate_report_service``
consume normally.

It **refuses** (``ValueError``) rather than silently producing a wrong merge: every source must share the
same dataset folder and identical ``model`` + ``detection`` params + ``grouping.column``, and any image
present in more than one source must have byte-identical rows in every source that has it (so a genuine
re-analysis with different results can't be quietly stitched together).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from . import manifest as _manifest
from .config import get_timestamped_output_dir
from .artifacts import IMAGE_SUMMARY, ALL_DEPOSITS, RUN_MANIFEST


def _resolve(p) -> str:
    try:
        return str(Path(p).expanduser().resolve())
    except OSError:
        return str(p)


def _load(d):
    """Return (resolved_dir, manifest|None, summary_df, deposits_df|None). Raises if no image_summary."""
    dd = Path(d)
    m = None
    mp = dd / RUN_MANIFEST
    if mp.is_file():
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = None
    sp = dd / IMAGE_SUMMARY
    if not sp.is_file():
        raise ValueError(f"{d}: no image_summary.csv")
    summary = pd.read_csv(sp)
    if "filename" not in summary.columns:
        raise ValueError(f"{d}: image_summary.csv has no 'filename' column")
    dp = dd / ALL_DEPOSITS
    deposits = pd.read_csv(dp) if dp.is_file() else None
    return _resolve(d), m, summary, deposits


def _rows_for(df, col, fn):
    if df is None:
        return None
    sub = df[col == fn]                       # col = precomputed str filename column (no re-cast)
    return sub.sort_index(axis=1).sort_values(list(sub.columns)).reset_index(drop=True)


def combine_results_service(results_dirs, output_dir: Optional[str] = None) -> dict:
    if not results_dirs or len(results_dirs) < 2:
        raise ValueError("combine_results needs at least 2 results dirs")
    loaded = [_load(d) for d in results_dirs]
    manifests = [m for _, m, _, _ in loaded]
    summaries = [s for _, _, s, _ in loaded]
    deposits = [dep for _, _, _, dep in loaded]

    # --- compatibility gate on manifests (refuse on any mismatch) ---
    if any(m is None for m in manifests):
        raise ValueError("cannot combine: a source is missing/unreadable run_manifest.json (params unknown)")

    def _dspath(m):
        p = (m.get("dataset") or {}).get("path")
        return _resolve(p) if p else None

    dpaths = {_dspath(m) for m in manifests}
    if None in dpaths or len(dpaths) != 1:
        raise ValueError(f"cannot combine: sources are from different datasets: {sorted(x for x in dpaths if x)}")
    if len({json.dumps(m.get("model"), sort_keys=True, default=str) for m in manifests}) != 1:
        raise ValueError("cannot combine: sources used different model settings")
    if len({json.dumps(m.get("detection"), sort_keys=True, default=str) for m in manifests}) != 1:
        raise ValueError("cannot combine: sources used different detection settings")
    gcols = {((m.get("grouping") or {}) or {}).get("column") for m in manifests}
    if len(gcols) != 1:
        raise ValueError("cannot combine: sources used different grouping columns")
    gcol = next(iter(gcols))

    # --- overlap check: a shared basename must be byte-identical across every source that has it ---
    # Precompute each source's str filename column + set ONCE (the per-duplicate loop below would
    # otherwise rebuild them repeatedly → O(dupes × sources × rows)).
    summ_fn = [s["filename"].astype(str) for s in summaries]
    summ_sets = [set(f) for f in summ_fn]
    dep_fn = [d["filename"].astype(str) if d is not None else None for d in deposits]
    dep_sets = [set(f) if f is not None else set() for f in dep_fn]

    counts = Counter()
    for fset in summ_sets:
        counts.update(fset)
    for fn in [f for f, c in counts.items() if c > 1]:
        srows = [_rows_for(s, sf, fn) for s, sf, fset in zip(summaries, summ_fn, summ_sets) if fn in fset]
        if any(not srows[0].equals(o) for o in srows[1:]):
            raise ValueError(f"cannot combine: image '{fn}' has differing image_summary rows between sources")
        drows = [_rows_for(d, df, fn) for d, df, dset in zip(deposits, dep_fn, dep_sets) if d is not None and fn in dset]
        if any(not drows[0].equals(o) for o in drows[1:]):
            raise ValueError(f"cannot combine: image '{fn}' has differing deposit rows between sources")

    # --- build combined, keeping the first source per filename (overlaps proven identical above) ---
    seen: set[str] = set()
    keep_summ, keep_dep = [], []
    for (_, _, s, d) in loaded:
        fns = s["filename"].astype(str)
        new = fns[~fns.isin(seen)]
        kept_here = set(new)
        keep_summ.append(s[fns.isin(kept_here)])
        if d is not None and "filename" in d.columns:
            keep_dep.append(d[d["filename"].astype(str).isin(kept_here)])
        seen |= kept_here
    combined_summary = pd.concat(keep_summ, ignore_index=True)
    combined_deposits = pd.concat(keep_dep, ignore_index=True) if keep_dep else None

    # --- write merged dir + a run_manifest superset (discoverable + consumable) ---
    out = Path(output_dir) if output_dir else get_timestamped_output_dir(
        Path(results_dirs[0]).parent, "results_combined")
    out.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(out / IMAGE_SUMMARY, index=False)
    if combined_deposits is not None:
        combined_deposits.to_csv(out / ALL_DEPOSITS, index=False)

    names = combined_summary["filename"].astype(str)
    groups = (sorted(set(combined_summary[gcol].dropna().astype(str)))
              if gcol and gcol in combined_summary.columns else [])
    mapping = (dict(zip(names, combined_summary[gcol].astype(str)))
               if gcol and gcol in combined_summary.columns else None)
    dataset_path = next(iter(dpaths))
    # If the merge reconstitutes the WHOLE source folder (the common resume case: full run +
    # pending run == every image), recompute the real whole-dataset fingerprint so a later scan
    # recognizes this combined dir as fingerprint-verified complete. Otherwise leave it None (honest:
    # a partial union is not the whole folder). Best-effort — the folder may have moved.
    sha = None
    try:
        from .pipeline import list_images
        folder_imgs = list_images(dataset_path)
        folder_bn = [Path(p).name for p in folder_imgs]
        if folder_bn and set(folder_bn) == set(names) and len(folder_bn) == len(set(folder_bn)):
            sha = _manifest.dataset_fingerprint([str(p) for p in folder_imgs]).get("sha256")
    except Exception:
        sha = None
    man = {
        "schema": _manifest.SCHEMA, "created_at": _manifest._now_iso(), **_manifest.run_context(),
        "dataset": {"path": dataset_path, "n_images": int(names.nunique()),
                    "sha256": sha, "sample": sorted(set(names))[:10]},
        "model": manifests[0].get("model"), "detection": manifests[0].get("detection"),
        "grouping": ({"column": gcol, "mapping": mapping} if mapping else None),
        "warnings": [], "combined_from": [dir_ for dir_, _, _, _ in loaded],
    }
    try:
        (out / RUN_MANIFEST).write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass
    return {"output_dir": str(out), "n_images": int(names.nunique()),
            "groups": groups, "sources": [dir_ for dir_, _, _, _ in loaded]}
