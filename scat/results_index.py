"""On-disk results ledger (core — no agent/LLM deps).

Reads back the durable artifacts every analysis run writes (T2.1's ``run_manifest.json`` + the
``image_summary.csv``) to answer "what has already been analyzed?" so the agent can resume instead of
re-running, and can discover prior results without a hand-pasted path.

Two consumers: ``scan_folder_service`` attaches an ``already_analyzed`` summary (the recipe calls it
first every turn), and the ``list_analyses`` tool enumerates prior runs. Everything here is best-effort
and never raises into the caller.

Design notes (see docs/superpowers/specs/2026-07-15-scat-context-ledger.md):
- ``image_summary.csv``'s ``filename`` column is the image BASENAME only (``analyzer.py`` sets
  ``filename=image_path.name``), so per-image resume is inherently basename-level. The whole-dataset
  ``run_manifest.json`` fingerprint (``dataset.sha256`` over sorted ``relpath:size``) is the strong,
  ambiguity-immune guard used for "fully analyzed"; basename matching is only a guarded fallback.
- ``run_manifest.json`` is written LAST (after the CSVs), so its presence implies a finished run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .artifacts import RUN_MANIFEST as _MANIFEST, IMAGE_SUMMARY as _SUMMARY

_PENDING_CAP = 50


@dataclass(frozen=True)
class AnalysisRecord:
    results_dir: str            # resolved abspath
    status: str                 # "complete" | "partial"  (unreadable dirs go to the skip channel)
    created_at: str | None
    dataset_path: str | None    # resolved; from manifest.dataset.path (None when the manifest is missing)
    dataset_sha256: str | None
    n_images: int
    analyzed_basenames: frozenset[str]
    basename_dupes: bool        # image_summary.csv had duplicate 'filename' rows (delta math unsafe)
    groups: list[str]           # sorted labels from manifest.grouping.mapping ([] if ungrouped)
    group_column: str | None
    model: dict | None
    detection: dict | None
    warnings: list[str]


# --------------------------------------------------------------------------- helpers

def _resolve(p) -> Path:
    try:
        return Path(p).expanduser().resolve()
    except OSError:
        return Path(p)


def _norm_roots(roots) -> list[Path]:
    """Resolve, drop falsy/empty/non-existent, dedupe (stable order)."""
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots or []:
        if not r:
            continue
        rp = _resolve(r)
        key = str(rp)
        if key in seen or not rp.exists():
            continue
        seen.add(key)
        out.append(rp)
    return out


def _parse_ts(s) -> float:
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _read_manifest(results_dir: Path) -> dict | None:
    try:
        return json.loads((results_dir / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_basenames(results_dir: Path) -> tuple[frozenset[str], bool] | None:
    """Return (basename set, had_duplicate_rows) or None if the CSV is missing/unreadable."""
    import pandas as pd
    try:
        df = pd.read_csv(results_dir / _SUMMARY)
    except (OSError, ValueError):
        return None
    if "filename" not in df.columns:
        return None
    names = [str(x) for x in df["filename"].tolist()]
    return frozenset(names), len(names) != len(set(names))


def _is_result_dir(d: Path) -> bool:
    return (d / _MANIFEST).is_file() or (d / _SUMMARY).is_file()


def _iter_result_dirs(root: Path):
    """root itself (if it is a results dir) plus its DIRECT children that are. Direct children only —
    results dirs are siblings of inputs, never nested — so this never walks a deep tree."""
    if _is_result_dir(root):
        yield root
    try:
        children = sorted(root.iterdir())
    except OSError:
        return
    for child in children:
        if child.is_dir() and _is_result_dir(child):
            yield child


def _record_for(results_dir: Path) -> tuple[AnalysisRecord | None, str | None]:
    """Build a record for one results dir. Returns (record, None) or (None, skip_reason)."""
    m = _read_manifest(results_dir)
    bn = _read_basenames(results_dir)
    if bn is None and m is None:
        return None, "no readable run_manifest.json or image_summary.csv"
    basenames, dupes = bn if bn is not None else (frozenset(), False)
    dataset = (m or {}).get("dataset") or {}
    grouping = (m or {}).get("grouping") or {}
    mapping = grouping.get("mapping") or {}
    groups = sorted({str(v) for v in mapping.values()}) if mapping else []
    dpath = dataset.get("path")
    return AnalysisRecord(
        results_dir=str(results_dir),
        status="complete" if m is not None else "partial",   # manifest is written last ⇒ finished run
        created_at=(m or {}).get("created_at"),
        dataset_path=str(_resolve(dpath)) if dpath else None,
        dataset_sha256=dataset.get("sha256"),
        n_images=int(dataset.get("n_images", len(basenames))),
        analyzed_basenames=basenames,
        basename_dupes=dupes,
        groups=groups,
        group_column=grouping.get("column"),
        model=(m or {}).get("model"),
        detection=(m or {}).get("detection"),
        warnings=list((m or {}).get("warnings") or []),
    ), None


# --------------------------------------------------------------------------- public API

def find_analyses_with_skips(search_roots):
    """Discover prior runs under the given roots. Returns (records, skipped) where skipped is a list of
    {dir, reason} for dirs that looked like results but couldn't be read (surfaced, not silently dropped)."""
    records: list[AnalysisRecord] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    for root in _norm_roots(search_roots):
        for d in _iter_result_dirs(root):
            key = str(_resolve(d))
            if key in seen:
                continue
            seen.add(key)
            rec, reason = _record_for(d)
            if rec is not None:
                records.append(rec)
            else:
                skipped.append({"dir": key, "reason": reason})
    records.sort(key=lambda r: (_parse_ts(r.created_at), r.results_dir))
    return records, skipped


def find_analyses(search_roots) -> list[AnalysisRecord]:
    return find_analyses_with_skips(search_roots)[0]


def run_brief(record: AnalysisRecord) -> dict:
    """JSON-safe subset for tool output. Includes model/detection so the agent can judge run compatibility."""
    return {"results_dir": record.results_dir, "dataset_path": record.dataset_path,
            "created_at": record.created_at, "n_images": record.n_images,
            "groups": record.groups, "group_column": record.group_column,
            "model": record.model, "detection": record.detection, "status": record.status}


def analysis_status(folder, *, image_paths=None, search_roots=None) -> dict:
    """Analysed-vs-pending delta for one target folder, in two tiers of confidence.

    Tier 1 (strong): if a discovered run's whole-dataset fingerprint equals the current image set's, the
    set was analyzed exactly once → complete/verified. Tier 2 (guarded): otherwise a basename seen/not-seen
    split, but only when basenames are unambiguous — duplicate basenames yield status "ambiguous" with no
    numeric counts (never emit an unsafe split)."""
    from . import manifest as _manifest
    from .pipeline import list_images

    if image_paths is not None:
        current = [Path(p) for p in image_paths]
    else:
        try:
            current = list_images(folder)
        except OSError:
            current = []
    n_current = len(current)
    if n_current == 0:
        return {"status": "empty", "folder": str(folder)}

    roots = search_roots if search_roots is not None else [_resolve(folder).parent]
    target = _resolve(folder)
    records = [r for r in find_analyses(roots)
               if r.dataset_path and _resolve(r.dataset_path) == target]
    runs = [run_brief(r) for r in records]

    # Tier 1 — fingerprint-verified complete.
    fp = _manifest.dataset_fingerprint([str(p) for p in current]).get("sha256")
    verified = [r for r in records if r.dataset_sha256 and r.dataset_sha256 == fp]
    if verified:
        return {"status": "complete", "verified": True, "n_current": n_current,
                "n_analyzed": n_current, "n_pending": 0,
                "results_dir": verified[-1].results_dir, "runs": runs}

    # Tier 2 — guarded basename delta.
    cur_bn = [p.name for p in current]
    if len(set(cur_bn)) != len(cur_bn) or any(r.basename_dupes for r in records):
        return {"status": "ambiguous",
                "reason": "duplicate image basenames — cannot map prior results by basename",
                "n_current": n_current, "runs": runs}
    seen: set[str] = set()
    for r in records:
        seen |= r.analyzed_basenames
    pending_p = [p for p in current if p.name not in seen]      # keep the real paths, not just names
    n_analyzed = n_current - len(pending_p)
    return {"status": "partial" if records else "none", "verified": False,
            "n_current": n_current, "n_analyzed": n_analyzed, "n_pending": len(pending_p),
            "pending": [p.name for p in pending_p][:_PENDING_CAP],
            "pending_paths": [str(p) for p in pending_p][:_PENDING_CAP],   # pass THESE to analyze_folder
            "pending_truncated": len(pending_p) > _PENDING_CAP,
            "latest_results_dir": records[-1].results_dir if records else None, "runs": runs,
            "note": "analyzed = basename seen in a prior run; runs may use different params — not "
                    "reusable for whole-experiment stats without combine_results"}
