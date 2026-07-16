"""Reproducibility manifest primitives (core — no agent/GUI deps).

Two consumers: `analyze_folder_service` writes a `run_manifest.json` into every results dir (a
self-describing record that travels with the outputs — the basis for a "methods" section and the
seed of future result bundles), and the agent provenance log writes a session header. Everything
here is best-effort and never raises into the analysis path.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from scat import metrics as _metrics

from .artifacts import RUN_MANIFEST

SCHEMA = "scat.run_manifest/1"


@lru_cache(maxsize=1)
def run_context() -> dict:
    """Static run context: SCAT version, git commit (best-effort), python, platform. Cached —
    computed once per process (the git subprocess never runs in a per-image loop)."""
    try:
        import scat
        version = getattr(scat, "__version__", "unknown")
    except Exception:
        version = "unknown"
    return {
        "scat_version": version,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv0": Path(sys.argv[0]).name if sys.argv else "",
    }


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parent.parent
        # Only report a commit when SCAT itself IS the git root — not when it's pip-installed
        # inside some other repo (git rev-parse otherwise walks up and records that repo's HEAD).
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                            capture_output=True, text=True, timeout=3)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
            return None
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None   # not a git checkout (installed/packaged), git missing, or timeout


def sha256_file(path, _chunk: int = 1 << 20) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def dataset_fingerprint(image_paths) -> dict:
    """Order-independent, content-agnostic fingerprint: sha256 over sorted 'relpath:size' lines,
    where relpath is relative to the selection's common root — so same-named files in different
    subfolders don't collide (Codex F1), while staying portable if the whole tree moves. Cheap
    (stat, not bytes) and stable regardless of selection order."""
    import os
    strs = [str(p) for p in image_paths]
    try:
        root = os.path.commonpath([str(Path(s).parent) for s in strs]) if strs else ""
    except (ValueError, OSError):
        root = ""   # mixed drives / relative-vs-absolute — fall back to basenames
    items = []
    for s in strs:
        try:
            size = Path(s).stat().st_size
        except OSError:
            size = -1
        try:
            rel = os.path.relpath(s, root) if root else Path(s).name
        except (ValueError, OSError):
            rel = Path(s).name
        items.append((rel, size))
    items.sort()
    h = hashlib.sha256()
    for rel, size in items:
        h.update(f"{rel}:{size}\n".encode("utf-8"))
    return {"n_images": len(items), "sha256": h.hexdigest(),
            "sample": [r for r, _ in items[:10]]}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _abspath(p) -> str:
    """Store an ABSOLUTE dataset path: a relative one would later be resolved against a different
    process cwd and silently fail to match the same folder on a resume (results_index)."""
    try:
        return str(Path(p).resolve())
    except OSError:
        return str(p)


def write_run_manifest(output_dir, *, path=None, image_paths, model_type=None, model_path=None,
                       circularity=None, groups=None, group_column=None, detection=None,
                       warnings=None, primary_metric=None, normalization=None,
                       confidence_threshold=None) -> dict:
    """Write <output_dir>/run_manifest.json describing this analysis run. Best-effort (OSError
    is swallowed) and additive — a new sidecar file that never touches the CSV outputs."""
    def _thr(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return _metrics.DEFAULT_THRESHOLD
    manifest = {
        "schema": SCHEMA,
        "created_at": _now_iso(),
        **run_context(),
        "dataset": {"path": _abspath(path) if path is not None else None,
                    **dataset_fingerprint(image_paths)},
        # classifier settings (type/model/circularity) vs detector knobs are kept separate
        "model": {"type": model_type,
                  "path": str(model_path) if model_path else None,
                  "sha256": sha256_file(model_path) if model_path else None,
                  "circularity": circularity},
        "grouping": ({"column": group_column, "mapping": groups} if groups else None),
        "detection": dict(detection) if detection else {},
        # the analysis contract — guarded so a bad value degrades to the registry default, never raises
        "analysis": {
            "primary_metric": _metrics.resolve_metric(primary_metric),
            "normalization": (normalization if normalization in _metrics.NORMALIZATIONS
                              else _metrics.DEFAULT_NORMALIZATION),
            "confidence_threshold": _thr(confidence_threshold),
        },
        "warnings": list(warnings) if warnings else [],
    }
    try:
        (Path(output_dir) / RUN_MANIFEST).write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass
    return manifest
