"""SCAT pipeline services — plain Python, no agent/LLM deps.

The single canonical implementation the CLI, the @tool adapters, and (phase 2)
the GUI all call. Nothing here imports pydantic/anthropic/scat.agent/scat.tools.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd

from .detector import DepositDetector
from .classifier import ClassifierConfig
from .analyzer import Analyzer, ReportGenerator
from .config import get_timestamped_output_dir
from .artifacts import IMAGE_SUMMARY, ALL_DEPOSITS

IMAGE_GLOBS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def list_images(path: str) -> list[Path]:
    from .pathutils import normalize_path
    p = Path(normalize_path(path))
    if p.is_file():
        return [p]
    imgs: list[Path] = []
    for g in IMAGE_GLOBS:
        imgs += p.rglob(g)
    return sorted(set(imgs))


def resolve_model_type(model_type: Optional[str], model_path: Optional[str]) -> tuple[str, Optional[str]]:
    """Canonical default: rf if a model file is available, else threshold. RF chosen without a path
    (e.g. GUI Method=Random Forest with the model-file field blank, or `scat analyze --model-type rf`)
    falls back to the bundled models/model_rf.pkl — otherwise the classifier fails to load and every
    image reports 0 deposits."""
    default_model = Path(__file__).parent.parent / "models" / "model_rf.pkl"
    if model_type:
        if model_type == "rf" and not model_path and default_model.exists():
            return "rf", str(default_model)
        return model_type, model_path
    if model_path:
        return "rf", model_path
    if default_model.exists():
        return "rf", str(default_model)
    return "threshold", None


@dataclass
class AnalyzeResult:
    output_dir: str
    n_images: int
    n_normal: int
    n_rod: int
    n_artifact: int
    n_failed: int
    groups: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resolve_n_flies(n_flies, images) -> Optional[dict]:
    """Normalize the n_flies argument to a {basename: positive-int} map (or None). An int/float applies
    to EVERY image; a dict is keyed by basename (a bare name or a full path both work). Non-positive /
    unparseable counts are dropped."""
    if n_flies is None:
        return None
    if isinstance(n_flies, (int, float)):
        v = int(n_flies)
        return {p.name: v for p in images} if v > 0 else None
    if isinstance(n_flies, dict):
        norm: dict = {}
        for k, val in n_flies.items():
            try:
                iv = int(val)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                norm[Path(str(k)).name] = iv
        return norm or None
    return None


_SCAN_NAME_CAP = 500


def scan_folder_service(path: str) -> dict:
    imgs = list_images(path)
    exts = sorted({p.suffix.lower() for p in imgs})
    root = Path(path)
    subdirs = sorted({p.parent.name for p in imgs if p.parent != root})
    # Return ALL basenames so the agent can read them and decide the grouping itself
    # (subfolder is exposed per file so the agent can group by folder when present).
    files = [{"filename": p.name, "subfolder": (p.parent.name if p.parent != root else None)} for p in imgs]
    truncated = len(files) > _SCAN_NAME_CAP
    result = {"path": str(path), "n_images": len(imgs), "extensions": exts,
              "subfolders": subdirs, "files": files[:_SCAN_NAME_CAP],
              "files_truncated": truncated}
    # Resume support (T3.1): report which of these images were already analyzed, from on-disk
    # results. Best-effort and additive — never break scan; scan output is not in the parity gate.
    try:
        from . import results_index
        result["already_analyzed"] = results_index.analysis_status(path)
    except Exception:
        pass
    return result


def analyze_folder_service(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                           model_path: Optional[str] = None, min_area: int = 20, max_area: int = 10000,
                           edge_margin: int = 20, circularity: float = 0.6,
                           sensitive_mode: bool = False, unet_model_path: Optional[str] = None,
                           annotate: bool = True, visualize: bool = False, spatial: bool = False,
                           significance_mode: str = 'auto', show_ns: bool = False,
                           condition_matrix: Optional[dict] = None,
                           parallel: bool = True, max_workers: int = 0, save_json: bool = True,
                           image_paths: Optional[list] = None, progress_callback=None,
                           ambient_progress: bool = False,
                           output_dir: Optional[str] = None,
                           primary_metric: Optional[str] = None,
                           normalization: Optional[str] = None,
                           confidence_threshold: Optional[float] = None,
                           palette=None, n_flies=None) -> AnalyzeResult:
    """Canonical folder analysis. Superset of every caller (CLI + GUI); every param
    beyond the original signature defaults to the value that reproduces the pre-slimdown
    CLI/parity behaviour, so tests/test_pipeline_parity.py (default kwargs) stays byte-identical.

    image_paths: analyse exactly this explicit list (the GUI is a multi-file picker); when
    None, rglob the folder as before. progress_callback(current, total) drives the GUI bar.
    """
    from .grouping_util import build_group_metadata, duplicate_basenames
    from .pathutils import normalize_path
    # Accept Windows or WSL path forms interchangeably (SCAT may run in either environment while the
    # images live in the other) — translate everything to what this OS can actually open.
    path = normalize_path(path)
    model_path = normalize_path(model_path) if model_path else model_path
    unet_model_path = normalize_path(unet_model_path) if unet_model_path else unet_model_path
    output_dir = normalize_path(output_dir) if output_dir else output_dir
    if image_paths is not None:
        image_paths = [normalize_path(p) for p in image_paths]
        # Resolve entries that were passed as bare basenames / folder-relative (e.g. an agent
        # forwarding scan_folder's pending list): try them relative to `path` before giving up, so a
        # resume doesn't fail just because the process cwd differs from the dataset folder.
        base = Path(path)
        images = []
        for p in image_paths:
            pp = Path(p)
            if not pp.is_absolute() and not pp.exists() and (base / p).exists():
                pp = base / p
            images.append(pp)
    else:
        images = list_images(path)
    if not images:
        raise ValueError(f"No images found in {path}")
    if groups:
        dups = duplicate_basenames(images)
        if dups:
            raise ValueError(
                f"Cannot group: duplicate basenames {dups[:5]} — SCAT joins group metadata on the "
                "image basename, so grouping would mis-join. Use unique filenames or a flat folder.")
    mtype, mpath = resolve_model_type(model_type, model_path)
    detector = DepositDetector(min_area=min_area, max_area=max_area, edge_margin=edge_margin,
                               sensitive_mode=sensitive_mode, unet_model_path=unet_model_path)
    cfg = ClassifierConfig(model_type=mtype, circularity_threshold=circularity, model_path=mpath)
    analyzer = Analyzer(detector=detector, classifier_config=cfg)

    metadata = None
    group_by = None
    group_names: list[str] = []
    warnings: list[str] = []
    # Resolve per-image fly counts: an int applies to every image; a {basename: count} map is keyed by
    # basename (matching how metadata joins). Rides into image_summary.csv so deposits/IOD normalize
    # per fly (fly_normalize) — the meaningful readout when vials hold different numbers of flies.
    n_flies_map = _resolve_n_flies(n_flies, images)
    # A per-fly run needs a count for EVERY image; a partial map (some images uncovered) silently falls
    # back to totals, so flag it. covered<total -> partial; empty map -> none.
    _covered = sum(1 for p in images if n_flies_map and p.name in n_flies_map)
    if n_flies_map and _covered < len(images):
        warnings.append(f"partial n_flies ({_covered}/{len(images)} images) — deposit/IOD stay per-image "
                        "totals, NOT fly-normalized; provide a count for every image to normalize per fly")
    if groups:
        metadata, group_by = build_group_metadata(groups, n_flies=n_flies_map)
        group_names = sorted(set(metadata["group"]) - {"ungrouped"})
        if len(group_names) < 2:
            warnings.append(f"stats skipped: {len(group_names)} group(s) — need >=2 to compare")
        elif not n_flies_map:
            warnings.append("no n_flies provided — deposit/IOD comparisons are per-image totals, NOT "
                            "fly-normalized (misleading if vials hold different numbers of flies)")
    elif n_flies_map:
        metadata = pd.DataFrame([{"filename": p.name, "n_flies": n_flies_map.get(p.name)} for p in images])

    # Compose the per-image callback. It always drives the caller's explicit callback (the GUI
    # Run button's Qt bar). It ALSO drives the process-global progress/cancel channel ONLY when
    # ambient_progress=True — the agent tool path opts in; the GUI Run path never does, so a chat
    # turn's open sink can't leak into (or cancel) a concurrent GUI Run analysis.
    _cb = progress_callback
    if ambient_progress:
        from . import progress as _pg

        def _cb(current, total):
            if progress_callback:
                progress_callback(current, total)
            _pg.report_progress(current, total, "analyzing images")
            _pg.raise_if_cancelled()

    results = analyzer.analyze_batch(images, metadata=metadata, progress_callback=_cb,
                                     parallel=parallel, max_workers=max_workers)
    out = Path(output_dir) if output_dir else get_timestamped_output_dir(Path(path).parent, "results")
    reporter = ReportGenerator(out)
    reports = reporter.save_all(results, metadata, group_by, save_json=save_json)
    n_failed = sum(1 for r in results if getattr(r, "failed", False))

    if annotate:
        from PIL import Image
        import numpy as np
        from . import parallel as _parallel
        ann_dir = out / "annotated"; ann_dir.mkdir(exist_ok=True)

        def _annotate_one(item):
            # Best-effort per image: a single bad/locked file must not discard the whole
            # completed run (annotate is default-ON in the GUI). Return an error string instead.
            img_path, res = item
            try:
                from .analyzer import to_rgb
                arr = np.array(to_rgb(Image.open(img_path)))   # composite alpha → visible annotations
                annotated = analyzer.generate_annotated_image(
                    arr, res.deposits, show_labels=True, skip_artifacts=True)
                Image.fromarray(annotated).save(ann_dir / f"{img_path.stem}_annotated.png")
                return None
            except Exception as e:
                return f"{img_path.name}: {e}"

        # Each image writes a distinct PNG, so order is irrelevant. Decode (PIL), draw
        # (cv2), and PNG encode all release the GIL, so a thread pool overlaps this
        # I/O-and-native-encode work without process overhead (no model needed here).
        todo = [(p, r) for p, r in zip(images, results) if r.n_total > 0]
        if len(todo) > 1:
            from concurrent.futures import ThreadPoolExecutor
            workers = max(1, min(_parallel.usable_cores(), len(todo), 8))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                ann_errs = [e for e in ex.map(_annotate_one, todo) if e]
        else:
            ann_errs = [e for e in (_annotate_one(item) for item in todo) if e]
        if ann_errs:
            warnings.append(f"{len(ann_errs)} image(s) could not be annotated (e.g. {ann_errs[0]})")

    # Validate the color override once (so it persists in the manifest even when visualize=False,
    # e.g. for a later generate_report) — a typo'd color is dropped with a warning, never crashes.
    clean_palette = _valid_palette(palette, warnings) if palette else None

    if visualize:
        try:
            from .visualization import generate_all_visualizations
            generate_all_visualizations(reports["film_summary"], reports["deposit_data"],
                                        out / "visualizations",
                                        group_by=group_by[0] if group_by else None,
                                        significance_mode=significance_mode, show_ns=show_ns,
                                        condition_matrix=condition_matrix,
                                        palette=clean_palette)
        except ImportError as e:
            warnings.append(f"visualizations skipped (missing deps): {e}")
        except Exception as e:  # never let a plotting error abort the analysis or the manifest write
            warnings.append(f"visualizations failed (analysis still completed): {e}")

    if spatial:
        try:
            import json
            import numpy as np
            from PIL import Image
            from .spatial import SpatialAnalyzer, aggregate_spatial_stats
            spatial_analyzer = SpatialAnalyzer()
            spatial_results = []
            for img_path, res in zip(images, results):
                _sz = Image.open(img_path).size  # (W, H) from the header — no full pixel decode
                shape = (_sz[1], _sz[0])
                spatial_results.append(spatial_analyzer.analyze(res.deposits, shape))
            agg = aggregate_spatial_stats(spatial_results)
            with open(out / "spatial_stats.json", "w") as f:
                json.dump(agg, f, default=str)
            try:
                from .visualization import generate_spatial_visualizations
                generate_spatial_visualizations(spatial_results, out / "visualizations")
            except Exception as e:
                warnings.append(f"spatial visualizations skipped: {e}")
        except Exception as e:  # spatial is best-effort; never abort the batch
            warnings.append(f"spatial analysis skipped: {e}")

    # Reproducibility sidecar: a self-describing manifest that travels with the results.
    # Additive (new file) — does not touch the CSVs the parity gate diffs.
    from . import manifest as _manifest
    _manifest.write_run_manifest(
        out, path=path, image_paths=[str(p) for p in images], model_type=mtype, model_path=mpath,
        circularity=circularity, groups=groups, group_column=(group_by[0] if group_by else None),
        detection={"min_area": min_area, "max_area": max_area, "edge_margin": edge_margin,
                   "sensitive_mode": sensitive_mode, "unet_model_path": unet_model_path},
        warnings=warnings,
        primary_metric=primary_metric, normalization=normalization,
        confidence_threshold=confidence_threshold, palette=clean_palette)

    summary = reports["film_summary"]
    return AnalyzeResult(
        output_dir=str(out), n_images=len(results),
        n_normal=int(summary["n_normal"].sum()), n_rod=int(summary["n_rod"].sum()),
        n_artifact=int(summary["n_artifact"].sum()), n_failed=n_failed,
        groups=group_names, warnings=warnings)


def run_statistics_service(results_dir: str, group_col: str = "group") -> dict:
    from .statistics import run_comprehensive_analysis
    rd = Path(results_dir)
    film = pd.read_csv(rd / IMAGE_SUMMARY)
    deposits = pd.read_csv(rd / ALL_DEPOSITS) if (rd / ALL_DEPOSITS).exists() else None
    if group_col not in film.columns:
        return {"skipped": True, "reason": f"column '{group_col}' not found"}
    # Exclude the 'ungrouped' sentinel so a run with 1 real group (+ ungrouped images) is skipped
    # instead of fabricating a comparison that treats 'ungrouped' as a second group.
    n_real = film.loc[film[group_col] != "ungrouped", group_col].dropna().nunique()
    if n_real < 2:
        return {"skipped": True, "reason": f"<2 groups in column '{group_col}'"}
    return run_comprehensive_analysis(film, deposits_df=deposits, group_column=group_col)


def generate_report_service(results_dir: str, statistical_results: Optional[dict] = None,
                            group_by: Optional[str] = None) -> str:
    from .report import generate_report
    rd = Path(results_dir)
    # Reproducibility sidecar carries the run's analysis contract (predeclared primary metric,
    # normalization, confidence threshold). Thread it into the report so the HTML reflects the
    # run's own choices rather than global config. Best-effort — a missing/bad manifest degrades
    # to an empty dict, never raises into the report path.
    import json
    analysis = {}
    mpath = rd / "run_manifest.json"
    if mpath.exists():
        try:
            analysis = (json.loads(mpath.read_text()).get("analysis") or {})
        except Exception:
            analysis = {}
    film = pd.read_csv(rd / IMAGE_SUMMARY)
    deposits = pd.read_csv(rd / ALL_DEPOSITS) if (rd / ALL_DEPOSITS).exists() else None
    # report expects the FLAT metrics mapping, not the whole run_comprehensive_analysis dict.
    metrics = None
    if statistical_results and not statistical_results.get("skipped"):
        metrics = statistical_results.get("basic", {}).get("metrics") or statistical_results
    # The report renders a Spatial Analysis section from spatial_stats; pick up the sidecar
    # analyze_folder_service(spatial=True) writes so the HTML matches the Results tab.
    spatial_stats = None
    sp = rd / "spatial_stats.json"
    if sp.exists():
        import json
        try:
            spatial_stats = json.loads(sp.read_text())
        except Exception:
            spatial_stats = None
    # Honor display overrides a prior rerender persisted into the manifest, so a later plain
    # generate_report renders the SAME group order / reference group / colors (not just the metric).
    return generate_report(film, output_dir=rd, deposit_data=deposits,
                           statistical_results=metrics, spatial_stats=spatial_stats,
                           group_by=group_by, format="html", analysis=analysis,
                           group_order=analysis.get("group_order"),
                           control_group=analysis.get("control_group"),
                           palette=analysis.get("palette"))


def _valid_palette(palette, warnings: list):
    """Filter a color override down to colors matplotlib can actually parse, so a typo'd hex/name
    can never crash a plot. Dict: drop bad entries. List: replace bad entries with None (which keeps
    that group on the default color) to preserve positional alignment. Returns the cleaned override
    (dict or list) or None if nothing valid remains."""
    try:
        from matplotlib.colors import is_color_like
    except Exception:
        return palette  # matplotlib absent — let downstream handle it
    if isinstance(palette, dict):
        out, bad = {}, []
        for k, c in palette.items():
            (out.__setitem__(k, c) if is_color_like(c) else bad.append(f"{k}={c!r}"))
        if bad:
            warnings.append(f"ignored unparseable palette color(s): {', '.join(bad)}")
        return out or None
    if isinstance(palette, (list, tuple)):
        out, bad = [], []
        for c in palette:
            if c and is_color_like(c):
                out.append(c)
            else:
                out.append(None)
                if c:
                    bad.append(repr(c))
        if bad:
            warnings.append(f"ignored unparseable palette color(s): {', '.join(bad)}")
        return out if any(out) else None
    warnings.append(f"palette must be a dict or list, got {type(palette).__name__} — ignored")
    return None


@dataclass
class RerenderResult:
    output_dir: str
    report_path: str
    primary_metric: str
    group_by: Optional[str]
    group_order: Optional[list]
    n_groups: int
    stats_recomputed: bool
    stats_skipped_reason: Optional[str]
    n_visualizations: int
    changed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def rerender_results_service(results_dir: str,
                             primary_metric: Optional[str] = None,
                             group_order: Optional[list] = None,
                             group_by: Optional[str] = None,
                             control_group: Optional[str] = None,
                             significance_mode: str = "auto",
                             show_ns: bool = False,
                             condition_matrix: Optional[dict] = None,
                             palette=None, n_flies=None,
                             regenerate_visualizations: bool = True) -> RerenderResult:
    """Re-render statistics + comparison plots + the HTML report from an EXISTING results dir
    WITHOUT re-detecting. Reads the on-disk image_summary.csv / all_deposits.csv (which may have been
    hand-edited during manual review) and rebuilds every DERIVED artifact from them. It NEVER runs the
    detector/classifier, so manual label corrections survive — this is the "review, then produce the
    outputs" and "re-graph the same detections differently" path.

    Optional overrides: primary_metric (the declared endpoint the report headlines — one of the six
    metrics keys), group_order (explicit x-axis order), control_group, significance_mode/show_ns
    (bracket style on the standalone plots), condition_matrix (factorial design table). Any override
    given is persisted into run_manifest.json's `analysis` block so a later plain generate_report
    stays consistent. Statistics are ALWAYS recomputed from the current CSVs so edits are reflected.

    Source-of-truth note: image_summary.csv is the FILM-level source of truth for the group statistics
    (per-image counts, rod_fraction, area/hue/IOD aggregates); all_deposits.csv drives the deposit-level
    distributions. The labeling GUI's edit-save rewrites BOTH in lock-step, so review edits are picked
    up here. This service does NOT re-derive image_summary.csv from all_deposits.csv — editing only the
    deposits CSV by hand would leave the film-level metrics stale.
    """
    import json
    from .pathutils import normalize_path
    rd = Path(normalize_path(results_dir))
    warnings: list[str] = []
    changed: list[str] = []

    film_path = rd / IMAGE_SUMMARY
    if not film_path.exists():
        raise FileNotFoundError(f"{IMAGE_SUMMARY} not found in {rd} — not a SCAT results dir")
    film = pd.read_csv(film_path)

    # Add/replace per-image fly counts on an existing run, then normalize per fly. This augments the
    # vial metadata (an n_flies column) — it does NOT change any detection — and is persisted so the
    # recomputed stats + rebuilt report read it. Enables "I already analyzed; now normalize per fly".
    if n_flies is not None and "filename" in film.columns:
        nmap = _resolve_n_flies(n_flies, [Path(str(f)) for f in film["filename"]])
        if nmap:
            film = film.copy()
            film["n_flies"] = film["filename"].map(lambda f: nmap.get(Path(str(f)).name))
            film.to_csv(film_path, index=False)
            changed.append("n_flies")
            missing = int(film["n_flies"].isna().sum())
            if missing:
                warnings.append(f"partial n_flies ({len(film) - missing}/{len(film)} images) — "
                                "deposit/IOD stay per-image totals until every image has a count")

    deposits = pd.read_csv(rd / ALL_DEPOSITS) if (rd / ALL_DEPOSITS).exists() else None

    mpath = rd / "run_manifest.json"
    manifest = {}
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text())
        except Exception:
            manifest = {}
    analysis = dict(manifest.get("analysis") or {})

    # Resolve grouping column: explicit arg > manifest grouping.column > 'group' if present.
    if group_by is None:
        group_by = (manifest.get("grouping") or {}).get("column")
    if group_by is None and "group" in film.columns:
        group_by = "group"
    if group_by is not None and group_by not in film.columns:
        warnings.append(f"group_by '{group_by}' not in {IMAGE_SUMMARY}; report will be ungrouped")
        group_by = None

    # Validate + apply the primary-metric override (resolve_metric silently falls back, so warn here).
    from . import metrics as _metrics
    if primary_metric is not None:
        if primary_metric not in _metrics.METRICS:
            warnings.append(f"unknown primary_metric '{primary_metric}' ignored; valid: {sorted(_metrics.METRICS)}")
        else:
            if analysis.get("primary_metric") != primary_metric:
                changed.append(f"primary_metric -> {primary_metric}")
            analysis["primary_metric"] = primary_metric
    pm = _metrics.resolve_metric(analysis.get("primary_metric"))

    if group_order:
        if analysis.get("group_order") != list(group_order):
            changed.append("group_order")
        analysis["group_order"] = list(group_order)
    if control_group:
        if analysis.get("control_group") != control_group:
            changed.append(f"control_group -> {control_group}")
        analysis["control_group"] = control_group
    if palette:
        # Drop colors matplotlib can't parse (typo guard) so a bad hex never crashes a plot.
        clean = _valid_palette(palette, warnings)
        if clean:
            # Warn about color keys that match no group here (otherwise the color silently vanishes).
            if isinstance(clean, dict) and group_by and group_by in film.columns:
                from .visualization import _display_label as _dl
                gvals = [str(g) for g in film[group_by].dropna().unique()]
                gdisp = {_dl(g) for g in gvals}
                for k in list(clean):
                    if str(k) not in gvals and str(k) not in gdisp:
                        warnings.append(f"palette color for '{k}' ignored — no matching group")
            if analysis.get("palette") != clean:
                changed.append("palette")
            analysis["palette"] = clean
    # Re-consume persisted display overrides so a BARE rerender (no display args) reproduces the last
    # look — order, reference group, and colors — instead of reverting to the automatic defaults.
    effective_order = list(group_order) if group_order else analysis.get("group_order")
    effective_control = control_group if control_group else analysis.get("control_group")
    effective_palette = analysis.get("palette")

    # Recompute statistics fresh from the (possibly hand-edited) CSVs.
    stats = (run_statistics_service(str(rd), group_col=group_by) if group_by
             else {"skipped": True, "reason": "no grouping column"})
    stats_skipped_reason = stats.get("reason") if stats.get("skipped") else None
    metrics_flat = None
    if not stats.get("skipped"):
        metrics_flat = stats.get("basic", {}).get("metrics") or stats
    n_groups = 0
    if group_by and group_by in film.columns:
        n_groups = int(film.loc[film[group_by] != "ungrouped", group_by].dropna().nunique())

    # Persist the updated analysis block into the manifest (additive — never touches the dataset
    # fingerprint / model / detection record). Only when a manifest already exists.
    if changed and mpath.exists() and manifest:
        manifest["analysis"] = analysis
        try:
            import os as _os
            tmp = mpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(manifest, indent=2, default=str))
            _os.replace(tmp, mpath)  # atomic swap: a torn write can never corrupt the real manifest
        except Exception as e:
            warnings.append(f"could not update run_manifest.json: {e}")
    elif changed and not mpath.exists():
        warnings.append("no run_manifest.json — overrides applied to this render but not persisted")

    # Regenerate the standalone comparison plots from the CSVs (no detection).
    n_viz = 0
    if regenerate_visualizations:
        vdir = rd / "visualizations"
        # Drop the previous PNGs first so a plot that is now skipped, renamed, or no longer applies
        # (e.g. group_by changed, condition_matrix dropped) cannot leave a STALE graph behind.
        if vdir.exists():
            for old in vdir.glob("*.png"):
                try:
                    old.unlink()
                except OSError:
                    pass
        try:
            from .visualization import generate_all_visualizations
            res = generate_all_visualizations(film, deposits, vdir,
                                               group_by=group_by, control_group=effective_control,
                                               significance_mode=significance_mode, show_ns=show_ns,
                                               condition_matrix=condition_matrix,
                                               group_order=effective_order, palette=effective_palette)
            n_viz = len(res)
        except ImportError as e:
            warnings.append(f"visualizations skipped (missing deps): {e}")
        except Exception as e:
            warnings.append(f"visualizations failed (report still rebuilt): {e}")

    # Rebuild the HTML report from the CSVs + fresh stats + overridden analysis (no detection).
    spatial_stats = None
    sp = rd / "spatial_stats.json"
    if sp.exists():
        try:
            spatial_stats = json.loads(sp.read_text())
        except Exception:
            spatial_stats = None
    from .report import generate_report as _gen
    report_path = _gen(film, output_dir=rd, deposit_data=deposits,
                       statistical_results=metrics_flat, spatial_stats=spatial_stats,
                       group_by=group_by, format="html", analysis=analysis,
                       group_order=effective_order, control_group=effective_control,
                       palette=effective_palette)
    changed.append("report.html")

    return RerenderResult(
        output_dir=str(rd), report_path=str(report_path), primary_metric=pm,
        group_by=group_by, group_order=effective_order, n_groups=n_groups,
        stats_recomputed=not stats.get("skipped"), stats_skipped_reason=stats_skipped_reason,
        n_visualizations=n_viz, changed=changed, warnings=warnings)


# --------------------------------------------------------------------------- model training
@dataclass
class TrainResult:
    output_path: str
    model_type: str
    n_samples: int
    class_counts: dict
    accuracy: Optional[float]
    cv_mean: Optional[float]
    cv_std: Optional[float]
    top_features: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _label_source_from_results_dir(rd: Path, warnings: list) -> Optional[tuple]:
    """(image_dir, label_dir) for a results dir: labels live in <rd>/deposits, images at the analyzed
    folder recorded in run_manifest.json (dataset.path). Returns None (with a warning) if either is
    unresolvable."""
    import json
    from .pathutils import normalize_path
    label_dir = rd / "deposits"
    if not label_dir.exists() or not any(label_dir.glob("*.labels.json")):
        warnings.append(f"{rd.name}: no deposits/*.labels.json to train on")
        return None
    img = None
    mp = rd / "run_manifest.json"
    if mp.exists():
        try:
            img = ((json.loads(mp.read_text()).get("dataset") or {}).get("path"))
        except Exception:
            img = None
    if not img:
        warnings.append(f"{rd.name}: run_manifest.json has no dataset.path; pass image_dir explicitly")
        return None
    img = Path(normalize_path(img))
    if img.is_file():
        img = img.parent           # a single-image analysis records the file; DataLoader wants the dir
    if not img.exists():
        warnings.append(f"{rd.name}: image folder '{img}' not found; pass image_dir explicitly")
        return None
    return (img, label_dir)


def train_model_service(results_dirs: Optional[list] = None, image_dir: Optional[str] = None,
                        label_dir: Optional[str] = None, output: Optional[str] = None,
                        model_type: str = "rf", n_estimators: int = 100,
                        cross_validate: bool = True) -> TrainResult:
    """Train (or retrain) a deposit classifier from LABELED data — the S5/S6 path.

    Sources are combined (union) so "update the existing model" = retrain on the union of the old
    labels + the new ones (there is NO warm-start; RFTrainer.train fits a fresh forest every time):
      - results_dirs: a list of SCAT results dirs whose deposits/*.labels.json carry the labels
        (after a manual review, these are the CORRECTED labels — the labeling GUI's edit-save writes
        them). Images are taken from each run's run_manifest.json dataset.path.
      - image_dir + label_dir: an explicit pair (CLI-style), e.g. a curated ground-truth set to fold in.

    output: where to write the model. If omitted, a timestamped file under the repo models/ dir (so the
    bundled models/model_rf.pkl is never overwritten unless the caller explicitly targets it — which is
    how you "update the active model"). Returns training metrics; raises ValueError if no labels load.
    """
    from .pathutils import normalize_path
    warnings: list = []
    sources: list = []
    for r in (results_dirs or []):
        pair = _label_source_from_results_dir(Path(normalize_path(str(r))), warnings)
        if pair:
            sources.append(pair)
    if image_dir:
        sources.append((Path(normalize_path(image_dir)),
                        Path(normalize_path(label_dir)) if label_dir else Path(normalize_path(image_dir))))
    if not sources:
        reason = ("; ".join(warnings)) if warnings else "none given"
        raise ValueError("no trainable label sources — pass results_dirs (with deposits/*.labels.json) "
                         f"and/or an explicit image_dir + label_dir. Details: {reason}")

    # Dedup by resolved (image_dir, label_dir) so the SAME labels passed twice are not double-weighted
    # (a real "update" union uses distinct sources; an accidental repeat should not skew the model).
    seen, deduped = set(), []
    for img, lbl in sources:
        key = (str(Path(img).resolve()), str(Path(lbl).resolve()))
        if key in seen:
            warnings.append(f"duplicate label source skipped: {lbl}")
            continue
        seen.add(key)
        deduped.append((img, lbl))
    sources = deduped

    from .trainer import DataLoader, RFTrainer, CNNTrainer  # lazy: pulls sklearn/torch only on train
    all_patches, all_features, all_labels = [], [], []
    for img, lbl in sources:
        patches, features, labels = DataLoader(img, lbl).load_labeled_data()
        all_patches += patches
        all_features += features
        all_labels += labels
    if not all_labels:
        raise ValueError("no labeled deposits found in the given sources (normal/rod/artifact) — "
                         "review/label the detections first, then train")

    class_counts = {c: all_labels.count(c) for c in ("normal", "rod", "artifact")}

    # The trainer uses a STRATIFIED train/test split (needs >=2 per class) and 5-fold CV (needs >=5
    # per class). Guard both so a tiny/imbalanced set gives an actionable error instead of a raw
    # sklearn ValueError, and CV auto-disables rather than crashing.
    present = {c: n for c, n in ((c, all_labels.count(c)) for c in set(all_labels)) if n}
    min_class, min_n = min(present.items(), key=lambda kv: kv[1])
    if len(present) < 2:
        raise ValueError(f"only one class present ({min_class}); need >=2 labeled classes to train a "
                         "classifier — review/label a mix of normal/rod/artifact")
    if min_n < 2:
        raise ValueError(f"class '{min_class}' has only {min_n} labeled sample(s); need >=2 per class "
                         "to train (>=5 for cross-validation). Review/label more of that class")
    do_cv = cross_validate and min_n >= 5
    if cross_validate and not do_cv:
        warnings.append(f"cross-validation disabled: smallest class '{min_class}' has {min_n} samples (<5)")

    if output:
        out_path = Path(normalize_path(output))
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # microseconds: no same-second collisions
        out_path = Path(__file__).parent.parent / "models" / f"model_{model_type}_{ts}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "rf":
        trainer = RFTrainer()
        results = trainer.train(all_features, all_labels, n_estimators=n_estimators,
                                cross_validate=do_cv)
    elif model_type == "cnn":
        trainer = CNNTrainer()
        results = trainer.train(all_patches, all_labels)
    else:
        raise ValueError(f"unknown model_type '{model_type}' (use 'rf' or 'cnn')")
    trainer.save(out_path)

    top = dict(sorted((results.get("feature_importance") or {}).items(),
                      key=lambda kv: -kv[1])[:5])
    return TrainResult(
        output_path=str(out_path), model_type=model_type, n_samples=len(all_labels),
        class_counts=class_counts,
        accuracy=results.get("accuracy", results.get("final_accuracy")),   # CNN reports final_accuracy
        cv_mean=results.get("cv_mean"), cv_std=results.get("cv_std"),
        top_features=top, sources=[str(l) for _, l in sources], warnings=warnings)


# --------------------------------------------------------------------------- clustering (labeling assist)
@dataclass
class ClusterSummary:
    output_dir: str
    n_images: int
    n_deposits: int
    n_clusters: int
    n_noise: int
    health: list


@dataclass
class PropagateSummary:
    n_labeled: int
    n_skipped: int
    readiness: str
    reasons: list
    class_counts: dict
    source_images: Optional[str] = None
    labels_dir: Optional[str] = None


def cluster_folder_service(path: str, output_dir: Optional[str] = None, method: str = "hdbscan",
                           min_cluster_size: Optional[int] = None, k: Optional[int] = None,
                           reps_per_cluster: int = 6) -> ClusterSummary:
    """Detect deposits (unsupervised), cluster them by feature, and write everything the user
    needs to label whole clusters: a labels.json per image (label='unknown' + informational
    cluster_id), the authoritative cluster_assignments.csv, representative thumbnails, a
    cluster_report.html, and a cluster_labels.csv template. Works from in-memory results so it
    keeps ALL deposits (including RF-called 'artifact')."""
    from .pathutils import normalize_path
    path = normalize_path(path)
    output_dir = normalize_path(output_dir) if output_dir else output_dir
    import cv2
    from . import clustering as C
    from .features import FeatureExtractor
    from .grouping_util import duplicate_basenames

    images = list_images(str(path))
    # labels.json / assignments key deposits by (basename, id); colliding basenames (e.g. from
    # a recursive folder with same-named files) would cross-apply labels — fail fast instead.
    dups = duplicate_basenames(images)
    if dups:
        raise ValueError(f"duplicate image basenames would collide: {dups[:5]}"
                         f"{'...' if len(dups) > 5 else ''}. Run on a flat folder of unique names.")
    mtype, mpath = resolve_model_type(None, None)
    az = Analyzer(detector=DepositDetector(),
                  classifier_config=ClassifierConfig(model_type=mtype, model_path=mpath))
    results = az.analyze_batch(images)

    rows = []
    for res in results:
        fx = FeatureExtractor(dpi=res.dpi)
        for d in res.deposits:
            fd = fx.to_feature_dict(d)
            cnt = d.contour
            area = cv2.contourArea(cnt) if cnt is not None else 0.0
            # solidity = area / convex-hull area: consolidates irregular/unusual deposits into a
            # real (labelable) cluster instead of scattering them into noise.
            hull_area = cv2.contourArea(cv2.convexHull(cnt)) if cnt is not None else 0.0
            fd["solidity"] = float(area / hull_area) if hull_area > 0 else 0.0
            # Rotation-invariant thinness for line-artifact detection (aspect_ratio from an
            # axis-aligned bbox misses DIAGONAL film boundaries). minAreaRect gives an
            # orientation-free elongation + how much of the rotated box the shape fills.
            if cnt is not None and len(cnt) >= 3:
                (rw, rh) = cv2.minAreaRect(cnt)[1]
                maj, mnr = max(rw, rh), min(rw, rh)
                fd["elongation"] = float(maj / mnr) if mnr > 0 else 1.0
                fd["rect_fill"] = float(area / (maj * mnr)) if maj * mnr > 0 else 0.0
            else:
                fd["elongation"], fd["rect_fill"] = 1.0, 0.0
            fd["filename"] = res.filename
            fd["deposit_id"] = d.id
            rows.append(fd)
    df = pd.DataFrame(rows)

    out = Path(output_dir) if output_dir else get_timestamped_output_dir(Path(path).parent, "clusters")
    out.mkdir(parents=True, exist_ok=True)

    if df.empty:
        (out / "cluster_assignments.csv").write_text("filename,deposit_id,cluster_id\n")
        return ClusterSummary(str(out), len(images), 0, 0, 0, ["no deposits detected"])

    X, _ = C.build_feature_matrix(df)
    cres = C.cluster_deposits(X, method=method, min_cluster_size=min_cluster_size, k=k)
    df["cluster_id"] = cres.labels

    df[["filename", "deposit_id", "cluster_id"]].to_csv(out / "cluster_assignments.csv", index=False)
    import json as _json
    (out / "cluster_meta.json").write_text(_json.dumps({"source_images": str(Path(path).resolve())}))
    _write_cluster_labels_json(out / "deposits", results, df)
    reps = C.representatives(X, cres.labels, per_kind=reps_per_cluster)
    _export_cluster_thumbnails(out / "clusters", df, reps, images)
    profile = C.cluster_profile(df, cres.labels)
    # Surface the shape-unusual deposits (elongated/irregular, ROD-like) — ranked across ALL
    # deposits (not only noise), line artifacts excluded (-inf), so the real atypical cohort
    # (which solidity consolidates into a cluster) is shown WITH its cluster_id for labeling.
    import numpy as _np
    df["unusual"] = C.unusual_ranking(df)
    unusual = df[_np.isfinite(df["unusual"])].sort_values("unusual", ascending=False).head(24)
    _export_unusual_thumbnails(out / "unusual", unusual, images)
    n_lines = int(C.line_flag(df).sum())
    _write_cluster_report_html(out / "cluster_report.html", profile, reps, cres, unusual, n_lines)
    _write_cluster_labels_csv(out / "cluster_labels.csv", profile, cres)

    return ClusterSummary(str(out), len(images), len(df), cres.n_clusters, cres.n_noise, cres.health)


def propagate_service(results_dir: str, csv_path: Optional[str] = None) -> PropagateSummary:
    """Apply the user's cluster->label mapping (cluster_labels.csv) to every member deposit via
    the authoritative cluster_assignments.csv (NOT the GUI-mutable labels.json), rewrite the
    labels.json labels, and run the training-readiness guard."""
    import json
    from . import clustering as C

    rd = Path(results_dir)
    assignments = pd.read_csv(rd / "cluster_assignments.csv")
    mapping = C.parse_cluster_labels_csv(Path(csv_path) if csv_path else rd / "cluster_labels.csv")
    labels, summary = C.propagate_labels(assignments, mapping)

    for p in (rd / "deposits").glob("*.labels.json"):
        data = json.loads(p.read_text())
        fn = data.get("image_file", p.name.replace(".labels.json", ""))
        for d in data["deposits"]:
            d["label"] = labels.get((fn, int(d["id"])), d.get("label", "unknown"))
        p.write_text(json.dumps(data, indent=2))

    share = None
    if summary["n_labeled"]:
        lab = [labels.get((str(f), int(i)), "unknown")
               for f, i in zip(assignments["filename"], assignments["deposit_id"])]
        by = assignments.assign(_lab=lab)
        counts = by[by["_lab"] != "unknown"].groupby("cluster_id").size()
        share = float(counts.max() / counts.sum()) if len(counts) else None
    rep = C.training_readiness(list(labels.values()), largest_cluster_share=share)
    source = None
    meta = rd / "cluster_meta.json"
    if meta.exists():
        try:
            source = json.loads(meta.read_text()).get("source_images")
        except Exception:
            source = None
    return PropagateSummary(summary["n_labeled"], summary["n_skipped"], rep.verdict,
                            rep.reasons, rep.class_counts, source_images=source,
                            labels_dir=str(rd / "deposits"))


def _write_cluster_labels_json(deposits_dir, results, df):
    import json
    deposits_dir.mkdir(parents=True, exist_ok=True)
    cid_by_key = {(str(r.filename), int(r.deposit_id)): int(r.cluster_id) for r in df.itertuples()}
    for res in results:
        stem = Path(res.filename).stem
        deps = []
        for d in res.deposits:
            deps.append({
                "id": d.id,
                "contour": d.contour.squeeze().tolist() if d.contour is not None else [],
                "x": d.centroid[0], "y": d.centroid[1], "width": d.width, "height": d.height,
                "area": float(d.area), "circularity": float(d.circularity),
                "label": "unknown", "confidence": 0.0,
                "cluster_id": cid_by_key.get((res.filename, d.id), -1),
                "merged": getattr(d, "merged", False), "group_id": getattr(d, "group_id", None),
            })
        with open(deposits_dir / f"{stem}.labels.json", "w") as f:
            json.dump({"image_file": res.filename, "next_group_id": 1, "deposits": deps}, f, indent=2)


def _crop_deposit(cache, img_by_name, fn, x, y, w, h, pad=6):
    import numpy as np
    from PIL import Image
    if fn not in cache:
        cache[fn] = np.array(Image.open(img_by_name[fn]))
    arr = cache[fn]
    x, y = int(x), int(y)
    w, h = int(w or 20), int(h or 20)
    y0, y1 = max(0, y - h // 2 - pad), min(arr.shape[0], y + h // 2 + pad)
    x0, x1 = max(0, x - w // 2 - pad), min(arr.shape[1], x + w // 2 + pad)
    crop = arr[y0:y1, x0:x1]
    return Image.fromarray(crop) if crop.size else None


def _export_cluster_thumbnails(clusters_dir, df, reps, images):
    img_by_name = {Path(p).name: p for p in images}
    rows = list(df.itertuples())
    cache = {}
    for cid, kinds in reps.items():
        cdir = clusters_dir / f"cluster_{cid}"
        cdir.mkdir(parents=True, exist_ok=True)
        for kind, idxs in kinds.items():
            for j, pos in enumerate(idxs):
                try:
                    r = rows[pos]
                    im = _crop_deposit(cache, img_by_name, str(r.filename), r.x, r.y,
                                       getattr(r, "width", 20), getattr(r, "height", 20))
                    if im is not None:
                        im.save(cdir / f"{kind}_{j}.png")
                except Exception:
                    continue  # thumbnails are best-effort; never fail the run on one crop


def _export_unusual_thumbnails(unusual_dir, unusual_df, images):
    import cv2
    import numpy as np
    from PIL import Image
    img_by_name = {Path(p).name: p for p in images}
    unusual_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for j, (_, r) in enumerate(unusual_df.iterrows()):
        try:
            im = _crop_deposit(cache, img_by_name, str(r["filename"]), r["x"], r["y"],
                               r.get("width", 20), r.get("height", 20), pad=10)
            if im is None:
                continue
            # stamp the cluster_id so the user knows which cluster to label (noise = -1)
            arr = np.array(im)
            cid = int(r.get("cluster_id", -1))
            tag = f"c{cid}" if cid >= 0 else "noise"
            cv2.putText(arr, tag, (2, arr.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (32, 96, 200), 1, cv2.LINE_AA)
            Image.fromarray(arr).save(unusual_dir / f"rank_{j:02d}.png")
        except Exception:
            continue


def _write_cluster_labels_csv(path, profile, cres):
    keep = ["cluster_id", "size", "kind", "area_px", "circularity", "aspect_ratio", "solidity"]
    cols = [c for c in keep if c in profile.columns]
    out = profile[profile["cluster_id"] != -1][cols].copy()
    total_deposits = max(1, int(profile["size"].sum()))  # size sums over all rows incl. -1 noise
    out["noise_frac"] = round(cres.n_noise / total_deposits, 3)
    out["label"] = ""  # user fills: normal / rod / artifact (or blank to skip)
    out.to_csv(path, index=False)


def _write_cluster_report_html(path, profile, reps, cres, unusual=None, n_lines=0):
    import base64
    root = Path(path).parent

    def _b64(p):
        return base64.b64encode(Path(p).read_bytes()).decode() if Path(p).exists() else ""

    parts = ["<!DOCTYPE html><meta charset='utf-8'><title>SCAT clusters</title>",
             "<style>body{font-family:sans-serif;max-width:1000px;margin:2rem auto}"
             ".c{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}"
             ".u{border:2px solid #2F6B9E;border-radius:8px;padding:12px;margin:12px 0;background:#f4f8fb}"
             "img{height:72px;margin:2px;border:1px solid #eee}.warn{color:#b26a00}"
             ".k{color:#2F6B9E;font-weight:600}</style>",
             f"<h1>Cluster report — {cres.n_clusters} clusters, {cres.n_noise} noise</h1>"]
    for h in cres.health:
        parts.append(f"<p class='warn'>&#9888; {h}</p>")
    # Unusual deposits — the shape-atypical ones pulled out of the noise bucket
    if unusual is not None and len(unusual):
        parts.append("<div class='u'><h2>&#11088; Unusual deposits (review — likely ROD / atypical)</h2>"
                     "<p>Shape-atypical deposits surfaced from the noise bucket (line artifacts "
                     "down-ranked). Label the cluster they fall in, or fix individually in the "
                     "labeling GUI.</p>")
        for j in range(len(unusual)):
            b = _b64(root / "unusual" / f"rank_{j:02d}.png")
            if b:
                parts.append(f"<img src='data:image/png;base64,{b}'>")
        if n_lines:
            parts.append(f"<p class='warn'>~{n_lines} deposit(s) look like line artifacts (film "
                         f"boundaries: very elongated + near-zero circularity) — likely 'artifact'.</p>")
        parts.append("</div>")
    for _, row in profile.iterrows():
        cid = int(row["cluster_id"])
        kind = row.get("kind", "")
        title = "Outliers (noise -1)" if cid == -1 else f"Cluster {cid}"
        parts.append(f"<div class='c'><h3>{title} — size {int(row['size'])} "
                     f"<span class='k'>[{kind}]</span></h3>")
        if cid in reps:
            for k in ("medoid", "random", "boundary"):
                for j in range(len(reps[cid][k])):
                    b = _b64(root / "clusters" / f"cluster_{cid}" / f"{k}_{j}.png")
                    if b:
                        parts.append(f"<img title='{k}' src='data:image/png;base64,{b}'>")
        feats = ", ".join(f"{c}={row[c]:.2f}" for c in ("area_px", "circularity", "aspect_ratio",
                          "solidity") if c in profile.columns and pd.notna(row[c]))
        parts.append(f"<p>{feats}</p></div>")
    Path(path).write_text("".join(parts), encoding="utf-8")
