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
    p = Path(path)
    if p.is_file():
        return [p]
    imgs: list[Path] = []
    for g in IMAGE_GLOBS:
        imgs += p.rglob(g)
    return sorted(set(imgs))


def resolve_model_type(model_type: Optional[str], model_path: Optional[str]) -> tuple[str, Optional[str]]:
    """Canonical default: rf if a model file is available, else threshold."""
    if model_type:
        return model_type, model_path
    default_model = Path(__file__).parent.parent / "models" / "model_rf.pkl"
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
                           output_dir: Optional[str] = None) -> AnalyzeResult:
    """Canonical folder analysis. Superset of every caller (CLI + GUI); every param
    beyond the original signature defaults to the value that reproduces the pre-slimdown
    CLI/parity behaviour, so tests/test_pipeline_parity.py (default kwargs) stays byte-identical.

    image_paths: analyse exactly this explicit list (the GUI is a multi-file picker); when
    None, rglob the folder as before. progress_callback(current, total) drives the GUI bar.
    """
    from .grouping_util import build_group_metadata, duplicate_basenames
    if image_paths is not None:
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
    if groups:
        metadata, group_by = build_group_metadata(groups)
        group_names = sorted(set(metadata["group"]) - {"ungrouped"})
        if len(group_names) < 2:
            warnings.append(f"stats skipped: {len(group_names)} group(s) — need >=2 to compare")

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
    n_failed = sum(1 for r in results if r.n_total == 0)

    if annotate:
        from PIL import Image
        import numpy as np
        ann_dir = out / "annotated"; ann_dir.mkdir(exist_ok=True)
        for img_path, res in zip(images, results):
            if res.n_total == 0:
                continue
            arr = np.array(Image.open(img_path))
            annotated = analyzer.generate_annotated_image(arr, res.deposits, show_labels=True, skip_artifacts=True)
            Image.fromarray(annotated).save(ann_dir / f"{img_path.stem}_annotated.png")

    if visualize:
        try:
            from .visualization import generate_all_visualizations
            generate_all_visualizations(reports["film_summary"], reports["deposit_data"],
                                        out / "visualizations",
                                        group_by=group_by[0] if group_by else None,
                                        significance_mode=significance_mode, show_ns=show_ns,
                                        condition_matrix=condition_matrix)
        except ImportError as e:
            warnings.append(f"visualizations skipped (missing deps): {e}")

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
        warnings=warnings)

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
    if group_col not in film.columns or film[group_col].dropna().nunique() < 2:
        return {"skipped": True, "reason": f"<2 groups in column '{group_col}'"}
    return run_comprehensive_analysis(film, deposits_df=deposits, group_column=group_col)


def generate_report_service(results_dir: str, statistical_results: Optional[dict] = None,
                            group_by: Optional[str] = None) -> str:
    from .report import generate_report
    rd = Path(results_dir)
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
    return generate_report(film, output_dir=rd, deposit_data=deposits,
                           statistical_results=metrics, spatial_stats=spatial_stats,
                           group_by=group_by, format="html")
