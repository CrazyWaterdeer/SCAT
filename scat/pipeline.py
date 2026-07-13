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


def scan_folder_service(path: str) -> dict:
    imgs = list_images(path)
    exts = sorted({p.suffix.lower() for p in imgs})
    root = Path(path)
    subdirs = sorted({p.parent.name for p in imgs if p.parent != root})
    return {"path": str(path), "n_images": len(imgs), "extensions": exts,
            "subfolders": subdirs, "sample_names": [p.name for p in imgs[:8]]}


def analyze_folder_service(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                           model_path: Optional[str] = None, min_area: int = 20, max_area: int = 10000,
                           edge_margin: int = 20, circularity: float = 0.6, annotate: bool = True,
                           visualize: bool = False, output_dir: Optional[str] = None) -> AnalyzeResult:
    from .grouping_util import build_group_metadata
    images = list_images(path)
    if not images:
        raise ValueError(f"No images found in {path}")
    mtype, mpath = resolve_model_type(model_type, model_path)
    detector = DepositDetector(min_area=min_area, max_area=max_area, edge_margin=edge_margin)
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

    results = analyzer.analyze_batch(images, metadata=metadata)
    out = Path(output_dir) if output_dir else get_timestamped_output_dir(Path(path).parent, "results")
    reporter = ReportGenerator(out)
    reports = reporter.save_all(results, metadata, group_by)
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
                                        group_by=group_by[0] if group_by else None)
        except ImportError as e:
            warnings.append(f"visualizations skipped (missing deps): {e}")

    summary = reports["film_summary"]
    return AnalyzeResult(
        output_dir=str(out), n_images=len(results),
        n_normal=int(summary["n_normal"].sum()), n_rod=int(summary["n_rod"].sum()),
        n_artifact=int(summary["n_artifact"].sum()), n_failed=n_failed,
        groups=group_names, warnings=warnings)


def run_statistics_service(results_dir: str, group_col: str = "group") -> dict:
    from .statistics import run_comprehensive_analysis
    rd = Path(results_dir)
    film = pd.read_csv(rd / "image_summary.csv")
    deposits = pd.read_csv(rd / "all_deposits.csv") if (rd / "all_deposits.csv").exists() else None
    if group_col not in film.columns or film[group_col].dropna().nunique() < 2:
        return {"skipped": True, "reason": f"<2 groups in column '{group_col}'"}
    return run_comprehensive_analysis(film, deposits_df=deposits, group_column=group_col)


def generate_report_service(results_dir: str, statistical_results: Optional[dict] = None,
                            group_by: Optional[str] = None) -> str:
    from .report import generate_report
    rd = Path(results_dir)
    film = pd.read_csv(rd / "image_summary.csv")
    deposits = pd.read_csv(rd / "all_deposits.csv") if (rd / "all_deposits.csv").exists() else None
    # report expects the FLAT metrics mapping, not the whole run_comprehensive_analysis dict.
    metrics = None
    if statistical_results and not statistical_results.get("skipped"):
        metrics = statistical_results.get("basic", {}).get("metrics") or statistical_results
    return generate_report(film, output_dir=rd, deposit_data=deposits,
                           statistical_results=metrics, group_by=group_by, format="html")
