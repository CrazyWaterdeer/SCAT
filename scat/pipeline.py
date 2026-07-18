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
                           confidence_threshold: Optional[float] = None) -> AnalyzeResult:
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
                arr = np.array(Image.open(img_path))
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
        warnings=warnings,
        primary_metric=primary_metric, normalization=normalization,
        confidence_threshold=confidence_threshold)

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
    return generate_report(film, output_dir=rd, deposit_data=deposits,
                           statistical_results=metrics, spatial_stats=spatial_stats,
                           group_by=group_by, format="html", analysis=analysis)


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
