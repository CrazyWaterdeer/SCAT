import os
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from scat.agent.registry import tool
from scat.pipeline import (analyze_folder_service, run_statistics_service, generate_report_service,
                           rerender_results_service, train_model_service)


@tool(description="Detect + classify deposits across a folder and write CSVs/JSON/annotations to a timestamped results dir. Pass groups={filename: group_label} — the mapping YOU inferred from the filenames — to enable group comparison. Errors on duplicate basenames (unsafe to group). Pass image_paths=[...] to analyze ONLY those images (e.g. the pending ones from scan_folder's already_analyzed) instead of the whole folder. Set visualize=True to also render comparison plots; significance_mode chooses which significance brackets to draw ('auto'|'vs_control'|'adjacent'|'pairwise'|'none') — YOU decide it from the design. For a FACTORIAL design (2+ crossed factors, e.g. Drug × Light), pass condition_matrix={factor_name: {group: true/false}} to also render bar charts with an open/closed-circle condition table beneath (● = factor present, ○ = absent). Pass primary_metric (total_deposits [DEFAULT], rod_fraction, mean_area, mean_hue, total_iod, mean_circularity) — the predeclared endpoint this experiment measures; CONFIRM it with the user before running, never silently guess. Pass palette={group_label: color} (hex like '#4C72B0' or CSS name like 'tomato') to set plot COLORS per group; unlisted groups keep the default palette and pH/hue plots keep their Bromophenol-Blue coloring. To CHANGE colors/metric/order on an ALREADY-analyzed folder, use rerender_report instead (no re-detection). Pass n_flies to normalize the count metrics (deposit count, IOD) PER FLY — this is the meaningful readout when vials hold different numbers of flies, so total-vs-total is not. n_flies is a JSON object mapping EACH image's filename to its integer fly count, e.g. {\"CS mF 24h 3 flies (1).tif\": 3, \"CS mF 24h 2 flies.tif\": 2}; read the counts from the filenames (e.g. '... 3 flies ...' -> 3). If every vial has the same count, still pass the full map with that number for every image. Cover EVERY image — a partial map falls back to per-image totals with a warning. If some filenames encode a count and others don't, ASK the user for the missing ones. If fly counts are truly unavailable, omit n_flies (per-image totals + a warning). Pass bar_groups to draw a CLUSTERED bar chart: the bars of related groups sit adjacent and a gap separates each cluster (with a label beneath) — set it to a JSON object {cluster_label: [group, ...]} where each value lists the groups (in the left-to-right order you want) that belong together, e.g. {\"mF\": [\"mF 24h\",\"mF 48h\"], \"Saline\": [\"saline 24h\",\"saline 48h\"]}. Requires visualize=True. With bar_groups you can also pass bar_colors to REPEAT a color across clusters: a JSON object {token_or_group: color} where a bar is colored if its group name contains the token — e.g. {\"24h\":\"#4C72B0\", \"48h\":\"#DD8452\"} paints every '…24h' group blue and every '…48h' orange in EVERY cluster (with a legend). So the user picks their viewing angle: cluster by one dimension (bar_groups), color-repeat by another (bar_colors).")
def analyze_folder(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                   min_area: int = 20, max_area: int = 10000, circularity: float = 0.6,
                   annotate: bool = True, image_paths: Optional[list[str]] = None,
                   visualize: bool = False, significance_mode: str = "auto",
                   show_ns: bool = False, condition_matrix: Optional[dict] = None,
                   primary_metric: Optional[str] = None, palette: Optional[dict] = None,
                   n_flies: Optional[dict[str, int]] = None,
                   bar_groups: Optional[dict] = None, bar_colors: Optional[dict] = None) -> dict:
    """Run detection + classification over a folder; returns counts + output dir."""
    return asdict(analyze_folder_service(path, groups=groups, model_type=model_type,
                                         min_area=min_area, max_area=max_area,
                                         circularity=circularity, annotate=annotate,
                                         image_paths=image_paths, visualize=visualize,
                                         significance_mode=significance_mode, show_ns=show_ns,
                                         condition_matrix=condition_matrix,
                                         primary_metric=primary_metric, palette=palette,
                                         n_flies=n_flies, bar_groups=bar_groups, bar_colors=bar_colors,
                                         ambient_progress=True))  # stream progress + honor Stop


@tool(description="Run group statistics on a completed results dir. No-ops (with a reason) if <2 groups.")
def run_statistics(results_dir: str, group_col: str = "group") -> dict:
    """Group comparison statistics over a results directory."""
    return run_statistics_service(results_dir, group_col=group_col)


@tool(description="Generate the HTML report for a results dir. Pass the statistics dict if available.")
def generate_report(results_dir: str, statistical_results: Optional[dict] = None,
                    group_by: Optional[str] = None) -> dict:
    """Build the self-contained HTML report; returns its path."""
    return {"report_path": generate_report_service(results_dir, statistical_results, group_by)}


@tool(description="Re-render statistics + comparison graphs + the HTML report from an EXISTING results dir WITHOUT re-detecting. Use this to (a) produce the outputs AFTER the user has manually reviewed/relabeled the detections (it rebuilds everything from the on-disk CSVs, so their label edits are preserved and the statistics are recomputed to match), and (b) re-graph the SAME detections differently. It NEVER runs detection — the reviewed detections are untouched. Optional overrides: primary_metric (change the headline endpoint the report/graph is keyed to — one of total_deposits|rod_fraction|mean_area|mean_hue|total_iod|mean_circularity); group_order (a list giving the explicit left-to-right order of groups on the x-axis); control_group (pin the reference group, drawn first and in gray); palette (change the COLORS — a dict {group_label: color} where color is a hex like '#4C72B0' or a CSS name like 'tomato'; unlisted groups keep the default palette; pH/hue plots keep their Bromophenol-Blue coloring); significance_mode ('auto'|'vs_control'|'adjacent'|'pairwise'|'none') and show_ns for the bracket style; condition_matrix for a factorial design table; n_flies (a JSON object mapping each image filename to its integer fly count, covering EVERY image) to ADD per-image fly counts to an existing run so the deposit/IOD comparison becomes per-fly (persisted into image_summary.csv; this augments vial metadata, it does not re-detect); bar_groups to draw a CLUSTERED bar chart on the existing results — a JSON object {cluster_label: [group, ...]} grouping related bars adjacent with a gap between clusters (no re-detection); bar_colors ({token_or_group: color}) to REPEAT a color across clusters (e.g. {\"24h\":\"#4C72B0\",\"48h\":\"#DD8452\"} colors every '…24h' the same in every cluster, with a legend). Overrides are saved into run_manifest.json so later reports stay consistent. Prefer this over generate_report whenever the user wants edits reflected or a different metric/order/colors/brackets/fly-normalization on already-detected results.")
def rerender_report(results_dir: str, primary_metric: Optional[str] = None,
                    group_order: Optional[list] = None, control_group: Optional[str] = None,
                    group_by: Optional[str] = None, significance_mode: str = "auto",
                    show_ns: bool = False, condition_matrix: Optional[dict] = None,
                    palette: Optional[dict] = None, n_flies: Optional[dict[str, int]] = None,
                    bar_groups: Optional[dict] = None, bar_colors: Optional[dict] = None,
                    regenerate_visualizations: bool = True) -> dict:
    """Rebuild stats + plots + report from an existing results dir (no re-detection)."""
    return asdict(rerender_results_service(
        results_dir, primary_metric=primary_metric, group_order=group_order,
        control_group=control_group, group_by=group_by, significance_mode=significance_mode,
        show_ns=show_ns, condition_matrix=condition_matrix, palette=palette, n_flies=n_flies,
        bar_groups=bar_groups, bar_colors=bar_colors,
        regenerate_visualizations=regenerate_visualizations))


@tool(description="Train (or retrain) the deposit classifier (Random Forest, or CNN) from LABELED results — the way to UPDATE the active model or make a NEW one. Labels come from a MANUAL review: after the user reviews/relabels a results dir, its deposits/*.labels.json hold the corrected labels, so pass results_dirs=[that dir]. There is NO warm-start/incremental learning — training fits a FRESH model each time, so 'update the existing model' means retrain on the UNION of the old labels + the new ones: pass ALL the relevant results_dirs (and/or an explicit image_dir+label_dir for a curated ground-truth set) together. output=path to write the model; OMIT it to write a timestamped file under models/ (safe — never overwrites the bundled models/model_rf.pkl); to make the trained model the ACTIVE one, set output to '<repo>/models/model_rf.pkl' after confirming with the user. model_type 'rf' (default) or 'cnn'. Returns sample/class counts, accuracy, cross-val, top features, and the model path. IMPORTANT: only train on REVIEWED labels — training on raw un-reviewed detections just re-learns the current model. This tool does NOT detect or label; do that first (analyze_folder, then the user reviews).")
def train_model(results_dirs: Optional[list] = None, image_dir: Optional[str] = None,
                label_dir: Optional[str] = None, output: Optional[str] = None,
                model_type: str = "rf", n_estimators: int = 100) -> dict:
    """Train an RF/CNN classifier from labeled results dirs and/or an explicit image_dir+label_dir."""
    return asdict(train_model_service(results_dirs=results_dirs, image_dir=image_dir,
                                      label_dir=label_dir, output=output, model_type=model_type,
                                      n_estimators=n_estimators))


def _search_roots(folder: Optional[str] = None) -> list[str]:
    """Roots to scan for prior results. Results dirs are siblings of the analyzed folder, so a folder
    argument must include its PARENT (searching the folder alone finds nothing)."""
    from scat.config import config
    roots = [folder, str(Path(folder).parent)] if folder else [os.getcwd()]
    roots += list(config.get("agent.results_search_roots", []) or [])
    return roots  # results_index normalizes/dedupes/drops non-existent


@tool(description="List prior SCAT results dirs discoverable near a folder (or the default search roots): each run's results_dir, dataset_path, created_at, n_images, groups, model/detection, status. Use this to resume/reuse prior analyses instead of re-running, or to find a results dir to run stats/report on.")
def list_analyses(folder: Optional[str] = None) -> dict:
    """Enumerate on-disk results dirs (a per-run summary each)."""
    from scat.results_index import find_analyses, run_brief
    roots = _search_roots(folder)
    return {"analyses": [run_brief(r) for r in find_analyses(roots)],
            "search_roots": [str(r) for r in roots]}


@tool(description="Merge several COMPATIBLE results dirs (same dataset folder, identical model/detection/grouping) into ONE new results dir you can then run_statistics/generate_report on. Use after a pending-only resume run to get correct whole-experiment stats over old+new images. Refuses (with a reason) if the runs are incompatible or an overlapping image differs between them.")
def combine_results(results_dirs: list[str], output_dir: Optional[str] = None) -> dict:
    """Concatenate compatible results dirs into one merged dir; returns its path + counts."""
    from scat.combine import combine_results_service
    return combine_results_service(results_dirs, output_dir=output_dir)
