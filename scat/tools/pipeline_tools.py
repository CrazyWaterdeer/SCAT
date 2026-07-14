import os
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from scat.agent.registry import tool
from scat.pipeline import analyze_folder_service, run_statistics_service, generate_report_service


@tool(description="Detect + classify deposits across a folder and write CSVs/JSON/annotations to a timestamped results dir. Pass groups={filename: group_label} — the mapping YOU inferred from the filenames — to enable group comparison. Errors on duplicate basenames (unsafe to group). Pass image_paths=[...] to analyze ONLY those images (e.g. the pending ones from scan_folder's already_analyzed) instead of the whole folder.")
def analyze_folder(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                   min_area: int = 20, max_area: int = 10000, circularity: float = 0.6,
                   annotate: bool = True, image_paths: Optional[list[str]] = None) -> dict:
    """Run detection + classification over a folder; returns counts + output dir."""
    return asdict(analyze_folder_service(path, groups=groups, model_type=model_type,
                                         min_area=min_area, max_area=max_area,
                                         circularity=circularity, annotate=annotate,
                                         image_paths=image_paths,
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
