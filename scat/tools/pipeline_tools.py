from typing import Optional
from dataclasses import asdict
from scat.agent.registry import tool
from scat.pipeline import analyze_folder_service, run_statistics_service, generate_report_service


@tool(description="Detect + classify deposits across a folder and write CSVs/JSON/annotations to a timestamped results dir. Pass groups={filename: group_label} — the mapping YOU inferred from the filenames — to enable group comparison. Errors on duplicate basenames (unsafe to group).")
def analyze_folder(path: str, groups: Optional[dict] = None, model_type: Optional[str] = None,
                   min_area: int = 20, max_area: int = 10000, circularity: float = 0.6,
                   annotate: bool = True) -> dict:
    """Run detection + classification over a folder; returns counts + output dir."""
    return asdict(analyze_folder_service(path, groups=groups, model_type=model_type,
                                         min_area=min_area, max_area=max_area,
                                         circularity=circularity, annotate=annotate))


@tool(description="Run group statistics on a completed results dir. No-ops (with a reason) if <2 groups.")
def run_statistics(results_dir: str, group_col: str = "group") -> dict:
    """Group comparison statistics over a results directory."""
    return run_statistics_service(results_dir, group_col=group_col)


@tool(description="Generate the HTML report for a results dir. Pass the statistics dict if available.")
def generate_report(results_dir: str, statistical_results: Optional[dict] = None,
                    group_by: Optional[str] = None) -> dict:
    """Build the self-contained HTML report; returns its path."""
    return {"report_path": generate_report_service(results_dir, statistical_results, group_by)}
