from scat.agent.registry import tool
from scat.grouping_util import infer_groups_from_folder


@tool(description="Infer experimental groups from filenames/subfolders. Discovers ARBITRARY condition names and ANY number of groups (genotypes, doses, timepoints, etc.) — subfolder names, or the <condition>_<replicate> prefix, not a fixed control/treated list. Returns {basename: group}, the detected groups, and a confidence. State the mapping to the user before analyzing; if confidence is 'low', recommend confirmation or a metadata CSV.")
def infer_groups(path: str) -> dict:
    """Infer experimental groups from filenames or subfolders (deterministic, arbitrary conditions)."""
    return infer_groups_from_folder(path)
