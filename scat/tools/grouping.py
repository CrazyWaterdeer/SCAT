from scat.agent.registry import tool
from scat.grouping_util import infer_groups_from_folder


@tool(description="Infer experimental groups from filenames/subfolders. Returns {basename: group} plus confidence. State the mapping to the user before analyzing; if confidence is 'low', recommend confirmation or a metadata CSV.")
def infer_groups(path: str) -> dict:
    """Infer experimental groups from filenames or subfolders (deterministic)."""
    return infer_groups_from_folder(path)
