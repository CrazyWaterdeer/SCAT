from scat.agent.registry import tool
from scat.pipeline import scan_folder_service


@tool(description="List a folder's images with their filenames and subfolders. Call this first — read the filenames to decide the experimental grouping yourself.")
def scan_folder(path: str) -> dict:
    """List images (filename + subfolder) and extensions so you can infer groups from the names."""
    return scan_folder_service(path)
