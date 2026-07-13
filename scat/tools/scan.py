from scat.agent.registry import tool
from scat.pipeline import scan_folder_service


@tool(description="List images in a folder and summarize filename structure. Call this first.")
def scan_folder(path: str) -> dict:
    """List images in a folder and summarize extensions/subfolders/sample names."""
    return scan_folder_service(path)
