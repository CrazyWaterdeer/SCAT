#!/usr/bin/env python
"""
SCAT Build Script
Builds the executable and sets up the distribution folder structure.

Usage:
    python build.py          # Full build
    python build.py --clean  # Clean build (removes previous build artifacts)
"""

import subprocess
import shutil
import sys
from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
SCAT_DIST = DIST_DIR / "SCAT"


def clean():
    """Remove previous build artifacts."""
    print("Cleaning previous build...")
    
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"  Removed {BUILD_DIR}")
    
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
        print(f"  Removed {DIST_DIR}")
    
    print("Clean complete.\n")


def build_exe():
    """Run PyInstaller to build the executable."""
    print("Building executable with PyInstaller...")
    print("-" * 50)
    
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "SCAT.spec", "--noconfirm"],
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print("\nERROR: PyInstaller build failed!")
        sys.exit(1)
    
    print("-" * 50)
    print("PyInstaller build complete.\n")


def setup_distribution():
    """Set up the distribution folder structure."""
    print("Setting up distribution structure...")
    
    if not SCAT_DIST.exists():
        print(f"ERROR: {SCAT_DIST} not found. Build may have failed.")
        sys.exit(1)
    
    # Create data folders
    folders = [
        SCAT_DIST / "Models",
        SCAT_DIST / "Data" / "Images",
        SCAT_DIST / "Data" / "Results",
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {folder.relative_to(DIST_DIR)}")
    
    # Copy pre-trained model
    model_src = PROJECT_ROOT / "models" / "model_rf.pkl"
    model_dst = SCAT_DIST / "Models" / "model_rf.pkl"
    if model_src.exists():
        shutil.copy2(model_src, model_dst)
        print(f"  Copied:  model_rf.pkl -> Models/")
    else:
        print(f"  Warning: models/model_rf.pkl not found, skipping")
    
    # Copy documentation files
    docs = ["README.md", "WORKFLOW.md", "LICENSE"]
    
    for doc in docs:
        src = PROJECT_ROOT / doc
        dst = SCAT_DIST / doc
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied:  {doc}")
        else:
            print(f"  Warning: {doc} not found, skipping")
    
    print("\nDistribution setup complete.")


def print_summary():
    """Print the final distribution structure."""
    print("\n" + "=" * 50)
    print("BUILD SUCCESSFUL!")
    print("=" * 50)
    print(f"\nOutput: {SCAT_DIST}\n")
    print("Distribution structure:")
    print("  SCAT/")
    print("  ├── SCAT.exe")
    print("  ├── _internal/")
    print("  ├── Models/")
    print("  │   └── model_rf.pkl")
    print("  ├── Data/")
    print("  │   ├── Images/")
    print("  │   └── Results/")
    print("  ├── README.md")
    print("  ├── WORKFLOW.md")
    print("  └── LICENSE")
    print("\nNext steps:")
    print("  1. Zip the SCAT folder for distribution")
    print("  2. Users can place images in Data/Images/ and run SCAT.exe")


def main():
    print("\n" + "=" * 50)
    print("SCAT Build Script")
    print("=" * 50 + "\n")
    
    # Check for --clean flag
    if "--clean" in sys.argv:
        clean()
    
    # Build
    build_exe()
    
    # Setup distribution structure
    setup_distribution()
    
    # Summary
    print_summary()


if __name__ == "__main__":
    main()
