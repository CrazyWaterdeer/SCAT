#!/usr/bin/env python
"""
SCAT Launcher for PyInstaller.
This script is the entry point for the packaged EXE.
Handles frozen environment setup (paths, working directory).
"""

import sys
import os

# Fix working directory for PyInstaller frozen environment
# This ensures relative paths (Models/, Data/) work correctly
# regardless of where the EXE is launched from (e.g., shortcuts, command line)
if getattr(sys, 'frozen', False):
    # Running as compiled EXE
    exe_dir = os.path.dirname(sys.executable)
    os.chdir(exe_dir)
    
    # Add _internal to path for imports
    application_path = sys._MEIPASS
    sys.path.insert(0, application_path)

# Import and run the main GUI
# AppUserModelID is set in main_gui.py at import time
from scat.main_gui import run_gui

if __name__ == "__main__":
    run_gui()
