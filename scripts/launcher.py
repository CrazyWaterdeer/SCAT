#!/usr/bin/env python
"""
SCAT Launcher for PyInstaller.
This script is the entry point for the packaged EXE.
"""

import sys
import os

# Set Windows AppUserModelID for proper taskbar icon display
# This MUST be called before any Qt imports
if sys.platform == 'win32':
    import ctypes
    # Unique identifier for this application
    APP_ID = 'SCAT.DepositAnalyzer.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

# Add the parent directory to path for proper imports
if getattr(sys, 'frozen', False):
    # Running as compiled EXE
    application_path = sys._MEIPASS
    sys.path.insert(0, application_path)
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))

# Now import and run the main GUI
from scat.main_gui import run_gui

if __name__ == "__main__":
    run_gui()
