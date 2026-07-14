# -*- mode: python ; coding: utf-8 -*-
"""
SCAT PyInstaller Spec File
- Target: Windows 11 x64
- Mode: onedir (folder distribution)
- Console: hidden (windowed)
- Deep learning: EXCLUDED (torch, torchvision)

After build, run build.py to complete the distribution structure:
    SCAT/
    ├── SCAT.exe
    ├── _internal/
    ├── Models/
    ├── Data/
    │   ├── Images/
    │   └── Results/
    ├── README.md
    ├── WORKFLOW.md
    └── LICENSE
"""

import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(SPECPATH)
SCAT_PACKAGE = PROJECT_ROOT / 'scat'
RESOURCES_DIR = SCAT_PACKAGE / 'resources'

# =============================================================================
# Analysis: Dependency collection
# =============================================================================
a = Analysis(
    # Entry point
    scripts=[str(PROJECT_ROOT / 'scripts' / 'launcher.py')],
    
    # Additional search paths
    pathex=[str(PROJECT_ROOT)],
    
    # Binary files (DLLs, .so files) - auto-detected, add manual ones here
    binaries=[],
    
    # Data files: (source, destination_in_bundle)
    # destination is relative to _internal/ in onedir mode
    datas=[
        # Resources folder (fonts, icons)
        (str(RESOURCES_DIR), 'scat/resources'),
    ],
    
    # Hidden imports: modules not auto-detected by PyInstaller
    hiddenimports=[
        # SCAT package modules
        'scat',
        'scat.analyzer',
        'scat.classifier',
        'scat.cli',
        'scat.config',
        'scat.detector',
        'scat.features',
        'scat.labeling_gui',
        'scat.main_gui',
        'scat.report',
        'scat.segmentation',
        'scat.spatial',
        'scat.statistics',
        'scat.trainer',
        'scat.ui_common',
        'scat.visualization',
        
        # PySide6 plugins (commonly missed)
        'PySide6.QtSvg',
        'PySide6.QtPrintSupport',
        
        # Scientific stack
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'pandas',
        'pandas._libs.tslibs.base',
        'sklearn',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree._utils',
        'sklearn.utils._weight_vector',
        'scipy',
        'scipy.special._ufuncs_cxx',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy.sparse.csgraph._validation',
        
        # Image processing
        'cv2',
        'PIL',
        'PIL._tkinter_finder',
        
        # Visualization
        'matplotlib',
        'matplotlib.backends.backend_qtagg',
        'seaborn',
        
        # Templating (for reports)
        'jinja2',
        'markupsafe',
    ],
    
    # Packages to collect entirely (submodules + data files)
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    
    # Excludes: reduce bundle size by removing unnecessary modules
    excludes=[
        # Deep learning (not included in this build)
        'torch',
        'torchvision',
        
        # Test frameworks (unittest is needed by scipy/sklearn)
        'pytest',
        'test',
        
        # Development tools
        'IPython',
        'ipykernel',
        'jupyter',
        'notebook',
        
        # Unused GUI frameworks
        'tkinter',
        '_tkinter',
        'PyQt5',
        'PyQt6',
        'wx',
        
        # Unused backends
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_wxagg',
    ],
    
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# =============================================================================
# PYZ: Python bytecode archive
# =============================================================================
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# =============================================================================
# EXE: Executable configuration
# =============================================================================
exe = EXE(
    pyz,
    a.scripts,
    [],  # Empty for onedir mode
    exclude_binaries=True,  # True for onedir mode
    name='SCAT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress with UPX if available
    console=False,  # Windowed mode (no console)
    disable_windowed_traceback=False,
    
    # Windows-specific options
    icon=str(RESOURCES_DIR / 'icon.ico'),
    
    # Version info (optional, can add version.txt later)
    version=None,
    
    # UAC settings for Windows 11
    uac_admin=False,  # Set True if admin rights needed
    uac_uiaccess=False,
    
    # Runtime temp directory name
    argv_emulation=False,
    target_arch=None,  # Auto-detect (x64 for Win11)
    codesign_identity=None,
    entitlements_file=None,
)

# =============================================================================
# COLLECT: Bundle everything into dist folder
# =============================================================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # Files that don't compress well or break when compressed
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'msvcp140.dll',
        'python*.dll',
        'Qt*.dll',
    ],
    name='SCAT',  # Output folder name: dist/SCAT/
)
