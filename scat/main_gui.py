"""
Main GUI application for SCAT.
Integrates labeling, training, analysis, and results viewing.
"""

import sys
import os
import subprocess
import ctypes

# Set AppUserModelID for Windows taskbar icon
# Must be called before QApplication is created
if sys.platform == 'win32':
    try:
        # Unique identifier for this application
        app_id = 'SCAT.DepositAnalyzer.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # Ignore on non-Windows or if it fails
import json
from pathlib import Path
from typing import List
import pandas as pd
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QMessageBox, QScrollArea,
    QDialog, QKeySequenceEdit, QDialogButtonBox,
    QMenu, QInputDialog,
    QTreeWidget, QTreeWidgetItem, QDockWidget,
    QSizePolicy, QGridLayout, QStackedWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (
    QFont, QPixmap, QIcon, QKeySequence,
    QColor, QShortcut, QFontDatabase
)

# Import SCAT modules
from .config import config, get_timestamped_output_dir
from .artifacts import IMAGE_SUMMARY, ALL_DEPOSITS
from . import metrics as _metrics
from . import confidence as _conf
from .ui_common import (
    Theme, NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox,
    CollapsibleSection, CenteredCap, ToggleSwitch, setting_row,
    NumericTableWidgetItem, icon, load_custom_fonts, get_icon_path
)
from . import ui_motion

# Note: trainer is imported lazily when needed to avoid loading sklearn at startup


class ShortcutEditor(QWidget):
    """Widget for editing a single shortcut."""
    
    def __init__(self, action_name: str, display_name: str, parent=None):
        super().__init__(parent)
        self.action_name = action_name
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(display_name)
        self.label.setMinimumWidth(150)
        
        self.key_edit = QKeySequenceEdit()
        current = config.get_shortcut(action_name)
        if current:
            self.key_edit.setKeySequence(QKeySequence(current))
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        
        layout.addWidget(self.label)
        layout.addWidget(self.key_edit, 1)
        layout.addWidget(self.clear_btn)
    
    def _clear(self):
        self.key_edit.clear()
    
    def get_shortcut(self) -> str:
        return self.key_edit.keySequence().toString()


class SettingsDialog(QDialog):
    """Settings dialog for customizing shortcuts and preferences."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 600)
        
        layout = QVBoxLayout(self)
        
        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Shortcuts tab
        shortcuts_widget = QWidget()
        shortcuts_layout = QVBoxLayout(shortcuts_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Shortcut editors
        self.shortcut_editors = []
        
        shortcut_definitions = [
            ("Global", [
                ("save", "Save"),
                ("quit", "Quit Application"),
                ("undo", "Undo"),
                ("run_analysis", "Run Analysis"),
            ]),
            ("Labeling / Edit", [
                ("label_normal", "Label as Normal"),
                ("label_rod", "Label as ROD"),
                ("label_artifact", "Label as Artifact"),
                ("pan_mode", "Pan Mode"),
                ("select_mode", "Select Mode"),
                ("add_mode", "Add Mode"),
                ("delete", "Delete Selected"),
                ("merge", "Merge Selected"),
                ("group", "Group Selected"),
                ("ungroup", "Ungroup Selected"),
            ]),
        ]
        
        for group_name, shortcuts in shortcut_definitions:
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()
            
            for action_name, display_name in shortcuts:
                editor = ShortcutEditor(action_name, display_name)
                self.shortcut_editors.append(editor)
                group_layout.addWidget(editor)
            
            group.setLayout(group_layout)
            scroll_layout.addWidget(group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        shortcuts_layout.addWidget(scroll)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_shortcuts)
        shortcuts_layout.addWidget(reset_btn)
        
        tabs.addTab(shortcuts_widget, "Shortcuts")
        
        # Performance tab
        perf_widget = QWidget()
        perf_layout = QVBoxLayout(perf_widget)
        
        parallel_group = QGroupBox("Parallel Processing")
        parallel_layout = QFormLayout()
        
        self.parallel_check = QCheckBox("Enable parallel image processing")
        self.parallel_check.setChecked(config.get("performance.parallel_enabled", True))
        self.parallel_check.setToolTip("Process multiple images simultaneously (faster on multi-core systems)")
        parallel_layout.addRow(self.parallel_check)
        
        # Worker options + the "Auto" count come from the hardware-aware engine
        # (scat.parallel) that the analysis actually uses, so the displayed "Auto (N)"
        # matches what runs — a fork process pool sized to the usable cores + free RAM.
        from .parallel import auto_worker_count, usable_cores
        cpu_count = usable_cores()
        self.auto_worker_count = auto_worker_count(10 ** 6)  # hardware cap for a large batch

        # Available worker counts (up to 32, but limited by usable cores)
        all_worker_options = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
        self.available_workers = [w for w in all_worker_options if w <= cpu_count]
        
        # Build combo box items
        self.workers_combo = QComboBox()
        worker_items = [f"Auto ({self.auto_worker_count})", "1 (sequential)"]
        for w in self.available_workers[1:]:  # Skip 1, already added
            worker_items.append(str(w))
        self.workers_combo.addItems(worker_items)
        
        # Set current value
        worker_setting = config.get("performance.worker_count", 0)  # 0 = auto
        if worker_setting == 0:
            self.workers_combo.setCurrentIndex(0)
        elif worker_setting == 1:
            self.workers_combo.setCurrentIndex(1)
        else:
            # Find index for this worker count
            try:
                idx = self.available_workers.index(worker_setting) + 1  # +1 for Auto at index 0
                self.workers_combo.setCurrentIndex(idx)
            except ValueError:
                self.workers_combo.setCurrentIndex(0)  # Default to Auto if not found
        
        parallel_layout.addRow("Workers:", self.workers_combo)

        # System info
        from .parallel import _available_gb
        _mem = _available_gb()
        sys_info = (f"Detected: {cpu_count} usable CPU cores"
                    + (f", {_mem:.1f} GB available RAM" if _mem else ""))
        
        info_label = QLabel(f"ℹ️ {sys_info}")
        info_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 11px;")
        parallel_layout.addRow(info_label)
        
        parallel_group.setLayout(parallel_layout)
        perf_layout.addWidget(parallel_group)
        perf_layout.addStretch()
        
        tabs.addTab(perf_widget, "Performance")

        # (Detection parameters live on the Analysis tab — the single place they are edited
        #  and the values actually used by a run. No duplicate Settings-dialog copy.)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._save_and_close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _reset_shortcuts(self):
        config.reset_shortcuts()
        # Reload editors
        for editor in self.shortcut_editors:
            current = config.get_shortcut(editor.action_name)
            if current:
                editor.key_edit.setKeySequence(QKeySequence(current))
            else:
                editor.key_edit.clear()
        QMessageBox.information(self, "Reset", "Shortcuts reset to defaults")
    
    def _save_and_close(self):
        # Save shortcuts
        for editor in self.shortcut_editors:
            config.set_shortcut(editor.action_name, editor.get_shortcut())
        
        # Save performance settings
        config.set("performance.parallel_enabled", self.parallel_check.isChecked())
        
        # Get worker count from combo index
        combo_idx = self.workers_combo.currentIndex()
        if combo_idx == 0:
            worker_count = 0  # Auto
        elif combo_idx == 1:
            worker_count = 1  # Sequential
        else:
            # Index 2+ corresponds to available_workers[1+]
            worker_count = self.available_workers[combo_idx - 1] if combo_idx - 1 < len(self.available_workers) else 0
        config.set("performance.worker_count", worker_count)
        
        self.accept()


class WorkerThread(QThread):
    """Background worker for long-running tasks."""
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class PathSelector(QWidget):
    """Widget for selecting file/folder paths."""
    
    pathChanged = Signal(str)  # Emitted when path changes
    
    def __init__(self, label: str, is_folder: bool = False, filter: str = "", config_key: str = "", default_path: str = ""):
        super().__init__()
        self.is_folder = is_folder
        self.filter = filter
        self.config_key = config_key
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        
        # Label - bold, no colon, no separate box
        label_text = label.rstrip(':')  # Remove colon if present
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(70)
        self.label.setStyleSheet(f"""
            font-weight: bold;
            color: {Theme.TEXT_PRIMARY};
            background-color: transparent;
        """)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(f"Select {label_text.lower()}...")
        self.path_edit.textChanged.connect(self.pathChanged.emit)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setMinimumWidth(80)
        self.browse_btn.clicked.connect(self._browse)
        
        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_btn)
        
        # Load from config, fallback to default_path
        if config_key:
            saved_path = config.get(config_key, "")
            if saved_path:
                self.path_edit.setText(saved_path)
            elif default_path:
                self.path_edit.setText(default_path)
    
    def _browse(self):
        start_dir = self.path_edit.text().replace('/', '\\') or ""  # Convert back for dialog
        
        if self.is_folder:
            path = QFileDialog.getExistingDirectory(self, f"Select {self.label.text()}", start_dir)
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {self.label.text()}", start_dir, self.filter)
        
        if path:
            # Display with forward slashes to avoid KRW symbol on Korean Windows
            display_path = path.replace('\\', '/')
            self.path_edit.setText(display_path)
            if self.config_key:
                config.set(self.config_key, path)  # Store original path
    
    def path(self) -> str:
        # Return path with system-appropriate separators
        return self.path_edit.text().replace('/', '\\') if sys.platform == 'win32' else self.path_edit.text()
    
    def set_path(self, path: str):
        # Display with forward slashes
        display_path = path.replace('\\', '/') if path else ''
        self.path_edit.setText(display_path)
        if self.config_key:
            config.set(self.config_key, path)


class TrainingTab(QWidget):
    """Training tab for model training."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        # Main layout for the tab
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        # Scroll content — centered, width-capped column so the form doesn't stretch on widescreen.
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        outer = QVBoxLayout(scroll_content)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(0)
        _cap = CenteredCap(760)
        _cap.content_layout.setSpacing(14)
        outer.addWidget(_cap)
        outer.addStretch(1)
        layout = _cap.content_layout   # the group boxes below are added to this capped column

        # Data paths
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout()
        
        self.image_dir = PathSelector("Image Folder:", is_folder=True, config_key="last_image_dir")
        self.label_dir = PathSelector("Label Folder:", is_folder=True, config_key="last_label_dir")
        self.same_folder = QCheckBox("Same as image folder")
        self.same_folder.setChecked(True)
        self.same_folder.toggled.connect(self._toggle_label_dir)
        self.image_dir.pathChanged.connect(self._on_image_dir_changed)
        
        data_layout.addWidget(self.image_dir)
        data_layout.addWidget(self.label_dir)
        data_layout.addWidget(self.same_folder)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        self._toggle_label_dir(True)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        self.model_type = NoScrollComboBox()
        self.model_type.addItems(["Random Forest", "CNN (PyTorch)", "U-Net Segmentation"])
        model_type_map = {"rf": 0, "cnn": 1, "unet": 2}
        self.model_type.setCurrentIndex(model_type_map.get(config.get("training.model_type", "rf"), 0))
        self.model_type.currentIndexChanged.connect(self._on_model_type_changed)
        model_layout.addRow("Model Type:", self.model_type)
        
        self.output_path = PathSelector("Output:", filter="Model (*.pkl *.pt)")
        model_layout.addRow(self.output_path)
        
        self.n_estimators = NoScrollSpinBox()
        self.n_estimators.setRange(10, 1000)
        self.n_estimators.setValue(config.get("training.n_estimators", 100))
        self.n_estimators.setButtonSymbols(QSpinBox.NoButtons)
        model_layout.addRow("RF Trees:", self.n_estimators)
        
        self.epochs = NoScrollSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(config.get("training.epochs", 20))
        self.epochs.setButtonSymbols(QSpinBox.NoButtons)
        model_layout.addRow("CNN/U-Net Epochs:", self.epochs)
        
        self.image_size = NoScrollSpinBox()
        self.image_size.setRange(128, 512)
        self.image_size.setValue(256)
        self.image_size.setSingleStep(64)
        self.image_size.setButtonSymbols(QSpinBox.NoButtons)
        model_layout.addRow("U-Net Image Size:", self.image_size)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Train button (primary action) — centered, not a full-width banner
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setMinimumHeight(44)
        self.train_btn.setMinimumWidth(200)
        self.train_btn.setStyleSheet(
            Theme.button_style(Theme.PRIMARY, "#FFFFFF", Theme.PRIMARY_LIGHT, Theme.PRIMARY_DARK))
        ui_motion.attach_button_motion(self.train_btn, primary=True)
        self.train_btn.clicked.connect(self._train)
        _train_row = QHBoxLayout()
        _train_row.addStretch(1)
        _train_row.addWidget(self.train_btn)
        _train_row.addStretch(1)
        layout.addLayout(_train_row)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(200)
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        # Set scroll content and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
    
    def _toggle_label_dir(self, checked):
        self.label_dir.setEnabled(not checked)
        if checked:
            # Sync label_dir path with image_dir path
            self.label_dir.set_path(self.image_dir.path())
    
    def _on_image_dir_changed(self, path):
        """When image dir changes, sync label dir if same_folder is checked."""
        if self.same_folder.isChecked():
            self.label_dir.set_path(path)
    
    def _on_model_type_changed(self, index):
        """Show/hide settings based on model type."""
        is_rf = (index == 0)
        is_unet = (index == 2)
        
        self.n_estimators.setEnabled(is_rf)
        self.image_size.setEnabled(is_unet)
    
    def _train(self):
        image_dir = self.image_dir.path()
        if not image_dir:
            QMessageBox.warning(self, "Error", "Please select image folder")
            return
        
        label_dir = image_dir if self.same_folder.isChecked() else self.label_dir.path()
        output_path = self.output_path.path()
        
        model_type_idx = self.model_type.currentIndex()
        model_type = ["rf", "cnn", "unet"][model_type_idx]
        
        if not output_path:
            ext = ".pkl" if model_type == "rf" else ".pt"
            output_path = str(Path(image_dir) / f"model_{model_type}{ext}")
            self.output_path.set_path(output_path)
        
        # Save settings
        config.set("training.model_type", model_type)
        config.set("training.n_estimators", self.n_estimators.value())
        config.set("training.epochs", self.epochs.value())
        
        self.log.clear()
        self.log.append(f"Starting {model_type.upper()} training...")
        self.train_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        
        if model_type == "unet":
            # U-Net segmentation training
            from .segmentation import train_segmentation_model
            
            def progress_callback(epoch, train_loss, val_loss, val_iou):
                self.log.append(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, IoU={val_iou:.3f}")
            
            self.worker = WorkerThread(
                train_segmentation_model,
                image_dir=image_dir,
                label_dir=label_dir,
                output_path=output_path,
                epochs=self.epochs.value(),
                image_size=self.image_size.value(),
                progress_callback=progress_callback
            )
        else:
            # RF or CNN classifier training
            kwargs = {}
            if model_type == "rf":
                kwargs['n_estimators'] = self.n_estimators.value()
            else:
                kwargs['epochs'] = self.epochs.value()
            
            # Lazy import trainer module
            from .trainer import train_from_labels
            
            self.worker = WorkerThread(
                train_from_labels,
                image_dir=image_dir,
                label_dir=label_dir,
                output_path=output_path,
                model_type=model_type,
                **kwargs
            )
        
        self.worker.finished.connect(self._on_train_finished)
        self.worker.error.connect(self._on_train_error)
        self.worker.start()
    
    def _on_train_finished(self, results):
        self.train_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        self.log.append("\n" + "="*40)
        self.log.append("TRAINING COMPLETE!")
        self.log.append("="*40)
        
        if 'accuracy' in results:
            self.log.append(f"Accuracy: {results.get('accuracy', 0)*100:.1f}%")
        
        if 'cv_mean' in results:
            self.log.append(f"Cross-validation: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
        
        if 'best_val_loss' in results:
            self.log.append(f"Best validation loss: {results['best_val_loss']:.4f}")
        
        if 'val_iou' in results and results['val_iou']:
            self.log.append(f"Final IoU: {results['val_iou'][-1]:.3f}")
        
        QMessageBox.information(self, "Success", f"Model saved to:\n{self.output_path.path()}")
    
    def _on_train_error(self, error_msg):
        self.train_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.log.append(f"\nERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Training failed:\n{error_msg}")


def _results_dict_from_output(output_dir, group_by=None, image_paths=None, stats=None):
    """Rebuild the Results-tab dict from an analysis output directory (the source of truth).

    analyze_folder_service returns only aggregate scalars; the per-image/per-deposit data,
    visualizations and spatial stats live on disk. Shared by the Run flow and 'Load results'.
    `stats` (the full run_statistics_service dict) is flattened to basic.metrics because the
    Results-tab renderer expects a flat {metric: comparison} mapping.
    """
    import json as _json
    out = Path(output_dir)
    summary_path = out / IMAGE_SUMMARY
    if not summary_path.exists():
        summary_path = out / "film_summary.csv"  # backward compatibility
    film_summary = pd.read_csv(summary_path)
    dep_path = out / ALL_DEPOSITS
    deposit_data = pd.read_csv(dep_path) if dep_path.exists() else None

    viz_results = {}
    viz_dir = out / "visualizations"
    if viz_dir.exists():
        for png in viz_dir.glob("*.png"):
            viz_results[png.stem] = str(png)

    spatial_stats = {}
    sp_path = out / "spatial_stats.json"
    if sp_path.exists():
        try:
            spatial_stats = _json.loads(sp_path.read_text())
        except Exception:
            spatial_stats = {}

    if group_by is None and "group" in film_summary.columns:
        if film_summary["group"].dropna().nunique() > 1:
            group_by = "group"

    stats_results = {}
    if stats and not stats.get("skipped"):
        stats_results = stats.get("basic", {}).get("metrics", {}) or {}

    analysis = {}
    mpath = out / "run_manifest.json"
    if mpath.exists():
        try:
            analysis = (_json.loads(mpath.read_text()) or {}).get("analysis", {}) or {}
        except Exception:
            analysis = {}

    return {
        "output_dir": str(out),
        "film_summary": film_summary,   # holds image_summary; key kept for compatibility
        "deposit_data": deposit_data,
        "viz_results": viz_results,
        "spatial_stats": spatial_stats,
        "stats_results": stats_results,
        "group_by": group_by,
        "image_paths": list(image_paths) if image_paths else [],
        "primary_metric": _metrics.resolve_metric(analysis.get("primary_metric")),
        "normalization": analysis.get("normalization") or _metrics.DEFAULT_NORMALIZATION,
        "confidence_threshold": float(analysis.get("confidence_threshold", _metrics.DEFAULT_THRESHOLD)),
        "run_meta": {},   # n_flies/roi_area/duration land here in the metadata-capture task
    }


class DropZone(QWidget):
    """A drag-and-drop hero for image input. Accepts dropped image files, folders (searched
    recursively), and multiple items at once; also two Browse actions (images / folder).
    Emits ``filesSelected(list[str])`` with the resolved image file paths."""

    IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    filesSelected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self._active = False
        self._compact_mode = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Hero: shown BEFORE any images are chosen ----
        self._hero = QWidget()
        hv = QVBoxLayout(self._hero)
        hv.setContentsMargins(18, 22, 18, 20)
        hv.setSpacing(6)
        self._icon = QLabel()
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setPixmap(icon("add_photo_alternate", Theme.TEXT_SECONDARY, 36).pixmap(36, 36))
        hv.addWidget(self._icon, 0, Qt.AlignHCenter)
        self._title = QLabel("Drop images or a folder here")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet(
            f"color:{Theme.TEXT_PRIMARY}; font-weight:{Theme.WEIGHT_TITLE}; background:transparent;")
        hv.addWidget(self._title)
        self._sub = QLabel("TIFF · PNG · JPG  —  folders are searched recursively")
        self._sub.setAlignment(Qt.AlignCenter)
        self._sub.setStyleSheet(f"color:{Theme.TEXT_MUTED}; font-size:{Theme.FS_XS}px; background:transparent;")
        hv.addWidget(self._sub)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        self._img_btn = QPushButton("Choose images…")
        self._img_btn.setIcon(icon("image"))
        self._img_btn.clicked.connect(self._choose_images)
        self._dir_btn = QPushButton("Choose folder…")
        self._dir_btn.setIcon(icon("folder_open"))
        self._dir_btn.clicked.connect(self._choose_folder)
        btn_row.addWidget(self._img_btn)
        btn_row.addWidget(self._dir_btn)
        btn_row.addStretch(1)
        hv.addSpacing(4)
        hv.addLayout(btn_row)
        root.addWidget(self._hero)

        # ---- Compact chip: shown once images are chosen ----
        self._compact = QWidget()
        cv = QHBoxLayout(self._compact)
        cv.setContentsMargins(14, 10, 12, 10)
        cv.setSpacing(10)
        c_icon = QLabel()
        c_icon.setPixmap(icon("photo_library", Theme.NORMAL, 22).pixmap(22, 22))
        cv.addWidget(c_icon)
        self._c_label = QLabel("")
        self._c_label.setStyleSheet(
            f"color:{Theme.TEXT_PRIMARY}; font-weight:{Theme.WEIGHT_LABEL}; background:transparent;")
        cv.addWidget(self._c_label, 1)
        change_btn = QPushButton("Change")
        change_btn.setIcon(icon("refresh"))
        change_btn.setToolTip("Choose a different set of images (or drop new ones)")
        change_btn.clicked.connect(self._show_hero)
        cv.addWidget(change_btn)
        self._compact.setVisible(False)
        root.addWidget(self._compact)

        self._apply_style()

    # -- appearance --
    def _apply_style(self):
        border = Theme.PRIMARY if self._active else Theme.BORDER
        if self._compact_mode:
            self.setStyleSheet(
                f"QWidget#dropZone {{ background-color: {Theme.BG_SURFACE}; border: 1px solid {border}; "
                f"border-radius: {Theme.RADIUS_CONTAINER}px; }}")
        else:
            bg = "rgba(218,78,66,0.10)" if self._active else Theme.BG_INSET
            self.setStyleSheet(
                f"QWidget#dropZone {{ background-color: {bg}; border: 2px dashed {border}; "
                f"border-radius: {Theme.RADIUS_CONTAINER}px; }}")

    def _set_active(self, on):
        if on != self._active:
            self._active = on
            self._apply_style()

    def _show_hero(self):
        """Revert to the drop hero so the user can drop/browse a different set. The current
        selection stays active until replaced."""
        self._compact_mode = False
        self._compact.setVisible(False)
        self._hero.setVisible(True)
        self._apply_style()

    def set_count(self, n: int):
        self._compact_mode = n > 0
        if n > 0:
            self._c_label.setText(f"{n} image{'s' if n != 1 else ''} selected")
            self._hero.setVisible(False)
            self._compact.setVisible(True)
        else:
            self._hero.setVisible(True)
            self._compact.setVisible(False)
        self._apply_style()

    # -- drag & drop --
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_active(True)

    def dragLeaveEvent(self, event):
        self._set_active(False)

    def dropEvent(self, event):
        self._set_active(False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        files = self._collect(paths)
        if files:
            event.acceptProposedAction()
            self.filesSelected.emit(files)

    # -- browse --
    def _choose_images(self):
        start = config.get("last_input_dir", "")
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select image files", start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if files:
            self.filesSelected.emit(self._collect(files))

    def _choose_folder(self):
        start = config.get("last_input_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of images", start)
        if folder:
            files = self._collect([folder])
            if files:
                self.filesSelected.emit(files)
            else:
                QMessageBox.information(self, "No images", "No supported images found in that folder.")

    def _collect(self, paths) -> list:
        """Expand a mix of files and folders into a sorted, de-duplicated list of image paths."""
        out = []
        for pth in paths:
            p = Path(pth)
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in self.IMAGE_EXTS:
                        out.append(str(f))
            elif p.is_file() and p.suffix.lower() in self.IMAGE_EXTS:
                out.append(str(p))
        return sorted(dict.fromkeys(out))


class AnalysisTab(QWidget):
    """Analysis tab for running analysis on images."""

    analysis_complete = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self._metadata = None  # In-memory metadata from GroupEditor
        self._group_data = {}  # {group_name: [file1, file2, ...]}
        self._selected_files: List[str] = []  # Individual file selection
        self._input_mode = 'folder'  # 'folder' or 'files'
        self._image_files_for_analysis: List[str] = []  # Files to analyze

        # A stateful workspace, not two tabs: CONFIGURE ↔ RESULTS. Before a run the config is
        # the content; once results exist they take over and the config collapses to a one-line
        # summary bar (Apple/Claude: state drives the view, not a persistent mode switch).
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)
        self._configure_page = QWidget()
        self._results_page = QWidget()
        self.stack.addWidget(self._configure_page)
        self.stack.addWidget(self._results_page)
        self._setup_ui()               # builds the configure page
        self._build_results_page()     # builds the results page (embeds ResultsTab)
        self.stack.setCurrentWidget(self._configure_page)

    def _setup_ui(self):
        # Configure page: the form + live summary/preview panel + sticky Run footer.
        main_layout = QVBoxLayout(self._configure_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        outer = QVBoxLayout(scroll_content)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        # Body: configuration form (left, readable width) + a live run-summary / preview panel
        # (right) that FILLS the remaining width — empty gutters become useful context
        # (Apple content-first, Claude helpful). Sticky Run footer stays below.
        body = QHBoxLayout()
        body.setSpacing(16)

        form_col_w = QWidget()
        form_col_w.setMaximumWidth(560)
        form_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_col = QVBoxLayout(form_col_w)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(12)

        default_input = str(Path.home() / "SCAT" / "data" / "images")
        default_output = str(Path.home() / "SCAT" / "data" / "results")

        # ---- Input: a drag-and-drop hero (images, folders, multiple at once) ----
        self.dropzone = DropZone()
        self.dropzone.filesSelected.connect(self._on_files_selected)
        left_col.addWidget(self.dropzone)

        # ---- Output + model ----
        io_group = QGroupBox("Output")
        io_layout = QVBoxLayout()
        io_layout.setSpacing(6)
        io_layout.setContentsMargins(14, 16, 14, 14)
        self.output_dir = PathSelector("Results folder", is_folder=True, config_key="last_output_dir", default_path=default_output)
        # "Classifier model" (the model file) — distinct from the "Method" picker in Options.
        self.model_path = PathSelector("Classifier model", filter="Model (*.pkl *.pt)", config_key="last_model_path")
        # U-Net detection model lives under Advanced (optional / rarely changed).
        self.detection_model_path = PathSelector("Detection (U-Net)", filter="Model (*.pt)", config_key="last_detection_model_path")
        self.detection_model_path.setToolTip("Optional: U-Net model for improved deposit detection")
        io_layout.addWidget(self.output_dir)
        io_layout.addWidget(self.model_path)
        io_group.setLayout(io_layout)
        left_col.addWidget(io_group)

        # ---- Options: Apple-style rows (label + description on the left, control on the right) ----
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_layout.setSpacing(2)
        options_layout.setContentsMargins(14, 16, 14, 12)

        self.model_type = NoScrollComboBox()
        self.model_type.addItems(["Threshold", "Random Forest", "CNN"])
        model_type_map = {"threshold": 0, "rf": 1, "cnn": 2}
        self.model_type.setCurrentIndex(model_type_map.get(config.get("analysis.model_type", "rf"), 1))
        self.model_type.setMinimumWidth(150)
        self.model_type.currentIndexChanged.connect(self._update_run_summary)
        options_layout.addWidget(setting_row("Method", self.model_type, "How deposits are classified"))
        options_layout.addWidget(self._divider())

        self.annotate = ToggleSwitch()
        self.annotate.setChecked(config.get("analysis.annotate", True))
        self.annotate.toggled.connect(self._update_run_summary)
        options_layout.addWidget(setting_row("Annotated images", self.annotate, "Save each image with deposits outlined"))
        self.visualize = ToggleSwitch()
        self.visualize.setChecked(config.get("analysis.visualize", True))
        self.visualize.toggled.connect(self._update_run_summary)
        options_layout.addWidget(setting_row("Visualizations", self.visualize, "Distribution plots and comparisons"))
        self.report = ToggleSwitch()
        self.report.setChecked(config.get("analysis.report", True))
        self.report.toggled.connect(self._update_run_summary)
        options_layout.addWidget(setting_row("HTML report", self.report, "A shareable summary document"))
        options_group.setLayout(options_layout)
        left_col.addWidget(options_group)

        # ---- Groups ----
        groups_group = QGroupBox("Groups")
        groups_layout = QVBoxLayout()
        groups_layout.setSpacing(8)
        groups_layout.setContentsMargins(14, 16, 14, 14)
        self.use_groups = ToggleSwitch()
        self.use_groups.setChecked(config.get("analysis.use_groups", True))
        self.use_groups.toggled.connect(self._on_use_groups_toggled)
        groups_layout.addWidget(setting_row("Compare groups", self.use_groups,
                                            "Run statistics across experimental conditions"))

        # Grouping actions side by side (no longer full-width banner buttons).
        group_btn_row = QHBoxLayout()
        group_btn_row.setSpacing(8)
        self.autogroup_btn = QPushButton("Group by subfolder")
        self.autogroup_btn.setToolTip(
            "Assign each selected image to a group named after its parent subfolder.\n"
            "Double-click a group in the list below to rename it.")
        # wrap: QPushButton.clicked passes a `checked` bool that would shadow `announce`
        self.autogroup_btn.clicked.connect(lambda: self._autogroup_by_subfolder(announce=True))
        self.load_groups_csv_btn = QPushButton("Load grouping CSV…")
        self.load_groups_csv_btn.setToolTip(
            "Load an explicit filename→group mapping from a CSV (columns: filename, group).\n"
            "Overrides subfolder grouping; matches images by filename.")
        self.load_groups_csv_btn.clicked.connect(self._load_grouping_csv)
        group_btn_row.addWidget(self.autogroup_btn, 1)
        group_btn_row.addWidget(self.load_groups_csv_btn, 1)
        groups_layout.addLayout(group_btn_row)

        self.groups_hint = QLabel("Select images, then group by subfolder. Double-click a group to rename.")
        self.groups_hint.setWordWrap(True)
        self.groups_hint.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: {Theme.FS_XS}px;")
        groups_layout.addWidget(self.groups_hint)

        self.groups_tree = QTreeWidget()
        self.groups_tree.setHeaderHidden(True)
        self.groups_tree.setMinimumHeight(140)
        self.groups_tree.setIndentation(15)
        self.groups_tree.setAnimated(True)
        self.groups_tree.itemClicked.connect(self._on_group_tree_clicked)
        self.groups_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.groups_tree.customContextMenuRequested.connect(self._on_groups_context_menu)
        groups_layout.addWidget(self.groups_tree, 1)
        groups_group.setLayout(groups_layout)
        left_col.addWidget(groups_group)

        # ---- Advanced settings (collapsible) ----
        advanced = CollapsibleSection("Advanced settings", expanded=False)
        advanced.add_widget(self.detection_model_path)

        detect_form_w = QWidget()
        detect_layout = QFormLayout(detect_form_w)
        detect_layout.setVerticalSpacing(8)
        detect_layout.setHorizontalSpacing(12)
        detect_layout.setContentsMargins(0, 0, 0, 0)
        self.min_area = NoScrollSpinBox()
        self.min_area.setRange(1, 1000)
        self.min_area.setValue(config.get("detection.min_area", 20))
        self.min_area.setButtonSymbols(QSpinBox.NoButtons)
        detect_layout.addRow("Min Area", self.min_area)
        self.max_area = NoScrollSpinBox()
        self.max_area.setRange(100, 50000)
        self.max_area.setValue(config.get("detection.max_area", 10000))
        self.max_area.setButtonSymbols(QSpinBox.NoButtons)
        detect_layout.addRow("Max Area", self.max_area)
        self.threshold = NoScrollDoubleSpinBox()
        self.threshold.setRange(0.1, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(config.get("detection.threshold", 0.6))
        self.threshold.setButtonSymbols(QDoubleSpinBox.NoButtons)
        detect_layout.addRow("Circularity", self.threshold)
        advanced.add_widget(detect_form_w)

        self.spatial = ToggleSwitch()
        self.spatial.setChecked(config.get("analysis.spatial", True))
        self.spatial.toggled.connect(self._update_run_summary)
        advanced.add_widget(setting_row("Spatial analysis", self.spatial, "Clustering / dispersion metrics"))
        self.stats = ToggleSwitch()
        self.stats.setChecked(config.get("analysis.stats", True))
        self.stats.toggled.connect(self._update_run_summary)
        advanced.add_widget(setting_row("Statistical analysis", self.stats, "Group comparison tests"))
        self.save_json = ToggleSwitch()
        self.save_json.setChecked(config.get("analysis.save_json", True))
        self.save_json.setToolTip("Save contour data for model retraining. Disable to reduce file size.")
        advanced.add_widget(setting_row("Save for retraining", self.save_json, "Store contour data (JSON)"))
        left_col.addWidget(advanced)
        left_col.addStretch(1)

        body.addWidget(form_col_w, 0)

        # ---- Right: live run summary + input preview (fills the width) ----
        summary_panel = self._build_summary_panel()
        body.addWidget(summary_panel, 1)
        outer.addLayout(body, 1)

        # Refresh the summary when the output path is edited too.
        self.output_dir.pathChanged.connect(self._update_run_summary)

        # Update groups UI state
        self._on_use_groups_toggled(self.use_groups.isChecked())

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)

        # ---- Sticky footer: progress + Run (always reachable) ----
        footer = QWidget()
        footer.setObjectName("runFooter")
        footer.setStyleSheet(
            f"QWidget#runFooter {{ background-color: {Theme.BG_BASE}; border-top: 1px solid {Theme.BORDER}; }}")
        footer_v = QVBoxLayout(footer)
        footer_v.setContentsMargins(6, 8, 6, 8)
        footer_center = QHBoxLayout()
        footer_center.addStretch(1)
        footer_inner = QWidget()
        footer_inner.setMaximumWidth(1080)
        footer_row = QHBoxLayout(footer_inner)
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setMinimumWidth(320)
        self.progress_label = QLabel("")
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        self.progress.setVisible(False)          # shown once a run starts
        self.progress_label.setVisible(False)
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setIcon(icon("play_arrow", "#FFFFFF"))
        self.run_btn.setMinimumHeight(48)
        self.run_btn.setMinimumWidth(240)
        self.run_btn.setStyleSheet(
            Theme.button_style(Theme.PRIMARY, "#FFFFFF", Theme.PRIMARY_LIGHT, Theme.PRIMARY_DARK))
        # Responsive depth on the primary action: a coral-tinted shadow lifts on hover and
        # depresses on press (coral reads on the near-black theme where a neutral shadow can't).
        ui_motion.attach_button_motion(self.run_btn, primary=True)
        self.run_btn.clicked.connect(self._run_analysis)
        self.load_prev_btn = QPushButton("Load previous results…")
        self.load_prev_btn.setToolTip("Open results from a previous analysis session")
        self.load_prev_btn.clicked.connect(self._open_previous_results)
        footer_row.addWidget(self.load_prev_btn)
        footer_row.addStretch(1)
        footer_row.addWidget(self.progress)
        footer_row.addWidget(self.progress_label)
        footer_row.addWidget(self.eta_label)
        footer_row.addWidget(self.run_btn)
        footer_center.addWidget(footer_inner, 0)
        footer_center.addStretch(1)
        footer_v.addLayout(footer_center)
        main_layout.addWidget(footer)

        self._start_time = None
        self._update_run_summary()
        self._update_preview()

    # ---- Live run-summary / preview panel -------------------------------------
    def _build_summary_panel(self):
        """Right-hand context panel: a live 'Run summary' + input thumbnail preview. Fills the
        width the form doesn't use with self-updating context (Apple content-first / Claude helpful)."""
        panel = QGroupBox("Run summary")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 18, 16, 16)
        v.setSpacing(12)

        self.summary_status = QLabel("Select images to begin")
        self.summary_status.setWordWrap(True)
        self.summary_status.setStyleSheet(
            f"color: {Theme.TEXT_PRIMARY}; font-size: {Theme.FS_TITLE}px; font-weight: {Theme.WEIGHT_TITLE};")
        v.addWidget(self.summary_status)

        self.summary_detail = QLabel("")
        self.summary_detail.setWordWrap(True)
        self.summary_detail.setTextFormat(Qt.RichText)
        self.summary_detail.setStyleSheet("background: transparent;")
        v.addWidget(self.summary_detail)

        self.preview_title = QLabel("Preview")
        self.preview_title.setStyleSheet(
            f"color: {Theme.TEXT_SECONDARY}; font-size: {Theme.FS_XS}px; letter-spacing: 1px;")
        self.preview_title.setVisible(False)
        v.addWidget(self.preview_title)

        self.preview_holder = QWidget()
        self.preview_grid = QGridLayout(self.preview_holder)
        self.preview_grid.setContentsMargins(0, 0, 0, 0)
        self.preview_grid.setSpacing(8)
        self.preview_grid.setColumnStretch(4, 1)   # pack thumbnails left; extra width to the right
        v.addWidget(self.preview_holder)

        v.addStretch(1)
        return panel

    def _update_run_summary(self, *args):
        """Refresh the live run-summary text from the current configuration."""
        if not hasattr(self, "summary_status"):
            return
        sec = Theme.TEXT_SECONDARY
        n = len(self._selected_files)
        if n == 0:
            self.summary_status.setText("Select images to begin")
            step = (lambda num, title, hint:
                    f"<div style='color:{Theme.TEXT_PRIMARY}; margin:5px 0'>"
                    f"<span style='color:{Theme.PRIMARY}; font-weight:600'>{num}</span>&nbsp;&nbsp;{title}"
                    f" &nbsp;<span style='color:{sec}'>{hint}</span></div>")
            self.summary_detail.setText(
                f"<div style='color:{sec}; margin-bottom:16px'>Choose one or more images, then "
                f"configure the options on the left.</div>"
                f"<div style='color:{sec}; letter-spacing:1px; font-size:11px; margin-bottom:6px'>HOW IT WORKS</div>"
                + step("1", "Add images", "— drop files or a folder, or Browse")
                + step("2", "Group by condition", "— optional, from subfolders")
                + step("3", "Choose outputs &amp; Run", "— report, plots, statistics"))
            return

        self.summary_status.setText(f"Ready to analyze {n} image{'s' if n != 1 else ''}")

        if self.use_groups.isChecked() and self._group_data:
            g = len(self._group_data)
            grouping = f"{g} group{'s' if g != 1 else ''}"
        elif self.use_groups.isChecked():
            grouping = "on (not derived yet)"
        else:
            grouping = "single group"

        method = self.model_type.currentText()
        out = self.output_dir.path() or "default (next to input)"
        out_disp = ("…" + out[-40:]) if len(out) > 42 else out

        produces = []
        if self.annotate.isChecked(): produces.append("Annotated images")
        if self.visualize.isChecked(): produces.append("Visualizations")
        if self.stats.isChecked(): produces.append("Statistics")
        if self.report.isChecked(): produces.append("HTML report")
        prod = "".join(
            f"<div style='margin:2px 0; color:{Theme.TEXT_PRIMARY}'>&#10003;&nbsp; {p}</div>"
            for p in produces) or f"<div style='color:{sec}'>—</div>"

        est = max(1, round(n * 1.5))
        est_txt = f"~{est} sec" if est < 90 else f"~{round(est / 60)} min"

        rows = [("Images", str(n)), ("Grouping", grouping), ("Method", method),
                ("Output", out_disp), ("Est. time", est_txt)]
        table = "".join(
            f"<tr><td style='color:{sec}; padding:2px 16px 2px 0'>{k}</td>"
            f"<td style='color:{Theme.TEXT_PRIMARY}'>{val}</td></tr>" for k, val in rows)
        self.summary_detail.setText(
            f"<table cellspacing='0'>{table}</table>"
            f"<div style='color:{sec}; margin-top:14px; margin-bottom:2px; "
            f"letter-spacing:1px; font-size:11px'>WILL PRODUCE</div>{prod}")

    def _update_preview(self):
        """Refresh the input thumbnail grid (first few selected images)."""
        if not hasattr(self, "preview_grid"):
            return
        while self.preview_grid.count():
            it = self.preview_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        files = self._selected_files[:6]
        self.preview_title.setVisible(bool(files))
        base = f"background: {Theme.BG_INSET}; border: 1px solid {Theme.BORDER}; border-radius: 6px;"
        for idx, f in enumerate(files):
            cell = QLabel()
            cell.setFixedSize(92, 92)
            cell.setAlignment(Qt.AlignCenter)
            pm = QPixmap(f)
            if not pm.isNull():
                cell.setStyleSheet(base)
                cell.setPixmap(pm.scaled(88, 88, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                cell.setStyleSheet(base + f" color: {Theme.TEXT_MUTED}; font-size: 10px;")
                cell.setText(Path(f).suffix.lstrip('.').upper() or "IMG")
            cell.setToolTip(Path(f).name)
            self.preview_grid.addWidget(cell, idx // 3, idx % 3)
        extra = len(self._selected_files) - len(files)
        if extra > 0:
            more = QLabel(f"+{extra}")
            more.setFixedSize(92, 92)
            more.setAlignment(Qt.AlignCenter)
            more.setStyleSheet(base + f" color: {Theme.TEXT_SECONDARY};")
            self.preview_grid.addWidget(more, len(files) // 3, len(files) % 3)

    # ---- Stateful workspace: RESULTS state ------------------------------------
    def _build_results_page(self):
        """Results state: a compact config-summary bar (New / Re-run) above the embedded
        results view. Once results exist they are the content; the config collapses to one line."""
        v = QVBoxLayout(self._results_page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QWidget()
        bar.setObjectName("resultsBar")
        bar.setStyleSheet(
            f"QWidget#resultsBar {{ background-color: {Theme.BG_BASE}; border-bottom: 1px solid {Theme.BORDER}; }}")
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(14, 10, 14, 10)
        bh.setSpacing(12)
        back_btn = QPushButton("New analysis")
        back_btn.setIcon(icon("arrow_back"))
        back_btn.setToolTip("Back to configuration")
        back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self._configure_page))
        self.results_bar_label = QLabel("")
        self.results_bar_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        rerun_btn = QPushButton("Re-run")
        rerun_btn.setIcon(icon("refresh"))
        rerun_btn.setToolTip("Run again with the current settings")
        rerun_btn.clicked.connect(self._rerun)
        bh.addWidget(back_btn)
        bh.addWidget(self.results_bar_label, 1)
        bh.addWidget(rerun_btn)
        v.addWidget(bar)

        self.results_view = ResultsTab()
        v.addWidget(self.results_view, 1)

    def _show_results(self, results):
        """Switch the workspace into the RESULTS state, loading the given results."""
        self.results_view.load_results(results)
        self._update_results_bar(results)
        self.stack.setCurrentWidget(self._results_page)

    def _update_results_bar(self, results):
        fs = results.get('film_summary') if isinstance(results, dict) else None
        try:
            n = len(fs) if fs is not None else len(self._selected_files)
        except Exception:
            n = len(self._selected_files)
        method = self.model_type.currentText()
        if self.use_groups.isChecked() and self._group_data:
            g = len(self._group_data)
            grp = f"{g} group{'s' if g != 1 else ''}"
        else:
            grp = "single group"
        self.results_bar_label.setText(
            f"✓  Analyzed {n} image{'s' if n != 1 else ''}   ·   {method}   ·   {grp}")

    def _rerun(self):
        self.stack.setCurrentWidget(self._configure_page)
        self._run_analysis()

    def _open_previous_results(self):
        """Load a previous session's results and jump straight to the RESULTS state."""
        self.results_view._load_previous_results()
        if getattr(self.results_view, "results", None):
            self._update_results_bar(self.results_view.results)
            self.stack.setCurrentWidget(self._results_page)

    def _save_settings(self):
        """Save current settings to config."""
        model_types = ['threshold', 'rf', 'cnn']
        config.set("analysis.model_type", model_types[self.model_type.currentIndex()])
        config.set("analysis.use_groups", self.use_groups.isChecked())
        config.set("analysis.annotate", self.annotate.isChecked())
        config.set("analysis.visualize", self.visualize.isChecked())
        config.set("analysis.spatial", self.spatial.isChecked())
        config.set("analysis.stats", self.stats.isChecked())
        config.set("analysis.report", self.report.isChecked())
        config.set("analysis.save_json", self.save_json.isChecked())
        config.set("detection.min_area", self.min_area.value())
        config.set("detection.max_area", self.max_area.value())
        config.set("detection.threshold", self.threshold.value())
    
    def _on_use_groups_toggled(self, checked):
        """Enable/disable groups UI based on checkbox."""
        self.autogroup_btn.setEnabled(checked)
        self.load_groups_csv_btn.setEnabled(checked)
        self.groups_tree.setEnabled(checked)
        self.groups_hint.setEnabled(checked)
        self._update_run_summary()
    
    def _on_group_tree_clicked(self, item, column):
        """Toggle expand/collapse on single click for parent items."""
        if item.childCount() > 0:  # Only for parent items (groups)
            item.setExpanded(not item.isExpanded())
    
    def _update_groups_list(self, group_data: dict):
        """Update the groups tree widget with group data."""
        self._group_data = group_data
        self.groups_tree.clear()
        
        for group_name, files in sorted(group_data.items()):
            # Parent item (group name with count); stash the raw name for rename lookup
            parent = QTreeWidgetItem([f"{group_name} ({len(files)} files)"])
            parent.setData(0, Qt.UserRole, group_name)
            parent.setExpanded(False)

            # Child items (file names)
            for filename in sorted(files):
                child = QTreeWidgetItem([f"  {filename}"])
                child.setForeground(0, QColor(Theme.TEXT_SECONDARY))
                parent.addChild(child)

            self.groups_tree.addTopLevelItem(parent)
        self._update_run_summary()

    def _divider(self):
        """A thin 1px horizontal rule for separating settings rows."""
        d = QWidget()
        d.setFixedHeight(1)
        d.setStyleSheet(f"background-color: {Theme.BORDER};")
        return d

    def _on_files_selected(self, files):
        """Handle images chosen via the drop zone (drag-drop or Browse — files or a folder)."""
        if not files:
            return
        self._selected_files = list(files)
        # Store the common ancestor (not just files[0].parent) so a later 'Load Previous
        # Results' + edit can recursively find originals across sibling condition folders.
        try:
            root = os.path.commonpath([str(Path(f).parent) for f in files])
        except ValueError:
            root = str(Path(files[0]).parent)
        config.set("last_input_dir", root)
        self._metadata = None
        self._group_data = {}
        self.groups_tree.clear()
        if self.use_groups.isChecked():
            self._autogroup_by_subfolder(announce=False)
        self._update_input_display()

    def _update_input_display(self):
        """Reflect the current selection in the drop zone + live run summary / preview."""
        if hasattr(self, "dropzone"):
            self.dropzone.set_count(len(self._selected_files))
        self._update_run_summary()
        self._update_preview()
    
    def _get_image_files(self) -> List[Path]:
        """Get list of selected image files."""
        return [Path(f) for f in self._selected_files]
    
    def _autogroup_by_subfolder(self, announce=True):
        """Derive {group: [basename]} deterministically from each file's parent subfolder.

        No LLM / no [agent] extra — groups come from the folder layout the user already has
        ("one folder per condition"). Basenames must be unique across the selection (SCAT joins
        group metadata on basename); on a collision we clear grouping and warn. Grouping only
        engages when there are >=2 distinct subfolders; a single common folder means "no groups".
        """
        files = self._get_image_files()
        if not files:
            if announce:
                QMessageBox.warning(self, "No Input", "Please select input image files first.")
            return
        from .grouping_util import duplicate_basenames
        dups = duplicate_basenames(files)
        if dups:
            self._group_data = {}
            self._metadata = None
            self.groups_tree.clear()
            self.groups_hint.setText(
                f"Duplicate filenames across folders ({', '.join(dups[:3])}…) — grouping skipped. "
                "Use unique filenames or a flat folder.")
            if announce:
                QMessageBox.warning(self, "Duplicate filenames",
                    "Some images share a filename across subfolders. SCAT joins group metadata on "
                    "the filename, so grouping was skipped. Use unique names or a flat folder.")
            return
        group_data: dict = {}
        for f in files:
            group_data.setdefault(f.parent.name or "ungrouped", []).append(f.name)
        real = {g: v for g, v in group_data.items() if g and g != "ungrouped"}
        if len(real) < 2:
            real = {}  # a single shared folder is not a comparison grouping
        self._metadata = None
        self._update_groups_list(real)
        if real:
            self.groups_hint.setText(
                f"{len(real)} group(s) from subfolders. Right-click a group to rename.")
        else:
            self.groups_hint.setText(
                "All selected files share one folder — no groups. "
                "Organize images into per-condition subfolders to compare.")

    @staticmethod
    def _grouping_from_csv(df, selected):
        """Build {group: [basename]} from a metadata DataFrame, restricted to the `selected` basenames.
        Columns: 'filename' plus a group column (prefers 'group', else the first other column). Returns
        (group_data, matched_count); raises ValueError with a user-facing message on a bad schema."""
        cols = {str(c).lower(): c for c in df.columns}
        if "filename" not in cols:
            raise ValueError("The CSV needs a 'filename' column and a group column.")
        fn_col = cols["filename"]
        group_col = cols.get("group") or next((c for c in df.columns if c != fn_col), None)
        if group_col is None:
            raise ValueError("The CSV needs a group column beside 'filename'.")
        group_data: dict = {}
        for _, row in df.iterrows():
            fn, g = str(row[fn_col]).strip(), str(row[group_col]).strip()
            if not g or g.lower() == "nan" or fn not in selected:
                continue
            group_data.setdefault(g, []).append(fn)
        return group_data, sum(len(v) for v in group_data.values())

    def _load_grouping_csv(self):
        """Load an explicit filename→group mapping from a CSV (mirrors CLI --metadata), overriding
        subfolder grouping for users whose folder layout doesn't encode the design."""
        files = self._get_image_files()
        if not files:
            QMessageBox.warning(self, "No Input", "Please select input image files first.")
            return
        # Same invariant the subfolder path enforces: SCAT joins group metadata on basename, so a
        # CSV grouping over duplicate basenames would be rejected by analyze_folder_service at run time.
        from .grouping_util import duplicate_basenames
        dups = duplicate_basenames(files)
        if dups:
            QMessageBox.warning(self, "Duplicate filenames",
                f"Some images share a filename across folders ({', '.join(dups[:3])}…). SCAT joins "
                "group metadata on the filename, so CSV grouping can't be applied. Use unique names "
                "or a flat folder.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load grouping CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        import pandas as pd
        try:
            df = pd.read_csv(path)
            group_data, matched = self._grouping_from_csv(df, {f.name for f in files})
        except ValueError as e:
            QMessageBox.warning(self, "Invalid grouping CSV", str(e))
            return
        except Exception as e:
            QMessageBox.warning(self, "Could not read CSV", str(e))
            return
        if matched == 0:
            QMessageBox.warning(self, "No matches",
                "None of the CSV filenames match the selected images (matched on filename).")
            return
        self._metadata = None
        self._update_groups_list(group_data)
        unmatched = len(files) - matched
        self.groups_hint.setText(
            f"{len(group_data)} group(s) from CSV; {matched} image(s) matched"
            + (f", {unmatched} left ungrouped." if unmatched else "."))

    def _on_groups_context_menu(self, pos):
        """Right-click a top-level group node to rename it (review/correct)."""
        item = self.groups_tree.itemAt(pos)
        if item is None or item.parent() is not None:
            return  # only group (top-level) nodes are renameable
        old = item.data(0, Qt.UserRole)
        if not old:
            return
        menu = QMenu(self)
        rename_action = menu.addAction("Rename group…")
        chosen = menu.exec(self.groups_tree.viewport().mapToGlobal(pos))
        if chosen != rename_action:
            return
        new, ok = QInputDialog.getText(self, "Rename group", "New group name:", text=str(old))
        if ok:
            self._rename_group(old, (new or "").strip())

    def _rename_group(self, old, new):
        """Rename group `old` -> `new`, merging into `new` when it already exists. No-op on an
        empty/unchanged/unknown name. Extracted from the context menu so the review/correct
        flow is unit-testable without driving the modal dialog."""
        if not new or new == old or old not in self._group_data:
            return
        data = dict(self._group_data)
        moved = data.pop(old, [])
        data[new] = data.get(new, []) + moved  # merge if the target name already exists
        self._update_groups_list(data)
        self.groups_hint.setText(f"{len(data)} group(s). Right-click a group to rename.")
    
    def _run_analysis(self):
        image_files = self._get_image_files()
        output_base = self.output_dir.path()
        
        if not image_files:
            QMessageBox.warning(self, "Error", "Please select image files")
            return
        
        if not output_base:
            # Use parent of first file's folder
            output_base = str(Path(image_files[0]).parent.parent)
        
        # Create timestamped output folder
        output_dir = get_timestamped_output_dir(Path(output_base))
        
        self._save_settings()
        
        # Store image files for worker
        self._image_files_for_analysis = [str(f) for f in image_files]
        
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress.setValue(0)
        self.eta_label.setText("")
        self._output_dir = str(output_dir)
        
        # Record start time for ETA calculation
        import time
        self._start_time = time.time()
        
        self.worker = WorkerThread(self._do_analysis, str(output_dir))
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _do_analysis(self, output_dir: str):
        """Run the canonical pipeline services (mirrors cli.analyze_command), then rebuild the
        Results-tab dict from the output dir. Executes on the WorkerThread."""
        from .pipeline import analyze_folder_service, run_statistics_service, generate_report_service

        image_files = list(self._image_files_for_analysis)
        if not image_files:
            raise ValueError("No images to analyze")

        model_type = ['threshold', 'rf', 'cnn'][self.model_type.currentIndex()]

        # Groups: invert the {group: [basename]} review panel into {basename: group}
        groups = None
        if self.use_groups.isChecked() and self._group_data:
            groups = {fn: g for g, files in self._group_data.items() for fn in files}

        res = analyze_folder_service(
            path=str(Path(image_files[0]).parent),
            image_paths=image_files,
            groups=groups,
            model_type=model_type,
            model_path=self.model_path.path() or None,
            min_area=self.min_area.value(),
            max_area=self.max_area.value(),
            circularity=self.threshold.value(),
            sensitive_mode=True,  # historical GUI default (sensitive multi-level detection)
            unet_model_path=self.detection_model_path.path() or None,
            annotate=self.annotate.isChecked(),
            visualize=self.visualize.isChecked(),
            spatial=self.spatial.isChecked(),
            parallel=config.get("performance.parallel_enabled", True),
            max_workers=config.get("performance.worker_count", 0),
            save_json=self.save_json.isChecked(),
            progress_callback=lambda c, t: self.worker.progress.emit(c, t),
            output_dir=output_dir,
        )

        stats = None
        if self.stats.isChecked() and len(res.groups) >= 2:
            stats = run_statistics_service(res.output_dir, group_col="group")

        if self.report.isChecked():
            generate_report_service(res.output_dir, statistical_results=stats, group_by="group")

        group_by = "group" if res.groups else None
        return _results_dict_from_output(res.output_dir, group_by=group_by,
                                         image_paths=image_files, stats=stats)
    
    def _on_progress(self, current, total):
        import time
        
        self.progress.setMaximum(total)
        ui_motion.animate_value(self.progress, current)   # eased, interruptible (never hops)
        self.progress_label.setText(f"{current}/{total}")
        
        # Calculate ETA
        if self._start_time and current > 0:
            elapsed = time.time() - self._start_time
            avg_per_item = elapsed / current
            remaining = total - current
            eta_seconds = avg_per_item * remaining
            
            if eta_seconds < 60:
                eta_text = f"ETA: {int(eta_seconds)}s remaining"
            elif eta_seconds < 3600:
                minutes = int(eta_seconds // 60)
                seconds = int(eta_seconds % 60)
                eta_text = f"ETA: {minutes}m {seconds}s remaining"
            else:
                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                eta_text = f"ETA: {hours}h {minutes}m remaining"
            
            self.eta_label.setText(eta_text)
    
    def _on_finished(self, results):
        self.run_btn.setEnabled(True)
        self.progress_label.setText("Complete!")
        self.eta_label.setText("")
        self.analysis_complete.emit(results)
        # The state transition IS the completion feedback (no blocking modal) — results take over.
        self._show_results(results)
    
    def _on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.progress_label.setText("Error!")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error_msg}")


class ResultsTab(QWidget):
    """Results tab for viewing analysis results."""
    
    def __init__(self):
        super().__init__()
        self.results = None
        self._setup_ui()
    
    def _setup_ui(self):
        """One calm, answer-first scroll (no sub-tabs): hero summary → per-image table →
        visualizations & statistics, in reading order — matching the app's tabless direction and
        the HTML report it produces."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("scrollContent")
        oc = QVBoxLayout(content)
        oc.setContentsMargins(16, 18, 16, 28)
        oc.setSpacing(0)
        cap = CenteredCap(1120)
        cap.content_layout.setSpacing(20)
        oc.addWidget(cap)
        oc.addStretch(1)
        self.col = cap.content_layout           # everything lands in the width-capped column
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ---- Hero card: lead with the answer (rebuilt on each load) ----
        self._hero_card = QWidget()
        self._hero_card.setObjectName("heroCard")
        self._hero_card.setStyleSheet(
            f"QWidget#heroCard {{ background-color: {Theme.BG_SURFACE}; border: 1px solid {Theme.BORDER};"
            f" border-top: 1px solid {Theme.BORDER_LIT}; border-radius: {Theme.RADIUS_CONTAINER}px; }}")
        hv = QVBoxLayout(self._hero_card)
        hv.setContentsMargins(24, 22, 24, 22)
        hv.setSpacing(16)

        top = QHBoxLayout()
        top.setSpacing(20)
        numcol = QVBoxLayout()
        numcol.setSpacing(2)
        self.hero_kicker = QLabel("MEAN ROD FRACTION")
        self.hero_kicker.setStyleSheet(
            f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_XS}px; font-weight: {Theme.WEIGHT_TITLE};"
            f" letter-spacing: {Theme.TRACK_CAPS};")
        self.hero_value = QLabel("—")
        self.hero_value.setStyleSheet(
            f"color: {Theme.TEXT_PRIMARY}; font-size: 34px; font-weight: 700;"
            f" letter-spacing: {Theme.TRACK_DISPLAY};")
        self.hero_sub = QLabel("")
        self.hero_sub.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: {Theme.FS_BODY}px;")
        # One factual trust line: a neutral count vs the fixed confidence threshold — muted, no
        # colored verdict/dot (a reliability claim the uncalibrated score can't support).
        self.trust_line = QLabel("")
        self.trust_line.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_SM}px;")
        numcol.addWidget(self.hero_kicker)
        numcol.addWidget(self.hero_value)
        numcol.addWidget(self.hero_sub)
        numcol.addWidget(self.trust_line)
        top.addLayout(numcol)
        top.addStretch(1)

        act = QVBoxLayout()
        act.setSpacing(8)
        # Button hierarchy: Open report is the payoff → coral primary once a report exists.
        self.open_report_btn = QPushButton("Open report")
        self.open_report_btn.setIcon(icon("open_in_new", "#FFFFFF"))
        self.open_report_btn.setMinimumHeight(38)
        self.open_report_btn.setStyleSheet(
            Theme.button_style(Theme.PRIMARY, "#FFFFFF", Theme.PRIMARY_LIGHT, Theme.PRIMARY_DARK))
        ui_motion.attach_button_motion(self.open_report_btn, primary=True)
        self.open_report_btn.clicked.connect(self._open_report)
        # Generate/rebuild is a utility → quiet secondary.
        self.generate_report_btn = QPushButton("Rebuild after edits")
        self.generate_report_btn.setIcon(icon("refresh"))
        self.generate_report_btn.setToolTip(
            "Regenerate annotated images, statistics, visualizations and the HTML report.\n"
            "Use after editing/correcting results in the labeling window.")
        self.generate_report_btn.clicked.connect(self._generate_report)
        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.setIcon(icon("folder_open"))
        self.open_folder_btn.clicked.connect(self._open_folder)
        for b in (self.open_report_btn, self.generate_report_btn, self.open_folder_btn):
            b.setCursor(Qt.PointingHandCursor)
            act.addWidget(b)
        top.addLayout(act)
        hv.addLayout(top)

        # Composition strip (one thin semantic line — replaces the six KPI tiles).
        self.composition_line = QLabel("")
        self.composition_line.setStyleSheet(
            f"color: {Theme.TEXT_SECONDARY}; font-size: {Theme.FS_BODY}px; padding-top: 4px;")
        self.composition_line.setTextFormat(Qt.RichText)
        hv.addWidget(self.composition_line)

        self.col.addWidget(self._hero_card)

        # Quiet secondary: load a previous session (belongs to the "open" family, de-emphasized).
        self.load_results_btn = QPushButton("Load a previous results folder…")
        self.load_results_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {Theme.TEXT_MUTED};"
            f" font-size: {Theme.FS_SM}px; text-align: left; padding: 2px 0; }}"
            f"QPushButton:hover {{ color: {Theme.TEXT_SECONDARY}; }}")
        self.load_results_btn.setCursor(Qt.PointingHandCursor)
        self.load_results_btn.clicked.connect(self._load_previous_results)
        # Added to the column as a quiet footer below the results (see end of _setup_ui) — a
        # session-switch action shouldn't interrupt the hero→table reading flow.

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.col.addWidget(self.progress)

        # ---- Per-image table ----
        self.col.addWidget(self._section_label("PER-IMAGE RESULTS"))
        hint = QLabel("Double-click a row to view and edit an image's deposits.")
        hint.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_SM}px;")
        self.col.addWidget(hint)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(7)
        self.summary_table.setHorizontalHeaderLabels(
            ["Filename", "Review", "Normal", "ROD", "Artifact", "ROD %", "Total IOD"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.setSortingEnabled(True)
        self.summary_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.summary_table.doubleClicked.connect(self._on_table_double_click)
        self.col.addWidget(self.summary_table)

        # ---- Visualizations + statistics (added by _load_statistics_tab) ----
        self.stats_host = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_host)
        self.stats_layout.setContentsMargins(0, 0, 0, 0)
        self.stats_layout.setSpacing(20)
        self.col.addWidget(self.stats_host)

        # Session switch — a quiet footer below the current results, out of the reading flow.
        self.col.addWidget(self.load_results_btn)

        # Empty until results arrive.
        for b in (self.open_report_btn, self.generate_report_btn, self.open_folder_btn):
            b.setVisible(False)

    # ---- small building blocks ----
    def _section_label(self, text: str) -> QLabel:
        """A small muted uppercase section label (the app's calm-card heading idiom)."""
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_XS}px; font-weight: {Theme.WEIGHT_TITLE};"
            f" letter-spacing: {Theme.TRACK_CAPS}; margin-top: 4px;")
        return lbl

    def _apply_action_state(self, report_exists: bool):
        """Swap the primary action by state: Open report leads once a report exists; otherwise
        Generate becomes the coral primary (the way to create one)."""
        self.open_report_btn.setVisible(report_exists)
        if report_exists:
            self.generate_report_btn.setText("Rebuild after edits")
            self.generate_report_btn.setStyleSheet("")   # inherit the app's secondary QPushButton QSS
        else:
            self.generate_report_btn.setText("Generate report")
            self.generate_report_btn.setStyleSheet(
                Theme.button_style(Theme.PRIMARY, "#FFFFFF", Theme.PRIMARY_LIGHT, Theme.PRIMARY_DARK))

    def _set_num(self, row: int, col: int, value, fmt: str, color: str = None):
        """A right-aligned, sort-correct numeric cell (magnitudes line up like a ledger)."""
        item = NumericTableWidgetItem(value, fmt)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if color:
            item.setForeground(QColor(color))
        self.summary_table.setItem(row, col, item)

    def _fit_table_height(self):
        """Size the table to its rows so the page is one scroll — capped so a huge run still
        scrolls internally rather than making the page absurdly tall."""
        t = self.summary_table
        if t.rowCount() == 0:
            h = t.horizontalHeader().height() + 48
        else:
            h = (t.horizontalHeader().height() + 2 * t.frameWidth() + 4
                 + sum(t.rowHeight(r) for r in range(t.rowCount())))
        h = min(h, 560)
        t.setMinimumHeight(h)
        t.setMaximumHeight(h)
    
    def load_results(self, results: dict):
        self.results = results
        film_summary = results['film_summary']

        total_normal = film_summary['n_normal'].sum()
        total_rod = film_summary['n_rod'].sum()
        total_artifact = film_summary['n_artifact'].sum()
        mean_rod_frac = film_summary['rod_fraction'].mean()
        std_rod_frac = film_summary['rod_fraction'].std()
        n = len(film_summary)

        # Hero: lead with the headline answer, driven by the predeclared primary metric.
        pm = _metrics.resolve_metric(results.get("primary_metric"))
        norm = results.get("normalization", _metrics.DEFAULT_NORMALIZATION)
        meta = results.get("run_meta", {})
        m = _metrics.METRICS[pm]
        self.hero_kicker.setText(m.label.upper())
        self.hero_value.setText(_metrics.format_headline(film_summary, pm, norm, meta))
        vals = _metrics.metric_values(film_summary, pm).dropna()
        sd = vals.std() if len(vals) > 1 else 0.0
        self.hero_sub.setText(f"±{sd:.1f}{m.unit}   ·   across {n} image{'s' if n != 1 else ''}")
        # Grouped runs: the hero value is a grand mean across distinct conditions — flag it as pooled
        # and (for non-circular metrics) give the honest per-group range so the pooled number isn't
        # read as a single condition's effect.
        gcol = results.get("group_by")
        groups = []
        if gcol and gcol in film_summary.columns:
            for g in film_summary[gcol].dropna().unique():
                s = str(g).strip()
                if s and s != "ungrouped" and s not in groups:
                    groups.append(s)
        if len(groups) >= 2:
            note = f"pooled across {len(groups)} groups"
            if pm != "mean_hue":     # hue is circular — a min/max range is misleading
                per_group = [_metrics.metric_values(
                    film_summary[film_summary[gcol].astype(str).str.strip() == g], pm).dropna().mean()
                    for g in groups]
                per_group = [v for v in per_group if v == v]
                if per_group:
                    note += (f" (group image-means {m.fmt.format(min(per_group))}"
                             f"–{m.fmt.format(max(per_group))}{m.unit})")
            self.hero_sub.setText(self.hero_sub.text() + "  ·  " + note)
        # Factual trust line: count of deposits below the fixed confidence threshold (no verdict).
        self.trust_line.setText(
            _conf.run_trust(results.get("deposit_data"),
                            results.get("confidence_threshold", 0.60))["line"])

        # Composition strip (one thin semantic line — replaces the six KPI tiles).
        # "Deposits" = Normal + ROD (artifacts are the reject class) — matches the report's
        # total_deposits so the app and the report it produces agree on the headline count.
        sep = " &nbsp;·&nbsp; "
        parts = [
            f"Deposits <b>{total_normal + total_rod:.0f}</b>",
            f"Normal <b style='color:{Theme.NORMAL}'>{total_normal:.0f}</b>",
            f"ROD <b style='color:{Theme.ROD}'>{total_rod:.0f}</b>",
            f"ROD fraction <b>{mean_rod_frac*100:.1f}%</b>",
            f"Artifact <span style='color:{Theme.TEXT_MUTED}'>{total_artifact:.0f}</span>",
        ]
        if 'total_iod' in film_summary.columns:              # omit when absent — never show a fake "0"
            parts.append(f"Total IOD <b>{film_summary['total_iod'].sum():.0f}</b>")
        self.composition_line.setText(sep.join(parts))

        # Actions: state-driven hierarchy (Open report leads once one exists).
        output_dir = results.get('output_dir', '')
        report_exists = bool(output_dir) and (Path(output_dir) / 'report.html').exists()
        self.generate_report_btn.setVisible(True)
        self.open_folder_btn.setVisible(bool(output_dir))
        self._apply_action_state(report_exists)

        # Per-image low-confidence counts (triage signal, not a reliability claim) — Review column.
        flagged = _metrics.flagged_by_image(
            results.get("deposit_data"), results.get("confidence_threshold", 0.60))

        # Per-image table — right-aligned tabular numbers, gentle semantic tint on the class counts.
        self.summary_table.setSortingEnabled(False)
        self.summary_table.setRowCount(n)
        for i, (_, row) in enumerate(film_summary.iterrows()):
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(row['filename'])))
            info = flagged.get(str(row['filename']))
            rev = NumericTableWidgetItem(info["flagged"] if info else 0, "{:.0f}")
            rev.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if info and info["flagged"]:
                rev.setForeground(QColor(Theme.PRIMARY_LIGHT))
                frac = 100.0 * info["flagged"] / info["total"] if info["total"] else 0.0
                rev.setToolTip(
                    f"{info['flagged']} of {info['total']} deposits below the "
                    f"confidence-score threshold ({frac:.0f}%)")
            else:
                rev.setText("—")
            self.summary_table.setItem(i, 1, rev)
            self._set_num(i, 2, row['n_normal'], "{:.0f}", Theme.NORMAL)
            self._set_num(i, 3, row['n_rod'], "{:.0f}", Theme.ROD)
            self._set_num(i, 4, row['n_artifact'], "{:.0f}", Theme.TEXT_MUTED)
            self._set_num(i, 5, row['rod_fraction'] * 100, "{:.1f}%")
            self._set_num(i, 6, row.get('total_iod', 0), "{:.0f}")
        self.summary_table.setSortingEnabled(True)
        self._fit_table_height()

        self._load_statistics_tab(results)
    
    def _load_statistics_tab(self, results: dict):
        """The working view is for triage; the report carries the full distributions, group comparisons
        and statistics. Show ONE quiet, report-state-aware pointer instead of duplicating (and out-dumping)
        the report. stats_layout is retained so re-loads (edit -> _reload_results -> load_results) work."""
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        out = results.get("output_dir", "")
        report_exists = bool(out) and (Path(out) / "report.html").exists()
        text = ("Full distributions, group comparisons and statistics are in the report."
                if report_exists else
                "Generate a report to see the full distributions, group comparisons and statistics.")
        pointer = QLabel(text)
        pointer.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: {Theme.FS_SM}px; padding-top: 8px;")
        pointer.setWordWrap(True)
        self.stats_layout.addWidget(pointer)
    
    def _on_table_double_click(self, index):
        if not self.results:
            return
        
        row = index.row()
        filename = self.summary_table.item(row, 0).text()
        output_dir = Path(self.results['output_dir'])
        
        # Find original image: in-session exact paths → recursive fallback → annotated image
        stem = Path(filename).stem
        found = self._find_original_image(filename)
        image_path = str(found) if found else None
        if not image_path:
            annotated_path = output_dir / 'annotated' / f"{stem}_annotated.png"
            if annotated_path.exists():
                image_path = str(annotated_path)
        if not image_path:
            QMessageBox.warning(self, "Not Found", f"Original image not found for {filename}")
            return
        
        # Load deposits from JSON (includes artifacts and contours)
        contour_data = {}
        file_deposits = None
        next_group_id = 1
        
        json_path = output_dir / 'deposits' / f"{stem}.labels.json"
        if not json_path.exists():
            json_path = output_dir / 'deposits' / f"{stem}_deposits.json"
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    next_group_id = data.get('next_group_id', 1)
                    
                    # Build DataFrame from JSON (includes artifacts)
                    deposits_list = []
                    for dep in data.get('deposits', []):
                        dep_id = dep.get('id', len(deposits_list) + 1)
                        deposits_list.append({
                            'id': dep_id,
                            'filename': filename,
                            'label': dep.get('label', 'unknown'),
                            'centroid_x': dep.get('centroid', [0, 0])[0],
                            'centroid_y': dep.get('centroid', [0, 0])[1],
                            'area': dep.get('area', 0),
                            'circularity': dep.get('circularity', 0),
                            'iod': dep.get('iod', 0),
                            'mean_hue': dep.get('mean_hue', 0),
                            'mean_saturation': dep.get('mean_saturation', 0),
                            'mean_value': dep.get('mean_value', 0),
                            'bbox': dep.get('bbox', [0, 0, 0, 0])
                        })
                        contour_data[dep_id] = {
                            'contour': dep.get('contour', []),
                            'merged': dep.get('merged', False),
                            'group_id': dep.get('group_id', None)
                        }
                    
                    if deposits_list:
                        file_deposits = pd.DataFrame(deposits_list)
            except Exception as e:
                print(f"Error loading JSON: {e}")
        
        # Fallback to CSV if JSON failed
        if file_deposits is None and 'deposit_data' in self.results and self.results['deposit_data'] is not None:
            deposits_df = self.results['deposit_data']
            file_deposits = deposits_df[deposits_df['filename'] == filename].copy()
        
        if file_deposits is None or len(file_deposits) == 0:
            # Still allow editing even with no deposits
            file_deposits = pd.DataFrame(columns=['id', 'filename', 'label', 'centroid_x', 'centroid_y', 
                                                   'area', 'circularity', 'iod'])
        
        # Open LabelingWindow in EDIT_MODE
        from .labeling_gui import LabelingWindow
        
        edit_data = {
            'image_path': image_path,
            'output_dir': str(output_dir),
            'filename': filename,
            'deposits_df': file_deposits,
            'contour_data': contour_data,
            'next_group_id': next_group_id
        }
        
        self._edit_window = LabelingWindow(
            mode=LabelingWindow.MODE_EDIT,
            edit_data=edit_data
        )
        self._edit_window.data_saved.connect(self._reload_results)
        self._edit_window.show()
    
    def _reload_results(self):
        """Reload results after editing."""
        if not self.results or 'output_dir' not in self.results:
            return
        
        output_dir = Path(self.results['output_dir'])
        
        # Reload film_summary
        summary_path = output_dir / IMAGE_SUMMARY
        if summary_path.exists():
            self.results['film_summary'] = pd.read_csv(summary_path)
        
        # Reload deposit_data
        all_deposits_path = output_dir / ALL_DEPOSITS
        if all_deposits_path.exists():
            self.results['deposit_data'] = pd.read_csv(all_deposits_path)

        # Refresh display (regenerate the report on demand via the always-on button)
        self.load_results(self.results)
    
    def _open_folder(self):
        if self.results and 'output_dir' in self.results:
            path = self.results['output_dir']
            # Cross-platform folder opening
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', path])
            else:  # Linux
                subprocess.run(['xdg-open', path])
    
    def _find_original_image(self, filename):
        """Locate an original image. Prefers an EXACT basename match among the paths captured
        at analysis time (handles per-condition subfolders), then a stem match, then a recursive
        search under the common input root (for disk-loaded results). Returns a Path or None."""
        stem = Path(filename).stem
        exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        session = [Path(p) for p in ((self.results or {}).get('image_paths') or [])]
        for pp in session:                        # 1a. exact basename, in-session
            if pp.name == filename and pp.exists():
                return pp
        for pp in session:                        # 1b. stem match, in-session
            if pp.stem == stem and pp.exists():
                return pp
        input_dir = config.get("last_input_dir", "")
        if input_dir and Path(input_dir).exists():
            base = Path(input_dir)
            for m in base.rglob(filename):        # 2a. exact filename under the root
                if m.is_file():
                    return m
            for m in base.rglob(f"{stem}.*"):     # 2b. stem.* image fallback
                if m.suffix.lower() in exts:
                    return m
        return None

    def _generate_report(self):
        """Regenerate annotated images, statistics and report after editing."""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No analysis results to regenerate.")
            return
        
        output_dir = Path(self.results['output_dir'])

        if not output_dir.exists():
            QMessageBox.critical(self, "Error", f"Output directory not found: {output_dir}")
            return
        
        # Show progress bar
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        QApplication.processEvents()
        
        try:
            # Reload data from CSV files (support both new and old naming)
            summary_path = output_dir / IMAGE_SUMMARY
            if not summary_path.exists():
                summary_path = output_dir / 'film_summary.csv'  # Backward compatibility
            all_deposits_path = output_dir / ALL_DEPOSITS
            
            if not summary_path.exists() or not all_deposits_path.exists():
                self.progress.setVisible(False)
                QMessageBox.critical(self, "Error", "Required CSV files not found.")
                return
            
            image_summary = pd.read_csv(summary_path)
            deposit_data = pd.read_csv(all_deposits_path)
            
            self.progress.setValue(10)
            QApplication.processEvents()
            
            # 1. Regenerate annotated images from the (possibly edited) labels JSON,
            #    reusing analyzer.generate_annotated_image instead of an inline cv2 annotator.
            from .analyzer import Analyzer, deposits_from_labels_json
            annotated_dir = output_dir / 'annotated'
            annotated_dir.mkdir(exist_ok=True)
            deposits_dir = output_dir / 'deposits'

            for idx, row in image_summary.iterrows():
                stem = Path(row['filename']).stem
                json_path = deposits_dir / f"{stem}.labels.json"
                image_path = self._find_original_image(row['filename'])
                # save_json=False runs write no labels JSON -> nothing to redraw; skip cleanly.
                if image_path and json_path.exists():
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        deposits = deposits_from_labels_json(json_path)
                        annotated = Analyzer.generate_annotated_image(
                            image, deposits, show_labels=True, skip_artifacts=True)
                        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(annotated_dir / f"{stem}_annotated.png"), annotated_bgr)
                self.progress.setValue(10 + int(40 * (idx + 1) / max(len(image_summary), 1)))
                QApplication.processEvents()

            # 2. Visualizations + statistics + HTML report via the canonical services.
            from .pipeline import run_statistics_service, generate_report_service
            from .visualization import generate_all_visualizations

            group_by = self.results.get('group_by')
            viz_dir = output_dir / 'visualizations'
            viz_results = generate_all_visualizations(
                image_summary, deposit_data, viz_dir, group_by=group_by)
            self.progress.setValue(70)
            QApplication.processEvents()

            stats = run_statistics_service(str(output_dir), group_col=group_by or 'group')
            self.progress.setValue(80)
            QApplication.processEvents()

            generate_report_service(str(output_dir), statistical_results=stats, group_by=group_by)
            self.progress.setValue(100)

            # Update results and refresh display (Results tab wants the flat metrics mapping)
            self.results['film_summary'] = image_summary  # Keep key for compatibility
            self.results['comprehensive_stats'] = stats
            self.results['deposit_data'] = deposit_data
            self.results['viz_results'] = viz_results
            self.results['stats_results'] = (
                {} if not stats or stats.get('skipped')
                else stats.get('basic', {}).get('metrics', {}) or {})

            self.progress.setVisible(False)
            # The refresh IS the feedback — load_results re-reveals the now-current "Open report"
            # primary; no blocking success modal.
            self.load_results(self.results)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Regeneration failed: {str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.progress.setVisible(False)
            self.progress.setValue(0)
    
    def _open_report(self):
        if self.results and 'output_dir' in self.results:
            report_path = Path(self.results['output_dir']) / 'report.html'
            if report_path.exists():
                import webbrowser
                webbrowser.open(str(report_path))
    
    def _load_previous_results(self):
        """Load results from a previous analysis session."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Results Folder",
            config.get("last_output_dir", "")
        )
        
        if not folder:
            return
        
        output_dir = Path(folder)
        
        # Check for required files (support both new and old naming)
        summary_path = output_dir / IMAGE_SUMMARY
        if not summary_path.exists():
            summary_path = output_dir / 'film_summary.csv'  # Backward compatibility
        deposits_path = output_dir / ALL_DEPOSITS
        
        if not summary_path.exists():
            QMessageBox.critical(self, "Error", "image_summary.csv not found in selected folder.")
            return
        
        try:
            # Rebuild via the shared helper (also picks up spatial_stats.json + auto group_by).
            self.results = _results_dict_from_output(output_dir)
            self.load_results(self.results)
            QMessageBox.information(self, "Success", f"Loaded results from:\n{output_dir}")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to load results: {str(e)}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCAT - Spot Classification and Analysis Tool")
        self.setMinimumSize(900, 700)
        
        # Set icon
        icon_path = get_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        
        self._setup_ui()
        self._setup_shortcuts()
        self._load_window_state()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Slim top bar: wordmark + icon actions. A single stateful workspace needs no tab
        #     chrome; rare setup lives behind a "More" menu (Apple: reduce navigation). ---
        topbar = QWidget()
        topbar.setObjectName("topBar")
        topbar.setStyleSheet(
            f"QWidget#topBar {{ background-color: {Theme.BG_BASE}; border-bottom: 1px solid {Theme.BORDER}; }}")
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(18, 8, 12, 8)
        tb.setSpacing(10)

        wordmark = QLabel(
            f'<span style="color:{Theme.PRIMARY};">S</span>'
            f'<span style="color:{Theme.TEXT_PRIMARY};">CAT</span>')
        wordmark.setFont(QFont("Noto Sans", 16, QFont.Bold))
        tb.addWidget(wordmark)
        tagline = QLabel("Drosophila excreta analysis")
        tagline.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px; padding-top: 5px;")
        tb.addWidget(tagline)
        tb.addStretch(1)

        self.assistant_btn = QPushButton("  Assistant")
        self.assistant_btn.setIcon(icon("forum"))
        self.assistant_btn.setCheckable(True)
        self.assistant_btn.setToolTip("Show/hide the conversational assistant panel")
        self.assistant_btn.clicked.connect(self._toggle_chat_dock)
        tb.addWidget(self.assistant_btn)

        more_btn = QPushButton("  More")
        more_btn.setIcon(icon("tune"))
        more_menu = QMenu(self)
        more_menu.addAction(icon("photo_library", Theme.TEXT_PRIMARY, 18), "Labeling…", self._launch_labeling)
        more_menu.addAction(icon("insights", Theme.TEXT_PRIMARY, 18), "Train model…", self._open_training)
        more_menu.addSeparator()
        more_menu.addAction(icon("settings", Theme.TEXT_PRIMARY, 18), "Settings…", self._open_settings)
        more_btn.setMenu(more_menu)
        tb.addWidget(more_btn)

        layout.addWidget(topbar)

        # --- Body: the Analyze workspace, directly (no tab bar) ---
        self.analysis_tab = AnalysisTab()
        layout.addWidget(self.analysis_tab, 1)

        # Conversational assistant dock (lazy — the agent stack is optional)
        self._build_chat_dock()

        self.statusBar().showMessage("Ready")

        # Post-build polish QSS can't express: pointer cursors on controls (see scat.ui_motion).
        ui_motion.apply_ui_polish(self)

    def _open_training(self):
        """Open the model-training UI in its own window (rare / one-time action)."""
        if getattr(self, "_training_win", None) is None:
            self._training_win = QDialog(self)
            self._training_win.setWindowTitle("SCAT — Train model")
            self._training_win.resize(760, 760)
            self._training_win.setStyleSheet(Theme.get_app_stylesheet())
            tv = QVBoxLayout(self._training_win)
            tv.setContentsMargins(0, 0, 0, 0)
            self.training_tab = TrainingTab()
            tv.addWidget(self.training_tab)
            ui_motion.apply_ui_polish(self._training_win)
        self._training_win.show()
        self._training_win.raise_()

    def _build_chat_dock(self):
        """Add the Assistant dock. Imported lazily so `import scat.main_gui` and the core GUI
        keep working without the [agent] extra; the widget itself only needs PySide6."""
        self.chat_widget = None
        self.chat_dock = QDockWidget("Assistant", self)
        self.chat_dock.setObjectName("assistantDock")
        try:
            from .agent.chat_widget import ChatDockWidget
            self.chat_widget = ChatDockWidget()
            self.chat_dock.setWidget(self.chat_widget)
        except Exception as exc:  # PySide-only import should not fail, but degrade gracefully
            placeholder = QLabel(f"Assistant unavailable: {exc}")
            placeholder.setWordWrap(True)
            placeholder.setStyleSheet("color: gray; padding: 12px;")
            self.chat_dock.setWidget(placeholder)
        self.addDockWidget(Qt.RightDockWidgetArea, self.chat_dock)
        self.chat_dock.setMinimumWidth(320)
        # Open with a roomy panel (the sizeHint prefers ~440; this pins the initial split).
        self.resizeDocks([self.chat_dock], [460], Qt.Horizontal)
        visible = config.get("window.chat_visible", True)
        self.chat_dock.setVisible(visible)
        self.assistant_btn.setChecked(visible)
        self.chat_dock.visibilityChanged.connect(self.assistant_btn.setChecked)

    def _toggle_chat_dock(self, checked):
        # Fade the dock content in/out (drawer curve) rather than popping it.
        content = self.chat_dock.widget()
        if checked:
            self.chat_dock.setVisible(True)
            if content is not None:
                ui_motion.fade_in(content, dur=ui_motion.DUR_DOCK, curve=ui_motion.CURVE_DRAWER)
        elif content is not None:
            ui_motion.fade_out(content, on_finished=lambda: self.chat_dock.setVisible(False))
        else:
            self.chat_dock.setVisible(False)

    def _setup_shortcuts(self):
        # Global shortcuts
        QShortcut(QKeySequence(config.get_shortcut("quit")), self, self.close)
        QShortcut(QKeySequence(config.get_shortcut("run_analysis")), self, self._run_analysis_shortcut)

    def _run_analysis_shortcut(self):
        self.analysis_tab._run_analysis()
    
    def _open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Shortcuts changed - would need restart to take effect
            QMessageBox.information(
                self, "Settings Saved",
                "Settings saved. Some shortcut changes may require restart."
            )
    
    def _launch_labeling(self):
        from .labeling_gui import LabelingWindow
        self.labeling_window = LabelingWindow()
        icon_path = get_icon_path()
        if icon_path:
            self.labeling_window.setWindowIcon(QIcon(icon_path))
        self.labeling_window.show()
    
    def _load_window_state(self):
        w = config.get("window.width", 1200)
        h = config.get("window.height", 800)
        self.resize(w, h)
        if config.get("window.maximized", False):
            self.showMaximized()
    
    def _save_window_state(self):
        config.set("window.width", self.width(), auto_save=False)
        config.set("window.height", self.height(), auto_save=False)
        config.set("window.maximized", self.isMaximized())
        if getattr(self, "chat_dock", None) is not None:
            config.set("window.chat_visible", self.chat_dock.isVisible())

    def closeEvent(self, event):
        # Tear down the agent runner (subscription backend owns a daemon asyncio loop)
        if getattr(self, "chat_widget", None) is not None:
            self.chat_widget.shutdown()
        self._save_window_state()
        event.accept()


def run_gui():
    """Launch the main GUI application."""
    # AppUserModelID is already set at module import time (top of file)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Load custom fonts (Noto Sans)
    load_custom_fonts()
    
    # Set application-wide font
    app_font = QFont("Noto Sans", 10)
    if not QFontDatabase.hasFamily("Noto Sans"):
        # Fallback if Noto Sans not available
        app_font = QFont("Segoe UI", 10)  # Windows fallback
    app.setFont(app_font)
    
    # Apply dark theme
    app.setStyleSheet(Theme.get_app_stylesheet())
    
    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
