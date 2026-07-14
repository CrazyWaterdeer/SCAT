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
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QLineEdit, QMessageBox, QScrollArea,
    QDialog, QKeySequenceEdit, QDialogButtonBox,
    QGraphicsView, QGraphicsScene, QListWidget, QMenu, QInputDialog,
    QTreeWidget, QTreeWidgetItem, QFrame, QDockWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QRectF
from PySide6.QtGui import (
    QFont, QPixmap, QIcon, QKeySequence,
    QColor, QShortcut, QFontDatabase, QPainter
)

# Import SCAT modules
from .detector import DepositDetector
from .classifier import ClassifierConfig
from .analyzer import Analyzer, ReportGenerator
from .config import config, get_timestamped_output_dir
from .ui_common import (
    Theme, NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox,
    load_custom_fonts, get_icon_path
)

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
        
        # Get CPU thread count for dynamic worker options
        import os
        cpu_count = os.cpu_count() or 1
        
        # Calculate auto worker count for display
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            memory_workers = max(1, int(available_gb / 0.3))
        except ImportError:
            memory_workers = 4
        cpu_workers = max(1, cpu_count // 2)
        self.auto_worker_count = min(cpu_workers, memory_workers, 20)  # Max 20 for auto
        
        # Available worker counts (up to 32, but limited by CPU threads)
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
        
        parallel_layout.addRow("Worker threads:", self.workers_combo)
        
        # System info
        try:
            import psutil
            mem_gb = psutil.virtual_memory().available / (1024**3)
            sys_info = f"Detected: {cpu_count} CPU threads, {mem_gb:.1f} GB available RAM"
        except ImportError:
            sys_info = f"Detected: {cpu_count} CPU threads"
        
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


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView with Ctrl+wheel zoom support."""
    
    def __init__(self, scene=None):
        super().__init__(scene) if scene else super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
    
    def wheelEvent(self, event):
        """Handle Ctrl+wheel for zoom."""
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


class ImageViewerDialog(QDialog):
    """Dialog for viewing images in full size with fit to window."""
    
    def __init__(self, image_path: str, title: str = "Image Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 700)
        self.image_path = image_path
        
        layout = QVBoxLayout(self)
        
        # Use ZoomableGraphicsView for scaling with Ctrl+wheel
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))
        
        layout.addWidget(self.view)
        
        # Hint label
        hint_label = QLabel("Tip: Ctrl + Mouse wheel to zoom")
        hint_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 11px;")
        hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        fit_btn = QPushButton("Fit to Window")
        fit_btn.clicked.connect(self._fit_to_window)
        btn_layout.addWidget(fit_btn)
        
        actual_btn = QPushButton("Actual Size (100%)")
        actual_btn.clicked.connect(self._actual_size)
        btn_layout.addWidget(actual_btn)
        
        close_btn = QPushButton("Close (Esc)")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Escape to close
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)
        
        # Fit to window on open
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def _fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def _actual_size(self):
        self.view.resetTransform()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Auto-fit on resize
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def showEvent(self, event):
        super().showEvent(event)
        # Fit after dialog is shown
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically."""
    
    def __init__(self, value, display_format: str = None):
        if display_format:
            super().__init__(display_format.format(value))
        else:
            super().__init__(str(value))
        self._value = value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._value < other._value
        return super().__lt__(other)


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
        
        # Scroll content widget
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
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
        
        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self._train)
        layout.addWidget(self.train_btn)
        
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
    summary_path = out / "image_summary.csv"
    if not summary_path.exists():
        summary_path = out / "film_summary.csv"  # backward compatibility
    film_summary = pd.read_csv(summary_path)
    dep_path = out / "all_deposits.csv"
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

    return {
        "output_dir": str(out),
        "film_summary": film_summary,   # holds image_summary; key kept for compatibility
        "deposit_data": deposit_data,
        "viz_results": viz_results,
        "spatial_stats": spatial_stats,
        "stats_results": stats_results,
        "group_by": group_by,
        "image_paths": list(image_paths) if image_paths else [],
    }


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
        
        # Container widget for scroll area content
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        # Input/Output
        io_group = QGroupBox("Input / Output")
        io_layout = QVBoxLayout()
        io_layout.setSpacing(6)
        io_layout.setContentsMargins(10, 12, 10, 10)
        
        # Default paths
        default_input = str(Path.home() / "SCAT" / "data" / "images")
        default_output = str(Path.home() / "SCAT" / "data" / "results")
        
        # Input: File selector (select one or multiple files)
        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        
        input_label = QLabel("Input")
        input_label.setMinimumWidth(70)
        input_label.setStyleSheet(f"font-weight: bold; color: {Theme.TEXT_PRIMARY}; background-color: transparent;")
        
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select image files...")
        self.input_path_edit.setReadOnly(True)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setMinimumWidth(80)
        self.browse_btn.setToolTip("Select one or more image files (Ctrl+A to select all in folder)")
        self.browse_btn.clicked.connect(self._browse_input)
        
        input_row.addWidget(input_label)
        input_row.addWidget(self.input_path_edit, 1)
        input_row.addWidget(self.browse_btn)
        
        io_layout.addLayout(input_row)
        
        self.output_dir = PathSelector("Output", is_folder=True, config_key="last_output_dir", default_path=default_output)
        self.model_path = PathSelector("Classifier", filter="Model (*.pkl *.pt)", config_key="last_model_path")
        self.detection_model_path = PathSelector("Detection (U-Net)", filter="Model (*.pt)", config_key="last_detection_model_path")
        self.detection_model_path.setToolTip("Optional: U-Net model for improved deposit detection")
        
        # Groups section
        groups_group = QGroupBox("Groups")
        groups_layout = QVBoxLayout()
        groups_layout.setSpacing(6)
        groups_layout.setContentsMargins(10, 12, 10, 10)
        
        # Use groups checkbox
        self.use_groups = QCheckBox("Use groups for comparison")
        self.use_groups.setChecked(config.get("analysis.use_groups", True))
        self.use_groups.toggled.connect(self._on_use_groups_toggled)
        groups_layout.addWidget(self.use_groups)
        
        # Deterministic grouping: derive {file: group} from each file's parent subfolder
        # (SCAT's "one folder per condition" convention). Replaces the drag-drop editor.
        self.autogroup_btn = QPushButton("Group by subfolder")
        self.autogroup_btn.setToolTip(
            "Assign each selected image to a group named after its parent subfolder.\n"
            "Double-click a group in the list below to rename it.")
        # wrap: QPushButton.clicked passes a `checked` bool that would shadow `announce`
        self.autogroup_btn.clicked.connect(lambda: self._autogroup_by_subfolder(announce=True))
        groups_layout.addWidget(self.autogroup_btn)

        self.groups_hint = QLabel("Select images, then group by subfolder. Double-click a group to rename.")
        self.groups_hint.setWordWrap(True)
        self.groups_hint.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 11px;")
        groups_layout.addWidget(self.groups_hint)

        # Groups tree (expandable - click to show files; group nodes are editable to rename)
        self.groups_tree = QTreeWidget()
        self.groups_tree.setHeaderHidden(True)
        self.groups_tree.setMinimumHeight(150)
        self.groups_tree.setMaximumHeight(200)
        self.groups_tree.setIndentation(15)
        self.groups_tree.setAnimated(True)
        # Single click to expand/collapse
        self.groups_tree.itemClicked.connect(self._on_group_tree_clicked)
        # Right-click a group to rename it (the review/correct affordance)
        self.groups_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.groups_tree.customContextMenuRequested.connect(self._on_groups_context_menu)
        groups_layout.addWidget(self.groups_tree)
        
        groups_group.setLayout(groups_layout)
        
        # Note: input row already added above
        io_layout.addWidget(self.output_dir)
        io_layout.addWidget(self.model_path)
        io_layout.addWidget(self.detection_model_path)
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
        # Add groups section after I/O
        layout.addWidget(groups_group)
        
        # Update groups UI state
        self._on_use_groups_toggled(self.use_groups.isChecked())
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QFormLayout()
        options_layout.setVerticalSpacing(8)
        options_layout.setHorizontalSpacing(12)
        options_layout.setContentsMargins(10, 12, 10, 10)
        
        self.model_type = NoScrollComboBox()
        self.model_type.addItems(["Threshold", "Random Forest", "CNN"])
        model_type_map = {"threshold": 0, "rf": 1, "cnn": 2}
        self.model_type.setCurrentIndex(model_type_map.get(config.get("analysis.model_type", "rf"), 1))
        classifier_label = QLabel("Classifier")
        classifier_label.setStyleSheet("background-color: transparent;")
        options_layout.addRow(classifier_label, self.model_type)
        
        # Report / visualization options (always available; the report is a follow-up action)
        self.annotate = QCheckBox("Generate annotated images")
        self.annotate.setChecked(config.get("analysis.annotate", True))
        options_layout.addRow(self.annotate)
        
        self.visualize = QCheckBox("Generate visualizations")
        self.visualize.setChecked(config.get("analysis.visualize", True))
        options_layout.addRow(self.visualize)
        
        self.spatial = QCheckBox("Spatial analysis")
        self.spatial.setChecked(config.get("analysis.spatial", True))
        options_layout.addRow(self.spatial)
        
        self.stats = QCheckBox("Statistical analysis")
        self.stats.setChecked(config.get("analysis.stats", True))
        options_layout.addRow(self.stats)
        
        self.report = QCheckBox("Generate HTML report")
        self.report.setChecked(config.get("analysis.report", True))
        options_layout.addRow(self.report)
        
        self.save_json = QCheckBox("Save for retraining (JSON)")
        self.save_json.setChecked(config.get("analysis.save_json", True))
        self.save_json.setToolTip("Save contour data for model retraining. Disable to reduce file size.")
        options_layout.addRow(self.save_json)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Detection settings
        detect_group = QGroupBox("Detection Settings")
        detect_layout = QFormLayout()
        detect_layout.setVerticalSpacing(8)
        detect_layout.setHorizontalSpacing(12)
        detect_layout.setContentsMargins(10, 12, 10, 10)
        
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
        
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)
        
        # Run button - PRIMARY color for emphasis
        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.setMinimumHeight(48)
        self.run_btn.setStyleSheet(Theme.button_style(Theme.PRIMARY, "#FFFFFF", Theme.PRIMARY_LIGHT))
        self.run_btn.clicked.connect(self._run_analysis)
        layout.addWidget(self.run_btn)
        
        # Progress
        progress_layout = QVBoxLayout()
        
        progress_bar_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress_label = QLabel("")
        progress_bar_layout.addWidget(self.progress)
        progress_bar_layout.addWidget(self.progress_label)
        progress_layout.addLayout(progress_bar_layout)
        
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        progress_layout.addWidget(self.eta_label)
        
        layout.addLayout(progress_layout)
        
        layout.addStretch()
        
        # Initialize timing variables
        self._start_time = None
        
        # Set scroll content and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
    
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
        self.groups_tree.setEnabled(checked)
        self.groups_hint.setEnabled(checked)
    
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
    
    def _browse_input(self):
        """Browse for image files (one or multiple)."""
        start_dir = ""
        if self._selected_files:
            start_dir = str(Path(self._selected_files[0]).parent)
        elif config.get("last_input_dir"):
            start_dir = config.get("last_input_dir")
        
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", start_dir,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if files:
            self._selected_files = files
            # Store the common ancestor (not just files[0].parent) so a later 'Load Previous
            # Results' + edit can recursively find originals across sibling condition folders.
            try:
                root = os.path.commonpath([str(Path(f).parent) for f in files])
            except ValueError:
                root = str(Path(files[0]).parent)
            config.set("last_input_dir", root)
            self._update_input_display()

            # Re-derive groups from the new selection's subfolders (deterministic, no dialog)
            self._group_data = {}
            self._metadata = None
            self.groups_tree.clear()
            if self.use_groups.isChecked():
                self._autogroup_by_subfolder(announce=False)
    
    def _update_input_display(self):
        """Update the input path display."""
        if not self._selected_files:
            self.input_path_edit.clear()
            return
        
        # Show folder path and count
        folder = Path(self._selected_files[0]).parent
        count = len(self._selected_files)
        display_text = f"{str(folder).replace(chr(92), '/')} ({count} file{'s' if count > 1 else ''})"
        self.input_path_edit.setText(display_text)
    
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
        self.progress.setValue(current)
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
        QMessageBox.information(self, "Analysis Complete", f"Results saved to:\n{results['output_dir']}")
    
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
        layout = QVBoxLayout(self)
        
        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)
        
        # Overview tab
        self.overview_widget = QWidget()
        overview_layout = QVBoxLayout(self.overview_widget)
        
        # Top row: Summary + Buttons
        top_row = QHBoxLayout()
        
        # Summary (left)
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout()
        self.summary_label = QLabel("No results loaded. Run analysis first.")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        
        # Open folder button under summary
        self.open_folder_btn = QPushButton("📁 Open Output Folder")
        self.open_folder_btn.clicked.connect(self._open_folder)
        self.open_folder_btn.setVisible(False)
        summary_layout.addWidget(self.open_folder_btn)
        
        summary_group.setLayout(summary_layout)
        top_row.addWidget(summary_group, 2)
        
        # Buttons (right)
        buttons_group = QGroupBox("Actions")
        buttons_layout = QVBoxLayout()
        
        self.load_results_btn = QPushButton("📂 Load Previous Results")
        self.load_results_btn.setToolTip("Load results from a previous analysis session")
        self.load_results_btn.clicked.connect(self._load_previous_results)
        buttons_layout.addWidget(self.load_results_btn)
        
        buttons_layout.addSpacing(10)
        
        self.generate_report_btn = QPushButton("📊 Generate Report")
        self.generate_report_btn.setToolTip(
            "Regenerate annotated images, statistics, visualizations, and the HTML report.\n"
            "Use after editing/correcting results in the labeling window."
        )
        self.generate_report_btn.clicked.connect(self._generate_report)
        buttons_layout.addWidget(self.generate_report_btn)
        
        self.open_report_btn = QPushButton("📄 Open HTML Report")
        self.open_report_btn.clicked.connect(self._open_report)
        self.open_report_btn.setVisible(False)  # Hidden until report is generated
        buttons_layout.addWidget(self.open_report_btn)
        
        buttons_layout.addStretch()
        buttons_group.setLayout(buttons_layout)
        top_row.addWidget(buttons_group, 1)
        
        overview_layout.addLayout(top_row)
        
        # Progress bar for report generation
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        overview_layout.addWidget(self.progress)
        
        # Table (double-click to view image)
        table_label = QLabel("Double-click a row to view and edit image details:")
        overview_layout.addWidget(table_label)
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(6)
        self.summary_table.setHorizontalHeaderLabels(["Filename", "Normal", "ROD", "Artifact", "ROD %", "Total IOD"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)  # Select entire row
        self.summary_table.setSortingEnabled(True)  # Enable column sorting
        self.summary_table.doubleClicked.connect(self._on_table_double_click)
        overview_layout.addWidget(self.summary_table)
        
        self.sub_tabs.addTab(self.overview_widget, "Overview")
        
        # Statistics tab (merged with Visualizations)
        self.stats_widget = QScrollArea()
        self.stats_widget.setWidgetResizable(True)
        self.stats_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stats_content = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_content)
        self.stats_widget.setWidget(self.stats_content)
        self.sub_tabs.addTab(self.stats_widget, "Statistics")
    
    def load_results(self, results: dict):
        self.results = results
        film_summary = results['film_summary']

        total_normal = film_summary['n_normal'].sum()
        total_rod = film_summary['n_rod'].sum()
        total_artifact = film_summary['n_artifact'].sum()
        mean_rod_frac = film_summary['rod_fraction'].mean()

        summary_text = f"""
        <h3>Summary</h3>
        <p><b>Total Images:</b> {len(film_summary)}</p>
        <p><b>Total Deposits:</b> {film_summary['n_total'].sum():.0f}</p>
        <p><b>Normal:</b> {total_normal:.0f} | <b>ROD:</b> {total_rod:.0f} | <b>Artifact:</b> {total_artifact:.0f}</p>
        <p><b>Mean ROD Fraction:</b> {mean_rod_frac*100:.1f}% (±{film_summary['rod_fraction'].std()*100:.1f}%)</p>
        <p><b>Output:</b> {results.get('output_dir', '')}</p>
        """
        self.summary_label.setText(summary_text)
        
        # Show/hide buttons based on state
        output_dir = results.get('output_dir', '')
        self.open_folder_btn.setVisible(bool(output_dir))
        
        # Check if report.html exists
        report_exists = False
        if output_dir:
            report_path = Path(output_dir) / 'report.html'
            report_exists = report_path.exists()
        self.open_report_btn.setVisible(report_exists)
        
        self.summary_table.setSortingEnabled(False)  # Disable during population
        self.summary_table.setRowCount(len(film_summary))
        for i, (_, row) in enumerate(film_summary.iterrows()):
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(row['filename'])))
            # Use NumericTableWidgetItem for numeric columns for proper sorting
            self.summary_table.setItem(i, 1, NumericTableWidgetItem(row['n_normal'], "{:.0f}"))
            self.summary_table.setItem(i, 2, NumericTableWidgetItem(row['n_rod'], "{:.0f}"))
            self.summary_table.setItem(i, 3, NumericTableWidgetItem(row['n_artifact'], "{:.0f}"))
            self.summary_table.setItem(i, 4, NumericTableWidgetItem(row['rod_fraction']*100, "{:.1f}%"))
            self.summary_table.setItem(i, 5, NumericTableWidgetItem(row.get('total_iod', 0), "{:.0f}"))
        self.summary_table.setSortingEnabled(True)  # Re-enable after population
        
        self._load_statistics_tab(results)
    
    def _load_statistics_tab(self, results: dict):
        """Load combined statistics and visualizations."""
        # Clear existing
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        film_summary = results['film_summary']
        viz_results = results.get('viz_results', {})
        stats_results = results.get('stats_results', {})
        spatial_stats = results.get('spatial_stats', {})
        
        # ===== Visualizations Section =====
        if viz_results:
            viz_group = QGroupBox("📊 Visualizations")
            viz_inner = QVBoxLayout()
            
            # Grid layout for images (2 columns)
            from PySide6.QtWidgets import QGridLayout
            viz_grid = QGridLayout()
            viz_grid.setSpacing(15)
            
            for idx, (name, path) in enumerate(viz_results.items()):
                btn = QPushButton()
                btn.setToolTip(f"Click to enlarge: {name}")
                
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(380, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    btn.setIcon(QIcon(scaled))
                    btn.setIconSize(scaled.size())
                    btn.setFixedSize(scaled.size() + QSize(10, 10))
                    btn.clicked.connect(lambda checked, p=path, n=name: self._show_image_dialog(p, n))
                
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.addWidget(btn)
                label = QLabel(self._format_viz_name(name))
                label.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(label)
                
                row = idx // 2
                col = idx % 2
                viz_grid.addWidget(container, row, col)
            
            viz_inner.addLayout(viz_grid)
            viz_group.setLayout(viz_inner)
            self.stats_layout.addWidget(viz_group)
        
        # ===== Descriptive Statistics Section =====
        desc_group = QGroupBox("📈 Descriptive Statistics")
        desc_layout = QVBoxLayout()
        
        desc_text = self._generate_descriptive_stats(film_summary)
        desc_label = QLabel(desc_text)
        desc_label.setWordWrap(True)
        desc_label.setTextFormat(Qt.RichText)
        desc_layout.addWidget(desc_label)
        
        desc_group.setLayout(desc_layout)
        self.stats_layout.addWidget(desc_group)
        
        # ===== Group Comparison Section =====
        if stats_results:
            comp_group = QGroupBox("📉 Group Comparisons")
            comp_layout = QVBoxLayout()
            
            comp_text = self._generate_comparison_stats(stats_results)
            comp_label = QLabel(comp_text)
            comp_label.setWordWrap(True)
            comp_label.setTextFormat(Qt.RichText)
            comp_layout.addWidget(comp_label)
            
            comp_group.setLayout(comp_layout)
            self.stats_layout.addWidget(comp_group)
        
        # ===== Spatial Analysis Section =====
        if spatial_stats:
            spatial_group = QGroupBox("🗺️ Spatial Analysis")
            spatial_layout = QVBoxLayout()
            
            spatial_text = self._generate_spatial_stats(spatial_stats)
            spatial_label = QLabel(spatial_text)
            spatial_label.setWordWrap(True)
            spatial_label.setTextFormat(Qt.RichText)
            spatial_layout.addWidget(spatial_label)
            
            spatial_group.setLayout(spatial_layout)
            self.stats_layout.addWidget(spatial_group)
        
        self.stats_layout.addStretch()
    
    def _format_viz_name(self, name: str) -> str:
        """Format visualization key names for display."""
        # Special mappings for known keys
        name_map = {
            'dashboard': 'Dashboard',
            'pca': 'PCA Analysis',
            'heatmap': 'Feature Heatmap',
            'scatter_matrix': 'Feature Relationships',
            'area_iod': 'Area vs IOD',
            'nnd_histogram': 'Nearest Neighbor Distance',
            'clark_evans': 'Clark-Evans Index',
            'density_map': 'Deposit Density Map',
            'quadrant_plot': 'Quadrant Analysis',
            'violin_total_iod': 'Total IOD Distribution',
            'violin_rod_fraction': 'ROD Fraction Distribution',
            'violin_n_deposits': 'Deposit Count Distribution',
            'violin_mean_area': 'Mean Area Distribution',
        }
        
        if name in name_map:
            return name_map[name]
        
        # Generic formatting: remove prefix, replace underscores, proper case
        formatted = name
        for prefix in ['violin_', 'box_', 'bar_', 'scatter_']:
            if formatted.startswith(prefix):
                formatted = formatted[len(prefix):]
                break
        
        # Special abbreviations that should stay uppercase
        upper_words = {'iod': 'IOD', 'rod': 'ROD', 'nnd': 'NND', 'pca': 'PCA'}
        words = formatted.split('_')
        formatted_words = []
        for w in words:
            if w.lower() in upper_words:
                formatted_words.append(upper_words[w.lower()])
            else:
                formatted_words.append(w.capitalize())
        
        return ' '.join(formatted_words)
    
    def _generate_descriptive_stats(self, film_summary: pd.DataFrame) -> str:
        """Generate detailed descriptive statistics."""
        from scipy import stats as sp_stats
        
        text = "<table style='border-collapse: collapse; width: 100%;'>"
        text += "<tr style='background-color: #3D3D4D;'>"
        text += "<th style='padding: 8px; text-align: left;'>Metric</th>"
        text += "<th style='padding: 8px;'>Mean</th>"
        text += "<th style='padding: 8px;'>SD</th>"
        text += "<th style='padding: 8px;'>Median</th>"
        text += "<th style='padding: 8px;'>IQR</th>"
        text += "<th style='padding: 8px;'>95% CI</th>"
        text += "<th style='padding: 8px;'>Normal?</th>"
        text += "</tr>"
        
        metrics = [
            ('ROD Fraction', film_summary['rod_fraction'] * 100, '%'),
            ('Total Deposits', film_summary['n_total'], ''),
            ('Normal Count', film_summary['n_normal'], ''),
            ('ROD Count', film_summary['n_rod'], ''),
        ]
        
        if 'total_iod' in film_summary.columns:
            metrics.append(('Total IOD', film_summary['total_iod'], ''))
        
        for name, data, unit in metrics:
            data = data.dropna()
            if len(data) < 2:
                continue
            
            mean = data.mean()
            std = data.std()
            median = data.median()
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            sem = std / np.sqrt(len(data))
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem
            
            # Normality test
            if len(data) >= 3:
                _, p_norm = sp_stats.shapiro(data[:5000])
                is_normal = "✓" if p_norm > 0.05 else "✗"
            else:
                is_normal = "-"
            
            text += f"<tr>"
            text += f"<td style='padding: 6px;'><b>{name}</b></td>"
            text += f"<td style='padding: 6px; text-align: center;'>{mean:.2f}{unit}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{std:.2f}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{median:.2f}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{iqr:.2f}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>[{ci_low:.2f}, {ci_high:.2f}]</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{is_normal}</td>"
            text += "</tr>"
        
        text += "</table>"
        text += f"<p style='color: #B0B0B0; font-size: 11px;'>n = {len(film_summary)} images | CI = Confidence Interval | Normal? = Shapiro-Wilk p > 0.05</p>"
        
        return text
    
    def _generate_comparison_stats(self, stats_results: dict) -> str:
        """Generate group comparison statistics."""
        text = ""
        
        if not isinstance(stats_results, dict):
            return text
        
        for metric, result in stats_results.items():
            # Skip if result is not a dict
            if not isinstance(result, dict):
                continue
            if 'error' in result:
                text += f"<p><b>{metric}:</b> {result['error']}</p>"
                continue
            
            text += f"<h4 style='color: {Theme.PRIMARY_LIGHT};'>{metric}</h4>"
            
            if 'overall_test' in result:
                # Multiple group comparison
                text += f"<p><b>Test:</b> {result['overall_test']}</p>"
                text += f"<p><b>Statistic:</b> {result['overall_statistic']:.3f}</p>"
                p = result['overall_p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                text += f"<p><b>p-value:</b> {p:.4f} {sig}</p>"
                
                if result.get('pairwise_comparisons'):
                    text += "<p><b>Pairwise:</b></p><ul>"
                    for pair in result['pairwise_comparisons']:
                        p_corr = pair.get('p_value_corrected', pair['p_value'])
                        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else ""
                        text += f"<li>{pair['group1_name']} vs {pair['group2_name']}: "
                        text += f"p={p_corr:.4f}{sig}, d={pair['cohens_d']:.2f} ({pair['effect_size']})</li>"
                    text += "</ul>"
            else:
                # Two group comparison
                text += f"<p><b>Groups:</b> {result['group1_name']} (n={result['n1']}) vs {result['group2_name']} (n={result['n2']})</p>"
                text += f"<p><b>Means:</b> {result['mean1']:.3f} ± {result['std1']:.3f} vs {result['mean2']:.3f} ± {result['std2']:.3f}</p>"
                text += f"<p><b>Test:</b> {result['test_name']}</p>"
                
                p = result['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                color = Theme.ROD if p < 0.05 else Theme.TEXT_SECONDARY
                text += f"<p><b>p-value:</b> <span style='color: {color};'>{p:.4f} {sig}</span></p>"
                text += f"<p><b>Effect size:</b> Cohen's d = {result['cohens_d']:.2f} ({result['effect_size']})</p>"
        
        text += "<p style='color: #B0B0B0; font-size: 11px;'>* p < 0.05 | ** p < 0.01 | *** p < 0.001</p>"
        return text
    
    def _generate_spatial_stats(self, spatial_stats: dict) -> str:
        """Generate spatial analysis statistics."""
        text = "<table style='border-collapse: collapse;'>"
        
        if 'mean_nnd' in spatial_stats:
            text += f"<tr><td style='padding: 6px;'><b>Mean Nearest Neighbor Distance:</b></td>"
            text += f"<td style='padding: 6px;'>{spatial_stats['mean_nnd']:.1f} px</td></tr>"
        
        if 'mean_clark_evans' in spatial_stats:
            r = spatial_stats['mean_clark_evans']
            pattern = "clustered" if r < 1 else "regular" if r > 1 else "random"
            text += f"<tr><td style='padding: 6px;'><b>Clark-Evans R:</b></td>"
            text += f"<td style='padding: 6px;'>{r:.3f} ({pattern})</td></tr>"
        
        if 'density_per_mm2' in spatial_stats:
            text += f"<tr><td style='padding: 6px;'><b>Deposit Density:</b></td>"
            text += f"<td style='padding: 6px;'>{spatial_stats['density_per_mm2']:.2f} /mm²</td></tr>"
        
        text += "</table>"
        text += "<p style='color: #B0B0B0; font-size: 11px;'>Clark-Evans R: &lt;1 clustered, =1 random, &gt;1 regular</p>"
        return text
    
    def _show_image_dialog(self, path: str, title: str):
        dialog = ImageViewerDialog(path, title, self)
        dialog.exec()
    
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
        summary_path = output_dir / 'image_summary.csv'
        if summary_path.exists():
            self.results['film_summary'] = pd.read_csv(summary_path)
        
        # Reload deposit_data
        all_deposits_path = output_dir / 'all_deposits.csv'
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
            summary_path = output_dir / 'image_summary.csv'
            if not summary_path.exists():
                summary_path = output_dir / 'film_summary.csv'  # Backward compatibility
            all_deposits_path = output_dir / 'all_deposits.csv'
            
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
            self.load_results(self.results)
            QMessageBox.information(self, "Success", "Report generated successfully!")
            
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
        summary_path = output_dir / 'image_summary.csv'
        if not summary_path.exists():
            summary_path = output_dir / 'film_summary.csv'  # Backward compatibility
        deposits_path = output_dir / 'all_deposits.csv'
        
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
        
        # Header with settings button
        header_layout = QHBoxLayout()
        
        header = QLabel(f'<span style="color: {Theme.PRIMARY};">S</span><span style="color: {Theme.TEXT_SECONDARY};">CAT</span>')
        header.setFont(QFont("Noto Sans", 24, QFont.Bold))
        header.setStyleSheet("padding: 10px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()

        self.assistant_btn = QPushButton("💬 Assistant")
        self.assistant_btn.setCheckable(True)
        self.assistant_btn.setToolTip("Show/hide the conversational assistant panel")
        self.assistant_btn.clicked.connect(self._toggle_chat_dock)
        header_layout.addWidget(self.assistant_btn)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(settings_btn)

        layout.addLayout(header_layout)
        
        subtitle = QLabel("Spot Classification and Analysis Tool for Drosophila Excreta")
        subtitle.setStyleSheet("color: #666; padding-left: 10px;")
        layout.addWidget(subtitle)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Analysis tab (primary)
        self.analysis_tab = AnalysisTab()
        self.analysis_tab.analysis_complete.connect(self._on_analysis_complete)
        self.tabs.addTab(self.analysis_tab, "Analysis")
        
        # Results tab
        self.results_tab = ResultsTab()
        self.tabs.addTab(self.results_tab, "Results")
        
        # Setup tab (contains Labeling and Training as sub-tabs)
        self.setup_tab = QWidget()
        setup_layout = QVBoxLayout(self.setup_tab)
        
        setup_intro = QLabel(
            "Initial setup for your analysis environment. "
            "Use Labeling to create training data, then Training to build a custom model."
        )
        setup_intro.setWordWrap(True)
        setup_intro.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; padding: 10px;")
        setup_layout.addWidget(setup_intro)
        
        self.setup_sub_tabs = QTabWidget()
        
        # Labeling sub-tab
        self.labeling_widget = QWidget()
        labeling_layout = QVBoxLayout(self.labeling_widget)
        
        labeling_desc = QLabel(
            "Create labeled training data by manually marking deposits in images.\n"
            "This is typically done once per microscope/camera setup."
        )
        labeling_desc.setWordWrap(True)
        labeling_desc.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        labeling_layout.addWidget(labeling_desc)
        
        launch_labeling = QPushButton("Launch Labeling Window")
        launch_labeling.setMinimumHeight(60)
        launch_labeling.clicked.connect(self._launch_labeling)
        labeling_layout.addWidget(launch_labeling)
        labeling_layout.addStretch()
        
        self.setup_sub_tabs.addTab(self.labeling_widget, "Labeling")
        
        # Training sub-tab
        self.training_tab = TrainingTab()
        self.setup_sub_tabs.addTab(self.training_tab, "Training")
        
        setup_layout.addWidget(self.setup_sub_tabs)
        self.tabs.addTab(self.setup_tab, "Setup")

        # Conversational assistant dock (lazy — the agent stack is optional)
        self._build_chat_dock()

        self.statusBar().showMessage("Ready")

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
        self.chat_dock.setVisible(checked)

    def _setup_shortcuts(self):
        # Global shortcuts
        QShortcut(QKeySequence(config.get_shortcut("quit")), self, self.close)
        QShortcut(QKeySequence(config.get_shortcut("run_analysis")), self, self._run_analysis_shortcut)
    
    def _run_analysis_shortcut(self):
        self.tabs.setCurrentWidget(self.analysis_tab)
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
    
    def _on_analysis_complete(self, results):
        self.results_tab.load_results(results)
        self.tabs.setCurrentWidget(self.results_tab)
    
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
