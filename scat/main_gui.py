"""
Main GUI application for SCAT.
Integrates labeling, training, analysis, and results viewing.
"""

import sys
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
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QGroupBox, QSpinBox,
    QFormLayout, QComboBox, QCheckBox, QProgressBar, QTextEdit,
    QMessageBox, QScrollArea, QDialog, QDialogButtonBox,
    QLineEdit, QMenu, QDockWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QFont, QIcon, QKeySequence, QShortcut,
    QFontDatabase
)

# Import SCAT modules
from .config import config
from .ui_common import (
    Theme, NoScrollSpinBox, NoScrollComboBox, CenteredCap,
    icon, load_custom_fonts, get_icon_path
)
from . import ui_motion

# Note: trainer is imported lazily when needed to avoid loading sklearn at startup
# GUI tabs and shared widgets now live in scat.gui.*; re-exported here so existing
# `from scat.main_gui import ...` call sites (tests, cli) keep working unchanged.
from .gui.widgets import ShortcutEditor, WorkerThread, PathSelector, DropZone  # noqa: F401
from .gui.results_tab import ResultsTab, _results_dict_from_output  # noqa: F401
from .gui.analysis_tab import AnalysisTab  # noqa: F401


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
        shortcuts_widget.setObjectName("scrollContent")   # dark themed pane (else default light bg)
        shortcuts_layout = QVBoxLayout(shortcuts_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
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
        perf_widget.setObjectName("scrollContent")
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

        # Assistant tab — the Anthropic API key for the billed API backend. The subscription
        # backend (a logged-in `claude` CLI) needs no key, so this is optional.
        assistant_widget = QWidget()
        assistant_widget.setObjectName("scrollContent")
        assistant_layout = QVBoxLayout(assistant_widget)

        key_group = QGroupBox("Anthropic API")
        key_form = QFormLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-ant-…  (leave blank to use your Claude subscription)")
        self.api_key_edit.setText(config.get("agent.api_key", ""))
        key_form.addRow("API key:", self.api_key_edit)
        key_group.setLayout(key_form)
        assistant_layout.addWidget(key_group)

        key_note = QLabel(
            "Only needed for the billed API backend. If you're logged in to Claude (the "
            "`claude` CLI), the Assistant uses your subscription and no key is required.\n\n"
            "The key is stored in this app's config file in plain text; the ANTHROPIC_API_KEY "
            "environment variable, if set, takes precedence over it.")
        key_note.setWordWrap(True)
        key_note.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 11px;")
        assistant_layout.addWidget(key_note)
        assistant_layout.addStretch()

        tabs.addTab(assistant_widget, "Assistant")

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

        # Save the assistant API key (stripped; blank clears it)
        config.set("agent.api_key", self.api_key_edit.text().strip())

        self.accept()


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
            # When the Assistant finishes an analysis, load its results into the workspace so they
            # are reviewable there (the agent produced the files but never drove the results view).
            self.chat_widget.analysis_ready.connect(self._on_agent_analysis_ready)
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

    def _on_agent_analysis_ready(self, output_dir):
        """The Assistant finished an analysis (or (re)built a report): load its results into the
        workspace so they are reviewable there, exactly like a manual run."""
        try:
            self.analysis_tab.load_results_from_dir(output_dir)
        except Exception:
            pass

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
