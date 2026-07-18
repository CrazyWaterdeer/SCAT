"""The analysis setup tab (AnalysisTab): input drop zone, options, grouping, run flow."""
import os
from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QProgressBar, QMessageBox, QScrollArea, QMenu, QInputDialog,
    QTreeWidget, QTreeWidgetItem, QSizePolicy, QGridLayout,
    QStackedWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QPixmap, QColor
)

from ..config import config, get_timestamped_output_dir
from ..ui_common import (
    Theme, NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox,
    CollapsibleSection, ToggleSwitch, setting_row, icon
)
from .. import ui_motion

from .widgets import DropZone, PathSelector, WorkerThread
from .results_tab import ResultsTab, _results_dict_from_output


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
            Theme.primary_button_style())
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
        # Derive grouping from the LOADED results (not the configure-page _group_data),
        # so a loaded previous grouped run labels correctly too.
        gcol = results.get("group_by") if isinstance(results, dict) else None
        groups = []
        if gcol and fs is not None and gcol in getattr(fs, "columns", []):
            for g in fs[gcol].dropna().unique():
                s = str(g).strip()
                if s and s != "ungrouped" and s not in groups:
                    groups.append(s)
        grp = f"{len(groups)} groups" if len(groups) >= 2 else "single group"
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

    def load_results_from_dir(self, output_dir):
        """Load an analysis output dir into the results surface and switch to it — used when the
        Assistant (agent) finishes an analysis, so its results are reviewable in the workspace just
        like a manual run (the agent produces the files but never entered this stateful workspace)."""
        from .results_tab import _results_dict_from_output
        try:
            results = _results_dict_from_output(str(output_dir))
        except Exception:
            return
        self.results_view.load_results(results)
        fs = results.get("film_summary")
        try:
            n = len(fs)
        except Exception:
            n = 0
        self.results_bar_label.setText(
            f"✓  Assistant analyzed {n} image{'s' if n != 1 else ''}  ·  {Path(output_dir).name}")
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
        from ..grouping_util import duplicate_basenames
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
        from ..grouping_util import duplicate_basenames
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
        # Re-entrancy guard: the Run keyboard shortcut bypasses the disabled button, and
        # _rerun also calls in — never start a second concurrent worker over a second dir.
        if getattr(self, 'worker', None) is not None and self.worker.isRunning():
            return
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
        from ..pipeline import analyze_folder_service, run_statistics_service, generate_report_service

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
        # Friendly one-line summary up front; keep the full payload (incl. traceback) in the
        # collapsible details pane so a bench scientist isn't shown a raw Python exception.
        text = str(error_msg).strip()
        summary = text.splitlines()[0] if text else "Unknown error"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Analysis failed")
        box.setText("The analysis could not be completed.")
        box.setInformativeText(summary)
        box.setDetailedText(text)
        box.exec()
