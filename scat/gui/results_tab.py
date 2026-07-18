"""The result window (ResultsTab) — the "Verdict + Worklist" triage surface — plus the
_results_dict_from_output helper that builds its data dict from an analysis output dir."""
import json
from pathlib import Path
import pandas as pd
import cv2

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor
)

from ..config import config
from ..artifacts import IMAGE_SUMMARY, ALL_DEPOSITS
from .. import metrics as _metrics
from .. import confidence as _conf
from ..ui_common import (
    Theme, CenteredCap, NumericTableWidgetItem, icon
)
from .. import ui_motion


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
            if not m.is_circular:    # hue is circular — a min/max range is misleading
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
                            results.get("confidence_threshold", _metrics.DEFAULT_THRESHOLD))["line"])

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
            results.get("deposit_data"), results.get("confidence_threshold", _metrics.DEFAULT_THRESHOLD))

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
        from ..labeling_gui import LabelingWindow
        
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
            from ..pathutils import open_in_os
            if not open_in_os(path):
                QMessageBox.information(
                    self, "Open folder",
                    f"Couldn't open the folder automatically. It is at:\n\n{path}")
    
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
            from ..analyzer import Analyzer, deposits_from_labels_json
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
            from ..pipeline import run_statistics_service, generate_report_service
            from ..visualization import generate_all_visualizations

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
                from ..pathutils import open_in_os
                if not open_in_os(report_path):
                    QMessageBox.information(
                        self, "Open report",
                        f"Couldn't open the report automatically. Open it manually:\n\n{report_path}")
    
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
