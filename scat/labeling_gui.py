"""
Labeling GUI for deposit annotation with manual editing.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene, 
    QGraphicsPathItem, QGraphicsRectItem, QToolBar, QSplitter, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QTableWidget, QCheckBox, QComboBox,
    QTableWidgetItem, QHeaderView, QButtonGroup, QRadioButton, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, QRectF, QTimer, Signal
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, 
    QPainterPath, QAction, QShortcut, QKeySequence,
    QFont, QFontDatabase
)

from .detector import DepositDetector, Deposit
from .features import FeatureExtractor
from .config import config
from .ui_common import (
    Theme, NoScrollSpinBox, NoScrollDoubleSpinBox, NumericTableWidgetItem,
    load_custom_fonts
)


class DepositGraphicsItem(QGraphicsPathItem):
    COLORS = {
        'rod': QColor(255, 0, 0, 150),
        'normal': QColor(0, 255, 0, 150),
        'artifact': QColor(128, 128, 128, 150),
        'unknown': QColor(255, 255, 0, 150)
    }
    
    # Colors for group visualization
    GROUP_COLORS = [
        QColor(255, 165, 0),    # Orange
        QColor(0, 255, 255),    # Cyan
        QColor(255, 0, 255),    # Magenta
        QColor(255, 255, 0),    # Yellow
        QColor(0, 128, 255),    # Light blue
        QColor(255, 128, 0),    # Dark orange
        QColor(128, 0, 255),    # Purple
        QColor(0, 255, 128),    # Mint
    ]
    
    def __init__(self, deposit: Deposit, scale: float = 1.0):
        super().__init__()
        self.deposit = deposit
        self.scale = scale
        self.selected = False
        self._create_path()
        self._update_appearance()
    
    def _create_path(self):
        path = QPainterPath()
        contour = self.deposit.contour.squeeze()
        if len(contour.shape) == 1:
            contour = contour.reshape(1, 2)
        path.moveTo(contour[0][0] * self.scale, contour[0][1] * self.scale)
        for point in contour[1:]:
            path.lineTo(point[0] * self.scale, point[1] * self.scale)
        path.closeSubpath()
        self.setPath(path)
    
    def _update_appearance(self):
        color = self.COLORS.get(self.deposit.label, self.COLORS['unknown'])
        
        # Line width: 0.8 when selected, 0.4 when not
        # Alpha: 30 when selected, 10 when not
        line_width = 0.8 if self.selected else 0.4
        alpha = 30 if self.selected else 10
        
        # If grouped, use dashed line with group color
        if self.deposit.group_id is not None:
            group_color = self.GROUP_COLORS[self.deposit.group_id % len(self.GROUP_COLORS)]
            pen = QPen(group_color, line_width)
            pen.setStyle(Qt.DashLine)
            self.setPen(pen)
        else:
            self.setPen(QPen(color.darker(150), line_width))
        
        fill = QColor(color)
        fill.setAlpha(alpha)
        self.setBrush(QBrush(fill))
    
    def set_label(self, label: str):
        self.deposit.label = label
        self._update_appearance()
    
    def set_selected(self, selected: bool):
        self.selected = selected
        self._update_appearance()
    
    def paint(self, painter, option, widget=None):
        """Override to prevent Qt's default selection dotted line and optionally draw ID."""
        from PySide6.QtWidgets import QStyle
        from PySide6.QtGui import QFont, QFontDatabase
        
        # Remove the selected state to prevent default selection rectangle
        option.state &= ~QStyle.State_Selected
        super().paint(painter, option, widget)
        
        # Draw ID number if enabled
        if hasattr(self, 'show_id') and self.show_id:
            # Use Noto Sans if available, fallback to system font
            font = QFont("Noto Sans", 9, QFont.Bold)
            painter.setFont(font)
            
            # Use deposit label color for ID text
            color = self.COLORS.get(self.deposit.label, self.COLORS['unknown'])
            painter.setPen(QPen(color.darker(120)))
            
            # Draw at centroid
            cx, cy = self.deposit.centroid
            painter.drawText(int(cx * self.scale) - 5, int(cy * self.scale) + 3, str(self.deposit.id))
    
    def set_show_id(self, show: bool):
        """Set whether to display ID number."""
        self.show_id = show
        self.update()
    
    def update_group_visual(self, group_id):
        """Update visual appearance for grouping."""
        self.deposit.group_id = group_id
        self._update_appearance()


class ImageViewer(QGraphicsView):
    MODE_PAN = 0      # Pan/move view
    MODE_SELECT = 1   # Select deposits (with box selection)
    MODE_ADD = 2      # Add new deposits
    
    # Add shape modes
    ADD_RECT = 0
    ADD_CIRCLE = 1
    ADD_FREEFORM = 2
    ADD_MANUAL = 3    # Manual mode: draw area directly as deposit
    
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.pixmap_item = None
        self.deposit_items: List[DepositGraphicsItem] = []
        self.selected_item: Optional[DepositGraphicsItem] = None
        self.selected_items: List[DepositGraphicsItem] = []
        self.scale_factor = 1.0
        
        self.edit_mode = self.MODE_PAN  # Default to pan mode
        self.add_shape = self.ADD_RECT  # Default add shape
        self.drawing = False
        self.start_point = None
        self.rect_item = None
        self.ellipse_item = None  # For circle preview
        self.freeform_points = []  # For freeform mode
        self.freeform_path_item = None
        self.manual_points = []  # For manual mode
        self.manual_path_item = None
        self.selection_rect = None  # For box selection
        
        # Space bar temporary pan mode
        self._space_pressed = False
        self._mode_before_space = None
        self.setFocusPolicy(Qt.StrongFocus)  # Enable key events
    
    def keyPressEvent(self, event):
        """Handle key press - Space for temporary pan mode."""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self.edit_mode != self.MODE_PAN and not self._space_pressed:
                self._space_pressed = True
                self._mode_before_space = self.edit_mode
                self.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle key release - restore mode after Space."""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self._space_pressed and self._mode_before_space is not None:
                self._space_pressed = False
                if self._mode_before_space == self.MODE_PAN:
                    self.setDragMode(QGraphicsView.ScrollHandDrag)
                else:
                    self.setDragMode(QGraphicsView.NoDrag)
                self._mode_before_space = None
        else:
            super().keyReleaseEvent(event)
    
    def set_mode(self, mode: int):
        self.edit_mode = mode
        self.selected_items.clear()
        if mode == self.MODE_PAN:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
    
    def load_image(self, image: np.ndarray):
        self.scene.clear()
        self.deposit_items.clear()
        self.selected_item = None
        self.selected_items.clear()
        h, w, ch = image.shape
        qimg = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
    
    def add_deposit(self, deposit: Deposit):
        item = DepositGraphicsItem(deposit, self.scale_factor)
        item.setFlag(QGraphicsPathItem.ItemIsSelectable)
        self.scene.addItem(item)
        self.deposit_items.append(item)
    
    def remove_deposit_item(self, item: DepositGraphicsItem):
        if item in self.deposit_items:
            self.deposit_items.remove(item)
            self.scene.removeItem(item)
            if self.selected_item == item:
                self.selected_item = None
            if item in self.selected_items:
                self.selected_items.remove(item)
    
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
    
    def mousePressEvent(self, event):
        # Space bar temporary pan mode - let Qt handle pan
        if self._space_pressed:
            super().mousePressEvent(event)
            return
        
        if self.edit_mode == self.MODE_PAN:
            # Let Qt handle pan with ScrollHandDrag
            super().mousePressEvent(event)
            
        elif self.edit_mode == self.MODE_SELECT:
            item = self.itemAt(event.pos())
            
            # Ctrl+click: multi-select
            if event.modifiers() & Qt.ControlModifier:
                if isinstance(item, DepositGraphicsItem):
                    # Add existing selected_item to selected_items if present
                    if self.selected_item and self.selected_item not in self.selected_items:
                        self.selected_items.append(self.selected_item)
                        self.selected_item = None
                    
                    if item in self.selected_items:
                        item.set_selected(False)
                        self.selected_items.remove(item)
                    else:
                        item.set_selected(True)
                        self.selected_items.append(item)
                    # Set selection source to image
                    if self.parent_window:
                        self.parent_window._selection_source = 'image'
            elif isinstance(item, DepositGraphicsItem):
                # Clicked on deposit: single selection
                # Clear existing multi-selection
                for sel_item in self.selected_items:
                    sel_item.set_selected(False)
                self.selected_items.clear()
                
                if self.selected_item and self.selected_item != item:
                    self.selected_item.set_selected(False)
                self.selected_item = item
                item.set_selected(True)
                # Set selection source to image
                if self.parent_window:
                    self.parent_window._selection_source = 'image'
            else:
                # Clicked on empty space: start box selection or clear selection
                # Clear existing selection
                for sel_item in self.selected_items:
                    sel_item.set_selected(False)
                self.selected_items.clear()
                if self.selected_item:
                    self.selected_item.set_selected(False)
                    self.selected_item = None
                
                # Start box selection
                self.drawing = True
                self.start_point = self.mapToScene(event.pos())
                self.selection_rect = QGraphicsRectItem()
                self.selection_rect.setPen(QPen(QColor(100, 150, 255), 1, Qt.DashLine))
                self.selection_rect.setBrush(QBrush(QColor(100, 150, 255, 30)))
                self.scene.addItem(self.selection_rect)
            
        elif self.edit_mode == self.MODE_ADD:
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            
            if self.add_shape == self.ADD_FREEFORM:
                # Freeform: start collecting points
                self.freeform_points = [self.start_point]
                self._update_freeform_preview()
            elif self.add_shape == self.ADD_MANUAL:
                # Manual: like freeform but creates deposit directly from drawn shape
                self.manual_points = [self.start_point]
                self._update_manual_preview()
            elif self.add_shape == self.ADD_CIRCLE:
                # Circle: use ellipse item for preview
                from PySide6.QtWidgets import QGraphicsEllipseItem
                self.ellipse_item = QGraphicsEllipseItem()
                self.ellipse_item.setPen(QPen(QColor(0, 0, 255), 2, Qt.DashLine))
                self.scene.addItem(self.ellipse_item)
            else:
                # Rectangle
                self.rect_item = QGraphicsRectItem()
                self.rect_item.setPen(QPen(QColor(0, 0, 255), 2, Qt.DashLine))
                self.scene.addItem(self.rect_item)
    
    def mouseMoveEvent(self, event):
        # Space bar temporary pan mode - let Qt handle pan
        if self._space_pressed:
            super().mouseMoveEvent(event)
            return
        
        if self.edit_mode == self.MODE_SELECT and self.drawing and self.selection_rect:
            # Update selection rectangle
            current = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current).normalized()
            self.selection_rect.setRect(rect)
        elif self.edit_mode == self.MODE_ADD and self.drawing:
            current = self.mapToScene(event.pos())
            
            if self.add_shape == self.ADD_FREEFORM:
                # Add point if far enough from last point (spacing)
                if self.freeform_points:
                    last = self.freeform_points[-1]
                    dist = ((current.x() - last.x())**2 + (current.y() - last.y())**2)**0.5
                    if dist > 3:  # Minimum spacing
                        self.freeform_points.append(current)
                        self._update_freeform_preview()
            elif self.add_shape == self.ADD_MANUAL:
                # Manual: add point if far enough from last point
                if self.manual_points:
                    last = self.manual_points[-1]
                    dist = ((current.x() - last.x())**2 + (current.y() - last.y())**2)**0.5
                    if dist > 3:  # Minimum spacing
                        self.manual_points.append(current)
                        self._update_manual_preview()
            elif self.add_shape == self.ADD_CIRCLE:
                # Circle: center at start_point, radius to current
                radius = ((current.x() - self.start_point.x())**2 + 
                         (current.y() - self.start_point.y())**2)**0.5
                rect = QRectF(
                    self.start_point.x() - radius,
                    self.start_point.y() - radius,
                    radius * 2, radius * 2
                )
                if hasattr(self, 'ellipse_item') and self.ellipse_item:
                    self.ellipse_item.setRect(rect)
            else:
                # Rectangle
                rect = QRectF(self.start_point, current).normalized()
                if self.rect_item:
                    self.rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        # Space bar temporary pan mode - let Qt handle pan
        if self._space_pressed:
            super().mouseReleaseEvent(event)
            return
        
        if self.edit_mode == self.MODE_SELECT and self.drawing and self.selection_rect:
            # Complete box selection
            self.drawing = False
            rect = self.selection_rect.rect()
            self.scene.removeItem(self.selection_rect)
            self.selection_rect = None
            
            # Select all deposits inside the rectangle
            if rect.width() > 5 and rect.height() > 5:
                for item in self.deposit_items:
                    # Check if deposit center is inside selection rect
                    item_center = item.boundingRect().center() + item.pos()
                    if rect.contains(item_center):
                        item.set_selected(True)
                        if item not in self.selected_items:
                            self.selected_items.append(item)
                
                if self.parent_window and self.selected_items:
                    self.parent_window._selection_source = 'image'
        elif self.edit_mode == self.MODE_ADD and self.drawing:
            self.drawing = False
            
            if self.add_shape == self.ADD_FREEFORM:
                # Complete freeform shape
                if len(self.freeform_points) >= 3 and self.parent_window:
                    self.parent_window._add_deposit_from_freeform(self.freeform_points)
                self._clear_freeform()
            elif self.add_shape == self.ADD_MANUAL:
                # Complete manual shape - directly use drawn area as deposit
                if len(self.manual_points) >= 3 and self.parent_window:
                    self.parent_window._add_deposit_from_manual(self.manual_points)
                self._clear_manual()
            elif self.add_shape == self.ADD_CIRCLE:
                if hasattr(self, 'ellipse_item') and self.ellipse_item:
                    rect = self.ellipse_item.rect()
                    self.scene.removeItem(self.ellipse_item)
                    self.ellipse_item = None
                    if rect.width() > 5 and rect.height() > 5 and self.parent_window:
                        self.parent_window._add_deposit_from_circle(rect)
            else:
                # Rectangle
                if self.rect_item:
                    rect = self.rect_item.rect()
                    self.scene.removeItem(self.rect_item)
                    self.rect_item = None
                    if rect.width() > 5 and rect.height() > 5 and self.parent_window:
                        self.parent_window._add_deposit_from_rect(rect)
        else:
            super().mouseReleaseEvent(event)
    
    def _update_freeform_preview(self):
        """Update the freeform path preview."""
        if self.freeform_path_item:
            self.scene.removeItem(self.freeform_path_item)
        
        if len(self.freeform_points) < 2:
            return
        
        path = QPainterPath()
        path.moveTo(self.freeform_points[0])
        for point in self.freeform_points[1:]:
            path.lineTo(point)
        
        self.freeform_path_item = self.scene.addPath(
            path, QPen(QColor(0, 0, 255), 0.3, Qt.DashLine)
        )
    
    def _clear_freeform(self):
        """Clear freeform drawing state."""
        self.freeform_points.clear()
        if self.freeform_path_item:
            self.scene.removeItem(self.freeform_path_item)
            self.freeform_path_item = None
    
    def _update_manual_preview(self):
        """Update the manual path preview."""
        if self.manual_path_item:
            self.scene.removeItem(self.manual_path_item)
        
        if len(self.manual_points) < 2:
            return
        
        path = QPainterPath()
        path.moveTo(self.manual_points[0])
        for point in self.manual_points[1:]:
            path.lineTo(point)
        
        # Use a different color (green) to distinguish from freeform
        self.manual_path_item = self.scene.addPath(
            path, QPen(QColor(0, 200, 0), 0.3, Qt.DashLine)
        )
    
    def _clear_manual(self):
        """Clear manual drawing state."""
        self.manual_points.clear()
        if self.manual_path_item:
            self.scene.removeItem(self.manual_path_item)
            self.manual_path_item = None


class LabelingWindow(QMainWindow):
    MAX_UNDO = 5  # Maximum undo steps
    AUTO_SAVE_INTERVAL = 60000  # 1 minute in milliseconds
    
    # Operating modes
    MODE_LABELING = 0  # New labeling from scratch
    MODE_EDIT = 1      # Edit existing analysis results
    
    # Signal to notify parent when data is saved (for Results tab refresh)
    data_saved = Signal()
    
    def __init__(self, mode: int = 0, edit_data: dict = None):
        """
        Initialize LabelingWindow.
        
        Args:
            mode: MODE_LABELING (0) or MODE_EDIT (1)
            edit_data: Dict with keys for EDIT_MODE:
                - 'image_path': Path to original image
                - 'output_dir': Results output directory
                - 'filename': Original filename
                - 'deposits_df': DataFrame of deposits
                - 'contour_data': Dict of contour info
                - 'next_group_id': Next group ID
        """
        super().__init__()
        
        self.mode = mode
        self.edit_data = edit_data or {}
        
        title = "SCAT - Edit Deposits" if mode == self.MODE_EDIT else "SCAT - Labeling Tool"
        self.setWindowTitle(title)
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.setStyleSheet(Theme.get_labeling_stylesheet())
        
        self.image: Optional[np.ndarray] = None
        self.deposits: List[Deposit] = []
        self.current_file: Optional[Path] = None
        self._last_saved_path: Optional[Path] = None  # For auto-save
        self._has_unsaved_changes = False
        
        # Use sensitive_mode=True to match Analysis tab detection
        self.detector = DepositDetector(sensitive_mode=True)
        self.extractor = FeatureExtractor()
        self.next_id = 1  # Start from 1, not 0
        self.next_group_id = 1  # For grouping deposits
        
        # Undo history
        self._undo_stack: List[List[dict]] = []
        
        # Selection source: 'image' or 'table'
        # Used to determine if auto-advance should happen
        self._selection_source = 'image'
        
        # Auto-save timer
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.timeout.connect(self._auto_save)
        self._auto_save_timer.start(self.AUTO_SAVE_INTERVAL)
        
        self._setup_ui()
        self._setup_shortcuts()
        
        # If EDIT_MODE, load the data
        if mode == self.MODE_EDIT and edit_data:
            self._load_edit_data()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        self.viewer = ImageViewer(parent=self)
        splitter.addWidget(self.viewer)
        
        # Right panel with scroll area
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setMinimumWidth(280)
        right_scroll.setMaximumWidth(350)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(8, 8, 8, 8)
        
        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)
        
        # Edit Mode
        edit_group = QGroupBox("Edit Mode")
        edit_layout = QVBoxLayout()
        
        # Style to completely hide radio button indicator (circle) in all states
        radio_style = """
            QRadioButton::indicator {
                width: 0px;
                height: 0px;
                margin: 0px;
                padding: 0px;
                border: none;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                width: 0px;
                height: 0px;
                border: none;
                background: transparent;
            }
            QRadioButton::indicator:unchecked {
                width: 0px;
                height: 0px;
                border: none;
                background: transparent;
            }
        """
        
        self.mode_group = QButtonGroup()
        self.radio_pan = QRadioButton("Pan (Q)")
        self.radio_pan.setStyleSheet(radio_style)
        self.radio_pan.setChecked(True)  # Default to pan mode
        self.radio_pan.setToolTip("Drag to pan the view")
        self.radio_select = QRadioButton("Select (S)")
        self.radio_select.setStyleSheet(radio_style)
        self.radio_select.setToolTip("Click to select, drag to box-select multiple")
        self.radio_add = QRadioButton("Add Deposit (A)")
        self.radio_add.setStyleSheet(radio_style)
        
        self.mode_group.addButton(self.radio_pan, ImageViewer.MODE_PAN)
        self.mode_group.addButton(self.radio_select, ImageViewer.MODE_SELECT)
        self.mode_group.addButton(self.radio_add, ImageViewer.MODE_ADD)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        
        edit_layout.addWidget(self.radio_pan)
        edit_layout.addWidget(self.radio_select)
        edit_layout.addWidget(self.radio_add)
        
        # Add shape selection (under radio_add)
        shape_layout = QHBoxLayout()
        shape_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        shape_label = QLabel("Shape:")
        shape_label.setFixedWidth(45)
        shape_label.setStyleSheet("background-color: transparent;")
        self.add_shape_combo = QComboBox()
        self.add_shape_combo.addItems(["Rectangle", "Circle", "Freeform", "Manual"])
        self.add_shape_combo.setCurrentIndex(config.get("labeling.add_shape", 0))
        self.add_shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        self.add_shape_combo.setToolTip(
            "Rectangle/Circle/Freeform: Auto-detect deposits in selected area\n"
            "Manual: Draw deposit boundary directly (no auto-detection)"
        )
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.add_shape_combo)
        edit_layout.addLayout(shape_layout)
        
        edit_btn_layout = QHBoxLayout()
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setToolTip("Delete selected deposits (Delete key)")
        self.btn_delete.clicked.connect(self._delete_selected)
        edit_btn_layout.addWidget(self.btn_delete)
        
        self.btn_merge = QPushButton("Merge (R)")
        self.btn_merge.setToolTip("Merge selected deposits into one (Ctrl+Click to multi-select)")
        self.btn_merge.clicked.connect(self._merge_selected)
        edit_btn_layout.addWidget(self.btn_merge)
        
        edit_layout.addLayout(edit_btn_layout)
        
        # Group buttons (separate row)
        group_btn_layout = QHBoxLayout()
        self.btn_group = QPushButton("Group (G)")
        self.btn_group.setToolTip("Group selected deposits (keeps individual contours)")
        self.btn_group.clicked.connect(self._group_selected)
        group_btn_layout.addWidget(self.btn_group)
        
        self.btn_ungroup = QPushButton("Ungroup (F)")
        self.btn_ungroup.setToolTip("Remove selected deposits from their group")
        self.btn_ungroup.clicked.connect(self._ungroup_selected)
        group_btn_layout.addWidget(self.btn_ungroup)
        
        edit_layout.addLayout(group_btn_layout)
        
        edit_group.setLayout(edit_layout)
        right_layout.addWidget(edit_group)
        
        # View Options
        view_group = QGroupBox("View Options")
        view_layout = QVBoxLayout()
        
        self.show_ids_check = QCheckBox("Show ID numbers")
        self.show_ids_check.setStyleSheet("background-color: transparent;")
        self.show_ids_check.setChecked(False)
        self.show_ids_check.toggled.connect(self._toggle_show_ids)
        view_layout.addWidget(self.show_ids_check)
        
        view_group.setLayout(view_layout)
        right_layout.addWidget(view_group)
        
        # Detection settings (hidden in EDIT_MODE)
        self.detect_group = QGroupBox("Detection Settings")
        detect_layout = QVBoxLayout()
        detect_layout.setSpacing(8)
        
        # Common width for spin boxes
        SPIN_WIDTH = 80
        
        # Min Area row
        min_area_row = QHBoxLayout()
        min_area_label = QLabel("Min Area")
        min_area_label.setStyleSheet("background-color: transparent;")
        min_area_row.addWidget(min_area_label)
        min_area_row.addStretch()
        self.min_area_spin = NoScrollSpinBox()
        self.min_area_spin.setRange(1, 1000)
        self.min_area_spin.setValue(20)
        self.min_area_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.min_area_spin.setFixedWidth(SPIN_WIDTH)
        min_area_row.addWidget(self.min_area_spin)
        detect_layout.addLayout(min_area_row)
        
        # Max Area row
        max_area_row = QHBoxLayout()
        max_area_label = QLabel("Max Area")
        max_area_label.setStyleSheet("background-color: transparent;")
        max_area_row.addWidget(max_area_label)
        max_area_row.addStretch()
        self.max_area_spin = NoScrollSpinBox()
        self.max_area_spin.setRange(100, 50000)
        self.max_area_spin.setValue(10000)
        self.max_area_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.max_area_spin.setFixedWidth(SPIN_WIDTH)
        max_area_row.addWidget(self.max_area_spin)
        detect_layout.addLayout(max_area_row)
        
        # ROD Threshold row
        threshold_row = QHBoxLayout()
        threshold_label = QLabel("ROD Threshold")
        threshold_label.setStyleSheet("background-color: transparent;")
        threshold_row.addWidget(threshold_label)
        threshold_row.addStretch()
        self.threshold_spin = NoScrollDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.6)
        self.threshold_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.threshold_spin.setFixedWidth(SPIN_WIDTH)
        threshold_row.addWidget(self.threshold_spin)
        detect_layout.addLayout(threshold_row)
        
        detect_btn = QPushButton("Re-detect")
        detect_btn.clicked.connect(self._detect_deposits)
        detect_layout.addWidget(detect_btn)
        self.detect_group.setLayout(detect_layout)
        right_layout.addWidget(self.detect_group)
        
        # Hide detection settings in EDIT_MODE
        if self.mode == self.MODE_EDIT:
            self.detect_group.hide()
        
        # Labeling
        label_group = QGroupBox("Labeling (1=Normal, 2=ROD, 3=Artifact)")
        label_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_normal = QPushButton("Normal (1)")
        self.btn_normal.setStyleSheet(f"background-color: {Theme.NORMAL}; color: white;")
        self.btn_normal.clicked.connect(lambda: self._set_selected_label("normal"))
        btn_layout.addWidget(self.btn_normal)
        
        self.btn_rod = QPushButton("ROD (2)")
        self.btn_rod.setStyleSheet(f"background-color: {Theme.ROD}; color: white;")
        self.btn_rod.clicked.connect(lambda: self._set_selected_label("rod"))
        btn_layout.addWidget(self.btn_rod)
        
        self.btn_artifact = QPushButton("Artifact (3)")
        self.btn_artifact.setStyleSheet(f"background-color: {Theme.ARTIFACT}; color: white;")
        self.btn_artifact.clicked.connect(lambda: self._set_selected_label("artifact"))
        btn_layout.addWidget(self.btn_artifact)
        label_layout.addLayout(btn_layout)
        
        auto_threshold = QPushButton("Auto (Threshold)")
        auto_threshold.clicked.connect(self._auto_label_threshold)
        label_layout.addWidget(auto_threshold)
        label_group.setLayout(label_layout)
        right_layout.addWidget(label_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        stats_layout.setHorizontalSpacing(15)  # Increase spacing between label and value
        
        # Style for row labels (transparent background, center aligned)
        label_style = "background-color: transparent;"
        # Style for value labels (rounded dark background)
        value_style = f"""
            background-color: {Theme.BG_MEDIUM};
            border-radius: 4px;
            padding: 4px 8px;
            min-width: 40px;
        """
        
        def create_stat_row(text):
            label = QLabel(text)
            label.setStyleSheet(label_style)
            label.setAlignment(Qt.AlignCenter)
            value = QLabel("0")
            value.setStyleSheet(value_style)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return label, value
        
        total_label, self.label_total = create_stat_row("Total")
        normal_label, self.label_normal = create_stat_row("Normal")
        rod_label, self.label_rod = create_stat_row("ROD")
        artifact_label, self.label_artifact = create_stat_row("Artifact")
        unlabeled_label, self.label_unknown = create_stat_row("Unlabeled")
        
        stats_layout.addRow(total_label, self.label_total)
        stats_layout.addRow(normal_label, self.label_normal)
        stats_layout.addRow(rod_label, self.label_rod)
        stats_layout.addRow(artifact_label, self.label_artifact)
        stats_layout.addRow(unlabeled_label, self.label_unknown)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # Table with rounded frame
        table_frame = QFrame()
        table_frame.setObjectName("tableFrame")
        table_frame_layout = QVBoxLayout(table_frame)
        table_frame_layout.setContentsMargins(0, 0, 0, 0)
        table_frame_layout.setSpacing(0)
        
        self.deposit_table = QTableWidget()
        self.deposit_table.setColumnCount(5)
        self.deposit_table.setHorizontalHeaderLabels(["ID", "Area", "Circ", "Hue", "Label"])
        self.deposit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.deposit_table.setMinimumHeight(280)  # ~10 rows visible
        
        # Table cleanup: disable editing, hide row numbers, enable sorting, row selection
        self.deposit_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.deposit_table.verticalHeader().setVisible(False)
        self.deposit_table.setSortingEnabled(True)
        self.deposit_table.setSelectionBehavior(QTableWidget.SelectRows)  # Select entire row
        self.deposit_table.setSelectionMode(QTableWidget.SingleSelection)
        
        self.deposit_table.itemSelectionChanged.connect(self._on_table_select)
        self.deposit_table.doubleClicked.connect(self._on_table_double_click)
        
        table_frame_layout.addWidget(self.deposit_table)
        right_layout.addWidget(table_frame)
        
        right_layout.addStretch()
        
        # Toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        if self.mode == self.MODE_LABELING:
            # Labeling mode: Open Image, Save Labels, Load Labels, Export
            open_action = QAction("Open Image", self)
            open_action.triggered.connect(self._open_image)
            toolbar.addAction(open_action)
            
            save_action = QAction("Save Labels", self)
            save_action.triggered.connect(self._save_labels)
            toolbar.addAction(save_action)
            
            load_action = QAction("Load Labels", self)
            load_action.triggered.connect(self._load_labels)
            toolbar.addAction(load_action)
            
            export_action = QAction("Export Training Data", self)
            export_action.triggered.connect(self._export_training_data)
            toolbar.addAction(export_action)
            
            self.statusBar().showMessage("S: Select, A: Add, Ctrl+Click: Multi-select, M: Merge, D: Delete")
        else:
            # Edit mode: Save Changes only
            save_action = QAction("Save Changes", self)
            save_action.triggered.connect(self._save_edit_changes)
            toolbar.addAction(save_action)
            
            discard_action = QAction("Discard & Close", self)
            discard_action.triggered.connect(self._discard_and_close)
            toolbar.addAction(discard_action)
            
            self.statusBar().showMessage("Edit mode: 1/2/3 to label, R to merge, G to group")
        
        splitter.setSizes([800, 400])
        
        # Initialize viewer mode (fixes Pan mode not working on startup)
        self.viewer.set_mode(ImageViewer.MODE_PAN)
    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("1"), self, lambda: self._set_selected_label("normal"))
        QShortcut(QKeySequence("2"), self, lambda: self._set_selected_label("rod"))
        QShortcut(QKeySequence("3"), self, lambda: self._set_selected_label("artifact"))
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_current)  # Mode-aware save
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected)
        # Mode shortcuts (left hand: Q W E area)
        QShortcut(QKeySequence("Q"), self, lambda: self._set_mode(0))  # Pan
        QShortcut(QKeySequence("S"), self, lambda: self._set_mode(1))  # Select
        QShortcut(QKeySequence("A"), self, lambda: self._set_mode(2))  # Add
        # Action shortcuts
        QShortcut(QKeySequence("R"), self, self._merge_selected)  # meRge
        QShortcut(QKeySequence("G"), self, self._group_selected)  # Group
        QShortcut(QKeySequence("F"), self, self._ungroup_selected)  # Free from group
    
    def _save_current(self):
        """Save based on current mode - Ctrl+S handler."""
        if self.mode == self.MODE_EDIT:
            self._save_edit_changes()
        else:
            self._save_labels()
    
    def _save_state(self):
        """Save current state to undo stack."""
        state = []
        for d in self.deposits:
            state.append({
                'id': d.id,
                'contour': d.contour.copy(),
                'x': d.x,
                'y': d.y,
                'width': d.width,
                'height': d.height,
                'area': d.area,
                'perimeter': d.perimeter,
                'circularity': d.circularity,
                'aspect_ratio': d.aspect_ratio,
                'centroid': d.centroid,
                'mean_hue': d.mean_hue,
                'mean_saturation': d.mean_saturation,
                'mean_lightness': d.mean_lightness,
                'mean_r': d.mean_r,
                'mean_g': d.mean_g,
                'mean_b': d.mean_b,
                'iod': d.iod,
                'label': d.label,
                'confidence': d.confidence,
                'merged': d.merged,
                'group_id': d.group_id,
            })
        
        self._undo_stack.append(state)
        
        # Maintain maximum size
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        
        # Mark as having unsaved changes
        self._mark_unsaved()
    
    def _undo(self):
        """Restore to previous state."""
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return
        
        state = self._undo_stack.pop()
        
        # Remove current viewer items
        for item in self.viewer.deposit_items:
            self.viewer.scene.removeItem(item)
        self.viewer.deposit_items.clear()
        self.viewer.selected_item = None
        self.viewer.selected_items.clear()
        self.deposits.clear()
        
        # Restore state
        for saved in state:
            d = Deposit(
                id=saved['id'],
                contour=saved['contour'],
                x=saved['x'],
                y=saved['y'],
                width=saved['width'],
                height=saved['height'],
                area=saved['area'],
                perimeter=saved['perimeter'],
                circularity=saved['circularity'],
                aspect_ratio=saved['aspect_ratio'],
                centroid=saved['centroid'],
                mean_hue=saved['mean_hue'],
                mean_saturation=saved['mean_saturation'],
                mean_lightness=saved['mean_lightness'],
                mean_r=saved['mean_r'],
                mean_g=saved['mean_g'],
                mean_b=saved['mean_b'],
                iod=saved['iod'],
                label=saved['label'],
                confidence=saved['confidence'],
                merged=saved['merged'],
                group_id=saved['group_id'],
            )
            self.deposits.append(d)
            self.viewer.add_deposit(d)
        
        # Update next_id
        if self.deposits:
            self.next_id = max(d.id for d in self.deposits) + 1
        else:
            self.next_id = 1  # Start from 1, not 0
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Undo ({len(self._undo_stack)} remaining)")
    
    def _set_mode(self, mode):
        if mode == 0:
            self.radio_pan.setChecked(True)
        elif mode == 1:
            self.radio_select.setChecked(True)
        elif mode == 2:
            self.radio_add.setChecked(True)
        self._on_mode_changed()
    
    def _on_mode_changed(self):
        mode = self.mode_group.checkedId()
        self.viewer.set_mode(mode)
        shape_names = ["Rectangle", "Circle", "Freeform", "Manual"]
        shape = shape_names[self.viewer.add_shape] if mode == ImageViewer.MODE_ADD else ""
        modes = {0: "Pan", 1: "Select", 2: f"Add ({shape})"}
        self.statusBar().showMessage(f"Mode: {modes.get(mode, 'Unknown')}")
    
    def _on_shape_changed(self, index: int):
        """Change add shape mode and switch to Add mode."""
        self.viewer.add_shape = index
        self.viewer._clear_freeform()  # Clear any pending freeform
        if hasattr(self.viewer, '_clear_manual'):
            self.viewer._clear_manual()  # Clear any pending manual
        config.set("labeling.add_shape", index)  # Save preference
        
        # Auto-switch to Add mode when shape is selected
        self.radio_add.setChecked(True)
        self._on_mode_changed()
        
        shape_names = [
            "Rectangle - drag to draw", 
            "Circle - drag center to edge", 
            "Freeform - drag to draw (auto-detect)",
            "Manual - drag to draw (exact shape)"
        ]
        self.statusBar().showMessage(f"Add shape: {shape_names[index]}")
    
    def _open_image(self):
        # Use last image directory
        start_dir = config.get("last_image_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", start_dir, 
            "Images (*.tif *.tiff *.png *.jpg)"
        )
        if path:
            # Save directory for next time
            config.set("last_image_dir", str(Path(path).parent))
            self._load_image(Path(path))
    
    def _load_image(self, path: Path):
        from PIL import Image
        self.current_file = path
        self.image = np.array(Image.open(path))
        img = Image.open(path)
        dpi = img.info.get('dpi', (600, 600))[0]
        self.extractor = FeatureExtractor(dpi=dpi)
        self.viewer.load_image(self.image)
        self._detect_deposits()
        self.setWindowTitle(f"SCAT - {path.name}")
    
    def _detect_deposits(self):
        if self.image is None:
            return
        self.detector.min_area = self.min_area_spin.value()
        self.detector.max_area = self.max_area_spin.value()
        self.detector.sensitive_mode = True  # Always match Analysis tab
        self.deposits = self.detector.detect(self.image)
        self.deposits = self.extractor.extract_features(self.image, self.deposits)
        self.next_id = len(self.deposits)
        
        for item in self.viewer.deposit_items:
            self.viewer.scene.removeItem(item)
        self.viewer.deposit_items.clear()
        self.viewer.selected_item = None
        self.viewer.selected_items.clear()
        
        for deposit in self.deposits:
            self.viewer.add_deposit(deposit)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Detected {len(self.deposits)} deposits")
    
    def _add_deposit_from_rect(self, rect: QRectF):
        if self.image is None:
            return
        
        self._save_state()  # Save state before adding
        
        x, y = int(rect.x()), int(rect.y())
        w, h = int(rect.width()), int(rect.height())
        
        h_img, w_img = self.image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        
        roi = self.image[y:y+h, x:x+w]
        if len(roi) > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contour = largest + np.array([x, y])
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
        
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2
        
        deposit = Deposit(
            id=self.next_id, contour=contour,
            x=bx, y=by, width=bw, height=bh,
            area=area, perimeter=perimeter,
            circularity=circularity, aspect_ratio=aspect_ratio,
            centroid=(cx, cy)
        )
        self.next_id += 1
        
        self.deposits.append(deposit)
        self.extractor.extract_features(self.image, [deposit])
        self.viewer.add_deposit(deposit)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Added deposit {deposit.id}")
    
    def _add_deposit_from_circle(self, rect: QRectF):
        """Add deposit from circle (rect is bounding box of circle)."""
        if self.image is None:
            return
        
        self._save_state()
        
        # Circle parameters from bounding rect
        cx = int(rect.center().x())
        cy = int(rect.center().y())
        radius = int(rect.width() / 2)
        
        h_img, w_img = self.image.shape[:2]
        
        # Create circular contour
        num_points = 32
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        contour = np.array([
            [int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))]
            for a in angles
        ])
        
        # Clip to image bounds
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w_img, cx + radius), min(h_img, cy + radius)
        
        roi = self.image[y1:y2, x1:x2]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contour = largest + np.array([x1, y1])
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
        
        M = cv2.moments(contour)
        cx_m = int(M["m10"] / M["m00"]) if M["m00"] > 0 else cx
        cy_m = int(M["m01"] / M["m00"]) if M["m00"] > 0 else cy
        
        deposit = Deposit(
            id=self.next_id, contour=contour,
            x=bx, y=by, width=bw, height=bh,
            area=area, perimeter=perimeter,
            circularity=circularity, aspect_ratio=aspect_ratio,
            centroid=(cx_m, cy_m)
        )
        self.next_id += 1
        
        self.deposits.append(deposit)
        self.extractor.extract_features(self.image, [deposit])
        self.viewer.add_deposit(deposit)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Added deposit {deposit.id}")
    
    def _add_deposit_from_freeform(self, points: list):
        """Add deposit from freeform polygon."""
        if self.image is None or len(points) < 3:
            return
        
        self._save_state()
        
        h_img, w_img = self.image.shape[:2]
        
        # Convert QPointF to numpy array
        contour = np.array([[int(p.x()), int(p.y())] for p in points])
        
        # Get bounding rect
        bx, by, bw, bh = cv2.boundingRect(contour)
        
        # Clip to image bounds
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
        
        roi = self.image[y1:y2, x1:x2]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Create mask from freeform polygon
            mask = np.zeros_like(thresh)
            local_contour = contour - np.array([x1, y1])
            cv2.fillPoly(mask, [local_contour], 255)
            
            # Apply mask to threshold
            thresh = cv2.bitwise_and(thresh, mask)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contour = largest + np.array([x1, y1])
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
        
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else bx + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else by + bh // 2
        
        deposit = Deposit(
            id=self.next_id, contour=contour,
            x=bx, y=by, width=bw, height=bh,
            area=area, perimeter=perimeter,
            circularity=circularity, aspect_ratio=aspect_ratio,
            centroid=(cx, cy)
        )
        self.next_id += 1
        
        self.deposits.append(deposit)
        self.extractor.extract_features(self.image, [deposit])
        self.viewer.add_deposit(deposit)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Added deposit {deposit.id}")
    
    def _add_deposit_from_manual(self, points: list):
        """Add deposit directly from drawn polygon (no auto-detection)."""
        if self.image is None or len(points) < 3:
            return
        
        self._save_state()
        
        # Convert QPointF to numpy array - use drawn shape directly
        contour = np.array([[int(p.x()), int(p.y())] for p in points], dtype=np.int32)
        
        # Close the contour
        contour = contour.reshape((-1, 1, 2))
        
        area = cv2.contourArea(contour)
        if area < 5:  # Too small
            return
            
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
        
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else bx + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else by + bh // 2
        
        deposit = Deposit(
            id=self.next_id, contour=contour,
            x=bx, y=by, width=bw, height=bh,
            area=area, perimeter=perimeter,
            circularity=circularity, aspect_ratio=aspect_ratio,
            centroid=(cx, cy)
        )
        self.next_id += 1
        
        self.deposits.append(deposit)
        self.extractor.extract_features(self.image, [deposit])
        self.viewer.add_deposit(deposit)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Added manual deposit {deposit.id}")
    
    def _delete_selected(self):
        deleted = []
        has_selection = self.viewer.selected_items or self.viewer.selected_item
        
        if has_selection:
            self._save_state()  # Save state before deleting
        
        if self.viewer.selected_items:
            for item in self.viewer.selected_items[:]:
                self.deposits.remove(item.deposit)
                deleted.append(item.deposit.id)
                self.viewer.remove_deposit_item(item)
        elif self.viewer.selected_item:
            item = self.viewer.selected_item
            self.deposits.remove(item.deposit)
            deleted.append(item.deposit.id)
            self.viewer.remove_deposit_item(item)
        
        if deleted:
            self._update_table()
            self._update_stats()
            self.statusBar().showMessage(f"Deleted: {deleted}")
    
    def _merge_selected(self):
        if len(self.viewer.selected_items) < 2:
            self.statusBar().showMessage("Select 2+ deposits to merge")
            return
        
        self._save_state()  # Save state before merge
        
        all_points = []
        for item in self.viewer.selected_items:
            contour = item.deposit.contour.squeeze()
            if len(contour.shape) == 1:
                contour = contour.reshape(1, 2)
            all_points.extend(contour.tolist())
        
        hull = cv2.convexHull(np.array(all_points))
        
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        bx, by, bw, bh = cv2.boundingRect(hull)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
        
        M = cv2.moments(hull)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else bx + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else by + bh // 2
        
        merged = Deposit(
            id=self.next_id, contour=hull,
            x=bx, y=by, width=bw, height=bh,
            area=area, perimeter=perimeter,
            circularity=circularity, aspect_ratio=aspect_ratio,
            centroid=(cx, cy),
            merged=True  # Mark as merged for U-Net training
        )
        self.next_id += 1
        
        for item in self.viewer.selected_items[:]:
            self.deposits.remove(item.deposit)
            self.viewer.remove_deposit_item(item)
        
        self.deposits.append(merged)
        self.extractor.extract_features(self.image, [merged])
        self.viewer.add_deposit(merged)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Merged into deposit {merged.id}")
    
    def _group_selected(self):
        """Group selected deposits together (keeps individual contours)."""
        if len(self.viewer.selected_items) < 2:
            self.statusBar().showMessage("Select 2+ deposits to group")
            return
        
        self._save_state()  # Save state before grouping
        
        # Assign same group_id to all selected deposits
        group_id = self.next_group_id
        self.next_group_id += 1
        
        grouped_ids = []
        for item in self.viewer.selected_items:
            item.deposit.group_id = group_id
            grouped_ids.append(item.deposit.id)
            # Update visual
            item.update_group_visual(group_id)
        
        self._update_table()
        self.statusBar().showMessage(f"Grouped deposits {grouped_ids} into group {group_id}")
    
    def _ungroup_selected(self):
        """Remove selected deposits from their group."""
        items = self.viewer.selected_items if self.viewer.selected_items else (
            [self.viewer.selected_item] if self.viewer.selected_item else []
        )
        
        # Check if there are grouped items
        has_grouped = any(item and item.deposit.group_id is not None for item in items)
        
        if has_grouped:
            self._save_state()  # Save state before ungrouping
        
        ungrouped_ids = []
        for item in items:
            if item and item.deposit.group_id is not None:
                item.deposit.group_id = None
                ungrouped_ids.append(item.deposit.id)
                item.update_group_visual(None)
        
        if ungrouped_ids:
            self._update_table()
            self.statusBar().showMessage(f"Ungrouped deposits {ungrouped_ids}")
        else:
            self.statusBar().showMessage("No grouped deposits selected")

    
    def _update_table(self):
        self.deposit_table.setSortingEnabled(False)  # Disable during update
        self.deposit_table.setRowCount(len(self.deposits))
        for i, d in enumerate(self.deposits):
            # Use NumericTableWidgetItem for proper numeric sorting
            self.deposit_table.setItem(i, 0, NumericTableWidgetItem(d.id))
            self.deposit_table.setItem(i, 1, NumericTableWidgetItem(d.area, "{:.0f}"))
            self.deposit_table.setItem(i, 2, NumericTableWidgetItem(d.circularity, "{:.3f}"))
            self.deposit_table.setItem(i, 3, NumericTableWidgetItem(d.mean_hue, "{:.1f}"))
            self.deposit_table.setItem(i, 4, QTableWidgetItem(d.label))
        self.deposit_table.setSortingEnabled(True)  # Re-enable sorting
    
    def _toggle_show_ids(self, show: bool):
        """Toggle ID number display on deposits."""
        for item in self.viewer.deposit_items:
            item.set_show_id(show)
    
    def _update_stats(self):
        labels = [d.label for d in self.deposits]
        self.label_total.setText(str(len(labels)))
        self.label_normal.setText(str(labels.count("normal")))
        self.label_rod.setText(str(labels.count("rod")))
        self.label_artifact.setText(str(labels.count("artifact")))
        self.label_unknown.setText(str(labels.count("unknown")))
    
    def _set_selected_label(self, label: str):
        has_selection = self.viewer.selected_items or self.viewer.selected_item
        
        if has_selection:
            self._save_state()  # Save state before label change
        
        # Label all if multi-selected
        if self.viewer.selected_items:
            for item in self.viewer.selected_items:
                item.set_label(label)
            self._update_table()
            self._update_stats()
            return
        
        # Single selection
        if self.viewer.selected_item:
            self.viewer.selected_item.set_label(label)
            self._update_table()
            self._update_stats()
            
            # Move to next only if selected from table
            if self._selection_source == 'table':
                self._select_next()
    
    def _auto_label_threshold(self):
        threshold = self.threshold_spin.value()
        for item in self.viewer.deposit_items:
            d = item.deposit
            # ROD: elongated AND concentrated (low lightness)
            if d.circularity < threshold and d.mean_lightness < 0.80:
                label = "rod"
            else:
                label = "normal"
            item.set_label(label)
        self._update_table()
        self._update_stats()
    
    def _select_next(self):
        if not self.viewer.deposit_items:
            return
        current_idx = -1
        if self.viewer.selected_item:
            try:
                current_idx = self.viewer.deposit_items.index(self.viewer.selected_item)
            except ValueError:
                pass
        next_idx = (current_idx + 1) % len(self.viewer.deposit_items)
        if self.viewer.selected_item:
            self.viewer.selected_item.set_selected(False)
        self.viewer.selected_item = self.viewer.deposit_items[next_idx]
        self.viewer.selected_item.set_selected(True)
        self.viewer.centerOn(self.viewer.selected_item)
        self.deposit_table.selectRow(next_idx)
    
    def _select_prev(self):
        if not self.viewer.deposit_items:
            return
        current_idx = 0
        if self.viewer.selected_item:
            try:
                current_idx = self.viewer.deposit_items.index(self.viewer.selected_item)
            except ValueError:
                pass
        prev_idx = (current_idx - 1) % len(self.viewer.deposit_items)
        if self.viewer.selected_item:
            self.viewer.selected_item.set_selected(False)
        self.viewer.selected_item = self.viewer.deposit_items[prev_idx]
        self.viewer.selected_item.set_selected(True)
        self.viewer.centerOn(self.viewer.selected_item)
        self.deposit_table.selectRow(prev_idx)
    
    def _on_table_select(self):
        rows = self.deposit_table.selectionModel().selectedRows()
        if rows:
            row_idx = rows[0].row()
            # Get ID from the table (column 0) - sorted may differ from list order
            id_item = self.deposit_table.item(row_idx, 0)
            if id_item:
                deposit_id = int(id_item.text())
                # Find the corresponding deposit item by ID
                for item in self.viewer.deposit_items:
                    if item.deposit.id == deposit_id:
                        if self.viewer.selected_item:
                            self.viewer.selected_item.set_selected(False)
                        self.viewer.selected_item = item
                        self.viewer.selected_item.set_selected(True)
                        # Mark selection source
                        self._selection_source = 'table'
                        break
    
    def _on_table_double_click(self, index):
        """Navigate to deposit location on double-click."""
        row_idx = index.row()
        id_item = self.deposit_table.item(row_idx, 0)
        if id_item:
            deposit_id = int(id_item.text())
            # Find the corresponding deposit item by ID
            for item in self.viewer.deposit_items:
                if item.deposit.id == deposit_id:
                    # Center view on the deposit
                    self.viewer.centerOn(item)
                    # Optionally zoom in a bit
                    self.viewer.scale(1.5, 1.5)
                    break
    
    def _save_labels(self):
        """Save labels - if path exists, overwrite; otherwise ask for path."""
        if not self.current_file:
            return
        
        # If we have a saved path, just save there (overwrite)
        if self._last_saved_path and self._last_saved_path.exists():
            self._save_to_path(self._last_saved_path)
            self._has_unsaved_changes = False
            self.statusBar().showMessage(f"Saved to {self._last_saved_path}")
            return
        
        # Otherwise, ask for save location (Save As)
        start_dir = config.get("last_label_dir", "")
        if not start_dir:
            start_dir = str(self.current_file.parent)
        default_name = self.current_file.stem + '.labels.json'
        default_path = Path(start_dir) / default_name
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Labels", str(default_path), "JSON (*.json)"
        )
        if path:
            self._save_to_path(Path(path))
            self._last_saved_path = Path(path)
            self._has_unsaved_changes = False
            config.set("last_label_dir", str(Path(path).parent))
            self.statusBar().showMessage(f"Saved to {path}")
    
    def _save_to_path(self, path: Path):
        """Save labels to the specified path."""
        data = {
            'image_file': str(self.current_file),
            'next_group_id': self.next_group_id,
            'deposits': []
        }
        for d in self.deposits:
            deposit_data = {
                'id': d.id,
                'contour': d.contour.tolist(),
                'x': d.centroid[0], 
                'y': d.centroid[1],
                'width': d.width,
                'height': d.height,
                'area': d.area, 
                'circularity': d.circularity, 
                'label': d.label,
                'confidence': d.confidence,
                'merged': d.merged,
                'group_id': d.group_id
            }
            data['deposits'].append(deposit_data)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _auto_save(self):
        """Auto-save if there are unsaved changes and a save path exists."""
        if self._has_unsaved_changes and self._last_saved_path:
            try:
                self._save_to_path(self._last_saved_path)
                self._has_unsaved_changes = False
                self.statusBar().showMessage("Auto-saved", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"Auto-save failed: {e}", 5000)
    
    def _mark_unsaved(self):
        """Mark that there are unsaved changes."""
        self._has_unsaved_changes = True
    
    def closeEvent(self, event):
        """Handle window close with unsaved changes warning."""
        if self._has_unsaved_changes:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                if self._last_saved_path:
                    self._save_to_path(self._last_saved_path)
                else:
                    self._save_labels()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    def _load_labels(self):
        # Use last label directory
        start_dir = config.get("last_label_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Labels", start_dir, "JSON (*.json)"
        )
        if path:
            # Save directory for next time
            config.set("last_label_dir", str(Path(path).parent))
            
            with open(path) as f:
                data = json.load(f)
            
            # Load next_group_id if present
            self.next_group_id = data.get('next_group_id', 1)
            
            has_contours = data['deposits'] and 'contour' in data['deposits'][0]
            
            if has_contours:
                for item in self.viewer.deposit_items:
                    self.viewer.scene.removeItem(item)
                self.viewer.deposit_items.clear()
                self.viewer.selected_item = None
                self.deposits.clear()
                
                for saved in data['deposits']:
                    contour = np.array(saved['contour'])
                    d = Deposit(
                        id=saved['id'], contour=contour,
                        x=0, y=0, width=0, height=0,
                        area=saved['area'], perimeter=cv2.arcLength(contour, True),
                        circularity=saved['circularity'], aspect_ratio=1.0,
                        centroid=(saved['x'], saved['y']),
                        merged=saved.get('merged', False),
                        group_id=saved.get('group_id', None)
                    )
                    d.label = saved['label']
                    self.deposits.append(d)
                    self.extractor.extract_features(self.image, [d])
                    self.viewer.add_deposit(d)
                
                self.next_id = max(d.id for d in self.deposits) + 1 if self.deposits else 0
                
                # Update next_group_id based on loaded deposits
                group_ids = [d.group_id for d in self.deposits if d.group_id is not None]
                if group_ids:
                    self.next_group_id = max(max(group_ids) + 1, self.next_group_id)
            else:
                for saved in data['deposits']:
                    for item in self.viewer.deposit_items:
                        d = item.deposit
                        if abs(d.centroid[0] - saved['x']) < 5 and abs(d.centroid[1] - saved['y']) < 5:
                            item.set_label(saved['label'])
                            break
            
            self._update_table()
            self._update_stats()
            self.statusBar().showMessage(f"Loaded from {path}")
    
    def _export_training_data(self):
        if not self.deposits:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
        folder = Path(folder)
        for label in ['rod', 'normal', 'artifact']:
            (folder / label).mkdir(exist_ok=True)
        from PIL import Image
        count = 0
        for d in self.deposits:
            if d.label in ['rod', 'normal', 'artifact']:
                patch = d.get_patch(self.image, padding=10)
                filename = f"{self.current_file.stem}_{d.id}.png"
                Image.fromarray(patch).save(folder / d.label / filename)
                count += 1
        self.statusBar().showMessage(f"Exported {count} patches to {folder}")
    
    # =========================================================================
    # EDIT_MODE specific methods
    # =========================================================================
    
    def _load_edit_data(self):
        """Load data from edit_data dict (for EDIT_MODE)."""
        from PIL import Image
        import pandas as pd
        
        image_path = self.edit_data.get('image_path')
        if image_path and Path(image_path).exists():
            self.image = np.array(Image.open(image_path))
            self.current_file = Path(image_path)
            self.viewer.load_image(self.image)
        
        # Load deposits from DataFrame
        deposits_df = self.edit_data.get('deposits_df')
        contour_data = self.edit_data.get('contour_data', {})
        self.next_group_id = self.edit_data.get('next_group_id', 1)
        
        if deposits_df is not None:
            self.deposits = []
            for idx, row in deposits_df.iterrows():
                dep_id = int(row.get('id', idx))
                info = contour_data.get(dep_id, {})
                
                # Get contour
                contour = info.get('contour', []) if isinstance(info, dict) else info
                if not contour:
                    # Create approximate contour from bounding box
                    x, y = int(row.get('x', 0)), int(row.get('y', 0))
                    w, h = int(row.get('width', 20)), int(row.get('height', 20))
                    contour = [[x-w//2, y-h//2], [x+w//2, y-h//2], 
                              [x+w//2, y+h//2], [x-w//2, y+h//2]]
                
                contour_np = np.array(contour)
                
                # Create Deposit object
                area = float(row.get('area', row.get('area_px', 0)))
                perimeter = cv2.arcLength(contour_np, True) if len(contour_np) > 2 else 0
                circularity = float(row.get('circularity', 0))
                
                x, y, bw, bh = cv2.boundingRect(contour_np) if len(contour_np) > 2 else (0, 0, 20, 20)
                
                deposit = Deposit(
                    id=dep_id,
                    contour=contour_np,
                    x=x, y=y,
                    width=bw, height=bh,
                    area=area,
                    perimeter=perimeter,
                    circularity=circularity,
                    aspect_ratio=max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1,
                    centroid=(int(row.get('x', x + bw//2)), int(row.get('y', y + bh//2))),
                    label=row.get('label', 'unknown'),
                    confidence=float(row.get('confidence', 1.0)),
                    merged=info.get('merged', False) if isinstance(info, dict) else False,
                    group_id=info.get('group_id', None) if isinstance(info, dict) else None
                )
                
                # Add color features if available
                deposit.mean_hue = float(row.get('mean_hue', 0))
                deposit.mean_saturation = float(row.get('mean_saturation', 0))
                deposit.mean_lightness = float(row.get('mean_lightness', 0))
                
                self.deposits.append(deposit)
            
            # Update next_id
            if self.deposits:
                self.next_id = max(d.id for d in self.deposits) + 1
            
            # Add to viewer
            for d in self.deposits:
                item = DepositGraphicsItem(d)
                self.viewer.scene.addItem(item)
                self.viewer.deposit_items.append(item)
            
            self._update_table()
            self._update_stats()
    
    def _save_edit_changes(self):
        """Save changes in EDIT_MODE (updates CSV + JSON + film_summary)."""
        if not self._has_unsaved_changes:
            self.statusBar().showMessage("No changes to save")
            return
        
        import pandas as pd
        
        output_dir = Path(self.edit_data.get('output_dir', ''))
        filename = self.edit_data.get('filename', '')
        stem = Path(filename).stem
        
        if not output_dir.exists():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Output directory not found: {output_dir}")
            return
        
        try:
            # 1. Update individual CSV
            deposits_dir = output_dir / 'deposits'
            if deposits_dir.exists():
                csv_path = deposits_dir / f"{stem}_deposits.csv"
                
                # Create DataFrame from deposits
                rows = []
                for d in self.deposits:
                    rows.append({
                        'id': d.id,
                        'filename': filename,
                        'x': d.centroid[0],
                        'y': d.centroid[1],
                        'width': d.width,
                        'height': d.height,
                        'area_px': d.area,
                        'circularity': d.circularity,
                        'aspect_ratio': d.aspect_ratio,
                        'mean_hue': d.mean_hue,
                        'mean_saturation': d.mean_saturation,
                        'mean_lightness': d.mean_lightness,
                        'label': d.label,
                        'confidence': d.confidence
                    })
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                
                # 2. Update JSON (unified format)
                json_path = deposits_dir / f"{stem}.labels.json"
                deposits_data = []
                for d in self.deposits:
                    deposits_data.append({
                        'id': d.id,
                        'contour': d.contour.tolist(),
                        'x': d.centroid[0],
                        'y': d.centroid[1],
                        'width': d.width,
                        'height': d.height,
                        'area': d.area,
                        'circularity': d.circularity,
                        'label': d.label,
                        'confidence': d.confidence,
                        'merged': d.merged,
                        'group_id': d.group_id
                    })
                
                with open(json_path, 'w') as f:
                    json.dump({
                        'image_file': filename,
                        'next_group_id': self.next_group_id,
                        'deposits': deposits_data
                    }, f, indent=2)
            
            # 3. Update all_deposits.csv
            all_deposits_path = output_dir / 'all_deposits.csv'
            if all_deposits_path.exists():
                all_df = pd.read_csv(all_deposits_path)
                all_df = all_df[all_df['filename'] != filename]
                all_df = pd.concat([all_df, df], ignore_index=True)
                all_df.to_csv(all_deposits_path, index=False)
            
            # 4. Update film_summary.csv
            summary_path = output_dir / 'image_summary.csv'
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                
                labels = [d.label for d in self.deposits]
                n_normal = labels.count('normal')
                n_rod = labels.count('rod')
                n_artifact = labels.count('artifact')
                n_total = n_normal + n_rod
                rod_fraction = n_rod / n_total if n_total > 0 else 0
                
                mask = summary_df['filename'] == filename
                if mask.any():
                    summary_df.loc[mask, 'n_normal'] = n_normal
                    summary_df.loc[mask, 'n_rod'] = n_rod
                    summary_df.loc[mask, 'n_artifact'] = n_artifact
                    summary_df.loc[mask, 'n_total'] = n_total
                    summary_df.loc[mask, 'rod_fraction'] = rod_fraction
                    summary_df.to_csv(summary_path, index=False)
            
            self._has_unsaved_changes = False
            self.data_saved.emit()  # Notify parent to refresh
            self.statusBar().showMessage("Changes saved successfully!")
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def _discard_and_close(self):
        """Discard changes and close (EDIT_MODE)."""
        if self._has_unsaved_changes:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "Discard Changes?",
                "You have unsaved changes. Discard them?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self._has_unsaved_changes = False
        self.close()


def run_labeling_gui():
    app = QApplication(sys.argv)
    
    # Load custom fonts (Noto Sans)
    load_custom_fonts()
    
    # Set application-wide font
    app_font = QFont("Noto Sans", 10)
    if not QFontDatabase.hasFamily("Noto Sans"):
        app_font = QFont("Segoe UI", 10)  # Windows fallback
    app.setFont(app_font)
    
    window = LabelingWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_labeling_gui()
