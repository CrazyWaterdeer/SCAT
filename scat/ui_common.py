"""
Common UI components shared between main_gui and labeling_gui.
Consolidates Theme, custom widgets, and utility functions.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QSpinBox, QDoubleSpinBox, QComboBox, QTableWidgetItem,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
)
from PySide6.QtGui import QWheelEvent, QFontDatabase
from PySide6.QtCore import Qt


# =============================================================================
# Theme Colors - Dark Theme with Coral Accent (DIC2497 #DA4E42)
# =============================================================================
class Theme:
    """SCAT Application Color Theme — dark, with a single coral accent (DIC2497).

    Token discipline (docs/superpowers/specs/2026-07-15-scat-design-elevation.md):
      * Surface tiers are role-named and monotonic in lightness
        (BG_BASE < BG_INSET < BG_SURFACE < BG_HOVER < BG_ACTIVE).
      * PRIMARY / SECONDARY are CHROME tokens — selection, focus, state — never a data marker.
      * NORMAL / ROD / ARTIFACT are SEMANTIC data colors, byte-identical to the report tokens.
    """

    # ---- Chrome accents (UI state only — never a data marker) ----
    PRIMARY = "#DA4E42"        # Coral (DIC2497) — selected tab, focus ring, primary action
    PRIMARY_DARK = "#C44539"
    PRIMARY_LIGHT = "#E8695E"

    SECONDARY = "#636867"      # Gray-green (DIC540) — neutral hover / secondary
    SECONDARY_DARK = "#4E5251" # pressed feedback
    SECONDARY_LIGHT = "#7A7F7E"

    # ---- Semantic deposit colors (IDENTICAL to report --normal/--rod/--artifact) ----
    NORMAL = "#1F9E77"         # Teal — color-blind-safe, ties the pH-axis mid, separates from ROD
    NORMAL_DARK = "#177A5C"
    ROD = "#DA4E42"            # Coral — RODs (matches report --rod)
    ROD_DARK = "#C44539"
    ARTIFACT = "#636867"       # Gray-green — artifacts (matches report --artifact)
    ARTIFACT_DARK = "#4E5251"

    # ---- Surface tiers (role-named, monotonic in lightness) ----
    BG_BASE    = "#0A0A0A"     # window
    BG_INSET   = "#0E0E0E"     # input fields, table alternate rows
    BG_SURFACE = "#171717"     # group boxes / cards — a visible step above BASE
    BG_HOVER   = "#242424"     # hover
    BG_ACTIVE  = "#2E2E2E"     # active / pressed
    # Backwards-compatible aliases (old names → new tiers). Prefer the role names above.
    BG_DARKEST = BG_BASE
    BG_MEDIUM  = BG_INSET
    BG_DARK    = BG_SURFACE
    BG_LIGHT   = BG_HOVER
    BG_LIGHTER = BG_ACTIVE

    # ---- Text (contrast-checked; see spec §6) ----
    TEXT_PRIMARY   = "#EDEDED"  # near-white — softer than pure #FFF
    TEXT_SECONDARY = "#9A9A9A"  # the one muted/secondary token (6.66:1 on BASE)
    TEXT_MUTED     = "#8A8A8A"  # was #5A5A5A (FAIL) → 5.43:1 on SURFACE
    TEXT_DISABLED  = "#6E6E6E"  # disabled labels — 3.27:1 on #1E1E1E (was #404040, 1.61:1)

    # ---- Borders / focus ----
    BORDER = "#2A2A2A"
    BORDER_FOCUS = "#DA4E42"   # focus ring (rendered 2px — 4.67:1 on the inset field)
    FOCUS_RING = "#DA4E42"

    # ---- Scales (spacing 4/8 · radius · weight · font-size) ----
    RADIUS_CONTROL = 6         # button, input, combo, checkbox, menu item, tooltip
    RADIUS_CONTAINER = 8       # group box, tab, table, list, tree, textedit
    RADIUS_PILL = 17           # chat send button (34px control → full round)
    SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5, SPACE_6 = 4, 8, 12, 16, 20, 24
    WEIGHT_BODY, WEIGHT_LABEL, WEIGHT_TITLE = 400, 500, 600
    FS_XS, FS_SM, FS_BODY, FS_TITLE, FS_DISPLAY = 11, 12, 13, 15, 24

    # Cached stylesheet
    _cached_app_stylesheet = None
    _cached_labeling_stylesheet = None
    
    @staticmethod
    def button_style(bg_color: str, text_color: str = "#FFFFFF",
                     hover_color: str = None, pressed_color: str = None) -> str:
        """Generate a filled action-button stylesheet with specific colors (run button, the
        semantic labeling buttons, …). A 2px focus ring is reserved as a transparent border so
        keyboard focus never reflows the button. Pass ``pressed_color`` (usually the color's
        own dark shade) so a coloured button darkens on press instead of flashing neutral gray."""
        if hover_color is None:
            hover_color = Theme.BG_ACTIVE
        if pressed_color is None:
            pressed_color = Theme.BG_HOVER
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: 2px solid transparent;
                padding: 8px 16px;
                border-radius: {Theme.RADIUS_CONTROL}px;
                font-weight: bold;
                font-size: {Theme.FS_BODY}px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
            QPushButton:focus {{
                border: 2px solid {Theme.FOCUS_RING};
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: {Theme.TEXT_DISABLED};
            }}
        """
    
    @classmethod
    def get_app_stylesheet(cls) -> str:
        """Return the complete application stylesheet for main_gui."""
        # Always regenerate (remove caching for debugging)
        cls._cached_app_stylesheet = f"""
            /* Main Window */
            QMainWindow {{
                background-color: {cls.BG_DARKEST};
                color: {cls.TEXT_PRIMARY};
            }}
            QDialog {{
                background-color: {cls.BG_DARKEST};
                color: {cls.TEXT_PRIMARY};
            }}
            
            /* Scroll content area - dark background */
            QWidget#scrollContent {{
                background-color: {cls.BG_DARKEST};
            }}
            
            /* Tab Widget */
            QTabWidget::pane {{
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
                background-color: {cls.BG_DARKEST};
                margin-top: -1px;
            }}
            QTabBar::tab {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_SECONDARY};
                padding: 12px 24px;
                margin-right: 3px;
                border-top-left-radius: {cls.RADIUS_CONTAINER}px;
                border-top-right-radius: {cls.RADIUS_CONTAINER}px;
                font-weight: {cls.WEIGHT_TITLE};
            }}
            QTabBar::tab:selected {{
                background-color: {cls.PRIMARY};
                color: #FFFFFF;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
            }}
            
            /* Group Box — a calm card with a small MUTED section label above it (not a loud
               coral heading). Coral is reserved for actions/active state. */
            QGroupBox {{
                background-color: {cls.BG_SURFACE};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
                margin-top: 18px;
                padding: 18px 16px 16px 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 4px;
                top: 0px;
                padding: 0px 2px;
                color: {cls.TEXT_SECONDARY};
                font-size: {cls.FS_XS}px;
                font-weight: {cls.WEIGHT_TITLE};
                letter-spacing: 1px;
            }}

            /* Scroll Area */
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}

            /* Buttons - Secondary (gray) by default with visible background */
            QPushButton {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                padding: 8px 16px;
                border-radius: {cls.RADIUS_CONTROL}px;
                font-weight: {cls.WEIGHT_TITLE};
            }}
            QPushButton:hover {{
                background-color: {cls.SECONDARY};
                border-color: {cls.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {cls.SECONDARY_DARK};
            }}
            QPushButton:focus {{
                border: 2px solid {cls.FOCUS_RING};
                padding: 7px 15px;
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: {cls.TEXT_DISABLED};
                border-color: {cls.BORDER};
            }}
            
            /* Input Fields - background matches surroundings */
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTROL}px;
                padding: 8px 10px;
                color: {cls.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {cls.SECONDARY};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 2px solid {cls.PRIMARY};
                padding: 7px 9px;
            }}

            /* ComboBox - separate styling */
            QComboBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTROL}px;
                padding: 8px 28px 8px 10px;
                color: {cls.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {cls.SECONDARY};
            }}
            QComboBox:focus {{
                border: 2px solid {cls.PRIMARY};
                padding: 7px 27px 7px 9px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 22px;
                border: none;
                border-left: 1px solid {cls.BORDER};
                border-top-right-radius: {cls.RADIUS_CONTROL}px;
                border-bottom-right-radius: {cls.RADIUS_CONTROL}px;
                background-color: transparent;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                image: url("{qss_icon('caret-down.svg')}");
            }}
            QComboBox:hover::down-arrow, QComboBox:focus::down-arrow, QComboBox:on::down-arrow {{
                image: url("{qss_icon('caret-down-hover.svg')}");
            }}
            QComboBox QAbstractItemView {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                selection-background-color: {cls.SECONDARY};
                padding: 4px;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 10px;
                min-height: 24px;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            
            /* Labels - body weight by default; emphasis comes from titles/section headers */
            QLabel {{
                color: {cls.TEXT_PRIMARY};
                background-color: transparent;
                padding: 0px;
                font-weight: {cls.WEIGHT_BODY};
            }}

            /* Tables */
            QTableWidget {{
                background-color: {cls.BG_DARK};
                alternate-background-color: {cls.BG_INSET};
                gridline-color: {cls.BORDER};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
            QTableWidget::item:hover {{
                background-color: #1A1A1A;
            }}
            QTableWidget::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            QHeaderView::section {{
                background-color: {cls.BG_DARK};
                color: {cls.TEXT_PRIMARY};
                padding: 10px 8px;
                border: none;
                border-bottom: 1px solid {cls.BORDER};
                font-weight: {cls.WEIGHT_TITLE};
            }}
            QTableCornerButton::section {{
                background-color: {cls.BG_DARK};
                border: none;
                border-bottom: 1px solid {cls.BORDER};
            }}
            
            /* List Widget */
            QListWidget {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
            }}
            QListWidget::item {{
                padding: 10px;
            }}
            QListWidget::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {cls.BG_LIGHT};
            }}
            
            /* Tree Widget */
            QTreeWidget {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
            }}
            QTreeWidget::item {{
                padding: 4px 8px;
            }}
            QTreeWidget::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            QTreeWidget::item:hover:!selected {{
                background-color: {cls.BG_LIGHT};
            }}
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {{
                border-image: none;
                image: url(none);
            }}
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {{
                border-image: none;
                image: url(none);
            }}
            
            /* ScrollBar */
            QScrollBar:vertical {{
                background-color: {cls.BG_DARK};
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #484848;
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {cls.SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {cls.BG_DARK};
                height: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: #484848;
                border-radius: 5px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {cls.SECONDARY};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* Progress Bar */
            QProgressBar {{
                background-color: {cls.BG_MEDIUM};
                border-radius: {cls.RADIUS_CONTROL}px;
                text-align: center;
                color: {cls.TEXT_PRIMARY};
                min-height: 22px;
                border: 1px solid {cls.BORDER};
            }}
            QProgressBar::chunk {{
                background-color: {cls.SECONDARY};
                border-radius: 4px;
            }}
            
            /* Text Edit */
            QTextEdit {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
                padding: 10px;
            }}
            
            /* CheckBox - larger with more spacing */
            QCheckBox {{
                spacing: 10px;
                color: {cls.TEXT_PRIMARY};
                padding: 6px 0px;
                min-height: 26px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {cls.BORDER};
                background-color: {cls.BG_DARK};
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls.PRIMARY};
                border-color: {cls.PRIMARY};
                image: url("{qss_icon('check.svg')}");
            }}
            QCheckBox::indicator:hover {{
                border-color: {cls.PRIMARY_LIGHT};
            }}
            QCheckBox:focus::indicator {{
                border-color: {cls.PRIMARY};
            }}

            /* RadioButton - same style as CheckBox */
            QRadioButton {{
                spacing: 10px;
                color: {cls.TEXT_PRIMARY};
                padding: 6px 0px;
                min-height: 26px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid {cls.BORDER};
                background-color: {cls.BG_DARK};
            }}
            QRadioButton::indicator:checked {{
                background-color: {cls.PRIMARY};
                border-color: {cls.PRIMARY};
            }}
            QRadioButton::indicator:hover {{
                border-color: {cls.PRIMARY_LIGHT};
            }}
            QRadioButton:focus::indicator {{
                border-color: {cls.PRIMARY};
            }}

            /* Scroll Area */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            /* Splitter */
            QSplitter::handle {{
                background-color: {cls.BORDER};
            }}
            QSplitter::handle:hover {{
                background-color: {cls.SECONDARY};
            }}
            
            /* Menu */
            QMenu {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 24px;
                border-radius: {cls.RADIUS_CONTROL}px;
            }}
            QMenu::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            
            /* ToolTip */
            QToolTip {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER};
                padding: 8px;
                border-radius: {cls.RADIUS_CONTROL}px;
            }}
        """
        return cls._cached_app_stylesheet
    
    @classmethod
    def get_labeling_stylesheet(cls) -> str:
        """Return the stylesheet for labeling_gui (simplified version)."""
        if cls._cached_labeling_stylesheet is None:
            cls._cached_labeling_stylesheet = f"""
            QMainWindow, QWidget {{
                background-color: {cls.BG_DARKEST};
                color: {cls.TEXT_PRIMARY};
            }}
            QGroupBox {{
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
                margin-top: 18px;
                padding: 18px 16px 16px 16px;
                background-color: {cls.BG_SURFACE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 4px;
                top: 0px;
                padding: 0px 2px;
                color: {cls.TEXT_SECONDARY};
                font-size: {cls.FS_XS}px;
                font-weight: {cls.WEIGHT_TITLE};
                letter-spacing: 1px;
            }}
            QPushButton {{
                background-color: {cls.BG_LIGHT};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTROL}px;
                padding: 6px 12px;
                min-height: 28px;
                color: {cls.TEXT_PRIMARY};
                font-weight: {cls.WEIGHT_TITLE};
            }}
            QPushButton:hover {{
                background-color: {cls.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {cls.SECONDARY_DARK};
            }}
            QPushButton:focus {{
                border: 2px solid {cls.FOCUS_RING};
                padding: 5px 11px;
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: {cls.TEXT_DISABLED};
                border-color: {cls.BORDER};
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTROL}px;
                padding: 4px 8px;
                min-height: 20px;
                color: {cls.TEXT_PRIMARY};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 2px solid {cls.PRIMARY};
                padding: 3px 7px;
            }}

            /* Table Frame with rounded corners */
            QFrame#tableFrame {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTAINER}px;
            }}
            QFrame#tableFrame QTableWidget {{
                background-color: transparent;
                border: none;
                gridline-color: {cls.BORDER};
                color: {cls.TEXT_PRIMARY};
            }}
            QTableWidget {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                gridline-color: {cls.BORDER};
                color: {cls.TEXT_PRIMARY};
            }}
            QTableWidget QTableCornerButton::section {{
                background-color: {cls.BG_DARK};
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            QHeaderView {{
                background-color: transparent;
            }}
            QHeaderView::section {{
                background-color: {cls.BG_DARK};
                color: {cls.TEXT_PRIMARY};
                padding: 6px;
                border: none;
                border-bottom: 1px solid {cls.BORDER};
                font-weight: {cls.WEIGHT_TITLE};
            }}
            QRadioButton {{
                color: {cls.TEXT_PRIMARY};
                spacing: 8px;
                padding: 6px 12px;
                border-radius: {cls.RADIUS_CONTROL}px;
                background-color: {cls.BG_DARK};
            }}
            QRadioButton:hover {{
                border: 1px solid {cls.SECONDARY};
                background-color: {cls.BG_LIGHT};
            }}
            QRadioButton:checked {{
                background-color: {cls.PRIMARY};
                color: white;
                font-weight: bold;
            }}
            QRadioButton::indicator {{
                width: 0px;
                height: 0px;
            }}
            QLabel {{
                color: {cls.TEXT_PRIMARY};
                font-weight: {cls.WEIGHT_BODY};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QToolBar {{
                background-color: {cls.BG_DARK};
                border-bottom: 1px solid {cls.BORDER};
                spacing: 4px;
                padding: 4px;
            }}
            QToolBar QToolButton {{
                background-color: {cls.BG_DARK};
                border: 1px solid transparent;
                border-radius: {cls.RADIUS_CONTROL}px;
                padding: 6px 12px;
                color: {cls.TEXT_PRIMARY};
                font-weight: {cls.WEIGHT_TITLE};
            }}
            QToolBar QToolButton:hover {{
                background-color: {cls.SECONDARY};
                border-color: {cls.BORDER};
            }}
            QToolBar QToolButton:pressed {{
                background-color: {cls.SECONDARY_DARK};
            }}
            QToolBar QToolButton:checked {{
                background-color: {cls.PRIMARY};
                color: #FFFFFF;
            }}
            QSplitter::handle {{
                background-color: {cls.BORDER};
            }}
            QComboBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_CONTROL}px;
                padding: 4px 24px 4px 8px;
                min-height: 20px;
                color: {cls.TEXT_PRIMARY};
            }}
            QComboBox:focus {{
                border: 2px solid {cls.PRIMARY};
                padding: 3px 23px 3px 7px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 20px;
                border: none;
                border-left: 1px solid {cls.BORDER};
                background-color: transparent;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                image: url("{qss_icon('caret-down.svg')}");
            }}
            QComboBox:hover::down-arrow, QComboBox:focus::down-arrow, QComboBox:on::down-arrow {{
                image: url("{qss_icon('caret-down-hover.svg')}");
            }}
            QComboBox QAbstractItemView {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                selection-background-color: {cls.SECONDARY};
            }}
            QCheckBox {{
                spacing: 8px;
                color: {cls.TEXT_PRIMARY};
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 2px solid {cls.BORDER};
                background-color: {cls.BG_DARK};
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls.PRIMARY};
                border-color: {cls.PRIMARY};
                image: url("{qss_icon('check.svg')}");
            }}
            QCheckBox:focus::indicator {{
                border-color: {cls.PRIMARY};
            }}
            
            /* ScrollBar */
            QScrollBar:vertical {{
                background-color: {cls.BG_DARK};
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #484848;
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {cls.SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {cls.BG_DARK};
                height: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: #484848;
                border-radius: 5px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {cls.SECONDARY};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """
        return cls._cached_labeling_stylesheet


# =============================================================================
# Custom Widgets - SpinBox/ComboBox without scroll wheel
# =============================================================================
class NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores mouse wheel events to prevent accidental changes."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()  # Pass to parent for scrolling


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores mouse wheel events."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores mouse wheel events and always drops down below."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Force popup to always appear below the combobox
        self.view().window().setWindowFlags(
            Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint
        )
    
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()
    
    def showPopup(self):
        """Override to ensure dropdown always appears below."""
        # Call parent showPopup first
        super().showPopup()
        
        # Get the popup and move it below the combobox
        popup = self.view().window()
        pos = self.mapToGlobal(self.rect().bottomLeft())
        popup.move(pos)


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


class CenteredCap(QWidget):
    """A widget that horizontally centers a width-capped content column. Add content to
    ``.content_layout`` (a QVBoxLayout). Used so single-column forms don't stretch across a
    16:9 / 16:10 display — the column caps at ``max_width`` and is centered with side gutters."""

    def __init__(self, max_width: int = 1080, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addStretch(1)
        self._inner = QWidget()
        self._inner.setMaximumWidth(max_width)
        self.content_layout = QVBoxLayout(self._inner)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        row.addWidget(self._inner, 0)
        row.addStretch(1)

    def add_widget(self, w):
        self.content_layout.addWidget(w)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class CollapsibleSection(QWidget):
    """A titled section whose body shows/hides via a header toggle — progressive disclosure
    for advanced settings. The header is a full-width bar with a rotating chevron; the body
    animates open/closed (height), honoring reduced motion. Add content with ``add_widget`` /
    ``add_layout`` or by laying out on ``.body``."""

    _MAX = 16777215  # Qt's QWIDGETSIZE_MAX

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = QPushButton()
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 9px 12px;
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_CONTROL}px;
                background-color: {Theme.BG_INSET};
                color: {Theme.TEXT_SECONDARY};
                font-weight: {Theme.WEIGHT_TITLE};
            }}
            QPushButton:hover {{ background-color: {Theme.BG_HOVER}; color: {Theme.TEXT_PRIMARY}; }}
            QPushButton:checked {{ color: {Theme.TEXT_PRIMARY}; }}
            QPushButton:focus {{ border: 2px solid {Theme.FOCUS_RING}; padding: 8px 11px; }}
        """)
        self.header.clicked.connect(self._toggle)
        root.addWidget(self.header)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 8, 4, 4)
        self.body_layout.setSpacing(6)
        root.addWidget(self.body)
        self.body.setVisible(expanded)
        self._update_header()

    # -- content API --
    def add_widget(self, w):
        self.body_layout.addWidget(w)

    def add_layout(self, layout):
        self.body_layout.addLayout(layout)

    # -- behaviour --
    def _update_header(self):
        chevron = "▾" if self._expanded else "▸"   # ▾ / ▸
        self.header.setText(f"{chevron}   {self._title}")

    def _toggle(self):
        from scat import ui_motion
        self._expanded = self.header.isChecked()
        self._update_header()
        if self._expanded:
            self.body.setVisible(True)
            target = self.body.sizeHint().height()
            anim = ui_motion.animate(self.body, b"maximumHeight", target,
                                     ui_motion.DUR_TAB, ui_motion.CURVE_OUT, start=0)
            # After opening, lift the cap so the body can grow with its content.
            anim.finished.connect(lambda: self.body.setMaximumHeight(self._MAX))
        else:
            start = self.body.height()
            anim = ui_motion.animate(self.body, b"maximumHeight", 0,
                                     ui_motion.DUR_TAB, ui_motion.CURVE_OUT, start=start)
            anim.finished.connect(lambda: self.body.setVisible(False))


# =============================================================================
# Utility Functions
# =============================================================================
def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to a bundled resource under scat/resources/.

    Args:
        relative_path: Path relative to scat/resources/ (e.g., 'fonts/NotoSans-Regular.ttf')
    """
    return Path(__file__).parent / 'resources' / relative_path


_icon_cache = {}


def icon(name: str, color: str = None, size: int = 20):
    """Load a bundled Material Symbol (``scat/resources/icons/ms_<name>.svg``) as a recolored
    ``QIcon`` — replaces emoji glyphs that render as tofu in the bundled font. Cached. These are
    Google Material Symbols (Apache-2.0). Returns an empty QIcon if the name is missing."""
    from PySide6.QtGui import QIcon, QPixmap, QPainter
    color = color or Theme.TEXT_PRIMARY
    key = (name, color, size)
    if key in _icon_cache:
        return _icon_cache[key]
    ic = QIcon()
    path = Path(__file__).parent / "resources" / "icons" / f"ms_{name}.svg"
    if path.exists():
        try:
            from PySide6.QtSvg import QSvgRenderer
            from PySide6.QtCore import Qt, QByteArray
            svg = path.read_text().replace("<svg ", f'<svg fill="{color}" ', 1)
            r = QSvgRenderer(QByteArray(svg.encode()))
            pm = QPixmap(size, size)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            r.render(p)
            p.end()
            ic = QIcon(pm)
        except Exception:
            ic = QIcon()
    _icon_cache[key] = ic
    return ic


def qss_icon(name: str) -> str:
    """Absolute POSIX path to a bundled QSS icon (``scat/resources/icons/<name>``), for use in
    a QSS ``image: url(...)``. Forward slashes so Qt accepts it on every platform. Qt's QSS
    engine does not support the CSS border-triangle caret trick or ``data:`` URIs (verified),
    so combobox carets and the checkbox glyph are shipped as real SVG files referenced here."""
    return (Path(__file__).parent / "resources" / "icons" / name).as_posix()


def load_custom_fonts():
    """Load custom fonts bundled under scat/resources/fonts/."""
    fonts_dir = Path(__file__).parent / "resources" / "fonts"
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id < 0:
                print(f"Warning: Failed to load font {font_file.name}")


def get_icon_path() -> str:
    """Get the path to the application icon (scat/resources/icon.ico)."""
    for path in (
        Path(__file__).parent / "resources" / "icon.ico",
        Path(__file__).parent.parent / "scat" / "resources" / "icon.ico",
        Path(__file__).parent.parent / "resources" / "icon.ico",
    ):
        if path.exists():
            return str(path)
    return ""
