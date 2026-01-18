"""
Common UI components shared between main_gui and labeling_gui.
Consolidates Theme, custom widgets, and utility functions.
"""

from pathlib import Path

from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox, QTableWidgetItem
from PySide6.QtGui import QWheelEvent, QFontDatabase
from PySide6.QtCore import Qt


# =============================================================================
# Theme Colors - Dark Theme with Coral Accent (DIC2497 #DA4E42)
# =============================================================================
class Theme:
    """SCAT Application Color Theme - Dark with Coral Accent (minimal use)"""
    
    # Primary accent - DIC2497 (minimal use for accents only)
    PRIMARY = "#DA4E42"        # Coral red - selected tabs, primary buttons only
    PRIMARY_DARK = "#C44539"
    PRIMARY_LIGHT = "#E8695E"
    
    # Secondary - DIC540 (general UI elements)
    SECONDARY = "#636867"      # Gray-green
    SECONDARY_DARK = "#525756"
    SECONDARY_LIGHT = "#7A7F7E"
    
    # Semantic colors for deposits
    NORMAL = "#4CAF50"         # Green - normal deposits
    NORMAL_DARK = "#388E3C"
    
    ROD = "#DA4E42"            # Same as primary - RODs  
    ROD_DARK = "#C44539"
    
    ARTIFACT = "#636867"       # Same as secondary - artifacts
    ARTIFACT_DARK = "#525756"
    
    # Background layers (very dark, almost black)
    BG_DARKEST = "#0A0A0A"     # Main window background
    BG_DARK = "#121212"        # Card/group background
    BG_MEDIUM = "#101010"      # Input field background
    BG_LIGHT = "#242424"       # Hover state
    BG_LIGHTER = "#2E2E2E"     # Active/pressed state
    
    # Text
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#9A9A9A"
    TEXT_MUTED = "#5A5A5A"
    
    # Borders
    BORDER = "#2A2A2A"
    BORDER_FOCUS = "#DA4E42"   # Focus uses primary
    
    # Cached stylesheet
    _cached_app_stylesheet = None
    _cached_labeling_stylesheet = None
    
    @staticmethod
    def button_style(bg_color: str, text_color: str = "#FFFFFF", hover_color: str = None) -> str:
        """Generate button stylesheet with specific colors."""
        if hover_color is None:
            hover_color = Theme.BG_LIGHTER
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_LIGHT};
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: #404040;
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
                border-radius: 5px;
                background-color: {cls.BG_DARKEST};
                margin-top: -1px;
            }}
            QTabBar::tab {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_SECONDARY};
                padding: 12px 28px;
                margin-right: 3px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {cls.PRIMARY};
                color: {cls.TEXT_PRIMARY};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_PRIMARY};
            }}
            
            /* Group Box */
            QGroupBox {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: 8px;
                margin-top: 20px;
                padding: 20px 12px 12px 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 6px;
                padding: 2px 10px;
                color: {cls.PRIMARY};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(10, 10, 10, 255),
                    stop:0.55 rgba(10, 10, 10, 255),
                    stop:0.65 rgba(18, 18, 18, 255),
                    stop:1 rgba(18, 18, 18, 255));
                font-size: 13px;
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
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {cls.SECONDARY};
                border-color: {cls.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {cls.SECONDARY_DARK};
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: #404040;
            }}
            
            /* Input Fields - background matches surroundings */
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: 5px;
                padding: 8px 10px;
                color: {cls.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {cls.SECONDARY};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {cls.PRIMARY};
            }}
            
            /* ComboBox - separate styling */
            QComboBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: 5px;
                padding: 8px 10px;
                color: {cls.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {cls.SECONDARY};
            }}
            QComboBox:focus {{
                border-color: {cls.PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 12px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {cls.TEXT_SECONDARY};
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
            
            /* Labels - no background, bold for form labels */
            QLabel {{
                color: {cls.TEXT_PRIMARY};
                background-color: transparent;
                padding: 0px;
                font-weight: bold;
            }}
            
            /* Tables */
            QTableWidget {{
                background-color: {cls.BG_DARK};
                gridline-color: {cls.BORDER};
                border: 1px solid {cls.BORDER};
                border-radius: 5px;
            }}
            QTableWidget::item {{
                padding: 8px;
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
                font-weight: bold;
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
                border-radius: 5px;
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
                border-radius: 5px;
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
                background-color: {cls.BG_LIGHTER};
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
                background-color: {cls.BG_LIGHTER};
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
                border-radius: 5px;
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
                border-radius: 5px;
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
            }}
            QCheckBox::indicator:hover {{
                border-color: {cls.PRIMARY_LIGHT};
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
                border-radius: 6px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 10px 24px;
                border-radius: 4px;
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
                border-radius: 4px;
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
                font-weight: bold;
                border: 1px solid {cls.BORDER};
                border-radius: 8px;
                margin-top: 20px;
                padding: 20px 12px 12px 12px;
                background-color: {cls.BG_DARK};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 6px;
                padding: 2px 10px;
                color: {cls.PRIMARY};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(10, 10, 10, 255),
                    stop:0.55 rgba(10, 10, 10, 255),
                    stop:0.65 rgba(18, 18, 18, 255),
                    stop:1 rgba(18, 18, 18, 255));
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {cls.BG_LIGHT};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 24px;
                color: {cls.TEXT_PRIMARY};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {cls.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {cls.PRIMARY};
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
                color: {cls.TEXT_PRIMARY};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {cls.PRIMARY};
            }}
            
            /* Table Frame with rounded corners */
            QFrame#tableFrame {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                border-radius: 6px;
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
                font-weight: bold;
            }}
            QRadioButton {{
                color: {cls.TEXT_PRIMARY};
                spacing: 8px;
                padding: 6px 12px;
                border-radius: 4px;
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
                font-weight: bold;
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
                border-radius: 4px;
                padding: 6px 12px;
                color: {cls.TEXT_PRIMARY};
                font-weight: bold;
            }}
            QToolBar QToolButton:hover {{
                background-color: {cls.SECONDARY};
                border-color: {cls.BORDER};
            }}
            QToolBar QToolButton:pressed {{
                background-color: {cls.PRIMARY};
            }}
            QSplitter::handle {{
                background-color: {cls.BORDER};
            }}
            QComboBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
                color: {cls.TEXT_PRIMARY};
            }}
            QComboBox:focus {{
                border-color: {cls.PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {cls.TEXT_SECONDARY};
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
                border-radius: 3px;
                border: 2px solid {cls.BORDER};
                background-color: {cls.BG_DARK};
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls.PRIMARY};
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
                background-color: {cls.BG_LIGHTER};
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
                background-color: {cls.BG_LIGHTER};
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


# =============================================================================
# Utility Functions
# =============================================================================
def load_custom_fonts():
    """Load custom fonts bundled with the application."""
    fonts_dir = Path(__file__).parent / "resources" / "fonts"
    
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id < 0:
                print(f"Warning: Failed to load font {font_file.name}")


def get_icon_path() -> str:
    """Get the path to the application icon."""
    # Try multiple locations
    possible_paths = [
        Path(__file__).parent.parent / "scat" / "resources" / "scat.ico",
        Path(__file__).parent / "resources" / "scat.ico",
        Path(__file__).parent.parent / "resources" / "scat.ico",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return ""
