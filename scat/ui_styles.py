"""QSS stylesheet builders for SCAT's Qt GUIs.

Split out of ui_common.Theme to keep that module focused on design tokens + widgets.
`theme` is the Theme class (the token holder); these builders read tokens off it and are
called — with caching — by Theme.get_app_stylesheet / Theme.get_labeling_stylesheet.
"""
from .ui_common import qss_icon


def build_app_stylesheet(theme) -> str:
    """The complete application stylesheet for main_gui."""
    return f"""
            /* Main Window */
            QMainWindow {{
                background-color: {theme.BG_DARKEST};
                color: {theme.TEXT_PRIMARY};
            }}
            QDialog {{
                background-color: {theme.BG_DARKEST};
                color: {theme.TEXT_PRIMARY};
            }}
            
            /* Scroll content area - dark background */
            QWidget#scrollContent {{
                background-color: {theme.BG_DARKEST};
            }}
            
            /* Tab Widget */
            QTabWidget::pane {{
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
                background-color: {theme.BG_DARKEST};
                margin-top: -1px;
            }}
            QTabBar::tab {{
                background-color: {theme.BG_MEDIUM};
                color: {theme.TEXT_SECONDARY};
                padding: 12px 24px;
                margin-right: 3px;
                border-top-left-radius: {theme.RADIUS_CONTAINER}px;
                border-top-right-radius: {theme.RADIUS_CONTAINER}px;
                font-weight: {theme.WEIGHT_TITLE};
            }}
            QTabBar::tab:selected {{
                background-color: {theme.PRIMARY};
                color: #FFFFFF;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {theme.BG_LIGHT};
                color: {theme.TEXT_PRIMARY};
            }}
            
            /* Group Box — a calm card with a small MUTED section label above it (not a loud
               coral heading). Coral is reserved for actions/active state. */
            QGroupBox {{
                background-color: {theme.BG_SURFACE};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
                margin-top: 18px;
                padding: 18px 16px 16px 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 4px;
                top: 0px;
                padding: 0px 2px;
                color: {theme.TEXT_SECONDARY};
                font-size: {theme.FS_XS}px;
                font-weight: {theme.WEIGHT_TITLE};
                letter-spacing: 1px;
            }}

            /* Scroll Area */
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}

            /* Buttons - Secondary (gray) by default with visible background */
            QPushButton {{
                background-color: {theme.BG_LIGHT};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER};
                padding: 8px 16px;
                border-radius: {theme.RADIUS_CONTROL}px;
                font-weight: {theme.WEIGHT_TITLE};
            }}
            QPushButton:hover {{
                background-color: {theme.SECONDARY};
                border-color: {theme.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {theme.SECONDARY_DARK};
            }}
            QPushButton:focus {{
                border: 2px solid {theme.FOCUS_RING};
                padding: 7px 15px;
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: {theme.TEXT_DISABLED};
                border-color: {theme.BORDER};
            }}
            /* A QPushButton with a dropdown menu (e.g. the top-bar "More") otherwise renders its
               text with the native palette (black) instead of the QSS color — styling the
               menu-indicator forces full-QSS rendering so `color` above applies. */
            QPushButton::menu-indicator {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                right: 8px;
                width: 10px;
            }}

            /* Input Fields - background matches surroundings */
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {theme.BG_MEDIUM};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTROL}px;
                padding: 8px 10px;
                color: {theme.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {theme.SECONDARY};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 2px solid {theme.PRIMARY};
                padding: 7px 9px;
            }}

            /* ComboBox - separate styling */
            QComboBox {{
                background-color: {theme.BG_MEDIUM};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTROL}px;
                padding: 8px 28px 8px 10px;
                color: {theme.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {theme.SECONDARY};
            }}
            QComboBox:focus {{
                border: 2px solid {theme.PRIMARY};
                padding: 7px 27px 7px 9px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border: none;
                background-color: transparent;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                subcontrol-position: center;
                image: url("{qss_icon('caret-down.svg')}");
            }}
            /* Ghost picker (Claude-style): reads as plain text, becomes a subtle button on
               hover, opens the list on click. Used for the chat model/provider selectors. */
            QComboBox#ghostPicker {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: {theme.RADIUS_CONTROL}px;
                padding: 5px 22px 5px 10px;
                color: {theme.TEXT_SECONDARY};
                min-height: 18px;
                font-weight: {theme.WEIGHT_TITLE};
            }}
            QComboBox#ghostPicker:hover {{
                background-color: {theme.BG_HOVER};
                color: {theme.TEXT_PRIMARY};
            }}
            QComboBox#ghostPicker:focus {{
                border: 1px solid transparent;
                background-color: {theme.BG_HOVER};
            }}
            QComboBox#ghostPicker::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 20px;
                border: none;
            }}
            QComboBox#ghostPicker::down-arrow {{
                width: 11px;
                height: 11px;
                subcontrol-position: center;
                image: url("{qss_icon('caret-down.svg')}");
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                selection-background-color: {theme.SECONDARY};
                padding: 4px;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 10px;
                min-height: 24px;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {theme.SECONDARY};
            }}
            
            /* Labels - body weight by default; emphasis comes from titles/section headers */
            QLabel {{
                color: {theme.TEXT_PRIMARY};
                background-color: transparent;
                padding: 0px;
                font-weight: {theme.WEIGHT_BODY};
            }}

            /* Tables */
            QTableWidget {{
                background-color: {theme.BG_DARK};
                alternate-background-color: {theme.BG_INSET};
                gridline-color: {theme.BORDER};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
            QTableWidget::item:hover {{
                background-color: #1A1A1A;
            }}
            QTableWidget::item:selected {{
                background-color: {theme.SECONDARY};
            }}
            QHeaderView::section {{
                background-color: {theme.BG_DARK};
                color: {theme.TEXT_PRIMARY};
                padding: 10px 8px;
                border: none;
                border-bottom: 1px solid {theme.BORDER};
                font-weight: {theme.WEIGHT_TITLE};
            }}
            QTableCornerButton::section {{
                background-color: {theme.BG_DARK};
                border: none;
                border-bottom: 1px solid {theme.BORDER};
            }}
            
            /* List Widget */
            QListWidget {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
            }}
            QListWidget::item {{
                padding: 10px;
            }}
            QListWidget::item:selected {{
                background-color: {theme.SECONDARY};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {theme.BG_LIGHT};
            }}
            
            /* Tree Widget */
            QTreeWidget {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
            }}
            QTreeWidget::item {{
                padding: 4px 8px;
            }}
            QTreeWidget::item:selected {{
                background-color: {theme.SECONDARY};
            }}
            QTreeWidget::item:hover:!selected {{
                background-color: {theme.BG_LIGHT};
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
                background-color: {theme.BG_DARK};
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
                background-color: {theme.SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {theme.BG_DARK};
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
                background-color: {theme.SECONDARY};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* Progress Bar */
            QProgressBar {{
                background-color: {theme.BG_MEDIUM};
                border-radius: {theme.RADIUS_CONTROL}px;
                text-align: center;
                color: {theme.TEXT_PRIMARY};
                min-height: 22px;
                border: 1px solid {theme.BORDER};
            }}
            QProgressBar::chunk {{
                background-color: {theme.SECONDARY};
                border-radius: 4px;
            }}
            
            /* Text Edit */
            QTextEdit {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
                padding: 10px;
            }}
            
            /* CheckBox - larger with more spacing */
            QCheckBox {{
                spacing: 10px;
                color: {theme.TEXT_PRIMARY};
                padding: 6px 0px;
                min-height: 26px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {theme.BORDER};
                background-color: {theme.BG_DARK};
            }}
            QCheckBox::indicator:checked {{
                background-color: {theme.PRIMARY};
                border-color: {theme.PRIMARY};
                image: url("{qss_icon('check.svg')}");
            }}
            QCheckBox::indicator:hover {{
                border-color: {theme.PRIMARY_LIGHT};
            }}
            QCheckBox:focus::indicator {{
                border-color: {theme.PRIMARY};
            }}

            /* RadioButton - same style as CheckBox */
            QRadioButton {{
                spacing: 10px;
                color: {theme.TEXT_PRIMARY};
                padding: 6px 0px;
                min-height: 26px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid {theme.BORDER};
                background-color: {theme.BG_DARK};
            }}
            QRadioButton::indicator:checked {{
                background-color: {theme.PRIMARY};
                border-color: {theme.PRIMARY};
            }}
            QRadioButton::indicator:hover {{
                border-color: {theme.PRIMARY_LIGHT};
            }}
            QRadioButton:focus::indicator {{
                border-color: {theme.PRIMARY};
            }}

            /* Scroll Area */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            /* Splitter */
            QSplitter::handle {{
                background-color: {theme.BORDER};
            }}
            QSplitter::handle:hover {{
                background-color: {theme.SECONDARY};
            }}
            
            /* Menu */
            QMenu {{
                background-color: {theme.BG_DARK};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
                padding: 6px;
            }}
            QMenu::item {{
                color: {theme.TEXT_PRIMARY};
                background-color: transparent;
                padding: 8px 24px;
                border-radius: {theme.RADIUS_CONTROL}px;
            }}
            QMenu::item:selected {{
                background-color: {theme.SECONDARY};
                color: {theme.TEXT_PRIMARY};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {theme.BORDER};
                margin: 6px 8px;
            }}
            
            /* ToolTip */
            QToolTip {{
                background-color: {theme.BG_MEDIUM};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER};
                padding: 8px;
                border-radius: {theme.RADIUS_CONTROL}px;
            }}

            /* Status bar — the outermost bottom strip (was default light chrome) */
            QStatusBar {{
                background-color: {theme.BG_BASE};
                color: {theme.TEXT_MUTED};
                border-top: 1px solid {theme.BORDER};
            }}
            QStatusBar::item {{
                border: none;
            }}

            /* Assistant dock — theme its title bar/frame (was default light chrome) */
            QDockWidget {{
                color: {theme.TEXT_PRIMARY};
            }}
            QDockWidget::title {{
                background-color: {theme.BG_BASE};
                color: {theme.TEXT_PRIMARY};
                text-align: left;
                padding: 8px 12px;
                border-bottom: 1px solid {theme.BORDER};
            }}
            QDockWidget::close-button, QDockWidget::float-button {{
                background: transparent;
                border: none;
                padding: 2px;
            }}
            QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
                background-color: {theme.BG_HOVER};
                border-radius: {theme.RADIUS_CONTROL}px;
            }}
        """


def build_labeling_stylesheet(theme) -> str:
    """The stylesheet for labeling_gui (simplified version)."""
    return f"""
            QMainWindow, QWidget {{
                background-color: {theme.BG_DARKEST};
                color: {theme.TEXT_PRIMARY};
            }}
            QGroupBox {{
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
                margin-top: 18px;
                padding: 18px 16px 16px 16px;
                background-color: {theme.BG_SURFACE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 4px;
                top: 0px;
                padding: 0px 2px;
                color: {theme.TEXT_SECONDARY};
                font-size: {theme.FS_XS}px;
                font-weight: {theme.WEIGHT_TITLE};
                letter-spacing: 1px;
            }}
            QPushButton {{
                background-color: {theme.BG_LIGHT};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTROL}px;
                padding: 6px 12px;
                min-height: 28px;
                color: {theme.TEXT_PRIMARY};
                font-weight: {theme.WEIGHT_TITLE};
            }}
            QPushButton:hover {{
                background-color: {theme.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {theme.SECONDARY_DARK};
            }}
            QPushButton:focus {{
                border: 2px solid {theme.FOCUS_RING};
                padding: 5px 11px;
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: {theme.TEXT_DISABLED};
                border-color: {theme.BORDER};
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTROL}px;
                padding: 4px 8px;
                min-height: 20px;
                color: {theme.TEXT_PRIMARY};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 2px solid {theme.PRIMARY};
                padding: 3px 7px;
            }}

            /* Table Frame with rounded corners */
            QFrame#tableFrame {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTAINER}px;
            }}
            QFrame#tableFrame QTableWidget {{
                background-color: transparent;
                border: none;
                gridline-color: {theme.BORDER};
                color: {theme.TEXT_PRIMARY};
            }}
            QTableWidget {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                gridline-color: {theme.BORDER};
                color: {theme.TEXT_PRIMARY};
            }}
            QTableWidget QTableCornerButton::section {{
                background-color: {theme.BG_DARK};
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: {theme.SECONDARY};
            }}
            QHeaderView {{
                background-color: transparent;
            }}
            QHeaderView::section {{
                background-color: {theme.BG_DARK};
                color: {theme.TEXT_PRIMARY};
                padding: 6px;
                border: none;
                border-bottom: 1px solid {theme.BORDER};
                font-weight: {theme.WEIGHT_TITLE};
            }}
            QRadioButton {{
                color: {theme.TEXT_PRIMARY};
                spacing: 8px;
                padding: 6px 12px;
                border-radius: {theme.RADIUS_CONTROL}px;
                background-color: {theme.BG_DARK};
            }}
            QRadioButton:hover {{
                border: 1px solid {theme.SECONDARY};
                background-color: {theme.BG_LIGHT};
            }}
            QRadioButton:checked {{
                background-color: {theme.PRIMARY};
                color: white;
                font-weight: bold;
            }}
            QRadioButton::indicator {{
                width: 0px;
                height: 0px;
            }}
            QLabel {{
                color: {theme.TEXT_PRIMARY};
                font-weight: {theme.WEIGHT_BODY};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QToolBar {{
                background-color: {theme.BG_DARK};
                border-bottom: 1px solid {theme.BORDER};
                spacing: 4px;
                padding: 4px;
            }}
            QToolBar QToolButton {{
                background-color: {theme.BG_DARK};
                border: 1px solid transparent;
                border-radius: {theme.RADIUS_CONTROL}px;
                padding: 6px 12px;
                color: {theme.TEXT_PRIMARY};
                font-weight: {theme.WEIGHT_TITLE};
            }}
            QToolBar QToolButton:hover {{
                background-color: {theme.SECONDARY};
                border-color: {theme.BORDER};
            }}
            QToolBar QToolButton:pressed {{
                background-color: {theme.SECONDARY_DARK};
            }}
            QToolBar QToolButton:checked {{
                background-color: {theme.PRIMARY};
                color: #FFFFFF;
            }}
            QSplitter::handle {{
                background-color: {theme.BORDER};
            }}
            QComboBox {{
                background-color: {theme.BG_MEDIUM};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_CONTROL}px;
                padding: 4px 24px 4px 8px;
                min-height: 20px;
                color: {theme.TEXT_PRIMARY};
            }}
            QComboBox:focus {{
                border: 2px solid {theme.PRIMARY};
                padding: 3px 23px 3px 7px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border: none;
                background-color: transparent;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                subcontrol-position: center;
                image: url("{qss_icon('caret-down.svg')}");
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme.BG_DARK};
                border: 1px solid {theme.BORDER};
                selection-background-color: {theme.SECONDARY};
            }}
            QCheckBox {{
                spacing: 8px;
                color: {theme.TEXT_PRIMARY};
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 2px solid {theme.BORDER};
                background-color: {theme.BG_DARK};
            }}
            QCheckBox::indicator:checked {{
                background-color: {theme.PRIMARY};
                border-color: {theme.PRIMARY};
                image: url("{qss_icon('check.svg')}");
            }}
            QCheckBox:focus::indicator {{
                border-color: {theme.PRIMARY};
            }}
            
            /* ScrollBar */
            QScrollBar:vertical {{
                background-color: {theme.BG_DARK};
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
                background-color: {theme.SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {theme.BG_DARK};
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
                background-color: {theme.SECONDARY};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """
