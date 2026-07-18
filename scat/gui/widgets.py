"""Leaf Qt widgets shared across SCAT's GUI tabs: keyboard-shortcut editor, the
background WorkerThread, the file/folder PathSelector, and the drag-drop DropZone."""
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QMessageBox, QKeySequenceEdit
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (
    QKeySequence
)

from ..config import config
from ..ui_common import (
    Theme, icon
)


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
        """Expand a mix of files and folders into a sorted, de-duplicated list of image paths.
        Paths may arrive in Windows or WSL form (drops/pastes from either side) — normalize each."""
        from ..pathutils import normalize_path
        out = []
        for pth in paths:
            p = Path(normalize_path(pth))
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in self.IMAGE_EXTS:
                        out.append(str(f))
            elif p.is_file() and p.suffix.lower() in self.IMAGE_EXTS:
                out.append(str(p))
        return sorted(dict.fromkeys(out))
