"""
Configuration management for SCAT.
Saves and loads user settings to/from JSON file.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    if Path.home().exists():
        config_dir = Path.home() / ".scat"
    else:
        config_dir = Path(".scat")
    
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the configuration file path."""
    return get_config_dir() / "config.json"


DEFAULT_CONFIG = {
    # Paths
    "last_input_dir": "",
    "last_output_dir": "",
    "last_model_path": "",
    "last_metadata_path": "",
    "last_image_dir": "",      # For labeling - image open
    "last_label_dir": "",      # For labeling - label save/load
    
    # Detection settings
    "detection": {
        "min_area": 20,
        "max_area": 10000,
        "threshold": 0.6,
        "edge_margin": 20,
        "sensitive_mode": False
    },
    
    # Analysis options
    "analysis": {
        "model_type": "rf",
        "annotate": True,
        "visualize": True,
        "spatial": True,
        "stats": True,
        "report": True,
        "group_by": ""
    },
    
    # Training settings
    "training": {
        "model_type": "rf",
        "n_estimators": 100,
        "epochs": 20
    },
    
    # Keyboard shortcuts (customizable)
    "shortcuts": {
        # Global
        "open": "Ctrl+O",
        "save": "Ctrl+S",
        "quit": "Ctrl+Q",
        "help": "F1",
        
        # Labeling
        "label_normal": "1",
        "label_rod": "2",
        "label_artifact": "3",
        "add_mode": "A",
        "select_mode": "S",
        "delete": "Delete",
        "merge": "M",
        "next_image": "N",
        "prev_image": "P",
        "next_deposit": ".",
        "prev_deposit": ",",
        
        # Analysis
        "run_analysis": "Ctrl+R",
        
        # Results
        "export_excel": "Ctrl+E",
        "open_detail": "Return"
    },
    
    # Window state
    "window": {
        "width": 1200,
        "height": 800,
        "maximized": False
    }
}


class Config:
    """Configuration manager with auto-save."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config_path = get_config_path()
        self._data = self._load()
        self._initialized = True
    
    def _load(self) -> Dict:
        """Load configuration from file."""
        if self._config_path.exists():
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # Merge with defaults (in case new keys were added)
                return self._merge_defaults(loaded)
            except (json.JSONDecodeError, IOError):
                pass
        return DEFAULT_CONFIG.copy()
    
    def _merge_defaults(self, loaded: Dict) -> Dict:
        """Merge loaded config with defaults for missing keys."""
        result = DEFAULT_CONFIG.copy()
        
        def deep_update(base: Dict, update: Dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(result, loaded)
        return result
    
    def save(self):
        """Save configuration to file."""
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value. Supports dot notation (e.g., 'detection.min_area')."""
        keys = key.split('.')
        value = self._data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, auto_save: bool = True):
        """Set a configuration value. Supports dot notation."""
        keys = key.split('.')
        data = self._data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
        
        if auto_save:
            self.save()
    
    def get_shortcut(self, action: str) -> str:
        """Get keyboard shortcut for an action."""
        return self.get(f"shortcuts.{action}", "")
    
    def set_shortcut(self, action: str, shortcut: str):
        """Set keyboard shortcut for an action."""
        self.set(f"shortcuts.{action}", shortcut)
    
    def reset_shortcuts(self):
        """Reset all shortcuts to defaults."""
        self._data['shortcuts'] = DEFAULT_CONFIG['shortcuts'].copy()
        self.save()
    
    @property
    def data(self) -> Dict:
        """Get the raw configuration data."""
        return self._data


def generate_output_folder_name(base_name: str = "results") -> str:
    """Generate output folder name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def get_timestamped_output_dir(parent_dir: Path, base_name: str = "results") -> Path:
    """Create and return a timestamped output directory."""
    folder_name = generate_output_folder_name(base_name)
    output_dir = Path(parent_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Singleton instance
config = Config()
