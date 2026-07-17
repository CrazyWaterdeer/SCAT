"""
Configuration management for SCAT.
Saves and loads user settings to/from JSON file.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict
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
    "last_image_dir": "",      # For labeling - image open
    "last_label_dir": "",      # For labeling - label save/load
    
    # Detection settings
    "detection": {
        "min_area": 20,
        "max_area": 10000,
        "threshold": 0.6
    },

    # Analysis options
    "analysis": {
        "model_type": "rf",
        "annotate": True,
        "visualize": True,
        "spatial": True,
        "stats": True,
        "report": True,
        "primary_metric": "total_deposits",   # predeclared endpoint (metrics.DEFAULT_METRIC)
        "normalization": "per_image",          # per_image | per_fly | per_area | per_time
        "confidence_threshold": 0.60,          # review/triage threshold on deposit confidence (display only; metrics.DEFAULT_THRESHOLD)
        "use_groups": True,                    # GUI: default the grouping toggle on
        "save_json": True,                     # GUI: also write results.json alongside the CSVs
    },

    # Performance / parallel engine
    "performance": {
        "parallel_enabled": True,   # fork process engine (falls back to sequential when unsafe)
        "worker_count": 0,          # 0 = auto (cpu-based)
    },

    # Labeling tool
    "labeling": {
        "add_shape": 0,             # default add-deposit shape index in the labeling window
    },

    # AI agent (non-secret selection only; ANTHROPIC_API_KEY comes from the env)
    "agent": {
        "backend": "auto",   # auto | subscription | api
        "model": "claude-opus-4-8",
        "max_loops": 40,
        "max_tokens": 4096,   # API backend: max output tokens per request
        "max_retries": 3,     # API backend: SDK retry count for 408/409/429/5xx (SDK does the backoff)
        # Extra dirs to scan for prior results (T3.1 resume). Empty = just the analyzed
        # folder's parent (where results dirs are written as siblings).
        "results_search_roots": []
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
        "save": "Ctrl+S",
        "quit": "Ctrl+Q",
        "undo": "Ctrl+Z",
        
        # Labeling / Edit mode
        "label_normal": "1",
        "label_rod": "2",
        "label_artifact": "3",
        "pan_mode": "Q",
        "select_mode": "S",
        "add_mode": "A",
        "delete": "Delete",
        "merge": "R",
        "group": "G",
        "ungroup": "F",
        
        # Analysis
        "run_analysis": "Ctrl+R"
    },
    
    # Window state
    "window": {
        "width": 1200,
        "height": 800,
        "maximized": False,
        "chat_visible": True
    },

    # UI / motion preferences
    "ui": {
        # Motion gate honored by scat.ui_motion. "auto" = animate normally;
        # "on" = reduce motion (animations land on their end value, no movement);
        # "off" = force full motion. The SCAT_REDUCED_MOTION env var forces "on".
        "reduced_motion": "auto"
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
        return copy.deepcopy(DEFAULT_CONFIG)

    def _merge_defaults(self, loaded: Dict) -> Dict:
        """Merge loaded config with defaults for missing keys."""
        # deepcopy, NOT shallow .copy(): deep_update recurses into nested dicts, so a shallow copy
        # would mutate the shared nested dicts of the module-global DEFAULT_CONFIG in place.
        result = copy.deepcopy(DEFAULT_CONFIG)

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
        self._data['shortcuts'] = copy.deepcopy(DEFAULT_CONFIG['shortcuts'])
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
