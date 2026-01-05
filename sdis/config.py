"""
SDIS Configuration Management
=============================

Centralized configuration for paths, benchmarks, and hyperparameters.
Uses pydantic for validation where available, falls back to dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os
import json

# Try pydantic for enhanced validation, fallback to dataclass
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# =============================================================================
# LEAGUE BENCHMARKS (Statistical thresholds)
# =============================================================================

LEAGUE_BENCHMARKS: Dict[str, float] = {
    'elite_spm': 3.0,            # Top 10% SPM threshold (scaled -10 to +10)
    'elite_slfi': 3.0,           # Alias for SPM
    'fatigue_spm': -3.0,         # Bottom 25% (fatigued) SPM threshold
    'fatigue_slfi': -3.0,        # Alias
    'avg_q4_drop': 3.5,          # Average FG% drop Q1->Q4 (percentage points)
    'b2b_risk_multiplier': 1.35, # B2B fatigue risk multiplier
    'age_risk_threshold': 28,    # Age where fatigue risk increases
}


# =============================================================================
# SPM CONFIGURATION
# =============================================================================

SPM_SCALE_FACTOR: int = 10  # Scale SPM to -10 to +10 range

# Manual weights for SPM components (validated against data)
MANUAL_SPM_WEIGHTS: Dict[str, float] = {
    'fg_change': 0.40,      # FG% change is primary signal
    'pts_change': 0.25,     # Points change
    'tov_resil': 0.15,      # Turnover resilience
    'blk_change': 0.10,     # Blocks change
    'stl_change': 0.10,     # Steals change
}

# SPM component features
SPM_FEATURES: List[str] = ['fg_change', 'pts_change', 'tov_resil', 'blk_change', 'stl_change']


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for ML models."""

    # Gradient Boosting Regressor
    gb_n_estimators: int = 100
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.1
    gb_min_samples_leaf: int = 5
    gb_subsample: float = 0.8

    # Ridge Regression
    ridge_alpha: float = 1.0

    # Neural Network (LSTM)
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    mlp_hidden_size: int = 32
    sequence_length: int = 10

    # Training
    train_test_split: float = 0.7
    random_state: int = 42
    min_samples_for_model: int = 50


@dataclass
class PathConfig:
    """Configuration for file paths."""

    base_dir: Path = field(default_factory=lambda: Path(r"C:\Users\user"))

    @property
    def data_file(self) -> Path:
        return self.base_dir / "NBA_Quarter_ALL_Combined.xlsx"

    @property
    def parquet_file(self) -> Path:
        return self.base_dir / "NBA_Quarter_ALL_Combined.parquet"

    @property
    def injury_file(self) -> Path:
        return self.base_dir / "NBA_Injuries_Combined.xlsx"

    @property
    def salary_file(self) -> Path:
        return self.base_dir / "Sixers Salary Cap.xlsx"

    @property
    def neural_model_file(self) -> Path:
        return self.base_dir / "player_performance_model.pth"

    @property
    def scalers_file(self) -> Path:
        return self.base_dir / "player_model_scalers.pkl"

    @property
    def fatigue_data_file(self) -> Path:
        return self.base_dir / "Sixers Fatigue Data.xlsx"


@dataclass
class UIConfig:
    """Configuration for UI elements."""

    # Chart colors
    primary_color: str = "#4A90D9"
    secondary_color: str = "#10B981"
    warning_color: str = "#F59E0B"
    danger_color: str = "#EF4444"

    # Chart dimensions
    default_chart_height: int = 400
    small_chart_height: int = 300

    # Dark theme colors
    paper_bgcolor: str = "rgba(0,0,0,0)"
    plot_bgcolor: str = "rgba(26,45,74,0.3)"
    font_color: str = "#E5E7EB"
    grid_color: str = "rgba(255,255,255,0.1)"


# =============================================================================
# PLAYER NAME NORMALIZATION
# =============================================================================

PLAYER_NAME_MAP: Dict[str, str] = {
    'Valdez Edgecombe': 'VJ Edgecombe',
    'V. Edgecombe': 'VJ Edgecombe',
    # Add more mappings as needed
}


# =============================================================================
# MAIN CONFIG CLASS
# =============================================================================

@dataclass
class Config:
    """Main configuration container."""

    paths: PathConfig = field(default_factory=PathConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Expose benchmarks
    benchmarks: Dict[str, float] = field(default_factory=lambda: LEAGUE_BENCHMARKS.copy())
    spm_weights: Dict[str, float] = field(default_factory=lambda: MANUAL_SPM_WEIGHTS.copy())
    player_name_map: Dict[str, str] = field(default_factory=lambda: PLAYER_NAME_MAP.copy())

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            # TODO: Parse YAML into config objects
            return cls()
        except ImportError:
            print("PyYAML not installed, using default config")
            return cls()
        except FileNotFoundError:
            return cls()

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # TODO: Parse JSON into config objects
            return cls()
        except FileNotFoundError:
            return cls()

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            'benchmarks': self.benchmarks,
            'spm_weights': self.spm_weights,
            'models': {
                'gb_n_estimators': self.models.gb_n_estimators,
                'gb_max_depth': self.models.gb_max_depth,
                'ridge_alpha': self.models.ridge_alpha,
                'lstm_hidden_size': self.models.lstm_hidden_size,
            },
            'paths': {
                'base_dir': str(self.paths.base_dir),
                'data_file': str(self.paths.data_file),
            }
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
