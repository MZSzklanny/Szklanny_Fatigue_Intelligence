"""
SDIS - Szklanny Decision Intelligence System
=============================================

A comprehensive NBA analytics platform featuring:
- SPRS: Szklanny Player Resilience System (fatigue analysis)
- Cap Lab: Salary cap management tools
- SNM: Szklanny Neural Model (predictive analytics)

Modules:
    config: Configuration management
    data_utils: Data loading and processing
    metrics: SPM calculations and statistics
    models: ML predictors (regression, classification, neural)
    visuals: Chart layouts and Plotly helpers
"""

__version__ = "2.0.0"
__author__ = "Szklanny Analytics"

from .config import Config, LEAGUE_BENCHMARKS
from .data_utils import (
    load_data,
    load_combined_quarter_data,
    load_injury_data,
    enforce_parquet,
)
from .metrics import (
    calculate_szklanny_metrics,
    compute_spm_weights_pca,
    compute_league_benchmarks,
)
from .models import (
    build_fatigue_regression_predictor,
    build_rf_regression_predictor,
    load_neural_model,
    predict_player_next_game,
)
from .visuals import (
    get_chart_layout,
    render_header,
    apply_custom_css,
    render_metric_card,
    render_status_badge,
    render_cap_bar,
    render_spm_gauge,
    get_color_scale,
    load_external_css,
)
