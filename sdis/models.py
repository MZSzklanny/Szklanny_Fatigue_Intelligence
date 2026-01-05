"""
SDIS Models Module
==================

Machine learning models for fatigue prediction and performance analysis.
Includes regression, classification, and neural network predictors.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import os

from .config import get_config


# =============================================================================
# FATIGUE REGRESSION PREDICTOR
# =============================================================================

def build_fatigue_regression_predictor(
    metrics_data: pd.DataFrame,
    predict_rolling: bool = True
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict], Optional[pd.DataFrame], List[str], Optional[np.ndarray]]:
    """
    Build regression model to predict SPM (rolling average for stability).

    Key insight: Single-game SPM is too noisy (R²~0.01). Predicting 3-game
    rolling average is more stable and actionable.

    Args:
        metrics_data: DataFrame with SPM features and rolling history
        predict_rolling: If True, predict 3-game rolling avg (more stable)

    Returns:
        Tuple of (model, scaler, metrics_dict, importance_df, feature_cols, X_train_scaled)
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    config = get_config()

    # Create rolling target (3-game forward average) for stability
    metrics_data = metrics_data.copy()
    metrics_data = metrics_data.sort_values(['player', 'game_date'])

    # Target: 3-game FORWARD rolling average (what we're trying to predict)
    if predict_rolling:
        metrics_data['target_spm'] = metrics_data.groupby('player')['spm'].transform(
            lambda x: x.shift(-1).rolling(3, min_periods=1).mean()
        )
    else:
        metrics_data['target_spm'] = metrics_data['spm']

    # Enhanced feature set - prioritize rolling averages over single-game data
    feature_cols = [
        'slfi_avg_last5',       # PRIMARY: 5-game trend (most stable)
        'slfi_avg_last3',       # Recent 3-game trend
        'slfi_avg_last10',      # Long-term baseline
        'slfi_momentum',        # Trend direction (5g avg - last game)
        'slfi_trend',           # Short vs long trend (3g - 10g)
        'slfi_last1',           # Single game (lower weight naturally)
        'minutes_avg_last5',    # Workload
        'age',                  # Base physiological factor
        'age_load',             # Age-adjusted workload
        'age_b2b',              # Age × B2B interaction
        'recovery_penalty',     # Rest penalty
        'effort_index_last5',   # Effort trend
    ]

    # Add consistency/variance features if available
    if 'slfi_std_last10' in metrics_data.columns:
        feature_cols.append('slfi_std_last10')

    # Add B2B indicator directly
    if 'is_b2b' in metrics_data.columns:
        metrics_data['is_b2b_num'] = metrics_data['is_b2b'].astype(float)
        feature_cols.append('is_b2b_num')

    # Filter to valid data
    model_data = metrics_data.dropna(subset=feature_cols + ['target_spm'])
    model_data = model_data.copy()

    if len(model_data) < config.models.min_samples_for_model:
        return None, None, None, None, feature_cols, None

    # Sort by date for proper time-based split
    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data['target_spm'].values

    # Time-based split
    split_idx = int(len(X) * config.models.train_test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # GradientBoosting captures non-linear relationships
    model = GradientBoostingRegressor(
        n_estimators=config.models.gb_n_estimators,
        max_depth=config.models.gb_max_depth,
        learning_rate=config.models.gb_learning_rate,
        min_samples_leaf=config.models.gb_min_samples_leaf,
        subsample=config.models.gb_subsample,
        random_state=config.models.random_state
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    metrics_result = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'model_type': 'GradientBoosting',
        'target': '3-game rolling avg' if predict_rolling else 'single game'
    }

    # Feature importance names
    feature_names = [
        'Avg SPM (5g)', 'Avg SPM (3g)', 'Avg SPM (10g)', 'SPM Momentum',
        'SPM Trend (3g-10g)', 'Last SPM', 'Avg Minutes (5g)', 'Age',
        'Age-Adjusted Load', 'Age × B2B', 'Recovery Penalty', 'Effort Index (5g)'
    ]
    if 'slfi_std_last10' in feature_cols:
        feature_names.append('SPM Volatility (10g)')
    if 'is_b2b_num' in feature_cols:
        feature_names.append('Is B2B')

    importance = pd.DataFrame({
        'Feature': feature_names[:len(model.feature_importances_)],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, scaler, metrics_result, importance, feature_cols, X_train_scaled


# =============================================================================
# RANDOM FOREST REGRESSION PREDICTOR
# =============================================================================

def build_rf_regression_predictor(
    metrics_data: pd.DataFrame
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict], Optional[pd.DataFrame], List[str], Optional[np.ndarray]]:
    """
    Build Random Forest regression model to predict Q4 FG% drop with bootstrap CI.

    Uses ensemble of trees to provide prediction intervals via bootstrap.

    Args:
        metrics_data: DataFrame with SPM features

    Returns:
        Tuple of (model, scaler, metrics_dict, importance_df, feature_cols, X_train_scaled)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    config = get_config()

    feature_cols = [
        'slfi_avg_last5', 'slfi_avg_last3', 'slfi_avg_last10',
        'slfi_momentum', 'slfi_trend', 'slfi_last1',
        'minutes_avg_last5', 'age', 'age_load',
        'age_b2b', 'recovery_penalty', 'effort_index_last5'
    ]

    target_col = 'fg_change' if 'fg_change' in metrics_data.columns else 'slfi'

    model_data = metrics_data.dropna(subset=feature_cols + [target_col])
    model_data = model_data.copy()

    if len(model_data) < config.models.min_samples_for_model:
        return None, None, None, None, feature_cols, None

    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data[target_col].values

    split_idx = int(len(X) * config.models.train_test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=config.models.random_state,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    metrics_result = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'model_type': 'RandomForest'
    }

    importance = pd.DataFrame({
        'Feature': [
            'Avg SPM (5g)', 'Avg SPM (3g)', 'Avg SPM (10g)', 'SPM Momentum',
            'SPM Trend (3g-10g)', 'Last SPM', 'Avg Minutes (5g)', 'Age',
            'Age-Adjusted Load', 'Age × B2B', 'Recovery Penalty', 'Effort Index (5g)'
        ],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, scaler, metrics_result, importance, feature_cols, X_train_scaled


def predict_with_bootstrap_ci(
    rf_model: Any,
    X_pred_scaled: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Get prediction with confidence interval from Random Forest.

    Uses individual tree predictions as bootstrap samples.

    Args:
        rf_model: Fitted RandomForestRegressor
        X_pred_scaled: Scaled feature vector(s) to predict
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (prediction, ci_low, ci_high)
    """
    tree_preds = np.array([tree.predict(X_pred_scaled) for tree in rf_model.estimators_])
    pred = tree_preds.mean(axis=0)
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(tree_preds, alpha * 100, axis=0)
    ci_high = np.percentile(tree_preds, (1 - alpha) * 100, axis=0)

    if X_pred_scaled.shape[0] == 1:
        return float(pred[0]), float(ci_low[0]), float(ci_high[0])

    return pred, ci_low, ci_high


# =============================================================================
# IMPACT PREDICTOR (CLASSIFICATION)
# =============================================================================

def build_impact_predictor(
    metrics_data: pd.DataFrame
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict], Optional[pd.DataFrame], List[str]]:
    """
    Build logistic regression model to predict SPM < 0 (low impact) next game.

    Args:
        metrics_data: DataFrame with SPM features

    Returns:
        Tuple of (model, scaler, metrics_dict, importance_df, feature_cols)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    config = get_config()

    feature_cols = ['slis_last1', 'slis_avg_last5', 'slis_std_last10',
                    'minutes_avg_last5', 'is_b2b', 'days_rest']

    model_data = metrics_data.dropna(subset=feature_cols + ['slis'])
    model_data = model_data.copy()
    model_data['target'] = (model_data['slis'] < 0).astype(int)

    if len(model_data) < config.models.min_samples_for_model:
        return None, None, None, None, feature_cols

    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data['target'].values

    split_idx = int(len(X) * config.models.train_test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    importance = pd.DataFrame({
        'Feature': ['Last SPM', 'Avg SPM (5g)', 'SPM Volatility (10g)',
                    'Avg Minutes (5g)', 'B2B', 'Days Rest'],
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    return model, scaler, {
        'train_acc': model.score(X_train_scaled, y_train),
        'test_acc': model.score(X_test_scaled, y_test)
    }, importance, feature_cols


# Legacy alias
def build_spm_predictor(spm_data: pd.DataFrame):
    """Legacy function - calls build_impact_predictor."""
    return build_impact_predictor(spm_data)


# =============================================================================
# NEURAL NETWORK MODEL LOADER
# =============================================================================

def load_neural_model() -> Tuple[Optional[Any], Optional[Dict], Optional[List[str]]]:
    """
    Load the trained neural network model and scalers.

    Returns:
        Tuple of (model, scalers_dict, target_cols) or (None, None, None) if not found
    """
    try:
        import torch
        import joblib
        from sdis_neural_models import PlayerPerformanceModel
    except ImportError:
        return None, None, None

    config = get_config()
    model_path = str(config.paths.neural_model_file)
    scalers_path = str(config.paths.scalers_file)

    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        return None, None, None

    try:
        scalers = joblib.load(scalers_path)
        seq_scaler = scalers['seq_scaler']
        target_scaler = scalers['target_scaler']
        target_cols = scalers['target_cols']
        seq_features = scalers['seq_features']

        model = PlayerPerformanceModel(
            seq_input_dim=len(seq_features),
            static_input_dim=3,
            target_dims={name: 1 for name in target_cols},
            seq_hidden_dim=config.models.lstm_hidden_size,
            mlp_hidden_dims=[64, config.models.mlp_hidden_size],
            encoder_type='lstm',
            dropout=config.models.lstm_dropout
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        return model, scalers, target_cols
    except Exception:
        return None, None, None


def predict_player_next_game(
    model: Any,
    scalers: Dict,
    player_history: pd.DataFrame,
    target_cols: List[str]
) -> Optional[Dict[str, float]]:
    """
    Generate prediction for a player's next game.

    Args:
        model: Trained neural network model
        scalers: Dict with 'seq_scaler', 'target_scaler', 'seq_features'
        player_history: DataFrame with player's game-level history
        target_cols: List of target column names

    Returns:
        Dict mapping target names to predicted values, or None if insufficient data
    """
    import torch

    config = get_config()

    if len(player_history) < 5:
        return None

    seq_scaler = scalers['seq_scaler']
    target_scaler = scalers['target_scaler']
    seq_features = scalers['seq_features']

    # Get last N games
    recent = player_history.tail(config.models.sequence_length)
    sequence = recent[seq_features].values.astype(np.float32)

    # Pad if needed
    if len(sequence) < config.models.sequence_length:
        padding = np.zeros((config.models.sequence_length - len(sequence), len(seq_features)))
        sequence = np.vstack([padding, sequence])

    sequence_scaled = seq_scaler.transform(sequence)
    seq_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)
    static_tensor = torch.tensor([0, 2, 27], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predictions = model(seq_tensor, static_tensor)

    pred_array = np.array([[predictions[name].numpy()[0, 0] for name in target_cols]])
    pred_denorm = target_scaler.inverse_transform(pred_array)[0]

    return {name: pred_denorm[i] for i, name in enumerate(target_cols)}


# =============================================================================
# PHYSIOLOGICAL RISK FUNCTIONS
# =============================================================================

def apply_fatigue_risk_floor(
    model_probability: float,
    age: int,
    minutes_avg: float,
    is_b2b: bool,
    days_rest: int
) -> Tuple[float, float]:
    """
    Apply the Physiological Risk Floor to the model's predicted probability.

    Final fatigue risk = max(model_probability, PRF)

    Args:
        model_probability: Model's predicted fatigue probability
        age: Player age
        minutes_avg: Average minutes played
        is_b2b: Whether it's a back-to-back
        days_rest: Days since last game

    Returns:
        Tuple of (final_risk, prf)
    """
    from .metrics import compute_physiological_risk_floor

    prf = compute_physiological_risk_floor(age, minutes_avg, is_b2b, days_rest)
    final_risk = max(model_probability, prf)
    return final_risk, prf
