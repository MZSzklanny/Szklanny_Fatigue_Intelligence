"""
SDIS Metrics Module
====================

Szklanny Performance Metric (SPM) calculations and statistical utilities.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import streamlit as st

from .config import (
    get_config,
    SPM_SCALE_FACTOR,
    SPM_FEATURES,
    MANUAL_SPM_WEIGHTS,
    LEAGUE_BENCHMARKS,
)


# =============================================================================
# SPM WEIGHT COMPUTATION
# =============================================================================

def compute_spm_weights_pca(
    change_data: pd.DataFrame,
    variance_threshold: float = 0.8
) -> Tuple[Dict[str, float], float, Optional[object]]:
    """
    Compute SPM weights using PCA on Q1-3→Q4 change data.

    Args:
        change_data: DataFrame with change columns (fg_change, pts_change, etc.)
        variance_threshold: Minimum variance to explain

    Returns:
        Tuple of (weights_dict, variance_explained, pca_object)
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return MANUAL_SPM_WEIGHTS.copy(), 0.0, None

    # Check for required columns
    missing = [c for c in SPM_FEATURES if c not in change_data.columns]
    if missing:
        return MANUAL_SPM_WEIGHTS.copy(), 0.0, None

    # Get valid data
    valid_data = change_data[SPM_FEATURES].dropna()

    if len(valid_data) < 50:
        return MANUAL_SPM_WEIGHTS.copy(), 0.0, None

    # Standardize and run PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid_data)

    pca = PCA(n_components=1)
    pca.fit(X_scaled)

    # Get weights from first principal component
    weights_raw = np.abs(pca.components_[0])
    weights_raw = weights_raw / weights_raw.sum()  # Normalize to sum to 1

    variance_explained = pca.explained_variance_ratio_[0]

    # Build weights dictionary
    weights = dict(zip(SPM_FEATURES, weights_raw))

    return weights, variance_explained, pca


@st.cache_data(ttl=3600)
def compute_league_benchmarks(all_data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute NBA league-wide benchmarks for SPM normalization.

    Args:
        all_data: DataFrame with all NBA quarter data

    Returns:
        Dict with mean/std for each SPM component
    """
    benchmarks = {}

    # Calculate Q1-Q3 vs Q4 changes for league
    if 'qtr_num' not in all_data.columns:
        return benchmarks

    q1_q3 = all_data[all_data['qtr_num'].isin([1, 2, 3])]
    q4 = all_data[all_data['qtr_num'] == 4]

    # FG% change
    q1_q3_fg = q1_q3.groupby(['game_id', 'player']).agg({
        'fgm': 'sum', 'fga': 'sum'
    }).reset_index()
    q1_q3_fg['fg_pct'] = np.where(
        q1_q3_fg['fga'] > 0,
        q1_q3_fg['fgm'] / q1_q3_fg['fga'] * 100,
        np.nan
    )

    q4_fg = q4.groupby(['game_id', 'player']).agg({
        'fgm': 'sum', 'fga': 'sum'
    }).reset_index()
    q4_fg['q4_fg_pct'] = np.where(
        q4_fg['fga'] > 0,
        q4_fg['fgm'] / q4_fg['fga'] * 100,
        np.nan
    )

    merged = q1_q3_fg.merge(
        q4_fg[['game_id', 'player', 'q4_fg_pct']],
        on=['game_id', 'player']
    )
    merged['fg_change'] = merged['q4_fg_pct'] - merged['fg_pct']

    if len(merged) > 100:
        benchmarks['fg_change_mean'] = merged['fg_change'].mean()
        benchmarks['fg_change_std'] = merged['fg_change'].std()
        benchmarks['avg_q4_drop'] = -merged['fg_change'].mean()

    return benchmarks


# =============================================================================
# SPM CALCULATION
# =============================================================================

def calculate_spm_scores(
    metrics_data: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    return_details: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate SPM (Szklanny Performance-resilience Metric) for a dataset.

    This is the standalone SPM calculation that can be cached.

    Args:
        metrics_data: DataFrame with SPM component columns
        weights: Optional custom weights (defaults to manual weights)
        return_details: If True, return DataFrame with component breakdown

    Returns:
        If return_details=False: Series of SPM scores
        If return_details=True: DataFrame with SPM and components
    """
    if weights is None:
        weights = MANUAL_SPM_WEIGHTS.copy()

    # Calculate weighted sum
    spm_scores = pd.Series(0.0, index=metrics_data.index)

    for feature in SPM_FEATURES:
        if feature in metrics_data.columns:
            # Z-score the feature
            col_mean = metrics_data[feature].mean()
            col_std = metrics_data[feature].std()
            if col_std > 0:
                z_scored = (metrics_data[feature] - col_mean) / col_std
                spm_scores += weights.get(feature, 0) * z_scored

    # Scale to -10 to +10 range
    spm_scores = spm_scores * SPM_SCALE_FACTOR

    if return_details:
        result = metrics_data.copy()
        result['spm'] = spm_scores
        return result

    return spm_scores


def calculate_szklanny_metrics(
    data: pd.DataFrame,
    use_pca_weights: bool = False,
    league_benchmarks: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Calculate Szklanny Performance Metric (SPM) - consolidated late-game resilience score.

    Args:
        data: DataFrame with quarter-level data
        use_pca_weights: If True, compute weights via PCA
        league_benchmarks: Optional dict with 'mean' and 'std' for each SPM component

    Returns:
        DataFrame with SPM scores and component breakdowns
    """
    required_cols = ['player', 'game_date', 'qtr_num', 'fgm', 'fga', 'pts',
                     'stl', 'blk', 'tov', 'pf', 'minutes']
    missing = [c for c in required_cols if c not in data.columns]

    if missing:
        st.warning(f"Missing columns for SPM calculation: {missing}")
        return pd.DataFrame()

    # Handle minutes column naming
    if 'minutes' not in data.columns and 'total_minutes' in data.columns:
        data = data.rename(columns={'total_minutes': 'minutes'})

    # STEP 1: Calculate Q1-Q3 averages (baseline performance)
    q1_q3 = data[data['qtr_num'].isin([1, 2, 3])].copy()
    q1_q3_agg = q1_q3.groupby(['player', 'game_date']).agg({
        'fgm': 'sum',
        'fga': 'sum',
        'pts': 'sum',
        'stl': 'sum',
        'blk': 'sum',
        'tov': 'sum',
        'pf': 'sum',
        'minutes': 'sum'
    }).reset_index()

    # Calculate FG% for Q1-Q3
    q1_q3_agg['fg_pct_early'] = np.where(
        q1_q3_agg['fga'] > 0,
        q1_q3_agg['fgm'] / q1_q3_agg['fga'] * 100,
        np.nan
    )

    # Per-minute rates for Q1-Q3
    q1_q3_agg['pts_per_min_early'] = np.where(
        q1_q3_agg['minutes'] > 0,
        q1_q3_agg['pts'] / q1_q3_agg['minutes'],
        0
    )

    # Rename for clarity
    q1_q3_agg = q1_q3_agg.rename(columns={
        'stl': 'stl_early',
        'blk': 'blk_early',
        'tov': 'tov_early',
        'pf': 'pf_early',
        'pts': 'pts_early',
        'minutes': 'minutes_early'
    })

    # STEP 2: Calculate Q4 stats
    q4 = data[data['qtr_num'] == 4].copy()
    q4_agg = q4.groupby(['player', 'game_date']).agg({
        'fgm': 'sum',
        'fga': 'sum',
        'pts': 'sum',
        'stl': 'sum',
        'blk': 'sum',
        'tov': 'sum',
        'pf': 'sum',
        'minutes': 'sum'
    }).reset_index()

    q4_agg['fg_pct_q4'] = np.where(
        q4_agg['fga'] > 0,
        q4_agg['fgm'] / q4_agg['fga'] * 100,
        np.nan
    )

    q4_agg['pts_per_min_q4'] = np.where(
        q4_agg['minutes'] > 0,
        q4_agg['pts'] / q4_agg['minutes'],
        0
    )

    q4_agg = q4_agg.rename(columns={
        'stl': 'stl_q4',
        'blk': 'blk_q4',
        'tov': 'tov_q4',
        'pf': 'pf_q4',
        'pts': 'pts_q4',
        'minutes': 'minutes_q4',
        'fgm': 'fgm_q4',
        'fga': 'fga_q4'
    })

    # STEP 3: Merge and calculate changes
    metrics_data = q1_q3_agg.merge(
        q4_agg[['player', 'game_date', 'fg_pct_q4', 'pts_per_min_q4',
                'stl_q4', 'blk_q4', 'tov_q4', 'pf_q4', 'pts_q4',
                'minutes_q4', 'fgm_q4', 'fga_q4']],
        on=['player', 'game_date'],
        how='inner'
    )

    # Calculate change metrics (Q4 - baseline)
    metrics_data['fg_change'] = metrics_data['fg_pct_q4'] - metrics_data['fg_pct_early']
    metrics_data['pts_change'] = metrics_data['pts_per_min_q4'] - metrics_data['pts_per_min_early']

    # Resilience metrics (positive = maintained/improved)
    metrics_data['tov_resil'] = -(metrics_data['tov_q4'] - metrics_data['tov_early'] / 3)
    metrics_data['stl_change'] = metrics_data['stl_q4'] - metrics_data['stl_early'] / 3
    metrics_data['blk_change'] = metrics_data['blk_q4'] - metrics_data['blk_early'] / 3

    # Add game-level aggregates
    game_totals = data.groupby(['player', 'game_date']).agg({
        'minutes': 'sum',
        'pts': 'sum',
        'fgm': 'sum',
        'fga': 'sum'
    }).reset_index()
    game_totals = game_totals.rename(columns={
        'minutes': 'total_minutes',
        'pts': 'total_pts',
        'fgm': 'total_fgm',
        'fga': 'total_fga'
    })

    metrics_data = metrics_data.merge(
        game_totals,
        on=['player', 'game_date'],
        how='left'
    )

    # Add metadata from original data
    game_meta = data.groupby(['player', 'game_date']).agg({
        'is_b2b': 'first',
        'is_win': 'first',
        'age': 'first',
        'days_rest': 'first'
    }).reset_index()

    # Only merge columns that exist
    merge_cols = ['player', 'game_date']
    for col in ['is_b2b', 'is_win', 'age', 'days_rest']:
        if col in game_meta.columns:
            merge_cols.append(col)

    if len(merge_cols) > 2:
        metrics_data = metrics_data.merge(
            game_meta[merge_cols],
            on=['player', 'game_date'],
            how='left'
        )

    # Fill defaults
    if 'is_b2b' not in metrics_data.columns:
        metrics_data['is_b2b'] = False
    if 'age' not in metrics_data.columns:
        metrics_data['age'] = 27
    if 'days_rest' not in metrics_data.columns:
        metrics_data['days_rest'] = 2

    # STEP 4: Calculate SPM
    if use_pca_weights and len(metrics_data) >= 50:
        spm_weights, variance_explained, _ = compute_spm_weights_pca(metrics_data)
    else:
        spm_weights = MANUAL_SPM_WEIGHTS.copy()

    # Compute SPM as weighted sum of z-scored components
    metrics_data['spm'] = 0.0

    for feature, weight in spm_weights.items():
        if feature in metrics_data.columns:
            col_mean = metrics_data[feature].mean()
            col_std = metrics_data[feature].std()
            if col_std > 0:
                z_scored = (metrics_data[feature] - col_mean) / col_std
                metrics_data['spm'] += weight * z_scored

    # Scale SPM to -10 to +10 range
    metrics_data['spm'] = metrics_data['spm'] * SPM_SCALE_FACTOR

    # STEP 5: Add rolling features for prediction
    metrics_data = metrics_data.sort_values(['player', 'game_date'])

    # SPM rolling features
    metrics_data['slfi'] = metrics_data['spm']  # Alias
    metrics_data['slfi_last1'] = metrics_data.groupby('player')['slfi'].shift(1)
    metrics_data['slfi_avg_last3'] = metrics_data.groupby('player')['slfi'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    metrics_data['slfi_avg_last5'] = metrics_data.groupby('player')['slfi'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    metrics_data['slfi_avg_last10'] = metrics_data.groupby('player')['slfi'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    metrics_data['slfi_momentum'] = metrics_data['slfi_avg_last5'] - metrics_data['slfi_last1']
    metrics_data['slfi_trend'] = metrics_data['slfi_avg_last3'] - metrics_data['slfi_avg_last10']
    metrics_data['slfi_std_last10'] = metrics_data.groupby('player')['slfi'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()
    )

    # Minutes features
    metrics_data['minutes_last'] = metrics_data.groupby('player')['total_minutes'].shift(1)
    metrics_data['minutes_avg_last5'] = metrics_data.groupby('player')['total_minutes'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Age-adjusted features
    metrics_data['age'] = metrics_data['age'].fillna(27)
    metrics_data['age_load'] = metrics_data['minutes_avg_last5'].fillna(30) * (
        1 + np.maximum(0, metrics_data['age'] - 28) * 0.03
    )

    if 'is_b2b' in metrics_data.columns:
        metrics_data['age_b2b'] = metrics_data['is_b2b'].astype(float) * np.maximum(
            0, metrics_data['age'] - 30
        )
    else:
        metrics_data['age_b2b'] = 0

    if 'days_rest' in metrics_data.columns:
        metrics_data['recovery_penalty'] = np.maximum(0, 2 - metrics_data['days_rest']) * np.maximum(
            0, metrics_data['age'] - 30
        )
    else:
        metrics_data['recovery_penalty'] = 0

    # Effort index
    metrics_data['effort_q1'] = (
        metrics_data['stl_early'] + metrics_data['blk_early'] -
        metrics_data['pf_early'] - metrics_data['tov_early']
    )
    metrics_data['effort_q4'] = (
        metrics_data['stl_q4'] + metrics_data['blk_q4'] -
        metrics_data['pf_q4'] - metrics_data['tov_q4']
    )
    metrics_data['effort_change'] = metrics_data['effort_q4'] - metrics_data['effort_q1']

    effort_mean = metrics_data['effort_change'].mean()
    effort_std = metrics_data['effort_change'].std()
    metrics_data['effort_index'] = (metrics_data['effort_change'] - effort_mean) / (effort_std + 1e-6)

    metrics_data['effort_index_last5'] = metrics_data.groupby('player')['effort_index'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Legacy aliases for backward compatibility
    metrics_data['slis'] = metrics_data['spm']
    metrics_data['slis_calibrated'] = metrics_data['spm']

    return metrics_data


def compute_physiological_risk_floor(
    age: int,
    minutes: float,
    is_b2b: bool,
    days_rest: int
) -> float:
    """
    Compute physiological risk floor based on age, workload, and rest.

    Args:
        age: Player age
        minutes: Average minutes played
        is_b2b: Whether it's a back-to-back game
        days_rest: Days since last game

    Returns:
        Risk floor as fraction (0-1)
    """
    config = get_config()

    # Base risk from minutes
    minutes_risk = min(minutes / 48.0, 1.0) ** 1.5 * 0.4

    # Age multiplier
    if age >= config.benchmarks.get('age_risk_threshold', 28):
        age_mult = 1.0 + (age - 28) * 0.03
    else:
        age_mult = 1.0 - (28 - age) * 0.01

    age_mult = max(0.8, min(age_mult, 1.5))

    # B2B multiplier
    b2b_mult = config.benchmarks.get('b2b_risk_multiplier', 1.35) if is_b2b else 1.0

    # Rest penalty
    rest_penalty = max(0, (2 - days_rest) * 0.1)

    return min(minutes_risk * age_mult * b2b_mult + rest_penalty, 1.0)
