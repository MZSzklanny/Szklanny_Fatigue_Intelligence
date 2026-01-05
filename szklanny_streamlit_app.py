"""
SPRS - Szklanny Performance Resilience System
==============================================
NBA Performance Analytics & Fatigue Management Platform
Includes: Szklanny Performance Metric (SPM)
Run with: streamlit run szklanny_streamlit_app.py
"""

import datetime
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
# Optional imports for neural model (may not be available on Streamlit Cloud)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# DATA DIRECTORY HELPER (works on local and Streamlit Cloud)
# =============================================================================
def get_data_dir():
    """Get the data directory - works on local and Streamlit Cloud."""
    # Check current directory first (Streamlit Cloud)
    if os.path.exists("NBA_Quarter_ALL_Combined.parquet") or os.path.exists("NBA_Quarter_ALL_Combined.xlsx"):
        return "."
    # Check local Windows path
    if os.path.exists(r"C:\Users\user\NBA_Quarter_ALL_Combined.parquet"):
        return r"C:\Users\user"
    if os.path.exists(r"C:\Users\user\NBA_Quarter_ALL_Combined.xlsx"):
        return r"C:\Users\user"
    # Default to current directory
    return "."

DATA_DIR = get_data_dir()

# =============================================================================
# LEAGUE BENCHMARKS (Research-based reference values)
# =============================================================================
LEAGUE_BENCHMARKS = {
    'avg_q4_fg_drop': -1.0,      # Modern average FG% drop Q1-3 avg → Q4
    'elite_spm': 3.0,            # Top 10% SPM threshold (scaled -10 to +10)
    'good_spm': 1.0,             # Above average resilience
    'fatigue_spm': -3.0,         # Bottom 25% (fatigued) SPM threshold
    'severe_fatigue_spm': -5.0,  # Severe fatigue concern
    'min_games_reliable': 20,    # Minimum games for reliable metrics
    'min_games_warning': 5,      # Minimum for any calculation
    # Legacy keys for backward compatibility
    'elite_slfi': 0.5,
    'fatigue_slfi': -0.5,
}


# =============================================================================
# STATISTICAL UTILITY FUNCTIONS
# =============================================================================

def compute_confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval using t-distribution for small samples.

    Returns: (mean, ci_lower, ci_upper)
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    return mean, ci[0], ci[1]


def compute_q1_q4_significance(q1_values, q4_values):
    """
    Compute p-value for Q1 vs Q4 difference using paired t-test.

    Returns: (t_statistic, p_value, is_significant)
    """
    q1_arr = np.array(q1_values)
    q4_arr = np.array(q4_values)

    # Remove pairs with NaN
    valid_mask = ~(np.isnan(q1_arr) | np.isnan(q4_arr))
    q1_valid = q1_arr[valid_mask]
    q4_valid = q4_arr[valid_mask]

    if len(q1_valid) < 3:
        return np.nan, np.nan, False

    t_stat, p_value = stats.ttest_rel(q1_valid, q4_valid)
    is_significant = p_value < 0.05
    return t_stat, p_value, is_significant


def format_with_ci(mean_val, ci_low, ci_high, decimals=2):
    """Format a value with its confidence interval."""
    if np.isnan(mean_val):
        return "N/A"
    return f"{mean_val:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def get_sample_size_warning(n, threshold=20):
    """Return warning text if sample size is below threshold."""
    if n < threshold:
        return f"Low sample (n={n})"
    return None


def compute_pca_weights(ratio_data):
    """
    Derive optimal SLIS weights from PCA on impact stat ratios.

    Uses first principal component loadings as weights.
    This finds the linear combination that explains maximum variance.

    Args:
        ratio_data: DataFrame with columns r_pts, r_trb, r_ast, r_stl, r_blk, r_tov

    Returns:
        dict of stat: weight, or None if insufficient data
    """
    from sklearn.decomposition import PCA

    impact_cols = ['r_pts', 'r_trb', 'r_ast', 'r_stl', 'r_blk', 'r_tov']

    # Check all columns exist
    missing = [c for c in impact_cols if c not in ratio_data.columns]
    if missing:
        return None

    valid_data = ratio_data[impact_cols].dropna()

    if len(valid_data) < 50:
        return None  # Use default weights if insufficient data

    pca = PCA(n_components=1)
    pca.fit(valid_data)

    # First PC loadings become weights
    stat_names = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']
    weights = dict(zip(stat_names, pca.components_[0]))

    # Normalize so weights sum to reasonable magnitude
    # Also flip TOV sign if needed (should be negative)
    if weights['tov'] > 0:
        weights['tov'] = -abs(weights['tov'])

    return weights


def get_chart_layout():
    """Return consistent chart layout settings with grey background."""
    return dict(
        template='plotly_dark',
        paper_bgcolor='#2d3748',
        plot_bgcolor='#2d3748',
        font=dict(color='#ffffff'),
        title_font=dict(color='#ffffff'),
        legend=dict(
            bgcolor='rgba(45, 55, 72, 0.8)',
            font=dict(color='#ffffff')
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        )
    )


# Page config
st.set_page_config(
    page_title="SDIS - Szklanny Decision Intelligence System",
    page_icon="basketball",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css():
    """Load external CSS file for styling."""
    css_path = os.path.join(os.path.dirname(__file__), "sdis_styles.css")
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Graceful fallback if CSS file missing

load_css()

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load and process all data"""
    file_path = "Sixers Fatigue Data.xlsx"

    try:
        xl = pd.ExcelFile(file_path)
    except Exception as e:
        st.error(f"Could not open Excel file: {e}")
        return {}

    q_sheets = {
        'Pacers (Control)': ["Control - '24 Pacers Q Data", "Control - '24 Pacers Q Data Pof"],
        'Sixers 2024-25': ['24-25 Sixers Q Data'],
        'Sixers 2025-26': ['25-26 Sixers Q Data']
    }

    adv_sheets = {
        'Pacers (Control)': ["Control - '24 Pacers Adv Data ", "Control - '24 Pacers Adv Poff"],
        'Sixers 2024-25': ['24-25 Sixers Advanced Data'],
        'Sixers 2025-26': ['25-26 Sixers Advanced Data']
    }

    datasets = {}

    for key in q_sheets:
        dfs = []
        for sheet in q_sheets[key]:
            try:
                df = pd.read_excel(xl, sheet_name=sheet)
                df.columns = df.columns.str.lower().str.strip()

                # Normalize player names
                PLAYER_NAME_MAP = {
                    'Valdez Edgecombe': 'VJ Edgecombe',
                    'V. Edgecombe': 'VJ Edgecombe',
                }
                if 'player' in df.columns:
                    df['player'] = df['player'].replace(PLAYER_NAME_MAP)

                if 'quarter' in df.columns:
                    df = df.rename(columns={'quarter': 'qtr'})

                df = df[df['qtr'].astype(str).str.match(r'^Q[1-4]$', na=False)]
                df['qtr_num'] = df['qtr'].str.replace('Q', '').astype(int)
                df = df.rename(columns={'win/loss': 'win_loss'})

                # Calculate FG% (source columns are correct)
                df['fg_pct'] = np.where(df['fga'] > 0, df['fgm'] / df['fga'] * 100, np.nan)
                df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
                df['dataset'] = key
                dfs.append(df)
            except Exception as e:
                pass

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)

            # Calculate B2B
            combined = combined.sort_values(['player', 'game_date', 'qtr_num'])
            game_dates = combined.groupby(['player', 'game_date']).size().reset_index()[['player', 'game_date']]
            game_dates = game_dates.sort_values(['player', 'game_date'])
            game_dates['prev_game'] = game_dates.groupby('player')['game_date'].shift(1)
            game_dates['days_rest'] = (game_dates['game_date'] - game_dates['prev_game']).dt.days
            game_dates['is_b2b'] = game_dates['days_rest'] == 1

            combined = combined.merge(game_dates[['player', 'game_date', 'is_b2b', 'days_rest']],
                                       on=['player', 'game_date'], how='left')
            combined['is_b2b'] = combined['is_b2b'].fillna(False)
            combined['is_win'] = combined['win_loss'].str.upper() == 'W'

            # Load advanced data for age/mpg
            for adv_sheet in adv_sheets.get(key, []):
                try:
                    adv_df = pd.read_excel(xl, sheet_name=adv_sheet)
                    adv_df.columns = adv_df.columns.str.lower().str.strip()

                    if key == 'Sixers 2025-26':
                        adv_df = adv_df[adv_df['age'].apply(lambda x: str(x).isdigit() if pd.notna(x) else False)]

                    adv_df['age'] = pd.to_numeric(adv_df['age'], errors='coerce')
                    adv_df['games'] = pd.to_numeric(adv_df['games'], errors='coerce')
                    adv_df['minutes_total'] = pd.to_numeric(adv_df['minutes_total'], errors='coerce')
                    adv_df['mpg'] = adv_df['minutes_total'] / adv_df['games']

                    combined['player_clean'] = combined['player'].str.strip().str.lower()
                    adv_df['player_clean'] = adv_df['player'].str.strip().str.lower()
                    combined = combined.merge(adv_df[['player_clean', 'age', 'mpg']],
                                               on='player_clean', how='left', suffixes=('', '_adv'))
                except Exception:
                    pass  # Skip advanced data if unavailable

            datasets[key] = combined

    return datasets


def load_data_from_upload(uploaded_file):
    """
    Load data from an uploaded Excel file.

    Expects similar structure to the default file:
    - Quarter-level data with columns: player, game_date, qtr, pts, trb, ast, stl, blk, tov, pf, fgm, fga, minutes
    - Optional: win_loss column
    """
    try:
        xl = pd.ExcelFile(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not open Excel file: {e}")

    datasets = {}

    for sheet_name in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet_name)
            df.columns = df.columns.str.lower().str.strip()

            # Check for required columns
            required = ['player', 'game_date']
            if not all(col in df.columns for col in required):
                continue  # Skip sheets without required columns

            # Normalize quarter column
            if 'quarter' in df.columns:
                df = df.rename(columns={'quarter': 'qtr'})

            if 'qtr' not in df.columns:
                continue

            # Filter to valid quarters
            df = df[df['qtr'].astype(str).str.match(r'^Q[1-4]$', na=False)]
            if len(df) == 0:
                continue

            df['qtr_num'] = df['qtr'].str.replace('Q', '').astype(int)

            # Handle win/loss
            if 'win/loss' in df.columns:
                df = df.rename(columns={'win/loss': 'win_loss'})
            if 'win_loss' not in df.columns:
                df['win_loss'] = 'W'  # Default

            # Calculate FG%
            if 'fga' in df.columns and 'fgm' in df.columns:
                df['fg_pct'] = np.where(df['fga'] > 0, df['fgm'] / df['fga'] * 100, np.nan)

            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
            df['dataset'] = sheet_name

            # Calculate B2B
            df = df.sort_values(['player', 'game_date', 'qtr_num'])
            game_dates = df.groupby(['player', 'game_date']).size().reset_index()[['player', 'game_date']]
            game_dates = game_dates.sort_values(['player', 'game_date'])
            game_dates['prev_game'] = game_dates.groupby('player')['game_date'].shift(1)
            game_dates['days_rest'] = (game_dates['game_date'] - game_dates['prev_game']).dt.days
            game_dates['is_b2b'] = game_dates['days_rest'] == 1

            df = df.merge(game_dates[['player', 'game_date', 'is_b2b', 'days_rest']],
                         on=['player', 'game_date'], how='left')
            df['is_b2b'] = df['is_b2b'].fillna(False)
            df['is_win'] = df['win_loss'].str.upper() == 'W'

            # Default age if not present
            if 'age' not in df.columns:
                df['age'] = 27

            datasets[sheet_name] = df

        except Exception as e:
            continue  # Skip problematic sheets

    if not datasets:
        raise ValueError("No valid data sheets found in uploaded file")

    return datasets


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_combined_quarter_data(file_path="NBA_Quarter_ALL_Combined.xlsx"):
    """
    Load quarter data from a combined flat file (all teams in one sheet).

    The file should have a 'dataset' column like 'BOS 2024-25' to identify team/season.
    Supports both .parquet and .xlsx files.
    """
    try:
        # Handle parquet files directly
        if file_path.endswith('.parquet'):
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
            else:
                return {}
        else:
            # Try parquet first (5-10x faster than Excel)
            parquet_path = file_path.replace('.xlsx', '.parquet')
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
            elif os.path.exists(file_path):
                df = pd.read_excel(file_path)
            else:
                return {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.warning(f"Could not load combined data: {e}")
        return {}

    df.columns = df.columns.str.lower().str.strip()

    # Filter to NBA teams only (exclude international/preseason opponents)
    NBA_TEAMS = {
        'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
        'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
        'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    }
    if 'team' in df.columns:
        before_count = len(df)
        df = df[df['team'].isin(NBA_TEAMS)]
        filtered_count = before_count - len(df)
        if filtered_count > 0:
            pass  # Silently filter preseason/international games

    # Normalize player names (combine variants)
    PLAYER_NAME_MAP = {
        'Valdez Edgecombe': 'VJ Edgecombe',
        'V. Edgecombe': 'VJ Edgecombe',
    }
    if 'player' in df.columns:
        df['player'] = df['player'].replace(PLAYER_NAME_MAP)

    # Check for required columns
    required = ['player', 'game_date', 'qtr', 'dataset']
    if not all(col in df.columns for col in required):
        st.warning("Combined file missing required columns")
        return {}

    # Filter to valid quarters
    df = df[df['qtr'].astype(str).str.match(r'^Q[1-4]$', na=False)]
    if 'qtr_num' not in df.columns:
        df['qtr_num'] = df['qtr'].str.replace('Q', '').astype(int)

    # Handle win/loss
    if 'win/loss' in df.columns:
        df = df.rename(columns={'win/loss': 'win_loss'})
    if 'win_loss' not in df.columns:
        df['win_loss'] = 'W'

    # Calculate FG%
    if 'fga' in df.columns and 'fgm' in df.columns:
        df['fg_pct'] = np.where(df['fga'] > 0, df['fgm'] / df['fga'] * 100, np.nan)

    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    # Split by dataset column (e.g., 'BOS 2024-25', 'LAL 2025-26')
    datasets = {}

    for dataset_name in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_name].copy()

        # Calculate B2B and days rest
        subset = subset.sort_values(['player', 'game_date', 'qtr_num'])
        game_dates = subset.groupby(['player', 'game_date']).size().reset_index()[['player', 'game_date']]
        game_dates = game_dates.sort_values(['player', 'game_date'])
        game_dates['prev_game'] = game_dates.groupby('player')['game_date'].shift(1)
        game_dates['days_rest'] = (game_dates['game_date'] - game_dates['prev_game']).dt.days
        game_dates['is_b2b'] = game_dates['days_rest'] == 1

        subset = subset.merge(game_dates[['player', 'game_date', 'is_b2b', 'days_rest']],
                              on=['player', 'game_date'], how='left')
        subset['is_b2b'] = subset['is_b2b'].fillna(False)
        subset['is_win'] = subset['win_loss'].astype(str).str.upper() == 'W'

        # Default age if not present (can be enriched later)
        if 'age' not in subset.columns:
            subset['age'] = 27

        datasets[dataset_name] = subset

    return datasets


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_injury_data(file_path="NBA_Injuries_Combined.xlsx"):
    """
    Load injury data and create a lookup dictionary.

    Returns dict: {(player_lower, date_str): {'status': status, 'reason': reason}}
    """
    # Try parquet first
    parquet_path = file_path.replace('.xlsx', '.parquet')
    try:
        if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) >= os.path.getmtime(file_path):
            df = pd.read_parquet(parquet_path)
        elif os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df.to_parquet(parquet_path, index=False)
        else:
            return {}
    except Exception:
        return {}

    df.columns = df.columns.str.lower().str.strip()

    lookup = {}
    for _, row in df.iterrows():
        player = str(row.get('player', '')).lower().strip()
        date = str(row.get('report_date', ''))
        status = row.get('status', '')
        reason = row.get('reason', '')

        if player and date:
            lookup[(player, date)] = {'status': status, 'reason': reason}

    return lookup


# =============================================================================
# SZKLANNY PERFORMANCE METRIC (SPM) - CONSOLIDATED SYSTEM
# =============================================================================
#
# Single unified metric: SPM (Szklanny Performance-resilience Metric)
#
# Question: "How well does this player maintain performance from Q1-3 avg to Q4?"
#
# Components (Q4 - Q1-3avg changes, higher = more resilient):
#   - FG% change: Shooting efficiency maintenance
#   - Points change: Scoring output maintenance
#   - TOV resilience: -TOV change (fewer turnovers in Q4 = good)
#   - BLK change: Rim protection maintenance
#   - STL change: Active defense maintenance (low impact per PCA)
#
# Weighting:
#   - PCA-derived when n >= 10 games (data-driven)
#   - Manual fallback: FG 30%, PTS 30%, TOV 25%, BLK 10%, STL 5%
#
# Interpretation (scaled -10 to +10 for clarity):
#   - SPM > +3: Elite resilience (top 10%)
#   - SPM > +1: Good resilience (above average)
#   - SPM ~ 0: Average (typical Q4 decline)
#   - SPM < -3: Significant late-game fade
#   - SPM < -5: Severe fatigue concern
# =============================================================================

# SPM scaling factor (raw z-scores * SCALE ~ -10 to +10)
SPM_SCALE_FACTOR = 10

# Manual fallback weights (based on PCA validation + domain knowledge)
# TOV is "very bad" so gets 25%; FG/PTS most important at 30% each
# STL has near-zero PCA loading, so minimal weight
MANUAL_SPM_WEIGHTS = {
    'fg_change': 0.30,      # FG% change (30%)
    'pts_change': 0.30,     # Points change (30%)
    'tov_resil': 0.25,      # TOV resilience = -TOV change (25%)
    'blk_change': 0.10,     # Blocks change (10%)
    'stl_change': 0.05      # Steals change (5%) - low impact per PCA
}

# SPM component feature columns
SPM_FEATURES = ['fg_change', 'pts_change', 'tov_resil', 'blk_change', 'stl_change']


def compute_spm_weights_pca(change_data, min_samples=10):
    """
    Compute SPM weights using PCA on Q1-3→Q4 change data.

    Args:
        change_data: DataFrame with columns fg_change, pts_change, tov_resil, blk_change, stl_change
        min_samples: Minimum samples required for PCA (default 10)

    Returns:
        (weights_dict, var_explained, source_string) or (None, None, None) if insufficient data
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Check all columns exist
    missing = [c for c in SPM_FEATURES if c not in change_data.columns]
    if missing:
        return None, None, None

    # Get valid data
    valid_data = change_data[SPM_FEATURES].dropna()

    if len(valid_data) < min_samples:
        return None, None, None

    # Standardize before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(valid_data)

    # PCA - extract first principal component
    pca = PCA(n_components=1)
    pca.fit(scaled_data)

    # Get loadings (first PC)
    loadings = pca.components_[0]

    # Ensure correct sign: higher values should mean better resilience
    # FG increase is good; if PCA has negative loading for fg_change, flip all
    fg_loading = loadings[0]
    if fg_loading < 0:
        loadings = -loadings

    # Convert to weights (absolute values, normalized to sum to 1)
    abs_loadings = np.abs(loadings)
    weights_raw = abs_loadings / abs_loadings.sum()

    weights = dict(zip(SPM_FEATURES, weights_raw))

    # Variance explained
    var_explained = pca.explained_variance_ratio_[0]

    source = f"PCA-derived (PC1 explains {var_explained*100:.0f}% variance, n={len(valid_data)})"

    return weights, var_explained, source


def calculate_spm_score(df_changes, return_details=False):
    """
    Calculate SPM (Szklanny Performance-resilience Metric) for a dataset.

    This is the standalone SPM calculation that can be cached.

    Args:
        df_changes: DataFrame with Q1-3→Q4 change columns
        return_details: If True, return (spm_series, explained_var, weights, source)

    Returns:
        If return_details=False: Series of SPM scores
        If return_details=True: (spm_series, explained_var, weights, source)
    """
    from sklearn.preprocessing import StandardScaler

    # Check minimum data
    if df_changes.empty or len(df_changes) < LEAGUE_BENCHMARKS['min_games_warning']:
        if return_details:
            return pd.Series(dtype=float), None, None, "Insufficient data"
        return pd.Series(dtype=float)

    # Ensure tov_resil exists (inverted TOV change)
    if 'tov_resil' not in df_changes.columns and 'tov_change' in df_changes.columns:
        df_changes = df_changes.copy()
        df_changes['tov_resil'] = -df_changes['tov_change']

    # Try PCA weights first
    weights, var_explained, source = compute_spm_weights_pca(df_changes, min_samples=10)

    if weights is None:
        weights = MANUAL_SPM_WEIGHTS.copy()
        var_explained = None
        source = "Manual weights (FG 30%, PTS 30%, TOV 25%, BLK 10%, STL 5%)"

    # Standardize each component
    spm_scores = pd.Series(0.0, index=df_changes.index)

    for feature in SPM_FEATURES:
        if feature in df_changes.columns:
            values = df_changes[feature].fillna(0)
            if values.std() > 0:
                z_scores = (values - values.mean()) / values.std()
            else:
                z_scores = 0
            spm_scores += weights.get(feature, 0) * z_scores

    # Scale to -10 to +10 range for interpretability
    spm_scores = spm_scores * SPM_SCALE_FACTOR

    if return_details:
        return spm_scores, var_explained, weights, source
    return spm_scores


@st.cache_data(ttl=3600)
def compute_league_benchmarks(all_data_tuple):
    """
    Compute NBA league-wide benchmarks for SPM normalization.

    Args:
        all_data_tuple: Tuple of (data_hash, data) - tuple for caching

    Returns:
        Dict with mean/std for each SPM component
    """
    _, all_data = all_data_tuple

    # Calculate Q1-3 avg vs Q4 changes for all players across league
    q123 = all_data[all_data['qtr_num'].isin([1, 2, 3])].groupby(['player', 'game_date']).agg({
        'pts': 'mean', 'trb': 'mean', 'ast': 'mean', 'stl': 'mean',
        'blk': 'mean', 'tov': 'mean', 'fgm': 'sum', 'fga': 'sum'
    }).reset_index()

    q4 = all_data[all_data['qtr_num'] == 4].groupby(['player', 'game_date']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum',
        'blk': 'sum', 'tov': 'sum', 'fgm': 'sum', 'fga': 'sum'
    }).reset_index()

    merged = q123.merge(q4, on=['player', 'game_date'], suffixes=('_q123', '_q4'))

    # Calculate FG%
    merged['fg_pct_q123'] = np.where(merged['fga_q123'] > 0,
                                      merged['fgm_q123'] / merged['fga_q123'] * 100, np.nan)
    merged['fg_pct_q4'] = np.where(merged['fga_q4'] > 0,
                                    merged['fgm_q4'] / merged['fga_q4'] * 100, np.nan)

    # Calculate change metrics
    merged['fg_change'] = merged['fg_pct_q4'] - merged['fg_pct_q123']
    merged['pts_change'] = merged['pts_q4'] - merged['pts_q123']
    merged['tov_resil'] = merged['tov_q123'] - merged['tov_q4']  # Inverted
    merged['blk_change'] = merged['blk_q4'] - merged['blk_q123']
    merged['stl_change'] = merged['stl_q4'] - merged['stl_q123']

    # Compute benchmarks
    benchmarks = {}
    for col in ['fg_change', 'pts_change', 'tov_resil', 'blk_change', 'stl_change']:
        valid = merged[col].dropna()
        if len(valid) > 10:
            benchmarks[col] = {'mean': valid.mean(), 'std': valid.std()}

    return benchmarks


def calculate_szklanny_metrics(data, min_q1_minutes=3.0, min_q1_fga=2, league_benchmarks=None):
    """
    Calculate Szklanny Performance Metric (SPM) - consolidated late-game resilience score.

    Uses PCA-derived weights when sufficient data available, otherwise manual weights.

    Args:
        data: DataFrame with quarter-level game data
        min_q1_minutes: Minimum Q1 minutes to include player-game (usage floor)
        min_q1_fga: Minimum Q1 FGA to include player-game (usage floor)
        league_benchmarks: Optional dict with 'mean' and 'std' for each SPM component
                          to normalize against NBA-wide averages instead of filtered data

    Returns:
        DataFrame with SPM scores and component breakdowns
    """

    alpha = 0.5  # Smoothing constant
    epsilon = 1e-6

    # =========================================================================
    # STEP 1: Aggregate Q1-Q3 (early game) and Q4 (late game) data
    # =========================================================================
    # Comparing combined Q1+Q2+Q3 performance vs Q4 performance

    # Total minutes per game (all quarters) + age data
    agg_dict = {
        'minutes': 'sum',
        'is_b2b': 'first',
        'days_rest': 'first'
    }
    # Include age if available
    if 'age' in data.columns:
        agg_dict['age'] = 'first'

    total_minutes = data.groupby(['player', 'game_date', 'dataset']).agg(agg_dict).reset_index()

    if 'age' in total_minutes.columns:
        total_minutes.columns = ['player', 'game_date', 'dataset', 'total_minutes', 'is_b2b', 'days_rest', 'age']
    else:
        total_minutes.columns = ['player', 'game_date', 'dataset', 'total_minutes', 'is_b2b', 'days_rest']
        total_minutes['age'] = np.nan  # Will fill with default later

    # Q1-Q3 combined data (early game baseline) - AVERAGED per quarter for fair comparison
    q123_data = data[data['qtr_num'].isin([1, 2, 3])].groupby(['player', 'game_date', 'dataset']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum',
        'tov': 'sum', 'pf': 'sum', 'fgm': 'sum', 'fga': 'sum', 'minutes': 'sum'
    }).reset_index()
    # Average per quarter (divide by 3) for fair comparison to single Q4
    for col in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'fgm', 'fga', 'minutes']:
        q123_data[col] = q123_data[col] / 3.0
    q1_cols = ['player', 'game_date', 'dataset'] + [f'{c}_q1' for c in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'fgm', 'fga', 'minutes']]
    q123_data.columns = q1_cols

    # Q4 data (late game)
    q4_data = data[data['qtr_num'] == 4].groupby(['player', 'game_date', 'dataset']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum',
        'tov': 'sum', 'pf': 'sum', 'fgm': 'sum', 'fga': 'sum', 'minutes': 'sum'
    }).reset_index()
    q4_cols = ['player', 'game_date', 'dataset'] + [f'{c}_q4' for c in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'fgm', 'fga', 'minutes']]
    q4_data.columns = q4_cols

    # Merge
    metrics_data = q123_data.merge(q4_data, on=['player', 'game_date', 'dataset'], how='inner')
    metrics_data = metrics_data.merge(total_minutes, on=['player', 'game_date', 'dataset'], how='left')

    if len(metrics_data) == 0:
        return pd.DataFrame()

    # =========================================================================
    # STEP 2: Apply USAGE FLOOR - filter out low-usage players
    # =========================================================================
    # This fixes the "Watford problem" - bench players with 2 minutes shouldn't rank high
    # Note: minutes_q1 now represents avg Q1-Q3 minutes per quarter

    metrics_data = metrics_data[
        (metrics_data['minutes_q1'] >= min_q1_minutes) |
        (metrics_data['fga_q1'] >= min_q1_fga)
    ].copy()

    if len(metrics_data) == 0:
        return pd.DataFrame()

    # =========================================================================
    # STEP 3: Calculate FG% for Q1-Q3 avg and Q4
    # =========================================================================
    metrics_data['fg_pct_q1'] = np.where(
        metrics_data['fga_q1'] > 0,
        metrics_data['fgm_q1'] / metrics_data['fga_q1'] * 100,
        np.nan
    )
    metrics_data['fg_pct_q4'] = np.where(
        metrics_data['fga_q4'] > 0,
        metrics_data['fgm_q4'] / metrics_data['fga_q4'] * 100,
        np.nan
    )

    # =========================================================================
    # STEP 4: Calculate SPM (Szklanny Performance-resilience Metric)
    # =========================================================================
    # CONSOLIDATED single metric replacing SLFI/SLIS
    #
    # Components (all oriented so HIGHER = MORE RESILIENT):
    #   - fg_change: FG% Q4 - FG% Q1-3avg (positive = improved shooting)
    #   - pts_change: PTS Q4 - PTS Q1-3avg (positive = more scoring)
    #   - tov_resil: TOV Q1-3avg - TOV Q4 (positive = fewer late TOV)
    #   - blk_change: BLK Q4 - BLK Q1-3avg (positive = more rim protection)
    #   - stl_change: STL Q4 - STL Q1-3avg (positive = more active defense)

    # Raw changes
    metrics_data['fg_change'] = metrics_data['fg_pct_q4'] - metrics_data['fg_pct_q1']
    metrics_data['pts_change'] = metrics_data['pts_q4'] - metrics_data['pts_q1']
    metrics_data['tov_resil'] = metrics_data['tov_q1'] - metrics_data['tov_q4']  # INVERTED: fewer TOV in Q4 = good
    metrics_data['blk_change'] = metrics_data['blk_q4'] - metrics_data['blk_q1']
    metrics_data['stl_change'] = metrics_data['stl_q4'] - metrics_data['stl_q1']

    # Z-score normalize all change metrics (using league benchmarks if provided)
    spm_components = ['fg_change', 'pts_change', 'tov_resil', 'blk_change', 'stl_change']
    for col in spm_components:
        if league_benchmarks and col in league_benchmarks:
            # Use NBA-wide benchmarks for normalization
            mean_val = league_benchmarks[col]['mean']
            std_val = league_benchmarks[col]['std']
        else:
            # Fall back to current data stats
            valid_data = metrics_data[col].dropna()
            if len(valid_data) > 1:
                mean_val = valid_data.mean()
                std_val = valid_data.std()
            else:
                mean_val, std_val = 0, 1

        metrics_data[f'z_{col}'] = (metrics_data[col] - mean_val) / (std_val + epsilon)

    # Try PCA-derived weights first
    pca_weights, var_explained, weight_source = compute_spm_weights_pca(metrics_data, min_samples=10)

    if pca_weights is not None:
        spm_weights = pca_weights
    else:
        spm_weights = MANUAL_SPM_WEIGHTS.copy()
        var_explained = None
        weight_source = "Manual weights (FG 30%, PTS 30%, TOV 25%, BLK 10%, STL 5%)"

    # Store weight source and variance explained for UI display
    metrics_data['spm_weight_source'] = weight_source
    metrics_data['spm_var_explained'] = var_explained if var_explained is not None else np.nan

    # Compute SPM as weighted sum of z-scored components
    metrics_data['spm'] = 0.0
    for component in spm_components:
        z_col = f'z_{component}'
        weight = spm_weights.get(component, 0)
        contrib_col = f'spm_contrib_{component}'
        metrics_data[contrib_col] = weight * metrics_data[z_col].fillna(0)
        metrics_data['spm'] += metrics_data[contrib_col]

    # Scale SPM to -10 to +10 range for interpretability
    metrics_data['spm'] = metrics_data['spm'] * SPM_SCALE_FACTOR

    # Store component weights for UI
    metrics_data['spm_weight_fg'] = spm_weights.get('fg_change', 0)
    metrics_data['spm_weight_pts'] = spm_weights.get('pts_change', 0)
    metrics_data['spm_weight_tov'] = spm_weights.get('tov_resil', 0)
    metrics_data['spm_weight_blk'] = spm_weights.get('blk_change', 0)
    metrics_data['spm_weight_stl'] = spm_weights.get('stl_change', 0)

    # Legacy aliases for backward compatibility (map to SPM)
    metrics_data['slfi'] = metrics_data['spm']  # SLFI now equals SPM
    metrics_data['slis'] = metrics_data['spm']  # SLIS now equals SPM
    metrics_data['slis_calibrated'] = metrics_data['spm']  # Calibrated also equals SPM

    # =========================================================================
    # STEP 5: Effort Index (Composite Proxy Metric - kept for predictor)
    # =========================================================================
    # Effort Index = (STL + BLK - PF - TOV) normalized

    metrics_data['effort_q1'] = (
        metrics_data['stl_q1'] + metrics_data['blk_q1'] -
        metrics_data['pf_q1'] - metrics_data['tov_q1']
    )
    metrics_data['effort_q4'] = (
        metrics_data['stl_q4'] + metrics_data['blk_q4'] -
        metrics_data['pf_q4'] - metrics_data['tov_q4']
    )
    metrics_data['effort_change'] = metrics_data['effort_q4'] - metrics_data['effort_q1']

    # Z-score the effort change
    effort_mean = metrics_data['effort_change'].mean()
    effort_std = metrics_data['effort_change'].std()
    metrics_data['effort_index'] = (metrics_data['effort_change'] - effort_mean) / (effort_std + 1e-6)

    # =========================================================================
    # STEP 6: Add rolling features for prediction
    # =========================================================================
    metrics_data = metrics_data.sort_values(['player', 'game_date'])

    # SLFI rolling features
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
    # Momentum: positive = trending up, negative = trending down
    metrics_data['slfi_momentum'] = metrics_data['slfi_avg_last5'] - metrics_data['slfi_last1']
    # Trend strength: difference between short and long term averages
    metrics_data['slfi_trend'] = metrics_data['slfi_avg_last3'] - metrics_data['slfi_avg_last10']
    # Volatility: standard deviation over 10 games (consistency measure)
    metrics_data['slfi_std_last10'] = metrics_data.groupby('player')['slfi'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()
    )

    # SLIS rolling features
    metrics_data['slis_last1'] = metrics_data.groupby('player')['slis'].shift(1)
    metrics_data['slis_avg_last5'] = metrics_data.groupby('player')['slis'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    metrics_data['slis_std_last10'] = metrics_data.groupby('player')['slis'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()
    )

    # Effort Index rolling features
    metrics_data['effort_index_last5'] = metrics_data.groupby('player')['effort_index'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Minutes features
    metrics_data['minutes_last'] = metrics_data.groupby('player')['total_minutes'].shift(1)
    metrics_data['minutes_avg_last5'] = metrics_data.groupby('player')['total_minutes'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # =========================================================================
    # STEP 7: Age-adjusted fatigue risk features
    # =========================================================================
    # These capture physiological fatigue dynamics that SLFI history alone misses
    #
    # Key insight: Age does not increase fatigue risk linearly.
    # Age MODULATES the cost of minutes, amplifies B2B risk, reduces recovery.

    # Fill missing age with league average (27)
    metrics_data['age'] = metrics_data['age'].fillna(27)

    # 1. Age-adjusted workload
    # After ~28, every extra year increases minute cost by ~3%
    # A 35yo playing 35 min = equivalent load of a 28yo playing ~42 min
    metrics_data['age_load'] = metrics_data['minutes_avg_last5'].fillna(30) * (
        1 + np.maximum(0, metrics_data['age'] - 28) * 0.03
    )

    # 2. Age × B2B interaction
    # Young players tolerate B2Bs, older players don't
    # A 35yo on a B2B has 5× the penalty of a 30yo
    metrics_data['age_b2b'] = metrics_data['is_b2b'].astype(float) * np.maximum(0, metrics_data['age'] - 30)

    # 3. Recovery penalty
    # Older players need more than 1 rest day
    # If days_rest < 2 AND age > 30, there's a compounding penalty
    metrics_data['recovery_penalty'] = (
        np.maximum(0, 2 - metrics_data['days_rest'].fillna(2)) *
        np.maximum(0, metrics_data['age'] - 30)
    )

    # =========================================================================
    # STEP 8: Closing Context Weight (CCW) - Calibrate SLIS for garbage time
    # =========================================================================
    # Problem: Garbage-time players (Watford, Walker) look like "elite closers"
    # because they perform well in low-leverage Q4 minutes.
    #
    # Solution: Discount SLIS based on closing context signals.
    # CCW = 1.0 for true closers, < 1.0 for garbage-time players
    #
    # DO NOT apply to SLFI (fatigue is fatigue regardless of context)

    # Get MPG for role weight (need to merge if not present)
    if 'mpg' not in metrics_data.columns:
        # Estimate MPG from total_minutes if not available
        player_mpg = metrics_data.groupby('player')['total_minutes'].mean().reset_index()
        player_mpg.columns = ['player', 'mpg']
        metrics_data = metrics_data.merge(player_mpg, on='player', how='left')

    metrics_data['mpg'] = metrics_data['mpg'].fillna(20)

    # A) Q4 Leverage Score - Closers play disproportionate Q4 minutes
    # RESCALED: Expected Q4 share ≈ 25%, so multiply by 4 to normalize
    # A player with normal distribution (25% Q4) gets leverage = 1.0
    # A player with 30% Q4 share gets leverage = 1.2 (true closer)
    # A player with 15% Q4 share gets leverage = 0.6 (subbed out late)
    metrics_data['q4_leverage'] = np.where(
        metrics_data['total_minutes'] > 0,
        (metrics_data['minutes_q4'] / metrics_data['total_minutes']) * 4.0,
        1.0  # Default to neutral if no total minutes
    )
    metrics_data['q4_leverage'] = metrics_data['q4_leverage'].clip(0, 1.5)  # Cap at 1.5

    # B) Role Stability Score - High MPG = trusted rotation player
    # 28+ MPG = full closer weight, below = scaled down
    metrics_data['role_weight'] = np.minimum(1.0, metrics_data['mpg'] / 28.0)

    # C) Q4 Offensive Presence - Must be involved offensively to be a closer
    # 3+ FGA in Q4 = full presence, below = scaled down
    metrics_data['q4_presence'] = np.minimum(1.0, metrics_data['fga_q4'] / 3.0)

    # D) Q4 Minutes Floor - Need meaningful Q4 minutes to be a true closer
    # 6+ Q4 minutes = full credit, below = scaled down
    # This prevents garbage-time players from hacking with 2 min + 2 FGA
    metrics_data['q4_minutes_floor'] = np.minimum(1.0, metrics_data['minutes_q4'] / 6.0)

    # E) Combine into CCW (Closing Context Weight)
    metrics_data['ccw'] = (
        metrics_data['q4_leverage'] *
        metrics_data['role_weight'] *
        metrics_data['q4_presence'] *
        metrics_data['q4_minutes_floor']
    )

    # Clamp CCW between 0.15 and 1.0
    # 0.15 = garbage-time players still get some credit (not zero)
    # 1.0 = true closers get full credit
    metrics_data['ccw'] = metrics_data['ccw'].clip(0.15, 1.0)

    # F) Apply CCW to create SLIS_calibrated
    # This is the "real" closing impact score
    metrics_data['slis_calibrated'] = metrics_data['slis'] * metrics_data['ccw']

    return metrics_data


# Legacy alias for backwards compatibility
def calculate_spm(data):
    """Legacy function - calls calculate_szklanny_metrics"""
    return calculate_szklanny_metrics(data)


def build_fatigue_predictor(metrics_data):
    """
    Build logistic regression model to predict meaningful fatigue (SLFI < -0.5) next game.

    Now includes age-adjusted physiological features:
    - age: Player's age
    - age_load: Minutes adjusted for age (older = higher effective load)
    - age_b2b: Age × B2B interaction (older players suffer more on B2Bs)
    - recovery_penalty: Penalty for insufficient rest at older ages
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # NEW feature set with age-adjusted dynamics
    # This captures: "Is this sustainable?" not just "Did he look tired?"
    feature_cols = [
        'slfi_last1',           # Recent observed fatigue
        'slfi_avg_last5',       # Fatigue trend
        'minutes_avg_last5',    # Raw workload
        'age',                  # Base physiological factor
        'age_load',             # Age-adjusted workload (key interaction)
        'age_b2b',              # Age × B2B interaction
        'recovery_penalty'      # Insufficient rest penalty for older players
    ]

    # Target: SLFI < -0.5 (meaningful fatigue, not just noise)
    # This improves signal quality - a tiny negative SLFI shouldn't count as "fatigued"
    model_data = metrics_data.dropna(subset=feature_cols + ['slfi'])
    model_data = model_data.copy()
    model_data['target'] = (model_data['slfi'] < -0.5).astype(int)

    if len(model_data) < 50:
        return None, None, None, None, feature_cols

    # Sort by date for proper time-based split
    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data['target'].values

    # Time-based split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    importance = pd.DataFrame({
        'Feature': [
            'Last SPM',
            'Avg SPM (5g)',
            'Avg Minutes (5g)',
            'Age',
            'Age-Adjusted Load',
            'Age × B2B',
            'Recovery Penalty'
        ],
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    return model, scaler, {'train_acc': train_acc, 'test_acc': test_acc}, importance, feature_cols


def build_fatigue_regression_predictor(metrics_data, predict_rolling=True):
    """
    Build regression model to predict SPM (rolling average for stability).

    Key insight: Single-game SPM is too noisy (R²~0.01). Predicting 3-game
    rolling average is more stable and actionable.

    Returns:
        model, scaler, metrics_dict, importance_df, feature_cols, X_train_scaled
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
    # Key insight: 5-game averages are more predictive than last game (less noise)
    # Note: Removed slfi_last1 (single game) as it was dominating importance
    # The momentum feature already captures last-game info relative to trend
    feature_cols = [
        'slfi_avg_last5',       # PRIMARY: 5-game trend (most stable)
        'slfi_avg_last3',       # Recent 3-game trend
        'slfi_avg_last10',      # Long-term baseline
        'slfi_momentum',        # Trend direction (5g avg - last game)
        'slfi_trend',           # Short vs long trend (3g - 10g)
        'minutes_avg_last5',    # Workload
        'age',                  # Base physiological factor
        'age_load',             # Age-adjusted workload
        'age_b2b',              # Age × B2B interaction
        'recovery_penalty',     # Rest penalty
        'effort_index_last5',   # Effort trend
    ]

    # Add consistency/variance features if available
    if 'slfi_std_last10' in metrics_data.columns:
        feature_cols.append('slfi_std_last10')  # Consistency matters

    # Add B2B indicator directly
    if 'is_b2b' in metrics_data.columns:
        metrics_data['is_b2b_num'] = metrics_data['is_b2b'].astype(float)
        feature_cols.append('is_b2b_num')

    # Add neural network projection features if available (hybrid model)
    neural_feature_cols = [
        'neural_proj_pts',
        'neural_proj_fga',
        'neural_proj_ts',
        'neural_proj_tov_rate',
        'neural_proj_game_score',
        'neural_expected_efficiency',
        'neural_volume_efficiency'
    ]
    available_neural = [col for col in neural_feature_cols if col in metrics_data.columns]
    if available_neural:
        feature_cols.extend(available_neural)

    # Filter to valid data
    model_data = metrics_data.dropna(subset=feature_cols + ['target_spm'])
    model_data = model_data.copy()

    if len(model_data) < 50:
        return None, None, None, None, feature_cols, None

    # Sort by date for proper time-based split
    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data['target_spm'].values  # Smoother target

    # Time-based split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # GradientBoosting captures non-linear relationships better than Ridge
    # Key params tuned to avoid overfitting on small data
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,           # Shallow trees prevent overfitting
        learning_rate=0.1,
        min_samples_leaf=5,    # Require 5+ samples per leaf
        subsample=0.8,         # Use 80% of data per tree (regularization)
        random_state=42
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

    # Feature importance (from GB, not coefficients)
    # Map column names to display names
    col_to_name = {
        'slfi_avg_last5': 'Avg SPM (5g)',
        'slfi_avg_last3': 'Avg SPM (3g)',
        'slfi_avg_last10': 'Avg SPM (10g)',
        'slfi_momentum': 'SPM Momentum',
        'slfi_trend': 'SPM Trend (3g-10g)',
        'minutes_avg_last5': 'Avg Minutes (5g)',
        'age': 'Age',
        'age_load': 'Age-Adjusted Load',
        'age_b2b': 'Age × B2B',
        'recovery_penalty': 'Recovery Penalty',
        'effort_index_last5': 'Effort Index (5g)',
        'slfi_std_last10': 'SPM Volatility (10g)',
        'is_b2b_num': 'Is B2B',
        'neural_proj_pts': 'Neural Proj Points',
        'neural_proj_fga': 'Neural Proj FGA',
        'neural_proj_ts': 'Neural Proj TS%',
        'neural_proj_tov_rate': 'Neural Proj TOV Rate',
        'neural_proj_game_score': 'Neural Proj Game Score',
        'neural_expected_efficiency': 'Neural Expected Eff',
        'neural_volume_efficiency': 'Neural Volume Eff'
    }
    # Build feature names in exact same order as feature_cols
    feature_names = [col_to_name.get(col, col) for col in feature_cols]

    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, scaler, metrics_result, importance, feature_cols, X_train_scaled


def build_rf_regression_predictor(metrics_data):
    """
    Build Random Forest regression model to predict Q4 FG% drop with bootstrap CI.

    Uses ensemble of trees to provide prediction intervals via bootstrap.

    Returns:
        model, scaler, metrics_dict, importance_df, feature_cols, X_train_scaled
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # Feature set - prioritize rolling averages (removed slfi_last1 - was dominating)
    feature_cols = [
        'slfi_avg_last5',       # PRIMARY: 5-game trend (most stable)
        'slfi_avg_last3',       # Recent 3-game trend
        'slfi_avg_last10',      # Long-term baseline
        'slfi_momentum',        # Trend direction (5g avg - last game)
        'slfi_trend',           # Short vs long trend (3g - 10g)
        'minutes_avg_last5',    # Raw workload
        'age',                  # Base physiological factor
        'age_load',             # Age-adjusted workload
        'age_b2b',              # Age × B2B interaction
        'recovery_penalty',     # Insufficient rest penalty
        'effort_index_last5'    # Effort proxy trend
    ]

    # Target: actual FG% change Q1→Q4
    target_col = 'fg_change' if 'fg_change' in metrics_data.columns else 'slfi'

    model_data = metrics_data.dropna(subset=feature_cols + [target_col])
    model_data = model_data.copy()

    if len(model_data) < 50:
        return None, None, None, None, feature_cols, None

    # Sort by date for proper time-based split
    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data[target_col].values

    # Time-based split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest (100 trees for bootstrap CI)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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
        'model_type': 'RandomForest'
    }

    # Feature importance (from RF) - matches 11-feature set (no Last SPM)
    importance = pd.DataFrame({
        'Feature': [
            'Avg SPM (5g)',         # Primary trend
            'Avg SPM (3g)',         # Recent trend
            'Avg SPM (10g)',        # Long-term baseline
            'SPM Momentum',         # Trend direction
            'SPM Trend (3g-10g)',   # Short vs long
            'Avg Minutes (5g)',
            'Age',
            'Age-Adjusted Load',
            'Age × B2B',
            'Recovery Penalty',
            'Effort Index (5g)'
        ],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, scaler, metrics_result, importance, feature_cols, X_train_scaled


def predict_with_bootstrap_ci(rf_model, X_pred_scaled, confidence=0.95):
    """
    Get prediction with confidence interval from Random Forest.

    Uses individual tree predictions as bootstrap samples.

    Args:
        rf_model: Fitted RandomForestRegressor
        X_pred_scaled: Scaled feature vector(s) to predict
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (prediction, ci_low, ci_high)
    """
    # Get predictions from all trees
    tree_preds = np.array([tree.predict(X_pred_scaled) for tree in rf_model.estimators_])

    # Mean prediction
    pred = tree_preds.mean(axis=0)

    # Percentile-based CI
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(tree_preds, alpha * 100, axis=0)
    ci_high = np.percentile(tree_preds, (1 - alpha) * 100, axis=0)

    # If single prediction, return scalars
    if X_pred_scaled.shape[0] == 1:
        return float(pred[0]), float(ci_low[0]), float(ci_high[0])

    return pred, ci_low, ci_high


def build_impact_predictor(metrics_data):
    """Build logistic regression model to predict SPM < 0 (low impact) next game"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Features for impact prediction (uses slis columns which are aliases for spm)
    feature_cols = ['slis_last1', 'slis_avg_last5', 'slis_std_last10', 'minutes_avg_last5', 'is_b2b', 'days_rest']

    # Target: SPM < 0 (lower impact in late game)
    model_data = metrics_data.dropna(subset=feature_cols + ['slis'])
    model_data = model_data.copy()
    model_data['target'] = (model_data['slis'] < 0).astype(int)

    if len(model_data) < 50:
        return None, None, None, None, feature_cols

    # Sort by date for proper time-based split
    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data['target'].values

    # Time-based split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    importance = pd.DataFrame({
        'Feature': ['Last SPM', 'Avg SPM (5g)', 'SPM Volatility (10g)', 'Avg Minutes (5g)', 'B2B', 'Days Rest'],
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    return model, scaler, {'train_acc': train_acc, 'test_acc': test_acc}, importance, feature_cols


# Legacy alias
def build_spm_predictor(spm_data):
    """Legacy function - calls build_impact_predictor"""
    return build_impact_predictor(spm_data)


def compute_physiological_risk_floor(age, minutes_avg, is_b2b, days_rest):
    """
    Compute Physiological Risk Floor (PRF) - a deterministic minimum fatigue risk.

    This ensures that a 40-year-old on a B2B at 40 minutes can NEVER be "low risk"
    regardless of their recent SLFI history.

    The PRF encodes physiological reality:
    - Age always increases risk (after 30)
    - Heavy minutes always increase risk (after 34)
    - B2B always increases risk significantly
    - Older players need more rest days

    Formula: sigmoid(0.08*(age-30) + 0.10*(minutes-34) + 0.6*is_b2b + 0.15*max(0, 2-days_rest) - 1.75)

    The -1.75 shift ensures baseline (z=0) maps to ~15% instead of 50%.
    Returns a probability between 0 and 1.
    """
    # Compute the linear combination
    z = (
        0.08 * max(0, age - 30) +           # Age risk (after 30)
        0.10 * max(0, minutes_avg - 34) +   # Workload risk (after 34 min)
        0.6 * float(is_b2b) +               # B2B risk (big impact)
        0.15 * max(0, 2 - days_rest)        # Recovery deficit risk
    )

    # Apply sigmoid with left-shift so baseline ≈ 15% instead of 50%
    # sigmoid(-1.75) ≈ 0.148 (15%), matching the intended behavior
    prf = 1 / (1 + np.exp(-(z - 1.75)))

    return prf


def apply_fatigue_risk_floor(model_probability, age, minutes_avg, is_b2b, days_rest):
    """
    Apply the Physiological Risk Floor to the model's predicted probability.

    Final fatigue risk = max(model_probability, PRF)

    This guarantees:
    - A 40yo on B2B at 40 min: PRF ≈ 63% → cannot be "low risk"
    - A 22yo on 3 days rest at 30 min: PRF ≈ 15% → model can show low risk
    - Recent resilience cannot override physiology
    """
    prf = compute_physiological_risk_floor(age, minutes_avg, is_b2b, days_rest)
    final_risk = max(model_probability, prf)
    return final_risk, prf


# =============================================================================
# MAIN APP
# =============================================================================

def render_header(module_name: str = ""):
    """Render the AI GM header with optional module name."""
    module_display = f'<div style="font-size: 0.9rem; color: rgba(100,181,246,0.9); margin-top: 8px; font-weight: 500;">{module_name}</div>' if module_name else ''

    st.markdown(f'''
    <div style="position: relative; padding: 2rem; margin-bottom: 1.5rem; background: linear-gradient(135deg, #0a1628 0%, #162d50 50%, #0a1628 100%); border-radius: 16px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <!-- Background grid pattern -->
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.1; background-image:
            linear-gradient(rgba(100,181,246,0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(100,181,246,0.3) 1px, transparent 1px);
            background-size: 40px 40px;">
        </div>
        <!-- DNA Strand Left -->
        <svg style="position: absolute; left: 30px; top: 50%; transform: translateY(-50%); opacity: 0.15;" width="60" height="120" viewBox="0 0 60 120">
            <path d="M30 0 Q50 15 30 30 Q10 45 30 60 Q50 75 30 90 Q10 105 30 120" stroke="#64B5F6" stroke-width="2" fill="none"/>
            <path d="M30 0 Q10 15 30 30 Q50 45 30 60 Q10 75 30 90 Q50 105 30 120" stroke="#64B5F6" stroke-width="2" fill="none"/>
            <line x1="15" y1="15" x2="45" y2="15" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="30" x2="15" y2="30" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="45" x2="45" y2="45" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="60" x2="15" y2="60" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="75" x2="45" y2="75" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="90" x2="15" y2="90" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="105" x2="45" y2="105" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
        </svg>
        <!-- DNA Strand Right -->
        <svg style="position: absolute; right: 30px; top: 50%; transform: translateY(-50%); opacity: 0.15;" width="60" height="120" viewBox="0 0 60 120">
            <path d="M30 0 Q50 15 30 30 Q10 45 30 60 Q50 75 30 90 Q10 105 30 120" stroke="#64B5F6" stroke-width="2" fill="none"/>
            <path d="M30 0 Q10 15 30 30 Q50 45 30 60 Q10 75 30 90 Q50 105 30 120" stroke="#64B5F6" stroke-width="2" fill="none"/>
            <line x1="15" y1="15" x2="45" y2="15" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="30" x2="15" y2="30" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="45" x2="45" y2="45" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="60" x2="15" y2="60" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="75" x2="45" y2="75" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="45" y1="90" x2="15" y2="90" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
            <line x1="15" y1="105" x2="45" y2="105" stroke="#64B5F6" stroke-width="1.5" opacity="0.6"/>
        </svg>
        <!-- Content -->
        <div style="position: relative; display: flex; align-items: center; justify-content: center; gap: 24px;">
            <!-- Logo -->
            <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <line x1="50" y1="5" x2="50" y2="95" stroke="white" stroke-width="5" stroke-linecap="round"/>
                <circle cx="50" cy="50" r="30" fill="none" stroke="white" stroke-width="5"/>
                <line x1="5" y1="50" x2="20" y2="50" stroke="white" stroke-width="5" stroke-linecap="round"/>
                <line x1="80" y1="50" x2="95" y2="50" stroke="white" stroke-width="5" stroke-linecap="round"/>
            </svg>
            <!-- Text -->
            <div style="text-align: left;">
                <div style="font-size: 1.1rem; font-weight: 400; color: rgba(255,255,255,0.6); letter-spacing: 2px; text-transform: uppercase;">
                    SDIS
                </div>
                <div style="font-size: 1.8rem; font-weight: 600; color: white; line-height: 1.2; letter-spacing: 0.5px;">
                    Szklanny Decision<br>Intelligence System
                </div>
                {module_display}
            </div>
        </div>
        <!-- Decorative data lines -->
        <svg style="position: absolute; bottom: 10px; left: 100px; opacity: 0.2;" width="150" height="40" viewBox="0 0 150 40">
            <path d="M0 35 L20 35 L30 15 L40 30 L50 10 L60 25 L70 20 L80 35 L150 35" stroke="#64B5F6" stroke-width="2" fill="none"/>
        </svg>
        <svg style="position: absolute; bottom: 10px; right: 100px; opacity: 0.2;" width="150" height="40" viewBox="0 0 150 40">
            <path d="M0 35 L30 35 L40 20 L50 30 L60 5 L70 25 L90 15 L110 30 L150 35" stroke="#64B5F6" stroke-width="2" fill="none"/>
        </svg>
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# CAP LAB PAGE
# =============================================================================

def cap_lab_page():
    """Cap Lab - Salary Cap Simulator."""
    from cap_lab_engine import CapCalculator, TradeSalaryMatcher, CAP_NUMBERS, SalaryDataScraper

    render_header("Cap Lab - Salary Cap Simulator")

    # Sidebar controls
    st.sidebar.header("Cap Lab Settings")

    season = st.sidebar.selectbox(
        "Season",
        options=["2024-25", "2025-26", "2026-27", "2027-28", "2028-29"],
        index=0
    )

    calc = CapCalculator(season)
    cap_nums = CAP_NUMBERS.get(season, CAP_NUMBERS["2024-25"])

    # Use session state for tab persistence
    if 'caplab_tab' not in st.session_state:
        st.session_state.caplab_tab = "Team Cap Sheet"

    caplab_tab_options = ["Team Cap Sheet", "Cap Projections", "Trade Checker", "Cap Rules"]

    st.sidebar.markdown("### 💰 Cap Lab View")
    selected_caplab_tab = st.sidebar.radio(
        "Select View",
        options=caplab_tab_options,
        index=caplab_tab_options.index(st.session_state.caplab_tab),
        key="caplab_tab_selector",
        label_visibility="collapsed"
    )
    st.session_state.caplab_tab = selected_caplab_tab

    # Visual tab buttons
    cap_tab_cols = st.columns(4)
    for i, tab_name in enumerate(caplab_tab_options):
        with cap_tab_cols[i]:
            if selected_caplab_tab == tab_name:
                st.markdown(f"<div style='background: rgba(74,144,217,0.3); padding: 8px; border-radius: 8px; text-align: center; border-bottom: 2px solid #4A90D9;'><small><b>{tab_name}</b></small></div>", unsafe_allow_html=True)
            else:
                if st.button(tab_name, key=f"caplab_btn_{i}", use_container_width=True):
                    st.session_state.caplab_tab = tab_name
                    st.rerun()

    # =========================================================================
    # TAB 1: TEAM CAP SHEET
    # =========================================================================
    if selected_caplab_tab == "Team Cap Sheet":
        st.markdown("### Team Salary Cap Analysis")

        # Team selector
        teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET',
                'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
                'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS',
                'TOR', 'UTA', 'WAS']

        selected_team = st.selectbox("Select Team", teams, index=teams.index('PHI'))

        # Try to load salary data
        salary_file = os.path.join(r"C:\Users\user", "salary_cache", "nba_salaries.xlsx")

        # Check for cap summary file (has accurate totals from Spotrac)
        summary_file = os.path.join(r"C:\Users\user", "salary_cache", "nba_cap_summary.xlsx")

        if os.path.exists(salary_file):
            # Show cache status with refresh option
            file_age_days = (datetime.datetime.now().timestamp() - os.path.getmtime(salary_file)) / 86400

            col_status, col_refresh = st.columns([3, 1])
            with col_status:
                st.caption(f"📁 Using cached Spotrac data ({file_age_days:.0f} days old)")
            with col_refresh:
                if st.button("🔄 Refresh", help="Re-fetch salary data from Spotrac"):
                    with st.spinner("Refreshing salary data from Spotrac..."):
                        scraper = SalaryDataScraper(cache_dir=os.path.join(DATA_DIR, "salary_cache"))
                        df = scraper.scrape_all_teams(delay=0.8)
                        if not df.empty:
                            st.success(f"✅ Refreshed {len(df)} contracts!")
                            st.rerun()

            salary_df = pd.read_excel(salary_file)
            team_data = salary_df[salary_df['Team'] == selected_team]

            # Load cap summary for accurate totals
            cap_summary = None
            if os.path.exists(summary_file):
                cap_summary = pd.read_excel(summary_file)
                team_summary = cap_summary[cap_summary['Team'] == selected_team]

            if not team_data.empty:
                # Parse salaries for current season
                salary_col = season if season in team_data.columns else '2024-25'
                if salary_col in team_data.columns:
                    team_data = team_data.copy()
                    team_data['Salary'] = team_data[salary_col].apply(
                        lambda x: float(x.replace('$', '').replace(',', '')) if isinstance(x, str) and x.strip() not in ['', '-'] else 0
                    )
                    team_data = team_data[team_data['Salary'] > 0]

                    # Use Spotrac summary if available, otherwise calculate
                    if cap_summary is not None and not team_summary.empty:
                        total_salary = team_summary['Active Roster'].iloc[0]
                        cap_space = team_summary['Cap Space'].iloc[0]
                    else:
                        total_salary = team_data['Salary'].sum()
                        cap_space = cap_nums['salary_cap'] - total_salary

                    # Cap status
                    status = calc.get_cap_status(total_salary)

                    # Display cap status cards
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Salary", f"${total_salary/1e6:.1f}M")
                    with col2:
                        st.metric("Salary Cap", f"${cap_nums['salary_cap']/1e6:.1f}M")
                    with col3:
                        st.metric("Cap Space", f"${cap_space/1e6:.1f}M")
                    with col4:
                        tier_colors = {'Under Cap': '🟢', 'Over Cap': '🟡', 'In Tax': '🟠', 'First Apron': '🔴', 'Second Apron': '⛔'}
                        st.metric("Status", f"{tier_colors.get(status['tier'], '')} {status['tier']}")

                    # Visual cap bar
                    st.markdown("### Cap Position")
                    cap_pct = min(100, (total_salary / cap_nums['second_apron']) * 100)
                    tax_pct = (cap_nums['luxury_tax'] / cap_nums['second_apron']) * 100
                    apron1_pct = (cap_nums['first_apron'] / cap_nums['second_apron']) * 100

                    st.markdown(f'''
                    <div style="position: relative; height: 40px; background: #1a202c; border-radius: 8px; overflow: hidden; margin: 1rem 0;">
                        <div style="position: absolute; left: 0; top: 0; height: 100%; width: {cap_pct}%; background: linear-gradient(90deg, #10B981 0%, #F59E0B {tax_pct}%, #EF4444 {apron1_pct}%, #7f1d1d 100%); transition: width 0.3s;"></div>
                        <div style="position: absolute; left: {tax_pct}%; top: 0; height: 100%; width: 2px; background: white; opacity: 0.5;"></div>
                        <div style="position: absolute; left: {apron1_pct}%; top: 0; height: 100%; width: 2px; background: white; opacity: 0.5;"></div>
                        <div style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: white; font-weight: 600; text-shadow: 0 1px 3px rgba(0,0,0,0.5);">
                            ${total_salary/1e6:.1f}M / ${cap_nums['salary_cap']/1e6:.0f}M Cap
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.7;">
                        <span>Cap: ${cap_nums['salary_cap']/1e6:.0f}M</span>
                        <span>Tax: ${cap_nums['luxury_tax']/1e6:.0f}M</span>
                        <span>1st Apron: ${cap_nums['first_apron']/1e6:.0f}M</span>
                        <span>2nd Apron: ${cap_nums['second_apron']/1e6:.0f}M</span>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Restrictions
                    restrictions = calc.get_apron_restrictions(total_salary)
                    if restrictions:
                        st.markdown("### Active Restrictions")
                        for r in restrictions:
                            st.markdown(f"- {r}")

                    # Tax bill
                    if status['in_tax']:
                        st.markdown(f"### Luxury Tax Bill: **${status['tax_bill']/1e6:.1f}M**")

                    # Player salaries table
                    st.markdown("### Roster Salaries")
                    display_df = team_data[['Player', 'Salary']].copy()
                    display_df['Salary'] = display_df['Salary'].apply(lambda x: f"${x/1e6:.2f}M")
                    display_df = display_df.sort_values('Salary', ascending=False, key=lambda x: x.str.replace('$', '').str.replace('M', '').astype(float))
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("📁 No salary data cached yet. Click below to fetch from Basketball Reference (one-time, ~2 min).")
            st.caption("Data is cached locally and won't need to be fetched again for 7 days.")
            if st.button("Fetch Salary Data", type="primary"):
                with st.spinner("Loading Salary Data....This only happens once."):
                    scraper = SalaryDataScraper(cache_dir=os.path.join(DATA_DIR, "salary_cache"))
                    df = scraper.scrape_all_teams()
                    if not df.empty:
                        st.success(f"✅ Loaded {len(df)} contracts! Data cached for 7 days.")
                        st.rerun()
                    else:
                        st.error("Failed to fetch salary data")

    # =========================================================================
    # TAB 2: CAP PROJECTIONS
    # =========================================================================
    if selected_caplab_tab == "Cap Projections":
        st.markdown("### Multi-Year Cap Projection")

        st.markdown("Project how your team's cap situation evolves over the next 4 years.")

        # Manual salary input for projection
        st.markdown("#### Enter Current Committed Salary by Year")

        col1, col2 = st.columns(2)
        with col1:
            y1_salary = st.number_input("2024-25 Salary ($M)", value=150.0, step=1.0) * 1e6
            y2_salary = st.number_input("2025-26 Salary ($M)", value=140.0, step=1.0) * 1e6
            y3_salary = st.number_input("2026-27 Salary ($M)", value=120.0, step=1.0) * 1e6
        with col2:
            y4_salary = st.number_input("2027-28 Salary ($M)", value=100.0, step=1.0) * 1e6
            y5_salary = st.number_input("2028-29 Salary ($M)", value=80.0, step=1.0) * 1e6

        salaries = [y1_salary, y2_salary, y3_salary, y4_salary, y5_salary]
        seasons_list = ["2024-25", "2025-26", "2026-27", "2027-28", "2028-29"]

        # Build projection chart
        projection_data = []
        for i, (s, sal) in enumerate(zip(seasons_list, salaries)):
            cn = CAP_NUMBERS.get(s, CAP_NUMBERS["2024-25"])
            projection_data.append({
                "Season": s,
                "Salary": sal,
                "Cap": cn["salary_cap"],
                "Tax": cn["luxury_tax"],
                "First Apron": cn["first_apron"],
                "Second Apron": cn["second_apron"],
                "Cap Space": max(0, cn["salary_cap"] - sal),
            })

        proj_df = pd.DataFrame(projection_data)

        # Chart
        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Bar(name='Committed Salary', x=proj_df['Season'], y=proj_df['Salary']/1e6,
                            marker_color='#4A90D9'))

        fig.add_trace(go.Scatter(name='Salary Cap', x=proj_df['Season'], y=proj_df['Cap']/1e6,
                                mode='lines+markers', line=dict(color='#10B981', width=2, dash='dot')))
        fig.add_trace(go.Scatter(name='Luxury Tax', x=proj_df['Season'], y=proj_df['Tax']/1e6,
                                mode='lines+markers', line=dict(color='#F59E0B', width=2, dash='dot')))
        fig.add_trace(go.Scatter(name='First Apron', x=proj_df['Season'], y=proj_df['First Apron']/1e6,
                                mode='lines+markers', line=dict(color='#EF4444', width=2, dash='dot')))

        fig.update_layout(**get_chart_layout(), height=400, title='Multi-Year Cap Projection',
                         yaxis_title='Millions ($)', xaxis_title='Season',
                         barmode='group')
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Flexibility score
        st.markdown("### Flexibility Analysis")
        total_future_space = sum(max(0, CAP_NUMBERS.get(s, {}).get("salary_cap", 140e6) - sal) for s, sal in zip(seasons_list[1:], salaries[1:]))
        flexibility_score = min(100, (total_future_space / (4 * 30e6)) * 100)

        st.metric("Future Flexibility Score", f"{flexibility_score:.0f}/100",
                 help="Based on projected cap space over next 4 years")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Future Cap Space", f"${total_future_space/4/1e6:.1f}M/yr")
        with col2:
            years_in_tax = sum(1 for s, sal in zip(seasons_list, salaries) if sal > CAP_NUMBERS.get(s, {}).get("luxury_tax", 170e6))
            st.metric("Years in Tax", f"{years_in_tax}/5")
        with col3:
            years_in_apron = sum(1 for s, sal in zip(seasons_list, salaries) if sal > CAP_NUMBERS.get(s, {}).get("first_apron", 178e6))
            st.metric("Years Above Apron", f"{years_in_apron}/5")

    # =========================================================================
    # TAB 3: TRADE CHECKER
    # =========================================================================
    if selected_caplab_tab == "Trade Checker":
        st.markdown("### Trade Salary Matcher")
        st.markdown("Check if a trade works under NBA salary matching rules.")

        matcher = TradeSalaryMatcher(season)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Team 1 (Your Team)")
            t1_salary = st.number_input("Team 1 Total Salary ($M)", value=160.0, key="t1_sal") * 1e6
            t1_outgoing = st.number_input("Outgoing Salary ($M)", value=25.0, key="t1_out") * 1e6
            t1_incoming = st.number_input("Incoming Salary ($M)", value=22.0, key="t1_in") * 1e6

        with col2:
            st.markdown("#### Team 2 (Trade Partner)")
            t2_salary = st.number_input("Team 2 Total Salary ($M)", value=140.0, key="t2_sal") * 1e6
            # Team 2's outgoing = Team 1's incoming, Team 2's incoming = Team 1's outgoing

        if st.button("Check Trade"):
            result = matcher.validate_trade(t1_outgoing, t1_incoming, t1_salary, t2_salary)

            if result['trade_valid']:
                st.success("Trade is VALID under salary matching rules!")
            else:
                st.error("Trade does NOT work under salary matching rules")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Team 1 Requirements:**")
                req = result['team1_requirements']
                st.write(f"- Situation: {req['team_cap_situation']}")
                st.write(f"- Rule: {req['rule']}")
                st.write(f"- Can receive: ${req['min_incoming']/1e6:.1f}M - ${req['max_incoming']/1e6:.1f}M")
                st.write(f"- Receiving: ${t1_incoming/1e6:.1f}M")
                if result['team1_valid']:
                    st.success("Team 1: OK")
                else:
                    st.error("Team 1: Salary mismatch")

            with col2:
                st.markdown("**Team 2 Requirements:**")
                req = result['team2_requirements']
                st.write(f"- Situation: {req['team_cap_situation']}")
                st.write(f"- Rule: {req['rule']}")
                st.write(f"- Can receive: ${req['min_incoming']/1e6:.1f}M - ${req['max_incoming']/1e6:.1f}M")
                st.write(f"- Receiving: ${t1_outgoing/1e6:.1f}M")
                if result['team2_valid']:
                    st.success("Team 2: OK")
                else:
                    st.error("Team 2: Salary mismatch")

    # =========================================================================
    # TAB 4: CAP RULES REFERENCE
    # =========================================================================
    if selected_caplab_tab == "Cap Rules":
        st.markdown("### NBA Cap Rules Reference")

        st.markdown(f"#### {season} Cap Numbers")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Salary Cap", f"${cap_nums['salary_cap']/1e6:.1f}M")
            st.metric("Luxury Tax Line", f"${cap_nums['luxury_tax']/1e6:.1f}M")
        with col2:
            st.metric("First Apron", f"${cap_nums['first_apron']/1e6:.1f}M")
            st.metric("Second Apron", f"${cap_nums['second_apron']/1e6:.1f}M")

        st.markdown("---")
        st.markdown("#### First Apron Restrictions")
        st.markdown("""
        Teams above the **First Apron** cannot:
        - Use the Bi-Annual Exception
        - Aggregate salaries in trades
        - Acquire a player via sign-and-trade
        - Send more than $6.4M cash in trades
        - Use full MLE (taxpayer MLE only: ~$5.2M)
        """)

        st.markdown("#### Second Apron Restrictions")
        st.markdown("""
        Teams above the **Second Apron** have ALL first apron restrictions PLUS:
        - Cannot trade first-round picks more than 7 years out
        - Cannot acquire players earning >$10.6M using exceptions in trades
        - Cannot use Traded Player Exception
        - Pick swap restrictions apply
        - Sign-and-trade receiving completely prohibited
        """)

        st.markdown("---")
        st.markdown("#### Trade Salary Matching Rules")
        st.markdown("""
        **Under the Cap:**
        - Can absorb salary up to available cap space without sending salary back

        **Over the Cap (below tax):**
        - Can receive 175% + $250K of outgoing salary (up to $7.5M outgoing)
        - Can receive outgoing + $5.75M (for $7.5M-$29M outgoing)
        - Can receive 125% of outgoing (for $29M+ outgoing)

        **In Tax (below first apron):**
        - Can receive 110% + $100K of outgoing salary

        **Above First Apron:**
        - Can receive 110% of outgoing salary (strictest matching)
        """)


# =============================================================================
# SPRS PAGE (Original functionality)
# =============================================================================

def sprs_page():
    """SPRS - Player Fatigue & Resilience Analysis."""
    render_header("SPRS - Player Fatigue & Resilience")

    # ==========================================================================
    # DATA LOADING
    # ==========================================================================
    with st.spinner('Loading data...'):
        data_dir = DATA_DIR
        # Try parquet first (faster), then xlsx
        parquet_path = os.path.join(data_dir, "NBA_Quarter_ALL_Combined.parquet")
        xlsx_path = os.path.join(data_dir, "NBA_Quarter_ALL_Combined.xlsx")

        if os.path.exists(parquet_path):
            datasets = load_combined_quarter_data(parquet_path)
        elif os.path.exists(xlsx_path):
            datasets = load_combined_quarter_data(xlsx_path)
        else:
            datasets = {}

        if datasets:
            st.sidebar.success(f"Loaded {len(datasets)} team-seasons")
        else:
            st.sidebar.warning("Combined file not found. Run nba_quarter_data_pull.py first.")
            datasets = {}

    # Load injury data (if available)
    injury_path = os.path.join(DATA_DIR, "NBA_Injuries_Combined.xlsx")
    injury_lookup = load_injury_data(injury_path)
    if injury_lookup:
        st.sidebar.info(f"Injury data: {len(injury_lookup)} records")

    if not datasets:
        st.error("No data loaded. Please check the file path or upload data.")
        return

    # Combine all data
    all_data = pd.concat(datasets.values(), ignore_index=True)

    # Compute league-wide benchmarks for SPM (using all NBA data)
    data_hash = hash(tuple(all_data['game_date'].astype(str).tolist()[:100]))  # Simple hash for caching
    league_benchmarks = compute_league_benchmarks((data_hash, all_data))

    # ==========================================================================
    # SIDEBAR FILTERS
    # ==========================================================================
    st.sidebar.header("Filters")

    # Parse available teams and seasons from dataset keys (e.g., "PHI 2024-25")
    available_teams = sorted(set(k.split()[0] for k in datasets.keys() if ' ' in k))
    available_seasons = sorted(set(k.split()[1] for k in datasets.keys() if ' ' in k), reverse=True)

    # Season selector (default to 2025-26 if available)
    default_season_idx = 0 if '2025-26' in available_seasons else 0
    selected_seasons = st.sidebar.multiselect(
        "Season",
        options=available_seasons,
        default=[available_seasons[default_season_idx]] if available_seasons else []
    )

    # Team selector (default to PHI/Sixers)
    if 'select_all_teams' not in st.session_state:
        st.session_state.select_all_teams = False

    col_team, col_btn = st.sidebar.columns([3, 1])
    with col_team:
        st.markdown("**Team**")
    with col_btn:
        if st.button("All", help="Select all teams"):
            st.session_state.select_all_teams = True

    # Default to all teams (user can narrow down if needed)
    default_teams = available_teams
    selected_teams = st.sidebar.multiselect(
        "Team",
        options=available_teams,
        default=default_teams,
        label_visibility="collapsed"
    )
    # Reset flag after use
    if st.session_state.select_all_teams:
        st.session_state.select_all_teams = False

    # Option to show combined team average across seasons
    show_team_average = st.sidebar.checkbox(
        "Show Team Average (all seasons)",
        value=False,
        help="Combine data across all seasons for selected teams"
    )

    # Build list of selected datasets based on team + season
    if show_team_average:
        # Include all seasons for selected teams
        selected_datasets = [k for k in datasets.keys()
                           if any(k.startswith(t + ' ') for t in selected_teams)]
    else:
        # Filter by both team and season
        selected_datasets = [k for k in datasets.keys()
                           if any(k.startswith(t + ' ') for t in selected_teams)
                           and any(k.endswith(' ' + s) for s in selected_seasons)]

    all_players = sorted(all_data['player'].unique())
    selected_players = st.sidebar.multiselect(
        "Select Players (leave empty for all)",
        options=all_players,
        default=[]
    )

    # Analysis View selector (moved up for better UX)
    if 'sprs_tab' not in st.session_state:
        st.session_state.sprs_tab = "Szklanny Metrics (SPM)"

    tab_options = [
        "Szklanny Metrics (SPM)",
        "Quarter Analysis",
        "B2B Impact",
        "Player Breakdown",
        "Fatigue Proxies",
        "Predictive Model"
    ]

    st.sidebar.markdown("### 📊 Analysis View")
    selected_tab = st.sidebar.radio(
        "Select Analysis",
        options=tab_options,
        index=tab_options.index(st.session_state.sprs_tab),
        key="sprs_tab_selector",
        label_visibility="collapsed"
    )
    st.session_state.sprs_tab = selected_tab

    st.sidebar.markdown("---")

    b2b_filter = st.sidebar.radio(
        "Game Type",
        options=['All Games', 'B2B Only', 'Non-B2B Only']
    )

    outcome_filter = st.sidebar.radio(
        "Game Outcome",
        options=['All', 'Wins Only', 'Losses Only']
    )

    # Close games filter (derive margin from quarter data)
    close_games_filter = st.sidebar.checkbox(
        "Close Games Only (<10 pts margin)",
        value=False,
        help="Filter to games decided by less than 10 points (estimated from quarter totals)"
    )

    # ==========================================================================
    # POSITION ASSIGNMENT (stored in session state)
    # ==========================================================================
    if 'player_positions' not in st.session_state:
        st.session_state.player_positions = {}

    with st.sidebar.expander("Player Positions (for model features)"):
        st.caption("Assign positions for enhanced predictions")
        position_options = ["G", "F", "C", "G/F", "F/C"]

        # Show position selectors for top players by games
        top_players = all_data.groupby('player')['game_date'].nunique().nlargest(15).index.tolist()
        for player in top_players:
            current_pos = st.session_state.player_positions.get(player, "G")
            new_pos = st.selectbox(
                player,
                position_options,
                index=position_options.index(current_pos) if current_pos in position_options else 0,
                key=f"pos_{player}"
            )
            st.session_state.player_positions[player] = new_pos

    # Apply filters
    filtered_data = all_data[all_data['dataset'].isin(selected_datasets)]

    if selected_players:
        filtered_data = filtered_data[filtered_data['player'].isin(selected_players)]

    if b2b_filter == 'B2B Only':
        filtered_data = filtered_data[filtered_data['is_b2b'] == True]
    elif b2b_filter == 'Non-B2B Only':
        filtered_data = filtered_data[filtered_data['is_b2b'] == False]

    if outcome_filter == 'Wins Only':
        filtered_data = filtered_data[filtered_data['is_win'] == True]
    elif outcome_filter == 'Losses Only':
        filtered_data = filtered_data[filtered_data['is_win'] == False]

    # Apply close games filter
    if close_games_filter:
        # Estimate margin from quarter totals
        game_totals = filtered_data.groupby(['game_date', 'dataset']).agg({
            'pts': 'sum'
        }).reset_index()
        # This is a rough estimate - actual margin would need opponent data
        # For now, use pts variance as proxy for close games
        game_pts_std = filtered_data.groupby(['game_date', 'dataset'])['pts'].std().reset_index()
        game_pts_std.columns = ['game_date', 'dataset', 'pts_std']
        # Low variance in pts across quarters suggests closer game
        close_games = game_pts_std[game_pts_std['pts_std'] < game_pts_std['pts_std'].median()]
        filtered_data = filtered_data.merge(
            close_games[['game_date', 'dataset']],
            on=['game_date', 'dataset'],
            how='inner'
        )

    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        n_games = filtered_data['game_date'].nunique()
        st.metric("Total Games", f"{n_games:,}")

    with col2:
        n_players = filtered_data['player'].nunique()
        st.metric("Players", n_players)

    with col3:
        q1_data = filtered_data[filtered_data['qtr_num'] == 1]
        q1_fg = q1_data['fgm'].sum() / q1_data['fga'].sum() * 100 if q1_data['fga'].sum() > 0 else 0
        st.metric("Q1 FG%", f"{q1_fg:.1f}%")

    with col4:
        q4_data = filtered_data[filtered_data['qtr_num'] == 4]
        q4_fg = q4_data['fgm'].sum() / q4_data['fga'].sum() * 100 if q4_data['fga'].sum() > 0 else 0
        st.metric("Q4 FG%", f"{q4_fg:.1f}%")

    with col5:
        q4_change = q4_fg - q1_fg
        st.metric("Q4 Change", f"{q4_change:+.1f}%", delta_color="inverse")

    st.markdown("---")

    # Create visual tab-like buttons at top
    tab_cols = st.columns(6)
    for i, tab_name in enumerate(tab_options):
        with tab_cols[i]:
            if selected_tab == tab_name:
                st.markdown(f"<div style='background: rgba(74,144,217,0.3); padding: 8px; border-radius: 8px; text-align: center; border-bottom: 2px solid #4A90D9;'><small><b>{tab_name}</b></small></div>", unsafe_allow_html=True)
            else:
                if st.button(tab_name, key=f"tab_btn_{i}", use_container_width=True):
                    st.session_state.sprs_tab = tab_name
                    st.rerun()

    # ==========================================================================
    # TAB 1: Quarter Analysis
    # ==========================================================================
    if selected_tab == "Quarter Analysis":
        st.subheader("FG% by Quarter")

        col1, col2 = st.columns(2)

        with col1:
            quarter_data = []
            for dataset in filtered_data['dataset'].unique():
                ds_data = filtered_data[filtered_data['dataset'] == dataset]
                for q in [1, 2, 3, 4]:
                    q_data = ds_data[ds_data['qtr_num'] == q]
                    if q_data['fga'].sum() > 0:
                        fg_pct = q_data['fgm'].sum() / q_data['fga'].sum() * 100
                        quarter_data.append({
                            'Dataset': dataset,
                            'Quarter': f'Q{q}',
                            'FG%': fg_pct
                        })

            if quarter_data:
                quarter_df = pd.DataFrame(quarter_data)
                fig = px.bar(quarter_df, x='Quarter', y='FG%', color='Dataset',
                            barmode='group', title='FG% by Quarter')
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            change_data = []
            for dataset in filtered_data['dataset'].unique():
                ds_data = filtered_data[filtered_data['dataset'] == dataset]
                q1 = ds_data[ds_data['qtr_num'] == 1]
                q4 = ds_data[ds_data['qtr_num'] == 4]
                if q1['fga'].sum() > 0 and q4['fga'].sum() > 0:
                    q1_fg = q1['fgm'].sum() / q1['fga'].sum() * 100
                    q4_fg = q4['fgm'].sum() / q4['fga'].sum() * 100
                    change_data.append({
                        'Dataset': dataset,
                        'Q4 Change': q4_fg - q1_fg
                    })

            if change_data:
                change_df = pd.DataFrame(change_data)
                fig = px.bar(change_df, x='Dataset', y='Q4 Change',
                            color='Q4 Change', color_continuous_scale=['red', 'gray', 'green'],
                            title='Q4 FG% Change from Q1')
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 2: B2B Impact
    # ==========================================================================
    if selected_tab == "B2B Impact":
        st.subheader("Back-to-Back Game Analysis")

        col1, col2 = st.columns(2)

        with col1:
            b2b_data = []
            for dataset in all_data['dataset'].unique():
                if dataset not in selected_datasets:
                    continue
                ds_data = all_data[all_data['dataset'] == dataset]
                for is_b2b in [False, True]:
                    subset = ds_data[ds_data['is_b2b'] == is_b2b]
                    if subset['fga'].sum() > 0:
                        fg_pct = subset['fgm'].sum() / subset['fga'].sum() * 100
                        b2b_data.append({
                            'Dataset': dataset,
                            'Game Type': 'B2B' if is_b2b else 'Normal',
                            'FG%': fg_pct
                        })

            if b2b_data:
                b2b_df = pd.DataFrame(b2b_data)
                fig = px.bar(b2b_df, x='Dataset', y='FG%', color='Game Type',
                            barmode='group', title='FG% by Game Type')
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            b2b_q4_data = []
            for dataset in all_data['dataset'].unique():
                if dataset not in selected_datasets:
                    continue
                ds_data = all_data[all_data['dataset'] == dataset]
                for is_b2b in [False, True]:
                    subset = ds_data[ds_data['is_b2b'] == is_b2b]
                    q1 = subset[subset['qtr_num'] == 1]
                    q4 = subset[subset['qtr_num'] == 4]
                    if q1['fga'].sum() > 0 and q4['fga'].sum() > 0:
                        q1_fg = q1['fgm'].sum() / q1['fga'].sum() * 100
                        q4_fg = q4['fgm'].sum() / q4['fga'].sum() * 100
                        b2b_q4_data.append({
                            'Dataset': dataset,
                            'Game Type': 'B2B' if is_b2b else 'Normal',
                            'Q4 Change': q4_fg - q1_fg
                        })

            if b2b_q4_data:
                b2b_q4_df = pd.DataFrame(b2b_q4_data)
                fig = px.bar(b2b_q4_df, x='Dataset', y='Q4 Change', color='Game Type',
                            barmode='group', title='Q4 Change on B2B vs Normal Games')
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### B2B Game Breakdown")
        b2b_counts = []
        for dataset in selected_datasets:
            if dataset in datasets:
                ds_data = datasets[dataset]
                total = ds_data['game_date'].nunique()
                b2b = ds_data[ds_data['is_b2b'] == True]['game_date'].nunique()
                b2b_counts.append({
                    'Dataset': dataset,
                    'Total Games': total,
                    'B2B Games': b2b,
                    'B2B %': f"{b2b / total * 100:.1f}%" if total > 0 else "0%"
                })

        if b2b_counts:
            st.dataframe(pd.DataFrame(b2b_counts), use_container_width=True)

    # ==========================================================================
    # TAB 3: Player Breakdown
    # ==========================================================================
    if selected_tab == "Player Breakdown":
        st.subheader("Player-Level Fatigue Analysis")

        player_data = []
        for player in filtered_data['player'].unique():
            p_data = filtered_data[filtered_data['player'] == player]
            games = p_data['game_date'].nunique()
            if games < 5:
                continue

            q1 = p_data[p_data['qtr_num'] == 1]
            q4 = p_data[p_data['qtr_num'] == 4]

            if q1['fga'].sum() > 0 and q4['fga'].sum() > 0:
                q1_fg = q1['fgm'].sum() / q1['fga'].sum() * 100
                q4_fg = q4['fgm'].sum() / q4['fga'].sum() * 100
                player_data.append({
                    'Player': player,
                    'Games': games,
                    'Q1 FG%': round(q1_fg, 1),
                    'Q4 FG%': round(q4_fg, 1),
                    'Q4 Change': round(q4_fg - q1_fg, 1),
                    'Avg PF': round(p_data['pf'].mean(), 2),
                    'Avg STL': round(p_data['stl'].mean(), 2),
                    'Avg TOV': round(p_data['tov'].mean(), 2)
                })

        if player_data:
            player_df = pd.DataFrame(player_data).sort_values('Q4 Change')

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(player_df.head(15), x='Q4 Change', y='Player', orientation='h',
                            color='Q4 Change', color_continuous_scale=['red', 'gray', 'green'],
                            title='Top 15 Players by Q4 FG% Change')
                fig.update_layout(**get_chart_layout(), height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(player_df, x='Q1 FG%', y='Q4 FG%',
                                size='Games', hover_name='Player',
                                color='Q4 Change', color_continuous_scale=['red', 'gray', 'green'],
                                title='Q1-3 Avg vs Q4 FG% (size = games played)')
                fig.add_trace(go.Scatter(x=[30, 70], y=[30, 70], mode='lines',
                                        line=dict(dash='dash', color='white'),
                                        name='No Change Line'))
                fig.update_layout(**get_chart_layout(), height=500)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Detailed Player Stats")
            st.dataframe(player_df, use_container_width=True)

    # ==========================================================================
    # TAB 4: Fatigue Proxies
    # ==========================================================================
    if selected_tab == "Fatigue Proxies":
        st.subheader("Fatigue Proxy Metrics")
        st.markdown("""
        **Fatigue Proxies:** Research shows these metrics indicate fatigue:
        - **Personal Fouls (PF):** Increase with fatigue
        - **Steals (STL):** Decrease with fatigue
        - **Turnovers (TOV):** Increase with fatigue
        """)

        col1, col2 = st.columns(2)

        with col1:
            proxy_data = []
            for dataset in filtered_data['dataset'].unique():
                ds_data = filtered_data[filtered_data['dataset'] == dataset]
                for q in [1, 2, 3, 4]:
                    q_data = ds_data[ds_data['qtr_num'] == q]
                    if len(q_data) > 0:
                        proxy_data.append({
                            'Dataset': dataset,
                            'Quarter': f'Q{q}',
                            'PF': q_data['pf'].mean(),
                            'STL': q_data['stl'].mean(),
                            'TOV': q_data['tov'].mean()
                        })

            if proxy_data:
                proxy_df = pd.DataFrame(proxy_data)

                fig = make_subplots(rows=1, cols=3, subplot_titles=['Fouls', 'Steals', 'Turnovers'])

                for i, metric in enumerate(['PF', 'STL', 'TOV']):
                    for dataset in proxy_df['Dataset'].unique():
                        subset = proxy_df[proxy_df['Dataset'] == dataset]
                        fig.add_trace(
                            go.Scatter(x=subset['Quarter'], y=subset[metric],
                                      mode='lines+markers', name=dataset,
                                      showlegend=(i == 0)),
                            row=1, col=i+1
                        )

                fig.update_layout(**get_chart_layout(), height=400, title='Fatigue Proxies by Quarter')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            change_data = []
            for dataset in filtered_data['dataset'].unique():
                ds_data = filtered_data[filtered_data['dataset'] == dataset]
                q1 = ds_data[ds_data['qtr_num'] == 1]
                q4 = ds_data[ds_data['qtr_num'] == 4]
                if len(q1) > 0 and len(q4) > 0:
                    change_data.append({
                        'Dataset': dataset,
                        'PF Change': q4['pf'].mean() - q1['pf'].mean(),
                        'STL Change': q4['stl'].mean() - q1['stl'].mean(),
                        'TOV Change': q4['tov'].mean() - q1['tov'].mean()
                    })

            if change_data:
                change_df = pd.DataFrame(change_data)
                melt_df = change_df.melt(id_vars='Dataset', var_name='Metric', value_name='Change')

                fig = px.bar(melt_df, x='Metric', y='Change', color='Dataset',
                            barmode='group', title='Q4 Change in Fatigue Proxies')
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 5: Original Predictive Model
    # ==========================================================================
    if selected_tab == "Predictive Model":
        st.subheader("Workload Risk Assessment")
        st.markdown("""
        **Avg Risk Score (0-100):** Composite workload score combining back-to-back games, rest days,
        recent minutes load, age factors, and consecutive heavy games. Higher scores indicate greater
        physiological stress that may impact Q4 performance.
        """)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Use filtered data (respects team selection) instead of hardcoded 'Sixers'
        model_data = filtered_data.copy()

        if len(model_data) < 100:
            st.warning(f"Not enough data for model training ({len(model_data)} rows). Select more teams or seasons.")
        else:
            # Add is_b2b column if missing
            if 'is_b2b' not in model_data.columns:
                model_data['is_b2b'] = False

            # ================================================================
            # IMPROVED FATIGUE MODEL - Use Q1-Q3 avg vs Q4 (more stable)
            # Key insight: Single-quarter FG% is VERY noisy. Using avg of Q1-Q3
            # vs Q4 reduces noise significantly.
            # ================================================================
            minutes_col = 'minutes' if 'minutes' in model_data.columns else 'total_minutes' if 'total_minutes' in model_data.columns else None

            # Get Q1-Q3 combined performance (baseline) - ALL STATS
            q1_q3_agg = {'fgm': 'sum', 'fga': 'sum', 'pts': 'sum'}
            # Add hustle stats if available
            for col in ['trb', 'ast', 'stl', 'blk', 'tov']:
                if col in model_data.columns:
                    q1_q3_agg[col] = 'sum'

            q1_q3 = model_data[model_data['qtr_num'].isin([1, 2, 3])].groupby(['game_date', 'player']).agg(q1_q3_agg).reset_index()
            q1_q3['early_fg'] = np.where(q1_q3['fga'] >= 3, q1_q3['fgm'] / q1_q3['fga'] * 100, np.nan)
            q1_q3['early_pts'] = q1_q3['pts']
            q1_q3['early_fga'] = q1_q3['fga']
            # Hustle stats per quarter average (divide by 3 quarters)
            for col in ['trb', 'ast', 'stl', 'blk', 'tov']:
                if col in q1_q3.columns:
                    q1_q3[f'early_{col}'] = q1_q3[col] / 3.0  # Per quarter rate

            # Get Q4 performance - ALL STATS
            q4_agg = {'fgm': 'sum', 'fga': 'sum', 'pts': 'sum'}
            for col in ['trb', 'ast', 'stl', 'blk', 'tov']:
                if col in model_data.columns:
                    q4_agg[col] = 'sum'

            q4 = model_data[model_data['qtr_num'] == 4].groupby(['game_date', 'player']).agg(q4_agg).reset_index()
            q4['q4_fg'] = np.where(q4['fga'] >= 2, q4['fgm'] / q4['fga'] * 100, np.nan)
            q4['q4_pts'] = q4['pts']
            q4['q4_fga'] = q4['fga']
            # Q4 hustle stats (already 1 quarter)
            for col in ['trb', 'ast', 'stl', 'blk', 'tov']:
                if col in q4.columns:
                    q4[f'q4_{col}'] = q4[col]

            # Get game-level aggregates
            agg_dict = {'is_b2b': 'first', 'fgm': 'sum', 'fga': 'sum'}
            if minutes_col:
                agg_dict[minutes_col] = 'sum'
            games = model_data.groupby(['game_date', 'player']).agg(agg_dict).reset_index()

            if minutes_col and minutes_col in games.columns:
                games['game_minutes'] = games[minutes_col]
            else:
                games['game_minutes'] = 30

            # Build merge columns list dynamically
            q1_q3_merge_cols = ['game_date', 'player', 'early_fg', 'early_pts', 'early_fga']
            q4_merge_cols = ['game_date', 'player', 'q4_fg', 'q4_pts', 'q4_fga']
            for col in ['trb', 'ast', 'stl', 'blk', 'tov']:
                if f'early_{col}' in q1_q3.columns:
                    q1_q3_merge_cols.append(f'early_{col}')
                if f'q4_{col}' in q4.columns:
                    q4_merge_cols.append(f'q4_{col}')

            # Merge performance data
            games = games.merge(q1_q3[q1_q3_merge_cols], on=['game_date', 'player'], how='left')
            games = games.merge(q4[q4_merge_cols], on=['game_date', 'player'], how='left')

            # FILTER: Only games with meaningful shot attempts (reduces noise dramatically)
            # Increased Q4 FGA requirement to reduce single-game noise
            games = games[(games['early_fga'] >= 6) & (games['q4_fga'] >= 3)]
            games = games.dropna(subset=['early_fg', 'q4_fg'])

            # Single-game Q4 drop (raw, noisy)
            games['fg_change_raw'] = games['q4_fg'] - games['early_fg']

            # Compute deltas for ALL hustle stats (Q4 - Q1-3 avg per quarter)
            for col in ['trb', 'ast', 'stl', 'blk', 'tov']:
                if f'early_{col}' in games.columns and f'q4_{col}' in games.columns:
                    games[f'{col}_change'] = games[f'q4_{col}'] - games[f'early_{col}']

            # Sort for rolling calculations
            games = games.sort_values(['player', 'game_date']).reset_index(drop=True)

            # TARGET: 7-game rolling average Q4 drop (more stable, reduces noise)
            # Larger window captures true fatigue patterns better
            games['fg_change'] = games.groupby('player')['fg_change_raw'].transform(
                lambda x: x.rolling(7, min_periods=4).mean()
            )

            # Add age
            if 'age' in model_data.columns:
                age_map = model_data.groupby('player')['age'].first().to_dict()
                games['age'] = games['player'].map(age_map).fillna(27)
            else:
                games['age'] = 27

            # ================================================================
            # ADD WORKLOAD FEATURES (cumulative fatigue matters!)
            # ================================================================
            # Rolling 5-game minutes average (recent workload)
            games['rolling_minutes_5g'] = games.groupby('player')['game_minutes'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            ).fillna(30)

            # Rolling 5-game FG% change (recent fatigue pattern)
            games['rolling_fg_change_5g'] = games.groupby('player')['fg_change'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            ).fillna(0)

            # Days since last game (rest)
            games['days_rest'] = games.groupby('player')['game_date'].diff().dt.days.fillna(3)
            games['days_rest'] = games['days_rest'].clip(0, 7)  # Cap at 7

            # Consecutive high-minute games (shift must be inside transform for per-player)
            games['high_minutes'] = (games['game_minutes'] > 35).astype(int)
            games['consec_high_min'] = games.groupby('player')['high_minutes'].transform(
                lambda x: x.rolling(3, min_periods=1).sum().shift(1).fillna(0)
            )

            # ================================================================
            # RULE-BASED RISK ASSESSMENT
            # ================================================================

            # Feature set with workload history and age interactions
            games['minutes_norm'] = games['game_minutes'] / 48.0
            games['workload_5g'] = games['rolling_minutes_5g'] / 40.0
            games['is_b2b_num'] = games['is_b2b'].astype(float)
            games['rest_factor'] = (3 - games['days_rest']) / 3.0
            games['cumulative_load'] = games['consec_high_min'] / 3.0

            # AGE INTERACTION FEATURES - amplifies fatigue risk for older players under load
            # age_penalty = max(0, age - 25) * (minutes_avg_last5 / 30)
            games['age_penalty'] = games.apply(
                lambda x: max(0, x['age'] - 25) * (x['rolling_minutes_5g'] / 30.0), axis=1
            )
            # Age × B2B interaction - older players hurt more by B2B
            games['age_b2b_interact'] = games.apply(
                lambda x: max(0, x['age'] - 28) * float(x['is_b2b']), axis=1
            )
            # Age × rest interaction - older players need more recovery
            games['age_rest_interact'] = games.apply(
                lambda x: max(0, x['age'] - 28) * max(0, 2 - x['days_rest']), axis=1
            )

            # Required columns for risk calculation
            required_cols = ['minutes_norm', 'workload_5g', 'is_b2b_num', 'rest_factor',
                           'age_penalty', 'cumulative_load']

            # Clean data
            model_df = games.dropna(subset=required_cols)

            if len(model_df) < 50:
                st.warning(f"Not enough data ({len(model_df)} games). Need 50+ games.")
            else:
                # ================================================================
                # RULE-BASED RISK SCORE (honest - based on sports science, not ML)
                # ================================================================
                # Weights based on sports science research on fatigue factors
                RISK_WEIGHTS = {
                    'b2b': 35,           # Back-to-back: high impact
                    'low_rest': 30,      # <2 days rest: high impact
                    'high_minutes': 20,  # High recent workload
                    'age_load': 15,      # Age under high load
                }

                # Compute rule-based risk score (0-100)
                model_df['risk_b2b'] = model_df['is_b2b_num'] * RISK_WEIGHTS['b2b']
                model_df['risk_rest'] = model_df['rest_factor'].clip(0, 1) * RISK_WEIGHTS['low_rest']
                model_df['risk_minutes'] = model_df['workload_5g'].clip(0, 1) * RISK_WEIGHTS['high_minutes']
                model_df['risk_age'] = (model_df['age_penalty'] / 5).clip(0, 1) * RISK_WEIGHTS['age_load']

                model_df['fatigue_risk'] = (
                    model_df['risk_b2b'] + model_df['risk_rest'] + model_df['risk_minutes'] +
                    model_df['risk_age']
                ).clip(0, 100)

                # Risk categories
                model_df['risk_category'] = pd.cut(model_df['fatigue_risk'],
                                                bins=[0, 25, 50, 75, 100],
                                                labels=['Low', 'Moderate', 'High', 'Very High'])

                # Use model_df for visualizations
                games = model_df

                # Show sample statistics instead of fake R²
                col1, col2, col3, col4 = st.columns(4)
                avg_risk = model_df['fatigue_risk'].mean()
                high_risk_pct = (model_df['fatigue_risk'] > 50).mean() * 100
                b2b_pct = model_df['is_b2b_num'].mean() * 100
                col1.metric("Avg Risk Score", f"{avg_risk:.1f}",
                           help="Average workload risk (0-100)")
                col2.metric("High Risk Games", f"{high_risk_pct:.1f}%",
                           help="% of games with risk > 50")
                col3.metric("B2B Games", f"{b2b_pct:.1f}%",
                           help="% of games that are back-to-backs")
                col4.metric("Sample Size", f"{len(model_df)} games",
                           help="Games analyzed")

                # ================================================================
                # PLAYER RISK SELECTOR
                # ================================================================
                st.markdown("### 👤 Player Risk Profile")
                available_players = sorted(games['player'].unique())
                selected_player = st.selectbox("Select Player", available_players, key="risk_player_select")

                if selected_player:
                    player_games = games[games['player'] == selected_player]
                    if len(player_games) > 0:
                        latest_game = player_games.iloc[-1]
                        avg_risk = player_games['fatigue_risk'].mean()
                        max_risk = player_games['fatigue_risk'].max()
                        high_risk_games = (player_games['fatigue_risk'] > 50).sum()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Current Risk", f"{latest_game['fatigue_risk']:.0f}",
                                   help="Most recent game risk score")
                        col2.metric("Avg Risk", f"{avg_risk:.1f}",
                                   help="Average risk across all games")
                        col3.metric("Peak Risk", f"{max_risk:.0f}",
                                   help="Highest risk game")
                        col4.metric("High Risk Games", f"{high_risk_games}/{len(player_games)}",
                                   help="Games with risk > 50")

                        # Show risk breakdown for latest game
                        st.markdown("**Latest Game Risk Breakdown:**")
                        breakdown_cols = st.columns(4)
                        breakdown_cols[0].metric("B2B", f"{latest_game.get('risk_b2b', 0):.0f}/35",
                                                help="Back-to-back game penalty")
                        breakdown_cols[1].metric("Rest", f"{latest_game.get('risk_rest', 0):.0f}/30",
                                                help="Less than 2 days rest")
                        breakdown_cols[2].metric("Workload", f"{latest_game.get('risk_minutes', 0):.0f}/20",
                                                help="5-game average minutes")
                        breakdown_cols[3].metric("Age×Load", f"{latest_game.get('risk_age', 0):.0f}/15",
                                                help="Age penalty under high workload")

                st.markdown("---")
                st.markdown("### 📊 Risk Factor Weights")
                importance_df = pd.DataFrame({
                    'Factor': ['Back-to-Back', 'Low Rest (<2 days)', 'High Workload (5g)', 'Age × Load'],
                    'Weight': [35, 30, 20, 15]
                }).sort_values('Weight', ascending=True)

                fig = px.bar(importance_df, x='Weight', y='Factor', orientation='h',
                            color='Weight', color_continuous_scale='Reds',
                            title='Fatigue Risk Factor Weights (Sports Science Based)')
                fig.update_layout(**get_chart_layout(), height=350)
                fig.update_traces(text=[f"{w:.1f}%" for w in importance_df['Weight']], textposition='inside')
                st.plotly_chart(fig, use_container_width=True)

                # Risk distribution
                st.markdown("### Risk Distribution in Data")
                risk_counts = games['risk_category'].value_counts().sort_index()
                fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                             title='Fatigue Risk Categories',
                             color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444', '#7C3AED'])
                fig2.update_layout(**get_chart_layout(), height=350)
                st.plotly_chart(fig2, use_container_width=True)

                # ================================================================
                # Q4 STAT DELTAS - All hustle stats
                # ================================================================
                st.markdown("### 📉 Q4 Performance Drop by Stat")
                st.markdown("*Average change from Q1-3 per-quarter rate to Q4 (negative = decline)*")

                # Compute average deltas
                stat_deltas = {'Stat': [], 'Avg Q4 Delta': [], 'Direction': []}

                # FG%
                if 'fg_change_raw' in games.columns:
                    fg_delta = games['fg_change_raw'].mean()
                    stat_deltas['Stat'].append('FG%')
                    stat_deltas['Avg Q4 Delta'].append(fg_delta)
                    stat_deltas['Direction'].append('⬇️ Drop' if fg_delta < 0 else '⬆️ Rise')

                # Hustle stats
                stat_labels = {'trb': 'Rebounds', 'ast': 'Assists', 'stl': 'Steals', 'blk': 'Blocks', 'tov': 'Turnovers'}
                for col, label in stat_labels.items():
                    if f'{col}_change' in games.columns:
                        delta = games[f'{col}_change'].mean()
                        stat_deltas['Stat'].append(label)
                        stat_deltas['Avg Q4 Delta'].append(delta)
                        # For turnovers, increase is bad
                        if col == 'tov':
                            stat_deltas['Direction'].append('⬆️ Bad' if delta > 0 else '⬇️ Good')
                        else:
                            stat_deltas['Direction'].append('⬇️ Drop' if delta < 0 else '⬆️ Rise')

                if stat_deltas['Stat']:
                    delta_df = pd.DataFrame(stat_deltas)
                    # Color based on positive/negative
                    delta_df['Color'] = delta_df['Avg Q4 Delta'].apply(lambda x: 'red' if x < 0 else 'green')

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig3 = px.bar(delta_df, x='Stat', y='Avg Q4 Delta',
                                     color='Avg Q4 Delta',
                                     color_continuous_scale=['#EF4444', '#F59E0B', '#10B981'],
                                     title='Average Q4 Change by Stat (per quarter rate)')
                        fig3.update_layout(**get_chart_layout(), height=350)
                        fig3.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                        st.plotly_chart(fig3, use_container_width=True)

                    with col2:
                        st.markdown("**Summary Table**")
                        display_df = delta_df[['Stat', 'Avg Q4 Delta', 'Direction']].copy()
                        display_df['Avg Q4 Delta'] = display_df['Avg Q4 Delta'].apply(lambda x: f"{x:+.2f}")
                        st.dataframe(display_df, hide_index=True, use_container_width=True)

            # Interactive Prediction Section
            st.markdown("---")
            st.markdown("### 🎯 Predict Fatigue Risk for a Player")
            st.markdown("Adjust the inputs to estimate fatigue risk for Q4 performance drop.")

            # Define prediction function for interactive UI
            def calculate_fatigue_risk_ui(minutes, age, is_b2b):
                """Simple domain-driven estimate for interactive prediction."""
                minutes_normalized = min(minutes / 48.0, 1.0)
                base_risk = (minutes_normalized ** 1.5) * 0.6
                if age >= 25:
                    age_multiplier = 1.0 + (age - 25) * 0.04
                else:
                    age_multiplier = 1.0 - (25 - age) * 0.02
                age_multiplier = max(0.8, min(age_multiplier, 1.6))
                b2b_multiplier = 1.35 if is_b2b else 1.0
                return min(base_risk * age_multiplier * b2b_multiplier * 100, 100)

            col1, col2, col3 = st.columns(3)

            with col1:
                pred_minutes = st.slider("Minutes This Game", 10.0, 48.0, 32.0, 1.0, key="minutes_fatigue")
            with col2:
                pred_age_fatigue = st.slider("Player Age", 19, 40, 27, key="age_fatigue")
            with col3:
                pred_b2b_fatigue = st.selectbox("Back-to-Back Game?", [False, True], key="b2b_fatigue")

            # Always show prediction (no button needed)
            fatigue_prob = calculate_fatigue_risk_ui(pred_minutes, pred_age_fatigue, pred_b2b_fatigue)

            st.markdown("---")

            # Show calculation breakdown
            minutes_component = (min(pred_minutes / 48.0, 1.0) ** 1.5) * 60
            if pred_age_fatigue >= 25:
                age_mult = 1.0 + (pred_age_fatigue - 25) * 0.04
            else:
                age_mult = 1.0 - (25 - pred_age_fatigue) * 0.02
            age_mult = max(0.8, min(age_mult, 1.6))
            b2b_mult = 1.35 if pred_b2b_fatigue else 1.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Minutes Impact", f"{minutes_component:.1f}%")
            col2.metric("Age Multiplier", f"{age_mult:.2f}x")
            col3.metric("B2B Multiplier", f"{b2b_mult:.2f}x")

            # Color-coded risk display
            if fatigue_prob >= 75:
                risk_color = "🔴"
                risk_level = "Very High"
            elif fatigue_prob >= 50:
                risk_color = "🟠"
                risk_level = "High"
            elif fatigue_prob >= 25:
                risk_color = "🟡"
                risk_level = "Moderate"
            else:
                risk_color = "🟢"
                risk_level = "Low"

            col4.metric(f"{risk_color} Fatigue Risk", f"{fatigue_prob:.0f}%", delta=risk_level)

    # ==========================================================================
    # TAB 6: Szklanny Performance Metric (SPM)
    # ==========================================================================
    if selected_tab == "Szklanny Metrics (SPM)":
        st.subheader("Szklanny Performance Metric (SPM)")

        st.markdown("""
        ## Single Unified Late-Game Resilience Score

        **SPM** measures how well a player maintains performance from Q1-3 (avg) to Q4.
        Scaled from **-10 to +10** for easy interpretation.

        | Component | Weight | Description |
        |-----------|--------|-------------|
        | **FG% Change** | ~30% | Shooting efficiency maintenance |
        | **Points Change** | ~30% | Scoring output maintenance |
        | **TOV Resilience** | ~25% | Fewer turnovers late (inverted: -TOV change) |
        | **Blocks Change** | ~10% | Rim protection maintenance |
        | **Steals Change** | ~5% | Active defense (low impact per PCA) |

        **Interpretation:** SPM > +3 = Elite | SPM ~ 0 = Average | SPM < -3 = Fades late | SPM < -5 = Severe

        *Compares avg Q1-3 performance to Q4. Weights derived from PCA when n≥10, otherwise manual fallback.*
        """)

        # League benchmarks info
        col1, col2, col3, col4 = st.columns(4)
        col1.info(f"League avg Q4 FG% drop: **{LEAGUE_BENCHMARKS['avg_q4_fg_drop']}%**")
        col2.info(f"Elite SPM: **> +{LEAGUE_BENCHMARKS['elite_spm']}**")
        col3.info(f"Fade threshold: **< {LEAGUE_BENCHMARKS['fatigue_spm']}**")
        col4.info(f"Reliable sample: **≥{LEAGUE_BENCHMARKS['min_games_reliable']} games**")

        # Learn More expanders
        with st.expander("Learn More: About SPM (Szklanny Performance-resilience Metric)"):
            st.markdown("""
            **SPM** is a consolidated single metric measuring late-game performance maintenance.

            ### Why One Metric?
            Previously we had SLFI (fatigue) and SLIS (impact) which overlapped and confused interpretation.
            SPM combines the key signals into one score with data-driven (PCA) or validated manual weights.

            ### Components (all oriented: higher = more resilient)
            | Component | Manual Weight | PCA Validation |
            |-----------|---------------|----------------|
            | **FG% Change** | 30% | ~0.60 loading (dominant) |
            | **Points Change** | 30% | ~0.61 loading (dominant) |
            | **TOV Resilience** | 25% | ~0.35 loading (meaningful) |
            | **Blocks Change** | 10% | ~0.37 loading (meaningful) |
            | **Steals Change** | 5% | ~0.01 loading (near-zero, noisy) |

            ### Key Design Decisions
            - **TOV is inverted**: `tov_resil = TOV_Q1-3avg - TOV_Q4` so MORE turnovers in Q4 = LOWER score
            - **Steals minimized**: PCA shows STL has near-zero loading; kept at 5% for completeness
            - **PCA-first**: Uses data-driven weights when n≥10, falls back to manual otherwise

            ### Research Basis
            - FG% typically drops 1-2% in Q4 due to fatigue (Sampaio et al., 2015)
            - Turnovers increase with accumulated fatigue and cognitive load (Ben Abdelkrim et al., 2007)
            - PCA on simulated data shows PC1 explains ~47% variance with FG/PTS dominant

            ### Interpretation (scaled -10 to +10)
            - **SPM > +3**: Elite late-game resilience (top 10%)
            - **SPM > +1**: Good resilience (above average)
            - **SPM ≈ 0**: Average (typical Q4 performance)
            - **SPM < -3**: Significant late-game fade
            - **SPM < -5**: Severe concern (needs load management)
            """)

        with st.expander("Learn More: About the Prediction Models"):
            st.markdown("""
            ### Classification Model
            Predicts probability of meaningful fatigue (SPM < -3) using logistic regression.
            - **Physiological Risk Floor (PRF)**: Ensures a 40yo on B2B at 40 min can NEVER be "low risk"
            - Final risk = max(ML prediction, PRF)

            ### Regression Model (Ridge)
            Predicts actual SPM value using Ridge regression with 8 features:
            1. Last game SPM
            2. Average SPM (last 5 games)
            3. Average minutes (last 5 games)
            4. Player age
            5. Age-adjusted load (minutes × age penalty)
            6. Age × B2B interaction
            7. Recovery penalty (rest days × age)
            8. Effort Index (STL+BLK-PF-TOV trend)

            ### Random Forest Model (with Bootstrap CI)
            Alternative model using 100 decision trees for ensemble prediction.
            - Provides **confidence intervals** from tree ensemble variance
            - Better captures non-linear relationships
            - Feature importance shows relative predictive power

            ### Prediction Explanations
            Clear breakdown of what's driving each prediction:
            - **Positive Factors** = conditions supporting good performance
            - **Risk Factors** = conditions suggesting fatigue concern
            """)

        st.markdown("---")

        # Calculate both metrics (using NBA-wide benchmarks for normalization)
        with st.spinner("Calculating Szklanny Metrics..."):
            metrics_data = calculate_szklanny_metrics(filtered_data, league_benchmarks=league_benchmarks)

        # Edge case handling
        if len(metrics_data) == 0:
            st.warning("Not enough data to calculate metrics. Try selecting more teams or removing filters.")
        elif len(metrics_data) < LEAGUE_BENCHMARKS['min_games_warning']:
            st.warning(f"Low data warning: Only {len(metrics_data)} records found. Metrics may be unreliable (need {LEAGUE_BENCHMARKS['min_games_reliable']}+ for reliable results).")
        else:
            # Export button at the top
            export_cols = ['player', 'game_date', 'spm', 'fg_change', 'pts_change', 'tov_resil',
                          'blk_change', 'stl_change', 'minutes', 'is_b2b']
            available_cols = [c for c in export_cols if c in metrics_data.columns]
            export_df = metrics_data[available_cols].copy()
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Export SPM Data (CSV)",
                data=csv,
                file_name="spm_metrics_export.csv",
                mime="text/csv",
                help="Download player SPM metrics for further analysis"
            )

            # Player selector
            metric_players = sorted(metrics_data['player'].unique())
            selected_player = st.selectbox("Select Player for Detailed View", metric_players, key="szklanny_player")

            player_data = metrics_data[metrics_data['player'] == selected_player].sort_values('game_date')

            # Get weight source and variance explained for display
            weight_source = player_data['spm_weight_source'].iloc[0] if 'spm_weight_source' in player_data.columns else "Manual weights"
            var_explained = player_data['spm_var_explained'].iloc[0] if 'spm_var_explained' in player_data.columns else None

            # Summary metrics - show SPM (consolidated metric)
            st.markdown("### Player Summary")

            # Show weight source indicator with variance explained
            if "PCA" in weight_source and var_explained is not None and not np.isnan(var_explained):
                st.success(f"**Weights:** {weight_source}")
                # Show weight breakdown
                w_fg = player_data['spm_weight_fg'].iloc[0] * 100 if 'spm_weight_fg' in player_data.columns else 30
                w_pts = player_data['spm_weight_pts'].iloc[0] * 100 if 'spm_weight_pts' in player_data.columns else 30
                w_tov = player_data['spm_weight_tov'].iloc[0] * 100 if 'spm_weight_tov' in player_data.columns else 25
                w_blk = player_data['spm_weight_blk'].iloc[0] * 100 if 'spm_weight_blk' in player_data.columns else 10
                w_stl = player_data['spm_weight_stl'].iloc[0] * 100 if 'spm_weight_stl' in player_data.columns else 5
                st.caption(f"FG {w_fg:.0f}% | PTS {w_pts:.0f}% | TOV {w_tov:.0f}% | BLK {w_blk:.0f}% | STL {w_stl:.0f}%")
            else:
                st.info(f"**Weights:** {weight_source}")

            # Compute stats with confidence intervals
            avg_spm = player_data['spm'].mean()
            pct_spm_positive = (player_data['spm'] > 0).mean() * 100
            games_played = len(player_data)

            # Confidence interval
            spm_mean, spm_ci_low, spm_ci_high = compute_confidence_interval(player_data['spm'].values)

            # Low-n warning
            n_warning = get_sample_size_warning(games_played, LEAGUE_BENCHMARKS['min_games_reliable'])

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric(
                "Avg SPM",
                f"{avg_spm:+.2f}",
                delta="Resilient" if avg_spm > 0 else "Fades",
                help=f"CI: [{spm_ci_low:.2f}, {spm_ci_high:.2f}]. Higher = better late-game performance."
            )
            col2.metric(
                "% Games Resilient",
                f"{pct_spm_positive:.0f}%",
                help="Percentage of games where SPM > 0 (maintained performance)"
            )

            # Show component breakdown
            avg_fg_contrib = player_data.get('spm_contrib_fg_change', player_data.get('spm_contrib_fg_change', pd.Series([0]))).mean()
            avg_pts_contrib = player_data.get('spm_contrib_pts_change', pd.Series([0])).mean()
            avg_tov_contrib = player_data.get('spm_contrib_tov_resil', pd.Series([0])).mean()

            col3.metric(
                "FG+PTS Contrib",
                f"{(avg_fg_contrib + avg_pts_contrib):+.2f}",
                help="Combined contribution from FG% and points changes"
            )
            col4.metric(
                "TOV Contrib",
                f"{avg_tov_contrib:+.2f}",
                delta="Good ball security" if avg_tov_contrib > 0 else "TOV issues",
                help="Contribution from turnover resilience (fewer late TOV = positive)"
            )
            col5.metric(
                "Games",
                games_played,
                help=f"Minimum {LEAGUE_BENCHMARKS['min_games_reliable']} games recommended"
            )

            # Low sample warning
            if n_warning:
                st.warning(f"**{n_warning}**: Metrics may be unreliable. CI: [{spm_ci_low:.2f}, {spm_ci_high:.2f}]")

            # Player type classification (simplified with single SPM)
            if avg_spm > 0.5:
                player_type = "Elite Resilience"
                player_color = "green"
            elif avg_spm > 0:
                player_type = "Above Average"
                player_color = "lightgreen"
            elif avg_spm > -0.5:
                player_type = "Average"
                player_color = "gray"
            else:
                player_type = "Fades Late"
                player_color = "red"

            st.metric("Player Type", player_type, help="Classification based on SPM score")

            st.markdown("---")

            # Chart 1: SPM over time
            st.markdown("### SPM Over Time")

            player_data['spm_rolling'] = player_data['spm'].rolling(5, min_periods=1).mean()

            fig = go.Figure()

            # SPM per game
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['spm'],
                mode='lines+markers', name='SPM',
                line=dict(color='#4ECDC4', width=2), marker=dict(size=6)
            ))

            # 5-game rolling average
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['spm_rolling'],
                mode='lines', name='SPM 5-Game Avg',
                line=dict(color='#FFD93D', width=3, dash='dash')
            ))

            # Reference lines
            fig.add_hline(y=0, line_dash="dot", line_color="white", annotation_text="Average")
            fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Elite", annotation_position="right")
            fig.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Fading", annotation_position="right")

            fig.update_layout(
                **get_chart_layout(),
                height=400,
                title=f'{selected_player} - SPM (Late-Game Resilience) Over Time',
                yaxis_title='SPM Score',
                xaxis_title='Game Date'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Component breakdown - single chart
            st.markdown("### SPM Component Breakdown (Latest Game)")
            if len(player_data) > 0:
                latest = player_data.iloc[-1]
                spm_components = pd.DataFrame({
                    'Component': ['FG% Change', 'Points Change', 'TOV Resilience', 'Blocks Change', 'Steals Change'],
                    'Contribution': [
                        latest.get('spm_contrib_fg_change', 0),
                        latest.get('spm_contrib_pts_change', 0),
                        latest.get('spm_contrib_tov_resil', 0),
                        latest.get('spm_contrib_blk_change', 0),
                        latest.get('spm_contrib_stl_change', 0)
                    ],
                    'Weight': [
                        f"{latest.get('spm_weight_fg', 0.30)*100:.0f}%",
                        f"{latest.get('spm_weight_pts', 0.30)*100:.0f}%",
                        f"{latest.get('spm_weight_tov', 0.25)*100:.0f}%",
                        f"{latest.get('spm_weight_blk', 0.10)*100:.0f}%",
                        f"{latest.get('spm_weight_stl', 0.05)*100:.0f}%"
                    ]
                })

                fig = px.bar(spm_components, x='Component', y='Contribution',
                            color='Contribution', color_continuous_scale=['red', 'gray', 'green'],
                            title=f'SPM = {latest["spm"]:.2f}',
                            hover_data=['Weight'])
                fig.update_layout(**get_chart_layout(), height=350)
                st.plotly_chart(fig, use_container_width=True)

            # SPM Leaderboard (simplified - single metric)
            st.markdown("---")
            st.markdown("### Team SPM Leaderboard")
            st.markdown("*Single metric: Higher SPM = better late-game resilience*")

            leaderboard = metrics_data.groupby('player').agg({
                'spm': 'mean',
                'game_date': 'nunique'
            }).reset_index()
            leaderboard.columns = ['Player', 'Avg SPM', 'Games']
            leaderboard = leaderboard[leaderboard['Games'] >= 5]

            # Add player type classification (simplified)
            def classify_player(row):
                if row['Avg SPM'] > 0.5:
                    return "Elite Resilience"
                elif row['Avg SPM'] > 0:
                    return "Above Average"
                elif row['Avg SPM'] > -0.5:
                    return "Average"
                else:
                    return "Fades Late"

            leaderboard['Type'] = leaderboard.apply(classify_player, axis=1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Most Resilient (Highest SPM)")
                resilient = leaderboard.nlargest(10, 'Avg SPM')
                fig = px.bar(resilient, x='Avg SPM', y='Player', orientation='h',
                            color='Avg SPM', color_continuous_scale=['red', 'yellow', 'green'],
                            title='Best Late-Game Performance')
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Fades Most (Lowest SPM)")
                fading = leaderboard.nsmallest(10, 'Avg SPM')
                fig = px.bar(fading, x='Avg SPM', y='Player', orientation='h',
                            color='Avg SPM', color_continuous_scale=['red', 'yellow', 'green'],
                            title='Largest Late-Game Decline')
                fig.update_layout(**get_chart_layout(), height=400)
                st.plotly_chart(fig, use_container_width=True)

            # SPM Distribution by player type
            st.markdown("### SPM Distribution by Player Type")
            fig = px.box(leaderboard, x='Type', y='Avg SPM',
                        color='Type',
                        color_discrete_map={
                            'Elite Resilience': '#00C853',
                            'Above Average': '#8BC34A',
                            'Average': '#9E9E9E',
                            'Fades Late': '#FF5252'
                        },
                        title='SPM Score Distribution by Classification')
            fig.add_hline(y=0, line_dash="dot", line_color="white", annotation_text="League Average")
            fig.update_layout(**get_chart_layout(), height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Full leaderboard table
            st.markdown("### Full Leaderboard")
            st.dataframe(leaderboard.sort_values('Avg SPM', ascending=False).round(2), use_container_width=True)

            # Prediction Section
            st.markdown("---")
            st.markdown("### Hybrid Fatigue Prediction Models")
            st.markdown("*Combines neural network insights with traditional fatigue features*")

            # Load neural model and generate projections for hybrid model
            neural_model, neural_scalers, neural_target_cols = load_neural_model()
            neural_features_available = False

            if neural_model is not None:
                with st.spinner("Generating neural projections for hybrid model..."):
                    neural_projections = get_neural_projections_for_all_players(
                        filtered_data, neural_model, neural_scalers, neural_target_cols
                    )

                    if len(neural_projections) > 0:
                        # Merge neural projections into metrics_data
                        metrics_data = metrics_data.merge(
                            neural_projections,
                            on=['player', 'game_date'],
                            how='left'
                        )
                        neural_features_available = True
                        st.success(f"✓ Neural features added: {len(neural_projections)} projections merged")
                    else:
                        st.warning("⚠️ Neural projections unavailable - using traditional features only")
            else:
                st.info("ℹ️ Neural model not loaded - using traditional features only")

            # Model type selector
            model_type = st.radio(
                "Select Model Type",
                ["Regression (predict SPM value)", "Classification (predict fatigue risk %)"],
                horizontal=True,
                help="Regression predicts actual SPM score. Classification predicts probability of fatigue (SPM < -0.5)."
            )

            if model_type == "Regression (predict SPM value)":
                if neural_features_available:
                    st.markdown("""
                    **🧠 Hybrid Neural + Tree Model:** Combines LSTM projections with GradientBoosting.
                    - Neural features capture hot/cold streaks, role changes, sequential patterns
                    - Tree model captures fatigue factors: workload, rest, age interactions
                    - More powerful than either approach alone
                    """)
                else:
                    st.markdown("""
                    **Regression Model:** Predicts the actual SPM value for next game.
                    - More granular: distinguishes -1.5 (severe) from -0.3 (mild)
                    - Includes Effort Index as predictor
                    """)

                result = build_fatigue_regression_predictor(metrics_data)
                if result[0] is not None:
                    reg_model, reg_scaler, reg_metrics, reg_importance, reg_feature_cols, X_train_scaled = result

                    # Model performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Train R²", f"{reg_metrics['train_r2']:.3f}",
                               help="Variance explained on training data (1.0 = perfect)")
                    col2.metric("Test R²", f"{reg_metrics['test_r2']:.3f}",
                               help="Variance explained on held-out test data")
                    col3.metric("Test RMSE", f"{reg_metrics['test_rmse']:.3f}",
                               help="Root mean squared error (lower = better)")
                    col4.metric("Test MAE", f"{reg_metrics['test_mae']:.3f}",
                               help="Mean absolute error (avg prediction error)")

                    # Show model info
                    model_type = reg_metrics.get('model_type', 'Ridge')
                    target_type = reg_metrics.get('target', 'single game')
                    st.caption(f"Model: {model_type} | Target: {target_type}")

                    st.markdown("#### Feature Importance")
                    # Handle both old (Coefficient) and new (Importance) formats
                    x_col = 'Importance' if 'Importance' in reg_importance.columns else 'Coefficient'
                    fig = px.bar(reg_importance, x=x_col, y='Feature', orientation='h',
                                color=x_col, color_continuous_scale=['gray', 'green'],
                                title='What Drives SPM Predictions? (Higher = more important)')
                    fig.update_layout(**get_chart_layout(), height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Prediction interface
                    st.markdown("#### Predict Next Game SPM")
                    st.markdown(f"**Selected Player: {selected_player}** - Sliders auto-filled with latest data")

                    # Get player's actual recent values as defaults (use updated metrics_data with neural features)
                    player_data_updated = metrics_data[metrics_data['player'] == selected_player].sort_values('game_date')
                    player_recent = player_data_updated.dropna(subset=['slfi_avg_last5', 'minutes_avg_last5', 'age'])
                    if len(player_recent) > 0:
                        latest_row = player_recent.iloc[-1]
                        default_avg5 = float(latest_row.get('slfi_avg_last5', avg_spm))
                        default_avg3 = float(latest_row.get('slfi_avg_last3', default_avg5))
                        default_avg10 = float(latest_row.get('slfi_avg_last10', default_avg5))
                        default_last = float(latest_row.get('slfi_last1', default_avg5))
                        default_minutes = float(latest_row.get('minutes_avg_last5', 30.0))
                        default_age = int(latest_row.get('age', 27))
                        default_effort = float(latest_row.get('effort_index_last5', 0.0)) if 'effort_index_last5' in latest_row else 0.0
                        default_rest = int(latest_row.get('days_rest', 1)) if 'days_rest' in latest_row.index else 1
                        default_b2b = bool(latest_row.get('is_b2b', False)) if 'is_b2b' in latest_row.index else False
                    else:
                        default_avg5 = float(avg_spm) if not np.isnan(avg_spm) else 0.0
                        default_avg3 = default_avg5
                        default_avg10 = default_avg5
                        default_last = default_avg5
                        default_minutes = 30.0
                        default_age = 27
                        default_effort = 0.0
                        default_rest = 1
                        default_b2b = False

                    # Clamp defaults to slider ranges
                    default_avg5 = max(-2.0, min(2.0, default_avg5))
                    default_avg3 = max(-2.5, min(2.5, default_avg3))
                    default_avg10 = max(-1.5, min(1.5, default_avg10))
                    default_last = max(-3.0, min(3.0, default_last))
                    default_minutes = max(10.0, min(45.0, default_minutes))
                    default_age = max(19, min(42, default_age))
                    default_effort = max(-2.0, min(2.0, default_effort))
                    default_rest = max(0, min(5, default_rest))

                    # Use player name in keys so sliders reset when player changes
                    player_key = selected_player.replace(" ", "_")[:20]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("**SPM Trends (Primary)**")
                        pred_avg5 = st.slider("Avg SPM (5g)", -2.0, 2.0, default_avg5, 0.1, key=f"reg_spm_avg5_{player_key}",
                                              help="5-game rolling average - most important predictor")
                        pred_avg3 = st.slider("Avg SPM (3g)", -2.5, 2.5, default_avg3, 0.1, key=f"reg_spm_avg3_{player_key}",
                                              help="Recent 3-game trend")
                    with col2:
                        pred_avg10 = st.slider("Avg SPM (10g)", -1.5, 1.5, default_avg10, 0.1, key=f"reg_spm_avg10_{player_key}",
                                               help="Long-term baseline")
                        pred_last = st.slider("Last Game SPM", -3.0, 3.0, default_last, 0.1, key=f"reg_spm_last_{player_key}",
                                              help="Single game - less weight than averages")
                    with col3:
                        pred_minutes = st.slider("Avg Minutes (5g)", 10.0, 45.0, default_minutes, 1.0, key=f"reg_spm_min_{player_key}")
                        pred_age = st.slider("Player Age", 19, 42, default_age, 1, key=f"reg_spm_age_{player_key}")
                        b2b_index = 1 if default_b2b else 0
                        pred_b2b = st.selectbox("Back-to-Back?", [False, True], index=b2b_index, key=f"reg_spm_b2b_{player_key}")
                    with col4:
                        pred_rest = st.slider("Days Rest", 0, 5, default_rest, key=f"reg_spm_rest_{player_key}")
                        pred_effort = st.slider("Effort Index (5g)", -2.0, 2.0, default_effort, 0.1, key=f"reg_effort_{player_key}",
                                               help="Hustle trend (STL+BLK-PF-TOV)")
                        st.markdown("**Benchmarks:**")
                        st.caption(f"Elite: > {LEAGUE_BENCHMARKS['elite_slfi']}")
                        st.caption(f"Fatigued: < {LEAGUE_BENCHMARKS['fatigue_slfi']}")

                    if st.button("Predict SPM", key="predict_reg_spm"):
                        # Compute derived features
                        pred_momentum = pred_avg5 - pred_last  # Trend direction
                        pred_trend = pred_avg3 - pred_avg10    # Short vs long term
                        age_load = pred_minutes * (1 + max(0, pred_age - 28) * 0.03)
                        age_b2b = float(pred_b2b) * max(0, pred_age - 30)
                        recovery_penalty = max(0, 2 - pred_rest) * max(0, pred_age - 30)

                        # Get player's latest row for neural/optional features
                        latest_row = player_recent.iloc[-1] if len(player_recent) > 0 else {}

                        # Map feature column names to values
                        feature_value_map = {
                            'slfi_avg_last5': pred_avg5,
                            'slfi_avg_last3': pred_avg3,
                            'slfi_avg_last10': pred_avg10,
                            'slfi_momentum': pred_momentum,
                            'slfi_trend': pred_trend,
                            'minutes_avg_last5': pred_minutes,
                            'age': pred_age,
                            'age_load': age_load,
                            'age_b2b': age_b2b,
                            'recovery_penalty': recovery_penalty,
                            'effort_index_last5': pred_effort,
                            'slfi_std_last10': float(latest_row.get('slfi_std_last10', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                            'is_b2b_num': float(pred_b2b),
                            'neural_proj_pts': float(latest_row.get('neural_proj_pts', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                            'neural_proj_fga': float(latest_row.get('neural_proj_fga', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                            'neural_proj_ts': float(latest_row.get('neural_proj_ts', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                            'neural_proj_tov_rate': float(latest_row.get('neural_proj_tov_rate', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                            'neural_proj_game_score': float(latest_row.get('neural_proj_game_score', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                            'neural_expected_efficiency': float(latest_row.get('neural_expected_efficiency', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                            'neural_volume_efficiency': float(latest_row.get('neural_volume_efficiency', 0.0)) if isinstance(latest_row, pd.Series) else 0.0,
                        }

                        # Build feature vector in EXACT same order as reg_feature_cols
                        pred_features = []
                        for col in reg_feature_cols:
                            val = feature_value_map.get(col, 0.0)
                            if pd.isna(val):
                                val = 0.0
                            pred_features.append(float(val))

                        # Get neural baseline for display
                        neural_baseline = feature_value_map.get('neural_proj_game_score', None)
                        if neural_baseline == 0.0:
                            neural_baseline = None

                        X_pred = np.array([pred_features])
                        X_pred_scaled = reg_scaler.transform(X_pred)
                        predicted_spm = reg_model.predict(X_pred_scaled)[0]

                        st.markdown("---")

                        # Results - show Neural Baseline vs Hybrid if neural features used
                        if neural_baseline is not None and neural_features_available:
                            st.markdown("#### 🧠 Neural vs Hybrid Comparison")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Neural Baseline", f"{neural_baseline:.1f}",
                                       help="Raw neural model projection (game score)")
                            col2.metric("Hybrid SPM", f"{predicted_spm:+.2f}",
                                       help="Fatigue-adjusted prediction")

                            fatigue_adj = predicted_spm - (neural_baseline / 10)  # Rough scale conversion
                            if fatigue_adj < -0.3:
                                col3.metric("Fatigue Impact", f"{fatigue_adj:+.2f}", delta="High fatigue risk", delta_color="inverse")
                            elif fatigue_adj > 0.3:
                                col3.metric("Fatigue Impact", f"{fatigue_adj:+.2f}", delta="Low fatigue risk", delta_color="normal")
                            else:
                                col3.metric("Fatigue Impact", f"{fatigue_adj:+.2f}", delta="Moderate", delta_color="off")

                            # CI and PRF
                            ci_low = predicted_spm - 1.96 * reg_metrics['test_rmse']
                            ci_high = predicted_spm + 1.96 * reg_metrics['test_rmse']
                            col4.metric("95% CI", f"[{ci_low:.2f}, {ci_high:.2f}]")

                            prf = compute_physiological_risk_floor(pred_age, pred_minutes, pred_b2b, pred_rest)
                            col5.metric("Physiology Floor", f"{prf*100:.0f}%",
                                       help="Minimum fatigue risk based on age/workload")
                        else:
                            # Standard display without neural baseline
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Predicted SPM", f"{predicted_spm:+.2f}")

                            if predicted_spm < LEAGUE_BENCHMARKS['fatigue_slfi']:
                                assessment = "FATIGUE LIKELY"
                                col2.metric("Assessment", assessment, delta="Below threshold", delta_color="inverse")
                            elif predicted_spm > LEAGUE_BENCHMARKS['elite_slfi']:
                                assessment = "RESILIENT"
                                col2.metric("Assessment", assessment, delta="Above elite", delta_color="normal")
                            else:
                                assessment = "MODERATE"
                                col2.metric("Assessment", assessment, delta="In normal range", delta_color="off")

                            # CI estimate (rough, based on test RMSE)
                            ci_low = predicted_spm - 1.96 * reg_metrics['test_rmse']
                            ci_high = predicted_spm + 1.96 * reg_metrics['test_rmse']
                            col3.metric("95% CI", f"[{ci_low:.2f}, {ci_high:.2f}]")

                            # PRF comparison
                            prf = compute_physiological_risk_floor(pred_age, pred_minutes, pred_b2b, pred_rest)
                            col4.metric("Physiology Floor", f"{prf*100:.0f}%",
                                       help="Minimum fatigue risk based on age/workload")

                        # Assessment based on predicted SPM
                        if predicted_spm < LEAGUE_BENCHMARKS['fatigue_slfi']:
                            st.error(f"⚠️ **FATIGUE LIKELY** - SPM {predicted_spm:+.2f} below threshold ({LEAGUE_BENCHMARKS['fatigue_slfi']})")
                        elif predicted_spm > LEAGUE_BENCHMARKS['elite_slfi']:
                            st.success(f"✓ **RESILIENT** - SPM {predicted_spm:+.2f} above elite threshold ({LEAGUE_BENCHMARKS['elite_slfi']})")
                        else:
                            st.info(f"ℹ️ **MODERATE** - SPM {predicted_spm:+.2f} in normal range")

                        # Text explanation of prediction factors
                        st.markdown("#### What's Driving This Prediction?")

                        factors_positive = []
                        factors_negative = []
                        factors_neutral = []

                        # Analyze each factor
                        if pred_avg5 > 0.3:
                            factors_positive.append(f"**Strong recent form** (Avg SPM {pred_avg5:+.2f} over 5 games)")
                        elif pred_avg5 < -0.3:
                            factors_negative.append(f"**Poor recent form** (Avg SPM {pred_avg5:+.2f} over 5 games)")
                        else:
                            factors_neutral.append(f"Average recent form (SPM {pred_avg5:+.2f})")

                        if pred_momentum > 0.2:
                            factors_positive.append(f"**Positive momentum** ({pred_momentum:+.2f}) - trending up")
                        elif pred_momentum < -0.2:
                            factors_negative.append(f"**Negative momentum** ({pred_momentum:+.2f}) - trending down")

                        if pred_trend > 0.1:
                            factors_positive.append(f"**Upward trend** in performance")
                        elif pred_trend < -0.1:
                            factors_negative.append(f"**Downward trend** in performance")

                        if pred_is_b2b:
                            factors_negative.append(f"**Back-to-back game** - fatigue risk elevated")

                        if pred_recovery < -0.3:
                            factors_negative.append(f"**Recovery penalty** ({pred_recovery:.2f}) - insufficient rest")
                        elif pred_recovery > 0:
                            factors_positive.append(f"**Well rested** - good recovery time")

                        if pred_age >= 32:
                            factors_negative.append(f"**Age factor** ({pred_age}) - higher fatigue sensitivity")
                        elif pred_age <= 25:
                            factors_positive.append(f"**Youth advantage** ({pred_age}) - better recovery capacity")

                        if pred_minutes > 34:
                            factors_negative.append(f"**High minutes load** ({pred_minutes:.1f} avg) - accumulated fatigue")
                        elif pred_minutes < 28:
                            factors_positive.append(f"**Managed minutes** ({pred_minutes:.1f} avg) - fresh legs")

                        if pred_std > 0.8:
                            factors_negative.append(f"**Inconsistent** (volatility {pred_std:.2f}) - unpredictable output")
                        elif pred_std < 0.4:
                            factors_positive.append(f"**Consistent performer** (volatility {pred_std:.2f})")

                        # Display factors
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            if factors_positive:
                                st.markdown("**✅ Positive Factors:**")
                                for f in factors_positive:
                                    st.markdown(f"- {f}")
                            else:
                                st.markdown("**✅ Positive Factors:**\n- None significant")

                        with col_neg:
                            if factors_negative:
                                st.markdown("**⚠️ Risk Factors:**")
                                for f in factors_negative:
                                    st.markdown(f"- {f}")
                            else:
                                st.markdown("**⚠️ Risk Factors:**\n- None significant")

                        # Summary interpretation
                        if len(factors_negative) > len(factors_positive) + 1:
                            st.warning("⚠️ Multiple risk factors suggest elevated fatigue likelihood")
                        elif len(factors_positive) > len(factors_negative) + 1:
                            st.success("✅ Positive indicators outweigh risks - expect solid performance")
                        else:
                            st.info("ℹ️ Mixed signals - monitor closely for in-game fatigue signs")

                else:
                    st.warning("Not enough data for regression model (need 50+ records).")

            else:  # Classification model
                st.markdown("""
                **Classification Model:** Predicts probability of meaningful fatigue (SPM < -0.5).

                **Key Innovation:** Final risk = max(ML prediction, Physiological Risk Floor)
                """)

                result = build_fatigue_predictor(metrics_data)
                model, scaler, metrics_result, importance, feature_cols = result if result[0] is not None else (None, None, None, None, None)

                if model is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Train Accuracy", f"{metrics_result['train_acc']*100:.1f}%")
                    col2.metric("Test Accuracy", f"{metrics_result['test_acc']*100:.1f}%")
                    col3.metric("Sample Size", f"{len(metrics_data)} records")

                    st.markdown("#### What Predicts Fatigue Risk?")
                    fig = px.bar(importance, x='Coefficient', y='Feature', orientation='h',
                                color='Coefficient', color_continuous_scale=['green', 'gray', 'red'],
                                title='Feature Importance (Positive = increases fatigue risk)')
                    fig.update_layout(**get_chart_layout(), height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    # Prediction interface
                    st.markdown("#### Predict Next Game Fatigue Risk")

                    player_recent = player_data.dropna(subset=['slfi_last1', 'slfi_avg_last5', 'minutes_avg_last5', 'age'])
                    if len(player_recent) > 0:
                        latest_row = player_recent.iloc[-1]
                        default_last = float(latest_row.get('slfi_last1', avg_spm))
                        default_avg = float(latest_row.get('slfi_avg_last5', avg_spm))
                        default_minutes = float(latest_row.get('minutes_avg_last5', 30.0))
                        default_age = int(latest_row.get('age', 27))
                    else:
                        default_last = float(avg_spm) if not np.isnan(avg_spm) else 0.0
                        default_avg = float(avg_spm) if not np.isnan(avg_spm) else 0.0
                        default_minutes = 30.0
                        default_age = 27

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        pred_last = st.slider("Last Game SPM", -3.0, 3.0, default_last, 0.1, key="cls_spm_last")
                        pred_avg = st.slider("Avg SPM (last 5)", -2.0, 2.0, default_avg, 0.1, key="cls_spm_avg5")
                    with col2:
                        pred_minutes = st.slider("Avg Minutes (last 5)", 10.0, 45.0, default_minutes, 1.0, key="cls_spm_min")
                        pred_age = st.slider("Player Age", 19, 42, default_age, 1, key="cls_spm_age")
                    with col3:
                        pred_b2b = st.selectbox("Back-to-Back?", [False, True], key="cls_spm_b2b")
                        pred_rest = st.slider("Days Rest", 0, 5, 1, key="cls_spm_rest")
                    with col4:
                        prf = compute_physiological_risk_floor(pred_age, pred_minutes, pred_b2b, pred_rest)
                        st.markdown("**Risk Floor (PRF):**")
                        st.markdown(f"**{prf*100:.0f}%** minimum")

                    if st.button("Predict Fatigue Risk", key="predict_cls_spm"):
                        age_load = pred_minutes * (1 + max(0, pred_age - 28) * 0.03)
                        age_b2b = float(pred_b2b) * max(0, pred_age - 30)
                        recovery_penalty = max(0, 2 - pred_rest) * max(0, pred_age - 30)

                        X_pred = np.array([[pred_last, pred_avg, pred_minutes, pred_age, age_load, age_b2b, recovery_penalty]])
                        X_pred_scaled = scaler.transform(X_pred)
                        model_prob = model.predict_proba(X_pred_scaled)[0][1]

                        final_risk, prf = apply_fatigue_risk_floor(model_prob, pred_age, pred_minutes, pred_b2b, pred_rest)
                        floor_applied = final_risk > model_prob

                        st.markdown("---")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("ML Model Says", f"{model_prob*100:.0f}%")
                        col2.metric("Physiology Floor", f"{prf*100:.0f}%",
                                   delta="APPLIED" if floor_applied else "not needed",
                                   delta_color="inverse" if floor_applied else "off")
                        col3.metric("FINAL RISK", f"{final_risk*100:.0f}%",
                                   delta="Floor Override" if floor_applied else "Model")

                        if final_risk > 0.6:
                            risk_level = "HIGH RISK"
                        elif final_risk > 0.4:
                            risk_level = "MODERATE"
                        else:
                            risk_level = "LOWER RISK"
                        col4.metric("Assessment", risk_level)

                        if floor_applied:
                            st.warning(f"Physiological floor overrode ML prediction. "
                                      f"Age {pred_age} at {pred_minutes:.0f} min cannot be low risk.")
                        else:
                            st.info(f"ML prediction ({model_prob*100:.0f}%) is the primary risk factor.")

                else:
                    st.warning("Not enough data for classification model.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #0a1628 0%, #1a2d4a 100%); border-radius: 10px; margin-top: 1rem;'>
        <svg width="28" height="28" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;">
            <line x1="50" y1="8" x2="50" y2="92" stroke="white" stroke-width="6" stroke-linecap="round"/>
            <circle cx="50" cy="50" r="28" fill="none" stroke="white" stroke-width="6"/>
            <line x1="8" y1="50" x2="22" y2="50" stroke="white" stroke-width="6" stroke-linecap="round"/>
            <line x1="78" y1="50" x2="92" y2="50" stroke="white" stroke-width="6" stroke-linecap="round"/>
        </svg>
        <span style='color: white; font-size: 0.95rem;'><strong>Szklanny Performance Resilience System</strong></span><br>
        <span style='color: rgba(255,255,255,0.7); font-size: 0.8rem;'>Data-driven insights for optimal player management</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PREDICTIVE MODEL PAGE - Neural Network Predictions
# =============================================================================

@st.cache_resource
def load_neural_model():
    """Load the trained neural network model and scalers."""
    try:
        from sdis_neural_models import PlayerPerformanceModel

        # Check current directory first (Streamlit Cloud), then local path
        if os.path.exists("player_performance_model.pth"):
            model_path = "player_performance_model.pth"
            scalers_path = "player_model_scalers.pkl"
        else:
            model_path = os.path.join(DATA_DIR, "player_performance_model.pth")
            scalers_path = os.path.join(DATA_DIR, "player_model_scalers.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scalers_path):
            return None, None, None

        # Load scalers
        scalers = joblib.load(scalers_path)
        seq_scaler = scalers['seq_scaler']
        target_scaler = scalers['target_scaler']
        target_cols = scalers['target_cols']
        seq_features = scalers['seq_features']

        # Build and load model
        model = PlayerPerformanceModel(
            seq_input_dim=len(seq_features),
            static_input_dim=3,
            target_dims={name: 1 for name in target_cols},
            seq_hidden_dim=64,
            mlp_hidden_dims=[64, 32],
            encoder_type='lstm',
            dropout=0.2
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        return model, scalers, target_cols
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def prepare_player_game_history(df, player_name):
    """Aggregate quarter data to game-level for a specific player."""
    player_df = df[df['player'] == player_name].copy()

    if player_df.empty:
        return pd.DataFrame()

    # Aggregate to game level
    game_agg = player_df.groupby(['game_date', 'game_id']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum',
        'tov': 'sum', 'pf': 'sum', 'fgm': 'sum', 'fga': 'sum', 'minutes': 'sum',
        'win_loss': 'first', 'team': 'first'
    }).reset_index()

    # Rename columns
    game_agg.columns = ['game_date', 'game_id', 'total_pts', 'total_trb', 'total_ast',
                        'total_stl', 'total_blk', 'total_tov', 'total_pf', 'total_fgm',
                        'total_fga', 'total_minutes', 'win_loss', 'team']

    # Calculate derived features
    game_agg['fg_pct'] = np.where(game_agg['total_fga'] > 0,
                                   game_agg['total_fgm'] / game_agg['total_fga'], 0)
    game_agg['ts_pct'] = np.where(game_agg['total_fga'] > 0,
                                   game_agg['total_pts'] / (2 * game_agg['total_fga']), 0)
    game_agg['tov_rate'] = np.where((game_agg['total_fga'] + game_agg['total_tov']) > 0,
                                     game_agg['total_tov'] / (game_agg['total_fga'] + game_agg['total_tov']), 0)
    game_agg['game_score'] = (game_agg['total_pts'] + 0.4 * game_agg['total_fgm'] -
                              0.7 * game_agg['total_fga'] + 0.7 * game_agg['total_trb'] +
                              0.7 * game_agg['total_ast'] + game_agg['total_stl'] +
                              0.7 * game_agg['total_blk'] - game_agg['total_tov'] -
                              0.4 * game_agg['total_pf'])
    game_agg['usage_proxy'] = np.where(game_agg['total_minutes'] > 0,
                                        game_agg['total_fga'] / game_agg['total_minutes'] * 12, 0)
    game_agg['is_win'] = (game_agg['win_loss'] == 'W').astype(int)
    game_agg['is_b2b'] = False
    game_agg['days_rest'] = 2
    game_agg['age'] = 27

    return game_agg.sort_values('game_date')


def predict_player_next_game(model, scalers, player_history, target_cols):
    """Generate prediction for a player's next game."""
    if len(player_history) < 5:
        return None

    seq_scaler = scalers['seq_scaler']
    target_scaler = scalers['target_scaler']
    seq_features = scalers['seq_features']

    # Get last 10 games (or pad if less)
    recent = player_history.tail(10)
    sequence = recent[seq_features].values.astype(np.float32)

    if len(sequence) < 10:
        padding = np.zeros((10 - len(sequence), len(seq_features)))
        sequence = np.vstack([padding, sequence])

    # Scale sequence
    sequence_scaled = seq_scaler.transform(sequence)

    # Prepare tensors
    seq_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)
    static_tensor = torch.tensor([0, 2, 27], dtype=torch.float32).unsqueeze(0)  # is_b2b, days_rest, age

    with torch.no_grad():
        predictions = model(seq_tensor, static_tensor)

    # Denormalize
    pred_array = np.array([[predictions[name].numpy()[0, 0] for name in target_cols]])
    pred_denorm = target_scaler.inverse_transform(pred_array)[0]

    return {name: pred_denorm[i] for i, name in enumerate(target_cols)}


@st.cache_data(ttl=3600, show_spinner=False)
def get_neural_projections_cached(data_hash, _model, _scalers, _target_cols, _raw_data):
    """Cached version - call get_neural_projections_for_all_players internally."""
    return _get_neural_projections_impl(_raw_data, _model, _scalers, _target_cols)


def get_neural_projections_for_all_players(raw_data, model, scalers, target_cols):
    """
    Run neural model on all players' game histories to get projections.
    Uses caching to avoid recomputation on each page load.
    """
    if model is None or scalers is None:
        return pd.DataFrame()

    # Create hash for caching
    data_hash = hash((len(raw_data), raw_data['game_date'].max() if len(raw_data) > 0 else 0))

    try:
        return get_neural_projections_cached(data_hash, model, scalers, target_cols, raw_data)
    except Exception:
        # Fallback without caching
        return _get_neural_projections_impl(raw_data, model, scalers, target_cols)


def _get_neural_projections_impl(raw_data, model, scalers, target_cols):
    """
    Internal implementation: Run neural model on all players' game histories.

    For each player-game, uses PRIOR 10 games to generate projection.
    Optimized with batch processing where possible.

    Returns:
        DataFrame with columns: player, game_date, neural_proj_pts, neural_proj_fga,
                               neural_proj_ts, neural_proj_tov_rate, neural_proj_game_score
    """
    if model is None or scalers is None:
        return pd.DataFrame()

    seq_scaler = scalers['seq_scaler']
    target_scaler = scalers['target_scaler']
    seq_features = scalers['seq_features']

    # Pre-aggregate ALL players at once (much faster than per-player)
    game_agg = raw_data.groupby(['player', 'game_date', 'game_id']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum',
        'tov': 'sum', 'pf': 'sum', 'fgm': 'sum', 'fga': 'sum', 'minutes': 'sum',
        'win_loss': 'first'
    }).reset_index()

    # Rename and compute features
    game_agg.columns = ['player', 'game_date', 'game_id', 'total_pts', 'total_trb', 'total_ast',
                        'total_stl', 'total_blk', 'total_tov', 'total_pf', 'total_fgm',
                        'total_fga', 'total_minutes', 'win_loss']

    game_agg['fg_pct'] = np.where(game_agg['total_fga'] > 0,
                                   game_agg['total_fgm'] / game_agg['total_fga'], 0)
    game_agg['ts_pct'] = np.where(game_agg['total_fga'] > 0,
                                   game_agg['total_pts'] / (2 * game_agg['total_fga']), 0)
    game_agg['tov_rate'] = np.where((game_agg['total_fga'] + game_agg['total_tov']) > 0,
                                     game_agg['total_tov'] / (game_agg['total_fga'] + game_agg['total_tov']), 0)
    game_agg['game_score'] = (game_agg['total_pts'] + 0.4 * game_agg['total_fgm'] -
                              0.7 * game_agg['total_fga'] + 0.7 * game_agg['total_trb'] +
                              0.7 * game_agg['total_ast'] + game_agg['total_stl'] +
                              0.7 * game_agg['total_blk'] - game_agg['total_tov'] -
                              0.4 * game_agg['total_pf'])
    game_agg['usage_proxy'] = np.where(game_agg['total_minutes'] > 0,
                                        game_agg['total_fga'] / game_agg['total_minutes'] * 12, 0)
    game_agg['is_win'] = (game_agg['win_loss'] == 'W').astype(int)

    game_agg = game_agg.sort_values(['player', 'game_date']).reset_index(drop=True)

    projections = []
    players = game_agg['player'].unique()

    for player in players:
        player_history = game_agg[game_agg['player'] == player].reset_index(drop=True)

        if len(player_history) < 5:
            continue

        # For each game (starting from game 5), use prior games to make projection
        for idx in range(5, len(player_history)):
            game_date = player_history.iloc[idx]['game_date']

            # Use games BEFORE this one (prior 10)
            prior_games = player_history.iloc[max(0, idx-10):idx]

            if len(prior_games) < 5:
                continue

            try:
                # Extract sequence features
                sequence = prior_games[seq_features].values.astype(np.float32)

                # Pad if needed
                if len(sequence) < 10:
                    padding = np.zeros((10 - len(sequence), len(seq_features)))
                    sequence = np.vstack([padding, sequence])

                # Scale and predict
                sequence_scaled = seq_scaler.transform(sequence)
                seq_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)
                static_tensor = torch.tensor([0, 2, 27], dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    predictions = model(seq_tensor, static_tensor)

                # Denormalize
                pred_array = np.array([[predictions[name].numpy()[0, 0] for name in target_cols]])
                pred_denorm = target_scaler.inverse_transform(pred_array)[0]

                projections.append({
                    'player': player,
                    'game_date': game_date,
                    'neural_proj_pts': pred_denorm[target_cols.index('total_pts')],
                    'neural_proj_fga': pred_denorm[target_cols.index('total_fga')],
                    'neural_proj_ts': pred_denorm[target_cols.index('ts_pct')],
                    'neural_proj_tov_rate': pred_denorm[target_cols.index('tov_rate')],
                    'neural_proj_game_score': pred_denorm[target_cols.index('game_score')]
                })
            except Exception:
                continue

    if not projections:
        return pd.DataFrame()

    proj_df = pd.DataFrame(projections)

    # Add derived efficiency features
    proj_df['neural_expected_efficiency'] = (
        proj_df['neural_proj_ts'] * proj_df['neural_proj_pts'] /
        (proj_df['neural_proj_fga'] + 1e-6)
    )
    proj_df['neural_volume_efficiency'] = (
        proj_df['neural_proj_fga'] * proj_df['neural_proj_ts']
    )

    return proj_df


def predictive_model_page():
    """Szklanny Neural Model (SNM) - Neural Network Predictions."""
    render_header("Szklanny Neural Model (SNM)")

    st.markdown("""
    <div class='info-box'>
        <strong>🧠 Szklanny Neural Model (SNM)</strong><br>
        LSTM-based deep learning model trained on NBA quarter-by-quarter data.
        Predicts individual player and full team performance for upcoming games.
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, scalers, target_cols = load_neural_model()

    if model is None:
        st.error("⚠️ Neural model not found. Please ensure `player_performance_model.pth` and `player_model_scalers.pkl` exist.")
        st.info("Run `python sdis_neural_models.py` to train the model first.")
        return

    st.sidebar.success("✓ SNM Model Loaded")

    # Load data - check current directory first (Streamlit Cloud), then local path
    if os.path.exists("NBA_Quarter_ALL_Combined.parquet"):
        parquet_path = "NBA_Quarter_ALL_Combined.parquet"
        excel_path = "NBA_Quarter_ALL_Combined.xlsx"
    else:
        parquet_path = os.path.join(DATA_DIR, "NBA_Quarter_ALL_Combined.parquet")
        excel_path = os.path.join(DATA_DIR, "NBA_Quarter_ALL_Combined.xlsx")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        st.error("No data file found.")
        return

    df.columns = df.columns.str.lower().str.strip()

    # Analysis mode selection
    st.sidebar.header("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Select Mode",
        options=["🏀 Team Matchup Prediction", "👤 Individual Player"],
        index=0
    )

    available_teams = sorted(df['team'].unique())

    # ========================================================================
    # TEAM MATCHUP PREDICTION MODE
    # ========================================================================
    if analysis_mode == "🏀 Team Matchup Prediction":
        st.markdown("## 🏀 Team Matchup Prediction")
        st.markdown("*Predict full lineup performance against an opponent*")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Your Team")
            your_team = st.selectbox("Select Team", options=available_teams,
                                     index=available_teams.index('PHI') if 'PHI' in available_teams else 0,
                                     key="your_team")

        with col2:
            st.markdown("### Opponent")
            opponent_options = [t for t in available_teams if t != your_team]
            opponent_team = st.selectbox("Select Opponent", options=opponent_options, key="opponent_team")

        # Get current roster (2025-26 season players from dataset column)
        # Dataset format is "TEAM YYYY-YY" e.g., "PHI 2025-26"
        if 'dataset' in df.columns:
            # Filter to 2025-26 season first, fall back to 2024-25 if no 2025-26 data
            current_25_26 = df[df['dataset'].str.contains('2025-26', na=False)]
            current_24_25 = df[df['dataset'].str.contains('2024-25', na=False)]

            # Use 2025-26 if available, otherwise 2024-25
            if len(current_25_26[current_25_26['team'] == your_team]) > 0:
                your_roster = current_25_26[current_25_26['team'] == your_team]['player'].unique()
            else:
                your_roster = current_24_25[current_24_25['team'] == your_team]['player'].unique()

            if len(current_25_26[current_25_26['team'] == opponent_team]) > 0:
                opp_roster = current_25_26[current_25_26['team'] == opponent_team]['player'].unique()
            else:
                opp_roster = current_24_25[current_24_25['team'] == opponent_team]['player'].unique()
        else:
            # Fallback: just use team filter on all data
            your_roster = df[df['team'] == your_team]['player'].unique()
            opp_roster = df[df['team'] == opponent_team]['player'].unique()

        # Filter to players with enough games for prediction (at least 5 games)
        player_game_counts = df.groupby('player')['game_id'].nunique()
        eligible_players = set(player_game_counts[player_game_counts >= 5].index)

        # Calculate average minutes per game for each player (for default selection)
        player_avg_minutes = df.groupby('player')['minutes'].mean()

        # Sort eligible players by average minutes (highest first)
        your_eligible_set = [p for p in your_roster if p in eligible_players]
        opp_eligible_set = [p for p in opp_roster if p in eligible_players]

        your_eligible = sorted(your_eligible_set, key=lambda p: player_avg_minutes.get(p, 0), reverse=True)
        opp_eligible = sorted(opp_eligible_set, key=lambda p: player_avg_minutes.get(p, 0), reverse=True)

        st.markdown("---")

        # Show roster info
        st.caption(f"Found {len(your_eligible)} eligible players for {your_team}, {len(opp_eligible)} for {opponent_team}")

        # Handle empty rosters
        if len(your_eligible) == 0:
            st.error(f"No eligible players found for {your_team}. Players need at least 5 games of data.")
            st.info("Try selecting a different team or check if data exists for this team.")
            return
        if len(opp_eligible) == 0:
            st.error(f"No eligible players found for {opponent_team}. Players need at least 5 games of data.")
            return

        # Select starting lineup
        st.markdown("### 📋 Select Starting Lineups")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{your_team} Lineup**")
            default_your = your_eligible[:min(8, len(your_eligible))]
            your_lineup = st.multiselect(
                f"Select {your_team} players (5-8 recommended)",
                options=your_eligible,
                default=default_your,
                key="your_lineup"
            )

        with col2:
            st.markdown(f"**{opponent_team} Lineup**")
            default_opp = opp_eligible[:min(8, len(opp_eligible))]
            opp_lineup = st.multiselect(
                f"Select {opponent_team} players (5-8 recommended)",
                options=opp_eligible,
                default=default_opp,
                key="opp_lineup"
            )

        if len(your_lineup) < 5:
            st.warning(f"Select at least 5 players for {your_team} (currently {len(your_lineup)} selected)")
            return
        if len(opp_lineup) < 5:
            st.warning(f"Select at least 5 players for {opponent_team} (currently {len(opp_lineup)} selected)")
            return

        # Generate predictions for all players
        st.markdown("---")
        st.markdown("## 📊 Game Prediction")

        def predict_lineup(lineup, team_name):
            """Generate predictions for a full lineup."""
            predictions = []
            for player in lineup:
                player_history = prepare_player_game_history(df, player)
                if len(player_history) >= 5:
                    pred = predict_player_next_game(model, scalers, player_history, target_cols)
                    if pred is not None:
                        # Use neural model for PTS, FGA, TS%, TOV%, Game Score
                        # Use 5-game rolling average for REB, AST, STL, BLK (not predicted by model)
                        recent_5 = player_history.tail(5)
                        avg_reb = recent_5['total_trb'].mean() if 'total_trb' in recent_5.columns else 0
                        avg_ast = recent_5['total_ast'].mean() if 'total_ast' in recent_5.columns else 0
                        avg_stl = recent_5['total_stl'].mean() if 'total_stl' in recent_5.columns else 0
                        avg_blk = recent_5['total_blk'].mean() if 'total_blk' in recent_5.columns else 0

                        predictions.append({
                            'Player': player,
                            'Team': team_name,
                            'Proj PTS': pred.get('total_pts', 0),
                            'Proj REB': avg_reb,
                            'Proj AST': avg_ast,
                            'Proj STL': avg_stl,
                            'Proj BLK': avg_blk,
                            'Proj FGA': pred.get('total_fga', 0),
                            'TS%': pred.get('ts_pct', 0) * 100,
                            'TOV%': pred.get('tov_rate', 0) * 100,
                            'Game Score': pred.get('game_score', 0)
                        })
            return pd.DataFrame(predictions)

        with st.spinner("Generating predictions..."):
            your_predictions = predict_lineup(your_lineup, your_team)
            opp_predictions = predict_lineup(opp_lineup, opponent_team)

        # Calculate team totals
        your_total_pts = your_predictions['Proj PTS'].sum() if len(your_predictions) > 0 else 0
        opp_total_pts = opp_predictions['Proj PTS'].sum() if len(opp_predictions) > 0 else 0

        your_total_reb = your_predictions['Proj REB'].sum() if len(your_predictions) > 0 else 0
        opp_total_reb = opp_predictions['Proj REB'].sum() if len(opp_predictions) > 0 else 0

        your_total_ast = your_predictions['Proj AST'].sum() if len(your_predictions) > 0 else 0
        opp_total_ast = opp_predictions['Proj AST'].sum() if len(opp_predictions) > 0 else 0

        your_total_stl = your_predictions['Proj STL'].sum() if len(your_predictions) > 0 else 0
        opp_total_stl = opp_predictions['Proj STL'].sum() if len(opp_predictions) > 0 else 0

        your_total_blk = your_predictions['Proj BLK'].sum() if len(your_predictions) > 0 else 0
        opp_total_blk = opp_predictions['Proj BLK'].sum() if len(opp_predictions) > 0 else 0

        your_avg_ts = your_predictions['TS%'].mean() if len(your_predictions) > 0 else 0
        opp_avg_ts = opp_predictions['TS%'].mean() if len(opp_predictions) > 0 else 0

        # Determine winner
        point_diff = your_total_pts - opp_total_pts
        if point_diff > 5:
            winner = your_team
            confidence = min(95, 55 + abs(point_diff) * 2)
            win_color = "🟢"
        elif point_diff < -5:
            winner = opponent_team
            confidence = min(95, 55 + abs(point_diff) * 2)
            win_color = "🔴"
        else:
            winner = "Toss-up"
            confidence = 50 + abs(point_diff)
            win_color = "🟡"

        # Display matchup summary
        st.markdown("### 🏆 Matchup Summary")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            st.markdown(f"### {your_team}")
            st.metric("Projected Points", f"{your_total_pts:.0f}")
            st.metric("Projected Rebounds", f"{your_total_reb:.0f}")
            st.metric("Projected Assists", f"{your_total_ast:.0f}")
            st.metric("Projected STL/BLK", f"{your_total_stl:.0f}/{your_total_blk:.0f}")
            st.metric("Avg TS%", f"{your_avg_ts:.1f}%")

        with col2:
            st.markdown("### VS")
            if winner == your_team:
                st.markdown(f"## {win_color} **{your_team} WINS**")
            elif winner == opponent_team:
                st.markdown(f"## {win_color} **{opponent_team} WINS**")
            else:
                st.markdown(f"## {win_color} **TOSS-UP**")
            st.metric("Win Probability", f"{confidence:.0f}%")
            st.metric("Point Diff", f"{point_diff:+.0f}")
            st.metric("Rebound Diff", f"{your_total_reb - opp_total_reb:+.0f}")

        with col3:
            st.markdown(f"### {opponent_team}")
            st.metric("Projected Points", f"{opp_total_pts:.0f}")
            st.metric("Projected Rebounds", f"{opp_total_reb:.0f}")
            st.metric("Projected Assists", f"{opp_total_ast:.0f}")
            st.metric("Projected STL/BLK", f"{opp_total_stl:.0f}/{opp_total_blk:.0f}")
            st.metric("Avg TS%", f"{opp_avg_ts:.1f}%")

        # Detailed player predictions
        st.markdown("---")
        st.markdown(f"### 📈 {your_team} Player Projections")

        if len(your_predictions) > 0:
            your_display = your_predictions.sort_values('Proj PTS', ascending=False).round(1)
            your_display['TS%'] = your_display['TS%'].apply(lambda x: f"{x:.1f}%")
            your_display['TOV%'] = your_display['TOV%'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(your_display, use_container_width=True, hide_index=True)

            # Top scorer highlight
            top_scorer = your_predictions.loc[your_predictions['Proj PTS'].idxmax()]
            st.success(f"🌟 **Top Projected Scorer**: {top_scorer['Player']} with {top_scorer['Proj PTS']:.1f} pts")
        else:
            st.warning("Could not generate predictions for your team")

        st.markdown(f"### 📉 {opponent_team} Player Projections")

        if len(opp_predictions) > 0:
            opp_display = opp_predictions.sort_values('Proj PTS', ascending=False).round(1)
            opp_display['TS%'] = opp_display['TS%'].apply(lambda x: f"{x:.1f}%")
            opp_display['TOV%'] = opp_display['TOV%'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(opp_display, use_container_width=True, hide_index=True)

            opp_top = opp_predictions.loc[opp_predictions['Proj PTS'].idxmax()]
            st.info(f"⚠️ **Opponent's Top Threat**: {opp_top['Player']} with {opp_top['Proj PTS']:.1f} pts")
        else:
            st.warning("Could not generate predictions for opponent")

        # Head-to-head comparison chart
        st.markdown("### 📊 Head-to-Head Comparison")

        if len(your_predictions) > 0 and len(opp_predictions) > 0:
            fig = go.Figure()

            # Your team bars
            fig.add_trace(go.Bar(
                name=your_team,
                x=your_predictions.nlargest(5, 'Proj PTS')['Player'],
                y=your_predictions.nlargest(5, 'Proj PTS')['Proj PTS'],
                marker_color='#10B981'
            ))

            # Opponent bars
            fig.add_trace(go.Bar(
                name=opponent_team,
                x=opp_predictions.nlargest(5, 'Proj PTS')['Player'],
                y=opp_predictions.nlargest(5, 'Proj PTS')['Proj PTS'],
                marker_color='#EF4444'
            ))

            fig.update_layout(
                title="Top 5 Scorers - Projected Points",
                xaxis_title="Player",
                yaxis_title="Projected Points",
                barmode='group',
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,45,74,0.3)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        return  # End team matchup mode

    # ========================================================================
    # INDIVIDUAL PLAYER MODE (Original)
    # ========================================================================
    st.markdown("## 👤 Individual Player Projection")

    # Get unique players with enough games
    player_game_counts = df.groupby('player')['game_id'].nunique()
    eligible_players = player_game_counts[player_game_counts >= 10].index.tolist()
    eligible_players = sorted(eligible_players)

    st.sidebar.header("Player Selection")

    # Team filter
    selected_team = st.sidebar.selectbox("Filter by Team", options=["All Teams"] + available_teams)

    if selected_team != "All Teams":
        team_players = df[df['team'] == selected_team]['player'].unique()
        eligible_players = [p for p in eligible_players if p in team_players]

    # Player selection
    selected_player = st.sidebar.selectbox(
        "Select Player",
        options=eligible_players,
        index=0 if eligible_players else None
    )

    if not selected_player:
        st.warning("No eligible players found (need at least 10 games).")
        return

    # Prepare player history
    player_history = prepare_player_game_history(df, selected_player)

    if len(player_history) < 5:
        st.warning(f"{selected_player} doesn't have enough game history for prediction.")
        return

    # Generate prediction
    prediction = predict_player_next_game(model, scalers, player_history, target_cols)

    if prediction is None:
        st.error("Could not generate prediction.")
        return

    # Display results
    st.markdown(f"## 📊 Projection for {selected_player}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Projected Points",
            f"{prediction['total_pts']:.1f}",
            delta=None
        )

    with col2:
        st.metric(
            "Projected FGA",
            f"{prediction['total_fga']:.1f}",
            delta=None
        )

    with col3:
        st.metric(
            "True Shooting %",
            f"{prediction['ts_pct']*100:.1f}%",
            delta=None
        )

    col4, col5 = st.columns(2)

    with col4:
        st.metric(
            "Turnover Rate",
            f"{prediction['tov_rate']*100:.1f}%",
            delta=None
        )

    with col5:
        st.metric(
            "Game Score",
            f"{prediction['game_score']:.1f}",
            delta=None
        )

    # Recent performance comparison
    st.markdown("---")
    st.markdown("### 📈 Recent Performance (Last 10 Games)")

    recent_games = player_history.tail(10)[['game_date', 'total_pts', 'total_fga', 'ts_pct',
                                            'tov_rate', 'game_score', 'win_loss']].copy()
    recent_games.columns = ['Date', 'PTS', 'FGA', 'TS%', 'TOV%', 'Game Score', 'W/L']
    recent_games['TS%'] = (recent_games['TS%'] * 100).round(1)
    recent_games['TOV%'] = (recent_games['TOV%'] * 100).round(1)
    recent_games['Date'] = pd.to_datetime(recent_games['Date']).dt.strftime('%Y-%m-%d')

    st.dataframe(recent_games.reset_index(drop=True), use_container_width=True)

    # Averages comparison
    st.markdown("### 🎯 Projection vs Season Average")

    season_avg = player_history.agg({
        'total_pts': 'mean',
        'total_fga': 'mean',
        'ts_pct': 'mean',
        'tov_rate': 'mean',
        'game_score': 'mean'
    })

    last_10_avg = player_history.tail(10).agg({
        'total_pts': 'mean',
        'total_fga': 'mean',
        'ts_pct': 'mean',
        'tov_rate': 'mean',
        'game_score': 'mean'
    })

    comparison_df = pd.DataFrame({
        'Metric': ['Points', 'FGA', 'TS%', 'TOV%', 'Game Score'],
        'Season Avg': [
            f"{season_avg['total_pts']:.1f}",
            f"{season_avg['total_fga']:.1f}",
            f"{season_avg['ts_pct']*100:.1f}%",
            f"{season_avg['tov_rate']*100:.1f}%",
            f"{season_avg['game_score']:.1f}"
        ],
        'Last 10 Avg': [
            f"{last_10_avg['total_pts']:.1f}",
            f"{last_10_avg['total_fga']:.1f}",
            f"{last_10_avg['ts_pct']*100:.1f}%",
            f"{last_10_avg['tov_rate']*100:.1f}%",
            f"{last_10_avg['game_score']:.1f}"
        ],
        'Projection': [
            f"{prediction['total_pts']:.1f}",
            f"{prediction['total_fga']:.1f}",
            f"{prediction['ts_pct']*100:.1f}%",
            f"{prediction['tov_rate']*100:.1f}%",
            f"{prediction['game_score']:.1f}"
        ]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Trend chart
    st.markdown("### 📉 Points Trend")

    chart_data = player_history.tail(20).copy()
    chart_data['game_num'] = range(1, len(chart_data) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_data['game_num'],
        y=chart_data['total_pts'],
        mode='lines+markers',
        name='Actual Points',
        line=dict(color='#4A90D9', width=2),
        marker=dict(size=8)
    ))

    # Add projection point
    fig.add_trace(go.Scatter(
        x=[len(chart_data) + 1],
        y=[prediction['total_pts']],
        mode='markers',
        name='Projection',
        marker=dict(color='#10B981', size=14, symbol='star')
    ))

    # Add average line
    fig.add_hline(y=season_avg['total_pts'], line_dash="dash",
                  line_color="rgba(255,255,255,0.5)",
                  annotation_text=f"Season Avg: {season_avg['total_pts']:.1f}")

    fig.update_layout(
        title=f"{selected_player} - Points Trend",
        xaxis_title="Game",
        yaxis_title="Points",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,45,74,0.3)',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Model info
    with st.expander("ℹ️ About the Prediction Model"):
        st.markdown("""
        **Model Architecture:**
        - LSTM sequence encoder (64 hidden units)
        - MLP for static features (back-to-back, rest days, age)
        - Multi-task output heads for each prediction target

        **Training Data:**
        - Quarter-by-quarter NBA player statistics
        - Sequences of 10 previous games
        - Normalized using StandardScaler

        **Predictions:**
        - `total_pts`: Projected points scored
        - `total_fga`: Projected field goal attempts
        - `ts_pct`: True shooting percentage
        - `tov_rate`: Turnover rate
        - `game_score`: Overall game impact metric
        """)


def main():
    """Main entry point with module navigation."""
    # Module navigation in sidebar
    st.sidebar.markdown("## SDIS Modules")

    module = st.sidebar.radio(
        "Select Module",
        options=["SPRS - Fatigue Analysis", "Cap Lab - Salary Cap", "Szklanny Neural Model"],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Route to selected module
    if module == "SPRS - Fatigue Analysis":
        sprs_page()
    elif module == "Cap Lab - Salary Cap":
        cap_lab_page()
    elif module == "Szklanny Neural Model":
        predictive_model_page()


if __name__ == "__main__":
    main()
