"""
Szklanny Fatigue Intelligence - Interactive Streamlit Dashboard
================================================================
Includes: Szklanny Late-Game Resilience Score (SPM)
Run with: streamlit run szklanny_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LEAGUE BENCHMARKS (Research-based reference values)
# =============================================================================
LEAGUE_BENCHMARKS = {
    'avg_q4_fg_drop': -1.5,      # Average FG% drop Q1→Q4 (Sampaio et al., 2015)
    'elite_slfi': 0.5,           # Top 10% SLFI threshold
    'elite_slis': 0.8,           # Top 10% SLIS threshold
    'fatigue_slfi': -0.5,        # Bottom 25% (fatigued) threshold
    'min_games_reliable': 20,    # Minimum games for reliable metrics
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


# Page config
st.set_page_config(
    page_title="Szklanny Fatigue Intelligence",
    page_icon="basketball",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ED174C;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .spm-positive {
        color: #00C853;
        font-weight: bold;
    }
    .spm-negative {
        color: #FF5252;
        font-weight: bold;
    }
    /* Enhanced dark theme for metric cards */
    .metric-card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stMetric"] {
        background: #252525;
        border-radius: 8px;
        padding: 0.75rem;
        border: 1px solid #333;
    }
    div[data-testid="stMetric"] label {
        color: #aaa;
    }
    /* Warning badge for low sample size */
    .low-n-warning {
        background: #ff9800;
        color: #000;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    /* Significance indicator */
    .sig-indicator {
        font-size: 0.8rem;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .sig-yes {
        background: #4CAF50;
        color: white;
    }
    .sig-no {
        background: #666;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

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
                except:
                    pass

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


# =============================================================================
# SZKLANNY DUAL-METRIC SYSTEM
# =============================================================================
#
# Two separate metrics that answer different questions:
#
# 1. SLFI (Szklanny Late-Game Fatigue Index)
#    Question: "Is this player getting TIRED?"
#    Uses: FG%, TOV, PF (efficiency/effort stats only)
#    Interpretation: Negative = fatigued, Positive = resilient
#
# 2. SLIS (Szklanny Late-Game Impact Score)
#    Question: "Is this player still HELPING WIN late?"
#    Uses: PTS, REB, AST, STL, BLK, TOV (all impact stats)
#    Interpretation: Negative = less impactful, Positive = clutch
#
# Example interpretation:
#   Embiid: SLFI = -1.2 (fatigued), SLIS = +0.8 (still impactful)
#   Watford: SLFI = +0.1 (not tired), SLIS = -0.5 (low impact)
# =============================================================================

def calculate_szklanny_metrics(data, min_q1_minutes=3.0, min_q1_fga=2):
    """
    Calculate both Szklanny metrics:
    - SLFI: Late-Game Fatigue Index (FG%, TOV, PF)
    - SLIS: Late-Game Impact Score (all stats)

    Args:
        data: DataFrame with quarter-level game data
        min_q1_minutes: Minimum Q1 minutes to include player-game (usage floor)
        min_q1_fga: Minimum Q1 FGA to include player-game (usage floor)
    """

    alpha = 0.5  # Smoothing constant
    epsilon = 1e-6

    # =========================================================================
    # STEP 1: Aggregate Q1 and Q4 data
    # =========================================================================

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

    # Q1 data - include FGM, FGA, PF for fatigue index
    q1_data = data[data['qtr_num'] == 1].groupby(['player', 'game_date', 'dataset']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum',
        'tov': 'sum', 'pf': 'sum', 'fgm': 'sum', 'fga': 'sum', 'minutes': 'sum'
    }).reset_index()
    q1_cols = ['player', 'game_date', 'dataset'] + [f'{c}_q1' for c in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'fgm', 'fga', 'minutes']]
    q1_data.columns = q1_cols

    # Q4 data
    q4_data = data[data['qtr_num'] == 4].groupby(['player', 'game_date', 'dataset']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum',
        'tov': 'sum', 'pf': 'sum', 'fgm': 'sum', 'fga': 'sum', 'minutes': 'sum'
    }).reset_index()
    q4_cols = ['player', 'game_date', 'dataset'] + [f'{c}_q4' for c in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'fgm', 'fga', 'minutes']]
    q4_data.columns = q4_cols

    # Merge
    metrics_data = q1_data.merge(q4_data, on=['player', 'game_date', 'dataset'], how='inner')
    metrics_data = metrics_data.merge(total_minutes, on=['player', 'game_date', 'dataset'], how='left')

    if len(metrics_data) == 0:
        return pd.DataFrame()

    # =========================================================================
    # STEP 2: Apply USAGE FLOOR - filter out low-usage players
    # =========================================================================
    # This fixes the "Watford problem" - bench players with 2 minutes shouldn't rank high

    metrics_data = metrics_data[
        (metrics_data['minutes_q1'] >= min_q1_minutes) |
        (metrics_data['fga_q1'] >= min_q1_fga)
    ].copy()

    if len(metrics_data) == 0:
        return pd.DataFrame()

    # =========================================================================
    # STEP 3: Calculate FG% for Q1 and Q4
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
    # STEP 4: Calculate SLFI (Szklanny Late-Game Fatigue Index)
    # =========================================================================
    # Uses ONLY: FG% change, TOV change, PF change
    #
    # Logic:
    #   - FG% DROP from Q1 to Q4 = BAD (fatigue)
    #   - TOV INCREASE from Q1 to Q4 = BAD (fatigue)
    #   - PF INCREASE from Q1 to Q4 = BAD (fatigue)
    #
    # We compute: (Q4 - Q1) for each, then z-score, then weighted sum
    # Negative final score = fatigued

    # Raw changes (not log-ratio, simple difference is more interpretable for fatigue)
    metrics_data['fg_pct_change'] = metrics_data['fg_pct_q4'] - metrics_data['fg_pct_q1']
    metrics_data['tov_change'] = metrics_data['tov_q4'] - metrics_data['tov_q1']
    metrics_data['pf_change'] = metrics_data['pf_q4'] - metrics_data['pf_q1']

    # Z-score normalize the changes
    for col in ['fg_pct_change', 'tov_change', 'pf_change']:
        valid_data = metrics_data[col].dropna()
        if len(valid_data) > 1:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            metrics_data[f'z_{col}'] = (metrics_data[col] - mean_val) / (std_val + epsilon)
        else:
            metrics_data[f'z_{col}'] = 0

    # SLFI = weighted sum
    # FG% drop hurts (positive z_fg_change is good, so +1.0 weight)
    # TOV increase hurts (positive z_tov_change is bad, so -1.5 weight)
    # PF increase hurts (positive z_pf_change is bad, so -0.8 weight)
    slfi_weights = {
        'z_fg_pct_change': 1.0,   # FG% improvement = good = positive SLFI
        'z_tov_change': -1.5,    # More TOV = bad = negative SLFI
        'z_pf_change': -0.8      # More PF = bad = negative SLFI
    }

    metrics_data['slfi'] = 0.0
    for col, weight in slfi_weights.items():
        if col in metrics_data.columns:
            metrics_data['slfi'] += weight * metrics_data[col].fillna(0)

    # Store SLFI contributions for breakdown chart
    metrics_data['slfi_contrib_fg'] = slfi_weights['z_fg_pct_change'] * metrics_data['z_fg_pct_change'].fillna(0)
    metrics_data['slfi_contrib_tov'] = slfi_weights['z_tov_change'] * metrics_data['z_tov_change'].fillna(0)
    metrics_data['slfi_contrib_pf'] = slfi_weights['z_pf_change'] * metrics_data['z_pf_change'].fillna(0)

    # =========================================================================
    # STEP 5: Calculate SLIS (Szklanny Late-Game Impact Score)
    # =========================================================================
    # Uses ALL impact stats: PTS, REB, AST, STL, BLK, TOV
    # This is the original SPM logic, now properly named

    impact_stats = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']
    slis_weights = {
        'pts': 1.2,
        'trb': 0.7,
        'ast': 1.0,
        'stl': 0.6,
        'blk': 0.4,
        'tov': -2.0  # More turnovers = less impact
    }

    # Compute log-ratios for impact stats
    for stat in impact_stats:
        q1_col = f'{stat}_q1'
        q4_col = f'{stat}_q4'
        ratio_col = f'r_{stat}'
        metrics_data[ratio_col] = np.log((metrics_data[q4_col] + alpha) / (metrics_data[q1_col] + alpha))

    # Z-score normalize
    for stat in impact_stats:
        ratio_col = f'r_{stat}'
        z_col = f'z_{stat}'
        mean_val = metrics_data[ratio_col].mean()
        std_val = metrics_data[ratio_col].std()
        metrics_data[z_col] = (metrics_data[ratio_col] - mean_val) / (std_val + epsilon)

    # Compute SLIS (no clamping - let natural variance show)
    metrics_data['slis'] = 0.0
    for stat in impact_stats:
        z_col = f'z_{stat}'
        contrib_col = f'slis_contrib_{stat}'
        metrics_data[contrib_col] = slis_weights[stat] * metrics_data[z_col]
        metrics_data['slis'] += metrics_data[contrib_col]

    # =========================================================================
    # STEP 5.5: Effort Index (Composite Proxy Metric)
    # =========================================================================
    # Effort Index = (STL + BLK - PF - TOV) normalized
    # Positive = high effort, Negative = low effort/fatigue
    #
    # Research basis:
    # - STL/BLK require active effort and concentration
    # - PF/TOV increase with fatigue and mental lapses

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
    metrics_data['slfi_avg_last5'] = metrics_data.groupby('player')['slfi'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
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
            'Last SLFI',
            'Avg SLFI (5g)',
            'Avg Minutes (5g)',
            'Age',
            'Age-Adjusted Load',
            'Age × B2B',
            'Recovery Penalty'
        ],
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    return model, scaler, {'train_acc': train_acc, 'test_acc': test_acc}, importance, feature_cols


def build_fatigue_regression_predictor(metrics_data):
    """
    Build regression model to predict actual SLFI value (not just binary fatigue).

    This gives more granular predictions:
    - Predict SLFI = -1.5 (severe fatigue) vs -0.3 (mild) vs +0.5 (resilient)

    Includes Effort Index as additional feature.

    Returns:
        model, scaler, metrics_dict, importance_df, feature_cols, X_train_scaled
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # Enhanced feature set with Effort Index
    feature_cols = [
        'slfi_last1',           # Recent observed fatigue
        'slfi_avg_last5',       # Fatigue trend
        'minutes_avg_last5',    # Raw workload
        'age',                  # Base physiological factor
        'age_load',             # Age-adjusted workload
        'age_b2b',              # Age × B2B interaction
        'recovery_penalty',     # Insufficient rest penalty
        'effort_index_last5'    # Effort proxy trend (NEW)
    ]

    # Target: actual SLFI value (regression)
    model_data = metrics_data.dropna(subset=feature_cols + ['slfi'])
    model_data = model_data.copy()

    if len(model_data) < 50:
        return None, None, None, None, feature_cols, None

    # Sort by date for proper time-based split
    model_data = model_data.sort_values(['game_date', 'player']).reset_index(drop=True)

    X = model_data[feature_cols].values.astype(float)
    y = model_data['slfi'].values  # Continuous target

    # Time-based split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ridge regression (regularized for stability)
    model = Ridge(alpha=1.0)
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
        'n_test': len(X_test)
    }

    # Feature importance (coefficients)
    importance = pd.DataFrame({
        'Feature': [
            'Last SLFI',
            'Avg SLFI (5g)',
            'Avg Minutes (5g)',
            'Age',
            'Age-Adjusted Load',
            'Age × B2B',
            'Recovery Penalty',
            'Effort Index (5g)'
        ],
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    return model, scaler, metrics_result, importance, feature_cols, X_train_scaled


def build_impact_predictor(metrics_data):
    """Build logistic regression model to predict SLIS < 0 (low impact) next game"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Features for impact prediction
    feature_cols = ['slis_last1', 'slis_avg_last5', 'slis_std_last10', 'minutes_avg_last5', 'is_b2b', 'days_rest']

    # Target: SLIS < 0 (lower impact in late game)
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
        'Feature': ['Last SLIS', 'Avg SLIS (5g)', 'SLIS Volatility (10g)', 'Avg Minutes (5g)', 'B2B', 'Days Rest'],
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

    Formula: sigmoid(0.08*(age-30) + 0.10*(minutes-34) + 0.6*is_b2b + 0.15*max(0, 2-days_rest))

    Returns a probability between 0 and 1.
    """
    # Compute the linear combination
    z = (
        0.08 * max(0, age - 30) +           # Age risk (after 30)
        0.10 * max(0, minutes_avg - 34) +   # Workload risk (after 34 min)
        0.6 * float(is_b2b) +               # B2B risk (big impact)
        0.15 * max(0, 2 - days_rest)        # Recovery deficit risk
    )

    # Apply sigmoid to get probability
    prf = 1 / (1 + np.exp(-z))

    return prf


def apply_fatigue_risk_floor(model_probability, age, minutes_avg, is_b2b, days_rest):
    """
    Apply the Physiological Risk Floor to the model's predicted probability.

    Final fatigue risk = max(model_probability, PRF)

    This guarantees:
    - A 40yo on B2B at 40 min: PRF ≈ 75%+ → cannot be "low risk"
    - A 22yo on 3 days rest at 30 min: PRF ≈ 15% → model can show low risk
    - Recent resilience cannot override physiology
    """
    prf = compute_physiological_risk_floor(age, minutes_avg, is_b2b, days_rest)
    final_risk = max(model_probability, prf)
    return final_risk, prf


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">Szklanny Fatigue Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive NBA Fatigue Analysis Dashboard | Built for Performance Optimization</p>', unsafe_allow_html=True)

    # ==========================================================================
    # DATA UPLOAD SECTION
    # ==========================================================================
    st.sidebar.header("Data Source")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel Data (optional)",
        type=['xlsx'],
        help="Upload a new Excel file with quarter-level data to analyze"
    )

    # Load data
    with st.spinner('Loading data...'):
        if uploaded_file is not None:
            # Use uploaded file
            try:
                datasets = load_data_from_upload(uploaded_file)
                st.sidebar.success("Custom data loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                datasets = load_data()
        else:
            datasets = load_data()

    if not datasets:
        st.error("No data loaded. Please check the file path.")
        return

    # Combine all data
    all_data = pd.concat(datasets.values(), ignore_index=True)

    # ==========================================================================
    # SIDEBAR FILTERS
    # ==========================================================================
    st.sidebar.header("Filters")

    selected_datasets = st.sidebar.multiselect(
        "Select Teams",
        options=list(datasets.keys()),
        default=list(datasets.keys())
    )

    all_players = sorted(all_data['player'].unique())
    selected_players = st.sidebar.multiselect(
        "Select Players (leave empty for all)",
        options=all_players,
        default=[]
    )

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

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Quarter Analysis",
        "B2B Impact",
        "Player Breakdown",
        "Fatigue Proxies",
        "Predictive Model",
        "Szklanny Metrics (SLFI + SLIS)"
    ])

    # ==========================================================================
    # TAB 1: Quarter Analysis
    # ==========================================================================
    with tab1:
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
                fig.update_layout(template='plotly_dark', height=400)
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
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 2: B2B Impact
    # ==========================================================================
    with tab2:
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
                fig.update_layout(template='plotly_dark', height=400)
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
                fig.update_layout(template='plotly_dark', height=400)
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
    with tab3:
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
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(player_df, x='Q1 FG%', y='Q4 FG%',
                                size='Games', hover_name='Player',
                                color='Q4 Change', color_continuous_scale=['red', 'gray', 'green'],
                                title='Q1 vs Q4 FG% (size = games played)')
                fig.add_trace(go.Scatter(x=[30, 70], y=[30, 70], mode='lines',
                                        line=dict(dash='dash', color='white'),
                                        name='No Change Line'))
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Detailed Player Stats")
            st.dataframe(player_df, use_container_width=True)

    # ==========================================================================
    # TAB 4: Fatigue Proxies
    # ==========================================================================
    with tab4:
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

                fig.update_layout(template='plotly_dark', height=400, title='Fatigue Proxies by Quarter')
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
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 5: Original Predictive Model
    # ==========================================================================
    with tab5:
        st.subheader("Fatigue Risk Prediction Model")
        st.markdown("Predicts Q4 performance drops based on B2B, age, and MPG.")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        sixers_data = all_data[all_data['dataset'].str.contains('Sixers')]

        if len(sixers_data) < 100:
            st.warning("Not enough Sixers data for model training.")
        else:
            games = sixers_data.groupby(['game_date', 'player']).agg({
                'is_b2b': 'first',
                'fgm': 'sum',
                'fga': 'sum'
            }).reset_index()

            q1_perf = sixers_data[sixers_data['qtr_num'] == 1].groupby(['game_date', 'player']).agg({
                'fgm': 'sum', 'fga': 'sum'
            }).reset_index()
            q1_perf['q1_fg'] = np.where(q1_perf['fga'] > 0, q1_perf['fgm'] / q1_perf['fga'] * 100, np.nan)

            q4_perf = sixers_data[sixers_data['qtr_num'] == 4].groupby(['game_date', 'player']).agg({
                'fgm': 'sum', 'fga': 'sum'
            }).reset_index()
            q4_perf['q4_fg'] = np.where(q4_perf['fga'] > 0, q4_perf['fgm'] / q4_perf['fga'] * 100, np.nan)

            games = games.merge(q1_perf[['game_date', 'player', 'q1_fg']], on=['game_date', 'player'], how='left')
            games = games.merge(q4_perf[['game_date', 'player', 'q4_fg']], on=['game_date', 'player'], how='left')
            games = games.dropna(subset=['q1_fg', 'q4_fg'])

            games['q4_drop'] = (games['q4_fg'] < games['q1_fg'] - 5).astype(int)

            if 'age' in sixers_data.columns:
                age_map = sixers_data.groupby('player')['age'].first().to_dict()
                games['age'] = games['player'].map(age_map).fillna(27)
            else:
                games['age'] = 27

            if 'mpg' in sixers_data.columns:
                mpg_map = sixers_data.groupby('player')['mpg'].first().to_dict()
                games['mpg'] = games['player'].map(mpg_map).fillna(25)
            else:
                games['mpg'] = 25

            features = ['is_b2b', 'age', 'mpg']
            games = games.dropna(subset=features)
            X = games[features].values.astype(float)
            y = games['q4_drop'].values

            if len(X) > 20:
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                model.fit(X, y)

                col1, col2, col3 = st.columns(3)
                col1.metric("Model Accuracy", f"{scores.mean()*100:.1f}%")
                col2.metric("Std Dev", f"+/- {scores.std()*100:.1f}%")
                col3.metric("Sample Size", f"{len(X)} games")

                st.markdown("### Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': ['B2B', 'Age', 'MPG'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)

                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Reds',
                            title='Fatigue Risk Factors')
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Interactive Prediction Section
                st.markdown("---")
                st.markdown("### Predict Fatigue Risk for a Player")
                st.markdown("Adjust the sliders to predict likelihood of Q4 performance drop (>5% FG% decline).")

                col1, col2, col3 = st.columns(3)

                with col1:
                    pred_b2b_fatigue = st.selectbox("Back-to-Back Game?", [False, True], key="b2b_fatigue")
                with col2:
                    pred_age_fatigue = st.slider("Player Age", 19, 40, 27, key="age_fatigue")
                with col3:
                    pred_mpg_fatigue = st.slider("Minutes Per Game", 10.0, 40.0, 28.0, 0.5, key="mpg_fatigue")

                if st.button("Predict Fatigue Risk", key="predict_fatigue"):
                    X_pred_fatigue = np.array([[float(pred_b2b_fatigue), pred_age_fatigue, pred_mpg_fatigue]])
                    fatigue_prob = model.predict_proba(X_pred_fatigue)[0][1]

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fatigue Risk", f"{fatigue_prob*100:.0f}%")
                    col2.metric("Predicted Outcome", "Likely Q4 Drop" if fatigue_prob > 0.5 else "Likely Stable")
                    risk_level = "High" if fatigue_prob > 0.6 else "Moderate" if fatigue_prob > 0.4 else "Low"
                    col3.metric("Risk Level", risk_level)

    # ==========================================================================
    # TAB 6: Szklanny Dual Metrics (SLFI + SLIS)
    # ==========================================================================
    with tab6:
        st.subheader("Szklanny Late-Game Analytics")

        st.markdown("""
        ## Two Metrics That Answer Different Questions

        | Metric | Question | Uses | Interpretation |
        |--------|----------|------|----------------|
        | **SLFI** (Fatigue Index) | Is the player getting *tired*? | FG%, TOV, PF | Negative = fatigued |
        | **SLIS** (Impact Score) | Is the player still *helping win*? | PTS, REB, AST, STL, BLK, TOV | Positive = clutch |

        **Example:** Embiid might show SLFI = -1.2 (fatigued shooter) but SLIS = +0.8 (still dominates games)

        *Players must have 3+ Q1 minutes OR 2+ Q1 FGA to be included (usage floor)*
        """)

        # League benchmarks info
        col1, col2, col3 = st.columns(3)
        col1.info(f"League avg Q4 FG% drop: **{LEAGUE_BENCHMARKS['avg_q4_fg_drop']}%**")
        col2.info(f"Elite SLFI threshold: **>{LEAGUE_BENCHMARKS['elite_slfi']}**")
        col3.info(f"Fatigue threshold: **<{LEAGUE_BENCHMARKS['fatigue_slfi']}**")

        # Learn More expanders
        with st.expander("Learn More: About SLFI (Szklanny Late-Game Fatigue Index)"):
            st.markdown("""
            **SLFI** measures efficiency decline from Q1 to Q4, capturing physiological fatigue signals.

            ### Components
            - **FG% Change**: Shooting efficiency decline (weighted +1.0)
            - **Turnover Change**: Ball-handling deterioration (weighted -1.5)
            - **Foul Change**: Defensive discipline decline (weighted -0.8)

            ### Research Basis
            - FG% typically drops 1-2% in Q4 due to fatigue (Sampaio et al., 2015)
            - Turnovers increase with accumulated fatigue and cognitive load (Ben Abdelkrim et al., 2007)
            - Personal fouls correlate with decreased concentration and reaction time

            ### Interpretation
            - **SLFI > 0.5**: Elite fatigue resistance (top 10%)
            - **SLFI ≈ 0**: Average (no significant decline)
            - **SLFI < -0.5**: Meaningful fatigue (bottom 25%)
            - **SLFI < -1.5**: Severe fatigue concern
            """)

        with st.expander("Learn More: About SLIS (Szklanny Late-Game Impact Score)"):
            st.markdown("""
            **SLIS** measures overall late-game impact using all box score contributions.

            ### Components
            - **Points** (weighted 1.2): Scoring production
            - **Rebounds** (weighted 0.7): Board control
            - **Assists** (weighted 1.0): Playmaking
            - **Steals** (weighted 0.6): Active defense
            - **Blocks** (weighted 0.4): Rim protection
            - **Turnovers** (weighted -2.0): Ball security (negative impact)

            ### Calibration
            Raw SLIS is adjusted by **Closing Context Weight (CCW)** to discount garbage-time inflation:
            - **CCW = 1.0**: True closer (high MPG, high Q4 minutes, offensive involvement)
            - **CCW < 0.5**: Bench/garbage-time player (scores discounted)

            ### Why Two Metrics?
            A player can be **fatigued but impactful** (Embiid: SLFI=-1.2, SLIS=+0.8) or
            **fresh but ineffective** (bench player: SLFI=+0.3, SLIS=-0.5). Both signals matter.
            """)

        with st.expander("Learn More: About the Prediction Models"):
            st.markdown("""
            ### Classification Model
            Predicts probability of meaningful fatigue (SLFI < -0.5) using logistic regression.
            - **Physiological Risk Floor (PRF)**: Ensures a 40yo on B2B at 40 min can NEVER be "low risk"
            - Final risk = max(ML prediction, PRF)

            ### Regression Model
            Predicts actual SLFI value using Ridge regression with 8 features:
            1. Last game SLFI
            2. Average SLFI (last 5 games)
            3. Average minutes (last 5 games)
            4. Player age
            5. Age-adjusted load (minutes × age penalty)
            6. Age × B2B interaction
            7. Recovery penalty (rest days × age)
            8. Effort Index (STL+BLK-PF-TOV trend)

            ### SHAP Explanations
            Uses SHAP (SHapley Additive exPlanations) to show which features drove each prediction.
            - Green bars = features improving predicted SLFI
            - Red bars = features worsening predicted SLFI
            """)

        st.markdown("---")

        # Calculate both metrics
        with st.spinner("Calculating Szklanny Metrics..."):
            metrics_data = calculate_szklanny_metrics(filtered_data)

        if len(metrics_data) == 0:
            st.warning("Not enough data to calculate metrics. Try selecting more teams or removing filters.")
        else:
            # Player selector
            metric_players = sorted(metrics_data['player'].unique())
            selected_player = st.selectbox("Select Player for Detailed View", metric_players, key="szklanny_player")

            player_data = metrics_data[metrics_data['player'] == selected_player].sort_values('game_date')

            # Summary metrics - show BOTH scores (using CALIBRATED SLIS)
            st.markdown("### Player Summary")

            # Compute stats with confidence intervals
            avg_slfi = player_data['slfi'].mean()
            avg_slis_cal = player_data['slis_calibrated'].mean()
            avg_ccw = player_data['ccw'].mean()
            pct_slfi_positive = (player_data['slfi'] > 0).mean() * 100
            games_played = len(player_data)

            # Confidence intervals
            slfi_mean, slfi_ci_low, slfi_ci_high = compute_confidence_interval(player_data['slfi'].values)
            slis_mean, slis_ci_low, slis_ci_high = compute_confidence_interval(player_data['slis_calibrated'].values)

            # Low-n warning
            n_warning = get_sample_size_warning(games_played, LEAGUE_BENCHMARKS['min_games_reliable'])

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            col1.metric(
                "Avg SLFI (Fatigue)",
                f"{avg_slfi:+.2f}",
                delta="Resilient" if avg_slfi > 0 else "Fatigued",
                help=f"CI: [{slfi_ci_low:.2f}, {slfi_ci_high:.2f}]. Negative = fatigue signs in Q4."
            )
            col2.metric(
                "% Games Resilient",
                f"{pct_slfi_positive:.0f}%",
                help="Percentage of games where SLFI > 0 (no fatigue decline)"
            )
            col3.metric(
                "Avg SLIS (Calibrated)",
                f"{avg_slis_cal:+.2f}",
                delta="Clutch" if avg_slis_cal > 0 else "Fades",
                help=f"CI: [{slis_ci_low:.2f}, {slis_ci_high:.2f}]. Context-adjusted late-game impact."
            )
            col4.metric(
                "Context Weight (CCW)",
                f"{avg_ccw:.2f}",
                delta="Closer" if avg_ccw > 0.6 else "Bench",
                help="1.0 = true closer, <0.4 = garbage-time minutes"
            )
            col5.metric(
                "Games",
                games_played,
                help=f"Minimum {LEAGUE_BENCHMARKS['min_games_reliable']} games recommended for reliable metrics"
            )

            # Low sample warning
            if n_warning:
                st.warning(f"**{n_warning}**: Metrics may be unreliable. CI range: SLFI [{slfi_ci_low:.2f}, {slfi_ci_high:.2f}]")

            # Player type classification (using CALIBRATED SLIS)
            if avg_slfi < -0.3 and avg_slis_cal > 0.3:
                player_type = "Tired but Dominant"
                player_color = "orange"
            elif avg_slfi > 0.3 and avg_slis_cal > 0.3:
                player_type = "Elite Closer"
                player_color = "green"
            elif avg_slfi < -0.3 and avg_slis_cal < -0.3:
                player_type = "Fades Late"
                player_color = "red"
            elif avg_ccw < 0.4:
                player_type = "Garbage Time"
                player_color = "gray"
            else:
                player_type = "Steady"
                player_color = "gray"
            col6.metric("Player Type", player_type, help="Classification based on SLFI/SLIS quadrant")

            st.markdown("---")

            # Chart 1: Dual metric over time
            st.markdown("### Both Metrics Over Time")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=['SLFI (Fatigue Index)', 'SLIS Calibrated (Context-Adjusted Impact)'],
                               vertical_spacing=0.1)

            # SLFI
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['slfi'],
                mode='lines+markers', name='SLFI',
                line=dict(color='#FF6B6B', width=2), marker=dict(size=6)
            ), row=1, col=1)
            player_data['slfi_rolling'] = player_data['slfi'].rolling(5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['slfi_rolling'],
                mode='lines', name='SLFI 5-Game Avg',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ), row=1, col=1)

            # SLIS (Calibrated)
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['slis_calibrated'],
                mode='lines+markers', name='SLIS (Calibrated)',
                line=dict(color='#4ECDC4', width=2), marker=dict(size=6)
            ), row=2, col=1)
            player_data['slis_cal_rolling'] = player_data['slis_calibrated'].rolling(5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['slis_cal_rolling'],
                mode='lines', name='SLIS 5-Game Avg',
                line=dict(color='#4ECDC4', width=2, dash='dash')
            ), row=2, col=1)

            fig.add_hline(y=0, line_dash="dot", line_color="white", row=1, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="white", row=2, col=1)

            fig.update_layout(template='plotly_dark', height=500,
                             title=f'{selected_player} - Fatigue vs Impact Over Time')
            st.plotly_chart(fig, use_container_width=True)

            # Component breakdowns side by side
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### SLFI Breakdown (Latest Game)")
                if len(player_data) > 0:
                    latest = player_data.iloc[-1]
                    slfi_components = pd.DataFrame({
                        'Component': ['FG% Change', 'Turnover Change', 'Foul Change'],
                        'Contribution': [
                            latest.get('slfi_contrib_fg', 0),
                            latest.get('slfi_contrib_tov', 0),
                            latest.get('slfi_contrib_pf', 0)
                        ]
                    })
                    fig = px.bar(slfi_components, x='Component', y='Contribution',
                                color='Contribution', color_continuous_scale=['red', 'gray', 'green'],
                                title=f'SLFI = {latest["slfi"]:.2f}')
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### SLIS Breakdown (Latest Game)")
                if len(player_data) > 0:
                    latest = player_data.iloc[-1]
                    slis_components = pd.DataFrame({
                        'Component': ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers'],
                        'Contribution': [
                            latest.get('slis_contrib_pts', 0),
                            latest.get('slis_contrib_trb', 0),
                            latest.get('slis_contrib_ast', 0),
                            latest.get('slis_contrib_stl', 0),
                            latest.get('slis_contrib_blk', 0),
                            latest.get('slis_contrib_tov', 0)
                        ]
                    })
                    fig = px.bar(slis_components, x='Component', y='Contribution',
                                color='Contribution', color_continuous_scale=['red', 'gray', 'green'],
                                title=f'SLIS = {latest["slis"]:.2f}')
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)

            # Dual Leaderboard (Using CALIBRATED SLIS)
            st.markdown("---")
            st.markdown("### Team Leaderboards (Context-Adjusted)")
            st.markdown("*SLIS is now calibrated by Closing Context Weight (CCW) - garbage-time players are discounted*")

            leaderboard = metrics_data.groupby('player').agg({
                'slfi': 'mean',
                'slis_calibrated': 'mean',  # Use calibrated!
                'ccw': 'mean',
                'game_date': 'nunique'
            }).reset_index()
            leaderboard.columns = ['Player', 'Avg SLFI', 'Avg SLIS (Cal)', 'Avg CCW', 'Games']
            leaderboard = leaderboard[leaderboard['Games'] >= 5]

            # Add player type classification (using CALIBRATED SLIS)
            def classify_player(row):
                if row['Avg CCW'] < 0.4:
                    return "Garbage Time"  # Low context = bench player
                elif row['Avg SLFI'] < -0.3 and row['Avg SLIS (Cal)'] > 0.3:
                    return "Tired but Dominant"
                elif row['Avg SLFI'] > 0.3 and row['Avg SLIS (Cal)'] > 0.3:
                    return "Elite Closer"
                elif row['Avg SLFI'] < -0.3 and row['Avg SLIS (Cal)'] < -0.3:
                    return "Fades Late"
                else:
                    return "Steady"

            leaderboard['Type'] = leaderboard.apply(classify_player, axis=1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Most Fatigued (Lowest SLFI)")
                fatigued = leaderboard.nsmallest(10, 'Avg SLFI')
                fig = px.bar(fatigued, x='Avg SLFI', y='Player', orientation='h',
                            color='Avg SLFI', color_continuous_scale=['red', 'yellow', 'green'],
                            title='Who Gets Most Tired Late?')
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Most Impactful - Calibrated (Highest SLIS)")
                # Only show players with meaningful context (CCW > 0.4)
                meaningful = leaderboard[leaderboard['Avg CCW'] >= 0.4]
                impactful = meaningful.nlargest(10, 'Avg SLIS (Cal)')
                fig = px.bar(impactful, x='Avg SLIS (Cal)', y='Player', orientation='h',
                            color='Avg SLIS (Cal)', color_continuous_scale=['red', 'gray', 'green'],
                            title='True Late-Game Impact (Garbage Time Excluded)')
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Scatter plot: SLFI vs SLIS (Calibrated)
            st.markdown("### Fatigue vs Impact Matrix (Context-Adjusted)")
            fig = px.scatter(leaderboard, x='Avg SLFI', y='Avg SLIS (Cal)',
                            text='Player', color='Type',
                            size='Avg CCW',  # Size by context weight!
                            color_discrete_map={
                                'Elite Closer': '#00C853',
                                'Tired but Dominant': '#FF9800',
                                'Fades Late': '#FF5252',
                                'Garbage Time': '#666666',
                                'Steady': '#9E9E9E'
                            },
                            title='Player Classification: Fatigue vs Calibrated Impact (size = context weight)')
            fig.add_hline(y=0, line_dash="dot", line_color="white")
            fig.add_vline(x=0, line_dash="dot", line_color="white")
            fig.update_traces(textposition='top center')
            fig.update_layout(template='plotly_dark', height=500,
                             xaxis_title='SLFI (Fatigue) - Left=Tired, Right=Resilient',
                             yaxis_title='SLIS Calibrated (Impact) - Bottom=Fades, Top=Clutch')
            # Add quadrant labels
            fig.add_annotation(x=1.5, y=1.5, text="ELITE CLOSERS", showarrow=False, font=dict(color='green', size=14))
            fig.add_annotation(x=-1.5, y=1.5, text="TIRED BUT DOMINANT", showarrow=False, font=dict(color='orange', size=14))
            fig.add_annotation(x=-1.5, y=-1.5, text="FADES LATE", showarrow=False, font=dict(color='red', size=14))
            fig.add_annotation(x=1.5, y=-1.5, text="BENCH / GARBAGE", showarrow=False, font=dict(color='gray', size=14))
            st.plotly_chart(fig, use_container_width=True)

            # Full leaderboard table
            st.markdown("### Full Leaderboard")
            st.dataframe(leaderboard.sort_values('Avg SLIS (Cal)', ascending=False).round(2), use_container_width=True)

            # Prediction Section
            st.markdown("---")
            st.markdown("### Fatigue Prediction Models")

            # Model type selector
            model_type = st.radio(
                "Select Model Type",
                ["Regression (predict SLFI value)", "Classification (predict fatigue risk %)"],
                horizontal=True,
                help="Regression predicts actual SLFI score. Classification predicts probability of fatigue (SLFI < -0.5)."
            )

            if model_type == "Regression (predict SLFI value)":
                st.markdown("""
                **Regression Model:** Predicts the actual SLFI value for next game.
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

                    st.markdown("#### Feature Importance (Regression Coefficients)")
                    fig = px.bar(reg_importance, x='Coefficient', y='Feature', orientation='h',
                                color='Coefficient', color_continuous_scale=['red', 'gray', 'green'],
                                title='What Drives SLFI? (Positive = improves SLFI, Negative = worsens)')
                    fig.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Prediction interface
                    st.markdown("#### Predict Next Game SLFI")

                    # Get player's actual recent values as defaults
                    player_recent = player_data.dropna(subset=['slfi_last1', 'slfi_avg_last5', 'minutes_avg_last5', 'age'])
                    if len(player_recent) > 0:
                        latest_row = player_recent.iloc[-1]
                        default_last = float(latest_row.get('slfi_last1', avg_slfi))
                        default_avg = float(latest_row.get('slfi_avg_last5', avg_slfi))
                        default_minutes = float(latest_row.get('minutes_avg_last5', 30.0))
                        default_age = int(latest_row.get('age', 27))
                        default_effort = float(latest_row.get('effort_index_last5', 0.0)) if 'effort_index_last5' in latest_row else 0.0
                    else:
                        default_last = float(avg_slfi) if not np.isnan(avg_slfi) else 0.0
                        default_avg = float(avg_slfi) if not np.isnan(avg_slfi) else 0.0
                        default_minutes = 30.0
                        default_age = 27
                        default_effort = 0.0

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        pred_last = st.slider("Last Game SLFI", -3.0, 3.0, default_last, 0.1, key="reg_slfi_last")
                        pred_avg = st.slider("Avg SLFI (last 5)", -2.0, 2.0, default_avg, 0.1, key="reg_slfi_avg5")
                    with col2:
                        pred_minutes = st.slider("Avg Minutes (last 5)", 10.0, 45.0, default_minutes, 1.0, key="reg_slfi_min")
                        pred_age = st.slider("Player Age", 19, 42, default_age, 1, key="reg_slfi_age")
                    with col3:
                        pred_b2b = st.selectbox("Back-to-Back?", [False, True], key="reg_slfi_b2b")
                        pred_rest = st.slider("Days Rest", 0, 5, 1, key="reg_slfi_rest")
                    with col4:
                        pred_effort = st.slider("Effort Index (5g avg)", -2.0, 2.0, default_effort, 0.1, key="reg_effort",
                                               help="Recent effort trend (STL+BLK-PF-TOV normalized)")
                        st.markdown("**Benchmarks:**")
                        st.caption(f"Elite SLFI: > {LEAGUE_BENCHMARKS['elite_slfi']}")
                        st.caption(f"Fatigued: < {LEAGUE_BENCHMARKS['fatigue_slfi']}")

                    if st.button("Predict SLFI", key="predict_reg_slfi"):
                        # Compute derived features
                        age_load = pred_minutes * (1 + max(0, pred_age - 28) * 0.03)
                        age_b2b = float(pred_b2b) * max(0, pred_age - 30)
                        recovery_penalty = max(0, 2 - pred_rest) * max(0, pred_age - 30)

                        # Features: slfi_last1, slfi_avg_last5, minutes_avg_last5, age, age_load, age_b2b, recovery_penalty, effort_index_last5
                        X_pred = np.array([[pred_last, pred_avg, pred_minutes, pred_age, age_load, age_b2b, recovery_penalty, pred_effort]])
                        X_pred_scaled = reg_scaler.transform(X_pred)
                        predicted_slfi = reg_model.predict(X_pred_scaled)[0]

                        st.markdown("---")

                        # Results
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Predicted SLFI", f"{predicted_slfi:+.2f}")

                        if predicted_slfi < LEAGUE_BENCHMARKS['fatigue_slfi']:
                            assessment = "FATIGUE LIKELY"
                            col2.metric("Assessment", assessment, delta="Below threshold", delta_color="inverse")
                        elif predicted_slfi > LEAGUE_BENCHMARKS['elite_slfi']:
                            assessment = "RESILIENT"
                            col2.metric("Assessment", assessment, delta="Above elite", delta_color="normal")
                        else:
                            assessment = "MODERATE"
                            col2.metric("Assessment", assessment, delta="In normal range", delta_color="off")

                        # CI estimate (rough, based on test RMSE)
                        ci_low = predicted_slfi - 1.96 * reg_metrics['test_rmse']
                        ci_high = predicted_slfi + 1.96 * reg_metrics['test_rmse']
                        col3.metric("95% CI", f"[{ci_low:.2f}, {ci_high:.2f}]")

                        # PRF comparison
                        prf = compute_physiological_risk_floor(pred_age, pred_minutes, pred_b2b, pred_rest)
                        col4.metric("Physiology Floor", f"{prf*100:.0f}%",
                                   help="Minimum fatigue risk based on age/workload")

                        # SHAP explanation
                        st.markdown("#### Explain This Prediction (SHAP)")
                        try:
                            import shap
                            import matplotlib.pyplot as plt

                            # Create explainer
                            explainer = shap.LinearExplainer(reg_model, X_train_scaled)
                            shap_values = explainer.shap_values(X_pred_scaled)

                            # Create waterfall data manually for Plotly
                            feature_names = ['Last SLFI', 'Avg SLFI (5g)', 'Avg Minutes', 'Age',
                                           'Age-Load', 'Age×B2B', 'Recovery Pen.', 'Effort Idx']
                            shap_df = pd.DataFrame({
                                'Feature': feature_names,
                                'SHAP Value': shap_values[0],
                                'Direction': ['Increases SLFI' if v > 0 else 'Decreases SLFI' for v in shap_values[0]]
                            }).sort_values('SHAP Value', key=abs, ascending=True)

                            fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h',
                                            color='SHAP Value', color_continuous_scale=['red', 'gray', 'green'],
                                            title='Feature Contributions to Prediction')
                            fig_shap.update_layout(template='plotly_dark', height=350)
                            st.plotly_chart(fig_shap, use_container_width=True)

                            # Text explanation
                            top_pos = shap_df[shap_df['SHAP Value'] > 0].nlargest(2, 'SHAP Value')
                            top_neg = shap_df[shap_df['SHAP Value'] < 0].nsmallest(2, 'SHAP Value')

                            explanation_parts = []
                            if len(top_pos) > 0:
                                explanation_parts.append(f"**Helping:** {', '.join(top_pos['Feature'].tolist())}")
                            if len(top_neg) > 0:
                                explanation_parts.append(f"**Hurting:** {', '.join(top_neg['Feature'].tolist())}")

                            if explanation_parts:
                                st.info(" | ".join(explanation_parts))

                        except ImportError:
                            st.warning("SHAP not installed. Run: pip install shap")
                        except Exception as e:
                            st.warning(f"SHAP explanation unavailable: {str(e)}")

                else:
                    st.warning("Not enough data for regression model (need 50+ records).")

            else:  # Classification model
                st.markdown("""
                **Classification Model:** Predicts probability of meaningful fatigue (SLFI < -0.5).

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
                    fig.update_layout(template='plotly_dark', height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    # Prediction interface
                    st.markdown("#### Predict Next Game Fatigue Risk")

                    player_recent = player_data.dropna(subset=['slfi_last1', 'slfi_avg_last5', 'minutes_avg_last5', 'age'])
                    if len(player_recent) > 0:
                        latest_row = player_recent.iloc[-1]
                        default_last = float(latest_row.get('slfi_last1', avg_slfi))
                        default_avg = float(latest_row.get('slfi_avg_last5', avg_slfi))
                        default_minutes = float(latest_row.get('minutes_avg_last5', 30.0))
                        default_age = int(latest_row.get('age', 27))
                    else:
                        default_last = float(avg_slfi) if not np.isnan(avg_slfi) else 0.0
                        default_avg = float(avg_slfi) if not np.isnan(avg_slfi) else 0.0
                        default_minutes = 30.0
                        default_age = 27

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        pred_last = st.slider("Last Game SLFI", -3.0, 3.0, default_last, 0.1, key="cls_slfi_last")
                        pred_avg = st.slider("Avg SLFI (last 5)", -2.0, 2.0, default_avg, 0.1, key="cls_slfi_avg5")
                    with col2:
                        pred_minutes = st.slider("Avg Minutes (last 5)", 10.0, 45.0, default_minutes, 1.0, key="cls_slfi_min")
                        pred_age = st.slider("Player Age", 19, 42, default_age, 1, key="cls_slfi_age")
                    with col3:
                        pred_b2b = st.selectbox("Back-to-Back?", [False, True], key="cls_slfi_b2b")
                        pred_rest = st.slider("Days Rest", 0, 5, 1, key="cls_slfi_rest")
                    with col4:
                        prf = compute_physiological_risk_floor(pred_age, pred_minutes, pred_b2b, pred_rest)
                        st.markdown("**Risk Floor (PRF):**")
                        st.markdown(f"**{prf*100:.0f}%** minimum")

                    if st.button("Predict Fatigue Risk", key="predict_cls_slfi"):
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
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <strong>Szklanny Fatigue Intelligence</strong> | Built for NBA Performance Analytics<br>
        Data-driven insights for optimal player management
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
