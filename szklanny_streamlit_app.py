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


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">Szklanny Fatigue Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive NBA Fatigue Analysis Dashboard | Built for Performance Optimization</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading data...'):
        datasets = load_data()

    if not datasets:
        st.error("No data loaded. Please check the file path.")
        return

    # Combine all data
    all_data = pd.concat(datasets.values(), ignore_index=True)

    # Sidebar filters
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

            # Summary metrics - show BOTH scores
            st.markdown("### Player Summary")
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            avg_slfi = player_data['slfi'].mean()
            avg_slis = player_data['slis'].mean()
            pct_slfi_positive = (player_data['slfi'] > 0).mean() * 100
            pct_slis_positive = (player_data['slis'] > 0).mean() * 100
            games_played = len(player_data)

            col1.metric("Avg SLFI (Fatigue)", f"{avg_slfi:+.2f}",
                       delta="Resilient" if avg_slfi > 0 else "Fatigued")
            col2.metric("% Games Not Fatigued", f"{pct_slfi_positive:.0f}%")
            col3.metric("Avg SLIS (Impact)", f"{avg_slis:+.2f}",
                       delta="Clutch" if avg_slis > 0 else "Fades")
            col4.metric("% Games Impactful", f"{pct_slis_positive:.0f}%")
            col5.metric("Games", games_played)

            # Player type classification
            if avg_slfi < -0.3 and avg_slis > 0.3:
                player_type = "Tired but Dominant"
                player_color = "orange"
            elif avg_slfi > 0.3 and avg_slis > 0.3:
                player_type = "Elite Closer"
                player_color = "green"
            elif avg_slfi < -0.3 and avg_slis < -0.3:
                player_type = "Fades Late"
                player_color = "red"
            else:
                player_type = "Steady"
                player_color = "gray"
            col6.metric("Player Type", player_type)

            st.markdown("---")

            # Chart 1: Dual metric over time
            st.markdown("### Both Metrics Over Time")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=['SLFI (Fatigue Index)', 'SLIS (Impact Score)'],
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

            # SLIS
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['slis'],
                mode='lines+markers', name='SLIS',
                line=dict(color='#4ECDC4', width=2), marker=dict(size=6)
            ), row=2, col=1)
            player_data['slis_rolling'] = player_data['slis'].rolling(5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=player_data['game_date'], y=player_data['slis_rolling'],
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

            # Dual Leaderboard
            st.markdown("---")
            st.markdown("### Team Leaderboards")

            leaderboard = metrics_data.groupby('player').agg({
                'slfi': 'mean',
                'slis': 'mean',
                'game_date': 'nunique'
            }).reset_index()
            leaderboard.columns = ['Player', 'Avg SLFI', 'Avg SLIS', 'Games']
            leaderboard = leaderboard[leaderboard['Games'] >= 5]

            # Add player type classification
            def classify_player(row):
                if row['Avg SLFI'] < -0.3 and row['Avg SLIS'] > 0.3:
                    return "Tired but Dominant"
                elif row['Avg SLFI'] > 0.3 and row['Avg SLIS'] > 0.3:
                    return "Elite Closer"
                elif row['Avg SLFI'] < -0.3 and row['Avg SLIS'] < -0.3:
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
                st.markdown("#### Most Impactful (Highest SLIS)")
                impactful = leaderboard.nlargest(10, 'Avg SLIS')
                fig = px.bar(impactful, x='Avg SLIS', y='Player', orientation='h',
                            color='Avg SLIS', color_continuous_scale=['red', 'gray', 'green'],
                            title='Who Has Most Late-Game Impact?')
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Scatter plot: SLFI vs SLIS
            st.markdown("### Fatigue vs Impact Matrix")
            fig = px.scatter(leaderboard, x='Avg SLFI', y='Avg SLIS',
                            text='Player', color='Type',
                            color_discrete_map={
                                'Elite Closer': '#00C853',
                                'Tired but Dominant': '#FF9800',
                                'Fades Late': '#FF5252',
                                'Steady': '#9E9E9E'
                            },
                            title='Player Classification: Fatigue vs Impact')
            fig.add_hline(y=0, line_dash="dot", line_color="white")
            fig.add_vline(x=0, line_dash="dot", line_color="white")
            fig.update_traces(textposition='top center', marker=dict(size=12))
            fig.update_layout(template='plotly_dark', height=500,
                             xaxis_title='SLFI (Fatigue) - Left=Tired, Right=Resilient',
                             yaxis_title='SLIS (Impact) - Bottom=Fades, Top=Clutch')
            # Add quadrant labels
            fig.add_annotation(x=1.5, y=2, text="ELITE CLOSERS", showarrow=False, font=dict(color='green', size=14))
            fig.add_annotation(x=-1.5, y=2, text="TIRED BUT DOMINANT", showarrow=False, font=dict(color='orange', size=14))
            fig.add_annotation(x=-1.5, y=-2, text="FADES LATE", showarrow=False, font=dict(color='red', size=14))
            fig.add_annotation(x=1.5, y=-2, text="LOW USAGE/STEADY", showarrow=False, font=dict(color='gray', size=14))
            st.plotly_chart(fig, use_container_width=True)

            # Full leaderboard table
            st.markdown("### Full Leaderboard")
            st.dataframe(leaderboard.sort_values('Avg SLIS', ascending=False).round(2), use_container_width=True)

            # Prediction Section
            st.markdown("---")
            st.markdown("### Fatigue Prediction Model")
            st.markdown("Predicts probability of fatigue (SLFI < 0) for next game.")

            result = build_fatigue_predictor(metrics_data)
            model, scaler, metrics_result, importance, feature_cols = result if result[0] is not None else (None, None, None, None, None)

            if model is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("Train Accuracy", f"{metrics_result['train_acc']*100:.1f}%")
                col2.metric("Test Accuracy", f"{metrics_result['test_acc']*100:.1f}%")
                col3.metric("Sample Size", f"{len(metrics_data)} records")

                st.markdown("#### What Predicts Fatigue?")
                st.markdown("*Positive = increases fatigue risk, Negative = decreases risk*")
                fig = px.bar(importance, x='Coefficient', y='Feature', orientation='h',
                            color='Coefficient', color_continuous_scale=['green', 'gray', 'red'],
                            title='Feature Importance for Fatigue Prediction (Age-Adjusted Model)')
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)

                # Prediction interface
                st.markdown("#### Predict Next Game Fatigue Risk")
                st.markdown("*Now includes age-adjusted physiological factors*")

                # Get player's actual recent values as defaults
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
                    pred_last = st.slider("Last Game SLFI", -3.0, 3.0, default_last, 0.1, key="slfi_last")
                    pred_avg = st.slider("Avg SLFI (last 5)", -2.0, 2.0, default_avg, 0.1, key="slfi_avg5")
                with col2:
                    pred_minutes = st.slider("Avg Minutes (last 5)", 10.0, 45.0, default_minutes, 1.0, key="slfi_min")
                    pred_age = st.slider("Player Age", 19, 42, default_age, 1, key="slfi_age")
                with col3:
                    pred_b2b = st.selectbox("Back-to-Back?", [False, True], key="slfi_b2b")
                    pred_rest = st.slider("Days Rest", 0, 5, 1, key="slfi_rest")

                # Show derived features in real-time
                with col4:
                    # Compute derived features for display
                    age_load = pred_minutes * (1 + max(0, pred_age - 28) * 0.03)
                    age_b2b = float(pred_b2b) * max(0, pred_age - 30)
                    recovery_penalty = max(0, 2 - pred_rest) * max(0, pred_age - 30)

                    st.markdown("**Derived Features:**")
                    st.markdown(f"Age-Load: {age_load:.1f}")
                    st.markdown(f"Age×B2B: {age_b2b:.1f}")
                    st.markdown(f"Recovery Penalty: {recovery_penalty:.1f}")

                if st.button("Predict Fatigue Risk", key="predict_slfi"):
                    # Compute derived features
                    age_load = pred_minutes * (1 + max(0, pred_age - 28) * 0.03)
                    age_b2b = float(pred_b2b) * max(0, pred_age - 30)
                    recovery_penalty = max(0, 2 - pred_rest) * max(0, pred_age - 30)

                    # Features in exact order: slfi_last1, slfi_avg_last5, minutes_avg_last5, age, age_load, age_b2b, recovery_penalty
                    X_pred = np.array([[pred_last, pred_avg, pred_minutes, pred_age, age_load, age_b2b, recovery_penalty]])
                    X_pred_scaled = scaler.transform(X_pred)
                    prob = model.predict_proba(X_pred_scaled)[0][1]

                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Fatigue Risk", f"{prob*100:.0f}%")
                    col2.metric("Prediction", "High Fatigue Risk" if prob > 0.5 else "Lower Risk")
                    col3.metric("Confidence", "High" if abs(prob - 0.5) > 0.25 else "Moderate" if abs(prob - 0.5) > 0.1 else "Low")

                    # Explain what's driving the prediction
                    if pred_age >= 35 and pred_minutes >= 35:
                        col4.metric("Key Factor", "Age + Load")
                    elif pred_b2b and pred_age >= 32:
                        col4.metric("Key Factor", "Age × B2B")
                    elif recovery_penalty > 3:
                        col4.metric("Key Factor", "Recovery Deficit")
                    else:
                        col4.metric("Key Factor", "SLFI History")
            else:
                st.warning("Not enough data for prediction model.")

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
