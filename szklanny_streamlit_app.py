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
# SPM CALCULATION (Szklanny Late-Game Resilience Score)
# =============================================================================

def calculate_spm(data):
    """
    Calculate Szklanny Late-Game Resilience Score (SPM)

    Formula:
    1. Compute log-ratio for each stat: r_s = ln((s_Q4 + α) / (s_Q1 + α))
    2. Z-score normalize each ratio
    3. Weighted sum: SPM = 1.2*z_PTS + 0.7*z_REB + 1.0*z_AST + 0.6*z_STL + 0.4*z_BLK - 2.0*z_TOV
    """

    alpha = 0.5  # Smoothing constant

    # Weights for each stat
    weights = {
        'pts': 1.2,
        'trb': 0.7,  # Total rebounds
        'ast': 1.0,
        'stl': 0.6,
        'blk': 0.4,
        'tov': -2.0  # Negative weight for turnovers
    }

    stats = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']

    # Get Q1 and Q4 data per player per game
    q1_data = data[data['qtr_num'] == 1].groupby(['player', 'game_date', 'dataset']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum', 'tov': 'sum',
        'minutes': 'sum', 'is_b2b': 'first', 'days_rest': 'first'
    }).reset_index()
    q1_data.columns = ['player', 'game_date', 'dataset'] + [f'{c}_q1' for c in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'minutes']] + ['is_b2b', 'days_rest']

    q4_data = data[data['qtr_num'] == 4].groupby(['player', 'game_date', 'dataset']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'stl': 'sum', 'blk': 'sum', 'tov': 'sum',
        'minutes': 'sum'
    }).reset_index()
    q4_data.columns = ['player', 'game_date', 'dataset'] + [f'{c}_q4' for c in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'minutes']]

    # Merge Q1 and Q4
    spm_data = q1_data.merge(q4_data, on=['player', 'game_date', 'dataset'], how='inner')

    if len(spm_data) == 0:
        return pd.DataFrame()

    # Step A: Compute log-ratio for each stat
    for stat in stats:
        q1_col = f'{stat}_q1'
        q4_col = f'{stat}_q4'
        ratio_col = f'r_{stat}'

        spm_data[ratio_col] = np.log((spm_data[q4_col] + alpha) / (spm_data[q1_col] + alpha))

    # Step B: Z-score normalize each ratio (global normalization)
    epsilon = 1e-6
    for stat in stats:
        ratio_col = f'r_{stat}'
        z_col = f'z_{stat}'

        mean_val = spm_data[ratio_col].mean()
        std_val = spm_data[ratio_col].std()

        spm_data[z_col] = (spm_data[ratio_col] - mean_val) / (std_val + epsilon)

    # Step C: Compute weighted SPM
    spm_data['spm'] = 0
    for stat in stats:
        z_col = f'z_{stat}'
        contrib_col = f'contrib_{stat}'
        spm_data[contrib_col] = weights[stat] * spm_data[z_col]
        spm_data['spm'] += spm_data[contrib_col]

    # Clamp SPM to [-3, +3]
    spm_data['spm'] = spm_data['spm'].clip(-3, 3)

    # Sort by date
    spm_data = spm_data.sort_values(['player', 'game_date'])

    # Add rolling averages for prediction features
    spm_data['spm_last1'] = spm_data.groupby('player')['spm'].shift(1)
    spm_data['spm_avg_last5'] = spm_data.groupby('player')['spm'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    spm_data['spm_std_last10'] = spm_data.groupby('player')['spm'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()
    )
    spm_data['minutes_last'] = spm_data.groupby('player')['minutes_q1'].shift(1) + spm_data.groupby('player')['minutes_q4'].shift(1)
    spm_data['minutes_avg_last5'] = spm_data.groupby('player')['minutes_last'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    return spm_data


def build_spm_predictor(spm_data):
    """Build logistic regression model to predict SPM < 0 next game"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    # Features for prediction
    feature_cols = ['spm_last1', 'spm_avg_last5', 'spm_std_last10', 'minutes_avg_last5', 'is_b2b', 'days_rest']

    # Target: SPM < 0 (negative late-game performance)
    model_data = spm_data.dropna(subset=feature_cols + ['spm'])
    model_data['target'] = (model_data['spm'] < 0).astype(int)

    if len(model_data) < 50:
        return None, None, None, None

    X = model_data[feature_cols].values.astype(float)
    y = model_data['target'].values

    # Time-based split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    # Feature importance
    importance = pd.DataFrame({
        'Feature': ['Last SPM', 'Avg SPM (5g)', 'SPM Volatility', 'Avg Minutes', 'B2B', 'Days Rest'],
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    return model, scaler, {'train_acc': train_acc, 'test_acc': test_acc}, importance


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
        "SPM (Late-Game Resilience)"
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

    # ==========================================================================
    # TAB 6: SPM (Szklanny Late-Game Resilience Score)
    # ==========================================================================
    with tab6:
        st.subheader("Szklanny Late-Game Resilience Score (SPM)")

        st.markdown("""
        **What SPM Measures:** How a player's impact changes from early game (Q1) to late game (Q4),
        adjusted to be comparable across stats and roles.

        **Formula:** Log-ratio z-score weighted sum of PTS, REB, AST, STL, BLK, TOV changes

        - **SPM > 0:** Player performs BETTER in late game (clutch)
        - **SPM < 0:** Player performs WORSE in late game (fatigue)
        - **SPM = 0:** Stable performance throughout
        """)

        st.markdown("---")

        # Calculate SPM
        with st.spinner("Calculating SPM..."):
            spm_data = calculate_spm(filtered_data)

        if len(spm_data) == 0:
            st.warning("Not enough data to calculate SPM. Try selecting more teams or removing filters.")
        else:
            # Player selector for detailed view
            spm_players = sorted(spm_data['player'].unique())
            selected_spm_player = st.selectbox("Select Player for Detailed View", spm_players)

            player_spm = spm_data[spm_data['player'] == selected_spm_player].sort_values('game_date')

            col1, col2, col3, col4 = st.columns(4)

            avg_spm = player_spm['spm'].mean()
            std_spm = player_spm['spm'].std()
            pct_positive = (player_spm['spm'] > 0).mean() * 100
            games_played = len(player_spm)

            col1.metric("Avg SPM", f"{avg_spm:+.2f}",
                       delta="Clutch" if avg_spm > 0 else "Fatigues")
            col2.metric("SPM Volatility", f"{std_spm:.2f}")
            col3.metric("% Games SPM > 0", f"{pct_positive:.0f}%")
            col4.metric("Games", games_played)

            st.markdown("---")

            # Chart 1: SPM Over Time
            st.markdown("### Chart 1: SPM Over Time")

            fig = go.Figure()

            # Main SPM line
            fig.add_trace(go.Scatter(
                x=player_spm['game_date'],
                y=player_spm['spm'],
                mode='lines+markers',
                name='SPM',
                line=dict(color='#667eea', width=2),
                marker=dict(size=8)
            ))

            # Rolling average
            player_spm['spm_rolling'] = player_spm['spm'].rolling(5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=player_spm['game_date'],
                y=player_spm['spm_rolling'],
                mode='lines',
                name='5-Game Rolling Avg',
                line=dict(color='#f093fb', width=2, dash='dash')
            ))

            # Reference line at 0
            fig.add_hline(y=0, line_dash="dot", line_color="white", annotation_text="Baseline")

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title=f'{selected_spm_player} - SPM Over Time',
                xaxis_title='Game Date',
                yaxis_title='SPM Score',
                yaxis=dict(range=[-3.5, 3.5])
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            # Chart 2: Component Breakdown (latest game)
            with col1:
                st.markdown("### Chart 2: Latest Game Component Breakdown")

                if len(player_spm) > 0:
                    latest = player_spm.iloc[-1]

                    components = pd.DataFrame({
                        'Component': ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers'],
                        'Contribution': [
                            latest['contrib_pts'],
                            latest['contrib_trb'],
                            latest['contrib_ast'],
                            latest['contrib_stl'],
                            latest['contrib_blk'],
                            latest['contrib_tov']
                        ]
                    })

                    colors = ['green' if x > 0 else 'red' for x in components['Contribution']]

                    fig = px.bar(components, x='Component', y='Contribution',
                                color='Contribution',
                                color_continuous_scale=['red', 'gray', 'green'],
                                title=f'What drove SPM = {latest["spm"]:.2f}?')
                    fig.update_layout(template='plotly_dark', height=350)
                    st.plotly_chart(fig, use_container_width=True)

            # Chart 3: Distribution / Consistency
            with col2:
                st.markdown("### Chart 3: SPM Distribution")

                fig = px.histogram(player_spm, x='spm', nbins=15,
                                  title=f'{selected_spm_player} SPM Distribution',
                                  color_discrete_sequence=['#667eea'])
                fig.add_vline(x=0, line_dash="dot", line_color="white")
                fig.add_vline(x=avg_spm, line_dash="solid", line_color="yellow",
                             annotation_text=f"Avg: {avg_spm:.2f}")
                fig.update_layout(template='plotly_dark', height=350,
                                 xaxis_title='SPM Score', yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)

            # Chart 4: Leaderboard
            st.markdown("### Chart 4: SPM Leaderboard")

            leaderboard = spm_data.groupby('player').agg({
                'spm': ['mean', 'std', 'count'],
                'game_date': 'nunique'
            }).reset_index()
            leaderboard.columns = ['Player', 'Avg SPM', 'SPM Volatility', 'Records', 'Games']
            leaderboard['% SPM > 0'] = spm_data.groupby('player')['spm'].apply(lambda x: (x > 0).mean() * 100).values
            leaderboard = leaderboard[leaderboard['Games'] >= 5].sort_values('Avg SPM', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(leaderboard.head(10), x='Avg SPM', y='Player', orientation='h',
                            color='Avg SPM', color_continuous_scale=['red', 'gray', 'green'],
                            title='Top 10 Players by Avg SPM (Clutch Performers)')
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(leaderboard.round(2), use_container_width=True)

            # Prediction Section
            st.markdown("---")
            st.markdown("### SPM Prediction Model")
            st.markdown("Predicts probability of negative SPM (late-game drop-off) for next game.")

            model, scaler, metrics, importance = build_spm_predictor(spm_data)

            if model is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("Train Accuracy", f"{metrics['train_acc']*100:.1f}%")
                col2.metric("Test Accuracy", f"{metrics['test_acc']*100:.1f}%")
                col3.metric("Sample Size", f"{len(spm_data)} game-player records")

                st.markdown("#### Feature Importance (Coefficients)")
                fig = px.bar(importance, x='Coefficient', y='Feature', orientation='h',
                            color='Coefficient', color_continuous_scale=['green', 'gray', 'red'],
                            title='What predicts negative SPM?')
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Predict Next Game")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    pred_last_spm = st.slider("Last SPM", -3.0, 3.0, float(avg_spm), 0.1)
                with col2:
                    pred_avg_spm = st.slider("Avg SPM (5g)", -2.0, 2.0, float(avg_spm), 0.1)
                with col3:
                    pred_is_b2b = st.selectbox("Back-to-Back?", [False, True])
                with col4:
                    pred_days_rest = st.slider("Days Rest", 0, 5, 1)

                if st.button("Predict SPM Risk"):
                    X_pred = np.array([[pred_last_spm, pred_avg_spm, std_spm, 30, pred_is_b2b, pred_days_rest]])
                    X_pred_scaled = scaler.transform(X_pred)
                    prob = model.predict_proba(X_pred_scaled)[0][1]

                    col1, col2 = st.columns(2)
                    col1.metric("Risk of Negative SPM", f"{prob*100:.0f}%")
                    col2.metric("Predicted Outcome", "Likely Drop-Off" if prob > 0.5 else "Likely Resilient")
            else:
                st.warning("Not enough data for SPM prediction model. Need more game records.")

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
