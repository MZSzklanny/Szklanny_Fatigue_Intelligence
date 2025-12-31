"""
Szklanny Fatigue Intelligence - Interactive Streamlit Dashboard
================================================================
Run with: python -m streamlit run szklanny_streamlit_app.py
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
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load and process all data"""
    # Use relative path for Streamlit Cloud deployment
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

                # Calculate FG% (source columns are correct: FGM=makes, FGA=attempts)
                df['fg_pct'] = np.where(df['fga'] > 0, df['fgm'] / df['fga'] * 100, np.nan)
                df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
                df['dataset'] = key
                dfs.append(df)
            except Exception as e:
                st.warning(f"Could not load {sheet}: {e}")

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

            # Load advanced data
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

    # Combine all data for filtering
    all_data = pd.concat(datasets.values(), ignore_index=True)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Dataset selection
    selected_datasets = st.sidebar.multiselect(
        "Select Teams",
        options=list(datasets.keys()),
        default=list(datasets.keys())
    )

    # Player selection
    all_players = sorted(all_data['player'].unique())
    selected_players = st.sidebar.multiselect(
        "Select Players (leave empty for all)",
        options=all_players,
        default=[]
    )

    # B2B filter
    b2b_filter = st.sidebar.radio(
        "Game Type",
        options=['All Games', 'B2B Only', 'Non-B2B Only']
    )

    # Win/Loss filter
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

    # Main content
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

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Quarter Analysis",
        "B2B Impact",
        "Player Breakdown",
        "Fatigue Proxies",
        "Predictive Model"
    ])

    # ==========================================================================
    # TAB 1: Quarter Analysis
    # ==========================================================================
    with tab1:
        st.subheader("FG% by Quarter")

        col1, col2 = st.columns(2)

        with col1:
            # Quarter FG% by dataset
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
            # Q4 - Q1 change by dataset
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
            # B2B vs Non-B2B FG%
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
            # B2B Q4 impact specifically
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

        # B2B games count
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

        # Calculate player metrics
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
                # Player Q4 change chart
                fig = px.bar(player_df.head(15), x='Q4 Change', y='Player', orientation='h',
                            color='Q4 Change', color_continuous_scale=['red', 'gray', 'green'],
                            title='Top 15 Players by Q4 FG% Change')
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Q1 vs Q4 scatter
                fig = px.scatter(player_df, x='Q1 FG%', y='Q4 FG%',
                                size='Games', hover_name='Player',
                                color='Q4 Change', color_continuous_scale=['red', 'gray', 'green'],
                                title='Q1 vs Q4 FG% (size = games played)')
                fig.add_trace(go.Scatter(x=[30, 70], y=[30, 70], mode='lines',
                                        line=dict(dash='dash', color='white'),
                                        name='No Change Line'))
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)

            # Player data table
            st.markdown("### Detailed Player Stats")
            st.dataframe(
                player_df.style.format({
                    'Q1 FG%': '{:.1f}%',
                    'Q4 FG%': '{:.1f}%',
                    'Q4 Change': '{:+.1f}%',
                    'Avg PF': '{:.2f}',
                    'Avg STL': '{:.2f}',
                    'Avg TOV': '{:.2f}'
                }).background_gradient(subset=['Q4 Change'], cmap='RdYlGn'),
                use_container_width=True
            )

    # ==========================================================================
    # TAB 4: Fatigue Proxies
    # ==========================================================================
    with tab4:
        st.subheader("Fatigue Proxy Metrics")
        st.markdown("""
        **Fatigue Proxies:** Research shows these metrics indicate fatigue:
        - **Personal Fouls (PF):** Increase with fatigue (tired players commit more fouls)
        - **Steals (STL):** Decrease with fatigue (reduced defensive effort)
        - **Blocks (BLK):** Decrease with fatigue (slower reaction time)
        - **Turnovers (TOV):** Increase with fatigue (mental lapses)
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Fatigue proxies by quarter
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

                fig = make_subplots(rows=2, cols=2, subplot_titles=['Fouls', 'Steals', 'Blocks', 'Turnovers'])

                for i, (metric, title) in enumerate([('PF', 'Fouls'), ('STL', 'Steals'),
                                                       ('TOV', 'Turnovers')]):
                    if metric in proxy_df.columns:
                        row, col = (i // 2) + 1, (i % 2) + 1
                        for dataset in proxy_df['Dataset'].unique():
                            subset = proxy_df[proxy_df['Dataset'] == dataset]
                            fig.add_trace(
                                go.Scatter(x=subset['Quarter'], y=subset[metric],
                                          mode='lines+markers', name=dataset,
                                          showlegend=(i == 0)),
                                row=row, col=col
                            )

                fig.update_layout(template='plotly_dark', height=500, title='Fatigue Proxies by Quarter')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Q4 vs Q1 proxy change
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
                            barmode='group', title='Q4 Change in Fatigue Proxies',
                            labels={'Change': 'Q4 - Q1 Difference'})
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 5: Predictive Model
    # ==========================================================================
    with tab5:
        st.subheader("Fatigue Risk Prediction Model")
        st.markdown("""
        This model predicts the likelihood of Q4 performance drops based on:
        - Back-to-back game status
        - Player age
        - Minutes per game
        """)

        # Build simple model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Prepare data - use Sixers data
        sixers_data = all_data[all_data['dataset'].str.contains('Sixers')]

        if len(sixers_data) < 100:
            st.warning("Not enough Sixers data for model training. Select Sixers datasets in the sidebar.")
        else:
            # Get game-level data
            games = sixers_data.groupby(['game_date', 'player']).agg({
                'is_b2b': 'first',
                'is_win': 'first',
                'fgm': 'sum',
                'fga': 'sum'
            }).reset_index()

            # Q1 and Q4 performance
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

            # Get age and MPG
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

                with col1:
                    st.metric("Model Accuracy", f"{scores.mean()*100:.1f}%")

                with col2:
                    st.metric("Std Dev", f"+/- {scores.std()*100:.1f}%")

                with col3:
                    st.metric("Sample Size", f"{len(X)} games")

                # Feature importance
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

                # Risk prediction tool
                st.markdown("### Predict Player Fatigue Risk")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    pred_b2b = st.selectbox("Back-to-Back?", [False, True])
                with col2:
                    pred_age = st.slider("Player Age", 19, 40, 25)
                with col3:
                    pred_mpg = st.slider("Minutes/Game", 10, 40, 30)
                with col4:
                    if st.button("Predict Risk"):
                        X_pred = np.array([[pred_b2b, pred_age, pred_mpg]]).astype(float)
                        prob = model.predict_proba(X_pred)[0][1]
                        st.metric("Q4 Drop Risk", f"{prob*100:.0f}%")

                # B2B vs Normal risk comparison
                st.markdown("### Q4 Drop Rate Comparison")

                b2b_drop_rate = games[games['is_b2b'] == True]['q4_drop'].mean() * 100 if len(games[games['is_b2b'] == True]) > 0 else 0
                normal_drop_rate = games[games['is_b2b'] == False]['q4_drop'].mean() * 100 if len(games[games['is_b2b'] == False]) > 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Rest Q4 Drop Rate", f"{normal_drop_rate:.1f}%")
                with col2:
                    st.metric("B2B Q4 Drop Rate", f"{b2b_drop_rate:.1f}%")

            else:
                st.warning("Not enough data points for model training")

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
