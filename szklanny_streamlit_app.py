"""
Szklanny Fatigue Intelligence - Interactive Streamlit Dashboard (Rewritten)
==========================================================================

Key fixes in this rewrite:
1) Closing Context Weight (CCW) is rescaled + capped to avoid labeling everyone as "garbage time"
   and to prevent bench/garbage-minute players from looking like elite closers.
2) Fatigue predictor now predicts NEXT-GAME fatigue by shifting the target forward one game.
3) "Bench / Low Leverage" classification happens FIRST (before "Elite Closer") so low-context
   players can't be mislabeled as closers.

Run:
  streamlit run szklanny_streamlit_app_rewrite.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Page config + CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Szklanny Fatigue Intelligence",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ED174C;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.75rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load and process all data."""
    file_path = "Sixers Fatigue Data.xlsx"

    try:
        xl = pd.ExcelFile(file_path)
    except Exception as e:
        st.error(f"Could not open Excel file: {e}")
        return {}

    q_sheets = {
        "Pacers (Control)": ["Control - '24 Pacers Q Data", "Control - '24 Pacers Q Data Pof"],
        "Sixers 2024-25": ["24-25 Sixers Q Data"],
        "Sixers 2025-26": ["25-26 Sixers Q Data"],
    }

    adv_sheets = {
        "Pacers (Control)": ["Control - '24 Pacers Adv Data ", "Control - '24 Pacers Adv Poff"],
        "Sixers 2024-25": ["24-25 Sixers Advanced Data"],
        "Sixers 2025-26": ["25-26 Sixers Advanced Data"],
    }

    datasets = {}

    for key in q_sheets:
        dfs = []
        for sheet in q_sheets[key]:
            try:
                df = pd.read_excel(xl, sheet_name=sheet)
                df.columns = df.columns.str.lower().str.strip()

                if "quarter" in df.columns:
                    df = df.rename(columns={"quarter": "qtr"})
                if "win/loss" in df.columns:
                    df = df.rename(columns={"win/loss": "win_loss"})

                df = df[df["qtr"].astype(str).str.match(r"^Q[1-4]$", na=False)]
                df["qtr_num"] = df["qtr"].str.replace("Q", "").astype(int)

                # FG% (quarter-level)
                df["fg_pct"] = np.where(df["fga"] > 0, df["fgm"] / df["fga"] * 100, np.nan)

                df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
                df["dataset"] = key
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            continue

        combined = pd.concat(dfs, ignore_index=True)

        # B2B + days_rest at game level per player
        combined = combined.sort_values(["player", "game_date", "qtr_num"])
        game_dates = combined.groupby(["player", "game_date"]).size().reset_index()[["player", "game_date"]]
        game_dates = game_dates.sort_values(["player", "game_date"])
        game_dates["prev_game"] = game_dates.groupby("player")["game_date"].shift(1)
        game_dates["days_rest"] = (game_dates["game_date"] - game_dates["prev_game"]).dt.days
        game_dates["is_b2b"] = game_dates["days_rest"] == 1

        combined = combined.merge(
            game_dates[["player", "game_date", "is_b2b", "days_rest"]],
            on=["player", "game_date"],
            how="left",
        )
        combined["is_b2b"] = combined["is_b2b"].fillna(False)
        combined["days_rest"] = combined["days_rest"].fillna(2)

        combined["is_win"] = combined["win_loss"].astype(str).str.upper().eq("W")

        # merge age/mpg from advanced sheets (if present)
        combined["player_clean"] = combined["player"].astype(str).str.strip().str.lower()
        for adv_sheet in adv_sheets.get(key, []):
            try:
                adv_df = pd.read_excel(xl, sheet_name=adv_sheet)
                adv_df.columns = adv_df.columns.str.lower().str.strip()

                if key == "Sixers 2025-26" and "age" in adv_df.columns:
                    adv_df = adv_df[adv_df["age"].apply(lambda x: str(x).isdigit() if pd.notna(x) else False)]

                for c in ["age", "games", "minutes_total"]:
                    if c in adv_df.columns:
                        adv_df[c] = pd.to_numeric(adv_df[c], errors="coerce")

                if "games" in adv_df.columns and "minutes_total" in adv_df.columns:
                    adv_df["mpg"] = adv_df["minutes_total"] / adv_df["games"]

                adv_df["player_clean"] = adv_df["player"].astype(str).str.strip().str.lower()
                combined = combined.merge(
                    adv_df[["player_clean", "age", "mpg"]].drop_duplicates("player_clean"),
                    on="player_clean",
                    how="left",
                )
            except Exception:
                continue

        datasets[key] = combined

    return datasets


# =============================================================================
# SZKLANNY DUAL-METRIC SYSTEM
# =============================================================================

def calculate_szklanny_metrics(data: pd.DataFrame, min_q1_minutes: float = 3.0, min_q1_fga: int = 2) -> pd.DataFrame:
    """
    Computes:
      SLFI = fatigue signal (FG% change, TOV change, PF change)  -> negative means fatigued
      SLIS = impact signal  (PTS/REB/AST/STL/BLK/TOV log-ratios) -> positive means impactful

    Then calibrates SLIS using CCW (Closing Context Weight) to avoid "Watford is a closer" false positives.
    """
    alpha = 0.5
    eps = 1e-6

    # ---- game totals (for total_minutes + schedule + age) ----
    agg_dict = {"minutes": "sum", "is_b2b": "first", "days_rest": "first"}
    if "age" in data.columns:
        agg_dict["age"] = "first"
    if "mpg" in data.columns:
        agg_dict["mpg"] = "first"

    totals = data.groupby(["player", "game_date", "dataset"]).agg(agg_dict).reset_index()
    if "age" not in totals.columns:
        totals["age"] = np.nan
    if "mpg" not in totals.columns:
        totals["mpg"] = np.nan
    totals = totals.rename(columns={"minutes": "total_minutes"})

    # ---- Q1 / Q4 aggregates ----
    def agg_q(qnum: int) -> pd.DataFrame:
        dfq = data[data["qtr_num"] == qnum].groupby(["player", "game_date", "dataset"]).agg(
            pts=("pts", "sum"),
            trb=("trb", "sum"),
            ast=("ast", "sum"),
            stl=("stl", "sum"),
            blk=("blk", "sum"),
            tov=("tov", "sum"),
            pf=("pf", "sum"),
            fgm=("fgm", "sum"),
            fga=("fga", "sum"),
            minutes=("minutes", "sum"),
        ).reset_index()
        dfq.columns = ["player", "game_date", "dataset"] + [f"{c}_q{qnum}" for c in ["pts","trb","ast","stl","blk","tov","pf","fgm","fga","minutes"]]
        return dfq

    q1 = agg_q(1)
    q4 = agg_q(4)

    m = q1.merge(q4, on=["player","game_date","dataset"], how="inner").merge(totals, on=["player","game_date","dataset"], how="left")
    if m.empty:
        return pd.DataFrame()

    # ---- usage floor (bench filter) ----
    m = m[(m["minutes_q1"] >= min_q1_minutes) | (m["fga_q1"] >= min_q1_fga)].copy()
    if m.empty:
        return pd.DataFrame()

    # ---- FG% ----
    m["fg_pct_q1"] = np.where(m["fga_q1"] > 0, m["fgm_q1"] / m["fga_q1"] * 100, np.nan)
    m["fg_pct_q4"] = np.where(m["fga_q4"] > 0, m["fgm_q4"] / m["fga_q4"] * 100, np.nan)

    # =============================================================================
    # SLFI (fatigue) : simple diffs, z-scored, weighted sum
    # =============================================================================
    m["fg_pct_change"] = m["fg_pct_q4"] - m["fg_pct_q1"]
    m["tov_change"] = m["tov_q4"] - m["tov_q1"]
    m["pf_change"] = m["pf_q4"] - m["pf_q1"]

    for col in ["fg_pct_change","tov_change","pf_change"]:
        valid = m[col].dropna()
        if len(valid) > 1:
            mu, sd = valid.mean(), valid.std()
            m[f"z_{col}"] = (m[col] - mu) / (sd + eps)
        else:
            m[f"z_{col}"] = 0.0

    slfi_weights = {"z_fg_pct_change": 1.0, "z_tov_change": -1.5, "z_pf_change": -0.8}
    m["slfi"] = 0.0
    for k, w in slfi_weights.items():
        if k in m.columns:
            m["slfi"] += w * m[k].fillna(0)

    m["slfi_contrib_fg"] = slfi_weights["z_fg_pct_change"] * m["z_fg_pct_change"].fillna(0)
    m["slfi_contrib_tov"] = slfi_weights["z_tov_change"] * m["z_tov_change"].fillna(0)
    m["slfi_contrib_pf"] = slfi_weights["z_pf_change"] * m["z_pf_change"].fillna(0)

    # =============================================================================
    # SLIS (impact): log ratios, z-scored, weighted sum
    # =============================================================================
    impact_stats = ["pts","trb","ast","stl","blk","tov"]
    slis_weights = {"pts": 1.2, "trb": 0.7, "ast": 1.0, "stl": 0.6, "blk": 0.4, "tov": -2.0}

    for s in impact_stats:
        m[f"r_{s}"] = np.log((m[f"{s}_q4"] + alpha) / (m[f"{s}_q1"] + alpha))

    for s in impact_stats:
        mu, sd = m[f"r_{s}"].mean(), m[f"r_{s}"].std()
        m[f"z_{s}"] = (m[f"r_{s}"] - mu) / (sd + eps)

    m["slis"] = 0.0
    for s in impact_stats:
        m[f"slis_contrib_{s}"] = slis_weights[s] * m[f"z_{s}"]
        m["slis"] += m[f"slis_contrib_{s}"]

    # =============================================================================
    # Rolling features (lagged so we can predict next game)
    # =============================================================================
    m = m.sort_values(["player","game_date"]).reset_index(drop=True)

    m["slfi_last1"] = m.groupby("player")["slfi"].shift(1)
    m["slfi_avg_last5"] = m.groupby("player")["slfi"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    m["slis_last1"] = m.groupby("player")["slis"].shift(1)
    m["slis_avg_last5"] = m.groupby("player")["slis"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    m["slis_std_last10"] = m.groupby("player")["slis"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())

    m["minutes_last"] = m.groupby("player")["total_minutes"].shift(1)
    m["minutes_avg_last5"] = m.groupby("player")["total_minutes"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # =============================================================================
    # Age-adjusted features
    # =============================================================================
    m["age"] = pd.to_numeric(m["age"], errors="coerce").fillna(27)
    m["mpg"] = pd.to_numeric(m["mpg"], errors="coerce").fillna(m.groupby("player")["total_minutes"].transform("mean")).fillna(20)

    m["age_load"] = m["minutes_avg_last5"].fillna(30) * (1 + np.maximum(0, m["age"] - 28) * 0.03)
    m["age_b2b"] = m["is_b2b"].astype(float) * np.maximum(0, m["age"] - 30)
    m["recovery_penalty"] = np.maximum(0, 2 - m["days_rest"].fillna(2)) * np.maximum(0, m["age"] - 30)

    # =============================================================================
    # CCW (Closing Context Weight) - rewritten to match basketball reality
    # =============================================================================
    # Intuition:
    # - Closers play meaningful Q4 minutes AND have stable roles (MPG) AND are involved late.
    # - Bench/garbage-minute players must be discounted hard, or they "hack" late-game metrics.

    # Q4 share relative to "normal" expected share (~25%)
    m["q4_share"] = np.where(m["total_minutes"] > 0, m["minutes_q4"] / m["total_minutes"], 0.0)
    m["q4_leverage"] = (m["q4_share"] / 0.25).clip(0, 1.25)  # 1.0 = normal share, >1 = heavy Q4 closer usage

    # Role trust (30 MPG ~ full closer trust; 28 also fine, but 30 gives a little separation)
    m["role_weight"] = (m["mpg"] / 30.0).clip(0, 1.0)

    # Late involvement: shots + playmaking signal late
    # (bench guys can have 1-2 FGA and look inflated; we cap + we also add minutes floor below)
    involvement = m["fga_q4"] + 0.5 * m["ast_q4"]
    m["q4_presence"] = (involvement / 4.0).clip(0, 1.0)  # 4 "involvement units" -> full credit

    # Must have meaningful Q4 run
    m["q4_minutes_floor"] = (m["minutes_q4"] / 7.0).clip(0, 1.0)  # 7+ minutes in Q4 = real closing stint

    # Bench cap: if you barely played overall, you can't be a closer regardless of Q4 math
    m["bench_cap"] = (m["total_minutes"] / 24.0).clip(0, 1.0)  # 24+ total min -> full credit

    m["ccw"] = (m["q4_leverage"] * m["role_weight"] * m["q4_presence"] * m["q4_minutes_floor"] * m["bench_cap"]).clip(0.05, 1.0)

    # Calibrated impact score
    m["slis_calibrated"] = m["slis"] * m["ccw"]

    return m


# =============================================================================
# Predictive models
# =============================================================================

def build_fatigue_predictor(metrics_data: pd.DataFrame):
    """
    Logistic regression predicting NEXT-game meaningful fatigue:
      target_next = 1 if next game's SLFI < -0.5 else 0

    Critical fix: target is shifted forward by one game per player.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feature_cols = ["slfi_last1","slfi_avg_last5","minutes_avg_last5","age","age_load","age_b2b","recovery_penalty"]

    df = metrics_data.copy()
    df = df.sort_values(["player","game_date"]).reset_index(drop=True)

    # Next-game target (lead)
    df["slfi_next"] = df.groupby("player")["slfi"].shift(-1)
    df["target_next"] = (df["slfi_next"] < -0.5).astype(int)

    model_data = df.dropna(subset=feature_cols + ["slfi_next"])
    if len(model_data) < 80:
        return None, None, None, None, feature_cols

    # time-based split (by date)
    model_data = model_data.sort_values(["game_date","player"]).reset_index(drop=True)
    X = model_data[feature_cols].values.astype(float)
    y = model_data["target_next"].values.astype(int)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_s, y_train)

    metrics = {"train_acc": model.score(X_train_s, y_train), "test_acc": model.score(X_test_s, y_test)}

    importance = pd.DataFrame({
        "Feature": ["Last SLFI","Avg SLFI (5g)","Avg Minutes (5g)","Age","Age-Adjusted Load","Age × B2B","Recovery Penalty"],
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", key=np.abs, ascending=False)

    return model, scaler, metrics, importance, feature_cols


def compute_physiological_risk_floor(age: float, minutes_avg: float, is_b2b: bool, days_rest: float) -> float:
    """
    Deterministic minimum fatigue risk (PRF). Ensures:
      40yo + 40 min + B2B + 0 rest -> can't be low risk.
    """
    z = (
        0.08 * max(0.0, age - 30.0) +
        0.10 * max(0.0, minutes_avg - 34.0) +
        0.60 * float(bool(is_b2b)) +
        0.15 * max(0.0, 2.0 - float(days_rest))
    )
    return float(1.0 / (1.0 + np.exp(-z)))


def apply_fatigue_risk_floor(model_probability: float, age: float, minutes_avg: float, is_b2b: bool, days_rest: float):
    prf = compute_physiological_risk_floor(age, minutes_avg, is_b2b, days_rest)
    final_risk = max(float(model_probability), prf)
    return final_risk, prf


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<div class="main-header">Szklanny Fatigue Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Late-Game Fatigue + Impact | Public-data framework with closer calibration</div>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        datasets = load_data()

    if not datasets:
        st.error("No data loaded. Please check the Excel file path/name.")
        return

    all_data = pd.concat(datasets.values(), ignore_index=True)

    # Sidebar filters
    st.sidebar.header("Filters")

    selected_datasets = st.sidebar.multiselect("Select Teams", options=list(datasets.keys()), default=list(datasets.keys()))
    filtered_data = all_data[all_data["dataset"].isin(selected_datasets)].copy()

    all_players = sorted(filtered_data["player"].dropna().unique())
    selected_players = st.sidebar.multiselect("Select Players (optional)", options=all_players, default=[])

    if selected_players:
        filtered_data = filtered_data[filtered_data["player"].isin(selected_players)].copy()

    b2b_filter = st.sidebar.radio("Game Type", options=["All Games", "B2B Only", "Non-B2B Only"])
    if b2b_filter == "B2B Only":
        filtered_data = filtered_data[filtered_data["is_b2b"] == True]
    elif b2b_filter == "Non-B2B Only":
        filtered_data = filtered_data[filtered_data["is_b2b"] == False]

    outcome_filter = st.sidebar.radio("Game Outcome", options=["All", "Wins Only", "Losses Only"])
    if outcome_filter == "Wins Only":
        filtered_data = filtered_data[filtered_data["is_win"] == True]
    elif outcome_filter == "Losses Only":
        filtered_data = filtered_data[filtered_data["is_win"] == False]

    st.markdown("---")

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Games", f"{filtered_data['game_date'].nunique():,}")
    with col2:
        st.metric("Players", int(filtered_data["player"].nunique()))
    with col3:
        q1 = filtered_data[filtered_data["qtr_num"] == 1]
        q1_fg = (q1["fgm"].sum() / q1["fga"].sum() * 100) if q1["fga"].sum() > 0 else 0
        st.metric("Q1 FG%", f"{q1_fg:.1f}%")
    with col4:
        q4 = filtered_data[filtered_data["qtr_num"] == 4]
        q4_fg = (q4["fgm"].sum() / q4["fga"].sum() * 100) if q4["fga"].sum() > 0 else 0
        st.metric("Q4 FG%", f"{q4_fg:.1f}%")
    with col5:
        st.metric("Q4 Change", f"{(q4_fg-q1_fg):+.1f}%", delta_color="inverse")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Quarter Analysis",
        "B2B Impact",
        "Player Breakdown",
        "Fatigue Proxies",
        "Szklanny Metrics + Predictions"
    ])

    # -------------------------------------------------------------------------
    # TAB 1
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("FG% by Quarter")
        colA, colB = st.columns(2)

        with colA:
            quarter_data = []
            for ds in filtered_data["dataset"].unique():
                ds_data = filtered_data[filtered_data["dataset"] == ds]
                for qn in [1,2,3,4]:
                    qd = ds_data[ds_data["qtr_num"] == qn]
                    if qd["fga"].sum() > 0:
                        fg = qd["fgm"].sum() / qd["fga"].sum() * 100
                        quarter_data.append({"Dataset": ds, "Quarter": f"Q{qn}", "FG%": fg})
            if quarter_data:
                quarter_df = pd.DataFrame(quarter_data)
                fig = px.bar(quarter_df, x="Quarter", y="FG%", color="Dataset", barmode="group", title="FG% by Quarter")
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)

        with colB:
            change_data = []
            for ds in filtered_data["dataset"].unique():
                ds_data = filtered_data[filtered_data["dataset"] == ds]
                q1d = ds_data[ds_data["qtr_num"] == 1]
                q4d = ds_data[ds_data["qtr_num"] == 4]
                if q1d["fga"].sum() > 0 and q4d["fga"].sum() > 0:
                    q1fg = q1d["fgm"].sum()/q1d["fga"].sum()*100
                    q4fg = q4d["fgm"].sum()/q4d["fga"].sum()*100
                    change_data.append({"Dataset": ds, "Q4 Change": q4fg-q1fg})
            if change_data:
                change_df = pd.DataFrame(change_data)
                fig = px.bar(change_df, x="Dataset", y="Q4 Change", color="Q4 Change",
                             color_continuous_scale=["red","gray","green"], title="Q4 FG% Change from Q1")
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Back-to-Back Game Analysis")
        colA, colB = st.columns(2)

        with colA:
            rows = []
            for ds in all_data["dataset"].unique():
                if ds not in selected_datasets:
                    continue
                ds_data = all_data[all_data["dataset"] == ds]
                for flag in [False, True]:
                    sub = ds_data[ds_data["is_b2b"] == flag]
                    if sub["fga"].sum() > 0:
                        rows.append({"Dataset": ds, "Game Type": "B2B" if flag else "Normal", "FG%": sub["fgm"].sum()/sub["fga"].sum()*100})
            if rows:
                df = pd.DataFrame(rows)
                fig = px.bar(df, x="Dataset", y="FG%", color="Game Type", barmode="group", title="FG% by Game Type")
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)

        with colB:
            rows = []
            for ds in all_data["dataset"].unique():
                if ds not in selected_datasets:
                    continue
                ds_data = all_data[all_data["dataset"] == ds]
                for flag in [False, True]:
                    sub = ds_data[ds_data["is_b2b"] == flag]
                    q1d = sub[sub["qtr_num"] == 1]
                    q4d = sub[sub["qtr_num"] == 4]
                    if q1d["fga"].sum() > 0 and q4d["fga"].sum() > 0:
                        q1fg = q1d["fgm"].sum()/q1d["fga"].sum()*100
                        q4fg = q4d["fgm"].sum()/q4d["fga"].sum()*100
                        rows.append({"Dataset": ds, "Game Type":"B2B" if flag else "Normal", "Q4 Change": q4fg-q1fg})
            if rows:
                df = pd.DataFrame(rows)
                fig = px.bar(df, x="Dataset", y="Q4 Change", color="Game Type", barmode="group", title="Q4 Change: B2B vs Normal")
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Player-Level Quarter FG% (Quick Scan)")
        player_rows = []
        for player in filtered_data["player"].dropna().unique():
            p = filtered_data[filtered_data["player"] == player]
            games = p["game_date"].nunique()
            if games < 5:
                continue
            q1 = p[p["qtr_num"]==1]
            q4 = p[p["qtr_num"]==4]
            if q1["fga"].sum() > 0 and q4["fga"].sum() > 0:
                q1fg = q1["fgm"].sum()/q1["fga"].sum()*100
                q4fg = q4["fgm"].sum()/q4["fga"].sum()*100
                player_rows.append({"Player":player,"Games":games,"Q1 FG%":round(q1fg,1),"Q4 FG%":round(q4fg,1),"Q4 Change":round(q4fg-q1fg,1)})
        if player_rows:
            df = pd.DataFrame(player_rows).sort_values("Q4 Change")
            colA, colB = st.columns(2)
            with colA:
                fig = px.bar(df.head(15), x="Q4 Change", y="Player", orientation="h",
                             color="Q4 Change", color_continuous_scale=["red","gray","green"],
                             title="Bottom 15 by Q4 FG% Change")
                fig.update_layout(template="plotly_dark", height=520)
                st.plotly_chart(fig, use_container_width=True)
            with colB:
                fig = px.bar(df.tail(15), x="Q4 Change", y="Player", orientation="h",
                             color="Q4 Change", color_continuous_scale=["red","gray","green"],
                             title="Top 15 by Q4 FG% Change")
                fig.update_layout(template="plotly_dark", height=520)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Fatigue Proxy Metrics")
        st.markdown(
            """
Fatigue proxies (very rough, but useful):  
- **PF** tends to rise with fatigue  
- **STL** tends to fall with fatigue  
- **TOV** tends to rise with fatigue
"""
        )
        colA, colB = st.columns(2)
        with colA:
            rows = []
            for ds in filtered_data["dataset"].unique():
                ds_data = filtered_data[filtered_data["dataset"] == ds]
                for qn in [1,2,3,4]:
                    qd = ds_data[ds_data["qtr_num"] == qn]
                    if len(qd) == 0:
                        continue
                    rows.append({"Dataset":ds,"Quarter":f"Q{qn}","PF":qd["pf"].mean(),"STL":qd["stl"].mean(),"TOV":qd["tov"].mean()})
            if rows:
                df = pd.DataFrame(rows)
                fig = make_subplots(rows=1, cols=3, subplot_titles=["Fouls","Steals","Turnovers"])
                for i, metric in enumerate(["PF","STL","TOV"]):
                    for ds in df["Dataset"].unique():
                        sub = df[df["Dataset"] == ds]
                        fig.add_trace(go.Scatter(x=sub["Quarter"], y=sub[metric], mode="lines+markers", name=ds, showlegend=(i==0)), row=1, col=i+1)
                fig.update_layout(template="plotly_dark", height=420, title="Fatigue Proxies by Quarter")
                st.plotly_chart(fig, use_container_width=True)

        with colB:
            rows = []
            for ds in filtered_data["dataset"].unique():
                ds_data = filtered_data[filtered_data["dataset"] == ds]
                q1 = ds_data[ds_data["qtr_num"]==1]
                q4 = ds_data[ds_data["qtr_num"]==4]
                if len(q1)==0 or len(q4)==0:
                    continue
                rows.append({"Dataset":ds,"PF Change":q4["pf"].mean()-q1["pf"].mean(),"STL Change":q4["stl"].mean()-q1["stl"].mean(),"TOV Change":q4["tov"].mean()-q1["tov"].mean()})
            if rows:
                df = pd.DataFrame(rows).melt(id_vars="Dataset", var_name="Metric", value_name="Change")
                fig = px.bar(df, x="Metric", y="Change", color="Dataset", barmode="group", title="Q4 Change in Proxies")
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5 (Main)
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("Szklanny Metrics (SLFI + SLIS) + Predictions")
        st.markdown(
            """
**SLFI (Fatigue Index):** “Is he getting tired?” (FG% change, TOV change, PF change)  
- Negative = more fatigue

**SLIS (Impact Score):** “Is he still helping late?” (PTS/REB/AST/STL/BLK/TOV)  
- Positive = more late-game impact

**Key fix:** SLIS is calibrated with **CCW** (Closing Context Weight) so bench/garbage-minute players don't look like elite closers.
"""
        )

        with st.spinner("Calculating metrics..."):
            metrics_data = calculate_szklanny_metrics(filtered_data)

        if metrics_data.empty:
            st.warning("Not enough data after filters to compute metrics.")
            return

        players = sorted(metrics_data["player"].unique())
        sel_player = st.selectbox("Select Player", players, key="player_select")
        p = metrics_data[metrics_data["player"] == sel_player].sort_values("game_date")

        # Summary
        avg_slfi = float(p["slfi"].mean())
        avg_slis = float(p["slis_calibrated"].mean())
        avg_ccw = float(p["ccw"].mean())
        games = int(p["game_date"].nunique())

        colA, colB, colC, colD, colE = st.columns(5)
        colA.metric("Avg SLFI", f"{avg_slfi:+.2f}", delta="Resilient" if avg_slfi > 0 else "Fatigued")
        colB.metric("Avg SLIS (Cal)", f"{avg_slis:+.2f}", delta="Clutch" if avg_slis > 0 else "Fades")
        colC.metric("Avg CCW", f"{avg_ccw:.2f}", delta="Closer-ish" if avg_ccw >= 0.55 else "Low leverage")
        colD.metric("Games", games)

        # Classification (bench first!)
        if avg_ccw < 0.35:
            ptype = "Bench / Low Leverage"
        elif avg_slfi > 0.3 and avg_slis > 0.3:
            ptype = "Elite Closer"
        elif avg_slfi < -0.3 and avg_slis > 0.3:
            ptype = "Tired but Dominant"
        elif avg_slfi < -0.3 and avg_slis < -0.3:
            ptype = "Fades Late"
        else:
            ptype = "Steady"
        colE.metric("Player Type", ptype)

        st.markdown("---")

        # Time series
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["SLFI (Fatigue)","SLIS Calibrated (Impact)","CCW (Closing Context)"],
                            vertical_spacing=0.08)

        fig.add_trace(go.Scatter(x=p["game_date"], y=p["slfi"], mode="lines+markers", name="SLFI"), row=1, col=1)
        fig.add_trace(go.Scatter(x=p["game_date"], y=p["slfi"].rolling(5, min_periods=1).mean(), mode="lines", name="SLFI 5g"), row=1, col=1)

        fig.add_trace(go.Scatter(x=p["game_date"], y=p["slis_calibrated"], mode="lines+markers", name="SLIS Cal"), row=2, col=1)
        fig.add_trace(go.Scatter(x=p["game_date"], y=p["slis_calibrated"].rolling(5, min_periods=1).mean(), mode="lines", name="SLIS Cal 5g"), row=2, col=1)

        fig.add_trace(go.Scatter(x=p["game_date"], y=p["ccw"], mode="lines+markers", name="CCW"), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="white", row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="white", row=2, col=1)
        fig.add_hline(y=0.35, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(template="plotly_dark", height=650, title=f"{sel_player}: Fatigue vs Impact vs Context")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Team Leaderboard (Context-adjusted)")
        lb = metrics_data.groupby("player").agg(
            games=("game_date","nunique"),
            avg_slfi=("slfi","mean"),
            avg_slis_cal=("slis_calibrated","mean"),
            avg_ccw=("ccw","mean"),
            mpg=("mpg","mean")
        ).reset_index()
        lb = lb[lb["games"] >= 5].copy()

        def classify_row(r):
            if r["avg_ccw"] < 0.35:
                return "Bench / Low Leverage"
            if r["avg_slfi"] > 0.3 and r["avg_slis_cal"] > 0.3:
                return "Elite Closer"
            if r["avg_slfi"] < -0.3 and r["avg_slis_cal"] > 0.3:
                return "Tired but Dominant"
            if r["avg_slfi"] < -0.3 and r["avg_slis_cal"] < -0.3:
                return "Fades Late"
            return "Steady"

        lb["type"] = lb.apply(classify_row, axis=1)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Top Calibrated Closers (filters out bench hacks)")
            top = lb[lb["avg_ccw"] >= 0.35].nlargest(12, "avg_slis_cal")
            fig = px.bar(top, x="avg_slis_cal", y="player", orientation="h", title="Top SLIS (Calibrated)")
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            st.markdown("#### Most Fatigued (lowest SLFI)")
            bot = lb.nsmallest(12, "avg_slfi")
            fig = px.bar(bot, x="avg_slfi", y="player", orientation="h", title="Lowest SLFI (Most Fatigued)")
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(lb.sort_values(["avg_slis_cal","avg_ccw"], ascending=False).round(3), use_container_width=True)

        st.markdown("---")
        st.markdown("### Fatigue Prediction (NEXT game) + Physiological Floor")

        model, scaler, met, imp, feature_cols = build_fatigue_predictor(metrics_data)

        if model is None:
            st.warning("Not enough data to train the next-game fatigue model after filters.")
            return

        colA, colB, colC = st.columns(3)
        colA.metric("Train Acc", f"{met['train_acc']*100:.1f}%")
        colB.metric("Test Acc", f"{met['test_acc']*100:.1f}%")
        colC.metric("Training Rows", f"{len(metrics_data):,} (raw)")

        fig = px.bar(imp, x="Coefficient", y="Feature", orientation="h", title="Fatigue Model Coefficients (Next-game target)")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Predict a Scenario")
        col1, col2, col3, col4 = st.columns(4)

        # Defaults from player
        p_recent = p.dropna(subset=["slfi_last1","slfi_avg_last5","minutes_avg_last5","age"])
        if len(p_recent) > 0:
            latest = p_recent.iloc[-1]
            d_last = float(latest["slfi_last1"])
            d_avg = float(latest["slfi_avg_last5"])
            d_min = float(latest["minutes_avg_last5"])
            d_age = int(latest["age"])
        else:
            d_last, d_avg, d_min, d_age = 0.0, 0.0, 30.0, 27

        with col1:
            pred_last = st.slider("Last Game SLFI", -3.0, 3.0, d_last, 0.1)
            pred_avg = st.slider("Avg SLFI (last 5)", -2.0, 2.0, d_avg, 0.1)
        with col2:
            pred_minutes = st.slider("Avg Minutes (last 5)", 10.0, 45.0, d_min, 1.0)
            pred_age = st.slider("Age", 19, 42, d_age, 1)
        with col3:
            pred_b2b = st.selectbox("B2B?", [False, True], index=0)
            pred_rest = st.slider("Days Rest", 0, 5, 1)
        with col4:
            age_load = pred_minutes * (1 + max(0, pred_age - 28) * 0.03)
            age_b2b = float(pred_b2b) * max(0, pred_age - 30)
            recovery_penalty = max(0, 2 - pred_rest) * max(0, pred_age - 30)
            prf = compute_physiological_risk_floor(pred_age, pred_minutes, pred_b2b, pred_rest)
            st.markdown("**Physiology Floor (PRF)**")
            st.markdown(f"**{prf*100:.0f}%** minimum risk")
            st.caption(f"Age-load: {age_load:.1f} | Age×B2B: {age_b2b:.1f} | Recovery: {recovery_penalty:.1f}")

        if st.button("Predict Next-Game Fatigue Risk"):
            X_pred = np.array([[pred_last, pred_avg, pred_minutes, pred_age, age_load, age_b2b, recovery_penalty]], dtype=float)
            X_s = scaler.transform(X_pred)
            model_prob = float(model.predict_proba(X_s)[0][1])
            final_risk, prf = apply_fatigue_risk_floor(model_prob, pred_age, pred_minutes, pred_b2b, pred_rest)

            floor_applied = final_risk > model_prob + 1e-9
            colA, colB, colC, colD = st.columns(4)
            colA.metric("ML Model", f"{model_prob*100:.0f}%")
            colB.metric("PRF Floor", f"{prf*100:.0f}%", delta="APPLIED" if floor_applied else "not needed")
            colC.metric("FINAL", f"{final_risk*100:.0f}%")
            label = "HIGH" if final_risk > 0.6 else "MODERATE" if final_risk > 0.4 else "LOWER"
            colD.metric("Band", label)

            if floor_applied:
                st.warning("Physiology overrode the model. Older/heavy-minute/B2B combos cannot be 'low risk' even if recent SLFI looked fine.")
            else:
                st.info("Model probability already exceeds the physiology floor; recent fatigue signals are driving the risk.")

    st.markdown("---")
    st.caption("Szklanny Fatigue Intelligence | Rewritten model + CCW calibration so bench guys don't look like closers.")

if __name__ == "__main__":
    main()
