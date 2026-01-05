"""
SDIS Data Utilities
====================

Functions for loading, processing, and validating NBA data.
Supports both Excel and Parquet formats with automatic conversion.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import streamlit as st

from .config import get_config, PLAYER_NAME_MAP


# =============================================================================
# DATA LOADING
# =============================================================================

def enforce_parquet(excel_path: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load data, converting to Parquet if needed for faster subsequent loads.

    Args:
        excel_path: Path to Excel file
        force_refresh: If True, reload from Excel even if Parquet exists

    Returns:
        DataFrame with loaded data
    """
    parquet_path = excel_path.replace('.xlsx', '.parquet')

    if not force_refresh and os.path.exists(parquet_path):
        # Check if parquet is newer than excel
        if os.path.getmtime(parquet_path) >= os.path.getmtime(excel_path):
            return pd.read_parquet(parquet_path)

    # Load from Excel and save as Parquet
    df = pd.read_excel(excel_path)
    df.to_parquet(parquet_path, index=False)
    return df


def normalize_player_names(df: pd.DataFrame, name_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Normalize player names using the provided mapping.

    Args:
        df: DataFrame with 'player' column
        name_map: Dictionary mapping old names to new names

    Returns:
        DataFrame with normalized player names
    """
    if name_map is None:
        name_map = PLAYER_NAME_MAP

    if 'player' in df.columns:
        df['player'] = df['player'].replace(name_map)

    return df


@st.cache_data(ttl=3600)
def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load and process fatigue analysis data from Excel.

    Returns:
        Dictionary of DataFrames keyed by dataset name
    """
    config = get_config()
    file_path = str(config.paths.fatigue_data_file)

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
                df = normalize_player_names(df)

                if 'quarter' in df.columns:
                    df = df.rename(columns={'quarter': 'qtr'})

                df = df[df['qtr'].astype(str).str.match(r'^Q[1-4]$', na=False)]
                df['qtr_num'] = df['qtr'].str.replace('Q', '').astype(int)
                df = df.rename(columns={'win/loss': 'win_loss'})

                # Calculate FG%
                df['fg_pct'] = np.where(df['fga'] > 0, df['fgm'] / df['fga'] * 100, np.nan)
                df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
                df['dataset'] = key
                dfs.append(df)
            except Exception:
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

            combined = combined.merge(
                game_dates[['player', 'game_date', 'is_b2b', 'days_rest']],
                on=['player', 'game_date'],
                how='left'
            )
            combined['is_b2b'] = combined['is_b2b'].fillna(False)
            combined['is_win'] = combined['win_loss'].str.upper() == 'W'

            # Load advanced data for age/mpg
            for adv_sheet in adv_sheets.get(key, []):
                try:
                    adv_df = pd.read_excel(xl, sheet_name=adv_sheet)
                    adv_df.columns = adv_df.columns.str.lower().str.strip()

                    if key == 'Sixers 2025-26':
                        adv_df = adv_df[adv_df['age'].apply(
                            lambda x: str(x).isdigit() if pd.notna(x) else False
                        )]

                    adv_df['age'] = pd.to_numeric(adv_df['age'], errors='coerce')
                    adv_df['games'] = pd.to_numeric(adv_df['games'], errors='coerce')
                    adv_df['minutes_total'] = pd.to_numeric(adv_df['minutes_total'], errors='coerce')
                    adv_df['mpg'] = adv_df['minutes_total'] / adv_df['games']

                    combined['player_clean'] = combined['player'].str.strip().str.lower()
                    adv_df['player_clean'] = adv_df['player'].str.strip().str.lower()
                    combined = combined.merge(
                        adv_df[['player_clean', 'age', 'mpg']],
                        on='player_clean',
                        how='left',
                        suffixes=('', '_adv')
                    )
                except Exception:
                    pass

            datasets[key] = combined

    return datasets


@st.cache_data(ttl=3600)
def load_combined_quarter_data(file_path: str = "NBA_Quarter_ALL_Combined.xlsx") -> Dict[str, pd.DataFrame]:
    """
    Load quarter data from a combined flat file (all teams in one sheet).

    Args:
        file_path: Path to the combined data file

    Returns:
        Dictionary of DataFrames keyed by dataset name (e.g., 'BOS 2024-25')
    """
    config = get_config()

    # Use Parquet if available
    parquet_path = file_path.replace('.xlsx', '.parquet')

    try:
        if os.path.exists(parquet_path):
            if os.path.getmtime(parquet_path) >= os.path.getmtime(file_path):
                df = pd.read_parquet(parquet_path)
            else:
                df = pd.read_excel(file_path)
                df.to_parquet(parquet_path, index=False)
        else:
            df = pd.read_excel(file_path)
            df.to_parquet(parquet_path, index=False)
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.warning(f"Could not load combined data: {e}")
        return {}

    df.columns = df.columns.str.lower().str.strip()

    # Normalize player names
    df = normalize_player_names(df)

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

    # Split by dataset column
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

        subset = subset.merge(
            game_dates[['player', 'game_date', 'is_b2b', 'days_rest']],
            on=['player', 'game_date'],
            how='left'
        )
        subset['is_b2b'] = subset['is_b2b'].fillna(False)
        subset['is_win'] = subset['win_loss'].astype(str).str.upper() == 'W'

        # Default age if not present
        if 'age' not in subset.columns:
            subset['age'] = 27

        datasets[dataset_name] = subset

    return datasets


@st.cache_data(ttl=3600)
def load_injury_data(file_path: str = "NBA_Injuries_Combined.xlsx") -> pd.DataFrame:
    """
    Load NBA injury data.

    Args:
        file_path: Path to injury data file

    Returns:
        DataFrame with injury data
    """
    # Try parquet first
    parquet_path = file_path.replace('.xlsx', '.parquet')

    try:
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_excel(file_path)
            df.to_parquet(parquet_path, index=False)
    except Exception:
        return pd.DataFrame()

    df.columns = df.columns.str.lower().str.strip()
    return df


def prepare_player_game_history(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """
    Prepare aggregated game-level history for a player.

    Args:
        df: Raw quarter-level data
        player_name: Player name to filter for

    Returns:
        DataFrame with one row per game, aggregated stats
    """
    player_df = df[df['player'] == player_name].copy()

    if len(player_df) == 0:
        return pd.DataFrame()

    # Aggregate quarters to game level
    game_stats = player_df.groupby(['game_id', 'game_date']).agg({
        'pts': 'sum',
        'trb': 'sum',
        'ast': 'sum',
        'stl': 'sum',
        'blk': 'sum',
        'tov': 'sum',
        'pf': 'sum',
        'fgm': 'sum',
        'fga': 'sum',
        'minutes': 'sum',
        'team': 'first',
        'win_loss': 'first',
        'is_b2b': 'first' if 'is_b2b' in player_df.columns else lambda x: False,
        'days_rest': 'first' if 'days_rest' in player_df.columns else lambda x: 3,
    }).reset_index()

    # Rename for consistency
    game_stats = game_stats.rename(columns={
        'pts': 'total_pts',
        'trb': 'total_trb',
        'ast': 'total_ast',
        'stl': 'total_stl',
        'blk': 'total_blk',
        'tov': 'total_tov',
        'pf': 'total_pf',
        'fgm': 'total_fgm',
        'minutes': 'total_minutes',
    })

    # Calculate derived stats
    game_stats['fg_pct'] = np.where(
        game_stats['fga'] > 0,
        game_stats['total_fgm'] / game_stats['fga'] * 100,
        0
    )

    # True shooting percentage (simplified)
    game_stats['ts_pct'] = np.where(
        game_stats['fga'] > 0,
        game_stats['total_pts'] / (2 * game_stats['fga']),
        0
    )

    # Turnover rate
    game_stats['tov_rate'] = np.where(
        (game_stats['fga'] + game_stats['total_tov']) > 0,
        game_stats['total_tov'] / (game_stats['fga'] + game_stats['total_tov']),
        0
    )

    # Game score (simplified Hollinger formula)
    game_stats['game_score'] = (
        game_stats['total_pts'] +
        0.4 * game_stats['total_fgm'] -
        0.7 * game_stats['fga'] +
        0.7 * game_stats['total_trb'] +
        0.7 * game_stats['total_ast'] +
        game_stats['total_stl'] +
        0.7 * game_stats['total_blk'] -
        0.4 * game_stats['total_pf'] -
        game_stats['total_tov']
    )

    # Win indicator
    game_stats['is_win'] = game_stats['win_loss'].astype(str).str.upper() == 'W'

    # Sort by date
    game_stats = game_stats.sort_values('game_date').reset_index(drop=True)

    return game_stats


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing
