"""
NBA Full Model Retrain Script
- Merges all historical parquet files with current data
- Adds team vs team matchup analysis
- Adds player vs team performance tracking
- Retrains LSTM neural model on expanded dataset
- Auto-pushes to GitHub when complete
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import subprocess

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

print("="*60)
print("NBA FULL MODEL RETRAIN")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# =============================================================================
# STEP 1: MERGE ALL HISTORICAL DATA
# =============================================================================
print("\n[STEP 1] Merging all historical data...")

# Historical parquet files
hist_files = {
    '2020-21': 'C:/Users/user/NBA_Quarter_2020_21.parquet',
    '2021-22': 'C:/Users/user/NBA_Quarter_2021_22.parquet',
    '2022-23': 'C:/Users/user/NBA_Quarter_2022_23.parquet',
    '2023-24': 'C:/Users/user/NBA_Quarter_2023_24_progress.parquet',
}

# Load historical parquets
all_data = []
for season, filepath in hist_files.items():
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        df['season'] = season
        df['source'] = 'historical_parquet'
        all_data.append(df)
        print(f"  Loaded {season}: {len(df):,} rows")

# Load current combined file
combined_path = 'C:/Users/user/NBA_Quarter_ALL_Combined.xlsx'
if os.path.exists(combined_path):
    current_df = pd.read_excel(combined_path)
    # Assign seasons based on date
    current_df['game_date'] = pd.to_datetime(current_df['game_date'])

    def assign_season(date):
        if pd.isna(date):
            return None
        month, year = date.month, date.year
        if month >= 10:  # Oct-Dec = first year of season
            return f"{year}-{str(year+1)[-2:]}"
        else:  # Jan-Sep = second year of season
            return f"{year-1}-{str(year)[-2:]}"

    current_df['season'] = current_df['game_date'].apply(assign_season)
    current_df['source'] = 'excel_combined'
    all_data.append(current_df)
    print(f"  Loaded current combined: {len(current_df):,} rows")

# Concatenate all data
full_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal before dedup: {len(full_df):,} rows")

# Remove duplicates based on player, game_id, qtr
full_df['game_date'] = pd.to_datetime(full_df['game_date'])
full_df = full_df.drop_duplicates(subset=['player', 'game_id', 'qtr'], keep='last')
print(f"Total after dedup: {len(full_df):,} rows")

# Season breakdown
print("\nSeason breakdown:")
print(full_df['season'].value_counts().sort_index())

# =============================================================================
# STEP 2: AGGREGATE TO GAME LEVEL
# =============================================================================
print("\n[STEP 2] Aggregating to game level...")

# Aggregate quarter data to game totals
game_agg = full_df.groupby(['player', 'game_id', 'game_date', 'team', 'season']).agg({
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
    'win_loss': 'first'
}).reset_index()

# Rename columns
game_agg.columns = ['player', 'game_id', 'game_date', 'team', 'season',
                    'total_pts', 'total_trb', 'total_ast', 'total_stl', 'total_blk',
                    'total_tov', 'total_pf', 'total_fgm', 'total_fga', 'total_minutes', 'win_loss']

# Add derived stats
game_agg['fg_pct'] = game_agg['total_fgm'] / game_agg['total_fga'].replace(0, 1)
game_agg['ts_pct'] = game_agg['total_pts'] / (2 * game_agg['total_fga'].replace(0, 1))
game_agg['tov_rate'] = game_agg['total_tov'] / (game_agg['total_fga'] + 0.44 * game_agg['total_ast'] + game_agg['total_tov']).replace(0, 1)
game_agg['usage_proxy'] = (game_agg['total_fga'] + game_agg['total_ast'] + game_agg['total_tov']) / game_agg['total_minutes'].replace(0, 1)
game_agg['game_score'] = (game_agg['total_pts'] + 0.4 * game_agg['total_fgm'] - 0.7 * game_agg['total_fga']
                          + 0.7 * game_agg['total_trb'] + 0.7 * game_agg['total_ast'] + game_agg['total_stl']
                          + 0.7 * game_agg['total_blk'] - 0.4 * game_agg['total_pf'] - game_agg['total_tov'])
game_agg['is_win'] = (game_agg['win_loss'] == 'W').astype(int)

# SPM (Simple Player Metric) approximation
game_agg['spm'] = (game_agg['total_pts'] * 0.8 + game_agg['total_trb'] * 0.4 + game_agg['total_ast'] * 0.6
                   + game_agg['total_stl'] * 1.0 + game_agg['total_blk'] * 0.8
                   - game_agg['total_tov'] * 0.8 - (game_agg['total_fga'] - game_agg['total_fgm']) * 0.3) / game_agg['total_minutes'].replace(0, 1)

# Sort by player and date
game_agg = game_agg.sort_values(['player', 'game_date']).reset_index(drop=True)

print(f"Game-level records: {len(game_agg):,}")
print(f"Unique players: {game_agg['player'].nunique():,}")

# =============================================================================
# STEP 3: EXTRACT OPPONENT FROM GAME_ID
# =============================================================================
print("\n[STEP 3] Extracting opponent teams...")

def extract_opponent(row):
    """Extract opponent team from game_id"""
    game_id = str(row['game_id'])
    team = str(row['team'])

    # Game ID format varies - try to extract teams
    # Common format: "202312150PHI" or similar
    try:
        # Try to find team codes in game_id
        # If team is in game_id, the opponent might be after it
        if len(game_id) > 9:
            teams_in_id = game_id[9:]  # Usually team codes are at end
            # Check if our team is home or away
            # For now, we'll extract from win_loss context later
    except:
        pass
    return None

# We'll build opponent data from matchup patterns in the data
# For now, mark as unknown - will be populated from schedule data

# =============================================================================
# STEP 4: BUILD TEAM VS TEAM MATCHUP DATABASE
# =============================================================================
print("\n[STEP 4] Building team vs team matchup database...")

# Group games by game_date to find matchups
game_teams = game_agg.groupby(['game_date', 'team']).agg({
    'total_pts': 'sum',
    'total_trb': 'sum',
    'total_ast': 'sum',
    'game_id': 'first'
}).reset_index()

# Find games where two teams played on same date (same game_id prefix)
team_matchups = {}
for game_id in game_agg['game_id'].unique():
    teams_in_game = game_agg[game_agg['game_id'] == game_id]['team'].unique()
    if len(teams_in_game) == 2:
        team1, team2 = sorted(teams_in_game)
        matchup_key = f"{team1}_vs_{team2}"
        if matchup_key not in team_matchups:
            team_matchups[matchup_key] = []

        # Get stats for this game
        game_data = game_agg[game_agg['game_id'] == game_id]
        team1_pts = game_data[game_data['team'] == team1]['total_pts'].sum()
        team2_pts = game_data[game_data['team'] == team2]['total_pts'].sum()
        game_date = game_data['game_date'].iloc[0]

        team_matchups[matchup_key].append({
            'date': game_date,
            'team1_pts': team1_pts,
            'team2_pts': team2_pts
        })

print(f"  Found {len(team_matchups)} unique team matchups")

# Calculate historical matchup stats
matchup_stats = {}
for matchup_key, games in team_matchups.items():
    if len(games) >= 2:  # At least 2 games for meaningful stats
        team1, team2 = matchup_key.split('_vs_')
        t1_pts = [g['team1_pts'] for g in games]
        t2_pts = [g['team2_pts'] for g in games]

        matchup_stats[matchup_key] = {
            'games_played': len(games),
            'team1': team1,
            'team2': team2,
            'team1_avg_pts': np.mean(t1_pts),
            'team2_avg_pts': np.mean(t2_pts),
            'team1_wins': sum(1 for i in range(len(t1_pts)) if t1_pts[i] > t2_pts[i]),
            'team2_wins': sum(1 for i in range(len(t2_pts)) if t2_pts[i] > t1_pts[i]),
            'avg_total': np.mean([t1 + t2 for t1, t2 in zip(t1_pts, t2_pts)]),
            'recent_3_total': np.mean([g['team1_pts'] + g['team2_pts'] for g in sorted(games, key=lambda x: x['date'], reverse=True)[:3]]) if len(games) >= 3 else None
        }

print(f"  Calculated stats for {len(matchup_stats)} matchups")

# Save matchup stats
with open('C:/Users/user/team_matchup_stats.json', 'w') as f:
    # Convert dates to strings for JSON
    matchup_stats_json = {}
    for k, v in matchup_stats.items():
        matchup_stats_json[k] = v
    json.dump(matchup_stats_json, f, indent=2, default=str)
print("  Saved to team_matchup_stats.json")

# =============================================================================
# STEP 5: BUILD PLAYER VS TEAM PERFORMANCE DATABASE
# =============================================================================
print("\n[STEP 5] Building player vs team performance database...")

# For each player, track their performance against specific teams
# This requires knowing the opponent - we'll infer from game patterns

# First, let's build a game-to-opponent mapping
game_opponents = {}
for game_id in game_agg['game_id'].unique():
    teams_in_game = game_agg[game_agg['game_id'] == game_id]['team'].unique()
    if len(teams_in_game) == 2:
        team1, team2 = teams_in_game
        game_opponents[game_id] = {team1: team2, team2: team1}

# Add opponent column to game_agg
def get_opponent(row):
    game_id = row['game_id']
    team = row['team']
    if game_id in game_opponents and team in game_opponents[game_id]:
        return game_opponents[game_id][team]
    return None

game_agg['opponent'] = game_agg.apply(get_opponent, axis=1)
games_with_opponent = game_agg[game_agg['opponent'].notna()]
print(f"  Games with opponent identified: {len(games_with_opponent):,}")

# Calculate player vs team stats
player_vs_team = games_with_opponent.groupby(['player', 'opponent']).agg({
    'total_pts': ['mean', 'std', 'count', lambda x: np.percentile(x, 90) if len(x) > 3 else np.mean(x)],
    'total_trb': 'mean',
    'total_ast': 'mean',
    'game_score': 'mean',
    'total_minutes': 'mean',
    'is_win': 'mean'
}).reset_index()

# Flatten column names
player_vs_team.columns = ['player', 'opponent', 'avg_pts', 'std_pts', 'games', 'pts_90th',
                          'avg_trb', 'avg_ast', 'avg_game_score', 'avg_minutes', 'win_pct']

# Filter to meaningful samples (at least 3 games)
player_vs_team = player_vs_team[player_vs_team['games'] >= 3]
print(f"  Player-opponent combinations (3+ games): {len(player_vs_team):,}")

# Calculate performance differential vs player's overall average
player_overall = game_agg.groupby('player').agg({
    'total_pts': 'mean',
    'total_trb': 'mean',
    'total_ast': 'mean',
    'game_score': 'mean'
}).reset_index()
player_overall.columns = ['player', 'overall_pts', 'overall_trb', 'overall_ast', 'overall_game_score']

player_vs_team = player_vs_team.merge(player_overall, on='player', how='left')

# Calculate differentials
player_vs_team['pts_diff_pct'] = (player_vs_team['avg_pts'] - player_vs_team['overall_pts']) / player_vs_team['overall_pts'].replace(0, 1) * 100
player_vs_team['game_score_diff_pct'] = (player_vs_team['avg_game_score'] - player_vs_team['overall_game_score']) / player_vs_team['overall_game_score'].replace(0, 1) * 100

# Save player vs team stats
player_vs_team.to_pickle('C:/Users/user/player_vs_team_stats.pkl')
print("  Saved to player_vs_team_stats.pkl")

# Show some interesting matchups
print("\n  Players with biggest positive matchup boosts (10+ games):")
big_boosts = player_vs_team[player_vs_team['games'] >= 10].nlargest(10, 'pts_diff_pct')
for _, row in big_boosts.head(5).iterrows():
    print(f"    {row['player']} vs {row['opponent']}: +{row['pts_diff_pct']:.1f}% ({row['avg_pts']:.1f} vs {row['overall_pts']:.1f} avg)")

print("\n  Players with biggest negative matchup impact (10+ games):")
big_drops = player_vs_team[player_vs_team['games'] >= 10].nsmallest(10, 'pts_diff_pct')
for _, row in big_drops.head(5).iterrows():
    print(f"    {row['player']} vs {row['opponent']}: {row['pts_diff_pct']:.1f}% ({row['avg_pts']:.1f} vs {row['overall_pts']:.1f} avg)")

# =============================================================================
# STEP 6: ADD Q4 PERFORMANCE CHANGE METRIC
# =============================================================================
print("\n[STEP 6] Computing Q4 clutch performance metrics...")

# Go back to quarter-level data to calculate Q4 changes
q4_data = full_df[full_df['qtr_num'] == 4].copy()
q123_data = full_df[full_df['qtr_num'] < 4].groupby(['player', 'game_id']).agg({
    'pts': 'sum'
}).reset_index()
q123_data.columns = ['player', 'game_id', 'q123_pts']

q4_summary = q4_data.groupby(['player', 'game_id']).agg({'pts': 'sum'}).reset_index()
q4_summary.columns = ['player', 'game_id', 'q4_pts']

q4_change = q4_summary.merge(q123_data, on=['player', 'game_id'], how='left')
q4_change['q4_pts_change'] = q4_change['q4_pts'] - (q4_change['q123_pts'] / 3)  # vs average of Q1-3

# Merge back to game_agg
game_agg = game_agg.merge(q4_change[['player', 'game_id', 'q4_pts_change']],
                          on=['player', 'game_id'], how='left')
game_agg['q4_pts_change'] = game_agg['q4_pts_change'].fillna(0)

print(f"  Added Q4 clutch metric to {len(game_agg):,} games")

# =============================================================================
# STEP 7: PREPARE SEQUENCE DATA FOR LSTM
# =============================================================================
print("\n[STEP 7] Preparing sequence data for LSTM training...")

# Define features for sequences
seq_features = ['total_pts', 'total_trb', 'total_ast', 'total_stl', 'total_blk',
                'total_tov', 'total_pf', 'total_fgm', 'total_fga', 'total_minutes',
                'fg_pct', 'ts_pct', 'game_score', 'usage_proxy', 'is_win', 'spm', 'q4_pts_change']

target_cols = ['total_pts', 'total_fga', 'ts_pct', 'tov_rate', 'game_score', 'spm']

# Sort by player and date
game_agg = game_agg.sort_values(['player', 'game_date']).reset_index(drop=True)

# Filter to players with enough games
player_game_counts = game_agg.groupby('player').size()
valid_players = player_game_counts[player_game_counts >= 10].index
game_agg_filtered = game_agg[game_agg['player'].isin(valid_players)].copy()
print(f"  Players with 10+ games: {len(valid_players):,}")
print(f"  Records for training: {len(game_agg_filtered):,}")

# Replace NaN/inf values and ensure numeric types
for col in seq_features + target_cols:
    if col in game_agg_filtered.columns:
        game_agg_filtered[col] = pd.to_numeric(game_agg_filtered[col], errors='coerce')
        game_agg_filtered[col] = game_agg_filtered[col].replace([np.inf, -np.inf], np.nan)
        game_agg_filtered[col] = game_agg_filtered[col].fillna(game_agg_filtered[col].median())

# Build sequences (last 5 games -> predict next game)
SEQUENCE_LENGTH = 5
X_sequences = []
y_targets = []
player_indices = []

for player in valid_players:
    player_data = game_agg_filtered[game_agg_filtered['player'] == player].sort_values('game_date')

    if len(player_data) < SEQUENCE_LENGTH + 1:
        continue

    for i in range(SEQUENCE_LENGTH, len(player_data)):
        # Get sequence of last 5 games
        seq = player_data.iloc[i-SEQUENCE_LENGTH:i][seq_features].values.astype(np.float64)
        # Target is the current game
        target = player_data.iloc[i][target_cols].values.astype(np.float64)

        if not np.any(np.isnan(seq)) and not np.any(np.isnan(target)):
            X_sequences.append(seq)
            y_targets.append(target)
            player_indices.append(player)

X = np.array(X_sequences)
y = np.array(y_targets)
print(f"  Created {len(X):,} sequences for training")

# =============================================================================
# STEP 8: SCALE DATA
# =============================================================================
print("\n[STEP 8] Scaling data...")

# Fit scalers
seq_scaler = StandardScaler()
X_flat = X.reshape(-1, X.shape[-1])
X_scaled_flat = seq_scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(X.shape)

target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y)

print(f"  X shape: {X_scaled.shape}")
print(f"  y shape: {y_scaled.shape}")

# Save scalers
scalers = {
    'seq_scaler': seq_scaler,
    'target_scaler': target_scaler,
    'target_cols': target_cols,
    'seq_features': seq_features,
    'use_enhanced': True
}
with open('C:/Users/user/player_model_scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
print("  Saved scalers to player_model_scalers.pkl")

# =============================================================================
# STEP 9: TRAIN LSTM MODEL
# =============================================================================
print("\n[STEP 9] Training LSTM model...")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)
print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")

# Build model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(seq_features))),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(target_cols))
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='huber',
    metrics=['mae']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('C:/Users/user/player_model_best.keras', monitor='val_loss', save_best_only=True)
]

# Train
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test MAE: {test_mae:.4f}")

# Make predictions and inverse transform
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_original = target_scaler.inverse_transform(y_test)

# Per-target metrics
print("\nPer-target metrics:")
for i, col in enumerate(target_cols):
    mae = np.mean(np.abs(y_pred[:, i] - y_test_original[:, i]))
    print(f"  {col}: MAE = {mae:.2f}")

# Save final model
model.save('C:/Users/user/player_model_lstm.keras')
print("\nModel saved to player_model_lstm.keras")

# =============================================================================
# STEP 10: SAVE ENHANCED TRAINING DATA
# =============================================================================
print("\n[STEP 10] Saving enhanced training data...")

# Save the full merged dataset (convert game_id to string to avoid type issues)
game_agg['game_id'] = game_agg['game_id'].astype(str)
game_agg.to_parquet('C:/Users/user/NBA_Training_Full_5Seasons.parquet', index=False)
print(f"  Saved {len(game_agg):,} game records to NBA_Training_Full_5Seasons.parquet")

# Save player historical stats
player_stats = game_agg.groupby('player').agg({
    'total_pts': ['mean', 'std', 'max', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
    'total_trb': ['mean', 'std'],
    'total_ast': ['mean', 'std'],
    'total_fga': ['mean', 'std'],
    'total_minutes': ['mean', 'count'],
    'game_score': ['mean', 'std'],
    'ts_pct': 'mean',
    'spm': 'mean'
}).reset_index()

# Flatten columns
player_stats.columns = ['player', 'pts_mean', 'pts_std', 'pts_max', 'pts_10th', 'pts_90th',
                        'trb_mean', 'trb_std', 'ast_mean', 'ast_std', 'fga_mean', 'fga_std',
                        'min_mean', 'game_count', 'gs_mean', 'gs_std', 'ts_pct', 'spm']

player_stats.to_pickle('C:/Users/user/player_historical_stats.pkl')
print(f"  Saved stats for {len(player_stats):,} players to player_historical_stats.pkl")

# =============================================================================
# STEP 11: CREATE SUMMARY REPORT
# =============================================================================
print("\n[STEP 11] Creating training summary report...")

report = f"""
NBA Model Training Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SUMMARY
------------
Total game records: {len(game_agg):,}
Unique players: {game_agg['player'].nunique():,}
Date range: {game_agg['game_date'].min().strftime('%Y-%m-%d')} to {game_agg['game_date'].max().strftime('%Y-%m-%d')}

Seasons included:
{game_agg['season'].value_counts().sort_index().to_string()}

MATCHUP DATA
------------
Team vs Team matchups tracked: {len(matchup_stats):,}
Player vs Team combinations: {len(player_vs_team):,}

MODEL TRAINING
--------------
Training sequences: {len(X_train):,}
Test sequences: {len(X_test):,}
Sequence length: {SEQUENCE_LENGTH} games

Features used: {len(seq_features)}
{', '.join(seq_features)}

Target outputs: {len(target_cols)}
{', '.join(target_cols)}

Model Architecture:
- LSTM(128) -> BatchNorm -> Dropout(0.3)
- LSTM(64) -> BatchNorm -> Dropout(0.3)
- Dense(64, relu) -> Dense(32, relu) -> Dense({len(target_cols)})

RESULTS
-------
Test Loss (Huber): {test_loss:.4f}
Test MAE: {test_mae:.4f}

Per-target MAE:
"""

for i, col in enumerate(target_cols):
    mae = np.mean(np.abs(y_pred[:, i] - y_test_original[:, i]))
    report += f"  {col}: {mae:.2f}\n"

report += """
FILES CREATED
-------------
- player_model_lstm.keras (trained model)
- player_model_scalers.pkl (scalers)
- team_matchup_stats.json (team vs team history)
- player_vs_team_stats.pkl (player vs opponent history)
- player_historical_stats.pkl (player career stats)
- NBA_Training_Full_5Seasons.parquet (merged training data)
"""

with open('C:/Users/user/training_report.txt', 'w') as f:
    f.write(report)

print(report)

# =============================================================================
# STEP 12: AUTO-PUSH TO GITHUB (will run after 5am data update)
# =============================================================================
print("\n[STEP 12] Preparing GitHub auto-push...")

# Create a script that will be called after daily update
github_push_script = '''@echo off
echo Pushing updated model to GitHub...
cd /d C:\\Users\\user
git add player_model_lstm.keras player_model_scalers.pkl team_matchup_stats.json player_vs_team_stats.pkl player_historical_stats.pkl szklanny_streamlit_app.py training_report.txt
git commit -m "Auto-update: Retrained model with 5 seasons of data - %date%"
git push
echo GitHub push complete!
'''

with open('C:/Users/user/github_push.bat', 'w') as f:
    f.write(github_push_script)

print("  Created github_push.bat for auto-push after daily updates")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
