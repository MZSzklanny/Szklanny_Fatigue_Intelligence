"""
SDIS Neural Network Models
===========================
Player Performance Model (per-game + per-quarter)
Team Outcome Model (win prob / margin)

Uses PyTorch with tabular MLP + sequence heads.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# DATA PREPARATION
# =============================================================================

class PlayerSequenceDataset(Dataset):
    """
    Dataset for player performance prediction.

    Creates sequences of last N games/quarters for each player-game.
    Handles both game-level and quarter-level features.
    Includes StandardScaler normalization for stable training.
    """

    def __init__(self, df, sequence_length=10, quarter_sequence_length=20,
                 target_cols=None, is_quarter_level=False,
                 seq_scaler=None, target_scaler=None):
        """
        Args:
            df: DataFrame with player-game or player-quarter data
            sequence_length: Number of previous games to include
            quarter_sequence_length: Number of previous quarters to include
            target_cols: List of target column names
            is_quarter_level: If True, predicts quarter-level stats
            seq_scaler: Pre-fitted scaler for sequence features (for inference)
            target_scaler: Pre-fitted scaler for targets (for inference)
        """
        self.sequence_length = sequence_length
        self.quarter_sequence_length = quarter_sequence_length
        self.is_quarter_level = is_quarter_level

        # Default targets
        if target_cols is None:
            if is_quarter_level:
                target_cols = ['pts', 'fga', 'fgm', 'tov', 'pf', 'minutes']
            else:
                target_cols = ['total_pts', 'total_fga', 'ts_pct', 'tov_rate', 'game_score', 'spm']

        self.target_cols = target_cols
        self.df = df.copy()

        # Scalers for normalization
        self.seq_scaler = seq_scaler if seq_scaler else StandardScaler()
        self.target_scaler = target_scaler if target_scaler else StandardScaler()
        self._fit_scalers = seq_scaler is None  # Only fit if not provided

        # Prepare data
        self._prepare_features()
        self._create_sequences()

    def _prepare_features(self):
        """Engineer features for the model."""
        df = self.df

        # Ensure datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['player', 'game_date', 'qtr_num'])

        # Basic per-quarter features (already in data)
        quarter_features = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf',
                          'fgm', 'fga', 'minutes']

        # Calculate efficiency metrics per quarter
        df['fg_pct'] = np.where(df['fga'] > 0, df['fgm'] / df['fga'], 0)

        # Game-level aggregations
        game_agg = df.groupby(['player', 'game_date', 'game_id']).agg({
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
            'is_b2b': 'first',
            'days_rest': 'first',
            'age': 'first'
        }).reset_index()

        # Rename for clarity
        game_agg.columns = ['player', 'game_date', 'game_id',
                          'total_pts', 'total_trb', 'total_ast', 'total_stl',
                          'total_blk', 'total_tov', 'total_pf', 'total_fgm',
                          'total_fga', 'total_minutes', 'team', 'win_loss',
                          'is_b2b', 'days_rest', 'age']

        # Efficiency metrics
        game_agg['fg_pct'] = np.where(game_agg['total_fga'] > 0,
                                       game_agg['total_fgm'] / game_agg['total_fga'], 0)
        game_agg['ts_pct'] = np.where(game_agg['total_fga'] > 0,
                                       game_agg['total_pts'] / (2 * game_agg['total_fga']), 0)

        # Turnover rate
        game_agg['tov_rate'] = np.where(
            (game_agg['total_fga'] + game_agg['total_tov']) > 0,
            game_agg['total_tov'] / (game_agg['total_fga'] + game_agg['total_tov']),
            0
        )

        # Game Score (simplified)
        game_agg['game_score'] = (
            game_agg['total_pts'] +
            0.4 * game_agg['total_fgm'] -
            0.7 * game_agg['total_fga'] +
            0.7 * game_agg['total_trb'] +
            0.7 * game_agg['total_ast'] +
            game_agg['total_stl'] +
            0.7 * game_agg['total_blk'] -
            game_agg['total_tov'] -
            0.4 * game_agg['total_pf']
        )

        # Usage proxy (share of team shots when on floor)
        game_agg['usage_proxy'] = np.where(
            game_agg['total_minutes'] > 0,
            game_agg['total_fga'] / game_agg['total_minutes'] * 12,  # Per 12 min
            0
        )

        # Win indicator
        game_agg['is_win'] = (game_agg['win_loss'] == 'W').astype(int)

        # Calculate SPM (Szklanny Performance Metric) - Q1-3 to Q4 resilience
        spm_df = self._calculate_spm(df)
        if spm_df is not None and len(spm_df) > 0:
            game_agg = game_agg.merge(
                spm_df[['player', 'game_id', 'spm', 'q4_pts_change', 'q4_fg_change']],
                on=['player', 'game_id'], how='left'
            )
            game_agg['spm'] = game_agg['spm'].fillna(0)
            game_agg['q4_pts_change'] = game_agg['q4_pts_change'].fillna(0)
            game_agg['q4_fg_change'] = game_agg['q4_fg_change'].fillna(0)
        else:
            game_agg['spm'] = 0
            game_agg['q4_pts_change'] = 0
            game_agg['q4_fg_change'] = 0

        self.game_df = game_agg
        self.quarter_df = df

    def _calculate_spm(self, df):
        """Calculate SPM (late-game resilience) for each player-game."""
        try:
            if 'qtr_num' not in df.columns:
                return None
            q13 = df[df['qtr_num'].isin([1,2,3])].groupby(['player','game_id']).agg({
                'pts':'mean','fgm':'sum','fga':'sum','tov':'mean','blk':'mean','stl':'mean'
            }).reset_index()
            q13.columns = ['player','game_id','q13_pts','q13_fgm','q13_fga','q13_tov','q13_blk','q13_stl']
            q4 = df[df['qtr_num']==4].groupby(['player','game_id']).agg({
                'pts':'sum','fgm':'sum','fga':'sum','tov':'sum','blk':'sum','stl':'sum'
            }).reset_index()
            q4.columns = ['player','game_id','q4_pts','q4_fgm','q4_fga','q4_tov','q4_blk','q4_stl']
            spm_data = q13.merge(q4, on=['player','game_id'], how='inner')
            spm_data['q4_fg_pct'] = np.where(spm_data['q4_fga']>0, spm_data['q4_fgm']/spm_data['q4_fga'], 0)
            spm_data['q13_fg_pct'] = np.where(spm_data['q13_fga']>0, spm_data['q13_fgm']/spm_data['q13_fga'], 0)
            spm_data['q4_fg_change'] = spm_data['q4_fg_pct'] - spm_data['q13_fg_pct']
            spm_data['q4_pts_change'] = spm_data['q4_pts'] - spm_data['q13_pts']
            spm_data['tov_resil'] = spm_data['q13_tov'] - spm_data['q4_tov']
            spm_data['spm'] = spm_data['q4_fg_change']*15 + spm_data['q4_pts_change']*0.3 + spm_data['tov_resil']*2
            return spm_data
        except Exception as e:
            return None

    def _create_sequences(self):
        """Create sequences for each prediction point."""
        self.samples = []

        if self.is_quarter_level:
            self._create_quarter_sequences()
        else:
            self._create_game_sequences()

    def _create_game_sequences(self):
        """Create game-level sequences with normalization."""
        df = self.game_df.sort_values(['player', 'game_date'])

        # Features for sequence
        self.seq_features = ['total_pts', 'total_trb', 'total_ast', 'total_stl',
                            'total_blk', 'total_tov', 'total_pf', 'total_fgm',
                            'total_fga', 'total_minutes', 'fg_pct', 'ts_pct',
                            'game_score', 'usage_proxy', 'is_win', 'spm', 'q4_pts_change']

        # Static features (for current game context)
        static_features = ['is_b2b', 'days_rest', 'age']

        # Fit scalers on all data if needed
        if self._fit_scalers:
            all_seq_data = df[self.seq_features].values
            all_target_data = df[self.target_cols].values
            self.seq_scaler.fit(all_seq_data)
            self.target_scaler.fit(all_target_data)

        for player in df['player'].unique():
            player_df = df[df['player'] == player].reset_index(drop=True)

            for i in range(self.sequence_length, len(player_df)):
                # Get sequence of previous games
                seq_data = player_df.iloc[i-self.sequence_length:i][self.seq_features].values

                # Apply scaling to sequence
                seq_data_scaled = self.seq_scaler.transform(seq_data)

                # Current game static features
                static_data = player_df.iloc[i][static_features].values.astype(float)

                # Targets (scaled)
                targets = player_df.iloc[i][self.target_cols].values.astype(float).reshape(1, -1)
                targets_scaled = self.target_scaler.transform(targets).flatten()

                self.samples.append({
                    'sequence': seq_data_scaled.astype(np.float32),
                    'static': static_data.astype(np.float32),
                    'targets': targets_scaled.astype(np.float32),
                    'player': player,
                    'game_date': player_df.iloc[i]['game_date']
                })

    def _create_quarter_sequences(self):
        """Create quarter-level sequences with normalization."""
        df = self.quarter_df.sort_values(['player', 'game_date', 'qtr_num'])

        # Features for sequence
        self.seq_features = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf',
                            'fgm', 'fga', 'minutes', 'fg_pct', 'qtr_num']

        # Fit scalers on all data if needed
        if self._fit_scalers:
            all_seq_data = df[self.seq_features].values
            all_target_data = df[self.target_cols].values
            self.seq_scaler.fit(all_seq_data)
            self.target_scaler.fit(all_target_data)

        for player in df['player'].unique():
            player_df = df[df['player'] == player].reset_index(drop=True)

            for i in range(self.quarter_sequence_length, len(player_df)):
                # Get sequence of previous quarters
                seq_data = player_df.iloc[i-self.quarter_sequence_length:i][self.seq_features].values

                # Apply scaling
                seq_data_scaled = self.seq_scaler.transform(seq_data)

                # Current quarter context
                current_qtr = player_df.iloc[i]['qtr_num']

                # Targets (scaled)
                targets = player_df.iloc[i][self.target_cols].values.astype(float).reshape(1, -1)
                targets_scaled = self.target_scaler.transform(targets).flatten()

                self.samples.append({
                    'sequence': seq_data_scaled.astype(np.float32),
                    'static': np.array([current_qtr], dtype=np.float32),
                    'targets': targets_scaled.astype(np.float32),
                    'player': player,
                    'game_date': player_df.iloc[i]['game_date']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'sequence': torch.tensor(sample['sequence']),
            'static': torch.tensor(sample['static']),
            'targets': torch.tensor(sample['targets'])
        }


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class SequenceEncoder(nn.Module):
    """
    Encodes a sequence of previous games/quarters using LSTM or Transformer.

    ENHANCED VERSION (v2):
    - Bidirectional LSTM for better context capture
    - Layer normalization for training stability
    - Increased hidden_dim (128) for more expressive power
    - Support for MC Dropout during inference
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 encoder_type='lstm', dropout=0.2, bidirectional=True):
        super().__init__()

        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional if encoder_type in ['lstm', 'gru'] else False
        self.dropout = dropout

        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
            # Output dim doubles if bidirectional
            self.output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
            # Layer normalization for stability
            self.layer_norm = nn.LayerNorm(self.output_dim)

        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = input_dim
            self.layer_norm = nn.LayerNorm(self.output_dim)

        elif encoder_type == 'gru':
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
            self.output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
            self.layer_norm = nn.LayerNorm(self.output_dim)

        # Dropout layer for MC Dropout during inference
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, mc_dropout=False):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mc_dropout: If True, keep dropout active for MC inference
        Returns:
            (batch, output_dim)
        """
        if self.encoder_type in ['lstm', 'gru']:
            output, _ = self.encoder(x)
            # Take last timestep (for bidirectional, this contains both directions)
            encoded = output[:, -1, :]
        else:
            # Transformer
            output = self.encoder(x)
            # Mean pool over sequence
            encoded = output.mean(dim=1)

        # Apply layer normalization
        encoded = self.layer_norm(encoded)

        # Apply dropout (active during training OR when mc_dropout=True)
        if self.training or mc_dropout:
            encoded = self.output_dropout(encoded)

        return encoded


class TabularMLP(nn.Module):
    """
    MLP for processing static/tabular features.

    ENHANCED VERSION (v2):
    - Layer normalization option for stability
    - MC Dropout support for uncertainty estimation
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2, use_layer_norm=True):
        super().__init__()

        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            else:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x, mc_dropout=False):
        # For MC dropout, we need to manually handle dropout layers
        if mc_dropout and not self.training:
            # Apply layers manually with dropout forced on
            out = x
            for layer in self.mlp:
                if isinstance(layer, nn.Dropout):
                    out = nn.functional.dropout(out, p=self.dropout, training=True)
                else:
                    out = layer(out)
            return out
        return self.mlp(x)


class LegacySequenceEncoder(nn.Module):
    """Legacy sequence encoder matching original saved model structure."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        output, _ = self.encoder(x)
        return output[:, -1, :]


class LegacyTabularMLP(nn.Module):
    """Legacy MLP matching original saved model structure."""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.mlp(x)


class PlayerPerformanceModelLegacy(nn.Module):
    """
    Legacy model architecture for loading pre-trained models.
    Uses BatchNorm, unidirectional LSTM, simpler fusion.
    Structure matches original saved state_dict keys.
    """

    def __init__(self, seq_input_dim, static_input_dim, target_dims,
                 seq_hidden_dim=64, mlp_hidden_dims=[64, 32],
                 encoder_type='lstm', dropout=0.2):
        super().__init__()

        # Legacy sequence encoder (matches state_dict: seq_encoder.encoder.*)
        self.seq_encoder = LegacySequenceEncoder(
            input_dim=seq_input_dim,
            hidden_dim=seq_hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # Legacy static MLP (matches state_dict: static_mlp.mlp.*)
        self.static_mlp = LegacyTabularMLP(
            input_dim=static_input_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )

        # Combined dimension
        combined_dim = seq_hidden_dim + mlp_hidden_dims[-1]

        # Legacy fusion layer with BatchNorm
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )

        # Multi-task prediction heads (legacy: simpler)
        self.target_dims = target_dims
        self.heads = nn.ModuleDict()

        for target_name, dim in target_dims.items():
            self.heads[target_name] = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, dim)
            )

    def forward(self, sequence, static, mc_dropout=False):
        # Encode sequence
        seq_emb = self.seq_encoder(sequence)

        # Encode static features
        static_emb = self.static_mlp(static)

        # Combine and fuse
        combined = torch.cat([seq_emb, static_emb], dim=1)
        fused = self.fusion(combined)

        # Predict each target
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(fused)

        return outputs


class PlayerPerformanceModel(nn.Module):
    """
    Main model for predicting player performance.

    ENHANCED VERSION (v2):
    - Bidirectional LSTM (128 hidden) for better temporal patterns
    - Layer normalization throughout for training stability
    - MC Dropout support for uncertainty-aware predictions
    - Larger fusion layer (128) for better feature combination

    Combines:
    - Sequence encoder (LSTM/Transformer) for historical performance
    - Tabular MLP for static features
    - Multi-task prediction heads for different targets
    """

    def __init__(self, seq_input_dim, static_input_dim, target_dims,
                 seq_hidden_dim=128, mlp_hidden_dims=[64, 32],
                 encoder_type='lstm', dropout=0.2, bidirectional=True):
        super().__init__()

        self.dropout = dropout

        # Sequence encoder (enhanced with bidirectional + layer norm)
        self.seq_encoder = SequenceEncoder(
            input_dim=seq_input_dim,
            hidden_dim=seq_hidden_dim,
            encoder_type=encoder_type,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # Static feature MLP (enhanced with layer norm)
        self.static_mlp = TabularMLP(
            input_dim=static_input_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            use_layer_norm=True
        )

        # Combined dimension (note: bidirectional doubles seq encoder output)
        combined_dim = self.seq_encoder.output_dim + self.static_mlp.output_dim

        # Enhanced fusion layer with layer norm (128 units for more capacity)
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout)
        )

        # Multi-task prediction heads
        self.target_dims = target_dims
        self.heads = nn.ModuleDict()

        for target_name, dim in target_dims.items():
            self.heads[target_name] = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.LayerNorm(32),
                nn.Linear(32, dim)
            )

    def forward(self, sequence, static, mc_dropout=False):
        """
        Args:
            sequence: (batch, seq_len, seq_features)
            static: (batch, static_features)
            mc_dropout: If True, enable dropout during inference for uncertainty estimation
        Returns:
            dict of predictions for each target
        """
        # Encode sequence (with optional MC dropout)
        seq_emb = self.seq_encoder(sequence, mc_dropout=mc_dropout)

        # Encode static features (with optional MC dropout)
        static_emb = self.static_mlp(static, mc_dropout=mc_dropout)

        # Combine
        combined = torch.cat([seq_emb, static_emb], dim=1)

        # Apply fusion with MC dropout if requested
        if mc_dropout and not self.training:
            fused = combined
            for layer in self.fusion:
                if isinstance(layer, nn.Dropout):
                    fused = nn.functional.dropout(fused, p=self.dropout, training=True)
                else:
                    fused = layer(fused)
        else:
            fused = self.fusion(combined)

        # Predict each target
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(fused)

        return outputs

    def predict_with_uncertainty(self, sequence, static, n_samples=10):
        """
        Make predictions with uncertainty estimation using MC Dropout.

        Runs n_samples forward passes with dropout active and returns
        mean predictions plus standard deviations.

        Args:
            sequence: (batch, seq_len, seq_features)
            static: (batch, static_features)
            n_samples: Number of MC samples (default 10)

        Returns:
            dict with 'mean' and 'std' for each target
        """
        self.eval()  # Ensure eval mode (MC dropout handled separately)

        all_predictions = {name: [] for name in self.target_dims.keys()}

        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.forward(sequence, static, mc_dropout=True)
                for name, pred in preds.items():
                    all_predictions[name].append(pred.cpu().numpy())

        results = {}
        for name in self.target_dims.keys():
            stacked = np.stack(all_predictions[name], axis=0)  # (n_samples, batch, dim)
            results[name] = {
                'mean': np.mean(stacked, axis=0),
                'std': np.std(stacked, axis=0)
            }

        return results


# =============================================================================
# TEAM OUTCOME MODEL
# =============================================================================

class TeamOutcomeModel(nn.Module):
    """
    Model for predicting team-level outcomes.

    Targets:
    - Win probability
    - Point differential
    - Offensive/Defensive rating proxies
    - Pace
    """

    def __init__(self, team_feature_dim, player_feature_dim=None,
                 hidden_dims=[128, 64], dropout=0.2):
        super().__init__()

        input_dim = team_feature_dim
        if player_feature_dim:
            input_dim += player_feature_dim

        # Main MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Prediction heads
        self.win_prob_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.margin_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.ortg_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.drtg_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.pace_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, team_features, player_features=None):
        if player_features is not None:
            x = torch.cat([team_features, player_features], dim=1)
        else:
            x = team_features

        features = self.backbone(x)

        return {
            'win_prob': self.win_prob_head(features),
            'margin': self.margin_head(features),
            'ortg': self.ortg_head(features),
            'drtg': self.drtg_head(features),
            'pace': self.pace_head(features)
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable task weights.
    Uses uncertainty weighting (Kendall et al., 2018).
    """

    def __init__(self, task_names, task_types=None):
        super().__init__()

        self.task_names = task_names
        self.task_types = task_types or {name: 'regression' for name in task_names}

        # Learnable log variances for uncertainty weighting
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict of {task_name: prediction tensor}
            targets: dict of {task_name: target tensor}
        """
        total_loss = 0
        losses = {}

        for name in self.task_names:
            pred = predictions[name]
            target = targets[name]

            if self.task_types[name] == 'regression':
                task_loss = nn.MSELoss()(pred.squeeze(), target.squeeze())
            elif self.task_types[name] == 'classification':
                task_loss = nn.BCELoss()(pred.squeeze(), target.squeeze())
            else:
                task_loss = nn.MSELoss()(pred.squeeze(), target.squeeze())

            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[name])
            weighted_loss = precision * task_loss + self.log_vars[name]

            total_loss += weighted_loss
            losses[name] = task_loss.item()

        return total_loss, losses


def train_player_model(model, train_loader, val_loader,
                       target_names, epochs=50, lr=1e-3,
                       early_stop_patience=10):
    """
    Train the player performance model.
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    criterion = MultiTaskLoss(target_names)
    criterion = criterion.to(DEVICE)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            sequence = batch['sequence'].to(DEVICE)
            static = batch['static'].to(DEVICE)
            targets_tensor = batch['targets'].to(DEVICE)

            # Split targets
            targets = {name: targets_tensor[:, i] for i, name in enumerate(target_names)}

            optimizer.zero_grad()
            predictions = model(sequence, static)
            loss, task_losses = criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                sequence = batch['sequence'].to(DEVICE)
                static = batch['static'].to(DEVICE)
                targets_tensor = batch['targets'].to(DEVICE)

                targets = {name: targets_tensor[:, i] for i, name in enumerate(target_names)}

                predictions = model(sequence, static)
                loss, _ = criterion(predictions, targets)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history


# =============================================================================
# PREDICTION & INFERENCE
# =============================================================================

def predict_player_performance(model, player_history_df, static_features,
                               seq_scaler, target_scaler, target_names,
                               sequence_length=10):
    """
    Predict performance for a player given their recent history.

    Args:
        model: Trained PlayerPerformanceModel
        player_history_df: DataFrame with player's recent games (game-level aggregated)
        static_features: Dict of static features (is_b2b, days_rest, age)
        seq_scaler: Fitted StandardScaler for sequence features
        target_scaler: Fitted StandardScaler for targets
        target_names: List of target names
        sequence_length: Number of games in sequence

    Returns:
        Dict of predicted values (denormalized to original scale)
    """
    model.eval()

    # Prepare sequence
    seq_features = ['total_pts', 'total_trb', 'total_ast', 'total_stl',
                   'total_blk', 'total_tov', 'total_pf', 'total_fgm',
                   'total_fga', 'total_minutes', 'fg_pct', 'ts_pct',
                   'game_score', 'usage_proxy', 'is_win']

    # Get last N games
    recent = player_history_df.tail(sequence_length)
    sequence = recent[seq_features].values.astype(np.float32)

    # Pad if needed
    if len(sequence) < sequence_length:
        padding = np.zeros((sequence_length - len(sequence), len(seq_features)))
        sequence = np.vstack([padding, sequence])

    # Scale sequence
    sequence_scaled = seq_scaler.transform(sequence)

    # Prepare tensors
    seq_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor([
        static_features.get('is_b2b', 0),
        static_features.get('days_rest', 2),
        static_features.get('age', 27)
    ], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(seq_tensor, static_tensor)

    # Denormalize predictions
    pred_array = np.array([[predictions[name].cpu().numpy()[0, 0] for name in target_names]])
    pred_denorm = target_scaler.inverse_transform(pred_array)[0]

    results = {name: pred_denorm[i] for i, name in enumerate(target_names)}
    return results


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def build_and_train_player_model(quarter_df, game_level=True, use_enhanced=True):
    """
    Main function to build and train the player performance model.

    Args:
        quarter_df: DataFrame with quarter-level data
        game_level: If True, predict game totals; else quarter-level
        use_enhanced: If True, use enhanced v2 architecture (bidirectional LSTM, 128 hidden)

    Returns:
        Trained model, scalers, history
    """
    print("="*60)
    print("Building Player Performance Model")
    if use_enhanced:
        print("Using ENHANCED v2 architecture (bidirectional LSTM, 128 hidden)")
    else:
        print("Using legacy architecture (unidirectional LSTM, 64 hidden)")
    print("="*60)

    # Define targets
    if game_level:
        target_cols = ['total_pts', 'total_fga', 'ts_pct', 'tov_rate', 'game_score', 'spm']
    else:
        target_cols = ['pts', 'fga', 'fg_pct', 'tov', 'minutes']

    target_names = target_cols

    print(f"\nTargets: {target_names}")
    print(f"Level: {'Game' if game_level else 'Quarter'}")

    # Create dataset
    print("\nPreparing dataset...")
    dataset = PlayerSequenceDataset(
        quarter_df,
        sequence_length=10,
        target_cols=target_cols,
        is_quarter_level=not game_level
    )

    print(f"Total samples: {len(dataset)}")

    if len(dataset) < 100:
        print("ERROR: Not enough samples for training")
        return None, None, None

    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Get dimensions from sample
    sample = dataset[0]
    seq_input_dim = sample['sequence'].shape[1]
    static_input_dim = sample['static'].shape[0]

    print(f"\nSequence features: {seq_input_dim}")
    print(f"Static features: {static_input_dim}")

    # Build model
    target_dims = {name: 1 for name in target_names}

    if use_enhanced:
        # Enhanced v2 architecture for tighter variance
        model = PlayerPerformanceModel(
            seq_input_dim=seq_input_dim,
            static_input_dim=static_input_dim,
            target_dims=target_dims,
            seq_hidden_dim=128,  # Increased from 64
            mlp_hidden_dims=[64, 32],
            encoder_type='lstm',
            dropout=0.2,
            bidirectional=True  # Bidirectional for better context
        )
    else:
        # Legacy architecture
        model = PlayerPerformanceModel(
            seq_input_dim=seq_input_dim,
            static_input_dim=static_input_dim,
            target_dims=target_dims,
            seq_hidden_dim=64,
            mlp_hidden_dims=[64, 32],
            encoder_type='lstm',
            dropout=0.2,
            bidirectional=False
        )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\nTraining...")
    model, history = train_player_model(
        model, train_loader, val_loader,
        target_names=target_names,
        epochs=50,
        lr=1e-3,
        early_stop_patience=10
    )

    print("\nTraining complete!")
    return model, dataset, history


if __name__ == "__main__":
    # Test with sample data
    print("Loading data...")
    import os
    import joblib

    # Try parquet first (faster), fall back to Excel
    parquet_path = r"C:\Users\user\NBA_Quarter_ALL_Combined.parquet"
    excel_path = r"C:\Users\user\NBA_Quarter_ALL_Combined.xlsx"

    if os.path.exists(parquet_path):
        print(f"Loading from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(excel_path):
        print(f"Loading from Excel: {excel_path}")
        df = pd.read_excel(excel_path)
    else:
        print(f"Data file not found!")
        df = None

    if df is not None:
        df.columns = df.columns.str.lower().str.strip()

        # Add required columns if missing
        if 'is_b2b' not in df.columns:
            df['is_b2b'] = False
        if 'days_rest' not in df.columns:
            df['days_rest'] = 2
        if 'age' not in df.columns:
            df['age'] = 27
        if 'qtr_num' not in df.columns:
            df['qtr_num'] = df['qtr'].str.replace('Q', '').astype(int)

        print(f"Data loaded: {len(df):,} rows")

        # Train enhanced v2 model by default
        use_enhanced = True
        model, dataset, history = build_and_train_player_model(df, game_level=True, use_enhanced=use_enhanced)

        if model and dataset:
            # Save model with v2 suffix for enhanced version
            if use_enhanced:
                model_path = r"C:\Users\user\player_performance_model_v2.pth"
            else:
                model_path = r"C:\Users\user\player_performance_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"\nModel saved to {model_path}")

            # Save scalers for inference
            scalers_path = r"C:\Users\user\player_model_scalers.pkl"
            joblib.dump({
                'seq_scaler': dataset.seq_scaler,
                'target_scaler': dataset.target_scaler,
                'target_cols': dataset.target_cols,
                'seq_features': dataset.seq_features,
                'use_enhanced': use_enhanced  # Track which model architecture
            }, scalers_path)
            print(f"Scalers saved to {scalers_path}")
