"""
================================================================================
SIGNATURE-INFORMED TRANSFORMER (SIT) QUANT SYSTEM 2026
================================================================================

Architecture (based on ar5iv.labs.arxiv.org SIT paper):
1. COMPREHENSIVE FEATURE ENGINEERING (94+ features)
2. NLP SENTIMENT PROXY
3. PATH SIGNATURE COMPUTATION - Rough path theory for geometric features
4. LEAD-LAG SIGNATURE MATRIX - Cross-asset temporal relationships
5. SIGNATURE-AUGMENTED ATTENTION - Injects lead-lag bias into attention
6. CVaR LOSS - Direct risk optimization (not predict-then-optimize)
7. REGIME DETECTION (HMM)
8. PER-TICKER BUY/SELL SIGNALS with relative thresholds

Key SIT Innovations:
- Embeds mathematical finance structure (path signatures) into transformer
- End-to-end portfolio objective training (CVaR minimization)
- Lead-lag inductive bias in cross-asset attention

Training: 2015 → today
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

import os
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy import stats as scipy_stats

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Market Data
import yfinance as yf

# Regime Detection
from hmmlearn.hmm import GaussianHMM

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================
class MarketRegime(Enum):
    """Market regime states."""
    CRISIS = 0
    BEAR = 1
    SIDEWAYS = 2
    BULL = 3


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


# =============================================================================
# COMPREHENSIVE FEATURE LIST
# =============================================================================
TIER1_FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_21d',
    'log_return_1d', 'log_return_5d',
    'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
    'momentum_5d', 'momentum_10d', 'momentum_20d',
    'roc_5d', 'roc_10d', 'roc_20d',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_5', 'ema_10', 'ema_20', 'ema_50',
    'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
    'rsi_7', 'rsi_14', 'rsi_21',
    'macd', 'macd_signal', 'macd_hist', 'macd_cross',
    'bb_upper', 'bb_lower', 'bb_width', 'bb_pct',
    'volume_sma_20', 'volume_ratio', 'volume_change',
    'obv', 'obv_slope',
    'trend_sma10_50', 'trend_sma50_200',
    'atr_14', 'natr',
]

TIER2_FEATURES = [
    'high_low_ratio', 'close_open_ratio', 'range_pct', 'body_size',
    'upper_shadow', 'lower_shadow', 'overnight_gap',
    'plus_di', 'minus_di', 'adx_14',
    'stoch_k', 'stoch_d',
    'cci_20', 'williams_r', 'mfi_14',
    'cmf_20', 'vpt', 'ad_line',
    'volatility_ratio_5_20',
    'golden_cross',
    'distance_to_sma_50', 'distance_to_sma_200',
    'skewness_20', 'kurtosis_20',
    'day_of_week', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'is_monday', 'is_friday', 'is_month_end',
    'pct_from_52w_high', 'pct_from_52w_low',
]

SENTIMENT_FEATURES = [
    'sentiment_score', 'sentiment_momentum', 'sentiment_volatility',
    'news_volume', 'social_buzz', 'sentiment_divergence',
]

FEATURES = ['price'] + TIER1_FEATURES + TIER2_FEATURES + SENTIMENT_FEATURES


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Configuration for the SIT Quant System."""
    
    # Portfolio
    tickers: List[str] = field(default_factory=lambda: ['AAPL', 'GOOGL', 'MSFT'])
    
    # Data
    start_date: str = "2015-01-01"
    end_date: str = field(default_factory=lambda: datetime.today().strftime('%Y-%m-%d'))
    
    # Sequence settings
    time_steps: int = 20
    prediction_horizon: int = 5
    
    # Transformer Architecture
    d_model: int = 64
    ff_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout_rate: float = 0.2
    
    # SIT-Specific Parameters
    signature_depth: int = 3          # Depth of path signature truncation
    signature_dim: int = 16           # Dimension of signature embedding
    leadlag_gamma: float = 0.1        # Weight for lead-lag bias in attention
    use_cvar_loss: bool = True        # Use CVaR instead of MSE
    cvar_alpha: float = 0.05          # CVaR confidence level (5% = 95% CVaR)
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    
    # Data splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    
    # Trading
    initial_capital: float = 100000
    transaction_cost_bps: float = 5
    
    # Signal thresholds (percentile-based)
    strong_buy_percentile: float = 80
    buy_percentile: float = 60
    sell_percentile: float = 40
    strong_sell_percentile: float = 20


# =============================================================================
# PATH SIGNATURE COMPUTATION
# =============================================================================
class PathSignatureComputer:
    """
    Compute path signatures for financial time series.
    
    Path signatures are a mathematical tool from rough path theory that capture
    the geometric/sequential structure of a path. They provide:
    - Order-agnostic representation of path dynamics
    - Natural encoding of momentum, mean-reversion, oscillations
    - Invariance to time reparametrization
    
    For a path X: [0,T] -> R^d, the signature is the collection of iterated
    integrals: S(X)^{i1,...,ik} = ∫ dX^i1 ∘ ... ∘ dX^ik
    """
    
    def __init__(self, depth: int = 3, channels: int = 3):
        self.depth = depth
        self.channels = channels
        self._sig_dim = self._compute_sig_dim(channels, depth)
    
    def _compute_sig_dim(self, d: int, m: int) -> int:
        """Compute dimension of truncated signature."""
        # sig_dim = sum_{k=1}^{m} d^k = d(d^m - 1)/(d - 1)
        if d == 1:
            return m
        return int(d * (d**m - 1) / (d - 1))
    
    def compute_signature(self, path: np.ndarray) -> np.ndarray:
        """
        Compute truncated path signature.
        
        Args:
            path: (time_steps, channels) array
            
        Returns:
            signature: (sig_dim,) array
        """
        T, d = path.shape
        
        # Normalize path for numerical stability
        path = (path - path.mean(axis=0)) / (path.std(axis=0) + 1e-8)
        
        # Compute increments
        increments = np.diff(path, axis=0)  # (T-1, d)
        
        # Initialize signature terms
        sig_terms = []
        
        # Level 1: first-order iterated integrals = cumulative sums
        level1 = increments.sum(axis=0)  # (d,)
        sig_terms.extend(level1.tolist())
        
        if self.depth >= 2:
            # Level 2: second-order iterated integrals
            level2 = np.zeros((d, d))
            cumsum = np.zeros(d)
            for t in range(T - 1):
                level2 += np.outer(cumsum, increments[t])
                cumsum += increments[t]
            sig_terms.extend(level2.flatten().tolist())
        
        if self.depth >= 3:
            # Level 3: third-order (simplified approximation)
            level3 = np.zeros((d, d, d))
            cumsum1 = np.zeros(d)
            cumsum2 = np.zeros((d, d))
            for t in range(T - 1):
                inc = increments[t]
                for i in range(d):
                    for j in range(d):
                        level3[i, j] += cumsum2[i, j] * inc
                cumsum2 += np.outer(cumsum1, inc)
                cumsum1 += inc
            sig_terms.extend(level3.flatten()[:min(d**3, 27)].tolist())
        
        return np.array(sig_terms, dtype=np.float32)
    
    def compute_lead_lag_signature(self, path1: np.ndarray, path2: np.ndarray) -> np.ndarray:
        """
        Compute lead-lag signature between two paths.
        
        The lead-lag transform augments two paths into a single path that
        captures their relative dynamics and cross-correlations over time.
        
        This is key for detecting which asset leads/lags another.
        
        Args:
            path1, path2: (time_steps,) arrays
            
        Returns:
            lead_lag_features: (4,) array [lead_12, lag_12, cross_12, cross_21]
        """
        T = len(path1)
        
        # Normalize
        path1 = (path1 - path1.mean()) / (path1.std() + 1e-8)
        path2 = (path2 - path2.mean()) / (path2.std() + 1e-8)
        
        # Compute increments
        inc1 = np.diff(path1)
        inc2 = np.diff(path2)
        
        # Lead-lag features using cross-covariance at different lags
        # Lead: path1 at t predicts path2 at t+1
        lead_12 = np.corrcoef(path1[:-1], path2[1:])[0, 1] if T > 1 else 0
        
        # Lag: path2 at t predicts path1 at t+1
        lag_12 = np.corrcoef(path2[:-1], path1[1:])[0, 1] if T > 1 else 0
        
        # Cross-signature terms (second order iterated integrals)
        cross_12 = 0
        cross_21 = 0
        cumsum1 = 0
        cumsum2 = 0
        for t in range(T - 1):
            cross_12 += cumsum1 * inc2[t]
            cross_21 += cumsum2 * inc1[t]
            cumsum1 += inc1[t]
            cumsum2 += inc2[t]
        
        # Normalize
        norm = T + 1e-8
        
        return np.array([
            0 if np.isnan(lead_12) else lead_12,
            0 if np.isnan(lag_12) else lag_12,
            cross_12 / norm,
            cross_21 / norm
        ], dtype=np.float32)


# =============================================================================
# LEAD-LAG BIAS MATRIX
# =============================================================================
class LeadLagBiasComputer:
    """
    Compute lead-lag bias matrix for cross-asset attention.
    
    This implements the signature-informed attention bias from the SIT paper:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d) + γ * β) V
    
    where β is the lead-lag bias matrix derived from path signatures.
    """
    
    def __init__(self, n_assets: int, gamma: float = 0.1):
        self.n_assets = n_assets
        self.gamma = gamma
        self.signature_computer = PathSignatureComputer(depth=2, channels=1)
    
    def compute_bias_matrix(self, price_paths: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the lead-lag bias matrix β for all asset pairs.
        
        Args:
            price_paths: Dict[ticker -> (time_steps,) array of prices]
            
        Returns:
            bias_matrix: (n_assets, n_assets) array
        """
        tickers = list(price_paths.keys())
        n = len(tickers)
        bias = np.zeros((n, n), dtype=np.float32)
        
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i == j:
                    continue
                    
                # Compute lead-lag signature
                ll_sig = self.signature_computer.compute_lead_lag_signature(
                    price_paths[t1], price_paths[t2]
                )
                
                # Asymmetric bias: positive if t1 leads t2, negative if t1 lags t2
                bias[i, j] = ll_sig[0] - ll_sig[1]  # lead - lag
                
                # Add cross-signature contribution
                bias[i, j] += 0.5 * (ll_sig[2] - ll_sig[3])
        
        # Scale by gamma
        return self.gamma * bias


# =============================================================================
# CVaR (CONDITIONAL VALUE-AT-RISK) LOSS
# =============================================================================
class CVaRLoss(keras.losses.Loss):
    """
    Conditional Value-at-Risk (Expected Shortfall) loss.
    
    CVaR_α = E[X | X ≤ VaR_α]
    
    This aligns the model training with a risk management objective:
    minimize the expected loss in the worst α% of scenarios.
    
    For portfolio optimization, this is superior to MSE because:
    1. It focuses on tail risk (worst outcomes)
    2. It's coherent (subadditive, positive homogeneous)
    3. It directly optimizes what we care about in trading
    """
    
    def __init__(self, alpha: float = 0.05, name: str = 'cvar_loss'):
        """
        Args:
            alpha: Confidence level (e.g., 0.05 for 95% CVaR)
        """
        super().__init__(name=name)
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        """
        Compute CVaR loss.
        
        Args:
            y_true: (batch, n_assets) actual returns
            y_pred: (batch, n_assets) predicted returns
            
        Returns:
            CVaR loss (scalar)
        """
        # Portfolio loss = negative of prediction-weighted returns
        # Simple approach: use predictions as confidence weights
        pred_weights = tf.nn.softmax(y_pred, axis=-1)  # (batch, n_assets)
        portfolio_return = tf.reduce_sum(pred_weights * y_true, axis=-1)  # (batch,)
        
        # Convert to loss (negative return)
        losses = -portfolio_return
        
        # Compute CVaR: mean of losses above VaR threshold
        batch_size = tf.shape(losses)[0]
        k = tf.maximum(tf.cast(tf.cast(batch_size, tf.float32) * self.alpha, tf.int32), 1)
        
        # Sort losses in descending order (worst first)
        sorted_losses, _ = tf.nn.top_k(losses, k=k, sorted=True)
        
        # CVaR = mean of top-alpha losses
        cvar = tf.reduce_mean(sorted_losses)
        
        # Add prediction error component for gradient stability
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Weighted combination
        return 0.7 * cvar + 0.3 * mse


# =============================================================================
# SIGNATURE-AUGMENTED ATTENTION LAYER
# =============================================================================
class SignatureAugmentedAttention(layers.Layer):
    """
    Multi-head attention with signature-based lead-lag bias injection.
    
    Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + γ * β) V
    
    where β is the pre-computed lead-lag bias matrix.
    """
    
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self.scale = 1.0 / np.sqrt(key_dim)
        
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        
        # Query, Key, Value projections
        self.Wq = self.add_weight(
            name='Wq',
            shape=(self.d_model, self.num_heads * self.key_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Wk = self.add_weight(
            name='Wk',
            shape=(self.d_model, self.num_heads * self.key_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Wv = self.add_weight(
            name='Wv',
            shape=(self.d_model, self.num_heads * self.key_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Wo = self.add_weight(
            name='Wo',
            shape=(self.num_heads * self.key_dim, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Learnable lead-lag bias transformation
        self.bias_transform = self.add_weight(
            name='bias_transform',
            shape=(self.num_heads,),
            initializer='ones',
            trainable=True
        )
        
        self.dropout = layers.Dropout(self.dropout_rate)
        
    def call(self, inputs, lead_lag_bias=None, training=None):
        """
        Args:
            inputs: (batch, n_assets, d_model)
            lead_lag_bias: (n_assets, n_assets) static bias matrix
            training: bool
            
        Returns:
            output: (batch, n_assets, d_model)
        """
        batch_size = tf.shape(inputs)[0]
        n_assets = tf.shape(inputs)[1]
        
        # Project to Q, K, V
        Q = tf.matmul(inputs, self.Wq)  # (batch, n_assets, heads * key_dim)
        K = tf.matmul(inputs, self.Wk)
        V = tf.matmul(inputs, self.Wv)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, n_assets, self.num_heads, self.key_dim))
        Q = tf.transpose(Q, (0, 2, 1, 3))  # (batch, heads, n_assets, key_dim)
        K = tf.reshape(K, (batch_size, n_assets, self.num_heads, self.key_dim))
        K = tf.transpose(K, (0, 2, 1, 3))
        V = tf.reshape(V, (batch_size, n_assets, self.num_heads, self.key_dim))
        V = tf.transpose(V, (0, 2, 1, 3))
        
        # Compute attention scores
        scores = tf.matmul(Q, K, transpose_b=True) * self.scale  # (batch, heads, n, n)
        
        # Inject lead-lag bias if provided
        if lead_lag_bias is not None:
            # Expand bias for batch and heads
            # lead_lag_bias: (n_assets, n_assets)
            # Need: (1, heads, n_assets, n_assets)
            bias_expanded = tf.expand_dims(tf.expand_dims(lead_lag_bias, 0), 0)
            bias_expanded = tf.tile(bias_expanded, [1, self.num_heads, 1, 1])
            
            # Apply learnable per-head scaling
            head_scale = tf.reshape(self.bias_transform, (1, self.num_heads, 1, 1))
            bias_scaled = bias_expanded * head_scale
            
            scores = scores + bias_scaled
        
        # Softmax
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        
        # Apply to values
        output = tf.matmul(attn_weights, V)  # (batch, heads, n_assets, key_dim)
        
        # Reshape back
        output = tf.transpose(output, (0, 2, 1, 3))  # (batch, n_assets, heads, key_dim)
        output = tf.reshape(output, (batch_size, n_assets, self.num_heads * self.key_dim))
        
        # Final projection
        output = tf.matmul(output, self.Wo)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate
        })
        return config


# =============================================================================
# SIGNATURE-INFORMED TRANSFORMER MODEL
# =============================================================================
class SITModel(keras.Model):
    """
    Signature-Informed Transformer for multi-asset return prediction.
    
    Architecture:
    1. Per-asset temporal transformer (shared weights)
    2. Path signature feature extraction
    3. Signature-augmented cross-asset attention with lead-lag bias
    4. Output head for return predictions
    """
    
    def __init__(
        self,
        n_assets: int,
        n_features: int,
        time_steps: int,
        config: Config,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_assets = n_assets
        self.n_features = n_features
        self.time_steps = time_steps
        self.config = config
        
        # Signature computer
        self.sig_computer = PathSignatureComputer(
            depth=config.signature_depth,
            channels=min(n_features, 3)  # Use top 3 features for signature
        )
        
        # Feature projection
        self.input_projection = layers.Dense(config.d_model, activation=None)
        
        # Positional encoding (sinusoidal)
        self.pos_encoding = self._create_positional_encoding(time_steps, config.d_model)
        
        # Temporal transformer blocks
        self.temporal_attn_layers = []
        self.temporal_ff_layers = []
        self.temporal_ln1_layers = []
        self.temporal_ln2_layers = []
        
        for _ in range(config.num_layers):
            self.temporal_attn_layers.append(
                layers.MultiHeadAttention(
                    num_heads=config.num_heads,
                    key_dim=config.d_model // config.num_heads,
                    dropout=config.dropout_rate
                )
            )
            self.temporal_ff_layers.append(keras.Sequential([
                layers.Dense(config.ff_dim, activation='relu'),
                layers.Dense(config.d_model),
                layers.Dropout(config.dropout_rate)
            ]))
            self.temporal_ln1_layers.append(layers.LayerNormalization())
            self.temporal_ln2_layers.append(layers.LayerNormalization())
        
        # Signature embedding
        self.sig_embedding = layers.Dense(config.signature_dim, activation='relu')
        
        # Cross-asset attention with signature bias
        self.cross_attn = SignatureAugmentedAttention(
            num_heads=config.num_heads,
            key_dim=config.d_model // config.num_heads,
            dropout=config.dropout_rate
        )
        self.cross_ln = layers.LayerNormalization()
        
        # Output head
        self.output_ff = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(config.dropout_rate),
            layers.Dense(1)
        ])
        
        # Lead-lag bias (will be set during training)
        self.lead_lag_bias = None
        
    def _create_positional_encoding(self, time_steps: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        position = np.arange(time_steps)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((time_steps, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pe, dtype=tf.float32)
    
    def set_lead_lag_bias(self, bias: np.ndarray):
        """Set the lead-lag bias matrix computed from training data."""
        self.lead_lag_bias = tf.constant(bias, dtype=tf.float32)
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: (batch, n_assets, time_steps, n_features)
            
        Returns:
            predictions: (batch, n_assets)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Process each asset with shared temporal transformer
        asset_outputs = []
        
        for asset_idx in range(self.n_assets):
            # Extract single asset: (batch, time_steps, n_features)
            asset_input = inputs[:, asset_idx, :, :]
            
            # Project to d_model
            x = self.input_projection(asset_input)  # (batch, time_steps, d_model)
            
            # Add positional encoding
            x = x + self.pos_encoding
            
            # Temporal transformer blocks
            for i in range(self.config.num_layers):
                # Self-attention
                attn_out = self.temporal_attn_layers[i](x, x, training=training)
                x = self.temporal_ln1_layers[i](x + attn_out)
                
                # Feed-forward
                ff_out = self.temporal_ff_layers[i](x, training=training)
                x = self.temporal_ln2_layers[i](x + ff_out)
            
            # Take last time step: (batch, d_model)
            asset_output = x[:, -1, :]
            asset_outputs.append(asset_output)
        
        # Stack asset outputs: (batch, n_assets, d_model)
        combined = tf.stack(asset_outputs, axis=1)
        
        # Signature-augmented cross-asset attention
        cross_out = self.cross_attn(
            combined,
            lead_lag_bias=self.lead_lag_bias,
            training=training
        )
        combined = self.cross_ln(combined + cross_out)
        
        # Output predictions: (batch, n_assets, 1) -> (batch, n_assets)
        outputs = self.output_ff(combined)
        outputs = tf.squeeze(outputs, axis=-1)
        
        return outputs


def build_sit_model(
    n_assets: int,
    n_features: int,
    time_steps: int,
    config: Config
) -> SITModel:
    """Build and compile the SIT model."""
    
    model = SITModel(
        n_assets=n_assets,
        n_features=n_features,
        time_steps=time_steps,
        config=config
    )
    
    # Build model
    dummy_input = tf.zeros((1, n_assets, time_steps, n_features))
    _ = model(dummy_input)
    
    # Choose loss function
    if config.use_cvar_loss:
        loss_fn = CVaRLoss(alpha=config.cvar_alpha)
        logger.info("  Using CVaR loss (α=%.2f) for risk-aligned training", config.cvar_alpha)
    else:
        loss_fn = 'mse'
        logger.info("  Using MSE loss")
    
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=loss_fn,
        metrics=['mae']
    )
    
    return model


# =============================================================================
# DATA PIPELINE (same as base transformer)
# =============================================================================
class DataPipeline:
    """Fetch and process market data with comprehensive features."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def fetch_all_tickers(self) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all tickers."""
        data = {}
        
        for ticker in self.config.tickers:
            logger.info(f"  Fetching {ticker}...")
            try:
                t = yf.Ticker(ticker)
                df = t.history(start=self.config.start_date, end=self.config.end_date)
                df.reset_index(inplace=True)
                
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                df['price'] = df['close']
                
                if len(df) > 0:
                    data[ticker] = df
                    logger.info(f"    {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"    Error fetching {ticker}: {e}")
        
        return data
    
    def engineer_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Engineer comprehensive features for a single ticker."""
        df = df.copy()
        
        # Ensure we have OHLC data
        has_ohlc = all(c in df.columns for c in ['open', 'high', 'low', 'close'])
        
        price = df['price'].values
        
        # =================================================================
        # TIER 1: CORE FEATURES
        # =================================================================
        
        # Returns
        df['return_1d'] = df['price'].pct_change(1)
        df['return_5d'] = df['price'].pct_change(5)
        df['return_10d'] = df['price'].pct_change(10)
        df['return_21d'] = df['price'].pct_change(21)
        df['log_return_1d'] = np.log(df['price'] / df['price'].shift(1))
        df['log_return_5d'] = np.log(df['price'] / df['price'].shift(5))
        
        # Volatility
        for w in [5, 10, 20, 60]:
            df[f'volatility_{w}d'] = df['return_1d'].rolling(w).std() * np.sqrt(252)
        
        # Momentum
        for w in [5, 10, 20]:
            df[f'momentum_{w}d'] = df['price'] - df['price'].shift(w)
            df[f'roc_{w}d'] = df['price'].pct_change(w) * 100
        
        # Moving Averages
        for w in [5, 10, 20, 50, 200]:
            df[f'sma_{w}'] = df['price'].rolling(w).mean()
        for w in [5, 10, 20, 50]:
            df[f'ema_{w}'] = df['price'].ewm(span=w, adjust=False).mean()
        
        # Price to MA ratios
        df['price_to_sma_20'] = df['price'] / df['sma_20']
        df['price_to_sma_50'] = df['price'] / df['sma_50']
        df['price_to_sma_200'] = df['price'] / df['sma_200']
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['price'].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['price'].ewm(span=12, adjust=False).mean()
        ema26 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_ma = df['price'].rolling(bb_period).mean()
        bb_std_val = df['price'].rolling(bb_period).std()
        df['bb_upper'] = bb_ma + bb_std * bb_std_val
        df['bb_lower'] = bb_ma - bb_std * bb_std_val
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
        df['bb_pct'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1)
            df['volume_change'] = df['volume'].pct_change()
            
            # OBV
            obv = (np.sign(df['return_1d'].fillna(0)) * df['volume'].fillna(0)).cumsum()
            df['obv'] = obv
            df['obv_slope'] = obv.diff(5) / 5
        else:
            for f in ['volume_sma_20', 'volume_ratio', 'volume_change', 'obv', 'obv_slope']:
                df[f] = 0
        
        # Trend
        df['trend_sma10_50'] = (df['sma_10'] > df['sma_50']).astype(int)
        df['trend_sma50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # ATR
        if has_ohlc:
            high_low = df['high'] - df['low']
            high_cp = abs(df['high'] - df['close'].shift())
            low_cp = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()
            df['natr'] = df['atr_14'] / df['close'] * 100
        else:
            df['atr_14'] = df['price'].rolling(14).std() * 1.5
            df['natr'] = df['atr_14'] / df['price'] * 100
        
        # =================================================================
        # TIER 2: ADVANCED FEATURES
        # =================================================================
        
        if has_ohlc:
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            df['range_pct'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
            df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
            df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
            df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1
            
            # ADX
            df['plus_dm'] = np.where(
                (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                np.maximum(df['high'] - df['high'].shift(1), 0), 0
            )
            df['minus_dm'] = np.where(
                (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                np.maximum(df['low'].shift(1) - df['low'], 0), 0
            )
            df['plus_di'] = 100 * (pd.Series(df['plus_dm']).ewm(span=14).mean() / df['atr_14'])
            df['minus_di'] = 100 * (pd.Series(df['minus_dm']).ewm(span=14).mean() / df['atr_14'])
            df['adx_14'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)).ewm(span=14).mean()
            
            # Stochastic
            low14 = df['low'].rolling(14).min()
            high14 = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # CCI
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
            
            # Williams %R
            df['williams_r'] = -100 * (high14 - df['close']) / (high14 - low14 + 1e-10)
            
            # MFI
            if 'volume' in df.columns and df['volume'].sum() > 0:
                mf = tp * df['volume']
                pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
                neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
                df['mfi_14'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
                
                # CMF
                mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10) * df['volume']
                df['cmf_20'] = mfv.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1)
                
                # VPT
                df['vpt'] = (df['return_1d'] * df['volume']).cumsum()
                
                # A/D Line
                df['ad_line'] = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10) * df['volume']).cumsum()
            else:
                for f in ['mfi_14', 'cmf_20', 'vpt', 'ad_line']:
                    df[f] = 0
        else:
            for f in ['high_low_ratio', 'close_open_ratio', 'range_pct', 'body_size',
                      'upper_shadow', 'lower_shadow', 'overnight_gap', 'plus_di',
                      'minus_di', 'adx_14', 'stoch_k', 'stoch_d', 'cci_20',
                      'williams_r', 'mfi_14', 'cmf_20', 'vpt', 'ad_line']:
                df[f] = 0
        
        # Volatility ratio
        df['volatility_ratio_5_20'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-10)
        
        # Golden Cross
        df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & 
                              (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
        
        # Distance to MA
        df['distance_to_sma_50'] = (df['price'] - df['sma_50']) / df['sma_50']
        df['distance_to_sma_200'] = (df['price'] - df['sma_200']) / df['sma_200']
        
        # Statistical moments
        df['skewness_20'] = df['return_1d'].rolling(20).skew()
        df['kurtosis_20'] = df['return_1d'].rolling(20).kurt()
        
        # Calendar features
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        month = pd.to_datetime(df['date']).dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_end'] = pd.to_datetime(df['date']).dt.is_month_end.astype(int)
        
        # 52-week position
        df['pct_from_52w_high'] = df['price'] / df['price'].rolling(252).max() - 1
        df['pct_from_52w_low'] = df['price'] / df['price'].rolling(252).min() - 1
        
        # =================================================================
        # SENTIMENT PROXY (based on price/volume dynamics)
        # =================================================================
        df['sentiment_score'] = (
            0.3 * df['rsi_14'].fillna(50) / 100 +
            0.3 * df['bb_pct'].fillna(0.5) +
            0.2 * np.tanh(df['volume_ratio'].fillna(1) - 1) +
            0.2 * np.tanh(df['return_5d'].fillna(0) * 10)
        )
        df['sentiment_momentum'] = df['sentiment_score'].diff(5)
        df['sentiment_volatility'] = df['sentiment_score'].rolling(10).std()
        df['news_volume'] = df['volume_ratio'].fillna(1)
        df['social_buzz'] = df['volume_change'].fillna(0).rolling(5).mean()
        df['sentiment_divergence'] = df['sentiment_score'] - df['sentiment_score'].rolling(20).mean()
        
        # =================================================================
        # TARGET: FUTURE RETURN
        # =================================================================
        df['future_return'] = df['price'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
        
        return df
    
    def align_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all tickers to common dates."""
        date_sets = [set(df['date'].dt.strftime('%Y-%m-%d')) for df in data.values()]
        common_dates = set.intersection(*date_sets)
        
        aligned = {}
        for ticker, df in data.items():
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
            aligned[ticker] = df[df['date_str'].isin(common_dates)].drop('date_str', axis=1).reset_index(drop=True)
        
        logger.info(f"  Aligned to {len(common_dates)} common dates")
        return aligned


# =============================================================================
# REGIME DETECTOR
# =============================================================================
class RegimeDetector:
    """HMM-based market regime detection."""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.fitted = False
    
    def fit(self, returns: np.ndarray):
        """Fit HMM on return series."""
        X = returns.reshape(-1, 1).copy()
        X = np.where(np.isnan(X) | np.isinf(X), 0, X)
        try:
            self.model.fit(X)
            self.fitted = True
        except Exception as e:
            logger.warning(f"HMM fit failed: {e}")
    
    def predict(self, returns: np.ndarray) -> List[MarketRegime]:
        """Predict regimes."""
        if not self.fitted:
            return [MarketRegime.SIDEWAYS] * len(returns)
        
        X = returns.reshape(-1, 1).copy()
        X = np.where(np.isnan(X) | np.isinf(X), 0, X)
        
        try:
            states = self.model.predict(X)
            means = [self.model.means_[i][0] for i in range(self.n_regimes)]
            sorted_states = np.argsort(means)
            
            regime_map = {}
            if self.n_regimes >= 4:
                regime_map[sorted_states[0]] = MarketRegime.CRISIS
                regime_map[sorted_states[1]] = MarketRegime.BEAR
                regime_map[sorted_states[2]] = MarketRegime.SIDEWAYS
                regime_map[sorted_states[3]] = MarketRegime.BULL
            else:
                for i, s in enumerate(sorted_states):
                    regime_map[s] = list(MarketRegime)[min(i, 3)]
            
            return [regime_map.get(s, MarketRegime.SIDEWAYS) for s in states]
        except Exception:
            return [MarketRegime.SIDEWAYS] * len(returns)


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================
class SignalGenerator:
    """Generate BUY/SELL signals using percentile-based thresholds."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_signals(
        self,
        predictions: Dict[str, np.ndarray],
        dates: pd.Series,
        prices: Dict[str, pd.Series]
    ) -> Dict[str, pd.DataFrame]:
        """Generate signals using percentile-based approach."""
        signals = {}
        
        for ticker, preds in predictions.items():
            n = len(preds)
            
            # Rolling percentile rank
            window = 60
            pred_series = pd.Series(preds)
            rolling_rank = pred_series.rolling(window=window, min_periods=20).apply(
                lambda x: scipy_stats.percentileofscore(x, x.iloc[-1]), raw=False
            ).fillna(50)
            
            signal_types = []
            signal_names = []
            
            for i in range(n):
                rank = rolling_rank.iloc[i]
                
                if rank >= self.config.strong_buy_percentile:
                    signal_types.append(SignalType.STRONG_BUY.value)
                    signal_names.append('STRONG_BUY')
                elif rank >= self.config.buy_percentile:
                    signal_types.append(SignalType.BUY.value)
                    signal_names.append('BUY')
                elif rank <= self.config.strong_sell_percentile:
                    signal_types.append(SignalType.STRONG_SELL.value)
                    signal_names.append('STRONG_SELL')
                elif rank <= self.config.sell_percentile:
                    signal_types.append(SignalType.SELL.value)
                    signal_names.append('SELL')
                else:
                    signal_types.append(SignalType.HOLD.value)
                    signal_names.append('HOLD')
            
            signals[ticker] = pd.DataFrame({
                'date': dates.values[:n],
                'price': prices[ticker].values[:n],
                'prediction': preds,
                'percentile_rank': rolling_rank.values[:n],
                'signal': signal_types,
                'signal_name': signal_names,
                'confidence': np.abs(rolling_rank.values[:n] - 50) / 50
            })
        
        return signals


# =============================================================================
# VISUALIZER
# =============================================================================
class Visualizer:
    """Generate comprehensive visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.regime_colors = {
            MarketRegime.CRISIS: '#FF6B6B',
            MarketRegime.BEAR: '#FFB347',
            MarketRegime.SIDEWAYS: '#87CEEB',
            MarketRegime.BULL: '#98FB98'
        }
    
    def plot_per_ticker_signals(
        self,
        ticker_signals: Dict[str, pd.DataFrame],
        regimes: List[MarketRegime]
    ):
        """Plot buy/sell signals for each ticker."""
        n_tickers = len(ticker_signals)
        fig, axes = plt.subplots(n_tickers, 1, figsize=(16, 5 * n_tickers))
        
        if n_tickers == 1:
            axes = [axes]
        
        for idx, (ticker, signals) in enumerate(ticker_signals.items()):
            ax = axes[idx]
            
            dates = pd.to_datetime(signals['date'])
            prices = signals['price']
            
            # Plot price
            ax.plot(dates, prices, 'b-', linewidth=1.5, alpha=0.8, label='Price')
            
            # Add regime background
            if len(regimes) >= len(dates):
                for i in range(len(dates) - 1):
                    regime = regimes[i] if i < len(regimes) else MarketRegime.SIDEWAYS
                    ax.axvspan(dates.iloc[i], dates.iloc[i + 1],
                              alpha=0.15, color=self.regime_colors.get(regime, 'gray'))
            
            # Plot signals
            buy_mask = signals['signal_name'].isin(['BUY', 'STRONG_BUY'])
            sell_mask = signals['signal_name'].isin(['SELL', 'STRONG_SELL'])
            
            if buy_mask.any():
                buy_dates = dates[buy_mask]
                buy_prices = prices[buy_mask]
                buy_conf = signals.loc[buy_mask, 'confidence']
                sizes = 50 + buy_conf * 150
                ax.scatter(buy_dates, buy_prices, marker='^', c='green',
                          s=sizes, alpha=0.8, label=f'BUY ({buy_mask.sum()})', zorder=5)
            
            if sell_mask.any():
                sell_dates = dates[sell_mask]
                sell_prices = prices[sell_mask]
                sell_conf = signals.loc[sell_mask, 'confidence']
                sizes = 50 + sell_conf * 150
                ax.scatter(sell_dates, sell_prices, marker='v', c='red',
                          s=sizes, alpha=0.8, label=f'SELL ({sell_mask.sum()})', zorder=5)
            
            ax.set_title(f'{ticker} - SIT Trading Signals', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_signature_analysis(
        self,
        lead_lag_bias: np.ndarray,
        tickers: List[str]
    ):
        """Visualize the lead-lag signature matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(lead_lag_bias, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
        
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers)
        ax.set_yticklabels(tickers)
        
        # Add values
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                text = ax.text(j, i, f'{lead_lag_bias[i, j]:.3f}',
                              ha='center', va='center', fontsize=10)
        
        ax.set_title('Lead-Lag Signature Bias Matrix\n(+ve: row leads column, -ve: row lags column)',
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Lead-Lag Bias (γβ)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_portfolio_performance(
        self,
        signals: pd.DataFrame,
        regimes: List[MarketRegime]
    ):
        """Plot portfolio performance."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Panel 1: Portfolio value
        ax1 = axes[0]
        dates = pd.to_datetime(signals['date'])
        
        ax1.plot(dates, signals['portfolio_value'], 'b-', linewidth=2, label='SIT Strategy')
        ax1.plot(dates, signals['bh_value'], 'gray', linewidth=1.5, alpha=0.7, label='Buy & Hold')
        
        ax1.set_title('SIT Portfolio Performance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Cumulative returns
        ax2 = axes[1]
        strategy_returns = signals['portfolio_value'].pct_change().fillna(0)
        cumulative_returns = (1 + strategy_returns).cumprod() - 1
        
        ax2.fill_between(dates, 0, cumulative_returns * 100, 
                        where=cumulative_returns >= 0, color='green', alpha=0.3)
        ax2.fill_between(dates, 0, cumulative_returns * 100,
                        where=cumulative_returns < 0, color='red', alpha=0.3)
        ax2.plot(dates, cumulative_returns * 100, 'k-', linewidth=1.5)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN SYSTEM
# =============================================================================
class SITQuantSystem:
    """Signature-Informed Transformer Quant System."""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = DataPipeline(config)
        self.regime_detector = RegimeDetector()
        self.signal_generator = SignalGenerator(config)
        self.visualizer = Visualizer(config)
        self.sig_computer = PathSignatureComputer(depth=config.signature_depth)
        self.leadlag_computer = LeadLagBiasComputer(
            n_assets=len(config.tickers),
            gamma=config.leadlag_gamma
        )
        
        self.model: Optional[SITModel] = None
        self.scalers: Dict[str, StandardScaler] = {}
        self.selected_features: List[str] = []
        self.ticker_signals: Dict[str, pd.DataFrame] = {}
        self.lead_lag_bias: Optional[np.ndarray] = None
    
    def run(self) -> Dict[str, Any]:
        """Run the full SIT pipeline."""
        
        print("=" * 80)
        print("SIGNATURE-INFORMED TRANSFORMER (SIT) QUANT SYSTEM 2026")
        print("=" * 80)
        print("\nKey SIT Innovations:")
        print("  • Path Signature features from rough path theory")
        print("  • Lead-Lag bias injection in cross-asset attention")
        print(f"  • CVaR loss (α={self.config.cvar_alpha}) for risk-aligned training" if self.config.use_cvar_loss else "  • MSE loss")
        print("  • Percentile-based BUY/SELL signals (guarantees both)")
        print("=" * 80)
        
        # =====================================================================
        # PHASE 1: DATA FETCHING
        # =====================================================================
        logger.info("\n📥 PHASE 1: Fetching Market Data")
        raw_data = self.pipeline.fetch_all_tickers()
        
        if len(raw_data) == 0:
            raise ValueError("No data fetched!")
        
        # =====================================================================
        # PHASE 2: FEATURE ENGINEERING
        # =====================================================================
        logger.info("\n🔧 PHASE 2: Feature Engineering")
        
        processed_data = {}
        for ticker, df in raw_data.items():
            processed_data[ticker] = self.pipeline.engineer_features(df, ticker)
            logger.info(f"  {ticker}: {len(processed_data[ticker])} rows, {len(processed_data[ticker].columns)} features")
        
        aligned_data = self.pipeline.align_data(processed_data)
        
        # =====================================================================
        # PHASE 3: COMPUTE LEAD-LAG SIGNATURE BIAS
        # =====================================================================
        logger.info("\n🔢 PHASE 3: Computing Lead-Lag Signature Bias")
        
        # Use training period prices for signature computation
        first_ticker = self.config.tickers[0]
        n_samples = len(aligned_data[first_ticker])
        train_end = int(n_samples * self.config.train_ratio)
        
        train_prices = {
            ticker: df['price'].iloc[:train_end].values
            for ticker, df in aligned_data.items()
        }
        
        self.lead_lag_bias = self.leadlag_computer.compute_bias_matrix(train_prices)
        
        logger.info("  Lead-Lag Bias Matrix (γβ):")
        for i, t1 in enumerate(self.config.tickers):
            row = [f"{self.lead_lag_bias[i, j]:+.3f}" for j in range(len(self.config.tickers))]
            logger.info(f"    {t1}: [{', '.join(row)}]")
        
        # =====================================================================
        # PHASE 4: REGIME DETECTION
        # =====================================================================
        logger.info("\n🔍 PHASE 4: Regime Detection")
        returns = aligned_data[first_ticker]['return_1d'].dropna().values
        self.regime_detector.fit(returns)
        
        full_returns = aligned_data[first_ticker]['return_1d'].fillna(0).values
        regimes = self.regime_detector.predict(full_returns)
        
        regime_counts = {r: regimes.count(r) for r in MarketRegime}
        for regime, count in regime_counts.items():
            logger.info(f"  {regime.name}: {count} days ({count/len(regimes)*100:.1f}%)")
        
        # =====================================================================
        # PHASE 5: TRAIN/VAL/TEST SPLIT
        # =====================================================================
        logger.info("\n📊 PHASE 5: Data Splitting")
        
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))
        
        train_data = {t: df.iloc[:train_end] for t, df in aligned_data.items()}
        val_data = {t: df.iloc[train_end:val_end] for t, df in aligned_data.items()}
        test_data = {t: df.iloc[val_end:] for t, df in aligned_data.items()}
        
        logger.info(f"  Train: {train_end} samples")
        logger.info(f"  Val: {val_end - train_end} samples")
        logger.info(f"  Test: {n_samples - val_end} samples")
        
        # =====================================================================
        # PHASE 6: FEATURE SCALING
        # =====================================================================
        logger.info("\n📏 PHASE 6: Feature Scaling")
        
        available_cols = set(train_data[first_ticker].columns)
        self.selected_features = [f for f in FEATURES if f in available_cols]
        logger.info(f"  Using {len(self.selected_features)} features")
        
        scaled_train, scaled_val, scaled_test = {}, {}, {}
        
        for ticker in self.config.tickers:
            scaler = StandardScaler()
            scaled_train[ticker] = scaler.fit_transform(train_data[ticker][self.selected_features])
            scaled_val[ticker] = scaler.transform(val_data[ticker][self.selected_features])
            scaled_test[ticker] = scaler.transform(test_data[ticker][self.selected_features])
            self.scalers[ticker] = scaler
        
        # =====================================================================
        # PHASE 7: CREATE SEQUENCES
        # =====================================================================
        logger.info("\n🔢 PHASE 7: Creating Sequences")
        
        def create_sequences(scaled_data, raw_data, time_steps):
            n_assets = len(scaled_data)
            tickers = list(scaled_data.keys())
            min_len = min(len(v) for v in scaled_data.values())
            n_sequences = min_len - time_steps
            
            X = np.zeros((n_sequences, n_assets, time_steps, len(self.selected_features)))
            y = np.zeros((n_sequences, n_assets))
            
            for asset_idx, ticker in enumerate(tickers):
                features = scaled_data[ticker]
                targets = raw_data[ticker]['future_return'].values
                
                for i in range(n_sequences):
                    X[i, asset_idx] = features[i:i + time_steps]
                    y[i, asset_idx] = targets[i + time_steps]
            
            return X, y
        
        X_train, y_train = create_sequences(scaled_train, train_data, self.config.time_steps)
        X_val, y_val = create_sequences(scaled_val, val_data, self.config.time_steps)
        X_test, y_test = create_sequences(scaled_test, test_data, self.config.time_steps)
        
        # Handle NaN targets (NumPy 2.0 compatible)
        y_train = np.where(np.isnan(y_train), 0, y_train)
        y_val = np.where(np.isnan(y_val), 0, y_val)
        y_test = np.where(np.isnan(y_test), 0, y_test)
        
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Val: {X_val.shape}")
        logger.info(f"  Test: {X_test.shape}")
        
        # =====================================================================
        # PHASE 8: BUILD & TRAIN SIT MODEL
        # =====================================================================
        logger.info("\n🤖 PHASE 8: Building & Training SIT Model")
        
        self.model = build_sit_model(
            n_assets=len(self.config.tickers),
            n_features=len(self.selected_features),
            time_steps=self.config.time_steps,
            config=self.config
        )
        
        # Set lead-lag bias
        self.model.set_lead_lag_bias(self.lead_lag_bias)
        
        self.model.summary(print_fn=lambda x: logger.info(f"  {x}"))
        
        callbacks = [
            EarlyStopping(patience=self.config.patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # =====================================================================
        # PHASE 9: GENERATE PREDICTIONS
        # =====================================================================
        logger.info("\n📈 PHASE 9: Generating Predictions")
        
        test_preds = self.model.predict(X_test, verbose=0)
        
        predictions = {}
        for i, ticker in enumerate(self.config.tickers):
            predictions[ticker] = test_preds[:, i]
            logger.info(f"  {ticker}: mean={predictions[ticker].mean():.4f}, std={predictions[ticker].std():.4f}")
        
        # =====================================================================
        # PHASE 10: GENERATE SIGNALS
        # =====================================================================
        logger.info("\n🎯 PHASE 10: Generating Signals")
        
        test_dates = test_data[first_ticker]['date'].iloc[self.config.time_steps:].reset_index(drop=True)
        test_prices = {t: test_data[t]['price'].iloc[self.config.time_steps:].reset_index(drop=True)
                      for t in self.config.tickers}
        
        self.ticker_signals = self.signal_generator.generate_signals(
            predictions, test_dates, test_prices
        )
        
        # Add actual returns
        for ticker in self.config.tickers:
            actual_returns = test_data[ticker]['future_return'].iloc[self.config.time_steps:].values
            n = len(self.ticker_signals[ticker])
            if len(actual_returns) >= n:
                self.ticker_signals[ticker]['actual_return'] = actual_returns[:n]
            else:
                padded = np.zeros(n)
                padded[:len(actual_returns)] = actual_returns
                self.ticker_signals[ticker]['actual_return'] = padded
        
        # =====================================================================
        # PHASE 11: SIMULATE TRADING
        # =====================================================================
        logger.info("\n💰 PHASE 11: Simulating Trading")
        
        portfolio_signals = self._simulate_trading(test_prices, regimes[val_end:])
        
        # =====================================================================
        # PHASE 12: COMPUTE METRICS
        # =====================================================================
        logger.info("\n📊 PHASE 12: Computing Metrics")
        
        results = self._compute_metrics(portfolio_signals)
        
        # =====================================================================
        # PHASE 13: VISUALIZE
        # =====================================================================
        logger.info("\n📊 PHASE 13: Visualization")
        
        self.visualizer.plot_signature_analysis(self.lead_lag_bias, self.config.tickers)
        self.visualizer.plot_per_ticker_signals(self.ticker_signals, regimes[val_end:])
        self.visualizer.plot_portfolio_performance(portfolio_signals, regimes[val_end:])
        
        return results
    
    def _simulate_trading(
        self,
        prices: Dict[str, pd.Series],
        regimes: List[MarketRegime]
    ) -> pd.DataFrame:
        """Simulate trading based on generated signals."""
        
        n_periods = min(len(self.ticker_signals[t]) for t in self.config.tickers)
        
        # Initialize
        cash = self.config.initial_capital * 0.05
        n_assets = len(self.config.tickers)
        shares = {t: 0.0 for t in self.config.tickers}
        
        # Start 95% invested equally
        initial_alloc = self.config.initial_capital * 0.95 / n_assets
        for ticker in self.config.tickers:
            price = prices[ticker].iloc[0]
            shares[ticker] = initial_alloc / price
        
        portfolio_values = []
        bh_values = []
        
        # Buy & hold baseline
        bh_shares = {t: (self.config.initial_capital / n_assets) / prices[t].iloc[0]
                     for t in self.config.tickers}
        
        for t in range(n_periods):
            # Portfolio value
            port_val = cash + sum(
                shares[ticker] * prices[ticker].iloc[t]
                for ticker in self.config.tickers
            )
            portfolio_values.append(port_val)
            
            # B&H value
            bh_val = sum(
                bh_shares[ticker] * prices[ticker].iloc[t]
                for ticker in self.config.tickers
            )
            bh_values.append(bh_val)
            
            # Rebalance periodically based on signals
            if t > 0 and t % 5 == 0:  # Weekly rebalance
                for ticker in self.config.tickers:
                    sig = self.ticker_signals[ticker].iloc[t]
                    signal_val = sig['signal']
                    
                    current_val = shares[ticker] * prices[ticker].iloc[t]
                    target_weight = 1 / n_assets
                    
                    # Adjust based on signal
                    if signal_val >= 1:  # BUY
                        target_weight *= 1.2
                    elif signal_val <= -1:  # SELL
                        target_weight *= 0.8
                    
                    target_val = port_val * target_weight
                    trade_val = target_val - current_val
                    
                    if abs(trade_val) > port_val * 0.02:  # Min 2% change
                        cost = abs(trade_val) * self.config.transaction_cost_bps / 10000
                        if trade_val > 0:  # Buy
                            cash -= (trade_val + cost)
                            shares[ticker] += trade_val / prices[ticker].iloc[t]
                        else:  # Sell
                            cash += (-trade_val - cost)
                            shares[ticker] += trade_val / prices[ticker].iloc[t]
        
        # Build portfolio dataframe
        first_ticker = self.config.tickers[0]
        portfolio_df = pd.DataFrame({
            'date': self.ticker_signals[first_ticker]['date'].values[:n_periods],
            'portfolio_value': portfolio_values,
            'bh_value': bh_values
        })
        
        return portfolio_df
    
    def _compute_metrics(self, portfolio: pd.DataFrame) -> Dict[str, Any]:
        """Compute and display performance metrics."""
        
        results = {'portfolio': portfolio, 'ticker_signals': self.ticker_signals}
        
        print("\n" + "=" * 80)
        print("TRADING SUMMARY - SIGNATURE-INFORMED TRANSFORMER")
        print("=" * 80)
        
        # Portfolio metrics
        start_val = portfolio['portfolio_value'].iloc[0]
        end_val = portfolio['portfolio_value'].iloc[-1]
        strategy_return = (end_val / start_val - 1) * 100
        
        bh_start = portfolio['bh_value'].iloc[0]
        bh_end = portfolio['bh_value'].iloc[-1]
        bh_return = (bh_end / bh_start - 1) * 100
        
        alpha = strategy_return - bh_return
        
        print(f"\n📈 PORTFOLIO PERFORMANCE:")
        print(f"  Strategy Return: {strategy_return:.2f}%")
        print(f"  Buy & Hold Return: {bh_return:.2f}%")
        print(f"  ALPHA: {alpha:+.2f}%")
        
        # Per-ticker metrics
        print(f"\n📊 PER-TICKER METRICS:")
        
        for ticker in self.config.tickers:
            signals = self.ticker_signals[ticker]
            
            preds = signals['prediction'].values
            actuals = signals['actual_return'].values
            
            valid = ~(np.isnan(preds) | np.isnan(actuals))
            if valid.sum() > 10:
                corr = np.corrcoef(preds[valid], actuals[valid])[0, 1]
                direction_acc = ((preds[valid] > 0) == (actuals[valid] > 0)).mean() * 100
            else:
                corr = 0
                direction_acc = 50
            
            buy_count = signals['signal_name'].isin(['BUY', 'STRONG_BUY']).sum()
            sell_count = signals['signal_name'].isin(['SELL', 'STRONG_SELL']).sum()
            
            print(f"\n  {ticker}:")
            print(f"    Direction Accuracy: {direction_acc:.1f}%")
            print(f"    Correlation: {corr:.3f}")
            print(f"    Signals: {buy_count} BUYs, {sell_count} SELLs")
        
        # CVaR metrics
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        if len(daily_returns) > 0:
            var_95 = np.percentile(daily_returns, 5) * 100
            cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
            
            print(f"\n📉 RISK METRICS (what SIT optimizes):")
            print(f"  VaR (5%): {var_95:.2f}% daily")
            print(f"  CVaR (5%): {cvar_95:.2f}% daily")
            
            # Sharpe
            sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
            print(f"  Sharpe Ratio: {sharpe:.2f}")
        
        print("\n" + "=" * 80)
        
        results['strategy_return'] = strategy_return
        results['bh_return'] = bh_return
        results['alpha'] = alpha
        
        return results


# =============================================================================
# MAIN - Runs in both script and notebook environments
# =============================================================================
config = Config(
    tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
    start_date="2015-01-01",
    time_steps=20,
    prediction_horizon=5,
    d_model=64,
    num_heads=4,
    num_layers=2,
    signature_depth=3,
    leadlag_gamma=0.1,
    use_cvar_loss=True,
    cvar_alpha=0.05,
    epochs=100,
    patience=15
)

system = SITQuantSystem(config)
results = system.run()

print("\n✅ SIT Quant System Complete!")

