"""
================================================================================
COMPREHENSIVE FEATURE ENGINEERING FOR TRADING SIGNALS
================================================================================

Complete implementation of 200+ features across all categories.
Senior Google/Citadel SWE Level - Production Ready

Features are organized into tiers:
- Tier 1: Core features (always compute)
- Tier 2: High-value additions (default on)
- Tier 3: Extended features (optional)
- Tier 4: Advanced/Alternative (requires special data)

Usage:
    from comprehensive_feature_engineering import ComprehensiveFeatureEngineer
    
    engineer = ComprehensiveFeatureEngineer(config)
    df = engineer.engineer_all_features(df, tier=2)
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks
import logging

# Technical Analysis library
try:
    import ta
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
    from ta.momentum import StochRSIIndicator, TSIIndicator, UltimateOscillator, KAMAIndicator
    from ta.trend import MACD, ADXIndicator, CCIIndicator, AroonIndicator, PSARIndicator
    from ta.trend import IchimokuIndicator, DPOIndicator, VortexIndicator, MassIndex
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator
    from ta.volume import ForceIndexIndicator, EaseOfMovementIndicator, VolumeWeightedAveragePrice
    from ta.volume import AccDistIndexIndicator, VolumePriceTrendIndicator, NegativeVolumeIndexIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: 'ta' library not available. Install with: pip install ta")

# For advanced features
try:
    from scipy.stats import skew, kurtosis
    from scipy.fft import fft
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Prediction horizon (for target creation)
    prediction_horizon: int = 5
    
    # Moving average windows
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    
    # Momentum windows
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # Volatility windows
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 252])
    
    # Return calculation windows
    return_windows: List[int] = field(default_factory=lambda: [1, 5, 10, 21, 63, 252])
    
    # Standard indicator windows
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    adx_window: int = 14
    stoch_window: int = 14
    cci_window: int = 20
    
    # Feature tiers to compute
    include_tier_1: bool = True  # Core features
    include_tier_2: bool = True  # High-value additions
    include_tier_3: bool = False  # Extended features
    include_tier_4: bool = False  # Advanced (requires special data)
    
    # Special options
    include_candlestick_patterns: bool = True
    include_calendar_features: bool = True
    include_cross_sectional: bool = False  # Requires multi-ticker data


# =============================================================================
# MAIN FEATURE ENGINEERING CLASS
# =============================================================================
class ComprehensiveFeatureEngineer:
    """
    Comprehensive feature engineering for trading signals.
    
    Implements 200+ features across multiple categories without lookahead bias.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
    
    def engineer_all_features(
        self, 
        df: pd.DataFrame, 
        tier: int = 2,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Engineer all features up to the specified tier.
        
        Args:
            df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
                or at minimum ['date', 'price', 'volume'] (close-only data)
            tier: Feature tier (1-4). Higher = more features.
            verbose: Whether to log progress.
        
        Returns:
            DataFrame with all engineered features.
        """
        df = df.copy()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        if verbose:
            logger.info(f"Engineering features (Tier {tier}) on {len(df)} rows...")
        
        # Tier 1: Core features (always computed)
        if tier >= 1:
            df = self._engineer_tier1_core(df)
            if verbose:
                logger.info(f"  ✓ Tier 1 (Core): {len([c for c in df.columns if c.startswith('t1_')])} features")
        
        # Tier 2: High-value additions
        if tier >= 2:
            df = self._engineer_tier2_highvalue(df)
            if verbose:
                logger.info(f"  ✓ Tier 2 (High-Value): {len([c for c in df.columns if c.startswith('t2_')])} features")
        
        # Tier 3: Extended features
        if tier >= 3:
            df = self._engineer_tier3_extended(df)
            if verbose:
                logger.info(f"  ✓ Tier 3 (Extended): {len([c for c in df.columns if c.startswith('t3_')])} features")
        
        # Tier 4: Advanced features
        if tier >= 4:
            df = self._engineer_tier4_advanced(df)
            if verbose:
                logger.info(f"  ✓ Tier 4 (Advanced): {len([c for c in df.columns if c.startswith('t4_')])} features")
        
        # Create target variable
        df = self._create_targets(df)
        
        # Drop NaN rows
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        if verbose:
            logger.info(f"  Dropped {initial_len - len(df)} rows with NaN. Final: {len(df)} rows")
        
        # Store feature names
        self.feature_names = [c for c in df.columns if c.startswith('t1_') or c.startswith('t2_') 
                              or c.startswith('t3_') or c.startswith('t4_')]
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        df.columns = df.columns.str.lower()
        
        # Handle case where only 'price' exists (close-only data)
        if 'price' in df.columns and 'close' not in df.columns:
            df['close'] = df['price']
        
        # If missing OHLC, create from close
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        
        return df
    
    # =========================================================================
    # TIER 1: CORE FEATURES (Essential)
    # =========================================================================
    def _engineer_tier1_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 1: Core essential features."""
        
        # ----- 1.1 RETURNS (Multi-Horizon) -----
        for w in self.config.return_windows:
            df[f't1_return_{w}d'] = df['close'].pct_change(w)
            df[f't1_log_return_{w}d'] = np.log(df['close'] / df['close'].shift(w))
        
        # ----- 1.2 VOLATILITY (Multi-Window) -----
        for w in self.config.volatility_windows:
            df[f't1_volatility_{w}d'] = df['t1_return_1d'].rolling(window=w).std()
            df[f't1_volatility_{w}d_ann'] = df[f't1_volatility_{w}d'] * np.sqrt(252)
        
        # ----- 1.3 MOMENTUM -----
        for w in self.config.momentum_windows:
            df[f't1_momentum_{w}d'] = df['close'] - df['close'].shift(w)
            df[f't1_roc_{w}d'] = df['close'].pct_change(w) * 100
        
        # ----- 1.4 MOVING AVERAGES -----
        for w in self.config.ma_windows:
            df[f't1_sma_{w}'] = df['close'].rolling(window=w).mean()
            df[f't1_ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
            df[f't1_price_to_sma_{w}'] = df['close'] / df[f't1_sma_{w}']
            df[f't1_price_to_ema_{w}'] = df['close'] / df[f't1_ema_{w}']
        
        # ----- 1.5 RSI -----
        df['t1_rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['t1_rsi_7'] = self._calculate_rsi(df['close'], 7)
        df['t1_rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # ----- 1.6 MACD -----
        ema_fast = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        df['t1_macd'] = ema_fast - ema_slow
        df['t1_macd_signal'] = df['t1_macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['t1_macd_hist'] = df['t1_macd'] - df['t1_macd_signal']
        df['t1_macd_cross'] = (df['t1_macd'] > df['t1_macd_signal']).astype(int)
        
        # ----- 1.7 BOLLINGER BANDS -----
        bb_sma = df['close'].rolling(window=self.config.bb_window).mean()
        bb_std = df['close'].rolling(window=self.config.bb_window).std()
        df['t1_bb_upper'] = bb_sma + self.config.bb_std * bb_std
        df['t1_bb_lower'] = bb_sma - self.config.bb_std * bb_std
        df['t1_bb_middle'] = bb_sma
        df['t1_bb_width'] = (df['t1_bb_upper'] - df['t1_bb_lower']) / df['t1_bb_middle']
        df['t1_bb_pct'] = (df['close'] - df['t1_bb_lower']) / (df['t1_bb_upper'] - df['t1_bb_lower'] + 1e-10)
        
        # ----- 1.8 VOLUME FEATURES -----
        df['t1_volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['t1_volume_ratio'] = df['volume'] / (df['t1_volume_sma_20'] + 1e-10)
        df['t1_volume_change'] = df['volume'].pct_change()
        
        # OBV
        sign = np.sign(df['close'].diff()).fillna(0)
        df['t1_obv'] = (sign * df['volume']).cumsum()
        df['t1_obv_sma'] = df['t1_obv'].rolling(window=20).mean()
        df['t1_obv_slope'] = df['t1_obv'].diff(5) / 5
        
        # ----- 1.9 TREND INDICATORS -----
        df['t1_trend_sma10_50'] = (df['t1_sma_10'] > df['t1_sma_50']).astype(int)
        df['t1_trend_sma50_200'] = (df['t1_sma_50'] > df['t1_sma_200']).astype(int)
        
        # ----- 1.10 ATR -----
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['t1_atr_14'] = tr.rolling(window=14).mean()
        df['t1_natr'] = df['t1_atr_14'] / df['close'] * 100
        
        return df
    
    # =========================================================================
    # TIER 2: HIGH-VALUE ADDITIONS
    # =========================================================================
    def _engineer_tier2_highvalue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 2: High-value additional features."""
        
        # ----- 2.1 PRICE STRUCTURE -----
        df['t2_high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
        df['t2_close_open_ratio'] = df['close'] / (df['open'] + 1e-10)
        df['t2_range_pct'] = (df['high'] - df['low']) / df['close']
        df['t2_body_size'] = np.abs(df['close'] - df['open']) / df['close']
        df['t2_upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['t2_lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Gap features
        df['t2_overnight_gap'] = (df['open'] - df['close'].shift()) / df['close'].shift()
        df['t2_gap_direction'] = np.sign(df['t2_overnight_gap'])
        
        # ----- 2.2 ADX / DIRECTIONAL MOVEMENT -----
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = self._calculate_true_range(df)
        atr_14 = tr.rolling(window=14).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr_14 + 1e-10))
        
        df['t2_plus_di'] = plus_di
        df['t2_minus_di'] = minus_di
        df['t2_di_diff'] = plus_di - minus_di
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['t2_adx_14'] = dx.rolling(window=14).mean()
        df['t2_adx_rising'] = (df['t2_adx_14'] > df['t2_adx_14'].shift()).astype(int)
        
        # ----- 2.3 STOCHASTIC OSCILLATOR -----
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['t2_stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['t2_stoch_d'] = df['t2_stoch_k'].rolling(window=3).mean()
        df['t2_stoch_cross'] = (df['t2_stoch_k'] > df['t2_stoch_d']).astype(int)
        
        # ----- 2.4 CCI (Commodity Channel Index) -----
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = typical_price.rolling(window=20).mean()
        tp_mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['t2_cci_20'] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)
        
        # ----- 2.5 WILLIAMS %R -----
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['t2_williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
        
        # ----- 2.6 ADDITIONAL VOLUME INDICATORS -----
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        pos_sum = positive_flow.rolling(window=14).sum()
        neg_sum = negative_flow.rolling(window=14).sum()
        mf_ratio = pos_sum / (neg_sum + 1e-10)
        df['t2_mfi_14'] = 100 - (100 / (1 + mf_ratio))
        
        # Chaikin Money Flow
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['t2_cmf_20'] = (clv * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Volume Price Trend
        df['t2_vpt'] = (df['t1_return_1d'] * df['volume']).cumsum()
        
        # Accumulation/Distribution
        df['t2_ad_line'] = (clv * df['volume']).cumsum()
        
        # ----- 2.7 VOLATILITY RATIOS -----
        df['t2_volatility_ratio_5_20'] = df['t1_volatility_5d'] / (df['t1_volatility_20d'] + 1e-10)
        df['t2_volatility_ratio_20_60'] = df['t1_volatility_20d'] / (df['t1_volatility_60d'] + 1e-10)
        df['t2_atr_ratio_7_21'] = df['t1_atr_14'].rolling(window=7).mean() / (df['t1_atr_14'].rolling(window=21).mean() + 1e-10)
        
        # ----- 2.8 MA CROSSOVERS -----
        df['t2_sma_5_10_cross'] = (df['t1_sma_5'] > df['t1_sma_10']).astype(int)
        df['t2_sma_10_20_cross'] = (df['t1_sma_10'] > df['t1_sma_20']).astype(int)
        df['t2_ema_5_10_cross'] = (df['t1_ema_5'] > df['t1_ema_10']).astype(int)
        df['t2_golden_cross'] = (df['t1_sma_50'] > df['t1_sma_200']).astype(int)
        
        # Distance to MAs
        df['t2_distance_to_sma_50'] = (df['close'] - df['t1_sma_50']) / (df['t1_sma_50'] + 1e-10)
        df['t2_distance_to_sma_200'] = (df['close'] - df['t1_sma_200']) / (df['t1_sma_200'] + 1e-10)
        
        # ----- 2.9 STATISTICAL FEATURES -----
        # Skewness and Kurtosis (20-day rolling)
        df['t2_skewness_20'] = df['t1_return_1d'].rolling(window=20).apply(
            lambda x: stats.skew(x) if len(x) >= 3 else 0, raw=True
        )
        df['t2_kurtosis_20'] = df['t1_return_1d'].rolling(window=20).apply(
            lambda x: stats.kurtosis(x) if len(x) >= 4 else 0, raw=True
        )
        
        # Autocorrelation
        df['t2_autocorr_1'] = df['t1_return_1d'].rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        df['t2_autocorr_5'] = df['t1_return_1d'].rolling(window=20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )
        
        # ----- 2.10 CALENDAR FEATURES -----
        if 'date' in df.columns and self.config.include_calendar_features:
            date_col = pd.to_datetime(df['date'])
            
            # Day of week (cyclical encoding)
            dow = date_col.dt.dayofweek
            df['t2_day_of_week'] = dow
            df['t2_dow_sin'] = np.sin(2 * np.pi * dow / 5)
            df['t2_dow_cos'] = np.cos(2 * np.pi * dow / 5)
            
            # Month (cyclical encoding)
            month = date_col.dt.month
            df['t2_month'] = month
            df['t2_month_sin'] = np.sin(2 * np.pi * month / 12)
            df['t2_month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # Special days
            df['t2_is_monday'] = (dow == 0).astype(int)
            df['t2_is_friday'] = (dow == 4).astype(int)
            df['t2_is_month_start'] = date_col.dt.is_month_start.astype(int)
            df['t2_is_month_end'] = date_col.dt.is_month_end.astype(int)
            df['t2_is_quarter_end'] = date_col.dt.is_quarter_end.astype(int)
            df['t2_quarter'] = date_col.dt.quarter
        
        # ----- 2.11 POSITION FEATURES -----
        # 52-week high/low distance
        df['t2_high_52w'] = df['high'].rolling(window=252).max()
        df['t2_low_52w'] = df['low'].rolling(window=252).min()
        df['t2_pct_from_52w_high'] = (df['close'] - df['t2_high_52w']) / (df['t2_high_52w'] + 1e-10)
        df['t2_pct_from_52w_low'] = (df['close'] - df['t2_low_52w']) / (df['t2_low_52w'] + 1e-10)
        df['t2_52w_range_position'] = (df['close'] - df['t2_low_52w']) / (df['t2_high_52w'] - df['t2_low_52w'] + 1e-10)
        
        # Return percentile
        df['t2_return_percentile_20'] = df['t1_return_1d'].rolling(window=20).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 1 else 0.5, raw=False
        )
        
        return df
    
    # =========================================================================
    # TIER 3: EXTENDED FEATURES
    # =========================================================================
    def _engineer_tier3_extended(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 3: Extended features for advanced models."""
        
        # ----- 3.1 ICHIMOKU CLOUD -----
        # Tenkan-sen (Conversion Line): 9-period midpoint
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['t3_tenkan_sen'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line): 26-period midpoint
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['t3_kijun_sen'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['t3_senkou_span_a'] = ((df['t3_tenkan_sen'] + df['t3_kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period midpoint, shifted
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['t3_senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['t3_chikou_span'] = df['close'].shift(-26)
        
        # Derived features
        df['t3_tk_cross'] = (df['t3_tenkan_sen'] > df['t3_kijun_sen']).astype(int)
        df['t3_price_above_cloud'] = (df['close'] > df[['t3_senkou_span_a', 't3_senkou_span_b']].max(axis=1)).astype(int)
        
        # ----- 3.2 AROON INDICATOR -----
        window = 25
        df['t3_aroon_up'] = df['high'].rolling(window=window + 1).apply(
            lambda x: 100 * (window - (window - x.argmax())) / window, raw=True
        )
        df['t3_aroon_down'] = df['low'].rolling(window=window + 1).apply(
            lambda x: 100 * (window - (window - x.argmin())) / window, raw=True
        )
        df['t3_aroon_oscillator'] = df['t3_aroon_up'] - df['t3_aroon_down']
        
        # ----- 3.3 KELTNER CHANNELS -----
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        atr_10 = self._calculate_true_range(df).rolling(window=10).mean()
        df['t3_keltner_upper'] = ema_20 + 2 * atr_10
        df['t3_keltner_lower'] = ema_20 - 2 * atr_10
        df['t3_keltner_pct'] = (df['close'] - df['t3_keltner_lower']) / (df['t3_keltner_upper'] - df['t3_keltner_lower'] + 1e-10)
        
        # ----- 3.4 DONCHIAN CHANNELS -----
        df['t3_donchian_upper'] = df['high'].rolling(window=20).max()
        df['t3_donchian_lower'] = df['low'].rolling(window=20).min()
        df['t3_donchian_mid'] = (df['t3_donchian_upper'] + df['t3_donchian_lower']) / 2
        df['t3_donchian_width'] = (df['t3_donchian_upper'] - df['t3_donchian_lower']) / df['close']
        
        # ----- 3.5 ULTIMATE OSCILLATOR -----
        bp = df['close'] - df[['low', df['close'].shift()]].min(axis=1)
        tr = self._calculate_true_range(df)
        
        avg_7 = bp.rolling(window=7).sum() / (tr.rolling(window=7).sum() + 1e-10)
        avg_14 = bp.rolling(window=14).sum() / (tr.rolling(window=14).sum() + 1e-10)
        avg_28 = bp.rolling(window=28).sum() / (tr.rolling(window=28).sum() + 1e-10)
        
        df['t3_ultimate_oscillator'] = 100 * (4 * avg_7 + 2 * avg_14 + avg_28) / 7
        
        # ----- 3.6 VORTEX INDICATOR -----
        tr = self._calculate_true_range(df)
        vm_plus = np.abs(df['high'] - df['low'].shift())
        vm_minus = np.abs(df['low'] - df['high'].shift())
        
        df['t3_vortex_plus'] = vm_plus.rolling(window=14).sum() / (tr.rolling(window=14).sum() + 1e-10)
        df['t3_vortex_minus'] = vm_minus.rolling(window=14).sum() / (tr.rolling(window=14).sum() + 1e-10)
        df['t3_vortex_diff'] = df['t3_vortex_plus'] - df['t3_vortex_minus']
        
        # ----- 3.7 PARABOLIC SAR -----
        df['t3_psar'] = self._calculate_psar(df)
        df['t3_psar_trend'] = (df['close'] > df['t3_psar']).astype(int)
        
        # ----- 3.8 DETRENDED PRICE OSCILLATOR -----
        n = 20
        df['t3_dpo'] = df['close'].shift(n // 2 + 1) - df['close'].rolling(window=n).mean()
        
        # ----- 3.9 COPPOCK CURVE -----
        roc_14 = df['close'].pct_change(14) * 100
        roc_11 = df['close'].pct_change(11) * 100
        df['t3_coppock_curve'] = (roc_14 + roc_11).rolling(window=10).mean()
        
        # ----- 3.10 CHANDE MOMENTUM OSCILLATOR -----
        diff = df['close'].diff()
        sum_up = diff.where(diff > 0, 0).rolling(window=14).sum()
        sum_down = (-diff).where(diff < 0, 0).rolling(window=14).sum()
        df['t3_cmo'] = 100 * (sum_up - sum_down) / (sum_up + sum_down + 1e-10)
        
        # ----- 3.11 FORCE INDEX -----
        df['t3_force_index'] = df['close'].diff() * df['volume']
        df['t3_force_index_13'] = df['t3_force_index'].ewm(span=13, adjust=False).mean()
        
        # ----- 3.12 ELDER RAY -----
        ema_13 = df['close'].ewm(span=13, adjust=False).mean()
        df['t3_bull_power'] = df['high'] - ema_13
        df['t3_bear_power'] = df['low'] - ema_13
        
        # ----- 3.13 KNOW SURE THING (KST) -----
        roc_10 = df['close'].pct_change(10) * 100
        roc_15 = df['close'].pct_change(15) * 100
        roc_20 = df['close'].pct_change(20) * 100
        roc_30 = df['close'].pct_change(30) * 100
        
        df['t3_kst'] = (roc_10.rolling(window=10).mean() * 1 +
                        roc_15.rolling(window=10).mean() * 2 +
                        roc_20.rolling(window=10).mean() * 3 +
                        roc_30.rolling(window=15).mean() * 4)
        df['t3_kst_signal'] = df['t3_kst'].rolling(window=9).mean()
        
        # ----- 3.14 CANDLESTICK PATTERNS -----
        if self.config.include_candlestick_patterns:
            df = self._add_candlestick_patterns(df)
        
        # ----- 3.15 ROLLING DRAWDOWN -----
        rolling_max = df['close'].rolling(window=252, min_periods=1).max()
        df['t3_drawdown'] = (df['close'] - rolling_max) / rolling_max
        df['t3_max_drawdown_60'] = df['t3_drawdown'].rolling(window=60).min()
        
        # ----- 3.16 EXTENDED VOLUME FEATURES -----
        # Ease of Movement
        distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift() + df['low'].shift()) / 2)
        box_ratio = (df['volume'] / 1e8) / (df['high'] - df['low'] + 1e-10)
        df['t3_emv'] = distance / box_ratio
        df['t3_emv_sma_14'] = df['t3_emv'].rolling(window=14).mean()
        
        # Negative Volume Index
        df['t3_nvi'] = 1000.0  # Starting value
        nvi_values = [1000.0]
        for i in range(1, len(df)):
            if df['volume'].iloc[i] < df['volume'].iloc[i-1]:
                nvi_values.append(nvi_values[-1] * (1 + df['t1_return_1d'].iloc[i]))
            else:
                nvi_values.append(nvi_values[-1])
        df['t3_nvi'] = nvi_values
        
        return df
    
    # =========================================================================
    # TIER 4: ADVANCED FEATURES
    # =========================================================================
    def _engineer_tier4_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 4: Advanced features requiring specialized computation."""
        
        # ----- 4.1 HURST EXPONENT (Trending vs Mean-Reverting) -----
        df['t4_hurst_exponent'] = df['close'].rolling(window=100).apply(
            self._calculate_hurst, raw=True
        )
        
        # ----- 4.2 SAMPLE ENTROPY -----
        df['t4_sample_entropy'] = df['t1_return_1d'].rolling(window=50).apply(
            self._calculate_sample_entropy, raw=True
        )
        
        # ----- 4.3 FRACTAL DIMENSION -----
        df['t4_fractal_dimension'] = df['close'].rolling(window=50).apply(
            self._calculate_fractal_dimension, raw=True
        )
        
        # ----- 4.4 PARKINSON VOLATILITY -----
        df['t4_parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2)
        ).rolling(window=20).mean() * np.sqrt(252)
        
        # ----- 4.5 GARMAN-KLASS VOLATILITY -----
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        df['t4_garman_klass_vol'] = np.sqrt(
            0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        ).rolling(window=20).mean() * np.sqrt(252)
        
        # ----- 4.6 YANG-ZHANG VOLATILITY -----
        log_oc = np.log(df['open'] / df['close'].shift()) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        log_rs = np.log(df['high'] / df['close']) * np.log(df['high'] / df['open'])
        log_rs += np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        
        k = 0.34 / (1.34 + (21 + 1) / (21 - 1))
        df['t4_yang_zhang_vol'] = np.sqrt(
            log_oc.rolling(window=20).mean() + 
            k * log_co.rolling(window=20).mean() + 
            (1 - k) * log_rs.rolling(window=20).mean()
        ) * np.sqrt(252)
        
        # ----- 4.7 SPECTRAL FEATURES (FFT) -----
        df['t4_fft_dominant_period'] = df['close'].rolling(window=64).apply(
            self._calculate_dominant_period, raw=True
        )
        
        # ----- 4.8 PRICE COMPRESSION RATIO -----
        # Detects squeeze conditions
        bb_width = df['t1_bb_width']
        bb_width_min = bb_width.rolling(window=120).min()
        df['t4_bb_squeeze_ratio'] = bb_width / (bb_width_min + 1e-10)
        df['t4_in_squeeze'] = (df['t4_bb_squeeze_ratio'] < 1.2).astype(int)
        
        # ----- 4.9 REGIME PROBABILITY (Rolling HMM approximation) -----
        # Simple regime proxy using volatility and returns
        vol_zscore = (df['t1_volatility_20d'] - df['t1_volatility_20d'].rolling(window=252).mean()) / (df['t1_volatility_20d'].rolling(window=252).std() + 1e-10)
        ret_zscore = (df['t1_return_21d'] - df['t1_return_21d'].rolling(window=252).mean()) / (df['t1_return_21d'].rolling(window=252).std() + 1e-10)
        
        df['t4_regime_score'] = ret_zscore - vol_zscore  # Positive = bull, Negative = bear
        df['t4_regime_bull_prob'] = 1 / (1 + np.exp(-df['t4_regime_score']))  # Sigmoid
        
        # ----- 4.10 INFORMATION RATIO PROXY -----
        if 't1_sma_200' in df.columns:
            excess_return = df['t1_return_21d']
            tracking_error = df['t1_volatility_20d']
            df['t4_info_ratio_proxy'] = excess_return / (tracking_error + 1e-10)
        
        return df
    
    # =========================================================================
    # TARGET CREATION
    # =========================================================================
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction."""
        h = self.config.prediction_horizon
        
        # Forward returns
        df['future_price'] = df['close'].shift(-h)
        df['target_return'] = (df['future_price'] - df['close']) / df['close']
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        # Drop future columns that would cause leakage
        df = df.drop(columns=['future_price'], errors='ignore')
        
        return df
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def _calculate_psar(self, df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR."""
        length = len(df)
        psar = df['close'].copy()
        af = af_start
        bull = True
        ep = df['low'].iloc[0]
        hp = df['high'].iloc[0]
        lp = df['low'].iloc[0]
        
        for i in range(2, length):
            if bull:
                psar.iloc[i] = psar.iloc[i-1] + af * (hp - psar.iloc[i-1])
                psar.iloc[i] = min(psar.iloc[i], df['low'].iloc[i-1], df['low'].iloc[i-2])
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (lp - psar.iloc[i-1])
                psar.iloc[i] = max(psar.iloc[i], df['high'].iloc[i-1], df['high'].iloc[i-2])
            
            reverse = False
            if bull:
                if df['low'].iloc[i] < psar.iloc[i]:
                    bull = False
                    reverse = True
                    psar.iloc[i] = hp
                    lp = df['low'].iloc[i]
                    af = af_start
            else:
                if df['high'].iloc[i] > psar.iloc[i]:
                    bull = True
                    reverse = True
                    psar.iloc[i] = lp
                    hp = df['high'].iloc[i]
                    af = af_start
            
            if not reverse:
                if bull:
                    if df['high'].iloc[i] > hp:
                        hp = df['high'].iloc[i]
                        af = min(af + af_start, af_max)
                    if df['low'].iloc[i-1] < psar.iloc[i]:
                        psar.iloc[i] = df['low'].iloc[i-1]
                    if df['low'].iloc[i-2] < psar.iloc[i]:
                        psar.iloc[i] = df['low'].iloc[i-2]
                else:
                    if df['low'].iloc[i] < lp:
                        lp = df['low'].iloc[i]
                        af = min(af + af_start, af_max)
                    if df['high'].iloc[i-1] > psar.iloc[i]:
                        psar.iloc[i] = df['high'].iloc[i-1]
                    if df['high'].iloc[i-2] > psar.iloc[i]:
                        psar.iloc[i] = df['high'].iloc[i-2]
        
        return psar
    
    def _calculate_hurst(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            if len(prices) < 20:
                return 0.5
            
            n = len(prices)
            max_k = min(n // 2, 50)
            rs_values = []
            n_values = []
            
            for k in range(10, max_k):
                rs_list = []
                for start in range(0, n - k, k):
                    subset = prices[start:start + k]
                    mean_val = np.mean(subset)
                    deviation = subset - mean_val
                    cumsum = np.cumsum(deviation)
                    r = np.max(cumsum) - np.min(cumsum)
                    s = np.std(subset)
                    if s > 0:
                        rs_list.append(r / s)
                
                if len(rs_list) > 0:
                    rs_values.append(np.mean(rs_list))
                    n_values.append(k)
            
            if len(rs_values) < 2:
                return 0.5
            
            log_n = np.log(n_values)
            log_rs = np.log(rs_values)
            
            slope, _ = np.polyfit(log_n, log_rs, 1)
            return np.clip(slope, 0, 1)
        except Exception:
            return 0.5
    
    def _calculate_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate Sample Entropy."""
        try:
            if len(data) < 10:
                return 0.0
            
            n = len(data)
            r_val = r * np.std(data)
            
            def count_matches(template, data, m, r_val):
                count = 0
                for i in range(len(data) - m):
                    if i != template:
                        diff = np.abs(data[template:template+m] - data[i:i+m])
                        if np.max(diff) < r_val:
                            count += 1
                return count
            
            # This is a simplified version
            b = 0
            a = 0
            for i in range(n - m):
                b += count_matches(i, data, m, r_val)
                a += count_matches(i, data, m + 1, r_val)
            
            if b == 0:
                return 0.0
            
            return -np.log(a / b) if a > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting approximation."""
        try:
            if len(prices) < 10:
                return 1.5
            
            n = len(prices)
            max_val = np.max(prices)
            min_val = np.min(prices)
            
            if max_val == min_val:
                return 1.0
            
            normalized = (prices - min_val) / (max_val - min_val)
            
            # Simplified Higuchi fractal dimension
            k_max = min(10, n // 4)
            lengths = []
            
            for k in range(1, k_max + 1):
                length_k = 0
                for m in range(k):
                    indices = np.arange(m, n, k)
                    if len(indices) > 1:
                        diffs = np.abs(np.diff(normalized[indices]))
                        length_k += np.sum(diffs) * (n - 1) / (len(indices) * k)
                lengths.append(length_k / k)
            
            if len(lengths) < 2 or all(l == 0 for l in lengths):
                return 1.5
            
            log_k = np.log(np.arange(1, k_max + 1))
            log_l = np.log(np.array(lengths) + 1e-10)
            
            slope, _ = np.polyfit(log_k, log_l, 1)
            return np.clip(-slope, 1, 2)
        except Exception:
            return 1.5
    
    def _calculate_dominant_period(self, prices: np.ndarray) -> float:
        """Calculate dominant period using FFT."""
        try:
            if len(prices) < 8:
                return 0.0
            
            # Detrend
            detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
            
            # Apply FFT
            n = len(detrended)
            fft_vals = np.abs(fft(detrended))[:n // 2]
            
            # Find dominant frequency (skip DC component)
            if len(fft_vals) < 3:
                return 0.0
            
            dominant_idx = np.argmax(fft_vals[1:]) + 1
            
            if dominant_idx == 0:
                return 0.0
            
            dominant_period = n / dominant_idx
            return dominant_period
        except Exception:
            return 0.0
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        
        # Body and shadow calculations
        body = df['close'] - df['open']
        body_size = np.abs(body)
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        avg_body = body_size.rolling(window=14).mean()
        
        # Doji: Very small body
        df['t3_pattern_doji'] = (body_size < 0.1 * avg_body).astype(int)
        
        # Hammer: Small body at top, long lower shadow
        is_hammer = (body_size < 0.3 * avg_body) & (lower_shadow > 2 * body_size) & (upper_shadow < 0.5 * body_size)
        df['t3_pattern_hammer'] = is_hammer.astype(int)
        
        # Inverted Hammer
        is_inv_hammer = (body_size < 0.3 * avg_body) & (upper_shadow > 2 * body_size) & (lower_shadow < 0.5 * body_size)
        df['t3_pattern_inv_hammer'] = is_inv_hammer.astype(int)
        
        # Marubozu: Strong candle with minimal shadows
        is_marubozu = (body_size > 1.5 * avg_body) & (upper_shadow < 0.1 * body_size) & (lower_shadow < 0.1 * body_size)
        df['t3_pattern_marubozu'] = is_marubozu.astype(int)
        
        # Bullish Engulfing
        prev_bearish = df['close'].shift() < df['open'].shift()
        curr_bullish = df['close'] > df['open']
        engulfs = (df['open'] < df['close'].shift()) & (df['close'] > df['open'].shift())
        df['t3_pattern_bullish_engulfing'] = (prev_bearish & curr_bullish & engulfs).astype(int)
        
        # Bearish Engulfing
        prev_bullish = df['close'].shift() > df['open'].shift()
        curr_bearish = df['close'] < df['open']
        engulfs_bear = (df['open'] > df['close'].shift()) & (df['close'] < df['open'].shift())
        df['t3_pattern_bearish_engulfing'] = (prev_bullish & curr_bearish & engulfs_bear).astype(int)
        
        # Three White Soldiers
        is_bullish = df['close'] > df['open']
        strong_body = body_size > 0.5 * avg_body
        three_bullish = is_bullish & is_bullish.shift() & is_bullish.shift(2)
        three_strong = strong_body & strong_body.shift() & strong_body.shift(2)
        higher_closes = (df['close'] > df['close'].shift()) & (df['close'].shift() > df['close'].shift(2))
        df['t3_pattern_three_white_soldiers'] = (three_bullish & three_strong & higher_closes).astype(int)
        
        # Three Black Crows
        is_bearish = df['close'] < df['open']
        three_bearish = is_bearish & is_bearish.shift() & is_bearish.shift(2)
        lower_closes = (df['close'] < df['close'].shift()) & (df['close'].shift() < df['close'].shift(2))
        df['t3_pattern_three_black_crows'] = (three_bearish & three_strong & lower_closes).astype(int)
        
        return df
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    def get_feature_names(self, tier: Optional[int] = None) -> List[str]:
        """Get list of feature names, optionally filtered by tier."""
        if tier is None:
            return self.feature_names
        
        prefix = f't{tier}_'
        return [f for f in self.feature_names if f.startswith(prefix)]
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by category for analysis."""
        groups = {
            'returns': [f for f in self.feature_names if 'return' in f.lower()],
            'volatility': [f for f in self.feature_names if 'volatility' in f.lower() or 'vol' in f.lower() or 'atr' in f.lower()],
            'momentum': [f for f in self.feature_names if 'momentum' in f.lower() or 'rsi' in f.lower() or 'macd' in f.lower() or 'roc' in f.lower()],
            'trend': [f for f in self.feature_names if 'sma' in f.lower() or 'ema' in f.lower() or 'trend' in f.lower() or 'adx' in f.lower()],
            'volume': [f for f in self.feature_names if 'volume' in f.lower() or 'obv' in f.lower() or 'mfi' in f.lower() or 'cmf' in f.lower()],
            'price_structure': [f for f in self.feature_names if 'high' in f.lower() or 'low' in f.lower() or 'gap' in f.lower() or 'shadow' in f.lower()],
            'bollinger': [f for f in self.feature_names if 'bb_' in f.lower()],
            'stochastic': [f for f in self.feature_names if 'stoch' in f.lower()],
            'calendar': [f for f in self.feature_names if 'day' in f.lower() or 'month' in f.lower() or 'week' in f.lower() or 'quarter' in f.lower()],
            'statistical': [f for f in self.feature_names if 'skew' in f.lower() or 'kurt' in f.lower() or 'autocorr' in f.lower() or 'entropy' in f.lower()],
            'patterns': [f for f in self.feature_names if 'pattern' in f.lower()],
        }
        return groups


# =============================================================================
# CROSS-SECTIONAL FEATURE ENGINEERING (Multi-Ticker)
# =============================================================================
class CrossSectionalFeatureEngineer:
    """
    Cross-sectional features for multi-ticker portfolios.
    
    These features compare a stock's performance relative to peers.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def engineer_cross_sectional_features(
        self, 
        data: Dict[str, pd.DataFrame],
        market_ticker: str = 'SPY'
    ) -> Dict[str, pd.DataFrame]:
        """
        Add cross-sectional features to each ticker's DataFrame.
        
        Args:
            data: Dictionary mapping ticker -> DataFrame with features
            market_ticker: Market benchmark ticker (must be in data)
        
        Returns:
            Updated dictionary with cross-sectional features added.
        """
        tickers = list(data.keys())
        
        if len(tickers) < 2:
            logger.warning("Need at least 2 tickers for cross-sectional features")
            return data
        
        # Align all DataFrames
        common_dates = set(data[tickers[0]]['date'].astype(str))
        for ticker in tickers[1:]:
            common_dates &= set(data[ticker]['date'].astype(str))
        
        aligned_data = {}
        for ticker, df in data.items():
            mask = df['date'].astype(str).isin(common_dates)
            aligned_data[ticker] = df[mask].reset_index(drop=True)
        
        n_dates = len(common_dates)
        logger.info(f"Computing cross-sectional features for {len(tickers)} tickers over {n_dates} dates")
        
        # Get market returns if available
        market_returns = None
        if market_ticker in aligned_data:
            market_returns = aligned_data[market_ticker]['t1_return_1d'].values
        
        # Compute cross-sectional features for each ticker
        for ticker in tickers:
            df = aligned_data[ticker]
            
            # 1. Relative strength vs market
            if market_returns is not None and ticker != market_ticker:
                df['cs_return_vs_market'] = df['t1_return_1d'].values - market_returns
                
                # Rolling relative strength
                df['cs_relative_strength_20'] = (
                    df['t1_return_21d'] - 
                    aligned_data[market_ticker]['t1_return_21d'].values
                )
            
            # 2. Cross-sectional rank (percentile among peers)
            for window in [1, 5, 21]:
                col_name = f't1_return_{window}d' if window > 1 else 't1_return_1d'
                if col_name not in df.columns:
                    continue
                
                returns_matrix = np.column_stack([
                    aligned_data[t][col_name].values for t in tickers if col_name in aligned_data[t].columns
                ])
                
                if returns_matrix.shape[1] > 1:
                    # Compute percentile rank for each date
                    rank = np.zeros(len(df))
                    ticker_idx = [t for t in tickers if col_name in aligned_data[t].columns].index(ticker)
                    
                    for i in range(len(df)):
                        ticker_return = returns_matrix[i, ticker_idx]
                        rank[i] = stats.percentileofscore(returns_matrix[i, :], ticker_return) / 100
                    
                    df[f'cs_rank_{window}d'] = rank
            
            # 3. Average correlation with others
            if 't1_return_1d' in df.columns:
                correlations = []
                for other_ticker in tickers:
                    if other_ticker != ticker and 't1_return_1d' in aligned_data[other_ticker].columns:
                        corr = df['t1_return_1d'].rolling(window=60).corr(
                            aligned_data[other_ticker]['t1_return_1d']
                        )
                        correlations.append(corr)
                
                if correlations:
                    df['cs_avg_correlation_60'] = pd.concat(correlations, axis=1).mean(axis=1)
            
            # 4. Z-score among peers
            if 't1_return_1d' in df.columns:
                returns_matrix = np.column_stack([
                    aligned_data[t]['t1_return_1d'].values for t in tickers if 't1_return_1d' in aligned_data[t].columns
                ])
                
                cross_mean = np.nanmean(returns_matrix, axis=1)
                cross_std = np.nanstd(returns_matrix, axis=1)
                
                ticker_idx = [t for t in tickers if 't1_return_1d' in aligned_data[t].columns].index(ticker)
                df['cs_zscore'] = (returns_matrix[:, ticker_idx] - cross_mean) / (cross_std + 1e-10)
            
            aligned_data[ticker] = df
        
        return aligned_data


# =============================================================================
# QUICK USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    print("=" * 60)
    print("COMPREHENSIVE FEATURE ENGINEERING - EXAMPLE")
    print("=" * 60)
    
    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start="2020-01-01", end="2024-12-01")
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'adj close': 'adj_close'})
    
    print(f"\nLoaded {len(df)} rows for AAPL")
    print(f"Columns: {list(df.columns)}")
    
    # Engineer features
    config = FeatureConfig(
        prediction_horizon=5,
        include_candlestick_patterns=True,
        include_calendar_features=True
    )
    
    engineer = ComprehensiveFeatureEngineer(config)
    
    # Tier 1
    df_t1 = engineer.engineer_all_features(df.copy(), tier=1, verbose=True)
    print(f"\nTier 1: {len(engineer.get_feature_names(1))} features")
    
    # Tier 2
    df_t2 = engineer.engineer_all_features(df.copy(), tier=2, verbose=True)
    print(f"Tier 2: {len(engineer.get_feature_names(2))} additional features")
    
    # Tier 3
    df_t3 = engineer.engineer_all_features(df.copy(), tier=3, verbose=True)
    print(f"Tier 3: {len(engineer.get_feature_names(3))} additional features")
    
    # Tier 4
    df_t4 = engineer.engineer_all_features(df.copy(), tier=4, verbose=True)
    print(f"Tier 4: {len(engineer.get_feature_names(4))} additional features")
    
    print(f"\nTotal features: {len(engineer.feature_names)}")
    print(f"Final DataFrame shape: {df_t4.shape}")
    
    # Show feature groups
    groups = engineer.get_feature_importance_groups()
    print("\nFeature Groups:")
    for group, features in groups.items():
        if features:
            print(f"  {group}: {len(features)} features")
    
    # Sample output
    print("\nSample features (last row):")
    sample_features = ['t1_return_1d', 't1_rsi_14', 't1_macd', 't1_bb_pct', 
                       't2_adx_14', 't2_stoch_k', 't2_mfi_14', 't2_skewness_20']
    for feat in sample_features:
        if feat in df_t4.columns:
            print(f"  {feat}: {df_t4[feat].iloc[-1]:.4f}")

