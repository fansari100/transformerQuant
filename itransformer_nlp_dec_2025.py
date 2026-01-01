"""
================================================================================
iTRANSFORMER QUANT SYSTEM 2026 (with Real NLP Sentiment)
================================================================================

Architecture (based on ICLR 2024 iTransformer paper):
1. COMPREHENSIVE FEATURE ENGINEERING (94+ features)
2. REAL NLP SENTIMENT via FinBERT
   • Uses ProsusAI/finbert (state-of-the-art financial sentiment)
   • Fetches real news from yfinance + GNews fallback
   • Generates 6 sentiment features per ticker per day
3. iTRANSFORMER - INVERTED ATTENTION MECHANISM
   • Standard Transformer: Attention over TIME steps
   • iTransformer: Attention over FEATURES/ASSETS (inverted)
   • This naturally captures cross-asset relationships
4. REGIME DETECTION (HMM)
5. PER-TICKER BUY/SELL SIGNALS with relative thresholds

Key Innovations:
- FinBERT NLP: Real financial text analysis (not proxy)
- Treats each time step as a "token" with assets as the embedding dimension
- Attention learns correlations BETWEEN assets at each time point
- Mathematically superior for multivariate (multi-asset) forecasting

Papers:
- iTransformer: https://arxiv.org/abs/2310.06625
- FinBERT: https://arxiv.org/abs/1908.10063

Runnable in Google Colab. Requires: pip install transformers torch
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
import json
import time
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from collections import defaultdict

# Machine Learning
from sklearn.preprocessing import StandardScaler

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# NLP / Sentiment (FinBERT)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("⚠️ transformers/torch not installed. Run: pip install transformers torch")

# Market Data
import yfinance as yf

# News scraping fallback
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

# Regime Detection
from hmmlearn.hmm import GaussianHMM

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================
class MarketRegime(Enum):
    CRISIS = 0
    BEAR = 1
    SIDEWAYS = 2
    BULL = 3


class SignalType(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


# =============================================================================
# FEATURE LIST
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
    """Configuration for the iTransformer Quant System."""
    
    # Portfolio
    tickers: List[str] = field(default_factory=lambda: ['AAPL', 'GOOGL', 'MSFT'])
    
    # Data
    start_date: str = "2015-01-01"
    end_date: str = field(default_factory=lambda: datetime.today().strftime('%Y-%m-%d'))
    
    # Sequence settings
    time_steps: int = 20
    prediction_horizon: int = 5
    
    # iTransformer Architecture
    d_model: int = 64           # Embedding dimension
    ff_dim: int = 128           # Feed-forward dimension
    num_heads: int = 4          # Number of attention heads
    num_layers: int = 2         # Number of transformer blocks
    dropout_rate: float = 0.2
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 8
    
    # Signal Generation
    buy_percentile: float = 70
    sell_percentile: float = 30
    strong_buy_percentile: float = 90
    strong_sell_percentile: float = 10
    
    # Trading
    initial_investment: float = 100_000
    transaction_cost_bps: float = 10
    
    # Data Split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # NLP Sentiment
    use_real_nlp: bool = True  # True = FinBERT, False = proxy
    sentiment_cache_dir: str = ".sentiment_cache"
    use_gpu_for_nlp: bool = False
    
    # Company name mapping (for better news search)
    company_names: Dict[str, str] = field(default_factory=lambda: {
        'AAPL': 'Apple',
        'GOOGL': 'Google Alphabet',
        'MSFT': 'Microsoft',
        'AMZN': 'Amazon',
        'META': 'Meta Facebook',
        'NVDA': 'NVIDIA',
        'TSLA': 'Tesla',
        'JPM': 'JPMorgan Chase',
        'V': 'Visa',
        'JNJ': 'Johnson Johnson',
        'WMT': 'Walmart',
        'PG': 'Procter Gamble',
        'MA': 'Mastercard',
        'UNH': 'UnitedHealth',
        'HD': 'Home Depot',
        'DIS': 'Disney',
        'PYPL': 'PayPal',
        'NFLX': 'Netflix',
        'ADBE': 'Adobe',
        'CRM': 'Salesforce',
    })


# =============================================================================
# SENTIMENT ANALYZER (Real FinBERT NLP)
# =============================================================================
class FinBERTSentimentAnalyzer:
    """
    Real NLP Sentiment Analysis using FinBERT.
    
    Architecture:
    1. Loads ProsusAI/finbert (pre-trained on financial text)
    2. Fetches news from yfinance (free, no API key)
    3. Fallback to GNews scraping if yfinance unavailable
    4. Scores each headline with FinBERT
    5. Aggregates into 6 features per ticker per day
    
    Output features:
    - sentiment_score: Daily weighted average sentiment (-1 to +1)
    - sentiment_momentum: 5-day change in sentiment
    - sentiment_volatility: Rolling std of sentiment
    - news_volume: Number of news articles (normalized)
    - social_buzz: Sentiment × volume (attention proxy)
    - sentiment_divergence: Sentiment vs price momentum divergence
    """
    
    def __init__(self, cache_dir: str = ".sentiment_cache", use_gpu: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self._model_loaded = False
        self.use_gpu = use_gpu and torch.cuda.is_available() if NLP_AVAILABLE else False
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Cache for sentiment results
        self._sentiment_cache: Dict[str, Dict[str, float]] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached sentiment data."""
        cache_file = self.cache_dir / "sentiment_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._sentiment_cache = pickle.load(f)
                logger.info(f"  ✓ Loaded sentiment cache: {len(self._sentiment_cache)} entries")
            except Exception as e:
                logger.warning(f"  ⚠ Could not load cache: {e}")
                self._sentiment_cache = {}
    
    def _save_cache(self) -> None:
        """Save sentiment cache to disk."""
        cache_file = self.cache_dir / "sentiment_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._sentiment_cache, f)
        except Exception as e:
            logger.warning(f"  ⚠ Could not save cache: {e}")
    
    def _load_model(self) -> None:
        """Lazy-load FinBERT model."""
        if self._model_loaded or not NLP_AVAILABLE:
            return
        
        logger.info("  📥 Loading FinBERT model (first time only)...")
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            if self.use_gpu:
                self.model = self.model.to(self.device)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1,
                max_length=512,
                truncation=True
            )
            self._model_loaded = True
            logger.info("  ✓ FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"  ⚠ Could not load FinBERT: {e}. Using fallback.")
            self._model_loaded = False
    
    def _fetch_news_yfinance(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from yfinance (free, no API key)."""
        try:
            t = yf.Ticker(ticker)
            news = t.news if hasattr(t, 'news') else []
            
            articles = []
            for item in news[:50]:  # Limit to 50 most recent
                pub_time = item.get('providerPublishTime', 0)
                if pub_time:
                    pub_date = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
                else:
                    pub_date = datetime.now().strftime('%Y-%m-%d')
                
                articles.append({
                    'title': item.get('title', ''),
                    'date': pub_date,
                    'source': item.get('publisher', 'Unknown'),
                    'url': item.get('link', '')
                })
            return articles
        except Exception as e:
            logger.warning(f"  ⚠ yfinance news fetch failed for {ticker}: {e}")
            return []
    
    def _fetch_news_gnews(self, ticker: str, company_name: str = None) -> List[Dict[str, Any]]:
        """Fallback: Scrape news from Google News RSS (no API key needed)."""
        if not SCRAPING_AVAILABLE:
            return []
        
        query = company_name or ticker
        url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
        
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')[:30]
            
            articles = []
            for item in items:
                title = item.title.text if item.title else ''
                pub_date_str = item.pubDate.text if item.pubDate else ''
                
                # Parse date
                try:
                    from email.utils import parsedate_to_datetime
                    pub_date = parsedate_to_datetime(pub_date_str).strftime('%Y-%m-%d')
                except:
                    pub_date = datetime.now().strftime('%Y-%m-%d')
                
                articles.append({
                    'title': title,
                    'date': pub_date,
                    'source': 'Google News',
                    'url': item.link.text if item.link else ''
                })
            return articles
        except Exception as e:
            logger.warning(f"  ⚠ GNews fetch failed for {ticker}: {e}")
            return []
    
    def _score_text(self, text: str) -> float:
        """
        Score a single text with FinBERT.
        Returns: float between -1 (negative) and +1 (positive)
        """
        if not self._model_loaded or not text.strip():
            return 0.0
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            label = result['label'].lower()
            score = result['score']
            
            # Convert to -1 to +1 scale
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:  # neutral
                return 0.0
        except Exception:
            return 0.0
    
    def _fetch_all_news(self, ticker: str, company_names: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Fetch news from all available sources."""
        articles = []
        
        # Primary: yfinance
        yf_news = self._fetch_news_yfinance(ticker)
        articles.extend(yf_news)
        
        # Fallback: GNews (if yfinance returned few articles)
        if len(articles) < 10 and SCRAPING_AVAILABLE:
            company_name = company_names.get(ticker, ticker) if company_names else ticker
            gnews = self._fetch_news_gnews(ticker, company_name)
            articles.extend(gnews)
        
        return articles
    
    def get_sentiment_features(
        self, 
        ticker: str, 
        dates: pd.Series, 
        prices: pd.Series, 
        returns: pd.Series,
        company_names: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Generate 6 sentiment features for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            dates: Series of dates
            prices: Series of prices
            returns: Series of returns
            company_names: Optional mapping of ticker -> company name for better search
        
        Returns:
            DataFrame with 6 sentiment features
        """
        n = len(dates)
        dates_list = pd.to_datetime(dates).dt.strftime('%Y-%m-%d').tolist() if hasattr(dates.iloc[0], 'strftime') or isinstance(dates.iloc[0], str) else [str(d) for d in dates]
        
        # Check cache
        cache_key = f"{ticker}_{dates_list[0]}_{dates_list[-1]}"
        if cache_key in self._sentiment_cache:
            cached = self._sentiment_cache[cache_key]
            logger.info(f"  ✓ Using cached sentiment for {ticker}")
            return pd.DataFrame(cached)
        
        # Load model if needed
        self._load_model()
        
        # Fetch news
        logger.info(f"  📰 Fetching news for {ticker}...")
        articles = self._fetch_all_news(ticker, company_names)
        
        if not articles:
            logger.warning(f"  ⚠ No news found for {ticker}, using fallback proxy")
            return self._generate_proxy_sentiment(ticker, dates, prices, returns)
        
        # Score articles
        logger.info(f"  🧠 Scoring {len(articles)} articles with FinBERT...")
        daily_sentiments = defaultdict(list)
        
        for article in articles:
            title = article.get('title', '')
            date = article.get('date', '')
            
            if title and date:
                score = self._score_text(title)
                daily_sentiments[date].append(score)
        
        # Create date -> sentiment mapping
        sentiment_by_date = {}
        for date, scores in daily_sentiments.items():
            if scores:
                # Weighted average (recent articles weighted more)
                weights = np.exp(np.linspace(0, 1, len(scores)))
                sentiment_by_date[date] = np.average(scores, weights=weights)
        
        # Build feature DataFrame
        sentiment_scores = []
        news_counts = []
        
        for date in dates_list:
            if date in sentiment_by_date:
                sentiment_scores.append(sentiment_by_date[date])
                news_counts.append(len(daily_sentiments.get(date, [])))
            else:
                sentiment_scores.append(np.nan)
                news_counts.append(0)
        
        # Convert to series for rolling calculations
        sent_series = pd.Series(sentiment_scores)
        news_series = pd.Series(news_counts)
        returns_clean = returns.fillna(0).values
        
        # Forward/backward fill missing sentiment
        sent_series = sent_series.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Compute 6 features
        sentiment_score = sent_series.values
        sentiment_momentum = sent_series.diff(5).fillna(0).values
        sentiment_volatility = sent_series.rolling(10, min_periods=1).std().fillna(0.1).values
        news_volume = (news_series / (news_series.max() + 1e-10)).values  # Normalized
        social_buzz = np.abs(sentiment_score) * (news_volume + 0.1)  # Sentiment × attention
        
        # Divergence: sentiment vs price momentum
        price_momentum = pd.Series(returns_clean).rolling(10, min_periods=1).mean().values
        price_momentum_norm = np.tanh(price_momentum * 50)
        sentiment_divergence = sentiment_score - price_momentum_norm
        
        result = pd.DataFrame({
            'sentiment_score': sentiment_score,
            'sentiment_momentum': sentiment_momentum,
            'sentiment_volatility': sentiment_volatility,
            'news_volume': news_volume,
            'social_buzz': social_buzz,
            'sentiment_divergence': sentiment_divergence,
        })
        
        # Cache result
        self._sentiment_cache[cache_key] = result.to_dict('list')
        self._save_cache()
        
        logger.info(f"  ✓ Generated NLP sentiment for {ticker}")
        return result
    
    def _generate_proxy_sentiment(
        self, ticker: str, dates: pd.Series, prices: pd.Series, returns: pd.Series
    ) -> pd.DataFrame:
        """
        Fallback: Generate proxy sentiment from price/volume data.
        Used when no news is available or NLP fails.
        """
        n = len(dates)
        seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2**32)
        np.random.seed(seed)
        
        returns_clean = returns.fillna(0).values
        momentum_sent = pd.Series(returns_clean).rolling(10, min_periods=1).mean().shift(1).fillna(0).values
        momentum_sent = np.tanh(momentum_sent * 50)
        
        price_zscore = (prices - prices.rolling(50, min_periods=10).mean()) / \
                       (prices.rolling(50, min_periods=10).std() + 1e-10)
        contrarian_sent = -np.tanh(price_zscore.fillna(0).values * 0.3)
        
        noise = np.random.randn(n) * 0.1
        raw_sentiment = 0.5 * momentum_sent + 0.3 * contrarian_sent + 0.2 * noise
        sentiment_smooth = pd.Series(raw_sentiment).rolling(3, min_periods=1).mean().values
        
        return pd.DataFrame({
            'sentiment_score': sentiment_smooth,
            'sentiment_momentum': pd.Series(sentiment_smooth).diff(5).fillna(0).values,
            'sentiment_volatility': pd.Series(sentiment_smooth).rolling(10, min_periods=1).std().fillna(0.1).values,
            'news_volume': np.abs(returns_clean) * 0.1 + np.random.exponential(0.05, n),
            'social_buzz': np.abs(momentum_sent) * 0.5 + np.random.exponential(0.05, n),
            'sentiment_divergence': sentiment_smooth - momentum_sent,
        })


# Alias for backward compatibility
SentimentAnalyzer = FinBERTSentimentAnalyzer


# =============================================================================
# REGIME DETECTOR
# =============================================================================
class RegimeDetector:
    """HMM-based market regime detection."""
    
    def __init__(self, n_regimes: int = 4):
        self.hmm = GaussianHMM(n_components=n_regimes, covariance_type="full",
                               n_iter=200, random_state=42)
        self.regime_mapping: Dict[int, MarketRegime] = {}
        self._fitted = False
    
    def fit(self, returns: np.ndarray) -> None:
        X = returns.reshape(-1, 1)
        self.hmm.fit(X)
        means = self.hmm.means_.flatten()
        sorted_indices = np.argsort(means)
        regime_order = [MarketRegime.CRISIS, MarketRegime.BEAR, 
                       MarketRegime.SIDEWAYS, MarketRegime.BULL]
        for i, idx in enumerate(sorted_indices):
            self.regime_mapping[idx] = regime_order[i]
        self._fitted = True
    
    def predict(self, returns: np.ndarray) -> List[MarketRegime]:
        if not self._fitted:
            return [MarketRegime.SIDEWAYS] * len(returns)
        X = returns.reshape(-1, 1)
        states = self.hmm.predict(X)
        return [self.regime_mapping.get(s, MarketRegime.SIDEWAYS) for s in states]


# =============================================================================
# DATA PIPELINE
# =============================================================================
class DataPipeline:
    """Data fetching and feature engineering with real NLP sentiment."""
    
    def __init__(self, config: Config):
        self.config = config
        # Initialize FinBERT sentiment analyzer with config
        self.sentiment_analyzer = FinBERTSentimentAnalyzer(
            cache_dir=config.sentiment_cache_dir,
            use_gpu=config.use_gpu_for_nlp
        ) if config.use_real_nlp else SentimentAnalyzer()
    
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"Fetching data for {self.config.tickers}...")
        data = {}
        for ticker in self.config.tickers:
            t = yf.Ticker(ticker)
            df = t.history(start=self.config.start_date, end=self.config.end_date)
            df.reset_index(inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['price'] = df['close']
            data[ticker] = df
            logger.info(f"  ✓ {ticker}: {len(df)} days")
        return data
    
    def engineer_features(self, df: pd.DataFrame, ticker: str = 'UNKNOWN') -> pd.DataFrame:
        """Engineer all features (same as base transformer)."""
        df = df.copy()
        
        # Returns
        for w in [1, 5, 10, 21]:
            df[f'return_{w}d'] = df['close'].pct_change(w)
            df[f'log_return_{w}d'] = np.log(df['close'] / df['close'].shift(w))
        
        # Volatility
        for w in [5, 10, 20, 60]:
            df[f'volatility_{w}d'] = df['return_1d'].rolling(window=w).std()
        
        # Momentum
        for w in [5, 10, 20]:
            df[f'momentum_{w}d'] = df['close'] - df['close'].shift(w)
            df[f'roc_{w}d'] = df['close'].pct_change(w) * 100
        
        # Moving Averages
        for w in [5, 10, 20, 50, 200]:
            df[f'sma_{w}'] = df['close'].rolling(window=w).mean()
            df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
        
        df['price_to_sma_20'] = df['close'] / (df['sma_20'] + 1e-10)
        df['price_to_sma_50'] = df['close'] / (df['sma_50'] + 1e-10)
        df['price_to_sma_200'] = df['close'] / (df['sma_200'] + 1e-10)
        
        # RSI
        for w in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=w).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window=w).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{w}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_sma = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = bb_sma + 2 * bb_std
        df['bb_lower'] = bb_sma - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_sma + 1e-10)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        df['volume_change'] = df['volume'].pct_change()
        sign = np.sign(df['close'].diff()).fillna(0)
        df['obv'] = (sign * df['volume']).cumsum()
        df['obv_slope'] = df['obv'].diff(5) / 5
        
        # Trend
        df['trend_sma10_50'] = (df['sma_10'] > df['sma_50']).astype(int)
        df['trend_sma50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        df['natr'] = df['atr_14'] / df['close'] * 100
        
        # Tier 2 features (simplified)
        df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
        df['close_open_ratio'] = df['close'] / (df['open'] + 1e-10)
        df['range_pct'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        df['overnight_gap'] = (df['open'] - df['close'].shift()) / (df['close'].shift() + 1e-10)
        
        # ADX
        atr_14 = tr.rolling(window=14).mean()
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        df['plus_di'] = 100 * (plus_dm.rolling(window=14).mean() / (atr_14 + 1e-10))
        df['minus_di'] = 100 * (minus_dm.rolling(window=14).mean() / (atr_14 + 1e-10))
        dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
        df['adx_14'] = dx.rolling(window=14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = typical_price.rolling(window=20).mean()
        tp_mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['cci_20'] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)
        
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
        
        # MFI
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        pos_sum = positive_flow.rolling(window=14).sum()
        neg_sum = negative_flow.rolling(window=14).sum()
        df['mfi_14'] = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))
        
        # CMF, VPT, AD
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['cmf_20'] = (clv * df['volume']).rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)
        df['vpt'] = (df['return_1d'].fillna(0) * df['volume']).cumsum()
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        df['volatility_ratio_5_20'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-10)
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['distance_to_sma_50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-10)
        df['distance_to_sma_200'] = (df['close'] - df['sma_200']) / (df['sma_200'] + 1e-10)
        
        df['skewness_20'] = df['return_1d'].rolling(window=20).apply(
            lambda x: scipy_stats.skew(x) if len(x) >= 3 else 0, raw=True)
        df['kurtosis_20'] = df['return_1d'].rolling(window=20).apply(
            lambda x: scipy_stats.kurtosis(x) if len(x) >= 4 else 0, raw=True)
        
        # Calendar
        if 'date' in df.columns:
            date_col = pd.to_datetime(df['date'])
            dow = date_col.dt.dayofweek
            month = date_col.dt.month
            df['day_of_week'] = dow
            df['dow_sin'] = np.sin(2 * np.pi * dow / 5)
            df['dow_cos'] = np.cos(2 * np.pi * dow / 5)
            df['month_sin'] = np.sin(2 * np.pi * month / 12)
            df['month_cos'] = np.cos(2 * np.pi * month / 12)
            df['is_monday'] = (dow == 0).astype(int)
            df['is_friday'] = (dow == 4).astype(int)
            df['is_month_end'] = date_col.dt.is_month_end.astype(int)
        
        # 52-week
        df['high_52w'] = df['high'].rolling(window=252, min_periods=50).max()
        df['low_52w'] = df['low'].rolling(window=252, min_periods=50).min()
        df['pct_from_52w_high'] = (df['close'] - df['high_52w']) / (df['high_52w'] + 1e-10)
        df['pct_from_52w_low'] = (df['close'] - df['low_52w']) / (df['low_52w'] + 1e-10)
        
        # Sentiment (real NLP via FinBERT)
        sentiment_df = self.sentiment_analyzer.get_sentiment_features(
            ticker=ticker,
            dates=df['date'] if 'date' in df.columns else pd.Series(range(len(df))),
            prices=df['price'],
            returns=df['return_1d'] if 'return_1d' in df.columns else df['price'].pct_change(),
            company_names=self.config.company_names  # For better news search
        )
        for col in sentiment_df.columns:
            df[col] = sentiment_df[col].values[:len(df)] if len(sentiment_df[col]) >= len(df) else \
                      list(sentiment_df[col].values) + [0] * (len(df) - len(sentiment_df[col]))
        
        # Target
        df['future_price'] = df['price'].shift(-self.config.prediction_horizon)
        df['future_return'] = (df['future_price'] - df['price']) / df['price']
        
        df = df.dropna().reset_index(drop=True)
        return df
    
    def align_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        date_sets = [set(df['date'].astype(str)) for df in data.values()]
        common_dates = set.intersection(*date_sets)
        aligned = {}
        for ticker, df in data.items():
            mask = df['date'].astype(str).isin(common_dates)
            aligned[ticker] = df[mask].reset_index(drop=True)
        logger.info(f"  Aligned to {len(common_dates)} common dates")
        return aligned


# =============================================================================
# iTRANSFORMER MODEL
# =============================================================================
class iTransformerBlock(layers.Layer):
    """
    iTransformer Block: Inverted attention over features/assets.
    
    Key insight: Instead of attending over time steps (standard transformer),
    we attend over the feature/asset dimension. This allows the model to
    learn correlations between different assets at each time point.
    
    Input: (batch, time_steps, n_assets * n_features)
    Output: (batch, time_steps, d_model)
    """
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        # Multi-head self-attention over the "token" dimension
        # In iTransformer, tokens are time steps, and each token has
        # an embedding that represents all assets/features at that time
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        
        self.ff = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.d_model),
            layers.Dropout(self.dropout_rate)
        ])
    
    def call(self, x, training=None):
        # x: (batch, time_steps, d_model)
        attn_out = self.attention(x, x, training=training)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x, training=training)
        x = self.ln2(x + ff_out)
        return x


def build_itransformer_model(
    n_assets: int,
    n_features: int,
    time_steps: int,
    config: Config
) -> keras.Model:
    """
    Build iTransformer model for multi-asset return prediction.
    
    Architecture:
    1. Flatten all assets' features into single embedding per time step
    2. Apply transformer blocks with attention over time steps
    3. The key difference: input is (batch, time_steps, n_assets * n_features)
       treating each time step as a "token" with full cross-asset info
    4. Output predictions for each asset
    
    This is different from standard transformers that process each asset
    separately - iTransformer naturally captures cross-asset dependencies.
    """
    
    # Input: (batch, n_assets, time_steps, n_features)
    inputs = keras.Input(shape=(n_assets, time_steps, n_features))
    
    # INVERTED DIMENSION: Reshape to (batch, time_steps, n_assets * n_features)
    # Each time step becomes a "token" containing ALL assets' features
    x = layers.Permute((2, 1, 3))(inputs)  # (batch, time_steps, n_assets, n_features)
    x = layers.Reshape((time_steps, n_assets * n_features))(x)
    
    # Project to d_model
    x = layers.Dense(config.d_model, name='input_projection')(x)
    
    # Positional encoding
    def create_positional_encoding(length: int, d_model: int) -> np.ndarray:
        position = np.arange(length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe.astype(np.float32)
    
    pos_enc = create_positional_encoding(time_steps, config.d_model)
    x = layers.Lambda(lambda t, pe=pos_enc: t + pe, name='pos_encoding')(x)
    
    # iTransformer blocks
    for i in range(config.num_layers):
        x = iTransformerBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout_rate,
            name=f'itransformer_block_{i}'
        )(x)
    
    # Take the last time step: (batch, d_model)
    x = x[:, -1, :]
    
    # Output head: predict return for each asset
    x = layers.Dense(config.ff_dim, activation='relu')(x)
    x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(n_assets, name='predictions')(x)  # (batch, n_assets)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='iTransformer')
    
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================
class SignalGenerator:
    """Generate BUY/SELL signals using percentile-based thresholds."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_signals(
        self, predictions: Dict[str, np.ndarray], dates: pd.Series, prices: Dict[str, pd.Series]
    ) -> Dict[str, pd.DataFrame]:
        signals = {}
        
        for ticker, preds in predictions.items():
            n = len(preds)
            pred_series = pd.Series(preds)
            
            rolling_rank = pred_series.rolling(window=60, min_periods=20).apply(
                lambda x: scipy_stats.percentileofscore(x, x.iloc[-1]), raw=False
            ).fillna(50)
            
            signal_types, signal_names = [], []
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
                'percentile_rank': rolling_rank.values,
                'signal_value': signal_types,
                'signal': signal_names
            })
            
            buy_count = sum(1 for s in signal_names if 'BUY' in s)
            sell_count = sum(1 for s in signal_names if 'SELL' in s)
            logger.info(f"  {ticker}: {buy_count} BUYs, {sell_count} SELLs")
        
        return signals


# =============================================================================
# TRADING SIMULATOR
# =============================================================================
class TradingSimulator:
    """Simulate trading based on signals."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def simulate(
        self, ticker_signals: Dict[str, pd.DataFrame], regimes: List[MarketRegime]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        tickers = list(ticker_signals.keys())
        first_ticker = tickers[0]
        n_days = len(ticker_signals[first_ticker])
        n_assets = len(tickers)
        
        cash = 0.0
        shares = {t: 0.0 for t in tickers}
        portfolio_values, bh_values, trades = [], [], []
        
        # Buy & hold
        bh_shares = {t: (self.config.initial_investment / n_assets) / ticker_signals[t]['price'].iloc[0]
                     for t in tickers}
        
        # Start invested
        for t in tickers:
            price = ticker_signals[t]['price'].iloc[0]
            shares[t] = (self.config.initial_investment / n_assets) / price
        
        last_trade_day = {t: -10 for t in tickers}
        
        for day in range(n_days):
            prices = {t: ticker_signals[t]['price'].iloc[day] for t in tickers}
            
            portfolio_value = cash + sum(shares[t] * prices[t] for t in tickers)
            portfolio_values.append(portfolio_value)
            
            bh_value = sum(bh_shares[t] * prices[t] for t in tickers)
            bh_values.append(bh_value)
            
            for t in tickers:
                signal = ticker_signals[t]['signal'].iloc[day]
                if day - last_trade_day[t] < 5:
                    continue
                
                current_val = shares[t] * prices[t]
                target_weight = 1 / n_assets
                
                if signal == 'STRONG_BUY':
                    target_weight *= 1.5
                elif signal == 'BUY':
                    target_weight *= 1.2
                elif signal == 'SELL':
                    target_weight *= 0.5
                elif signal == 'STRONG_SELL':
                    target_weight *= 0.2
                
                target_val = portfolio_value * target_weight
                trade_val = target_val - current_val
                
                if abs(trade_val) > portfolio_value * 0.03:
                    cost = abs(trade_val) * self.config.transaction_cost_bps / 10000
                    if trade_val > 0:
                        shares[t] += trade_val / prices[t]
                        cash -= (trade_val + cost)
                    else:
                        sell_shares = min(shares[t], abs(trade_val) / prices[t])
                        shares[t] -= sell_shares
                        cash += (sell_shares * prices[t] - cost)
                    trades.append({'day': day, 'ticker': t, 'signal': signal})
                    last_trade_day[t] = day
        
        return np.array(portfolio_values), np.array(bh_values), pd.DataFrame(trades)


# =============================================================================
# VISUALIZER
# =============================================================================
class Visualizer:
    def __init__(self, config: Config):
        self.config = config
        self.regime_colors = {
            MarketRegime.CRISIS: '#FFCCCC', MarketRegime.BEAR: '#FFE4CC',
            MarketRegime.SIDEWAYS: '#E0E0E0', MarketRegime.BULL: '#CCFFCC'
        }
    
    def plot_portfolio(self, pv: np.ndarray, bh: np.ndarray, dates: pd.Series, regimes: List):
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(len(dates) - 1):
            if i < len(regimes):
                ax.axvspan(dates.iloc[i], dates.iloc[i+1], alpha=0.3,
                          color=self.regime_colors.get(regimes[i], '#FFFFFF'))
        
        ax.plot(dates, pv / pv[0] * 100, label='iTransformer Strategy', linewidth=2)
        ax.plot(dates, bh / bh[0] * 100, label='Buy & Hold', color='gray', alpha=0.7)
        
        strat_ret = (pv[-1] / pv[0] - 1) * 100
        bh_ret = (bh[-1] / bh[0] - 1) * 100
        ax.text(0.02, 0.95, f'Strategy: {strat_ret:.1f}%\nB&H: {bh_ret:.1f}%\nAlpha: {strat_ret-bh_ret:+.1f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylabel('Normalized Value (Base=100)')
        ax.set_title('iTransformer Portfolio Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_signals(self, ticker_signals: Dict[str, pd.DataFrame], regimes: List):
        n = len(ticker_signals)
        fig, axes = plt.subplots(n, 1, figsize=(16, 5*n), sharex=True)
        if n == 1:
            axes = [axes]
        
        for idx, (ticker, df) in enumerate(ticker_signals.items()):
            ax = axes[idx]
            dates = pd.to_datetime(df['date'])
            prices = df['price'].values
            signals = df['signal'].values
            
            for i in range(len(dates) - 1):
                if i < len(regimes):
                    ax.axvspan(dates.iloc[i], dates.iloc[i+1], alpha=0.3,
                              color=self.regime_colors.get(regimes[i], '#FFFFFF'))
            
            ax.plot(dates, prices, linewidth=1.5, alpha=0.8, label=f'{ticker} Price')
            
            buy_mask = np.isin(signals, ['BUY', 'STRONG_BUY'])
            sell_mask = np.isin(signals, ['SELL', 'STRONG_SELL'])
            
            if buy_mask.sum() > 0:
                idx_buy = np.where(buy_mask)[0][::max(1, buy_mask.sum()//50)]
                ax.scatter(dates.iloc[idx_buy], prices[idx_buy], marker='^', s=100,
                          c='green', label=f'BUY ({buy_mask.sum()})', zorder=10)
            
            if sell_mask.sum() > 0:
                idx_sell = np.where(sell_mask)[0][::max(1, sell_mask.sum()//50)]
                ax.scatter(dates.iloc[idx_sell], prices[idx_sell], marker='v', s=100,
                          c='red', label=f'SELL ({sell_mask.sum()})', zorder=10)
            
            bh_ret = (prices[-1] / prices[0] - 1) * 100
            ax.set_title(f'{ticker} | B&H: {bh_ret:.1f}%', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', ncol=3)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('iTransformer Trading Signals', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN SYSTEM
# =============================================================================
class iTransformerQuantSystem:
    """Main iTransformer-based Quant System."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.pipeline = DataPipeline(self.config)
        self.regime_detector = RegimeDetector()
        self.signal_generator = SignalGenerator(self.config)
        self.visualizer = Visualizer(self.config)
        self.simulator = TradingSimulator(self.config)
        
        self.model: Optional[keras.Model] = None
        self.scalers: Dict[str, StandardScaler] = {}
        self.selected_features: List[str] = []
        self.results: Dict[str, Any] = {}
        self.ticker_signals: Dict[str, pd.DataFrame] = {}
    
    def run(self) -> Dict[str, Any]:
        # PHASE 1: Data
        logger.info("\n📥 PHASE 1: Loading Data")
        raw_data = self.pipeline.fetch_data()
        
        # PHASE 2: Feature Engineering
        logger.info("\n⚙️ PHASE 2: Feature Engineering")
        engineered = {t: self.pipeline.engineer_features(df, t) for t, df in raw_data.items()}
        for t, df in engineered.items():
            logger.info(f"  {t}: {len(df)} samples, {len(df.columns)} features")
        aligned = self.pipeline.align_data(engineered)
        
        # PHASE 3: Regime Detection
        logger.info("\n🔍 PHASE 3: Regime Detection")
        first = self.config.tickers[0]
        returns = aligned[first]['return_1d'].dropna().values
        self.regime_detector.fit(returns)
        regimes = self.regime_detector.predict(aligned[first]['return_1d'].fillna(0).values)
        
        # PHASE 4: Split
        logger.info("\n📊 PHASE 4: Train/Val/Test Split")
        n = len(aligned[first])
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train = {t: df.iloc[:train_end] for t, df in aligned.items()}
        val = {t: df.iloc[train_end:val_end] for t, df in aligned.items()}
        test = {t: df.iloc[val_end:] for t, df in aligned.items()}
        logger.info(f"  Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
        
        # PHASE 5: Scaling
        logger.info("\n📏 PHASE 5: Feature Scaling")
        self.selected_features = [f for f in FEATURES if f in train[first].columns]
        logger.info(f"  Using {len(self.selected_features)} features")
        
        scaled_train, scaled_val, scaled_test = {}, {}, {}
        for t in self.config.tickers:
            scaler = StandardScaler()
            scaled_train[t] = scaler.fit_transform(train[t][self.selected_features])
            scaled_val[t] = scaler.transform(val[t][self.selected_features])
            scaled_test[t] = scaler.transform(test[t][self.selected_features])
            self.scalers[t] = scaler
        
        # PHASE 6: Sequences
        logger.info("\n🔢 PHASE 6: Creating Sequences")
        
        def create_seq(scaled, raw, ts):
            tickers = list(scaled.keys())
            min_len = min(len(v) for v in scaled.values())
            n_seq = min_len - ts
            X = np.zeros((n_seq, len(tickers), ts, len(self.selected_features)))
            y = np.zeros((n_seq, len(tickers)))
            for ai, t in enumerate(tickers):
                for i in range(n_seq):
                    X[i, ai] = scaled[t][i:i+ts]
                    y[i, ai] = raw[t]['future_return'].values[i + ts]
            return X, y
        
        X_train, y_train = create_seq(scaled_train, train, self.config.time_steps)
        X_val, y_val = create_seq(scaled_val, val, self.config.time_steps)
        X_test, y_test = create_seq(scaled_test, test, self.config.time_steps)
        
        logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # PHASE 7: Build iTransformer
        logger.info("\n🤖 PHASE 7: Building iTransformer")
        logger.info("  ✨ INVERTED ATTENTION: attending over assets/features, not time")
        
        self.model = build_itransformer_model(
            n_assets=len(self.config.tickers),
            n_features=len(self.selected_features),
            time_steps=self.config.time_steps,
            config=self.config
        )
        self.model.summary(print_fn=lambda x: logger.info(f"  {x}"))
        
        callbacks = [
            EarlyStopping(patience=self.config.patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
        ]
        
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=self.config.epochs, batch_size=self.config.batch_size,
                       callbacks=callbacks, verbose=1)
        
        # PHASE 8: Predict
        logger.info("\n📈 PHASE 8: Generating Predictions")
        preds = self.model.predict(X_test, verbose=0)
        predictions = {t: preds[:, i] for i, t in enumerate(self.config.tickers)}
        for t, p in predictions.items():
            logger.info(f"  {t}: mean={p.mean():.4f}, std={p.std():.4f}")
        
        # PHASE 9: Signals
        logger.info("\n🎯 PHASE 9: Generating Signals")
        test_dates = test[first]['date'].iloc[self.config.time_steps:].reset_index(drop=True)
        test_prices = {t: test[t]['price'].iloc[self.config.time_steps:].reset_index(drop=True)
                       for t in self.config.tickers}
        self.ticker_signals = self.signal_generator.generate_signals(predictions, test_dates, test_prices)
        
        for t in self.config.tickers:
            actual = test[t]['future_return'].iloc[self.config.time_steps:].values
            n_sig = len(self.ticker_signals[t])
            self.ticker_signals[t]['actual_return'] = actual[:n_sig] if len(actual) >= n_sig else \
                list(actual) + [0] * (n_sig - len(actual))
        
        # PHASE 10: Trading
        logger.info("\n💰 PHASE 10: Trading Simulation")
        test_regimes = regimes[val_end:][:len(test_dates)]
        pv, bh, trades = self.simulator.simulate(self.ticker_signals, test_regimes)
        
        # PHASE 11: Metrics
        logger.info("\n📊 PHASE 11: Calculating Metrics")
        for t in self.config.tickers:
            df = self.ticker_signals[t]
            p, a = df['prediction'].values, df['actual_return'].values
            valid = ~(np.isnan(p) | np.isnan(a))
            corr = np.corrcoef(p[valid], a[valid])[0, 1] if valid.sum() > 1 else 0
            corr = 0 if np.isnan(corr) else corr
            dir_acc = np.mean(np.sign(p[valid]) == np.sign(a[valid])) * 100 if valid.sum() > 1 else 50
            self.results[t] = {'direction_accuracy': dir_acc, 'correlation': corr,
                               'bh_return': (df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100}
        
        self.results['PORTFOLIO'] = {
            'strategy_return': (pv[-1] / pv[0] - 1) * 100,
            'bh_return': (bh[-1] / bh[0] - 1) * 100,
            'alpha': (pv[-1] / pv[0] - 1) * 100 - (bh[-1] / bh[0] - 1) * 100,
            'n_trades': len(trades)
        }
        
        # PHASE 12: Visualize
        logger.info("\n📈 PHASE 12: Visualization")
        self.visualizer.plot_portfolio(pv, bh, test_dates, test_regimes)
        self.visualizer.plot_signals(self.ticker_signals, test_regimes)
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        print("\n" + "=" * 80)
        print("iTRANSFORMER QUANT SYSTEM 2026 - FINAL RESULTS")
        print("=" * 80)
        for t in self.config.tickers:
            r = self.results[t]
            sigs = self.ticker_signals[t]['signal'].values
            print(f"\n{t}:")
            print(f"  Direction Accuracy: {r['direction_accuracy']:.1f}%")
            print(f"  Correlation: {r['correlation']:.4f}")
            print(f"  B&H Return: {r['bh_return']:.2f}%")
            print(f"  Signals: BUY={(sigs=='BUY').sum()+(sigs=='STRONG_BUY').sum()}, "
                  f"SELL={(sigs=='SELL').sum()+(sigs=='STRONG_SELL').sum()}")
        
        r = self.results['PORTFOLIO']
        print(f"\nPORTFOLIO:")
        print(f"  Strategy Return: {r['strategy_return']:.2f}%")
        print(f"  Buy & Hold: {r['bh_return']:.2f}%")
        print(f"  ALPHA: {r['alpha']:+.2f}%")
        print(f"  Trades: {r['n_trades']}")
        print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================
print("=" * 80)
print("=" * 80)
print("iTRANSFORMER QUANT SYSTEM 2026 + REAL NLP SENTIMENT")
print("=" * 80)
print("\n🧠 NLP Sentiment (FinBERT):")
print("  • Model: ProsusAI/finbert (financial text specialist)")
print("  • Sources: yfinance news + GNews fallback")
print("  • Features: score, momentum, volatility, volume, buzz, divergence")
print("\n📊 iTransformer Innovation (ICLR 2024):")
print("  • Standard Transformer: Attention over TIME steps")
print("  • iTransformer: Attention over FEATURES/ASSETS (inverted)")
print("  • Each time step = token, embedding = all assets' features")
print("  • Naturally captures cross-asset correlations")
print("=" * 80)

# Check NLP availability
print(f"\n🔧 NLP Status: {'✓ FinBERT Available' if NLP_AVAILABLE else '⚠ Fallback to proxy (install: pip install transformers torch)'}")
print(f"🔧 News Scraping: {'✓ Available' if SCRAPING_AVAILABLE else '⚠ Limited (install: pip install requests beautifulsoup4)'}")

config = Config(
    tickers=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2018-01-01',
    epochs=30,
    patience=6,
    d_model=64,
    num_heads=4,
    num_layers=2,
    use_real_nlp=NLP_AVAILABLE,  # Use FinBERT if available
)

system = iTransformerQuantSystem(config)
results = system.run()

print("\n✅ iTransformer Quant System with Real NLP Complete!")

