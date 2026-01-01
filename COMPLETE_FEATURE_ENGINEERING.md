# COMPLETE FEATURE ENGINEERING FOR TRADING SIGNALS
## Exhaustive List of ALL Possible Features (December 2025)

---

## CURRENTLY IMPLEMENTED IN YOUR CODE (12 features)

```python
FEATURES = [
    'price', 'Close_Return', 'Volatility', 'Momentum', 'RSI',
    'MA_Ratio', 'Volume_Change', 'Trend', 'MACD', 'Signal', 'MACD_Hist', 'OBV'
]
```

---

## COMPLETE FEATURE TAXONOMY (200+ features)

---

# CATEGORY 1: PRICE-BASED FEATURES (Raw & Derived)

## 1.1 Raw Price Data
| Feature | Formula/Description | Window | Rationale |
|---------|---------------------|--------|-----------|
| `open` | Raw open price | - | Opening auction price |
| `high` | Raw high price | - | Intraday maximum |
| `low` | Raw low price | - | Intraday minimum |
| `close` | Raw close price | - | Closing auction price |
| `adj_close` | Split/dividend adjusted close | - | True economic value |
| `typical_price` | (H + L + C) / 3 | - | Average price |
| `weighted_close` | (H + L + 2*C) / 4 | - | Close-weighted avg |
| `median_price` | (H + L) / 2 | - | Midpoint price |

## 1.2 Returns (Multi-Horizon)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `return_1d` | (P_t - P_{t-1}) / P_{t-1} | Daily momentum |
| `return_5d` | (P_t - P_{t-5}) / P_{t-5} | Weekly momentum |
| `return_10d` | (P_t - P_{t-10}) / P_{t-10} | Bi-weekly momentum |
| `return_21d` | (P_t - P_{t-21}) / P_{t-21} | Monthly momentum |
| `return_63d` | (P_t - P_{t-63}) / P_{t-63} | Quarterly momentum |
| `return_252d` | (P_t - P_{t-252}) / P_{t-252} | Annual momentum |
| `log_return_1d` | ln(P_t / P_{t-1}) | Log-scale momentum |
| `log_return_5d` | ln(P_t / P_{t-5}) | Log weekly |
| `cumulative_return` | Product of (1 + r_i) - 1 | Compounded growth |

## 1.3 Price Ratios
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `high_low_ratio` | high / low | Intraday range |
| `close_open_ratio` | close / open | Daily direction |
| `upper_shadow` | (high - max(open,close)) / close | Rejection signal |
| `lower_shadow` | (min(open,close) - low) / close | Support signal |
| `body_size` | abs(close - open) / close | Conviction |
| `range_pct` | (high - low) / close | Daily volatility |

## 1.4 Gap Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `overnight_gap` | (open_t - close_{t-1}) / close_{t-1} | Overnight news reaction |
| `gap_filled` | 1 if gap closed during day | Gap trading signal |
| `gap_direction` | sign(overnight_gap) | Gap sentiment |

---

# CATEGORY 2: MOVING AVERAGES & TREND INDICATORS

## 2.1 Simple Moving Averages
| Feature | Window | Rationale |
|---------|--------|-----------|
| `sma_5` | 5 | Very short-term trend |
| `sma_10` | 10 | Short-term trend |
| `sma_20` | 20 | Monthly trend |
| `sma_50` | 50 | Intermediate trend |
| `sma_100` | 100 | Medium trend |
| `sma_200` | 200 | Long-term trend |

## 2.2 Exponential Moving Averages
| Feature | Span | Rationale |
|---------|------|-----------|
| `ema_5` | 5 | Fast reaction |
| `ema_12` | 12 | MACD fast component |
| `ema_20` | 20 | Standard EMA |
| `ema_26` | 26 | MACD slow component |
| `ema_50` | 50 | Trend following |
| `ema_200` | 200 | Major trend |

## 2.3 Other Moving Averages
| Feature | Type | Rationale |
|---------|------|-----------|
| `wma_20` | Weighted (linear weights) | Recent emphasis |
| `dema_20` | Double EMA | Reduced lag |
| `tema_20` | Triple EMA | Further lag reduction |
| `kama_20` | Kaufman Adaptive | Volatility-adjusted |
| `hull_ma_20` | Hull Moving Average | Minimal lag |
| `vwma_20` | Volume-Weighted MA | Volume-confirmed |

## 2.4 MA-Based Signals
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `price_to_sma_5` | close / sma_5 | Short-term deviation |
| `price_to_sma_20` | close / sma_20 | Monthly deviation |
| `price_to_sma_50` | close / sma_50 | Intermediate deviation |
| `price_to_sma_200` | close / sma_200 | Long-term deviation |
| `sma_5_10_cross` | 1 if sma_5 > sma_10 else 0 | Short crossover |
| `sma_10_50_cross` | 1 if sma_10 > sma_50 else 0 | Golden cross signal |
| `sma_50_200_cross` | 1 if sma_50 > sma_200 else 0 | Major trend signal |
| `days_above_sma_50` | Consecutive days close > sma_50 | Trend strength |
| `distance_to_sma_200` | (close - sma_200) / sma_200 | Mean reversion potential |

---

# CATEGORY 3: MOMENTUM INDICATORS

## 3.1 Rate of Change Family
| Feature | Formula/Window | Rationale |
|---------|----------------|-----------|
| `roc_5` | (P_t - P_{t-5}) / P_{t-5} * 100 | 1-week momentum |
| `roc_10` | (P_t - P_{t-10}) / P_{t-10} * 100 | 2-week momentum |
| `roc_20` | (P_t - P_{t-20}) / P_{t-20} * 100 | Monthly momentum |
| `roc_60` | (P_t - P_{t-60}) / P_{t-60} * 100 | Quarterly momentum |
| `momentum_5` | P_t - P_{t-5} | Raw price momentum |
| `momentum_10` | P_t - P_{t-10} | 10-day momentum |
| `momentum_20` | P_t - P_{t-20} | Monthly momentum |

## 3.2 RSI Family
| Feature | Window | Rationale |
|---------|--------|-----------|
| `rsi_7` | 7 | Fast RSI |
| `rsi_14` | 14 | Standard RSI |
| `rsi_21` | 21 | Slow RSI |
| `rsi_divergence` | RSI trend vs price trend | Divergence signal |
| `rsi_slope` | d(RSI)/dt over 5 days | RSI momentum |
| `rsi_overbought` | 1 if RSI > 70 | Overbought signal |
| `rsi_oversold` | 1 if RSI < 30 | Oversold signal |
| `stoch_rsi` | Stochastic of RSI | RSI normalization |

## 3.3 Stochastic Oscillator
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `stoch_k` | (C - L14) / (H14 - L14) * 100 | Fast stochastic |
| `stoch_d` | SMA(stoch_k, 3) | Slow stochastic |
| `stoch_j` | 3*K - 2*D | Stochastic acceleration |
| `stoch_cross` | 1 if K crosses above D | Buy signal |

## 3.4 MACD Family
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `macd` | EMA_12 - EMA_26 | MACD line |
| `macd_signal` | EMA_9(MACD) | Signal line |
| `macd_hist` | MACD - Signal | MACD histogram |
| `macd_hist_slope` | d(histogram)/dt | Momentum of momentum |
| `macd_cross` | 1 if MACD > Signal | Bullish cross |
| `macd_zero_cross` | 1 if MACD > 0 | Bullish territory |
| `ppo` | (EMA_12 - EMA_26) / EMA_26 * 100 | Percentage Price Oscillator |
| `ppo_signal` | EMA_9(PPO) | PPO signal |

## 3.5 Other Momentum Indicators
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `cci_14` | Commodity Channel Index | Trend & overbought/sold |
| `cci_20` | CCI 20-day | Slower CCI |
| `williams_r` | Williams %R | Momentum oscillator |
| `uo` | Ultimate Oscillator | Multi-timeframe |
| `tsi` | True Strength Index | Double-smoothed |
| `ao` | Awesome Oscillator | 5/34 SMA momentum |
| `kst` | Know Sure Thing | Weighted ROC |
| `cmo` | Chande Momentum Oscillator | Trend strength |
| `rvi` | Relative Vigor Index | Open-close trend |

---

# CATEGORY 4: VOLATILITY INDICATORS

## 4.1 Historical Volatility
| Feature | Window | Rationale |
|---------|--------|-----------|
| `volatility_5` | 5-day std(returns) | Very short-term vol |
| `volatility_10` | 10-day std(returns) | Short-term vol |
| `volatility_20` | 20-day std(returns) | Monthly vol |
| `volatility_60` | 60-day std(returns) | Quarterly vol |
| `volatility_252` | 252-day std(returns) | Annual vol |
| `volatility_ratio` | vol_5 / vol_20 | Volatility change |
| `realized_vol` | Annualized std(returns) | Standard measure |
| `parkinson_vol` | Based on high-low range | Range-based estimator |
| `garman_klass_vol` | Uses OHLC | More efficient estimator |
| `yang_zhang_vol` | Combines overnight + intraday | Best OHLC estimator |

## 4.2 Bollinger Bands
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `bb_upper` | SMA_20 + 2*std_20 | Upper band |
| `bb_middle` | SMA_20 | Middle band |
| `bb_lower` | SMA_20 - 2*std_20 | Lower band |
| `bb_width` | (upper - lower) / middle | Volatility measure |
| `bb_pct` | (close - lower) / (upper - lower) | Position in bands |
| `bb_squeeze` | 1 if width < threshold | Low vol signal |
| `bb_breakout_up` | 1 if close > upper | Breakout signal |
| `bb_breakout_down` | 1 if close < lower | Breakdown signal |

## 4.3 Average True Range
| Feature | Window | Rationale |
|---------|--------|-----------|
| `atr_7` | 7-day ATR | Fast ATR |
| `atr_14` | 14-day ATR | Standard ATR |
| `atr_21` | 21-day ATR | Monthly ATR |
| `natr` | ATR / close * 100 | Normalized ATR |
| `atr_ratio` | atr_7 / atr_21 | Volatility expansion |
| `atr_percentile` | ATR percentile rank | Historical context |

## 4.4 Other Volatility Measures
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `keltner_upper` | EMA_20 + 2*ATR_10 | Volatility channel |
| `keltner_lower` | EMA_20 - 2*ATR_10 | Volatility channel |
| `donchian_upper` | Max(high, 20) | Breakout level |
| `donchian_lower` | Min(low, 20) | Breakdown level |
| `donchian_mid` | (upper + lower) / 2 | Centerline |
| `chaikin_volatility` | Volatility based on H-L range | Alternative vol |
| `ulcer_index` | Downside volatility | Risk measure |

---

# CATEGORY 5: VOLUME INDICATORS

## 5.1 Raw Volume Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `volume` | Raw volume | Activity level |
| `volume_sma_5` | SMA(volume, 5) | Short-term avg |
| `volume_sma_20` | SMA(volume, 20) | Monthly avg |
| `volume_ratio` | volume / volume_sma_20 | Relative volume |
| `volume_change` | (V_t - V_{t-1}) / V_{t-1} | Volume momentum |
| `volume_zscore` | (V - mean) / std | Volume anomaly |

## 5.2 Price-Volume Indicators
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `obv` | On-Balance Volume | Cumulative buying pressure |
| `obv_slope` | d(OBV)/dt | OBV momentum |
| `vpt` | Volume Price Trend | Price-weighted volume |
| `mfi_14` | Money Flow Index | Volume-weighted RSI |
| `ad_line` | Accumulation/Distribution | Buying/selling pressure |
| `ad_oscillator` | Chaikin A/D Oscillator | A/D momentum |
| `cmf_20` | Chaikin Money Flow | Money flow over period |
| `emv` | Ease of Movement | Price/volume relationship |
| `fi` | Force Index | Price × Volume momentum |
| `nvi` | Negative Volume Index | Down-day accumulation |
| `pvi` | Positive Volume Index | Up-day accumulation |
| `vwap` | Volume Weighted Avg Price | Fair value |

## 5.3 Volume Pattern Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `volume_price_confirm` | sign(return) == sign(vol_change) | Confirmation |
| `climax_volume` | 1 if volume > 2.5*sma_20 | Exhaustion signal |
| `dry_up_volume` | 1 if volume < 0.5*sma_20 | Low interest |
| `volume_streak` | Consecutive up-volume days | Buying persistence |

---

# CATEGORY 6: TREND INDICATORS

## 6.1 ADX Family
| Feature | Window | Rationale |
|---------|--------|-----------|
| `adx_14` | 14-day ADX | Trend strength |
| `adx_20` | 20-day ADX | Slower ADX |
| `plus_di` | +DI | Bullish direction |
| `minus_di` | -DI | Bearish direction |
| `di_cross` | 1 if +DI > -DI | Bullish signal |
| `adx_rising` | 1 if ADX increasing | Strengthening trend |
| `dx` | Directional Index | Base DX |

## 6.2 Ichimoku Cloud
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `tenkan_sen` | 9-period midpoint | Conversion line |
| `kijun_sen` | 26-period midpoint | Base line |
| `senkou_span_a` | (Tenkan + Kijun) / 2 | Leading span A |
| `senkou_span_b` | 52-period midpoint | Leading span B |
| `chikou_span` | Close shifted back 26 | Lagging span |
| `tk_cross` | Tenkan crosses Kijun | Trend signal |
| `price_vs_cloud` | Price position vs cloud | Trend confirmation |
| `cloud_thickness` | Span A - Span B | Trend strength |

## 6.3 Other Trend Indicators
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `psar` | Parabolic SAR | Trend & stops |
| `psar_direction` | 1 if price > PSAR | Trend direction |
| `aroon_up` | Days since highest high | Uptrend strength |
| `aroon_down` | Days since lowest low | Downtrend strength |
| `aroon_oscillator` | aroon_up - aroon_down | Net trend |
| `vortex_plus` | +VI | Positive vortex |
| `vortex_minus` | -VI | Negative vortex |
| `supertrend` | ATR-based trend | Trend with trailing stop |
| `mass_index` | Sum of EMA(H-L) range | Reversal signal |
| `dpo` | Detrended Price Oscillator | Cycle identification |

---

# CATEGORY 7: STATISTICAL FEATURES

## 7.1 Distribution Statistics
| Feature | Window | Rationale |
|---------|--------|-----------|
| `skewness_20` | Skewness of returns | Asymmetry |
| `skewness_60` | 60-day skewness | Longer-term asymmetry |
| `kurtosis_20` | Kurtosis of returns | Tail risk |
| `kurtosis_60` | 60-day kurtosis | Longer-term tails |
| `jarque_bera` | Normality test stat | Distribution shape |

## 7.2 Autocorrelation Features
| Feature | Lag | Rationale |
|---------|-----|-----------|
| `autocorr_1` | Lag-1 autocorrelation | Short-term persistence |
| `autocorr_5` | Lag-5 autocorrelation | Weekly persistence |
| `autocorr_10` | Lag-10 autocorrelation | Bi-weekly persistence |
| `partial_autocorr_1` | Partial autocorr | Direct effect |

## 7.3 Complexity & Entropy
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `hurst_exponent` | Hurst exponent (R/S analysis) | Mean reversion vs trending |
| `sample_entropy` | Sample entropy | Predictability |
| `approx_entropy` | Approximate entropy | Regularity |
| `permutation_entropy` | Permutation entropy | Ordinal patterns |
| `fractal_dimension` | Box-counting dimension | Market complexity |

## 7.4 Rolling Quantile Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `return_percentile` | Percentile rank of today's return | Historical context |
| `price_percentile_52w` | 52-week price percentile | Long-term position |
| `volume_percentile` | Volume percentile rank | Activity context |
| `rolling_max_drawdown` | Maximum drawdown over window | Risk measure |
| `days_from_52w_high` | Days since 52-week high | Relative strength |
| `pct_from_52w_high` | % below 52-week high | Pullback magnitude |
| `pct_from_52w_low` | % above 52-week low | Recovery magnitude |

---

# CATEGORY 8: CANDLESTICK PATTERNS (Encoded as Binary Features)

## 8.1 Single Candle Patterns
| Feature | Pattern | Signal |
|---------|---------|--------|
| `doji` | Small body, any size | Indecision |
| `hammer` | Small body, long lower shadow | Bullish reversal |
| `inverted_hammer` | Small body, long upper shadow | Bullish reversal |
| `hanging_man` | Hammer at top | Bearish reversal |
| `shooting_star` | Inv hammer at top | Bearish reversal |
| `spinning_top` | Small body, both shadows | Indecision |
| `marubozu` | No shadows | Strong conviction |

## 8.2 Two Candle Patterns
| Feature | Pattern | Signal |
|---------|---------|--------|
| `bullish_engulfing` | Green engulfs red | Bullish reversal |
| `bearish_engulfing` | Red engulfs green | Bearish reversal |
| `piercing_line` | Green opens lower, closes > 50% | Bullish |
| `dark_cloud_cover` | Red opens higher, closes < 50% | Bearish |
| `harami_bullish` | Small green inside red | Bullish |
| `harami_bearish` | Small red inside green | Bearish |
| `tweezer_top` | Two similar highs | Bearish |
| `tweezer_bottom` | Two similar lows | Bullish |

## 8.3 Three+ Candle Patterns
| Feature | Pattern | Signal |
|---------|---------|--------|
| `morning_star` | Down, doji, up | Bullish reversal |
| `evening_star` | Up, doji, down | Bearish reversal |
| `three_white_soldiers` | Three strong green | Bullish continuation |
| `three_black_crows` | Three strong red | Bearish continuation |
| `three_inside_up` | Harami + confirmation | Bullish |
| `three_inside_down` | Harami + confirmation | Bearish |

---

# CATEGORY 9: CHART PATTERNS (Encoded Features)

## 9.1 Reversal Patterns
| Feature | Detection Method | Signal |
|---------|------------------|--------|
| `head_shoulders` | Peak detection algorithm | Major reversal |
| `inv_head_shoulders` | Trough detection | Major bullish |
| `double_top` | Two similar peaks | Bearish reversal |
| `double_bottom` | Two similar troughs | Bullish reversal |
| `triple_top` | Three similar peaks | Bearish |
| `triple_bottom` | Three similar troughs | Bullish |

## 9.2 Continuation Patterns
| Feature | Detection Method | Signal |
|---------|------------------|--------|
| `ascending_triangle` | Higher lows + flat resistance | Bullish |
| `descending_triangle` | Lower highs + flat support | Bearish |
| `symmetric_triangle` | Converging trendlines | Breakout pending |
| `bull_flag` | Strong up + consolidation | Bullish continuation |
| `bear_flag` | Strong down + consolidation | Bearish continuation |
| `wedge_rising` | Rising converging lines | Bearish |
| `wedge_falling` | Falling converging lines | Bullish |

---

# CATEGORY 10: TIME-BASED FEATURES

## 10.1 Calendar Features
| Feature | Encoding | Rationale |
|---------|----------|-----------|
| `day_of_week` | 0-4 (Mon-Fri) | Weekly patterns |
| `day_of_week_sin` | sin(2π * dow / 5) | Cyclical encoding |
| `day_of_week_cos` | cos(2π * dow / 5) | Cyclical encoding |
| `day_of_month` | 1-31 | Monthly patterns |
| `day_of_month_sin` | sin(2π * dom / 31) | Cyclical encoding |
| `week_of_year` | 1-52 | Seasonal patterns |
| `month` | 1-12 | Monthly seasonality |
| `month_sin` | sin(2π * m / 12) | Cyclical encoding |
| `month_cos` | cos(2π * m / 12) | Cyclical encoding |
| `quarter` | 1-4 | Quarterly patterns |
| `is_quarter_end` | Binary | Rebalancing effects |
| `is_year_end` | Binary | Tax/window dressing |
| `is_month_start` | Binary | Flow effects |
| `is_month_end` | Binary | Rebalancing effects |
| `is_friday` | Binary | Weekend effect |
| `is_monday` | Binary | Monday effect |

## 10.2 Special Day Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `days_to_holiday` | Days until next market holiday | Holiday effect |
| `days_from_holiday` | Days since last holiday | Post-holiday effect |
| `is_opex_week` | Options expiration week | Volatility |
| `is_fomc_week` | Fed meeting week | Policy impact |
| `days_to_earnings` | Days to next earnings | Event premium |
| `days_from_earnings` | Days since earnings | Post-earnings drift |
| `is_witching_day` | Triple/quad witching | Expiration effects |

---

# CATEGORY 11: CROSS-ASSET FEATURES (Multi-Ticker)

## 11.1 Relative Strength Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `rs_vs_spy` | cumulative_return / SPY_return | Market-relative strength |
| `rs_vs_sector` | return / sector_etf_return | Sector-relative |
| `rank_in_sector` | Percentile rank in sector | Relative position |
| `rank_momentum_30d` | 30-day momentum rank | Cross-sectional momentum |
| `z_score_cross_section` | (return - mean) / std | Relative performance |

## 11.2 Correlation Features
| Feature | Window | Rationale |
|---------|--------|-----------|
| `corr_with_spy_20` | 20-day correlation with SPY | Market beta proxy |
| `corr_with_spy_60` | 60-day correlation | Longer-term beta |
| `corr_with_vix` | Correlation with VIX | Fear relationship |
| `avg_pairwise_corr` | Average correlation with others | Diversification |
| `corr_change` | Δ correlation over time | Regime change signal |

## 11.3 Cointegration & Lead-Lag
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `spread_from_cointegrated` | Deviation from cointegrated pair | Mean reversion |
| `lead_lag_indicator` | Lead/lag detection | Predictive relationship |
| `granger_causality` | Granger test p-value | Predictability |

## 11.4 Market Structure Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `advance_decline_ratio` | # advancers / # decliners | Market breadth |
| `new_highs_minus_lows` | 52w highs - 52w lows | Market strength |
| `pct_above_200dma` | % of stocks above 200 DMA | Market health |
| `sector_rotation_score` | Relative sector momentum | Rotation signal |

---

# CATEGORY 12: MARKET REGIME FEATURES

## 12.1 Volatility Regime
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `vix_level` | Current VIX | Fear gauge |
| `vix_percentile` | VIX percentile rank | Historical context |
| `vix_term_structure` | VIX1M / VIX3M | Contango/backwardation |
| `vix_change_5d` | 5-day VIX change | Fear momentum |
| `realized_vs_implied_vol` | RV / IV | Vol risk premium |

## 12.2 Economic Regime
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `yield_curve_slope` | 10Y - 2Y Treasury | Recession indicator |
| `credit_spread` | HY - IG spread | Risk appetite |
| `ted_spread` | 3M LIBOR - T-bill | Banking stress |
| `ism_manufacturing` | ISM PMI | Economic activity |

## 12.3 Detected Regime Features
| Feature | Method | Rationale |
|---------|--------|-----------|
| `hmm_regime` | HMM-detected regime | Market state |
| `regime_prob_bull` | P(bull regime) | Regime confidence |
| `regime_prob_bear` | P(bear regime) | Regime confidence |
| `regime_transition_prob` | P(switch to new regime) | Transition likelihood |
| `regime_duration` | Days in current regime | Persistence |

---

# CATEGORY 13: FUNDAMENTAL FEATURES

## 13.1 Valuation Ratios
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `pe_ratio` | Price / EPS | Valuation |
| `forward_pe` | Price / Forward EPS | Expected valuation |
| `peg_ratio` | PE / Growth | Growth-adjusted value |
| `pb_ratio` | Price / Book | Asset valuation |
| `ps_ratio` | Price / Sales | Revenue valuation |
| `pcf_ratio` | Price / Cash Flow | Cash valuation |
| `ev_ebitda` | EV / EBITDA | Enterprise value |
| `fcf_yield` | FCF / Market Cap | Cash return |
| `dividend_yield` | Div / Price | Income return |
| `earnings_yield` | EPS / Price | Earnings return |

## 13.2 Quality Metrics
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `roe` | Return on Equity | Profitability |
| `roa` | Return on Assets | Asset efficiency |
| `roic` | Return on Invested Capital | Capital efficiency |
| `gross_margin` | Gross Profit / Revenue | Pricing power |
| `operating_margin` | Operating Income / Revenue | Operational efficiency |
| `net_margin` | Net Income / Revenue | Overall profitability |
| `asset_turnover` | Revenue / Assets | Asset utilization |

## 13.3 Financial Health
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `debt_to_equity` | Total Debt / Equity | Leverage |
| `debt_to_ebitda` | Debt / EBITDA | Debt capacity |
| `interest_coverage` | EBIT / Interest Expense | Debt safety |
| `current_ratio` | Current Assets / Current Liabilities | Liquidity |
| `quick_ratio` | (CA - Inventory) / CL | Quick liquidity |
| `altman_z_score` | Bankruptcy predictor | Financial distress |
| `piotroski_f_score` | 9-point quality score | Value quality |

## 13.4 Growth Metrics
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `revenue_growth_yoy` | YoY revenue change | Growth |
| `eps_growth_yoy` | YoY EPS change | Earnings growth |
| `eps_surprise` | Actual EPS - Expected | Earnings beat/miss |
| `revenue_surprise` | Actual Rev - Expected | Revenue beat/miss |
| `guidance_revision` | Δ guidance | Forward expectations |
| `estimate_revisions` | # analyst upgrades - downgrades | Sentiment |

---

# CATEGORY 14: SENTIMENT FEATURES

## 14.1 Analyst Sentiment
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `analyst_rating_mean` | Avg rating (1-5) | Consensus view |
| `analyst_target_upside` | (Target - Price) / Price | Expected return |
| `rating_upgrades_30d` | # upgrades last 30d | Positive momentum |
| `rating_downgrades_30d` | # downgrades last 30d | Negative momentum |
| `estimate_eps_change_30d` | Δ EPS estimates | Earnings momentum |

## 14.2 News Sentiment
| Feature | Source | Rationale |
|---------|--------|-----------|
| `news_sentiment_score` | NLP on news headlines | Current sentiment |
| `news_sentiment_5d_avg` | 5-day avg sentiment | Trend sentiment |
| `news_volume` | # news articles | Attention |
| `news_sentiment_momentum` | Δ sentiment | Sentiment change |
| `earnings_call_sentiment` | Transcript NLP | Management tone |

## 14.3 Social Media Sentiment
| Feature | Source | Rationale |
|---------|--------|-----------|
| `twitter_sentiment` | Twitter/X analysis | Retail sentiment |
| `reddit_wsb_mentions` | r/wallstreetbets | Retail interest |
| `stocktwits_sentiment` | StockTwits analysis | Trader sentiment |
| `social_volume_change` | Δ mentions | Attention spike |
| `google_trends_score` | Google Trends | Search interest |

## 14.4 Positioning Sentiment
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `short_interest_pct` | Shares short / Float | Bearish bets |
| `short_interest_ratio` | Days to cover | Short squeeze risk |
| `short_interest_change` | Δ short interest | Sentiment shift |
| `institutional_ownership` | % held by institutions | Smart money |
| `inst_ownership_change` | Δ institutional holdings | Smart money flow |
| `insider_buy_sell_ratio` | Insider buys / sells | Insider confidence |
| `put_call_ratio` | Put volume / Call volume | Options sentiment |

---

# CATEGORY 15: ALTERNATIVE DATA FEATURES

## 15.1 Web/App Data
| Feature | Source | Rationale |
|---------|--------|-----------|
| `web_traffic_change` | SimilarWeb/Sensor Tower | Customer interest |
| `app_downloads` | App store data | Product adoption |
| `app_ratings_change` | Δ app ratings | Product quality |
| `job_postings_change` | LinkedIn/Indeed | Growth signal |

## 15.2 Satellite/Geolocation
| Feature | Source | Rationale |
|---------|--------|-----------|
| `store_foot_traffic` | Placer.ai, SafeGraph | Retail sales proxy |
| `parking_lot_fullness` | Satellite imagery | Store traffic |
| `shipping_activity` | AIS vessel data | Trade volume |
| `oil_storage_levels` | Satellite on tanks | Commodity supply |

## 15.3 Transaction Data
| Feature | Source | Rationale |
|---------|--------|-----------|
| `credit_card_spending` | Aggregated CC data | Consumer spending |
| `same_store_sales_proxy` | Transaction panel | Revenue leading |
| `consumer_spending_yoy` | YoY spending change | Demand signal |

## 15.4 Supply Chain
| Feature | Source | Rationale |
|---------|--------|-----------|
| `supplier_lead_time` | Supply chain data | Inventory stress |
| `inventory_days` | Days of inventory | Working capital |
| `shipping_rates` | Baltic Dry Index | Trade activity |

---

# CATEGORY 16: DERIVATIVES FEATURES

## 16.1 Options Metrics
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `iv_atm_30d` | 30-day ATM implied volatility | Expected vol |
| `iv_percentile` | IV rank in last 252 days | IV context |
| `iv_vs_hv` | IV / HV ratio | Vol risk premium |
| `iv_term_structure` | 30d IV / 90d IV | Vol term structure |
| `iv_skew` | 25δ put IV - 25δ call IV | Tail risk premium |
| `put_call_oi_ratio` | Put OI / Call OI | Positioning |
| `max_pain` | Options max pain price | Pinning target |
| `gamma_exposure` | Dealer gamma | Support/resistance |

## 16.2 Options Flow
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `call_volume_unusual` | Call vol / avg call vol | Bullish flow |
| `put_volume_unusual` | Put vol / avg put vol | Bearish flow |
| `large_call_orders` | # large call trades | Smart money calls |
| `large_put_orders` | # large put trades | Smart money puts |
| `options_sweep_ratio` | Sweeps / Block trades | Urgency signal |

---

# CATEGORY 17: MICROSTRUCTURE FEATURES

## 17.1 Liquidity Measures
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `bid_ask_spread` | Ask - Bid | Transaction cost |
| `relative_spread` | Spread / Mid | Normalized cost |
| `depth_at_bbo` | Volume at best bid/ask | Immediate liquidity |
| `total_depth_5levels` | Sum of 5 levels each side | Market depth |
| `amihud_illiquidity` | |Return| / Volume | Price impact |
| `kyle_lambda` | Price impact coefficient | Market quality |

## 17.2 Order Flow Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `order_flow_imbalance` | Buy volume - Sell volume | Net pressure |
| `ofi_normalized` | OFI / Total volume | Relative pressure |
| `vpin` | Volume-synced PIN | Informed trading |
| `trade_sign` | +1 buy, -1 sell (Lee-Ready) | Trade classification |
| `aggressive_ratio` | Market orders / Limit orders | Urgency |

## 17.3 Trade Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `avg_trade_size` | Total volume / # trades | Institutional vs retail |
| `trade_count` | Number of trades | Activity intensity |
| `block_trade_pct` | Large trades / Total | Institutional activity |
| `vwap_vs_close` | VWAP / Close - 1 | Execution quality |
| `twap_vs_close` | TWAP / Close - 1 | Time distribution |

---

# CATEGORY 18: MACHINE LEARNING-DERIVED FEATURES

## 18.1 Dimensionality Reduction
| Feature | Method | Rationale |
|---------|--------|-----------|
| `pca_1` | 1st principal component | Main variance |
| `pca_2` | 2nd principal component | Secondary variance |
| `pca_3` | 3rd principal component | Tertiary variance |
| `autoencoder_latent_1` | AE embedding dim 1 | Learned representation |
| `umap_1` | UMAP dim 1 | Non-linear structure |
| `umap_2` | UMAP dim 2 | Non-linear structure |

## 18.2 Clustering Features
| Feature | Method | Rationale |
|---------|--------|-----------|
| `cluster_id` | K-means cluster assignment | Market regime |
| `cluster_distance` | Distance to cluster center | Regime stability |
| `isolation_score` | Isolation Forest score | Anomaly detection |

## 18.3 Learned Signals
| Feature | Method | Rationale |
|---------|--------|-----------|
| `ensemble_prediction` | Blend of multiple models | Consensus signal |
| `model_uncertainty` | Prediction variance | Confidence |
| `feature_importance_weighted` | SHAP-weighted features | Important drivers |

---

# CATEGORY 19: GRAPH/NETWORK FEATURES

## 19.1 Stock Network Centrality
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `degree_centrality` | # of strong correlations | Connectivity |
| `betweenness_centrality` | Bridge position | Information flow |
| `pagerank` | PageRank in correlation network | Systemic importance |
| `clustering_coefficient` | Local clustering | Sector cohesion |

## 19.2 Sector/Industry Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `sector_momentum` | Sector ETF return | Sector trend |
| `sector_relative_strength` | Stock vs sector return | Outperformance |
| `industry_momentum` | Industry group return | Industry trend |
| `supply_chain_momentum` | Supplier/customer returns | Value chain signal |

---

# CATEGORY 20: PATH SIGNATURE FEATURES

## 20.1 Signature-Based Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `signature_level_1` | First level signature | Linear path |
| `signature_level_2` | Second level signature | Quadratic variation |
| `signature_level_3` | Third level signature | Higher-order dynamics |
| `log_signature` | Log-signature transform | Efficient signature |
| `rough_path_volatility` | Rough volatility estimate | Better vol model |

---

# CATEGORY 21: SPECTRAL FEATURES

## 21.1 Fourier Transform Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `fft_dominant_freq` | Dominant frequency | Cycle length |
| `fft_power_ratio` | Power in low vs high freq | Trend vs noise |
| `spectral_entropy` | Entropy of spectrum | Signal complexity |

## 21.2 Wavelet Features
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `wavelet_approx` | Approximation coefficients | Trend component |
| `wavelet_detail_1` | Detail level 1 | High-freq noise |
| `wavelet_detail_2` | Detail level 2 | Mid-freq cycles |
| `wavelet_energy_ratio` | Energy in different bands | Scale importance |

## 21.3 Empirical Mode Decomposition
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `imf_1` | Intrinsic Mode Function 1 | Fastest cycle |
| `imf_2` | IMF 2 | Second cycle |
| `imf_residual` | Residual trend | Underlying trend |

---

# CATEGORY 22: TARGET ENGINEERING (Label Features)

## 22.1 Return-Based Targets
| Feature | Horizon | Use Case |
|---------|---------|----------|
| `target_return_1d` | 1-day forward | Day trading |
| `target_return_5d` | 5-day forward | Swing trading |
| `target_return_21d` | 21-day forward | Monthly |
| `target_direction_5d` | Binary up/down | Classification |
| `target_magnitude_5d` | Abs(return) | Volatility target |

## 22.2 Risk-Adjusted Targets
| Feature | Description | Use Case |
|---------|-------------|----------|
| `target_sharpe_5d` | 5-day Sharpe-like | Risk-adjusted |
| `target_sortino_5d` | Downside-adjusted | Downside focus |
| `triple_barrier_label` | -1, 0, +1 (De Prado) | ML labeling |

---

# IMPLEMENTATION NOTES

## Feature Selection Priorities

### Tier 1 (Essential - Already Implemented + High Priority Additions)
1. All price returns (multi-horizon)
2. Volatility (multi-window)
3. RSI, MACD, Bollinger Bands
4. OBV, Volume ratios
5. SMA/EMA ratios

### Tier 2 (High Value - Add First)
1. ADX and Directional Indicators
2. ATR and normalized volatility
3. Stochastic oscillator
4. Cross-sectional relative strength
5. VIX and market regime proxies
6. Calendar features (day of week, month)
7. Skewness and kurtosis

### Tier 3 (Moderate Value)
1. Candlestick patterns
2. Additional momentum (CCI, Williams %R)
3. Ichimoku components
4. Options-derived features (if data available)
5. Correlation features

### Tier 4 (Advanced/Alternative)
1. Path signatures
2. Spectral/wavelet features
3. Microstructure (if HF data available)
4. Alternative data
5. Fundamental data

---

## Anti-Lookahead Checklist

✓ All features use ONLY past and present data
✓ Future returns are ONLY used as targets
✓ Rolling windows exclude current bar when appropriate
✓ No point-in-time bias in fundamental data
✓ Proper purging gap in train/val/test splits

---

## Implementation Code Follows Below

