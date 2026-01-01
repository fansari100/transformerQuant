# Cutting-Edge Methods for Live Equity Trading Signals (December 2025)

## For Portfolio of Tickers: Buy/Sell Signal Generation

**Context**: Methods specifically for generating live trading signals for a basket of equities (like AAPL, GOOGL, MSFT), with focus on practical implementation beyond Neural SDEs, Transformers, and PPO/RL.

---

## 📊 Quick Reference: Method Selection

| Your Need | Best Method (2025) | Difficulty |
|-----------|-------------------|------------|
| Better than PPO for trading | **MARS** (Multi-Agent Ensemble) | Medium |
| Adaptive to market regimes | **Regime-Aware RL** | Medium |
| Reduce false signals | **Meta-Labeling** | Easy |
| When to trade (not just what) | **DeepAries** | Medium |
| Cross-asset dependencies | **Attention-Enhanced RL** | Medium |
| Ultra-fast signals | **Extreme Learning Machine** | Easy |
| Robust to crashes | **DARL** (Diffusion-Augmented) | Hard |
| Interpretable patterns | **Shapelet Framework** | Medium |

---

## 🚀 TIER 1: Direct Upgrades to Your Current Architecture

### 1. Meta-Labeling (Corrective AI) — **Easiest Win**

**What it is**: A secondary model that evaluates your Transformer's signals BEFORE trading. It answers: "Should I trust this signal?" and "How much should I bet?"

**Why it's perfect for your code**: 
- Your Transformer predicts returns → Meta-label filters false positives
- Separates "direction" from "sizing" decisions
- De Prado (your purged CV reference) invented this!

**How it works with your architecture**:
```
Your Transformer → pred_return → Meta-Label Model → (confidence, position_size)
                                         ↓
                              Only trade when confidence > threshold
                              Size position by confidence
```

**Implementation** (drop-in addition to your code):

```python
class MetaLabeler:
    """
    Secondary model that filters primary signals.
    Train on: "Did the primary signal result in a profitable trade?"
    """
    
    def __init__(self, primary_model):
        self.primary = primary_model
        self.meta_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=50  # Prevent overfitting
        )
        self.calibrator = CalibratedClassifierCV(self.meta_model, cv=5)
    
    def create_meta_features(self, X, predictions):
        """
        Features for meta-model (NO lookahead):
        - Primary prediction magnitude
        - Primary prediction volatility
        - Recent prediction accuracy (rolling)
        - Market regime indicators
        """
        features = pd.DataFrame({
            'pred_magnitude': np.abs(predictions),
            'pred_sign': np.sign(predictions),
            'pred_volatility': pd.Series(predictions).rolling(20).std(),
            'market_volatility': X['Volatility'].values,
            'trend_alignment': X['Trend'].values * np.sign(predictions),
            'rsi_extreme': ((X['RSI'] < 30) | (X['RSI'] > 70)).astype(int).values,
        })
        return features.fillna(0)
    
    def fit(self, X_train, y_train, predictions_train, actual_returns_train):
        """
        y_meta = 1 if prediction direction matched actual AND was profitable
        """
        # Meta-label: was this trade profitable?
        pred_direction = np.sign(predictions_train)
        actual_direction = np.sign(actual_returns_train)
        profitable = (pred_direction == actual_direction) & (np.abs(actual_returns_train) > 0.001)
        
        meta_features = self.create_meta_features(X_train, predictions_train)
        self.calibrator.fit(meta_features, profitable.astype(int))
    
    def predict_with_confidence(self, X, predictions):
        """Returns (filtered_signal, confidence, position_size)"""
        meta_features = self.create_meta_features(X, predictions)
        
        # Calibrated probability = confidence
        confidence = self.calibrator.predict_proba(meta_features)[:, 1]
        
        # Only trade when confident
        filtered_signal = np.where(confidence > 0.6, predictions, 0)
        
        # Position size proportional to confidence
        position_size = np.clip(confidence - 0.5, 0, 0.5) * 2  # 0 to 1
        
        return filtered_signal, confidence, position_size


# Integration with your CitadelQuantSystemV3:
class EnhancedCitadelSystem(CitadelQuantSystemV3):
    def run_pipeline(self):
        # ... your existing code ...
        
        # After training transformer:
        train_preds = model.predict(X_train).flatten()
        
        # Add meta-labeling
        self.meta_labeler = MetaLabeler(model)
        self.meta_labeler.fit(
            df_train.iloc[self.config.time_steps:], 
            train_targets,
            train_preds,
            df_train['future_return'].iloc[self.config.time_steps:].values
        )
        
        # At prediction time:
        test_preds = model.predict(X_test).flatten()
        filtered_preds, confidence, sizing = self.meta_labeler.predict_with_confidence(
            df_test.iloc[self.config.time_steps:],
            test_preds
        )
```

---

### 2. MARS: Multi-Agent Risk-Aware RL — **Replace Single PPO**

**What it is**: Instead of one PPO agent, use an **ensemble of agents** with different risk profiles. A meta-controller decides which agent to follow based on market regime.

**Why it's better than your current PPO**:
- Your PPO: One policy for all market conditions
- MARS: Conservative agent for crashes, aggressive agent for bull markets
- Meta-controller switches between them dynamically

**Architecture**:
```
Market State → [Regime Detector] → regime ∈ {bull, bear, sideways, crisis}
                      ↓
              [Meta-Adaptive Controller]
                      ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
[Conservative    [Moderate      [Aggressive
   Agent]          Agent]         Agent]
(low risk tol)  (medium)       (high risk tol)
    ↓                 ↓                 ↓
    └─────────────────┼─────────────────┘
                      ↓
              Weighted Action Blend
```

**Implementation**:

```python
class SafetyCritic(nn.Module):
    """Estimates risk/tail-loss probability for an action"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability of large loss
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)  # P(loss > threshold)


class RiskAwareAgent:
    """Single agent with specific risk tolerance"""
    
    def __init__(self, risk_tolerance: float):
        self.risk_tol = risk_tolerance
        self.policy = PPO("MlpPolicy", env)
        self.safety_critic = SafetyCritic(state_dim, action_dim)
    
    def get_action(self, state):
        base_action, _ = self.policy.predict(state)
        
        # Constrain action by safety critic
        risk = self.safety_critic(state, base_action)
        if risk > self.risk_tol:
            # Scale down action to meet risk budget
            base_action *= (self.risk_tol / risk)
        
        return base_action


class RegimeDetector:
    """Detect market regime from recent data"""
    
    def __init__(self):
        self.hmm = GaussianHMM(n_components=4, covariance_type="full")
    
    def fit(self, returns):
        self.hmm.fit(returns.reshape(-1, 1))
    
    def predict_regime(self, recent_returns):
        state = self.hmm.predict(recent_returns.reshape(-1, 1))[-1]
        # Map to interpretable regimes based on state means
        means = self.hmm.means_.flatten()
        if means[state] > 0.001:
            return "bull"
        elif means[state] < -0.001:
            return "bear"
        else:
            return "sideways"


class MARS:
    """Multi-Agent Risk-aware System"""
    
    def __init__(self, env):
        # Ensemble of agents with different risk tolerances
        self.agents = {
            'conservative': RiskAwareAgent(risk_tolerance=0.1),
            'moderate': RiskAwareAgent(risk_tolerance=0.3),
            'aggressive': RiskAwareAgent(risk_tolerance=0.5),
        }
        self.regime_detector = RegimeDetector()
        
        # Meta-controller weights (learned)
        self.meta_weights = nn.Parameter(torch.ones(3) / 3)
    
    def get_action(self, state, recent_returns):
        regime = self.regime_detector.predict_regime(recent_returns)
        
        # Get actions from all agents
        actions = {
            name: agent.get_action(state) 
            for name, agent in self.agents.items()
        }
        
        # Regime-dependent weighting
        if regime == "bear" or regime == "crisis":
            weights = {'conservative': 0.7, 'moderate': 0.2, 'aggressive': 0.1}
        elif regime == "bull":
            weights = {'conservative': 0.1, 'moderate': 0.3, 'aggressive': 0.6}
        else:
            weights = {'conservative': 0.33, 'moderate': 0.34, 'aggressive': 0.33}
        
        # Blend actions
        final_action = sum(w * actions[name] for name, w in weights.items())
        return final_action, regime
```

---

### 3. DeepAries: Adaptive Rebalancing — **When to Trade**

**What it is**: Learns not just WHAT to trade, but WHEN to trade. Dynamically selects rebalancing intervals based on market conditions.

**Why it's better**:
- Your code: Fixed prediction horizon (5 days)
- DeepAries: "Should I trade now, or wait 3 more days?"
- Reduces transaction costs by avoiding unnecessary trades

**Key Innovation**: Outputs BOTH discrete (when) and continuous (how much) actions.

```python
class DeepAries:
    """Adaptive Rebalancing Interval Selection"""
    
    def __init__(self, tickers, max_interval=10):
        self.tickers = tickers
        self.max_interval = max_interval
        
        # Transformer encoder for market state
        self.encoder = TransformerEncoder(
            d_model=64, 
            nhead=4, 
            num_layers=3
        )
        
        # Discrete head: when to rebalance (1, 2, ..., max_interval days)
        self.interval_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, max_interval)  # Softmax over intervals
        )
        
        # Continuous head: portfolio weights
        self.allocation_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(tickers)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, market_history):
        """
        Returns: (rebalance_interval, portfolio_weights)
        """
        # Encode market state
        h = self.encoder(market_history)
        h = h.mean(dim=1)  # Pool over time
        
        # Discrete: when to rebalance
        interval_logits = self.interval_head(h)
        interval = torch.argmax(interval_logits, dim=-1) + 1
        
        # Continuous: how to allocate
        weights = self.allocation_head(h)
        
        return interval, weights
    
    def should_trade_today(self, days_since_last_trade, market_history):
        """Decision: trade now or wait?"""
        optimal_interval, weights = self.forward(market_history)
        
        if days_since_last_trade >= optimal_interval:
            return True, weights
        else:
            return False, None


# Integration with trading loop:
class AdaptiveTrader:
    def __init__(self, tickers):
        self.deeparies = DeepAries(tickers)
        self.days_since_trade = 0
        self.current_weights = None
    
    def daily_decision(self, market_history):
        should_trade, new_weights = self.deeparies.should_trade_today(
            self.days_since_trade, 
            market_history
        )
        
        if should_trade:
            self.execute_rebalance(new_weights)
            self.current_weights = new_weights
            self.days_since_trade = 0
        else:
            self.days_since_trade += 1
        
        return should_trade, self.current_weights
```

---

### 4. Attention-Enhanced Dirichlet RL — **Cross-Asset Dependencies**

**What it is**: Uses **cross-sectional attention** to model how AAPL, GOOGL, MSFT interact. Outputs **Dirichlet distribution** over weights (automatically sums to 1, always positive).

**Why it's better**:
- Your code: Separate transformer per ticker (no cross-asset learning)
- This: Joint model that learns "when AAPL drops, MSFT often follows"

**Key Innovation**: Dirichlet policy naturally satisfies portfolio constraints.

```python
class CrossAssetAttentionEncoder(nn.Module):
    """
    Captures cross-sectional dependencies between assets.
    Each asset has temporal encoder → then cross-asset attention.
    """
    
    def __init__(self, n_assets, d_model=64, n_heads=4):
        super().__init__()
        self.n_assets = n_assets
        
        # Per-asset temporal encoder (your existing transformer)
        self.temporal_encoders = nn.ModuleList([
            TransformerEncoder(d_model, n_heads, num_layers=2)
            for _ in range(n_assets)
        ])
        
        # Cross-asset attention (the key innovation)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads)
        
    def forward(self, asset_histories):
        """
        asset_histories: dict of {ticker: (batch, time_steps, features)}
        Returns: (batch, n_assets, d_model) - cross-asset aware embeddings
        """
        # Encode each asset's history
        asset_embeddings = []
        for i, (ticker, history) in enumerate(asset_histories.items()):
            h = self.temporal_encoders[i](history)
            h = h[:, -1, :]  # Last timestep embedding
            asset_embeddings.append(h)
        
        # Stack: (batch, n_assets, d_model)
        stacked = torch.stack(asset_embeddings, dim=1)
        
        # Cross-asset attention: each asset attends to others
        # Reshape for attention: (n_assets, batch, d_model)
        stacked_t = stacked.transpose(0, 1)
        cross_aware, _ = self.cross_attention(stacked_t, stacked_t, stacked_t)
        
        return cross_aware.transpose(0, 1)  # Back to (batch, n_assets, d_model)


class DirichletPolicy(nn.Module):
    """
    Outputs portfolio weights via Dirichlet distribution.
    Automatically satisfies: weights >= 0, sum(weights) = 1
    """
    
    def __init__(self, n_assets, d_model=64):
        super().__init__()
        self.n_assets = n_assets
        
        self.encoder = CrossAssetAttentionEncoder(n_assets, d_model)
        
        # Output Dirichlet concentration parameters (must be positive)
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, asset_histories):
        # Get cross-asset aware embeddings
        h = self.encoder(asset_histories)  # (batch, n_assets, d_model)
        
        # Concentration parameter per asset
        alphas = self.alpha_head(h).squeeze(-1)  # (batch, n_assets)
        alphas = alphas + 1.0  # Ensure alpha > 1 for unimodal
        
        return alphas
    
    def sample_weights(self, asset_histories):
        """Sample portfolio weights from Dirichlet"""
        alphas = self.forward(asset_histories)
        dist = torch.distributions.Dirichlet(alphas)
        weights = dist.sample()
        log_prob = dist.log_prob(weights)
        return weights, log_prob
    
    def expected_weights(self, asset_histories):
        """Expected value of Dirichlet = alpha / sum(alpha)"""
        alphas = self.forward(asset_histories)
        return alphas / alphas.sum(dim=-1, keepdim=True)


class AttentionEnhancedRL:
    """Full RL system with Dirichlet policy"""
    
    def __init__(self, tickers):
        self.policy = DirichletPolicy(len(tickers))
        self.value_net = ValueNetwork(len(tickers))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
    
    def compute_reward(self, weights, returns, prev_weights, tc_rate=0.001):
        """
        Reward = portfolio return - transaction costs - variance penalty
        """
        portfolio_return = (weights * returns).sum()
        turnover = torch.abs(weights - prev_weights).sum()
        transaction_cost = tc_rate * turnover
        variance_penalty = 0.5 * ((weights * returns) ** 2).sum()
        
        return portfolio_return - transaction_cost - variance_penalty
```

---

## 🔥 TIER 2: Advanced Methods (Higher Implementation Effort)

### 5. DARL: Diffusion-Augmented RL — **Crash Robustness**

**What it is**: Generates synthetic crash scenarios using diffusion models, then trains RL on these scenarios. Your agent learns to survive 2008-level crashes it's never seen.

**Implementation** (condensed):

```python
class MarketDiffusion(nn.Module):
    """Conditional diffusion for market scenario generation"""
    
    def __init__(self, n_assets, seq_len=252):
        super().__init__()
        self.denoiser = UNet1D(in_channels=n_assets, condition_dim=1)
        self.n_steps = 1000
    
    def generate_scenario(self, stress_intensity: float):
        """
        stress_intensity: 0.0 = normal, 1.0 = extreme crash
        Returns: (seq_len, n_assets) realistic market path
        """
        x = torch.randn(1, self.n_assets, self.seq_len)
        condition = torch.tensor([[stress_intensity]])
        
        for t in reversed(range(self.n_steps)):
            x = self.denoise_step(x, t, condition)
        
        return x.squeeze(0).T  # (seq_len, n_assets)


class DARLTrainer:
    """Train RL with diffusion-generated stress scenarios"""
    
    def __init__(self, tickers, historical_data):
        self.diffusion = MarketDiffusion(len(tickers))
        self.diffusion.train_on(historical_data)  # Learn market dynamics
        
        self.policy = DirichletPolicy(len(tickers))
    
    def train_robust(self, n_episodes=10000):
        for episode in range(n_episodes):
            # Curriculum: start easy, get harder
            progress = episode / n_episodes
            stress = np.random.beta(1 + 3*progress, 5 - 3*progress)
            
            # Generate scenario from diffusion
            scenario = self.diffusion.generate_scenario(stress)
            
            # Train on this scenario
            self.train_episode(scenario)
```

---

### 6. Regime-Aware RL — **Adapt to Market Conditions**

**What it is**: Detects market regime (bull/bear/crisis) and uses different reward functions for each.

```python
class RegimeAwareRL:
    """Different reward functions for different regimes"""
    
    def __init__(self, tickers):
        self.regime_hmm = GaussianHMM(n_components=4)
        self.policy = PPO("MlpPolicy", ...)
    
    def compute_reward(self, portfolio_return, volatility, regime):
        """Regime-dependent reward"""
        if regime == "crisis":
            # Survival mode: heavily penalize losses
            return portfolio_return - 5.0 * max(-portfolio_return, 0) - 2.0 * volatility
        elif regime == "bear":
            # Defensive: moderate loss penalty
            return portfolio_return - 2.0 * max(-portfolio_return, 0) - volatility
        elif regime == "bull":
            # Aggressive: reward gains, slight vol penalty
            return 1.5 * max(portfolio_return, 0) - 0.5 * volatility
        else:  # sideways
            # Balanced
            return portfolio_return - 0.5 * volatility
```

---

### 7. Shapelet-Based Signals — **Interpretable Patterns**

**What it is**: Extracts recurring patterns (shapelets) from price history, then classifies new data by pattern similarity. You can SEE what pattern triggered the signal.

**Why it's unique**:
- Transformer: Black box → "buy because..." ?
- Shapelets: "Buy because current pattern matches this historical breakout pattern"

```python
from tslearn.shapelets import LearningShapelets

class ShapeletSignalGenerator:
    """Interpretable pattern-based signals"""
    
    def __init__(self, n_shapelets=20, shapelet_length=10):
        self.model = LearningShapelets(
            n_shapelets_per_size={shapelet_length: n_shapelets},
            max_iter=500
        )
    
    def fit(self, X_train, y_train):
        """
        X_train: (n_samples, time_steps) - price patterns
        y_train: 1 = price went up after, 0 = price went down
        """
        self.model.fit(X_train, y_train)
        
        # Extract learned shapelets for interpretation
        self.shapelets = self.model.shapelets_
    
    def predict_with_explanation(self, X_new):
        """Returns prediction AND which shapelet matched"""
        proba = self.model.predict_proba(X_new)
        
        # Find which shapelet activated most
        distances = self.model.transform(X_new)
        best_match = np.argmin(distances, axis=1)
        
        return proba, best_match, self.shapelets[best_match]
```

---

### 8. Extreme Learning Machine — **Ultra-Fast Signals**

**What it is**: Single-layer network with random fixed hidden weights. Only output layer is trained analytically. **1000x faster** than deep networks.

**Use case**: Real-time signal generation, HFT-adjacent applications.

```python
class ExtremeLearnMachine:
    """Ultra-fast single-pass learning"""
    
    def __init__(self, input_dim, hidden_dim=500, output_dim=1):
        # Random fixed weights (never trained!)
        self.W_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_hidden = np.random.randn(hidden_dim)
        self.W_out = None
    
    def fit(self, X, y):
        """One-shot learning via pseudo-inverse (no iterations!)"""
        H = np.tanh(X @ self.W_hidden + self.b_hidden)
        self.W_out = np.linalg.pinv(H) @ y  # Analytical solution
    
    def predict(self, X):
        H = np.tanh(X @ self.W_hidden + self.b_hidden)
        return H @ self.W_out
    
    def incremental_update(self, X_new, y_new, forgetting_factor=0.99):
        """Online update for live trading"""
        H_new = np.tanh(X_new @ self.W_hidden + self.b_hidden)
        
        # Recursive least squares update
        # Much faster than retraining
        ...
```

---

## 📐 TIER 3: Architecture Improvements

### 9. Hierarchical Risk Parity (HRP) — **Better than Mean-Variance**

**What it is**: Builds portfolio using hierarchical clustering of asset correlations. No matrix inversion needed = more stable.

```python
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    """ML-based portfolio optimization"""
    
    def allocate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        returns: (n_samples, n_assets) DataFrame with ticker columns
        Returns: portfolio weights
        """
        cov = returns.cov()
        corr = returns.corr()
        
        # Step 1: Hierarchical clustering on correlation distance
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist), method='ward')
        order = leaves_list(link)
        
        # Step 2: Recursive bisection
        weights = self._recursive_bisection(cov, order)
        
        return weights
    
    def _recursive_bisection(self, cov, order):
        """Allocate by inverse variance down the hierarchy"""
        n = len(order)
        if n == 1:
            return np.array([1.0])
        
        # Split in half
        left = order[:n//2]
        right = order[n//2:]
        
        # Weights by inverse variance of each cluster
        var_left = self._cluster_variance(cov, left)
        var_right = self._cluster_variance(cov, right)
        
        alpha = 1 - var_left / (var_left + var_right)
        
        # Recursive
        w_left = self._recursive_bisection(cov, left) * alpha
        w_right = self._recursive_bisection(cov, right) * (1 - alpha)
        
        return np.concatenate([w_left, w_right])
```

---

## 🎯 Recommended Upgrade Path for Your Code

### Phase 1: Quick Wins (This Week)

1. **Add Meta-Labeling** to filter your Transformer signals
2. Replace naive thresholds with **confidence-based sizing**
3. Add **Hierarchical Risk Parity** for portfolio weights

### Phase 2: Core Upgrades (This Month)

4. Replace separate per-ticker Transformers with **Cross-Asset Attention**
5. Replace PPO with **MARS** multi-agent ensemble
6. Add **Regime Detection** to adapt strategy

### Phase 3: Advanced (Next Quarter)

7. Train **Diffusion model** on historical data for stress testing
8. Implement **DeepAries** for adaptive rebalancing timing
9. Add **Shapelet extraction** for interpretable signals

---

## 📊 Expected Improvements

| Upgrade | Expected Sharpe Improvement | Implementation Time |
|---------|---------------------------|-------------------|
| Meta-Labeling | +0.1 to +0.3 | 1-2 days |
| Cross-Asset Attention | +0.2 to +0.4 | 1 week |
| MARS Multi-Agent | +0.2 to +0.5 | 1-2 weeks |
| Regime Detection | +0.1 to +0.3 | 3-5 days |
| DARL Stress Training | +0.3 to +0.6 (in crashes) | 2-3 weeks |
| HRP Allocation | +0.1 to +0.2 | 1-2 days |

---

## Complete Upgraded Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          Market Data Stream             │
                    │     (AAPL, GOOGL, MSFT prices/vol)      │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Cross-Asset Attention Encoder                      │
│  (Replaces separate per-ticker Transformers)                         │
│  - Per-asset temporal encoding                                       │
│  - Cross-sectional attention for inter-asset dependencies           │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
            │ Return       │    │ Regime       │    │ Volatility   │
            │ Prediction   │    │ Detection    │    │ Forecast     │
            └──────────────┘    └──────────────┘    └──────────────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           Meta-Labeling Layer           │
                    │  - Filter low-confidence signals         │
                    │  - Compute position sizing               │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │         MARS Multi-Agent System          │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
                    │  │Conserv. │ │Moderate │ │Aggress. │    │
                    │  │ Agent   │ │ Agent   │ │ Agent   │    │
                    │  └─────────┘ └─────────┘ └─────────┘    │
                    │         Meta-Controller (regime-based)   │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │      DeepAries Rebalancing Timing        │
                    │  - Should I trade today or wait?         │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │    Hierarchical Risk Parity Weights      │
                    │  (Final portfolio allocation)            │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           BUY / SELL / HOLD              │
                    │      Signals for AAPL, GOOGL, MSFT       │
                    └─────────────────────────────────────────┘
```

---

## References

1. MARS (2025): arxiv.org/abs/2508.01173
2. DeepAries (2025): arxiv.org/abs/2510.14985  
3. Attention-Enhanced Dirichlet RL (2025): arxiv.org/abs/2510.06466
4. DARL (2025): arxiv.org/abs/2510.07099
5. Regime-Aware RL (2025): arxiv.org/abs/2509.14385
6. Shapelet Forecasting (2025): arxiv.org/abs/2509.15040
7. Large Execution Models (2025): arxiv.org/abs/2509.25211
8. Meta-Labeling: de Prado, "Advances in Financial ML" (2018)
9. HRP: de Prado, "Building Diversified Portfolios" (2016)

---

*Document updated: December 2025*
*Focus: Live buy/sell signals for equity portfolio*
