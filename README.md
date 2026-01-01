# Transformer Quant Systems dec_2025

Advanced transformer-based quantitative trading systems combining cutting-edge ML architectures with financial domain expertise.

## 🏗️ Architecture Evolution

| File | Architecture | Key Innovation |
|------|--------------|----------------|
| `transformer_quant_system_dec_2025.py` | Base Transformer | Cross-asset attention, purged splits |
| `leadlag_transformer_dec_2025.py` | + Lead-Lag Signatures | Path signatures + lead-lag bias injection |
| `sit_quant_system_dec_2025.py` | Signature-Informed Transformer | Full SIT with CVaR loss |
| `itransformer_quant_dec_2025.py` | iTransformer | Inverted attention (over features/assets) |
| `itransformer_nlp_dec_2025.py` | iTransformer + NLP | + FinBERT real news sentiment |
| `scaleformer_itransformer_nlp_dec_2025.py` | Scaleformer + iTransformer + NLP | Multi-scale refinement + inverted attention + NLP |
| `enhanced_quant_system_2025.py` | Full Enterprise System | MARS, Meta-Labeling, HRP, all features |

## 📊 Feature Engineering

- `comprehensive_feature_engineering.py` - 94+ technical/statistical features
- `COMPLETE_FEATURE_ENGINEERING.md` - Documentation of all features

## 📚 Research Documentation

- `CUTTING_EDGE_METHODS_2025.md` - Survey of cutting-edge ML methods for quant finance

## 🔬 Key Innovations

### 1. Scaleformer (arXiv:2206.04038)
- Multi-scale processing (coarse → fine temporal patterns)
- Shared weights across scales
- Iterative refinement
- +5-10% improvement over single-scale transformers

### 2. iTransformer (ICLR 2024, arXiv:2310.06625)
- Inverted attention: over features/assets, not time steps
- Each time step = token with all assets' features
- Naturally captures cross-asset correlations

### 3. Signature-Informed Transformer (arXiv:2510.03129)
- Path signatures from rough path theory
- Lead-lag bias injection in attention
- CVaR loss for risk-aware training

### 4. FinBERT NLP Sentiment
- Real financial news analysis via ProsusAI/finbert
- News from yfinance + GNews fallback
- 6 sentiment features per ticker

## 🚀 Quick Start

```bash
pip install tensorflow yfinance hmmlearn transformers torch requests beautifulsoup4
```

Run any system in Google Colab or locally:
```python
# Example: Scaleformer + iTransformer + NLP
exec(open('scaleformer_itransformer_nlp_dec_2025.py').read())
```

## 📈 Output

Each system generates:
- Per-ticker BUY/SELL signals with confidence
- Portfolio performance vs Buy & Hold
- Alpha calculation
- Signal visualizations

## 📄 License

MIT

## 🔗 References

- [Scaleformer Paper](https://arxiv.org/abs/2206.04038)
- [iTransformer Paper](https://arxiv.org/abs/2310.06625)
- [SIT Paper](https://arxiv.org/abs/2510.03129)
- [FinBERT](https://arxiv.org/abs/1908.10063)

