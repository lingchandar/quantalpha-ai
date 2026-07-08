# QuantAlpha AI — AI-Powered Investment Research Platform for Indian Equities

An end-to-end quantitative research pipeline for the Nifty 200 universe, built solo, combining
self-computed fundamental analysis, technical indicators, NLP-based news classification, and
rigorously validated statistical signals — with an explicit focus on **testing claims honestly**
rather than overselling accuracy.

> **Core philosophy:** every signal in this project was tested with proper time-split validation
> before being trusted. Several intuitive signals (technical timing, sector momentum) were tested
> and found to have **no real edge** — that's reported here as a finding, not hidden. One signal
> (fundamental quality investing) showed a genuine, if modest, statistical edge. This honesty
> about what works and what doesn't is the actual point of the project.

---

## What this project does

QuantAlpha AI scores all ~200 Nifty 200 stocks across multiple engines and produces explainable
BUY / WATCH / AVOID recommendations with confidence scores, entry zones, targets, and stop-loss
levels — modeled on how professional quant research platforms operate.

```
Raw NSE Data → Feature Engineering → Multi-Engine Scoring → Validated Signal Selection
    → Explainable AI Output (Recommendation + Confidence + Reasons + Risk Levels)
```

## Key Finding (the headline result)

| Signal tested | Method | Result |
|---|---|---|
| Technical timing (RSI, MACD, ADX, Supertrend, breakouts, pullbacks) | Rule-based, backtested across 5 market regimes (2016–2026, including COVID crash & 2022 correction) | **No real edge.** Actively harmful during crashes (-6.7 pts vs baseline) |
| Sector momentum | Properly time-split test (trailing momentum → forward return, no look-ahead) | **No real edge** (+0.5 pts, non-monotonic) |
| ML model (XGBoost) — absolute return target | Time-split train/test (2016–2023 train, 2024–2026 test) | **No edge.** AUC 0.509 (≈ random) |
| ML model (XGBoost) — relative return + sentiment + momentum features | Same rigorous time-split | **No edge.** AUC 0.509 (identical — confirms ceiling) |
| **Fundamental quality (Piotroski F-Score + ROCE)** | 6-month and 12-month forward returns | **Real edge: +3.26 pts (6mo), +5.28 pts (12mo)** win rate vs. rest of universe |

**Conclusion:** short-term stock-price prediction (technical or ML-based) shows no exploitable
edge on this universe/data — tested 5 independent ways. Longer-horizon fundamental quality
investing shows a real, modest, broad-based edge (57% win rate across 30 stocks, consistent
across sectors). The platform is built around this honest finding.

**Real-world, cost-adjusted check (Step 9):** simulating an actual diversified 15-stock portfolio
of "high quality" picks, with realistic Indian trading costs (STT, exchange charges, GST, stamp
duty, DP charges, slippage — ~0.52% round-trip) subtracted:

| Horizon | Portfolio (net of costs) | Rest of universe (net) | Net edge |
|---|---|---|---|
| 6-month | 3.10% | 0.98% | +2.12 pts |
| 12-month | 9.44% | 4.66% | +4.77 pts |

Transaction costs barely erode the edge at these holding periods (confirms costs matter far more
for short-term/Swing trading, which independently showed no edge anyway).

---

## Architecture

```
Data Layer (yfinance, NSE, Google News RSS)
        │
        ├── Technical Engine (self-computed via pandas-ta: RSI, MACD, ADX, Supertrend, ATR, Bollinger, VWAP, OBV)
        ├── Fundamental Engine (self-computed: Piotroski F-Score, Altman Z-Score, ROCE, growth rates)
        ├── Cash Flow Engine (OCF/FCF margins, computed from raw statements)
        ├── Sentiment/Event Engine (FinBERT sentiment + BART-MNLI zero-shot event classification)
        ├── Sector Engine (sector mapping + momentum — tested, not used in final scoring)
        │
        └── Ensemble Scoring (mode-aware: Swing / Position / Long-Term weights)
                │
                └── Explainable AI Layer (confidence-adjusted BUY/WATCH/AVOID + ATR-based entry/target/stop-loss)
```

## What's validated vs. experimental

**✅ Validated (backed by rigorous, time-split backtesting):**
- Fundamental quality scoring (F-Score, ROCE, cash flow margins) for **Long-Term mode (12mo+)**
  and the longer end of **Position mode (5-6mo)**
- Self-computed technical indicator library (correctness verified, though standalone predictive
  edge was not found)
- Event classification pipeline (FinBERT + zero-shot) — correctly distinguishes isolated/temporary
  news events from structural ones in spot-checks

**⚠️ Experimental / not recommended for real decisions yet:**
- **Swing Trading mode** — tested extensively (rule-based across 5 market regimes + 2 ML model
  variants), consistently shows no edge. Kept in the codebase for completeness, clearly flagged.
- **Sector-adjusted scoring** — tested with proper time-split methodology, showed no edge;
  intentionally excluded from the final scoring formula rather than forced in.
- Short-horizon Position mode (1-3 months) — behaves like Swing mode, no edge found.

**❌ Known limitations (explicit, not hidden):**
- Fundamental quality backtest uses **current** fundamental snapshots against **historical**
  price data — a look-ahead bias caveat. A fully rigorous version needs historical point-in-time
  quarterly fundamentals, which aren't freely available for Indian equities at scale — this is
  the single biggest documented next step.
- 10-year price history is used for technical/regime testing, but the fundamental-quality
  backtest sample is small (N=30 "high quality" stocks) — a real, broad-based effect, but not a
  large sample.
- No transaction cost / slippage modeling yet in reported returns.
- Not deployed as a live API — currently a set of validated research notebooks.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | yfinance, NSE/niftyindices.com constituent lists, Google News RSS (feedparser) |
| Technical Analysis | pandas-ta |
| NLP | Hugging Face Transformers — `ProsusAI/finbert` (sentiment), `facebook/bart-large-mnli` (zero-shot event classification) |
| ML | XGBoost, scikit-learn |
| Compute | Google Colab (Pro, T4 GPU) |
| Storage | Google Drive (Parquet + CSV) |

## Project Structure

```
data/                          # 3-year Nifty 200 dataset (OHLCV, fundamentals, technical, scores)
data_10yr/                     # 10-year extended dataset for multi-regime testing
notebooks/
  Step1_DataPipeline           # Data collection + validation layer
  Step2_ScoresAndEvents        # Piotroski F-Score, Altman Z-Score, ROCE + event classifier
  Step3_ScoringEngine          # Mode-aware ensemble scoring (initial version)
  Step3B_SignalComparison      # Technical signal variant testing
  Step3C_FundamentalBacktest   # The core validated finding (quality investing edge)
  Step4_SectorEngine           # Sector engine (found to have look-ahead bug, then no real edge)
  Step5_ExplainableAI          # Final recommendation engine with honest confidence scoring
  Step6_ExtendedHistory        # 10-year, multi-regime signal re-test
  Step7_XGBoost                # ML baseline (absolute return target)
  Step8_ImprovedModel          # ML with sentiment + momentum + relative-return target
  Step9_CostAdjustedBacktest   # Real-world, transaction-cost-adjusted portfolio backtest
```

## Sample Output

```
SYMBOL: MCX
Recommendation: BUY (Long-Term candidate)
Confidence: 88.8% (High)
Current Price: ₹2,814.00
Entry Zone: ₹2,763.46 - ₹2,864.54
Stop-Loss: ₹2,611.83
Target 1: ₹3,117.26  |  Target 2: ₹3,319.43

Reasons:
  - Strong Piotroski F-Score (9/9) — healthy profitability and efficiency trend
  - Strong ROCE (29.4%) — efficient use of capital
  - Revenue growing at 106.9% year-over-year
  - Healthy free cash flow margin (128.7%)
  - Sector: Financial Services

Note: this recommendation is based on a fundamental-quality signal that showed a real
backtested edge, but using current fundamentals against historical price data (a
limitation, not a guarantee). Treat as one input, not financial advice.
```

## Roadmap

- [ ] Source historical point-in-time fundamentals (quarterly) to remove the look-ahead caveat
- [ ] Extend fundamental backtest across a full bear/sideways market cycle
- [ ] Build FastAPI backend to serve recommendations
- [ ] Add transaction cost-adjusted return reporting
- [ ] Monthly fundamental snapshot capture (started manually, building real point-in-time history
      going forward)

## Disclaimer

This project is a research and educational platform, not financial advice. All recommendations
are outputs of a backtested statistical model with documented limitations (see above). Past
performance in backtests does not guarantee future results. Consult a licensed financial advisor
before making investment decisions.

## Author

Built solo by Lingchandar — B.E. Electronics and Communication Engineering (2026), as an
applied machine learning / quantitative research portfolio project.
