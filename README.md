# gates-engine

Fundamental-anchored stochastic price-scenario model with walk-forward backtesting.
Companion to the Five Gates framework / Watts-to-Tokens series.

## Usage

```bash
python3 engine.py EME                 # full run: fetch → fit → simulate → backtest → report
python3 engine.py WMB --no-backtest   # faster, skips walk-forward
python3 engine.py MU --horizon=5 --paths=8000 --bt-h=3
open reports/eme.html                 # styled report with fan chart + backtest
```

Stdlib-only (no pip installs). Data cached in `.cache/` (1 day).

## Model

```
ln P_t = ln EPS_ttm(t) + ln M_t
d lnE  = g dt + σ_E dW₁                     (earnings: drift + shocks)
d lnM  = κ(ln M̄ − lnM) dt + σ_M dW₂        (multiple: Ornstein–Uhlenbeck)
corr(dW₁, dW₂) = ρ                          (estimated; usually ≤ 0)
```

All parameters estimated per ticker from its own history. Scenario anchors
(bear/base/bull) = 25/50/75th percentiles of the stock's rolling-3y EPS CAGR
distribution and of its own historical P/E distribution. Monte Carlo (8k paths)
produces the fan; deterministic anchors overlay it.

## Data

- **SEC EDGAR XBRL** (`companyconcept`, free, no key): diluted EPS by quarter,
  fiscal-year-safe (quarterly = 80–100-day durations; fiscal Q4 synthesized as
  FY − interior quarters). 45-day publication lag applied so the backtest never
  uses numbers before the market had them.
- **Yahoo chart API** (free, no key): 25y monthly closes + adjusted closes
  (realized backtest returns are total-return via adjclose).

## Backtest protocol

Walk-forward, monthly origins, minimum 48 months of history, **past-only fits**
at every origin. Scored on: Pearson/Spearman IC of predicted-median vs realized
3-yr annualized return; MAE; sign hit-rate vs a 6% threshold; and distribution
calibration (coverage of the 50% and 90% predicted bands). Baselines: GBM with
trailing moments, and a constant-8% forecast.

## Honest results (Aug 2026, 3 tickers)

| | EME (stable compounder) | WMB (yieldco) | MU (cyclical) |
|---|---|---|---|
| IC (P/S) | **0.16 / 0.32** | −0.22 / −0.01 | −0.15 / 0.12 |
| MAE vs GBM vs const-8 | **21.6 / 23.1 / 25.1** | 30.1 / 17.7 / 17.1 | 31.9 / 19.4 / 19.2 |
| Coverage 90% band | 64% (overconfident) | 100% (over-wide) | 98% (over-wide) |
| Hit rate (±6%) | 80% | 64% | 56% |

**Interpretation:** the mean-reversion model adds value only for stable-multiple
compounders; for re-raters and cyclicals it is *worse than baselines* — multiple
reversion systematically fades regime changes. Calibration is regime-dependent
(EME's re-rating blew through the bands; WMB/MU bands are conservatively wide).
The model's honest use is as a *scenario framer and expectations translator*,
not a return predictor. Overlapping origins mean effective n ≈ 2–3 per ticker:
all ICs are descriptive, not significant.

## Roadmap

- [ ] Parameter-uncertainty widening (bootstrap the OU fit into the bands)
- [ ] Dividend-aware forecast paths (currently only realized side is total-return)
- [ ] Regime detection: flag non-stationary multiples, switch to wide-uncertainty mode
- [ ] Batch mode over a watchlist + email delivery via the existing routine webhook
- [ ] On-demand cloud runs (git-clone this repo in a scheduled Claude routine)

*Not investment advice. Free-data quirks possible; verify against filings.*
