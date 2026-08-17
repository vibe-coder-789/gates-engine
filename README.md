# gates-engine

**A single-name stock price prediction model.** Given a ticker, it produces an
honest 5-year price distribution: a Monte-Carlo fan, bear/base/bull scenario
paths, and a per-ticker reliability grade from a walk-forward backtest.
One model, one stock at a time — no factor zoo.

## Usage

```bash
python3 engine.py EME                 # full run: fetch → fit → simulate → backtest → report
python3 engine.py WMB --no-backtest   # faster, skips walk-forward
python3 engine.py MU --horizon=5 --paths=8000 --bt-h=3
python3 engine.py bundle EME MSFT AAPL --out=bundle.json   # batch → Terminal/Lab UI
open reports/eme.html                 # styled report with fan chart + backtest
```

Stdlib-only (no pip installs). Data cached in `.cache/` (1 day).

## Model

```
ln P_t = ln EPS_ttm(t) + ln M_t
d lnE  = g(t) dt + σ_E dW₁                  (earnings: drift + shocks)
d lnM  = κ(ln M̄ − lnM) dt + σ_M dW₂        (multiple: Ornstein–Uhlenbeck)
corr(dW₁, dW₂) = ρ                          (estimated; usually ≤ 0)
```

All parameters estimated per ticker from its own history, with guards learned
the hard way: split-basis truncation (as-filed EPS vs adjusted prices), 3–75×
multiple hygiene + winsorizing, near-unit-root anchor clamping, σ_M capping,
re-rating regime detection (anchor blended toward the recent mean when the last
3y sit >1σ from the long mean), and bootstrap parameter mixing so the fan
carries estimation risk, not just shock risk.

**One earnings drift — a single prediction, no modes.** g(t) is an
equal-weight blend of two legs:

- **Street leg** (years 1–2, fading out): analyst low/avg/high estimate chains
  (growth rates on Yahoo's basis applied to the GAAP base — levels never mixed
  across bases), from Yahoo earningsTrend.
- **History leg**: the stock's own rolling-3y EPS CAGR quartiles, with the
  bear's leg floored at ≤0 so a benign sample cannot rule out contraction.

Scenario anchors pair those growth chains with 25/50/75th-percentile exit
multiples from the stock's own P/E history. Names without analyst coverage
fall back to pure history — which is exactly the spine the walk-forward
backtest grades; the Street tilt itself is *unvalidated* on free data
(estimate history is paywalled) and every surface says so. The 50/50 weight is
a robustness default, not an estimate — revisit it if estimate history is
ever purchased.

## Data

- **SEC EDGAR XBRL** (free, no key): diluted GAAP EPS by quarter,
  fiscal-year-safe (duration-based extraction; fiscal Q4 synthesized as
  FY − interior quarters), 45-day publication lag against lookahead.
- **Yahoo chart API** (free, no key): 25y monthly closes + adjusted closes
  (realized backtest returns are total-return), split dates.
- **Yahoo earningsTrend** (free, cookie+crumb): analyst consensus EPS
  estimates + 30-day revision counts, for forward mode.

## Backtest protocol

Walk-forward, monthly origins, minimum 48 months of history, **past-only fits**
at every origin. Scored on: Pearson/Spearman IC of predicted-median vs realized
3-yr annualized return; MAE; sign hit-rate vs a 6% threshold; and distribution
calibration (coverage of the 50% and 90% predicted bands). Baselines: GBM with
trailing moments, and a constant-8% forecast. Overlapping origins mean
effective n ≈ n/(12·H): ICs are descriptive, not significant.

## Honest results (Aug 2026)

| | EME (stable compounder) | WMB (yieldco) | MU (cyclical) |
|---|---|---|---|
| IC (P/S) | **0.16 / 0.32** | −0.22 / −0.01 | −0.15 / 0.12 |
| MAE vs GBM vs const-8 | **21.6 / 23.1 / 25.1** | 30.1 / 17.7 / 17.1 | 31.9 / 19.4 / 19.2 |
| Coverage 90% band | 64% (overconfident) | 100% (over-wide) | 98% (over-wide) |
| Hit rate (±6%) | 80% | 64% | 56% |

**Interpretation:** the mean-reversion model adds value only for stable-multiple
compounders; for re-raters and cyclicals it is *worse than baselines* — multiple
reversion systematically fades regime changes. The model's honest use is as a
*scenario framer and expectations translator*, not a return predictor. The
per-ticker A/B/C grade in the Terminal encodes exactly this.

## Paper-trading books

Two $1,000 simulated books trade the system's outputs (fake money, real
prices, mechanical rules; state in Gmail drafts, runs via cloud routines,
live dashboard reads state read-only):

- **Long book** (`paper.py`, weekly): grade A/B + P(loss)<40% gate, top-4 by
  ensemble at 25% slots, unfilled slots stay cash.
- **Alpha book** (`alpha.py`, daily): −z(5d return) + 0.5·z(6m momentum) over
  ~100 liquid names, top-6, exit at rank>18; the earnings model can veto,
  never nominate. Execution is signal-on-completed-bars → fill-at-next-open,
  adopted after `backtest_books.py` (4.5y, 107 names, 10bps): next-open
  Sharpe 0.90 vs 0.67 for same-close fills; a SPY 200-dma entry filter and a
  tighter exit rank were tested and retired (no Sharpe gain). Survivorship
  caveat applies; ~130×/yr turnover means real-world costs would bite harder
  than 10bps.
- **Risk layer (both)**: a book >15% below its peak stops opening new
  positions until it recovers.

Design stance, informed by the 2025–26 open-source landscape (TradingAgents,
ai-hedge-fund, AI-Trader/Agent-Market-Arena benchmarks): deterministic code
makes every trading decision; LLM agents only orchestrate (state, delivery,
error handling). The live-benchmark literature keeps finding instability and
memorized-history contamination exactly where LLMs sit inside the decision
loop — so here they don't.

## Retired experiments

- **Rates-conditioned multiple anchor** (v0.4 experiment, retired): lnM̄ =
  a + b·(10y Treasury yield) with OU on the residual — theoretically sound
  (P/E ≈ inverse discount rate), empirically worse: walk-forward MAE degraded
  on 5 of 7 testable names, improved on 1. The multiple's own history beats
  yield conditioning at single-name horizon. Code remains (`--variant rates`
  in the experiment harness) as the record.
- **Cycle-position drift** (v0.4 experiment, retired *as an automatic
  adjustment*): shrink EPS drift toward the log-linear earnings trend. Big
  wins exactly where the model is weakest (MU MAE 0.32→0.25, COHR IC
  −0.22→+0.23) but losses on steady names — and no ex-ante covariate we have
  (σ_E included) separates the winners from the losers, so adopting it
  conditionally would be fitting the condition to the answer. The cycle
  position (EPS-vs-trend z) ships as a **flag** in the Terminal instead:
  real information, human judgment required.

- **Cross-sectional factor layer** (v0.3, `xsection.py`, removed): monthly
  VALUE/MOM/GROWTH/LOWVOL/REVGAP ranking over a 272-name large-cap universe,
  2012–2026. Result: no factor achieved |t| > 1.1; the composite long-short
  lost 6.8%/yr gross while the equal-weight market made 16.5%/yr — and fixing
  the split-basis contamination made the factors *worse*, proving the little
  apparent signal was partly a data bug. A clean negative: these signals carry
  no cross-sectional edge in recent large-cap US data, so the ranking machine
  was cut and the project refocused on single-name prediction. (Reproducible
  from git history.)

## Roadmap

- [x] Parameter-uncertainty widening (bootstrap OU refits mixed into the fan)
- [x] Regime detection + anchor blending for re-rated names
- [x] Split-basis guard (truncate at last split; as-filed EPS vs adjusted prices)
- [x] Batch bundle mode + Terminal/Lab UI + on-demand cloud runs ([ANALYZE] queue)
- [x] Street/history blended drift — one unified prediction (Street leg unvalidated, see above)
- [ ] Dividend-aware forecast paths (currently only realized side is total-return)
- [ ] Point-in-time estimate history (paid data) → validate forward mode, revisions signal
- [ ] Qualitative layer: agent reads filings/transcripts for what the statistics can't see

*Not investment advice. Free-data quirks possible; verify against filings.*
