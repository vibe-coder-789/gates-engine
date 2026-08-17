#!/usr/bin/env python3
"""
alpha.py — gates short-term alpha book (paper). Fake money, real prices,
mechanical rules, daily cadence. Separate thread from the long book: the
earnings model (signals.json) may VETO names, never nominate them.

    python3 alpha.py --state=astate.json [--signals=signals.json] [--out=new.json]
                     [--dry-run] [--vetoes=vetoes.json]

--dry-run: compute and print intended trades WITHOUT mutating state — used by
the LLM event-review node to see proposed entries before execution.
--vetoes: {"TICKER": "reason"} — entries for these names are BLOCKED and
logged to state["veto_log"] with the price at veto time, so the node's
judgment can be scored later (counterfactual: what the vetoed entry would
have returned). The veto node has asymmetric authority: it can only block
entries, never initiate trades, touch exits, or enlarge positions.

--universe: a screener-produced universe.json ({"tickers": [...]}) replaces
the built-in fallback list — discovery is mechanical (see screener.py) and
refreshed monthly by the Universe Scout routine.

STRATEGY (fixed; changing it means committing a new version):
- Universe: the screener's ~150 most liquid US names (fallback: the built-in
  list below), whoever has >=140 daily bars.
- Signals (close-to-close daily bars, z-scored, winsorized +-3):
    STR   5-day return, weight -1.0      (short-term reversal: buy dips)
    MOM   126d return skipping last 21d, weight +0.5
- Veto: a name graded C by the long model with ensemble < 4%/yr is excluded.
- Book: long top 6 equal-weight (1/6 slots); a holding is kept until its
  composite rank decays past 18 (turnover buffer), then sold.
- Costs 10 bps per side; fractional shares; cash earns nothing.
- Execution: signals from COMPLETED daily bars only (today's partial bar is
  excluded); fills at the live market price when the run executes — the
  routine is scheduled just after the US open, making this next-open
  execution, the variant that walk-forward-backtested best (Sharpe 0.90 vs
  0.67 for same-close fills over 4.5y; see backtest_books.py).
- Risk: book >15% below its peak halts new entries until it recovers.
- Benchmark: SPY buy-and-hold from inception.
"""

import json, math, statistics, sys, time, urllib.request
from datetime import date

UA = {"User-Agent": "gates-alpha/1.0 (github.com/vibe-coder-789/gates-engine)"}
TOP_N = 6
EXIT_RANK = 18
COST = 0.001
VETO_GRADE, VETO_ENS = "C", 0.04
KILL_DD = 0.15   # book down >15% from its peak: exits only, no new entries

UNIVERSE = """
AAPL MSFT AMZN META GOOGL AVGO ORCL CRM AMD QCOM TXN MU AMAT LRCX KLAC ANET
PANW CSCO IBM NOW INTU ACN ADI UBER ABNB PYPL SHOP BKNG NFLX CMCSA TMUS
JPM BAC WFC GS MS SCHW BLK AXP V MA
UNH JNJ LLY MRK ABBV TMO ABT AMGN GILD VRTX ISRG SYK
XOM CVX COP SLB PSX VLO KMI WMB OKE LNG
GE HON CAT DE UNP UPS LMT RTX NOC ETN PH ITW EME PWR FIX URI
WMT COST TGT HD LOW PG KO PEP
LIN APD SHW FCX NEM NEE DUK SO VST CEG
AMT PLD EQIX MCD SBUX CMG NKE GM F TSLA
""".split()


def daily_closes(ticker):
    """~1y of daily closes [(iso_date, px)]; [] on failure."""
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?range=1y&interval=1d"
           % ticker.upper())
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode())
            res = d["chart"]["result"][0]
            ts = res["timestamp"]
            cl = res["indicators"]["quote"][0]["close"]
            out = []
            for i in range(len(ts)):
                if cl[i]:
                    out.append((ts[i], float(cl[i])))
            time.sleep(0.25)
            return out
        except Exception:
            if attempt:
                return []
            time.sleep(4)
    return []


def zmap(vals):
    """{name: winsorized z} from {name: value}."""
    xs = list(vals.values())
    if len(xs) < 20:
        return {}
    mu = statistics.mean(xs); sd = statistics.pstdev(xs) or 1e-9
    return {k: max(-3.0, min(3.0, (v - mu) / sd)) for k, v in vals.items()}


def compute_scores(signals, universe=None):
    """{ticker: {score, px, r5, mom}} for the tradable universe today."""
    universe = universe or UNIVERSE
    veto = set()
    if signals:
        for t, s in signals.get("tickers", {}).items():
            if s.get("grade") == VETO_GRADE and (s.get("ensemble") or 0) < VETO_ENS:
                veto.add(t.upper())
    import datetime as _dt
    today_utc = _dt.datetime.utcnow().date()
    raw = {}
    for t in universe:
        if t in veto:
            continue
        bars = daily_closes(t)
        if len(bars) < 141:
            continue
        live = bars[-1][1]                     # latest price = fill price
        # signals use completed bars only: drop today's (partial) bar
        done = bars[:-1] if _dt.datetime.utcfromtimestamp(bars[-1][0]).date() == today_utc \
               else bars
        px = [p for _, p in done]
        if len(px) < 140:
            continue
        r5 = px[-1] / px[-6] - 1
        mom = px[-22] / px[-127] - 1
        raw[t] = {"px": live, "r5": r5, "mom": mom}
    z5 = zmap({t: v["r5"] for t, v in raw.items()})
    zm = zmap({t: v["mom"] for t, v in raw.items()})
    out = {}
    for t, v in raw.items():
        if t in z5 and t in zm:
            out[t] = dict(v, score=-1.0 * z5[t] + 0.5 * zm[t])
    return out


def run(state, signals, vetoes=None, universe=None):
    vetoes = vetoes or {}
    today = date.today().isoformat()
    scores = compute_scores(signals, universe=universe)
    if len(scores) < 30:
        return {"state": state, "trades": [],
                "valuation": {"date": today, "error": "only %d names scored — data problem, no trades" % len(scores)}}
    ranked = sorted(scores, key=lambda t: -scores[t]["score"])
    rank = {t: i + 1 for i, t in enumerate(ranked)}
    px = {t: scores[t]["px"] for t in scores}
    # SPY benchmark px via its own daily series
    spy_bars = daily_closes("SPY")
    spy = spy_bars[-1][1] if spy_bars else None
    if not state.get("spy_inception_px") and spy:
        state["spy_inception_px"] = spy
    for t in list(state["positions"]):        # value stale names at last known px
        if t not in px:
            px[t] = state["positions"][t]["avg_cost"]

    def port_value():
        return state["cash"] + sum(p["shares"] * px[t] for t, p in state["positions"].items())

    trades = []

    def sell(t, why):
        p = state["positions"].pop(t)
        state["cash"] += p["shares"] * px[t] * (1 - COST)
        trades.append({"date": today, "side": "SELL", "ticker": t,
                       "shares": round(p["shares"], 4), "px": round(px[t], 2),
                       "notional": round(p["shares"] * px[t], 2), "why": why})

    # exits: rank decayed past the buffer, or vetoed/unscored today
    for t in list(state["positions"]):
        if t not in rank:
            sell(t, "left universe / vetoed / no data")
        elif rank[t] > EXIT_RANK:
            sell(t, "rank decayed to %d (> %d)" % (rank[t], EXIT_RANK))
    # risk layer: drawdown kill switch — exits above still ran; block entries
    peak = max([h.get("v", state["initial"]) for h in (state.get("history") or [])]
               + [state["initial"], port_value()])
    halted = port_value() / peak - 1 < -KILL_DD
    # entries: fill empty slots with best-ranked names not held
    slots = 0 if halted else (TOP_N - len(state["positions"]))
    v = port_value()
    vetoed_today = []
    for t in ranked:
        if slots <= 0:
            break
        if t in state["positions"]:
            continue
        if t.upper() in vetoes:
            vetoed_today.append({"date": today, "ticker": t, "px": round(px[t], 2),
                                 "reason": str(vetoes[t.upper()])[:200]})
            continue
        notional = min(v / TOP_N, state["cash"] / (1 + COST))
        if notional < 5:
            break
        sh = notional / px[t]
        state["positions"][t] = {"shares": sh, "avg_cost": px[t]}
        state["cash"] -= notional * (1 + COST)
        trades.append({"date": today, "side": "BUY", "ticker": t,
                       "shares": round(sh, 4), "px": round(px[t], 2),
                       "notional": round(notional, 2),
                       "why": "rank %d, score %.2f (STR %.1f%%, MOM %.1f%%)"
                              % (rank[t], scores[t]["score"],
                                 scores[t]["r5"] * 100, scores[t]["mom"] * 100)})
        slots -= 1
    v = port_value()
    port_ret = v / state["initial"] - 1
    spy_ret = (spy / state["spy_inception_px"] - 1) if (spy and state.get("spy_inception_px")) else None
    valuation = {
        "date": today, "value": round(v, 2), "cash": round(state["cash"], 2),
        "return_pct": round(port_ret * 100, 2),
        "spy_return_pct": None if spy_ret is None else round(spy_ret * 100, 2),
        "vs_spy_pp": None if spy_ret is None else round((port_ret - spy_ret) * 100, 2),
        "positions": {t: {"shares": round(p["shares"], 4), "px": round(px[t], 2),
                          "value": round(p["shares"] * px[t], 2),
                          "rank_today": rank.get(t),
                          "pnl_pct": round((px[t] / p["avg_cost"] - 1) * 100, 1)}
                      for t, p in sorted(state["positions"].items())},
        "universe_scored": len(scores),
        "risk_halted": halted,
        "vetoed_today": vetoed_today,
        "top10_today": [{"t": t, "score": round(scores[t]["score"], 2)} for t in ranked[:10]]}
    state["veto_log"] = ((state.get("veto_log") or []) + vetoed_today)[-300:]
    state["log"] = ((state.get("log") or []) + trades)[-800:]
    # equity snapshot for the monitoring dashboard (one per date, latest wins)
    snap = {"d": today, "v": round(v, 2), "s": None if spy is None else round(spy, 2)}
    state["history"] = ([h for h in (state.get("history") or []) if h.get("d") != today]
                        + [snap])[-600:]
    state["last_run"] = today
    return {"state": state, "trades": trades, "valuation": valuation}


if __name__ == "__main__":
    import copy
    flags = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[1:]}
    with open(flags["--state"]) as f:
        state = json.load(f)
    signals = None
    if "--signals" in flags:
        try:
            with open(flags["--signals"]) as f:
                signals = json.load(f)
        except Exception:
            signals = None
    vetoes = {}
    if "--vetoes" in flags:
        try:
            with open(flags["--vetoes"]) as f:
                vetoes = {k.upper(): v for k, v in json.load(f).items()}
        except Exception:
            vetoes = {}
    universe = None
    if "--universe" in flags:
        try:
            with open(flags["--universe"]) as f:
                u = json.load(f).get("tickers")
            if u and len(u) >= 40:            # sanity floor: refuse a broken file
                universe = [t.upper() for t in u]
        except Exception:
            universe = None                    # fall back to built-in list
    dry = bool(flags.get("--dry-run"))
    out = run(copy.deepcopy(state) if dry else state, signals, vetoes=vetoes,
              universe=universe)
    if "--out" in flags and not dry:
        with open(flags["--out"], "w") as f:
            json.dump(out["state"], f, indent=1)
    print(json.dumps({"dry_run": dry, "trades": out["trades"],
                      "valuation": out["valuation"]}, indent=1))
