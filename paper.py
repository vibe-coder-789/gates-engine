#!/usr/bin/env python3
"""
paper.py — gates-engine paper-trading simulator. Fake money, real prices,
mechanical rules. Stdlib only.

    python3 paper.py --state=state.json --signals=signals.json [--out=newstate.json]

Reads portfolio state + model signals, fetches live Yahoo quotes, applies the
strategy, prints a JSON report {state, trades, valuation} and (with --out)
writes the new state.

STRATEGY (fixed; changing it means committing a new version):
- Eligible: model reliability grade A or B AND P(loss over 5y) < 40%.
- Rank eligible by ensemble expected return; target = top 4, each at 25% of
  portfolio value. Fewer than 4 eligible -> the unfilled slots stay in cash
  (the model finding few good bets is information, not a reason to concentrate).
- Sell anything held that is no longer eligible or drops out of the top 4.
- Trade only when composition changes or a position drifts >5pp from target.
- Cost: 10 bps of traded notional. Fractional shares. Cash earns nothing.
- Benchmark: SPY buy-and-hold from inception (recorded in state).

STATE (JSON): {inception, cash, spy_inception_px, positions:{T:{shares,avg_cost}},
               log:[...], last_run}
"""

import json, sys, time, urllib.request
from datetime import date

UA = {"User-Agent": "gates-paper/1.0 (github.com/vibe-coder-789/gates-engine)"}
TOP_N = 4
COST = 0.001          # 10 bps per side
DRIFT_PP = 0.05       # rebalance threshold, absolute weight
MAX_PLOSS = 0.40


def quote(ticker):
    """Latest close-ish price from Yahoo chart API (1d/1m tail)."""
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?range=1d&interval=5m"
           % ticker.upper())
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode())
            res = d["chart"]["result"][0]
            px = res["meta"].get("regularMarketPrice")
            if px:
                return float(px)
            closes = [c for c in res["indicators"]["quote"][0]["close"] if c]
            return float(closes[-1])
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("no quote for " + ticker)


def eligible(signals):
    """[(ticker, ensemble)] top-N target list, best first."""
    rows = []
    for t, s in signals["tickers"].items():
        if s.get("grade") in ("A", "B") and (s.get("p_loss") or 1.0) < MAX_PLOSS \
                and s.get("ensemble") is not None:
            rows.append((t, s["ensemble"]))
    rows.sort(key=lambda x: -x[1])
    return rows[:TOP_N]


def run(state, signals):
    today = date.today().isoformat()
    targets = eligible(signals)
    target_names = [t for t, _ in targets]
    held = list(state["positions"].keys())
    tickers = sorted(set(target_names) | set(held))
    px = {t: quote(t) for t in tickers}
    spy = quote("SPY")
    if not state.get("spy_inception_px"):
        state["spy_inception_px"] = spy

    def port_value():
        return state["cash"] + sum(p["shares"] * px[t]
                                   for t, p in state["positions"].items())

    trades = []

    def sell(t, why):
        p = state["positions"].pop(t)
        notional = p["shares"] * px[t]
        state["cash"] += notional * (1 - COST)
        trades.append({"date": today, "side": "SELL", "ticker": t,
                       "shares": round(p["shares"], 4), "px": round(px[t], 2),
                       "notional": round(notional, 2), "why": why})

    def trade_to(t, target_notional, why):
        cur = state["positions"].get(t, {"shares": 0.0, "avg_cost": 0.0})
        cur_notional = cur["shares"] * px[t]
        delta = target_notional - cur_notional
        if abs(delta) < 1.0:
            return
        if delta > 0:
            delta = min(delta, state["cash"] / (1 + COST))
            if delta < 1.0:
                return
            sh = delta / px[t]
            new_sh = cur["shares"] + sh
            cur["avg_cost"] = (cur["avg_cost"] * cur["shares"] + delta) / new_sh
            cur["shares"] = new_sh
            state["cash"] -= delta * (1 + COST)
            state["positions"][t] = cur
            trades.append({"date": today, "side": "BUY", "ticker": t,
                           "shares": round(sh, 4), "px": round(px[t], 2),
                           "notional": round(delta, 2), "why": why})
        else:
            sh = min(-delta / px[t], cur["shares"])
            cur["shares"] -= sh
            state["cash"] += sh * px[t] * (1 - COST)
            trades.append({"date": today, "side": "TRIM", "ticker": t,
                           "shares": round(sh, 4), "px": round(px[t], 2),
                           "notional": round(sh * px[t], 2), "why": why})
            if cur["shares"] * px[t] < 1.0:
                state["positions"].pop(t, None)
            else:
                state["positions"][t] = cur

    # 1. exit anything no longer targeted
    for t in held:
        if t not in target_names:
            sell(t, "no longer eligible / out of top %d" % TOP_N)
    # 2. rebalance toward equal weights (only past drift threshold, or new)
    if target_names:
        v = port_value()
        tw = 1.0 / TOP_N          # unfilled slots stay in cash by design
        for t in target_names:
            cur_w = state["positions"].get(t, {"shares": 0})["shares"] * px[t] / v \
                    if v > 0 else 0.0
            if t not in state["positions"] or abs(cur_w - tw) > DRIFT_PP:
                trade_to(t, v * tw, "target %.0f%% (was %.0f%%)" % (tw * 100, cur_w * 100))
    v = port_value()
    spy_ret = spy / state["spy_inception_px"] - 1
    port_ret = v / state["initial"] - 1
    valuation = {
        "date": today, "value": round(v, 2), "cash": round(state["cash"], 2),
        "return_pct": round(port_ret * 100, 2),
        "spy_return_pct": round(spy_ret * 100, 2),
        "vs_spy_pp": round((port_ret - spy_ret) * 100, 2),
        "positions": {t: {"shares": round(p["shares"], 4),
                          "px": round(px[t], 2),
                          "value": round(p["shares"] * px[t], 2),
                          "avg_cost": round(p["avg_cost"], 2),
                          "pnl_pct": round((px[t] / p["avg_cost"] - 1) * 100, 1)
                                     if p["avg_cost"] else None}
                      for t, p in sorted(state["positions"].items())},
        "signals_asof": signals.get("as_of"),
        "targets": [{"ticker": t, "ensemble_pct": round(e * 100, 1)} for t, e in targets]}
    state["log"] = (state.get("log") or []) + trades
    state["log"] = state["log"][-500:]
    state["last_run"] = today
    return {"state": state, "trades": trades, "valuation": valuation}


if __name__ == "__main__":
    flags = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a}
    with open(flags["--state"]) as f:
        state = json.load(f)
    with open(flags["--signals"]) as f:
        signals = json.load(f)
    out = run(state, signals)
    if "--out" in flags:
        with open(flags["--out"], "w") as f:
            json.dump(out["state"], f, indent=1)
    print(json.dumps({"trades": out["trades"], "valuation": out["valuation"]}, indent=1))
