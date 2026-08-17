#!/usr/bin/env python3
"""
backtest_books.py — walk-forward backtest of the ALPHA book's exact trading
rules on daily data, before trusting them with (fake) money. Adopt-or-retire.

    python3 backtest_books.py [--years=5]

Simulates the deployed strategy (score = -z(5d ret) + 0.5*z(6m mom), long
top-6 equal slots, exit when rank decays past EXIT_RANK, 10bps per side) over
the alpha universe, and compares variants:

  fill=close   fills at the SAME close the signal used (as deployed — this
               measures the optimistic same-bar bias the docstring admits)
  fill=open    fills at the NEXT day's open (honest execution)
  filter=spy   additionally blocks NEW entries while SPY < its 200-day MA
  exit=12      tighter exit rank vs the deployed 18

Costs 10 bps per side. Benchmark: SPY buy-and-hold. All from Yahoo daily bars
(cached 24h). Survivorship note: today's universe backtested into the past —
treat absolute levels as upper bounds, variant DELTAS as the real signal.
"""

import json, math, os, statistics, sys, time, urllib.request
from datetime import date

import alpha  # reuse the deployed universe + constants

UA = {"User-Agent": "gates-bt/1.0 (github.com/vibe-coder-789/gates-engine)"}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
COST = alpha.COST
TOP_N = alpha.TOP_N
WARMUP = 130

# AI / energy-to-compute lifecycle cohort (from the Watts-to-Tokens survey):
# energy & power infra -> chips & equipment -> networking/optics -> datacenter
# builders -> hyperscalers/models. Used by the ai-tilt experiment.
AI_SET = set("""NVDA AVGO AMD MU AMAT LRCX KLAC ANET COHR SMCI DELL VRT CEG VST
GEV PWR EME FIX MSFT META GOOGL AMZN AAPL ORCL TSM ARM ASML MRVL APP PLTR TER
NXPI ON MPWR CIEN GLW INTC QCOM TXN ADI MCHP CRM NOW PANW SNOW NET DDOG UBER
LIN ETN PH URI NEE""".split())


def daily_bars(ticker, years):
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, "btd_%s.json" % ticker.lower())
    if os.path.exists(key) and (time.time() - os.path.getmtime(key)) < 24 * 3600:
        with open(key) as f:
            return json.load(f)
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?range=%dy&interval=1d"
           % (ticker.upper(), years))
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode())
            res = d["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            rows = []
            for i in range(len(ts)):
                if q["close"][i] and q["open"][i]:
                    rows.append([ts[i] // 86400, round(q["open"][i], 4), round(q["close"][i], 4)])
            with open(key, "w") as f:
                json.dump(rows, f)
            time.sleep(0.25)
            return rows
        except Exception:
            if attempt == 2:
                return []
            time.sleep(5 * (attempt + 1))
    return []


def zmap(vals):
    xs = list(vals.values())
    if len(xs) < 20:
        return {}
    mu = statistics.mean(xs); sd = statistics.pstdev(xs) or 1e-9
    return {k: max(-3.0, min(3.0, (v - mu) / sd)) for k, v in vals.items()}


def load(years):
    data = {}
    for t in alpha.UNIVERSE:
        rows = daily_bars(t, years)
        if len(rows) >= WARMUP + 30:
            data[t] = rows
    spy = daily_bars("SPY", years)
    return data, spy


def simulate(data, spy, fill="close", spy_filter=False, exit_rank=None, ai_tilt=0.0):
    exit_rank = exit_rank or alpha.EXIT_RANK
    # calendar = SPY's days; per-ticker day->index map
    days = [r[0] for r in spy]
    idx = {t: {r[0]: i for i, r in enumerate(rows)} for t, rows in data.items()}
    spy_close = [r[2] for r in spy]
    cash, positions = 1000.0, {}          # t -> {shares, cost}
    curve, trades, gross_traded = [], 0, 0.0
    prev_holdings = set()
    for di in range(WARMUP, len(days) - 1):
        d = days[di]
        # scores from bars up to and including day d (completed closes)
        raw = {}
        for t, rows in data.items():
            i = idx[t].get(d)
            if i is None or i < 127:
                continue
            c = rows
            raw[t] = {"r5": c[i][2] / c[i - 5][2] - 1,
                      "mom": c[i - 21][2] / c[i - 126][2] - 1,
                      "i": i}
        z5, zm = zmap({t: v["r5"] for t, v in raw.items()}), zmap({t: v["mom"] for t, v in raw.items()})
        scores = {t: -1.0 * z5[t] + 0.5 * zm[t] + (ai_tilt if t in AI_SET else 0.0)
                  for t in raw if t in z5 and t in zm}
        if len(scores) < 30:
            continue
        ranked = sorted(scores, key=lambda t: -scores[t])
        rank = {t: i + 1 for i, t in enumerate(ranked)}

        def fill_px(t):
            i = raw[t]["i"]
            rows = data[t]
            if fill == "open" and i + 1 < len(rows):
                return rows[i + 1][1]
            return rows[i][2]

        # mark-to-market at close d
        def mtm():
            v = cash
            for t, p in positions.items():
                i = idx[t].get(d)
                v += p["shares"] * (data[t][i][2] if i is not None else p["cost"])
            return v

        # exits
        for t in list(positions):
            if t not in rank or rank[t] > exit_rank:
                px = fill_px(t) if t in raw else positions[t]["cost"]
                proceeds = positions[t]["shares"] * px
                cash += proceeds * (1 - COST)
                gross_traded += proceeds
                trades += 1
                del positions[t]
        # entries
        blocked = False
        if spy_filter and di >= 200:
            ma200 = sum(spy_close[di - 199:di + 1]) / 200
            blocked = spy_close[di] < ma200
        if not blocked:
            v = mtm()
            for t in ranked:
                if len(positions) >= TOP_N:
                    break
                if t in positions:
                    continue
                notional = min(v / TOP_N, cash / (1 + COST))
                if notional < 5:
                    break
                px = fill_px(t)
                positions[t] = {"shares": notional / px, "cost": px}
                cash -= notional * (1 + COST)
                gross_traded += notional
                trades += 1
        curve.append(mtm())
        prev_holdings = set(positions)
    # metrics
    rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve))]
    n_y = len(curve) / 252.0
    cagr = (curve[-1] / 1000.0) ** (1 / n_y) - 1 if n_y > 0 else 0
    vol = statistics.pstdev(rets) * math.sqrt(252) if rets else 0
    sharpe = cagr / vol if vol else 0
    peak, dd = 0.0, 0.0
    for v in curve:
        peak = max(peak, v); dd = min(dd, v / peak - 1)
    spy_cagr = (spy_close[len(days) - 2] / spy_close[WARMUP]) ** (1 / n_y) - 1
    return {"final": round(curve[-1], 2), "cagr": round(cagr, 4), "vol": round(vol, 4),
            "sharpe": round(sharpe, 2), "max_dd": round(dd, 3),
            "trades": trades, "turnover_x": round(gross_traded / 1000.0 / n_y, 1),
            "spy_cagr": round(spy_cagr, 4), "years": round(n_y, 1)}


if __name__ == "__main__":
    flags = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a}
    years = int(flags.get("--years", 5))
    print("loading %d tickers (daily, %dy)..." % (len(alpha.UNIVERSE), years), file=sys.stderr)
    data, spy = load(years)
    print("loaded %d ok" % len(data), file=sys.stderr)
    out = {"as_of": date.today().isoformat(), "universe_loaded": len(data), "variants": {}}
    for name, kw in (
            ("deployed (next-open, no tilt)", {"fill": "open"}),
            ("ai-tilt +0.3z", {"fill": "open", "ai_tilt": 0.3}),
            ("ai-tilt +0.6z", {"fill": "open", "ai_tilt": 0.6})):
        out["variants"][name] = simulate(data, spy, **kw)
        print("%-32s %s" % (name, out["variants"][name]), file=sys.stderr)
    print(json.dumps(out, indent=1))
