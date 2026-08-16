#!/usr/bin/env python3
"""
xsection.py — v0.3 cross-sectional factor layer for gates-engine.

    python3 xsection.py [--start=2012] [--html=reports/xsection.html]

Monthly cross-sectional ranking over a curated large-cap US-GAAP universe.
Factors (all point-in-time, 45d publication lag via engine.build_panel):

  VALUE   earnings yield  E/P (ttm)
  MOM     12-1 momentum   total return t-12 -> t-1 (skip most recent month)
  GROWTH  ln(EPS_ttm / EPS_ttm[-12m]), clipped
  LOWVOL  -stdev(12m monthly total returns)
  REVGAP  ln(expanding median multiple) - ln(current multiple)   [the OU gap,
          cross-sectionalized -- gates-engine's signature signal]

Composite = mean of available factor z-scores (winsorized +-3).
Validation: monthly cross-sectional Spearman IC per factor & composite
(t-stat legitimate: months are non-overlapping), quintile long-short (Q5-Q1)
equal-weight spread, turnover, and a cost-adjusted net figure.

HONESTY, PROMINENTLY:
- SURVIVORSHIP BIAS: the universe is a 2026 list of large liquid names
  backtested into the past. Companies that died or shrank are absent, which
  inflates absolute returns and flatters low-vol/quality especially. Treat
  long-short IC/spreads as upper bounds; deltas BETWEEN factors are more
  trustworthy than levels.
- Gross figures ignore costs until the stated cost line; shorting is
  modeled naively (no borrow costs/constraints).
- GAAP EPS basis; ~monthly rebalance; no risk model beyond equal weight.
- SPLIT-BASIS GUARD: Yahoo prices are retroactively split-adjusted but EDGAR
  EPS is as-filed, so for months before a name's most recent stock split the
  E/P-family factors would be garbage. EPS is masked (None) for those months:
  VALUE/GROWTH/REVGAP skip them, while MOM/LOWVOL (adjusted-return based,
  basis-safe) keep the full history.
"""

import json, math, os, statistics, sys
from datetime import date
import engine  # reuse fetchers/panel builder (cache shared)

UNIVERSE = """
AAPL MSFT NVDA AMZN META GOOGL AVGO ORCL CRM ADBE AMD INTC QCOM TXN MU AMAT
LRCX KLAC SNPS CDNS ANET PANW CSCO IBM NOW INTU ACN ADI NXPI MCHP ON SWKS
TER ENPH MPWR CRWD FTNT ZS DDOG NET TEAM WDAY ZM DOCU OKTA MDB SNOW PLTR
UBER ABNB DASH LYFT PYPL SQ COIN HOOD SHOP ETSY EBAY BKNG EXPE MAR HLT
DIS NFLX CMCSA T VZ TMUS
JPM BAC WFC C GS MS SCHW BLK BX KKR APO AXP V MA COF USB PNC TFC BK STT
AIG MET PRU ALL TRV CB PGR AFL HIG CINF
UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD VRTX REGN BIIB MRNA CVS
CI ELV HUM HCA ISRG SYK BSX MDT EW ZBH BAX BDX A IDXX IQV MTD RMD DXCM WAT
XOM CVX COP EOG SLB HAL BKR OXY PSX VLO MPC KMI WMB OKE LNG DVN FANG HES
APA EQT CTRA TRGP
GE HON MMM CAT DE UNP CSX NSC UPS FDX LMT RTX NOC GD BA LHX TDG HEI EMR ETN
PH ROK DOV ITW CMI PCAR JCI CARR OTIS AME FTV XYL IR PWR EME FIX MAS VMC MLM
NUE STLD X CLF FAST GWW URI SWK
WMT COST TGT HD LOW TJX ROST DG DLTR KR SYY PG KO PEP CL KMB GIS K HSY MKC
STZ MDLZ MO PM EL CHD CLX TSN HRL CAG CPB SJM
LIN APD SHW ECL DD DOW PPG LYB CE ALB FCX NEM
NEE DUK SO D EXC AEP SRE XEL ED PEG WEC ES EIX PPL FE AEE CMS DTE ETR CNP
VST CEG NRG AES
AMT PLD CCI EQIX PSA SPG O WELL DLR AVB EQR ESS MAA UDR VTR ARE BXP
MCD SBUX CMG YUM DPZ QSR TXRH NKE LULU RL VFC GM F TSLA RIVN HOG
CHTR ROKU TTD SPOT PINS SNAP
""".split()

START_DEFAULT = 2012
PUB_LAG_OK_MIN_MONTHS = 40   # min usable months to include a name at all


def month_key(d):
    return d.year * 12 + d.month

def load_universe(verbose=True):
    """{ticker: {month_key: (px, adj, eps_ttm)}} — reuses engine cache."""
    data, failed = {}, []
    for i, t in enumerate(UNIVERSE):
        try:
            cik, _ = engine.ticker_to_cik(t)
            eps_q = engine.quarterly_eps(cik, t)
            prices = engine.monthly_prices(t)
            panel = engine.build_panel(prices, eps_q, {})
            if len(panel) < PUB_LAG_OK_MIN_MONTHS:
                failed.append((t, "short"))
                continue
            # split-basis guard: EPS is valid only for months whose whole TTM
            # window sits after the last split (see module docstring)
            lsd = engine.last_split_date(t)
            if lsd is None:
                eps_ok = None                            # all months fine
            else:
                clean = engine.build_panel(prices, eps_q, {}, min_q_end=lsd)
                eps_ok = {month_key(r["date"]) for r in clean}
            data[t] = {month_key(r["date"]):
                       (r["px"], r["adj"],
                        r["eps"] if eps_ok is None or month_key(r["date"]) in eps_ok else None)
                       for r in panel}
        except SystemExit as e:
            failed.append((t, str(e)[:40]))
        except Exception as e:
            failed.append((t, type(e).__name__))
        if verbose and (i + 1) % 40 == 0:
            print("  ...%d/%d loaded (%d ok)" % (i + 1, len(UNIVERSE), len(data)),
                  file=sys.stderr)
    return data, failed

# ------------------------------------------------------------------ factors

def zscores(pairs):
    """[(name, value)] -> {name: winsorized z}"""
    vals = [v for _, v in pairs]
    if len(vals) < 20:
        return {}
    mu = statistics.mean(vals); sd = statistics.pstdev(vals) or 1e-9
    return {n: max(-3.0, min(3.0, (v - mu) / sd)) for n, v in pairs}

def spearman(a, b):
    n = len(a)
    ra = {}; rb = {}
    for r, (v, i) in enumerate(sorted((v, i) for i, v in enumerate(a))): ra[i] = r
    for r, (v, i) in enumerate(sorted((v, i) for i, v in enumerate(b))): rb[i] = r
    ma = mb = (n - 1) / 2.0
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)) *
                    sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / den if den else 0.0

def compute_month(data, mk):
    """Factor z-scores and forward returns for month key mk."""
    rows = {}
    for t, series in data.items():
        cur = series.get(mk)
        nxt = series.get(mk + 1)
        if not cur or not nxt:
            continue
        px, adj, eps = cur
        hist = [series.get(mk - k) for k in range(0, 13)]
        if any(h is None for h in hist):
            continue
        adjs = [h[1] for h in hist]                      # adj[mk-k], k=0..12
        rets12 = [adjs[k] / adjs[k + 1] - 1 for k in range(12)]
        f = {}
        if eps is not None:
            f["VALUE"] = eps / px                        # can be negative: fine
        f["MOM"] = adjs[1] / adjs[12] - 1                # t-12 .. t-1
        prior = series.get(mk - 12)
        if eps is not None and eps > 0 and prior and prior[2] is not None and prior[2] > 0:
            f["GROWTH"] = max(-1.0, min(1.0, math.log(eps / prior[2])))
        f["LOWVOL"] = -statistics.pstdev(rets12)
        # expanding median multiple (>=36 obs) for the reversion gap
        past_m = [series[k2][0] / series[k2][2]
                  for k2 in series if k2 <= mk and series[k2][2] is not None
                  and series[k2][2] > 0
                  and 3 <= series[k2][0] / series[k2][2] <= 75]
        if len(past_m) >= 36 and eps is not None and eps > 0 and 3 <= px / eps <= 75:
            med = sorted(past_m)[len(past_m) // 2]
            f["REVGAP"] = math.log(med) - math.log(px / eps)
        rows[t] = {"f": f, "fwd": nxt[1] / adj - 1}
    if len(rows) < 40:
        return None
    z = {}
    for fac in ("VALUE", "MOM", "GROWTH", "LOWVOL", "REVGAP"):
        z[fac] = zscores([(t, r["f"][fac]) for t, r in rows.items() if fac in r["f"]])
    comp = {}
    for t in rows:
        zs = [z[fac][t] for fac in z if t in z[fac]]
        if len(zs) >= 3:
            comp[t] = sum(zs) / len(zs)
    return {"rows": rows, "z": z, "comp": comp}

# ------------------------------------------------------------------ backtest

def run(start_year=START_DEFAULT, html_out=None):
    print("loading universe (%d tickers, cached after first run)..." % len(UNIVERSE),
          file=sys.stderr)
    data, failed = load_universe()
    print("loaded %d ok, %d failed/skipped" % (len(data), len(failed)), file=sys.stderr)
    mks = sorted({mk for s in data.values() for mk in s})
    mks = [mk for mk in mks if mk >= start_year * 12 + 1 and mk <= month_key(date.today()) - 1]
    ics = {f: [] for f in ("VALUE", "MOM", "GROWTH", "LOWVOL", "REVGAP", "COMPOSITE")}
    ls_rets, mkt_rets, months, prev_q = [], [], [], {}
    turnover = []
    for mk in mks:
        cm = compute_month(data, mk)
        if cm is None:
            continue
        rows, z, comp = cm["rows"], cm["z"], cm["comp"]
        names = list(comp.keys())
        fwd = {t: rows[t]["fwd"] for t in names}
        for fac in z:
            common = [t for t in names if t in z[fac]]
            if len(common) >= 40:
                ics[fac].append(spearman([z[fac][t] for t in common],
                                         [fwd[t] for t in common]))
        ics["COMPOSITE"].append(spearman([comp[t] for t in names],
                                         [fwd[t] for t in names]))
        ranked = sorted(names, key=lambda t: comp[t])
        q = max(10, len(ranked) // 5)
        bot, top = ranked[:q], ranked[-q:]
        ls = (sum(fwd[t] for t in top) / len(top)
              - sum(fwd[t] for t in bot) / len(bot))
        ls_rets.append(ls)
        mkt_rets.append(sum(fwd[t] for t in names) / len(names))
        months.append(mk)
        newq = {t: ("T" if t in top else "B") for t in list(top) + list(bot)}
        if prev_q:
            stay = sum(1 for t, s in newq.items() if prev_q.get(t) == s)
            turnover.append(1 - stay / max(1, len(newq)))
        prev_q = newq

    def icstats(xs):
        if len(xs) < 24:
            return None
        mu = statistics.mean(xs); sd = statistics.pstdev(xs) or 1e-9
        return {"mean": round(mu, 4), "t": round(mu / sd * math.sqrt(len(xs)), 2),
                "hit": round(sum(1 for x in xs if x > 0) / len(xs), 3), "n": len(xs)}

    n = len(ls_rets)
    ann = lambda xs: (math.prod(1 + x for x in xs)) ** (12 / len(xs)) - 1
    ls_ann = ann(ls_rets); mkt_ann = ann(mkt_rets)
    ls_sd = statistics.pstdev(ls_rets) * math.sqrt(12)
    sharpe = ls_ann / ls_sd if ls_sd else 0.0
    # cost model: one-way cost 20bps, monthly two-sided turnover on both legs
    avg_to = statistics.mean(turnover) if turnover else 0.0
    cost_drag = avg_to * 2 * 2 * 0.0020 * 12          # legs*sides*cost*months
    curve, eq = [], 1.0
    for i in range(n):
        eq *= (1 + ls_rets[i]); curve.append((months[i], eq))
    dd, peak = 0.0, 1.0
    for _, v in curve:
        peak = max(peak, v); dd = min(dd, v / peak - 1)
    out = {
        "as_of": date.today().isoformat(), "universe": len(UNIVERSE),
        "loaded": len(data), "start": start_year, "months": n,
        "ic": {f: icstats(v) for f, v in ics.items()},
        "quintile_ls": {"ann_gross": round(ls_ann, 4), "vol": round(ls_sd, 4),
                        "sharpe_gross": round(sharpe, 2),
                        "max_dd": round(dd, 3),
                        "avg_monthly_turnover": round(avg_to, 3),
                        "est_cost_drag": round(cost_drag, 4),
                        "ann_net_est": round(ls_ann - cost_drag, 4)},
        "market_ew_ann": round(mkt_ann, 4),
        "failed": len(failed),
        "warning": "Survivorship-biased universe (2026 list backtested into the past): treat levels as upper bounds; factor comparisons are more reliable than absolute spreads."
    }
    print(json.dumps(out, indent=2))
    if html_out:
        render(html_out, out, curve, ics, months)
        print("report -> " + html_out, file=sys.stderr)
    return out

# ------------------------------------------------------------------ report

def render(path, out, curve, ics, months):
    W, H, L, R, T, B = 900, 320, 60, 880, 24, 280
    vals = [v for _, v in curve]
    lo, hi = min(vals) * 0.95, max(vals) * 1.05
    X = lambda i: L + (R - L) * i / max(1, len(curve) - 1)
    Y = lambda v: B - (B - T) * (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    pts = " ".join("%.1f,%.1f" % (X(i), Y(v)) for i, (_, v) in enumerate(curve))
    yrs = sorted({mk // 12 for mk in months})
    xt = "".join('<text x="%.1f" y="300" text-anchor="middle" class="al">%d</text>'
                 % (X(months.index(next(m for m in months if m // 12 == y))), y)
                 for y in yrs if y % 2 == 0)
    gy = "".join('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="gr"/><text x="%d" y="%.1f" text-anchor="end" class="al">%.1fx</text>'
                 % (L, R, Y(v), Y(v), L - 6, Y(v) + 4, v)
                 for v in (1, 2, 4, 8) if lo <= v <= hi)
    ictab = ""
    for f in ("VALUE", "MOM", "GROWTH", "LOWVOL", "REVGAP", "COMPOSITE"):
        s = out["ic"][f]
        if s:
            ictab += ("<tr><td><b>%s</b></td><td>%.3f</td><td>%.2f</td><td>%.0f%%</td><td>%d</td></tr>"
                      % (f, s["mean"], s["t"], s["hit"] * 100, s["n"]))
    q = out["quintile_ls"]
    html = """<title>Cross-Section Lab</title>
<style>
:root{--paper:#F4F6F2;--card:#FFF;--ink:#1E2622;--muted:#5B6862;--accent:#1B6B4C;
--line:#D8DED8;--ls:#B8C2BA;--danger:#A33B32;--dangers:#A33B3212;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,Menlo,monospace;}
@media (prefers-color-scheme:dark){:root{--paper:#121714;--card:#1A211D;--ink:#DFE6E1;--muted:#9AA8A0;
--accent:#4FC08D;--line:#29322C;--ls:#3A463E;--danger:#DE8A80;--dangers:#DE8A801A;}}
:root[data-theme="dark"]{--paper:#121714;--card:#1A211D;--ink:#DFE6E1;--muted:#9AA8A0;
--accent:#4FC08D;--line:#29322C;--ls:#3A463E;--danger:#DE8A80;--dangers:#DE8A801A;}
:root[data-theme="light"]{--paper:#F4F6F2;--card:#FFF;--ink:#1E2622;--muted:#5B6862;
--accent:#1B6B4C;--line:#D8DED8;--ls:#B8C2BA;--danger:#A33B32;--dangers:#A33B3212;}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.6;margin:0}
.w{max-width:920px;margin:0 auto;padding:36px 16px 70px}
h1{font-size:28px;margin:0 0 6px}.ey{font-family:var(--mono);font-size:11px;letter-spacing:.15em;
text-transform:uppercase;color:var(--accent);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--ls);border-radius:10px;padding:16px 18px;margin:14px 0}
svg{max-width:100%%;height:auto;display:block}.gr{stroke:var(--line)}.al{font-size:10.5px;fill:var(--muted)}
table{border-collapse:collapse;width:100%%;font-size:13px}
th{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
padding:6px 10px;border-bottom:1.5px solid var(--ls)}
td{padding:6px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.warn{background:var(--dangers);border-left:3px solid var(--danger);padding:10px 14px;font-size:13px;margin:12px 0}
.note{color:var(--muted);font-size:12.5px}
</style>
<div class="w">
<p class="ey">gates-engine v0.3 &middot; cross-sectional factor layer &middot; %s</p>
<h1>Cross-Section Lab</h1>
<p class="note">%d-name curated large-cap universe (%d loaded) &middot; monthly since %d &middot; %d months of non-overlapping cross-sections.</p>
<div class="warn"><b>SURVIVORSHIP BIAS, LOUDLY:</b> %s</div>
<div class="card"><b>Quintile long-short (Q5&minus;Q1, equal weight, gross)</b> &mdash; cumulative growth of $1, log scale
<svg viewBox="0 0 %d %d" role="img" aria-label="Cumulative long-short equity curve">%s%s
<polyline points="%s" fill="none" stroke="var(--accent)" stroke-width="2"/></svg>
<table><tr><th>ann. gross</th><th>vol</th><th>Sharpe (gross)</th><th>max DD</th><th>turnover/mo</th><th>cost drag est.</th><th>ann. net est.</th><th>EW market</th></tr>
<tr><td>%.1f%%</td><td>%.1f%%</td><td>%.2f</td><td>%.0f%%</td><td>%.0f%%</td><td>%.1f%%</td><td>%.1f%%</td><td>%.1f%%</td></tr></table></div>
<div class="card"><b>Monthly cross-sectional Spearman IC</b> (t-stats are legitimate here: months don't overlap)
<table><tr><th>factor</th><th>mean IC</th><th>t-stat</th><th>hit rate</th><th>months</th></tr>%s</table>
<p class="note">|t| &gt; 2 &asymp; conventionally significant. Composite = equal-weight z of available factors. REVGAP is the single-name OU signal cross-sectionalized.</p></div>
<p class="note">gates-engine v0.3 &middot; data: SEC EDGAR XBRL + Yahoo, 45-day pub lag &middot; GAAP EPS &middot; split-basis guard: E/P-family factors skip months whose TTM EPS predates a name's last stock split &middot; not investment advice &middot; costs modeled naively at 20bps one-way.</p>
</div>""" % (out["as_of"], out["universe"], out["loaded"], out["start"], out["months"],
             out["warning"], W, H, gy, xt, pts,
             q["ann_gross"] * 100, q["vol"] * 100, q["sharpe_gross"], q["max_dd"] * 100,
             q["avg_monthly_turnover"] * 100, q["est_cost_drag"] * 100,
             q["ann_net_est"] * 100, out["market_ew_ann"] * 100, ictab)
    with open(path, "w") as f:
        f.write(html)

if __name__ == "__main__":
    flags = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a}
    run(start_year=int(flags.get("--start", START_DEFAULT)),
        html_out=flags.get("--html", "reports/xsection.html"))
