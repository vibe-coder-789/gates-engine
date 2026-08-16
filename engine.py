#!/usr/bin/env python3
"""
gates-engine: fundamental-anchored stochastic price scenarios + walk-forward backtest.

    python3 engine.py EME [--horizon 5] [--paths 8000] [--bt-h 3] [--no-backtest]

Model
-----
ln P_t = ln EPS_ttm(t) + ln M_t

  EPS: geometric drift with quarterly shocks     d lnE = g dt + sigma_E dW1
  M:   Ornstein-Uhlenbeck mean reversion         d lnM = kappa (lnM_bar - lnM) dt + sigma_M dW2
  corr(dW1, dW2) = rho   (estimated; typically negative for cyclicals)

All parameters estimated per ticker from SEC EDGAR XBRL (EPS, quarterly frames,
with a 45-day publication lag to avoid lookahead) and Yahoo monthly prices.
Scenario anchors (bear/base/bull) use the 25/50/75th percentiles of the stock's
own rolling-3y EPS CAGR distribution and of its own historical P/E distribution.
Monte Carlo gives the fan; the walk-forward backtest scores both the central
forecast (IC, hit rate) and the distribution (band coverage / PIT calibration)
against GBM and constant-return baselines.

Data: SEC EDGAR (free, stable, no key) + Yahoo chart API (free, no key).
Cache: ./.cache (prices 1 day, fundamentals 1 day, ticker map 30 days).

HONESTY NOTES (also printed in reports):
- Backtest origins overlap (monthly origins, multi-year horizon) -> observations
  are heavily autocorrelated; effective sample is ~n/ (12*H). Treat IC as
  descriptive, not significant.
- Multiple mean-reversion systematically fades re-ratings; the backtest will
  show exactly where that failed. That is a feature of the report, not a bug.
- Not investment advice. Point-in-time fundamentals only as good as XBRL frames.
"""

import json, math, os, random, statistics, sys, time, urllib.request
from datetime import date, datetime, timedelta

UA = {"User-Agent": os.environ.get("GATES_CONTACT", "gates-engine/0.2 (github.com/vibe-coder-789/gates-engine)")}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
PUB_LAG_DAYS = 45          # EPS for a quarter assumed known 45d after quarter end
MIN_FIT_MONTHS = 48        # minimum history to fit the model at a backtest origin
EPS_CLIP = (-0.15, 0.30)   # clip scenario growth to sane band

# ---------------------------------------------------------------- data layer

def _fetch(url, ttl_hours, tag):
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, tag + ".json")
    if os.path.exists(key) and (time.time() - os.path.getmtime(key)) < ttl_hours * 3600:
        with open(key) as f:
            return json.load(f)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    with open(key, "w") as f:
        json.dump(data, f)
    time.sleep(0.15)  # politeness (EDGAR fair-use)
    return data

def ticker_to_cik(ticker):
    m = _fetch("https://www.sec.gov/files/company_tickers.json", 720, "tickers")
    for row in m.values():
        if row["ticker"].upper() == ticker.upper():
            return "%010d" % row["cik_str"], row["title"]
    raise SystemExit("ticker %s not found in EDGAR map" % ticker)

def quarterly_eps(cik, ticker):
    """{quarter-end date: diluted EPS}, fiscal-year-safe.
    Built from raw XBRL durations (not calendar frames): quarterly = 80-100d
    periods; fiscal-Q4 synthesized as FY(350-380d) minus its three interior
    quarters. Handles off-calendar fiscal years (e.g. Micron's Aug year-end)."""
    d = _fetch("https://data.sec.gov/api/xbrl/companyconcept/CIK%s/us-gaap/EarningsPerShareDiluted.json"
               % cik, 24, ticker.lower() + "_eps")
    q, ann = {}, {}
    for e in d["units"].get("USD/shares", []):
        if "start" not in e or "end" not in e or e.get("val") is None:
            continue
        s = date.fromisoformat(e["start"]); en = date.fromisoformat(e["end"])
        dur = (en - s).days
        if 80 <= dur <= 100:
            q[en] = e["val"]                    # later filings overwrite (restated)
        elif 350 <= dur <= 380:
            ann[(s, en)] = e["val"]
    for (s, en), fy_val in ann.items():
        inside = [k for k in q if s < k <= en]
        if en in q:
            continue                            # Q4 already reported explicitly
        if len(inside) == 3:
            q[en] = fy_val - sum(q[k] for k in inside)
    return {k.isoformat(): v for k, v in sorted(q.items())}

def quarterly_dps(cik, ticker):
    """Best-effort dividends/share per quarter; {} if concept absent."""
    for concept in ("CommonStockDividendsPerShareDeclared",
                    "CommonStockDividendsPerShareCashPaid"):
        try:
            d = _fetch("https://data.sec.gov/api/xbrl/companyconcept/CIK%s/us-gaap/%s.json"
                       % (cik, concept), 24, ticker.lower() + "_dps_" + concept[:20])
            q, ann = {}, {}
            for e in d["units"].get("USD/shares", []):
                fr = e.get("frame") or ""
                if fr.startswith("CY") and "Q" in fr and len(fr) == 8:
                    q[(int(fr[2:6]), int(fr[7]))] = (e["end"], e["val"])
                elif fr.startswith("CY") and len(fr) == 6:
                    ann[int(fr[2:6])] = e["val"]
            for y, fy_val in ann.items():
                if (y, 4) not in q and all((y, k) in q for k in (1, 2, 3)):
                    q[(y, 4)] = ("%d-12-31" % y, fy_val - sum(q[(y, k)][1] for k in (1, 2, 3)))
            out = {end: val for (_, _), (end, val) in sorted(q.items())}
            if out:
                return dict(sorted(out.items()))
        except Exception:
            continue
    return {}

def monthly_prices(ticker):
    """[(date, close, adjclose)] month bars from Yahoo; drops current partial month."""
    d = _fetch("https://query1.finance.yahoo.com/v8/finance/chart/%s?range=25y&interval=1mo"
               % ticker.upper(), 24, ticker.lower() + "_px")
    r = d["chart"]["result"][0]
    ts = r["timestamp"]
    close = r["indicators"]["quote"][0]["close"]
    adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", close)
    rows = []
    for i in range(len(ts)):
        if close[i] is None:
            continue
        dt = datetime.utcfromtimestamp(ts[i]).date()
        rows.append((dt, close[i], adj[i] if adj[i] is not None else close[i]))
    today = date.today()
    if rows and rows[-1][0].year == today.year and rows[-1][0].month == today.month:
        rows[-1] = (rows[-1][0], rows[-1][1], rows[-1][2])  # keep partial as 'current'
    return rows

# ------------------------------------------------------------- series builder

def month_end(d):
    nxt = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return nxt - timedelta(days=1)

def build_panel(prices, eps_q, dps_q):
    """Monthly panel rows: dict(date, px, adj, eps_ttm, dps_ttm, m) with pub lag."""
    q_ends = sorted(date.fromisoformat(k) for k in eps_q)
    panel = []
    for (dt, px, adj) in prices:
        me = month_end(dt)
        known = [q for q in q_ends if q + timedelta(days=PUB_LAG_DAYS) <= me]
        if len(known) < 4:
            continue
        last4 = known[-4:]
        # require contiguity (~ within 3 quarters span check)
        if (last4[-1] - last4[0]).days > 310:
            continue
        eps_ttm = sum(eps_q[q.isoformat()] for q in last4)
        dps_ttm = sum(dps_q.get(q.isoformat(), 0.0) for q in last4) if dps_q else 0.0
        m = px / eps_ttm if eps_ttm > 0 else None
        panel.append({"date": me, "px": px, "adj": adj,
                      "eps": eps_ttm, "dps": dps_ttm, "m": m})
    return panel

# ------------------------------------------------------------------ estimation

def pctl(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f, c = int(math.floor(k)), int(math.ceil(k))
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)

def fit_ou(lnm):
    """AR(1) on monthly lnM -> (kappa/yr, lnM_bar, sigma_M/yr, halflife_yr, ok)."""
    x = lnm[:-1]; y = lnm[1:]
    n = len(x)
    if n < 24:
        return None
    mx, my = sum(x)/n, sum(y)/n
    sxx = sum((a-mx)**2 for a in x)
    if sxx == 0:
        return None
    b = sum((x[i]-mx)*(y[i]-my) for i in range(n)) / sxx
    a = my - b*mx
    resid = [y[i] - (a + b*x[i]) for i in range(n)]
    s = statistics.pstdev(resid)
    if not (0 < b < 0.9995):           # no reversion detected
        return {"ok": False, "b": b}
    kappa = -12.0 * math.log(b)
    lnbar = a / (1 - b)
    sigma = s * math.sqrt(2*kappa / (1 - b*b)) if b*b < 1 else s*math.sqrt(12)
    return {"ok": True, "kappa": kappa, "lnbar": lnbar, "sigma": sigma,
            "half_life": math.log(2)/kappa, "b": b, "resid_sd_m": s}

def estimate(panel):
    """All model parameters from a panel (used full-sample and walk-forward)."""
    # multiple hygiene: near-zero-EPS episodes create P/E spikes that are noise,
    # not signal — drop multiples outside a sane band before fitting the OU
    ms = [r["m"] for r in panel if r["m"] and 3.0 <= r["m"] <= 75.0]
    if len(ms) < MIN_FIT_MONTHS:
        return None
    lo, hi = pctl(ms, 0.02), pctl(ms, 0.98)
    lnm = [math.log(min(max(m, lo), hi)) for m in ms]
    ou = fit_ou(lnm)
    if ou is None:
        return None
    if ou.get("ok") and ou["sigma"] > 0.60:      # cap pathological vol, flag it
        ou["sigma"] = 0.60
        ou["sigma_capped"] = True
    # quarterly EPS-ttm growth series (every 3rd month to approximate quarters)
    eps_series = [r["eps"] for r in panel if r["eps"] > 0]
    dlnE_q = []
    for i in range(3, len(eps_series), 3):
        if eps_series[i-3] > 0 and eps_series[i] > 0:
            dlnE_q.append(math.log(eps_series[i]/eps_series[i-3]))
    # rolling 3y CAGR distribution for scenario growth anchors
    cagr3 = []
    for i in range(36, len(eps_series)):
        if eps_series[i-36] > 0 and eps_series[i] > 0:
            cagr3.append((eps_series[i]/eps_series[i-36])**(1/3.0) - 1)
    if not cagr3:
        return None
    clip = lambda g: min(max(g, EPS_CLIP[0]), EPS_CLIP[1])
    g25, g50, g75 = (clip(pctl(cagr3, p)) for p in (0.25, 0.50, 0.75))
    sigE = (statistics.pstdev(dlnE_q) * 2.0) if len(dlnE_q) > 8 else 0.15  # ->annual
    # shock correlation (quarterly)
    lnm_q = lnm[::3]
    dlnM_q = [lnm_q[i]-lnm_q[i-1] for i in range(1, len(lnm_q))]
    k = min(len(dlnE_q), len(dlnM_q))
    rho = 0.0
    if k > 8:
        a, bq = dlnE_q[-k:], dlnM_q[-k:]
        ma, mb = sum(a)/k, sum(bq)/k
        num = sum((a[i]-ma)*(bq[i]-mb) for i in range(k))
        den = math.sqrt(sum((v-ma)**2 for v in a) * sum((v-mb)**2 for v in bq))
        rho = num/den if den > 0 else 0.0
    m_p = {p: pctl(ms, p) for p in (0.25, 0.50, 0.75)}
    # --- regime detection: is the recent multiple regime far from the long mean?
    recent = lnm[-36:] if len(lnm) >= 36 else lnm
    rec_mean = sum(recent)/len(recent)
    full_sd = statistics.pstdev(lnm) or 1e-6
    regime_z = (rec_mean - ou["lnbar"]) / full_sd if ou.get("ok") else 0.0
    rerating = abs(regime_z) > 1.0
    if ou.get("ok") and rerating:
        # blend the reversion anchor halfway toward the recent regime:
        # full reversion to a decade-old mean is exactly what failed in backtests
        ou = dict(ou)
        ou["lnbar_raw"] = ou["lnbar"]
        ou["lnbar"] = 0.5*ou["lnbar"] + 0.5*rec_mean
    # --- parameter uncertainty: residual-bootstrap AR(1) refits (predictive mixing)
    boots = []
    if ou.get("ok"):
        b = ou["b"]; lnbar = ou["lnbar"]; sd = ou.get("resid_sd_m", 0.04)
        rngb = random.Random(11)
        n = len(lnm)
        for _ in range(48):
            path = [lnm[0]]
            for _i in range(1, n):
                path.append(lnbar*(1-b) + b*path[-1] + rngb.gauss(0, sd))
            f = fit_ou(path)
            if f and f.get("ok"):
                if f["sigma"] > 0.60: f["sigma"] = 0.60
                boots.append({"kappa": f["kappa"], "lnbar": f["lnbar"], "sigma": f["sigma"]})
    return {"ou": ou, "g": {"bear": g25, "base": g50, "bull": g75},
            "sigE": sigE, "rho": max(-0.9, min(0.9, rho)),
            "m_pct": m_p, "n_months": len(ms),
            "regime_z": round(regime_z, 2), "rerating": rerating,
            "boots": boots}

# ----------------------------------------------------------------- simulation

def simulate(px0, eps0, m0, est, horizon_y, n_paths, g_override=None, seed=7):
    """Monte Carlo fan. Returns monthly percentile curves + terminal returns."""
    rng = random.Random(seed)
    steps = int(horizon_y * 12)
    ou = est["ou"]
    g = est["g"]["base"] if g_override is None else g_override
    dt = 1/12.0
    boots = est.get("boots") or []
    sE_m = est["sigE"] * math.sqrt(dt)
    rho = est["rho"]
    curves = [[] for _ in range(steps)]
    term = []
    for _p in range(n_paths):
        # predictive mixing: each path draws one bootstrap parameter set,
        # so the fan carries estimation risk, not just shock risk
        if boots and ou.get("ok"):
            ps = boots[rng.randrange(len(boots))] if rng.random() < 0.7 else \
                 {"kappa": ou["kappa"], "lnbar": ou["lnbar"], "sigma": ou["sigma"]}
            e_k = math.exp(-ps["kappa"]*dt)
            tr_sd = ps["sigma"] * math.sqrt((1 - e_k*e_k) / (2*ps["kappa"]))
            lnbar = ps["lnbar"]
        elif ou.get("ok"):
            e_k = math.exp(-ou["kappa"]*dt)
            tr_sd = ou["sigma"] * math.sqrt((1 - e_k*e_k) / (2*ou["kappa"]))
            lnbar = ou["lnbar"]
        else:
            e_k, tr_sd, lnbar = 1.0, est["ou"].get("resid_sd_m", 0.05), math.log(m0)
        lnE, lnM = math.log(eps0), math.log(m0)
        for t in range(steps):
            z1 = rng.gauss(0, 1)
            z2 = rho*z1 + math.sqrt(max(0.0, 1-rho*rho))*rng.gauss(0, 1)
            lnE += g*dt + sE_m*z1
            lnM = lnbar + (lnM - lnbar)*e_k + tr_sd*z2
            curves[t].append(math.exp(lnE + lnM))
        term.append(curves[-1][-1])
    fan = {p: [pctl(c, p) for c in curves] for p in (0.05, 0.25, 0.50, 0.75, 0.95)}
    ann = sorted((v/px0)**(1/horizon_y) - 1 for v in term)
    return {"fan": fan, "terminal_px": sorted(term), "ann_returns": ann}

def scenario_lines(eps0, m0, est, horizon_y):
    """Deterministic anchors: EPS at g_x, multiple linearly -> its percentile."""
    steps = int(horizon_y*12)
    out = {}
    for name, gp, mp in (("bear", "bear", 0.25), ("base", "base", 0.50), ("bull", "bull", 0.75)):
        g = est["g"][gp]; m_t = est["m_pct"][mp]
        pts = []
        for t in range(1, steps+1):
            f = t/steps
            eps = eps0 * (1+g)**(t/12.0)
            mult = m0 + (m_t - m0)*f
            pts.append(eps*mult)
        out[name] = {"g": g, "m_exit": m_t, "path": pts}
    return out

# ------------------------------------------------------------------- backtest

def backtest(panel, bt_h, n_small=1500):
    """Walk-forward: at each monthly origin fit on past only, predict H-yr dist,
       compare with realized (total-return via adjclose). Plus baselines."""
    H = bt_h; steps_fwd = H*12
    res = []
    for i in range(MIN_FIT_MONTHS, len(panel)-steps_fwd):
        sub = panel[:i+1]
        row = sub[-1]
        if not row["m"] or row["m"] <= 0:
            continue
        est = estimate(sub)
        if est is None or not est["ou"]["ok"]:
            continue
        sim = simulate(row["px"], row["eps"], row["m"], est, H, n_small,
                       seed=100+i)
        ann = sim["ann_returns"]
        fut = panel[i+steps_fwd]
        realized = (fut["adj"]/row["adj"])**(1/H) - 1
        # GBM baseline from trailing adj returns
        rets = [math.log(sub[j]["adj"]/sub[j-1]["adj"]) for j in range(1, len(sub))]
        mu = statistics.mean(rets)*12; sd = statistics.pstdev(rets)*math.sqrt(12)
        # ensemble: inverse-MAE weights learned from PAST origins only
        pm = pctl(ann, 0.50)
        if len(res) >= 24:
            eps_ = 1e-4
            em = sum(abs(r0["pred_med"] - r0["realized"]) for r0 in res)/len(res)
            eg = sum(abs(r0["gbm_mu"] - r0["realized"]) for r0 in res)/len(res)
            ec = sum(abs(0.08 - r0["realized"]) for r0 in res)/len(res)
            wm, wg, wc = 1/(em+eps_), 1/(eg+eps_), 1/(ec+eps_)
            tot = wm+wg+wc
            comb = (wm*pm + wg*mu + wc*0.08)/tot
            wts = [round(wm/tot, 2), round(wg/tot, 2), round(wc/tot, 2)]
        else:
            comb = (pm + mu + 0.08)/3.0
            wts = [0.33, 0.33, 0.33]
        res.append({"date": row["date"].isoformat(),
                    "pred_med": pm, "pred_comb": comb, "weights": wts,
                    "p05": pctl(ann, 0.05), "p25": pctl(ann, 0.25),
                    "p75": pctl(ann, 0.75), "p95": pctl(ann, 0.95),
                    "gbm_mu": mu, "gbm_sd": sd, "realized": realized})
    return res

def bt_metrics(res, H):
    if len(res) < 12:
        return {"n": len(res), "note": "insufficient origins"}
    pred = [r["pred_med"] for r in res]; real = [r["realized"] for r in res]
    gbm = [r["gbm_mu"] for r in res]
    def pearson(a, b):
        n = len(a); ma, mb = sum(a)/n, sum(b)/n
        num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
        den = math.sqrt(sum((x-ma)**2 for x in a)*sum((x-mb)**2 for x in b))
        return num/den if den else 0.0
    def spearman(a, b):
        ra = {v: i for i, v in enumerate(sorted(a))}
        rb = {v: i for i, v in enumerate(sorted(b))}
        return pearson([ra[v] for v in a], [rb[v] for v in b])
    comb = [r["pred_comb"] for r in res]
    cov90 = sum(1 for r in res if r["p05"] <= r["realized"] <= r["p95"])/len(res)
    cov50 = sum(1 for r in res if r["p25"] <= r["realized"] <= r["p75"])/len(res)
    hit = sum(1 for r in res if (r["pred_med"] > 0.06) == (r["realized"] > 0.06))/len(res)
    hit_c = sum(1 for r in res if (r["pred_comb"] > 0.06) == (r["realized"] > 0.06))/len(res)
    mae_m = sum(abs(pred[i]-real[i]) for i in range(len(res)))/len(res)
    mae_g = sum(abs(gbm[i]-real[i]) for i in range(len(res)))/len(res)
    mae_c = sum(abs(0.08-r) for r in real)/len(res)
    mae_e = sum(abs(comb[i]-real[i]) for i in range(len(res)))/len(res)
    return {"n": len(res), "n_eff": round(len(res)/(12*H), 1),
            "ic_pearson": round(pearson(pred, real), 3),
            "ic_spearman": round(spearman(pred, real), 3),
            "ic_gbm": round(pearson(gbm, real), 3),
            "ic_comb": round(pearson(comb, real), 3),
            "cov90": round(cov90, 3), "cov50": round(cov50, 3),
            "hit_6pct": round(hit, 3), "hit_comb": round(hit_c, 3),
            "mae_model": round(mae_m, 4), "mae_gbm": round(mae_g, 4),
            "mae_const8": round(mae_c, 4), "mae_comb": round(mae_e, 4),
            "final_weights": res[-1]["weights"]}

# ------------------------------------------------------------------ reporting
# (SVG chart builders kept minimal; palette = session-validated pairs)

def _scale(vals, lo_px, hi_px, lo_v, hi_v):
    rng = (hi_v - lo_v) or 1.0
    return lambda v: lo_px + (hi_px - lo_px) * ((v - lo_v) / rng)

def fan_chart_svg(panel, sim, scen, horizon_y, ticker):
    hist = [(r["date"], r["px"]) for r in panel][-120:]
    steps = horizon_y*12
    last = panel[-1]["date"]
    def add_m(d, k):
        y = d.year + (d.month - 1 + k)//12
        m = (d.month - 1 + k) % 12 + 1
        return date(y, m, 1)
    fut = [add_m(last, k+1) for k in range(steps)]
    all_y = ([p for _, p in hist] + sim["fan"][0.95] + sim["fan"][0.05]
             + scen["bull"]["path"] + scen["bear"]["path"])
    lo, hi = min(all_y)*0.9, max(all_y)*1.06
    t0, t1 = hist[0][0], fut[-1]
    X = _scale(None, 70, 890, t0.toordinal(), t1.toordinal())
    Y = _scale(None, 330, 30, lo, hi)
    def pl(points):
        return " ".join("%.1f,%.1f" % (X(d.toordinal()), Y(v)) for d, v in points)
    band = ("M " + " L ".join("%.1f %.1f" % (X(fut[i].toordinal()), Y(sim["fan"][0.95][i])) for i in range(steps))
            + " L " + " L ".join("%.1f %.1f" % (X(fut[i].toordinal()), Y(sim["fan"][0.05][i])) for i in range(steps-1, -1, -1)) + " Z")
    band50 = ("M " + " L ".join("%.1f %.1f" % (X(fut[i].toordinal()), Y(sim["fan"][0.75][i])) for i in range(steps))
              + " L " + " L ".join("%.1f %.1f" % (X(fut[i].toordinal()), Y(sim["fan"][0.25][i])) for i in range(steps-1, -1, -1)) + " Z")
    gys = []
    v = lo
    stepv = (hi-lo)/5
    for k in range(6):
        gys.append(lo + k*stepv)
    grid = "".join('<line x1="70" x2="890" y1="%.1f" y2="%.1f" class="grid"/><text x="62" y="%.1f" text-anchor="end" class="axis-lbl">%d</text>'
                   % (Y(g), Y(g), Y(g)+4, round(g)) for g in gys)
    yrs = sorted({d.year for d, _ in hist} | {d.year for d in fut})
    xt = "".join('<text x="%.1f" y="350" text-anchor="middle" class="axis-lbl">%d</text>'
                 % (X(date(y,1,1).toordinal()), y) for y in yrs if y % 2 == 0)
    lastpt = "%.1f,%.1f" % (X(last.toordinal()), Y(panel[-1]["px"]))
    svg = ['<svg viewBox="0 0 960 370" role="img" aria-label="Price history and five-year scenario fan for %s: Monte Carlo percentile bands with bear, base, and bull deterministic anchors.">' % ticker]
    svg.append(grid); svg.append(xt)
    svg.append('<path d="%s" fill="var(--band90)" stroke="none"/>' % band)
    svg.append('<path d="%s" fill="var(--band50)" stroke="none"/>' % band50)
    svg.append('<polyline points="%s" fill="none" stroke="currentColor" stroke-width="1.8"/>' % pl(hist))
    med = list(zip(fut, sim["fan"][0.50]))
    svg.append('<polyline points="%s" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="5 4"/>' % pl(med))
    for name, color in (("bull", "var(--s-bull)"), ("bear", "var(--s-bear)")):
        pts = list(zip(fut, scen[name]["path"]))
        svg.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (pl(pts), color))
        svg.append('<text x="%.1f" y="%.1f" class="lbl" fill="%s">%s %.0f%%/yr, exit %.0fx</text>'
                   % (X(fut[-1].toordinal())-236, Y(scen[name]["path"][-1])-6, color,
                      name.upper(), scen[name]["g"]*100, scen[name]["m_exit"]))
    svg.append('<circle cx="%s" r="4" fill="currentColor"/>' % lastpt.replace(",", '" cy="'))
    svg.append('<text x="%.1f" y="%.1f" class="lbl">MC median (dashed) · shaded 5–95 / 25–75</text>'
               % (X(fut[2].toordinal()), 46))
    svg.append('</svg>')
    return "".join(svg)

def backtest_svg(res):
    if len(res) < 12:
        return "<p class='note'>Backtest skipped: insufficient origins.</p>"
    xs = [r["pred_med"] for r in res]; ys = [r["realized"] for r in res]
    lo = min(min(xs), min(ys)); hi = max(max(xs), max(ys))
    pad = (hi-lo)*0.08; lo -= pad; hi += pad
    X = _scale(None, 70, 500, lo, hi); Y = _scale(None, 330, 30, lo, hi)
    pts = "".join('<circle cx="%.1f" cy="%.1f" r="3.2" fill="var(--s-base)" fill-opacity="0.55"><title>%s: pred %.1f%%, realized %.1f%%</title></circle>'
                  % (X(xs[i]), Y(ys[i]), res[i]["date"], xs[i]*100, ys[i]*100) for i in range(len(res)))
    ticks = ""
    for k in range(5):
        v = lo + k*(hi-lo)/4
        ticks += ('<line x1="70" x2="500" y1="%.1f" y2="%.1f" class="grid"/>'
                  '<text x="62" y="%.1f" text-anchor="end" class="axis-lbl">%.0f%%</text>'
                  '<text x="%.1f" y="350" text-anchor="middle" class="axis-lbl">%.0f%%</text>'
                  % (Y(v), Y(v), Y(v)+4, v*100, X(v), v*100))
    diag = '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="var(--muted)" stroke-dasharray="4 4"/>' % (X(lo), Y(lo), X(hi), Y(hi))
    return ('<svg viewBox="0 0 540 372" role="img" aria-label="Backtest scatter: model-predicted median annualized return versus realized, with the diagonal marking perfect prediction.">'
            + ticks + diag + pts
            + '<text x="285" y="366" text-anchor="middle" class="axis-lbl">model predicted median (ann.)</text>'
            + '<text x="16" y="180" class="axis-lbl" transform="rotate(-90 16 180)">realized (ann., total return)</text></svg>')

REPORT_CSS = """
:root{--paper:#F4F6F2;--card:#FFFFFF;--ink:#1E2622;--muted:#5B6862;--accent:#1B6B4C;
--line:#D8DED8;--line-strong:#B8C2BA;--band90:#2E5AA818;--band50:#2E5AA830;
--s-bull:#2E5AA8;--s-bear:#B47222;--s-base:#1B6B4C;--danger:#A33B32;--danger-soft:#A33B3212;
--mono:ui-monospace,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme: dark){:root{--paper:#121714;--card:#1A211D;--ink:#DFE6E1;--muted:#9AA8A0;
--accent:#4FC08D;--line:#29322C;--line-strong:#3A463E;--band90:#5B8FE01E;--band50:#5B8FE038;
--s-bull:#5B8FE0;--s-bear:#C07E2E;--s-base:#4FC08D;--danger:#DE8A80;--danger-soft:#DE8A801A;}}
:root[data-theme="dark"]{--paper:#121714;--card:#1A211D;--ink:#DFE6E1;--muted:#9AA8A0;
--accent:#4FC08D;--line:#29322C;--line-strong:#3A463E;--band90:#5B8FE01E;--band50:#5B8FE038;
--s-bull:#5B8FE0;--s-bear:#C07E2E;--s-base:#4FC08D;--danger:#DE8A80;--danger-soft:#DE8A801A;}
:root[data-theme="light"]{--paper:#F4F6F2;--card:#FFFFFF;--ink:#1E2622;--muted:#5B6862;--accent:#1B6B4C;
--line:#D8DED8;--line-strong:#B8C2BA;--band90:#2E5AA818;--band50:#2E5AA830;
--s-bull:#2E5AA8;--s-bear:#B47222;--s-base:#1B6B4C;--danger:#A33B32;--danger-soft:#A33B3212;}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.6;margin:0;}
.wrap{max-width:960px;margin:0 auto;padding:40px 18px 70px;}
h1{font-size:28px;margin:0 0 6px;} h2{font-size:19px;margin:34px 0 8px;}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin:0 0 10px;}
.meta{color:var(--muted);font-size:13px;}
.card{background:var(--card);border:1px solid var(--line-strong);border-radius:10px;padding:16px 18px;margin:14px 0;}
svg{max-width:100%;height:auto;display:block;} .grid{stroke:var(--line);stroke-width:1;}
.axis-lbl{font-size:10.5px;fill:var(--muted);font-family:var(--sans);}
.lbl{font-size:11.5px;font-family:var(--sans);fill:currentColor;}
table{border-collapse:collapse;width:100%;font-size:13px;}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);padding:6px 10px;border-bottom:1.5px solid var(--line-strong);}
td{padding:6px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;}
.warn{background:var(--danger-soft);border-left:3px solid var(--danger);padding:10px 14px;font-size:13px;margin:14px 0;}
.note{color:var(--muted);font-size:12.5px;}
.mono{font-family:var(--mono);font-size:12px;}
"""

def render_report(ticker, title, panel, est, sim, scen, res, mets, horizon_y, bt_h):
    row = panel[-1]
    ou = est["ou"]
    ann = sim["ann_returns"]
    html = []
    A = html.append
    A("<title>%s Scenario Model</title>" % ticker.upper())
    A("<style>%s</style>" % REPORT_CSS)
    A('<div class="wrap">')
    A('<p class="eyebrow">gates-engine · stochastic scenario model · generated %s</p>' % date.today().isoformat())
    A("<h1>%s — %s</h1>" % (ticker.upper(), title))
    A('<p class="meta">Price $%.2f · EPS(ttm) $%.2f · P/E %.1f · data: SEC EDGAR XBRL + Yahoo (monthly, %d obs) · pub-lag %dd applied</p>'
      % (row["px"], row["eps"], row["m"], est["n_months"], PUB_LAG_DAYS))
    A('<div class="warn"><b>NOT INVESTMENT ADVICE.</b> A model of historical patterns, honestly backtested below — including where it fails. '
      'Mean-reverting multiples systematically fade re-ratings; parameters are sample-dependent; free-data quirks possible.</div>')
    A("<h2>1 · Five-year scenario fan</h2>")
    A('<div class="card">%s</div>' % fan_chart_svg(panel, sim, scen, horizon_y, ticker.upper()))
    A('<p class="note">Solid history; shaded Monte-Carlo bands (%d paths) from the coupled OU-multiple / drift-earnings process; '
      'dashed = MC median; colored lines = deterministic bear/base/bull anchors (own-history growth &amp; multiple percentiles).</p>' % len(sim["terminal_px"]))
    A("<h2>2 · Model parameters (estimated from this stock's own history)</h2>")
    A('<div class="card"><table><tr><th>Parameter</th><th>Value</th><th>Meaning</th></tr>')
    A("<tr><td>Multiple mean (OU)</td><td>%.1fx</td><td>where the P/E reverts to</td></tr>" % math.exp(ou["lnbar"]))
    A("<tr><td>Reversion half-life</td><td>%.1f yr</td><td>time for half a multiple gap to close</td></tr>" % ou["half_life"])
    A("<tr><td>Multiple vol σ_M</td><td>%.0f%%/yr</td><td>how noisy the multiple is</td></tr>" % (ou["sigma"]*100))
    A("<tr><td>EPS growth (bear/base/bull)</td><td>%.1f%% / %.1f%% / %.1f%%</td><td>25/50/75th pct of rolling 3y CAGRs</td></tr>"
      % (est["g"]["bear"]*100, est["g"]["base"]*100, est["g"]["bull"]*100))
    A("<tr><td>EPS vol σ_E</td><td>%.0f%%/yr</td><td>earnings shock size</td></tr>" % (est["sigE"]*100))
    A("<tr><td>Shock correlation ρ</td><td>%.2f</td><td>earnings vs multiple shocks</td></tr>" % est["rho"])
    A("<tr><td>Multiple percentiles</td><td>%.0fx / %.0fx / %.0fx</td><td>bear/base/bull exit anchors</td></tr>"
      % (est["m_pct"][0.25], est["m_pct"][0.50], est["m_pct"][0.75]))
    A("</table></div>")
    A("<h2>3 · Return distribution (%d-yr, annualized, MC)</h2>" % horizon_y)
    A('<div class="card"><table><tr><th>p5</th><th>p25</th><th>median</th><th>p75</th><th>p95</th><th>P(loss)</th></tr>')
    ploss = sum(1 for r in ann if r < 0)/len(ann)
    A("<tr>" + "".join("<td>%.1f%%</td>" % (pctl(ann, p)*100) for p in (0.05, 0.25, 0.50, 0.75, 0.95))
      + "<td>%.0f%%</td></tr></table></div>" % (ploss*100))
    A("<h2>4 · Walk-forward backtest (%d-yr horizon, monthly origins, past-only fits)</h2>" % bt_h)
    if isinstance(mets, dict) and "ic_pearson" in mets:
        A('<div class="card">%s</div>' % backtest_svg(res))
        A('<div class="card"><table><tr><th>Metric</th><th>Model</th><th>GBM baseline</th><th>Const 8%</th></tr>')
        A("<tr><td>IC (Pearson / Spearman)</td><td>%.2f / %.2f</td><td>%.2f</td><td>—</td></tr>"
          % (mets["ic_pearson"], mets["ic_spearman"], mets["ic_gbm"]))
        A("<tr><td>MAE (ann. return)</td><td>%.1f%%</td><td>%.1f%%</td><td>%.1f%%</td></tr>"
          % (mets["mae_model"]*100, mets["mae_gbm"]*100, mets["mae_const8"]*100))
        A("<tr><td>Coverage: 90%% band / 50%% band</td><td>%.0f%% / %.0f%%</td><td colspan=2>target ≈ 90%% / 50%%</td></tr>"
          % (mets["cov90"]*100, mets["cov50"]*100))
        A("<tr><td>Hit rate (above/below 6%%)</td><td>%.0f%%</td><td colspan=2>coin = 50%%</td></tr>" % (mets["hit_6pct"]*100))
        A("<tr><td>Origins n (effective ≈)</td><td colspan=3>%d (≈%s independent)</td></tr>" % (mets["n"], mets["n_eff"]))
        A("</table></div>")
        A('<p class="note">Origins overlap heavily (monthly origins, %d-yr windows) — treat IC as descriptive. '
          'Coverage below target ⇒ model overconfident; above ⇒ bands too wide.</p>' % bt_h)
    else:
        A('<p class="note">Backtest: %s</p>' % mets.get("note", "n/a"))
    A("<h2>5 · Where this model breaks (read before using)</h2>")
    A('<div class="warn">1) <b>Re-ratings:</b> OU reversion would have faded every great re-rating (and every collapse) — the scatter shows those misses as points far above/below the diagonal. '
      "2) <b>Regime change:</b> parameters assume the future resembles this stock's own past distributions. "
      "3) <b>Negative-EPS stretches</b> drop out of the multiple series. "
      "4) Dividends included in realized (adj-close) returns but only implicitly in forecasts for low-yield names — high-yield names are understated in the fan. "
      "5) Free-data caveats: XBRL frame gaps, Yahoo adjustments.</div>")
    A('<p class="note mono">gates-engine v0.1 · model: lnP = lnE + lnM; dlnE = g·dt + σ_E dW₁; dlnM = κ(ln M̄ − lnM)dt + σ_M dW₂; corr(dW₁,dW₂)=ρ</p>')
    A("</div>")
    return "\n".join(html)

# ------------------------------------------------------------------------ main

def run(ticker, horizon=5, paths=8000, bt_h=3, do_backtest=True):
    cik, title = ticker_to_cik(ticker)
    eps_q = quarterly_eps(cik, ticker)
    dps_q = quarterly_dps(cik, ticker)
    px = monthly_prices(ticker)
    panel = build_panel(px, eps_q, dps_q)
    if len(panel) < MIN_FIT_MONTHS:
        raise SystemExit("only %d usable months for %s — need %d" % (len(panel), ticker, MIN_FIT_MONTHS))
    est = estimate(panel)
    if est is None:
        raise SystemExit("estimation failed for %s" % ticker)
    row = panel[-1]
    if not row["m"] or row["m"] <= 0:
        raise SystemExit("%s: negative trailing EPS — this engine prices compounders only" % ticker)
    sim = simulate(row["px"], row["eps"], row["m"], est, horizon, paths)
    scen = scenario_lines(row["eps"], row["m"], est, horizon)
    res, mets = [], {"note": "skipped"}
    if do_backtest:
        res = backtest(panel, bt_h)
        mets = bt_metrics(res, bt_h)
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, ticker.lower() + ".html")
    with open(out, "w") as f:
        f.write(render_report(ticker, title, panel, est, sim, scen, res, mets, horizon, bt_h))
    ann = sim["ann_returns"]
    summary = {"ticker": ticker.upper(), "price": round(row["px"], 2),
               "eps_ttm": round(row["eps"], 2), "pe": round(row["m"], 1),
               "ou_mean_pe": round(math.exp(est["ou"]["lnbar"]), 1),
               "half_life_yr": round(est["ou"]["half_life"], 1),
               "g_base": round(est["g"]["base"], 3),
               "mc_median_ann": round(pctl(ann, 0.5), 3),
               "mc_p05": round(pctl(ann, 0.05), 3), "mc_p95": round(pctl(ann, 0.95), 3),
               "p_loss": round(sum(1 for r in ann if r < 0)/len(ann), 2),
               "backtest": mets, "report": out}
    print(json.dumps(summary, indent=2))
    return summary

def bundle(tickers, horizon=5, paths=5000, bt_h=3, out="bundle.json"):
    """Batch-run tickers -> one JSON bundle for the Terminal/Lab UI pages."""
    data = {"as_of": date.today().isoformat(), "horizon_y": horizon,
            "bt_h": bt_h, "tickers": {}, "failed": {}}
    for t in tickers:
        try:
            cik, title = ticker_to_cik(t)
            eps_q = quarterly_eps(cik, t); dps_q = quarterly_dps(cik, t)
            panel = build_panel(monthly_prices(t), eps_q, dps_q)
            est = estimate(panel)
            row = panel[-1] if panel else None
            if est is None or not row or not row["m"] or row["m"] <= 0:
                reason = ("negative trailing EPS — option, not a compounder; model declines to price it"
                          if row and row["eps"] <= 0 else "insufficient usable history (<%d months)" % MIN_FIT_MONTHS)
                data["failed"][t.upper()] = reason
                print("SKIP %s: %s" % (t, reason), file=sys.stderr)
                continue
            sim = simulate(row["px"], row["eps"], row["m"], est, horizon, paths)
            scen = scenario_lines(row["eps"], row["m"], est, horizon)
            res = backtest(panel, bt_h)
            mets = bt_metrics(res, bt_h)
            rets_now = [math.log(panel[j]["adj"]/panel[j-1]["adj"]) for j in range(1, len(panel))]
            gbm_now = {"mu": round(statistics.mean(rets_now)*12, 4),
                       "sd": round(statistics.pstdev(rets_now)*math.sqrt(12), 4)}
            dec = max(1, len(panel)//120)
            hist = [[r["date"].isoformat(), round(r["px"], 2)] for r in panel[::dec]]
            mult = [[r["date"].isoformat(), round(r["m"], 1)] for r in panel[::dec] if r["m"]]
            ann = sim["ann_returns"]
            data["tickers"][t.upper()] = {
                "title": title, "price": round(row["px"], 2),
                "eps_ttm": round(row["eps"], 2), "pe": round(row["m"], 1),
                "asof_month": row["date"].isoformat(),
                "params": {"ou_mean_pe": round(math.exp(est["ou"]["lnbar"]), 1),
                           "ou_mean_pe_raw": round(math.exp(est["ou"].get("lnbar_raw", est["ou"]["lnbar"])), 1),
                           "half_life": round(est["ou"]["half_life"], 2),
                           "sigma_m": round(est["ou"]["sigma"], 3),
                           "sigma_capped": est["ou"].get("sigma_capped", False),
                           "g": {k: round(v, 3) for k, v in est["g"].items()},
                           "sig_e": round(est["sigE"], 3), "rho": round(est["rho"], 2),
                           "m_pct": {str(k): round(v, 1) for k, v in est["m_pct"].items()},
                           "regime_z": est["regime_z"], "rerating": est["rerating"],
                           "n_months": est["n_months"], "n_boots": len(est.get("boots") or [])},
                "fan": {str(p): [round(v, 2) for v in sim["fan"][p]] for p in sim["fan"]},
                "scen": {k: {"g": round(v["g"], 3), "m_exit": round(v["m_exit"], 1),
                             "path": [round(x, 2) for x in v["path"]]} for k, v in scen.items()},
                "dist": {"p05": round(pctl(ann, .05), 3), "p25": round(pctl(ann, .25), 3),
                         "med": round(pctl(ann, .50), 3), "p75": round(pctl(ann, .75), 3),
                         "p95": round(pctl(ann, .95), 3),
                         "p_loss": round(sum(1 for r in ann if r < 0)/len(ann), 2)},
                "hist": hist, "mult": mult, "gbm_now": gbm_now,
                "bt": {"points": [[r["date"], round(r["pred_med"], 3), round(r["pred_comb"], 3),
                                   round(r["realized"], 3)] for r in res],
                       "metrics": mets}}
            print("OK   %s  px=%.2f pe=%.1f med=%.1f%%" %
                  (t, row["px"], row["m"], pctl(ann, .5)*100), file=sys.stderr)
        except SystemExit as e:
            data["failed"][t.upper()] = str(e); print("SKIP %s: %s" % (t, e), file=sys.stderr)
        except Exception as e:
            data["failed"][t.upper()] = "%s: %s" % (type(e).__name__, e)
            print("FAIL %s: %s" % (t, e), file=sys.stderr)
    with open(out, "w") as f:
        json.dump(data, f)
    print("bundle -> %s (%d ok, %d failed)" % (out, len(data["tickers"]), len(data["failed"])),
          file=sys.stderr)
    return data

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True)
             for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print(__doc__); sys.exit(1)
    if args[0] == "bundle":
        bundle(args[1:] or ["EME"],
               horizon=int(flags.get("--horizon", 5)),
               paths=int(flags.get("--paths", 5000)),
               bt_h=int(flags.get("--bt-h", 3)),
               out=str(flags.get("--out", "bundle.json")))
    else:
        run(args[0],
            horizon=int(flags.get("--horizon", 5)),
            paths=int(flags.get("--paths", 8000)),
            bt_h=int(flags.get("--bt-h", 3)),
            do_backtest=not flags.get("--no-backtest", False))
