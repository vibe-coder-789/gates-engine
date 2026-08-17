#!/usr/bin/env python3
"""
screener.py — mechanical universe discovery for the alpha book. No opinions,
no hand-picking: the tradable universe maintains itself.

    python3 screener.py [--size=150] [--out=universe.json] [--prev=universe.json]

Pipeline:
 1. Candidates = the SEC company_tickers.json top ~3x target size by file
    order (roughly cap-ranked — but only ROUGHLY: Micron sat at index 7228,
    so order alone is not trusted) UNION the previous universe (continuity:
    a discovered name stays until it fails liquidity) UNION alpha.py's
    built-in seed list.
 2. Symbol hygiene: 1-5 plain letters (skips class shares, units, warrants).
 3. Liquidity check from ~3 months of Yahoo daily bars: median daily dollar
    volume >= $50M, price >= $5, at least 55 bars.
 4. Keep the top `size` by median dollar volume.

Output JSON: {as_of, size, tickers: [...], added: [...], removed: [...]}
(diff vs --prev when given). The alpha trader loads this via --universe; the
monthly Universe Scout routine refreshes it and reports the diff by email.
Discovery is mechanical; the earnings-model veto and the LLM event veto still
apply downstream at trade time.
"""

import json, re, statistics, sys, time, urllib.request
from datetime import date

UA = {"User-Agent": "gates-screener/1.0 (github.com/vibe-coder-789/gates-engine)"}
MIN_DOLLAR_VOL = 50e6
MIN_PRICE = 5.0
MIN_BARS = 55


def fetch_json(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(4 * (attempt + 1))
    return None


def liquidity(ticker):
    d = fetch_json("https://query1.finance.yahoo.com/v8/finance/chart/%s?range=3mo&interval=1d"
                   % ticker, tries=2)
    time.sleep(0.15)
    try:
        res = d["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        dv = [q["close"][i] * q["volume"][i]
              for i in range(len(q["close"]))
              if q["close"][i] and q["volume"][i]]
        px = [c for c in q["close"] if c]
        if len(dv) < MIN_BARS or px[-1] < MIN_PRICE:
            return None
        return statistics.median(dv)
    except Exception:
        return None


def sec_map():
    """SEC ticker map, sharing engine's cache; a stale cache beats no map."""
    import os
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "tickers.json")
    if os.path.exists(cache) and (time.time() - os.path.getmtime(cache)) < 720 * 3600:
        with open(cache) as f:
            return json.load(f)
    m = fetch_json("https://www.sec.gov/files/company_tickers.json")
    if m:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w") as f:
                json.dump(m, f)
        except Exception:
            pass
        return m
    if os.path.exists(cache):                  # stale fallback
        with open(cache) as f:
            return json.load(f)
    return None


def run(size, prev_list):
    m = sec_map()
    if not m:
        raise SystemExit("SEC ticker map unavailable")
    seen, candidates = set(), []
    for i in range(len(m)):
        row = m.get(str(i))
        if not row:
            continue
        t = row["ticker"].upper()
        if t in seen or not re.fullmatch(r"[A-Z]{1,5}", t):
            continue
        seen.add(t)
        candidates.append(t)
        if len(candidates) >= size * 3:
            break
    # continuity + seeds: order-rank discovery misses real names (MU @ 7228)
    seeds = list(prev_list or [])
    try:
        import alpha as _alpha
        seeds += list(_alpha.UNIVERSE)
    except Exception:
        pass
    for t in seeds:
        t = t.upper()
        if t not in seen and re.fullmatch(r"[A-Z]{1,5}", t):
            seen.add(t)
            candidates.append(t)
    print("screening %d cap-ranked candidates..." % len(candidates), file=sys.stderr)
    scored, misses = [], []
    for i, t in enumerate(candidates):
        dv = liquidity(t)
        if dv and dv >= MIN_DOLLAR_VOL:
            scored.append((t, dv))
        elif dv is None:
            misses.append(t)
        if (i + 1) % 100 == 0:
            print("  ...%d/%d checked (%d pass)" % (i + 1, len(candidates), len(scored)),
                  file=sys.stderr)
    # second pass: a transient fetch failure must not eject a liquid name
    if misses:
        print("  retrying %d misses..." % len(misses), file=sys.stderr)
        time.sleep(10)
        for t in misses:
            dv = liquidity(t)
            if dv and dv >= MIN_DOLLAR_VOL:
                scored.append((t, dv))
    scored.sort(key=lambda x: -x[1])
    tickers = sorted(t for t, _ in scored[:size])
    prev = set(prev_list or [])
    out = {"as_of": date.today().isoformat(), "size": len(tickers),
           "tickers": tickers,
           "added": sorted(set(tickers) - prev) if prev else [],
           "removed": sorted(prev - set(tickers)) if prev else []}
    return out


if __name__ == "__main__":
    flags = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a}
    size = int(flags.get("--size", 150))
    prev = None
    if "--prev" in flags:
        try:
            with open(flags["--prev"]) as f:
                prev = json.load(f).get("tickers")
        except Exception:
            prev = None
    out = run(size, prev)
    dest = flags.get("--out", "universe.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print("universe -> %s (%d tickers, +%d/-%d)" %
          (dest, out["size"], len(out["added"]), len(out["removed"])), file=sys.stderr)
    print(json.dumps({k: out[k] for k in ("as_of", "size", "added", "removed")}, indent=1))
