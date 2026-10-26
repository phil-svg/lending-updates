"""The oracle candidates of the study (BRIEF.md), as pure functions on block-level data. Every candidate reads PAST data only.

Inputs (one pool, one coin):
  t_blk  int64 [n]   time of each block in which the pool traded (the block's timestamp)
  p_blk  float64 [n] the pool's spot (last_prices) AFTER that block's trades
  cl_t, cl_p         Chainlink: updated_at and answer of each round, sorted (optional)
  at     float64 [m] the times the oracle is asked for (the replay's minute ends)

The pool EMA as CurveTricryptoOptimizedWETH computes it (half-life 600 s): at a trade block k, before last_prices is set
to p_k, price_oracle <- p_{k-1} (1 - a) + price_oracle a with a = exp(-(t_k - t_{k-1}) / MA_RAW); the view price_oracle()
at a later time tau applies the same step from t_k to tau with p_k. So the EMA chases the spot of the PREVIOUS trade.
"""
import math

import numpy as np

MA_RAW = 866.0                       # ma_time raw of the three source pools (600 s half-life = 866 ln 2)
LN2 = math.log(2.0)


def _alpha(dt, raw=MA_RAW):
    return np.exp(-np.asarray(dt, float) / raw)


def _raw_blk(raw, n):
    """ma_time raw as a per-trade-block array (B1, 14 Oct 2026: TricryptoUSDT ran 601 s raw Jun-Aug 2026, USDC runs 865):
    a scalar is broadcast; an array [n] gives the raw in force at each trade block (used for the step INTO that block
    and for the view from it, as data/fetch_pools.py ema_replay / ema_view do)"""
    r = np.asarray(raw, float)
    return np.broadcast_to(r, (n,)) if r.ndim == 0 else r


def ema_at_blocks(t_blk, p_blk, raw=MA_RAW, p0=None):
    """price_oracle as stored at each trade block (after that block's update, before its own spot enters): float64 [n]"""
    t, p = np.asarray(t_blk, float), np.asarray(p_blk, float)
    rw = _raw_blk(raw, len(p))
    e = np.empty_like(p)
    e[0] = p[0] if p0 is None else p0
    for k in range(1, len(p)):
        a = math.exp(-(t[k] - t[k - 1]) / rw[k])
        e[k] = p[k - 1] * (1.0 - a) + e[k - 1] * a
    return e


def today(at, t_blk, p_blk, raw=MA_RAW, e_blk=None):
    """the live oracle: the pool EMA's view at the times `at` (float64 [m])"""
    t, p = np.asarray(t_blk, float), np.asarray(p_blk, float)
    e = ema_at_blocks(t, p, raw) if e_blk is None else e_blk
    k = np.searchsorted(t, np.asarray(at, float), side="right") - 1
    k = np.maximum(k, 0)
    a = _alpha(np.asarray(at, float) - t[k], _raw_blk(raw, len(p))[k])
    return p[k] * (1.0 - a) + e[k] * a


def cascade(at, t_blk, p_blk, raw=MA_RAW, h2=1200.0, vdown=None, vup=None, e_blk=None, p0=None):
    """the proposed oracle CONTRACT (round 4, 14 Oct 2026): the pool stays as it is; the contract reads the pool's
    price_oracle() (today's EMA, half-life raw*ln2, the pending step applied) at every call and keeps its own EMA of it
    with half-life h2 (lazily by timestamps, as the pool does), capped at vdown / vup per 12 s (log) on the way down / up.
    Every trade block and every sample time counts as a call (the contract is touched at every LLAMMA trade, liquidation,
    borrow and repay; here = the upper bound of its responsiveness). Returns the contract's value at the times `at`."""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    ev = np.concatenate([t, at]); kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable"); ev, kind = ev[o], kind[o]
    pool = today(ev, t, p, raw=raw, e_blk=e_blk)                 # the pool EMA's view at every event
    out = np.empty(len(at)); e2 = float(p[0] if p0 is None else p0); t_prev = ev[0]; ia = 0
    tau2 = h2 / LN2
    lo_v = -math.inf if vdown is None else -float(vdown); hi_v = math.inf if vup is None else float(vup)
    capped = vdown is not None or vup is not None
    for j in range(len(ev)):
        tau = ev[j]; dt = tau - t_prev
        if dt > 0:
            b = math.exp(-dt / tau2)
            e_new = pool[j] * (1.0 - b) + e2 * b
            if capped:
                r = math.log(e_new / e2); f = dt / 12.0
                e_new = e2 * math.exp(max(lo_v * f, min(hi_v * f, r)))
            e2 = e_new; t_prev = tau
        if kind[j] == 1:
            out[ia] = e2; ia += 1
    return out


def rate_limit(at, series, v, p0=None):
    """the oracle contract's output rate limiter (round 5): out_t = out_{t-1} * exp(clip(ln(series_t / out_{t-1}), -v dt/12, +v dt/12)),
    applied at the sample times `at` (dt = the time since the previous sample; v = log move per 12 s). Any inner oracle."""
    at, x = np.asarray(at, float), np.asarray(series, float)
    out = np.empty(len(x)); e = float(x[0] if p0 is None else p0); t_prev = at[0]
    for j in range(len(x)):
        if not np.isfinite(x[j]):
            out[j] = np.nan; continue
        if not np.isfinite(e): e = float(x[j])
        dt = at[j] - t_prev
        if dt > 0:
            lim = v * dt / 12.0; r = math.log(x[j] / e)
            e = e * math.exp(max(-lim, min(lim, r))); t_prev = at[j]
        out[j] = e
    return out


def robust_input(p_blk, K, how="max"):
    """the spike-proof input: max (or median) of the last K end-of-block prices (K = 1: the price itself)"""
    p = np.asarray(p_blk, float); K = int(K)
    if K <= 1: return p.copy()
    out = np.empty(len(p))
    for i in range(len(p)):
        w = p[max(0, i - K + 1):i + 1]
        out[i] = w.max() if how == "max" else float(np.median(w))
    return out


def radp(at, t_blk, p_blk, K=3, hl=318.3, hl_fast=36.90, g0=0.00606, g0_up=0.01630, g_exp=1.069, v=None, how="max"):
    """round 6 (14 Oct 2026): the adaptive oracle (fast mode when the slow EMA stands g0 above the fast 2-min EMA) fed with the
    spike-proof input max(last K block prices): a dump must persist K blocks to be seen at all; a real decline is seen K-1 blocks late"""
    return adaptive_r6(at, t_blk, robust_input(p_blk, K, how), hl=hl, hl_fast=hl_fast, g0=g0, g0_up=g0_up, g_exp=g_exp, v=v)


def cband(at, t_blk, p_blk, tau=120.0, ref="ema600", raw=MA_RAW, e_blk=None, p0=None, v=None, ref_hl=None):
    """round 8 (15 Oct 2026), the confidence-band oracle contract: state p (reported), [lo, hi] (the band), updated at every
    call (= every trade block and every sample time here). At a call: the band shrinks toward the reference r (the pool EMA's
    view, half-life raw*ln2; ref='ema400' = a 7-min EMA; ref='spot' = last_prices) by 1 - exp(-dt/tau); the trade price s
    (last_prices) widens the band to include it (lo = min(lo, s), hi = max(hi, s)); p does not move while it lies inside
    the band, and is clipped to the band's edge when the band has left it behind. A trade can only widen the band, never
    move p; p follows the reference with the band's shrink time."""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    ev = np.concatenate([t, at]); kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable"); ev, kind = ev[o], kind[o]
    if ref == "spot":
        r_ev = spot_at(ev, t, p)
    else:
        rw = (float(ref_hl) / LN2) if ref_hl is not None else (raw if ref == "ema600" else 432.8085)   # ref_hl: reference half-life (s)
        e = e_blk if (e_blk is not None and ref == "ema600" and ref_hl is None) else ema_at_blocks(t, p, raw=rw, p0=p0)
        r_ev = today(ev, t, p, raw=rw, e_blk=e)
    s_ev = spot_at(ev, t, p)
    out = np.empty(len(at)); x = float(p[0] if p0 is None else p0); lo = hi = x; t_prev = ev[0]; ia = 0
    for j in range(len(ev)):
        tau_j = ev[j]; dt = tau_j - t_prev
        if dt > 0:
            f = 1.0 - math.exp(-dt / tau); r = r_ev[j]
            lo += (r - lo) * f; hi += (r - hi) * f
            t_prev = tau_j
        sj = s_ev[j]
        if sj < lo: lo = sj
        if sj > hi: hi = sj
        xn = lo if x < lo else (hi if x > hi else x)
        if v is not None and xn != x:                              # optional cap on p's move per 12 s (log)
            lim = v * dt / 12.0; rr = math.log(xn / x)
            xn = x * math.exp(max(-lim, min(lim, rr)))
        x = xn
        if kind[j] == 1:
            out[ia] = x; ia += 1
    return out


def cband2(at, t_blk, p_blk, tau=300.0, W=60.0, v=None, p0=None, pmin=None, pmax=None, h=None, asym=0):
    """round 10 (15 Oct 2026), the window-extremes band oracle contract: it reads ONLY the pool's standing price (no pool EMA).
    Observations: the pool's standing price path; per trade block its lowest / highest observed value (pmin / pmax; with
    block-end coverage both = last_prices after the block); between trade blocks the standing price is the last block's
    last_prices. WMAX(t) / WMIN(t) = the highest / lowest standing price over the window [t - W, t] (the blocks inside it plus
    the standing price at t - W). State: hi follows WMAX and lo follows WMIN, each by 1 - exp(-dt/tau) per call (an EMA of
    the window extremes); p (reported) does not move while it lies inside [lo, hi] and is clipped to the edge that left it
    behind; hi and lo (and so p) move at most v per 12 s (log). A trade can only move p after the pool has stood at the new price for W
    seconds: a dump that is taken back inside W seconds never reaches p; a sustained move reaches it W seconds later.
    h (optional, s): hi / lo do not follow WMAX / WMIN directly but their EMAs of half-life h (the round-8 band's shape: a
    fast reference, a slower band; the response to a sustained move starts slowly and then catches up)."""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    pmn = p if pmin is None else np.asarray(pmin, float); pmx = p if pmax is None else np.asarray(pmax, float)
    ev = np.concatenate([t, at]); kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable"); ev, kind = ev[o], kind[o]
    out = np.empty(len(at)); x = float(p[0] if p0 is None else p0); lo = hi = x; t_prev = ev[0]; ia = 0
    rmax = rmin = x; lam = (LN2 / float(h)) if h else None
    from collections import deque
    dmax, dmin = deque(), deque()                                    # monotonic deques of block indexes inside the window
    i_in = 0                                                         # next block to enter the window (t_i <= t)
    i_out = 0                                                        # first block with t_i > t - W (blocks before it have left)
    nb = len(t)
    for j in range(len(ev)):
        tj = ev[j]
        while i_in < nb and t[i_in] <= tj:
            while dmax and pmx[dmax[-1]] <= pmx[i_in]: dmax.pop()
            dmax.append(i_in)
            while dmin and pmn[dmin[-1]] >= pmn[i_in]: dmin.pop()
            dmin.append(i_in)
            i_in += 1
        while i_out < nb and t[i_out] <= tj - W:
            i_out += 1
        while dmax and dmax[0] < i_out: dmax.popleft()
        while dmin and dmin[0] < i_out: dmin.popleft()
        stand = p[i_out - 1] if i_out > 0 else (p[0] if p0 is None else p0)   # the standing price at t - W
        wmax = max(stand, pmx[dmax[0]]) if dmax else stand
        wmin = min(stand, pmn[dmin[0]]) if dmin else stand
        dt = tj - t_prev
        if dt > 0:
            if lam is not None:                                      # the references: EMAs (half-life h) of the window extremes
                a = 1.0 - math.exp(-dt * lam); rmax += (wmax - rmax) * a; rmin += (wmin - rmin) * a
            else:
                rmax, rmin = wmax, wmin
            f = 1.0 - math.exp(-dt / tau)
            hn = hi + (rmax - hi) * f; ln = lo + (rmin - lo) * f
            if v is not None:                                        # the cap: hi, lo (and so p) move at most v per 12 s (log)
                lim = v * dt / 12.0; up, dn = (math.inf if asym == 2 else lim), (math.inf if asym == 1 else lim)   # asym 1 / 2: the cap limits only rises / only falls
                hn = hi * math.exp(max(-dn, min(up, math.log(hn / hi)))); ln = lo * math.exp(max(-dn, min(up, math.log(ln / lo))))
            hi, lo = hn, ln
            t_prev = tj
        x = lo if x < lo else (hi if x > hi else x)
        if kind[j] == 1:
            out[ia] = x; ia += 1
    return out


def spot_at(at, t_blk, p_blk):
    """the pool's last_prices at the times `at` (the spot of the last trade before)"""
    k = np.maximum(np.searchsorted(np.asarray(t_blk, float), np.asarray(at, float), side="right") - 1, 0)
    return np.asarray(p_blk, float)[k]


def chainlink_at(at, cl_t, cl_p, stale=86400.0):
    """the latest Chainlink answer at the times `at` by its updated_at; NaN where none or stale"""
    at = np.asarray(at, float)
    k = np.searchsorted(np.asarray(cl_t, float), at, side="right") - 1
    out = np.where(k >= 0, np.asarray(cl_p, float)[np.maximum(k, 0)], np.nan)
    age = at - np.asarray(cl_t, float)[np.maximum(k, 0)]
    return np.where((k >= 0) & (age <= stale), out, np.nan)


def clamp(o, cl, bound):
    """the contract's Chainlink clamp: min(max(o, cl (1 - b)), cl (1 + b)); where Chainlink is missing, o"""
    o, cl = np.asarray(o, float), np.asarray(cl, float)
    lo, hi = cl * (1.0 - bound), cl * (1.0 + bound)
    r = np.minimum(np.maximum(o, lo), hi)
    return np.where(np.isfinite(cl), r, o)


def last_moment(at, t_blk, p_blk, G, T, hl_fast=60.0, raw=MA_RAW, gate=None, gate_t=None, gate_G=None, p0=None):
    """today's EMA with a catch-up (downwards only): once the EMA has stood more than G (log) above the pool's spot for
    longer than T seconds without a break, it moves toward the spot with half-life hl_fast until the gap is under G.
    gate: an independent price series (gate_t, gate values) that must stand below the oracle by more than gate_G (log) at
    the START of the catch-up and while it runs; None = ungated. Returns the oracle at the times `at`.
    The state machine runs over the trade blocks and the asked times merged (what a contract sees at each call)."""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    ev = np.concatenate([t, at])
    kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable")
    ev, kind = ev[o], kind[o]
    gi = None
    if gate is not None:
        gi = np.maximum(np.searchsorted(np.asarray(gate_t, float), ev, side="right") - 1, 0)
        gate = np.asarray(gate, float)
    out = np.empty(len(at))
    e = p[0] if p0 is None else p0                             # the oracle
    spot = p[0]                                                # last_prices as the contract reads it (the last trade's)
    t_prev = ev[0]
    gap_since = -1.0                                           # when the gap first exceeded G (-1: not now)
    fast = False
    ib = 0                                                     # next trade block index
    ia = 0
    lnG = G
    for j in range(len(ev)):
        tau = ev[j]
        dt = tau - t_prev
        if dt > 0:
            hl = hl_fast if fast else raw * LN2
            a = math.exp(-dt / (hl / LN2))
            e = spot * (1.0 - a) + e * a
            t_prev = tau
        if kind[j] == 0:                                       # a trade block: the spot after it becomes the new last_prices
            spot = p[ib]; ib += 1
        else:
            out[ia] = e; ia += 1
        # the state for the NEXT interval, from what is known now
        gap = math.log(e / spot) if spot > 0 else 0.0
        if gap > lnG:
            if gap_since < 0:
                gap_since = tau
            held = tau - gap_since
            ok = True
            if gate is not None:
                g = gate[gi[j]]
                ok = np.isfinite(g) and g > 0 and math.log(e / g) > gate_G
            fast = held >= T and ok
        else:
            gap_since = -1.0
            fast = False
    return out


def last_moment2(at, t_blk, p_blk, G, T, hl_fast=60.0, raw=MA_RAW, K=0, rebound=None, p0=None):
    """the single-pool catch-ups of the study (BRIEF, candidates 1-3) in one state machine, down only:
    - timer: the gap must have exceeded G for T seconds without a break (gap_since, one stored timestamp);
    - K > 0 (recent-max): the gap is measured against the HIGHEST spot of the last K trade blocks, so a one-block dump
      that is arbitraged back within K blocks never opens a gap;
    - rebound (flow-confirmed): the fast mode stops as soon as the spot stands more than `rebound` (log) above its lowest
      value since the gap opened - an arbitrage back ends the catch-up at once; selling that goes on keeps it.
    Returns the oracle at the times `at`."""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    ev = np.concatenate([t, at]); kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable"); ev, kind = ev[o], kind[o]
    out = np.empty(len(at)); e = p[0] if p0 is None else p0; spot = p[0]; t_prev = ev[0]
    gap_since = -1.0; fast = False; ib = ia = 0; low = spot
    ring = [spot] * max(K, 1); ri = 0
    rw = _raw_blk(raw, len(p)).tolist(); raw_cur = rw[0]      # the pool's ma_time raw in force (per block)
    tf = hl_fast / LN2
    for j in range(len(ev)):
        tau = ev[j]; dt = tau - t_prev
        if dt > 0:
            a = math.exp(-dt / (tf if fast else raw_cur)); e = spot * (1.0 - a) + e * a; t_prev = tau
        if kind[j] == 0:
            raw_cur = rw[ib]; spot = p[ib]; ib += 1
            if K > 0:
                ring[ri] = spot; ri = (ri + 1) % K
        else:
            out[ia] = e; ia += 1
        ref = max(ring) if K > 0 else spot
        gap = math.log(e / ref) if ref > 0 else 0.0
        if gap > G:
            if gap_since < 0:
                gap_since = tau; low = spot
            low = min(low, spot)
            ok = True
            if rebound is not None and spot > low * math.exp(rebound):
                ok = False
            fast = (tau - gap_since >= T) and ok
        else:
            gap_since = -1.0; fast = False
    return out


def robust(at, t_blk, p_blk, K=10, how="median", G=None, T=None, hl_fast=60.0, raw=MA_RAW, p0=None):
    """a single-pool EMA whose INPUT is robust to one block: the EMA chases the median (how="median") or the highest
    (how="max": down moves delayed, up moves at once) of the last K trade blocks' spots instead of the last spot alone - a
    jump that is arbitraged back within K blocks never enters the oracle at all. With G and T set, the timer catch-up of
    last_moment2 runs on top (gap measured against the robust input). On chain: a ring of K spots (K x 32 bytes).
    Returns the oracle at the times `at`."""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    ev = np.concatenate([t, at]); kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable"); ev, kind = ev[o], kind[o]
    out = np.empty(len(at)); e = p[0] if p0 is None else p0; t_prev = ev[0]; ib = ia = 0
    ring = [p[0]] * K; ri = 0
    agg = (lambda r: float(np.median(r))) if how == "median" else (lambda r: max(r))
    inp = p[0]
    gap_since = -1.0; fast = False
    for j in range(len(ev)):
        tau = ev[j]; dt = tau - t_prev
        if dt > 0:
            hl = hl_fast if fast else raw * LN2
            a = math.exp(-dt / (hl / LN2)); e = inp * (1.0 - a) + e * a; t_prev = tau
        if kind[j] == 0:
            ring[ri] = p[ib]; ri = (ri + 1) % K; ib += 1
            inp = agg(ring)
        else:
            out[ia] = e; ia += 1
        if G is not None:
            gap = math.log(e / inp) if inp > 0 else 0.0
            if gap > G:
                if gap_since < 0:
                    gap_since = tau
                fast = tau - gap_since >= T
            else:
                gap_since = -1.0; fast = False
    return out


def regime(at, t_blk, p_blk, N=10, M=7, D=0.02, hf=60.0, calmK=0, vfast=None, vslow=None, raw=MA_RAW, eps=2e-4, p0=None):
    """the 'extended crash' designs of the 14 Oct afternoon study (single pool, past data only):
    the pool is in a CONFIRMED DECLINE while, of its last N trade blocks, at least M moved the spot down (by more than eps) and
    the spot stands at least D (log) under where it stood N blocks ago; any arbitrage back lowers the count and ends it.
      - calm regime: the EMA (half-life 600 s) chases the last spot (calmK = 0) or the MEDIAN of the last calmK spots (jump-proof);
      - crash regime: the oracle chases the LAST spot with half-life hf.
    vslow / vfast (log per 12-s block, optional): a rate limit on the oracle's move per block in the calm / crash regime
    (vslow alone, with M > N so the crash regime never comes: the plain rate-limited oracle).
    Returns the oracle at the times `at`."""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    ev = np.concatenate([t, at]); kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable"); ev, kind = ev[o], kind[o]
    out = np.empty(len(at)); e = p[0] if p0 is None else p0; t_prev = ev[0]; ib = ia = 0
    steps = [0] * N; si = 0; hist = [p[0]] * (N + 1); hi = 0           # the last N steps (-1 down, 0 flat, +1 up) and N+1 spots
    ring = [p[0]] * max(calmK, 1); ri = 0
    spot = p[0]; inp = p[0]; crash = False
    for j in range(len(ev)):
        tau = ev[j]; dt = tau - t_prev
        if dt > 0:
            hl = hf if crash else raw * LN2
            a = math.exp(-dt / (hl / LN2))
            target = spot if crash else inp
            e_new = target * (1.0 - a) + e * a
            v = vfast if crash else vslow
            if v is not None:                                      # the rate limit: at most v per 12 s
                lim = v * dt / 12.0
                r = math.log(e_new / e)
                e_new = e * math.exp(max(-lim, min(lim, r)))
            e = e_new; t_prev = tau
        if kind[j] == 0:
            pn = p[ib]; ib += 1
            st = -1 if pn < spot * (1.0 - eps) else (1 if pn > spot * (1.0 + eps) else 0)
            steps[si] = st; si = (si + 1) % N
            hist[hi] = pn; hi = (hi + 1) % (N + 1)
            spot = pn
            if calmK > 0:
                ring[ri] = pn; ri = (ri + 1) % calmK
                inp = float(np.median(ring))
            else:
                inp = pn
            oldest = hist[hi]                                      # the spot N blocks ago
            crash = (sum(1 for x in steps if x < 0) >= M) and (oldest > 0 and math.log(oldest / pn) >= D)
        else:
            out[ia] = e; ia += 1
    return out


def adaptive_r6(at, t_blk, p_blk, hl=318.3, hl_fast=36.90, g0=0.00606, g0_up=0.01630, g_exp=1.069, v=None):
    """the 1 Oct tuned adaptive oracle (round 6, oracle+HL fix+fee design) at block level: a slow EMA whose half-life
    switches toward hl_fast with weight 1 - exp(-(gap/g0)^g_exp), gap = ln(slow / fast 2-min EMA)"""
    t, p, at = np.asarray(t_blk, float), np.asarray(p_blk, float), np.asarray(at, float)
    ev = np.concatenate([t, at]); kind = np.concatenate([np.zeros(len(t), np.int8), np.ones(len(at), np.int8)])
    o = np.argsort(ev, kind="stable"); ev, kind = ev[o], kind[o]
    out = np.empty(len(at)); e = fast = spot = p[0]; t_prev = ev[0]; ib = ia = 0
    for j in range(len(ev)):
        tau = ev[j]; dt = tau - t_prev
        if dt > 0:
            fast += (spot - fast) * (1.0 - 2.0 ** (-dt / hl_fast))
            g = math.log(e / fast); gz = g / g0 if g >= 0 else -g / g0_up
            w = 1.0 - math.exp(-gz ** g_exp); he = hl * (1.0 - w) + hl_fast * w
            e_new = e + (spot - e) * (1.0 - 2.0 ** (-dt / he))
            if v is not None:                                      # round 5: the contract's cap per 12 s, inside the state
                lim = v * dt / 12.0; r = math.log(e_new / e)
                e_new = e * math.exp(max(-lim, min(lim, r)))
            e = e_new
            t_prev = tau
        if kind[j] == 0:
            spot = p[ib]; ib += 1
        else:
            out[ia] = e; ia += 1
    return out


# ----------------------------------------------------------------------------------------------- the synthetic map
def swap_response(fn, d, K, blocks=300, dt=12.0, **kw):
    """the deepest drop of an oracle candidate when the source pool is held d (fraction) low for K blocks of dt seconds
    and then restored; fn(at, t_blk, p_blk, **kw) -> oracle. Returns 1 - min(oracle) / 1."""
    t = np.arange(blocks + 1, dtype=float) * dt
    p = np.ones(blocks + 1); p[1:K + 1] = 1.0 - d
    o = fn(t[1:], t, p, **kw)
    return float(1.0 - np.min(o))


def swap_map(fn, ds=(0.01, 0.02, 0.05, 0.10), Ks=(1, 2, 3, 5, 10, 25, 50), **kw):
    return {d: {K: swap_response(fn, d, K, **kw) for K in Ks} for d in ds}


if __name__ == "__main__":
    print("source pool held d low for K blocks: the deepest drop, % (today | last-moment G 3 %, T 5 min | adaptive r6)")
    for d in (0.01, 0.02, 0.05, 0.10):
        row = []
        for K in (1, 3, 10, 25, 50):
            a = swap_response(today, d, K); b = swap_response(last_moment, d, K, G=0.03, T=300.0); c = swap_response(adaptive_r6, d, K)
            row.append(f"K {K}: {100 * a:.2f} | {100 * b:.2f} | {100 * c:.2f}")
        print(f"  d {100 * d:.0f} %:  " + "   ".join(row))
