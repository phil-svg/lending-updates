// window_oracle.h (v5, 22 Oct 2026, LLAMMA-Design/v2/verify/engine2): the concept oracle
// v2/concept/curve_stablecoin/price_oracles/v2/WindowOracle.vy (Vyper 0.4.3) in doubles, for replay_sim_x (--wo 1) and
// the parity harness wo_parity.cpp (the same code in both).
//
// PoolEMA      the oracle pool's price_oracle(): an EMA of block-end prices, folded in at the first trade of a block with the
//              PREVIOUS block-end price (tricrypto-ng tweak_price, MockCryptoPool.trade), viewed lazily:
//              view(t) = lp_k (1 - a) + e_k a, a = exp(-(t - t_k) / ma_k), k = the last trade block with t_k <= t.
//              Built from the same block series as the cb2 model and results/series/today_pool (oracles.ema_at_blocks / today).
// WindowOracle the contract's storage and its three entry points:
//              deploy(t, p_pool)      __init__: band = (p_pool, p_pool, p_pool), observation 0 = (t, p_pool), next slot 1
//              price(t, ...)  (view)  _band(packed_last)[2] * _factor_ro
//              price_w(t, ...)        _price_w: the same band, written with this block's observation (once per timestamp)
//              Every call takes the pool's price_oracle() and its ma_time AT THAT TIME (what the contract reads).
// Arithmetic: the contract's integer formulas in doubles. p_block is computed as p_last + (p_pool - p_last) / (1 - alpha)
// (= (p_pool - alpha p_last) / (1 - alpha), the contract's form; the rearranged form keeps the double rounding at ~1e-16
// instead of ~1e-16 / (1 - alpha)); 1 - exp(-x) via expm1. The USD factor (crvUSD aggregator x stableswap) is a constant
// AGG (1.0 in the engine, as the cb2 model and the today_pool series), re-read under the same STALE rule.
// unlock (engine --wo 2): the owner's fix of 22 Oct 2026 (v2/concept WindowOracle.vy _band: `if t_prev > 0 and dt < WINDOW:`, the
// file before it = v2/fork/oracle/WindowOracle.before_unlock_fix.vy = unlock false = --wo 1): the newest observation before the
// window counts only while the last write is less than WINDOW old; after that p_block alone covers the window.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct PoolEMA {
    std::vector<long> ts; std::vector<double> lp, raw, e;
    long init_ts = 0; double init_lp = 0, init_po = 0, init_raw = 865;
    long cur = -1;                                   // last block with ts <= the last query (queries may go back: binary search then)
    void build() {                                   // e[k] = the EMA as stored at block k (oracles.ema_at_blocks with the init row)
        e.resize(ts.size());
        double pe = init_po, pl = init_lp; long pt = init_ts;
        for (size_t k = 0; k < ts.size(); k++) {
            const double a = std::exp(-double(ts[k] - pt) / raw[k]);
            pe = pl * (1.0 - a) + pe * a; e[k] = pe; pt = ts[k]; pl = lp[k];
        }
    }
    long idx(long t) {
        const long n = long(ts.size());
        if (cur >= 0 && cur < n && ts[cur] <= t) { while (cur + 1 < n && ts[cur + 1] <= t) cur++; return cur; }
        cur = long(std::upper_bound(ts.begin(), ts.end(), t) - ts.begin()) - 1;
        return cur;
    }
    double view(long t) {                            // price_oracle() at time t
        const long k = idx(t);
        if (k < 0) { const double a = std::exp(-double(t - init_ts) / init_raw); return init_lp * (1.0 - a) + init_po * a; }
        const double a = std::exp(-double(t - ts[k]) / raw[k]);
        return lp[k] * (1.0 - a) + e[k] * a;
    }
    double ma(long t) { const long k = idx(t); return k < 0 ? init_raw : raw[k]; }   // the pool's ma_time now
    // file: 4 doubles [init_ts, init_lp, init_po, init_raw], then rows of 3 doubles [ts, lp, raw] (engine2/export_pool.py)
    bool load(const char *path) {
        FILE *f = std::fopen(path, "rb"); if (!f) return false;
        double h[4]; if (std::fread(h, 8, 4, f) != 4) { std::fclose(f); return false; }
        init_ts = long(h[0]); init_lp = h[1]; init_po = h[2]; init_raw = h[3];
        double r[3];
        while (std::fread(r, 8, 3, f) == 3) { ts.push_back(long(r[0])); lp.push_back(r[1]); raw.push_back(r[2]); }
        std::fclose(f); build(); return !ts.empty();
    }
};

struct WindowOracle {
    // immutables (the concept's deployment: WINDOW 36 s, TAU 300 s, CAP 0.3 % per 12 s, STALE 300 s; N_OBS = 4 is a constant)
    long WINDOW = 36; double TAU = 300.0, CAP = 0.003; int N_OBS = 4; long STALE = 300; double AGG = 1.0;
    // storage: packed_band = (hi, lo, p); packed_last = (t_last, slot, p_pool_last, factor, t_factor); packed_obs = N_OBS x (t, p)
    double hi = 0, lo = 0, p = 0;
    long t_last = 0; int slot = 0; double p_pool_last = 0, factor = 1.0; long t_factor = 0;
    std::vector<long> ot; std::vector<double> op;   // observation time (0 with price 0 = an empty slot), average block-end price
    bool deployed = false; long n_writes = 0;
    bool unlock = false;                            // false = WindowOracle.before_unlock_fix.vy, true = the fixed WindowOracle.vy
    int ASYM = 0;
    double FALL_CLAMP = 0.5;                        // round_r3c/ceiling: the per-write ceiling of a FALL (contract WAD // 2 until 24 Oct 2026 22:30; removed then); 0.5 = bit-exact
    double RISE_CLAMP = 0.5;                        // round_r3c/ceiling: the per-write ceiling of a RISE (the contract: WAD // 2 = 0.5); 0.5 = bit-exact; falls keep 0.5                                   // 1 = redesign/oracle WindowOracle: CAP limits only rises (falls: TAU alone); round_r1/oracle: 2 = only falls, 3 = cap >= the pool EMA's move
    bool anchor = false;                            // true = oracle_study/state/WindowOracleAnchor.vy (--wo 3): the pool's EMA is an observation too

    struct Band { double hi, lo, p, p_block, p_pool; };

    void deploy(long t, double p_pool) {
        hi = lo = p = p_pool; t_last = t; slot = 1 % N_OBS; p_pool_last = p_pool; factor = AGG; t_factor = t;
        ot.assign(N_OBS, 0); op.assign(N_OBS, 0.0); ot[0] = t; op[0] = p_pool;
        deployed = true;
    }
    // _band(packed_last): the band brought to time t; p_pool / ma = the pool's price_oracle() and ma_time at t
    Band band(long t, double p_pool, double ma) const {
        const long dt = t - t_last;
        if (dt == 0) return Band{hi, lo, p, p_pool_last, 0.0};
        // the pool EMA's own increment, inverted: p_block = (p_pool - a p_pool_last) / (1 - a), a = exp(-dt / ma_time)
        const double x = double(dt) / ma, a_pool = std::exp(-x);
        double p_block = p_pool;
        if (a_pool < 1.0 - 1e-12) p_block = std::max(p_pool_last + (p_pool - p_pool_last) / (-std::expm1(-x)), 0.0);
        // highest and lowest observation of the last WINDOW seconds, the newest one before them, and p_block
        double p_max = p_block, p_min = p_block, p_prev = 0; long t_prev = 0;
        const long t_start = t - WINDOW;
        for (int i = 0; i < N_OBS; i++) {
            if (ot[i] == 0 && op[i] == 0.0) continue;
            const long to = ot[i]; const double po = op[i];
            if (to > t_start) { p_max = std::max(p_max, po); p_min = std::min(p_min, po); }
            else if (to >= t_prev) { t_prev = to; p_prev = po; }
        }
        if (t_prev > 0 && (!unlock || dt < WINDOW)) { p_max = std::max(p_max, p_prev); p_min = std::min(p_min, p_prev); }
        if (anchor) { p_max = std::max(p_max, p_pool); p_min = std::min(p_min, p_pool); }
        // hi and lo follow p_max and p_min with time constant TAU, at most CAP per 12 s (never past them)
        const double alpha = -std::expm1(-double(dt) / TAU);
        double max_move = std::min(CAP * double(dt) / 12.0, 0.5);
        if (ASYM == 3 && p_pool_last > 0) max_move = std::min(std::max(max_move, std::fabs(p_pool / p_pool_last - 1.0)), 0.5);   // round_r1/oracle: cap >= the pool EMA's own move
        double h = hi, l = lo;
        const double max_fall = ASYM == 1 ? 1e300 : max_move, max_rise = ASYM == 2 ? 1e300 : max_move;
        const double max_rise_c = (RISE_CLAMP != 0.5 && ASYM == 0) ? std::min(CAP * double(dt) / 12.0, RISE_CLAMP) : max_rise;   // round_r3c/ceiling
        const double max_fall_c = (FALL_CLAMP != 0.5 && ASYM == 0) ? std::min(CAP * double(dt) / 12.0, FALL_CLAMP) : max_fall;   // round_r3c/ceiling
        if (p_max > h) h = h + std::min((p_max - h) * alpha, h * max_rise_c); else h = h - std::min((h - p_max) * alpha, h * max_fall_c);
        if (p_min > l) l = l + std::min((p_min - l) * alpha, l * max_rise_c); else l = l - std::min((l - p_min) * alpha, l * max_fall_c);
        return Band{h, l, std::min(std::max(p, l), h), p_block, p_pool};
    }
    double factor_ro(long t) const { return (t - t_factor > STALE) ? AGG : factor; }
    double price(long t, double p_pool, double ma) const { return band(t, p_pool, ma).p * factor_ro(t); }   // view
    double price_w(long t, double p_pool, double ma) {                                                     // _price_w
        const Band b = band(t, p_pool, ma);
        double fac = factor; long tf = t_factor;
        if (t - t_factor > STALE) { fac = AGG; tf = t; }        // STABLESWAP_AGGREGATOR.price_w() (x the stableswap's price_oracle)
        if (t > t_last) {
            ot[slot] = t; op[slot] = b.p_block;
            hi = b.hi; lo = b.lo; p = b.p;
            t_last = t; slot = (slot + 1) % N_OBS; p_pool_last = b.p_pool; factor = fac; t_factor = tf;
            n_writes++;
        }
        return b.p * fac;
    }
};
