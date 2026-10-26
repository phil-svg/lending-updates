#!/usr/bin/env python3
"""poolmath.py — round 3 (LLAMMA-Design/r3, 14 Oct 2026): exact tricrypto-ng float64 math for the two source pools,
TricryptoUSDC (WETH -> USDC) and TriCRV (CRV -> crvUSD), on the state of bdsim/state_latest.json (block 26189776,
14 Oct 2026 09:14 UTC). The formulas are r2a/false_swap.py exact_depth's (= research/speed_llamma/playground/tri_math.py,
validated against on-chain get_dy): invariant F with A = ANN / 1e4 / 27, y by bisection, the dynamic fee
mid f + out (1 - f), f = fee_gamma / (fee_gamma + 1 - K), K = N^N prod(xp) / sum(xp)^N on the post-trade xp.
D of a post-trade state (solve_D) is bracketed from the invariant's pole upward (round-3 fix after the T2 review): F(D) has
a pole at K0 = 1 + gamma, D_pole = N prod(xp)^(1/N) / (1 + gamma)^(1/N), is > 0 everywhere below it and falls from +inf to
-inf above it, so the unique root sits in [D_pole, hi] with hi = 1.5 sum(xp) doubled until F(hi) < 0. The former bracket
[0.5 sum(xp), 1.5 sum(xp)] returned its lower edge silently for sells deeper than d_eff ~79 % (X > $7.4M TricryptoUSDC /
$3.5M TriCRV), which priced the buy-back on a wrong invariant (roundtrip x3.8 at 80 %, x290 at 95 % too high).

THE POOL STATE IS FROZEN DURING THE ATTACK: every block boundary starts from the same balances, price_scale and D (the
attacker's own trades are undone by the buy-back; nobody else trades; price_scale is never re-pegged). The one state that
moves within a round trip is the post-sell state (balances after the sell with the fee kept in the pool, D re-solved as
tweak_price's newton_D does), from which the buy-back is priced.

Pool(name).sell(X_usd) -> X_tokens (sold, at the fee-inclusive marginal price spot0), got_usd (received after the fee),
  marginal_after (fee-inclusive marginal price after the sell, finite difference as manip_cost.py), effective_after =
  got_usd / X_tokens (= what the contract stores as last_prices after the trade: dy/dx after the fee, v1.0.0 _exchange),
  depth_marg = 1 - marginal_after / spot0, depth_eff = 1 - effective_after / spot0, fee_rate_in; the buy-back of the SAME
  token amount from the post-sell state: paid_back_usd, fee_rate_back, roundtrip_loss_usd = paid_back - got.
cost_per_boundary(d_eff): the attacker's cost of one block boundary (dump as the last trade of block n, buy back as the
  first trade of block n+1) at the effective depth d_eff. The sell size is the one whose depth_eff = d_eff (bisection on X).
  Rule of round 3: inside the manip_cost.json range of MARGINAL depth (d_marg <= 5 %, the file's own d) the cost is
  max(roundtrip_loss, open_cost_per_block + builder_cost_per_block interpolated log-log in d_marg), because the file is the
  measured per-block cost of holding such a position open (arbitrage against it + the builder's bribe) and a round trip
  cannot cost less than the exact fee + slippage; beyond the file's range (d_marg > 5 %) the exact roundtrip_loss alone
  (round 2 EXTRAPOLATED the file there - that is what this redo replaces). Below the file's first row (0.5 %) the file
  values are extrapolated log-log along its first segment.
  The fast path (cost_per_boundary without exact=True) interpolates log-log on a dense X grid; the grid carries two extra
  points at X_cliff (1 -+ 1e-7), X_cliff = the sell whose d_marg is exactly 5 %, so the 'rule' cliff is a step in the
  interpolation too (round-3 fix: before, the cell straddling the cliff was interpolated across it, -84 % / +31 % at
  0.5 % below / above the cliff depth).
Tables: X_usd log-spaced $10k..$20M (TricryptoUSDC) and $5k..$10M (TriCRV), 60 points, plus a dense cost grid for
  r3/attack.py -> r3/poolmath_tables.json. Validation (a)-(f) printed and stored under "validation": (e) solve_D against an
  independent Illinois root above the pole on 40 sizes up to the grid end + the deep round trips (75-95 %); (f) the fast
  cost against the exact one at 0.995 / 1.005 x the cliff depth and inside the former straddling cell.
Usage: cd r3 && nice -n 19 python3 poolmath.py"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
STATE = os.path.join(ROOT, "bdsim", "state_latest.json")
MANIP = os.environ.get("CURVE_SIM_UI", "curve-sim-ui") + "/research/speed_llamma/replay/liqtest/manip_cost.json"
# pool -> (manip_cost pool address, manipulated asset address, the sold coin index, the quote coin index)
POOLS = {
    "TricryptoUSDC": ("0x7f86bf177dd4f3494b841a37e810a34dd56c829b", "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 2, 0),
    "TriCRV": ("0x4ebdf703948ddcea3b11f675b4d1fba9d2414a14", "0xd533a949740bb3306d119cc777fa900ba034cd52", 2, 0),
}
TABLE_RANGE = {"TricryptoUSDC": (10e3, 20e6), "TriCRV": (5e3, 10e6)}
FILE_DMAX = 0.05                      # the manip_cost.json rows end at d_marg = 5 %
N = 3
ITERS = 100                           # bisection steps (float64 converges after ~55; exact_depth used 200)


class Manip:
    """manip_cost.json rows of one pool/asset: open + builder cost per block as a function of the MARGINAL depth d
    (log-log interpolation, extrapolated along the end segments as r2a/false_swap.Manip does)"""
    def __init__(self, pool):
        j = json.load(open(MANIP)); addr, asset, _, _ = POOLS[pool]
        rows = j["pools"][addr]["assets"][asset]["rows"]
        self.block = j["block"]; self.tvl_usd = j["pools"][addr]["tvl_usd"]; self.spot = j["pools"][addr]["assets"][asset]["spot"]
        self.rows = rows
        self.ld = np.log([r["d"] for r in rows]); self.lsell = np.log([r["sell_usd"] for r in rows])
        self.lopen = np.log([r["open_cost_per_block"] for r in rows]); self.lbuild = np.log([r["builder_cost_per_block"] for r in rows])
        self.dmax = max(r["d"] for r in rows)

    @staticmethod
    def _interp(x, xs, ys):
        if x <= xs[0]:
            s = (ys[1] - ys[0]) / (xs[1] - xs[0]); return ys[0] + s * (x - xs[0])
        if x >= xs[-1]:
            s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]); return ys[-1] + s * (x - xs[-1])
        return float(np.interp(x, xs, ys))

    def open_cost(self, d): return math.exp(self._interp(math.log(d), self.ld, self.lopen))
    def builder_cost(self, d): return math.exp(self._interp(math.log(d), self.ld, self.lbuild))
    def per_block(self, d): return self.open_cost(d) + self.builder_cost(d)
    def sell_usd(self, d): return math.exp(self._interp(math.log(d), self.ld, self.lsell))


class Pool:
    def __init__(self, name, state=STATE):
        SJ = json.load(open(state)); S = SJ["pools"][name]
        self.name = name; self.block = SJ["block"]; self.ts_utc = SJ["timestamp_utc"]
        self.coins = S["coins"]; self.dec = S["decimals"]
        _, _, self.i, self.j = POOLS[name]
        assert self.coins[self.i] in ("WETH", "CRV") and self.coins[self.j] in ("USDC", "crvUSD"), self.coins
        self.A = int(S["A"]) / 1e4 / 27; self.gamma = int(S["gamma"]) / 1e18
        self.mid, self.out, self.fg = int(S["mid_fee"]) / 1e10, int(S["out_fee"]) / 1e10, int(S["fee_gamma"]) / 1e18
        self.PS = [1.0] + [int(x) / 1e18 for x in S["price_scale"]]
        self.bal0 = [int(b) / 10 ** d for b, d in zip(S["balances"], self.dec)]
        self.D0 = int(S["D"]) / 1e18
        self.last_prices = S["last_prices"][self.i - 1]
        self.eps_tok = 1e-7 * self.bal0[self.i]                 # finite-difference step for the marginal price
        self.spot0 = self._dy(self.bal0, self.D0, self.i, self.j, self.eps_tok)[0] / self.eps_tok   # fee-inclusive marginal
        self.manip = Manip(name)
        self._grid = None

    # ---- invariant
    def F(self, xp, Dv):
        K0 = N ** N * math.prod(xp) / Dv ** N
        K = self.A * K0 * self.gamma ** 2 / (self.gamma + 1 - K0) ** 2
        return K * Dv ** (N - 1) * sum(xp) + math.prod(xp) - K * Dv ** N - (Dv / N) ** N

    def solve_y(self, xp, Dv, j):
        lo, hi = 1e-12 * Dv, Dv * 1.5
        for _ in range(ITERS):
            m = (lo + hi) / 2; x = list(xp); x[j] = m
            if self.F(x, Dv) > 0: hi = m
            else: lo = m
        return (lo + hi) / 2

    def D_pole(self, xp):
        """the pole of F in D: K0 = 1 + gamma  <=>  D = N prod(xp)^(1/N) / (1 + gamma)^(1/N); the root lies above it"""
        return N * math.prod(xp) ** (1.0 / N) / (1.0 + self.gamma) ** (1.0 / N)

    def solve_D(self, xp):
        """D of the state xp: bisection on [D_pole (1 + 1e-9), hi], F(lo) > 0 and F(hi) < 0 asserted (hi = 1.5 sum(xp),
        doubled while F(hi) > 0). Round-3 fix: the old bracket [0.5 sum, 1.5 sum] missed roots below 0.5 sum(xp)."""
        S = sum(xp)
        lo = self.D_pole(xp) * (1.0 + 1e-9); hi = 1.5 * S
        if not self.F(xp, lo) > 0:
            raise ArithmeticError(f"solve_D: F at the pole bracket is not positive ({self.F(xp, lo):.3e})")
        n_dbl = 0
        while self.F(xp, hi) > 0:
            hi *= 2.0; n_dbl += 1
            if n_dbl > 60: raise ArithmeticError("solve_D: no sign change above the pole")
        for _ in range(ITERS):
            m = (lo + hi) / 2
            if self.F(xp, m) > 0: lo = m
            else: hi = m
        return (lo + hi) / 2

    def fee(self, xp):
        K = N ** N * math.prod(xp) / sum(xp) ** N
        f = self.fg / (self.fg + 1 - K)
        return self.mid * f + self.out * (1 - f)

    def _dy(self, bal, Dv, i, j, dx):
        """sell dx of coin i for coin j on (bal, D): (dy after the fee, fee rate, balances after with the fee kept in the pool)"""
        xp = [b * p for b, p in zip(bal, self.PS)]; xp[i] += dx * self.PS[i]
        y = self.solve_y(xp, Dv, j); dy_xp = xp[j] - y; xp[j] -= dy_xp
        f = self.fee(xp); dy = dy_xp / self.PS[j] * (1 - f)
        post = list(bal); post[i] += dx; post[j] -= dy
        return dy, f, post

    def _dx_for(self, bal, Dv, i, j, dy_want, hint):
        """how much of coin i buys dy_want of coin j on (bal, D) (bisection on dx; inf when the pool cannot deliver)"""
        hi = max(hint, 1e-9)
        for _ in range(80):
            if self._dy(bal, Dv, i, j, hi)[0] >= dy_want: break
            hi *= 2.0
            if hi > 1e6 * self.D0: return float("inf"), float("nan")
        lo = 0.0
        for _ in range(ITERS):
            m = (lo + hi) / 2
            if self._dy(bal, Dv, i, j, m)[0] >= dy_want: hi = m
            else: lo = m
        dx = (lo + hi) / 2
        return dx, self._dy(bal, Dv, i, j, dx)[1]

    # ---- one round trip
    def sell(self, X_usd):
        X = X_usd / self.spot0
        got, f_in, post = self._dy(self.bal0, self.D0, self.i, self.j, X)
        got_eps = self._dy(self.bal0, self.D0, self.i, self.j, X + self.eps_tok)[0]
        marg = (got_eps - got) / self.eps_tok
        eff = got / X
        xp_post = [b * p for b, p in zip(post, self.PS)]
        D_post = self.solve_D(xp_post)
        paid, f_back = self._dx_for(post, D_post, self.j, self.i, X, hint=X_usd)
        return dict(X_usd=X_usd, X_tokens=X, got_usd=got, marginal_after=marg, effective_after=eff,
                    depth_marg=1.0 - marg / self.spot0, depth_eff=1.0 - eff / self.spot0, fee_rate_in=f_in,
                    paid_back_usd=paid, fee_rate_back=f_back, roundtrip_loss_usd=paid - got, D_post=D_post,
                    open_cost_usd=X_usd - got)        # the dump's slippage + fee = the loss when ARBITRAGEURS restore the price

    def X_for_depth(self, d, which="depth_eff", lo=1.0, hi=None):
        """the sell size (USD) whose depth (depth_eff | depth_marg) is d, by bisection on X (depth is monotone in X)"""
        if hi is None:
            hi = 1e4
            while self.sell(hi)[which] < d:
                hi *= 2.0
                if hi > 1e13: return float("inf")
        for _ in range(60):
            m = math.sqrt(lo * hi)
            if self.sell(m)[which] >= d: hi = m
            else: lo = m
        return math.sqrt(lo * hi)

    def cost_rule(self, r, mode="rule"):
        """the cost of one block boundary for a sell record r.
        mode 'rule' (the round-3 task's rule, see the module docstring): max(roundtrip_loss, file open + builder at d_marg) for
          d_marg <= 5 %, roundtrip_loss beyond. NOTE the cliff at d_marg = 5 %: the exact round trip is fees only (~1/15 of the
          file's open + builder), so the rule's cost DROPS there (not monotone in X) - a property of the rule, not of the pool.
        mode 'roundtrip': the exact round trip alone (the attacker buys back as the first trade of the next block: 2 fees, no
          slippage; a builder bribe is NOT included).
        mode 'open': the exact open-market cost X p0 - got (the dump's slippage + fee, lost when arbitrageurs restore the price);
          continuous, no file dependence; = the exact version of the file's open_cost_per_block on today's state.
        mode 'file_sum': the file's open + builder per block interpolated log-log in d_marg and EXTRAPOLATED beyond 5 % (round 2)."""
        if mode == "rule":
            if r["depth_marg"] <= FILE_DMAX:
                return max(r["roundtrip_loss_usd"], self.manip.per_block(max(r["depth_marg"], 1e-6)))
            return r["roundtrip_loss_usd"]
        if mode == "roundtrip": return r["roundtrip_loss_usd"]
        if mode == "open": return r["open_cost_usd"]
        if mode == "file_sum": return self.manip.per_block(max(r["depth_marg"], 1e-6))
        raise ValueError(mode)

    MODES = ("rule", "roundtrip", "open", "file_sum")

    def cost_per_boundary(self, d_eff, exact=False, mode="rule"):
        """attacker's cost of one block boundary at the EFFECTIVE depth d_eff (fraction), cost mode as cost_rule. exact=True:
        bisection on X and the exact round trip (~1 s); default: log-log interpolation on the dense grid (built once; the
        interpolation error is reported in the tables). Vectorised over an array d_eff in the default path."""
        if exact:
            X = self.X_for_depth(d_eff, "depth_eff")
            if not np.isfinite(X): return float("inf")
            return self.cost_rule(self.sell(X), mode)
        g = self.grid()
        d = np.asarray(d_eff, float)
        out = np.exp(np.interp(np.log(np.maximum(d, 1e-300)), g["ln_d_eff"], g["ln_cost"][mode]))
        out = np.where(d <= 0, 0.0, np.where(d > g["depth_eff"][-1], np.inf, out))
        return float(out) if np.ndim(d_eff) == 0 else out

    def X_cliff(self):
        """the sell size whose MARGINAL depth is exactly FILE_DMAX (5 %) = where the 'rule' cost steps down"""
        return self.X_for_depth(FILE_DMAX, "depth_marg")

    def grid(self, n=320, d_eff_max=0.96, X_lo=100.0, split=1e-7):
        """dense (X -> depth_eff, depth_marg, roundtrip_loss, cost_per_boundary) grid, log-spaced in X from X_lo to the
        sell size whose effective depth is d_eff_max, plus the two points X_cliff (1 -+ split) that make the 'rule' cliff a
        step of the interpolation (the point below has d_marg <= 5 %, the point above d_marg > 5 %: asserted)"""
        if self._grid is None:
            X_hi = self.X_for_depth(d_eff_max, "depth_eff")
            Xc = self.X_cliff()
            Xs = np.sort(np.concatenate([np.exp(np.linspace(math.log(X_lo), math.log(X_hi), n)), [Xc * (1 - split), Xc * (1 + split)]]))
            rows = [self.sell(float(x)) for x in Xs]
            kc = int(np.searchsorted(Xs, Xc * (1 - split)))
            assert rows[kc]["depth_marg"] <= FILE_DMAX < rows[kc + 1]["depth_marg"], (rows[kc]["depth_marg"], rows[kc + 1]["depth_marg"])
            self._cliff = dict(X_usd=Xc, depth_eff_below=rows[kc]["depth_eff"], depth_eff_above=rows[kc + 1]["depth_eff"],
                               rule_below=self.cost_rule(rows[kc]), rule_above=self.cost_rule(rows[kc + 1]), k=kc)
            de = np.array([r["depth_eff"] for r in rows])
            cost = {m: np.array([self.cost_rule(r, m) for r in rows]) for m in self.MODES}
            self._grid = dict(X_usd=Xs.tolist(), depth_eff=de.tolist(), depth_marg=[r["depth_marg"] for r in rows], cliff=self._cliff,
                              roundtrip_loss_usd=[r["roundtrip_loss_usd"] for r in rows], open_cost_usd=[r["open_cost_usd"] for r in rows],
                              cost_per_boundary={m: cost[m].tolist() for m in self.MODES},
                              ln_d_eff=np.log(de), ln_cost={m: np.log(cost[m]) for m in self.MODES})
        return self._grid

    def balanced_copy(self):
        """the same pool parameters with the balances set to a balanced state AT its price_scale (xp equal thirds of D): shows how
        much the sell size for a given depth depends on the distance to the peg (the manip_cost file was read at another block)"""
        import copy
        Q = copy.copy(self); Q._grid = None
        Q.bal0 = [self.D0 / 3.0 / p for p in self.PS]
        Q.eps_tok = 1e-7 * Q.bal0[Q.i]
        Q.spot0 = Q._dy(Q.bal0, Q.D0, Q.i, Q.j, Q.eps_tok)[0] / Q.eps_tok
        return Q


def illinois(f, a, b, xtol=1e-13, maxiter=1000):
    """independent bracketed root finder (Illinois regula falsi; the module's solvers bisect)"""
    fa, fb = f(a), f(b)
    if fa * fb > 0: raise ValueError(f"no bracket: f({a:.6g}) = {fa:.4e}, f({b:.6g}) = {fb:.4e}")
    side = 0; c = a
    for _ in range(maxiter):
        c = (a * fb - b * fa) / (fb - fa); fc = f(c)
        if fc == 0 or abs(b - a) < xtol * abs(c): return c
        if fc * fb < 0:
            a, fa = b, fb; side = 0
        else:
            if side == -1: fa /= 2
            side = -1
        b, fb = c, fc
    return c


def indep_D(P, xp):
    """D by Illinois on [D_pole (1 + 1e-9), 4 sum(xp)] (the reviewer's construction)"""
    a, b = P.D_pole(xp) * (1 + 1e-9), 4.0 * sum(xp)
    return illinois(lambda Dv: P.F(xp, Dv), a, b)


def validate_D(P, n=40):
    """(e) solve_D vs the independent root on n log-spaced sizes from the table's low end to the grid's end; the deep round
    trips at d_eff 75..95 % (loss, open cost, D) now priced on the right invariant"""
    g = P.grid(); lo, _ = TABLE_RANGE[P.name]
    rows = []
    for Xu in np.exp(np.linspace(math.log(lo), math.log(g["X_usd"][-1]), n)):
        r = P.sell(float(Xu)); post = list(P.bal0); post[P.i] += r["X_tokens"]; post[P.j] -= r["got_usd"]
        xpp = [b * p for b, p in zip(post, P.PS)]; S = sum(xpp)
        Di = indep_D(P, xpp)
        rows.append(dict(X_usd=float(Xu), depth_eff=r["depth_eff"], depth_marg=r["depth_marg"], D_post=r["D_post"], D_indep=Di, rel=r["D_post"] / Di - 1,
                         D_post_over_halfS=r["D_post"] / (0.5 * S), below_old_bracket=r["D_post"] < 0.5 * S, F_resid_rel=P.F(xpp, r["D_post"]) / (r["D_post"] / N) ** N))
    deep = []
    for d in (0.75, 0.80, 0.85, 0.90, 0.95):
        if d <= g["depth_eff"][-1]:
            X = P.X_for_depth(d, "depth_eff"); r = P.sell(X)
            deep.append(dict(d_eff=d, X_usd=X, D_post=r["D_post"], roundtrip_loss_usd=r["roundtrip_loss_usd"], open_cost_usd=r["open_cost_usd"],
                             rule=P.cost_rule(r), fee_rate_in=r["fee_rate_in"], fee_rate_back=r["fee_rate_back"]))
    return dict(rows=rows, max_abs_rel=max(abs(x["rel"]) for x in rows), n_below_old_bracket=sum(x["below_old_bracket"] for x in rows),
                pass_1e10=all(abs(x["rel"]) < 1e-10 for x in rows), deep=deep)


def validate_cliff(P):
    """(f) the fast 'rule' cost vs the exact one around the cliff: at 0.995 / 0.98 / 1.005 / 1.02 x the cliff depth and at 6 points
    inside the former straddling cell"""
    g = P.grid(); c = g["cliff"]; dc = c["depth_eff_below"]
    rows = []
    for d in (dc * 0.98, dc * 0.995, dc * 0.999, dc * 1.001, dc * 1.005, dc * 1.02):
        ce, cf = P.cost_per_boundary(d, exact=True, mode="rule"), P.cost_per_boundary(d, mode="rule")
        rows.append(dict(d_eff=float(d), exact=ce, fast=cf, rel_err=cf / ce - 1))
    # the former cell: the two plain grid neighbours of the cliff
    de = np.array(g["depth_eff"]); k = c["k"]
    lo_cell, hi_cell = float(de[k - 1]), float(de[k + 2])
    for d in np.linspace(lo_cell, hi_cell, 8)[1:-1]:
        ce, cf = P.cost_per_boundary(float(d), exact=True, mode="rule"), P.cost_per_boundary(float(d), mode="rule")
        rows.append(dict(d_eff=float(d), exact=ce, fast=cf, rel_err=cf / ce - 1, in_former_cell=True))
    return dict(cliff=c, former_cell=[lo_cell, hi_cell], rows=rows, max_abs_rel_err=max(abs(x["rel_err"]) for x in rows),
                pass_1e3=all(abs(x["rel_err"]) < 1e-3 for x in rows))


def validate(P, mp):
    """(a) sell_usd for d_marg 0.5/1/2/5 % vs the file within 10 %; (b) roundtrip_loss vs the file's open + builder per block
    (ratio); (c) effective_after > marginal_after; (d) monotonicity in X. Returns the dict; prints the lines."""
    out = {"a_sell_usd_vs_file": [], "b_roundtrip_vs_file_per_block": [], "pass_a": True, "pass_c": True, "pass_d": True}
    Qb = P.balanced_copy()
    out["peg"] = dict(spot0=P.spot0, price_scale=P.PS[P.i], spot0_vs_price_scale_pct=100 * (P.spot0 / P.PS[P.i] - 1),
                      xp_shares=[x / sum(P.bal0[k] * P.PS[k] for k in range(3)) for x in [P.bal0[k] * P.PS[k] for k in range(3)]])
    print(f"  peg: spot0 {P.spot0:.6g} vs price_scale {P.PS[P.i]:.6g} ({out['peg']['spot0_vs_price_scale_pct']:+.2f} %), xp shares {[round(x, 4) for x in out['peg']['xp_shares']]}")
    print(f"\n{P.name}: {P.coins[P.i]} -> {P.coins[P.j]}, block {P.block} ({P.ts_utc}), D ${P.D0:,.0f}, spot0 (fee-incl. marginal) {P.spot0:.6g}, "
          f"last_prices {P.last_prices:.6g}, bal {P.coins[P.i]} {P.bal0[P.i]:,.2f} / {P.coins[P.j]} {P.bal0[P.j]:,.0f}; file: block {mp.block}, spot {mp.spot:.6g}")
    for row in mp.rows:
        d = row["d"]; X = P.X_for_depth(d, "depth_marg"); r = P.sell(X)
        ratio = X / row["sell_usd"]; ok = abs(ratio - 1) <= 0.10
        fpb = row["open_cost_per_block"] + row["builder_cost_per_block"]
        Xb = Qb.X_for_depth(d, "depth_marg")
        out["a_sell_usd_vs_file"].append(dict(d_marg=d, sell_usd_exact=X, sell_usd_file=row["sell_usd"], ratio=ratio, within_10pct=ok,
                                               depth_eff_at_that_size=r["depth_eff"], sell_usd_exact_balanced_at_peg=Xb, ratio_balanced=Xb / row["sell_usd"]))
        out["b_roundtrip_vs_file_per_block"].append(dict(d_marg=d, roundtrip_loss_usd=r["roundtrip_loss_usd"], open_cost_usd=r["open_cost_usd"],
                                                          file_open=row["open_cost_per_block"], file_builder=row["builder_cost_per_block"], file_open_plus_builder=fpb,
                                                          ratio_roundtrip_over_file=r["roundtrip_loss_usd"] / fpb, ratio_open_over_file_open=r["open_cost_usd"] / row["open_cost_per_block"],
                                                          cost_per_boundary_rule=P.cost_rule(r), fee_rate_in=r["fee_rate_in"], fee_rate_back=r["fee_rate_back"]))
        out["pass_a"] &= ok
        print(f"  d_marg {100 * d:4.1f} %: sell exact ${X:11,.0f} vs file ${row['sell_usd']:11,.0f} (x{ratio:.3f}{' ok' if ok else ' OUT OF 10 %'}; balanced-at-peg state ${Xb:11,.0f} x{Xb / row['sell_usd']:.3f}) | d_eff {100 * r['depth_eff']:5.2f} % | "
              f"roundtrip ${r['roundtrip_loss_usd']:8,.0f} open ${r['open_cost_usd']:8,.0f} vs file open ${row['open_cost_per_block']:7,.0f} builder ${row['builder_cost_per_block']:7,.0f} (rt/file-sum x{r['roundtrip_loss_usd'] / fpb:.2f}, open/file-open x{r['open_cost_usd'] / row['open_cost_per_block']:.2f}) -> rule ${P.cost_rule(r):8,.0f} | fee in {100 * r['fee_rate_in']:.3f} % back {100 * r['fee_rate_back']:.3f} %")
    return out


def table(P, n=60):
    lo, hi = TABLE_RANGE[P.name]
    Xs = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    rows = []
    for x in Xs:
        r = P.sell(float(x))
        for m in P.MODES: r["cost_per_boundary_" + m] = P.cost_rule(r, m)
        rows.append(r)
    return rows


def mono_check(rows):
    keys = {"got_usd": 1, "marginal_after": -1, "effective_after": -1, "depth_marg": 1, "depth_eff": 1, "roundtrip_loss_usd": 1,
            "paid_back_usd": 1, "fee_rate_in": 1, "open_cost_usd": 1, "cost_per_boundary_roundtrip": 1, "cost_per_boundary_open": 1,
            "cost_per_boundary_file_sum": 1, "cost_per_boundary_rule": 1}
    res = {}
    for k, sgn in keys.items():
        v = np.array([r[k] for r in rows], float)
        dv = np.diff(v) * sgn
        res[k] = bool(np.all(dv >= -1e-12 * np.maximum(1.0, np.abs(v[1:]))))
    res["effective_gt_marginal_all"] = bool(all(r["effective_after"] > r["marginal_after"] for r in rows))
    return res


def main():
    T0 = time.time()
    out = {"state": STATE, "manip_cost": MANIP, "rule": "cost_per_boundary = max(roundtrip_loss, file open+builder at d_marg) for d_marg <= 5 %, roundtrip_loss beyond; pool state frozen",
           "pools": {}}
    for name in POOLS:
        t1 = time.time(); P = Pool(name); mp = P.manip
        val = validate(P, mp)
        rows = table(P)
        mono = mono_check(rows)
        val["pass_c"] = mono.pop("effective_gt_marginal_all")
        rule_mono = mono.pop("cost_per_boundary_rule")                 # the task's rule has a cliff at d_marg = 5 % by construction
        val["pass_d"] = all(mono.values()); val["d_monotone_by_field"] = mono; val["rule_cost_monotone"] = rule_mono
        g = P.grid()
        # interpolation error of the fast cost against the exact bisection at 8 depths
        errs = []
        for d in (0.003, 0.007, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5):
            if d <= g["depth_eff"][-1]:
                for m in ("rule", "open"):
                    ce, cf = P.cost_per_boundary(d, exact=True, mode=m), P.cost_per_boundary(d, mode=m)
                    errs.append(dict(d_eff=d, mode=m, exact=ce, fast=cf, rel_err=cf / ce - 1))
        val["fast_vs_exact_cost"] = errs; val["fast_cost_max_rel_err"] = max(abs(e["rel_err"]) for e in errs)
        val["e_solve_D"] = validate_D(P); val["pass_e"] = val["e_solve_D"]["pass_1e10"]
        val["f_cliff"] = validate_cliff(P); val["pass_f"] = val["f_cliff"]["pass_1e3"]
        e_ = val["e_solve_D"]; f_ = val["f_cliff"]
        print(f"  (e) solve_D vs independent Illinois root above the pole, 40 sizes to ${g['X_usd'][-1]:,.0f}: max |rel| {e_['max_abs_rel']:.1e}, "
              f"{e_['n_below_old_bracket']}/40 roots below the OLD bracket edge 0.5 sum(xp) -> {'PASS' if val['pass_e'] else 'FAIL'}")
        for x in e_["deep"]:
            print(f"      deep d_eff {100 * x['d_eff']:.0f} %: sell ${x['X_usd']:13,.0f} D_post {x['D_post']:.6g} roundtrip ${x['roundtrip_loss_usd']:11,.0f} open ${x['open_cost_usd']:13,.0f} rule ${x['rule']:11,.0f}")
        print(f"  (f) cliff: X ${f_['cliff']['X_usd']:,.0f}, d_eff {100 * f_['cliff']['depth_eff_below']:.4f} %: rule ${f_['cliff']['rule_below']:,.0f} below -> ${f_['cliff']['rule_above']:,.0f} above; "
              f"fast vs exact around it and inside the former cell {100 * f_['former_cell'][0]:.3f}..{100 * f_['former_cell'][1]:.3f} %: max |rel| {f_['max_abs_rel_err']:.1e} -> {'PASS' if val['pass_f'] else 'FAIL'}")
        # the cost at the grid's end and the X sizes for round depths
        marks = {}
        for d in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90, 0.95):
            if d <= g["depth_eff"][-1]:
                X = P.X_for_depth(d, "depth_eff"); r = P.sell(X)
                marks[f"{d:g}"] = dict(X_usd=X, got_usd=r["got_usd"], depth_marg=r["depth_marg"], roundtrip_loss_usd=r["roundtrip_loss_usd"], open_cost_usd=r["open_cost_usd"],
                                       cost_per_boundary={mm: P.cost_rule(r, mm) for mm in P.MODES}, fee_rate_in=r["fee_rate_in"])
        print(f"  monotone in X (exact quantities): {val['pass_d']} {mono}; the task's rule cost monotone: {rule_mono} (cliff at d_marg 5 %); effective > marginal everywhere: {val['pass_c']}; fast-cost interpolation max rel err {val['fast_cost_max_rel_err']:.2e}")
        print("  effective depth -> sell size; cost per boundary: rule | roundtrip (2 fees) | open (slippage + fee, arbs restore) | file open+builder (extrapolated > 5 %)")
        for k, m in marks.items():
            c = m["cost_per_boundary"]
            print(f"    d_eff {100 * float(k):5.1f} %: sell ${m['X_usd']:13,.0f} (d_marg {100 * m['depth_marg']:6.2f} %, fee {100 * m['fee_rate_in']:.3f} %)  rule ${c['rule']:13,.0f} | rt ${c['roundtrip']:12,.0f} | open ${c['open']:12,.0f} | file ${c['file_sum']:12,.0f}")
        gg = {k: v for k, v in g.items() if not k.startswith("ln_")}
        val["cliff"] = g["cliff"]
        out["pools"][name] = dict(coins=P.coins, sold=P.coins[P.i], quote=P.coins[P.j], block=P.block, ts_utc=P.ts_utc, spot0=P.spot0, last_prices=P.last_prices,
                                  D=P.D0, balances=P.bal0, A=P.A, gamma=P.gamma, mid_fee=P.mid, out_fee=P.out, fee_gamma=P.fg, price_scale=P.PS,
                                  manip_file=dict(block=mp.block, spot=mp.spot, tvl_usd=mp.tvl_usd, rows=mp.rows),
                                  validation=val, table=rows, depth_marks=marks, cost_grid=gg, secs=round(time.time() - t1, 1))
        print(f"  [{name} {time.time() - t1:.1f} s]")
    out["pass_all"] = all(out["pools"][n]["validation"][k] for n in POOLS for k in ("pass_a", "pass_c", "pass_d", "pass_e", "pass_f"))
    out["runtime_s"] = round(time.time() - T0, 1)
    json.dump(out, open(os.path.join(HERE, "poolmath_tables.json"), "w"), indent=1)
    print(f"\npoolmath {'PASS' if out['pass_all'] else 'FAIL'} in {out['runtime_s']} s -> r3/poolmath_tables.json")


if __name__ == "__main__":
    main()
