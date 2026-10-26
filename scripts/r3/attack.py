#!/usr/bin/env python3
"""attack.py — round 3 (LLAMMA-Design/r3, 14 Oct 2026): the attack model on the block grid and the cheapest path per design.
Rewritten after the T2 review (optimizer completeness was a blocker: the staircase/hold grid could not express a slope
matched to a design's rate limit nor a path that tracks the oracle at a fixed gap; today/open optimum is a long shallow ramp).

The grid: every block is a trade block, 12 s apart; a calm lead of LEAD = 50 blocks at depth 0, then the attack blocks
t = 1..T with EFFECTIVE depths d_t: the pool's last_prices after block t = spot0 (1 - d_t), spot0 = 1 (normalized).
Oracle sampling (the view semantics of the pool contract): the design's oracle is sampled at every block END + 12 s - 1e-6,
i.e. at the moment the NEXT block opens, before any trade of that block = what the first trade of block t+1 (a borrower, a
liquidator, the LLAMMA) sees. For the pool EMA that is e_t = e_{t-1} alpha + p_t (1 - alpha), alpha = exp(-12/865): the
spot of block t has entered once. (bdsim/run_bdsim.oracle_on_blocks samples at the block times = the same values shifted by
one index; r3/parity.py: max |ln| difference 6e-11.)
max_drop = 1 - min(oracle) (FRACTION; the engine's X rows price the book at oracle x (1 - m)); max_drop_log = -ln min(oracle).
cost(depths) = sum over the attack blocks with d_t > 0 of poolmath.Pool.cost_per_boundary(d_t): one block boundary =
the dump as the last trade of block t and the buy-back as the first trade of block t+1; the un-dump of the final block is
inside its round trip; no other terms. Cost modes (poolmath.Pool.cost_rule): 'rule' (the task's: max(round trip, file
open + builder) for d_marg <= 5 %, round trip beyond - has a cliff at 5 %), 'roundtrip', 'open', 'file_sum'.
Families (every one is a depth vector; the attacker knows the oracle, so any depth vector is an admissible attack):
  hold(d, K); alternating(d, cycles) = d, 0, d, 0, ...; sandwich(d, K) = hold (identical from the end-of-block view);
  staircase(d0, step, n, K2, newlow): n blocks d0, d0 + step, ..., then K2 blocks of new lows (the spot falls by `newlow`
    per block) at the reached depth; brutal(pool, X_usd, K) = hold at the effective depth of a sell of X_usd;
  ramp(d0, s): d0, d0 + s, d0 + 2 s, ... (capped at DMAX); two_speed(d0, s1, n1, s2): n1 blocks at slope s1 then slope s2;
  gap_track(label, gap, nl, d0): CAUSAL - block t's depth is set from the oracle the attacker sees at the open of block t:
    d_t = max(1 - o_{t-1} (1 - gap), new low by nl per block, d_{t-1} if nl = 0): the spot is held `gap` under the oracle
    (a chasing oracle with half-life hf moves (1 - exp(-12 ln2 / hf)) x gap per block, capped at vfast; the new lows keep a
    regime's decline test true);
  trigger(label, d_big, n_up, n_flat, nl, tiny) - the CHEAP TRIGGER of a crash regime (added after the T2 review, which found
    hand-built triggers 20-75 % cheaper than every family above): a prefix of M-1 TINY new lows (each `tiny` = 1.1 eps under the
    previous spot: a 'down' step of the ring, at a depth of 0.02-0.1 % that costs next to nothing), then ONE big step to d_big
    >= 1 - exp(-D) (the decline test: the spot N blocks ago is still the calm spot) = M downs in the ring -> the crash regime is
    ON and chases the LAST spot with half-life hf; then n_flat flat blocks at d_big (the ring keeps its M downs while the
    flats only overwrite the calm lead), then new lows at the rate nl per block (nl = 0: hold). The ring has N - M slots the
    prefix does not need: n_up of them are spent INSIDE the prefix as 'up' blocks (depth 0 = the arbitrageurs restore the
    spot, cost 0) that split the M-1 tiny downs into n_up + 1 runs of 1-3 steps, so the cumulative depth of the tiny steps
    (and their cost) restarts from 0 - cheaper than one run of M-1 ever deeper steps; n_up + n_flat <= N - M (the builder
    clamps n_up to N - M; n_flat may exceed the rest: the regime then lapses and the path is a calm hold with a head start);
  trigger_gap(label, d_big, n_up, n_flat, gap, nl, tiny): the same prefix + big step + flats, continued CAUSALLY by the
    gap track (block t's depth = max(1 - o_{t-1} (1 - gap), new low by nl, d_{t-1})).
  lagrangian (today, c1200, c1800 = the LINEAR designs: the oracle drop after block K is sum_t w_{K-t} d_t with the impulse
    weights w_j read off the oracle itself, verify() checks superposition): the exact optimum over ALL depth paths of length
    K <= KMAX on a 3000-point d grid by Lagrangian relaxation - min sum c(d_t) s.t. drop >= m; phi(mu) = min_d c(d) - mu d;
    the dual bound max_lambda [lambda m + sum_j phi(lambda w_j)] is a LOWER BOUND on the cost of every path of length
    K <= LAGK = 375 blocks (the longest path any family builds: a staircase of 75 steps + 300 new lows), stored per cell as
    'lower_bound' with 'bound_at_K_cap' = True when the bound's argmin K is the cap (a longer path could then be cheaper);
    the primal path at the smallest feasible lambda (+ a trim of the last block) is the path.
Optimizer min_cost: every family's paths are evaluated ONCE per design (the running maximum of the drop over the attack
  blocks gives the cheapest prefix reaching every m); hold over 60 log-spaced d in [0.25 %, 95 %] x K in 1..300; alternating
  over 40 d x cycles 1..150; staircase over D0S x STEPS x NS x K2 in 0..300 with newlow 0.05 % (the task's) and, for the
  designs with a crash regime, 1.05 D/N per block ('staircase_nl'); ramp over R_D0 x R_S (plus the design's own rates
  1.05 D/N, vfast, 1.1 vfast, vslow and the pools' cliff depths); two_speed over T_D0 x T_S1 x T_N1 x T_S2; gap_track over
  G_GAP x nl in {0, 1.05 D/N} x G_D0; trigger (regime designs only) over d_big in TRIG_DBIG x (1 - exp(-D)) plus the pools'
  cliff depths x TRIG_CLIFF_F (1.0-1.02) plus the absolute TRIG_DBIG_ABS (15-90 %) x n_up in a spread of 0..N-M x n_flat in {0, N-M-n_up, K2MAX (the hold variant, nl = 0)} x nl in {0, 1.05 / 1.5 /
  2.5 / 4 D/N, 1.05 D/(M-1), vfast}; trigger_gap over d_big in {1.002 (1 - exp(-D)), the cliffs, 5 %, 10 %} x TRIG_GAP x
  n_up in {0, N-M} x n_flat in {0, N-M-n_up}, nl = 1.05 D/N. Then per cell (design, m, pool, mode): the cheapest path is REFINED by a coordinate
  search on its family's continuous parameters (x (1 +- delta), delta 25 % -> 0.5 %; `tiny` never below 1.02 eps) and
  integers (+-1, +-2; n_up / n_flat may reach 0, the others stay >= 1), and its last
  block is trimmed to the smallest depth that still reaches m; every accepted step is re-evaluated with the real oracle; the
  runner-up family is refined the same way when its grid cost is within REFINE_SECOND = 15 % of the winner's (the cheaper wins).
  Returns per cell the cheapest path (with its depth vector), the best of every family and, for linear designs, the bound.
MS = the engine's 11 exposure drops (replay_sim_x.cpp ms[NM]) so a cost-vs-gain join covers every column.
Usage: cd r3 && nice -n 19 python3 attack.py [--workers 4] [--verify-only]  -> r3/attack_mincost.json, r3/attack_verify.json"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from r3 import common as R3  # noqa: E402
from r3 import poolmath as PM  # noqa: E402

LEAD = 50
DT = 12.0
TAIL = 60                                  # relaxation blocks after the attack (the max drop is inside the attack; the tail proves it)
DMIN, DMAX, ND = 0.0025, 0.95, 60
NDALT = 40
KMAX, CMAX = 300, 150
D0S = (0.0025, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
STEPS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02)
NS = (2, 3, 4, 5, 7, 10, 15, 25, 35, 50, 75)
K2MAX = 300
NEWLOW = 0.0005
R_D0 = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15)
R_S = (0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005, 0.006, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.05)
T_D0 = (0.003, 0.005, 0.01, 0.02, 0.025, 0.03, 0.05)
T_S1 = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02)
T_N1 = (2, 3, 5, 7, 10, 15, 20, 25, 30, 40)
T_S2 = (0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075)
G_GAP = (0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20)
G_D0 = (0.005, 0.01, 0.02, 0.026, 0.03)
TINY = 1.1 * R3.EPS                        # a tiny new low: 0.022 % under the previous spot (a 'down' step needs > eps = 0.02 %)
TINY_MIN = 1.02 * R3.EPS                   # the refine never goes below this (the step must stay a down step)
DBIG_MARGIN = 5e-4                         # d_big >= (1 - exp(-D)) (1 + margin): the decline test log(1 / (1 - d_big)) >= D holds with room
TRIG_DBIG = (1.002, 1.01, 1.03, 1.06, 1.1, 1.15, 1.2, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0)   # x (1 - exp(-D))
TRIG_DBIG_ABS = (0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90)   # absolute big steps too (one giant step is cheap under file_sum's concave extrapolation)
TRIG_GAP = (0.005, 0.008, 0.0125, 0.02, 0.03, 0.05, 0.08, 0.12)
TRIG_CLIFF_F = (1.0, 1.002, 1.005, 1.01, 1.02)   # d_big points just above each pool's 'rule' cliff (cliff_depths() x f)
TRIG_GAP_DBIG = (0.05, 0.10)               # plus 1.002 (1 - exp(-D)) and the cliff depths
TRIG_NL_STEEP = (0.01, 0.015, 0.025, 0.04) # steep new-low rates after the trigger (a cheap prefix + stairs), besides the design's own rates
MS = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30)       # = engine ms[NM]
POOLS = ("TricryptoUSDC", "TriCRV")
MODES = PM.Pool.MODES
LINEAR = ("today", "c1200", "c1800")       # EMA designs: the drop is linear in the depths (verify() checks superposition)
LAG_ND, LAG_NMU = 3000, 8000
LAGK = max(NS) + K2MAX                     # the longest path any family builds = the K range of the Lagrangian bound
REF_DELTAS = (0.25, 0.10, 0.04, 0.015, 0.005)
REF_DELTAS_SLOW = (0.20, 0.08)             # gap_track trials are causal (one oracle call per block): fewer steps
REFINE_SECOND = 0.15                       # the runner-up family is refined too when its grid cost is within 15 % of the winner's
_POOL = {}
_LAG = {}                                  # (pool, mode) -> phi / argmin tables of the Lagrangian


def pool(name):
    if name not in _POOL:
        _POOL[name] = PM.Pool(name); _POOL[name].grid()
    return _POOL[name]


def engine_ms():
    """the engine's exposure drops, parsed from replay_sim_x.cpp (ms[NM] = {...})"""
    src = open(os.path.join(ROOT, "engine", "replay_sim_x.cpp")).read()
    m = re.search(r"const double ms\[NM\] = \{([^}]*)\}", src)
    return tuple(float(x) for x in m.group(1).split(","))


# ------------------------------------------------------------------------------------------------------ the grid
def grid(depths, tail=TAIL):
    d = np.concatenate([np.zeros(LEAD), np.asarray(depths, float), np.zeros(tail)])
    t = np.arange(len(d), dtype=float) * DT
    return t, 1.0 - d, t + DT - 1e-6


RESTORE = False                            # round 10: True = the arbitrageurs' buy-back inside each block is read by the oracle (open
                                           # mode with reads inside blocks): the window oracle's in-block max is the market (1.0);
                                           # False = only the block end is read (the attacker's sell is the block's last trade)


def oracle_path(label, depths, tail=TAIL):
    """the design's oracle sampled at every block end + 12 s - 1e-6 on the lead + attack + tail grid; drops as fractions"""
    t, p, at = grid(depths, tail)
    if RESTORE and (R3.SPEC.get(label) or R3.spec(label))["kind"] == "cband2":
        o = np.asarray(R3.oracle(label, at, t, p, pmin=p, pmax=np.ones_like(p)), float)
    else:
        o = np.asarray(R3.oracle(label, at, t, p), float)
    drop = 1.0 - o
    k = int(np.argmax(drop))
    return dict(t=t, p=p, at=at, oracle=o, drop=drop, max_drop=float(drop[k]), max_drop_log=float(-math.log(o[k])),
                argmax_block=k, n_attack=len(depths))


def runmax(label, depths):
    """running maximum of the drop over the attack blocks (sampled at their block ends): runmax[j] = the deepest drop reached
    by the first j+1 attack blocks = the max drop of the path truncated after block j+1 (restoring the spot afterwards only
    lifts every design's oracle)"""
    return np.maximum.accumulate(oracle_path(label, depths, tail=0)["drop"][LEAD:])


# ------------------------------------------------------------------------------------------------------ families
def hold(d, K): return np.full(int(K), float(d))
def sandwich(d, K): return hold(d, K)
def alternating(d, cycles): return np.tile([float(d), 0.0], int(cycles))


def staircase(d0, step, n, K2, newlow=NEWLOW):
    s = d0 + step * np.arange(int(n))
    last = s[-1]
    h = 1.0 - (1.0 - last) * (1.0 - newlow) ** np.arange(1, int(K2) + 1)
    return np.concatenate([s, h])


def ramp(d0, s, n=KMAX, cap=DMAX):
    return np.minimum(d0 + s * np.arange(int(n)), cap)


def two_speed(d0, s1, n1, s2, n=KMAX, cap=DMAX):
    n1 = max(1, min(int(n1), n))
    a = d0 + s1 * np.arange(n1); b = a[-1] + s2 * np.arange(1, n - n1 + 1)
    return np.minimum(np.concatenate([a, b]), cap)


def gap_track(label, gap, nl, d0, n=KMAX, cap=DMAX, m_stop=None):
    """causal: block t's depth = max(1 - o_{t-1} (1 - gap), new low by nl per block, d_{t-1} if nl = 0), from d0; o_{t-1} = the
    oracle the attacker sees at the open of block t (one oracle call per block); stops once the drop seen is >= m_stop"""
    dep = [float(d0)]
    for t in range(1, int(n)):
        o = oracle_path(label, np.array(dep), tail=0)["oracle"][-1]
        if m_stop is not None and 1.0 - o >= m_stop: break
        d_gap = 1.0 - o * (1.0 - gap)
        d_nl = 1.0 - (1.0 - dep[-1]) * (1.0 - nl)
        d = min(cap, max(d_gap, d_nl, dep[-1] if nl == 0 else 0.0))
        dep.append(d)
    return np.array(dep)


def brutal(pool_name, X_usd, K):
    d = pool(pool_name).sell(X_usd)["depth_eff"]
    return hold(d, K), d


def prefix(M, n_up, tiny=TINY):
    """the trigger's prefix: M-1 tiny down steps (each `tiny` under the previous spot) split into n_up + 1 runs (as equal as
    possible, the longer runs first) separated by n_up 'up' blocks of depth 0; the depth after the k-th step of a run is
    1 - (1 - tiny)^k. Length M - 1 + n_up blocks; with the big step that follows, M + n_up <= N blocks = M downs in the ring."""
    nd, runs = M - 1, n_up + 1
    base, extra = divmod(nd, runs)
    out = []
    for r in range(runs):
        L = base + (1 if r < extra else 0)
        out.extend((1.0 - (1.0 - tiny) ** np.arange(1, L + 1)).tolist())
        if r < runs - 1: out.append(0.0)
    return np.array(out, float)


def _trigger_head(label, d_big, n_up, n_flat, tiny):
    """prefix + the big step + n_flat flat blocks at d_big (clamped: n_up in 0..N-M, n_flat >= 0, tiny >= TINY_MIN,
    d_big >= (1 - exp(-D)) (1 + DBIG_MARGIN) and <= DMAX); returns (depths, N, M)"""
    kw = (R3.SPEC.get(label) or R3.spec(label))["kwargs"]; N, M, D = kw["N"], kw["M"], kw["D"]
    n_up = int(max(0, min(int(n_up), N - M))); n_flat = int(max(0, int(n_flat)))
    tiny = max(float(tiny), TINY_MIN)
    d_big = min(max(float(d_big), (1.0 - math.exp(-D)) * (1.0 + DBIG_MARGIN)), DMAX)
    return np.concatenate([prefix(M, n_up, tiny), [d_big], np.full(n_flat, d_big)]), N, M


def trigger(label, d_big, n_up, n_flat, nl, tiny=TINY, K2=K2MAX):
    """the cheap trigger (module docstring): prefix(M, n_up, tiny), one step to d_big, n_flat flats, then K2 blocks of new
    lows at the rate nl per block (nl = 0: nothing more; the path ends after the flats)"""
    head, N, M = _trigger_head(label, d_big, n_up, n_flat, tiny)
    if nl > 0:
        lows = 1.0 - (1.0 - head[-1]) * (1.0 - nl) ** np.arange(1, int(K2) + 1)
        head = np.concatenate([head, lows])
    return np.minimum(head, DMAX)


def trigger_gap(label, d_big, n_up, n_flat, gap, nl, tiny=TINY, n=KMAX, cap=DMAX, m_stop=None):
    """the cheap trigger continued causally by the gap track: after the head, block t's depth = max(1 - o_{t-1} (1 - gap),
    new low by nl, d_{t-1}); o_{t-1} = the oracle seen at the open of block t (one oracle call per block)"""
    head, N, M = _trigger_head(label, d_big, n_up, n_flat, tiny)
    dep = head.tolist()
    for t in range(len(dep), int(n)):
        o = oracle_path(label, np.array(dep), tail=0)["oracle"][-1]
        if m_stop is not None and 1.0 - o >= m_stop: break
        d_gap = 1.0 - o * (1.0 - gap)
        d_nl = 1.0 - (1.0 - dep[-1]) * (1.0 - nl)
        dep.append(min(cap, max(d_gap, d_nl, dep[-1])))
    return np.array(dep)


def cost(depths, pool_name="TricryptoUSDC", mode="rule"):
    """sum over the attack blocks with d_t > 0 of the cost of one block boundary at d_t (inf when a depth is > 95 %)"""
    d = np.asarray(depths, float); d = d[d > 0]
    if len(d) == 0: return 0.0
    if d.max() > DMAX: return float("inf")
    return float(np.sum(pool(pool_name).cost_per_boundary(d, mode=mode)))


def design_rates(label):
    """the design's own per-block rates (fractions): 1.05 D/N (the smallest new low that keeps the regime's decline test true),
    vfast, 1.1 vfast, vslow - the slopes a rate-limited oracle can be made to follow"""
    s = R3.SPEC.get(label) or R3.spec(label); kw = s["kwargs"]; out = {}
    if R3.is_regime(label): out["DN"] = 1.0 - math.exp(-1.05 * kw["D"] / kw["N"])
    if s["kind"] == "regime":
        if kw.get("vfast") is not None: out["vfast"] = 1.0 - math.exp(-kw["vfast"]); out["vfast11"] = 1.0 - math.exp(-1.1 * kw["vfast"])
        if kw.get("vslow") is not None: out["vslow"] = 1.0 - math.exp(-kw["vslow"])
    return out


def cliff_depths():
    """the effective depth just above each pool's 'rule' cliff (d_marg = 5 %): the cheapest depth of the round-trip regime"""
    return tuple(float(pool(pn).grid()["cliff"]["depth_eff_above"] * 1.002) for pn in POOLS)


# ------------------------------------------------------------------------------------------------------ family builders
FAMILIES = {
    # family: (continuous params, integer params, builder(params, label, m) -> depths)
    "hold": (("d",), (), lambda p, l, m: hold(p["d"], KMAX)),
    "alternating": (("d",), (), lambda p, l, m: alternating(p["d"], CMAX)),
    "staircase": (("d0", "step", "newlow"), ("n",), lambda p, l, m: staircase(p["d0"], p["step"], p["n"], K2MAX, p["newlow"])),
    "staircase_nl": (("d0", "step", "newlow"), ("n",), lambda p, l, m: staircase(p["d0"], p["step"], p["n"], K2MAX, p["newlow"])),
    "ramp": (("d0", "s"), (), lambda p, l, m: ramp(p["d0"], p["s"])),
    "two_speed": (("d0", "s1", "s2"), ("n1",), lambda p, l, m: two_speed(p["d0"], p["s1"], p["n1"], p["s2"])),
    "gap_track": (("gap", "nl", "d0"), (), lambda p, l, m: gap_track(l, p["gap"], p["nl"], p["d0"], m_stop=m)),
    "trigger": (("d_big", "nl", "tiny"), ("n_up", "n_flat"), lambda p, l, m: trigger(l, p["d_big"], p["n_up"], p["n_flat"], p["nl"], p["tiny"])),
    "trigger_gap": (("d_big", "gap", "nl", "tiny"), ("n_up", "n_flat"),
                    lambda p, l, m: trigger_gap(l, p["d_big"], p["n_up"], p["n_flat"], p["gap"], p["nl"], p["tiny"], m_stop=m)),
    "lagrangian": ((), (), None),
}
INT_MIN = {"n_up": 0, "n_flat": 0}         # integer parameters that may be 0 (every other integer stays >= 1 in the refine)
CONT_MIN = {"tiny": TINY_MIN}              # continuous parameters with a floor in the refine
CAUSAL = ("gap_track", "trigger_gap")      # one oracle call per block when built: fewer refine steps


def canon_params(label, family, prm):
    """the trigger families' parameters as the builder applies them (n_up clamped to 0..N-M, n_flat >= 0, tiny >= TINY_MIN, d_big
    clamped to [(1 - exp(-D)) (1 + DBIG_MARGIN), DMAX]): the stored parameters then rebuild the stored path without a hidden clamp"""
    if family not in ("trigger", "trigger_gap"): return prm
    kw = (R3.SPEC.get(label) or R3.spec(label))["kwargs"]; N, M, D = kw["N"], kw["M"], kw["D"]
    out = dict(prm)
    out["n_up"] = int(max(0, min(int(prm["n_up"]), N - M))); out["n_flat"] = int(max(0, int(prm["n_flat"])))
    out["tiny"] = max(float(prm["tiny"]), TINY_MIN)
    out["d_big"] = min(max(float(prm["d_big"]), (1.0 - math.exp(-D)) * (1.0 + DBIG_MARGIN)), DMAX)
    return out


def build(family, params, label, m=None):
    if family in ("trigger", "trigger_gap"): params.update(canon_params(label, family, params))
    return FAMILIES[family][2](params, label, m)


def _paths(label):
    """every candidate path of the search for one design, with its running-max drop over the attack blocks:
    a list of (family, params, depths, runmax[n_attack])"""
    out = []
    rates = design_rates(label); cl = cliff_depths()
    for d in np.exp(np.linspace(math.log(DMIN), math.log(DMAX), ND)):
        dep = hold(d, KMAX); out.append(("hold", dict(d=float(d)), dep, runmax(label, dep)))
    for d in np.exp(np.linspace(math.log(DMIN), math.log(DMAX), NDALT)):
        dep = alternating(d, CMAX); out.append(("alternating", dict(d=float(d)), dep, runmax(label, dep)))
    nls = [("staircase", NEWLOW)]
    if "DN" in rates: nls.append(("staircase_nl", rates["DN"]))
    for fam, nl in nls:
        for d0 in D0S:
            for st in STEPS:
                for n in NS:
                    if d0 + st * (n - 1) > DMAX: continue
                    dep = staircase(d0, st, n, K2MAX, nl)
                    out.append((fam, dict(d0=d0, step=st, n=n, newlow=nl), dep, runmax(label, dep)))
    r_d0 = sorted(set(R_D0) | set(cl)); r_s = sorted(set(R_S) | set(rates.values()))
    for d0 in r_d0:
        for s in r_s:
            dep = ramp(d0, s); out.append(("ramp", dict(d0=float(d0), s=float(s)), dep, runmax(label, dep)))
    t_d0 = sorted(set(T_D0) | set(cl)); t_s1 = sorted(set(T_S1) | ({rates["DN"]} if "DN" in rates else set()))
    t_s2 = sorted(set(T_S2) | {v for k, v in rates.items() if k in ("DN", "vfast")})
    for d0 in t_d0:
        for s1 in t_s1:
            for n1 in T_N1:
                for s2 in t_s2:
                    dep = two_speed(d0, s1, n1, s2)
                    out.append(("two_speed", dict(d0=float(d0), s1=float(s1), n1=int(n1), s2=float(s2)), dep, runmax(label, dep)))
    g_nl = [0.0] + ([rates["DN"]] if "DN" in rates else [])
    for gp in G_GAP:
        for nl in g_nl:
            for d0 in G_D0:
                dep = gap_track(label, gp, nl, d0, m_stop=max(MS) * 1.02)
                out.append(("gap_track", dict(gap=float(gp), nl=float(nl), d0=float(d0)), dep, runmax(label, dep)))
    if R3.is_regime(label):
        kw = (R3.SPEC.get(label) or R3.spec(label))["kwargs"]; N, M, D = kw["N"], kw["M"], kw["D"]
        dD = 1.0 - math.exp(-D); U = N - M
        ups = sorted(set([0, U] + [int(round(U * f)) for f in (0.25, 0.5, 0.75)] + ([1, 2] if U >= 2 else [])))
        nls = sorted(set([0.0] + [1.0 - math.exp(-f * D / N) for f in (1.05, 1.5, 2.5, 4.0)] + [1.0 - math.exp(-1.05 * D / max(M - 1, 1))]
                         + ([rates["vfast"]] if "vfast" in rates else []) + list(TRIG_NL_STEEP)))
        cl_pts = [c * f for c in cl for f in TRIG_CLIFF_F if c * f >= dD * (1 + DBIG_MARGIN)]       # just above each pool's rule cliff
        dbigs = sorted(set([min(f * dD, DMAX) for f in TRIG_DBIG] + cl_pts + [x for x in TRIG_DBIG_ABS if x >= dD * (1 + DBIG_MARGIN)]))
        for d_big in dbigs:
            for n_up in ups:
                combos = sorted(set([(0, nl) for nl in nls] + [(U - n_up, nl) for nl in nls] + [(K2MAX, 0.0)]))
                for n_flat, nl in combos:
                    prm = canon_params(label, "trigger", dict(d_big=float(d_big), n_up=int(n_up), n_flat=int(n_flat), nl=float(nl), tiny=TINY))
                    dep = trigger(label, d_big, n_up, n_flat, nl, TINY)
                    out.append(("trigger", prm, dep, runmax(label, dep)))
        g_dbig = sorted(set([dD * (1.0 + 2e-3)] + cl_pts + [x for x in TRIG_GAP_DBIG if x >= dD * (1 + DBIG_MARGIN)]))
        nl_g = 1.0 - math.exp(-1.05 * D / N)
        for d_big in g_dbig:
            for gp in TRIG_GAP:
                for n_up in sorted(set([0, U])):
                    for n_flat in sorted(set([0, U - n_up])):
                        prm = canon_params(label, "trigger_gap", dict(d_big=float(d_big), n_up=int(n_up), n_flat=int(n_flat), gap=float(gp), nl=float(nl_g), tiny=TINY))
                        dep = trigger_gap(label, d_big, n_up, n_flat, gp, nl_g, TINY, m_stop=max(MS) * 1.02)
                        out.append(("trigger_gap", prm, dep, runmax(label, dep)))
    return out


# ------------------------------------------------------------------------------------------------------ the Lagrangian (linear designs)
def impulse_weights(label, delta=0.01, n=LAGK):
    """w_j, j = 0..n-1: the oracle drop seen at the sample after block K from a unit depth at block K-j (the drop after
    block K is sum_t w_{K-t} d_t for a linear design); read off the oracle itself with one dump of depth delta"""
    dep = np.zeros(n); dep[0] = delta
    r = oracle_path(label, dep, tail=0)["drop"][LEAD:]
    return r / delta


def linearity_check(label, w, seed=7, n=5):
    """superposition on random paths: max |drop(oracle) - sum w d| over the attack blocks"""
    rng = np.random.default_rng(seed); worst = 0.0; nk = len(w)
    for _ in range(n):
        dep = rng.uniform(0.0, 0.3, nk); dep[rng.uniform(size=nk) < 0.3] = 0.0
        r = oracle_path(label, dep, tail=0)["drop"][LEAD:]
        lin = np.convolve(dep, w)[:nk]
        worst = max(worst, float(np.max(np.abs(r - lin))))
    return worst


def lag_tables(pool_name, mode):
    """phi(mu) = min_d c(d) - mu d and its argmin on a log mu grid, for the cost c of (pool, mode) on the Lagrangian d grid"""
    key = (pool_name, mode)
    if key not in _LAG:
        P = pool(pool_name)
        dg = np.concatenate([[0.0], np.exp(np.linspace(math.log(1e-4), math.log(DMAX), LAG_ND))])
        cg = np.concatenate([[0.0], P.cost_per_boundary(dg[1:], mode=mode)])
        mu = np.exp(np.linspace(math.log(1e-4), math.log(1e13), LAG_NMU))
        phi = np.empty(LAG_NMU); arg = np.empty(LAG_NMU, int)
        for a in range(0, LAG_NMU, 500):
            v = cg[None, :] - mu[a:a + 500, None] * dg[None, :]
            arg[a:a + 500] = np.argmin(v, axis=1); phi[a:a + 500] = v[np.arange(len(v)), arg[a:a + 500]]
        _LAG[key] = dict(dg=dg, cg=cg, lmu=np.log(mu), phi=phi, arg=arg)
    return _LAG[key]


def lagrangian(label, w, m, pool_name, mode, margin=2e-6):
    """the exact optimum (on the d grid) over all paths of length K <= len(w) (= LAGK) for a linear design: dict(best primal, lower_bound).
    The primal targets m (1 + margin) so that the real oracle (sampled 1e-6 s before the block, weights off by ~1e-9) and the
    stored depths (rounded to 1e-9) still reach m; the bound is computed for m itself."""
    T = lag_tables(pool_name, mode); dg, cg, lmu, phi, arg = T["dg"], T["cg"], T["lmu"], T["phi"], T["arg"]
    m_bound = m; m = m * (1.0 + margin)
    lw = np.log(np.maximum(w, 1e-300)); nk = len(w)
    lam = np.exp(np.linspace(math.log(1e0), math.log(1e13), 700))

    def dual_all_K(lam_v):                                   # L(lambda, K) for every K at once: cumulative sums of phi(lambda w_j)
        ph = np.interp(np.log(lam_v)[:, None] + lw[None, :], lmu, phi)      # (nlam, KMAX)
        return lam_v[:, None] * m_bound + np.cumsum(ph, axis=1)

    L = dual_all_K(lam)                                       # (nlam, nk)
    DK = L.max(axis=0); kb = int(np.argmin(DK)); lb = float(DK[kb])
    # sharpen the bound of the best K by a golden-section on log lambda around its grid optimum
    i0 = int(np.argmax(L[:, kb])); a, b = math.log(lam[max(i0 - 1, 0)]), math.log(lam[min(i0 + 1, len(lam) - 1)])
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        c1, c2 = b - gr * (b - a), a + gr * (b - a)
        f1 = dual_all_K(np.array([math.exp(c1)]))[0, kb]; f2 = dual_all_K(np.array([math.exp(c2)]))[0, kb]
        if f1 < f2: a = c1
        else: b = c2
    lb = max(lb, float(dual_all_K(np.array([math.exp((a + b) / 2)]))[0, kb]))

    def primal(lam_s, K):                                     # depths of the argmins at lambda for the path of length K
        mu_j = np.log(lam_s) + lw[:K]                         # block t = K-j: weight w_j
        idx = arg[np.clip(np.searchsorted(lmu, mu_j), 0, len(lmu) - 1)]
        d = dg[idx][::-1].copy()                              # d_t for t = 1..K (j = K - t)
        return d, float(np.dot(w[:K][::-1], d)), float(cg[idx].sum())

    best = None
    for K in range(1, nk + 1):
        i0 = int(np.argmax(L[:, K - 1])); lam_used = lam[i0]
        d, drop, c = primal(lam_used, K)
        if drop < m:                                          # the smallest lambda whose primal reaches m (the drop is nondecreasing in lambda)
            lo, hi = lam_used, lam_used * 1e4
            if primal(hi, K)[1] < m: continue
            for _ in range(50):
                mid = math.sqrt(lo * hi)
                if primal(mid, K)[1] >= m: hi = mid
                else: lo = mid
            lam_used = hi; d, drop, c = primal(lam_used, K)
        # trim the last block to hit m exactly (linear model); keep when cheaper
        w0 = w[0]; rest = drop - w0 * d[-1]; dK = (m - rest) / w0
        if 0.0 < dK < d[-1]:
            d2 = d.copy(); d2[-1] = dK; c2 = cg[0] * 0 + float(np.sum(pool(pool_name).cost_per_boundary(d2[d2 > 0], mode=mode)))
            if c2 < c: d, drop, c = d2, float(np.dot(w[:K][::-1], d2)), c2
        if best is None or c < best["cost"]:
            best = dict(family="lagrangian", params=dict(K=K, lam=float(lam_used)), depths=d, cost=c, drop_linear=drop, K=K)
    return dict(best=best, lower_bound=lb, K_bound=kb + 1, K_max=nk, bound_at_K_cap=(kb + 1 == nk))


# ------------------------------------------------------------------------------------------------------ cells
def _record(fam, prm, dep, j, P, mode, run_j):
    pre = np.asarray(dep[:j + 1], float)
    c = float(np.sum(P.cost_per_boundary(pre[pre > 0], mode=mode))) if (pre > 0).any() else 0.0
    return dict(family=fam, params=prm, blocks=j + 1, cost=c, drop=float(run_j), cost_per_pct=c / (100.0 * run_j),
                depth_max=float(pre.max()), blocks_off=int((pre > 0).sum()), depths=pre)


def _best_of(paths, m, P, mode):
    """per family the cheapest (path prefix) that reaches the drop m; the cost of a prefix = the cumulative boundary cost"""
    best = {}
    for fam, prm, dep, run in paths:
        hit = np.where(run >= m)[0]
        if len(hit) == 0: continue
        j = int(hit[0])                                   # the first attack block (0-based) after which the drop is reached
        if (dep[:j + 1] > DMAX).any(): continue
        rec = _record(fam, prm, dep, j, P, mode, run[j])
        if not np.isfinite(rec["cost"]): continue
        if fam not in best or rec["cost"] < best[fam]["cost"]: best[fam] = rec
    return best


def eval_path(label, fam, prm, dep, m, P, mode):
    """the cheapest prefix of `dep` reaching m, or None"""
    run = runmax(label, dep); hit = np.where(run >= m)[0]
    if len(hit) == 0: return None
    j = int(hit[0])
    if (dep[:j + 1] > DMAX).any(): return None
    rec = _record(fam, prm, dep, j, P, mode, run[j])
    return rec if np.isfinite(rec["cost"]) else None


def refine(label, rec, m, P, mode):
    """coordinate search on the family's parameters around the best path (multiplicative steps on the continuous ones,
    +-1 / +-2 on the integers), greedy, then a trim of the last block; every step evaluated with the real oracle"""
    fam = rec["family"]; cont, ints, _ = FAMILIES[fam]
    n_eval = 0
    if fam != "lagrangian" and (cont or ints):
        deltas = REF_DELTAS_SLOW if fam in CAUSAL else REF_DELTAS
        for delta in deltas:
            for _round in range(3 if fam not in CAUSAL else 1):
                improved = False
                for k in cont:
                    for f in (1.0 + delta, 1.0 - delta):
                        prm = dict(rec["params"]); prm[k] = prm[k] * f
                        if k in CONT_MIN: prm[k] = max(prm[k], CONT_MIN[k])
                        if prm[k] < 0 or (k != "newlow" and k != "nl" and prm[k] <= 0) or prm[k] > DMAX: continue
                        if prm[k] == rec["params"][k]: continue
                        cand = eval_path(label, fam, prm, build(fam, prm, label, m), m, P, mode); n_eval += 1
                        if cand and cand["cost"] < rec["cost"] * (1 - 1e-9): rec = cand; improved = True
                for k in ints:
                    for dv in (1, -1, 2, -2):
                        prm = dict(rec["params"]); prm[k] = int(prm[k]) + dv
                        if prm[k] < INT_MIN.get(k, 1): continue
                        cand = eval_path(label, fam, prm, build(fam, prm, label, m), m, P, mode); n_eval += 1
                        if cand and cand["cost"] < rec["cost"] * (1 - 1e-9): rec = cand; improved = True
                if not improved: break
    # trim: the smallest last-block depth that still reaches m (bisection on a scale of the last depth; accepted when cheaper)
    dep = np.array(rec["depths"], float)
    if len(dep) and dep[-1] > 0:
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2; d2 = dep.copy(); d2[-1] = dep[-1] * mid
            r = runmax(label, d2); n_eval += 1
            if r[-1] >= m * (1.0 + 1e-7): hi = mid           # a 1e-7 margin: the stored depths are rounded to 1e-9
            else: lo = mid
        if hi < 1.0:
            d2 = dep.copy(); d2[-1] = dep[-1] * hi
            c2 = float(np.sum(P.cost_per_boundary(d2[d2 > 0], mode=mode))) if (d2 > 0).any() else 0.0
            run = runmax(label, d2)
            if run[-1] >= m * (1.0 + 1e-7) and c2 < rec["cost"]:
                rec = dict(rec, cost=c2, drop=float(run[-1]), cost_per_pct=c2 / (100.0 * run[-1]), depth_max=float(d2.max()),
                           blocks_off=int((d2 > 0).sum()), depths=d2, trimmed=True)
    rec["refine_evals"] = n_eval
    return rec


def min_cost_from_paths(paths, m, pool_name, mode, label=None, w=None, do_refine=True):
    P = pool(pool_name)
    best = _best_of(paths, m, P, mode)
    lb = None
    if label in LINEAR and w is not None:
        L = lagrangian(label, w, m, pool_name, mode); lb = L["lower_bound"]
        if L["best"] is not None:
            b = L["best"]; dep = b["depths"]; run = runmax(label, dep)
            if run[-1] >= m:
                best["lagrangian"] = dict(family="lagrangian", params=b["params"], blocks=len(dep), cost=b["cost"], drop=float(run[-1]),
                                          cost_per_pct=b["cost"] / (100.0 * run[-1]), depth_max=float(dep.max()), blocks_off=int((dep > 0).sum()), depths=dep)
    if not best:
        return dict(feasible=False, m=m, pool=pool_name, mode=mode, lower_bound=lb)
    order = sorted(best.values(), key=lambda r: r["cost"])
    top = order[0]; pre = top["cost"]; second_refined = None
    if do_refine and label is not None:
        top = refine(label, top, m, P, mode)
        # the runner-up family is refined too when its grid cost is within REFINE_SECOND of the winner's: a greedy coordinate
        # search on one family can stop at a local optimum a hair above another family's refined path
        if len(order) > 1 and order[1]["cost"] <= order[0]["cost"] * (1.0 + REFINE_SECOND):
            second_refined = refine(label, order[1], m, P, mode)
            if second_refined["cost"] < top["cost"]: top, second_refined = second_refined, top
    out = dict(feasible=True, m=m, pool=pool_name, mode=mode, best=_jsonable(top, full=True), cost_before_refine=pre,
               second=_jsonable(order[1]) if len(order) > 1 else None, second_refined=_jsonable(second_refined) if second_refined else None,
               by_family={k: _jsonable(v) for k, v in best.items()})
    if lb is not None: out.update(lower_bound=lb, cost_over_lb=top["cost"] / lb, bound_K=L["K_bound"], bound_K_max=L["K_max"], bound_at_K_cap=L["bound_at_K_cap"])
    return out


def _jsonable(rec, full=False):
    r = {k: v for k, v in rec.items() if k != "depths"}
    if full: r["depths_pct"] = [round(100.0 * float(x), 7) for x in rec["depths"]]
    return r


def min_cost(label, m, pool_name="TricryptoUSDC", mode="rule", paths=None):
    """the cheapest attack path that moves the design's oracle down by at least m (fraction): the task's search (see the
    module docstring). Returns dict(best=..., second=..., by_family=..., lower_bound for linear designs); best carries family,
    params, blocks, cost, drop (reached), cost_per_pct (cost per 1 % of drop), depth_max, blocks_off, depths_pct."""
    paths = _paths(label) if paths is None else paths
    w = impulse_weights(label) if label in LINEAR else None
    return min_cost_from_paths(paths, m, pool_name, mode, label=label, w=w)


def rebuild(rec):
    """the depth vector of a stored best record"""
    return np.array(rec["depths_pct"], float) / 100.0


def _worker(label):
    t0 = time.time()
    paths = _paths(label); t_paths = time.time() - t0
    w = impulse_weights(label) if label in LINEAR else None
    res = {}
    for pn in POOLS:
        for mode in MODES:
            for m in MS:
                res[f"{pn}|{mode}|{m:g}"] = min_cost_from_paths(paths, m, pn, mode, label=label, w=w)
    # every chosen path re-run with the relaxation tail: its max drop must equal the prefix read-off and reach m
    chk = 0.0; n_ok = 0
    for key, r in res.items():
        if not r["feasible"]: continue
        b = r["best"]; full = oracle_path(label, rebuild(b), tail=TAIL)["max_drop"]
        chk = max(chk, abs(full - b["drop"])); b["drop_full_path"] = full; b["reaches_m"] = bool(full >= r["m"]); n_ok += b["reaches_m"]
    fams = {}
    for r in res.values():
        if r["feasible"]: fams[r["best"]["family"]] = fams.get(r["best"]["family"], 0) + 1
    return label, res, dict(secs=round(time.time() - t0, 1), secs_paths=round(t_paths, 1), chk=chk, n_paths=len(paths),
                            n_cells=len(res), n_feasible=sum(r["feasible"] for r in res.values()), n_reach=n_ok, winners=fams,
                            refine_gain=float(np.mean([1 - r["best"]["cost"] / r["cost_before_refine"] for r in res.values() if r["feasible"] and r["cost_before_refine"] > 0])))


def run_optimizer(workers=4):
    from multiprocessing import get_context
    for pn in POOLS:
        pool(pn)                                           # grids built once, inherited by the forked workers
        for mode in MODES: lag_tables(pn, mode)
    T0 = time.time(); out = {"labels": {}, "stats": {}}
    with get_context("fork").Pool(workers) as pl:
        for label, res, st in pl.imap_unordered(_worker, R3.LABELS):
            out["labels"][label] = res; out["stats"][label] = st
            print(f"  {label:34s} {st['n_paths']:5d} paths ({st['secs_paths']:5.1f} s) {st['secs']:6.1f} s  cells {st['n_feasible']}/{st['n_cells']} feasible, {st['n_reach']} reach m; "
                  f"prefix vs full |d drop| {st['chk']:.1e}; refine gain {100 * st['refine_gain']:.1f} %; winners {st['winners']}", flush=True)
    out["wall_s"] = round(time.time() - T0, 1)
    return out


# ------------------------------------------------------------------------------------------------------ verification
def verify():
    """the task's checks; returns the dict (printed too)"""
    V = {}
    alpha = math.exp(-DT / R3.RAW)
    # 1) today's hold closed form m = d (1 - alpha^K)
    rows = []
    for d, K in ((0.01, 1), (0.05, 10), (0.10, 50), (0.30, 25), (0.60, 300)):
        r = oracle_path("today", hold(d, K)); cf = d * (1.0 - alpha ** K)
        rows.append(dict(d=d, K=K, closed_form=cf, oracle_path=r["max_drop"], abs_err=abs(cf - r["max_drop"]), argmax_block=r["argmax_block"], expected_argmax=LEAD + K - 1))
    V["today_closed_form"] = dict(rows=rows, pass_1e6=all(x["abs_err"] <= 1e-6 for x in rows), alpha=alpha)
    print("1) today hold closed form d (1 - alpha^K), alpha = exp(-12/865):")
    for x in rows: print(f"   d {100 * x['d']:5.1f} % K {x['K']:3d}: closed {100 * x['closed_form']:.6f} % path {100 * x['oracle_path']:.6f} % |err| {x['abs_err']:.1e}  (deepest at block {x['argmax_block']}, expected {x['expected_argmax']})")
    # 2) one brutal boundary moves today by 1 - alpha = 1.378 % of d_eff
    br = []
    for pn, X in (("TricryptoUSDC", 300e3), ("TricryptoUSDC", 1e6), ("TriCRV", 100e3)):
        dep, d = brutal(pn, X, 1); r = oracle_path("today", dep)
        br.append(dict(pool=pn, X_usd=X, d_eff=d, oracle_drop=r["max_drop"], ratio=r["max_drop"] / d, cost_rule=cost(dep, pn, "rule"), cost_open=cost(dep, pn, "open")))
    # tolerance 1e-8: the sample 1e-6 s before the next block shortens the step by 1e-6 s (1 - alpha moves by 1.1e-9)
    V["brutal_one_boundary"] = dict(rows=br, one_minus_alpha=1.0 - alpha, pass_=all(abs(x["ratio"] - (1 - alpha)) < 1e-8 for x in br))
    print(f"2) one brutal boundary, today: 1 - alpha = {100 * (1 - alpha):.4f} % of d_eff")
    for x in br: print(f"   {x['pool']} sell ${x['X_usd']:,.0f}: d_eff {100 * x['d_eff']:.2f} % -> oracle drop {100 * x['oracle_drop']:.4f} % (ratio {100 * x['ratio']:.4f} %), cost/boundary rule ${x['cost_rule']:,.0f} open ${x['cost_open']:,.0f}")
    # 3) the crash regime under 'tiny new lows' vs a plain hold (flags = r2c/episodes.crash_flags, the vectorised replica)
    sys.path.insert(0, os.path.join(ROOT, "r2c")); import episodes as E  # noqa: E402
    reg = {}
    print("3) crash regime after a staircase (d0 2 %, step 1 %, n = N blocks): blocks ON during the hold phase (K2 = 150)")
    for label in [l for l in R3.LABELS if R3.is_regime(l)]:
        kw = (R3.SPEC.get(label) or R3.spec(label))["kwargs"]; N, M, D, eps = kw["N"], kw["M"], kw["D"], kw["eps"]
        n = N; d0, st = 0.02, 0.01
        rec = {}
        for name, nl in (("newlow_0.05pct", NEWLOW), ("newlow_1.05D_over_N", 1.0 - math.exp(-1.05 * D / N)), ("plain_hold", 0.0)):
            dep = staircase(d0, st, n, 150, nl) if nl > 0 else np.concatenate([d0 + st * np.arange(n), np.full(150, d0 + st * (n - 1))])
            t, p, at = grid(dep, tail=0)
            on = E.crash_flags(t, p, N, M, D, eps)[LEAD:]
            stairs_on = int(on[:n].sum()); hold_on = int(on[n:].sum())
            r = oracle_path(label, dep, tail=0)
            o = r["oracle"][LEAD:]; sp = p[LEAD:]
            rec[name] = dict(newlow=nl, stairs_blocks_on=stairs_on, hold_blocks_on=hold_on, hold_blocks=150, on_at_end=bool(on[-1]),
                             first_off_in_hold=int(np.argmin(on[n:])) if not on[n:].all() else None,
                             gap_end_of_stairs_pct=100 * math.log(o[n - 1] / sp[n - 1]), gap_after_30_hold_pct=100 * math.log(o[n + 29] / sp[n + 29]),
                             gap_end_pct=100 * math.log(o[-1] / sp[-1]))
        rec["D_over_N_pct"] = 100 * D / N
        rec["tiny_newlow_keeps_on"] = rec["newlow_0.05pct"]["on_at_end"]
        rec["plain_hold_lapses_within_N"] = (rec["plain_hold"]["first_off_in_hold"] is not None and rec["plain_hold"]["first_off_in_hold"] < N)
        rec["DN_newlow_keeps_on"] = rec["newlow_1.05D_over_N"]["on_at_end"]
        reg[label] = rec
        print(f"   {label:34s} D/N {rec['D_over_N_pct']:.2f} %/blk | 0.05 % lows: ON {rec['newlow_0.05pct']['hold_blocks_on']:3d}/150 (off at {rec['newlow_0.05pct']['first_off_in_hold']}), gap stairs-end {rec['newlow_0.05pct']['gap_end_of_stairs_pct']:5.2f} % -> +30 {rec['newlow_0.05pct']['gap_after_30_hold_pct']:5.2f} % | "
              f"1.05 D/N lows: ON {rec['newlow_1.05D_over_N']['hold_blocks_on']:3d}/150 gap +30 {rec['newlow_1.05D_over_N']['gap_after_30_hold_pct']:5.2f} % | plain hold: off at {rec['plain_hold']['first_off_in_hold']} (N {N})")
    V["regime_newlows"] = reg
    V["regime_summary"] = dict(tiny_0_05pct_keeps_on_for=[l for l, r in reg.items() if r["tiny_newlow_keeps_on"]],
                               DN_keeps_on_for=[l for l, r in reg.items() if r["DN_newlow_keeps_on"]],
                               plain_hold_lapses_within_N_for_all=all(r["plain_hold_lapses_within_N"] for r in reg.values()))
    print(f"   -> 0.05 % new lows keep the regime on for: {V['regime_summary']['tiny_0_05pct_keeps_on_for'] or 'NONE'}; 1.05 D/N for: {len(V['regime_summary']['DN_keeps_on_for'])}/{len(reg)}; plain hold lapses within N for all: {V['regime_summary']['plain_hold_lapses_within_N_for_all']}")
    # 4) the rate limit per block: vfast in the CRASH regime (every design with vfast), vslow in the calm regime (the designs
    #    with vslow); the per-block move between two samples is attributed to the regime flag of the block between them
    rl = {}
    for label in R3.LABELS:
        kw = (R3.SPEC.get(label) or R3.spec(label))["kwargs"]
        if (R3.SPEC.get(label) or R3.spec(label))["kind"] != "regime" or kw.get("vfast") is None: continue
        N, M, D, eps = kw["N"], kw["M"], kw["D"], kw["eps"]
        vmax_crash = vmax_calm = vmax_all = 0.0; n_crash = 0
        for dep in (hold(0.5, 100), staircase(0.05, 0.02, 40, 100, 1.0 - math.exp(-1.05 * D / N)), staircase(0.02, 0.01, 60, 200, NEWLOW),
                    alternating(0.9, 50), hold(0.95, 300), ramp(0.02, 0.003), gap_track(label, 0.0125, 1.0 - math.exp(-1.05 * D / N), 0.02)):
            r = oracle_path(label, dep); o = r["oracle"]
            on = E.crash_flags(r["t"], r["p"], N, M, D, eps)
            dl = np.abs(np.diff(np.log(o))); fl = on[1:]
            vmax_all = max(vmax_all, float(dl.max()))
            if fl.any(): vmax_crash = max(vmax_crash, float(dl[fl].max())); n_crash += int(fl.sum())
            if (~fl).any(): vmax_calm = max(vmax_calm, float(dl[~fl].max()))
        # tolerance 1e-6: the sample sits 1e-6 s before the block (a crash -> calm transition exceeds vslow by 3e-7 relative)
        ok = vmax_crash <= kw["vfast"] * (1 + 1e-6) and (kw.get("vslow") is None or vmax_calm <= kw["vslow"] * (1 + 1e-6))
        rl[label] = dict(vfast=kw["vfast"], vslow=kw.get("vslow"), crash_blocks=n_crash, max_abs_dln_crash=vmax_crash, max_abs_dln_calm=vmax_calm,
                         max_abs_dln_all=vmax_all, pass_=ok)
    V["rate_limit"] = rl
    print("4) rate limit per block: crash regime <= vfast, calm regime <= vslow (where set); paths: hold 50 %/95 %, staircases, alternating 90 %, ramp, gap_track")
    for l, r in rl.items():
        print(f"   {l:34s} vfast {100 * r['vfast']:.2f} % vslow {('%.2f %%' % (100 * r['vslow'])) if r['vslow'] else 'none  '}: crash blocks {r['crash_blocks']:4d} max {100 * r['max_abs_dln_crash']:.4f} % | calm max {100 * r['max_abs_dln_calm']:7.4f} %  {'ok' if r['pass_'] else 'EXCEEDED'}")
    # 5) linearity of the LINEAR designs (the Lagrangian's premise): impulse weights vs superposition on random paths; today's
    #    weights vs the closed form (1 - alpha) alpha^j
    lin = {}
    for label in LINEAR:
        w = impulse_weights(label); err = linearity_check(label, w)
        kw = (R3.SPEC.get(label) or R3.spec(label))["kwargs"]; raw = R3.RAW if label == "today" else kw["raw"]; a = math.exp(-DT / raw)
        cf = (1 - a) * a ** np.arange(KMAX)
        cf = (1 - a) * a ** np.arange(len(w))
        lin[label] = dict(superposition_max_abs=err, w0=float(w[0]), w_sum=float(w.sum()), n_weights=len(w), closed_form_max_abs=float(np.max(np.abs(w - cf))), alpha=a, pass_1e8=err < 1e-8)
        print(f"5) {label:8s} linear: superposition max |err| {err:.1e}, w0 {w[0]:.6f} (1 - alpha {1 - a:.6f}), weights vs closed form max |err| {lin[label]['closed_form_max_abs']:.1e}")
    V["linearity"] = lin
    # 6) MS = the engine's exposure drops
    ems = engine_ms()
    V["ms_vs_engine"] = dict(attack=MS, engine=ems, equal=tuple(MS) == tuple(ems))
    print(f"6) MS {MS} == engine ms {ems}: {V['ms_vs_engine']['equal']}")
    # 7) the trigger's semantics on every regime design (flags = r2c/episodes.crash_flags; the move = the oracle itself):
    #    (a) prefix(M, n_up) + one step to d_big = (1 - exp(-D)) (1 + margin) turns the crash regime ON at the big block for
    #        n_up = 0 and n_up = N - M, with tiny = TINY and tiny = TINY_MIN; (b) with one tiny step fewer (M-2 downs) it stays OFF;
    #    (c) the oracle's move at the sample after the big block is the CRASH step: e = p (1 - a) + e_prev a with a = exp(-12 ln2 / hf),
    #        rate-limited to vfast (vs the calm step with alpha = exp(-12/raw): 1.38 % of the gap for raw 865 s); (d) after the big
    #        step n_flat = N - M - n_up flats keep the regime ON and the next flat turns it OFF.
    tr = {}
    print("7) trigger semantics: regime ON at the big block (M-1 tiny downs, n_up ups), OFF with one down fewer, the fast move at the sample after it, flats")
    for label in [l for l in R3.LABELS if R3.is_regime(l)]:
        kw = (R3.SPEC.get(label) or R3.spec(label))["kwargs"]; N, M, D, eps, hf = kw["N"], kw["M"], kw["D"], kw["eps"], kw["hf"]
        raw = kw["raw"]; vf = kw.get("vfast"); a_c = math.exp(-DT * math.log(2) / hf); a_calm = math.exp(-DT / raw)
        dD = (1.0 - math.exp(-D)) * (1.0 + DBIG_MARGIN); rows = []
        for n_up in sorted(set([0, N - M])):
            for tiny in (TINY, TINY_MIN):
                n_flat = N - M - n_up
                dep = trigger(label, dD, n_up, n_flat + 1, 0.0, tiny)             # head + the flats + ONE flat too many
                t, p, at = grid(dep, tail=0); on = E.crash_flags(t, p, N, M, D, eps)[LEAD:]
                kb = M - 1 + n_up                                                  # index of the big block in the attack
                r = oracle_path(label, dep, tail=0); o = r["oracle"][LEAD:]
                e_prev = o[kb - 1]; gap_pred = p[LEAD + kb] * (1 - a_c) + e_prev * a_c
                if vf is not None:
                    lim = vf; rr = math.log(gap_pred / e_prev); gap_pred = e_prev * math.exp(max(-lim, min(lim, rr)))
                calm_pred = p[LEAD + kb] * (1 - a_calm) + e_prev * a_calm
                dep2 = np.delete(dep, 0)                                           # one tiny step fewer: M-2 downs before the big step
                on2 = E.crash_flags(*grid(dep2, tail=0)[:2], N, M, D, eps)[LEAD:]
                rows.append(dict(n_up=n_up, tiny=tiny, n_flat=n_flat, big_block=kb, on_at_big=bool(on[kb]), off_before_big=bool(not on[:kb].any()),
                                 on_through_flats=bool(on[kb:kb + n_flat + 1].all()), off_after_flats=bool(not on[kb + n_flat + 1]),
                                 off_with_one_down_fewer=bool(not on2[:kb + n_flat + 1].any()),
                                 move_rel_err=abs(o[kb] / gap_pred - 1), calm_move_rel_err=abs(o[kb] / calm_pred - 1),
                                 drop_after_big_pct=100 * (1 - o[kb]), down_steps=int(sum(1 for k in range(0, kb + 1) if p[LEAD + k] < p[LEAD + k - 1] * (1 - eps)))))
        ok = all(x["on_at_big"] and x["off_before_big"] and x["on_through_flats"] and x["off_after_flats"] and x["off_with_one_down_fewer"]
                 and x["move_rel_err"] < 1e-6 and x["down_steps"] == M for x in rows)
        tr[label] = dict(rows=rows, pass_=ok, d_big=dD)
        x = rows[-1]
        print(f"   {label:34s} N {N} M {M} d_big {100 * dD:.3f} %: n_up {x['n_up']} tiny {100 * x['tiny']:.4f} %: ON at big block {x['big_block']} {x['on_at_big']}, "
              f"through {x['n_flat']} flats {x['on_through_flats']}, off after {x['off_after_flats']}, one down fewer stays off {x['off_with_one_down_fewer']}; "
              f"move = crash step (rel err {x['move_rel_err']:.1e}; vs calm step {x['calm_move_rel_err']:.2e}), drop after big {x['drop_after_big_pct']:.3f} %  {'ok' if tr[label]['pass_'] else 'FAIL'}")
    V["trigger_semantics"] = tr
    V["trigger_pass"] = all(r["pass_"] for r in tr.values())
    V["pass_all"] = bool(V["today_closed_form"]["pass_1e6"] and V["brutal_one_boundary"]["pass_"] and all(r["pass_"] for r in rl.values())
                         and all(x["pass_1e8"] for x in lin.values()) and V["ms_vs_engine"]["equal"] and V["trigger_pass"])
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    T0 = time.time()
    V = verify()
    V["secs"] = round(time.time() - T0, 1)
    json.dump(V, open(os.path.join(HERE, "attack_verify.json"), "w"), indent=1, default=float)
    print(f"verify {'PASS' if V['pass_all'] else 'FAIL'} in {V['secs']} s -> r3/attack_verify.json")
    if a.verify_only: return
    print(f"\noptimizer: {len(R3.LABELS)} designs x {len(MS)} drops x {len(POOLS)} pools x {len(MODES)} cost modes, {a.workers} workers")
    opt = run_optimizer(a.workers)
    opt.update(dict(ms=MS, pools=POOLS, modes=MODES, lead=LEAD, dt=DT, dmin=DMIN, dmax=DMAX, nd=ND, kmax=KMAX, cmax=CMAX,
                    staircase=dict(d0=D0S, step=STEPS, n=NS, k2max=K2MAX, newlow=NEWLOW), ramp=dict(d0=R_D0, s=R_S),
                    two_speed=dict(d0=T_D0, s1=T_S1, n1=T_N1, s2=T_S2), gap_track=dict(gap=G_GAP, d0=G_D0), linear=LINEAR,
                    trigger=dict(tiny=TINY, tiny_min=TINY_MIN, dbig_factors=TRIG_DBIG, dbig_abs=TRIG_DBIG_ABS, dbig_cliff_factors=TRIG_CLIFF_F, dbig_margin=DBIG_MARGIN, gap=TRIG_GAP, gap_dbig=TRIG_GAP_DBIG,
                                 n_up="0..N-M", n_flat="0, N-M-n_up, K2MAX", nl="0, 1.05/1.5/2.5/4 D/N, 1.05 D/(M-1), vfast, TRIG_NL_STEEP", nl_steep=TRIG_NL_STEEP),
                    lagrangian=dict(nd=LAG_ND, nmu=LAG_NMU, k_max=LAGK), refine=dict(deltas=REF_DELTAS, deltas_gap=REF_DELTAS_SLOW, second_within=REFINE_SECOND),
                    sampling="block end + 12 s - 1e-6", drop_definition="fraction 1 - min(oracle)", wall_s=opt["wall_s"], secs_total=round(time.time() - T0, 1)))
    json.dump(opt, open(os.path.join(HERE, "attack_mincost.json"), "w"), indent=1, default=float)
    print(f"optimizer wall {opt['wall_s']} s (max per design {max(s['secs'] for s in opt['stats'].values())} s) -> r3/attack_mincost.json")
    for pn in POOLS:
        for mode in ("rule", "open"):
            print(f"\n{pn}, cost mode {mode}: cheapest path per design (m = 1 % | 5 % | 20 %): family blocks depth_max cost [x LB for linear designs]")
            for l in R3.LABELS:
                cells = []
                for m in (0.01, 0.05, 0.20):
                    r = opt["labels"][l][f"{pn}|{mode}|{m:g}"]
                    if not r["feasible"]: cells.append(f"{'not reachable':>52s}"); continue
                    b = r["best"]
                    lbs = f" x{r['cost_over_lb']:.3f}" if r.get("lower_bound") else ""
                    cells.append(f"{b['family']:>12s} {b['blocks']:3d} blk dmax {100 * b['depth_max']:5.1f}% ${b['cost']:>11,.0f}{lbs:8s}")
                print(f"  {l:34s} " + " | ".join(cells))
    print(f"\ntotal {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
