"""driver.py - drive the REAL compiled WindowOracle (v2/concept, Vyper 0.4.3, cancun) with a mock tricrypto-ng pool
whose price_oracle() is a real EMA of block-end prices (v2/fork/oracle/mocks/MockCryptoPool.vy: the EMA folds in the
previous block-end price on the first trade of a block, snekmate exp, ma_time in the low 64 bits of
packed_rebalancing_params, read by the oracle the way it reads a crypto pool). titanoboa, no fork, no RPC.

run(...) feeds one block-end price path block by block and returns the oracle's price() at the requested sample times
under one write regime:
  'a'  price_w() in every block
  'b'  price_w() only in blocks with an AMM (LLAMMA) swap: every block in which the source pool trades, plus every block
       in which a LLAMMA arbitrage would trade = the oracle's view price() has moved by more than `theta` (relative)
       since the last write (LLAMMA's band prices move with p_o^3, so an oracle move opens an arbitrage when users sit
       in soft liquidation); theta=None = pool-trade blocks only
  'c'  no write after the deployment: price() only
"""
import math
from pathlib import Path

import boa
from boa.interpret import set_search_paths

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
CONCEPT = V2 / "concept"
ORACLE = CONCEPT / "curve_stablecoin/price_oracles/v2/WindowOracle.vy"
MOCKS = V2 / "fork/oracle/mocks"
WAD = 10**18
WINDOW, TAU, CAP, STALE = 36, 865, 2 * 10**15, 300     # the concept's parameters (22 Oct 2026: TAU = the pool's ma_time, CAP 0.2 %)
MA_TIME = 865                                         # TricryptoUSDC packed_rebalancing_params & (2**64-1), read on chain 22 Oct 2026 (block 26247824)

set_search_paths([str(CONCEPT)])
CARGS = {"evm_version": "cancun"}
_DEP = {}


def _deployers():
    if not _DEP:
        _DEP["oracle"] = boa.load_partial(str(ORACLE), compiler_args=CARGS)
        _DEP["pool"] = boa.load_partial(str(MOCKS / "MockCryptoPool.vy"), compiler_args=CARGS)
        _DEP["agg"] = boa.load_partial(str(MOCKS / "MockAggregator.vy"), compiler_args=CARGS)
    return _DEP


def set_time(t):
    t = int(t)
    ev = boa.env.evm.patch
    assert t >= ev.timestamp, (t, ev.timestamp)
    if t > ev.timestamp:
        ev.block_number += max(1, (t - ev.timestamp) // 12)
        ev.timestamp = t


def run(block_t, block_p, trade, sample_t, regime, p_scale=4000, theta=1e-3, deploy_t=None, deploy_p=None,
        ema0=None, lp0=None, ma_time=MA_TIME, warm_writes_until=None, progress=None, write_mask=None):
    """block_t: increasing integer block timestamps; block_p: the pool's block-end price after each block (float, in
    units of p_scale); trade: bool per block (the pool trades in that block -> mock pool.trade(p)); sample_t: increasing
    integer timestamps at which price() is read (a read at the same timestamp as a block comes after that block's
    trade and write: same value as before them). deploy_t / deploy_p: deployment of pool + oracle (default: the first
    block's time - 12 and its price); ema0 / lp0: the pool's EMA / last price at deployment (default deploy_p).
    warm_writes_until: in regimes b and c, every block before this time is written (a warm-up as in a); None = none.
    write_mask: regime b only - write in these blocks instead of the pool-trade blocks (the theta trigger stays).
    Returns dict(price=[oracle price() per sample], ema=[the mock pool's price_oracle(1) per sample = the current
    600 s EMA oracle], writes=int, wrote=[block indexes written])."""
    D = _deployers()
    with boa.env.anchor():
        t_dep = int(block_t[0] - 12 if deploy_t is None else deploy_t)
        boa.env.evm.patch.timestamp = t_dep
        boa.env.evm.patch.block_number = 20_000_000
        sc = p_scale * WAD
        pd = block_p[0] if deploy_p is None else deploy_p
        crv = boa.env.generate_address("crvusd"); coll = boa.env.generate_address("weth")
        agg = D["agg"].deploy(crv, WAD)
        pool = D["pool"].deploy([crv, coll], int(round(pd * sc)), ma_time)
        if ema0 is not None:
            pool.set_raw(int(round(ema0 * sc)), int(round((pd if lp0 is None else lp0) * sc)))
        orc = D["oracle"].deploy(pool.address, 1, "0x0000000000000000000000000000000000000000", agg.address,
                                 WINDOW, TAU, CAP, STALE)
        last_w = orc.price()
        out = []; ema = []; wrote = []
        js = 0; ns = len(sample_t); nb = len(block_t)
        for b in range(nb + 1):
            tb = int(block_t[b]) if b < nb else None
            while js < ns and (tb is None or sample_t[js] < tb):         # reads strictly before this block
                set_time(sample_t[js]); out.append(orc.price() / WAD / p_scale); ema.append(pool.price_oracle(1) / WAD / p_scale); js += 1
            if b == nb:
                break
            set_time(tb)
            if trade[b]:
                pool.trade(int(round(block_p[b] * sc)))
            w = regime == "a" or (warm_writes_until is not None and tb < warm_writes_until)
            if not w and regime == "b":
                if (trade if write_mask is None else write_mask)[b]:
                    w = True
                elif theta is not None:
                    v = orc.price()
                    w = abs(v - last_w) > theta * last_w
            if w:
                last_w = orc.price_w(); wrote.append(b)
            while js < ns and sample_t[js] == tb:                           # reads in this block (after its write)
                out.append(orc.price() / WAD / p_scale); ema.append(pool.price_oracle(1) / WAD / p_scale); js += 1
            if progress and b % progress == 0:
                print(f"  block {b}/{nb}", flush=True)
        return dict(price=out, ema=ema, writes=len(wrote), wrote=wrote)

