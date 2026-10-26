"""sep2026_run.py - one write regime of the REAL WindowOracle.vy on the real TricryptoUSDC ETH block-end prices
(data/pool_USDC.npz: last_prices after every block with a pool event, lp1 = ETH), 1 - 14 Sep 2026, read with price()
at the end of every candle minute (the model's sampling: candle t + 60 s, as r2a/build_series.at_times).

Block grid: every 12 s slot from the warm-up start (31 Aug 18:00 UTC) to 15 Sep 00:00 UTC (missed slots ignored: a write
in every slot); the pool trades (mock pool.trade(lp1)) in the blocks with a record in pool_USDC.npz.
Deployment at the warm-up start: mock pool with the real pool's EMA at that moment (oracles.today on the full series =
the replicated price_oracle) and its last block-end price; the oracle deployed in the same block.
  a       price_w() every slot
  b       price_w() in the pool's trade blocks + a LLAMMA-arb write when price() moved > theta since the last write
  bpool   price_w() in the pool's trade blocks only
  c       no write after the deployment
usage: nice -n 19 python3 sep2026_run.py <a|b|bpool|c> [hours]   -> sep_<regime>.npz (hours: a short test from the start)
"""
import datetime, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LD = HERE.parents[2]
sys.path.insert(0, str(LD)); sys.path.insert(0, str(HERE))
import oracles as O  # noqa: E402
import driver as DR  # noqa: E402

UTC = datetime.UTC
T_S = int(datetime.datetime(2026, 9, 1, tzinfo=UTC).timestamp()); T_E = int(datetime.datetime(2026, 9, 15, tzinfo=UTC).timestamp())
T_W = T_S - 6 * 3600


def inputs(hours=None):
    z = np.load(LD / "data/pool_USDC.npz", allow_pickle=True)
    ts = z["ts"].astype(np.int64); lp = z["lp1"].astype(float)
    import json
    meta = json.loads(str(z["meta"])); init = meta["init_state_before_span"]
    t_end = T_E if hours is None else T_W + int(hours * 3600)
    k0 = np.searchsorted(ts, T_W); k1 = np.searchsorted(ts, t_end)
    # the real pool EMA (price_oracle view) at T_W: the replicated EMA (O.today, raw 865 over the whole span)
    tt = np.concatenate([[float(init["ts"])], ts[:k0].astype(float)]); pp = np.concatenate([[float(init["lp1"])], lp[:k0]])
    e_blk = O.ema_at_blocks(tt, pp, raw=865.0, p0=float(init["po1"]))
    ema_w = float(O.today(np.array([float(T_W)]), tt, pp, raw=865.0, e_blk=e_blk)[0])
    lp_w = float(lp[k0 - 1])
    off = int(ts[k0] % 12)
    grid = np.arange(T_W + ((off - T_W) % 12), t_end, 12, dtype=np.int64)
    tr_t = ts[k0:k1]
    assert np.all((tr_t - off) % 12 == 0), "trade blocks off the slot grid"
    pos = np.searchsorted(grid, tr_t); assert np.all(grid[pos] == tr_t)
    trade = np.zeros(len(grid), bool); trade[pos] = True
    price = np.full(len(grid), np.nan); price[pos] = lp[k0:k1]
    # standing price per slot (only used where trade is True)
    last = lp_w; pr = np.empty(len(grid))
    for i in range(len(grid)):
        if trade[i]: last = price[i]
        pr[i] = last
    cc = np.fromfile(LD / "candles_0.bin", dtype=np.float64).reshape(-1, 4)
    kk = (cc[:, 0] >= T_S) & (cc[:, 0] < T_E)
    st = (cc[kk, 0] + 60).astype(np.int64)
    st = st[st <= t_end]
    return dict(grid=grid, p=pr, trade=trade, st=st, ema_w=ema_w, lp_w=lp_w, kk=kk)


if __name__ == "__main__":
    reg = sys.argv[1]; hours = float(sys.argv[2]) if len(sys.argv) > 2 else None
    I = inputs(hours)
    t0 = time.time()
    regime = {"a": "a", "b": "b", "bpool": "b", "c": "c"}[reg]
    theta = None if reg == "bpool" else 1e-3
    r = DR.run(I["grid"], I["p"], I["trade"], I["st"], regime, p_scale=1, theta=theta, deploy_t=T_W, deploy_p=I["lp_w"],
               ema0=I["ema_w"], lp0=I["lp_w"], progress=20000)
    secs = time.time() - t0
    tag = reg if hours is None else f"{reg}_test{hours:g}h"
    np.savez(HERE / f"sep_{tag}.npz", st=I["st"], price=np.array(r["price"]), ema=np.array(r["ema"]), wrote=np.array(r["wrote"]),
             n_blocks=len(I["grid"]), n_trade=int(I["trade"].sum()), secs=secs)
    print(f"{tag}: {len(I['grid'])} slots, {int(I['trade'].sum())} pool-trade blocks, {r['writes']} writes, {len(r['price'])} reads, {secs:.0f} s", flush=True)
