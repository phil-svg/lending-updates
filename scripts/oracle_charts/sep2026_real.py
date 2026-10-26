"""Two weeks of September 2026 (the page's ETH chart) on the real, FIXED WindowOracle.vy: TricryptoUSDC's real block-end ETH
prices (as oracle_charts/sep2026_run.py), the oracle written only in the blocks where the mint WETH market really had an AMM swap
(on-chain TokenExchange blocks, write_freq/swaps.json), price() read at every candle minute. -> sep2026_real.npz"""
import sys, os, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OC = HERE.parent / "oracle_charts"; WF = HERE.parent / "write_freq"
sys.path.insert(0, str(OC)); sys.path.insert(0, os.environ.get("CRVUSD_SIM_SRC", "crvUSD-sim/src"))
import chain as C  # noqa: E402
import sep2026_run as SR  # noqa: E402
import driver as DR  # noqa: E402
I = SR.inputs(); grid = I["grid"]
mk = max((m for m in json.load(open(WF / "swaps.json"))["markets"] if m["kind"] == "mint" and m["name"] == "WETH"), key=lambda m: len(m["swap_blocks"]))
cand = [b for b in mk["swap_blocks"] if 25_700_000 <= b <= 26_100_000]
ts = np.array(C.block_ts(cand), dtype=np.int64)
inw = (ts >= grid[0]) & (ts <= grid[-1]); ts = ts[inw]
pos = np.searchsorted(grid, ts); ok = grid[np.minimum(pos, len(grid) - 1)] == ts
wm = np.zeros(len(grid), bool); wm[pos[ok]] = True
print(f"{len(cand)} candidate swap blocks, {int(inw.sum())} inside the window, {int(ok.sum())} on the slot grid; write share {100 * wm.mean():.3f} % of slots", flush=True)
t0 = time.time()
r = DR.run(grid, I["p"], I["trade"], I["st"], "b", p_scale=1, theta=None, write_mask=wm, deploy_t=SR.T_W, deploy_p=I["lp_w"],
           ema0=I["ema_w"], lp0=I["lp_w"], progress=20000)
np.savez(HERE / "sep2026_real.npz", st=I["st"], price=np.array(r["price"]), ema=np.array(r["ema"]), writes=r["writes"], write_share=wm.mean(), secs=time.time() - t0)
print(f"done: {r['writes']} writes, {len(r['price'])} reads, {time.time() - t0:.0f} s", flush=True)
