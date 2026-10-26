"""Oracle drift scenario (the page's: $20M sold into TricryptoUSDC at the block end, arbitraged back next block, 50 times):
today's oracle (the pool's 600 s EMA; it does not depend on LLAMMA swaps) vs the concept's real WindowOracle.vy written only at
the blocks where mint WETH really had an AMM swap on its busiest day (12.6 % of blocks), the scenario laid at 280 offsets of that
day. -> drift_busy_day_vs_current.png"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; OC = HERE.parent / "oracle_charts"; LD = HERE.parents[2]
sys.path.insert(0, str(LD)); sys.path.insert(0, str(OC))
from r3 import attack as AT, poolmath as PM  # noqa: E402
import driver as DR  # noqa: E402
BG, INK, GRID, MUTED = "#0d1117", "#e6edf3", "#30363d", "#8b949e"; T0 = 1_790_000_003
d20 = PM.Pool("TricryptoUSDC").sell(20e6)["depth_eff"]; dep = AT.alternating(d20, 50)
cur = AT.oracle_path("today", dep)
t, p, at = AT.grid(dep); bt = (T0 + t).astype(np.int64); st = bt + 12
d = 1 - p; trade = (d > 0) | np.r_[False, d[:-1] > 0]; n = len(p); L = AT.LEAD; idx = np.arange(n); b = idx - L; keep = (b >= -2) & (b <= 130)
S = json.load(open(HERE / "swaps.json"))
mk = max((m for m in S["markets"] if m["kind"] == "mint" and m["name"] == "WETH"), key=lambda m: len(m["swap_blocks"]))
bl = np.array(mk["swap_blocks"]); i0 = max(range(len(bl)), key=lambda i: np.searchsorted(bl, bl[i] + 7200) - i); start = int(bl[i0])
paths, shares = [], []
for off in range(0, 7200 - n, 25):
    rel = bl[(bl >= start + off) & (bl < start + off + n)] - (start + off)
    wm = np.zeros(n, bool); wm[rel] = True; shares.append(wm.mean())
    r = DR.run(bt, p, trade, st, "b", theta=None, write_mask=wm, deploy_t=int(bt[0]) - 12, deploy_p=1.0, p_scale=4000)
    paths.append(np.array(r["price"]))
P = 100 * np.array(paths); lows = P[:, keep].min(axis=1); worst = P[int(np.argmin(lows))]
q10, q50, q90 = (np.percentile(P, q, axis=0) for q in (10, 50, 90))
fig, ax = plt.subplots(figsize=(16, 7), dpi=220, facecolor=BG); ax.set_facecolor(BG); ax.grid(color=GRID, linewidth=0.6); ax.tick_params(colors=INK, labelsize=12)
for s_ in ax.spines.values(): s_.set_color(GRID)
ax.step(b[keep], 100 * cur["oracle"][keep], where="post", color="#c9d1d9", linewidth=3, label=f"current oracle (600 s EMA)   lowest {100 * cur['oracle'][keep].min():.1f} %")
ax.fill_between(b[keep], q10[keep], q90[keep], step="post", color="#3fb950", alpha=0.20, linewidth=0, label="oracle concept (real code): middle 80 % of runs at each block (10th to 90th percentile)")
ax.step(b[keep], q50[keep], where="post", color="#3fb950", linewidth=3, label=f"oracle concept (real code): median across runs at each block   lowest point {q50[keep].min():.1f} %")
ax.step(b[keep], worst[keep], where="post", color="#3fb950", linewidth=1.4, linestyle="--", label=f"oracle concept (real code): worst run   lowest {lows.min():.1f} %")
ax.set_ylim(min(100 * cur["oracle"][keep].min(), q10[keep].min(), lows.min()) - 1.5, 102); ax.set_xlim(-2, 130)
ax.set_ylabel("oracle, % of the market price", color=INK, fontsize=13); ax.set_xlabel("blocks (12 s each)", color=INK, fontsize=13)
ax.set_title("Oracle drift ($20M sold into TricryptoUSDC at the block end, arbitraged back next block, 50 times) "
             f"with a busy day's swap timing: oracle written only where mint WETH had a swap ({100 * np.mean(shares):.1f} % of blocks), {len(paths)} runs",
             color=INK, fontsize=12, loc="left", pad=10, wrap=True)
ax.legend(loc="lower left", fontsize=11, facecolor="#161b22", edgecolor=GRID, labelcolor=INK); fig.tight_layout()
fig.savefig(HERE / "drift_busy_day_vs_current.png", facecolor=BG); plt.close(fig)
json.dump(dict(day_start_block=start, runs=len(paths), write_share=float(np.mean(shares)), current_low=float(100 * cur["oracle"][keep].min()),
               concept_low_median=float(np.median(lows)), concept_low_p10=float(np.percentile(lows, 10)), concept_low_min=float(lows.min()),
               concept_median_path_low=float(q50[keep].min())), open(HERE / "drift_busy_day_vs_current.json", "w"), indent=1)
print("saved", json.load(open(HERE / "drift_busy_day_vs_current.json")))
