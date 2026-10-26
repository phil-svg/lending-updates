"""The page's "Two weeks of September 2026" chart, same code and style as concept_site/build.py, with the concept line taken from
the real FIXED WindowOracle.vy written only at mint WETH's real swap blocks (sep2026_real.npz). -> sep2026_real.png"""
import datetime, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
BG, INK, GRID = "#0d1117", "#e6edf3", "#30363d"
cc = np.fromfile(f"{ROOT}/candles_0.bin", dtype=np.float64).reshape(-1, 4)
tt0 = datetime.datetime(2026, 9, 1, tzinfo=datetime.UTC).timestamp(); tt1 = datetime.datetime(2026, 9, 15, tzinfo=datetime.UTC).timestamp()
kk = (cc[:, 0] >= tt0) & (cc[:, 0] < tt1); tm = [datetime.datetime.fromtimestamp(x, datetime.UTC) for x in cc[kk, 0]]; spot = cc[kk, 3]
cur_w = np.fromfile(f"{ROOT}/results/series/today_pool/oracle_0.f64", dtype=np.float64)[kk]
z = np.load(HERE / "sep2026_real.npz"); con_w = z["price"]; assert len(con_w) == len(spot), (len(con_w), len(spot))
model = np.fromfile(f"{ROOT}/results/series/cb2_t300_w36_s0.3_cend/oracle_0.f64", dtype=np.float64)[kk]
fig, p1 = plt.subplots(figsize=(16, 7), dpi=200, facecolor=BG); p1.set_facecolor(BG); p1.grid(color=GRID, linewidth=0.6); p1.tick_params(colors=INK, labelsize=11)
for sp_ in p1.spines.values(): sp_.set_color(GRID)
p1.plot(tm, spot, color="#58a6ff", linewidth=1.0, label="ETH spot (Binance, 1-minute close)")
p1.plot(tm, con_w, color="#3fb950", linewidth=1.2, label=f"concept (real WindowOracle code on TricryptoUSDC, written only at the mint WETH market's {int(z['writes'])} real swaps)")
p1.plot(tm, cur_w, color="#c9d1d9", linewidth=1.0, linestyle=(0, (4, 3)), label="current oracle (TricryptoUSDC, 600 s EMA)")
p1.set_ylabel("USD per ETH", color=INK, fontsize=12); p1.set_title("ETH, 1 - 14 Sep 2026: spot vs what each oracle reports", color=INK, fontsize=14, loc="left", pad=10)
p1.legend(loc="upper left", fontsize=11, facecolor="#161b22", edgecolor=GRID, labelcolor=INK)
p1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); fig.savefig(HERE / "sep2026_real.png", facecolor=BG, bbox_inches="tight"); plt.close(fig)
d = lambda a, b: (float(100 * np.max(np.abs(a / b - 1))), float(100 * np.mean(np.abs(a / b - 1))))
out = dict(writes=int(z["writes"]), concept_vs_current_pct=d(con_w, cur_w), concept_vs_spot_pct=d(con_w, spot), current_vs_spot_pct=d(cur_w, spot), concept_vs_old_model_line_pct=d(con_w, model))
json.dump(out, open(HERE / "sep_chart.json", "w"), indent=1); print(json.dumps(out))
