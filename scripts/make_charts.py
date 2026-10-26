"""make_charts.py: the LLAMMA concept's charts, the same code as the concept page's builders (concept_site/build.py, by_pool.py,
contracts_build.py) without the page. Inputs, under ROOT (env LLAMMA_DESIGN, default "."):
  engine/runs/<run>/w_<market>.txt   weekly engine output per market (replay_sim_x.cpp; HL / SL / DL losses = columns 4 / 2 / 3,
                                     bad debt = written-off column 14 + the open shortfall column 7 of the last week)
  r10/onchain/engine21_oracle_pools.json   which pool family each market's oracle reads (by-pool charts)
  v2/fork/measure_v2.json, v2/fork/build/{base,concept}/sizes.json   fork gas measurement and compiled sizes (gas / size charts)
  r3/ + oracles.py                    the oracle models (model charts: a $20M sale held 10 blocks, the staleness pump)
Output: images/ (env OUT).  usage: python3 make_charts.py"""
import os, re, sys, json, datetime, textwrap, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.environ.get("LLAMMA_DESIGN", "."); OUT = os.environ.get("OUT", "images"); os.makedirs(f"{OUT}/by_pool", exist_ok=True)
sys.path.insert(0, ROOT)
from r3 import attack as AT, poolmath as PM, common as R3
BG, INK, GRID, MUTED = "#0d1117", "#e6edf3", "#30363d", "#8b949e"; T0 = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC).timestamp(); T1 = 1783167899
E = f"{ROOT}/engine/runs"
RUNS = {"current": f"{E}/rf6_today", "Concept; DL off": f"{E}/rf7_nodl", "Concept; DL on": f"{E}/rf7_dlon"}      # the page's runs
AAVE = {"Aave v3\nequivalent": f"{E}/aave1_r7", "Aave v4\nequivalent": f"{E}/aave4_r7"}
DROPPED = (5, 8, 14); MARKETS = [i for i in range(21) if i not in DROPPED]   # pufETH, ezETH, weETH: v2 cannot price a pool's stored rates
def totals(tag, t0=T0, t1=T1):
    sl = dl = hl = bd = 0.0
    for i in MARKETS:
        w = np.loadtxt(f"{tag}/w_{i}.txt", ndmin=2); k = (w[:, 1] >= t0) & (w[:, 1] <= t1)
        if k.any(): sl += w[k, 2].sum(); dl += w[k, 3].sum(); hl += w[k, 4].sum(); bd += w[k, 14].sum() + w[k][-1, 7]
    return dict(sl=sl / 1e6, dl=dl / 1e6, hl=hl / 1e6, bd=bd / 1e6)
T = {k: totals(v) for k, v in {**RUNS, **AAVE}.items()}

# ---- losses and bad debt (build.py) ----
def loss_and_bd_charts(names, suffix, bd=True):
    """Borrower-losses stack and bad-debt bars for the given runs (names index T); the Aave bars carry hard liquidation only"""
    nb = len(names); w = 0.5 if nb == 3 else 0.55; xl = (-0.7, nb - 0.3); fs = 16 if nb == 3 else 14
    fig, ax = plt.subplots(figsize=(16, 8), dpi=220, facecolor=BG); ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(GRID); tot = []
    for i, n in enumerate(names):
        hl, sl, dl = T[n]["hl"], T[n]["sl"], T[n]["dl"]
        ax.bar(i, hl, w, color="#f0883e", linewidth=0); ax.bar(i, sl, w, bottom=hl, color="#3fb950", linewidth=0); ax.bar(i, dl, w, bottom=hl + sl, color="#58a6ff", linewidth=0); tot.append(hl + sl + dl)
    top = max(tot); ymax = top * 1.22
    for i, n in enumerate(names):
        hl, sl, dl = T[n]["hl"], T[n]["sl"], T[n]["dl"]; y = tot[i] + top * 0.015
        # a segment of at least 5 % of the tallest bar is labelled inside; thinner ones are stacked above the bar, under its total, in their own colour
        # (labels beside a thin bar run into the neighbouring bars, inside labels of two thin segments overlap)
        for lab, val, base, col in (("HL", hl, 0, "#f0883e"), ("SL", sl, hl, "#3fb950"), ("DL", dl, hl + sl, "#58a6ff")):
            if val > top * 0.05: ax.text(i, base + val / 2, f"{lab} \\${val:.2f}M", ha="center", va="center", color=BG, fontsize=15, fontweight="bold")
            elif val > 0: ax.text(i, y, f"{lab} \\${val:.2f}M", ha="center", va="bottom", color=col, fontsize=15, fontweight="bold"); y += top * 0.06
        ax.text(i, y, f"\\${tot[i]:.2f}M" + (f"  ({100*(tot[i]/tot[0]-1):+.0f}%)" if i else ""), ha="center", va="bottom", color=INK, fontsize=19, fontweight="bold"); y += top * 0.08
        ymax = max(ymax, y + top * 0.02)
    ax.set_xlim(*xl); ax.set_xticks(range(nb)); ax.set_xticklabels(names, color=INK, fontsize=fs); ax.set_yticks([]); ax.tick_params(axis="x", length=0); ax.set_ylim(0, ymax)
    ax.set_title("Borrower losses: hard liquidation + soft/de-liquidation" + (" (Aave: hard liquidation only)" if nb > 3 else "") + f", 1 Jan 2025 - 4 Jul 2026,\n{len(MARKETS)} markets", color=INK, fontsize=16, loc="left", pad=12); fig.tight_layout(); fig.savefig(f"{OUT}/losses{suffix}.png", facecolor=BG); plt.close(fig)
    # bad debt
    if not bd: return   # no Aave bad-debt chart
    fig, ax = plt.subplots(figsize=(16, 8), dpi=220, facecolor=BG); ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(GRID); v = [T[n]["bd"] for n in names]; top = max(v)
    ax.bar(range(nb), v, w, color=(["#6e7681"] + ["#3fb950" if x <= v[0] else "#f85149" for x in v[1:3]] + ["#f0883e"] * (nb - 3))[:nb], linewidth=0)   # 25 Oct 2026: a concept bar above today's is red, not green
    for i, val in enumerate(v):
        ax.text(i, val + top * 0.015, f"\\${val:.2f}M", ha="center", va="bottom", color=INK, fontsize=19, fontweight="bold")
        if i: ax.text(i, val / 2, f"{'+' if val >= v[0] else '-'}\\${abs(val - v[0]):.2f}M", ha="center", va="center", color=BG, fontsize=18, fontweight="bold")   # 25 Oct 2026 (owner): the change in dollars, not in %
    ax.set_xlim(*xl); ax.set_xticks(range(nb)); ax.set_xticklabels(names, color=INK, fontsize=fs); ax.set_yticks([]); ax.tick_params(axis="x", length=0); ax.set_ylim(0, top * 1.22)
    ax.set_title(f"Bad debt, 1 Jan 2025 - 4 Jul 2026, {len(MARKETS)} markets", color=INK, fontsize=16, loc="left", pad=12); fig.tight_layout(); fig.savefig(f"{OUT}/bad_debt{suffix}.png", facecolor=BG); plt.close(fig)

loss_and_bd_charts(list(RUNS), "")
loss_and_bd_charts(list(RUNS) + list(AAVE), "_aave", bd=False)

# ---- the oracle model charts (build.py; the concept oracle = the shipped WindowOracle's constructor values) ----
_W = open(f"{ROOT}/v2/round_r1/oracle/WINNER").read().split()
ORA = dict(window=int(_W[0]), tau=int(_W[1]), cap=float(_W[2]), asym=int(_W[3])); ORA_MODEL = f"cb2_t{ORA['tau']}_w{ORA['window']}_s{100 * ORA['cap']:g}" + (f"_a{ORA['asym']}" if ORA["asym"] else "")
P = PM.Pool("TricryptoUSDC"); d20 = P.sell(20e6)["depth_eff"]
def oracle_chart(depths, nshow, title, note, fn, legend_loc, ymax, same_block=False):
    fig, ax = plt.subplots(figsize=(16, 7), dpi=220, facecolor=BG); ax.set_facecolor(BG); ax.grid(color=GRID, linewidth=0.6); ax.tick_params(colors=INK, labelsize=12)
    for s in ax.spines.values(): s.set_color(GRID)
    cur = AT.oracle_path("today", depths); con = AT.oracle_path(ORA_MODEL, depths); b = np.arange(len(cur["p"])) - AT.LEAD; keep = (b >= -2) & (b <= nshow)
    ax.step(b[keep], 100 * cur["p"][keep], where="post", color="#58a6ff", linewidth=1.5, label="pool spot price (block end)")
    ax.step(b[keep], 100 * cur["oracle"][keep], where="post", color="#c9d1d9", linewidth=3, label="current oracle (600 s EMA)")
    ax.step(b[keep], 100 * con["oracle"][keep], where="post", color="#3fb950", linewidth=3, label="concept (window oracle)", clip_on=False)
    lo = 100 * cur["oracle"][keep].min() - 1.5
    if same_block: lo = {105: 64.0, 101: 97.2}[ymax]
    ax.set_ylim(lo, ymax); ax.set_xlim(-2, nshow); ax.set_ylabel("oracle, % of the market price", color=INK, fontsize=13); ax.set_xlabel("blocks (12 s each)", color=INK, fontsize=13)
    if same_block: ax.text(0.5, 0.45, "sell and buy-back sit in the same block:\nthe block end never moves, neither oracle sees anything", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=14)
    ax.set_title(title, color=INK, fontsize=14, loc="left", pad=10); ax.legend(loc=legend_loc, fontsize=12, facecolor="#161b22", edgecolor=GRID, labelcolor=INK); fig.tight_layout(); fig.savefig(f"{OUT}/{fn}", facecolor=BG); plt.close(fig)
oracle_chart(AT.hold(d20, 10), 40, "Oracle response: \\$20M sold into TricryptoUSDC and held down for 10 blocks (2 minutes), arbitraged back after", "", "oracle_dump_held.png", "center right", 101)

# ---- the staleness pump (build.py) ----
FALL, HOLD, D = 75, 90, 0.20; b = np.arange(-5, HOLD + 30); market = np.where(b < 0, 1.0, np.where(b < FALL, 1 - D * b / FALL, 1 - D)); pumped = np.where((b >= 0) & (b < HOLD), 1.0, market)
tt = b * AT.DT; at = tt + AT.DT - 1e-6; cur_s = np.asarray(R3.oracle("today", at, tt, pumped), float); con_s = np.asarray(R3.oracle(ORA_MODEL, at, tt, pumped), float)
fig, ax = plt.subplots(figsize=(16, 7), dpi=220, facecolor=BG); ax.set_facecolor(BG); ax.grid(color=GRID, linewidth=0.6); ax.tick_params(colors=INK, labelsize=12)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.step(b, 100 * market, where="post", color="#e6edf3", linewidth=1.8, linestyle=":", label="market price (and the pool inside each block, after the arbs)")
ax.step(b, 100 * pumped, where="post", color="#58a6ff", linewidth=1.5, label="pool spot at the block end (the attacker's last trade)")
ax.step(b, 100 * cur_s, where="post", color="#c9d1d9", linewidth=3, label="current oracle (600 s EMA)")
ax.step(b, 100 * con_s, where="post", color="#3fb950", linewidth=2.2, linestyle="--", label="concept (window oracle)", clip_on=False)
ax.axvspan(0, HOLD, color="#f85149", alpha=0.08); ax.set_xlim(-5, HOLD + 30); ax.set_ylim(100 * (1 - D) - 2, 101)
ax.set_xlabel("blocks from the start of the crash (12 s each)", color=INK, fontsize=13); ax.set_ylabel("% of the pre-crash price", color=INK, fontsize=13)
ax.set_title("Oracle staleness: the market falls 20 % in 15 min while the attacker buys the pool back to the old price at the end of every block", color=INK, fontsize=14, loc="left", pad=10)
ax.legend(loc="lower left", fontsize=12, facecolor="#161b22", edgecolor=GRID, labelcolor=INK); fig.tight_layout(); fig.savefig(f"{OUT}/oracle_stale.png", facecolor=BG); plt.close(fig)

# ---- per oracle pool family (by_pool.py) ----
M = [m for m in json.load(open(f"{ROOT}/r10/onchain/engine21_oracle_pools.json")) if m["i"] in MARKETS]
fam = lambda m: (m["pool_kinds"][0] if m["pool_kinds"] else "pre-ng tricrypto2")
GROUPS = {}
for m in M: GROUPS.setdefault(fam(m), []).append(m)
def money(x, top):
    return f"\\${x:.2f}M" if top >= 1 else f"\\${x * 1000:,.0f}k"
def totals_idx(tag, idx):
    sl = dl = hl = bd = 0.0
    for i in idx:
        w = np.loadtxt(f"{tag}/w_{i}.txt", ndmin=2); k = (w[:, 1] >= T0) & (w[:, 1] <= T1)
        if k.any(): sl += w[k, 2].sum(); dl += w[k, 3].sum(); hl += w[k, 4].sum(); bd += w[k, 14].sum() + w[k][-1, 7]
    return dict(sl=sl / 1e6, dl=dl / 1e6, hl=hl / 1e6, bd=bd / 1e6)
def charts(family, ms):
    idx = [m["i"] for m in ms]; names = list(RUNS); T = {k: totals_idx(v, idx) for k, v in RUNS.items()}
    label = f"{len(ms)} market{'s' if len(ms) > 1 else ''} whose oracle reads a {family} pool: " + ", ".join(m["name"] for m in ms)  # titles without "(lend)" / "(mint)": the kind is in the page caption (kinds_caption)
    # losses, stacked HL + SL + DL (build.py chart 1)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=220, facecolor=BG); ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(GRID); tot = []
    for i, n in enumerate(names):
        hl, sl, dl = T[n]["hl"], T[n]["sl"], T[n]["dl"]
        ax.bar(i, hl, 0.5, color="#f0883e", linewidth=0); ax.bar(i, sl, 0.5, bottom=hl, color="#3fb950", linewidth=0); ax.bar(i, dl, 0.5, bottom=hl + sl, color="#58a6ff", linewidth=0); tot.append(hl + sl + dl)
    top = max(tot + [1e-9]); ymax = top * 1.22
    for i, n in enumerate(names):
        hl, sl, dl = T[n]["hl"], T[n]["sl"], T[n]["dl"]; y = tot[i] + top * 0.015
        # as build.py: segments of at least 5 % of the tallest bar labelled inside, thinner ones stacked above the bar under its total
        for lab, val, base, col in (("HL", hl, 0, "#f0883e"), ("SL", sl, hl, "#3fb950"), ("DL", dl, hl + sl, "#58a6ff")):
            if val > top * 0.05: ax.text(i, base + val / 2, f"{lab} " + money(val, top), ha="center", va="center", color=BG, fontsize=15, fontweight="bold")
            elif val > 0: ax.text(i, y, f"{lab} " + money(val, top), ha="center", va="bottom", color=col, fontsize=15, fontweight="bold"); y += top * 0.06
        ax.text(i, y, money(tot[i], top) + (f"  ({100*(tot[i]/tot[0]-1):+.0f}%)" if i and tot[0] else ""), ha="center", va="bottom", color=INK, fontsize=19, fontweight="bold"); ymax = max(ymax, y + top * 0.1)
    ax.set_xlim(-0.7, 2.7); ax.set_xticks(range(3)); ax.set_xticklabels(names, color=INK, fontsize=16); ax.set_yticks([]); ax.tick_params(axis="x", length=0); ax.set_ylim(0, ymax)
    ax.set_title(f"Borrower losses: hard liquidation + soft/de-liquidation, 1 Jan 2025 - 4 Jul 2026\n{label}", color=INK, fontsize=15, loc="left", pad=12); fig.tight_layout(); fig.savefig(f"{OUT}/by_pool/{family.replace(' ', '_')}_losses.png", facecolor=BG); plt.close(fig)
    # bad debt (build.py chart 2), only when the family has any
    v = [T[n]["bd"] for n in names]; top = max(v + [1e-9]); bd_png = None
    if max(v) >= 0.001:  # at least $1k of bad debt in one of the versions
        fig, ax = plt.subplots(figsize=(16, 8), dpi=220, facecolor=BG); ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(GRID)
        ax.bar(range(3), v, 0.5, color=["#6e7681", "#3fb950", "#3fb950"], linewidth=0)
        for i, val in enumerate(v):
            ax.text(i, val + top * 0.015, money(val, top), ha="center", va="bottom", color=INK, fontsize=19, fontweight="bold")
            if i and v[0] and val > top * 0.08: ax.text(i, val / 2, f"{100*(val/v[0]-1):+.0f}%", ha="center", va="center", color=BG, fontsize=18, fontweight="bold")
        ax.set_xlim(-0.7, 2.7); ax.set_xticks(range(3)); ax.set_xticklabels(names, color=INK, fontsize=16); ax.set_yticks([]); ax.tick_params(axis="x", length=0); ax.set_ylim(0, top * 1.22)
        ax.set_title(f"Bad debt, 1 Jan 2025 - 4 Jul 2026\n{label}", color=INK, fontsize=15, loc="left", pad=12); fig.tight_layout(); bd_png = f"{family.replace(' ', '_')}_bad_debt.png"; fig.savefig(f"{OUT}/by_pool/{bd_png}", facecolor=BG); plt.close(fig)
    return T, bd_png
for family, ms in GROUPS.items():
    T_f, _ = charts(family, ms)

# ---- gas and bytecode size per contract file (contracts_build.py) ----
GAS = json.load(open(f"{ROOT}/v2/fork/measure_v2.json")); M_ = GAS["meta"]
KEY_BASE, KEY_ON, KEY_OFF, KEY_BASE_ORACLE = "v2 base", "v2 concept; DL on", "v2 concept; DL off", "v2 base; v2 oracle"
COL_BASE, COL_ON, COL_OFF = "#6e7681", "#58a6ff", "#3fb950"; LIMIT = 24576
SZB, SZC = (json.load(open(f"{ROOT}/v2/fork/build/{t}/sizes.json")) for t in ("base", "concept"))
def _sz(lc, amm, view, mint=False):
    d = {"oracle contract": SZC["WindowOracle"]["deployed"]}
    for nm, src in ((("MintController" if mint else "LendController"), lc), ("AMM", amm), (("ControllerView" if mint else "LendControllerView"), view)):
        d[f"{nm} blueprint"], d[f"{nm} instance"] = src[nm]["blueprint_code"], src[nm]["deployed"]
    return d
SIZES = {"v2 base": _sz(SZB, SZB, SZB), "v2 concept; DL on": _sz(SZC, SZB, SZC), "v2 concept; DL off": _sz(SZC, SZC, SZC),
         "mint: v2 base": _sz(SZB, SZB, SZB, True), "mint: v2 concept; DL on": _sz(SZC, SZB, SZC, True), "mint: v2 concept; DL off": _sz(SZC, SZC, SZC, True)}
SIZES["v2 base; v2 oracle"] = {**SIZES["v2 base"], "oracle contract": SZB["CrvUSDAggregatorWrapper"]["deployed"] + SZB["OracleFromCurvePools"]["deployed"]}
INFO = GAS["info"]; CAP_FRAC = INFO[KEY_BASE]["v1 cap on this state (frac, f_remove)"][0]; CON_FRAC = INFO[KEY_ON]["liquidate (liquidator, frac 100 %)"]["repaid frac"]
SHOW = {"liquidate (liquidator, at the v1 cap's frac)": f"liquidate by a liquidator, frac {100 * CAP_FRAC:.1f} %", "liquidate (liquidator, frac 100 %)": f"liquidate by a liquidator, asks 100 % (concept stops at {100 * CON_FRAC:.1f} %)",
        "liquidate (liquidator, frac 50 %)": "liquidate by a liquidator, frac 50 %", "liquidate (self, frac 100 %)": "liquidate self, 100 %", "liquidate (self, frac 50 %)": "liquidate self, 50 %",
        "tokens_to_liquidate() view, liquidator": "tokens_to_liquidate() view, liquidator", "tokens_to_liquidate() view, self": "tokens_to_liquidate() view, self",
        "liquidate_health_preview() view, liquidator": "liquidate_health_preview() view, liquidator", "liquidate_health_preview() view, self 50 %": "liquidate_health_preview() view, self 50 %",
        "repay (full, closes the loan)": "repay, full (closes the loan)"}
def wrap(label, width):
    """x-axis label over lines: may break after "_", never inside a word or a short "( ... )" group"""
    label = re.sub(r"\(([^)]{1,12})\)", lambda m: "(" + m.group(1).replace(" ", "\u00a0") + ")", label)
    return textwrap.fill(label.replace("_", "_ "), width, break_long_words=False).replace("_ ", "_")
def chart_bars(title, labels, series, fn, unit, limit=None):
    """series: list of (legend label, values, colour); one group of bars per label; a value None where another series has one = the call reverts"""
    n_lab = len(labels); fig, ax = plt.subplots(figsize=(14, 6.4 if unit == "gas" else 6), dpi=200, facecolor=BG); ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(GRID); x = np.arange(n_lab); n = len(series); w = 0.8 / n
    top = max([v for _, vals, _ in series for v in vals if v] + [1]); top = max(top, limit or 0)
    for k, (lab, vals, col) in enumerate(series):
        off = (k - (n - 1) / 2) * w; ax.bar(x + off, [v or 0 for v in vals], w * 0.92, color=col, linewidth=0, label=lab)
        for i, v in enumerate(vals):
            if v is None and any(s[1][i] for s in series): ax.text(x[i] + off, top * 0.015, "reverts", rotation=90, ha="center", va="bottom", color="#f85149", fontsize=9, fontweight="bold")
            elif not v: continue
            elif unit == "gas": ax.text(x[i] + off, v + top * 0.012, f"{v:,}", rotation=90, ha="center", va="bottom", color=INK, fontsize=8.5, fontweight="bold")
            elif v > top * 0.3: ax.text(x[i] + off, v - top * 0.02, f"{v:,}", rotation=90, ha="center", va="top", color=BG, fontsize=10, fontweight="bold")
            else: ax.text(x[i] + off, v + top * 0.015, f"{v:,}", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    if limit: ax.axhline(limit, color="#f85149", linewidth=1.2, linestyle="--", label=f"EIP-170 limit {limit:,} B")
    ax.set_xticks(x); ax.set_xticklabels([wrap(l, 22 if n_lab < 6 else 17) for l in labels], color=INK, fontsize=10); ax.set_yticks([]); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, n_lab - 0.4); ax.set_ylim(0, top * (1.45 if unit == "gas" else 1.2))
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10); ax.legend(loc="upper left", ncol=n + (1 if limit else 0), fontsize=10, facecolor="#161b22", edgecolor=GRID, labelcolor=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}", facecolor=BG); plt.close(fig)
CONF = {   # per file: (DL-off only, base key, base label, gas charts (subtitle, rows), size rows (label, market prefix, size key)) as contracts_build.py
    "WindowOracle.vy": (False, KEY_BASE_ORACLE, "v2 base: v2's own USD oracle", [("", ["oracle.price_w()", "oracle.price() view", "AMM.price_oracle() view", "borrow_more", "AMM.exchange (0 -> 1)"])],
                        [("oracle contract(s)", "", "oracle contract")], "v2_WindowOracle"),
    "controller.vy": (False, KEY_BASE, "v2 base", [("borrower calls", ["create_loan (10 bands)", "borrow_more", "add_collateral", "remove_collateral", "repay", "repay (full, closes the loan)", "health() view"]),
                       ("liquidation", ["liquidate (liquidator, at the v1 cap's frac)", "liquidate (liquidator, frac 100 %)", "liquidate (liquidator, frac 50 %)", "liquidate (self, frac 100 %)", "liquidate (self, frac 50 %)", "tokens_to_liquidate() view, liquidator", "tokens_to_liquidate() view, self"])],
                      [("LendController blueprint", "", "LendController blueprint"), ("LendController instance", "", "LendController instance"), ("MintController blueprint", "mint: ", "MintController blueprint"), ("MintController instance", "mint: ", "MintController instance")], "v2_controller"),
    "ControllerView.vy": (False, KEY_BASE, "v2 base", [("", ["liquidate_health_preview() view, liquidator", "liquidate_health_preview() view, self 50 %", "users_to_liquidate() view", "user_state() view", "max_borrowable() view", "create_loan_health_preview() view", "borrow_more_health_preview() view", "repay_health_preview() view"])],
                          [("LendControllerView blueprint", "", "LendControllerView blueprint"), ("LendControllerView instance", "", "LendControllerView instance"), ("ControllerView blueprint (mint)", "mint: ", "ControllerView blueprint"), ("ControllerView instance (mint)", "mint: ", "ControllerView instance")], "v2_ControllerView"),
    "AMM.vy": (True, KEY_BASE, "v2 base (= the DL on market's AMM, 4x fee)", [("", ["AMM.exchange (0 -> 1)", "AMM.exchange_dy (0 -> 1)", "AMM.exchange (1 -> 0)", "AMM.get_dy(0,1) view", "AMM.get_dx(0,1) view", "AMM.get_x_down() view", "AMM.get_y_up() view", "AMM.get_amount_for_price() view", "create_loan (10 bands)"])],
               [("AMM blueprint", "", "AMM blueprint"), ("AMM instance", "", "AMM instance")], "v2_AMM"),
}
for name, (dl_off_only, base, base_label, gas, size, stem) in CONF.items():
    legend = [(base_label, base, COL_BASE)] + ([] if dl_off_only else [("v2 concept; DL on", KEY_ON, COL_ON)]) + [("v2 concept; DL off", KEY_OFF, COL_OFF)]
    size_of = lambda key, pre, k: SIZES.get(pre + key, {}).get(k)
    chart_bars(f"{name}: deployed bytecode size, bytes (a blueprint = 3 B preamble + initcode)", [s[0] for s in size],
               [(lab, [size_of(key, pre, k) for _, pre, k in size], col) for lab, key, col in legend], f"size_{stem}.png", "bytes", limit=LIMIT)
    for j, (sub, rows) in enumerate(gas):
        rows = [r for r in rows if any(r in GAS["gas"].get(key, {}) for _, key, _ in legend)]; fn = f"gas_{stem}{'_' + str(j + 1) if j else ''}.png"
        chart_bars(f"{name}: gas per transaction on a mainnet fork{', ' + sub if sub else ''} (block {M_['block']:,}, same fork state, fresh accounts, second calls)", [SHOW.get(r, r) for r in rows],
                   [(lab, [GAS["gas"].get(key, {}).get(r) for r in rows], col) for lab, key, col in legend], fn, "gas")
print("charts written to", OUT)
