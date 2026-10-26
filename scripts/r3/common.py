"""common.py — round 3 of the LLAMMA oracle study (LLAMMA-Design/r3, 14 Oct 2026): the 23 designs in their fixed order
(today first), their oracles.py specs, one colour per design, the oracle dispatcher and the chart style.

Label grammar (the round-3 family):  c{hl}[_n{N}_m{M}_d{D}_h{hf}_v{vfast}][_s{vslow}]
  = oracles.regime(N, M, D/100, hf=hf, raw=hl/ln2, vfast=vfast/100, vslow=vslow/100 if given), eps 2e-4;
  plain c{hl} = regime(N=10, M=11, ...) with raw=hl/ln2: the crash regime can never come = a plain slower EMA (half-life hl s).
Round-2 references: today = oracles.today (the live pool EMA); dec_N_M_D_hfX = regime(N, M, D/100, hf=X); rate_0.1_v =
  regime(10, 7, 0.02, hf=30, vslow=0.001, vfast=v/100); robust_max_K10 = robust(K=10, how='max'); adaptive = adaptive_r6.
RAW = 865 s is the ma_time raw of TricryptoUSDC and TriCRV over the whole span (data/pool_params.json; half-life 600 s,
  alpha = exp(-12/865) per 12-s block: one block boundary moves the EMA by 1.378 % of the gap). It is the scalar raw of every
  regime / robust design here (oracles.regime takes a scalar); `today` takes the per-block raw array (raw_blk) when one is
  given (the r2a pipeline's handling: USDT ran 601 s raw for a while), else RAW.
"""
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "r3")
sys.path.insert(0, ROOT)
import oracles as O  # noqa: E402

LN2 = math.log(2.0)
RAW = 865.0                       # ma_time raw (s) of TricryptoUSDC / TriCRV: alpha = exp(-dt / RAW)
EPS = 2e-4                        # regime(): a block counts as 'down' when the spot fell by more than eps
DT = 12.0                         # one block

LABELS = [
    "today", "dec_10_7_2_hf60", "dec_10_7_3_hf120", "rate_0.1_0.5", "rate_0.1_0.3", "robust_max_K10", "adaptive",
    "c1200", "c1800",
    "c600_n25_m18_d5_h60_v0.5", "c600_n25_m18_d5_h120_v0.3", "c600_n50_m35_d8_h60_v0.5", "c600_n50_m35_d8_h120_v0.3",
    "c1200_n25_m18_d5_h60_v0.5", "c1200_n25_m18_d5_h120_v0.3", "c1200_n50_m35_d8_h60_v0.5", "c1200_n50_m35_d8_h120_v0.3",
    "c1800_n25_m18_d5_h60_v0.5", "c1800_n25_m18_d5_h120_v0.3", "c1800_n50_m35_d8_h60_v0.5", "c1800_n50_m35_d8_h120_v0.3",
    "c600_n25_m18_d5_h120_v0.3_s0.1", "c1200_n25_m18_d5_h120_v0.3_s0.1",
]
assert len(LABELS) == 23 and len(set(LABELS)) == 23
LABELS += ['casc_h600', 'casc_h1200', 'casc_h1800', 'casc_h600_v0.25', 'casc_h600_v0.5', 'casc_h1200_v0.25', 'casc_h1200_v0.5', 'casc_h1800_v0.25', 'casc_h1800_v0.5']
LABELS += ['casc_h2400', 'casc_h3600']  # round 4 (14 Oct 2026): the cascade oracle contract

# one colour per design, the same in every chart: references grey / blue / teal / orange / pink / red (one hue each, as
# r2b/r2b.py COL does it), c600 greens, c1200 golds, c1800 purples, the _s0.1 pair cyan; dark shade = h60, light = h120
COLOR = {
    "today": "#8b949e", "dec_10_7_2_hf60": "#1f6feb", "dec_10_7_3_hf120": "#39c5bb",
    "rate_0.1_0.5": "#db6d28", "rate_0.1_0.3": "#f0a04b", "robust_max_K10": "#db61a2", "adaptive": "#f85149",
    "c1200": "#a16207", "c1800": "#6b21a8",
    "c600_n25_m18_d5_h60_v0.5": "#15803d", "c600_n25_m18_d5_h120_v0.3": "#4ade80",
    "c600_n50_m35_d8_h60_v0.5": "#4d7c0f", "c600_n50_m35_d8_h120_v0.3": "#a3e635",
    "c1200_n25_m18_d5_h60_v0.5": "#ca8a04", "c1200_n25_m18_d5_h120_v0.3": "#facc15",
    "c1200_n50_m35_d8_h60_v0.5": "#b45309", "c1200_n50_m35_d8_h120_v0.3": "#fbbf24",
    "c1800_n25_m18_d5_h60_v0.5": "#7e22ce", "c1800_n25_m18_d5_h120_v0.3": "#c084fc",
    "c1800_n50_m35_d8_h60_v0.5": "#5b21b6", "c1800_n50_m35_d8_h120_v0.3": "#a78bfa",
    "c600_n25_m18_d5_h120_v0.3_s0.1": "#0e7490", "c1200_n25_m18_d5_h120_v0.3_s0.1": "#22d3ee",
}
COLOR.update({'casc_h600': '#2dd4bf', 'casc_h1200': '#14b8a6', 'casc_h1800': '#0f766e', 'casc_h600_v0.25': '#a3e635', 'casc_h600_v0.5': '#65a30d', 'casc_h1200_v0.25': '#facc15', 'casc_h1200_v0.5': '#ca8a04', 'casc_h1800_v0.25': '#f472b6', 'casc_h1800_v0.5': '#be185d'})
COLOR.update({'casc_h2400': '#5eead4', 'casc_h3600': '#99f6e4'})
assert set(COLOR) == set(LABELS)
BG, INK, GRID, MUTED = "#0d1117", "#e6edf3", "#21262d", "#8b949e"

_CB = re.compile(r"^cb_t(\d+)_h(\d+)(?:_s([\d.]+))?$")
_CB2 = re.compile(r"^cb2_t(\d+)_w(\d+)(?:_h(\d+))?(?:_s([\d.]+))?(?:_c(all|end))?(?:_a([12]))?$")
_CBAND = re.compile(r"^cband_t(\d+)_r(ema600|ema400|spot)(?:_s([\d.]+))?$")
_ROB = re.compile(r"^rob_K(\d+)_h(\d+)$")
_RADP = re.compile(r"^radp_K(\d+)(?:_g([\d.]+))?(?:_h(\d+))?(?:_f(\d+))?(?:_s([\d.]+))?$")
_ADP = re.compile(r"^adp_s([\d.]+)$")
_CASC = re.compile(r"^casc_h(\d+)(?:_v([\d.]+))?$")
_C = re.compile(r"^c(\d+)(?:_n(\d+)_m(\d+)_d(\d+(?:\.\d+)?)_h(\d+(?:\.\d+)?)_v(\d+(?:\.\d+)?))?(?:_s(\d+(?:\.\d+)?))?$")


def spec(label):
    """-> dict(kind, kwargs): kind in today | regime | robust | adaptive; kwargs = exactly the keyword arguments of the
    oracles.py function of that kind (regime: N, M, D, hf, raw, eps[, vfast][, vslow][, calmK]; robust: K, how, raw)"""
    if label == "today":
        return dict(kind="today", kwargs={})
    if label == "adaptive":
        return dict(kind="adaptive", kwargs={})
    if label == "robust_max_K10":
        return dict(kind="robust", kwargs=dict(K=10, how="max", raw=RAW))
    f = label.split("_")
    if f[0] == "dec":                                 # dec_N_M_D_hfX
        N, M, D, hf = int(f[1]), int(f[2]), float(f[3]) / 100.0, float(f[4][2:])
        return dict(kind="regime", kwargs=dict(N=N, M=M, D=D, hf=hf, raw=RAW, eps=EPS))
    if f[0] == "rate":                                # rate_vslow_vfast (percent per block)
        return dict(kind="regime", kwargs=dict(N=10, M=7, D=0.02, hf=30.0, raw=RAW, eps=EPS,
                                               vslow=float(f[1]) / 100.0, vfast=float(f[2]) / 100.0))
    m = _C.match(label)
    if m:
        hl = float(m.group(1))
        kw = dict(raw=hl / LN2, eps=EPS)
        if m.group(2) is None:                        # plain c{hl}: M > N, the crash regime never comes
            kw.update(N=10, M=11, D=1.0, hf=60.0)
        else:
            kw.update(N=int(m.group(2)), M=int(m.group(3)), D=float(m.group(4)) / 100.0, hf=float(m.group(5)),
                      vfast=float(m.group(6)) / 100.0)
        if m.group(7) is not None:
            kw["vslow"] = float(m.group(7)) / 100.0
        return dict(kind="regime", kwargs=kw)
    mb2 = _CB2.match(label)
    if mb2:                                           # cb2_t{tau s}_w{W s}[_h{ref half-life s}][_s{cap}][_c{all|end}]: the window-extremes band (round 10)
        kw = dict(tau=float(mb2.group(1)), W=float(mb2.group(2)))
        if mb2.group(3) is not None: kw["h"] = float(mb2.group(3))
        if mb2.group(4) is not None: kw["v"] = float(mb2.group(4)) / 100.0
        kw["cov"] = mb2.group(5) or "all"
        if mb2.group(6): kw["asym"] = int(mb2.group(6))   # _a1 / _a2: the cap limits only rises / only falls (round_r1 oracle, 24 Oct 2026)
        return dict(kind="cband2", kwargs=kw)
    mc2 = _CB.match(label)
    if mc2:                                           # cb_t{tau}_h{ref half-life}[_s{cap}]: band oracle on an EMA of given half-life (round 9)
        kw = dict(tau=float(mc2.group(1)), ref="ema", ref_hl=float(mc2.group(2)))
        if mc2.group(3): kw["v"] = float(mc2.group(3)) / 100.0
        return dict(kind="cband", kwargs=kw)
    mb = _CBAND.match(label)
    if mb:                                            # cband_t{tau s}_r{ema600|ema400|spot}: the confidence-band oracle (round 8)
        kw = dict(tau=float(mb.group(1)), ref=mb.group(2))
        if mb.group(3): kw["v"] = float(mb.group(3)) / 100.0
        return dict(kind="cband", kwargs=kw)
    mo = _ROB.match(label)
    if mo:                                            # rob_K{K}_h{hl}: max of the last K block prices into a plain EMA of half-life hl (round 6)
        return dict(kind="robust", kwargs=dict(K=int(mo.group(1)), how="max", raw=float(mo.group(2)) / LN2))
    mr = _RADP.match(label)
    if mr:                                            # radp_K{K}[_g{g0 %}][_h{hl s}][_f{hl_fast s}][_s{cap %/block}]: robust-input adaptive (round 6)
        kw = dict(K=int(mr.group(1)))
        if mr.group(2): kw["g0"] = float(mr.group(2)) / 100.0
        if mr.group(3): kw["hl"] = float(mr.group(3))
        if mr.group(4): kw["hl_fast"] = float(mr.group(4))
        if mr.group(5): kw["v"] = float(mr.group(5)) / 100.0
        return dict(kind="radp", kwargs=kw)
    ma = _ADP.match(label)
    if ma:                                            # adp_s{cap}: the 1 Oct adaptive oracle with the contract's output cap (round 5)
        return dict(kind="adp_cap", kwargs=dict(v=float(ma.group(1)) / 100.0))
    mc = _CASC.match(label)
    if mc:                                            # casc_h{h2}[_v{cap %/block}]: the cascade oracle contract (round 4)
        v = None if mc.group(2) is None else float(mc.group(2)) / 100.0
        return dict(kind="cascade", kwargs=dict(raw=RAW, h2=float(mc.group(1)), vdown=v, vup=v))
    raise ValueError(f"unknown design label: {label}")


SPEC = {l: spec(l) for l in LABELS}


def is_regime(label):
    """True when the design has a crash regime that can come (regime kind with M <= N)"""
    s = SPEC.get(label) or spec(label)
    return s["kind"] == "regime" and s["kwargs"]["M"] <= s["kwargs"]["N"]


def regime_key(label):
    """(N, M, D, eps) of the crash regime, or None (r2c part 3 flags)"""
    if not is_regime(label):
        return None
    k = (SPEC.get(label) or spec(label))["kwargs"]
    return (k["N"], k["M"], k["D"], k["eps"])


def family(label):
    if label == "today": return "today"
    if label.startswith("dec"): return "dec"
    if label.startswith("rate"): return "rate"
    if label.startswith("robust"): return "robust"
    if label == "adaptive": return "adaptive"
    if label.startswith("casc"): return "casc"
    if label.startswith("cb2"): return "cband2"
    if label.startswith("cband"): return "cband"
    if label.startswith("radp"): return "radp"
    if label.startswith("adp"): return "adp"
    m = _C.match(label)
    if m:
        if m.group(7) is not None: return "s0.1"
        return f"c{m.group(1)}"
    return "?"


def oracle(label, at, t_blk, p_blk, raw_blk=None, p0=None, e_blk=None, pmin=None, pmax=None):
    """the design's oracle at the times `at` on the block series (t_blk, p_blk = last_prices after each trade block).
    today: raw_blk (scalar or per-block array, default RAW) and e_blk (the EMA state per block; built from p0 when absent)
           as r2a/build_series.py and r2c/episodes.py hand them over; regime / robust: scalar raw from the spec (RAW, or
           hl/ln2 for the c-designs), p0 = the oracle's start value (default: the first spot); adaptive: no parameters."""
    s = SPEC.get(label) or spec(label)
    kind, kw = s["kind"], s["kwargs"]
    t, p = np.asarray(t_blk, float), np.asarray(p_blk, float)
    if kind == "today":
        raw = RAW if raw_blk is None else raw_blk
        e = O.ema_at_blocks(t, p, raw=raw, p0=p0) if e_blk is None else e_blk
        return O.today(at, t, p, raw=raw, e_blk=e)
    if kind == "regime":
        return O.regime(at, t, p, p0=p0, **kw)
    if kind == "robust":
        return O.robust(at, t, p, p0=p0, **kw)
    if kind == "adaptive":
        return O.adaptive_r6(at, t, p)
    if kind == "cband":
        raw = RAW if raw_blk is None else raw_blk
        return O.cband(at, t, p, raw=raw, e_blk=e_blk, p0=p0, **kw)
    if kind == "cband2":                              # round 10: pmin / pmax = the in-block extremes (coverage 'all'); 'end' = block ends
        kw2 = {k: v for k, v in kw.items() if k != "cov"}
        if kw.get("cov", "all") == "end": pmin = pmax = None
        return O.cband2(at, t, p, p0=p0, pmin=pmin, pmax=pmax, **kw2)
    if kind == "radp":
        return O.radp(at, t, p, **kw)
    if kind == "adp_cap":
        return O.adaptive_r6(at, t, p, v=kw["v"])
    if kind == "cascade":
        raw = RAW if raw_blk is None else raw_blk
        e = O.ema_at_blocks(t, p, raw=raw, p0=p0) if e_blk is None else e_blk
        return O.cascade(at, t, p, raw=raw, h2=kw["h2"], vdown=kw["vdown"], vup=kw["vup"], e_blk=e, p0=p0)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------------- chart style
def style(ax, title, xlabel="", ylabel=""):
    """dark theme, one y axis, a short left-aligned title"""
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.7, axis="y")
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def spans_decades(v, decades=2.0):
    """True when the finite positive values span more than `decades` orders of magnitude (-> log scale)"""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x) and x > 0], float)
    return len(v) > 1 and math.log10(v.max() / v.min()) > decades


def fmt_value(y):
    if y is None or not np.isfinite(y):
        return "n/a"
    a = abs(y)
    if a >= 1e9: return f"{y / 1e9:.2f}G"
    if a >= 1e6: return f"{y / 1e6:.2f}M"
    if a >= 1e4: return f"{y / 1e3:.0f}k"
    if a >= 100: return f"{y:.0f}"
    if a >= 10: return f"{y:.1f}"
    if a >= 1: return f"{y:.2f}"
    return f"{y:.3g}"


def bar_chart(values, title, ylabel, fn, labels=None, fmt=fmt_value, log=None, width_px=2000, height_px=900, dpi=100):
    """one bar per design in the fixed LABELS order (today first), COLOR colours, the value written on every bar,
    log y when the values span decades (log=None: automatic), nothing clipped; values: {label: number | None}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = LABELS if labels is None else labels
    v = np.array([values.get(n) if values.get(n) is not None else np.nan for n in labels], float)
    uselog = spans_decades(v) if log is None else log
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    x = np.arange(len(labels))
    fin = np.isfinite(v)
    if uselog:
        pos = v[fin & (v > 0)]
        floor = (pos.min() / 3.0) if len(pos) else 1.0
        ax.bar(x[fin], np.maximum(v[fin], floor), bottom=floor * 0 + 0, width=0.72,
               color=[COLOR.get(n, MUTED) for n, f in zip(labels, fin) if f], linewidth=0)
        ax.set_yscale("log")
        top = pos.max() if len(pos) else 1.0
        ax.set_ylim(floor, top * 3.0)
        for k, y in enumerate(v):
            ax.text(k, (max(y, floor) if np.isfinite(y) else floor) * 1.12, fmt(y) if np.isfinite(y) else "n/a",
                    ha="center", va="bottom", color=INK, fontsize=9)
    else:
        ax.bar(x[fin], v[fin], width=0.72, color=[COLOR.get(n, MUTED) for n, f in zip(labels, fin) if f], linewidth=0)
        top = float(np.nanmax(v)) if fin.any() else 1.0
        bot = min(0.0, float(np.nanmin(v))) if fin.any() else 0.0
        span = max(top - bot, 1e-9)
        for k, y in enumerate(v):
            yy = y if np.isfinite(y) else 0.0
            ax.text(k, yy + span * 0.012 if yy >= 0 else yy - span * 0.012, fmt(y) if np.isfinite(y) else "n/a",
                    ha="center", va="bottom" if yy >= 0 else "top", color=INK, fontsize=9)
        ax.set_ylim(bot - (span * 0.08 if bot < 0 else 0), top + span * 0.12 if top > 0 else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    style(ax, title, "", ylabel)
    fig.tight_layout()
    fig.savefig(fn, facecolor=BG)
    plt.close(fig)
    return fn


if __name__ == "__main__":
    for l in LABELS:
        print(f"{l:34s} {COLOR[l]}  {family(l):8s} {SPEC[l]['kind']:8s} {SPEC[l]['kwargs']}")
