"""round_r6/engine (25 Oct 2026): the CRV market (engine market 3) of every engine run the concept page reads, re-run with TriCRV's
real on-chain sale quotes (--venue-grid venue_crv.txt) instead of the generic $20M venue. Each run is first reproduced WITHOUT the
grid on bin/replay_sim_x_r6 and must match its stored w_3/l_3/b_3 byte for byte (else it is left alone and reported), then run
with the grid into m3/<run>/. Arguments = the run's own log.txt "cmd:" line as run_v5s.sh / run_v4.sh expand it for market 3.
usage: nice -n 19 python3 run_m3.py"""
import os, re, shlex, subprocess, filecmp
from concurrent.futures import ThreadPoolExecutor
LD = os.environ.get("LLAMMA_DESIGN", "."); ENG = f"{LD}/engine"; H = f"{LD}/v2/round_r6/engine"
BIN = f"{H}/bin/replay_sim_x_r6"; GRID = f"{H}/venue_crv.txt"
RUNS = ["rf4_today", "rf4_nodl", "rf4_dlon", "aave1_r4", "aave4_r4", "rf4_today_g38", "rf4_nodl_g38", "rf4_dlon_g38", "rf4_nodl_oB0", "rf4_dlon_oB0", "rd_today", "vfy_today_exact"]
def args_of(run):
    cmd = [l for l in open(f"{ENG}/runs/{run}/log.txt") if l.startswith("cmd:")][-1][4:].strip()
    env = dict(re.findall(r"(\w+)=(\S*)", cmd.split(" run_v")[0]))
    rest = shlex.split(cmd.split(".sh ", 1)[1])[1:]          # drop the TAG
    x = []
    if env.get("ORA_SERIES"): x += ["--oracle-series", env["ORA_SERIES"] % 2]       # und 2 = CRV
    if env.get("WO_POOL"): x += ["--wo-pool", env["WO_POOL"] % 2]
    if env.get("WO_SWAPS"): x += ["--wo-swaps", env["WO_SWAPS"] % 3]
    return ["--in", "mkt_3.txt", "--candles", "../candles_2_onchain.bin", "--depth", "2e7"] + x + rest
def go(run):
    a = args_of(run); d = f"{H}/m3/{run}"; os.makedirs(d, exist_ok=True)
    def ex(tag, extra):
        out = [f"--out", f"{d}/w_{tag}.txt", "--liq", f"{d}/l_{tag}.txt", "--bd", f"{d}/b_{tag}.txt"]
        r = subprocess.run(["nice", "-n", "19", BIN] + a[:6] + out + a[6:] + extra, cwd=ENG, capture_output=True, text=True)
        open(f"{d}/log_{tag}.txt", "w").write(r.stderr); return r.returncode
    ex("repro", [])
    same = all(filecmp.cmp(f"{d}/{k}_repro.txt", f"{ENG}/runs/{run}/{k}_3.txt", shallow=False) for k in "wlb")
    if not same: return run, "NOT reproduced (left alone)"
    ex("grid", ["--venue-grid", GRID])
    return run, "reproduced byte for byte; grid run done"
with ThreadPoolExecutor(6) as P:
    for run, msg in P.map(go, RUNS): print(f"{run:16s} {msg}", flush=True)
