"""round_r6/engine run_th2.py (26 Oct 2026): the owner moved controller.vy's TARGET_HEALTH from 1.5 to 2.0 (health +100 % after a liquidation).
Re-runs the six concept page runs (rf6_nodl / rf6_dlon / *_g38 / *_oB0) on all 21 markets with --v2c-th 2 into engine/runs/rf7_*.
Each market's command = run_v5s.sh's expansion of the rf4 run's "cmd:" line (env ORA_SERIES / WO_POOL / WO_SWAPS, CDEC8 = "4 12 13 18" as
the rf4 runs used, ORA_DUMP when the source run has o_ files) on bin/replay_sim_x_r6, market 3 (CRV) with the venue grid as rf6.
Pass 1 (repro): without the new flag, every w/l/b file must equal the rf6 run's byte for byte, else the run is not made.
Pass 2: + --v2c-th 2. log.txt = the market lines + 'wall all' + the source cmd line with ' --v2c-th 2' appended + a note.
usage: nice -n 19 python3 run_th2.py"""
import os, re, shlex, subprocess, filecmp, time
from concurrent.futures import ThreadPoolExecutor, as_completed
LD = os.environ.get("LLAMMA_DESIGN", "."); ENG = f"{LD}/engine"; H = f"{LD}/v2/round_r6/engine"
BIN = f"{H}/bin/replay_sim_x_r6"; GRID = f"{H}/venue_crv.txt"; CDEC8 = (4, 12, 13, 18); TH = ["--v2c-th", "2"]
RUNS = {"rf6_nodl": "rf7_nodl", "rf6_dlon": "rf7_dlon", "rf6_nodl_g38": "rf7_nodl_g38", "rf6_dlon_g38": "rf7_dlon_g38", "rf6_nodl_oB0": "rf7_nodl_oB0", "rf6_dlon_oB0": "rf7_dlon_oB0"}
def src_cmd(run):
    return [l for l in open(f"{ENG}/runs/{run}/log.txt") if l.startswith("cmd:")][-1][4:].strip().split(" ; ROUND_R6")[0]
def margs(run, i, outdir):
    cmd = src_cmd(run); env = dict(re.findall(r"(\w+)=(\S*)", cmd.split(" run_v")[0])); rest = shlex.split(cmd.split(".sh ", 1)[1])[1:]
    und = int(open(f"{ENG}/mkt_{i}.txt").readline().split()[2])
    c = {0: "../candles_0.bin", 1: "../candles_1.bin", 2: "../candles_2_onchain.bin"}[und]; dep = "2e7" if und == 2 else "2e8"
    x = []
    if env.get("ORA_SERIES"): x += ["--oracle-series", env["ORA_SERIES"] % und]
    if os.path.exists(f"{ENG}/runs/{run}/o_{i}.f64"): x += ["--oracle-dump", f"{outdir}/o_{i}.f64"]
    if env.get("WO_POOL"): x += ["--wo-pool", env["WO_POOL"] % und]
    if env.get("WO_SWAPS"): x += ["--wo-swaps", env["WO_SWAPS"] % i]
    if i in CDEC8: x += ["--cdec", "8"]
    a = ["--in", f"mkt_{i}.txt", "--candles", c, "--depth", dep, "--out", f"{outdir}/w_{i}.txt", "--liq", f"{outdir}/l_{i}.txt", "--bd", f"{outdir}/b_{i}.txt"] + x + rest
    return a + (["--venue-grid", GRID] if i == 3 else [])
def one(run, new, tag, i):
    d = f"{H}/th2/{tag}/{new}" if tag == "repro" else f"{ENG}/runs/{new}"
    a = margs(run, i, d) + (TH if tag == "th2" else []); t = time.time()
    r = subprocess.run(["nice", "-n", "19", BIN] + a, cwd=ENG, capture_output=True, text=True)
    return run, new, tag, i, r.returncode, r.stderr + f"time mkt_{i} {time.time() - t:.0f} s rc {r.returncode}\n"
size = {i: int(open(f"{ENG}/mkt_{i}.txt").readline().split()[6]) for i in range(21)}
jobs = []
for run, new in RUNS.items():
    assert not os.path.exists(f"{ENG}/runs/{new}"), new
    for tag in ("repro", "th2"):
        os.makedirs(f"{H}/th2/{tag}/{new}" if tag == "repro" else f"{ENG}/runs/{new}", exist_ok=True)
        jobs += [(run, new, tag, i) for i in range(21)]
jobs.sort(key=lambda j: (-size[j[3]], j[2] != "repro"))
print(f"{len(jobs)} market runs", flush=True); T = time.time(); logs = {}
with ThreadPoolExecutor(6) as P:
    for f in as_completed([P.submit(one, *j) for j in jobs]):
        run, new, tag, i, rc, err = f.result(); assert rc == 0, (run, tag, i, err[-400:])
        logs.setdefault((new, tag), {})[i] = err
for run, new in RUNS.items():
    bad = [f"{k}_{i}" for i in range(21) for k in "wlb" if not filecmp.cmp(f"{H}/th2/repro/{new}/{k}_{i}.txt", f"{ENG}/runs/{run}/{k}_{i}.txt", shallow=False)]
    bad += [f"o_{i}" for i in range(21) if os.path.exists(f"{ENG}/runs/{run}/o_{i}.f64") and not filecmp.cmp(f"{H}/th2/repro/{new}/o_{i}.f64", f"{ENG}/runs/{run}/o_{i}.f64", shallow=False)]
    print(f"{run:14s} repro: {'byte for byte' if not bad else 'DIFFERS ' + ' '.join(bad)}", flush=True)
    note = f" ; ROUND_R6 TH2: all 21 markets on v2/round_r6/engine/bin/replay_sim_x_r6 (CRV on the venue grid v2/round_r6/engine/venue_crv.txt), CDEC8=4 12 13 18, TARGET_HEALTH 2.0 (--v2c-th 2); without the flag every file equals {run} (run_th2.py)"
    body = "".join(logs[(new, "th2")][i] for i in range(21))
    if bad: os.rename(f"{ENG}/runs/{new}", f"{H}/th2/NOT_REPRODUCED_{new}"); continue
    open(f"{ENG}/runs/{new}/log.txt", "w").write(body + f"wall all 21 markets {time.time() - T:.0f} s (6 workers, both passes)\ncmd: {src_cmd(run)} --v2c-th 2{note}\n")
print(f"done {time.time() - T:.0f} s", flush=True)
