"""round_r7 aave (26 Oct 2026): the Aave v3 / v4 equivalent page runs on Aave's real rules (owner: "Fix the Aave bars", after the audit
v2/round_r7_audit/C_aave). Binary bin/replay_sim_x_r7a = round_r6 + the --aave-lb / --aave-fee / --aave-slice / --aave-v33 / --aave-eqcap flags
(source header). Each market's command = run_v5s.sh's expansion of the aave*_r6 run's "cmd:" line (env ORA_SERIES / WO_POOL / WO_SWAPS),
market 3 (CRV) on the venue grid as round_r6. Pass 1 (repro): no new flag, every w/l/b file must equal aave*_r6's byte for byte, else no run.
Pass 2 -> engine/runs/aave1_r7, aave4_r7 with
  both: --aave-lb (the collateral's Aave bonus) --aave-fee (Aave's liquidation protocol fee) --aave-slice 1 --aave-eqcap 1
  v3:   --aave-v33 1; bonus / fee = Aave v3 Ethereum Core on chain (AaveProtocolDataProvider, block 26,276,527: C_aave/src/v3_onchain_params.txt)
  v4:   --thf 1.24 --hf-maxb 0.90 --lb-factor 0.90 = the live v4 Main Spoke's getLiquidationConfig (C_aave/src/v4_onchain_spokes.txt);
        max bonus = the Main Spoke's maxLiquidationBonus (C_aave/src/v4_mainspoke_reserves.txt), liquidation fee 10 %
  Collateral Aave does not list: proxies (v3: sfrxETH -> wstETH, ynETH / ynETHx / pufETH / ezETH -> weETH; v4 also tBTC / LBTC -> WBTC,
  CRV -> 1.111 x its v3 bonus, the Main Spoke's step from v3 for WETH / wstETH / weETH / WBTC). The liquidation line stays the LlamaLend
  market's own (--aave-lt-liq 1): Aave's thresholds (WETH 83 %, CRV 41 %) would make these loans liquidatable the minute they were opened.
usage: nice -n 19 python3 run_aave_r7.py"""
import os, re, shlex, subprocess, filecmp, time
from concurrent.futures import ThreadPoolExecutor, as_completed
LD = os.environ.get("LLAMMA_DESIGN", "."); ENG = f"{LD}/engine"; H = f"{LD}/v2/round_r7_aave/engine"
BIN = f"{H}/bin/replay_sim_x_r7a"; GRID = f"{LD}/v2/round_r6/engine/venue_crv.txt"
COLL = {0: "wstETH", 1: "WETH", 2: "tBTC", 3: "CRV", 4: "WBTC", 5: "pufETH", 6: "WETH", 7: "wstETH", 8: "ezETH", 9: "ynETH", 10: "ynETHx", 11: "tBTC",
        12: "LBTC", 13: "cbBTC", 14: "weETH", 15: "tBTC", 16: "sfrxETH", 17: "WETH", 18: "WBTC", 19: "wstETH", 20: "sfrxETH"}   # r10/onchain_markets.json
V3 = {"WETH": (0.05, 0.20), "wstETH": (0.06, 0.20), "WBTC": (0.05, 0.20), "cbBTC": (0.075, 0.10), "tBTC": (0.075, 0.10), "LBTC": (0.085, 0.10),
      "CRV": (0.083, 0.10), "weETH": (0.07, 0.10)}                                                       # (bonus, protocol fee), on chain
V3P = {"sfrxETH": "wstETH", "ynETH": "weETH", "ynETHx": "weETH", "pufETH": "weETH", "ezETH": "weETH"}     # not listed on v3: proxy
V4 = {"WETH": 0.0555, "wstETH": 0.0666, "weETH": 0.0777, "WBTC": 0.0555, "cbBTC": 0.0555}               # Main Spoke maxLiquidationBonus - 1
V4P = {"tBTC": "WBTC", "LBTC": "WBTC", "sfrxETH": "wstETH", "ynETH": "weETH", "ynETHx": "weETH", "pufETH": "weETH", "ezETH": "weETH"}
def v3(i): a = COLL[i]; return V3[V3P.get(a, a)]
def v4(i): a = COLL[i]; return (round(V3["CRV"][0] * 1.111, 4) if a == "CRV" else V4[V4P.get(a, a)]), 0.10
RUNS = {"aave1_r6": ("aave1_r7", lambda i: ["--aave-lb", f"{v3(i)[0]:g}", "--aave-fee", f"{v3(i)[1]:g}", "--aave-slice", "1", "--aave-eqcap", "1", "--aave-v33", "1"]),
        "aave4_r6": ("aave4_r7", lambda i: ["--thf", "1.24", "--hf-maxb", "0.90", "--lb-factor", "0.90", "--aave-lb", f"{v4(i)[0]:g}", "--aave-fee", f"{v4(i)[1]:g}", "--aave-slice", "1", "--aave-eqcap", "1"])}
def src_cmd(run):
    return [l for l in open(f"{ENG}/runs/{run}/log.txt") if l.startswith("cmd:")][-1][4:].strip().split(" ; ROUND_R6")[0]
def margs(run, i, outdir):
    cmd = src_cmd(run); env = dict(re.findall(r"(\w+)=(\S*)", cmd.split(" run_v")[0])); rest = shlex.split(cmd.split(".sh ", 1)[1])[1:]
    und = int(open(f"{ENG}/mkt_{i}.txt").readline().split()[2])
    c = {0: "../candles_0.bin", 1: "../candles_1.bin", 2: "../candles_2_onchain.bin"}[und]; dep = "2e7" if und == 2 else "2e8"
    x = []
    if env.get("ORA_SERIES"): x += ["--oracle-series", env["ORA_SERIES"] % und]
    if env.get("WO_POOL"): x += ["--wo-pool", env["WO_POOL"] % und]
    if env.get("WO_SWAPS"): x += ["--wo-swaps", env["WO_SWAPS"] % i]
    a = ["--in", f"mkt_{i}.txt", "--candles", c, "--depth", dep, "--out", f"{outdir}/w_{i}.txt", "--liq", f"{outdir}/l_{i}.txt", "--bd", f"{outdir}/b_{i}.txt"] + x + rest
    return a + (["--venue-grid", GRID] if i == 3 else [])
def one(run, new, tag, i):
    d = f"{H}/repro/{new}" if tag == "repro" else f"{ENG}/runs/{new}"
    a = margs(run, i, d) + (RUNS[run][1](i) if tag == "new" else []); t = time.time()
    r = subprocess.run(["nice", "-n", "19", BIN] + a, cwd=ENG, capture_output=True, text=True)
    return run, new, tag, i, r.returncode, r.stderr + f"time mkt_{i} {time.time() - t:.0f} s rc {r.returncode}\n"
size = {i: int(open(f"{ENG}/mkt_{i}.txt").readline().split()[6]) for i in range(21)}
jobs = []
for run, (new, _) in RUNS.items():
    assert not os.path.exists(f"{ENG}/runs/{new}"), new
    os.makedirs(f"{H}/repro/{new}", exist_ok=True); os.makedirs(f"{ENG}/runs/{new}")
    jobs += [(run, new, tag, i) for tag in ("repro", "new") for i in range(21)]
jobs.sort(key=lambda j: (-size[j[3]], j[2] != "repro"))
print(f"{len(jobs)} market runs", flush=True); T = time.time(); logs = {}
with ThreadPoolExecutor(6) as P:
    for f in as_completed([P.submit(one, *j) for j in jobs]):
        run, new, tag, i, rc, err = f.result(); assert rc == 0, (run, tag, i, err[-400:])
        logs.setdefault((new, tag), {})[i] = err
for run, (new, fl) in RUNS.items():
    bad = [f"{k}_{i}" for i in range(21) for k in "wlb" if not filecmp.cmp(f"{H}/repro/{new}/{k}_{i}.txt", f"{ENG}/runs/{run}/{k}_{i}.txt", shallow=False)]
    print(f"{run:10s} repro: {'byte for byte' if not bad else 'DIFFERS ' + ' '.join(bad)}", flush=True)
    if bad: os.rename(f"{ENG}/runs/{new}", f"{H}/NOT_REPRODUCED_{new}"); continue
    note = (f" ; ROUND_R7 AAVE: all 21 markets on v2/round_r7_aave/engine/bin/replay_sim_x_r7a (CRV on the venue grid), per market + "
            + " | ".join(f"m{i}: {' '.join(fl(i))}" for i in range(21)) + f"; without the new flags every file equals {run} (run_aave_r7.py)")
    body = "".join(logs[(new, "new")][i] for i in range(21))
    open(f"{ENG}/runs/{new}/log.txt", "w").write(body + f"wall all 21 markets {time.time() - T:.0f} s (6 workers, both passes)\ncmd: {src_cmd(run)} {' '.join(fl(1))}{note}\n")
print(f"done {time.time() - T:.0f} s", flush=True)
