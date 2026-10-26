"""research/concept_bd (25 Oct 2026): TriCRV's real on-chain sale quotes for CRV -> crvUSD over the CRV LlamaLend market's life, the
liquidators' (and arbitrageurs') venue in the bad-debt sim. At each sample block: price_oracle(1) (CRV in crvUSD), the marginal
price (get_dy of 1 CRV), and get_dy(2, 0, q) for q worth $1k ... $10M at the oracle price -> cost fraction 1 - dy / (q x mid).
Samples: daily 12:00 UTC from 11 Mar 2024 to 28 Sep 2026; every 10 min on 12-14 Jun 2024 and 10-11 Oct 2025. Block = arithmetic
from a measured anchor (post-merge 12 s slots; missed slots shift a daily sample by minutes). Endpoint: curve-sim-ui .env
WEB3_HTTP_MAINNET (never printed). usage: python3 fetch_tricrv_quotes.py -> tricrv_quotes.json"""
import json, os, urllib.request, datetime as dt
from eth_utils import keccak
env = dict(l.strip().split("=", 1) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env")) if "=" in l and not l.startswith("#"))
url = env["WEB3_HTTP_MAINNET"].strip().strip('"').strip("'")
TRICRV = "0x4ebdf703948ddcea3b11f675b4d1fba9d2414a14"
SIZES = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]
B_REF, T_REF = 23_549_898, 1760130323   # measured (chain_1010.json): block 23,549,898 at this timestamp
sel = lambda s: "0x" + keccak(text=s)[:4].hex()
def batch(reqs):
    out = []
    for k in range(0, len(reqs), 50):
        part = [dict(jsonrpc="2.0", id=i, method=m, params=p) for i, (m, p) in enumerate(reqs[k:k + 50])]
        r = sorted(json.loads(urllib.request.urlopen(urllib.request.Request(url, json.dumps(part).encode(), {"content-type": "application/json"}), timeout=120).read()), key=lambda x: x["id"])
        out += [x.get("result") for x in r]
    return out
ts = []
d = dt.datetime(2024, 3, 11, 12, tzinfo=dt.UTC)
while d <= dt.datetime(2026, 9, 28, 12, tzinfo=dt.UTC):
    ts.append(int(d.timestamp())); d += dt.timedelta(days=1)
for a, b in ((dt.datetime(2024, 6, 12, tzinfo=dt.UTC), dt.datetime(2024, 6, 15, tzinfo=dt.UTC)), (dt.datetime(2025, 10, 10, tzinfo=dt.UTC), dt.datetime(2025, 10, 12, tzinfo=dt.UTC))):
    t = int(a.timestamp())
    while t < int(b.timestamp()):
        ts.append(t); t += 600
ts = sorted(set(ts)); blocks = [B_REF + (t - T_REF) // 12 for t in ts]
call = lambda data, b: ("eth_call", [{"to": TRICRV, "data": data}, hex(b)])
po = [int(x, 16) if x else 0 for x in batch([call(sel("price_oracle(uint256)") + format(1, "064x"), b) for b in blocks])]
gd = lambda q: sel("get_dy(uint256,uint256,uint256)") + format(2, "064x") + format(0, "064x") + format(int(q), "064x")
mids = [int(x, 16) if x else 0 for x in batch([call(gd(10**18), b) for b in blocks])]
reqs = [call(gd(s / (p / 1e18) * 1e18), b) for b, p in zip(blocks, po) for s in SIZES] if all(po) else None
assert reqs, "a price_oracle read failed"
res = batch(reqs); rows = []
for k, (t, b, p, m) in enumerate(zip(ts, blocks, po, mids)):
    cost = []
    for j, s in enumerate(SIZES):
        x = res[k * len(SIZES) + j]; q = s / (p / 1e18)
        cost.append(None if not x else 1 - (int(x, 16) / 1e18) / (q * m / 1e18))
    rows.append(dict(t=t, block=b, p_oracle=p / 1e18, mid=m / 1e18, cost=cost))
json.dump(dict(sizes_usd=SIZES, rows=rows), open("tricrv_quotes.json", "w"))
for r in rows[::90] + [r for r in rows if r["t"] in (1760131380,)]:
    print(dt.datetime.fromtimestamp(r["t"], dt.UTC).strftime("%Y-%m-%d %H:%M"), f"p ${r['p_oracle']:.4f}", "cost at $100k / $1M:", None if r["cost"][4] is None else f"{100 * r['cost'][4]:.2f}%", None if r["cost"][6] is None else f"{100 * r['cost'][6]:.2f}%")
print("samples", len(rows), "failed quotes", sum(c is None for r in rows for c in r["cost"]))
