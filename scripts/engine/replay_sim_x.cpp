#include <array>
// round_r7 aave (26 Oct 2026): the Aave-equivalent runs on Aave's real rules (audit v2/round_r7_audit/C_aave). New flags, all off = bit-identical to round_r6:
//   --aave-lb X     the market's own Aave liquidation bonus (0.05 = 5 %; v4: the max bonus) instead of the per-underlying 5 / 5 / 8.3 %
//   --aave-fee F    Aave's liquidation protocol fee, the share of the bonus the treasury takes: the borrower still pays the whole bonus,
//                   the liquidator's profit check and sale use R x (1 + bonus x (1 - F))
//   --aave-slice 1  the liquidator picks debtToCover <= the maximum (profit-maximising slice on the venue, best_slice), as Aave allows
//   --aave-v33 1    Aave v3.3 rules in --aave 1: close factor 100 % when the debt or the collateral is below $2,000, and a partial
//                   liquidation must leave at least $1,000 (P.dust) of debt and of collateral (v4's dust rule is always on)
//   --aave-eqcap 1  borrower loss of a liquidation = min(bonus, the borrower's equity before it): a bonus beyond the equity is paid by
//                   the lenders and already sits in the written-off bad debt; a dust write-off is never negative
// round_r6 (25 Oct 2026): --venue-grid FILE = the market's real on-chain sale quotes over time (curve-sim-ui/research/concept_bd/
// fetch_tricrv_quotes.py -> round_r6/engine/venue_crv.txt: line 1 "K s_1..s_K" USD sizes, then "t c_1..c_K" = 1 - get_dy / (q x mid)
// of TriCRV at that block). With it the liquidators' sale cost is the measured cost curve C(q) = q c(q) (log-interpolated; the pool fee
// is inside the quote, so ext_fee is not added), drained by the minute's earlier sales (C(Qsell + q) - C(Qsell)), the liquidator's
// slice maximises its profit on that curve, and the arbitrage flow moves the market by the curve's slope dC/dq instead of q / depth.
// Without the flag the engine is bit-identical to round_r4 replay_sim_x_r4b.
// replay_sim.cpp — one continuous run per market (3 years, 1-minute steps) with every real borrower
// action replayed at its minute: loans are opened, topped up, repaid, deleveraged and closed exactly
// when the real borrowers did it. Nothing resets weekly, so a loan the real borrower rescued is
// rescued here too, and a loan the sim liquidates stays liquidated.
//
// Band math, oracle variants, the shared DEX venue and the liquidation rules are speed_sim.cpp's.
// Differences:
//   * band prices and debts grow with the market's rate (AMM base price, interpolated between Mondays)
//   * oracle = variant filter on the underlying's exchange mid x the collateral/underlying ratio
//   * actions (replay/prep_replay.py): 1 create, 2 adjust, 3 repay while underwater, 4 full repay,
//     5 self-liquidation, 6 third-party liquidation (a real OUTCOME, ignored; the sim decides itself)
//   * when the sim state forbids a real action (contract would revert: borrow / add / remove while in
//     soft liquidation) the add is turned into a repay of the same value, borrow and remove are dropped
//   * a loan the real world closed (liquidation, self-liquidation of a loan that is insolvent here)
//     but the sim did not is an "orphan": it lives on without borrower actions; with --orphan 1 the
//     borrower closes it when they open their next real loan, if it is healthy then
// Outputs: --out  weekly  "mkt t0 sl dl liq_eq n_liq bd_peak bd_end steps n_alive d_sum v_sum real_d n_orph
//                          bd_real n_bd_real fee_usd vol_usd sold_sl rng_lmin sl_lmin rng_dmin sl_dmin" (23 columns)
//          --liq  "S t uid debt value equity orphan Y X n1 n2 oracle h_oracle h_spot oracle spot" per sim hard
//                 liquidation (LLAMMA; partial: debt = repaid, value = seized, equity = bonus). Columns 13-15 (1-based,
//                 LLAMMA-Design 14 Oct 2026): h_spot = the loan's contract health with the exchange spot (this minute's
//                 Binance close x ratio) in place of the oracle, computed on the loan state just before the
//                 liquidation; oracle = the oracle used (same as column 12); spot = close x ratio. A liquidation with
//                 h_oracle < 0 and h_spot >= 0 is a false one (the oracle, not the market, caused it).
//                 Aave mode (--aave) prints the 7-column "S t uid debt value equity orphan" row unchanged.
//                 "R t uid sim_alive sim_health sim_equity debt" per real third-party liquidation
//          --bd   hourly "t bd_max" (only non-zero hours)
//          --exposure 1 adds to --liq, once per week (the week's first minute with the book priced at that minute's oracle):
//                 "X t d0 g0 d1 g1 ... d10 g10" (24 fields incl. 'X'; 14 before round 3): t = the minute's unix time; for each of the 11 oracle
//                 drops m = 0.25 0.5 1 2 3 5 7 10 15 20 30 % (round 3; the runs before 14 Oct 2026 16:30 carry the first 6):
//                 d = the debt ($) of the loans that drop would make liquidatable (health < 0; loans already liquidatable
//                 without an attack are skipped), g = the liquidators' gain ($) on them (partial mode: the repay R x (bonus-1);
//                 else collateral value - debt). The oracle itself is not moved: the snapshot is a what-if on the live book.
//
// External oracle (LLAMMA-Design, 14 Oct 2026):
//   --oracle-series FILE  float64, one value per row of the candle file (same row count, checked), the oracle ON THE
//                         UNDERLYING in USD (ETH, BTC or CRV; NOT multiplied by the market's collateral/underlying
//                         ratio). In a minute with a finite value > 0 it replaces nw_, the engine's own filtered
//                         price before the ratio (the EMA / adaptive filter and --cap are skipped that minute);
//                         NaN (or <= 0) = the engine's own oracle that minute, continuing from the last value.
//                         Everything after nw_ is unchanged: LLAMMA's limit_p_o (+-25 % per step), the dynamic fee
//                         from r, the ratio, the bands. The week-0 book is still priced with the minute-i0 mid.
//   --oracle-dump FILE    writes the engine's own per-minute oracle (the same quantity: the value nw_ that becomes
//                         ema_u after the limit, before the ratio) as float64, one per candle row, NaN outside the
//                         run's minutes [t_start, t_end). Feeding a dump back with --oracle-series reproduces the run.
//
// v2 concept parity (v4, 22 Oct 2026, LLAMMA-Design/v2/verify/engine; flags off = bit-exact v3 = replay_sim_x_v3):
//   --v2c 1    the rules of the v2 concept code (v2/concept): partial liquidation = Controller._calculate_debt_f_remove
//              (constants TARGET_HEALTH 1.4953, bonus 1.777 % -> 0.7429 % between health 0.9809 and 1 for EVERY market,
//              MIN_DEBT_LEFT 100; health without the full-health term; position valued at the AMM's p_o; no cap = the
//              liquidator repays the debt and gets the position in the same fraction, no bonus, no forced write-off);
//              one-way get_x_down keeps (x, y) when trading the band to p_o would buy y (needs --no-dl 1);
//              AMM fee capped at MAX_FEE = min(4 / A, 10 %) (v2 AMM constructor / _set_fee).
//   --exact 1  version-independent contract arithmetic: the AMM charges dx = x_dest / (1 - fee) (antifee), not
//              x_dest (1 + fee); a borrower's own partial self-liquidation withdraws f_remove = frac (health_limit 0).
//
// v5 (22 Oct 2026, LLAMMA-Design/v2/verify/engine2; --wo 0 = bit-exact v4): the concept oracle as the contract runs it
//   --wo 1 --wo-pool FILE   WindowOracle.vy (window_oracle.h) on the oracle pool's own price_oracle(), rebuilt from the pool's
//              block series (FILE = engine2/export_pool.py: the cb2 model's / today_pool's data, one file per underlying).
//              The minute's oracle = the contract's VIEW price() at the minute end (t + 60, the series' sampling time):
//              health, borrows, liquidations, the exposure snapshot and the AMM's p_o all read it. price_w() (a WRITE:
//              band + this block's observation) is called at that same time if the AMM traded in the minute (any loan's
//              x / y moved by a soft- or de-liquidation trade, or a replayed trade): LlamaLend v2 calls price_w() only in
//              AMM._exchange. In the block of a write price() = price_w(), so the write changes the later minutes only.
//              Deployed at the run's first minute end (>= the pool data's first block + 1 h, as the series' NaN rule; the
//              engine's own oracle before that). Replaces --oracle-series; limit_p_o / dynamic fee / ratio unchanged.
//   --wo 2     the same with the owner's unlock fix (22 Oct 2026, WindowOracle.vy _band: the observation before the window counts
//              only while the last write is < WINDOW old); --wo 1 = the code before the fix (v2/fork/oracle/WindowOracle.before_unlock_fix.vy)
//   --wo-window 36 --wo-tau 300 --wo-cap 0.003 --wo-stale 300 --wo-nobs 4   the contract's immutables (defaults = the concept)
//   --wo-log FILE  one line per write: "t crvUSD_volume_of_the_minute's_AMM_trades"
//   --wo-keep N    validation / sensitivity only (default 0 = off): a keeper also calls price_w() every N-th oracle minute
//                  (N = 1: a write every minute, the closest the 1-minute engine gets to the model's every-block oracle);
//                  keeper-only writes are not in --wo-log
//   every --wo run also reports the band at the minute reads (stderr 'band open': hi > lo, and wider than 1e-6 / 1e-4 / 1e-3 of p)
//   --wo-sub 1 --wo-swaps FILE   (23 Oct 2026, LLAMMA-Design/v2/fix/engine) the oracle on the block grid inside the minute:
//              FILE = float64 block timestamps (12 s slots, increasing) of the market's real AMM swaps (v2/verify/write_freq/
//              swaps.json via fix/engine/prep_swaps.py); each one in (t, t + 60] is a price_w() at its own block time, in time
//              order, before the minute-end view. The engine's own AMM trades write at the minute end as before; a second
//              write in the same block writes nothing (as the contract). Off (default) = bit-exact v5.
//   --gas-eth 1 --eth-candles FILE   (23 Oct 2026) the gas file is the base fee in gwei (liqtest/basefee_hourly.bin): a liquidation
//              costs gas_units x 1M gas x base fee x the ETH price (FILE = candles_0.bin, close of the last minute <= t);
//              off (default) = the file's value x gas_units, as before
//   --gas-eth 2 (23 Oct 2026, LLAMMA-Design/v2/redesign/base) the gas model measured from receipts (v2/fix/liq): a liquidation costs
//              gas_units x 1.07M gas x (base fee + 1 gwei tip) x the ETH price; give --gas-units 1. --gas-eth 1 unchanged.
//   --resync 2 (23 Oct 2026, LLAMMA-Design/v2/redesign/base) the real borrower's INTENT, never the chain's post-action STATE:
//              a real adjust (kind 2) is applied to the simulated loan as a delta (dcoll, ddebt) with the real band count N
//              (q.n2 - q.n1 + 1); outside soft liquidation the loan is re-placed by the simulated Controller's own
//              _calculate_debt_n1 (n1 formula at the simulated oracle and base, + the chain's constant formula offset q.n1 - q.nf,
//              capped at the simulated active band + 1024 - N); a borrow_more / remove_collateral whose new top band would sit
//              at or above the simulated oracle ("Debt too high") is cut to what the simulated Controller allows (the largest
//              borrow / smallest remaining collateral with the top band below the oracle; the real borrower's retry), and reverts
//              only when nothing is allowed. Inside soft liquidation (the simulated loan holds
//              converted crvUSD, x > 0) the action is what the contract does: add_collateral / borrow_more / remove_collateral
//              revert ("Underwater", counted as ignored, would revert), a repay reduces the debt and keeps the bands, a
//              deleverage (collateral out + debt down) is replayed as before. The chain's post-action collateral, debt and bands
//              are used only to re-take a loan the sim lost but the chain still has (as --resync 1). --resync 1 = the old rule
//              (the chain's state after every adjust; in_sl forced false); --resync 0 = no resync; both unchanged (bit-exact).
//              --liq: the resync-2 "I" row (sim loan in soft liquidation at a real adjust) carries a 6th field, the converted crvUSD / debt.
//   --shared-amm 5 (24 Oct 2026, LLAMMA-Design/v2/round_r1/engine; with --no-dl 1 --resync 2): DL off as the redesigned contracts run it, checked
//              against the compiled contracts by round_r1/engine/parity (peng.cpp + parity_dl.py). Mode 3's trade rule (only bands the oracle has
//              reached are sold) plus: g_act_m = the market's AMM.active_band() (deposits move it up past skipped bands, sales move it to the
//              deepest band they reached); health(full) adds the above-bands term only when n1 > g_act_m (--full-term 1, the 23 Oct contract)
//              and the loan holds no crvUSD (--full-term 2, default, the round r1 fix); new loans at the simulated Controller's n1 (formula at the
//              simulated oracle + the chain's offset), cut to the largest debt with the top band below the oracle ("Debt too high"), then below
//              any band holding crvUSD ("Band not empty"; the debt scaled by k per band); a repay of a loan at or above the active band keeps its
//              bands (_repay_partial); a loan holding crvUSD: add / borrow / remove = its crvUSD repays debt and the collateral is re-placed at
//              the simulated oracle, with the same two checks (--sl-reopen 1, default in mode 5; --sl-reopen 0 = the refusal of --resync 2).
//              Modes 0-4 and every other flag unchanged (bit-exact).
//   --arb-gas G (24 Oct 2026, diagnosis only, default 0 = off): the minute's AMM sale is skipped unless the arbitrageur's gain covers G M gas.
#include "window_oracle.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Candles { std::vector<double> t, h, l, c; };
static Candles load_candles(const char *path) {
    Candles C; FILE *f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "no candles %s\n", path); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f) / 32; std::fseek(f, 0, SEEK_SET);
    std::vector<double> b(4 * n); if (std::fread(b.data(), 8, 4 * n, f) != size_t(4 * n)) std::exit(2); std::fclose(f);
    C.t.resize(n); C.h.resize(n); C.l.resize(n); C.c.resize(n);
    for (long i = 0; i < n; i++) { C.t[i] = b[4*i]; C.h[i] = b[4*i+1]; C.l[i] = b[4*i+2]; C.c[i] = b[4*i+3]; }
    return C;
}

static double g_base = 1.0;
// fee accounting for real trades only (the venue bisection trades copies): fee kept in bands, crvUSD volume
static bool g_acc = false;
static int g_no_dl = 0;                   // --no-dl 1: one-way AMM, crvUSD in a band below the oracle stays crvUSD and counts at face in the health
static int g_m7 = 0;                        // --shared-amm 7 (round r3 route R2, 24 Oct 2026): mode 5, every placement accepted; a deposit into a band moves the
                                          // other loans' crvUSD in that band out of the band as their credit (Loan::xc): the band trades y only from its top
static bool g7_dirty = false;            // --m7v 3: a holder of a band with phantom x left / shrank: the band's slices are re-cut before the next trade
static int g_shared = 0;                  // --shared-amm 1|2|3 (23 Oct 2026, fix dloff / redesign dloff; with --no-dl 1): ONE AMM per market for placements: 1 = AMM.vy as now
                                          // (no skip over a band with crvUSD), 2 = the fix (bands with crvUSD only are skipped; x+y never; no x band inside a new range)
                                          // 5 (round r1, 24 Oct 2026): mode 3 as the contracts run it: the market's own active band (g_act_m) in health(full),
                                          // new loans at the simulated Controller's n1 ("Debt too high" at the simulated oracle, "Band not empty" at any x),
                                          // a loan holding crvUSD: add / borrow / remove apply its crvUSD to the debt and re-place the collateral (round r1 fixed
                                          // Controller._add_collateral_borrow; on the 23 Oct contract the same arithmetic is the borrower's full repay + create_loan:
                                          // --sl-reopen, on by default here, --sl-reopen 0 = the old refusal)
static int g_dlc = 0;                     // --dl-checks 1 (round r3b, 24 Oct 2026): the DL-on path (base AMM) gets the Controller checks of mode 5 / 7 (create n1 at the
                                          // simulated oracle + 'Debt too high' cut, repay refused unhealthy, repay covered by the bands' crvUSD = full repay, _repay_partial
                                          // keeps the bands when n1 <= the AMM's active band); off = bit-exact
static int g_ev = 0;                      // --ev 1 (round r3b): one 'EV' line on stderr per action cut or refused: mkt, action index, t, uid, kind, reason, asked, got
static int g_act_m = 0;                   // --shared-amm 5: AMM.active_band() of the market (deposits move it up past bands with crvUSD, sales move it down)
static int g_ft = 2;                      // --full-term (--shared-amm 5): 2 (default) = round r1 fixed Controller._health: the above-bands term only while the loan holds no x;
                                          // 1 = the contract of 23 Oct (the term whenever n1 > AMM.active_band(): audit3 F1, a loan placed above makes a converted loan unliquidatable)
static std::vector<double> g_vg_s, g_vg_t; static std::vector<std::vector<double>> g_vg_c;   // --venue-grid (round_r6)
static double g_th = 1.5, g_mbh = 0.95, g_maxb = 0.015, g_minb = 0.005, g_mdl = 0.0;   // 23 Oct 2026: the concept's own constants (was Aave v4's 1.4953 / 0.9809 / 1.777 % / 0.7429 % and a 100-coin dust rule)   // --v2c-th / -mbh / -maxb / -minb / -mdl (23 Oct 2026): the v2 concept's liquidation constants; --v2c-mdl 0 = no dust rule
static int g_v2c = 0;                     // --v2c 1 (v4, 22 Oct 2026): the v2 concept code's rules (one-way get_x_down in-band clamp, partial liquidation
                                          //   = Controller._calculate_debt_f_remove, AMM MAX_FEE); see the header note 'v2 concept parity'
static int g_exact = 0;                   // --exact 1 (v4): AMM fee charged as the contracts do (dx = x_dest / (1 - fee)), self-liquidation f_remove = frac
static int g_dyn_fee = 0;                 // --dyn-fee 1: the AMM's per-band dynamic fee, max(fee, |band price - oracle| x g_dyn_frac) (AMM.get_dynamic_fee)
static double g_dyn_frac = 0.25;          // --dyn-frac F: the share of the band's distance from the oracle charged as fee (AMM: 1/4)
static double g_fee_usd = 0, g_vol_usd = 0;          // AMM base price now (band prices and debts scale with it)

static int g_cdec = 18;
static inline double trc(double y) { if (g_cdec >= 18 || y <= 0) return y; const double u = std::pow(10.0, g_cdec); return std::floor(y * u * (1.0 + 1e-15)) / u; }
static int g_m7v = 1;                      // round_r4/engine: 4 = 3 + Loan::x_zero() in health(full) (the Controller's x == 0 at float resolution)
struct Loan {
    double A, k, sbr;
    int lo, hi, n1, n2, active;
    std::vector<double> bx, by, kp;   // kp[n - lo] = k^n
    double Dn = 0;                    // debt / base  (debt grows with the rate like the base price)
    double X = 0, Y = 0;
    double ycr = 0;                   // --m7v 2: collateral which left the loan's bands at a chain_end (counted in Y, in x_down at p_oracle)
    std::vector<double> ph, cr;        // --m7v 3: per band n1..n2: the phantom part of the slice's x (credited x kept for the curve), the loan's credit kept in the band
    std::vector<unsigned __int128> sh; // --m7v 2: the loan's shares per band n1..n2 (wei, v2's formula)
    double xc = 0;                    // --shared-amm 7: crvUSD which left the loan's bands at a deposit into them (counted in X, at face)
    double Ya = 0;                    // Aave mode: collateral tokens (debt = Dn * base as well)
    bool touched = false, alive = true, orphan = false;
    int uid = -1;
    double ld = 0;                    // this loan's liquidation discount (Controller.liquidation_discounts[user])
    long neg_since = -1;
    double sl_acc = 0;               // soft + de-liquidation loss booked on this loan so far ($)
    double sold_u = 0, sold_x = 0;   // net collateral sold by the AMM (units) and the crvUSD it got for it: the position vs holding
    long cov_since = -1;             // step since which the loan's crvUSD covers its debt
    // soft-liquidation statistics: minutes with the oracle inside/below the bands (should soft-liquidate), minutes with
    // collateral actually converted (x > 0), max converted share, converted share at the first hard liquidation
    long rng_min = 0, sl_min = 0, first_rng = -1; double maxconv = 0, conv_liq = -1, d_first = 0; int nliq_l = 0;
    double xdr = 0;                   // x_down / base while the oracle is above every band
    double p_o = 0, po3 = 0;
    int traded_n = -(1 << 30);        // --shared-amm 5: the deepest band a sale changed (AMM.active_band after the exchange)

    double D() const { return Dn * g_base; }
    double &x(int n) { return bx[n - lo]; }
    double &y(int n) { return by[n - lo]; }
    double pt(int n) const { return g_base * kp[n - lo]; }
    double p_down(int n) const { double pb = pt(n); return po3 / (pb * pb); }
    double p_up(int n) const { double pb = pt(n + 1); return po3 / (pb * pb); }
    bool in_range(int n) const { return n >= lo && n + 1 <= hi; }
    double get_y0(int n) {
        double xx = x(n), yy = y(n), ptn = pt(n);
        double b = ptn / p_o * (A - 1.0) * xx + p_o * p_o / ptn * A * yy;
        double a = p_o * A;
        return (b + std::sqrt(b * b + 4.0 * a * xx * yy)) / (2.0 * a);
    }
    double get_f(double y0, int n) { return y0 * p_o * p_o / pt(n) * A; }
    double get_g(double y0, int n) { return y0 * pt(n) / p_o * (A - 1.0); }
    double get_p() {
        int n = active; double xx = x(n), yy = y(n);
        if (xx == 0.0 && yy == 0.0) return std::sqrt(p_up(n) * p_down(n));
        double y0 = get_y0(n);
        return (get_f(y0, n) + xx) / (get_g(y0, n) + yy);
    }
    void trade_to_price(double price, double fe) {
        const double fe_base = fe;
        int bstep;
        if (x(active) == 0.0 && y(active) == 0.0) {
            if (price > p_up(active)) bstep = 1; else if (price < p_down(active)) bstep = -1; else return;
        } else {
            double cur = get_p();
            if (price > cur) bstep = 1; else if (price < cur) bstep = -1; else return;
            // an arbitrageur trades only if the fee-adjusted target lies beyond the AMM price; inside the fee
            // band the reference model would trade the wrong way (buying back / selling at a loss, paying the fee)
            if (bstep == 1 && price * (1.0 - fe) <= cur) return;
            if (bstep == -1 && price * (1.0 + fe) >= cur) return;
        }
        double original = price;
        if (g_m7 && bstep == -1) return;                     // mode 7: the one-way AMM never buys collateral back (a band set to x = 0 by a deposit sits above its price)
        if ((g_shared == 3 || g_shared == 5) && bstep == 1 && pt(active) < p_o) return;   // --shared-amm 3: p_oracle has not reached the active band
        while (true) {
            int n = active; double xx = x(n), yy = y(n);
            price = original;
            if (xx == 0.0 && yy == 0.0) {
                if (price >= p_down(n) && price <= p_up(n)) break;
                int nn = active + bstep;
                if (!in_range(nn)) break;
                if ((g_shared == 3 || g_shared == 5) && bstep == 1 && pt(nn) < p_o) break;   // one-way: no band below p_oracle is sold
                active = nn; continue;
            }
            double y0 = get_y0(n), g = get_g(y0, n), f = get_f(y0, n);
            double Inv = (f + xx) * (g + yy);
            double p_c_d = p_down(n), p_c_u = p_up(n);
            if (g_dyn_fee) {                                       // AMM.get_dynamic_fee: a quarter of the band's distance from the oracle
                double dfn = p_o < p_c_d ? (p_c_d - p_o) * g_dyn_frac / p_c_d : (p_o > p_c_u ? (p_o - p_c_u) * g_dyn_frac / p_o : 0.0);
                fe = std::max(fe_base, dfn);
            }
            const double fm = g_exact ? fe / (1.0 - fe) : fe;      // AMM: antifee = 1 / (1 - fee); fm == fe keeps the old arithmetic bit for bit
            if (bstep == 1) {
                price = price * (1.0 - fe);
                if (price < p_c_d) break;
                double y_dest = std::sqrt(Inv / price) - g, x_old = xx;
                if (y_dest >= 0.0) { y(n) = y_dest; x(n) = Inv / (g + y_dest) - f; if (g_acc) { g_fee_usd += fm * (x(n) - x_old); g_vol_usd += x(n) - x_old; } x(n) += fm * (x(n) - x_old); traded_n = n; break; }
                traded_n = n;
                y(n) = 0.0; x(n) = Inv / g - f; if (g_acc) { g_fee_usd += fm * (x(n) - x_old); g_vol_usd += x(n) - x_old; } x(n) += fm * (x(n) - x_old);
                if (!in_range(active + 1)) break;
                if ((g_shared == 3 || g_shared == 5) && pt(active + 1) < p_o) break;            // one-way: no band below p_oracle is sold
                active += 1;
            } else {
                price = price * (1.0 + fe);
                if (price > p_c_u) break;
                double x_dest = std::sqrt(Inv * price) - f, y_old = yy;
                if (x_dest >= 0.0) { x(n) = x_dest; y(n) = Inv / (f + x_dest) - g; if (g_acc) { g_fee_usd += fm * (y(n) - y_old) * original; g_vol_usd += xx - x_dest; } y(n) += fm * (y(n) - y_old); break; }
                x(n) = 0.0; y(n) = Inv / f - g; if (g_acc) { g_fee_usd += fm * (y(n) - y_old) * original; g_vol_usd += xx; } y(n) += fm * (y(n) - y_old);
                if (!in_range(active - 1)) break;
                active -= 1;
            }
        }
    }
    double get_x_down(int n) {
        if (!ph.empty() && ph[n - n1] != 0.0) {                // --m7v 3: AMM.get_xy_up: the pooled curve's x_o minus xc (pro rata), never below 0
            const double v = get_x_down_c(n) - ph[n - n1];
            return v > 0.0 ? v : 0.0;
        }
        return get_x_down_c(n);
    }
    double get_x_down_c(int n) {
        double xx = x(n), yy = y(n);
        if (xx == 0.0 && yy == 0.0) return 0.0;
        double p_o_up = pt(n), p_o_down = p_o_up * k;
        if (g_no_dl && p_o > p_o_up) return xx + yy * p_o_up / sbr;   // nothing is bought back: the crvUSD is worth its face, the collateral its band price
        double p_mid = p_o * p_o * p_o / (p_o_down * p_o_down) * k;
        if (xx == 0.0 || yy == 0.0) {
            if (p_o > p_o_up) { double y_eq = (yy == 0.0) ? xx / p_mid : yy; return y_eq * p_o_up / sbr; }
            else if (p_o < p_o_down) { double x_eq = (xx == 0.0) ? yy * p_mid : xx; return x_eq; }
        }
        double y0 = get_y0(n), g = get_g(y0, n), f = get_f(y0, n);
        double Inv = (f + xx) * (g + yy);
        if (p_o > p_o_up) { double y_o = std::max(Inv / f, g) - g; return y_o * p_o_up / sbr; }
        else if (p_o < p_o_down) { double x_o = std::max(Inv / g, f) - f; return x_o; }
        double y_o = A * y0 * (1.0 - p_o_down / p_o);
        double x_o = std::max(Inv / (g + y_o), f) - f;
        if (g_v2c && g_no_dl && x_o < xx) return xx + yy * std::sqrt(p_o_down * p_o);   // v2 concept AMM.get_xy_up: trading to p_o would buy y back; keep (x, y)
        return x_o + y_o * std::sqrt(p_o_down * p_o);
    }
    // --m7v 4 (round_r4/engine, 24 Oct 2026): Controller._health's 'xy[0] == 0' (AMM.get_sum_xy x, wei) at float resolution: X = xc + sum x - sum ph
    // is a sum of terms that cancel exactly in the contract; the float residue (|X| ~1e-13 on terms ~1e3, either sign) counts as 0 (round_r3c G4 N50, step 136.5 uid 69)
    bool x_zero() const { double m = std::fabs(xc); for (int n = n1; n <= n2; n++) m += std::fabs(bx[n - lo]); for (double v : ph) m += std::fabs(v); return std::fabs(X) <= 1e-12 * m; }
    void totals() { X = xc; Y = ycr; for (int n = n1; n <= n2; n++) { X += x(n); Y += y(n); } if (!ph.empty()) for (double v : ph) X -= v; }
    void scale(double f) { for (int n = n1; n <= n2; n++) { x(n) *= f; y(n) *= f; } if (!sh.empty()) for (auto &v : sh) v -= (unsigned __int128)((long double)v * (1.0L - (long double)f)); ycr *= f; xc *= f; for (auto &v : ph) v *= f; for (auto &v : cr) v *= f; if (!ph.empty()) g7_dirty = true; xdr *= f; sold_u *= f; sold_x *= f; totals(); }   // pro-rata slice
    double x_down_sum() {                 // AMM.get_x_down(user) (no liquidation discount, no full-health term)
        if (!touched && p_o > pt(n1)) return xdr * g_base;
        double xd = xc + ycr * p_o; for (int n = n1; n <= n2; n++) xd += get_x_down(n);
        return xd;
    }
    double health() {
        double xd, d = D();
        if (!touched && p_o > pt(n1)) xd = xdr * g_base;
        else { xd = xc + ycr * p_o; for (int n = n1; n <= n2; n++) xd += get_x_down(n); }
        double h = xd * (1.0 - ld) / d - 1.0;
        if (n1 > (g_shared == 5 ? g_act_m : active) && (g_shared != 5 || g_ft == 1 || X <= 0.0 || (g_m7v >= 4 && x_zero()))) { double pu = pt(n1); if (p_o > pu) h += (p_o - pu) * trc(Y) / d; }   // round r3c G4: the Controller's full term takes y from get_sum_xy (token units); mode 5: the market's active band
        return h;
    }
    double health_fast(double ema) {
        if (!touched && ema > pt(n1) * 1.001) return xdr * g_base * (1.0 - ld) / D() - 1.0 + (ema - pt(n1)) * trc(Y) / D();
        return health();
    }
    // deposit y collateral evenly over [a, b] (x = 0), market active band n_act
    void place(double yc, int a, int b, int n_act) {
        n1 = a; n2 = b;
        int na = std::max(n1 - 60, std::min(n_act, n2 + 60));
        lo = std::min(n1, na) - 3; hi = std::max(n2, na) + 4;
        int span = hi - lo + 1;
        bx.assign(span, 0.0); by.assign(span, 0.0); kp.resize(span + 1);
        for (int n = lo; n <= hi + 1; n++) kp[n - lo] = std::pow(k, n);
        int N = n2 - n1 + 1;
        for (int n = n1; n <= n2; n++) y(n) = yc / N;
        active = std::max(lo, std::min(na, hi - 1));
        if (!ph.empty()) g7_dirty = true;
        touched = false; neg_since = -1; xc = 0; ycr = 0; sh.clear(); ph.clear(); cr.clear();
        totals();
        xdr = 0; for (int n = n1; n <= n2; n++) xdr += y(n) * kp[n - lo] / sbr;
    }
};

// --m7v 2 (round r3b, 24 Oct 2026): the AMM's per-band share ledger in wei. U256 mulDiv for v2's ds = (s + DEAD_SHARES) * y // (total_y + 1).
static std::string g_ason;              // round r3c G5(a): --as-on FILE                 // round r3c G4: collateral decimals; < 18: a re-placement re-deposits AMM.withdraw's truncated y
typedef unsigned __int128 u128;
static long g7_close = 0, g7_chain = 0, g7_into = 0; static double g7_xcredit = 0, g7_ycredit = 0;
static u128 wei18(double v) { return v > 0 ? (u128)((long double)v * 1e18L) : 0; }
static u128 muldiv(u128 a, u128 b, u128 d) {               // floor(a * b / d), a * b < 2**256, result < 2**128
    const u128 M = ~(u128)0 >> 64; u128 a0 = a & M, a1 = a >> 64, b0 = b & M, b1 = b >> 64;
    u128 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    u128 mid = (p00 >> 64) + (p01 & M) + (p10 & M);
    u128 lo = (p00 & M) | (mid << 64), hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    if (hi == 0) return lo / d;
    u128 q = 0, r = 0;
    for (int i = 255; i >= 0; i--) {
        const bool top = r >> 127; r = (r << 1) | ((i >= 128 ? (hi >> (i - 128)) : (lo >> i)) & 1);
        if (top || r >= d) { r -= d; if (i < 128) q |= (u128)1 << i; }
    }
    return q;
}
static const u128 DEAD_SHARES_ = 1000, MAX_SHARES_PER_Y_ = 1000000000;
// AMM.deposit_range, the part before the new shares: for every band of [a, a + N - 1] held by others (H = the alive loans except the depositor):
// close the generation when the band holds x (in wei) or too little y for new shares; on a chain_end the holders' y leaves the band too.
static void m7_settle(std::vector<Loan*> &H, int a, int N) {
    std::vector<u128> S(N, 0), Yw(N, 0); std::vector<double> Xs(N, 0.0);
    for (Loan *o : H) for (int n = std::max(o->n1, a); n <= std::min(o->n2, a + N - 1); n++) {
        if (o->sh.empty()) continue;
        S[n - a] += o->sh[n - o->n1]; Yw[n - a] += wei18(o->y(n)); Xs[n - a] += o->x(n);
    }
    bool into = false;
    for (int j = 0; j < N; j++) {
        if (wei18(Xs[j]) != 0) into = true;
        if (S[j] == 0) continue;
        const bool chain = Yw[j] * MAX_SHARES_PER_Y_ < S[j] + DEAD_SHARES_;
        if (wei18(Xs[j]) == 0 && !chain) continue;
        g7_close++; if (chain) g7_chain++;
        const int n = a + j;
        for (Loan *o : H) {
            if (n < o->n1 || n > o->n2 || o->sh.empty()) continue;
            o->xc += o->x(n); g7_xcredit += o->x(n); o->x(n) = 0.0;
            if (chain) { o->ycr += o->y(n); g7_ycredit += o->y(n); o->y(n) = 0.0; o->sh[n - o->n1] = 0; }
            o->touched = true;
        }
    }
    if (into) g7_into++;
    for (Loan *o : H) if (std::max(o->n1, a) <= std::min(o->n2, a + N - 1)) o->totals();
}
static std::map<int, int> g7_pb;          // --m7v 3: bands holding phantom x (re-cut on g7_dirty)
// re-cut band n: every holder's slice = its shares' part of the pooled band (x incl. phantom, y); phantom = sum of the holders' kept credits (the contract's xc)
static void m7_recut(std::vector<Loan*> &Hn, int n, double xc_band) {
    u128 S = 0; double X = 0, Y = 0;
    for (Loan *o : Hn) { S += o->sh[n - o->n1]; X += o->x(n) - o->ph[n - o->n1]; Y += o->y(n); }
    if (S == 0) return;
    for (Loan *o : Hn) {
        const double w = (double)(long double)o->sh[n - o->n1] / (double)(long double)S;
        o->ph[n - o->n1] = xc_band * w; o->x(n) = (X + xc_band) * w; o->y(n) = Y * w;
    }
}
static void m7_reconcile(std::vector<Loan*> &A) {               // A = every alive loan
    if (!g7_dirty) return;
    g7_dirty = false;
    std::map<int, std::vector<Loan*>> hb;
    for (Loan *o : A) if (!o->ph.empty()) for (int n = o->n1; n <= o->n2; n++) if (g7_pb.count(n) && o->sh[n - o->n1] != 0) hb[n].push_back(o);
    for (auto it = g7_pb.begin(); it != g7_pb.end();) {
        auto h = hb.find(it->first);
        if (h == hb.end()) { it = g7_pb.erase(it); continue; }
        double xcb = 0; for (Loan *o : h->second) xcb += o->cr[it->first - o->n1];
        m7_recut(h->second, it->first, xcb);
        for (Loan *o : h->second) o->totals();
        ++it;
    }
}
// --m7v 3 deposit_range, before the new shares: the close as the contract decides it (live x = bands_x - xc != 0 in wei, or too little y);
// a close: the holders' live x becomes their credit (kept in the band: phantom); a chain_end: x and y leave the band
static void m7_settle3(std::vector<Loan*> &H, int a, int N) {
    m7_reconcile(H);
    std::vector<u128> S(N, 0), Yw(N, 0); std::vector<double> Xl(N, 0.0);
    for (Loan *o : H) for (int n = std::max(o->n1, a); n <= std::min(o->n2, a + N - 1); n++) {
        if (o->sh.empty() || o->sh[n - o->n1] == 0) continue;
        S[n - a] += o->sh[n - o->n1]; Yw[n - a] += wei18(o->y(n)); Xl[n - a] += o->x(n) - o->ph[n - o->n1];
    }
    bool into = false;
    for (int j = 0; j < N; j++) {
        const int n = a + j;
        if (S[j] == 0) continue;
        bool hx = false; for (Loan *o : H) if (n >= o->n1 && n <= o->n2 && !o->sh.empty() && o->sh[n - o->n1] != 0 && o->x(n) > 0.0) hx = true;
        if (hx) into = true;
        const bool chain = Yw[j] * MAX_SHARES_PER_Y_ < S[j] + DEAD_SHARES_;
        if (wei18(Xl[j]) == 0 && !chain) continue;
        g7_close++; if (chain) g7_chain++;
        for (Loan *o : H) {
            if (n < o->n1 || n > o->n2 || o->sh.empty() || o->sh[n - o->n1] == 0) continue;
            const int i = n - o->n1; const double live = o->x(n) - o->ph[i];
            o->xc += live; g7_xcredit += live;
            if (chain) { o->ycr += o->y(n); g7_ycredit += o->y(n); o->x(n) = 0.0; o->y(n) = 0.0; o->ph[i] = 0.0; o->cr[i] = 0.0; o->sh[i] = 0; }
            else { o->cr[i] += live; o->ph[i] = o->x(n); g7_pb[n] = 1; }
            o->touched = true;
        }
    }
    if (into) g7_into++;
    for (Loan *o : H) if (std::max(o->n1, a) <= std::min(o->n2, a + N - 1)) o->totals();
}

// the new shares of loan o (just placed over [o.n1, o.n2]) against the bands' shares and y of the others H
static void m7_mint(std::vector<Loan*> &H, Loan &o) {
    const int N = o.n2 - o.n1 + 1; std::vector<u128> S(N, 0), Yw(N, 0);
    for (Loan *h : H) for (int n = std::max(h->n1, o.n1); n <= std::min(h->n2, o.n2); n++) {
        if (h->sh.empty()) continue;
        S[n - o.n1] += h->sh[n - h->n1]; Yw[n - o.n1] += wei18(h->y(n));
    }
    o.sh.assign(N, 0);
    for (int j = 0; j < N; j++) o.sh[j] = muldiv(S[j] + DEAD_SHARES_, wei18(o.y(o.n1 + j)), Yw[j] + 1);
    if (g_m7v >= 3) {                                          // the newcomer's slice of a band with phantom x: re-cut the band (the curve does not move)
        o.ph.assign(N, 0.0); o.cr.assign(N, 0.0);
        for (int j = 0; j < N; j++) {
            const int n = o.n1 + j; if (!g7_pb.count(n)) continue;
            std::vector<Loan*> Hn; double xcb = 0;
            for (Loan *h : H) if (n >= h->n1 && n <= h->n2 && !h->ph.empty() && h->sh[n - h->n1] != 0) { Hn.push_back(h); xcb += h->cr[n - h->n1]; }
            if (xcb == 0.0) continue;
            Hn.push_back(&o); m7_recut(Hn, n, xcb); o.touched = true;
            for (Loan *h : Hn) h->totals();
        }
    }
}

// v2 concept Controller._calculate_debt_f_remove (v2/concept/curve_stablecoin/controller.vy 1238-1307) in floats, for a
// liquidator (health_limit = the user's liquidation discount). xd_raw = AMM.get_x_down(user), D = debt, V_or = x + y * p_o
// (the AMM's price_oracle(), NOT the market). Returns the most a liquidator may repay (frac_max x D) and the share of the
// position withdrawn for it; fallback (not capped) = frac_max = f_remove = 1, i.e. withdrawn in the same fraction as repaid.
struct PLq { double frac_max, f_remove, bonus; bool capped; };
static PLq pl_v2c(double xd_raw, double D, double ld, double V_or) {
    const double TH = g_th, MBH = g_mbh, MAXB = g_maxb, MINB = g_minb, MDL = g_mdl;   // TARGET_HEALTH ... MIN_DEBT_LEFT (flags, defaults = the contract's constants)
    PLq r{1.0, 1.0, 1.0 + MAXB, false};
    const double x_down = xd_raw * (1.0 - ld), health = x_down / D;                     // 10**18 + health(full=False)
    double bonus = 1.0 + MAXB;
    if (health > MBH) bonus -= (MAXB - MINB) * (std::min(health, 1.0) - MBH) / (1.0 - MBH);
    r.bonus = bonus;
    const double num = TH * D, den = V_or > 0 ? bonus * x_down / V_or : 0.0;          // unsafe_div by 0 -> 0
    if (num > x_down && den < TH) {
        const double repay = (num - x_down) / (TH - den);
        if ((MDL > 0 ? repay + MDL <= D : repay < D) && repay * bonus < V_or) { r.frac_max = repay / D; r.f_remove = repay * bonus / V_or; r.capped = true; }
    }
    return r;
}

// the loan's contract health with price p (the exchange spot) in place of the oracle; the loan's p_o is restored
static double spot_health(Loan &o, double p) {
    const double po = o.p_o, po3 = o.po3;
    o.p_o = p; o.po3 = p * p * p;
    const double h = o.health();
    o.p_o = po; o.po3 = po3;
    return h;
}

struct Params {
    double hl = 600, cap = 0, dfee_mult = 1.0;
    int oracle = 0;
    int down_only = 1;
    double hl_fast = 120, g0 = 0.01, g0_up = 0, g_exp = 2;
    double ld_mult = 1.0, fee_mult = 1.0;
    double A_mult = 1.0, vol_fee = 0.0, vol_ref = 0.0005; int vol_win = 30;   // round 10: wider bands (A x A_mult, bands remapped), volatility fee
    int aave = 0;
    double aave_lb = 0, aave_fee = 0; int aave_slice = 0, aave_v33 = 0, aave_eqcap = 0;   // round_r7 aave (header)
    double lb[3] = {0.05, 0.05, 0.083};
    double thf = 1.05, hf_maxb = 0.95, lb_factor = 0.5, dust = 1000;
    double ext_fee = 2e-3; int lat = 1; double gas = 10.0;
    double liq_cap = 0;               // crvUSD one liquidator brings per tx (0 = unlimited): bigger loans are liquidated in slices (liquidate_extended)
    int resync = 0;                   // 1: a borrower action outside soft liquidation sets the loan to the chain's state after it (collateral, debt, bands); 2: the action as a delta on the sim loan (header)
    int dyn_fee = 0;                  // 1: the AMM's per-band dynamic fee inside trades
    int liq_rate = 0;                 // > 0: at most this many liquidations per market-minute, the most profitable loans first (chain bots)
    int liq_every = 1;                // liquidations only every N-th minute (bots' throughput in bursts)
    double arb_gas = 0;               // --arb-gas G (round r1, diagnosis of today's reverted adjusts): the minute's AMM sale happens only if the
                                      // arbitrageur's gain (collateral at the minute high less crvUSD paid) covers G M gas at the hour's gas price; 0 = off
    int sl_reopen = -1;               // --sl-reopen 0|1 (audit3, round r1): --resync 2, a loan holding crvUSD: a refused add / borrow / remove = full repay with the crvUSD
                                      // applied + create_loan at the simulated oracle (no buy-back); -1 (default) = on with --shared-amm 5, off otherwise
    double conv_liq = 0;              // > 0: a loan whose converted share X / (X + Y Cl) exceeds this is closed out through the partial liquidator (full repay at the bonus)
    int auto_close = 0;               // 1: a loan whose crvUSD in the bands covers its debt is closed, leftover collateral returned, no loss
    int aave_oracle = 0;              // 1: the Aave-style loans are judged on the market's oracle series (ema) instead of the market price
    int v2c = 0, exact = 0;           // v4: see g_v2c / g_exact
    double fee_cap_set = 0;           // v4 --fee-cap X (> 0): AMM fee cap instead of 0.099 / the v2 MAX_FEE (attribution runs only)
    int no_dl = 0;                    // 1: one-way AMM: collateral is sold on the way down, never bought back (no de-liquidation); the crvUSD stays in the bands
    double A_set = 0;                 // > 0: set every market's A to this value (A_mult = A_set / A of the market)
    int auto_delay = 0;               // minutes a loan must stay covered before the auto-close fires (bot / borrower latency)
    int auto_mode = 0;                // 1: close only when the loan is also liquidatable (health < 0), i.e. inside today's liquidate trigger, surplus back to the borrower
    double auto_bounty = 0, auto_bounty_pct = 0, auto_bounty_cap = 1e18;   // what the caller of close_covered earns: max(fixed, pct x debt) capped, paid from the loan's surplus
    std::vector<double> auto_lat;     // --auto-lat m1,m2,m3,m4,m5,m6,mb: bot delay (minutes) for bounty/gas in [0,1) [1,3) [3,10) [10,30) [30,100) [100,inf) and for bursts of >= 15 waiting loans
    std::vector<long> dump_at;        // --dump-at T1,T2,...: 'D' lines (every alive loan) and 'T' lines (trade replay) in the liq file at those minutes   // ext_fee calibrated on 3,000 real health readings (replay/uwcal.py)
    int orphan = 1, noliq = 0;
    int dfee_blocks = 5, trace_uid = -1, partial = 0;
    double lb_scale = 1.0; long t_end = 0;
    double fee_und[3] = {0, 0, 0};
    int aave_lt_liq = 0;
    std::string gas_file; double gas_units = 1.0;
    std::string oracle_series, oracle_dump;   // external oracle on the underlying (float64 per candle row) / dump of the engine's own
    std::string trades;               // --trades FILE: float64 rows [t, y_sold, y_bought] per minute; replaces the arbitrage model by the real trades
    int exposure = 0;                 // 1 = weekly: debt an oracle drop of m would make liquidatable + liquidator gain   // real gas: USD per 1M gas by hour (liqtest/gas_usd_hourly.bin) x millions of gas per liquidation              // Aave clone: 1 = liquidation threshold at Curve's LIQUIDATION line (1-ld)((A-1)/A)^2 per loan, 0 = max borrow LTV    // absolute LLAMMA fee per underlying (ETH, BTC, CRV); 0 = the market's own fee
    long trace_from = 0, trace_to = 0;              // AMM trades per minute in a moving market (dynamic fee is updated per trade)
    int wo = 0;                       // v5 --wo 1: WindowOracle.vy (window_oracle.h), written by the AMM's trades only
    std::string wo_pool, wo_log;
    long wo_window = 36, wo_stale = 300; double wo_tau = 300.0, wo_cap = 0.003; int wo_nobs = 4; long wo_keep = 0; int wo_asym = 0; double wo_rise_clamp = 0.5, wo_fall_clamp = 0.5;
    int wo_sub = 0; std::string wo_swaps;     // --wo-sub 1: the real swap blocks write the oracle at their own block time
    int gas_eth = 0; std::string eth_candles; // --gas-eth 1: gas file = base fee in gwei, x 1M gas x gas_units x the ETH price
};

struct Week { long t; double base, fee; int act; double ratio, ldisc, lt, liqd; };
struct Act { long t; int uid, kind; double coll, debt; int n1, n2; double dcoll, ddebt; int nf; double remd = 0, remc = 0; };   // remd/remc: debt and collateral the real loan kept after a partial (self-)liquidation
struct Init { int uid; double d, x, y; int n1, n2; };
struct Trade { long t; double dir, y, p; };   // --trades: one real AMM trade, in chain order: dir +1 = the AMM sold y collateral / -1 bought it back, p = realised crvUSD per unit

int main(int argc, char **argv) {
    Params P; std::string in_path, out_path, liq_path, bd_path, cand; double dep = 2e8;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--hl") P.hl = std::stod(v);
        else if (a == "--cap") P.cap = std::stod(v);
        else if (a == "--dfee-mult") P.dfee_mult = std::stod(v);
        else if (a == "--oracle") P.oracle = std::stoi(v);
        else if (a == "--down-only") P.down_only = std::stoi(v);
        else if (a == "--hl-fast") P.hl_fast = std::stod(v);
        else if (a == "--g0") P.g0 = std::stod(v);
        else if (a == "--g0-up") P.g0_up = std::stod(v);
        else if (a == "--g-exp") P.g_exp = std::stod(v);
        else if (a == "--ld-mult") P.ld_mult = std::stod(v);
        else if (a == "--fee-mult") P.fee_mult = std::stod(v);
        else if (a == "--A-mult") P.A_mult = std::stod(v);
        else if (a == "--vol-fee") P.vol_fee = std::stod(v);
        else if (a == "--vol-ref") P.vol_ref = std::stod(v);
        else if (a == "--vol-win") P.vol_win = std::stoi(v);
        else if (a == "--aave") P.aave = std::stoi(v);
        else if (a == "--aave-lb") P.aave_lb = std::stod(v);
        else if (a == "--aave-fee") P.aave_fee = std::stod(v);
        else if (a == "--aave-slice") P.aave_slice = std::stoi(v);
        else if (a == "--aave-v33") P.aave_v33 = std::stoi(v);
        else if (a == "--aave-eqcap") P.aave_eqcap = std::stoi(v);
        else if (a == "--thf") P.thf = std::stod(v);
        else if (a == "--hf-maxb") P.hf_maxb = std::stod(v);
        else if (a == "--lb-factor") P.lb_factor = std::stod(v);
        else if (a == "--ext-fee") P.ext_fee = std::stod(v);
        else if (a == "--lat") P.lat = std::stoi(v);
        else if (a == "--liq-cap") P.liq_cap = std::stod(v);
        else if (a == "--resync") P.resync = std::stoi(v);
        else if (a == "--dyn-fee") { P.dyn_fee = std::stoi(v); g_dyn_fee = P.dyn_fee; }
        else if (a == "--dyn-frac") g_dyn_frac = std::stod(v);
        else if (a == "--liq-rate") P.liq_rate = std::stoi(v);
        else if (a == "--liq-every") P.liq_every = std::stoi(v);
        else if (a == "--conv-liq") P.conv_liq = std::stod(v);
        else if (a == "--auto-close") P.auto_close = std::stoi(v);
        else if (a == "--no-dl") { P.no_dl = std::stoi(v); g_no_dl = P.no_dl; }
        else if (a == "--shared-amm") { g_shared = std::stoi(v); if (g_shared == 7) { g_shared = 5; g_m7 = 1; } }
        else if (a == "--sl-reopen") P.sl_reopen = std::stoi(v);
        else if (a == "--dl-checks") g_dlc = std::stoi(v);
        else if (a == "--m7v") g_m7v = std::stoi(v);
        else if (a == "--cdec") g_cdec = std::stoi(v);
        else if (a == "--as-on") g_ason = v;
        else if (a == "--ev") g_ev = std::stoi(v);
        else if (a == "--arb-gas") P.arb_gas = std::stod(v);
        else if (a == "--full-term") g_ft = std::stoi(v);
        else if (a == "--v2c") { P.v2c = std::stoi(v); g_v2c = P.v2c; }
        else if (a == "--v2c-th") g_th = std::stod(v);
        else if (a == "--v2c-mbh") g_mbh = std::stod(v);
        else if (a == "--v2c-maxb") g_maxb = std::stod(v);
        else if (a == "--v2c-minb") g_minb = std::stod(v);
        else if (a == "--v2c-mdl") g_mdl = std::stod(v);
        else if (a == "--exact") { P.exact = std::stoi(v); g_exact = P.exact; }
        else if (a == "--fee-cap") P.fee_cap_set = std::stod(v);
        else if (a == "--A-set") P.A_set = std::stod(v);
        else if (a == "--auto-delay") P.auto_delay = std::stoi(v);
        else if (a == "--auto-mode") P.auto_mode = std::stoi(v);
        else if (a == "--auto-bounty") P.auto_bounty = std::stod(v);
        else if (a == "--auto-bounty-pct") P.auto_bounty_pct = std::stod(v);
        else if (a == "--auto-bounty-cap") P.auto_bounty_cap = std::stod(v);
        else if (a == "--auto-lat") { std::stringstream ds(v); std::string tok; while (std::getline(ds, tok, ',')) P.auto_lat.push_back(std::stod(tok)); }
        else if (a == "--dump-at") { std::stringstream ds(v); std::string tok; while (std::getline(ds, tok, ',')) P.dump_at.push_back(std::stol(tok)); }
        else if (a == "--gas") P.gas = std::stod(v);
        else if (a == "--gas-file") P.gas_file = v;
        else if (a == "--exposure") P.exposure = std::stoi(v);
        else if (a == "--oracle-series") P.oracle_series = v;
        else if (a == "--trades") P.trades = v;
        else if (a == "--oracle-dump") P.oracle_dump = v;
        else if (a == "--gas-units") P.gas_units = std::stod(v);
        else if (a == "--orphan") P.orphan = std::stoi(v);
        else if (a == "--noliq") P.noliq = std::stoi(v);
        else if (a == "--dfee-blocks") P.dfee_blocks = std::stoi(v);
        else if (a == "--trace-uid") P.trace_uid = std::stoi(v);
        else if (a == "--partial") P.partial = std::stoi(v);
        else if (a == "--fee-eth") P.fee_und[0] = std::stod(v);
        else if (a == "--aave-lt-liq") P.aave_lt_liq = std::stoi(v);
        else if (a == "--aave-oracle") P.aave_oracle = std::stoi(v);
        else if (a == "--fee-btc") P.fee_und[1] = std::stod(v);
        else if (a == "--fee-crv") P.fee_und[2] = std::stod(v);
        else if (a == "--lb-scale") P.lb_scale = std::stod(v);
        else if (a == "--t-end") P.t_end = std::stol(v);
        else if (a == "--trace-from") P.trace_from = std::stol(v);
        else if (a == "--trace-to") P.trace_to = std::stol(v);
        else if (a == "--wo") P.wo = std::stoi(v);
        else if (a == "--wo-pool") P.wo_pool = v;
        else if (a == "--wo-log") P.wo_log = v;
        else if (a == "--wo-window") P.wo_window = std::stol(v);
        else if (a == "--wo-tau") P.wo_tau = std::stod(v);
        else if (a == "--wo-cap") P.wo_cap = std::stod(v);
        else if (a == "--wo-stale") P.wo_stale = std::stol(v);
        else if (a == "--wo-nobs") P.wo_nobs = std::stoi(v);
        else if (a == "--wo-keep") P.wo_keep = std::stol(v);
        else if (a == "--wo-asym") P.wo_asym = std::stoi(v);   // round_r1/oracle: 1 = CAP on rises only, 2 = falls only, 3 = cap >= the pool EMA move
        else if (a == "--wo-rise-clamp") P.wo_rise_clamp = std::stod(v);   // round_r3c/ceiling: per-write ceiling of a rise (contract WAD // 2 = 0.5, default, bit-exact); 1e9 = none
        else if (a == "--wo-fall-clamp") P.wo_fall_clamp = std::stod(v);   // round_r3c/ceiling: per-write ceiling of a fall (0.5 = the old contract, bit-exact); 1e9 = none (the contract since 24 Oct 2026 22:30)
        else if (a == "--wo-sub") P.wo_sub = std::stoi(v);
        else if (a == "--wo-swaps") P.wo_swaps = v;
        else if (a == "--gas-eth") P.gas_eth = std::stoi(v);
        else if (a == "--eth-candles") P.eth_candles = v;
        else if (a == "--depth") dep = std::stod(v);
        else if (a == "--venue-grid") {
            std::ifstream vf(v); if (!vf) { std::fprintf(stderr, "cannot read %s\n", v.c_str()); return 1; }
            int K; vf >> K; g_vg_s.resize(K); for (auto &x : g_vg_s) vf >> x;
            double tt; while (vf >> tt) { std::vector<double> c(K); for (auto &x : c) vf >> x; g_vg_t.push_back(tt); g_vg_c.push_back(c); }
            std::fprintf(stderr, "venue grid %s: %zu rows x %d sizes\n", v.c_str(), g_vg_t.size(), K);
        }
        else if (a == "--candles") cand = v;
        else if (a == "--in") in_path = v;
        else if (a == "--out") out_path = v;
        else if (a == "--liq") liq_path = v;
        else if (a == "--bd") bd_path = v;
    }
    // ---- inputs
    std::ifstream f(in_path); std::string line;
    int mkt = 0, und = 0, nw = 0, nu = 0; double A = 0, liqd = 0;
    std::vector<Week> W; std::vector<Act> AC; std::vector<Init> IN; std::vector<double> real_d;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream s(line.substr(2));
        if (line[0] == 'M') s >> mkt >> und >> A >> liqd >> nw >> nu;
        else if (line[0] == 'W') { Week w; s >> w.t >> w.base >> w.fee >> w.act >> w.ratio >> w.ldisc >> w.lt >> w.liqd; W.push_back(w); }
        else if (line[0] == 'L') { Init q; s >> q.uid >> q.d >> q.x >> q.y >> q.n1 >> q.n2; IN.push_back(q); }
        else if (line[0] == 'A') { Act q; double t; s >> t >> q.uid >> q.kind >> q.coll >> q.debt >> q.n1 >> q.n2 >> q.dcoll >> q.ddebt >> q.nf; if (!(s >> q.remd >> q.remc)) { q.remd = 0; q.remc = 0; } q.t = long(t); AC.push_back(q); }
        else if (line[0] == 'R') { long t; double d; s >> t >> d; real_d.push_back(d); }
    }
    Candles Cd = load_candles(cand.c_str());
    std::vector<double> OS;                       // external oracle series (underlying, USD), one per candle row
    if (!P.oracle_series.empty()) {
        FILE *sf = std::fopen(P.oracle_series.c_str(), "rb");
        if (!sf) { std::fprintf(stderr, "no oracle series %s\n", P.oracle_series.c_str()); std::exit(2); }
        std::fseek(sf, 0, SEEK_END); long sn = std::ftell(sf) / 8; std::fseek(sf, 0, SEEK_SET);
        if (sn != long(Cd.t.size())) { std::fprintf(stderr, "oracle series rows %ld != candle rows %ld\n", sn, long(Cd.t.size())); std::exit(2); }
        OS.resize(sn); if (std::fread(OS.data(), 8, sn, sf) != size_t(sn)) std::exit(2); std::fclose(sf);
    }
    std::vector<Trade> TR; size_t tri = 0; double tr_short = 0;   // real trades (--trades), cursor, collateral value the sim could not deliver
    if (!P.trades.empty()) {
        FILE *tf = std::fopen(P.trades.c_str(), "rb");
        if (!tf) { std::fprintf(stderr, "no trades %s\n", P.trades.c_str()); std::exit(2); }
        double row[4];
        while (std::fread(row, 8, 4, tf) == 4) TR.push_back(Trade{long(row[0]), row[1], row[2], row[3]});
        std::fclose(tf);
    }
    PoolEMA PE; WindowOracle WO; long wo_t_ok = 0, wo_n_min = 0, wo_n_min25 = 0, wo_n_w25 = 0, wo_n_keep = 0; FILE *wlog = nullptr;   // v5 --wo
    if (P.wo) {
        if (P.wo_pool.empty() || !PE.load(P.wo_pool.c_str())) { std::fprintf(stderr, "--wo 1 needs a pool file (--wo-pool), cannot read '%s'\n", P.wo_pool.c_str()); std::exit(2); }
        WO.unlock = P.wo >= 2;
        WO.anchor = P.wo >= 3;
        WO.WINDOW = P.wo_window; WO.TAU = P.wo_tau; WO.CAP = P.wo_cap; WO.STALE = P.wo_stale; WO.N_OBS = P.wo_nobs; WO.ASYM = P.wo_asym; WO.RISE_CLAMP = P.wo_rise_clamp; WO.FALL_CLAMP = P.wo_fall_clamp;
        wo_t_ok = PE.ts[0] + 3600;                                            // the series' rule: NaN before the first record + 1 h
        if (!P.wo_log.empty()) wlog = std::fopen(P.wo_log.c_str(), "w");
    }
    std::vector<long> SWT; size_t swj = 0; long wo_n_sw = 0, wo_n_sw25 = 0;                 // --wo-sub: the market's real swap blocks
    long wo_nr = 0, wo_nr25 = 0, wo_open[4] = {0, 0, 0, 0}, wo_open25[4] = {0, 0, 0, 0};     // minute reads, band open (> 0, 1e-6, 1e-4, 1e-3)
    if (P.wo && P.wo_sub) {
        FILE *sf = P.wo_swaps.empty() ? nullptr : std::fopen(P.wo_swaps.c_str(), "rb");
        if (!sf) { std::fprintf(stderr, "--wo-sub 1 needs --wo-swaps FILE, cannot read '%s'\n", P.wo_swaps.c_str()); std::exit(2); }
        double x; while (std::fread(&x, 8, 1, sf) == 1) SWT.push_back(long(x));
        std::fclose(sf);
    }
    Candles CE; if (P.gas_eth) { if (P.eth_candles.empty()) { std::fprintf(stderr, "--gas-eth 1 needs --eth-candles FILE\n"); std::exit(2); } CE = load_candles(P.eth_candles.c_str()); }
    std::vector<double> OD;                       // --oracle-dump: the engine's own oracle per candle row (NaN outside the run)
    if (!P.oracle_dump.empty()) OD.assign(Cd.t.size(), std::nan(""));
    std::vector<double> gas_t, gas_u;
    if (!P.gas_file.empty()) {
        FILE *gf = std::fopen(P.gas_file.c_str(), "rb");
        std::fseek(gf, 0, SEEK_END); long gn = std::ftell(gf) / 16; std::fseek(gf, 0, SEEK_SET);
        std::vector<double> gb(2 * gn); if (std::fread(gb.data(), 8, 2 * gn, gf) != size_t(2 * gn)) std::exit(2); std::fclose(gf);
        for (long j = 0; j < gn; j++) { gas_t.push_back(gb[2 * j]); gas_u.push_back(gb[2 * j + 1]); }
    }
    if (P.A_set > 0) P.A_mult = P.A_set / A;
    if (P.A_mult != 1.0) {                                                  // wider / narrower bands: keep every loan's price range
        const double A_new = std::max(2.0, A * P.A_mult), f = std::log((A - 1.0) / A) / std::log((A_new - 1.0) / A_new);
        auto remap = [&](int n) { return int(std::lround(n * f)); };
        for (auto &q : IN) { q.n1 = remap(q.n1); q.n2 = remap(q.n2); if (q.n2 - q.n1 + 1 < 4) q.n2 = q.n1 + 3; }
        for (auto &q : AC) { q.n1 = remap(q.n1); q.n2 = remap(q.n2); q.nf = remap(q.nf); if (q.n2 - q.n1 + 1 < 4) q.n2 = q.n1 + 3; }
        A = A_new;
    }
    const double k = (A - 1.0) / A, sbr = std::sqrt(A / (A - 1.0)), lnA = std::log(A / (A - 1.0));
    (void)liqd;
    auto ld_at = [&](long t) { int w = int(std::upper_bound(W.begin(), W.end(), t, [](long a, const Week &b) { return a < b.t; }) - W.begin()) - 1; return W[std::max(0, std::min(nw - 1, w))].liqd * P.ld_mult; };
    FILE *out = std::fopen(out_path.c_str(), "w");
    FILE *lq = liq_path.empty() ? nullptr : std::fopen(liq_path.c_str(), "w");
    FILE *bdf = bd_path.empty() ? nullptr : std::fopen(bd_path.c_str(), "w");

    // start: week 0 if there is a book, otherwise one day before the first action (oracle warm-up)
    long t_start = W[0].t;
    if (IN.empty() && !AC.empty()) t_start = std::max(W[0].t, AC[0].t - 86400);
    long t_end = W.back().t + 7 * 86400;
    if (P.t_end > 0) t_end = std::min(t_end, P.t_end);
    long i0 = std::lower_bound(Cd.t.begin(), Cd.t.end(), double(t_start)) - Cd.t.begin();
    long i1 = std::lower_bound(Cd.t.begin(), Cd.t.end(), double(t_end)) - Cd.t.begin();

    auto week_at = [&](long t) { int w = int(std::upper_bound(W.begin(), W.end(), t, [](long a, const Week &b) { return a < b.t; }) - W.begin()) - 1; return std::max(0, std::min(nw - 1, w)); };
    auto base_at = [&](long t) {
        int w = week_at(t);
        if (w + 1 >= nw || t <= W[0].t) return W[w].base;
        double fr = double(t - W[w].t) / double(W[w + 1].t - W[w].t);
        return W[w].base * std::pow(W[w + 1].base / W[w].base, fr);
    };
    auto ratio_at = [&](long t) {
        int w = week_at(t);
        if (w + 1 >= nw || t <= W[0].t) return W[w].ratio;
        double fr = double(t - W[w].t) / double(W[w + 1].t - W[w].t);
        return W[w].ratio + (W[w + 1].ratio - W[w].ratio) * fr;
    };
    // Controller._calculate_debt_n1 (floor(log(y_eff * base / debt) / log(A / (A - 1)))), loan discount of the week
    auto n1_formula = [&](double coll, double debt, int N, long t) {
        double dsc = W[week_at(t)].ldisc, s = 0, kk = 1;
        for (int i = 0; i < N; i++) { s += kk; kk *= k; }
        double ye = coll * (1.0 - dsc) / (sbr * N) * s;
        return int(std::floor(std::log(std::max(ye * base_at(t) / debt, 1e-300)) / lnA));
    };

    std::vector<Loan> L; L.reserve(IN.size() + AC.size() + 16);
    std::vector<int> cur(nu + 1, -1), orph(nu + 1, -1);
    g_base = base_at(t_start);
    double r0 = ratio_at(t_start);
    double mid_u = 0.5 * (Cd.h[i0] + Cd.l[i0]);
    double ema_u = mid_u, ema_fast = mid_u, dfee = 0.0;
    std::vector<double> vol_buf(std::max(1, P.vol_win), 0.0); double vol_sum = 0.0, prev_mu = mid_u; int vol_i = 0;
    auto n_act_of = [&](double po, double P) { double s = std::sqrt(po * po * po / P); return int(std::floor(std::log(s / g_base) / std::log(k))); };
    auto new_loan = [&](int uid) { Loan o; o.A = A; o.k = k; o.sbr = sbr; o.uid = uid; L.push_back(o); return int(L.size()) - 1; };
    long n_pre = 0, n_resync_new = 0, n_autoclose = 0; double bounty_paid = 0, auto_wait_sum = 0;
    {   // week-0 book
        double po = ema_u * r0, Px = Cd.c[i0] * r0;
        for (auto &q : IN) {
            int li = new_loan(q.uid); Loan &o = L[li];
            o.ld = ld_at(t_start); o.place(q.y, q.n1, q.n2, n_act_of(po, Px));
            if (q.x > 0) {                       // already in soft liquidation: crvUSD above the active band, collateral below
                int act = std::max(o.lo, std::min(W[0].act, o.hi - 1)), N = q.n2 - q.n1 + 1;
                for (int n = q.n1; n <= q.n2; n++) o.y(n) = 0;
                if (act < q.n1) { for (int n = q.n1; n <= q.n2; n++) { o.y(n) = q.y / N; o.x(n) = q.x / N; } }
                else if (act > q.n2) { for (int n = q.n1; n <= q.n2; n++) { o.x(n) = q.x / N; o.y(n) = q.y / N; } }
                else {
                    int nx = act - q.n1 + 1, ny = q.n2 - act + 1;
                    for (int n = q.n1; n <= act; n++) o.x(n) += q.x / nx;
                    for (int n = act; n <= q.n2; n++) o.y(n) += q.y / ny;
                }
                o.active = act; o.touched = true; o.totals();
            }
            o.Dn = q.d / g_base; o.Ya = q.y + q.x / Px;
            o.p_o = po; o.po3 = po * po * po;
            bool bad = P.aave ? (o.Ya * Px * (P.aave_lt_liq ? (1.0 - o.ld) * k * k : W[0].lt) < q.d) : (o.health() < 0);
            if (bad) { o.alive = false; n_pre++; continue; }
            cur[q.uid] = li;
        }
    }
    const double a_up = std::pow(2.0, -60.0 / P.hl);
    const double al_f = 1.0 - std::pow(2.0, -60.0 / P.hl_fast);
    const double LB = (P.aave_lb > 0 ? P.aave_lb : P.lb[std::min(2, std::max(0, und))]) * P.lb_scale;
    size_t ai = 0; int exp_week = -1;
    // weekly tallies
    int wk = week_at(t_start);
    double sl = 0, dl = 0, liq_eq = 0, bd_peak = 0, bd_now = 0, bd_real = 0, sold_sl = 0, rng_dmin = 0, sl_dmin = 0, reb = 0, slm = 0; long rng_lmin = 0, sl_lmin = 0; long n_liq = 0, steps = 0, n_bd_real = 0;
    long hour = -1; double bd_hmax = 0;
    long n_orph_closed = 0, n_ignored = 0, n_conv = 0, n_revert = 0, n_capped = 0;   // --resync 2: actions the contract would revert (part of n_ignored), borrows / removals cut to the simulated limit
    long n_sh_shift = 0, n_sh_revert = 0, n_sh_place = 0; double sh_forgone = 0; int sh_maxshift = 0;   // --shared-amm counters
    long n_reopen = 0, n_cr_cut = 0, n_cr_revert = 0, n_arb_skip = 0, n_rep_unh = 0;
    long n_cr_band = 0, n_cr_band97 = 0; double cr_band_forgone = 0;   // round r2 (E4): creates whose asked range holds x ("Band not empty"), of them asked >= 0.97 x the engine's max_borrowable   // + partial repays refused: health(full=False) < 0 after them                    // --sl-reopen reopens; mode 5 creates cut / reverted at "Debt too high"
    const bool reopen = P.sl_reopen > 0 || (P.sl_reopen < 0 && g_shared == 5);
    if (g_shared == 5) g_act_m = W[0].act;                               // the chain's active band at the start
    const bool ctl = g_shared == 5 || g_dlc;                             // the simulated Controller's checks (mode 5 / 7, or --dl-checks 1 on the DL-on path)
    std::map<long, std::array<double, 3>> ASON;                         // round r3c G5(a): action index -> the DL-on loan's (Y, X, D)
    if (!g_ason.empty()) { FILE *fa = std::fopen(g_ason.c_str(), "r"); int mm; long aa; double y_, x_, d_;
        if (fa) { while (std::fscanf(fa, "%d %ld %lf %lf %lf", &mm, &aa, &y_, &x_, &d_) == 5) if (mm == mkt) ASON[aa] = {y_, x_, d_}; std::fclose(fa); } }
    long ev_ai = -1;
    auto ev = [&](const Act &q, const char *why, double asked, double got) { if (g_ev) std::fprintf(stderr, "EV %d %ld %ld %d %d %s %.6f %.6f\n", mkt, ev_ai, q.t, q.uid, q.kind, why, asked, got); };
    // --shared-amm: first band >= a where a deposit of N bands is allowed in the market's one shared one-way AMM. The crvUSD (x) bands
    // are the converted bands of the other alive loans (orphans included; the loan being re-placed has withdrawn first).
    // 1: n1 below every x band. 2: n1 below every band with x and y, and no x band inside [n1, n1 + N - 1].
    // 3 (redesign dloff, 23 Oct 2026): every band with x can be skipped (the AMM reads a skipped band's x when a trade or the
    //   health reaches it); only no x band inside [n1, n1 + N - 1] ("Band not empty"). Plus the trade rule in Loan::trade_to_price.
    auto shared_n1 = [&](int a, int N, int self_li) {
        if (g_m7) { n_sh_place++; return a; }                            // mode 7: no band blocks a deposit
        int bmax = -(1 << 30); std::vector<int> xb;
        for (int i = 0; i < (int)L.size(); i++) {
            Loan &o = L[i];
            if (!o.alive || !o.touched || i == self_li) continue;
            for (int n = o.n1; n <= o.n2; n++) {
                if (o.x(n) <= (g_shared == 5 ? 0.0 : 1e-6)) continue;   // mode 5: any x is "Band not empty"
                if (g_shared == 1 || (g_shared == 2 && o.y(n) > 1e-12)) bmax = std::max(bmax, n);
                if (g_shared >= 2) xb.push_back(n);
            }
        }
        int s = std::max(a, bmax + 1);
        std::sort(xb.begin(), xb.end());
        for (int n : xb) if (n >= s && n <= s + N - 1) s = n + 1;   // sorted: one pass moves s past every x band in the way
        n_sh_place++;
        return s;
    };
    long n_m7_settle = 0, n_m7_place_x = 0; double m7_credit = 0;     // mode 7: bands settled, placements whose range held others' crvUSD, crvUSD moved to credit
    std::vector<Loan*> H7;
    auto others7 = [&](int self_li) { H7.clear(); for (int i = 0; i < (int)L.size(); i++) if (L[i].alive && i != self_li) H7.push_back(&L[i]); return std::ref(H7); };
    auto settle7 = [&](int a, int N, int self_li) {                     // AMM.deposit_range: x of the other holders leaves each band of [a, a + N - 1]
        if (!g_m7) return;
        if (g_m7v >= 3) { if (self_li >= 0 && !L[self_li].ph.empty()) g7_dirty = true; m7_settle3(others7(self_li).get(), a, N); return; }
        if (g_m7v >= 2) { m7_settle(others7(self_li).get(), a, N); return; }
        bool hx = false;
        for (int i = 0; i < (int)L.size(); i++) {
            Loan &o = L[i];
            if (!o.alive || !o.touched || i == self_li) continue;
            for (int n = std::max(o.n1, a); n <= std::min(o.n2, a + N - 1); n++) {
                if (o.x(n) <= 0.0) continue;
                o.xc += o.x(n); m7_credit += o.x(n); o.x(n) = 0.0; n_m7_settle++; hx = true;
            }
            o.totals();                                                  // X / Y re-summed now: no float dust at the next trade
        }
        if (hx) n_m7_place_x++;
    };
    auto mint7 = [&](int li) { if (g_m7 && g_m7v >= 2) m7_mint(others7(li).get(), L[li]); };   // --m7v 2: the new shares of a loan just placed
    if (g_m7 && g_m7v >= 2) for (int i = 0; i < (int)L.size(); i++) if (L[i].alive) mint7(i);   // the week-0 book, in its order
    auto flush_week = [&](int w) {
        double dsum = 0, vsum = 0; long na = 0, no = 0;
        double Pn = Cd.c[std::max(i0, std::min(i1 - 1, long(std::lower_bound(Cd.t.begin(), Cd.t.end(), double(W[w].t + 7 * 86400)) - Cd.t.begin()) - 1))] * ratio_at(W[w].t);
        double reb_open = 0;
        for (auto &o : L) if (o.alive) { na++; if (o.orphan) no++; dsum += o.D(); vsum += P.aave ? o.Ya * Pn : o.X + o.Y * Pn; reb_open += o.sold_u * Pn - o.sold_x; }
        std::fprintf(out, "%d %ld %.4f %.4f %.4f %ld %.4f %.4f %ld %ld %.2f %.2f %.2f %ld %.2f %ld %.2f %.2f %.2f\n", mkt, W[w].t, sl, dl, liq_eq, n_liq, bd_peak, bd_now, steps, na, dsum, vsum,
                     w + 1 < (int)real_d.size() ? real_d[w + 1] : 0.0, no, bd_real, n_bd_real, g_fee_usd, g_vol_usd, sold_sl);
        std::fseek(out, -1, SEEK_CUR); std::fprintf(out, " %ld %ld %.0f %.0f %.2f %.2f %.4f\n", rng_lmin, sl_lmin, rng_dmin, sl_dmin, reb, reb_open, slm);
        sl = dl = liq_eq = bd_peak = bd_real = reb = slm = 0; n_liq = n_bd_real = 0; steps = 0; g_fee_usd = g_vol_usd = sold_sl = 0;
        rng_lmin = sl_lmin = 0; rng_dmin = sl_dmin = 0;
    };
    double Cl_cur = Cd.c[i0] * r0;
    int reb_site = 0; double rebs[4] = {0, 0, 0, 0};                     // where the vs-holding mark is booked: 1 chain events, 2 the sim's liquidator, 3 auto-close
    // the loan leaves the sim's book (closed, liquidated, auto-closed, re-taken from the chain): what the AMM's net sale is worth vs holding
    auto realise = [&](Loan &o) { if (!o.ph.empty()) g7_dirty = true; const double m = o.sold_u * Cl_cur - o.sold_x; reb += m; rebs[reb_site] += m; o.sold_u = o.sold_x = 0; };
    auto realise_part = [&](Loan &o, double fr) { const double m = fr * (o.sold_u * Cl_cur - o.sold_x); reb += m; rebs[reb_site] += m; };   // the slice a liquidator takes: its share of the net sale, settled now
    long t_now = 0;                                                    // round r3c: the minute, for the DEATH lines
    auto close_ = [&](int li, int line) { realise(L[li]); L[li].alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t_now, L[li].uid, line); if (cur[L[li].uid] == li) cur[L[li].uid] = -1; };
#define close(li) close_(li, __LINE__)
    double Cl_prev = Cd.c[i0] * r0;

    for (long i = i0; i < i1; i++) {
        long t = long(Cd.t[i]); t_now = t;
        int w = week_at(t);
        while (wk < w) { flush_week(wk); wk++; }
        g_base = base_at(t);
        const double rt = ratio_at(t);
        const double fee_cap = P.fee_cap_set > 0 ? P.fee_cap_set : P.v2c ? std::min(4.0 / A, 0.1) : 0.099;                      // v2 AMM: MIN_FEE <= fee <= MAX_FEE = min(4 / A, 10 %)
        const double fee = P.fee_und[und] > 0 ? std::min(P.fee_und[und], fee_cap) : std::min(W[w].fee * P.fee_mult, fee_cap);
        const double LT = W[w].lt;
        double gas_now = P.gas;
        if (!gas_t.empty()) {
            long gj = long(std::upper_bound(gas_t.begin(), gas_t.end(), double(t)) - gas_t.begin()) - 1;
            gas_now = P.gas_units * gas_u[std::max(0L, std::min(long(gas_u.size()) - 1, gj))];
            if (P.gas_eth == 2) gas_now = P.gas_units * 1.07 * (gas_u[std::max(0L, std::min(long(gas_u.size()) - 1, gj))] + 1.0);   // 1.07M gas at base fee + 1 gwei
            if (P.gas_eth) {                                                  // gwei x 1M gas = 1e-3 ETH per gwei and million gas
                const long ej = long(std::upper_bound(CE.t.begin(), CE.t.end(), double(t)) - CE.t.begin()) - 1;
                gas_now *= 1e-3 * CE.c[std::max(0L, std::min(long(CE.c.size()) - 1, ej))];
            }
        }
        auto LTo = [&](const Loan &o) { return P.aave_lt_liq ? (1.0 - o.ld) * k * k : LT; };
        const double H = Cd.h[i] * rt, Lw = Cd.l[i] * rt, Cl = Cd.c[i] * rt; Cl_cur = Cl;
        const double mu = 0.5 * (Cd.h[i] + Cd.l[i]);
        // ---- 1) oracle (on the underlying, scaled by the ratio)
        double nw_;
        if (P.oracle == 2) ema_fast = ema_fast * (1.0 - al_f) + mu * al_f;      // the fast tracker runs every minute
        const long t_now = t + 60;                                            // v5 --wo: the minute end (the series' sampling time)
        bool wo_on = false; double wo_v = 0.0;
        if (P.wo) {
            if (!WO.deployed && t_now >= wo_t_ok) WO.deploy(t_now, PE.view(t_now));   // with the market (the factory's price_w in the same block writes nothing)
            if (WO.deployed && P.wo_sub) {                                    // --wo-sub: the real swap blocks up to the minute end write first
                while (swj < SWT.size() && SWT[swj] <= t_now) {
                    const long tb = SWT[swj++];
                    if (tb <= WO.t_last) continue;
                    WO.price_w(tb, PE.view(tb), PE.ma(tb)); wo_n_sw++; if (tb > 1735689600L) wo_n_sw25++;
                }
            }
            if (WO.deployed) { wo_on = true; wo_v = WO.price(t_now, PE.view(t_now), PE.ma(t_now)); wo_n_min++; if (t_now > 1735689600L) wo_n_min25++; }
            if (WO.deployed) {                                                // the band at the read (stats only)
                const WindowOracle::Band bb = WO.band(t_now, PE.view(t_now), PE.ma(t_now));
                const double wr = (bb.hi - bb.lo) / bb.p, th[4] = {0.0, 1e-6, 1e-4, 1e-3};
                wo_nr++; if (t_now > 1735689600L) wo_nr25++;
                for (int q = 0; q < 4; q++) if (wr > th[q]) { wo_open[q]++; if (t_now > 1735689600L) wo_open25[q]++; }
            }
        }
        const bool ext = wo_on || (!OS.empty() && OS[i] > 0.0);               // NaN > 0 is false -> own oracle
        if (wo_on) nw_ = wo_v;
        else if (ext) nw_ = OS[i];
        else if (P.oracle == 2) {
            double g = std::log(ema_u / ema_fast), gz;
            if (g >= 0) gz = g / P.g0; else gz = P.down_only ? 0.0 : (-g) / (P.g0_up > 0 ? P.g0_up : P.g0);
            double wgt = 1.0 - std::exp(-std::pow(gz, P.g_exp));
            double hl_eff = P.hl * (1.0 - wgt) + P.hl_fast * wgt;
            double al = 1.0 - std::pow(2.0, -60.0 / hl_eff);
            nw_ = ema_u * (1.0 - al) + mu * al;
        } else nw_ = ema_u * a_up + mu * (1.0 - a_up);
        if (P.cap > 0 && !ext) nw_ = std::min(std::max(nw_, ema_u * std::exp(-P.cap)), ema_u * std::exp(P.cap));
        nw_ = std::min(std::max(nw_, ema_u / 1.25), ema_u * 1.25);           // LLAMMA limit_p_o: MAX_P_O_CHG 1.25 per step
        double r = std::min(ema_u, nw_) / std::max(ema_u, nw_);
        if (P.dfee_blocks <= 1) dfee = (dfee + (1.0 - r * r * r)) * 0.5;
        else {
            // LLAMMA.limit_p_o on every trade: dfee = (old_dfee + 1 - r^3) * (120 s - dt) / 120 s, old values
            // stored at each trade; with a trade every 60/B seconds the oracle moves r^(1/B) between trades
            const double rb3 = std::pow(r, 3.0 / P.dfee_blocks), keep = 1.0 - 1.0 / (2.0 * P.dfee_blocks);
            for (int b = 0; b < P.dfee_blocks; b++) dfee = std::min(1.0, (dfee + (1.0 - rb3)) * keep);
        }
        ema_u = nw_;
        if (!OD.empty()) OD[i] = ema_u;
        const double ema = ema_u * rt, po3 = ema * ema * ema;
        double fee_v = fee;
        if (P.vol_fee > 0) {                                                 // fee x (1 + K x excess realised volatility over the last vol_win minutes)
            const double rr = std::log(mu / prev_mu); prev_mu = mu;
            vol_sum += rr * rr - vol_buf[vol_i]; vol_buf[vol_i] = rr * rr; vol_i = (vol_i + 1) % vol_buf.size();
            const double sig = std::sqrt(std::max(vol_sum, 0.0) / vol_buf.size());
            fee_v = std::min(0.099, fee * (1.0 + P.vol_fee * std::max(0.0, sig / P.vol_ref - 1.0)));
        }
        const double fe = std::min(0.5, std::max(fee_v, P.dfee_mult * dfee));
        // ---- exposure snapshot (first minute of each week): what a manipulated oracle drop of m would make liquidatable
        if (P.exposure && !P.aave && w != exp_week && lq) {
            exp_week = w;
            // round 3 (LLAMMA-Design/r3, 14 Oct 2026): 11 drops (was 6: 0.25 .. 5 %); the X row grows to 1 + 2 * NM fields
            const int NM = 11;
            const double ms[NM] = {0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30};
            double dsum[NM] = {0}, gsum[NM] = {0};
            for (auto &o : L) {
                if (!o.alive) continue;
                o.p_o = ema; o.po3 = po3;
                if (o.health() < 0) continue;                       // already liquidatable without any attack
                const double V = o.X + o.Y * Cl, Dd = o.D();
                for (int kk = 0; kk < NM; kk++) {
                    const double pm = ema * (1.0 - ms[kk]); o.p_o = pm; o.po3 = pm * pm * pm;
                    const double hh = o.health();
                    if (hh >= 0) continue;
                    dsum[kk] += Dd;
                    if (P.partial) {
                        const double tgt = P.thf - 1.0, hf = 1.0 + hh, bmax = 1.0 + LB, bmin = 1.0 + LB * P.lb_factor;
                        const double b = hf <= P.hf_maxb ? bmax : bmax - (bmax - bmin) * (hf - P.hf_maxb) / (1.0 - P.hf_maxb);
                        const double Xd = (hh + 1.0) * Dd;
                        double R = ((1.0 + tgt) * Dd - Xd) / ((1.0 + tgt) - b * Xd / std::max(V, 1e-9));
                        if (!(R > 0) || R * b >= V) R = std::min(Dd, V / b);
                        gsum[kk] += std::min(R, Dd) * (b - 1.0);
                    } else gsum[kk] += std::max(0.0, V - Dd);
                }
                o.p_o = ema; o.po3 = po3;
            }
            std::fprintf(lq, "X %ld", t);
            for (int kk = 0; kk < NM; kk++) std::fprintf(lq, " %.0f %.0f", dsum[kk], gsum[kk]);
            std::fprintf(lq, "\n");
        }
        // ---- 2) real borrower actions of this minute
        while (ai < AC.size() && AC[ai].t < t + 60) {
            ev_ai = long(ai); const Act &q = AC[ai++]; reb_site = 1;
            if (g7_dirty) m7_reconcile(others7(-1).get());
            if (q.uid < 0 || q.uid > nu) continue;
            int li = cur[q.uid];
            if (g_ev >= 2) {                                         // --ev 2: the state each action meets (oracle, the sim loan's collateral / crvUSD / debt)
                if (li >= 0) std::fprintf(stderr, "AS %d %ld %ld %d %d %.12g %.9g %.9g %.9g\n", mkt, ev_ai, q.t, q.uid, q.kind, ema, L[li].Y, L[li].X, L[li].D());
                else std::fprintf(stderr, "AS %d %ld %ld %d %d %.12g -1 -1 -1\n", mkt, ev_ai, q.t, q.uid, q.kind, ema);
            }
            if (!ASON.empty() && li >= 0 && ASON.count(ev_ai)) {        // round r3c G5(a): the DL-off path's checks on the DL-on loan's state (dry run, nothing changes)
                const auto st = ASON[ev_ai]; const double Yv = st[0], Xv = st[1], Dv = st[2];
                auto dry = [&]() -> std::string {
                    if (q.kind != 2 && q.kind != 3) return "KIND" + std::to_string(q.kind);
                    bool isl = Xv > 1e-9 * Dv;
                    if (P.resync == 1 && q.kind == 2) isl = false;
                    if (q.kind == 3) {                                // real repay while underwater
                        const double Dx = Dv + q.ddebt;
                        if (Dx <= 1e-9) return "PASS_CLOSE";
                        if (ctl && Dx - Xv <= 1e-9) return "PASS_CLOSE";
                        return "PASS_NOXD";                           // REP_UNH needs the on-loan's x_down (not in as_on)
                    }
                    if (!isl) {
                        double yc = trc(Yv) + q.dcoll, Dx = Dv + q.ddebt;
                        if (P.resync == 1 && q.coll > 0 && q.debt > 0) { yc = q.coll; Dx = q.debt; }
                        if (P.resync && q.coll <= 0) return "PASS_CLOSE";
                        if (Dx <= 1e-9) return "PASS_CLOSE";
                        if (yc <= 0) return "ADJ_YC0";
                        if (ctl && q.dcoll == 0 && q.ddebt < 0) return "PASS_NOXD";   // _repay_partial branch: x_down of the on-loan not in as_on
                        int N = q.n2 - q.n1 + 1; int f = n1_formula(yc, Dx, N, t); int a = q.n1 + (f - q.nf);
                        if (q.nf - q.n1 > 20) a = std::min(f, n_act_of(ema, Cl) + 1024 - N);
                        if (P.resync == 1 && q.coll > 0 && q.debt > 0) a = q.n1;
                        std::string r = "PASS";
                        if (P.resync == 2 && (q.ddebt > 0 || q.dcoll < 0) && g_base * std::pow(k, a) >= ema) {
                            const int a_min = int(std::floor(std::log(ema / g_base) / std::log(k))) + 1, off = a - f;
                            double s = 0, kk = 1; for (int j = 0; j < N; j++) { s += kk; kk *= k; }
                            const double per = (1.0 - W[week_at(t)].ldisc) / (sbr * N) * s * base_at(t), need = std::exp(lnA * (a_min - off));
                            const double Dlim = yc * per / need * (1.0 - 1e-9), ylim = Dx * need / per * (1.0 + 1e-9);
                            const double Dx0 = Dx;
                            if (q.ddebt > 0 && q.dcoll >= 0 && Dlim > Dv) Dx = std::min(Dx, Dlim);
                            else if (q.dcoll < 0 && q.ddebt <= 0 && ylim < Yv) yc = std::max(yc, ylim);
                            else return "ADJ_REV_DTH";
                            a = n1_formula(yc, Dx, N, t) + off;
                            if (g_base * std::pow(k, a) >= ema) return "ADJ_REV_DTH";
                            r = Dx != Dx0 ? "ADJ_CAP_BORROW" : "ADJ_CAP_REMOVE";
                        }
                        if (g_shared) { int s5 = shared_n1(a, N, li); if (s5 > a) { const double Ds = Dx * std::pow(k, s5 - a); if (q.ddebt <= 0 || Ds < Dv) return "ADJ_REV_BNE"; return "ADJ_CUT_BNE"; } }
                        return r;
                    }
                    if (P.resync == 2 && !(q.dcoll < 0 && q.ddebt < 0)) {
                        if (reopen && (q.dcoll != 0 || q.ddebt > 0)) {
                            const int N = q.n2 - q.n1 + 1;
                            double Dx = Dv - Xv + q.ddebt, yc = trc(Yv) + q.dcoll;
                            if (Dx <= 1e-9) return "PASS_CLOSE";
                            if (yc <= 0 || N < 1) return "REO_YC0";
                            const int f = n1_formula(yc, Dx, N, t); int a = q.n1 + (f - q.nf);
                            if (q.nf - q.n1 > 20) a = std::min(f, n_act_of(ema, Cl) + 1024 - N);
                            std::string r = "PASS_REOPEN";
                            if (g_base * std::pow(k, a) >= ema) {
                                const int a_min = int(std::floor(std::log(ema / g_base) / std::log(k))) + 1, off = a - f;
                                double s = 0, kk = 1; for (int j = 0; j < N; j++) { s += kk; kk *= k; }
                                const double per = (1.0 - W[week_at(t)].ldisc) / (sbr * N) * s * base_at(t), need = std::exp(lnA * (a_min - off));
                                const double Dlim = yc * per / need * (1.0 - 1e-9), ylim = Dx * need / per * (1.0 + 1e-9);
                                const double Dx0 = Dx;
                                if (q.ddebt > 0 && q.dcoll >= 0 && Dlim > Dv - Xv) Dx = std::min(Dx, Dlim);
                                else if (q.dcoll < 0 && q.ddebt <= 0 && ylim < Yv) yc = std::max(yc, ylim);
                                else return "REO_REV_DTH";
                                a = n1_formula(yc, Dx, N, t) + off;
                                if (g_base * std::pow(k, a) >= ema) return "REO_REV_DTH";
                                r = Dx != Dx0 ? "REO_CAP_BORROW" : "REO_CAP_REMOVE";
                            }
                            if (g_shared) { const int s5 = shared_n1(a, N, li); if (s5 > a) { const double Ds = Dx * std::pow(k, s5 - a); if (q.ddebt <= 0 || Ds < Dv - Xv) return "REO_REV_BNE"; return "REO_CUT_BNE"; } }
                            return r;
                        }
                        if (q.dcoll != 0 || q.ddebt > 0) return "SL_REV_UNDERWATER";
                        return "PASS_NOXD";
                    }
                    return "PASS_OTHER";
                };
                std::fprintf(stderr, "DRY %d %ld %s %.9g %.9g %.9g\n", mkt, ev_ai, dry().c_str(), Yv, Xv, Dv);
            }
            if (!ASON.empty() && li < 0 && ASON.count(ev_ai)) std::fprintf(stderr, "DRY %d %ld NOLOAN_OFF\n", mkt, ev_ai);
            if (q.kind == 1) {
                for (int oi : {li, orph[q.uid]}) {                   // the sim's previous loan of this borrower
                    if (oi < 0 || !L[oi].alive) continue;
                    Loan &o = L[oi]; o.p_o = ema; o.po3 = po3;
                    bool healthy = P.aave ? (o.Ya * (P.aave_oracle ? ema : Cl) * LTo(o) >= o.D()) : (o.health() >= 0);
                    if (P.orphan >= 1 && healthy) { close(oi); n_orph_closed++; }
                    else o.orphan = true;
                }
                cur[q.uid] = -1; orph[q.uid] = -1;
                if (q.coll <= 0 || q.debt <= 0 || q.n2 < q.n1) continue;
                int a1 = q.n1, N1 = q.n2 - q.n1 + 1; double d1 = q.debt;
                if (ctl) {                                           // the simulated Controller's _calculate_debt_n1 (+ the chain's formula offset); "Debt too high" -> the most it allows
                    const int f = n1_formula(q.coll, q.debt, N1, t); a1 = q.n1 + (f - q.nf);
                    if (q.nf - q.n1 > 20) a1 = std::min(f, n_act_of(ema, Cl) + 1024 - N1);
                    if (g_base * std::pow(k, a1) >= ema) {
                        const int a_min = int(std::floor(std::log(ema / g_base) / std::log(k))) + 1, off = a1 - f;
                        double s = 0, kk = 1; for (int j = 0; j < N1; j++) { s += kk; kk *= k; }
                        const double per = (1.0 - W[week_at(t)].ldisc) / (sbr * N1) * s * base_at(t);
                        d1 = q.coll * per / std::exp(lnA * (a_min - off)) * (1.0 - 1e-9);
                        a1 = n1_formula(q.coll, d1, N1, t) + off;
                        if (g_base * std::pow(k, a1) >= ema) { n_ignored++; n_revert++; n_cr_revert++; ev(q, "CR_REV_DTH", q.debt, 0); continue; }
                        sh_forgone += q.debt - d1; n_cr_cut++; ev(q, "CR_CUT_DTH", q.debt, d1);
                    }
                    const int s5 = g_shared == 5 ? shared_n1(a1, N1, -1) : a1;
                    if (s5 > a1) {                                   // round r2 (E4): the asked create reverts "Band not empty"; the borrower takes the debt whose range clears the x bands
                        n_cr_band++; cr_band_forgone += d1 - d1 * std::pow(k, s5 - a1);
                        {   // the engine's max_borrowable for this collateral / N (Debt too high cut, then below the x bands = ControllerView): how close the ask was
                            const int a_min = int(std::floor(std::log(ema / g_base) / std::log(k))) + 1, off = a1 - n1_formula(q.coll, d1, N1, t);
                            double s = 0, kk = 1; for (int j = 0; j < N1; j++) { s += kk; kk *= k; }
                            const double per = (1.0 - W[week_at(t)].ldisc) / (sbr * N1) * s * base_at(t);
                            const double dt = q.coll * per / std::exp(lnA * (a_min - off)) * (1.0 - 1e-9); const int at = n1_formula(q.coll, dt, N1, t) + off;
                            const int st = shared_n1(at, N1, -1); const double mbx = dt * std::pow(k, st - at);
                            if (q.debt >= 0.97 * mbx) n_cr_band97++;
                        }
                        const double d5 = d1 * std::pow(k, s5 - a1); sh_forgone += d1 - d5; ev(q, "CR_CUT_BNE", d1, d5); d1 = d5; n_sh_shift++; sh_maxshift = std::max(sh_maxshift, s5 - a1); a1 = s5;
                    }
                } else if (g_shared) {                               // the Controller reverts at this debt: the borrower takes the most it allows
                    a1 = shared_n1(q.n1, N1, -1);
                    if (a1 > q.n1) { d1 = q.debt * std::pow(k, a1 - q.n1); sh_forgone += q.debt - d1; n_sh_shift++; sh_maxshift = std::max(sh_maxshift, a1 - q.n1); }
                }
                settle7(a1, N1, -1);
                int nl = new_loan(q.uid); Loan &o = L[nl];
                o.ld = ld_at(t); o.place(q.coll, a1, a1 + N1 - 1, n_act_of(ema, Cl)); mint7(nl);
                o.Dn = d1 / g_base; o.Ya = q.coll;
                if (g_shared == 5 && a1 <= g_act_m) g_act_m = a1 - 1;   // deposit_range: the active band moves up past the skipped bands
                cur[q.uid] = nl;
                continue;
            }
            if (li < 0) {
                if (P.resync && q.kind == 2 && q.coll > 0 && q.debt > 0 && q.n2 >= q.n1) {   // the chain has a loan the sim lost: take the chain's state
                    settle7(q.n1, q.n2 - q.n1 + 1, -1);
                    int nl = new_loan(q.uid); Loan &o = L[nl];
                    o.ld = ld_at(t); o.place(q.coll, q.n1, q.n2, n_act_of(ema, Cl)); mint7(nl);
                    o.Dn = q.debt / g_base; o.Ya = q.coll;
                    if (g_shared == 5 && q.n1 <= g_act_m) g_act_m = q.n1 - 1;
                    cur[q.uid] = nl; n_resync_new++;
                    continue;
                }
                if (q.kind <= 3) { n_ignored++; ev(q, "NO_SIM_LOAN", q.debt, 0); }
                if (q.kind == 6 && lq) std::fprintf(lq, "R %ld %d 0 0 0 0\n", t, q.uid);
                continue;
            }
            Loan &o = L[li]; o.p_o = ema; o.po3 = po3;
            if (q.kind == 4) { close(li); continue; }
            if (q.kind == 5 || q.kind == 6) {
                double val = P.aave ? o.Ya * Cl : o.X + o.Y * Cl;
                if (lq && q.kind == 6) std::fprintf(lq, "R %ld %d 1 %.5f %.2f %.2f\n", t, q.uid, P.aave ? o.Ya * (P.aave_oracle ? ema : Cl) * LTo(o) / o.D() - 1.0 : o.health(), val - o.D(), o.D());
                if (lq) std::fprintf(lq, "Q %ld %d %d %.2f %.2f %.8g %.8g %.2f %.2f %.8g %.8g %.6g %.8g %.8g %.6g\n", t, q.uid, q.kind, val, o.D(), o.Y, o.X, q.coll * Cl + q.dcoll, q.debt, q.coll, q.dcoll, P.aave ? 0.0 : o.health(), ema, Cl, o.ld);
                if (q.remd > 0) {                                      // partial (frac < 1): the real loan lives on with q.remd debt
                    double fr = q.debt / (q.debt + q.remd);
                    if (q.kind == 6 && val < o.D() && !P.noliq) {      // insolvent here: the slice's shortfall was covered like the real one was
                        bd_real += fr * (o.D() - val); n_bd_real++;
                        if (lq) std::fprintf(lq, "B %ld %d %.2f %.2f\n", t, q.uid, fr * o.D(), fr * val);
                        if (P.aave) { o.Ya *= (1.0 - fr); o.Dn *= (1.0 - fr); } else { realise_part(o, fr); o.scale(1.0 - fr); o.Dn *= (1.0 - fr); o.touched = true; }
                        if (o.D() <= 1e-6) close(li);
                        continue;
                    }
                    if (q.kind == 5) {                                 // the borrower's own slice: replayed like any borrower action
                        if (P.aave) { o.Ya *= (1.0 - fr); o.Dn *= (1.0 - fr); }
                        else { double fm = P.exact ? fr : ((1.0 + o.ld / 2) / (1.0 + o.ld) * (1.0 - fr) + fr) * fr; realise_part(o, fm); o.scale(1.0 - fm); o.Dn *= (1.0 - fr); o.touched = true; }   // --exact: approved caller, health_limit 0 -> f_remove = frac (v1 and v2)
                        if (o.D() <= 1e-6) close(li);
                    }
                    continue;                                          // kind 6: the sim's own liquidator decides; the loan stays the borrower's
                }
                if (q.kind == 5 && val >= o.D()) { close(li); continue; }
                if (val < o.D() && !P.noliq) {                         // insolvent here: covered like the real one was
                    bd_real += o.D() - val; n_bd_real++;
                    if (lq) std::fprintf(lq, "B %ld %d %.2f %.2f\n", t, q.uid, o.D(), val);
                    close(li); continue;
                }
                if (P.orphan == 2 && (P.aave ? o.Ya * (P.aave_oracle ? ema : Cl) * LTo(o) >= o.D() : o.health() >= 0)) { close(li); n_orph_closed++; continue; }
                o.orphan = true; cur[q.uid] = -1; orph[q.uid] = li;
                continue;
            }
            double Dnow = o.D();
            if (P.aave) {
                double Yn = o.Ya + q.dcoll, Dx = Dnow + q.ddebt;
                if (P.resync == 1 && q.kind == 2 && q.coll > 0 && q.debt > 0) { Yn = q.coll; Dx = q.debt; }   // the chain's state after the action is the truth
                if (P.resync && q.kind == 2 && q.coll <= 0) { close(li); continue; }
                if (Dx <= 1e-9) { close(li); continue; }
                if ((q.ddebt > 0 || q.dcoll < 0) && Yn * Cl * LTo(o) < Dx) { n_ignored++; continue; }   // would revert (health)
                o.Ya = std::max(0.0, Yn); o.Dn = Dx / g_base; o.ld = ld_at(t);
                continue;
            }
            bool in_sl = o.X > 1e-9 * o.D();
            if (P.resync == 1 && q.kind == 2) in_sl = false;            // the chain re-placed the loan (it was not underwater there): take its state
            if (q.kind == 3) {                                          // real repay while underwater
                if (lq) std::fprintf(lq, "U %ld %d %.6g %.2f %.8g %.8g %.2f\n", t, q.uid, o.health(), Dnow, o.Y, o.X, Cl);
                double Dx = Dnow + q.ddebt;
                if (Dx <= 1e-9) { close(li); continue; }
                if (ctl && Dx - o.X <= 1e-9) { close(li); continue; }   // repay(): wallet + the bands' crvUSD cover the debt -> _repay_full
                if (ctl && o.x_down_sum() * (1.0 - ld_at(t)) / Dx < 1.0) { n_ignored++; n_revert++; n_rep_unh++; ev(q, "REP_UNH", -q.ddebt, 0); continue; }   // "The action ends with unhealthy state"
                // the real loan was in soft liquidation and kept its bands; the sim loan (just outside soft
                // liquidation here, a knife-edge timing difference) keeps its bands too: only the debt changes
                o.Dn = Dx / g_base;
                continue;
            }
            // kind 2: adjust, real loan not underwater
            if (lq) std::fprintf(lq, "P %ld %d %d %.10g %.10g %.10g %.2f\n", t, q.uid, int(o.touched), o.X, o.Y, q.coll - q.dcoll, Cl);
            if (!in_sl) {
                double yc = trc(o.Y) + q.dcoll, Dx = Dnow + q.ddebt; bool rz = false;
                if (P.resync == 1 && q.coll > 0 && q.debt > 0) { rz = true; yc = q.coll; Dx = q.debt; }   // the chain's state after the action is the truth
                if (P.resync && q.coll <= 0) { close(li); continue; }                      // no collateral left on chain: nothing to keep here
                if (Dx <= 1e-9) { close(li); continue; }
                if (yc <= 0) { n_ignored++; ev(q, "ADJ_YC0", q.dcoll, 0); continue; }
                if (ctl && q.dcoll == 0 && q.ddebt < 0 && o.n1 <= (g_shared == 5 ? g_act_m : n_act_of(ema, Cl))) {   // _repay_partial: ns[0] <= active_band -> debt only, bands kept (DL on: the band of the AMM's price)
                    if (o.x_down_sum() * (1.0 - ld_at(t)) / Dx < 1.0) { n_ignored++; n_revert++; n_rep_unh++; ev(q, "REP_UNH", -q.ddebt, 0); continue; }
                    o.Dn = Dx / g_base; continue;
                }   // _repay_partial: ns[0] <= active_band -> debt only, bands kept
                int N = q.n2 - q.n1 + 1;
                int f = n1_formula(yc, Dx, N, t);
                int a = q.n1 + (f - q.nf);
                // the Controller caps n1 at n0 + 1024 - N (debt repaid to dust): the real n1 carries no formula offset then
                if (q.nf - q.n1 > 20) a = std::min(f, n_act_of(ema, Cl) + 1024 - N);
                if (P.resync == 1 && q.coll > 0 && q.debt > 0) a = q.n1;                 // the Controller's own band placement
                if (P.resync == 2 && (q.ddebt > 0 || q.dcoll < 0) && g_base * std::pow(k, a) >= ema) {   // "Debt too high" at the simulated oracle:
                    // the borrower gets what the simulated Controller allows (max borrowable / min collateral), else the action reverts
                    const int a_min = int(std::floor(std::log(ema / g_base) / std::log(k))) + 1, off = a - f;
                    double s = 0, kk = 1; for (int j = 0; j < N; j++) { s += kk; kk *= k; }
                    const double per = (1.0 - W[week_at(t)].ldisc) / (sbr * N) * s * base_at(t), need = std::exp(lnA * (a_min - off));
                    const double Dlim = yc * per / need * (1.0 - 1e-9), ylim = Dx * need / per * (1.0 + 1e-9);
                    const double Dx0 = Dx, yc0 = yc;
                    if (q.ddebt > 0 && q.dcoll >= 0 && Dlim > Dnow) Dx = std::min(Dx, Dlim);
                    else if (q.dcoll < 0 && q.ddebt <= 0 && ylim < o.Y) yc = std::max(yc, ylim);
                    else { n_ignored++; n_revert++; ev(q, "ADJ_REV_DTH", q.ddebt > 0 ? q.ddebt : -q.dcoll, 0); continue; }
                    a = n1_formula(yc, Dx, N, t) + off;
                    if (g_base * std::pow(k, a) >= ema) { n_ignored++; n_revert++; ev(q, "ADJ_REV_DTH", q.ddebt > 0 ? q.ddebt : -q.dcoll, 0); continue; }
                    n_capped++; if (Dx != Dx0) ev(q, "ADJ_CAP_BORROW", Dx0 - Dnow, Dx - Dnow); else ev(q, "ADJ_CAP_REMOVE", o.Y - yc0, o.Y - yc);
                }
                if (g_shared) {                                          // borrow_more takes less; any other action reverts here
                    int s = shared_n1(a, N, li);
                    if (s > a) {
                        double Ds = Dx * std::pow(k, s - a);
                        if (q.ddebt <= 0 || Ds < Dnow) { n_sh_revert++; ev(q, "ADJ_REV_BNE", Dx, 0); continue; }
                        ev(q, "ADJ_CUT_BNE", Dx - Dnow, Ds - Dnow); sh_forgone += Dx - Ds; n_sh_shift++; sh_maxshift = std::max(sh_maxshift, s - a); Dx = Ds; a = s;
                    }
                }
                if (rz) realise(o);                                      // settled only when the action goes through (was: before the checks)
                o.Dn = Dx / g_base;
                settle7(a, N, li);
                o.ld = ld_at(t); o.place(yc, a, a + N - 1, n_act_of(ema, Cl)); mint7(li);
                if (g_shared == 5 && a <= g_act_m) g_act_m = a - 1;
                continue;
            }
            // sim is in soft liquidation: add/borrow/remove would revert
            if (P.resync == 2 && !(q.dcoll < 0 && q.ddebt < 0)) {       // the contract: add / borrow / remove revert "Underwater", a repay keeps the bands
                if (lq) std::fprintf(lq, "I %ld %d %.6g %.6g %.6g %.6g\n", t, q.uid, q.dcoll, q.ddebt, o.health(), o.X / o.D());   // --resync 2: + converted crvUSD / debt
                if (reopen && (q.dcoll != 0 || q.ddebt > 0)) {        // "Underwater": the borrower closes (full repay, the crvUSD applied) and reopens at the simulated oracle, no buy-back
                    const int N = q.n2 - q.n1 + 1;
                    double Dx = Dnow - o.X + q.ddebt, yc = trc(o.Y) + q.dcoll;
                    if (Dx <= 1e-9) { close(li); n_reopen++; continue; }
                    if (yc <= 0 || N < 1) { n_ignored++; n_revert++; ev(q, "REO_YC0", q.dcoll, 0); continue; }
                    const int f = n1_formula(yc, Dx, N, t); int a = q.n1 + (f - q.nf);
                    if (q.nf - q.n1 > 20) a = std::min(f, n_act_of(ema, Cl) + 1024 - N);
                    if (g_base * std::pow(k, a) >= ema) {             // "Debt too high": the borrow is cut to what the Controller allows, else the reopen is not done
                        const int a_min = int(std::floor(std::log(ema / g_base) / std::log(k))) + 1, off = a - f;
                        double s = 0, kk = 1; for (int j = 0; j < N; j++) { s += kk; kk *= k; }
                        const double per = (1.0 - W[week_at(t)].ldisc) / (sbr * N) * s * base_at(t), need = std::exp(lnA * (a_min - off));
                        const double Dlim = yc * per / need * (1.0 - 1e-9), ylim = Dx * need / per * (1.0 + 1e-9);
                        const double Dx0 = Dx, yc0 = yc;
                        if (q.ddebt > 0 && q.dcoll >= 0 && Dlim > Dnow - o.X) Dx = std::min(Dx, Dlim);
                        else if (q.dcoll < 0 && q.ddebt <= 0 && ylim < o.Y) yc = std::max(yc, ylim);
                        else { n_ignored++; n_revert++; ev(q, "REO_REV_DTH", q.ddebt > 0 ? q.ddebt : -q.dcoll, 0); continue; }
                        a = n1_formula(yc, Dx, N, t) + off;
                        if (g_base * std::pow(k, a) >= ema) { n_ignored++; n_revert++; ev(q, "REO_REV_DTH", q.ddebt > 0 ? q.ddebt : -q.dcoll, 0); continue; }
                        n_capped++; if (Dx != Dx0) ev(q, "REO_CAP_BORROW", Dx0 - (Dnow - o.X), Dx - (Dnow - o.X)); else ev(q, "REO_CAP_REMOVE", o.Y - yc0, o.Y - yc);
                    }
                    if (g_shared) {                                   // "Band not empty": a borrow takes less, anything else is not done
                        const int s5 = shared_n1(a, N, li);
                        if (s5 > a) {
                            const double Ds = Dx * std::pow(k, s5 - a);
                            if (q.ddebt <= 0 || Ds < Dnow - o.X) { n_sh_revert++; n_ignored++; n_revert++; ev(q, "REO_REV_BNE", Dx, 0); continue; }
                            ev(q, "REO_CUT_BNE", Dx, Ds); sh_forgone += Dx - Ds; n_sh_shift++; sh_maxshift = std::max(sh_maxshift, s5 - a); Dx = Ds; a = s5;
                        }
                    }
                    realise(o); o.Dn = Dx / g_base; settle7(a, N, li); o.ld = ld_at(t); o.place(yc, a, a + N - 1, n_act_of(ema, Cl)); mint7(li);
                    if (g_shared == 5 && a <= g_act_m) g_act_m = a - 1;
                    n_reopen++;
                    continue;
                }
                if (q.dcoll != 0 || q.ddebt > 0) { n_ignored++; n_revert++; ev(q, "SL_REV_UNDERWATER", q.ddebt, 0); }
                if (q.ddebt >= 0) continue;
                const double Dx = Dnow + q.ddebt;
                if (Dx <= 1e-9) { close(li); continue; }
                if (ctl && Dx - o.X <= 1e-9) { close(li); continue; }   // repay(): wallet + the bands' crvUSD cover the debt -> _repay_full
                if (ctl && o.x_down_sum() * (1.0 - ld_at(t)) / Dx < 1.0) { n_ignored++; n_revert++; n_rep_unh++; ev(q, "REP_UNH", -q.ddebt, 0); continue; }   // "The action ends with unhealthy state"
                o.Dn = Dx / g_base;
                continue;
            }
            if (q.dcoll < 0 && q.ddebt < 0) {                          // deleverage (repay_extended withdraws everything)
                double sold = std::min(-q.dcoll, o.Y);
                double Dx = Dnow - o.X - sold * Cl;
                if (Dx <= 1e-9) { close(li); continue; }
                double yc = trc(o.Y - sold); int N = o.n2 - o.n1 + 1;
                if (yc <= 0) { n_ignored++; ev(q, "DLV_YC0", q.dcoll, 0); continue; }
                int a = n1_formula(yc, Dx, N, t);
                if (g_shared && shared_n1(a, N, li) > a) { n_sh_revert++; ev(q, "DLV_REV_BNE", q.ddebt, 0); continue; }   // re-deposit blocked: the repay reverts
                realise(o);                                             // the crvUSD repaid, the rest sold at market: the sale is settled
                o.Dn = Dx / g_base; settle7(a, N, li); o.ld = ld_at(t); o.place(yc, a, a + N - 1, n_act_of(ema, Cl)); mint7(li);
                if (g_shared == 5 && a <= g_act_m) g_act_m = a - 1;
                n_conv++;
                continue;
            }
            double rep = std::max(0.0, -q.ddebt) + std::max(0.0, q.dcoll) * Cl;
            if (q.dcoll > 0) n_conv++;
            if (lq) std::fprintf(lq, "I %ld %d %.6g %.6g %.6g\n", t, q.uid, q.dcoll, q.ddebt, o.health());
            if (rep <= 0) { n_ignored++; ev(q, "ADJ_NOREP", 0, 0); continue; }
            double Dx = Dnow - rep;
            if (Dx <= 1e-9) { close(li); continue; }
            o.Dn = Dx / g_base;
        }
        // ---- 3) market: arbitrage (LLAMMA) or nothing (Aave), liquidations, shortfall
        if (g7_dirty) m7_reconcile(others7(-1).get());
        bd_now = 0;
        bool amm_traded = false; double amm_vol = 0;                          // v5: did the AMM trade this minute (-> price_w), crvUSD volume
        double tr_ys_log = 0, tr_ys_done = 0, tr_t1_log = 0, tr_yb_log = 0, tr_yb_done = 0, tr_t2_log = 0, tr_xs_real = 0, tr_xs_eng = 0, tr_xb_real = 0, tr_xb_eng = 0;   // crvUSD legs: real (y x realised price) vs the engine's bands
        double Qsell = 0;
        const std::vector<double>* vgc = nullptr;                        // --venue-grid: the latest quote row at or before t
        if (!g_vg_t.empty()) { long vj = long(std::upper_bound(g_vg_t.begin(), g_vg_t.end(), double(t)) - g_vg_t.begin()) - 1; vgc = &g_vg_c[std::max(0L, vj)]; }
        auto Cv = [&](double q) {                                        // measured cost of selling q (USD) into the venue
            if (q <= 0) return 0.0;
            const auto &S = g_vg_s; const auto &c = *vgc; const size_t K = S.size(); double f;
            if (q <= S[0]) f = c[0] * q / S[0];
            else if (q >= S[K - 1]) f = c[K - 1] + (c[K - 1] - c[K - 2]) * (q - S[K - 1]) / (S[K - 1] - S[K - 2]);
            else { size_t k = 1; while (S[k] < q) k++; const double u = std::log(q / S[k - 1]) / std::log(S[k] / S[k - 1]); f = c[k - 1] + (c[k] - c[k - 1]) * u; }
            return q * std::min(0.99, std::max(0.0, f));
        };
        auto sell_cost = [&](double qq) { return vgc ? Cv(Qsell + qq) - Cv(Qsell) : qq * P.ext_fee + qq * std::min(0.99, Qsell / dep) + qq * qq / (2.0 * dep); };
        auto impact = [&](double q) { if (!vgc) return std::min(0.99, q / dep); if (!(q > 0)) return 0.0; return std::min(0.99, std::max(0.0, (Cv(q * 1.01) - Cv(q * 0.99)) / (0.02 * q))); };   // no flow = no move (0 / 0 would read as the 0.99 cap)
        auto best_slice = [&](double R, double b, double s_c) {           // --venue-grid: the repay that maximises R b - R - sale cost
            double lo = 0.0, hi = R; const double ph = 0.6180339887498949;
            auto f = [&](double r) { return r * (b - 1.0) - sell_cost(r * b * s_c); };
            double x1 = hi - ph * (hi - lo), x2 = lo + ph * (hi - lo), f1 = f(x1), f2 = f(x2);
            for (int it = 0; it < 80; it++) { if (f1 < f2) { lo = x1; x1 = x2; f1 = f2; x2 = lo + ph * (hi - lo); f2 = f(x2); } else { hi = x2; x2 = x1; f2 = f1; x1 = hi - ph * (hi - lo); f1 = f(x1); } }
            const double r = 0.5 * (lo + hi); return f(R) > f(r) ? R : r;
        };
        if (P.aave) {
            const double por = P.aave_oracle ? ema : Cl;
            double gain_floor = -1e300; int n_liq_min = 0;
            if (P.liq_rate > 0) {                                              // the bots: at most liq_rate liquidations a minute, biggest gain first
                std::vector<double> gains;
                for (auto &o : L) {
                    if (!o.alive || o.neg_since < 0 || steps - o.neg_since < P.lat) continue;
                    double Dd = o.D(); if (o.Ya * por * LTo(o) / Dd >= 1.0) continue;
                    gains.push_back(std::min(Dd, o.Ya * por) * LB);
                }
                if (int(gains.size()) > P.liq_rate) { std::nth_element(gains.begin(), gains.begin() + (P.liq_rate - 1), gains.end(), std::greater<double>()); gain_floor = gains[P.liq_rate - 1]; }
            }
            for (auto &o : L) {
                if (!o.alive) continue;
                for (int call = 0; call < 2; call++) {
                    double Dd = o.D();
                    const double LT = LTo(o);
                    double hf = o.Ya * por * LT / Dd;
                    if (hf >= 1.0) { o.neg_since = -1; break; }
                    if (o.neg_since < 0) o.neg_since = steps;
                    if (steps - o.neg_since < P.lat) break;
                    if (P.liq_rate > 0 && (n_liq_min >= P.liq_rate || std::min(Dd, o.Ya * por) * LB < gain_floor || (P.liq_every > 1 && steps % P.liq_every != 0))) break;
                    double R, bmul = 1.0 + LB;
                    if (P.aave == 4) {
                        double bmax = 1.0 + LB, bmin = 1.0 + LB * P.lb_factor;
                        bmul = hf <= P.hf_maxb ? bmax : bmax - (bmax - bmin) * (hf - P.hf_maxb) / (1.0 - P.hf_maxb);
                        double C = o.Ya * por;
                        R = (P.thf * Dd - C * LT) / (P.thf - bmul * LT);
                        R = std::min(std::max(R, 0.0), Dd);
                        if (Dd - R < P.dust || C - R * bmul < P.dust) R = Dd;
                        if (R * bmul > C) R = C / bmul;
                    } else {
                        R = (hf < 0.95 || (P.aave_v33 && (Dd < 2000.0 || o.Ya * por < 2000.0)) ? 1.0 : 0.5) * Dd;
                        if (R * bmul > o.Ya * por) R = o.Ya * por / bmul;
                    }
                    if (P.liq_cap > 0 && R > P.liq_cap) R = P.liq_cap;    // one liquidator's crvUSD per tx
                    const double bl = 1.0 + (bmul - 1.0) * (1.0 - P.aave_fee);   // round_r7: the liquidator's share of the bonus
                    if (P.aave_slice) {                                          // round_r7: the liquidator picks debtToCover <= R
                        const double C = o.Ya * por, sc = Cl / por;
                        auto ok = [&](double r) { return (P.aave == 1 && !P.aave_v33) || r >= Dd * (1.0 - 1e-12) || r * bmul >= C * (1.0 - 1e-12) || (Dd - r >= P.dust && C - r * bmul >= P.dust); };
                        auto pf = [&](double r) { return r * (bl - 1.0) - sell_cost(r * bl * sc); };
                        double r = best_slice(R, bl, sc);
                        if (!ok(r)) {                                            // leftover below $1,000: shrink the slice, or take the maximum
                            const double r2 = std::min(r, std::min(Dd - P.dust, (C - P.dust) / bmul)); const bool o2 = r2 > 0 && ok(r2), oR = ok(R);
                            if (!o2 && !oR) break;
                            r = (o2 && (!oR || pf(r2) >= pf(R))) ? r2 : R;
                        }
                        R = r;
                    }
                    double seize = R * bmul / por;
                    if (seize > o.Ya) { seize = o.Ya; R = o.Ya * por / bmul; }
                    double sold = (P.aave_fee > 0 ? R * bl / por : seize) * Cl, cost = gas_now + sell_cost(sold);
                    if (sold - R - cost <= 0) break;
                    if (call == 0) n_liq_min++;
                    const double hl_b = P.aave_eqcap ? std::min(seize * por - R, std::max(0.0, o.Ya * por - Dd)) : seize * por - R;   // round_r7: the borrower's part
                    o.Dn -= R / g_base; o.Ya -= seize; n_liq++; liq_eq += hl_b; Qsell += sold;
                    if (lq) std::fprintf(lq, "S %ld %d %.2f %.2f %.2f %d\n", t, o.uid, R, seize * por, hl_b, int(o.orphan));
                    if (o.D() <= 1e-6) { realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1; break; }
                    if (o.Ya * por < P.dust && o.D() > 0) {          // collateral gone: v3.3 deficit, written off now
                        bd_real += P.aave_eqcap ? std::max(0.0, o.D() - o.Ya * por) : o.D() - o.Ya * por; n_bd_real++;
                        if (lq) std::fprintf(lq, "B %ld %d %.2f %.2f\n", t, o.uid, o.D(), o.Ya * por);
                        realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1; break;
                    }
                }
                if (!o.alive) continue;
                double val = o.Ya * Cl;
                if (o.Ya <= 1e-12 && o.D() > 0) { /* deficit: collateral gone, debt left */ }
                if (val < o.D()) bd_now += o.D() - val;
            }
        } else {
            std::vector<Loan*> act;
            for (auto &o : L) if (o.alive) { o.p_o = ema; o.po3 = po3; act.push_back(&o); }
            const double Hn = H * (1.0 - P.ext_fee), Ln = Lw * (1.0 + P.ext_fee);
            auto can_up = [&](const Loan &o, double tt) {
                if (o.touched) return true;
                double pf = po3 / (o.pt(o.n1) * o.pt(o.n1));
                return tt * (1.0 - fe) >= pf;
            };
            auto flow_y = [&](const std::vector<Loan*> &set, double tt, int dir) {   // collateral the set sells (dir > 0) / buys back (dir < 0) when moved to price tt
                double qq = 0;
                for (Loan *o : set) { Loan c = *o; c.trade_to_price(tt, fe); c.totals(); double dy = c.Y - o->Y; if (dir > 0 && dy < 0) qq -= dy; if (dir < 0 && dy > 0) qq += dy; }
                return qq;
            };
            if (!TR.empty()) {                                                // trade replay: every real trade of this minute, in chain order
                while (tri < TR.size() && TR[tri].t < t + 60) {
                    const Trade q = TR[tri++];
                    if (q.t < t) continue;
                    if (P.no_dl && q.dir < 0) continue;                      // one-way AMM: nobody can sell collateral back to it
                    const double pr = q.p;                                // realised crvUSD per collateral unit (already in collateral terms)
                    std::vector<Loan*> set;
                    if (q.dir > 0) {                                          // the AMM sold q.y collateral: loans move up to the clearing price
                        for (Loan *o : act) set.push_back(o);
                        double lo = 0.0, hi = std::max(H * 1.003, pr * 1.02), q_hi = flow_y(set, hi, +1), t1;
                        if (q_hi <= q.y) { t1 = hi; tr_short += (q.y - q_hi) * Cl; tr_ys_done += q_hi; }
                        else { for (int it = 0; it < 40; it++) { double m = 0.5 * (lo + hi); if (flow_y(set, m, +1) >= q.y) hi = m; else lo = m; } t1 = hi; tr_ys_done += q.y; }
                        tr_ys_log += q.y; tr_t1_log = t1; tr_xs_real += q.y * pr;
                        for (Loan *o : set) {
                            double X0 = o->X, Y0 = o->Y;
                            g_acc = true; o->trade_to_price(t1, fe); g_acc = false; o->totals();
                            double dx = o->X - X0, dy = o->Y - Y0; tr_xs_eng += dx;
                            if (g_shared == 5 && dy < 0) g_act_m = std::max(g_act_m, o->traded_n);
                            if (dy != 0 || dx != 0) o->touched = true;
                            if (dy != 0 || dx != 0) { amm_traded = true; amm_vol += std::fabs(dx); }
                            double loss = -(dx + dy * H); o->sl_acc += loss; o->sold_u -= dy; o->sold_x += dx; slm += -(dx + dy * Cl);
                            if (dy < 0) { sl += loss; Qsell += -dy * t1; sold_sl += -dy * H; } else dl += loss;
                        }
                    } else {                                                  // the AMM bought q.y collateral back: loans move down to the clearing price
                        for (Loan *o : act) if (o->X > 1e-12) set.push_back(o);
                        double lo = std::min(Lw * 0.997, pr * 0.98), hi = std::max(H * 1.003, pr * 1.02), q_lo = flow_y(set, lo, -1), t2;
                        if (q_lo <= q.y) { t2 = lo; tr_short += (q.y - q_lo) * Cl; tr_yb_done += q_lo; }
                        else { for (int it = 0; it < 40; it++) { double m = 0.5 * (lo + hi); if (flow_y(set, m, -1) >= q.y) lo = m; else hi = m; } t2 = lo; tr_yb_done += q.y; }
                        tr_yb_log += q.y; tr_t2_log = t2; tr_xb_real += q.y * pr;
                        for (Loan *o : set) {
                            double X0 = o->X, Y0 = o->Y;
                            g_acc = true; o->trade_to_price(t2, fe); g_acc = false; o->totals();
                            double dx = o->X - X0, dy = o->Y - Y0; tr_xb_eng -= dx;
                            if (dy != 0 || dx != 0) { amm_traded = true; amm_vol += std::fabs(dx); }
                            double loss = -(dx + dy * Lw); o->sl_acc += loss; o->sold_u -= dy; o->sold_x += dx; slm += -(dx + dy * Cl);
                            if (dy < 0) { sl += loss; sold_sl += -dy * Lw; } else dl += loss;
                        }
                    }
                }
            }
            std::vector<Loan*> up, dn;
            if (TR.empty()) { for (Loan *o : act) if (can_up(*o, Hn) && (!o->touched || o->get_p() < Hn)) up.push_back(o); }
            auto flow = [&](const std::vector<Loan*> &set, double tt, int dir) {
                double qq = 0;
                for (Loan *o : set) {
                    Loan c = *o; c.trade_to_price(tt, fe); c.totals();
                    double dy = c.Y - o->Y;
                    if (dir > 0 && dy < 0) qq -= dy * tt;
                    if (dir < 0 && dy > 0) qq += dy * tt;
                }
                return qq;
            };
            double t1 = Hn;
            if (!up.empty()) {
                double q0 = flow(up, Hn, +1);
                if (q0 > (vgc ? 1.0 : dep * 1e-4)) {
                    double lo = Hn * (1.0 - impact(q0)), hi = Hn;
                    for (int it = 0; it < 30; it++) {
                        double m = 0.5 * (lo + hi), g = m - Hn * (1.0 - impact(flow(up, m, +1)));
                        if (g > 0) hi = m; else lo = m;
                    }
                    t1 = 0.5 * (lo + hi);
                }
            }
            if (P.arb_gas > 0 && !up.empty()) {                              // --arb-gas: no sale unless the arbitrageur's gain pays the gas
                double gain = 0;
                for (Loan *o : up) { Loan c = *o; c.trade_to_price(t1, fe); c.totals(); const double dx = c.X - o->X, dy = c.Y - o->Y; if (!(P.no_dl && dy > 0)) gain += -(dx + dy * H); }
                if (gain < gas_now / (1.07 * std::max(P.gas_units, 1e-9)) * P.arb_gas * P.gas_units) { up.clear(); n_arb_skip++; }
            }
            for (Loan *o : up) {
                double X0 = o->X, Y0 = o->Y;
                Loan bak; if (P.no_dl) bak = *o;
                g_acc = true; o->trade_to_price(t1, fe); g_acc = false; o->totals();
                double dx = o->X - X0, dy = o->Y - Y0;
                if (P.no_dl && dy > 0) { *o = bak; continue; }               // that would have been a buy-back
                if (g_shared == 5 && dy < 0) g_act_m = std::max(g_act_m, o->traded_n);   // the market's active band: the deepest band a sale reached
                if (dy != 0 || dx != 0) { amm_traded = true; amm_vol += std::fabs(dx); }
                if (dy != 0 || dx != 0) o->touched = true;
                double loss = -(dx + dy * H); o->sl_acc += loss; o->sold_u -= dy; o->sold_x += dx; slm += -(dx + dy * Cl);
                if (dy < 0) { sl += loss; Qsell += -dy * t1; sold_sl += -dy * H; } else dl += loss;
            }
            if (TR.empty() && !P.no_dl) { for (Loan *o : act) if (o->touched && o->get_p() > Ln) dn.push_back(o); }
            double t2 = Ln;
            if (!dn.empty()) {
                double q0 = flow(dn, Ln, -1);
                if (q0 > (vgc ? 1.0 : dep * 1e-4)) {
                    double lo = Ln, hi = Ln * (1.0 + impact(q0));
                    for (int it = 0; it < 30; it++) {
                        double m = 0.5 * (lo + hi), g = m - Ln * (1.0 + impact(flow(dn, m, -1)));
                        if (g > 0) hi = m; else lo = m;
                    }
                    t2 = 0.5 * (lo + hi);
                }
            }
            for (Loan *o : dn) {
                double X0 = o->X, Y0 = o->Y;
                g_acc = true; o->trade_to_price(t2, fe); g_acc = false; o->totals();
                double dx = o->X - X0, dy = o->Y - Y0;
                if (dy != 0 || dx != 0) { amm_traded = true; amm_vol += std::fabs(dx); }
                double loss = -(dx + dy * Lw); o->sl_acc += loss; o->sold_u -= dy; o->sold_x += dx; slm += -(dx + dy * Cl);
                if (dy < 0) { sl += loss; sold_sl += -dy * Lw; } else dl += loss;
            }
            if (wo_on && amm_traded) {                                        // v5: AMM._exchange -> price_w() (same value as this minute's view)
                WO.price_w(t_now, PE.view(t_now), PE.ma(t_now));
                if (t_now > 1735689600L) wo_n_w25++;
                if (wlog) std::fprintf(wlog, "%ld %.2f\n", t_now, amm_vol);
            } else if (wo_on && P.wo_keep > 0 && wo_n_min % P.wo_keep == 0) { WO.price_w(t_now, PE.view(t_now), PE.ma(t_now)); wo_n_keep++; }
            double gain_floor = -1e300; int n_liq_min = 0;
            if (P.liq_rate > 0) {                                              // the bots: at most liq_rate liquidations a minute, biggest gain first
                std::vector<double> gains;
                for (Loan *op : act) {
                    Loan &o = *op;
                    if (o.neg_since < 0 || steps - o.neg_since < P.lat || o.health_fast(ema) >= 0) continue;
                    gains.push_back(o.X + o.Y * Cl - o.D());
                }
                if (int(gains.size()) > P.liq_rate) { std::nth_element(gains.begin(), gains.begin() + (P.liq_rate - 1), gains.end(), std::greater<double>()); gain_floor = gains[P.liq_rate - 1]; }
            }
            int n_cov_wait = 0;
            if (P.auto_close) for (Loan *op : act) if (op->cov_since >= 0) n_cov_wait++;
            for (Loan *op : act) {
                if (g7_dirty) m7_reconcile(others7(-1).get());       // --m7v 3: a liquidation in this loop changed a band with phantom x: re-cut before the next loan
                Loan &o = *op; reb_site = 2;
                const double bounty = std::min(P.auto_bounty_cap, std::max(P.auto_bounty, P.auto_bounty_pct * o.D()));
                if (P.auto_close && o.X >= o.D() + bounty && o.D() > 0) {     // self-covered (incl. the caller's bounty): close it, nothing more to lose
                    if (o.cov_since < 0) o.cov_since = steps;
                    double need = P.auto_delay;
                    if (!P.auto_lat.empty()) {                                 // the bots: delay from the bounty / gas ratio and the queue, fitted on the chain's liquidations
                        const double gas_close = gas_now / std::max(P.gas_units, 1e-9) * 0.96;   // ~0.96M gas per close at the hour's gas price
                        const double r = bounty / std::max(gas_close, 1e-9);
                        int b = r < 1 ? 0 : r < 3 ? 1 : r < 10 ? 2 : r < 30 ? 3 : r < 100 ? 4 : 5;
                        need = P.auto_lat[std::min<size_t>(b, P.auto_lat.size() - 1)];
                        if (P.auto_lat.size() > 6 && n_cov_wait >= 15) need = std::max(need, P.auto_lat[6]);
                    }
                    const bool ok_delay = steps - o.cov_since >= need;
                    const bool ok_mode = P.auto_mode == 0 || o.health_fast(ema) < 0;
                    if (ok_delay && ok_mode) {
                        reb_site = 3;
                        if (lq) std::fprintf(lq, "C %ld %d %.2f %.2f %.8g %.2f %ld\n", t, o.uid, o.D(), o.X, o.Y, bounty, steps - o.cov_since);
                        bounty_paid += bounty; auto_wait_sum += steps - o.cov_since;
                        realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1; n_autoclose++; continue;
                    }
                } else o.cov_since = -1;
                double h = o.health_fast(ema);
                double val = o.X + o.Y * Cl, Dd = o.D();
                const bool allowed = P.liq_rate <= 0 || (n_liq_min < P.liq_rate && val - Dd >= gain_floor && (P.liq_every <= 1 || steps % P.liq_every == 0));
                const bool deep = P.conv_liq > 0 && val > 0 && o.X / val > P.conv_liq && val > Dd;   // converted past the threshold: close it out now rather than bleed on
                {
                    const bool inr = ema < o.pt(o.n1), conv = o.X > 1e-9 * Dd;
                    if (inr) { o.rng_min++; rng_lmin++; rng_dmin += Dd; if (o.first_rng < 0) { o.first_rng = t; o.d_first = Dd; } }
                    if (conv) { o.sl_min++; sl_lmin++; sl_dmin += Dd; }
                    if (val > 0) o.maxconv = std::max(o.maxconv, o.X / val);
                }
                if (h < 0 || (deep && P.partial)) {
                    if (o.neg_since < 0) o.neg_since = steps;
                    if (steps - o.neg_since >= P.lat && !P.noliq && P.partial && P.v2c && allowed) {
                        // v2 concept Controller.liquidate with a liquidator's frac: repay at most frac_max x D, get f_remove x frac / frac_max
                        // of the position (valued here at the market: what the liquidator can sell it for); depth / gas / sell-cost
                        // model of the liquidator as in the branch below
                        n_liq_min++;
                        for (int call = 0; call < 3 && o.alive; call++) {
                            double hh = call == 0 ? h : o.health();
                            if (hh >= 0) break;
                            double Dd = o.D(), Vm = o.X + o.Y * Cl, Vo = o.X + o.Y * ema;
                            if (Vm <= 0) break;
                            const PLq q = pl_v2c(o.x_down_sum(), Dd, o.ld, Vo);
                            const double R = q.frac_max * Dd, beff = q.f_remove * Vm / R;   // market value received per unit repaid
                            if (!(beff > 1.0)) break;
                            double s_c = std::max(o.Y * Cl / Vm, 1e-12);
                            double q_opt = dep * (beff - 1.0) / (beff * s_c) - Qsell;
                            double Rx = vgc ? best_slice(R, beff, s_c) : std::min(R, q_opt > 0 ? q_opt / (beff * s_c) : 0.0);
                            if (Rx <= 0) break;
                            double qq = Rx * beff * s_c, prof = Rx * (beff - 1.0) - gas_now - sell_cost(qq);
                            if (prof <= 0) break;
                            double phi = Rx >= Dd ? 1.0 : std::min(1.0, q.f_remove * Rx / R);   // final_debt == 0 -> f_remove = WAD
                            if (o.conv_liq < 0) o.conv_liq = o.X / Vm;
                            o.nliq_l++;
                            const double h_spot = spot_health(o, Cl);
                            realise_part(o, phi); o.scale(1.0 - phi); o.Dn -= std::min(Rx, Dd) / g_base; o.touched = true;
                            n_liq++; liq_eq += phi * Vm - Rx; Qsell += qq;
                            if (lq) std::fprintf(lq, "S %ld %d %.2f %.2f %.2f %d %.8g %.8g %d %d %.8g %.6g %.6g %.8g %.8g %.2f\n", t, o.uid, Rx, phi * Vm, phi * Vm - Rx, int(o.orphan), o.Y, o.X, o.n1, o.n2, ema, hh, h_spot, ema, Cl, o.sl_acc);
                            if (o.D() <= 1e-6) { realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1; break; }
                            // no dust write-off: after a capped liquidation >= MIN_DEBT_LEFT stays a normal loan (health +49.53 %)
                        }
                        if (!o.alive) continue;
                        val = o.X + o.Y * Cl; Dd = o.D();
                    } else if (steps - o.neg_since >= P.lat && !P.noliq && P.partial && allowed) {
                        n_liq_min++;
                        const double tgt = P.thf - 1.0;
                        for (int call = 0; call < 3 && o.alive; call++) {
                            double hh = call == 0 ? h : o.health();
                            if (hh >= 0 && !deep) break;
                            double Dd = o.D(), V = o.X + o.Y * Cl;
                            double bmax = 1.0 + LB, bmin = 1.0 + LB * P.lb_factor, hf = 1.0 + hh;
                            double b = hf <= P.hf_maxb ? bmax : bmax - (bmax - bmin) * (hf - P.hf_maxb) / (1.0 - P.hf_maxb);
                            if (V <= 0) break;
                            double Xd = (hh + 1.0) * Dd;                          // x_down * (1 - ld)
                            double R = ((1.0 + tgt) * Dd - Xd) / ((1.0 + tgt) - b * Xd / V);
                            if (!(R > 0) || R * b >= V || Dd - R < P.dust || deep) R = std::min(Dd, V / b);   // deep: repay everything at the bonus, the borrower keeps the rest
                            double s_c = std::max(o.Y * Cl / V, 1e-12);            // collateral share of what is seized
                            double q_opt = dep * (b - 1.0) / (b * s_c) - Qsell;
                            double Rx = vgc ? best_slice(R, b, s_c) : std::min(R, q_opt > 0 ? q_opt / (b * s_c) : 0.0);
                            if (Rx <= 0) break;
                            double qq = Rx * b * s_c, prof = Rx * (b - 1.0) - gas_now - sell_cost(qq);
                            if (prof <= 0) break;
                            double phi = std::min(1.0, Rx * b / V);
                            if (o.conv_liq < 0) o.conv_liq = o.X / V;
                            o.nliq_l++;
                            const double h_spot = spot_health(o, Cl);            // before the state changes
                            realise_part(o, phi); o.scale(1.0 - phi); o.Dn -= Rx / g_base; o.touched = true;
                            n_liq++; liq_eq += Rx * b - Rx; Qsell += qq;
                            if (lq) std::fprintf(lq, "S %ld %d %.2f %.2f %.2f %d %.8g %.8g %d %d %.8g %.6g %.6g %.8g %.8g %.2f\n", t, o.uid, Rx, Rx * b, Rx * (b - 1.0), int(o.orphan), o.Y, o.X, o.n1, o.n2, ema, hh, h_spot, ema, Cl, o.sl_acc);
                            if (o.D() <= 1e-6) { realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1; break; }
                            if (o.X + o.Y * Cl < P.dust) {                          // nothing left to seize: write off the rest
                                bd_real += o.D() - (o.X + o.Y * Cl); n_bd_real++;
                                if (lq) std::fprintf(lq, "B %ld %d %.2f %.2f\n", t, o.uid, o.D(), o.X + o.Y * Cl);
                                realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1; break;
                            }
                        }
                        if (!o.alive) continue;
                        val = o.X + o.Y * Cl; Dd = o.D();
                    } else if (steps - o.neg_since >= P.lat && !P.noliq && allowed) {
                        double sold = o.Y * Cl, cost = gas_now + sell_cost(sold);
                        n_liq_min++;
                        if (P.liq_cap > 0 && Dd - o.X > P.liq_cap) {          // more crvUSD than one liquidator brings: a slice (liquidate_extended)
                            // frac of the debt repaid, f_remove(frac) of the bands withdrawn, f_remove = ((1 + ld/2) / (1 + ld) (1 - frac) + frac) frac;
                            // the liquidator pays frac D - f_remove X: frac solves that = liq_cap
                            auto f_rem = [&](double fr) { return fr >= 1.0 ? 1.0 : ((1.0 + o.ld / 2) / (1.0 + o.ld) * (1.0 - fr) + fr) * fr; };
                            double lo = 0.0, hi = 1.0;
                            for (int it = 0; it < 50; it++) { double m = 0.5 * (lo + hi); if (m * Dd - f_rem(m) * o.X > P.liq_cap) hi = m; else lo = m; }
                            double fr = lo, fm = f_rem(fr), Rx = fr * Dd, seized = fm * val, sold_p = fm * o.Y * Cl;
                            if (Rx > 0 && seized - Rx - gas_now - sell_cost(sold_p) > 0) {
                                if (o.conv_liq < 0) o.conv_liq = val > 0 ? o.X / val : 0;
                                o.nliq_l++;
                                const double h_spot = spot_health(o, Cl);
                                realise_part(o, fm); o.scale(1.0 - fm); o.Dn -= Rx / g_base; o.touched = true;
                                n_liq++; liq_eq += seized - Rx; Qsell += sold_p;
                                if (lq) std::fprintf(lq, "S %ld %d %.2f %.2f %.2f %d %.8g %.8g %d %d %.8g %.6g %.6g %.8g %.8g %.2f\n", t, o.uid, Rx, seized, seized - Rx, int(o.orphan), o.Y, o.X, o.n1, o.n2, ema, h, h_spot, ema, Cl, o.sl_acc);
                                if (o.D() <= 1e-6) { realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1; continue; }
                                val = o.X + o.Y * Cl; Dd = o.D();
                            }
                        } else if (val - Dd - cost > 0) {
                            if (o.conv_liq < 0) o.conv_liq = val > 0 ? o.X / val : 0;
                            o.nliq_l++;
                            const double h_spot = spot_health(o, Cl);
                            realise(o); o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); if (cur[o.uid] >= 0 && &L[cur[o.uid]] == &o) cur[o.uid] = -1;
                            n_liq++; liq_eq += val - Dd; Qsell += sold;
                            if (lq) std::fprintf(lq, "S %ld %d %.2f %.2f %.2f %d %.8g %.8g %d %d %.8g %.6g %.6g %.8g %.8g %.2f\n", t, o.uid, Dd, val, val - Dd, int(o.orphan), o.Y, o.X, o.n1, o.n2, ema, h, h_spot, ema, Cl, o.sl_acc);
                            continue;
                        }
                    }
                } else o.neg_since = -1;
                if (val < Dd) bd_now += Dd - val;
            }
        }
        if (lq && !TR.empty() && (tr_ys_log > 0 || tr_yb_log > 0))
            std::fprintf(lq, "T %ld %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.6g %.8g %.8g %.8g %.8g\n", t, tr_ys_log, tr_ys_done, tr_t1_log, H, tr_yb_log, tr_yb_done, tr_t2_log, Lw, ema, fe, tr_xs_real, tr_xs_eng, tr_xb_real, tr_xb_eng);
        if (lq && !P.dump_at.empty() && std::find(P.dump_at.begin(), P.dump_at.end(), t) != P.dump_at.end())
        {
            std::map<int, std::pair<double, double>> agg;
            for (auto &o : L) if (o.alive) { o.p_o = ema; o.po3 = po3; std::fprintf(lq, "D %ld %d %.8g %.8g %.2f %d %d %.6g %d %d %.2f\n", t, o.uid, o.X, o.Y, o.D(), o.n1, o.n2, o.health(), o.active, int(o.orphan), o.sl_acc); for (int n = o.n1; n <= o.n2; n++) { agg[n].first += o.x(n); agg[n].second += o.y(n); } }
            for (auto &kv : agg) std::fprintf(lq, "N %ld %d %.8g %.8g %.6g\n", t, kv.first, kv.second.first, kv.second.second, ema);
        }
        if (P.orphan == 3) for (auto &o : L) {
            if (!o.alive || !o.orphan) continue;
            o.p_o = ema; o.po3 = po3;
            bool ok = P.aave ? (o.Ya * (P.aave_oracle ? ema : Cl) * LTo(o) >= o.D()) : (o.X <= 1e-9 * o.D() && o.health() >= 0);
            if (ok) { o.alive = false; if (g_ev >= 3) std::fprintf(stderr, "DEATH %d %ld %d %d\n", mkt, t, o.uid, __LINE__); n_orph_closed++; if (!o.ph.empty()) g7_dirty = true; }
        }
        bd_peak = std::max(bd_peak, bd_now);
        if (P.trace_uid >= 0 && t >= P.trace_from && t <= P.trace_to && lq)
            for (auto &o : L) if (o.uid == P.trace_uid && o.alive) {
                o.p_o = ema; o.po3 = po3;
                std::fprintf(lq, "T %ld %.4f %.4f %.4f %.4f %.6f %.8g %.8g %.2f %.6f %d %d %d %.2f\n", t, H, Lw, Cl, ema, fe, o.Y, o.X, o.D(), P.aave ? 0.0 : o.health(), o.active, o.n1, o.n2, o.sl_acc);
            }
        long hr = t / 3600;
        if (hr != hour) { if (bdf && bd_hmax > 0) std::fprintf(bdf, "%ld %.2f\n", hour * 3600, bd_hmax); hour = hr; bd_hmax = 0; }
        bd_hmax = std::max(bd_hmax, bd_now);
        steps++;
        Cl_prev = Cl;
        // compact dead loans now and then (keeps the per-minute loops short)
        if ((i & 4095) == 0) {
            std::vector<Loan> keep; keep.reserve(L.size());
            std::vector<int> remap(L.size(), -1);
            for (size_t j = 0; j < L.size(); j++) {
                if (L[j].alive) { remap[j] = int(keep.size()); keep.push_back(std::move(L[j])); continue; }
                const Loan &o = L[j];
                if (lq) std::fprintf(lq, "E %d %ld %ld %.5f %d %.5f %ld %.2f\n", o.uid, o.rng_min, o.sl_min, o.maxconv, o.nliq_l, o.conv_liq, o.first_rng, o.d_first);
            }
            for (auto &c : cur) if (c >= 0) c = remap[c];
            for (auto &c : orph) if (c >= 0) c = remap[c];
            L.swap(keep);
            L.reserve(L.size() + AC.size() - ai + 16);
        }
    }
    if (bdf && bd_hmax > 0) std::fprintf(bdf, "%ld %.2f\n", hour * 3600, bd_hmax);
    flush_week(wk);
    for (const Loan &o : L) if (lq) std::fprintf(lq, "E %d %ld %ld %.5f %d %.5f %ld %.2f\n", o.uid, o.rng_min, o.sl_min, o.maxconv, o.nliq_l, o.conv_liq, o.first_rng, o.d_first);
    int werr = std::ferror(out) || (lq && std::ferror(lq)) || (bdf && std::ferror(bdf));   // round r3c: the disk filled up during runs -> lost buffered lines; detect it
    werr |= std::fclose(out) != 0; if (lq) werr |= std::fclose(lq) != 0; if (bdf) werr |= std::fclose(bdf) != 0;
    if (!OD.empty()) {
        FILE *df = std::fopen(P.oracle_dump.c_str(), "wb");
        if (!df || std::fwrite(OD.data(), 8, OD.size(), df) != OD.size()) { std::fprintf(stderr, "cannot write %s\n", P.oracle_dump.c_str()); std::exit(2); }
        std::fclose(df);
    }
    if (g_m7 && g_m7v >= 2) std::fprintf(stderr, "mkt %d: mode 7 r3b: generation closes %ld, chain ends %ld, deposits into bands holding x %ld, crvUSD credited $%.0f, collateral credited %.6g\n", mkt, g7_close, g7_chain, g7_into, g7_xcredit, g7_ycredit);
    if (g_m7) std::fprintf(stderr, "mkt %d: mode 7: placements whose range held others' crvUSD %ld, band holdings moved to credit %ld, crvUSD moved $%.0f\n", mkt, n_m7_place_x, n_m7_settle, m7_credit);
    if (g_shared) std::fprintf(stderr, "mkt %d: shared AMM %d: placements checked %ld, shifted %ld (max %d bands), forgone debt $%.0f, reverted %ld\n", mkt, g_shared, n_sh_place, n_sh_shift, sh_maxshift, sh_forgone, n_sh_revert);
    if (P.arb_gas > 0) std::fprintf(stderr, "mkt %d: arb-gas %.3g M gas: minutes with the AMM sale skipped %ld\n", mkt, P.arb_gas, n_arb_skip);
    if (g_shared == 5) std::fprintf(stderr, "mkt %d: creates Band not empty %ld (asked >= 0.97 x max_borrowable %ld), taken less debt $%.0f\n", mkt, n_cr_band, n_cr_band97, cr_band_forgone);
    if (g_shared == 5 || reopen) std::fprintf(stderr, "mkt %d: mode 5 / reopen: reopens %ld, creates cut at Debt too high %ld, creates reverted %ld, repays refused unhealthy %ld, active band at the end %d\n", mkt, n_reopen, n_cr_cut, n_cr_revert, n_rep_unh, g_act_m);
    std::fprintf(stderr, "mkt %d: pre-liquidatable at start %ld, orphans closed %ld, actions ignored %ld, converted %ld, trade replay short $%.0f, loans re-taken from the chain %ld, auto-closed %ld, bounties $%.0f, mean wait %.1f min\n", mkt, n_pre, n_orph_closed, n_ignored, n_conv, tr_short, n_resync_new, n_autoclose, bounty_paid, n_autoclose ? auto_wait_sum / n_autoclose : 0.0);
    if (P.resync == 2) std::fprintf(stderr, "mkt %d: resync 2: actions the contract would revert %ld (in 'actions ignored'), borrows / removals cut to the simulated limit %ld\n", mkt, n_revert, n_capped);
    std::fprintf(stderr, "mkt %d: vs-holding booked at chain events %.0f, by the liquidator %.0f, at auto-close %.0f\n", mkt, rebs[1], rebs[2], rebs[3]);
    if (P.wo) {
        std::fprintf(stderr, "mkt %d: window oracle writes %ld in %ld minutes (%.4f %% of blocks at 5 per minute); from 1 Jan 2025: %ld in %ld minutes (%.4f %%)\n", mkt, WO.n_writes, wo_n_min,
                     wo_n_min ? 100.0 * WO.n_writes / (5.0 * wo_n_min) : 0.0, wo_n_w25, wo_n_min25, wo_n_min25 ? 100.0 * wo_n_w25 / (5.0 * wo_n_min25) : 0.0);
        if (P.wo_keep > 0) std::fprintf(stderr, "mkt %d: keeper-only writes %ld (every %ld oracle minutes, not counted above)\n", mkt, wo_n_keep, P.wo_keep);
        if (P.wo_sub) std::fprintf(stderr, "mkt %d: real swap-block writes %ld (from 1 Jan 2025: %ld) of %zu swap blocks in the file\n", mkt, wo_n_sw, wo_n_sw25, SWT.size());
        std::fprintf(stderr, "mkt %d: band open at minute reads, all %ld: >0 %ld >1e-6 %ld >1e-4 %ld >1e-3 %ld; from 1 Jan 2025 %ld: >0 %ld >1e-6 %ld >1e-4 %ld >1e-3 %ld\n", mkt,
                     wo_nr, wo_open[0], wo_open[1], wo_open[2], wo_open[3], wo_nr25, wo_open25[0], wo_open25[1], wo_open25[2], wo_open25[3]);
        if (wlog) std::fclose(wlog);
    }
    std::fflush(stderr);
    if (werr || std::ferror(stderr)) { std::fprintf(stderr, "WRITE_ERROR mkt %d\n", mkt); return 3; }
    return 0;
}
