#!/bin/zsh
# round r3c engine copy: + CDEC8="4 12 13 18" (--cdec 8 for those markets: WBTC / LBTC / cbBTC / WBTC), ASON=file (--as-on), BIN required# run_v5.sh TAG [engine args...] -> <OUTROOT>/TAG/{w,l,b}_<mkt>.txt (+ wo_<mkt>.txt with WO_LOG=1), all 21 markets, nice -n 19,
# at most 6 workers. = ../engine/run_v4.sh (same env hooks ORA_SERIES / ORA_SERIES_MKT / TRADES_MKT / ORA_DUMP / DEPTH_* /
# CRV_CANDLES, BIN defaults to the v5 engine of this folder) plus
#   WO_POOL=pattern  --wo-pool $(printf pattern und) per market (und from the mkt file: 0 ETH, 1 BTC, 2 CRV); give --wo 1 yourself
#   WO_LOG=1         --wo-log OUTROOT/TAG/wo_<mkt>.txt (one line per oracle write)
#   OUTROOT=dir      where the run folder goes (default LLAMMA-Design/engine/runs; the bit-exact checks use engine2/checks)
# Runs from LLAMMA-Design/engine (relative inputs).
# fix/engine copy (23 Oct 2026): + WO_SWAPS=pattern  --wo-swaps $(printf pattern mkt) per market (fix/engine/swaps/sw_%d.f64)
HERE=${LLAMMA_DESIGN:-.}/v2/verify/engine2
cd $HERE/../../../engine || exit 1
BIN=${BIN:?}
OUTROOT=${OUTROOT:-runs}
TAG=$1; shift
[[ -e $OUTROOT/$TAG ]] && { echo "$OUTROOT/$TAG exists, not overwriting"; exit 1; }
mkdir -p $OUTROOT/$TAG
NW=${NW:-6}; (( NW > 6 )) && NW=6
T_ALL=$(date +%s)
for i in $(seq 0 20); do
  und=$(head -1 mkt_$i.txt | awk '{print $3}')
  case $und in 0) C=../candles_0.bin; D=${DEPTH_EB:-2e8};; 1) C=../candles_1.bin; D=${DEPTH_EB:-2e8};; 2) C=${CRV_CANDLES:-../candles_2_onchain.bin}; D=${DEPTH_CRV:-2e7};; esac
  X=""
  [[ -n "$ORA_SERIES" ]] && X="$X --oracle-series $(printf "$ORA_SERIES" $und)"
  [[ -n "$ORA_SERIES_MKT" ]] && X="$X --oracle-series $(printf "$ORA_SERIES_MKT" $i)"
  [[ -n "$TRADES_MKT" ]] && X="$X --trades $(printf "$TRADES_MKT" $i)"
  [[ -n "$ORA_DUMP" ]] && X="$X --oracle-dump $OUTROOT/$TAG/o_$i.f64"
  [[ -n "$WO_POOL" ]] && X="$X --wo-pool $(printf "$WO_POOL" $und)"
  [[ -n "$WO_SWAPS" ]] && X="$X --wo-swaps $(printf "$WO_SWAPS" $i)"
  [[ -n "$CDEC8" && " $CDEC8 " == *" $i "* ]] && X="$X --cdec 8"
  [[ -n "$ASON" ]] && X="$X --as-on $ASON"
  [[ -n "$WO_LOG" ]] && X="$X --wo-log $OUTROOT/$TAG/wo_$i.txt"
  echo "T0=\$(date +%s); $BIN --in mkt_$i.txt --candles $C --depth $D --out $OUTROOT/$TAG/w_$i.txt --liq $OUTROOT/$TAG/l_$i.txt --bd $OUTROOT/$TAG/b_$i.txt $X $*; rc=\$?; echo \"time mkt_$i \$(( \$(date +%s) - T0 )) s rc \$rc\" >&2"
done | xargs -P $NW -S 4096 -I{} nice -n 19 zsh -c "{}" 2> $OUTROOT/$TAG/log.txt
echo "wall all 21 markets $(( $(date +%s) - T_ALL )) s ($NW workers)" >> $OUTROOT/$TAG/log.txt
echo "cmd: BIN=$BIN ORA_SERIES=$ORA_SERIES WO_POOL=$WO_POOL WO_SWAPS=$WO_SWAPS run_v5s.sh $TAG $*" >> $OUTROOT/$TAG/log.txt
tail -2 $OUTROOT/$TAG/log.txt
