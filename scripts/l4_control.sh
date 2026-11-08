#!/usr/bin/env bash
# Scorer-minutes paper, section 4.3 [L4] control (Akanksha, Nov 6: "run all the jobs you need for the research track
# data collection"): g31_v12e (7 scorer-min cap) on train55 on Kaggle 4x L4 with 10 parallel sessions, twice, in
# parallel (Kaggle allows 2 GPU kernels):
#   * converted: cap 15 min wall (7 x 2.125, eval_31b.sh's default for this setup);
#   * nominal:   cap 7 min wall (the cap not converted), i.e. what a naive rerun on this hardware would use.
# Waits until NOT_BEFORE (the weekly GPU quota resets Sat 00:00 UTC), then launches both. ~4-5 GPU h in total.
#   setsid nohup bash scripts/paper/l4_control.sh [NOT_BEFORE=2026-11-07T00:05:00Z] >> ~/gemma4_work/logs/l4_control.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
NOT_BEFORE="${1:-2026-11-07T00:05:00Z}"; L=$HOME/gemma4_work/logs
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting until $NOT_BEFORE"
while [ "$(date -u +%s)" -lt "$(date -u -d "$NOT_BEFORE" +%s)" ]; do sleep 60; done
log "launching the converted run (15 min wall)"
setsid nohup bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/train55.txt g31_v12e_l4ctl_conv15 \
  > $L/kaggle_g31_v12e_l4ctl_conv15.log 2>&1 < /dev/null &
sleep 120  # do not push two kernels in the same second
log "launching the nominal run (7 min wall)"
setsid nohup bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/train55.txt g31_v12e_l4ctl_nom7 \
  --eval-args=--max-time-minutes=7 > $L/kaggle_g31_v12e_l4ctl_nom7.log 2>&1 < /dev/null &
log "both launched; logs: $L/kaggle_g31_v12e_l4ctl_{conv15,nom7}.log"
