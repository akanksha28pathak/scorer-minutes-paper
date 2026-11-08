#!/usr/bin/env bash
# Scorer-minutes paper: a sequential speed anchor and scorer reference on the scorer setup itself (Gemini review G0,
# Nov 6: the anchor rested on 5 sessions; the reference was an A100). g31_v12e on ALL 55 train55 tasks on Kaggle
# 4x L4, TP4, ONE session at a time, 7-min cap = exactly how the scorer runs it: a 55-session anchor and a true scorer
# reference on the same tasks as the [L4] converted / nominal pair.
# Budget (Akanksha, Nov 6: "use all my Kaggle quota for this research"): a 31B 4x L4 run is charged ~3.6 GPU h per wall
# hour; this run is ~4.5 h wall, ~16 GPU h; the pair ~7 GPU h; 30 h weekly quota. Waits until the L4 pair has finished
# and launches only if the quota used since the reset leaves room for it (<= MAX_USED, 13.0 h).
# If WSL restarts while it runs on Kaggle, collect later with:
#   NO_PUSH=1 bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/train55.txt g31_v12e_l4seq55 --workers 1 --eval-args=--max-time-minutes=7
#   setsid nohup bash scripts/paper/l4_sequential.sh >> ~/gemma4_work/logs/l4_sequential.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
MAX_USED="${MAX_USED:-13.0}"; L=$HOME/gemma4_work/logs
log() { echo "[$(date '+%F %T')] $*"; }
used() { .venv/bin/kaggle quota 2>/dev/null | awk '$1=="GPU"{sub("h","",$2); print $2}'; }
log "waiting for the quota reset and the L4 pair (logs kaggle_g31_v12e_l4ctl_{conv15,nom7}.log) to finish"
while [ "$(date -u +%s)" -lt "$(date -u -d 2026-11-07T00:15:00Z +%s)" ]; do sleep 60; done
while :; do
  done_n=0
  for t in conv15 nom7; do grep -q 'GPU quota used after' "$L/kaggle_g31_v12e_l4ctl_$t.log" 2>/dev/null && done_n=$((done_n + 1)); done
  [ $done_n -eq 2 ] && break
  sleep 120
done
U=$(used); log "L4 pair finished; GPU quota used since the reset: ${U} h (limit for launching: $MAX_USED h)"
awk "BEGIN{exit !(${U:-99} <= $MAX_USED)}" || { log "not launching: the 30 h weekly quota would run out"; exit 0; }
log "launching the sequential scorer reference (55 tasks, 1 session, 7 min)"
bash scripts/kaggle/eval_31b.sh agents/g31_v12e splits/train55.txt g31_v12e_l4seq55 \
  --workers 1 --eval-args=--max-time-minutes=7 > "$L/kaggle_g31_v12e_l4seq55.log" 2>&1 < /dev/null
log "eval_31b exited $?; quota used now: $(used) h"
