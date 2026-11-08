#!/usr/bin/env bash
# Every number the scorer-minutes paper needs from the PC, in one go (CPU only, no GPU, ~minutes).
#
#   bash scripts/paper/scorer_minutes_pc.sh
#
# 1. regenerates docs/RESULTS.md + docs/TASK_MATRIX.md (CLAUDE.md rule; D24-D29 runs were missing on Nov 5)
# 2. lists every run folder with model-call logs and checks one record for usage.prompt_tokens (prefill split)
# 3. speed table over all logged runs (analysis/speed_table.py; labels from docs/paper/aggregates/run_labels.tsv)
# 4. timing JSON for each logged run (tools/scorer_minutes timing)
# 5. noise table from the regenerated matrix (+ docs/paper/aggregates/extra_runs.csv if present)
# 6. copies the aggregate outputs (no task text, no traces) to docs/paper/aggregates/ for commit
#
# Outputs: results/paper/ (all), docs/paper/aggregates/ (aggregates to commit). Then:
#   git add docs/paper/aggregates docs/RESULTS.md docs/TASK_MATRIX.md && git commit -m "paper: scorer-minutes data" && git push
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
OUT=results/paper; DATA=docs/paper/aggregates
mkdir -p "$OUT/timing" "$DATA"
export PYTHONPATH="$PWD/tools/scorer_minutes/src${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1. regenerate RESULTS.md and TASK_MATRIX.md"
"$PY" analysis/make_results_doc.py
"$PY" analysis/make_task_matrix.py

echo "== 2. runs with model-call logs"
RUNS=()
for d in results/kaggle_runs/*/; do
  d=${d%/}
  case "$d" in *-aborted*) continue;; esac  # partial runs (e.g. the Nov 6 credit outage) are not measurements
  f=$(find "$d/run" -path '*llm_calls/*.jsonl' -print -quit 2>/dev/null || true)
  [ -n "$f" ] && RUNS+=("$d")
done
echo "${#RUNS[@]} logged runs"
[ "${#RUNS[@]}" -gt 0 ] || { echo "no logged runs under results/kaggle_runs"; exit 1; }
"$PY" - "${RUNS[@]}" <<'EOF' | tee "$OUT/prompt_tokens_check.txt"
import json, sys
from pathlib import Path
have = miss = 0
for run in sys.argv[1:]:
    f = next(Path(run, "run").glob("**/llm_calls/*.jsonl"), None)
    rec = json.loads(f.open().readline())
    u = ((rec.get("response") or {}).get("usage") or {})
    ok = isinstance(u.get("prompt_tokens"), int)
    have += ok; miss += not ok
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"{Path(run).name}\tprompt_tokens={'yes' if ok else 'NO'}\tcached_tokens={cached}")
print(f"# runs with prompt_tokens: {have}, without: {miss}")
EOF

echo "== 3. speed table"
# factors against the logged sequential 4x L4 run (the real scorer setup, grant review Nov 6); the A100 proxy
# (31.6 tok/s) is kept as a secondary table
REF=${REF:-results/kaggle_runs/gemma4-eval-g31-v11m-seq5}
"$PY" analysis/speed_table.py "${RUNS[@]}" --reference "$REF" --cap-scorer-min 7 --out "$OUT/speed_table.csv" | tee "$OUT/speed_table.md"
"$PY" analysis/speed_table.py "${RUNS[@]}" --scorer-tok-s 31.6 --cap-scorer-min 7 --out "$OUT/speed_table_a100proxy.csv" > "$OUT/speed_table_a100proxy.md"

echo "== 4. timing per run"
for r in "${RUNS[@]}"; do
  "$PY" -m scorer_minutes timing "$r" --glob 'run/**/llm_calls/*.jsonl' > "$OUT/timing/$(basename "$r").json" \
    || echo "timing failed: $r"
done

echo "== 5. noise table"
EXTRA=(); [ -f "$DATA/extra_runs.csv" ] && EXTRA=(--extra-csv "$DATA/extra_runs.csv")
"$PY" analysis/noise_table.py "${EXTRA[@]}" --pool-exclude g31_v12e --out "$OUT/noise_long.csv" --markdown "$OUT/noise_table.md" > /dev/null
cat "$OUT/noise_table.md"

echo "== 6. aggregates for the paper -> $DATA"
cp "$OUT/speed_table.csv" "$OUT/speed_table.md" "$OUT/speed_table_a100proxy.csv" "$OUT/speed_table_a100proxy.md" "$OUT/noise_table.md" "$OUT/noise_long.csv" \
   "$OUT/prompt_tokens_check.txt" "$DATA/"
mkdir -p "$DATA/timing"
for j in "$OUT"/timing/*.json; do
  # drop per-session flags (session keys are task ids; keep counts only)
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); d['flagged_sessions']=len(d.get('flagged_sessions',[])); print(json.dumps(d,indent=1))" "$j" \
    > "$DATA/timing/$(basename "$j")"
done
echo "== 7. paper tables 4.1, 4.3, 4.4 and figures"
"$PY" analysis/speed_groups.py > "$DATA/speed_groups.md"
"$PY" analysis/control_table.py > /dev/null
"$PY" analysis/timing_coef_table.py > "$DATA/timing_coefs.md"
"$PY" analysis/lora_sessions.py > /dev/null
"$PY" analysis/budget_curve.py > "$DATA/budget_curve.md"
"$PY" scripts/paper/figures.py || echo "figures failed (matplotlib missing?)"
echo "done. Review docs/paper/aggregates/, label runs in $DATA/run_labels.tsv (then rerun), and commit."
