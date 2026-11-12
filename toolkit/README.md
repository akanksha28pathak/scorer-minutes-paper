# scorer-minutes

Hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets.

Agent benchmarks with a time cap (per session, or a total limit) mix two things: what the agent does, and how fast the
GPU decodes. A 7-minute cap on a setup that decodes half as fast as the scorer is a 3.5-minute cap. And the same agent
solves a different number of tasks on every run. This toolkit measures both from standard per-call logs:

| Command | Question | Output |
|---|---|---|
| `calibrate` | How fast is this setup vs the scorer, and what wall cap equals a scorer-minute budget? | factor `f = scorer tok/s / median session tok/s` (bootstrap CI), `cap_wall = cap_scorer x f`, per-session scorer minutes |
| `timing` | Is one factor enough? (sessions decode-bound, prefill small, speed stable as context grows) | model vs tool time, a fitted per-call cost `t = c + a*new_prompt + b*gen + d*gen*context`, tok/s per context bin, early vs late calls, flagged sessions |
| `noise` | How much do identical runs differ, and can two agents be told apart? | per-config SD, pooled SD per split, task-level SD, difference CIs (run-noise and task bootstrap), runs needed |
| `draws` | How should one leaderboard score be read? | 95% interval of one draw, P(wrong order) for a true margin, runs needed |

Standard library only (Python >= 3.9). Deterministic (seeded bootstrap). Paper: *Scorer-minutes: hardware-normalized,
noise-aware evaluation of coding agents* (Kaggle Gemma 4 Developer Agent paper track, 2026).

**What we found with it** (Gemma 4 31B agents, 106 runs, six GPU models): per-session speed of one model ranges from 13
to 52 tokens/s across setups; the competition's own 4x L4 is 2.1-2.6x slower per session at 10 parallel sessions than
when it scores one at a time; budgets converted with the measured factor reproduced the scorer's score (19-20 vs 19 of
55 tasks on four setups), while the unconverted run solved 15 with 37 of 55 sessions timed out.

## Install / run

```bash
pip install .                               # from a clone; then: scorer-minutes ...
PYTHONPATH=src python3 -m scorer_minutes ...  # or without installing
```

## Inputs

**Call logs** (`calibrate`, `timing`): JSONL, one record per model call, with start / end timestamps (epoch seconds or
ISO 8601) and the completion token count; prompt tokens (and cached tokens) enable the prefill split. Fields are dotted
paths, defaults match OpenAI-style usage blocks:

| Option | Default |
|---|---|
| `--field-session` | `task,attempt` (missing first key: the file name) |
| `--field-t-start` / `--field-t-end` | `t_start` / `t_end` |
| `--field-completion-tokens` | `response.usage.completion_tokens` |
| `--field-prompt-tokens` | `response.usage.prompt_tokens` |
| `--field-cached-tokens` | `response.usage.prompt_tokens_details.cached_tokens` |

Records without timestamps are skipped and counted. Without a cached-token count the tool assumes prefix caching
(vLLM's default): a call prefills only what the transcript gained since the previous call (`--no-prefix-cache` to
turn this off).

**Results** (`noise`): CSV, long `config,split,run,task,solved` (preferred) or summary `config,split,run,solved,n`.

## Examples

```bash
# speed factor of a run vs a scorer that does 33.8 tok/s per session, and the wall cap equal to 7 scorer-minutes
scorer-minutes calibrate <run_dir> --scorer-tok-s 33.8 --glob 'run/**/llm_calls/*.jsonl' --cap-scorer-min 7 --csv sessions.csv

# the same, with the scorer speed measured from a run on the scoring setup itself
scorer-minutes calibrate <run> --reference <run_on_scorer_hardware> --glob 'run/**/llm_calls/*.jsonl' --cap-scorer-min 7

scorer-minutes timing <run> --glob 'run/**/llm_calls/*.jsonl'

# other log layouts: point the field options at your own keys (dotted paths; ISO 8601 or epoch timestamps)
scorer-minutes calibrate my_logs/ --glob '*.jsonl' --field-session session_id \
  --field-t-start start_time --field-t-end end_time --field-completion-tokens usage.completion_tokens --scorer-tok-s 33.8
scorer-minutes noise results_long.csv --compare agentA agentB --split train --markdown noise.md
scorer-minutes draws --sd 1.6 --score 9
```

## Definitions and assumptions

* **Session tok/s** = completion tokens / time inside model calls (prefill included). It is the throughput an agent
  gets per second of model time, the quantity a time cap rations. `timing` separates prefill from decode.
* **Factor** uses the median over sessions (robust to stalled or very short sessions); its CI is a percentile
  bootstrap over sessions.
* **Scorer minutes** = wall minutes / factor. Valid when sessions are decode-bound: model time dominates wall time and
  the factor does not drift with context length. `timing` checks both and flags sessions that break them
  (default: model share < 0.9 or prefill share > 0.2).
* **Wall time** of a session = first call start to last call end; time before the first and after the last call is
  not in call logs.
* **Run noise**: the pooled SD needs >= 2 runs of the same config; with few runs it is unstable, so the task-level SD
  (each task an independent coin with its observed solve rate) is reported too, and comparison tables use the larger.

## The protocol (one page)

1. Log every model call with timestamps and token usage.
2. Measure the reference speed on the scoring setup at its real concurrency (`calibrate --reference`).
3. Calibrate every run: wall cap = budget x f; report f with its CI.
4. Check the assumption (`timing`): model share, prefill share, late / early ratio.
5. Report budgets in scorer minutes, with timeouts and projected total time.
6. Run every configuration at least twice; report the pooled SD (`noise`) and the runs needed for a margin.
7. Read a single leaderboard draw as an interval (`draws`), not a rank.

## Paper data and notebook

`docs/paper/aggregates/` (repo root) holds the aggregates behind every table of the paper (no task content; data card
in its README):
`speed_table.csv` (one row per logged run: speed, factor, time split), `speed_groups.csv` (table 4.1),
`control_43.csv` and `control_twin.csv` (table 4.2, the control), `lora_sessions.csv` (section 4.1), `timing_coefs.csv` and `timing/` (section 4.3, per-run fits),
`noise_long.csv` (task x run solved matrix, section 5), `run_labels.tsv` (hardware per run).
`notebook/scorer_minutes_demo.ipynb` recovers known speeds from synthetic logs and rebuilds the paper's tables and
figures from those aggregates.

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q tests
```

Tests use synthetic logs with known prefill / decode speeds (`tests/synth.py`); no real traces.

## License and citation

Apache-2.0 (`LICENSE`). Citation: `CITATION.cff`.
