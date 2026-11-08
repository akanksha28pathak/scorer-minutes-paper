# Data card: scorer-minutes paper aggregates

Aggregates from 106 logged runs (4,757 sessions) of Gemma 4 coding agents (31B, E4B, 26B-A4B; quantized w4a16
checkpoints served by vLLM) in the Kaggle Gemma 4 Developer Agent harness, Oct-Nov 2026. Six GPU models: Kaggle 4x L4,
A100 80 GB, H100 SXM, L40S, RTX PRO 6000, RTX 5080. No task text, code, patches, prompts or model outputs are included;
task ids in `noise_long.csv` are replaced by `t001`... (stable within a split). License: Apache-2.0, as the code.

| File | One row per | Columns (main) |
|---|---|---|
| `speed_table.csv` | logged run | `run`, `label` (hardware, tensor parallelism, sessions in parallel, model), `sessions`, `median_tok_s` / `p25_tok_s` / `p75_tok_s` (per-session tokens/s), `factor` and its 95% CI vs the scorer reference (33.8 tokens/s), `cap_wall_min` (wall cap for 7 scorer minutes), `model_share`, `fit_r2`, `prefill_share` / `decode_share` / `overhead_share` of model time, `decode_tok_s_ctx0` / `decode_tok_s_ctx24k`, `late_over_early` |
| `speed_groups.csv` | setup (paper table 4.1) | `gpu`, `model`, `sessions_in_parallel` (attempts x workers when attempts ran in parallel), `note`, `runs`, `sessions`, tokens/s median / min / max, factor median / min / max |
| `control_43.csv` | control setup (paper table 4.2) | `setup`, `runs`, `factor` (full run), `wall_cap_min` (set before the run), `scorer_min` = cap / factor, `solved`, `timed_out` (harness timeout field), `timed_out_solved`, paired counts vs the scorer (`both`, `scorer_only`, `run_only`), exact McNemar p |
| `control_twin.csv` | pair | converted vs nominal cap on the same hardware: tasks solved only by each, exact McNemar p |
| `budget_curve.csv` | run (paper section 4.2) | one agent, one setup, caps for 3.5 / 7 / 10 scorer minutes: factor, scorer minutes, solved, timed out |
| `lora_sessions.csv` | setup (paper section 4.4) | mean / median session seconds with and without adapter B (110 sessions each), ratio, projected hours |
| `timing_coefs.csv` | setup (paper table 4.3) | fitted `t = c + a*new_prompt + b*gen + d*gen*context` (seconds, tokens in thousands): `c_overhead_s`, `a_s_per_k_prompt`, `b_s_per_k_gen`, `d_s_per_k_gen_per_k_ctx`, decode tokens/s at 0k and 24k context |
| `timing/<run>.json` | logged run | output of `scorer-minutes timing`: model share, time split, tokens/s per context bin, early vs late calls, the fit (per-session flags reduced to a count) |
| `noise_long.csv` | (run, task) | `config` (agent version), `split` (`train`, `held-out`; 55 tasks each), `run`, `task` (anonymized), `solved` (0/1) |
| `run_labels.tsv` | logged run | run name, hardware label |

Grading: the competition's sandbox image, strict (all hidden tests must pass). The scorer reference is a sequential run
(one session at a time) on Kaggle 4x L4 with tensor parallelism 4, the competition's scoring setup.
