# Idea 1. Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets

## Question
Coding-agent benchmarks with a wall-clock budget (this competition: a per-session time cap, a 12-h total limit, 4x L4)
quietly mix two things: what the agent does and how fast the GPU decodes. When teams develop on other hardware (A100,
RTX 6000, Kaggle 4x L4 at different concurrency), are their local numbers comparable to the scorer's, and how much of
the reported spread between agents is noise?

## Claim
Measured in **scorer-minutes** (wall time divided by the run's decode-speed factor against the scorer), results become
comparable across GPUs; without it, a time cap silently changes by 2x or more between machines. And run-to-run noise
is large enough (several tasks per 55, about 1.5 tasks per 58 on the leaderboard) that most single-run comparisons,
including leaderboard ones, cannot separate agents.

## Evidence we already have (all in this repo)
* **Speed factors** (`scripts/vast/session_speed.py` on the model-call logs, median generated tok/s per session):
  scorer reference 31.6 tok/s (Gemma 4 31B, A100, 5 sessions); RTX PRO 6000 (Blackwell) at 5 sessions 36.9-37.2
  (factor 0.85-0.86); A100 SWE-smith runs 31.0 (1.02); Kaggle 4x L4 at 10 workers 14.0-15.1 (2.1-2.26).
* **Control** (D25): g31_v12e with its 7 scorer-minutes on the RTX PRO 6000 (6.0 min wall) solved 19/55 vs 17 and 20 on
  the A100 at scorer speed; time-cap hits 15 vs 12. The normalization reproduces the reference.
* **What goes wrong without it:** the same agent at a fixed wall cap on a 2x slower setup gets half the effective
  budget; v9c at 4.5 scorer-min hit its cap in 28-32 of 55 sessions, v12e at 7 in 12-15.
* **Where the time goes:** about 98% of a 31B session is generation; ~3,400 shell commands had median < 0.1 s and
  p99 <= 8 s (D28), so decode speed, not tool time, sets the budget. This is why a single factor works.
* **Noise:** identical configurations on our splits differ by 3-4 tasks per 55 (CLAUDE.md, docs/RESULTS.md); on the
  public leaderboard identical bundles vary by 1.4-1.8 tasks SD and one 0.22 bundle drew 7-13 of 58 on reruns
  (docs/COMMUNITY_INSIGHTS.md); local rank and public rank disagree (v12e > v9c locally, 0.06 vs 0.12 publicly;
  ref_davis resubmission 57996854 pending).

## Method to finish
1. Formalize: factor f = scorer tok/s / median session tok/s; cap_wall = cap_scorer x f; report scorer-minutes per
   session. Show the decode-bound assumption holds (tool-time share) and where it breaks (prefill-heavy sessions).
2. One more matched control on a third setup (e.g. Kaggle 4x L4 at 10 workers, already run for H1/H2, factor 2.1).
3. Noise table: all repeated runs in docs/RESULTS.md (same agent, same split) -> SD per split size; bootstrap CIs; the
   number of runs needed to detect a 2-task difference.
4. Practical guide: how many runs, which normalization, how to read a single leaderboard draw.

## Resource (Best New Resource)
The calibration protocol, `session_speed.py` (reads standard OpenAI-style call logs), a small table of speed factors for
common GPUs at common concurrencies for Gemma 4 31B / E4B, and the noise table. All aggregate numbers; no task data.

## Why it is novel and relevant
Agent papers usually report one run on one machine with a fixed timeout. We know of no protocol that converts
wall-clock budgets between hardware for agent evaluation (to verify in related work: SWE-bench, SWE-agent, OpenHands
evaluation notes). Every team in this competition develops off the scorer's hardware.

## Risks
Prefill share grows with context, so the factor is not constant over a session; must quantify. Leaderboard noise
numbers come partly from community posts (cite, and add our own resubmissions).

## Effort
2-3 days, CPU only (one optional GPU control run, ~1 h). Scripts: `scripts/vast/session_speed.py`,
`analysis/make_results_doc.py`; new: `analysis/noise_table.py`.
