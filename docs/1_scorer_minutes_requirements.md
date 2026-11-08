# Idea 1 (scorer-minutes): requirements checklist for a Best New Resource paper

Written Nov 5 from the committed docs and scripts only; the raw call logs and run folders are git ignored, so every
"have" below is a number a doc states, not one recomputed from logs. Idea: [1_scorer_minutes.md](1_scorer_minutes.md).
Toolkit being built for it: `tools/scorer_minutes/` (see its README).

Status: ✅ have · 🟡 partial / needs rework · ❌ missing

**Update Nov 5 evening (PC aggregates in docs/paper/aggregates/, draft filled):** C0 ✅ logged sequential 4x L4 anchor
33.8 tok/s (5 sessions); A100 proxy 6-10% slower. C1 ✅ 81 runs, 5 GPU types. C4 ✅ model share 0.965-0.997, prefill
median 2.7% (max 5.5%), late/early 0.97-1.05 single GPU, up to 1.27 at 10 sessions. C5 ✅ pooled SD 1.19 (dof 6),
task-level 1.61; 11 runs for a 2-task margin. R4 ✅ incl. E4B and 26B-A4B. **Open: C3 [L4] control runs (Kaggle quota
resets Nov 7 00:00 UTC); citations to verify; final trim.** Correction: v12e reference is one A100 run (17/55), not 17
and 20; the 9/10 fair10 cap was 5.6 scorer-min (assumed factor 2.5, measured 1.98).

**Bottom line (Nov 5 morning):** the evidence that the problem exists is solid. Thin: the anchor to the real scorer and the formal
analysis (noise table, prefill split). The packaged resource is mostly unbuilt. About 2-3 days of CPU work plus one
GPU control run.

## 1. Claims

| # | Claim | Status | What the repo has | What is missing |
|---|---|---|---|---|
| C0 | What "scorer speed" is | 🟡 | The scorer runs tasks **sequentially** on 4x L4, TP4, vLLM 0.19.1 (COMMUNITY_INSIGHTS.md:27, :380; host post 743063). One sequential Kaggle check measured **32 tok/s** (DECISIONS_31B.md D1). The 31.6 "reference" is an **A100 with 5 sessions in parallel** (D3: 30-33 tok/s) used as a stand-in. | A **logged** sequential 4x L4 run through the calibrator, so the anchor has a session count and spread. Otherwise reviewers ask why the A100 is "the scorer". |
| C1 | Speed factors differ a lot between setups | ✅ | RTX PRO 6000: 36.9 / 37.2 tok/s -> x0.86 (D24, D25). Kaggle L4 at 10 workers: 15.1 -> x2.09 (HARD_TASKS.md:88). Also 13 tok/s at 10 workers (D1), ~17 at 12 on the A100 (D3), 32 at 1 worker. | One table with session count and spread per row; today the numbers sit in 4 docs. |
| C2 | A fixed wall cap on slower hardware hurts results | ✅ | **Same agent, same 10 tasks:** g31_v9b 4/10 at 10 parallel (~13 tok/s) vs **9/10** at the scaled cap; g31_v11m 5/71 with 66 timeouts at 10 parallel (RETRO_31B_BUDGET.md:35-38, D1). | The v9c vs v12e cap-hit numbers in the idea doc compare **different agents and budgets**; lead with v9b 4 -> 9 instead. Ideally one 55-task same-agent fixed-wall run (can come from the L4 control). |
| C3 | Normalization reproduces the reference | 🟡 | D25: v12e on the RTX PRO 6000 at 6.0 min wall: **19/55, 15 at the cap** vs A100 17 and 20/55, 12 at the cap. | Only tested at **f = 0.86** (close to 1). Decisive test: **v12e train55 on Kaggle 4x L4, 10 workers, cap ~ 7 x 2.1 ~ 15 min wall**. H1/H2 used x2.1 but with other agents on the hard tasks and no A100 match: not a control. ~1-1.5 h wall, ~2-3 GPU h quota (needs Akanksha's OK). |
| C4 | Sessions are decode-bound, so one factor works | 🟡 | "98% of 31B session time is generation" (D4, RETRO); ~3,400 run_command calls: median < 0.1 s, p99 <= 8 s (D28). | `session_speed.py` divides completion tokens by **all** time inside model calls, prefill included; the "98%" is model vs tool time, not decode vs prefill. Need the per-call fit t = c + a * new prompt tokens + b * completion tokens, prefill share vs context length, factor in early vs late calls (`scorer-minutes timing`). Check one llm_calls file for `usage.prompt_tokens` (LiteLLM usually logs it). |
| C5 | Run-to-run noise is large | 🟡 | **First numbers (Nov 5, `analysis/noise_table.py` on the stale matrix):** train55 pooled run SD **0.91** (3 configs, dof 3), task-level SD **1.58**: for 31B that is ~1-1.6 tasks of 55, **not the "3-4 per 55"** in the idea doc (that came from E4B on kaggle_clean: 12 vs 8 of 71). 1 run each: P(wrong order) 0.19 at a 2-task margin; 10 runs each for 80% power. Rephrase the claim. Repeated runs: train55 v12 20/19, v12d 20/18, v9d 19/19, v12e 17/20/19, v9c 15/13, v15 19/21; kaggle_clean e4b_v1 12/8, e4b_v11m 10/10. | Noise table: SD per split size, bootstrap CIs, runs needed (`scorer-minutes noise`, `analysis/noise_table.py`). **docs/RESULTS.md and TASK_MATRIX.md are stale** (Nov 3 22:19): D24-D29 runs (v9c x2, the D25 control, v15 x2) are missing. Regenerate on the PC first. |
| C6 | Leaderboard is noisy; local rank != public rank | ✅ | Community: SD 1.4-1.8 tasks; Budget-Fit 8-11/58 over 7-10 draws; the 0.22 bundle drew 7 and 9 for others (COMMUNITY_INSIGHTS.md:52-54, :73-75). **Our own ref_davis resubmission: 0.15 (~9/58)** vs the notebook's 0.22. v12e >= v9c locally, 0.06 vs 0.12 public (SUBMISSIONS.md). | Cite community draws as a source. g31_v15 (58028054) pending; useful only next to its local numbers. |
| C7 | Novelty | 🟡 **Nov 5: checked by web search** (arXiv not reachable from the cloud session; verify the PDFs). Nobody converts wall-clock caps between hardware: OpenHarmony Bench treats wall time as within-environment only; On the Clock (2610.10833) is about agents obeying a fixed clock. **Noise is not new**: On Randomness in Agentic Evals (2602.07150, ICLR 2026) and Miller's error bars (2411.00640): cite and frame ours as applying them to time budgets. TraceLab (2606.30560): tool time rivals generation for frontier agents, our contrast for C4. Written into docs/paper/SCORER_MINUTES.md section 2. | "To verify" notes; docs/paper/V9B_V9C.md:84-90 has a related-work list to reuse. | Check SWE-bench, SWE-agent, OpenHands, mini-SWE-agent, HAL / agent-eval reliability work, METR time horizons, MLPerf-style hardware normalization. |

## 2. Resource deliverables

| # | Item | Status | Gap |
|---|---|---|---|
| R1 | `calibrate`: logs -> factor -> wall cap | 🟡 -> in progress | `scripts/vast/session_speed.py` works on our layout; generalized as `scorer-minutes calibrate` (any JSONL call log with timestamps + usage). Its docstring says 31.6 = "10 sessions on an A100"; D3 / HARD_TASKS say 5 in parallel: confirm which. |
| R2 | `timing`: model vs tool time, prefill vs decode, flags non-decode-bound sessions | ❌ -> in progress | Built on C4. |
| R3 | `noise`: SD, CIs, runs needed, reading one leaderboard draw | ❌ -> in progress | `scorer-minutes noise` + repo adapter `analysis/noise_table.py`. |
| R4 | Speed-factor table (GPU x concurrency x model) | 🟡 | 31B only (L4 x4 at 1 and 10 workers, A100 at 5 and 12, RTX PRO 6000 at 5). **No E4B numbers**: if E4B Kaggle runs logged calls, CPU only to add. |
| R5 | Session-level dataset (tok/s, time split, cap hit, solved; no task content) | ❌ | Decide what is shareable (rules: aggregates and task ids only). |
| R6 | Repeated-runs table | 🟡 | Regenerate RESULTS / TASK_MATRIX, export CSV (`analysis/noise_table.py --long-csv`). |
| R7 | Reporting protocol / checklist | ❌ | One page. |
| R8 | Usage demo | 🟡 | Re-score v9c vs v12e (local vs public) and v9b 4/10 vs 9/10 with the tool; show the conclusions change. |

## 3. Paper sections (<= 3,000 words)

| Section | Status |
|---|---|
| Introduction | 🟡 argument in 1_scorer_minutes.md; rewrite resource-first |
| Related work | ❌ (reuse the V9B_V9C list) |
| Method: factor, cap conversion, assumptions | 🟡 formula exists; prefill caveat not quantified |
| Resource description | ❌ |
| Validation: controls | 🟡 D25 + v9b 4 -> 9; L4 control missing |
| Noise analysis | ❌ |
| Limitations | 🟡 listed, not measured |
| Reproducibility: vLLM version, TP, quantization, driver per row | 🟡 scorer side known (vLLM 0.19.1, TP4, w4a16); our Vast / Kaggle rows need versions |

## 4. Problems found in the repo

1. The "scorer reference" is not the scorer: an A100 stand-in for one sequential 4x L4 measurement. Anchor on a logged
   sequential L4 run, or state the stand-in plainly.
2. `session_speed.py` measures model-call throughput, not pure decode speed. Fine as a practical factor; the paper must
   say so (C4).
3. docs/RESULTS.md and docs/TASK_MATRIX.md are stale (Nov 3), and they are the generated sources the paper cites.
4. `session_speed.py` docstring: session count behind 31.6 (10 vs 5) disagrees with D3.
5. Paper plan conflict: CLAUDE.md says "one merged paper: V9B_V9C"; docs/paper-track/README.md recommends idea 1.
   Decide, then update CLAUDE.md.

## 5. Shortest path to Nov 10

| Day | Work | Where |
|---|---|---|
| Nov 5-6 | Regenerate RESULTS / TASK_MATRIX; check llm_calls for prompt_tokens; **launch the L4 control** (v12e train55, 10 workers, ~15-min cap) and, if cheap, a sequential 4x L4 anchor run on ~10 tasks | PC + Kaggle GPU (needs OK) |
| Nov 6-7 | Run `scorer-minutes calibrate / timing` on every logged run (factor table incl. E4B); `analysis/noise_table.py` | CPU (PC or private Kaggle kernel) |
| Nov 7-8 | Finish packaging, export CSVs (R4-R6), write R7 | here |
| Nov 8-10 | Related work, write-up, figures; submit | |

## 6. Execution plan v2 (Nov 6, grant review)

Review of this checklist + the Nov 5-6 results + the paper-track public notebooks (Kaggle CLI listing). Decision:
**stay on scorer-minutes, no pivot after Nov 7.** Localization/graph papers are crowded (~12 public notebooks) and our
own result there is negative; the hinted-trace leak finding (D40) is novel but needs a week.

**Differentiation** (cite each in related work and say what it lacks): pablodiegoo "L4 Decode Law & 12-Call Cliff"
(one GPU, one concurrency, a latency law); dariushafshar "58 Tasks, Bronze Is a 124-Team Tie" (leaderboard noise
only); velumurugan12 "Budget Feasibility Audit" and netaro "YaoNeM Pareto" (config arithmetic). Ours, uncovered: (a) a
cross-hardware cap conversion validated by a control at f ~ 2; (b) local repeated-run noise (15 full runs, 5 GPU types,
SD 1.1-1.7 of 55) equal to leaderboard noise (1.4-1.8 of 58), so the noise is the agent's, not the scorer's; (c) a LoRA
adapter changes the speed factor (D41: +12% session time, +19% decode; v15 10.8 h -> 12.1 h of the 12-h limit);
(d) a pip-installable library + per-session dataset.

**Fixes for the top judge objections:** (1) the [L4] pair (only necessary GPU work); paired per-task tables
(both / only-A / only-B, McNemar) and the minimal detectable effect; (2) open with (a)-(d), LoRA in its own subsection;
(3) re-anchor every factor on the logged sequential 4x L4 run (`analysis/speed_table.py --reference
gemma4-eval-g31-v11m-seq5`, 33.75 tok/s), A100 proxy as a secondary column; (4) chi-square CI on the pooled SD
(`tools/scorer_minutes/noise.py`), pool train55 + val55 (dof 8), runs-needed as a range, lead with "4-8 of 55 tasks
flip between identical runs"; (5) verifiable resource: tests runnable (pytest into the venv), public tool-only repo +
public notebook (needs Akanksha's decision: the main repo stays private until the deadline); trim to <= 2,800 words;
two figures.

Plan of action, progress and decisions: **[PLAN.md](PLAN.md)** (moved there Nov 6).
