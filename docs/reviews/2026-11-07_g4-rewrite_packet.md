# Review packet: g4-rewrite (2026-11-07)

You are a judge for the Kaggle "Google - The Gemma 4 Developer Agent Paper Track" (judges are Google
researchers in graph ML and agentic software engineering). Submissions are Kaggle Writeups of at most 3,000 words
reporting original research on agentic software engineering. Each is scored 0-5 on five equally weighted criteria:
- Novelty: new insights, deeper understanding, or important properties of existing methods.
- Quality: how general the approach is beyond the competition; does it translate to other problems?
- Relevance: significance and potential impact for software engineering and agentic learning.
- Verifiability: enough information to understand how it works and how data was obtained, analyzed and interpreted.
- Clarity: clear presentation and writing.
Prizes: Best Paper (highest average), Best New Resource (a new dataset, library, tool or software), Best New
Application. This paper targets Best New Resource and a high average.

Review the material below as a strict judge. Answer in this structure:
1. Scores 0-5 for each criterion, one sentence of justification each, and the average.
2. The 5 most serious weaknesses, most serious first. For each: where (section), why a judge would mark it down,
   and a concrete fix that fits in the remaining time (the deadline is days away; no new large experiments).
3. Claims that are not supported by the numbers shown, or numbers that look inconsistent with each other.
4. Anything unclear to a reader outside this competition (jargon, undefined terms, missing setup details).
5. One change that would raise the score the most.
Be specific and quote the text you object to. Do not rewrite the paper.

## What changed in this step

Full rewrite to 2,800 words (limit 3,000), with every G1b objection fixed (fitted timing coefficients now in table 4.3; the two anomalous speed rows explained as parallel attempts, i.e. 10 and 12 concurrent sessions; run overlap; McNemar wording; 2.1-2.6x; v_ref explanation). Since G1b, a fresh-agent cold read found and we fixed 10 number errors (e.g. 56 full 31B runs, not 58), and a second reviewer restructured the abstract and introduction around the idea and cut two claims not traceable to the released data. Sections renumbered: 4.2 control, 4.3 timing fit, 4.4 LoRA. Judge the full draft as submitted.

## Paper (full draft; full draft is 2800 words, not counting markdown table pipes)

# Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets

## Abstract

Coding-agent benchmarks increasingly cap each session by wall-clock time, as this competition does. Such a cap mixes
what the agent does with how fast the serving stack decodes: teams developing on other GPUs, or at other concurrency,
get a different amount of model work from the same minutes. We propose measuring budgets in **scorer
minutes**: wall minutes divided by a speed factor *f*, the scorer's tokens per second over the development setup's,
measured from ordinary per-call logs. We release **scorer-minutes**, a dependency-free toolkit and protocol that
computes the factor and the wall cap, checks from the same logs whether one factor is enough, and reports run noise. Across 103 runs (4,592 sessions) of Gemma 4 agents on six GPU models, the same 31B model decodes at 13 to 52
tokens/s per session, and the scorer's own 4x L4 at 10 concurrent sessions is **2.1-2.6x** slower per session than the scorer
reference (one session at a time). In a control (one agent, 55 tasks), the scorer setup solved **19**; runs converted to the same
budget on four other setups (factors 0.65-2.40) solved **19-20**; the unconverted run on the scorer's hardware at 10
sessions solved **15**, with 37 of 55 sessions timed out. Identical runs differ by an SD of **1.1-1.7 tasks of 55**:
one run each ranks a true 2-task difference wrongly 11-20% of the time.

## 1. Introduction

This competition scores an agent (prompts, tools, optional LoRA adapter) on about 120 hidden tasks from private Python
repositories: the share whose hidden tests pass. Each session runs under a cap set by the submitter, and all sessions
must finish within 12 hours on 4x NVIDIA L4, one session at a time, one vLLM server with tensor parallelism over the 4
GPUs ("TP4"), serving quantized (w4a16) Gemma 4 checkpoints.

Nobody develops on the scorer: teams test on rented GPUs or on Kaggle's 4x L4 with many parallel sessions, each
decoding at a different speed, so "7 minutes" is a different budget on each. Even the scorer's own hardware
is a different machine at development concurrency: at 10 parallel sessions a 7-minute cap buys 2.7 scorer minutes and
the same agent loses 4 of 19 tasks (Section 4.2). The reverse error costs too: a cap tuned on a fast machine overruns the
12-hour limit and scores zero. Noise compounds this: the same agent solves different tasks on every run, so
one- or two-task differences are read as progress.

**The idea** is to treat the cap as a budget of model work, not of minutes: one number per run, the speed factor *f*,
converts between the two, and a check on the same logs says whether one number is enough.

**Contributions (the resource).** (1) `scorer-minutes`, a Python-standard-library toolkit (`calibrate`, `timing`,
`noise`, `draws`) reading OpenAI-style per-call logs and reporting the speed factor, the wall cap, a fitted overhead /
prefill / decode split, and noise and power estimates. (2) A speed and timing table for 103 runs (79
of Gemma 4 31B, 14 of E4B, 9 of 26B-A4B, 1 of 12B) across six GPU models and 19 serving setups. (3) A control
validating the conversion against the scorer's own setup. (4) A repeated-run matrix, its noise table and a one-page
protocol.

Terms: `g31_<version>` is one Gemma 4 31B agent configuration; the **train** split is 55 public
tasks we develop on, the **held-out** split 55 others we only measure on; a **nominal** cap is the scorer's wall cap
applied unconverted elsewhere.

## 2. Related work

Wang et al. [On the Clock, arXiv 2610.10833] study agents told a wall-clock budget in the prompt: behavior under a
fixed clock, not what the clock means on different hardware. Benchmarks report wall-clock time as a
within-environment diagnostic [OpenHarmony Bench, arXiv 2608.16022] or one un-normalized limit [AnalysisBench, arXiv
2604.11270]; we know of no protocol that converts a cap between setups. For noise, Miller [arXiv 2411.00640] gives standard errors, paired tests and power analysis for LLM evals,
"On Randomness in Agentic Evals" [arXiv 2602.07150] finds SWE-bench Verified pass@1 shifting 2.2-6.0 points across runs,
and HAL [arXiv 2510.11977] builds reproducible agent evaluation; we apply these statistics under a time budget. Here,
public notebooks measure 31B decode speed at one concurrency [P. D. Pereira, "L4 Decode Law & 12-Call
Cliff"], leaderboard noise [D. Afshar, "58 Tasks, Bronze Is a 124-Team Tie"] and config feasibility [V. Murugan,
"Budget Feasibility Audit"; netaro, "YaoNeM Pareto Analysis"]; none converts a cap or measures repeated-run noise.
TraceLab [arXiv 2606.30560] finds frontier-model agent sessions dominated by long contexts and tool time; ours are the
opposite (Section 4.3).

## 3. The resource

### 3.1 Definitions

From each session's per-call logs: session speed `v_s` = completion tokens / time inside model calls (prefill included:
what a time cap rations); run speed `v = median_s v_s` with a bootstrap CI; **factor** `f = v_ref / v` (f > 1:
slower than the reference); **scorer minutes** = wall minutes / f, so a budget of *B* scorer minutes is a wall cap of
`B x f`. `v_ref` is measured on the scoring setup at its real concurrency (`calibrate --reference`): a sequential run on
4x L4 gave **33.8 tokens/s** (5 sessions, IQR 32.8-34.4), fixed before the Section 4.2 control, whose 56 scorer
sessions then measured 33.4 and 32.9 in two parts, within 2.5%.

### 3.2 When one factor is enough

One factor converts budgets only if session time is dominated by model calls at a roughly constant speed. `timing`
reports the **model share** (time inside model calls / session wall time), model-free checks (tokens/s by context;
**late / early**, tokens/s of the last third of each session's calls over the first third), and
a per-call least-squares fit

`t = c + a * new_prompt + b * gen + d * gen * context` (seconds; tokens in thousands),

with `c` per-call overhead, `a` prefill per 1k prompt tokens added since the previous call (prefix caching covers the
rest), `b` decode per 1k generated tokens, and `d` the decode slowdown with context. It flags sessions with model share
< 0.9 or prefill share > 0.2, from client timestamps and token counts alone.

### 3.3 Noise and design

`noise` takes `config, split, run, task, solved` rows and reports per-config and pooled SD (chi-square CI), a task-level
SD (`sqrt(sum_i p_i(1-p_i) k/(k-1))`, each task a coin with its solve rate p_i over k runs; all tasks, not a few run
totals), difference CIs, runs needed to detect a margin, and the chance one
run each ranks two agents wrongly; `draws` does the same for single leaderboard scores. The toolkit is standard library
only, reads OpenAI, LiteLLM or vLLM logs via configurable field paths, is deterministic, and has 16 tests (the fit
recovers known prefill and decode speeds from synthetic logs). One script regenerates every table and figure.

## 4. Validation

All runs use Gemma 4 agents in the competition's harness and sandbox image, vLLM serving, graded as the scorer does
(all hidden tests must pass). Of 103 logged runs, 73 are full (>= 20 sessions; 4,377
sessions), the rest smoke runs. The 56 full 31B runs give the speed table (4.1); one agent under
converted caps on seven setups gives the control (4.2); 24 full 55-task runs (7 configurations run twice, the
cross-hardware g31_v12e pair, 8 single runs) give the noise matrix (Section 5).

### 4.1 Speed depends on the GPU and on concurrency

All 56 full 31B runs by setup (Figure 1); tokens/s = each run's median per-session speed (E4B and 26B-A4B:
`speed_groups.csv`).

| Setup (31B) | Sessions in parallel | Runs (sessions) | Tokens/s per session, median (range) | Factor vs scorer |
|---|---|---|---|---|
| **Kaggle 4x L4, TP4 (scorer setup)** | 1 | 2 (56) | 33.2 (32.9-33.4) | 1.01-1.02 |
| H100 SXM 80 GB | 5 | 1 (55) | 52.0 | 0.65 |
| RTX PRO 6000 96 GB | 2 / 5 | 1 (24) / 22 (1,375) | 36.1 / 38.2 (36.5-38.8) | 0.94 / 0.87-0.92 |
| RTX PRO 6000, + LoRA adapter B | 5 | 3 (165) | 35.4 (35.1-35.6) | 0.95-0.96 |
| RTX PRO 6000, 2 attempts x 5 | 10 | 2 (104) | 26.4 (25.7-27.2) | 1.24-1.31 |
| A100 80 GB | 5 | 16 (948) | 30.5 (29.2-33.4) | 1.01-1.16 |
| A100 80 GB, + LoRA adapter A | 5 | 1 (55) | 26.1 | 1.29 |
| A100 80 GB, 3 attempts x 4 | 12 | 1 (25) | 16.9 | 1.99 |
| L40S 48 GB | 5 | 1 (55) | 20.7 | 1.63 |
| Kaggle 4x L4, TP4 | 10 | 6 (261) | 14.1 (13.2-15.8) | 2.13-2.55 |

(1) **Concurrency matters as much as the GPU**: the scorer's own hardware at 10 sessions runs at factors 2.1-2.6, and
the "attempts" rows (2 or 3 attempts per task at once, so 10 or 12 sessions) drop the RTX PRO 6000 from 38.2 to 26.4
tokens/s and the A100 (partial run) from 30.5 to 16.9. (2) **Small changes move the factor**: a LoRA adapter cost
7-16% of speed (Section 4.4). (3) **Speed is stable within a setup** (A100: factors 1.01-1.16 over 16 runs) **but not
across nominally equal ones**: five E4B runs on one RTX 5080 at 5 sessions ran at 88-90 (three) and at 24 and 40
tokens/s, for reasons the metadata does not record. Calibrate every run.

### 4.2 Normalization reproduces the scorer

One agent (g31_v12e, budget 7 scorer minutes), the 55 train tasks, seven setups: if the conversion is right, converted
caps should solve about as many tasks as the scorer and the nominal cap should not. The reference is the scorer setup
itself: 4x L4, TP4, one session at a time, 7 minutes (two parts, as our kernel has a 150-minute deadline; one cut
session rerun, hence 56 sessions). Converted caps were set before each run from 5
calibration sessions (H100, L40S), an earlier run (RTX PRO 6000) or a guess from earlier 10-session runs (4x L4 at 10
sessions: 2.13). The table gives the factor measured on the full run, so the realized budget is wall cap / f
(Figure 2).

| Setup | f (full run) | Wall cap (set in advance) | Scorer min (cap / f) | Solved / 55 | Timed out | Overlap with scorer (both / scorer only / run only) |
|---|---|---|---|---|---|---|
| **Kaggle 4x L4, 1 session (scorer)** | 1.02 (reference run) | 7 min | 6.9 | **19** | 11 | - |
| H100 SXM, 5 sessions (v12e) | 0.65 | 4.8 min | 7.4 | 19 | 11 | 16 / 3 / 3 |
| RTX PRO 6000, 5 sessions (v15) | 0.90 | 6.0 min | 6.7 | 19 | 15 | 18 / 1 / 1 |
| L40S, 5 sessions (v12e) | 1.63 | 11.7 min | 7.2 | 20 | 7 | 17 / 2 / 3 |
| Kaggle 4x L4, 10 sessions, converted | 2.40 | 15 min | 6.3 | 20 | 8 | 17 / 2 / 3 |
| Kaggle 4x L4, 10 sessions, **nominal** | 2.55 | 7 min | **2.7** | **15** | **37** | 14 / 5 / 1 |
| A100, 5 sessions (development reference) | 1.08 | 7 min | 6.5 | 17 | 12 | 14 / 5 / 3 |

The four pre-calibrated runs land within one task of the scorer at factors from 0.65 to 2.40, although the pre-run
factors missed the full-run factor by up to 12% (realized budgets 6.3-7.4 scorer minutes); the A100 run is within two. Per-task disagreements with the scorer are symmetric (McNemar p = 1.0 for the
four pre-calibrated runs, 0.73 for the A100): converted runs solve different tasks, as repeated runs do, not fewer. The
nominal run gets 2.7 scorer minutes: 37 of 55 sessions time out (no timed-out session was solved in any run), and it
solves 4 fewer tasks than the scorer and 5 fewer than its converted twin on identical hardware. That gap is about 3 SD
of a run-to-run difference (Section 5: run SD 1.13, so 1.6 for a difference); the paired per-task test (6 solved only
when converted, 1 only when not; McNemar p = 0.13) is weaker, seeing only the 7 disagreeing tasks; the timeouts (37 vs
7-15) show the mechanism.

### 4.3 Why one factor works here, and what the fit shows

One factor suffices when sessions are decode-bound. In all 73 full runs, model calls take 0.965-0.998 of session wall
time; the per-call fit has median R^2 0.987 and puts prefill at 2.3% of model time (median; max 5.5%). Fitted
coefficients (`timing_coefs.csv`):

| Setup (agent) | Calls (min R^2) | c, s | a, s / 1k prompt | b, s / 1k gen | d, s / 1k gen / 1k ctx | Decode tok/s, 0k / 24k ctx |
|---|---|---|---|---|---|---|
| 4x L4, 1 session, scorer (v12e) | 1,417 (0.999) | 0.12 | 0.46 | 26.6 | 0.19 | 37.5 / 32.1 |
| 4x L4, 10 sessions (v12e) | 2,480 (0.955) | 0.41 | 0.89 | 62.5 | 0.23 | 16.1 / 14.7 |
| H100 SXM, 5 sessions (v12e) | 1,498 (0.985) | 0.08 | 0.25 | 17.0 | 0.14 | 58.8 / 48.9 |
| L40S, 5 sessions (v12e) | 1,549 (0.988) | 0.22 | 0.39 | 45.5 | 0.07 | 22.0 / 21.2 |
| RTX PRO 6000, 5 sessions (v15) | 3,047 (0.990) | 0.26 | 0.17 | 23.5 | 0.08 | 42.6 / 39.5 |
| RTX PRO 6000, + adapter B (v15) | 2,659 (0.991) | 0.08 | 0.24 | 26.3 | 0.07 | 38.0 / 35.6 |

Decode `b` dominates and carries the hardware difference: ten sessions on 4x L4 more than double it and triple per-call
overhead `c`, while prefill `a` stays under a second per 1k new tokens thanks to prefix caching. `d` varies (decode at 24k
is 14% slower than at 0k on the scorer, 4% on the L40S), but one agent builds similar contexts everywhere, so a
run-level factor absorbs it. The exception is heavy sharing: at 10 sessions, late calls run up to
27% faster than early ones as sessions finish, and table 4.3's lowest R^2 (0.955) is there, so a run-level factor under-converts a run's start and over-converts its end.

### 4.4 A LoRA adapter changes the factor

Adapter B (rank 16, tuned on solved sessions) on the RTX PRO 6000: 35.4 vs 38.2 tokens/s; the fit puts the
cost in decode (`b` +12%) and prefill (`a` +41%), not overhead. In two held-out runs the adapter changed no score
beyond noise (20.5 vs 19.0 of 55) but lengthened sessions by 12%, moving the projected scorer time from 10.3-10.8 h to
11.6-12.1 h of the 12-hour limit. Adapter A on an A100: 16% slower than its paired base run. Recalibrate after any
change to what is served.

## 5. Noise

### 5.1 Local runs

Repeated full runs on the same 55 tasks (7 scorer minutes; A100 for the first four configs, RTX PRO 6000 converted
otherwise):

| Config | Split | Solved per run |
|---|---|---|
| g31_v12 / g31_v12d / g31_v9d | train | 20, 19 / 20, 18 / 19, 19 |
| g31_v12e (A100 / RTX PRO 6000) | train | 17 / 19 |
| g31_v9c / g31_v15 | train | 15, 13 / 19, 21 |
| g31_v15 / g31_v15 + adapter B | held-out | 18, 20 / 20, 21 |

4-6 of 55 tasks flip between identical runs. The pool excludes the cross-hardware g31_v12e pair, so it is independent of
the conversion. Pooled run SD: 1.14 tasks on train, 1.12 on held-out, **1.13 over both (95% CI
0.75-2.31, 7 dof)**; task-level SD 1.52 and 1.66. Reading comparisons with the larger (1.66, held-out): one run each ranks a true 2-task margin wrongly **20%** of the time (11% with two runs each), a
1-task margin 34%; 80% power for a 2-task margin takes **11 runs** per agent. The cross-hardware pair differs by 0.9
task-level SD of a difference, no more than identical runs.

### 5.2 The leaderboard

Public scores are single draws on ~58 hidden tasks. Resubmissions of identical files vary with an SD of 1.4-1.8 tasks
[D. Afshar; forum]: one public notebook that scored 13/58 scored 7 and 9 for others who resubmitted it, and 9 for us.
With SD 1.6, one draw's 95% interval is about +-3 tasks (`draws --sd 1.6 --score 9`: 5.9-12.1). Levels also differ
(24-40% of local splits vs 5-17% of hidden tasks: unseen private repositories), which no noise model explains.

## 6. Protocol

1. **Log every model call** with timestamps and token usage.
2. **Measure the reference speed** on the scoring setup at its real concurrency.
3. **Calibrate every run**: wall cap = budget x f; report f with its CI.
4. **Check the assumption** (`timing`); if flagged, measure on the scorer.
5. **Report budgets in scorer minutes**, with timeouts and projected total time.
6. **Run every configuration at least twice**; report the pooled SD and runs needed before claiming a margin.
7. **Read a leaderboard draw as an interval**, not a rank.

## 7. Limitations

Heavy concurrency makes speed drift within a run (late / early up to 1.27 at 10 sessions; 0.96-1.05 for 31B
otherwise), so a run-level factor is an average. One model family, one harness and Python repositories: tool-heavy
agents are less model-bound, which `timing` detects but one factor would not fix. The scorer reference rests on one
machine type (61 sequential sessions on Kaggle 4x L4, 32.9-33.8 tokens/s); host-to-host variation is not measured.
Each converted setup ran once: the control rules out errors as large as the nominal cap's (4-5 tasks) but cannot
resolve a 10% factor error. Noise estimates rest on seven repeated pairs, the task-level SD assumes independent tasks,
and leaderboard noise is partly community-reported.

## 8. Conclusion

A wall-clock cap is a hardware setting unless converted. One factor from per-call logs, checked against the same logs,
makes local results comparable to the scorer's; two runs and a pooled SD make them comparable to each other. Nothing
here is specific to Gemma or this competition: any timed, decode-bound agent benchmark can use the same logs and
commands.

**Reproducibility.** Toolkit, tests, data and notebook: github.com/akanksha28pathak/scorer-minutes (Apache-2.0). Data: `speed_table.csv`
(103 runs), `speed_groups.csv` (4.1), `control_43.csv` and `control_twin.csv` (4.2), `lora_sessions.csv` (4.4), `timing_coefs.csv` and `timing/` (4.3), `noise_long.csv`
(task id x run matrix), `run_labels.tsv`; no task content. Serving: vLLM 0.19.1, it-qat-w4a16 checkpoints.

