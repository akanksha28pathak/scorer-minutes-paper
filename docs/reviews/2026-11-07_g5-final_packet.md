# Review packet: g5-final (2026-11-07)

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

Final check before submission. Since G4 (Gemini 4.6): the table 4.2 row label error is fixed (all seven rows are g31_v12e), the control / matrix overlap is stated, table 4.3's lead-in explains summed calls, lowest R^2 and the decode formula, section 4.4 no longer reads prefill and overhead separately, section 5.1 names each SD, E4B / 26B-A4B are defined again. NEW: a budget-response experiment answers 'each converted setup ran once': on one setup, caps for 3.5 / 7 / 10 scorer minutes solved 16 / 18 / 21 (the 7-minute run also repeats the control row, 19); slope 0.76 tasks per scorer minute, so a 10% factor error costs about half a task (section 4.2). Totals now 106 runs. The toolkit, data and notebook live in the main project repository. 2,922 words. Included: the full draft, the Kaggle Writeup fields and the release layout and notebook. Look for anything a judge would mark down that can still be fixed in a day, and inconsistencies between the paper, the Writeup fields and the release.

## Paper (full draft; full draft is 2922 words, not counting markdown table pipes)

# Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets

## Abstract

Coding-agent benchmarks increasingly cap each session by wall-clock time, as this competition does. Such a cap mixes
what the agent does with how fast the serving stack decodes: teams developing on other GPUs, or at other concurrency,
get a different amount of model work from the same minutes. We propose measuring budgets in **scorer
minutes**: wall minutes divided by a speed factor *f*, the scorer's tokens per second over the development setup's,
measured from ordinary per-call logs. We release **scorer-minutes**, a dependency-free toolkit and protocol that
computes the factor and the wall cap, checks from the same logs whether one factor is enough, and reports run noise. Across 106 runs (4,757 sessions) of Gemma 4 agents on six GPU models, the same 31B model decodes at 13 to 52
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
prefill / decode split, and noise and power estimates. (2) A speed and timing table for 106 runs (82
of Gemma 4 31B, 14 of E4B, an effective-4B on-device model; 9 of 26B-A4B, a mixture of experts with 4B active parameters; 1 of 12B) across six GPU models and 19 serving setups. (3) A control
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
TraceLab [arXiv 2606.30560] finds frontier-model sessions tool-heavy; ours are model-bound (Section 4.3).

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
(all hidden tests must pass). Of 106 logged runs, 76 are full (>= 20 sessions; 4,542
sessions), the rest smoke runs. The 59 full 31B runs give the speed table (4.1); one agent under
converted caps on seven setups (8 runs, 2 shared with the matrix) gives the control (4.2), with 3 budget-response runs on one setup; 24 full 55-task runs (7 configurations run twice, the
cross-hardware g31_v12e pair, 8 single runs) give the noise matrix (Section 5).

### 4.1 Speed depends on the GPU and on concurrency

All 59 full 31B runs by setup (Figure 1); tokens/s = each run's median per-session speed (E4B and 26B-A4B:
`speed_groups.csv`).

| Setup (31B) | Sessions in parallel | Runs (sessions) | Tokens/s per session, median (range) | Factor vs scorer |
|---|---|---|---|---|
| **Kaggle 4x L4, TP4 (scorer setup)** | 1 | 2 (56) | 33.2 (32.9-33.4) | 1.01-1.02 |
| H100 SXM 80 GB | 5 | 1 (55) | 52.0 | 0.65 |
| RTX PRO 6000 96 GB | 2 / 5 | 1 (24) / 25 (1,540) | 36.1 / 38.1 (36.5-38.8) | 0.94 / 0.87-0.92 |
| RTX PRO 6000, + LoRA adapter B | 5 | 3 (165) | 35.4 (35.1-35.6) | 0.95-0.96 |
| RTX PRO 6000, 5 workers x 2 parallel attempts | 10 | 2 (104) | 26.4 (25.7-27.2) | 1.24-1.31 |
| A100 80 GB | 5 | 16 (948) | 30.5 (29.2-33.4) | 1.01-1.16 |
| A100 80 GB, + LoRA adapter A | 5 | 1 (55) | 26.1 | 1.29 |
| A100 80 GB, 4 workers x 3 parallel attempts | 12 | 1 (25) | 16.9 | 1.99 |
| L40S 48 GB | 5 | 1 (55) | 20.7 | 1.63 |
| Kaggle 4x L4, TP4 | 10 | 6 (261) | 14.1 (13.2-15.8) | 2.13-2.55 |

(1) **Concurrency matters as much as the GPU**: the scorer's own hardware at 10 sessions runs at factors 2.1-2.6, and
the "attempts" rows (2 or 3 attempts per task at once, so 10 or 12 sessions) drop the RTX PRO 6000 from 38.2 to 26.4
tokens/s and the A100 (partial run) from 30.5 to 16.9. (2) **Small changes move the factor**: a LoRA adapter cost
7-16% of speed (Section 4.4). (3) **Speed is stable within a setup** (A100: factors 1.01-1.16 over 16 runs) **but not
across nominally equal ones**: E4B runs on one RTX 5080 at 5 sessions ranged from 24 to 90 tokens/s, unexplained by
the metadata. Calibrate every run.

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
| H100 SXM, 5 sessions | 0.65 | 4.8 min | 7.4 | 19 | 11 | 16 / 3 / 3 |
| RTX PRO 6000, 5 sessions | 0.90 | 6.0 min | 6.7 | 19 | 15 | 18 / 1 / 1 |
| L40S, 5 sessions | 1.63 | 11.7 min | 7.2 | 20 | 7 | 17 / 2 / 3 |
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

**How much does a factor error cost?** On one setup (RTX PRO 6000, 5 sessions) we converted caps for 3.5, 7 and 10
scorer minutes (`budget_curve.csv`): 16, 18 and 21 solved, with 35, 11 and 0 sessions timed out; the 7-minute point
repeats the control row (19). The fitted slope is 0.76 tasks per scorer minute, so a 10% factor error
at 7 minutes costs about half a task, below run noise, while the nominal cap's 2.6x error cost 4-5.

### 4.3 Why one factor works here, and what the fit shows

One factor suffices when sessions are decode-bound. In all 76 full runs, model calls take 0.965-0.998 of session wall
time; the per-call fit has median R^2 0.988 and puts prefill at 2.3% of model time (median; max 5.5%). Fitted
coefficients (`timing_coefs.csv`; calls summed and the lowest R^2 shown over a setup's 1-2 runs; decode tokens/s =
1000 / (b + d x context in thousands)):

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
cost in decode (`b` +12%; +10% at 24k context). Overhead `c` falls while prefill `a` rises; the
two are hard to separate in the fit, so we do not read them individually. In two held-out runs the adapter changed no score
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
0.75-2.31, 7 dof)**; task-level SD 1.52 and 1.66. One run each ranks a true 2-task margin wrongly 11% of the time with the run SD and **20%** with the larger task-level SD
(1.66, held-out; 11% with two runs each), which we use for comparisons; a 1-task margin 34%; 80% power for a 2-task margin takes **11 runs** per agent. The cross-hardware pair differs by 0.9
train task-level SD of a difference (1.3 run-level SD), no more than identical runs.

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

A run-level factor averages over the drift that heavy concurrency causes (Section 4.3). One model family, one harness and Python repositories: tool-heavy
agents are less model-bound, which `timing` detects but one factor would not fix. The scorer reference rests on one
machine type (61 sequential sessions on Kaggle 4x L4, 32.9-33.8 tokens/s); host-to-host variation is not measured.
Only one converted setup was repeated (19, 18). Noise estimates rest on seven repeated pairs, the task-level SD assumes independent tasks,
and leaderboard noise is partly community-reported.

## 8. Conclusion

A wall-clock cap is a hardware setting unless converted. One factor from per-call logs, checked against the same logs,
makes local results comparable to the scorer's; two runs and a pooled SD make them comparable to each other. Nothing
here is specific to Gemma or this competition: any timed, decode-bound agent benchmark can use the same logs and
commands.

**Reproducibility.** Toolkit and tests (`tools/scorer_minutes/`, Apache-2.0), notebook
(`tools/scorer_minutes/notebook/`) and data (`docs/paper/aggregates/`): github.com/akanksha28pathak/gemma-4-developer-agent. Data: `speed_table.csv`
(106 runs), `speed_groups.csv` (4.1), `control_43.csv`, `control_twin.csv` and `budget_curve.csv` (4.2), `lora_sessions.csv` (4.4), `timing_coefs.csv` and `timing/` (4.3), `noise_long.csv`
(task id x run matrix), `run_labels.tsv`; no task content. Serving: vLLM 0.19.1, it-qat-w4a16 checkpoints.


## Extra: docs/paper/WRITEUP_FIELDS.md

# Kaggle Writeup fields (paper track)

Fill these into the Writeup form (Akanksha, Nov 9). Body = `docs/paper/SCORER_MINUTES.md` from "## Abstract" on.

* **Title:** Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets
* **Subtitle (one line):** One measured speed factor turns a wall-clock cap into the same budget on any GPU; a control
  on six setups reproduces the scorer's score, and repeated runs show what a one-task gap means.
* **Track / prize:** Best New Resource (toolkit + per-run dataset + protocol).
* **Card image:** `docs/paper/figures/fig2_control.png`.
* **Figures in the body:**
  * Figure 1 (`fig1_speed.png`, after table 4.1): "Per-session speed of every full Gemma 4 31B run (one dot per run),
    by setup. The same model ranges from 14 to 52 tokens/s; the dashed line is the scorer reference (33.8)."
  * Figure 2 (`fig2_control.png`, after table 4.2): "The control: one agent, the same 55 tasks. Converted caps (blue)
    match the scorer (black); the nominal cap on 4x L4 at 10 sessions (orange) times out 37 sessions and solves 15."
* **Links:** the main project repo only (Akanksha, Nov 7: no separate repo): github.com/akanksha28pathak/gemma-4-developer-agent
  (`tools/scorer_minutes/`, `docs/paper/aggregates/`, notebook in `tools/scorer_minutes/notebook/`). **Open:** the repo is
  private until Dec 2 (CLAUDE.md), so judges cannot open it unless it is made public or shared; decide before submitting.
* **Word count:** body 2,922 of 3,000 (`python3 scripts/paper/word_count.py`).
* **Authors:** Akanksha Pathak (sole author, confirmed Nov 7).


## Extra: docs/paper-track/release_notebook_outline.md

# Public release staging: layout and Kaggle notebook outline (G3)

Staged by `scripts/paper/stage_release.py` (not published; goes public Nov 9-10 after Akanksha's OK). The staging
audit checks every file for competition task ids, repository-qualified task names and task-text fields.

## Release layout

```
.gitignore
CITATION.cff
LICENSE
README.md
data/control_43.csv
data/noise_long.csv
data/run_labels.tsv
data/speed_groups.csv
data/speed_table.csv
data/timing_coefs.csv
notebook/scorer_minutes_demo.ipynb
pyproject.toml
src/scorer_minutes/__init__.py
src/scorer_minutes/__main__.py
src/scorer_minutes/calibrate.py
src/scorer_minutes/cli.py
src/scorer_minutes/logs.py
src/scorer_minutes/noise.py
src/scorer_minutes/stats.py
src/scorer_minutes/timing.py
tests/synth.py
tests/test_calibrate_timing.py
tests/test_noise.py
data/timing/*.json (103 per-run fit files)
```

## Notebook `notebook/scorer_minutes_demo.ipynb` (executes end to end on CPU in under a minute)

# scorer-minutes: convert wall-clock budgets between GPUs, and read agent scores with their noise

Companion notebook of the paper *Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under
wall-clock budgets*. It (1) installs the toolkit, (2) shows on synthetic logs that `calibrate` and `timing` recover known
speeds, and (3) rebuilds the paper's tables and figures from the released aggregates (`data/`). No GPU needed.

```python
import os, subprocess, sys
REPO_URL = "https://github.com/akanksha28pathak/scorer-minutes"
HERE = os.path.abspath("..") if os.path.exists("../data") else None
if HERE is None:  # Kaggle / Colab: clone the public repo
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "scorer-minutes"], check=True)
    HERE = os.path.abspath("scorer-minutes")
# standard library only: running from the source tree needs no install (pip install HERE also works)
os.environ["PYTHONPATH"] = os.path.join(HERE, "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
CLI = [sys.executable, "-m", "scorer_minutes"]
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(HERE, "tests"))
print("toolkit at", HERE)
```

Output (excerpt):

```
toolkit at /home/akanksha/gemma4_work/release/scorer-minutes
```

## 1. Synthetic check: a fast setup and a slow, shared one

Two setups with known costs per call: a "scorer" (decode 37.5 tokens/s, prefill 2,200 tokens/s, 0.12 s overhead) and a
"dev box" shared by ten sessions (16 tokens/s, 1,100 tokens/s, 0.4 s). `calibrate` reports the factor between
their session speeds (about 2.2: session speed includes prefill and the context slowdown, so it is not exactly
37.5 / 16), and `timing` recovers the per-call costs exactly (c = 0.4, a = 1/1.1, b = 1000/16, d = 0.23).

```python
import tempfile, json, pathlib
from synth import make_session, write_run
tmp = pathlib.Path(tempfile.mkdtemp())
def run(name, **kw):
    sessions = {f"s{i}": make_session(f"s{i}", seed=i, calls=25, **kw) for i in range(12)}
    return str(write_run(tmp / name, sessions, extra_rows=False))
scorer = run("scorer", decode_tok_s=37.5, prefill_tok_s=2200, overhead_s=0.12, ctx_slowdown=0.19)
dev = run("dev", decode_tok_s=16.0, prefill_tok_s=1100, overhead_s=0.40, ctx_slowdown=0.23)
g = "run/**/llm_calls/*.jsonl"
print(subprocess.run(CLI + ["calibrate", dev, "--reference", scorer, "--glob", g, "--cap-scorer-min", "7"],
                     capture_output=True, text=True).stdout[-1500:])
fit = json.loads(subprocess.run(CLI + ["timing", dev, "--glob", g], capture_output=True, text=True).stdout)["fit"]
print("fitted c, a, b, d:", fit["coef"], " R^2", fit["r2"])
```

Output (excerpt):

```
{
  "calibration": {
    "sessions": 12,
    "median_tok_s": 14.416,
    "p25_tok_s": 14.365,
    "p75_tok_s": 14.435,
    "scorer_tok_s": 31.901,
    "factor": 2.213,
    "factor_ci95": [
      2.21,
      2.221
    ],
    "cap_scorer_min": 7.0,
    "cap_wall_min": 15.49
...
```

## 2. Table 4.1: speed by GPU and concurrency (all full runs)

```python
import csv
rows = list(csv.DictReader(open(os.path.join(DATA, "speed_groups.csv"))))
for r in rows:
    print(f"{r['model']:8} {r['gpu']:20} {r['note']:22} x{r['sessions_in_parallel']:>2}  runs {r['runs']:>2}  "
          f"{r['tok_s_median']:>5} tok/s  factor {r['factor_min']}-{r['factor_max']}")
print("full runs:", sum(int(r["runs"]) for r in rows), " sessions:", sum(int(r["sessions"]) for r in rows))
```

Output (excerpt):

```
26B-A4B  Kaggle 4x L4                                x 4  runs  1   28.3 tok/s  factor 1.19-1.19
26B-A4B  Kaggle 4x L4         2 servers TP2          x 6  runs  1   22.1 tok/s  factor 1.52-1.52
26B-A4B  Kaggle 4x L4                                x 7  runs  2   14.6 tok/s  factor 2.24-2.4
26B-A4B  Kaggle 4x L4                                x 8  runs  2   22.2 tok/s  factor 1.43-1.64
26B-A4B  Kaggle 4x L4         2 servers TP2          x 8  runs  2   24.2 tok/s  factor 1.39-1.41
31B      A100 80 GB                                  x 5  runs 16   30.5 tok/s  factor 1.01-1.16
31B      A100 80 GB           + LoRA adapter         x 5  runs  1   26.1 tok/s  factor 1.29-1.29
31B      A100 80 GB           parallel attempts      x12  runs  1   16.9 tok/s  factor 1.99-1.99
31B      H100 SXM 80 GB                              x 5  runs  1   52.0 tok/s  factor 0.65-0.65
31B      Kaggle 4x L4                                x 1  runs  2   33.2 tok/s  factor 1.01-1.02
31B      Kaggle 4x L4                                x10  runs  6   14.1 tok/s  factor 2.13-2.55
31B      L40S 48 GB                                  x 5  runs  1   20.7 tok/s  factor 1.63-1.63
31B      RTX PRO 6000 96 GB                          x 2  runs  1   36.1 tok/s  factor 0.94-0.94
31B      RTX PRO 6000 96 GB                          x 5  runs 22   38.2 tok/s  factor 0.87-0.92
...
```

## 3. Table 4.2: the control (one agent, 55 tasks, seven setups) with exact McNemar tests

```python
from math import comb
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
for r in csv.DictReader(open(os.path.join(DATA, "control_43.csv"))):
    p = mcnemar(int(r["scorer_only"]), int(r["run_only"])) if r["mcnemar_p_vs_scorer"] else None
    print(f"{r['setup']:42} f={r['factor']:>5} cap={r['wall_cap_min']:>5} scorer-min={r['scorer_min']:>4} "
          f"solved={r['solved']:>2} timed-out={r['timed_out']:>2} McNemar p={'-' if p is None else round(p, 2)}")
```

Output (excerpt):

```
Kaggle 4x L4, 1 session (scorer)           f= 1.02 cap=  7.0 scorer-min= 6.9 solved=19 timed-out=11 McNemar p=-
H100 SXM, 5 sessions, converted            f= 0.65 cap=  4.8 scorer-min= 7.4 solved=19 timed-out=11 McNemar p=1.0
RTX PRO 6000, 5 sessions, converted        f=  0.9 cap=  6.0 scorer-min= 6.7 solved=19 timed-out=15 McNemar p=1.0
L40S, 5 sessions, converted                f= 1.63 cap= 11.7 scorer-min= 7.2 solved=20 timed-out= 7 McNemar p=1.0
Kaggle 4x L4, 10 sessions, converted       f=  2.4 cap= 15.0 scorer-min= 6.3 solved=20 timed-out= 8 McNemar p=1.0
Kaggle 4x L4, 10 sessions, nominal         f= 2.55 cap=  7.0 scorer-min= 2.7 solved=15 timed-out=37 McNemar p=0.22
A100, 5 sessions (earlier reference)       f= 1.08 cap=  7.0 scorer-min= 6.5 solved=17 timed-out=12 McNemar p=0.73
```

## 4. Table 4.3: fitted per-call timing coefficients

```python
for r in csv.DictReader(open(os.path.join(DATA, "timing_coefs.csv"))):
    print(f"{r['setup']:50} c={r['c_overhead_s']:>5}  a={r['a_s_per_k_prompt']:>5}  b={r['b_s_per_k_gen']:>5}  "
          f"d={r['d_s_per_k_gen_per_k_ctx']:>5}  decode {r['decode_tok_s_0k']}/{r['decode_tok_s_24k']} tok/s")
```

Output (excerpt):

```
Kaggle 4x L4, 1 session (scorer)                   c= 0.12  a= 0.46  b= 26.6  d= 0.19  decode 37.5/32.1 tok/s
Kaggle 4x L4, 10 sessions                          c= 0.41  a= 0.89  b= 62.5  d= 0.23  decode 16.1/14.7 tok/s
H100 SXM, 5 sessions                               c= 0.08  a= 0.25  b= 17.0  d= 0.14  decode 58.8/48.9 tok/s
L40S, 5 sessions                                   c= 0.22  a= 0.39  b= 45.5  d= 0.07  decode 22.0/21.2 tok/s
RTX PRO 6000, 5 sessions (v15, held-out)           c= 0.26  a= 0.17  b= 23.5  d= 0.08  decode 42.6/39.5 tok/s
RTX PRO 6000, 5 sessions, + adapter B (held-out)   c= 0.08  a= 0.24  b= 26.3  d= 0.07  decode 38.0/35.6 tok/s
```

## 5. Section 5: run noise from the repeated-run matrix

The noise pool leaves out `g31_v12e`, whose two runs used different GPUs (so the pool does not depend on the conversion
it is used to judge).

```python
pool = os.path.join(tmp, "noise_pool.csv")
with open(os.path.join(DATA, "noise_long.csv")) as src, open(pool, "w") as dst:
    for i, line in enumerate(src):
        if i == 0 or not line.startswith("g31_v12e,"):
            dst.write(line)
report = os.path.join(tmp, "noise.md")
subprocess.run(CLI + ["noise", pool, "--markdown", report], capture_output=True, text=True, check=True)
print(open(report).read())
print(subprocess.run(CLI + ["draws", "--sd", "1.6", "--score", "9"], capture_output=True, text=True).stdout)
```

Output (excerpt):

```
## Runs per config

| Split | Config | n | Runs | Solved per run | Mean | SD | Task-level SD | Flaky tasks |
|---|---|---|---|---|---|---|---|---|
| held-out | g31_refdavis | 55 | 1 | 17 | 17.00 | - | - | 0 |
| train | g31_refdavis | 55 | 1 | 20 | 20.00 | - | - | 0 |
| train | g31_v12 | 55 | 2 | 20, 19 | 19.50 | 0.71 | 1.58 | 5 |
| held-out | g31_v12b | 55 | 1 | 18 | 18.00 | - | - | 0 |
| train | g31_v12b | 55 | 1 | 18 | 18.00 | - | - | 0 |
| train | g31_v12d | 55 | 2 | 20, 18 | 19.00 | 1.41 | 1.41 | 4 |
| train | g31_v12e_rft | 55 | 1 | 17 | 17.00 | - | - | 0 |
| held-out | g31_v15 | 55 | 2 | 18, 20 | 19.00 | 1.41 | 1.73 | 6 |
| train | g31_v15 | 55 | 2 | 19, 21 | 20.00 | 1.41 | 1.41 | 4 |
| held-out | g31_v15_sft | 55 | 2 | 20, 21 | 20.50 | 0.71 | 1.58 | 5 |
...
```

## 6. Figure: per-session speed by setup

```python
import matplotlib.pyplot as plt
r31 = [r for r in rows if r["model"] == "31B"]
r31.sort(key=lambda r: float(r["tok_s_median"]))
labels = [f"{r['gpu']} {r['note']} x{r['sessions_in_parallel']}".replace("  ", " ") for r in r31]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(labels, [float(r["tok_s_median"]) for r in r31], color="#4a6fa5")
ax.axvline(33.8, color="#333", ls="--", lw=1)
ax.text(34.5, 0, "scorer reference 33.8", fontsize=8)
ax.set_xlabel("tokens/s per session (median of run medians)")
ax.set_title("Gemma 4 31B: per-session speed by setup")
plt.tight_layout(); plt.show()
```

Output: a bar chart of per-session speed by setup.


