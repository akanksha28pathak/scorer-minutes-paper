# Review packet: g3-resource (2026-11-07)

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

G3 reviews the resource itself, not the paper: the toolkit (tools/scorer_minutes/ in the main project repository: README, license, citation), the paper aggregates with their data card (docs/paper/aggregates/), and the companion notebook (tools/scorer_minutes/notebook/). The notebook was executed end to end on CPU; its outputs (excerpted below) reproduce the paper's tables 4.1-4.3 and the pooled noise SD 1.13 (95% CI 0.75-2.31). Judge it as for Best New Resource: is it usable by someone outside this competition, is it documented enough, what is missing? Section 3 of the paper is included for context.

## Paper (sections 3; full draft is 2922 words, not counting markdown table pipes)

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



## Extra: tools/scorer_minutes/README.md

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
noise-aware evaluation of coding agents under wall-clock budgets* (Kaggle Gemma 4 Developer Agent paper track, 2026).

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
`control_43.csv` and `control_twin.csv` (table 4.2, the control), `lora_sessions.csv` (section 4.4), `timing_coefs.csv` and `timing/` (table 4.3, per-run fits),
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


