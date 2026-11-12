# Research-track submission: Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents

Kaggle "Google - The Gemma 4 Developer Agent" paper track. Author: Akanksha Pathak. Target prize: **Best New Resource**
(also judged for Best Paper). Deadline Thu Nov 12 23:59 UTC (6:59 PM ET).

## What the paper says

Coding-agent benchmarks cap each session by wall-clock time, but the same cap buys a different amount of model work on
different hardware and concurrency. The paper proposes **scorer minutes** (wall minutes divided by a speed factor measured
from ordinary per-call logs), releases a dependency-free toolkit that computes it, checks when one factor is enough, and
reports run-to-run noise, plus a one-page reporting protocol.

Key results (all from `data/`):
* 106 logged runs (4,757 sessions) of Gemma 4 agents on six GPU models: the same 31B model runs at 13-52 tokens/s per
  session; the scorer's own 4x L4 at 10 parallel sessions is 2.1-2.6x slower than when it scores one session at a time.
* Control (one agent, 55 tasks, seven setups): the scorer setup solved 19; converted caps on four other setups solved
  19-20; the unconverted cap solved 15 with 37 of 55 sessions timed out.
* Budget curve (3.5 / 7 / 10 scorer minutes on one setup): 16 / 18 / 21 solved, about 0.76 tasks per scorer minute, so a
  10% factor error costs about half a task.
* Per-call timing fit (overhead, prefill, decode, context slowdown) from client logs alone; a LoRA adapter makes
  sessions 12% longer and pushes the agent toward the 12-hour limit.
* Noise: identical runs differ by an SD of 1.1-1.7 tasks of 55; one run each ranks a 2-task gap wrongly 11-20% of the time.

## This folder (everything for the submission; the shareable public copy is built separately)

| File | Use |
|---|---|
| `CHECKLIST.md` | Step-by-step for filling the Kaggle Writeup form (start here) |
| `FIELDS.md` | Title, subtitle, track, card image, figure captions, links, author, word count |
| `body.md` | Writeup body to paste (2,985 words with captions; limit 3,000), figures placed after tables 4.1 and 4.2 |
| `card.png`, `fig1_speed.png`, `fig2_control.png` | Card image and the two figures |
| `notebook/scorer_minutes_demo.ipynb` | Demo notebook, executed: synthetic check of the toolkit, then tables 4.1-4.2, the timing fits of section 4.3 and the noise numbers rebuilt from the data |
| `data/` | Aggregates behind every table, with a data card (`data/README.md`: files, tasks and splits); task ids anonymized, no task content |
| `scorer-minutes.zip` | The toolkit (Python standard library only): source, tests, README, Apache-2.0 license, citation |

## Sources and reproduction (main repo)

* Draft: `docs/paper/SCORER_MINUTES.md`; Writeup fields: `docs/paper/WRITEUP_FIELDS.md`; figures: `docs/paper/figures/`.
* Toolkit: `tools/scorer_minutes/` (16 tests); data: `docs/paper/aggregates/`.
* Regenerate every table and figure from the run logs: `bash scripts/paper/scorer_minutes_pc.sh`; rebuild this folder:
  `python3 scripts/paper/build_submission.py` (also audits it for competition task content).
* Plan, review gates and every reviewer objection with its fix: `docs/paper-track/PLAN.md` and
  `docs/paper-track/reviews/` (Gemini: G0 3.6, G1 4.1, G1b 4.3, G4 4.6).

## Public copy

github.com/akanksha28pathak/scorer-minutes-paper (public since Nov 11): toolkit, data, notebook, scripts and this folder.
