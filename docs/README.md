# Paper track: three independent research ideas (Nov 4)

**Current: idea 1 (scorer-minutes) chosen. Plan of action and log: [PLAN.md](PLAN.md).**

Competition: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper. Deadline **Thu Nov 12, 23:59 UTC**,
Kaggle Writeup of at most 3,000 words, judged equally on Novelty, Quality (generality), Relevance, Verifiability,
Clarity (0-5 each). Prizes include **Best New Resource** ($10k, our target).

Three ideas, each built on work already in this repo, each independent of the others and of the existing drafts
(`docs/paper/V9B_V9C.md`: the v9b -> v9c harness change + localization; `docs/RESEARCH_OPTIONS.md` and
`docs/BASELINE.md`: the localization benchmark). Pick one; they do not depend on each other.

| # | Idea | Core claim | Resource | What is already measured | Missing before Nov 12 |
|---|---|---|---|---|---|
| 1 | [Scorer-minutes](1_scorer_minutes.md) | Agent results under wall-clock budgets are not comparable across GPUs or between runs unless time is normalized to the scorer's decode speed and noise is reported | Calibration protocol + `scripts/vast/session_speed.py` + a noise table from ~30 runs | Speed factors on 4 GPU setups, a control that reproduces A100 results on a different GPU, run-to-run and leaderboard noise | One more matched control, the noise table, the write-up |
| 2 | [Green but wrong](2_green_but_wrong.md) | Coding agents verify fixes with tests that pass on incomplete fixes; local green is a weak signal of hidden-test success | A per-session "local verification vs hidden grade" table over ~1,500 sessions (aggregates only) + a tiny analysis script | H2: 9 of 10 green sessions still failed the hidden tests; failure-mode table over 4 agents | Compute the confusion matrix over all logged sessions; one prompt ablation |
| 3 | [Hinted rationalization](3_hinted_rationalization.md) | Hint-conditioned trace synthesis (STaR-style) for software agents works only with explicit leak auditing; we give the method, the audit and the trade-off between hint strength, solve rate and exploration fidelity | Open SWE-smith-based hinted + unhinted trace set (MIT upstream, no competition data) + strip / leak-audit tool | Hint levels, stripping verified on 432 prompts, leak filter false-positive rate, the collection running tonight | Tonight's numbers, leak and fidelity metrics, ideally one LoRA ablation |

Recommendation: **1** is the safest (most of the evidence exists, fully reproducible on CPU, clear resource) and fits
"Best New Resource"; **3** has the highest upside but depends on tonight's collection and ideally a training run;
**2** is a strong, simple finding that needs one day of CPU analysis.

Rules that apply to all three: no competition task text, code, patches or traces in the paper or the released resource
(aggregates and task ids only); every number regenerable by a script in this repo (CLAUDE.md).
