# Kaggle Writeup fields (paper track)

Fill these into the Writeup form (Akanksha, Nov 9). Everything to paste or upload is in `docs/paper/submission/`
(built by `scripts/paper/build_submission.py`; body = the draft from "## Abstract" on, with the figures placed).

* **Title:** Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets
* **Subtitle (one line):** One measured speed factor turns a wall-clock cap into the same budget on any GPU; a control
  on six setups reproduces the scorer's score, and repeated runs show what a one-task gap means.
* **Track / prize:** Best New Resource (toolkit + per-run dataset + protocol).
* **Card image:** `docs/paper/figures/fig2_control.png`.
* **Figures in the body** (placed by `scripts/paper/build_submission.py`):
  * Figure 1 (`fig1_speed.png`, after table 4.1): "One dot per run; dashed line: the scorer reference (33.8 tokens/s)."
  * Figure 2 (`fig2_control.png`, after table 4.2): "Converted caps (blue) match the scorer (black); the nominal cap
    (orange) does not."
* **Links:** github.com/akanksha28pathak/scorer-minutes-paper (the paper repo: toolkit, data, notebook, scripts,
  submission folder; built from the main repo by `scripts/paper/build_paper_repo.py`, no competition data). **Open:**
  it is private; make it public (or share it) before submitting, or judges cannot open it.
* **Word count:** body 2,922 of 3,000 (2,944 with the figure captions; `python3 scripts/paper/build_submission.py`).
* **Authors:** Akanksha Pathak (sole author, confirmed Nov 7).
