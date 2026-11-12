# Kaggle Writeup fields (paper track)

Fill these into the Writeup form (Akanksha, Nov 9). Everything to paste or upload is in `docs/paper/submission/`
(built by `scripts/paper/build_submission.py`; body = the draft from "## Abstract" on, with the figures placed).

* **Title (76 characters; the form allows 80):** Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents
* **Subtitle (one line, 66 characters):** A toolkit for comparing time-limited coding agents across machines
* **Track / prize:** Best New Resource (toolkit + per-run dataset + protocol).
* **Card image:** `docs/paper/figures/fig2_control.png`.
* **Figures in the body** (placed by `scripts/paper/build_submission.py`):
  * Figure 1 (`fig1_speed.png`, after table 4.1): "One dot per run; the dashed line is the scorer's speed (33.8 tokens/s)."
  * Figure 2 (`fig2_control.png`, after table 4.2): "Converted time limits (blue) match the scorer (black); the
    unconverted limit (orange) does not. The A100 row of table 4.2 is not shown."
* **Links:** github.com/akanksha28pathak/scorer-minutes-paper (the paper repo: toolkit, data, notebook, scripts,
  submission folder; built from the main repo by `scripts/paper/build_paper_repo.py`, no competition data). Public
  since Nov 11.
* **Media gallery:** `fig2_control.png` and `fig1_speed.png` (no video).
* **Word count:** body 2,941 of 3,000 (2,985 with the figure captions; `python3 scripts/paper/build_submission.py`).
* **Authors:** Akanksha Pathak (sole author, confirmed Nov 7).
