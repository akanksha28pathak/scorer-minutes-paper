# Scorer-minutes: research-track paper (all artifacts)

Kaggle "Google - The Gemma 4 Developer Agent" paper track. Author: Akanksha Pathak. Target: Best New Resource.
Built from the main project repo by `scripts/paper/build_paper_repo.py`; do not edit here, edit there and rebuild.

| Folder | Contents |
|---|---|
| `submission/` | Everything for the Kaggle Writeup; start with `submission/README.md` and `submission/CHECKLIST.md` |
| `paper/` | The draft (`SCORER_MINUTES.md`), the Writeup fields, the figures |
| `toolkit/` | The `scorer-minutes` toolkit (Python standard library only; 16 tests; Apache-2.0) and its demo notebook |
| `data/` | Aggregates behind every table, with a data card (`data/README.md`); task ids anonymized, no task content |
| `analysis/`, `scripts/` | Scripts that build the tables and figures from run logs (they expect the main repo's `results/` layout) |
| `docs/` | Plan, requirements, review gates, and every reviewer's response with the action taken |

Run the toolkit tests: `cd toolkit && PYTHONPATH=src python3 -m pytest -q tests`.
