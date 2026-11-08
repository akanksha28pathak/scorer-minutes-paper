"""Build a standalone git repo with every artifact of the research-track paper (Akanksha, Nov 7), outside the main repo.

  python3 scripts/paper/build_paper_repo.py [--out ~/gemma4_work/scorer-minutes-paper]

Layout of the new repo:
  submission/   the Kaggle Writeup folder (README, CHECKLIST, FIELDS, body.md, figures, notebook, data, toolkit zip)
  paper/        the draft, the Writeup fields, the figures
  toolkit/      tools/scorer_minutes (source, tests, notebook, README, LICENSE, CITATION)
  data/         the aggregates behind every table, with the data card (task ids anonymized)
  analysis/     the scripts that build the tables from run logs (they expect the main repo's results/ layout)
  scripts/      paper scripts (regeneration, figures, review packets, submission and release builders)
  docs/         plan, requirements, review gates and every reviewer's response
Rebuilds the folder from scratch each time and commits the result (one commit per rebuild). Audited for competition
task ids, task names and task text before committing; pushed only with --push (origin:
github.com/akanksha28pathak/scorer-minutes-paper, private, created by Akanksha Nov 7).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper"))
import stage_release as sr  # noqa: E402  (data list, data card, anonymization and audit shared with the release)

ANALYSIS = ["speed_table.py", "speed_groups.py", "noise_table.py", "control_table.py", "timing_coef_table.py",
            "lora_sessions.py", "budget_curve.py"]
PAPER_TRACK = ["README.md", "PLAN.md", "1_scorer_minutes.md", "1_scorer_minutes_requirements.md",
               "release_notebook_outline.md"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.egg-info", "build")

README = """# Scorer-minutes: research-track paper (all artifacts)

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
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/gemma4_work/scorer-minutes-paper"))
    ap.add_argument("--push", action="store_true", help="push to the repo's origin after committing")
    a = ap.parse_args()
    out = Path(a.out)
    subprocess.run([sys.executable, str(ROOT / "scripts/paper/build_submission.py")], check=True)
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():  # keep .git, rebuild everything else
        if child.name != ".git":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(ROOT / "docs/paper/submission", out / "submission", ignore=IGNORE)
    (out / "paper").mkdir()
    for f in ("SCORER_MINUTES.md", "WRITEUP_FIELDS.md"):
        shutil.copy(ROOT / "docs/paper" / f, out / "paper" / f)
    shutil.copytree(ROOT / "docs/paper/figures", out / "paper/figures")
    shutil.copytree(sr.TOOL, out / "toolkit", ignore=IGNORE)
    (out / "data").mkdir()
    for f in sr.DATA_FILES:
        shutil.copy(sr.AGG / f, out / "data" / f)
    shutil.copytree(sr.AGG / "timing", out / "data/timing")
    sr.anonymize_noise(out / "data/noise_long.csv")
    (out / "data/README.md").write_text(sr.DATA_CARD)
    (out / "analysis").mkdir()
    for f in ANALYSIS:
        shutil.copy(ROOT / "analysis" / f, out / "analysis" / f)
    shutil.copytree(ROOT / "scripts/paper", out / "scripts", ignore=IGNORE)
    (out / "docs").mkdir()
    for f in PAPER_TRACK:
        shutil.copy(ROOT / "docs/paper-track" / f, out / "docs" / f)
    shutil.copytree(ROOT / "docs/paper-track/reviews", out / "docs/reviews")
    shutil.copy(sr.TOOL / "LICENSE", out / "LICENSE")
    (out / "README.md").write_text(README)
    (out / ".gitignore").write_text("__pycache__/\n*.egg-info/\nbuild/\n")
    problems = sr.audit(out)
    if problems:
        sys.exit("AUDIT FAILED (nothing committed):\n  " + "\n  ".join(problems[:30]))
    git = lambda *args: subprocess.run(["git", "-C", str(out), *args], check=True, capture_output=True, text=True)
    if not (out / ".git").exists():
        git("init", "-q", "-b", "main")
    git("add", "-A")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain").stdout.strip():
        git("commit", "-q", "-m", f"Rebuild from main repo {head} (scripts/paper/build_paper_repo.py)")
    if a.push:
        git("push", "-q", "origin", "main")
    files = sum(1 for f in out.rglob("*") if f.is_file() and ".git" not in f.parts)
    print(f"{out}: {files} files, audit OK, {git('log', '--oneline', '-1').stdout.strip()}")


if __name__ == "__main__":
    main()
