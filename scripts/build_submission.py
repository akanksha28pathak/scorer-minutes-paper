"""Build the paper-track submission folder docs/paper/submission/ from the single sources (the draft, the Writeup fields
and the figures), so it never drifts from them. Rerun after every edit to the draft.

  python3 scripts/paper/build_submission.py

Writes every submission artifact in one place (Akanksha, Nov 7; the shareable copy from stage_release.py is separate):
body.md (the Writeup body: the draft from "## Abstract" on, with the two figures placed after tables 4.1 and 4.2),
fig1_speed.png, fig2_control.png, card.png (the card image), FIELDS.md (title, subtitle, track, captions, links),
CHECKLIST.md (the order to fill the Kaggle form), notebook/ (the demo notebook), data/ (the aggregates behind every
table, data card included; task ids in noise_long.csv anonymized as in the shareable copy) and scorer-minutes.zip (the
toolkit: src, tests, README, LICENSE, CITATION). Prints the body word count (limit 3,000).
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper/SCORER_MINUTES.md"
FIELDS = ROOT / "docs/paper/WRITEUP_FIELDS.md"
FIG = ROOT / "docs/paper/figures"
OUT = ROOT / "docs/paper/submission"
FIG1 = ("![Figure 1](fig1_speed.png)\n\n"
        "*Figure 1. One dot per run; the dashed line is the scorer's speed (33.8 tokens/s).*\n\n")
FIG2 = ("![Figure 2](fig2_control.png)\n\n"
        "*Figure 2. Converted time limits (blue) match the scorer (black); the unconverted limit (orange) does not. The A100 row of table 4.2 is not shown.*\n\n")


README = """# Research-track submission: {title}

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
| `body.md` | Writeup body to paste ({n:,} words with captions; limit 3,000), figures placed after tables 4.1 and 4.2 |
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
"""


def words(text: str) -> int:
    return sum(1 for w in text.split() if re.search(r"[A-Za-z0-9]", w))


def main() -> None:
    text = PAPER.read_text()
    title = text.splitlines()[0].lstrip("# ").strip()
    body = text[text.index("## Abstract"):]
    for head, fig in (("### 4.2 ", FIG1), ("### 4.3 ", FIG2)):
        i = body.index(head)
        body = body[:i] + fig + body[i:]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "body.md").write_text(body)
    for name in ("fig1_speed.png", "fig2_control.png"):
        shutil.copy(FIG / name, OUT / name)
    shutil.copy(FIG / "fig2_control.png", OUT / "card.png")
    sys.path.insert(0, str(ROOT / "scripts/paper"))
    import stage_release as sr  # the same data list, data card and task-id anonymization as the shareable copy
    data = OUT / "data"
    shutil.rmtree(data, ignore_errors=True)
    data.mkdir()
    for f in sr.DATA_FILES:
        shutil.copy(sr.AGG / f, data / f)
    shutil.copytree(sr.AGG / "timing", data / "timing")
    sr.anonymize_noise(data / "noise_long.csv")
    (data / "README.md").write_text(sr.DATA_CARD)
    (OUT / "notebook").mkdir(exist_ok=True)
    shutil.copy(sr.TOOL / "notebook/scorer_minutes_demo.ipynb", OUT / "notebook/scorer_minutes_demo.ipynb")
    with zipfile.ZipFile(OUT / "scorer-minutes.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(sr.TOOL.rglob("*")):
            rel = f.relative_to(sr.TOOL)
            if f.is_file() and not ({"__pycache__", ".pytest_cache", "build", "notebook"} & set(rel.parts)) and ".egg-info" not in str(rel):
                z.write(f, Path("scorer-minutes") / rel)
    problems = sr.audit(OUT)
    if problems:
        sys.exit("AUDIT FAILED:\n  " + "\n  ".join(problems[:20]))
    fields = FIELDS.read_text().replace("`docs/paper/figures/fig2_control.png`", "`card.png`")
    (OUT / "FIELDS.md").write_text(fields)
    n = words(body)
    (OUT / "CHECKLIST.md").write_text(f"""# Submitting the Writeup (paper track)

Generated by `scripts/paper/build_submission.py` from `docs/paper/SCORER_MINUTES.md` and `WRITEUP_FIELDS.md`; edit
those, then rerun. Body: **{n:,} words** (limit 3,000). Deadline Thu Nov 12 23:59 UTC = 6:59 PM ET
(ties go to the earliest entry; a saved draft is not judged, press Submit).

1. Open https://www.kaggle.com/competitions/gemma-4-developer-agent-paper and start a new Writeup.
2. **Title** (max 80 characters): {title}
3. **Subtitle**: from `FIELDS.md`.
4. **Card image**: upload `card.png`. **Media gallery**: `fig2_control.png` and `fig1_speed.png`.
5. **Body**: paste `body.md`. Upload `fig1_speed.png` and `fig2_control.png` and put each where its `![...]` line is
   (if the editor does not take markdown images, insert the image there and keep the italic caption below it).
6. **Links / attachments**: https://github.com/akanksha28pathak/scorer-minutes-paper (public); if
   files can be attached, `scorer-minutes.zip` (toolkit), `data/` (aggregates + data card) and
   `notebook/scorer_minutes_demo.ipynb`.
7. Check: tables render, both figures show, the word counter reads <= 3,000, no draft notes left.
8. Submit; then tell Claude so NEXT_STEPS.md and PLAN.md get the time and link.
""")
    (OUT / "README.md").write_text(README.format(title=title, n=n))
    print(f"{OUT.relative_to(ROOT)}: README.md, body.md ({n:,} words), 3 images, FIELDS.md, CHECKLIST.md, notebook/, data/, "
          f"scorer-minutes.zip; audit OK")


if __name__ == "__main__":
    main()
