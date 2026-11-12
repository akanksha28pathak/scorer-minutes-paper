"""Build the scorer-minutes demo notebook (tools/scorer_minutes/notebook/) and stage a copy of what would be shared
(toolkit + paper aggregates + notebook) outside the repo, then audit it. Akanksha, Nov 7: everything stays in the main
repo (no separate repo); the staged copy is only for the no-task-content audit.

  python3 scripts/paper/stage_release.py [--out ~/gemma4_work/release/scorer-minutes] [--repo-url URL] [--run-notebook]

Copies tools/scorer_minutes (src, tests, pyproject, README, LICENSE, CITATION), the paper aggregates in
docs/paper/aggregates (task ids in noise_long.csv replaced by t001...; the mapping is never written), and builds
notebook/scorer_minutes_demo.ipynb. The audit fails on any competition task id, repository-qualified task name or
task-text field in the staged tree. Nothing is pushed or uploaded: publishing needs Akanksha's OK (PLAN.md, Nov 9).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/scorer_minutes"
AGG = ROOT / "docs/paper/aggregates"
DATA_FILES = ["speed_table.csv", "speed_groups.csv", "control_43.csv", "control_twin.csv", "timing_coefs.csv",
              "lora_sessions.csv", "budget_curve.csv", "run_labels.tsv"]
FORBIDDEN = ["problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS", "test_patch", '"patch"', "diff --git"]


DATA_CARD = """# Data card: scorer-minutes paper aggregates

Aggregates from 106 logged runs (4,757 sessions) of Gemma 4 coding agents (31B, E4B, 26B-A4B; quantized w4a16
checkpoints served by vLLM) in the Kaggle Gemma 4 Developer Agent harness, Oct-Nov 2026. Six GPU models: Kaggle 4x L4,
A100 80 GB, H100 SXM, L40S, RTX PRO 6000, RTX 5080. No task text, code, patches, prompts or model outputs are included;
task ids in `noise_long.csv` are replaced by `t001`... (stable within a split). License: Apache-2.0, as the code.

| File | One row per | Columns (main) |
|---|---|---|
| `speed_table.csv` | logged run | `run`, `label` (hardware, tensor parallelism, sessions in parallel, model), `sessions`, `median_tok_s` / `p25_tok_s` / `p75_tok_s` (per-session tokens/s), `factor` and its 95% CI vs the scorer reference (33.8 tokens/s), `cap_wall_min` (wall cap for 7 scorer minutes), `model_share`, `fit_r2`, `prefill_share` / `decode_share` / `overhead_share` of model time, `decode_tok_s_ctx0` / `decode_tok_s_ctx24k`, `late_over_early` |
| `speed_groups.csv` | setup (paper table 4.1) | `gpu`, `model`, `sessions_in_parallel` (attempts x workers when attempts ran in parallel), `note`, `runs`, `sessions`, tokens/s median / min / max, factor median / min / max |
| `control_43.csv` | control setup (paper table 4.2) | `setup`, `runs`, `factor` (full run), `wall_cap_min` (set before the run), `scorer_min` = cap / factor, `solved`, `timed_out` (harness timeout field), `timed_out_solved`, paired counts vs the scorer (`both`, `scorer_only`, `run_only`), exact McNemar p |
| `control_twin.csv` | pair | converted vs nominal cap on the same hardware: tasks solved only by each, exact McNemar p |
| `budget_curve.csv` | run (paper section 4.2) | one agent, one setup, caps for 3.5 / 7 / 10 scorer minutes: factor, scorer minutes, solved, timed out |
| `lora_sessions.csv` | setup (paper section 4.1) | mean / median session seconds with and without adapter B (110 sessions each), ratio, projected hours |
| `timing_coefs.csv` | setup (paper section 4.3) | fitted `t = c + a*new_prompt + b*gen + d*gen*context` (seconds, tokens in thousands): `c_overhead_s`, `a_s_per_k_prompt`, `b_s_per_k_gen`, `d_s_per_k_gen_per_k_ctx`, decode tokens/s at 0k and 24k context |
| `timing/<run>.json` | logged run | output of `scorer-minutes timing`: model share, time split, tokens/s per context bin, early vs late calls, the fit (per-session flags reduced to a count) |
| `noise_long.csv` | (run, task) | `config` (agent version), `split` (`train`, `held-out`; 55 tasks each), `run`, `task` (anonymized), `solved` (0/1) |
| `run_labels.tsv` | logged run | run name, hardware label |

Grading: the competition's sandbox image, strict (all hidden tests must pass). The scorer reference is a sequential run
(one session at a time) on Kaggle 4x L4 with tensor parallelism 4, the competition's scoring setup.

## Tasks and splits

The tasks themselves are not in this repository: the competition rules forbid redistributing them. They are on the
competition's data page (kaggle.com/competitions/gemma-4-developer-agent/data), together with the agent program and its
documentation (`HARNESS_README.md`).

* **Public tasks:** 129, from four open-source Python projects (FastAPI 67, Rich 48, Requests 13, httpx 1). Each task
  carries the project's code just before a real bug fix (a repository snapshot at a named commit), the bug report, the
  maintainers' fix, and the tests that the fix makes pass.
* **Pool used here:** the 110 tasks (63 FastAPI, 47 Rich) on which the maintainers' fix passes in our grading sandbox.
* **`train`** (55 tasks: 31 FastAPI, 24 Rich): used to develop agents. The control experiment (paper section 4.2) runs
  on it.
* **`held-out`** (55 tasks: 32 FastAPI, 23 Rich): never used for development, only measured. The two splits share no
  task.
* **Hidden test set:** about 120 tasks from private repositories, scored only by Kaggle (about 58 of them on the public
  leaderboard). None of our files contains results on it except the public scores quoted in the paper.

Task ids are anonymized (`t001`...), so rows of `noise_long.csv` can be compared across runs but not mapped back to a
competition task.
"""


def task_ids() -> set[str]:
    ids = {r["task"] for r in csv.DictReader(open(AGG / "noise_long.csv"))}
    for f in (ROOT / "splits").glob("*.txt"):
        ids |= {line.strip() for line in open(f) if line.strip() and not line.startswith("#")}
    return {i for i in ids if len(i) >= 6}


def anonymize_noise(dst: Path) -> None:
    rows = list(csv.DictReader(open(AGG / "noise_long.csv")))
    order = sorted({(r["split"], r["task"]) for r in rows})
    alias = {key: f"t{i + 1:03d}" for i, key in enumerate(order)}
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "task": alias[(r["split"], r["task"])],
                        "split": {"train55": "train", "val55": "held-out"}.get(r["split"], r["split"])})


_ID = itertools.count(1)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": f"c{next(_ID)}", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": f"c{next(_ID)}", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(repo_url: str) -> dict:
    cells = [
        md("""
# scorer-minutes: convert wall-clock budgets between GPUs, and read agent scores with their noise

Companion notebook of the paper *Scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under
wall-clock budgets*. It (1) installs the toolkit, (2) shows on synthetic logs that `calibrate` and `timing` recover known
speeds, and (3) rebuilds the paper's tables and figures from the released aggregates (`data/`). No GPU needed.
"""),
        code(f"""
import os, subprocess, sys
REPO_URL = "{repo_url}"
HERE, DATA = None, None
if os.path.exists("../data"):  # staged copy: tools + data side by side
    HERE, DATA = os.path.abspath(".."), os.path.abspath("../data")
elif os.path.exists("../../../docs/paper/aggregates"):  # inside the main repo: tools/scorer_minutes/notebook
    HERE, DATA = os.path.abspath(".."), os.path.abspath("../../../docs/paper/aggregates")
else:  # Kaggle / Colab: clone the project repo
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "repo"], check=True)
    HERE, DATA = os.path.abspath("repo/tools/scorer_minutes"), os.path.abspath("repo/docs/paper/aggregates")
# standard library only: running from the source tree needs no install (pip install HERE also works)
os.environ["PYTHONPATH"] = os.path.join(HERE, "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
CLI = [sys.executable, "-m", "scorer_minutes"]
sys.path.insert(0, os.path.join(HERE, "tests"))
print("data:", os.path.relpath(DATA))
"""),
        md("""
## 1. Synthetic check: a fast setup and a slow, shared one

Two setups with known costs per call: a "scorer" (decode 37.5 tokens/s, prefill 2,200 tokens/s, 0.12 s overhead) and a
"dev box" shared by ten sessions (16 tokens/s, 1,100 tokens/s, 0.4 s). `calibrate` reports the factor between
their session speeds (about 2.2: session speed includes prefill and the context slowdown, so it is not exactly
37.5 / 16), and `timing` recovers the per-call costs exactly (c = 0.4, a = 1/1.1, b = 1000/16, d = 0.23).
"""),
        code("""
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
"""),
        md("""
## 2. Table 4.1: speed by GPU and concurrency (all full runs)
"""),
        code("""
import csv
rows = list(csv.DictReader(open(os.path.join(DATA, "speed_groups.csv"))))
for r in rows:
    print(f"{r['model']:8} {r['gpu']:20} {r['note']:22} x{r['sessions_in_parallel']:>2}  runs {r['runs']:>2}  "
          f"{r['tok_s_median']:>5} tok/s  factor {r['factor_min']}-{r['factor_max']}")
print("full runs:", sum(int(r["runs"]) for r in rows), " sessions:", sum(int(r["sessions"]) for r in rows))
"""),
        md("""
## 3. Table 4.2: the control (one agent, 55 tasks, seven setups) with exact McNemar tests
"""),
        code("""
from math import comb
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
for r in csv.DictReader(open(os.path.join(DATA, "control_43.csv"))):
    p = mcnemar(int(r["scorer_only"]), int(r["run_only"])) if r["mcnemar_p_vs_scorer"] else None
    print(f"{r['setup']:42} f={r['factor']:>5} cap={r['wall_cap_min']:>5} scorer-min={r['scorer_min']:>4} "
          f"solved={r['solved']:>2} timed-out={r['timed_out']:>2} McNemar p={'-' if p is None else round(p, 2)}")
"""),
        md("""
## 4. Section 4.3: fitted per-call timing coefficients
"""),
        code("""
for r in csv.DictReader(open(os.path.join(DATA, "timing_coefs.csv"))):
    print(f"{r['setup']:50} c={r['c_overhead_s']:>5}  a={r['a_s_per_k_prompt']:>5}  b={r['b_s_per_k_gen']:>5}  "
          f"d={r['d_s_per_k_gen_per_k_ctx']:>5}  decode {r['decode_tok_s_0k']}/{r['decode_tok_s_24k']} tok/s")
"""),
        md("""
## 5. Section 5: run noise from the repeated-run matrix

The noise pool leaves out `g31_v12e`, whose two runs used different GPUs (so the pool does not depend on the conversion
it is used to judge).
"""),
        code("""
pool = os.path.join(tmp, "noise_pool.csv")
with open(os.path.join(DATA, "noise_long.csv")) as src, open(pool, "w") as dst:
    for i, line in enumerate(src):
        if i == 0 or not line.startswith("g31_v12e,"):
            dst.write(line)
report = os.path.join(tmp, "noise.md")
subprocess.run(CLI + ["noise", pool, "--markdown", report], capture_output=True, text=True, check=True)
print(open(report).read())
print(subprocess.run(CLI + ["draws", "--sd", "1.6", "--score", "9"], capture_output=True, text=True).stdout)
"""),
        md("""
## 6. Figure: per-session speed by setup
"""),
        code("""
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
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def audit(out: Path) -> list[str]:
    ids = task_ids()
    pat = re.compile(r"\b(fastapi|rich|requests|httpx)[_-]{1,2}\d{2,}\b")
    problems = []
    for f in out.rglob("*"):
        if not f.is_file() or ".git" in f.parts:
            continue
        text = f.read_text(errors="ignore")
        for word in FORBIDDEN if f.name != "stage_release.py" else []:  # this file holds the word list itself
            if word in text:
                problems.append(f"{f.relative_to(out)}: contains {word!r}")
        if pat.search(text):
            problems.append(f"{f.relative_to(out)}: task-like id {pat.search(text).group(0)!r}")
        tokens = set(re.findall(r"[A-Za-z0-9_.\-]{6,}", text))
        hit = sorted(tokens & ids)
        if hit:
            problems.append(f"{f.relative_to(out)}: task ids {hit[:3]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/gemma4_work/release/scorer-minutes"))
    ap.add_argument("--repo-url", default="https://github.com/akanksha28pathak/gemma-4-developer-agent")
    ap.add_argument("--run-notebook", action="store_true", help="execute the notebook in place (needs nbclient)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(TOOL / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(TOOL / "tests", out / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ["pyproject.toml", "README.md", "LICENSE", "CITATION.cff"]:
        shutil.copy(TOOL / f, out / f)
    (out / "data").mkdir()
    for f in DATA_FILES:
        shutil.copy(AGG / f, out / "data" / f)
    shutil.copytree(AGG / "timing", out / "data/timing")
    anonymize_noise(out / "data/noise_long.csv")
    (out / "data/README.md").write_text(DATA_CARD)
    (out / ".gitignore").write_text("__pycache__/\n*.egg-info/\nbuild/\n")
    (out / "notebook").mkdir()
    nb = out / "notebook/scorer_minutes_demo.ipynb"
    nb.write_text(json.dumps(notebook(a.repo_url), indent=1))
    repo_nb = TOOL / "notebook/scorer_minutes_demo.ipynb"  # the copy that lives in the main repo
    repo_nb.parent.mkdir(exist_ok=True)
    repo_nb.write_text(json.dumps(notebook(a.repo_url), indent=1))
    (AGG / "README.md").write_text(DATA_CARD.replace("task ids in `noise_long.csv` are replaced by `t001`... (stable within a split)",
                                                     "the shared copy replaces task ids in `noise_long.csv` by `t001`..."))
    if a.run_notebook:
        for path in (nb, repo_nb):
            subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            str(path)], check=True, cwd=path.parent)
    problems = audit(out)
    files = sum(1 for f in out.rglob("*") if f.is_file())
    if problems:
        print("AUDIT FAILED:\n  " + "\n  ".join(problems[:40]))
        sys.exit(1)
    print(f"staged {files} files in {out}; audit OK (no task ids, task names or task text)")


if __name__ == "__main__":
    main()
