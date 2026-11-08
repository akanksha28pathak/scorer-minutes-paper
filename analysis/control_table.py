"""Section 4.3 control table of the scorer-minutes paper: one agent (g31_v12e), the same 55 train55 tasks, six setups,
compared task by task with the scorer reference (Kaggle 4x L4, one session at a time, 7 min).

  python3 analysis/control_table.py [--out docs/paper/aggregates/control_43.csv]

Reads results/kaggle_runs/<run>/run/**/rows.jsonl (solved, duration) and docs/paper/aggregates/speed_table.csv
(measured factor). Writes one row per setup: factor, wall cap, scorer minutes, solved, sessions the harness timed out
("Agent exceeded session timeout"; a timed-out session can still be solved, since the working tree is graded), and the
paired counts vs the scorer (both / scorer only / run only) with an exact two-sided McNemar p. Task ids and counts only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = "results/kaggle_runs"
REF = ["gemma4-eval-g31-v12e-l4seq55", "gemma4-eval-g31-v12e-l4seq55b"]  # two parts (kernel deadline, K-note Nov 6)
SETUPS = [  # (label, run folders, wall cap in minutes)
    ("Kaggle 4x L4, 1 session (scorer)", REF, 7.0),
    ("H100 SXM, 5 sessions, converted", ["gemma4-eval-g31-v12e-h100-conv"], 4.8),
    ("RTX PRO 6000, 5 sessions, converted", ["gemma4-eval-g31-v12e-train55-ctl"], 6.0),
    ("L40S, 5 sessions, converted", ["gemma4-eval-g31-v12e-l40s-conv"], 11.7),
    ("Kaggle 4x L4, 10 sessions, converted", ["gemma4-eval-g31-v12e-l4ctl-conv15"], 15.0),
    ("Kaggle 4x L4, 10 sessions, nominal", ["gemma4-eval-g31-v12e-l4ctl-nom7"], 7.0),
    ("A100, 5 sessions (earlier reference)", ["gemma4-eval-g31-v12e-train55"], 7.0),
]


def rows(runs: list[str]) -> dict[str, dict]:
    out = {}
    for run in runs:
        for f in glob.glob(str(ROOT / RUNS / run / "run" / "**" / "rows.jsonl"), recursive=True):
            for line in open(f):
                if line.startswith("{"):
                    r = json.loads(line)
                    out[r["id"]] = r
    return out


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/paper/aggregates/control_43.csv"))
    a = ap.parse_args()
    speed = {r["run"]: r for r in csv.DictReader(open(ROOT / "docs/paper/aggregates/speed_table.csv"))}
    ref = {t: int(bool(r["resolved"])) for t, r in rows(REF).items()}
    out = []
    for label, runs, cap in SETUPS:
        rs = rows(runs)
        solved = {t: int(bool(r["resolved"])) for t, r in rs.items()}
        facs = [float(speed[r]["factor"]) * int(speed[r]["sessions"]) for r in runs if r in speed]
        n_s = sum(int(speed[r]["sessions"]) for r in runs if r in speed)
        f = sum(facs) / n_s if n_s else float("nan")
        common = sorted(set(ref) & set(solved))
        both = sum(ref[t] & solved[t] for t in common)
        ref_only = sum(ref[t] & (1 - solved[t]) for t in common)
        run_only = sum((1 - ref[t]) & solved[t] for t in common)
        out.append({"setup": label, "runs": "+".join(runs), "factor": round(f, 2), "wall_cap_min": cap,
                    "scorer_min": round(cap / f, 1), "tasks": len(solved), "solved": sum(solved.values()),
                    "timed_out": sum(1 for r in rs.values() if "exceeded session timeout" in (r.get("error") or "")),
                    "timed_out_solved": sum(1 for r in rs.values() if "exceeded session timeout" in (r.get("error") or "")
                                            and r["resolved"]),
                    "both": both, "scorer_only": ref_only, "run_only": run_only,
                    "mcnemar_p_vs_scorer": round(mcnemar(ref_only, run_only), 3) if runs != REF else ""})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    # the twin pair: same hardware (4x L4, 10 sessions), converted vs nominal cap
    conv, nom = (rows(runs) for label, runs, _ in SETUPS if "10 sessions" in label)
    common = sorted(set(conv) & set(nom))
    c_only = sum(bool(conv[t]["resolved"]) and not nom[t]["resolved"] for t in common)
    n_only = sum(bool(nom[t]["resolved"]) and not conv[t]["resolved"] for t in common)
    with open(Path(a.out).with_name("control_twin.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "tasks", "converted_only", "nominal_only", "mcnemar_p"])
        w.writerow(["4x L4 10 sessions: converted vs nominal", len(common), c_only, n_only,
                    round(mcnemar(c_only, n_only), 3)])
    for r in out:
        print(r)
    print("twin:", c_only, n_only, round(mcnemar(c_only, n_only), 3))


if __name__ == "__main__":
    main()
