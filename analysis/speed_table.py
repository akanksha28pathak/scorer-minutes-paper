"""Speed-factor table for the scorer-minutes paper (idea 1): one row per logged run.

  python3 analysis/speed_table.py RUN_DIR... [--scorer-tok-s 31.6 | --reference REF_RUN_DIR] [--cap-scorer-min 7]
                                  [--label RUN_DIR=label ...] [--out results/paper/speed_table.csv]

RUN_DIR is a run folder with model-call logs at run/**/llm_calls/*.jsonl (scripts/llm_call_log.py, --log-llm-calls),
e.g. results/kaggle_runs/<run> or a Vast run copied back to ~/gemma4_work. Per run: sessions, median / IQR tok/s,
factor vs the scorer (with a bootstrap CI), the wall cap for --cap-scorer-min, median model share of wall time, and the
fitted prefill / decode split with decode speed at 0 and 24k context (tools/scorer_minutes timing). Hardware and
concurrency are not in the logs: give them with --label (e.g. "Kaggle 4xL4 TP4, 10 workers, 31B").
Aggregates only (no task ids or content). CPU only, stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "scorer_minutes" / "src"))
from scorer_minutes.calibrate import calibrate, reference_tok_s  # noqa: E402
from scorer_minutes.logs import load_sessions  # noqa: E402
from scorer_minutes.timing import timing_report  # noqa: E402

GLOB = "run/**/llm_calls/*.jsonl"


def row(run: Path, label: str, scorer: float, cap: float | None) -> dict:
    res = load_sessions([run], glob=GLOB)
    if not res.sessions:
        return None
    cal = calibrate(res.sessions, scorer, cap)
    rep = timing_report(res.sessions)
    fit = rep.fit
    r = {"run": run.name, "label": label, "sessions": cal.sessions, "median_tok_s": round(cal.median_tok_s, 2),
         "p25_tok_s": round(cal.p25_tok_s, 2), "p75_tok_s": round(cal.p75_tok_s, 2), "factor": round(cal.factor, 3),
         "factor_lo": round(cal.factor_ci95[0], 3), "factor_hi": round(cal.factor_ci95[1], 3),
         "cap_wall_min": None if cal.cap_wall_min is None else round(cal.cap_wall_min, 2),
         "model_share": rep.model_share_median, "fit_r2": None if not fit else round(fit.r2, 3),
         "prefill_share": rep.shares.get("prefill"), "decode_share": rep.shares.get("decode"),
         "overhead_share": rep.shares.get("overhead"),
         "decode_tok_s_ctx0": None if not fit or fit.decode_tok_s(0) is None else round(fit.decode_tok_s(0), 2),
         "decode_tok_s_ctx24k": None if not fit or fit.decode_tok_s(24) is None else round(fit.decode_tok_s(24), 2),
         "late_over_early": rep.drift.get("late_over_early"),
         "calls_without_prompt_tokens": res.missing_prompt_tokens}
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--scorer-tok-s", type=float, default=31.6)
    g.add_argument("--reference", help="run folder measured on the scoring setup (sets the scorer speed)")
    ap.add_argument("--cap-scorer-min", type=float, default=7.0)
    ap.add_argument("--label", nargs="*", default=[], help="RUN_DIR=label pairs (hardware, workers, model)")
    ap.add_argument("--labels-file", default=str(ROOT / "docs" / "paper" / "aggregates" / "run_labels.tsv"),
                    help="TSV: run folder name <TAB> label (hardware, workers, model); --label wins")
    ap.add_argument("--out", default=str(ROOT / "results" / "paper" / "speed_table.csv"))
    a = ap.parse_args()
    labels = dict(x.split("=", 1) for x in a.label)
    if a.labels_file and Path(a.labels_file).exists():
        for ln in Path(a.labels_file).read_text(encoding="utf-8").splitlines():
            if ln.strip() and not ln.startswith("#") and "\t" in ln:
                k, v = ln.split("\t", 1)
                labels.setdefault(k.strip(), v.strip())
    scorer = a.scorer_tok_s
    if a.reference:
        scorer = reference_tok_s(load_sessions([a.reference], glob=GLOB).sessions)
        print(f"scorer speed from {a.reference}: {scorer:.2f} tok/s", file=sys.stderr)
    rows = []
    for r in a.runs:
        p = Path(r)
        out = row(p, labels.get(r, labels.get(p.name, "")), scorer, a.cap_scorer_min)
        if out is None:
            print(f"skip {r}: no model-call logs under {GLOB}", file=sys.stderr)
        else:
            rows.append(out)
    if not rows:
        sys.exit("no run with model-call logs")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["run", "label", "sessions", "median_tok_s", "factor", "cap_wall_min", "model_share", "prefill_share",
            "decode_tok_s_ctx0", "decode_tok_s_ctx24k", "late_over_early"]
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join("-" if r[c] is None else str(r[c]) for c in cols) + " |")
    print(f"\nscorer speed {scorer:.2f} tok/s; CSV: {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
