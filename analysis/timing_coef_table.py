"""Fitted per-call timing coefficients for selected runs (scorer-minutes paper, section 4.4).

  python3 analysis/timing_coef_table.py [--out docs/paper/aggregates/timing_coefs.csv]

Reads docs/paper/aggregates/timing/<run>.json (written by `scorer-minutes timing`). The fit is
t = c + a * new_prompt + b * gen + d * gen * context (seconds; tokens in thousands): c per-call overhead, a seconds per
1k new prompt tokens (prefill), b seconds per 1k generated tokens at empty context (decode), d the extra decode seconds
per 1k generated tokens per 1k tokens of context. Runs of one setup are listed separately and averaged in the table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMING = ROOT / "docs/paper/aggregates/timing"
SETUPS = [  # (label, runs)
    ("Kaggle 4x L4, 1 session (scorer)", ["gemma4-eval-g31-v12e-l4seq55", "gemma4-eval-g31-v12e-l4seq55b"]),
    ("Kaggle 4x L4, 10 sessions", ["gemma4-eval-g31-v12e-l4ctl-conv15", "gemma4-eval-g31-v12e-l4ctl-nom7"]),
    ("H100 SXM, 5 sessions", ["gemma4-eval-g31-v12e-h100-conv"]),
    ("L40S, 5 sessions", ["gemma4-eval-g31-v12e-l40s-conv"]),
    ("RTX PRO 6000, 5 sessions (v15, held-out)", ["gemma4-eval-g31-v15-val55-p1", "gemma4-eval-g31-v15-val55-p2"]),
    ("RTX PRO 6000, 5 sessions, + adapter B (held-out)",
     ["gemma4-eval-g31-v15-sft-val55-p1", "gemma4-eval-g31-v15-sft-val55-p2"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/paper/aggregates/timing_coefs.csv"))
    a = ap.parse_args()
    out = []
    for label, runs in SETUPS:
        fits = [json.load(open(TIMING / f"{r}.json"))["fit"] for r in runs]
        mean = lambda key: sum(f["coef"][key] for f in fits) / len(fits)
        out.append({"setup": label, "runs": "+".join(runs), "calls": sum(f["calls"] for f in fits),
                    "r2_min": min(f["r2"] for f in fits), "c_overhead_s": round(mean("overhead"), 2),
                    "a_s_per_k_prompt": round(mean("new_prompt_k"), 2), "b_s_per_k_gen": round(mean("gen_k"), 1),
                    "d_s_per_k_gen_per_k_ctx": round(mean("gen_x_ctx_k"), 2),
                    "decode_tok_s_0k": round(sum(f["decode_tok_s_at_context_k"]["0"] for f in fits) / len(fits), 1),
                    "decode_tok_s_24k": round(sum(f["decode_tok_s_at_context_k"]["24"] for f in fits) / len(fits), 1)})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("| Setup | Calls (R^2 min) | c overhead, s | a prefill, s / 1k tok | b decode, s / 1k tok | d, s / 1k tok / 1k ctx | Decode tok/s at 0k / 24k |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['setup']} | {r['calls']:,} ({r['r2_min']}) | {r['c_overhead_s']} | {r['a_s_per_k_prompt']} | "
              f"{r['b_s_per_k_gen']} | {r['d_s_per_k_gen_per_k_ctx']} | {r['decode_tok_s_0k']} / {r['decode_tok_s_24k']} |")


if __name__ == "__main__":
    main()
