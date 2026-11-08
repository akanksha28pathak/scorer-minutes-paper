"""scorer-minutes: hardware-normalized, noise-aware evaluation of coding agents under wall-clock budgets.

  scorer-minutes calibrate LOGS... [--scorer-tok-s 31.6 | --reference REF_LOGS...] [--cap-scorer-min 7] [--csv out.csv]
  scorer-minutes timing LOGS... [--csv out.csv]
  scorer-minutes noise RESULTS.csv... [--compare A B --split S] [--sd X] [--markdown out.md]
  scorer-minutes draws --sd 1.6 [--score 9]

LOGS are JSONL files or folders of them (one record per model call). Field names are configurable (--field-*).
All output is JSON on stdout (plus optional CSV / Markdown files).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

from . import __version__
from .calibrate import calibrate, reference_tok_s, session_rows
from .logs import Fields, load_sessions
from .noise import compare, draw_table, pooled, read_csv, report_markdown, single_draw_interval, summarize
from .timing import timing_report


def _log_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("logs", nargs="+", help="JSONL call logs or folders of them")
    p.add_argument("--glob", default="**/*.jsonl", help="file pattern inside folders (default **/*.jsonl)")
    d = Fields()
    p.add_argument("--field-session", default=",".join(d.session), help="comma-separated keys naming a session")
    p.add_argument("--field-t-start", default=d.t_start)
    p.add_argument("--field-t-end", default=d.t_end)
    p.add_argument("--field-prompt-tokens", default=d.prompt_tokens)
    p.add_argument("--field-completion-tokens", default=d.completion_tokens)
    p.add_argument("--field-cached-tokens", default=d.cached_tokens)
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="server ran without prefix caching: every call prefills its whole prompt")


def _fields(a) -> Fields:
    return Fields(tuple(a.field_session.split(",")), a.field_t_start, a.field_t_end, a.field_prompt_tokens,
                  a.field_completion_tokens, a.field_cached_tokens)


def _load(a, paths=None):
    res = load_sessions(paths or a.logs, _fields(a), a.glob, prefix_cache=not a.no_prefix_cache)
    if not res.sessions:
        sys.exit(f"no usable model-call records in {paths or a.logs} ({res.files} files, {res.skipped} skipped)")
    return res


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_calibrate(a) -> dict:
    res = _load(a)
    scorer = a.scorer_tok_s
    ref = None
    if a.reference:
        ref_res = _load(a, a.reference)
        scorer = reference_tok_s(ref_res.sessions)
        ref = {"sessions": len(ref_res.sessions), "median_tok_s": round(scorer, 3)}
    cal = calibrate(res.sessions, scorer, a.cap_scorer_min, seed=a.seed)
    rows = session_rows(res.sessions, cal.factor, a.cap_scorer_min)
    if a.csv:
        _write_csv(a.csv, rows)
    out = {"calibration": cal.as_dict(), "reference": ref,
           "input": {"files": res.files, "records": res.records, "skipped": res.skipped}}
    if a.cap_scorer_min is not None:
        out["sessions_at_cap"] = sum(bool(r["at_cap"]) for r in rows)
    return out


def cmd_timing(a) -> dict:
    res = _load(a)
    rep = timing_report(res.sessions, a.min_model_share, a.max_prefill_share)
    if a.csv:
        _write_csv(a.csv, [{"session": f["session"], "reasons": "; ".join(f["reasons"])} for f in rep.flagged])
    out = rep.as_dict()
    out["input"] = {"files": res.files, "records": res.records, "skipped": res.skipped,
                    "calls_without_prompt_tokens": res.missing_prompt_tokens}
    return out


def cmd_noise(a) -> dict:
    t = read_csv(a.results)
    if not t.runs:
        sys.exit("no runs in the input")
    out = {"configs": [dataclasses.asdict(s) for s in summarize(t)],
           "pooled": [dataclasses.asdict(pooled(t, s)) for s in t.splits()]}
    if a.compare:
        if not a.split:
            sys.exit("--compare needs --split")
        out["comparison"] = dataclasses.asdict(compare(t, a.compare[0], a.compare[1], a.split, a.sd, seed=a.seed))
    if a.markdown:
        Path(a.markdown).write_text(report_markdown(t), encoding="utf-8")
    return out


def cmd_draws(a) -> dict:
    out = {"sd": a.sd, "table": draw_table(a.sd, tuple(a.deltas))}
    if a.score is not None:
        lo, hi = single_draw_interval(a.score, a.sd)
        out["single_draw_95"] = [round(lo, 2), round(hi, 2)]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="scorer-minutes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calibrate", help="speed factor vs the scorer and the equivalent wall cap")
    _log_args(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scorer-tok-s", type=float, default=31.6,
                   help="scorer's median per-session tok/s (default 31.6: Gemma 4 31B, see README)")
    g.add_argument("--reference", nargs="+", help="logs of a run on the scoring setup; its median sets the speed")
    p.add_argument("--cap-scorer-min", type=float, help="session budget in scorer minutes")
    p.add_argument("--csv", help="write per-session rows here")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("timing", help="model vs tool time, prefill vs decode, drift over a session")
    _log_args(p)
    p.add_argument("--min-model-share", type=float, default=0.9)
    p.add_argument("--max-prefill-share", type=float, default=0.2)
    p.add_argument("--csv", help="write flagged sessions here")
    p.set_defaults(fn=cmd_timing)

    p = sub.add_parser("noise", help="run-to-run noise, pooled SD, comparisons")
    p.add_argument("results", nargs="+", help="CSV: config,split,run,task,solved or config,split,run,solved,n")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--split")
    p.add_argument("--sd", type=float, help="run SD to use instead of the pooled one")
    p.add_argument("--markdown", help="write a Markdown report here")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_noise)

    p = sub.add_parser("draws", help="how to read single leaderboard draws given a run SD")
    p.add_argument("--sd", type=float, required=True, help="SD of one draw, in tasks")
    p.add_argument("--deltas", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--score", type=float, help="one observed score (tasks): print its 95%% interval")
    p.set_defaults(fn=cmd_draws)

    a = ap.parse_args(argv)
    print(json.dumps(a.fn(a), indent=2, default=str))


if __name__ == "__main__":
    main()
