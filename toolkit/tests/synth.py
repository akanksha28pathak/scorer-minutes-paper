"""Synthetic call logs with known speeds (no real traces are needed or used in tests)."""
from __future__ import annotations

import json
import random
from pathlib import Path


def make_session(task: str, *, decode_tok_s: float, prefill_tok_s: float = 4000.0, overhead_s: float = 0.2,
                 ctx_slowdown: float = 0.0, calls: int = 20, tool_s: float = 1.0, t0: float = 1_000_000.0,
                 attempt: int = 0, seed: int = 0, with_prompt: bool = True, noise: float = 0.0) -> list[dict]:
    """One agent session. Decode cost per 1k tokens = 1000/decode_tok_s + ctx_slowdown * context_k."""
    rng = random.Random(seed)
    t, ctx = t0, 3000
    recs = []
    for i in range(calls):
        new = rng.randint(200, 1500) if i else ctx
        gen = rng.randint(100, 900)
        prompt = ctx + (new if i else 0)
        dec = gen / 1000 * (1000 / decode_tok_s + ctx_slowdown * prompt / 1000)
        dur = overhead_s + new / prefill_tok_s + dec
        dur *= 1 + rng.uniform(-noise, noise)
        usage = {"completion_tokens": gen}
        if with_prompt:
            usage["prompt_tokens"] = prompt
        recs.append({"task": task, "attempt": attempt, "call": i, "t_start": t, "t_end": t + dur,
                     "response": {"usage": usage}})
        t += dur + tool_s
        ctx = prompt + gen
    return recs


def write_run(dir_: Path, sessions: dict[str, list[dict]], extra_rows: bool = True) -> Path:
    d = dir_ / "run" / "shard0" / "llm_calls"
    d.mkdir(parents=True, exist_ok=True)
    for task, recs in sessions.items():
        (d / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    if extra_rows:  # graded rows without timestamps live in the same run folder; they must be skipped
        (dir_ / "rows.jsonl").write_text(json.dumps({"task": "x", "resolved": True}) + "\n")
    return dir_
