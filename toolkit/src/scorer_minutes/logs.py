"""Read per-call model logs (one JSON object per model call, JSONL files) into sessions.

A record needs a start and end timestamp and the completion token count; everything else is optional. Field names are
dotted paths into the record, so OpenAI / LiteLLM / vLLM style logs work without conversion. Defaults match
`scripts/llm_call_log.py` in this repo:

    {"task", "attempt", "call", "t_start", "t_end", "request": {...},
     "response": {"usage": {"prompt_tokens", "completion_tokens", "prompt_tokens_details": {"cached_tokens"}}}}

Records without the timestamps (for example graded rows in the same folder) are skipped and counted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Fields:
    session: Sequence[str] = ("task", "attempt")
    t_start: str = "t_start"
    t_end: str = "t_end"
    prompt_tokens: str = "response.usage.prompt_tokens"
    completion_tokens: str = "response.usage.completion_tokens"
    cached_tokens: str = "response.usage.prompt_tokens_details.cached_tokens"


@dataclass
class Call:
    t_start: float
    t_end: float
    completion_tokens: int
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    new_prompt_tokens: int | None = None  # filled by Session.finalize

    @property
    def seconds(self) -> float:
        return max(self.t_end - self.t_start, 0.0)


@dataclass
class Session:
    key: tuple
    calls: list[Call] = field(default_factory=list)

    def finalize(self, prefix_cache: bool = True) -> "Session":
        """Sort calls and estimate the prompt tokens each call had to prefill.

        With a reported cached-token count: prompt - cached. Otherwise, with prefix caching (vLLM's default), an agent
        transcript only grows, so a call prefills what was added since the previous call; if the prompt shrank
        (context trimmed, a sub-agent), the whole prompt counts. Without prefix caching: the whole prompt.
        """
        self.calls.sort(key=lambda c: c.t_start)
        prev = None
        for c in self.calls:
            if c.prompt_tokens is None:
                c.new_prompt_tokens = None
            elif c.cached_tokens is not None:
                c.new_prompt_tokens = max(c.prompt_tokens - c.cached_tokens, 0)
            elif prefix_cache and prev is not None and c.prompt_tokens >= prev:
                c.new_prompt_tokens = c.prompt_tokens - prev
            else:
                c.new_prompt_tokens = c.prompt_tokens
            if c.prompt_tokens is not None:
                prev = c.prompt_tokens + c.completion_tokens
        return self

    @property
    def gen_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def model_seconds(self) -> float:
        """Time inside model calls (overlapping calls, e.g. parallel sub-agents, are merged, not double counted)."""
        total, cur_s, cur_e = 0.0, None, None
        for c in sorted(self.calls, key=lambda c: c.t_start):
            if cur_e is None or c.t_start > cur_e:
                if cur_e is not None:
                    total += cur_e - cur_s
                cur_s, cur_e = c.t_start, c.t_end
            else:
                cur_e = max(cur_e, c.t_end)
        if cur_e is not None:
            total += cur_e - cur_s
        return total

    @property
    def wall_seconds(self) -> float:
        """First call start to last call end (time before the first and after the last call is not in the logs)."""
        if not self.calls:
            return 0.0
        return max(c.t_end for c in self.calls) - min(c.t_start for c in self.calls)

    @property
    def tok_s(self) -> float | None:
        t = sum(c.seconds for c in self.calls)
        return self.gen_tokens / t if t > 0 else None


def get_path(rec: Any, dotted: str) -> Any:
    cur = rec
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _time(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    return None


def _int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def iter_files(paths: Iterable[str | Path], glob: str = "**/*.jsonl") -> Iterator[Path]:
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(p.glob(glob))
        elif p.exists():
            yield p
        else:
            raise FileNotFoundError(p)


@dataclass
class LoadResult:
    sessions: list[Session]
    files: int
    records: int
    skipped: int
    missing_prompt_tokens: int


def load_sessions(paths: Iterable[str | Path], fields: Fields = Fields(), glob: str = "**/*.jsonl",
                  prefix_cache: bool = True) -> LoadResult:
    by_key: dict[tuple, Session] = {}
    files = records = skipped = no_prompt = 0
    for f in iter_files(paths, glob):
        files += 1
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                t0, t1 = _time(get_path(rec, fields.t_start)), _time(get_path(rec, fields.t_end))
                gen = _int(get_path(rec, fields.completion_tokens))
                if t0 is None or t1 is None or gen is None:
                    skipped += 1
                    continue
                key = tuple(get_path(rec, k) if get_path(rec, k) is not None else (f.stem if i == 0 else 0)
                            for i, k in enumerate(fields.session))
                prompt = _int(get_path(rec, fields.prompt_tokens))
                no_prompt += prompt is None
                by_key.setdefault(key, Session(key)).calls.append(
                    Call(t0, t1, gen, prompt, _int(get_path(rec, fields.cached_tokens))))
    sessions = [s.finalize(prefix_cache) for _, s in sorted(by_key.items(), key=lambda kv: tuple(map(str, kv[0])))]
    return LoadResult(sessions, files, records, skipped, no_prompt)
