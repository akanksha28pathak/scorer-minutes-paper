import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth import make_session, write_run  # noqa: E402

from scorer_minutes.calibrate import calibrate
from scorer_minutes.cli import main
from scorer_minutes.logs import load_sessions
from scorer_minutes.timing import timing_report


def _run(tmp_path, decode, n=8, **kw):
    return write_run(tmp_path, {f"t{i}": make_session(f"t{i}", decode_tok_s=decode, seed=i, **kw) for i in range(n)})


def test_load_skips_rows_and_groups_sessions(tmp_path):
    run = _run(tmp_path, 30)
    res = load_sessions([run])
    assert len(res.sessions) == 8
    assert res.skipped == 1  # rows.jsonl
    s = res.sessions[0]
    assert s.calls == sorted(s.calls, key=lambda c: c.t_start)
    # prefix caching: the second call prefills only what was added since the first
    assert s.calls[1].new_prompt_tokens == s.calls[1].prompt_tokens - (s.calls[0].prompt_tokens + s.calls[0].completion_tokens)


def test_slow_setup_factor_and_cap(tmp_path):
    # pure decode (no prefill, no overhead): tok/s equals the decode speed exactly
    run = _run(tmp_path, 15.0, prefill_tok_s=1e12, overhead_s=0.0)
    cal = calibrate(load_sessions([run]).sessions, scorer_tok_s=30.0, cap_scorer_min=7)
    assert cal.median_tok_s == pytest.approx(15.0, rel=1e-6)
    assert cal.factor == pytest.approx(2.0, rel=1e-6)
    assert cal.cap_wall_min == pytest.approx(14.0, rel=1e-6)
    assert cal.factor_ci95[0] <= cal.factor <= cal.factor_ci95[1]


def test_reference_run_sets_scorer_speed(tmp_path, capsys):
    ref = _run(tmp_path / "ref", 32.0, prefill_tok_s=1e12, overhead_s=0.0)
    run = _run(tmp_path / "run", 16.0, prefill_tok_s=1e12, overhead_s=0.0)
    main(["calibrate", str(run), "--reference", str(ref), "--cap-scorer-min", "7"])
    out = json.loads(capsys.readouterr().out)
    assert out["calibration"]["factor"] == pytest.approx(2.0, rel=1e-3)
    assert out["reference"]["sessions"] == 8


def test_timing_fit_recovers_known_costs(tmp_path):
    run = _run(tmp_path, 30.0, n=20, prefill_tok_s=2000.0, overhead_s=0.3, ctx_slowdown=0.5)
    rep = timing_report(load_sessions([run]).sessions)
    f = rep.fit
    assert f.terms == ["overhead", "new_prompt_k", "gen_k", "gen_x_ctx_k"]
    assert f.r2 > 0.999
    assert f.coef["overhead"] == pytest.approx(0.3, abs=1e-6)
    assert f.prefill_tok_s == pytest.approx(2000.0, rel=1e-6)
    assert f.decode_tok_s(0) == pytest.approx(30.0, rel=1e-6)
    assert f.decode_tok_s(16) < f.decode_tok_s(0)
    assert rep.drift["late_over_early"] < 1  # long context slows late calls
    assert sum(rep.shares.values()) == pytest.approx(1.0, rel=1e-6)


def test_timing_flags_tool_heavy_sessions(tmp_path):
    sessions = {"fast": make_session("fast", decode_tok_s=30, tool_s=0.1, seed=1),
                "toolheavy": make_session("toolheavy", decode_tok_s=30, tool_s=30.0, seed=2)}
    run = write_run(tmp_path, sessions)
    rep = timing_report(load_sessions([run]).sessions)
    flagged = {f["session"].split("/")[0] for f in rep.flagged}
    assert flagged == {"toolheavy"}


def test_timing_without_prompt_tokens_falls_back(tmp_path):
    run = _run(tmp_path, 25.0, n=10, with_prompt=False)
    rep = timing_report(load_sessions([run]).sessions)
    assert rep.fit.terms == ["overhead", "gen_k"]
    assert rep.context[0]["calls"] == 0  # no context bins without prompt tokens


def test_custom_fields_and_iso_times(tmp_path):
    f = tmp_path / "calls.jsonl"
    recs = [{"sid": "a", "start": "2026-11-05T10:00:00Z", "end": "2026-11-05T10:00:10Z", "out_tok": 300},
            {"sid": "a", "start": "2026-11-05T10:00:12Z", "end": "2026-11-05T10:00:22Z", "out_tok": 300}]
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    from scorer_minutes.logs import Fields
    res = load_sessions([f], Fields(session=("sid",), t_start="start", t_end="end", completion_tokens="out_tok"))
    s = res.sessions[0]
    assert s.tok_s == pytest.approx(30.0)
    assert s.wall_seconds == pytest.approx(22.0)
    assert s.model_seconds == pytest.approx(20.0)


def test_overlapping_calls_not_double_counted(tmp_path):
    f = tmp_path / "calls.jsonl"
    recs = [{"task": "a", "t_start": 0, "t_end": 10, "response": {"usage": {"completion_tokens": 100}}},
            {"task": "a", "t_start": 5, "t_end": 12, "response": {"usage": {"completion_tokens": 100}}}]
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = load_sessions([f]).sessions[0]
    assert s.model_seconds == pytest.approx(12.0)
