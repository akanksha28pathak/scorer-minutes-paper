import json
import math

import pytest

from scorer_minutes.cli import main
from scorer_minutes.noise import compare, from_rows, pooled, read_csv, report_markdown, summarize
from scorer_minutes.stats import p_wrong_order, runs_needed


def _long(config, split, run, solved_tasks, n=10):
    return [{"config": config, "split": split, "run": run, "task": f"t{i}", "solved": int(i in solved_tasks)}
            for i in range(n)]


def test_summary_and_pooled_sd():
    rows = (_long("A", "s", "a1", {0, 1, 2, 3}) + _long("A", "s", "a2", {0, 1, 2, 3, 4, 5})
            + _long("B", "s", "b1", {0, 1}) + _long("B", "s", "b2", {0, 1}) + _long("C", "s", "c1", {0}))
    t = from_rows(rows)
    by = {s.config: s for s in summarize(t)}
    assert by["A"].solved == [4, 6] and by["A"].mean == 5 and by["A"].sd == pytest.approx(math.sqrt(2))
    assert by["A"].flaky_tasks == 2 and by["B"].flaky_tasks == 0
    p = pooled(t, "s")
    assert p.configs == 2 and p.dof == 2  # C has one run: excluded
    assert p.sd == pytest.approx(1.0)  # (2 + 0) / 2
    # task-level SD for A: two tasks with p = 0.5, k = 2 -> var 2 * 0.25 * 2 = 1
    assert by["A"].task_sd == pytest.approx(1.0)


def test_summary_csv_shape(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("config,split,run,solved,n\nv12,train55,p1,20,55\nv12,train55,p2,19,55\nv9d,train55,a,19,55\n"
                 "v9d,train55,b,19,55\n")
    t = read_csv([f])
    p = pooled(t, "train55")
    assert p.n == 55 and p.dof == 2 and p.sd == pytest.approx(0.5)
    assert summarize(t)[0].task_sd is None


def test_compare_with_task_bootstrap():
    rows = []
    for r in range(3):
        rows += _long("A", "s", f"a{r}", set(range(6)) | ({6} if r == 0 else set()), n=20)
        rows += _long("B", "s", f"b{r}", set(range(3)), n=20)
    c = compare(from_rows(rows), "A", "B", "s", n_boot=2000)
    assert c.mean_diff == pytest.approx(6 + 1 / 3 - 3)
    lo, hi = c.task_bootstrap_ci95
    assert lo < c.mean_diff < hi and lo > 0
    assert c.p_value < 0.05


def test_power_and_wrong_order():
    assert p_wrong_order(0, 2.0) == pytest.approx(0.5)
    assert p_wrong_order(2, 2.0, runs=5) < p_wrong_order(2, 2.0, runs=1)
    # SD 2, margin 2: n = 2 * (1.96 + 0.84)^2 * 1 = 15.7 -> 16
    assert runs_needed(2.0, 2.0) == 16


def test_cli_noise_and_draws(tmp_path, capsys):
    f = tmp_path / "long.csv"
    rows = _long("A", "s", "a1", {0, 1, 2}) + _long("A", "s", "a2", {0, 1})
    f.write_text("config,split,run,task,solved\n" + "".join(
        f"{r['config']},{r['split']},{r['run']},{r['task']},{r['solved']}\n" for r in rows))
    md = tmp_path / "noise.md"
    main(["noise", str(f), "--markdown", str(md)])
    out = json.loads(capsys.readouterr().out)
    assert out["pooled"][0]["dof"] == 1
    assert "Pooled run noise" in md.read_text()
    main(["draws", "--sd", "1.6", "--score", "9"])
    d = json.loads(capsys.readouterr().out)
    assert d["single_draw_95"] == [pytest.approx(9 - 1.96 * 1.6, abs=0.01), pytest.approx(9 + 1.96 * 1.6, abs=0.01)]
    assert d["table"][0]["delta"] == 1


def test_markdown_has_reading_table():
    rows = _long("A", "s", "a1", {0, 1, 2}) + _long("A", "s", "a2", {0})
    md = report_markdown(from_rows(rows))
    assert "reading comparisons" in md and "Runs each for 80% power" in md


def test_chi2_and_sd_ci():
    from scorer_minutes.stats import chi2_quantile, sd_ci
    assert abs(chi2_quantile(0.975, 6) - 14.449) < 0.01   # table values
    assert abs(chi2_quantile(0.025, 6) - 1.237) < 0.01
    lo, hi = sd_ci(1.19, 6)
    assert lo < 1.19 < hi and abs(lo - 0.767) < 0.01 and abs(hi - 2.62) < 0.02


def test_pooled_over_splits():
    from scorer_minutes.noise import Run, Table, pooled
    t = Table([Run("a", "s1", "1", 10, 55), Run("a", "s1", "2", 12, 55),
               Run("b", "s2", "1", 20, 55), Run("b", "s2", "2", 21, 55)])
    p = pooled(t, ["s1", "s2"])
    assert p.dof == 2 and p.split == "s1+s2" and abs(p.sd - (2.5 / 2) ** 0.5) < 1e-9
