"""The one-node refutation test: a penalty TREND, not a winner.

Picking the fastest share per mesh turns a one-millisecond difference into a
verdict.  The registered statistic is the penalty for keeping the production
share, and how it moves as the local problem shrinks, compared against the
uncertainty carried from the repeats.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts/bench/_summarise_share_vs_size.py"
_spec = importlib.util.spec_from_file_location("svs", _SRC)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _arm(p, sub, cpt, rep, ms):
    (p / f"sub{sub}_cpt{cpt}_r{rep}.jsonl").write_text(
        json.dumps({"steady_median_ms": ms}) + "\n")


def _census(p, mapping):
    for cpt, w in mapping.items():
        (p / f"threads_cpt{cpt}.json.0").write_text(
            json.dumps({"threads": {"tf_XLAEigen": w}}))


def _run(p, capsys, expect=0):
    mod.main.__globals__["sys"].argv = ["x", str(p)]
    assert mod.main() == expect
    return capsys.readouterr().out


def test_a_rising_penalty_is_supported(tmp_path, capsys):
    # Big mesh: 64 cores is best (penalty 1.0). Small mesh: 64 costs 2.5x.
    for rep in (1, 2):
        _arm(tmp_path, 8, 64, rep, 300.0); _arm(tmp_path, 8, 8, rep, 900.0)
        _arm(tmp_path, 5, 64, rep, 25.0);  _arm(tmp_path, 5, 8, rep, 10.0)
    _census(tmp_path, {8: 8, 64: 64})
    out = _run(tmp_path, capsys)
    assert "SUPPORTED" in out, out


def test_a_flat_penalty_truncates_rather_than_cancels(tmp_path, capsys):
    # 64 cores is best at every size: the LOCAL-SIZE mechanism is dead, but
    # the multi-node sweep is truncated, not cancelled -- one node cannot
    # see the inter-node mechanisms.
    for rep in (1, 2):
        _arm(tmp_path, 8, 64, rep, 300.0); _arm(tmp_path, 8, 8, rep, 900.0)
        _arm(tmp_path, 5, 64, rep, 10.0);  _arm(tmp_path, 5, 8, rep, 30.0)
    _census(tmp_path, {8: 8, 64: 64})
    out = _run(tmp_path, capsys)
    assert "REFUTED" in out and "TRUNCATE" in out, out
    assert "cancel" not in out.lower().replace("cancelling", ""), out


def test_a_difference_inside_the_spread_is_inconclusive(tmp_path, capsys):
    # The penalty barely moves and the repeats are noisy: neither branch.
    _arm(tmp_path, 8, 64, 1, 300.0); _arm(tmp_path, 8, 64, 2, 340.0)
    _arm(tmp_path, 8, 8, 1, 298.0);  _arm(tmp_path, 8, 8, 2, 345.0)
    _arm(tmp_path, 5, 64, 1, 100.0); _arm(tmp_path, 5, 64, 2, 115.0)
    _arm(tmp_path, 5, 8, 1, 99.0);   _arm(tmp_path, 5, 8, 2, 118.0)
    _census(tmp_path, {8: 8, 64: 64})
    out = _run(tmp_path, capsys)
    assert "INCONCLUSIVE" in out, out


def test_a_dead_knob_is_invalid(tmp_path, capsys):
    for rep in (1, 2):
        _arm(tmp_path, 8, 64, rep, 300.0); _arm(tmp_path, 8, 8, rep, 900.0)
        _arm(tmp_path, 5, 64, rep, 25.0);  _arm(tmp_path, 5, 8, rep, 10.0)
    _census(tmp_path, {8: 64, 64: 64})
    assert "INVALID" in _run(tmp_path, capsys, expect=2)


def test_the_production_share_winning_everywhere_is_not_inconclusive(tmp_path, capsys):
    """The real 2026-09-23 data, which the first version mislabelled.

    64 cores was best at all four mesh sizes, so every penalty is 1.000 by
    construction.  Deriving an uncertainty from the repeat spread anyway put
    a tolerance band around a ratio that cannot vary, and the INCONCLUSIVE
    branch fired on the cleanest result the job can produce.
    """
    for rep, jitter in ((1, 0.0), (2, 3.0)):
        _arm(tmp_path, 5, 8, rep, 46.6 + jitter); _arm(tmp_path, 5, 16, rep, 28.5 + jitter)
        _arm(tmp_path, 5, 32, rep, 19.8 + jitter); _arm(tmp_path, 5, 64, rep, 17.0 + jitter)
        _arm(tmp_path, 8, 8, rep, 1874.2 + jitter); _arm(tmp_path, 8, 16, rep, 1126.0 + jitter)
        _arm(tmp_path, 8, 32, rep, 731.9 + jitter); _arm(tmp_path, 8, 64, rep, 475.8 + jitter)
    _census(tmp_path, {8: 8, 16: 16, 32: 32, 64: 64})
    out = _run(tmp_path, capsys)
    assert "REFUTED" in out and "INCONCLUSIVE" not in out, out
    assert "TRUNCATE" in out, out
