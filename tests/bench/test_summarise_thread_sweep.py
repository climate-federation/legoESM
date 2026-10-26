"""The sweep summariser must not call a winner inside the repeat spread."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts/bench/_summarise_thread_sweep.py"
_spec = importlib.util.spec_from_file_location("ts", _SRC)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _arm(tmp_path, nd, cpt, rep, ms):
    (tmp_path / f"nd{nd}_cpt{cpt}_r{rep}.jsonl").write_text(
        json.dumps({"steady_median_ms": ms, "n_devices": nd}) + "\n")


def _census(tmp_path, mapping):
    for cpt, workers in mapping.items():
        (tmp_path / f"threads_cpt{cpt}.json.0").write_text(
            json.dumps({"threads": {"tf_XLAEigen": workers}}))


def _run(tmp_path, capsys):
    mod.main.__globals__["sys"].argv = ["x", str(tmp_path)]
    assert mod.main() == 0
    return capsys.readouterr().out


def test_a_gain_inside_the_repeat_spread_is_refuted(tmp_path, capsys):
    # 8 cores looks 5 ms faster, but its own repeats differ by 40 ms.
    _arm(tmp_path, 128, 64, 1, 1000.0); _arm(tmp_path, 128, 64, 2, 1002.0)
    _arm(tmp_path, 128, 8, 1, 995.0);   _arm(tmp_path, 128, 8, 2, 1035.0)
    _census(tmp_path, {64: 64, 8: 8})
    out = _run(tmp_path, capsys)
    assert "REFUTED" in out, out


def test_a_real_gain_with_a_moving_optimum_is_confirmed(tmp_path, capsys):
    _arm(tmp_path, 128, 64, 1, 1240.0); _arm(tmp_path, 128, 64, 2, 1245.0)
    _arm(tmp_path, 128, 8, 1, 700.0);   _arm(tmp_path, 128, 8, 2, 705.0)
    # 16 ranks have a bigger local problem and prefer the bigger share.
    _arm(tmp_path, 16, 64, 1, 919.0);   _arm(tmp_path, 16, 64, 2, 921.0)
    _arm(tmp_path, 16, 8, 1, 1500.0);   _arm(tmp_path, 16, 8, 2, 1510.0)
    _census(tmp_path, {64: 64, 8: 8})
    out = _run(tmp_path, capsys)
    assert "CONFIRMED" in out and "optimum moves" in out, out


def test_a_real_gain_without_a_moving_optimum_is_not_confirmed(tmp_path, capsys):
    # Faster at 128, but 16 ranks prefer the SAME small share: the speed-up
    # is real and the stated mechanism is not established.
    _arm(tmp_path, 128, 64, 1, 1240.0); _arm(tmp_path, 128, 64, 2, 1245.0)
    _arm(tmp_path, 128, 8, 1, 700.0);   _arm(tmp_path, 128, 8, 2, 705.0)
    _arm(tmp_path, 16, 64, 1, 1900.0);  _arm(tmp_path, 16, 64, 2, 1910.0)
    _arm(tmp_path, 16, 8, 1, 900.0);    _arm(tmp_path, 16, 8, 2, 905.0)
    _census(tmp_path, {64: 64, 8: 8})
    out = _run(tmp_path, capsys)
    assert "MECHANISM NOT CONFIRMED" in out, out


def test_a_dead_knob_is_invalid_not_refuted(tmp_path, capsys):
    # Every share ran 64 workers: the sweep varied pinning, not threads.
    # The old code called this REFUTED, i.e. "the thread pool is not the
    # cause" -- a conclusion the data cannot support.
    _arm(tmp_path, 128, 64, 1, 1240.0); _arm(tmp_path, 128, 64, 2, 1245.0)
    _arm(tmp_path, 128, 8, 1, 700.0);   _arm(tmp_path, 128, 8, 2, 705.0)
    _census(tmp_path, {64: 64, 8: 64})
    mod.main.__globals__["sys"].argv = ["x", str(tmp_path)]
    assert mod.main() == 2
    assert "INVALID" in capsys.readouterr().out


def test_a_live_knob_still_reaches_a_verdict(tmp_path, capsys):
    _arm(tmp_path, 128, 64, 1, 1240.0); _arm(tmp_path, 128, 64, 2, 1245.0)
    _arm(tmp_path, 128, 8, 1, 700.0);   _arm(tmp_path, 128, 8, 2, 705.0)
    _arm(tmp_path, 16, 64, 1, 919.0);   _arm(tmp_path, 16, 64, 2, 921.0)
    _arm(tmp_path, 16, 8, 1, 1500.0);   _arm(tmp_path, 16, 8, 2, 1510.0)
    _census(tmp_path, {64: 64, 8: 8})
    mod.main.__globals__["sys"].argv = ["x", str(tmp_path)]
    assert mod.main() == 0
    assert "CONFIRMED" in capsys.readouterr().out
