"""Direct tests for the vertical-diffusion microbenchmark: the gate's exit codes
and one tiny end-to-end run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_BENCH = (Path(__file__).resolve().parents[2]
          / "scripts" / "bench" / "bench_vertical_diffusion_micro.py")
_spec = importlib.util.spec_from_file_location("bench_vdiff_micro", _BENCH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _row(fwd=0.8, grad=0.9, vd=1e-16, gd=1e-15, path=True):
    return {"fwd_ratio": fwd, "grad_ratio": grad,
            "value_rel_diff": vd, "grad_rel_diff": gd, "in_sweep_path": path}


def test_gate_codes():
    assert mod.gate(_row(), None) == 0
    assert mod.gate(_row(), 1.0) == 0
    assert mod.gate(_row(grad=1.2), 1.0) == 1
    assert mod.gate(_row(fwd=1.2), 1.0) == 1
    assert mod.gate(_row(fwd=5.0), None) == 0
    assert mod.gate(_row(vd=1e-6), None) == 2
    assert mod.gate(_row(gd=1e-6, fwd=5.0), 1.0) == 2
    assert mod.gate(_row(vd=float("nan")), None) == 2
    assert mod.gate(_row(fwd=float("nan")), 1.0) == 2
    assert mod.gate(_row(path=False), 1.0) == 3


def test_tiny_run_writes_receipt(tmp_path):
    out = tmp_path / "r.json"
    rc = mod.main(["--cols", "64", "--nlev", "6", "--repeats", "2",
                   "--warmup", "1", "--out", str(out)])
    assert rc == 0
    row = json.loads(out.read_text())["rows"][0]
    assert row["cols"] == 64 and row["nlev"] == 6
    assert row["in_sweep_path"] is True
    assert row["value_rel_diff"] < 1e-12 and row["grad_rel_diff"] < 1e-12
    for k in ("prod_fwd_ms", "ref_fwd_ms", "prod_grad_ms", "ref_grad_ms"):
        assert row[k] > 0.0


def test_slow_run_exits_1_and_still_writes_receipt(tmp_path):
    out = tmp_path / "r.json"
    rc = mod.main(["--cols", "64", "--nlev", "6", "--repeats", "2",
                   "--warmup", "1", "--max-ratio", "1e-6", "--out", str(out)])
    assert rc == 1
    assert json.loads(out.read_text())["rows"][0]["exit"] == 1


def test_zero_repeats_refused():
    with pytest.raises(SystemExit):
        mod.main(["--repeats", "0"])


@pytest.mark.parametrize("bad", ["nan", "inf", "0", "-1"])
def test_bad_max_ratio_refused(bad):
    with pytest.raises(SystemExit):
        mod.main(["--max-ratio", bad])
