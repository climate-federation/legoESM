"""Direct tests for the fail-closed SWG online GPU smoke entry point."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate"
          / "ocean_fidelity" / "testcases"
          / "nemo_testcase_l1_swg_online_gpu_smoke.py")
SPEC = importlib.util.spec_from_file_location("swg_online_gpu_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)


def test_cpu_gpu_comparator_accepts_roundoff_and_rejects_live_plant():
    reference = tuple(tuple(np.zeros((3, 3), dtype=np.float64)
                            for _ in range(3)) for _ in range(2))
    within = tuple(tuple(value + 1.0e-15 for value in operator)
                   for operator in reference)
    rows = smoke._compare_samples(reference, within)
    assert len(rows) == 6
    planted = tuple(
        tuple(value.copy() for value in operator) for operator in reference)
    planted[1][1][1, 1] = 1.0e-10
    with pytest.raises(AssertionError, match="filtered u_m_s"):
        smoke._compare_samples(reference, planted)
    nonfinite = tuple(
        tuple(value.copy() for value in operator) for operator in reference)
    nonfinite[0][0][0, 0] = np.nan
    matching_nonfinite = tuple(
        tuple(value.copy() for value in operator) for operator in nonfinite)
    with pytest.raises(AssertionError, match="box ssh_m.*non-finite"):
        smoke._compare_samples(nonfinite, matching_nonfinite)


def test_smoke_requires_explicit_memory_fraction_before_gpu_import(
        tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    result = smoke.main([
        "0", "--restart", str(tmp_path / "restart.npz"),
        "--output", str(tmp_path / "smoke.json"),
    ])
    assert result == 2
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION must be set" in capsys.readouterr().err
