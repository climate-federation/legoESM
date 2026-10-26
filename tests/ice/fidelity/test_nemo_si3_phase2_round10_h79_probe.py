"""Behavioural tests for the Round-10 H79 scalar-libm discriminator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "scripts"
    / "validate"
    / "ocean_fidelity"
    / "testcases"
    / "nemo_si3_phase2_round10_h79_probe.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("nemo_si3_phase2_round10_h79_probe", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_libm_arm_is_exact_and_vector_arm_binds() -> None:
    probe = _load_probe()
    for oracle_root in (probe.ROOT, probe.PROBE_ROOT):
        if not oracle_root.is_dir():
            pytest.skip(f"oracle run root absent: {oracle_root}")
    report, code = probe.run_probe()
    assert code == 0
    assert report["status"] == "BIT-EXACT"
    assert report["predicates"] == {
        "vector_arm_binds": True,
        "scalar_arm_is_bit_exact": True,
    }
    assert report["vector_np_exp_arm"]["bitwise_nonzero_over_n"] == "94986 / 1008016"
    assert report["vector_np_exp_arm"]["max_abs"] == 0.00390625
    assert report["scalar_libm_exp_arm"]["bitwise_nonzero_over_n"] == "0 / 1008016"
