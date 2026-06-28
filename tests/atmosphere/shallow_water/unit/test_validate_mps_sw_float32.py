"""Tests for ``scripts/validate/validate_mps_sw_float32.py``.

Two layers:
* in-process — exercise the pure ``check_validation`` threshold logic with
  synthetic metric dicts (no JAX, precision-policy independent);
* subprocess — run the real float32 SW dycore end-to-end on CPU with x64 OFF
  and assert it PASSes (the validator forces float32 and asserts physical
  sanity).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate.validate_mps_sw_float32 import check_validation

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO_ROOT / "scripts" / "validate" / "validate_mps_sw_float32.py"


def _good_metrics(**over):
    m = {
        "backend": "cpu",
        "dtype": "float32",
        "n_steps": 50,
        "finite": True,
        "h_min": 1097.0,
        "h_max": 2994.0,
        "h_drift_rel": 6.0e-5,
        "mass_rel": 2.0e-5,
    }
    m.update(over)
    return m


def test_check_validation_passes_on_healthy_metrics():
    assert check_validation(_good_metrics()) == []


@pytest.mark.parametrize(
    "override, needle",
    [
        ({"finite": False}, "NaN/Inf"),
        ({"h_drift_rel": 1.0}, "drift"),
        ({"mass_rel": 1.0}, "mass drift"),
        ({"dtype": "float64"}, "dtype"),
    ],
)
def test_check_validation_flags_each_failure(override, needle):
    failures = check_validation(_good_metrics(**override))
    assert any(needle in f for f in failures), failures


def test_check_validation_respects_custom_tolerances():
    # A drift that passes the default tol must fail a tighter one.
    m = _good_metrics(h_drift_rel=1.0e-4)
    assert check_validation(m) == []
    assert check_validation(m, drift_tol=5.0e-5) != []


def _run_validator(x64: str):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = x64
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--steps", "20"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_validator_passes_float32_on_cpu():
    """End-to-end: the real dycore runs float32 on CPU (x64 off) and PASSes."""
    proc = _run_validator("0")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "PASS" in proc.stdout
    assert "dtype=float32" in proc.stdout


@pytest.mark.slow
def test_validator_rejects_x64_on_cleanly():
    """Under x64 the validator must refuse with clear guidance, not a traceback.

    x64 promotes the dycore to float64 via the conservation accumulator + mass
    fixer (mixing dtypes in the scan carry diverges to NaN), so single-precision
    validation is invalid there — the validator detects this precondition and
    exits non-zero with a ``JAX_ENABLE_X64=0`` hint rather than reporting a
    bogus PASS or crashing.
    """
    proc = _run_validator("1")
    assert proc.returncode != 0
    assert "JAX_ENABLE_X64=0" in proc.stdout
    assert "Traceback" not in proc.stderr  # clean message, not an uncaught raise
