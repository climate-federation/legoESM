"""Regression test for the iter-71 ``scripts/run_rce.py`` CLI input
validation (mirrors iter-67/70 plane-CRM driver validation).

iter-71 added guards rejecting bad numeric inputs with clean
SystemExit + exit 1, instead of:
* ``--days -1`` → silent "Complete: 0.0s wall time" zero-row run.
* ``--resolution 0`` → deep grid-creation crash.
* ``--sst-init nan`` → NaN propagation through all column ICs.

This test parametric-asserts each guard produces:
* non-zero exit code
* concise ``error: <flag> rejected: ...`` marker
* no raw Python traceback (catches a future regression that
  re-introduces the raw exception path)

Wall: ~10 s per case (lightweight hydrostatic driver, no JAX MPI
init). ~50 s total for 5 cases.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER = REPO_ROOT / "scripts" / "run_rce.py"


@pytest.mark.parametrize("flag,value,expected_err", [
    ("--days", "-1", "must be positive integer"),
    ("--days", "0", "must be positive integer"),
    ("--resolution", "0", "must be positive integer"),
    ("--nlev", "-1", "must be positive integer"),
    ("--diag-days", "-5", "must be positive integer"),
    ("--dt", "nan", "must be finite"),
    ("--dt", "0.0", "must be positive"),
    ("--dt", "-1.0", "must be positive"),
    ("--sst-init", "nan", "must be finite"),
    ("--truncation", "0", "must be positive integer"),
])
def test_run_rce_rejects_bad_cli_args(tmp_path, flag, value, expected_err):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--grid-type", "cubed_sphere",
        "--discretization", "cdgrid",
        "--resolution", "12",
        "--days", "1",
        "--nlev", "20",
        flag, value,
        "--output", str(tmp_path / f"rce_bad_{flag.strip('-')}"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, (
        f"run_rce.py exited 0 with {flag}={value}; expected non-zero.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    combined = result.stdout + result.stderr
    assert "rejected" in combined and expected_err in combined, (
        f"run_rce.py exit message missing 'rejected' + "
        f"'{expected_err}'.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    assert "Traceback" not in combined, (
        f"run_rce.py emitted Python traceback for {flag}={value}; "
        f"iter-71 contract: clean SystemExit.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
