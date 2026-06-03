"""Shared helpers for the bench-smoke tests (iter-238..260 chain).

iter-261 (Codex iter-256 LOW): the 6 bench smoke tests added in
iter-238..260 (test_run_rcemip_long_cross_grid_smoke,
test_bench_plane_crm_dd_scaling_smoke, test_bench_halo_ops_scaling_smoke,
test_bench_dd_scaling_smoke, test_bench_halo_exchange_smoke,
test_bench_mpi_scaling_smoke, test_bench_plane_dycore_smoke) all
share subprocess invocation + env setup + returncode-fail
formatting. Extract here.

This file is NOT a test file (no ``test_`` prefix on functions);
pytest's ``test_*.py`` glob skips it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]


def run_bench(
    script: Path,
    cmd_args: list[str],
    *,
    timeout_s: int = 180,
) -> subprocess.CompletedProcess:
    """Invoke a bench script as a subprocess with the standard
    CPU-only / x64-enabled env.

    Parameters
    ----------
    script : Path
        Path to the ``scripts/bench_*.py`` (or similar) under test.
    cmd_args : list[str]
        Arg vector AFTER the script path (e.g. ``["--nx", "12", ...]``).
    timeout_s : int, default 180
        Subprocess timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process. Caller asserts ``returncode == 0`` +
        further-introspects stdout/stderr; this helper does NOT
        ``pytest.fail`` on nonzero returncode (caller's job, with
        the test's own context-specific message).
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [sys.executable, str(script), *cmd_args]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
    )


def fail_on_nonzero(
    result: subprocess.CompletedProcess,
    script_name: str,
    extra_kwargs: str = "",
) -> None:
    """Standard ``pytest.fail`` path for a nonzero subprocess
    returncode. Used by every bench smoke to keep the failure
    message format consistent.

    Parameters
    ----------
    result : subprocess.CompletedProcess
        The result of ``run_bench(...)``.
    script_name : str
        Short name for the failure message (e.g. ``"bench_dd_scaling.py"``).
    extra_kwargs : str, optional
        Optional ``" --mode strong"`` suffix for parametrised tests.
    """
    if result.returncode == 0:
        return
    pytest.fail(
        f"{script_name}{extra_kwargs} exited {result.returncode}\n"
        f"stdout tail:\n{result.stdout[-1500:]}\n"
        f"stderr tail:\n{result.stderr[-1500:]}"
    )
