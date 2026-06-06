"""iter-36 AMIP dt-safety warning regression.

When ``scripts/run/run_amip.py`` is invoked with ``--dt`` > 2× the
iter-13/iter-26 cross-grid ladder for the (grid_type, resolution)
combination, it should print a WARNING to stderr pointing the
operator at CRM_implementation.md. The warning was added in iter-36
after Codex iter-31/32 flagged that the AMIP wrapper's iter-32
hard-coded overrides only catch the specific resolutions the
wrapper hits; a direct ``run_amip.py`` invocation (e.g. from a
training script, a sweep, or a manual run) bypasses the wrapper
entirely and could silently land in the iter-13-banned dt=600 / C48
configuration.

These tests subprocess `run_amip.py --resolution 48 --dt 600 --days 0`
and check for the WARNING line.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run" / "run_amip.py"


def _run_amip_dryrun(*extra_args):
    """Invoke run_amip.py with --days 0 so the dt-advisory print
    happens before any heavy compute. Capture stderr."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [sys.executable, str(SCRIPT)] + list(extra_args)
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=60,
    )


def test_dt_warning_fires_on_c48_dt600():
    """Forced bad config: C48 cubed_sphere @ dt=600. The iter-13
    measurement said dt=600 BLEW UP C48 at day 25. iter-36 dt
    advisory must surface that on the way in."""
    result = _run_amip_dryrun(
        "--grid-type", "cubed_sphere",
        "--discretization", "cdgrid",
        "--resolution", "48",
        "--dt", "600",
        "--days", "0",
    )
    # stdout/stderr both inspected since some Python configurations
    # mix them.
    combined = (result.stderr or "") + (result.stdout or "")
    assert "WARNING: --dt 600" in combined, (
        f"AMIP at C48/dt=600 should emit the iter-36 dt warning; "
        f"got:\n---stdout---\n{result.stdout}\n---stderr---\n{result.stderr}"
    )
    assert "ladder" in combined.lower()


def test_dt_warning_silent_at_c24_dt600():
    """C24 at dt=600 is the iter-12-validated ladder value. The
    advisory must NOT fire."""
    result = _run_amip_dryrun(
        "--grid-type", "cubed_sphere",
        "--discretization", "cdgrid",
        "--resolution", "24",
        "--dt", "600",
        "--days", "0",
    )
    combined = (result.stderr or "") + (result.stdout or "")
    assert "WARNING: --dt" not in combined, (
        f"AMIP at C24/dt=600 (valid ladder value) should not emit "
        f"the dt warning; got:\n{combined}"
    )


def test_dt_warning_silent_at_n_gt_96():
    """N>96 makes the ladder raise; the warning path catches the
    ValueError and skips silently so the user doesn't get a
    spurious dt-warning on top of a different downstream error."""
    result = _run_amip_dryrun(
        "--grid-type", "cubed_sphere",
        "--discretization", "cdgrid",
        "--resolution", "144",
        "--dt", "600",
        "--days", "0",
    )
    combined = (result.stderr or "") + (result.stdout or "")
    # The dt warning compares against the ladder; ValueError means
    # we can't compute the ratio, so no warning should fire.
    assert "WARNING: --dt" not in combined, (
        f"AMIP at N>96 should silently skip the dt warning (ladder "
        f"refuses); got:\n{combined}"
    )
