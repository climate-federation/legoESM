"""The sedimentation sub-step count reaches a REAL run's log.

Both reviewers (codex + GLM, 2026-09-22) rejected the unit tests that drive
``report_sed_substep_overflow`` / ``sed_substeps_window_max`` directly: they
stay green if the driver's own calls are deleted.  This runs the real MPAS
loop (``ModelDriver._run_mpas`` via the coupled voronoi driver, the same
harness as the surface-flux export regression) for fewer steps than the
report cadence, so the ONLY thing that can emit the line is the post-loop
flush.  Deleting the flush, the slot publication, the microphysics count or
the tendency field all turn this red.  Subprocess, ~1 min.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER = REPO_ROOT / "scripts" / "run" / "run_coupled.py"


def test_mpas_run_logs_the_required_sedimentation_substeps(tmp_path):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    # a smaller compile footprint: the shared node also runs the campaign's
    # 60-day arms, and LLVM there failed to allocate memory mid-compile
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + " --xla_cpu_enable_fast_math=false").strip()
    env["OMP_NUM_THREADS"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--grid", "voronoi", "-n", "3",
        "--ocean", "slab", "--radiation", "gray",
        "--days", "1", "--diag-days", "1",
        "--output", str(tmp_path / "sed_substeps_report"),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=900)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"coupled voronoi run failed: exit {result.returncode}\n"
        f"{combined[-2000:]}")
    _hits = re.findall(
        r"microphysics sedimentation: max (\d+) CFL sub-steps required of "
        r"(\d+) available, at step (\d+)", combined)
    m = re.search(r"microphysics sedimentation: max (\d+) CFL sub-steps "
                  r"required of (\d+) available", combined)
    assert m is not None, (
        "the run never reported its sedimentation sub-step count: the count "
        "does not reach the driver (publication slot, tendency field, or the "
        "post-loop flush is missing).\n" + combined[-2000:])
    required, cap = int(m.group(1)), int(m.group(2))
    assert required >= 1                      # a real count, not a placeholder
    assert cap == 256                         # the effective cap, not a literal
    # The run is shorter than the report cadence, so the ONLY reporter is the
    # post-loop flush, and it reports at the last step -- deleting the flush
    # leaves nothing behind (codex 2026-09-22).
    _steps = sorted(int(h[2]) for h in _hits)
    assert len(_hits) == 1, (
        "the run is longer than the report cadence, so a mid-run report "
        "could satisfy this test without the flush: shorten the case "
        f"(reports at steps {_steps})")
    assert _steps and _steps[-1] > 0, (
        "only the opening report survived: the post-loop flush is missing.\n"
        + combined[-2000:])
