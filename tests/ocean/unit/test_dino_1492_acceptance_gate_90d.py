"""#1492 item 2.2: pytest wrapper for the 90-day twin acceptance gate.

Runs ``acceptance_gate_90d.py --self-test`` (the synthetic-violation
non-vacuity mode: identical candidate must PASS, a >>10x-threshold field
perturbation must FAIL every metric) as a subprocess, skipping cleanly when
the NEMO oracle tree is absent (the multistep-replay suite's pattern --
importing the gate module opens mesh_mask.nc, so the check must run first).
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
GATE = _REPO / "scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py"

_DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
_RUN_TRAJ = os.environ.get("DINO_NEMO_RUN_TRAJ", f"{_DINO}/RUN_TRAJ")
_RUN_90D = os.environ.get("DINO_NEMO_RUN_90D_TWIN", f"{_DINO}/RUN_90D_TWIN")
_N20_T = f"{_DINO}/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_T.nc"
_N20_U = f"{_DINO}/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_U.nc"


def _have_artifacts() -> bool:
    return (os.path.isfile(f"{_RUN_TRAJ}/mesh_mask.nc")
            and os.path.isfile(_N20_T) and os.path.isfile(_N20_U)
            and len(glob.glob(f"{_RUN_90D}/DINO_00008640_restart_*.nc")) > 0)


def test_acceptance_gate_self_test():
    if not _have_artifacts():
        pytest.skip(
            "NEMO oracle tree not found on this machine (RUN_TRAJ mesh_mask.nc"
            " / RUN_20Y_REBUILD grid_T / RUN_90D_TWIN kt-8640 restart tiles)"
            " -- set $DINO_NEMO_RUN_TRAJ / $DINO_NEMO_RUN_90D_TWIN to point at"
            " a built oracle, or run on a machine with one.")
    r = subprocess.run([sys.executable, str(GATE), "--self-test"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"self-test failed:\n{r.stdout}\n{r.stderr}"
    assert "SELF-TEST PASS" in r.stdout
    assert "[self-checks PASS]" in r.stdout
