"""The stage-split MPAS water-closure probe runs and prints its budget lines."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts/validate/mpas_water_closure_stages.py"


@pytest.mark.slow
def test_probe_two_steps_reports_the_kessler_residual():
    days = 2 * 600.0 / 86400.0                      # two 600 s steps
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    run = subprocess.run([sys.executable, str(PROBE), repr(days), "full"],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=3600)
    out = run.stdout
    assert run.returncode == 0, run.stderr[-2000:]
    assert re.search(r"^status \S+ steps 2$", out, re.M), out[-2000:]
    kres = re.search(r"kessler own column residual .* = (\S+) kg over (\d+) calls", out)
    assert kres and int(kres.group(2)) == 2, out[-2000:]
    float(kres.group(1))
    for key in ("SUM step_dW=", "SUM phys=", "SUM E="):
        assert key in out, key
