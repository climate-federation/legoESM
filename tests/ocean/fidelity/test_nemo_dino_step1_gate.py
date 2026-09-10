"""The gate that scores legoESM's FIRST STEP against NEMO's kt=1 record.

The gate itself is the thing under test here.  It is easy to write a
comparison that cannot fail, and easier still to write one that happily
scores the WRONG step -- so the three tests are: it passes when handed the
oracle's own answer, it fails on a one-ulp lie about that answer, and it
REFUSES a snapshot that is not step one rather than reporting a number for it.

Skipped (never silently passed) when NEMO's record or netCDF4 is absent.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_GATE = os.path.join(_REPO, "scripts", "validate", "ocean_fidelity", "dino_1226",
                     "nemo_dino_step1_gate.py")
_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
            "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")

_HAVE = bool(glob.glob(_RESTART))
_needs_oracle = pytest.mark.skipif(
    not _HAVE, reason=f"NEMO from-rest restart not here: {_RESTART}")
_needs_nc = pytest.mark.skipif(
    importlib.util.find_spec("netCDF4") is None,
    reason="netCDF4 not installed: the gate cannot read the oracle restart, "
           "so a failure here would not be a gate regression")


def _run_gate(*extra):
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO, "packages", p) for p in
         ("core", "ocean", "atmosphere", "coupler", "ice", "land", "ml", "tools")]
        + [os.path.join(_REPO, "src"), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, _GATE, *extra],
                          capture_output=True, text=True, env=env, cwd=_REPO)


@_needs_oracle
@_needs_nc
def test_gate_passes_on_the_oracles_own_answer():
    """A gate that cannot pass proves nothing when the model fails it."""
    r = _run_gate("--oracle-self-test")
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]
    assert "GATE PASS" in r.stdout
    assert "0 unaccounted" in r.stdout


@_needs_oracle
@_needs_nc
def test_gate_fails_on_a_one_ulp_lie():
    r = _run_gate("--oracle-self-test", "--plant")
    assert r.returncode != 0
    assert "GATE FAIL" in r.stdout
    assert "PLANT ACTIVE" in r.stdout          # the plant was not a no-op


@_needs_oracle
@_needs_nc
def test_gate_refuses_a_snapshot_that_is_not_step_one(tmp_path):
    """Scoring step 2 against NEMO's kt=1 would be a number with no meaning."""
    snaps = tmp_path / "snapshots"
    snaps.mkdir(parents=True)
    np.savez_compressed(
        snaps / "snapshot_00001.npz",
        time_seconds=5400.0, time_days=5400.0 / 86400.0,
        eta=np.zeros((199, 52)), T=np.zeros((199, 52, 36)),
        S=np.zeros((199, 52, 36)), u=np.zeros((199, 53, 36)),
        v=np.zeros((200, 52, 36)),
        land_mask=np.ones((199, 52)), H_bathy=np.ones((199, 52)))
    r = _run_gate("--run-dir", str(tmp_path))
    assert r.returncode != 0
    assert "scores STEP 1 only" in (r.stdout + r.stderr)
