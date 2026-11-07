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
import json
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


def _fake_run(tmp_path, *, dt=2700.0, stride_days=2700.0 / 86400.0,
              arrays=None, land_mask=None, metadata=True):
    """A run directory that never ran. Every test below is an ATTACK."""
    snaps = tmp_path / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    a = arrays or {}
    np.savez_compressed(
        snaps / "snapshot_00001.npz",
        time_seconds=dt, time_days=dt / 86400.0,
        eta=a.get("eta", np.zeros((199, 52))),
        T=a.get("T", np.zeros((199, 52, 36))),
        S=a.get("S", np.zeros((199, 52, 36))),
        u=a.get("u", np.zeros((199, 53, 36))),
        v=a.get("v", np.zeros((200, 52, 36))),
        land_mask=(np.ones((199, 52)) if land_mask is None else land_mask),
        H_bathy=np.ones((199, 52)))
    if metadata:
        (tmp_path / "run_metadata.json").write_text(json.dumps({
            "args": {"dt": dt, "snapshot_every_days": stride_days,
                     "grid": "latlon", "nemo_faithful_grid": True,
                     "recipe": "nemo_dino_kamm_mlf"}}))
    return tmp_path


@_needs_oracle
@_needs_nc
def test_gate_refuses_a_snapshot_with_no_run_behind_it(tmp_path):
    """THE ATTACK THAT WORKED, kept so it cannot work again.

    A reviewer hand-wrote a snapshot holding NEMO's own kt=1 answer over an
    all-land domain, with no legoESM run anywhere. The first version of this
    gate read five arrays and a timestamp, reported every row AT BAR, and
    exited zero under a header claiming it had scored the production driver.
    """
    r = _run_gate("--run-dir", str(_fake_run(tmp_path, metadata=False)))
    assert r.returncode != 0
    assert "no run behind it" in (r.stdout + r.stderr)


@_needs_oracle
@_needs_nc
def test_gate_refuses_a_domain_that_is_not_NEMOs(tmp_path):
    """The forgery's other half: metadata that says the right card, over a
    domain that is not the card's. The land mask is what refuses it."""
    r = _run_gate("--run-dir", str(_fake_run(tmp_path)))
    assert r.returncode != 0
    assert "not NEMO's DINO domain" in (r.stdout + r.stderr)


@_needs_oracle
@_needs_nc
def test_gate_refuses_two_half_steps_that_land_on_the_same_clock(tmp_path):
    """t = 2700 s is not the same claim as ONE step of 2700 s.

    ``--dt 1350`` reaches that clock after TWO steps, and the snapshot itself
    carries no step count -- so the step number has to come from the run's
    own dt and stride, not from the timestamp.
    """
    run = _fake_run(tmp_path, dt=1350.0, stride_days=2700.0 / 86400.0)
    # keep the file's stamp at NEMO's clock, which is the whole trap
    snaps = run / "snapshots"
    d = dict(np.load(snaps / "snapshot_00001.npz"))
    d["time_seconds"] = np.array(2700.0)
    d["time_days"] = np.array(2700.0 / 86400.0)
    np.savez_compressed(snaps / "snapshot_00001.npz", **d)
    r = _run_gate("--run-dir", str(run))
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "not comparable" in out or "STEP 1 only" in out
