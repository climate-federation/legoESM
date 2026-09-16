"""NEMO's DINO initial state, and the gate that pins the from-rest start.

Two layers, for the same reason as ``test_nemo_dino_mesh.py``:

* FILE-FREE tests check what is knowable without the oracle's restart --
  which namelist branch is transcribed, that the meridional-blend anchors are
  taken from the model fields rather than from the namelist, and that a
  non-NEMO card is left on the paper profile.
* The GATE test runs the real bit-for-bit comparison against NEMO's
  ``RUN_FROMREST_KT1`` restart.  It is SKIPPED (never silently passed) when
  that record is not on this machine, and it also runs the gate's own
  ``--plant``, so a gate that could not fail would be caught here rather than
  believed.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
from legoesm.ocean.fidelity import nemo_dino_mesh as ndm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_GATE = os.path.join(_REPO, "scripts", "validate", "ocean_fidelity", "dino_1226",
                     "nemo_dino_fromrest_gate.py")
_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
            "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")


def test_only_the_transcribed_initcase_is_accepted():
    g = ndm.nemo_dino_mesh()
    for case in (0, 1, 2, 3, 5):
        with pytest.raises(ValueError, match="nn_initcase"):
            ndm.nemo_dino_istate(g, nn_initcase=case)


def test_blend_anchors_come_from_the_model_fields_not_the_namelist():
    """usrdef_istate.F90:151-155 -- MAXVAL(gphit) and a MINVAL over WET cells.

    Using ``rn_phi_max`` (70) or the deepest REFERENCE level instead is the
    defect issue #1729 measured, so pin both anchors by recovering them from
    the built field rather than trusting the docstring (Rule 10).
    """
    g = ndm.nemo_dino_mesh()
    T, S = ndm.nemo_dino_istate(g)
    wet = g.tmask > 0.5
    # At the poleward-most row the blend factor is 0, so T there IS zTbot.
    j_max = int(np.argmax(g.gphit[:, 0]))
    t_bot = float(T[wet].min())
    assert np.allclose(T[j_max][wet[j_max]], t_bot, rtol=0, atol=0)
    # zphiMAX is MAXVAL(gphit) = 69.85..., NOT rn_phi_max = 70: with 70 the
    # poleward row would keep a residual (1 - |phi|/70) of the profile.
    assert float(g.gphit.max()) < 70.0
    # And zTbot is the deepest WET value, not the deepest REFERENCE level:
    # level jpk is dry everywhere, so its profile value never appears.
    assert g.tmask[:, :, -1].max() == 0.0
    assert t_bot > 0.0 and float(S[wet].min()) > 30.0


def test_a_uniform_gdept_is_required():
    g = ndm.nemo_dino_mesh()
    bad = np.array(g.gdept_0, copy=True)
    bad[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="horizontally uniform"):
        ndm.nemo_dino_istate(g._replace(gdept_0=bad))


def _run_gate(*extra):
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO, "packages", p) for p in
         ("core", "ocean", "atmosphere", "coupler", "ice", "land", "ml", "tools")]
        + [os.path.join(_REPO, "src"), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, _GATE, *extra],
                          capture_output=True, text=True, env=env, cwd=_REPO)


_HAVE_RESTART = bool(glob.glob(_RESTART))
_skip = [
    pytest.mark.skipif(not _HAVE_RESTART,
                       reason=f"NEMO from-rest restart not here: {_RESTART}"),
    pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                       reason="netCDF4 not installed: the gate cannot read the "
                              "oracle restart, so a failure here would not be "
                              "a gate regression"),
]


@pytest.mark.skipif(not _HAVE_RESTART, reason="NEMO from-rest restart not here")
@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_gate_passes_against_the_oracle_restart():
    r = _run_gate()
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]
    assert "GATE PASS" in r.stdout
    assert "0 unaccounted" in r.stdout


@pytest.mark.skipif(not _HAVE_RESTART, reason="NEMO from-rest restart not here")
@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_gate_plant_fails():
    # Non-vacuity: a 1-ulp lie about the initial temperature AND about the
    # source latitudes must both be caught.  The gate refuses to run a plant
    # it has measured to be a no-op, so a green --plant here cannot be
    # vacuous.
    r = _run_gate("--plant")
    assert r.returncode != 0
    assert "GATE FAIL" in r.stdout
    assert "PLANT ACTIVE" in r.stdout
