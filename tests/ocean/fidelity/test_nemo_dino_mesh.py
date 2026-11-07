"""The analytic NEMO DINO mesh, and the gate that pins it bit-for-bit.

Two layers, because they can fail for different reasons:

* FILE-FREE tests run everywhere and check the parts of NEMO's construction
  whose answers are known independently of the oracle file -- the domain size
  the namelist implies, the shape of the land ring, and the refusal to build a
  namelist branch that was never transcribed.
* The GATE test runs the real bit-for-bit comparison against NEMO's own
  ``mesh_mask.nc``.  It is SKIPPED (never silently passed) when that file is
  not on this machine, and it also runs the gate's own ``--plant``, so a gate
  that could not fail would be caught here rather than believed.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
from legoesm.ocean.fidelity import nemo_dino_mesh as ndm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_GATE = os.path.join(_REPO, "scripts", "validate", "ocean_fidelity",
                     "dino_1226", "nemo_dino_mesh_gate.py")
_MESH = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/"
         "mesh_mask.nc")


def test_domain_size_is_nemos():
    # usrdef_nam.F90:141-155.  Cross-checked against RUN_TRAJ/ocean.output:39-40
    # ("nn_piglo = 52", "nn_pjglo = 199"); jpiglo=56/jpjglo=203 there are these
    # plus 2*nn_hls and are NOT the mesh_mask's shape.
    assert ndm.nemo_dino_domain_size() == (52, 199, 36)


def test_merc_proj_matches_nemos_printed_indices():
    # ocean.output:28-29 prints these two, so they are the oracle's own answer.
    assert ndm.merc_proj(70.0, 1.0) == -98
    assert ndm.merc_proj(-70.0, 1.0) == 100
    # ...and the channel edges the land ring is cut at.
    assert ndm.merc_proj(-65.0, 1.0) == 87
    assert ndm.merc_proj(-45.0, 1.0) == 51


def test_land_ring_is_the_partial_periodic_channel():
    g = ndm.nemo_dino_mesh()
    surf = g.tmask[:, :, 0] > 0.5
    # First and last ROW closed (domzgr.F90:308-314, l_Jperio false).
    assert not surf[0].any() and not surf[-1].any()
    # First and last COLUMN wet ONLY in the ACC band (usrdef_zgr.F90:150-163):
    # rows 14..48 inclusive, i.e. the 35-row re-entrant channel.
    for col in (0, -1):
        wet_rows = np.nonzero(surf[:, col])[0]
        assert wet_rows.min() == 14 and wet_rows.max() == 48
        assert wet_rows.size == 35
    # Everything between the walls is wet on every open row.
    assert surf[1:-1, 1:-1].all()


def test_masks_are_the_neighbour_products_nemo_builds():
    # dommsk.F90:150-151, with i wrapping (ln_Iperio) and j closed.
    g = ndm.nemo_dino_mesh()
    assert np.array_equal(g.umask, g.tmask * np.roll(g.tmask, -1, axis=1))
    assert np.array_equal(g.vmask[:-1], g.tmask[:-1] * g.tmask[1:])
    assert not g.vmask[-1].any()


def test_ladders_are_fp64_and_close_the_column():
    g = ndm.nemo_dino_mesh()
    for a in (g.e3t_1d, g.gdept_1d, g.gdepw_1d, g.glamt, g.gphit, g.e1t):
        assert np.asarray(a).dtype == np.float64
    # 35 wet levels span exactly the basin depth; the 36th is NEMO's dummy.
    assert float(np.sum(g.e3t_1d[:35])) == pytest.approx(4000.0, abs=1e-9)
    assert not (g.tmask[:, :, 35] > 0.5).any()


@pytest.mark.parametrize("field,value", [
    ("nn_botcase", 0), ("ln_zco_nam", False), ("ln_mid_ridge", True),
    ("ln_Iperio", False), ("ln_drake_sill", False),
])
def test_untranscribed_namelist_branches_raise(field, value):
    # Dispatch hardening: DINO's namelist has branches this transcription does
    # not cover, and silently returning the wrong domain is the failure mode
    # this whole module exists to remove.
    nml = ndm.NEMO_DINO_R1._replace(**{field: value})
    with pytest.raises(ValueError):
        ndm.nemo_dino_mesh(nml)


def test_bathymetry_outside_the_ladder_band_raises():
    # NEMO's z2d in zgr_msk_top_bot is uninitialised outside
    # (gdept_1d(1), gdept_1d(jpk)] (usrdef_zgr.F90:456,473), so no
    # transcription can match it there. Refuse rather than diverge.
    hgr = ndm.nemo_dino_hgr()
    lad = ndm.vertical_ladders(ndm.NEMO_DINO_R1, 36)
    with pytest.raises(ValueError, match="uninitialised"):
        ndm.nemo_dino_bathymetry(
            ndm.NEMO_DINO_R1._replace(rn_H=9.0, rn_hborder=1.0),
            hgr, lad["gdept_1d"])


def _run_gate(*extra):
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO, "packages", p) for p in
         ("core", "ocean", "atmosphere", "coupler", "ice", "land", "ml", "tools")]
        + [os.path.join(_REPO, "src"), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, _GATE, *extra],
                          capture_output=True, text=True, env=env, cwd=_REPO)


@pytest.mark.skipif(not os.path.exists(_MESH),
                    reason=f"NEMO oracle mesh not on this machine: {_MESH}")
@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed: the gate cannot read the "
                           "oracle mesh, so a failure here would not be a "
                           "gate regression")
def test_gate_passes_against_the_oracle_mesh():
    r = _run_gate()
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "GATE PASS" in r.stdout
    assert "0 unaccounted" in r.stdout


@pytest.mark.skipif(not os.path.exists(_MESH),
                    reason=f"NEMO oracle mesh not on this machine: {_MESH}")
@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed: the gate cannot read the "
                           "oracle mesh, so a failure here would not be a "
                           "gate regression")
def test_gate_plant_fails():
    # Non-vacuity: a one-cell lie about the bathymetry must be caught.
    r = _run_gate("--plant")
    assert r.returncode != 0
    assert "GATE FAIL" in r.stdout


@pytest.mark.skipif(not os.path.exists(_MESH),
                    reason=f"NEMO oracle mesh not on this machine: {_MESH}")
@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed: the gate cannot read the "
                           "oracle mesh, so a failure here would not be a "
                           "gate regression")
def test_model_consumed_geometry_is_nemos_own_ladder():
    """Section 4: the card RUNS on NEMO's e3t_0, not only carries it.

    Section 1+2 is blind to this (Rule 2: a mesh gate cannot see which of the
    two ladders the model integrates), which is how the card spent months
    cutting its staircase from `e3t_1d` while NEMO's DINO integrates `e3t_0`
    (`key_vco_3d`, domzgr_substitute.h90:102).
    """
    r = _run_gate()
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    body = r.stdout.split("SECTION 4")[-1]
    assert "FAIL" not in body, body
    for row in ("e3t(Kmm) live thickness", "e3t_0 reference ladder",
                "gdept_0 T-depth ladder", "e3u_0 live face thickness",
                "ht_0 column depth", "r1_hu_0"):
        assert row in body, f"section 4 lost its {row!r} row:\n{body}"


@pytest.mark.skipif(not os.path.exists(_MESH),
                    reason=f"NEMO oracle mesh not on this machine: {_MESH}")
@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed: the gate cannot read the "
                           "oracle mesh, so a failure here would not be a "
                           "gate regression")
def test_ladder_plant_flips_every_derived_row():
    """Non-vacuity for section 4, row by row rather than "the gate failed".

    The plant moves the CONSTRUCTED ladder by 1 ulp at every level.  Each
    ladder-derived row must flip; each raw mesh row must NOT, because it is
    carried straight off the mesh and is not derived from the planted value.
    That asymmetry is what proves the derived rows are the ones under test.
    """
    r = _run_gate("--plant-ladder")
    assert r.returncode != 0
    assert "8 of 8 ladder-derived rows flipped to FAIL" in r.stdout, \
        r.stdout[-4000:]
    assert "PLANT DID NOT REACH" not in r.stdout
    assert "PLANT REACHED A RAW MESH ROW" not in r.stdout


@pytest.mark.skipif(not os.path.exists(_MESH),
                    reason=f"NEMO oracle mesh not on this machine: {_MESH}")
@pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed: the gate cannot read the "
                           "oracle mesh, so a failure here would not be a "
                           "gate regression")
def test_a_lie_about_the_dry_level_is_refused():
    """The permanently-dry level is SCORED, not waived (2026-09-10).

    DINO's level 36 is NEMO's permanently dry dummy.  It used to be excluded
    from every scored row and covered by a waiver, because the bridge left
    the unrelated 1-D ladder's value there.  The user's instruction was to do
    exactly what NEMO does, so the level now takes NEMO's own e3t_0/gdept_0
    and the two ladder rows cover all 36 levels.

    This test is the non-vacuity check for that: an arbitrary value planted
    on the dry level must make those rows go red and the gate exit non-zero.
    Before the change, the same plant passed the whole gate.
    """
    r = _run_gate("--plant-dry-level")
    assert r.returncode != 0, r.stdout[-4000:]
    assert "e3t_0 reference ladder" in r.stdout, r.stdout[-4000:]
    # the two ladder rows are the ones that must catch it
    for row in ("e3t_0 reference ladder", "gdept_0 T-depth ladder"):
        line = [ln for ln in r.stdout.splitlines() if ln.startswith(row)]
        assert line and "FAIL" in line[0], (row, r.stdout[-4000:])
    # and it must not fire by accident on the honest run
    ok = _run_gate()
    assert ok.returncode == 0
    assert "scored, not waived" in ok.stdout
    # the coordinate-branch guard must have actually looked
    assert "partial-cell: True" in ok.stdout, ok.stdout[-4000:]
