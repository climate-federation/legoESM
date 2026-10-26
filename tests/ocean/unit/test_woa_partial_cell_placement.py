"""Observed T/S profiles are sampled at each cell's TRUE centre.

A partial bottom cell's centre sits ABOVE the reference full-cell centre.
Sampling the observation at the reference depth therefore puts water from the
wrong level into every cut column, and the error is largest exactly where cells
are cut -- along the shelf break.  Measured on the production ico8/ETOPO mesh
before this fix: the sampling depth was a median 22 m (max 112 m) from the true
centre, and the first-step pressure-gradient acceleration reached 4.5e-3 m/s^2,
against 4.5e-6 for the same mesh with uniform stratification.

Every expectation below is computed INDEPENDENTLY of the code under test (cell
faces built by hand from the reference thicknesses and the bathymetry), so a
wrong sign or an off-by-one level cannot regenerate its own golden value.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.init_woa import _interp_profile_to_z_coord
from legoesm.ocean.vertical import (
    compute_centroid_depth,
    create_ocean_z_star,
    create_partial_cell_coordinate,
)

# a thermocline sharp enough that a level of offset is unmistakable
WOA_D = np.array([0.0, 100.0, 200.0, 400.0, 700.0, 1000.0, 1500.0])
PROF = np.array([20.0, 18.0, 14.0, 8.0, 4.0, 2.0, 1.0])
N_LEV, H_MAX = 6, 1000.0


def _faces_and_centres(z, H):
    """Cell faces and centres of ONE column, built by hand from the reference
    thicknesses -- deliberately not via compute_centroid_depth."""
    zref = np.abs(np.asarray(z.z_full_ref))
    faces = np.empty(zref.size + 1)
    faces[0] = 0.0
    for k in range(zref.size):
        faces[k + 1] = 2.0 * zref[k] - faces[k]
    h = np.diff(faces)
    h_cut = np.clip(H - faces[:-1], 0.0, h)          # partial thickness
    top = np.concatenate([[0.0], np.cumsum(h_cut)[:-1]])
    return faces, top + 0.5 * h_cut


def test_full_column_is_bit_identical_to_the_reference_sampling():
    z = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX)
    ref = _interp_profile_to_z_coord(PROF, WOA_D, z)
    assert np.array_equal(ref, np.interp(np.abs(np.asarray(z.z_full_ref)), WOA_D, PROF))

    pc = create_partial_cell_coordinate(z, jnp.array([H_MAX, H_MAX]))
    cd = np.asarray(compute_centroid_depth(jnp.zeros(2), jnp.array([H_MAX, H_MAX]), pc))
    # Through the centroid helper an uncut column reproduces the reference
    # sampling to the coordinate's own float32 depth precision (~1e-5 m), not
    # bitwise; the DEFAULT path above is the bitwise one.
    assert _interp_profile_to_z_coord(PROF, WOA_D, pc, cd[0]) == pytest.approx(
        ref, abs=1e-6), "an uncut column must not move"


def test_cut_column_samples_at_its_own_centre():
    """The value in a cut bottom cell is the profile at THAT cell's centre."""
    H = jnp.array([H_MAX, 640.0])
    z = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX)
    pc = create_partial_cell_coordinate(z, H)
    cd = np.asarray(compute_centroid_depth(jnp.zeros_like(H), H, pc))

    _, centres_hand = _faces_and_centres(z, 640.0)
    assert cd[1] == pytest.approx(centres_hand, abs=1e-3), \
        "the helper's centres must match hand-built faces (mm tolerance: the "\
        "coordinate arrays are float32; an off-by-one level is metres)"
    assert np.all(np.diff(cd[1]) >= 0), "centres must not decrease downward"
    assert cd[1].max() <= 640.0 + 1e-6, "no centre may lie below the seafloor"

    got = _interp_profile_to_z_coord(PROF, WOA_D, pc, cd[1])
    want = np.interp(centres_hand, WOA_D, PROF)          # independent expectation
    # 1e-5 K: the helper's depths are float32 (~1e-5 m), and the profile's
    # steepest slope is ~0.02 K/m.  Five orders below the 1.7 K the fix moves.
    assert got == pytest.approx(want, abs=1e-5)

    old = _interp_profile_to_z_coord(PROF, WOA_D, pc)    # reference-depth sampling
    assert np.abs(got - old).max() > 1.0, \
        "the fix must move a cut column, or this test proves nothing"
    # ... and it moves in the physically correct direction: a shallower sampling
    # depth in a profile that cools downward means WARMER water.
    kb = int(np.abs(got - old).argmax())
    assert centres_hand[kb] < np.abs(np.asarray(z.z_full_ref))[kb]
    assert got[kb] > old[kb], "sampling higher up a cooling profile must warm the cell"


def test_column_heat_content_follows_the_profile():
    """Off-by-one in the depths would break the column integral."""
    H = 640.0
    z = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX)
    pc = create_partial_cell_coordinate(z, jnp.array([H_MAX, H]))
    cd = np.asarray(compute_centroid_depth(jnp.zeros(2), jnp.array([H_MAX, H]), pc))
    faces, centres = _faces_and_centres(z, H)
    h_cut = np.clip(H - faces[:-1], 0.0, np.diff(faces))

    got = _interp_profile_to_z_coord(PROF, WOA_D, pc, cd[1])
    model_int = float((got * h_cut).sum())
    fine = np.linspace(0.0, H, 20001)
    exact_int = float(np.trapezoid(np.interp(fine, WOA_D, PROF), fine))
    assert model_int == pytest.approx(exact_int, rel=0.02)

    old = _interp_profile_to_z_coord(PROF, WOA_D, pc)
    assert abs(float((old * h_cut).sum()) - exact_int) > abs(model_int - exact_int), \
        "the reference-depth sampling must be the worse column integral"


def test_bad_depths_are_refused():
    z = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX)
    with pytest.raises(ValueError, match="finite"):
        _interp_profile_to_z_coord(PROF, WOA_D, z, np.full(N_LEV, np.nan))
    with pytest.raises(ValueError, match="decrease"):
        _interp_profile_to_z_coord(PROF, WOA_D, z, np.arange(N_LEV, 0, -1.0))
    # ... but a FLAT tail is legitimate: cells below the seafloor carry zero
    # thickness and inherit the bottom depth.  This is what a real cut column
    # hands the routine, and rejecting it aborted a production run.
    flat_tail = np.array([5.0, 20.0, 60.0, 100.0, 100.0, 100.0])
    assert _interp_profile_to_z_coord(PROF, WOA_D, z, flat_tail) == pytest.approx(
        np.interp(flat_tail, WOA_D, PROF))


def test_driver_forwards_true_cell_depths_to_the_initializer():
    """The leaf helper above is exercised directly; this pins the DRIVER wiring
    (codex review: removing the centroid argument left every test green).
    ``run_omip_single`` is the function that runs the initialisation.  This is a
    source check: it catches the argument being dropped, not ``_cell_depths``
    being reset to ``None`` upstream (executing the driver's placement block in
    a unit test would need the whole grid/bathymetry setup)."""
    import importlib.util
    import inspect
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("run_omip", root / "scripts/run/run_omip.py")
    mod = sys.modules.get("run_omip") or importlib.util.module_from_spec(spec)
    if "run_omip" not in sys.modules:
        sys.modules["run_omip"] = mod
        spec.loader.exec_module(mod)
    src = inspect.getsource(mod.run_omip_single)
    assert "compute_centroid_depth(" in src
    assert "cell_center_depths=_cell_depths" in src
