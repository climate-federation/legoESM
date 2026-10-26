"""The implicit vertical solve's u/v-face control volume, on a staircase.

WHAT THIS PINS.  ``_apply_implicit_vertical_mixing`` builds the face-cell
thickness it uses both as the tridiagonal control volume and as the divisor of
its own barotropic split (``depth_mean(u_solve_in, dz_u, ...)``).  legoESM's
cell-centre thickness is ALREADY ZERO below the sea floor (``ocean/vertical.py``:
``h_partial = jnp.where(is_active, h_full, 0.0)``), so a cell->face ARITHMETIC
mean hands the solve HALF a reference cell at every face whose two columns have
different bottom levels -- water on a level that exists on neither column, summed
into the column divisor.  Multiplying that average by the
both-cells-wet face mask -- the mask the same block already builds for the
viscosity, three lines away -- gives exactly zero there.

NEMO's own value for the same face column is ``hu_0 = SUM_k e3u_0 * umask``
(``src/OCE/DOM/domain.F90:145``), with ``e3u_0 = MIN(e3t_0(i), e3t_0(i+1))`` on
the partial-step builder (``tools/DOMAINcfg/src/domzgr.F90:1166``) and the
arithmetic mean of the two ``e3t_0`` on the full-step builder
(``cfgs/DINO/MY_SRC/zgr_lib.F90:231``) -- identical on the horizontally uniform
reference ladder DINO's ``ln_zco_nam=.true.`` card has.  Either way the level is
REMOVED, not halved, so the reference below is computed the NEMO way: sum the
reference thickness over the levels where BOTH adjacent columns are wet.

OWNERSHIP.  The construction under test is PR #1642's, merged to main on
2026-08-22 and ported onto this branch verbatim rather than re-invented; this
file is the branch's own gate on it, and it is deliberately written against the
NEMO VALUE rather than against either candidate rule, so it passes for main's
masked average and would also pass for the min rule -- what it forbids is the
half.

NON-VACUITY, measured not reasoned.  Reverting the two lines under test
(``dz_u = dz_u * _act_u3`` and its ``dz_v`` sibling) makes
``test_u_face_column_depth_is_the_nemo_value`` and
``test_v_face_column_depth_is_the_nemo_value`` FAIL, by exactly the half-cell
sum that ``test_the_staircase_is_real_and_the_two_rules_disagree`` computes
independently.  That third test is the premise check: it uses NO model code, so
if the fixture ever stopped containing a staircase the other two would become
vacuous and it goes red first.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N_LAT, N_LON, N_LEV, H_MAX = 6, 8, 6, 3000.0
_DT = 1800.0


def _staircase_channel():
    """Rest channel whose bathymetry steps DOWN in i and in j.

    The step is placed on a reference-ladder INTERFACE so every cell is a full
    step (DINO's ln_zco card), which keeps the expected numbers exact and makes
    the only difference between the two rules the masked level itself.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import (
        create_ocean_z_star,
        create_partial_cell_coordinate,
    )

    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z0c = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX)
    zhalf = np.asarray(z0c.z_half_ref, dtype=np.float64)   # interface depths
    deep = float(abs(zhalf[N_LEV]))                        # full column
    shallow = float(abs(zhalf[N_LEV - 2]))                 # two levels less

    H = np.full((N_LAT, N_LON), deep)
    H[:, :3] = shallow            # zonal step -> staircase u-faces at i=2/3
    H[:2, :] = shallow            # meridional step -> staircase v-faces at j=1/2
    H_bathy = jnp.asarray(H)
    z = create_partial_cell_coordinate(z0c, H_bathy)
    state = rest_state_latlon_cgrid_ocean(
        grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy)
    config = LatLonCGridOceanConfig.from_flat(
        A_v=1.0e-4, K_v=1.0e-5, A_h=0.0, K_h=0.0,
        implicit_vertical_mixing=True, zdf_baroclinic_only=True,
    )
    return grid, z, state, config


def _nemo_reference_face_depths(z):
    """``hu_0``/``hv_0`` the NEMO way: the level is REMOVED, never halved.

    Uses only the coordinate's own ``is_active`` mask and reference ladder --
    no model face operator -- so this cannot agree with the code under test by
    construction.
    """
    act = np.asarray(z.is_active).astype(bool)              # (lat, lon, lev)
    dz = np.asarray(z.dz_ref, dtype=np.float64)             # (lev,)
    both_u = act[:, :-1, :] & act[:, 1:, :]                 # face between i,i+1
    both_v = act[:-1, :, :] & act[1:, :, :]                 # face between j,j+1
    return (both_u * dz).sum(-1), (both_v * dz).sum(-1)


def _arithmetic_mean_face_depths(z):
    """What the arithmetic mean of the ALREADY-MASKED cell thickness gives."""
    act = np.asarray(z.is_active).astype(np.float64)
    dz_cell = act * np.asarray(z.dz_ref, dtype=np.float64)
    mean_u = 0.5 * (dz_cell[:, :-1, :] + dz_cell[:, 1:, :])
    mean_v = 0.5 * (dz_cell[:-1, :, :] + dz_cell[1:, :, :])
    return mean_u.sum(-1), mean_v.sum(-1)


def _capture_solve_face_thickness():
    """Run one step and return the (u, v) face thickness the solve divided by.

    Captured at ``depth_mean``, which is the barotropic split's own divisor
    call inside ``_apply_implicit_vertical_mixing`` -- i.e. the quantity under
    test, not a re-derivation of it.
    """
    # ``jax.disable_jit`` is load-bearing for the capture, not a different
    # code path: it keeps the intercepted thickness a concrete array instead of
    # a tracer, and stops a cached compiled step from ignoring the patch.  The
    # Python statements executed are the production ones.
    from unittest import mock

    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as omlc

    grid, z, state, config = _staircase_channel()
    model = omlc.LatLonCGridOceanModel(grid, z, config)
    real = omlc.depth_mean
    seen = []

    def _spy(field, h_face, *a, **kw):
        seen.append(np.asarray(h_face, dtype=np.float64))
        return real(field, h_face, *a, **kw)

    # disable_jit so the captured thickness is a CONCRETE array, not a tracer.
    with jax.disable_jit(), mock.patch.object(omlc, "depth_mean", _spy):
        model._step_impl(state, dt=_DT)

    u_shape = (N_LAT, N_LON + 1, N_LEV)
    v_shape = (N_LAT + 1, N_LON, N_LEV)
    h_u = [h for h in seen if h.shape == u_shape]
    h_v = [h for h in seen if h.shape == v_shape]
    # Exactly one call each: taking "the first match" silently starts
    # measuring a different stage if any earlier one ever makes a call of the
    # same shape, and the test would stay green.
    assert len(h_u) == 1 and len(h_v) == 1, (
        f"expected exactly one u-face and one v-face depth_mean call, saw "
        f"{len(h_u)} and {len(h_v)} -- the capture may be reading a "
        "different stage than the implicit solve")
    assert h_u and h_v, (
        "the implicit solve's depth_mean was never reached with a 3-D face "
        f"thickness -- shapes seen: {[h.shape for h in seen]}.  The capture, "
        "not the model, is what failed; fix it before reading any number.")
    return h_u[0].sum(-1), h_v[0].sum(-1), z


def test_the_staircase_is_real_and_the_two_rules_disagree():
    """Premise check, model-free: the fixture DOES contain staircase faces and
    the two rules give different column depths there, by the half-cell sum."""
    _, z, _, _ = _staircase_channel()
    nemo_u, nemo_v = _nemo_reference_face_depths(z)
    mean_u, mean_v = _arithmetic_mean_face_depths(z)
    du, dv = mean_u - nemo_u, mean_v - nemo_v
    dz = np.asarray(z.dz_ref, dtype=np.float64)
    half_two_levels = 0.5 * (dz[N_LEV - 2] + dz[N_LEV - 1])
    assert np.count_nonzero(du > 1e-9) > 0, "no staircase u-face in the fixture"
    assert np.count_nonzero(dv > 1e-9) > 0, "no staircase v-face in the fixture"
    np.testing.assert_allclose(du.max(), half_two_levels, rtol=1e-12)
    np.testing.assert_allclose(dv.max(), half_two_levels, rtol=1e-12)


def test_u_face_column_depth_is_the_nemo_value():
    H_u, _, z = _capture_solve_face_thickness()
    nemo_u, _ = _nemo_reference_face_depths(z)
    mean_u, _ = _arithmetic_mean_face_depths(z)
    # interior u-faces only: index 0 is the west-wall column, index -1 the
    # periodic wrap of it, and neither is a face between two of these columns.
    np.testing.assert_allclose(H_u[:, 1:-1], nemo_u, rtol=0, atol=1e-9)
    assert np.abs(H_u[:, 1:-1] - mean_u).max() > 1e-3, (
        "the masked and unmasked averages agree everywhere on this fixture "
        "-- the test proves nothing")


def test_v_face_column_depth_is_the_nemo_value():
    _, H_v, z = _capture_solve_face_thickness()
    _, nemo_v = _nemo_reference_face_depths(z)
    _, mean_v = _arithmetic_mean_face_depths(z)
    # v-face row 0 is the south wall and row -1 the north wall (both zeroed);
    # rows 1..N_LAT-1 are the faces between adjacent cell rows.
    np.testing.assert_allclose(H_v[1:-1], nemo_v, rtol=0, atol=1e-9)
    assert np.abs(H_v[1:-1] - mean_v).max() > 1e-3, (
        "the masked and unmasked averages agree everywhere on this fixture "
        "-- the test proves nothing")


def test_no_face_carries_water_on_a_level_neither_column_has():
    """The property the fix exists for, stated directly."""
    H_u, H_v, z = _capture_solve_face_thickness()
    nemo_u, nemo_v = _nemo_reference_face_depths(z)
    assert (H_u[:, 1:-1] - nemo_u).max() <= 1e-9
    assert (H_v[1:-1] - nemo_v).max() <= 1e-9
