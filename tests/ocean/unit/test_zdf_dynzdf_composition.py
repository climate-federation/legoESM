"""Tests for the #1226 NEMO dynzdf composition options.

``LatLonCGridOceanConfig.zdf_drag_in_matrix`` / ``zdf_baroclinic_only``
transcribe two dynzdf.F90 pieces of NEMO's implicit vertical-friction
composition (see the field docstrings in ``ocean/state.py`` for the full
line-cited derivation):

1. ``zdf_drag_in_matrix`` (dynzdf.F90:293-305 + 148-171): semi-implicit
   bottom friction goes INTO the tridiagonal diagonal at the deepest wet
   cell, instead of an explicit RHS kick — and the model's own OUTSIDE
   bottom-drag application (``_bc_bottom_drag``) is disabled on that path
   (single-owner; no double-drag).
2. ``zdf_baroclinic_only`` (dynzdf.F90:119-171): the implicit solve acts on
   the baroclinic residual only (depth mean stripped before, re-added after
   — exactly conservative at A_v=0).

Both default False -> BIT-IDENTICAL to the pre-existing code path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


CD0, CDMAX, Z0, KE0 = 1.0e-3, 0.1, 3.0e-3, 2.5e-3


def _partial_cell_channel(n_lat=6, n_lon=8, n_levels=5, H_max=3000.0,
                          **cfg_kw):
    """A rest-state channel on an OceanPartialCellCoordinate (masked_zco-
    style; DINO's kamm cards always use a partial-cell coordinate) with a
    NEMO bottom-drag scheme wired, for the drag-in-matrix / no-double-drag
    tests."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate,
    )

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z0c = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    # Uniform depth (no slope) keeps the analytic bottom-cell check simple;
    # partial-cell-ness only matters for is_bot_u_3d/rate construction.
    H_bathy = jnp.full((n_lat, n_lon), H_max * 0.62)
    z = create_partial_cell_coordinate(z0c, H_bathy)
    state = rest_state_latlon_cgrid_ocean(
        grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy)
    cfg_kw.setdefault("bottom_drag_scheme", "nemo_quadratic")
    cfg_kw.setdefault("bottom_drag_cd0", CD0)
    cfg_kw.setdefault("bottom_drag_cdmax", CDMAX)
    cfg_kw.setdefault("bottom_drag_z0", Z0)
    cfg_kw.setdefault("bottom_drag_ke0", KE0)
    cfg_kw.setdefault("A_v", 0.0)
    cfg_kw.setdefault("K_v", 0.0)
    cfg_kw.setdefault("A_h", 0.0)
    cfg_kw.setdefault("K_h", 0.0)
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    config = LatLonCGridOceanConfig.from_flat(**cfg_kw)
    return grid, z, state, config


def _uniform_flow(state, u0, v0):
    return state._replace(
        u=state.u.replace(data=jnp.full_like(state.u.data, u0)),
        v=state.v.replace(data=jnp.full_like(state.v.data, v0)),
    )


# --------------------------------------------------------------- test 1 ---

def test_default_flags_are_false_and_model_bit_identical():
    """Both new fields default False, and building/stepping a model without
    passing them is IDENTICAL to passing them explicitly False (the flags
    add no new code path when off)."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg_default = LatLonCGridOceanConfig.from_flat()
    assert cfg_default.zdf_drag_in_matrix is False
    assert cfg_default.zdf_baroclinic_only is False
    assert cfg_default.barotropic_drag_substep is False

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config_a = _partial_cell_channel(
        bottom_drag_scheme="legacy", bottom_drag_r=1.0e-3, A_v=1.0e-3,
        K_v=1.0e-4)
    config_b = config_a._replace(
        zdf_drag_in_matrix=False, zdf_baroclinic_only=False,
        barotropic_drag_substep=False)
    state = _uniform_flow(state, 0.1, -0.05)
    model_a = LatLonCGridOceanModel(grid, z, config_a)
    model_b = LatLonCGridOceanModel(grid, z, config_b)
    s_a = model_a.step(state, dt=1800.0)
    s_b = model_b.step(state, dt=1800.0)
    np.testing.assert_array_equal(np.asarray(s_a.u.data), np.asarray(s_b.u.data))
    np.testing.assert_array_equal(np.asarray(s_a.v.data), np.asarray(s_b.v.data))
    np.testing.assert_array_equal(np.asarray(s_a.T.data), np.asarray(s_b.T.data))


# --------------------------------------------------------------- test 2 ---

def test_drag_in_matrix_analytic_bottom_cell():
    """Single-column analytic check (A_v=0): the drag-in-matrix diagonal
    term decouples the tridiagonal system into independent per-level scalar
    equations (K=0 -> a=c=0 everywhere), so

        b_bot = 1 + 2 * dt_mom * r_eff / e3_bot   (extra_diag at the bottom only)
        u_new_bot = u_old_bot / b_bot

    and every level ABOVE the bottom is unchanged (b=1 there).  b_bot is
    DERIVED from NEMO's dynzdf.F90:293-296, NOT from the legoESM code under
    test: ``zwd(iku) -= zDt_2*(rCdU_bot(i+1)+rCdU_bot(i))/e3u(iku)`` is a SUM
    of the two T-point rates (rCdU_bot<=0, so the subtraction ADDS damping);
    ``r_eff = -rCdU_bot`` here is the 0.5-AVERAGE of those same two rates
    (``nemo_bottom_drag_rate_faces``'s shared convention), so reproducing
    NEMO's SUM from the AVERAGE needs the explicit factor of 2:
    b gains ``+2*dt_mom*r_eff/e3u`` at the bottom cell, where
    ``r_eff = Cd*sqrt(u^2+v^2+ke0)`` (nemo_quadratic)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, barotropic_solver="rigid_lid")
    u0, v0 = 0.25, -0.15
    state = _uniform_flow(state, u0, v0)
    dt_mom = 1800.0
    model = LatLonCGridOceanModel(grid, z, config)
    s_new = model._apply_implicit_vertical_mixing(
        state, dt_mom, surface_forcing=None, do_tracers=False,
    )

    # Analytic bottom-cell rate: uniform flow -> the NEMO t-point rate is
    # the SAME at every wet column, so the 2-point face average collapses
    # to that same constant (mirrors test_nemo_bottom_drag.py's uniform-flow
    # construction).
    bl = int(np.asarray(z.bottom_level)[2, 3])
    h_bot = float(np.asarray(z.h_partial)[2, 3, bl])
    r_eff = CD0 * float(np.sqrt(u0 * u0 + v0 * v0 + KE0))
    b_bot = 1.0 + 2.0 * dt_mom * r_eff / h_bot
    expect_u_bot = u0 / b_bot
    expect_v_bot = v0 / b_bot

    u_new = np.asarray(s_new.u.data)
    v_new = np.asarray(s_new.v.data)
    # Interior u-faces, bottom level: matches the analytic decoupled solve.
    np.testing.assert_allclose(
        u_new[2, 2:-1, bl], expect_u_bot, rtol=1e-6)
    np.testing.assert_allclose(
        v_new[2:-2, 3, bl], expect_v_bot, rtol=1e-6)
    # Every level ABOVE the bottom is untouched (A_v=0, b=1 there).
    np.testing.assert_allclose(
        u_new[2, 2:-1, :bl], u0, rtol=1e-10)


def test_drag_in_matrix_rejects_legacy_scheme():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, bottom_drag_scheme="legacy",
        bottom_drag_r=1e-3)
    with pytest.raises(ValueError, match="zdf_drag_in_matrix"):
        LatLonCGridOceanModel(grid, z, config)


def test_drag_in_matrix_requires_implicit_vmix():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, implicit_vertical_mixing=False)
    with pytest.raises(ValueError, match="zdf_drag_in_matrix"):
        LatLonCGridOceanModel(grid, z, config)


def test_drag_in_matrix_rejects_explicit_substep_barotropic():
    """zdf_drag_in_matrix=True with barotropic_solver="explicit_substep"
    and WITHOUT barotropic_drag_substep must still hard-error: that
    combination skips _bc_bottom_drag (single-owner guard) while the
    explicit_substep barotropic loop's ONLY default drag source IS
    _bc_bottom_drag's RHS kick — the barotropic mode would silently run
    undamped unless the NEMO dyn_drg in-subcycle drag
    (barotropic_drag_substep, dynspg_ts.F90:700-706 + 1584-1642) takes
    over."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, barotropic_solver="explicit_substep")
    with pytest.raises(ValueError, match="barotropic_drag_substep"):
        LatLonCGridOceanModel(grid, z, config)


# ----------------------------------------------- barotropic_drag_substep ---
# NEMO dyn_drg (#1226): the split-explicit barotropic drag composition —
# per-substep explicit bottom stress (dynspg_ts.F90:700-706, the DINO-active
# .NOT.ll_wd branch) + the once-per-step pu_RHSi baroclinic-residual
# correction into F_slow (dyn_drg_init, :1584-1642).


def test_barotropic_drag_substep_requires_drag_in_matrix():
    """barotropic_drag_substep without zdf_drag_in_matrix double-counts (the
    default _bc_bottom_drag depth-mean already reaches F_slow) — must
    raise."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        barotropic_drag_substep=True, barotropic_solver="explicit_substep")
    with pytest.raises(ValueError, match="barotropic_drag_substep"):
        LatLonCGridOceanModel(grid, z, config)


def test_barotropic_drag_substep_requires_explicit_substep_solver():
    """barotropic_drag_substep under a non-substep solver is a partial
    mechanism (only the F_slow correction would apply) — must raise."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True, barotropic_solver="rigid_lid")
    with pytest.raises(ValueError, match="explicit_substep"):
        LatLonCGridOceanModel(grid, z, config)


def test_nemo_dino_drag_composition_constructs():
    """The full NEMO-DINO drag composition (ln_drgimp=T + ln_dynspg_ts=T ⇒
    zdf_drag_in_matrix=True + zdf_baroclinic_only=True +
    barotropic_drag_substep=True + explicit_substep) must construct — this
    is exactly the combination the old guard rejected before dyn_drg
    landed.  baroclinic_only is REQUIRED: NEMO removes the barotropic mean
    from the 3-D implicit solve unconditionally in this composition
    (dynzdf.F90:147-159), so the matrix drag acts on the baroclinic
    residual only and the barotropic mode is dragged once, by the substep
    drag."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    LatLonCGridOceanModel(grid, z, config)   # must not raise


def test_barotropic_drag_substep_requires_baroclinic_only():
    """substep drag + matrix drag WITHOUT the barotropic-mean removal
    (zdf_baroclinic_only / nemo_stage_mean_imposition both off) must raise:
    the matrix diagonal would drag the FULL bottom velocity (barotropic
    included) and the substep drag would double-count the barotropic mode
    (adversarial-review finding 1; NEMO's removal at dynzdf.F90:147-159 is
    unconditional under ln_drgimp + ln_dynspg_ts)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    with pytest.raises(ValueError, match="zdf_baroclinic_only"):
        LatLonCGridOceanModel(grid, z, config)


def test_barotropic_drag_substep_analytic_one_substep():
    """One substep, uniform zonal u0, v0=0, flat bottom, Coriolis off, no
    slow forcing: uniform zonal flow is exactly divergence-free on the
    C-grid (equal fluxes on both u-faces of every cell), so eta stays 0 and
    the PGF vanishes — the substep update reduces to the transcribed NEMO
    form alone.  DERIVED from dynspg_ts.F90:701-724 (NOT from the code under
    test): ``zu_trd += zCdU_u*un_e*hur_e`` then ``ua_e = un_e +
    rDt_e*(spg+trd+frc)`` with zCdU_u = -r_eff, hur_e = 1/H_u gives

        U_1 = u0 * (1 - dt * r_eff / H_u),   r_eff = Cd0*sqrt(u0^2 + ke0)

    (uniform flow → the 2-point face average of t-point rates collapses to
    the constant rate; flat bottom → H_u = H_bathy at every face)."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    grid, z, state, config = _partial_cell_channel(
        barotropic_drag_substep=True)
    u0 = 0.2
    state = _uniform_flow(state, u0, 0.0)
    dt_s = 600.0
    s_new, _ = barotropic_substeps_latlon_cgrid(
        state, dt_s, 1, grid, z, config, add_barotropic_coriolis=False)

    H_u = float(np.asarray(state.H_bathy.data)[2, 3])   # flat: 1860 m
    r_eff = CD0 * float(np.sqrt(u0 * u0 + KE0))
    expect = u0 * (1.0 - dt_s * r_eff / H_u)
    assert 0.0 < expect < u0
    u_new = np.asarray(s_new.u.data)
    # rtol 1e-6: the rest state is float32 (matches test 2's tolerance).
    np.testing.assert_allclose(u_new[2, 2:-1, :], expect, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(s_new.v.data), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(s_new.eta.data), 0.0, atol=1e-15)


def test_barotropic_drag_substep_damps_over_substeps():
    """Multiple substeps, drag on, all forcing off: the barotropic velocity
    magnitude must strictly DECREASE (positive-r damping sign gate — a
    flipped sign would amplify), and stay positive (no overshoot at this
    dt*r/H << 1)."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    grid, z, state, config = _partial_cell_channel(
        barotropic_drag_substep=True)
    u0 = 0.2
    state = _uniform_flow(state, u0, 0.0)
    s_new, _ = barotropic_substeps_latlon_cgrid(
        state, 600.0, 8, grid, z, config, add_barotropic_coriolis=False)
    u_new = np.asarray(s_new.u.data)
    assert np.all(u_new[2, 2:-1, :] > 0.0)
    assert np.all(u_new[2, 2:-1, :] < u0)


def test_barotropic_drag_substep_off_is_bit_identical():
    """Flag off (default) vs explicitly False: the substep loop must be
    byte-identical (the drag branch is statically not built)."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    grid, z, state, config = _partial_cell_channel()
    assert config.barotropic_drag_substep is False
    config_b = config._replace(barotropic_drag_substep=False)
    state = _uniform_flow(state, 0.15, -0.05)
    s_a, (hu_a, hv_a) = barotropic_substeps_latlon_cgrid(
        state, 600.0, 4, grid, z, config)
    s_b, (hu_b, hv_b) = barotropic_substeps_latlon_cgrid(
        state, 600.0, 4, grid, z, config_b)
    np.testing.assert_array_equal(np.asarray(s_a.u.data), np.asarray(s_b.u.data))
    np.testing.assert_array_equal(np.asarray(s_a.v.data), np.asarray(s_b.v.data))
    np.testing.assert_array_equal(
        np.asarray(s_a.eta.data), np.asarray(s_b.eta.data))
    np.testing.assert_array_equal(np.asarray(hu_a), np.asarray(hu_b))
    # And flag ON differs (the term is live, not silently dropped).
    s_c, _ = barotropic_substeps_latlon_cgrid(
        state, 600.0, 4, grid, z,
        config._replace(barotropic_drag_substep=True))
    assert not np.array_equal(np.asarray(s_c.u.data), np.asarray(s_a.u.data))


def test_baroclinic_only_plus_drag_in_matrix_bt_correction_damps():
    """Combined zdf_drag_in_matrix + zdf_baroclinic_only, A_v=0, uniform u0
    everywhere (so u_bt_mean == u0 at every column): the BT-drag RHS
    correction (dynzdf.F90:156-159, rCdU_bot<=0 -> the term OPPOSES uu_b) must
    DAMP the bottom-cell velocity in magnitude, not amplify or flip its sign.

    Trace: baroclinic strip zeroes u_solve_in everywhere (uniform column);
    the BT correction then sets the bottom cell to
    ``-2*dt_mom*r_eff/e3u * u0``; the decoupled A_v=0 solve divides that by
    ``b_bot = 1 + 2*dt_mom*r_eff/e3u``; re-adding u_bt_mean=u0 afterward gives

        u_final_bot = u0 - 2*dt_mom*r_eff/e3u*u0 / b_bot = u0 / b_bot

    i.e. the SAME damped form as the drag-in-matrix-alone case (test 2),
    with |u_final_bot| < |u0| since b_bot > 1 (r_eff > 0)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_solver="rigid_lid")
    u0 = 0.2
    state = _uniform_flow(state, u0, 0.0)
    dt_mom = 1800.0
    model = LatLonCGridOceanModel(grid, z, config)
    s_new = model._apply_implicit_vertical_mixing(
        state, dt_mom, surface_forcing=None, do_tracers=False,
    )

    bl = int(np.asarray(z.bottom_level)[2, 3])
    h_bot = float(np.asarray(z.h_partial)[2, 3, bl])
    r_eff = CD0 * float(np.sqrt(u0 * u0 + KE0))
    b_bot = 1.0 + 2.0 * dt_mom * r_eff / h_bot
    expect_u_bot = u0 / b_bot
    assert abs(expect_u_bot) < abs(u0)

    u_new = np.asarray(s_new.u.data)
    np.testing.assert_allclose(u_new[2, 2:-1, bl], expect_u_bot, rtol=1e-6)
    # Above the bottom (A_v=0, no drag there): the baroclinic-only round
    # trip is exact, so those levels are untouched at u0.
    np.testing.assert_allclose(u_new[2, 2:-1, :bl], u0, rtol=1e-6)


# --------------------------------------------------------------- test 3 ---

def test_baroclinic_only_round_trip_conservative_at_zero_Av():
    """A_v=0, no drag -> the implicit solve is the identity, so stripping
    the depth mean, "solving" (a no-op), and re-adding the SAME depth mean
    must round-trip EXACTLY to the input (dynzdf.F90:148-150 "remove
    barotropic velocities" ... re-spliced after dyn_zdf at mlf_baro_corr)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z, state, config = _partial_cell_channel(
        bottom_drag_scheme="legacy", bottom_drag_r=0.0,
        zdf_baroclinic_only=True)
    rng = np.random.default_rng(7)
    u = rng.uniform(-0.3, 0.3, size=state.u.data.shape)
    v = rng.uniform(-0.3, 0.3, size=state.v.data.shape)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u) * state.u_mask.data[..., None]),
        v=state.v.replace(data=jnp.asarray(v) * state.v_mask.data[..., None]),
    )
    model = LatLonCGridOceanModel(grid, z, config)
    s_new = model._apply_implicit_vertical_mixing(
        state, 1800.0, surface_forcing=None, do_tracers=False,
    )
    np.testing.assert_allclose(
        np.asarray(s_new.u.data), np.asarray(state.u.data),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(s_new.v.data), np.asarray(state.v.data),
        rtol=0, atol=1e-9)


# --------------------------------------------------------------- test 4 ---

def test_no_double_drag_outside_application_disabled():
    """With zdf_drag_in_matrix=True, the tendency-stage explicit bottom-drag
    application (_bc_bottom_drag) must be SKIPPED — the baroclinic
    tendencies' isolated bottom-drag diagnostic must be exactly zero, even
    though the SAME bottom_drag_scheme would produce a nonzero rate with the
    flag off (single-owner: drag now lives ONLY in the implicit matrix)."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    grid, z, state, config_off = _partial_cell_channel()
    config_on = config_off._replace(zdf_drag_in_matrix=True)
    state = _uniform_flow(state, 0.25, -0.15)

    _, diag_off = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z, config_off, diagnose_momentum=True)
    _, diag_on = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z, config_on, diagnose_momentum=True)

    # Sanity: with the flag OFF, the existing NEMO explicit branch is
    # active (nonzero bottom-cell drag), matching test_nemo_bottom_drag.py.
    assert float(jnp.max(jnp.abs(diag_off.botdrag_u.data))) > 0.0
    # With the flag ON, the outside application is disabled everywhere.
    np.testing.assert_array_equal(
        np.asarray(diag_on.botdrag_u.data), 0.0)
    np.testing.assert_array_equal(
        np.asarray(diag_on.botdrag_v.data), 0.0)


def test_implicit_solver_extra_diag_default_zero_is_bit_identical():
    """implicit_vertical_diffusion_ocean's new extra_diag kwarg defaults to
    0.0 and must not perturb the diagonal at all (b unchanged)."""
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        build_dz_half, implicit_vertical_diffusion_ocean,
    )
    nlev = 6
    field = jnp.linspace(20.0, 5.0, nlev)
    dz = jnp.full((nlev,), 10.0)
    dz_half = build_dz_half(dz)
    K = jnp.full((nlev - 1,), 1.0e-2)
    out_default = implicit_vertical_diffusion_ocean(field, K, dz, dz_half, dt=3600.0)
    out_explicit_zero = implicit_vertical_diffusion_ocean(
        field, K, dz, dz_half, dt=3600.0, extra_diag=0.0)
    np.testing.assert_array_equal(
        np.asarray(out_default), np.asarray(out_explicit_zero))


def test_implicit_solver_extra_diag_damps_targeted_level():
    """extra_diag > 0 at ONE level (K=0 elsewhere) reproduces the analytic
    backward-Euler damping u_new = u_old / (1 + extra_diag) at exactly that
    level, and leaves every other (undamped) level unchanged."""
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        build_dz_half, implicit_vertical_diffusion_ocean,
    )
    nlev = 5
    field = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dz = jnp.full((nlev,), 10.0)
    dz_half = build_dz_half(dz)
    K = jnp.zeros((nlev - 1,))
    extra_diag = jnp.array([0.0, 0.0, 0.0, 0.0, 0.4])
    out = implicit_vertical_diffusion_ocean(
        field, K, dz, dz_half, dt=600.0, extra_diag=extra_diag)
    expect = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0 / 1.4])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expect), rtol=1e-12)


# --------------------------------------------- #1455 drag-rate time level ---

def test_barotropic_drag_rate_uses_u_now_time_level():
    """The barotropic bottom-drag RATE must be built from the NOW-level
    (NEMO ``Kmm``) velocity handed in as ``u_now``/``v_now``, not from the
    velocity carried by ``state``.

    NEMO evaluates ``rCdU_bot`` in ``zdf_drg_nonlin`` from ``uu(ji,jj,imk,Kmm)``
    (zdfdrg.F90:174-181), inside ``zdf_phy`` at stpmlf.F90:190 — BEFORE
    ``dyn_adv``/``dyn_vor``/``dyn_ldf``/``dyn_hpg``/``dyn_spg`` — and
    ``dyn_drg_init`` (dynspg_ts.F90:1616) freezes that same array across the
    substep window.  The production caller hands this solver a POST-momentum
    ``state`` (u* = u^n + dt·RHS + the Matsuno rotation), so the now level has
    to arrive separately.

    Same analytic setup and same DERIVED form as
    ``test_barotropic_drag_substep_analytic_one_substep`` (uniform zonal flow
    is divergence-free on the C-grid ⇒ eta and the PGF stay zero), except that
    the drag velocity and the integrated velocity are now DIFFERENT:

        U_1 = u_state * (1 - dt * r_eff(u_drag) / H_u),
        r_eff(u) = Cd0 * sqrt(u^2 + ke0)
    """
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    # Shallow column (H_u = 124 m): dt*r/H is then O(1e-2), so the two
    # candidate predictions are separated by ~4 orders of magnitude more than
    # the 1e-6 tolerance asserted below.
    grid, z, state, config = _partial_cell_channel(
        H_max=200.0, barotropic_drag_substep=True)
    u_state, u_drag = 0.2, 1.4          # r_eff differs by ~7x between them
    state = _uniform_flow(state, u_state, 0.0)
    now = _uniform_flow(state, u_drag, 0.0)
    dt_s = 600.0
    s_new, _ = barotropic_substeps_latlon_cgrid(
        state, dt_s, 1, grid, z, config, add_barotropic_coriolis=False,
        u_now=now.u.data, v_now=now.v.data)

    H_u = float(np.asarray(state.H_bathy.data)[2, 3])   # flat: 124 m
    r_drag = CD0 * float(np.sqrt(u_drag * u_drag + KE0))
    r_state = CD0 * float(np.sqrt(u_state * u_state + KE0))
    expect = u_state * (1.0 - dt_s * r_drag / H_u)
    wrong = u_state * (1.0 - dt_s * r_state / H_u)
    # The two predictions must be separable at the tolerance asserted below,
    # or the test could not fail (a control on the control).
    assert abs(expect - wrong) / abs(expect) > 1e-3
    u_new = np.asarray(s_new.u.data)
    np.testing.assert_allclose(u_new[2, 2:-1, :], expect, rtol=1e-6)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(u_new[2, 2:-1, :], wrong, rtol=1e-6)


def test_barotropic_drag_rate_u_now_default_is_state_velocity():
    """Omitting ``u_now``/``v_now`` is byte-identical to passing ``state``'s
    own velocity — the default keeps every direct caller (which hands this
    solver the un-advanced NOW state) on exactly its previous numbers."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    grid, z, state, config = _partial_cell_channel(
        barotropic_drag_substep=True)
    state = _uniform_flow(state, 0.15, -0.05)
    kw = dict(add_barotropic_coriolis=False)
    s_a, (hu_a, _) = barotropic_substeps_latlon_cgrid(
        state, 600.0, 4, grid, z, config, **kw)
    s_b, (hu_b, _) = barotropic_substeps_latlon_cgrid(
        state, 600.0, 4, grid, z, config,
        u_now=state.u.data, v_now=state.v.data, **kw)
    np.testing.assert_array_equal(np.asarray(s_a.u.data),
                                  np.asarray(s_b.u.data))
    np.testing.assert_array_equal(np.asarray(s_a.v.data),
                                  np.asarray(s_b.v.data))
    np.testing.assert_array_equal(np.asarray(hu_a), np.asarray(hu_b))
