"""Unit tests for the Veros-faithful prognostic-TKE SUPERBEE ADVECTION option
(``TKEConfig.advection_scheme``; Veros ``enable_tke_superbee_advection``,
veros/core/tke.py:286-323 + veros/core/advection.py:117-244).

Covers:

1.  dispatch hardening — unknown scheme raises at model construction;
    advection with a DIAGNOSTIC TKE (prognostic=False) raises; the runtime
    dispatch raises on an unknown literal too;
2.  default-off bit-identity — advection_scheme="none" (default) leaves a
    multi-step prognostic-TKE run bitwise identical to the explicit-"none"
    config, and never creates state.dtke;
3.  conservation — the volume integral of the advective tendency over the
    closed/periodic domain vanishes to machine precision;
4.  surface/bottom special-cell handling — a CONSTANT TKE field has exactly
    zero advective tendency at every interface except the topmost, which
    absorbs the W-grid column-divergence residual (the role Veros's surface
    half-cell plays); the bottom interface is Veros-exact (zero flux below);
5.  the superbee flux formula — one hand-computed Veros ``_adv_superbee``
    face-flux check (non-vacuous reference);
6.  AB2-on-tendency carry — the FIRST step applies dt·(1.5+eps)·dtke^n
    (dtke^{n-1}=0, Veros's zero-initialised dtke[taum1]); the carried
    state.dtke equals the tendency of the advected tke[tau]; the advection
    increment touches ONLY tke (T/S/u/v bitwise unchanged in the step);
7.  differentiability — jax.grad through 2 advecting steps is finite+nonzero;
8.  scan-compat — integrate_scan seeds dtke; the carry pytree is stable.

Fixtures mirror tests/ocean/unit/test_tke_prognostic.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.advection import (
    adv_flux_superbee_wgrid_latlon_cgrid,
    wgrid_advection_tendency_latlon_cgrid,
    wgrid_velocities_latlon_cgrid,
    wgrid_vertical_velocity_latlon_cgrid,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _fp64_policy():
    """fp64 storage policy so the x64 state survives the model's storage cast
    (mirrors test_tke_prognostic.py)."""
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


N_LAT, N_LON, NLEV = 6, 12, 8
H_MAX = 4000.0


def _grid_z():
    return (create_latlon_grid(n_lat=N_LAT, n_lon=N_LON),
            create_ocean_z_star(n_levels=NLEV, H_max=H_MAX))


def _perturbed_state(grid, z_coord, seed=0):
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=H_MAX,
    )
    rng = np.random.default_rng(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.2 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.2 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    T_prof = 5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))
    T = T_prof[None, None, :] + 0.2 * rng.standard_normal((n_lat, n_lon, nlev))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u) * s.u_mask.data[:, :, None]),
        v=s.v.replace(data=jnp.asarray(v) * s.v_mask.data[:, :, None]),
        T=s.T.replace(data=jnp.asarray(T)),
    )


def _physics(tke_cfg):
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke_cfg),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def _model(tke_cfg, **cfg_kwargs):
    grid, z_coord = _grid_z()
    cfg_kwargs.setdefault("implicit_vertical_mixing", True)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4,
        K_v=0.0,
        physics=_physics(tke_cfg),
        **cfg_kwargs,
    )
    return LatLonCGridOceanModel(grid, z_coord, cfg), grid, z_coord


def _seed_tke_random(state, z_coord, seed=7, base=1.0e-3):
    """Seed a SPATIALLY-VARYING positive TKE so the advective tendency is
    nontrivial (a uniform field advects to ~zero tendency)."""
    rng = np.random.default_rng(seed)
    lm = np.asarray(state.land_mask.data)
    nlev = z_coord.n_levels
    tke0 = base * (1.0 + 0.5 * rng.random((lm.shape[0], lm.shape[1], nlev - 1)))
    tke0 = jnp.asarray(tke0 * (lm[:, :, None] > 0.5),
                       dtype=state.T.data.dtype)
    return state._replace(
        tke=Field(data=tke0, name="tke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"))


def _adv_inputs(seed=3):
    """Common raw inputs for operator-level tests."""
    grid, z_coord = _grid_z()
    s = _perturbed_state(grid, z_coord, seed=seed)
    s = _seed_tke_random(s, z_coord, seed=seed + 1)
    dz_ref = jnp.asarray(z_coord.dz_ref)
    return grid, z_coord, s, dz_ref


def _dzw(dz_ref):
    return 0.5 * (dz_ref[:-1] + dz_ref[1:])


# ---------------------------------------------------------------------------
# (1) dispatch hardening
# ---------------------------------------------------------------------------


def test_unknown_scheme_raises_at_construction():
    with pytest.raises(ValueError, match="advection_scheme"):
        _model(TKEConfig(prognostic=True, advection_scheme="superbe"))


def test_advection_without_prognostic_raises():
    with pytest.raises(ValueError, match="prognostic"):
        _model(TKEConfig(prognostic=False, advection_scheme="superbee"))


def test_advection_without_implicit_vmix_raises():
    """Without the implicit vertical-mixing solve the prognostic TKE never
    advances, so the advection would be a SILENT no-op — must fail fast."""
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        _model(TKEConfig(prognostic=True, advection_scheme="superbee"),
               implicit_vertical_mixing=False)


def test_runtime_dispatch_raises_on_unknown():
    """The dispatch site itself raises (tripwire independent of the
    construction gate)."""
    model, grid, z_coord = _model(
        TKEConfig(prognostic=True, advection_scheme="superbee"))
    s = _seed_tke_random(_perturbed_state(grid, z_coord), z_coord)
    # Forge an invalid literal past the construction gate.
    bad_tke = model.config.physics.vertical_mixing.tke._replace(
        advection_scheme="bogus")
    bad_vm = model.config.physics.vertical_mixing._replace(tke=bad_tke)
    model.config = model.config._replace(
        physics=model.config.physics._replace(vertical_mixing=bad_vm))
    with pytest.raises(ValueError, match="bogus"):
        model._apply_tke_advection(s, s.tke.data, 3600.0)


# ---------------------------------------------------------------------------
# (2) default-off bit-identity
# ---------------------------------------------------------------------------


def test_default_off_bitwise_identical_multistep():
    """A 3-step prognostic-TKE run with the DEFAULT TKEConfig (advection field
    untouched) is bitwise identical to an explicit advection_scheme='none'
    run, and neither creates state.dtke. (The byte-identity of the default
    against the PRE-CHANGE code is verified out-of-band against a pristine
    HEAD worktree — see the PR validation notes; this test locks the
    none==default equivalence and the inert dtke carry.)"""
    grid, z_coord = _grid_z()
    m_default, _, _ = _model(TKEConfig(prognostic=True))
    m_none, _, _ = _model(TKEConfig(prognostic=True, advection_scheme="none"))
    s0 = _seed_tke_random(_perturbed_state(grid, z_coord), z_coord)
    sa, sb = s0, s0
    for _ in range(3):
        sa = m_default._step_impl(sa, 3600.0)
        sb = m_none._step_impl(sb, 3600.0)
        assert sa.dtke is None and sb.dtke is None
    for name in ("tke", "T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(sa, name).data),
            np.asarray(getattr(sb, name).data), err_msg=name)
    np.testing.assert_array_equal(np.asarray(sa.u.data), np.asarray(sb.u.data))
    np.testing.assert_array_equal(np.asarray(sa.v.data), np.asarray(sb.v.data))


# ---------------------------------------------------------------------------
# (3) conservation of the advective tendency
# ---------------------------------------------------------------------------


def test_advective_tendency_conserves_volume_integral():
    """Sum over the wet volume of the advective tendency = 0 to machine
    precision: the horizontal divergence is flux-form (periodic lon, zero-flux
    walls) and the vertical fluxes are zero at both column ends."""
    grid, z_coord, s, dz_ref = _adv_inputs()
    dtke = wgrid_advection_tendency_latlon_cgrid(
        s.tke.data, s.u.data, s.v.data, grid, dz_ref, 3600.0,
        s.land_mask.data, s.u_mask.data, s.v_mask.data)
    dzw = _dzw(dz_ref)
    vol = grid.area[:, :, None] * dzw  # (n_lat, n_lon, M)
    total = float(jnp.sum(dtke * vol))
    scale = float(jnp.sum(jnp.abs(dtke) * vol))
    assert scale > 0.0, "vacuous: tendency is identically zero"
    assert abs(total) < 1e-12 * scale


# ---------------------------------------------------------------------------
# (4) surface/bottom special-cell handling (constant-field column residual)
# ---------------------------------------------------------------------------


def test_constant_field_residual_lands_on_top_interface_only():
    """For a CONSTANT TKE field the superbee fluxes reduce to c·(velocity)
    (the limited anti-diffusion vanishes), so the tendency is exactly
    -c·(total W-cell divergence)/dzw — zero at every interface where the
    continuity-built w closes the budget (interfaces 1..M-1, including the
    Veros-exact bottom cell with zero flux below), with the column residual
    absorbed at interface 0 (the role of Veros's surface half-cell)."""
    grid, z_coord, s, dz_ref = _adv_inputs(seed=11)
    M = NLEV - 1
    c = 2.5e-3
    E_const = jnp.full((N_LAT, N_LON, M), c, dtype=s.T.data.dtype)
    dtke = wgrid_advection_tendency_latlon_cgrid(
        E_const, s.u.data, s.v.data, grid, dz_ref, 3600.0,
        s.land_mask.data, s.u_mask.data, s.v_mask.data)
    dtke = np.asarray(dtke)
    # Interfaces 1..M-1: exactly closed (machine precision).
    interior_scale = float(np.max(np.abs(dtke))) + 1e-30
    assert np.max(np.abs(dtke[:, :, 1:])) < 1e-12 * max(interior_scale, 1.0)
    # Interface 0 absorbs the column residual -c·Σ dzw·div / dzw[0]:
    # cross-check against the SAME public W-grid velocity helpers.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
    u_w, v_w = wgrid_velocities_latlon_cgrid(
        s.u.data, s.v.data, dz_ref, s.u_mask.data, s.v_mask.data)
    div = divergence_cgrid(u_w, v_w, grid)
    dzw = _dzw(jnp.asarray(dz_ref))
    expected_top = np.asarray(
        -c * jnp.sum(div * dzw, axis=-1) / dzw[0]
        * s.land_mask.data)
    np.testing.assert_allclose(dtke[:, :, 0], expected_top,
                               rtol=1e-10, atol=1e-22)
    assert np.any(np.abs(expected_top) > 0.0)  # non-vacuous


def test_wgrid_velocity_weights():
    """Interior interfaces carry the dz-weighted 2-cell average (= the plain
    average on this uniform grid); the deepest interface absorbs the bottom
    cell's lower half (Veros's bottom redirect)."""
    grid, z_coord, s, dz_ref = _adv_inputs(seed=5)
    u_w, v_w = wgrid_velocities_latlon_cgrid(
        s.u.data, s.v.data, dz_ref, s.u_mask.data, s.v_mask.data)
    dz = np.asarray(dz_ref)
    dzw = 0.5 * (dz[:-1] + dz[1:])
    u = np.asarray(s.u.data)
    um = np.asarray(s.u_mask.data)[:, :, None]
    exp_int = ((u[..., :-2] * 0.5 * dz[:-2] + u[..., 1:-1] * 0.5 * dz[1:-1])
               / dzw[:-1]) * um
    # Compare the n_lon owned faces; the wrap column (face n_lon) is
    # REBUILT from face 0 by the operator (periodic consistency).
    np.testing.assert_allclose(np.asarray(u_w)[:, :N_LON, :-1],
                               exp_int[:, :N_LON], rtol=1e-12)
    exp_bot = ((u[..., -2] * 0.5 * dz[-2] + u[..., -1] * dz[-1])
               / dzw[-1]) * um[..., 0]
    np.testing.assert_allclose(np.asarray(u_w)[:, :N_LON, -1],
                               exp_bot[:, :N_LON], rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(u_w)[:, -1],
                                  np.asarray(u_w)[:, 0])


# ---------------------------------------------------------------------------
# (5) Veros superbee flux formula — hand-computed reference
# ---------------------------------------------------------------------------


def test_vertical_superbee_flux_matches_hand_formula():
    """One vertical face flux against the literal Veros ``_adv_superbee``
    expression F = w·(E_above+E_below)/2 − |w|·((1−cr)+uCFL·cr)·rj/2 with
    cr = superbee(rjm/rj) for upward flow (donor below)."""
    grid, z_coord, s, dz_ref = _adv_inputs(seed=13)
    M = NLEV - 1
    rng = np.random.default_rng(2)
    E = jnp.asarray(1e-3 * (1.0 + rng.random((N_LAT, N_LON, M))))
    w_w = jnp.asarray(1e-4 * rng.standard_normal((N_LAT, N_LON, M - 1)))
    lm = jnp.ones((N_LAT, N_LON))
    um = jnp.ones((N_LAT, N_LON + 1))
    vm = jnp.zeros((N_LAT + 1, N_LON)).at[1:-1].set(1.0)
    dt = 3600.0
    _, _, ft = adv_flux_superbee_wgrid_latlon_cgrid(
        E, jnp.zeros((N_LAT, N_LON + 1, M)), jnp.zeros((N_LAT + 1, N_LON, M)),
        w_w, grid, dz_ref, dt, lm, um, vm)
    # Hand evaluation at one interior flux position j (between interfaces j
    # above and j+1 below), one column.
    j, iy, ix = 2, 3, 5
    En = np.asarray(E)[iy, ix]
    w = float(np.asarray(w_w)[iy, ix, j])
    dz = np.asarray(dz_ref)
    dzw = 0.5 * (dz[:-1] + dz[1:])
    var_0, var_1 = En[j + 1], En[j]          # below (donor for w>0), above
    var_m1, var_2 = En[j + 2], En[j - 1]
    rj = var_1 - var_0
    rjm = var_0 - var_m1
    rjp = var_2 - var_1
    eps = 1e-20
    r = (rjm if w > 0 else rjp) / (rj if abs(rj) >= eps else eps)
    cr = max(0.0, min(1.0, 2.0 * r), min(2.0, r))
    ucfl = abs(w) * dt / dzw[j + 1]
    expected = (w * (var_1 + var_0) * 0.5
                - abs(w) * ((1.0 - cr) + ucfl * cr) * rj * 0.5)
    np.testing.assert_allclose(float(np.asarray(ft)[iy, ix, j]), expected,
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# (6) AB2 carry behaviour
# ---------------------------------------------------------------------------


def test_first_step_forward_weighted_and_local_to_tke():
    """First advecting step: tke_on − tke_off = dt·(1.5+eps)·dtke^n with
    dtke^{n-1}=0 (Veros's zero-initialised dtke[taum1]); state.dtke carries
    dtke^n; and the advection touches NOTHING but tke."""
    grid, z_coord = _grid_z()
    m_on, _, _ = _model(TKEConfig(prognostic=True, advection_scheme="superbee"))
    m_off, _, _ = _model(TKEConfig(prognostic=True))
    s0 = _seed_tke_random(_perturbed_state(grid, z_coord), z_coord)
    dt = 3600.0
    s_on = m_on._step_impl(s0, dt)
    s_off = m_off._step_impl(s0, dt)

    dtke_ref = wgrid_advection_tendency_latlon_cgrid(
        s0.tke.data, s0.u.data, s0.v.data, grid,
        jnp.asarray(z_coord.dz_ref), dt,
        s0.land_mask.data, s0.u_mask.data, s0.v_mask.data)
    eps = m_on.config.ab2_epsilon
    expected_incr = dt * (1.5 + eps) * np.asarray(dtke_ref)
    actual_incr = np.asarray(s_on.tke.data) - np.asarray(s_off.tke.data)
    np.testing.assert_allclose(actual_incr, expected_incr,
                               rtol=1e-9, atol=1e-20)
    assert np.any(np.abs(expected_incr) > 0.0)  # non-vacuous

    # The carry equals the just-computed tendency.
    assert s_on.dtke is not None
    np.testing.assert_allclose(np.asarray(s_on.dtke.data),
                               np.asarray(dtke_ref), rtol=1e-12, atol=0)
    # Locality: only tke (and the dtke carry) differ.
    for name in ("T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(s_on, name).data),
            np.asarray(getattr(s_off, name).data), err_msg=name)
    np.testing.assert_array_equal(np.asarray(s_on.u.data),
                                  np.asarray(s_off.u.data))
    np.testing.assert_array_equal(np.asarray(s_on.v.data),
                                  np.asarray(s_off.v.data))


def test_second_step_uses_carried_dtke():
    """Step 2's increment uses BOTH AB2 weights: reconstruct it from the
    carried dtke^1 and the recomputed dtke^2 of the step-2 inputs."""
    grid, z_coord = _grid_z()
    m_on, _, _ = _model(TKEConfig(prognostic=True, advection_scheme="superbee"))
    s0 = _seed_tke_random(_perturbed_state(grid, z_coord), z_coord)
    dt = 3600.0
    s1 = m_on._step_impl(s0, dt)
    s2 = m_on._step_impl(s1, dt)
    # dtke^2 from step 2's tau-state (s1's tke, u, v) via the public operator.
    dtke2_ref = wgrid_advection_tendency_latlon_cgrid(
        s1.tke.data, s1.u.data, s1.v.data, grid,
        jnp.asarray(z_coord.dz_ref), dt,
        s1.land_mask.data, s1.u_mask.data, s1.v_mask.data)
    np.testing.assert_allclose(np.asarray(s2.dtke.data),
                               np.asarray(dtke2_ref), rtol=1e-12, atol=0)
    # And the AB2 history actually changed between steps (genuine carry).
    assert not np.allclose(np.asarray(s2.dtke.data), np.asarray(s1.dtke.data))


def test_ab2_outer_integrator_path_advects():
    """The faithful-AB2 outer path threads the advection too (the second
    store-back site)."""
    grid, z_coord = _grid_z()
    m_on, _, _ = _model(
        TKEConfig(prognostic=True, advection_scheme="superbee"),
        outer_integrator="ab2")
    m_off, _, _ = _model(TKEConfig(prognostic=True), outer_integrator="ab2")
    s0 = _seed_tke_random(_perturbed_state(grid, z_coord), z_coord)
    s_on = m_on.step(s0, 3600.0)
    s_off = m_off.step(s0, 3600.0)
    assert s_on.dtke is not None and s_off.dtke is None
    assert np.max(np.abs(np.asarray(s_on.tke.data)
                         - np.asarray(s_off.tke.data))) > 0.0
    assert np.all(np.isfinite(np.asarray(s_on.tke.data)))


# ---------------------------------------------------------------------------
# (7) differentiability
# ---------------------------------------------------------------------------


def test_grad_finite_through_two_advecting_steps():
    grid, z_coord = _grid_z()
    model, _, _ = _model(
        TKEConfig(prognostic=True, advection_scheme="superbee"))
    s0 = _seed_tke_random(_perturbed_state(grid, z_coord), z_coord)
    dt = 3600.0

    def loss(u_data):
        s = s0._replace(u=s0.u.replace(data=u_data))
        s = model._step_impl(s, dt)
        s = model._step_impl(s, dt)
        return jnp.sum(s.tke.data ** 2)

    g = jax.grad(loss)(s0.u.data)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.abs(np.asarray(g)) > 0.0)


# ---------------------------------------------------------------------------
# (8) scan-compat
# ---------------------------------------------------------------------------


def test_scan_four_steps_constant_pytree_with_advection():
    """integrate_scan pre-seeds state.dtke (zero AB2 history) so the lax.scan
    carry pytree is constant; tke + dtke trajectories are stacked."""
    grid, z_coord = _grid_z()
    model, _, _ = _model(
        TKEConfig(prognostic=True, advection_scheme="superbee"))
    s0 = _perturbed_state(grid, z_coord)  # neither tke nor dtke pre-seeded
    final, traj = model.integrate_scan(s0, dt=3600.0, n_steps=4)
    assert final.tke is not None and final.dtke is not None
    assert np.all(np.isfinite(np.asarray(final.tke.data)))
    assert np.all(np.isfinite(np.asarray(final.dtke.data)))
    assert traj.tke.data.shape[0] == 4
    assert traj.dtke.data.shape[0] == 4
    M = NLEV - 1
    assert final.dtke.data.shape == (N_LAT, N_LON, M)


def test_tripolar_grid_rejected():
    """The W-grid superbee fluxes have no fold-aware meridional stencil; a
    tripolar grid must fail fast, not silently mis-advect across the fold."""
    grid, z_coord, s, dz_ref = _adv_inputs(seed=23)

    class _ActiveFold:
        is_active = True

    class _FakeTripolar:
        """Duck-typed grid carrying an active ``fold`` (the ``is_tripolar``
        sentinel) — the guard must raise before any metric is touched."""
        fold = _ActiveFold()

        def __getattr__(self, name):  # pragma: no cover - guard fires first
            raise AssertionError(f"metric {name!r} touched before the guard")

    from legoesm.ocean.dynamics.latlon_cgrid_operators import is_tripolar
    fake = _FakeTripolar()
    if not is_tripolar(fake):
        pytest.skip("is_tripolar sentinel changed; update the fake")
    with pytest.raises(NotImplementedError, match="tripolar"):
        adv_flux_superbee_wgrid_latlon_cgrid(
            s.tke.data, jnp.zeros((N_LAT, N_LON + 1, NLEV - 1)),
            jnp.zeros((N_LAT + 1, N_LON, NLEV - 1)),
            jnp.zeros((N_LAT, N_LON, NLEV - 2)),
            fake, dz_ref, 3600.0,
            s.land_mask.data, s.u_mask.data, s.v_mask.data)


def test_continuity_w_shape_and_finiteness():
    """Direct exercise of the continuity-built W-grid vertical velocity."""
    grid, z_coord, s, dz_ref = _adv_inputs(seed=17)
    u_w, v_w = wgrid_velocities_latlon_cgrid(
        s.u.data, s.v.data, dz_ref, s.u_mask.data, s.v_mask.data)
    w_w = wgrid_vertical_velocity_latlon_cgrid(u_w, v_w, dz_ref, grid)
    assert w_w.shape == (N_LAT, N_LON, NLEV - 2)
    assert np.all(np.isfinite(np.asarray(w_w)))
    assert np.any(np.abs(np.asarray(w_w)) > 0.0)
