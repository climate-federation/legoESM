"""IMPLICIT surface-forcing placement (``surface_forcing_implicit``).

Veros applies the surface TRACER forcing (``forc_temp_surface``/``forc_salt_surface``
restoring + prescribed q_net + penetrating shortwave) IMPLICITLY — it enters the
backward-Euler vertical-mixing tridiagonal RHS at weight 1.0
(``veros/core/thermodynamics.py``), NOT as an AB2-extrapolated explicit tendency.
legoESM applies it EXPLICITLY by default (summed into ``dT_dt``/``dS_dt``), which the
faithful AB2 outer integrator (#44) over-applies 1.6×.

``LatLonCGridOceanConfig.surface_forcing_implicit=True`` withholds the surface TRACER
forcing from the explicit ``dT_dt``/``dS_dt`` and instead adds ``dt_tracer·S_surf`` to
the T/S solve INPUT inside ``_apply_implicit_vertical_mixing`` (weight 1.0), realising
the backward-Euler RHS-source identity ``(I − dt·L)·X_new = X_old + dt·S_surf``. WIND
STRESS (→ du_dt/dv_dt) is unaffected — it is explicit/AB2'd in BOTH models.

Gates exercised here (the task's gate 4):
  (a) default-off (``surface_forcing_implicit=False``) is bit-identical to the explicit
      path — covered separately by the before/after byte probe; here we assert the
      tendency builder leaves ``surface_tracer_forcing=None`` and the field unchanged;
  (b) the implicit option runs finite AND the surface forcing IS applied (surface T
      relaxes toward T*);
  (c) ``surface_forcing_implicit=True`` + ``implicit_vertical_mixing=False`` raises;
  (d) jax.grad is finite through a step with the implicit option.

Run in fp64 + on CPU for determinism.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_N_LAT, _N_LON = 8, 16
_DT = 3600.0
# Short restoring timescale so the surface relaxation is clear over the few-day
# test horizon (the PLACEMENT is what is under test, not a specific timescale).
_TAU_T_DAYS = 5.0
_SECONDS_PER_DAY = 86400.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _restoring_physics(t_star_eq=25.0, t_star_pole=2.0):
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig, RestoringConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    restoring = RestoringConfig(
        tau_T=_TAU_T_DAYS * _SECONDS_PER_DAY, tau_S=1.0e30,
        T_star_eq=t_star_eq, T_star_pole=t_star_pole, T_profile="cosine",
        implicit=False,
    )
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="restoring",
                                             restoring=restoring),
    )


def _basin(*, surface_forcing_implicit, outer_integrator="ab2",
           implicit_vertical_mixing=True, barotropic_solver="rigid_lid",
           dt_mom_ratio=1.0):
    """Closed flat-bottom channel (land walls N/S, periodic x) with a warm SST
    perturbation so the restoring (T* cosine) drives a clear surface cooling."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((_N_LAT, _N_LON), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(lm),
        H_bathy_override=jnp.asarray(Hb))
    # Surface much WARMER than T* so the restoring cools the surface layer.
    T = np.array(state.T.data)
    T[:, :, 0] = 40.0
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3,
        implicit_vertical_mixing=implicit_vertical_mixing,
        enable_runtime_checks=False, barotropic_solver=barotropic_solver,
        outer_integrator=outer_integrator, dt_mom_ratio=dt_mom_ratio,
        surface_forcing_implicit=surface_forcing_implicit,
        physics=_restoring_physics(),
    )
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


# --------------------------------------------------------------------------
# (a) Default-off — tendency builder leaves the new field None, field unchanged
# --------------------------------------------------------------------------

def test_default_off_leaves_surface_tracer_forcing_none():
    state, model = _basin(surface_forcing_implicit=False)
    tend = model.tendencies(state, surface_forcing=None, dt=_DT)
    assert tend.surface_tracer_forcing is None
    # The restoring is still present in the EXPLICIT dT_dt (summed via physics).
    dT = np.asarray(tend.dT_dt.data)
    assert np.any(np.abs(dT[:, :, 0]) > 0.0), (
        "default-off: restoring must still be in the explicit surface dT_dt")


def test_implicit_on_withholds_restoring_from_explicit_dT_dt():
    """With the flag on, the restoring is WITHHELD from the explicit dT_dt and
    routed onto surface_tracer_forcing instead."""
    state, model = _basin(surface_forcing_implicit=True)
    tend = model.tendencies(state, surface_forcing=None, dt=_DT)
    assert tend.surface_tracer_forcing is not None
    dT_expl = np.asarray(tend.dT_dt.data)
    # The explicit surface tendency must NO LONGER carry the restoring (the only
    # tracer forcing here) — the surface layer explicit dT_dt is ~0 (vmix is
    # apply_diffusion=False under implicit_vertical_mixing).
    assert np.max(np.abs(dT_expl[:, :, 0])) < 1.0e-12, (
        "implicit-on: restoring must be withheld from the explicit surface dT_dt")
    # The restoring RATE now lives on surface_tracer_forcing (surface only).
    dT_surf = np.asarray(tend.surface_tracer_forcing.dT_dt.data)
    assert np.any(np.abs(dT_surf[:, :, 0]) > 0.0)
    assert np.max(np.abs(dT_surf[:, :, 1:])) == 0.0, (
        "restoring is a surface-layer source only")


def test_implicit_matches_explicit_restoring_rate():
    """The routed surface_tracer_forcing rate equals the explicit restoring rate
    (same numerics, just a different placement)."""
    st_e, m_e = _basin(surface_forcing_implicit=False)
    st_i, m_i = _basin(surface_forcing_implicit=True)
    tend_e = m_e.tendencies(st_e, surface_forcing=None, dt=_DT)
    tend_i = m_i.tendencies(st_i, surface_forcing=None, dt=_DT)
    rate_explicit = np.asarray(tend_e.dT_dt.data)[:, :, 0]
    rate_implicit = np.asarray(tend_i.surface_tracer_forcing.dT_dt.data)[:, :, 0]
    np.testing.assert_allclose(rate_implicit, rate_explicit, rtol=0, atol=0)


# --------------------------------------------------------------------------
# (b) Runs finite + the surface forcing IS applied (surface T relaxes to T*)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("outer_integrator", ["forward_euler", "ab2"])
def test_implicit_runs_finite_and_relaxes_surface(outer_integrator):
    state, model = _basin(surface_forcing_implicit=True,
                          outer_integrator=outer_integrator)
    # Pre-build the rigid-lid island data on the concrete state (host-side
    # flood-fill cannot run inside the jitted step).
    model._ensure_rigid_lid_data(state)
    lm = np.asarray(state.land_mask.data).astype(bool)
    T0 = np.asarray(state.T.data)[lm, 0]
    s = state
    for _ in range(48):
        s = model.step(s, dt=_DT, surface_forcing=None)
    jax.block_until_ready(s.T.data)
    assert bool(jnp.all(jnp.isfinite(s.T.data)))
    assert bool(jnp.all(jnp.isfinite(s.S.data)))
    assert bool(jnp.all(jnp.isfinite(s.u.data)))
    T1 = np.asarray(s.T.data)[lm, 0]
    # Surface starts at 40 °C, far above T* (<=25) — implicit restoring must cool it.
    assert np.mean(T1) < np.mean(T0) - 1.0, (
        f"implicit restoring did not cool the surface: {np.mean(T0):.2f} -> "
        f"{np.mean(T1):.2f}")


def test_implicit_vs_explicit_forward_euler_close():
    """Under forward_euler (no AB2 extrapolation) the implicit and explicit
    placements should give a SIMILAR (not identical) surface relaxation — both
    converge toward T*; the implicit form is the backward-Euler τ→τ+dt variant,
    so it relaxes slightly slower per step but to the same target. We just check
    both move the surface toward T* and stay finite/ordered."""
    st_e, m_e = _basin(surface_forcing_implicit=False,
                       outer_integrator="forward_euler")
    st_i, m_i = _basin(surface_forcing_implicit=True,
                       outer_integrator="forward_euler")
    m_e._ensure_rigid_lid_data(st_e)
    m_i._ensure_rigid_lid_data(st_i)
    lm = np.asarray(st_e.land_mask.data).astype(bool)
    for _ in range(48):
        st_e = m_e.step(st_e, dt=_DT, surface_forcing=None)
        st_i = m_i.step(st_i, dt=_DT, surface_forcing=None)
    Te = np.asarray(st_e.T.data)[lm, 0].mean()
    Ti = np.asarray(st_i.T.data)[lm, 0].mean()
    # Both cooled measurably below the 40 °C start, toward T* (<=25).
    assert Te < 38.0 and Ti < 38.0
    # The two placements agree to within a few degrees over the run.
    assert abs(Te - Ti) < 5.0


# --------------------------------------------------------------------------
# (b') Conservation: closed basin, NO surface forcing → new path is a no-op
# --------------------------------------------------------------------------

def _tracer_content(model, state, field):
    from legoesm.ocean.vertical import compute_layer_thickness
    h_k = np.asarray(compute_layer_thickness(
        state.eta.data, state.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m))
    area = np.asarray(model.grid.area)[:, :, None]
    lm = np.asarray(state.land_mask.data)[:, :, None]
    return float(np.sum(np.asarray(field) * h_k * area * lm, dtype=np.float64))


def test_no_forcing_conserves_tracer_mass():
    """surface_forcing_implicit=True + NO surface forcing (no physics, no wind)
    under rigid-lid AB2 dt_mom_ratio=9: the new path is a no-op and tracer mass
    is conserved to f64."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((_N_LAT, _N_LON), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(lm),
        H_bathy_override=jnp.asarray(Hb))
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        enable_runtime_checks=False, barotropic_solver="rigid_lid",
        outer_integrator="ab2", dt_mom_ratio=9.0,
        surface_forcing_implicit=True,  # ON, but NO surface forcing present
        physics=None,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    model._ensure_rigid_lid_data(state)
    heat0 = _tracer_content(model, state, state.T.data)
    salt0 = _tracer_content(model, state, state.S.data)
    s = state
    for _ in range(30):
        s = model.step(s, dt=_DT, surface_forcing=None)
    jax.block_until_ready(s.T.data)
    heat1 = _tracer_content(model, s, s.T.data)
    salt1 = _tracer_content(model, s, s.S.data)
    assert bool(jnp.all(jnp.isfinite(s.T.data)))
    # Rigid-lid fixed-thickness ⇒ flux-form + zero-flux implicit mixing conserve.
    assert abs(heat1 - heat0) / max(abs(heat0), 1.0) < 1.0e-12, (
        f"heat not conserved: {heat0:.6e} -> {heat1:.6e}")
    assert abs(salt1 - salt0) / max(abs(salt0), 1.0) < 1.0e-12, (
        f"salt not conserved: {salt0:.6e} -> {salt1:.6e}")


# --------------------------------------------------------------------------
# (c) Validation: implicit surface forcing requires implicit vertical mixing
# --------------------------------------------------------------------------

def test_requires_implicit_vertical_mixing():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        surface_forcing_implicit=True, implicit_vertical_mixing=False)
    with pytest.raises(ValueError, match="requires implicit_vertical_mixing"):
        LatLonCGridOceanModel(grid, z_coord, cfg)


# --------------------------------------------------------------------------
# (d) Differentiability through a step with the implicit option
# --------------------------------------------------------------------------

def test_grad_finite_through_implicit_step():
    # Use the explicit_substep barotropic solver so the differentiated step has
    # no host-side rigid-lid island flood-fill (which cannot run under tracing).
    # The implicit surface-forcing path is solver-agnostic.
    state, model = _basin(surface_forcing_implicit=True,
                          outer_integrator="ab2",
                          barotropic_solver="explicit_substep",
                          dt_mom_ratio=1.0)

    def loss(T_data):
        s = state._replace(T=state.T.replace(data=T_data))
        s2 = model.step(s, dt=_DT, surface_forcing=None)
        return jnp.sum(s2.T.data ** 2)

    g = jax.grad(loss)(state.T.data)
    jax.block_until_ready(g)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


# --------------------------------------------------------------------------
# q_net / shortwave routing through _bc_external_surface_forcing
# --------------------------------------------------------------------------

def test_q_net_routed_to_surface_tracer_forcing():
    """A prescribed q_net (via OceanSurfaceForcing) is withheld from the explicit
    dT_dt and routed onto surface_tracer_forcing when the flag is on."""
    from legoesm.ocean.state import OceanSurfaceForcing
    # No physics restoring here — isolate the q_net path.
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(lm))
    q_net = jnp.full((_N_LAT, _N_LON), 50.0)   # +50 W/m² into ocean
    sf = OceanSurfaceForcing(q_net=q_net)

    cfg_e = LatLonCGridOceanConfig.from_flat(
        implicit_vertical_mixing=True, enable_runtime_checks=False,
        surface_forcing_implicit=False, physics=None)
    cfg_i = cfg_e._replace(surface_forcing_implicit=True)
    m_e = LatLonCGridOceanModel(grid, z_coord, cfg_e)
    m_i = LatLonCGridOceanModel(grid, z_coord, cfg_i)
    t_e = m_e.tendencies(state, surface_forcing=sf, dt=_DT)
    t_i = m_i.tendencies(state, surface_forcing=sf, dt=_DT)

    # Explicit: q_net heats the surface dT_dt; implicit: dT_dt surface ~0,
    # the heating lives on surface_tracer_forcing instead.
    dT_e_surf = np.asarray(t_e.dT_dt.data)[:, :, 0]
    assert np.any(dT_e_surf > 0.0)
    assert t_i.surface_tracer_forcing is not None
    dT_i_expl_surf = np.asarray(t_i.dT_dt.data)[:, :, 0]
    assert np.max(np.abs(dT_i_expl_surf)) < 1.0e-15
    dT_i_surf = np.asarray(t_i.surface_tracer_forcing.dT_dt.data)[:, :, 0]
    np.testing.assert_allclose(dT_i_surf, dT_e_surf, rtol=0, atol=0)
