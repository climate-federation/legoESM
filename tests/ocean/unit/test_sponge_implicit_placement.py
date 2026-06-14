"""EXT-N2: implicit (weight-1.0 pre-vmix) sponge placement
(``LatLonCGridOceanConfig.sponge_forcing_implicit``).

Veros applies its tempsalt sources (the NA sponge zones) forward-Euler at
taup1, AFTER the AB2'd advection/diffusion and BEFORE the implicit vertical
mixing, at weight 1.0 — NOT AB2-extrapolated
(``veros/core/thermodynamics.py:419`` → ``veros/core/diffusion.py:132-141``:
``temp[taup1] += dt_tracer·temp_source·maskT``).  legoESM's stage-10c sponge
rides the explicit ``dT_dt`` and is therefore 1.6×-extrapolated by the
faithful AB2 outer integrator — the same placement class the Rung-6 oracle
tendency-match exposed for surface forcing.

``sponge_forcing_implicit=True`` withholds the sponge T/S rates from the
explicit tendency, routes them onto the NEW
``LatLonCGridOceanTendencies.tracer_source`` slot, and applies them at weight
1.0 inside the backward-Euler vertical-mixing solve.  CRITICALLY, the rates do
NOT ride ``surface_tracer_forcing``: the post-mixing TKE surface buoyancy-flux
reconstruction column-sums that slot to rebuild Veros's ``forc_temp_surface``,
and Veros keeps tempsalt_sources OUT of ``forc_rho_surface``
(thermodynamics.py:312-314) — covered by the exclusion tripwire below.

Gates:
  (a) default off → ``tracer_source`` is None and the sponge stays in the
      explicit ``dT_dt`` (bit-identical legacy placement);
  (b) on → the explicit tendency excludes the sponge and ``tracer_source``
      carries exactly the stage-10c rate (rate-for-rate);
  (c) placement equality: ONE forward-Euler step explicit vs implicit is
      bit-near-identical (the same dt·rate enters either way);
  (d) AB2 weight-1 semantics, hand-computed: implicit follows the plain
      forward recursion; explicit follows the (1.5+ε)/(0.5+ε) AB2 recursion
      (1.6× bootstrap) — the two demonstrably differ;
  (e) ``sponge_forcing_implicit=True`` + ``implicit_vertical_mixing=False``
      raises;
  (f) TKE exclusion tripwire (the documented EXT-N2 caveat);
  (g) jax.grad finite through the implicit-sponge AB2 step.

Run in fp64 on CPU for determinism.
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

_N_LAT, _N_LON, _NLEV = 8, 16, 4
_H_MAX = 4000.0
_DT = 1800.0
_GAMMA = 1.0 / (3.0 * 86400.0)   # 3-day restoring (the NA rest_tscl max class)


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _grid_z():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    return (create_latlon_grid(_N_LAT, _N_LON),
            create_ocean_z_star(n_levels=_NLEV, H_max=_H_MAX))


def _basin_state(grid, z_coord, *, uniform_T=None):
    """Closed flat-bottom channel (land walls N/S, periodic x)."""
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((_N_LAT, _N_LON), _H_MAX)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(lm),
        H_bathy_override=jnp.asarray(Hb))
    if uniform_T is not None:
        # z-UNIFORM tracers: the implicit vmix solve is then an exact identity
        # on a per-column-uniform update (zero-flux BCs), so the sponge is the
        # ONLY active tendency and the trajectories are hand-computable.
        state = state._replace(
            T=state.T.replace(data=jnp.full_like(state.T.data, uniform_T)),
            S=state.S.replace(data=jnp.full_like(state.S.data, 35.0)),
        )
    return state, lm


def _sponge_full_column(state, lm, seed=0):
    """3-D gamma (exercises EXT-N1 × EXT-N2 composition), constant along z so
    each column stays z-uniform — horizontally varying."""
    from legoesm.ocean.sponge import SpongeForcing
    rng = np.random.default_rng(seed)
    g2 = rng.uniform(0.2, 1.0, size=(_N_LAT, _N_LON)) * _GAMMA * lm
    g3 = np.repeat(g2[..., None], _NLEV, axis=-1)
    T_ref = np.asarray(state.T.data) + 2.0
    S_ref = np.asarray(state.S.data) - 0.5
    return SpongeForcing(gamma=jnp.asarray(g3), T_ref=jnp.asarray(T_ref),
                         S_ref=jnp.asarray(S_ref)), g3, T_ref, S_ref


def _model(grid, z_coord, state, **kw):
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    cfg = LatLonCGridOceanConfig(
        A_h=2.0e4, implicit_vertical_mixing=True,
        enable_runtime_checks=False, barotropic_solver="rigid_lid", **kw)
    m = LatLonCGridOceanModel(grid, z_coord, cfg)
    # Host-side island flood-fill cannot run inside the jitted step.
    m._ensure_rigid_lid_data(state)
    return m


# ---------------------------------------------------------------------------
# (a) default off — bit-identical legacy placement
# ---------------------------------------------------------------------------

def test_default_off_tracer_source_none_and_sponge_explicit():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    grid, z_coord = _grid_z()
    state, lm = _basin_state(grid, z_coord)
    sponge, g3, T_ref, _ = _sponge_full_column(state, lm)
    m = _model(grid, z_coord, state, sponge_forcing_implicit=False)
    tend = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, m.config, sponge=sponge, dt=_DT)
    tend0 = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, m.config, sponge=None, dt=_DT)
    assert tend.tracer_source is None
    assert tend0.tracer_source is None
    # The sponge rate sits in the explicit dT_dt (legacy stage-10c placement).
    rate = g3 * (T_ref - np.asarray(state.T.data)) * lm[..., None]
    np.testing.assert_allclose(
        np.asarray(tend.dT_dt.data - tend0.dT_dt.data), rate,
        rtol=0, atol=1e-18)


# ---------------------------------------------------------------------------
# (b) on — withheld from explicit, routed rate-for-rate onto tracer_source
# ---------------------------------------------------------------------------

def test_implicit_withholds_and_routes_exact_rate():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    grid, z_coord = _grid_z()
    state, lm = _basin_state(grid, z_coord)
    sponge, g3, T_ref, S_ref = _sponge_full_column(state, lm)
    m_on = _model(grid, z_coord, state, sponge_forcing_implicit=True)
    m_off = _model(grid, z_coord, state, sponge_forcing_implicit=False)
    t_on = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, m_on.config, sponge=sponge, dt=_DT)
    t_none = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, m_off.config, sponge=None, dt=_DT)
    # Explicit tendency excludes the sponge entirely…
    np.testing.assert_array_equal(np.asarray(t_on.dT_dt.data),
                                  np.asarray(t_none.dT_dt.data))
    np.testing.assert_array_equal(np.asarray(t_on.dS_dt.data),
                                  np.asarray(t_none.dS_dt.data))
    # …and tracer_source carries EXACTLY the masked stage-10c rate.
    rate_T = g3 * (T_ref - np.asarray(state.T.data)) * lm[..., None]
    rate_S = g3 * (S_ref - np.asarray(state.S.data)) * lm[..., None]
    np.testing.assert_array_equal(np.asarray(t_on.tracer_source.dT_dt.data),
                                  rate_T)
    np.testing.assert_array_equal(np.asarray(t_on.tracer_source.dS_dt.data),
                                  rate_S)
    # The surface slot is untouched (the TKE-exclusion contract is on slots).
    assert t_on.surface_tracer_forcing is None


# ---------------------------------------------------------------------------
# (c) placement equality: one forward-Euler step, rate-for-rate
# ---------------------------------------------------------------------------

def test_placement_equality_forward_euler_one_step():
    grid, z_coord = _grid_z()
    state, lm = _basin_state(grid, z_coord)
    sponge, _, _, _ = _sponge_full_column(state, lm)
    m_exp = _model(grid, z_coord, state, outer_integrator="forward_euler",
                   sponge_forcing_implicit=False)
    m_imp = _model(grid, z_coord, state, outer_integrator="forward_euler",
                   sponge_forcing_implicit=True)
    s_exp = m_exp.step(state, _DT, sponge=sponge)
    s_imp = m_imp.step(state, _DT, sponge=sponge)
    # Forward Euler applies the explicit bucket at weight 1.0, so moving the
    # dt·rate from the bucket to the solve input is the SAME update up to
    # floating-point association (measured bit-exact on this basin).
    np.testing.assert_allclose(np.asarray(s_exp.T.data),
                               np.asarray(s_imp.T.data), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(s_exp.S.data),
                               np.asarray(s_imp.S.data), rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# (d) AB2: hand-computed weight-1 vs AB2-extrapolated recursions
# ---------------------------------------------------------------------------

def test_ab2_weight_one_hand_computed_two_steps():
    grid, z_coord = _grid_z()
    state, lm = _basin_state(grid, z_coord, uniform_T=10.0)
    sponge, g3, T_ref, S_ref = _sponge_full_column(state, lm, seed=3)
    m_exp = _model(grid, z_coord, state, outer_integrator="ab2",
                   sponge_forcing_implicit=False)
    m_imp = _model(grid, z_coord, state, outer_integrator="ab2",
                   sponge_forcing_implicit=True)
    eps = m_exp.config.ab2_epsilon
    a_n, a_p = 1.5 + eps, 0.5 + eps
    wet = np.broadcast_to(lm[..., None] > 0.5, (_N_LAT, _N_LON, _NLEV))

    s_e1 = m_exp.step(state, _DT, sponge=sponge)
    s_e2 = m_exp.step(s_e1, _DT, sponge=sponge)
    s_i1 = m_imp.step(state, _DT, sponge=sponge)
    s_i2 = m_imp.step(s_i1, _DT, sponge=sponge)

    T0 = np.asarray(state.T.data)
    rate = lambda T: g3 * (T_ref - T)            # noqa: E731

    # Step-2 tolerance: after step 1 the horizontally-varying sponge leaves a
    # horizontal T pattern (~0.01 K) whose PGF spins up u ~ 1e-4 m/s WITHIN
    # step 2; the step-2 tracer advection by that velocity is a genuine
    # physical coupling of ~1e-9 K that the scalar recursions ignore.  The
    # PLACEMENT signal under test (the 0.6×dt·rate bootstrap gap, ~8e-3 K) is
    # six orders larger.  Probed: step-1 exact to 2e-15, u(after step 1)=0.
    _ATOL2 = 1.0e-8

    # IMPLICIT (Veros tempsalt_sources): plain weight-1 forward recursion —
    # the rate is NEVER AB2-extrapolated (and never enters the AB2 carry).
    Ti1 = T0 + _DT * rate(T0)
    Ti2 = Ti1 + _DT * rate(Ti1)
    np.testing.assert_allclose(np.asarray(s_i1.T.data)[wet], Ti1[wet],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(s_i2.T.data)[wet], Ti2[wet],
                               rtol=0, atol=_ATOL2)

    # EXPLICIT legacy: AB2 recursion — 1.6× bootstrap, then (1.5+ε)/(0.5+ε).
    Te1 = T0 + a_n * _DT * rate(T0)
    Te2 = Te1 + a_n * _DT * rate(Te1) - a_p * _DT * rate(T0)
    np.testing.assert_allclose(np.asarray(s_e1.T.data)[wet], Te1[wet],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(s_e2.T.data)[wet], Te2[wet],
                               rtol=0, atol=_ATOL2)

    # The placements demonstrably differ under AB2 (the point of EXT-N2):
    # bootstrap over-application is 60% of the one-step sponge increment.
    diff = np.abs(np.asarray(s_e1.T.data) - np.asarray(s_i1.T.data))[wet]
    incr = np.abs(_DT * rate(T0))[wet]
    assert diff.max() > 0.5 * incr.max()

    # Salinity follows the same algebra.
    S0 = np.asarray(state.S.data)
    Si1 = S0 + _DT * g3 * (S_ref - S0)
    np.testing.assert_allclose(np.asarray(s_i1.S.data)[wet], Si1[wet],
                               rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# (e) guard
# ---------------------------------------------------------------------------

def test_requires_implicit_vertical_mixing():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid, z_coord = _grid_z()
    with pytest.raises(ValueError, match="sponge_forcing_implicit"):
        LatLonCGridOceanModel(
            grid, z_coord,
            LatLonCGridOceanConfig(sponge_forcing_implicit=True,
                                   implicit_vertical_mixing=False,
                                   enable_runtime_checks=False))


# ---------------------------------------------------------------------------
# (f) TKE exclusion tripwire (the documented EXT-N2 caveat)
# ---------------------------------------------------------------------------

def test_tke_surface_flux_reconstruction_excludes_sponge():
    """Direct tripwire for the documented EXT-N2 caveat: the post-mixing TKE
    surface buoyancy-flux reconstruction column-sums ``surface_tracer_forcing``
    (rebuilding Veros's forc_temp_surface) and must EXCLUDE the column source
    slot ``tracer_source`` (Veros keeps tempsalt_sources out of
    forc_rho_surface, thermodynamics.py:312-314).

    The SAME deep column rate is routed through both slots of
    ``_apply_implicit_vertical_mixing`` (identical solve inputs ⇒ identical
    T/S either way; K profiles are recomputed from the same ``state``):

      * run A — rate on ``tracer_source``    (the EXT-N2 placement);
      * run B — rate on ``surface_tracer_forcing`` (the WRONG slot: its
        column sum lands in the surface buoyancy flux — synthetic violation);
      * run C — no source (baseline).

    A leak is loud by construction: the deep rate's column sum here is
    ~2.5 K·m/s of spurious surface heat flux, so |tke_B − tke_C| (the leak
    signal this tripwire would produce) must dwarf |tke_A − tke_C| (the
    legitimate second-order N²-mediated effect of the source on the post-
    mixing TKE budget).  Asserting BOTH proves the exclusion AND that the
    tripwire is non-vacuous."""
    from legoesm.core.field import Field
    from legoesm.ocean.state import SurfaceTracerForcing
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, TKEConfig,
    )

    grid, z_coord = _grid_z()
    state, lm = _basin_state(grid, z_coord)
    tke0 = (jnp.where(lm[..., None] > 0.5, 1.0e-3, 0.0)
            * jnp.ones((1, 1, _NLEV - 1))).astype(state.T.data.dtype)
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(prognostic=True,
                          buoyancy_timing="post_mixing_veros",
                          veros_dz_slots=True, n2_mode="adiabatic",
                          positivity="veros_surface_correction")),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    m = _model(grid, z_coord, state, outer_integrator="forward_euler",
               sponge_forcing_implicit=True, K_v=0.0, physics=physics)

    # Strong DEEP column rate (k >= 2): zero surface signature, large column
    # sum (≈ (1/3600)·5 K · 2 levels · 1000 m ≈ 2.8 K·m/s if leaked).
    rate = np.zeros((_N_LAT, _N_LON, _NLEV))
    rate[:, :, 2:] = (5.0 / 3600.0) * lm[..., None]
    src = SurfaceTracerForcing(
        dT_dt=Field(data=jnp.asarray(rate), name="dT_source_rate",
                    dims=("lat", "lon", "level"), units="degC/s"),
        dS_dt=Field(data=jnp.zeros_like(jnp.asarray(rate)),
                    name="dS_source_rate",
                    dims=("lat", "lon", "level"), units="PSU/s"),
    )

    def run(**source_kw):
        return m._apply_implicit_vertical_mixing(
            state, _DT, None, tke_old=tke0, return_tke=True, **source_kw)

    st_a, tke_a = run(tracer_source=src)              # EXT-N2 slot
    st_b, tke_b = run(surface_tracer_forcing=src)     # wrong slot (synthetic)
    st_c, tke_c = run()                               # baseline

    # Identical solve inputs ⇒ identical tracers regardless of slot.
    np.testing.assert_array_equal(np.asarray(st_a.T.data),
                                  np.asarray(st_b.T.data))
    assert np.abs(np.asarray(st_a.T.data - st_c.T.data)).max() > 0.0

    tke_a, tke_b, tke_c = (np.asarray(x) for x in (tke_a, tke_b, tke_c))
    assert np.all(np.isfinite(tke_a))
    leak_signal = np.abs(tke_b - tke_c).max()
    residual = np.abs(tke_a - tke_c).max()
    # Non-vacuity: the wrong slot WOULD corrupt the TKE surface flux…
    assert leak_signal > 0.0
    # …and the EXT-N2 slot is excluded from it (only the second-order
    # N²-mediated effect of the mixed tracers remains).
    assert leak_signal > 50.0 * max(residual, 1e-30), (
        f"tracer_source appears to leak into the TKE surface-flux "
        f"reconstruction: leak_signal={leak_signal:.3e}, "
        f"residual={residual:.3e}")




# ---------------------------------------------------------------------------
# (g) differentiability
# ---------------------------------------------------------------------------

def test_grad_finite_through_implicit_sponge_ab2_step():
    grid, z_coord = _grid_z()
    state, lm = _basin_state(grid, z_coord)
    sponge, _, _, _ = _sponge_full_column(state, lm)
    m = _model(grid, z_coord, state, outer_integrator="ab2",
               sponge_forcing_implicit=True)

    def loss(T0):
        s = state._replace(T=state.T.replace(data=T0))
        s1 = m.step(s, _DT, sponge=sponge)
        return jnp.mean(s1.T.data ** 2) + jnp.mean(s1.S.data ** 2)

    g = jax.grad(loss)(state.T.data)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0
