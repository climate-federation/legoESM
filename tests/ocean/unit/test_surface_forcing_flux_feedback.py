"""Veros-faithful "flux + feedback" surface forcing (``scheme="flux_feedback"``).

Gates (global_4deg transfer, SCOPING.md §A.2-A.4):

1. EXACT-VALUE: the scheme reproduces a numpy transcription of the Veros
   global_4deg ``set_forcing_kernel`` heat/salt block
   (veros/setups/global_4deg/global_4deg.py:249-266) followed by the tracer
   core's ``forc/dzt`` rate conversion (veros/core/thermodynamics.py:276-282),
   to 1e-14 relative.
2. Ice-mask logic: all four quadrants of ``(T_surf < -1.8 °C, Q < 0)``.
3. Dispatch: ``flux_feedback`` is a ``make_surface_forcing_physics`` citizen;
   unknown schemes still raise.
4. Bit-identity of existing paths: ``OceanSurfaceForcing`` treedef is
   unchanged by the new default-None channels; the ``restoring`` scheme
   output is bitwise identical with/without the new fields spelled out.
5. Differentiability: d(dT_dt)/dT_surf finite and NEGATIVE where
   q_feedback > 0 (warmer SST ⇒ less heating).
6. Implicit-seam routing: with ``surface_forcing_implicit=True`` the scheme's
   rates reach the backward-Euler solve via ``surface_tracer_forcing`` (rate
   equality vs the explicit placement; 1-step both placements finite).

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

from legoesm import constants
from legoesm.ocean.physics.surface_forcing.config import (
    FluxFeedbackConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.surface_forcing.flux_feedback import (
    flux_feedback_surface_forcing,
)
from legoesm.ocean.physics.surface_forcing.integration import (
    make_surface_forcing_physics,
)
from legoesm.ocean.state import OceanSurfaceForcing

# Veros global_4deg setup-kernel values (the cp_0 literal is a setup-kernel
# hardcode — NOT legoesm.constants.c_sw = 3994.0; the 0.05% mismatch is
# documented on FluxFeedbackConfig).
_CP0_VEROS = 3991.86795711963
_RHO0_VEROS = 1024.0
_T_REST = 30 * 86400.0
_DZT_SURF = 50.0

_NLAT, _NLON, _NLEV = 4, 5, 3


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _manufactured_fields():
    """Small deterministic fields covering: land, warm/cold surface, heating/
    cooling, so every branch of the kernel is exercised."""
    rng = np.random.default_rng(42)
    maskT = np.ones((_NLAT, _NLON))
    maskT[0, 0] = 0.0
    maskT[3, 4] = 0.0
    T = rng.uniform(2.0, 20.0, size=(_NLAT, _NLON, _NLEV))
    # Cold (sub-freezing) surface cells for the ice mask:
    T[1, 1, 0] = -2.5    # cold
    T[1, 2, 0] = -2.5    # cold
    S = rng.uniform(33.0, 36.0, size=(_NLAT, _NLON, _NLEV))
    qnet = rng.uniform(-200.0, 200.0, size=(_NLAT, _NLON))
    qnet[1, 1] = -150.0  # cold + cooling -> ice-masked
    qnet[1, 2] = +150.0  # cold + heating -> NOT masked
    qnec = rng.uniform(0.0, 50.0, size=(_NLAT, _NLON))
    # Make the feedback term small in the ice rows so qnet controls the sign.
    qnec[1, 1] = 1.0
    qnec[1, 2] = 1.0
    sst = rng.uniform(-2.0, 25.0, size=(_NLAT, _NLON))
    sst[1, 1] = -1.9
    sst[1, 2] = -1.9
    sss = rng.uniform(33.0, 37.0, size=(_NLAT, _NLON))
    dz_0 = _DZT_SURF * maskT  # zero on land = wet/dry mask
    return maskT, T, S, qnet, qnec, sst, sss, dz_0


def _veros_kernel_replica(maskT, T, S, qnet, qnec, sst, sss):
    """Numpy transcription of global_4deg.py:249-266 (+ the core's /dzt rate
    conversion, thermodynamics.py:276-282).  Veros works in K·m/s / PSU·m/s;
    the rates come out in K/s / PSU/s."""
    T_surf = T[..., 0]
    S_surf = S[..., 0]
    # heat flux : W/m^2 K kg/J m^3/kg = K m/s
    forc_temp = (qnet + qnec * (sst - T_surf)) * maskT / _CP0_VEROS / _RHO0_VEROS
    # salinity restoring
    forc_salt = 1.0 / _T_REST * (sss - S_surf) * maskT * _DZT_SURF
    # apply simple ice mask
    mask_ice = np.logical_and(T_surf * maskT < -1.8, forc_temp < 0.0)
    forc_temp = np.where(mask_ice, 0.0, forc_temp)
    forc_salt = np.where(mask_ice, 0.0, forc_salt)
    # tracer core: source rate = forc / dzt[surface]
    return forc_temp / _DZT_SURF, forc_salt / _DZT_SURF, mask_ice


def _cfg_veros():
    return FluxFeedbackConfig(c_sw=_CP0_VEROS, rho_0=_RHO0_VEROS,
                              tau_restore_s=_T_REST)


def _forcing(qnet, qnec, sst, sss):
    return OceanSurfaceForcing(
        q_prescribed=jnp.asarray(qnet), q_feedback=jnp.asarray(qnec),
        T_feedback_target=jnp.asarray(sst), S_restore_target=jnp.asarray(sss),
    )


# --------------------------------------------------------------------------
# 1. Exact-value vs the Veros kernel replica
# --------------------------------------------------------------------------

def test_matches_veros_kernel_replica_to_1e14():
    maskT, T, S, qnet, qnec, sst, sss, dz_0 = _manufactured_fields()
    dT_ref, dS_ref, mask_ice = _veros_kernel_replica(
        maskT, T, S, qnet, qnec, sst, sss)
    assert mask_ice.sum() == 1, "fixture must exercise the ice mask"

    out = flux_feedback_surface_forcing(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(dz_0),
        _forcing(qnet, qnec, sst, sss), _cfg_veros(),
    )
    np.testing.assert_allclose(np.asarray(out.dT_dt[..., 0]), dT_ref,
                               rtol=1e-14, atol=1e-25)
    np.testing.assert_allclose(np.asarray(out.dS_dt[..., 0]), dS_ref,
                               rtol=1e-14, atol=1e-25)
    # Surface-layer source only; no momentum forcing from this scheme.
    assert np.max(np.abs(np.asarray(out.dT_dt[..., 1:]))) == 0.0
    assert np.max(np.abs(np.asarray(out.dS_dt[..., 1:]))) == 0.0
    assert np.max(np.abs(np.asarray(out.du_dt))) == 0.0
    assert np.max(np.abs(np.asarray(out.dv_dt))) == 0.0
    # Land columns carry zero forcing (maskT).
    assert np.asarray(out.dT_dt)[0, 0, 0] == 0.0
    assert np.asarray(out.dS_dt)[3, 4, 0] == 0.0


def test_units_prescribed_only_is_q_over_rho_cp_dz():
    """q_feedback=0 ⇒ dT_dt = Q/(rho_0·c_sw·dz_0) exactly (the single W/m² →
    K/s conversion, at the Veros factor placement)."""
    T = jnp.zeros((2, 2, _NLEV)) + 10.0
    S = jnp.zeros((2, 2, _NLEV)) + 35.0
    dz_0 = jnp.full((2, 2), _DZT_SURF)
    Q = 100.0
    sf = OceanSurfaceForcing(q_prescribed=jnp.full((2, 2), Q))
    out = flux_feedback_surface_forcing(T, S, dz_0, sf, _cfg_veros())
    expected = Q / (_RHO0_VEROS * _CP0_VEROS * _DZT_SURF)
    np.testing.assert_allclose(np.asarray(out.dT_dt[..., 0]), expected,
                               rtol=1e-15)
    # No salinity channel -> zero dS.
    assert np.max(np.abs(np.asarray(out.dS_dt))) == 0.0


def test_all_channels_none_gives_zero():
    T = jnp.zeros((2, 2, _NLEV)) + 10.0
    S = jnp.zeros((2, 2, _NLEV)) + 35.0
    dz_0 = jnp.full((2, 2), _DZT_SURF)
    out = flux_feedback_surface_forcing(
        T, S, dz_0, OceanSurfaceForcing(), _cfg_veros())
    for arr in (out.dT_dt, out.dS_dt, out.du_dt, out.dv_dt, out.Q_net):
        assert np.max(np.abs(np.asarray(arr))) == 0.0


def test_feedback_without_target_raises():
    T = jnp.zeros((2, 2, _NLEV))
    S = jnp.zeros((2, 2, _NLEV))
    dz_0 = jnp.full((2, 2), _DZT_SURF)
    sf = OceanSurfaceForcing(q_feedback=jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="T_feedback_target"):
        flux_feedback_surface_forcing(T, S, dz_0, sf, _cfg_veros())


def test_target_without_feedback_raises():
    """A target SST without a piston coefficient would be silently inert."""
    T = jnp.zeros((2, 2, _NLEV))
    S = jnp.zeros((2, 2, _NLEV))
    dz_0 = jnp.full((2, 2), _DZT_SURF)
    sf = OceanSurfaceForcing(T_feedback_target=jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="q_feedback"):
        flux_feedback_surface_forcing(T, S, dz_0, sf, _cfg_veros())


def test_nonpositive_restore_timescale_raises():
    T = jnp.zeros((2, 2, _NLEV))
    S = jnp.zeros((2, 2, _NLEV))
    dz_0 = jnp.full((2, 2), _DZT_SURF)
    sf = OceanSurfaceForcing(S_restore_target=jnp.full((2, 2), 35.0))
    with pytest.raises(ValueError, match="tau_restore_s"):
        flux_feedback_surface_forcing(
            T, S, dz_0, sf, _cfg_veros()._replace(tau_restore_s=0.0))


# --------------------------------------------------------------------------
# 2. Ice-mask quadrants: (T < -1.8) x (Q < 0)
# --------------------------------------------------------------------------

def test_ice_mask_all_four_quadrants():
    # 4 wet cells: (cold, cooling), (cold, heating), (warm, cooling),
    # (warm, heating). Only the first is zeroed.
    T = jnp.asarray(
        np.stack([np.array([[-2.5, -2.5], [5.0, 5.0]])] + [np.full((2, 2), 4.0)] * (_NLEV - 1),
                 axis=-1))
    S = jnp.full((2, 2, _NLEV), 34.0)
    dz_0 = jnp.full((2, 2), _DZT_SURF)
    qnet = np.array([[-50.0, +50.0], [-50.0, +50.0]])
    sss = np.full((2, 2), 35.0)
    sf = OceanSurfaceForcing(q_prescribed=jnp.asarray(qnet),
                             S_restore_target=jnp.asarray(sss))
    out = flux_feedback_surface_forcing(T, S, dz_0, sf, _cfg_veros())
    dT = np.asarray(out.dT_dt[..., 0])
    dS = np.asarray(out.dS_dt[..., 0])
    # (cold, cooling): BOTH zeroed.
    assert dT[0, 0] == 0.0 and dS[0, 0] == 0.0
    # (cold, heating): both live.
    assert dT[0, 1] > 0.0 and dS[0, 1] > 0.0
    # (warm, cooling): both live.
    assert dT[1, 0] < 0.0 and dS[1, 0] > 0.0
    # (warm, heating): both live.
    assert dT[1, 1] > 0.0 and dS[1, 1] > 0.0
    # Q_net diagnostic reflects the zeroing.
    Qd = np.asarray(out.Q_net)
    assert Qd[0, 0] == 0.0 and Qd[0, 1] > 0.0


def test_ice_threshold_comes_from_constants():
    cfg = FluxFeedbackConfig()
    assert cfg.ice_threshold_C == constants.T_freeze_ocean - constants.T_freeze
    # -1.8 °C to float representation (documented 4.6e-14 gap vs the literal).
    assert abs(cfg.ice_threshold_C - (-1.8)) < 1e-13


def test_ice_mask_can_be_disabled():
    T = jnp.full((1, 1, _NLEV), -2.5)
    S = jnp.full((1, 1, _NLEV), 34.0)
    dz_0 = jnp.full((1, 1), _DZT_SURF)
    sf = OceanSurfaceForcing(q_prescribed=jnp.full((1, 1), -50.0))
    out_on = flux_feedback_surface_forcing(T, S, dz_0, sf, _cfg_veros())
    out_off = flux_feedback_surface_forcing(
        T, S, dz_0, sf, _cfg_veros()._replace(ice_mask=False))
    assert np.asarray(out_on.dT_dt)[0, 0, 0] == 0.0
    assert np.asarray(out_off.dT_dt)[0, 0, 0] < 0.0


# --------------------------------------------------------------------------
# 3. Dispatch
# --------------------------------------------------------------------------

def test_factory_dispatches_flux_feedback():
    fn = make_surface_forcing_physics(
        SurfaceForcingConfig(scheme="flux_feedback"))
    assert callable(fn)


def test_factory_unknown_scheme_still_raises():
    with pytest.raises(ValueError, match="Unknown surface forcing scheme"):
        make_surface_forcing_physics(SurfaceForcingConfig(scheme="flux_fedback"))


# --------------------------------------------------------------------------
# 4. Bit-identity of existing paths (new channels default None)
# --------------------------------------------------------------------------

def test_surface_forcing_pytree_treedef_stable():
    q = jnp.full((2, 2), 5.0)
    sf_default = OceanSurfaceForcing(q_net=q)
    sf_spelled = OceanSurfaceForcing(
        q_net=q, q_prescribed=None, q_feedback=None,
        T_feedback_target=None, S_restore_target=None)
    td_a = jax.tree_util.tree_structure(sf_default)
    td_b = jax.tree_util.tree_structure(sf_spelled)
    assert td_a == td_b
    # None channels contribute NO leaves (pytree-stable for jit/scan carries).
    assert len(jax.tree_util.tree_leaves(sf_default)) == 1
    assert OceanSurfaceForcing().q_prescribed is None


def test_restoring_scheme_bitwise_unchanged_by_new_channels():
    """ACC-style restoring config: identical output whether the passed
    OceanSurfaceForcing spells out the new channels or not (and the scheme
    ignores them entirely)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.surface_forcing.config import RestoringConfig

    grid = create_latlon_grid(_NLAT, _NLON)
    z_coord = create_ocean_z_star(n_levels=_NLEV, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    cfg = SurfaceForcingConfig(
        scheme="restoring",
        restoring=RestoringConfig(tau_T=_T_REST, tau_S=_T_REST,
                                  T_star_eq=25.0, T_star_pole=0.0))
    fn = make_surface_forcing_physics(cfg)
    sf_a = OceanSurfaceForcing(q_net=jnp.full((_NLAT, _NLON), 5.0))
    sf_b = OceanSurfaceForcing(q_net=jnp.full((_NLAT, _NLON), 5.0),
                               q_prescribed=None, q_feedback=None,
                               T_feedback_target=None, S_restore_target=None)
    out_a = fn(state, grid, z_coord, sf_a)
    out_b = fn(state, grid, z_coord, sf_b)
    np.testing.assert_array_equal(np.asarray(out_a.dT_dt.data),
                                  np.asarray(out_b.dT_dt.data))
    np.testing.assert_array_equal(np.asarray(out_a.dS_dt.data),
                                  np.asarray(out_b.dS_dt.data))
    assert np.any(np.asarray(out_a.dT_dt.data) != 0.0)


# --------------------------------------------------------------------------
# 5. Differentiability + feedback sign
# --------------------------------------------------------------------------

def test_grad_wrt_T_surf_finite_and_negative_with_positive_feedback():
    """d(dT_dt)/dT_surf = -q_feedback/(rho_0 c_sw dz_0) < 0: warmer SST ⇒
    less heating (the piston damping)."""
    qnec = 40.0
    dz = _DZT_SURF

    def heating(T_surf_scalar):
        T = jnp.full((1, 1, _NLEV), 10.0).at[0, 0, 0].set(T_surf_scalar)
        S = jnp.full((1, 1, _NLEV), 35.0)
        sf = OceanSurfaceForcing(
            q_prescribed=jnp.full((1, 1), 20.0),
            q_feedback=jnp.full((1, 1), qnec),
            T_feedback_target=jnp.full((1, 1), 15.0))
        out = flux_feedback_surface_forcing(
            T, S, jnp.full((1, 1), dz), sf, _cfg_veros())
        return out.dT_dt[0, 0, 0]

    g = jax.grad(heating)(jnp.asarray(10.0))
    assert bool(jnp.isfinite(g))
    expected = -qnec / (_RHO0_VEROS * _CP0_VEROS * dz)
    np.testing.assert_allclose(float(g), expected, rtol=1e-12)


# --------------------------------------------------------------------------
# 6. Implicit-seam routing (surface_forcing_implicit=True)
# --------------------------------------------------------------------------

def _basin_flux_feedback(*, surface_forcing_implicit, barotropic_solver="explicit_substep"):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig

    n_lat, n_lon = 8, 16
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    lm = np.ones((n_lat, n_lon)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((n_lat, n_lon), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(lm),
        H_bathy_override=jnp.asarray(Hb))
    phys = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(
            scheme="flux_feedback", flux_feedback=_cfg_veros()),
    )
    cfg = LatLonCGridOceanConfig(
        A_h=2.0e4, implicit_vertical_mixing=True,
        enable_runtime_checks=False, barotropic_solver=barotropic_solver,
        outer_integrator="ab2",
        surface_forcing_implicit=surface_forcing_implicit,
        physics=phys,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    sf = OceanSurfaceForcing(
        q_prescribed=jnp.full((n_lat, n_lon), 80.0),
        q_feedback=jnp.full((n_lat, n_lon), 30.0),
        T_feedback_target=jnp.full((n_lat, n_lon), 18.0),
        S_restore_target=jnp.full((n_lat, n_lon), 36.0),
    )
    return state, model, sf


def test_implicit_seam_routes_flux_feedback_rates():
    """Explicit placement puts the rates on dT_dt/dS_dt; implicit placement
    withholds them and routes the SAME rates onto surface_tracer_forcing."""
    dt = 3600.0
    st_e, m_e, sf = _basin_flux_feedback(surface_forcing_implicit=False)
    st_i, m_i, _ = _basin_flux_feedback(surface_forcing_implicit=True)
    t_e = m_e.tendencies(st_e, surface_forcing=sf, dt=dt)
    t_i = m_i.tendencies(st_i, surface_forcing=sf, dt=dt)

    assert t_e.surface_tracer_forcing is None
    dT_e = np.asarray(t_e.dT_dt.data)[:, :, 0]
    dS_e = np.asarray(t_e.dS_dt.data)[:, :, 0]
    assert np.any(np.abs(dT_e) > 0.0) and np.any(np.abs(dS_e) > 0.0)

    assert t_i.surface_tracer_forcing is not None
    # Withheld from the explicit tendencies...
    assert np.max(np.abs(np.asarray(t_i.dT_dt.data)[:, :, 0])) < 1e-15
    assert np.max(np.abs(np.asarray(t_i.dS_dt.data)[:, :, 0])) < 1e-15
    # ... and routed, EXACTLY, onto the implicit surface forcing (the ice
    # mask was applied INSIDE the scheme, so the placement cannot change
    # the masking semantics).
    dT_i = np.asarray(t_i.surface_tracer_forcing.dT_dt.data)[:, :, 0]
    dS_i = np.asarray(t_i.surface_tracer_forcing.dS_dt.data)[:, :, 0]
    np.testing.assert_allclose(dT_i, dT_e, rtol=0, atol=0)
    np.testing.assert_allclose(dS_i, dS_e, rtol=0, atol=0)
    # Surface-layer source only.
    assert np.max(np.abs(np.asarray(
        t_i.surface_tracer_forcing.dT_dt.data)[:, :, 1:])) == 0.0


def test_one_step_explicit_and_implicit_placements_finite():
    dt = 3600.0
    st_e, m_e, sf = _basin_flux_feedback(surface_forcing_implicit=False)
    st_i, m_i, _ = _basin_flux_feedback(surface_forcing_implicit=True)
    s_e = m_e.step(st_e, dt=dt, surface_forcing=sf)
    s_i = m_i.step(st_i, dt=dt, surface_forcing=sf)
    for s in (s_e, s_i):
        jax.block_until_ready(s.T.data)
        assert bool(jnp.all(jnp.isfinite(s.T.data)))
        assert bool(jnp.all(jnp.isfinite(s.S.data)))
        assert bool(jnp.all(jnp.isfinite(s.u.data)))
    # The forcing was actually applied in BOTH placements (warming +
    # salinification toward the targets on wet cells).
    lm = np.asarray(st_e.land_mask.data).astype(bool)
    T0 = np.asarray(st_e.T.data)[lm, 0]
    for s in (s_e, s_i):
        T1 = np.asarray(s.T.data)[lm, 0]
        assert np.mean(T1) != pytest.approx(np.mean(T0), abs=0.0)


def test_implicit_seam_with_none_forcing_is_inert():
    """flux_feedback + surface_forcing_implicit=True + NO forcing passed:
    the routed surface forcing is exactly zero (T-shaped, no staggered-shape
    clash from the full C-grid state at the 10b'' seam)."""
    dt = 3600.0
    state, model, _ = _basin_flux_feedback(surface_forcing_implicit=True)
    tend = model.tendencies(state, surface_forcing=None, dt=dt)
    assert tend.surface_tracer_forcing is not None
    assert np.max(np.abs(np.asarray(
        tend.surface_tracer_forcing.dT_dt.data))) == 0.0
    assert np.max(np.abs(np.asarray(
        tend.surface_tracer_forcing.dS_dt.data))) == 0.0


def test_grad_finite_through_implicit_step():
    dt = 3600.0
    state, model, sf = _basin_flux_feedback(surface_forcing_implicit=True)

    def loss(T_data):
        s = state._replace(T=state.T.replace(data=T_data))
        s2 = model.step(s, dt=dt, surface_forcing=sf)
        return jnp.sum(s2.T.data ** 2)

    g = jax.grad(loss)(state.T.data)
    jax.block_until_ready(g)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_q_net_plus_q_prescribed_raises_double_count_guard():
    """Dispatch hardening (review advisory): under scheme="flux_feedback" a
    simultaneous q_net would ALSO be routed by the external-surface-forcing
    stage and double-count the heat — must raise at trace time."""
    st, model, sf = _basin_flux_feedback(surface_forcing_implicit=False)
    n_lat, n_lon = sf.q_prescribed.shape
    sf_bad = sf._replace(q_net=jnp.full((n_lat, n_lon), 5.0))
    with pytest.raises(ValueError, match="double-count"):
        model.tendencies(st, surface_forcing=sf_bad, dt=3600.0)
    # And the good form still works.
    t = model.tendencies(st, surface_forcing=sf, dt=3600.0)
    assert np.any(np.abs(np.asarray(t.dT_dt.data)[:, :, 0]) > 0.0)
