"""Ice-gated penetrative-shortwave channel on ``flux_feedback`` (EXT-F2).

The Veros global_flexible / global_1deg solar handling as a config option:
``OceanSurfaceForcing.q_solar`` [W/m²] deposited through the SHARED two-band
Jerlov column (type "I" ≡ the Veros literals R=0.58, ζ1=0.35, ζ2=23.0),
cfg-pinned cp_0/rho_0, gated by the SAME simple ice mask (evaluated on the
TOTAL flux incl. solar — Veros's ``forc_temp_surface`` whose qnet includes
solar — and multiplying the FULL 3-D source, ``ice[..., None]``).

Gates:
1. Jerlov column vs a hand-computed two-band exponential, incl. the
   pen(0)-at-surface convention equivalence (Veros zero-column-sum
   redistribution on solar-inclusive qnet ≡ legoESM I(0)=1 full deposition
   on non-solar q_prescribed).
2. Column-integral heat closure: rho_0·c_sw·Σ dz_k·dT_k == q_solar (machine
   precision, full-depth column).
3. Ice-mask quadrants gating the full solar column, incl. the
   solar-flips-the-sign cell that distinguishes mask-on-total from
   mask-on-nonsolar.
4. Veros-kernel replica (numpy transcription of the global setups' solar
   block, Veros orientation + maskT + shallow column) to 1e-14.
5. Double-count / misuse guards raise at trace time.
6. Implicit-vs-explicit placement rate equality (full column) + 1-step finite.
7. AD: analytic d(dT_k)/d(q_solar); grad finite through the step.
8. Defaults inert: q_solar=None bitwise identical; pytree treedef stable.

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

from legoesm.ocean.physics.shortwave_penetration import JERLOV_TYPES
from legoesm.ocean.physics.surface_forcing.config import (
    FluxFeedbackConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.surface_forcing.flux_feedback import (
    flux_feedback_surface_forcing,
)
from legoesm.ocean.state import OceanSurfaceForcing

# Veros setup-kernel literals (global_flexible.py / global_1deg.py
# set_forcing_kernel; cp_0 is a setup hardcode, NOT legoesm.constants.c_sw).
_CP0_VEROS = 3991.86795711963
_RHO0_VEROS = 1024.0
_T_REST = 30 * 86400.0

_NLAT, _NLON, _NLEV = 3, 4, 6
# Non-uniform reference grid (Vinokur-like upper refinement).
_DZ_REF = np.array([10.0, 14.0, 22.0, 40.0, 90.0, 224.0])


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _column_geometry():
    dz_ref = jnp.asarray(_DZ_REF)
    z_half_ref = jnp.concatenate(
        [jnp.zeros(1), -jnp.cumsum(dz_ref)])          # (nlev+1,), z[0]=0
    return dz_ref, z_half_ref


def _cfg_solar(**over):
    base = dict(c_sw=_CP0_VEROS, rho_0=_RHO0_VEROS, tau_restore_s=_T_REST,
                penetrative_shortwave=True, shortwave_water_type="I")
    base.update(over)
    return FluxFeedbackConfig(**base)


def _kernel(T, S, dz_0, sf, cfg, maskT3=None, jacobian=None):
    dz_ref, z_half_ref = _column_geometry()
    if maskT3 is None:
        maskT3 = jnp.ones(T.shape)
    if jacobian is None:
        jacobian = jnp.ones(T.shape[:-1])
    return flux_feedback_surface_forcing(
        T, S, dz_0, sf, cfg,
        dz_ref=dz_ref, z_half_ref=z_half_ref, jacobian=jacobian,
        wet_3d=maskT3,
    )


def _hand_two_band_fracs():
    """Per-cell absorbed fractions from the hand-computed two-band
    exponential I(z) = R·e^{z/ζ1} + (1−R)·e^{z/ζ2}, surface I(0) = 1,
    seabed-residual catch in the deepest grid cell."""
    p = JERLOV_TYPES["I"]
    # Document the Veros-literal identity (global_flexible.py:211-213).
    assert (p.R, p.zeta1, p.zeta2) == (0.58, 0.35, 23.0)
    z_half = np.concatenate([[0.0], -np.cumsum(_DZ_REF)])
    I = p.R * np.exp(z_half / p.zeta1) + (1.0 - p.R) * np.exp(z_half / p.zeta2)
    frac = I[:-1] - I[1:]
    frac[-1] += I[-1]
    return I, frac


# --------------------------------------------------------------------------
# 1. Jerlov column vs hand-computed two-band exponential + pen(0) convention
# --------------------------------------------------------------------------

def test_jerlov_column_matches_hand_two_band():
    Qs = 240.0
    T = jnp.full((_NLAT, _NLON, _NLEV), 10.0)
    S = jnp.full((_NLAT, _NLON, _NLEV), 35.0)
    dz_0 = jnp.full((_NLAT, _NLON), _DZ_REF[0])
    sf = OceanSurfaceForcing(q_solar=jnp.full((_NLAT, _NLON), Qs))
    out = _kernel(T, S, dz_0, sf, _cfg_solar())
    _, frac = _hand_two_band_fracs()
    expected = Qs * frac / (_RHO0_VEROS * _CP0_VEROS * _DZ_REF)
    np.testing.assert_allclose(
        np.asarray(out.dT_dt)[1, 2, :], expected, rtol=1e-14, atol=0)
    # Every level heated (deposition positive), all columns identical.
    assert np.all(np.asarray(out.dT_dt) > 0.0)


def test_pen0_surface_convention_equivalence():
    """Top-cell algebra: legoESM (q_prescribed = qnet_total − qsol, I(0)=1
    full deposition) == Veros (solar-inclusive qnet in the top cell + the
    pen(0)=0 redistribution, top-cell term −qsol·I(z1)): both give
    (qnet_total − qsol·I(z1)) / (rho·cp·dz0)."""
    qnet_total, qsol = 130.0, 210.0
    T = jnp.full((1, 1, _NLEV), 12.0)
    S = jnp.full((1, 1, _NLEV), 35.0)
    dz_0 = jnp.full((1, 1), _DZ_REF[0])
    sf = OceanSurfaceForcing(
        q_prescribed=jnp.full((1, 1), qnet_total - qsol),
        q_solar=jnp.full((1, 1), qsol))
    out = _kernel(T, S, dz_0, sf, _cfg_solar())
    I, _ = _hand_two_band_fracs()
    veros_top = (qnet_total - qsol * I[1]) / (
        _RHO0_VEROS * _CP0_VEROS * _DZ_REF[0])
    np.testing.assert_allclose(
        float(out.dT_dt[0, 0, 0]), veros_top, rtol=1e-14)


# --------------------------------------------------------------------------
# 2. Column-integral heat closure (conservation)
# --------------------------------------------------------------------------

def test_column_integral_of_deposition_equals_surface_flux():
    Qs = 187.5
    T = jnp.full((_NLAT, _NLON, _NLEV), 8.0)
    S = jnp.full((_NLAT, _NLON, _NLEV), 34.0)
    dz_0 = jnp.full((_NLAT, _NLON), _DZ_REF[0])
    sf = OceanSurfaceForcing(q_solar=jnp.full((_NLAT, _NLON), Qs))
    out = _kernel(T, S, dz_0, sf, _cfg_solar())
    column_heat = _RHO0_VEROS * _CP0_VEROS * np.sum(
        np.asarray(out.dT_dt) * _DZ_REF, axis=-1)
    np.testing.assert_allclose(column_heat, Qs, rtol=1e-14)


# --------------------------------------------------------------------------
# 3. Ice-mask quadrants gate the FULL solar column
# --------------------------------------------------------------------------

def test_ice_mask_quadrants_gate_solar_column():
    # 4 wet cells:
    #  (0,0) cold + total cooling (qns=-100, qsol=20  -> -80 < 0): ALL zeroed
    #  (0,1) cold + solar flips sign (qns=-50, qsol=200 -> +150): live
    #        (distinguishes Veros's mask-on-TOTAL from mask-on-nonsolar)
    #  (1,0) warm + total cooling: live
    #  (1,1) warm + warming: live
    T = jnp.asarray(np.stack(
        [np.array([[-2.5, -2.5], [5.0, 5.0]])]
        + [np.full((2, 2), 4.0)] * (_NLEV - 1), axis=-1))
    S = jnp.full((2, 2, _NLEV), 34.0)
    dz_0 = jnp.full((2, 2), _DZ_REF[0])
    qns = np.array([[-100.0, -50.0], [-100.0, +50.0]])
    qsol = np.array([[20.0, 200.0], [20.0, 200.0]])
    sf = OceanSurfaceForcing(
        q_prescribed=jnp.asarray(qns), q_solar=jnp.asarray(qsol),
        S_restore_target=jnp.full((2, 2), 35.0))
    out = _kernel(T, S, dz_0, sf, _cfg_solar())
    dT = np.asarray(out.dT_dt)
    dS = np.asarray(out.dS_dt)
    Qd = np.asarray(out.Q_net)
    # (cold, total cooling): heat, salt AND every solar level zeroed.
    assert np.all(dT[0, 0, :] == 0.0) and dS[0, 0, 0] == 0.0 and Qd[0, 0] == 0.0
    # (cold, solar-flipped warming): full column live.
    assert np.all(dT[0, 1, :] > 0.0) and dS[0, 1, 0] > 0.0
    # (warm, total cooling): live (subsurface solar heats, top cools).
    assert np.all(dT[1, 0, 1:] > 0.0) and dS[1, 0, 0] > 0.0
    # (warm, warming): live.
    assert np.all(dT[1, 1, :] > 0.0)
    # Q_net diagnostic carries the TOTAL heat incl. the solar column.
    np.testing.assert_allclose(Qd[1, 1], 50.0 + 200.0, rtol=1e-14)


def test_ice_mask_disabled_keeps_solar():
    T = jnp.full((1, 1, _NLEV), -2.5)
    S = jnp.full((1, 1, _NLEV), 34.0)
    dz_0 = jnp.full((1, 1), _DZ_REF[0])
    sf = OceanSurfaceForcing(q_prescribed=jnp.full((1, 1), -100.0),
                             q_solar=jnp.full((1, 1), 20.0))
    out_on = _kernel(T, S, dz_0, sf, _cfg_solar())
    out_off = _kernel(T, S, dz_0, sf, _cfg_solar(ice_mask=False))
    assert np.all(np.asarray(out_on.dT_dt) == 0.0)
    assert np.all(np.asarray(out_off.dT_dt)[0, 0, 1:] > 0.0)


# --------------------------------------------------------------------------
# 4. Veros global-setup kernel replica (numpy transcription, 1e-14)
# --------------------------------------------------------------------------

def _veros_solar_replica(maskT3_v, T_surf, qnet_total, qnec, sst, qsol):
    """Numpy transcription of the global_flexible/global_1deg solar handling:
    set_initial_conditions divpen block (global_flexible.py:292-301) +
    set_forcing_kernel heat/ice/solar block (global_flexible.py:381-405 ≡
    global_1deg.py:322-346).  Veros orientation: level 0 = DEEPEST, level
    -1 = top; zw = upper cell faces, zw[-1] = 0.  Returns the per-cell total
    heating RATE [K/s] (forc/dzt on the top cell + temp_source), Veros
    orientation."""
    R, z1, z2 = 0.58, 0.35, 23.0                       # setup literals
    dzt_v = _DZ_REF[::-1].copy()
    z_half = np.concatenate([[0.0], -np.cumsum(_DZ_REF)])
    zw_v = z_half[:_NLEV][::-1]                        # upper faces, zw[-1]=0
    # --- divpen_shortwave (pen set to 0 at the surface "to compensate for
    #     the shortwave part of the total surface flux") ---
    pen = R * np.exp(zw_v / z1) + (1.0 - R) * np.exp(zw_v / z2)
    pen[-1] = 0.0
    divpen = np.empty(_NLEV)
    divpen[1:] = (pen[1:] - pen[:-1]) / dzt_v[1:]
    divpen[0] = pen[0] / dzt_v[0]
    # --- forcing kernel ---
    maskT_surf = maskT3_v[..., -1]
    forc_temp = (qnet_total + qnec * (sst - T_surf)) * maskT_surf \
        / _CP0_VEROS / _RHO0_VEROS                     # [K m/s]
    mask1 = T_surf * maskT_surf > -1.8
    mask2 = forc_temp > 0.0
    ice = np.logical_or(mask1, mask2)                  # open water = 1
    forc_temp = forc_temp * ice
    temp_source = (
        (qsol * maskT_surf)[..., None]                 # qsol masked at load
        * divpen[None, None, :]
        * ice[..., None]
        * maskT3_v
        / _CP0_VEROS / _RHO0_VEROS
    )                                                  # [K/s]
    rate_v = temp_source.copy()
    rate_v[..., -1] += forc_temp / dzt_v[-1]           # core: forc/dzt[-1]
    return rate_v


def test_matches_veros_global_setup_solar_replica_to_1e14():
    rng = np.random.default_rng(7)
    T = rng.uniform(2.0, 20.0, size=(_NLAT, _NLON, _NLEV))
    T[1, 1, :] = -2.5                                  # ice candidate
    S = rng.uniform(33.0, 36.0, size=(_NLAT, _NLON, _NLEV))
    qnet_total = rng.uniform(-200.0, 200.0, size=(_NLAT, _NLON))
    qnet_total[1, 1] = -150.0                          # cold + cooling -> ice
    qnec = rng.uniform(0.0, 30.0, size=(_NLAT, _NLON))
    qnec[1, 1] = 1.0
    sst = rng.uniform(-2.0, 25.0, size=(_NLAT, _NLON))
    sst[1, 1] = -1.9
    qsol = rng.uniform(0.0, 300.0, size=(_NLAT, _NLON))
    qsol[1, 1] = 30.0                                  # keep total cooling
    # legoESM 3-D wet mask: one land column, one shallow (3 wet levels).
    maskT3 = np.ones((_NLAT, _NLON, _NLEV))
    maskT3[0, 0, :] = 0.0                              # land
    maskT3[2, 3, 3:] = 0.0                             # shallow column
    dz_0 = _DZ_REF[0] * maskT3[..., 0]

    rate_v = _veros_solar_replica(
        maskT3[..., ::-1], T[..., 0], qnet_total, qnec, sst, qsol)
    rate_ref = rate_v[..., ::-1]                       # -> legoESM orientation
    assert np.sum((T[..., 0] * maskT3[..., 0] < -1.8)
                  & (rate_ref[..., 0] <= 0.0)) >= 1, "must exercise ice"

    sf = OceanSurfaceForcing(
        q_prescribed=jnp.asarray(qnet_total - qsol),   # NON-solar remainder
        q_feedback=jnp.asarray(qnec),
        T_feedback_target=jnp.asarray(sst),
        q_solar=jnp.asarray(qsol))
    out = _kernel(jnp.asarray(T), jnp.asarray(S), jnp.asarray(dz_0), sf,
                  _cfg_solar(), maskT3=jnp.asarray(maskT3))
    np.testing.assert_allclose(np.asarray(out.dT_dt), rate_ref,
                               rtol=1e-14, atol=1e-22)


# --------------------------------------------------------------------------
# 5. Guards
# --------------------------------------------------------------------------

def test_q_solar_without_penetrative_option_raises():
    T = jnp.zeros((1, 1, _NLEV))
    S = jnp.zeros((1, 1, _NLEV))
    dz_0 = jnp.full((1, 1), _DZ_REF[0])
    sf = OceanSurfaceForcing(q_solar=jnp.full((1, 1), 100.0))
    with pytest.raises(ValueError, match="penetrative_shortwave"):
        _kernel(T, S, dz_0, sf, _cfg_solar(penetrative_shortwave=False))


def test_q_solar_missing_column_geometry_raises():
    T = jnp.zeros((1, 1, _NLEV))
    S = jnp.zeros((1, 1, _NLEV))
    dz_0 = jnp.full((1, 1), _DZ_REF[0])
    sf = OceanSurfaceForcing(q_solar=jnp.full((1, 1), 100.0))
    with pytest.raises(ValueError, match="z_half_ref"):
        flux_feedback_surface_forcing(T, S, dz_0, sf, _cfg_solar())


def _basin(*, surface_forcing_implicit, scheme="flux_feedback",
           penetrative=True):
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
            scheme=scheme,
            flux_feedback=_cfg_solar(penetrative_shortwave=penetrative)),
    )
    cfg = LatLonCGridOceanConfig(
        A_h=2.0e4, implicit_vertical_mixing=True,
        enable_runtime_checks=False, outer_integrator="ab2",
        surface_forcing_implicit=surface_forcing_implicit, physics=phys,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    sf = OceanSurfaceForcing(
        q_prescribed=jnp.full((n_lat, n_lon), -60.0),  # non-solar remainder
        q_feedback=jnp.full((n_lat, n_lon), 30.0),
        T_feedback_target=jnp.full((n_lat, n_lon), 18.0),
        S_restore_target=jnp.full((n_lat, n_lon), 36.0),
        q_solar=jnp.full((n_lat, n_lon), 180.0),
    )
    return state, model, sf


def test_q_solar_plus_sw_down_raises_double_count_guard():
    st, model, sf = _basin(surface_forcing_implicit=False)
    sf_bad = sf._replace(sw_down=jnp.full(sf.q_solar.shape, 180.0))
    with pytest.raises(ValueError, match="double-count"):
        model.tendencies(st, surface_forcing=sf_bad, dt=3600.0)


def test_q_solar_under_other_scheme_raises():
    st, model, sf = _basin(surface_forcing_implicit=False, scheme="none")
    with pytest.raises(ValueError, match="q_solar"):
        model.tendencies(st, surface_forcing=sf, dt=3600.0)


# --------------------------------------------------------------------------
# 6. Implicit-vs-explicit placement (full-column rate equality)
# --------------------------------------------------------------------------

def test_implicit_and_explicit_placements_carry_same_solar_column():
    dt = 3600.0
    st_e, m_e, sf = _basin(surface_forcing_implicit=False)
    st_i, m_i, _ = _basin(surface_forcing_implicit=True)
    t_e = m_e.tendencies(st_e, surface_forcing=sf, dt=dt)
    t_i = m_i.tendencies(st_i, surface_forcing=sf, dt=dt)

    assert t_e.surface_tracer_forcing is None
    dT_e = np.asarray(t_e.dT_dt.data)                  # rest state: dynamics=0
    # Solar column reaches BELOW the surface layer in the explicit placement.
    assert np.any(np.abs(dT_e[:, :, 1:]) > 0.0)

    assert t_i.surface_tracer_forcing is not None
    # Withheld from the explicit tendencies (every level)...
    assert np.max(np.abs(np.asarray(t_i.dT_dt.data))) < 1e-18
    # ... and routed EXACTLY (full column) onto the implicit seam.
    dT_i = np.asarray(t_i.surface_tracer_forcing.dT_dt.data)
    np.testing.assert_allclose(dT_i, dT_e, rtol=0, atol=0)


def test_implicit_seam_column_sum_is_veros_forc_temp_surface():
    """The TKE surface-buoyancy consumer reconstructs Veros's
    forc_temp_surface as the COLUMN SUM of the implicit rate × dz
    (ocean_model_latlon_cgrid).  Pin the telescoping property at the seam:
    Σ_k dT_dt[...,k]·dz_k == (q_prescribed + feedback·(T*−T) + q_solar)
    / (rho_0·cp) on wet full-depth columns — the solar-INCLUSIVE total
    (the top-cell rate alone would be short by qsol·I(z₁))."""
    from legoesm.ocean.vertical import create_ocean_z_star

    dt = 3600.0
    st_i, m_i, sf = _basin(surface_forcing_implicit=True)
    t_i = m_i.tendencies(st_i, surface_forcing=sf, dt=dt)
    dT_rate = np.asarray(t_i.surface_tracer_forcing.dT_dt.data)
    dz = np.asarray(create_ocean_z_star(n_levels=4, H_max=4000.0).dz_ref)
    col = np.sum(dT_rate * dz, axis=-1)
    lm = np.asarray(st_i.land_mask.data).astype(bool)
    T_surf = np.asarray(st_i.T.data)[..., 0]
    q_total = (-60.0 + 30.0 * (18.0 - T_surf) + 180.0)  # = forc_temp·cp·rho
    expected = q_total / (_RHO0_VEROS * _CP0_VEROS)
    np.testing.assert_allclose(col[lm], expected[lm], rtol=1e-13)
    # And the top-cell rate alone is NOT the total (the I(z₁) deficit).
    top_only = dT_rate[..., 0] * dz[0]
    assert np.all(top_only[lm] < col[lm])


def test_one_step_both_placements_finite_and_heated_at_depth():
    dt = 3600.0
    st_e, m_e, sf = _basin(surface_forcing_implicit=False)
    st_i, m_i, _ = _basin(surface_forcing_implicit=True)
    lm = np.asarray(st_e.land_mask.data).astype(bool)
    T0_deep = np.asarray(st_e.T.data)[lm, 1]
    for st, m in ((st_e, m_e), (st_i, m_i)):
        s = m.step(st, dt=dt, surface_forcing=sf)
        jax.block_until_ready(s.T.data)
        assert bool(jnp.all(jnp.isfinite(s.T.data)))
        # Subsurface solar heating actually arrived at level 1.
        T1_deep = np.asarray(s.T.data)[lm, 1]
        assert np.mean(T1_deep) > np.mean(T0_deep)


# --------------------------------------------------------------------------
# 7. Differentiability
# --------------------------------------------------------------------------

def test_grad_wrt_q_solar_analytic():
    """d(dT_k)/d(q_solar) = frac_k / (rho_0 c_sw dz_k) — the Jerlov fractions."""
    _, frac = _hand_two_band_fracs()

    def level_rate(q, k):
        T = jnp.full((1, 1, _NLEV), 10.0)
        S = jnp.full((1, 1, _NLEV), 35.0)
        sf = OceanSurfaceForcing(q_solar=jnp.full((1, 1), q))
        out = _kernel(T, S, jnp.full((1, 1), _DZ_REF[0]), sf, _cfg_solar())
        return out.dT_dt[0, 0, k]

    for k in (0, 2, _NLEV - 1):
        g = jax.grad(level_rate)(jnp.asarray(150.0), k)
        expected = frac[k] / (_RHO0_VEROS * _CP0_VEROS * _DZ_REF[k])
        np.testing.assert_allclose(float(g), expected, rtol=1e-13)


def test_grad_finite_through_step_wrt_q_solar():
    dt = 3600.0
    state, model, sf = _basin(surface_forcing_implicit=True)

    def loss(q_solar):
        s2 = model.step(state, dt=dt,
                        surface_forcing=sf._replace(q_solar=q_solar))
        return jnp.sum(s2.T.data ** 2)

    g = jax.grad(loss)(sf.q_solar)
    jax.block_until_ready(g)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


# --------------------------------------------------------------------------
# 8. Defaults inert (bit-identity + pytree stability)
# --------------------------------------------------------------------------

def test_q_solar_none_is_bitwise_inert():
    """penetrative_shortwave=True with q_solar=None produces output bitwise
    identical to the option being off entirely."""
    rng = np.random.default_rng(3)
    T = jnp.asarray(rng.uniform(2.0, 20.0, size=(_NLAT, _NLON, _NLEV)))
    S = jnp.asarray(rng.uniform(33.0, 36.0, size=(_NLAT, _NLON, _NLEV)))
    dz_0 = jnp.full((_NLAT, _NLON), _DZ_REF[0])
    sf = OceanSurfaceForcing(
        q_prescribed=jnp.asarray(rng.uniform(-200, 200, size=(_NLAT, _NLON))),
        S_restore_target=jnp.full((_NLAT, _NLON), 35.0))
    out_on = _kernel(T, S, dz_0, sf, _cfg_solar())
    out_off = flux_feedback_surface_forcing(
        T, S, dz_0, sf, _cfg_solar(penetrative_shortwave=False))
    for a, b in zip(out_on, out_off):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_pytree_treedef_stable_with_q_solar():
    q = jnp.full((2, 2), 5.0)
    sf_default = OceanSurfaceForcing(q_prescribed=q)
    sf_spelled = OceanSurfaceForcing(q_prescribed=q, q_solar=None)
    assert (jax.tree_util.tree_structure(sf_default)
            == jax.tree_util.tree_structure(sf_spelled))
    assert len(jax.tree_util.tree_leaves(sf_default)) == 1
    assert OceanSurfaceForcing().q_solar is None
