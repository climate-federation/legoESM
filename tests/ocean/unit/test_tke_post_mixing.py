"""Unit tests for the POST-MIXING TKE step order
(``TKEConfig.buoyancy_timing="post_mixing_veros"``) and companions.

Veros solves the TKE budget AFTER the implicit T/S vertical mixing
(veros.py:263-285): the kappa profiles consumed by the TRACER solve come from
the PREVIOUS step's TKE (``set_tke_diffusivities`` at tau, tke.py:20-113) and
the TKE forcing charges ``P_diss_v = kappaH·Nsqr[taup1]`` — the POST-mixing
stratification (thermodynamics.py:385) — plus the surface buoyancy-flux
``P_diss_v`` slot (386-388 + surf_densityf 304-317) and the REALIZED
implicit-friction dissipation ``K_diss_v`` (friction.py:131-151).

Covers:
- validation (unknown literals / missing prerequisites fail loudly; the
  pre-mixing orchestrator rejects the post-mixing config);
- the surface buoyancy-flux term vs the Veros formula (linear EOS, exact);
- the realized K_diss_v vs the hand-computed ``κ·g_new·g_old``;
- a manufactured convective column: PRE-mixing N² < 0, POST-mixing N² ≥ 0,
  and the solved TKE matches an independent numpy transcription of the Veros
  kernel (nlev W rows incl. the surface half-volume) to machine precision —
  the source difference vs charging the pre-mixing N² is hand-computable;
- ordering: the tracer solve under post-mixing uses kappa from the CARRIED
  (previous-step) TKE;
- the additive-friction K_diss_v threading (momentum-only call returns the
  realized field);
- AD: gradients are finite through the reordered model step;
- the EKE-side adiabatic-N² dzw slot (``EKEConfig.n2_over_dzw``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    TKEPostMixingContext,
    compute_surface_buoyancy_P_diss_v,
    realized_implicit_friction_dissipation,
    tke_integrate_post_mixing,
    tke_set_diffusivities,
    tke_vertical_mixing,
)

# Post-mixing prerequisites (the faithful-recipe class).
PM_CFG = TKEConfig(
    n2_mode="adiabatic", veros_dz_slots=True,
    positivity="veros_surface_correction", kappa_convention="veros_sqrte",
    prandtl_mode="constant", Prandtl_tke0=10.0, prognostic=True,
    buoyancy_timing="post_mixing_veros", shear_production="pre_solve",
)

RHO_0 = 1024.0
G = 9.81


def _linear_eos(alpha=0.2, beta=0.8, gamma=4.0e-6):
    """rho = 1000 - alpha*T + beta*S + gamma*p (hand-differentiable)."""
    def eos_fn(T, S, p):
        return 1000.0 - alpha * T + beta * S + gamma * p
    return eos_fn


def _u_centered_coord(dz):
    """OceanZStarCoordinate with the Veros u_centered centres (dzw != midpoint)."""
    from legoesm.ocean.fidelity.veros_state_bridge import (
        veros_u_centered_z_centres,
    )
    from legoesm.ocean.vertical import OceanZStarCoordinate
    dz = jnp.asarray(dz, dtype=jnp.float64)
    z_half = jnp.concatenate([jnp.zeros(1, dtype=dz.dtype), -jnp.cumsum(dz)])
    z_full = jnp.asarray(veros_u_centered_z_centres(np.asarray(dz)),
                         dtype=dz.dtype)
    dz_half = jnp.abs(z_full[:-1] - z_full[1:])
    return OceanZStarCoordinate(
        n_levels=int(dz.shape[0]), H_max=float(jnp.sum(dz)),
        z_full_ref=z_full, z_half_ref=z_half, dz_ref=dz, dz_half_ref=dz_half)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_buoyancy_timing_rejected():
    cfg = PM_CFG._replace(buoyancy_timing="typo")
    with pytest.raises(ValueError, match="buoyancy_timing"):
        tke_integrate_post_mixing(
            None, None, None, None, dt=1.0, cfg=cfg)  # validated first


def test_unknown_shear_production_rejected():
    cfg = PM_CFG._replace(shear_production="typo")
    with pytest.raises(ValueError, match="shear_production"):
        tke_integrate_post_mixing(None, None, None, None, dt=1.0, cfg=cfg)


@pytest.mark.parametrize("bad", [
    dict(prognostic=False),
    dict(veros_dz_slots=False),
    dict(n2_mode="insitu"),
    dict(positivity="floor"),
])
def test_post_mixing_prerequisites_enforced(bad):
    cfg = PM_CFG._replace(**bad)
    with pytest.raises(ValueError, match="post_mixing_veros"):
        tke_integrate_post_mixing(None, None, None, None, dt=1.0, cfg=cfg)


def test_realized_requires_post_mixing():
    cfg = PM_CFG._replace(buoyancy_timing="pre_mixing",
                          shear_production="realized_veros")
    with pytest.raises(ValueError, match="realized_veros"):
        tke_integrate_post_mixing(None, None, None, None, dt=1.0, cfg=cfg)


def test_pre_mixing_orchestrator_rejects_post_mixing_config():
    """tke_vertical_mixing IS the pre-mixing solve — it must fail loudly."""
    nlev = 4
    one = jnp.ones((1, 1, nlev))
    with pytest.raises(ValueError, match="post_mixing_veros"):
        tke_vertical_mixing(
            one, one, one, 35.0 * one, 1025.0 * one,
            jnp.ones((1, 1, nlev - 1)) * 10.0,
            tke_old=jnp.full((1, 1, nlev - 1), 1e-6),
            tau_x_surface=None, tau_y_surface=None,
            dt=100.0, cfg=PM_CFG, rho_0=RHO_0, g=G)


# ---------------------------------------------------------------------------
# Surface buoyancy-flux P_diss_v slot (Veros thermodynamics.py:304-317,386-388)
# ---------------------------------------------------------------------------


def test_surface_buoyancy_flux_matches_veros_formula():
    alpha, beta = 0.2, 0.8
    eos_fn = _linear_eos(alpha=alpha, beta=beta)
    T_sfc = jnp.asarray([[10.0, 5.0]])
    S_sfc = jnp.asarray([[35.0, 34.0]])
    p_sfc = jnp.asarray([[1.0e5, 1.0e5]])
    forc_T = jnp.asarray([[2.0e-4, -3.0e-4]])   # [K·m/s]
    forc_S = jnp.asarray([[1.0e-5, 0.0]])       # [PSU·m/s]
    got = compute_surface_buoyancy_P_diss_v(
        T_sfc, S_sfc, p_sfc, forc_T, forc_S, eos_fn, rho_0=RHO_0, g=G)
    # Veros: forc_rho_surface = drhodT·forc_T + drhodS·forc_S;
    # P_diss_v_sfc = -(g/rho_0)·forc_rho_surface. Linear EOS: drhodT=-alpha,
    # drhodS=+beta, independent of state.
    want = -(G / RHO_0) * (-alpha * forc_T + beta * forc_S)
    # rtol bounded by the EOS compute-precision policy (the derivative is
    # evaluated at the policy dtype, float32 under the default test policy).
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6)
    # Sign physics: surface COOLING (forc_T < 0) must be a TKE SOURCE
    # (P_diss_v < 0 so forc = -P_diss_v > 0).
    assert float(got[0, 1]) < 0.0


def test_surface_buoyancy_flux_finite_at_land_fill_gsw():
    """REGRESSION (global_4deg day-5 blowup): the TEOS-10/gsw EOS carries
    sqrt(S) terms whose S-derivative is NaN/inf at the land-fill S=0 (and
    S<0); NaN·mask does not mask NaN. The salinity floor must keep the
    surface term finite at S=0 and S<0 evaluation points."""
    from legoesm.ocean.eos import make_eos_fn
    eos_fn = make_eos_fn("veros_gsw", None)
    T_sfc = jnp.asarray([[0.0, 2.0, 10.0]])
    S_sfc = jnp.asarray([[0.0, -1.0e-6, 35.0]])     # land fill / transient / wet
    p_sfc = jnp.zeros_like(T_sfc)
    forc_T = jnp.full_like(T_sfc, 1.0e-4)
    forc_S = jnp.full_like(T_sfc, 1.0e-5)
    out = compute_surface_buoyancy_P_diss_v(
        T_sfc, S_sfc, p_sfc, forc_T, forc_S, eos_fn, rho_0=RHO_0, g=G)
    assert bool(jnp.all(jnp.isfinite(out))), out


# ---------------------------------------------------------------------------
# Realized implicit-friction K_diss_v (Veros friction.py:131-151)
# ---------------------------------------------------------------------------


def test_realized_kdiss_hand_computed():
    u_old = jnp.asarray([[[0.3, 0.1, -0.05]]])
    u_new = jnp.asarray([[[0.22, 0.12, -0.02]]])
    A_f = jnp.asarray([[[1.0e-3, 5.0e-4]]])
    dzw = jnp.asarray([[[10.0, 20.0]]])
    got = realized_implicit_friction_dissipation(u_old, u_new, A_f, dzw)
    g_old = (np.array([0.3, 0.1]) - np.array([0.1, -0.05])) / np.array([10., 20.])
    g_new = (np.array([0.22, 0.12]) - np.array([0.12, -0.02])) / np.array([10., 20.])
    want = np.array([1.0e-3, 5.0e-4]) * g_new * g_old
    np.testing.assert_allclose(np.asarray(got)[0, 0], want, rtol=1e-13)


# ---------------------------------------------------------------------------
# The manufactured convective column + the independent Veros-kernel reference
# ---------------------------------------------------------------------------


def _column_setup():
    """Stretched u_centered 4-level column with a surface inversion."""
    dz = np.array([20.0, 28.0, 40.0, 56.0])
    zc = _u_centered_coord(dz)
    J = jnp.ones((1, 1))
    eos_fn = _linear_eos(gamma=0.0)  # pressure-independent: N2 sign = -dT sign
    T = jnp.asarray([[[4.0, 8.0, 6.0, 5.0]]])     # cold over warm: N2[0] < 0
    S = jnp.full((1, 1, 4), 35.0)
    p = jnp.asarray(
        RHO_0 * G * (-np.asarray(zc.z_full_ref)))[None, None, :]
    dz_half = (zc.dz_half_ref * J[..., None]).reshape(1, 1, 3)
    dz_surface = (-zc.z_full_ref[0]) * J
    return zc, J, eos_fn, T, S, p, dz_half, dz_surface


def _veros_kernel_post_mixing(ctx, N2_post, K_diss_v, P_sfc, dt, cfg):
    """Independent numpy transcription of Veros integrate_tke (tke.py:185-227)
    with nlev W rows top-down (row 0 = the surface half-volume point)."""
    e_int = np.asarray(ctx.tke_old)[0, 0]
    kM = np.asarray(ctx.K_M_old)[0, 0]
    kH = np.asarray(ctx.K_H_old)[0, 0]
    mxl = np.asarray(ctx.mxl)[0, 0]
    sq = np.asarray(ctx.sqrttke)[0, 0]
    dzw = np.asarray(ctx.dz_half)[0, 0]
    dzt = np.asarray(ctx.dz_cell)[0, 0]
    n = dzt.shape[0]
    forc_int = np.asarray(K_diss_v)[0, 0] - kH * np.asarray(N2_post)[0, 0]
    forc = np.concatenate([[-float(P_sfc)], forc_int])
    e_w = np.concatenate([e_int[:1], e_int])
    kM_w = np.concatenate([kM[:1], kM])
    sq_w = np.concatenate([sq[:1], sq])
    mxl_w = np.concatenate([mxl[:1], mxl])
    vol = np.concatenate([[float(ctx.dz_surface[0, 0])], dzw])
    delta = dt * cfg.alpha_tke * 0.5 * (kM_w[:-1] + kM_w[1:]) / dzt[:n - 1]
    a = np.zeros(n)
    c = np.zeros(n)
    a[1:] = -delta / vol[1:]
    c[:-1] = -delta / vol[:-1]
    b = 1.0 - (a + c) + dt * cfg.c_eps * sq_w / np.maximum(mxl_w, cfg.mxl_min)
    d = e_w + dt * forc
    d[0] += dt * float(ctx.surface_flux[0, 0]) / vol[0]
    # Thomas
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, n):
        den = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / den
        dp[k] = (d[k] - a[k] * dp[k - 1]) / den
    x = np.zeros(n)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x[1:]


def test_convective_column_post_mixing_n2_and_kernel_match():
    zc, J, eos_fn, T, S, p, dz_half, dz_surface = _column_setup()
    u = jnp.zeros_like(T)
    tke_old = jnp.full((1, 1, 3), 2.0e-4)
    tau0 = jnp.zeros((1, 1))
    K_M, K_H, ctx = tke_set_diffusivities(
        u, u, T, S, eos_fn(T, S, p), dz_half, tke_old, tau0, tau0,
        PM_CFG, RHO_0, G, p_cell=p, dz_ref=zc.dz_ref, jacobian=J,
        eos_fn=eos_fn, z_interface=zc.z_half_ref[1:-1], dz_surface=dz_surface)

    # PRE-mixing N2: surface inversion -> negative at interface 0.
    N2_pre = compute_buoyancy_frequency_adiabatic(
        T, S, p, zc.dz_ref, J, eos_fn=eos_fn, rho_ref=RHO_0, g=G,
        dz_half=dz_half)
    assert float(N2_pre[0, 0, 0]) < 0.0

    # Implicit tracer mixing with kappa from the CARRIED tke (phase 1), big
    # enough dt·K to remove the inversion -> POST-mixing N2 >= 0 there.
    from legoesm.ocean.physics.vertical_mixing import (
        implicit_vertical_diffusion_ocean,
    )
    dz_cell = (zc.dz_ref * J[..., None]).reshape(1, 1, 4)
    dt_tr = 43200.0
    K_strong = jnp.maximum(K_H, 1.0)   # convective-class kappa (kappaM_max regime)
    T_mixed = implicit_vertical_diffusion_ocean(
        T, K_strong, dz_cell, dz_half, dt_tr)
    N2_post = compute_buoyancy_frequency_adiabatic(
        T_mixed, S, p, zc.dz_ref, J, eos_fn=eos_fn, rho_ref=RHO_0, g=G,
        dz_half=dz_half)
    # The backward-Euler solve drives the inversion toward neutral (never
    # exactly zero in one step): >= 99% of the instability must be gone.
    assert (abs(float(N2_post[0, 0, 0]))
            < 0.01 * abs(float(N2_pre[0, 0, 0]))), \
        "implicit solve must stabilise the inversion"
    # The hand-computable source difference vs charging the pre-mixing N2:
    src_diff = np.asarray(ctx.K_H_old * (N2_pre - N2_post))[0, 0]
    assert src_diff[0] < 0.0  # pre-mixing charge is MORE production at k=0

    # Solved TKE == the independent Veros-kernel transcription (machine prec).
    P_sfc = jnp.zeros((1, 1))
    kdiss = ctx.K_M_old * ctx.shear_sq
    dt_tke = 4800.0
    got = tke_integrate_post_mixing(
        ctx, N2_post, kdiss, P_sfc, dt=dt_tke, cfg=PM_CFG)
    want = _veros_kernel_post_mixing(ctx, N2_post, kdiss, 0.0, dt_tke, PM_CFG)
    np.testing.assert_allclose(np.asarray(got)[0, 0], want, rtol=1e-12)

    # And it must DIFFER from charging the pre-mixing N2 by a hand-computable
    # amount in the forcing (regression teeth for the timing itself).
    got_pre = tke_integrate_post_mixing(
        ctx, N2_pre, kdiss, P_sfc, dt=dt_tke, cfg=PM_CFG)
    assert not np.allclose(np.asarray(got), np.asarray(got_pre))


def test_surface_flux_injected_at_surface_w_row():
    """The wind-work injection enters the surface half-volume row and reaches
    the carried interfaces only via the implicit coupling (Veros tke.py:225)."""
    zc, J, eos_fn, T, S, p, dz_half, dz_surface = _column_setup()
    u = jnp.zeros_like(T)
    T_stable = jnp.asarray([[[10.0, 8.0, 6.0, 5.0]]])
    tke_old = jnp.full((1, 1, 3), 2.0e-4)
    tau = jnp.full((1, 1), 0.1)
    K_M, K_H, ctx = tke_set_diffusivities(
        u, u, T_stable, S, eos_fn(T_stable, S, p), dz_half, tke_old, tau,
        jnp.zeros((1, 1)), PM_CFG, RHO_0, G, p_cell=p, dz_ref=zc.dz_ref,
        jacobian=J, eos_fn=eos_fn, z_interface=zc.z_half_ref[1:-1],
        dz_surface=dz_surface)
    N2 = compute_buoyancy_frequency_adiabatic(
        T_stable, S, p, zc.dz_ref, J, eos_fn=eos_fn, rho_ref=RHO_0, g=G,
        dz_half=dz_half)
    got = tke_integrate_post_mixing(
        ctx, N2, ctx.K_M_old * ctx.shear_sq, jnp.zeros((1, 1)),
        dt=4800.0, cfg=PM_CFG)
    want = _veros_kernel_post_mixing(
        ctx, N2, ctx.K_M_old * ctx.shear_sq, 0.0, 4800.0, PM_CFG)
    np.testing.assert_allclose(np.asarray(got)[0, 0], want, rtol=1e-12)
    # With wind on, the top carried interface must gain TKE vs wind off.
    ctx_off = ctx._replace(surface_flux=jnp.zeros((1, 1)))
    got_off = tke_integrate_post_mixing(
        ctx_off, N2, ctx.K_M_old * ctx.shear_sq, jnp.zeros((1, 1)),
        dt=4800.0, cfg=PM_CFG)
    assert float(got[0, 0, 0]) > float(got_off[0, 0, 0])


# ---------------------------------------------------------------------------
# Model-step ordering + threading (small 3-D model)
# ---------------------------------------------------------------------------


def _small_model(tke_cfg, outer="forward_euler", extra=None):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z_coord = create_ocean_z_star(n_levels=6, H_max=2000.0)
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke_cfg),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    kw = dict(A_h=1.0e4, implicit_vertical_mixing=True, physics=physics,
              outer_integrator=outer)
    if extra:
        kw.update(extra)
    cfg = LatLonCGridOceanConfig.from_flat(**kw)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0,
        H_max=2000.0)
    rng = np.random.default_rng(1)
    s = s._replace(
        u=s.u.replace(data=jnp.asarray(0.05 * rng.standard_normal(
            s.u.data.shape))),
        v=s.v.replace(data=jnp.asarray(0.05 * rng.standard_normal(
            s.v.data.shape))),
        T=s.T.replace(data=s.T.data + jnp.asarray(
            0.2 * rng.standard_normal(s.T.data.shape))))
    tke0 = jnp.full(s.T.data.shape[:-1] + (z_coord.n_levels - 1,),
                    1.0e-5, dtype=s.T.data.dtype)
    s = s._replace(tke=Field(data=tke0, name="tke",
                             dims=("lat", "lon", "level"), units="m^2/s^2"))
    return model, s, z_coord


def test_tracer_kappa_comes_from_carried_tke():
    """ORDERING: under post-mixing the implicit tracer solve must use kappa
    derived from the PREVIOUS step's TKE (state.tke), reproducible by an
    independent solve with K from compute_vertical_K_profiles' phase-1."""
    from legoesm.ocean.physics.vertical_mixing import (
        compute_vertical_K_profiles, implicit_vertical_diffusion_ocean,
        build_dz_half,
    )
    from legoesm.ocean.vertical import compute_ocean_jacobian
    from legoesm.ocean.eos import make_eos_fn

    model, s, zc = _small_model(PM_CFG)
    # Double the carried TKE -> phase-1 kappa changes -> T output must change
    # (the kappa is a function of the CARRIED field, not of a fresh solve).
    out1, tke1 = model._apply_implicit_vertical_mixing(
        s, 3600.0, None, tke_old=s.tke.data, return_tke=True)
    s2 = s._replace(tke=s.tke.replace(data=200.0 * s.tke.data))
    out2, tke2 = model._apply_implicit_vertical_mixing(
        s2, 3600.0, None, tke_old=s2.tke.data, return_tke=True)
    assert not np.allclose(np.asarray(out1.T.data), np.asarray(out2.T.data))

    # Exact reproduction: T_new == independent implicit solve with the
    # phase-1 K profiles (K from tke_old) + the config background.
    u_c = 0.5 * (s.u.data[:, :-1, :] + s.u.data[:, 1:, :])
    v_c = 0.5 * (s.v.data[:-1, :, :] + s.v.data[1:, :, :])
    cc = s._replace(u=s.u.replace(data=u_c), v=s.v.replace(data=v_c))
    eos_fn = make_eos_fn(eos=model.config.eos,
                         eos_linear=model.config.eos_linear)
    K_v, A_v, ctx = compute_vertical_K_profiles(
        cc, zc, None, model.config.physics,
        A_v_background=float(model.config.A_v),
        K_v_background=float(model.config.K_v),
        eos_fn=eos_fn, tke_old=s.tke.data, dt_tke=3600.0, return_tke=True)
    assert isinstance(ctx, TKEPostMixingContext)
    J = compute_ocean_jacobian(s.eta.data, s.H_bathy.data, zc)
    dz_cell = zc.dz_ref * J[..., None]
    dz_half_cell = build_dz_half(dz_cell)
    # K_v from compute_vertical_K_profiles already includes the config
    # background (K_v_background) — the model's fallback path applies it as-is.
    T_want = implicit_vertical_diffusion_ocean(
        s.T.data, K_v.astype(s.T.data.dtype),
        dz_cell, dz_half_cell, 3600.0)
    T_want = jnp.where(s.land_mask.data[..., None] > 0.5, T_want, s.T.data)
    np.testing.assert_allclose(np.asarray(out1.T.data), np.asarray(T_want),
                               rtol=1e-12, atol=1e-12)


def test_momentum_only_call_returns_realized_kdiss():
    cfg = PM_CFG._replace(shear_production="realized_veros")
    model, s, zc = _small_model(cfg)
    out = model._apply_implicit_vertical_mixing(
        s, 3600.0, None, dt_mom=400.0, do_tracers=False,
        tke_old=s.tke.data, return_K_diss_v=True)
    state_fric, kdv = out
    assert kdv is not None and kdv.shape == s.tke.data.shape
    assert bool(jnp.all(jnp.isfinite(kdv)))
    # Sanity: the implicit solve damps shear, so kappa*g_new*g_old >= 0 a.e.
    assert float(jnp.mean((kdv >= 0).astype(jnp.float64))) > 0.95


def test_realized_kdiss_missing_raises():
    cfg = PM_CFG._replace(shear_production="realized_veros")
    model, s, zc = _small_model(cfg)
    with pytest.raises(ValueError, match="realized"):
        model._apply_implicit_vertical_mixing(
            s, 3600.0, None, do_momentum=False,
            tke_old=s.tke.data, return_tke=True)


def test_ad_finite_through_post_mixing_step():
    cfg = PM_CFG._replace(shear_production="realized_veros")
    model, s, zc = _small_model(cfg)

    def loss(T0):
        st = s._replace(T=s.T.replace(data=T0))
        st1 = model._step_impl(st, 3600.0)
        return (jnp.mean(st1.T.data ** 2) + jnp.mean(st1.u.data ** 2)
                + jnp.mean(st1.tke.data ** 2))

    g = jax.grad(loss)(s.T.data)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


# ---------------------------------------------------------------------------
# EKE-side adiabatic-N² dzw slot (EKEConfig.n2_over_dzw)
# ---------------------------------------------------------------------------


def test_eke_n2_over_dzw_slot():
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        _eady_growth_and_length,
    )
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig

    zc, J2, eos_fn, T, S, p, dz_half, _dzs = _column_setup()
    T = jnp.asarray([[[10.0, 8.0, 6.0, 5.0]]])
    rho = eos_fn(T, S, p)
    J = jnp.ones((1, 1))
    Sx = jnp.full((1, 1, 3), 1.0e-4)
    Sy = jnp.zeros((1, 1, 3))
    f = jnp.full((1, 1), 1.0e-4)
    cfg = VisbeckConfig()

    def run(n2_over_dzw):
        sigma_bar, L, wet, intN, sigma, _dzh = _eady_growth_and_length(
            rho, Sx, Sy, zc, J, f, cfg, RHO_0, n2_mode="adiabatic",
            n2_over_dzw=n2_over_dzw, T=T, S=S, p_cell=p, eos_fn=eos_fn)
        return np.asarray(sigma)[0, 0]

    sig_dzw = run(True)
    sig_mid = run(False)
    # Independent references: N over the u_centered dzw vs the midpoint
    # metric. The Eady chain fixes g = constants.g internally.
    from legoesm import constants as _const
    N2_dzw = np.asarray(compute_buoyancy_frequency_adiabatic(
        T, S, p, zc.dz_ref, J, eos_fn=eos_fn, rho_ref=RHO_0, g=_const.g,
        dz_half=dz_half))[0, 0]
    N2_mid = np.asarray(compute_buoyancy_frequency_adiabatic(
        T, S, p, zc.dz_ref, J, eos_fn=eos_fn, rho_ref=RHO_0, g=_const.g))[0, 0]
    np.testing.assert_allclose(
        sig_dzw, np.sqrt(np.maximum(N2_dzw, 1e-30)) * np.sqrt(1e-8 + 1e-30),
        rtol=1e-6)
    np.testing.assert_allclose(
        sig_mid, np.sqrt(np.maximum(N2_mid, 1e-30)) * np.sqrt(1e-8 + 1e-30),
        rtol=1e-6)
    # The u_centered coordinate genuinely distinguishes the slots.
    assert not np.allclose(sig_dzw, sig_mid)


def test_orchestrator_consumes_post_mixing_n2():
    """Orchestrator-level pin (review coverage gap): the model step's carried
    TKE must equal the reference computed with the POST-mixing N2 (T after the
    implicit tracer solve) and clearly differ from the PRE-mixing reference —
    a regression to ``T_n2 = state.T`` at the orchestration site
    (ocean_model_latlon_cgrid.py:~2903) goes red here. Construction from the
    review probe (.physics-validator/tke_postmix_review/attack9_n2timing_gap.py):
    a rest-state surface inversion (cold over warm), no forcing, u=v=0 — so the
    step's T_new is the implicit vertical solve alone and the function-level
    reference reproduces the orchestrator exactly.
    """
    import numpy as np
    from legoesm.core.field import Field
    from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic, make_eos_fn
    from legoesm.ocean.physics.vertical_mixing.tke import tke_integrate_post_mixing
    from legoesm.ocean.physics.vertical_mixing import (
        compute_vertical_K_profiles, implicit_vertical_diffusion_ocean,
        build_dz_half,
    )
    from legoesm.ocean.vertical import compute_ocean_jacobian, create_ocean_z_star
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        TKEConfig, VerticalMixingConfig,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    dt = 4800.0
    pm = TKEConfig(
        n2_mode="adiabatic", veros_dz_slots=True,
        positivity="veros_surface_correction", kappa_convention="veros_sqrte",
        prandtl_mode="constant", prognostic=True,
        buoyancy_timing="post_mixing_veros", shear_production="pre_solve")
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=pm),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None)
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    zc = create_ocean_z_star(n_levels=6, H_max=2000.0)
    cfg = LatLonCGridOceanConfig.from_flat(A_h=1e4, implicit_vertical_mixing=True,
                                 physics=physics)
    model = LatLonCGridOceanModel(grid, zc, cfg)
    s = rest_state_latlon_cgrid_ocean(grid, zc, T_water_init_C=2.0,
                                      T_deep=18.0, S_uniform=35.0,
                                      H_max=2000.0)
    tke0 = jnp.full(s.T.data.shape[:-1] + (zc.n_levels - 1,), 1e-4,
                    dtype=s.T.data.dtype)
    s = s._replace(tke=Field(data=tke0, name="tke",
                             dims=("lat", "lon", "level"), units="m^2/s^2"))

    s1 = model.step(s, dt)
    carried = np.asarray(s1.tke.data)

    # Function-level references reproducing the orchestrator (rest state, no
    # forcing => T_new is the implicit vertical solve alone).
    cc = s._replace(
        u=s.u.replace(data=0.5 * (s.u.data[:, :-1, :] + s.u.data[:, 1:, :])),
        v=s.v.replace(data=0.5 * (s.v.data[:-1, :, :] + s.v.data[1:, :, :])))
    eos_fn = make_eos_fn(eos=cfg.eos, eos_linear=cfg.eos_linear)
    K_v, A_v, ctx = compute_vertical_K_profiles(
        cc, zc, None, physics, A_v_background=float(cfg.A_v),
        K_v_background=float(cfg.K_v), eos_fn=eos_fn, tke_old=s.tke.data,
        dt_tke=dt, return_tke=True)
    J = compute_ocean_jacobian(s.eta.data, s.H_bathy.data, zc)
    dz_cell = zc.dz_ref * J[..., None]
    dz_half = build_dz_half(dz_cell)
    T_new = implicit_vertical_diffusion_ocean(
        s.T.data, K_v.astype(s.T.data.dtype), dz_cell, dz_half, dt)

    def n2_of(T):
        return compute_buoyancy_frequency_adiabatic(
            T, s.S.data, ctx.p_cell, zc.dz_ref, J, eos_fn=ctx.eos_fn,
            rho_ref=ctx.rho_0, g=ctx.g, dz_half=ctx.dz_half)

    kdiss = ctx.K_M_old * ctx.shear_sq
    psfc = jnp.zeros(s.T.data.shape[:-1])
    ref_post = np.asarray(tke_integrate_post_mixing(
        ctx, n2_of(T_new), kdiss, psfc, dt, pm))
    ref_pre = np.asarray(tke_integrate_post_mixing(
        ctx, n2_of(s.T.data), kdiss, psfc, dt, pm))

    wet = np.asarray(s.land_mask.data)[..., None] > 0.5
    wet = np.broadcast_to(wet, carried.shape)
    # The discrimination signal is ~1.9e-2; the reproduction error is ~0.
    assert np.max(np.abs((carried - ref_post)[wet])) < 1e-6, \
        "carried TKE does not match the post-mixing-N2 reference"
    assert np.max(np.abs((carried - ref_pre)[wet])) > 1e-3, \
        "pre/post references coincide - test construction is vacuous"


def test_nemo_dirichlet_surface_bc_production_path():
    """surface_bc='nemo_dirichlet' through the PRODUCTION route
    (tke_set_diffusivities -> ctx.surface_dirichlet -> tke_integrate_post_mixing).

    Geometric note (review 2026-07-16): this path holds the DISCARDED z=0
    surface W row at e_sfc (the more NEMO-faithful location); the CARRIED top
    interface couples to it through the surface half-volume and lands slightly
    BELOW e_sfc — unlike tke_vertical_mixing, which pins the carried top
    interface exactly. Assert: (a) ctx carries the exact NEMO value, (b) the
    carried surface TKE is pulled to the e_sfc scale, far above the Veros
    flux-BC response.
    """
    zc, J, eos_fn, T, S, p, dz_half, dz_surface = _column_setup()
    u = jnp.zeros_like(T)
    tke_old = jnp.full((1, 1, 3), 2.0e-4)
    tau = jnp.full((1, 1), 0.1)          # |tau| = 0.1 Pa wind
    tau0 = jnp.zeros((1, 1))
    e_sfc = max(1.0e-4, 67.83 / RHO_0 * 0.1)

    def run(cfg):
        _, _, ctx = tke_set_diffusivities(
            u, u, T, S, eos_fn(T, S, p), dz_half, tke_old, tau, tau0,
            cfg, RHO_0, G, p_cell=p, dz_ref=zc.dz_ref, jacobian=J,
            eos_fn=eos_fn, z_interface=zc.z_half_ref[1:-1],
            dz_surface=dz_surface)
        n2 = jnp.zeros_like(dz_half)
        tke_new = tke_integrate_post_mixing(
            ctx, n2, jnp.zeros_like(dz_half), jnp.zeros((1, 1)),
            dt=43200.0, cfg=cfg)
        return ctx, tke_new

    ctx_d, tke_d = run(PM_CFG._replace(surface_bc="nemo_dirichlet"))
    assert ctx_d.surface_dirichlet is not None
    np.testing.assert_allclose(
        np.asarray(ctx_d.surface_dirichlet), e_sfc, rtol=1e-12)

    ctx_f, tke_f = run(PM_CFG)
    assert ctx_f.surface_dirichlet is None

    top_d = float(tke_d[0, 0, 0])
    top_f = float(tke_f[0, 0, 0])
    # pulled to the e_sfc scale (coupled, not exact) and >> the flux response
    assert 0.3 * e_sfc < top_d <= 1.01 * e_sfc
    assert top_d > 5.0 * top_f


def test_nn_mxl_choices_3_and_4_end_to_end_anchor_and_jacobian():
    """END-TO-END cover for the two guards nn_mxl=2 (choice 4) changed.

    The dedicated choice-4 unit tests inject ``l_surface_anchor`` and
    ``dz_cell`` directly, so they never exercise (a) the ``(3, 4)`` guard in the
    ln_mxl0 anchor helper or (b) the ``(3, 4)`` guard that builds ``dz_cell``
    from ``dz_ref``/``jacobian`` (codex).  Both live behind
    ``tke_set_diffusivities``, so drive them from there.

    Failure modes this catches:
      * a missed anchor guard -> choice 4 becomes WIND-INSENSITIVE at the
        surface (identical K for very different stress);
      * a missed geometry guard -> a raise, or silently reference thicknesses.
    """
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

    zc, J, eos_fn, T, S, p, dz_half, dz_surface = _column_setup()
    u = jnp.zeros_like(T)
    tke_old = jnp.full((1, 1, 3), 2.0e-4)
    calm = jnp.zeros((1, 1))
    windy = jnp.full((1, 1), 0.4)          # |tau| = 0.4 N/m^2

    def _run(cfg, taum):
        # taum_surface is the NEMO stress-modulus channel the anchor reads.
        return tke_set_diffusivities(
            u, u, T, S, eos_fn(T, S, p), dz_half, tke_old, None, None,
            cfg, RHO_0, G, taum_surface=taum, p_cell=p,
            dz_ref=zc.dz_ref, jacobian=J, eos_fn=eos_fn,
            z_interface=zc.z_half_ref[1:-1], dz_surface=dz_surface)

    for choice in (3, 4):
        cfg = PM_CFG._replace(tke_mxl_choice=choice)
        # (b) the dz_ref/jacobian geometry path must work WITHOUT dz_cell
        K_calm, _, _ = _run(cfg, calm)
        K_wind, _, _ = _run(cfg, windy)
        assert bool(jnp.all(jnp.isfinite(K_calm))), choice
        assert bool(jnp.all(jnp.isfinite(K_wind))), choice
        # (a) the ln_mxl0 anchor must respond to wind stress for BOTH choices
        assert not jnp.allclose(K_calm, K_wind), (
            f"tke_mxl_choice={choice}: K is wind-insensitive at the surface, "
            "so the ln_mxl0 anchor guard is not firing for this choice")

    # How the two choices must and must NOT differ, under identical forcing.
    K3, H3, ctx3 = _run(PM_CFG._replace(tke_mxl_choice=3), windy)
    K4, H4, ctx4 = _run(PM_CFG._replace(tke_mxl_choice=4), windy)

    # (i) the EDDY COEFFICIENTS must be IDENTICAL at a fixed TKE.  avm/avt use
    # l_k, which choices 3 and 4 share (it is computed BEFORE the split), so a
    # difference here would mean l_k was disturbed -- not the intended change.
    # This encodes the correction that nn_mxl=2 does not directly "mix less":
    # it changes DISSIPATION, and a shallower mixed layer is a coupled
    # consequence over time, not an instantaneous algebraic one.  (An earlier
    # version of this test asserted K3 != K4 and failed for exactly that
    # reason -- the code was right and the assertion was wrong.)
    assert jnp.allclose(K3, K4, rtol=0.0, atol=0.0), (
        "l_k must be identical between nn_mxl=3 and nn_mxl=2")
    assert jnp.allclose(H3, H4, rtol=0.0, atol=0.0)

    # (ii) the DISSIPATION length must differ, and choice 4's must be smaller
    # (min(lup,ldn) <= sqrt(lup*ldn)).  This is the fallthrough detector.
    assert ctx3.l_eps is not None and ctx4.l_eps is not None, \
        "ctx.l_eps is required to tell the two schemes apart end-to-end"
    assert not jnp.allclose(ctx3.l_eps, ctx4.l_eps), (
        "choice 4 produced the choice-3 dissipation length end-to-end -- "
        "it fell through to nn_mxl=3")
    assert bool(jnp.all(ctx4.l_eps <= ctx3.l_eps + 1e-12))
