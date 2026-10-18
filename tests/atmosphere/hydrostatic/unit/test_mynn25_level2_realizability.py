"""MYNN-2.5 level-2 q2 from NN09 eq A1 (Helfand-Labraga alpha_c engages without shear).

With no resolved shear the old eq-A2 form ``B1 L^2 SM2 (1-Rf) S^2`` collapsed to
the qke floor, so alpha_c stayed 1, G_H at floor qke was ~1e7, SH25 went negative
and the Kh >= 0 clamp zeroed buoyancy production: a windless convective column
never left the floor. Eq A1 ``B1 q^2 (SM2 G_M + SH2 G_H)`` restores it and is
identical where the Ri -> Rf relation is live and unclipped (values below were
produced by the eq-A2 code).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    compute_rho,
    exner_function,
)
from legoesm.atmosphere.physics.turbulence.config import MYNN25Config
from legoesm.atmosphere.physics.turbulence.integration import (
    fold_prescribed_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.mynn25 import (
    _compute_SM_SH,
    mynn25_turbulence,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    surface_fluxes_at_lowest_level,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_specific_humidity

from legoesm import constants

jax.config.update("jax_enable_x64", True)

NLEV = 32
_SHFLX = 70.0     # W/m^2, ~ the Nieuwstadt CBL's 0.06 K m/s


def test_shear_free_unstable_state_keeps_sh_positive():
    # measured windless-CBL interface: SH25 was -7.7e-10 here before the fix
    sm, sh = _compute_SM_SH(jnp.zeros((1, 1)), jnp.full((1, 1), 3.956e7),
                            jnp.full((1, 1), 85.9), jnp.full((1, 1), 1e-5), MYNN25Config())
    assert float(sh[0, 0]) > 0.0 and float(sm[0, 0]) >= 0.0
    assert np.isfinite(float(sh[0, 0]))


@pytest.mark.parametrize("g_m, g_h, q, sm_ref, sh_ref", [
    (0.001, -1.0, 0.001, 0.19172061111164834, 0.02919686711739747),
    (0.1, -1.0, 0.001, 0.15252511705214308, 0.02607936182495483),
    (1.0, 0.0, 0.001, 0.1202289482612237, 0.16247646912420216),
    (0.001, 0.01, 0.3, 0.7492373975234176, 0.8452835817556567),
    (0.001, 0.03, 0.3, 0.9006675130050619, 1.4516988593949316),
    (1.0, 0.03, 0.3, 0.12201944615803771, 0.1680832680077125),
])
def test_sheared_states_unchanged(g_m, g_h, q, sm_ref, sh_ref):
    sm, sh = _compute_SM_SH(jnp.full((1, 1), g_m), jnp.full((1, 1), g_h),
                            jnp.full((1, 1), 50.0), jnp.full((1, 1), q), MYNN25Config())
    np.testing.assert_allclose([float(sm[0, 0]), float(sh[0, 0])], [sm_ref, sh_ref],
                               rtol=1e-12)


def _column(u_wind, qke0):
    """Top-first windless/sheared column: well mixed theta = 300 K, unstable
    0.5 K at the lowest level, prescribed sensible heat flux."""
    sc = create_sigma_coordinate(NLEV, sigma_top=0.8)
    p_s = jnp.asarray(1.0e5)
    p_full = jnp.asarray(sc.pressure_at_full(p_s), jnp.float64)[None]   # sigma is f32
    p_half = jnp.asarray(sc.pressure_at_half(p_s), jnp.float64)[None]
    theta = jnp.full((1, NLEV), 300.0).at[:, -1].add(0.5)
    temp = theta * exner_function(p_full)
    q_v = jnp.zeros_like(temp)
    z_full, z_half = compute_heights_from_sigma(temp, p_half, q_v=q_v)
    rho = compute_rho(temp, p_full, q_v=q_v)
    u = jnp.full_like(temp, u_wind)
    cfg = fold_prescribed_surface_fluxes(
        MYNN25Config(), shflx_w_m2=jnp.asarray([_SHFLX]), lhflx_w_m2=jnp.zeros(1),
        evap_kg_m2_s=jnp.zeros(1))
    t_sfc = temp[:, -1]
    q_sfc = saturation_specific_humidity(t_sfc, p_full[:, -1])
    return dict(u=u, v=jnp.zeros_like(temp), T=temp, q_v=q_v,
                qke=jnp.full_like(temp, qke0), p_full=p_full, p_half=p_half,
                z_full=z_full, z_half=z_half, T_sfc=t_sfc, q_sfc=q_sfc, rho=rho), cfg


def test_windless_convective_column_grows_tke_from_the_floor():
    col, cfg = _column(0.0, 1e-10)
    dt = 10.0
    for _ in range(120):
        out, qke = mynn25_turbulence(**col, dt=dt, config=cfg)
        col = {**col, "T": col["T"] + dt * out.dT_dt, "qke": qke}
    interior = np.asarray(col["qke"])[0, :-1]          # top-first; [-1] = surface reset
    assert interior.max() > 1e-3, interior.max()


@pytest.mark.parametrize("u_wind, qke0", [(0.0, 1e-4), (10.0, 0.3)])
def test_heat_and_momentum_budgets_close(u_wind, qke0):
    col, cfg = _column(u_wind, qke0)
    out, _ = mynn25_turbulence(**col, dt=10.0, config=cfg)
    rho = col["rho"]
    dz_raw = jnp.abs(col["z_half"][:, :-1] - col["z_half"][:, 1:])
    assert float(jnp.min(dz_raw)) > 1.0      # the kernel's 1 m layer floor is inert
    dz = dz_raw
    exner = exner_function(col["p_full"])
    # theta-form diffusion: sum(rho dz dtheta/dt) = shflx / (c_p Pi_lowest)
    d_theta = jnp.sum(rho * dz * out.dT_dt / exner)
    f_theta = out.shflx[0] / constants.c_pd / exner[0, -1]
    assert abs(float(d_theta - f_theta)) < 1e-10 * abs(float(f_theta))
    tau_x = surface_fluxes_at_lowest_level(
        col["u"][:, -1], col["v"][:, -1], col["T"][:, -1], col["q_v"][:, -1],
        col["T_sfc"], col["q_sfc"], rho[:, -1], cfg.surface,
        col["z_full"][:, -1] - col["z_half"][:, -1])[0]
    d_mom = jnp.sum(rho * dz * out.du_dt)
    assert abs(float(d_mom - tau_x[0])) <= 1e-10 * max(abs(float(tau_x[0])), 1e-12)


def test_ad_matches_finite_difference_on_a1_in_the_alpha_c_regime():
    col, cfg = _column(0.0, 1e-4)

    def loss(a1):
        out, qke = mynn25_turbulence(**col, dt=10.0, config=cfg._replace(A1=a1))
        return jnp.sum(qke[:, :-1]) + jnp.sum(out.dT_dt ** 2)
    a1 = float(cfg.A1)
    g = float(jax.grad(loss)(a1))
    h = 1e-6 * a1
    fd = (float(loss(a1 + h)) - float(loss(a1 - h))) / (2 * h)
    assert g != 0.0 and abs(g / fd - 1.0) < 1e-4, (g, fd)
