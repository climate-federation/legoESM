"""P-hydro (Joshi profit strategy on the SPA supply): pure-module behaviour.

Supply magnitudes and drought collapse, optimality certificates
(stationarity + profit dominance over perturbations), monotone responses
(drier soil -> lower capacities; larger gamma -> smaller draw-down), slope
mappings for all three stomatal models, differentiability, loud no-state
error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land import phydro as ph
from legoesm.land.p_model import PModelConfig, init_pmodel_acclim, optimal_chi
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig


def _acclim(ncol=1, t_c=25.0, vpd=1000.0):
    a = init_pmodel_acclim(ncol, t_init_K=constants.T_freeze + t_c,
                           ps_init_pa=constants.p_atm_std,
                           cfg=PModelConfig())
    return a._replace(vpd_mean_pa=jnp.full((ncol,), vpd))


def _supply(psi_s=-0.2, lsc=2e-3, ncol=1):
    return ph.PhydroSupply(
        lsc_mol=jnp.full((ncol,), lsc),
        psi_s_mpa=jnp.full((ncol,), psi_s))


def _soil_column(psi_m=-5.0, nlev=6, ncol=2):
    hyd = SoilHydraulicsConfig()
    psi = jnp.full((ncol, nlev), psi_m)
    theta = jnp.full((ncol, nlev), 0.3)
    dz = jnp.full((nlev,), 0.25)
    rf = jnp.full((ncol, nlev), 1.0 / nlev)
    lai = jnp.full((ncol,), 3.0)
    return psi, theta, dz, rf, lai, hyd


def test_supply_wet_soil_is_xylem_limited_and_dry_collapses():
    cfg = ph.PHydroConfig()
    psi, theta, dz, rf, lai, hyd = _soil_column(psi_m=-1.0)  # wet (~-0.01 MPa)
    sup = ph.soil_root_supply(psi, theta, dz, rf, lai, hyd, cfg)
    assert np.all(np.isfinite(np.asarray(sup.lsc_mol)))
    # Wet soil: whole-plant conductance approaches (but never exceeds) gplant.
    assert float(sup.lsc_mol[0]) < cfg.gplant_mol
    assert float(sup.lsc_mol[0]) > 0.2 * cfg.gplant_mol
    assert float(sup.psi_s_mpa[0]) > -0.2
    # Very dry soil: psi_s collapses toward minlwp and lsc drops.
    psi_d, theta_d, *_ = _soil_column(psi_m=-300.0)  # ~ -2.9 MPa
    sup_d = ph.soil_root_supply(psi_d, jnp.full_like(theta_d, 0.12), dz, rf,
                                lai, hyd, cfg)
    assert float(sup_d.psi_s_mpa[0]) <= cfg.minlwp_mpa + 0.2
    assert float(sup_d.lsc_mol[0]) < float(sup.lsc_mol[0])


def test_optimum_certificates_and_regime():
    cfg = ph.PHydroConfig()
    pcfg = PModelConfig()
    acclim = _acclim()
    sup = _supply()
    caps = ph.phydro_optimum(acclim, sup, cfg, pcfg)
    v = float(caps.vcmax25_leaf[0])
    assert 1.0 < v < 200.0
    assert 0.05 < float(caps.chi[0]) < 0.98
    d_star = float(caps.dpsi_mpa[0])
    assert 0.0 < d_star < float(sup.psi_s_mpa[0]) - cfg.minlwp_mpa + 0.2

    # Optimality certificate: the returned dpsi beats +/- 10% perturbations
    # of the SAME profit (GLM review: verify stationarity, not just output).
    def profit_at(dpsi):
        gs = sup.lsc_mol * dpsi / (1.6 * jnp.maximum(
            acclim.vpd_mean_pa, pcfg.vpd_min_pa) / acclim.ps_ema)
        _, xi, ci0, gs_pa, K = optimal_chi(
            acclim.t_mean_K, acclim.vpd_mean_pa, acclim.co2_mean_ppm,
            acclim.ps_ema, pcfg)
        del xi, ci0
        from legoesm.land.p_model import C_STAR, _phi0, _smooth_floor
        ca_pa = acclim.co2_mean_ppm * 1e-6 * acclim.ps_ema
        chi = ph._chi_supply_demand(
            gs, acclim.t_mean_K, ca_pa, gs_pa, _phi0(acclim.t_mean_K, pcfg),
            acclim.iabs_mean, acclim.ps_ema, pcfg)
        ci = chi * ca_pa
        mj = (ci - gs_pa) / (ci + 2.0 * gs_pa)
        mjs = _smooth_floor(mj, C_STAR + pcfg.mj_floor_eps, pcfg.mj_floor_width)
        k = (C_STAR / mjs) ** (2.0 / 3.0)
        a = _phi0(acclim.t_mean_K, pcfg) * acclim.iabs_mean * mjs * jnp.sqrt(1.0 - k)
        return float((a - cfg.gamma_cost * dpsi ** 2)[0])

    p_star = profit_at(jnp.full((1,), d_star))
    assert p_star >= profit_at(jnp.full((1,), d_star * 0.9)) - 1e-6
    assert p_star >= profit_at(jnp.full((1,), d_star * 1.1)) - 1e-6


def test_drought_monotone_and_smooth_to_zero():
    cfg = ph.PHydroConfig()
    pcfg = PModelConfig()
    acclim = _acclim()
    v = []
    for psi_s in (-0.1, -0.8, -1.5, -1.9, -1.99):
        caps = ph.phydro_optimum(acclim, _supply(psi_s=psi_s), cfg, pcfg)
        v.append(float(caps.vcmax25_leaf[0]))
        assert np.isfinite(v[-1])
    assert all(a >= b for a, b in zip(v, v[1:]))  # drier -> lower capacity
    assert v[-1] < 0.15 * v[0]                     # near-collapse at minlwp


def test_gamma_reduces_drawdown():
    pcfg = PModelConfig()
    acclim = _acclim()
    d1 = float(ph.phydro_optimum(
        acclim, _supply(), ph.PHydroConfig(gamma_cost=0.5), pcfg).dpsi_mpa[0])
    d2 = float(ph.phydro_optimum(
        acclim, _supply(), ph.PHydroConfig(gamma_cost=5.0), pcfg).dpsi_mpa[0])
    assert d2 < d1


def test_slope_mappings_reproduce_target_gs():
    """Each model's gs with the mapped slope equals 1.6*A/(ca*(1-chi))."""
    from legoesm.land.stomata import ball_berry_gs, leuning_gs, medlyn_gs
    from legoesm.thermo import saturation_vapor_pressure
    acclim = _acclim()
    pcfg = PModelConfig()
    chi = jnp.full((1,), 0.75)
    _, _, _, gamma_star, _ = optimal_chi(
        acclim.t_mean_K, acclim.vpd_mean_pa, acclim.co2_mean_ppm,
        acclim.ps_ema, pcfg)
    A = jnp.full((1,), 12.0)
    ca_ppm = acclim.co2_mean_ppm
    gs_target = 1.6 * A / (ca_ppm * (1.0 - chi))
    d_kpa = acclim.vpd_mean_pa / 1000.0
    es = saturation_vapor_pressure(acclim.t_mean_K)
    rh = 1.0 - acclim.vpd_mean_pa / es
    gamma_ppm = gamma_star / (1e-6 * acclim.ps_ema)

    g1 = ph.slope_for_model("medlyn", chi, acclim, gamma_star)
    gs_med = medlyn_gs(A, d_kpa, ca_ppm, g1, 0.0)
    np.testing.assert_allclose(np.asarray(gs_med), np.asarray(gs_target),
                               rtol=1e-6)
    m = ph.slope_for_model("ball_berry", chi, acclim, gamma_star)
    gs_bb = ball_berry_gs(A, rh, ca_ppm, m, 0.0)
    np.testing.assert_allclose(np.asarray(gs_bb), np.asarray(gs_target),
                               rtol=1e-6)
    a1 = ph.slope_for_model("leuning", chi, acclim, gamma_star,
                            d0_leuning_kpa=1.5)
    gs_le = leuning_gs(A, d_kpa, ca_ppm, gamma_ppm, a1, 1.5, 0.0)
    np.testing.assert_allclose(np.asarray(gs_le), np.asarray(gs_target),
                               rtol=1e-6)
    with pytest.raises(ValueError, match="stomatal_model"):
        ph.slope_for_model("wue", chi, acclim, gamma_star)
    with pytest.raises(ValueError, match="d0_leuning"):
        ph.slope_for_model("leuning", chi, acclim, gamma_star)


def test_gradients_finite_through_optimum():
    pcfg = PModelConfig()
    acclim = _acclim()

    def loss(gamma, psi_s, lsc):
        caps = ph.phydro_optimum(
            acclim, ph.PhydroSupply(lsc_mol=jnp.full((1,), lsc),
                                    psi_s_mpa=jnp.full((1,), psi_s)),
            ph.PHydroConfig(gamma_cost=gamma), pcfg)
        return caps.vcmax25_leaf[0] + caps.rjv25[0] + caps.dpsi_mpa[0]

    g = jax.grad(loss, argnums=(0, 1, 2))(1.0, -0.5, 2e-3)
    assert all(np.isfinite(float(x)) for x in g)
    # Near the minlwp corner too (smooth cap).
    g2 = jax.grad(loss, argnums=(0, 1, 2))(1.0, -1.97, 2e-3)
    assert all(np.isfinite(float(x)) for x in g2)


def test_no_state_raises():
    with pytest.raises(ValueError, match="acclimation state"):
        ph.phydro_optimum(None, _supply(), ph.PHydroConfig(), PModelConfig())
