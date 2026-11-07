"""Soil freeze/thaw (apparent-heat-capacity) validation.

Covers: default-off bit-identity, freezing-curve properties, discrete energy
closure (an exact identity of the flux-form backward-Euler solver), true
nonlinear enthalpy closure (O(dt) convergence, insulated column), the
zero-curtain plateau, config validation, and differentiability.
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import (
    SoilThermalConfig,
    compute_apparent_heat_capacity,
    compute_heat_capacity,
    liquid_water_content,
    solve_soil_thermal,
)

jax.config.update("jax_enable_x64", True)

HYDRO = SoilHydraulicsConfig()


def _column(ncol=1, T0=274.0, theta_frac=0.9):
    grid = make_soil_grid()
    n = grid.dz.shape[0]
    T = jnp.full((ncol, n), T0)
    theta = jnp.full((ncol, n), theta_frac * HYDRO.theta_sat)
    return grid, T, theta, n


def test_default_off_is_sensible_only():
    """enable_freeze_thaw=False is bit-identical sensible-heat diffusion:
    C_eff equals the sensible formula, the solve is INDEPENDENT of the
    freeze/thaw params, and it DIFFERS from the on-path below freezing."""
    grid, T, theta, _ = _column(T0=272.0)  # below freezing: latent would bite if on
    G = jnp.array([10.0])
    off = SoilThermalConfig(enable_freeze_thaw=False)
    # C_eff exactly the sensible-only mixing formula.
    assert jnp.array_equal(
        compute_heat_capacity(theta, HYDRO, off),
        (1.0 - HYDRO.theta_sat) * off.C_soil + theta * off.C_water_vol
        + (HYDRO.theta_sat - theta) * off.C_air,
    )
    T_ref = solve_soil_thermal(T, theta, grid, HYDRO, off, G, dt=3600.0)
    # Off-path output is inert to the freeze/thaw params (proves the gate).
    off2 = SoilThermalConfig(enable_freeze_thaw=False, freeze_curve_width_K=2.0,
                             theta_liq_residual_frac=0.5)
    assert jnp.array_equal(T_ref, solve_soil_thermal(T, theta, grid, HYDRO, off2, G, dt=3600.0))
    # On-path genuinely differs below freezing (the branch does something).
    on = SoilThermalConfig(enable_freeze_thaw=True)
    T_on = solve_soil_thermal(T, theta, grid, HYDRO, on, G, dt=3600.0)
    assert not bool(jnp.allclose(T_ref, T_on))


def test_liquid_curve_properties():
    """theta_liq -> theta warm, -> theta_min cold, monotone, derivative >= 0."""
    cfg = SoilThermalConfig(enable_freeze_thaw=True)
    theta = jnp.full((1, 5), 0.9 * HYDRO.theta_sat)
    # +10 K = 20 sigmoid widths above freezing -> frozen tail ~2e-9 << atol.
    T_warm = jnp.full((1, 5), constants.T_freeze + 10.0)
    T_cold = jnp.full((1, 5), constants.T_freeze - 10.0)
    liq_w, d_w = liquid_water_content(T_warm, theta, cfg)
    liq_c, d_c = liquid_water_content(T_cold, theta, cfg)
    theta_min = cfg.theta_liq_residual_frac * theta
    assert jnp.allclose(liq_w, theta, atol=1e-6)
    assert jnp.allclose(liq_c, theta_min, atol=1e-6)
    assert jnp.all(d_w >= 0.0) and jnp.all(d_c >= 0.0)
    Ts = jnp.linspace(constants.T_freeze - 3.0, constants.T_freeze + 3.0, 40)
    liq = jax.vmap(lambda t: liquid_water_content(
        jnp.full((1,), t), jnp.full((1,), 0.9 * HYDRO.theta_sat), cfg)[0])(Ts)
    assert jnp.all(jnp.diff(liq[:, 0]) >= -1e-12)


def test_latent_enthalpy_total():
    """Total fusion enthalpy = rho_water*L_f*(theta - theta_min), independent of
    the curve width (the latent apparent-HC integrates to the full L_f)."""
    theta = jnp.array([[0.9 * HYDRO.theta_sat]])
    for w in (0.25, 0.5, 1.0):
        cfg = SoilThermalConfig(enable_freeze_thaw=True, freeze_curve_width_K=w)
        Ts = jnp.linspace(constants.T_freeze - 8.0 * w, constants.T_freeze + 8.0 * w, 200000)
        dT = Ts[1] - Ts[0]
        dliq = jax.vmap(lambda t: liquid_water_content(
            jnp.full((1, 1), t), theta, cfg)[1][0, 0])(Ts)
        integral = jnp.sum(dliq) * dT * constants.rho_water * constants.L_f
        expected = constants.rho_water * constants.L_f * (
            theta[0, 0] - cfg.theta_liq_residual_frac * theta[0, 0])
        assert jnp.abs(integral - expected) / expected < 1e-3


def test_discrete_energy_closes():
    """Exact flux-form identity: sum C_app*dz*(T_new-T_old) == net boundary
    flux * dt (machine precision) with freeze/thaw active."""
    grid, T, theta, _ = _column(T0=constants.T_freeze + 0.2)
    G = jnp.array([-30.0])
    cfg = SoilThermalConfig(enable_freeze_thaw=True, Q_geothermal=0.0)
    dt = 1800.0
    C_app = compute_apparent_heat_capacity(T, theta, HYDRO, cfg)  # at T_old
    T_new = solve_soil_thermal(T, theta, grid, HYDRO, cfg, G, dt=dt)
    stored = jnp.sum(C_app * grid.dz[None, :] * (T_new - T), axis=1)
    assert jnp.allclose(stored, (G + cfg.Q_geothermal) * dt, rtol=1e-9, atol=1e-6)


def _enthalpy_change_per_cell(T_i, T_f, theta, cfg, n=400):
    """Per-cell true enthalpy change int_{T_i}^{T_f} C_app(T';theta) dT'
    (trapezoid), the nonlinear enthalpy the scheme should conserve."""
    s = jnp.linspace(0.0, 1.0, n)
    ds = s[1] - s[0]
    Tg = T_i[..., None] + (T_f - T_i)[..., None] * s          # (ncol, nlayers, n)
    Capp = jax.vmap(
        lambda k: compute_apparent_heat_capacity(Tg[..., k], theta, HYDRO, cfg),
        in_axes=0, out_axes=-1,
    )(jnp.arange(n))                                           # (ncol, nlayers, n)
    trap = (jnp.sum(Capp, axis=-1) - 0.5 * Capp[..., 0] - 0.5 * Capp[..., -1]) * ds
    return trap * (T_f - T_i)


def test_enthalpy_convergence():
    """Insulated column (zero flux) conserves TRUE enthalpy; the linearised
    apparent-HC scheme drifts O(dt), so a 4x finer dt gives a smaller drift."""
    grid, _, theta, n = _column(theta_frac=1.0)
    T0 = jnp.linspace(constants.T_freeze - 1.0, constants.T_freeze + 1.0, n)[None, :]
    cfg = SoilThermalConfig(enable_freeze_thaw=True, Q_geothermal=0.0)
    G = jnp.zeros((1,))
    t_end = 6.0 * 3600.0

    def drift(dt):
        nsteps = int(round(t_end / dt))
        T = T0
        for _ in range(nsteps):
            T = solve_soil_thermal(T, theta, grid, HYDRO, cfg, G, dt=dt)
        dH = jnp.sum(grid.dz[None, :] * _enthalpy_change_per_cell(T0, T, theta, cfg))
        return float(jnp.abs(dH))

    coarse = drift(t_end / 15)
    fine = drift(t_end / 60)          # 4x finer
    assert fine < coarse              # O(dt) convergence of true enthalpy


def test_zero_curtain_plateau():
    """A saturated column cooled through freezing lingers near T_freeze WITH
    freeze/thaw (latent release) vs cooling straight through WITHOUT it."""
    grid, T, theta, _ = _column(T0=constants.T_freeze + 0.5, theta_frac=1.0)
    G = jnp.array([-40.0])
    dt = 1800.0
    on = SoilThermalConfig(enable_freeze_thaw=True, Q_geothermal=0.0)
    off = SoilThermalConfig(enable_freeze_thaw=False, Q_geothermal=0.0)
    T_on, T_off = T, T
    for _ in range(40):
        T_on = solve_soil_thermal(T_on, theta, grid, HYDRO, on, G, dt=dt)
        T_off = solve_soil_thermal(T_off, theta, grid, HYDRO, off, G, dt=dt)
    assert T_on[0, 0] > T_off[0, 0] + 1.0
    assert jnp.abs(T_on[0, 0] - constants.T_freeze) < 3.0
    liq, _ = liquid_water_content(T_on, theta, on)
    assert jnp.any(theta - liq > 1e-3)


def _true_enthalpy_change(T0, T1, theta, cfg):
    """Latent part exact, sensible part trapezoid (sensible C varies slowly)."""
    def sens(T):
        tl, _ = liquid_water_content(T, theta, cfg)
        ti = jnp.maximum(theta - tl, 0.0)
        return ((1 - HYDRO.theta_sat) * cfg.C_soil + tl * cfg.C_water_vol
                + ti * cfg.C_ice_vol + (HYDRO.theta_sat - theta) * cfg.C_air)
    l0, _ = liquid_water_content(T0, theta, cfg)
    l1, _ = liquid_water_content(T1, theta, cfg)
    return (0.5 * (sens(T0) + sens(T1)) * (T1 - T0)
            + constants.rho_water * constants.L_f * (l1 - l0))


@pytest.mark.parametrize("T0,G", [(275.0, -30.0), (275.0, -100.0),
                                  (275.0, -270.0), (262.0, 100.0)])
def test_production_grid_curtain_crossing_energy(T0, G):
    """AMIP production soil (10 layers to 3 m, 2.9 mm top layer), land step
    300 s, 10 days of one-signed forcing through the curtain (freeze and thaw).
    The start-of-step apparent heat capacity lets a thin layer overshoot the
    curtain in single steps (worst step measured 2 / 28 / 319 W/m2 at
    |G| = 30 / 100 / 270), leaving a small one-signed episode error: measured
    +0.003 / +0.07 / +0.35 W/m2, i.e. <= 0.13 % of the forcing.  Tolerance
    0.3 % of |G|."""
    from legoesm.land.soil_grid import SoilGridConfig
    grid = make_soil_grid(SoilGridConfig(n_layers=10, total_depth=3.0))
    n = grid.dz.shape[0]
    theta = jnp.full((1, n), 0.30)
    cfg = SoilThermalConfig(enable_freeze_thaw=True)
    dt, nsteps = 300.0, 12 * 24 * 10
    flux = (G + cfg.Q_geothermal) * dt

    def body(T, _):
        Tn = solve_soil_thermal(T, theta, grid, HYDRO, cfg, jnp.array([G]), dt=dt)
        return Tn, jnp.sum(_true_enthalpy_change(T, Tn, theta, cfg) * grid.dz) - flux

    T_end, errs = jax.jit(lambda T: jax.lax.scan(body, T, None, length=nsteps))(
        jnp.full((1, n), T0))
    assert jnp.all(jnp.isfinite(T_end))
    # the episode really crossed the curtain in the top layers
    assert (T_end[0, 0] - constants.T_freeze) * (T0 - constants.T_freeze) < 0
    assert abs(float(errs.sum()) / (nsteps * dt)) < 3e-3 * abs(G)


def test_config_validation():
    """freeze_curve_width_K <= 0 raises (div-by-zero / sign-reversal guard)."""
    theta = jnp.full((1, 3), 0.3)
    T = jnp.full((1, 3), constants.T_freeze)
    with pytest.raises(ValueError):
        liquid_water_content(T, theta, SoilThermalConfig(freeze_curve_width_K=0.0))
    with pytest.raises(ValueError):
        liquid_water_content(T, theta, SoilThermalConfig(freeze_curve_width_K=-1.0))


def test_differentiable():
    """jax.grad flows through the freeze/thaw solve (AD-safe)."""
    grid, T, theta, _ = _column(T0=constants.T_freeze + 0.1)
    cfg = SoilThermalConfig(enable_freeze_thaw=True)

    def loss(G_scalar):
        return jnp.mean(solve_soil_thermal(
            T, theta, grid, HYDRO, cfg, jnp.array([G_scalar]), dt=1800.0))

    g = jax.grad(loss)(-20.0)
    assert jnp.isfinite(g) and g != 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
