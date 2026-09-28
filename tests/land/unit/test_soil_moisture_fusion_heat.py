"""Fusion heat of ice changed by soil-water movement at fixed temperature.

With freeze/thaw on, the apparent heat capacity charges latent heat only for
ice that changes with T.  Richards moves water at fixed T, which changes the
diagnosed ice ``theta - theta_liq(T, theta)`` with no fusion heat.  These tests
pin the explicit per-layer source that closes the solver's linearised identity

    sum_k dz [C_app(T0, th1) (T1 - T0) - rho_w L_f (ice(T0, th1) - ice(T0, th0))]
        = dt (G + Q_geo - lambda (T1_top - T0_top))

both for the thermal kernel and for the full multilayer land step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import init_multilayer_land_state, step_multilayer_land
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import (
    SoilThermalConfig,
    compute_apparent_heat_capacity,
    liquid_water_content,
    moisture_fusion_heat_source,
    solve_soil_thermal,
)

jax.config.update("jax_enable_x64", True)

HYDRO = SoilHydraulicsConfig()
ON = SoilThermalConfig(enable_freeze_thaw=True)
_RLF = constants.rho_water * constants.L_f


def _ice(T, th, cfg=ON):
    return th - liquid_water_content(T, th, cfg)[0]


def _residual(T0, T1, th0, th1, dz, G, lam, dt, cfg=ON):
    """Linearised energy residual [J/m2] and the moisture fusion term [J/m2]."""
    C = compute_apparent_heat_capacity(T0, th1, HYDRO, cfg)
    stored = jnp.sum(C * dz * (T1 - T0), axis=1)
    fusion = _RLF * jnp.sum(dz * (_ice(T0, th1, cfg) - _ice(T0, th0, cfg)), axis=1)
    flux = (G + cfg.Q_geothermal - lam * (T1[:, 0] - T0[:, 0])) * dt
    return stored - fusion - flux, fusion


def _kernel_case(T0_val, dtheta):
    grid = make_soil_grid()
    n = grid.dz.shape[0]
    T0 = jnp.full((1, n), T0_val)
    th0 = jnp.full((1, n), 0.6 * HYDRO.theta_sat)
    th1 = th0 + dtheta * jnp.linspace(1.0, -0.5, n)[None, :]  # wets top, dries bottom
    return grid, T0, th0, th1


@pytest.mark.parametrize("T0_val", [constants.T_freeze - 5.0, constants.T_freeze - 0.2])
def test_kernel_closes_with_moisture_driven_ice_change(T0_val):
    grid, T0, th0, th1 = _kernel_case(T0_val, 0.02)
    G, dt = jnp.array([-15.0]), 1800.0
    src = moisture_fusion_heat_source(T0, th0, th1, grid.dz, ON, dt)
    T1 = solve_soil_thermal(T0, th1, grid, HYDRO, ON, G, dt, layer_source=src)
    res, fusion = _residual(T0, T1, th0, th1, grid.dz, G, 0.0, dt)
    assert abs(float(fusion[0])) > 1.0e3            # the term is not a zero
    assert abs(float(res[0])) < 1.0e-9 * abs(float(fusion[0])) + 1.0e-6
    # Without the source the same step misses exactly the fusion term.
    T1_leak = solve_soil_thermal(T0, th1, grid, HYDRO, ON, G, dt)
    res_leak, _ = _residual(T0, T1_leak, th0, th1, grid.dz, G, 0.0, dt)
    assert jnp.allclose(res_leak, -fusion, rtol=1e-9)


def test_source_sign_freezing_new_water_warms():
    """Water added to a deeply frozen layer freezes and releases heat."""
    grid, T0, th0, _ = _kernel_case(constants.T_freeze - 5.0, 0.0)
    th1 = th0.at[:, 0].add(0.02)
    src = moisture_fusion_heat_source(T0, th0, th1, grid.dz, ON, 1800.0)
    assert float(src[0, 0]) > 0.0
    assert jnp.all(src[0, 1:] == 0.0)


def test_fixed_moisture_is_bit_identical():
    grid, T0, th0, _ = _kernel_case(constants.T_freeze - 0.2, 0.0)
    src = moisture_fusion_heat_source(T0, th0, th0, grid.dz, ON, 1800.0)
    assert jnp.all(src == 0.0)
    G = jnp.array([-15.0])
    a = solve_soil_thermal(T0, th0, grid, HYDRO, ON, G, 1800.0, layer_source=src)
    b = solve_soil_thermal(T0, th0, grid, HYDRO, ON, G, 1800.0)
    assert jnp.array_equal(a, b)


def test_mixed_precision_unchanged_water_gives_zero():
    grid, T0, th0, _ = _kernel_case(constants.T_freeze - 0.2, 0.0)
    th32 = (th0 + 1.0e-3).astype(jnp.float32)
    src = moisture_fusion_heat_source(T0, th32, th32.astype(jnp.float64), grid.dz, ON, 1800.0)
    assert jnp.all(src == 0.0)


def test_source_jit_and_grad_finite():
    grid, T0, th0, th1 = _kernel_case(constants.T_freeze - 0.2, 0.02)

    def f(th, detach=False):
        src = moisture_fusion_heat_source(T0, th0, th, grid.dz, ON, 1800.0)
        src = jax.lax.stop_gradient(src) if detach else src
        return jnp.sum(solve_soil_thermal(
            T0, th, grid, HYDRO, ON, jnp.array([-15.0]), 1800.0, layer_source=src))

    assert jnp.allclose(f(th1), jax.jit(f)(th1), rtol=1e-12)
    g = jax.grad(f)(th1)
    assert bool(jnp.all(jnp.isfinite(g)))
    # the source itself carries gradient (not only C_app / conductivity)
    assert float(jnp.max(jnp.abs(g - jax.grad(lambda th: f(th, True))(th1)))) > 1.0e-3


# ── full multilayer land step ────────────────────────────────────────────────
def _forcing(ncol, *, T_air, q_air, precip):
    o = jnp.ones(ncol)
    p_s = 1.0e5 * o
    return AtmToSurface(
        sw_down=150.0 * o, lw_down=280.0 * o, precip_total=precip * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=q_air * o,
        u_lowest=4.0 * o, v_lowest=0.0 * o, p_lowest=0.99 * p_s, p_surface=p_s,
        rho_lowest=p_s / (constants.R_d * T_air), cos_zenith=0.5 * o,
        co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)


def _captured_step(monkeypatch, cfg, forcing, T_init, theta_init, dt=1800.0):
    calls = []
    real = multilayer_land.solve_soil_thermal

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append((args, kwargs, out))
        return out

    monkeypatch.setattr(multilayer_land, "solve_soil_thermal", spy)
    st = init_multilayer_land_state(1, cfg, T_init=T_init, theta_init=theta_init)
    new, _, _ = step_multilayer_land(st, forcing, cfg, 1.0, dt, lat=jnp.full(1, 0.9))
    return st, new, calls[-1]


_ARMS = {
    # rain onto deeply frozen soil: infiltrating water freezes
    "rain_on_frozen": dict(T_init=constants.T_freeze - 4.0, theta_init=0.20,
                           forcing=dict(T_air=276.0, q_air=0.004, precip=2.0e-4)),
    # near-freezing column draining and evaporating (ice leaves)
    "drain_near_freezing": dict(T_init=constants.T_freeze - 0.2, theta_init=0.40,
                                forcing=dict(T_air=280.0, q_air=0.001, precip=0.0)),
    # heavy pulse onto frozen soil (stress: large source)
    "heavy_pulse": dict(T_init=constants.T_freeze - 1.0, theta_init=0.15,
                        forcing=dict(T_air=277.0, q_air=0.005, precip=2.0e-3)),
}


def _exact_residual(T0, T1, th0, th1, dz, G, dt, cfg, n=4000):
    """Enthalpy residual [J/m2] of a step: the exact integral of C_app over the
    temperature change at fixed th1, minus the moisture fusion term, minus the
    boundary heat (surface conductance None)."""
    s = (jnp.arange(n) + 0.5) / n
    Tq = T0[..., None] + (T1 - T0)[..., None] * s
    Cq = jax.vmap(lambda Tk: compute_apparent_heat_capacity(Tk, th1, HYDRO, cfg),
                  in_axes=-1, out_axes=-1)(Tq)
    dH = jnp.sum(jnp.mean(Cq, -1) * (T1 - T0) * dz, axis=1)
    fusion = _RLF * jnp.sum(dz * (_ice(T0, th1, cfg) - _ice(T0, th0, cfg)), axis=1)
    return dH - fusion - (G + cfg.Q_geothermal) * dt, fusion


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_full_step_closes_soil_energy(monkeypatch, arm):
    a = _ARMS[arm]
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0),
                               thermal=ON)
    st, new, (args, kwargs, T1) = _captured_step(
        monkeypatch, cfg, _forcing(1, **a["forcing"]), a["T_init"], a["theta_init"])
    T0, th1, grid, _, _, G, dt = args
    # Final solve: start-of-step T, post-Richards water, result accepted.
    assert jnp.array_equal(T0, st.T_soil)
    assert jnp.array_equal(th1.astype(new.theta_soil.dtype), new.theta_soil)
    assert jnp.array_equal(T1.astype(new.T_soil.dtype), new.T_soil)
    assert kwargs.get("surface_conductance") is None
    assert kwargs["n_substeps"] == multilayer_land.FINAL_THERMAL_SUBSTEPS > 1
    dz = jnp.asarray(grid.dz)
    assert jnp.allclose(kwargs["layer_source"], moisture_fusion_heat_source(
        T0, st.theta_soil, th1, dz, cfg.thermal, dt), rtol=1e-12)
    res, fusion = _exact_residual(T0, T1, st.theta_soil, th1, dz, G, dt, cfg.thermal)
    assert abs(float(fusion[0])) > 1.0e3, fusion       # water moved, ice changed
    # The same solve as one 1800 s step overshoots the curtain: measured first-
    # step residuals 44 / 44 / 0.3 W/m2 sub-stepped vs 998 / 389 / 4.4 in one
    # step (rain / heavy / drain).
    one = solve_soil_thermal(*args, **{**kwargs, "n_substeps": 1})
    res1, _ = _exact_residual(T0, one, st.theta_soil, th1, dz, G, dt, cfg.thermal)
    assert abs(float(res[0])) < 0.5 * abs(float(res1[0])), (res, res1)


def test_full_step_off_passes_no_source_and_one_step(monkeypatch):
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0))
    assert not cfg.thermal.enable_freeze_thaw
    _, _, (_, kwargs, _) = _captured_step(
        monkeypatch, cfg, _forcing(1, **_ARMS["rain_on_frozen"]["forcing"]),
        constants.T_freeze - 4.0, 0.20)
    assert kwargs.get("layer_source") is None
    assert kwargs["n_substeps"] == 1


def test_one_substep_is_the_single_step_solve():
    grid, T0, th0, th1 = _kernel_case(constants.T_freeze - 0.2, 0.02)
    G = jnp.array([-15.0])
    src = moisture_fusion_heat_source(T0, th0, th1, grid.dz, ON, 1800.0)
    a = solve_soil_thermal(T0, th1, grid, HYDRO, ON, G, 1800.0, layer_source=src)
    b = solve_soil_thermal(T0, th1, grid, HYDRO, ON, G, 1800.0, layer_source=src,
                           n_substeps=1)
    assert jnp.array_equal(a, b)
    # Sub-steps conserve the linearised identity sub-step by sub-step: with
    # moisture fixed and freeze/thaw off, n sub-steps deliver exactly G*dt.
    off = SoilThermalConfig(enable_freeze_thaw=False)
    T6 = solve_soil_thermal(T0, th1, grid, HYDRO, off, G, 1800.0, n_substeps=6)
    from legoesm.land.soil_thermal import compute_heat_capacity
    stored = jnp.sum(compute_heat_capacity(th1, HYDRO, off) * grid.dz * (T6 - T0), axis=1)
    assert jnp.allclose(stored, (G + off.Q_geothermal) * 1800.0, rtol=1e-9)
