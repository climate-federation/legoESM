"""Frozen soil impedes water flow: CLM5 ice impedance 10**(-e * icefrac).

The multiplier reaches the Richards solve as a per-layer log factor: interface
conductivity (log-form geometric mean), surface infiltration (top layer's
factor) and free drainage (bottom layer's factor).  Freeze/thaw off passes no
factor and runs the original path.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land
from legoesm.land.config import MultiLayerLandConfig, inactive_land_param_names
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    soil_ice_log_impedance,
    step_multilayer_land,
)
from legoesm.land.richards import RichardsConfig, solve_richards
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig,
    interblock_K,
    psi_from_theta,
)
from legoesm.land.soil_thermal import liquid_water_content

jax.config.update("jax_enable_x64", True)

_DT = 1800.0
_LN10 = math.log(10.0)


def _column(hyd, theta_frac=0.8, ncol=1):
    grid = make_soil_grid()
    nl = grid.n_layers
    theta = jnp.full((ncol, nl), theta_frac * float(hyd.theta_sat))
    return grid, psi_from_theta(theta, hyd), theta, nl


def _run(hyd, rcfg, log_imp, n_steps=48, flux=0.0, theta_frac=0.8, sink=None):
    """n_steps Richards steps; returns final state and summed fluxes [m]."""
    grid, psi, theta, nl = _column(hyd, theta_frac)
    sink = jnp.zeros((1, nl)) if sink is None else sink
    flux_top = jnp.full(1, flux)
    sw = jnp.zeros(1)
    w0 = float(jnp.sum(theta * grid.dz))
    drain = runoff = 0.0
    for _ in range(n_steps):
        out = solve_richards(psi, theta, grid, hyd, rcfg, flux_top, sink, _DT,
                             surface_water=sw, log_impedance=log_imp)
        psi, theta, sw = out.psi_new, out.theta_new, out.surface_water
        drain += float(out.runoff_subsurface[0]) / constants.rho_water * _DT
        runoff += float(out.runoff_surface[0]) / constants.rho_water * _DT
    resid = (float(jnp.sum(theta * grid.dz)) - w0 + float(sw[0]) + drain + runoff
             - n_steps * _DT * (flux - float(jnp.sum(sink * grid.dz))))
    return dict(theta=theta, drain=drain, runoff=runoff, pond=float(sw[0]), resid=resid)


def _log_imp(nl, icefrac, e=6.0):
    f = jnp.broadcast_to(jnp.asarray(icefrac, dtype=jnp.float64), (1, nl))
    return -e * _LN10 * f


# ── interface form ───────────────────────────────────────────────────────────
def test_log_form_matches_clm_multiplier_unfloored():
    rng = np.random.default_rng(0)
    Ka = jnp.asarray(10.0 ** rng.uniform(-12, -4, (5, 9)))
    Kb = jnp.asarray(10.0 ** rng.uniform(-12, -4, (5, 9)))
    fa = jnp.asarray(rng.uniform(0, 1, (5, 9)))
    fb = jnp.asarray(rng.uniform(0, 1, (5, 9)))
    e = 6.0
    got = interblock_K(Ka, Kb, -e * _LN10 * fa, -e * _LN10 * fb)
    clm = jnp.sqrt(Ka * Kb) * 10.0 ** (-e * 0.5 * (fa + fb))
    np.testing.assert_allclose(got, clm, rtol=1e-12)
    # zero factors reproduce the plain geometric mean
    z = jnp.zeros_like(Ka)
    np.testing.assert_allclose(interblock_K(Ka, Kb, z, z), interblock_K(Ka, Kb), rtol=1e-12)


def test_log_form_floored_pair_returns_the_floor():
    K = jnp.full(3, 1e-30, dtype=jnp.float32)
    z = jnp.zeros(3, dtype=jnp.float32)
    got = np.asarray(interblock_K(K, K, z, z))
    np.testing.assert_allclose(got, 1e-20, rtol=1e-5, atol=0)
    with pytest.raises(AssertionError):   # an underflowed zero would not pass
        np.testing.assert_allclose(np.zeros(3), 1e-20, rtol=1e-5, atol=0)
    with pytest.raises(ValueError, match="both"):
        interblock_K(K, K, z, None)


# ── Richards: identity, impedance, closure ───────────────────────────────────
def test_exponent_zero_equals_no_factor():
    hyd = SoilHydraulicsConfig()
    rcfg = RichardsConfig()
    a = _run(hyd, rcfg, None, n_steps=4, flux=2e-6)
    nl = a["theta"].shape[1]
    b = _run(hyd, rcfg, _log_imp(nl, 0.7, e=0.0), n_steps=4, flux=2e-6)
    np.testing.assert_allclose(b["theta"], a["theta"], rtol=1e-12)
    assert b["drain"] == pytest.approx(a["drain"], rel=1e-12)


@pytest.mark.parametrize("curve", ["van_genuchten", "clapp_hornberger", "pdi"])
def test_frozen_column_barely_drains(curve):
    hyd = SoilHydraulicsConfig(retention_curve=curve)
    rcfg = RichardsConfig()
    nl = make_soil_grid().n_layers
    thawed = _run(hyd, rcfg, None)
    frozen = _run(hyd, rcfg, _log_imp(nl, 0.8))
    assert thawed["drain"] > 1e-4
    assert frozen["drain"] < 1e-3 * thawed["drain"]
    assert abs(frozen["resid"]) < 1e-9


def test_frozen_top_layer_turns_rain_into_runoff():
    hyd = SoilHydraulicsConfig(K_sat=2e-5)
    rcfg = RichardsConfig()
    nl = make_soil_grid().n_layers
    f = jnp.zeros((1, nl)).at[:, 0].set(1.0)
    thawed = _run(hyd, rcfg, None, n_steps=6, flux=1e-5, theta_frac=0.5)
    frozen = _run(hyd, rcfg, -6.0 * _LN10 * f, n_steps=6, flux=1e-5, theta_frac=0.5)
    assert frozen["runoff"] + frozen["pond"] > thawed["runoff"] + thawed["pond"] + 1e-3
    # the surface conductance itself is impeded: in one step the frozen top layer
    # takes in almost nothing of the 18 mm offered (thawed, it would fill)
    grid, _, th0, _ = _column(hyd, 0.5)
    one = _run(hyd, rcfg, -6.0 * _LN10 * f, n_steps=1, flux=1e-5, theta_frac=0.5)
    assert float((one["theta"] - th0)[0, 0] * grid.dz[0]) < 1e-4


def test_frozen_middle_layer_blocks_the_interfaces():
    """Only one mid-column layer frozen: the interfaces around it throttle the
    drainage out of the wet layers above (bottom layer itself thawed)."""
    hyd = SoilHydraulicsConfig()
    rcfg = RichardsConfig()
    nl = make_soil_grid().n_layers
    f = jnp.zeros((1, nl)).at[:, 3].set(1.0)
    thawed = _run(hyd, rcfg, None, n_steps=96)
    frozen = _run(hyd, rcfg, -6.0 * _LN10 * f, n_steps=96)
    above = slice(0, 3)
    assert float(jnp.sum(frozen["theta"][:, above])) > float(jnp.sum(thawed["theta"][:, above])) + 1e-3
    assert abs(frozen["resid"]) < 1e-9


@pytest.mark.parametrize("case", [
    dict(bc="free_drainage", fc=0.0, K=3e-6, flux=5e-6, sink=True),
    dict(bc="zero_flux", fc=0.0, K=3e-6, flux=5e-6, sink=False),
    dict(bc="free_drainage", fc=0.5, K=3e-6, flux=0.0, sink=True),
    dict(bc="free_drainage", fc=0.0, K=1e-3, flux=5e-4, sink=False),  # CFL caps bind
])
def test_water_closes_with_impedance(case):
    """soil + pond + runoff + drainage balances input - root sink; the reported
    bottom drainage is the debited one (else the residual shows it)."""
    hyd = SoilHydraulicsConfig(K_sat=case["K"])
    rcfg = RichardsConfig(bottom_bc=case["bc"], fc_drain_saturation=case["fc"])
    grid = make_soil_grid()
    nl = grid.n_layers
    sink = (jnp.zeros((1, nl)).at[:, 1:4].set(1e-8) if case["sink"] else None)
    icefrac = jnp.linspace(0.9, 0.1, nl)[None, :]
    out = _run(hyd, rcfg, -6.0 * _LN10 * icefrac, n_steps=12, flux=case["flux"],
               theta_frac=0.6, sink=sink)
    assert abs(out["resid"]) < 1e-9, out["resid"]


# ── land step wiring ─────────────────────────────────────────────────────────
def _forcing(T_air=270.0, precip=0.0):
    o = jnp.ones(1)
    p_s = 1.0e5 * o
    return AtmToSurface(
        sw_down=0.0 * o, lw_down=250.0 * o, precip_total=precip * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=0.002 * o,
        u_lowest=4.0 * o, v_lowest=0.0 * o, p_lowest=0.99 * p_s, p_surface=p_s,
        rho_lowest=p_s / (constants.R_d * T_air), cos_zenith=0.0 * o,
        co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)


def _cfg(freeze, e=6.0):
    base = MultiLayerLandConfig()
    return base._replace(
        thermal=base.thermal._replace(enable_freeze_thaw=freeze),
        richards=base.richards._replace(ice_impedance_exponent=e))


def _spy_step(monkeypatch, cfg, T_init, theta_init=0.35):
    """One jitted land step; returns (start state, the Richards call's
    flux_top, sink and log_impedance) as traced out of the step."""
    calls = []
    real = multilayer_land.solve_richards

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(multilayer_land, "solve_richards", spy)
    st = init_multilayer_land_state(1, cfg, T_init=T_init, theta_init=theta_init)

    @jax.jit
    def run(st):
        step_multilayer_land(st, _forcing(), cfg, 1.0, _DT, lat=jnp.full(1, 1.2))
        args, kw = calls[-1]
        return args[5], args[6], kw["log_impedance"]

    return st, run(st)


def test_freeze_off_passes_no_factor(monkeypatch):
    _, (_, _, li) = _spy_step(monkeypatch, _cfg(False), constants.T_freeze - 5.0)
    assert li is None


def test_frozen_step_passes_clm_factor(monkeypatch):
    cfg = _cfg(True)
    st, (_, _, li) = _spy_step(monkeypatch, cfg, constants.T_freeze - 5.0)
    liq, _ = liquid_water_content(st.T_soil, st.theta_soil, cfg.thermal)
    ice = st.theta_soil - liq
    want = -6.0 * _LN10 * jnp.clip(
        ice * constants.rho_water / constants.rho_ice / cfg.hydraulics.theta_sat, 0.0, 1.0)
    np.testing.assert_allclose(li, want, rtol=1e-12)
    assert float(jnp.min(li)) < -5.0 * _LN10  # deeply frozen


def test_thawed_column_is_unimpeded(monkeypatch):
    _, (_, _, li) = _spy_step(monkeypatch, _cfg(True), constants.T_freeze + 8.0)
    assert float(jnp.min(li)) > -1e-5  # factor > 0.99998 (sigmoid tail)


def test_roots_and_evaporation_unchanged_by_impedance(monkeypatch):
    """D6, land step: the soil water sink and top flux reaching Richards do not
    depend on the impedance."""
    T = constants.T_freeze - 3.0
    _, a0 = _spy_step(monkeypatch, _cfg(True, e=0.0), T)
    _, a6 = _spy_step(monkeypatch, _cfg(True, e=6.0), T)
    for i in (0, 1):  # flux_top, sink
        assert jnp.array_equal(a0[i], a6[i])


def test_evaporation_and_root_uptake_delivered_from_frozen_soil():
    """D6, Richards: with no pond, bare-soil evaporation and root uptake are
    withdrawn in full from a fully impeded column (the surface cell's supply cap
    carries evaporation, not the impeded conductance)."""
    hyd = SoilHydraulicsConfig()
    grid = make_soil_grid()
    nl = grid.n_layers
    sink = jnp.zeros((1, nl)).at[:, 1:4].set(2e-8)
    evap = -3e-8                                   # m/s, upward
    rcfg = RichardsConfig(bottom_bc="zero_flux")
    want = _DT * (evap - float(jnp.sum(sink * grid.dz)))
    for li in (None, _log_imp(nl, 1.0)):
        out = _run(hyd, rcfg, li, n_steps=1, flux=evap, theta_frac=0.6, sink=sink)
        _, _, th0, _ = _column(hyd, 0.6)
        dW = float(jnp.sum((out["theta"] - th0) * grid.dz))
        assert out["pond"] == 0.0 and out["runoff"] == 0.0
        assert dW == pytest.approx(want, rel=1e-6)


def test_zero_porosity_cell_gets_no_ice():
    cfg = _cfg(True)
    cfg = cfg._replace(hydraulics=cfg.hydraulics._replace(
        theta_sat=jnp.array([[0.0], [0.45]])))
    T = jnp.full((2, 3), constants.T_freeze - 5.0)
    th = jnp.full((2, 3), 0.2)
    li = soil_ice_log_impedance(T, th, cfg)
    assert bool(jnp.all(jnp.isfinite(li)))
    assert float(jnp.max(jnp.abs(li[0]))) == 0.0


# ── gradients ────────────────────────────────────────────────────────────────
def _drain_of(T0, e, k_sat):
    cfg = _cfg(True, e=e)
    cfg = cfg._replace(hydraulics=cfg.hydraulics._replace(K_sat=k_sat))
    grid = make_soil_grid()
    nl = grid.n_layers
    theta = jnp.full((1, nl), 0.8 * float(cfg.hydraulics.theta_sat))
    T = jnp.full((1, nl), T0)
    psi = psi_from_theta(theta, cfg.hydraulics)
    out = solve_richards(psi, theta, grid, cfg.hydraulics, cfg.richards,
                         jnp.zeros(1), jnp.zeros((1, nl)), _DT,
                         surface_water=jnp.zeros(1),
                         log_impedance=soil_ice_log_impedance(T, theta, cfg))
    return out.runoff_subsurface[0] + jnp.sum(out.theta_new * grid.dz)


def test_gradient_partially_frozen_matches_fd():
    """Partially frozen (mid-curtain), unclamped: gradient w.r.t. temperature,
    exponent and K_sat is nonzero and matches central differences."""
    x0 = (constants.T_freeze - 0.3, 6.0, 3e-6)
    g = jax.grad(lambda *x: _drain_of(*x) * 1e3, argnums=(0, 1, 2))(*x0)
    for i, (gi, h) in enumerate(zip(g, (1e-4, 1e-4, 1e-9))):
        xp = list(x0); xm = list(x0)
        xp[i] += h; xm[i] -= h
        fd = (_drain_of(*xp) - _drain_of(*xm)) * 1e3 / (2 * h)
        assert abs(float(gi)) > 0.0, i
        assert float(gi) == pytest.approx(float(fd), rel=1e-4), i


def test_float32_frozen_dry_column_under_jit_scan():
    hyd = SoilHydraulicsConfig()
    g64 = make_soil_grid()
    grid = g64._replace(**{k: jnp.asarray(getattr(g64, k), jnp.float32)
                           for k in ("dz", "dz_interface", "z_node", "z_interface")})
    nl = grid.n_layers
    cfg = _cfg(True)
    th0 = jnp.full((2, nl), 0.05, dtype=jnp.float32)  # dry
    th0 = th0.at[1].set(float(hyd.theta_sat))          # saturated, fully frozen
    T = jnp.full((2, nl), constants.T_freeze - 20.0, dtype=jnp.float32)

    def roll(T, e):
        c = cfg._replace(richards=cfg.richards._replace(ice_impedance_exponent=e))
        li = soil_ice_log_impedance(T, th0, c).astype(jnp.float32)

        def body(carry, _):
            psi, th = carry
            o = solve_richards(psi, th, grid, hyd, c.richards,
                               jnp.zeros(2, jnp.float32), jnp.zeros((2, nl), jnp.float32),
                               _DT, log_impedance=li)
            return (o.psi_new.astype(jnp.float32), o.theta_new.astype(jnp.float32)), None

        psi0 = psi_from_theta(th0, hyd).astype(jnp.float32)
        (psi, th), _ = jax.lax.scan(body, (psi0, th0), None, length=8)
        return jnp.sum(th)

    # the solve itself runs in float32 (no promotion from the grid or factor)
    o = solve_richards(psi_from_theta(th0, hyd).astype(jnp.float32), th0, grid, hyd,
                       cfg.richards, jnp.zeros(2, jnp.float32),
                       jnp.zeros((2, nl), jnp.float32), _DT,
                       log_impedance=soil_ice_log_impedance(T, th0, cfg).astype(jnp.float32))
    assert o.theta_new.dtype == jnp.float32 and o.runoff_subsurface.dtype == jnp.float32

    val = jax.jit(roll)(T, jnp.float32(6.0))
    gT, ge = jax.jit(jax.grad(roll, argnums=(0, 1)))(T, jnp.float32(6.0))
    assert bool(jnp.isfinite(val))
    assert bool(jnp.all(jnp.isfinite(gT))) and bool(jnp.isfinite(ge))


def test_exponent_carries_gradient_through_a_land_step():
    """No inert parameter: with freeze/thaw on the exponent moves the land state."""
    def loss(e):
        cfg = _cfg(True, e=e)
        st = init_multilayer_land_state(1, cfg, T_init=constants.T_freeze - 0.3,
                                        theta_init=0.40)
        new, _, _ = step_multilayer_land(st, _forcing(T_air=272.0), cfg, 1.0, _DT,
                                         lat=jnp.full(1, 1.2))
        return jnp.sum(new.theta_soil[:, -3:])
    g = jax.jit(jax.grad(loss))(6.0)
    assert bool(jnp.isfinite(g)) and abs(float(g)) > 0.0


def test_inactive_names_track_freeze_thaw():
    assert inactive_land_param_names(_cfg(False)) == {"land.richards.ice_impedance_exponent"}
    assert inactive_land_param_names(_cfg(True)) == frozenset()
