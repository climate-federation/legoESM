"""Air-entry van Genuchten curve and the dry-layer theta-step switch of the Richards solve.

Columns recorded from a production AMIP day (res6, 16 ranks, dt 1800 s) on which the
former solver never converged: a dry top layer receiving a 0.05 mm/h input (the step
damping crawled, 3550 such columns that day) and a ponded saturated column (the layers
flipped across psi = 0 every iteration, 669 such columns).  Run under JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.pedotransfer import soil_hydraulics_config_from_texture
from legoesm.land.richards import _CFL_SAFETY, psi_dry_floor, solve_richards
from legoesm.land.soil_grid import make_soil_grid_custom
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig, hydraulic_conductivity, interblock_K, moisture_capacity,
    psi_from_theta, theta_from_psi)

jax.config.update("jax_enable_x64", True)

# Expected on curves the dry-layer switch does not apply to (thresholds at their
# nonzero library defaults): harmless here, silenced explicitly.
_SWITCH_INACTIVE_OK = ("ignore:solve_richards. se_switch / c_switch_per_m are set but the "
                       "dry-layer switch is inactive:UserWarning")

_DZ = (0.002932551319648094, 0.005865102639296188, 0.011730205278592375,
       0.02346041055718475, 0.0469208211143695, 0.093841642228739,
       0.187683284457478, 0.375366568914956, 0.750733137829912,
       1.501466275659824)
_DT = 1800.0

# Dry top layer (Se ~ 0.02) over moister soil, 0.048 mm/h input, no pond.
_DRY_HYDRO = SoilHydraulicsConfig(theta_r=0.095, theta_sat=0.41, alpha_vg=1.9,
                                  n_vg=1.31, K_sat=7.22e-07)
_DRY = dict(
    psi=(-130118.36662129851, -9.161072304315486, -7.209692950559303,
         -6.454787553673726, -5.845751838352613, -4.630129141632636,
         -2.799781004203452, -2.389053537976291, -3.3870262083410907,
         -4.4681073765959125),
    theta=(0.10170582108041551, 0.2242071828600309, 0.23388733207953258,
           0.23856590648019396, 0.24287020544960777, 0.2534074223543996,
           0.27797046481490306, 0.2861508872260405, 0.2684027783384482,
           0.25506729496245195),
    sink=(0.0,) * 10, flux_top=1.33462175721415e-08, pond=0.0, log_imp=(0.0,) * 10)

# Ponded, partly frozen column, wet top over a drier subsoil (loam, the
# SoilHydraulicsConfig defaults).  Converges with the air-entry curve, not with the unmodified one.
_WET_HYDRO = SoilHydraulicsConfig()
_WET = dict(
    psi=(-0.0002279387582892228, -0.0067143662197155325, -1.164777926967595,
         -1.737768226677235, -1.7052006805757465, -1.6454146329339383,
         -1.5602040567783657, -1.4574971686304572, -1.3501918389205319,
         -1.1939986187287832),
    theta=(0.4299980607677621, 0.4296209643864074, 0.23008573818701544,
           0.20157725674988347, 0.20282082407248422, 0.20519524650737966,
           0.20880194475483338, 0.21354108506278252, 0.2190237287760218,
           0.22818765445360728),
    sink=(1.1990123592254016e-11, 5.702755198134389e-12, 6.310198097809234e-13,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    flux_top=3.601757613601539e-08, pond=0.0005473411097446029,
    # frozen-soil conductivity impedance the run applied (ln of the K multiplier)
    log_imp=(-0.28492946637123095, -2.1790697612313594, -4.3604673241217995,
             -6.272378617489319, -6.7316007283841754, -6.8299045843035495,
             -6.950039788678326, -7.1077836206209675, -7.290275178107841,
             -7.595299388387436))


def _grid(dtype):
    g = make_soil_grid_custom(_DZ)
    return g._replace(**{k: jnp.asarray(getattr(g, k), dtype=dtype)
                         for k in ("dz", "z_node", "z_interface", "dz_interface")})


def _solve(col, hydro, psi=None, dtype=jnp.float64, flux_top=None):
    f = lambda x: jnp.asarray([x], dtype=dtype)
    return solve_richards(
        f(col["psi"] if psi is None else psi), f(col["theta"]),
        _grid(dtype), hydro, MultiLayerLandConfig().richards,
        f(col["flux_top"]) if flux_top is None else flux_top, f(col["sink"]), _DT,
        surface_water=f(col["pond"]), log_impedance=f(col["log_imp"]))


def _assert_fixed_point(col, hydro):
    """Converged, and a fixed point of the step equations: restarted from its own
    answer (same time-level-n water), the solve accepts it on the first iteration
    with no change, so every layer's water balance holds before any repair."""
    out = _solve(col, hydro)
    assert bool(out.converged[0])
    again = _solve(col, hydro, psi=np.asarray(out.psi_new[0]))
    assert bool(again.converged[0]) and float(again.n_iter[0]) == 1.0
    assert float(jnp.max(jnp.abs(again.theta_new - out.theta_new))) < 1e-6
    assert abs(float(out.water_created[0])) < 1e-12


def test_dry_top_recorded_column_converges_to_a_fixed_point():
    _assert_fixed_point(_DRY, _DRY_HYDRO)


def test_ponded_recorded_column_converges():
    # The recorded heads came from the unmodified curve; a run on the air-entry
    # curve carries heads consistent with its own water content, so seed from it.
    # The solver API seeds the surface cell at the old pond, so the restart check
    # above does not apply to a column whose pond changes.
    seed = np.asarray(psi_from_theta(jnp.asarray(_WET["theta"]), _WET_HYDRO))
    out = _solve(_WET, _WET_HYDRO, psi=seed)
    assert bool(out.converged[0])
    assert abs(float(out.water_created[0])) < 1e-12


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradient_finite_through_switched_dry_layers(dtype):
    def loss(ft):
        return jnp.sum(_solve(_DRY, _DRY_HYDRO, dtype=dtype, flux_top=ft).theta_new)
    ft = jnp.asarray([_DRY["flux_top"]], dtype=dtype)
    assert _solve(_DRY, _DRY_HYDRO, dtype=dtype, flux_top=ft).theta_new.dtype == dtype
    g = jax.grad(loss)(ft)
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize("curve,h_e,psi0", [
    ("van_genuchten", -0.02, -1000.0), ("van_genuchten", -0.02, -0.03),
    ("van_genuchten", 0.0, -1e-12), ("pdi", -0.02, -1000.0), ("lu", -0.02, -1000.0),
    ("brooks_corey", -0.02, -1000.0), ("clapp_hornberger", -0.02, -1000.0)])
@pytest.mark.filterwarnings(_SWITCH_INACTIVE_OK)
def test_column_at_rest_stays_at_rest(curve, h_e, psi0):
    """Hydrostatic column (head rises with depth by the depth), zero flux at both
    ends: nothing may move.  (The theta-step switch with an inexact inverse moved
    such a column: PDI / Lu -1000 m -> -3.9e6 m; the unmodified van Genuchten near
    saturation -1e-12 m -> -7.6e-5 m.)"""
    h = SoilHydraulicsConfig(retention_curve=curve, vg_air_entry_m=h_e)
    grid = make_soil_grid_custom((0.05,) * 6)
    psi = psi0 + jnp.asarray(grid.z_node)[None, :] - float(grid.z_node[0])
    out = solve_richards(psi, theta_from_psi(psi, h), grid, h,
                         # no field-capacity drainage limiter: it scales gravity only,
                         # so a hydrostatic profile is not at rest under it
                         MultiLayerLandConfig().richards._replace(
                             bottom_bc="zero_flux", fc_drain_saturation=0.0),
                         jnp.zeros(1), jnp.zeros((1, 6)), _DT)
    assert bool(out.converged[0])
    assert float(jnp.max(jnp.abs(out.psi_new - psi))) < 1e-9


@pytest.mark.parametrize("h_e", [0.01, float("nan"), float("-inf")])
def test_positive_or_nonfinite_air_entry_head_is_rejected(h_e):
    h = SoilHydraulicsConfig(vg_air_entry_m=h_e)
    with pytest.raises(ValueError, match="vg_air_entry_m"):
        theta_from_psi(jnp.asarray(-1.0), h)
    with pytest.raises(ValueError, match="vg_air_entry_m"):
        psi_from_theta(jnp.asarray(0.3), h)
    with pytest.raises(ValueError, match="vg_air_entry_m"):
        hydraulic_conductivity(jnp.asarray(-1.0), jnp.asarray(0.3), h)
    with pytest.raises(ValueError, match="vg_air_entry_m"):
        solve_richards(jnp.full((1, 6), -1.0), jnp.full((1, 6), 0.3), make_soil_grid_custom((0.05,) * 6),
                       h, MultiLayerLandConfig().richards, jnp.zeros(1), jnp.zeros((1, 6)), _DT)


# --- curve -----------------------------------------------------------------------
_H = _WET_HYDRO
_HE = _H.vg_air_entry_m


def test_air_entry_curve_is_on_by_default():
    assert _HE == -0.02


def test_capacity_is_the_derivative_of_water_content():
    psi = jnp.concatenate([-jnp.logspace(-4, 4, 200), jnp.linspace(-0.05, 0.05, 101)])
    grad = jax.vmap(jax.grad(lambda p: theta_from_psi(p, _H)))(psi)
    C = moisture_capacity(psi, None, _H)
    off_kink = jnp.abs(psi - _HE) > 1e-9
    assert float(jnp.max(jnp.where(off_kink, jnp.abs(C - grad) / jnp.maximum(grad, 1e-12), 0.0))) < 1e-9


def test_inverse_round_trips_including_the_elastic_branch():
    psi = jnp.concatenate([-jnp.logspace(-1.5, 4, 200), jnp.linspace(_HE, 1.0, 50)])
    back = psi_from_theta(theta_from_psi(psi, _H), _H)
    assert float(jnp.max(jnp.abs(back - psi) / jnp.maximum(jnp.abs(psi), 1.0))) < 1e-8
    # and water content survives the round trip right up to saturation
    theta = _H.theta_sat - jnp.logspace(-12, -3, 40) * (_H.theta_sat - _H.theta_r)
    again = theta_from_psi(psi_from_theta(theta, _H), _H)
    assert float(jnp.max(jnp.abs(again - theta))) < 1e-12


def test_saturated_and_continuous_at_the_air_entry_head():
    below, at = jnp.asarray(_HE * (1 + 1e-12)), jnp.asarray(_HE)
    assert float(theta_from_psi(below, _H)) == pytest.approx(_H.theta_sat, abs=1e-12)
    assert float(theta_from_psi(at, _H)) == _H.theta_sat
    assert float(hydraulic_conductivity(below, None, _H)) == pytest.approx(_H.K_sat, rel=1e-9)
    assert float(hydraulic_conductivity(at, None, _H)) == pytest.approx(_H.K_sat, rel=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("h_e", [-0.02, -1e-6])
def test_conductivity_gradients_finite_at_saturation(dtype, h_e):
    """A tiny air-entry head rounds Sc to 1 in float32, which put the Mualem pow
    at a zero base (NaN gradient in alpha_vg)."""
    for psi in (h_e * 1.5, h_e, 0.0, 0.1, -10.0):
        def K(a, n):
            h = SoilHydraulicsConfig(alpha_vg=a, n_vg=n, vg_air_entry_m=h_e)
            return hydraulic_conductivity(jnp.asarray(psi, dtype), None, h)
        g = jax.grad(K, argnums=(0, 1))(jnp.asarray(3.6, dtype), jnp.asarray(1.56, dtype))
        assert all(bool(jnp.isfinite(x)) for x in g), (psi, g)
    # and the inverse, across saturation into the elastic branch
    for theta in (0.2, 0.4299, 0.43, 0.44):
        def P(th, a):
            return psi_from_theta(th, SoilHydraulicsConfig(alpha_vg=a, vg_air_entry_m=h_e))
        g = jax.grad(P, argnums=(0, 1))(jnp.asarray(theta, dtype), jnp.asarray(3.6, dtype))
        assert all(bool(jnp.isfinite(x)) for x in g), (theta, g)


@pytest.mark.parametrize("curve", ["clapp_hornberger", "brooks_corey", "pdi", "lu"])
def test_other_curves_ignore_the_air_entry_head(curve):
    psi = jnp.concatenate([-jnp.logspace(-3, 4, 50), jnp.linspace(0.0, 0.5, 5)])
    a = SoilHydraulicsConfig(retention_curve=curve)
    b = a._replace(vg_air_entry_m=0.0)
    th_a, th_b = theta_from_psi(psi, a), theta_from_psi(psi, b)
    assert np.array_equal(np.asarray(th_a), np.asarray(th_b))
    assert np.array_equal(np.asarray(psi_from_theta(th_a, a)), np.asarray(psi_from_theta(th_b, b)))
    assert np.array_equal(np.asarray(hydraulic_conductivity(psi, th_a, a)),
                          np.asarray(hydraulic_conductivity(psi, th_b, b)))
    assert np.array_equal(np.asarray(moisture_capacity(psi, th_a, a)),
                          np.asarray(moisture_capacity(psi, th_b, b)))


def test_zero_air_entry_head_is_the_unmodified_curve():
    from legoesm.land.soil_hydraulics import van_genuchten_theta, van_genuchten_K
    psi = -jnp.logspace(-3, 4, 50)
    h0 = _H._replace(vg_air_entry_m=0.0)
    assert np.array_equal(np.asarray(theta_from_psi(psi, h0)), np.asarray(van_genuchten_theta(psi, _H, 0.0)))
    assert np.array_equal(np.asarray(hydraulic_conductivity(psi, None, h0)),
                          np.asarray(van_genuchten_K(psi, _H, 0.0)))


def test_modified_curve_matches_the_published_closed_forms():
    """Vogel, van Genuchten & Cislerova (2001) / Ippisch, Vogel & Bastian (2006,
    eqs 9-11), transcribed independently here with the math module:
      Sc = (1 + (a|h_e|)^n)^-m,  Se = (1 + (a|h|)^n)^-m / Sc  (h < h_e; 1 above),
      theta = theta_r + (theta_s - theta_r) Se,
      K = K_s Se^0.5 [(1 - (1 - (Se Sc)^(1/m))^m) / (1 - (1 - Sc^(1/m))^m)]^2,
      C = (theta_s - theta_r) a n m (a|h|)^(n-1) (1 + (a|h|)^n)^(-m-1) / Sc."""
    import math
    for hy in (_H, _DRY_HYDRO, _H._replace(vg_air_entry_m=-0.5, n_vg=2.7, alpha_vg=0.8)):
        a, n, he = hy.alpha_vg, hy.n_vg, hy.vg_air_entry_m
        m = 1.0 - 1.0 / n
        Sc = (1.0 + (a * abs(he)) ** n) ** -m
        for h in (he * 1.0001, -0.1, -1.0, -3.3, -10.0, -150.0, -1.0e4):
            if h >= he:
                continue
            u = (1.0 + (a * abs(h)) ** n) ** -m
            Se = u / Sc
            th = hy.theta_r + (hy.theta_sat - hy.theta_r) * Se
            K = hy.K_sat * math.sqrt(Se) * ((1.0 - (1.0 - (Se * Sc) ** (1.0 / m)) ** m)
                                            / (1.0 - (1.0 - Sc ** (1.0 / m)) ** m)) ** 2
            C = ((hy.theta_sat - hy.theta_r) * a * n * m * (a * abs(h)) ** (n - 1.0)
                 * (1.0 + (a * abs(h)) ** n) ** (-m - 1.0) / Sc)
            hj = jnp.asarray(h)
            assert float(theta_from_psi(hj, hy)) == pytest.approx(th, rel=1e-12), h
            assert float(hydraulic_conductivity(hj, None, hy)) == pytest.approx(K, rel=1e-10), h
            assert float(moisture_capacity(hj, None, hy)) == pytest.approx(C, rel=1e-10), h
            if Se > 1e-6:      # the inverse's documented Se clip
                assert float(psi_from_theta(jnp.asarray(th), hy)) == pytest.approx(h, rel=1e-9), h


def _layer_residual(out, col, hydro, rc, dt, grid):
    """Per-layer water balance [m] of the mixed-form step at the RETURNED state
    (after the solver's budget repair, which the output does not expose
    separately), transcribed from the discretisation (Celia et al. 1990):
    storage change + sink*dt - (flux in - flux out)*dt, layer by layer, plus the
    surface cell.  Uniform conductivity in depth (k_sat_decay_m = 0)."""
    f = lambda x: jnp.asarray([x])
    psi = jnp.maximum(out.psi_new, psi_dry_floor(hydro))
    limp, dz, dz_if = f(col["log_imp"]), grid.dz, grid.dz_interface
    th = theta_from_psi(psi, hydro)
    K = hydraulic_conductivity(psi, th, hydro)
    Kh = jnp.minimum(interblock_K(K[:, :-1], K[:, 1:], limp[:, :-1], limp[:, 1:]),
                     _CFL_SAFETY * jnp.minimum(dz[:-1], dz[1:]) / dt)
    fd = jnp.ones_like(th)                       # field-capacity gravity limiter
    if rc.fc_drain_saturation > 0.0:
        se = jnp.clip((th - hydro.theta_r) / (hydro.theta_sat - hydro.theta_r), 0.0, 1.0)
        fd = jnp.clip((se - rc.fc_drain_saturation) / (1.0 - rc.fc_drain_saturation), 0.0, 1.0)
    F = Kh / dz_if * (psi[:, :-1] - psi[:, 1:]) + Kh * fd[:, :-1]   # downward face flux
    q01_cap = f(col["pond"]) / dt + f(col["flux_top"])
    Ks0 = jnp.minimum(jnp.broadcast_to(hydro.K_sat, th.shape)[:, 0] * jnp.exp(limp[:, 0]),
                      _CFL_SAFETY * dz[0] / dt)
    h_s = out.surface_water + out.runoff_surface / constants.rho_water * dt
    q01 = jnp.minimum(Ks0 * ((h_s - psi[:, 0]) / (0.5 * dz[0]) + 1.0), q01_cap)
    Kb = jnp.minimum(K[:, -1] * jnp.exp(limp[:, -1]), _CFL_SAFETY * dz[-1] / dt) * fd[:, -1]
    flux_in = jnp.concatenate([q01[:, None], F], axis=1)
    flux_out = jnp.concatenate([F, Kb[:, None]], axis=1)
    res = (th - f(col["theta"])) * dz + (f(col["sink"]) * dz - flux_in + flux_out) * dt
    surf = (h_s - f(col["pond"])) - (f(col["flux_top"]) - q01) * dt
    return jnp.concatenate([res, surf[:, None]], axis=1)


def test_converged_dry_column_returns_a_state_closing_every_layer():
    out = _solve(_DRY, _DRY_HYDRO)
    assert bool(out.converged[0])
    rc = MultiLayerLandConfig().richards
    res = _layer_residual(out, _DRY, _DRY_HYDRO, rc, _DT, make_soil_grid_custom(_DZ))
    # converged = every layer's theta change below theta_tol on the last iterate
    bound = rc.theta_tol * jnp.asarray(_DZ)
    assert bool(jnp.all(jnp.abs(res[0, :-1]) <= bound)), res
    assert abs(float(res[0, -1])) < 1e-12


# A Cosby (1984) loam through the committed pedotransfer (Clapp-Hornberger,
# theta_r = 0): the dry-layer switch stays off for this curve.
_CH = soil_hydraulics_config_from_texture(40.0, 20.0)


def test_switch_thresholds_set_on_an_inactive_curve_warn_and_illegal_ones_raise():
    grid = make_soil_grid_custom((0.05,) * 6)
    psi = jnp.full((1, 6), -1.0)
    def run(h, rc):
        th = theta_from_psi(psi, h)
        return solve_richards(psi, th, grid, h, rc, jnp.zeros(1), jnp.zeros((1, 6)), _DT)
    rc = MultiLayerLandConfig().richards
    for h in (SoilHydraulicsConfig(retention_curve="brooks_corey"), _CH,
              SoilHydraulicsConfig(vg_air_entry_m=0.0)):
        with pytest.warns(UserWarning, match="inactive"):
            run(h, rc)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        run(SoilHydraulicsConfig(retention_curve="brooks_corey"),
            rc._replace(se_switch=0.0, c_switch_per_m=0.0))
        run(SoilHydraulicsConfig(), rc)
    for bad in (dict(se_switch=-0.1), dict(se_switch=1.0), dict(se_switch=float("nan")),
                dict(c_switch_per_m=-1e-5), dict(c_switch_per_m=float("inf"))):
        with pytest.raises(ValueError, match=next(iter(bad))):
            run(SoilHydraulicsConfig(), rc._replace(**bad))


# --- Clapp-Hornberger saturation at psi_sat --------------------------------------
# The power law is flat at theta_sat from psi_sat up; the elastic branch and its
# capacity start there (formerly at 0, which left [psi_sat, 0) with flat theta but a
# positive capacity: ponded saturated columns stopped early and lost 1-10 mm/step).
@pytest.mark.parametrize("curve", ["clapp_hornberger", "campbell"])
def test_clapp_hornberger_is_continuous_and_elastic_from_psi_sat(curve):
    h = _CH._replace(retention_curve=curve)
    ps, ss = h.psi_sat, h.S_s * h.theta_sat
    assert float(theta_from_psi(jnp.asarray(ps), h)) == h.theta_sat
    assert float(theta_from_psi(jnp.asarray(ps * (1 + 1e-12)), h)) == pytest.approx(h.theta_sat, abs=1e-12)
    for d in (1e-3, 0.1, -ps, 1.0):
        assert float(theta_from_psi(jnp.asarray(ps + d), h)) == pytest.approx(h.theta_sat + ss * d, rel=1e-14)


@pytest.mark.parametrize("curve", ["clapp_hornberger", "campbell"])
def test_clapp_hornberger_capacity_is_the_derivative_on_both_sides_of_psi_sat(curve):
    h = _CH._replace(retention_curve=curve)
    psi = jnp.concatenate([-jnp.logspace(np.log10(-h.psi_sat) + 1e-6, 4, 200),
                           jnp.linspace(h.psi_sat * (1 - 1e-6), 1.0, 101)])
    grad = jax.vmap(jax.grad(lambda p: theta_from_psi(p, h)))(psi)
    C = moisture_capacity(psi, theta_from_psi(psi, h), h)
    assert float(jnp.max(jnp.abs(C - grad) / jnp.maximum(jnp.abs(grad), 1e-12))) < 1e-9


@pytest.mark.parametrize("curve", ["clapp_hornberger", "campbell"])
def test_clapp_hornberger_inverse_round_trips_including_the_elastic_branch(curve):
    h = _CH._replace(retention_curve=curve)
    floor = float(psi_dry_floor(h))
    psi = jnp.concatenate([-jnp.logspace(np.log10(-h.psi_sat) + 1e-9, np.log10(-floor), 300),
                           jnp.linspace(h.psi_sat, 2.0, 60)])
    back = psi_from_theta(theta_from_psi(psi, h), h)
    assert float(jnp.max(jnp.abs(back - psi) / jnp.maximum(jnp.abs(psi), 1.0))) < 1e-9
    theta = jnp.linspace(float(theta_from_psi(jnp.asarray(floor), h)), h.theta_sat * (1 + 1e-4), 300)
    assert float(jnp.max(jnp.abs(theta_from_psi(psi_from_theta(theta, h), h) - theta))) < 1e-12


@pytest.mark.filterwarnings(_SWITCH_INACTIVE_OK)
def test_saturated_clapp_hornberger_column_at_rest_stays_at_rest():
    rc = MultiLayerLandConfig().richards._replace(bottom_bc="zero_flux", fc_drain_saturation=0.0)
    grid = make_soil_grid_custom((0.05,) * 6)
    # head from inside [psi_sat, 0) at the top to positive below
    psi = 0.5 * _CH.psi_sat + jnp.asarray(grid.z_node)[None, :] - float(grid.z_node[0])
    out = solve_richards(psi, theta_from_psi(psi, _CH), grid, _CH, rc, jnp.zeros(1), jnp.zeros((1, 6)), _DT)
    assert bool(out.converged[0])
    assert float(jnp.max(jnp.abs(out.psi_new - psi))) < 1e-9
    assert abs(float(out.water_created[0])) < 1e-15


# Worst converged ponded column of the Cosby set (SYNTHETIC recorded-style: a recorded
# production column's saturation profile, sink and pond on the Cosby loam): fully
# saturated under a 17 mm pond.  With saturation at 0 instead of psi_sat it was
# accepted after one iteration and lost 10.3 mm.
_CH_PONDED = dict(
    psi=(_CH.psi_sat,) * 10, theta=(_CH.theta_sat,) * 10,
    sink=(3.522576108618152e-07, 1.7454831614930273e-07, 8.571487890630423e-08,
          4.1339699585326274e-08, 1.9231781346737493e-08, 8.324418412732043e-09,
          3.1192678840829778e-09, 8.759518105194935e-10, 1.3815458161779797e-10,
          6.873302253148755e-12),
    flux_top=-1.341545671866834e-08, pond=0.01713178520088233,
    log_imp=(-6.090200219258314e-13, -6.690642494396457e-13, -6.840753063180994e-13,
             -4.010096623244031e-13, -3.6455423847673004e-14, 0.0, 0.0, 0.0, 0.0, 0.0))


@pytest.mark.filterwarnings(_SWITCH_INACTIVE_OK)
def test_ponded_saturated_clapp_hornberger_column_closes_its_water_budget():
    out = _solve(_CH_PONDED, _CH)
    assert bool(out.converged[0])
    assert abs(float(out.water_created[0])) < 1e-12
    rc = MultiLayerLandConfig().richards
    res = _layer_residual(out, _CH_PONDED, _CH, rc, _DT, make_soil_grid_custom(_DZ))
    assert bool(jnp.all(jnp.abs(res[0, :-1]) <= rc.theta_tol * jnp.asarray(_DZ))), res


@pytest.mark.parametrize("name", ["van_genuchten_Se", "van_genuchten_theta", "van_genuchten_psi",
                                  "van_genuchten_K", "van_genuchten_C"])
def test_van_genuchten_leaves_require_the_saturation_head(name):
    import legoesm.land.soil_hydraulics as sh
    with pytest.raises(TypeError):
        getattr(sh, name)(jnp.asarray([-1.0]), _H)
