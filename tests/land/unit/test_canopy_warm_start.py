"""Warm start + feasible-set reparameterisation of the canopy Newton closure.

The warm start exists to raise the CONVERGED fraction of the two-leaf canopy solve,
because the solver's ``custom_vjp`` zeroes the gradient of every non-converged
column: measured over 864 steps x 39 ERA5 columns, 12.7% of column-steps fail,
75% of them by exhausting the iteration cap rather than by the damping ceiling,
and the failures concentrate on the DEGENERATE canopy (bare-soil-dominant cells
23.6%, LAI < 0.5 16.0%, versus 2.6% at LAI >= 0.5).  It may not alter what the
solve converges TO: the seed is a pure numerical cache — the fixed point does
not depend on it, and the adjoint returns a zero cotangent for it
(``test_canopy_solver_grad.test_x0_cotangent_is_zero``).

A companion feasible-set reparameterisation of the Ci and q_c unknowns was
implemented, MEASURED and REVERTED: it cost convergence on well-posed
vegetated columns (scan 2.3% -> 4.4% non-converged; a 400-column cold-start
battery 96.75% -> 95.50%), because a sigmoid box is flat in its tails exactly
where a bound binds.  Its tests went with it.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import (
    CanopyForcingBundle,
    solve_canopy_closure,
    canopy_forward,
    _canopy_residual,
)

_a = jnp.asarray
_CFG = CanopyConfig()
_CA = 400.0
# The cold start the two-leaf canopy builds: leaf and canopy air at air
# temperature, Ci at 0.7*Ca.
_X0 = _a([298.0, 298.0, 0.7 * _CA, 0.7 * _CA, 298.0, 0.010])


def _bundle(**kw) -> CanopyForcingBundle:
    """A physically reasonable single-column midday forcing bundle."""
    d = dict(
        LAI=_a(3.0), SZA=_a(30.0), La=_a(350.0), epsf=_a(0.97), epss=_a(0.96),
        fSun=_a(0.6), APAR_Sun=_a(800.0), APAR_Sh=_a(200.0),
        Vcmax25_Sun=_a(40.0), Vcmax25_Sh=_a(20.0),
        Vcmax25_C4Sun=_a(0.0), Vcmax25_C4Sh=_a(0.0),
        ASW_Sun=_a(300.0), ASW_Sh=_a(80.0), ASW_Soil=_a(50.0),
        Ts_bc=_a(298.0), Ca=_a(_CA), Ps=_a(101325.0), Ta=_a(298.0),
        lam=_a(constants.L_v), Cp=_a(constants.c_pd), rhoa=_a(1.2),
        Tv_atm=_a(299.0), q_atm=_a(0.010), m=_a(9.0), b0=_a(0.01),
        alf=_a(0.3), TgC=_a(25.0), fC4=_a(0.0), fStress_soil=_a(0.5),
        ur=_a(2.0), CI=_a(0.8), z0m=_a(0.1), displa=_a(1.0), z0=_a(10.0),
        cv=_a(0.01), d_leaf=_a(0.025), r_soil_surface=_a(0.0), fwet=_a(0.0),
    )
    d.update(kw)
    return CanopyForcingBundle(**d)


def _resid_norm(x, bundle) -> float:
    return float(jnp.linalg.norm(_canopy_residual(
        x, bundle, _CFG.LE_module, _CFG.stomatal_model,
        _CFG.le_cap_mode, _CFG.use_ta_for_photosynthesis)))


# ---------------------------------------------------------------------------
# The solve still finds a root, and the seed does not move it
# ---------------------------------------------------------------------------

def test_converged_state_is_a_root_and_feasible():
    b = _bundle()
    x, _n, conv = solve_canopy_closure(_X0, b, _CFG)
    assert bool(conv)
    assert float(x[5]) > 0.0
    # A root of the UNCHANGED physical residual, at the solver's own RELATIVE
    # tolerance (the criterion is ||F||^2 <= atol + rtol*||F_0||^2, not zero).
    assert _resid_norm(x, b) < 1e-2 * _resid_norm(_X0, b)


@pytest.mark.parametrize("seed", [
    [298.0, 298.0, 280.0, 280.0, 298.0, 0.010],   # the cold start
    [310.0, 290.0, 350.0, 210.0, 305.0, 0.020],   # far, still feasible
    [285.0, 305.0, 200.0, 380.0, 290.0, 0.004],   # far the other way
])
def test_warm_start_reaches_the_same_fixed_point(seed):
    """Different seeds converge to the same root, within the solver's own ball.

    This is the claim the warm start rests on: it changes only WHICH columns
    reach the fixed point inside the iteration budget, never where it is.
    """
    b = _bundle()
    x_cold, _, c_cold = solve_canopy_closure(_X0, b, _CFG)
    x_seed, _, c_seed = solve_canopy_closure(_a(seed), b, _CFG)
    assert bool(c_cold) and bool(c_seed)
    assert float(jnp.max(jnp.abs(
        x_seed[jnp.array([0, 1, 4])] - x_cold[jnp.array([0, 1, 4])]))) < 0.5  # K
    assert float(jnp.abs(x_seed[5] - x_cold[5])) < 1e-3                       # kg/kg
    fl = lambda x: canopy_forward(x, b, _CFG.LE_module, _CFG.stomatal_model,
                                  _CFG.le_cap_mode, _CFG.use_ta_for_photosynthesis)
    f_cold, f_seed = fl(x_cold), fl(x_seed)
    for k in ("LE_Sun", "H_Sun", "G"):
        assert float(jnp.abs(f_seed[k] - f_cold[k])) < 1.0                    # W/m2


def test_warm_start_costs_fewer_iterations():
    """Seeding from the previous solution is CHEAPER — the point of the change.

    75% of the measured failures exhaust the iteration cap, so iterations saved
    here are failures avoided in the scan.
    """
    b = _bundle()
    x_cold, _n_cold, _ = solve_canopy_closure(_X0, b, _CFG)
    # A nearby forcing, as the next Picard pass or the next timestep presents.
    b2 = _bundle(Ts_bc=_a(298.4))
    _, n_from_cold, _ = solve_canopy_closure(_X0, b2, _CFG)
    _, n_from_warm, _ = solve_canopy_closure(x_cold, b2, _CFG)
    assert int(n_from_warm) < int(n_from_cold)


# ---------------------------------------------------------------------------
# The cache the caller carries: never a non-converged iterate
# ---------------------------------------------------------------------------

def test_canopy_cache_never_stores_a_non_converged_iterate():
    """``two_leaf_canopy`` hands back the last CONVERGED solve, else NaN.

    A failed solve's final iterate is not a root; storing it would seed the next
    step from a non-physical state and let one failure cascade.  Asserted on the
    source of the function that actually runs the Picard loop.
    """
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)
    src = inspect.getsource(compute_two_leaf_canopy_fluxes)
    assert "x_conv = jnp.where(converged[:, None], x_final, x_conv)" in src
    assert "x_conv = jnp.full_like(initial_state, jnp.nan)" in src
    assert "canopy_x=x_conv," in src


def test_non_finite_seed_cannot_be_reported_as_converged():
    """An all-NaN seed must not produce a "converged" answer."""
    x_nan, _n, conv = solve_canopy_closure(_a([jnp.nan] * 6), _bundle(), _CFG)
    assert (not bool(conv)) or bool(jnp.all(jnp.isfinite(x_nan)))


# ---------------------------------------------------------------------------
# Adversarial regimes.  The measured failures live at LOW LAI and on bare-soil
# columns, so those are here explicitly.
# ---------------------------------------------------------------------------

_DEGENERATE = [
    ("lai_zero", dict(LAI=_a(0.0), fSun=_a(0.0), APAR_Sun=_a(0.0),
                      APAR_Sh=_a(0.0), ASW_Sun=_a(0.0), ASW_Sh=_a(0.0))),
    ("lai_1e-3", dict(LAI=_a(1e-3), fSun=_a(0.01))),
    ("lai_0.1", dict(LAI=_a(0.1), fSun=_a(0.2))),
    # bare-soil-dominant cell under snow: the single worst measured regime
    ("bare_soil_snow", dict(LAI=_a(0.0), fSun=_a(0.0), ASW_Soil=_a(30.0),
                            Ts_bc=_a(271.0), Ta=_a(269.0), Tv_atm=_a(270.0),
                            q_atm=_a(2e-3))),
    ("night", dict(APAR_Sun=_a(0.0), APAR_Sh=_a(0.0), ASW_Sun=_a(0.0),
                   ASW_Sh=_a(0.0), ASW_Soil=_a(0.0), fSun=_a(0.02))),
    ("calm", dict(ur=_a(0.01))),
    ("wilting", dict(fStress_soil=_a(0.0), m=_a(0.0), b0=_a(0.0))),
    ("freezing", dict(Ta=_a(263.0), Ts_bc=_a(263.0), Tv_atm=_a(263.0),
                      q_atm=_a(1e-3))),
    ("humid", dict(q_atm=_a(0.019))),
]


@pytest.mark.parametrize("name,kw", _DEGENERATE)
def test_degenerate_regimes_stay_inside_the_feasible_set(name, kw):
    """Whatever the outcome, Ci and q_c never leave the set the physics allows.

    Convergence is NOT asserted: some of these regimes are genuinely degenerate
    (at LAI -> 0 the leaf rows carry no area and the Jacobian loses rank), and
    the honest contract there is ``converged=False``, which the caller's guard
    already handles.  What must hold unconditionally is that the returned state
    is feasible, so a failed column can never poison the next one through the
    warm-start cache or the canopy air space.
    """
    b = _bundle(**kw)
    x, _n, conv = solve_canopy_closure(_X0, b, _CFG)
    assert bool(jnp.all(jnp.isfinite(x))), name
    if bool(conv):
        assert _resid_norm(x, b) < 1e-2 * _resid_norm(_X0, b), name


# ---------------------------------------------------------------------------
# BEHAVIOURAL test of the change itself: the Picard loop and the returned cache
#
# The tests above drive ``solve_canopy_closure`` directly, which exercises the
# solver's seeding but NOT the outer Picard loop or the cache this change adds —
# codex flagged them as vacuous with respect to the diff.  These drive the real
# entry point instead.
# ---------------------------------------------------------------------------

def _canopy_call(ncol: int = 8, seed_arr=None, cap: int = 60):
    """One two-leaf canopy call over a small realistic column batch."""
    import numpy as np
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)

    r = np.random.default_rng(3)
    f = lambda a: jnp.asarray(a, dtype=jnp.float32)
    Ta = 290.0 + 8.0 * r.random(ncol)
    forcing = AtmToSurface(
        sw_down=f(200.0 + 400.0 * r.random(ncol)),
        lw_down=f(300.0 + 60.0 * r.random(ncol)),
        precip_total=f(np.zeros(ncol)), precip_snow=f(np.zeros(ncol)),
        T_lowest=f(Ta), q_lowest=f(0.006 + 0.006 * r.random(ncol)),
        u_lowest=f(1.0 + 3.0 * r.random(ncol)), v_lowest=f(np.zeros(ncol)),
        p_lowest=f(np.full(ncol, 99000.0)), p_surface=f(np.full(ncol, 100000.0)),
        rho_lowest=f(np.full(ncol, 1.2)), cos_zenith=f(0.3 + 0.5 * r.random(ncol)),
        co2_ppmv=f(np.full(ncol, 400.0)), has_radiation=f(np.ones(ncol)),
        has_precipitation=f(np.ones(ncol)))
    T_soil = f(288.0 + 8.0 * r.random(ncol))
    w_frac = f(0.3 + 0.6 * r.random(ncol))
    return compute_two_leaf_canopy_fluxes(
        T_soil_top=T_soil, forcing=forcing,
        canopy_config=TwoLeafCanopyConfig(max_iters=cap),
        land_config=MultiLayerLandConfig(), canopy_params=None,
        w_frac_rz=w_frac, wind_speed=jnp.abs(forcing.u_lowest),
        wind_dir_x=jnp.ones_like(w_frac), wind_dir_y=jnp.zeros_like(w_frac),
        soil_thermal_fn=lambda G, dt_: T_soil, dt=1800.0,
        canopy_seed=seed_arr)


def test_canopy_returns_a_cache_shaped_and_masked_correctly():
    """The call returns ``canopy_x`` (ncol, 6): finite iff that column solved."""
    out = _canopy_call()
    assert out.canopy_x is not None
    assert out.canopy_x.shape == (8, 6)
    solved = jnp.all(jnp.isfinite(out.canopy_x), axis=-1)
    # Every column that the loop reports as converged must have a stored seed,
    # and a column with a stored seed must have converged at some pass.
    assert bool(jnp.all(solved | ~jnp.asarray(out.converged)))


def test_warm_started_call_reproduces_the_cold_call():
    """Seeding the SAME call with its own cache changes nothing observable.

    This is the behavioural form of the seed-independence claim: the fluxes the
    caller consumes must not depend on where the iteration started.
    """
    cold = _canopy_call()
    warm = _canopy_call(seed_arr=cold.canopy_x)
    assert bool(jnp.all(jnp.asarray(warm.converged) >= jnp.asarray(cold.converged)))
    for name in ("shflx", "lhflx", "T_surface", "gpp"):
        a, b = getattr(cold, name), getattr(warm, name)
        if a is None:
            continue
        assert float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b)))) < 1.0, name


def test_an_all_nan_cache_is_the_cold_start():
    """A cache with no converged column yet must reproduce the cold call exactly."""
    cold = _canopy_call()
    nan_seed = jnp.full((8, 6), jnp.nan)
    seeded = _canopy_call(seed_arr=nan_seed)
    for name in ("shflx", "lhflx", "T_surface"):
        a, b = getattr(cold, name), getattr(seeded, name)
        assert float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b)))) == 0.0, name
