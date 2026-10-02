"""A land column the model could not solve must not take the model with it.

Two independent pieces are pinned here.

**The canopy Newton solver must not call a discarded step "converged".**  A
singular Jacobian makes the linear solve return a non-finite Newton step.  That
step is replaced by zero so the iterate stays finite — but a zero step also has
zero norm, so the convergence test read the substitution as success: the solver
reported a root on exactly the column whose Jacobian had just collapsed, and the
backward pass's convergence mask (there to zero the gradient of a non-root)
never fired.

**A non-finite column is held, not spent.**  In the coupled model one column of
2562 going non-finite reached the atmosphere through the land skin temperature
and, through the dynamical core's global mass fixer, made every column
non-finite one step later — a 5-day run died 7 hours in.  The guard freezes such
a column and leaves every other column untouched.  A FINITE unsolved column is
held here only without ``fallback_ok``; with it (two-leaf canopy, no elevation
bands) it is accepted with energy-closed fallback fluxes, tested in
``test_land_unsolved_fallback.py``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.land.config import MultiLayerLandConfig, SoilGridConfig
from legoesm.land.state import MultiLayerLandState
from legoesm.land.multilayer_land import _hold_unsolved_columns
from legoesm.land.surface_scheme.base import SurfaceFluxOutput
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure


# --------------------------------------------------------------------------- #
# the solver's convergence flag                                               #
# --------------------------------------------------------------------------- #

def _bundle(**over):
    base = dict(
        LAI=0.02, SZA=30.0, La=340.0, epsf=0.97, epss=0.96, fSun=0.5,
        APAR_Sun=200.0, APAR_Sh=50.0, Vcmax25_Sun=40.0, Vcmax25_Sh=20.0,
        Vcmax25_C4Sun=0.0, Vcmax25_C4Sh=0.0,
        ASW_Sun=200.0, ASW_Sh=60.0, ASW_Soil=400.0, Ts_bc=290.0,
        Ca=400.0, Ps=95000.0, Ta=288.0, lam=2.45e6, Cp=1005.0, rhoa=1.15,
        Tv_atm=289.0, q_atm=0.006, m=9.0, b0=0.01, alf=0.3, TgC=20.0,
        fC4=0.0, fStress_soil=0.5, ur=2.0, CI=0.75, z0m=0.02, displa=0.01,
        z0=10.0, cv=0.0135, d_leaf=0.025, r_soil_surface=100.0, fwet=0.0,
    )
    base.update(over)
    return CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in base.items()})


_X0 = jnp.array([288.0, 288.0, 280.0, 280.0, 288.0, 0.006])


def test_solver_reports_convergence_on_a_normal_column():
    _, n_iters, converged = solve_canopy_closure(_X0, _bundle(), CanopyConfig())
    assert bool(converged), "a well-posed column did not converge"
    assert int(n_iters) < CanopyConfig().max_iters


def test_a_discarded_non_finite_newton_step_is_not_convergence():
    # REGRESSION.  A non-finite forcing makes the residual and its Jacobian
    # non-finite, so the Newton step is non-finite and gets replaced by zero.
    # The zero step must NOT be read as "converged": before this was fixed the
    # solver returned success here on the first iteration.
    _, n_iters, converged = solve_canopy_closure(
        _X0, _bundle(La=jnp.nan), CanopyConfig())
    assert not bool(converged), (
        "the solver called a discarded (zeroed) step a converged "
        "solution — the caller would spend its fluxes as physics")
    # The damped least-squares solver bails a non-finite column early via the
    # damping ceiling (lambda -> lambda_max) rather than running the full
    # max_iters, so the count is <= the cap, not exactly it.  The invariant that
    # matters is `not converged`, asserted above.
    assert int(n_iters) <= CanopyConfig().max_iters


# --------------------------------------------------------------------------- #
# the per-column hold                                                          #
# --------------------------------------------------------------------------- #

_NCOL = 3
_CFG = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=4, total_depth=2.0))


def _state(T0, theta0):
    n = _NCOL
    return MultiLayerLandState(
        T_soil=jnp.broadcast_to(jnp.asarray(T0)[:, None], (n, 4)) * 1.0,
        psi_soil=jnp.full((n, 4), -1.0),
        theta_soil=jnp.broadcast_to(jnp.asarray(theta0)[:, None], (n, 4)) * 1.0,
        runoff_surface=jnp.zeros(n), runoff_subsurface=jnp.zeros(n),
        snow_depth=jnp.zeros(n), snow_age=jnp.zeros(n),
        surface_water=jnp.zeros(n),
    )


def _new_state():
    return _state([285.0, 286.0, 287.0], [0.33, 0.34, 0.35])  # const-ok: fixed test soil T[K]/moisture


def _response(**over):
    n = _NCOL
    base = {f: jnp.zeros(n) for f in TileResponse._fields}
    base.update(T_sfc=jnp.full(n, 290.0), T_rad=jnp.full(n, 290.0),
                albedo=jnp.full(n, 0.25), emissivity=jnp.full(n, 0.97),
                z0=jnp.full(n, 0.1), q_surface=jnp.full(n, 0.008),
                shflx=jnp.full(n, 40.0), lhflx=jnp.full(n, 60.0),
                lw_up=jnp.full(n, 390.0))
    base.update(over)
    return TileResponse(**base)


def _forcing():
    n = _NCOL
    return AtmToSurface(
        T_lowest=jnp.full(n, 288.0), q_lowest=jnp.full(n, 0.005),
        u_lowest=jnp.full(n, 2.0), v_lowest=jnp.zeros(n),
        p_lowest=jnp.full(n, 95000.0), p_surface=jnp.full(n, 97000.0),
        rho_lowest=jnp.full(n, 1.15), sw_down=jnp.full(n, 300.0),
        lw_down=jnp.full(n, 320.0), cos_zenith=jnp.full(n, 0.5),
        precip_total=jnp.zeros(n), precip_snow=jnp.zeros(n),
        co2_ppmv=jnp.full(n, 420.0), has_radiation=jnp.ones(n),
        has_precipitation=jnp.ones(n))


def _run(new_state, response, converged=None):
    old = _state([280.0, 281.0, 282.0], [0.30, 0.31, 0.32])
    out = SurfaceFluxOutput(
        shflx=response.shflx, lhflx=response.lhflx, tau_x=response.tau_x,
        tau_y=response.tau_y, sw_net=jnp.zeros(_NCOL), lw_net=jnp.zeros(_NCOL),
        lw_up=response.lw_up, G_soil=jnp.zeros(_NCOL),
        T_surface=response.T_sfc, q_surface=response.q_surface,
        albedo=response.albedo, emissivity=response.emissivity, z0=response.z0,
        converged=converged)
    held, held_resp, _carbon, mask, n_held = _hold_unsolved_columns(
        old, new_state, response, out, _forcing(), _CFG, _NCOL)[:5]
    return old, (held, held_resp, mask, n_held)


def test_a_healthy_step_is_untouched():
    new = _new_state()
    resp = _response()
    _, (held, held_resp, mask, n_held) = _run(new, resp, converged=jnp.ones(_NCOL, bool))
    for f in new._fields:
        a, b = getattr(new, f), getattr(held, f)
        if a is None:
            continue
        assert jnp.array_equal(a, b), f"healthy column state changed in {f}"
    for f in ("shflx", "lhflx", "T_sfc", "albedo"):
        assert jnp.array_equal(getattr(resp, f), getattr(held_resp, f))
    assert int(n_held) == 0, "a healthy step reported held columns"
    assert not any(bool(v) for v in mask)


def test_a_non_finite_column_is_reverted_and_its_neighbours_are_not():
    new = _new_state()
    bad_T = new.T_soil.at[1, 0].set(jnp.nan)
    new = new._replace(T_soil=bad_T)
    old, (held, held_resp, mask, n_held) = _run(new, _response())
    # column 1 reverted to the start of the step
    assert np.allclose(np.asarray(held.T_soil[1]), np.asarray(old.T_soil[1]))
    assert np.allclose(np.asarray(held.theta_soil[1]),
                       np.asarray(old.theta_soil[1]))
    assert np.all(np.isfinite(np.asarray(held.T_soil)))
    # columns 0 and 2 advanced as computed
    for c in (0, 2):
        assert np.allclose(np.asarray(held.T_soil[c]), np.asarray(new.T_soil[c]))
    # the held column exchanges nothing
    assert float(held_resp.shflx[1]) == 0.0
    assert float(held_resp.lhflx[1]) == 0.0
    assert float(held_resp.T_sfc[1]) == pytest.approx(float(old.T_soil[1, 0]))
    assert float(held_resp.q_surface[1]) == pytest.approx(0.005)  # = q_air
    # the count and mask report exactly that one column
    assert int(n_held) == 1
    assert [bool(v) for v in mask] == [False, True, False]
    # neighbours keep their fluxes
    assert float(held_resp.shflx[0]) == 40.0
    assert float(held_resp.lhflx[2]) == 60.0


def test_a_non_converged_column_is_held_even_though_it_is_finite():
    # Without ``fallback_ok`` (a scheme whose unsolved energy the caller did not
    # close) a FINITE stopped iterate is still reverted: a canopy closure that
    # hit its iteration cap can return large, plausible-looking fluxes.  The
    # accepted-fallback path is tested in test_land_unsolved_fallback.py.
    new = _new_state()
    resp = _response(shflx=jnp.array([40.0, -2040.0, 40.0]),
                     lhflx=jnp.array([60.0, -3231.0, 60.0]))
    old, (held, held_resp, mask, n_held) = _run(
        new, resp, converged=jnp.array([True, False, True]))
    assert np.allclose(np.asarray(held.T_soil[1]), np.asarray(old.T_soil[1]))
    assert float(held_resp.shflx[1]) == 0.0
    assert float(held_resp.lhflx[1]) == 0.0
    for c in (0, 2):
        assert np.allclose(np.asarray(held.T_soil[c]), np.asarray(new.T_soil[c]))
    assert float(held_resp.shflx[0]) == 40.0
    assert int(n_held) == 1


def test_the_guard_survives_jit():
    new = _new_state()
    new = new._replace(T_soil=new.T_soil.at[0, 2].set(jnp.inf))
    old = _state([280.0, 281.0, 282.0], [0.30, 0.31, 0.32])
    resp = _response()
    out = SurfaceFluxOutput(
        shflx=resp.shflx, lhflx=resp.lhflx, tau_x=resp.tau_x, tau_y=resp.tau_y,
        sw_net=jnp.zeros(_NCOL), lw_net=jnp.zeros(_NCOL), lw_up=resp.lw_up,
        G_soil=jnp.zeros(_NCOL), T_surface=resp.T_sfc,
        q_surface=resp.q_surface, albedo=resp.albedo,
        emissivity=resp.emissivity, z0=resp.z0, converged=None)
    f = jax.jit(lambda a, b, c, d: _hold_unsolved_columns(
        a, b, c, d, _forcing(), _CFG, _NCOL))
    held, held_resp, _carbon, mask, n_held = f(old, new, resp, out)[:5]
    assert int(n_held) == 1 and bool(mask[0])
    assert np.all(np.isfinite(np.asarray(held.T_soil)))
    assert np.allclose(np.asarray(held.T_soil[0]), np.asarray(old.T_soil[0]))


def test_a_bfloat16_leaf_is_still_checked():
    # REGRESSION: the non-finiteness test filtered leaves by ``dtype.kind``,
    # and bfloat16 (an extension dtype) reports kind "V" — so a bfloat16 NaN
    # was skipped entirely and the column was never held (reproduced by review).
    new = _new_state()
    bf = new.T_soil.astype(jnp.bfloat16).at[1, 0].set(jnp.nan)
    new = new._replace(T_soil=bf)
    old, (held, held_resp, mask, n_held) = _run(new, _response())
    assert int(n_held) == 1, "a bfloat16 NaN went unheld"
    assert bool(mask[1])


def test_a_nested_carrier_is_held_too():
    # The land state can carry a nested pytree (the multilayer-canopy state).
    # Holding only the top-level array fields left a NaN inside the carrier.
    new = _new_state()
    good = {"inner": jnp.arange(_NCOL, dtype=float)}
    bad_inner = {"inner": jnp.arange(_NCOL, dtype=float).at[1].set(jnp.nan)}
    old_state = _state([280.0, 281.0, 282.0], [0.30, 0.31, 0.32])._replace(
        canopy_state=good)
    new = new._replace(canopy_state=bad_inner)
    out = SurfaceFluxOutput(
        shflx=jnp.zeros(_NCOL), lhflx=jnp.zeros(_NCOL), tau_x=jnp.zeros(_NCOL),
        tau_y=jnp.zeros(_NCOL), sw_net=jnp.zeros(_NCOL),
        lw_net=jnp.zeros(_NCOL), lw_up=jnp.zeros(_NCOL),
        G_soil=jnp.zeros(_NCOL), T_surface=jnp.full(_NCOL, 290.0),
        q_surface=jnp.zeros(_NCOL), albedo=jnp.full(_NCOL, 0.2),
        emissivity=jnp.full(_NCOL, 0.97), z0=jnp.full(_NCOL, 0.1))
    held, _resp, _carbon, mask, n_held = _hold_unsolved_columns(
        old_state, new, _response(), out, _forcing(), _CFG, _NCOL)[:5]
    assert int(n_held) == 1
    assert np.all(np.isfinite(np.asarray(held.canopy_state["inner"]))), \
        "the nested carrier kept its non-finite value"


def test_carbon_pools_revert_with_the_column():
    # The carbon pools advance from the same rejected surface state, so a
    # column held in the soil but advanced in carbon would carry that
    # inconsistency into the restart file.
    new = _new_state()
    new = new._replace(T_soil=new.T_soil.at[1, 0].set(jnp.nan))
    old = _state([280.0, 281.0, 282.0], [0.30, 0.31, 0.32])
    carbon_old = {"C_fol": jnp.array([100.0, 110.0, 120.0])}
    carbon_new = {"C_fol": jnp.array([101.0, 999.0, 121.0])}
    out = SurfaceFluxOutput(
        shflx=jnp.zeros(_NCOL), lhflx=jnp.zeros(_NCOL), tau_x=jnp.zeros(_NCOL),
        tau_y=jnp.zeros(_NCOL), sw_net=jnp.zeros(_NCOL),
        lw_net=jnp.zeros(_NCOL), lw_up=jnp.zeros(_NCOL),
        G_soil=jnp.zeros(_NCOL), T_surface=jnp.full(_NCOL, 290.0),
        q_surface=jnp.zeros(_NCOL), albedo=jnp.full(_NCOL, 0.2),
        emissivity=jnp.full(_NCOL, 0.97), z0=jnp.full(_NCOL, 0.1))
    _held, _resp, carbon, _mask, n_held = _hold_unsolved_columns(
        old, new, _response(), out, _forcing(), _CFG, _NCOL,
        carbon_old=carbon_old, carbon_new=carbon_new)[:5]
    assert int(n_held) == 1
    assert float(carbon["C_fol"][1]) == 110.0, "carbon advanced on a held column"
    assert float(carbon["C_fol"][0]) == 101.0, "carbon froze on a healthy column"


def _run_mpas_source():
    import inspect

    from legoesm.driver.model_driver import ModelDriver

    return inspect.getsource(ModelDriver._run_mpas)


def test_the_production_lane_carries_the_hold_count_out_of_the_land_step():
    """A hold that nothing reports is a run quietly freezing part of its land.

    The containment guard's own docstring says the caller must surface the
    count, and no caller did: the coupled driver unpacked the three-value
    result and the status went nowhere. Keyed off the function that actually
    runs, not a wrapper around it.
    """
    src = _run_mpas_source()
    assert "step_multilayer_land_with_diagnostics" in src, (
        "the production land step uses the three-value form, which discards "
        "the containment status")
    assert "n_held" in src


def test_the_hold_count_is_actually_logged():
    """Carrying the number out of the step is not the same as reporting it.

    An earlier version of this test searched the whole function for a word and
    passed with the warning deleted, because a nearby comment still mentioned
    holds. This one finds the logging calls themselves, so removing the call
    goes red whatever the comments say.
    """
    import ast

    tree = ast.parse("if True:\n" + _run_mpas_source())
    logged = []
    for node in ast.walk(tree):
        fn = getattr(node, "func", None)
        if not isinstance(node, ast.Call) or not isinstance(fn, ast.Attribute):
            continue
        if fn.attr not in ("warning", "error"):
            continue
        if not (isinstance(fn.value, ast.Name) and fn.value.id == "logger"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            logged.append(str(node.args[0].value))
    assert logged, "no logger warnings found in the production lane at all"
    hold_messages = [m for m in logged if "held" in m.lower()]
    assert hold_messages, (
        "the production lane logs no warning about held columns, so the count "
        f"is computed and thrown away. Messages found: {logged[:6]}")
    # Column-steps alone cannot separate one column failing every step from
    # many columns failing once, and those are different problems, so the
    # number of steps that held has to be reported too.
    assert "steps" in hold_messages[0].lower()
    # Accepted fallbacks and guard rejections are reported too (D4).
    assert any("fallback" in m.lower() and "rejected" in m.lower()
               for m in logged), logged[:8]


def test_the_hold_count_is_read_at_a_cadence_and_cannot_overflow():
    """Two ways this reporting could quietly stop working.

    Reading a device scalar every step stalls the accelerator once per step
    for a number that is almost always zero. And a 32-bit counter accumulating
    a whole run's column-steps wraps: ten thousand columns at a seventy-five
    second timestep pass two billion in about half a simulated year, after
    which a comparison against a previous maximum would never fire again. So
    the device counter is reset at each read and the running totals live on
    the host as Python integers.
    """
    src = _run_mpas_source()
    assert "_HARD_SAT_LOG_CADENCE_STEPS" in src
    resets = src.count("_land_n_held_accum = jnp.zeros((), jnp.int32)")
    assert resets >= 2, (
        "the device hold counter is never reset, so it accumulates the whole "
        "run in 32 bits and wraps")
    assert "self._land_n_held_total += " in src
