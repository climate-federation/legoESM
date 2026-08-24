"""The leaf-temperature solver must not cycle forever, and must not call a
halved step a solution.

Runs in the model's own single precision: the oscillation is a property of the
production float32 path, so forcing double precision here would be testing a
different solver than the one that ships.

Background.  A small share of land columns -- systematically the hot, bright,
calm ones -- never satisfied the solver's stopping test no matter how many
iterations it was given: the unsolved share was flat from 25 iterations through
50.  They were not converging slowly.  They were OSCILLATING: one extra
iteration moved such a column's leaf temperature by several kelvin while every
solved column moved exactly zero.  The step limiter sustained it -- the iterate
overshot, was clipped to the same magnitude each time, and cycled.

The fix requires a step to actually reduce the residual before it is accepted,
halving it until it does, armed only once the plain step has had ~18 iterations
to work (under batching every column pays for the extra residual evaluations
the hardest one needs, so arming it from the start is pure cost).

Both tests below are written to FAIL if the safeguard is disabled.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import legoesm.land.canopy.solver as solver
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.two_leaf_canopy import (
    compute_two_leaf_canopy_fluxes)


def _hard_regime_forcing(ncol: int, seed: int = 0) -> AtmToSurface:
    """Forcing built IN the regime that defeats the plain step.

    The columns that oscillate were measured to be hot (~301 K against a 293 K
    mean), bright (~360 W/m2 against 235) and calm (~1.4 m/s against 5.2).
    Sampling a realistic global spread and hoping such a column turns up needs
    hundreds of columns before even one appears -- at 48 columns none did, and
    the test would have passed while proving nothing.  Building the regime
    directly makes a handful of columns enough, and makes the test deterministic
    rather than a lottery on the seed.
    """
    r = np.random.default_rng(seed)
    f = lambda a: jnp.asarray(a, dtype=jnp.float32)
    return AtmToSurface(
        T_lowest=f(299.0 + 9.0 * r.random(ncol)),        # hot
        q_lowest=f(0.008 + 0.006 * r.random(ncol)),
        u_lowest=f(0.3 + 1.6 * r.random(ncol)),          # calm
        v_lowest=f(np.zeros(ncol)),
        p_lowest=f(np.full(ncol, 98000.0)),
        p_surface=f(np.full(ncol, 101325.0)),
        rho_lowest=f(np.full(ncol, 1.10)),
        sw_down=f(300.0 + 500.0 * r.random(ncol)),       # bright
        lw_down=f(340.0 + 40.0 * r.random(ncol)),
        cos_zenith=f(0.4 + 0.5 * r.random(ncol)),
        precip_total=f(np.zeros(ncol)),
        precip_snow=f(np.zeros(ncol)),
        co2_ppmv=f(np.full(ncol, 410.0)),
        has_radiation=True,
        has_precipitation=False,
    )


def _unsolved_fraction(ncol: int, cap: int, seed: int = 0) -> float:
    forcing = _hard_regime_forcing(ncol, seed)
    T_soil = jnp.asarray(285.0 + 15.0 * np.random.default_rng(seed + 1).random(ncol),
                         dtype=jnp.float32)
    w_frac = jnp.asarray(np.random.default_rng(seed + 2).random(ncol), dtype=jnp.float32)
    out = compute_two_leaf_canopy_fluxes(
        T_soil_top=T_soil, forcing=forcing,
        canopy_config=TwoLeafCanopyConfig(max_iters=cap),
        land_config=MultiLayerLandConfig(), canopy_params=None,
        w_frac_rz=w_frac, wind_speed=jnp.abs(forcing.u_lowest),
        wind_dir_x=jnp.ones_like(w_frac), wind_dir_y=jnp.zeros_like(w_frac),
        soil_thermal_fn=lambda G, dt_: T_soil, dt=1800.0)
    return float(1.0 - jnp.mean(jnp.asarray(out.converged, jnp.float32)))


# WHY THESE TESTS NEED A GPU, AND WHY THEY ARE NOT MADE CHEAPER.
# The safeguard acts on a small minority of columns, through six outer
# canopy-soil passes.  Both were trimmed to make these tests fit a CPU runner --
# 48 columns, one outer pass -- and the effect VANISHED: with and without the
# safeguard both left 6.25% of columns unsolved.  The shortcut had removed the
# very conditions the fix addresses, so the test measured nothing while still
# passing its non-vacuity check.  A cheaper version of this test is a test of a
# different solver.  So the batch and the outer loop are left at realistic
# values and the behavioural tests are skipped where that is unaffordable; the
# source-level test below has no such cost and always runs.
_GPU = any(d.platform == "gpu" for d in jax.devices())
_needs_gpu = pytest.mark.skipif(
    not _GPU, reason="the safeguard's effect only appears at a realistic batch "
                     "size and outer-loop count, which is too slow on CPU")
NCOL = 512


@_needs_gpu
def test_safeguard_solves_columns_the_plain_step_cannot(monkeypatch):
    """Fewer columns are left unsolved with the safeguard than without it.

    Disabling it means arming it past the iteration limit, so the solver falls
    back to the plain clipped step it used before -- which is exactly the
    reverted state this test has to fail in.
    """
    with_guard = _unsolved_fraction(NCOL, cap=25)

    monkeypatch.setattr(solver, "_LINESEARCH_ARM_ITER", 10_000)
    without_guard = _unsolved_fraction(NCOL, cap=25)

    assert without_guard > 0.0, (
        "no column failed even without the safeguard, so this forcing cannot "
        "detect whether the safeguard works — widen the spread")
    assert with_guard < without_guard, (
        f"the safeguard left {with_guard:.4%} of columns unsolved but the plain "
        f"step left {without_guard:.4%}: it is not breaking the oscillation")


def test_convergence_is_judged_on_the_newton_step_not_the_halved_one():
    """Backtracking must not decide convergence, in either direction.

    Testing the TAKEN step would let a column look converged merely because its
    step was cut by up to 8x — the same trap a zeroed step posed before it.
    Requiring that no backtracking happened at all makes the opposite error:
    a column that needed one smaller step and then settled would be reported
    unsolved, and whatever the caller does with unsolved columns would be
    applied to a genuine root.  The test is therefore on the step the solver
    WANTED, which stays far above tolerance while a column oscillates.
    """
    src = __import__("inspect").getsource(solver._make_implicit_newton_solver)
    assert "new_converged = step_finite & (jnp.linalg.norm(delta) < tol)" in src, (
        "the convergence flag no longer tests the unbacktracked Newton step")
    assert "x_new = x + delta_taken" in src, (
        "the iterate no longer advances by the backtracked step")


@_needs_gpu
@pytest.mark.parametrize("seed", [0, 1])
def test_the_solver_still_solves_almost_everything(seed):
    """The safeguard must still solve the great majority of hard columns."""
    # Deliberately the HARD regime, so this is a demanding bound: even where
    # the plain step oscillates, the safeguarded one must still solve most
    # columns.
    assert _unsolved_fraction(NCOL, cap=25, seed=seed) < 0.35
