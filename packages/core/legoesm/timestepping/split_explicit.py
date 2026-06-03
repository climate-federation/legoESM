"""Split-explicit time integrator for non-hydrostatic dynamics.

Uses SSP-RK3 for slow modes (advection, Coriolis, diffusion) with
forward-backward acoustic substeps for fast modes (sound and gravity waves).

Within each RK3 stage:
1. Compute slow tendencies once (advection, Coriolis, horizontal PGF, diffusion)
2. Run N_s acoustic substeps via jax.lax.fori_loop:
   a. Forward: update w using vertical pressure gradient + buoyancy
   b. Backward: update rho', theta' using divergence

The horizontal winds (u, v) are only updated by the slow tendencies
(once per RK stage), not by acoustic substeps. This is valid because
horizontal acoustic propagation is slower than the advective CFL limit
at atmospheric scales.

References
----------
- Klemp, Skamarock & Dudhia (2007): An Upper Gravity-Wave Absorbing Layer
  for NWP Applications. Mon. Wea. Rev.
- Skamarock & Klemp (2008): A Time-Split Nonhydrostatic Atmospheric Model
  for Weather Research and Forecasting Applications.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, TypeVar

import jax

from legoesm.timestepping.pytree_ops import pytree_axpy as _pytree_axpy
from legoesm.timestepping.pytree_ops import pytree_linear_combination as _pytree_linear_combination

State = TypeVar("State")


class SplitExplicitConfig(NamedTuple):
    """Configuration for split-explicit time integration."""
    n_substeps: int = 6
    outer_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"


def select_n_outer_split(
    dt_outer: float,
    dx: float,
    max_wind_safe: float = 300.0,
    cfl_safe: float = 0.4,
) -> int:
    """Trace-time FV3-style outer-subcycle count from a conservative
    max-wind estimate.

    FV3's atmospheric solver picks ``n_split`` once at startup from a
    conservative max wind (the worst-case advective velocity expected
    over the run, not the instantaneous max). The chosen integer
    is then used for the entire run as a STATIC Python int — no
    while-loop, no traced control flow, no XLA retrace, fully
    reverse-differentiable through ``eqx.filter_value_and_grad``
    because the integer is a compile-time constant from JAX's
    perspective.

    This is the scan-friendly alternative to mid-run dt-shrinkage
    (iter-228 ``--adaptive-dt`` stub): instead of monitoring
    Ca_adv post-step and halving dt on violation (which requires
    a while-loop refactor + breaks the fori_loop), we pick a
    sufficiently-large static n_split upfront that covers the
    worst-case wind we expect.

    Parameters
    ----------
    dt_outer : float
        User-requested outer time step [s] (e.g. 20.0 for iter-183).
    dx : float
        Grid spacing [m] (e.g. 2000.0 for iter-183).
    max_wind_safe : float, default 300.0
        Conservative upper bound on max|w| or max|u| over the run
        [m/s]. iter-223 F11 cascade hit max|w|=225 m/s before NaN;
        300 m/s adds a 33% safety margin.
    cfl_safe : float, default 0.4
        Target advective CFL (dimensionless). 0.4 is the standard
        SK08 / FV3 conservative target (advective CFL=1 is the
        formal stability limit; 0.4 gives 2.5× margin for
        accuracy).

    Returns
    -------
    n_split : int
        Number of inner sub-steps to run per outer step. Always
        ``>= 1``. The inner step size is ``dt_outer / n_split``.

    Examples
    --------
    iter-183 production (dt=20, dx=2000):
        select_n_outer_split(20.0, 2000.0) = ceil(20·300/(0.4·2000))
        = ceil(7.5) = 8

    Steady plateau iter-183 (max|w| < 0.05 m/s):
        select_n_outer_split(20.0, 2000.0, max_wind_safe=1.0) = 1
        (the user knows the wind ceiling, opts down)

    LES dx=500 m, dt=0.5 s, F11-cascade ceiling:
        select_n_outer_split(0.5, 500.0) = ceil(0.5·300/(0.4·500))
        = ceil(0.75) = 1

    Notes
    -----
    The output is a Python ``int``, NOT a ``jnp.array``: callers
    must use it as a static argument (e.g. ``range(n_split)`` in
    Python or ``static_argnums`` in JIT). Passing it as a traced
    int forces a retrace per call.

    Reference: SHiELD / FV3 atmos_top.F90 ``n_split`` calc;
    Skamarock & Klemp 2008 split-explicit CFL discussion.
    """
    if dt_outer <= 0.0:
        raise ValueError(
            f"select_n_outer_split: dt_outer={dt_outer} must be > 0."
        )
    if dx <= 0.0:
        raise ValueError(
            f"select_n_outer_split: dx={dx} must be > 0."
        )
    if max_wind_safe <= 0.0:
        raise ValueError(
            f"select_n_outer_split: max_wind_safe={max_wind_safe} "
            f"must be > 0."
        )
    if not (0.0 < cfl_safe <= 1.0):
        raise ValueError(
            f"select_n_outer_split: cfl_safe={cfl_safe} must be in "
            f"(0, 1]."
        )
    import math
    raw = dt_outer * max_wind_safe / (cfl_safe * dx)
    return max(1, int(math.ceil(raw)))


def split_explicit_step(
    state: State,
    slow_tendency_fn: Callable[[State], State],
    acoustic_update_fn: Callable,
    dt: float,
    config: SplitExplicitConfig,
) -> State:
    """One full time step: RK3 outer + forward-backward acoustic inner.

    Parameters
    ----------
    state : pytree
        Current NonHydrostaticState.
    slow_tendency_fn : callable
        Computes slow tendencies: state -> tendencies (same pytree).
        Called once per RK stage. Updates u, v, tracers, and slow
        contributions to w, theta', rho'.
    acoustic_update_fn : callable
        Performs acoustic substeps on the fast variables (w, theta', rho').
        Signature: (state, slow_tend, dt_substep, n_substeps, config) -> state.
    dt : float
        Full time step [seconds].
    config : SplitExplicitConfig
        Substep configuration.

    Returns
    -------
    state : pytree
        State after one time step.
    """
    integrator = config.outer_integrator.lower()
    if integrator in ("ssp_rk3", "ssp3", "rk3"):
        return _split_explicit_ssp_rk3(
            state, slow_tendency_fn, acoustic_update_fn, dt, config,
        )
    if integrator in ("ssp_rk34", "ssp34", "rk34"):
        return _split_explicit_ssp_rk34(
            state, slow_tendency_fn, acoustic_update_fn, dt, config,
        )
    if integrator in ("ssp_rk54", "ssp54", "ssp45", "rk54"):
        return _split_explicit_ssp_rk54(
            state, slow_tendency_fn, acoustic_update_fn, dt, config,
        )
    raise ValueError(f"Unsupported outer_integrator={config.outer_integrator!r}")


def _split_explicit_ssp_rk3(
    state: State,
    slow_tendency_fn: Callable[[State], State],
    acoustic_update_fn: Callable,
    dt: float,
    config: SplitExplicitConfig,
) -> State:
    """SSP-RK3 outer integrator with acoustic substeps at each stage."""
    # Stage 1: k1 = state + dt * F(state) + acoustic_update
    tend_0 = slow_tendency_fn(state)
    k1 = _rk_stage_with_acoustics(
        state, tend_0, dt, 1.0, acoustic_update_fn, config,
    )

    # Stage 2: k2 = 3/4 * state + 1/4 * (k1 + dt * F(k1) + acoustic)
    tend_1 = slow_tendency_fn(k1)
    k1_stepped = _rk_stage_with_acoustics(
        k1, tend_1, dt, 1.0, acoustic_update_fn, config,
    )
    k2 = _pytree_linear_combination(state, k1_stepped, 0.75, 0.25)

    # Stage 3: k3 = 1/3 * state + 2/3 * (k2 + dt * F(k2) + acoustic)
    tend_2 = slow_tendency_fn(k2)
    k2_stepped = _rk_stage_with_acoustics(
        k2, tend_2, dt, 1.0, acoustic_update_fn, config,
    )
    return _pytree_linear_combination(state, k2_stepped, 1.0 / 3.0, 2.0 / 3.0)


def _split_explicit_ssp_rk54(
    state: State,
    slow_tendency_fn: Callable[[State], State],
    acoustic_update_fn: Callable,
    dt: float,
    config: SplitExplicitConfig,
) -> State:
    """SSP-RK(5,4) outer integrator with acoustic substeps at each stage."""
    # Coefficients from Spiteri & Ruuth (2002), same as ssp_rk54_step.
    a20, a21 = 0.444370493651235, 0.555629506348765
    a30, a32 = 0.620101851488403, 0.379898148511597
    a40, a43 = 0.178079954393132, 0.821920045606868
    a52, a53, a54 = 0.517231671970585, 0.096059710526147, 0.386708617503269
    b10 = 0.391752226571890
    b21 = 0.368410593050371
    b32 = 0.251891774271694
    b43 = 0.544974750228521
    b53 = 0.063692468666290
    b54 = 0.226007483236906

    # Stage 1
    f0 = slow_tendency_fn(state)
    u1 = _rk_stage_with_acoustics(
        state, f0, dt, b10, acoustic_update_fn, config,
    )

    # Stage 2
    f1 = slow_tendency_fn(u1)
    u2_base = _pytree_linear_combination(state, u1, a20, a21)
    u2 = _rk_stage_with_acoustics(
        u2_base, f1, dt, b21, acoustic_update_fn, config,
    )

    # Stage 3
    f2 = slow_tendency_fn(u2)
    u3_base = _pytree_linear_combination(state, u2, a30, a32)
    u3 = _rk_stage_with_acoustics(
        u3_base, f2, dt, b32, acoustic_update_fn, config,
    )

    # Stage 4
    f3 = slow_tendency_fn(u3)
    u4_base = _pytree_linear_combination(state, u3, a40, a43)
    u4 = _rk_stage_with_acoustics(
        u4_base, f3, dt, b43, acoustic_update_fn, config,
    )

    # Stage 5
    f4 = slow_tendency_fn(u4)
    u5_base = jax.tree.map(
        lambda s2, s3, s4: a52 * s2 + a53 * s3 + a54 * s4,
        u2, u3, u4,
    )
    f54 = jax.tree.map(lambda ff3, ff4: b53 * ff3 + b54 * ff4, f3, f4)
    return _rk_stage_with_acoustics(
        u5_base, f54, dt, 1.0, acoustic_update_fn, config,
    )


def _split_explicit_ssp_rk34(
    state: State,
    slow_tendency_fn: Callable[[State], State],
    acoustic_update_fn: Callable,
    dt: float,
    config: SplitExplicitConfig,
) -> State:
    """SSP-RK(4,3) outer integrator with acoustic substeps at each stage."""
    # Stage 1: u1 = u0 + 1/2 dt F(u0)
    f0 = slow_tendency_fn(state)
    u1 = _rk_stage_with_acoustics(
        state, f0, dt, 0.5, acoustic_update_fn, config,
    )

    # Stage 2: u2 = u1 + 1/2 dt F(u1)
    f1 = slow_tendency_fn(u1)
    u2 = _rk_stage_with_acoustics(
        u1, f1, dt, 0.5, acoustic_update_fn, config,
    )

    # Stage 3: u3 = 2/3 u0 + 1/3 u2 + 1/6 dt F(u2)
    f2 = slow_tendency_fn(u2)
    u3_base = _pytree_linear_combination(state, u2, 2.0 / 3.0, 1.0 / 3.0)
    u3 = _rk_stage_with_acoustics(
        u3_base, f2, dt, 1.0 / 6.0, acoustic_update_fn, config,
    )

    # Stage 4: u4 = u3 + 1/2 dt F(u3)
    f3 = slow_tendency_fn(u3)
    return _rk_stage_with_acoustics(
        u3, f3, dt, 0.5, acoustic_update_fn, config,
    )


def _rk_stage_with_acoustics(
    state, slow_tend, dt, weight, acoustic_update_fn, config,
):
    """Apply slow tendency to u/v/tracers, then run acoustic substeps."""
    # First, apply the slow tendency update to the full state
    state_slow = _pytree_axpy(state, slow_tend, dt * weight)

    # Then run acoustic substeps to update w, theta', rho'
    dt_substep = dt * weight / config.n_substeps
    state_out = acoustic_update_fn(
        state_slow, slow_tend, dt_substep, config.n_substeps, config,
    )
    return state_out


