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
import jax.numpy as jnp

State = TypeVar("State")


class SplitExplicitConfig(NamedTuple):
    """Configuration for split-explicit time integration."""
    n_substeps: int = 6
    off_centering: float = 0.5
    divergence_damping: float = 0.0


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
    k3 = _pytree_linear_combination(state, k2_stepped, 1.0 / 3.0, 2.0 / 3.0)

    return k3


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


def _pytree_axpy(x, y, alpha):
    """Compute x + alpha * y for two pytrees."""
    return jax.tree.map(lambda xi, yi: xi + alpha * yi, x, y)


def _pytree_linear_combination(x, y, a, b):
    """Compute a * x + b * y for two pytrees."""
    return jax.tree.map(lambda xi, yi: a * xi + b * yi, x, y)
