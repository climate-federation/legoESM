"""Newtonian relaxation toward an ANALYTIC RCE-style temperature target.

NOT a gray-radiation equilibrium solver (Codex iter-1): this module
applies a Newtonian relaxation toward a HAND-PRESCRIBED analytic
target T_eq(z), which mimics the gross shape of a tropical RCE
profile (warm surface, exp decay with height, stratospheric cap).
The output is NOT in radiative flux balance and is NOT a solve of
the radiative-transfer equation; it is a CHEAP PRECONDITIONER that
removes the worst cold-start shock when the downstream dynamics
+ gray radiation are first activated.

Usage
-----
Call once at IC setup time to replace the HeightCoordinate's
reference state with the preconditioned profile. The downstream
dynamics + radiation then start from a state that is closer to
true RCE than the isothermal default — the radiation shock during
the first sim-hours is muted but not eliminated.

For a TRUE column gray-radiation equilibrium, run the gray
radiation backend (``legoesm.atmosphere.physics.radiation.gray``)
in a column-only no-dynamics loop until heating rates fall below
a tolerance; that's a separate harness, out of scope here.

Stability
---------
The relaxation uses the EXACT exponential update
``T_new = T_eq + (T - T_eq) · exp(-dt / tau)`` (Codex iter-1: the
explicit Euler form was unstable above dt/tau > 2). Stable for
any positive dt, tau.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.vertical import (
    HeightCoordinate, compute_reference_state,
)


def relax_to_analytic_rce_temperature_target(
    height_coord: HeightCoordinate,
    T_sfc: float = 300.0,
    n_steps: int = 240,
    dt: float = 360.0,
    tau: float = 86_400.0,
    H_scale: float = 7_500.0,
    softening_factor: float = 4.0,
    T_strato_floor: float = 200.0,
) -> HeightCoordinate:
    """Newtonian-relax column T toward an analytic RCE temperature target.

    NOT a gray-radiation equilibrium solver. The target is a
    HAND-PRESCRIBED tropical-mean-style profile::

        T_eq(z) = max(T_sfc · exp(-z / (softening_factor · H_scale)),
                      T_strato_floor)

    Suitable as a CHEAP IC PRECONDITIONER to mute the cold-start
    radiation shock; downstream caller should still allow ~1 sim-
    day of radiation+dynamics equilibration before treating the
    state as RCE.

    Parameters
    ----------
    height_coord : HeightCoordinate
        Starting vertical grid + reference state.
    T_sfc : float
        Surface T anchor [K].
    n_steps, dt, tau : float
        Relaxation iterations + step + e-folding timescale [s].
        Uses the EXACT exponential update so any positive ratio is
        stable.
    H_scale : float
        Atmospheric scale-height for the analytic target [m]. Wing
        2018-style default 7.5 km. Codex iter-1 fix — was hardcoded.
    softening_factor : float
        Dimensionless multiplier broadening the exponential decay
        (target ≈ T_sfc at the surface, T_sfc · e⁻¹ at z ≈
        softening_factor·H_scale). Default 4 keeps the target above
        the stratospheric floor for the typical NLEV=40, H=33km
        configuration.
    T_strato_floor : float
        Stratospheric temperature floor [K]. Default 200 K.

    Returns
    -------
    HeightCoordinate
        Same z grid; theta_ref / rho_ref / exner_ref replaced with
        the relaxed profile via Poisson reconstruction.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps={n_steps} must be >= 1.")
    if tau <= 0.0:
        raise ValueError(f"tau={tau} must be positive.")
    if dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive.")
    if H_scale <= 0.0:
        raise ValueError(f"H_scale={H_scale} must be positive.")
    if softening_factor <= 0.0:
        raise ValueError(
            f"softening_factor={softening_factor} must be positive."
        )
    z_full = height_coord.z_full
    T_eq = T_sfc * jnp.exp(
        -z_full / (softening_factor * H_scale)
    )
    T_eq = jnp.maximum(T_eq, T_strato_floor)

    T_current = height_coord.theta_ref * height_coord.exner_ref

    # Exact exponential relaxation: T_new = T_eq + (T - T_eq)·e^{-dt/tau}
    # — Codex iter-1: stable for any positive dt/tau (explicit Euler
    # was unstable above dt/tau > 2).
    decay_per_step = jnp.exp(-dt / tau)

    def _step(_, T):
        return T_eq + (T - T_eq) * decay_per_step

    T_final = jax.lax.fori_loop(
        0, n_steps, _step, T_current,
    )

    # Reconstruct θ from T via the Poisson identity, using the
    # supplied exner_ref as the pressure proxy (consistent with the
    # original HeightCoordinate construction).
    theta_final = T_final / jnp.maximum(
        height_coord.exner_ref, 1.0e-6,
    )

    # Build a closure for compute_reference_state that returns the
    # relaxed θ as a function of z (matched at the existing z_full
    # via interpolation).
    def _theta_ref_fn(z):
        # Use jnp.interp for piecewise-linear sampling on z_full.
        # The grid is top-to-bottom (z_full[0] = top); jnp.interp
        # requires sorted input → flip to surface→top.
        z_s2t = z_full[::-1]
        theta_s2t = theta_final[::-1]
        return jnp.interp(z, z_s2t, theta_s2t)

    rho_ref, theta_ref, exner_ref = compute_reference_state(
        z_full, _theta_ref_fn,
    )
    rho_ref_half, _, exner_ref_half = compute_reference_state(
        height_coord.z_half, _theta_ref_fn,
    )

    return HeightCoordinate(
        n_levels=height_coord.n_levels,
        H=height_coord.H,
        z_full=height_coord.z_full,
        z_half=height_coord.z_half,
        dz=height_coord.dz,
        dz_half=height_coord.dz_half,
        rho_ref=rho_ref,
        theta_ref=theta_ref,
        exner_ref=exner_ref,
        exner_ref_half=exner_ref_half,
        rho_ref_half=rho_ref_half,
    )


def relax_to_radiative_equilibrium_column(*args, **kwargs):
    """DEPRECATED: use :func:`relax_to_analytic_rce_temperature_target`.

    The original name implied a gray-radiation equilibrium solve;
    the implementation is actually a Newtonian relaxation toward a
    HAND-PRESCRIBED analytic target (Codex iter-1 / iter-2). Kept
    as a deprecated wrapper for one release cycle so downstream
    imports do not break; emits ``DeprecationWarning``.
    """
    import warnings
    warnings.warn(
        "relax_to_radiative_equilibrium_column is deprecated; use "
        "relax_to_analytic_rce_temperature_target — the function is "
        "a Newtonian preconditioner toward an analytic target, NOT "
        "a gray-radiation equilibrium solver.",
        DeprecationWarning, stacklevel=2,
    )
    return relax_to_analytic_rce_temperature_target(*args, **kwargs)
