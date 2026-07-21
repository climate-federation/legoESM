"""Runtime CFL diagnostic for the CRM dycores.

Provides JAX-traceable helpers to compute the per-step
horizontal advective, vertical advective, and acoustic Courant
numbers from a state + grid + dt; plus a hard-fail wrapper that
raises a Python ``ValueError`` *after* the trace if a configured
bound is exceeded.

The diagnostic itself runs inside JIT (pure-JAX reductions) and
returns a scalar Courant number per category. The bound check is
host-side via ``jax.block_until_ready`` + ``float(...)``; suitable
for an outer integration loop that checks every N steps.

Reuses :data:`legoesm.constants` for the dry-air sound speed:
``c_s ≈ sqrt(γ R_d T)`` with γ = c_pd / c_vd.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants


class CourantNumbers(NamedTuple):
    """Per-step Courant numbers (dimensionless).

    Attributes
    ----------
    horizontal_advective : jax.Array
        ``max(|u|·dt/dx, |v|·dt/dy)`` over the domain.
    vertical_advective : jax.Array
        ``max(|w|·dt/dz)`` over the domain.
    acoustic : jax.Array
        ``max(c_s·dt/min(dx, dy, dz))`` over the domain.
    """
    horizontal_advective: jax.Array
    vertical_advective: jax.Array
    acoustic: jax.Array


def column_sound_speed_upper_bound(height_coord, theta_prime=None) -> jax.Array:
    """Conservative UPPER BOUND on the dry-air sound speed ``c_s ≈ sqrt(γ R_d T_max)``
    over a column, ``γ = c_pd/c_vd``.

    ``T_max ≈ max(θ_ref + |θ'|) · max(π_ref)`` — the two maxima may occur at DIFFERENT
    levels, so the product OVERestimates the true ``max(T)`` (intentional: it only ever
    inflates the reported acoustic Courant, never under-reports).  ``theta_prime``
    (``state.theta_prime.data``) adds the warm-perturbation contribution; ``None`` uses
    only ``θ_ref`` — the REST-state bound, exact for a pre-run pre-flight where no eddies
    have developed.  The single home for the sound-speed bound, reused by the Courant
    diagnostics (no re-derivation).
    """
    gamma = constants.c_pd / constants.c_vd
    theta_total = height_coord.theta_ref
    if theta_prime is not None:
        theta_total = theta_total + jnp.abs(theta_prime)
    T_max = jnp.max(theta_total) * jnp.max(height_coord.exner_ref)
    return jnp.sqrt(gamma * constants.R_d * T_max)


def acoustic_courant_horizontal(
    height_coord, grid, dt: float, n_acoustic_substeps: int = 1, theta_prime=None,
) -> jax.Array:
    """HORIZONTAL acoustic Courant ``c_s · (dt/n_acoustic) / min(dx, dy)``.

    Distinct from :class:`CourantNumbers`.``acoustic`` (which uses ``min(dx, dy, dz)``):
    when the dycore solves the VERTICAL acoustic mode SEMI-IMPLICITLY (unconditionally
    stable in the vertical — e.g. the plane LES with ``semi_implicit_acoustic=True`` +
    ``substep_horizontal_acoustic=True``), the small ``dz`` does NOT bind the acoustic
    CFL; only the explicit HORIZONTAL acoustic mode does, so the binding length is the
    horizontal spacing.  Wind-independent (depends on ``c_s``, ``dt``, ``n_acoustic``,
    ``dx`` only), so it is meaningful on a REST state — a valid PRE-FLIGHT before the run.
    """
    if n_acoustic_substeps < 1:
        raise ValueError(
            f"n_acoustic_substeps={n_acoustic_substeps} must be >= 1 "
            "(acoustic CFL meaningless for zero/negative substep counts).")
    c_sound = column_sound_speed_upper_bound(height_coord, theta_prime)
    dx_min = jnp.minimum(grid.dx, grid.dy)
    return c_sound * (dt / float(n_acoustic_substeps)) / dx_min


def compute_courant_numbers_plane(
    state, height_coord, grid, dt: float,
    n_acoustic_substeps: int = 1,
) -> CourantNumbers:
    """Compute Courant numbers for a ``PlaneNonHydrostaticState``.

    Parameters
    ----------
    state : PlaneNonHydrostaticState
    height_coord : HeightCoordinate
        Provides ``dz`` per level.
    grid : PlaneGrid
        Provides ``dx``, ``dy``.
    dt : float
        Outer time step [s].
    n_acoustic_substeps : int
        Codex iter-1: split-explicit acoustic substepping divides
        the outer ``dt`` by this factor before checking the acoustic
        CFL. Default 1 (= acoustic uses the full outer dt, the most
        conservative assumption); for the standard plane CRM config
        with ``n_acoustic_substeps=6`` the acoustic Courant is
        6× smaller than this default would report.

    Notes
    -----
    Sound speed is a CONSERVATIVE UPPER BOUND:
    ``c_sound ≲ sqrt(γ · R_d · max(θ_ref + |θ'|) · max(exner_ref))``.
    The two maxima may occur at DIFFERENT vertical levels (e.g.
    max θ in the warm BL but max exner_ref at the surface), so the
    product overestimates the true ``max(T)``. This is intentional
    — overestimating c_sound only ever inflates the reported
    acoustic Courant, never under-reports. A perturbation-aware
    Exner reconstruction (calling ``pressure_from_eos`` + Poisson
    inversion) would be exact but adds an EOS evaluation per
    diagnostic call; out of scope for the smoke diagnostic.
    """
    if n_acoustic_substeps < 1:
        raise ValueError(
            f"n_acoustic_substeps={n_acoustic_substeps} must be >= 1 "
            f"(acoustic CFL meaningless for zero/negative substep counts)."
        )
    max_abs_u = jnp.max(jnp.abs(state.u.data))
    max_abs_v = jnp.max(jnp.abs(state.v.data))
    max_abs_w = jnp.max(jnp.abs(state.w.data))
    c_h = jnp.maximum(
        max_abs_u * dt / grid.dx,
        max_abs_v * dt / grid.dy,
    )
    dz_min = jnp.min(height_coord.dz)
    c_v = max_abs_w * dt / dz_min
    # Acoustic: include the θ' contribution to T so warm perturbations
    # raise c_s (a conservative upper bound; see
    # :func:`column_sound_speed_upper_bound` — the single sound-speed home).
    c_sound = column_sound_speed_upper_bound(height_coord, state.theta_prime.data)
    delta_min = jnp.minimum(
        jnp.minimum(grid.dx, grid.dy), dz_min,
    )
    # Substep correction: split-explicit advances acoustic modes
    # over dt/n_substeps per substep.
    dt_acoustic = dt / float(n_acoustic_substeps)
    c_a = c_sound * dt_acoustic / delta_min
    return CourantNumbers(
        horizontal_advective=c_h,
        vertical_advective=c_v,
        acoustic=c_a,
    )


def assert_courant_below(
    courant: CourantNumbers,
    horizontal_max: float = 1.0,
    vertical_max: float = 1.0,
    acoustic_max: float = 1.0,
) -> None:
    """Host-side bound check on the Courant numbers.

    Raises ``ValueError`` if any component exceeds its bound.
    Convert the JAX arrays to Python floats so the check is
    deterministic and not silently traced under JIT.

    Parameters
    ----------
    courant : CourantNumbers
    horizontal_max, vertical_max, acoustic_max : float
        Maximum allowed Courant number per category. CONSERVATIVE
        DEFAULTS at 1.0 — the classical CFL limit for explicit Euler
        on a Cartesian uniform grid. Codex iter-1: the actual stable
        limit depends on the time integrator (SSP-RK3 ~1.5), the
        acoustic substep ratio (already folded into the diagnostic's
        ``n_acoustic_substeps`` if passed), and safety margin
        convention. Callers SHOULD override these defaults using the
        bounds appropriate to their dycore + integrator combination
        (e.g. ``horizontal_max=1.5`` for SSP-RK3, ``acoustic_max=
        0.5`` if you want a safety margin below the formal bound).
    """
    c_h = float(courant.horizontal_advective)
    c_v = float(courant.vertical_advective)
    c_a = float(courant.acoustic)
    if c_h > horizontal_max:
        raise ValueError(
            f"Horizontal advective Courant {c_h:.3e} > limit "
            f"{horizontal_max} (decrease dt or refine the horizontal "
            f"grid)."
        )
    if c_v > vertical_max:
        raise ValueError(
            f"Vertical advective Courant {c_v:.3e} > limit "
            f"{vertical_max} (decrease dt or refine the vertical "
            f"grid)."
        )
    if c_a > acoustic_max:
        raise ValueError(
            f"Acoustic Courant {c_a:.3e} > limit {acoustic_max} "
            f"(decrease dt or increase n_acoustic_substeps)."
        )


def suggest_stable_dt(
    state, height_coord, grid,
    courant_target: float = 0.8,
    n_acoustic_substeps: int = 1,
) -> float:
    """Suggest the largest OUTER dt that keeps every Courant below
    ``courant_target``.

    HOST-SIDE ONLY (Codex iter-1): the function calls ``float(...)``
    on every JAX maximum, which would raise ``TracerArrayConversionError``
    under JIT. Use this from an outer integration loop, NOT from
    inside a compiled hot loop. A JAX-traceable variant is left as
    a follow-up.

    Returns ``min(dt_h, dt_v, dt_acoustic)`` with the acoustic
    bound scaled by ``n_acoustic_substeps`` (split-explicit
    substepping divides the outer dt before applying the acoustic
    CFL).
    """
    if n_acoustic_substeps < 1:
        raise ValueError(
            f"n_acoustic_substeps={n_acoustic_substeps} must be >= 1."
        )
    max_abs_u = float(jnp.max(jnp.abs(state.u.data)))
    max_abs_v = float(jnp.max(jnp.abs(state.v.data)))
    max_abs_w = float(jnp.max(jnp.abs(state.w.data)))
    dz_min = float(jnp.min(height_coord.dz))
    eps = 1.0e-30
    dt_h = courant_target / max(
        max_abs_u / grid.dx, max_abs_v / grid.dy, eps,
    )
    dt_v = courant_target * dz_min / max(max_abs_w, eps)
    # Sound speed from the total state (incl. θ' so warm perturbations raise c_s) —
    # the SAME conservative bound the Courant diagnostic uses (no re-derivation).
    c_sound = float(column_sound_speed_upper_bound(
        height_coord, state.theta_prime.data))
    delta_min = min(grid.dx, grid.dy, dz_min)
    # Acoustic substep budget: outer dt can be n_substeps × the
    # bare acoustic limit before the substep CFL bites.
    dt_a = (
        courant_target * delta_min / c_sound
    ) * float(n_acoustic_substeps)
    return float(min(dt_h, dt_v, dt_a))
