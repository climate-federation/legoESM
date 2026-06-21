"""Leapfrog + Adams-Bashforth + Robert-Asselin time integrator.

Veros's canonical time integrator. Provides a leapfrog step with
Robert-Asselin time filter to damp the leapfrog computational mode,
plus an Adams-Bashforth-2 corrector path used by many MOM-derived
ocean models for advection terms.

Schemes
-------

**Leapfrog (centred-in-time)**::

    state^{n+1} = state^{n-1} + 2 dt · F(state^n)

Unconditionally stable for linear hyperbolic problems (no CFL on the
linear-advection part) and conserves quadratic invariants exactly in
the absence of forcing. The downside is the **computational mode** —
a 2-Δt oscillation between even and odd steps — which grows without
damping. Mitigated by the Robert-Asselin filter below.

**Robert-Asselin filter** (Asselin 1972)::

    state^n_filtered = state^n + ν · (state^{n+1} - 2 state^n + state^{n-1})

Applied AFTER the leapfrog step, before the carry advance. ``ν ≈ 0.01``
to ``0.1`` is the common range; ``ν = 0.05`` is a Veros-typical default.
The filter is linear in the three state levels, so its forward and
reverse-mode AD are well-defined.

**Adams-Bashforth-2 (AB2)** with optional ε offset::

    state^{n+1} = state^n + dt · ((1.5 + ε) · F(state^n)
                                  - (0.5 + ε) · F(state^{n-1}))

Used by MOM4 / MOM5 / Veros for the advection corrector inside the
leapfrog step. The optional ``ε`` parameter (typically 0.1) damps
the leapfrog computational mode somewhat without the Robert-Asselin
filter's amplitude loss; setting ``ε = 0`` recovers the standard
AB2 scheme.

Differentiability
-----------------

All operations in this module are linear combinations of pytree
leaves and therefore differentiable end-to-end. The Robert-Asselin
filter's three-level carry is the only non-standard feature; the
``LeapfrogCarry`` NamedTuple is a flat pytree (registered via
``register_pytree_node`` in JAX) so it works with
:func:`jax.lax.scan`, :func:`jax.grad`, and :func:`jax.vmap`.

Integration into ``LatLonCGridOceanModel`` and the ``timestepping/
split_explicit.py`` ``outer_integrator`` dispatch is tracked under
Phase G.2 in ``docs/ocean/fidelity/phase_g_veros_recipe_audit.md`` —
this module provides the standalone closure ready for that wiring.

References
----------
- Robert, A. J. (1969). The integration of a low-order spectral form
  of the primitive meteorological equations. *J. Meteorol. Soc. Japan*,
  44, 237-244.
- Asselin, R. (1972). Frequency Filter for Time Integrations.
  *Mon. Weather Rev.*, 100, 487-490.
- Williams, P. D. (2011). The RAW filter: An improvement to the
  Robert-Asselin filter. *Mon. Weather Rev.*, 139, 1996-2007.
- Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics*,
  2nd ed., Springer, Ch. 2.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, TypeVar

import jax


State = TypeVar("State")


class LeapfrogCarry(NamedTuple):
    """Carry pytree for the leapfrog + Robert-Asselin integrator.

    ``prev`` is ``state^{n-1}`` (used as the leapfrog jump-off point
    next step). ``curr`` is ``state^n`` after Robert-Asselin filtering.
    Both have the same pytree structure as the integrator's input
    state.

    The carry is registered as a JAX pytree automatically because it
    is a NamedTuple of two JAX-compatible state pytrees — no manual
    registration required.
    """
    prev: State
    curr: State


def leapfrog_ra_step(
    carry: LeapfrogCarry,
    tendency_fn: Callable[[State], State],
    dt: float,
    asselin_nu: float = 0.05,
) -> LeapfrogCarry:
    """One leapfrog + Robert-Asselin step.

    Parameters
    ----------
    carry : LeapfrogCarry
        ``(state^{n-1}, state^n)`` from the previous step.
    tendency_fn : callable
        ``F(state) -> tendencies`` matching the state pytree.
    dt : float
        Time step [s].
    asselin_nu : float
        Robert-Asselin filter strength. ``0.05`` is a Veros-typical
        default. Set ``0.0`` to disable filtering (leapfrog only —
        will likely diverge over many steps).

    Returns
    -------
    LeapfrogCarry
        ``(state^n_filtered, state^{n+1})`` ready for the next call.
    """
    prev, curr = carry.prev, carry.curr

    # Leapfrog: state^{n+1} = state^{n-1} + 2 dt · F(state^n)
    tend = tendency_fn(curr)
    new = jax.tree.map(lambda p, t: p + 2.0 * dt * t, prev, tend)

    if asselin_nu == 0.0:
        return LeapfrogCarry(prev=curr, curr=new)

    # Robert-Asselin filter on state^n:
    #   state^n_filtered = state^n + ν · (state^{n+1} - 2 state^n + state^{n-1})
    nu = asselin_nu
    curr_filtered = jax.tree.map(
        lambda c, p, n: (1.0 - 2.0 * nu) * c + nu * p + nu * n,
        curr, prev, new,
    )
    return LeapfrogCarry(prev=curr_filtered, curr=new)


class AB2Carry(NamedTuple):
    """Carry pytree for Adams-Bashforth-2.

    ``prev_tend`` is ``F(state^{n-1})`` — the tendency from the
    previous step, cached to avoid recomputation. ``curr`` is the
    current state ``state^n``.
    """
    prev_tend: State
    curr: State


def ab2_step(
    carry: AB2Carry,
    tendency_fn: Callable[[State], State],
    dt: float,
    epsilon: float = 0.0,
) -> AB2Carry:
    """One Adams-Bashforth-2 step.

    state^{n+1} = state^n + dt · ((1.5 + ε)·F(state^n) - (0.5 + ε)·F(state^{n-1}))

    Parameters
    ----------
    carry : AB2Carry
        Holds ``F(state^{n-1})`` and ``state^n``.
    tendency_fn : callable
    dt : float
    epsilon : float
        Damping offset on the AB2 coefficients (MOM4 / Veros use
        ``ε = 0.1`` in some configurations to suppress the leapfrog
        computational mode without the Robert-Asselin amplitude loss).
        ``0`` recovers the standard scheme.

    Returns
    -------
    AB2Carry
    """
    a_curr = 1.5 + epsilon
    a_prev = -(0.5 + epsilon)
    curr_tend = tendency_fn(carry.curr)
    # state^{n+1} = state^n + dt · (a_curr·curr_tend + a_prev·prev_tend)
    new = jax.tree.map(
        lambda s, tc, tp: s + dt * (a_curr * tc + a_prev * tp),
        carry.curr, curr_tend, carry.prev_tend,
    )
    return AB2Carry(prev_tend=curr_tend, curr=new)


def initialize_leapfrog_carry(state: State) -> LeapfrogCarry:
    """Seed the leapfrog carry from a single initial state.

    Sets both ``prev`` and ``curr`` to the same value — equivalent to
    a forward-Euler bootstrap step. After the first call,
    ``leapfrog_ra_step`` will produce a meaningful ``state^{n+1}``.

    Veros production runs typically take a forward-Euler step for the
    very first iteration (so the leapfrog jump-off has a sensible
    ``state^{n-1}``); this seed mimics that.
    """
    return LeapfrogCarry(prev=state, curr=state)


def initialize_ab2_carry(
    state: State,
    tendency_fn: Callable[[State], State],
) -> AB2Carry:
    """Seed the AB2 carry from a single initial state.

    Sets ``prev_tend`` to ``F(state)`` so the first AB2 step degenerates
    to a forward-Euler step ((1.5 + ε - 0.5 - ε) · F = F).
    """
    return AB2Carry(prev_tend=tendency_fn(state), curr=state)


__all__ = (
    "AB2Carry",
    "LeapfrogCarry",
    "ab2_step",
    "initialize_ab2_carry",
    "initialize_leapfrog_carry",
    "leapfrog_ra_step",
)
