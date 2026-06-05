"""Tracer positivity filters for CRM tracer transport.

Pseudo-spectral horizontal advection, sharp microphysical sources,
and limited-precision upwind reconstruction can all leave tracer
fields with slightly-negative values that microphysical schemes
read as nonsensical (negative q_v injects energy on condensation,
negative q_c skips evaporation). These helpers provide:

* :func:`clip_positive` — pointwise ``max(q, 0)``. Fastest; does
  not conserve the column mean.
* :func:`clip_positive_compensated` — clips, then borrows the
  clipped negative mass from the positive cells weighted by cell
  volume so the global tracer mean is preserved bit-exactly.
* :func:`apply_positive_filter_state` — bulk-apply a chosen filter
  to every tracer slot in a ``PlaneNonHydrostaticState`` or
  spectral counterpart.

Reuses existing tracer dtype / shape patterns; does NOT introduce
a new state type.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def clip_positive(q: jax.Array) -> jax.Array:
    """Pointwise non-negativity clip: ``max(q, 0)``.

    Fastest filter; produces a slight POSITIVE bias on the column
    mean equal to ``-sum(min(q, 0))``. Suitable when the upstream
    advection is intrinsically positive-preserving and only floating-
    point noise needs to be stripped. Use
    :func:`clip_positive_compensated` if the upstream produces real
    negative excursions whose mass must be conserved.
    """
    return jnp.maximum(q, 0.0)


def clip_positive_compensated(
    q: jax.Array, weights: jax.Array | None = None,
) -> jax.Array:
    """Clip + uniform downward redistribution.

    1. Deficit ``D = -sum(min(q, 0) · w) ≥ 0`` (mass removed by the
       pointwise clip).
    2. ``q_pos_new = q_pos · max(1 - D / sum(q_pos · w), 0)``.
    3. Negatives → 0.

    Conservation contract (Codex iter-1)
    ------------------------------------
    The weighted mean ``sum(q · w)`` is preserved **to floating-point
    roundoff (i.e. ``rtol ≲ 1e-12`` under fp64; the implementation
    is purely linear in ``q`` so reduction order alone sets the
    bound) only when** ``sum(q · w) ≥ 0`` AND ``sum(q_pos · w) > 0``.
    In the other regimes, mass is dropped:

    * ``sum(q · w) < 0`` (mostly-negative input): the positive mass
      cannot cover the deficit, so ``scale → 0`` and every cell is
      zeroed. Output weighted mass = 0 ≥ original (mass GAINED to
      preserve non-negativity, not lost). The caller should treat
      this as a hard signal that the upstream advection has
      malfunctioned — positivity filters are not a substitute for
      a positivity-preserving transport scheme.
    * ``sum(q_pos · w) == 0`` (all values ≤ 0): same as above, all
      cells zeroed.

    Validate input via :func:`_validate_compensated_weights`: shape
    must match ``q`` exactly; values must be non-negative.

    Differentiability
    -----------------
    Uses ``jnp.where`` and ``maximum`` at clipping thresholds, so
    gradients are piecewise smooth. ``jax.grad`` returns the
    subgradient at the kinks; values stay finite everywhere.
    """
    if weights is None:
        weights = jnp.ones_like(q)
    else:
        _validate_compensated_weights(q, weights)
    q_pos = jnp.maximum(q, 0.0)
    q_neg = jnp.minimum(q, 0.0)        # all <= 0
    deficit = -jnp.sum(q_neg * weights)    # >= 0 (since w >= 0)
    pos_mass = jnp.sum(q_pos * weights)    # >= 0
    safe_pos_mass = jnp.where(pos_mass > 0.0, pos_mass, 1.0)
    scale = jnp.where(
        pos_mass > 0.0,
        jnp.maximum(1.0 - deficit / safe_pos_mass, 0.0),
        0.0,
    )
    return q_pos * scale


def _validate_compensated_weights(q: jax.Array, weights: jax.Array) -> None:
    """Compensated mode requires weights with matching shape and
    non-negative values.

    Static shape check is always enforced. The non-negativity check
    runs ONLY on host arrays (outside a JIT trace); under JIT,
    negative weights would propagate through the filter and break
    the conservation guarantee silently. Callers using this filter
    inside a JIT-compiled hot loop MUST pre-validate weights on the
    host or use ``jax.experimental.checkify.check`` upstream
    (Codex iter-2). Standard CRM use builds weights as ``rho · dz ·
    area`` once at setup → naturally host-static.
    """
    if weights.shape != q.shape:
        raise ValueError(
            f"clip_positive_compensated: weights.shape "
            f"{weights.shape} must equal q.shape {q.shape} "
            f"(no broadcasting allowed; the weighted mean is then "
            f"unambiguous)."
        )
    # Host-only non-negativity check; under JIT the float() raises
    # TracerArrayConversionError which we trap so the function
    # remains traceable. The trapping is INTENTIONAL — see docstring.
    try:
        w_min = float(jnp.min(weights))
        if w_min < 0.0:
            raise ValueError(
                f"clip_positive_compensated: weights must be "
                f"non-negative; min(weights) = {w_min:.3e}"
            )
    except (jax.errors.TracerArrayConversionError, TypeError):
        # Inside a JIT trace; non-neg invariant must be enforced
        # by the caller (e.g. compute weights as rho·dz·area at
        # host setup, or wrap in jax.experimental.checkify.check).
        pass


def apply_positive_filter_state(
    state, tracer_slots_to_filter=None, mode: str = "clip",
    weights: jax.Array | None = None,
):
    """Apply a positivity filter to selected tracer slots of any
    plane/CS/MPAS NH state.

    Required state protocol (Codex iter-1)
    --------------------------------------
    ``state`` MUST expose a ``tracers`` attribute that is a Field-
    like object with ``.data`` (jax.Array, trailing tracer axis)
    and a ``replace(data=new_data)`` method, AND must support
    ``state._replace(tracers=new_field)`` (NamedTuple convention).
    All NH state classes in the codebase
    (``PlaneNonHydrostaticState``, ``NonHydrostaticState``,
    ``MPASNonHydrostaticState``, ``SpectralPlanePhysicsState``)
    satisfy this. Callers passing raw tracer arrays should use
    :func:`clip_positive` or :func:`clip_positive_compensated`
    directly.

    Parameters
    ----------
    state : Any NH state with a ``tracers`` Field whose ``.data`` has
        a trailing ``(..., n_tracers)`` axis.
    tracer_slots_to_filter : tuple of int or None
        Which slot indices to filter. Default ``None`` filters every
        slot.
    mode : {"clip", "compensated"}
        ``clip`` → pointwise :func:`clip_positive`.
        ``compensated`` → :func:`clip_positive_compensated` (mass-
        preserving). When compensated, ``weights`` is forwarded.
    weights : jax.Array or None
        Per-cell weights, only used with mode="compensated".

    Returns
    -------
    Same state type with filtered tracers; non-tracer fields
    untouched.
    """
    if mode not in ("clip", "compensated"):
        raise ValueError(
            f"Unknown positivity filter mode {mode!r}; "
            f"choose from 'clip', 'compensated'."
        )
    if not (
        hasattr(state, "tracers") and hasattr(state.tracers, "data")
        and hasattr(state.tracers, "replace") and hasattr(state, "_replace")
    ):
        raise TypeError(
            f"apply_positive_filter_state requires a state with a "
            f"`tracers` Field (with .data + .replace) and a "
            f"NamedTuple `_replace`; got {type(state).__name__} "
            f"which lacks the required protocol. Use "
            f"clip_positive / clip_positive_compensated directly "
            f"on raw arrays instead."
        )
    tracers = state.tracers.data       # (..., n_tracers)
    n_tracers = tracers.shape[-1]
    if tracer_slots_to_filter is None:
        tracer_slots_to_filter = tuple(range(n_tracers))

    def _filter_slot(slot):
        if mode == "clip":
            return clip_positive(slot)
        return clip_positive_compensated(slot, weights=weights)

    new_tracers = tracers
    for idx in tracer_slots_to_filter:
        if idx < 0 or idx >= n_tracers:
            raise ValueError(
                f"tracer slot {idx} out of range [0, {n_tracers})."
            )
        slot = new_tracers[..., idx]
        new_tracers = new_tracers.at[..., idx].set(_filter_slot(slot))

    return state._replace(
        tracers=state.tracers.replace(data=new_tracers),
    )
