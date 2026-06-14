"""Generic single-column time-stepping kernels shared by the atmosphere and
ocean single-column models (SCMs).

Both SCMs integrate a column state forward in time by

  1. evaluating a tendency closure ``f(state, aux, t) -> (tend, aux_out)``
     (column physics + external forcing), then
  2. advancing ``state`` with an explicit integrator (forward Euler, RK2,
     RK4, or the AB2 first-step fallback).

The integrator *numerics* — the RK stage abscissae and weights, the AB2
multistep weights, and the integrator registry / arity validation — are
identical whether ``state`` is a
:class:`~legoesm.core.state.HydrostaticState` (atmosphere) or an
:class:`~legoesm.ocean.state.OceanState` (ocean).  They are factored here
so neither SCM re-derives them (CLAUDE.md: "No duplicate numerics across
dycores/physics/grids/tests").

The only state-specific operations are

  * ``apply_tendencies(state, tend, dt) -> state``     (``state + dt·tend``)
  * ``average_tendencies(*tends, weights) -> tend``    (weighted sum)

which each SCM supplies.  :func:`build_explicit_integrators` closes the
integrator bodies over those two callables and returns the registry dict;
:func:`register_integrator` validates and installs custom integrators into
such a registry; :func:`ab2_effective_tendency` encodes the AB2 multistep
weights for the driver's per-instance previous-tendency cache.

Integrator contract
--------------------
Every step function has signature ``step_fn(state, aux, f, dt, t)`` and
returns ``(new_state, new_aux)`` where

  * ``aux`` is an opaque per-step carry (the atmosphere's ``PhysicsState``;
    ``None`` for the stateless ocean column physics),
  * ``f(state, aux, t) -> (tend, aux_out)`` is the tendency closure, and
  * ``t`` is the **stage time** in seconds.  Multi-stage integrators
    advance ``t`` to the correct sub-step abscissa so time-dependent
    forcing is sampled consistently with the update.

``aux`` is advanced once per outer step (from the stage-1 evaluation) and
reused across RK stages, so multi-stage integrators are exact only for
autonomous, stateless physics.  Each SCM enforces that constraint with its
own ``forward_euler``-only validation when stateful / implicit physics is
active.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Callable


def build_explicit_integrators(
    apply_tendencies: Callable,
    average_tendencies: Callable,
) -> dict[str, Callable]:
    """Build the default integrator registry for a given state type.

    The returned step functions close over the state-specific
    ``apply_tendencies`` / ``average_tendencies`` operators so the
    RK / Euler numerics live here exactly once.

    Parameters
    ----------
    apply_tendencies
        ``apply_tendencies(state, tend, dt) -> state`` computing
        ``state + dt·tend`` for the SCM's state / tendency types.
    average_tendencies
        ``average_tendencies(*tends, weights=...) -> tend`` computing a
        weighted linear combination of like-typed tendencies.

    Returns
    -------
    dict[str, Callable]
        Keys ``"forward_euler"``, ``"rk2"``, ``"rk4"``, ``"ab2"``.  The
        ``"ab2"`` entry is the first-step forward-Euler fallback; the true
        AB2 multistep update needs a per-instance previous-tendency cache
        and lives in the SCM driver (see :func:`ab2_effective_tendency`).
    """

    def euler_step(state, aux, f, dt, t):
        """Forward Euler (1st order)."""
        tend, aux_out = f(state, aux, t)
        return apply_tendencies(state, tend, dt), aux_out

    def rk2_step(state, aux, f, dt, t):
        """Heun's method (RK2, midpoint-correction variant).

        ``aux`` is advanced once per outer step (stage-1 update) to keep
        the prognostic-physics carry single-valued.  Stage time advances
        to ``t + dt`` for the corrector so time-dependent forcing is
        sampled consistently with the Heun update.
        """
        tend1, aux_mid = f(state, aux, t)
        mid = apply_tendencies(state, tend1, dt)
        tend2, _ = f(mid, aux_mid, t + dt)
        avg = average_tendencies(tend1, tend2, weights=(0.5, 0.5))
        return apply_tendencies(state, avg, dt), aux_mid

    def rk4_step(state, aux, f, dt, t):
        """Classical RK4.  ``aux`` carry advanced from stage 1 only.

        Stage times ``t``, ``t + dt/2``, ``t + dt/2``, ``t + dt`` — the
        canonical RK4 abscissae, so time-dependent forcing is sampled at
        the correct sub-step instants.
        """
        k1, aux_mid = f(state, aux, t)
        s2 = apply_tendencies(state, k1, 0.5 * dt)
        k2, _ = f(s2, aux_mid, t + 0.5 * dt)
        s3 = apply_tendencies(state, k2, 0.5 * dt)
        k3, _ = f(s3, aux_mid, t + 0.5 * dt)
        s4 = apply_tendencies(state, k3, dt)
        k4, _ = f(s4, aux_mid, t + dt)
        avg = average_tendencies(
            k1, k2, k3, k4, weights=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
        )
        return apply_tendencies(state, avg, dt), aux_mid

    def ab2_step_first(state, aux, f, dt, t):
        """First-step fallback for AB2 (no previous tendency): Euler.

        The actual AB2 update lives in the SCM driver because it needs a
        per-instance previous-tendency cache that the
        ``(state, aux, f, dt, t) -> (state', aux')`` contract does not
        expose.  This registry entry exists so ``"ab2"`` passes the
        driver's ``time_integrator`` validation; it only fires for the
        very first step (when ``prev_tend`` is ``None``).
        """
        return euler_step(state, aux, f, dt, t)

    return {
        "forward_euler": euler_step,
        "rk2": rk2_step,
        "rk4": rk4_step,
        "ab2": ab2_step_first,
    }


def ab2_effective_tendency(tend_n, prev_tend, average_tendencies):
    """Adams-Bashforth 2 effective tendency ``1.5·tend_n − 0.5·tend_{n-1}``.

    The second-order linear-multistep combination of the current and
    previous outer-step tendencies.  The SCM driver caches ``prev_tend``
    per instance and applies ``apply_tendencies(state, this, dt)``.
    """
    return average_tendencies(tend_n, prev_tend, weights=(1.5, -0.5))


def register_integrator(
    registry: dict[str, Callable],
    name: str,
    step_fn: Callable,
) -> None:
    """Validate ``step_fn`` and install it into ``registry`` under ``name``.

    The canonical signature is ``step_fn(state, aux, f, dt, t)`` where
    ``f`` is the tendency closure ``f(state, aux, stage_t_seconds)``.
    ``t`` is the base-step time in seconds; multi-stage integrators MUST
    advance ``t`` to the correct sub-step abscissa for each ``f`` call so
    time-dependent SCM forcing is sampled consistently with the update.
    The integrator must return ``(new_state, new_aux)``.

    Backwards compatibility
    -----------------------
    Pre-forcing legacy integrators registered with the older
    ``step_fn(state, aux, f, dt)`` 4-arg signature are wrapped
    transparently: the wrapper captures the outer-step ``t`` once and
    forwards a 2-arg tendency closure that ignores stage time.  Such
    integrators therefore behave correctly only when forcing is ``None``
    or every channel is time-independent; a ``DeprecationWarning`` flags
    this at registration, and the wrapped callable is tagged
    ``_is_legacy_4arg = True`` so the SCM driver can hard-error if it is
    paired with time-dependent forcing.  Anything other than 4 or 5
    positional parameters is rejected with ``TypeError`` at registration
    rather than at first ``step()`` call.
    """
    if not callable(step_fn):
        raise TypeError(
            f"register_integrator({name!r}): step_fn must be callable; "
            f"got {type(step_fn).__name__}."
        )
    try:
        sig = inspect.signature(step_fn)
    except (TypeError, ValueError) as exc:
        # Fail closed: C-extensions, descriptors, and some wrappers
        # refuse signature inspection.  Refusing to register here forces
        # callers to expose a true ``__signature__`` rather than letting
        # an opaque object slip through and crash on the first step.
        raise TypeError(
            f"register_integrator({name!r}): cannot inspect step_fn "
            f"signature ({exc}). Wrap the callable so it exposes a "
            "concrete signature (e.g. ``functools.wraps`` with "
            "``__wrapped__`` set, or a plain ``def`` form)."
        ) from exc

    # Count parameters that can be passed positionally.  The driver
    # invokes ``step_fn(state, aux, f, dt, t)`` positionally, so only the
    # positional-or-positional-or-keyword count must equal 4 or 5.
    # ``*args`` is rejected below (arity unverifiable); required
    # keyword-only params are rejected below (cannot be supplied
    # positionally).  A trailing ``**kwargs`` is harmless — it is never
    # populated by the positional call — so it is intentionally allowed.
    _POSITIONAL = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    positional_params = [
        p for p in sig.parameters.values() if p.kind in _POSITIONAL
    ]
    n_params = len(positional_params)
    has_varargs = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL
        for p in sig.parameters.values()
    )
    if has_varargs:
        raise TypeError(
            f"register_integrator({name!r}): step_fn must declare its "
            "parameters explicitly; ``*args`` / variadic positional "
            "signatures are rejected because arity cannot be verified."
        )

    # Reject any required keyword-only parameters — the driver invokes
    # the integrator positionally so a required kwonly arg (commonly
    # ``def step(state, aux, f, dt, *, t)``) would crash on the first
    # call.  Optional kwonlys (with a default) are harmless.
    required_kwonly = [
        p.name for p in sig.parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
    ]
    if required_kwonly:
        raise TypeError(
            f"register_integrator({name!r}): step_fn declares required "
            f"keyword-only parameter(s) {required_kwonly}; the driver "
            "invokes integrators positionally so keyword-only parameters "
            "cannot be supplied at call time. Move them to "
            "positional-or-keyword."
        )

    if n_params == 5:
        registry[name] = step_fn
    elif n_params == 4:
        warnings.warn(
            f"Time integrator {name!r} uses the legacy 4-arg signature "
            "``(state, aux, f, dt)``. Wrapped for compatibility, but "
            "time-dependent SCM forcing will be sampled at the outer-step "
            "time only — migrate to the 5-arg signature "
            "``(state, aux, f, dt, t)`` to sample forcing at sub-step "
            "abscissae.",
            DeprecationWarning,
            stacklevel=2,
        )

        def wrapped(state, aux, f, dt, t, _legacy=step_fn):
            return _legacy(state, aux, lambda s, a: f(s, a, t), dt)

        # Tag as a legacy shim so the SCM driver can hard-error if the
        # user pairs it with time-dependent forcing — the wrapper would
        # silently sample forcing at the outer-step ``t`` only.
        wrapped._is_legacy_4arg = True
        registry[name] = wrapped
    else:
        raise TypeError(
            f"register_integrator({name!r}): step_fn must take 4 or 5 "
            f"positionally-callable args (got {n_params} positional, full "
            f"signature {list(sig.parameters)})."
        )
