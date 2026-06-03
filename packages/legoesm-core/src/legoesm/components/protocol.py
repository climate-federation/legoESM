"""Component contracts (Stage A2 / design Phase 0).

These Protocols and abstract bases are the structural seams the Stage-B refactor
builds on.  They describe what a dynamical core / surface component / physics
scheme must expose so the registry can *discover* them and the coupler can
*compose* them — and they pin the cross-cutting differentiability invariants
every implementation must uphold:

* **D1 — pure pytree functions.**  State is a registered pytree; ``step`` /
  ``tendency`` are pure ``state -> state`` (no host side effects, no Python
  control flow on *traced* values), so ``jax.grad`` / ``lax.scan`` trace through
  them regardless of which package defines them.
* **D2 — donation × grad.**  A JIT ``step`` used inside ``jax.grad`` /
  ``eqx.filter_value_and_grad`` must also expose a non-donating ``.raw`` variant
  (``donate_argnums`` conflicts with reverse-mode AD).  See ``build_segment_fn``.
* **D3 — implicit solves differentiated properly.**  Surface energy balance and
  any implicit coupling use the implicit-function theorem
  (``optimistix`` / ``lax.custom_root``), never backprop through an unbounded
  iteration; no hard regime branches in flux/closure code — smooth blends keep
  VJPs continuous.

These are *contracts*, not implementations: existing solver classes already
satisfy :class:`DycoreProtocol` structurally (``__init__(grid, config)`` then
``step(state, dt)``); Stage B migrates the components onto
:class:`AbstractComponent`.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

# The physics-scheme contract already lives in core.state — re-export it so the
# component seams are all discoverable from one place (do NOT duplicate it).
from legoesm.core.state import PhysicsModuleProtocol  # noqa: F401


@runtime_checkable
class DycoreProtocol(Protocol):
    """A dynamical core: constructed on a grid, advances a state pytree by ``dt``.

    Concrete dycores (``CDGridShallowWaterModel``,
    ``SpectralPrimitiveEquationModel``, ``MPASCompressibleEulerModel``, ...) are
    built as ``Model(grid, config)`` and then expose ``step(state, dt) -> state``
    where ``state`` is a registered pytree (D1).  A differentiable training
    consumer additionally requires a non-donating ``.raw`` step variant (D2).

    ``isinstance(model, DycoreProtocol)`` is a NECESSARY structural check, but a
    ``runtime_checkable`` Protocol only verifies attribute *presence* — it would
    pass ``step = 1`` or a zero-arg ``step``.  The registry must therefore call
    :func:`validate_dycore`, which also checks ``step`` is callable with the
    ``(state, dt)`` arity.
    """

    def step(self, state: Any, dt: float) -> Any:
        """Advance one time step: pure ``state -> state`` (D1)."""
        ...


def validate_dycore(obj: Any) -> None:
    """Raise ``TypeError`` unless *obj* is a usable dycore.

    Stronger than ``isinstance(obj, DycoreProtocol)``: confirms ``step`` is
    *callable* and accepts the ``(state, dt)`` arity, so a malformed plugin fails
    at registration rather than deep inside a rollout.  The registry uses this,
    not the bare isinstance check.
    """
    step = getattr(obj, "step", None)
    if not callable(step):
        raise TypeError(
            f"dycore {obj!r} does not expose a callable 'step' method"
        )
    try:
        sig = inspect.signature(step)
    except (TypeError, ValueError) as exc:
        # An uninspectable callable cannot be proven to accept (state, dt), so we
        # do NOT silently pass it — that would reinstate the late-failure mode.
        # A native/C step should expose ``__signature__`` to be accepted.
        raise TypeError(
            f"dycore.step signature could not be inspected ({exc}); a dycore "
            f"step must be an introspectable callable (set __signature__ on a "
            f"native implementation)"
        ) from exc
    # Prove ``step(state, dt)`` is actually a valid call: bind() raises if a
    # required argument is missing (too few params, or an extra REQUIRED
    # positional / keyword-only one) or if there are too many.  This catches the
    # late-failure cases a mere positional-count check would let through.
    _sentinel = object()
    try:
        sig.bind(_sentinel, _sentinel)
    except TypeError as exc:
        raise TypeError(
            f"dycore.step cannot be called as step(state, dt) "
            f"(signature {sig}): {exc}"
        ) from exc


@runtime_checkable
class SurfaceComponentProtocol(Protocol):
    """A surface tile (land / slab-ocean / sea-ice) that exchanges fluxes.

    Computes a pure tendency from the grid, its own state, the forcing handed in
    by the coupler interface, and its parameters.
    """

    def tendency(self, grid: Any, state: Any, forcing: Any, params: Any) -> Any:
        """Pure ``dstate = f(grid, state, forcing, params)`` (D1)."""
        ...


class _ComponentBase(ABC):
    """Shared metadata of an interchangeable Earth-system 'lego' brick (design L4).

    A component declares what it needs (``required_forcing``), what it evolves
    (``prognostic_variables``), and what it hands back to its partners
    (``provided_fluxes``).  Two update flavours subclass this base — explicit-RHS
    :class:`AbstractComponent` (computes a ``tendency``) and implicitly-stepped
    :class:`StepComponent` (advances via ``step``) — so a brick is never forced to
    fake the seam it does not have.
    """

    @property
    @abstractmethod
    def prognostic_variables(self) -> tuple[str, ...]:
        """Names of the state fields this component evolves."""
        ...

    @property
    @abstractmethod
    def required_forcing(self) -> tuple[str, ...]:
        """Names of the forcing fields this component needs from its partners."""
        ...

    @property
    @abstractmethod
    def provided_fluxes(self) -> tuple[str, ...]:
        """Names of the fluxes this component provides back to its partners."""
        ...


class AbstractComponent(_ComponentBase):
    """An explicit-RHS Earth-system 'lego' brick (design L4).

    Computes a pure ``tendency`` (``dstate = f(...)``) — the dynamical cores
    (atmosphere / 3-D ocean).  **Hard rule:** every component must run standalone on
    a single-column grid — its cheapest gradient-check harness — with partners
    supplied by ``Prescribed*`` bricks.  Implicitly-stepped components (surface
    tiles whose energy balance is solved per column) use :class:`StepComponent`
    instead, so they never fake a ``tendency``.
    """

    @abstractmethod
    def tendency(self, grid: Any, state: Any, forcing: Any, params: Any) -> Any:
        """Pure ``dstate = f(grid, state, forcing, params)`` (D1)."""
        ...


class StepComponent(_ComponentBase):
    """An implicitly-stepped Earth-system 'lego' brick (design L4).

    The surface tiles (slab / multilayer land, slab / dynamic sea ice, slab ocean)
    advance by an **implicit step** — the per-column surface energy balance is solved
    (D3) — returning ``(new_state, response)``, not an explicit continuous tendency.
    A ``StepComponent`` exposes that as ``step(grid, state, forcing, dt)`` and carries
    the SAME metadata as :class:`AbstractComponent`, so the two are interchangeable
    bricks that differ only in their update seam (no faked ``tendency``).
    """

    @abstractmethod
    def step(self, grid: Any, state: Any, forcing: Any, dt: Any) -> Any:
        """Advance one step: ``(new_state, response) = step(grid, state, forcing, dt)``.

        Pure (D1); the per-column surface energy balance is solved implicitly (D3).
        """
        ...
