"""The four-brick coupler ``Interface`` contract (Stage A2 / design §5).

An ``Interface`` between two components is a *composition of four orthogonal,
individually grad-safe sub-bricks* — so any combination is valid and
differentiability/conservation are enforced per brick (D3/D4), not prayed for
over the whole system:

* **form** — WHAT is exchanged: ``FluxForm`` (turbulent/radiative fluxes; conserves
  by construction), ``StateForm`` (state across; flagged non-conservative),
  ``MixedForm`` (per-quantity choice).
* **transfer** — WHAT computes it: ``PhysicsTransfer`` (Monin-Obukhov / bulk, with a
  differentiable implicit surface solve — IFT, no hard regime branches: D3),
  ``MLTransfer`` (must opt into a conservation constraint: D4), ``HybridTransfer``.
* **regrid** — HOW grids are bridged: ``Identity`` / ``ConservativeRemap`` (area-
  weighted, conserves: D4) / ``BilinearRemap``.
* **coupling** — HOW it advances in time: ``Explicit`` / ``ImplicitSurface``
  (IFT-differentiated solve) / ``Concurrent``.

This module is the *contract* — the structural seam the registry/coupler compose
against.  Stage B5 supplies the concrete bricks; they satisfy these Protocols.
The signatures follow design §5.1-5.4 and are intentionally minimal (B5 fixes the
exact state/field types).
"""

from __future__ import annotations

import inspect
from typing import Any, NamedTuple, Protocol, runtime_checkable


@runtime_checkable
class FormBrick(Protocol):
    """Maps an exchanged quantity onto each side as boundary conditions.

    ``conservative`` records whether the form conserves by construction
    (``FluxForm`` -> True; ``StateForm`` -> False, flagged per D4).
    """

    conservative: bool

    def apply(self, exchanged: Any, state_a: Any, state_b: Any) -> tuple[Any, Any]:
        """Return ``(update_a, update_b)`` — the BC applied to each side."""
        ...


@runtime_checkable
class TransferBrick(Protocol):
    """Computes the exchanged quantity from both sides' states (the learnable brick).

    Pure and grad-safe (D1).  An implicit surface solve is differentiated via the
    implicit-function theorem, never backprop through an unbounded loop (D3).

    ``conservative`` records whether the transfer preserves the exchanged budget:
    physics/bulk and hybrid transfers do by construction; an ``MLTransfer`` does
    NOT unless it opts into a conservation constraint (D4).  This flag is part of
    :meth:`Interface.conserves` precisely so an unconstrained learned transfer
    cannot be reported as end-to-end conserving.
    """

    conservative: bool

    def __call__(self, state_a: Any, state_b: Any, surface_props: Any) -> Any:
        ...


@runtime_checkable
class RegridBrick(Protocol):
    """A differentiable remap of a field between two component grids.

    ``conservative`` records whether the remap conserves area-weighted integrals
    (``ConservativeRemap`` -> True; ``BilinearRemap``/``Identity`` may not).
    """

    conservative: bool

    def __call__(self, field: Any, src_grid: Any, dst_grid: Any) -> Any:
        ...


@runtime_checkable
class CouplingBrick(Protocol):
    """Advances the exchange in time (explicit / implicit-surface / concurrent)."""

    def advance(
        self, interface: Interface, state_a: Any, state_b: Any, dt: float
    ) -> tuple[Any, Any]:
        """Return the advanced ``(state_a, state_b)`` for one coupling window."""
        ...


class Interface(NamedTuple):
    """One component-pair exchange = form x transfer x regrid x coupling.

    The coupler holds one ``Interface`` per component pair (atmos-ocean,
    atmos-land, atmos-ice, ocean-ice).  Components never compute each other's
    exchange — the interface does, once, and applies it conservatively to both
    sides.
    """

    side_a: str
    side_b: str
    form: FormBrick
    transfer: TransferBrick
    regrid: RegridBrick
    coupling: CouplingBrick

    def conserves(self) -> bool:
        """True iff this interface conserves the exchanged quantity end-to-end.

        Requires a conservative **form** (flux removed from one side == added to
        the other), a conservative **transfer** (a learned transfer must opt in —
        an unconstrained ``MLTransfer`` is NOT conservative), AND a conservative
        **regrid** (area-weighted remap).  Any non-conservative brick breaks the
        chain — the D4 invariant the per-interface conservation test checks.
        """
        return bool(
            self.form.conservative
            and self.transfer.conservative
            and self.regrid.conservative
        )


def _validate_call(fn: Any, n_args: int, label: str) -> None:
    """Raise ``TypeError`` unless *fn* is callable with *n_args* positional args."""
    if not callable(fn):
        raise TypeError(f"{label} is not callable")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{label} signature could not be inspected ({exc}); expose "
            f"__signature__ for a native implementation"
        ) from exc
    try:
        sig.bind(*([object()] * n_args))
    except TypeError as exc:
        raise TypeError(
            f"{label} must accept {n_args} positional argument(s) "
            f"(signature {sig}): {exc}"
        ) from exc


def validate_interface(iface: Interface) -> None:
    """Raise ``TypeError`` unless *iface*'s four bricks satisfy the contract.

    Stronger than ``isinstance(brick, Protocol)`` (which a ``runtime_checkable``
    Protocol passes on mere attribute presence): confirms each brick method is
    callable with its contract arity, and that ``form``/``transfer``/``regrid``
    expose a ``bool`` ``conservative`` flag.  The registry/coupler uses this when
    composing an interface, so a malformed brick fails at composition, not deep in
    a coupled rollout.
    """
    _validate_call(iface.form.apply, 3, "form.apply(exchanged, state_a, state_b)")
    _validate_call(iface.transfer, 3, "transfer(state_a, state_b, surface_props)")
    _validate_call(iface.regrid, 3, "regrid(field, src_grid, dst_grid)")
    _validate_call(iface.coupling.advance, 4, "coupling.advance(interface, a, b, dt)")
    for brick, name in (
        (iface.form, "form"),
        (iface.transfer, "transfer"),
        (iface.regrid, "regrid"),
    ):
        if not isinstance(getattr(brick, "conservative", None), bool):
            raise TypeError(f"{name} brick must expose a bool 'conservative' flag")
