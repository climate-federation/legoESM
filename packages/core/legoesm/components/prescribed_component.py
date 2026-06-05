"""A prescribed-data ``AbstractComponent`` brick (Stage B).

The :class:`~legoesm.components.protocol.AbstractComponent` hard rule is that every
component must run standalone on a single-column grid "with partners supplied by
``Prescribed*`` bricks".  :class:`PrescribedComponent` is that partner: a brick
that **evolves nothing** and instead **provides a fixed (or time-varying) payload**
to whatever consumes it — a prescribed-SST slab feeding the atmosphere, or a
prescribed physics tendency driving a forcing-coupled dycore
(:class:`~legoesm.components.dycore_component.ForcedDycoreComponent`).

It is the *producer* half of the component seam (the dycore wrappers are the
*consumer* half): ``prognostic_variables`` is empty (no state advances),
``required_forcing`` is empty (it needs no partner), and its value is
``provided_fluxes`` — the names of what it hands over.  The payload is a **mapping
keyed by ``provided_fluxes``**, so :attr:`provided_fluxes` is load-bearing: a
consumer routes by name (``producer.provide(grid, state)[flux_name]``), and that
name is exactly what the consumer lists in its ``required_forcing``.

**Differentiability (D1).**  :meth:`provide` is a pure passthrough, so a gradient
flows back through the payload values *when they are traced* — i.e. the component
is constructed inside the differentiated scope, or ``fluxes`` is a pure closure
over the traced inputs.  ``PrescribedComponent`` is NOT an Equinox/pytree module:
its payload is a plain Python attribute, so a component built OUTSIDE and captured
by ``jit``/``scan`` holds the payload as a *static constant* (not a trainable leaf).
Pass the payload through as a traced value (or via a closure) to differentiate it.
Treat the payload as **immutable** — :meth:`provide` returns it without copying.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from legoesm.components.protocol import AbstractComponent


class PrescribedComponent(AbstractComponent):
    """An inert ``AbstractComponent`` that supplies a prescribed payload to partners.

    Parameters
    ----------
    fluxes
        The prescribed payload — a ``Mapping`` ``{flux_name: value}`` keyed EXACTLY
        by *provided_fluxes*, or a pure ``callable(grid, state) -> such a mapping``
        (e.g. a time/space-varying SST closure).
    provided_fluxes
        Names of what this brick provides (its :attr:`provided_fluxes`); a partner
        lists these in its ``required_forcing``.

    A static (non-callable) payload is validated at construction — its keys must
    equal *provided_fluxes* — so a misdeclared producer fails loudly rather than
    handing a partner a flux it never named.  A callable payload must return the
    same key set (validated lazily on the first :meth:`provide`).
    """

    def __init__(
        self,
        fluxes: Mapping[str, Any] | Callable[[Any, Any], Mapping[str, Any]],
        *,
        provided_fluxes: tuple[str, ...],
    ) -> None:
        self._provided_fluxes = tuple(provided_fluxes)
        self._fluxes = fluxes
        if not callable(fluxes):
            self._check_keys(fluxes)

    def _check_keys(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError(
                "PrescribedComponent payload must be a Mapping "
                "{flux_name: value} keyed by provided_fluxes (or a callable "
                f"returning one); got {type(payload).__name__}"
            )
        keys = set(payload.keys())
        if keys != set(self._provided_fluxes):
            raise ValueError(
                f"PrescribedComponent payload keys {sorted(keys)} must equal "
                f"provided_fluxes {sorted(self._provided_fluxes)}"
            )

    @property
    def prognostic_variables(self) -> tuple[str, ...]:
        return ()  # a prescribed brick evolves nothing

    @property
    def required_forcing(self) -> tuple[str, ...]:
        return ()  # it needs no partner forcing

    @property
    def provided_fluxes(self) -> tuple[str, ...]:
        return self._provided_fluxes

    def provide(self, grid: Any = None, state: Any = None) -> Mapping[str, Any]:
        """Return the prescribed ``{flux_name: value}`` mapping (the producer handoff).

        A partner routes by name — ``provide(grid, state)[flux_name]`` — and feeds
        that into its consuming ``tendency``.  Pure, so a gradient flows back
        through the payload values (D1).  Treat the result as immutable.
        """
        payload = self._fluxes(grid, state) if callable(self._fluxes) else self._fluxes
        if callable(self._fluxes):
            self._check_keys(payload)  # a callable's output must match the names too
        return payload

    def tendency(self, grid: Any, state: Any, forcing: Any, params: Any) -> Any:
        """A no-op: a prescribed brick advances no state, so ``dstate`` is the empty
        pytree ``()`` (matching ``prognostic_variables == ()``).

        An inert producer takes no ``forcing``/``params`` — a non-``None`` payload is
        rejected rather than silently dropped (its output is read via
        :meth:`provide`, not produced through this seam).
        """
        if forcing is not None or params is not None:
            raise ValueError(
                "PrescribedComponent is an inert producer (it evolves nothing and "
                "consumes no forcing/params); got a non-None forcing/params.  Read "
                "its prescribed output via provide()."
            )
        return ()
