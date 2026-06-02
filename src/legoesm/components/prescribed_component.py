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
``provided_fluxes`` — the names of what it hands over, returned by :meth:`provide`.
Because :meth:`provide` simply returns a pytree (or calls a pure closure), a
gradient flows from the consumer's loss back through the prescribed payload (D1),
so an end-to-end ``producer -> consumer`` assembly stays differentiable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from legoesm.components.protocol import AbstractComponent


class PrescribedComponent(AbstractComponent):
    """An inert ``AbstractComponent`` that supplies a prescribed payload to partners.

    Parameters
    ----------
    fluxes
        The prescribed payload handed to partners — either a pytree (static) or a
        pure ``callable(grid, state) -> pytree`` (e.g. time/space-varying SST).
    provided_fluxes
        Names of what this brick provides (its :attr:`provided_fluxes`).  These are
        what a partner lists in its ``required_forcing``.
    """

    def __init__(
        self,
        fluxes: Any | Callable[[Any, Any], Any],
        *,
        provided_fluxes: tuple[str, ...],
    ) -> None:
        self._fluxes = fluxes
        self._provided_fluxes = tuple(provided_fluxes)

    @property
    def prognostic_variables(self) -> tuple[str, ...]:
        return ()  # a prescribed brick evolves nothing

    @property
    def required_forcing(self) -> tuple[str, ...]:
        return ()  # it needs no partner forcing

    @property
    def provided_fluxes(self) -> tuple[str, ...]:
        return self._provided_fluxes

    def provide(self, grid: Any = None, state: Any = None) -> Any:
        """Return the prescribed payload (calls the closure if ``fluxes`` is one).

        This is the producer handoff a partner consumes — e.g. feed the result as
        the ``forcing`` argument of a consumer component's ``tendency``.  Pure, so
        a gradient flows back through it (D1).
        """
        if callable(self._fluxes):
            return self._fluxes(grid, state)
        return self._fluxes

    def tendency(self, grid: Any, state: Any, forcing: Any, params: Any) -> Any:
        """A no-op: a prescribed brick advances no state, so ``dstate`` is empty.

        ``forcing``/``params`` are ignored (a producer is inert to them — nothing
        that should be used is silently dropped, because it evolves nothing).
        """
        return ()
