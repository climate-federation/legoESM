"""Wrap a live surface model's step as an ``AbstractComponent`` brick (Stage B).

The atmosphere/ocean *dynamical cores* expose an explicit right-hand side
(``tendencies(state)``) and migrate onto :class:`AbstractComponent` through the
:mod:`~legoesm.components.dycore_component` wrappers.  The *surface tiles*
(slab / multilayer land, slab / dynamic sea ice, slab ocean) are different: they
advance by an **implicit step** — the per-column surface energy balance is solved
(D3) — returning ``(new_state, TileResponse)`` rather than an explicit continuous
tendency.  :class:`SurfaceComponentAdapter` migrates those live surface steps onto
the component seam WITHOUT rewriting the physics: it wraps the existing step
(same numerics, same ``jax.grad`` path), declares the brick metadata, and exposes
a uniform :meth:`step`.

Because a surface update is an implicit *step*, the explicit-RHS
:meth:`tendency` seam (used by dycores) is not applicable and raises with a
pointer to :meth:`step` — the coupler advances a surface brick by stepping it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from legoesm.components.protocol import AbstractComponent


class SurfaceComponentAdapter(AbstractComponent):
    """An :class:`AbstractComponent` over a live surface model's implicit step.

    Parameters
    ----------
    step_fn
        The surface step as a pure ``callable(grid, state, forcing) -> (new_state,
        response)`` with its static config / ``dt`` bound by the caller — mirroring
        the coupler's per-tile closures (``_step_land_tile`` etc.).  Must be
        ``jax.grad``-compatible (D1).
    prognostic_variables
        Names of the surface state fields this tile evolves.
    required_forcing
        Names of the forcing it needs from the atmosphere (e.g. ``"atm_forcing"``
        for the :class:`AtmToSurface` bundle, plus partner inputs like
        ``"ocean_state"`` for sea ice).
    provided_fluxes
        Names of the :class:`~legoesm.core.coupling_fields.TileResponse` channels it
        hands back to the atmosphere (``shflx``, ``lhflx``, ``lw_up``, ...).
    """

    def __init__(
        self,
        step_fn: Callable[[Any, Any, Any], Any],
        *,
        prognostic_variables: tuple[str, ...],
        required_forcing: tuple[str, ...],
        provided_fluxes: tuple[str, ...],
    ) -> None:
        if not callable(step_fn):
            raise TypeError(
                f"SurfaceComponentAdapter needs a callable step_fn(grid, state, "
                f"forcing) -> (new_state, response); got {step_fn!r}"
            )
        self._step_fn = step_fn
        self._prognostic_variables = tuple(prognostic_variables)
        self._required_forcing = tuple(required_forcing)
        self._provided_fluxes = tuple(provided_fluxes)

    @property
    def prognostic_variables(self) -> tuple[str, ...]:
        return self._prognostic_variables

    @property
    def required_forcing(self) -> tuple[str, ...]:
        return self._required_forcing

    @property
    def provided_fluxes(self) -> tuple[str, ...]:
        return self._provided_fluxes

    def step(self, grid: Any, state: Any, forcing: Any) -> Any:
        """Advance the surface state one implicit step.

        Returns ``(new_state, response)`` from the wrapped surface model — pure, so
        ``jax.grad`` flows through the step (D1); the ``response`` is the
        :class:`~legoesm.core.coupling_fields.TileResponse` of fluxes back to the
        atmosphere.
        """
        return self._step_fn(grid, state, forcing)

    def tendency(self, grid: Any, state: Any, forcing: Any, params: Any) -> Any:
        """Not applicable — a surface tile advances by an implicit :meth:`step`.

        The surface energy balance is solved implicitly (D3), so the component has no
        explicit continuous tendency; advance it via ``step(grid, state, forcing)``.
        ``AbstractComponent.tendency`` is the explicit-RHS seam used by dycores.
        """
        raise NotImplementedError(
            "SurfaceComponentAdapter wraps an implicitly-STEPPED surface model "
            "(surface energy balance, D3); advance it via step(grid, state, "
            "forcing), not tendency().  The tendency seam is for explicit-RHS "
            "components such as dynamical cores."
        )
