"""Wrap a live surface model's step as a ``StepComponent`` brick (Stage B).

The atmosphere / 3-D ocean *dynamical cores* expose an explicit right-hand side
(``tendencies(state)``) and migrate onto :class:`AbstractComponent` through the
:mod:`~legoesm.components.dycore_component` wrappers.  The *surface tiles*
(slab / multilayer land, slab / dynamic sea ice, slab ocean) are different: they
advance by an **implicit step** — the per-column surface energy balance is solved
(D3) — returning ``(new_state, response)`` rather than an explicit continuous
tendency.  They therefore migrate onto :class:`StepComponent` (the implicit-step
sibling of ``AbstractComponent``), NOT a faked ``tendency``.

:class:`SurfaceComponentAdapter` migrates those live surface steps onto the
component seam WITHOUT rewriting the physics: it wraps the existing step (same
numerics, same ``jax.grad`` path), declares the brick metadata, and exposes the
uniform ``step(grid, state, forcing, dt)``.

**Static wrapper (not a pytree).**  Like the other component bricks this is a plain
Python wrapper, not an Equinox/pytree module: the wrapped ``step_fn`` (and any
config it closes over) is a static attribute.  ``dt`` is an explicit ``step``
argument — NOT closed over — so a variable timestep flows through the seam without
rebuilding the wrapper or risking a stale captured ``dt``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from legoesm.components.protocol import StepComponent


class SurfaceComponentAdapter(StepComponent):
    """A :class:`StepComponent` over a live surface model's implicit step.

    Parameters
    ----------
    step_fn
        The surface step as a pure ``callable(grid, state, forcing, dt) ->
        (new_state, response)`` with its static config bound by the caller —
        mirroring the coupler's per-tile closures (``_step_land_tile`` etc.).  Must
        be ``jax.grad``-compatible (D1).
    prognostic_variables
        Names of the surface state fields this tile evolves.
    required_forcing
        Names of the forcing it needs (e.g. ``"atm_forcing"`` for the
        :class:`AtmToSurface` bundle, plus partner inputs like ``"ocean_state"`` for
        sea ice).
    provided_fluxes
        Names of what it hands to its partners.  For land/ice these are
        :class:`~legoesm.core.coupling_fields.TileResponse` channels (``shflx``,
        ``lhflx``, ``lw_up``, ...); for the ocean *model* they are the surface fields
        the coupler turns into atmospheric fluxes (``sst``, ``u_ocean_sfc``,
        ``v_ocean_sfc``).  Validated against the ``response`` on each :meth:`step`
        (present as attributes for a ``NamedTuple`` response, or keys for a mapping).
    """

    def __init__(
        self,
        step_fn: Callable[[Any, Any, Any, Any], Any],
        *,
        prognostic_variables: tuple[str, ...],
        required_forcing: tuple[str, ...],
        provided_fluxes: tuple[str, ...],
    ) -> None:
        if not callable(step_fn):
            raise TypeError(
                f"SurfaceComponentAdapter needs a callable step_fn(grid, state, "
                f"forcing, dt) -> (new_state, response); got {step_fn!r}"
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

    def _check_response(self, response: Any) -> None:
        """Verify the wrapped step actually hands back every declared flux."""
        if isinstance(response, Mapping):
            present = set(response.keys())
        else:
            present = {f for f in self._provided_fluxes if hasattr(response, f)}
        missing = set(self._provided_fluxes) - present
        if missing:
            raise ValueError(
                f"SurfaceComponentAdapter declares provided_fluxes "
                f"{self._provided_fluxes} but the step response is missing "
                f"{sorted(missing)} ({type(response).__name__})"
            )

    def step(self, grid: Any, state: Any, forcing: Any, dt: Any) -> Any:
        """Advance the surface state one implicit step -> ``(new_state, response)``.

        Pure, so ``jax.grad`` flows through the step (D1); ``response`` carries the
        fluxes/fields the brick hands its partners (validated against
        :attr:`provided_fluxes`).
        """
        new_state, response = self._step_fn(grid, state, forcing, dt)
        self._check_response(response)
        return new_state, response
