"""Wrap a dynamical core as an :class:`AbstractComponent` brick (Stage B4).

The production dycores are ``step(state, dt)`` solvers (``DycoreProtocol``) that
ALSO expose their dynamical right-hand side as ``tendencies(state)``.  This wrapper
turns one into an :class:`AbstractComponent` — the interchangeable Earth-system
lego brick — additively, without touching the dycore: it declares the brick's
metadata (``prognostic_variables`` it evolves; ``required_forcing`` it needs from
partners — empty for a dry core; ``provided_fluxes`` it hands back — none for the
pure dynamics) and routes the abstract ``tendency`` seam to ``model.tendencies``.

It is **generic** — it holds a model *instance* and never imports a concrete
dycore — so it lives in ``components`` without a component<->component import; a
caller (or test) supplies the atmosphere/ocean dycore.  Per the AbstractComponent
hard rule the resulting brick is pure and runs standalone (its dynamical tendency
is ``jax.grad``-differentiable — D1); see ``tests/unit/test_dycore_component.py``.

**DRY-only.**  This wrapper is for a *dry* dynamical core whose tendency is a
function of its own state alone: ``required_forcing`` and ``provided_fluxes`` are
therefore empty, and :meth:`tendency` rejects any ``forcing``/``params`` payload
rather than silently dropping it.  A physics/forcing-coupled component (e.g. a
primitive-equation core whose ``tendencies`` takes a ``physics_tendency``) needs a
distinct wrapper that maps the forcing into that signature — NOT this one.
"""

from __future__ import annotations

from typing import Any

from legoesm.components.protocol import AbstractComponent


class DycoreComponent(AbstractComponent):
    """An :class:`AbstractComponent` over a DRY dynamical core's tendency RHS."""

    def __init__(
        self,
        model: Any,
        *,
        prognostic_variables: tuple[str, ...],
    ) -> None:
        tendencies = getattr(model, "tendencies", None)
        if not callable(tendencies):
            raise TypeError(
                f"dycore {model!r} does not expose a callable 'tendencies(state)' "
                "method, so it cannot be wrapped as an AbstractComponent brick"
            )
        self._model = model
        self._prognostic_variables = tuple(prognostic_variables)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def prognostic_variables(self) -> tuple[str, ...]:
        return self._prognostic_variables

    @property
    def required_forcing(self) -> tuple[str, ...]:
        return ()  # a dry dynamical core needs no partner forcing

    @property
    def provided_fluxes(self) -> tuple[str, ...]:
        return ()  # the pure dynamics provides no surface flux

    def tendency(self, grid: Any, state: Any, forcing: Any, params: Any) -> Any:
        """The dynamical RHS of the wrapped core (pure ``state -> dstate``, D1).

        A dry dynamical core's tendency is a function of its own state alone; the
        ``grid`` is carried by the model.  ``forcing``/``params`` MUST be ``None``
        — a non-None payload is rejected rather than silently dropped (this dry
        wrapper does not route physics/partner forcing into the core).
        """
        if forcing is not None or params is not None:
            raise ValueError(
                "DycoreComponent wraps a DRY dynamical core and ignores external "
                "forcing/params; got non-None forcing/params.  A forcing-coupled "
                "component needs a wrapper that maps the payload into the dycore's "
                "tendencies(..., physics_tendency=...) signature."
            )
        return self._model.tendencies(state)
