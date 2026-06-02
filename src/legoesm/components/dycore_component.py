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
rather than silently dropping it.  A physics/forcing-coupled core (e.g. a
primitive-equation core whose ``tendencies`` takes a ``physics_tendency``) is
wrapped by :class:`ForcedDycoreComponent` instead — it maps the AbstractComponent
``forcing`` payload into that ``physics_tendency`` parameter.

**Instantaneous (dt-independent) RHS.**  :meth:`tendency` calls
``model.tendencies(state)`` with no time step, so a core whose ``tendencies`` takes
an optional ``dt`` (e.g. MPAS shallow water, whose anticipated-potential-vorticity
APVM stabilizer is gated on ``dt > 0``) yields its continuous, ``dt``-independent
right-hand side.  This is the correct seam for an ``AbstractComponent`` tendency:
``dt``-scaled *numerical* stabilizers belong to the time-discrete ``step``, not the
continuous dynamics, and are intentionally excluded here — by contract, not by
silent drop.
"""

from __future__ import annotations

import inspect
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
                "tendencies(..., physics_tendency=...) signature — see "
                "ForcedDycoreComponent."
            )
        return self._model.tendencies(state)


class ForcedDycoreComponent(AbstractComponent):
    """An :class:`AbstractComponent` over a dynamical core that ACCEPTS an external
    physics tendency.

    The counterpart to :class:`DycoreComponent` for cores whose
    ``tendencies(state, physics_tendency=...)`` ADD an externally supplied physics/
    forcing tendency (a state-tendency-shaped pytree produced by the physics
    parameterizations — same structure as the dynamical tendency the core returns)
    to their dynamical right-hand side.  This wrapper routes the AbstractComponent
    ``forcing`` payload straight into that ``physics_tendency`` parameter, so a
    primitive-equation core becomes a forcing-coupled brick that a coupler can drive
    with a physics tendency.

    Contract:

    * ``required_forcing`` = ``(forcing_name,)`` — the physics tendency it consumes
      (default ``"physics_tendency"``).
    * ``forcing`` MAY be ``None`` -> the core runs pure dynamics (the
      ``physics_tendency`` parameter is left at its default).  A non-``None``
      ``forcing`` is passed through unmodified — the core adds it leaf-wise, so it
      must match the core's tendency structure (the caller/coupler owns that).
    * ``params`` MUST be ``None`` — this wrapper exposes no tunable-parameter seam
      and rejects a payload rather than silently dropping it.
    * Same **instantaneous (dt-independent) RHS** contract as
      :class:`DycoreComponent`: ``tendencies`` is called with no ``dt``.

    Construction validates that the core's ``tendencies`` actually accepts a
    ``physics_tendency`` keyword; a DRY core (no such parameter) is rejected with a
    pointer back to :class:`DycoreComponent`.
    """

    def __init__(
        self,
        model: Any,
        *,
        prognostic_variables: tuple[str, ...],
        forcing_name: str = "physics_tendency",
    ) -> None:
        tendencies = getattr(model, "tendencies", None)
        if not callable(tendencies):
            raise TypeError(
                f"dycore {model!r} does not expose a callable 'tendencies' method, "
                "so it cannot be wrapped as an AbstractComponent brick"
            )
        try:
            sig = inspect.signature(tendencies)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"dycore {model!r}.tendencies signature could not be inspected "
                f"({exc}); cannot confirm it accepts a physics tendency"
            ) from exc
        if "physics_tendency" not in sig.parameters:
            raise TypeError(
                f"ForcedDycoreComponent requires a core whose tendencies(...) accepts "
                f"a 'physics_tendency' parameter; {model!r} does not (signature "
                f"{sig}).  Use DycoreComponent for a DRY core."
            )
        self._model = model
        self._prognostic_variables = tuple(prognostic_variables)
        self._forcing_name = str(forcing_name)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def prognostic_variables(self) -> tuple[str, ...]:
        return self._prognostic_variables

    @property
    def required_forcing(self) -> tuple[str, ...]:
        return (self._forcing_name,)  # the physics tendency the core consumes

    @property
    def provided_fluxes(self) -> tuple[str, ...]:
        return ()  # still a dynamical core — provides no surface flux

    def tendency(self, grid: Any, state: Any, forcing: Any, params: Any) -> Any:
        """The dynamical RHS plus the supplied physics tendency (pure, D1).

        ``forcing`` is the physics tendency (or ``None`` for pure dynamics) and is
        routed into ``model.tendencies(state, physics_tendency=forcing)``.  ``grid``
        is carried by the model; ``params`` MUST be ``None`` (no tunable-param seam).
        """
        if params is not None:
            raise ValueError(
                "ForcedDycoreComponent exposes no tunable-parameter seam; got "
                "non-None params.  Pass the physics tendency via `forcing`."
            )
        return self._model.tendencies(state, physics_tendency=forcing)
