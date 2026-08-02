"""Land<->atmosphere turbulent-flux INTERFACE: one owner for the shared limits.

A surface turbulent flux is exchanged between two components, so any bound on it
is a property of the INTERFACE, not of one participant.  Before this module the
two ends of the land tile disagreed:

* the land surface-energy balance (``legoesm.land.surface_scheme.simple_seb``)
  computed its flux with a transfer-coefficient ceiling AND a condensation
  floor;
* the atmospheric land tile
  (``legoesm.atmosphere.physics.turbulence.surface_layer``) computed the SAME
  flux with neither.

Whatever one end emitted the other was free to reject, so the coupling budget
``in - out - dStorage = 0`` did not close whenever either limit bound.  Both
limits now live in :class:`LandInterfaceFluxConfig` and are applied by the
single entry point :func:`land_interface_most_fluxes`, which BOTH ends call: the
two sides therefore agree by CONSTRUCTION for matched inputs, not by two
independently-maintained copies of the same two numbers.

Sign convention (positive UPWARD, surface -> atmosphere)
-------------------------------------------------------
Every flux here follows the convention of
:func:`legoesm.core.bulk_flux.compute_most_fluxes` and of the atmospheric
surface-layer contract: ``shflx`` and ``lhflx`` are POSITIVE UPWARD, i.e. out of
the surface into the atmosphere.

* ``lhflx > 0``  evaporation / sublimation — the surface loses water, the
  lowest atmospheric layer gains it.
* ``lhflx < 0``  condensation (dew / frost / fog deposition) — the atmosphere
  loses water, the surface gains it, and the latent heat of that phase change is
  released INTO the surface.

Both consumers already use this convention with the same sign:
the land SEB closes ``G_soil = SW_net + LW_net - shflx - lhflx`` (positive-up
turbulent fluxes SUBTRACTED from the into-surface radiative terms), and the
atmosphere adds ``dq/dt|_BL = g (lhflx / L) / dp_low`` to its bottom level
(positive ``lhflx`` moistens the air).  A condensation event is therefore a
negative ``lhflx`` at BOTH ends: it dries the bottom atmospheric layer and
deposits ``-lhflx`` of latent heat into the soil.

Why the condensation floor exists (do NOT drop it without re-testing #730)
-------------------------------------------------------------------------
A cold, dry skin under moister advected air produces a large negative (i.e.
condensing) latent flux; the released latent heat enters ``G_soil`` with a PLUS
sign (``-lhflx > 0``) and the surface energy balance answers with an
unphysically hot skin temperature, which the thin top soil layer at the
radiation timestep turns into a runaway (issue #730: land skin 224 K -> 1156 K
-> NaN in coupled AMIP).  The floor bounds only the CONDENSING branch;
evaporation stays unbounded so the SEB keeps its self-limiting feedback.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.bulk_flux import compute_most_fluxes


class LandInterfaceFluxConfig(NamedTuple):
    """Limits both ends of the land<->atmosphere turbulent interface apply.

    Fields
    ------
    max_exchange_coeff : float
        Ceiling on the neutral-equivalent bulk transfer coefficient
        ``C = kappa^2 / (denom_m * denom_h)`` passed to
        :func:`legoesm.core.bulk_flux.compute_most_fluxes`.  The MOST solver's
        default denominator floor of 0.5 admits ``C <= kappa^2/0.25 ~ 0.64``,
        about 190x the neutral land value ~3.4e-3; that turns a ~0.6 g/kg
        humidity gradient into a ~4900 W/m^2 latent shock on a cold-start
        column and NaNs the thin top soil layer.  0.02 is a generous
        strong-instability bound.  MEASURED to bind (2026-08-02, x64, MOST
        ``scheme="most"``, ``z_ref=10 m``): 1.38x at ``z0 = 0.1 m`` with
        ``|U| <= 2 m/s`` and ``dT >= 30 K`` (hot arid daytime surface layer),
        and up to 21x at ``z0 = 1 m`` (forest) — it is NOT an inert guard.
    condensation_floor_w : float
        Floor [W/m^2] on the CONDENSING (negative, positive-up) latent flux;
        see the module docstring for the #730 instability it guards.  Chosen
        safely above any real frost/dew flux (~O(10-100) W/m^2), so it binds
        only on pathological columns and on the strong warm-advection-over-
        frozen-ground corner.  Evaporation (positive ``lhflx``) is never
        limited.
    """

    max_exchange_coeff: float = 0.02
    condensation_floor_w: float = -150.0


__param_spec__ = {
    "LandInterfaceFluxConfig": {
        "scheme_key": "core.land_interface_flux",
        "excluded": {
            "max_exchange_coeff": (
                "numerics: cold-start ceiling on the MOST bulk transfer "
                "coefficient (regulariser bounding an unbounded log-law "
                "denominator, not a physical closure); it is also a shared "
                "INTERFACE contract — training it on one end would silently "
                "re-open the land/atmosphere flux disagreement"
            ),
            "condensation_floor_w": (
                "numerics: cold-start floor on the condensing latent flux "
                "(issue #730 thin-top-soil-layer runaway guard, not a "
                "physical closure); also a shared INTERFACE contract"
            ),
        },
        "params": {},
    },
}


#: Default limits.  Both ends of the interface read THIS instance unless a
#: caller threads an explicit config, so the two sides cannot drift apart.
LAND_INTERFACE_FLUX = LandInterfaceFluxConfig()


def apply_condensation_floor(
    lhflx: jax.Array,
    config: LandInterfaceFluxConfig = LAND_INTERFACE_FLUX,
) -> jax.Array:
    """Bound the CONDENSING branch of a positive-upward latent heat flux.

    Sign walk (convention: POSITIVE UPWARD, surface -> atmosphere).
    ``config.condensation_floor_w < 0``, so ``jnp.maximum`` raises the most
    negative (strongest condensation) values TOWARD zero — it reduces the
    magnitude of condensation and can never turn condensation into
    evaporation or change the sign of the flux.  Evaporation
    (``lhflx > 0 > condensation_floor_w``) passes through untouched, so the
    surface energy balance keeps its self-limiting evaporative feedback.

    ``jnp.maximum`` (not a Python ``if``): the branch is data-dependent per
    column, so it must be a traced select.  Differentiable a.e.; the gradient
    is 1 above the floor and 0 below it (a clamp, matching every other
    limiter in the flux chain).
    """
    return jnp.maximum(lhflx, config.condensation_floor_w)


def land_interface_most_fluxes(
    u_rel: jax.Array,
    v_rel: jax.Array,
    T_atm: jax.Array,
    q_atm: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    *,
    config: LandInterfaceFluxConfig = LAND_INTERFACE_FLUX,
    **most_kwargs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The SINGLE land-tile turbulent-flux entry point (both ends call this).

    Thin, limit-carrying wrapper over
    :func:`legoesm.core.bulk_flux.compute_most_fluxes`: it injects
    ``max_exchange_coeff`` from ``config`` and applies
    :func:`apply_condensation_floor` to the returned latent flux.  No flux
    numerics are re-derived here.

    Callers pass every remaining ``compute_most_fluxes`` argument through
    ``**most_kwargs`` (``z_ref``, ``z0_init``, ``scheme``, ``n_iter``,
    ``L_latent``, ...).  ``max_exchange_coeff`` may NOT be passed there — the
    interface owns it, and a caller-supplied value would be exactly the
    one-sided override this module exists to prevent.

    Returns
    -------
    tau_x, tau_y, shflx, lhflx, ustar
        As :func:`compute_most_fluxes`, with ``shflx``/``lhflx`` POSITIVE
        UPWARD and ``lhflx`` floored at ``config.condensation_floor_w``.
        ``return_2m`` / ``return_convergence`` are rejected: they change the
        return arity, and silently reshaping the tuple here would make the two
        ends unpack different things.
    """
    if "max_exchange_coeff" in most_kwargs:
        raise ValueError(
            "max_exchange_coeff is owned by LandInterfaceFluxConfig and must "
            "not be passed to land_interface_most_fluxes; thread a custom "
            "config= instead so BOTH ends of the interface see the same "
            "ceiling."
        )
    for _extra in ("return_2m", "return_convergence"):
        if most_kwargs.get(_extra):
            raise ValueError(
                f"{_extra}=True is not supported by "
                "land_interface_most_fluxes (it changes the return arity of "
                "the shared land-interface flux tuple); call "
                "compute_most_fluxes directly for that diagnostic."
            )
        most_kwargs.pop(_extra, None)

    tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
        u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho,
        max_exchange_coeff=config.max_exchange_coeff,
        **most_kwargs,
    )
    # Sensible heat carries NO extra interface limit: it has no analogue of the
    # #730 condensation runaway (the SEB's longwave + sensible slopes both damp)
    # and both ends already agree on it once they share max_exchange_coeff,
    # which caps u* and theta* consistently.
    return tau_x, tau_y, shflx, apply_condensation_floor(lhflx, config), ustar
