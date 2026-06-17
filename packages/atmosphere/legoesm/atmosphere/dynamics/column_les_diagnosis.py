"""Diagnose a closure coefficient from a finished column-LES run.

Stage 6 composition of ``docs/COMPARE_REANALYSIS.md``: after the spun-off LES for
a flagged column reaches statistical steady state, turn its resolved turbulent
fluxes into the per-column closure coefficient that feeds the spatially-varying
parameter field.  This is the glue between the two already-tested leaves:

* resolved fluxes — :func:`legoesm.atmosphere.dynamics.rce_diagnostics.resolved_turbulent_fluxes_plane`,
* closure inversion — :mod:`legoesm.atmosphere.dynamics.les_closure_diagnosis`.

Two methods (selected by ``method``; an unknown method raises — dispatch
hardening):

* ``"eddy_diffusivity"`` — the down-gradient eddy diffusivity ``K`` profile from
  ``w'θ'`` and the domain-mean ``θ`` gradient.
* ``"entrainment"`` — the boundary-layer entrainment velocity ``w_e`` from the
  resolved buoyancy flux ``w'θ_v'`` at the inversion (the doc's headline
  coefficient).

Vertical-orientation handling: the plane state / height coordinate store fields
**top-down** (index 0 = model top), but the closure functions require ascending
height; this module reverses the relevant arrays consistently before calling
them (resolved fluxes at the ``nlev-1`` interior interfaces reverse to the same
ascending interfaces as the reversed mean-profile gradient).  Pure-JAX, AD-safe.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
    EntrainmentDiagnosis,
    eddy_diffusivity_from_flux,
    entrainment_velocity_from_buoyancy_flux,
)
from legoesm.atmosphere.dynamics.rce_diagnostics import (
    resolved_turbulent_fluxes_plane,
)

_METHODS = ("eddy_diffusivity", "entrainment")


class EddyDiffusivityProfile(NamedTuple):
    """Down-gradient eddy diffusivity ``K`` per interior interface (ascending z)."""

    z_m: jax.Array     # (nlev-1,) interior-interface heights [m], ascending
    K: jax.Array       # (nlev-1,) eddy diffusivity [m²/s]
    valid: jax.Array   # (nlev-1,) bool: significant gradient AND K>=0


def _domain_mean_theta(les_state, height_coord) -> jax.Array:
    """Domain-mean potential temperature profile ``⟨θ⟩(z)`` [K], top-down."""
    return height_coord.theta_ref + jnp.mean(
        les_state.theta_prime.data, axis=(0, 1)
    )


def diagnose_eddy_diffusivity(
    les_state, height_coord, qv_slot: int = 0
) -> EddyDiffusivityProfile:
    """Diagnose ``K = -w'θ'/(∂⟨θ⟩/∂z)`` per interior interface from the LES.

    Reverses the top-down resolved heat flux + mean-θ + heights to ascending so
    the flux and the gradient are co-located at the same ascending interfaces.
    """
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    theta_mean = _domain_mean_theta(les_state, height_coord)        # (nlev,) top-down
    z_full = jnp.asarray(height_coord.z_full)                       # (nlev,) top-down
    # Reverse to ascending z; the interior-interface flux reverses to the same
    # ascending interfaces as the reversed full-level gradient.
    diag = eddy_diffusivity_from_flux(
        fluxes.w_theta[::-1], theta_mean[::-1], z_full[::-1]
    )
    return EddyDiffusivityProfile(
        z_m=fluxes.z_half_interior[::-1], K=diag.K, valid=diag.valid
    )


def diagnose_entrainment(
    les_state, height_coord, qv_slot: int = 0
) -> EntrainmentDiagnosis:
    """Diagnose the entrainment velocity ``w_e`` from the LES buoyancy flux.

    Builds the domain-mean virtual potential temperature ``⟨θ_v⟩`` (same ``ε``
    convention as :func:`compute_cape`), reverses to ascending, and inverts the
    buoyancy-flux minimum at the inversion.
    """
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    theta_total = height_coord.theta_ref + les_state.theta_prime.data
    q_v = les_state.tracers.data[..., qv_slot]
    coeff = 1.0 / constants.epsilon - 1.0
    theta_v = theta_total * (1.0 + coeff * q_v)
    thetav_mean = jnp.mean(theta_v, axis=(0, 1))                    # (nlev,) top-down
    return entrainment_velocity_from_buoyancy_flux(
        fluxes.w_thetav[::-1], thetav_mean[::-1], fluxes.z_half_interior[::-1]
    )


def diagnose_column_coefficient(
    les_state, height_coord, *, method: str = "eddy_diffusivity", qv_slot: int = 0
):
    """Dispatch to the chosen closure-coefficient diagnosis (raises on unknown).

    Returns an :class:`EddyDiffusivityProfile` for ``"eddy_diffusivity"`` or an
    :class:`~legoesm.atmosphere.dynamics.les_closure_diagnosis.EntrainmentDiagnosis`
    for ``"entrainment"``.
    """
    if method == "eddy_diffusivity":
        return diagnose_eddy_diffusivity(les_state, height_coord, qv_slot)
    if method == "entrainment":
        return diagnose_entrainment(les_state, height_coord, qv_slot)
    raise ValueError(
        f"Unknown column-LES diagnosis method {method!r}; choose from {_METHODS}."
    )
