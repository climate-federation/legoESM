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
    clubb_coefficient_from_diffusivity,
    eddy_diffusivity_from_flux,
    entrainment_velocity_from_buoyancy_flux,
    momentum_diffusivity_from_fluxes,
    prandtl_number_from_diffusivities,
)
from legoesm.atmosphere.dynamics.rce_diagnostics import (
    resolved_turbulent_fluxes_plane,
    vertical_velocity_variance_plane,
)
from legoesm.atmosphere.physics._shared import mixing_length

_METHODS = (
    "eddy_diffusivity", "entrainment", "clubb_coefficient", "prandtl_number",
)

# Minimum number of valid interior interfaces for a trustworthy column C_K
# (a single anomalous-shear interface should not define the whole column).
_MIN_VALID_CK_LEVELS = 3


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


class ClubbCoefficientProfile(NamedTuple):
    """Dimensionless CLUBB-lite eddy coefficient ``C_K`` per interior interface."""

    z_m: jax.Array     # (nlev-1,) interior-interface heights [m], ascending
    C_K: jax.Array     # (nlev-1,) dimensionless C_K = K_m/(ℓ·√wp2)
    valid: jax.Array   # (nlev-1,) bool


def diagnose_clubb_coefficient(
    les_state,
    height_coord,
    *,
    l_mix_max: float,
    qv_slot: int = 0,
    min_valid_levels: int = _MIN_VALID_CK_LEVELS,
) -> ClubbCoefficientProfile:
    """Diagnose the DIMENSIONLESS CLUBB-lite coefficient ``C_K = K_m/(ℓ·√wp2)``.

    Unlike :func:`diagnose_eddy_diffusivity` (which returns a *dimensional* heat
    diffusivity ``K`` [m²/s]), this returns the actual GCM parameter: the
    dimensionless coefficient that, with the GCM's own mixing length and velocity
    scale, reproduces the LES MOMENTUM diffusivity — the exact inverse of
    ``Km = C_K·ℓ·√wp2`` (``clubb_lite.py``).  Using the momentum flux (not heat)
    avoids conflating ``C_K`` with the turbulent Prandtl number ``Pr_t``.

    Pipeline (all co-located at the ``nlev-1`` interior interfaces, reversed
    top-down→ascending together — the plane convention):
      * ``K_m`` from the resolved ``⟨w'u'⟩, ⟨w'v'⟩`` + mean-wind shear
        (:func:`momentum_diffusivity_from_fluxes`);
      * ``ℓ`` from the SAME Blackadar :func:`mixing_length` the GCM uses, with
        the GCM's ``l_mix_max``;
      * ``√wp2`` from the resolved vertical-velocity variance
        (:func:`vertical_velocity_variance_plane`, interior half levels);
      * ``C_K`` and the valid mask from
        :func:`clubb_coefficient_from_diffusivity`.
    Fewer than ``min_valid_levels`` valid interfaces ⇒ the WHOLE column is
    flagged invalid (a single anomalous-shear interface must not define ``C_K``);
    the loop then keeps the background.  Pure-JAX, AD-safe.
    """
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    wp2_half = vertical_velocity_variance_plane(les_state, height_coord)  # (nlev+1,) top-down
    u_mean = jnp.mean(les_state.u.data, axis=(0, 1))                      # (nlev,) top-down
    v_mean = jnp.mean(les_state.v.data, axis=(0, 1))
    z_full = jnp.asarray(height_coord.z_full)                            # (nlev,) top-down
    # Reverse EVERY array top-down→ascending together so K_m's flux + shear,
    # √wp2 and ℓ(z) are all co-located at the same ascending interior interfaces.
    w_u = fluxes.w_u[::-1]
    w_v = fluxes.w_v[::-1]
    u_asc = u_mean[::-1]
    v_asc = v_mean[::-1]
    z_asc = z_full[::-1]
    wp2_interior = wp2_half[1:-1][::-1]                                  # (nlev-1,) ascending
    z_m = fluxes.z_half_interior[::-1]                                   # (nlev-1,) ascending
    K_m, km_valid = momentum_diffusivity_from_fluxes(
        w_u, w_v, u_asc, v_asc, z_asc
    )
    l_mix = mixing_length(z_m, l_mix_max)
    C_K, valid = clubb_coefficient_from_diffusivity(
        K_m, km_valid, l_mix, wp2_interior
    )
    # Column-level trust guard: too few valid interfaces ⇒ invalidate the column.
    enough = jnp.sum(valid) >= int(min_valid_levels)
    valid = valid & enough
    return ClubbCoefficientProfile(z_m=z_m, C_K=C_K, valid=valid)


class PrandtlProfile(NamedTuple):
    """Turbulent Prandtl number ``Pr_t = K_m/K_h`` per interior interface."""

    z_m: jax.Array     # (nlev-1,) interior-interface heights [m], ascending
    Pr_t: jax.Array    # (nlev-1,) dimensionless Pr_t = K_m/K_h
    valid: jax.Array   # (nlev-1,) bool


def diagnose_prandtl_number(
    les_state,
    height_coord,
    *,
    qv_slot: int = 0,
    min_valid_levels: int = _MIN_VALID_CK_LEVELS,
) -> PrandtlProfile:
    """Diagnose the DIMENSIONLESS turbulent Prandtl number ``Pr_t = K_m/K_h``.

    The ratio of the LES MOMENTUM diffusivity ``K_m`` (shear-projected, from
    ``⟨w'u'⟩,⟨w'v'⟩``) to the HEAT diffusivity ``K_h`` (from ``⟨w'θ'⟩``), both at
    the same interior interfaces — the inverse of the GCM ``K_h = K_m/Pr_t``.
    Being a pure ratio, ``Pr_t`` carries NO wp2-identification assumption (unlike
    ``C_K``), so it transfers cleanly.  ``K_h`` reuses
    :func:`diagnose_eddy_diffusivity` (ascending, co-located with the momentum
    interfaces).  Fewer than ``min_valid_levels`` valid interfaces ⇒ the column
    is flagged invalid.  Pure-JAX, AD-safe.
    """
    kh = diagnose_eddy_diffusivity(les_state, height_coord, qv_slot)  # heat K, ascending
    fluxes = resolved_turbulent_fluxes_plane(les_state, height_coord, qv_slot)
    u_mean = jnp.mean(les_state.u.data, axis=(0, 1))
    v_mean = jnp.mean(les_state.v.data, axis=(0, 1))
    z_full = jnp.asarray(height_coord.z_full)
    K_m, km_valid = momentum_diffusivity_from_fluxes(
        fluxes.w_u[::-1], fluxes.w_v[::-1],
        u_mean[::-1], v_mean[::-1], z_full[::-1],
    )
    Pr_t, valid = prandtl_number_from_diffusivities(
        K_m, km_valid, kh.K, kh.valid
    )
    enough = jnp.sum(valid) >= int(min_valid_levels)
    return PrandtlProfile(z_m=kh.z_m, Pr_t=Pr_t, valid=valid & enough)


def diagnose_column_coefficient(
    les_state,
    height_coord,
    *,
    method: str = "eddy_diffusivity",
    qv_slot: int = 0,
    l_mix_max: float | None = None,
):
    """Dispatch to the chosen closure-coefficient diagnosis (raises on unknown).

    Returns an :class:`EddyDiffusivityProfile` for ``"eddy_diffusivity"``, an
    :class:`~legoesm.atmosphere.dynamics.les_closure_diagnosis.EntrainmentDiagnosis`
    for ``"entrainment"``, a :class:`ClubbCoefficientProfile` for
    ``"clubb_coefficient"`` (which requires the GCM ``l_mix_max``), or a
    :class:`PrandtlProfile` for ``"prandtl_number"``.
    """
    if method == "eddy_diffusivity":
        return diagnose_eddy_diffusivity(les_state, height_coord, qv_slot)
    if method == "entrainment":
        return diagnose_entrainment(les_state, height_coord, qv_slot)
    if method == "prandtl_number":
        return diagnose_prandtl_number(les_state, height_coord, qv_slot=qv_slot)
    if method == "clubb_coefficient":
        if l_mix_max is None:
            raise ValueError(
                "method='clubb_coefficient' requires l_mix_max (the GCM "
                "CLUBBLiteConfig.l_mix_max) to evaluate the mixing length."
            )
        return diagnose_clubb_coefficient(
            les_state, height_coord, l_mix_max=l_mix_max, qv_slot=qv_slot
        )
    raise ValueError(
        f"Unknown column-LES diagnosis method {method!r}; choose from {_METHODS}."
    )
