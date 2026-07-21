"""Snow-on-ice column physics.

A single bulk snow layer is tracked per ice category with depth
``h_snow`` and constant density ``rho_snow``.  Snow modifies the
sea-ice column in four ways:

1. **Accumulation** — falling snow (forcing ``precip_snow``) adds
   mass on top of existing ice.  Snow falling on open water is
   either treated as rain (delivered to the ocean as freshwater) or
   absorbed into the lead-freeze new-ice flux; default here is to
   convert it to ocean freshwater.
2. **Conductivity** — combined snow + ice conductive resistance is
   the harmonic sum ``h_snow/k_snow + h_ice/k_ice``.  Snow reduces
   the conductive flux that warms (cools) the ice surface,
   slowing growth and melt.
3. **Melt / sublimation** — surface melt energy and sublimation
   mass deplete snow first; ice only when snow is exhausted.
4. **Snow-ice flooding (white-ice)** — when freeboard goes
   negative (snow load sinks the snow-ice interface below sea
   level), snow at the bottom of the snow column consolidates
   into white ice.  Leppäranta 1983 / Notz 2002 flotation form: the
   flooded pore space draws ``(rho_ice - rho_snow) d`` of seawater
   that the caller debits from the ocean (see ``snow_ice_flooding``).

All kernels are pure JAX (differentiable, JIT-friendly) and operate
on arbitrary leading spatial shape with optional trailing category
axis.

Faithfulness
------------
The two closed forms are pinned by
``tests/ice/unit/test_ice_snow_faithful.py`` against an independent scalar
oracle (rel 1e-9):

  * ``combined_conductance`` / ``combined_conductive_flux`` (Semtner 1976 /
    Maykut-Untersteiner 1971 SERIES thermal resistance): K = 1 / (h_snow/k_snow
    + h_ice/k_ice) with hard max(h, h_min) thickness floors, F = K (T_base -
    T_sfc).
  * ``snow_ice_flooding`` (Leppäranta 1983 / Notz 2002 Archimedes flotation):
    fb = ((rho_ocean - rho_ice) h_ice - rho_snow h_snow)/rho_ocean, and when
    fb < 0 the flooded thickness d = -fb rho_ocean/(rho_ocean - rho_ice +
    rho_snow) (capped at h_snow) with h_ice' = h_ice + d, h_snow' = h_snow - d.
    The pin includes the DEFINING flotation identity — un-capped flooding drives
    the new freeboard to EXACTLY zero — and the (rho_ice - rho_snow) d seawater
    mass draw.

The rho_ice/rho_snow/rho_ocean/k_ice/k_snow coefficients are transcribed as
independent oracle literals and canaried against ``legoesm.constants`` (CICE
defaults; constants == literal == value).  DEPARTURES / guards (reproduced by
the oracle): the max(h, h_min) resistance floors, the max(-fb, 0) flooding gate,
and the max(h_snow - d, 0) floor.  The d <= h_snow cap is an UNREACHABLE safety
for physical densities (rho_snow < rho_ice < rho_ocean): an active flood always
has d = (rho_snow h_snow - (rho_ocean - rho_ice) h_ice)/(rho_ocean - rho_ice +
rho_snow) < h_snow, so it is retained only as a defensive floor.
"""

from __future__ import annotations

import jax.numpy as jnp


# ==============================================================================
# Combined snow + ice conductive flux
# ==============================================================================

def combined_conductive_flux(
    T_base: jnp.ndarray,
    T_sfc: jnp.ndarray,
    h_ice: jnp.ndarray,
    h_snow: jnp.ndarray,
    k_ice: float,
    k_snow: float,
    h_ice_min: float,
    h_snow_min: float,
) -> jnp.ndarray:
    """Conductive flux through the combined snow+ice column.

    F_cond = (T_base - T_sfc) / (h_snow/k_snow + h_ice/k_ice)

    Hard ``max(h, h_min)`` floors prevent ``1/0`` when a layer is absent.
    Sign convention: positive = upward energy into the surface
    (from a warm base towards a cold surface).

    Parameters
    ----------
    T_base : array
        Bottom (ice-ocean interface) temperature [K].  Sea-ice models
        typically use ``T_freeze_ocean ≈ 271.35 K``.
    T_sfc : array
        Snow / ice top surface temperature [K].
    h_ice, h_snow : array
        Ice and snow column thickness [m].
    k_ice, k_snow : float
        Thermal conductivities [W/(m·K)].
    h_ice_min, h_snow_min : float
        Hard max(h, h_min) lower floors on layer thickness used inside the
        resistance sum to avoid divisions by zero in JIT.
    """
    return combined_conductance(
        h_ice, h_snow, k_ice, k_snow, h_ice_min, h_snow_min,
    ) * (T_base - T_sfc)


def combined_conductance(
    h_ice: jnp.ndarray,
    h_snow: jnp.ndarray,
    k_ice: float,
    k_snow: float,
    h_ice_min: float,
    h_snow_min: float,
) -> jnp.ndarray:
    """Series snow+ice thermal conductance ``K = 1 / R_total`` [W/(m^2 K)].

    ``R_total = h_snow/k_snow + h_ice/k_ice`` (with hard max(h, h_min)
    thickness floors).  The conductive flux is ``K * (T_base - T_sfc)``.  Exposed
    separately from :func:`combined_conductive_flux` so the surface energy
    balance can treat the conductive term semi-implicitly (evaluate it at
    the *new* surface temperature), which is unconditionally stable for thin
    ice where the explicit ``K*dt/skin_cap`` greatly exceeds 1.
    """
    h_i_eff = jnp.maximum(h_ice, h_ice_min)
    h_s_eff = jnp.maximum(h_snow, h_snow_min)
    R_total = h_s_eff / k_snow + h_i_eff / k_ice
    return 1.0 / R_total


# ==============================================================================
# Snow accumulation
# ==============================================================================

def accumulate_snowfall(
    h_snow: jnp.ndarray,
    precip_snow: jnp.ndarray,
    ice_mask: jnp.ndarray,
    dt: float,
    rho_snow: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add new snowfall to the snow column.

    Snow that falls on open water (``ice_mask`` False) is returned
    as a freshwater flux equivalent rather than ice-column snow —
    callers route this to the ocean as freshwater.

    Parameters
    ----------
    h_snow : array
        Current snow depth [m].
    precip_snow : array
        Snow precipitation rate [kg/m²/s].
    ice_mask : bool array
        True where ice exists (precip_snow is captured); False
        where it falls on open water.
    dt : float
        Time step [s].
    rho_snow : float
        Snow density [kg/m³].

    Returns
    -------
    h_snow_new : array
        Updated snow depth [m].
    snow_to_ocean_kg_m2_s : array
        Snow mass per area per second that fell on open water and
        becomes freshwater to the ocean [kg/m²/s].
    """
    snowfall_m = (precip_snow * dt) / rho_snow
    delta_on_ice = jnp.where(ice_mask, snowfall_m, 0.0)
    snow_to_ocean = jnp.where(ice_mask, 0.0, precip_snow)
    return h_snow + delta_on_ice, snow_to_ocean


# ==============================================================================
# Snow melt and sublimation (drawn from snow first, then ice)
# ==============================================================================

def consume_from_snow_then_ice(
    energy_per_area: jnp.ndarray,
    h_snow: jnp.ndarray,
    h_ice: jnp.ndarray,
    rho_snow: float,
    rho_ice: float,
    L_f: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert melt energy into snow then ice removal.

    Snow is consumed first; any remaining energy melts ice.  Both
    arrays are clamped to be non-negative.

    Parameters
    ----------
    energy_per_area : array
        Available melt energy [J/m²] (positive = melt).  Negative
        values are clipped to zero — refreezing is handled
        separately.
    h_snow, h_ice : array
        Current snow and ice thickness [m].
    rho_snow, rho_ice : float
        Densities [kg/m³].
    L_f : float
        Latent heat of fusion [J/kg].

    Returns
    -------
    h_snow_new : array
    h_ice_new : array
    snow_melt_m : array
        Snow column actually melted [m, of snow].  Useful for
        freshwater accounting.
    ice_melt_m : array
        Ice column actually melted [m, of ice].
    unconsumed_energy_per_area : array
        Surface melt energy [J/m²] left over AFTER the snow + the FULL
        ice column latent capacity has been consumed (i.e. the column
        melted out with surplus heat).  Non-negative.  Sign/energy
        convention: this is heat that arrived at the surface but found no
        ice/snow left to melt; the caller MUST route it to the ocean
        mixed layer (ocean GAINS this heat) so the column+ocean energy
        budget closes.  It was previously dropped (lost) on a melt-out
        step — finding #6.
    """
    E_pos = jnp.maximum(energy_per_area, 0.0)

    # Snow melt, capped at the available snow latent capacity.
    snow_melt_capacity_kg_m2 = jnp.maximum(h_snow, 0.0) * rho_snow
    energy_for_snow = jnp.minimum(E_pos, snow_melt_capacity_kg_m2 * L_f)
    snow_melt_m = (energy_for_snow / L_f) / rho_snow
    h_snow_new = jnp.maximum(h_snow - snow_melt_m, 0.0)

    # Ice melt from the remaining energy, CAPPED at the available ice latent
    # capacity.  Without the cap the *reported* ``ice_melt_m`` could exceed
    # ``h_ice`` (while ``h_ice_new`` clamped to 0), inflating the freshwater /
    # salt / pond diagnostics the callers build from it — reporting more ice
    # melted than ever existed (energy/mass non-closure).
    energy_remaining = E_pos - energy_for_snow
    ice_melt_capacity_kg_m2 = jnp.maximum(h_ice, 0.0) * rho_ice
    energy_for_ice = jnp.minimum(energy_remaining, ice_melt_capacity_kg_m2 * L_f)
    ice_melt_m = (energy_for_ice / L_f) / rho_ice
    h_ice_new = jnp.maximum(h_ice - ice_melt_m, 0.0)

    # Energy left after the snow AND the full ice column have melted (column
    # fully ablated with surplus heat).  Previously dropped (silently lost);
    # now RETURNED so the caller credits the ocean mixed layer (finding #6).
    # >= 0 by construction (energy_for_snow + energy_for_ice <= E_pos).
    unconsumed_energy_per_area = jnp.maximum(
        E_pos - energy_for_snow - energy_for_ice, 0.0,
    )

    return (
        h_snow_new, h_ice_new, snow_melt_m, ice_melt_m,
        unconsumed_energy_per_area,
    )


def consume_sublimation_from_snow_then_ice(
    sublim_mass_per_area: jnp.ndarray,
    h_snow: jnp.ndarray,
    h_ice: jnp.ndarray,
    rho_snow: float,
    rho_ice: float,
    sublim_partition: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sublimate snow first, then ice.

    ``sublim_mass_per_area`` is positive when mass leaves the
    surface (drying) and negative when there is deposition.

    Parameters
    ----------
    sublim_mass_per_area : array
        Net sublimation mass loss per area [kg/m²].  Positive = ice
        / snow mass removed; negative = deposition added to snow.
    sublim_partition : float
        Fraction of positive sublimation that draws from snow
        first.  ``1.0`` (default) drains snow before tapping ice;
        ``0.0`` would draw both layers simultaneously.

    Returns
    -------
    h_snow_new, h_ice_new : arrays
    snow_sublim_m, ice_sublim_m : arrays
        Layer thickness changes from sublimation [m] (positive =
        loss).
    """
    # Deposition (mass < 0) deposits onto snow column.
    deposition_kg = jnp.maximum(-sublim_mass_per_area, 0.0)
    h_snow_dep = h_snow + deposition_kg / rho_snow

    # Sublimation (mass > 0) draws from snow first via
    # ``sublim_partition`` then any remainder from ice.
    sublim_pos_kg = jnp.maximum(sublim_mass_per_area, 0.0)
    snow_avail_kg = h_snow_dep * rho_snow
    snow_target_kg = sublim_partition * sublim_pos_kg
    snow_sublim_kg = jnp.minimum(snow_target_kg, snow_avail_kg)
    snow_sublim_m = snow_sublim_kg / rho_snow
    h_snow_new = jnp.maximum(h_snow_dep - snow_sublim_m, 0.0)

    # Ice sublimation from the remainder, CAPPED at the available ice mass
    # so the reported ``ice_sublim_m`` can never exceed ``h_ice`` (same
    # over-removal failure mode as the melt path: an uncapped value would
    # inflate the concentration retreat and the brine delta_V_sublim budget
    # in _thermo_v2 while h_ice_new clamps to 0).  Any sublimation demand
    # beyond the column simply ablates it fully.
    ice_target_kg = jnp.maximum(sublim_pos_kg - snow_sublim_kg, 0.0)
    ice_avail_kg = jnp.maximum(h_ice, 0.0) * rho_ice
    ice_sublim_kg = jnp.minimum(ice_target_kg, ice_avail_kg)
    ice_sublim_m = ice_sublim_kg / rho_ice
    h_ice_new = jnp.maximum(h_ice - ice_sublim_m, 0.0)

    return h_snow_new, h_ice_new, snow_sublim_m, ice_sublim_m


# ==============================================================================
# Snow-ice flooding (Archimedes / white-ice formation)
# ==============================================================================

def snow_ice_flooding(
    h_ice: jnp.ndarray,
    h_snow: jnp.ndarray,
    rho_ice: float,
    rho_snow: float,
    rho_ocean: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply Archimedes snow-ice flooding (white-ice formation).

    When ``freeboard < 0`` the bottom of the snow column sinks
    below sea level, seawater wets the basal snow, and the snow
    consolidates into white ice.  In the Leppäranta 1983 / Notz 2002
    flotation solution the snow+ice column GAINS ``(rho_ice - rho_snow) d`` of
    mass — the seawater drawn into the flooded pore space — which the caller
    debits from the ocean (see the mass balance below); its salt is handled in
    :mod:`legoesm.ice.brine`.  (The ice gain ``rho_ice d`` minus the snow loss
    ``rho_snow d`` equals that seawater draw.)

    Freeboard:
        fb = ((rho_ocean - rho_ice) * h_ice - rho_snow * h_snow) / rho_ocean

    When ``fb < 0`` (snow-load sinks the surface), Leppäranta 1983 / Notz 2002:
    the submerged basal snow is FLOODED by seawater filling its pore space and
    then freezes, so a snow layer of thickness ``d`` becomes snow-ice of the
    SAME thickness at ice density.  The flotation solve raises the freeboard to
    exactly zero (``h_ice' = h_ice + d``, ``h_snow' = h_snow - d``):
        d = rho_ocean * (-fb) / (rho_ocean - rho_ice + rho_snow)
    Mass balance (per unit area):
        rho_ice * d          (new snow-ice)
      = rho_snow * d         (snow consumed)
      + (rho_ice - rho_snow) * d   (SEAWATER drawn from the ocean into pores)
    The ``(rho_ice - rho_snow) * d`` seawater is a real OCEAN mass sink (water +
    its salt); the caller debits it from the ocean so flooding does not
    spuriously freshen it (audit: the old ``d_s = (rho_ice/rho_snow)*d_i``
    snow->ice form exchanged NO ocean mass yet the brine module still charged
    the ocean the white-ice salt -> salt-without-water freshening).

    Parameters
    ----------
    h_ice, h_snow : array
        Ice and snow thickness [m].
    rho_ice, rho_snow, rho_ocean : float
        Densities [kg/m³].

    Returns
    -------
    h_ice_new : array
        Ice thickness after flooding [m].
    h_snow_new : array
        Snow thickness after flooding [m].
    h_si_formed : array
        Snow-ice thickness ``d`` formed this step [m].  The caller derives the
        ocean seawater withdrawal ``(rho_ice - rho_snow) * h_si_formed`` and the
        brine module the white-ice salt it carries.
    """
    fb_num = (rho_ocean - rho_ice) * h_ice - rho_snow * h_snow
    fb = fb_num / rho_ocean
    # Flotation-to-zero flooding thickness; denom > 0 for physical densities
    # (rho_snow < rho_ice < rho_ocean).  Floor guards the divide only.
    denom = jnp.maximum(rho_ocean - rho_ice + rho_snow, 1.0e-30)
    delta = jnp.maximum(-fb, 0.0) * rho_ocean / denom
    # Defensive cap: cannot flood more snow than the column holds.  Unreachable
    # for physical densities (rho_snow < rho_ice < rho_ocean) — an active flood
    # always has d < h_snow — but kept so a non-physical density set cannot flood
    # negative snow.  Ice gain, snow loss and seawater draw all scale with the
    # same d, so mass stays consistent even under the cap.
    delta = jnp.minimum(delta, h_snow)

    h_ice_new = h_ice + delta
    h_snow_new = jnp.maximum(h_snow - delta, 0.0)
    return h_ice_new, h_snow_new, delta
