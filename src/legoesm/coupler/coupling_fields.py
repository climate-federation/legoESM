"""Fixed-slot coupling field containers.

All fields pre-allocated with fixed shape — no dicts, no Optional types.
These NamedTuples define the strict interface between atmosphere and surface.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class AtmToSurface(NamedTuple):
    """Atmosphere -> surface coupling fields. Shape (6, n, n)."""
    sw_down: jax.Array           # Downward shortwave [W/m2]
    lw_down: jax.Array           # Downward longwave [W/m2]
    precip_total: jax.Array      # Total precipitation rate [kg/m2/s]
    precip_snow: jax.Array       # Snow precipitation rate [kg/m2/s]
    T_lowest: jax.Array          # Lowest-level temperature [K]
    q_lowest: jax.Array          # Lowest-level specific humidity [kg/kg]
    u_lowest: jax.Array          # Lowest-level zonal wind [m/s]
    v_lowest: jax.Array          # Lowest-level meridional wind [m/s]
    p_lowest: jax.Array          # Lowest-level pressure [Pa]
    p_surface: jax.Array         # Surface pressure [Pa]
    rho_lowest: jax.Array        # Lowest-level air density [kg/m3]
    cos_zenith: jax.Array        # Cosine solar zenith angle
    co2_ppmv: jax.Array          # CO2 concentration [ppmv]
    has_radiation: jax.Array     # 1.0 = radiation fields valid, 0.0 = not
    has_precipitation: jax.Array # 1.0 = precip fields valid, 0.0 = not


class TileResponse(NamedTuple):
    """Per-tile surface response returned to the coupler.

    **Temporal semantics**: state fields (``T_surface``, ``q_surface``,
    ``albedo``, ``lw_up``) reflect the **end-of-step** surface state.
    Turbulent fluxes (``shflx``, ``lhflx``, ``tau_x``, ``tau_y``) are
    computed from the **beginning-of-step** state and represent the
    time-step-averaged exchange.  This is standard practice in land
    surface models (the fluxes drove the state update, so they are
    self-consistent with the energy/water budget over the step).

    **freshwater_flux** (kg/m²/s, positive INTO ocean / surface) is
    the net liquid-water mass flux from this tile to the ocean — for
    land it is `runoff_surface + runoff_subsurface`, for sea-ice it is
    melt + brine + sublimation mass, for ocean it is `precip - evap`.
    Tiles that do not produce a freshwater channel return zeros.
    Added in the Physical_Consistency cycle (audit F4).
    """
    T_surface: jax.Array         # [end-of-step] Surface skin temperature [K]
    albedo: jax.Array            # [end-of-step] Surface albedo [0-1]
    emissivity: jax.Array        # Surface emissivity [0-1]
    z0: jax.Array                # Roughness length [m]
    q_surface: jax.Array         # [end-of-step] Surface specific humidity [kg/kg]
    shflx: jax.Array             # [step-averaged] Sensible heat flux [W/m2] (positive up)
    lhflx: jax.Array             # [step-averaged] Latent heat flux [W/m2] (positive up)
    tau_x: jax.Array             # [step-averaged] Zonal surface stress [Pa]
    tau_y: jax.Array             # [step-averaged] Meridional surface stress [Pa]
    lw_up: jax.Array             # [end-of-step] Upward longwave [W/m2]
    u_ocean_sfc: jax.Array       # Ocean surface zonal current [m/s]
    v_ocean_sfc: jax.Array       # Ocean surface meridional current [m/s]
    co2_flux: jax.Array          # CO2 flux [kg/m2/s] (positive up)
    # Net liquid-water mass flux delivered by this tile [kg/m²/s,
    # positive into the ocean / receiving body].  Zero for tiles that
    # do not produce a freshwater channel — but must always be a
    # populated array, never None, so the structure is pytree-uniform
    # across all four tiles (jax.tree.map otherwise raises a
    # tree-prefix mismatch when one tile has None and others have
    # arrays).
    freshwater_flux: jax.Array
    # Heat flux extracted from the ocean by this tile [W/m²,
    # positive = ocean LOSES energy to this tile].  Sea-ice draws
    # heat from the warm ocean to melt at its base (F_ocean), and
    # latent heat of fusion is removed from the ocean when open
    # water freezes — both should be subtracted from the ocean
    # column heat budget.  Land/lake tiles return zeros (no direct
    # ocean exchange).  Ocean tile reports zero (it is the source,
    # not a sink).  Audit F8.
    ocean_heat_extraction: jax.Array
    # Stress applied by this tile back onto the ocean surface [Pa,
    # positive = eastward / northward force on the ocean].  By
    # Newton's third law, the air→ice and ocean→ice stresses produce
    # an equal-and-opposite reaction force on the ocean column under
    # the ice.  Sea-ice tiles deliver −tau_ocean_from_ice (the
    # ocean→ice drag inverted).  Land/lake tiles return zeros.
    # Ocean tile returns zero (its own wind stress is already in
    # tau_x/tau_y).  Audit F9.
    ocean_stress_x: jax.Array
    ocean_stress_y: jax.Array
    # Phase-aware surface moisture mass flux [kg/m²/s, positive = up
    # = drying].  Each tile populates this directly using the
    # appropriate phase latent heat (L_v for liquid surfaces, L_s
    # for frozen surfaces, mixed for snow-on-land).  Consumers
    # should use this field rather than back-deriving evaporation
    # from ``lhflx / L_v`` — the latter under-counts mass by ~13%
    # over any tile that sublimates rather than evaporates.
    # Audit F3.
    surface_mass_flux: jax.Array
    # Net salt mass flux delivered by this tile to the ocean
    # [kg(salt)/m²/s, positive = salt INTO ocean].  Note: ocean
    # *salinity* rises during freezing not because salt is added but
    # because water mass leaves (negative ``freshwater_flux``); the
    # absolute salt mass balance is what this channel tracks.
    # Sea-ice tiles report:
    #   * negative flux during freezing of new lead ice — a small
    #     amount of salt is locked into the new ice at ``S_lead_ice``
    #     (≈ 4 PSU), so the ocean column loses that salt mass.  Ocean
    #     salinity still rises because the water mass loss is much
    #     larger (classical "brine rejection" rise in S).
    #   * positive flux during basal / surface melt — ice salt at
    #     ``S_ice`` returns to the ocean column.
    #   * negative flux during snow-ice flooding — white ice traps
    #     pore-water salt at ``S_white_ice`` (≈ 17 PSU).
    # Tiles without a salt channel (atmosphere, land, lake) return
    # zeros.  Always a populated array — pytree-uniform with
    # ``freshwater_flux``.
    salt_flux: jax.Array


class SurfaceToAtm(NamedTuple):
    """Blended surface -> atmosphere coupling fields. Shape (6, n, n)."""
    T_surface: jax.Array
    albedo: jax.Array
    emissivity: jax.Array
    z0: jax.Array
    q_surface: jax.Array
    shflx: jax.Array
    lhflx: jax.Array
    tau_x: jax.Array
    tau_y: jax.Array
    lw_up: jax.Array
    u_ocean_sfc: jax.Array
    v_ocean_sfc: jax.Array
    co2_flux: jax.Array
    # Net freshwater flux from blended surface tiles to the receiving
    # body [kg/m²/s, positive into the ocean / surface].  Always a
    # populated array (zeros if no tile reports a freshwater channel).
    freshwater_flux: jax.Array
    # Tile-blended heat flux extracted from the ocean [W/m²,
    # positive = ocean LOSES energy to surface tiles].  See
    # ``TileResponse.ocean_heat_extraction``.
    ocean_heat_extraction: jax.Array
    # Tile-blended stress applied to the ocean surface [Pa,
    # positive = eastward / northward force on the ocean].  See
    # ``TileResponse.ocean_stress_x / y``.
    ocean_stress_x: jax.Array
    ocean_stress_y: jax.Array
    # Tile-blended phase-aware surface moisture mass flux
    # [kg/m²/s, positive up].  See ``TileResponse.surface_mass_flux``.
    surface_mass_flux: jax.Array
    # Tile-blended salt mass flux to ocean [kg(salt)/m²/s, positive
    # = INTO ocean].  See ``TileResponse.salt_flux``.
    salt_flux: jax.Array
