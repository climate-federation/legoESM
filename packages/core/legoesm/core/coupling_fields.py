"""Fixed-slot coupling field containers.

Physical fields are pre-allocated with fixed shape. The optional forcing height
distinguishes observed reference-height input from model-level input.
These NamedTuples define the strict interface between atmosphere and surface.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants


def lowest_level_height(T_lowest, p_half, p_full):
    """Hydrostatic lowest full-level height above the LOCAL surface [m]."""
    return (constants.R_d * T_lowest / constants.g
            * jnp.log(p_half[..., -1] / p_full[..., -1]))


def require_surface_radiation_aux(aux, *, radiation_active: bool,
                                  precip_active: bool, lane: str) -> None:
    """Raise when a coupled atmosphere lane advanced a segment but stashed no
    surface radiation / precipitation for the coupler to consume.

    Both coupled drivers (``CoupledESMDriver._build_atm_forcing`` and
    ``EarthSystemDriver._build_atm_forcing``) read ``held_sw_net_sfc`` /
    ``held_lw_net_sfc`` / ``seg_precip`` out of ``ModelDriver._carry_aux`` and
    historically fell back to ``zeros`` for anything absent.  Lanes that never
    write them therefore forced the ocean/land/ice tiles with ``sw_down=0`` and
    ``precip=0``: perpetual polar night (~-240 W/m^2 global-mean, ice-albedo
    runaway) plus an evaporation-only freshwater budget (unbounded
    salinification).  The failure is SILENT -- no NaN, no exception, and the
    reconstructed ``lw_down`` collapses to a plausible ``sigma*T_sfc**4``
    because the emissivity cancels exactly.  Per dispatch-hardening doctrine a
    silent wrong-physics fallback is worse than a hard failure.

    Gated on the atmosphere CONFIG, not on bare key presence: zero IS the
    physically correct forcing for a dry run or ``radiation="none"``, and
    ``_run_column`` stashes each key only when its source produced one
    (model_driver.py:5795-5802).  An unconditional "require all three keys"
    check would raise on those legitimate coupled runs.

    Tests ``aux.get(k) is None`` rather than ``k not in aux`` because
    ``_run_per_step`` writes the key unconditionally with a ``None`` value when
    its source is inactive (model_driver.py:9762); a present-but-None entry is
    just as unusable to the consumer as a missing one.

    Pure and JAX-free (dict lookups only) so it is directly unit testable with
    no driver construction and no model run.
    """
    missing = []
    if radiation_active:
        missing += [k for k in ("held_sw_net_sfc", "held_lw_net_sfc")
                    if aux.get(k) is None]
    if precip_active and aux.get("seg_precip") is None:
        missing.append("seg_precip")
    if not missing:
        return
    raise RuntimeError(
        f"coupled atmosphere lane {lane!r} advanced a segment but did not "
        f"stash {missing} into _carry_aux, while the atmosphere config has an "
        "active radiation / precipitation source. The surface would be forced "
        "with zero shortwave and/or zero precipitation (silent: no NaN, and "
        "the reconstructed lw_down is a plausible sigma*T_sfc**4). Populate "
        "the held fields in that lane, or run a lane that does (the compiled "
        "single-device lane).")


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
    # Appended for positional compatibility. None denotes observed forcing
    # at the land config reference height; model-level producers supply metres.
    z_lowest: jax.Array | None = None


class TileResponse(NamedTuple):
    """Per-tile surface response returned to the coupler.

    **Temporal semantics**: state fields (``T_sfc``, ``q_surface``,
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
    T_sfc: jax.Array         # [end-of-step] Surface skin temperature [K]
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
    # Radiative (emission-equivalent) skin temperature [K] for the LW boundary:
    # the temperature T such that ``emissivity * sigma * T_rad^4`` equals THIS
    # tile's actual upward LW emission.  For most tiles this is just the skin
    # temperature, so it defaults to ``None`` and the blender falls back to
    # ``T_sfc``.  The two-leaf canopy is the exception: it emits with the canopy
    # column temperature ``surface_out.T_surface`` (so ``eps_col*sigma*T_rad^4 =
    # LW_emit``) while reporting ``T_sfc = T_soil`` for the (linear) sensible-heat
    # path — so it MUST set ``T_rad`` explicitly, else the tile blend would use
    # the soil temperature and break LW conservation for vegetated cells.
    T_rad: jax.Array | None = None
    # Sea-ice tiles only: the AGGREGATE ice concentration [0-1] the
    # THERMODYNAMICS integrated its atmospheric fluxes over (post-transport,
    # pre-thermo — under transport='advect' the ice moves BEFORE the thermo,
    # so the pre-call concentration is not the flux time level).  Forced-ocean
    # drivers partition the open-water atmospheric forcing with (1 - A) at
    # THIS time level so ice + open water together receive exactly the
    # incident flux (codex r5 #1).  ``None`` for every non-ice tile and for
    # the slab-ice paths (consumers fall back to the pre-call concentration).
    ice_concentration_thermo: jax.Array | None = None
    # Sea-ice tile only: the REALIZED latent heat paired with
    # ``surface_mass_flux`` on that field's per-water-cell basis, i.e. the
    # sum over categories of L_s(T_k) * m_k [W/m2, positive up].  ``lhflx``
    # stays per ice area (on the max(pre, post) concentration basis), so the
    # tile blend cannot rebuild the cell latent from it once L_s varies with
    # the category temperature; blend_tiles weights this by f_water exactly
    # like the mass.  None for every other tile (their lhflx is already the
    # latent paired with their mass flux, per tile area).
    lhflx_exchange: jax.Array | None = None


class SurfaceToAtm(NamedTuple):
    """Blended surface -> atmosphere coupling fields. Shape (6, n, n)."""
    T_sfc: jax.Array
    albedo: jax.Array
    emissivity: jax.Array
    # Radiative-equivalent surface temperature for the LW boundary.  Area-
    # averaging T_sfc and emissivity INDEPENDENTLY does not conserve the upward
    # LW flux of mixed land/ocean/ice cells (T^4 and eps*T^4 are nonlinear), so
    # the tile blender derives T_rad from the area-weighted EMISSION FLUX:
    # T_rad = (sum_i f_i*eps_i*sigma*T_i^4 / (eps_grid*sigma))^(1/4), with
    # eps_grid = sum_i f_i*eps_i (= ``emissivity``).  The atmosphere LW boundary
    # eps_grid*sigma*T_rad^4 + (1-eps_grid)*La then equals sum_i f_i*lw_up_i
    # EXACTLY.  Equals T_sfc for single-tile (pure) cells.  ``T_sfc`` stays the
    # area-weighted skin temperature for the (linear) sensible-heat / diagnostics
    # paths; only radiation uses T_rad.
    T_rad: jax.Array
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
    # River runoff sub-component of ``freshwater_flux`` [kg/m²/s, positive
    # = INTO ocean] = ``f_land * land.freshwater_flux`` only.  Kept SEPARATE
    # because a dynamic ocean depth-spreads river runoff over the top
    # ``runoff_depth_spread_m`` metres (NEMO ``rn_dep_max``) while ocean P-E,
    # ice melt and lake P-E stay at the top cell.  It is a subset of
    # ``freshwater_flux`` (NOT additive) — the surface (top-cell) freshwater is
    # ``freshwater_flux - river_runoff_flux``.  Zero for tiles/runs without land.
    # APPENDED at the struct end so the positional pytree ABI of the prior 19
    # fields is unchanged.
    river_runoff_flux: jax.Array
    # Surface (top-cell) NON-ocean freshwater sub-component of ``freshwater_flux``
    # [kg/m²/s, +INTO ocean] = ice melt/freeze (f_water-weighted, F11) + lake P-E
    # (f_lake-weighted).  Kept SEPARATE from ``river_runoff_flux`` because these
    # are surface fluxes (NOT depth-spread) and SEPARATE from the ocean P-E so a
    # consumer can keep ocean P-E CURRENT (it depends on the current SST) while
    # lagging the land/ice/lake exchange one coupling sub-step.  Together
    # ``river_runoff_flux + ice_lake_freshwater_flux`` is the full non-ocean
    # freshwater = ``freshwater_flux - f_ocean*(ocean P-E)``; both are zero for an
    # aquaplanet (no land/ice/lake tile).
    ice_lake_freshwater_flux: jax.Array
