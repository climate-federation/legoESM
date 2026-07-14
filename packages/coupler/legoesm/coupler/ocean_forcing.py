"""Sea-ice -> ocean surface-forcing mappers for forced (OMIP) and coupled runs.

Three public entry points:

* ``ice_ocean_forcing_from_ice_response(ice_resp, fracs)`` — the validated
  ICE-ONLY mapper (sign / conservation suite ``tests/unit/test_ice_ocean_two_way.py``)
  that turns a sea-ice ``TileResponse`` + resolved ``TileFractions`` into the
  ``(FreshwaterForcing, OceanSurfaceForcing)`` a prognostic ocean ingests.
* ``blend_omip_ice_ocean_forcing(...)`` — the ONE shared, mask-aware
  forced-ocean (OMIP) flux partitioning: scale the caller's UNMASKED
  full-cell open-ocean bulk forcing by ``f_open = 1 - A`` (heat, SW, stress,
  evaporation — ONE post-step concentration time level), add each ice->ocean
  exchange exactly once, keep precip/runoff full-cell, and populate the
  KPP/vmix freshwater-buoyancy channel.
* ``omip_sea_ice_surface_forcing(...)`` — the forced-ocean (OMIP) driver step:
  advance a slab sea-ice tile one step, then delegate to the shared blend,
  returning the BLENDED forcing + new ice state.

Promoted out of ``coupler/_future/`` (2026-06-30) for the OMIP runner's
prognostic sea-ice tile — the first forced-ocean use of the F11 ice->ocean
back-reaction (``scripts/run/run_omip.py --jra55-sea-ice``).  Kept OUT of
``coupler.py`` so the coupler core stays ocean-model-agnostic.

ICE-ONLY by construction: the mapper takes the RAW sea-ice ``TileResponse`` and
the resolved ``TileFractions`` and extracts ONLY the ice tile's contribution,
using the SAME blend weights ``blend_tiles`` applies (``f_water`` for the
freshwater / heat EXCHANGE channels, ``f_ice`` for the back-reaction STRESS).
It does NOT take the blended ``SurfaceToAtm`` (whose ``freshwater_flux`` is the
TOTAL of ocean P-E + ice melt + land runoff + lake P-E — using that as
``ice_fw`` would double-count the atmospheric terms).

Sign conventions (verified against ``tests/unit/test_ice_ocean_two_way.py``):

* ``ice_fw = f_water * ice_resp.freshwater_flux`` — positive INTO the ocean on
  melt (dilutes salinity), negative on freeze (extracts -> salinifies).
* ``q_net = -(f_water * ice_resp.ocean_heat_extraction)`` —
  ``ocean_heat_extraction`` is positive when the ocean LOSES heat to the ice
  base; ``q_net`` is positive INTO the ocean, so it is the negative.
* ``tau_x/tau_y = -(f_ice * ice_resp.ocean_stress_x/ocean_stress_y)`` —
  ``ocean_stress_*`` is the force ON the ocean, but the ocean consumer applies
  external ``tau`` with an atmosphere-convention flip (``ocean force = -tau``),
  so the negation makes the net applied force equal the intended on-ocean ice
  stress.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.coupling_fields import TileResponse
from legoesm.coupler.tile_fractions import TileFractions
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    physical_net_freshwater_flux,
)
from legoesm.ocean.state import OceanSurfaceForcing


def ice_ocean_forcing_from_ice_response(
    ice_resp: TileResponse,
    fracs: TileFractions,
):
    """Build the ICE-ONLY ``(FreshwaterForcing, OceanSurfaceForcing)`` from the
    raw sea-ice ``TileResponse`` and the resolved ``TileFractions``.

    The ice contribution to each cell uses ``blend_tiles``' weights:

    * freshwater / heat EXCHANGE are per-grid-cell and weighted by
      ``f_water = f_ocean + f_ice`` (F11);
    * the back-reaction STRESS is per-ice-tile and weighted by ``f_ice``.

    ``sw_down``/``freshwater`` on ``OceanSurfaceForcing``: ``sw_down`` is left
    ``None`` (shortwave is the driver's atmospheric concern); ``freshwater`` is
    set to ``ice_fw`` so the KPP boundary-layer scheme sees the ice-freshwater
    BUOYANCY flux (a DIFFERENT consumer than the salinity tendency, which reads
    ``FreshwaterForcing.ice_fw`` — not double counting).
    """
    f_water = fracs.f_ocean + fracs.f_ice
    f_ice = fracs.f_ice
    z = jnp.zeros_like(ice_resp.freshwater_flux)
    ice_fw = f_water * ice_resp.freshwater_flux
    freshwater = FreshwaterForcing(
        precip=z, evap=z, runoff=z,
        ice_fw=ice_fw,
        restoring=z,
    )
    surface_forcing = OceanSurfaceForcing(
        sw_down=None,
        q_net=-(f_water * ice_resp.ocean_heat_extraction),
        tau_x=-(f_ice * ice_resp.ocean_stress_x),
        tau_y=-(f_ice * ice_resp.ocean_stress_y),
        # Real brine salt-mass flux (kg/m2/s, +into ocean), per-grid-cell
        # (f_water weight like the other exchange channels).  Applied to top
        # salinity as a real salt source, distinct from the freshwater
        # virtual-salt dilution above (#F11).
        salt_flux=f_water * ice_resp.salt_flux,
        # KPP buoyancy channel (see docstring) — NOT double counting.
        freshwater=ice_fw,
    )
    return freshwater, surface_forcing


def blend_omip_ice_ocean_forcing(
    *,
    ice_resp: TileResponse,
    ice_concentration,
    open_ocean_sf: OceanSurfaceForcing,
    open_ocean_fw: FreshwaterForcing,
    ocean_mask=None,
):
    """Partition a FULL-CELL open-ocean forcing against a stepped sea-ice tile
    (the ONE shared, mask-aware OMIP forced-ocean partitioning).

    The caller supplies the UNMASKED full-cell open-ocean bulk forcing (built
    with NO ice attenuation) plus the sea-ice ``TileResponse`` and ONE ice
    concentration ``A`` — the POST-step concentration, used consistently for
    ALL four open-water channels (heat, SW, stress, evaporation).  With
    ``f_open = 1 - A`` (masked, see below):

    * OPEN-WATER channels scale by ``f_open`` ONCE: heat ``q_net``,
      penetrative ``sw_down``, wind stress ``tau_x``/``tau_y`` (atmospheric
      convention — the core applies the ``-tau`` ocean reaction), and
      evaporation.  Under full ice no direct atmospheric flux reaches the
      ocean (the ice intercepts it in its own surface energy balance; SW
      transmittance through ice is the ice model's concern, not re-added
      here).
    * ICE->ocean exchange enters EXACTLY ONCE via
      :func:`ice_ocean_forcing_from_ice_response` (TileResponse conventions
      verified in ``ice/sea_ice.py``: basal heat / melt-freeze freshwater /
      brine salt are PER-GRID-CELL — already concentration-weighted by the
      ice model — weighted here by ``f_water``; the ice-ocean stress is
      PER-ICE-TILE, weighted by ``f_ice = A``, and carries the same
      atmospheric-convention sign flip as the open stress).
    * Precipitation + runoff stay FULL-CELL (no-snow-reservoir policy: the
      slab ice stores no precip), and the ``restoring`` channel passes
      through untouched.

    ``ocean_mask`` (1 = ocean, 0 = land; ``None`` = all ocean): the mask
    gates ONLY the ice->ocean terms (``A -> A*mask`` so ``f_ice`` and the
    ``f_water = mask`` exchange weights vanish on land — a spurious land-ice
    budget never reaches the ocean).  The OPEN forcing passes through land
    cells UNCHANGED (``f_open = 1 - A*mask = 1`` there), bit-identical to
    the ice-free path — the ocean core's own land mask zeroes it, exactly as
    it does without ice.

    KPP/vmix buoyancy contract: the returned ``sf.freshwater`` carries the
    PHYSICAL net freshwater (P - E_open + R + ice_fw, EXCLUDING the numerical
    SSS-restoring channel) via
    :func:`legoesm.ocean.freshwater.physical_net_freshwater_flux`.  On the
    direct OMIP-forced latlon/tripole/MPAS paths this channel is consumed
    ONLY by the vertical-mixing closures (surface buoyancy + non-local
    transport); the freshwater MASS is applied exactly once by the caller
    via ``model.step(freshwater=fw)``.  The caller must NOT have pre-set
    ``open_ocean_sf.freshwater``/``salt_flux`` (raises — those channels are
    owned by this blend).

    Returns ``(FreshwaterForcing, OceanSurfaceForcing)``.
    """
    if open_ocean_sf.freshwater is not None:
        raise ValueError(
            "blend_omip_ice_ocean_forcing owns OceanSurfaceForcing.freshwater "
            "(the KPP buoyancy channel is rebuilt from the blended freshwater); "
            "pass open_ocean_sf with freshwater=None.")
    if open_ocean_sf.salt_flux is not None:
        raise ValueError(
            "blend_omip_ice_ocean_forcing owns OceanSurfaceForcing.salt_flux "
            "(the brine real-salt channel comes from the ice TileResponse and "
            "must be applied exactly once); pass open_ocean_sf with "
            "salt_flux=None.")

    A = jnp.clip(jnp.asarray(ice_concentration), 0.0, 1.0)
    z = jnp.zeros_like(A)
    if ocean_mask is not None:
        m = jnp.asarray(ocean_mask, dtype=A.dtype)
        A = A * m            # land carries no ice tile
        f_land = 1.0 - m
    else:
        f_land = z
    # Open-water scale: 1 on land (open forcing passes through; the core's
    # land mask owns it), 1 - A on wet cells.
    f_open = 1.0 - A
    # Tile fractions sum to 1: wet cell (f_open - 0) + A + 0; land 0 + 0 + 1.
    # => f_water = f_ocean + f_ice = mask: every ice EXCHANGE channel is
    # land-masked by the mapper's own weights (no ad-hoc extra factor).
    fracs = TileFractions(f_ocean=f_open - f_land, f_ice=A, f_land=f_land,
                          f_lake=z)
    fw_ice, sf_ice = ice_ocean_forcing_from_ice_response(ice_resp, fracs)

    # --- Blend surface energy / momentum: open-ocean fluxes act on f_open;
    #     the ice tile adds its ice->ocean exchange (each term once). ---
    def _blend(open_field, ice_field):
        # open_field may be None on the OceanSurfaceForcing struct.
        scaled = (open_field * f_open) if open_field is not None else 0.0
        return scaled + ice_field

    sw_down = (open_ocean_sf.sw_down * f_open
               if open_ocean_sf.sw_down is not None else None)

    # --- Blend freshwater: P + runoff over the whole cell; E over open ocean
    #     only; ice melt/freeze via ice_fw; restoring passes through. ---
    evap = (open_ocean_fw.evap * f_open
            if open_ocean_fw.evap is not None else z)
    fw = FreshwaterForcing(
        precip=open_ocean_fw.precip,
        evap=evap,
        runoff=open_ocean_fw.runoff,
        ice_fw=fw_ice.ice_fw,
        restoring=open_ocean_fw.restoring,
    )
    # _replace (not a fresh struct) so every untouched channel the caller set
    # (e.g. ``chl`` for the RGB shortwave penetration) passes through.
    sf = open_ocean_sf._replace(
        sw_down=sw_down,
        q_net=_blend(open_ocean_sf.q_net, sf_ice.q_net),
        tau_x=_blend(open_ocean_sf.tau_x, sf_ice.tau_x),
        tau_y=_blend(open_ocean_sf.tau_y, sf_ice.tau_y),
        salt_flux=sf_ice.salt_flux,           # brine real salt, applied once
        # KPP/vmix buoyancy channel (see docstring): the PHYSICAL net
        # freshwater the mass path applies once via step(freshwater=fw).
        freshwater=physical_net_freshwater_flux(fw),
    )
    return fw, sf


def omip_sea_ice_surface_forcing(
    *,
    ice_state,
    ice_config,
    atm,
    ocean_sst_K,
    open_ocean_sf: OceanSurfaceForcing,
    open_ocean_fw: FreshwaterForcing,
    dt: float,
    u_ocean=None,
    v_ocean=None,
    U_min: float = 1.0,
    grid=None,
    ocean_mask=None,
):
    """Advance a slab sea-ice tile one OMIP step and return the BLENDED
    ``(new_ice_state, FreshwaterForcing, OceanSurfaceForcing)`` for the ocean.

    Thin step-then-blend wrapper: advances ``legoesm.ice.step_sea_ice`` and
    delegates the flux partitioning to the ONE shared implementation,
    :func:`blend_omip_ice_ocean_forcing` (see its docstring for the
    ``f_open = 1 - A`` partitioning, the POST-step concentration time level,
    the land-mask semantics, and the KPP freshwater-buoyancy contract).

    Parameters
    ----------
    ice_state : SeaIceState | DynamicSeaIceState
        Sea-ice state at the start of the step (carried by the driver).
    ice_config : SeaIceConfig
        Static (compile-time) ice configuration.
    atm : AtmToSurface
        Atmospheric forcing this step (same struct the ocean bulk flux uses).
    ocean_sst_K : jnp.ndarray
        Ocean surface temperature [K] (NOT degC) for the basal heat exchange.
    open_ocean_sf, open_ocean_fw : OceanSurfaceForcing, FreshwaterForcing
        The full-cell (no-ice, UNATTENUATED) bulk-flux forcings the OMIP loop
        already builds (``freshwater``/``salt_flux`` unset — the blend owns
        those channels).
    dt : float
        Timestep [s].
    u_ocean, v_ocean : jnp.ndarray or None
        Ocean surface currents [m/s] (default zeros — the 1-deg forced-ocean
        approximation drops the small current correction).
    U_min : float
        Minimum wind-speed floor passed to the ice bulk flux [m/s].
    grid : grid or None
        Required only for ice dynamics / advection / ridging; a pure
        thermodynamic slab (``dynamics="none"``) accepts ``None``.
    ocean_mask : array or None
        1 = ocean, 0 = land (``None`` = all ocean, the legacy flat-bottom
        behaviour).  Gates the ice->ocean terms only; see the blend docstring.

    Returns
    -------
    (new_ice_state, FreshwaterForcing, OceanSurfaceForcing)
    """
    from legoesm.ice.sea_ice import step_sea_ice

    u_o = u_ocean if u_ocean is not None else jnp.zeros_like(ocean_sst_K)
    v_o = v_ocean if v_ocean is not None else jnp.zeros_like(ocean_sst_K)

    new_ice, ice_resp = step_sea_ice(
        ice_state, atm, ocean_sst_K, u_o, v_o, ice_config, U_min, dt, grid,
    )

    # POST-step ice concentration sets the open-ocean fraction — the ONE
    # documented concentration time level for heat / SW / stress / evap.
    conc = new_ice.concentration.data
    if conc.ndim > ocean_sst_K.ndim:      # multi-category: aggregate
        conc = jnp.sum(conc, axis=-1)
    fw, sf = blend_omip_ice_ocean_forcing(
        ice_resp=ice_resp,
        ice_concentration=conc,
        open_ocean_sf=open_ocean_sf,
        open_ocean_fw=open_ocean_fw,
        ocean_mask=ocean_mask,
    )
    return new_ice, fw, sf


__all__ = [
    "ice_ocean_forcing_from_ice_response",
    "blend_omip_ice_ocean_forcing",
    "omip_sea_ice_surface_forcing",
]
