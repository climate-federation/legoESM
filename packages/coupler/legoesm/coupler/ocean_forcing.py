"""Sea-ice -> ocean surface-forcing mappers for forced (OMIP) and coupled runs.

Two public entry points:

* ``ice_ocean_forcing_from_ice_response(ice_resp, fracs)`` — the validated
  ICE-ONLY mapper (sign / conservation suite ``tests/unit/test_ice_ocean_two_way.py``)
  that turns a sea-ice ``TileResponse`` + resolved ``TileFractions`` into the
  ``(FreshwaterForcing, OceanSurfaceForcing)`` a prognostic ocean ingests.
* ``omip_sea_ice_surface_forcing(...)`` — the forced-ocean (OMIP) driver step:
  advance a slab sea-ice tile one step and partition the surface forcing
  between the open-ocean fraction ``f_ocean = 1 - A`` (the caller's full-cell
  bulk-flux forcing) and the ice tile (basal heat, melt/freeze freshwater,
  brine salt, ice-ocean stress), returning the BLENDED forcing + new ice state.

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
from legoesm.ocean.freshwater import FreshwaterForcing
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
):
    """Advance a slab sea-ice tile one OMIP step and return the BLENDED
    ``(new_ice_state, FreshwaterForcing, OceanSurfaceForcing)`` for the ocean.

    Forced-ocean (OMIP) flux partitioning between the open-ocean fraction
    ``f_ocean = 1 - A`` and the ice fraction ``A`` (post-step concentration):

    * Heat / wind stress / penetrative SW from the atmosphere act only on the
      OPEN-OCEAN fraction (the ice intercepts them over its own area, handling
      them in its surface energy balance), so the caller's full-cell
      ``open_ocean_sf`` is scaled by ``f_ocean``.  The ice tile then ADDS its
      ice->ocean exchange (basal heat, melt/freeze freshwater, brine salt,
      ice-ocean stress) via :func:`ice_ocean_forcing_from_ice_response`.
    * Freshwater: precipitation + runoff reach the ocean over the whole cell
      (this slab ice has no snow reservoir — ``SnowConfig`` off by default — so
      precip is not stored on ice); evaporation acts only on the open-ocean
      fraction (``f_ocean``); ice melt/freeze enters via ``ice_fw``.

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
        The full-cell (no-ice) bulk-flux forcings the OMIP loop already builds.
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

    # Post-step ice concentration sets the open-ocean fraction.  No land/lake
    # in an OMIP ocean-only run: f_water = 1, f_ice = A, f_ocean = 1 - A.
    A = jnp.clip(new_ice.concentration.data, 0.0, 1.0)
    f_ocean = 1.0 - A
    z = jnp.zeros_like(A)
    fracs = TileFractions(f_ocean=f_ocean, f_ice=A, f_land=z, f_lake=z)
    fw_ice, sf_ice = ice_ocean_forcing_from_ice_response(ice_resp, fracs)

    # --- Blend surface energy / momentum: open-ocean fluxes act on (1-A);
    #     the ice tile adds its ice->ocean exchange. ---
    def _blend(open_field, ice_field):
        # open_field may be None on the OceanSurfaceForcing struct.
        scaled = (open_field * f_ocean) if open_field is not None else 0.0
        return scaled + ice_field

    sw_down = (open_ocean_sf.sw_down * f_ocean
               if open_ocean_sf.sw_down is not None else None)
    sf = OceanSurfaceForcing(
        sw_down=sw_down,
        q_net=_blend(open_ocean_sf.q_net, sf_ice.q_net),
        tau_x=_blend(open_ocean_sf.tau_x, sf_ice.tau_x),
        tau_y=_blend(open_ocean_sf.tau_y, sf_ice.tau_y),
        salt_flux=sf_ice.salt_flux,           # open ocean has no salt flux
        freshwater=sf_ice.freshwater,         # KPP ice-buoyancy channel
    )

    # --- Blend freshwater: P + runoff over the whole cell; E over open ocean
    #     only; ice melt/freeze via ice_fw. ---
    evap = (open_ocean_fw.evap * f_ocean
            if open_ocean_fw.evap is not None else z)
    fw = FreshwaterForcing(
        precip=open_ocean_fw.precip,
        evap=evap,
        runoff=open_ocean_fw.runoff,
        ice_fw=fw_ice.ice_fw,
        restoring=open_ocean_fw.restoring,
    )
    return new_ice, fw, sf


__all__ = [
    "ice_ocean_forcing_from_ice_response",
    "omip_sea_ice_surface_forcing",
]
