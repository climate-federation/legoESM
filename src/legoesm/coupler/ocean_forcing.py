"""Assemble PROGNOSTIC-ocean surface forcing from the SEA-ICE tile's ice->ocean
exchange (F11 two-way ice<->ocean coupling).

A prognostic ocean ingests surface forcing through ``OceanSurfaceForcing``
(heat / wind stress) and ``FreshwaterForcing`` (the ``ice_fw`` melt/freeze
channel -> virtual salt flux).  This maps the sea-ice tile's ice->ocean
exchange into those structs so a coupled driver can close the ICE -> OCEAN
feedback prognostically (salinity freshens on melt, currents respond to ice
stress, SST responds to basal heat) instead of dropping it (which the
slab-ocean driver does intentionally).

ICE-ONLY by construction: it takes the RAW sea-ice ``TileResponse`` and the
resolved ``TileFractions`` and extracts ONLY the ice tile's contribution, using
the SAME blend weights ``blend_tiles`` applies (``f_water`` for the freshwater /
heat EXCHANGE channels, ``f_ice`` for the back-reaction STRESS).  It does NOT
take the blended ``SurfaceToAtm`` (whose ``freshwater_flux`` is the TOTAL of
ocean P-E + ice melt + land runoff + lake P-E — using that as ``ice_fw`` would
double-count the atmospheric terms).  A driver adds the atmosphere/ocean/land/
lake surface forcing separately and merges.

SCOPE: ``LatLonCGridOceanModel`` (its ``step`` takes a separate ``freshwater=``
argument that drives salinity via the virtual-salt-flux closure, and applies
``OceanSurfaceForcing.tau_x/tau_y/q_net`` unconditionally).  NOT wired for the
cubed-sphere ``OceanModel`` (``step`` has no ``freshwater=`` arg -> ``ice_fw``
would be dropped) nor MPAS (external ``tau``/``q_net`` are applied only inside
its OPTIONAL physics function, so they can be silently ignored depending on
``MPASOceanConfig.physics``).  Kept OUT of ``coupler.py`` so the coupler core
stays ocean-model-agnostic.

NOTE on salt: the ice melt/freeze FRESHWATER (``ice_fw``) drives salinity via
the ocean's virtual-salt-flux closure — melt dilutes, freeze concentrates — so
the dominant brine effect is captured.  The EXPLICIT sea-ice ``salt_flux``
(real brine-rejection salt mass) is NOT separately wired here; a real ocean
salt-mass source would be needed for that refinement.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.coupler.coupling_fields import TileResponse
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

    Sign conventions:

    * ``ice_fw = f_water * ice_resp.freshwater_flux`` — positive INTO the ocean
      on melt (dilutes salinity), negative on freeze (extracts -> salinifies).
    * ``q_net = -(f_water * ice_resp.ocean_heat_extraction)`` —
      ``ocean_heat_extraction`` is positive when the ocean LOSES heat to the ice
      base; ``q_net`` is positive INTO the ocean, so it is the negative.
    * ``tau_x/tau_y = -(f_ice * ice_resp.ocean_stress_x/ocean_stress_y)`` —
      ``ocean_stress_*`` is the force ON the ocean, but the ocean consumer
      applies external ``tau`` with an atmosphere-convention flip
      (``ocean force = -tau``; see ``ocean_pe_latlon_cgrid.py`` /
      ``mpas_physics.py``), so the negation makes the net applied force equal the
      intended on-ocean ice stress.

    ``sw_down``/``freshwater`` on ``OceanSurfaceForcing`` are left ``None``
    (shortwave is the driver's atmospheric concern; salinity freshwater is
    delivered via ``FreshwaterForcing.ice_fw``, the path the ocean salinity
    tendency reads — no double counting).
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
        # Also expose the net ice freshwater on the surface-forcing channel:
        # the KPP boundary-layer scheme derives its salt-BUOYANCY flux (melt
        # stabilizes, freeze/brine destabilizes) from surface_forcing.freshwater
        # — a DIFFERENT consumer than the salinity tendency, which reads the
        # FreshwaterForcing.ice_fw above via virtual_salt_flux — so setting both
        # is NOT double counting (verified: k_profiles.py / integration.py read
        # surface_forcing.freshwater only for buoyancy; salinity reads the
        # freshwater= arg).  Without this, KPP would see zero ice freshwater
        # buoyancy while salinity still changed.
        freshwater=ice_fw,
    )
    return freshwater, surface_forcing
