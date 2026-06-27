"""**NOT YET WIRED INTO ANY PRODUCTION DRIVER** (F11 two-way ice<->ocean).

This module is the VALIDATED ice->ocean forcing mapper (see the extensive sign /
conservation test suite ``tests/unit/test_ice_ocean_two_way.py``) for the
two-way sea-ice <-> prognostic-ocean back-reaction.  It has no production caller:
``CoupledESMDriver`` runs an aquaplanet Phase-1 dynamic ocean with NO sea-ice
tile (ice concentration == 0 everywhere), so this mapper would return all-zeros
there and its non-zero ice path cannot be exercised / sign-validated by any
in-scope coupled run.  It is parked here (mirroring ``ice/_future/
bitz_lipscomb.py``) until an ice-coupled dynamic-ocean configuration exists.

Wire it into ``CoupledESMDriver._assemble_ocean_forcing`` by:

1. surfacing the RAW ice ``TileResponse`` + the resolved ``TileFractions`` from
   ``coupler.step_surface`` (which today returns only the blended
   ``SurfaceToAtm``) -- e.g. an optional extra return or a stashed attribute;
2. calling ``ice_ocean_forcing_from_ice_response(ice_resp, fracs)`` and ADDING
   its ``q_net`` (basal heat extraction), ``tau_x/tau_y`` (ice stress
   back-reaction), ``salt_flux`` (brine) and ``ice_fw`` (melt/freeze freshwater)
   to the ocean-tile forcing already assembled there -- remapped onto the ocean
   grid (``_grid_remapper.a2o``) like the other lagged sub-channels;
3. verifying each exchanged flux carries the SAME sign at BOTH ends (basal heat
   leaving the ocean == entering the ice; brine +into ocean on freeze; ice
   stress on the ocean == -tau atmosphere convention) against an ICE-COUPLED
   validation run -- the verification this audit could not complete.

The detailed per-ocean-model consumption contract and sign conventions follow.

Assemble PROGNOSTIC-ocean surface forcing from the SEA-ICE tile's ice->ocean
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

SCOPE — the returned structs are model-agnostic; consumption differs per grid:
  - ``LatLonCGridOceanModel``: applies ``OceanSurfaceForcing.tau/q_net/salt``
    directly in its dynamics + the separate ``freshwater=`` arg for the salinity
    virtual-salt closure (pass BOTH returned structs).
  - cubed-sphere ``OceanModel``: routes surface forcing through ``physics_fn`` —
    configure
    ``OceanPhysicsConfig(surface_forcing=SurfaceForcingConfig(scheme="external"))``
    so the ``external`` scheme applies the SAME ``OceanSurfaceForcing``
    (tau / q_net / freshwater / salt).  Here ``freshwater`` is a VIRTUAL salt
    flux (salinity dilution, fixed volume) — no ``eta`` mass source yet (deferred
    with MPAS); pass only the ``OceanSurfaceForcing`` (no ``freshwater=`` arg).
  - ``MPASOceanModel``: pass BOTH structs to ``step(freshwater=, surface_forcing=)``.
    The ``freshwater=`` arg drives eta + virtual-salt salinity; the
    ``surface_forcing`` drives tau / q_net AND the real ``salt_flux`` through
    ``make_mpas_ocean_physics`` (its own external block — enabled by
    ``SurfaceForcingConfig(scheme="external")`` or ``"none"``).  The external
    block does NOT read ``surface_forcing.freshwater`` (the net salinity-
    freshwater injection is the ``freshwater=`` arg's job).  KPP, if enabled,
    DOES read ``surface_forcing.freshwater`` — but only for buoyancy and a
    NON-LOCAL salinity redistribution whose column integral is zero (no net
    surface salt), so it does not double-count the ``freshwater=`` virtual salt.
The returned ``OceanSurfaceForcing`` carries tau/q_net/freshwater/salt and the
``FreshwaterForcing`` carries ``ice_fw``; pass whichever the chosen model
consumes.  Kept OUT of ``coupler.py`` so the coupler core stays
ocean-model-agnostic.

Salt: BOTH channels are wired.  The ice melt/freeze FRESHWATER (``ice_fw``)
drives salinity via the ocean's virtual-salt-flux closure (melt dilutes, freeze
concentrates), AND the explicit sea-ice ``salt_flux`` (real brine-rejection
salt mass) is delivered through ``OceanSurfaceForcing.salt_flux`` -> a real
top-layer salt source (``salt_flux_salinity_tendency``).  An ocean model must
apply ``OceanSurfaceForcing.salt_flux`` for the latter to take effect (the
lat-lon C-grid does; see ``ocean_pe_latlon_cgrid.py``).
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
        # Real brine salt-mass flux (kg/m2/s, +into ocean on freeze / melt),
        # per-grid-cell (f_water weight like the other exchange channels).
        # Applied to top-layer salinity as a real salt source, distinct from the
        # freshwater virtual-salt dilution above (#F11).
        salt_flux=f_water * ice_resp.salt_flux,
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
