"""Sea-ice -> ocean surface-forcing mappers for forced (OMIP) and coupled runs.

Three public entry points:

* ``ice_ocean_forcing_from_ice_response(ice_resp, fracs)`` — the validated
  ICE-ONLY mapper (sign / conservation suite ``tests/unit/test_ice_ocean_two_way.py``)
  that turns a sea-ice ``TileResponse`` + resolved ``TileFractions`` into the
  ``(FreshwaterForcing, OceanSurfaceForcing)`` a prognostic ocean ingests.
* ``blend_ice_ocean_forcing(...)`` — the ONE shared, mask-aware open-water /
  ice partition: scales the caller's full-cell open-ocean forcing by
  ``f_open = 1 - A`` (stress, evaporation, heat/SW), adds the ice->ocean
  exchange exactly once, and sets the KPP freshwater-buoyancy channel.
* ``omip_sea_ice_surface_forcing(...)`` — the forced-ocean (OMIP) driver step:
  advance a slab sea-ice tile one step and hand the partition to
  ``blend_ice_ocean_forcing``, returning the BLENDED forcing + new ice state.

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

# --- under-ice shortwave transmittance (Grenfell & Maykut 1977 / NEMO
#     fr_sw-under-ice order of magnitude; same default as
#     ocean.coupler.omip2_applicator._ice_surface_heat / --ice-thermo-sw-trans) ---
_SW_TRANSMITTANCE_ICE = 0.03    # [-] fraction of SW penetrating ice+snow to ocean


def add_frazil_ice(ice_state, ice_config, ice_mass_kg_m2):
    """Deposit frazil ice exported by the ocean column into the slab ice tile.

    ``ice_mass_kg_m2`` is :func:`legoesm.ocean.physics.frazil.apply_frazil`'s
    surface export (kg of ice per unit cell area, >= 0).  The ocean side has
    already removed the liquid mass, rejected the salt at
    ``ice_config.S_ice_new`` and kept the latent heat, so the tile only gains
    volume, with the CICE ``add_new_ice`` convention the lead-freeze path in
    :mod:`legoesm.ice.sea_ice` uses: new ice fills lead area at the nominal
    thickness ``h_new_ice`` (capped at the open-water fraction) and the mean
    thickness follows from volume conservation.  Single-category slab only.
    """
    h, A = ice_state.h_ice.data, ice_state.concentration.data
    V_new = jnp.maximum(ice_mass_kg_m2, 0.0) / ice_config.rho_ice
    dA = jnp.minimum(V_new / ice_config.h_new_ice, jnp.maximum(1.0 - A, 0.0))
    A_new = jnp.clip(A + dA, 0.0, 1.0)
    V_new_total = h * A + V_new
    h_new = jnp.where(A_new > 0.0, V_new_total / jnp.maximum(A_new, 1e-12), 0.0)
    return ice_state._replace(
        h_ice=ice_state.h_ice.replace(data=h_new),
        concentration=ice_state.concentration.replace(data=A_new),
    )


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


def blend_ice_ocean_forcing(
    *,
    open_sf: OceanSurfaceForcing,
    open_fw: FreshwaterForcing | None,
    ice_resp: TileResponse,
    ice_concentration,
    ocean_mask=None,
    sw_partition: str = "prescaled",
    alpha_ocean: float | None = None,
    sw_transmittance_ice: float = _SW_TRANSMITTANCE_ICE,
    ice_owns_snow_reservoir: bool = False,
):
    """Partition a full-cell open-ocean forcing between the open-water fraction
    ``f_open = 1 - A`` and the sea-ice tile, and add the ice->ocean exchange —
    the ONE shared, mask-aware implementation of the OMIP ice/ocean forcing
    split (used by :func:`omip_sea_ice_surface_forcing` and the CORE-II runner's
    prognostic-ice path; do not re-derive the weights elsewhere).

    ``ice_concentration`` (``A``) is a SINGLE concentration time level chosen by
    the caller and used consistently for heat, SW, stress, AND evaporation.
    The CORE-II runner passes the PRE-``step_sea_ice`` concentration — the
    state the ice integrated its atmospheric fluxes over, so ice + open water
    together receive exactly the incident flux (``A + (1-A) = 1``; codex r4);
    ``omip_sea_ice_surface_forcing`` keeps its original post-step contract.
    Partition:

    * open-ocean stress, evaporation, and heat/SW scale with ``f_open = 1 - A``;
    * ice basal heat, brine salt, melt/freeze freshwater, and ice stress are
      each added exactly ONCE, via :func:`ice_ocean_forcing_from_ice_response`
      (its validated F11 sign conventions: ``tau_ice = -A*ocean_stress`` so the
      core's ``-tau`` consumer applies ``+A*stress`` on the ocean);
    * precipitation and runoff stay full-cell inputs UNLESS the ice model owns a
      snow reservoir (``ice_owns_snow_reservoir=True``, i.e. ``uses_new_physics``
      / the ``_thermo_v2`` path): a snow-reservoir ice model routes its
      ice-fraction rain/snow to the ocean via ``ice_resp.freshwater_flux`` ->
      ``ice_fw``, so the DIRECT precip channel must then carry only the
      open-water share ``f_open*P`` to avoid counting the ``A*P`` ice-fraction
      precip twice (once full-cell here, once via ``ice_fw``).  For a
      reservoir-less slab / legacy-dynamic model (default False) precip stays
      full-cell and ``ice_fw`` carries melt/freeze only, so P is counted once.
      This matches the coupled-ESM gate (``coupled_esm_driver`` uses_new_physics
      -> ``f_ocean*P``).  ``runoff`` is always full-cell; only ``evap`` (and now,
      when gated, ``precip``) is rescaled in ``open_fw``.

    ``ocean_mask`` (1 = ocean, 0 = land; ``None`` = all ocean) zeroes the ice
    concentration AND every ice->ocean channel on land cells, so a spurious
    land-ice budget never reaches the ocean and land keeps the (irrelevant,
    core-masked) full-cell open forcing.

    ``sw_partition`` selects the shortwave/heat convention of ``open_sf``:

    * ``"prescaled"`` — ``open_sf.sw_down``/``q_net`` are already the FINAL
      open-water values (any albedo applied by the caller): both simply scale
      by ``f_open`` (the original ``omip_sea_ice_surface_forcing`` behaviour).
    * ``"raw_core2"`` — ``open_sf`` is the UNMASKED CORE-II bulk forcing built
      with ``ice_albedo=None`` (``sw_down`` = raw downwelling SW, ``q_net`` =
      ``q_non_sw + sw_down``).  The open-water/under-ice SW split is applied
      HERE (and only here — the caller must NOT also attenuate in
      ``compute_omip2_surface_forcing``), reproducing ``_ice_surface_heat``'s
      under-ice partition bit-exactly::

          sw_ocean = sw_down * (f_open*(1 - alpha_ocean) + A*sw_transmittance_ice)
          q_net    = sw_ocean + f_open*(q_net_open - sw_down) + q_ice

      ``alpha_ocean`` is required in this mode (pass
      ``constants.alpha_ocean_broadband``).

    KPP freshwater-buoyancy contract: when ``open_fw`` is given, the returned
    ``sf.freshwater`` is set to ``net_freshwater_flux(blended fw)`` — the
    PHYSICAL ``P - E + R + ice_fw`` signal the vertical-mixing surface-buoyancy
    diagnosis reads.  This is a BUOYANCY-ONLY channel on the direct-forced OMIP
    paths (surface-forcing scheme ``"none"``): the freshwater MASS/salinity is
    applied exactly once via ``model.step(freshwater=fw)``, never from
    ``sf.freshwater``.  Callers running the ``"external"`` surface-forcing
    scheme must NOT use this helper's ``sf`` (it would double-apply — the
    CORE-II runner guards this at setup).  When ``open_fw`` is ``None`` the
    channel carries the masked ice freshwater alone.

    Returns ``(fw, sf)`` (blended; ``fw`` is ``None`` iff ``open_fw`` was).
    """
    from legoesm.ocean.freshwater import net_freshwater_flux

    # Ownership contract: this blend OWNS sf.freshwater and sf.salt_flux (it
    # assembles both below).  A caller pre-setting either would be silently
    # overwritten — hiding a double application or discarding another
    # salt/freshwater source — so raise loudly instead (trace-time structural
    # check; every production caller builds sf with both channels unset).
    if open_sf.freshwater is not None:
        raise ValueError(
            "blend_ice_ocean_forcing owns OceanSurfaceForcing.freshwater "
            "(the KPP surface-buoyancy channel); the caller must pass it "
            "unset (None) — a preset value would be silently overwritten.")
    if open_sf.salt_flux is not None:
        raise ValueError(
            "blend_ice_ocean_forcing owns OceanSurfaceForcing.salt_flux "
            "(the ice brine-rejection channel); the caller must pass it "
            "unset (None) — a preset value would be silently overwritten.")

    A_raw = jnp.clip(jnp.asarray(ice_concentration), 0.0, 1.0)
    if ocean_mask is None:
        m = jnp.ones_like(A_raw)
    else:
        m = jnp.asarray(ocean_mask, dtype=A_raw.dtype)
    wet = m > 0.0
    # jnp.where, NOT multiply-by-mask: a NaN concentration / ice flux in a dry
    # cell survives `NaN * 0` (and jnp.clip passes NaN through), so masking by
    # multiplication would poison f_open and every blended field (codex r2 #2).
    A = jnp.where(wet, A_raw, 0.0)    # land: A -> 0 (no partition, no ice flux)
    f_open = 1.0 - A
    z = jnp.zeros_like(A)

    fracs = TileFractions(f_ocean=f_open, f_ice=A, f_land=z, f_lake=z)
    fw_ice, sf_ice = ice_ocean_forcing_from_ice_response(ice_resp, fracs)
    # The exchange channels are weighted by f_water = f_open + A = 1 even on
    # land — mask them explicitly (the stress is already ∝ A = 0 on land, but
    # a dry-cell NaN would survive the multiply, hence jnp.where).
    ice_fw = jnp.where(wet, fw_ice.ice_fw, 0.0)
    q_ice = jnp.where(wet, sf_ice.q_net, 0.0)
    salt_ice = jnp.where(wet, sf_ice.salt_flux, 0.0)
    tau_x_ice = jnp.where(wet, sf_ice.tau_x, 0.0)
    tau_y_ice = jnp.where(wet, sf_ice.tau_y, 0.0)

    if sw_partition == "prescaled":
        sw = (open_sf.sw_down * f_open
              if open_sf.sw_down is not None else None)
        q_open_part = (open_sf.q_net * f_open
                       if open_sf.q_net is not None else 0.0)
        q = q_open_part + q_ice
    elif sw_partition == "raw_core2":
        if alpha_ocean is None:
            raise ValueError(
                "blend_ice_ocean_forcing: sw_partition='raw_core2' requires "
                "alpha_ocean (pass constants.alpha_ocean_broadband)")
        swd = open_sf.sw_down
        # open q_net was assembled as q_non_sw + sw_down (no albedo) — recover.
        q_non_sw = open_sf.q_net - swd
        sw = swd * (f_open * (1.0 - float(alpha_ocean))
                    + A * float(sw_transmittance_ice))
        q = sw + f_open * q_non_sw + q_ice
    else:
        raise ValueError(
            f"blend_ice_ocean_forcing: unknown sw_partition {sw_partition!r} "
            "(expected 'prescaled' or 'raw_core2')")

    tau_x = ((open_sf.tau_x * f_open if open_sf.tau_x is not None else 0.0)
             + tau_x_ice)
    tau_y = ((open_sf.tau_y * f_open if open_sf.tau_y is not None else 0.0)
             + tau_y_ice)

    if open_fw is not None:
        # No-snow-reservoir policy: full-cell precip reaches the ocean directly.
        # But when the ice model OWNS a snow reservoir (v2 / uses_new_physics ->
        # the _thermo_v2 path), it delivers the ice-fraction rain/snow to the
        # ocean via ice_fw (= f_water*ice_resp.freshwater_flux), so the DIRECT
        # channel must carry only the open-water share f_open*P (kg/m2/s, +into
        # ocean; f_open in [0,1] is a pure fraction -> positive scaling, no sign
        # flip) or the A*P ice-fraction precip is counted twice.  Mirrors
        # coupled_esm_driver's f_ocean*P gate.
        precip = (
            open_fw.precip * f_open
            if ice_owns_snow_reservoir else open_fw.precip
        )
        fw = FreshwaterForcing(
            precip=precip,
            evap=(open_fw.evap * f_open if open_fw.evap is not None else z),
            runoff=open_fw.runoff,
            ice_fw=ice_fw,
            restoring=open_fw.restoring,
        )
        # KPP buoyancy channel = PHYSICAL net freshwater ONLY: the numerical
        # SSS-restoring virtual flux stays on the mass channel (fw.restoring,
        # applied once by the core) but is EXCLUDED from the boundary-layer
        # buoyancy signal (codex r1 #3 — a restoring correction is not a
        # physical surface buoyancy flux).
        kpp_freshwater = net_freshwater_flux(fw._replace(restoring=None))
    else:
        fw = None
        kpp_freshwater = ice_fw

    sf = open_sf._replace(
        sw_down=sw,
        q_net=q,
        tau_x=tau_x,
        tau_y=tau_y,
        salt_flux=salt_ice,
        freshwater=kpp_freshwater,
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

    Forced-ocean (OMIP) flux partitioning between the open-ocean fraction
    ``f_ocean = 1 - A`` and the ice fraction ``A`` (post-step concentration):

    * Heat / wind stress / penetrative SW from the atmosphere act only on the
      OPEN-OCEAN fraction (the ice intercepts them over its own area, handling
      them in its surface energy balance), so the caller's full-cell
      ``open_ocean_sf`` is scaled by ``f_ocean``.  The ice tile then ADDS its
      ice->ocean exchange (basal heat, melt/freeze freshwater, brine salt,
      ice-ocean stress) via :func:`ice_ocean_forcing_from_ice_response`.
    * Freshwater: runoff reaches the ocean over the whole cell; evaporation acts
      only on the open-ocean fraction (``f_ocean``); ice melt/freeze enters via
      ``ice_fw``.  Precipitation reaches the ocean full-cell for a reservoir-less
      ice model (slab / legacy-dynamic: no snow reservoir, ``ice_fw`` carries no
      precip), but only over the open-water fraction ``f_ocean*P`` when the
      configured ice model owns a snow reservoir (``uses_new_physics`` / the
      ``_thermo_v2`` path), which then delivers the ``A*P`` ice-fraction precip
      itself via ``ice_fw`` -- see :func:`blend_ice_ocean_forcing`'s
      ``ice_owns_snow_reservoir`` gate (so precip is counted exactly once).
    * ``sf.freshwater`` carries the PHYSICAL net freshwater ``P - E + R +
      ice_fw`` as the KPP surface-buoyancy signal (buoyancy-only under the
      direct-forced scheme ``"none"``; the mass is applied once via the
      ``model.step(freshwater=fw)`` channel — see
      :func:`blend_ice_ocean_forcing`).

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
    ocean_mask : array or None
        1 = ocean, 0 = land; ``None`` = all ocean (the flat-bottom / aquaplanet
        legacy behaviour).  Threaded to :func:`blend_ice_ocean_forcing`, which
        zeroes the ice concentration and every ice->ocean channel on land
        cells.  Callers whose domain has land (the JRA55 latlon-bathy /
        tripole lanes in ``run_omip.py``) MUST pass their land mask, or
        spurious land-cell ice budgets reach the blend boundary.

    Returns
    -------
    (new_ice_state, FreshwaterForcing, OceanSurfaceForcing)
    """
    from legoesm.ice import uses_new_physics
    from legoesm.ice.sea_ice import step_sea_ice

    u_o = u_ocean if u_ocean is not None else jnp.zeros_like(ocean_sst_K)
    v_o = v_ocean if v_ocean is not None else jnp.zeros_like(ocean_sst_K)

    new_ice, ice_resp = step_sea_ice(
        ice_state, atm, ocean_sst_K, u_o, v_o, ice_config, U_min, dt, grid,
    )

    # Post-step ice concentration sets the open-ocean fraction (the ONE
    # documented concentration time level for heat / SW / stress / evap).
    # ocean_mask defaults to None (all ocean) for flat-bottom callers;
    # land-bearing lanes thread their land mask through.  The caller's sf
    # carries FINAL open-water values -> sw_partition="prescaled".  The
    # partition weights live ONLY in blend_ice_ocean_forcing.
    fw, sf = blend_ice_ocean_forcing(
        open_sf=open_ocean_sf,
        open_fw=open_ocean_fw,
        ice_resp=ice_resp,
        ice_concentration=new_ice.concentration.data,
        ocean_mask=ocean_mask,
        sw_partition="prescaled",
        ice_owns_snow_reservoir=uses_new_physics(ice_config),
    )
    return new_ice, fw, sf


__all__ = [
    "blend_ice_ocean_forcing",
    "ice_ocean_forcing_from_ice_response",
    "omip_sea_ice_surface_forcing",
]
