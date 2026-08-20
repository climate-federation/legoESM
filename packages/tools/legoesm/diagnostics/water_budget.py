"""Global water-conservation residual diagnostics for the coupled ESM.

These are DIAGNOSTIC-ONLY residuals: host-side, off the differentiated
segment loss, and NEVER fed back into physics.  They are built from the
ACTUAL per-tile conserved-water flows (each tile's real freshwater flux /
storage), respecting that land and ice have INTERNAL storage, so a coupler
water-routing bug shows up as a non-zero residual early rather than as a
silent multi-year drift.

WHY NOT A NAIVE ``atm bottom flux == sum_i f_i*(P-E)`` PARTITION
-----------------------------------------------------------------
The LAND tile does not return P-E (its ``freshwater_flux`` is RUNOFF only)
and neither land nor ice is storage-free, so over land/ice
``P - E - outflow = d(storage)/dt != 0`` on any timestep.  A residual that
assumes per-tile P-E balance false-flags every land/ice cell and is useless.
The helpers here instead compare either the atmosphere's OWN moisture drain
(``atm_moisture_residual`` -- an atmosphere-only budget, immune to land/ice
storage) or a conserved interface integral (``runoff_conservation_residual``)
-- neither assumes tile P-E balance.

THESE ARE TRIPWIRES, NOT MACHINE-ZERO IDENTITIES.  ``atm_moisture_residual``
uses the coupler-BLENDED surface moisture flux (not the atmosphere's own
internal evaporation) and samples CWV at segment boundaries, so a correct run
sits at a small floor (bounded by the ~1e-4 kg/m^2/s hydrological rate) far
below a sensible ~1e-3 threshold, while a mis-scaled precip overshoots it by
~1e6x.  ``runoff_conservation_residual`` differences two integrals the caller
supplies; when the caller RECONSTRUCTS the ocean wet-mask gating, a small
cross-grid conservative-remap residual / fractional-coast wet fraction sits at
its floor and the balanced all-wet case is exactly zero.

MULTI-RANK REDUCTION
--------------------
Every area-integral SUM in these helpers is routed through
``global_sum_if_distributed`` (the only AD-safe collective, ``allreduce(SUM)``
with full VJP): a no-op returning the rank-local value when serial (BYTE-
IDENTICAL), an ``allreduce(SUM)`` when a shard/Voronoi decomposition is armed.
So a SHARDED coupled run gets a globally-closed residual -- advective moisture
divergence moves water across rank boundaries, so a per-rank budget would not
close.  This is CONTINGENT on each field being a DISJOINT INTERIOR shard (no
replication -> N-count, no halo inclusion -> boundary double-count) over a
shared ``COMM_WORLD`` rank set; certified by the MPI gate
``tests/distributed/test_water_budget_global_reduction_mpi.py`` (np1 residual
== np2 residual on a small coupled config), never by a login-node run.
``global_max_mpi``/``global_min_mpi`` are not AD-safe and are deliberately not
used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.diagnostics.energy_budget import area_weighted_mean
from legoesm.parallel.reductions import (
    global_sum_if_distributed,
    is_multi_process,
)

from legoesm import constants


def area_integral(field: jax.Array, area: jax.Array) -> jax.Array:
    """Area integral ``sum_i field_i * area_i``, globally reduced when sharded.

    Units are ``[field-units] * m^2`` (e.g. a ``kg/m^2/s`` flux integrates to
    ``kg/s``).  The local partial sum is routed through
    :func:`legoesm.parallel.reductions.global_sum_if_distributed` -- the only
    AD-safe reduction (``allreduce(SUM)`` with full VJP).  A SINGLE-process run
    pays no reduction and the value is BYTE-IDENTICAL to the rank-local integral;
    a SHARDED coupled run gets the true GLOBAL integral so the water budget
    closes across ranks (advective moisture divergence crosses rank boundaries).
    Correctness under sharding REQUIRES ``field``/``area`` to be the DISJOINT
    INTERIOR shard on their grid -- no replication (would N-count) and no halo
    inclusion (would double-count boundary cells), the same owned-shard contract
    every ``global_sum_if_distributed`` caller relies on.
    """
    return global_sum_if_distributed(jnp.sum(field * area))


def _area_weighted_mean_maybe_global(field, area):
    """``area_weighted_mean`` reduced across shards when distributed.

    SERIAL (or ``area is None`` / horizontal-shape-mismatch): delegates to
    :func:`area_weighted_mean` so the result is BYTE-IDENTICAL to the rank-local
    diagnostic.  DISTRIBUTED well-formed WEIGHTED case: forms the mean from the
    GLOBALLY reduced ``sum(field*area)`` and ``sum(area)`` (AD-safe
    ``allreduce(SUM)``) so the closed-sphere moisture budget closes across ranks.
    ``area_weighted_mean`` itself is LEFT UNTOUCHED -- it is shared far beyond the
    water budget (``sst_mean``, ML losses, bias metrics); wrapping it in place
    would silently make those collective (out of scope).  NOTE the DEGENERATE
    distributed ``area is None`` / shape-mismatch branch delegates to the LOCAL
    unweighted ``jnp.mean`` and therefore stays RANK-LOCAL (not globally
    reduced) -- acceptable only because a real sharded coupled call always
    passes the atm ``grid_area`` matching the field's horizontal shape (the
    weighted branch); tripwire A is global ONLY on the weighted branch.
    """
    if (not is_multi_process() or area is None
            or field.ndim < area.ndim
            or tuple(field.shape[:area.ndim]) != tuple(area.shape)):
        return area_weighted_mean(field, area)
    if field.ndim > area.ndim:
        field = jnp.mean(field, axis=tuple(range(area.ndim, field.ndim)))
    num = global_sum_if_distributed(jnp.sum(field * area))
    den = global_sum_if_distributed(jnp.sum(area))
    return num / den


def atm_moisture_residual(cwv_now, cwv_prev, evap, precip, area, dt_s):
    """Global atmospheric moisture-budget residual [kg/m^2/s], RANK-LOCAL.

    Closed-sphere moisture conservation integrates the horizontal moisture
    divergence to zero, so the area-mean column-water-vapour tendency must
    equal the area-mean net surface moisture source ``E - P``::

        R_atm = d/dt <CWV> - (<E> - <P>)

    with ``E = evap`` the surface evaporation (+up, moistening the atmosphere)
    and ``P = precip`` the precipitation rate (+down, drying it).
    ``area_weighted_mean`` supplies the area weighting (``area=None`` falls
    back to an unweighted mean).  ``<CWV>`` is an INDEPENDENT witness of the
    atmosphere's true precip drain, computed from ``q_v`` and unaffected by
    the coupler's packaged precip export -- so a coupler that hands the
    surface a precip rate O(1e5) too large (a segment ACCUMULATION mis-
    delivered as a RATE) inflates ``<P>`` to ~14 kg/m^2/s, ~1e6x the ~3.5e-5
    kg/m^2/s physical hydrological rate, and ``|R_atm|`` jumps far above any
    sane threshold.

    THIS IS A TRIPWIRE, NOT A MACHINE-ZERO IDENTITY: ``evap`` is the coupler-
    BLENDED surface moisture flux, which need not equal the atmosphere's own
    internal evaporation source to machine precision, and CWV is sampled at
    segment boundaries.  The mismatch is bounded by the hydrological rate
    (<~1e-4 kg/m^2/s), far below a sensible tripwire threshold (~1e-3
    kg/m^2/s), so a correct-physics ``R_atm`` sits near that small floor while
    a mis-scaled precip overshoots it by ~1e6x.

    It detects an inflation in the coupler-DELIVERED precip (the drained
    ``q_v`` still reflects the true per-step precip); it would be blind only
    to a hypothetical bug that also mis-scaled the atmosphere's OWN q_v precip
    sink (then CWV would move with the inflated precip).
    """
    dcwv_dt = (_area_weighted_mean_maybe_global(cwv_now, area)
               - _area_weighted_mean_maybe_global(cwv_prev, area)) / dt_s
    e_mean = _area_weighted_mean_maybe_global(evap, area)
    p_mean = _area_weighted_mean_maybe_global(precip, area)
    return dcwv_dt - (e_mean - p_mean)


def runoff_conservation_residual(river_atm, area_atm,
                                 runoff_applied_ocean, area_ocean):
    """River-runoff remap/wet-mask conservation residual [kg/s], RANK-LOCAL.

    The conservative atm->ocean remap preserves the runoff INTEGRAL; the
    ocean wet mask then drops runoff that landed on fully-dry ocean cells
    (interior-land runoff with no river-routing map).  That discarded water
    is a silent freshwater leak::

        R_runoff = INT_atm(river) - INT_ocean(applied)

    ``river_atm`` is the exported land river-runoff flux on the ATM grid
    (``SurfaceToAtm.river_runoff_flux``, +into ocean); ``runoff_applied_ocean``
    is the runoff surviving the ocean wet mask on the OCEAN grid.  Both are
    area-INTEGRATED on their own grids because conservative remap conserves
    the integral, not the mean.  Zero for a correct routing; equals the
    discarded runoff otherwise.  This is a TRIPWIRE: if the caller supplies
    a RECONSTRUCTED wet-mask gating (``river*wet_mask``) rather than the ocean
    core's literally-applied value, a small cross-grid conservative-remap
    residual / fractional-coast wet fraction sits at its floor, while the
    balanced all-wet case is exactly zero.
    """
    return (area_integral(river_atm, area_atm)
            - area_integral(runoff_applied_ocean, area_ocean))


def ice_water_content(h_ice, concentration, cell_water_fraction, *,
                      h_snow=None):
    """Whole-cell frozen-water inventory [kg/m^2] (tripwire-C store term).

    W_ice = rho_ice * <sum_cat h_ice*conc> * f_water
            [+ rho_snow * <sum_cat h_snow*conc> * f_water]

    ``concentration`` is the sea-ice-tile state variable: the ice fraction of
    the WATER (ocean) area, NOT of the whole grid cell -- so the whole-cell ice
    area fraction is ``cell_water_fraction * concentration`` (matches
    ``coupler.tile_fractions.compute_tile_fractions``: ``f_ice = f_water*conc``,
    ``f_water = 1 - f_land - f_lake``).  A trailing ITD category axis
    (``concentration.ndim > cell_water_fraction.ndim``) is summed so multi-
    category ice collapses to the per-cell volume.  ``h_ice``/``h_snow`` are
    per-ice-area thicknesses [m]; densities are ``constants.rho_ice`` /
    ``constants.rho_snow``.  Slab ``SeaIceState`` carries no snow reservoir ->
    pass ``h_snow=None``.  Pure; rank-local by construction (the caller
    area-integrates on the atm grid).
    """
    _extra = concentration.ndim - cell_water_fraction.ndim

    def _cell_volume(x):
        prod = x * concentration
        if _extra > 0:
            prod = jnp.sum(prod, axis=tuple(range(-_extra, 0)))
        return prod

    w = constants.rho_ice * _cell_volume(h_ice)
    if h_snow is not None:
        w = w + constants.rho_snow * _cell_volume(h_snow)
    return w * cell_water_fraction


def land_water_content_slab(w_bucket, snow_depth, land_fraction):
    """Whole-cell slab-land water inventory [kg/m^2] (tripwire-C store term).

    ``LandState.W_bucket`` and ``.snow_depth`` are already [kg/m^2] PER LAND
    AREA, so the whole-cell store is ``f_land * (W_bucket + snow_depth)``.  The
    ``land_fraction`` weight is essential: without it a coastal (partial-land)
    cell's storage tendency would not cancel the ``f_land*P`` precip it actually
    received, false-flagging every water-storing coastal cell.  Pure.
    """
    return land_fraction * (w_bucket + snow_depth)


def land_water_content_multilayer(theta_soil, dz, snow_depth, land_fraction, *,
                                  surface_water=None):
    """Whole-cell multilayer-land water inventory [kg/m^2] (tripwire-C term).

    W_land = f_land * ( sum_k theta_soil_k*dz_k*rho_w + snow_depth
                        + surface_water*rho_w )

    ``theta_soil`` [m^3/m^3] has shape ``(ncol, n_soil_layers)``; ``dz`` [m] is
    the ``make_soil_grid(config.soil_grid).dz`` layer-thickness vector
    ``(n_soil_layers,)``; ``snow_depth`` [kg/m^2] and ``surface_water`` [m]
    (ponding; ``None`` -> 0) are ``(ncol,)``.  ``rho_w = constants.rho_water``.
    Mirrors the extractable-water integral in ``multilayer_land.py`` (theta*dz
    summed over layers, x rho_w) but uses TOTAL theta_soil for storage.  The
    ``f_land`` weight is essential (see ``land_water_content_slab``).  Pure.
    """
    soil_col = jnp.sum(
        theta_soil * jnp.asarray(dz)[None, :], axis=-1) * constants.rho_water
    w = soil_col + snow_depth
    if surface_water is not None:
        w = w + surface_water * constants.rho_water
    return land_fraction * w


def water_inventory_residual(store_integral_now, store_integral_prev,
                             f_ocean_applied_integral, dt_s):
    """Closed global water-inventory tendency residual [kg/s], RANK-LOCAL.

        R_water = d/dt INT[W_atm + W_land + W_ice + W_lake] dA
                  + INT[F_ocean_applied] dA

    ``store_integral_now``/``_prev`` are the area-INTEGRATED atm-grid water
    inventory [kg] (vapour[+condensate] + whole-cell land + whole-cell ice;
    lake=0) at consecutive diagnostic segment boundaries.
    ``f_ocean_applied_integral`` [kg/s] is the ocean-grid integral of the
    freshwater the OCEAN MODEL actually applied POST wet-mask,
    ``INT(precip) - INT(evap) + INT(runoff_applied) + INT(ice_fw)``, with
    +into ocean = water LEAVING the tracked inventory.

    Because storage tendencies are IN the budget, a land/ice cell with a
    legitimate ``P - E - outflow != 0`` CANCELS (no false land/ice-storage
    flag).  The H2 class (frozen precip destroyed on the ice fraction) shows up
    because ``W_ice`` fails to rise by ``INT(f_ice*P)`` and that water reaches
    neither ``F_ocean`` nor another store -> ``R_water != 0``.

    TRIPWIRE, NOT a machine-zero identity.  Unlike ``atm_moisture_residual``
    (an atmosphere-only budget immune to land/ice storage), this CLOSED
    inventory couples every tile's storage tendency to its blended surface
    flux, so its floor sums each tile's evap-weighting / area-remap consistency
    gap plus the segment-boundary sampling (``F_ocean`` sampled at the last
    coupling sub-step vs the segment-mean store tendency) and the one-substep
    ice_fw lag -- a larger floor than A/B.  Choose any hard threshold well
    ABOVE that floor and below the H2 signal (~INT(f_ice*P)); as an emitted
    diagnostic it needs no threshold.  RANK-LOCAL: a sharded coupled run MUST
    route the underlying SUMs through ``global_sum_mpi`` before this is
    globally meaningful -- advective moisture divergence crosses rank
    boundaries, so a per-rank inventory does not close (hence the ``_ranklocal``
    key suffix).  DYNAMIC-OCEAN ONLY: ``F_ocean_applied`` is assembled only on
    the dynamic-ocean path; the storage-free slab ocean is a heat-only sink and
    admits no closed water inventory.
    """
    return (store_integral_now - store_integral_prev) / dt_s \
        + f_ocean_applied_integral
