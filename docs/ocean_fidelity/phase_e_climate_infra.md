# Phase E — Climate-scale forcing infrastructure

## What landed

Five modules cover the prerequisites for Phase F multi-decade OMIP-2
runs + Bryan-style THC spinups:

### Forcing loaders

* `src/legoesm/ocean/forcing/jra55_do.py` -- JRA55-do reanalysis
  loader (Tsujino 2018/2020 OMIP-2 standard). Seven channels:
  ``u10``, ``v10``, ``T_air``, ``q_air``, ``sw_down``, ``lw_down``,
  ``precip``, ``runoff``. Caches each calendar year as a zarr store
  under ``$LEGOESM_CACHE/forcing/jra55_do/<year>.zarr``.
* `src/legoesm/ocean/forcing/core2.py` -- CORE-II Normal-Year Forcing
  (Large & Yeager 2009). Single perpetual-year ``OceanForcing`` for
  Bryan THC spinups.
* `src/legoesm/ocean/forcing/__init__.py` -- module facade exporting
  ``OceanForcing``, ``load_jra55_do``, ``load_core2_nyf``, and
  ``synthetic_ocean_forcing``.

Both loaders fall back to a deterministic synthetic climatology when
the on-disk cache is missing so the matrix smoke tests do not need
~50 GB of JRA55-do data. ``allow_synthetic=False`` raises
``FileNotFoundError`` to flag missing real data in production runs.

### Bulk-flux coupler

* `src/legoesm/ocean/bulk_flux_omip.py` -- Large & Yeager 2009 open-
  ocean bulk formulae:
  - ``large_yeager_cd(u10)`` -- wind-speed-dependent drag coefficient
    (``Cd * 1e3 = 2.7/u10 + 0.142 + 0.0764 u10``).
  - ``large_yeager_ch(T_air, T_sfc)`` -- two-regime transfer
    coefficient (1.46e-3 unstable, 1.18e-3 stable).
  - ``air_sea_fluxes(...)`` -- converts (u10, v10, T_air, q_air,
    T_sfc, q_sfc, rho_air) -> (tau_x, tau_y, shflx, lhflx).

  Reuses the ``simple_bulk_fluxes`` core in
  ``legoesm.coupler.bulk_flux`` so the legoESM single-coupler flux
  surface stays unified.

### Climate diagnostics

* `src/legoesm/ocean/diagnostics_climate.py`:
  - ``amoc_at_latitude(psi, lat, depth, target_lat=26.5)`` -> max
    of the MOC streamfunction at 26.5 deg N. Returns
    ``AmocResult(target_lat_deg, streamfunction_Sv, depth_of_max_m)``.
  - ``acc_transport(psi_barotropic, lat, drake_south=-65,
    drake_north=-45)`` -> Drake-passage zonal volume transport as
    ``max - min`` of the barotropic streamfunction inside the band.
  - ``sst_climatology_bias(sst_model, sst_ref, area, mask=None)`` ->
    area-weighted bias + RMSE vs a reference climatology.

  Acceptance bars (Phase F):
  - AMOC @ 26.5 deg N -- 15 +/- 3 Sv (Cunningham 2007 / RAPID).
  - ACC transport @ Drake -- 130 +/- 15 Sv (Donohue 2016).
  - SST bias -- < 1.5 deg C globally vs WOA.

## Tests

`tests/ocean/unit/test_phase_e_climate.py` -- **11 / 11 pass**:

* Synthetic forcing channel shapes + physical bounds.
* JRA55-do / CORE-II synthetic-fallback round-trip.
* ``load_jra55_do(..., allow_synthetic=False)`` raises on missing
  cache.
* L&Y drag increases with wind speed at the high-wind branch.
* L&Y heat coefficient picks ``unstable`` vs ``stable``.
* ``air_sea_fluxes`` sign conventions (drag opposes wind; SST > T_air
  -> heat into atmosphere; q_air < q_sfc -> evaporative cooling).
* AMOC max located at target latitude + depth.
* ACC transport = max - min of barotropic streamfunction band.
* SST bias zero when model matches reference.
* SST bias / RMSE pick up a uniform offset.

## Acceptance gate

All four pillars of the Phase E plan land:

1. JRA55-do loader (synthetic fallback + zarr real-data path).
2. CORE-II NYF loader (same interface).
3. Large & Yeager 2009 bulk-flux module.
4. AMOC / ACC / SST climate-diagnostics package.

Phase F (OMIP-2 + Bryan THC long runs) can now drive a multi-decade
forced ocean integration and diff the output against the published
ensembles via the climate-diagnostics package.
