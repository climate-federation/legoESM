# OMIP-2 30-year forced ocean run -- results skeleton

This document is the **acceptance template** for the Phase F
production OMIP-2 run. The smoke path (one model day at 36x72) is
verified locally and runs in ~1 s; the production run at 1 deg
(180x360, MPAS 100 km) drives 30 years of JRA55-do forcing and needs
cluster compute. Fill in the table cells below with the actual values
once the cluster run finishes.

## Setup

* Driver: ``scripts/ocean_long_runs/run_omip2.py``
* Grids: lat-lon C-grid @ 1 deg (180x360) + MPAS @ 100 km
* Forcing: JRA55-do (Tsujino 2018) 1958-present, 3-hourly
* Bulk fluxes: Large & Yeager 2009 (``bulk_flux_omip.py``)
* Years: 30 (1980-2010 default)
* Diagnostics per year: RPE, energy budget, tracer budget, AMOC @
  26.5 N, ACC @ Drake, SST bias vs WOA (optional)
* Restart cadence: every model year (``restart_year_NNNN.npz``)

## Acceptance bars (Tsujino et al. 2020 multi-model ensemble)

| metric | acceptance | rationale |
|---|---|---|
| AMOC @ 26.5 deg N | 15 +/- 3 Sv | Cunningham 2007 / RAPID; Tsujino 2020 ensemble |
| ACC @ Drake | 130 +/- 15 Sv | Donohue 2016 transport mooring |
| SST bias vs WOA | < 1.5 deg C globally | Tsujino 2020 evaluation |
| RPE drift | < 0.5 mW/m^2 | Petersen 2015 (MPAS-Ocean 0.1-0.3 mW/m^2) |
| Volume drift | < 1e-10 / year | tracer-conservation reference |
| Heat-content drift | < 1 W/m^2 after 100 yr | Bryan 1987 |

## Cluster fill-in table (placeholder)

Replace each ``...`` with the actual production value:

| year | AMOC [Sv] | ACC [Sv] | SST bias [K] | RPE flux [mW/m^2] |
|---|---|---|---|---|
|   1 | ... | ... | ... | ... |
|   5 | ... | ... | ... | ... |
|  10 | ... | ... | ... | ... |
|  20 | ... | ... | ... | ... |
|  30 | ... | ... | ... | ... |

## Smoke confirmation (local)

```
==> Building global rest-state on latlon/36x72
   RPE_0 = -7.7476e+25 J  |  KE_0 = 0.000e+00  |  vol_0 = 2.764e+18
==> Year 1/1 (48 steps)
   RPE_flux = -3.683e-02 W/m^2  |  KE = 0.000e+00  |  vol_drift = 0.000e+00
Wall time: 1.1s
```

Smoke run is a sanity check that the driver:
* builds the grid + model on the requested resolution,
* applies forcing via the JRA55-do loader (synthetic fallback),
* steps the model through ``model.step``,
* computes the RPE / energy / tracer diagnostics each year,
* writes the per-year restart .npz,
* dumps the ``summary.json`` payload that the production run will fill
  in for 30 years.

## Known open items before production cluster run

* **Surface-flux applicator** -- the smoke driver computes
  ``(tau_x, tau_y, shflx, lhflx)`` per timestep but does not yet apply
  them to the state's surface boundary fields. The plumbing lives in
  ``legoesm/ocean/physics/surface_forcing/prescribed.py`` (used by
  the gyre runners) and the OMIP-2 driver needs a thin coupler hook
  that translates the bulk-flux output to the state's
  ``surface_taux`` / ``surface_tauy`` / ``forc_temp_surface`` etc.
  This is a follow-up commit; the present driver stops short of the
  end-to-end forcing so the climate-scale acceptance gate is still
  cluster work, not local-laptop work.
* **JRA55-do regridding** -- the smoke path applies a global-mean
  forcing to every cell. Production needs the
  ``legoesm.grids.conservative_regrid`` weights baked once + cached
  per (forcing grid, model grid) pair.
* **WOA SST climatology** -- ``compute_sst_bias`` needs a WOA
  reference passed in. Add a thin loader under
  ``legoesm/ocean/forcing/woa.py`` (Phase F.4 follow-up).
