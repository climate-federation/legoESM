"""Surface-data **producer** tools for legoESM (host-side, not traced).

This subpackage builds legoESM's geophysical surface boundary data from primary
sources (FAO HWSD v2.0 soil, later GIMMS LAI4g vegetation, ESA-CCI biomass, ...)
and writes a harmonized regular lat-lon ``legoesm_surfdata`` NetCDF.  That file
is then consumed by the **runtime** loader
:mod:`legoesm.land.global_surface_data`, which regrids it once to the model grid
and time-interpolates per step.

Producer vs consumer — the same split as ``training/era5_to_state.py`` (producer)
→ the runtime forcing readers (consumer):

  - **Producer (here):** expensive, run-once, host-only NumPy/pandas/xarray; reads
    1 km rasters and attribute databases; aggregates to a coarse regular grid.
    *Never* imported into the traced model or a ``lax.scan`` body.
  - **Consumer (``global_surface_data``):** cheap per-run regrid of the coarse
    file to the active model grid; pure-JAX per-step interpolation.

Modules:
  - :mod:`~legoesm.land.surface_data.raster` — ENVI/BIL raster reader (pure NumPy).
  - :mod:`~legoesm.land.surface_data.aggregate` — area-weighted fine→coarse block
    aggregation.
  - :mod:`~legoesm.land.surface_data.sources` — one module per primary dataset
    (``hwsd2`` first).
  - :mod:`~legoesm.land.surface_data.schema` — harmonized surfdata NetCDF schema
    + writer.
"""

from __future__ import annotations
