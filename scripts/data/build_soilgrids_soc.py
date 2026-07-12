#!/usr/bin/env python
"""Build a gridded soil-organic-carbon (SOC) OBSERVATION field on the global
carbon initial-condition grid, for the per-cell / zonal / per-biome bias
validation of the model SOC map (``scripts/validate/carbon_soc_vs_gridded.py``).

Motivation
----------
The CUE-fixed global carbon IC (``results/global_carbon_ic_cuefix2/global_carbon_ic.npz``)
is physically realistic in STRUCTURE (corr(NPP,SOC)>0, corr(MAT,SOC)<0) but has only
been validated against a single global-mean SOC (~11 kgC/m2).  A single mean cannot say
WHERE the model is biased (uniform? high-latitude peat? tropics?).  This builder produces
a gridded SOC obs field -- on the SAME grid as the IC -- so the validator can compute a
per-cell, area-weighted bias, a zonal-mean profile, and a per-biome table.

Two obs products, written to the SAME NetCDF schema (dims ``(lat, lon)``, var ``soc``
[kgC/m2], on the IC grid) so the validator is product-agnostic:

* ``--product surfdata_organic`` (DEFAULT; depth-matched; no network).  The CLM5 surfdata
  ``ORGANIC`` soil-organic-MATTER density [kg/m3] -- an IGBP / soil-database-derived
  gridded product (Lawrence & Slater 2008 soil organic column, the SAME field the
  Stage-B SOC calibration loss targets via
  :func:`legoesm.land.carbon.soc_observations.column_soc`) -- vertically integrated over
  the MODEL soil column (``make_soil_grid`` with the IC's own ``n_layers`` / ``soil_depth``
  / growth factor) and scaled organic-matter -> organic-carbon by ``--om-to-oc``
  (van Bemmelen 0.58 gC/gOM).  DEPTH-CONSISTENT with the model's full-column prognostic
  SOM, so the bias it reports is the same quantity the calibration targets.

* ``--product soilgrids`` (external, independent; ISRIC SoilGrids 2.0).  A 0-100 cm SOC
  STOCK assembled from the ``soc`` (content), ``bdod`` (bulk density) and ``cfvo`` (coarse
  fragments) coverages at the 5 standard depth intervals, fetched from the ISRIC WCS in
  EPSG:4326 and CONSERVATIVELY regridded (``conservative_regrid_latlon``) to the IC grid.
  DEPTH CAVEAT: SoilGrids reaches 100 cm (max 200 cm), shallower than the model's 3 m
  column, so a SoilGrids-vs-model comparison UNDER-credits the model's deep carbon; a
  "model low" finding there is conservative.  Provided as the independent external
  cross-check requested; the depth-matched surfdata_organic product is the headline.

Units (kgC/m2, no gC/m2 vs kgC/m2 ambiguity)
--------------------------------------------
Both products write kgC/m2.  The model SOM in the IC npz is stored gC/m2 (the validator
divides by 1000); this builder emits kgC/m2 so the validator compares like with like.

Login-node policy
-----------------
The networked SoilGrids WCS fetch AND any code path importing ``legoesm`` (constants /
regridding pull JAX) must run on a COMPUTE node (sbatch / srun), never the login node.
The pure reduction / unit / assembly helpers are offline-unit-tested.
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np

# --- SoilGrids 2.0 mapped-unit conventions (ISRIC; https://www.isric.org/
#     explore/soilgrids/faq-soilgrids -> "Available soil properties") ----------
# soc  mapped in dg/kg     -> g/kg          (x 0.1)
# bdod mapped in cg/cm3    -> kg/m3         (x 10 ; 1 cg/cm3 = 0.01 g/cm3 = 10 kg/m3)
# cfvo mapped in cm3/dm3   -> volume frac   (/ 1000)
_SOC_MAP_TO_G_PER_KG = 0.1
_BDOD_MAP_TO_KG_PER_M3 = 10.0
_CFVO_MAP_TO_FRACTION = 1.0 / 1000.0
_G_TO_KG = 1.0e-3

# SoilGrids standard depth intervals for a 0-100 cm stock: (coverage token, dz [m]).
_SOILGRIDS_DEPTHS: tuple[tuple[str, float], ...] = (
    ("0-5cm", 0.05),
    ("5-15cm", 0.10),
    ("15-30cm", 0.15),
    ("30-60cm", 0.30),
    ("60-100cm", 0.40),
)

# Organic-CARBON per organic-MATTER mass fraction (van Bemmelen 1/1.724 = 0.58); the
# raw CLM5 surfdata ORGANIC field is organic MATTER (its units attribute declares
# 0.58 gC/gOM).  Matches scripts/run/train_carbon_params.py::_VAN_BEMMELEN_OC_PER_OM
# so the obs here is carbon-consistent with the Stage-B SOC calibration target.
_VAN_BEMMELEN_OC_PER_OM = 0.58

_OUT_VAR = "soc"

# ISRIC SoilGrids 2.0 WCS (public, no auth).
_WCS_HOST = "https://maps.isric.org"
_WCS_CRS_4326 = "http://www.opengis.net/def/crs/EPSG/0/4326"


# ===========================================================================
# Pure reductions / unit conversions (offline-unit-tested)
# ===========================================================================
def soilgrids_layer_stock(soc_mapped, bdod_mapped, cfvo_mapped, dz_m):
    """Per-layer SOC STOCK [kgC/m2] from SoilGrids mapped ``soc``/``bdod``/``cfvo``.

    ``stock = SOC_content[gC/kg] * bulk_density[kg/m3] * (1 - coarse_frac) * dz[m]``,
    converted gC->kgC.  NaN in any input propagates to NaN (missing soil).  Coarse
    fragments REDUCE the fine-earth stock, so the ``(1 - cfvo)`` factor is <= 1.
    """
    soc = np.asarray(soc_mapped, dtype=np.float64) * _SOC_MAP_TO_G_PER_KG      # gC/kg
    bd = np.asarray(bdod_mapped, dtype=np.float64) * _BDOD_MAP_TO_KG_PER_M3    # kg/m3
    cfvo = np.asarray(cfvo_mapped, dtype=np.float64) * _CFVO_MAP_TO_FRACTION   # frac
    fine_earth = np.clip(1.0 - cfvo, 0.0, 1.0)
    stock_g_per_m2 = soc * bd * fine_earth * float(dz_m)                       # gC/m2
    return stock_g_per_m2 * _G_TO_KG                                           # kgC/m2


def assemble_ocs_0_100(layer_stocks):
    """Sum a list of per-layer stocks [kgC/m2] -> 0-100 cm stock, NaN-aware.

    A cell is NaN only where EVERY depth layer is missing; otherwise the finite
    layers are summed (a partially-observed column is not discarded).
    """
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in layer_stocks], axis=0)
    all_missing = np.all(~np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    return np.where(all_missing, np.nan, total)


def integrate_organic_column(organic, dz, om_to_oc):
    """Column SOC [kgC/m2] from per-layer organic-matter density and thicknesses.

    ``organic`` ``(..., n_layer)`` [kg OM/m3], ``dz`` ``(n_layer,)`` [m].  Reuses the
    shared :func:`legoesm.land.carbon.soc_observations.column_soc` (no re-derived
    vertical integral), then scales organic MATTER -> organic CARBON by ``om_to_oc``.
    """
    from legoesm.land.carbon.soc_observations import column_soc

    col_om = column_soc(np.asarray(organic, dtype=np.float64), np.asarray(dz, dtype=np.float64))
    return col_om * float(om_to_oc)


def regrid_to_target(field2d, src_lat, src_lon, tgt_lat, tgt_lon):
    """Conservatively regrid a ``(nlat, nlon)`` obs field to the IC grid.

    Thin wrapper over the shared, NaN-aware, area-conserving
    :func:`legoesm.grids.regridding.conservative_regrid_latlon`; when the source
    and target centres coincide (surfdata already on the IC grid) it is the
    identity, which also exercises the regrid path as a self-consistency check.
    """
    from legoesm.grids.regridding import conservative_regrid_latlon

    return conservative_regrid_latlon(
        np.asarray(field2d, dtype=np.float64),
        np.asarray(src_lat, dtype=np.float64), np.asarray(src_lon, dtype=np.float64),
        np.asarray(tgt_lat, dtype=np.float64), np.asarray(tgt_lon, dtype=np.float64))


# ===========================================================================
# IC grid + surfdata reading
# ===========================================================================
def read_ic_grid(ic_npz_path):
    """1-D ascending ``(lat, lon)`` [deg] and soil geometry from the IC npz.

    Returns ``(lat, lon, n_layers, soil_depth)`` where lat/lon are the UNIQUE
    sorted centres of the flattened IC grid (the target of the regrid) and the
    soil geometry drives the depth-matched surfdata integration.
    """
    d = np.load(ic_npz_path, allow_pickle=True)
    lat = np.unique(np.asarray(d["lat"], dtype=np.float64))
    lon = np.unique(np.asarray(d["lon"], dtype=np.float64))
    n_layers = int(d["n_layers"]) if "n_layers" in d else 10
    soil_depth = float(d["soil_depth"]) if "soil_depth" in d else 3.0
    return lat, lon, n_layers, soil_depth


def model_soil_dz(n_layers, soil_depth):
    """Model soil-layer thicknesses ``dz`` ``(n_layers,)`` [m] via the SHARED
    :func:`legoesm.land.soil_grid.make_soil_grid` (same geometric grid the carbon
    equilibration uses -- ``_SOIL_GROWTH_FACTOR`` from ``carbon.global_init``), so
    the obs column integrates over the SAME depth the modelled SOM represents.

    Deferred import: ``make_soil_grid`` pulls JAX -> compute-node only.
    """
    from legoesm.land.carbon.global_init import _SOIL_GROWTH_FACTOR
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid

    grid = make_soil_grid(SoilGridConfig(
        n_layers=int(n_layers), total_depth=float(soil_depth),
        growth_factor=_SOIL_GROWTH_FACTOR))
    return np.asarray(grid.dz, dtype=np.float64)


def read_surfdata_organic(surf_path):
    """CLM5 surfdata ``ORGANIC`` ``(nlat, nlon, nlev)`` [kg OM/m3] + 1-D lat/lon.

    Reads the canonical ``LATIXY``/``LONGXY`` 2-D coordinate arrays (reduced to 1-D
    centres, the SAME reduction ``read_clm5_cover_veg`` uses) and moves the level
    axis last so the column integral is over the trailing axis.  Ocean cells carry
    ORGANIC 0 (no fill) -- kept as 0 and excluded downstream by the land mask.
    """
    import xarray as xr

    ds = xr.open_dataset(surf_path, decode_times=False)
    try:
        var = next((v for v in ("ORGANIC", "organic") if v in ds), None)
        if var is None:
            raise SystemExit(
                f"surfdata {surf_path} has no ORGANIC/organic variable "
                f"(available: {sorted(ds.data_vars)[:30]}...)")
        org = np.asarray(ds[var].values, dtype=np.float64)          # (nlev, nlat, nlon)
        lat = np.asarray(ds["LATIXY"].values, dtype=np.float64)[:, 0]
        lon = np.asarray(ds["LONGXY"].values, dtype=np.float64)[0, :]
    finally:
        ds.close()
    if org.ndim != 3:
        raise SystemExit(f"ORGANIC must be (nlev, nlat, nlon); got {org.shape}.")
    return np.moveaxis(org, 0, -1), lat, lon                        # (nlat, nlon, nlev)


def build_surfdata_obs(surf_path, tgt_lat, tgt_lon, *, n_layers, soil_depth, om_to_oc):
    """Depth-matched surfdata-ORGANIC SOC obs [kgC/m2] on the IC grid + provenance."""
    org, src_lat, src_lon = read_surfdata_organic(surf_path)       # (nlat, nlon, nlev)
    dz = model_soil_dz(n_layers, soil_depth)
    if org.shape[-1] != dz.shape[-1]:
        raise SystemExit(
            f"surfdata ORGANIC has {org.shape[-1]} levels but the model soil column "
            f"has {dz.shape[-1]} (n_layers={n_layers}); cannot depth-match.")
    col = integrate_organic_column(org, dz, om_to_oc)              # (nlat, nlon) kgC/m2
    soc = regrid_to_target(col, src_lat, src_lon, tgt_lat, tgt_lon)
    prov = dict(product="surfdata_organic", source=os.path.abspath(surf_path),
                depth_cm=float(soil_depth) * 100.0, om_to_oc=float(om_to_oc),
                n_layers=int(n_layers), dz_sum_m=float(dz.sum()))
    return soc, prov


# ===========================================================================
# SoilGrids 2.0 WCS fetch (networked; compute-node only)
# ===========================================================================
def _raster_to_grid(arr, bbox, *, nodata):
    """Map a north-up 2-D raster to ``(field2d, lat, lon)`` on ascending latitude.

    The georeferencing is taken from the REQUESTED ``bbox`` (x0, x1, y0, y1) and the
    raster shape (cell-centred), NOT from any embedded geotransform -- so no rasterio /
    gdal is needed (absent in this env).  GeoTIFF rows run north (top) -> south; the
    result is reordered to ascending latitude.  ``nodata`` (and non-finite) -> NaN.
    Pure array logic (unit-tested); the imageio decode is the thin wrapper below.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"WCS raster must be 2-D; got shape {arr.shape}.")
    ny, nx = arr.shape
    x0, x1, y0, y1 = bbox
    lon = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    lat_top_down = y1 - (np.arange(ny) + 0.5) * (y1 - y0) / ny
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr[::-1, :], lat_top_down[::-1], lon


def _wcs_geotiff_to_grid(content, bbox, *, nodata):
    """Decode a WCS EPSG:4326 GeoTIFF byte payload (imageio, no rasterio) then map it to
    ``(field2d, lat, lon)`` via :func:`_raster_to_grid`."""
    import imageio.v3 as iio

    arr = np.asarray(iio.imread(io.BytesIO(content)), dtype=np.float64)
    return _raster_to_grid(arr, bbox, nodata=nodata)


def fetch_soilgrids_coverage(prop, depth_token, *, nx, ny, host=_WCS_HOST,
                             nodata=None, timeout=300):
    """Fetch one SoilGrids 2.0 coverage (``{prop}_{depth}_mean``) global, EPSG:4326.

    Returns ``(field2d [mapped units], lat, lon)`` on a regular ``ny x nx`` lat-lon grid.
    Thin networked wrapper (mirrors the ERA5/SIF builders' ``_fetch_*``); the science is
    the tested pure conversion/assembly helpers.  Requests a GeoTIFF and derives the grid
    from the request bbox + output size (server reprojects Homolosine -> 4326).
    """
    import requests

    coverage = f"{prop}_{depth_token}_mean"
    bbox = (-180.0, 180.0, -90.0, 90.0)
    params = {
        "map": f"/map/{prop}.map",
        "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage",
        "COVERAGEID": coverage, "FORMAT": "GEOTIFF_INT16",
        "SUBSETTINGCRS": _WCS_CRS_4326, "OUTPUTCRS": _WCS_CRS_4326,
        # Subset + scale by the coverage's EPSG:4326 axis labels (Lat/Long).  The
        # generic OGC i/j axis URIs are rejected by the ISRIC MapServer WCS
        # (InvalidAxisLabel); Long(nx),Lat(ny) sets the output lon x lat pixel counts.
        "SUBSET": [f"Lat({bbox[2]},{bbox[3]})", f"Long({bbox[0]},{bbox[1]})"],
        "SCALESIZE": f"Long({nx}),Lat({ny})",
    }
    url = f"{host}/mapserv"
    print(f"# WCS GetCoverage {coverage} ({ny}x{nx}, EPSG:4326)")
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200 or not r.content or r.content[:6].lstrip().startswith(b"<"):
        raise SystemExit(
            f"SoilGrids WCS {coverage} failed: HTTP {r.status_code}, "
            f"{len(r.content)} bytes; body head: {r.content[:300]!r}\n  URL: {r.url}")
    return _wcs_geotiff_to_grid(r.content, bbox, nodata=nodata)


def build_soilgrids_obs(tgt_lat, tgt_lon, *, nx, ny, host=_WCS_HOST, nodata=None):
    """SoilGrids 2.0 0-100 cm SOC STOCK obs [kgC/m2] on the IC grid + provenance.

    Fetches soc/bdod/cfvo at the 5 standard depths, assembles the fine-earth stock per
    layer, sums to 0-100 cm, then conservatively regrids to the IC grid.  All source
    layers share one 4326 grid (same request geometry), so the per-layer product is
    cell-aligned before the single regrid.
    """
    src_lat = src_lon = None
    layer_stocks = []
    for token, dz_m in _SOILGRIDS_DEPTHS:
        soc, la, lo = fetch_soilgrids_coverage("soc", token, nx=nx, ny=ny, host=host, nodata=nodata)
        bdod, _, _ = fetch_soilgrids_coverage("bdod", token, nx=nx, ny=ny, host=host, nodata=nodata)
        cfvo, _, _ = fetch_soilgrids_coverage("cfvo", token, nx=nx, ny=ny, host=host, nodata=nodata)
        layer_stocks.append(soilgrids_layer_stock(soc, bdod, cfvo, dz_m))
        if src_lat is None:
            src_lat, src_lon = la, lo
    ocs = assemble_ocs_0_100(layer_stocks)                          # (ny, nx) kgC/m2
    soc = regrid_to_target(ocs, src_lat, src_lon, tgt_lat, tgt_lon)
    prov = dict(product="soilgrids", source=f"{host}/mapserv (SoilGrids 2.0 WCS)",
                depth_cm=100.0, source_grid=f"{ny}x{nx} EPSG:4326",
                depths=[t for t, _ in _SOILGRIDS_DEPTHS])
    return soc, prov


# ===========================================================================
# NetCDF write + CLI
# ===========================================================================
def write_obs_netcdf(path, soc2d, lat, lon, provenance):
    """Write the obs field to the validator's NetCDF schema (dims lat, lon; var soc)."""
    import xarray as xr

    soc2d = np.asarray(soc2d, dtype=np.float64)
    if soc2d.shape != (lat.size, lon.size):
        raise ValueError(
            f"soc grid {soc2d.shape} != (lat={lat.size}, lon={lon.size}).")
    ds = xr.Dataset(
        {_OUT_VAR: (("lat", "lon"), soc2d)},
        coords={"lat": ("lat", np.asarray(lat, dtype=np.float64)),
                "lon": ("lon", np.asarray(lon, dtype=np.float64))})
    ds[_OUT_VAR].attrs.update(units="kgC/m2", long_name="soil organic carbon stock")
    ds.attrs.update(title="Gridded SOC observation on the carbon-IC grid",
                    **{k: str(v) for k, v in provenance.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds.to_netcdf(path)
    return path


def _print_sanity(soc2d, lat, provenance):
    a = np.asarray(soc2d, dtype=np.float64)
    finite = np.isfinite(a)
    w = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)[:, None] * np.ones_like(a)
    wm = float(np.nansum(np.where(finite, a, 0.0) * w) / np.nansum(np.where(finite, w, 0.0)))
    print(f"# {_OUT_VAR} [kgC/m2]: finite {int(finite.sum())}/{a.size} "
          f"({100.0 * finite.sum() / a.size:.1f}%)  min {np.nanmin(a):.3g}  "
          f"cos-lat-mean {wm:.3g}  max {np.nanmax(a):.3g}")
    print(f"# provenance: {provenance}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--product", choices=("surfdata_organic", "soilgrids"),
                   default="surfdata_organic",
                   help="obs source (default surfdata_organic: depth-matched, no network)")
    p.add_argument("--ic-npz", default="results/global_carbon_ic_cuefix2/global_carbon_ic.npz",
                   help="model carbon-IC npz defining the target grid + soil geometry")
    p.add_argument("--surf-path",
                   default="/burg-archive/glab/users/pg2328/legoESM/data/clm/"
                           "surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc",
                   help="CLM5 surfdata NetCDF with ORGANIC (surfdata_organic product)")
    p.add_argument("--om-to-oc", type=float, default=_VAN_BEMMELEN_OC_PER_OM,
                   help="organic-matter -> organic-carbon mass fraction (van Bemmelen 0.58)")
    p.add_argument("--out", default="results/soc_obs/soc_obs.nc",
                   help="output obs NetCDF (dims lat,lon; var 'soc' [kgC/m2])")
    # SoilGrids WCS options.
    p.add_argument("--sg-nx", type=int, default=720,
                   help="SoilGrids WCS output lon size (720 = 0.5 deg global)")
    p.add_argument("--sg-ny", type=int, default=360,
                   help="SoilGrids WCS output lat size (360 = 0.5 deg global)")
    p.add_argument("--sg-nodata", type=float, default=None,
                   help="SoilGrids GeoTIFF nodata sentinel to mask -> NaN")
    p.add_argument("--sg-host", default=_WCS_HOST)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    tgt_lat, tgt_lon, n_layers, soil_depth = read_ic_grid(args.ic_npz)
    print(f"# IC target grid from {args.ic_npz}: ({tgt_lat.size} lat, {tgt_lon.size} lon); "
          f"soil {n_layers} layers to {soil_depth} m")

    if args.product == "surfdata_organic":
        soc, prov = build_surfdata_obs(
            args.surf_path, tgt_lat, tgt_lon,
            n_layers=n_layers, soil_depth=soil_depth, om_to_oc=args.om_to_oc)
    else:
        soc, prov = build_soilgrids_obs(
            tgt_lat, tgt_lon, nx=args.sg_nx, ny=args.sg_ny,
            host=args.sg_host, nodata=args.sg_nodata)

    _print_sanity(soc, tgt_lat, prov)
    out = write_obs_netcdf(args.out, soc, tgt_lat, tgt_lon, prov)
    print(f"# wrote {out}  (var '{_OUT_VAR}' [kgC/m2]; product {prov['product']})")
    return out


if __name__ == "__main__":
    main()
