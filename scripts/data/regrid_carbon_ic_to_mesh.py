#!/usr/bin/env python
"""Put an existing land-carbon finidat onto an unstructured mesh.

The published carbon initial condition (`global_carbon_ic.npz`, the release
`land-carbon-ic-v1`) is on a regular lat-lon grid. `load_finidat_carbon_ic`
matches the finidat's cells to the run's columns ELEMENT-FOR-ELEMENT and raises
on any mismatch, deliberately: the pools are per-area stocks pinned to their own
cells, so reshaping them onto a different grid would corrupt the initial
condition. Its docstring therefore says a Voronoi run "must build the carbon IC
on its own grid".

Rebuilding it on the mesh means re-running the archetype spin-up. Interpolating
it does not, and for a coarse mesh whose cells are larger than the source's it
is the cheaper and more faithful of the two: the equilibrium was computed for a
(plant type x climate) archetype, and a cell of the mesh inherits the same
archetypes as the lat-lon cells it covers.

Reuses the repo's lat-lon -> unstructured interpolator; no new numerics. The
NaN-aware variant is the one that matters here, because the source is land-only:
it drops missing neighbours and renormalises, so ocean does not bleed into
coastal mesh cells and a cell is only left empty when every neighbour is.

WHAT THIS DOES NOT DO. Interpolation is not conservative -- the global carbon
stock is not preserved to roundoff, and the script prints the change so the
error is visible rather than assumed. It also cannot invent land: a mesh cell
whose neighbours are all ocean gets the cold-start value, and the count is
printed.

Usage:
    python scripts/data/regrid_carbon_ic_to_mesh.py \
        --source data/land_carbon_ic/global_carbon_ic.npz \
        --resolution 5 --output data/land_carbon_ic/global_carbon_ic_mpas5.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from legoesm.grids.regridding import (
    compute_latlon_to_voronoi_weights,
    regrid_scalar_nan_aware,
)
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.land.carbon.config import CarbonState

# Per-cell fields the finidat carries alongside the pools.  Everything else in
# the file is metadata (scalars, PFT names) and is copied through untouched.
_EXTRA_CELL_FIELDS = ("soil_frozen_fraction", "land_mask", "dominant_pft",
                      "pft_weights")


def _axes_from_flat(lat_flat, lon_flat):
    """Recover the (n_lat,), (n_lon,) axes of a flattened regular grid.

    The finidat stores lat/lon per cell, so the regular structure has to be
    recovered before the interpolator (which wants the two axes) can be used.
    Row-major with longitude fastest is asserted, not assumed: a file stored the
    other way round would otherwise be silently transposed.
    """
    lat_1d = np.unique(lat_flat)
    lon_1d = np.unique(lon_flat)
    if lat_1d.size * lon_1d.size != lat_flat.size:
        raise SystemExit(
            f"FATAL: {lat_flat.size} cells is not {lat_1d.size} x {lon_1d.size}; "
            "the source is not a complete regular lat-lon grid")
    expect = np.repeat(lat_1d, lon_1d.size)
    if not np.allclose(np.sort(lat_flat), np.sort(expect)):
        raise SystemExit("FATAL: source lat/lon do not form a regular grid")
    if not np.allclose(lat_flat, expect):
        raise SystemExit(
            "FATAL: source is not row-major with longitude fastest; the "
            "reshape below would transpose the map")
    return lat_1d, lon_1d


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--resolution", type=int, required=True,
                    help="SCVT subdivision level (4 = 2562 cells, 5 = 10242)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--k-neighbors", type=int, default=4)
    args = ap.parse_args(argv)

    z = np.load(args.source)
    pool_fields = CarbonState._fields
    missing = [f for f in pool_fields if f not in z.files]
    if missing:
        raise SystemExit(f"FATAL: {args.source} is not a carbon finidat; "
                         f"missing pools {missing}")

    lat_flat = np.asarray(z["lat"], dtype=np.float64)
    lon_flat = np.asarray(z["lon"], dtype=np.float64)
    lat_1d, lon_1d = _axes_from_flat(lat_flat, lon_flat)
    shape = (lat_1d.size, lon_1d.size)

    mesh = create_voronoi_mesh(args.resolution)
    tgt_lat = np.asarray(mesh.latCell, dtype=np.float64)
    tgt_lon = np.asarray(mesh.lonCell, dtype=np.float64)

    land = np.asarray(z["land_mask"], dtype=np.float64) > 0
    w = compute_latlon_to_voronoi_weights(
        np.deg2rad(lat_1d), np.deg2rad(lon_1d), tgt_lat, tgt_lon,
        k_neighbors=args.k_neighbors)

    # Source area weights, for the stock comparison printed at the end.
    src_area = np.cos(np.deg2rad(lat_flat)) * land
    tgt_area = np.asarray(mesh.areaCell, dtype=np.float64)

    out = {}
    n_empty = 0
    for name in list(pool_fields) + list(_EXTRA_CELL_FIELDS):
        if name not in z.files:
            continue
        src = np.asarray(z[name], dtype=np.float64)
        lead = src.shape[0] if src.ndim > 1 else src.size
        if lead != lat_flat.size:          # not a per-cell field
            out[name] = z[name]
            continue
        # Ocean -> NaN so the NaN-aware interpolator drops it instead of
        # averaging a zero into the coast.
        masked = np.where(land.reshape(-1, *([1] * (src.ndim - 1))), src, np.nan)
        got = np.asarray(regrid_scalar_nan_aware(
            masked.reshape(*shape, *src.shape[1:]), w), dtype=np.float64)
        n_empty = max(n_empty, int(np.sum(~np.isfinite(got.reshape(got.shape[0], -1)[:, 0]))))
        out[name] = np.nan_to_num(got, nan=0.0)

    # A cell only counts as land on the mesh if the interpolation found land.
    tgt_land = np.isfinite(np.asarray(regrid_scalar_nan_aware(
        np.where(land, 1.0, np.nan).reshape(shape), w), dtype=np.float64))
    out["land_mask"] = tgt_land.astype(np.float64)
    out["lat"] = np.rad2deg(tgt_lat)
    out["lon"] = np.rad2deg(tgt_lon) % 360.0
    for k in z.files:                      # metadata straight through
        out.setdefault(k, z[k])

    np.savez(args.output, **out)

    soc = sum(np.asarray(z[f]) for f in
              ("C_som_active", "C_som_slow", "C_som_passive"))
    soc_t = sum(out[f] for f in ("C_som_active", "C_som_slow", "C_som_passive"))
    src_mean = float((soc * src_area).sum() / src_area.sum()) / 1000.0
    ta = tgt_area * tgt_land
    tgt_mean = float((soc_t * ta).sum() / max(ta.sum(), 1e-30)) / 1000.0
    print(f"  source: {int(land.sum())} land cells of {lat_flat.size}, "
          f"area-weighted soil carbon {src_mean:.3f} kgC/m2")
    print(f"  mesh:   {int(tgt_land.sum())} land cells of {tgt_lat.size}, "
          f"area-weighted soil carbon {tgt_mean:.3f} kgC/m2")
    print(f"  change {100.0 * (tgt_mean - src_mean) / src_mean:+.2f} % -- "
          f"interpolation is NOT conservative; this is the size of that error")
    if n_empty:
        print(f"  {n_empty} mesh cells found no land neighbour and are zero")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
