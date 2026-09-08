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
interpolator is given the source LAND MASK as its valid set, so its neighbour
tree is built from land cells only: every mesh cell takes the value of its
nearest actual land, ocean never bleeds in, and no cell is left empty. That
last part is not a nicety. An earlier version left unmatched cells at zero,
which two reviewers independently flagged: the loader does not read a land
mask, so a zeroed cell becomes a real land column holding no carbon and, worse,
a permafrost index of zero -- which passes the loader's range check and silently
switches OFF the frozen-ground protection. Islands and ice margins are exactly
the cells with no nearby source land, and exactly the cells where that is most
wrong.

Which cells are land is decided by the RUN, from its own land fraction. This
file only says what the carbon is if a cell is land.

WHAT THIS DOES NOT DO. Interpolation is not conservative -- the global carbon
stock is not preserved to roundoff, and the script prints the change so the
error is visible rather than assumed.

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
    regrid_scalar,
)
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.land.carbon.config import CarbonState

# Per-cell fields interpolated alongside the pools.  ``dominant_pft`` and
# ``pft_present`` are deliberately NOT in this list: the first is a category
# and the second a boolean, and interpolating either gives a number that is not
# one -- a "dominant plant type" of 4.37.  Both are RECOMPUTED from the
# interpolated cover weights instead.  Everything else in the file is metadata
# (scalars, plant-type names) and is copied through untouched.
_EXTRA_CELL_FIELDS = ("soil_frozen_fraction", "pft_weights")
_DERIVED_FROM_WEIGHTS = ("dominant_pft", "pft_present", "land_mask")


def _axes_from_flat(lat_flat, lon_flat):
    """Recover the (n_lat,), (n_lon,) axes of a flattened regular grid.

    The finidat stores lat/lon per cell, so the regular structure has to be
    recovered before the interpolator (which wants the two axes) can be used.
    Row-major with longitude fastest is asserted, not assumed: a file stored the
    other way round would otherwise be silently transposed.
    """
    n_lat = np.unique(lat_flat).size
    n_lon = np.unique(lon_flat).size
    if n_lat * n_lon != lat_flat.size:
        raise SystemExit(
            f"FATAL: {lat_flat.size} cells is not {n_lat} x {n_lon}; "
            "the source is not a complete regular lat-lon grid")
    # Take the axes in STORAGE order, not sorted order: a descending-latitude
    # file is perfectly legal and sorting would silently flip the map. The
    # reshape below is then checked against the stored arrays element for
    # element, which is what makes this guard bite -- a reordered or cyclically
    # shifted longitude row fails here rather than being mapped against a
    # sorted axis it does not have.
    lat_1d = lat_flat.reshape(n_lat, n_lon)[:, 0]
    lon_1d = lon_flat.reshape(n_lat, n_lon)[0, :]
    if not np.allclose(lat_flat, np.repeat(lat_1d, n_lon)):
        raise SystemExit(
            "FATAL: source is not row-major with longitude fastest; the "
            "reshape would transpose the map")
    if not np.allclose(lon_flat, np.tile(lon_1d, n_lat)):
        raise SystemExit(
            "FATAL: source longitude row is not the same for every latitude; "
            "the reshape would misalign the map")
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
    # src_valid = the land mask: the neighbour tree is built from LAND cells
    # only, so every mesh cell maps to its nearest actual land and nothing is
    # left unmatched.  This is the repo's own facility, not a new fill rule.
    w = compute_latlon_to_voronoi_weights(
        np.deg2rad(lat_1d), np.deg2rad(lon_1d), tgt_lat, tgt_lon,
        k_neighbors=args.k_neighbors, src_valid=land)

    # Source area weights, for the stock comparison printed at the end.
    src_area = np.cos(np.deg2rad(lat_flat)) * land
    tgt_area = np.asarray(mesh.areaCell, dtype=np.float64)

    out = {}
    for name in list(pool_fields) + list(_EXTRA_CELL_FIELDS):
        if name not in z.files:
            continue
        src = np.asarray(z[name], dtype=np.float64)
        lead = src.shape[0] if src.ndim > 1 else src.size
        if lead != lat_flat.size:          # not a per-cell field
            out[name] = z[name]
            continue
        if src.ndim > 2:
            raise SystemExit(
                f"FATAL: {name} has rank {src.ndim}; the interpolator flattens "
                "every trailing dimension into one, so a rank-3 field would "
                "come back with its dimensions fused")
        got = np.asarray(regrid_scalar(src.reshape(*shape, *src.shape[1:]), w),
                         dtype=np.float64)
        if not np.all(np.isfinite(got)):
            raise SystemExit(f"FATAL: {name} interpolated to a non-finite "
                             "value; the land-only neighbour tree should make "
                             "that impossible")
        out[name] = got

    # Category and boolean companions are RECOMPUTED, never interpolated.
    if "pft_weights" in out:
        pw = out["pft_weights"]
        out["dominant_pft"] = np.argmax(pw, axis=1).astype(
            np.asarray(z["dominant_pft"]).dtype if "dominant_pft" in z.files
            else np.int64)
        if "pft_present" in z.files:
            out["pft_present"] = (pw > 0.0).astype(
                np.asarray(z["pft_present"]).dtype)
    # Every mesh cell now carries its nearest land cell's carbon.  Where land
    # actually IS, is the run's decision, from its own land fraction -- and the
    # loader does not read this field at all.  Written as all-land so nothing
    # downstream mistakes it for a mask that gates anything.
    out["land_mask"] = np.ones(tgt_lat.size, dtype=np.float64)
    out["lat"] = np.rad2deg(tgt_lat)
    out["lon"] = np.rad2deg(tgt_lon) % 360.0
    for k in z.files:                      # metadata straight through
        if k in _DERIVED_FROM_WEIGHTS:
            continue                       # already recomputed above
        out.setdefault(k, z[k])
    stale = [k for k, v in out.items()
             if isinstance(v, np.ndarray) and v.ndim >= 1
             and v.shape[0] == lat_flat.size and lat_flat.size != tgt_lat.size]
    if stale:
        raise SystemExit(f"FATAL: {stale} still carry the SOURCE cell count; "
                         "a consumer reading them would index a different grid")

    np.savez(args.output, **out)

    soc = sum(np.asarray(z[f]) for f in
              ("C_som_active", "C_som_slow", "C_som_passive"))
    soc_t = sum(out[f] for f in ("C_som_active", "C_som_slow", "C_som_passive"))
    src_mean = float((soc * src_area).sum() / src_area.sum()) / 1000.0
    # Like-for-like needs a LAND-AREA weight on both sides.  The mesh's land
    # fraction comes from interpolating the source mask with an UNMASKED tree --
    # the land-only tree used for the pools would return 1 everywhere by
    # construction and quietly turn this into a whole-globe mean.
    w_all = compute_latlon_to_voronoi_weights(
        np.deg2rad(lat_1d), np.deg2rad(lon_1d), tgt_lat, tgt_lon,
        k_neighbors=args.k_neighbors)
    tgt_landfrac = np.clip(np.asarray(regrid_scalar(
        land.astype(np.float64).reshape(shape), w_all), dtype=np.float64),
        0.0, 1.0)
    ta = tgt_area * tgt_landfrac
    tgt_mean = float((soc_t * ta).sum() / max(ta.sum(), 1e-30)) / 1000.0
    print(f"  source: {int(land.sum())} land cells of {lat_flat.size}, "
          f"area-weighted soil carbon {src_mean:.3f} kgC/m2")
    print(f"  mesh:   {tgt_landfrac.sum():.0f} land-cell equivalents of "
          f"{tgt_lat.size}, land-area-weighted soil carbon {tgt_mean:.3f} kgC/m2")
    print(f"  change {100.0 * (tgt_mean - src_mean) / src_mean:+.2f} % -- "
          f"interpolation is NOT conservative; this is the size of that error")
    print(f"  every mesh cell carries its nearest land cell's carbon; the RUN's "
          f"land fraction decides where land is")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
