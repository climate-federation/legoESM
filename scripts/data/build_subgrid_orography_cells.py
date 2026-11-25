#!/usr/bin/env python
"""Per-cell subgrid orography (sgh) over each cell's own footprint.

For a saved MPAS mesh (``save_mpas_mesh.py`` / ``save_voronoi_mesh``), sgh of
every cell is the area-weighted standard deviation of max(z, 0) over the
high-resolution elevation points the cell owns
(``legoesm.grids.topography.subgrid_orography_per_cell``), instead of one
block file interpolated to the cell centres.  The output .npz carries the
mesh's SHA-256 and is refused by the run on any other mesh
(``load_subgrid_orography_cells``).

Usage:
    build_subgrid_orography_cells.py --mesh-file M.npz --mesh-sha256 H \
        --source ETOPO_2022_v1_60s_N90W180_surface.nc --out sgh_cells.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def build(mesh_file, mesh_sha256, source, out, *, var="z", rows_per_chunk=200):
    import xarray as xr
    from legoesm.grids.voronoi_mesh_file import load_voronoi_mesh
    from legoesm.grids.topography import subgrid_orography_per_cell

    mesh = load_voronoi_mesh(mesh_file, expected_sha256=mesh_sha256)
    ds = xr.open_dataset(source)
    try:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        flip = lat[0] > lat[-1]
        z = ds[var]
        nlat = lat.size

        def read_rows(i0, i1):
            if flip:
                return np.asarray(z[nlat - i1:nlat - i0, :].values)[::-1, :]
            return np.asarray(z[i0:i1, :].values)

        sgh, mean, cnt = subgrid_orography_per_cell(
            np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
            lat[::-1] if flip else lat, lon, read_rows,
            rows_per_chunk=rows_per_chunk)
    finally:
        ds.close()
    meta = {"method": "area-weighted std of max(z, 0) over each Voronoi cell "
                      "(nearest generator ownership)",
            "source": os.path.abspath(source), "var": var,
            "mesh_file": os.path.abspath(mesh_file),
            "min_points_per_cell": int(cnt.min())}
    np.savez(out, sgh=sgh, mean_elev=mean, n_src=cnt,
             mesh_sha256=np.asarray(str(mesh_sha256).lower()),
             metadata=np.asarray(json.dumps(meta, sort_keys=True)))
    return sgh, mean, cnt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh-file", required=True)
    p.add_argument("--mesh-sha256", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--var", default="z")
    p.add_argument("--rows-per-chunk", type=int, default=200)
    a = p.parse_args(argv)
    sgh, _, cnt = build(a.mesh_file, a.mesh_sha256, a.source, a.out,
                        var=a.var, rows_per_chunk=a.rows_per_chunk)
    print(f"wrote {a.out}: {sgh.size} cells, sgh max {sgh.max():.1f} m, "
          f"min points per cell {cnt.min()}")


if __name__ == "__main__":
    main()
