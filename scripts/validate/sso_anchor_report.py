"""Compare subgrid-orography (SSO_STDH) files against a reference (#1712).

For each file: area-weighted global and 40-60S means of ``sgh`` and ``sgh^2``
on the file's own lat-lon grid, the same moments after the loader regrids it
onto an MPAS mesh (area-weighted by cell area), the ratio of each ``<sgh^2>``
to the reference's, and the loader's own scale-check verdict for that mesh.

``<sgh^2>`` is an INPUT-FIELD statistic: the launch stress goes as ``sgh^2``
only before the Froude cap and flow dependence, so it is not a drag factor.
Prints one table and writes the numbers as JSON next to ``--json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

SO_BAND = (-60.0, -40.0)


def latlon_moments(path):
    """Area-weighted means of sgh and sgh^2 on the file's own grid."""
    with xr.open_dataset(path) as ds:
        sgh = np.asarray(ds["SSO_STDH"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], sgh.shape)
    so = np.broadcast_to(((lat >= SO_BAND[0]) & (lat <= SO_BAND[1]))[:, None],
                         sgh.shape)
    return _moments(sgh, w, so)


def mesh_moments(path, mesh):
    """Same moments after ``load_subgrid_orography`` regrids onto ``mesh``
    (a VoronoiMesh), area-weighted by its cell areas."""
    from legoesm.grids.topography import load_subgrid_orography

    sgh = np.asarray(load_subgrid_orography(mesh, str(path), scale_check="off"),
                     dtype=np.float64)
    lat_deg = np.rad2deg(np.asarray(mesh.latCell))
    so = (lat_deg >= SO_BAND[0]) & (lat_deg <= SO_BAND[1])
    return _moments(sgh, np.asarray(mesh.areaCell, dtype=np.float64), so)


def guard_verdict(path, ncells):
    """The loader's scale check for this file on an ``ncells`` mesh."""
    from legoesm.grids.topography import (
        _check_sso_scale_decomposition, _sso_file_construction,
        voronoi_cell_spacing_deg)

    with xr.open_dataset(path) as ds:
        built = _sso_file_construction(ds)
    msg = _check_sso_scale_decomposition(
        built, voronoi_cell_spacing_deg(ncells), str(path), "warn")
    return "PASS" if msg is None else "MISMATCH"


def _moments(sgh, w, so):
    if not np.all(np.isfinite(sgh)):
        raise ValueError("non-finite SSO_STDH values")
    return {
        "mean": float((w * sgh).sum() / w.sum()),
        "ms": float((w * sgh ** 2).sum() / w.sum()),
        "so_mean": float((w * sgh)[so].sum() / w[so].sum()),
        "so_ms": float((w * sgh ** 2)[so].sum() / w[so].sum()),
    }


def report(reference, files, mesh=None):
    """Rows of moments and ``<sgh^2>`` ratios to ``reference``."""
    rows = []
    for path in [reference, *files]:
        row = {"file": str(path), "latlon": latlon_moments(path)}
        if mesh is not None:
            row["mesh"] = mesh_moments(path, mesh)
            row["guard"] = guard_verdict(path, int(np.asarray(mesh.latCell).size))
        rows.append(row)
    ref = rows[0]
    for row in rows:
        for space in ("latlon", "mesh"):
            if space in row:
                row[space]["ratio_ms"] = row[space]["ms"] / ref[space]["ms"]
                row[space]["ratio_so_ms"] = (row[space]["so_ms"]
                                             / ref[space]["so_ms"])
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--reference", required=True,
                   help="the SSO file production reads")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--mesh-level", type=int, default=None,
                   help="also regrid onto this icosahedral MPAS level")
    p.add_argument("--json", required=True)
    args = p.parse_args(argv)

    mesh = None
    if args.mesh_level is not None:
        from legoesm.grids.voronoi import create_voronoi_mesh

        mesh = create_voronoi_mesh(args.mesh_level)
    rows = report(args.reference, args.files, mesh)
    Path(args.json).write_text(json.dumps(rows, indent=1))
    print("(first row = the reference; its guard verdict is about the file "
          "production reads today)")
    for row in rows:
        ll = row["latlon"]
        line = (f"{Path(row['file']).name:45s} latlon: mean {ll['mean']:7.2f} "
                f"SO {ll['so_mean']:6.2f} <s2>x {ll['ratio_ms']:5.2f} "
                f"SO<s2>x {ll['ratio_so_ms']:5.2f}")
        if "mesh" in row:
            ms = row["mesh"]
            line += (f" | mesh: mean {ms['mean']:7.2f} <s2>x "
                     f"{ms['ratio_ms']:5.2f} SO<s2>x {ms['ratio_so_ms']:5.2f} "
                     f"guard {row['guard']}")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
