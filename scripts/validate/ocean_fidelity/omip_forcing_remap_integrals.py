"""Area integral of OMIP flux forcing after each ocean grid's sampling (F20).

For precipitation and downwelling shortwave on CORE-II NYF day means, compare
the area integral over each ocean grid's footprint after nearest-neighbour and
bilinear point sampling against a first-order conservative reference on the
same footprint (the existing ``make_regular_to_curvilinear_weights``: tile the
source, assign each sub-area to the nearest target centre).  On a GLOBAL
target the conservative integral must reproduce the source global integral --
that is the instrument self-check, printed as ``cons-vs-src``.

Net shortwave without sea ice is ``sw_down * (1 - constant albedo)``, so its
relative error equals sw_down's.

Measured 2026-10-11 (review 2026-10-10 F20; Jan 1 and Jul 1): every method on
every grid within 0.37% of the conservative integral; MPAS cons-vs-src
<= 0.007%.  The grids differed pointwise (precip ~5-8% area-RMS), which is why
every point grid now samples bilinearly.

Usage (compute node; imports jax):
  omip_forcing_remap_integrals.py --forcing-dir D --mesh-mask eORCA1.nc
      [--mpas-level 7 --mpas-lloyd 50] [--fesom-mesh-dir M --fesom-repo R]
      [--grids tripole,mpas,fesom]
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.grids.conservative_regrid import apply_conservative_regrid
from legoesm.grids.conservative_regrid_curvilinear import (
    make_regular_to_curvilinear_weights,
)
from legoesm.ocean.coupler.omip2_applicator import (
    _bilinear_interp_to_points,
    _edges_from_centers_deg,
    _nn_interp_to_points,
)

CHANNELS = ("precip", "sw_down")
DAYS = {"jan01": 0, "jul01": 181}


def source_grid(src_lat_deg, src_lon_deg):
    """(regular-grid namespace for the weight builder, source cell areas m^2).

    Edges are inferred from uniformly spaced centres exactly as the lat-lon
    forcing path does (``_edges_from_centers_deg``), clamped to the poles."""
    src_lat = np.asarray(src_lat_deg, dtype=np.float64)
    src_lon = np.asarray(src_lon_deg, dtype=np.float64)
    if src_lat[0] > src_lat[-1]:
        raise ValueError("descending source latitude is not supported")
    lat_e = np.clip(_edges_from_centers_deg(src_lat), -np.pi / 2, np.pi / 2)
    r = float(constants.R_earth)
    dlam = np.radians(abs(src_lon[1] - src_lon[0]))
    area = (r * r * dlam * np.abs(np.diff(np.sin(lat_e))))[:, None] \
        * np.ones(src_lon.size)[None, :]
    reg = types.SimpleNamespace(lat_v=lat_e, lon=np.radians(src_lon))
    return reg, area


def compare_methods(field, src_lat_deg, src_lon_deg, lat_deg, lon_deg, area,
                    footprints, *, weights=None):
    """Integrals of ``field`` sampled at points ``(lat_deg, lon_deg)``.

    Returns ``{footprint: {"cons": I, "src": I_src (global only),
    "nearest": rel, "bilinear": rel, "nearest_rms": r, "bilinear_rms": r}}``
    with ``rel = I_method / I_cons - 1`` and ``r`` the area-RMS pointwise
    difference to the conservative field divided by its area mean.  A
    footprint named ``global`` also gets ``src`` (source global integral) and
    ``cons_vs_src``.  ``weights`` may be passed to reuse a built remap."""
    reg, src_area = source_grid(src_lat_deg, src_lon_deg)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    if weights is None:
        weights = make_conservative_weights(src_lat_deg, src_lon_deg,
                                            lat_deg, lon_deg)
    f = np.asarray(field, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("non-finite source field")
    cons = np.asarray(apply_conservative_regrid(f, weights)).reshape(-1)
    sampled = {
        "nearest": np.asarray(_nn_interp_to_points(
            f, src_lat_deg, src_lon_deg, lat_deg, lon_deg)),
        "bilinear": np.asarray(_bilinear_interp_to_points(
            f, src_lat_deg, src_lon_deg, lat_deg, lon_deg)),
    }
    out = {}
    for name, m in footprints.items():
        a = area[m]
        i_c = float(np.sum(cons[m] * a))
        mean = i_c / float(np.sum(a))
        row = {"cons": i_c}
        if name == "global":
            row["src"] = float(np.sum(f * src_area))
            row["cons_vs_src"] = i_c / row["src"] - 1.0
        for k, g in sampled.items():
            row[k] = float(np.sum(g[m] * a)) / i_c - 1.0
            row[k + "_rms"] = float(np.sqrt(
                np.sum((g[m] - cons[m]) ** 2 * a) / np.sum(a))) / mean
        out[name] = row
    return out


def make_conservative_weights(src_lat_deg, src_lon_deg, lat_deg, lon_deg):
    """First-order conservative weights onto an arbitrary point set, via the
    existing tripole builder with the points as a (1, n) 'tripole'."""
    reg, _ = source_grid(src_lat_deg, src_lon_deg)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    tgt = types.SimpleNamespace(
        lat_T=np.radians(lat_deg)[None, :],
        lon_T=np.radians(np.asarray(lon_deg, dtype=np.float64))[None, :],
        n_lat=1, n_lon=lat_deg.size)
    return make_regular_to_curvilinear_weights(reg, tgt)


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], -1)


def _report(label, forcing, lat, lon, area, footprints):
    w = make_conservative_weights(forcing.lat, forcing.lon, lat, lon)
    for c in CHANNELS:
        for d, k in DAYS.items():
            f = np.asarray(getattr(forcing, c)[4 * k:4 * k + 4]).mean(axis=0)
            res = compare_methods(f, forcing.lat, forcing.lon, lat, lon, area,
                                  footprints, weights=w)
            for fp, r in res.items():
                extra = (f" cons-vs-src={100 * r['cons_vs_src']:+.3f}%"
                         if "cons_vs_src" in r else "")
                print(f"[{label}] {c:8s} {d} {fp:6s} I_cons={r['cons']:.6e}"
                      f"{extra}  nearest={100 * r['nearest']:+.3f}% "
                      f"(rms/mean {r['nearest_rms']:.3f})  "
                      f"bilinear={100 * r['bilinear']:+.3f}% "
                      f"(rms/mean {r['bilinear_rms']:.3f})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcing-dir", required=True)
    p.add_argument("--mesh-mask", required=True, help="eORCA1 mesh_mask.nc")
    p.add_argument("--mpas-level", type=int, default=7)
    p.add_argument("--mpas-lloyd", type=int, default=50)
    p.add_argument("--fesom-mesh-dir")
    p.add_argument("--fesom-repo", help="fesom_jax checkout, put on sys.path")
    p.add_argument("--grids", default="tripole,mpas,fesom")
    a = p.parse_args(argv)
    grids = set(a.grids.split(","))
    unknown = grids - {"tripole", "mpas", "fesom"}
    if unknown:
        raise SystemExit(f"unknown grids {sorted(unknown)}")

    import xarray as xr
    from scipy.spatial import cKDTree

    from legoesm.ocean.forcing import load_core2_nyf

    forcing = load_core2_nyf(allow_synthetic=False,
                             cache_dir=Path(a.forcing_dir))
    r = float(constants.R_earth)
    ds = xr.open_dataset(a.mesh_mask)
    sq = lambda v: np.asarray(ds[v]).squeeze()  # noqa: E731
    glamt, gphit = sq("glamt"), sq("gphit")
    area_t = sq("e1t") * sq("e2t")
    tm = sq("tmask")
    wet = (tm[0] if tm.ndim == 3 else tm) > 0.5
    inner = np.zeros_like(wet)
    inner[:-1, 1:-1] = True       # drop cyclic halo columns and the fold row
    if "tripole" in grids:
        _report("tripole", forcing, gphit[inner], glamt[inner], area_t[inner],
                {"global": np.ones(int(inner.sum()), bool),
                 "ocean": wet[inner]})
    if "mpas" in grids:
        from legoesm.grids.voronoi import create_voronoi_mesh
        mesh = create_voronoi_mesh(a.mpas_level,
                                   lloyd_iterations=a.mpas_lloyd)
        mlat = np.degrees(np.asarray(mesh.latCell))
        mlon = np.degrees(np.asarray(mesh.lonCell))
        marea = np.asarray(mesh.areaCell, dtype=np.float64)
        print(f"[mpas] level {a.mpas_level} lloyd {a.mpas_lloyd}: "
              f"sum area / 4piR^2 = {marea.sum() / (4 * np.pi * r * r):.6f}")
        # ocean = eORCA1 wet flag of the nearest eORCA T point (same mask
        # for every method, so the comparison is footprint-controlled)
        _, k = cKDTree(_xyz(gphit[inner], glamt[inner])).query(
            _xyz(mlat, mlon))
        _report("mpas", forcing, mlat, mlon, marea,
                {"global": np.ones(mlat.size, bool), "ocean": wet[inner][k]})
    if "fesom" in grids:
        if not a.fesom_mesh_dir:
            raise SystemExit("--grids fesom needs --fesom-mesh-dir")
        if a.fesom_repo:
            sys.path.insert(0, a.fesom_repo)
        from fesom_jax.mesh import load_mesh
        fm = load_mesh(a.fesom_mesh_dir)
        geo = np.asarray(fm.geo_coord_nod2D, dtype=np.float64)
        farea = np.asarray(fm.areasvol[:, 0], dtype=np.float64)
        # FESOM mesh is ocean-only: no global footprint, so no self-check.
        _report("fesom", forcing, np.degrees(geo[:, 1]),
                np.degrees(geo[:, 0]), farea,
                {"ocean": np.ones(farea.size, bool)})
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
