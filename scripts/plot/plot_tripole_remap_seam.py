#!/usr/bin/env python
"""Fold-seam visual gate for the Phase-2 regular-lat-lon <-> tripole remap.

CLAUDE.md visual-verify rule: passing conservation norms are NECESSARY but NOT
SUFFICIENT for grid-coupling correctness — a remap bug at the bipolar cap shows
up only as a SEAM artifact (a discontinuity along the north-fold) that norms
hide.  This renders a SMOOTH analytic field remapped regular-atm -> eORCA1
tripole (and the round-trip back), so the bipolar cap can be eyeballed for
seam/duplication artifacts.

A correct first-order conservative remap of a smooth field must itself look
smooth everywhere INCLUDING across the fold; any sharp line along the top rows
(j ~ cap_j..n_lat) is a fold-handling bug.

Usage (compute node / sbatch, NOT the login node):
    python scripts/plot/plot_tripole_remap_seam.py \
        --mesh data/grids/eORCA1.2_mesh_mask.nc --out docs/scaling/tripole_remap_seam.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="data/grids/eORCA1.2_mesh_mask.nc")
    ap.add_argument("--out", default="docs/scaling/tripole_remap_seam.png")
    ap.add_argument("--atm-nlat", type=int, default=72)  # 2.5deg atm
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.tripole import create_tripole_grid
    from legoesm.coupler.grid_remap import make_curvilinear_latlon_remapper
    from legoesm.grids.conservative_regrid import apply_conservative_regrid

    atm = create_latlon_grid(args.atm_nlat)
    trip = create_tripole_grid(args.mesh)
    print(f"atm {atm.n_lat}x{atm.n_lon}  tripole {trip.n_lat}x{trip.n_lon} "
          f"(fold cap_j={trip.fold.cap_j})")

    # Smooth analytic field on the atm grid: a low-order spherical pattern.
    lat = np.asarray(atm.lat)
    lon = np.asarray(atm.lon)
    LON, LAT = np.meshgrid(lon, lat)
    field_atm = (1.0 + 0.6 * np.sin(LAT) + 0.4 * np.cos(LAT) * np.cos(2 * LON))

    a2o = make_curvilinear_latlon_remapper(atm, trip)
    o2a = make_curvilinear_latlon_remapper(trip, atm)
    field_trip = np.asarray(apply_conservative_regrid(jnp.asarray(field_atm), a2o))
    field_back = np.asarray(apply_conservative_regrid(jnp.asarray(field_trip), o2a))

    # Conservation / range diagnostics (printed; the figure is the visual gate).
    print(f"atm  field range [{field_atm.min():.4f}, {field_atm.max():.4f}]")
    print(f"trip field range [{field_trip.min():.4f}, {field_trip.max():.4f}] "
          f"(must stay within atm range — no overshoot)")
    rt_err = np.abs(field_back - field_atm)
    print(f"round-trip max|err| {rt_err.max():.4f}, mean {rt_err.mean():.4f}")

    lat_t = np.degrees(np.asarray(trip.lat_T))
    lon_t = np.degrees(np.asarray(trip.lon_T))
    cap_j = int(trip.fold.cap_j)

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    vmin, vmax = field_atm.min(), field_atm.max()

    axs[0, 0].pcolormesh(LON * 180 / np.pi, LAT * 180 / np.pi, field_atm,
                         vmin=vmin, vmax=vmax, shading="auto", cmap="viridis")
    axs[0, 0].set_title(f"source: smooth field on atm {atm.n_lat}x{atm.n_lon}")

    sc = axs[0, 1].scatter(lon_t.ravel(), lat_t.ravel(), c=field_trip.ravel(),
                           s=2, vmin=vmin, vmax=vmax, cmap="viridis")
    axs[0, 1].set_title("remapped onto eORCA1 tripole (global)")
    fig.colorbar(sc, ax=axs[0, 1])

    # Zoom on the Arctic bipolar cap: the seam-prone region.
    capslice = slice(max(0, cap_j - 4), trip.n_lat)
    lo = lon_t[capslice]; la = lat_t[capslice]; fv = field_trip[capslice]
    sc2 = axs[1, 0].scatter(lo.ravel(), la.ravel(), c=fv.ravel(), s=8,
                            vmin=vmin, vmax=vmax, cmap="viridis")
    axs[1, 0].set_title(f"BIPOLAR CAP (rows {capslice.start}..{trip.n_lat}) — "
                        "seam check (must be smooth)")
    fig.colorbar(sc2, ax=axs[1, 0])

    im = axs[1, 1].pcolormesh(LON * 180 / np.pi, LAT * 180 / np.pi, rt_err,
                              shading="auto", cmap="magma")
    axs[1, 1].set_title(f"round-trip |error| (max {rt_err.max():.3f})")
    fig.colorbar(im, ax=axs[1, 1])

    fig.suptitle("Phase-2 atm(lat-lon) <-> eORCA1 tripole conservative remap — "
                 "fold-seam visual gate", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
