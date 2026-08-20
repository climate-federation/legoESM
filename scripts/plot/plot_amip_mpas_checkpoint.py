#!/usr/bin/env python
"""Realism snapshot from an AMIP MPAS checkpoint: maps + zonal circulation.

The century chain writes ``checkpoint_day_NNNN.npz`` on the native Voronoi
mesh (cell-centred scalars, edge-normal winds).  ``plot_amip_snapshot_maps.py``
covers lat-lon ``snapshots.npz`` runs and cannot read these; this is the MPAS
counterpart the loop task needs every iteration — the spatial and circulation
check that scalar health lines (mean T, CWV, |u|max) cannot see.

Four panels:

* near-surface temperature map (lowest model level),
* column water vapour map,
* zonal-mean zonal wind — the jet structure / Hadley signature,
* zonal-mean temperature — the equator-pole gradient and cold point.

Shared utilities only (CLAUDE.md): the mesh comes from
``legoesm.grids.voronoi.create_voronoi_mesh``, cell winds from
``reconstruct_cell_velocity``, CWV from
``legoesm.diagnostics.column_integrals.column_water_vapor``, and the sole
constant is ``constants.T_freeze`` for the degC axis.  No re-derived
saturation curves or column integrals.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/plot/plot_amip_mpas_checkpoint.py \
        <run_dir> [--day N] [--out fig.png]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def latest_checkpoint(run_dir: Path, day: int | None = None) -> Path:
    """The requested checkpoint, or the highest-numbered one."""
    found = {}
    for f in run_dir.glob("checkpoint_day_*.npz"):
        m = re.search(r"checkpoint_day_0*(\d+)\.npz", f.name)
        if m:
            found[int(m.group(1))] = f
    if not found:
        raise SystemExit(f"no checkpoint_day_*.npz in {run_dir}")
    if day is not None:
        if day not in found:
            raise SystemExit(
                f"no checkpoint for day {day}; have {sorted(found)}")
        return found[day]
    return found[max(found)]


def subdivisions_for(n_cells: int) -> int:
    """Icosahedral subdivision level giving ``n_cells`` (10*4^k + 2).

    Raises rather than guessing: a wrong level silently pairs fields with the
    wrong cell centres, which would render a plausible-looking but fictitious
    map.
    """
    for k in range(12):
        if 10 * 4 ** k + 2 == n_cells:
            return k
    raise SystemExit(
        f"{n_cells} cells is not an icosahedral count (10*4^k+2); "
        "this checkpoint is not on a uniform SCVT mesh")


def load_state(path: Path):
    """(T, q_v, u_edge, p_s, dsigma) from a checkpoint, as float64 numpy."""
    d = np.load(path, allow_pickle=True)
    sigma_half = np.asarray(d["meta_vgrid"], dtype=np.float64)[1]
    return (np.asarray(d["T"], dtype=np.float64),
            np.asarray(d["trc_q_v"], dtype=np.float64),
            np.asarray(d["u"], dtype=np.float64),
            np.asarray(d["p_s"], dtype=np.float64),
            np.diff(sigma_half))


def zonal_mean(field_2d, lat_cell, n_bins=36):
    """Area-naive zonal mean on equal-area latitude bins.

    Equal-area (uniform in sin(lat)) rather than uniform-in-latitude so each
    bin holds a comparable cell count on an SCVT mesh — a uniform-in-lat
    binning leaves the polar bins with a handful of cells and a noisy mean.
    Returns (bin_centre_lat, mean_profile) with empty bins dropped.
    """
    edges = np.arcsin(np.linspace(-1.0, 1.0, n_bins + 1)) * 180.0 / np.pi
    idx = np.clip(np.searchsorted(edges, lat_cell, side="right") - 1,
                  0, n_bins - 1)
    lats, prof = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            lats.append(0.5 * (edges[b] + edges[b + 1]))
            prof.append(field_2d[sel].mean(axis=0))
    return np.array(lats), np.array(prof)


def build_figure(run_dir: Path, day: int | None, out: Path | None):
    import jax
    # The checkpoint is float64 and the CWV integral runs through jax; without
    # this the shared helper silently truncates to float32 and the printed
    # column mass differs from the model's own diagnostic.
    jax.config.update("jax_enable_x64", True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from legoesm.diagnostics.column_integrals import column_water_vapor
    from legoesm.grids.voronoi import (
        create_voronoi_mesh,
        reconstruct_cell_velocity,
    )

    from legoesm import constants

    ckpt = latest_checkpoint(run_dir, day)
    T, q_v, u_edge, p_s, dsigma = load_state(ckpt)  # noqa: N806 (T = temperature, domain convention)
    sim_day = int(re.search(r"checkpoint_day_0*(\d+)", ckpt.name).group(1))

    mesh = create_voronoi_mesh(subdivisions_for(T.shape[0]))
    lat = np.asarray(mesh.latCell) * 180.0 / np.pi
    lon = (np.asarray(mesh.lonCell) * 180.0 / np.pi + 180.0) % 360.0 - 180.0

    cwv = np.asarray(column_water_vapor(q_v, p_s, dsigma))
    u_east, _v_north = reconstruct_cell_velocity(u_edge, mesh)
    u_zonal = np.asarray(u_east)

    sigma_mid = np.cumsum(dsigma) - 0.5 * dsigma
    lat_u, prof_u = zonal_mean(u_zonal, lat)
    lat_t, prof_t = zonal_mean(T, lat)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{run_dir.name} — AMIP MPAS day {sim_day}", fontsize=13)

    s = ax[0, 0].scatter(lon, lat, c=T[:, -1] - constants.T_freeze, s=4,
                         cmap="RdBu_r", vmin=-40, vmax=40)
    ax[0, 0].set_title("near-surface T [degC]")
    fig.colorbar(s, ax=ax[0, 0])

    s = ax[0, 1].scatter(lon, lat, c=cwv, s=4, cmap="viridis",
                         vmin=0, vmax=60)
    ax[0, 1].set_title("column water vapour [kg/m2]")
    fig.colorbar(s, ax=ax[0, 1])

    for a in (ax[0, 0], ax[0, 1]):
        a.set_xlim(-180, 180)
        a.set_ylim(-90, 90)
        a.set_xlabel("lon")
        a.set_ylabel("lat")

    lim = float(np.abs(prof_u).max()) or 1.0
    c = ax[1, 0].contourf(lat_u, sigma_mid, prof_u.T, levels=21,
                          cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_title("zonal-mean zonal wind [m/s]")
    ax[1, 0].set_xlabel("lat")
    ax[1, 0].set_ylabel("sigma")
    fig.colorbar(c, ax=ax[1, 0])

    c = ax[1, 1].contourf(lat_t, sigma_mid, prof_t.T, levels=21, cmap="magma")
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title("zonal-mean T [K]")
    ax[1, 1].set_xlabel("lat")
    ax[1, 1].set_ylabel("sigma")
    fig.colorbar(c, ax=ax[1, 1])

    fig.tight_layout()
    out = out or run_dir / f"snapshot_day_{sim_day:04d}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"day {sim_day}: T_sfc [{T[:, -1].min():.1f}, {T[:, -1].max():.1f}] K"
          f" | CWV mean {cwv.mean():.1f} kg/m2"
          f" | u_zonal max {np.abs(u_zonal).max():.1f} m/s")
    print(f"wrote {out}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--day", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if not a.run_dir.is_dir():
        raise SystemExit(f"no such run directory: {a.run_dir}")
    build_figure(a.run_dir, a.day, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
