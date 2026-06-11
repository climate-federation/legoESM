#!/usr/bin/env python
"""Lat-lon spatial-map + zonal-mean plotter for AMIP ``snapshots.npz``.

The driver's snapshot accumulator writes a handful of 2-D surface /
low-level fields per snapshot day (``dayNNN_<field>`` keys, each
shape ``(n_lat, n_lon)``).  ``scripts/plot/plot_amip.py`` covers
the scalar timeseries + global-mean vertical profiles; this companion
covers the *spatial* realism check the loop task needs — lat-lon
gradients, Hadley/ITCZ precip banding, equator-pole temperature
contrast, and grid-imprint artifacts — which scalar norms cannot see
(CLAUDE.md: "Visual verify spatial/grid artifacts").

Grid reconstruction matches ``legoesm.grids.latlon.create_latlon_grid``
exactly: cell-center latitudes ``linspace(-90+dlat/2, 90-dlat/2, n_lat)``
with ``dlat = 180/n_lat`` and ``n_lon = 2*n_lat``.  No saturation curves
are re-derived here (raw model fields only); the only physical constant
used is ``constants.T_freeze`` for the K->°C plot-axis label.

Usage::

    JAX_PLATFORMS=cpu python scripts/plot/plot_amip_snapshot_maps.py \
        --run results/amip_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from legoesm import constants  # noqa: E402


# Fields we know how to render, with display metadata.  Keys are the
# ``<field>`` suffix after the ``dayNNN_`` prefix.
FIELD_META = {
    "SST": dict(title="SST [°C]", cmap="RdYlBu_r", to_celsius=True),
    "T_sfc": dict(title="Surface T [°C]", cmap="RdYlBu_r", to_celsius=True),
    "T_low": dict(title="Low-level T [°C]", cmap="RdYlBu_r", to_celsius=True),
    # Driver already stores low-level moisture in g/kg (raw q_v_low ~3-27).
    "q_v_low": dict(title="Low-level q_v [g/kg]", cmap="YlGnBu"),
    "q_c_low": dict(title="Low-level q_c [g/kg]", cmap="Blues"),
    "q_r_low": dict(title="Low-level q_r [g/kg]", cmap="Purples"),
    "precip": dict(title="Precip [mm/day]", cmap="viridis"),
    "wind": dict(title="Low-level wind speed [m/s]", cmap="magma"),
    "SIC": dict(title="Sea-ice conc [-]", cmap="Blues"),
}


def _latlon_axes(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center lat/lon in degrees, matching create_latlon_grid."""
    dlat = 180.0 / n_lat
    lat = np.linspace(-90.0 + dlat / 2.0, 90.0 - dlat / 2.0, n_lat)
    dlon = 360.0 / n_lon
    lon = np.linspace(0.0, 360.0 - dlon, n_lon)
    return lat, lon


def _latest_day(keys) -> int:
    days = set()
    for k in keys:
        if k.startswith("day") and "_" in k:
            tok = k[3:].split("_", 1)[0]
            if tok.isdigit():
                days.add(int(tok))
    if not days:
        raise SystemExit("ERROR: no dayNNN_<field> snapshot keys found.")
    return max(days)


def _prep(field: str, arr: np.ndarray) -> np.ndarray:
    meta = FIELD_META.get(field, {})
    out = np.asarray(arr, dtype=np.float64)
    if meta.get("to_celsius"):
        out = out - constants.T_freeze
    if "scale" in meta:
        out = out * meta["scale"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run dir with snapshots.npz")
    p.add_argument("--day", type=int, default=None,
                   help="snapshot day to plot (default: latest)")
    p.add_argument("--out", default=None, help="output PNG path")
    args = p.parse_args(argv)

    run = Path(args.run)
    snap_path = run / "snapshots.npz"
    if not snap_path.exists():
        print(f"ERROR: {snap_path} not found.  Under MPI, snapshots are "
              f"skipped unless diagnostics_perf_mode='never' or "
              f"--cmip-output.", file=sys.stderr)
        return 1

    d = np.load(snap_path, allow_pickle=True)
    day = args.day if args.day is not None else _latest_day(d.files)
    prefix = f"day{day:03d}_"

    present = [(f, k) for f in FIELD_META
               for k in (prefix + f,) if k in d.files]
    if not present:
        print(f"ERROR: no known fields for day {day}.  Keys: {list(d.files)}",
              file=sys.stderr)
        return 1

    sample = d[present[0][1]]
    n_lat, n_lon = sample.shape
    lat, lon = _latlon_axes(n_lat, n_lon)

    n = len(present)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i, (field, key) in enumerate(present):
        r, c = divmod(i, ncol)
        ax = axes[r][c]
        ax.axis("on")
        meta = FIELD_META[field]
        data = _prep(field, d[key])
        im = ax.pcolormesh(lon, lat, data, cmap=meta["cmap"], shading="auto")
        ax.set_title(meta["title"], fontsize=9)
        ax.set_xlabel("lon", fontsize=7)
        ax.set_ylabel("lat", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle(f"AMIP lat-lon snapshot maps — day {day}  "
                 f"({n_lat}×{n_lon})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out) if args.out else run / f"snapshot_maps_day{day:03d}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")

    # Zonal-mean figure: temperature panel + moisture/precip/wind panel.
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    for field, key in present:
        if field in ("SIC", "q_c_low", "q_r_low"):
            continue
        data = _prep(field, d[key])
        zm = data.mean(axis=1)
        target = ax2[0] if field in ("SST", "T_sfc", "T_low") else ax2[1]
        target.plot(lat, zm, label=FIELD_META[field]["title"])
    ax2[0].set_title("Zonal-mean temperature"); ax2[0].set_xlabel("lat")
    ax2[0].set_ylabel("°C"); ax2[0].grid(alpha=0.3); ax2[0].legend(fontsize=7)
    ax2[1].set_title("Zonal-mean moisture / precip / wind")
    ax2[1].set_xlabel("lat"); ax2[1].grid(alpha=0.3); ax2[1].legend(fontsize=7)
    fig2.suptitle(f"AMIP zonal means — day {day}", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = run / f"zonal_means_day{day:03d}.png"
    fig2.savefig(out2, dpi=120)
    plt.close(fig2)
    print(f"Wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
