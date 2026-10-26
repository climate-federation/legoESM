"""Map WHERE the eORCA1 zonal-metric misalignment perturbed continuity.

The defect is already measured and fixed (15c6670ae): the mesh loader gave each
u-face the meridional length of the face one column EAST.  The numbers say it
is exactly zero in the southern mid-latitudes and reaches tens of percent of
the local divergence north of 30N.  This draws that, because a latitude-band
table does not show which SEAS were affected.

Both metrics are built here -- the OLD append-first-column form and the CURRENT
prepend-last-column form -- and the model's own ``divergence_cgrid`` is run on
the production restart with each.  The difference is the error the old build
injected into the continuity equation for the state the model actually carried.

Panels: the absolute difference, the same relative to the local divergence, and
the along-row metric variation that drives it.
"""
from __future__ import annotations

import numpy as np

MESH = ("/burg-archive/glab/users/pg2328/legoESM/data/grids/"
        "eORCA1.2_mesh_mask.nc")
RESTART = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
           "trp_orca1faithtint_d30/restart_leg2.npz")
OUT = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
       "figs_metric_error")


def main() -> int:
    import os

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.colors import LogNorm

    from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
    from legoesm.grids.tripole import DEFAULT_MIN_DX_M, create_tripole_grid

    os.makedirs(OUT, exist_ok=True)
    grid = create_tripole_grid(MESH)
    dy_now = np.asarray(grid.dy_u)
    area_t = np.asarray(grid.area_T, dtype=np.float64)
    n_lat, n_lon = area_t.shape
    lat = np.degrees(np.asarray(grid.lat_T, dtype=np.float64))
    lon = np.degrees(np.asarray(grid.lon_T, dtype=np.float64)) if hasattr(
        grid, "lon_T") else None

    dm = xr.open_dataset(MESH, decode_times=False)
    e2u = np.asarray(dm["e2u"].values)
    while e2u.ndim > 2:
        e2u = e2u[0]
    # the OLD build: identity plus the FIRST column appended as the wrap
    dy_old = np.concatenate([e2u, e2u[:, :1]], axis=1)
    if DEFAULT_MIN_DX_M > 0.0:
        dy_old = np.maximum(dy_old, DEFAULT_MIN_DX_M)
    dy_old = dy_old.astype(dy_now.dtype)

    z = np.load(RESTART, allow_pickle=True)
    u, v = jnp.asarray(z["u"]), jnp.asarray(z["v"])

    for name in ("_replace", "replace"):
        fn = getattr(grid, name, None)
        if callable(fn):
            grid_old = fn(dy_u=jnp.asarray(dy_old))
            break
    else:
        import dataclasses
        grid_old = dataclasses.replace(grid, dy_u=jnp.asarray(dy_old))

    d_now = np.asarray(divergence_cgrid(u, v, grid), dtype=np.float64)
    d_old = np.asarray(divergence_cgrid(u, v, grid_old), dtype=np.float64)
    if d_now.ndim == 3:
        d_now, d_old = d_now[..., 0], d_old[..., 0]
    delta = np.abs(d_old - d_now)
    rel = np.where(np.abs(d_now) > 0, delta / np.abs(d_now), np.nan)
    var = np.abs(np.roll(e2u, -1, axis=1) - e2u) / np.maximum(e2u, 1.0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    for ax, field, title, norm in (
        (axes[0], np.where(delta > 0, delta, np.nan),
         "Error the old metric injected into continuity  [1/s]",
         LogNorm(vmin=1e-12, vmax=max(1e-11, np.nanmax(delta)))),
        (axes[1], np.where(np.isfinite(rel) & (rel > 0), rel, np.nan),
         "Same, as a fraction of the local divergence",
         LogNorm(vmin=1e-6, vmax=1.0)),
        (axes[2], np.where(var > 0, var, np.nan),
         "Along-row variation of the face metric (the driver)",
         LogNorm(vmin=1e-8, vmax=1.0)),
    ):
        im = ax.pcolormesh(field, norm=norm, cmap="magma", shading="auto")
        # latitude guides: the bands the table quoted
        for target, style in ((30.0, "-"), (-30.0, "--"), (-60.0, "--")):
            rows = np.argmin(np.abs(lat - target), axis=0)
            ax.plot(np.arange(n_lon), rows, style, color="cyan", lw=1.0,
                    alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("i"); ax.set_ylabel("j")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("eORCA1 zonal-metric misalignment: where it actually bit\\n"
                 "cyan: 30N (solid), 30S and 60S (dashed)", y=0.995)
    fig.tight_layout()
    path = os.path.join(OUT, "tripole_metric_error_map.png")
    fig.savefig(path, dpi=130)
    print(f"[fig] wrote {path}")
    print(f"[fig] max absolute error {np.nanmax(delta):.3e} 1/s; "
          f"cells affected {int((delta > 0).sum())} of {delta.size}")
    for lo, hi in ((-90, -60), (-60, -30), (-30, 30), (30, 60), (60, 90)):
        m = (lat >= lo) & (lat < hi) & np.isfinite(rel)
        if m.any():
            print(f"[fig] {lo:+4d}..{hi:+4d}  median rel "
                  f"{np.nanmedian(rel[m]):.3e}  p99 {np.nanpercentile(rel[m], 99):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
