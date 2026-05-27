"""Geostrophic adjustment with logarithmic-in-time snapshots.

Runs lat-lon 36x72 and MPAS ico4 at matched config (A_h=1e4, Wright EOS,
TVD tracer advection), recording SSH at t = {0, 2h, 6h, 12h, 1d, 2d, 4d,
7d, 10d}. Saves NPZ + slide-ready PNG film-strips.
"""

from __future__ import annotations

import os

# Must set before importing jax/legoesm — module-level constants get cached.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.experiments.geostrophic_adjustment import (
    create_initial_conditions as ga_ic,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

OUT_DIR = Path("results/ocean/slide28_validation/geostrophic_adjustment_logtime")
DT = 300.0                  # seconds
TOTAL_DAYS = 10.0
SNAPSHOT_DAYS = [0.0, 2 / 24, 6 / 24, 12 / 24, 1.0, 2.0, 4.0, 7.0, 10.0]
NLEV = 10
H_MAX = 5500.0


def _snap_steps(snapshot_days, dt):
    steps = sorted({0} | {int(round(d * 86400.0 / dt)) for d in snapshot_days})
    return steps


def _promote_state_to_f64(state):
    """Promote all float32 leaves of a state pytree to float64.

    The IC builder leaves most fields as float32 even under JAX_ENABLE_X64=1.
    The matrix runner's ``_run_timeloop`` doesn't hit this because the very
    first ``model.step`` happens before a sharded scan; here we call
    ``model.step`` in a Python loop so the dtype mismatch surfaces
    immediately. Cast explicitly to keep the carry dtypes self-consistent.
    """
    def _cast(x):
        if hasattr(x, "dtype") and x.dtype == jnp.float32:
            return x.astype(jnp.float64)
        return x
    return jax.tree.map(_cast, state)


def run_latlon():
    grid = create_latlon_grid(36, 72)
    z = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    cfg = LatLonCGridOceanConfig(n_barotropic_substeps=30)
    model = LatLonCGridOceanModel(grid, z, cfg)
    state = ga_ic("latlon", grid, z)
    state = _promote_state_to_f64(state)
    return _run("latlon", model, state, grid)


def run_mpas():
    mesh = create_voronoi_mesh(4)
    z = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    cfg = MPASOceanConfig(n_barotropic_substeps=30, tracer_advection="tvd")
    model = MPASOceanModel(mesh, z, cfg)
    state = ga_ic("mpas", mesh, z)
    state = _promote_state_to_f64(state)
    return _run("mpas", model, state, mesh)


def _arr(x):
    """Unwrap Field → ndarray; otherwise asarray."""
    if hasattr(x, "data"):
        return np.asarray(x.data)
    return np.asarray(x)


def _extract(state, grid_type, grid):
    eta = _arr(state.eta if hasattr(state, "eta") else state.eta_field)
    if grid_type == "latlon":
        lon = np.asarray(grid.lon) * 180 / np.pi
        lat = np.asarray(grid.lat) * 180 / np.pi
        # u on western faces, shape (nlat, nlon+1, nlev) — average adjacent
        # faces to cell centres for plotting. Take surface level (k=0).
        u_raw = _arr(state.u)[..., 0]                 # (nlat, nlon+1)
        u_cell = 0.5 * (u_raw[:, :-1] + u_raw[:, 1:])  # (nlat, nlon)
        u = np.asarray(u_cell)
    else:  # mpas
        lon = np.asarray(grid.lonCell) * 180 / np.pi
        lat = np.asarray(grid.latCell) * 180 / np.pi
        # Perot reconstruction: edge-normal u_edge → (u_east, v_north) at cell
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity
        u_edge_sfc = _arr(state.u)[..., 0]            # (nEdges,)
        u_east, _v_north = reconstruct_cell_velocity(u_edge_sfc, grid)
        u = np.asarray(u_east)
    lm = _arr(state.land_mask) if hasattr(state, "land_mask") else None
    return eta, u, lon, lat, lm


def _run(grid_type, model, state, grid):
    n_total = int(TOTAL_DAYS * 86400.0 / DT)
    targets = set(_snap_steps(SNAPSHOT_DAYS, DT))
    eta_snaps, u_snaps, steps = [], [], []
    eta0, u0, lon, lat, lm = _extract(state, grid_type, grid)
    eta_snaps.append(eta0); u_snaps.append(u0); steps.append(0)
    print(f"[{grid_type}] running {n_total} steps, dt={DT}s, "
          f"snapshots at steps {sorted(targets)}")
    for i in range(n_total):
        state = model.step(state, DT)
        step = i + 1
        if step in targets:
            eta_i, u_i, _, _, _ = _extract(state, grid_type, grid)
            eta_snaps.append(eta_i); u_snaps.append(u_i); steps.append(step)
            print(f"  [{grid_type}] step {step} (t={step*DT/86400:.3f} d)")
    times_days = np.array([s * DT / 86400.0 for s in steps])
    return {
        "grid_type": grid_type,
        "lon": lon, "lat": lat,
        "land_mask": lm,
        "times_days": times_days,
        "eta": np.stack(eta_snaps),
        "u": np.stack(u_snaps),
    }


def save_npz(data, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             lon=data["lon"], lat=data["lat"],
             land_mask=data["land_mask"],
             times_days=data["times_days"],
             eta=data["eta"], u=data["u"])
    print(f"saved {out_path}")


def _regrid_mpas_to_latlon(mpas_data, n_lat=181, n_lon=360):
    """Cheap nearest-neighbour regrid from Voronoi cells to a regular lat-lon
    grid for plotting. Only used for visualisation."""
    from scipy.spatial import cKDTree

    src_lon = np.asarray(mpas_data["lon"])
    src_lat = np.asarray(mpas_data["lat"])
    # Convert to unit-sphere XYZ for nearest-neighbour
    sl = np.deg2rad(src_lon); st = np.deg2rad(src_lat)
    src_xyz = np.stack([np.cos(st) * np.cos(sl),
                        np.cos(st) * np.sin(sl),
                        np.sin(st)], axis=1)
    tree = cKDTree(src_xyz)
    tgt_lat = np.linspace(-90, 90, n_lat)
    tgt_lon = np.linspace(0, 360, n_lon, endpoint=False)
    LON, LAT = np.meshgrid(tgt_lon, tgt_lat)
    tl = np.deg2rad(LON.ravel()); tt = np.deg2rad(LAT.ravel())
    tgt_xyz = np.stack([np.cos(tt) * np.cos(tl),
                        np.cos(tt) * np.sin(tl),
                        np.sin(tt)], axis=1)
    _, idx = tree.query(tgt_xyz, k=1)
    lm_native = mpas_data["land_mask"]
    lm_regrid = lm_native[idx].reshape(n_lat, n_lon)
    eta_regrid = np.stack([
        mpas_data["eta"][t][idx].reshape(n_lat, n_lon)
        for t in range(mpas_data["eta"].shape[0])
    ])
    u_regrid = np.stack([
        mpas_data["u"][t][idx].reshape(n_lat, n_lon)
        for t in range(mpas_data["u"].shape[0])
    ])
    return {
        "lon": tgt_lon, "lat": tgt_lat,
        "land_mask": lm_regrid,
        "times_days": mpas_data["times_days"],
        "eta": eta_regrid,
        "u": u_regrid,
        "grid_type": "mpas_regridded",
    }


def _filmstrip_field(latlon, mpas_ll, field, cmap, label_unit, out_path, title):
    """Generic 2-row (lat-lon, MPAS) × N-times film-strip of a given field."""
    import matplotlib.pyplot as plt

    times = latlon["times_days"]
    n = len(times)
    ll = np.where(latlon["land_mask"][None, :, :] > 0.5,
                  latlon[field], np.nan)
    mp = np.where(mpas_ll["land_mask"][None, :, :] > 0.5,
                  mpas_ll[field], np.nan)
    vmax = max(np.nanmax(np.abs(ll)), np.nanmax(np.abs(mp)))

    fig, axes = plt.subplots(2, n, figsize=(2.0 * n + 1.2, 4.6),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    for j in range(n):
        for i, (data, fld, gridlabel) in enumerate([
            (latlon, ll, "lat-lon\n36×72"),
            (mpas_ll, mp, "MPAS ico4\n(~445 km)"),
        ]):
            ax = axes[i, j]
            im = ax.pcolormesh(data["lon"], data["lat"], fld[j],
                               cmap=cmap, vmin=-vmax, vmax=vmax,
                               shading="auto")
            if i == 0:
                t = times[j]
                t_str = f"{t * 24:.0f} h" if t < 1.0 else f"{t:.0f} d"
                ax.set_title(t_str, fontsize=9)
            if j == 0:
                ax.set_ylabel(gridlabel, fontsize=9)
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_xticks([])
            ax.set_yticks([-60, 0, 60] if j == 0 else [])
    fig.colorbar(im, ax=axes, shrink=0.7, label=label_unit)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def _filmstrip_eta_u_combined(latlon, mpas_ll, out_path):
    """4-row (lat-lon η, MPAS η, lat-lon u, MPAS u) × N-times film-strip."""
    import matplotlib.pyplot as plt

    times = latlon["times_days"]
    n = len(times)

    ll_eta = np.where(latlon["land_mask"][None, :, :] > 0.5,
                      latlon["eta"], np.nan)
    mp_eta = np.where(mpas_ll["land_mask"][None, :, :] > 0.5,
                      mpas_ll["eta"], np.nan)
    ll_u = np.where(latlon["land_mask"][None, :, :] > 0.5,
                    latlon["u"], np.nan)
    mp_u = np.where(mpas_ll["land_mask"][None, :, :] > 0.5,
                    mpas_ll["u"], np.nan)

    eta_vmax = max(np.nanmax(np.abs(ll_eta)), np.nanmax(np.abs(mp_eta)))
    u_vmax = max(np.nanmax(np.abs(ll_u)), np.nanmax(np.abs(mp_u)))

    fig, axes = plt.subplots(4, n, figsize=(2.0 * n + 1.4, 8.2),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    rows = [
        (latlon, ll_eta, "η  lat-lon", "RdBu_r", eta_vmax),
        (mpas_ll, mp_eta, "η  MPAS", "RdBu_r", eta_vmax),
        (latlon, ll_u, "u  lat-lon", "PuOr_r", u_vmax),
        (mpas_ll, mp_u, "u  MPAS", "PuOr_r", u_vmax),
    ]
    ims = {}
    for j in range(n):
        for i, (data, fld, ylabel, cmap, vmax) in enumerate(rows):
            ax = axes[i, j]
            im = ax.pcolormesh(data["lon"], data["lat"], fld[j],
                               cmap=cmap, vmin=-vmax, vmax=vmax,
                               shading="auto")
            ims[i] = im
            if i == 0:
                t = times[j]
                t_str = f"{t * 24:.0f} h" if t < 1.0 else f"{t:.0f} d"
                ax.set_title(t_str, fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_xticks([])
            ax.set_yticks([-60, 0, 60] if j == 0 else [])
    # Two colorbars — one for η, one for u.
    fig.colorbar(ims[0], ax=axes[:2, :].ravel().tolist(),
                 shrink=0.8, label="SSH η (m)", location="right")
    fig.colorbar(ims[2], ax=axes[2:, :].ravel().tolist(),
                 shrink=0.8, label="surface u (m/s)", location="right")
    fig.suptitle(
        "Geostrophic adjustment — SSH and surface zonal velocity "
        "(log-spaced snapshots)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_filmstrip(latlon, mpas, out_path):
    mpas_ll = _regrid_mpas_to_latlon(mpas)
    _filmstrip_field(
        latlon, mpas_ll, "eta", "RdBu_r", "SSH η (m)",
        out_path.parent / "slide28_logtime_filmstrip.png",
        "Geostrophic adjustment — SSH evolution (log-spaced snapshots)",
    )
    _filmstrip_field(
        latlon, mpas_ll, "u", "PuOr_r", "surface u (m/s)",
        out_path.parent / "slide28_logtime_filmstrip_u.png",
        "Geostrophic adjustment — surface zonal velocity u (log-spaced snapshots)",
    )
    _filmstrip_eta_u_combined(
        latlon, mpas_ll, out_path.parent / "slide28_logtime_filmstrip_eta_u.png"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ll = run_latlon()
    save_npz(ll, OUT_DIR / "latlon_36x72.npz")
    mp = run_mpas()
    save_npz(mp, OUT_DIR / "mpas_ico4.npz")
    plot_filmstrip(ll, mp, OUT_DIR / "slide28_logtime_filmstrip.png")


if __name__ == "__main__":
    main()
