"""Diagnostic 3 for the ETOPO 30-day instability.

Re-runs the linear-T frozen-T case (or z-only T-init frozen-T with
``--zonly-T-init``) and analyzes the spatial structure of the day-30
|u| field to distinguish:

- **Localized at coastal partial-cell step boundaries** → SMC03 /
  momentum stencil bug at lateral steps (dycore-expert prediction).
- **Distributed across the abyssal ocean** → C-grid topographic
  computational mode (Mesinger 1973; ocean-expert prediction).

3a: ``--zonly-T-init`` strips the curvature contribution
(d²ρ/dz² ≈ 0 at fixed reference depths).  If the v-direction
step-face concentration persists in this regime, the dominant
bug isn't the bottom-cell σ × curvature — it's a structural
asymmetry in the y-PGF or momentum stencil.

3b: stats are reported BOTH including pole rows AND excluding the
near-pole band (|lat| > 80°), to test whether the step-localized
signal is purely a high-latitude grid-convergence artefact or
domain-wide.

Outputs:
- ``umax_map.png``: |u|max(lat, lon) over levels with bot_level
  contour overlay and step-face dots.
- ``step_vs_interior.txt``: full vs no-pole RMS comparisons.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


N_LAT, N_LON = 60, 120
N_LEVELS = 20
H_MAX = 5000.0
DT = 600.0
DAYS = 30.0
STEPS_PER_DAY = int(86400.0 / DT)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--zonly-T-init", action="store_true",
        help="3a: use z-only T(k) instead of linear T(z) — strips "
        "curvature in ρ at the partial-bottom σ stencil.",
    )
    args = parser.parse_args()
    tag = "zonlyT_frozenTS" if args.zonly_T_init else "linearTz_frozenTS"
    out_dir = Path(f"results/realistic_geometry_validation/diag3_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = create_latlon_grid(N_LAT, N_LON)
    z_coord_base = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX, dz_surface=20.0, dz_deep=500.0,
    )

    bathy_cfg = BathymetryConfig(
        source="file", path="data/bathymetry/etopo_1deg.nc",
        H_max=H_MAX, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True,
    )
    H_bathy, ocean_mask = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy.astype(jnp.float32)
    ocean_mask = ocean_mask.astype(jnp.float32)
    z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_bathy_override=H_bathy, land_mask_override=ocean_mask,
    )

    # T-init: linear T(z) by default (3 / partial original); z-only T(k)
    # with --zonly-T-init for 3a.
    T_deep, T_surf = 2.0, 20.0
    if args.zonly_T_init:
        # T(k) from z_full_ref (column-independent; cell-mean T at level
        # k identical across columns; linear in z so curvature ≈ 0).
        z_abs = jnp.abs(z_coord_base.z_full_ref).astype(jnp.float32)
        T_1d = T_surf + (T_deep - T_surf) * z_abs / H_MAX
        T_per_cell = jnp.broadcast_to(
            T_1d[jnp.newaxis, jnp.newaxis, :],
            (N_LAT, N_LON, N_LEVELS),
        )
    else:
        # Linear T(z), centroid-aware.
        centroid = compute_centroid_depth(
            jnp.zeros_like(H_bathy), H_bathy, z_coord,
        )
        T_per_cell = T_surf + (T_deep - T_surf) * centroid / H_MAX
    T_per_cell = jnp.where(z_coord.is_active, T_per_cell, T_deep)
    T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
    )

    cfg = LatLonCGridOceanConfig.from_flat(
        barotropic_solver="implicit_cn",
        physics=None,
        bottom_drag_r=1.0e-3,
        bottom_drag_bbl_thickness=100.0,
        pgf_scheme="smc03",
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)

    # Frozen-T scan body
    initial_T = state.T.data
    initial_S = state.S.data

    def scan_body(state, _):
        new_state = model.step(state, DT)
        new_state = new_state._replace(
            T=new_state.T.replace(data=initial_T),
            S=new_state.S.replace(data=initial_S),
        )
        return new_state, None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    # Integrate 30 days
    n_total = int(DAYS * STEPS_PER_DAY)
    print(f"Integrating {DAYS:.0f} days ({n_total} steps, dt={DT}s) ...")
    t0 = time.time()
    s = block_fn(state, n_total)
    jax.block_until_ready(s.eta.data)
    print(f"  Done in {time.time() - t0:.0f}s")
    print(f"  Final |u|max = {float(jnp.max(jnp.abs(s.u.data)))*1000:.3f} mm/s")

    # ----------------------------------------------------------------
    # Spatial analysis
    # ----------------------------------------------------------------
    u = np.asarray(s.u.data)               # (n_lat, n_lon+1, nlev)
    v = np.asarray(s.v.data)               # (n_lat+1, n_lon, nlev)
    bot_level = np.asarray(z_coord.bottom_level)
    H_np = np.asarray(H_bathy)
    om = np.asarray(ocean_mask)

    # u-face bot-level mismatch indicator (interior u-faces, j=0..n_lon-1).
    bot_W = np.roll(bot_level, 1, axis=1)
    u_face_step = (bot_level != bot_W) & (om > 0.5) & (np.roll(om, 1, axis=1) > 0.5)
    # v-face bot-level mismatch (interior, i=1..n_lat-1).
    v_face_step = np.zeros((N_LAT - 1, N_LON), dtype=bool)
    v_face_step = (bot_level[1:] != bot_level[:-1]) & \
                  (om[1:] > 0.5) & (om[:-1] > 0.5)

    # Drop the periodic-wrap face from u for stats, drop pole rows from v.
    u_int = u[:, :-1, :]                   # (n_lat, n_lon, nlev)
    v_int = v[1:-1, :, :]                  # (n_lat-1, n_lon, nlev)

    # Wet-wet face masks (per level k uses bot_level >= k).
    k_idx = np.arange(N_LEVELS)
    is_active_3d = (k_idx[None, None, :] <= bot_level[:, :, None])  # (n_lat,n_lon,nlev)
    u_face_active_3d = is_active_3d & np.roll(is_active_3d, 1, axis=1)
    v_face_active_3d = is_active_3d[1:] & is_active_3d[:-1]

    u_face_step_3d = u_face_step[:, :, None] & u_face_active_3d
    u_face_int_3d = (~u_face_step[:, :, None]) & u_face_active_3d
    v_face_step_3d = v_face_step[:, :, None] & v_face_active_3d
    v_face_int_3d = (~v_face_step[:, :, None]) & v_face_active_3d

    u_sq = u_int ** 2
    v_sq = v_int ** 2

    # |u/v|max location and whether it sits on a step face.
    u_abs = np.abs(u_int)
    v_abs = np.abs(v_int)
    iu_max = np.unravel_index(np.argmax(u_abs), u_abs.shape)
    iv_max = np.unravel_index(np.argmax(v_abs), v_abs.shape)

    lat_deg_full = np.asarray(grid.lat) * 180.0 / np.pi   # (n_lat,)
    POLE_BAND = 80.0

    def stats(u_lat_mask, v_lat_mask, label):
        # u_lat_mask: (n_lat,) bool — which lat rows to include for u.
        # v_lat_mask: (n_lat-1,) bool — which v-face interior lat rows.
        u_inc = u_lat_mask[:, None, None]
        v_inc = v_lat_mask[:, None, None]
        u_step_mask = u_face_step_3d & u_inc
        u_int_mask = u_face_int_3d & u_inc
        v_step_mask = v_face_step_3d & v_inc
        v_int_mask = v_face_int_3d & v_inc

        nu_s, nu_i = int(u_step_mask.sum()), int(u_int_mask.sum())
        nv_s, nv_i = int(v_step_mask.sum()), int(v_int_mask.sum())
        rms = lambda m, x2: (np.sqrt(np.sum(x2[m]) / max(int(m.sum()), 1)) * 1000)
        ru_s = rms(u_step_mask, u_sq)
        ru_i = rms(u_int_mask, u_sq)
        rv_s = rms(v_step_mask, v_sq)
        rv_i = rms(v_int_mask, v_sq)
        return [
            f"\n--- {label} ---\n",
            f"u-faces  step/interior: {nu_s} / {nu_i}  "
            f"(step fraction = {nu_s/max(nu_s+nu_i,1):.3f})\n",
            f"v-faces  step/interior: {nv_s} / {nv_i}  "
            f"(step fraction = {nv_s/max(nv_s+nv_i,1):.3f})\n",
            f"RMS |u| (mm/s):  step = {ru_s:.4f}   interior = {ru_i:.4f}   "
            f"ratio = {ru_s/max(ru_i,1e-12):.2f}\n",
            f"RMS |v| (mm/s):  step = {rv_s:.4f}   interior = {rv_i:.4f}   "
            f"ratio = {rv_s/max(rv_i,1e-12):.2f}\n",
        ]

    u_full_mask = np.ones(N_LAT, dtype=bool)
    v_full_mask = np.ones(N_LAT - 1, dtype=bool)

    nopole_lat_mask = np.abs(lat_deg_full) <= POLE_BAND
    # v-face i is between cell i-1 and cell i (interior); both must be non-pole.
    v_lat_centers = 0.5 * (lat_deg_full[:-1] + lat_deg_full[1:])
    nopole_v_mask = np.abs(v_lat_centers) <= POLE_BAND

    label_full = "FULL DOMAIN (incl. poles)"
    label_nopole = f"|lat| ≤ {POLE_BAND:.0f}° (excludes pole rows)"

    report = []
    report.append("Diagnostic 3: spatial concentration of frozen-T day-30 |u|\n")
    report.append("=" * 72 + "\n")
    report.append(f"Mode: {'z-only T-init (3a — kills σ × curvature)' if args.zonly_T_init else 'linear T(z)'}\n")
    report.extend(stats(u_full_mask, v_full_mask, label_full))
    report.extend(stats(nopole_lat_mask, nopole_v_mask, label_nopole))
    report.append("\n")
    report.append(f"|u|max location: i={iu_max[0]} (lat={lat_deg_full[iu_max[0]]:.1f}°), "
                  f"j={iu_max[1]}, k={iu_max[2]}, |u|={u_abs[iu_max]*1000:.3f} mm/s, "
                  f"step-face={bool(u_face_step_3d[iu_max])}\n")
    report.append(f"|v|max location: i={iv_max[0]} (v-face lat={v_lat_centers[iv_max[0]]:.1f}°), "
                  f"j={iv_max[1]}, k={iv_max[2]}, |v|={v_abs[iv_max]*1000:.3f} mm/s, "
                  f"step-face={bool(v_face_step_3d[iv_max])}\n")

    report_str = "".join(report)
    print()
    print(report_str)
    (out_dir / "step_vs_interior.txt").write_text(report_str)
    print(f"Saved {out_dir / 'step_vs_interior.txt'}")

    # Spatial map: depth-max |u| (over levels) at each cell-center
    u_cell = 0.5 * (u_int + np.roll(u_int, -1, axis=1))   # cell-centered u
    v_cell = 0.5 * (v_int + np.roll(v_int, -1, axis=0)) if v_int.shape[0] > 1 else v_int
    speed_cell = np.sqrt(u_cell ** 2 + v_cell[:N_LAT-1].mean(axis=0, keepdims=True).repeat(N_LAT, axis=0)[:u_cell.shape[0]] ** 2 if False else u_cell ** 2)
    # Simpler: just |u| max over depth at each (i, j) on the u-grid.
    u_max_over_z = np.max(np.abs(u_int), axis=-1) * 1000  # mm/s, (n_lat, n_lon)

    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    fig, ax = plt.subplots(figsize=(11, 5))
    speed_plot = np.where(om > 0.5, u_max_over_z, np.nan)
    im = ax.pcolormesh(
        lon_deg, lat_deg, speed_plot,
        cmap="hot_r", shading="auto", vmin=0, vmax=10,
    )
    plt.colorbar(im, ax=ax, label="max|u| over depth (mm/s)", fraction=0.025)

    # Overlay bot_level contours (every 5 levels)
    cs = ax.contour(
        lon_deg, lat_deg, np.where(om > 0.5, bot_level, np.nan),
        levels=[3, 5, 8, 12, 15, 18], colors="cyan",
        linewidths=0.5, alpha=0.7,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="bot=%g")
    # Mark step faces (where adjacent columns have different bot_level)
    step_lat, step_lon = np.where(u_face_step)
    ax.scatter(
        lon_deg[step_lon], lat_deg[step_lat],
        s=2, color="cyan", alpha=0.4, label=f"u-face bot-step (n={len(step_lat)})",
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"Diag-3 frozen-T linear-T(z) — depth-max |u| at day {DAYS:.0f}\n"
        f"Cyan dots = bot_level step faces.  RMS step/interior = "
        f"see step_vs_interior.txt for ratios"
    )
    ax.legend(loc="lower left", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "umax_map.png", dpi=130)
    plt.close()
    print(f"Saved {out_dir / 'umax_map.png'}")


if __name__ == "__main__":
    main()
