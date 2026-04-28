#!/usr/bin/env python
"""Continue the 50-yr global overturning + GM/Redi run for 10 more years.

Restarts from
  results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz
and integrates 10 additional years with monthly (~30.4-day) snapshots.

Same model setup as run_global_overturning_50yr_gmredi.py:
  lat-lon 36x72 (5-deg), 20 levels, dt=600s, two-belt wind,
  A_h=2e5, surface-T restoring, GM/Redi triads + Visbeck.

Outputs (results/ocean/global_overturning_10yr_continue/):
  - snapshots.npz: 121 monthly samples of eta, SST, speed_sfc, T_zonal_mean
                   (initial restart + 120 end-of-month samples)
  - time_mean.npz: 10-year time mean of full 3D fields (T, S, u, v, w, eta)
                   accumulated over the 120 end-of-month samples
  - moc.npz: meridional overturning streamfunction from time-mean v [Sv]
  - diagnostics.npz: ~10/year scalar diagnostics
  - plots: timeseries, snapshot evolution, time-mean maps + sections, MOC
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config, create_gm_redi_config,
)


RESTART_PATH = Path(
    "results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz"
)
OUTPUT_DIR = Path("results/ocean/global_overturning_10yr_continue")


def _restore_state_from_restart(template_state, restart_path: Path):
    """Replace the ``data`` of every Field in ``template_state`` with the
    array of the same name in the restart npz.

    The template_state carries the correct Field metadata (dims, units,
    staggering); we only swap in the saved numerical data.
    """
    npz = np.load(restart_path, allow_pickle=False)
    saved_day = float(npz["time_days"])
    saved_step = int(npz["step"])

    new_fields = {}
    for fname in template_state._fields:
        field_obj = getattr(template_state, fname)
        if field_obj is None:
            continue  # T_som / S_som when SOM advection is off
        if not hasattr(field_obj, "data"):
            continue
        if fname not in npz.files:
            raise KeyError(
                f"Restart {restart_path} is missing field '{fname}'"
            )
        new_fields[fname] = field_obj.replace(
            data=jnp.asarray(npz[fname], dtype=field_obj.data.dtype)
        )
    return template_state._replace(**new_fields), saved_day, saved_step


def main():
    # ---- Configuration ----
    total_years = 10
    diag_every_days = 36.5         # ~10 diagnostics per year
    months_per_year = 12
    dt = 600.0                     # 10-min timestep (matches 50yr run)
    n_lat, n_lon = 36, 72
    n_barotropic_substeps = 30

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RESTART_PATH.exists():
        raise FileNotFoundError(
            f"Restart file not found: {RESTART_PATH}\n"
            "Run scripts/run_global_overturning_50yr_gmredi.py first."
        )

    config = GlobalOverturningConfig(use_gm_redi=True)
    gm_redi_cfg = create_gm_redi_config(config)
    assert gm_redi_cfg is not None

    total_days = total_years * 365.0
    n_steps = int(total_days * 86400 / dt)
    diag_every = max(1, int(diag_every_days * 86400 / dt))

    # Monthly cadence: 365/12 = 30.4167 days → 4380 steps at dt=600s.
    # 10 years × 12 months = 120 end-of-month ticks (n_steps/snapshot_every = 120).
    snapshot_every_days = 365.0 / months_per_year
    snapshot_every = int(round(snapshot_every_days * 86400 / dt))
    expected_snaps = n_steps // snapshot_every
    assert expected_snaps == total_years * months_per_year, (
        f"Expected {total_years * months_per_year} monthly snapshots, "
        f"got {expected_snaps}"
    )

    print(f"Global Overturning continuation (10 yr from day 18250, GM/Redi triads ON)")
    print(f"  Restart: {RESTART_PATH}")
    print(f"  Grid: lat-lon {n_lat}x{n_lon} (5-deg)")
    print(f"  Vertical: {config.n_levels} levels, H_max={config.H_max}m")
    print(f"  dt={dt}s, n_steps={n_steps:,}")
    print(f"  Snapshot cadence: every {snapshot_every_days:.3f} d "
          f"({snapshot_every} steps) → {expected_snaps} samples")
    print(f"  Diagnostics every {diag_every_days} days")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # ---- Setup model (same as 50yr script) ----
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels,
        H_max=config.H_max,
        dz_surface=config.dz_surface,
        dz_deep=config.dz_deep,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=n_barotropic_substeps,
        physics=physics,
        A_h=config.A_h,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    template_state = create_initial_conditions("latlon", grid, z_coord, config)
    state, day_offset, _step_offset = _restore_state_from_restart(
        template_state, RESTART_PATH
    )
    print(f"  Restarted state at simulation day {day_offset:.1f} "
          f"(year {day_offset/365.0:.2f})")
    print()

    # ---- Coordinate arrays for diagnostics / plotting ----
    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    dz_ref = np.asarray(z_coord.dz_ref, dtype=np.float64)         # (nlev,)
    z_full_ref = np.asarray(z_coord.z_full_ref, dtype=np.float64) # (nlev,)
    z_half_ref = np.asarray(z_coord.z_half_ref, dtype=np.float64) # (nlev+1,)
    # v-face latitudes (n_lat+1): interfaces between cell centers, plus
    # outer half-cells at the south/north edges.
    dlat_rad = float(grid.dlat)
    lat_v_rad = np.concatenate([
        [lat_deg[0] * np.pi / 180 - 0.5 * dlat_rad],
        0.5 * (np.asarray(grid.lat, dtype=np.float64)[:-1]
               + np.asarray(grid.lat, dtype=np.float64)[1:]),
        [lat_deg[-1] * np.pi / 180 + 0.5 * dlat_rad],
    ])
    lat_v_deg = lat_v_rad * 180 / np.pi
    cos_lat_v = np.cos(np.clip(lat_v_rad, -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
    dlon_rad = float(grid.dlon)
    dx_v = cos_lat_v * float(grid.radius) * dlon_rad              # (n_lat+1,)

    # ---- Diagnostic storage ----
    diag_times = []
    diag_max_speed = []
    diag_mean_sst = []
    diag_mean_T = []
    diag_T_deep = []
    diag_mean_eta = []
    snapshots = {}

    # ---- Time-mean accumulators (over end-of-month ticks) ----
    sum_T = np.zeros_like(np.asarray(state.T.data), dtype=np.float64)
    sum_S = np.zeros_like(np.asarray(state.S.data), dtype=np.float64)
    sum_u = np.zeros_like(np.asarray(state.u.data), dtype=np.float64)
    sum_v = np.zeros_like(np.asarray(state.v.data), dtype=np.float64)
    sum_w = np.zeros_like(np.asarray(state.w.data), dtype=np.float64)
    sum_eta = np.zeros_like(np.asarray(state.eta.data), dtype=np.float64)
    n_mean_samples = 0

    def record_diagnostics(state, day):
        eta = np.asarray(state.eta.data)
        T = np.asarray(state.T.data)
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        mask = np.asarray(state.land_mask.data)

        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        speed = np.sqrt(u_c**2 + v_c**2)

        ocean_mask = mask > 0.5
        sst = T[:, :, 0]
        T_bot = T[:, :, -1]

        diag_times.append(day)
        diag_max_speed.append(float(np.max(speed)))
        diag_mean_sst.append(
            float(np.mean(sst[ocean_mask])) if ocean_mask.any() else 0.0
        )
        diag_mean_T.append(
            float(np.mean(T[ocean_mask])) if ocean_mask.any() else 0.0
        )
        diag_T_deep.append(
            float(np.mean(T_bot[ocean_mask])) if ocean_mask.any() else 0.0
        )
        diag_mean_eta.append(
            float(np.mean(eta[ocean_mask])) if ocean_mask.any() else 0.0
        )

    def save_snapshot(state, day):
        eta = np.asarray(state.eta.data)
        T = np.asarray(state.T.data)
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        snapshots[day] = {
            "eta": eta.copy(),
            "SST": T[:, :, 0].copy(),
            "speed_sfc": np.sqrt(u_c**2 + v_c**2),
            "T_zonal_mean": np.mean(T, axis=1).copy(),
        }

    def accumulate_mean(state):
        nonlocal n_mean_samples
        sum_T[:] += np.asarray(state.T.data)
        sum_S[:] += np.asarray(state.S.data)
        sum_u[:] += np.asarray(state.u.data)
        sum_v[:] += np.asarray(state.v.data)
        sum_w[:] += np.asarray(state.w.data)
        sum_eta[:] += np.asarray(state.eta.data)
        n_mean_samples += 1

    def save_restart_file(state, step, day):
        restart = {
            "step": step,
            "time_days": day,
            "grid_type": "latlon",
        }
        for field_name in state._fields:
            field_obj = getattr(state, field_name)
            if field_obj is not None and hasattr(field_obj, "data"):
                restart[field_name] = np.asarray(field_obj.data)
        fname = OUTPUT_DIR / f"restart_day{int(day):06d}.npz"
        np.savez_compressed(fname, **restart)
        print(f"  Restart saved: {fname}")

    # ---- Initial diagnostics & snapshot (NOT counted in time mean) ----
    record_diagnostics(state, day_offset)
    save_snapshot(state, day_offset)

    # ---- Time integration ----
    print(f"Starting {total_years}-year continuation ({n_steps:,} steps)...")
    t0 = time.time()
    last_print = t0

    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1
        day = day_offset + step * dt / 86400.0

        if step % 100 == 0:
            eta_max = float(jnp.max(jnp.abs(state.eta.data)))
            if not np.isfinite(eta_max) or eta_max > 100.0:
                print(f"\n  BLOWUP at step {step} (day {day:.1f}), "
                      f"eta_max={eta_max}")
                save_restart_file(state, step, day)
                break

        if step % diag_every == 0:
            record_diagnostics(state, day)
            now = time.time()
            if now - last_print > 30:
                yr = (day - day_offset) / 365.0
                elapsed = now - t0
                eta_s = elapsed / yr * total_years - elapsed if yr > 0 else 0
                print(f"  Year {yr:5.2f}/{total_years} | "
                      f"spd={diag_max_speed[-1]:.3f} | "
                      f"SST={diag_mean_sst[-1]:.2f} | "
                      f"T_deep={diag_T_deep[-1]:.2f} | "
                      f"eta={diag_mean_eta[-1]:.2e} | "
                      f"ETA {eta_s/3600:.1f}h", flush=True)
                last_print = now

        if step % snapshot_every == 0:
            save_snapshot(state, day)
            accumulate_mean(state)

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    print(f"\nCompleted in {wall:.0f}s ({wall/3600:.1f}h)")
    print(f"  Time-mean built from {n_mean_samples} monthly samples")

    final_day = day_offset + n_steps * dt / 86400.0
    save_restart_file(state, n_steps, final_day)

    # ---- Save diagnostics ----
    np.savez_compressed(
        OUTPUT_DIR / "diagnostics.npz",
        times=np.array(diag_times),
        max_speed=np.array(diag_max_speed),
        mean_sst=np.array(diag_mean_sst),
        mean_T=np.array(diag_mean_T),
        T_deep=np.array(diag_T_deep),
        mean_eta=np.array(diag_mean_eta),
    )
    print(f"  Diagnostics saved: {OUTPUT_DIR / 'diagnostics.npz'}")

    # ---- Save monthly snapshots ----
    snap_data = {}
    for day_key, fields in snapshots.items():
        for fname, arr in fields.items():
            snap_data[f"{fname}_day{int(day_key):06d}"] = arr
    snap_data["snapshot_days"] = np.array(sorted(snapshots.keys()))
    np.savez_compressed(OUTPUT_DIR / "snapshots.npz", **snap_data)
    print(f"  Snapshots saved: {OUTPUT_DIR / 'snapshots.npz'} "
          f"({len(snapshots)} samples)")

    # ---- Compute time means ----
    if n_mean_samples == 0:
        raise RuntimeError("No time-mean samples accumulated; aborting.")
    mean_T = sum_T / n_mean_samples
    mean_S = sum_S / n_mean_samples
    mean_u = sum_u / n_mean_samples
    mean_v = sum_v / n_mean_samples
    mean_w = sum_w / n_mean_samples
    mean_eta = sum_eta / n_mean_samples

    np.savez_compressed(
        OUTPUT_DIR / "time_mean.npz",
        T=mean_T, S=mean_S, u=mean_u, v=mean_v, w=mean_w, eta=mean_eta,
        n_samples=n_mean_samples,
        day_start=day_offset,
        day_end=final_day,
        lat_deg=lat_deg, lon_deg=lon_deg,
        z_full_ref=z_full_ref, z_half_ref=z_half_ref, dz_ref=dz_ref,
    )
    print(f"  Time mean saved: {OUTPUT_DIR / 'time_mean.npz'}")

    # ---- Meridional overturning streamfunction from time-mean v ----
    v_mask_arr = np.asarray(state.v_mask.data)         # (n_lat+1, n_lon)
    # Mask v across walls; sum v*dx zonally per (lat, level)
    masked_v = mean_v * v_mask_arr[:, :, None]         # (n_lat+1, n_lon, nlev)
    U_zonal = np.sum(masked_v * dx_v[:, None, None], axis=1)  # (n_lat+1, nlev)
    # Cumulative integral from bottom (deepest level) up: ψ at top of layer k
    # is the meridional volume transport occurring below depth z_half[k+1].
    # ψ(j, k) = -Σ_{k'>=k} U(j, k') * dz[k']      [integration upward]
    # Sign convention: positive = clockwise in (y, z) plane (NH-like northward
    # at surface, sinking, southward at depth).
    UV = U_zonal * dz_ref[None, :]                     # (n_lat+1, nlev)
    # cumulative from deepest to shallowest:
    psi = -np.cumsum(UV[:, ::-1], axis=1)[:, ::-1]      # (n_lat+1, nlev)
    psi_Sv = psi / 1.0e6                                # m^3/s -> Sv

    np.savez_compressed(
        OUTPUT_DIR / "moc.npz",
        psi_Sv=psi_Sv,
        lat_v_deg=lat_v_deg,
        z_half_ref=z_half_ref,
        z_full_ref=z_full_ref,
        n_samples=n_mean_samples,
    )
    print(f"  MOC saved: {OUTPUT_DIR / 'moc.npz'}")

    # ---- Plots ----
    _plot_diagnostics(diag_times, diag_max_speed, diag_mean_sst,
                      diag_mean_T, diag_T_deep, diag_mean_eta,
                      snapshots, lat_deg, lon_deg, day_offset, total_years,
                      mean_T, mean_S, mean_u, mean_v, mean_eta,
                      psi_Sv, lat_v_deg, z_half_ref, z_full_ref,
                      np.asarray(state.land_mask.data))
    print(f"\nAll output in: {OUTPUT_DIR}")


def _plot_diagnostics(times, max_speed, mean_sst, mean_T, T_deep, mean_eta,
                      snapshots, lat_deg, lon_deg, day_offset, total_years,
                      mean_T_3d, mean_S_3d, mean_u_3d, mean_v_3d, mean_eta_2d,
                      psi_Sv, lat_v_deg, z_half_ref, z_full_ref,
                      land_mask_2d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years_since_restart = (np.array(times) - day_offset) / 365.0

    # ---- Timeseries ----
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(years_since_restart, max_speed)
    axes[0].set_ylabel("Max speed (m/s)")
    axes[0].set_title(
        f"10-yr continuation (from day {int(day_offset)}) — diagnostics"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(years_since_restart, mean_sst)
    axes[1].set_ylabel("Mean SST (degC)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(years_since_restart, mean_T, label="All-depth")
    axes[2].plot(years_since_restart, T_deep, label="Bottom layer")
    axes[2].set_ylabel("Mean T (degC)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(years_since_restart, mean_eta)
    axes[3].set_ylabel("Mean SSH (m)")
    axes[3].set_xlabel("Year (since restart)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"timeseries_{total_years}yr.png", dpi=150)
    plt.close()

    # ---- Snapshot evolution (subset to keep figure manageable) ----
    snap_days = sorted(snapshots.keys())
    # Plot every Nth snapshot so the figure stays readable
    n_show = min(13, len(snap_days))  # initial + ~12 evenly spaced
    if n_show <= 1:
        idx = [0]
    else:
        idx = np.linspace(0, len(snap_days) - 1, n_show).round().astype(int)
    show_days = [snap_days[k] for k in idx]

    fig, axes = plt.subplots(len(show_days), 3,
                             figsize=(15, 3 * len(show_days)))
    if len(show_days) == 1:
        axes = axes[np.newaxis, :]
    for i, day in enumerate(show_days):
        s = snapshots[day]
        for j, (key, label, cmap) in enumerate([
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("speed_sfc", "Speed (m/s)", "magma"),
        ]):
            im = axes[i, j].imshow(
                s[key], aspect="auto", origin="lower", cmap=cmap)
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
            if i == 0:
                axes[i, j].set_title(label)
            yr = (day - day_offset) / 365.0
            axes[i, j].set_ylabel(f"Year {yr:.2f}")
    plt.suptitle("Monthly snapshots (subset)", y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"snapshots_{total_years}yr.png", dpi=120)
    plt.close()

    # ---- Time-mean maps ----
    ocean_mask = land_mask_2d > 0.5
    sst_mean = np.where(ocean_mask, mean_T_3d[:, :, 0], np.nan)
    eta_mean = np.where(ocean_mask, mean_eta_2d, np.nan)

    # Time-mean surface speed from u, v cell-center reconstruction
    u_c = 0.5 * (mean_u_3d[:, :-1, 0] + mean_u_3d[:, 1:, 0])
    v_c = 0.5 * (mean_v_3d[:-1, :, 0] + mean_v_3d[1:, :, 0])
    speed_mean = np.where(ocean_mask, np.sqrt(u_c**2 + v_c**2), np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    extent = [lon_deg[0], lon_deg[-1], lat_deg[0], lat_deg[-1]]
    for ax, field, label, cmap in [
        (axes[0], sst_mean, "10-yr mean SST (degC)", "RdYlBu_r"),
        (axes[1], eta_mean, "10-yr mean SSH (m)", "RdBu_r"),
        (axes[2], speed_mean, "10-yr mean surface speed (m/s)", "magma"),
    ]:
        im = ax.imshow(field, aspect="auto", origin="lower",
                       extent=extent, cmap=cmap)
        ax.set_title(label)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_mean_surface.png", dpi=150)
    plt.close()

    # ---- Time-mean zonal-mean T section ----
    T_zm_mean = np.mean(mean_T_3d, axis=1)  # (n_lat, nlev)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        lat_deg, z_full_ref, T_zm_mean.T,
        cmap="RdYlBu_r", shading="auto"
    )
    plt.colorbar(im, ax=ax, label="T (degC)")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"10-yr time-mean zonal-mean potential temperature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_mean_T_zonal.png", dpi=150)
    plt.close()

    # ---- MOC ----
    fig, ax = plt.subplots(figsize=(8, 5))
    # psi_Sv lives at v-face latitudes (n_lat+1) and full levels (nlev)
    levels = np.linspace(-np.nanmax(np.abs(psi_Sv)),
                         np.nanmax(np.abs(psi_Sv)), 21)
    if not np.isfinite(levels).all() or levels[-1] == levels[0]:
        levels = np.linspace(-1, 1, 21)
    cf = ax.contourf(lat_v_deg, z_full_ref, psi_Sv.T,
                     levels=levels, cmap="RdBu_r", extend="both")
    cs = ax.contour(lat_v_deg, z_full_ref, psi_Sv.T,
                    levels=levels[::2], colors="k", linewidths=0.5)
    ax.clabel(cs, fmt="%.0f", fontsize=7)
    plt.colorbar(cf, ax=ax, label="ψ (Sv)")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(
        f"10-yr time-mean meridional overturning streamfunction"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_mean_MOC.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
