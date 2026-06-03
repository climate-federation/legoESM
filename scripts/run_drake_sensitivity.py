#!/usr/bin/env python
"""Drake Passage barotropic-flow sensitivity tests.

Restarts from the 50yr GM/Redi run (day 18250) and integrates 10 years
under one of two modifications, to diagnose why the original run shows
a strong westward depth-mean flow in the Drake band:

  --case bottom_drag    Reduce ``bottom_drag_coeff`` by ``--drag-factor``
                         (default 0.2 = 5× weaker drag).  Increases the
                         barotropic decay timescale; tests whether the
                         absent ACC is bottom-drag-limited.

  --case ridge          Add a meridional ridge in the Drake band: at
                         lon index ``--ridge-lon-idx`` (default 30),
                         set ``H_bathy[2:7, idx] = ridge_depth`` (default
                         2000 m).  Tests whether topographic form drag
                         is the missing element for a realistic ACC.

Each run writes to a case-specific subdirectory under
``results/ocean/`` and produces:

  - snapshots.npz: 121 monthly samples (eta, SST, speed_sfc,
                                        T_zonal_mean, drake_T_section)
  - drake_timeseries.npz: barotropic Drake transport at every snapshot
  - time_mean.npz: 10yr time mean of full 3D fields
  - moc.npz: meridional overturning streamfunction from time-mean v
  - diagnostics.npz: ~10/year scalars including drake transport
  - plots: timeseries, drake transport, time-mean zonal-mean u, MOC

Usage:
    JAX_ENABLE_X64=1 python scripts/run_drake_sensitivity.py --case bottom_drag
    JAX_ENABLE_X64=1 python scripts/run_drake_sensitivity.py --case ridge
"""

from __future__ import annotations

import argparse
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
J_DRAKE = np.arange(2, 7)  # lat indices for Drake band (-77.5..-57.5)


def _restore_state_from_restart(template_state, restart_path: Path,
                                bathy_override: np.ndarray | None = None):
    """Replace each Field's data from the restart npz; optionally override
    H_bathy with `bathy_override` (shape (n_lat, n_lon))."""
    npz = np.load(restart_path, allow_pickle=False)
    saved_day = float(npz["time_days"])
    saved_step = int(npz["step"])

    new_fields = {}
    for fname in template_state._fields:
        field_obj = getattr(template_state, fname)
        if field_obj is None:
            continue
        if not hasattr(field_obj, "data"):
            continue
        if fname not in npz.files:
            raise KeyError(f"Restart missing field '{fname}'")
        data = npz[fname]
        if fname == "H_bathy" and bathy_override is not None:
            data = bathy_override
        new_fields[fname] = field_obj.replace(
            data=jnp.asarray(data, dtype=field_obj.data.dtype)
        )
    return template_state._replace(**new_fields), saved_day, saved_step


def _drake_metrics(state_T, state_u, state_u_mask, dz, dy, lat_deg):
    """Return (drake_transport_Sv, drake_baro_u_cms, drake_T_section)."""
    u = np.asarray(state_u)
    um = np.asarray(state_u_mask)
    T = np.asarray(state_T)

    m = um[:, :, None]
    den = m.sum(axis=1)
    zm_u = np.where(den > 0, (u * m).sum(axis=1) / np.where(den > 0, den, 1.0),
                    np.nan)                                      # (n_lat, nlev)
    H = float(dz.sum())
    u_baro_lat = np.nansum(zm_u * dz[None, :], axis=1) / H       # (n_lat,)
    drake_baro = float(np.nanmean(u_baro_lat[J_DRAKE]))
    transport = 0.0
    for j in J_DRAKE:
        transport += np.nansum(zm_u[j] * dz) * dy
    drake_T_zm = np.mean(T, axis=1)[J_DRAKE]                     # (5, nlev)
    return transport / 1e6, drake_baro * 100.0, drake_T_zm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["bottom_drag", "ridge"],
                        required=True)
    parser.add_argument("--drag-factor", type=float, default=0.2,
                        help="multiplier on bottom_drag_coeff (case=bottom_drag)")
    parser.add_argument("--ridge-depth", type=float, default=2000.0,
                        help="ridge crest H_bathy [m] (case=ridge)")
    parser.add_argument("--ridge-lon-idx", type=int, default=30,
                        help="longitude index for ridge (0..71)")
    parser.add_argument("--total-years", type=float, default=10.0)
    args = parser.parse_args()

    # ---- Configuration ----
    total_years = args.total_years
    diag_every_days = 36.5
    months_per_year = 12
    dt = 600.0
    n_lat, n_lon = 36, 72
    n_barotropic_substeps = 30

    case_tag = (
        f"lower_drag_x{args.drag_factor:g}".replace(".", "p")
        if args.case == "bottom_drag"
        else f"ridge_{int(args.ridge_depth)}m_lon{args.ridge_lon_idx}"
    )
    output_dir = Path(f"results/ocean/drake_sensitivity_{case_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not RESTART_PATH.exists():
        raise FileNotFoundError(f"Restart not found: {RESTART_PATH}")

    config = GlobalOverturningConfig(use_gm_redi=True)
    gm_redi_cfg = create_gm_redi_config(config)
    bottom_drag = config.bottom_drag_coeff
    if args.case == "bottom_drag":
        bottom_drag = config.bottom_drag_coeff * args.drag_factor

    n_steps = int(total_years * 365.0 * 86400 / dt)
    diag_every = max(1, int(diag_every_days * 86400 / dt))
    snapshot_every = int(round((365.0 / months_per_year) * 86400 / dt))

    print(f"=== Drake sensitivity: {args.case} ===")
    print(f"  Restart: {RESTART_PATH}")
    print(f"  Output : {output_dir}")
    print(f"  bottom_drag_coeff: {config.bottom_drag_coeff} -> {bottom_drag} "
          f"(timescale at H=4000m: "
          f"{4000.0 / max(bottom_drag, 1e-12) / 86400:.0f} d)")
    if args.case == "ridge":
        print(f"  Ridge: H_bathy[{J_DRAKE.tolist()}, {args.ridge_lon_idx}] "
              f"= {args.ridge_depth} m")
    print(f"  dt={dt}s, n_steps={n_steps:,}, "
          f"snapshot every {snapshot_every} steps "
          f"({n_steps // snapshot_every} samples)")
    print()

    # ---- Setup ----
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
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
        bottom_drag_r=bottom_drag,
        eos="linear",
        eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    template_state = create_initial_conditions("latlon", grid, z_coord, config)

    # Build optional bathymetry override
    bathy_override = None
    if args.case == "ridge":
        H0 = np.asarray(template_state.H_bathy.data, dtype=np.float64)
        bathy_override = H0.copy()
        bathy_override[J_DRAKE, args.ridge_lon_idx] = args.ridge_depth
        # Keep land cells unchanged regardless
        land = np.asarray(template_state.land_mask.data) < 0.5
        bathy_override[land] = H0[land]

    state, day_offset, _ = _restore_state_from_restart(
        template_state, RESTART_PATH, bathy_override=bathy_override
    )
    print(f"  Restarted at sim day {day_offset:.1f} (year {day_offset/365:.2f})")
    if args.case == "ridge":
        H_check = np.asarray(state.H_bathy.data)
        print(f"  Ridge H_bathy now: {H_check[J_DRAKE, args.ridge_lon_idx]}")
        # Zero u, v on faces touching the ridge column to avoid the restart
        # shock from existing momentum colliding with the new shallower H.
        u_arr = np.asarray(state.u.data).copy()
        v_arr = np.asarray(state.v.data).copy()
        j_lo, j_hi = int(J_DRAKE[0]), int(J_DRAKE[-1])
        ridge_i = args.ridge_lon_idx
        u_arr[j_lo:j_hi + 1, ridge_i,     :] = 0.0
        u_arr[j_lo:j_hi + 1, ridge_i + 1, :] = 0.0
        v_arr[j_lo:j_hi + 2, ridge_i,     :] = 0.0
        state = state._replace(
            u=state.u.replace(
                data=jnp.asarray(u_arr, dtype=state.u.data.dtype)),
            v=state.v.replace(
                data=jnp.asarray(v_arr, dtype=state.v.data.dtype)),
        )
        print(f"  Zeroed u/v on faces of ridge column "
              f"(rows {j_lo}..{j_hi}, lon idx {ridge_i})")
    print()

    # ---- Coordinates / metrics ----
    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    dz_ref = np.asarray(z_coord.dz_ref, dtype=np.float64)
    z_full_ref = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    z_half_ref = np.asarray(z_coord.z_half_ref, dtype=np.float64)
    dy = float(grid.radius) * float(grid.dlat)
    # v-face latitudes for MOC
    lat_v_rad = np.concatenate([
        [np.asarray(grid.lat)[0] - 0.5 * float(grid.dlat)],
        0.5 * (np.asarray(grid.lat)[:-1] + np.asarray(grid.lat)[1:]),
        [np.asarray(grid.lat)[-1] + 0.5 * float(grid.dlat)],
    ])
    lat_v_deg = lat_v_rad * 180 / np.pi
    cos_lat_v = np.cos(np.clip(lat_v_rad, -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
    dx_v = cos_lat_v * float(grid.radius) * float(grid.dlon)

    # ---- Storage ----
    diag_times, diag_max_speed, diag_mean_sst, diag_mean_T = [], [], [], []
    diag_T_deep, diag_mean_eta = [], []
    diag_drake_T = []                     # Drake transport (Sv) at diag cadence
    diag_drake_baro = []                  # Drake-band barotropic u (cm/s)
    snap_times, snap_drake_T, snap_drake_baro = [], [], []
    snapshots = {}

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
        diag_times.append(day)
        diag_max_speed.append(float(np.max(speed)))
        diag_mean_sst.append(
            float(np.mean(T[:, :, 0][ocean_mask])) if ocean_mask.any() else 0.0)
        diag_mean_T.append(
            float(np.mean(T[ocean_mask])) if ocean_mask.any() else 0.0)
        diag_T_deep.append(
            float(np.mean(T[:, :, -1][ocean_mask])) if ocean_mask.any() else 0.0)
        diag_mean_eta.append(
            float(np.mean(eta[ocean_mask])) if ocean_mask.any() else 0.0)
        T_Sv, baro_cms, _ = _drake_metrics(
            T, u, np.asarray(state.u_mask.data), dz_ref, dy, lat_deg)
        diag_drake_T.append(T_Sv)
        diag_drake_baro.append(baro_cms)

    def save_snapshot(state, day):
        eta = np.asarray(state.eta.data)
        T = np.asarray(state.T.data)
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        T_Sv, baro_cms, drake_T_sec = _drake_metrics(
            T, u, np.asarray(state.u_mask.data), dz_ref, dy, lat_deg)
        snapshots[day] = {
            "eta": eta.copy(),
            "SST": T[:, :, 0].copy(),
            "speed_sfc": np.sqrt(u_c**2 + v_c**2),
            "T_zonal_mean": np.mean(T, axis=1).copy(),
            "drake_T_section": drake_T_sec.copy(),  # (5, nlev)
        }
        snap_times.append(day)
        snap_drake_T.append(T_Sv)
        snap_drake_baro.append(baro_cms)

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
        restart = {"step": step, "time_days": day, "grid_type": "latlon"}
        for f in state._fields:
            obj = getattr(state, f)
            if obj is not None and hasattr(obj, "data"):
                restart[f] = np.asarray(obj.data)
        fname = output_dir / f"restart_day{int(day):06d}.npz"
        np.savez_compressed(fname, **restart)

    record_diagnostics(state, day_offset)
    save_snapshot(state, day_offset)

    print(f"Starting {total_years}-year sensitivity run "
          f"({n_steps:,} steps)...")
    t0 = time.time()
    last_print = t0

    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1
        day = day_offset + step * dt / 86400.0

        if step % 100 == 0:
            eta_max = float(jnp.max(jnp.abs(state.eta.data)))
            if not np.isfinite(eta_max) or eta_max > 100.0:
                print(f"\n  BLOWUP at step {step} (day {day:.1f})")
                save_restart_file(state, step, day)
                break

        if step % diag_every == 0:
            record_diagnostics(state, day)
            now = time.time()
            if now - last_print > 30:
                yr = (day - day_offset) / 365.0
                elapsed = now - t0
                eta_s = elapsed / yr * total_years - elapsed if yr > 0 else 0
                print(f"  Year {yr:5.2f}/{int(total_years):2d} | "
                      f"spd={diag_max_speed[-1]:.3f} | "
                      f"DrakeT={diag_drake_T[-1]:+6.1f}Sv | "
                      f"u_baro={diag_drake_baro[-1]:+5.2f}cm/s | "
                      f"SST={diag_mean_sst[-1]:.2f} | "
                      f"ETA {eta_s/3600:.1f}h", flush=True)
                last_print = now

        if step % snapshot_every == 0:
            save_snapshot(state, day)
            accumulate_mean(state)

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    print(f"\nCompleted in {wall:.0f}s ({wall/3600:.2f}h)")
    final_day = day_offset + n_steps * dt / 86400.0
    save_restart_file(state, n_steps, final_day)
    print(f"  Restart saved (day {final_day:.0f})")
    print(f"  Time-mean over {n_mean_samples} monthly samples")

    # ---- Save outputs ----
    np.savez_compressed(
        output_dir / "diagnostics.npz",
        times=np.array(diag_times),
        max_speed=np.array(diag_max_speed),
        mean_sst=np.array(diag_mean_sst),
        mean_T=np.array(diag_mean_T),
        T_deep=np.array(diag_T_deep),
        mean_eta=np.array(diag_mean_eta),
        drake_T_Sv=np.array(diag_drake_T),
        drake_baro_cms=np.array(diag_drake_baro),
    )
    np.savez_compressed(
        output_dir / "drake_timeseries.npz",
        snap_times=np.array(snap_times),
        snap_drake_T_Sv=np.array(snap_drake_T),
        snap_drake_baro_cms=np.array(snap_drake_baro),
    )

    snap_data = {}
    for day_key, fields in snapshots.items():
        for fn, arr in fields.items():
            snap_data[f"{fn}_day{int(day_key):06d}"] = arr
    snap_data["snapshot_days"] = np.array(sorted(snapshots.keys()))
    np.savez_compressed(output_dir / "snapshots.npz", **snap_data)

    if n_mean_samples > 0:
        mean_T = sum_T / n_mean_samples
        mean_S = sum_S / n_mean_samples
        mean_u = sum_u / n_mean_samples
        mean_v = sum_v / n_mean_samples
        mean_w = sum_w / n_mean_samples
        mean_eta = sum_eta / n_mean_samples
        np.savez_compressed(
            output_dir / "time_mean.npz",
            T=mean_T, S=mean_S, u=mean_u, v=mean_v, w=mean_w, eta=mean_eta,
            n_samples=n_mean_samples,
            day_start=day_offset, day_end=final_day,
            lat_deg=lat_deg, lon_deg=lon_deg,
            z_full_ref=z_full_ref, z_half_ref=z_half_ref, dz_ref=dz_ref,
        )
        # MOC from time-mean v
        v_mask = np.asarray(state.v_mask.data)
        masked_v = mean_v * v_mask[:, :, None]
        U = np.sum(masked_v * dx_v[:, None, None], axis=1)
        UV = U * dz_ref[None, :]
        psi = -np.cumsum(UV[:, ::-1], axis=1)[:, ::-1]
        psi_Sv = psi / 1.0e6
        np.savez_compressed(
            output_dir / "moc.npz",
            psi_Sv=psi_Sv, lat_v_deg=lat_v_deg,
            z_half_ref=z_half_ref, z_full_ref=z_full_ref,
            n_samples=n_mean_samples,
        )

    _make_plots(output_dir, args, day_offset,
                diag_times, diag_drake_T, diag_drake_baro,
                diag_max_speed, diag_mean_sst, diag_T_deep, diag_mean_eta,
                snap_times, snap_drake_T, snap_drake_baro,
                snapshots, lat_deg, lon_deg, z_full_ref,
                mean_T if n_mean_samples > 0 else None,
                mean_u if n_mean_samples > 0 else None,
                psi_Sv if n_mean_samples > 0 else None,
                lat_v_deg, z_full_ref,
                np.asarray(state.land_mask.data),
                np.asarray(state.u_mask.data),
                np.asarray(state.H_bathy.data),
                dz_ref)
    print(f"\nAll output in: {output_dir}")


def _make_plots(output_dir, args, day_offset,
                diag_times, diag_drake_T, diag_drake_baro,
                diag_max_speed, diag_mean_sst, diag_T_deep, diag_mean_eta,
                snap_times, snap_drake_T, snap_drake_baro,
                snapshots, lat_deg, lon_deg, z_full,
                mean_T, mean_u, psi_Sv, lat_v_deg, z_full_ref,
                land_mask, u_mask, H_bathy, dz_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yrs_diag = (np.array(diag_times) - day_offset) / 365.0
    yrs_snap = (np.array(snap_times) - day_offset) / 365.0
    case_label = (f"lower_drag (×{args.drag_factor})"
                  if args.case == "bottom_drag"
                  else f"ridge (depth {int(args.ridge_depth)} m at lon idx "
                       f"{args.ridge_lon_idx})")

    # ---- Drake transport timeseries (the headline diagnostic) ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(yrs_snap, snap_drake_T, "C0-", lw=1.0, alpha=0.7,
                 label="monthly")
    axes[0].plot(yrs_diag, diag_drake_T, "C3-o", ms=4, lw=0.8,
                 label="diag (~36 d)")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].axhline(-407, color="k", lw=0.5, ls=":",
                    label="50yr endpoint (−407 Sv)")
    axes[0].set_ylabel("Drake transport (Sv)")
    axes[0].set_title(f"Drake Passage transport — sensitivity: {case_label}")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(yrs_snap, snap_drake_baro, "C2-", lw=1.0, alpha=0.7)
    axes[1].plot(yrs_diag, diag_drake_baro, "C2-o", ms=4)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("Drake-band depth-mean u (cm/s)")
    axes[1].set_xlabel("Year (since day 18250 restart)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "drake_transport_timeseries.png", dpi=150)
    plt.close()

    # ---- Time-mean zonal-mean u(lat, depth) ----
    if mean_u is not None:
        m = u_mask[:, :, None]
        den = m.sum(axis=1)
        zm_u = np.where(den > 0,
                        (mean_u * m).sum(axis=1) / np.where(den > 0, den, 1.0),
                        np.nan)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        vmax = np.nanmax(np.abs(zm_u[:18]))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 0.1
        levels = np.linspace(-vmax, vmax, 21)
        for ax, lat_slice, title in [
            (axes[0], slice(2, 18), "Zonal-mean u, SH (10yr time mean)"),
            (axes[1], slice(2, 7),  "Drake band zoom"),
        ]:
            cf = ax.contourf(lat_deg[lat_slice], z_full_ref,
                             zm_u[lat_slice].T,
                             levels=levels, cmap="RdBu_r", extend="both")
            cs = ax.contour(lat_deg[lat_slice], z_full_ref, zm_u[lat_slice].T,
                            levels=levels[::2], colors="k", linewidths=0.4)
            try:
                ax.clabel(cs, fmt="%.2f", fontsize=6)
            except Exception:
                pass
            plt.colorbar(cf, ax=ax, label="u (m/s)", fraction=0.04)
            ax.set_title(title)
            ax.set_xlabel("Latitude (deg)")
            ax.set_ylabel("Depth (m)")
        plt.suptitle(f"{case_label}", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "time_mean_zonal_u.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    # ---- MOC ----
    if psi_Sv is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        vmax = np.nanmax(np.abs(psi_Sv))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        levels = np.linspace(-vmax, vmax, 21)
        cf = ax.contourf(lat_v_deg, z_full_ref, psi_Sv.T,
                         levels=levels, cmap="RdBu_r", extend="both")
        cs = ax.contour(lat_v_deg, z_full_ref, psi_Sv.T,
                        levels=levels[::2], colors="k", linewidths=0.5)
        ax.clabel(cs, fmt="%.0f", fontsize=7)
        plt.colorbar(cf, ax=ax, label="ψ (Sv)")
        ax.set_xlabel("Latitude (deg)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"10yr-mean MOC — {case_label}")
        plt.tight_layout()
        plt.savefig(output_dir / "time_mean_MOC.png", dpi=150)
        plt.close()

    # ---- Bathymetry sanity (ridge case) ----
    if args.case == "ridge":
        fig, ax = plt.subplots(figsize=(10, 4))
        H = np.where(land_mask > 0.5, H_bathy, np.nan)
        im = ax.imshow(H, aspect="auto", origin="lower",
                       extent=[lon_deg[0], lon_deg[-1], lat_deg[0], lat_deg[-1]],
                       cmap="viridis_r")
        plt.colorbar(im, ax=ax, label="H_bathy (m)")
        ax.set_title("Bathymetry with ridge")
        ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
        plt.tight_layout()
        plt.savefig(output_dir / "bathymetry_with_ridge.png", dpi=150)
        plt.close()

    print(f"  Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
