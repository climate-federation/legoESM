#!/usr/bin/env python
"""Run spectral ocean test cases with diagnostics and plots.

Cases (simple -> complex):
  1. Rest state adjustment
  2. Barotropic gravity wave (Gaussian SSH perturbation)
  3. Baroclinic adjustment (meridional thermal front)

Outputs per case:
  - SSH and speed snapshots
  - profile evolution at a representative wet column
  - mean-state and conservation time series
  - latitude-depth and longitude-depth sections
  - CSV time series and run summary JSON/TXT
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _ensure_mpl_config_dir() -> None:
    current = os.environ.get("MPLCONFIGDIR", "")
    if current and os.path.isdir(current) and os.access(current, os.W_OK):
        return
    candidate = os.path.join(tempfile.gettempdir(), "legoesm_mplconfig")
    os.makedirs(candidate, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = candidate


_ensure_mpl_config_dir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis,
    sh_analysis_3d,
    sh_synthesis,
    sh_synthesis_3d,
    uv_from_vordiv_3d,
)
from legoesm.ocean import (
    SpectralOceanConfig,
    SpectralOceanModel,
    create_ocean_z_star,
    rest_state_spectral_ocean,
)
from legoesm.ocean.vertical import compute_layer_thickness


def _cell_area(grid):
    dlon = 2.0 * np.pi / grid.n_lon
    return (float(grid.radius) ** 2) * np.asarray(grid.weights, dtype=np.float64)[:, None] * dlon


def _edges_from_centers(centers):
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    e = np.empty(c.size + 1, dtype=np.float64)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - (e[1] - c[0])
    e[-1] = c[-1] + (c[-1] - e[-2])
    return e


def _fill_missing_bands(section, *, periodic=False):
    out = np.asarray(section, dtype=np.float64).copy()
    if out.ndim != 2:
        return out
    n_bins = out.shape[0]
    idx = np.arange(n_bins, dtype=np.float64)
    for k in range(out.shape[1]):
        col = out[:, k]
        finite = np.isfinite(col)
        if np.sum(finite) == 0:
            continue
        if np.sum(finite) == 1:
            col[~finite] = col[finite][0]
        else:
            if periodic:
                x = idx[finite]
                y = col[finite]
                x_ext = np.concatenate([x - float(n_bins), x, x + float(n_bins)])
                y_ext = np.concatenate([y, y, y])
                col[~finite] = np.interp(idx[~finite], x_ext, y_ext)
            else:
                col[~finite] = np.interp(idx[~finite], idx[finite], col[finite])
        out[:, k] = col

    finite_level = np.any(np.isfinite(out), axis=0)
    if np.any(finite_level):
        level_idx = np.arange(out.shape[1], dtype=np.int64)
        for k in level_idx[~finite_level]:
            nearest = level_idx[finite_level][
                np.argmin(np.abs(level_idx[finite_level] - k))
            ]
            out[:, k] = out[:, nearest]

    finite_all = np.isfinite(out)
    if np.any(finite_all):
        fill = float(np.nanmean(out[finite_all]))
        out[~finite_all] = fill
    else:
        out[...] = 0.0
    return out


def _compute_binned_depth_section(field_3d, coord_2d_deg, area_2d, mask_2d, bins_deg, *, periodic=False):
    field = np.asarray(field_3d, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError(f"Expected field_3d (lat, lon, level), got {field.shape}")
    coord = np.asarray(coord_2d_deg, dtype=np.float64)
    area = np.asarray(area_2d, dtype=np.float64)
    mask = np.asarray(mask_2d, dtype=np.float64)

    n_levels = field.shape[-1]
    n_bins = int(len(bins_deg) - 1)
    section = np.full((n_bins, n_levels), np.nan, dtype=np.float64)

    coord_flat = coord.ravel()
    weight_flat = (area * mask).ravel()
    idx_flat = np.digitize(coord_flat, bins_deg, right=False) - 1
    idx_flat = np.clip(idx_flat, 0, n_bins - 1)
    valid_geom = np.isfinite(coord_flat) & np.isfinite(weight_flat) & (weight_flat > 0.0)

    for k in range(n_levels):
        data_flat = np.asarray(field[..., k], dtype=np.float64).ravel()
        valid = valid_geom & np.isfinite(data_flat)
        if not np.any(valid):
            continue
        idx = idx_flat[valid]
        w = weight_flat[valid]
        v = data_flat[valid]
        sum_w = np.bincount(idx, weights=w, minlength=n_bins)
        sum_v = np.bincount(idx, weights=w * v, minlength=n_bins)
        wet = sum_w > 0.0
        section[wet, k] = sum_v[wet] / sum_w[wet]

    section = _fill_missing_bands(section, periodic=periodic)
    centers = 0.5 * (np.asarray(bins_deg[:-1]) + np.asarray(bins_deg[1:]))
    return centers, section


def _plot_depth_sections(output_path, z_coord, coord_centers, coord_label, panels, suptitle):
    depth = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    if float(np.nanmean(depth)) < 0.0:
        depth = -depth

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 6.0), squeeze=False)
    axes = axes.ravel()

    for ax, (title, section, cmap, symmetric, cbar_label) in zip(axes, panels):
        data = np.asarray(section, dtype=np.float64).T
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "No ocean data", ha="center", va="center")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(coord_label)
            ax.set_ylabel("Depth [m]")
            continue

        if symmetric:
            vmax = max(float(np.max(np.abs(finite))), 1.0e-12)
            vmin = -vmax
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if abs(vmax - vmin) < 1.0e-12:
                vmax = vmin + 1.0e-12

        coord_edges = _edges_from_centers(coord_centers)
        depth_edges = _edges_from_centers(depth)
        cs = ax.pcolormesh(
            coord_edges,
            depth_edges,
            np.ma.masked_invalid(data),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel(coord_label)
        ax.set_ylabel("Depth [m]")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(float(np.nanmax(depth)), float(np.nanmin(depth)))
        ax.grid(True, alpha=0.25)
        fig.colorbar(cs, ax=ax, orientation="vertical", pad=0.02, label=cbar_label)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_latlon_snapshots(output_path, snapshots, dt, grid, field_getter, title, cmap, cbar_label, *, symmetric=False):
    items = sorted(snapshots.items())
    if not items:
        return
    if len(items) > 4:
        idx = np.linspace(0, len(items) - 1, 4, dtype=int)
        items = [items[i] for i in idx]

    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180.0 / np.pi
    lon_edges = _edges_from_centers(lon_deg)
    lat_edges = _edges_from_centers(lat_deg)

    data_all = []
    converted = []
    for step_num, fields in items:
        arr = np.asarray(field_getter(fields), dtype=np.float64)
        converted.append((step_num, arr))
        finite = arr[np.isfinite(arr)]
        if finite.size:
            data_all.append(finite)

    if data_all:
        stacked = np.concatenate(data_all)
        if symmetric:
            vmax = max(float(np.nanmax(np.abs(stacked))), 1.0e-12)
            vmin = -vmax
        else:
            vmin = float(np.nanmin(stacked))
            vmax = float(np.nanmax(stacked))
            if abs(vmax - vmin) < 1.0e-12:
                vmax = vmin + 1.0e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.ravel()
    pcm = None
    for i, ax in enumerate(axes):
        if i >= len(converted):
            ax.axis("off")
            continue
        step_num, arr = converted[i]
        day = step_num * dt / 86400.0
        pcm = ax.pcolormesh(
            lon_edges,
            lat_edges,
            np.ma.masked_invalid(arr),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlim(float(np.min(lon_edges)), float(np.max(lon_edges)))
        ax.set_ylim(float(np.min(lat_edges)), float(np.max(lat_edges)))
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(f"step {step_num}, t={day:.2f} d", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)

    if pcm is not None:
        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.90, wspace=0.13, hspace=0.32)
        cax = fig.add_axes([0.92, 0.13, 0.02, 0.72])
        fig.colorbar(pcm, cax=cax, orientation="vertical", label=cbar_label)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_point_profile_evolution(output_path, z_coord, profiles, dt, x_label, title):
    if not profiles:
        return
    z_ref = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    if float(np.nanmean(z_ref)) > 0.0:
        z_ref = -z_ref
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    for step_num, profile in sorted(profiles.items()):
        day = step_num * dt / 86400.0
        arr = np.asarray(profile, dtype=np.float64)
        if arr.shape != z_ref.shape:
            continue
        ax.plot(arr, z_ref, linewidth=1.5, label=f"Day {day:.2f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Depth [m]")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _state_to_grid_fields(state, grid):
    mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
    mask_3d = mask[..., None]
    eta = np.asarray(sh_synthesis(grid, state.eta_hat.data).real, dtype=np.float64) * mask
    H_bathy = np.asarray(sh_synthesis(grid, state.H_bathy_hat.data).real, dtype=np.float64)
    H_bathy = np.maximum(H_bathy, 1.0) * mask + 1.0 * (1.0 - mask)

    T = np.asarray(sh_synthesis_3d(grid, state.T_hat.data).real, dtype=np.float64)
    S = np.asarray(sh_synthesis_3d(grid, state.S_hat.data).real, dtype=np.float64)
    u_cos, v_cos = uv_from_vordiv_3d(grid, state.vor_hat.data, state.div_hat.data)
    cos_lat = np.asarray(grid.cos_lat, dtype=np.float64)[:, None, None]
    cos_safe = np.maximum(cos_lat, 1.0e-6)
    u = np.asarray(u_cos, dtype=np.float64) / cos_safe * mask_3d
    v = np.asarray(v_cos, dtype=np.float64) / cos_safe * mask_3d
    speed = np.sqrt(u * u + v * v)

    return {
        "mask": mask,
        "eta": eta,
        "H_bathy": H_bathy,
        "T": T,
        "S": S,
        "u": u,
        "v": v,
        "speed": speed,
    }


def _compute_diagnostics(fields, grid, z_coord, min_water_column_m):
    mask = np.asarray(fields["mask"], dtype=np.float64)
    area = _cell_area(grid)
    weighted_area = area * mask

    h_k = compute_layer_thickness(
        jnp.asarray(fields["eta"]),
        jnp.asarray(fields["H_bathy"]),
        z_coord,
        min_water_column_m=min_water_column_m,
    )
    h_k = np.asarray(h_k, dtype=np.float64)

    volume = float(np.sum(fields["eta"] * weighted_area))
    heat = float(np.sum(np.sum(fields["T"] * h_k, axis=-1) * weighted_area))
    salt = float(np.sum(np.sum(fields["S"] * h_k, axis=-1) * weighted_area))
    ke = float(np.sum(np.sum(0.5 * (fields["u"] ** 2 + fields["v"] ** 2) * h_k, axis=-1) * weighted_area))
    ocean_area = max(float(np.sum(weighted_area)), 1.0)

    sst = np.asarray(fields["T"][..., 0], dtype=np.float64)
    eta = np.asarray(fields["eta"], dtype=np.float64)
    speed = np.asarray(fields["speed"], dtype=np.float64)

    return {
        "volume": volume,
        "heat": heat,
        "salt": salt,
        "ocean_area": ocean_area,
        "kinetic_energy": ke,
        "SST_mean": float(np.sum(sst * weighted_area) / ocean_area),
        "SST_max": float(np.max(np.where(mask > 0.5, sst, -999.0))),
        "SST_min": float(np.min(np.where(mask > 0.5, sst, 999.0))),
        "SSH_min": float(np.min(np.where(mask > 0.5, eta, 999.0))),
        "SSH_max": float(np.max(np.where(mask > 0.5, eta, -999.0))),
        "u_max": float(np.max(np.abs(fields["u"]))),
        "v_max": float(np.max(np.abs(fields["v"]))),
        "speed_max": float(np.max(speed)),
    }


def _summarize_case_metrics(fields_final, diagnostics):
    first = diagnostics[0]
    last = diagnostics[-1]
    mask = np.asarray(fields_final["mask"], dtype=np.float64)
    land = mask < 0.5
    land_3d = land[..., None]

    finite_checks = [
        np.all(np.isfinite(fields_final["eta"])),
        np.all(np.isfinite(fields_final["T"])),
        np.all(np.isfinite(fields_final["S"])),
        np.all(np.isfinite(fields_final["u"])),
        np.all(np.isfinite(fields_final["v"])),
    ]
    all_finite = bool(np.all(finite_checks))

    if np.any(land):
        max_land_u = float(np.max(np.abs(np.where(land_3d, fields_final["u"], 0.0))))
        max_land_v = float(np.max(np.abs(np.where(land_3d, fields_final["v"], 0.0))))
        max_land_eta = float(np.max(np.abs(np.where(land, fields_final["eta"], 0.0))))
        land_zero = max(max_land_u, max_land_v, max_land_eta) <= 1.0e-10
    else:
        max_land_u = max_land_v = max_land_eta = 0.0
        land_zero = True

    ocean_area = max(float(first.get("ocean_area", 1.0)), 1.0)
    heat_drift_rel = (
        (last["heat"] - first["heat"]) / abs(first["heat"])
        if abs(first["heat"]) > 1.0e-12 else 0.0
    )
    salt_drift_rel = (
        (last["salt"] - first["salt"]) / abs(first["salt"])
        if abs(first["salt"]) > 1.0e-12 else 0.0
    )
    volume_mean_eta_drift = (last["volume"] - first["volume"]) / ocean_area

    return {
        "SSH_min": float(last["SSH_min"]),
        "SSH_max": float(last["SSH_max"]),
        "u_max": float(last["u_max"]),
        "v_max": float(last["v_max"]),
        "speed_max": float(last["speed_max"]),
        "SST_min": float(last["SST_min"]),
        "SST_max": float(last["SST_max"]),
        "SST_mean": float(last["SST_mean"]),
        "kinetic_energy": float(last["kinetic_energy"]),
        "heat_drift_rel": float(heat_drift_rel),
        "salt_drift_rel": float(salt_drift_rel),
        "volume_mean_eta_drift": float(volume_mean_eta_drift),
        "all_finite": bool(all_finite),
        "land_zero": bool(land_zero),
        "max_land_u": float(max_land_u),
        "max_land_v": float(max_land_v),
        "max_land_eta": float(max_land_eta),
    }


def _save_case_timeseries(case_dir, diagnostics, dt, n_steps):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    n_diag = len(diagnostics)
    times_days = np.linspace(0.0, n_steps * dt / 86400.0, n_diag)

    with open(case_dir / "timeseries.csv", "w") as f:
        f.write(
            "time_day,ssh_min,ssh_max,sst_mean,speed_max,kinetic_energy,volume,heat,salt\n",
        )
        for t, d in zip(times_days, diagnostics):
            f.write(
                f"{t:.8f},{d['SSH_min']:.12e},{d['SSH_max']:.12e},"
                f"{d['SST_mean']:.12e},{d['speed_max']:.12e},{d['kinetic_energy']:.12e},"
                f"{d['volume']:.12e},{d['heat']:.12e},{d['salt']:.12e}\n",
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(times_days, [d["SSH_max"] for d in diagnostics], "b-", label="SSH max")
    axes[0, 0].plot(times_days, [d["SSH_min"] for d in diagnostics], "c-", label="SSH min")
    axes[0, 0].set_ylabel("SSH [m]")
    axes[0, 0].set_title("Sea Surface Height Extremes")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    ke = np.array([d["kinetic_energy"] for d in diagnostics], dtype=np.float64)
    axes[0, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[0, 1].set_ylabel("Kinetic energy")
    axes[0, 1].set_title("Kinetic Energy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    axes[1, 0].plot(times_days, [d["speed_max"] for d in diagnostics], "r-", linewidth=1.5)
    axes[1, 0].set_ylabel("Max speed [m/s]")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_title("Maximum Velocity Magnitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times_days, [d["SST_mean"] for d in diagnostics], "g-", linewidth=1.5)
    axes[1, 1].set_ylabel("Mean SST [degC]")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Mean Sea Surface Temperature")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Mean-State Time Series", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(case_dir / "mean_state_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    vol = np.array([d["volume"] for d in diagnostics], dtype=np.float64)
    heat = np.array([d["heat"] for d in diagnostics], dtype=np.float64)
    salt = np.array([d["salt"] for d in diagnostics], dtype=np.float64)
    heat_rel = (heat - heat[0]) / max(abs(heat[0]), 1.0)
    salt_rel = (salt - salt[0]) / max(abs(salt[0]), 1.0)
    ocean_area = max(float(diagnostics[0].get("ocean_area", 1.0)), 1.0)
    eta_mean_drift = (vol - vol[0]) / ocean_area

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(times_days, vol, "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("Volume integral [m^3]")
    axes[0, 0].set_title("Volume Integral")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    axes[0, 1].plot(times_days, eta_mean_drift, "c-", linewidth=1.5)
    axes[0, 1].set_ylabel("Mean eta drift [m]")
    axes[0, 1].set_title("Volume Drift / Ocean Area")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    axes[1, 0].plot(times_days, heat_rel, "r-", linewidth=1.5)
    axes[1, 0].set_ylabel("Relative heat drift")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_title("Heat Conservation")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    axes[1, 1].plot(times_days, salt_rel, "g-", linewidth=1.5)
    axes[1, 1].set_ylabel("Relative salt drift")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Salt Conservation")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    fig.suptitle("Conservation Time Series", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(case_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()


def _pick_representative_ocean_point(mask, lat_deg, lon_deg):
    wet = np.argwhere(mask > 0.5)
    if wet.size == 0:
        raise ValueError("No wet cells found for profile diagnostics.")
    target_lon = 180.0
    dist = np.abs(lat_deg[wet[:, 0], wet[:, 1]]) + np.abs(
        ((lon_deg[wet[:, 0], wet[:, 1]] - target_lon + 180.0) % 360.0) - 180.0,
    )
    idx = int(np.argmin(dist))
    i, j = wet[idx]
    return int(i), int(j)


def _run_case(case_title, case_name, state0, model, grid, z_coord, config, dt, n_steps, output_dir):
    case_dir = Path(output_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    snapshot_steps = {0, max(1, n_steps // 4), max(1, n_steps // 2), max(1, (3 * n_steps) // 4), n_steps}
    diag_interval = max(1, n_steps // 50)

    print("\n" + "=" * 70)
    print(case_title)
    print("=" * 70)

    state = state0
    fields = _state_to_grid_fields(state, grid)
    diagnostics = [_compute_diagnostics(fields, grid, z_coord, config.min_water_column_m)]
    snapshots = {0: fields}
    profile_T = {}
    profile_speed = {}

    lat_deg_2d = np.asarray(grid.lat2d, dtype=np.float64) * 180.0 / np.pi
    lon_deg_2d = (np.asarray(grid.lon2d, dtype=np.float64) * 180.0 / np.pi + 360.0) % 360.0
    i0, j0 = _pick_representative_ocean_point(fields["mask"], lat_deg_2d, lon_deg_2d)
    profile_T[0] = fields["T"][i0, j0, :]
    profile_speed[0] = fields["speed"][i0, j0, :]

    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    warm = model.step(state, dt)
    jax.block_until_ready(warm.eta_hat.data)
    print(f"done ({time.time()-t0:.1f}s)")

    print(f"\n{'Step':>7s}  {'Day':>7s}  {'SSH_min':>9s}  {'SSH_max':>9s}  {'|v|_max':>9s}  {'SST_mean':>9s}  {'KE':>12s}")
    print("-" * 82)
    t_start = time.time()

    for step in range(1, n_steps + 1):
        state = model.step(state, dt)

        if step % diag_interval == 0 or step == n_steps or step in snapshot_steps:
            jax.block_until_ready(state.eta_hat.data)
            fields = _state_to_grid_fields(state, grid)
            if step % diag_interval == 0 or step == n_steps:
                diag = _compute_diagnostics(fields, grid, z_coord, config.min_water_column_m)
                diagnostics.append(diag)
                day = step * dt / 86400.0
                print(
                    f"{step:7d}  {day:7.2f}  {diag['SSH_min']:9.3e}  {diag['SSH_max']:9.3e}  "
                    f"{diag['speed_max']:9.3e}  {diag['SST_mean']:9.3e}  {diag['kinetic_energy']:12.4e}",
                )
            if step in snapshot_steps:
                snapshots[step] = fields
                profile_T[step] = fields["T"][i0, j0, :]
                profile_speed[step] = fields["speed"][i0, j0, :]

        if step % max(1, n_steps // 20) == 0:
            leaves = [
                np.asarray(state.vor_hat.data),
                np.asarray(state.div_hat.data),
                np.asarray(state.T_hat.data),
                np.asarray(state.S_hat.data),
                np.asarray(state.eta_hat.data),
            ]
            if not all(np.all(np.isfinite(x)) for x in leaves):
                raise RuntimeError(f"Non-finite spectral coefficients at step={step}")

    wall = time.time() - t_start
    print(f"  Completed in {wall:.1f}s ({n_steps / max(wall, 1.0e-12):.1f} steps/s)")

    fields_final = _state_to_grid_fields(state, grid)

    _plot_latlon_snapshots(
        case_dir / "ssh_snapshots.png",
        snapshots,
        dt,
        grid,
        lambda f: f["eta"],
        f"{case_title} — SSH Snapshots",
        "RdBu_r",
        "SSH [m]",
        symmetric=True,
    )
    _plot_latlon_snapshots(
        case_dir / "speed_surface_snapshots.png",
        snapshots,
        dt,
        grid,
        lambda f: f["speed"][..., 0],
        f"{case_title} — Surface Speed Snapshots",
        "magma",
        "Speed [m/s]",
        symmetric=False,
    )

    area = _cell_area(grid)
    mask = fields_final["mask"]
    lat_deg = np.asarray(grid.lat2d, dtype=np.float64) * 180.0 / np.pi
    lon_deg = (np.asarray(grid.lon2d, dtype=np.float64) * 180.0 / np.pi + 360.0) % 360.0
    wet_lat = lat_deg[mask > 0.5]
    lat_lo = float(np.nanmin(wet_lat)) if wet_lat.size else -90.0
    lat_hi = float(np.nanmax(wet_lat)) if wet_lat.size else 90.0
    n_lat_bins = int(np.clip(grid.n_lat, 12, 72))
    n_lon_bins = int(np.clip(grid.n_lon, 24, 180))
    lat_bins = np.linspace(lat_lo, lat_hi, n_lat_bins + 1)
    lon_bins = np.linspace(0.0, 360.0, n_lon_bins + 1)

    lat_centers, T_lat = _compute_binned_depth_section(
        fields_final["T"], lat_deg, area, mask, lat_bins, periodic=False,
    )
    _, speed_lat = _compute_binned_depth_section(
        fields_final["speed"], lat_deg, area, mask, lat_bins, periodic=False,
    )
    _plot_depth_sections(
        case_dir / "lat_depth_sections.png",
        z_coord,
        lat_centers,
        "Latitude [deg]",
        [
            ("Final temperature T", T_lat, "RdYlBu_r", False, "T [degC]"),
            ("Final speed", speed_lat, "magma", False, "Speed [m/s]"),
        ],
        f"{case_title} — Latitude-Depth Sections",
    )

    lon_centers, T_lon = _compute_binned_depth_section(
        fields_final["T"], lon_deg, area, mask, lon_bins, periodic=True,
    )
    _, speed_lon = _compute_binned_depth_section(
        fields_final["speed"], lon_deg, area, mask, lon_bins, periodic=True,
    )
    _plot_depth_sections(
        case_dir / "lon_depth_sections.png",
        z_coord,
        lon_centers,
        "Longitude [deg]",
        [
            ("Final temperature T", T_lon, "RdYlBu_r", False, "T [degC]"),
            ("Final speed", speed_lon, "magma", False, "Speed [m/s]"),
        ],
        f"{case_title} — Longitude-Depth Sections",
    )

    _plot_point_profile_evolution(
        case_dir / "T_profile_over_time.png",
        z_coord,
        profile_T,
        dt,
        "Temperature [degC]",
        f"{case_title} — Temperature Profile Evolution",
    )
    _plot_point_profile_evolution(
        case_dir / "speed_profile_over_time.png",
        z_coord,
        profile_speed,
        dt,
        "Speed [m/s]",
        f"{case_title} — Speed Profile Evolution",
    )

    _save_case_timeseries(case_dir, diagnostics, dt, n_steps)
    metrics = _summarize_case_metrics(fields_final, diagnostics)
    return metrics, diagnostics


def _make_rest_case(grid, z_coord):
    return rest_state_spectral_ocean(
        grid,
        z_coord,
        land_lat_threshold=90.0,
    )


def _make_gravity_wave_case(grid, z_coord):
    state = _make_rest_case(grid, z_coord)
    mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
    eta0 = np.asarray(sh_synthesis(grid, state.eta_hat.data).real, dtype=np.float64)
    lon = np.asarray(grid.lon2d, dtype=np.float64)
    lat = np.asarray(grid.lat2d, dtype=np.float64)
    lon0 = np.pi
    lat0 = 0.0
    dlon = ((lon - lon0 + np.pi) % (2.0 * np.pi)) - np.pi
    dlat = lat - lat0
    amp = 1.0
    width = 0.18
    eta_pert = amp * np.exp(-0.5 * ((dlon / width) ** 2 + (dlat / width) ** 2))
    eta = (eta0 + eta_pert) * mask
    return state._replace(
        eta_hat=state.eta_hat.replace(data=sh_analysis(grid, jnp.asarray(eta))),
    )


def _make_baroclinic_front_case(grid, z_coord):
    state = _make_rest_case(grid, z_coord)
    mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
    lat_deg = np.asarray(grid.lat2d, dtype=np.float64) * 180.0 / np.pi
    front_lat = 25.0
    front_width = 15.0
    front = 0.5 * (1.0 + np.tanh((lat_deg - front_lat) / (0.5 * front_width)))
    T_south = 22.0
    T_north = 8.0
    T_2d = T_south + (T_north - T_south) * front
    T_3d = np.broadcast_to(T_2d[..., None], (grid.n_lat, grid.n_lon, z_coord.n_levels))
    T_3d = T_3d * mask[..., None]
    return state._replace(
        T_hat=state.T_hat.replace(data=sh_analysis_3d(grid, jnp.asarray(T_3d))),
    )


def main():
    parser = argparse.ArgumentParser(description="Spectral ocean test suite")
    parser.add_argument("--x64", action="store_true", help="Enable float64 mode (required for spectral).")
    parser.add_argument("--truncation", "-n", type=int, default=8, help="Spectral triangular truncation Tn.")
    parser.add_argument("--levels", "-l", type=int, default=10, help="Vertical levels.")
    parser.add_argument("--dt", type=float, default=1800.0, help="Time step [s].")
    parser.add_argument("--days", "-d", type=float, default=3.0, help="Integration length [days].")
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        default="all",
        choices=["all", "rest", "wave", "baroclinic"],
        help="Which test to run.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: results/ocean_spectral_tests_T{n}_L{l}).",
    )
    args = parser.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise ValueError(
            "Spectral ocean requires float64/complex128. Re-run with --x64 or set JAX_ENABLE_X64=1.",
        )
    if args.truncation < 2:
        raise ValueError(f"--truncation must be >= 2, got {args.truncation!r}")
    if args.levels < 1:
        raise ValueError(f"--levels must be >= 1, got {args.levels!r}")
    if args.dt <= 0.0:
        raise ValueError(f"--dt must be > 0, got {args.dt!r}")
    if args.days <= 0.0:
        raise ValueError(f"--days must be > 0, got {args.days!r}")

    n_steps = int(args.days * 86400.0 / args.dt)
    if n_steps < 1:
        raise ValueError(
            "Integration has zero steps. Increase --days or reduce --dt "
            f"(got days={args.days}, dt={args.dt}).",
        )

    output_dir = Path(args.output or f"results/ocean_spectral_tests_T{args.truncation}_L{args.levels}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("legoESM — Spectral Ocean Tests")
    print("=" * 70)
    print(f"  Backend:      {jax.default_backend()}")
    print(f"  Devices:      {jax.device_count()}")
    print(f"  Float dtype:  {jnp.zeros(1).dtype}")
    print(f"  Truncation:   T{args.truncation}")
    print(f"  Levels:       {args.levels}")
    print(f"  Integration:  {args.days:.2f} days, dt={args.dt:.0f}s, steps={n_steps}")

    print("\nCreating Gaussian grid...")
    t0 = time.time()
    grid = create_gaussian_grid(args.truncation, allow_unsupported_backend=True)
    print(f"  Grid {grid.n_lat}x{grid.n_lon} created in {time.time() - t0:.1f}s")

    print("Creating z-star coordinate...")
    z_coord = create_ocean_z_star(n_levels=args.levels)
    print(f"  Surface dz={float(z_coord.dz_ref[0]):.1f} m, bottom dz={float(z_coord.dz_ref[-1]):.1f} m")

    config = SpectralOceanConfig(
        A_h=1.0e4,
        K_h=1.0e3,
        A_v=1.0e-3,
        K_v=1.0e-4,
        hyperdiff_coeff=1.0e15,
        hyperdiff_order=2,
        use_conservation_fixer=True,
        min_water_column_m=0.5,
    )
    model = SpectralOceanModel(grid, z_coord, config, allow_unsupported_backend=True)

    cases = []
    if args.test in ("all", "rest"):
        cases.append(("rest_state", "TEST 1: Spectral Rest State", _make_rest_case))
    if args.test in ("all", "wave"):
        cases.append(("gravity_wave", "TEST 2: Spectral Barotropic Gravity Wave", _make_gravity_wave_case))
    if args.test in ("all", "baroclinic"):
        cases.append(("baroclinic_adjustment", "TEST 3: Spectral Baroclinic Adjustment", _make_baroclinic_front_case))

    summary_cases = {}
    case_errors = {}
    for case_name, case_title, init_fn in cases:
        try:
            state0 = init_fn(grid, z_coord)
            metrics, _ = _run_case(
                case_title=case_title,
                case_name=case_name,
                state0=state0,
                model=model,
                grid=grid,
                z_coord=z_coord,
                config=config,
                dt=args.dt,
                n_steps=n_steps,
                output_dir=output_dir,
            )
            summary_cases[case_name] = metrics
        except Exception as exc:  # pragma: no cover
            case_errors[case_name] = f"{type(exc).__name__}: {exc}"
            print(f"\n  !!! CASE FAILED: {case_name} -> {case_errors[case_name]}")
            traceback.print_exc()

    summary = {
        "suite": "ocean_spectral_tests",
        "meta": {
            "backend": jax.default_backend(),
            "device_count": int(jax.device_count()),
            "float_dtype": str(jnp.zeros(1).dtype),
            "truncation": int(args.truncation),
            "levels": int(args.levels),
            "dt": float(args.dt),
            "days": float(args.days),
            "n_steps": int(n_steps),
            "n_lat": int(grid.n_lat),
            "n_lon": int(grid.n_lon),
            "n_sh": int(grid.n_sh),
        },
        "config": {
            "A_h": float(config.A_h),
            "K_h": float(config.K_h),
            "A_v": float(config.A_v),
            "K_v": float(config.K_v),
            "hyperdiff_coeff": float(config.hyperdiff_coeff),
            "hyperdiff_order": int(config.hyperdiff_order),
            "use_conservation_fixer": bool(config.use_conservation_fixer),
            "min_water_column_m": float(config.min_water_column_m),
        },
        "cases": summary_cases,
        "case_errors": case_errors,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(output_dir / "summary.txt", "w") as f:
        f.write("legoESM Spectral Ocean Tests\n")
        f.write("=" * 34 + "\n")
        f.write(f"T{args.truncation}, levels={args.levels}, dt={args.dt:.0f}s, days={args.days:.2f}, steps={n_steps}\n")
        f.write(f"n_lat={grid.n_lat}, n_lon={grid.n_lon}, n_sh={grid.n_sh}\n\n")
        for name, m in summary_cases.items():
            f.write(f"{name}\n")
            f.write(
                f"  SSH=[{m['SSH_min']:.4e}, {m['SSH_max']:.4e}] "
                f"u_max={m['u_max']:.4e} v_max={m['v_max']:.4e} "
                f"speed_max={m['speed_max']:.4e}\n",
            )
            f.write(
                f"  heat_drift_rel={m['heat_drift_rel']:.3e} "
                f"salt_drift_rel={m['salt_drift_rel']:.3e} "
                f"eta_mean_drift={m['volume_mean_eta_drift']:.3e}\n",
            )
            f.write(f"  all_finite={m['all_finite']} land_zero={m['land_zero']}\n\n")
        if case_errors:
            f.write("failed_cases\n")
            for name, msg in case_errors.items():
                f.write(f"  {name}: {msg}\n")
            f.write("\n")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, m in summary_cases.items():
        print(f"  {name}: speed_max={m['speed_max']:.3e}, heat_drift_rel={m['heat_drift_rel']:.3e}, all_finite={m['all_finite']}")
    if case_errors:
        print("  failed_cases:")
        for name, msg in case_errors.items():
            print(f"    - {name}: {msg}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)
    return 1 if case_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
