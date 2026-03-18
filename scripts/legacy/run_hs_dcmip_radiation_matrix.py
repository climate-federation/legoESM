#!/usr/bin/env python
"""Run Held-Suarez and DCMIP radiation matrix with diagnostics.

Coverage:
- Held-Suarez hydrostatic:
  - FV cubed-sphere
  - FV lat-lon
  - Spectral (gaussian lat-lon)
- DCMIP-2025 non-hydrostatic (FV cubed-sphere):
  - TC1, TC2a, TC3
- Non-hydrostatic FV lat-lon:
  - Dry-rest dynamical stability (diagnostic coverage case)

Radiation schemes:
- gray
- rrtmgp

Diagnostics per case:
- field_snapshots.png
- snapshot_times.txt
- integrated_timeseries.csv / integrated_timeseries.png
- conservation_timeseries.csv / conservation_timeseries.png
- vertical_profiles.csv / vertical_profiles.png
- zonal_cross_sections.png (latitude vs sigma/height, averaged across longitudes)
- meridional_cross_sections.png (longitude vs sigma/height, averaged across latitudes)
- results.txt
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree  # type: ignore

    HAS_SCIPY = True
except Exception:
    cKDTree = None
    HAS_SCIPY = False

# Reuse plotting/time-series helpers already used by atmosphere dycore suite.
try:
    import scripts.run_atmosphere_25deg_ssp45_full as atm25
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    atm25 = importlib.import_module("scripts.run_atmosphere_25deg_ssp45_full")

from legoesm import constants
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.core.conservation import (
    compute_hydrostatic_energy,
    compute_nh_dry_mass,
    compute_nh_energy,
)


@dataclass(frozen=True)
class ResolutionPreset:
    name: str
    cube_resolution: int
    latlon_nlat: int
    latlon_nlon: int
    spectral_truncation: int
    hydro_levels: int
    nh_levels: int


PRESETS = {
    "low": ResolutionPreset(
        name="low",
        cube_resolution=8,
        latlon_nlat=24,
        latlon_nlon=48,
        spectral_truncation=10,
        hydro_levels=10,
        nh_levels=10,
    ),
    "medium": ResolutionPreset(
        name="medium",
        cube_resolution=12,
        latlon_nlat=36,
        latlon_nlon=72,
        spectral_truncation=15,
        hydro_levels=15,
        nh_levels=15,
    ),
    "high": ResolutionPreset(
        name="high",
        cube_resolution=24,
        latlon_nlat=120,
        latlon_nlon=240,
        spectral_truncation=85,
        hydro_levels=20,
        nh_levels=20,
    ),
    "atm2deg": ResolutionPreset(
        name="atm2deg",
        cube_resolution=32,
        latlon_nlat=90,
        latlon_nlon=180,
        spectral_truncation=63,
        hydro_levels=20,
        nh_levels=20,
    ),
}


def _gaussian_area_weights(grid) -> np.ndarray:
    w_lat = np.asarray(grid.weights, dtype=np.float64)
    return np.broadcast_to(w_lat[:, None] / float(grid.n_lon), (w_lat.size, int(grid.n_lon)))


def _regrid_faces_to_latlon(
    field_2d_faces: np.ndarray,
    lon_faces_deg: np.ndarray,
    lat_faces_deg: np.ndarray,
    n_lon_out: int = 240,
    n_lat_out: int = 121,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.asarray(lon_faces_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(lat_faces_deg, dtype=np.float64).reshape(-1)
    val = np.asarray(field_2d_faces, dtype=np.float64).reshape(-1)

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon = ((lon[valid] + 180.0) % 360.0) - 180.0
    lat = np.clip(lat[valid], -90.0, 90.0)
    val = val[valid]

    lon_cent = np.linspace(-180.0, 180.0, n_lon_out, endpoint=False) + 180.0 / n_lon_out
    lat_cent = np.linspace(-90.0, 90.0, n_lat_out)
    lon2d, lat2d = np.meshgrid(lon_cent, lat_cent)

    if HAS_SCIPY:
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        src_xyz = np.column_stack(
            [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)],
        )
        lon_t = np.deg2rad(lon2d.reshape(-1))
        lat_t = np.deg2rad(lat2d.reshape(-1))
        tgt_xyz = np.column_stack(
            [np.cos(lat_t) * np.cos(lon_t), np.cos(lat_t) * np.sin(lon_t), np.sin(lat_t)],
        )
        tree = cKDTree(src_xyz)
        k = min(8, src_xyz.shape[0])
        dist, idx = tree.query(tgt_xyz, k=k)
        if k == 1:
            field_ll = val[idx].reshape(lon2d.shape)
        else:
            dist = np.maximum(dist, 1.0e-12)
            w = 1.0 / dist
            w /= np.sum(w, axis=1, keepdims=True)
            field_ll = np.sum(val[idx] * w, axis=1).reshape(lon2d.shape)
        return lon2d, lat2d, field_ll

    lon_edges = np.linspace(-180.0, 180.0, n_lon_out + 1)
    lat_edges = np.linspace(-90.0, 90.0, n_lat_out + 1)
    sum_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges), weights=val)
    cnt_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        field_ll = np.where(cnt_grid > 0.0, sum_grid / cnt_grid, np.nan)
    return lon2d, lat2d, field_ll


def _latbin_mean_3d(
    field_3d: np.ndarray,
    lat_deg_2d: np.ndarray,
    area_2d: np.ndarray,
    n_lat_bins: int = 121,
) -> tuple[np.ndarray, np.ndarray]:
    lat_flat = np.asarray(lat_deg_2d, dtype=np.float64).reshape(-1)
    area_flat = np.asarray(area_2d, dtype=np.float64).reshape(-1)
    fld = np.asarray(field_3d, dtype=np.float64).reshape(lat_flat.size, -1)

    edges = np.linspace(-90.0, 90.0, n_lat_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full((n_lat_bins, fld.shape[1]), np.nan, dtype=np.float64)

    for i in range(n_lat_bins):
        if i == n_lat_bins - 1:
            mask = (lat_flat >= edges[i]) & (lat_flat <= edges[i + 1])
        else:
            mask = (lat_flat >= edges[i]) & (lat_flat < edges[i + 1])
        if not np.any(mask):
            continue
        w = area_flat[mask]
        wsum = np.sum(w)
        if wsum <= 0.0:
            continue
        prof[i, :] = np.sum(fld[mask, :] * w[:, None], axis=0) / wsum

    # Fill empty bins by 1D interpolation per level.
    x = centers
    for k in range(prof.shape[1]):
        y = prof[:, k]
        valid = np.isfinite(y)
        if not np.any(valid):
            prof[:, k] = 0.0
            continue
        if np.sum(valid) == 1:
            prof[:, k] = y[valid][0]
            continue
        prof[:, k] = np.interp(x, x[valid], y[valid])

    return centers, prof


def _lonbin_mean_3d(
    field_3d: np.ndarray,
    lon_deg_2d: np.ndarray,
    area_2d: np.ndarray,
    n_lon_bins: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    lon_flat = np.asarray(lon_deg_2d, dtype=np.float64).reshape(-1)
    lon_flat = np.mod(lon_flat, 360.0)
    area_flat = np.asarray(area_2d, dtype=np.float64).reshape(-1)
    fld = np.asarray(field_3d, dtype=np.float64).reshape(lon_flat.size, -1)

    edges = np.linspace(0.0, 360.0, n_lon_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full((n_lon_bins, fld.shape[1]), np.nan, dtype=np.float64)

    for i in range(n_lon_bins):
        if i == n_lon_bins - 1:
            mask = (lon_flat >= edges[i]) & (lon_flat <= edges[i + 1])
        else:
            mask = (lon_flat >= edges[i]) & (lon_flat < edges[i + 1])
        if not np.any(mask):
            continue
        w = area_flat[mask]
        wsum = np.sum(w)
        if wsum <= 0.0:
            continue
        prof[i, :] = np.sum(fld[mask, :] * w[:, None], axis=0) / wsum

    x = centers
    for k in range(prof.shape[1]):
        y = prof[:, k]
        valid = np.isfinite(y)
        if not np.any(valid):
            prof[:, k] = 0.0
            continue
        if np.sum(valid) == 1:
            prof[:, k] = y[valid][0]
            continue
        x_valid = x[valid]
        y_valid = y[valid]
        x_ext = np.concatenate([x_valid - 360.0, x_valid, x_valid + 360.0])
        y_ext = np.concatenate([y_valid, y_valid, y_valid])
        prof[:, k] = np.interp(x, x_ext, y_ext)

    return centers, prof


def _save_conservation(out_dir: Path, case_name: str, rows: list[dict[str, float]]) -> tuple[float, float]:
    if not rows:
        return float("nan"), float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "conservation_timeseries.csv"
    with out_csv.open("w") as f:
        f.write("step,time_days,mass,energy,mass_rel,energy_rel\n")
        for r in rows:
            f.write(
                f"{int(r['step'])},{r['time_days']:.8f},{r['mass']:.12e},{r['energy']:.12e},"
                f"{r['mass_rel']:.12e},{r['energy_rel']:.12e}\n",
            )

    t = np.asarray([r["time_days"] for r in rows], dtype=float)
    mrel = np.asarray([r["mass_rel"] for r in rows], dtype=float)
    erel = np.asarray([r["energy_rel"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, mrel, lw=1.8, color="tab:blue")
    axes[0].axhline(0.0, color="0.3", lw=0.8, ls="--")
    axes[0].set_ylabel("Mass drift (rel.)")
    axes[0].set_title("Mass conservation")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, erel, lw=1.8, color="tab:red")
    axes[1].axhline(0.0, color="0.3", lw=0.8, ls="--")
    axes[1].set_ylabel("Energy drift (rel.)")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title("Energy conservation")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{case_name} - Conservation", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return float(mrel[-1]), float(erel[-1])


def _save_zonal_cross_sections(
    out_dir: Path,
    case_name: str,
    cross_sections: dict[int, dict[str, np.ndarray]],
    dt: float,
    level_values: np.ndarray,
    level_label: str,
    invert_y: bool,
) -> None:
    if not cross_sections:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(cross_sections.keys())
    keys = [k for k in cross_sections[steps[0]].keys() if k != "lat_deg"]
    n_rows = len(keys)
    n_cols = len(steps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), sharey=True)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = np.array([[axes[r]] for r in range(n_rows)])

    for r, key in enumerate(keys):
        values = [np.asarray(cross_sections[s][key], dtype=np.float64) for s in steps]
        vmin = float(min(np.nanmin(v) for v in values))
        vmax = float(max(np.nanmax(v) for v in values))
        if vmin < 0.0 < vmax:
            m = max(abs(vmin), abs(vmax), 1.0e-12)
            vmin, vmax = -m, m
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin), 1.0) * 1.0e-6
            vmin -= pad
            vmax += pad

        for c, st in enumerate(steps):
            ax = axes[r, c]
            lat = np.asarray(cross_sections[st]["lat_deg"], dtype=np.float64)
            fld = np.asarray(cross_sections[st][key], dtype=np.float64)
            im = ax.pcolormesh(lat, level_values, fld.T, cmap="RdBu_r" if vmin < 0 < vmax else "viridis",
                               vmin=vmin, vmax=vmax, shading="auto")
            if invert_y:
                ax.invert_yaxis()
            if r == n_rows - 1:
                ax.set_xlabel("Latitude [deg]")
            if c == 0:
                ax.set_ylabel(level_label)
            ax.set_title(f"{key}\n{atm25._format_sim_time(st, dt)}", fontsize=9)
            ax.grid(True, alpha=0.2)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"{case_name} - Zonal Mean Cross Sections", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "zonal_cross_sections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_meridional_cross_sections(
    out_dir: Path,
    case_name: str,
    cross_sections: dict[int, dict[str, np.ndarray]],
    dt: float,
    level_values: np.ndarray,
    level_label: str,
    invert_y: bool,
) -> None:
    if not cross_sections:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(cross_sections.keys())
    keys = [k for k in cross_sections[steps[0]].keys() if k != "lon_deg"]
    n_rows = len(keys)
    n_cols = len(steps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), sharey=True)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = np.array([[axes[r]] for r in range(n_rows)])

    for r, key in enumerate(keys):
        values = [np.asarray(cross_sections[s][key], dtype=np.float64) for s in steps]
        vmin = float(min(np.nanmin(v) for v in values))
        vmax = float(max(np.nanmax(v) for v in values))
        if vmin < 0.0 < vmax:
            m = max(abs(vmin), abs(vmax), 1.0e-12)
            vmin, vmax = -m, m
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin), 1.0) * 1.0e-6
            vmin -= pad
            vmax += pad

        for c, st in enumerate(steps):
            ax = axes[r, c]
            lon = np.asarray(cross_sections[st]["lon_deg"], dtype=np.float64)
            fld = np.asarray(cross_sections[st][key], dtype=np.float64)
            im = ax.pcolormesh(lon, level_values, fld.T, cmap="RdBu_r" if vmin < 0 < vmax else "viridis",
                               vmin=vmin, vmax=vmax, shading="auto")
            if invert_y:
                ax.invert_yaxis()
            if r == n_rows - 1:
                ax.set_xlabel("Longitude [deg]")
            if c == 0:
                ax.set_ylabel(level_label)
            ax.set_title(f"{key}\n{atm25._format_sim_time(st, dt)}", fontsize=9)
            ax.grid(True, alpha=0.2)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"{case_name} - Meridional Mean Cross Sections", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "meridional_cross_sections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_radiation_only_physics(scheme: str, model_type: str, dt: float):
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme=scheme),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    return make_physics(cfg, model_type=model_type, dt=dt)


def _write_results_txt(path: Path, rows: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for k, v in rows.items():
            f.write(f"{k}: {v}\n")


def _safe_hydro_energy(state, grid, sigma) -> float:
    try:
        return float(compute_hydrostatic_energy(state, grid, sigma)["total_energy"])
    except Exception:
        return float("nan")


def run_hs_fv_cube(case_dir: Path, preset: ResolutionPreset, scheme: str, days: float, mean_every: int, solver: str) -> dict[str, object]:
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq import PrimitiveEquationModel, PrimitiveEquationConfig
    from legoesm.atmosphere.physics.held_suarez import held_suarez_init

    dt = 600.0
    case_dir.mkdir(parents=True, exist_ok=True)
    n_steps = int(days * 86400.0 / dt)
    n = preset.cube_resolution
    nlev = preset.hydro_levels

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    model = PrimitiveEquationModel(
        grid,
        sigma,
        PrimitiveEquationConfig(
            hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
            hyperdiff_ps_coeff=5.0e16 * (48.0 / n) ** 4,
            use_conservation_fixer=True,
            fix_mass=True,
            time_integrator=solver,
            edge_blend_uv=0.15,
            edge_blend_T=0.10,
            edge_blend_p_s=0.20,
            edge_blend_width=2,
        ),
    )
    physics_fn = _build_radiation_only_physics(scheme, "hydrostatic", dt)

    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    sigma_full = np.asarray(sigma.sigma_full, dtype=np.float64)
    dsigma = np.asarray(sigma.dsigma, dtype=np.float64)

    snap_targets = atm25._snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {}
    cross_sections: dict[int, dict[str, np.ndarray]] = {}
    series = atm25._series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])
    cons_rows: list[dict[str, float]] = []

    mass0 = float(np.sum(np.asarray(state.p_s.data) * area))
    energy0 = _safe_hydro_energy(state, grid, sigma)

    def _extract_snapshot(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        return {
            "wind_sfc": np.sqrt(u[..., -1] * u[..., -1] + v[..., -1] * v[..., -1]),
            "p_s": np.asarray(s.p_s.data),
            "T_sfc": np.asarray(s.T.data)[..., -1],
        }

    def _extract_profile(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        T = np.asarray(s.T.data)
        wind = np.sqrt(u * u + v * v)
        return {
            "wind_profile": atm25._horizontal_profile_3d(wind, area),
            "T_profile": atm25._horizontal_profile_3d(T, area),
        }

    def _extract_cross(s):
        u = np.asarray(s.u.data)
        T = np.asarray(s.T.data)
        lat_cent, u_z = _latbin_mean_3d(u, lat_deg, area)
        lon_cent, u_m = _lonbin_mean_3d(u, lon_deg, area)
        _, T_z = _latbin_mean_3d(T, lat_deg, area)
        _, T_m = _lonbin_mean_3d(T, lon_deg, area)
        return {
            "lat_deg": lat_cent,
            "u_zonal": u_z,
            "T_zonal": T_z,
            "lon_deg": lon_cent,
            "u_meridional": u_m,
            "T_meridional": T_m,
        }

    def _record(step_i: int, s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        T = np.asarray(s.T.data)
        p_s = np.asarray(s.p_s.data)
        wind_prof = atm25._horizontal_profile_3d(np.sqrt(u * u + v * v), area)
        T_prof = atm25._horizontal_profile_3d(T, area)
        atm25._series_push(
            series,
            step_i,
            {
                "mean_wind_3d": atm25._vertical_integral_from_profile(wind_prof, dsigma),
                "mean_T_3d": atm25._vertical_integral_from_profile(T_prof, dsigma),
                "mean_p_s": atm25._weighted_mean_2d(p_s, area),
            },
        )
        mass = float(np.sum(p_s * area))
        energy = _safe_hydro_energy(s, grid, sigma)
        cons_rows.append(
            {
                "step": float(step_i),
                "time_days": float(step_i * dt / 86400.0),
                "mass": mass,
                "energy": energy,
                "mass_rel": (mass - mass0) / max(abs(mass0), 1.0e-30),
                "energy_rel": (energy - energy0) / max(abs(energy0), 1.0e-30),
            },
        )

    snapshots[0] = _extract_snapshot(state)
    profiles[0] = _extract_profile(state)
    cross_sections[0] = _extract_cross(state)
    _record(0, state)

    stable = True
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)
        st = i + 1
        if st in snap_targets:
            snapshots[st] = _extract_snapshot(state)
            profiles[st] = _extract_profile(state)
            cross_sections[st] = _extract_cross(state)
        if st % max(1, mean_every) == 0 or st == n_steps:
            _record(st, state)
        if st % progress_every == 0:
            print(f"      hs_fv_cube progress: {st}/{n_steps}")
        if not bool(jnp.all(jnp.isfinite(state.u.data))):
            stable = False
            break
    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    case_name = f"Held-Suarez FV Cube ({scheme}, {preset.name})"
    atm25._save_snapshots(
        case_dir,
        case_name,
        snapshots,
        dt,
        [
            ("wind_sfc", "Surface wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
            ("T_sfc", "Surface temperature (K)", "coolwarm"),
        ],
        coord_kind="cube",
        lon_deg=lon_deg,
        lat_deg=lat_deg,
    )
    atm25._save_timeseries(
        case_dir,
        case_name,
        series,
        dt,
        {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
    )
    atm25._save_profiles(
        case_dir,
        case_name,
        profiles,
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
        units={"wind_profile": "m/s", "T_profile": "K"},
    )
    mass_end, energy_end = _save_conservation(case_dir, case_name, cons_rows)
    _save_zonal_cross_sections(
        case_dir,
        case_name,
        {k: {"lat_deg": v["lat_deg"], "u_zonal": v["u_zonal"], "T_zonal": v["T_zonal"]} for k, v in cross_sections.items()},
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
    )
    _save_meridional_cross_sections(
        case_dir,
        case_name,
        {k: {"lon_deg": v["lon_deg"], "u_meridional": v["u_meridional"], "T_meridional": v["T_meridional"]} for k, v in cross_sections.items()},
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
    )

    _write_results_txt(
        case_dir / "results.txt",
        {
            "case": "hs_fv_cube",
            "scheme": scheme,
            "preset": preset.name,
            "resolution": f"C{n}",
            "levels": nlev,
            "dt": dt,
            "days": days,
            "n_steps": n_steps,
            "stable": stable,
            "mass_drift_rel": f"{mass_end:.12e}",
            "energy_drift_rel": f"{energy_end:.12e}",
            "wall_time_s": f"{wall:.2f}",
        },
    )
    return {
        "case": "hs_fv_cube",
        "scheme": scheme,
        "preset": preset.name,
        "status": "PASS" if stable else "FAIL",
        "stable": stable,
        "mass_drift_rel": mass_end,
        "energy_drift_rel": energy_end,
        "wall_time_s": wall,
        "output": str(case_dir),
    }


def run_hs_fv_latlon(case_dir: Path, preset: ResolutionPreset, scheme: str, days: float, mean_every: int, solver: str) -> dict[str, object]:
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_latlon import LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig
    from legoesm.atmosphere.physics.held_suarez_latlon import held_suarez_init_latlon

    dt = 600.0
    case_dir.mkdir(parents=True, exist_ok=True)
    n_steps = int(days * 86400.0 / dt)
    nlat = preset.latlon_nlat
    nlon = preset.latlon_nlon
    nlev = preset.hydro_levels

    grid = create_latlon_grid(nlat, nlon)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init_latlon(grid, sigma)
    model = LatLonPrimitiveEquationModel(
        grid,
        sigma,
        LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=2.0e16 * (64.0 / nlat) ** 4,
            hyperdiff_ps_coeff=2.0e16 * (64.0 / nlat) ** 4,
            use_conservation_fixer=True,
            fix_mass=True,
            use_polar_filter=True,
            polar_filter_cutoff_deg=60.0,
        ),
        dt=dt,
    )
    physics_fn = _build_radiation_only_physics(scheme, "hydrostatic", dt)

    area = np.asarray(grid.area)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    sigma_full = np.asarray(sigma.sigma_full, dtype=np.float64)
    dsigma = np.asarray(sigma.dsigma, dtype=np.float64)

    snap_targets = atm25._snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {}
    cross_sections: dict[int, dict[str, np.ndarray]] = {}
    series = atm25._series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])
    cons_rows: list[dict[str, float]] = []

    mass0 = float(np.sum(np.asarray(state.p_s.data) * area))
    energy0 = _safe_hydro_energy(state, grid, sigma)

    def _extract_snapshot(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        return {
            "wind_sfc": np.sqrt(u[..., -1] * u[..., -1] + v[..., -1] * v[..., -1]),
            "p_s": np.asarray(s.p_s.data),
            "T_sfc": np.asarray(s.T.data)[..., -1],
        }

    def _extract_profile(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        T = np.asarray(s.T.data)
        wind = np.sqrt(u * u + v * v)
        return {
            "wind_profile": atm25._horizontal_profile_3d(wind, area),
            "T_profile": atm25._horizontal_profile_3d(T, area),
        }

    def _extract_cross(s):
        u = np.asarray(s.u.data)
        T = np.asarray(s.T.data)
        lon2d = np.broadcast_to(lon_deg[None, :], (u.shape[0], u.shape[1]))
        lon_cent, u_m = _lonbin_mean_3d(u, lon2d, area, n_lon_bins=u.shape[1])
        _, T_m = _lonbin_mean_3d(T, lon2d, area, n_lon_bins=u.shape[1])
        return {
            "lat_deg": lat_deg,
            "u_zonal": np.mean(u, axis=1),
            "T_zonal": np.mean(T, axis=1),
            "lon_deg": lon_cent,
            "u_meridional": u_m,
            "T_meridional": T_m,
        }

    def _record(step_i: int, s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        T = np.asarray(s.T.data)
        p_s = np.asarray(s.p_s.data)
        wind_prof = atm25._horizontal_profile_3d(np.sqrt(u * u + v * v), area)
        T_prof = atm25._horizontal_profile_3d(T, area)
        atm25._series_push(
            series,
            step_i,
            {
                "mean_wind_3d": atm25._vertical_integral_from_profile(wind_prof, dsigma),
                "mean_T_3d": atm25._vertical_integral_from_profile(T_prof, dsigma),
                "mean_p_s": atm25._weighted_mean_2d(p_s, area),
            },
        )
        mass = float(np.sum(p_s * area))
        energy = _safe_hydro_energy(s, grid, sigma)
        cons_rows.append(
            {
                "step": float(step_i),
                "time_days": float(step_i * dt / 86400.0),
                "mass": mass,
                "energy": energy,
                "mass_rel": (mass - mass0) / max(abs(mass0), 1.0e-30),
                "energy_rel": (energy - energy0) / max(abs(energy0), 1.0e-30),
            },
        )

    snapshots[0] = _extract_snapshot(state)
    profiles[0] = _extract_profile(state)
    cross_sections[0] = _extract_cross(state)
    _record(0, state)

    stable = True
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)
        st = i + 1
        if st in snap_targets:
            snapshots[st] = _extract_snapshot(state)
            profiles[st] = _extract_profile(state)
            cross_sections[st] = _extract_cross(state)
        if st % max(1, mean_every) == 0 or st == n_steps:
            _record(st, state)
        if st % progress_every == 0:
            print(f"      hs_fv_latlon progress: {st}/{n_steps}")
        if not bool(jnp.all(jnp.isfinite(state.u.data))):
            stable = False
            break
    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    case_name = f"Held-Suarez FV LatLon ({scheme}, {preset.name})"
    atm25._save_snapshots(
        case_dir,
        case_name,
        snapshots,
        dt,
        [
            ("wind_sfc", "Surface wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
            ("T_sfc", "Surface temperature (K)", "coolwarm"),
        ],
        coord_kind="latlon",
        lon_deg=lon_deg,
        lat_deg=lat_deg,
    )
    atm25._save_timeseries(
        case_dir,
        case_name,
        series,
        dt,
        {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
    )
    atm25._save_profiles(
        case_dir,
        case_name,
        profiles,
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
        units={"wind_profile": "m/s", "T_profile": "K"},
    )
    mass_end, energy_end = _save_conservation(case_dir, case_name, cons_rows)
    _save_zonal_cross_sections(
        case_dir,
        case_name,
        {k: {"lat_deg": v["lat_deg"], "u_zonal": v["u_zonal"], "T_zonal": v["T_zonal"]} for k, v in cross_sections.items()},
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
    )
    _save_meridional_cross_sections(
        case_dir,
        case_name,
        {k: {"lon_deg": v["lon_deg"], "u_meridional": v["u_meridional"], "T_meridional": v["T_meridional"]} for k, v in cross_sections.items()},
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
    )

    _write_results_txt(
        case_dir / "results.txt",
        {
            "case": "hs_fv_latlon",
            "scheme": scheme,
            "preset": preset.name,
            "resolution": f"{nlat}x{nlon}",
            "levels": nlev,
            "dt": dt,
            "days": days,
            "n_steps": n_steps,
            "stable": stable,
            "mass_drift_rel": f"{mass_end:.12e}",
            "energy_drift_rel": f"{energy_end:.12e}",
            "wall_time_s": f"{wall:.2f}",
        },
    )
    return {
        "case": "hs_fv_latlon",
        "scheme": scheme,
        "preset": preset.name,
        "status": "PASS" if stable else "FAIL",
        "stable": stable,
        "mass_drift_rel": mass_end,
        "energy_drift_rel": energy_end,
        "wall_time_s": wall,
        "output": str(case_dir),
    }


def run_hs_spectral(case_dir: Path, preset: ResolutionPreset, scheme: str, days: float, mean_every: int) -> dict[str, object]:
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPEConfig,
        SpectralPrimitiveEquationModel,
        isothermal_rest_state_spectral,
        spectral_pe_to_grid,
    )

    dt = 240.0
    case_dir.mkdir(parents=True, exist_ok=True)
    n_steps = int(days * 86400.0 / dt)
    trunc = preset.spectral_truncation
    nlev = preset.hydro_levels

    grid = create_gaussian_grid(trunc)
    sigma = create_sigma_coordinate(nlev)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0, p_s_init=1.0e5)
    a = float(grid.radius)
    eig_max = trunc * (trunc + 1) / (a * a)
    model = SpectralPrimitiveEquationModel(
        grid,
        sigma,
        SpectralPEConfig(
            hyperdiff_coeff=1.0 / (0.5 * 3600.0 * eig_max ** 2),
            hyperdiff_order=2,
            semi_implicit=False,
            time_integrator="ssp_rk54",
        ),
    )
    physics_fn = _build_radiation_only_physics(scheme, "spectral_pe", dt)

    area = _gaussian_area_weights(grid)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    sigma_full = np.asarray(sigma.sigma_full, dtype=np.float64)
    dsigma = np.asarray(sigma.dsigma, dtype=np.float64)

    snap_targets = atm25._snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {}
    cross_sections: dict[int, dict[str, np.ndarray]] = {}
    series = atm25._series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])
    cons_rows: list[dict[str, float]] = []

    def _to_grid(s):
        return spectral_pe_to_grid(s, grid, sigma)

    f0 = _to_grid(state)
    mass0 = float(np.sum(np.asarray(f0["p_s"]) * area))
    ke0 = np.sum(
        np.sum(0.5 * (np.asarray(f0["u"]) ** 2 + np.asarray(f0["v"]) ** 2) * np.asarray(f0["p_s"])[..., None] * dsigma[None, None, :], axis=-1)
        * area,
    )
    ie0 = np.sum(
        np.sum(constants.c_vd * np.asarray(f0["T"]) * np.asarray(f0["p_s"])[..., None] * dsigma[None, None, :], axis=-1)
        * area,
    )
    energy0 = float((ke0 + ie0) / constants.g)

    def _extract_snapshot(fields):
        u = np.asarray(fields["u"])
        v = np.asarray(fields["v"])
        return {
            "wind_sfc": np.sqrt(u[..., -1] * u[..., -1] + v[..., -1] * v[..., -1]),
            "p_s": np.asarray(fields["p_s"]),
            "T_sfc": np.asarray(fields["T"])[..., -1],
        }

    def _extract_profile(fields):
        u = np.asarray(fields["u"])
        v = np.asarray(fields["v"])
        T = np.asarray(fields["T"])
        wind = np.sqrt(u * u + v * v)
        return {
            "wind_profile": atm25._horizontal_profile_3d(wind, area),
            "T_profile": atm25._horizontal_profile_3d(T, area),
        }

    def _extract_cross(fields):
        u = np.asarray(fields["u"])
        T = np.asarray(fields["T"])
        lon2d = np.broadcast_to(lon_deg[None, :], (u.shape[0], u.shape[1]))
        lon_cent, u_m = _lonbin_mean_3d(u, lon2d, area, n_lon_bins=u.shape[1])
        _, T_m = _lonbin_mean_3d(T, lon2d, area, n_lon_bins=u.shape[1])
        return {
            "lat_deg": lat_deg,
            "u_zonal": np.mean(u, axis=1),
            "T_zonal": np.mean(T, axis=1),
            "lon_deg": lon_cent,
            "u_meridional": u_m,
            "T_meridional": T_m,
        }

    def _record(step_i: int, fields):
        u = np.asarray(fields["u"])
        v = np.asarray(fields["v"])
        T = np.asarray(fields["T"])
        p_s = np.asarray(fields["p_s"])
        wind_prof = atm25._horizontal_profile_3d(np.sqrt(u * u + v * v), area)
        T_prof = atm25._horizontal_profile_3d(T, area)
        atm25._series_push(
            series,
            step_i,
            {
                "mean_wind_3d": atm25._vertical_integral_from_profile(wind_prof, dsigma),
                "mean_T_3d": atm25._vertical_integral_from_profile(T_prof, dsigma),
                "mean_p_s": atm25._weighted_mean_2d(p_s, area),
            },
        )
        mass = float(np.sum(p_s * area))
        ke = np.sum(np.sum(0.5 * (u * u + v * v) * p_s[..., None] * dsigma[None, None, :], axis=-1) * area)
        ie = np.sum(np.sum(constants.c_vd * T * p_s[..., None] * dsigma[None, None, :], axis=-1) * area)
        energy = float((ke + ie) / constants.g)
        cons_rows.append(
            {
                "step": float(step_i),
                "time_days": float(step_i * dt / 86400.0),
                "mass": mass,
                "energy": energy,
                "mass_rel": (mass - mass0) / max(abs(mass0), 1.0e-30),
                "energy_rel": (energy - energy0) / max(abs(energy0), 1.0e-30),
            },
        )

    snapshots[0] = _extract_snapshot(f0)
    profiles[0] = _extract_profile(f0)
    cross_sections[0] = _extract_cross(f0)
    _record(0, f0)

    stable = True
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)
        st = i + 1
        fields = _to_grid(state)
        if st in snap_targets:
            snapshots[st] = _extract_snapshot(fields)
            profiles[st] = _extract_profile(fields)
            cross_sections[st] = _extract_cross(fields)
        if st % max(1, mean_every) == 0 or st == n_steps:
            _record(st, fields)
        if st % progress_every == 0:
            print(f"      hs_spectral progress: {st}/{n_steps}")
        if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
            stable = False
            break
    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    case_name = f"Held-Suarez Spectral ({scheme}, {preset.name})"
    atm25._save_snapshots(
        case_dir,
        case_name,
        snapshots,
        dt,
        [
            ("wind_sfc", "Surface wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
            ("T_sfc", "Surface temperature (K)", "coolwarm"),
        ],
        coord_kind="gaussian",
        lon_deg=lon_deg,
        lat_deg=lat_deg,
    )
    atm25._save_timeseries(
        case_dir,
        case_name,
        series,
        dt,
        {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
    )
    atm25._save_profiles(
        case_dir,
        case_name,
        profiles,
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
        units={"wind_profile": "m/s", "T_profile": "K"},
    )
    mass_end, energy_end = _save_conservation(case_dir, case_name, cons_rows)
    _save_zonal_cross_sections(
        case_dir,
        case_name,
        {k: {"lat_deg": v["lat_deg"], "u_zonal": v["u_zonal"], "T_zonal": v["T_zonal"]} for k, v in cross_sections.items()},
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
    )
    _save_meridional_cross_sections(
        case_dir,
        case_name,
        {k: {"lon_deg": v["lon_deg"], "u_meridional": v["u_meridional"], "T_meridional": v["T_meridional"]} for k, v in cross_sections.items()},
        dt,
        sigma_full,
        "Sigma",
        invert_y=True,
    )

    _write_results_txt(
        case_dir / "results.txt",
        {
            "case": "hs_spectral",
            "scheme": scheme,
            "preset": preset.name,
            "truncation": f"T{trunc}",
            "levels": nlev,
            "dt": dt,
            "days": days,
            "n_steps": n_steps,
            "stable": stable,
            "mass_drift_rel": f"{mass_end:.12e}",
            "energy_drift_rel": f"{energy_end:.12e}",
            "wall_time_s": f"{wall:.2f}",
        },
    )
    return {
        "case": "hs_spectral",
        "scheme": scheme,
        "preset": preset.name,
        "status": "PASS" if stable else "FAIL",
        "stable": stable,
        "mass_drift_rel": mass_end,
        "energy_drift_rel": energy_end,
        "wall_time_s": wall,
        "output": str(case_dir),
    }


def _run_nh_fv_case(
    case_name: str,
    case_dir: Path,
    scheme: str,
    dt: float,
    duration_s: float,
    state0,
    grid,
    height_coord,
    terrain_metric,
    config_kwargs: dict[str, object],
    mean_every: int,
) -> dict[str, object]:
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig, CompressibleEulerModel

    n_steps = int(duration_s / dt)
    case_dir.mkdir(parents=True, exist_ok=True)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    z_full = np.asarray(height_coord.z_full)
    dz = np.asarray(height_coord.dz)

    config = CompressibleEulerConfig(**config_kwargs)
    model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)
    physics_fn = _build_radiation_only_physics(scheme, "nonhydrostatic", dt)
    state = state0

    snap_targets = atm25._snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {}
    cross_sections: dict[int, dict[str, np.ndarray]] = {}
    series = atm25._series_init(["mean_wind_3d", "mean_abs_w_3d", "mean_theta_prime_3d", "mean_rho_prime_3d"])
    cons_rows: list[dict[str, float]] = []

    mass0 = float(compute_nh_dry_mass(state.rho_prime.data, height_coord, terrain_metric, grid))
    energy0 = float(compute_nh_energy(state, grid, height_coord, terrain_metric)["total_energy"])

    def _extract_snapshot(s):
        u_low = np.asarray(s.u.data)[..., -1]
        v_low = np.asarray(s.v.data)[..., -1]
        w_half = np.asarray(s.w.data)
        out = {
            "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
            "rho_prime_low": np.asarray(s.rho_prime.data)[..., -1],
            "w_mid": w_half[..., w_half.shape[-1] // 2],
        }
        tr = np.asarray(s.tracers.data)
        if tr.ndim == 5 and tr.shape[-1] >= 3:
            out["q_rain_low"] = np.clip(tr[..., -1, 2], 0.0, None)
        return out

    def _extract_profile(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        w_half = np.asarray(s.w.data)
        w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
        wind = np.sqrt(u * u + v * v)
        return {
            "wind_profile": atm25._horizontal_profile_3d(wind, area),
            "abs_w_profile": atm25._horizontal_profile_3d(np.abs(w_full), area),
            "theta_prime_profile": atm25._horizontal_profile_3d(np.asarray(s.theta_prime.data), area),
            "rho_prime_profile": atm25._horizontal_profile_3d(np.asarray(s.rho_prime.data), area),
        }

    def _extract_cross(s):
        u = np.asarray(s.u.data)
        theta_p = np.asarray(s.theta_prime.data)
        lat_cent, uz = _latbin_mean_3d(u, lat_deg, area)
        lon_cent, um = _lonbin_mean_3d(u, lon_deg, area)
        _, thz = _latbin_mean_3d(theta_p, lat_deg, area)
        _, thm = _lonbin_mean_3d(theta_p, lon_deg, area)
        return {
            "lat_deg": lat_cent,
            "u_zonal": uz,
            "theta_prime_zonal": thz,
            "lon_deg": lon_cent,
            "u_meridional": um,
            "theta_prime_meridional": thm,
        }

    def _record(step_i: int, s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        w_half = np.asarray(s.w.data)
        w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
        wind_prof = atm25._horizontal_profile_3d(np.sqrt(u * u + v * v), area)
        abs_w_prof = atm25._horizontal_profile_3d(np.abs(w_full), area)
        th_prof = atm25._horizontal_profile_3d(np.asarray(s.theta_prime.data), area)
        rho_prof = atm25._horizontal_profile_3d(np.asarray(s.rho_prime.data), area)
        atm25._series_push(
            series,
            step_i,
            {
                "mean_wind_3d": atm25._vertical_integral_from_profile(wind_prof, dz),
                "mean_abs_w_3d": atm25._vertical_integral_from_profile(abs_w_prof, dz),
                "mean_theta_prime_3d": atm25._vertical_integral_from_profile(th_prof, dz),
                "mean_rho_prime_3d": atm25._vertical_integral_from_profile(rho_prof, dz),
            },
        )
        mass = float(compute_nh_dry_mass(s.rho_prime.data, height_coord, terrain_metric, grid))
        energy = float(compute_nh_energy(s, grid, height_coord, terrain_metric)["total_energy"])
        cons_rows.append(
            {
                "step": float(step_i),
                "time_days": float(step_i * dt / 86400.0),
                "mass": mass,
                "energy": energy,
                "mass_rel": (mass - mass0) / max(abs(mass0), 1.0e-30),
                "energy_rel": (energy - energy0) / max(abs(energy0), 1.0e-30),
            },
        )

    snapshots[0] = _extract_snapshot(state)
    profiles[0] = _extract_profile(state)
    cross_sections[0] = _extract_cross(state)
    _record(0, state)

    stable = True
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)
        st = i + 1
        if st in snap_targets:
            snapshots[st] = _extract_snapshot(state)
            profiles[st] = _extract_profile(state)
            cross_sections[st] = _extract_cross(state)
        if st % max(1, mean_every) == 0 or st == n_steps:
            _record(st, state)
        if st % progress_every == 0:
            print(f"      {case_name} progress: {st}/{n_steps}")
        u_max = float(jnp.max(jnp.abs(state.u.data)))
        if (not bool(jnp.all(jnp.isfinite(state.u.data)))) or (u_max > 1500.0):
            stable = False
            break
    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    field_specs = [
        ("wind_low", "Low-level wind speed (m/s)", "magma"),
        ("rho_prime_low", "Low-level rho' (kg/m3)", "RdBu_r"),
        ("w_mid", "Mid-level w (m/s)", "RdBu_r"),
    ]
    if any("q_rain_low" in s for s in snapshots.values()):
        field_specs.append(("q_rain_low", "Low-level q_rain", "Blues"))

    pretty_case = f"{case_name} ({scheme})"
    atm25._save_snapshots(case_dir, pretty_case, snapshots, dt, field_specs, coord_kind="cube", lon_deg=lon_deg, lat_deg=lat_deg)
    atm25._save_timeseries(
        case_dir,
        pretty_case,
        series,
        dt,
        {
            "mean_wind_3d": "m/s",
            "mean_abs_w_3d": "m/s",
            "mean_theta_prime_3d": "K",
            "mean_rho_prime_3d": "kg/m3",
        },
    )
    atm25._save_profiles(
        case_dir,
        pretty_case,
        profiles,
        dt,
        z_full,
        "Height z (m)",
        invert_y=True,
        units={
            "wind_profile": "m/s",
            "abs_w_profile": "m/s",
            "theta_prime_profile": "K",
            "rho_prime_profile": "kg/m3",
        },
    )
    mass_end, energy_end = _save_conservation(case_dir, pretty_case, cons_rows)
    _save_zonal_cross_sections(
        case_dir,
        pretty_case,
        {
            k: {
                "lat_deg": v["lat_deg"],
                "u_zonal": v["u_zonal"],
                "theta_prime_zonal": v["theta_prime_zonal"],
            }
            for k, v in cross_sections.items()
        },
        dt,
        z_full,
        "Height z (m)",
        invert_y=False,
    )
    _save_meridional_cross_sections(
        case_dir,
        pretty_case,
        {
            k: {
                "lon_deg": v["lon_deg"],
                "u_meridional": v["u_meridional"],
                "theta_prime_meridional": v["theta_prime_meridional"],
            }
            for k, v in cross_sections.items()
        },
        dt,
        z_full,
        "Height z (m)",
        invert_y=False,
    )

    _write_results_txt(
        case_dir / "results.txt",
        {
            "case": case_name,
            "scheme": scheme,
            "dt": dt,
            "duration_s": duration_s,
            "n_steps": n_steps,
            "stable": stable,
            "mass_drift_rel": f"{mass_end:.12e}",
            "energy_drift_rel": f"{energy_end:.12e}",
            "wall_time_s": f"{wall:.2f}",
        },
    )

    return {
        "case": case_name,
        "scheme": scheme,
        "status": "PASS" if stable else "FAIL",
        "stable": stable,
        "mass_drift_rel": mass_end,
        "energy_drift_rel": energy_end,
        "wall_time_s": wall,
        "output": str(case_dir),
    }


def _make_nh_latlon_rest_state(grid, height_coord, n_tracers: int = 4) -> NonHydrostaticState:
    nlev = int(height_coord.n_levels)
    shape_3d = (int(grid.n_lat), int(grid.n_lon), nlev)
    shape_w = (int(grid.n_lat), int(grid.n_lon), nlev + 1)
    shape_2d = (int(grid.n_lat), int(grid.n_lon))
    shape_tr = (*shape_3d, int(max(0, n_tracers)))

    dims_3d = ("lat", "lon", "level")
    dims_w = ("lat", "lon", "level_half")
    dims_2d = ("lat", "lon")
    dims_tr = ("lat", "lon", "level", "tracer")

    return NonHydrostaticState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros(shape_w), name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros(shape_tr), name="tracers", dims=dims_tr, units="kg/kg"),
    )


def run_nh_fv_latlon_rest(case_dir: Path, preset: ResolutionPreset, scheme: str, tc1_hours: float, mean_every: int, solver: str):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric
    from legoesm.atmosphere.dynamics.compressible_euler_fv_latlon import (
        FVCompressibleEulerLatLonConfig,
        FVCompressibleEulerLatLonModel,
    )

    dt = 0.5
    duration_s = max(tc1_hours * 3600.0, dt)
    n_steps = int(duration_s / dt)
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_latlon_grid(preset.latlon_nlat, preset.latlon_nlon)
    height_coord = create_height_coordinate(preset.nh_levels, 30000.0)
    z_s = jnp.zeros((grid.n_lat, grid.n_lon), dtype=jnp.float64)
    terrain_metric = compute_terrain_metric(z_s, height_coord)

    state0 = _make_nh_latlon_rest_state(grid, height_coord, n_tracers=4)
    state = state0

    area = np.asarray(grid.area, dtype=np.float64)
    lon_deg_1d = np.asarray(grid.lon, dtype=np.float64) * 180.0 / np.pi
    lat_deg_1d = np.asarray(grid.lat, dtype=np.float64) * 180.0 / np.pi
    lon_deg_2d, lat_deg_2d = np.meshgrid(lon_deg_1d, lat_deg_1d)
    z_full = np.asarray(height_coord.z_full, dtype=np.float64)
    dz = np.asarray(height_coord.dz, dtype=np.float64)

    scale = (90.0 / float(max(grid.n_lat, 1))) ** 4
    config = FVCompressibleEulerLatLonConfig(
        hyperdiff_coeff=2.0e15 * scale,
        hyperdiff_rho_coeff=5.0e14 * scale,
        hyperdiff_w_coeff=1.0e14 * scale,
        sponge_width=12000.0,
        sponge_coeff=0.03,
        n_acoustic_substeps=8,
        outer_integrator="ssp_rk3",
        use_polar_filter=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
    )
    model = FVCompressibleEulerLatLonModel(grid, height_coord, terrain_metric, config, dt=dt)
    # Nonhydro radiation integration is currently cube-oriented in this workspace.
    # Keep this standard case as dry dynamics until a lat-lon NH radiation bridge is added.
    physics_fn = None

    snap_targets = atm25._snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {}
    cross_sections: dict[int, dict[str, np.ndarray]] = {}
    series = atm25._series_init(["mean_wind_3d", "mean_abs_w_3d", "mean_theta_prime_3d", "mean_rho_prime_3d"])
    cons_rows: list[dict[str, float]] = []

    mass0 = float(compute_nh_dry_mass(state.rho_prime.data, height_coord, terrain_metric, grid))
    energy0 = float(compute_nh_energy(state, grid, height_coord, terrain_metric)["total_energy"])

    def _extract_snapshot(s):
        u_low = np.asarray(s.u.data)[..., -1]
        v_low = np.asarray(s.v.data)[..., -1]
        w_half = np.asarray(s.w.data)
        out = {
            "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
            "rho_prime_low": np.asarray(s.rho_prime.data)[..., -1],
            "w_mid": w_half[..., w_half.shape[-1] // 2],
        }
        tr = np.asarray(s.tracers.data)
        if tr.ndim == 4 and tr.shape[-1] >= 1:
            out["qv_low"] = np.clip(tr[..., -1, 0], 0.0, None)
        return out

    def _extract_profile(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        w_half = np.asarray(s.w.data)
        w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
        wind = np.sqrt(u * u + v * v)
        return {
            "wind_profile": atm25._horizontal_profile_3d(wind, area),
            "abs_w_profile": atm25._horizontal_profile_3d(np.abs(w_full), area),
            "theta_prime_profile": atm25._horizontal_profile_3d(np.asarray(s.theta_prime.data), area),
            "rho_prime_profile": atm25._horizontal_profile_3d(np.asarray(s.rho_prime.data), area),
        }

    def _extract_cross(s):
        u = np.asarray(s.u.data)
        theta_p = np.asarray(s.theta_prime.data)
        lat_cent, uz = _latbin_mean_3d(u, lat_deg_2d, area)
        lon_cent, um = _lonbin_mean_3d(u, lon_deg_2d, area)
        _, thz = _latbin_mean_3d(theta_p, lat_deg_2d, area)
        _, thm = _lonbin_mean_3d(theta_p, lon_deg_2d, area)
        return {
            "lat_deg": lat_cent,
            "u_zonal": uz,
            "theta_prime_zonal": thz,
            "lon_deg": lon_cent,
            "u_meridional": um,
            "theta_prime_meridional": thm,
        }

    def _record(step_i: int, s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        w_half = np.asarray(s.w.data)
        w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
        wind_prof = atm25._horizontal_profile_3d(np.sqrt(u * u + v * v), area)
        abs_w_prof = atm25._horizontal_profile_3d(np.abs(w_full), area)
        th_prof = atm25._horizontal_profile_3d(np.asarray(s.theta_prime.data), area)
        rho_prof = atm25._horizontal_profile_3d(np.asarray(s.rho_prime.data), area)
        atm25._series_push(
            series,
            step_i,
            {
                "mean_wind_3d": atm25._vertical_integral_from_profile(wind_prof, dz),
                "mean_abs_w_3d": atm25._vertical_integral_from_profile(abs_w_prof, dz),
                "mean_theta_prime_3d": atm25._vertical_integral_from_profile(th_prof, dz),
                "mean_rho_prime_3d": atm25._vertical_integral_from_profile(rho_prof, dz),
            },
        )
        mass = float(compute_nh_dry_mass(s.rho_prime.data, height_coord, terrain_metric, grid))
        energy = float(compute_nh_energy(s, grid, height_coord, terrain_metric)["total_energy"])
        cons_rows.append(
            {
                "step": float(step_i),
                "time_days": float(step_i * dt / 86400.0),
                "mass": mass,
                "energy": energy,
                "mass_rel": (mass - mass0) / max(abs(mass0), 1.0e-30),
                "energy_rel": (energy - energy0) / max(abs(energy0), 1.0e-30),
            },
        )

    snapshots[0] = _extract_snapshot(state)
    profiles[0] = _extract_profile(state)
    cross_sections[0] = _extract_cross(state)
    _record(0, state)

    stable = True
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)
        st = i + 1
        if st in snap_targets:
            snapshots[st] = _extract_snapshot(state)
            profiles[st] = _extract_profile(state)
            cross_sections[st] = _extract_cross(state)
        if st % max(1, mean_every) == 0 or st == n_steps:
            _record(st, state)
        if st % progress_every == 0:
            print(f"      nh_fv_latlon_rest progress: {st}/{n_steps}")
        u_max = float(jnp.max(jnp.abs(state.u.data)))
        if (not bool(jnp.all(jnp.isfinite(state.u.data)))) or (u_max > 1500.0):
            stable = False
            break
    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    case_name = f"nh_fv_latlon_rest_{preset.name}"
    pretty_case = f"NH FV LatLon Rest ({scheme}, {preset.name})"
    field_specs = [
        ("wind_low", "Low-level wind speed (m/s)", "magma"),
        ("rho_prime_low", "Low-level rho' (kg/m3)", "RdBu_r"),
        ("w_mid", "Mid-level w (m/s)", "RdBu_r"),
        ("qv_low", "Low-level qv (kg/kg)", "Blues"),
    ]
    atm25._save_snapshots(
        case_dir,
        pretty_case,
        snapshots,
        dt,
        field_specs,
        coord_kind="latlon",
        lon_deg=lon_deg_1d,
        lat_deg=lat_deg_1d,
    )
    atm25._save_timeseries(
        case_dir,
        pretty_case,
        series,
        dt,
        {
            "mean_wind_3d": "m/s",
            "mean_abs_w_3d": "m/s",
            "mean_theta_prime_3d": "K",
            "mean_rho_prime_3d": "kg/m3",
        },
    )
    atm25._save_profiles(
        case_dir,
        pretty_case,
        profiles,
        dt,
        z_full,
        "Height z (m)",
        invert_y=False,
        units={
            "wind_profile": "m/s",
            "abs_w_profile": "m/s",
            "theta_prime_profile": "K",
            "rho_prime_profile": "kg/m3",
        },
    )
    mass_end, energy_end = _save_conservation(case_dir, pretty_case, cons_rows)
    _save_zonal_cross_sections(
        case_dir,
        pretty_case,
        {
            k: {
                "lat_deg": v["lat_deg"],
                "u_zonal": v["u_zonal"],
                "theta_prime_zonal": v["theta_prime_zonal"],
            }
            for k, v in cross_sections.items()
        },
        dt,
        z_full,
        "Height z (m)",
        invert_y=False,
    )
    _save_meridional_cross_sections(
        case_dir,
        pretty_case,
        {
            k: {
                "lon_deg": v["lon_deg"],
                "u_meridional": v["u_meridional"],
                "theta_prime_meridional": v["theta_prime_meridional"],
            }
            for k, v in cross_sections.items()
        },
        dt,
        z_full,
        "Height z (m)",
        invert_y=False,
    )
    _write_results_txt(
        case_dir / "results.txt",
        {
            "case": case_name,
            "scheme": scheme,
            "preset": preset.name,
            "dt": dt,
            "duration_s": duration_s,
            "n_steps": n_steps,
            "stable": stable,
            "mass_drift_rel": f"{mass_end:.12e}",
            "energy_drift_rel": f"{energy_end:.12e}",
            "notes": "dry_dynamics_only; nh_latlon_radiation_bridge_pending",
            "wall_time_s": f"{wall:.2f}",
        },
    )

    return {
        "case": "nh_fv_latlon_rest",
        "scheme": scheme,
        "preset": preset.name,
        "status": "PASS" if stable else "FAIL",
        "stable": stable,
        "mass_drift_rel": mass_end,
        "energy_drift_rel": energy_end,
        "notes": "dry_dynamics_only; nh_latlon_radiation_bridge_pending",
        "wall_time_s": wall,
        "output": str(case_dir),
    }


def run_dcmip_fv_tc1(case_dir: Path, preset: ResolutionPreset, scheme: str, tc1_hours: float, mean_every: int, solver: str):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init

    grid = create_cubed_sphere(preset.cube_resolution)
    state0, height_coord, terrain_metric = dcmip25_tc1_init(grid, n_levels=preset.nh_levels)
    return _run_nh_fv_case(
        case_name=f"dcmip_tc1_fv_cube_{preset.name}",
        case_dir=case_dir,
        scheme=scheme,
        dt=2.0,
        duration_s=tc1_hours * 3600.0,
        state0=state0,
        grid=grid,
        height_coord=height_coord,
        terrain_metric=terrain_metric,
        config_kwargs={
            "n_acoustic_substeps": 8,
            "sponge_width": 10000.0,
            "sponge_coeff": 0.06,
            "outer_integrator": solver,
            "edge_blend_uv": 0.22,
            "edge_blend_w": 0.14,
            "edge_blend_theta": 0.12,
            "edge_blend_rho": 0.18,
            "edge_blend_width": 3,
        },
        mean_every=mean_every,
    )


def run_dcmip_fv_tc2a(case_dir: Path, preset: ResolutionPreset, scheme: str, tc23_minutes: float, mean_every: int, solver: str):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from tests.test_cases.dcmip2025 import dcmip25_tc2_init

    grid = create_cubed_sphere(preset.cube_resolution)
    state0, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(
        grid, n_levels=preset.nh_levels, subcase="a",
    )
    return _run_nh_fv_case(
        case_name=f"dcmip_tc2a_fv_cube_{preset.name}",
        case_dir=case_dir,
        scheme=scheme,
        dt=1.0,
        duration_s=tc23_minutes * 60.0,
        state0=state0,
        grid=small_grid,
        height_coord=height_coord,
        terrain_metric=terrain_metric,
        config_kwargs={
            "n_acoustic_substeps": 6,
            "sponge_width": 15000.0,
            "sponge_coeff": 1.0 / (0.1 * 86400.0),
            "small_earth_factor": 20.0,
            "outer_integrator": solver,
            "edge_blend_uv": 0.22,
            "edge_blend_w": 0.22,
            "edge_blend_theta": 0.12,
            "edge_blend_rho": 0.26,
            "edge_blend_width": 4,
        },
        mean_every=mean_every,
    )


def run_dcmip_fv_tc3(case_dir: Path, preset: ResolutionPreset, scheme: str, tc23_minutes: float, mean_every: int, solver: str):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from tests.test_cases.dcmip2025 import dcmip25_tc3_init

    grid = create_cubed_sphere(preset.cube_resolution)
    state0, height_coord, terrain_metric, small_grid = dcmip25_tc3_init(grid, n_levels=preset.nh_levels)
    return _run_nh_fv_case(
        case_name=f"dcmip_tc3_fv_cube_{preset.name}",
        case_dir=case_dir,
        scheme=scheme,
        dt=1.0,
        duration_s=tc23_minutes * 60.0,
        state0=state0,
        grid=small_grid,
        height_coord=height_coord,
        terrain_metric=terrain_metric,
        config_kwargs={
            "n_acoustic_substeps": 6,
            "sponge_width": 5000.0,
            "sponge_coeff": 0.05,
            "small_earth_factor": 60.0,
            "use_coriolis": False,
            "outer_integrator": solver,
        },
        mean_every=mean_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-Suarez/DCMIP radiation matrix")
    parser.add_argument("--output", type=Path, default=Path("results/atmosphere/hydrostatic/hs_dcmip_radiation_matrix"))
    parser.add_argument("--schemes", type=str, default="gray,rrtmgp", help="Comma-separated: gray,rrtmgp")
    parser.add_argument("--presets", type=str, default="atm2deg", help="Comma-separated: low,medium,high,atm2deg")
    parser.add_argument("--hs-days", type=float, default=2.0)
    parser.add_argument("--tc1-hours", type=float, default=1.0)
    parser.add_argument("--tc23-minutes", type=float, default=3.0)
    parser.add_argument("--mean-every", type=int, default=12)
    parser.add_argument("--solver", type=str, default="ssp45")
    args = parser.parse_args()

    schemes = [s.strip().lower() for s in args.schemes.split(",") if s.strip()]
    preset_names = [p.strip().lower() for p in args.presets.split(",") if p.strip()]
    presets = [PRESETS[p] for p in preset_names if p in PRESETS]
    if not presets:
        raise ValueError(f"No valid presets selected from {preset_names}; choose from {sorted(PRESETS)}")
    for s in schemes:
        if s not in ("gray", "rrtmgp"):
            raise ValueError(f"Unsupported radiation scheme {s!r}; choose gray,rrtmgp")

    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    print("Running Held-Suarez / DCMIP radiation matrix")
    print(f"  output={out_root}")
    print(f"  schemes={schemes}")
    print(f"  presets={[p.name for p in presets]}")
    print(f"  hs_days={args.hs_days}, tc1_hours={args.tc1_hours}, tc23_minutes={args.tc23_minutes}")

    records: list[dict[str, object]] = []
    t0 = time.time()

    for scheme in schemes:
        for preset in presets:
            block_root = out_root / scheme / preset.name
            block_root.mkdir(parents=True, exist_ok=True)
            print(f"\n=== scheme={scheme}, preset={preset.name} ===")

            runs = [
                ("hs_fv_cube", lambda: run_hs_fv_cube(block_root / "01_hs_fv_cube", preset, scheme, args.hs_days, args.mean_every, args.solver)),
                ("hs_fv_latlon", lambda: run_hs_fv_latlon(block_root / "02_hs_fv_latlon", preset, scheme, args.hs_days, args.mean_every, args.solver)),
                ("hs_spectral", lambda: run_hs_spectral(block_root / "03_hs_spectral", preset, scheme, args.hs_days, args.mean_every)),
                ("dcmip_tc1_fv_cube", lambda: run_dcmip_fv_tc1(block_root / "04_dcmip_tc1_fv_cube", preset, scheme, args.tc1_hours, args.mean_every, args.solver)),
                ("dcmip_tc2a_fv_cube", lambda: run_dcmip_fv_tc2a(block_root / "05_dcmip_tc2a_fv_cube", preset, scheme, args.tc23_minutes, args.mean_every, args.solver)),
                ("dcmip_tc3_fv_cube", lambda: run_dcmip_fv_tc3(block_root / "06_dcmip_tc3_fv_cube", preset, scheme, args.tc23_minutes, args.mean_every, args.solver)),
                ("nh_fv_latlon_rest", lambda: run_nh_fv_latlon_rest(block_root / "07_nh_fv_latlon_rest", preset, scheme, args.tc1_hours, args.mean_every, args.solver)),
            ]

            for name, fn in runs:
                print(f"  -> {name}")
                try:
                    rec = fn()
                    rec["preset"] = preset.name
                    rec["scheme"] = scheme
                    records.append(rec)
                except Exception as exc:
                    traceback.print_exc()
                    records.append(
                        {
                            "case": name,
                            "preset": preset.name,
                            "scheme": scheme,
                            "status": "ERROR",
                            "stable": False,
                            "mass_drift_rel": float("nan"),
                            "energy_drift_rel": float("nan"),
                            "wall_time_s": 0.0,
                            "output": str(block_root / name),
                            "notes": str(exc),
                        },
                    )

            # Explicitly mark currently unsupported requested combinations.
            records.append(
                {
                    "case": "dcmip_spectral_radiation",
                    "preset": preset.name,
                    "scheme": scheme,
                    "status": "SKIPPED",
                    "stable": False,
                    "mass_drift_rel": float("nan"),
                    "energy_drift_rel": float("nan"),
                    "wall_time_s": 0.0,
                    "output": "",
                    "notes": "Not available: spectral NH model lacks step_with_physics radiation coupling path.",
                },
            )
            records.append(
                {
                    "case": "dcmip_latlon_radiation",
                    "preset": preset.name,
                    "scheme": scheme,
                    "status": "SKIPPED",
                    "stable": False,
                    "mass_drift_rel": float("nan"),
                    "energy_drift_rel": float("nan"),
                    "wall_time_s": 0.0,
                    "output": "",
                    "notes": "Not available: no standard DCMIP lat-lon initialization/benchmark path in workspace scripts.",
                },
            )

    total_wall = time.time() - t0
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "schemes": schemes,
            "presets": [asdict(p) for p in presets],
            "hs_days": args.hs_days,
            "tc1_hours": args.tc1_hours,
            "tc23_minutes": args.tc23_minutes,
            "mean_every": args.mean_every,
            "solver": args.solver,
            "total_wall_time_s": total_wall,
        },
        "results": records,
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    n_pass = sum(1 for r in records if r.get("status") == "PASS")
    n_fail = sum(1 for r in records if r.get("status") == "FAIL")
    n_err = sum(1 for r in records if r.get("status") == "ERROR")
    n_skip = sum(1 for r in records if r.get("status") == "SKIPPED")
    print("\nDone.")
    print(f"  PASS={n_pass} FAIL={n_fail} ERROR={n_err} SKIPPED={n_skip} TOTAL={len(records)}")
    print(f"  Wall time: {total_wall:.1f}s")
    print(f"  Summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
