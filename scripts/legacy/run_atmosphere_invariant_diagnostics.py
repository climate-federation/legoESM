#!/usr/bin/env python
"""Run atmosphere invariant/regression cases with diagnostics and plots.

Cases:
1. Rest-state invariance (hydrostatic PE, cubed-sphere)
2. Solid-body rotation (shallow water Williamson 2, cubed-sphere)
3. Single-column radiation (gray)
4. Single-column radiation (RRTMGP)

Outputs per case:
- field_snapshots.png
- diagnostics_timeseries.png
- conservation_timeseries.png
- diagnostics.csv
- summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if os.environ.get("LEGOESM_DISABLE_CARTOPY", "0") == "1":
    ccrs = None
    HAS_CARTOPY = False
else:
    try:
        import cartopy.crs as ccrs

        HAS_CARTOPY = True
    except ImportError:
        ccrs = None
        HAS_CARTOPY = False

from legoesm import constants
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig as PrimitiveEquationConfig,
    CDGridPrimitiveEquationModel as PrimitiveEquationModel,
    cdgrid_hydrostatic_tendencies as hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterConfig as ShallowWaterConfig, CDGridShallowWaterModel as ShallowWaterModel
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig, RRTMGPConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
from legoesm.core.conservation import compute_conservation_diagnostics
from legoesm.core.field import Field
from legoesm.core.operators import curl_z, global_integral
from legoesm.core.state import HydrostaticState, ShallowWaterState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate


def _snapshot_steps(n_steps: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    return sorted({0, max(1, n_steps // 2), n_steps})


def _format_sim_time(step: int, dt: float) -> str:
    t_sec = float(step) * float(dt)
    if t_sec < 3600.0:
        return f"{t_sec / 60.0:.1f} min"
    if t_sec < 86400.0:
        return f"{t_sec / 3600.0:.2f} h"
    return f"{t_sec / 86400.0:.2f} d"


def _color_limits(fields: list[np.ndarray]) -> tuple[float, float]:
    chunks = [f[np.isfinite(f)] for f in fields if f is not None]
    chunks = [c for c in chunks if c.size > 0]
    if not chunks:
        return -1.0, 1.0
    vals = np.concatenate(chunks)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmin < 0.0 < vmax:
        vmax_abs = max(abs(vmin), abs(vmax), 1e-12)
        vmin, vmax = -vmax_abs, vmax_abs
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1.0e-6
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _plot_cube_snapshots(
    out_path: Path,
    case_name: str,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    field_specs: list[tuple[str, str, str]],
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> None:
    steps = sorted(snapshots.keys())
    n_rows = len(field_specs)
    n_cols = len(steps)

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(5.0 * n_cols + 1.2, 3.3 * n_rows))
        gs = fig.add_gridspec(
            n_rows,
            n_cols + 1,
            width_ratios=[1.0] * n_cols + [0.06],
            hspace=0.3,
            wspace=0.2,
        )
        proj = ccrs.Robinson()
        axes = [
            [fig.add_subplot(gs[r, c], projection=proj) for c in range(n_cols)]
            for r in range(n_rows)
        ]
        caxes = [fig.add_subplot(gs[r, n_cols]) for r in range(n_rows)]
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.1 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = np.array([[axes[r]] for r in range(n_rows)])
        caxes = [None] * n_rows

    lon_flat = np.asarray(lon_deg, dtype=np.float64).reshape(-1)
    lat_flat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)

    for row, (key, label, cmap) in enumerate(field_specs):
        row_vals = [np.asarray(snapshots[s][key], dtype=np.float64) for s in steps]
        vmin, vmax = _color_limits(row_vals)
        im = None

        for col, step in enumerate(steps):
            ax = axes[row][col]
            vals = row_vals[col].reshape(-1)
            valid = np.isfinite(vals) & np.isfinite(lon_flat) & np.isfinite(lat_flat)
            marker_size = max(0.8, 2200.0 / float(vals.size))

            if HAS_CARTOPY:
                im = ax.scatter(
                    lon_flat[valid],
                    lat_flat[valid],
                    c=vals[valid],
                    s=marker_size,
                    linewidths=0.0,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    rasterized=True,
                )
                ax.set_global()
                ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.35)
            else:
                n_face = int(np.round((vals.size / 6.0) ** 0.5))
                mosaic = vals.reshape(6, n_face, n_face)
                panel = np.full((2 * n_face, 3 * n_face), np.nan, dtype=np.float64)
                for i in range(6):
                    rr = 0 if i < 3 else 1
                    cc = i % 3
                    panel[rr * n_face:(rr + 1) * n_face, cc * n_face:(cc + 1) * n_face] = mosaic[i]
                im = ax.imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])

            ax.set_title(f"{label}\nstep {step}, t={_format_sim_time(step, dt)}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)

        if HAS_CARTOPY and im is not None:
            fig.colorbar(im, cax=caxes[row], orientation="vertical")
        elif (not HAS_CARTOPY) and im is not None and col == n_cols - 1:
            fig.colorbar(im, ax=[axes[row][c] for c in range(n_cols)], orientation="vertical", shrink=0.9)

    fig.suptitle(f"{case_name} - Field Snapshots", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_timeseries(
    out_path: Path,
    title: str,
    time_days: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
) -> None:
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.9 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, values, color) in zip(axes, series):
        ax.plot(time_days, values, color=color, lw=1.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [days]")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _make_stratified_rest_state(grid, sigma) -> HydrostaticState:
    shape_3d = (6, grid.n, grid.n, sigma.n_levels)
    shape_2d = (6, grid.n, grid.n)
    t_profile = jnp.linspace(210.0, 290.0, sigma.n_levels, dtype=jnp.float32)
    t_data = jnp.broadcast_to(t_profile[None, None, None, :], shape_3d)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=jnp.zeros(shape_3d, dtype=jnp.float32), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d, dtype=jnp.float32), name="v", dims=dims_3d, units="m/s"),
        T=Field(data=t_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.full(shape_2d, 1.0e5, dtype=jnp.float32), name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d, dtype=jnp.float32), name="phis", dims=dims_2d, units="m^2/s^2"),
    )


def _williamson_test2_state(grid) -> ShallowWaterState:
    """Williamson test 2 initial state on cubed-sphere."""
    r = grid.radius
    omega = constants.Omega
    g = constants.g

    u0 = 2.0 * jnp.pi * r / (12.0 * 86400.0)
    gh0 = 2.94e4
    h0 = gh0 / g

    lat = grid.lat
    u_east = u0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)

    cos_a = jnp.cos(grid.angle)
    sin_a = jnp.sin(grid.angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north

    h = h0 - (r * omega * u0 + 0.5 * u0 * u0) * jnp.sin(lat) ** 2 / g
    h_s = jnp.zeros_like(h)
    dims = ("face", "x", "y")

    return ShallowWaterState(
        h=Field(data=h, name="h", dims=dims, units="m"),
        u=Field(data=u_grid, name="u", dims=dims, units="m/s"),
        v=Field(data=v_grid, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s, name="h_s", dims=dims, units="m"),
    )


def _hydro_ke_proxy(state: HydrostaticState, grid, sigma) -> jnp.ndarray:
    u2v2 = state.u.data * state.u.data + state.v.data * state.v.data
    layer_mass = (
        state.p_s.data[..., None]
        * sigma.dsigma[None, None, None, :]
        / constants.g
    )  # kg m^-2
    ke = 0.5 * u2v2 * layer_mass  # J m^-2
    return jnp.sum(ke * grid.area[..., None])


def _run_rest_state_case(case_dir: Path) -> dict[str, float]:
    print("\n[1/4] Rest-state invariance with diagnostics")
    grid = create_cubed_sphere(8)
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(12)
    state0 = _make_stratified_rest_state(grid, sigma)
    state = state0

    cfg = PrimitiveEquationConfig(
        hyperdiff_coeff=0.0,
        hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False,
    )
    model = PrimitiveEquationModel(grid, sigma, cfg)

    dt = 600.0
    n_steps = 144  # 1 day
    snap_steps = _snapshot_steps(n_steps)
    sigma_idx = int(np.argmin(np.abs(np.asarray(sigma.sigma_full) - 0.85)))

    snapshots: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, float]] = []

    mass0 = float(global_integral(state0.p_s, grid))
    ke0 = float(_hydro_ke_proxy(state0, grid, sigma))

    t0 = time.time()
    for step in range(n_steps + 1):
        # Diagnose tendencies directly for strict rest-state invariance.
        tend = hydrostatic_tendencies(state, grid, sigma, cdgrid, cfg)
        wind_speed = jnp.sqrt(state.u.data * state.u.data + state.v.data * state.v.data)
        mass = float(global_integral(state.p_s, grid))
        ke = float(_hydro_ke_proxy(state, grid, sigma))
        diagnostics.append(
            {
                "step": float(step),
                "time_days": float(step * dt / 86400.0),
                "mean_T": float(jnp.mean(state.T.data)),
                "max_wind": float(jnp.max(wind_speed)),
                "mass": mass,
                "mass_rel_drift": (mass - mass0) / mass0,
                "ke_proxy": ke,
                "ke_proxy_drift": ke - ke0,
                "max_abs_du_dt": float(jnp.max(jnp.abs(tend.du_dt.data))),
                "max_abs_dv_dt": float(jnp.max(jnp.abs(tend.dv_dt.data))),
                "max_abs_dT_dt": float(jnp.max(jnp.abs(tend.dT_dt.data))),
                "max_abs_dps_dt": float(jnp.max(jnp.abs(tend.dp_s_dt.data))),
            }
        )

        if step in snap_steps:
            snapshots[step] = {
                "wind_speed": np.asarray(wind_speed[..., sigma_idx]),
                "temp_anomaly": np.asarray(state.T.data[..., sigma_idx] - state0.T.data[..., sigma_idx]),
                "ps_anomaly": np.asarray(state.p_s.data - state0.p_s.data),
            }

        if step < n_steps:
            state = model.step(state, dt)

    _plot_cube_snapshots(
        case_dir / "field_snapshots.png",
        "Rest State Invariance",
        snapshots,
        dt,
        [
            ("wind_speed", "Wind speed @sigma~0.85 (m/s)", "magma"),
            ("temp_anomaly", "Temperature anomaly (K)", "RdBu_r"),
            ("ps_anomaly", "Surface pressure anomaly (Pa)", "RdBu_r"),
        ],
        np.asarray(grid.lon) * 180.0 / np.pi,
        np.asarray(grid.lat) * 180.0 / np.pi,
    )

    time_days = np.array([d["time_days"] for d in diagnostics], dtype=np.float64)
    _plot_timeseries(
        case_dir / "diagnostics_timeseries.png",
        "Rest State - Mean Diagnostics",
        time_days,
        [
            ("Global mean T (K)", np.array([d["mean_T"] for d in diagnostics]), "tab:blue"),
            ("Max wind (m/s)", np.array([d["max_wind"] for d in diagnostics]), "tab:orange"),
            ("Max |du/dt| (m/s^2)", np.array([d["max_abs_du_dt"] for d in diagnostics]), "tab:red"),
        ],
    )
    _plot_timeseries(
        case_dir / "conservation_timeseries.png",
        "Rest State - Conservation",
        time_days,
        [
            ("Mass relative drift", np.array([d["mass_rel_drift"] for d in diagnostics]), "tab:green"),
            ("KE proxy drift (J)", np.array([d["ke_proxy_drift"] for d in diagnostics]), "tab:purple"),
            ("Max |dp_s/dt| (Pa/s)", np.array([d["max_abs_dps_dt"] for d in diagnostics]), "tab:brown"),
        ],
    )

    _write_csv(case_dir / "diagnostics.csv", diagnostics)

    wall = time.time() - t0
    summary = {
        "case": "rest_state_invariance",
        "dt_s": dt,
        "n_steps": n_steps,
        "wall_time_s": wall,
        "max_wind_final": diagnostics[-1]["max_wind"],
        "max_abs_du_dt": max(d["max_abs_du_dt"] for d in diagnostics),
        "max_abs_dv_dt": max(d["max_abs_dv_dt"] for d in diagnostics),
        "max_abs_dT_dt": max(d["max_abs_dT_dt"] for d in diagnostics),
        "max_abs_dps_dt": max(d["max_abs_dps_dt"] for d in diagnostics),
        "mass_rel_drift_final": diagnostics[-1]["mass_rel_drift"],
    }
    with (case_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(
        "  done | "
        f"max|du/dt|={summary['max_abs_du_dt']:.3e}, "
        f"max|dT/dt|={summary['max_abs_dT_dt']:.3e}, "
        f"mass drift={summary['mass_rel_drift_final']:+.3e}"
    )
    return summary


def _sw_enstrophy(state, grid) -> jnp.ndarray:
    zeta = curl_z(state.u, state.v, grid).data
    abs_vor = zeta + grid.f
    h = jnp.clip(state.h.data, 1.0, None)
    return jnp.sum(0.5 * abs_vor * abs_vor / h * grid.area)


def _run_solid_body_case(case_dir: Path) -> dict[str, float]:
    print("\n[2/4] Solid-body rotation (Williamson 2) with diagnostics")
    grid = create_cubed_sphere(12)
    state0 = _williamson_test2_state(grid)
    state = state0
    model = ShallowWaterModel(
        grid,
        ShallowWaterConfig(
            hyperdiff_coeff=1.0e15,
            use_conservation_fixer=False,
            time_integrator="ssp45",
        ),
    )

    dt = 120.0
    n_steps = 120  # Match regression test horizon
    snap_steps = _snapshot_steps(n_steps)

    diagnostics: list[dict[str, float]] = []
    snapshots: dict[int, dict[str, np.ndarray]] = {}

    d0 = compute_conservation_diagnostics(state0, grid)
    mass0 = float(d0["total_mass"])
    energy0 = float(d0["total_energy"])
    enst0 = float(_sw_enstrophy(state0, grid))

    t0 = time.time()
    for step in range(n_steps + 1):
        d = compute_conservation_diagnostics(state, grid)
        mass = float(d["total_mass"])
        energy = float(d["total_energy"])
        enst = float(_sw_enstrophy(state, grid))
        speed = jnp.sqrt(state.u.data * state.u.data + state.v.data * state.v.data)
        diagnostics.append(
            {
                "step": float(step),
                "time_days": float(step * dt / 86400.0),
                "mean_h": float(jnp.mean(state.h.data)),
                "max_wind": float(jnp.max(speed)),
                "mass": mass,
                "mass_rel_drift": (mass - mass0) / mass0,
                "energy": energy,
                "energy_rel_drift": (energy - energy0) / energy0,
                "enstrophy": enst,
                "enstrophy_rel_drift": (enst - enst0) / enst0,
            }
        )

        if step in snap_steps:
            snapshots[step] = {
                "wind_speed": np.asarray(speed),
                "h_anomaly": np.asarray(state.h.data - state0.h.data),
                "vorticity": np.asarray(curl_z(state.u, state.v, grid).data),
            }

        if step < n_steps:
            state = model.step(state, dt)

    _plot_cube_snapshots(
        case_dir / "field_snapshots.png",
        "Solid-Body Rotation Williamson 2",
        snapshots,
        dt,
        [
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("h_anomaly", "Fluid depth anomaly (m)", "RdBu_r"),
            ("vorticity", "Relative vorticity (1/s)", "coolwarm"),
        ],
        np.asarray(grid.lon) * 180.0 / np.pi,
        np.asarray(grid.lat) * 180.0 / np.pi,
    )

    time_days = np.array([d["time_days"] for d in diagnostics], dtype=np.float64)
    _plot_timeseries(
        case_dir / "diagnostics_timeseries.png",
        "Solid-Body Rotation - Mean Diagnostics",
        time_days,
        [
            ("Global mean h (m)", np.array([d["mean_h"] for d in diagnostics]), "tab:blue"),
            ("Max wind (m/s)", np.array([d["max_wind"] for d in diagnostics]), "tab:orange"),
            ("Enstrophy", np.array([d["enstrophy"] for d in diagnostics]), "tab:red"),
        ],
    )
    _plot_timeseries(
        case_dir / "conservation_timeseries.png",
        "Solid-Body Rotation - Conservation",
        time_days,
        [
            ("Mass rel. drift", np.array([d["mass_rel_drift"] for d in diagnostics]), "tab:green"),
            ("Energy rel. drift", np.array([d["energy_rel_drift"] for d in diagnostics]), "tab:purple"),
            ("Enstrophy rel. drift", np.array([d["enstrophy_rel_drift"] for d in diagnostics]), "tab:brown"),
        ],
    )

    _write_csv(case_dir / "diagnostics.csv", diagnostics)

    wall = time.time() - t0
    summary = {
        "case": "solid_body_rotation_williamson2",
        "dt_s": dt,
        "n_steps": n_steps,
        "wall_time_s": wall,
        "mass_rel_drift_final": diagnostics[-1]["mass_rel_drift"],
        "energy_rel_drift_final": diagnostics[-1]["energy_rel_drift"],
        "enstrophy_rel_drift_final": diagnostics[-1]["enstrophy_rel_drift"],
        "max_wind_final": diagnostics[-1]["max_wind"],
    }
    with (case_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(
        "  done | "
        f"mass drift={summary['mass_rel_drift_final']:+.3e}, "
        f"energy drift={summary['energy_rel_drift_final']:+.3e}, "
        f"enstrophy drift={summary['enstrophy_rel_drift_final']:+.3e}"
    )
    return summary


def _plot_column_snapshots(
    out_path: Path,
    case_name: str,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
) -> None:
    steps = sorted(snapshots.keys())
    n_cols = len(steps)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.2 * n_cols, 10.0), sharey=True)
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for col, step in enumerate(steps):
        snap = snapshots[step]
        p_hpa = snap["p_full_pa"] / 100.0
        p_int_hpa = snap["p_half_pa"] / 100.0

        time_label = _format_sim_time(step, dt)

        axes[0, col].plot(snap["temperature_k"], p_hpa, color="tab:red", lw=2)
        axes[0, col].set_title(f"Temperature profile\nstep {step}, t={time_label}", fontsize=10)
        axes[0, col].set_xlabel("T (K)")

        axes[1, col].plot(snap["heating_kday"], p_hpa, color="tab:blue", lw=2)
        axes[1, col].axvline(0.0, color="0.4", lw=0.8)
        axes[1, col].set_title(f"Heating profile\nstep {step}, t={time_label}", fontsize=10)
        axes[1, col].set_xlabel("dT/dt (K/day)")

        axes[2, col].plot(snap["net_down_flux"], p_int_hpa, color="tab:green", lw=2)
        axes[2, col].axvline(0.0, color="0.4", lw=0.8)
        axes[2, col].set_title(f"Net flux profile\nstep {step}, t={time_label}", fontsize=10)
        axes[2, col].set_xlabel("Net down flux (W/m^2)")

        for r in range(3):
            axes[r, col].invert_yaxis()
            axes[r, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Pressure (hPa)")
    axes[1, 0].set_ylabel("Pressure (hPa)")
    axes[2, 0].set_ylabel("Pressure (hPa)")

    fig.suptitle(f"{case_name} - Profile Snapshots", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_single_column_radiation_case(case_dir: Path, scheme: str) -> dict[str, float]:
    print(f"\n[3/4 or 4/4] Single-column radiation ({scheme}) with diagnostics")
    ncol, nlev = 1, 24
    dt = 1800.0
    n_steps = 96  # 2 days

    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1, dtype=jnp.float32)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    sigma = p_full / p_half[:, -1:]

    T = jnp.broadcast_to(
        jnp.linspace(220.0, 295.0, nlev, dtype=jnp.float32)[None, :],
        (ncol, nlev),
    )
    q_v = jnp.clip(0.015 * sigma * sigma, 1.0e-5, 0.02)
    t_sfc = jnp.full((ncol,), 295.0, dtype=jnp.float32)

    if scheme == "gray":
        lat = jnp.zeros((ncol,), dtype=jnp.float32)
        insol = jnp.full((ncol,), 340.0, dtype=jnp.float32)
        cfg_gray = GrayRadiationConfig()

        def rad_fn(temp):
            return gray_radiation(temp, p_full, p_half, t_sfc, lat, q_v, insol, cfg_gray)

    elif scheme == "rrtmgp":
        cosz = jnp.full((ncol,), 0.5, dtype=jnp.float32)
        cfg_rrtmgp = RRTMGPConfig()

        def rad_fn(temp):
            return rrtmgp_radiation(temp, p_full, p_half, t_sfc, q_v, cosz, cfg_rrtmgp)

    else:
        raise ValueError(f"Unsupported scheme={scheme!r}")

    snap_steps = _snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, float]] = []

    dp = p_half[:, 1:] - p_half[:, :-1]
    h0 = float(jnp.sum(constants.c_pd / constants.g * T * dp))
    cumulative_flux_input = 0.0

    t0 = time.time()
    for step in range(n_steps + 1):
        out = rad_fn(T)
        hr = out.heating_rate  # K/s

        net_up = (out.lw_flux_up + out.sw_flux_up) - (out.lw_flux_down + out.sw_flux_down)
        net_down = -net_up
        f_toa_down = float(net_down[0, 0])
        f_sfc_down = float(net_down[0, -1])
        f_atm = f_toa_down - f_sfc_down

        col_heat_flux = float(jnp.sum(constants.c_pd / constants.g * hr * dp))
        closure_residual = col_heat_flux - f_atm

        hcol = float(jnp.sum(constants.c_pd / constants.g * T * dp))
        cum_heat_change = hcol - h0
        if step > 0:
            cumulative_flux_input += f_atm * dt
        cum_residual = cum_heat_change - cumulative_flux_input

        diagnostics.append(
            {
                "step": float(step),
                "time_days": float(step * dt / 86400.0),
                "mean_T": float(jnp.mean(T)),
                "column_heat_Jm2": hcol,
                "toa_net_down_Wm2": f_toa_down,
                "sfc_net_down_Wm2": f_sfc_down,
                "atm_net_radiative_input_Wm2": f_atm,
                "column_heating_flux_Wm2": col_heat_flux,
                "instant_closure_residual_Wm2": closure_residual,
                "cumulative_heat_change_Jm2": cum_heat_change,
                "cumulative_flux_input_Jm2": cumulative_flux_input,
                "cumulative_budget_residual_Jm2": cum_residual,
            }
        )

        if step in snap_steps:
            snapshots[step] = {
                "p_full_pa": np.asarray(p_full[0]),
                "p_half_pa": np.asarray(p_half[0]),
                "temperature_k": np.asarray(T[0]),
                "heating_kday": np.asarray(hr[0] * 86400.0),
                "net_down_flux": np.asarray(net_down[0]),
            }

        if step < n_steps:
            T = jnp.clip(T + dt * hr, 150.0, 360.0)

    _plot_column_snapshots(
        case_dir / "field_snapshots.png",
        f"Single-Column Radiation ({scheme})",
        snapshots,
        dt,
    )

    time_days = np.array([d["time_days"] for d in diagnostics], dtype=np.float64)
    _plot_timeseries(
        case_dir / "diagnostics_timeseries.png",
        f"Single-Column Radiation ({scheme}) - Mean Diagnostics",
        time_days,
        [
            ("Column mean T (K)", np.array([d["mean_T"] for d in diagnostics]), "tab:red"),
            (
                "Net flux TOA/SFC (W/m^2)",
                np.array([d["atm_net_radiative_input_Wm2"] for d in diagnostics]),
                "tab:blue",
            ),
            (
                "Column heating flux (W/m^2)",
                np.array([d["column_heating_flux_Wm2"] for d in diagnostics]),
                "tab:green",
            ),
        ],
    )
    _plot_timeseries(
        case_dir / "conservation_timeseries.png",
        f"Single-Column Radiation ({scheme}) - Energy Budget",
        time_days,
        [
            (
                "Instant closure residual (W/m^2)",
                np.array([d["instant_closure_residual_Wm2"] for d in diagnostics]),
                "tab:purple",
            ),
            (
                "Cumulative budget residual (J/m^2)",
                np.array([d["cumulative_budget_residual_Jm2"] for d in diagnostics]),
                "tab:brown",
            ),
        ],
    )

    _write_csv(case_dir / "diagnostics.csv", diagnostics)

    wall = time.time() - t0
    summary = {
        "case": f"single_column_radiation_{scheme}",
        "dt_s": dt,
        "n_steps": n_steps,
        "wall_time_s": wall,
        "mean_T_final": diagnostics[-1]["mean_T"],
        "mean_T_change": diagnostics[-1]["mean_T"] - diagnostics[0]["mean_T"],
        "instant_closure_residual_rms_Wm2": float(
            np.sqrt(np.mean(np.square([d["instant_closure_residual_Wm2"] for d in diagnostics])))
        ),
        "cumulative_budget_residual_final_Jm2": diagnostics[-1]["cumulative_budget_residual_Jm2"],
    }
    with (case_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(
        "  done | "
        f"dT_mean={summary['mean_T_change']:+.3f} K, "
        f"rms closure={summary['instant_closure_residual_rms_Wm2']:.3e} W/m^2, "
        f"cum residual={summary['cumulative_budget_residual_final_Jm2']:+.3e} J/m^2"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run atmosphere invariant diagnostics suite")
    parser.add_argument(
        "--output",
        type=str,
        default="results/atmosphere_invariant_diagnostics",
        help="Output directory",
    )
    args = parser.parse_args()

    # Reduce noisy dtype warnings from x64-disabled JAX paths in RRTMGP.
    warnings.filterwarnings("ignore", message="Explicitly requested dtype")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Atmosphere Invariant Diagnostics")
    print("=" * 78)
    print(f"Backend: {jax.default_backend()} | x64={jax.config.jax_enable_x64}")
    print(f"Output directory: {outdir}")

    summaries: list[dict[str, float]] = []
    t0 = time.time()

    case1 = outdir / "rest_state_invariance"
    case1.mkdir(parents=True, exist_ok=True)
    summaries.append(_run_rest_state_case(case1))

    case2 = outdir / "solid_body_rotation"
    case2.mkdir(parents=True, exist_ok=True)
    summaries.append(_run_solid_body_case(case2))

    case3 = outdir / "single_column_radiation_gray"
    case3.mkdir(parents=True, exist_ok=True)
    summaries.append(_run_single_column_radiation_case(case3, "gray"))

    case4 = outdir / "single_column_radiation_rrtmgp"
    case4.mkdir(parents=True, exist_ok=True)
    summaries.append(_run_single_column_radiation_case(case4, "rrtmgp"))

    total_wall = time.time() - t0
    suite_summary = {
        "total_wall_time_s": total_wall,
        "cases": summaries,
    }
    with (outdir / "summary.json").open("w") as f:
        json.dump(suite_summary, f, indent=2)

    print("\nCompleted all cases.")
    print(f"Total wall time: {total_wall:.1f}s")
    print(f"Suite summary: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
