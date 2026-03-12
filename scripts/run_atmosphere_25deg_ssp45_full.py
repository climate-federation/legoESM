#!/usr/bin/env python
"""Atmospheric dycore full suite at ~2.5 degree with SSP(5,4).

Runs a broad atmosphere dycore matrix with:
- finite volume (cubed-sphere + lat-lon hydro)
- spectral (Gaussian grid)
- SSP45 outer time integration with per-case dt fallback

Per case outputs:
- horizontal field snapshots at start/mid/end
- integrated mean-field time series
- vertical mean profiles at start/mid/end (non-shallow-water cases)
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:
    ccrs = None
    HAS_CARTOPY = False

MAP_PROJECTION = "platecarree"
DRAW_COASTLINES = False


def _snapshot_steps(n_steps: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    return sorted({0, max(1, n_steps // 2), n_steps})


def _format_sim_time(step: int, dt: float) -> str:
    t_sec = step * dt
    if t_sec < 3600.0:
        return f"{t_sec / 60.0:.1f} min"
    if t_sec < 86400.0:
        return f"{t_sec / 3600.0:.2f} h"
    return f"{t_sec / 86400.0:.2f} d"


def _capture_snapshot(
    snapshots: dict[int, dict[str, np.ndarray]],
    targets: list[int],
    step: int,
    extractor,
    state,
):
    if step in targets and step not in snapshots:
        snapshots[step] = extractor(state)


def _capture_profile(
    profiles: dict[int, dict[str, np.ndarray]],
    targets: list[int],
    step: int,
    extractor,
    state,
):
    if step in targets and step not in profiles:
        profiles[step] = extractor(state)


def _color_limits(panels: list[np.ndarray]) -> tuple[float, float]:
    finite_chunks = [p[np.isfinite(p)] for p in panels if p is not None]
    finite_chunks = [c for c in finite_chunks if c.size > 0]
    if not finite_chunks:
        return -1.0, 1.0
    values = np.concatenate(finite_chunks)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin < 0.0 < vmax:
        vmax_abs = max(abs(vmin), abs(vmax), 1.0e-12)
        vmin, vmax = -vmax_abs, vmax_abs
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1.0e-6
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _weighted_mean_2d(field: np.ndarray, area: np.ndarray) -> float:
    num = np.sum(field * area)
    den = np.sum(area)
    return float(num / den)


def _horizontal_profile_3d(field: np.ndarray, area: np.ndarray) -> np.ndarray:
    num = np.sum(field * area[..., None], axis=tuple(range(field.ndim - 1)))
    den = np.sum(area)
    return np.asarray(num / den)


def _vertical_integral_from_profile(profile: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights)
    return float(np.sum(profile * w) / np.sum(w))


def _save_snapshots(
    out_dir: Path,
    case_name: str,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    field_specs: list[tuple[str, str, str]],
    *,
    coord_kind: str,
    lon_deg: np.ndarray | None = None,
    lat_deg: np.ndarray | None = None,
):
    if not snapshots:
        return

    steps = sorted(snapshots.keys())
    n_rows = len(field_specs)
    n_cols = len(steps)

    use_cube_proj = (
        coord_kind == "cube"
        and HAS_CARTOPY
        and lon_deg is not None
        and lat_deg is not None
    )
    use_latlon_map = coord_kind in ("latlon", "gaussian") and lon_deg is not None and lat_deg is not None

    fig = plt.figure(figsize=((5.0 if (use_cube_proj or use_latlon_map) else 4.3) * n_cols + 0.9, 3.4 * n_rows))
    width_ratios = [1.0] * n_cols + [0.06]
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=width_ratios,
        hspace=0.28,
        wspace=0.18,
    )

    if use_cube_proj or use_latlon_map:
        if MAP_PROJECTION == "robinson":
            proj = ccrs.Robinson(central_longitude=0.0)
        else:
            proj = ccrs.PlateCarree()
        axes = [[fig.add_subplot(gs[r, c], projection=proj) for c in range(n_cols)] for r in range(n_rows)]
    else:
        axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    caxes = [fig.add_subplot(gs[r, n_cols]) for r in range(n_rows)]

    if use_latlon_map:
        lon_1d = np.asarray(lon_deg)
        lat_1d = np.asarray(lat_deg)
        if lon_1d.ndim == 2:
            lon_2d = lon_1d
            lat_2d = np.asarray(lat_deg)
        else:
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon_pts = lon_2d.reshape(-1)
        lat_pts = lat_2d.reshape(-1)
        n_pts = lon_pts.size
        marker_size_latlon = max(0.6, 2400.0 / float(max(n_pts, 1)))

    for r, (key, label, cmap) in enumerate(field_specs):
        row_panels = []
        for st in steps:
            fld = snapshots[st].get(key)
            row_panels.append(None if fld is None else np.asarray(fld))

        valid_for_limits = []
        for p in row_panels:
            if p is None:
                continue
            if p.ndim == 3 and p.shape[0] == 6:
                valid_for_limits.append(p.reshape(-1))
            else:
                valid_for_limits.append(p)
        vmin, vmax = _color_limits([np.asarray(v).reshape(-1) for v in valid_for_limits])
        im = None

        for c, st in enumerate(steps):
            ax = axes[r][c]
            panel = row_panels[c]
            tlabel = _format_sim_time(st, dt)

            if panel is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if use_cube_proj and panel.ndim == 3 and panel.shape[0] == 6:
                    lon_pts = np.asarray(lon_deg).reshape(-1)
                    lat_pts = np.asarray(lat_deg).reshape(-1)
                    val_pts = panel.reshape(-1)
                    valid = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(val_pts)
                    lon_plot = lon_pts[valid]
                    if MAP_PROJECTION == "platecarree":
                        lon_plot = ((lon_plot + 180.0) % 360.0) - 180.0
                    n_face = panel.shape[1]
                    marker_size = max(0.8, 2200.0 / float(n_face * n_face))
                    im = ax.scatter(
                        lon_plot,
                        lat_pts[valid],
                        c=val_pts[valid],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=marker_size,
                        linewidths=0.0,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                    )
                    if MAP_PROJECTION == "platecarree":
                        ax.set_extent([-180.0, 180.0, -90.0, 90.0], ccrs.PlateCarree())
                    else:
                        ax.set_global()
                    if DRAW_COASTLINES:
                        ax.coastlines(linewidth=0.35, color="0.35")
                    ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.35)
                elif use_latlon_map and panel.ndim == 2:
                    vals = panel.reshape(-1)
                    valid = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(vals)
                    lon_plot = lon_pts[valid]
                    if MAP_PROJECTION == "platecarree":
                        lon_plot = ((lon_plot + 180.0) % 360.0) - 180.0
                    im = ax.scatter(
                        lon_plot,
                        lat_pts[valid],
                        c=vals[valid],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=marker_size_latlon,
                        linewidths=0.0,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                    )
                    if MAP_PROJECTION == "platecarree":
                        ax.set_extent([-180.0, 180.0, -90.0, 90.0], ccrs.PlateCarree())
                    else:
                        ax.set_global()
                    if DRAW_COASTLINES:
                        ax.coastlines(linewidth=0.35, color="0.35")
                    ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.35)
                else:
                    im = ax.imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                    ax.set_xticks([])
                    ax.set_yticks([])

            ax.set_title(f"{label}\nstep {st}, t={tlabel}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)

        if im is not None:
            fig.colorbar(im, cax=caxes[r], orientation="vertical")
        else:
            caxes[r].axis("off")

    fig.suptitle(f"{case_name} - Field Snapshots", fontsize=13)
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_dir / "field_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "snapshot_times.txt", "w") as f:
        f.write("step,time_seconds,time_days\n")
        for st in steps:
            t_sec = st * dt
            f.write(f"{st},{t_sec:.6f},{t_sec/86400.0:.8f}\n")


def _save_timeseries(
    out_dir: Path,
    case_name: str,
    series: dict[str, list[float]],
    dt: float,
    units: dict[str, str] | None = None,
):
    units = units or {}
    steps = np.asarray(series.get("step", []), dtype=float)
    if steps.size == 0:
        return

    keys = [k for k in series if k != "step"]
    t_sec = steps * dt
    t_days = t_sec / 86400.0

    with open(out_dir / "integrated_timeseries.csv", "w") as f:
        header = ["step", "time_seconds", "time_days"] + keys
        f.write(",".join(header) + "\n")
        for i in range(steps.size):
            row = [f"{steps[i]:.0f}", f"{t_sec[i]:.6f}", f"{t_days[i]:.8f}"]
            row.extend(f"{series[k][i]:.10e}" for k in keys)
            f.write(",".join(row) + "\n")

    fig, axes = plt.subplots(len(keys), 1, figsize=(9.0, 2.8 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ax.plot(t_days, np.asarray(series[key]), lw=1.7)
        unit = units.get(key, "")
        ax.set_ylabel(key if not unit else f"{key} ({unit})")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} - Integrated Mean Fields", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "integrated_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_profiles(
    out_dir: Path,
    case_name: str,
    profiles: dict[int, dict[str, np.ndarray]],
    dt: float,
    level_values: np.ndarray,
    level_label: str,
    invert_y: bool,
    units: dict[str, str] | None = None,
):
    if not profiles:
        return

    units = units or {}
    steps = sorted(profiles.keys())
    keys = sorted({k for p in profiles.values() for k in p.keys()})
    lev = np.asarray(level_values, dtype=float)

    with open(out_dir / "vertical_profiles.csv", "w") as f:
        f.write("step,time_seconds,time_days,level,variable,value\n")
        for st in steps:
            t_sec = st * dt
            t_day = t_sec / 86400.0
            for k in keys:
                if k not in profiles[st]:
                    continue
                vec = np.asarray(profiles[st][k], dtype=float)
                for li, vv in enumerate(vec):
                    f.write(f"{st},{t_sec:.6f},{t_day:.8f},{lev[li]:.10e},{k},{vv:.10e}\n")

    fig, axes = plt.subplots(1, len(keys), figsize=(4.8 * len(keys), 6.0), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        for st in steps:
            if key not in profiles[st]:
                continue
            vec = np.asarray(profiles[st][key], dtype=float)
            ax.plot(vec, lev, lw=1.7, label=f"step {st} ({_format_sim_time(st, dt)})")
        unit = units.get(key, "")
        ax.set_xlabel(key if not unit else f"{key} ({unit})")
        ax.grid(True, alpha=0.25)
        ax.set_title(key)
        ax.legend(fontsize=7)

    axes[0].set_ylabel(level_label)
    if invert_y:
        axes[0].invert_yaxis()
    fig.suptitle(f"{case_name} - Vertical Mean Profiles", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "vertical_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_with_dt_fallback(case_name: str, dt_candidates: list[float], runner):
    errors: list[str] = []
    for dt in dt_candidates:
        print(f"    Trying dt={dt:.3f}s")
        try:
            res = runner(dt)
        except Exception as exc:
            errors.append(f"dt={dt}: {exc}")
            traceback.print_exc()
            continue
        if res.get("stable", False):
            res["dt_selected"] = dt
            return res
        errors.append(f"dt={dt}: unstable")
        print(f"    Unstable at dt={dt:.3f}s, trying smaller dt.")

    return {
        "status": "FAIL",
        "stable": False,
        "metric": "dt_fallback_exhausted",
        "value": np.nan,
        "wall_time_s": 0.0,
        "notes": "; ".join(errors),
        "dt_selected": None,
        "case": case_name,
    }


def _series_init(keys: list[str]) -> dict[str, list[float]]:
    out = {"step": []}
    for k in keys:
        out[k] = []
    return out


def _series_push(series: dict[str, list[float]], step: int, vals: dict[str, float]):
    series["step"].append(float(step))
    for k, v in vals.items():
        series[k].append(float(v))


def _run_sw_fv_w2(out_dir: Path, n: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import ShallowWaterModel, ShallowWaterConfig
    from tests.test_cases.williamson import williamson_test2

    case_name = "SW FV Williamson 2"
    case_dir = out_dir / "01_sw_fv_williamson2"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    state0 = williamson_test2(grid)

    def _runner(dt):
        local_steps = int(5 * 86400 / dt)
        config = ShallowWaterConfig(
            hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
            time_integrator=solver,
            edge_blend_strength=0.25,
        )
        model = ShallowWaterModel(grid, config)
        state = state0

        snapshots = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_speed", "mean_height"])

        def _extract(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            return {
                "wind_speed": np.sqrt(u * u + v * v),
                "height": np.asarray(s.h.data),
            }

        def _push(step_i: int, fields: dict[str, np.ndarray]):
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_speed": _weighted_mean_2d(fields["wind_speed"], area),
                    "mean_height": _weighted_mean_2d(fields["height"], area),
                },
            )

        fields0 = _extract(state)
        _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
        _push(0, fields0)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, _extract(state))
            if not bool(jnp.all(jnp.isfinite(state.h.data))):
                stable = False
                break
        jax.block_until_ready(state.h.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} C{n} {solver}",
            snapshots,
            dt,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
            coord_kind="cube",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} C{n} {solver}",
            series,
            dt,
            {"mean_wind_speed": "m/s", "mean_height": "m"},
        )

        h_err = float(jnp.sqrt(jnp.mean((state.h.data - state0.h.data) ** 2)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\ndt: {dt}\n")
            f.write(f"n_steps: {local_steps}\n")
            f.write(f"l2_error_h: {h_err:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "l2_error_h",
            "value": h_err,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [450.0, 300.0, 200.0], _runner)


def _run_sw_fv_w5(out_dir: Path, n: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import ShallowWaterModel, ShallowWaterConfig
    from tests.test_cases.williamson import williamson_test5

    case_name = "SW FV Williamson 5"
    case_dir = out_dir / "02_sw_fv_williamson5"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    state0 = williamson_test5(grid)

    def _runner(dt):
        local_steps = int(15 * 86400 / dt)
        config = ShallowWaterConfig(
            hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
            time_integrator=solver,
            edge_blend_strength=0.25,
        )
        model = ShallowWaterModel(grid, config)
        state = state0

        snapshots = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_speed", "mean_height"])

        def _extract(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            return {
                "wind_speed": np.sqrt(u * u + v * v),
                "height": np.asarray(s.h.data),
            }

        def _push(step_i: int, fields: dict[str, np.ndarray]):
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_speed": _weighted_mean_2d(fields["wind_speed"], area),
                    "mean_height": _weighted_mean_2d(fields["height"], area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
        _push(0, _extract(state))

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, _extract(state))
            if not bool(jnp.all(jnp.isfinite(state.h.data))):
                stable = False
                break
        jax.block_until_ready(state.h.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} C{n} {solver}",
            snapshots,
            dt,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
            coord_kind="cube",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} C{n} {solver}",
            series,
            dt,
            {"mean_wind_speed": "m/s", "mean_height": "m"},
        )

        h0 = np.asarray(state0.h.data)
        h1 = np.asarray(state.h.data)
        mass_drift = float(abs(np.mean(h1) - np.mean(h0)) / max(abs(np.mean(h0)), 1.0e-12))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\ndt: {dt}\n")
            f.write(f"n_steps: {local_steps}\n")
            f.write(f"mass_drift: {mass_drift:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "mass_drift",
            "value": mass_drift,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [450.0, 300.0, 200.0], _runner)


def _run_hydro_fv_hs(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq import PrimitiveEquationModel, PrimitiveEquationConfig
    from legoesm.atmosphere.physics.held_suarez import held_suarez_init, held_suarez_forcing
    from legoesm.core.operators import global_integral

    case_name = "Hydro FV Held-Suarez"
    case_dir = out_dir / "03_hydro_fv_held_suarez"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full)
    dsigma = np.asarray(sigma.dsigma)

    def _runner(dt):
        local_steps = int(10 * 86400 / dt)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
            hyperdiff_ps_coeff=5.0e16 * (48.0 / n) ** 4,
            use_conservation_fixer=True,
            fix_mass=True,
            time_integrator=solver,
            edge_blend_uv=0.15,
            edge_blend_T=0.10,
            edge_blend_p_s=0.20,
            edge_blend_width=2,
        )
        model = PrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init(grid, sigma)
        mass_init = float(global_integral(state.p_s, grid))

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _extract_snapshot(s):
            u_sfc = np.asarray(s.u.data)[..., -1]
            v_sfc = np.asarray(s.v.data)[..., -1]
            return {
                "wind_sfc": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
                "p_s": np.asarray(s.p_s.data),
                "T_sfc": np.asarray(s.T.data)[..., -1],
            }

        def _extract_profile(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind = np.sqrt(u * u + v * v)
            wind_prof = _horizontal_profile_3d(wind, area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(s.p_s.data), area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step_with_physics(state, dt, held_suarez_forcing)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not bool(jnp.all(jnp.isfinite(state.u.data))) or u_max > 700.0:
                stable = False
                break
        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
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
        _save_timeseries(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        mass_final = float(global_integral(state.p_s, grid))
        mass_drift = abs(mass_final - mass_init) / max(abs(mass_init), 1.0e-12)
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"mass_drift: {mass_drift:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "mass_drift",
            "value": mass_drift,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [450.0, 300.0, 200.0, 150.0], _runner)


def _run_hydro_fv_bw(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq import PrimitiveEquationModel, PrimitiveEquationConfig
    from legoesm.atmosphere.physics.baroclinic_wave import baroclinic_wave_init
    from legoesm.core.operators import global_integral

    case_name = "Hydro FV Baroclinic Wave"
    case_dir = out_dir / "04_hydro_fv_baroclinic"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full)
    dsigma = np.asarray(sigma.dsigma)

    def _runner(dt):
        local_steps = int(10 * 86400 / dt)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
            hyperdiff_ps_coeff=5.0e16 * (48.0 / n) ** 4,
            use_conservation_fixer=True,
            fix_mass=True,
            time_integrator=solver,
            edge_blend_uv=0.15,
            edge_blend_T=0.10,
            edge_blend_p_s=0.20,
            edge_blend_width=2,
        )
        model = PrimitiveEquationModel(grid, sigma, config)
        state = baroclinic_wave_init(grid, sigma, perturbed=True)
        mass_init = float(global_integral(state.p_s, grid))

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _extract_snapshot(s):
            u_sfc = np.asarray(s.u.data)[..., -1]
            v_sfc = np.asarray(s.v.data)[..., -1]
            return {
                "wind_sfc": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
                "p_s": np.asarray(s.p_s.data),
                "T_sfc": np.asarray(s.T.data)[..., -1],
            }

        def _extract_profile(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind = np.sqrt(u * u + v * v)
            wind_prof = _horizontal_profile_3d(wind, area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(s.p_s.data), area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not bool(jnp.all(jnp.isfinite(state.u.data))) or u_max > 700.0:
                stable = False
                break
        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
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
        _save_timeseries(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        mass_final = float(global_integral(state.p_s, grid))
        mass_drift = abs(mass_final - mass_init) / max(abs(mass_init), 1.0e-12)
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"mass_drift: {mass_drift:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "mass_drift",
            "value": mass_drift,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [300.0, 200.0, 150.0], _runner)


def _run_nh_fv_tc1(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerModel, CompressibleEulerConfig
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init

    case_name = "NH FV DCMIP TC1"
    case_dir = out_dir / "05_nh_fv_dcmip_tc1"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi

    state0, height_coord, terrain_metric = dcmip25_tc1_init(grid, n_levels=nlev)
    z_full = np.asarray(height_coord.z_full)
    dz = np.asarray(height_coord.dz)

    def _runner(dt):
        local_steps = int(1.0 * 3600.0 / dt)
        config = CompressibleEulerConfig(
            n_acoustic_substeps=8,
            sponge_width=10000.0,
            sponge_coeff=0.06,
            outer_integrator=solver,
            edge_blend_uv=0.22,
            edge_blend_w=0.14,
            edge_blend_theta=0.12,
            edge_blend_rho=0.18,
            edge_blend_width=3,
        )
        model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)
        state = state0

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_abs_w_3d", "mean_theta_prime_3d", "mean_rho_prime_3d"])

        def _extract_snapshot(s):
            u_low = np.asarray(s.u.data)[..., -1]
            v_low = np.asarray(s.v.data)[..., -1]
            w = np.asarray(s.w.data)
            return {
                "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
                "rho_prime_low": np.asarray(s.rho_prime.data)[..., -1],
                "w_mid": w[..., w.shape[-1] // 2],
            }

        def _extract_profile(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            w_half = np.asarray(s.w.data)
            w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "abs_w_profile": _horizontal_profile_3d(np.abs(w_full), area),
                "theta_prime_profile": _horizontal_profile_3d(np.asarray(s.theta_prime.data), area),
                "rho_prime_profile": _horizontal_profile_3d(np.asarray(s.rho_prime.data), area),
            }

        def _push(step_i: int, s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            w_half = np.asarray(s.w.data)
            w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            abs_w_prof = _horizontal_profile_3d(np.abs(w_full), area)
            th_prof = _horizontal_profile_3d(np.asarray(s.theta_prime.data), area)
            rho_prof = _horizontal_profile_3d(np.asarray(s.rho_prime.data), area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dz),
                    "mean_abs_w_3d": _vertical_integral_from_profile(abs_w_prof, dz),
                    "mean_theta_prime_3d": _vertical_integral_from_profile(th_prof, dz),
                    "mean_rho_prime_3d": _vertical_integral_from_profile(rho_prof, dz),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not bool(jnp.all(jnp.isfinite(state.u.data))) or u_max > 1200.0:
                stable = False
                break
        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            snapshots,
            dt,
            [
                ("wind_low", "Low-level wind speed (m/s)", "magma"),
                ("rho_prime_low", "Low-level rho' (kg/m3)", "RdBu_r"),
                ("w_mid", "Mid-level w (m/s)", "RdBu_r"),
            ],
            coord_kind="cube",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            series,
            dt,
            {
                "mean_wind_3d": "m/s",
                "mean_abs_w_3d": "m/s",
                "mean_theta_prime_3d": "K",
                "mean_rho_prime_3d": "kg/m3",
            },
        )
        _save_profiles(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
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

        w_max = float(jnp.max(jnp.abs(state.w.data)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_abs_w: {w_max:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_abs_w",
            "value": w_max,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [2.0, 1.0, 0.5], _runner)


def _run_nh_fv_tc2a(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerModel, CompressibleEulerConfig
    from tests.test_cases.dcmip2025 import dcmip25_tc2_init

    case_name = "NH FV DCMIP TC2a"
    case_dir = out_dir / "06_nh_fv_dcmip_tc2a"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    state0, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(grid, n_levels=nlev, subcase="a")
    area = np.asarray(small_grid.area)
    lon_deg = np.asarray(small_grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(small_grid.lat) * 180.0 / np.pi
    z_full = np.asarray(height_coord.z_full)
    dz = np.asarray(height_coord.dz)

    def _runner(dt):
        local_steps = int(0.05 * 3600.0 / dt)
        config = CompressibleEulerConfig(
            n_acoustic_substeps=6,
            sponge_width=15000.0,
            sponge_coeff=1.0 / (0.1 * 86400.0),
            small_earth_factor=20.0,
            outer_integrator=solver,
            edge_blend_uv=0.22,
            edge_blend_w=0.22,
            edge_blend_theta=0.12,
            edge_blend_rho=0.26,
            edge_blend_width=4,
        )
        model = CompressibleEulerModel(small_grid, height_coord, terrain_metric, config)
        state = state0

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_abs_w_3d", "mean_theta_prime_3d", "mean_rho_prime_3d"])

        def _extract_snapshot(s):
            u_low = np.asarray(s.u.data)[..., -1]
            v_low = np.asarray(s.v.data)[..., -1]
            w = np.asarray(s.w.data)
            return {
                "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
                "rho_prime_low": np.asarray(s.rho_prime.data)[..., -1],
                "w_mid": w[..., w.shape[-1] // 2],
            }

        def _extract_profile(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            w_half = np.asarray(s.w.data)
            w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "abs_w_profile": _horizontal_profile_3d(np.abs(w_full), area),
                "theta_prime_profile": _horizontal_profile_3d(np.asarray(s.theta_prime.data), area),
                "rho_prime_profile": _horizontal_profile_3d(np.asarray(s.rho_prime.data), area),
            }

        def _push(step_i: int, s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            w_half = np.asarray(s.w.data)
            w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            abs_w_prof = _horizontal_profile_3d(np.abs(w_full), area)
            th_prof = _horizontal_profile_3d(np.asarray(s.theta_prime.data), area)
            rho_prof = _horizontal_profile_3d(np.asarray(s.rho_prime.data), area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dz),
                    "mean_abs_w_3d": _vertical_integral_from_profile(abs_w_prof, dz),
                    "mean_theta_prime_3d": _vertical_integral_from_profile(th_prof, dz),
                    "mean_rho_prime_3d": _vertical_integral_from_profile(rho_prof, dz),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.u.data))):
                stable = False
                break
        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            snapshots,
            dt,
            [
                ("wind_low", "Low-level wind speed (m/s)", "magma"),
                ("rho_prime_low", "Low-level rho' (kg/m3)", "RdBu_r"),
                ("w_mid", "Mid-level w (m/s)", "RdBu_r"),
            ],
            coord_kind="cube",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            series,
            dt,
            {
                "mean_wind_3d": "m/s",
                "mean_abs_w_3d": "m/s",
                "mean_theta_prime_3d": "K",
                "mean_rho_prime_3d": "kg/m3",
            },
        )
        _save_profiles(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
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

        w_max = float(jnp.max(jnp.abs(state.w.data)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_abs_w: {w_max:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_abs_w",
            "value": w_max,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [1.0, 0.5, 0.2], _runner)


def _run_nh_fv_tc3(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerModel, CompressibleEulerConfig
    from tests.test_cases.dcmip2025 import dcmip25_tc3_init

    case_name = "NH FV DCMIP TC3"
    case_dir = out_dir / "07_nh_fv_dcmip_tc3"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    state0, height_coord, terrain_metric, small_grid = dcmip25_tc3_init(grid, n_levels=nlev)
    area = np.asarray(small_grid.area)
    lon_deg = np.asarray(small_grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(small_grid.lat) * 180.0 / np.pi
    z_full = np.asarray(height_coord.z_full)
    dz = np.asarray(height_coord.dz)

    def _runner(dt):
        local_steps = int(0.05 * 3600.0 / dt)
        config = CompressibleEulerConfig(
            n_acoustic_substeps=6,
            sponge_width=5000.0,
            sponge_coeff=0.05,
            small_earth_factor=60.0,
            use_coriolis=False,
            outer_integrator=solver,
        )
        model = CompressibleEulerModel(small_grid, height_coord, terrain_metric, config)
        state = state0

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_abs_w_3d", "mean_theta_prime_3d", "mean_rho_prime_3d"])

        def _extract_snapshot(s):
            u_low = np.asarray(s.u.data)[..., -1]
            v_low = np.asarray(s.v.data)[..., -1]
            w = np.asarray(s.w.data)
            out = {
                "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
                "rho_prime_low": np.asarray(s.rho_prime.data)[..., -1],
                "w_mid": w[..., w.shape[-1] // 2],
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
                "wind_profile": _horizontal_profile_3d(wind, area),
                "abs_w_profile": _horizontal_profile_3d(np.abs(w_full), area),
                "theta_prime_profile": _horizontal_profile_3d(np.asarray(s.theta_prime.data), area),
                "rho_prime_profile": _horizontal_profile_3d(np.asarray(s.rho_prime.data), area),
            }

        def _push(step_i: int, s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            w_half = np.asarray(s.w.data)
            w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            abs_w_prof = _horizontal_profile_3d(np.abs(w_full), area)
            th_prof = _horizontal_profile_3d(np.asarray(s.theta_prime.data), area)
            rho_prof = _horizontal_profile_3d(np.asarray(s.rho_prime.data), area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dz),
                    "mean_abs_w_3d": _vertical_integral_from_profile(abs_w_prof, dz),
                    "mean_theta_prime_3d": _vertical_integral_from_profile(th_prof, dz),
                    "mean_rho_prime_3d": _vertical_integral_from_profile(rho_prof, dz),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.u.data))):
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

        _save_snapshots(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            snapshots,
            dt,
            field_specs,
            coord_kind="cube",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            series,
            dt,
            {
                "mean_wind_3d": "m/s",
                "mean_abs_w_3d": "m/s",
                "mean_theta_prime_3d": "K",
                "mean_rho_prime_3d": "kg/m3",
            },
        )
        _save_profiles(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
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

        w_max = float(jnp.max(jnp.abs(state.w.data)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_abs_w: {w_max:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_abs_w",
            "value": w_max,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [1.0, 0.5, 0.2], _runner)


def _run_transport_fv(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.tracer_transport import TracerTransportModel, TracerTransportConfig
    from tests.test_cases.dcmip_transport import (
        dcmip11_wind,
        dcmip11_init,
        compute_tracer_error_norms,
        create_dcmip_sigma,
    )

    case_name = "Transport FV DCMIP 1-1"
    case_dir = out_dir / "08_transport_fv_dcmip11"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    sigma = create_dcmip_sigma(nlev)
    sigma_full = np.asarray(sigma.sigma_full)
    dsigma = np.asarray(sigma.dsigma)
    state0 = dcmip11_init(grid, sigma)

    def _runner(dt):
        local_steps = int(12 * 86400 / dt)
        config = TracerTransportConfig(hyperdiff_coeff=0.0, time_integrator=solver)
        model = TracerTransportModel(grid, sigma, dcmip11_wind, config)
        state = state0

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_q1_3d", "mean_q2_3d"])

        def _extract_snapshot(s):
            tr = np.asarray(s.tracers.data)
            out = {"q1_sfc": np.clip(tr[..., -1, 0], 0.0, None)}
            if tr.shape[-1] >= 2:
                out["q2_sfc"] = np.clip(tr[..., -1, 1], 0.0, None)
            return out

        def _extract_profile(s):
            tr = np.asarray(s.tracers.data)
            out = {"q1_profile": _horizontal_profile_3d(np.clip(tr[..., 0], 0.0, None), area)}
            if tr.shape[-1] >= 2:
                out["q2_profile"] = _horizontal_profile_3d(np.clip(tr[..., 1], 0.0, None), area)
            return out

        def _push(step_i: int, s):
            tr = np.asarray(s.tracers.data)
            q1_prof = _horizontal_profile_3d(np.clip(tr[..., 0], 0.0, None), area)
            q2_prof = _horizontal_profile_3d(np.clip(tr[..., 1], 0.0, None), area) if tr.shape[-1] >= 2 else q1_prof
            _series_push(
                series,
                step_i,
                {
                    "mean_q1_3d": _vertical_integral_from_profile(q1_prof, dsigma),
                    "mean_q2_3d": _vertical_integral_from_profile(q2_prof, dsigma),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.tracers.data))):
                stable = False
                break
        jax.block_until_ready(state.tracers.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            snapshots,
            dt,
            [
                ("q1_sfc", "Tracer q1 (surface)", "viridis"),
                ("q2_sfc", "Tracer q2 (surface)", "plasma"),
            ],
            coord_kind="cube",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            series,
            dt,
            {"mean_q1_3d": "kg/kg", "mean_q2_3d": "kg/kg"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} C{n}/L{nlev} {solver}",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"q1_profile": "kg/kg", "q2_profile": "kg/kg"},
        )

        norms = compute_tracer_error_norms(state, state0, grid)
        l2_q1 = float(norms["l2"][0])
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: {solver}\nresolution: C{n}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"l2_q1: {l2_q1:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "l2_q1",
            "value": l2_q1,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [900.0, 600.0, 450.0], _runner)


def _run_hydro_latlon_hs(out_dir: Path, n_lat: int, n_lon: int, nlev: int, mean_every: int):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
        LatLonPrimitiveEquationConfig,
        LatLonPrimitiveEquationModel,
        latlon_hydrostatic_tendencies,
        _filter_state,
    )
    from legoesm.core.state import HydrostaticState
    from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
    from legoesm.atmosphere.physics.held_suarez_latlon import held_suarez_forcing_latlon, held_suarez_init_latlon
    from legoesm.grids.polar_filter import compute_polar_filter_mask
    from legoesm.core.conservation import fix_mass_hydrostatic_latlon
    from legoesm.core.operators_latlon import global_integral

    case_name = "Hydro LatLon Held-Suarez"
    case_dir = out_dir / "09_hydro_latlon_held_suarez"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_latlon_grid(n_lat, n_lon)
    area = np.asarray(grid.area)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full)
    dsigma = np.asarray(sigma.dsigma)

    def _runner(dt):
        local_steps = int(10 * 86400 / dt)
        hyper = 2.0e16 * (64.0 / n_lat) ** 4
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=hyper,
            hyperdiff_ps_coeff=hyper,
            use_conservation_fixer=True,
            fix_mass=True,
            use_polar_filter=True,
            polar_filter_cutoff_deg=60.0,
        )
        polar_mask = compute_polar_filter_mask(
            grid,
            dt,
            config.polar_filter_max_wave_speed,
            config.polar_filter_cutoff_deg,
        )

        state = held_suarez_init_latlon(grid, sigma)
        mass_init = float(global_integral(state.p_s, grid))

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _extract_snapshot(s):
            u_sfc = np.asarray(s.u.data)[..., -1]
            v_sfc = np.asarray(s.v.data)[..., -1]
            return {
                "wind_sfc": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
                "p_s": np.asarray(s.p_s.data),
                "T_sfc": np.asarray(s.T.data)[..., -1],
            }

        def _extract_profile(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(s.p_s.data), area),
                },
            )

        def _step_with_physics_ssp45(s0, dt0):
            if polar_mask is not None:
                s_work = _filter_state(s0, grid, polar_mask)
            else:
                s_work = s0

            def _tendency_fn(s):
                phys = held_suarez_forcing_latlon(s, grid, sigma)
                tend = latlon_hydrostatic_tendencies(
                    s,
                    grid,
                    sigma,
                    config,
                    phys,
                    polar_mask,
                )
                return HydrostaticState(
                    u=s.u.replace(data=tend.du_dt.data),
                    v=s.v.replace(data=tend.dv_dt.data),
                    T=s.T.replace(data=tend.dT_dt.data),
                    p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                    phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                )

            s1 = ssp_rk54_step(s_work, _tendency_fn, dt0)
            if config.use_conservation_fixer and config.fix_mass:
                s1 = fix_mass_hydrostatic_latlon(s1, s_work, grid)
            return s1

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = _step_with_physics_ssp45(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not bool(jnp.all(jnp.isfinite(state.u.data))) or u_max > 900.0:
                stable = False
                break
        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} {n_lat}x{n_lon}/L{nlev} ssp45",
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
        _save_timeseries(
            case_dir,
            f"{case_name} {n_lat}x{n_lon}/L{nlev} ssp45",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} {n_lat}x{n_lon}/L{nlev} ssp45",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        mass_final = float(global_integral(state.p_s, grid))
        mass_drift = abs(mass_final - mass_init) / max(abs(mass_init), 1.0e-12)
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: ssp45\nresolution: {n_lat}x{n_lon}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"mass_drift: {mass_drift:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "mass_drift",
            "value": mass_drift,
            "wall_time_s": wall,
            "notes": "",
        }

    res = _run_with_dt_fallback(case_name, [300.0, 200.0, 150.0, 100.0], _runner)
    if res.get("status") == "PASS":
        return res

    print("    SSP45 exhausted; falling back to lat-lon RK3 for this case.")

    def _runner_rk3(dt):
        local_steps = int(10 * 86400 / dt)
        hyper = 2.0e16 * (64.0 / n_lat) ** 4
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=hyper,
            hyperdiff_ps_coeff=hyper,
            use_conservation_fixer=True,
            fix_mass=True,
            use_polar_filter=True,
            polar_filter_cutoff_deg=60.0,
        )
        model = LatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
        state = held_suarez_init_latlon(grid, sigma)
        mass_init = float(global_integral(state.p_s, grid))

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _extract_snapshot(s):
            u_sfc = np.asarray(s.u.data)[..., -1]
            v_sfc = np.asarray(s.v.data)[..., -1]
            return {
                "wind_sfc": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
                "p_s": np.asarray(s.p_s.data),
                "T_sfc": np.asarray(s.T.data)[..., -1],
            }

        def _extract_profile(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            T = np.asarray(s.T.data)
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(s.p_s.data), area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step_with_physics(state, dt, held_suarez_forcing_latlon)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.u.data))):
                stable = False
                break
        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} {n_lat}x{n_lon}/L{nlev} rk3-fallback",
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
        _save_timeseries(
            case_dir,
            f"{case_name} {n_lat}x{n_lon}/L{nlev} rk3-fallback",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} {n_lat}x{n_lon}/L{nlev} rk3-fallback",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        mass_final = float(global_integral(state.p_s, grid))
        mass_drift = abs(mass_final - mass_init) / max(abs(mass_init), 1.0e-12)
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: rk3_fallback\nresolution: {n_lat}x{n_lon}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"mass_drift: {mass_drift:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "mass_drift",
            "value": mass_drift,
            "wall_time_s": wall,
            "notes": "fallback_integrator=rk3_latlon",
        }

    res_fallback = _run_with_dt_fallback(case_name, [300.0, 200.0, 150.0, 100.0], _runner_rk3)
    if res_fallback.get("notes"):
        res_fallback["notes"] = f"{res_fallback['notes']}; fallback_integrator=rk3_latlon"
    else:
        res_fallback["notes"] = "fallback_integrator=rk3_latlon"
    return res_fallback


def _gaussian_area_weights(grid) -> np.ndarray:
    w_lat = np.asarray(grid.weights, dtype=np.float64)
    n_lon = int(grid.n_lon)
    return np.broadcast_to(w_lat[:, None] / float(n_lon), (w_lat.shape[0], n_lon))


def _run_sw_spec_w2(out_dir: Path, trunc: int, mean_every: int):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralSWConfig,
        spectral_sw_tendencies,
        williamson_test2_spectral,
        spectral_to_grid,
    )
    from legoesm.timestepping.ssp_rk54 import ssp_rk54_step

    case_name = "SW Spectral Williamson 2"
    case_dir = out_dir / "10_sw_spectral_williamson2"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_gaussian_grid(trunc)
    area = _gaussian_area_weights(grid)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    state0 = williamson_test2_spectral(grid)

    def _runner(dt):
        local_steps = int(5 * 86400 / dt)
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        config = SpectralSWConfig(hyperdiff_coeff=1.0 / (1.0 * 3600.0 * eig_max ** 2))

        def _tendency(s):
            return spectral_sw_tendencies(s, grid, config)

        step = jax.jit(lambda s, dt_step: ssp_rk54_step(s, _tendency, dt_step))
        state = state0

        snapshots = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_speed", "mean_height"])

        def _extract(s):
            fields = spectral_to_grid(s, grid)
            u = np.asarray(fields["u"])
            v = np.asarray(fields["v"])
            h = np.asarray(fields["h"])
            return {"wind_speed": np.sqrt(u * u + v * v), "height": h}

        def _push(step_i: int, fields):
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_speed": _weighted_mean_2d(fields["wind_speed"], area),
                    "mean_height": _weighted_mean_2d(fields["height"], area),
                },
            )

        f0 = _extract(state)
        _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
        _push(0, f0)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, _extract(state))
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                stable = False
                break
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} T{trunc} ssp45",
            snapshots,
            dt,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
            coord_kind="gaussian",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} T{trunc} ssp45",
            series,
            dt,
            {"mean_wind_speed": "m/s", "mean_height": "m"},
        )

        fields0 = spectral_to_grid(state0, grid)
        fields1 = spectral_to_grid(state, grid)
        h_err = float(jnp.sqrt(jnp.mean((fields1["h"] - fields0["h"]) ** 2)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: ssp45\ntruncation: T{trunc}\ndt: {dt}\n")
            f.write(f"n_steps: {local_steps}\nl2_error_h: {h_err:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "l2_error_h",
            "value": h_err,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [120.0, 90.0, 60.0], _runner)


def _run_sw_spec_w5(out_dir: Path, trunc: int, mean_every: int):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralSWConfig,
        spectral_sw_tendencies,
        williamson_test5_spectral,
        spectral_to_grid,
    )
    from legoesm.timestepping.ssp_rk54 import ssp_rk54_step

    case_name = "SW Spectral Williamson 5"
    case_dir = out_dir / "11_sw_spectral_williamson5"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_gaussian_grid(trunc)
    area = _gaussian_area_weights(grid)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    state0 = williamson_test5_spectral(grid)

    def _runner(dt):
        local_steps = int(15 * 86400 / dt)
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        config = SpectralSWConfig(
            mean_depth=5960.0,
            hyperdiff_coeff=1.0 / (1.0 * 3600.0 * eig_max ** 2),
            hyperdiff_order=2,
        )

        def _tendency(s):
            return spectral_sw_tendencies(s, grid, config)

        step = jax.jit(lambda s, dt_step: ssp_rk54_step(s, _tendency, dt_step))
        state = state0

        snapshots = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_speed", "mean_height"])

        def _extract(s):
            fields = spectral_to_grid(s, grid)
            u = np.asarray(fields["u"])
            v = np.asarray(fields["v"])
            h = np.asarray(fields["h"])
            return {"wind_speed": np.sqrt(u * u + v * v), "height": h}

        def _push(step_i: int, fields):
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_speed": _weighted_mean_2d(fields["wind_speed"], area),
                    "mean_height": _weighted_mean_2d(fields["height"], area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
        _push(0, _extract(state))

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, _extract(state))
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                stable = False
                break
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} T{trunc} ssp45",
            snapshots,
            dt,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
            coord_kind="gaussian",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} T{trunc} ssp45",
            series,
            dt,
            {"mean_wind_speed": "m/s", "mean_height": "m"},
        )

        h0 = np.asarray(spectral_to_grid(state0, grid)["h"])
        h1 = np.asarray(spectral_to_grid(state, grid)["h"])
        mass_drift = float(abs(np.mean(h1) - np.mean(h0)) / max(abs(np.mean(h0)), 1.0e-12))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: ssp45\ntruncation: T{trunc}\ndt: {dt}\n")
            f.write(f"n_steps: {local_steps}\nmass_drift: {mass_drift:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "mass_drift",
            "value": mass_drift,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [120.0, 90.0, 60.0], _runner)


def _run_hydro_spec_hs(out_dir: Path, trunc: int, nlev: int, mean_every: int):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel,
        SpectralPEConfig,
        SpectralHydrostaticState,
        isothermal_rest_state_spectral,
        spectral_pe_tendencies,
        spectral_pe_to_grid,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral
    from legoesm.timestepping.ssp_rk54 import ssp_rk54_step

    case_name = "Hydro Spectral Held-Suarez"
    case_dir = out_dir / "12_hydro_spectral_held_suarez"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_gaussian_grid(trunc)
    area = _gaussian_area_weights(grid)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full)
    dsigma = np.asarray(sigma.dsigma)
    k_jet = int(np.argmin(np.abs(sigma_full - 0.25)))
    k_mid = int(np.argmin(np.abs(sigma_full - 0.55)))

    state0 = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

    def _runner(dt):
        local_steps = int(10 * 86400 / dt)
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        config = SpectralPEConfig(
            hyperdiff_coeff=1.0 / (0.5 * 3600.0 * eig_max ** 2),
            hyperdiff_order=2,
            semi_implicit=False,
        )

        def _tendency(s):
            phys = held_suarez_forcing_spectral(s, grid, sigma)
            tend = spectral_pe_tendencies(s, grid, sigma, config, phys)
            return SpectralHydrostaticState(
                vor_hat=s.vor_hat.replace(data=tend.vor_hat.data),
                div_hat=s.div_hat.replace(data=tend.div_hat.data),
                T_hat=s.T_hat.replace(data=tend.T_hat.data),
                lnps_hat=s.lnps_hat.replace(data=tend.lnps_hat.data),
                phis_hat=s.phis_hat.replace(data=jnp.zeros_like(s.phis_hat.data)),
            )

        step = jax.jit(lambda s, dt_step: ssp_rk54_step(s, _tendency, dt_step))
        state = state0

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _fields(s):
            return spectral_pe_to_grid(s, grid, sigma)

        def _extract_snapshot(s):
            f = _fields(s)
            u_jet = np.asarray(f["u"])[..., k_jet]
            v_jet = np.asarray(f["v"])[..., k_jet]
            ps = np.asarray(f["p_s"])
            return {
                "wind_jet": np.sqrt(u_jet * u_jet + v_jet * v_jet),
                "p_s_anom": ps - np.mean(ps, axis=1, keepdims=True),
                "T_mid": np.asarray(f["T"])[..., k_mid],
            }

        def _extract_profile(s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(f["p_s"]), area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                stable = False
                break
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            snapshots,
            dt,
            [
                ("wind_jet", "Wind speed @ jet level (m/s)", "magma"),
                ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
                ("T_mid", "Temperature @ mid level (K)", "coolwarm"),
            ],
            coord_kind="gaussian",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        f_end = _fields(state)
        max_wind = float(np.max(np.sqrt(np.asarray(f_end["u"]) ** 2 + np.asarray(f_end["v"]) ** 2)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: ssp45\ntruncation: T{trunc}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_wind: {max_wind:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_wind",
            "value": max_wind,
            "wall_time_s": wall,
            "notes": "",
        }

    res = _run_with_dt_fallback(case_name, [180.0, 120.0], _runner)
    if res.get("status") == "PASS":
        return res

    print("    SSP45 exhausted; falling back to SI-RK3 spectral PE for this case.")

    def _runner_si(dt):
        local_steps = int(10 * 86400 / dt)
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        config_si = SpectralPEConfig(
            hyperdiff_coeff=1.0 / (0.5 * 3600.0 * eig_max ** 2),
            hyperdiff_order=2,
            semi_implicit=True,
            si_T_ref=300.0,
            si_alpha=0.5,
            si_substeps=4,
            si_hyperdiff_boost=8.0,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, config_si)
        state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _fields(s):
            return spectral_pe_to_grid(s, grid, sigma)

        def _extract_snapshot(s):
            f = _fields(s)
            u_jet = np.asarray(f["u"])[..., k_jet]
            v_jet = np.asarray(f["v"])[..., k_jet]
            ps = np.asarray(f["p_s"])
            return {
                "wind_jet": np.sqrt(u_jet * u_jet + v_jet * v_jet),
                "p_s_anom": ps - np.mean(ps, axis=1, keepdims=True),
                "T_mid": np.asarray(f["T"])[..., k_mid],
            }

        def _extract_profile(s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(f["p_s"]), area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step_with_physics(state, dt, held_suarez_forcing_spectral)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                stable = False
                break
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} si-rk3-fallback",
            snapshots,
            dt,
            [
                ("wind_jet", "Wind speed @ jet level (m/s)", "magma"),
                ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
                ("T_mid", "Temperature @ mid level (K)", "coolwarm"),
            ],
            coord_kind="gaussian",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} si-rk3-fallback",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} si-rk3-fallback",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        f_end = _fields(state)
        max_wind = float(np.max(np.sqrt(np.asarray(f_end["u"]) ** 2 + np.asarray(f_end["v"]) ** 2)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: si_rk3_fallback\ntruncation: T{trunc}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_wind: {max_wind:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_wind",
            "value": max_wind,
            "wall_time_s": wall,
            "notes": "fallback_integrator=si_rk3_spectral_pe",
        }

    res_fallback = _run_with_dt_fallback(case_name, [600.0, 450.0, 300.0, 200.0], _runner_si)
    if res_fallback.get("notes"):
        res_fallback["notes"] = f"{res_fallback['notes']}; fallback_integrator=si_rk3_spectral_pe"
    else:
        res_fallback["notes"] = "fallback_integrator=si_rk3_spectral_pe"
    return res_fallback


def _run_hydro_spec_bw(out_dir: Path, trunc: int, nlev: int, mean_every: int):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel,
        SpectralPEConfig,
        SpectralHydrostaticState,
        baroclinic_wave_init_spectral,
        spectral_pe_tendencies,
        spectral_pe_to_grid,
    )
    from legoesm.timestepping.ssp_rk54 import ssp_rk54_step

    case_name = "Hydro Spectral Baroclinic Wave"
    case_dir = out_dir / "13_hydro_spectral_baroclinic"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_gaussian_grid(trunc)
    area = _gaussian_area_weights(grid)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full)
    dsigma = np.asarray(sigma.dsigma)
    k_jet = int(np.argmin(np.abs(sigma_full - 0.25)))
    k_mid = int(np.argmin(np.abs(sigma_full - 0.55)))

    state0 = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)

    def _runner(dt):
        local_steps = int(10 * 86400 / dt)
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        config = SpectralPEConfig(
            hyperdiff_coeff=1.0 / (0.1 * 3600.0 * eig_max ** 2),
            hyperdiff_order=2,
            semi_implicit=False,
        )

        def _tendency(s):
            tend = spectral_pe_tendencies(s, grid, sigma, config, None)
            return SpectralHydrostaticState(
                vor_hat=s.vor_hat.replace(data=tend.vor_hat.data),
                div_hat=s.div_hat.replace(data=tend.div_hat.data),
                T_hat=s.T_hat.replace(data=tend.T_hat.data),
                lnps_hat=s.lnps_hat.replace(data=tend.lnps_hat.data),
                phis_hat=s.phis_hat.replace(data=jnp.zeros_like(s.phis_hat.data)),
            )

        step = jax.jit(lambda s, dt_step: ssp_rk54_step(s, _tendency, dt_step))
        state = state0

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _fields(s):
            return spectral_pe_to_grid(s, grid, sigma)

        def _extract_snapshot(s):
            f = _fields(s)
            u_jet = np.asarray(f["u"])[..., k_jet]
            v_jet = np.asarray(f["v"])[..., k_jet]
            ps = np.asarray(f["p_s"])
            return {
                "wind_jet": np.sqrt(u_jet * u_jet + v_jet * v_jet),
                "p_s_anom": ps - np.mean(ps, axis=1, keepdims=True),
                "T_mid": np.asarray(f["T"])[..., k_mid],
            }

        def _extract_profile(s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(f["p_s"]), area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                stable = False
                break
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            snapshots,
            dt,
            [
                ("wind_jet", "Wind speed @ jet level (m/s)", "magma"),
                ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
                ("T_mid", "Temperature @ mid level (K)", "coolwarm"),
            ],
            coord_kind="gaussian",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        f_end = _fields(state)
        max_wind = float(np.max(np.sqrt(np.asarray(f_end["u"]) ** 2 + np.asarray(f_end["v"]) ** 2)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: ssp45\ntruncation: T{trunc}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_wind: {max_wind:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_wind",
            "value": max_wind,
            "wall_time_s": wall,
            "notes": "",
        }

    res = _run_with_dt_fallback(case_name, [120.0, 90.0], _runner)
    if res.get("status") == "PASS":
        return res

    print("    SSP45 exhausted; falling back to SI-RK3 spectral PE for this case.")

    def _runner_si(dt):
        local_steps = int(10 * 86400 / dt)
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        config_si = SpectralPEConfig(
            hyperdiff_coeff=1.0 / (0.2 * 3600.0 * eig_max ** 2),
            hyperdiff_order=2,
            semi_implicit=True,
            si_T_ref=300.0,
            si_alpha=0.5,
            si_substeps=4,
            si_hyperdiff_boost=8.0,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, config_si)
        state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)

        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_T_3d", "mean_p_s"])

        def _fields(s):
            return spectral_pe_to_grid(s, grid, sigma)

        def _extract_snapshot(s):
            f = _fields(s)
            u_jet = np.asarray(f["u"])[..., k_jet]
            v_jet = np.asarray(f["v"])[..., k_jet]
            ps = np.asarray(f["p_s"])
            return {
                "wind_jet": np.sqrt(u_jet * u_jet + v_jet * v_jet),
                "p_s_anom": ps - np.mean(ps, axis=1, keepdims=True),
                "T_mid": np.asarray(f["T"])[..., k_mid],
            }

        def _extract_profile(s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "T_profile": _horizontal_profile_3d(T, area),
            }

        def _push(step_i: int, s):
            f = _fields(s)
            u = np.asarray(f["u"])
            v = np.asarray(f["v"])
            T = np.asarray(f["T"])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            T_prof = _horizontal_profile_3d(T, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dsigma),
                    "mean_T_3d": _vertical_integral_from_profile(T_prof, dsigma),
                    "mean_p_s": _weighted_mean_2d(np.asarray(f["p_s"]), area),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = model.step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                stable = False
                break
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} si-rk3-fallback",
            snapshots,
            dt,
            [
                ("wind_jet", "Wind speed @ jet level (m/s)", "magma"),
                ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
                ("T_mid", "Temperature @ mid level (K)", "coolwarm"),
            ],
            coord_kind="gaussian",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} si-rk3-fallback",
            series,
            dt,
            {"mean_wind_3d": "m/s", "mean_T_3d": "K", "mean_p_s": "Pa"},
        )
        _save_profiles(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} si-rk3-fallback",
            profiles,
            dt,
            sigma_full,
            "Sigma",
            invert_y=True,
            units={"wind_profile": "m/s", "T_profile": "K"},
        )

        f_end = _fields(state)
        max_wind = float(np.max(np.sqrt(np.asarray(f_end["u"]) ** 2 + np.asarray(f_end["v"]) ** 2)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: si_rk3_fallback\ntruncation: T{trunc}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_wind: {max_wind:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_wind",
            "value": max_wind,
            "wall_time_s": wall,
            "notes": "fallback_integrator=si_rk3_spectral_pe",
        }

    res_fallback = _run_with_dt_fallback(case_name, [600.0, 450.0, 300.0, 200.0], _runner_si)
    if res_fallback.get("notes"):
        res_fallback["notes"] = f"{res_fallback['notes']}; fallback_integrator=si_rk3_spectral_pe"
    else:
        res_fallback["notes"] = "fallback_integrator=si_rk3_spectral_pe"
    return res_fallback


def _run_nh_spec_tc1(out_dir: Path, trunc: int, nlev: int, mean_every: int):
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis_3d, uv_from_vordiv_3d
    from legoesm.atmosphere.dynamics.spectral_nh import (
        SpectralCompressibleEulerModel,
        SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )
    from legoesm.timestepping.split_explicit import split_explicit_step, SplitExplicitConfig

    case_name = "NH Spectral DCMIP TC1"
    case_dir = out_dir / "14_nh_spectral_dcmip_tc1"
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = create_gaussian_grid(trunc)
    area = _gaussian_area_weights(grid)
    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    state0, height_coord, terrain_metric = dcmip25_tc1_init_spectral(grid, n_levels=nlev)
    z_full = np.asarray(height_coord.z_full)
    dz = np.asarray(height_coord.dz)

    def _runner(dt):
        local_steps = int(1.0 * 3600.0 / dt)
        a = grid.radius
        eig_max = trunc * (trunc + 1) / (a * a)
        config = SpectralNHConfig(
            n_acoustic_substeps=6,
            sponge_width=10000.0,
            sponge_coeff=0.05,
            hyperdiff_coeff=1.0 / (0.5 * 3600.0 * eig_max ** 2),
        )
        model = SpectralCompressibleEulerModel(grid, height_coord, terrain_metric, config)
        slow_fn, acoustic_fn = model._build_se_functions()
        se_cfg = SplitExplicitConfig(n_substeps=config.n_acoustic_substeps, outer_integrator="ssp45")
        step = jax.jit(lambda s, dt_step: split_explicit_step(s, slow_fn, acoustic_fn, dt_step, se_cfg))

        state = state0
        snapshots = {}
        profiles = {}
        snap_targets = _snapshot_steps(local_steps)
        series = _series_init(["mean_wind_3d", "mean_abs_w_3d", "mean_theta_prime_3d", "mean_rho_prime_3d"])

        def _to_grid(s):
            u_cos, v_cos = uv_from_vordiv_3d(grid, s.vor_hat.data, s.div_hat.data)
            cos_lat_3d = np.asarray(grid.cos_lat)[:, None, None]
            u = np.asarray(u_cos) / cos_lat_3d
            v = np.asarray(v_cos) / cos_lat_3d
            w_half = np.asarray(sh_synthesis_3d(grid, s.w_hat.data))
            theta_p = np.asarray(sh_synthesis_3d(grid, s.theta_prime_hat.data))
            rho_p = np.asarray(sh_synthesis_3d(grid, s.rho_prime_hat.data))
            return u, v, w_half, theta_p, rho_p

        def _extract_snapshot(s):
            u, v, w_half, theta_p, rho_p = _to_grid(s)
            u_low = u[..., -1]
            v_low = v[..., -1]
            return {
                "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
                "rho_prime_low": rho_p[..., -1],
                "w_mid": w_half[..., w_half.shape[-1] // 2],
            }

        def _extract_profile(s):
            u, v, w_half, theta_p, rho_p = _to_grid(s)
            w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
            wind = np.sqrt(u * u + v * v)
            return {
                "wind_profile": _horizontal_profile_3d(wind, area),
                "abs_w_profile": _horizontal_profile_3d(np.abs(w_full), area),
                "theta_prime_profile": _horizontal_profile_3d(theta_p, area),
                "rho_prime_profile": _horizontal_profile_3d(rho_p, area),
            }

        def _push(step_i: int, s):
            u, v, w_half, theta_p, rho_p = _to_grid(s)
            w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
            wind_prof = _horizontal_profile_3d(np.sqrt(u * u + v * v), area)
            abs_w_prof = _horizontal_profile_3d(np.abs(w_full), area)
            th_prof = _horizontal_profile_3d(theta_p, area)
            rho_prof = _horizontal_profile_3d(rho_p, area)
            _series_push(
                series,
                step_i,
                {
                    "mean_wind_3d": _vertical_integral_from_profile(wind_prof, dz),
                    "mean_abs_w_3d": _vertical_integral_from_profile(abs_w_prof, dz),
                    "mean_theta_prime_3d": _vertical_integral_from_profile(th_prof, dz),
                    "mean_rho_prime_3d": _vertical_integral_from_profile(rho_prof, dz),
                },
            )

        _capture_snapshot(snapshots, snap_targets, 0, _extract_snapshot, state)
        _capture_profile(profiles, snap_targets, 0, _extract_profile, state)
        _push(0, state)

        t0 = time.time()
        stable = True
        for i in range(local_steps):
            state = step(state, dt)
            st = i + 1
            _capture_snapshot(snapshots, snap_targets, st, _extract_snapshot, state)
            _capture_profile(profiles, snap_targets, st, _extract_profile, state)
            if st % mean_every == 0 or st == local_steps:
                _push(st, state)
            if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
                stable = False
                break
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_snapshots(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            snapshots,
            dt,
            [
                ("wind_low", "Low-level wind speed (m/s)", "magma"),
                ("rho_prime_low", "Low-level rho' (kg/m3)", "RdBu_r"),
                ("w_mid", "Mid-level w (m/s)", "RdBu_r"),
            ],
            coord_kind="gaussian",
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )
        _save_timeseries(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
            series,
            dt,
            {
                "mean_wind_3d": "m/s",
                "mean_abs_w_3d": "m/s",
                "mean_theta_prime_3d": "K",
                "mean_rho_prime_3d": "kg/m3",
            },
        )
        _save_profiles(
            case_dir,
            f"{case_name} T{trunc}/L{nlev} ssp45",
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

        u, v, w_half, _, _ = _to_grid(state)
        max_w = float(np.max(np.abs(w_half)))
        status = "PASS" if stable else "FAIL"
        with open(case_dir / "results.txt", "w") as f:
            f.write(f"solver: ssp45\ntruncation: T{trunc}\nlevels: {nlev}\n")
            f.write(f"dt: {dt}\nn_steps: {local_steps}\n")
            f.write(f"max_abs_w: {max_w:.8e}\n")
            f.write(f"stable: {stable}\nwall_time_s: {wall:.2f}\n")
        return {
            "case": case_name,
            "status": status,
            "stable": stable,
            "metric": "max_abs_w",
            "value": max_w,
            "wall_time_s": wall,
            "notes": "",
        }

    return _run_with_dt_fallback(case_name, [5.0, 3.0, 2.0, 1.0], _runner)


def _write_summary(out_dir: Path, cfg: dict, results: list[dict]):
    total_wall = float(sum(r.get("wall_time_s", 0.0) for r in results))
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_err = sum(1 for r in results if r.get("status") == "ERROR")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
        "summary": {
            "total_cases": len(results),
            "pass": n_pass,
            "fail": n_fail,
            "error": n_err,
            "total_wall_time_s": total_wall,
        },
        "cases": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Atmosphere Dycore ~2.5 Degree SSP45 Suite",
        "",
        f"Generated: {payload['generated']}",
        "",
        "## Configuration",
        "",
        f"- Cubed-sphere FV resolution: C{cfg['cube_resolution']}",
        f"- Lat-lon hydro resolution: {cfg['latlon_nlat']}x{cfg['latlon_nlon']}",
        f"- Spectral truncation: T{cfg['spectral_truncation']}",
        f"- Hydro levels: {cfg['hydro_levels']}",
        f"- NH levels: {cfg['nh_levels']}",
        f"- Solver: {cfg['solver']}",
        "",
        "## Case Results",
        "",
        "| Case | Status | dt (s) | Metric | Value | Wall Time (s) | Notes |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for r in results:
        val = r.get("value")
        if isinstance(val, (float, int, np.floating)):
            val_str = f"{float(val):.6e}"
        else:
            val_str = str(val)
        lines.append(
            f"| {r.get('case')} | {r.get('status')} | {r.get('dt_selected')} | "
            f"{r.get('metric')} | {val_str} | {r.get('wall_time_s', 0.0):.2f} | {r.get('notes', '')} |",
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Total cases: {len(results)}",
            f"- PASS: {n_pass}",
            f"- FAIL: {n_fail}",
            f"- ERROR: {n_err}",
            f"- Total wall time: {total_wall:.1f} s ({total_wall/60.0:.2f} min)",
            "",
            "Each case folder contains:",
            "- `field_snapshots.png`",
            "- `integrated_timeseries.png` and `integrated_timeseries.csv`",
            "- `vertical_profiles.png` and `vertical_profiles.csv` (non-shallow-water cases)",
            "- `results.txt`",
        ],
    )
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    global MAP_PROJECTION, DRAW_COASTLINES

    parser = argparse.ArgumentParser(description="Atmosphere dycore full suite (~2.5 degree, SSP45)")
    parser.add_argument("--cube-resolution", type=int, default=36, help="Cubed-sphere resolution Cn")
    parser.add_argument("--latlon-nlat", type=int, default=72, help="Lat-lon n_lat")
    parser.add_argument("--latlon-nlon", type=int, default=144, help="Lat-lon n_lon")
    parser.add_argument("--spectral-truncation", type=int, default=42, help="Spectral triangular truncation")
    parser.add_argument("--hydro-levels", type=int, default=20, help="Hydro/transport vertical levels")
    parser.add_argument("--nh-levels", type=int, default=20, help="NH vertical levels")
    parser.add_argument("--solver", type=str, default="ssp45", help="FV solver label for configurable models")
    parser.add_argument("--mean-every", type=int, default=20, help="Record integrated means every N steps")
    parser.add_argument(
        "--projection",
        type=str,
        default="platecarree",
        choices=("platecarree", "robinson"),
        help="Map projection for horizontal snapshot plots.",
    )
    parser.add_argument(
        "--coastlines",
        action="store_true",
        help="Draw coastlines on projected map snapshots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/atmosphere_dycore_25deg_ssp45_full)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=(
            "sw_fv_w2",
            "sw_fv_w5",
            "hydro_fv_hs",
            "hydro_fv_bw",
            "nh_fv_tc1",
            "nh_fv_tc2a",
            "nh_fv_tc3",
            "transport_fv",
            "hydro_latlon_hs",
            "sw_spec_w2",
            "sw_spec_w5",
            "hydro_spec_hs",
            "hydro_spec_bw",
            "nh_spec_tc1",
        ),
        help="Run only one case",
    )
    args = parser.parse_args()
    MAP_PROJECTION = args.projection
    DRAW_COASTLINES = bool(args.coastlines)

    out_dir = args.output or Path("results/atmosphere_dycore_25deg_ssp45_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Atmosphere Dycore Full Suite (~2.5 degree, SSP45)")
    print("=" * 78)
    print(f"Backend: {jax.default_backend()}")
    print(f"X64: {jax.config.jax_enable_x64}")
    print(f"Devices: {jax.devices()}")
    print(f"Output: {out_dir}")
    print(f"Cartopy: {HAS_CARTOPY}")
    print(f"Projection: {MAP_PROJECTION} | Coastlines: {DRAW_COASTLINES}")

    runners = [
        ("sw_fv_w2", lambda: _run_sw_fv_w2(out_dir, args.cube_resolution, args.solver, args.mean_every)),
        ("sw_fv_w5", lambda: _run_sw_fv_w5(out_dir, args.cube_resolution, args.solver, args.mean_every)),
        (
            "hydro_fv_hs",
            lambda: _run_hydro_fv_hs(out_dir, args.cube_resolution, args.hydro_levels, args.solver, args.mean_every),
        ),
        (
            "hydro_fv_bw",
            lambda: _run_hydro_fv_bw(out_dir, args.cube_resolution, args.hydro_levels, args.solver, args.mean_every),
        ),
        (
            "nh_fv_tc1",
            lambda: _run_nh_fv_tc1(out_dir, args.cube_resolution, args.nh_levels, args.solver, args.mean_every),
        ),
        (
            "nh_fv_tc2a",
            lambda: _run_nh_fv_tc2a(out_dir, args.cube_resolution, args.nh_levels, args.solver, args.mean_every),
        ),
        (
            "nh_fv_tc3",
            lambda: _run_nh_fv_tc3(out_dir, args.cube_resolution, args.nh_levels, args.solver, args.mean_every),
        ),
        (
            "transport_fv",
            lambda: _run_transport_fv(out_dir, args.cube_resolution, args.hydro_levels, args.solver, args.mean_every),
        ),
        (
            "hydro_latlon_hs",
            lambda: _run_hydro_latlon_hs(out_dir, args.latlon_nlat, args.latlon_nlon, args.hydro_levels, args.mean_every),
        ),
        ("sw_spec_w2", lambda: _run_sw_spec_w2(out_dir, args.spectral_truncation, args.mean_every)),
        ("sw_spec_w5", lambda: _run_sw_spec_w5(out_dir, args.spectral_truncation, args.mean_every)),
        (
            "hydro_spec_hs",
            lambda: _run_hydro_spec_hs(out_dir, args.spectral_truncation, args.hydro_levels, args.mean_every),
        ),
        (
            "hydro_spec_bw",
            lambda: _run_hydro_spec_bw(out_dir, args.spectral_truncation, args.hydro_levels, args.mean_every),
        ),
        (
            "nh_spec_tc1",
            lambda: _run_nh_spec_tc1(out_dir, args.spectral_truncation, args.nh_levels, args.mean_every),
        ),
    ]

    if args.only is not None:
        runners = [r for r in runners if r[0] == args.only]

    results = []
    t_all = time.time()
    for key, fn in runners:
        print("\n" + "-" * 78)
        print(f"Running case: {key}")
        print("-" * 78)
        try:
            res = fn()
        except Exception as exc:
            traceback.print_exc()
            res = {
                "case": key,
                "status": "ERROR",
                "stable": False,
                "metric": "error",
                "value": str(exc),
                "wall_time_s": 0.0,
                "notes": str(exc),
                "dt_selected": None,
            }
        results.append(res)
        print(
            f"  {res.get('status')} | dt={res.get('dt_selected')} | "
            f"{res.get('metric')}={res.get('value')} | {res.get('wall_time_s', 0.0):.1f}s",
        )

    total = time.time() - t_all
    cfg = {
        "cube_resolution": args.cube_resolution,
        "latlon_nlat": args.latlon_nlat,
        "latlon_nlon": args.latlon_nlon,
        "spectral_truncation": args.spectral_truncation,
        "hydro_levels": args.hydro_levels,
        "nh_levels": args.nh_levels,
        "solver": args.solver,
        "mean_every": args.mean_every,
        "projection": args.projection,
        "coastlines": bool(args.coastlines),
    }
    _write_summary(out_dir, cfg, results)

    print("\n" + "=" * 78)
    print(f"Suite complete in {total:.1f}s ({total/60.0:.2f} min)")
    print("=" * 78)
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_err = sum(1 for r in results if r.get("status") == "ERROR")
    print(f"PASS: {n_pass} | FAIL: {n_fail} | ERROR: {n_err} | TOTAL: {len(results)}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
