#!/usr/bin/env python
"""Ocean test matrix: organized test runner for legoESM ocean dynamical cores.

Runs rest-state adjustment, barotropic gravity wave, wind-driven gyre,
and baroclinic adjustment tests across cubed-sphere, lat-lon, MPAS, and
spectral (Gaussian) grids at ~5 degree resolution.

Output structure:
    results/ocean/<case>/<grid_type>/<resolution>/

Each case folder contains:
    - mean_timeseries.csv / .png      (domain-averaged scalar time series)
    - conservation_timeseries.csv/.png (volume, heat & salt drift)
    - field_snapshots.png              (2D field maps at selected times)
    - snapshots_<field>.png            (per-field snapshot evolution)
    - snapshots_native.npz             (snapshot arrays in native grid coords)
    - snapshots_latlon.npz             (snapshot arrays regridded to 181x360 lat-lon)
    - vertical_profiles.png            (vertical profile evolution)
    - latitude_vertical_cross_sections.png
    - longitude_vertical_cross_sections.png
    - results.txt                      (run metadata)

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --quick
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --only rest_state
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --grid cubed_sphere
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --grid spectral --only baroclinic
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.stdout.reconfigure(line_buffering=True)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Configuration constants
# ===========================================================================

# ~5 degree resolutions per grid type (ocean is more expensive than atm)
GRID_RESOLUTIONS: dict[str, str] = {
    "cubed_sphere": "C24",
    "latlon": "36x72",
    "mpas": "ico3",
    "spectral": "T21",
}

GRID_TYPES = list(GRID_RESOLUTIONS.keys())

DEFAULT_NLEV = 10
DEFAULT_H_MAX = 5500.0
DEFAULT_DT = 900.0  # seconds


# ===========================================================================
# TestCase dataclass
# ===========================================================================

@dataclass
class TestCase:
    """A single test case in the ocean matrix."""
    case: str               # rest_state, barotropic_wave, wind_gyre, baroclinic
    grid_type: str          # cubed_sphere, latlon, mpas, spectral
    resolution: str         # C24, 36x72, ico3, T21
    duration_days: float
    quick_days: float
    run_kwargs: dict = field(default_factory=dict)

    @property
    def output_path(self) -> str:
        return f"{self.case}/{self.grid_type}/{self.resolution}"


# ===========================================================================
# Test matrix generation
# ===========================================================================

def _build_test_matrix() -> list[TestCase]:
    """Generate the full test matrix from grid x case."""
    matrix: list[TestCase] = []
    res = GRID_RESOLUTIONS

    # --- Rest state adjustment: all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "rest_state", g, res[g], 1.0, 0.1))

    # --- Barotropic gravity wave: all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "barotropic_wave", g, res[g], 2.0, 0.2))

    # --- Wind-driven gyre: cubed_sphere, latlon, mpas ---
    for g in ["cubed_sphere", "latlon", "mpas"]:
        matrix.append(TestCase(
            "wind_gyre", g, res[g], 30.0, 2.0))

    # --- Baroclinic adjustment: all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "baroclinic", g, res[g], 10.0, 1.0))

    # --- Phillips two-layer baroclinic: all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "phillips_two_layer", g, res[g], 10.0, 1.0))

    return matrix


TEST_MATRIX = _build_test_matrix()


# ===========================================================================
# Results tracking
# ===========================================================================

ALL_RESULTS: list[dict[str, Any]] = []


def record(tc: TestCase, status: str, wall_time: float, notes: str = ""):
    icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!", "SKIP": "--"}[status]
    ALL_RESULTS.append({
        "test": tc.case, "grid": tc.grid_type,
        "resolution": tc.resolution, "status": status,
        "wall_time": wall_time, "notes": notes,
    })
    label = f"{tc.case}/{tc.grid_type}/{tc.resolution}"
    print(f"  {icon} {status:5s} | {label:<45s} | {wall_time:7.1f}s | {notes}")


# ===========================================================================
# Shared utilities
# ===========================================================================

def check_finite(arrays: dict[str, Any]) -> bool:
    for arr in arrays.values():
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False
    return True


def _snapshot_steps(n_steps: int, n_snaps: int = 10) -> set[int]:
    """Return step numbers at which to save snapshots."""
    if n_steps <= 0:
        return set()
    steps = {0, n_steps}
    for i in range(1, n_snaps):
        steps.add(max(1, int(i * n_steps / n_snaps)))
    return steps


def _compute_drift(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return abs(values[-1] - values[0]) / max(abs(values[0]), 1e-30)


# ===========================================================================
# Generic time loop
# ===========================================================================

def _run_timeloop(
    step_fn: Callable,
    state: Any,
    dt: float,
    n_steps: int,
    check_fn: Callable,
    scalar_fn: Callable,
    extract_fn: Callable,
    diag_every: int,
    key_array_fn: Callable,
    *,
    label: str = "",
    total_days: float = 0,
    blowup_threshold: float = 100.0,
    n_snaps: int = 10,
) -> tuple[Any, dict, dict, float, bool]:
    """Run time loop with diagnostics.

    Returns (final_state, snapshots, diag, wall_time, ok).
    """
    snap_targets = _snapshot_steps(n_steps, n_snaps)
    snapshots: dict[int, dict[str, np.ndarray]] = {0: extract_fn(state)}
    diag: dict[str, list] = {"times": [], "steps": []}

    t0 = time.time()
    last_print = t0
    blown_up = False

    for i in range(n_steps):
        state = step_fn(state, dt)
        step = i + 1

        if step in snap_targets:
            snapshots[step] = extract_fn(state)

        if step % 100 == 0:
            is_finite, metric = check_fn(state)
            if not is_finite or metric > blowup_threshold:
                print(f"  BLOWUP at step {step}, metric={metric:.1f}")
                blown_up = True
                break

        if step % diag_every == 0:
            day = step * dt / 86400.0
            scalars = scalar_fn(state)
            diag["times"].append(day)
            diag["steps"].append(step)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            now = time.time()
            if now - last_print > 30:
                summary = " | ".join(
                    f"{k}={v:.4g}" for k, v in list(scalars.items())[:3])
                print(f"    Day {day:7.1f}/{total_days} | {summary}")
                last_print = now

    jax.block_until_ready(key_array_fn(state))
    wall = time.time() - t0

    is_finite, _ = check_fn(state)
    ok = is_finite and not blown_up

    return state, snapshots, diag, wall, ok


# ===========================================================================
# Diagnostic saving
# ===========================================================================

def _build_latlon_weights(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    n_lat: int = 181,
    n_lon: int = 360,
    k: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build KDTree interpolation weights from unstructured to lat-lon grid.

    Returns (idxs, weights) arrays of shape (n_lat*n_lon, K) for K-nearest-
    neighbor inverse-distance weighting in 3-D Cartesian coordinates.
    """
    lon = ((np.asarray(lon_deg, dtype=np.float64).ravel() + 180) % 360) - 180
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64).ravel(), -90, 90)
    d2r = np.pi / 180.0
    src = np.column_stack([
        np.cos(lat * d2r) * np.cos(lon * d2r),
        np.cos(lat * d2r) * np.sin(lon * d2r),
        np.sin(lat * d2r)])
    lat_1d = np.linspace(-90.0, 90.0, n_lat)
    lon_1d = np.linspace(-180.0, 180.0, n_lon)
    lo, la = np.meshgrid(lon_1d, lat_1d)
    tgt = np.column_stack([
        np.cos(la.ravel() * d2r) * np.cos(lo.ravel() * d2r),
        np.cos(la.ravel() * d2r) * np.sin(lo.ravel() * d2r),
        np.sin(la.ravel() * d2r)])
    tree = cKDTree(src)
    K = min(k, src.shape[0])
    dists, idxs = tree.query(tgt, k=K)
    if K == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    w = 1.0 / np.maximum(dists, 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    return idxs, w


def _apply_weights(vals: np.ndarray, idxs: np.ndarray, w: np.ndarray,
                   n_lat: int, n_lon: int) -> np.ndarray:
    """Apply precomputed IDW weights, handling NaN source values."""
    v = vals[idxs]
    v_valid = np.isfinite(v)
    v_safe = np.where(v_valid, v, 0.0)
    wm = w * v_valid
    ws = wm.sum(axis=1, keepdims=True)
    wn = np.where(ws > 0, wm / np.maximum(ws, 1e-30), 0)
    result = np.sum(v_safe * wn, axis=1)
    return np.where(ws.ravel() > 0, result, np.nan).reshape(n_lat, n_lon)


def _bin_to_latlon(
    values: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    n_lat: int = 181,
    n_lon: int = 360,
) -> np.ndarray:
    """Interpolate unstructured points onto a regular lat-lon grid."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    if not np.any(np.isfinite(vals)):
        return np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon)
    return _apply_weights(vals, idxs, w, n_lat, n_lon)


def _regrid_2d(field_arr: np.ndarray, lon_deg: np.ndarray,
               lat_deg: np.ndarray, coord_kind: str) -> np.ndarray:
    """Regrid a 2D field to (181, 360) lat-lon."""
    if coord_kind in ("latlon", "gaussian"):
        return np.asarray(field_arr, dtype=np.float64)
    return _bin_to_latlon(field_arr.ravel(), lon_deg.ravel(), lat_deg.ravel())


def _regrid_3d_level(field_3d: np.ndarray, lon_deg: np.ndarray,
                     lat_deg: np.ndarray, coord_kind: str) -> np.ndarray:
    """Regrid a 3D field (*, nlev) to (181, 360, nlev)."""
    arr = np.asarray(field_3d, dtype=np.float64)
    if coord_kind in ("latlon", "gaussian"):
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr
    if arr.ndim == 1:
        arr = arr[:, None]
    nlev = arr.shape[-1]
    n_lat, n_lon = 181, 360
    flat = arr.reshape(-1, nlev)
    idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon)
    out = np.full((n_lat, n_lon, nlev), np.nan, dtype=np.float64)
    for k in range(nlev):
        out[..., k] = _apply_weights(flat[:, k], idxs, w, n_lat, n_lon)
    return out


def _fill_nan_profile(profile: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 1D profile by linear interpolation along index."""
    prof = np.asarray(profile, dtype=np.float64).copy()
    valid = np.isfinite(prof)
    if not np.any(valid):
        return prof
    if np.count_nonzero(valid) == 1:
        prof[:] = prof[valid][0]
        return prof
    x = np.arange(prof.size, dtype=np.float64)
    prof[:] = np.interp(x, x[valid], prof[valid])
    return prof


def _fill_nan_section(section: np.ndarray) -> np.ndarray:
    """Fill NaNs along the horizontal axis for each vertical level."""
    sec = np.asarray(section, dtype=np.float64).copy()
    if sec.ndim != 2:
        return sec
    for k in range(sec.shape[1]):
        sec[:, k] = _fill_nan_profile(sec[:, k])
    return sec


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_results_txt(output_dir: Path, rows: dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        for k, v in rows.items():
            f.write(f"{k}: {v}\n")


def _save_timeseries_csv(output_dir: Path, diag: dict, dt: float):
    keys = [k for k in diag if k not in ("steps", "times")]
    if not keys or not diag["steps"]:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "mean_timeseries.csv", "w") as f:
        f.write("step,time_days," + ",".join(keys) + "\n")
        for i in range(len(diag["steps"])):
            vals = ",".join(f"{diag[k][i]:.12e}" for k in keys)
            f.write(f"{diag['steps'][i]},{diag['times'][i]:.8f},{vals}\n")


def _save_timeseries_plot(output_dir: Path, case_name: str, diag: dict,
                          scalar_units: dict[str, str]):
    keys = [k for k in diag if k not in ("steps", "times")]
    times = diag.get("times", [])
    if not keys or not times:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ax.plot(times, diag[key], lw=1.5)
        unit = scalar_units.get(key, "")
        ax.set_ylabel(f"{key}" + (f" ({unit})" if unit else ""))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} — domain-averaged time series", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_conservation(output_dir: Path, case_name: str, diag: dict,
                       vol_key: str, heat_key: str, salt_key: str):
    vol_vals = diag.get(vol_key, [])
    heat_vals = diag.get(heat_key, [])
    salt_vals = diag.get(salt_key, [])
    times = diag.get("times", [])
    if not vol_vals or not heat_vals or not times:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    vol = np.array(vol_vals, dtype=np.float64)
    heat = np.array(heat_vals, dtype=np.float64)
    t = np.array(times, dtype=np.float64)
    vol_rel = (vol - vol[0]) / max(abs(vol[0]), 1e-30)
    heat_rel = (heat - heat[0]) / max(abs(heat[0]), 1e-30)

    n_panels = 2
    has_salt = len(salt_vals) == len(times)
    if has_salt:
        salt = np.array(salt_vals, dtype=np.float64)
        salt_rel = (salt - salt[0]) / max(abs(salt[0]), 1e-30)
        n_panels = 3

    with open(output_dir / "conservation_timeseries.csv", "w") as f:
        header = "time_days,volume,heat,vol_rel,heat_rel"
        if has_salt:
            header += ",salt,salt_rel"
        f.write(header + "\n")
        for i in range(t.size):
            line = (f"{t[i]:.8f},{vol[i]:.12e},{heat[i]:.12e},"
                    f"{vol_rel[i]:.12e},{heat_rel[i]:.12e}")
            if has_salt:
                line += f",{salt[i]:.12e},{salt_rel[i]:.12e}"
            f.write(line + "\n")

    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), sharex=True)
    axes[0].plot(t, vol_rel, lw=1.5)
    axes[0].axhline(0, color="0.3", ls="--", lw=0.8)
    axes[0].set_ylabel(f"Relative {vol_key} drift")
    axes[0].set_title("Volume conservation")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, heat_rel, lw=1.5, color="tab:red")
    axes[1].axhline(0, color="0.3", ls="--", lw=0.8)
    axes[1].set_ylabel(f"Relative {heat_key} drift")
    axes[1].set_title("Heat conservation")
    axes[1].grid(True, alpha=0.25)
    if has_salt:
        axes[2].plot(t, salt_rel, lw=1.5, color="tab:green")
        axes[2].axhline(0, color="0.3", ls="--", lw=0.8)
        axes[2].set_ylabel(f"Relative {salt_key} drift")
        axes[2].set_title("Salt conservation")
        axes[2].grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} — conservation diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        output_dir / "conservation_timeseries.png", dpi=150,
        bbox_inches="tight")
    plt.close(fig)


def _save_snapshot_plots(output_dir: Path, case_name: str, snapshots: dict,
                         dt: float, field_specs: list[tuple[str, str, str]],
                         coord_kind: str, lon_deg: np.ndarray,
                         lat_deg: np.ndarray):
    """Save snapshot evolution plots for each 2D field."""
    if not snapshots:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_steps = sorted(snapshots.keys())
    first_saved = None

    for field_key, field_label, cmap in field_specs:
        steps = [s for s in valid_steps if field_key in snapshots[s]]
        if not steps:
            continue
        if len(steps) > 8:
            idx = np.linspace(0, len(steps) - 1, 8).astype(int)
            steps = [steps[i] for i in idx]

        n_cols = min(4, len(steps))
        n_rows = (len(steps) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
        axes = np.atleast_2d(axes)
        im = None

        for idx, step in enumerate(steps):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            raw = np.asarray(snapshots[step][field_key], dtype=np.float64)
            regridded = _regrid_2d(raw, lon_deg, lat_deg, coord_kind)
            im = ax.imshow(
                regridded, origin="lower", aspect="auto", cmap=cmap,
                extent=[-180, 180, -90, 90])
            day = step * dt / 86400.0
            ax.set_title(f"t={day:.2f} d", fontsize=9)
            if c == 0:
                ax.set_ylabel("Latitude")
            if r == n_rows - 1:
                ax.set_xlabel("Longitude")

        for idx in range(len(steps), n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].set_visible(False)

        if im is not None:
            fig.colorbar(
                im, ax=axes.ravel().tolist(), orientation="vertical",
                fraction=0.02, pad=0.02, label=field_label)
        fig.suptitle(f"{case_name} — {field_key}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.96, 0.95])
        fname = f"snapshots_{field_key}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if first_saved is None:
            first_saved = fname

    # Alias first field as field_snapshots.png
    if first_saved and (output_dir / first_saved).exists():
        shutil.copy2(output_dir / first_saved,
                     output_dir / "field_snapshots.png")


def _save_cross_sections(output_dir: Path, case_name: str, snapshots: dict,
                         dt: float, field_3d_key: str, coord_kind: str,
                         lon_deg: np.ndarray, lat_deg: np.ndarray,
                         levels: np.ndarray, level_label: str):
    """Save latitude-vertical and longitude-vertical cross-sections."""
    valid_steps = sorted(
        s for s in snapshots if field_3d_key in snapshots[s])
    if not valid_steps:
        return
    if len(valid_steps) > 6:
        idx = np.linspace(0, len(valid_steps) - 1, 6).astype(int)
        valid_steps = [valid_steps[i] for i in idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    lat_axis = np.linspace(-90, 90, 181)
    lon_axis = np.linspace(-180, 180, 360)

    for fname, axis_vals, axis_key, mean_axis, xlabel in [
        ("latitude_vertical_cross_sections.png", lat_axis, "lat", 1, "Latitude"),
        ("longitude_vertical_cross_sections.png", lon_axis, "lon", 0, "Longitude"),
    ]:
        nc = len(valid_steps)
        fig, axes_arr = plt.subplots(
            1, nc, figsize=(4.5 * nc, 5), sharey=True)
        if nc == 1:
            axes_arr = [axes_arr]
        im = None

        for ax, step in zip(axes_arr, valid_steps):
            f3d = np.asarray(snapshots[step][field_3d_key], dtype=np.float64)
            ll = _regrid_3d_level(f3d, lon_deg, lat_deg, coord_kind)
            section = np.nanmean(ll, axis=mean_axis)
            section = _fill_nan_section(section)
            im = ax.imshow(
                section.T, origin="upper", aspect="auto", cmap="RdBu_r",
                extent=[axis_vals[0], axis_vals[-1],
                        float(levels[-1]), float(levels[0])])
            day = step * dt / 86400.0
            ax.set_title(f"t={day:.2f} d", fontsize=9)
            ax.set_xlabel(xlabel)

        axes_arr[0].set_ylabel(level_label)
        if im is not None:
            fig.colorbar(
                im, ax=axes_arr, orientation="vertical", fraction=0.028,
                pad=0.02, label=field_3d_key)
        fig.suptitle(
            f"{case_name} — {field_3d_key} cross-sections", fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.96, 0.95])
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_profiles(output_dir: Path, case_name: str, snapshots: dict,
                   dt: float, field_3d_key: str, levels: np.ndarray,
                   level_label: str):
    """Save vertical profile evolution plot."""
    valid_steps = sorted(
        s for s in snapshots if field_3d_key in snapshots[s])
    if not valid_steps:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_steps)))
    for step, color in zip(valid_steps, colors):
        f3d = np.asarray(snapshots[step][field_3d_key], dtype=np.float64)
        profile = np.nanmean(f3d, axis=tuple(range(f3d.ndim - 1)))
        day = step * dt / 86400.0
        ax.plot(profile, levels, color=color, lw=1.5, label=f"day {day:.1f}")
    ax.set_xlabel(field_3d_key)
    ax.set_ylabel(level_label)
    ax.invert_yaxis()
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_title(f"{case_name} — vertical profile evolution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "vertical_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_snapshot_times(output_dir: Path, snapshots: dict, dt: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "snapshot_times.txt", "w") as f:
        f.write("step,time_seconds,time_days\n")
        for step in sorted(snapshots.keys()):
            t_s = step * dt
            f.write(f"{step},{t_s:.2f},{t_s / 86400:.6f}\n")


def _save_snapshot_data(
    output_dir: Path,
    snapshots: dict,
    dt: float,
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
):
    """Save snapshot field arrays as NPZ files."""
    if not snapshots:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_steps = sorted(snapshots.keys())
    times_days = np.array([s * dt / 86400.0 for s in sorted_steps],
                          dtype=np.float64)

    native_arrays: dict[str, np.ndarray] = {
        "steps": np.array(sorted_steps, dtype=np.int64),
        "times_days": times_days,
    }
    latlon_arrays: dict[str, np.ndarray] = {
        "steps": np.array(sorted_steps, dtype=np.int64),
        "times_days": times_days,
        "lat": np.linspace(-90.0, 90.0, 181),
        "lon": np.linspace(-180.0, 180.0, 360),
    }

    for step in sorted_steps:
        for field_key, field_val in snapshots[step].items():
            arr = np.asarray(field_val, dtype=np.float64)
            key = f"{field_key}_step{step}"
            native_arrays[key] = arr
            if field_key.endswith("_3d"):
                latlon_arrays[key] = _regrid_3d_level(
                    arr, lon_deg, lat_deg, coord_kind)
            else:
                latlon_arrays[key] = _regrid_2d(
                    arr, lon_deg, lat_deg, coord_kind)

    np.savez_compressed(output_dir / "snapshots_native.npz", **native_arrays)
    np.savez_compressed(output_dir / "snapshots_latlon.npz", **latlon_arrays)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _save_case_diagnostics(
    output_dir: Path,
    case_name: str,
    dt: float,
    diag: dict,
    snapshots: dict,
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    field_specs_2d: list[tuple[str, str, str]],
    *,
    field_3d_key: str | None = None,
    level_values: np.ndarray | None = None,
    level_label: str = "Depth (m)",
    vol_key: str | None = None,
    heat_key: str | None = None,
    salt_key: str | None = None,
    scalar_units: dict[str, str] | None = None,
):
    """Save all standard diagnostic outputs for a test case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    su = scalar_units or {}

    _save_timeseries_csv(output_dir, diag, dt)
    _save_timeseries_plot(output_dir, case_name, diag, su)

    if vol_key and heat_key:
        _save_conservation(output_dir, case_name, diag,
                           vol_key, heat_key, salt_key or "")

    _save_snapshot_plots(
        output_dir, case_name, snapshots, dt, field_specs_2d,
        coord_kind, lon_deg, lat_deg)
    _save_snapshot_times(output_dir, snapshots, dt)
    _save_snapshot_data(
        output_dir, snapshots, dt, coord_kind, lon_deg, lat_deg)

    if field_3d_key and level_values is not None:
        _save_cross_sections(
            output_dir, case_name, snapshots, dt, field_3d_key,
            coord_kind, lon_deg, lat_deg, level_values, level_label)
        _save_profiles(
            output_dir, case_name, snapshots, dt, field_3d_key,
            level_values, level_label)


def _placeholder_plot(path: Path, title: str, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.6, title, ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(0.5, 0.3, text, ha="center", va="center", fontsize=8)
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _ensure_required_artifacts(output_dir: Path):
    """Guarantee standardized files exist in each case folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_name in ["mean_timeseries.csv", "conservation_timeseries.csv"]:
        p = output_dir / csv_name
        if not p.exists():
            with open(p, "w") as f:
                f.write("# No data produced\n")
    if not (output_dir / "snapshot_times.txt").exists():
        with open(output_dir / "snapshot_times.txt", "w") as f:
            f.write("step,time_seconds,time_days\n")
    # Note: PNGs are only created by the diagnostic routines when applicable.
    # No placeholder images are generated to avoid masking real issues.


# ===========================================================================
# Grid / model setup helpers
# ===========================================================================

def _parse_resolution(tc: TestCase):
    """Parse resolution string and return grid-appropriate parameters."""
    if tc.grid_type == "cubed_sphere":
        return {"n": int(tc.resolution[1:])}
    elif tc.grid_type == "latlon":
        parts = tc.resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif tc.grid_type == "mpas":
        return {"level": int(tc.resolution.replace("ico", ""))}
    elif tc.grid_type == "spectral":
        return {"truncation": int(tc.resolution[1:])}
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


def _create_ocean_setup(tc: TestCase, nlev: int = DEFAULT_NLEV,
                        H_max: float = DEFAULT_H_MAX):
    """Create grid, z_coord, and rest-state for any grid type.

    Returns (grid, z_coord, state, model, coord_kind, lon_deg, lat_deg).
    """
    from legoesm.ocean.vertical import create_ocean_z_star

    z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    params = _parse_resolution(tc)

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig

        n = params["n"]
        grid = create_cubed_sphere(n)
        config = OceanConfig(n_barotropic_substeps=30)
        model = OceanModel(grid, z_coord, config)
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon import LatLonOceanModel
        from legoesm.ocean.state import LatLonOceanConfig

        grid = create_latlon_grid(params["n_lat"], params["n_lon"])
        config = LatLonOceanConfig(n_barotropic_substeps=30)
        model = LatLonOceanModel(grid, z_coord, config)
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        mesh = create_voronoi_mesh(params["level"])
        config = MPASOceanConfig(n_barotropic_substeps=30)
        model = MPASOceanModel(mesh, z_coord, config)
        coord_kind = "mpas"
        lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        return mesh, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        from legoesm.ocean.state import SpectralOceanConfig

        trunc = params["truncation"]
        grid = create_gaussian_grid(trunc)
        config = SpectralOceanConfig()
        model = SpectralOceanModel(grid, z_coord, config)
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, config, model, coord_kind, lon_deg, lat_deg

    raise ValueError(f"Unknown grid type: {tc.grid_type}")


def _create_rest_state(tc: TestCase, grid, z_coord, H_max=DEFAULT_H_MAX):
    """Create rest-state initial condition for any grid type."""
    if tc.grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(grid, z_coord, H_max=H_max)
    elif tc.grid_type == "latlon":
        from legoesm.ocean.init_latlon import rest_state_latlon_ocean
        return rest_state_latlon_ocean(grid, z_coord, H_max=H_max)
    elif tc.grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(grid, z_coord, H_max=H_max)
    elif tc.grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        # No land mask for idealized spectral tests (avoids Gibbs ringing)
        return rest_state_spectral_ocean(grid, z_coord, H_max=H_max,
                                         land_lat_threshold=90.0)
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


# ===========================================================================
# Field extraction helpers
# ===========================================================================

def _extract_fv_ocean(state, grid_type: str):
    """Extract snapshot fields for cubed-sphere or lat-lon ocean state."""
    eta = np.asarray(state.eta.data, dtype=np.float64)
    T_3d = np.asarray(state.T.data, dtype=np.float64)
    S_3d = np.asarray(state.S.data, dtype=np.float64)
    u_sfc = np.asarray(state.u.data[..., 0], dtype=np.float64)
    result = {
        "eta": eta,
        "SST": np.asarray(state.T.data[..., 0], dtype=np.float64),
        "SSS": np.asarray(state.S.data[..., 0], dtype=np.float64),
        "u_sfc": u_sfc,
        "T_3d": T_3d,
        "S_3d": S_3d,
    }
    if hasattr(state, "v"):
        v_sfc = np.asarray(state.v.data[..., 0], dtype=np.float64)
        result["v_sfc"] = v_sfc
        result["speed_sfc"] = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    return result


def _extract_mpas_ocean(state, lon_deg, lat_deg):
    """Extract snapshot fields for MPAS ocean state.

    Returns raw cell-center arrays; regridding to lat-lon is done by the
    plotting functions via _regrid_2d / _regrid_3d_level.
    """
    eta = np.asarray(state.eta.data, dtype=np.float64)
    T_3d = np.asarray(state.T.data, dtype=np.float64)
    S_3d = np.asarray(state.S.data, dtype=np.float64)
    SST = T_3d[..., 0] if T_3d.ndim >= 2 else T_3d

    result = {
        "eta": eta,
        "SST": SST,
        "T_3d": T_3d,
        "S_3d": S_3d,
    }
    return result


def _extract_spectral_ocean(state, grid):
    """Extract snapshot fields for spectral ocean state."""
    from legoesm.grids.gaussian import sh_synthesis, sh_synthesis_3d

    eta_grid = np.asarray(
        sh_synthesis(grid, state.eta_hat.data), dtype=np.float64)
    T_grid = np.asarray(
        sh_synthesis_3d(grid, state.T_hat.data), dtype=np.float64)
    S_grid = np.asarray(
        sh_synthesis_3d(grid, state.S_hat.data), dtype=np.float64)
    SST = T_grid[..., 0] if T_grid.ndim >= 3 else T_grid
    SSS = S_grid[..., 0] if S_grid.ndim >= 3 else S_grid
    return {
        "eta": eta_grid,
        "SST": SST,
        "SSS": SSS,
        "T_3d": T_grid,
        "S_3d": S_grid,
    }


# ===========================================================================
# Scalar / check functions per grid type
# ===========================================================================

def _make_check_fn(grid_type: str):
    """Return a check_fn(state) -> (is_finite, metric)."""
    if grid_type == "spectral":
        def check_fn(s):
            eta_max = float(jnp.max(jnp.abs(s.eta_hat.data)))
            fin = (bool(jnp.all(jnp.isfinite(s.eta_hat.data))) and
                   bool(jnp.all(jnp.isfinite(s.T_hat.data))))
            return fin, eta_max
        return check_fn
    elif grid_type == "mpas":
        def check_fn(s):
            fin = check_finite({"eta": s.eta.data, "T": s.T.data,
                                "u": s.u.data})
            metric = float(jnp.max(jnp.abs(s.eta.data)))
            return fin, metric
        return check_fn
    else:
        def check_fn(s):
            fin = check_finite({"eta": s.eta.data, "T": s.T.data,
                                "u": s.u.data})
            metric = float(jnp.max(jnp.abs(s.eta.data)))
            return fin, metric
        return check_fn


def _make_scalar_fn(grid_type: str):
    """Return a scalar_fn(state) -> dict for diagnostics."""
    if grid_type == "spectral":
        def scalar_fn(s):
            return {
                "mean_eta_hat_abs": float(jnp.mean(jnp.abs(s.eta_hat.data))),
                "max_eta_hat_abs": float(jnp.max(jnp.abs(s.eta_hat.data))),
                "mean_T_hat_abs": float(jnp.mean(jnp.abs(s.T_hat.data))),
                "mean_S_hat_abs": float(jnp.mean(jnp.abs(s.S_hat.data))),
            }
        return scalar_fn
    elif grid_type == "mpas":
        def scalar_fn(s):
            return {
                "mean_eta": float(jnp.mean(s.eta.data)),
                "max_abs_eta": float(jnp.max(jnp.abs(s.eta.data))),
                "mean_T": float(jnp.mean(s.T.data)),
                "mean_S": float(jnp.mean(s.S.data)),
                "max_abs_u": float(jnp.max(jnp.abs(s.u.data))),
            }
        return scalar_fn
    else:
        def scalar_fn(s):
            return {
                "mean_eta": float(jnp.mean(s.eta.data)),
                "max_abs_eta": float(jnp.max(jnp.abs(s.eta.data))),
                "mean_T": float(jnp.mean(s.T.data)),
                "mean_S": float(jnp.mean(s.S.data)),
                "max_speed": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
            }
        return scalar_fn


def _make_extract_fn(grid_type: str, grid, lon_deg, lat_deg):
    """Return an extract_fn(state) -> dict for snapshots."""
    if grid_type == "spectral":
        def extract_fn(s):
            return _extract_spectral_ocean(s, grid)
        return extract_fn
    elif grid_type == "mpas":
        def extract_fn(s):
            return _extract_mpas_ocean(s, lon_deg, lat_deg)
        return extract_fn
    else:
        def extract_fn(s):
            return _extract_fv_ocean(s, grid_type)
        return extract_fn


def _key_array_fn(state, grid_type: str):
    """Return a key array for jax.block_until_ready."""
    if grid_type == "spectral":
        return state.eta_hat.data
    return state.eta.data


# ===========================================================================
# Barotropic wave perturbation
# ===========================================================================

def _add_barotropic_wave_perturbation(state, grid_type: str, grid, z_coord):
    """Add a Gaussian SSH perturbation to the rest state."""
    from legoesm.core.field import Field

    eta_amp = 1.0  # 1 m SSH perturbation
    sigma_deg = 10.0  # Gaussian width in degrees

    if grid_type == "cubed_sphere":
        lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        perturb = eta_amp * np.exp(
            -(lon ** 2 + lat ** 2) / (2 * sigma_deg ** 2))
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(new_eta))

    elif grid_type == "latlon":
        lon_1d = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        perturb = eta_amp * np.exp(
            -(lon_2d ** 2 + lat_2d ** 2) / (2 * sigma_deg ** 2))
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(new_eta))

    elif grid_type == "mpas":
        lon = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
        lat = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi
        perturb = eta_amp * np.exp(
            -(lon ** 2 + lat ** 2) / (2 * sigma_deg ** 2))
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(new_eta))

    elif grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis, sh_synthesis
        lon_1d = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        perturb = eta_amp * np.exp(
            -(lon_2d ** 2 + lat_2d ** 2) / (2 * sigma_deg ** 2))
        perturb_hat = sh_analysis(grid, jnp.array(perturb))
        new_eta_hat = state.eta_hat.data + perturb_hat
        return state._replace(eta_hat=Field(new_eta_hat))

    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Wind-driven gyre forcing
# ===========================================================================

def _add_wind_gyre_forcing(state, grid_type: str, grid, z_coord):
    """Create wind-gyre initial state (starts at rest with bathymetry)."""
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import wind_driven_gyre_init
        return wind_driven_gyre_init(grid, z_coord)
    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon import wind_driven_gyre_latlon
        return wind_driven_gyre_latlon(grid, z_coord)
    elif grid_type == "mpas":
        # MPAS doesn't have a dedicated gyre init; use rest state
        return state
    raise ValueError(f"Wind gyre not available for grid: {grid_type}")


# ===========================================================================
# Baroclinic adjustment perturbation
# ===========================================================================

def _add_baroclinic_perturbation(state, grid_type: str, grid, z_coord):
    """Add a meridional temperature front for baroclinic adjustment."""
    from legoesm.core.field import Field

    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis_3d
        lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        # Add meridional temperature gradient: +5C at equator, -5C at poles
        T_pert = 5.0 * np.cos(np.radians(lat))
        # Extend to 3D
        T_grid_hat = state.T_hat.data
        nlev = T_grid_hat.shape[-1]
        from legoesm.grids.gaussian import sh_synthesis_3d
        T_grid = np.array(sh_synthesis_3d(grid, T_grid_hat), dtype=np.float64)
        # T_pert is (n_lat,) → broadcast to (n_lat, n_lon)
        T_pert_2d = T_pert[:, np.newaxis] * np.ones((1, grid.n_lon))
        for k in range(nlev):
            decay = np.exp(-k / max(nlev / 3, 1))
            T_grid[..., k] += T_pert_2d * decay
        from legoesm.grids.gaussian import sh_analysis_3d
        new_T_hat = sh_analysis_3d(grid, jnp.array(T_grid))
        return state._replace(T_hat=Field(new_T_hat))

    else:
        # FV grids (cube, latlon, mpas)
        if grid_type == "mpas":
            lat = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi
        else:
            lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

        T_data = np.asarray(state.T.data, dtype=np.float64)
        nlev = T_data.shape[-1]
        T_pert = 5.0 * np.cos(np.radians(lat))
        # Reshape T_pert to broadcast with T_data[..., k]
        # For lat-lon: (n_lat,) → (n_lat, 1); for cube: (6, n, n) OK; for mpas: (nCells,) OK
        pert_shape = T_data.shape[:-1]  # spatial dims
        T_pert = np.broadcast_to(T_pert.reshape(
            T_pert.shape + (1,) * (len(pert_shape) - T_pert.ndim)), pert_shape)

        for k in range(nlev):
            decay = np.exp(-k / max(nlev / 3, 1))
            T_data[..., k] += T_pert * decay

        return state._replace(T=Field(jnp.array(T_data)))


# ===========================================================================
# Runner: Rest State
# ===========================================================================

def run_rest_state(tc: TestCase, output_dir: Path, days: float
                   ) -> tuple[str, float, str]:
    """Rest state adjustment: model should remain near initial condition."""
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state(tc, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State ({tc.grid_type})", total_days=days)

    # Check drift is small
    if tc.grid_type == "spectral":
        eta_drift = _compute_drift(diag.get("max_eta_hat_abs", []))
        T_drift = _compute_drift(diag.get("mean_T_hat_abs", []))
    else:
        eta_drift = _compute_drift(diag.get("mean_eta", []))
        T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full  # positive downward for plotting

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Rest State {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta" if tc.grid_type != "spectral" else "mean_eta_hat_abs",
        heat_key="mean_T" if tc.grid_type != "spectral" else "mean_T_hat_abs",
        salt_key="mean_S" if tc.grid_type != "spectral" else "mean_S_hat_abs",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Barotropic Wave
# ===========================================================================

def run_barotropic_wave(tc: TestCase, output_dir: Path, days: float
                        ) -> tuple[str, float, str]:
    """Barotropic gravity wave: Gaussian SSH perturbation propagation."""
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state(tc, grid, z_coord)
    state = _add_barotropic_wave_perturbation(
        state, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Barotropic Wave ({tc.grid_type})", total_days=days)

    if tc.grid_type == "spectral":
        eta_key = "max_eta_hat_abs"
    else:
        eta_key = "max_abs_eta"
    eta_max = diag[eta_key][-1] if diag.get(eta_key) else 0
    notes = f"max|eta|={eta_max:.4f} m"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Barotropic Wave {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta" if tc.grid_type != "spectral" else "mean_eta_hat_abs",
        heat_key="mean_T" if tc.grid_type != "spectral" else "mean_T_hat_abs",
        salt_key="mean_S" if tc.grid_type != "spectral" else "mean_S_hat_abs",
        scalar_units={"mean_eta": "m", "max_abs_eta": "m",
                      "mean_T": "degC", "mean_S": "PSU"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Wind-Driven Gyre
# ===========================================================================

def run_wind_gyre(tc: TestCase, output_dir: Path, days: float
                  ) -> tuple[str, float, str]:
    """Wind-driven double-gyre circulation."""
    if tc.grid_type == "spectral":
        raise NotImplementedError(
            "Wind-driven gyre not implemented for spectral grid")

    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    # Create rest state first (needed for grids without dedicated gyre init)
    rest = _create_rest_state(tc, grid, z_coord)
    state = _add_wind_gyre_forcing(rest, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Wind Gyre ({tc.grid_type})", total_days=days)

    if tc.grid_type == "mpas":
        speed_key = "max_abs_u"
    else:
        speed_key = "max_speed"
    max_speed = diag[speed_key][-1] if diag.get(speed_key) else 0
    eta_drift = _compute_drift(diag.get("mean_eta", []))
    notes = f"max speed={max_speed:.4f} m/s, eta drift={eta_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})

    field_specs = [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
    ]
    if tc.grid_type != "mpas":
        field_specs.append(("speed_sfc", "Surface speed (m/s)", "magma"))

    _save_case_diagnostics(
        output_dir, f"Wind Gyre {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=field_specs,
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Baroclinic Adjustment
# ===========================================================================

def run_baroclinic(tc: TestCase, output_dir: Path, days: float
                   ) -> tuple[str, float, str]:
    """Baroclinic adjustment: meridional temperature front relaxation."""
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state(tc, grid, z_coord)
    state = _add_baroclinic_perturbation(
        state, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Baroclinic ({tc.grid_type})", total_days=days)

    if tc.grid_type == "spectral":
        T_drift = _compute_drift(diag.get("mean_T_hat_abs", []))
    else:
        T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Baroclinic Adj {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("SSS", "SSS (PSU)", "YlGnBu"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta" if tc.grid_type != "spectral" else "mean_eta_hat_abs",
        heat_key="mean_T" if tc.grid_type != "spectral" else "mean_T_hat_abs",
        salt_key="mean_S" if tc.grid_type != "spectral" else "mean_S_hat_abs",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Phillips Two-Layer
# ===========================================================================

def _add_phillips_perturbation(state, grid_type: str, grid, z_coord):
    """Set up Phillips two-layer initial conditions: jet + SSH perturbation."""
    from legoesm.core.field import Field

    if grid_type == "spectral":
        from legoesm.grids.gaussian import (
            sh_analysis, sh_analysis_3d, sh_synthesis_3d,
            sh_analysis_oc2_3d, sh_analysis_dmu_3d)
        lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='xy')
        mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)

        # Temperature with land mask (consistent with FV grids)
        T_hat = state.T_hat.data
        T_grid = np.array(sh_synthesis_3d(grid, T_hat), dtype=np.float64)
        nlev = T_grid.shape[-1]
        T_grid[..., 0] = (16.0 - 10.0 * np.sin(np.radians(lat_2d)) ** 2) * mask
        if nlev > 1:
            T_grid[..., 1] = (8.0 - 4.0 * np.sin(np.radians(lat_2d)) ** 2) * mask
        new_T_hat = sh_analysis_3d(grid, jnp.array(T_grid))

        # Zonal jet -> convert to vorticity/divergence
        cos_lat = np.asarray(grid.cos_lat[:, None], dtype=np.float64)
        u_jet = 0.30 * np.exp(-((lat_2d - 45.0) / 14.0) ** 2) * mask
        u_grid = np.zeros(T_grid.shape, dtype=np.float64)
        u_grid[..., 0] = u_jet
        if nlev > 1:
            u_grid[..., 1] = -0.20 * u_jet
        u_cos = jnp.array(u_grid * cos_lat[..., None])
        v_cos = jnp.zeros_like(u_cos)
        a = grid.radius
        im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        one_over_a = 1.0 / a
        vor_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
                   + one_over_a * sh_analysis_dmu_3d(grid, u_cos))
        div_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
                   - one_over_a * sh_analysis_dmu_3d(grid, v_cos))

        # SSH perturbation with mask
        eta_pert = 0.05 * np.sin(3.0 * np.radians(lon_2d)) * np.cos(
            2.0 * np.radians(lat_2d)) * mask
        eta_pert -= np.mean(eta_pert)
        eta_hat = state.eta_hat.data + sh_analysis(grid, jnp.array(eta_pert))

        return state._replace(
            vor_hat=Field(vor_hat),
            div_hat=Field(div_hat),
            T_hat=Field(new_T_hat),
            eta_hat=Field(eta_hat))

    else:
        # FV grids (cube, latlon, mpas)
        if grid_type == "mpas":
            lat_rad = np.asarray(grid.latCell, dtype=np.float64)
            lon_rad = np.asarray(grid.lonCell, dtype=np.float64)
            area = np.asarray(grid.areaCell, dtype=np.float64)
        else:
            lat_rad = np.asarray(grid.lat, dtype=np.float64)
            lon_rad = np.asarray(grid.lon, dtype=np.float64)
            area = np.asarray(grid.area, dtype=np.float64)

        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        T_data = np.array(state.T.data, dtype=np.float64, copy=True)
        nlev = T_data.shape[-1]

        # Broadcast lat to match spatial dimensions of mask
        lat_deg_arr = lat_rad * 180 / np.pi
        if lat_deg_arr.ndim < mask.ndim:
            # latlon grid: lat is (n_lat,), mask is (n_lat, n_lon)
            lat_deg_arr = np.broadcast_to(
                lat_deg_arr.reshape(lat_deg_arr.shape + (1,) * (mask.ndim - lat_deg_arr.ndim)),
                mask.shape)
            lat_rad_bc = np.broadcast_to(
                lat_rad.reshape(lat_rad.shape + (1,) * (mask.ndim - lat_rad.ndim)),
                mask.shape)
            lon_rad_bc = np.broadcast_to(
                lon_rad.reshape((1,) * (mask.ndim - lon_rad.ndim) + lon_rad.shape),
                mask.shape)
        else:
            lat_rad_bc = lat_rad
            lon_rad_bc = lon_rad

        # Zonal jet
        u_data = np.array(state.u.data, dtype=np.float64, copy=True)
        if grid_type == "mpas":
            # MPAS: u on edges, compute jet on edge latitudes
            lat_edge_deg = np.asarray(grid.latEdge, dtype=np.float64) * 180 / np.pi
            u_jet_edge = 0.30 * np.exp(-((lat_edge_deg - 45.0) / 14.0) ** 2)
            u_data[..., 0] = u_jet_edge
            if nlev > 1:
                u_data[..., 1] = -0.20 * u_jet_edge
        else:
            u_jet = 0.30 * np.exp(-((lat_deg_arr - 45.0) / 14.0) ** 2) * mask
            u_data[..., 0] = u_jet
            if nlev > 1:
                u_data[..., 1] = -0.20 * u_jet

        # Target temperatures
        T_data[..., 0] = (16.0 - 10.0 * np.sin(lat_rad_bc) ** 2) * mask
        if nlev > 1:
            T_data[..., 1] = (8.0 - 4.0 * np.sin(lat_rad_bc) ** 2) * mask

        # SSH perturbation
        eta_seed = 0.05 * np.sin(3.0 * lon_rad_bc) * np.cos(2.0 * lat_rad_bc) * mask
        area_w = mask * area
        eta_seed -= np.sum(eta_seed * area_w) / np.maximum(np.sum(area_w), 1.0)
        new_eta = state.eta.data + jnp.array(eta_seed)

        state = state._replace(
            u=Field(jnp.array(u_data)),
            T=Field(jnp.array(T_data)),
            eta=Field(new_eta))
        if hasattr(state, "v"):
            v_data = np.array(state.v.data, dtype=np.float64, copy=True)
            v_data[..., 0] *= 0  # start with no meridional flow
            if nlev > 1:
                v_data[..., 1] *= 0
            state = state._replace(v=Field(jnp.array(v_data)))
        return state


def run_phillips_two_layer(tc: TestCase, output_dir: Path, days: float
                           ) -> tuple[str, float, str]:
    """Phillips two-layer baroclinic test with zonal-mean relaxation."""
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=2, H_max=3500.0))
    state = _create_rest_state(tc, grid, z_coord, H_max=3500.0)
    state = _add_phillips_perturbation(state, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    # Relaxation forcing toward target temperature profiles
    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            sh_analysis, sh_analysis_3d, sh_synthesis_3d)
        lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        _, lat_2d = np.meshgrid(lon, lat, indexing='xy')
        T_star_upper = 16.0 - 10.0 * np.sin(np.radians(lat_2d)) ** 2
        T_star_lower = 8.0 - 4.0 * np.sin(np.radians(lat_2d)) ** 2
        T_star_3d = np.stack([T_star_upper, T_star_lower], axis=-1)
        T_star_hat = sh_analysis_3d(grid, jnp.array(T_star_3d))
        tau_relax = 15.0 * 86400.0
        drag_factor = float(jnp.exp(-dt / (25.0 * 86400.0)))

        def forcing_fn(s, dt_):
            from legoesm.core.field import Field
            T_hat = s.T_hat.data
            dT_hat = -(T_hat - T_star_hat) / tau_relax
            new_T_hat = T_hat + dt_ * dT_hat
            # Spectral ocean uses vor_hat/div_hat, not u_hat/v_hat
            new_vor_hat = s.vor_hat.data * drag_factor
            new_div_hat = s.div_hat.data * drag_factor
            return s._replace(
                T_hat=Field(new_T_hat),
                vor_hat=Field(new_vor_hat),
                div_hat=Field(new_div_hat))

    else:
        if tc.grid_type == "mpas":
            lat_rad = np.asarray(grid.latCell, dtype=np.float64)
        elif tc.grid_type == "latlon":
            # lat is 1D (n_lat,) — broadcast to (n_lat, n_lon)
            lat_1d = np.asarray(grid.lat, dtype=np.float64)
            n_lon = grid.n_lon if hasattr(grid, "n_lon") else grid.lon.shape[0]
            lat_rad = np.broadcast_to(lat_1d[:, None], (lat_1d.size, n_lon))
        else:
            lat_rad = np.asarray(grid.lat, dtype=np.float64)
        T_star_upper = jnp.array(16.0 - 10.0 * np.sin(lat_rad) ** 2)
        T_star_lower = jnp.array(8.0 - 4.0 * np.sin(lat_rad) ** 2)
        tau_relax = 15.0 * 86400.0
        drag_factor = float(jnp.exp(-dt / (25.0 * 86400.0)))

        _is_mpas = (tc.grid_type == "mpas")

        def forcing_fn(s, dt_):
            from legoesm.core.field import Field
            T_data = s.T.data
            mask = s.land_mask.data
            dT0 = -(T_data[..., 0] - T_star_upper) / tau_relax * mask
            dT1 = -(T_data[..., 1] - T_star_lower) / tau_relax * mask
            T_new = T_data.at[..., 0].set(T_data[..., 0] + dt_ * dT0)
            T_new = T_new.at[..., 1].set(T_data[..., 1] + dt_ * dT1)
            # For MPAS, u is on edges — can't multiply by cell mask,
            # and there is no separate v field.
            u_new = s.u.data * drag_factor
            if _is_mpas:
                return s._replace(
                    T=Field(T_new),
                    u=Field(u_new))
            mask_3d = mask[..., jnp.newaxis]
            u_new = u_new * mask_3d
            v_new = s.v.data * drag_factor * mask_3d
            return s._replace(
                T=Field(T_new),
                u=Field(u_new),
                v=Field(v_new))

    def step_fn(s, dt_):
        s = model.step(s, dt_)
        return forcing_fn(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Phillips 2-layer ({tc.grid_type})", total_days=days)

    if tc.grid_type == "spectral":
        T_drift = _compute_drift(diag.get("mean_T_hat_abs", []))
    else:
        T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"Phillips 2-Layer {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta" if tc.grid_type != "spectral" else "mean_eta_hat_abs",
        heat_key="mean_T" if tc.grid_type != "spectral" else "mean_T_hat_abs",
        salt_key="mean_S" if tc.grid_type != "spectral" else "mean_S_hat_abs",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner dispatch
# ===========================================================================

RUNNERS: dict[str, Callable] = {
    "rest_state": run_rest_state,
    "barotropic_wave": run_barotropic_wave,
    "wind_gyre": run_wind_gyre,
    "baroclinic": run_baroclinic,
    "phillips_two_layer": run_phillips_two_layer,
}


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ocean test matrix for legoESM ocean dynamical cores.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--only", type=str, default="all",
        help="Run only cases matching this name "
             "(e.g. rest_state, barotropic_wave, wind_gyre, baroclinic)")
    p.add_argument(
        "--grid", type=str, default="all",
        choices=["cubed_sphere", "latlon", "mpas", "spectral", "all"],
        help="Run only a specific grid type (default: all)")
    p.add_argument(
        "--resolution", type=str, default=None,
        help="Override baseline resolution (e.g. C48, 72x144, ico4, T42)")
    p.add_argument(
        "--levels", type=int, default=DEFAULT_NLEV,
        help=f"Number of vertical levels (default: {DEFAULT_NLEV})")
    p.add_argument(
        "--dt", type=float, default=DEFAULT_DT,
        help=f"Time step in seconds (default: {DEFAULT_DT})")
    p.add_argument(
        "--output", "-o", type=str, default="results/ocean",
        help="Base output directory (default: results/ocean)")
    p.add_argument(
        "--quick", action="store_true",
        help="Use shorter durations for quick verification")
    p.add_argument(
        "--list", action="store_true",
        help="List all test cases and exit")
    return p


def filter_tests(tests: list[TestCase], args) -> list[TestCase]:
    filtered = tests
    if args.only != "all":
        filtered = [t for t in filtered if args.only in t.case]
    if args.grid != "all":
        filtered = [t for t in filtered if t.grid_type == args.grid]
    return filtered


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Override global defaults if specified
    global DEFAULT_NLEV, DEFAULT_DT
    DEFAULT_NLEV = args.levels
    DEFAULT_DT = args.dt

    tests = filter_tests(TEST_MATRIX, args)

    if args.list:
        print(f"{'#':>3}  {'Case':<22}  {'Grid':<14}  "
              f"{'Resolution':<10}  {'Days':>8}  {'Quick':>8}")
        print("-" * 78)
        for i, tc in enumerate(tests, 1):
            print(f"{i:3d}  {tc.case:<22}  {tc.grid_type:<14}  "
                  f"{tc.resolution:<10}  {tc.duration_days:8.2f}  "
                  f"{tc.quick_days:8.2f}")
        print(f"\nTotal: {len(tests)} test cases "
              f"(of {len(TEST_MATRIX)} in full matrix)")
        return

    if not tests:
        print("No tests match the given filters.")
        return

    if args.resolution:
        tests = [TestCase(
            t.case, t.grid_type, args.resolution,
            t.duration_days, t.quick_days, t.run_kwargs)
            for t in tests]

    output_base = Path(args.output)

    print("=" * 78)
    print("  legoESM Ocean Test Matrix")
    print("=" * 78)
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  X64:        {jax.config.jax_enable_x64}")
    print(f"  Devices:    {jax.devices()}")
    print(f"  Output:     {output_base}")
    print(f"  Levels:     {DEFAULT_NLEV}")
    print(f"  dt:         {DEFAULT_DT}s")
    print(f"  Quick mode: {args.quick}")
    print(f"  Tests:      {len(tests)} / {len(TEST_MATRIX)}")
    print("=" * 78)
    print()

    t_start_all = time.time()

    for i, tc in enumerate(tests, 1):
        days = tc.quick_days if args.quick else tc.duration_days
        out_dir = output_base / tc.output_path

        label = f"{tc.case}/{tc.grid_type}/{tc.resolution}"
        print(f"\n[{i}/{len(tests)}] {label} ({days:.4g} days)")
        print("-" * 60)

        runner = RUNNERS.get(tc.case)
        if runner is None:
            record(tc, "ERROR", 0, f"Unknown runner for case: {tc.case}")
            continue

        try:
            status, wall, notes = runner(tc, out_dir, days)
            record(tc, status, wall, notes)
        except NotImplementedError as e:
            record(tc, "SKIP", 0, str(e)[:120])
        except Exception as e:
            record(tc, "ERROR", 0, str(e)[:120])
            traceback.print_exc()
        finally:
            _ensure_required_artifacts(out_dir)

    total_wall = time.time() - t_start_all

    # --- Summary ---
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'Status':6}  {'Grid':<14}  {'Case':<22}  "
          f"{'Resolution':<10}  {'Time':>8}  Notes")
    print("-" * 90)

    n_pass = n_fail = n_error = n_skip = 0
    for r in ALL_RESULTS:
        icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!",
                "SKIP": "--"}[r["status"]]
        print(f"  {icon}{r['status']:5}  {r['grid']:<14}  "
              f"{r['test']:<22}  {r['resolution']:<10}  "
              f"{r['wall_time']:7.1f}s  {r['notes']}")
        if r["status"] == "PASS":
            n_pass += 1
        elif r["status"] == "FAIL":
            n_fail += 1
        elif r["status"] == "SKIP":
            n_skip += 1
        else:
            n_error += 1

    print("-" * 90)
    print(f"  Total: {len(ALL_RESULTS)} tests | "
          f"PASS: {n_pass} | FAIL: {n_fail} | SKIP: {n_skip} | "
          f"ERROR: {n_error} | Wall: {total_wall:.1f}s "
          f"({total_wall / 60:.1f} min)")
    print("=" * 78)

    # Save summary
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "summary.json", "w") as f:
        json.dump({
            "results": ALL_RESULTS, "total_wall_time": total_wall,
            "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
            "n_error": n_error, "quick_mode": args.quick,
            "levels": DEFAULT_NLEV, "dt": DEFAULT_DT,
        }, f, indent=2)
    with open(output_base / "summary.txt", "w") as f:
        f.write("legoESM Ocean Test Matrix Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(ALL_RESULTS)} tests | "
                f"PASS: {n_pass} | FAIL: {n_fail} | SKIP: {n_skip} | "
                f"ERROR: {n_error}\n")
        f.write(f"Wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)\n")
        f.write(f"Levels: {DEFAULT_NLEV}, dt: {DEFAULT_DT}s\n")
        f.write(f"Quick mode: {args.quick}\n\n")
        for r in ALL_RESULTS:
            f.write(f"{r['status']:5}  {r['grid']:<14}  "
                    f"{r['test']:<22}  {r['resolution']:<10}  "
                    f"{r['wall_time']:7.1f}s  {r['notes']}\n")

    print(f"\n  Summary: {output_base / 'summary.json'}")

    if n_fail > 0 or n_error > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
