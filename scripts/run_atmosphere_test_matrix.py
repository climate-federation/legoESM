#!/usr/bin/env python
"""Atmosphere test matrix: organized test runner for legoESM dynamical cores.

Runs shallow-water, hydrostatic, and non-hydrostatic tests across
cubed-sphere, lat-lon, and icosahedral grids at ~1.25 degree resolution.

Output structure:
    results/atmosphere/<equation_set>/<case>/<grid_type>/<resolution>/<vertical_coord>/

Each case folder contains:
    - mean_timeseries.csv / .png      (domain-averaged scalar time series)
    - conservation_timeseries.csv/.png (mass & energy drift)
    - field_snapshots.png              (2D field maps at selected times)
    - snapshots_<field>.png            (per-field snapshot evolution)
    - snapshots_native.npz             (snapshot arrays in native grid coords)
    - snapshots_latlon.npz             (snapshot arrays regridded to 181x360 lat-lon)
    - vertical_profiles.png            (vertical profile evolution)
    - latitude_vertical_cross_sections.png
    - longitude_vertical_cross_sections.png
    - results.txt                      (run metadata)

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py --quick
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py --only sw
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py --only hydro --grid cubed_sphere
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py --radiation rrtmgp
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

# ~1.25 degree resolutions per grid type
GRID_RESOLUTIONS: dict[str, str] = {
    "cubed_sphere": "C36",
    "latlon": "72x144",
    "icosahedral": "ico5",
    "spectral": "T21",
}

GRID_TYPES = list(GRID_RESOLUTIONS.keys())

DEFAULT_NLEV = 40


# ===========================================================================
# TestCase dataclass
# ===========================================================================

@dataclass
class TestCase:
    """A single test case in the matrix."""
    equation_set: str       # shallow_water, hydrostatic, nonhydrostatic
    case: str               # williamson2, held_suarez, dcmip_tc1, etc.
    grid_type: str          # cubed_sphere, latlon, icosahedral, spectral
    resolution: str         # C36, 72x144, ico5
    vertical_coord: str     # none, sigma, hybrid, height
    duration_days: float
    quick_days: float
    run_kwargs: dict = field(default_factory=dict)

    @property
    def output_path(self) -> str:
        parts = [self.equation_set, self.case, self.grid_type, self.resolution]
        if self.vertical_coord != "none":
            parts.append(self.vertical_coord)
        return "/".join(parts)


# ===========================================================================
# Test matrix generation
# ===========================================================================

def _build_test_matrix() -> list[TestCase]:
    """Generate the full test matrix from grid × case × vertical coord."""
    matrix: list[TestCase] = []
    res = GRID_RESOLUTIONS

    # --- Shallow water: all grids, no vertical coord ---
    for g in GRID_TYPES:
        for case, dur, quick, kw in [
            ("williamson2", 5, 1, {"test_num": 2}),
            ("williamson5", 15, 1, {"test_num": 5}),
        ]:
            matrix.append(TestCase(
                "shallow_water", case, g, res[g], "none", dur, quick, dict(kw)))

    # --- Hydrostatic: all grids, sigma + hybrid ---
    for g in GRID_TYPES:
        for vert in ["sigma", "hybrid"]:
            matrix.append(TestCase(
                "hydrostatic", "held_suarez", g, res[g], vert, 200, 30))
            matrix.append(TestCase(
                "hydrostatic", "baroclinic", g, res[g], vert, 10, 2))
        # DCMIP transport: sigma only
        for tn in [11, 12, 13]:
            matrix.append(TestCase(
                "hydrostatic", f"dcmip_transport_{tn}", g, res[g], "sigma",
                12, 1, {"test_num": tn}))
        # AMIP: hybrid only
        matrix.append(TestCase(
            "hydrostatic", "amip", g, res[g], "hybrid", 365, 30))

    # --- Non-hydrostatic: cubed-sphere, icosahedral, and spectral only ---
    # (lat-lon NH dycore does not exist)
    nh_grids = ["cubed_sphere", "icosahedral", "spectral"]
    for g in nh_grids:
        for case, dur, quick, kw in [
            ("dcmip_tc1", 3 / 24, 0.5 / 24, {"test_case": "tc1"}),
            ("dcmip_tc2", 6 / 24, 5 / (24 * 60), {"test_case": "tc2a"}),
            ("dcmip_tc3", 2 / 24, 5 / (24 * 60), {"test_case": "tc3"}),
        ]:
            matrix.append(TestCase(
                "nonhydrostatic", case, g, res[g], "height", dur, quick,
                dict(kw)))

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
        "equation_set": tc.equation_set, "resolution": tc.resolution,
        "vertical_coord": tc.vertical_coord, "status": status,
        "wall_time": wall_time, "notes": notes,
    })
    label = f"{tc.equation_set}/{tc.case}/{tc.grid_type}"
    print(f"  {icon} {status:5s} | {label:<55s} | {wall_time:7.1f}s | {notes}")


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


# ---------------------------------------------------------------------------
# Hyperdiffusion helpers
# ---------------------------------------------------------------------------

def _hyperdiff_cube(n: int, ref_n: int = 48, ref_coeff: float = 1e16) -> float:
    return ref_coeff * (ref_n / n) ** 4


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    """Scale second-order divergence damping for cubed-sphere (FV3-style)."""
    return ref_coeff * (ref_n / n) ** 2


def _hyperdiff_latlon(n_lat: int, ref_n: int = 64, ref_coeff: float = 2e16) -> float:
    return ref_coeff * (ref_n / n_lat) ** 4


def _div_damp_latlon(n_lat: int, ref_n: int = 64, ref_coeff: float = 5e6) -> float:
    """Scale second-order divergence damping coefficient with grid spacing."""
    return ref_coeff * (ref_n / n_lat) ** 2


def _hyperdiff_ico(mesh) -> float:
    """Biharmonic hyperdiffusion for icosahedral mesh.

    Uses the minimum of dcEdge and dvEdge because the TRiSK vector
    Laplacian has eigenvalues governed by min(dc, dv), not the mean
    cell spacing.  A 48-hour e-folding time keeps the coefficient
    conservative.
    """
    dx = float(jnp.minimum(jnp.min(mesh.dcEdge), jnp.min(mesh.dvEdge)))
    return dx ** 4 / (48.0 * 3600.0)


def _laplacian_visc_cube(n: int, frac: float = 0.05) -> float:
    """Laplacian viscosity A_h = frac * c_gw * dx for cubed-sphere.

    A modest Laplacian viscosity (frac=0.05) is needed alongside
    biharmonic hyperdiffusion to damp grid-scale energy that the C-D
    grid staggering does not fully resolve.
    """
    import math
    from legoesm import constants
    dx = math.pi * constants.R_earth / (2.0 * n)
    c_gw = math.sqrt(constants.R_d * 300.0)
    return frac * c_gw * dx


def _laplacian_visc_latlon(n_lat: int, frac: float = 0.1) -> float:
    """Laplacian viscosity A_h = frac * c_gw * dy for lat-lon grid."""
    import math
    from legoesm import constants
    dy = math.pi * constants.R_earth / n_lat
    c_gw = math.sqrt(constants.R_d * 300.0)
    return frac * c_gw * dy


def _laplacian_visc_ico(mesh, frac: float = 0.1) -> float:
    """Laplacian viscosity A_h = frac * c_gw * dx for icosahedral grid."""
    import math
    from legoesm import constants
    dx_mean = float(jnp.sqrt(4.0 * jnp.pi * mesh.radius ** 2 / mesh.nCells))
    c_gw = math.sqrt(constants.R_d * 300.0)
    return frac * c_gw * dx_mean


# ---------------------------------------------------------------------------
# Vertical coordinate creation
# ---------------------------------------------------------------------------

def _create_vertical(nlev: int, vertical_coord: str):
    """Create vertical coordinate (sigma or hybrid)."""
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    if vertical_coord == "hybrid":
        return standard_hybrid_levels(nlev)
    return create_sigma_coordinate(nlev)


# ---------------------------------------------------------------------------
# RRTMGP physics factory
# ---------------------------------------------------------------------------

def _make_rrtmgp_physics(model_type: str, dt: float):
    """Create RRTMGP-based physics function.

    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe", "mpas".
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

    phys_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="rrtmgp"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    return make_physics(phys_cfg, model_type=model_type, dt=dt)


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
    blowup_threshold: float = 1000.0,
    n_snaps: int = 10,
) -> tuple[Any, dict, dict, float, bool]:
    """Run time loop with diagnostics.

    Returns (final_state, snapshots, diag, wall_time, ok).
    """
    snap_targets = _snapshot_steps(n_steps, n_snaps)
    snapshots: dict[int, dict[str, np.ndarray]] = {0: extract_fn(state)}
    # Record step-0 diagnostics so conservation plots have the true
    # initial value (important for perturbation variables starting at 0).
    scalars_0 = scalar_fn(state)
    diag: dict[str, list] = {"times": [0.0], "steps": [0]}
    for k, v in scalars_0.items():
        diag.setdefault(k, []).append(v)

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


def _interp_gaussian_to_latlon(field: np.ndarray, lat_gauss_deg: np.ndarray,
                               n_lat_out: int = 181) -> np.ndarray:
    """Interpolate a Gaussian-grid latitude axis to a regular lat-lon grid.

    Uses linear interpolation along the latitude dimension so that
    cross-section plots have smooth rendering instead of blocky stripes.

    Parameters
    ----------
    field : (n_lat_gauss, ...) — data on Gaussian latitudes
    lat_gauss_deg : (n_lat_gauss,) — Gaussian latitudes in degrees
    n_lat_out : int — number of output latitudes (default 181 for 1-deg)

    Returns
    -------
    out : (n_lat_out, ...) — interpolated to regular latitudes
    """
    from scipy.interpolate import interp1d
    lat_out = np.linspace(-90.0, 90.0, n_lat_out)
    lat_g = np.asarray(lat_gauss_deg, dtype=np.float64).ravel()
    # interp1d along axis 0
    f = interp1d(lat_g, field, axis=0, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    return f(lat_out)


def _get_cs_weights(n: int, n_lat: int = 181, n_lon: int = 360):
    """Get (or compute and cache) face-aware bilinear CS→latlon weights."""
    from legoesm.grids.regridding import get_cubedsphere_to_latlon_weights
    return get_cubedsphere_to_latlon_weights(n, n_lon=n_lon, n_lat=n_lat)


def _regrid_2d(field: np.ndarray, lon_deg: np.ndarray, lat_deg: np.ndarray,
               coord_kind: str) -> np.ndarray:
    """Regrid a 2D field to (181, 360) lat-lon."""
    arr = np.asarray(field, dtype=np.float64)
    if coord_kind == "latlon":
        return arr
    if coord_kind == "gaussian":
        # Gaussian grid: interpolate lat axis to regular spacing
        lat_gauss = np.asarray(lat_deg, dtype=np.float64).ravel()
        return _interp_gaussian_to_latlon(arr, lat_gauss)
    if coord_kind == "icosa":
        # Some MPAS extractors already return regular lat-lon fields for 2D
        # quantities because edge- and cell-based variables need different
        # source coordinates. Do not remap those arrays again.
        if arr.shape == (181, 360):
            return arr
        idxs, w = _build_latlon_weights(lon_deg, lat_deg, k=20)
        return _apply_weights(arr.ravel(), idxs, w, 181, 360)
    # Cubed-sphere: use face-aware bilinear interpolation.
    # If a field was already regridded (e.g. wind from corner-based remap),
    # its shape is (n_lat, n_lon) — return it as-is.
    from legoesm.grids.regridding import apply_cubedsphere_to_latlon
    if arr.ndim == 2 and arr.shape[0] != 6:
        return arr
    if arr.ndim >= 3 and arr.shape[0] == 6:
        n = arr.shape[1]
    else:
        n = int(round(np.sqrt(arr.size / 6)))
        arr = arr.reshape(6, n, n)
    w = _get_cs_weights(n)
    return apply_cubedsphere_to_latlon(arr, w)


def _regrid_3d_level(field_3d: np.ndarray, lon_deg: np.ndarray,
                     lat_deg: np.ndarray, coord_kind: str) -> np.ndarray:
    """Regrid a 3D field (*, nlev) to (n_lat, n_lon, nlev)."""
    arr = np.asarray(field_3d, dtype=np.float64)
    if coord_kind == "latlon":
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr
    if coord_kind == "gaussian":
        if arr.ndim == 2:
            arr = arr[..., None]
        # Interpolate the Gaussian latitude axis to regular 1-deg spacing
        lat_gauss = np.asarray(lat_deg, dtype=np.float64).ravel()
        return _interp_gaussian_to_latlon(arr, lat_gauss)
    if coord_kind == "icosa":
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim >= 3 and arr.shape[:2] == (181, 360):
            return arr
        nlev = arr.shape[-1]
        idxs, w = _build_latlon_weights(lon_deg, lat_deg, k=20)
        flat = arr.reshape(-1, nlev)
        out = np.full((181, 360, nlev), np.nan, dtype=np.float64)
        for k in range(nlev):
            out[..., k] = _apply_weights(flat[:, k], idxs, w, 181, 360)
        return out
    # Cubed-sphere: use face-aware bilinear interpolation
    from legoesm.grids.regridding import apply_cubedsphere_to_latlon_3d
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim >= 4 and arr.shape[0] == 6:
        n = arr.shape[1]
    else:
        nlev = arr.shape[-1]
        n = int(round(np.sqrt(arr.size / (6 * nlev))))
        arr = arr.reshape(6, n, n, nlev)
    w = _get_cs_weights(n)
    return apply_cubedsphere_to_latlon_3d(arr, w)


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
                       mass_key: str, energy_key: str):
    mass_vals = diag.get(mass_key, [])
    energy_vals = diag.get(energy_key, [])
    times = diag.get("times", [])
    if not mass_vals or not energy_vals or not times:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    mass = np.array(mass_vals, dtype=np.float64)
    energy = np.array(energy_vals, dtype=np.float64)
    t = np.array(times, dtype=np.float64)

    # Decide between relative-drift and absolute-value mode.
    # For quantities with a large initial value (e.g. total mass,
    # mean height), normalise by the initial value to show fractional
    # drift.  For perturbation variables that start near zero
    # (e.g. rho_prime, theta_prime in NH), plot absolute values
    # directly since relative drift is meaningless.
    _PERTURBATION_THRESHOLD = 1e-10  # initial value below this ⇒ perturbation mode
    mass_is_perturbation = abs(mass[0]) < _PERTURBATION_THRESHOLD
    energy_is_perturbation = abs(energy[0]) < _PERTURBATION_THRESHOLD

    if mass_is_perturbation:
        mass_plot = mass
        mass_ylabel = f"{mass_key} (absolute)"
    else:
        mass_plot = (mass - mass[0]) / abs(mass[0])
        mass_ylabel = f"Relative {mass_key} drift"

    if energy_is_perturbation:
        energy_plot = energy
        energy_ylabel = f"{energy_key} (absolute)"
    else:
        energy_plot = (energy - energy[0]) / abs(energy[0])
        energy_ylabel = f"Relative {energy_key} drift"

    with open(output_dir / "conservation_timeseries.csv", "w") as f:
        f.write("time_days,mass_proxy,energy_proxy,mass_plot,energy_plot\n")
        for i in range(t.size):
            f.write(f"{t[i]:.8f},{mass[i]:.12e},{energy[i]:.12e},"
                    f"{mass_plot[i]:.12e},{energy_plot[i]:.12e}\n")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, mass_plot, lw=1.5)
    axes[0].axhline(0, color="0.3", ls="--", lw=0.8)
    axes[0].set_ylabel(mass_ylabel)
    axes[0].set_title(f"{mass_key} conservation")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, energy_plot, lw=1.5, color="tab:red")
    axes[1].axhline(0, color="0.3", ls="--", lw=0.8)
    axes[1].set_ylabel(energy_ylabel)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title(f"{energy_key} conservation")
    axes[1].grid(True, alpha=0.25)
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

        # Compute shared color limits across all panels
        all_vals = np.concatenate([
            _regrid_2d(
                np.asarray(snapshots[s][field_key], dtype=np.float64),
                lon_deg, lat_deg, coord_kind,
            ).ravel() for s in steps
        ])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) > 0:
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
        else:
            vmin, vmax = 0.0, 1.0

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
                extent=[-180, 180, -90, 90], vmin=vmin, vmax=vmax)
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
        # Pre-compute all sections to determine shared color limits
        sections = []
        for step in valid_steps:
            f3d = np.asarray(snapshots[step][field_3d_key], dtype=np.float64)
            ll = _regrid_3d_level(f3d, lon_deg, lat_deg, coord_kind)
            section = np.nanmean(ll, axis=mean_axis)
            sections.append(_fill_nan_section(section))

        all_vals = np.concatenate([s.ravel() for s in sections])
        all_finite = all_vals[np.isfinite(all_vals)]
        if all_finite.size > 0:
            vmin, vmax = float(all_finite.min()), float(all_finite.max())
        else:
            vmin, vmax = 0.0, 1.0

        nc = len(valid_steps)
        fig, axes_arr = plt.subplots(
            1, nc, figsize=(4.5 * nc, 5), sharey=True)
        if nc == 1:
            axes_arr = [axes_arr]
        im = None

        for ax, step, section in zip(axes_arr, valid_steps, sections):
            im = ax.imshow(
                section.T, origin="lower", aspect="auto", cmap="RdBu_r",
                vmin=vmin, vmax=vmax,
                extent=[axis_vals[0], axis_vals[-1],
                        float(levels[0]), float(levels[-1])])
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
                   level_label: str, invert_y: bool = True):
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
    if invert_y:
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
    """Save snapshot field arrays as NPZ files in both native grid and lat-lon.

    Produces:
        snapshots_native.npz   – raw arrays keyed as ``{field}_step{step}``
        snapshots_latlon.npz   – regridded to (181, 360) regular lat-lon,
                                 same key convention.  For 3-D fields the
                                 shape is (181, 360, nlev).
    Both files also contain ``times_days`` and ``steps`` metadata arrays.
    """
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
            # Fields with "_3d" suffix are (*, nlev) – regrid per level
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
    level_label: str = "Level",
    mass_key: str | None = None,
    energy_key: str | None = None,
    scalar_units: dict[str, str] | None = None,
    invert_levels: bool = True,
):
    """Save all standard diagnostic outputs for a test case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    su = scalar_units or {}

    _save_timeseries_csv(output_dir, diag, dt)
    _save_timeseries_plot(output_dir, case_name, diag, su)

    if mass_key and energy_key:
        _save_conservation(output_dir, case_name, diag, mass_key, energy_key)

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
            level_values, level_label, invert_levels)


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
# Shared field extraction helpers
# ===========================================================================

def _extract_hydro_cube_latlon(s, cos_angle=None, sin_angle=None):
    """Extract hydrostatic snapshot fields for cubed-sphere or lat-lon.

    For cubed-sphere grids, pass cos_angle and sin_angle to rotate
    face-local (u, v) to geographic (u_east, v_north) coordinates.
    """
    u_sfc = np.asarray(s.u.data[..., -1], dtype=np.float64)
    v_sfc = np.asarray(s.v.data[..., -1], dtype=np.float64)
    if cos_angle is not None:
        ca = np.asarray(cos_angle, dtype=np.float64)
        sa = np.asarray(sin_angle, dtype=np.float64)
        u_sfc, v_sfc = ca * u_sfc - sa * v_sfc, sa * u_sfc + ca * v_sfc
    return {
        "u": u_sfc,
        "v": v_sfc,
        "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
        "p_s": np.asarray(s.p_s.data, dtype=np.float64),
        "T_3d": np.asarray(s.T.data, dtype=np.float64),
    }


def _extract_hydro_mpas(s, mesh, lon_cell, lat_cell):
    """Extract hydrostatic snapshot fields for icosahedral (MPAS).

    Reconstructs cell-centered (u_east, v_north) from edge-normal
    velocities, then regrids all fields to a regular lat-lon grid.
    """
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity

    u_edge = s.u.data
    if u_edge.ndim > 1:
        u_edge = u_edge[:, -1]
    u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
    u_e = np.asarray(u_east, dtype=np.float64)
    v_n = np.asarray(v_north, dtype=np.float64)

    u_ll = _bin_to_latlon(u_e, lon_cell, lat_cell)
    v_ll = _bin_to_latlon(v_n, lon_cell, lat_cell)
    ps_ll = _bin_to_latlon(
        np.asarray(s.p_s.data, dtype=np.float64), lon_cell, lat_cell)

    return {
        "u": u_ll,
        "v": v_ll,
        "wind_speed": np.sqrt(u_ll ** 2 + v_ll ** 2),
        "p_s": ps_ll,
        "T_3d": np.asarray(s.T.data, dtype=np.float64),
    }


# ===========================================================================
# Runner: Shallow Water
# ===========================================================================

def run_shallow_water(tc: TestCase, output_dir: Path, days: float, *,
                      radiation: str = "gray") -> tuple[str, float, str]:
    test_num = tc.run_kwargs["test_num"]

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
            CDGridShallowWaterConfig)
        from tests.test_cases.williamson import (
            williamson_test2, williamson_test5,
            williamson_test2_exact, compute_error_norms)

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        dt = 300.0
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=_hyperdiff_cube(n),
            div_damp=_div_damp_cube(n))
        model = FV3EdgeShallowWaterModel(grid, config)
        cdgrid = model.cdgrid

        # Initialise edge-midpoint D-grid winds analytically.
        sw = williamson_test2(grid) if test_num == 2 else williamson_test5(grid)
        u0 = (2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
              if test_num == 2 else 20.0)
        u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
        u_d = cdgrid.cos_angle_edge_x * u_east_x
        u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
        v_d = -cdgrid.sin_angle_edge_y * u_east_y
        state = FV3EdgeShallowWaterState(
            h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
        model.set_initial_mass(state)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            return (check_finite({"h": s.h, "u_d": s.u_d}),
                    float(jnp.max(jnp.abs(s.u_d))))

        def scalar_fn(s):
            return {
                "mean_height": float(jnp.mean(s.h)),
                "max_wind": float(jnp.max(jnp.abs(s.u_d))),
            }

        # Regrid wind from cell-centre averages of edge-midpoint winds.
        _cs_w = _get_cs_weights(n)

        def extract_fn(s):
            # Average edge-midpoint winds to cell centres, then regrid
            u_cc = 0.5 * (np.asarray(s.u_d, dtype=np.float64)[:, :, :-1]
                          + np.asarray(s.u_d, dtype=np.float64)[:, :, 1:])
            v_cc = 0.5 * (np.asarray(s.v_d, dtype=np.float64)[:, :-1, :]
                          + np.asarray(s.v_d, dtype=np.float64)[:, 1:, :])
            ca = np.asarray(grid.cos_angle, dtype=np.float64)
            sa = np.asarray(grid.sin_angle, dtype=np.float64)
            u_east = ca * u_cc - sa * v_cc
            v_north = sa * u_cc + ca * v_cc
            u_ll = _regrid_2d(u_east, lon_deg, lat_deg, coord_kind)
            v_ll = _regrid_2d(v_north, lon_deg, lat_deg, coord_kind)
            return {"u": u_ll, "v": v_ll,
                    "wind_speed": np.sqrt(u_ll ** 2 + v_ll ** 2),
                    "height": np.asarray(s.h, dtype=np.float64)}

        key_array_fn = lambda s: s.h
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
            FVShallowWaterLatLonModel, FVShallowWaterLatLonConfig)
        from tests.test_cases.williamson_latlon import (
            williamson_test2_latlon, williamson_test5_latlon,
            williamson_test2_exact_latlon, compute_error_norms_latlon)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        dt = 300.0
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=_hyperdiff_latlon(n_lat))
        model = FVShallowWaterLatLonModel(grid, config)
        state = (williamson_test2_latlon(grid) if test_num == 2
                 else williamson_test5_latlon(grid))

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            return (check_finite({"h": s.h.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mean_height": float(jnp.mean(s.h.data)),
                "max_wind": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
            }

        def extract_fn(s):
            u = np.asarray(s.u.data, dtype=np.float64)
            v = np.asarray(s.v.data, dtype=np.float64)
            return {"u": u, "v": v,
                    "wind_speed": np.sqrt(u ** 2 + v ** 2),
                    "height": np.asarray(s.h.data, dtype=np.float64)}

        key_array_fn = lambda s: s.h.data
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.shallow_water_mpas import (
            MPASShallowWaterModel, MPASShallowWaterConfig)
        from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
            williamson_test2_mpas, williamson_test5_mpas,
            compute_error_norms_mpas)
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        dt = 300.0
        config = MPASShallowWaterConfig(nu_del4=_hyperdiff_ico(mesh))
        model = MPASShallowWaterModel(mesh, config)
        init_fns = {2: williamson_test2_mpas, 5: williamson_test5_mpas}
        state = init_fns[test_num](mesh)
        grid = mesh  # for consistent naming

        lon_cell = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_cell = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            return (check_finite({"h": s.h.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mean_height": float(jnp.mean(s.h.data)),
                "max_wind": float(jnp.max(jnp.abs(s.u.data))),
            }

        def extract_fn(s):
            u_e, v_n = reconstruct_cell_velocity(s.u.data, mesh)
            u = np.asarray(u_e, dtype=np.float64)
            v = np.asarray(v_n, dtype=np.float64)
            return {
                "u": _bin_to_latlon(u, lon_cell, lat_cell),
                "v": _bin_to_latlon(v, lon_cell, lat_cell),
                "wind_speed": _bin_to_latlon(
                    np.sqrt(u ** 2 + v ** 2), lon_cell, lat_cell),
                "height": _bin_to_latlon(
                    np.asarray(s.h.data, dtype=np.float64),
                    lon_cell, lat_cell),
            }

        key_array_fn = lambda s: s.h.data
        coord_kind = "latlon"  # already regridded
        lon_deg = np.linspace(-180, 180, 360, endpoint=False)
        lat_deg = np.linspace(-90, 90, 181)

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            create_gaussian_grid, sh_synthesis, uv_from_vordiv)
        from legoesm.atmosphere.dynamics.spectral_sw import (
            SpectralShallowWaterModel, SpectralSWConfig,
            williamson_test2_spectral, williamson_test5_spectral,
        )
        from legoesm import constants

        n_max = int(tc.resolution.replace("T", ""))
        grid = create_gaussian_grid(n_max)
        # CFL-safe dt for explicit SSP-RK3: gravity wave CFL ≈ 0.5
        import math
        _c_gw = math.sqrt(constants.g * 5960.0)  # shallow-water wave speed
        dt = min(600.0, 0.5 * grid.radius / (n_max * _c_gw))
        config = SpectralSWConfig(
            spectral_filter_order=8 if test_num == 5 else 0,
        )
        model = SpectralShallowWaterModel(grid, config)
        state = (williamson_test2_spectral(grid) if test_num == 2
                 else williamson_test5_spectral(grid))
        if test_num == 5:
            state = model.filter_initial_state(state)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            phi = sh_synthesis(grid, s.phi_hat.data)
            # Use wind speed for blowup metric (phi is O(10^4), not comparable)
            vor = sh_synthesis(grid, s.vor_hat.data)
            return (check_finite({"phi": phi}),
                    float(jnp.max(jnp.abs(vor))))

        def scalar_fn(s):
            phi = sh_synthesis(grid, s.phi_hat.data)
            u_cos, v_cos = uv_from_vordiv(
                grid, s.vor_hat.data, s.div_hat.data)
            cos2d = grid.cos_lat[:, None]
            ws = jnp.sqrt((u_cos / cos2d) ** 2 + (v_cos / cos2d) ** 2)
            return {
                "mean_height": float(jnp.mean(phi / constants.g)),
                "max_wind": float(jnp.max(ws)),
            }

        def extract_fn(s):
            phi = np.asarray(sh_synthesis(grid, s.phi_hat.data),
                             dtype=np.float64)
            u_cos, v_cos = uv_from_vordiv(
                grid, s.vor_hat.data, s.div_hat.data)
            cos2d = np.asarray(grid.cos_lat[:, None], dtype=np.float64)
            u = np.asarray(u_cos, dtype=np.float64) / cos2d
            v = np.asarray(v_cos, dtype=np.float64) / cos2d
            return {
                "u": u,
                "v": v,
                "wind_speed": np.sqrt(u ** 2 + v ** 2),
                "height": phi / float(constants.g),
            }

        key_array_fn = lambda s: s.phi_hat.data
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    else:
        raise NotImplementedError(
            f"Shallow water not implemented for grid '{tc.grid_type}'")

    # --- Time loop ---
    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis as _sh
        mass_init = float(jnp.mean(
            _sh(grid, state.phi_hat.data) / constants.g))
    else:
        h_data = state.h if isinstance(state.h, jnp.ndarray) else state.h.data
        mass_init = float(jnp.mean(h_data))
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, key_array_fn,
        label=f"SW W{test_num} ({tc.grid_type})", total_days=days)

    # --- Error norms for TC2 ---
    notes = ""
    if test_num == 2 and tc.grid_type == "cubed_sphere":
        exact = williamson_test2_exact(grid, days * 86400.0)
        h_final = state.h if isinstance(state.h, jnp.ndarray) else state.h.data
        h_exact = exact.h.data
        # Manual L2 / Linf norms for CDGrid state
        err = h_final - h_exact
        area = grid.area
        l2 = float(jnp.sqrt(jnp.sum(err**2 * area) / jnp.sum(h_exact**2 * area)))
        linf = float(jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(h_exact)))
        norms = {"l2": l2, "linf": linf}
        notes = f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    elif test_num == 2 and tc.grid_type == "latlon":
        exact = williamson_test2_exact_latlon(grid, days * 86400.0)
        norms = compute_error_norms_latlon(state, exact, grid)
        notes = f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    elif test_num == 2 and tc.grid_type == "icosahedral":
        norms = compute_error_norms_mpas(state.h.data, init_fns[2](mesh).h.data, mesh)
        notes = f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    elif diag.get("mean_height"):
        notes = f"mass drift={_compute_drift(diag['mean_height']):.2e}"

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "status": "PASS" if ok else "FAIL",
        "notes": notes})
    _save_case_diagnostics(
        output_dir, f"SW Williamson {test_num} {tc.resolution}", dt,
        diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("u", "Zonal wind u (m/s)", "RdBu_r"),
            ("v", "Meridional wind v (m/s)", "RdBu_r"),
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("height", "Fluid depth h (m)", "viridis"),
        ],
        mass_key="mean_height", energy_key="max_wind",
        scalar_units={"mean_height": "m", "max_wind": "m/s"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Held-Suarez
# ===========================================================================

def run_held_suarez(tc: TestCase, output_dir: Path, days: float, *,
                    radiation: str = "gray") -> tuple[str, float, str]:
    nlev = DEFAULT_NLEV

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel as PrimitiveEquationModel,
            CDGridPrimitiveEquationConfig as PrimitiveEquationConfig)
        from legoesm.atmosphere.physics.held_suarez import (
            held_suarez_forcing, held_suarez_init)
        from legoesm.core.operators import global_integral

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        hd = _hyperdiff_cube(n)
        dd = _div_damp_cube(n)
        ah = _laplacian_visc_cube(n)
        dt = 200.0
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = PrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init(grid, sigma)

        physics_fn = (_make_rrtmgp_physics("hydrostatic", dt)
                      if radiation == "rrtmgp" else held_suarez_forcing)

        def step_fn(s, dt_):
            return model.step_with_physics(s, dt_, physics_fn)

        mass_fn = lambda s: float(global_integral(s.p_s, grid))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
                "mean_T": float(jnp.mean(s.T.data)),
            }

        _cos_a = np.asarray(grid.cos_angle, dtype=np.float64)
        _sin_a = np.asarray(grid.sin_angle, dtype=np.float64)
        extract_fn = lambda s: _extract_hydro_cube_latlon(s, _cos_a, _sin_a)
        key_array_fn = lambda s: s.T.data
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
            LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig)
        from legoesm.atmosphere.physics.held_suarez_latlon import (
            held_suarez_forcing_latlon, held_suarez_init_latlon)
        from legoesm.core.operators_latlon import (
            global_integral as global_integral_ll)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        hd = _hyperdiff_latlon(n_lat)
        dd = _div_damp_latlon(n_lat)
        ah = _laplacian_visc_latlon(n_lat)
        dt = 200.0
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = LatLonPrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init_latlon(grid, sigma)

        physics_fn = (_make_rrtmgp_physics("hydrostatic", dt)
                      if radiation == "rrtmgp" else held_suarez_forcing_latlon)

        def step_fn(s, dt_):
            return model.step_with_physics(s, dt_, physics_fn)

        mass_fn = lambda s: float(global_integral_ll(s.p_s, grid))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
                "mean_T": float(jnp.mean(s.T.data)),
            }

        extract_fn = _extract_hydro_cube_latlon
        key_array_fn = lambda s: s.T.data
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
        from legoesm.atmosphere.physics.held_suarez_mpas import (
            held_suarez_forcing_mpas, held_suarez_init_mpas)

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        # MPAS PE currently uses sigma only
        sigma = create_sigma_coordinate(nlev)
        dt = 200.0
        ah = _laplacian_visc_ico(mesh)
        config = MPASPrimitiveEquationConfig(
            nu_del4=_hyperdiff_ico(mesh), nu_del2=ah, fix_mass=True)
        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        state = held_suarez_init_mpas(mesh, sigma)
        grid = mesh

        physics_fn_mpas = (_make_rrtmgp_physics("mpas", dt)
                           if radiation == "rrtmgp" else held_suarez_forcing_mpas)

        def step_fn(s, dt_):
            return model.step(s, dt_, physics_fn_mpas)

        mass_fn = lambda s: float(jnp.sum(s.p_s.data * mesh.areaCell))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.abs(s.u.data))),
                "mean_T": float(jnp.mean(s.T.data)),
            }

        lon_cell = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_cell = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi

        def extract_fn(s):
            return _extract_hydro_mpas(s, mesh, lon_cell, lat_cell)

        key_array_fn = lambda s: s.T.data
        coord_kind = "icosa"
        lon_deg = lon_cell
        lat_deg = lat_cell

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            create_gaussian_grid, sh_synthesis_3d,
        )
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
            isothermal_rest_state_spectral, spectral_pe_to_grid,
        )
        from legoesm.atmosphere.physics.held_suarez import (
            held_suarez_forcing_spectral,
        )

        n_max = int(tc.resolution.replace("T", ""))
        grid = create_gaussian_grid(n_max)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        dt = 600.0
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=2.338e15 * (21.0 / n_max) ** 4,
            spectral_filter_order=8,
            spectral_filter_strength=0.01,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)
        state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

        physics_fn = (_make_rrtmgp_physics("spectral_pe", dt)
                      if radiation == "rrtmgp" else held_suarez_forcing_spectral)

        def step_fn(s, dt_):
            return model.step(s, dt_, physics_fn=physics_fn)

        def check_fn(s):
            T = sh_synthesis_3d(grid, s.T_hat.data)
            return (check_finite({"T": T}),
                    float(jnp.max(jnp.abs(T))))

        def scalar_fn(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            return {
                "mass": float(jnp.mean(fields['p_s'])),
                "max_wind": float(jnp.max(jnp.sqrt(
                    fields['u'] ** 2 + fields['v'] ** 2))),
                "mean_T": float(jnp.mean(fields['T'])),
            }

        def extract_fn(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            u_sfc = np.asarray(fields['u'][..., -1], dtype=np.float64)
            v_sfc = np.asarray(fields['v'][..., -1], dtype=np.float64)
            return {
                "u": u_sfc,
                "v": v_sfc,
                "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
                "p_s": np.asarray(fields['p_s'], dtype=np.float64),
                "T_3d": np.asarray(fields['T'], dtype=np.float64),
            }

        key_array_fn = lambda s: s.T_hat.data
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    else:
        raise NotImplementedError(
            f"Held-Suarez not implemented for grid '{tc.grid_type}'")

    # --- Time loop ---
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, key_array_fn,
        label=f"Held-Suarez ({tc.grid_type})", total_days=days)

    mass_drift = _compute_drift(diag.get("mass", []))
    max_wind = diag["max_wind"][-1] if diag.get("max_wind") else 0
    notes = f"mass drift={mass_drift:.2e}, max|v|={max_wind:.1f}"

    level_values = np.asarray(
        getattr(sigma, "sigma_full", np.arange(nlev)), dtype=np.float64)

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "vertical_coord": tc.vertical_coord, "radiation": radiation,
        "days": days, "dt": dt, "levels": nlev,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir,
        f"Held-Suarez {radiation} {tc.resolution} {tc.vertical_coord}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("u", "Zonal wind u (m/s)", "RdBu_r"),
            ("v", "Meridional wind v (m/s)", "RdBu_r"),
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
        ],
        field_3d_key="T_3d", level_values=level_values,
        level_label="Sigma level",
        mass_key="mass", energy_key="mean_T",
        scalar_units={"mass": "Pa*sr", "max_wind": "m/s", "mean_T": "K"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Baroclinic Wave
# ===========================================================================

def run_baroclinic(tc: TestCase, output_dir: Path, days: float, *,
                   radiation: str = "gray") -> tuple[str, float, str]:
    nlev = DEFAULT_NLEV

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel as PrimitiveEquationModel,
            CDGridPrimitiveEquationConfig as PrimitiveEquationConfig)
        from legoesm.atmosphere.physics.baroclinic_wave import (
            baroclinic_wave_init)
        from legoesm.core.operators import global_integral

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        hd = _hyperdiff_cube(n)
        dd = _div_damp_cube(n)
        ah = _laplacian_visc_cube(n)
        dt = 200.0
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = PrimitiveEquationModel(grid, sigma, config)
        state = baroclinic_wave_init(grid, sigma_for_init, perturbed=True)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        mass_fn = lambda s: float(global_integral(s.p_s, grid))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        ps_init = np.array(state.p_s.data)

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
                "ps_perturbation": float(jnp.max(
                    jnp.abs(s.p_s.data - ps_init))),
            }

        _cos_a = np.asarray(grid.cos_angle, dtype=np.float64)
        _sin_a = np.asarray(grid.sin_angle, dtype=np.float64)
        extract_fn = lambda s: _extract_hydro_cube_latlon(s, _cos_a, _sin_a)
        key_array_fn = lambda s: s.T.data
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
            LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig)
        from legoesm.atmosphere.physics.baroclinic_wave import (
            baroclinic_wave_init_latlon)
        from legoesm.core.operators_latlon import (
            global_integral as global_integral_ll)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        hd = _hyperdiff_latlon(n_lat)
        dd = _div_damp_latlon(n_lat)
        ah = _laplacian_visc_latlon(n_lat)
        dt = 200.0
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = LatLonPrimitiveEquationModel(grid, sigma, config)
        state = baroclinic_wave_init_latlon(grid, sigma_for_init, perturbed=True)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        mass_fn = lambda s: float(global_integral_ll(s.p_s, grid))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        ps_init = np.array(state.p_s.data)

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
                "ps_perturbation": float(jnp.max(
                    jnp.abs(s.p_s.data - ps_init))),
            }

        extract_fn = _extract_hydro_cube_latlon
        key_array_fn = lambda s: s.T.data
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
        from legoesm.atmosphere.physics.held_suarez_mpas import (
            baroclinic_wave_init_mpas)

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        dt = 200.0
        ah = _laplacian_visc_ico(mesh)
        config = MPASPrimitiveEquationConfig(
            nu_del4=_hyperdiff_ico(mesh), nu_del2=ah, fix_mass=True)
        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        state = baroclinic_wave_init_mpas(mesh, sigma_for_init, perturbed=True)
        grid = mesh

        def step_fn(s, dt_):
            return model.step(s, dt_)

        mass_fn = lambda s: float(jnp.sum(s.p_s.data * mesh.areaCell))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.abs(s.u.data))),
            }

        lon_cell = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_cell = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi

        def extract_fn(s):
            return _extract_hydro_mpas(s, mesh, lon_cell, lat_cell)

        key_array_fn = lambda s: s.T.data
        coord_kind = "icosa"
        lon_deg = lon_cell
        lat_deg = lat_cell

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            create_gaussian_grid, sh_synthesis, sh_synthesis_3d,
        )
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
            baroclinic_wave_init_spectral, spectral_pe_to_grid,
        )

        n_max = int(tc.resolution.replace("T", ""))
        grid = create_gaussian_grid(n_max)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        dt = 600.0
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=2.338e15 * (21.0 / n_max) ** 4,
            spectral_filter_order=8,
            spectral_filter_strength=0.01,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)
        state = baroclinic_wave_init_spectral(
            grid, sigma_for_init, perturbed=True)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            T = sh_synthesis_3d(grid, s.T_hat.data)
            return (check_finite({"T": T}),
                    float(jnp.max(jnp.abs(T))))

        ps_init_spec = np.asarray(
            jnp.exp(sh_synthesis(grid, state.lnps_hat.data)),
            dtype=np.float64,
        )

        def scalar_fn(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            return {
                "mass": float(jnp.mean(fields['p_s'])),
                "max_wind": float(jnp.max(jnp.sqrt(
                    fields['u'] ** 2 + fields['v'] ** 2))),
                "ps_perturbation": float(jnp.max(
                    jnp.abs(fields['p_s'] - ps_init_spec))),
            }

        def extract_fn(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            u_sfc = np.asarray(fields['u'][..., -1], dtype=np.float64)
            v_sfc = np.asarray(fields['v'][..., -1], dtype=np.float64)
            return {
                "u": u_sfc,
                "v": v_sfc,
                "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
                "p_s": np.asarray(fields['p_s'], dtype=np.float64),
                "T_3d": np.asarray(fields['T'], dtype=np.float64),
            }

        key_array_fn = lambda s: s.T_hat.data
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    else:
        raise NotImplementedError(
            f"Baroclinic wave not implemented for grid '{tc.grid_type}'")

    # --- Time loop ---
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, key_array_fn,
        label=f"Baroclinic ({tc.grid_type})", total_days=days)

    mass_drift = _compute_drift(diag.get("mass", []))
    notes = f"mass drift={mass_drift:.2e}"
    if diag.get("max_wind"):
        notes += f", max|v|={diag['max_wind'][-1]:.1f}"

    level_values = np.asarray(
        getattr(sigma, "sigma_full", np.arange(nlev)), dtype=np.float64)

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "vertical_coord": tc.vertical_coord, "days": days, "dt": dt,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir,
        f"Baroclinic {tc.resolution} {tc.vertical_coord}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("u", "Zonal wind u (m/s)", "RdBu_r"),
            ("v", "Meridional wind v (m/s)", "RdBu_r"),
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
        ],
        field_3d_key="T_3d", level_values=level_values,
        level_label="Sigma level",
        mass_key="mass", energy_key="max_wind",
        scalar_units={
            "mass": "Pa*sr", "max_wind": "m/s",
            "ps_perturbation": "Pa"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: DCMIP Transport
# ===========================================================================

def run_dcmip_transport(tc: TestCase, output_dir: Path, days: float, *,
                        radiation: str = "gray") -> tuple[str, float, str]:
    """Run DCMIP-2012 transport test cases (Tests 1-1, 1-2, 1-3).

    Supports cubed-sphere, lat-lon, and icosahedral grids.
    Spectral grids are skipped (no tracer advection infrastructure).

    Produces tracer snapshots, lat/lon cross-sections, and vertical profiles
    via _save_case_diagnostics.
    """
    test_num = tc.run_kwargs["test_num"]

    if tc.grid_type == "spectral":
        record(tc, "SKIP", 0.0,
               "DCMIP transport not implemented for spectral grid")
        return "SKIP", 0.0, ""

    # --- Shared imports (grid-independent IC/wind helpers) ---
    from tests.test_cases.dcmip_transport import (
        create_dcmip_sigma,
        dcmip11_wind_geo, dcmip11_tracers_at_points,
        dcmip12_wind_geo, dcmip12_tracers_at_points,
        dcmip13_wind_geo, dcmip13_tracers_at_points,
    )

    TC_CFGS = {
        11: {"wind_geo": dcmip11_wind_geo, "period": 12.0,
             "dt": 1800.0, "n_tracers": 4},
        12: {"wind_geo": dcmip12_wind_geo, "period": 1.0,
             "dt": 600.0, "n_tracers": 1},
        13: {"wind_geo": dcmip13_wind_geo, "period": 12.0,
             "dt": 1800.0, "n_tracers": 4},
    }
    cfg = TC_CFGS[test_num]
    nlev = 30
    dt = cfg["dt"]
    sigma_coord = create_dcmip_sigma(nlev)

    # --- Grid-specific setup ---
    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import (
            create_cubed_sphere, rotate_winds_geo_to_grid)
        from legoesm.atmosphere.dynamics.tracer_transport import (
            TracerTransportModel, TracerTransportConfig)
        from tests.test_cases.dcmip_transport import (
            dcmip11_wind, dcmip11_init, dcmip12_wind, dcmip12_init,
            dcmip13_wind, dcmip13_init, compute_tracer_error_norms)

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        wind_fns = {11: dcmip11_wind, 12: dcmip12_wind, 13: dcmip13_wind}
        init_fns = {11: dcmip11_init, 12: dcmip12_init, 13: dcmip13_init}
        state_init = init_fns[test_num](grid, sigma_coord)
        model = TracerTransportModel(
            grid, sigma_coord, wind_fns[test_num],
            TracerTransportConfig(hyperdiff_coeff=0.0))
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.tracer_transport_latlon import (
            TracerTransportLatLonModel, TracerTransportLatLonConfig)
        from tests.test_cases.dcmip_transport import (
            dcmip11_init_latlon, dcmip12_init_latlon, dcmip13_init_latlon)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)

        # Wind wrapper: shared _geo returns geographic (u, v, sigma_dot),
        # no rotation needed for lat-lon.  Must meshgrid 1D lon/lat.
        wind_geo_fn = cfg["wind_geo"]

        def ll_wind(t, g, sc):
            return wind_geo_fn(t, g.lon2d, g.lat2d, sc)

        init_fns = {
            11: dcmip11_init_latlon,
            12: dcmip12_init_latlon,
            13: dcmip13_init_latlon,
        }
        state_init = init_fns[test_num](grid, sigma_coord)
        model = TracerTransportLatLonModel(
            grid, sigma_coord, ll_wind,
            TracerTransportLatLonConfig(hyperdiff_coeff=0.0))
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.tracer_transport_mpas import (
            TracerTransportMPASModel, TracerTransportMPASConfig)
        from tests.test_cases.dcmip_transport import (
            dcmip11_init_mpas, dcmip12_init_mpas, dcmip13_init_mpas)

        level = int(tc.resolution.replace("ico", ""))
        grid = create_voronoi_mesh(level)

        # Wind wrapper: convert geographic (u_east, v_north) to edge-normal.
        wind_geo_fn = cfg["wind_geo"]

        def mpas_wind(t, mesh, sc):
            u_east, v_north, sigma_dot = wind_geo_fn(
                t, mesh.lonCell, mesh.latCell, sc)
            # Project geographic winds to edge-normal at each level.
            # u_edge = u_east(cell) * cos(angleEdge) + v_north(cell) * sin(angleEdge)
            # Average the two cells straddling each edge.
            c1 = mesh.cellsOnEdge[0]  # (nEdges,)
            c2 = mesh.cellsOnEdge[1]
            cos_a = jnp.cos(mesh.angleEdge)  # (nEdges,)
            sin_a = jnp.sin(mesh.angleEdge)
            ue_c1 = u_east[c1] * cos_a[:, None] + v_north[c1] * sin_a[:, None]
            ue_c2 = u_east[c2] * cos_a[:, None] + v_north[c2] * sin_a[:, None]
            u_edge = 0.5 * (ue_c1 + ue_c2)  # (nEdges, nlev)
            return u_edge, sigma_dot

        init_fns = {
            11: dcmip11_init_mpas,
            12: dcmip12_init_mpas,
            13: dcmip13_init_mpas,
        }
        state_init = init_fns[test_num](grid, sigma_coord)
        model = TracerTransportMPASModel(
            grid, sigma_coord, mpas_wind,
            TracerTransportMPASConfig(hyperdiff_coeff=0.0))
        coord_kind = "icosa"
        lon_deg = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi
    else:
        record(tc, "SKIP", 0.0,
               f"DCMIP transport not implemented for grid '{tc.grid_type}'")
        return "SKIP", 0.0, ""

    # --- Extract function for tracer snapshots ---
    n_tracers = cfg["n_tracers"]

    def extract_fn(s):
        q = np.asarray(s.tracers.data, dtype=np.float64)
        out = {}
        # Surface-level (bottom) tracer for 2D map snapshots
        for i in range(min(n_tracers, 4)):
            out[f"q{i+1}"] = q[..., -1, i]
        # Full 3D for cross-sections (first tracer)
        out["q1_3d"] = q[..., 0]
        return out

    def step_fn(s, dt_):
        return model.step(s, dt_)

    def check_fn(s):
        return (check_finite({"tracers": s.tracers.data}),
                float(jnp.max(jnp.abs(s.tracers.data))))

    def scalar_fn(s):
        q = s.tracers.data
        return {
            "q1_min": float(jnp.min(q[..., 0])),
            "q1_max": float(jnp.max(q[..., 0])),
            "q1_mean": float(jnp.mean(q[..., 0])),
        }

    def key_array_fn(s):
        return s.tracers.data

    # --- Time loop ---
    period = min(days, cfg["period"])
    n_steps = int(period * 86400.0 / dt)
    diag_every = max(1, n_steps // 20)

    state_final, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state_init, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, key_array_fn,
        label=f"DCMIP transport {test_num} ({tc.grid_type})",
        total_days=period)

    # --- Error norms (for flow-reversal tests on cubed-sphere) ---
    notes = ""
    if test_num in (11, 12) and tc.grid_type == "cubed_sphere":
        norms = compute_tracer_error_norms(state_final, state_init, grid)
        norm_strs = [f"q{i + 1} L2={norms['l2'][i]:.4e}"
                     for i in range(n_tracers)]
        notes = ", ".join(norm_strs)

    # --- Diagnostics output ---
    level_values = np.asarray(sigma_coord.sigma_full, dtype=np.float64)
    field_specs_2d = [(f"q{i+1}", f"Tracer q{i+1}", "viridis")
                      for i in range(min(n_tracers, 4))]

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "levels": nlev, "period_days": period, "dt": dt,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir,
        f"DCMIP transport {test_num} {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=field_specs_2d,
        field_3d_key="q1_3d", level_values=level_values,
        level_label="Sigma level",
        scalar_units={"q1_min": "kg/kg", "q1_max": "kg/kg",
                      "q1_mean": "kg/kg"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: AMIP
# ===========================================================================

def run_amip(tc: TestCase, output_dir: Path, days: float, *,
             radiation: str = "gray") -> tuple[str, float, str]:
    nlev = DEFAULT_NLEV

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import standard_hybrid_levels
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel as PrimitiveEquationModel,
            CDGridPrimitiveEquationConfig as PrimitiveEquationConfig)
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init
        from legoesm.core.operators import global_integral

        from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        sigma = standard_hybrid_levels(nlev)
        hd = _hyperdiff_cube(n)
        dd = _div_damp_cube(n)
        ah = _laplacian_visc_cube(n)
        dt = 300.0
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = PrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init(grid, sigma, T_init=280.0)

        physics_fn = held_suarez_forcing

        def step_fn(s, dt_):
            return model.step_with_physics(s, dt_, physics_fn)

        mass_fn = lambda s: float(global_integral(s.p_s, grid))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
                "mean_T": float(jnp.mean(s.T.data)),
                "mean_p_s": float(jnp.mean(s.p_s.data)),
            }

        _cos_a = np.asarray(grid.cos_angle, dtype=np.float64)
        _sin_a = np.asarray(grid.sin_angle, dtype=np.float64)
        extract_fn = lambda s: _extract_hydro_cube_latlon(s, _cos_a, _sin_a)
        key_array_fn = lambda s: s.T.data
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import standard_hybrid_levels
        from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
            LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig)
        from legoesm.atmosphere.physics.held_suarez_latlon import (
            held_suarez_init_latlon, held_suarez_forcing_latlon)
        from legoesm.core.operators_latlon import (
            global_integral as global_integral_ll)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        sigma = standard_hybrid_levels(nlev)
        hd = _hyperdiff_latlon(n_lat)
        dd = _div_damp_latlon(n_lat)
        ah = _laplacian_visc_latlon(n_lat)
        dt = 300.0
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = LatLonPrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init_latlon(grid, sigma)

        physics_fn = held_suarez_forcing_latlon

        def step_fn(s, dt_):
            return model.step_with_physics(s, dt_, physics_fn)

        mass_fn = lambda s: float(global_integral_ll(s.p_s, grid))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(
                    s.u.data ** 2 + s.v.data ** 2))),
                "mean_T": float(jnp.mean(s.T.data)),
                "mean_p_s": float(jnp.mean(s.p_s.data)),
            }

        extract_fn = _extract_hydro_cube_latlon
        key_array_fn = lambda s: s.T.data
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
        from legoesm.atmosphere.physics.held_suarez_mpas import (
            held_suarez_forcing_mpas, held_suarez_init_mpas)
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        sigma = create_sigma_coordinate(nlev)
        dt = 200.0
        ah = _laplacian_visc_ico(mesh)
        config = MPASPrimitiveEquationConfig(
            nu_del4=_hyperdiff_ico(mesh), nu_del2=ah, fix_mass=True)
        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        state = held_suarez_init_mpas(mesh, sigma, T_init=280.0)
        grid = mesh

        def step_fn(s, dt_):
            return model.step(s, dt_, held_suarez_forcing_mpas)

        mass_fn = lambda s: float(jnp.sum(s.p_s.data * mesh.areaCell))

        def check_fn(s):
            return (check_finite({"T": s.T.data, "u": s.u.data}),
                    float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.abs(s.u.data))),
                "mean_T": float(jnp.mean(s.T.data)),
                "mean_p_s": float(jnp.mean(s.p_s.data)),
            }

        lon_cell = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_cell = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi

        def extract_fn(s):
            return _extract_hydro_mpas(s, mesh, lon_cell, lat_cell)

        key_array_fn = lambda s: s.T.data
        coord_kind = "icosa"
        lon_deg = lon_cell
        lat_deg = lat_cell

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            create_gaussian_grid, sh_synthesis_3d,
        )
        from legoesm.grids.vertical import standard_hybrid_levels
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
            isothermal_rest_state_spectral, spectral_pe_to_grid,
        )
        from legoesm.atmosphere.physics.held_suarez import (
            held_suarez_forcing_spectral,
        )

        n_max = int(tc.resolution.replace("T", ""))
        grid = create_gaussian_grid(n_max)
        sigma = standard_hybrid_levels(nlev)
        dt = 600.0
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=2.338e15 * (21.0 / n_max) ** 4,
            spectral_filter_order=8,
            spectral_filter_strength=0.01,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)
        state = isothermal_rest_state_spectral(
            grid, sigma, T_init=280.0)

        physics_fn = held_suarez_forcing_spectral

        def step_fn(s, dt_):
            return model.step(s, dt_, physics_fn=physics_fn)

        def check_fn(s):
            T = sh_synthesis_3d(grid, s.T_hat.data)
            return (check_finite({"T": T}),
                    float(jnp.max(jnp.abs(T))))

        def scalar_fn(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            return {
                "mass": float(jnp.mean(fields['p_s'])),
                "max_wind": float(jnp.max(jnp.sqrt(
                    fields['u'] ** 2 + fields['v'] ** 2))),
                "mean_T": float(jnp.mean(fields['T'])),
                "mean_p_s": float(jnp.mean(fields['p_s'])),
            }

        def extract_fn(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            u_sfc = np.asarray(fields['u'][..., -1], dtype=np.float64)
            v_sfc = np.asarray(fields['v'][..., -1], dtype=np.float64)
            return {
                "u": u_sfc,
                "v": v_sfc,
                "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
                "p_s": np.asarray(fields['p_s'], dtype=np.float64),
                "T_3d": np.asarray(fields['T'], dtype=np.float64),
            }

        key_array_fn = lambda s: s.T_hat.data
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    else:
        raise NotImplementedError(
            f"AMIP not implemented for grid '{tc.grid_type}'")

    # --- Time loop ---
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(24 * 3600 / dt))

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, key_array_fn,
        label=f"AMIP ({tc.grid_type})", total_days=days)

    mass_drift = _compute_drift(diag.get("mass", []))
    notes = f"mass drift={mass_drift:.2e}"

    level_values = np.asarray(
        getattr(sigma, "sigma_full", np.arange(nlev)), dtype=np.float64)

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "radiation": radiation, "days": days, "dt": dt,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir,
        f"AMIP {radiation} {tc.resolution} hybrid",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("u", "Zonal wind u (m/s)", "RdBu_r"),
            ("v", "Meridional wind v (m/s)", "RdBu_r"),
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
        ],
        field_3d_key="T_3d", level_values=level_values,
        level_label="Sigma level",
        mass_key="mass", energy_key="mean_T",
        scalar_units={
            "mass": "Pa*sr", "max_wind": "m/s", "mean_T": "K",
            "mean_p_s": "Pa"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Non-Hydrostatic (DCMIP-2025)
# ===========================================================================

def run_nonhydrostatic(tc: TestCase, output_dir: Path, days: float, *,
                       radiation: str = "gray") -> tuple[str, float, str]:
    test_case = tc.run_kwargs["test_case"]
    nlev = DEFAULT_NLEV

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
            CDGridCompressibleEulerModel as CompressibleEulerModel,
            CDGridCompressibleEulerConfig as CompressibleEulerConfig)

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        hd = _hyperdiff_cube(n)

        if test_case == "tc1":
            from tests.test_cases.dcmip2025 import dcmip25_tc1_init
            state, hcoord, tmetric = dcmip25_tc1_init(grid, n_levels=nlev)
            dt = max(0.2, 6.0 * (16.0 / n))
            nh_config = CompressibleEulerConfig(
                n_acoustic_substeps=10, semi_implicit_acoustic=True,
                sponge_width=10000.0, sponge_coeff=0.05,
                hyperdiff_coeff=hd,
                acoustic_off_centering=0.1)
        elif test_case == "tc2a":
            from tests.test_cases.dcmip2025 import dcmip25_tc2_init
            state, hcoord, tmetric, small_grid = dcmip25_tc2_init(
                grid, n_levels=nlev)
            grid = small_grid
            dt = max(0.2, 4.0 * (16.0 / n))
            nh_config = CompressibleEulerConfig(
                n_acoustic_substeps=10, semi_implicit_acoustic=True,
                sponge_width=5000.0, sponge_coeff=0.1,
                hyperdiff_coeff=hd,
                acoustic_off_centering=0.1)
        elif test_case == "tc3":
            from tests.test_cases.dcmip2025 import dcmip25_tc3_init
            state, hcoord, tmetric, small_grid = dcmip25_tc3_init(
                grid, n_levels=nlev)
            grid = small_grid
            dt = max(0.1, 2.0 * (16.0 / n))
            nh_config = CompressibleEulerConfig(
                n_acoustic_substeps=10, semi_implicit_acoustic=True,
                sponge_width=5000.0, sponge_coeff=0.1,
                hyperdiff_coeff=hd,
                acoustic_off_centering=0.1)
        else:
            raise ValueError(f"Unknown NH test case: {test_case}")

        model = CompressibleEulerModel(grid, hcoord, tmetric, nh_config)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            return (check_finite({
                "u": s.u.data, "w": s.w.data,
                "theta": s.theta_prime.data}),
                float(jnp.max(jnp.abs(s.u.data))))

        def scalar_fn(s):
            return {
                "max_abs_w": float(jnp.max(jnp.abs(s.w.data))),
                "mean_theta_prime": float(jnp.mean(s.theta_prime.data)),
                "mean_rho_prime": float(jnp.mean(s.rho_prime.data)),
            }

        _cos_a_nh = np.asarray(grid.cos_angle, dtype=np.float64)
        _sin_a_nh = np.asarray(grid.sin_angle, dtype=np.float64)

        def extract_fn(s):
            # Use surface level (k=-1) instead of model top (k=0) which
            # is inside the sponge layer and gets damped to zero.
            u = np.asarray(s.u.data[..., -1], dtype=np.float64)
            v = np.asarray(s.v.data[..., -1], dtype=np.float64)
            # Rotate face-local to geographic
            u, v = _cos_a_nh * u - _sin_a_nh * v, _sin_a_nh * u + _cos_a_nh * v
            w_idx = min(s.w.data.shape[-1] // 2, s.w.data.shape[-1] - 1)
            return {
                "u": u, "v": v,
                "w_mid": np.asarray(s.w.data[..., w_idx], dtype=np.float64),
                "wind_speed": np.sqrt(u ** 2 + v ** 2),
                "theta_prime_3d": np.asarray(
                    s.theta_prime.data, dtype=np.float64),
                "rho_prime_3d": np.asarray(
                    s.rho_prime.data, dtype=np.float64),
            }

        key_array_fn = lambda s: s.u.data
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        z_full = np.asarray(
            getattr(hcoord, "z_full", np.arange(nlev)), dtype=np.float64)

    elif tc.grid_type == "icosahedral":
        if test_case != "tc1":
            record(tc, "SKIP", 0.0, f"MPAS NH only supports tc1, got {test_case}")
            return "SKIP", 0.0, ""

        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.compressible_euler_mpas import (
            MPASCompressibleEulerModel, MPASCompressibleEulerConfig)
        from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas import (
            dcmip25_tc1_init_mpas)

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        state, hcoord, tmetric = dcmip25_tc1_init_mpas(
            mesh, n_levels=nlev)
        grid = mesh

        dx_mean = float(jnp.sqrt(
            4.0 * jnp.pi * mesh.radius ** 2 / mesh.nCells))
        dt = min(max(0.2, 6.0 * (200.0 / (dx_mean / 1000.0))), 6.0)
        nh_config = MPASCompressibleEulerConfig(
            n_acoustic_substeps=10, sponge_width=10000.0,
            sponge_coeff=0.05,
            nu_del4=dx_mean ** 4 / (48.0 * 3600.0))
        model = MPASCompressibleEulerModel(
            mesh, hcoord, tmetric, nh_config)

        lon_cell = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_cell = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        lon_edge = np.asarray(mesh.lonEdge, dtype=np.float64) * 180 / np.pi
        lat_edge = np.asarray(mesh.latEdge, dtype=np.float64) * 180 / np.pi

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            # u is on edges (nEdges, nlev), w/theta on cells (nCells, nlev)
            # check each independently to avoid shape mismatch
            u_ok = check_finite({"u": s.u.data})
            cell_ok = check_finite({
                "w": s.w.data, "theta": s.theta_prime.data})
            metric = float(jnp.max(jnp.abs(s.w.data)))
            return (u_ok and cell_ok, metric)

        def scalar_fn(s):
            return {
                "max_abs_w": float(jnp.max(jnp.abs(s.w.data))),
                "mean_theta_prime": float(jnp.mean(s.theta_prime.data)),
                "mean_rho_prime": float(jnp.mean(s.rho_prime.data)),
            }

        def extract_fn(s):
            # u is on edges (nEdges, nlev) — use edge coordinates
            u_sfc = np.asarray(s.u.data, dtype=np.float64)
            if u_sfc.ndim > 1:
                u_sfc = u_sfc[:, -1]  # surface level
            u_ll = _bin_to_latlon(u_sfc, lon_edge, lat_edge)

            # w is on cells (nCells, nlev+1) — use cell coordinates
            w_arr = np.asarray(s.w.data, dtype=np.float64)
            if w_arr.ndim >= 2 and w_arr.shape[0] == lon_cell.size:
                w_mid = w_arr[:, min(
                    w_arr.shape[-1] // 2, w_arr.shape[-1] - 1)]
                w_ll = _bin_to_latlon(w_mid, lon_cell, lat_cell)
            else:
                w_ll = np.full((181, 360), np.nan)
            return {
                "u": u_ll, "w_mid": w_ll,
                "theta_prime_3d": np.asarray(
                    s.theta_prime.data, dtype=np.float64),
                "rho_prime_3d": np.asarray(
                    s.rho_prime.data, dtype=np.float64),
            }

        key_array_fn = lambda s: s.u.data
        coord_kind = "icosa"
        lon_deg = lon_cell
        lat_deg = lat_cell
        z_full = np.asarray(
            getattr(hcoord, "z_full", np.arange(nlev)), dtype=np.float64)

    elif tc.grid_type == "latlon":
        record(tc, "SKIP", 0.0, "DCMIP NH init requires CubedSphereGrid; lat-lon NH not yet available")
        return "SKIP", 0.0, ""

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            create_gaussian_grid, sh_synthesis_3d,
        )
        from legoesm.atmosphere.dynamics.spectral_nh import (
            SpectralCompressibleEulerModel, SpectralNHConfig,
            dcmip25_tc1_init_spectral,
        )

        if test_case != "tc1":
            raise NotImplementedError(
                f"Spectral NH only supports tc1, got {test_case}")

        n_max = int(tc.resolution.replace("T", ""))
        grid = create_gaussian_grid(n_max)
        state, hcoord, tmetric = dcmip25_tc1_init_spectral(
            grid, n_levels=nlev)

        dt = max(0.5, 6.0 * (21.0 / n_max))
        nh_config = SpectralNHConfig(
            n_acoustic_substeps=10,
            semi_implicit_acoustic=True,
            sponge_width=10000.0,
            sponge_coeff=0.05,
            hyperdiff_coeff=2.338e15 * (21.0 / n_max) ** 4,
        )
        model = SpectralCompressibleEulerModel(
            grid, hcoord, tmetric, nh_config)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            w = sh_synthesis_3d(grid, s.w_hat.data)
            theta_p = sh_synthesis_3d(grid, s.theta_prime_hat.data)
            return (check_finite({"w": w, "theta": theta_p}),
                    float(jnp.max(jnp.abs(w))))

        def scalar_fn(s):
            w = sh_synthesis_3d(grid, s.w_hat.data)
            theta_p = sh_synthesis_3d(grid, s.theta_prime_hat.data)
            rho_p = sh_synthesis_3d(grid, s.rho_prime_hat.data)
            return {
                "max_abs_w": float(jnp.max(jnp.abs(w))),
                "mean_theta_prime": float(jnp.mean(theta_p)),
                "mean_rho_prime": float(jnp.mean(rho_p)),
            }

        def extract_fn(s):
            from legoesm.grids.gaussian import uv_from_vordiv_3d
            u_cos, v_cos = uv_from_vordiv_3d(
                grid, s.vor_hat.data, s.div_hat.data)
            _COS_MIN = 1.0e-6
            cos3 = jnp.clip(grid.cos_lat[:, None, None], _COS_MIN, None)
            u_grid = u_cos / cos3
            v_grid = v_cos / cos3
            # Use surface level (k=-1) instead of model top (k=0)
            # which is inside the sponge layer and gets damped to zero.
            u_sfc = np.asarray(u_grid[..., -1], dtype=np.float64)
            v_sfc = np.asarray(v_grid[..., -1], dtype=np.float64)
            w = sh_synthesis_3d(grid, s.w_hat.data)
            w_idx = min(w.shape[-1] // 2, w.shape[-1] - 1)
            return {
                "u": u_sfc,
                "v": v_sfc,
                "w_mid": np.asarray(w[..., w_idx], dtype=np.float64),
                "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
                "theta_prime_3d": np.asarray(
                    sh_synthesis_3d(grid, s.theta_prime_hat.data),
                    dtype=np.float64),
                "rho_prime_3d": np.asarray(
                    sh_synthesis_3d(grid, s.rho_prime_hat.data),
                    dtype=np.float64),
            }

        key_array_fn = lambda s: s.vor_hat.data
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        z_full = np.asarray(
            getattr(hcoord, "z_full", np.arange(nlev)), dtype=np.float64)

    else:
        raise NotImplementedError(
            f"NH not implemented for grid '{tc.grid_type}'")

    # --- Time loop ---
    duration_hours = days * 24.0
    n_steps = int(duration_hours * 3600.0 / dt)
    diag_every = max(1, n_steps // 20)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, key_array_fn,
        label=f"NH {test_case} ({tc.grid_type})", total_days=days)

    if ok:
        if hasattr(state, 'w'):
            w_max = float(jnp.max(jnp.abs(state.w.data)))
        elif hasattr(state, 'w_hat'):
            from legoesm.grids.gaussian import sh_synthesis_3d
            w_max = float(jnp.max(jnp.abs(sh_synthesis_3d(grid, state.w_hat.data))))
        else:
            w_max = float("nan")
    else:
        w_max = float("nan")
    notes = f"|w|_max={w_max:.4f} m/s, dt={dt:.2f}s"

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "levels": nlev, "duration_hours": duration_hours, "dt": dt,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"})
    _save_case_diagnostics(
        output_dir, f"NH DCMIP {test_case} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("u", "Zonal wind u (m/s)", "RdBu_r"),
            ("v", "Meridional wind v (m/s)", "RdBu_r"),
            ("w_mid", "Vertical velocity w mid (m/s)", "RdBu_r"),
            ("wind_speed", "Wind speed (m/s)", "magma"),
        ],
        field_3d_key="theta_prime_3d", level_values=z_full,
        level_label="Height (m)", invert_levels=False,
        mass_key="mean_rho_prime", energy_key="mean_theta_prime",
        scalar_units={
            "max_abs_w": "m/s", "mean_theta_prime": "K",
            "mean_rho_prime": "kg/m^3"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner dispatch
# ===========================================================================

RUNNERS: dict[str, Callable] = {
    "williamson2": run_shallow_water,
    "williamson5": run_shallow_water,
    "held_suarez": run_held_suarez,
    "baroclinic": run_baroclinic,
    "dcmip_transport_11": run_dcmip_transport,
    "dcmip_transport_12": run_dcmip_transport,
    "dcmip_transport_13": run_dcmip_transport,
    "amip": run_amip,
    "dcmip_tc1": run_nonhydrostatic,
    "dcmip_tc2": run_nonhydrostatic,
    "dcmip_tc3": run_nonhydrostatic,
}

CATEGORY_RUNNER_HINTS: dict[str, str] = {
    "shallow_water": "scripts/atmosphere/run_shallow_water_tests.py",
    "hydrostatic": "scripts/atmosphere/run_hydrostatic_tests.py",
    "nonhydrostatic": "scripts/atmosphere/run_nonhydrostatic_tests.py",
    "rce": "scripts/atmosphere/run_rce_tests.py",
    "aquaplanet": "scripts/atmosphere/run_aquaplanet_tests.py",
    "ocean": "scripts/ocean/run_ocean_category_tests.py",
}


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Atmosphere test matrix for legoESM dynamical cores.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--only", type=str, default="all",
        choices=["sw", "hydro", "nh", "all"],
        help="Run only a specific equation set (default: all)")
    p.add_argument(
        "--grid", type=str, default="all",
        choices=["cubed_sphere", "latlon", "icosahedral", "spectral", "all"],
        help="Run only a specific grid type (default: all)")
    p.add_argument(
        "--test", type=str, default=None,
        help="Run only cases matching this name (e.g. held_suarez)")
    p.add_argument(
        "--radiation", type=str, default="gray",
        choices=["gray", "rrtmgp"],
        help="Radiation scheme for applicable tests (default: gray)")
    p.add_argument(
        "--resolution", type=str, default=None,
        help="Override baseline resolution (e.g. C48, 90x180, ico6)")
    p.add_argument(
        "--output", "-o", type=str, default="results/atmosphere",
        help="Base output directory (default: results/atmosphere)")
    p.add_argument(
        "--quick", action="store_true",
        help="Use shorter durations for quick verification")
    p.add_argument(
        "--list", action="store_true",
        help="List all test cases and exit")
    p.add_argument(
        "--list-category-scripts", action="store_true",
        help="List canonical per-category runner scripts and exit")
    return p


def filter_tests(tests: list[TestCase], args) -> list[TestCase]:
    filtered = tests
    eq_map = {"sw": "shallow_water", "hydro": "hydrostatic",
              "nh": "nonhydrostatic"}
    if args.only != "all":
        eq_set = eq_map[args.only]
        filtered = [t for t in filtered if t.equation_set == eq_set]
    if args.grid != "all":
        filtered = [t for t in filtered if t.grid_type == args.grid]
    if args.test:
        filtered = [t for t in filtered if args.test in t.case]
    return filtered


def main():
    parser = build_parser()
    args = parser.parse_args()

    tests = filter_tests(TEST_MATRIX, args)

    if args.list_category_scripts:
        print("Canonical category runner scripts:")
        for cat, path in CATEGORY_RUNNER_HINTS.items():
            print(f"  - {cat:<14} {path}")
        return

    if args.list:
        print(f"{'#':>3}  {'Equation Set':<16}  {'Case':<22}  "
              f"{'Grid':<14}  {'Resolution':<10}  {'Vert':<7}  "
              f"{'Days':>8}  {'Quick':>8}")
        print("-" * 105)
        for i, tc in enumerate(tests, 1):
            print(f"{i:3d}  {tc.equation_set:<16}  {tc.case:<22}  "
                  f"{tc.grid_type:<14}  {tc.resolution:<10}  "
                  f"{tc.vertical_coord:<7}  {tc.duration_days:8.4f}  "
                  f"{tc.quick_days:8.4f}")
        print(f"\nTotal: {len(tests)} test cases "
              f"(of {len(TEST_MATRIX)} in full matrix)")
        return
    if not tests:
        print("No tests match the given filters.")
        return

    if args.resolution:
        tests = [TestCase(
            t.equation_set, t.case, t.grid_type, args.resolution,
            t.vertical_coord, t.duration_days, t.quick_days, t.run_kwargs)
            for t in tests]

    output_base = Path(args.output)

    print("=" * 78)
    print("  legoESM Atmosphere Test Matrix")
    print("=" * 78)
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  X64:        {jax.config.jax_enable_x64}")
    print(f"  Devices:    {jax.devices()}")
    print(f"  Output:     {output_base}")
    print(f"  Radiation:  {args.radiation}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Tests:      {len(tests)} / {len(TEST_MATRIX)}")
    print("=" * 78)
    print()

    t_start_all = time.time()

    for i, tc in enumerate(tests, 1):
        days = tc.quick_days if args.quick else tc.duration_days
        out_dir = output_base / tc.output_path

        label = f"{tc.equation_set}/{tc.case}/{tc.grid_type}"
        print(f"\n[{i}/{len(tests)}] {label} ({tc.resolution}, "
              f"{tc.vertical_coord}, {days:.4g} days)")
        print("-" * 60)

        runner = RUNNERS.get(tc.case)
        if runner is None:
            record(tc, "ERROR", 0, f"Unknown runner for case: {tc.case}")
            continue

        try:
            status, wall, notes = runner(
                tc, out_dir, days, radiation=args.radiation)
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
    print(f"  {'Status':6}  {'Grid':<14}  {'Equation Set':<16}  "
          f"{'Case':<22}  {'Time':>8}  Notes")
    print("-" * 105)

    n_pass = n_fail = n_error = n_skip = 0
    for r in ALL_RESULTS:
        icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!",
                "SKIP": "--"}[r["status"]]
        print(f"  {icon}{r['status']:5}  {r['grid']:<14}  "
              f"{r['equation_set']:<16}  {r['test']:<22}  "
              f"{r['wall_time']:7.1f}s  {r['notes']}")
        if r["status"] == "PASS":
            n_pass += 1
        elif r["status"] == "FAIL":
            n_fail += 1
        elif r["status"] == "SKIP":
            n_skip += 1
        else:
            n_error += 1

    print("-" * 105)
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
            "radiation": args.radiation,
        }, f, indent=2)
    with open(output_base / "summary.txt", "w") as f:
        f.write("legoESM Atmosphere Test Matrix Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(ALL_RESULTS)} tests | "
                f"PASS: {n_pass} | FAIL: {n_fail} | SKIP: {n_skip} | "
                f"ERROR: {n_error}\n")
        f.write(f"Wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)\n")
        f.write(f"Radiation: {args.radiation}\n")
        f.write(f"Quick mode: {args.quick}\n\n")
        for r in ALL_RESULTS:
            f.write(f"{r['status']:5}  {r['grid']:<14}  "
                    f"{r['equation_set']:<16}  {r['test']:<22}  "
                    f"{r['wall_time']:7.1f}s  {r['notes']}\n")

    print(f"\n  Summary: {output_base / 'summary.json'}")

    if n_fail > 0 or n_error > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
