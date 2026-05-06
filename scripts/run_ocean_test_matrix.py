#!/usr/bin/env python
"""Ocean test matrix: organized test runner for legoESM ocean dynamical cores.

Runs a comprehensive suite of ocean test cases across cubed-sphere, lat-lon,
and MPAS grids at ~5 degree resolution.

Test cases:
  Existing:
    - rest_state          Rest-state adjustment (stability check)
    - barotropic_wave     Gaussian SSH perturbation propagation
    - barotropic_gyre     Wind-driven single barotropic gyre (Stommel/Munk)
    - barotropic_double_gyre  Wind-driven barotropic double gyre (Holland & Lin)
    - baroclinic          Meridional temperature front relaxation
    - phillips_two_layer  Phillips 2-layer baroclinic instability

  Barotropic dynamics (Bishnu et al. 2024):
    - inertia_gravity_wave  Inertia-gravity (Poincare) wave propagation

  Numerical mixing & dianeutral transport (NEMO; Petersen et al. 2015):
    - lock_exchange       Density-driven gravity current (RPE diagnostic)
    - overflow            Dense water descending a bathymetric slope

  Tracer transport (Hecht et al. 2000):
    - stommel_gyre_tracer Passive tracer in wind-driven Stommel gyre

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

References:
    Bishnu et al. (2024), JAMES. DOI: 10.1029/2022MS003545
    Petersen et al. (2015), Ocean Modelling 86, 93-113.
        DOI: 10.1016/j.ocemod.2014.12.004
    Hecht et al. (2000), Ocean Modelling 2, 1-15.
        DOI: 10.1016/S1463-5003(00)00004-4

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --quick
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --only lock_exchange
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --grid cubed_sphere
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

# Line-buffered stdout for CI/log visibility.
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import shutil
import time
import traceback
from dataclasses import dataclass, field

# Ensure project root is on sys.path (for legoesm imports).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Ensure scripts/ is on sys.path (for ocean_test_matrix package).
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# JAX configuration must happen before any jax.numpy import.
import jax
jax.config.update("jax_enable_x64", True)

from legoesm.runtime.backend import ensure_metal_or_fallback
ensure_metal_or_fallback()

import jax.numpy as jnp
import numpy as np
import pandas as pd  # iter-19 fix: pd was used (lines 4617, 6106) but
                    # never imported, silently disabling the
                    # cross-grid comparison block introduced by the
                    # iter-1-predecessor's
                    # ``cross_grid_comparison_plots_plan.md`` work.
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Configuration constants
# ===========================================================================

# ~2.5 degree resolutions per grid type
GRID_RESOLUTIONS: dict[str, str] = {
    "cubed_sphere": "C24",
    "latlon": "36x72",
    "mpas": "ico3",
    "mpas_regional": "300km",
    "latlon_regional": "24x48",
    "cs_regional": "C24",
    "spectral": "T21",
}

# Standard grid types for the full test matrix.
# Regional grids are only added to specific test cases (gyre experiments).
GRID_TYPES = ["cubed_sphere", "latlon", "mpas"]
REGIONAL_GRID_TYPES = ["mpas_regional", "latlon_regional", "cs_regional"]

DEFAULT_NLEV = 10
DEFAULT_H_MAX = 5500.0
DEFAULT_DT = 300.0  # seconds (scaled for ~2.5 deg resolution CFL)

# Physical constants for idealized ocean test cases — use canonical values.
from legoesm import constants as _C
_A_EARTH = _C.R_earth   # Earth radius (m)
_OMEGA_E = _C.Omega      # Earth rotation rate (rad/s)
_G_EARTH = _C.g           # gravitational acceleration (m/s^2)

# Field ranges for consistent plotting across grid types
FIELD_RANGES = {
    "rest_state_stratified_with_land": {
        "eta": (-1e-6, 1e-6),      # meters - rest state should have tiny SSH
        "SST": (1.5, 21.0),        # °C - range from deep to surface T
    },
    "baroclinic_gyre": {
        "eta": (None, None),       # meters - adaptive range for circulation patterns
        "SST": (None, None),       # °C - adaptive range for circulation-driven T patterns
    },
    "rest_state_stratified_no_land": {
        "eta": (-1e-6, 1e-6),      # meters - rest state should have tiny SSH (pure ocean)
        "SST": (1.5, 21.0),        # °C - range from deep to surface T
    },
    "barotropic_wave": {
        "eta": (-1.5, 1.5),        # meters - wave amplitude ~1m  
        "SST": (1.5, 21.0),        # °C - background temperature range
    },
    "barotropic_gyre": {
        "eta": (-0.02, 0.02),      # meters - gyre SSH (small after quick spin-up)
        "speed_sfc": (0, 0.15),     # m/s - surface speeds
        "SST": (9.5, 10.5),        # °C - uniform 10°C (barotropic)
    },
    "barotropic_double_gyre": {
        "eta": (-0.02, 0.02),      # meters - gyre SSH (small after quick spin-up)
        "speed_sfc": (0, 0.15),     # m/s - surface speeds
        "SST": (9.5, 10.5),        # °C - uniform 10°C (barotropic)
    },
    "baroclinic_gyre": {
        "eta": (-0.05, 0.05),      # meters - larger SSH with baroclinic dynamics
        "speed_sfc": (0, 0.5),      # m/s - higher speeds with thermal wind
        "SST": (None, None),        # °C - adaptive range for circulation-driven T patterns
    },
    "geostrophic_adjustment": {
        "eta": (-0.1, 0.1),        # meters - adjustment process
        "SST": (1.5, 21.0),        # °C - background temperature range
    },
    "phillips_two_layer": {
        "eta": (-0.2, 0.2),        # meters - 2-layer dynamics
        "SST": (8, 16),             # °C - 2-layer temperature range
    },
    "inertia_gravity_wave": {
        "eta": (-1.2, 1.2),        # meters - IGW amplitude
        "SST": (1.5, 21.0),        # °C
    },
    "lock_exchange": {
        "eta": (-0.05, 0.05),      # meters - density current adjustment
        "SST": (-1, 21),            # °C - cold/warm water exchange
    },
    "overflow": {
        "eta": (-0.1, 0.1),        # meters - dense water overflow
        "SST": (-1, 21),            # °C - cold dense water
    },
    "stommel_gyre_tracer": {
        "eta": (-0.02, 0.02),      # meters - gyre circulation (small during spin-up)
        "SST": (1.5, 21.0),        # °C
        "SSS": (33, 37),            # PSU - tracer salinity range
    },
}


# ===========================================================================
# TestCase dataclass
# ===========================================================================

@dataclass
class TestCase:
    """A single test case in the ocean matrix."""
    case: str               # rest_state, barotropic_gyre, barotropic_double_gyre, etc.
    grid_type: str          # cubed_sphere, latlon, mpas, spectral
    resolution: str         # C24, 36x72, ico3, T21
    duration_days: float
    quick_days: float
    run_kwargs: dict = field(default_factory=dict)

    # Rest-state variants are grouped under a single rest_state/ folder
    _REST_STATE_GROUP = {
        "rest_state_stratified_with_land",
        "rest_state_uniform_with_land",
        "rest_state_stratified_no_land",
        "rest_state_uniform_no_land",
    }

    @property
    def output_path(self) -> str:
        if self.case in self._REST_STATE_GROUP:
            return f"rest_state/{self.case}/{self.grid_type}/{self.resolution}"
        return f"{self.case}/{self.grid_type}/{self.resolution}"


# ===========================================================================
# Test matrix generation
# ===========================================================================

def _build_test_matrix() -> list[TestCase]:
    """Generate the full test matrix from grid x case."""
    matrix: list[TestCase] = []
    res = GRID_RESOLUTIONS

    # --- Rest state adjustment (with land): all grids except spectral ---
    for g in GRID_TYPES:
        if g == "spectral":
            continue
        matrix.append(TestCase(
            "rest_state_stratified_with_land", g, res[g], 1.0, 0.1))

    # --- Rest state with uniform T/S (with land): isolates barotropic PGF ---
    for g in GRID_TYPES:
        if g == "spectral":
            continue
        matrix.append(TestCase(
            "rest_state_uniform_with_land", g, res[g], 1.0, 0.1))

    # --- Rest state adjustment without land: all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "rest_state_stratified_no_land", g, res[g], 1.0, 0.1))

    # --- Rest state uniform T/S without land: control ---
    for g in GRID_TYPES:
        if g == "spectral":
            continue
        matrix.append(TestCase(
            "rest_state_uniform_no_land", g, res[g], 1.0, 0.1))

    # --- Barotropic gravity wave: resolution-matched grids (~384-446 km dx) ---
    bwave_res = {"cubed_sphere": "C24", "latlon": "48x72",
                 "mpas": "ico4", "spectral": "T21"}
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "barotropic_wave", g, bwave_res[g], 2.0, 0.2))

    # --- Wind-driven regional barotropic double gyre: regional grids ---
    # cs_regional excluded: ocean init assumes 6-face arrays (TODO: adapt)
    for g in ["mpas_regional", "latlon_regional"]:
        matrix.append(TestCase(
            "barotropic_double_gyre", g, res[g], 30.0, 2.0))

    # sin² wind variant (5° edge taper) — exercises the
    # ``wind_buffer_deg`` path in PrescribedForcingConfig that is not
    # otherwise covered by the cosine-wind double-gyre case above.
    for g in ["mpas_regional", "latlon_regional"]:
        matrix.append(TestCase(
            "barotropic_double_gyre_sin2", g, res[g], 30.0, 2.0))

    # --- Wind-driven regional baroclinic gyre: regional grids ---
    # Tests Coriolis double-counting fix (#103) with realistic stratification
    for g in ["mpas_regional", "latlon_regional"]:
        matrix.append(TestCase(
            "baroclinic_gyre", g, res[g], 60.0, 5.0))

    # Cosine-wind / no-edge-taper variant — pairs with the default
    # sin² baroclinic_gyre (BaroclinicGyreConfig default) to compare
    # the effect of the edge-buffer treatment on the western boundary
    # current and gyre asymmetry.
    for g in ["mpas_regional", "latlon_regional"]:
        matrix.append(TestCase(
            "baroclinic_gyre_cos", g, res[g], 60.0, 5.0))

    # --- Global barotropic wind-driven: latlon, mpas ---
    # (cubed_sphere excluded — face-boundary instability produces unphysical speeds)
    for g in ["latlon", "mpas"]:
        matrix.append(TestCase(
            "global_barotropic_wind", g, res[g], 60.0, 5.0))

    # --- Global barotropic wind, single layer (truly barotropic) ---
    # 1-level eliminates vertical coupling issues; wind and bottom drag
    # act on the same layer → clean Sverdrup-like equilibrium.
    for g in ["latlon", "mpas"]:
        matrix.append(TestCase(
            "global_barotropic_wind_1lev", g, res[g], 60.0, 5.0,
            run_kwargs={"nlev": 1}))

    # --- Geostrophic adjustment: all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "geostrophic_adjustment", g, res[g], 10.0, 1.0))

    # --- Phillips two-layer baroclinic: all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "phillips_two_layer", g, res[g], 10.0, 1.0))

    # --- Inertia-Gravity Wave (Bishnu et al. 2024): all grids ---
    for g in GRID_TYPES:
        matrix.append(TestCase(
            "inertia_gravity_wave", g, res[g], 2.0, 0.2))

    # --- Lock Exchange (NEMO / Petersen et al. 2015): cubed_sphere, latlon ---
    for g in ["cubed_sphere", "latlon"]:
        matrix.append(TestCase(
            "lock_exchange", g, res[g], 1.0, 0.1))

    # --- Overflow (NEMO / Petersen et al. 2015): cubed_sphere, latlon ---
    for g in ["cubed_sphere", "latlon"]:
        matrix.append(TestCase(
            "overflow", g, res[g], 0.5, 0.1))

    # --- Stommel Gyre Tracer (Hecht et al. 2000): cubed_sphere, latlon, mpas ---
    for g in ["cubed_sphere", "latlon", "mpas"]:
        matrix.append(TestCase(
            "stommel_gyre_tracer", g, res[g], 60.0, 5.0))

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
    """Scalar drift wrapper.

    iter-90 (codex review HIGH-1): the previous inline implementation
    used ``max(abs(values[0]), 1e-30)`` — the same iter-78/80
    pathology that was already fixed in
    ``run_atmosphere_test_matrix.py:_compute_drift`` (iter-83) and
    ``run_held_suarez_rrtmgp_4grids.py`` (iter-87) and factored into
    ``legoesm.diagnostics.conservation_drift`` (iter-88).  This
    function was missed in the iter-88 refactor; iter-90 routes it
    through the shared helper so all 10 ocean callsites
    (``T_drift``, ``PE_drift``, ``S_integral_drift``) inherit the
    1.0 floor.
    """
    from legoesm.diagnostics.conservation_drift import compute_relative_drift
    return compute_relative_drift(values)


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
                # iter-105 (codex iter-104 MEDIUM-3): mirror the
                # iter-98 atmosphere matrix BLOWUP-info fix.
                # Without this, results.txt would emit notes
                # derived from the last *clean* diagnostic
                # (same false-improvement risk as iter-96 OMIP).
                diag["_blowup_info"] = {
                    "step": step,
                    "day": step * dt / 86400.0,
                    "metric": float(metric),
                    "is_finite": bool(is_finite),
                    "threshold": float(blowup_threshold),
                    "reason": (
                        "state non-finite (NaN/Inf)" if not is_finite
                        else f"metric {float(metric):.1f} > "
                             f"threshold {float(blowup_threshold):.1f}"
                    ),
                }
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
    max_dist: float | None = None,
    target_lat: np.ndarray | None = None,
    target_lon: np.ndarray | None = None,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build KDTree interpolation weights from unstructured to lat-lon grid.

    Returns (idxs, weights) arrays of shape (n_lat*n_lon, K) for K-nearest-
    neighbor inverse-distance weighting in 3-D Cartesian coordinates.

    Parameters
    ----------
    max_dist : float, optional
        Maximum 3-D Cartesian distance (on the unit sphere) for a valid
        neighbour.  Target points whose nearest source cell is farther
        than this get zero weight and will produce NaN after
        ``_apply_weights``.  Prevents extrapolation artefacts in
        regional meshes.
    ocean_mask : array, optional
        Boolean-like mask where >0.5 means ocean.  When provided, only
        ocean cells are included in the KDTree so land values never
        contaminate interpolated ocean fields.  Returned ``idxs`` refer
        to the *original* (unmasked) array so ``_apply_weights`` works
        unchanged.
    """
    lon = np.asarray(lon_deg, dtype=np.float64).ravel() % 360
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64).ravel(), -90, 90)
    d2r = np.pi / 180.0
    src_all = np.column_stack([
        np.cos(lat * d2r) * np.cos(lon * d2r),
        np.cos(lat * d2r) * np.sin(lon * d2r),
        np.sin(lat * d2r)])

    # When an ocean mask is provided, build the tree from ocean cells only
    # but map indices back to the full array for _apply_weights.
    if ocean_mask is not None:
        omask = np.asarray(ocean_mask, dtype=np.float64).ravel() > 0.5
        ocean_idx = np.where(omask)[0]
        if ocean_idx.size == 0:
            # All land — return zero weights
            n_tgt = n_lat * n_lon
            return (np.zeros((n_tgt, 1), dtype=int),
                    np.zeros((n_tgt, 1), dtype=np.float64))
        src = src_all[ocean_idx]
    else:
        ocean_idx = None
        src = src_all

    # Use target grid if provided
    if target_lat is not None and target_lon is not None:
        lat_1d = target_lat
        lon_1d = target_lon
        n_lat, n_lon = len(lat_1d), len(lon_1d)
    else:
        lat_1d = np.linspace(-90.0, 90.0, n_lat)
        lon_1d = np.linspace(0.0, 360.0, n_lon)
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

    # Map indices back to the full (unmasked) array
    if ocean_idx is not None:
        idxs = ocean_idx[idxs]

    w = 1.0 / np.maximum(dists, 1e-12)
    # Zero out weights for target points too far from any source cell.
    if max_dist is not None:
        too_far = dists[:, 0] > max_dist
        w[too_far] = 0.0
    w_sum = w.sum(axis=1, keepdims=True)
    w = np.where(w_sum > 0, w / np.maximum(w_sum, 1e-30), 0.0)
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


def _build_voronoi_polygons(mesh) -> tuple[list, np.ndarray]:
    """Pre-compute Voronoi cell polygons from an MPAS VoronoiMesh.

    Returns
    -------
    polygons : list of (nv, 2) arrays
        Each polygon is an array of (lon_deg, lat_deg) vertices.
    cell_indices : int array, shape (n_polygons,)
        The cell index for each polygon (some cells may be skipped
        if vertex connectivity is invalid).
    """
    lat_v = np.degrees(np.asarray(mesh.latVertex, dtype=np.float64))
    lon_v = np.degrees(np.asarray(mesh.lonVertex, dtype=np.float64))
    verts_on_cell = np.asarray(mesh.verticesOnCell, dtype=int)  # (maxEdges, nCells)
    n_edges = np.asarray(mesh.nEdgesOnCell, dtype=int)          # (nCells,)

    polygons = []
    cell_indices = []
    for i in range(mesh.nCells):
        nv = int(n_edges[i])
        if nv < 3:
            continue
        vidx = verts_on_cell[:nv, i]
        if np.any(vidx < 0):
            continue
        poly = np.column_stack([lon_v[vidx], lat_v[vidx]])
        polygons.append(poly)
        cell_indices.append(i)
    return polygons, np.array(cell_indices, dtype=int)


def _plot_voronoi_field(ax, mesh, field: np.ndarray,
                        land_mask: np.ndarray | None = None,
                        cmap: str = "RdYlBu_r",
                        vmin: float | None = None,
                        vmax: float | None = None):
    """Plot a cell-centered field on the native Voronoi mesh.

    Uses matplotlib PolyCollection — no interpolation, so there are
    zero land-bleed artifacts.

    Parameters
    ----------
    ax : matplotlib Axes
    mesh : VoronoiMesh
    field : array, shape (nCells,)
    land_mask : array, shape (nCells,), optional
        >0.5 means ocean.  Land cells drawn in light gray.
    cmap, vmin, vmax : colormap parameters for ocean cells.

    Returns
    -------
    pc : PolyCollection for the ocean cells (for colorbar).
    """
    from matplotlib.collections import PolyCollection

    polygons, cell_idx = _build_voronoi_polygons(mesh)
    values = np.asarray(field, dtype=np.float64).ravel()

    if land_mask is not None:
        mask = np.asarray(land_mask, dtype=np.float64).ravel()
        ocean_polys, ocean_vals = [], []
        land_polys = []
        for poly, ci in zip(polygons, cell_idx):
            if mask[ci] > 0.5:
                ocean_polys.append(poly)
                ocean_vals.append(values[ci])
            else:
                land_polys.append(poly)
        # Draw land cells
        if land_polys:
            land_pc = PolyCollection(land_polys, facecolor="#d9d9d9",
                                     edgecolor="#bfbfbf", linewidth=0.3)
            ax.add_collection(land_pc)
        # Draw ocean cells
        pc = None
        if ocean_polys:
            pc = PolyCollection(ocean_polys, array=np.array(ocean_vals),
                                cmap=cmap, edgecolor="face", linewidth=0.1)
            if vmin is not None and vmax is not None:
                pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
    else:
        vals_arr = values[cell_idx]
        pc = PolyCollection(polygons, array=vals_arr, cmap=cmap,
                            edgecolor="face", linewidth=0.1)
        if vmin is not None and vmax is not None:
            pc.set_clim(vmin, vmax)
        ax.add_collection(pc)

    ax.autoscale_view()
    ax.set_aspect("equal")
    return pc


def _bin_to_latlon(
    values: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    n_lat: int = 181,
    n_lon: int = 360,
    max_dist: float | None = None,
    target_lat: np.ndarray | None = None,
    target_lon: np.ndarray | None = None,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate unstructured points onto a regular lat-lon grid.

    Parameters
    ----------
    max_dist : float, optional
        If given, target grid points whose nearest source cell is farther
        than this (3-D Cartesian distance on unit sphere) are set to NaN.
        Automatically estimated for regional meshes when not provided.
    ocean_mask : array, optional
        Boolean-like mask (>0.5 = ocean).  When provided, land cells are
        excluded from the KDTree so they cannot contaminate interpolated
        ocean values.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    # Use target grid if provided, otherwise use default dimensions
    if target_lat is not None and target_lon is not None:
        n_lat, n_lon = len(target_lat), len(target_lon)
    if not np.any(np.isfinite(vals)):
        return np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    # Auto-detect regional mesh: if the source points span < 80% of
    # the globe in latitude, apply a distance cutoff to prevent
    # extrapolation artefacts outside the mesh.
    if max_dist is None:
        lat = np.asarray(lat_deg, dtype=np.float64).ravel()
        lat_span = lat.max() - lat.min()
        if lat_span < 0.8 * 180:
            # Regional: max_dist ≈ 3× median cell spacing (on unit sphere)
            lon = np.asarray(lon_deg, dtype=np.float64).ravel()
            d2r = np.pi / 180.0
            # Rough estimate: sqrt(4π / N) gives mean angular cell spacing
            n_pts = len(lat)
            mean_spacing = np.sqrt(
                d2r**2 * lat_span * min(360, lon.max() - lon.min()) / n_pts)
            # Convert angular spacing to 3-D chord distance
            max_dist = 2.0 * np.sin(0.5 * mean_spacing * 3.0)
    idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon,
                                     max_dist=max_dist,
                                     ocean_mask=ocean_mask)
    return _apply_weights(vals, idxs, w, n_lat, n_lon)


def _regrid_land_mask(mask_arr: np.ndarray, lon_deg: np.ndarray,
                     lat_deg: np.ndarray, coord_kind: str,
                     target_lat: np.ndarray | None = None,
                     target_lon: np.ndarray | None = None) -> np.ndarray:
    """Regrid a binary land mask using nearest neighbor interpolation."""
    if coord_kind in ("latlon", "gaussian"):
        return np.asarray(mask_arr, dtype=np.float64)
    
    # For unstructured grids, use nearest neighbor interpolation
    # (cKDTree already imported at module level)

    # Source points (unstructured)
    lon_src = np.asarray(lon_deg, dtype=np.float64).ravel() % 360
    lat_src = np.clip(np.asarray(lat_deg, dtype=np.float64).ravel(), -90, 90)
    mask_src = np.asarray(mask_arr, dtype=np.float64).ravel()
    
    # Target grid
    if target_lat is not None and target_lon is not None:
        n_lat, n_lon = len(target_lat), len(target_lon)
        lat_1d, lon_1d = target_lat, target_lon
    else:
        n_lat, n_lon = 181, 360
        lat_1d = np.linspace(-90.0, 90.0, n_lat)
        lon_1d = np.linspace(0.0, 360.0, n_lon)
    
    # Convert to 3D Cartesian coordinates for accurate distance calculation
    d2r = np.pi / 180.0
    
    # Source points in 3D
    src_3d = np.column_stack([
        np.cos(lat_src * d2r) * np.cos(lon_src * d2r),
        np.cos(lat_src * d2r) * np.sin(lon_src * d2r),
        np.sin(lat_src * d2r)
    ])
    
    # Target points in 3D  
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    tgt_3d = np.column_stack([
        np.cos(lat_2d.ravel() * d2r) * np.cos(lon_2d.ravel() * d2r),
        np.cos(lat_2d.ravel() * d2r) * np.sin(lon_2d.ravel() * d2r),
        np.sin(lat_2d.ravel() * d2r)
    ])
    
    # Build KDTree and find nearest neighbors
    tree = cKDTree(src_3d)
    distances, indices = tree.query(tgt_3d, k=1)
    
    # Get mask values at nearest neighbors
    mask_interp = mask_src[indices]
    
    # For land mask, apply threshold to ensure binary values
    mask_interp = np.where(mask_interp > 0.5, 1.0, 0.0)
    
    return mask_interp.reshape(n_lat, n_lon)


def _regrid_2d(field_arr: np.ndarray, lon_deg: np.ndarray,
               lat_deg: np.ndarray, coord_kind: str,
               target_lat: np.ndarray | None = None,
               target_lon: np.ndarray | None = None,
               ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """Regrid a 2D field to target lat-lon grid (default 181x360).

    Parameters
    ----------
    ocean_mask : array, optional
        Boolean-like mask (>0.5 = ocean).  For unstructured grids, land
        cells are excluded from the interpolation KDTree.
    """
    if coord_kind in ("latlon", "gaussian"):
        return np.asarray(field_arr, dtype=np.float64)
    # Cubed-sphere: use face-aware bilinear interpolation (no edge artifacts).
    if coord_kind == "cube":
        from legoesm.grids.regridding import (
            apply_cubedsphere_to_latlon, get_cubedsphere_to_latlon_weights)
        arr = np.asarray(field_arr, dtype=np.float64)
        if arr.ndim >= 3 and arr.shape[0] == 6:
            n = arr.shape[1]
        else:
            n = int(round(np.sqrt(arr.size / 6)))
            arr = arr.reshape(6, n, n)
        w = get_cubedsphere_to_latlon_weights(n)
        return apply_cubedsphere_to_latlon(arr, w)
    return _bin_to_latlon(field_arr.ravel(), lon_deg.ravel(), lat_deg.ravel(),
                          target_lat=target_lat, target_lon=target_lon,
                          ocean_mask=ocean_mask)


def _regrid_3d_level(field_3d: np.ndarray, lon_deg: np.ndarray,
                     lat_deg: np.ndarray, coord_kind: str,
                     target_lat: np.ndarray | None = None,
                     target_lon: np.ndarray | None = None,
                     ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """Regrid a 3D field (*, nlev) to target lat-lon grid (default 181x360x nlev)."""
    arr = np.asarray(field_3d, dtype=np.float64)
    if coord_kind in ("latlon", "gaussian"):
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr
    # Cubed-sphere: use face-aware bilinear interpolation.
    if coord_kind == "cube":
        from legoesm.grids.regridding import (
            apply_cubedsphere_to_latlon_3d, get_cubedsphere_to_latlon_weights)
        if arr.ndim >= 3 and arr.shape[0] == 6:
            n = arr.shape[1]
        else:
            nlev = arr.shape[-1]
            n = int(round(np.sqrt(arr.size / (6 * nlev))))
            arr = arr.reshape(6, n, n, nlev)
        w = get_cubedsphere_to_latlon_weights(n)
        return apply_cubedsphere_to_latlon_3d(arr, w)
    if arr.ndim == 1:
        arr = arr[:, None]
    nlev = arr.shape[-1]
    if target_lat is not None and target_lon is not None:
        n_lat, n_lon = len(target_lat), len(target_lon)
    else:
        n_lat, n_lon = 181, 360
    flat = arr.reshape(-1, nlev)
    # Auto-detect regional mesh and apply distance cutoff
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    max_dist = None
    if lat.max() - lat.min() < 0.8 * 180:
        lon = np.asarray(lon_deg, dtype=np.float64).ravel()
        d2r = np.pi / 180.0
        n_pts = len(lat)
        mean_spacing = np.sqrt(
            d2r**2 * (lat.max() - lat.min())
            * min(360, lon.max() - lon.min()) / n_pts)
        max_dist = 2.0 * np.sin(0.5 * mean_spacing * 3.0)
    idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon,
                                     max_dist=max_dist, target_lat=target_lat,
                                     target_lon=target_lon,
                                     ocean_mask=ocean_mask)
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

def _write_results_txt(output_dir: Path, rows: dict[str, Any],
                       *, diag: dict | None = None,
                       blowup_info: dict | None = None):
    """Write results.txt for an ocean-matrix-runner case.

    iter-105 (codex iter-104 MEDIUM-3): added ``diag`` and
    ``blowup_info`` kwargs mirroring the iter-98 atmosphere
    matrix fix.  When ``diag`` is passed, ``_blowup_info`` is
    auto-extracted from it.  When BLOWUP info is present and
    ``rows.get("status") == "FAIL"``, the function prepends a
    BLOWUP marker to the ``notes`` field so a reader of
    ``results.txt`` sees the failure mode unambiguously
    instead of last-clean-diagnostic notes.
    """
    if diag is not None and blowup_info is None:
        blowup_info = diag.get("_blowup_info")
    output_dir.mkdir(parents=True, exist_ok=True)
    if blowup_info is not None and rows.get("status") == "FAIL":
        original_notes = rows.get("notes", "")
        blowup_str = (
            f"BLOWUP at step {blowup_info['step']} "
            f"(day {blowup_info.get('day', 0):.2f}), "
            f"reason: {blowup_info['reason']}"
        )
        if original_notes:
            rows = {**rows,
                    "notes": f"{blowup_str}; last clean: {original_notes}"}
        else:
            rows = {**rows, "notes": blowup_str}
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
    # iter-88: the iter-80 baseline-zero floor logic (originally
    # written inline here as ``_MIN_RELATIVE_BASELINE = 1.0`` and
    # ``vol_denom = max(abs(vol[0]), _MIN_RELATIVE_BASELINE)``) now
    # lives in ``legoesm.diagnostics.conservation_drift`` so that
    # the same floor convention is shared by the atmosphere,
    # ocean, and HS+RRTMGP cross-grid drivers.  See that module's
    # docstring for the full iter-78/80/83/87 history.
    #
    # iter-81 codex MEDIUM: the column name ``vol_rel`` is
    # unit-ambiguous (relative when baseline ≥ 1, absolute when
    # < 1).  This is acceptable for the cross-variant comparison
    # plot's purposes (both forms convey "is the system drifting
    # away from initial state?") but downstream consumers reading
    # these columns should treat them as drift magnitudes, NOT as
    # dimensionless relative deviations.
    from legoesm.diagnostics.conservation_drift import relative_drift_series
    vol_rel = relative_drift_series(vol)
    heat_rel = relative_drift_series(heat)

    n_panels = 2
    has_salt = len(salt_vals) == len(times)
    if has_salt:
        salt = np.array(salt_vals, dtype=np.float64)
        salt_rel = relative_drift_series(salt)
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


def _extract_test_case_name(case_name: str) -> str:
    """Extract test case name from full case name (e.g. 'Rest State spectral T21' -> 'rest_state')."""
    case_lower = case_name.lower()
    for test_case in FIELD_RANGES.keys():
        test_case_words = test_case.replace('_', ' ')
        if test_case_words in case_lower:
            return test_case
    # Fallback to first word if no match
    return case_lower.split()[0].replace(' ', '_')

def _save_snapshot_plots(output_dir: Path, case_name: str, snapshots: dict,
                         dt: float, field_specs: list[tuple[str, str, str]],
                         coord_kind: str, lon_deg: np.ndarray,
                         lat_deg: np.ndarray,
                         domain_extent: tuple[float,float,float,float] | None = None,
                         mesh=None):
    """Save snapshot evolution plots for each 2D field.

    Parameters
    ----------
    mesh : VoronoiMesh, optional
        When provided and coord_kind is ``"mpas"``, snapshot plots use
        native Voronoi polygon rendering (PolyCollection) instead of
        regrid-then-imshow, eliminating land-bleed interpolation artifacts.
    """
    if not snapshots:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_steps = sorted(snapshots.keys())
    first_saved = None
    
    # Extract test case for consistent field ranges
    test_case = _extract_test_case_name(case_name)
    field_ranges = FIELD_RANGES.get(test_case, {})

    # Use native Voronoi polygon rendering for MPAS grids when mesh is
    # available.  This eliminates all interpolation artifacts at land
    # boundaries.  (Falls back to regrid+imshow when mesh is not provided.)
    use_native_voronoi = (mesh is not None
                          and coord_kind in ("mpas", "mpas_regional"))

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

        if use_native_voronoi:
            # --- Native Voronoi polygon path (MPAS) -----------------------
            for idx, step in enumerate(steps):
                r, c = divmod(idx, n_cols)
                ax = axes[r, c]
                raw = np.asarray(snapshots[step][field_key], dtype=np.float64)
                lm = (np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
                      if "land_mask" in snapshots[step] else None)
                # Per-panel color range: use explicit range if configured,
                # otherwise fit to this panel's ocean data.
                if field_key in field_ranges and field_ranges[field_key] != (None, None):
                    vmin, vmax = field_ranges[field_key]
                else:
                    ocean_vals = raw.ravel()
                    if lm is not None:
                        ocean_vals = ocean_vals[lm.ravel() > 0.5]
                    finite = ocean_vals[np.isfinite(ocean_vals)]
                    vmin = float(finite.min()) if finite.size else None
                    vmax = float(finite.max()) if finite.size else None
                im = _plot_voronoi_field(
                    ax, mesh, raw, land_mask=lm, cmap=cmap,
                    vmin=vmin, vmax=vmax)
                if domain_extent is not None:
                    ax.set_xlim(domain_extent[0], domain_extent[1])
                    ax.set_ylim(domain_extent[2], domain_extent[3])
                if im is not None:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                day = step * dt / 86400.0
                ax.set_title(f"t={day:.2f} d", fontsize=9)
                if c == 0:
                    ax.set_ylabel("Latitude")
                if r == n_rows - 1:
                    ax.set_xlabel("Longitude")
        else:
            # --- Regrid + imshow path (latlon, cubed-sphere, fallback) ----
            # Pre-compute regridded fields for all steps
            all_regridded = []
            for step in steps:
                raw = np.asarray(snapshots[step][field_key], dtype=np.float64)
                # Use mask-aware IDW: pass ocean_mask so land cells are
                # excluded from the KDTree and cannot contaminate ocean values.
                land_mask_raw = None
                if "land_mask" in snapshots[step]:
                    land_mask_raw = np.asarray(snapshots[step]["land_mask"],
                                                dtype=np.float64)
                regridded = _regrid_2d(raw, lon_deg, lat_deg, coord_kind,
                                       ocean_mask=land_mask_raw)
                # Still apply land mask on the regridded grid for display
                if land_mask_raw is not None:
                    land_mask_reg = _regrid_land_mask(
                        land_mask_raw, lon_deg, lat_deg, coord_kind)
                    regridded = np.where(land_mask_reg > 0.5, regridded, np.nan)
                all_regridded.append(regridded)

            for idx, step in enumerate(steps):
                r, c = divmod(idx, n_cols)
                ax = axes[r, c]
                regridded = all_regridded[idx]

                # Per-panel color range
                if field_key in field_ranges and field_ranges[field_key] != (None, None):
                    vmin, vmax = field_ranges[field_key]
                else:
                    panel_finite = regridded.ravel()
                    panel_finite = panel_finite[np.isfinite(panel_finite)]
                    if len(panel_finite) > 0:
                        vmin, vmax = float(panel_finite.min()), float(panel_finite.max())
                    else:
                        vmin, vmax = None, None

                # Compute extent from actual coordinates.
                # When domain_extent is given (regional experiments), use it
                # directly.  Otherwise infer from coordinate arrays or the
                # non-NaN bounding box of the regridded field.
                if domain_extent is not None:
                    # domain_extent = (lon_west, lon_east, lat_south, lat_north)
                    lon_ext = [domain_extent[0], domain_extent[1]]
                    lat_ext = [domain_extent[2], domain_extent[3]]
                    if coord_kind not in ("latlon", "gaussian"):
                        lat_1d = np.linspace(-90, 90, regridded.shape[0])
                        lon_1d = np.linspace(0, 360, regridded.shape[1])
                        r0 = max(int(np.searchsorted(lat_1d, lat_ext[0])) - 1, 0)
                        r1 = min(int(np.searchsorted(lat_1d, lat_ext[1])) + 2,
                                 len(lat_1d))
                        c0 = max(int(np.searchsorted(lon_1d, lon_ext[0])) - 1, 0)
                        c1 = min(int(np.searchsorted(lon_1d, lon_ext[1])) + 2,
                                 len(lon_1d))
                        plot_data = regridded[r0:r1, c0:c1]
                        lon_ext = [float(lon_1d[c0]),
                                   float(lon_1d[min(c1, len(lon_1d)-1)])]
                        lat_ext = [float(lat_1d[r0]),
                                   float(lat_1d[min(r1, len(lat_1d)-1)])]
                    else:
                        plot_data = regridded
                elif coord_kind in ("latlon", "gaussian"):
                    lon_flat = np.asarray(lon_deg, dtype=np.float64).ravel()
                    lat_flat = np.asarray(lat_deg, dtype=np.float64).ravel()
                    lon_ext = [float(lon_flat.min()), float(lon_flat.max())]
                    lat_ext = [float(lat_flat.min()), float(lat_flat.max())]
                    plot_data = regridded
                else:
                    lat_1d = np.linspace(-90, 90, regridded.shape[0])
                    lon_1d = np.linspace(0, 360, regridded.shape[1])
                    valid = np.isfinite(regridded)
                    if valid.any():
                        rows = np.where(valid.any(axis=1))[0]
                        cols = np.where(valid.any(axis=0))[0]
                        r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, len(lat_1d))
                        c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, len(lon_1d))
                        plot_data = regridded[r0:r1, c0:c1]
                        lon_ext = [float(lon_1d[c0]), float(lon_1d[min(c1, len(lon_1d)-1)])]
                        lat_ext = [float(lat_1d[r0]), float(lat_1d[min(r1, len(lat_1d)-1)])]
                    else:
                        plot_data = regridded
                        lon_ext = [0, 360]
                        lat_ext = [-90, 90]
                im = ax.imshow(
                    plot_data, origin="lower", aspect="auto", cmap=cmap,
                    extent=[lon_ext[0], lon_ext[1], lat_ext[0], lat_ext[1]],
                    vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Add velocity vectors for circulation visualization
                if field_key in ("SST", "speed_sfc") and "u_sfc" in snapshots[step] and "v_sfc" in snapshots[step]:
                    u_raw = np.asarray(snapshots[step]["u_sfc"], dtype=np.float64)
                    v_raw = np.asarray(snapshots[step]["v_sfc"], dtype=np.float64)
                    # Iter-19 fix: latlon C-grid stores u on east-west
                    # faces (n_lat, n_lon+1) and v on north-south faces
                    # (n_lat+1, n_lon).  Average both to cell centres
                    # before regridding so the quiver overlay doesn't
                    # try to broadcast incompatible shapes.  Mirrors
                    # the same averaging done in
                    # ``_extract_latlon_cgrid_ocean`` for ``speed_sfc``.
                    # iter-21 codex review MEDIUM: removed the silent
                    # ``[:ny, :nx]`` clamp that masked unexpected
                    # shape mismatches.  Now we ONLY apply the
                    # canonical face→cell averagings; if the post-
                    # averaging shapes still differ, skip the quiver
                    # overlay rather than silently truncate.
                    if u_raw.shape != v_raw.shape:
                        if u_raw.shape[1] == v_raw.shape[1] + 1:
                            u_raw = 0.5 * (u_raw[:, :-1] + u_raw[:, 1:])
                        if v_raw.shape[0] == u_raw.shape[0] + 1:
                            v_raw = 0.5 * (v_raw[:-1, :] + v_raw[1:, :])
                    if u_raw.shape != v_raw.shape:
                        # Non-canonical mismatch — skip quiver overlay
                        # rather than truncate-and-mislead.  Earlier
                        # logic silently dropped the high-index edge.
                        continue
                    _lm = (np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
                            if "land_mask" in snapshots[step] else None)
                    u_reg = _regrid_2d(u_raw, lon_deg, lat_deg, coord_kind,
                                       ocean_mask=_lm)
                    v_reg = _regrid_2d(v_raw, lon_deg, lat_deg, coord_kind,
                                       ocean_mask=_lm)

                    if domain_extent is not None and coord_kind not in ("latlon", "gaussian"):
                        lat_1d = np.linspace(-90, 90, u_reg.shape[0])
                        lon_1d = np.linspace(0, 360, u_reg.shape[1])
                        r0 = max(int(np.searchsorted(lat_1d, lat_ext[0])) - 1, 0)
                        r1 = min(int(np.searchsorted(lat_1d, lat_ext[1])) + 2, len(lat_1d))
                        c0 = max(int(np.searchsorted(lon_1d, lon_ext[0])) - 1, 0)
                        c1 = min(int(np.searchsorted(lon_1d, lon_ext[1])) + 2, len(lon_1d))
                        u_plot = u_reg[r0:r1, c0:c1]
                        v_plot = v_reg[r0:r1, c0:c1]
                        lat_plot = lat_1d[r0:r1]
                        lon_plot = lon_1d[c0:c1]
                    else:
                        u_plot = u_reg
                        v_plot = v_reg
                        lat_plot = np.linspace(lat_ext[0], lat_ext[1], u_reg.shape[0])
                        lon_plot = np.linspace(lon_ext[0], lon_ext[1], u_reg.shape[1])

                    skip = 4
                    X, Y = np.meshgrid(lon_plot[::skip], lat_plot[::skip])
                    U = u_plot[::skip, ::skip]
                    V = v_plot[::skip, ::skip]

                    mask = np.isfinite(U) & np.isfinite(V) & ((np.abs(U) + np.abs(V)) > 1e-6)
                    if np.any(mask):
                        speed = np.sqrt(U[mask]**2 + V[mask]**2)
                        max_speed = np.nanmax(speed) if len(speed) > 0 else 0.01
                        scale = max_speed * 100

                        ax.quiver(X[mask], Y[mask], U[mask], V[mask],
                                 color='white', alpha=0.8, scale=scale, scale_units='xy',
                                 width=0.003, headwidth=4, headlength=6,
                                 edgecolors='black', linewidth=0.5)

                day = step * dt / 86400.0
                ax.set_title(f"t={day:.2f} d", fontsize=9)
                if c == 0:
                    ax.set_ylabel("Latitude")
                if r == n_rows - 1:
                    ax.set_xlabel("Longitude")

        for idx in range(len(steps), n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].set_visible(False)

        fig.suptitle(f"{case_name} — {field_key} ({field_label})", fontsize=11)
        fig.tight_layout()
        fname = f"snapshots_{field_key}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if first_saved is None:
            first_saved = fname

    # Alias first field as field_snapshots.png
    if first_saved and (output_dir / first_saved).exists():
        shutil.copy2(output_dir / first_saved,
                     output_dir / "field_snapshots.png")


def _bin_cross_section(
    f3d: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    coord_kind: str,
    mean_axis: int,
    n_lat: int = 181,
    n_lon: int = 360,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a cross-section by direct binning (no interpolation).

    For regular grids (latlon, gaussian): regrid then nanmean as before.
    For unstructured grids (cube, mpas): bin source points directly into
    latitude or longitude bins with adaptive bin count ensuring >= ~10
    points per bin on average, avoiding IDW interpolation artifacts.

    Parameters
    ----------
    mean_axis : int
        0 = average over latitude → longitude-vertical section
        1 = average over longitude → latitude-vertical section
    ocean_mask : array, optional
        Boolean-like mask (>0.5 = ocean).  Excludes land cells from KDTree.

    Returns
    -------
    section : np.ndarray, shape (n_bins, nlev)
    bin_centers : np.ndarray, shape (n_bins,) — for plotting extents
    """
    # Regrid to regular lat-lon, then average over the requested axis.
    # Use k=20 neighbors for unstructured grids to avoid aliasing from
    # cubed-sphere face boundaries or icosahedral grid structure.
    if coord_kind not in ("latlon", "gaussian"):
        arr = np.asarray(f3d, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        flat = arr.reshape(-1, arr.shape[-1])
        nlev = flat.shape[1]
        idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon, k=20,
                                         ocean_mask=ocean_mask)
        ll = np.full((n_lat, n_lon, nlev), np.nan, dtype=np.float64)
        for lev in range(nlev):
            ll[..., lev] = _apply_weights(flat[:, lev], idxs, w, n_lat, n_lon)
    else:
        ll = _regrid_3d_level(f3d, lon_deg, lat_deg, coord_kind)

    section = np.nanmean(ll, axis=mean_axis)
    if mean_axis == 0:
        return section, np.linspace(0, 360, section.shape[0])
    return section, np.linspace(-90, 90, section.shape[0])


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

    for fname, mean_axis, xlabel in [
        ("latitude_vertical_cross_sections.png", 1, "Latitude"),
        ("longitude_vertical_cross_sections.png", 0, "Longitude"),
    ]:
        nc = len(valid_steps)
        fig, axes_arr = plt.subplots(
            1, nc, figsize=(4.5 * nc, 5), sharey=True)
        if nc == 1:
            axes_arr = [axes_arr]
        im = None

        # Pre-compute all cross-sections and find shared color range
        all_sections = []
        all_bin_centers = []
        for step in valid_steps:
            f3d = np.asarray(snapshots[step][field_3d_key], dtype=np.float64)
            _lm = None
            if "land_mask" in snapshots[step]:
                _lm = np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
                land_mask_3d = _lm[..., np.newaxis]
                f3d = np.where(land_mask_3d > 0.5, f3d, np.nan)
            section, bin_centers = _bin_cross_section(
                f3d, lon_deg, lat_deg, coord_kind, mean_axis,
                ocean_mask=_lm)
            section = _fill_nan_section(section)
            all_sections.append(section)
            all_bin_centers.append(bin_centers)

        all_vals = np.concatenate([s.ravel() for s in all_sections])
        all_finite = all_vals[np.isfinite(all_vals)]
        if len(all_finite) > 0:
            cs_vmin, cs_vmax = float(np.nanmin(all_finite)), float(np.nanmax(all_finite))
        else:
            cs_vmin, cs_vmax = None, None

        # Compute level interfaces for pcolormesh (accurate vertical grid representation)
        def compute_level_interfaces(level_centers):
            """Compute interface depths from level center depths."""
            if len(level_centers) == 1:
                return np.array([0.0, 2 * level_centers[0]])
            
            # Compute interfaces as midpoints between level centers
            interfaces = np.zeros(len(level_centers) + 1)
            interfaces[0] = 0.0  # Surface
            interfaces[1:-1] = 0.5 * (level_centers[:-1] + level_centers[1:])
            interfaces[-1] = level_centers[-1] + (level_centers[-1] - interfaces[-2])
            return interfaces

        level_interfaces = compute_level_interfaces(levels)
        
        for ax, step, section, bin_centers in zip(
                axes_arr, valid_steps, all_sections, all_bin_centers):
            # Use pcolormesh to show true model grid structure instead of imshow
            # This accurately represents the variable vertical grid spacing
            bin_interfaces = np.linspace(bin_centers[0] - 0.5 * (bin_centers[1] - bin_centers[0]),
                                       bin_centers[-1] + 0.5 * (bin_centers[-1] - bin_centers[-2]),
                                       len(bin_centers) + 1)
            X, Y = np.meshgrid(bin_interfaces, level_interfaces)
            im = ax.pcolormesh(
                X, Y, section.T, cmap="RdBu_r", shading='flat',
                vmin=cs_vmin, vmax=cs_vmax)
            
            day = step * dt / 86400.0
            ax.set_title(f"t={day:.2f} d", fontsize=9)
            ax.set_xlabel(xlabel)
            ax.invert_yaxis()  # Depth increases downward

        axes_arr[0].set_ylabel(level_label)
        if im is not None:
            fig.colorbar(
                im, ax=axes_arr, orientation="vertical", fraction=0.046,
                pad=0.04, label=field_3d_key)
        fig.suptitle(
            f"{case_name} — {field_3d_key} cross-sections", fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.88, 0.95])  # More space for colorbar
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
        
        # Apply land masking to 3D field before computing profile if available
        if "land_mask" in snapshots[step]:
            land_mask_raw = np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
            land_mask_3d = land_mask_raw[..., np.newaxis]  # Expand to 3D
            f3d = np.where(land_mask_3d > 0.5, f3d, np.nan)
        
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


def _save_velocity_profiles(output_dir: Path, case_name: str,
                            snapshots: dict, dt: float,
                            levels: np.ndarray, level_label: str = "Depth (m)"):
    """Save vertical velocity profile diagnostics (mean, max, structure).

    Requires snapshots to contain 'u_3d', 'v_3d', 'speed_3d' keys
    (set include_velocity_3d=True in _make_extract_fn).
    """
    valid_steps = sorted(
        s for s in snapshots
        if "u_3d" in snapshots[s] and "v_3d" in snapshots[s])
    if not valid_steps:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    n_snaps = len(valid_steps)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, n_snaps))

    def _mask_3d(snap):
        mask = snap["land_mask"]
        if mask.ndim == 1:  # MPAS: (nCells,)
            return mask[:, np.newaxis]
        return mask[..., np.newaxis]  # latlon: (n_lat, n_lon)

    def _horiz_axes(m3d):
        return tuple(range(m3d.ndim - 1))

    # ── 1. Domain-mean |u|, |v|, speed vs depth ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    for idx, step in enumerate(valid_steps):
        snap = snapshots[step]
        m3d = _mask_3d(snap)
        hax = _horiz_axes(m3d)
        ocean_count = np.maximum(np.sum(m3d > 0.5, axis=hax), 1)
        day = step * dt / 86400.0
        lbl = f"day {day:.1f}"

        mean_abs_u = np.sum(np.abs(snap["u_3d"]) * m3d, axis=hax) / ocean_count
        mean_abs_v = np.sum(np.abs(snap["v_3d"]) * m3d, axis=hax) / ocean_count
        mean_spd = np.sum(snap["speed_3d"] * m3d, axis=hax) / ocean_count

        axes[0].plot(mean_abs_u, levels, color=cmap[idx], label=lbl)
        axes[1].plot(mean_abs_v, levels, color=cmap[idx], label=lbl)
        axes[2].plot(mean_spd, levels, color=cmap[idx], label=lbl)

    axes[0].set_xlabel(r"Mean $|u|$ (m/s)"); axes[0].set_title("Zonal velocity")
    axes[1].set_xlabel(r"Mean $|v|$ (m/s)"); axes[1].set_title("Meridional velocity")
    axes[2].set_xlabel("Mean speed (m/s)"); axes[2].set_title("Speed")
    axes[0].set_ylabel(level_label)
    axes[2].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{case_name} — domain-mean velocity profiles", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_profiles_mean.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 2. Max |u|, |v|, speed vs depth ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    for idx, step in enumerate(valid_steps):
        snap = snapshots[step]
        m3d = _mask_3d(snap)
        hax = _horiz_axes(m3d)
        day = step * dt / 86400.0
        lbl = f"day {day:.1f}"

        max_u = np.max(np.where(m3d > 0.5, np.abs(snap["u_3d"]), 0.0), axis=hax)
        max_v = np.max(np.where(m3d > 0.5, np.abs(snap["v_3d"]), 0.0), axis=hax)
        max_spd = np.max(np.where(m3d > 0.5, snap["speed_3d"], 0.0), axis=hax)

        axes[0].plot(max_u, levels, color=cmap[idx], label=lbl)
        axes[1].plot(max_v, levels, color=cmap[idx], label=lbl)
        axes[2].plot(max_spd, levels, color=cmap[idx], label=lbl)

    axes[0].set_xlabel(r"Max $|u|$ (m/s)"); axes[0].set_title("Zonal velocity")
    axes[1].set_xlabel(r"Max $|v|$ (m/s)"); axes[1].set_title("Meridional velocity")
    axes[2].set_xlabel("Max speed (m/s)"); axes[2].set_title("Speed")
    axes[0].set_ylabel(level_label)
    axes[2].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{case_name} — max velocity profiles", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_profiles_max.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 3. Final-snapshot velocity structure: mean + RMS per level ──
    last_step = valid_steps[-1]
    snap = snapshots[last_step]
    m3d = _mask_3d(snap)
    hax = _horiz_axes(m3d)
    day = last_step * dt / 86400.0
    ocean_count = np.maximum(np.sum(m3d > 0.5, axis=hax), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    mean_u = np.sum(snap["u_3d"] * m3d, axis=hax) / ocean_count
    mean_v = np.sum(snap["v_3d"] * m3d, axis=hax) / ocean_count
    rms_u = np.sqrt(np.sum(snap["u_3d"]**2 * m3d, axis=hax) / ocean_count)
    rms_v = np.sqrt(np.sum(snap["v_3d"]**2 * m3d, axis=hax) / ocean_count)

    axes[0].plot(mean_u, levels, 'b-o', ms=4, label="mean u")
    axes[0].plot(rms_u, levels, 'b--s', ms=4, label="RMS u")
    axes[0].set_xlabel("Velocity (m/s)"); axes[0].set_ylabel(level_label)
    axes[0].set_title(f"Zonal velocity (day {day:.1f})")
    axes[0].legend(); axes[0].axvline(0, color='k', lw=0.5, ls=':')

    axes[1].plot(mean_v, levels, 'r-o', ms=4, label="mean v")
    axes[1].plot(rms_v, levels, 'r--s', ms=4, label="RMS v")
    axes[1].set_xlabel("Velocity (m/s)")
    axes[1].set_title(f"Meridional velocity (day {day:.1f})")
    axes[1].legend(); axes[1].axvline(0, color='k', lw=0.5, ls=':')

    for ax in axes:
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{case_name} — velocity structure (final)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_structure_final.png", dpi=150,
                bbox_inches="tight")
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
    domain_extent: tuple[float, float, float, float] | None = None,
):
    """Save snapshot field arrays as NPZ files with proper time series format.
    
    NEW FORMAT: Each field is saved as a time series array with shape (n_times, ...).
    This replaces the old format where each timestep was a separate variable.
    """
    if not snapshots:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_steps = sorted(snapshots.keys())
    times_days = np.array([s * dt / 86400.0 for s in sorted_steps],
                          dtype=np.float64)

    # Common metadata for both files
    common_metadata = {
        "steps": np.array(sorted_steps, dtype=np.int64),
        "times_days": times_days,
    }

    # Collect all unique field keys across all timesteps
    all_field_keys = set()
    for step_data in snapshots.values():
        all_field_keys.update(step_data.keys())

    # Build time-series arrays for native grid
    native_arrays = dict(common_metadata)
    latlon_arrays = dict(common_metadata)
    # Store source coordinate range so downstream plotters know the
    # actual domain extent (important for regional unstructured meshes
    # where nearest-neighbour regridding bleeds beyond the domain).
    # When domain_extent is provided (regional experiments), use it
    # instead of the mesh cell coordinate range (which can extend
    # beyond the nominal domain for Voronoi meshes).
    src_lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    src_lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    if coord_kind in ("latlon", "gaussian"):
        # Native grids have lon in [0, 360); store actual coordinates
        latlon_arrays["lat"] = src_lat
        latlon_arrays["lon"] = src_lon
    else:
        # Regridded grids (cube, mpas) use [0, 360] target
        latlon_arrays["lat"] = np.linspace(-90.0, 90.0, 181)
        latlon_arrays["lon"] = np.linspace(0.0, 360.0, 360)
    if domain_extent is not None:
        latlon_arrays["source_lon_range"] = np.array(
            [domain_extent[0], domain_extent[1]])
        latlon_arrays["source_lat_range"] = np.array(
            [domain_extent[2], domain_extent[3]])
    else:
        latlon_arrays["source_lon_range"] = np.array(
            [float(src_lon.min()), float(src_lon.max())])
        latlon_arrays["source_lat_range"] = np.array(
            [float(src_lat.min()), float(src_lat.max())])

    for field_key in all_field_keys:
        # Collect this field across all timesteps
        field_timesteps = []
        latlon_timesteps = []

        for step in sorted_steps:
            if field_key in snapshots[step]:
                arr = np.asarray(snapshots[step][field_key], dtype=np.float64)
                field_timesteps.append(arr)

                # Get ocean mask for mask-aware IDW (skip for land_mask itself)
                _lm = None
                if field_key != "land_mask" and "land_mask" in snapshots[step]:
                    _lm = np.asarray(snapshots[step]["land_mask"],
                                     dtype=np.float64)

                # Regrid to lat-lon
                target_lat = latlon_arrays["lat"]
                target_lon = latlon_arrays["lon"]
                if field_key.endswith("_3d"):
                    regridded = _regrid_3d_level(arr, lon_deg, lat_deg, coord_kind,
                                               target_lat=target_lat,
                                               target_lon=target_lon,
                                               ocean_mask=_lm)
                else:
                    regridded = _regrid_2d(arr, lon_deg, lat_deg, coord_kind,
                                         target_lat=target_lat,
                                         target_lon=target_lon,
                                         ocean_mask=_lm)
                latlon_timesteps.append(regridded)
            else:
                # Field not available at this timestep - skip or use NaN
                # For now, we'll skip incomplete time series
                break
        
        # Only save fields that are available at all timesteps
        if len(field_timesteps) == len(sorted_steps):
            # Stack into time series: shape (n_times, ...)
            native_arrays[field_key] = np.stack(field_timesteps, axis=0)
            latlon_arrays[field_key] = np.stack(latlon_timesteps, axis=0)

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
    domain_extent: tuple[float, float, float, float] | None = None,
    mesh=None,
):
    """Save all standard diagnostic outputs for a test case.

    Parameters
    ----------
    domain_extent : (lon_west, lon_east, lat_south, lat_north) or None
        When set, snapshot plots are cropped to this geographic extent.
        Useful for regional experiments on unstructured grids where the
        auto-crop heuristic fails due to nearest-neighbour extrapolation.
    mesh : VoronoiMesh, optional
        When provided for MPAS grids, snapshot plots use native Voronoi
        polygon rendering instead of regrid+imshow.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    su = scalar_units or {}

    _save_timeseries_csv(output_dir, diag, dt)
    _save_timeseries_plot(output_dir, case_name, diag, su)

    if vol_key and heat_key:
        _save_conservation(output_dir, case_name, diag,
                           vol_key, heat_key, salt_key or "")

    _save_snapshot_plots(
        output_dir, case_name, snapshots, dt, field_specs_2d,
        coord_kind, lon_deg, lat_deg, domain_extent=domain_extent,
        mesh=mesh)
    _save_snapshot_times(output_dir, snapshots, dt)
    _save_snapshot_data(
        output_dir, snapshots, dt, coord_kind, lon_deg, lat_deg,
        domain_extent=domain_extent)

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
    """Guarantee standardized files exist in each case folder.

    Creates placeholder CSVs, snapshot_times.txt, and placeholder PNGs
    so that every test case directory has the full set of expected outputs,
    even when the test crashed (ERROR) or was skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_name in ["mean_timeseries.csv", "conservation_timeseries.csv"]:
        p = output_dir / csv_name
        if not p.exists():
            with open(p, "w") as f:
                f.write("# No data produced\n")
    if not (output_dir / "snapshot_times.txt").exists():
        with open(output_dir / "snapshot_times.txt", "w") as f:
            f.write("step,time_seconds,time_days\n")
    # Generate placeholder PNGs for any missing visualization files.
    case_label = "/".join(output_dir.parts[-4:])
    for png_name in [
        "mean_timeseries.png",
        "conservation_timeseries.png",
        "field_snapshots.png",
    ]:
        p = output_dir / png_name
        if not p.exists():
            _placeholder_plot(p, png_name.replace(".png", ""),
                              f"No data — {case_label}")


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
    elif tc.grid_type == "mpas_regional":
        return {"resolution_km": int(tc.resolution.replace("km", ""))}
    elif tc.grid_type == "latlon_regional":
        parts = tc.resolution.split("x")
        return {"n_lat": int(parts[0]), "n_lon": int(parts[1])}
    elif tc.grid_type == "cs_regional":
        return {"n": int(tc.resolution[1:])}
    elif tc.grid_type == "spectral":
        return {"truncation": int(tc.resolution[1:])}
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


def _create_ocean_setup(tc: TestCase, nlev: int | None = None,
                        H_max: float | None = None, physics=None,
                        A_h: float | None = None,
                        A_v: float | None = None):
    """Create grid, z_coord, and rest-state for any grid type.

    Parameters
    ----------
    tc : TestCase
    nlev : int
    H_max : float
    physics : OceanPhysicsConfig or None
        If provided, passed to the model config to enable physics
        (e.g. prescribed surface forcing for wind-driven experiments).
    A_h : float or None
        Override horizontal viscosity [m^2/s]. If None, uses config default.

    Returns (grid, z_coord, config, model, coord_kind, lon_deg, lat_deg).
    """
    if nlev is None:
        nlev = DEFAULT_NLEV
    if H_max is None:
        H_max = DEFAULT_H_MAX
    from legoesm.ocean.vertical import create_ocean_z_star

    z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    params = _parse_resolution(tc)

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig

        n = params["n"]
        grid = create_cubed_sphere(n)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        config = OceanConfig(**kw)
        model = OceanModel(grid, z_coord, config)
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        grid = create_latlon_grid(params["n_lat"], params["n_lon"])
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        config = LatLonCGridOceanConfig(**kw)
        model = LatLonCGridOceanModel(grid, z_coord, config)
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        mesh = create_voronoi_mesh(params["level"])
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        config = MPASOceanConfig(**kw)
        model = MPASOceanModel(mesh, z_coord, config)
        coord_kind = "mpas"
        lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        return mesh, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "mpas_regional":
        from legoesm.grids.voronoi import create_regional_voronoi_mesh
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.mpas_config import MPASOceanConfig

        res_km = params["resolution_km"]
        # Default gyre basin bounds
        lon_w = tc.run_kwargs.get("lon_west", 0.0)
        lon_e = tc.run_kwargs.get("lon_east", 120.0)
        lat_s = tc.run_kwargs.get("lat_south", 15.0)
        lat_n = tc.run_kwargs.get("lat_north", 75.0)
        mesh = create_regional_voronoi_mesh(
            (lon_w, lon_e), (lat_s, lat_n), resolution_km=res_km)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        config = MPASOceanConfig(**kw)
        model = MPASOceanModel(mesh, z_coord, config)
        coord_kind = "mpas"
        lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        return mesh, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "latlon_regional":
        from legoesm.grids.latlon import create_regional_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.state import LatLonCGridOceanConfig

        n_lat, n_lon = params["n_lat"], params["n_lon"]
        lon_w = tc.run_kwargs.get("lon_west", 0.0)
        lon_e = tc.run_kwargs.get("lon_east", 120.0)
        lat_s = tc.run_kwargs.get("lat_south", 15.0)
        lat_n = tc.run_kwargs.get("lat_north", 75.0)
        grid, wall_mask = create_regional_latlon_grid(
            n_lat, n_lon, lat_s, lat_n, lon_w, lon_e)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        config = LatLonCGridOceanConfig(**kw)
        model = LatLonCGridOceanModel(grid, z_coord, config)
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, config, model, coord_kind, lon_deg, lat_deg

    elif tc.grid_type == "cs_regional":
        from legoesm.grids.cubed_sphere import create_cubed_sphere_panel
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig

        n = params["n"]
        grid = create_cubed_sphere_panel(n, face_id=0, return_cdgrid=False)
        kw = dict(n_barotropic_substeps=30, physics=physics)
        if A_h is not None:
            kw["A_h"] = A_h
        if A_v is not None:
            kw["A_v"] = A_v
        config = OceanConfig(**kw)
        model = OceanModel(grid, z_coord, config)
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        return grid, z_coord, config, model, coord_kind, lon_deg, lat_deg

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
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        return rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=H_max)
    elif tc.grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(grid, z_coord, H_max=H_max)
    elif tc.grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        # Use land with tanh taper (same as other grids); hyperdiffusion mitigates Gibbs
        return rest_state_spectral_ocean(grid, z_coord, H_max=H_max,
                                         land_lat_threshold=80.0)
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


def _create_rest_state_uniform_ts(tc: TestCase, grid, z_coord, H_max=DEFAULT_H_MAX):
    """Create rest-state with uniform T/S (no stratification) + land."""
    if tc.grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(grid, z_coord, H_max=H_max, T_surface=10.0, T_deep=10.0)
    elif tc.grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        return rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=H_max, T_surface=10.0, T_deep=10.0)
    elif tc.grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(grid, z_coord, H_max=H_max, T_surface=10.0, T_deep=10.0)
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


def _create_rest_state_uniform_ts_no_land(tc: TestCase, grid, z_coord, H_max=DEFAULT_H_MAX):
    """Create rest-state with uniform T/S and no land."""
    if tc.grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(grid, z_coord, H_max=H_max, T_surface=10.0, T_deep=10.0,
                                 land_lat_threshold=90.0)
    elif tc.grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        return rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=H_max, T_surface=10.0, T_deep=10.0,
                                        land_lat_threshold=90.0)
    elif tc.grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(grid, z_coord, H_max=H_max, T_surface=10.0, T_deep=10.0,
                                      land_lat_threshold=90.0)
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


def _create_rest_state_no_land(tc: TestCase, grid, z_coord, H_max=DEFAULT_H_MAX):
    """Create rest-state initial condition with no land for any grid type."""
    if tc.grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(grid, z_coord, H_max=H_max, land_lat_threshold=90.0)
    elif tc.grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        return rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=H_max, land_lat_threshold=90.0)
    elif tc.grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(grid, z_coord, H_max=H_max, land_lat_threshold=90.0)
    elif tc.grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        return rest_state_spectral_ocean(grid, z_coord, H_max=H_max,
                                         land_lat_threshold=90.0)
    raise ValueError(f"Unknown grid type: {tc.grid_type}")


# ===========================================================================
# Field extraction helpers
# ===========================================================================

def _extract_fv_ocean(state, grid_type: str, include_velocity_3d: bool = False):
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
        "land_mask": np.asarray(state.land_mask.data, dtype=np.float64),
    }
    if hasattr(state, "v"):
        v_sfc = np.asarray(state.v.data[..., 0], dtype=np.float64)
        result["v_sfc"] = v_sfc
        # C-grid: u and v have different shapes; skip speed_sfc
        if u_sfc.shape == v_sfc.shape:
            result["speed_sfc"] = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    if include_velocity_3d:
        u_raw = np.asarray(state.u.data, dtype=np.float64)
        if hasattr(state, "v"):
            v_raw = np.asarray(state.v.data, dtype=np.float64)
            # C-grid: interpolate staggered u/v to cell centers
            if u_raw.shape[:-1] != v_raw.shape[:-1]:
                u_cc = 0.5 * (u_raw[:, :-1, :] + u_raw[:, 1:, :])
                v_cc = 0.5 * (v_raw[:-1, :, :] + v_raw[1:, :, :])
            else:
                u_cc = u_raw
                v_cc = v_raw
            result["u_3d"] = u_cc
            result["v_3d"] = v_cc
            result["speed_3d"] = np.sqrt(u_cc**2 + v_cc**2)
        else:
            result["u_3d"] = u_raw
        # Add vertical velocity if available
        if hasattr(state, "w"):
            w_3d = np.asarray(state.w.data, dtype=np.float64)
            result["w_3d"] = w_3d
            # Extract vertical velocity below Ekman layer (level 1 = 133.9m depth)
            result["w_133m"] = w_3d[..., 1]
            # Also keep surface for comparison if needed
            result["w_sfc"] = w_3d[..., 0]
    
    # Create surface speed field by interpolating u,v to common grid
    if "u_sfc" in result and "v_sfc" in result:
        u_sfc = result["u_sfc"]
        v_sfc = result["v_sfc"]
        # For C-grid: interpolate staggered velocities to cell centers for speed
        if u_sfc.shape != v_sfc.shape:
            # u is on east-west faces, v on north-south faces
            # Interpolate both to cell centers
            u_cc = 0.5 * (u_sfc[:, :-1] + u_sfc[:, 1:])  # avg in lon direction
            v_cc = 0.5 * (v_sfc[:-1, :] + v_sfc[1:, :])  # avg in lat direction
            # Make sure they have same shape (min of both)
            ny_min = min(u_cc.shape[0], v_cc.shape[0])
            nx_min = min(u_cc.shape[1], v_cc.shape[1])
            u_final = u_cc[:ny_min, :nx_min]
            v_final = v_cc[:ny_min, :nx_min]
            result["speed_sfc"] = np.sqrt(u_final**2 + v_final**2)
    return result


def _extract_mpas_ocean(state, lon_deg, lat_deg, mesh=None,
                        include_velocity_3d: bool = False):
    """Extract snapshot fields for MPAS ocean state.

    Returns raw cell-center arrays; regridding to lat-lon is done by the
    plotting functions via _regrid_2d / _regrid_3d_level.

    When *include_velocity_3d* is True and *mesh* is provided, reconstructs
    cell-center (u_east, v_north) at all levels via Perot reconstruction.
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
        "land_mask": np.asarray(state.land_mask.data, dtype=np.float64),
    }
    if include_velocity_3d and mesh is not None:
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity
        u_edge = np.asarray(state.u.data, dtype=np.float64)  # (nEdges, nlev)
        nlev = u_edge.shape[-1]
        nCells = eta.shape[0]
        u_cc = np.zeros((nCells, nlev), dtype=np.float64)
        v_cc = np.zeros((nCells, nlev), dtype=np.float64)
        for k in range(nlev):
            ue, vn = reconstruct_cell_velocity(state.u.data[:, k], mesh)
            u_cc[:, k] = np.asarray(ue, dtype=np.float64)
            v_cc[:, k] = np.asarray(vn, dtype=np.float64)
        result["u_3d"] = u_cc
        result["v_3d"] = v_cc
        result["speed_3d"] = np.sqrt(u_cc**2 + v_cc**2)
        
        # Add vertical velocity if available (for future MPAS implementation)
        if hasattr(state, "w"):
            w_3d = np.asarray(state.w.data, dtype=np.float64)
            result["w_3d"] = w_3d
            # Extract vertical velocity below Ekman layer (level 1 = 133.9m depth)
            result["w_133m"] = w_3d[..., 1]
            # Also keep surface for comparison if needed
            result["w_sfc"] = w_3d[..., 0]
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
    
    # For spectral grids, synthesize land mask if available (typically all ocean for spectral)
    if hasattr(state, 'land_mask') and hasattr(state.land_mask, 'data'):
        land_mask_grid = np.asarray(
            sh_synthesis(grid, state.land_mask.data), dtype=np.float64)
    else:
        # Default to all ocean for spectral grids (consistent with rest_state config)
        land_mask_grid = np.ones_like(eta_grid, dtype=np.float64)
    
    return {
        "eta": eta_grid,
        "SST": SST,
        "SSS": SSS,
        "T_3d": T_grid,
        "S_3d": S_grid,
        "land_mask": land_mask_grid,
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
    elif grid_type in ("mpas", "mpas_regional"):
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


def _make_scalar_fn(grid_type: str, grid=None, z_coord=None):
    """Return a scalar_fn(state) -> dict for diagnostics.

    When *z_coord* is provided, mean_T and mean_S are volume-weighted
    (sum(T * dz * area * mask) / sum(dz * area * mask)).  Without it
    the diagnostic falls back to an unweighted nanmean, which drifts
    spuriously when vertical diffusion redistributes heat across layers
    of different thickness.
    """
    if grid_type == "spectral":
        if grid is None:
            raise ValueError("Grid object required for spectral scalar function")
        
        def scalar_fn(s):
            from legoesm.grids.gaussian import sh_synthesis, sh_synthesis_3d
            
            # Convert spectral coefficients to physical fields
            eta_phys = sh_synthesis(grid, s.eta_hat.data)        # meters
            T_phys = sh_synthesis_3d(grid, s.T_hat.data)         # °C  
            S_phys = sh_synthesis_3d(grid, s.S_hat.data)         # PSU
            
            # Apply ocean masking if available
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                land_mask_phys = sh_synthesis(grid, s.land_mask.data)
                ocean_mask = land_mask_phys > 0.5
                
                # Compute ocean-only statistics
                eta_ocean = jnp.where(ocean_mask, eta_phys, jnp.nan)
                T_ocean = jnp.where(ocean_mask, T_phys, jnp.nan)
                S_ocean = jnp.where(ocean_mask, S_phys, jnp.nan)
                
                return {
                    "mean_eta": float(jnp.nanmean(eta_ocean)),           # meters
                    "max_abs_eta": float(jnp.nanmax(jnp.abs(eta_ocean))), # meters
                    "mean_T": float(jnp.nanmean(T_ocean)),               # °C
                    "mean_S": float(jnp.nanmean(S_ocean)),               # PSU
                }
            else:
                # Fallback for spectral (typically all ocean anyway)
                return {
                    "mean_eta": float(jnp.mean(eta_phys)),           # meters
                    "max_abs_eta": float(jnp.max(jnp.abs(eta_phys))), # meters
                    "mean_T": float(jnp.mean(T_phys)),               # °C
                    "mean_S": float(jnp.mean(S_phys)),               # PSU
                }
        return scalar_fn
    elif grid_type in ("mpas", "mpas_regional"):
        # Capture z_coord layer thicknesses and cell areas for
        # volume-weighted diagnostics.
        # Use actual h_k (which depends on eta) rather than reference dz_ref,
        # so that the diagnostic tracks the true conserved quantity h*T.
        _area = grid.areaCell if grid is not None else None
        _z_coord = z_coord
        _min_wc = 0.5  # default min_water_column_m

        def scalar_fn(s):
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                mask = s.land_mask.data > 0.5

                # Area-weighted eta mean (nanmean is biased on non-uniform grids)
                if _area is not None:
                    area_ocean = _area * mask.astype(_area.dtype)
                    mean_eta = float(jnp.sum(s.eta.data * area_ocean) / jnp.maximum(jnp.sum(area_ocean), 1e-30))
                else:
                    eta_ocean = jnp.where(mask, s.eta.data, jnp.nan)
                    mean_eta = float(jnp.nanmean(eta_ocean))
                eta_ocean = jnp.where(mask, s.eta.data, jnp.nan)

                # Volume-weighted T and S using actual layer thickness h_k
                if _area is not None and _z_coord is not None:
                    from legoesm.ocean.vertical import compute_layer_thickness
                    h_k = compute_layer_thickness(
                        s.eta.data, s.H_bathy.data, _z_coord,
                        min_water_column_m=_min_wc)
                    vol = _area[:, None] * h_k * mask[:, None]
                    vol_sum = jnp.sum(vol)
                    mean_T = float(jnp.sum(s.T.data * vol) / vol_sum)
                    mean_S = float(jnp.sum(s.S.data * vol) / vol_sum)
                else:
                    T_ocean = jnp.where(mask[:, None], s.T.data, jnp.nan)
                    S_ocean = jnp.where(mask[:, None], s.S.data, jnp.nan)
                    mean_T = float(jnp.nanmean(T_ocean))
                    mean_S = float(jnp.nanmean(S_ocean))

                max_abs_u = float(jnp.max(jnp.abs(s.u.data)))

                # Reconstruct cell-center velocity for comparable speed diagnostic
                from legoesm.ocean.init_mpas import reconstruct_cell_velocity
                u_east, v_north = reconstruct_cell_velocity(s.u.data[:, 0], grid)
                speed_cell = jnp.sqrt(u_east**2 + v_north**2)
                max_speed = float(jnp.max(jnp.where(mask, speed_cell, 0.0)))

                # Volume-weighted KE: 0.5 * sum(speed^2 * h_k * area) / sum(h_k * area)
                if _area is not None and _z_coord is not None:
                    from legoesm.ocean.vertical import compute_layer_thickness
                    h_k = compute_layer_thickness(
                        s.eta.data, s.H_bathy.data, _z_coord,
                        min_water_column_m=_min_wc)
                    nlev = h_k.shape[-1]
                    # Reconstruct cell speed at all levels
                    ke_sum = jnp.zeros(())
                    vol_ke = jnp.zeros(())
                    for lev in range(nlev):
                        ue, vn = reconstruct_cell_velocity(s.u.data[:, lev], grid)
                        spd2 = ue**2 + vn**2
                        cell_vol = _area * h_k[:, lev] * mask
                        ke_sum = ke_sum + jnp.sum(0.5 * spd2 * cell_vol)
                        vol_ke = vol_ke + jnp.sum(cell_vol)
                    mean_ke = float(ke_sum / jnp.maximum(vol_ke, 1e-10))
                else:
                    mean_ke = 0.0

                return {
                    "mean_eta": mean_eta,
                    "max_abs_eta": float(jnp.nanmax(jnp.abs(eta_ocean))),
                    "mean_T": mean_T,
                    "mean_S": mean_S,
                    "max_abs_u": max_abs_u,
                    "max_speed": max_speed,
                    "mean_ke": mean_ke,
                }
            else:
                max_abs_u = float(jnp.max(jnp.abs(s.u.data)))
                from legoesm.ocean.init_mpas import reconstruct_cell_velocity
                u_east, v_north = reconstruct_cell_velocity(s.u.data[:, 0], grid)
                max_speed = float(jnp.max(jnp.sqrt(u_east**2 + v_north**2)))
                return {
                    "mean_eta": float(jnp.mean(s.eta.data)),
                    "max_abs_eta": float(jnp.max(jnp.abs(s.eta.data))),
                    "mean_T": float(jnp.mean(s.T.data)),
                    "mean_S": float(jnp.mean(s.S.data)),
                    "max_abs_u": max_abs_u,
                    "max_speed": max_speed,
                }
        return scalar_fn
    else:
        # Cubed-sphere and lat-lon grids.
        # Capture grid area and z_coord for volume-weighted diagnostics.
        # Use actual h_k (which depends on eta) rather than reference dz_ref,
        # so that the diagnostic tracks the true conserved quantity h*T.
        _area = grid.area if (grid is not None and hasattr(grid, 'area')) else None
        _z_coord = z_coord
        _min_wc = 0.5  # default min_water_column_m

        def scalar_fn(s):
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                ocean_mask = s.land_mask.data > 0.5

                # Area-weighted eta mean (nanmean is biased on non-uniform grids)
                if _area is not None:
                    area_ocean = _area * ocean_mask.astype(_area.dtype)
                    mean_eta = float(jnp.sum(s.eta.data * area_ocean) / jnp.maximum(jnp.sum(area_ocean), 1e-30))
                else:
                    mean_eta = float(jnp.nanmean(jnp.where(ocean_mask, s.eta.data, jnp.nan)))
                eta_ocean = jnp.where(ocean_mask, s.eta.data, jnp.nan)

                # Volume-weighted T and S using actual layer thickness h_k
                if _area is not None and _z_coord is not None:
                    from legoesm.ocean.vertical import compute_layer_thickness
                    h_k = compute_layer_thickness(
                        s.eta.data, s.H_bathy.data, _z_coord,
                        min_water_column_m=_min_wc)
                    vol = _area[..., None] * h_k * ocean_mask[..., None]
                    vol_sum = jnp.sum(vol)
                    mean_T = float(jnp.sum(s.T.data * vol) / vol_sum)
                    mean_S = float(jnp.sum(s.S.data * vol) / vol_sum)
                else:
                    ocean_mask_3d = ocean_mask[..., jnp.newaxis]
                    T_ocean = jnp.where(ocean_mask_3d, s.T.data, jnp.nan)
                    S_ocean = jnp.where(ocean_mask_3d, s.S.data, jnp.nan)
                    mean_T = float(jnp.nanmean(T_ocean))
                    mean_S = float(jnp.nanmean(S_ocean))

                if hasattr(s, 'u') and hasattr(s, 'v'):
                    u_sfc = s.u.data[..., 0]
                    v_sfc = s.v.data[..., 0]
                    # C-grid: u (n_lat, n_lon+1) and v (n_lat+1, n_lon)
                    # can't compute sqrt(u²+v²) directly. Use max(|u|, |v|).
                    if u_sfc.shape != v_sfc.shape:
                        max_speed = float(jnp.maximum(
                            jnp.max(jnp.abs(u_sfc)),
                            jnp.max(jnp.abs(v_sfc))))
                    else:
                        speed_sfc = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2)
                        speed_ocean = jnp.where(ocean_mask, speed_sfc, jnp.nan)
                        max_speed = float(jnp.nanmax(speed_ocean))
                else:
                    max_speed = 0.0

                # Volume-weighted KE
                mean_ke = 0.0
                if hasattr(s, 'u') and hasattr(s, 'v') and _area is not None and _z_coord is not None:
                    u = s.u.data; v = s.v.data
                    # Interpolate to cell centers for KE
                    if u.shape[:-1] != v.shape[:-1]:
                        # C-grid: u (n_lat, n_lon+1, nlev), v (n_lat+1, n_lon, nlev)
                        u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
                        v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
                    else:
                        u_cc = u; v_cc = v
                    spd2 = u_cc**2 + v_cc**2
                    ke_vol = 0.5 * spd2 * _area[..., None] * h_k * ocean_mask[..., None]
                    vol_total = jnp.sum(_area[..., None] * h_k * ocean_mask[..., None])
                    mean_ke = float(jnp.sum(ke_vol) / jnp.maximum(vol_total, 1e-10))

                return {
                    "mean_eta": mean_eta,
                    "max_abs_eta": float(jnp.nanmax(jnp.abs(eta_ocean))),
                    "mean_T": mean_T,
                    "mean_S": mean_S,
                    "max_speed": max_speed,
                    "mean_ke": mean_ke,
                }
            else:
                # Fallback without masking
                return {
                    "mean_eta": float(jnp.mean(s.eta.data)),
                    "max_abs_eta": float(jnp.max(jnp.abs(s.eta.data))),
                    "mean_T": float(jnp.mean(s.T.data)),
                    "mean_S": float(jnp.mean(s.S.data)),
                    "max_speed": float(jnp.maximum(
                        jnp.max(jnp.abs(s.u.data)),
                        jnp.max(jnp.abs(s.v.data)))),
                    "mean_ke": 0.0,
                }
        return scalar_fn


def _make_extract_fn(grid_type: str, grid, lon_deg, lat_deg,
                     include_velocity_3d: bool = False):
    """Return an extract_fn(state) -> dict for snapshots."""
    if grid_type == "spectral":
        def extract_fn(s):
            return _extract_spectral_ocean(s, grid)
        return extract_fn
    elif grid_type in ("mpas", "mpas_regional"):
        _mesh = grid if include_velocity_3d else None
        def extract_fn(s):
            return _extract_mpas_ocean(s, lon_deg, lat_deg, mesh=_mesh,
                                       include_velocity_3d=include_velocity_3d)
        return extract_fn
    else:
        def extract_fn(s):
            return _extract_fv_ocean(s, grid_type,
                                     include_velocity_3d=include_velocity_3d)
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
    """Add a Gaussian SSH perturbation to the rest state.

    Uses great-circle distance centered at (180E, 0N) with sigma=10 deg,
    consistent across all grid types.
    """
    from legoesm.core.field import Field

    eta_amp = 1.0  # 1 m SSH perturbation
    sigma_rad = 10.0 * np.pi / 180.0  # Gaussian width in radians
    lon0 = np.pi   # 180 degrees east
    lat0 = 0.0     # equator

    def _great_circle_perturbation(lon_rad, lat_rad):
        """Compute Gaussian SSH perturbation using great-circle distance."""
        dlon = lon_rad - lon0
        dist = np.arccos(np.clip(
            np.sin(lat_rad) * np.sin(lat0)
            + np.cos(lat_rad) * np.cos(lat0) * np.cos(dlon),
            -1.0, 1.0,
        ))
        return eta_amp * np.exp(-0.5 * (dist / sigma_rad) ** 2)

    if grid_type == "cubed_sphere":
        lon = np.asarray(grid.lon, dtype=np.float64)
        lat = np.asarray(grid.lat, dtype=np.float64)
        perturb = _great_circle_perturbation(lon, lat)
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(new_eta))

    elif grid_type == "latlon":
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        perturb = _great_circle_perturbation(lon_2d, lat_2d)
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(new_eta))

    elif grid_type == "mpas":
        lon = np.asarray(grid.lonCell, dtype=np.float64)
        lat = np.asarray(grid.latCell, dtype=np.float64)
        perturb = _great_circle_perturbation(lon, lat)
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(new_eta))

    elif grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        perturb = _great_circle_perturbation(lon_2d, lat_2d)
        perturb_hat = sh_analysis(grid, jnp.array(perturb))
        new_eta_hat = state.eta_hat.data + perturb_hat
        return state._replace(eta_hat=Field(new_eta_hat))

    raise ValueError(f"Unknown grid type: {grid_type}")


# ===========================================================================
# Wind-driven gyre forcing
# ===========================================================================

def _add_wind_gyre_forcing(state, grid_type: str, grid, z_coord,
                           lon_west=0.0, lon_east=120.0,
                           lat_south=15.0, lat_north=75.0):
    """Create wind-gyre initial state with rectangular basin boundaries."""
    if grid_type in ("cubed_sphere", "cs_regional"):
        from legoesm.ocean.init import wind_driven_gyre_init
        return wind_driven_gyre_init(
            grid, z_coord,
            lon_west=lon_west, lon_east=lon_east,
            lat_south=lat_south, lat_north=lat_north,
        )
    elif grid_type in ("latlon", "latlon_regional"):
        from legoesm.ocean.init_latlon_cgrid import wind_driven_gyre_latlon_cgrid
        return wind_driven_gyre_latlon_cgrid(
            grid, z_coord,
            lon_west=lon_west, lon_east=lon_east,
            lat_south=lat_south, lat_north=lat_north,
        )
    elif grid_type in ("mpas", "mpas_regional"):
        from legoesm.ocean.init_mpas import wind_driven_gyre_mpas
        return wind_driven_gyre_mpas(
            grid, z_coord,
            lon_west=lon_west, lon_east=lon_east,
            lat_south=lat_south, lat_north=lat_north,
        )
    raise ValueError(f"Wind gyre not available for grid: {grid_type}")


# ===========================================================================
# Geostrophic adjustment perturbation
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

        T_data = np.array(state.T.data, dtype=np.float64)  # writeable copy
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
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State ({tc.grid_type})", total_days=days)

    # Check drift is small - all grids now use same physical units
    # Use absolute eta drift in meters rather than relative drift since 
    # initial mean_eta is ~0 in rest state, making relative drift meaningless (division by ~0).
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    # Spectral land-leakage diagnostic: check that eta stays near zero in land cells
    if tc.grid_type == "spectral" and hasattr(state, 'land_mask_grid'):
        from legoesm.grids.gaussian import sh_synthesis
        mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
        eta_grid = np.asarray(sh_synthesis(grid, state.eta_hat.data).real,
                              dtype=np.float64)
        land_leakage = np.max(np.abs(eta_grid * (1.0 - mask)))
        notes += f", land_leakage={land_leakage:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full  # positive downward for plotting

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Rest State {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_rest_state_no_land(tc: TestCase, output_dir: Path, days: float
                   ) -> tuple[str, float, str]:
    """Rest state adjustment with no land: pure ocean should remain near initial condition."""
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state_no_land(tc, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State No Land ({tc.grid_type})", total_days=days)

    # Check drift is small - all grids now use same physical units
    # Use absolute eta drift in meters rather than relative drift since 
    # initial mean_eta is ~0 in rest state, making relative drift meaningless (division by ~0).
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full  # positive downward for plotting

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Rest State No Land {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_rest_state_uniform_ts(tc: TestCase, output_dir: Path, days: float
                              ) -> tuple[str, float, str]:
    """Rest state with land but uniform T/S — diagnostic for baroclinic PGF hypothesis.

    If the stratified rest_state (with land) blows up but this test stays
    stable, it confirms the baroclinic pressure gradient is the primary
    trigger for land-boundary instabilities.
    """
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state_uniform_ts(tc, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State Uniform T/S ({tc.grid_type})", total_days=days)

    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Rest State Uniform T/S {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_rest_state_uniform_ts_no_land(tc: TestCase, output_dir: Path, days: float
                                      ) -> tuple[str, float, str]:
    """Rest state with uniform T/S and no land — control for uniform_ts diagnostic."""
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state_uniform_ts_no_land(tc, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Rest State Uniform T/S No Land ({tc.grid_type})", total_days=days)

    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"eta drift={eta_drift:.2e}, T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Rest State Uniform T/S No Land {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Barotropic Wave
# ===========================================================================

def run_barotropic_wave(tc: TestCase, output_dir: Path, days: float
                        ) -> tuple[str, float, str]:
    """Barotropic gravity wave: Gaussian SSH perturbation propagation."""
    if tc.grid_type == "spectral":
        raise NotImplementedError(
            "Barotropic wave skipped for spectral grid (land masking issues)")
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state(tc, grid, z_coord)
    state = _add_barotropic_wave_perturbation(
        state, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Barotropic Wave ({tc.grid_type})", total_days=days)

    # All grids now use same physical units
    eta_max = diag["max_abs_eta"][-1] if diag.get("max_abs_eta") else 0
    notes = f"max|eta|={eta_max:.4f} m"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Barotropic Wave {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_abs_eta": "m",
                      "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Wind-Driven Gyre
# ===========================================================================

def _make_gyre_physics(
    wind_profile: str = "single_gyre",
    wind_buffer_deg: float = 0.0,
):
    """Create OceanPhysicsConfig with prescribed gyre wind forcing.

    Parameters
    ----------
    wind_profile : str
        "single_gyre", "double_gyre", or "double_gyre_sin2".
    wind_buffer_deg : float
        Buffer zone width [degrees] for the sin² wind taper. Only
        meaningful for ``wind_profile="double_gyre_sin2"``; ignored
        otherwise.

    Notes
    -----
    Lateral viscosity is NOT in the physics config because the physics
    pipeline's harmonic mixing is cubed-sphere-only. Instead, A_h is
    set on the ocean config via _create_ocean_setup (the dynamics
    applies it natively on both cubed-sphere and latlon when
    physics_fn is None for that module). When physics_fn IS set,
    the dynamics skips config.A_h — so we keep lateral_mixing="none"
    in the physics config and set A_h on the base config.
    See _create_ocean_setup where A_h is passed.
    """
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import (
        BottomDragConfig, LinearDragConfig,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    # Linear bottom drag r = 1e-4 s^-1 (Stommel, design doc §4.2).
    # Lateral viscosity A_h handled by the base ocean config (see above).
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile=wind_profile,
                wind_buffer_deg=wind_buffer_deg,
                tau_max=0.1,
                lat_south_deg=15.0,
                lat_north_deg=75.0,
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(
            scheme="linear",
            linear=LinearDragConfig(r=1e-4),
        ),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def _make_global_wind_physics():
    """Create OceanPhysicsConfig with global 3-belt wind forcing."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import (
        BottomDragConfig, LinearDragConfig,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind",
                tau_max=0.1,
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(
            scheme="linear",
            linear=LinearDragConfig(r=1e-4),
        ),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def _create_simplified_continent_mask(lon_deg, lat_deg,
                                       continent_lon_west=20.0,
                                       continent_lon_east=60.0,
                                       continent_lat_south=-55.0,
                                       polar_cap_lat=80.0):
    """Create a simplified continent land mask for global wind-driven tests.

    Geometry:
    - North polar cap: land poleward of +polar_cap_lat
    - South polar cap: land poleward of -polar_cap_lat
    - Single meridional continent from north cap to continent_lat_south
    - Open Drake Passage south of continent_lat_south
    - Everything else is ocean (including circumpolar band)

    The continent is 40° wide (20-60°E). This is the minimum needed
    for full latitude coverage on the ico3 MPAS mesh (~4-5° spacing)
    and to prevent roll-based operators from communicating through
    the land barrier on the latlon grid (8 cells at 5° resolution).

    Parameters
    ----------
    lon_deg, lat_deg : array
        Cell-center coordinates in degrees.
    continent_lon_west, continent_lon_east : float
        Longitude bounds of the continent [degrees].
    continent_lat_south : float
        Southern tip of the continent [degrees]. Drake Passage opens
        south of this latitude.
    polar_cap_lat : float
        Latitude of polar caps [degrees]. Land poleward of ±this value.

    Returns
    -------
    land_mask : array
        1 = ocean, 0 = land.
    """
    lon = jnp.asarray(lon_deg)
    lat = jnp.asarray(lat_deg)

    # Start with all ocean
    ocean = jnp.ones_like(lat)

    # Polar caps: land
    ocean = jnp.where(jnp.abs(lat) > polar_cap_lat, 0.0, ocean)

    # Single continent: land where inside lon bounds AND north of Drake Passage
    in_continent = (
        (lon >= continent_lon_west) & (lon <= continent_lon_east) &
        (lat >= continent_lat_south)
    )
    ocean = jnp.where(in_continent, 0.0, ocean)

    return ocean


def _run_gyre_experiment(tc: TestCase, output_dir: Path, days: float,
                         wind_profile: str, label: str,
                         wind_buffer_deg: float = 0.0,
                         ) -> tuple[str, float, str]:
    """Shared runner for barotropic gyre experiments."""
    _supported = ("cubed_sphere", "latlon", "mpas", "mpas_regional",
                   "latlon_regional", "cs_regional")
    if tc.grid_type not in _supported:
        raise NotImplementedError(
            f"{label} not implemented for {tc.grid_type} grid "
            f"(no surface forcing support)")

    physics = _make_gyre_physics(wind_profile, wind_buffer_deg=wind_buffer_deg)
    # A_h = 5e5 m^2/s: Munk layer delta_M ~ 300 km, needed to
    # stabilise long integrations at ~5-degree resolution.
    # Default A_v = 1e-3 (higher values destabilise latlon).
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics, A_h=5e5))
    state = _add_wind_gyre_forcing(
        None, tc.grid_type, grid, z_coord,
        lon_west=0.0, lon_east=120.0, lat_south=15.0, lat_north=75.0,
    )

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg,
                                  include_velocity_3d=True)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"{label} ({tc.grid_type})", total_days=days)

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    notes = f"max speed={max_speed:.4f} m/s, eta drift={eta_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any

    field_specs = [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("speed_sfc", "Surface speed (m/s)", "magma"),
    ]

    case_label = f"{label} {tc.grid_type} {tc.resolution}"
    # For regional grids, pass domain extent so plots are cropped correctly
    # (the auto-crop heuristic fails on unstructured meshes because
    # nearest-neighbour regridding bleeds beyond the domain).
    is_regional = tc.grid_type in ("mpas_regional", "latlon_regional",
                                    "cs_regional")
    extent = (0.0, 120.0, 15.0, 75.0) if is_regional else None
    _save_case_diagnostics(
        output_dir, case_label,
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=field_specs,
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU"},
        domain_extent=extent,
        mesh=grid if coord_kind == "mpas" else None)

    _save_velocity_profiles(output_dir, case_label, snapshots, dt,
                            depth, level_label="Depth (m)")

    return "PASS" if ok else "FAIL", wall, notes


def run_barotropic_gyre(tc: TestCase, output_dir: Path, days: float
                        ) -> tuple[str, float, str]:
    """Wind-driven single barotropic gyre (Stommel 1948, Munk 1950)."""
    return _run_gyre_experiment(tc, output_dir, days,
                                wind_profile="single_gyre",
                                label="Barotropic Gyre")


def run_barotropic_double_gyre(tc: TestCase, output_dir: Path, days: float
                               ) -> tuple[str, float, str]:
    """Wind-driven barotropic double gyre (Holland & Lin 1975) — cosine wind."""
    return _run_gyre_experiment(tc, output_dir, days,
                                wind_profile="double_gyre",
                                label="Barotropic Double Gyre")


def run_barotropic_double_gyre_sin2(tc: TestCase, output_dir: Path, days: float
                                    ) -> tuple[str, float, str]:
    """Wind-driven barotropic double gyre with sin² wind profile (5° taper)."""
    return _run_gyre_experiment(tc, output_dir, days,
                                wind_profile="double_gyre_sin2",
                                wind_buffer_deg=5.0,
                                label="Barotropic Double Gyre sin2")


def _make_baroclinic_scalar_fn(grid_type: str, grid=None, z_coord=None, config=None):
    """Enhanced scalar function for baroclinic gyre with N-S temperature gradient diagnostics."""
    import jax.numpy as jnp
    
    # Get base scalar function
    base_scalar_fn = _make_scalar_fn(grid_type, grid, z_coord)
    
    # Domain bounds for North-South analysis
    lat_south = config.lat_south if config else 15.0
    lat_north = config.lat_north if config else 75.0
    lat_center = (lat_south + lat_north) / 2.0
    
    if grid_type == "latlon_regional":
        # Get latitude coordinates
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        
        # Find indices for north/south split
        center_idx = np.argmin(np.abs(lat_deg - lat_center))
        
        def scalar_fn(s):
            base_diag = base_scalar_fn(s)
            
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                ocean_mask = s.land_mask.data > 0.5
                
                # Compute North-South temperature differences at key levels
                T_data = s.T.data
                
                # Surface level (0) and thermocline level (4, ~681m depth)
                for level, level_name in [(0, 'surface'), (4, 'thermocline')]:
                    if level < T_data.shape[-1]:
                        T_level = T_data[..., level]
                        T_level_ocean = jnp.where(ocean_mask, T_level, jnp.nan)
                        
                        # Split domain at center latitude
                        T_north = T_level_ocean[center_idx:, :]
                        T_south = T_level_ocean[:center_idx, :]
                        
                        # Compute mean temperatures in each region
                        T_north_mean = jnp.nanmean(T_north)
                        T_south_mean = jnp.nanmean(T_south)
                        
                        # North-South temperature difference (positive = north warmer)
                        dT_ns = T_north_mean - T_south_mean
                        
                        # Add to diagnostics
                        base_diag[f"dT_ns_{level_name}"] = float(dT_ns)
                        base_diag[f"T_north_{level_name}"] = float(T_north_mean)
                        base_diag[f"T_south_{level_name}"] = float(T_south_mean)
                        
                        # Spatial standard deviation (measure of baroclinic development)
                        spatial_std = jnp.nanstd(T_level_ocean)
                        base_diag[f"T_spatial_std_{level_name}"] = float(spatial_std)
            
            return base_diag
            
        return scalar_fn
        
    elif grid_type == "mpas_regional":
        # For MPAS, use cell latitude coordinates
        lat_deg = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi
        
        # Find cells in north vs south regions
        north_mask = lat_deg >= lat_center
        south_mask = lat_deg < lat_center
        
        def scalar_fn(s):
            base_diag = base_scalar_fn(s)
            
            if hasattr(s, 'land_mask') and hasattr(s.land_mask, 'data'):
                ocean_mask = s.land_mask.data > 0.5
                
                # Compute North-South temperature differences at key levels
                T_data = s.T.data
                
                for level, level_name in [(0, 'surface'), (4, 'thermocline')]:
                    if level < T_data.shape[-1]:
                        T_level = T_data[..., level]
                        T_level_ocean = jnp.where(ocean_mask, T_level, jnp.nan)
                        
                        # Extract north and south regions
                        T_north = jnp.where(north_mask & ocean_mask, T_level, jnp.nan)
                        T_south = jnp.where(south_mask & ocean_mask, T_level, jnp.nan)
                        
                        # Compute mean temperatures in each region
                        T_north_mean = jnp.nanmean(T_north)
                        T_south_mean = jnp.nanmean(T_south)
                        
                        # North-South temperature difference
                        dT_ns = T_north_mean - T_south_mean
                        
                        # Add to diagnostics
                        base_diag[f"dT_ns_{level_name}"] = float(dT_ns)
                        base_diag[f"T_north_{level_name}"] = float(T_north_mean)
                        base_diag[f"T_south_{level_name}"] = float(T_south_mean)
                        
                        # Spatial standard deviation
                        spatial_std = jnp.nanstd(T_level_ocean)
                        base_diag[f"T_spatial_std_{level_name}"] = float(spatial_std)
            
            return base_diag
            
        return scalar_fn
    
    else:
        # For other grids, just return the base function
        return base_scalar_fn


def run_baroclinic_gyre(tc: TestCase, output_dir: Path, days: float,
                        gyre_config=None, label: str = "Baroclinic Gyre",
                        ) -> tuple[str, float, str]:
    """Regional wind-driven baroclinic gyre with surface restoring.

    ``gyre_config`` defaults to ``BaroclinicGyreConfig()`` (sin² wind
    profile with 5° edge taper). Pass an explicit config to select a
    different wind profile / taper combination — see
    ``run_baroclinic_gyre_cos`` for the cosine-wind / no-taper variant.
    """
    if tc.grid_type not in ("mpas_regional", "latlon_regional"):
        raise NotImplementedError(
            f"{label} only implemented for regional grids, not {tc.grid_type}")

    from legoesm.ocean.experiments.baroclinic_gyre import (
        BaroclinicGyreConfig, create_initial_conditions, create_forcings)

    config = gyre_config if gyre_config is not None else BaroclinicGyreConfig()
    physics = create_forcings(tc.grid_type, None, config)
    
    grid, z_coord, ocean_config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics, A_h=config.A_h))
    
    state = create_initial_conditions(tc.grid_type, grid, z_coord, config)
    
    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)
    
    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_baroclinic_scalar_fn(tc.grid_type, grid, z_coord, config)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg,
                                  include_velocity_3d=True)
    
    def step_fn(s, dt_):
        return model.step(s, dt_)
    
    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"{label} ({tc.grid_type})", total_days=days)
    
    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    T_list = diag.get("mean_T", [])
    T_drift = (abs(T_list[-1] - T_list[0])
               if len(T_list) >= 2 else 0.0)
    
    # Extract baroclinic diagnostics
    dT_ns_surface = diag.get("dT_ns_surface", [0])[-1] if diag.get("dT_ns_surface") else 0
    dT_ns_thermocline = diag.get("dT_ns_thermocline", [0])[-1] if diag.get("dT_ns_thermocline") else 0
    T_spatial_std_surface = diag.get("T_spatial_std_surface", [0])[-1] if diag.get("T_spatial_std_surface") else 0
    T_spatial_std_thermocline = diag.get("T_spatial_std_thermocline", [0])[-1] if diag.get("T_spatial_std_thermocline") else 0
    
    notes = (f"max_speed={max_speed:.4f}m/s, eta_drift={eta_drift:.2e}, "
             f"T_drift={T_drift:.3f}degC, dT_NS_sfc={dT_ns_surface:.6f}degC, "
             f"dT_NS_thermo={dT_ns_thermocline:.6f}degC")
    
    # Save results  
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full
    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "max_speed": max_speed, "eta_drift": eta_drift, "T_drift": T_drift,
        "depth": depth.tolist(), "notes": notes,
    }, diag=diag)  # iter-105: surface BLOWUP info if any
    
    # Regional extent for proper plotting
    extent = (config.lon_west, config.lon_east, config.lat_south, config.lat_north)
    _save_case_diagnostics(
        output_dir, f"{label} {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("speed_sfc", "Surface speed (m/s)", "magma"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("w_133m", "Vertical velocity at 134m (m/s)", "RdBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU",
                      "dT_ns_surface": "degC", "dT_ns_thermocline": "degC",
                      "T_north_surface": "degC", "T_south_surface": "degC",
                      "T_north_thermocline": "degC", "T_south_thermocline": "degC",
                      "T_spatial_std_surface": "degC", "T_spatial_std_thermocline": "degC"},
        domain_extent=extent,
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


def run_baroclinic_gyre_cos(tc: TestCase, output_dir: Path, days: float
                            ) -> tuple[str, float, str]:
    """Regional baroclinic gyre with cosine wind profile (no edge taper)."""
    from legoesm.ocean.experiments.baroclinic_gyre import BaroclinicGyreConfig
    config_ = BaroclinicGyreConfig(
        wind_profile="double_gyre", wind_buffer_deg=0.0,
    )
    return run_baroclinic_gyre(
        tc, output_dir, days,
        gyre_config=config_, label="Baroclinic Gyre cos",
    )


# ===========================================================================
# Runner: Global Wind-Driven Circulation
# ===========================================================================

def run_global_barotropic_wind(tc: TestCase, output_dir: Path, days: float
                                ) -> tuple[str, float, str]:
    """Global barotropic wind-driven circulation with simplified continent.

    Tests the barotropic response to a global 3-belt wind stress
    (trades, westerlies, polar easterlies) in a basin with:
    - One meridional continent (30-90°E) from the north polar cap to 55°S
    - Open Drake Passage south of 55°S → circumpolar current
    - Polar caps (land poleward of ±80°)

    Expected features: subtropical/subpolar gyres in Atlantic-like and
    Pacific-like basins, western boundary currents, and ACC-like flow.
    """
    if tc.grid_type not in ("cubed_sphere", "latlon", "mpas"):
        raise NotImplementedError(
            f"Global wind not implemented for {tc.grid_type}")

    physics = _make_global_wind_physics()
    nlev_override = tc.run_kwargs.get("nlev", None)
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics, A_h=5e5, nlev=nlev_override))

    # Build initial state with simplified continent land mask
    # Uniform T/S for a truly barotropic experiment (no baroclinic modes)
    T_uniform = 10.0  # degC
    S_uniform = 35.0  # PSU
    if tc.grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(grid, z_coord,
                                 T_surface=T_uniform, T_deep=T_uniform,
                                 S_uniform=S_uniform)
        # Override land mask with simplified continent
        lon_flat = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_flat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        mask = _create_simplified_continent_mask(lon_flat, lat_flat)
        from legoesm.core.field import Field
        state = state._replace(land_mask=Field(data=mask.astype(state.eta.data.dtype)))
    elif tc.grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(grid, z_coord,
                                              T_surface=T_uniform, T_deep=T_uniform,
                                              S_uniform=S_uniform)
        lon_2d = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon_grid, lat_grid = np.meshgrid(lon_2d, lat_1d) if lon_2d.ndim == 1 else (lon_2d, lat_1d)
        if lat_grid.ndim == 1:
            lat_grid = lat_1d[:, None] * np.ones((1, len(lon_2d)))
            lon_grid = lon_2d[None, :] * np.ones((len(lat_1d), 1))
        mask = _create_simplified_continent_mask(lon_grid, lat_grid)
        from legoesm.core.field import Field
        from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
        mask_typed = mask.astype(state.eta.data.dtype)
        u_mask_new, v_mask_new = compute_face_masks(mask_typed)
        state = state._replace(
            land_mask=Field(data=mask_typed),
            u_mask=Field(data=u_mask_new),
            v_mask=Field(data=v_mask_new),
        )
    elif tc.grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(grid, z_coord,
                                      T_surface=T_uniform, T_deep=T_uniform,
                                      S_uniform=S_uniform)
        lon_deg_c = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg_c = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi
        mask = _create_simplified_continent_mask(lon_deg_c, lat_deg_c)
        from legoesm.core.field import Field
        state = state._replace(land_mask=Field(data=mask.astype(state.eta.data.dtype)))

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Global Wind ({tc.grid_type})", total_days=days)

    max_speed = diag["max_speed"][-1] if diag.get("max_speed") else 0
    eta_list = diag.get("mean_eta", [])
    eta_drift = (abs(eta_list[-1] - eta_list[0])
                 if len(eta_list) >= 2 else 0.0)
    notes = f"max speed={max_speed:.4f} m/s, eta drift={eta_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any

    _save_case_diagnostics(
        output_dir, f"Global Wind {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("speed_sfc", "Surface speed (m/s)", "magma"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "max_speed": "m/s",
                      "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Geostrophic Adjustment
# ===========================================================================

def run_geostrophic_adjustment(tc: TestCase, output_dir: Path, days: float
                   ) -> tuple[str, float, str]:
    """Geostrophic adjustment: meridional temperature front relaxation."""
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc))
    state = _create_rest_state(tc, grid, z_coord)
    state = _add_baroclinic_perturbation(
        state, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    def step_fn(s, dt_):
        return model.step(s, dt_)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Geostrophic Adj ({tc.grid_type})", total_days=days)

    # All grids now use same physical units
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Geostrophic Adj {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
            ("SSS", "SSS (PSU)", "YlGnBu"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

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

        # SSH perturbation with mask (area-weighted mean subtraction)
        eta_pert = 0.05 * np.sin(3.0 * np.radians(lon_2d)) * np.cos(
            2.0 * np.radians(lat_2d)) * mask
        w = np.asarray(grid.weights, dtype=np.float64)[:, None] * mask
        eta_pert -= np.sum(eta_pert * w) / np.maximum(np.sum(w), 1e-30)
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
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
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

    # All grids now use same physical units
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = f"T drift={T_drift:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Phillips 2-Layer {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Helper: cell-center lat/lon for any grid type
# ===========================================================================

def _get_cell_latlon_rad(grid_type, grid):
    """Return (lat, lon) in radians, broadcast to match cell shape."""
    if grid_type == "mpas":
        return (np.asarray(grid.latCell, dtype=np.float64),
                np.asarray(grid.lonCell, dtype=np.float64))
    elif grid_type in ("latlon", "spectral"):
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        return lat_2d, lon_2d
    else:  # cubed_sphere
        return (np.asarray(grid.lat, dtype=np.float64),
                np.asarray(grid.lon, dtype=np.float64))


# ===========================================================================
# Runner: Inertia-Gravity Wave (Bishnu et al. 2024)
# ===========================================================================
# Reference: Bishnu et al. (2024), "A Verification Suite of Test Cases for
# the Barotropic Solver of Ocean Models", JAMES.
# DOI: 10.1029/2022MS003545
#
# Sinusoidal inertia-gravity (Poincare) wave on the sphere.
# Analytical dispersion: omega^2 = f^2 + g*H*(kx^2 + ky^2)
# Tests the barotropic pressure-gradient and Coriolis terms.
# ===========================================================================

def _init_inertia_gravity_wave(state, grid_type, grid, z_coord):
    """Initialize a sinusoidal inertia-gravity wave perturbation.

    Uses wavenumber-2 pattern in both longitude and latitude.
    Analytical solution for comparison after propagation.
    """
    from legoesm.core.field import Field

    H = float(z_coord.H_max)
    f0 = 1.0e-4  # Coriolis parameter (mid-latitude f-plane value)

    # Wavenumber-2 pattern
    lat, lon = _get_cell_latlon_rad(grid_type, grid)
    kx = 2.0  # wavenumber in zonal direction (cycles)
    ky = 2.0  # wavenumber in meridional direction (cycles)

    # Physical wavenumbers on the sphere (approximate for low wavenumbers)
    k_phys = kx / _A_EARTH
    l_phys = ky / _A_EARTH

    # Dispersion relation
    omega = np.sqrt(f0**2 + _G_EARTH * H * (k_phys**2 + l_phys**2))

    # Initial perturbation (t=0)
    eta_amp = 1.0  # 1 m amplitude
    phase = kx * lon + ky * lat
    eta_pert = eta_amp * np.cos(phase)

    # Velocity from linearized SWE: u, v from eta at t=0
    # u = g/(omega^2 - f^2) * (omega*kx*cos(phase) - f*ky*sin(phase))
    # v = g/(omega^2 - f^2) * (omega*ky*cos(phase) + f*kx*sin(phase))
    denom = omega**2 - f0**2
    if abs(denom) < 1e-30:
        denom = 1e-30
    u_pert = (_G_EARTH / denom) * (
        omega * k_phys * np.cos(phase) - f0 * l_phys * np.sin(phase))
    v_pert = (_G_EARTH / denom) * (
        omega * l_phys * np.cos(phase) + f0 * k_phys * np.sin(phase))

    if grid_type == "spectral":
        from legoesm.grids.gaussian import (
            sh_analysis, sh_analysis_oc2_3d, sh_analysis_dmu_3d)
        eta_hat = sh_analysis(grid, jnp.array(eta_pert))
        cos_lat = np.asarray(grid.cos_lat[:, None], dtype=np.float64)
        a = grid.radius
        nlev = state.T_hat.data.shape[-1]
        # Only perturb level 0 (consistent with cubed-sphere / lat-lon init)
        u_cos_2d = jnp.array(u_pert * cos_lat)    # (n_lat, n_lon)
        v_cos_2d = jnp.array(v_pert * cos_lat)
        u_cos = jnp.concatenate([u_cos_2d[..., None],
                                 jnp.zeros((*u_cos_2d.shape, nlev - 1))], axis=-1)
        v_cos = jnp.concatenate([v_cos_2d[..., None],
                                 jnp.zeros((*v_cos_2d.shape, nlev - 1))], axis=-1)
        im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        one_over_a = 1.0 / a
        vor_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
                   + one_over_a * sh_analysis_dmu_3d(grid, u_cos))
        div_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
                   - one_over_a * sh_analysis_dmu_3d(grid, v_cos))
        return state._replace(
            eta_hat=Field(eta_hat),
            vor_hat=Field(vor_hat),
            div_hat=Field(div_hat))

    elif grid_type == "mpas":
        lat_e = np.asarray(grid.latEdge, dtype=np.float64)
        lon_e = np.asarray(grid.lonEdge, dtype=np.float64)
        u_data = np.array(state.u.data, dtype=np.float64, copy=True)
        phase_e = kx * lon_e + ky * lat_e
        u_e = (_G_EARTH / denom) * (
            omega * k_phys * np.cos(phase_e) - f0 * l_phys * np.sin(phase_e))
        v_e = (_G_EARTH / denom) * (
            omega * l_phys * np.cos(phase_e) + f0 * k_phys * np.sin(phase_e))
        # Project onto edge normals
        angle = np.asarray(grid.angleEdge, dtype=np.float64)
        u_data[..., 0] = u_e * np.cos(angle) + v_e * np.sin(angle)
        eta_cell = eta_amp * np.cos(kx * lon + ky * lat)
        return state._replace(
            eta=Field(jnp.array(eta_cell)),
            u=Field(jnp.array(u_data)))

    else:  # cubed_sphere, latlon
        u_data = np.array(state.u.data, dtype=np.float64, copy=True)
        v_data = np.array(state.v.data, dtype=np.float64, copy=True)
        u_data[..., 0] = u_pert
        v_data[..., 0] = v_pert
        return state._replace(
            eta=Field(jnp.array(eta_pert)),
            u=Field(jnp.array(u_data)),
            v=Field(jnp.array(v_data)))


def run_inertia_gravity_wave(tc: TestCase, output_dir: Path, days: float
                              ) -> tuple[str, float, str]:
    """Bishnu et al. 2024: inertia-gravity (Poincare) wave propagation.

    Single-level (SW-equivalent) ocean model with sinusoidal IGW initial
    condition. Measures L2 error against analytical solution and checks
    dispersion properties.
    """
    H_max = 1000.0  # equivalent depth (m)
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=2, H_max=H_max))
    state = _create_rest_state(tc, grid, z_coord, H_max=H_max)
    state = _init_inertia_gravity_wave(state, tc.grid_type, grid, z_coord)

    # Store initial eta for error computation
    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis
        eta_init = np.asarray(
            sh_synthesis(grid, state.eta_hat.data), dtype=np.float64)
    else:
        eta_init = np.asarray(state.eta.data, dtype=np.float64)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    check_fn = _make_check_fn(tc.grid_type)
    scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"IGW ({tc.grid_type})", total_days=days)

    # Compute analytical solution at t_final
    t_final = days * 86400.0
    f0 = 1.0e-4
    kx, ky = 2.0, 2.0
    k_phys = kx / _A_EARTH
    l_phys = ky / _A_EARTH
    omega = np.sqrt(f0**2 + _G_EARTH * H_max * (k_phys**2 + l_phys**2))
    lat, lon = _get_cell_latlon_rad(tc.grid_type, grid)
    eta_exact = np.cos(kx * lon + ky * lat - omega * t_final)

    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis
        eta_final = np.asarray(
            sh_synthesis(grid, state.eta_hat.data), dtype=np.float64)
    else:
        eta_final = np.asarray(state.eta.data, dtype=np.float64)

    l2_err = float(np.sqrt(np.mean((eta_final - eta_exact)**2)) /
                   max(np.sqrt(np.mean(eta_exact**2)), 1e-30))
    max_eta = float(np.max(np.abs(eta_final)))
    notes = f"L2={l2_err:.4f}, max|eta|={max_eta:.3f}m, omega={omega:.2e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "H_max": H_max,
        "reference": "Bishnu et al. 2024, DOI:10.1029/2022MS003545",
        "L2_error": l2_err, "omega_analytical": omega,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"IGW Bishnu {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[("eta", "SSH (m)", "RdBu_r")],
        vol_key="mean_eta",
        heat_key="mean_T",
        scalar_units={"mean_eta": "m"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Lock Exchange (NEMO / Petersen et al. 2015)
# ===========================================================================
# Reference: Petersen et al. (2015), Ocean Modelling 86, 93-113.
# DOI: 10.1016/j.ocemod.2014.12.004
# Also: Ilicak et al. (2012), Ocean Modelling 45-46, 37-49.
# NEMO test cases: https://sites.nemo-ocean.io/user-guide/tests.html
#
# Two fluids of different densities separated by a vertical front.
# Dense cold water on one side, light warm water on the other.
# Gravity currents form when the "lock" is removed (t=0).
# Key diagnostic: Reference Potential Energy (RPE) measures spurious mixing.
# RPE(t) = g * integral(rho * z_star dV) where z_star is the equilibrium
# parcel height in a minimum-energy sorted state.
# ===========================================================================

def _init_lock_exchange(state, grid_type, grid, z_coord):
    """Initialize lock-exchange: cold dense (western hemisphere) / warm light (eastern).

    Adapted to global ocean grids following Petersen et al. (2015):
      - Left (lon < 0): T = 5 degC  (dense, rho ~ 1027 kg/m^3)
      - Right (lon > 0): T = 30 degC (light, rho ~ 1022 kg/m^3)
      - Salinity: uniform 35 PSU
      - Velocity: zero (lock released at t=0)
    """
    from legoesm.core.field import Field

    T_cold = 5.0    # degC (dense side)
    T_warm = 30.0   # degC (light side)

    lat, lon = _get_cell_latlon_rad(grid_type, grid)

    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis_3d
        T_hat = state.T_hat.data
        from legoesm.grids.gaussian import sh_synthesis_3d
        T_grid = np.array(sh_synthesis_3d(grid, T_hat), dtype=np.float64)
        nlev = T_grid.shape[-1]
        mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
        # Front at prime meridian (lon=0)
        T_field = np.where(lon[..., None] < 0, T_cold, T_warm) * mask[..., None]
        new_T_hat = sh_analysis_3d(grid, jnp.array(T_field))
        return state._replace(T_hat=Field(new_T_hat))

    else:
        T_data = np.array(state.T.data, dtype=np.float64, copy=True)
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        nlev = T_data.shape[-1]
        # Temperature front at prime meridian
        for k in range(nlev):
            T_data[..., k] = np.where(lon < 0, T_cold, T_warm) * mask
        return state._replace(T=Field(jnp.array(T_data)))


def _compute_rpe(state, grid_type, grid, z_coord):
    """Compute Reference Potential Energy (Ilicak et al. 2012).

    RPE = g * sum(rho_sorted * z_ref * dz * area)
    Approximation: sort density profile at each column and compute
    domain-integrated rho * z.
    """
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d
        T = np.asarray(sh_synthesis_3d(grid, state.T_hat.data), dtype=np.float64)
        S = np.asarray(sh_synthesis_3d(grid, state.S_hat.data), dtype=np.float64)
        area = np.asarray(grid.area, dtype=np.float64)
    elif grid_type == "mpas":
        T = np.asarray(state.T.data, dtype=np.float64)
        S = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.areaCell, dtype=np.float64)
    else:
        T = np.asarray(state.T.data, dtype=np.float64)
        S = np.asarray(state.S.data, dtype=np.float64)
        area = np.asarray(grid.area, dtype=np.float64)

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)

    # Compute density at each point using linearized EOS
    from legoesm.ocean.eos import linear_eos
    rho = np.asarray(linear_eos(
        jnp.array(T), jnp.array(S), jnp.zeros_like(jnp.array(T)),
        rho_ref=1025.0, alpha_T=2.0e-4, beta_S=0.0, T_ref=15.0,
    ), dtype=np.float64)

    # Potential energy: PE = g * sum(rho * z * dz * area)
    # For RPE, we'd sort density globally, but as approximation compute PE
    spatial_shape = T.shape[:-1]
    area_bc = area.reshape(spatial_shape)
    pe = 0.0
    for k in range(len(z_full)):
        pe += float(np.nansum(rho[..., k] * z_full[k] * dz[k] * area_bc))
    return _G_EARTH * pe


def run_lock_exchange(tc: TestCase, output_dir: Path, days: float
                      ) -> tuple[str, float, str]:
    """Lock exchange: density-driven gravity currents (Petersen et al. 2015).

    Cold dense water in western hemisphere, warm light in eastern.
    Monitors potential energy evolution as a proxy for spurious mixing.
    """
    H_max = 500.0  # shallow basin
    nlev = 20
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=nlev, H_max=H_max))
    state = _create_rest_state(tc, grid, z_coord, H_max=H_max)
    state = _init_lock_exchange(state, tc.grid_type, grid, z_coord)

    # Compute initial PE
    pe_init = _compute_rpe(state, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)

    # Custom scalar function that includes PE
    base_scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)

    def scalar_fn(s):
        scalars = base_scalar_fn(s)
        pe = _compute_rpe(s, tc.grid_type, grid, z_coord)
        scalars["PE"] = pe
        if abs(pe_init) > 1e-30:
            scalars["PE_rel"] = (pe - pe_init) / abs(pe_init)
        else:
            scalars["PE_rel"] = 0.0
        return scalars

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Lock Exchange ({tc.grid_type})", total_days=days,
        blowup_threshold=200.0)

    pe_drift = _compute_drift(diag.get("PE", []))
    pe_rel_final = diag["PE_rel"][-1] if diag.get("PE_rel") else 0.0
    notes = f"PE drift={pe_drift:.2e}, PE_rel_final={pe_rel_final:.4e}"

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": nlev, "H_max": H_max,
        "reference": "Petersen et al. 2015, DOI:10.1016/j.ocemod.2014.12.004",
        "PE_drift": pe_drift, "PE_rel_final": pe_rel_final,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Lock Exchange {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "PE": "J",
                      "PE_rel": ""},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Overflow (NEMO / Petersen et al. 2015)
# ===========================================================================
# Reference: Petersen et al. (2015), Ocean Modelling 86, 93-113.
# DOI: 10.1016/j.ocemod.2014.12.004
# NEMO test cases: https://sites.nemo-ocean.io/user-guide/tests.html
#
# Dense water on a shallow shelf overflows and descends a continental slope.
# Tests numerical mixing near sloping topography.
# Adapted to global grids: cold dense water at high latitudes flows
# equatorward over a mid-latitude bathymetric ridge.
# ===========================================================================

def _init_overflow(state, grid_type, grid, z_coord):
    """Initialize overflow: dense water at high latitudes over bathymetric slope.

    Adapted from Petersen et al. (2015):
      - Poleward of 50 deg: cold (T=5 degC, dense)
      - Equatorward of 50 deg: warm (T=20 degC, light)
      - Smooth tanh transition at 50 deg latitude
      - Bathymetric ridge at ~40 deg: shelf at 500m, deep basin at 2000m
    """
    from legoesm.core.field import Field

    T_cold = 5.0
    T_warm = 20.0
    lat_front = np.radians(50.0)   # front position
    sigma_front = np.radians(5.0)  # transition width

    # Bathymetry: shelf (500m) poleward of 40 deg, deep (2000m) equatorward
    lat_shelf = np.radians(40.0)
    sigma_shelf = np.radians(7.0)
    d_shallow = 500.0
    d_deep = float(z_coord.H_max)

    lat, lon = _get_cell_latlon_rad(grid_type, grid)
    abs_lat = np.abs(lat)

    # Temperature: tanh transition at lat_front
    T_profile = T_warm + (T_cold - T_warm) * 0.5 * (
        1.0 + np.tanh((abs_lat - lat_front) / sigma_front))

    # Bathymetry: tanh transition at lat_shelf
    H_bathy_new = d_shallow + (d_deep - d_shallow) * 0.5 * (
        1.0 - np.tanh((abs_lat - lat_shelf) / sigma_shelf))

    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
        T_hat = state.T_hat.data
        T_grid = np.array(sh_synthesis_3d(grid, T_hat), dtype=np.float64)
        nlev = T_grid.shape[-1]
        mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
        for k in range(nlev):
            # Decay temperature perturbation with depth
            depth_frac = float(z_coord.z_full_ref[k] / z_coord.z_full_ref[-1])
            T_grid[..., k] = (T_profile * (1.0 - 0.5 * depth_frac) + 2.0 * depth_frac) * mask
        new_T_hat = sh_analysis_3d(grid, jnp.array(T_grid))
        # Note: spectral model doesn't easily support variable bathymetry
        return state._replace(T_hat=Field(new_T_hat))

    else:
        T_data = np.array(state.T.data, dtype=np.float64, copy=True)
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        nlev = T_data.shape[-1]
        for k in range(nlev):
            depth_frac = float(z_coord.z_full_ref[k] / z_coord.z_full_ref[-1])
            T_data[..., k] = (T_profile * (1.0 - 0.5 * depth_frac) + 2.0 * depth_frac) * mask

        # Update bathymetry
        H_bathy_new_masked = H_bathy_new * mask
        # Ensure minimum depth where ocean exists
        H_bathy_new_masked = np.where(mask > 0.5, np.maximum(H_bathy_new_masked, 50.0), 0.0)

        return state._replace(
            T=Field(jnp.array(T_data)),
            H_bathy=Field(jnp.array(H_bathy_new_masked)))


def run_overflow(tc: TestCase, output_dir: Path, days: float
                 ) -> tuple[str, float, str]:
    """Overflow: dense water descending a bathymetric slope (Petersen et al. 2015).

    Cold dense water at high latitudes flows equatorward over a mid-latitude
    ridge. Monitors PE evolution and plume descent.
    """
    H_max = 2000.0
    nlev = 20
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, nlev=nlev, H_max=H_max))
    state = _create_rest_state(tc, grid, z_coord, H_max=H_max)
    state = _init_overflow(state, tc.grid_type, grid, z_coord)

    pe_init = _compute_rpe(state, tc.grid_type, grid, z_coord)

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 30)

    check_fn = _make_check_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)
    base_scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)

    def scalar_fn(s):
        scalars = base_scalar_fn(s)
        pe = _compute_rpe(s, tc.grid_type, grid, z_coord)
        scalars["PE"] = pe
        if abs(pe_init) > 1e-30:
            scalars["PE_rel"] = (pe - pe_init) / abs(pe_init)
        else:
            scalars["PE_rel"] = 0.0
        return scalars

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Overflow ({tc.grid_type})", total_days=days,
        blowup_threshold=200.0)

    pe_drift = _compute_drift(diag.get("PE", []))
    pe_rel_final = diag["PE_rel"][-1] if diag.get("PE_rel") else 0.0
    # All grids now use same physical units
    T_drift = _compute_drift(diag.get("mean_T", []))
    notes = (f"PE drift={pe_drift:.2e}, PE_rel={pe_rel_final:.4e}, "
             f"T drift={T_drift:.2e}")

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": nlev, "H_max": H_max,
        "reference": "Petersen et al. 2015, DOI:10.1016/j.ocemod.2014.12.004",
        "PE_drift": pe_drift, "PE_rel_final": pe_rel_final,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Overflow {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("eta", "SSH (m)", "RdBu_r"),
            ("SST", "SST (degC)", "RdYlBu_r"),
        ],
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta",
        heat_key="mean_T",
        salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "PE": "J"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Stommel Gyre Tracer (Hecht et al. 2000)
# ===========================================================================
# Reference: Hecht, Wingate, Kasahara (2000), "A better, more discriminating
# test problem for ocean tracer transport", Ocean Modelling 2, 1-15.
# DOI: 10.1016/S1463-5003(00)00004-4
#
# Wind-driven Stommel gyre with a passive tracer (salinity field).
# The tracer blob is advected through the highly sheared western boundary
# current, which is a severe test of advection scheme accuracy.
# Based on MITgcm barotropic gyre setup:
#   - Domain: global (~1200 km effective gyre scale)
#   - Wind: tau_x = -tau0 * cos(pi * y / L_y)
#   - Viscosity: A_h to resolve Munk layer
#   - Linear bottom drag
# ===========================================================================

def _init_stommel_gyre_tracer(state, grid_type, grid, z_coord):
    """Initialize Stommel gyre with passive salinity tracer blob.

    Uses the existing wind-driven gyre initialization for dynamics,
    then sets salinity as a passive tracer with a Gaussian blob
    in the subtropical gyre interior.
    """
    from legoesm.core.field import Field

    # First set up the wind-gyre dynamics (basin: 0-60E, 15-75N)
    state = _add_wind_gyre_forcing(
        state, grid_type, grid, z_coord,
        lon_west=0.0, lon_east=120.0, lat_south=15.0, lat_north=75.0,
    )

    # Add salinity tracer blob (Gaussian, centered at 35N, 30E — inside basin)
    lat, lon = _get_cell_latlon_rad(grid_type, grid)
    lat_c = np.radians(35.0)   # blob center latitude
    lon_c = np.radians(60.0)   # blob center longitude (basin midpoint)
    sigma = np.radians(10.0)   # blob width (~10 deg)
    S_bg = 35.0                # background salinity (PSU)
    S_amp = 2.0                # tracer perturbation amplitude

    r2 = (lat - lat_c)**2 + (np.cos(lat_c) * (lon - lon_c))**2
    S_blob = S_bg + S_amp * np.exp(-r2 / (2.0 * sigma**2))

    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
        S_hat = state.S_hat.data
        S_grid = np.array(sh_synthesis_3d(grid, S_hat), dtype=np.float64)
        nlev = S_grid.shape[-1]
        mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
        # Set top-level salinity as tracer, keep deeper levels uniform
        S_grid[..., 0] = S_blob * mask
        new_S_hat = sh_analysis_3d(grid, jnp.array(S_grid))
        return state._replace(S_hat=Field(new_S_hat))

    else:
        S_data = np.array(state.S.data, dtype=np.float64, copy=True)
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        # Surface salinity blob
        S_data[..., 0] = S_blob * mask
        return state._replace(S=Field(jnp.array(S_data)))


def run_stommel_gyre_tracer(tc: TestCase, output_dir: Path, days: float
                             ) -> tuple[str, float, str]:
    """Stommel gyre with passive tracer (Hecht et al. 2000).

    Wind-driven gyre with a salinity blob advected through the western
    boundary current. Monitors tracer conservation (integral, min, max)
    and transport through the sheared flow.
    """
    if tc.grid_type not in ("cubed_sphere", "latlon", "mpas"):
        raise NotImplementedError(
            f"Stommel gyre tracer not implemented for {tc.grid_type} grid "
            f"(no surface forcing support)")

    physics = _make_gyre_physics("single_gyre")
    grid, z_coord, config, model, coord_kind, lon_deg, lat_deg = (
        _create_ocean_setup(tc, physics=physics))
    rest = _create_rest_state(tc, grid, z_coord)
    state = _init_stommel_gyre_tracer(rest, tc.grid_type, grid, z_coord)

    # Store initial tracer integral for conservation check
    area = np.asarray(grid.grid_area, dtype=np.float64)
    S_init_sfc = np.asarray(state.S.data[..., 0], dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    S_integral_init = float(np.sum(S_init_sfc * area * mask))
    S_min_init = float(np.min(S_init_sfc[mask > 0.5]))
    S_max_init = float(np.max(S_init_sfc[mask > 0.5]))

    dt = DEFAULT_DT
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 40)

    check_fn = _make_check_fn(tc.grid_type)
    extract_fn = _make_extract_fn(tc.grid_type, grid, lon_deg, lat_deg)
    base_scalar_fn = _make_scalar_fn(tc.grid_type, grid, z_coord)

    def scalar_fn(s):
        scalars = base_scalar_fn(s)
        S_sfc = np.asarray(s.S.data[..., 0], dtype=np.float64)
        ocean = mask > 0.5
        scalars["S_min"] = float(np.min(S_sfc[ocean]))
        scalars["S_max"] = float(np.max(S_sfc[ocean]))
        scalars["S_integral"] = float(np.sum(S_sfc * area * mask))
        if abs(S_integral_init) > 1e-30:
            scalars["S_integral_rel"] = (
                (scalars["S_integral"] - S_integral_init) / abs(S_integral_init))
        else:
            scalars["S_integral_rel"] = 0.0
        return scalars

    state, snapshots, diag, wall, ok = _run_timeloop(
        lambda s, dt_: model.step(s, dt_), state, dt, n_steps,
        check_fn, scalar_fn, extract_fn, diag_every,
        lambda s: _key_array_fn(s, tc.grid_type),
        label=f"Stommel Tracer ({tc.grid_type})", total_days=days)

    S_int_drift = _compute_drift(diag.get("S_integral", []))
    S_min_final = diag["S_min"][-1] if diag.get("S_min") else 0
    S_max_final = diag["S_max"][-1] if diag.get("S_max") else 0
    # Check for new extrema (overshoots/undershoots)
    overshoot = max(0, S_max_final - S_max_init)
    undershoot = max(0, S_min_init - S_min_final)
    notes = (f"S integral drift={S_int_drift:.2e}, "
             f"overshoot={overshoot:.3f}, undershoot={undershoot:.3f}")

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth = -z_full

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "levels": z_coord.n_levels,
        "reference": "Hecht et al. 2000, DOI:10.1016/S1463-5003(00)00004-4",
        "S_integral_drift": S_int_drift,
        "S_overshoot": overshoot, "S_undershoot": undershoot,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-105: surface BLOWUP info if any

    field_specs = [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("SSS", "SSS (PSU)", "YlGnBu"),
    ]
    if tc.grid_type != "mpas":
        field_specs.append(("speed_sfc", "Surface speed (m/s)", "magma"))

    _save_case_diagnostics(
        output_dir, f"Stommel Tracer {tc.grid_type} {tc.resolution}",
        dt, diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=field_specs,
        field_3d_key="T_3d", level_values=depth,
        level_label="Depth (m)",
        vol_key="mean_eta", heat_key="mean_T", salt_key="mean_S",
        scalar_units={"mean_eta": "m", "mean_T": "degC", "mean_S": "PSU",
                      "S_min": "PSU", "S_max": "PSU", "S_integral": "PSU*m^2"},
        mesh=grid if coord_kind == "mpas" else None)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner dispatch
# ===========================================================================

RUNNERS: dict[str, Callable] = {
    "rest_state_stratified_with_land": run_rest_state,
    "rest_state_uniform_with_land": run_rest_state_uniform_ts,
    "rest_state_stratified_no_land": run_rest_state_no_land,
    "rest_state_uniform_no_land": run_rest_state_uniform_ts_no_land,
    "barotropic_wave": run_barotropic_wave,
    "barotropic_gyre": run_barotropic_gyre,
    "barotropic_double_gyre": run_barotropic_double_gyre,
    "barotropic_double_gyre_sin2": run_barotropic_double_gyre_sin2,
    "baroclinic_gyre": run_baroclinic_gyre,
    "baroclinic_gyre_cos": run_baroclinic_gyre_cos,
    "global_barotropic_wind": run_global_barotropic_wind,
    "global_barotropic_wind_1lev": run_global_barotropic_wind,
    "geostrophic_adjustment": run_geostrophic_adjustment,
    "phillips_two_layer": run_phillips_two_layer,
    "inertia_gravity_wave": run_inertia_gravity_wave,
    "lock_exchange": run_lock_exchange,
    "overflow": run_overflow,
    "stommel_gyre_tracer": run_stommel_gyre_tracer,
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
             "(e.g. rest_state, barotropic_gyre, barotropic_double_gyre)")
    p.add_argument(
        "--grid", type=str, default="all",
        choices=["cubed_sphere", "latlon", "mpas",
                 "mpas_regional", "latlon_regional", "cs_regional",
                 "all"],
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
        "--days", type=float, default=None,
        help="Override duration in days (overrides both normal and quick mode durations)")
    p.add_argument(
        "--list", action="store_true",
        help="List all test cases and exit")
    p.add_argument(
        "--replot", action="store_true",
        help="Skip simulations; regenerate all plots from existing NPZ data")
    return p


def filter_tests(tests: list[TestCase], args) -> list[TestCase]:
    filtered = tests
    if args.only != "all":
        # Support exact matching with "=" prefix (e.g., "=rest_state")
        if args.only.startswith("="):
            exact_name = args.only[1:]
            filtered = [t for t in filtered if t.case == exact_name]
        else:
            # Default substring matching
            filtered = [t for t in filtered if args.only in t.case]
    if args.grid != "all":
        filtered = [t for t in filtered if t.grid_type == args.grid]
    return filtered


_OCEAN_RES_DIR_WARNED: set = set()


def _select_ocean_resolution_dir(grid_dir, resolution_dirs):
    """iter-108 (codex iter-104 MEDIUM-6): prefer grid-typed
    resolution dirs over bare-numeric ones.

    Pre-iter-108 the ocean collector took ``resolution_dirs[0]``
    (filesystem order) which left stale ``16/`` dirs from
    pre-iter-102 runs shadowing fresh ``C16/`` / ``36x32/`` /
    ``ico3/`` dirs.  Now match the per-grid format from the
    iter-95/102 dispatch table:

    * cubed_sphere → ``C*``
    * latlon → ``*x*``
    * mpas → ``ico*``
    * spectral → ``T*``
    * regional grids → ``*km``

    If a typed dir is present, it wins over bare-numeric.  If
    no typed dir is found (pre-iter-95 pure-legacy tree), fall
    back to ``resolution_dirs[0]`` for graceful degradation.
    Warn ONCE per ``grid_dir`` if stale dirs are filtered out.
    """
    grid_name = grid_dir.name
    grid_typed_pattern = {
        "cubed_sphere": lambda n: n.startswith("C") and n[1:].isdigit(),
        "latlon": lambda n: "x" in n and all(p.isdigit() for p in n.split("x") if p),
        "mpas": lambda n: n.startswith("ico") and n[3:].isdigit(),
        "spectral": lambda n: n.startswith("T") and n[1:].isdigit(),
        "mpas_regional": lambda n: n.endswith("km") and n[:-2].isdigit(),
        "latlon_regional": lambda n: n.endswith("km") and n[:-2].isdigit(),
        "cs_regional": lambda n: n.endswith("km") and n[:-2].isdigit(),
    }
    matcher = grid_typed_pattern.get(grid_name)
    if matcher is not None:
        typed = [d for d in resolution_dirs if matcher(d.name)]
        if typed:
            chosen = sorted(typed)[0]
            stale = [d.name for d in resolution_dirs if d not in typed]
            if stale and grid_dir not in _OCEAN_RES_DIR_WARNED:
                _OCEAN_RES_DIR_WARNED.add(grid_dir)
                print(
                    f"    [comparison] {grid_name}: ignoring "
                    f"non-grid-typed resolution dirs "
                    f"{sorted(stale)} in favor of {chosen.name} "
                    f"(iter-108 prefers the grid-typed format "
                    f"from iter-95/102 dispatch).  Re-run with "
                    f"a clean output tree to remove stale dirs."
                )
            return chosen
    # Fall back to filesystem order for graceful degradation.
    return resolution_dirs[0]


def _collect_grid_results(test_case_dir: Path) -> dict:
    """Collect results from all grids that completed for this test case.

    iter-49: relaxed to match the iter-26 atmosphere-matrix collector
    pattern.  Previously required ALL THREE of ``mean_timeseries.csv``,
    ``snapshots_latlon.npz``, and ``results.txt``.  Now accepts EITHER
    ``snapshots_latlon.npz`` OR (``mean_timeseries.csv`` AND
    ``results.txt``) so that timeseries-only ocean runs (e.g.
    ``run_omip.py`` output without snapshots) can be cross-grid-plotted
    too.  The downstream ``_create_comparison_*`` functions already
    handle missing snapshots gracefully (they emit only the timeseries
    plot when snapshots are absent).

    Returns:
        dict mapping grid_type -> {timeseries, snapshots, metadata}
    """
    grid_results = {}

    for grid_dir in test_case_dir.iterdir():
        if not grid_dir.is_dir():
            continue

        # Find resolution subdirectory (e.g., C24, 36x72, ico3, T21)
        resolution_dirs = [d for d in grid_dir.iterdir() if d.is_dir()]
        if not resolution_dirs:
            continue
        # iter-108 (codex iter-104 MEDIUM-6): prefer grid-typed
        # resolution dirs (e.g., ``C24`` over ``24``) so stale
        # pre-iter-102 bare-numeric output trees don't shadow
        # fresh post-iter-102 grid-typed dirs.  Mirrors the
        # atmosphere collector iter-108 fix.
        resolution_dir = _select_ocean_resolution_dir(
            grid_dir, resolution_dirs)

        # iter-49: relaxed predicate.
        csv_file = resolution_dir / "mean_timeseries.csv"
        npz_file = resolution_dir / "snapshots_latlon.npz"
        results_file = resolution_dir / "results.txt"
        has_snapshots = npz_file.exists()
        has_timeseries = csv_file.exists() and results_file.exists()
        if not (has_snapshots or has_timeseries):
            continue

        # iter-50 codex MEDIUM: per-artifact load isolation.  A
        # corrupt/stale optional file should NOT drop an otherwise-
        # usable run.  Previously, ``except Exception`` around the
        # full load block meant a bad npz would also throw away the
        # CSV + results.txt.  Now each artifact load is wrapped
        # individually; a corrupted artifact becomes ``None`` and
        # the run is still collected on the other artifact.
        timeseries_df = None
        if csv_file.exists():
            try:
                timeseries_df = pd.read_csv(csv_file)
            except Exception as e:
                print(
                    f"Warning: Failed to read {csv_file} for "
                    f"{grid_dir.name}: {e}"
                )

        snapshots_data = None
        if has_snapshots:
            try:
                snapshots_data = np.load(npz_file)
            except Exception as e:
                print(
                    f"Warning: Failed to load {npz_file} for "
                    f"{grid_dir.name}: {e}"
                )

        metadata: dict = {}
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key.strip()] = value.strip()
            except Exception as e:
                print(
                    f"Warning: Failed to parse {results_file} for "
                    f"{grid_dir.name}: {e}"
                )

        # Re-check: after loading, the run must still satisfy the
        # collector predicate (at least one DATA half remains
        # usable).  iter-51 codex MEDIUM: the iter-50 check
        # ``... or not metadata`` was too strict — a valid
        # timeseries_df should keep the run even if results.txt
        # parsing failed.  The downstream
        # ``_create_comparison_summary`` already uses
        # ``metadata.get(..., 'N/A')`` so missing metadata is
        # rendered as N/A rather than crashing.
        if snapshots_data is None and timeseries_df is None:
            print(
                f"Warning: {grid_dir.name} had artifacts but all "
                f"data loads failed; dropping from cross-grid "
                f"collection"
            )
            continue

        grid_results[grid_dir.name] = {
            'timeseries': timeseries_df,
            'snapshots': snapshots_data,
            'metadata': metadata,
            'resolution': resolution_dir.name,
        }

    return grid_results


def _create_comparison_timeseries(test_case_dir: Path, grid_results: dict) -> None:
    """Create 4-panel time series comparison plot across all grids.

    iter-50 codex LOW: early-return if no grid has timeseries data;
    otherwise we'd emit an empty ``comparison_timeseries.png`` with
    legend warnings.  This is the snapshots-only-cross-grid case.
    """
    import matplotlib.pyplot as plt

    grids_with_timeseries = {
        name: data for name, data in grid_results.items()
        if data.get('timeseries') is not None
    }
    if not grids_with_timeseries:
        print(
            "    [iter-50] no grid has timeseries data; skipping "
            "comparison_timeseries.png"
        )
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Time Series Comparison - {test_case_dir.name}', fontsize=14, fontweight='bold')
    
    # Colors for different grids
    colors = {
        'cubed_sphere': 'blue', 'latlon': 'red', 'mpas': 'green',
        'spectral': 'orange', 'mpas_regional': 'tab:green',
        'latlon_regional': 'tab:red', 'cs_regional': 'tab:blue',
    }
    linestyles = {
        'cubed_sphere': '-', 'latlon': '-', 'mpas': '-',
        'spectral': '-', 'mpas_regional': '--',
        'latlon_regional': '--', 'cs_regional': '--',
    }
    
    for grid_name, data in grid_results.items():
        df = data['timeseries']
        # iter-49: skip grids that have only snapshots (no timeseries CSV).
        if df is None:
            continue
        color = colors.get(grid_name, 'black')
        ls = linestyles.get(grid_name, '-')

        # Panel 1: Mean eta evolution
        if 'mean_eta' in df.columns:
            axes[0,0].plot(df['time_days'], df['mean_eta'], label=grid_name, color=color, ls=ls)

        # Panel 2: Max |eta|
        if 'max_abs_eta' in df.columns:
            axes[0,1].plot(df['time_days'], df['max_abs_eta'], label=grid_name, color=color, ls=ls)

        # Panel 3: Max speed
        if 'max_speed' in df.columns:
            axes[1,0].plot(df['time_days'], df['max_speed'], label=grid_name, color=color, ls=ls)

        # Panel 4: Volume-weighted KE
        if 'mean_ke' in df.columns:
            axes[1,1].plot(df['time_days'], df['mean_ke'], label=grid_name, color=color, ls=ls)

    axes[0,0].set_ylabel('Mean η (m)')
    axes[0,0].set_title('Mean Sea Surface Height')
    axes[0,1].set_ylabel('Max |η| (m)')
    axes[0,1].set_title('Maximum SSH Amplitude')
    axes[1,0].set_ylabel('Max Speed (m/s)')
    axes[1,0].set_title('Maximum Speed')
    axes[1,1].set_ylabel('Mean KE (m²/s²)')
    axes[1,1].set_title('Volume-Weighted Kinetic Energy')

    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Set common x-label
    for ax in axes[1,:]:
        ax.set_xlabel('Time (days)')

    from datetime import datetime
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.tight_layout()

    # Save plot
    output_file = test_case_dir / "comparison_timeseries.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")


def _create_comparison_snapshots(test_case_dir: Path, grid_results: dict, field: str = 'eta') -> None:
    """Create 4-panel final snapshot comparison for a given field."""
    import matplotlib.pyplot as plt
    
    # Determine the test case for field ranges
    test_case = _extract_test_case_name(test_case_dir.name)
    field_ranges = FIELD_RANGES.get(test_case, {})
    vmin, vmax = field_ranges.get(field, (None, None))

    # If no explicit range, compute shared range across all grids
    if vmin is None or vmax is None:
        all_vals = []
        for data in grid_results.values():
            snapshots = data['snapshots']
            # iter-49: skip timeseries-only grids (no snapshots payload).
            if snapshots is None:
                continue
            if field in snapshots.files:
                fd = snapshots[field]
                if fd.ndim == 3:
                    fd = fd[-1]
                if 'land_mask' in snapshots.files:
                    lm = snapshots['land_mask']
                    if lm.ndim == 3:
                        lm = lm[-1]
                    fd = np.where(lm > 0.5, fd, np.nan)
                all_vals.append(fd.ravel())
            else:
                ffiles = [f for f in snapshots.files if f.startswith(f'{field}_step')]
                if ffiles:
                    steps = [int(f.split('_step')[1]) for f in ffiles]
                    fd = snapshots[f'{field}_step{max(steps)}']
                    all_vals.append(fd.ravel())
        if all_vals:
            combined = np.concatenate(all_vals)
            finite = combined[np.isfinite(combined)]
            if len(finite) > 0:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    
    # Set up grid layout (2x2 for up to 4 grids + space for colorbar)
    n_grids = len(grid_results)
    if n_grids <= 2:
        nrows, ncols = 1, 3  # Extra column for colorbar
    else:
        nrows, ncols = 2, 3  # Extra column for colorbar
        
    fig = plt.figure(figsize=(14, 10))  # Wider to accommodate colorbar
    
    # Create subplots with specific width ratios: plots get most space, colorbar gets less
    gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1, 0.05] if ncols == 3 else [1, 1, 1, 0.05])
    
    axes = []
    for i in range(nrows):
        for j in range(ncols - 1):  # Don't include colorbar column
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Determine simulation time of the final snapshot from any grid's data
    sim_time_str = ""
    for data in grid_results.values():
        snapshots_any = data['snapshots']
        # iter-49: timeseries-only grids have snapshots=None.
        if snapshots_any is None:
            continue
        if 'times_days' in snapshots_any.files:
            t_final = float(snapshots_any['times_days'][-1])
            sim_time_str = f" (t = {t_final:.2f} days)"
            break

    fig.suptitle(f'Final {field.upper()} Snapshots - {test_case_dir.name}{sim_time_str}',
                 fontsize=14, fontweight='bold')
    
    # Choose colormap based on field
    if field == 'eta':
        cmap = 'RdBu_r'
    elif field in ['SST', 'T']:
        cmap = 'plasma'
    elif field in ['SSS', 'S']:
        cmap = 'viridis'
    elif field in ('w_133m', 'w_sfc'):
        cmap = 'RdBu_r'  # Diverging colormap for vertical velocity (upwelling/downwelling)
    else:
        cmap = 'viridis'
    
    im = None
    for i, (grid_name, data) in enumerate(grid_results.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        snapshots = data['snapshots']
        # iter-49: skip timeseries-only grids.
        if snapshots is None:
            ax.set_visible(False)
            continue

        # Determine plot extent: prefer source coordinate range (accurate
        # for regional unstructured meshes) over the regridded grid range.
        if 'source_lon_range' in snapshots.files:
            src_lon = snapshots['source_lon_range']
            src_lat = snapshots['source_lat_range']
            plot_extent = [float(src_lon[0]), float(src_lon[1]),
                          float(src_lat[0]), float(src_lat[1])]
        elif 'lon' in snapshots.files:
            lon_arr = snapshots['lon']
            lat_arr = snapshots['lat']
            plot_extent = [float(lon_arr.min()), float(lon_arr.max()),
                          float(lat_arr.min()), float(lat_arr.max())]
        else:
            plot_extent = [0, 360, -90, 90]

        # Handle both new format (field as time series) and old format (field_stepN)
        if field in snapshots.files:
            # NEW FORMAT: field is a time series array, take final timestep
            field_data = snapshots[field]
            if field_data.ndim >= 2:  # At least 2D (could be 3D with time)
                if field_data.ndim == 3:  # Time series: (n_times, nlat, nlon)
                    final_field = field_data[-1]  # Last time
                else:  # Single timestep: (nlat, nlon) - shouldn't happen with new format
                    final_field = field_data

                # Apply land masking if available
                if 'land_mask' in snapshots.files:
                    land_mask_data = snapshots['land_mask']
                    if land_mask_data.ndim == 3:  # Time series
                        land_mask_final = land_mask_data[-1]  # Last time
                    else:  # Single timestep
                        land_mask_final = land_mask_data
                    # Mask land areas (where land_mask ≤ 0.5) with NaN
                    final_field = np.where(land_mask_final > 0.5, final_field, np.nan)

                # Crop regridded array to source domain extent
                if 'lon' in snapshots.files:
                    lat_1d = np.asarray(snapshots['lat'])
                    lon_1d = np.asarray(snapshots['lon'])
                    r0 = max(int(np.searchsorted(lat_1d, plot_extent[2])) - 1, 0)
                    r1 = min(int(np.searchsorted(lat_1d, plot_extent[3])) + 2,
                             len(lat_1d))
                    c0 = max(int(np.searchsorted(lon_1d, plot_extent[0])) - 1, 0)
                    c1 = min(int(np.searchsorted(lon_1d, plot_extent[1])) + 2,
                             len(lon_1d))
                    if (r1 - r0) < final_field.shape[0] or (c1 - c0) < final_field.shape[1]:
                        final_field = final_field[r0:r1, c0:c1]
                        plot_extent = [float(lon_1d[c0]),
                                       float(lon_1d[min(c1, len(lon_1d)-1)]),
                                       float(lat_1d[r0]),
                                       float(lat_1d[min(r1, len(lat_1d)-1)])]

                # Create the plot with consistent color scale
                im = ax.imshow(final_field, origin='lower', aspect='auto', cmap=cmap,
                              extent=plot_extent, vmin=vmin, vmax=vmax)
                ax.set_title(f'{grid_name} ({data["resolution"]})')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            else:
                ax.text(0.5, 0.5, f'{field} wrong shape', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f'{grid_name} ({data["resolution"]})')
        else:
            # OLD FORMAT: Find final timestep field (fields stored as field_stepN)
            field_files = [f for f in snapshots.files if f.startswith(f'{field}_step')]
            if field_files:
                # Get the highest step number
                step_numbers = [int(f.split('_step')[1]) for f in field_files]
                final_step = max(step_numbers)
                final_field_name = f'{field}_step{final_step}'
                final_field = snapshots[final_field_name]
                
                # Apply land masking if available (old format)
                land_mask_name = f'land_mask_step{final_step}'
                if land_mask_name in snapshots.files:
                    land_mask_final = snapshots[land_mask_name]
                    # Mask land areas (where land_mask ≤ 0.5) with NaN
                    final_field = np.where(land_mask_final > 0.5, final_field, np.nan)

                # Crop to bounding box of non-NaN data for regional grids
                valid = np.isfinite(final_field)
                if valid.any():
                    rows = np.where(valid.any(axis=1))[0]
                    cols = np.where(valid.any(axis=0))[0]
                    r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, final_field.shape[0])
                    c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, final_field.shape[1])
                    if (r1 - r0) * (c1 - c0) < 0.8 * final_field.size:
                        lat_1d = np.linspace(plot_extent[2], plot_extent[3], final_field.shape[0])
                        lon_1d = np.linspace(plot_extent[0], plot_extent[1], final_field.shape[1])
                        final_field = final_field[r0:r1, c0:c1]
                        plot_extent = [float(lon_1d[c0]), float(lon_1d[min(c1, len(lon_1d)-1)]),
                                       float(lat_1d[r0]), float(lat_1d[min(r1, len(lat_1d)-1)])]

                # Create the plot with consistent color scale
                im = ax.imshow(final_field, origin='lower', aspect='auto', cmap=cmap,
                              extent=plot_extent, vmin=vmin, vmax=vmax)
                ax.set_title(f'{grid_name} ({data["resolution"]})')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            else:
                ax.text(0.5, 0.5, f'{field} not available', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f'{grid_name} ({data["resolution"]})')
    
    # Hide unused subplots
    for i in range(n_grids, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar in dedicated space
    if im is not None:
        # Create colorbar axes in the rightmost column, spanning all rows
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        if field == 'eta':
            cbar.set_label('Sea Surface Height (m)')
        elif field in ['SST', 'T']:
            cbar.set_label('Temperature (°C)')
        elif field in ['SSS', 'S']:
            cbar.set_label('Salinity (PSU)')

    from datetime import datetime
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.tight_layout()

    # Save plot
    output_file = test_case_dir / f"comparison_snapshots_{field}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")


def _create_comparison_summary(test_case_dir: Path, grid_results: dict) -> None:
    """Create cross-grid metrics comparison table."""
    
    summary_file = test_case_dir / "comparison_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"{test_case_dir.name} - Cross-Grid Comparison\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Grid':<12} {'Status':<8} {'Resolution':<10} {'Wall Time':<10} {'Notes':<30}\n")
        f.write("-" * 80 + "\n")
        
        # Find metrics for comparison
        eta_drifts = []
        T_drifts = []
        wall_times = []
        
        for grid_name, data in grid_results.items():
            metadata = data['metadata']
            status = metadata.get('status', 'N/A')
            resolution = data['resolution']
            wall_time = metadata.get('wall_time', 'N/A')
            notes = metadata.get('notes', '')
            
            f.write(f"{grid_name:<12} {status:<8} {resolution:<10} {wall_time:<10} {notes:<30}\n")
            
            # Extract drift metrics from notes if available
            if 'eta drift=' in notes:
                try:
                    eta_drift_str = notes.split('eta drift=')[1].split(',')[0].split()[0]
                    eta_drifts.append((grid_name, float(eta_drift_str)))
                except:
                    pass
            
            if 'T drift=' in notes:
                try:
                    T_drift_str = notes.split('T drift=')[1].split(',')[0].split()[0]
                    T_drifts.append((grid_name, float(T_drift_str)))
                except:
                    pass
            
            if wall_time != 'N/A':
                try:
                    wall_time_val = float(wall_time.replace('s', ''))
                    wall_times.append((grid_name, wall_time_val))
                except:
                    pass
        
        f.write("-" * 80 + "\n")
        
        # Summary statistics
        if eta_drifts:
            best_eta = min(eta_drifts, key=lambda x: abs(x[1]))
            worst_eta = max(eta_drifts, key=lambda x: abs(x[1]))
            f.write(f"Best η drift:    {best_eta[0]} ({best_eta[1]:.2e})\n")
            f.write(f"Worst η drift:   {worst_eta[0]} ({worst_eta[1]:.2e})\n")
        
        if T_drifts:
            best_T = min(T_drifts, key=lambda x: abs(x[1]))
            worst_T = max(T_drifts, key=lambda x: abs(x[1]))
            f.write(f"Best T drift:    {best_T[0]} ({best_T[1]:.2e})\n")
            f.write(f"Worst T drift:   {worst_T[0]} ({worst_T[1]:.2e})\n")
        
        if wall_times:
            fastest = min(wall_times, key=lambda x: x[1])
            slowest = max(wall_times, key=lambda x: x[1])
            f.write(f"Fastest:         {fastest[0]} ({fastest[1]:.1f}s)\n")
            f.write(f"Slowest:         {slowest[0]} ({slowest[1]:.1f}s)\n")
    
    print(f"    Saved: {summary_file.name}")


def _create_comparison_evolution(test_case_dir: Path, grid_results: dict,
                                  field: str = 'eta', max_times: int = 6) -> None:
    """Create a grid-vs-time evolution comparison: rows=grids, cols=time steps.

    Each row is a different grid, each column a snapshot in time.
    All panels share a common colorbar so fields are directly comparable.
    Skipped for rest-state cases where fields don't evolve.
    """
    import matplotlib.pyplot as plt

    # Skip for rest states — nothing evolves
    if test_case_dir.name.startswith("rest_state"):
        return

    test_case = _extract_test_case_name(test_case_dir.name)
    field_ranges = FIELD_RANGES.get(test_case, {})

    # Choose colormap
    if field == 'eta':
        cmap = 'RdBu_r'
    elif field in ('SST', 'T'):
        cmap = 'plasma'
    elif field in ('speed_sfc',):
        cmap = 'magma'
    elif field in ('w_133m', 'w_sfc'):
        cmap = 'RdBu_r'  # Diverging colormap for vertical velocity (upwelling/downwelling)
    else:
        cmap = 'viridis'

    grid_names = list(grid_results.keys())
    n_grids = len(grid_names)

    # Determine number of time steps available (use the grid with most)
    n_times_per_grid = {}
    for gname, data in grid_results.items():
        snaps = data['snapshots']
        # iter-49: skip timeseries-only grids.
        if snaps is None:
            continue
        if field in snaps.files and snaps[field].ndim == 3:
            n_times_per_grid[gname] = snaps[field].shape[0]
        elif 'times_days' in snaps.files:
            # Grid doesn't have this field, but use times from other data
            n_times_per_grid[gname] = len(snaps['times_days'])
        else:
            # Old format: count step files or use a default
            ffiles = [f for f in snaps.files if f.startswith(f'{field}_step')]
            n_times_per_grid[gname] = len(ffiles) if ffiles else max_times

    if not n_times_per_grid:
        return  # No grids at all

    # Ensure we have at least 2 time steps from grids that actually have the field
    has_field_times = [
        nt for gname, nt in n_times_per_grid.items()
        if grid_results[gname]['snapshots'] is not None  # iter-49 guard
        and field in grid_results[gname]['snapshots'].files
    ]
    if not has_field_times or max(has_field_times) < 2:
        return  # Need at least 2 time steps for evolution

    n_times_max = max(n_times_per_grid.values())
    n_cols = min(max_times, n_times_max)

    # Collect all fields for shared color range
    all_fields = []  # list of 2D arrays
    grid_field_data = {}  # gname -> list of (time_label, 2D_array or None)
    for gname, data in grid_results.items():
        snaps = data['snapshots']
        # iter-49: skip timeseries-only grids.
        if snaps is None:
            continue
        times_days = snaps['times_days'] if 'times_days' in snaps.files else None

        if field in snaps.files and snaps[field].ndim == 3:
            field_3d = snaps[field]  # (n_times, nlat, nlon)
            nt = field_3d.shape[0]
            # Select evenly spaced time indices
            if nt > n_cols:
                indices = np.linspace(0, nt - 1, n_cols).astype(int)
            else:
                indices = np.arange(nt)

            entries = []
            for idx in indices:
                f2d = field_3d[idx].copy()
                if 'land_mask' in snaps.files:
                    lm = snaps['land_mask']
                    lm2d = lm[idx] if lm.ndim == 3 else lm
                    f2d = np.where(lm2d > 0.5, f2d, np.nan)
                t_label = f"{times_days[idx]:.1f}d" if times_days is not None else f"t{idx}"
                entries.append((t_label, f2d))
                all_fields.append(f2d)
            grid_field_data[gname] = entries
        else:
            # Field not available - create placeholder entries
            entries = []
            for i in range(n_cols):
                if times_days is not None and len(times_days) > i:
                    t_label = f"{times_days[i]:.1f}d"
                else:
                    t_label = f"t{i}"
                entries.append((t_label, None))  # None indicates missing data
            grid_field_data[gname] = entries

    if not grid_field_data or not all_fields:
        return

    # Actual number of columns (may differ per grid; use max)
    actual_cols = max(len(v) for v in grid_field_data.values())
    # iter-49: only iterate over grids that actually contributed
    # snapshot data — timeseries-only grids were skipped above.
    actual_grids = list(grid_field_data.keys())
    n_rows = len(actual_grids)

    fig, axes = plt.subplots(n_rows, actual_cols,
                              figsize=(3.5 * actual_cols, 3.0 * n_rows),
                              squeeze=False)

    for i_row, gname in enumerate(actual_grids):
        entries = grid_field_data[gname]

        # Color range is computed per-panel below so that spatial patterns
        # within each snapshot are visible (temporal trends otherwise
        # dominate the shared range and wash out spatial structure).

        # Get plot extent — prefer source coordinate range for accurate
        # regional domain cropping on unstructured meshes.
        snaps = grid_results[gname]['snapshots']
        if 'source_lon_range' in snaps.files:
            src_lon = snaps['source_lon_range']
            src_lat = snaps['source_lat_range']
            extent = [float(src_lon[0]), float(src_lon[1]),
                      float(src_lat[0]), float(src_lat[1])]
        elif 'lon' in snaps.files:
            lon_arr, lat_arr = snaps['lon'], snaps['lat']
            extent = [float(lon_arr.min()), float(lon_arr.max()),
                      float(lat_arr.min()), float(lat_arr.max())]
        else:
            extent = [0, 360, -90, 90]

        # Crop regridded array to source domain extent
        crop_slices = None
        if 'lon' in snaps.files:
            lat_1d = np.asarray(snaps['lat'])
            lon_1d = np.asarray(snaps['lon'])
            r0 = max(int(np.searchsorted(lat_1d, extent[2])) - 1, 0)
            r1 = min(int(np.searchsorted(lat_1d, extent[3])) + 2,
                     len(lat_1d))
            c0 = max(int(np.searchsorted(lon_1d, extent[0])) - 1, 0)
            c1 = min(int(np.searchsorted(lon_1d, extent[1])) + 2,
                     len(lon_1d))
            # Find a non-None entry to get the shape
            sample_shape = None
            if entries:
                for _, f2d in entries:
                    if f2d is not None:
                        sample_shape = f2d.shape
                        break
            if sample_shape and ((r1 - r0) < sample_shape[0]
                                 or (c1 - c0) < sample_shape[1]):
                crop_slices = (r0, r1, c0, c1)
                extent = [float(lon_1d[c0]),
                          float(lon_1d[min(c1, len(lon_1d)-1)]),
                          float(lat_1d[r0]),
                          float(lat_1d[min(r1, len(lat_1d)-1)])]

        for i_col in range(actual_cols):
            ax = axes[i_row, i_col]
            if i_col < len(entries):
                t_label, f2d = entries[i_col]
                if f2d is not None:
                    # Field data available
                    if crop_slices is not None:
                        r0, r1, c0, c1 = crop_slices
                        f2d = f2d[r0:r1, c0:c1]
                    # Per-panel color range
                    if field in field_ranges and field_ranges[field] != (None, None):
                        p_vmin, p_vmax = field_ranges[field]
                    else:
                        pf = f2d.ravel()
                        pf = pf[np.isfinite(pf)]
                        if pf.size:
                            p_vmin, p_vmax = float(pf.min()), float(pf.max())
                        else:
                            p_vmin, p_vmax = None, None
                    panel_im = ax.imshow(f2d, origin='lower', aspect='auto',
                                         cmap=cmap, extent=extent,
                                         vmin=p_vmin, vmax=p_vmax)
                    fig.colorbar(panel_im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, f'{field} not available',
                            transform=ax.transAxes,
                            ha='center', va='center', fontsize=9)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                if i_row == 0:
                    ax.set_title(t_label, fontsize=9)
                if i_col == 0:
                    res = grid_results[gname]['resolution']
                    ax.set_ylabel(f"{gname}\n({res})", fontsize=9)
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("")
                ax.tick_params(labelsize=7)
            else:
                ax.set_visible(False)

    fig.suptitle(f"{field.upper()} Evolution — {test_case_dir.name}",
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    out = test_case_dir / f"comparison_evolution_{field}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.name}")


def _create_comparison_vertical_section(test_case_dir: Path, grid_results: dict) -> None:
    """Create meridional vertical cross-section comparison for baroclinic gyre."""
    import numpy as np
    
    # Check that we have T_3d data for all grids
    grids_with_T3d = {}
    for grid_name, data in grid_results.items():
        snapshots = data['snapshots']
        # iter-49: skip timeseries-only grids.
        if snapshots is None:
            continue
        if 'T_3d' in snapshots.files:
            grids_with_T3d[grid_name] = data

    if len(grids_with_T3d) < 2:
        return
    
    # Domain parameters for baroclinic gyre (60° longitude × 60° latitude)
    lon_middle = 60.0  # Middle of 0-120° domain
    lat_range = (-90, 90)  # Full latitude range for averaging
    
    fig, axes = plt.subplots(1, len(grids_with_T3d), figsize=(5 * len(grids_with_T3d), 6),
                             sharey=True)
    if len(grids_with_T3d) == 1:
        axes = [axes]
    
    for idx, (grid_name, data) in enumerate(grids_with_T3d.items()):
        ax = axes[idx]
        snapshots = data['snapshots']
        
        # Load final temperature field (use last time step)
        T_3d_all = np.asarray(snapshots['T_3d'], dtype=np.float64)
        lon_deg = np.asarray(snapshots['lon'], dtype=np.float64) 
        lat_deg = np.asarray(snapshots['lat'], dtype=np.float64)
        
        # Handle different data structures - T_3d is (time, lat, lon, lev) 
        if T_3d_all.ndim == 4:  # (time, lat, lon, lev)
            T_3d = T_3d_all[-1]  # Take final time step → (lat, lon, lev)
        elif T_3d_all.ndim == 3:  # (lat, lon, lev) - already final state  
            T_3d = T_3d_all
        else:
            print(f"Warning: unexpected T_3d shape {T_3d_all.shape} for {grid_name}")
            continue
        
        # Get depth levels from the first grid (they should be the same)  
        if idx == 0:
            # Try to get depth data from results.txt or default levels
            results_file = test_case_dir / grid_name / 'results.txt'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    content = f.read()
                    # Look for depth line
                    for line in content.split('\n'):
                        if line.startswith('depth:'):
                            import ast
                            depth_str = line.split(':', 1)[1].strip()
                            depth_data = np.array(ast.literal_eval(depth_str))
                            break
                    else:
                        # Fallback to default depth levels
                        depth_data = np.array([26.19, 133.86, 352.12, 680.95, 1120.37, 
                                             1670.37, 2330.95, 3102.12, 3983.86, 4976.19])
            else:
                # Fallback to default depth levels
                depth_data = np.array([26.19, 133.86, 352.12, 680.95, 1120.37,
                                     1670.37, 2330.95, 3102.12, 3983.86, 4976.19])
        
        # Extract meridional section at domain middle (longitude = 60°)
        # Find closest longitude index to domain middle  
        lon_idx = np.argmin(np.abs(lon_deg - lon_middle))
        
        # Average over a few longitude points for smoother section
        lon_indices = slice(max(0, lon_idx-2), min(len(lon_deg), lon_idx+3))
        T_section = np.nanmean(T_3d[:, lon_indices, :], axis=1)  # T_3d is (lat, lon, lev) → (lat, lev)
        
        # Apply land mask if available (simplified for now)
        if 'land_mask' in snapshots.files:
            land_mask = np.asarray(snapshots['land_mask'], dtype=np.float64)
            # For now, skip land masking to get basic functionality working
            # TODO: Fix land mask broadcasting for vertical sections
            pass
        
        # Compute depth interfaces for proper plotting
        def compute_level_interfaces(level_centers):
            if len(level_centers) == 1:
                return np.array([0.0, 2 * level_centers[0]])
            interfaces = np.zeros(len(level_centers) + 1)
            interfaces[0] = 0.0
            interfaces[1:-1] = 0.5 * (level_centers[:-1] + level_centers[1:])
            interfaces[-1] = level_centers[-1] + (level_centers[-1] - interfaces[-2])
            return interfaces
        
        level_interfaces = compute_level_interfaces(depth_data)
        
        # Create coordinate meshes for pcolormesh
        # Ensure lat_interfaces matches the T_section shape
        if len(lat_deg) > 1:
            lat_step = (lat_deg[-1] - lat_deg[0]) / (len(lat_deg) - 1)
            lat_interfaces = np.linspace(lat_deg[0] - 0.5 * lat_step,
                                       lat_deg[-1] + 0.5 * lat_step,
                                       len(lat_deg) + 1)
        else:
            lat_interfaces = np.array([lat_deg[0] - 1.0, lat_deg[0] + 1.0])
        
        X, Y = np.meshgrid(lat_interfaces, level_interfaces)
        
        # Verify dimensions match for pcolormesh
        expected_lat_size = len(lat_interfaces) - 1  # pcolormesh expects one less than interfaces
        expected_lev_size = len(level_interfaces) - 1
        if T_section.shape != (expected_lat_size, expected_lev_size):
            print(f"Warning: T_section shape {T_section.shape} doesn't match expected "
                  f"({expected_lat_size}, {expected_lev_size}) for {grid_name}")
            continue
        
        # Plot with adaptive colormap range
        # For baroclinic_gyre, use data-adaptive range to show circulation patterns
        T_finite = T_section.T[np.isfinite(T_section.T)]
        if len(T_finite) > 0:
            vmin, vmax = float(np.nanmin(T_finite)), float(np.nanmax(T_finite))
        else:
            vmin, vmax = 2, 20  # fallback for edge cases
        
        im = ax.pcolormesh(X, Y, T_section.T, cmap='RdYlBu_r', 
                          vmin=vmin, vmax=vmax, shading='flat')
        
        ax.set_title(f'{grid_name}', fontsize=12)
        ax.set_xlabel('Latitude (°)')
        if idx == 0:
            ax.set_ylabel('Depth (m)')
        ax.set_ylim(0, level_interfaces[-1])  # Set y-limits to actual depth range
        ax.invert_yaxis()  # Surface at top, deep at bottom
        ax.grid(True, alpha=0.3)
        
        # Set x-limits to regional domain extent for consistent comparison
        # Use the actual experiment domain, not the interpolation grid extent
        regional_lat_min, regional_lat_max = 15.0, 75.0  # Baroclinic gyre domain
        ax.set_xlim(regional_lat_min, regional_lat_max)
        
        # Add domain boundaries for regional grids
        if 'regional' in grid_name:
            ax.axvline(15, color='white', linewidth=2, linestyle='--', alpha=0.8)
            ax.axvline(75, color='white', linewidth=2, linestyle='--', alpha=0.8)
    
    # Add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    fig.suptitle(f'{test_case_dir.name} — Meridional Temperature Section (60°E)', 
                 fontsize=14, y=0.95)
    
    # Save the plot
    out_file = test_case_dir / 'comparison_vertical_section.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_file.name}")


def _create_comparison_vertical_evolution(
    test_case_dir: Path, grid_results: dict, max_times: int = 6,
) -> None:
    """Meridional T cross-section evolution: rows = grids, cols = time steps.

    Each panel shows the latitude–depth temperature section at 60°E,
    with a per-panel colorbar so that spatial patterns within each
    snapshot are visible even when the full-depth stratification
    dominates the overall range.
    """
    import matplotlib.pyplot as plt

    # Skip rest-state cases
    if test_case_dir.name.startswith("rest_state"):
        return

    # Gather grids that have T_3d
    grids_with_T3d = {}
    for gname, data in grid_results.items():
        snaps = data["snapshots"]
        # iter-49: skip timeseries-only grids.
        if snaps is None:
            continue
        if "T_3d" in snaps.files:
            grids_with_T3d[gname] = data
    if len(grids_with_T3d) < 1:
        return

    # Read depth levels from results.txt of first grid
    first_grid = next(iter(grids_with_T3d))
    results_file = (test_case_dir / first_grid / "results.txt")
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.startswith("depth:"):
                    import ast as _ast
                    depth_data = np.array(
                        _ast.literal_eval(line.split(":", 1)[1].strip()))
                    break
            else:
                depth_data = np.array([26.19, 133.86, 352.12, 680.95,
                                       1120.37, 1670.37, 2330.95, 3102.12,
                                       3983.86, 4976.19])
    else:
        depth_data = np.array([26.19, 133.86, 352.12, 680.95, 1120.37,
                               1670.37, 2330.95, 3102.12, 3983.86, 4976.19])

    # Level interfaces for pcolormesh
    def _level_interfaces(centers):
        if len(centers) == 1:
            return np.array([0.0, 2 * centers[0]])
        ifc = np.zeros(len(centers) + 1)
        ifc[0] = 0.0
        ifc[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        ifc[-1] = centers[-1] + (centers[-1] - ifc[-2])
        return ifc

    level_ifc = _level_interfaces(depth_data)
    lon_middle = 60.0

    grid_names = list(grids_with_T3d.keys())
    n_grids = len(grid_names)

    # Determine number of time columns
    n_times_max = max(
        grids_with_T3d[g]["snapshots"]["T_3d"].shape[0]
        for g in grid_names
        if grids_with_T3d[g]["snapshots"]["T_3d"].ndim == 4)
    n_cols = min(max_times, n_times_max)

    fig, axes = plt.subplots(n_grids, n_cols,
                              figsize=(3.5 * n_cols, 3.5 * n_grids),
                              squeeze=False)

    for i_row, gname in enumerate(grid_names):
        snaps = grids_with_T3d[gname]["snapshots"]
        T_3d_all = np.asarray(snaps["T_3d"], dtype=np.float64)
        lon_deg = np.asarray(snaps["lon"], dtype=np.float64)
        lat_deg = np.asarray(snaps["lat"], dtype=np.float64)
        times_days = (np.asarray(snaps["times_days"], dtype=np.float64)
                      if "times_days" in snaps.files else None)

        if T_3d_all.ndim != 4:
            continue  # need (time, lat, lon, lev)
        nt = T_3d_all.shape[0]
        indices = (np.linspace(0, nt - 1, n_cols).astype(int)
                   if nt > n_cols else np.arange(nt))

        # Longitude slice
        lon_idx = int(np.argmin(np.abs(lon_deg - lon_middle)))
        lon_sl = slice(max(0, lon_idx - 2), min(len(lon_deg), lon_idx + 3))

        # Latitude interfaces
        if len(lat_deg) > 1:
            dlat = (lat_deg[-1] - lat_deg[0]) / (len(lat_deg) - 1)
            lat_ifc = np.linspace(lat_deg[0] - 0.5 * dlat,
                                  lat_deg[-1] + 0.5 * dlat,
                                  len(lat_deg) + 1)
        else:
            lat_ifc = np.array([lat_deg[0] - 1.0, lat_deg[0] + 1.0])

        X, Y = np.meshgrid(lat_ifc, level_ifc)

        # Use the initial condition (t=0) range for all panels so that
        # any departure from the initial stratification is visible.
        T_ic = T_3d_all[0]  # (lat, lon, lev)
        T_sec_ic = np.nanmean(T_ic[:, lon_sl, :], axis=1)
        ic_finite = T_sec_ic[np.isfinite(T_sec_ic)]
        if ic_finite.size:
            row_vmin, row_vmax = float(ic_finite.min()), float(ic_finite.max())
        else:
            row_vmin, row_vmax = 2, 20

        for i_col, ti in enumerate(indices):
            ax = axes[i_row, i_col]
            T_3d = T_3d_all[ti]  # (lat, lon, lev)
            T_sec = np.nanmean(T_3d[:, lon_sl, :], axis=1)  # (lat, lev)

            if T_sec.shape != (len(lat_ifc) - 1, len(level_ifc) - 1):
                ax.set_visible(False)
                continue

            im = ax.pcolormesh(X, Y, T_sec.T, cmap="RdYlBu_r",
                               vmin=row_vmin, vmax=row_vmax, shading="flat")
            ax.invert_yaxis()
            ax.set_xlim(15, 75)
            ax.set_ylim(level_ifc[-1], 0)
            ax.tick_params(labelsize=7)

            if i_row == 0:
                t_label = (f"{times_days[ti]:.1f}d"
                           if times_days is not None else f"t{ti}")
                ax.set_title(t_label, fontsize=9)
            if i_col == 0:
                res = grid_results[gname]["resolution"]
                ax.set_ylabel(f"{gname}\n({res})\nDepth (m)", fontsize=9)
            else:
                ax.set_ylabel("")
            if i_row == n_grids - 1:
                ax.set_xlabel("Latitude (°)", fontsize=8)
            else:
                ax.set_xlabel("")

        # Hide unused columns
        for i_col in range(len(indices), n_cols):
            axes[i_row, i_col].set_visible(False)

        # One shared colorbar per row (all panels use the IC range)
        if im is not None:
            row_axes = [axes[i_row, c] for c in range(n_cols)
                        if axes[i_row, c].get_visible()]
            fig.colorbar(im, ax=row_axes, fraction=0.02, pad=0.02,
                         label="T (°C)")

    fig.suptitle(
        f"T Section Evolution at 60°E — {test_case_dir.name}",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = test_case_dir / "comparison_evolution_vertical_section.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.name}")


def _create_cross_grid_comparisons(test_case_dir: Path, grid_results: dict) -> None:
    """Create all cross-grid comparison plots and summary for a test case.

    iter-49: gracefully handle the timeseries-only case (e.g. OMIP runs
    that don't emit ``snapshots_latlon.npz``).  When no grid has a
    snapshots payload, skip the snapshot-based plots and only emit the
    timeseries comparison.
    """
    if len(grid_results) < 2:
        return  # Need at least 2 grids for comparison

    print(f"  Creating cross-grid comparisons for {test_case_dir.name}...")

    # Time series comparison (works whether or not snapshots are present).
    _create_comparison_timeseries(test_case_dir, grid_results)

    # iter-49: only attempt snapshot-based comparisons if at least one
    # grid has a snapshots payload.  ``data['snapshots']`` is None for
    # timeseries-only runs (the relaxed collector emits None there).
    grids_with_snapshots = {
        name: data for name, data in grid_results.items()
        if data.get('snapshots') is not None
    }
    if not grids_with_snapshots:
        print("    [iter-49] timeseries-only run; skipping snapshot plots")
        return

    # iter-50 codex MEDIUM: a "cross-grid" snapshot plot needs at
    # least 2 grids with snapshots — otherwise we'd produce a
    # single-grid plot mislabelled as cross-grid.
    if len(grids_with_snapshots) < 2:
        print(
            "    [iter-50] only 1 grid has snapshots; skipping cross-"
            "grid snapshot plots (would be single-grid)"
        )
        return

    # Final snapshot comparisons (use only the grids that actually
    # have snapshots).  iter-50 codex MEDIUM: pass the filtered
    # ``grids_with_snapshots`` rather than the full ``grid_results``
    # so the downstream plotters don't have to do the same skip
    # twice (and a future plotter that forgets the guard cannot
    # silently emit a single-grid plot).
    for field in ['eta', 'SST', 'w_133m']:
        field_available = any(
            field in data['snapshots'].files or
            any(f.startswith(f'{field}_step') for f in data['snapshots'].files)
            for data in grids_with_snapshots.values()
        )
        if field_available:
            _create_comparison_snapshots(
                test_case_dir, grids_with_snapshots, field,
            )
            _create_comparison_evolution(
                test_case_dir, grids_with_snapshots, field,
            )

    # Forcing profile plot for wind-driven cases
    if 'barotropic_wind' in test_case_dir.name:
        _save_forcing_profile(test_case_dir)

    # Vertical cross-section comparison for baroclinic gyre
    if 'baroclinic_gyre' in test_case_dir.name:
        _create_comparison_vertical_section(test_case_dir, grid_results)
        _create_comparison_vertical_evolution(test_case_dir, grid_results)

    # Summary table
    _create_comparison_summary(test_case_dir, grid_results)


def _save_forcing_profile(test_case_dir: Path) -> None:
    """Save a wind forcing profile plot (tau_x vs latitude)."""
    import matplotlib.pyplot as plt

    lat_deg = np.linspace(-90, 90, 361)
    lat_rad = lat_deg * np.pi / 180.0
    s2 = np.sin(lat_rad) ** 2
    tau_x = (-0.08 - 0.0397 * s2 + 1.9487 * s2**2 - 2.0397 * s2**3) * np.cos(lat_rad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Prescribed Wind Forcing (Nikurashin & Vallis style)',
                 fontsize=14, fontweight='bold')

    # Panel 1: Wind stress profile
    axes[0].plot(lat_deg, tau_x, 'b-', linewidth=2)
    axes[0].axhline(0, color='k', linewidth=0.5, linestyle=':')
    axes[0].axvspan(-90, -55, alpha=0.1, color='gray', label='Drake Passage open')
    axes[0].set_xlabel('Latitude (deg)')
    axes[0].set_ylabel(r'$\tau_x$ [Pa]')
    axes[0].set_title('Zonal Wind Stress')
    axes[0].set_xlim(-80, 80)
    axes[0].set_ylim(-0.12, 0.12)
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('Trades', xy=(0, -0.065), ha='center', fontsize=9, color='blue')
    axes[0].annotate('Westerlies', xy=(50, 0.085), ha='center', fontsize=9, color='red')
    axes[0].annotate('Westerlies', xy=(-50, 0.085), ha='center', fontsize=9, color='red')

    # Panel 2: Wind stress curl (proportional to Sverdrup transport)
    R = 6.371e6
    dtau_dlat = np.gradient(tau_x, lat_rad)
    curl_z = dtau_dlat / R
    axes[1].plot(lat_deg, curl_z * 1e7, 'b-', linewidth=2)
    axes[1].axhline(0, color='k', linewidth=0.5, linestyle=':')
    axes[1].axvspan(-90, -55, alpha=0.1, color='gray')
    axes[1].set_xlabel('Latitude (deg)')
    axes[1].set_ylabel(r'curl$_z(\tau)$ ($\times 10^{-7}$ N/m$^3$)')
    axes[1].set_title('Wind Stress Curl')
    axes[1].set_xlim(-80, 80)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = test_case_dir / "forcing_profile.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")


def _check_and_generate_comparisons(output_base: Path, test_case_name: str, all_results: list) -> None:
    """Check if all grids completed for a test case and generate cross-grid comparisons."""
    # Find all results for this test case
    test_results = [r for r in all_results if r['test'] == test_case_name]
    
    if len(test_results) < 2:
        return  # Need at least 2 grids for comparison
    
    # Check if we have results for multiple grids
    grid_types = set(r['grid'] for r in test_results)
    if len(grid_types) < 2:
        return  # Need different grids, not just multiple resolutions
        
    # Rest-state variants are grouped under rest_state/
    if test_case_name in TestCase._REST_STATE_GROUP:
        test_case_dir = output_base / "rest_state" / test_case_name
    else:
        test_case_dir = output_base / test_case_name
    if not test_case_dir.exists():
        return
    
    # Try to collect grid results
    grid_results = _collect_grid_results(test_case_dir)
    if len(grid_results) > 1:
        print("\n" + "-" * 60)
        print(f"  CROSS-GRID COMPARISON: {test_case_name}")
        print("-" * 60)
        _create_cross_grid_comparisons(test_case_dir, grid_results)
        print("-" * 60)
    else:
        print(f"  Note: Insufficient grid data for {test_case_name} cross-grid comparison")


def _replot_case_snapshots(case_dir: Path) -> None:
    """Regenerate per-case snapshot plots from saved NPZ data.

    Reads ``snapshots_latlon.npz`` (regridded) and ``results.txt``
    to reconstruct the snapshot evolution plots, cross-sections, and
    vertical profiles without rerunning the simulation.
    """
    npz_path = case_dir / "snapshots_latlon.npz"
    results_path = case_dir / "results.txt"
    if not npz_path.exists():
        return

    data = np.load(npz_path)
    if "times_days" not in data.files:
        return

    times = data["times_days"]
    lat = data["lat"]
    lon = data["lon"]
    n_times = len(times)

    # Parse metadata from results.txt
    meta = {}
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    meta[k.strip()] = v.strip()
    dt_val = float(meta.get("dt", DEFAULT_DT))
    grid_type = meta.get("grid", "")
    resolution = meta.get("resolution", "")

    # Determine coord_kind for the regridded data — it's always on a
    # regular lat-lon grid after regridding, so we use "latlon".
    coord_kind = "latlon"

    # Build field specs from what's available
    field_specs_2d = []
    if "eta" in data.files:
        field_specs_2d.append(("eta", "SSH (m)", "RdBu_r"))
    if "SST" in data.files:
        field_specs_2d.append(("SST", "SST (degC)", "RdYlBu_r"))
    if "speed_sfc" in data.files:
        field_specs_2d.append(("speed_sfc", "Surface speed (m/s)", "magma"))
    if "w_133m" in data.files:
        field_specs_2d.append(("w_133m", "w at 134m (m/s)", "RdBu_r"))

    if not field_specs_2d:
        return

    # Reconstruct the snapshots dict expected by _save_snapshot_plots.
    # The NPZ stores (n_times, lat, lon) arrays.  The plotting code
    # expects  snapshots = {step: {field: 2D_array, ...}, ...}
    # with step numbers as keys.  We fake step numbers from dt.
    snapshots = {}
    steps_arr = data["steps"] if "steps" in data.files else np.arange(n_times)
    for i, step in enumerate(steps_arr):
        step = int(step)
        snap = {}
        for fk, _, _ in field_specs_2d:
            if fk in data.files:
                arr = data[fk]
                if arr.ndim == 3:  # (time, lat, lon)
                    snap[fk] = arr[i]
                elif arr.ndim == 2:
                    snap[fk] = arr
        if "land_mask" in data.files:
            lm = data["land_mask"]
            snap["land_mask"] = lm[i] if lm.ndim == 3 else lm
        # Velocity vectors for quiver overlay
        for vk in ("u_sfc", "v_sfc"):
            if vk in data.files:
                va = data[vk]
                snap[vk] = va[i] if va.ndim == 3 else va
        snapshots[step] = snap

    # Domain extent
    domain_extent = None
    if "source_lon_range" in data.files and "source_lat_range" in data.files:
        slon = data["source_lon_range"]
        slat = data["source_lat_range"]
        domain_extent = (float(slon[0]), float(slon[1]),
                         float(slat[0]), float(slat[1]))

    case_label = f"{case_dir.parent.name} {grid_type} {resolution}".strip()

    # Regenerate snapshot plots (regridded data is already on lat-lon)
    _save_snapshot_plots(
        case_dir, case_label, snapshots, dt_val, field_specs_2d,
        coord_kind, lon, lat, domain_extent=domain_extent)

    # Regenerate cross-sections and profiles if 3D data exists
    if "T_3d" in data.files:
        import ast as _ast
        T_3d_all = data["T_3d"]  # (time, lat, lon, lev)
        depth_data = None
        if results_path.exists():
            with open(results_path) as f:
                for line in f:
                    if line.startswith("depth:"):
                        depth_data = np.array(
                            _ast.literal_eval(line.split(":", 1)[1].strip()))
                        break
        if depth_data is not None:
            # Add T_3d to snapshots
            for i, step in enumerate(steps_arr):
                step = int(step)
                if step in snapshots and T_3d_all.ndim == 4:
                    snapshots[step]["T_3d"] = T_3d_all[i]

            _save_cross_sections(
                case_dir, case_label, snapshots, dt_val, "T_3d",
                coord_kind, lon, lat, depth_data, "Depth (m)")
            _save_profiles(
                case_dir, case_label, snapshots, dt_val, "T_3d",
                depth_data, "Depth (m)")


def _run_replot(args) -> None:
    """Replot mode: regenerate all plots from existing NPZ data."""
    output_base = Path(args.output)
    if not output_base.exists():
        print(f"Output directory {output_base} does not exist.")
        return

    print("=" * 78)
    print("  legoESM Ocean Test Matrix — REPLOT MODE")
    print("=" * 78)
    print(f"  Output:     {output_base}")
    print(f"  Filter:     --only {args.only}  --grid {args.grid}")
    print("=" * 78)

    t_start = time.time()

    # Discover all test case directories with results.
    # iter-53: ALSO discover timeseries-only test cases (OMIP-style)
    # by globbing for ``mean_timeseries.csv``.  Without this, iter-49's
    # collector relaxation has no entry point — replot would silently
    # skip OMIP runs because they don't emit ``snapshots_latlon.npz``.
    test_case_dirs = set()
    replotted = 0

    discovery_globs = ["snapshots_latlon.npz", "mean_timeseries.csv"]
    seen_res_dirs: set[Path] = set()

    for pattern in discovery_globs:
        for marker in sorted(output_base.rglob(pattern)):
            res_dir = marker.parent
            if res_dir in seen_res_dirs:
                continue
            seen_res_dirs.add(res_dir)
            grid_dir = res_dir.parent
            test_dir = grid_dir.parent

            grid_name = grid_dir.name
            test_name = test_dir.name

            # Apply filters
            if args.only != "all" and args.only not in test_name:
                continue
            if args.grid != "all" and args.grid != grid_name:
                continue

            # iter-53: only the snapshot-marker path replots
            # individual case snapshots.  Timeseries-only cases skip
            # ``_replot_case_snapshots`` (it requires npz data) but
            # still add to ``test_case_dirs`` so the cross-grid
            # comparison block below picks them up.
            #
            # iter-55 codex HIGH: the iter-53 version had the
            # ``try/except _replot_case_snapshots`` block at the
            # OUTER ``for pattern`` indent, so it ran once per
            # discovery_glob (using whatever ``res_dir`` was last
            # bound) — and worse, on the timeseries-only pass it
            # would call ``_replot_case_snapshots`` on a directory
            # without ``snapshots_latlon.npz``.  Re-indented so the
            # snapshot-replot step is INSIDE the marker loop AND
            # only runs on the snapshot-marker branch.
            if pattern == "snapshots_latlon.npz":
                print(
                    f"  Replotting {test_name}/{grid_name}/"
                    f"{res_dir.name} ..."
                )
                try:
                    _replot_case_snapshots(res_dir)
                    replotted += 1
                except Exception as e:
                    print(f"    ERROR: {e}")
                    traceback.print_exc()
                test_case_dirs.add(test_dir)
            else:
                print(
                    f"  Discovered timeseries-only "
                    f"{test_name}/{grid_name}/{res_dir.name}"
                )
                # Skip the per-case snapshot-replot step;
                # cross-grid plot below will use the iter-49
                # relaxed collector.
                test_case_dirs.add(test_dir)

    # Regenerate cross-grid comparisons
    if test_case_dirs:
        print()
        print("=" * 78)
        print("  REGENERATING CROSS-GRID COMPARISONS")
        print("=" * 78)
        for test_dir in sorted(test_case_dirs):
            grid_results = _collect_grid_results(test_dir)
            if len(grid_results) >= 2:
                _create_cross_grid_comparisons(test_dir, grid_results)
            elif len(grid_results) == 1:
                print(f"  Skipping {test_dir.name}: only 1 grid available")

    # Rest-state cross-variant comparison
    _create_rest_state_cross_variant_comparison(output_base)

    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  Replotted {replotted} case(s) in {elapsed:.1f}s")
    print("=" * 78)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Use float64 precision for the validation test matrix.
    #
    # The default PrecisionPolicy is float32, which is faster on GPUs
    # and suitable for production runs.  However, float32 introduces
    # rounding noise (~1e-7 relative per step) that accumulates in
    # conservation diagnostics and masks real discretisation errors.
    # For example, the stratified rest-state test shows a spurious
    # temperature drift of ~3e-5 degC/day in float32 that vanishes
    # entirely in float64 — the vertical diffusion operator is in
    # fact perfectly conservative.
    #
    # Running the test matrix in float64 ensures that any drift we
    # detect is a genuine bug in the numerics, not arithmetic noise.
    # ---------------------------------------------------------------
    from legoesm.core.precision import set_policy, PrecisionPolicy
    set_policy(PrecisionPolicy.fp64())

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

    if args.replot:
        _run_replot(args)
        return

    if not tests:
        print("No tests match the given filters.")
        return

    if args.resolution:
        # iter-102 fix: ``--resolution N`` (integer) was previously
        # applied verbatim to every grid type, breaking 3+ of 4
        # parsers (mirrors the iter-95 atmosphere matrix fix).
        # Specifically:
        #   cubed_sphere ``int(res[1:])``: "24" → 4 (silent wrong)
        #   latlon ``res.split("x")``: "24" → unpack error
        #   mpas ``res.replace("ico","")``: "24" → level=24
        #     (4.29e+10 cells, ValueError)
        #   spectral ``int(res[1:])``: "24" → 4 (silent wrong)
        #
        # iter-102: bare integer N expands per-grid; pre-formatted
        # strings (``"C24"``, ``"ico3"``, ``"36x72"``, ``"T21"``)
        # pass through unchanged.
        cli_res = args.resolution
        try:
            N = int(cli_res)
            is_bare_int = True
        except ValueError:
            is_bare_int = False

        # iter-107 (codex iter-104 LOW-7): reject N <= 0 BEFORE
        # ``_expand_cli_res`` is invoked.  See atmosphere matrix
        # runner for the rationale.
        if is_bare_int and N <= 0:
            import sys as _sys
            print(
                f"error: --resolution must be a positive integer "
                f"(or a per-grid format string like 'C24', 'ico3', "
                f"'36x72', 'T21', '50km').  Got N={N}.",
                file=_sys.stderr,
            )
            _sys.exit(2)

        def _expand_cli_res(grid_type: str) -> str:
            if not is_bare_int:
                return cli_res
            if grid_type == "cubed_sphere":
                return f"C{N}"
            elif grid_type == "latlon":
                return f"{N}x{2 * N}"
            elif grid_type == "mpas":
                # mpas (icosahedral): level so 10·4^level + 2 ≈ 2N²
                # (latlon-coverage).  iter-95 dispatch table.
                # N=16 → ico3 (642 cells), N=32 → ico4 (2562),
                # N=72 → ico5 (10242).
                import math
                raw_level = round(
                    math.log(2 * N * N / 10) / math.log(4))
                level = max(2, min(8, raw_level))
                # iter-107: warn when clipping to ico8 (max
                # supported level per ``voronoi.py:1063``).
                if raw_level > 8:
                    print(
                        f"warning: --resolution {N} maps to "
                        f"icosahedral level {raw_level} which "
                        f"exceeds the maximum supported level "
                        f"(8 = 655,362 cells); clipping to "
                        f"ico8.  Use load_mpas_mesh() with a "
                        f"pre-built mesh file for higher "
                        f"resolutions.",
                    )
                return f"ico{level}"
            elif grid_type == "spectral":
                return f"T{N}"
            elif grid_type in ("mpas_regional", "latlon_regional", "cs_regional"):
                # Regional grids use km resolution; pass through
                # the bare integer as ``Nkm`` form.
                return f"{N}km"
            else:
                return cli_res

        tests = [TestCase(
            t.case, t.grid_type, _expand_cli_res(t.grid_type),
            t.duration_days, t.quick_days, t.run_kwargs)
            for t in tests]

    output_base = Path(args.output)

    print("=" * 78)
    print("  legoESM Ocean Test Matrix")
    print("=" * 78)
    from legoesm.core.precision import get_policy
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  Precision:  {get_policy().storage.__name__}")
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

    # Keep track of completed test cases for cross-grid comparison
    completed_test_cases = set()
    
    for i, tc in enumerate(tests, 1):
        if args.days is not None:
            days = args.days
        else:
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
        
        # Check if this test case just completed across all its grids
        if tc.case not in completed_test_cases:
            # Find how many grids are supposed to run for this test case
            test_case_tests = [t for t in tests if t.case == tc.case]
            test_case_results = [r for r in ALL_RESULTS if r['test'] == tc.case]
            
            # If we have results for all grids of this test case, generate comparisons
            if len(test_case_results) >= len(test_case_tests):
                _check_and_generate_comparisons(output_base, tc.case, ALL_RESULTS)
                completed_test_cases.add(tc.case)

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

    # Generate cross-grid comparison plots for any test cases that weren't completed during the run
    # (This handles cases where the run was filtered or interrupted)
    remaining_test_cases = set()
    test_cases = {}
    for result in ALL_RESULTS:
        test_name = result['test']
        if test_name not in test_cases:
            test_cases[test_name] = []
        test_cases[test_name].append(result)
        if test_name not in completed_test_cases:
            remaining_test_cases.add(test_name)
    
    if remaining_test_cases:
        print("\n" + "=" * 78)
        print("  GENERATING REMAINING CROSS-GRID COMPARISONS")
        print("=" * 78)
        
        # Create comparisons for test cases that weren't processed during the main loop
        for test_name in remaining_test_cases:
            results = test_cases[test_name]
            if len(results) > 1:  # Only create comparisons if multiple grids were run
                if test_name in TestCase._REST_STATE_GROUP:
                    test_case_dir = output_base / "rest_state" / test_name
                else:
                    test_case_dir = output_base / test_name
                if test_case_dir.exists():
                    grid_results = _collect_grid_results(test_case_dir)
                    if len(grid_results) > 1:
                        print(f"\n  Creating cross-grid comparisons for {test_name}...")
                        _create_cross_grid_comparisons(test_case_dir, grid_results)
                    else:
                        print(f"  Skipping {test_name}: insufficient grid data")
                else:
                    print(f"  Skipping {test_name}: directory not found")
            else:
                print(f"  Skipping {test_name}: only {len(results)} grid(s) run")
    else:
        print("\n" + "=" * 78)
        print("  ALL CROSS-GRID COMPARISONS COMPLETED DURING RUN")
        print("=" * 78)

    # Generate rest-state cross-variant comparison
    _create_rest_state_cross_variant_comparison(output_base)

    if n_fail > 0 or n_error > 0:
        sys.exit(1)


# ===========================================================================
# Rest-state cross-variant comparison
# ===========================================================================

REST_STATE_VARIANTS_LIST = [
    "rest_state_stratified_with_land",
    "rest_state_uniform_with_land",
    "rest_state_stratified_no_land",
    "rest_state_uniform_no_land",
]

REST_STATE_VARIANT_LABELS = {
    "rest_state_stratified_with_land": "Stratified + Land",
    "rest_state_uniform_with_land":    "Uniform T/S + Land",
    "rest_state_stratified_no_land":   "Stratified, No Land",
    "rest_state_uniform_no_land":      "Uniform T/S, No Land",
}

REST_STATE_VARIANT_COLORS = {
    "rest_state_stratified_with_land": "red",
    "rest_state_uniform_with_land":    "orange",
    "rest_state_stratified_no_land":   "blue",
    "rest_state_uniform_no_land":      "green",
}


def _create_rest_state_cross_variant_comparison(output_base: Path):
    """Create a combined comparison plot across all rest-state variants and grids.

    Produces a (3 rows × 3 cols) figure:
    - Rows: grids (cubed_sphere, latlon, mpas)
    - Cols: eta drift, T drift, S drift
    Each panel overlays the 4 rest-state variants as different colored lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = ["cubed_sphere", "latlon", "mpas"]

    # Collect conservation timeseries for all variants × grids
    data = {}  # (variant, grid) -> DataFrame
    rest_base = output_base / "rest_state"
    for variant in REST_STATE_VARIANTS_LIST:
        for grid in grids:
            # Find the resolution dir (rest states grouped under rest_state/)
            variant_dir = rest_base / variant / grid
            if not variant_dir.exists():
                continue
            res_dirs = [d for d in variant_dir.iterdir() if d.is_dir()]
            if not res_dirs:
                continue
            csv_file = res_dirs[0] / "conservation_timeseries.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    data[(variant, grid)] = df
                except Exception:
                    continue

    if not data:
        print("  No rest-state conservation data found for cross-variant comparison.")
        return

    print("\n" + "=" * 78)
    print("  REST-STATE CROSS-VARIANT COMPARISON")
    print("=" * 78)

    fig, axes = plt.subplots(len(grids), 3, figsize=(15, 3.5 * len(grids)),
                              sharex=True)
    if len(grids) == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Volume (eta) relative drift", "Heat (T) relative drift",
                  "Salt (S) relative drift"]
    col_keys = ["vol_rel", "heat_rel", "salt_rel"]

    for i_grid, grid in enumerate(grids):
        for i_col, (col_key, col_label) in enumerate(zip(col_keys, col_labels)):
            ax = axes[i_grid, i_col]
            any_plotted = False
            for variant in REST_STATE_VARIANTS_LIST:
                key = (variant, grid)
                if key not in data:
                    continue
                df = data[key]
                if col_key not in df.columns:
                    continue
                label = REST_STATE_VARIANT_LABELS.get(variant, variant)
                color = REST_STATE_VARIANT_COLORS.get(variant, "gray")
                ax.plot(df["time_days"], df[col_key], label=label, color=color,
                        linewidth=1.5)
                any_plotted = True

            ax.set_ylabel(col_label if i_grid == 0 else "")
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
            ax.grid(True, alpha=0.3)
            if i_grid == 0:
                ax.set_title(col_label, fontsize=10)
            if i_grid == len(grids) - 1:
                ax.set_xlabel("Time (days)")
            if i_col == 0:
                ax.text(-0.25, 0.5, grid, transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center',
                        rotation=90)
            if any_plotted and i_grid == 0 and i_col == 2:
                ax.legend(fontsize=7, loc='upper left')

    fig.suptitle("Rest-State Cross-Variant Comparison\n"
                 "(4 variants × 3 grids, conservation drift)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])

    rest_base.mkdir(parents=True, exist_ok=True)
    out_path = rest_base / "cross_variant_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also create a summary table
    summary_path = rest_base / "cross_variant_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Rest-State Cross-Variant Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Variant':<35} {'Grid':<15} {'eta drift':>12} {'T drift':>12} {'S drift':>12}\n")
        f.write("-" * 86 + "\n")
        for variant in REST_STATE_VARIANTS_LIST:
            for grid in grids:
                key = (variant, grid)
                if key not in data:
                    continue
                df = data[key]
                last = df.iloc[-1]
                eta_d = last.get("vol_rel", 0)
                t_d = last.get("heat_rel", 0)
                s_d = last.get("salt_rel", 0)
                label = REST_STATE_VARIANT_LABELS.get(variant, variant)
                f.write(f"{label:<35} {grid:<15} {eta_d:>12.2e} {t_d:>12.2e} {s_d:>12.2e}\n")
            f.write("\n")
    print(f"  Saved: {summary_path}")


if __name__ == "__main__":
    main()
