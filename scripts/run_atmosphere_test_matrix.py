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
import os
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

from legoesm.runtime.backend import ensure_metal_or_fallback
ensure_metal_or_fallback()

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
    family: str = ""        # Hughes-tutorial family (filled by lookup)

    @property
    def output_path(self) -> str:
        parts = [self.equation_set, self.case, self.grid_type, self.resolution]
        if self.vertical_coord != "none":
            parts.append(self.vertical_coord)
        return "/".join(parts)


# ---------------------------------------------------------------------------
# Hughes (2026) catalog: which families each case belongs to
# ---------------------------------------------------------------------------
#
# Each case is tagged with a *set* of family memberships so a single case
# can simultaneously belong to an equation-set family ("hydro", "tracer",
# …) and to a paper-vintage family ("dcmip2008", "dcmip2012", …) and to
# the Hughes-tutorial meta-tag ("hughes").  ``--family X`` selects the
# cases whose membership set contains X.
#
# Tagging conventions:
#   - equation-set family: sw / hydro / nh / climate / tracer / moist
#       (mutually exclusive within a single case)
#   - paper vintage:       dcmip2008 / dcmip2012 / dcmip2016
#       (a case may carry zero or one vintage tag)
#   - hughes (meta):       any case that appears in Hughes (2026) §2-§7
#       — including the Williamson "extras" (W1/W2/W5/W6) implied as
#       prerequisites and the DCMIP 2025 NH cases (which are partial
#       coverage of the canonical DCMIP 2012 set, see catalog).
#
# See ``docs/dycore_validation_catalog.md`` for the canonical inventory.
_CASE_FAMILIES: dict[str, frozenset[str]] = {
    # Williamson SW (W1=cosine_bell, W2, W5 wired pre-M1; W6 added M1.a)
    "williamson2":        frozenset({"sw", "hughes"}),
    "williamson5":        frozenset({"sw", "hughes"}),
    "williamson6":        frozenset({"sw", "hughes"}),
    "cosine_bell":        frozenset({"sw", "hughes"}),
    # Hydrostatic dry
    "baroclinic":         frozenset({"hydro", "hughes"}),  # canonical J-W
    "rotated_baroclinic": frozenset({"hydro", "dcmip2008", "hughes"}),
    "rotated_steady":     frozenset({"hydro", "dcmip2008", "hughes"}),
    "gravity_wave_3_1":   frozenset({"hydro", "dcmip2008", "hughes"}),
    "inertio_gravity_3_2": frozenset({"hydro", "dcmip2008", "hughes"}),
    "mountain_rossby_5_0": frozenset({"hydro", "dcmip2008", "hughes"}),
    "rossby_haurwitz_6_0": frozenset({"hydro", "dcmip2008", "hughes"}),
    "rest_state_topo":    frozenset({"hydro", "dcmip2012", "hughes"}),
    # Tracer transport
    "dcmip_transport_11": frozenset({"tracer", "dcmip2012", "hughes"}),
    "dcmip_transport_12": frozenset({"tracer", "dcmip2012", "hughes"}),
    "dcmip_transport_13": frozenset({"tracer", "dcmip2012", "hughes"}),
    # Climate-timescale
    "held_suarez":        frozenset({"climate", "hughes"}),
    "held_suarez_topo":   frozenset({"climate", "hughes"}),
    "held_suarez_small_planet": frozenset({"climate", "hughes"}),
    # Production AMIP — NOT a canonical Hughes idealized test
    "amip":               frozenset({"climate"}),
    # Non-hydrostatic (DCMIP 2025; partial coverage of Hughes §4 / §6)
    "dcmip_tc1":          frozenset({"nh", "hughes"}),
    "dcmip_tc2":          frozenset({"nh", "moist", "hughes"}),
    "dcmip_tc3":          frozenset({"nh", "moist", "hughes"}),
}

# Primary equation-set tag, used only for the ``--list`` display column.
# Falls back to "" for unrecognised cases.
_EQUATION_SET_TAGS = frozenset(
    {"sw", "hydro", "nh", "tracer", "climate", "moist"})


def _primary_family(case: str) -> str:
    families = _CASE_FAMILIES.get(case, frozenset())
    primary = families & _EQUATION_SET_TAGS
    return next(iter(sorted(primary)), "")


# Family tags that are reserved for forthcoming milestones — they must
# remain valid CLI choices (``--family dcmip2016`` returns the empty set
# today but is documented in the catalog) so users do not get an
# argparse error before the M3 cases land.
_RESERVED_FAMILIES: frozenset[str] = frozenset({"dcmip2016"})

# Allowed values of the ``--family`` CLI flag.  Constructed from the
# union of every tag mentioned in ``_CASE_FAMILIES`` plus the reserved
# set, so adding a new case automatically extends the CLI surface and
# pre-announced families stay accepted.
_FAMILY_CHOICES: list[str] = sorted(
    {tag for tags in _CASE_FAMILIES.values() for tag in tags}
    | _RESERVED_FAMILIES
) + ["all"]


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
            ("cosine_bell", 12, 1, {}),
        ]:
            matrix.append(TestCase(
                "shallow_water", case, g, res[g], "none", dur, quick, dict(kw)))
        # Williamson 6 (Rossby-Haurwitz wave-4) — Hughes-tutorial extended
        # SW set.  Wired for icosahedral (MPAS) and spectral grids in
        # M1.a.  Cubed-sphere (FV3 D-grid) and lat-lon C-grid require
        # analytic edge-/face-midpoint wind init at non-trivial lon/lat
        # offsets (W6 winds depend on both lon and lat, unlike W2/W5),
        # so they are deferred to M1.b.  See
        # ``docs/dycore_validation_catalog.md``.
        if g in ("icosahedral", "spectral"):
            matrix.append(TestCase(
                "shallow_water", "williamson6", g, res[g], "none", 14, 1,
                {"test_num": 6}))

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

        # --- Hughes-tutorial extensions (M1.a) ---
        # Rotated Jablonowski-Williamson — DCMIP 2008 §4-1 (steady) /
        # §4-2 (baroclinic). Hybrid coord only by default since the
        # canonical DCMIP setup uses pressure-based vertical levels.
        matrix.append(TestCase(
            "hydrostatic", "rotated_baroclinic", g, res[g], "hybrid", 10, 2,
            {"alpha": 0.7853981633974483, "perturbed": True}))
        matrix.append(TestCase(
            "hydrostatic", "rotated_steady", g, res[g], "hybrid", 30, 2,
            {"alpha": 0.7853981633974483, "perturbed": False}))
        # DCMIP 2012 §2-0-0 — atmosphere at rest with steep topography.
        # Hybrid coord only (sigma cannot represent the ridged mountain
        # consistently for hydrostatic-balance tests).
        matrix.append(TestCase(
            "hydrostatic", "rest_state_topo", g, res[g], "hybrid", 7, 1,
            {"h_0": 2000.0}))
        # Held-Suarez over idealized Gaussian/cosine-bell mountain.
        # Quick-mode duration is short (2 days) so the smoke test
        # finishes in a few minutes per grid; full duration (200 days)
        # is needed only for the actual HS climatology.
        matrix.append(TestCase(
            "hydrostatic", "held_suarez_topo", g, res[g], "hybrid", 200, 2,
            {"h_0": 2000.0}))
        # DCMIP 2008 §3-1 / §3-2 / §5-0 / §6-0 — dry-3D Hughes-tutorial
        # tests on hydrostatic dycores.  Short integrations (1-3 days) so
        # the wave packets / Rossby trains have time to develop without
        # consuming AMIP-scale wall-clock.
        matrix.append(TestCase(
            "hydrostatic", "gravity_wave_3_1", g, res[g], "hybrid", 1.0, 0.25, {}))
        matrix.append(TestCase(
            "hydrostatic", "inertio_gravity_3_2", g, res[g], "hybrid",
            3.0, 0.5, {}))
        matrix.append(TestCase(
            "hydrostatic", "mountain_rossby_5_0", g, res[g], "hybrid",
            10.0, 2.0, {}))
        matrix.append(TestCase(
            "hydrostatic", "rossby_haurwitz_6_0", g, res[g], "hybrid",
            14.0, 2.0, {}))
        # Wedi-Smolarkiewicz 2009 small-planet Held-Suarez (X=125).
        # The IC + grid factories are in place at
        # ``src/legoesm/atmosphere/idealized/small_planet.py``; matrix
        # runner wiring is deferred to M1.b because the existing
        # ``run_held_suarez`` builds grids with default Earth radius.

    # --- Non-hydrostatic: cubed-sphere, icosahedral, and spectral only ---
    # (lat-lon NH dycore does not exist)
    nh_grids = ["cubed_sphere", "icosahedral", "spectral"]
    for g in nh_grids:
        for case, dur, quick, kw in [
            ("dcmip_tc1", 3 / 24, 0.5 / 24, {"test_case": "tc1"}),
            ("dcmip_tc2", 6 / 24, 5 / (24 * 60), {"test_case": "tc2a"}),
            # TC3 quick is 4 min — moist squall-line test limited by
            # Kessler microphysics coupling efficiency in the NH solver.
            ("dcmip_tc3", 2 / 24, 4 / (24 * 60), {"test_case": "tc3"}),
        ]:
            matrix.append(TestCase(
                "nonhydrostatic", case, g, res[g], "height", dur, quick,
                dict(kw)))

    # Fill in the primary equation-set family tag for the --list column.
    # Multi-tag membership (including paper vintage) is consulted
    # directly from ``_CASE_FAMILIES`` inside ``filter_tests``.
    for tc in matrix:
        tc.family = _primary_family(tc.case)

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


def has_collectable_atmosphere_outputs(d: Path) -> bool:
    """Return True iff ``d`` looks like a collectable per-grid output.

    iter-43 codex MEDIUM: extracted to module scope so the iter-42
    AMIP format converter (and its tests) can rely on a SINGLE source
    of truth.  Previously the predicate lived inside
    ``_create_atmosphere_comparison`` so anyone wanting to verify it
    had to either re-implement it or assert against the source string.

    Criterion (iter-26/iter-27 codex review):

      * a ``snapshots_latlon.npz`` file (hydro-style snapshot output), OR
      * BOTH ``mean_timeseries.csv`` AND ``results.txt`` (timeseries-
        only run, e.g. RCE / OMIP / AMIP-via-iter-42-converter).

    A bare ``mean_timeseries.csv`` with no ``results.txt`` is
    rejected so a stale CSV without provenance doesn't produce a
    phantom cross-grid combo.
    """
    if (d / "snapshots_latlon.npz").exists():
        return True
    return (d / "mean_timeseries.csv").exists() and (d / "results.txt").exists()


def _compute_drift(values: list[float]) -> float:
    """Relative drift of a scalar diagnostic series.

    Thin wrapper that delegates to the shared helper
    ``legoesm.diagnostics.conservation_drift.compute_relative_drift``.
    The wrapper exists so that the dozens of internal callsites in
    this script can stay terse (``_compute_drift(diag["mass"])``) and
    so that the iter-83 / iter-86 unit tests retain a stable target
    in this module — no external API contract is being preserved.

    iter-90 codex LOW-8: the previous docstring said "backward
    compatibility" which was misleading; the rationale is purely
    "internal callsite ergonomics" (≥4 callsites in this module).

    Returns ``|values[-1] - values[0]| / max(|values[0]|, 1.0)``.
    """
    from legoesm.diagnostics.conservation_drift import compute_relative_drift
    return compute_relative_drift(values)


def _apply_mass_drift_tolerance(
    ok: bool, notes: str, mass_drift: float, tol: float,
    *, n_samples: int | None = None,
) -> tuple[bool, str]:
    """Thin wrapper that delegates to the centralized
    ``legoesm.diagnostics.conservation_drift.apply_drift_tolerance``
    helper specialized for ``label="mass"``
    (iter-128 codex iter-127-followup LOW-1: previously the
    atmosphere runner had its own near-identical copy of the
    same gate logic; iter-128 collapses to a single source of
    truth in the diagnostics package).

    Behaviour and signature are unchanged from iter-117–iter-120:
    NaN-aware, n_samples-aware, idempotent on already-failed runs.
    """
    from legoesm.diagnostics.conservation_drift import (
        apply_drift_tolerance,
    )
    return apply_drift_tolerance(
        ok, notes, mass_drift, tol,
        label="mass", n_samples=n_samples,
    )


def _grid_cell_area(grid_or_mesh):
    """Return the cell-area array for any supported grid object.

    Cubed-sphere and lat-lon grids expose ``.area``; Voronoi /
    icosahedral meshes expose ``.areaCell``; Gaussian grids expose
    ``.grid_area``.  Pick whichever attribute exists in the order
    most-specific → most-generic to avoid surprises.
    """
    for attr in ("areaCell", "grid_area", "area"):
        a = getattr(grid_or_mesh, attr, None)
        if a is not None:
            return a
    raise AttributeError(
        f"_grid_cell_area: object {type(grid_or_mesh).__name__} exposes "
        f"none of areaCell/grid_area/area"
    )


def _area_weighted_mean(field, area) -> float:
    """Area-weighted scalar mean.

    Returns ``sum(field * area) / sum(area)`` as a Python float when
    ``field`` and ``area`` have the same shape.  When ``field`` has
    *more* dimensions than ``area`` (typical case: 3-D atmospheric
    field ``(nlat, nlon, nlev)`` with 2-D horizontal area
    ``(nlat, nlon)``, or cubed-sphere ``(6, n, n, nlev)`` with area
    ``(6, n, n)``, or Voronoi ``(n_cells, nlev)`` with area
    ``(n_cells,)``), the extra trailing axes are collapsed via a
    uniform mean *first* and the horizontal weighting is then
    applied.  This keeps the cross-grid mean physically meaningful
    (vertical-uniform-mean of a horizontally area-weighted average)
    while being grid-agnostic.

    Use this instead of bare ``jnp.mean`` whenever a field lives on
    cells with non-uniform area.  Bare ``jnp.mean`` over-weights
    high-latitude cells on lat-lon and Gaussian grids and disagrees
    with cube/icosahedral domain-mean h by ~15 % for Williamson 2 (the
    inconsistency exposed by the iter-1 cross-grid time-series plot
    and resolved in iter-2 for shallow-water; iter-3 extends the fix
    to the hydrostatic / AMIP scalar diagnostics, which have the same
    bug pattern but on 3-D fields).
    """
    f = jnp.asarray(field, dtype=jnp.float64)
    a = jnp.asarray(area, dtype=jnp.float64)
    if f.ndim == a.ndim:
        return float(jnp.sum(f * a) / jnp.sum(a))
    if f.ndim < a.ndim:
        raise ValueError(
            f"_area_weighted_mean: field.ndim={f.ndim} < area.ndim={a.ndim}; "
            f"area must have <= field dims (got shapes {f.shape} vs {a.shape})"
        )
    # Collapse extra trailing axes via uniform mean, then horizontal
    # area-weighted average over the leading axes.
    extra_axes = tuple(range(a.ndim, f.ndim))
    f_collapsed = jnp.mean(f, axis=extra_axes)
    return float(jnp.sum(f_collapsed * a) / jnp.sum(a))


# ---------------------------------------------------------------------------
# Hyperdiffusion helpers
# ---------------------------------------------------------------------------

def _hyperdiff_cube(n: int, ref_n: int = 48, ref_coeff: float = 1e16) -> float:
    return ref_coeff * (ref_n / n) ** 4


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    """Scale second-order divergence damping for cubed-sphere (FV3-style)."""
    return ref_coeff * (ref_n / n) ** 2


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

    .. warning::
       This heuristic gives ``A_h ∝ 1/n``, which is INSUFFICIENT at
       C72+ resolutions for the FV3 3D HS hybrid path
       (`primitive_eq_cdgrid.py`).  iter 33 found C72 NaN at default
       ``A_h`` but stable at 10x.  iter 32 traced the unstable mode to
       the interior (synoptic-scale, not grid-scale), which del-2
       Laplacian viscosity damps better than del-4 hyperdiff or
       cube-vertex damping.

       For C72+ users: set ``LEGOESM_AH_SCALE=10.0`` (iter 34)
       when invoking the matrix, OR construct the
       ``CDGridPrimitiveEquationConfig`` with ``A_h`` ~ 2.0e+07
       directly.  See ``FV3_3D.md`` iter 33-35 for the full diagnosis.
    """
    import math
    from legoesm import constants
    dx = math.pi * constants.R_earth / (2.0 * n)
    c_gw = math.sqrt(constants.R_d * 300.0)
    return frac * c_gw * dx


def _laplacian_visc_cube_v2(n: int) -> float:
    """Empirically calibrated Laplacian viscosity for cubed-sphere
    HS hybrid path (iter 33-37).

    Returns the LEGOESM_AH_SCALE-equivalent ``A_h`` directly:

    | n   | recommended A_h | LEGOESM_AH_SCALE multiplier vs v1 |
    |----:|----------------:|----------------------------------:|
    |  36 |     4.08e+06    |                              1.0  |
    |  48 |     6.12e+06    |                              2.0  |
    |  72 |     2.04e+07    |                             10.0  |

    For ``n`` not in the calibration set, this function uses a
    log-linear interpolation in ``log(A_h) ~ log(n)``.  The slope
    between C36 → C72 (factor 5x in A_h for factor 2x in n) is
    captured by ``A_h ∝ n^2.32``; intermediate values fit a
    quadratic-in-log fit to the 3 calibration points.

    .. warning::
       UNTESTED at C96+ resolutions.  This function extrapolates
       under the iter-37-observed pattern but will need empirical
       confirmation.  Run a stability check before climate-relevant
       integration at any new resolution.

    See ``FV3_3D.md`` iter 33-37 for the calibration history.

    This is an OPT-IN function — ``_laplacian_visc_cube`` (v1)
    remains the matrix default to avoid regressing the C36 / C48
    iter-17 / iter-19 / iter-24 calibrations which are tuned for
    the v1 ``A_h``.

    To opt in, replace the matrix's ``ah = _laplacian_visc_cube(n)``
    line with ``ah = _laplacian_visc_cube_v2(n)``, OR set
    ``LEGOESM_AH_SCALE`` to match the v2 / v1 ratio at each
    resolution.
    """
    import math
    # Calibration points from iter 33-37.
    calib = {36: 4.08e6, 48: 6.12e6, 72: 2.04e7}
    if n in calib:
        return calib[n]
    # Log-linear interpolation/extrapolation.  Fit:
    # log10(A_h) = a * log10(n) + b
    # Through C36 and C72: slope = (log10(2.04e7) - log10(4.08e6))
    #                            / (log10(72) - log10(36))
    #                    = log10(5) / log10(2) ≈ 2.322
    # i.e. A_h ∝ n^2.322
    log_n_36 = math.log10(36.0)
    log_a36 = math.log10(4.08e6)
    log_n_72 = math.log10(72.0)
    log_a72 = math.log10(2.04e7)
    slope = (log_a72 - log_a36) / (log_n_72 - log_n_36)
    log_a = log_a36 + slope * (math.log10(float(n)) - log_n_36)
    return float(10.0 ** log_a)


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

# Iter-31: per-run GHG overrides set from CLI flags by ``main()``
# before any runner invokes ``_make_rrtmgp_physics``.  The runner
# functions don't take additional kwargs (they all share the
# ``runner(tc, out_dir, days, radiation=...)`` signature), so this
# module-level pattern threads the values through without changing
# the runner interface.  Iter-34 added ``cloud_scheme`` to the
# same dict for the same reason.
_RUNTIME_RRTMGP_OVERRIDES: dict[str, float | str | None] = {
    "co2_ppmv": None,
    "ch4_ppbv": None,
    "n2o_ppbv": None,
    "cloud_scheme": None,   # iter-34
    # iter-39: optional analytical-ozone-profile knobs.  These map
    # directly onto ``OzoneProfileConfig`` fields and are forwarded
    # via ``RadiationConfig.ozone`` only when at least one is set.
    # iter-39 codex MEDIUM: keys are lower-case + lower-case unit
    # suffix to match the iter-31 GHG keys (``co2_ppmv``,
    # ``ch4_ppbv``, ``n2o_ppbv``).  The ``OzoneProfileConfig``
    # internal field is still ``p_peak_hPa`` (mixed-case unit
    # suffix); the override dict layer stays lower-case so
    # ``results.txt`` columns are consistent.
    "ozone_source": None,    # "standard" | "analytical" | "none"
    "ozone_peak_hpa": None,  # float (analytical-source only)
    "ozone_max_vmr": None,   # float (analytical-source only); 0 < vmr <= 1
}


def _make_rrtmgp_physics(model_type: str, dt: float, hs_fn=None,
                          co2_ppmv: float | None = None,
                          ch4_ppbv: float | None = None,
                          n2o_ppbv: float | None = None):
    """Create RRTMGP-based physics function, optionally combined with Held-Suarez.

    When *hs_fn* is provided the returned function sums the Held-Suarez
    Newtonian relaxation / Rayleigh drag tendencies with the RRTMGP
    radiative tendencies so the experiment remains a valid HS benchmark.

    Parameters
    ----------
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe", "mpas".
    dt : float
        Physics time step [s].
    hs_fn : callable, optional
        Held-Suarez forcing function ``(state, grid, sigma_coord) -> tendencies``.
        When ``None``, only RRTMGP radiation is applied (no HS forcing).
    co2_ppmv, ch4_ppbv, n2o_ppbv : float, optional
        Iter-31: per-experiment GHG concentration overrides for the
        AMIP "realistic forcing" prompt item.  When ``None``, the
        ``RRTMGPConfig`` defaults are used (415 ppm CO2, 1900 ppb
        CH4, 332 ppb N2O — present-day values).  Pass e.g.
        ``co2_ppmv=280`` for a pre-industrial AMIP run.  Time-varying
        CMIP6 input4MIPs forcing requires a deeper integration with
        ``forcing/external.py:get_ghg_at_time`` that's tracked as a
        post-Ralph follow-up.
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import (
        OzoneProfileConfig, RadiationConfig, RRTMGPConfig,
    )
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

    # Iter-31: build RRTMGPConfig with optional GHG overrides.
    # Argument values take precedence; otherwise fall back to the
    # module-level runtime overrides set from CLI flags.
    eff_co2 = co2_ppmv if co2_ppmv is not None else _RUNTIME_RRTMGP_OVERRIDES.get("co2_ppmv")
    eff_ch4 = ch4_ppbv if ch4_ppbv is not None else _RUNTIME_RRTMGP_OVERRIDES.get("ch4_ppbv")
    eff_n2o = n2o_ppbv if n2o_ppbv is not None else _RUNTIME_RRTMGP_OVERRIDES.get("n2o_ppbv")
    eff_cloud = _RUNTIME_RRTMGP_OVERRIDES.get("cloud_scheme")
    rrtmgp_kwargs = {}
    if eff_co2 is not None:
        rrtmgp_kwargs["co2_ppmv"] = eff_co2
    if eff_ch4 is not None:
        rrtmgp_kwargs["ch4_ppbv"] = eff_ch4
    if eff_n2o is not None:
        rrtmgp_kwargs["n2o_ppbv"] = eff_n2o
    # Iter-36 codex HIGH: ``RRTMGPConfig.include_clouds`` defaults to
    # False, which gates the RRTMGP cloud-optics path off even when
    # ``RadiationConfig.cloud_scheme != "none"``.  When the user
    # asks for a non-trivial cloud_scheme via ``--cloud-scheme``,
    # also flip ``include_clouds`` so RRTMGP actually consumes the
    # computed cloud properties.
    if eff_cloud is not None and eff_cloud != "none":
        rrtmgp_kwargs["include_clouds"] = True
    rrtmgp_cfg = RRTMGPConfig(**rrtmgp_kwargs) if rrtmgp_kwargs else RRTMGPConfig()

    # Iter-34: route ``--cloud-scheme`` through RadiationConfig
    # for AMIP-with-clouds runs.  iter-36: ``include_clouds`` flag
    # above ensures the scheme actually fires.
    rad_kwargs = {"scheme": "rrtmgp", "rrtmgp": rrtmgp_cfg}
    if eff_cloud is not None:
        rad_kwargs["cloud_scheme"] = eff_cloud

    # Iter-39: optional ozone-profile overrides.  Only construct an
    # ``OzoneProfileConfig`` if at least one of the three knobs is
    # set; otherwise leave the radiation default.
    #
    # iter-39 codex MEDIUM: ``--ozone-peak-hpa`` / ``--ozone-max-vmr``
    # only affect the analytical Gaussian path
    # (``_compute_ozone_vmr`` source=="analytical"); they're silently
    # ignored when source is ``standard`` (built-in profile) or
    # ``none`` (zero ozone).  When the user sets one of these
    # without explicitly setting source, auto-promote source to
    # ``"analytical"`` — matches the iter-36 ``include_clouds``
    # auto-flip pattern.  CLI-side parser_error guards against the
    # explicit-contradiction case (``--ozone-source standard
    # --ozone-peak-hpa 50``).
    eff_o3_src = _RUNTIME_RRTMGP_OVERRIDES.get("ozone_source")
    eff_o3_peak = _RUNTIME_RRTMGP_OVERRIDES.get("ozone_peak_hpa")
    eff_o3_vmr = _RUNTIME_RRTMGP_OVERRIDES.get("ozone_max_vmr")
    if eff_o3_src is not None or eff_o3_peak is not None or eff_o3_vmr is not None:
        o3_kwargs = {}
        if eff_o3_src is not None:
            o3_kwargs["source"] = eff_o3_src
        elif eff_o3_peak is not None or eff_o3_vmr is not None:
            # Implicit promotion so the analytical-only knobs
            # actually take effect.
            o3_kwargs["source"] = "analytical"
        if eff_o3_peak is not None:
            o3_kwargs["p_peak_hPa"] = eff_o3_peak
        if eff_o3_vmr is not None:
            o3_kwargs["o3_max_vmr"] = eff_o3_vmr
        rad_kwargs["ozone"] = OzoneProfileConfig(**o3_kwargs)

    phys_cfg = PhysicsConfig(
        radiation=RadiationConfig(**rad_kwargs),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    rrtmgp_fn = make_physics(phys_cfg, model_type=model_type, dt=dt)

    if hs_fn is None:
        return rrtmgp_fn

    # --- Build wrapper that sums HS + RRTMGP tendencies ---
    if model_type == "spectral_pe":
        def combined_fn(state, grid, sigma_coord, phys_state=None):
            rrtmgp_result = rrtmgp_fn(state, grid, sigma_coord, phys_state=phys_state)
            rrtmgp_tend = rrtmgp_result[0] if isinstance(rrtmgp_result, tuple) else rrtmgp_result
            phys_state_out = rrtmgp_result[1] if isinstance(rrtmgp_result, tuple) else None
            hs_tend = hs_fn(state, grid, sigma_coord)
            from legoesm.atmosphere.dynamics.spectral_pe import SpectralHydrostaticState
            summed = SpectralHydrostaticState(
                vor_hat=rrtmgp_tend.vor_hat.replace(
                    data=rrtmgp_tend.vor_hat.data + hs_tend.vor_hat.data),
                div_hat=rrtmgp_tend.div_hat.replace(
                    data=rrtmgp_tend.div_hat.data + hs_tend.div_hat.data),
                T_hat=rrtmgp_tend.T_hat.replace(
                    data=rrtmgp_tend.T_hat.data + hs_tend.T_hat.data),
                lnps_hat=rrtmgp_tend.lnps_hat.replace(
                    data=rrtmgp_tend.lnps_hat.data + hs_tend.lnps_hat.data),
                phis_hat=rrtmgp_tend.phis_hat.replace(
                    data=rrtmgp_tend.phis_hat.data + hs_tend.phis_hat.data),
            )
            return summed, phys_state_out
    else:
        # hydrostatic, nonhydrostatic, mpas — all use HydrostaticTendencies
        def combined_fn(state, grid, sigma_coord, phys_state=None):
            rrtmgp_result = rrtmgp_fn(state, grid, sigma_coord, phys_state=phys_state)
            rrtmgp_tend = rrtmgp_result[0] if isinstance(rrtmgp_result, tuple) else rrtmgp_result
            phys_state_out = rrtmgp_result[1] if isinstance(rrtmgp_result, tuple) else None
            hs_tend = hs_fn(state, grid, sigma_coord)
            from legoesm.core.state import HydrostaticTendencies
            dv_dt = None
            if rrtmgp_tend.dv_dt is not None and hs_tend.dv_dt is not None:
                dv_dt = rrtmgp_tend.dv_dt.replace(
                    data=rrtmgp_tend.dv_dt.data + hs_tend.dv_dt.data)
            elif rrtmgp_tend.dv_dt is not None:
                dv_dt = rrtmgp_tend.dv_dt
            elif hs_tend.dv_dt is not None:
                dv_dt = hs_tend.dv_dt
            summed = HydrostaticTendencies(
                du_dt=rrtmgp_tend.du_dt.replace(
                    data=rrtmgp_tend.du_dt.data + hs_tend.du_dt.data),
                dT_dt=rrtmgp_tend.dT_dt.replace(
                    data=rrtmgp_tend.dT_dt.data + hs_tend.dT_dt.data),
                dp_s_dt=rrtmgp_tend.dp_s_dt.replace(
                    data=rrtmgp_tend.dp_s_dt.data + hs_tend.dp_s_dt.data),
                dphis_dt=rrtmgp_tend.dphis_dt.replace(
                    data=rrtmgp_tend.dphis_dt.data + hs_tend.dphis_dt.data),
                dv_dt=dv_dt,
                tracer_tendencies=rrtmgp_tend.tracer_tendencies,
            )
            return summed, phys_state_out

    # Forward set_time / reset_state from the RRTMGP combined function
    if hasattr(rrtmgp_fn, 'set_time'):
        combined_fn.set_time = rrtmgp_fn.set_time
    if hasattr(rrtmgp_fn, 'reset_state'):
        combined_fn.reset_state = rrtmgp_fn.reset_state

    return combined_fn


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
    blowup_check_interval: int = 100,
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

        if step % blowup_check_interval == 0:
            is_finite, metric = check_fn(state)
            if not is_finite or metric > blowup_threshold:
                print(f"  BLOWUP at step {step}, metric={metric:.1f}")
                # iter-98: store the BLOWUP details in the diag
                # dict (private ``_blowup_info`` key) so
                # ``_write_results_txt`` callers can surface them
                # in results.txt instead of leaving the reader to
                # parse stdout.  Mirrors iter-97 fix for OMIP.
                # When ``ok=False``, the last ``diag`` entries are
                # from BEFORE the BLOWUP step — without this info
                # results.txt could falsely report "PASS-shaped"
                # last-clean values.
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

def _augment_with_rrtmgp_overrides(rows: dict[str, Any], radiation: str) -> dict[str, Any]:
    """If RRTMGP is active, append the effective GHG / cloud overrides
    to ``rows`` so they're recorded in ``results.txt`` for
    reproducibility.  Iter-32 LOW2 finding (originally inlined in
    ``run_amip``); iter-35 factored out so other runners
    (``run_held_suarez`` etc.) get the same metadata.
    """
    if radiation != "rrtmgp":
        return rows
    rows = dict(rows)  # shallow copy; preserve caller's dict
    # iter-39: ozone-profile knobs join the existing GHG/cloud trio.
    # iter-39 codex LOW: keys are lower-case + lower-case unit
    # suffix for column-name consistency with the iter-31 GHG keys.
    keys = (
        "co2_ppmv", "ch4_ppbv", "n2o_ppbv", "cloud_scheme",
        "ozone_source", "ozone_peak_hpa", "ozone_max_vmr",
    )
    for key in keys:
        val = _RUNTIME_RRTMGP_OVERRIDES.get(key)
        if val is not None:
            rows[key] = val
    return rows


def _write_results_txt(output_dir: Path, rows: dict[str, Any],
                       *, diag: dict | None = None,
                       blowup_info: dict | None = None):
    """Write results.txt for a matrix-runner case.

    iter-98: ``diag`` and ``blowup_info`` (both optional, default
    None) carry BLOWUP details from ``_run_timeloop`` (iter-98
    stored them in ``diag["_blowup_info"]``).  When ``diag`` is
    passed, the function auto-extracts ``_blowup_info`` from it,
    so callers can opt in with one-line ``diag=diag`` additions.
    When ``blowup_info`` is non-None and ``rows.get("status") ==
    "FAIL"``, the function prepends a BLOWUP marker to the
    ``notes`` field so a reader of ``results.txt`` sees the
    failure mode unambiguously instead of just "FAIL" with
    last-clean diagnostic values.
    """
    if diag is not None and blowup_info is None:
        blowup_info = diag.get("_blowup_info")
    output_dir.mkdir(parents=True, exist_ok=True)
    if blowup_info is not None and rows.get("status") == "FAIL":
        # Prepend BLOWUP info to the notes field if any.
        original_notes = rows.get("notes", "")
        blowup_str = (
            f"BLOWUP at step {blowup_info['step']} "
            f"(day {blowup_info.get('day', 0):.2f}), "
            f"reason: {blowup_info['reason']}"
        )
        if original_notes:
            rows = {**rows, "notes": f"{blowup_str}; last clean: {original_notes}"}
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
    """Save snapshot field arrays as NPZ files with proper time series format.

    NEW FORMAT: Each field is saved as a time series array with shape (n_times, ...).
    This replaces the old format where each timestep was a separate variable.

    Produces:
        snapshots_native.npz   – raw arrays keyed as ``{field}`` with time series
        snapshots_latlon.npz   – regridded to (181, 360) regular lat-lon,
                                 same key convention.  For 3-D fields the
                                 shape is (n_times, 181, 360, nlev).
    Both files also contain ``times_days`` and ``steps`` metadata arrays.
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
    latlon_arrays["lat"] = np.linspace(-90.0, 90.0, 181)
    latlon_arrays["lon"] = np.linspace(-180.0, 180.0, 360)

    for field_key in all_field_keys:
        # Collect this field across all timesteps
        field_timesteps = []
        latlon_timesteps = []

        for step in sorted_steps:
            if field_key in snapshots[step]:
                arr = np.asarray(snapshots[step][field_key], dtype=np.float64)
                field_timesteps.append(arr)

                # Regrid to lat-lon
                if field_key.endswith("_3d"):
                    regridded = _regrid_3d_level(arr, lon_deg, lat_deg, coord_kind)
                else:
                    regridded = _regrid_2d(arr, lon_deg, lat_deg, coord_kind)
                latlon_timesteps.append(regridded)
            else:
                # Field not available at this timestep - skip incomplete time series
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
        # Iter-760: switch to Fortran-faithful del-n vorticity damping
        # (sw_core.F90:1948-1999) instead of the scalar bilaplacian on
        # geographic wind components.  del6_vt_flux damps relative
        # vorticity (a true scalar, no 1/cos(lat) polar singularity)
        # via the post-step `damp_v > 0` hook in FV3EdgeShallowWaterModel.
        # Per iter-755b ablation, damp_v=0.06, nord_v=2 reduces W2
        # v_ll_Linf from 0.303 (legacy hyperdiff) to 0.214 m/s
        # (-29.4%) — the Fortran-prescribed structural fix.  Legacy
        # `hyperdiff_coeff`-on-geographic-winds path retained in
        # fv3_sw_tendencies but disabled by default here.
        #
        # Iter-761: tune `div_damp` coefficient 8× higher (8*_div_damp_cube)
        # to further reduce cube-corner mode A at lat ±35°.  Per iter-761
        # sweep, at 8× the W2 v_ll_Linf drops to 0.159 m/s (-48% from
        # 0.303 legacy) and h_L2 drops to 2.07e-4 (from 2.42e-4 legacy).
        # Stable at C16-C48.  Blowup limit at ~16×.  This tuning is
        # PRAGMATIC within the existing aggregated-div_damp API; a
        # Fortran-faithful port of d_sw5's structured `d2_bg, dddmp,
        # d4_bg, nord` (sw_core.F90:1720) is iter-759's ongoing work.
        # Iter-893: enable `apply_fortran_xppm_boundary=True` on the
        # canonical W2/W5 LEGACY production config.  iter-892's
        # off-by-one fix in `_ppm_reconstruct_1d` revealed that
        # Fortran's iord<7 cube-edge boundary formulas
        # (tp_core.F90:357-369) reduce W2 v_north Linf at C36 1-day by
        # 19.6% (0.189 → 0.152 m/s on Linf, 10.1% on L2).  iter-892
        # locked the improvement behind a default-OFF kwarg; iter-893
        # activates it on the production matrix.  W5 is essentially
        # unchanged (max|h| ≈ 5966.72 in both ON/OFF).
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=8.0 * _div_damp_cube(n),
            boundary_fix=True,
            damp_v=0.06,
            nord_v=2,
            apply_fortran_xppm_boundary=True)
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
                "mean_height": _area_weighted_mean(s.h, grid.area),
                "max_wind": float(jnp.max(jnp.abs(s.u_d))),
            }

        # Regrid wind from cell-centre averages of edge-midpoint winds.
        _cs_w = _get_cs_weights(n)

        # iter-528/529: pre-compute the 4-edge averaged cos/sin
        # angles via the canonical helper.  This replaces the inline
        # 4-edge averaging that lived here.
        from legoesm.grids.cubed_sphere_cdgrid import (
            cell_centre_angles_from_4edge,
        )
        _ca_4edge, _sa_4edge = cell_centre_angles_from_4edge(cdgrid)
        _ca_4edge_np = np.asarray(_ca_4edge, dtype=np.float64)
        _sa_4edge_np = np.asarray(_sa_4edge, dtype=np.float64)

        def extract_fn(s):
            # Average edge-midpoint winds to cell centres, then regrid.
            # The cell-centre angles for the rotation come from the
            # 4-surrounding-edge mean (helper
            # `cell_centre_angles_from_4edge`).  Cell-centre angles
            # differ from edge-averaged angles by O(dx), creating a
            # 0.39 m/s v_north residual for Williamson 2; the 4-edge
            # mean reduces this to 0.008 m/s at t=0 (47x improvement;
            # see iter-25/26 of docs/fv3_fortran_fidelity_review.md).
            u_cc = 0.5 * (np.asarray(s.u_d, dtype=np.float64)[:, :, :-1]
                          + np.asarray(s.u_d, dtype=np.float64)[:, :, 1:])
            v_cc = 0.5 * (np.asarray(s.v_d, dtype=np.float64)[:, :-1, :]
                          + np.asarray(s.v_d, dtype=np.float64)[:, 1:, :])
            u_east = _ca_4edge_np * u_cc - _sa_4edge_np * v_cc
            v_north = _sa_4edge_np * u_cc + _ca_4edge_np * v_cc
            u_ll = _regrid_2d(u_east, lon_deg, lat_deg, coord_kind)
            v_ll = _regrid_2d(v_north, lon_deg, lat_deg, coord_kind)
            # Expose face-native cell-centre geographic winds for
            # face-resolved diagnostics (iter-119).  These bypass the
            # lat-lon regridding so per-face symmetry / seam jumps
            # can be inspected directly.  Shape: (6, n, n).
            return {"u": u_ll, "v": v_ll,
                    "wind_speed": np.sqrt(u_ll ** 2 + v_ll ** 2),
                    "height": np.asarray(s.h, dtype=np.float64),
                    "u_cc_east": np.asarray(u_east, dtype=np.float64),
                    "v_cc_north": np.asarray(v_north, dtype=np.float64)}

        key_array_fn = lambda s: s.h
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
            CGridLatLonShallowWaterModel, CGridLatLonShallowWaterConfig,
            williamson_test2_cgrid, williamson_test5_cgrid,
            williamson_test2_exact_cgrid, compute_error_norms_cgrid)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        # CFL-safe dt for gravity waves near poles
        import math as _m
        _dx_pole = float(grid.radius) * grid.dlon * _m.cos(
            _m.pi / 2 - grid.dlat / 2)
        _c_grav = _m.sqrt(9.81 * 3000.0)
        dt = min(300.0, 0.5 * _dx_pole / _c_grav)
        # A_h must respect diffusion CFL: A_h*dt/dx_pole^2 < 0.5
        _A_h_max = 0.4 * _dx_pole**2 / dt
        _A_h = min(_laplacian_visc_latlon(n_lat), _A_h_max)
        config = CGridLatLonShallowWaterConfig(A_h=_A_h)
        model = CGridLatLonShallowWaterModel(grid, config, dt=dt)
        state = (williamson_test2_cgrid(grid) if test_num == 2
                 else williamson_test5_cgrid(grid))

        def step_fn(s, dt_):
            return model.step(s, dt_)

        def check_fn(s):
            return (check_finite({"h": s.h, "u": s.u}),
                    float(jnp.max(jnp.abs(s.u))))

        def scalar_fn(s):
            u_c = 0.5 * (s.u[:, :-1] + s.u[:, 1:])
            v_c = 0.5 * (s.v[:-1] + s.v[1:])
            return {
                "mean_height": _area_weighted_mean(s.h, grid.area),
                "max_wind": float(jnp.max(jnp.sqrt(u_c ** 2 + v_c ** 2))),
            }

        def extract_fn(s):
            u = np.asarray(0.5 * (s.u[:, :-1] + s.u[:, 1:]), dtype=np.float64)
            v = np.asarray(0.5 * (s.v[:-1] + s.v[1:]), dtype=np.float64)
            return {"u": u, "v": v,
                    "wind_speed": np.sqrt(u ** 2 + v ** 2),
                    "height": np.asarray(s.h, dtype=np.float64)}

        key_array_fn = lambda s: s.h
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
        if test_num == 6:
            from tests.test_cases.williamson_extended import (
                williamson_test6_mpas)
            init_fns = {6: williamson_test6_mpas}
        else:
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
                "mean_height": _area_weighted_mean(s.h.data, mesh.areaCell),
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
            spectral_filter_order=8 if test_num in (5, 6) else 0,
        )
        model = SpectralShallowWaterModel(grid, config)
        if test_num == 6:
            from tests.test_cases.williamson_extended import (
                williamson_test6_spectral)
            state = williamson_test6_spectral(grid)
            state = model.filter_initial_state(state)
        elif test_num == 2:
            state = williamson_test2_spectral(grid)
        else:
            state = williamson_test5_spectral(grid)
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
                "mean_height": _area_weighted_mean(
                    phi / constants.g, grid.grid_area),
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
    # Area-weighted reference mass: cross-grid consistency requires the
    # SAME definition on every grid.  Bare ``jnp.mean`` over-weights
    # the shrunken pole cells on lat-lon and Gaussian grids — see
    # ``_area_weighted_mean`` docstring and iter-2 commit.
    if tc.grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis as _sh
        h_init = _sh(grid, state.phi_hat.data) / constants.g
        mass_init = _area_weighted_mean(h_init, grid.grid_area)
    else:
        h_data = state.h if isinstance(state.h, jnp.ndarray) else state.h.data
        if tc.grid_type == "icosahedral":
            mass_init = _area_weighted_mean(h_data, grid.areaCell)
        else:  # cubed_sphere, latlon
            mass_init = _area_weighted_mean(h_data, grid.area)
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
        # Iter-742 (corrects iter-741 per Codex stop-time review):
        # iter-740 measured v_d-error, iter-741 corrected to pre-
        # regrid v_north, but the actual plotted field in
        # snapshots_v.png is the POST-REGRID v_ll =
        # _regrid_2d(v_north, lon_deg, lat_deg, coord_kind=cube).
        # The regrid step interpolates cube-face-native to lat-lon
        # and can shift/smooth the peak; the PNG colourbar reflects
        # v_ll magnitude, not v_north magnitude.  Iter-742 measures
        # the same quantity the snapshot plots — matching the
        # visual artifact exactly.
        #
        # The exact W2 geographic v_north is ZERO everywhere at all
        # times.  Regridding a zero field yields zero, so max|v_ll|
        # is itself the error (no subtraction).
        u_cc = 0.5 * (np.asarray(state.u_d, dtype=np.float64)[:, :, :-1]
                      + np.asarray(state.u_d, dtype=np.float64)[:, :, 1:])
        v_cc = 0.5 * (np.asarray(state.v_d, dtype=np.float64)[:, :-1, :]
                      + np.asarray(state.v_d, dtype=np.float64)[:, 1:, :])
        v_north_face = _sa_4edge_np * u_cc + _ca_4edge_np * v_cc
        v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, coord_kind)
        v_linf = float(np.max(np.abs(v_ll)))
        v_north_linf_face = float(np.max(np.abs(v_north_face)))
        norms["v_linf"] = v_linf
        notes = (f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}, "
                 f"v_ll_Linf={v_linf:.2e} "
                 f"(pre-regrid {v_north_linf_face:.2e})")
    elif test_num == 2 and tc.grid_type == "latlon":
        exact = williamson_test2_exact_cgrid(grid, days * 86400.0)
        norms = compute_error_norms_cgrid(state, exact, grid)
        notes = f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    elif test_num == 2 and tc.grid_type == "icosahedral":
        norms = compute_error_norms_mpas(state.h.data, init_fns[2](mesh).h.data, mesh)
        notes = f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    elif test_num == 2 and tc.grid_type == "spectral":
        # iter-116 (codex iter-114 HIGH-5): pre-iter-116, the
        # spectral W2 branch fell through to ``mass drift=0``
        # below — codex correctly flagged this as not a valid
        # Williamson L2 comparison.  iter-116 computes the
        # area-weighted L2/Linf height-error norm against the
        # analytical steady-state solution (which IS the
        # initial state for W2: ``williamson_test2_spectral``
        # is itself the closed-form geostrophic balance).
        from legoesm.grids.gaussian import sh_synthesis as _sh_syn
        from legoesm.atmosphere.dynamics.spectral_sw import (
            williamson_test2_spectral as _w2_spec)
        from legoesm import constants as _consts
        _init_state = _w2_spec(grid)
        _init_h = np.asarray(
            _sh_syn(grid, _init_state.phi_hat.data) / _consts.g,
            dtype=np.float64,
        )
        _final_h = np.asarray(
            _sh_syn(grid, state.phi_hat.data) / _consts.g,
            dtype=np.float64,
        )
        _err = _final_h - _init_h
        _area = np.asarray(grid.grid_area, dtype=np.float64)
        _l2 = float(np.sqrt(
            np.sum(_err ** 2 * _area)
            / np.sum(_init_h ** 2 * _area)
        ))
        _linf = float(
            np.max(np.abs(_err)) / np.max(np.abs(_init_h)))
        notes = f"L2={_l2:.2e}, Linf={_linf:.2e}"
    elif diag.get("mean_height"):
        notes = f"mass drift={_compute_drift(diag['mean_height']):.2e}"

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "status": "PASS" if ok else "FAIL",
        "notes": notes,
        "wall_time": f"{wall:.1f}s"},  # iter-29: enable iter-28 GPU efficiency table on SW
        diag=diag)  # iter-98: surface BLOWUP info if any
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
# Runner: Cosine bell advection (Putman & Lin 2007, Section 4.1)
# ===========================================================================

def run_cosine_bell(tc: TestCase, output_dir: Path, days: float, *,
                    radiation: str = "gray") -> tuple[str, float, str]:
    """Solid body rotation of a cosine bell — pure transport test.

    Implements Section 4.1 of Putman & Lin (2007).  Winds are prescribed
    (frozen) via transport-only tendency wrappers; only the height/mass
    field evolves.  After 12 days the bell returns to its initial position
    and error norms are computed against the initial condition.

    Flow angle beta = pi/4 sends the bell over the cubed-sphere corners.
    """
    from tests.test_cases.cosine_bell import (
        cosine_bell_cubesphere, cosine_bell_latlon,
        cosine_bell_mpas, cosine_bell_spectral,
        cosine_bell_error_norms, cosine_bell_exact,
    )

    beta = jnp.pi / 4.0  # Southeastward flow over corners

    # The cosine bell peak is ~1000 m, so use a larger blowup threshold.
    _CB_BLOWUP = 5000.0

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
            CDGridShallowWaterConfig)

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        dt = 1800.0
        # NOTE (iter-760b): config is DECLARATION-ONLY for this test.
        # Cosine bell is PURE HORIZONTAL ADVECTION — `model.step()` is
        # NEVER called.  Instead, the `step_fn` below uses
        # `transport_step` directly with pre-computed frozen winds
        # (_d2a2c_vect output).  The CDGridShallowWaterConfig fields
        # (div_damp, damp_v, nord_v, hyperdiff_coeff) are NOT READ by
        # the cosine-bell stepping code.  Kept in sync with the W2/W5
        # config for declaration consistency but the numerical
        # behaviour is independent of these fields.
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=_div_damp_cube(n),
            boundary_fix=True,
            damp_v=0.06,
            nord_v=2)
        model = FV3EdgeShallowWaterModel(grid, config)
        cdgrid = model.cdgrid
        state = cosine_bell_cubesphere(grid, cdgrid, beta)
        h_init = state.h.copy()
        model.set_initial_mass(state)

        # FV3-faithful transport: d2a2c_vect for contravariant velocities,
        # then Lin-Rood split transport with Courant-number PPM.
        from legoesm.core.fv3_sw_core import _d2a2c_vect
        from legoesm.core.fv_tp_2d import transport_step
        from legoesm.core.conservation import _accumulation_dtype

        # Pre-compute contravariant velocities (winds are frozen)
        _ua, _va, _uc, _vc, ut, vt = _d2a2c_vect(
            state.u_d, state.v_d, cdgrid)

        # Pre-compute initial mass for conservation fixer
        _mass_target = float(jnp.sum(state.h * grid.area))

        @jax.jit
        def step_fn(s, dt_):
            h_new = transport_step(s.h, ut, vt, dt_, cdgrid,
                                   mass_target=_mass_target)
            return s._replace(h=h_new)

        def check_fn(s):
            return (check_finite({"h": s.h}),
                    float(jnp.max(jnp.abs(s.h))))

        def scalar_fn(s):
            return {"mean_height": _area_weighted_mean(s.h, grid.area),
                    "max_height": float(jnp.max(s.h))}

        _cs_w = _get_cs_weights(n)

        def extract_fn(s):
            h_np = np.asarray(s.h, dtype=np.float64)
            return {"height": h_np}

        key_array_fn = lambda s: s.h
        coord_kind = "cube"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

        def error_fn(s, t):
            h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius,
                                        t, beta)
            return cosine_bell_error_norms(s.h, h_exact, grid.area)

    elif tc.grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
            CGridLatLonShallowWaterState, cell_to_cgrid_winds)
        from legoesm.core.operators_fv_latlon import cgrid_fv_flux_divergence_latlon
        from legoesm.timestepping.dispatch import dispatch_integrator

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        # CFL-safe dt for advection near poles (beta=pi/4 rotated flow)
        import math as _m
        _u0 = 2.0 * _m.pi * float(grid.radius) / (12.0 * 86400.0)
        _dx_pole = float(grid.radius) * grid.dlon * _m.cos(
            _m.pi / 2 - grid.dlat / 2)
        dt = min(1800.0, 0.8 * _dx_pole / _u0)

        # Build C-grid cosine bell IC from cell-centered version
        _cb_ll = cosine_bell_latlon(grid, beta)
        _u_face, _v_face = cell_to_cgrid_winds(_cb_ll.u.data, _cb_ll.v.data)
        state = CGridLatLonShallowWaterState(
            h=_cb_ll.h.data, u=_u_face, v=_v_face,
            h_s=jnp.zeros_like(_cb_ll.h.data))
        h_init = state.h.copy()

        # Transport-only step: freeze winds, only advect h.
        # Uses the same PPM operator the shipped model calls internally.
        # NO mass correction — raw transport conservation error is visible
        # in the benchmark norms.
        _u_frozen = _u_face
        _v_frozen = _v_face
        _mass_init = float(jnp.sum(state.h * grid.area))

        @jax.jit
        def step_fn(s, dt_):
            def tendency_fn(st):
                dh = cgrid_fv_flux_divergence_latlon(
                    st.h, _u_frozen, _v_frozen, grid)
                return st._replace(
                    h=dh,
                    u=jnp.zeros_like(st.u),
                    v=jnp.zeros_like(st.v),
                    h_s=jnp.zeros_like(st.h_s))
            return dispatch_integrator(
                s, tendency_fn, dt_, "ssp_rk3")

        def check_fn(s):
            return (check_finite({"h": s.h}),
                    float(jnp.max(jnp.abs(s.h))))

        def scalar_fn(s):
            return {"mean_height": _area_weighted_mean(s.h, grid.area),
                    "max_height": float(jnp.max(s.h))}

        def extract_fn(s):
            return {"height": np.asarray(s.h, dtype=np.float64)}

        key_array_fn = lambda s: s.h
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

        def error_fn(s, t):
            h_exact = cosine_bell_exact(grid.lon2d, grid.lat2d,
                                        grid.radius, t, beta)
            norms = cosine_bell_error_norms(s.h, h_exact, grid.area)
            # Report raw mass drift (no correction applied).
            #
            # iter-157: migrated from inline
            # ``abs(mass_final - _mass_init) / abs(_mass_init)`` to the
            # ``_compute_drift`` wrapper (which delegates to the centralized
            # ``legoesm.diagnostics.conservation_drift.compute_relative_drift``).
            # For cosine_bell ``_mass_init`` is always large positive (cosine
            # bell has a positive background), so the happy-path numerical
            # behavior is unchanged.  The helper adds NaN-aware behavior so
            # that a blown-up run reports NaN explicitly instead of silently
            # returning a non-finite that downstream comparisons treat as
            # False.
            mass_final = float(jnp.sum(s.h * grid.area))
            norms["mass_drift"] = _compute_drift([_mass_init, mass_final])
            return norms

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.shallow_water_mpas import (
            MPASShallowWaterModel, MPASShallowWaterConfig)
        from legoesm.core.state import MPASShallowWaterState

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        dt = 1800.0
        # Use SSP-RK3 for positivity; RK4 default can overshoot.
        config = MPASShallowWaterConfig(
            nu_del4=_hyperdiff_ico(mesh), time_integrator="ssp_rk3")
        model = MPASShallowWaterModel(mesh, config)
        state = cosine_bell_mpas(mesh, beta)
        h_init = state.h.data.copy()
        grid = mesh

        from legoesm.timestepping.dispatch import dispatch_integrator

        # Compute ONLY mass flux divergence, bypassing the momentum
        # equation entirely.  The PV computation in the full tendency
        # divides by h, producing inf where h=0 (outside the bell).
        from legoesm.core.operators_voronoi import (
            thickness_flux, divergence_cell)

        @jax.jit
        def step_fn(s, dt_):
            def tendency_fn_transport(st):
                h_flux = thickness_flux(
                    st.h.data, st.u.data, mesh,
                    order=config.thickness_order)
                dh_dt_data = -divergence_cell(h_flux, mesh)
                return MPASShallowWaterState(
                    h=st.h.replace(data=dh_dt_data),
                    u=st.u.replace(data=jnp.zeros_like(st.u.data)),
                    h_s=st.h_s.replace(data=jnp.zeros_like(st.h_s.data)))
            s_new = dispatch_integrator(
                s, tendency_fn_transport, dt_, config.time_integrator)
            # Positivity limiter + mass conservation fixer.
            # The centred thickness flux can produce negative h;
            # clamp to zero then restore total mass.
            h_new = jnp.maximum(s_new.h.data, 0.0)
            area = mesh.areaCell
            mass_old = jnp.sum(s.h.data * area)
            mass_new = jnp.sum(h_new * area)
            total_area = jnp.sum(area)
            correction = (mass_old - mass_new) / total_area
            h_new = h_new + correction
            s_new = s_new._replace(
                h=s_new.h.replace(data=h_new))
            return s_new

        lon_cell = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_cell = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi

        def check_fn(s):
            return (check_finite({"h": s.h.data}),
                    float(jnp.max(jnp.abs(s.h.data))))

        def scalar_fn(s):
            return {"mean_height": _area_weighted_mean(
                        s.h.data, mesh.areaCell),
                    "max_height": float(jnp.max(s.h.data))}

        def extract_fn(s):
            return {"height": _bin_to_latlon(
                np.asarray(s.h.data, dtype=np.float64),
                lon_cell, lat_cell)}

        key_array_fn = lambda s: s.h.data
        coord_kind = "latlon"  # already regridded
        lon_deg = np.linspace(-180, 180, 360, endpoint=False)
        lat_deg = np.linspace(-90, 90, 181)

        def error_fn(s, t):
            h_exact = cosine_bell_exact(mesh.lonCell, mesh.latCell,
                                        mesh.radius, t, beta)
            return cosine_bell_error_norms(s.h.data, h_exact, mesh.areaCell)

    elif tc.grid_type == "spectral":
        from legoesm.grids.gaussian import (
            create_gaussian_grid, sh_synthesis, sh_analysis)
        from legoesm.atmosphere.dynamics.spectral_sw import (
            SpectralShallowWaterModel, SpectralSWConfig,
            SpectralSWState, spectral_sw_tendencies)
        from legoesm import constants as C

        n_max = int(tc.resolution.replace("T", ""))
        grid = create_gaussian_grid(n_max)
        dt = min(1800.0, 0.5 * grid.radius / (n_max * 40.0))
        config = SpectralSWConfig(spectral_filter_order=8)
        model = SpectralShallowWaterModel(grid, config)
        state = cosine_bell_spectral(grid, beta)
        phi_init = sh_synthesis(grid, state.phi_hat.data).copy()

        from legoesm.timestepping.dispatch import dispatch_integrator

        @jax.jit
        def step_fn(s, dt_):
            def tendency_fn_transport(st):
                full = spectral_sw_tendencies(st, grid, config)
                # Keep only phi tendency; freeze vor and div (winds)
                return SpectralSWState(
                    vor_hat=st.vor_hat.replace(
                        data=jnp.zeros_like(st.vor_hat.data)),
                    div_hat=st.div_hat.replace(
                        data=jnp.zeros_like(st.div_hat.data)),
                    phi_hat=full.phi_hat,
                    phis_hat=st.phis_hat.replace(
                        data=jnp.zeros_like(st.phis_hat.data)))
            result = dispatch_integrator(
                s, tendency_fn_transport, dt_, 'ssp_rk3')
            return model._apply_filter(result)

        def check_fn(s):
            phi = sh_synthesis(grid, s.phi_hat.data)
            return (check_finite({"phi": phi}),
                    float(jnp.max(phi / C.g)))

        def scalar_fn(s):
            phi = sh_synthesis(grid, s.phi_hat.data)
            return {"mean_height": _area_weighted_mean(
                        phi / C.g, grid.grid_area),
                    "max_height": float(jnp.max(phi / C.g))}

        def extract_fn(s):
            phi = np.asarray(sh_synthesis(grid, s.phi_hat.data),
                             dtype=np.float64)
            return {"height": phi / float(C.g)}

        key_array_fn = lambda s: s.phi_hat.data
        coord_kind = "gaussian"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

        def error_fn(s, t):
            phi = sh_synthesis(grid, s.phi_hat.data)
            h_exact = cosine_bell_exact(grid.lon2d, grid.lat2d,
                                        grid.radius, t, beta)
            return cosine_bell_error_norms(phi / C.g, h_exact, grid.grid_area)

    else:
        raise NotImplementedError(
            f"Cosine bell not implemented for grid '{tc.grid_type}'")

    # --- Time loop ---
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    state, snapshots, diag, wall, ok = _run_timeloop(
        step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
        diag_every, key_array_fn,
        label=f"Cosine Bell ({tc.grid_type})", total_days=days,
        blowup_threshold=_CB_BLOWUP)

    # --- Error norms (compare to initial condition after full revolution) ---
    notes = ""
    norms = error_fn(state, days * 86400.0)
    notes = f"L1={norms['l1']:.2e}, L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    # iter-120 (codex iter-119-followup MEDIUM-2): unified mass
    # drift across all 4 cosine_bell grids.  Pre-iter-120 only
    # the latlon error_fn computed ``mass_drift`` (line ~2079);
    # cube/ico/spectral skipped the gate entirely.  iter-120
    # falls back to ``_compute_drift(diag["mean_height"])`` for
    # grids whose ``error_fn`` doesn't supply ``mass_drift``.
    # ``mean_height`` is the area-weighted mean of h, so its
    # relative drift equals the relative mass drift (linear).
    if "mass_drift" in norms:
        mass_drift = norms["mass_drift"]
        notes += f", mass_drift={mass_drift:.2e}"
        n_mass_samples = len(diag.get("mean_height", []))
    else:
        mass_drift = _compute_drift(diag.get("mean_height", []))
        n_mass_samples = len(diag.get("mean_height", []))
        if mass_drift > 0 or n_mass_samples >= 2:
            notes += f", mass_drift={mass_drift:.2e}"
    if mass_drift > 0.01:
        logger.warning(
            "Cosine bell %s: mass drift %.2e exceeds 1%% threshold",
            tc.grid_type, mass_drift)
    # iter-119 (codex iter-118-followup MEDIUM-1): apply
    # the iter-117/118 mass-drift PASS gate.  Pre-iter-119
    # cosine_bell only WARNED.  iter-120: now applies to ALL
    # 4 grids via the unified mass_drift / n_mass_samples
    # path above.
    ok, notes = _apply_mass_drift_tolerance(
        ok, notes, mass_drift, 1e-2,
        n_samples=n_mass_samples)

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "status": "PASS" if ok else "FAIL",
        "notes": notes,
        "wall_time": f"{wall:.1f}s"},  # iter-29: enable iter-28 GPU efficiency table on cosine bell
        diag=diag)  # iter-99: surface BLOWUP info if any
    _save_case_diagnostics(
        output_dir, f"Cosine Bell PL07 {tc.resolution}", dt,
        diag, snapshots, coord_kind, lon_deg, lat_deg,
        field_specs_2d=[
            ("height", "Height h (m)", "viridis"),
        ],
        mass_key="mean_height", energy_key="max_height",
        scalar_units={"mean_height": "m", "max_height": "m"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner: Held-Suarez
# ===========================================================================

def run_held_suarez(tc: TestCase, output_dir: Path, days: float, *,
                    radiation: str = "gray") -> tuple[str, float, str]:
    nlev = DEFAULT_NLEV
    # When tc.case == "held_suarez_topo" we swap the init for the
    # topography-aware version (forcing function is unchanged — see
    # ``src/legoesm/atmosphere/idealized/held_suarez_topo.py``).
    _topo = tc.case == "held_suarez_topo"
    _topo_h0 = float(tc.run_kwargs.get("h_0", 2000.0)) if _topo else 0.0

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel as PrimitiveEquationModel,
            CDGridPrimitiveEquationConfig as PrimitiveEquationConfig)
        from legoesm.atmosphere.held_suarez import (
            held_suarez_forcing, held_suarez_init)
        from legoesm.core.operators import global_integral

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        hd = _hyperdiff_cube(n)
        dd = _div_damp_cube(n)
        ah = _laplacian_visc_cube(n)
        # FV3_3D iter 33/34: scale A_h via env var.  matrix default
        # is INSUFFICIENT at C72+ (iter 33 found C72 NaN at default
        # A_h but stable at 10x).  Default 1.0 preserves iter-17/24
        # C36/C48 behaviour; set LEGOESM_AH_SCALE=10.0 at C72.
        _ah_scale = float(os.environ.get("LEGOESM_AH_SCALE", "1.0"))
        ah = ah * _ah_scale
        # FV3_3D iter 41: print a stability tip at C72+ when the
        # user has NOT set LEGOESM_AH_SCALE — the iter-33 finding
        # showed default A_h NaNs at C72 (~13 days into the 30-day
        # HS spin-up).  Recommend ah_scale=2.0 at C48, 10.0 at C72.
        if n >= 72 and _ah_scale < 5.0:
            print(
                f"[FV3_3D iter 41 stability tip] At C{n} the matrix "
                f"default A_h={ah:.2e} is INSUFFICIENT and the run "
                f"will likely NaN around day 13.  Recommended: set "
                f"LEGOESM_AH_SCALE=10.0 (or use _laplacian_visc_cube_v2(n) "
                f"directly) per FV3_3D.md iter 33-39.",
                flush=True,
            )
        dt = 200.0
        # Iter-15 NOTE on the cubed-sphere upper-atmosphere sponge:
        # The default ``sponge_tau_sec = 3600`` (1 hour) is FAR more
        # aggressive than the FV3 Fortran reference
        # (``../FV3/atmos_cubed_sphere-symmetryclean/model/dyn_core.F90``,
        # subroutine ``Ray_fast`` line 2922-2985) which uses ``tau``
        # in DAYS — typical production setting is 5-10 days, i.e.
        # ~430-860x weaker damping.  However, EMPIRICALLY for this
        # SPECIFIC HS configuration (gray radiation, hydrostatic,
        # ``sigma`` vertical coord, C36 / 72×144 / ico5 / T21
        # resolutions, ``DEFAULT_NLEV = 40``, 30-day quick spin-up,
        # cross-grid mean_T gap measured at t=30 d as
        # ``mean_T(latlon, t=30) - mean_T(cubed, t=30)``):
        #   - τ = 1 h (default): cube-vs-latlon mean_T gap  -5.0 K
        #   - τ = 7 d (FV3-like):                          -10.8 K
        #   - τ = ∞ (sponge OFF, iter-13):                 -11.3 K
        # The aggressive 1-h sponge produces the BEST cross-grid
        # agreement on this metric, despite being non-canonical for
        # HS.  Suspected cause: the cubed-sphere hyperdiffusion +
        # sponge combination is empirically tuned to roughly match
        # the effective dissipation that lat-lon's Laplacian
        # viscosity provides; a weaker sponge under-damps the
        # cubed-sphere upper troposphere and the climatology drifts
        # further from the lat-lon / icosahedral / spectral cluster.
        # NOT yet established for HYBRID coord, RRTMGP radiation,
        # or 200-day spin-up — those may have different optimal τ.
        # Keep the 1-h default for this HS test until a more
        # principled retuning is done.
        # FV3_3D iter 13: optional FV3-faithful post-step vorticity
        # damping (SW backbone reuse).  Set LEGOESM_DAMP_V=0.30 to
        # opt in (~17 % mid-level cube-imprint reduction at C36).
        _damp_v_env = float(os.environ.get("LEGOESM_DAMP_V", "0.0"))
        # FV3_3D iter 16: optional FV3-faithful B-grid corner-divergence
        # damping (port of sw_core.F90:divergence_corner + d_sw5
        # adaptive damping).  Set LEGOESM_CDD_D2BG=0.001 to opt in
        # (~71 % mid-level cube-imprint reduction at C36 — best result
        # to date).  Use values 0.001-0.005; 0.010 destabilises.
        _cdd_d2_bg_env = float(os.environ.get("LEGOESM_CDD_D2BG", "0.0"))
        # FV3_3D iter 18: optional higher-order del-(2*(nord+1)) corner-
        # divergence damping (port of sw_core.F90:1725-1822 nord>0
        # branch).  Active only when BOTH LEGOESM_CDD_D4BG > 0 AND
        # LEGOESM_CDD_NORD > 0.  Typical FV3 production: d4_bg=0.16,
        # nord=2.  At C36 the unit-equivalent values scale down by
        # (96/36)^2 ~ 7.1, so d4_bg ~ 0.02 for nord=1 / nord=2.
        _cdd_d4_bg_env = float(os.environ.get("LEGOESM_CDD_D4BG", "0.0"))
        _cdd_nord_env = int(os.environ.get("LEGOESM_CDD_NORD", "0"))
        # FV3_3D iter 22: optional FV3-fully-faithful vector cube-
        # vertex fill (sw_core.F90:1762).  At nord=1 it is a
        # mathematical no-op (bit-for-bit preserves iter-18); flagged
        # here for users who want to verify FV3 fidelity end-to-end.
        # Set LEGOESM_CDD_FV3_VFILL=1 (or 'true', 'yes', 'on') to opt in.
        _cdd_fv3_vfill_env = (
            os.environ.get("LEGOESM_CDD_FV3_VFILL", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            damp_v=_damp_v_env, nord_v=2,
            corner_div_damp_d2_bg=_cdd_d2_bg_env,
            corner_div_damp_dddmp=0.20,
            corner_div_damp_d4_bg=_cdd_d4_bg_env,
            corner_div_damp_nord=_cdd_nord_env,
            corner_div_damp_fv3_vector_fill=_cdd_fv3_vfill_env,
            use_conservation_fixer=True, fix_mass=True)
        model = PrimitiveEquationModel(grid, sigma, config)
        if _topo:
            from legoesm.atmosphere.idealized.held_suarez_topo import (
                held_suarez_topo_init)
            state = held_suarez_topo_init(grid, sigma, h_0=_topo_h0)
        else:
            state = held_suarez_init(grid, sigma)

        physics_fn = (_make_rrtmgp_physics("hydrostatic", dt, hs_fn=held_suarez_forcing)
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
                "mean_T": _area_weighted_mean(s.T.data, grid.area),
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
        from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
            hydrostatic_to_cgrid)
        from legoesm.atmosphere.held_suarez import (
            held_suarez_forcing_latlon, held_suarez_init_latlon)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        ah = _laplacian_visc_latlon(n_lat)
        # CFL-safe dt for explicit RK on lat-lon polar cells
        import math as _m
        _dx_pole = float(grid.radius) * grid.dlon * _m.cos(
            _m.pi / 2 - grid.dlat / 2)
        dt = min(200.0, 0.5 * _dx_pole / 300.0)
        _A_h_max = 0.4 * _dx_pole**2 / dt
        ah = min(ah, _A_h_max)
        config = CGridLatLonPrimitiveEquationConfig(A_h=ah, fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
        if _topo:
            from legoesm.atmosphere.idealized.held_suarez_topo import (
                held_suarez_topo_init_latlon)
            state_cc = held_suarez_topo_init_latlon(
                grid, sigma, h_0=_topo_h0)
        else:
            state_cc = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(state_cc, grid)

        physics_fn = (_make_rrtmgp_physics("hydrostatic", dt, hs_fn=held_suarez_forcing_latlon)
                      if radiation == "rrtmgp" else held_suarez_forcing_latlon)

        def step_fn(s, dt_):
            return model.step_with_physics(s, dt_, physics_fn)

        mass_fn = lambda s: float(jnp.sum(s.p_s * grid.area))

        def check_fn(s):
            return (check_finite({"T": s.T, "u": s.u}),
                    float(jnp.max(jnp.abs(s.u))))

        def scalar_fn(s):
            u_c = 0.5 * (s.u[:, :-1, :] + s.u[:, 1:, :])
            v_c = 0.5 * (s.v[:-1, :, :] + s.v[1:, :, :])
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(u_c ** 2 + v_c ** 2))),
                "mean_T": _area_weighted_mean(s.T, grid.area),
            }

        def extract_fn(s):
            u_sfc = np.asarray(0.5 * (s.u[:, :-1, -1] + s.u[:, 1:, -1]), dtype=np.float64)
            v_sfc = np.asarray(0.5 * (s.v[:-1, :, -1] + s.v[1:, :, -1]), dtype=np.float64)
            return {
                "u": u_sfc, "v": v_sfc,
                "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
                "p_s": np.asarray(s.p_s, dtype=np.float64),
                "T_3d": np.asarray(s.T, dtype=np.float64),
            }

        key_array_fn = lambda s: s.T
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
        from legoesm.atmosphere.held_suarez import (
            held_suarez_forcing_mpas, held_suarez_init_mpas)

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        dt = 200.0
        ah = _laplacian_visc_ico(mesh)
        config = MPASPrimitiveEquationConfig(
            nu_del4=_hyperdiff_ico(mesh), nu_del2=ah, fix_mass=True)
        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        if _topo:
            from legoesm.atmosphere.idealized.held_suarez_topo import (
                held_suarez_topo_init_mpas)
            state = held_suarez_topo_init_mpas(mesh, sigma, h_0=_topo_h0)
        else:
            state = held_suarez_init_mpas(mesh, sigma)
        grid = mesh

        physics_fn_mpas = (_make_rrtmgp_physics("mpas", dt, hs_fn=held_suarez_forcing_mpas)
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
                "mean_T": _area_weighted_mean(s.T.data, mesh.areaCell),
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
        from legoesm.atmosphere.held_suarez import (
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
        # iter-47 codex MEDIUM + post-merge with main HS-topo
        # extension: rely on the canonical ``T_init=300.0``
        # default for the no-topo branch (keeps the spectral
        # branch consistent with cube/latlon/mpas in the same
        # function); pass T_init explicitly only on the topo
        # branch where ``held_suarez_topo_init_spectral`` may
        # have a different default.
        if _topo:
            from legoesm.atmosphere.idealized.held_suarez_topo import (
                held_suarez_topo_init_spectral)
            state = held_suarez_topo_init_spectral(
                grid, sigma, h_0=_topo_h0, T_init=300.0)
        else:
            state = isothermal_rest_state_spectral(grid, sigma)

        physics_fn = (_make_rrtmgp_physics("spectral_pe", dt, hs_fn=held_suarez_forcing_spectral)
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
                # ``mass`` is the area integral ``∫ p_s dA`` (Pa·m²),
                # consistent with the cube/latlon/ico paths
                # (``mass_fn = lambda s: float(jnp.sum(s.p_s * area))``).
                # iter-5 codex review H1 caught my iter-3 mistake of
                # using the area-weighted MEAN here (Pa) — that broke
                # cross-grid ``mass`` time-series comparability.
                "mass": float(jnp.sum(fields['p_s'] * grid.grid_area)),
                "max_wind": float(jnp.max(jnp.sqrt(
                    fields['u'] ** 2 + fields['v'] ** 2))),
                "mean_T": _area_weighted_mean(fields['T'], grid.grid_area),
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

    # iter-117 (codex iter-114 HIGH-5 followup): apply a
    # mass-drift tolerance to the HS PASS criteria.  Pre-iter-117
    # ``ok`` only checked finiteness + non-blown-up, allowing
    # latlon/spectral mass drift ~1e-4 to PASS while cube/ico
    # are at machine precision (~1e-11).  iter-117 adds 1e-2.
    # iter-118 (codex iter-117 followup MEDIUM-2): also fail
    # on non-finite drift via ``_apply_mass_drift_tolerance``.
    # iter-120 (codex iter-119 followup MEDIUM-1): also fail
    # on series with < 2 samples (pre-iter-120 returned 0.0
    # sentinel that silently passed).
    HELD_SUAREZ_MASS_DRIFT_TOL = 1e-2
    ok, notes = _apply_mass_drift_tolerance(
        ok, notes, mass_drift, HELD_SUAREZ_MASS_DRIFT_TOL,
        n_samples=len(diag.get("mass", [])))

    level_values = np.asarray(
        getattr(sigma, "sigma_full", np.arange(nlev)), dtype=np.float64)

    _write_results_txt(output_dir, _augment_with_rrtmgp_overrides({
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "vertical_coord": tc.vertical_coord, "radiation": radiation,
        "days": days, "dt": dt, "levels": nlev,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"}, radiation),
        diag=diag)  # iter-98: surface BLOWUP info if any
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
        scalar_units={"mass": "Pa*m^2", "max_wind": "m/s", "mean_T": "K"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# DCMIP 2008 dry-3D dispatch helper
# ===========================================================================

def _build_dcmip2008_state(case: str, grid_kind: str, grid_or_mesh, sigma):
    """Dispatch ``(case, grid_kind) -> init function`` for DCMIP-2008
    dry-3D Hughes-tutorial cases.

    ``grid_kind`` is one of ``"cube" / "latlon" / "mpas" / "spectral"``;
    the returned state matches the tendency dispatch each downstream
    runner expects.
    """
    if case == "gravity_wave_3_1":
        from tests.test_cases.dcmip2008 import gravity_wave_3_1 as _m
        fn = {
            "cube": _m.gravity_wave_init,
            "latlon": _m.gravity_wave_init_latlon,
            "mpas": _m.gravity_wave_init_mpas,
            "spectral": _m.gravity_wave_init_spectral,
        }[grid_kind]
    elif case == "inertio_gravity_3_2":
        from tests.test_cases.dcmip2008 import inertio_gravity_3_2 as _m
        fn = {
            "cube": _m.inertio_gravity_init,
            "latlon": _m.inertio_gravity_init_latlon,
            "mpas": _m.inertio_gravity_init_mpas,
            "spectral": _m.inertio_gravity_init_spectral,
        }[grid_kind]
    elif case == "mountain_rossby_5_0":
        from tests.test_cases.dcmip2008 import mountain_rossby_5_0 as _m
        fn = {
            "cube": _m.mountain_rossby_init,
            "latlon": _m.mountain_rossby_init_latlon,
            "mpas": _m.mountain_rossby_init_mpas,
            "spectral": _m.mountain_rossby_init_spectral,
        }[grid_kind]
    elif case == "rossby_haurwitz_6_0":
        from tests.test_cases.dcmip2008 import rossby_haurwitz_6_0 as _m
        fn = {
            "cube": _m.rossby_haurwitz_init,
            "latlon": _m.rossby_haurwitz_init_latlon,
            "mpas": _m.rossby_haurwitz_init_mpas,
            "spectral": _m.rossby_haurwitz_init_spectral,
        }[grid_kind]
    else:
        raise ValueError(f"Unknown DCMIP 2008 dry-3D case: {case!r}")
    return fn(grid_or_mesh, sigma)


# ===========================================================================
# Runner: Baroclinic Wave
# ===========================================================================

def run_baroclinic(tc: TestCase, output_dir: Path, days: float, *,
                   radiation: str = "gray") -> tuple[str, float, str]:
    nlev = DEFAULT_NLEV
    # Rotated J-W variant (DCMIP 2008 §4-1 / §4-2).  ``alpha`` defaults
    # to π/4 (45°) per DCMIP convention; ``perturbed=False`` selects the
    # steady-state DCMIP §4-1 rotated test.
    _rotated = tc.case in ("rotated_baroclinic", "rotated_steady")
    _rot_alpha = float(tc.run_kwargs.get("alpha", jnp.pi / 4.0)) if _rotated else 0.0
    _rot_perturbed = bool(tc.run_kwargs.get(
        "perturbed", tc.case == "rotated_baroclinic")) if _rotated else True
    # DCMIP 2012 §2-0-0 rest-state-with-topography variant.  Replaces the
    # baroclinic init with a true rest state over a ridged cosine-bell
    # mountain.  The dycore should preserve rest indefinitely; spurious
    # max-wind growth diagnoses pressure-gradient-force errors on
    # terrain-following coordinates.
    _rest = tc.case == "rest_state_topo"
    _rest_h0 = float(tc.run_kwargs.get("h_0", 2000.0)) if _rest else 0.0
    # DCMIP 2008 dry-3D family: gravity wave (§3-1), inertio-gravity
    # wave (§3-2), mountain Rossby (§5-0), Rossby-Haurwitz (§6-0).
    # All four use isothermal hydrostatic state + zonal solid-body
    # flow as the base; they differ in perturbation / topography only.
    _dcmip2008_dry = tc.case in (
        "gravity_wave_3_1", "inertio_gravity_3_2",
        "mountain_rossby_5_0", "rossby_haurwitz_6_0",
    )
    # DCMIP 2008 §3-1 is canonical *only* on a non-rotating sphere:
    # the gravity-wave packet is supposed to disperse free of inertial
    # effects.  When this case is selected we override the planetary
    # rotation rate at grid-construction time (the four grid factories
    # all accept ``omega``); ``None`` leaves the factory's default
    # (Earth) Ω in place.
    _omega_override: float | None = (
        0.0 if tc.case == "gravity_wave_3_1" else None
    )

    if tc.grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel as PrimitiveEquationModel,
            CDGridPrimitiveEquationConfig as PrimitiveEquationConfig)
        from tests.test_cases.baroclinic_wave import (
            baroclinic_wave_init)
        from legoesm.core.operators import global_integral

        n = int(tc.resolution[1:])
        _cube_kwargs = {}
        if _omega_override is not None:
            _cube_kwargs["omega"] = _omega_override
        grid = create_cubed_sphere(n, **_cube_kwargs)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        hd = _hyperdiff_cube(n)
        dd = _div_damp_cube(n)
        ah = _laplacian_visc_cube(n)
        # FV3_3D iter 33/34: scale A_h via env var.  matrix default
        # is INSUFFICIENT at C72+ (iter 33 found C72 NaN at default
        # A_h but stable at 10x).  Default 1.0 preserves iter-17/24
        # C36/C48 behaviour; set LEGOESM_AH_SCALE=10.0 at C72.
        _ah_scale = float(os.environ.get("LEGOESM_AH_SCALE", "1.0"))
        ah = ah * _ah_scale
        # FV3_3D iter 41: print a stability tip at C72+ when the
        # user has NOT set LEGOESM_AH_SCALE — the iter-33 finding
        # showed default A_h NaNs at C72 (~13 days into the 30-day
        # HS spin-up).  Recommend ah_scale=2.0 at C48, 10.0 at C72.
        if n >= 72 and _ah_scale < 5.0:
            print(
                f"[FV3_3D iter 41 stability tip] At C{n} the matrix "
                f"default A_h={ah:.2e} is INSUFFICIENT and the run "
                f"will likely NaN around day 13.  Recommended: set "
                f"LEGOESM_AH_SCALE=10.0 (or use _laplacian_visc_cube_v2(n) "
                f"directly) per FV3_3D.md iter 33-39.",
                flush=True,
            )
        dt = 200.0
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = PrimitiveEquationModel(grid, sigma, config)
        if _rotated:
            from tests.test_cases.dcmip2008.jablonowski_rotated import (
                rotated_baroclinic_init)
            state = rotated_baroclinic_init(
                grid, sigma_for_init,
                perturbed=_rot_perturbed, alpha=_rot_alpha)
        elif _rest:
            from tests.test_cases.dcmip2012.rest_state_topography import (
                rest_state_topography_init)
            state = rest_state_topography_init(grid, sigma, h_0=_rest_h0)
        elif _dcmip2008_dry:
            state = _build_dcmip2008_state(tc.case, "cube", grid, sigma)
        else:
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
        from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
            hydrostatic_to_cgrid)
        from tests.test_cases.baroclinic_wave import (
            baroclinic_wave_init_latlon)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        _ll_kwargs = {}
        if _omega_override is not None:
            _ll_kwargs["omega"] = _omega_override
        grid = create_latlon_grid(n_lat, n_lon, **_ll_kwargs)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        ah = _laplacian_visc_latlon(n_lat)
        import math as _m
        _dx_pole = float(grid.radius) * grid.dlon * _m.cos(
            _m.pi / 2 - grid.dlat / 2)
        dt = min(200.0, 0.5 * _dx_pole / 300.0)
        _A_h_max = 0.4 * _dx_pole**2 / dt
        ah = min(ah, _A_h_max)
        config = CGridLatLonPrimitiveEquationConfig(A_h=ah, fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
        if _rotated:
            from tests.test_cases.dcmip2008.jablonowski_rotated import (
                rotated_baroclinic_init_latlon)
            state_cc = rotated_baroclinic_init_latlon(
                grid, sigma_for_init,
                perturbed=_rot_perturbed, alpha=_rot_alpha)
        elif _rest:
            from tests.test_cases.dcmip2012.rest_state_topography import (
                rest_state_topography_init_latlon)
            state_cc = rest_state_topography_init_latlon(
                grid, sigma, h_0=_rest_h0)
        elif _dcmip2008_dry:
            state_cc = _build_dcmip2008_state(
                tc.case, "latlon", grid, sigma)
        else:
            state_cc = baroclinic_wave_init_latlon(
                grid, sigma_for_init, perturbed=True)
        state = hydrostatic_to_cgrid(state_cc, grid)

        def step_fn(s, dt_):
            return model.step(s, dt_)

        mass_fn = lambda s: float(jnp.sum(s.p_s * grid.area))

        def check_fn(s):
            return (check_finite({"T": s.T, "u": s.u}),
                    float(jnp.max(jnp.abs(s.u))))

        ps_init = np.array(state.p_s)

        def scalar_fn(s):
            u_c = 0.5 * (s.u[:, :-1, :] + s.u[:, 1:, :])
            v_c = 0.5 * (s.v[:-1, :, :] + s.v[1:, :, :])
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(u_c ** 2 + v_c ** 2))),
                "ps_perturbation": float(jnp.max(jnp.abs(s.p_s - ps_init))),
            }

        def extract_fn(s):
            u_sfc = np.asarray(0.5 * (s.u[:, :-1, -1] + s.u[:, 1:, -1]), dtype=np.float64)
            v_sfc = np.asarray(0.5 * (s.v[:-1, :, -1] + s.v[1:, :, -1]), dtype=np.float64)
            return {
                "u": u_sfc, "v": v_sfc,
                "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
                "p_s": np.asarray(s.p_s, dtype=np.float64),
                "T_3d": np.asarray(s.T, dtype=np.float64),
            }

        key_array_fn = lambda s: s.T
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
        from tests.test_cases.baroclinic_wave import (
            baroclinic_wave_init_mpas)

        level = int(tc.resolution.replace("ico", ""))
        _vor_kwargs = {}
        if _omega_override is not None:
            _vor_kwargs["omega"] = _omega_override
        mesh = create_voronoi_mesh(level, **_vor_kwargs)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        dt = 200.0
        ah = _laplacian_visc_ico(mesh)
        config = MPASPrimitiveEquationConfig(
            nu_del4=_hyperdiff_ico(mesh), nu_del2=ah, fix_mass=True)
        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        if _rotated:
            from tests.test_cases.dcmip2008.jablonowski_rotated import (
                rotated_baroclinic_init_mpas)
            state = rotated_baroclinic_init_mpas(
                mesh, sigma_for_init,
                perturbed=_rot_perturbed, alpha=_rot_alpha)
        elif _rest:
            from tests.test_cases.dcmip2012.rest_state_topography import (
                rest_state_topography_init_mpas)
            state = rest_state_topography_init_mpas(
                mesh, sigma, h_0=_rest_h0)
        elif _dcmip2008_dry:
            state = _build_dcmip2008_state(tc.case, "mpas", mesh, sigma)
        else:
            state = baroclinic_wave_init_mpas(
                mesh, sigma_for_init, perturbed=True)
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
            spectral_pe_to_grid,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral

        n_max = int(tc.resolution.replace("T", ""))
        _gauss_kwargs = {}
        if _omega_override is not None:
            _gauss_kwargs["omega"] = _omega_override
        grid = create_gaussian_grid(n_max, **_gauss_kwargs)
        sigma_for_init = create_sigma_coordinate(nlev)
        sigma = _create_vertical(nlev, tc.vertical_coord)
        dt = 600.0
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=2.338e15 * (21.0 / n_max) ** 4,
            spectral_filter_order=8,
            spectral_filter_strength=0.01,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)
        if _rotated:
            from tests.test_cases.dcmip2008.jablonowski_rotated import (
                rotated_baroclinic_init_spectral)
            state = rotated_baroclinic_init_spectral(
                grid, sigma_for_init,
                perturbed=_rot_perturbed, alpha=_rot_alpha)
        elif _rest:
            from tests.test_cases.dcmip2012.rest_state_topography import (
                rest_state_topography_init_spectral)
            state = rest_state_topography_init_spectral(
                grid, sigma, h_0=_rest_h0)
        elif _dcmip2008_dry:
            state = _build_dcmip2008_state(
                tc.case, "spectral", grid, sigma)
        else:
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
                # See iter-5 H1: ``mass`` is the integral, not the mean.
                "mass": float(jnp.sum(fields['p_s'] * grid.grid_area)),
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
    # iter-118 (codex iter-117 followup MEDIUM-3): apply
    # a 1e-2 mass-drift tolerance to baroclinic.  Same
    # rationale as iter-117 HS.  iter-120 also gates on
    # n_samples >= 2 (codex iter-119 followup MEDIUM-1).
    ok, notes = _apply_mass_drift_tolerance(
        ok, notes, mass_drift, 1e-2,
        n_samples=len(diag.get("mass", [])))

    level_values = np.asarray(
        getattr(sigma, "sigma_full", np.arange(nlev)), dtype=np.float64)

    _write_results_txt(output_dir, {
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "vertical_coord": tc.vertical_coord, "days": days, "dt": dt,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-99: surface BLOWUP info if any
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
            "mass": "Pa*m^2", "max_wind": "m/s",
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

    # TC13 grid-specific dt: the tilted solid-body rotation (u0~38 m/s)
    # violates the horizontal CFL on lat-lon grids near the poles where
    # dx -> 0.  On icosahedral meshes the centered advective scheme
    # requires a moderately smaller dt to avoid dispersive instability.
    if test_num == 13:
        if tc.grid_type == "latlon":
            dt = 120.0   # CFL < 0.8 at pole-nearest grid point
        elif tc.grid_type == "icosahedral":
            dt = 600.0   # reduce dispersion errors in centered scheme
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

        # Wind wrapper: convert geographic (u_east, v_north, sigma_dot) API
        # to the (t, lon2d, lat2d, sigma_coord) signature.
        wind_geo_fn = cfg["wind_geo"]

        def latlon_wind(t, lon2d, lat2d, sc):
            return wind_geo_fn(t, lon2d, lat2d, sc)

        init_fns = {
            11: dcmip11_init_latlon,
            12: dcmip12_init_latlon,
            13: dcmip13_init_latlon,
        }
        state_init = init_fns[test_num](grid, sigma_coord)
        model = TracerTransportLatLonModel(
            grid, sigma_coord, latlon_wind,
            TracerTransportLatLonConfig())
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

    _area_for_mean = _grid_cell_area(grid)

    def scalar_fn(s):
        q = s.tracers.data
        return {
            "q1_min": float(jnp.min(q[..., 0])),
            "q1_max": float(jnp.max(q[..., 0])),
            "q1_mean": _area_weighted_mean(q[..., 0], _area_for_mean),
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
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-99: surface BLOWUP info if any
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
        from legoesm.atmosphere.held_suarez import held_suarez_init
        from legoesm.core.operators import global_integral

        from legoesm.atmosphere.held_suarez import held_suarez_forcing

        n = int(tc.resolution[1:])
        grid = create_cubed_sphere(n)
        sigma = standard_hybrid_levels(nlev)
        hd = _hyperdiff_cube(n)
        dd = _div_damp_cube(n)
        ah = _laplacian_visc_cube(n)
        # FV3_3D iter 33/34: scale A_h via env var.  matrix default
        # is INSUFFICIENT at C72+ (iter 33 found C72 NaN at default
        # A_h but stable at 10x).  Default 1.0 preserves iter-17/24
        # C36/C48 behaviour; set LEGOESM_AH_SCALE=10.0 at C72.
        _ah_scale = float(os.environ.get("LEGOESM_AH_SCALE", "1.0"))
        ah = ah * _ah_scale
        # FV3_3D iter 41: print a stability tip at C72+ when the
        # user has NOT set LEGOESM_AH_SCALE — the iter-33 finding
        # showed default A_h NaNs at C72 (~13 days into the 30-day
        # HS spin-up).  Recommend ah_scale=2.0 at C48, 10.0 at C72.
        if n >= 72 and _ah_scale < 5.0:
            print(
                f"[FV3_3D iter 41 stability tip] At C{n} the matrix "
                f"default A_h={ah:.2e} is INSUFFICIENT and the run "
                f"will likely NaN around day 13.  Recommended: set "
                f"LEGOESM_AH_SCALE=10.0 (or use _laplacian_visc_cube_v2(n) "
                f"directly) per FV3_3D.md iter 33-39.",
                flush=True,
            )
        dt = 300.0
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=hd, hyperdiff_ps_coeff=hd,
            div_damp_coeff=dd, A_h=ah,
            use_conservation_fixer=True, fix_mass=True)
        model = PrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init(grid, sigma, T_init=280.0)

        # iter-22: route ``--radiation`` flag through.  Previously
        # AMIP ignored it and always used HS forcing.  When
        # ``--radiation rrtmgp`` is set the user gets actual
        # RRTMGP radiation (which itself uses the GHG / aerosol /
        # ozone defaults baked into ``forcing/external.py``); the
        # canonical AMIP forcing wiring (CMIP6 input4MIPs realistic
        # GHG/aerosol/ozone) is a follow-up that needs the
        # ``forcing/external.py`` loaders threaded into RadiationConfig.
        physics_fn = (_make_rrtmgp_physics("hydrostatic", dt, hs_fn=held_suarez_forcing)
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
                "mean_T":   _area_weighted_mean(s.T.data,   grid.area),
                "mean_p_s": _area_weighted_mean(s.p_s.data, grid.area),
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
        from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
            hydrostatic_to_cgrid)
        from legoesm.atmosphere.held_suarez import (
            held_suarez_init_latlon, held_suarez_forcing_latlon)

        n_lat, n_lon = (int(x) for x in tc.resolution.split("x"))
        grid = create_latlon_grid(n_lat, n_lon)
        sigma = standard_hybrid_levels(nlev)
        ah = _laplacian_visc_latlon(n_lat)
        import math as _m
        _dx_pole = float(grid.radius) * grid.dlon * _m.cos(
            _m.pi / 2 - grid.dlat / 2)
        dt = min(300.0, 0.5 * _dx_pole / 300.0)
        _A_h_max = 0.4 * _dx_pole**2 / dt
        ah = min(ah, _A_h_max)
        config = CGridLatLonPrimitiveEquationConfig(A_h=ah, fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
        state_cc = held_suarez_init_latlon(grid, sigma, T_init=280.0)
        state = hydrostatic_to_cgrid(state_cc, grid)

        # iter-22: route --radiation flag (was: always HS forcing).
        physics_fn = (_make_rrtmgp_physics("hydrostatic", dt, hs_fn=held_suarez_forcing_latlon)
                      if radiation == "rrtmgp" else held_suarez_forcing_latlon)

        def step_fn(s, dt_):
            return model.step_with_physics(s, dt_, physics_fn)

        mass_fn = lambda s: float(jnp.sum(s.p_s * grid.area))

        def check_fn(s):
            return (check_finite({"T": s.T, "u": s.u}),
                    float(jnp.max(jnp.abs(s.u))))

        def scalar_fn(s):
            u_c = 0.5 * (s.u[:, :-1, :] + s.u[:, 1:, :])
            v_c = 0.5 * (s.v[:-1, :, :] + s.v[1:, :, :])
            return {
                "mass": mass_fn(s),
                "max_wind": float(jnp.max(jnp.sqrt(u_c ** 2 + v_c ** 2))),
                "mean_T":   _area_weighted_mean(s.T,   grid.area),
                "mean_p_s": _area_weighted_mean(s.p_s, grid.area),
            }

        def extract_fn(s):
            u_sfc = np.asarray(0.5 * (s.u[:, :-1, -1] + s.u[:, 1:, -1]), dtype=np.float64)
            v_sfc = np.asarray(0.5 * (s.v[:-1, :, -1] + s.v[1:, :, -1]), dtype=np.float64)
            return {
                "u": u_sfc, "v": v_sfc,
                "wind_speed": np.sqrt(u_sfc ** 2 + v_sfc ** 2),
                "p_s": np.asarray(s.p_s, dtype=np.float64),
                "T_3d": np.asarray(s.T, dtype=np.float64),
            }

        key_array_fn = lambda s: s.T
        coord_kind = "latlon"
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif tc.grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.grids.vertical import standard_hybrid_levels
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
        from legoesm.atmosphere.held_suarez import (
            held_suarez_forcing_mpas, held_suarez_init_mpas)
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)
        sigma = standard_hybrid_levels(nlev)
        dt = 200.0
        ah = _laplacian_visc_ico(mesh)
        config = MPASPrimitiveEquationConfig(
            nu_del4=_hyperdiff_ico(mesh), nu_del2=ah, fix_mass=True)
        model = MPASPrimitiveEquationModel(mesh, sigma, config)
        state = held_suarez_init_mpas(mesh, sigma, T_init=280.0)
        grid = mesh

        # iter-22: route --radiation flag (was: always HS forcing).
        physics_fn_mpas = (_make_rrtmgp_physics("mpas", dt, hs_fn=held_suarez_forcing_mpas)
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
                "mean_T":   _area_weighted_mean(s.T.data,   mesh.areaCell),
                "mean_p_s": _area_weighted_mean(s.p_s.data, mesh.areaCell),
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
        from legoesm.atmosphere.held_suarez import (
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

        # iter-22: route --radiation flag (was: always HS forcing).
        physics_fn = (_make_rrtmgp_physics("spectral_pe", dt, hs_fn=held_suarez_forcing_spectral)
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
                # See iter-5 H1: ``mass`` is the integral (Pa·m²),
                # ``mean_p_s`` is the area-weighted mean (Pa).
                "mass":     float(jnp.sum(fields['p_s'] * grid.grid_area)),
                "max_wind": float(jnp.max(jnp.sqrt(
                    fields['u'] ** 2 + fields['v'] ** 2))),
                "mean_T":   _area_weighted_mean(fields['T'],   grid.grid_area),
                "mean_p_s": _area_weighted_mean(fields['p_s'], grid.grid_area),
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
    # iter-118 (codex iter-117 followup MEDIUM-3): apply
    # a 1e-2 mass-drift tolerance to AMIP.  Same rationale
    # as iter-117 HS / iter-118 baroclinic.  iter-120 also
    # gates on n_samples >= 2.
    ok, notes = _apply_mass_drift_tolerance(
        ok, notes, mass_drift, 1e-2,
        n_samples=len(diag.get("mass", [])))

    level_values = np.asarray(
        getattr(sigma, "sigma_full", np.arange(nlev)), dtype=np.float64)

    # iter-32 codex LOW + iter-35 factored: record effective GHG
    # concentrations in ``results.txt`` for reproducibility when
    # RRTMGP is active.  ``_augment_with_rrtmgp_overrides`` is the
    # shared helper now also used by ``run_held_suarez``.
    _write_results_txt(output_dir, _augment_with_rrtmgp_overrides({
        "test": tc.case, "grid": tc.grid_type, "resolution": tc.resolution,
        "radiation": radiation, "days": days, "dt": dt,
        "status": "PASS" if ok else "FAIL", "notes": notes,
        "wall_time": f"{wall:.1f}s",
    }, radiation),
        diag=diag)  # iter-99: surface BLOWUP info if any
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
            "mass": "Pa*m^2", "max_wind": "m/s", "mean_T": "K",
            "mean_p_s": "Pa"})

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Kessler microphysics adapter for Non-Hydrostatic solver
# ===========================================================================


def _make_kessler_nh_physics_fn(height_coord, dt_phys, tendencies_cls):
    """Return a physics_fn suitable for the NH solver that wraps Kessler.

    DCMIP TC3 (squall line) requires warm-rain microphysics to bound the
    convective instability from the warm bubble perturbations.

    Parameters
    ----------
    height_coord : HeightCoordinate
        Vertical coordinate with reference profiles.
    dt_phys : float
        Outer time step [s] used for Kessler saturation adjustment rate.
    tendencies_cls : type
        NonHydrostaticTendencies or MPASNonHydrostaticTendencies.
    """
    from legoesm.atmosphere.dynamics.compressible_euler import (
        compute_exner_perturbation,
    )
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from legoesm.core.field import Field
    from legoesm import constants as _c

    p_ref = _c.p_ref
    c_pd = _c.c_pd
    R_d = _c.R_d
    L_v = _c.L_v

    theta_ref = height_coord.theta_ref   # (nlev,)
    rho_ref = height_coord.rho_ref       # (nlev,)
    exner_ref = height_coord.exner_ref   # (nlev,)
    dz = height_coord.dz                 # (nlev,)
    nlev = int(dz.shape[0])
    config = KesslerConfig()
    # Cubed-sphere tendencies have dv_dt; MPAS does not
    has_dv = "dv_dt" in tendencies_cls._fields

    def physics_fn(state, grid, hcoord, tmetric):
        # Recover thermodynamic fields from perturbation state
        theta_total = theta_ref + state.theta_prime.data
        rho_total = rho_ref + state.rho_prime.data

        pi_prime = compute_exner_perturbation(
            state.rho_prime.data, state.theta_prime.data, hcoord,
        )
        exner_total = exner_ref + pi_prime
        T = theta_total * exner_total
        p_full = p_ref * exner_total ** (c_pd / R_d)

        # Flatten all spatial dims → (ncol, nlev)
        flat_shape = (-1, nlev)
        T_flat = T.reshape(flat_shape)
        p_flat = p_full.reshape(flat_shape)
        rho_flat = rho_total.reshape(flat_shape)
        ncol = T_flat.shape[0]

        # Tracers: (..., nlev, 3) → separate (ncol, nlev)
        tr = state.tracers.data
        q_v = tr[..., 0].reshape(flat_shape)
        q_c = tr[..., 1].reshape(flat_shape)
        q_r = tr[..., 2].reshape(flat_shape)

        # Half-level pressure (linearly interpolated)
        p_half = jnp.zeros((ncol, nlev + 1))
        p_half = p_half.at[:, 1:-1].set(
            0.5 * (p_flat[:, :-1] + p_flat[:, 1:]),
        )
        p_half = p_half.at[:, -1].set(
            p_flat[:, -1] + 0.5 * (p_flat[:, -1] - p_flat[:, -2]),
        )

        dz_flat = jnp.broadcast_to(dz, (ncol, nlev))
        z_ncol = jnp.zeros((ncol, nlev))

        hydro = HydrometeorState(
            q_c=q_c, q_r=q_r,
            q_i=z_ncol, q_s=z_ncol, q_g=z_ncol,
            N_c=z_ncol, N_r=z_ncol, N_i=z_ncol,
        )

        out = kessler_microphysics(
            T_flat, q_v, hydro, p_flat, p_half, rho_flat,
            dz_flat, dt_phys, config,
        )

        # Convert dT/dt → dtheta_prime/dt ≈ dT/dt / exner_total
        dtheta_dt = out.dT_dt.reshape(T.shape) / jnp.clip(exner_total, 0.5, None)

        # Tracer tendencies
        dtracers = jnp.stack([
            out.dq_v_dt.reshape(T.shape),
            out.dq_c_dt.reshape(T.shape),
            out.dq_r_dt.reshape(T.shape),
        ], axis=-1)

        z3d = jnp.zeros_like(state.theta_prime.data)
        z_w = jnp.zeros_like(state.w.data)
        z2d = jnp.zeros_like(state.phis.data)
        d3 = state.theta_prime.dims
        d_w = state.w.dims
        d2 = state.phis.dims

        fields = dict(
            du_dt=Field(data=jnp.zeros_like(state.u.data),
                        name="du_dt", dims=state.u.dims, units="m/s^2"),
            dw_dt=Field(data=z_w, name="dw_dt", dims=d_w, units="m/s^2"),
            dtheta_prime_dt=Field(
                data=dtheta_dt, name="dtheta_prime_dt", dims=d3, units="K/s",
            ),
            drho_prime_dt=Field(
                data=z3d, name="drho_prime_dt", dims=d3, units="kg/m^3/s",
            ),
            dphis_dt=Field(data=z2d, name="dphis_dt", dims=d2, units="m^2/s^3"),
            dtracers_dt=Field(
                data=dtracers, name="dtracers_dt",
                dims=state.tracers.dims, units="kg/kg/s",
            ),
        )
        if has_dv:
            fields["dv_dt"] = Field(
                data=z3d, name="dv_dt", dims=d3, units="m/s^2",
            )

        return tendencies_cls(**fields)

    return physics_fn


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
            # Scale hyperdiffusion for 20x smaller Earth: coeff ∝ dx⁴.
            # Use a shorter e-folding time (4x stronger diffusion) than
            # the full-Earth default because mountain-generated flow
            # disturbances produce grid-scale noise that the standard
            # 52-hour e-fold cannot damp in a 6-hour simulation.
            hd_tc2 = hd / 20.0 ** 4 * 4.0
            dt = max(0.15, 3.0 * (16.0 / n))
            nh_config = CompressibleEulerConfig(
                n_acoustic_substeps=20, semi_implicit_acoustic=True,
                sponge_width=15000.0,
                sponge_coeff=1.0 / (0.1 * 86400.0),
                hyperdiff_coeff=hd_tc2,
                hyperdiff_w_coeff=hd_tc2,
                acoustic_off_centering=0.15)
        elif test_case == "tc3":
            from tests.test_cases.dcmip2025 import dcmip25_tc3_init
            state, hcoord, tmetric, small_grid = dcmip25_tc3_init(
                grid, n_levels=nlev)
            grid = small_grid
            # Scale hyperdiffusion for 60x smaller Earth: coeff ∝ dx⁴.
            # 8x stronger than default scaling to stabilize the
            # convective dynamics in the squall line.
            hd_tc3 = hd / 60.0 ** 4 * 8.0
            dt = max(0.05, 0.5 * (16.0 / n))
            nh_config = CompressibleEulerConfig(
                n_acoustic_substeps=25, semi_implicit_acoustic=True,
                sponge_width=12000.0, sponge_coeff=0.3,
                hyperdiff_coeff=hd_tc3,
                hyperdiff_w_coeff=hd_tc3,
                acoustic_off_centering=0.2)
        else:
            raise ValueError(f"Unknown NH test case: {test_case}")

        model = CompressibleEulerModel(grid, hcoord, tmetric, nh_config)

        # TC3 requires Kessler warm-rain microphysics to bound the
        # buoyancy-driven convective instability from the warm bubbles.
        if test_case == "tc3":
            from legoesm.core.state import NonHydrostaticTendencies
            _kessler_fn = _make_kessler_nh_physics_fn(
                hcoord, dt, NonHydrostaticTendencies)

            def step_fn(s, dt_):
                return model.step_with_physics(s, dt_, physics_fn=_kessler_fn)
        else:
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
                "mean_theta_prime": _area_weighted_mean(
                    s.theta_prime.data, grid.area),
                "mean_rho_prime":   _area_weighted_mean(
                    s.rho_prime.data,   grid.area),
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
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.compressible_euler_mpas import (
            MPASCompressibleEulerModel, MPASCompressibleEulerConfig)

        level = int(tc.resolution.replace("ico", ""))
        mesh = create_voronoi_mesh(level)

        if test_case == "tc1":
            from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas import (
                dcmip25_tc1_init_mpas)
            state, hcoord, tmetric = dcmip25_tc1_init_mpas(
                mesh, n_levels=nlev)
        elif test_case == "tc2a":
            from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_2_mpas import (
                dcmip25_tc2_init_mpas)
            state, hcoord, tmetric, mesh = dcmip25_tc2_init_mpas(
                mesh, n_levels=nlev, subcase="a")
        elif test_case == "tc3":
            from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_3_mpas import (
                dcmip25_tc3_init_mpas)
            state, hcoord, tmetric, mesh = dcmip25_tc3_init_mpas(
                mesh, n_levels=nlev)
        else:
            record(tc, "SKIP", 0.0, f"MPAS NH: unsupported test case {test_case}")
            return "SKIP", 0.0, ""
        grid = mesh

        dx_mean = float(jnp.sqrt(
            4.0 * jnp.pi * mesh.radius ** 2 / mesh.nCells))
        _dt_scale = 6.0
        _dt_cap = 0.5 if test_case == "tc3" else 6.0
        dt = min(max(0.2, _dt_scale * (200.0 / (dx_mean / 1000.0))), _dt_cap)
        # Sponge config per test case
        if test_case == "tc2a":
            _sponge_w, _sponge_c = 15000.0, 1.0 / (0.1 * 86400.0)
        elif test_case == "tc3":
            _sponge_w, _sponge_c = 12000.0, 0.3
        else:
            _sponge_w, _sponge_c = 10000.0, 0.05
        # TC2/TC3 need more acoustic substeps and stronger diffusion
        # than TC1 due to mountain-generated and convective instabilities.
        if test_case == "tc2a":
            _n_acoustic = 20
            _nu_del4 = dx_mean ** 4 / (12.0 * 3600.0)  # 4x stronger
        elif test_case == "tc3":
            _n_acoustic = 25
            _nu_del4 = dx_mean ** 4 / (6.0 * 3600.0)   # 8x stronger
        else:
            _n_acoustic = 10
            _nu_del4 = dx_mean ** 4 / (48.0 * 3600.0)
        nh_config = MPASCompressibleEulerConfig(
            n_acoustic_substeps=_n_acoustic, sponge_width=_sponge_w,
            sponge_coeff=_sponge_c,
            nu_del4=_nu_del4)
        model = MPASCompressibleEulerModel(
            mesh, hcoord, tmetric, nh_config)

        lon_cell = np.asarray(mesh.lonCell, dtype=np.float64) * 180 / np.pi
        lat_cell = np.asarray(mesh.latCell, dtype=np.float64) * 180 / np.pi
        lon_edge = np.asarray(mesh.lonEdge, dtype=np.float64) * 180 / np.pi
        lat_edge = np.asarray(mesh.latEdge, dtype=np.float64) * 180 / np.pi

        if test_case == "tc3":
            from legoesm.core.state import MPASNonHydrostaticTendencies
            _kessler_fn = _make_kessler_nh_physics_fn(
                hcoord, dt, MPASNonHydrostaticTendencies)

            def step_fn(s, dt_):
                return model.step(s, dt_, physics_fn=_kessler_fn)
        else:
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
                "mean_theta_prime": _area_weighted_mean(
                    s.theta_prime.data, mesh.areaCell),
                "mean_rho_prime":   _area_weighted_mean(
                    s.rho_prime.data,   mesh.areaCell),
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
        # TC3 requires Kessler microphysics; the spectral solver does not
        # yet support a physics_fn callback (state is in spectral space).
        if test_case == "tc3":
            record(tc, "SKIP", 0.0,
                   "Spectral NH TC3 requires Kessler microphysics; "
                   "physics_fn not yet wired for spectral solver")
            return "SKIP", 0.0, ""

        from legoesm.grids.gaussian import (
            create_gaussian_grid, sh_synthesis_3d,
        )
        from legoesm.atmosphere.dynamics.spectral_nh import (
            SpectralCompressibleEulerModel, SpectralNHConfig,
            dcmip25_tc1_init_spectral,
            dcmip25_tc2_init_spectral,
            dcmip25_tc3_init_spectral,
        )

        n_max = int(tc.resolution.replace("T", ""))
        grid = create_gaussian_grid(n_max)

        if test_case == "tc1":
            state, hcoord, tmetric = dcmip25_tc1_init_spectral(
                grid, n_levels=nlev)
        elif test_case == "tc2a":
            state, hcoord, tmetric = dcmip25_tc2_init_spectral(
                grid, n_levels=nlev, subcase="a")
        elif test_case == "tc3":
            state, hcoord, tmetric = dcmip25_tc3_init_spectral(
                grid, n_levels=nlev)
        else:
            raise NotImplementedError(
                f"Spectral NH: unsupported test case {test_case}")

        # Small-Earth factor and sponge config per test case
        if test_case == "tc2a":
            sef = 20.0
            sponge_w, sponge_c = 15000.0, 1.0 / (0.1 * 86400.0)
        elif test_case == "tc3":
            sef = 60.0
            sponge_w, sponge_c = 8000.0, 0.15
        else:
            sef = 1.0
            sponge_w, sponge_c = 10000.0, 0.05

        _dt_scale_sp = 3.0 if test_case == "tc3" else 6.0
        dt = max(0.25 if test_case == "tc3" else 0.5,
                 _dt_scale_sp * (21.0 / n_max))
        # TC2/TC3 need more acoustic substeps and stronger diffusion.
        _base_hd = 2.338e15 * (21.0 / n_max) ** 4
        if test_case == "tc2a":
            _n_acoustic_sp = 20
            _hd_sp = _base_hd * 4.0
        elif test_case == "tc3":
            _n_acoustic_sp = 25
            _hd_sp = _base_hd * 8.0
        else:
            _n_acoustic_sp = 10
            _hd_sp = _base_hd
        nh_config = SpectralNHConfig(
            n_acoustic_substeps=_n_acoustic_sp,
            semi_implicit_acoustic=True,
            sponge_width=sponge_w,
            sponge_coeff=sponge_c,
            hyperdiff_coeff=_hd_sp,
            small_earth_factor=sef,
        )
        # Use the small-Earth grid for tc2/tc3
        if sef != 1.0:
            from legoesm import constants as _c
            grid = create_gaussian_grid(
                n_max, radius=_c.R_earth / sef)
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
                "mean_theta_prime": _area_weighted_mean(theta_p, grid.grid_area),
                "mean_rho_prime":   _area_weighted_mean(rho_p,   grid.grid_area),
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
        label=f"NH {test_case} ({tc.grid_type})", total_days=days,
        blowup_check_interval=25)

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
        "wall_time": f"{wall:.1f}s"},
        diag=diag)  # iter-99: surface BLOWUP info if any
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
    "williamson6": run_shallow_water,                # M1.a (mpas + spectral)
    "cosine_bell": run_cosine_bell,
    "held_suarez": run_held_suarez,
    "held_suarez_topo": run_held_suarez,             # M1.a (HS over topo)
    "baroclinic": run_baroclinic,
    "rotated_baroclinic": run_baroclinic,            # M1.a (DCMIP 2008 §4-2 rotated)
    "rotated_steady": run_baroclinic,                # M1.a (DCMIP 2008 §4-1 rotated)
    "rest_state_topo": run_baroclinic,               # M1.a (DCMIP 2012 §2-0-0)
    "gravity_wave_3_1":   run_baroclinic,            # M1.a (DCMIP 2008 §3-1)
    "inertio_gravity_3_2": run_baroclinic,           # M1.a (DCMIP 2008 §3-2)
    "mountain_rossby_5_0": run_baroclinic,           # M1.a (DCMIP 2008 §5-0)
    "rossby_haurwitz_6_0": run_baroclinic,           # M1.a (DCMIP 2008 §6-0)
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
# Cross-grid comparison plots
# ===========================================================================
# Per-case field metadata for shared-colorbar / shared-projection comparison
# plots.  ``range`` is ``(vmin, vmax)`` — use ``(None, None)`` to autoscale
# from the data across all grids (recommended for adaptive cases).
# ``cmap`` is the matplotlib colormap; diverging maps for signed wind/v;
# sequential for height / scalar.  ``units`` flows into the colorbar label.

ATMOSPHERE_COMPARISON_FIELDS: dict[str, list[dict]] = {
    # Williamson 2: solid-body steady state — height ~ 2.94e3 m,
    # zonal wind ~38 m/s.  Use adaptive ranges so any drift from the
    # analytic state is visible.
    "williamson2": [
        {"field": "height",     "vmin": None,  "vmax": None,  "cmap": "viridis", "units": "m"},
        {"field": "u",          "vmin": -50,   "vmax": 50,    "cmap": "RdBu_r",  "units": "m/s"},
        {"field": "wind_speed", "vmin": 0,     "vmax": 50,    "cmap": "viridis", "units": "m/s"},
    ],
    # Williamson 5: flow over an isolated mountain — height develops a
    # standing-wave pattern downstream, ~5400 ± 500 m.
    "williamson5": [
        {"field": "height",     "vmin": None,  "vmax": None,  "cmap": "viridis", "units": "m"},
        {"field": "u",          "vmin": -30,   "vmax": 60,    "cmap": "RdBu_r",  "units": "m/s"},
        {"field": "wind_speed", "vmin": 0,     "vmax": 60,    "cmap": "viridis", "units": "m/s"},
    ],
    # Cosine-bell tracer: height field passively advects.
    "cosine_bell": [
        {"field": "height",     "vmin": None,  "vmax": None,  "cmap": "viridis", "units": "m"},
        {"field": "wind_speed", "vmin": 0,     "vmax": 50,    "cmap": "viridis", "units": "m/s"},
    ],
    # Held-Suarez: zonally averaged steady-state climate.  ``T_3d`` is
    # the 3-D temperature field as saved by the runners; the 4-D-array
    # branch in ``_atm_extract_field_2d`` slices the lowest model level
    # (surface T proxy) for the comparison panel.
    "held_suarez": [
        {"field": "T_3d",       "vmin": 230,   "vmax": 310,   "cmap": "plasma",  "units": "K"},
        {"field": "u",          "vmin": -40,   "vmax": 60,    "cmap": "RdBu_r",  "units": "m/s"},
        {"field": "p_s",        "vmin": 95000, "vmax": 105000,"cmap": "viridis", "units": "Pa"},
        {"field": "wind_speed", "vmin": 0,     "vmax": 60,    "cmap": "viridis", "units": "m/s"},
    ],
    "baroclinic": [
        {"field": "T_3d",       "vmin": 220,   "vmax": 310,   "cmap": "plasma",  "units": "K"},
        {"field": "u",          "vmin": -40,   "vmax": 80,    "cmap": "RdBu_r",  "units": "m/s"},
        {"field": "p_s",        "vmin": 95000, "vmax": 105000,"cmap": "viridis", "units": "Pa"},
    ],
    "amip": [
        # AMIP runners save ``T_3d`` (4-D) and ``p_s`` / ``u`` /
        # ``wind_speed`` (3-D).  The plotter slices the lowest level
        # of T_3d as a surface-T proxy.  ``T_sfc`` and ``precip`` are
        # NOT currently saved by any AMIP extractor; iter-5 codex
        # review M2 caught my iter-1 mistake of advertising fields
        # the runners don't produce.  Restore those panels once the
        # AMIP extractors emit them.
        {"field": "T_3d",       "vmin": 220,   "vmax": 305,   "cmap": "plasma",  "units": "K"},
        {"field": "p_s",        "vmin": 95000, "vmax": 105000,"cmap": "viridis", "units": "Pa"},
        {"field": "wind_speed", "vmin": 0,     "vmax": 60,    "cmap": "viridis", "units": "m/s"},
    ],
    # DCMIP transport tests save tracer fields (``q1``..``q4`` 2-D and
    # ``q1_3d`` 3-D).  Solid-body / divergent-flow advection should
    # preserve the tracer pattern across grids; cross-grid panels
    # expose grid-specific dispersion / monotonicity differences.
    # iter-5 codex review M3.
    "dcmip_transport_11": [
        {"field": "q1",         "vmin": 0,     "vmax": 1.1,   "cmap": "viridis", "units": "kg/kg"},
        {"field": "q2",         "vmin": 0,     "vmax": 1.1,   "cmap": "viridis", "units": "kg/kg"},
    ],
    "dcmip_transport_12": [
        {"field": "q1",         "vmin": 0,     "vmax": 1.1,   "cmap": "viridis", "units": "kg/kg"},
    ],
    "dcmip_transport_13": [
        {"field": "q1",         "vmin": 0,     "vmax": 1.1,   "cmap": "viridis", "units": "kg/kg"},
        {"field": "q2",         "vmin": 0,     "vmax": 1.1,   "cmap": "viridis", "units": "kg/kg"},
    ],
}


# Iter-8: per-case 3-D fields whose zonal-mean cross-section
# ``(latitude, sigma)`` is the canonical inter-model comparison
# (e.g. Held & Suarez 1994 Fig. 3-4).  Each entry mirrors the
# ATMOSPHERE_COMPARISON_FIELDS schema but the ``field`` key MUST
# resolve to a 4-D ``(n_times, lat, lon, nlev)`` array in
# ``snapshots_latlon.npz`` so the plotter can compute
# ``mean(field[t_final], axis=lon-axis)``.
ATMOSPHERE_ZONAL_MEAN_FIELDS: dict[str, list[dict]] = {
    "held_suarez": [
        {"field": "T_3d", "vmin": 200,   "vmax": 310,  "cmap": "plasma",
         "units": "K",   "longname": "Zonal-mean temperature"},
    ],
    "baroclinic": [
        {"field": "T_3d", "vmin": 220,   "vmax": 310,  "cmap": "plasma",
         "units": "K",   "longname": "Zonal-mean temperature"},
    ],
    "amip": [
        {"field": "T_3d", "vmin": 200,   "vmax": 305,  "cmap": "plasma",
         "units": "K",   "longname": "Zonal-mean temperature"},
    ],
}


def _atmosphere_grid_color() -> dict[str, str]:
    return {
        "cubed_sphere": "tab:blue",
        "latlon":       "tab:red",
        "icosahedral":  "tab:green",
        "spectral":     "tab:orange",
    }


_RES_DIR_WARNED: set[Path] = set()


class _EmptyNpzShim:
    """Mimics ``numpy.lib.npyio.NpzFile`` with no fields.  Used by
    ``_collect_grid_results_atmosphere`` (iter-26) when a grid has a
    valid timeseries but no snapshot NPZ — snapshot-based plots see
    no fields and cleanly no-op while the timeseries plot still runs.
    """
    files: list[str] = []
    def __getitem__(self, key):
        raise KeyError(f"empty snapshot stub does not contain {key!r}")
    def __contains__(self, key):
        return False


_EMPTY_SNAPSHOTS_STUB = _EmptyNpzShim()


def _select_resolution_dir(grid_dir: Path) -> Path | None:
    """Pick the (single) resolution subdirectory under ``grid_dir``.

    When more than one resolution directory exists (stale + fresh
    output mixed in the same tree), prefer the directory whose
    name matches the per-grid format from iter-95 (``C*`` for
    cubed_sphere, ``*x*`` for latlon, ``ico*`` for icosahedral,
    ``T*`` for spectral).  Only fall back to the
    alphabetically-first dir if no grid-typed candidate exists
    (graceful degradation for pre-iter-95 output trees).

    iter-108 (codex iter-104 MEDIUM-6): the original
    ``sorted(res_dirs)[0]`` rule picked alphabetically-first,
    so a stale ``16/`` dir would win over a fresh ``C16/`` dir
    after iter-95's per-grid dispatch fix.  Now prefer the
    grid-typed name explicitly.

    Centralised so ``_collect_grid_results_atmosphere`` and
    ``_vertical_coords_for_case`` use identical selection logic
    and don't emit duplicate warnings.
    """
    # iter-110 codex MEDIUM-3: filter hidden / internal /
    # tooling directories so collector fallback can't pick
    # ``.ipynb_checkpoints`` or ``__pycache__`` over a valid
    # legacy ``16/`` dir.
    _BAD_DIRNAMES = {"__pycache__", ".ipynb_checkpoints"}
    res_dirs = [
        d for d in grid_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and d.name not in _BAD_DIRNAMES
    ]
    if not res_dirs:
        return None

    # iter-108: prefer grid-typed format over bare numeric.
    grid_name = grid_dir.name
    grid_typed_pattern = {
        "cubed_sphere": lambda n: n.startswith("C") and n[1:].isdigit(),
        "latlon": lambda n: "x" in n and all(p.isdigit() for p in n.split("x") if p),
        "icosahedral": lambda n: n.startswith("ico") and n[3:].isdigit(),
        "spectral": lambda n: n.startswith("T") and n[1:].isdigit(),
        # Ocean grid types
        "mpas": lambda n: n.startswith("ico") and n[3:].isdigit(),
        "mpas_regional": lambda n: n.endswith("km") and n[:-2].isdigit(),
        "latlon_regional": lambda n: n.endswith("km") and n[:-2].isdigit(),
        "cs_regional": lambda n: n.endswith("km") and n[:-2].isdigit(),
    }
    matcher = grid_typed_pattern.get(grid_name)

    if matcher is not None:
        typed = [d for d in res_dirs if matcher(d.name)]
        if typed:
            chosen = sorted(typed)[0]
            # If we filtered out stale dirs, warn once.
            stale = [d.name for d in res_dirs if d not in typed]
            if stale and grid_dir not in _RES_DIR_WARNED:
                _RES_DIR_WARNED.add(grid_dir)
                print(
                    f"    [comparison] {grid_name}: ignoring "
                    f"non-grid-typed resolution dirs {sorted(stale)} "
                    f"in favor of {chosen.name} (iter-108 prefers "
                    f"the grid-typed format from iter-95 "
                    f"dispatch).  Re-run with a clean output tree "
                    f"to remove stale dirs."
                )
            return chosen
    # iter-112 codex LOW-4: fallback prefers bare-numeric
    # dirs (legacy pre-iter-95 form) over arbitrary names.
    bare_numeric = [d for d in res_dirs if d.name.isdigit()]
    if bare_numeric:
        if len(res_dirs) > 1 and grid_dir not in _RES_DIR_WARNED:
            _RES_DIR_WARNED.add(grid_dir)
            print(
                f"    [comparison] {grid_dir.name}: multiple "
                f"resolution dirs found "
                f"({sorted(d.name for d in res_dirs)}); using "
                f"{sorted(bare_numeric)[0].name} — re-run with "
                f"a clean output tree if a different one is "
                f"intended."
            )
        return sorted(bare_numeric)[0]
    # Final fallback: alphabetically-first.
    if len(res_dirs) > 1 and grid_dir not in _RES_DIR_WARNED:
        _RES_DIR_WARNED.add(grid_dir)
        print(
            f"    [comparison] {grid_dir.name}: multiple resolution "
            f"dirs found ({sorted(d.name for d in res_dirs)}); using "
            f"{sorted(res_dirs)[0].name} — re-run with a clean output "
            f"tree if a different one is intended."
        )
    return sorted(res_dirs)[0]


def _collect_grid_results_atmosphere(
    test_case_dir: Path,
    vertical_coord: str | None = None,
    *,
    allowed_grids: set[str] | None = None,
) -> dict[str, dict]:
    """Walk ``<grid>/<resolution>/[<vertical>/]`` under ``test_case_dir`` and
    load each grid's outputs.

    Atmosphere layout has an optional ``<vertical_coord>`` level beyond
    ``<grid>/<resolution>/`` (e.g. ``hydrostatic/held_suarez/cubed_sphere/C36/sigma/``).
    For shallow-water cases there is no vertical coord and the layout is
    ``<grid>/<resolution>/`` only.

    When ``vertical_coord`` is given, only the matching subdir under each
    grid's resolution directory is collected.  Grids that lack that
    specific vertical-coord subdir are skipped (allows separate
    sigma vs. hybrid comparisons for hydrostatic runs).

    Returns
    -------
    dict mapping ``grid_type`` to ``{timeseries, snapshots, metadata, resolution}``.
    Grids whose required artifacts are missing or unreadable are skipped.
    """
    import pandas as pd

    grid_results: dict[str, dict] = {}
    if not test_case_dir.exists():
        return grid_results

    for grid_dir in sorted(test_case_dir.iterdir()):
        if not grid_dir.is_dir() or grid_dir.name.startswith("."):
            continue
        if grid_dir.name not in GRID_TYPES:
            continue
        if allowed_grids is not None and grid_dir.name not in allowed_grids:
            continue
        # iter-6 L1: centralised resolution selection (warns once
        # per grid_dir on multi-resolution trees).
        res_dir = _select_resolution_dir(grid_dir)
        if res_dir is None:
            continue

        # Determine the leaf directory.  When vertical_coord is given,
        # it MUST exist as a subdir; otherwise skip this grid.  When not
        # given, accept the resolution directory directly (SW case);
        # if the resolution dir itself has subdirs (hydro case) but no
        # ``vertical_coord`` filter was specified, fall through and use
        # the resolution dir (no leaf-files there → grid is skipped).
        if vertical_coord is not None:
            leaf = res_dir / vertical_coord
            if not leaf.is_dir():
                continue
        else:
            leaf = res_dir

        csv_file = leaf / "mean_timeseries.csv"
        npz_file = leaf / "snapshots_latlon.npz"
        results_file = leaf / "results.txt"
        # iter-26 codex review HIGH: relax the all-three requirement.
        # Cross-grid TIMESERIES + SUMMARY plots only need
        # ``mean_timeseries.csv`` + ``results.txt``.  The snapshot
        # NPZ is only needed for the snapshot panels and zonal-mean
        # plot.  Allow runs without snapshots (e.g. RCE iter-24
        # output) to participate in TIMESERIES comparisons by
        # providing an empty-snapshot stub when the NPZ is missing.
        if not (csv_file.exists() and results_file.exists()):
            continue

        try:
            timeseries_df = pd.read_csv(csv_file)
            if npz_file.exists():
                snapshots = np.load(npz_file)
            else:
                # Empty stub so downstream snapshot-only plotters
                # (which iterate ``snapshots.files``) cleanly skip.
                snapshots = _EMPTY_SNAPSHOTS_STUB
            metadata: dict[str, str] = {}
            with open(results_file, "r") as fh:
                for line in fh:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip()] = v.strip()
            grid_results[grid_dir.name] = {
                "timeseries": timeseries_df,
                "snapshots":  snapshots,
                "metadata":   metadata,
                "resolution": res_dir.name,
                "vertical_coord": vertical_coord or "",
            }
        except Exception as e:  # noqa: BLE001 — best-effort post-hoc collection
            print(f"    [comparison] skipping {grid_dir.name}: {e}")
            continue

    return grid_results


def _vertical_coords_for_case(
    test_case_dir: Path, *, allowed_grids: set[str] | None = None,
) -> list[str | None]:
    """Enumerate the vertical-coord variants present across grids for a
    given case directory.

    Returns ``[None]`` for shallow-water-style layouts where each grid
    has the leaf files directly under ``<grid>/<resolution>/``.
    Returns the union of vertical-coord subdirectory names (e.g.
    ``["sigma", "hybrid"]``) for hydrostatic-style layouts.  An empty
    list is returned only if no recognised grid layout exists.
    """
    if not test_case_dir.exists():
        return []
    found_sw = False
    verts: set[str] = set()
    for grid_dir in test_case_dir.iterdir():
        if not grid_dir.is_dir() or grid_dir.name not in GRID_TYPES:
            continue
        if allowed_grids is not None and grid_dir.name not in allowed_grids:
            continue
        # iter-6 L1: shared resolution-selection helper, identical
        # warnings as the collector (deduplicated via _RES_DIR_WARNED).
        res_dir = _select_resolution_dir(grid_dir)
        if res_dir is None:
            continue
        # Detect SW-style (leaf files at resolution level) vs. hydro-style
        # (vertical-coord subdir level).  iter-26 codex review HIGH:
        # accept timeseries-only runs (e.g. RCE iter-24 output without
        # snapshots) — but match the COLLECTION criteria
        # (``mean_timeseries.csv`` AND ``results.txt``, OR
        # ``snapshots_latlon.npz``) so a stale CSV without
        # ``results.txt`` doesn't produce a phantom no-op combo.
        # iter-27 codex review LOW.  iter-43 codex MEDIUM: predicate
        # lifted to ``has_collectable_atmosphere_outputs`` at module
        # scope so the iter-42 converter and its tests bind to a
        # single source of truth.
        if has_collectable_atmosphere_outputs(res_dir):
            found_sw = True
        for sub in res_dir.iterdir():
            if sub.is_dir() and has_collectable_atmosphere_outputs(sub):
                verts.add(sub.name)
    if verts and found_sw:
        # Mixed layout: SOME grids have leaf files at the resolution
        # level (SW-style) and others have a vertical-coord subdir
        # level (hydro-style).  Almost always indicates stale output
        # mixed with a re-run.  iter-5 codex review L1: warn the user
        # rather than silently emitting two comparison sets.
        print(
            f"  [comparison] {test_case_dir.name}: mixed SW-style and "
            f"vertical-coord layouts detected — emitting BOTH SW-style "
            f"and per-vertical-coord comparisons.  This usually means "
            f"the output tree contains stale data; consider clearing "
            f"the case directory before re-running."
        )
        return [None] + sorted(verts)
    if verts:
        return sorted(verts)
    if found_sw:
        return [None]
    return []


def _atm_extract_field_2d(snapshots: np.lib.npyio.NpzFile, field: str) -> np.ndarray | None:
    """Pull the final-timestep 2-D lat-lon slice of ``field``.

    Handles 2-D (lat, lon), 3-D (n_times, lat, lon), and
    4-D (n_times, lat, lon, nlev) — for 4-D, takes the lowest model
    level (index -1, surface).  Returns ``None`` if the field is absent
    or the shape is unrecognised.
    """
    if field not in snapshots.files:
        return None
    arr = np.asarray(snapshots[field])
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[-1]
    if arr.ndim == 4:
        return arr[-1, :, :, -1]
    return None


def _create_atmosphere_comparison_snapshots(
    test_case_dir: Path, grid_results: dict, fields: list[dict],
    *, label: str | None = None,
) -> None:
    """4-panel snapshot comparison across grids using a SHARED PlateCarrée
    projection and SHARED colorbar per field.

    Output PNGs are written into ``test_case_dir`` (caller is
    responsible for choosing the right output directory — see
    ``_create_cross_grid_comparisons_atmosphere``).  Pass ``label`` to
    override the default suptitle text (which is ``test_case_dir.name``);
    callers should set ``label`` to e.g. ``"held_suarez/sigma"`` so
    sigma vs hybrid comparisons are visually distinguishable.

    One PNG is written per field: ``comparison_snapshots_<field>.png``.
    Missing-grid panels are blanked but the layout slot is preserved so
    the colorbar alignment stays consistent across runs.
    """
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        have_cartopy = True
    except Exception:
        have_cartopy = False

    case_name = label if label is not None else test_case_dir.name

    for spec in fields:
        field = spec["field"]
        vmin = spec["vmin"]
        vmax = spec["vmax"]
        cmap = spec["cmap"]
        units = spec.get("units", "")

        fields_2d: dict[str, np.ndarray] = {}
        for grid_name, data in grid_results.items():
            f2 = _atm_extract_field_2d(data["snapshots"], field)
            if f2 is not None:
                fields_2d[grid_name] = f2
        if not fields_2d:
            continue

        # Auto-range from data across grids if vmin/vmax is None.
        if vmin is None or vmax is None:
            stacked = np.concatenate([f.ravel() for f in fields_2d.values()])
            finite = stacked[np.isfinite(stacked)]
            if finite.size > 0:
                if vmin is None:
                    vmin = float(np.nanmin(finite))
                if vmax is None:
                    vmax = float(np.nanmax(finite))

        # Layout: 2 columns × ⌈n/2⌉ rows.  Always allocate 4 slots so the
        # colorbar geometry is identical across runs.
        nrows, ncols = 2, 2
        if have_cartopy:
            proj = ccrs.PlateCarree()
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(13, 6.5),
                subplot_kw={"projection": proj},
            )
        else:
            fig, axes = plt.subplots(nrows, ncols, figsize=(13, 6.5))

        # Get final time across grids for the title (use first grid's metadata).
        sim_time_str = ""
        for data in grid_results.values():
            snap = data["snapshots"]
            if "times_days" in snap.files:
                sim_time_str = f" (t = {float(snap['times_days'][-1]):.2f} d)"
                break

        fig.suptitle(
            f"{case_name} — {field} cross-grid comparison{sim_time_str}",
            fontsize=13, fontweight="bold",
        )

        slot_order = [g for g in GRID_TYPES if g in grid_results]
        # Append any extra grid names (e.g. regional variants) at the end.
        slot_order += [g for g in grid_results if g not in slot_order]

        im = None
        for slot, ax in zip(range(nrows * ncols), axes.flat):
            if slot >= len(slot_order):
                ax.set_visible(False)
                continue
            grid_name = slot_order[slot]
            data = grid_results[grid_name]
            if grid_name not in fields_2d:
                ax.set_title(f"{grid_name} ({data['resolution']}) — {field} N/A")
                ax.set_visible(False)
                continue
            f2 = fields_2d[grid_name]
            # The snapshot file may carry canonical 181x360 lat/lon
            # metadata even when the data array was saved at native
            # resolution.  Always derive the plotting grid from the
            # actual array shape to keep dimensions consistent.
            n_lat, n_lon = f2.shape
            md_lat = np.asarray(data["snapshots"]["lat"])
            md_lon = np.asarray(data["snapshots"]["lon"])
            if md_lat.size == n_lat:
                lat = md_lat
            else:
                lat = np.linspace(-90.0, 90.0, n_lat)
            if md_lon.size == n_lon:
                lon = md_lon
            else:
                lon = np.linspace(-180.0, 180.0, n_lon)
            if have_cartopy:
                im = ax.pcolormesh(
                    lon, lat, f2, cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree(), shading="auto",
                )
                ax.set_global()
                ax.coastlines(linewidth=0.4, color="black", alpha=0.6)
                ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
            else:
                im = ax.imshow(
                    f2, origin="lower", aspect="auto",
                    extent=[float(lon.min()), float(lon.max()),
                            float(lat.min()), float(lat.max())],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
            ax.set_title(f"{grid_name} ({data['resolution']})")

        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label(f"{field}" + (f" ({units})" if units else ""))

        from datetime import datetime
        fig.text(0.99, 0.01,
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ha="right", va="bottom", fontsize=7, color="gray")

        # Cartopy GeoAxes are incompatible with ``tight_layout``; use
        # explicit margin control instead to silence the UserWarning.
        fig.subplots_adjust(left=0.04, right=0.9, top=0.92, bottom=0.05,
                            wspace=0.05, hspace=0.15)

        out_file = test_case_dir / f"comparison_snapshots_{field}.png"
        plt.savefig(out_file, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_file.name}")


# Iter-10: fraction of trailing snapshots to average for zonal-mean
# climatology cross-sections.  ``0.5`` averages the last 50 % of the
# snapshots — for an 11-snapshot HS run that's
# ``ceil(11 × 0.5) = 6`` snapshots (days ~15-30 of a 30-day quick
# run).  Iter-16 codex review LOW: this is a variance/transient
# COMPROMISE — it does NOT cleanly exclude the early spin-up; it
# trades samples-for-noise against samples-for-transient-bias.  The
# canonical HS climatology averages 1000+ days after a 200-day
# spin-up; we don't reach that here, but a 50 % trailing window
# already substantially reduces cross-grid RMS vs. a single-
# snapshot diagnostic.  If a tighter window is needed, expose
# ``--climatology-fraction`` as a CLI knob in a future iteration.
CLIMATOLOGY_AVG_FRACTION = 0.5


def _n_climatology_avg(n_times: int) -> int:
    """Number of trailing snapshots to average for a climatology mean."""
    return max(1, int(np.ceil(n_times * CLIMATOLOGY_AVG_FRACTION)))


def _zonal_mean_at_final_time(arr_4d: np.ndarray) -> np.ndarray:
    """Reduce a 4-D ``(n_times, n_lat, n_lon, n_lev)`` snapshot array to
    the zonal-mean cross-section at the final time, returning a 2-D
    array shaped ``(n_lat, n_lev)``.

    Used by the zonal-mean cross-grid comparison.  Iter-8.
    """
    if arr_4d.ndim != 4:
        raise ValueError(
            f"_zonal_mean_at_final_time: expected 4-D (n_times, n_lat, "
            f"n_lon, n_lev), got shape {arr_4d.shape}"
        )
    return np.nanmean(arr_4d[-1], axis=1)  # collapse longitude axis


def _zonal_mean_climatology(
    arr_4d: np.ndarray, n_avg: int = 1,
) -> np.ndarray:
    """Reduce a 4-D ``(n_times, n_lat, n_lon, n_lev)`` snapshot array to
    a TIME-MEAN zonal-mean cross-section, returning a 2-D
    ``(n_lat, n_lev)`` array.

    ``n_avg`` is the number of trailing snapshots to average over.
    For Held-Suarez climatology comparison, this should ideally be
    100s of days of snapshots — but in practice we cap at the
    available number of snapshots.

    Iter-10: addresses the snapshot-vs-climatology distinction
    surfaced when investigating the user's iter-5
    HS-cross-grid-agreement question.  A single-snapshot zonal
    mean carries large sampling variance from the eddy field;
    a multi-snapshot time mean drives that variance toward zero
    and exposes the actual cross-grid CLIMATOLOGY agreement.
    """
    if arr_4d.ndim != 4:
        raise ValueError(
            f"_zonal_mean_climatology: expected 4-D (n_times, n_lat, "
            f"n_lon, n_lev), got shape {arr_4d.shape}"
        )
    n_times = arr_4d.shape[0]
    n_use = max(1, min(int(n_avg), n_times))
    # Average over the last n_use snapshots, then over longitude.
    return np.nanmean(np.nanmean(arr_4d[-n_use:], axis=2), axis=0)


def _create_atmosphere_comparison_zonal_mean(
    test_case_dir: Path, grid_results: dict, fields: list[dict],
    *, label: str | None = None,
) -> None:
    """4-panel zonal-mean ``(latitude, sigma-level)`` cross-section
    comparison across grids using a SHARED colormap.

    For Held-Suarez this is the canonical Fig. 3 / Fig. 4 of
    Held & Suarez (1994): the climatological zonal-mean temperature
    and zonal-mean zonal wind versus latitude and pressure level.
    Cross-grid disagreement in this plot is the principal physical-
    consistency check the user asked about in iter-5 ("ideally those
    Held and Suarez cases should all be the same after a few days").

    One PNG is written per requested 4-D field:
    ``comparison_zonal_mean_<field>.png``.

    The vertical axis is the model-level INDEX (0 = top, nlev-1 =
    surface) — labelled "Sigma level (top → bottom)" so the
    convention is unambiguous regardless of which vertical
    coordinate (sigma vs hybrid vs height) the run used.
    """
    case_name = label if label is not None else test_case_dir.name

    for spec in fields:
        field = spec["field"]
        vmin = spec["vmin"]
        vmax = spec["vmax"]
        cmap = spec["cmap"]
        units = spec.get("units", "")
        longname = spec.get("longname", field)

        zonal_means: dict[str, np.ndarray] = {}
        n_avg_used: int | None = None
        for grid_name, data in grid_results.items():
            snap = data["snapshots"]
            if field not in snap.files:
                continue
            arr = np.asarray(snap[field])
            if arr.ndim != 4:
                continue
            try:
                # Iter-10: use climatology (time-mean over trailing
                # snapshots) instead of single t_final snapshot.
                # Removes eddy variance that contaminated single-time
                # cross-grid RMS at finite spin-up.
                n_avg = _n_climatology_avg(arr.shape[0])
                zm = _zonal_mean_climatology(arr, n_avg=n_avg)
                if n_avg_used is None:
                    n_avg_used = n_avg
            except ValueError:
                continue
            zonal_means[grid_name] = zm

        if len(zonal_means) < 2:
            continue

        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8))

        sim_time_str = ""
        for data in grid_results.values():
            snap = data["snapshots"]
            if "times_days" in snap.files:
                t_days = np.asarray(snap["times_days"])
                t_final = float(t_days[-1])
                if n_avg_used is not None and len(t_days) >= n_avg_used:
                    t_avg_start = float(t_days[-n_avg_used])
                    sim_time_str = (
                        f" (climatology t = {t_avg_start:.1f}–{t_final:.1f} d, "
                        f"{n_avg_used} snapshots)"
                    )
                else:
                    sim_time_str = f" (t = {t_final:.2f} d)"
                break

        fig.suptitle(
            f"{case_name} — {longname} cross-grid comparison{sim_time_str}",
            fontsize=13, fontweight="bold",
        )

        slot_order = [g for g in GRID_TYPES if g in grid_results]
        slot_order += [g for g in grid_results if g not in slot_order]

        im = None
        for slot, ax in zip(range(nrows * ncols), axes.flat):
            if slot >= len(slot_order):
                ax.set_visible(False)
                continue
            grid_name = slot_order[slot]
            data = grid_results[grid_name]
            if grid_name not in zonal_means:
                ax.set_title(f"{grid_name} ({data['resolution']}) — {field} N/A")
                ax.set_visible(False)
                continue
            zm = zonal_means[grid_name]              # (n_lat, n_lev)
            n_lat, n_lev = zm.shape
            md_lat = np.asarray(data["snapshots"]["lat"])
            lat = md_lat if md_lat.size == n_lat else np.linspace(-90.0, 90.0, n_lat)
            # Plot with sigma level on vertical (top-down: 0 at top)
            im = ax.pcolormesh(
                lat, np.arange(n_lev), zm.T, cmap=cmap, vmin=vmin, vmax=vmax,
                shading="auto",
            )
            ax.invert_yaxis()  # so model top is at the top of the plot
            ax.set_title(f"{grid_name} ({data['resolution']})")
            ax.set_xlabel("Latitude (deg)")
            ax.set_ylabel("Sigma level (top → bottom)")
            ax.set_xlim(-90, 90)
            ax.set_xticks([-60, -30, 0, 30, 60])
            ax.grid(True, alpha=0.25, linewidth=0.4)

        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label(f"{field}" + (f" ({units})" if units else ""))

        from datetime import datetime
        fig.text(0.99, 0.01,
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ha="right", va="bottom", fontsize=7, color="gray")

        fig.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.07,
                            wspace=0.18, hspace=0.25)

        out_file = test_case_dir / f"comparison_zonal_mean_{field}.png"
        plt.savefig(out_file, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_file.name}")


def _interp_zonal_mean_to_target(
    zm_native: np.ndarray, lat_native: np.ndarray, lat_target: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a zonal-mean ``(n_lat_native, n_lev)`` array
    onto a target latitude axis ``lat_target`` (in degrees).  Returns
    ``(n_lat_target, n_lev)``.

    Used for cross-grid RMS comparison so each grid's native
    latitudes don't bias the metric toward the higher-resolution one.
    Iter-9.
    """
    n_lev = zm_native.shape[1]
    out = np.empty((lat_target.size, n_lev), dtype=np.float64)
    # Ensure latitude is monotonically increasing for np.interp.
    if lat_native[0] > lat_native[-1]:
        lat_native = lat_native[::-1]
        zm_native = zm_native[::-1]
    for k in range(n_lev):
        out[:, k] = np.interp(lat_target, lat_native, zm_native[:, k])
    return out


def _compute_cross_grid_rms_agreement(
    grid_results: dict, field: str, *, target_n_lat: int = 72,
) -> dict | None:
    """Compute pair-wise and ensemble cross-grid RMS agreement of the
    zonal-mean cross-section of ``field`` at the final timestep.

    Workflow:
    1. For each grid, compute ``_zonal_mean_at_final_time(field)``
       → ``(n_lat_native, n_lev)``.
    2. Interpolate each grid's zonal mean onto a common target
       latitude axis (``-90..90`` linspace, default 72 points).
    3. Vertical level count must match across grids; if it doesn't,
       skip that grid (warn).
    4. RMS pair-wise:  ``rms_AB = sqrt(mean((zm_A - zm_B)^2))``.
    5. Ensemble:       ``rms_ens = sqrt(mean((zm_grid - zm_mean_ensemble)^2))``.

    Returns a dict with keys ``pairwise`` (mapping ``"a-b"`` strings
    to floats), ``ensemble`` (mapping grid names to floats), and
    ``ensemble_mean`` (float — the average of all per-grid ensemble
    RMS values).  Returns ``None`` if fewer than 2 grids have valid
    zonal-mean data.

    Iter-9: addresses the user's iter-5 observation quantitatively.
    """
    zonal_means: dict[str, np.ndarray] = {}
    lat_natives: dict[str, np.ndarray] = {}
    n_avg_used: int | None = None
    for grid_name, data in grid_results.items():
        snap = data["snapshots"]
        if field not in snap.files:
            continue
        arr = np.asarray(snap[field])
        if arr.ndim != 4:
            continue
        try:
            # Iter-10: climatology mean (trailing N snapshots) instead
            # of single t_final, matches what the plot now shows.
            n_avg = _n_climatology_avg(arr.shape[0])
            zm = _zonal_mean_climatology(arr, n_avg=n_avg)
            if n_avg_used is None:
                n_avg_used = n_avg
        except ValueError:
            continue
        n_lat = zm.shape[0]
        md_lat = np.asarray(snap["lat"])
        lat_native = md_lat if md_lat.size == n_lat else np.linspace(-90.0, 90.0, n_lat)
        zonal_means[grid_name] = zm
        lat_natives[grid_name] = lat_native

    if len(zonal_means) < 2:
        return None

    # Verify nlev consistency across grids.
    nlev_per_grid = {g: zm.shape[1] for g, zm in zonal_means.items()}
    nlev_set = set(nlev_per_grid.values())
    if len(nlev_set) > 1:
        print(
            f"    [comparison] cross-grid RMS for {field}: vertical-level "
            f"count differs across grids ({nlev_per_grid}); skipping metric."
        )
        return None

    # Interpolate each grid's zonal mean to common latitude axis.
    lat_target = np.linspace(-90.0, 90.0, target_n_lat)
    zm_common: dict[str, np.ndarray] = {}
    for grid_name, zm in zonal_means.items():
        zm_common[grid_name] = _interp_zonal_mean_to_target(
            zm, lat_natives[grid_name], lat_target,
        )

    # Pair-wise RMS.
    pairwise: dict[str, float] = {}
    grid_names = sorted(zm_common.keys())
    for i, ga in enumerate(grid_names):
        for gb in grid_names[i + 1:]:
            diff = zm_common[ga] - zm_common[gb]
            pairwise[f"{ga} vs {gb}"] = float(np.sqrt(np.nanmean(diff ** 2)))

    # Iter-14: per-level cross-grid spread (max - min across grids
    # of each grid's COS-LATITUDE-weighted horizontal mean at each
    # level).  Iter-16 codex review MEDIUM: switched from uniform
    # latitude mean to cos-lat-weighted mean to match the area-
    # weighted convention used everywhere else in this script.
    # Without the weighting, the 72 equally-spaced latitude samples
    # over-weight the polar cells where actual cell area shrinks
    # as cos(φ).
    #
    # Exposes WHICH levels carry the cross-grid disagreement —
    # e.g. for the iter-13 60-day HS data the upper-troposphere
    # disagreement was much larger than the surface disagreement.
    n_lev = next(iter(nlev_set))
    per_level_spread = np.zeros(n_lev, dtype=np.float64)
    per_level_max_grid = ["?"] * n_lev
    per_level_min_grid = ["?"] * n_lev
    stack = np.stack([zm_common[g] for g in grid_names], axis=0)  # (n_grids, n_lat, n_lev)
    cos_lat_weights = np.cos(np.deg2rad(lat_target))               # (n_lat,)
    for k in range(n_lev):
        slab = stack[:, :, k]  # (n_grids, n_lat)
        # Cos-lat-weighted horizontal mean per grid, NaN-aware.
        weights_b = np.broadcast_to(cos_lat_weights[None, :], slab.shape)
        finite = np.isfinite(slab)
        masked_slab = np.where(finite, slab, 0.0)
        masked_w = np.where(finite, weights_b, 0.0)
        w_sum = masked_w.sum(axis=1)                               # (n_grids,)
        # Per-grid mean: nan when no finite samples.
        slab_horizmean = np.where(
            w_sum > 0, (masked_slab * masked_w).sum(axis=1) / np.maximum(w_sum, 1e-30), np.nan,
        )
        finite_grid = np.isfinite(slab_horizmean)
        if int(finite_grid.sum()) < 2:
            per_level_spread[k] = float("nan")
            per_level_max_grid[k] = "?"
            per_level_min_grid[k] = "?"
            continue
        finite_vals = slab_horizmean[finite_grid]
        finite_idx = np.where(finite_grid)[0]
        per_level_spread[k] = float(np.max(finite_vals) - np.min(finite_vals))
        per_level_max_grid[k] = grid_names[int(finite_idx[int(np.argmax(finite_vals))])]
        per_level_min_grid[k] = grid_names[int(finite_idx[int(np.argmin(finite_vals))])]

    # Ensemble (deviation from cross-grid mean).
    ens_mean_field = np.mean(stack, axis=0)
    ensemble: dict[str, float] = {}
    for grid_name, zm in zm_common.items():
        diff = zm - ens_mean_field
        ensemble[grid_name] = float(np.sqrt(np.nanmean(diff ** 2)))
    ensemble_mean = float(np.mean(list(ensemble.values())))

    return {
        "pairwise": pairwise,
        "ensemble": ensemble,
        "ensemble_mean": ensemble_mean,
        "n_grids": len(zm_common),
        "n_lat_target": target_n_lat,
        "n_lev": next(iter(nlev_set)),
        "n_avg_used": n_avg_used,
        "per_level_spread": per_level_spread,
        "per_level_max_grid": per_level_max_grid,
        "per_level_min_grid": per_level_min_grid,
        "grid_names": grid_names,
    }


def _create_atmosphere_comparison_timeseries(
    test_case_dir: Path, grid_results: dict,
    *, label: str | None = None,
) -> None:
    """Overlay the common scalar columns from ``mean_timeseries.csv`` across
    grids.  Plots up to 4 columns auto-detected as numeric scalar TS.

    Pass ``label`` to override the default suptitle text.  Output is
    written to ``test_case_dir/comparison_timeseries.png``.
    """
    common_cols: list[str] = []
    candidates = [
        "mean_height", "max_height", "min_height",
        "mean_T", "mean_T_sfc", "max_T", "min_T",
        "mean_u", "max_speed", "max_wind", "max_abs_w",
        # iter-5 M1: ``mass`` is the integral ``∫ p_s dA`` (Pa·m²)
        # written by the cube/latlon/ico/spectral hydro/AMIP runners.
        # Adding it to the TS candidate list lets the cross-grid
        # comparison overlay this mass diagnostic across grids.
        "mass", "mean_p_s", "mass_drift", "energy_drift",
        "mean_q_v", "max_q_v", "global_precip",
        # iter-26: RCE diagnostics — make moist-RCE timeseries
        # comparable across grids.  iter-27 codex MEDIUM: moved
        # ahead of NH ``mean_theta_prime`` etc. so the four-panel
        # cap doesn't drop them for typical RCE CSVs (which have
        # ``mean_T``, ``mean_T_sfc``, ``max_wind``, ``mean_precip``,
        # ``mean_cwv`` as the 5 useful columns).  ``mean_precip``
        # is now slot 4, ``mean_cwv`` slot 5 — still capped.
        "mean_precip", "mean_cwv",
        # DCMIP transport tracer scalars
        "q1_min", "q1_max", "q1_mean",
        # NH dycore scalars
        "mean_theta_prime", "mean_rho_prime",
    ]
    for col in candidates:
        if all(col in data["timeseries"].columns
               for data in grid_results.values()):
            common_cols.append(col)
    if not common_cols:
        return  # No common columns to compare.

    # iter-27 codex MEDIUM: cap at 6 panels (was 4) so RCE
    # diagnostics (``mean_T_sfc``, ``mean_T``, ``max_wind``,
    # ``mean_precip``, ``mean_cwv``) fit in a single 3×2 grid.
    common_cols = common_cols[:6]

    n = len(common_cols)
    nrows = (n + 1) // 2
    ncols = 1 if n == 1 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    title_label = label if label is not None else test_case_dir.name
    fig.suptitle(
        f"{title_label} — cross-grid time series",
        fontsize=13, fontweight="bold",
    )
    colors = _atmosphere_grid_color()

    time_col = None
    for data in grid_results.values():
        for c in ("time_days", "time_d", "t_days", "time"):
            if c in data["timeseries"].columns:
                time_col = c
                break
        if time_col is not None:
            break
    if time_col is None:
        time_col = grid_results[next(iter(grid_results))]["timeseries"].columns[0]

    for idx, col in enumerate(common_cols):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        for grid_name, data in grid_results.items():
            df = data["timeseries"]
            ax.plot(
                df[time_col], df[col],
                label=f"{grid_name} ({data['resolution']})",
                color=colors.get(grid_name, "black"),
                lw=1.2,
            )
        ax.set_ylabel(col)
        ax.set_xlabel("Time (days)" if "day" in time_col else time_col)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused panels.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, 2)
        axes[r, c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = test_case_dir / "comparison_timeseries.png"
    plt.savefig(out_file, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_file.name}")


def _create_atmosphere_comparison_summary(
    test_case_dir: Path, grid_results: dict,
    *, label: str | None = None,
    case_name: str | None = None,
) -> None:
    """Write a plain-text cross-grid summary table to
    ``comparison_summary.txt``.  Columns are auto-selected from the
    metadata that the runners write into ``results.txt`` (status,
    wall_time, plus any numeric metric the runner recorded).

    Pass ``label`` to override the default header (which is
    ``test_case_dir.name``).  ``case_name`` controls which entry of
    ``ATMOSPHERE_ZONAL_MEAN_FIELDS`` is consulted for the iter-9
    quantitative cross-grid RMS metric.  When ``None``, falls back
    to ``test_case_dir.name`` for case lookup, so the metric still
    fires for HS/baroclinic/AMIP regardless of whether the writer
    is invoked from the SW or hydrostatic path.
    """
    out_file = test_case_dir / "comparison_summary.txt"
    grids_sorted = sorted(grid_results.keys())
    if not grids_sorted:
        return

    # Union of all metadata keys across grids.
    all_keys: list[str] = []
    seen: set[str] = set()
    for g in grids_sorted:
        for k in grid_results[g]["metadata"]:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    title_label = label if label is not None else test_case_dir.name
    lookup_key = case_name if case_name is not None else test_case_dir.name
    zm_fields = ATMOSPHERE_ZONAL_MEAN_FIELDS.get(lookup_key, [])

    with open(out_file, "w") as fh:
        fh.write(f"{title_label} — Cross-grid comparison\n")
        fh.write("=" * 70 + "\n")
        header = f"{'metric':<32}  " + "  ".join(f"{g:<14}" for g in grids_sorted)
        fh.write(header + "\n")
        fh.write("-" * len(header) + "\n")
        for k in all_keys:
            row = f"{k:<32}  " + "  ".join(
                f"{grid_results[g]['metadata'].get(k, '-'):<14}"
                for g in grids_sorted
            )
            fh.write(row + "\n")
        fh.write("=" * 70 + "\n")

        # Iter-28: GPU/MPI efficiency snapshot for the user's prompt
        # item "ensure code runs efficiently on GPUs and MPI".
        # Rank grids by wall_time for this case.  Metric value is
        # wall-time PER SIMULATED DAY so cases of different
        # ``--days`` are roughly comparable.
        wall_per_day: list[tuple[str, float]] = []
        for g in grids_sorted:
            md = grid_results[g]["metadata"]
            wt_str = md.get("wall_time", "")
            # iter-32 codex MEDIUM: ``run_dcmip_transport`` writes
            # ``period_days`` and ``run_nonhydrostatic`` writes
            # ``duration_hours`` instead of ``days``.  Try all 3
            # so the GPU efficiency table fires on those cases too.
            days_str = md.get("days") or md.get("period_days") or ""
            duration_hours_str = md.get("duration_hours", "")
            if not wt_str or (not days_str and not duration_hours_str):
                continue
            try:
                wt = float(wt_str.rstrip("s").strip())
                if days_str:
                    d = float(days_str)
                else:
                    d = float(duration_hours_str) / 24.0
            except ValueError:
                continue
            # iter-32 codex caveat: skip non-finite or non-positive d.
            import math as _m
            if d <= 0 or not _m.isfinite(d) or not _m.isfinite(wt):
                continue
            wall_per_day.append((g, wt / d))
        if wall_per_day:
            wall_per_day.sort(key=lambda kv: kv[1])
            # iter-32 codex LOW: removed unused ``fastest`` local
            # (Ruff F841 lint failure).  ``slowest`` is the value
            # we actually use for the speedup denominator.
            fh.write(
                "\nGPU / MPI efficiency — wall-time per simulated day "
                "(faster = better):\n"
            )
            fh.write(f"  {'rank':<5}  {'grid':<14}  {'s/day':>10}  "
                     f"{'speedup vs slowest':>18}\n")
            slowest = wall_per_day[-1][1]
            for rank, (g, sd) in enumerate(wall_per_day, 1):
                speedup = slowest / sd if sd > 0 else float("inf")
                fh.write(
                    f"  {rank:<5d}  {g:<14}  {sd:10.2f}  {speedup:18.2f}x\n"
                )
            fh.write("=" * 70 + "\n")

        # Iter-9: quantitative cross-grid RMS agreement on the
        # zonal-mean cross-section, for cases where that diagnostic
        # is the canonical inter-model metric.
        for spec in zm_fields:
            field = spec["field"]
            units = spec.get("units", "")
            metric = _compute_cross_grid_rms_agreement(grid_results, field)
            if metric is None:
                continue
            n_avg = metric.get("n_avg_used")
            avg_label = (
                f"climatology = trailing {n_avg} snapshot(s)"
                if n_avg and n_avg > 1
                else "single t_final snapshot"
            )
            fh.write(
                f"\nQuantitative cross-grid RMS — zonal-mean {field}"
                f" ({avg_label}; interpolated to "
                f"{metric['n_lat_target']} lat × "
                f"{metric['n_lev']} lev)\n"
            )
            fh.write("-" * 70 + "\n")
            unit_suffix = f" {units}" if units else ""
            fh.write("Pair-wise RMS:\n")
            for pair, rms in sorted(
                metric["pairwise"].items(), key=lambda kv: kv[1]
            ):
                fh.write(f"  {pair:<40}  {rms:8.3f}{unit_suffix}\n")
            fh.write("\nDeviation from ensemble mean (per grid):\n")
            for g, rms in sorted(
                metric["ensemble"].items(), key=lambda kv: kv[1]
            ):
                fh.write(f"  {g:<40}  {rms:8.3f}{unit_suffix}\n")
            fh.write(
                f"\nEnsemble-averaged RMS: {metric['ensemble_mean']:.3f}"
                f"{unit_suffix} ({metric['n_grids']} grids)\n"
            )

            # Iter-14: top-5 vertical levels with largest cross-grid
            # spread.  Diagnostic: HS shows largest disagreement in
            # the upper troposphere (different sponge / hyperdiffusion
            # treatments).  Levels are 0=top, n_lev-1=surface.
            spread = metric.get("per_level_spread")
            if spread is not None and len(spread) > 0:
                fh.write(
                    "\nTop-5 levels by cross-grid spread (max-min of "
                    "horizontally-averaged value per grid; level 0 = "
                    "model top, level N-1 = surface):\n"
                )
                # argsort descending; skip NaN rows (iter-16 codex
                # LOW: NaN-robust ordering — only finite spreads
                # contribute to the top-5 list).
                finite_idx = np.where(np.isfinite(spread))[0]
                if finite_idx.size > 0:
                    ordered = finite_idx[np.argsort(-spread[finite_idx])][:5]
                    fh.write(
                        f"  {'level':<8}  {'spread':<10}  "
                        f"{'max grid':<14}  {'min grid':<14}\n"
                    )
                    for k in ordered:
                        fh.write(
                            f"  {int(k):<8d}  {spread[k]:8.3f}{unit_suffix:<2}  "
                            f"{metric['per_level_max_grid'][k]:<14}  "
                            f"{metric['per_level_min_grid'][k]:<14}\n"
                        )
                else:
                    fh.write("  (no finite per-level spreads)\n")
            fh.write("=" * 70 + "\n")
    print(f"    Saved: {out_file.name}")


def _create_cross_grid_comparisons_atmosphere(
    test_case_dir: Path,
    vertical_coord: str | None = None,
    *,
    allowed_grids: set[str] | None = None,
) -> None:
    """Top-level entry point: collect per-grid outputs under
    ``test_case_dir`` for ONE vertical-coord variant and emit shared-
    colorbar / shared-projection comparison plots plus a summary
    table.

    For hydrostatic-style cases (multiple vertical coords per grid),
    output is written to ``test_case_dir/<vertical_coord>/`` so that
    sigma and hybrid comparisons live in separate sub-directories and
    don't overwrite each other.  For SW-style cases (no vertical
    coord), output goes to ``test_case_dir/`` directly.

    No-ops if fewer than 2 grids produced output for the requested
    vertical coord.  ``allowed_grids`` (iter-6 L3) restricts which
    grid sub-directories are loaded; ``None`` loads every grid.
    """
    grid_results = _collect_grid_results_atmosphere(
        test_case_dir,
        vertical_coord=vertical_coord,
        allowed_grids=allowed_grids,
    )
    if len(grid_results) < 2:
        return

    case_name = test_case_dir.name
    fields = ATMOSPHERE_COMPARISON_FIELDS.get(
        case_name,
        # Default for unmapped cases: pick whatever shared field is available.
        [
            {"field": "height",     "vmin": None, "vmax": None, "cmap": "viridis", "units": "m"},
            {"field": "T",          "vmin": None, "vmax": None, "cmap": "plasma",  "units": "K"},
            {"field": "wind_speed", "vmin": 0,    "vmax": None, "cmap": "viridis", "units": "m/s"},
        ],
    )

    # Output directory: case_dir/<vert>/ when vertical_coord is set,
    # else case_dir/ (SW).  Create if missing.
    out_dir = test_case_dir / vertical_coord if vertical_coord else test_case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label = case_name + (f"/{vertical_coord}" if vertical_coord else "")
    print(f"  Creating cross-grid comparisons for {label}...")
    _create_atmosphere_comparison_snapshots(
        out_dir, grid_results, fields, label=label,
    )
    _create_atmosphere_comparison_timeseries(
        out_dir, grid_results, label=label,
    )
    _create_atmosphere_comparison_summary(
        out_dir, grid_results, label=label, case_name=case_name,
    )
    # Iter-8: zonal-mean cross-section comparison.  Only emitted for
    # cases registered in ATMOSPHERE_ZONAL_MEAN_FIELDS (currently HS,
    # baroclinic, AMIP) — these are the cases where a steady-state /
    # statistical-equilibrium zonal climatology is the canonical
    # cross-grid agreement metric (Held & Suarez 1994 Fig. 3-4).
    zm_fields = ATMOSPHERE_ZONAL_MEAN_FIELDS.get(case_name)
    if zm_fields:
        _create_atmosphere_comparison_zonal_mean(
            out_dir, grid_results, zm_fields, label=label,
        )


def _walk_atmosphere_test_cases(output_base: Path) -> list[Path]:
    """Enumerate per-case directories under ``output_base`` that have at
    least one grid subdirectory.  Layout is
    ``<output_base>/<equation_set>/<case>/<grid>/<resolution>/[<vert>/]``.

    For hydrostatic runs with a vertical-coord layer the *case* directory
    is one level above the grid subdir, i.e.
    ``<output_base>/hydrostatic/held_suarez/`` — this is the right level
    for the comparison plots because each grid's output sits below it.
    """
    cases: list[Path] = []
    if not output_base.exists():
        return cases
    for eq_dir in output_base.iterdir():
        if not eq_dir.is_dir():
            continue
        for case_dir in eq_dir.iterdir():
            if not case_dir.is_dir():
                continue
            # Has at least one known grid subdir?
            has_grid = any(
                (case_dir / g).is_dir() for g in GRID_TYPES
            )
            if has_grid:
                cases.append(case_dir)
    return cases


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
        "--family", type=str, default="all",
        choices=_FAMILY_CHOICES,
        help=("Run only cases tagged with the given Hughes (2026) "
              "tutorial family. 'hughes' selects the full canonical "
              "3D-spherical-dycore validation set; 'dcmip2008' / "
              "'dcmip2012' / 'dcmip2016' select paper-vintage groups; "
              "'sw'/'hydro'/'nh'/'moist'/'climate'/'tracer' select "
              "equation-set families. A single case may belong to "
              "multiple families simultaneously."))
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
    # Iter-31: AMIP "realistic GHG forcing" CLI knobs.  Applied
    # only when ``--radiation rrtmgp``.  Defaults to RRTMGP's own
    # present-day defaults (415 ppm CO2, 1900 ppb CH4, 332 ppb N2O).
    # Set ``--co2-ppmv 280`` for pre-industrial AMIP runs.  Time-
    # varying CMIP6 input4MIPs forcing is a deeper integration
    # tracked as a post-Ralph follow-up.
    p.add_argument(
        "--co2-ppmv", type=float, default=None,
        help="Override RRTMGP CO2 concentration [ppmv].  Default: 415 "
             "(present-day).  Use 280 for pre-industrial.")
    p.add_argument(
        "--ch4-ppbv", type=float, default=None,
        help="Override RRTMGP CH4 concentration [ppbv].  Default: 1900.")
    p.add_argument(
        "--n2o-ppbv", type=float, default=None,
        help="Override RRTMGP N2O concentration [ppbv].  Default: 332.")
    p.add_argument(
        "--cloud-scheme", type=str, default=None,
        choices=["none", "sundqvist"],
        help="iter-34: cloud fraction scheme for RRTMGP cloud-radiation "
             "coupling.  Default: ``none`` (clear-sky).  Only takes "
             "effect with ``--radiation rrtmgp``.  iter-37: "
             "``xu_randall`` removed from choices because it requires "
             "``q_cloud`` + ``q_ice`` condensate tracers, which the "
             "matrix runners' ``MicrophysicsConfig(scheme=\"none\")`` "
             "does not produce — would silently give zero cloud "
             "fraction and clear-sky radiation.  Re-add when a runner "
             "with active microphysics + condensate tracers exists.")
    # iter-39: ozone-profile knobs.  Mirrors the iter-31 GHG pattern.
    # Only applies under ``--radiation rrtmgp``; gray radiation
    # ignores ozone entirely.  The "standard" source is the existing
    # US-Standard-1976 climatology (no latitude dependence); the
    # "analytical" source is a Gaussian peak with optional
    # sin²(lat) scaling, useful for sensitivity studies of
    # stratospheric ozone amplitude on tropospheric circulation.
    p.add_argument(
        "--ozone-source", type=str, default=None,
        choices=["standard", "analytical", "none"],
        help="Override RRTMGP ozone profile source.  Default: "
             "``standard`` (US-Standard-1976, no latitude dependence). "
             "``analytical`` enables the latitude-dependent Gaussian "
             "profile.  ``none`` disables ozone absorption entirely. "
             "Only takes effect with ``--radiation rrtmgp``.")
    p.add_argument(
        "--ozone-peak-hpa", type=float, default=None,
        dest="ozone_peak_hpa",
        help="Override analytical-ozone-profile peak pressure [hPa]. "
             "Default: 30.  Setting this auto-promotes "
             "``--ozone-source`` to ``analytical`` if not already "
             "set.  iter-39 codex review: lower-case unit suffix "
             "matches the iter-31 GHG flags.")
    p.add_argument(
        "--ozone-max-vmr", type=float, default=None,
        help="Override analytical-ozone-profile peak volume mixing "
             "ratio (fraction, 0 < vmr <= 1).  Default: 8.0e-6 "
             "(8 ppmv).  Setting this auto-promotes "
             "``--ozone-source`` to ``analytical`` if not already "
             "set.  iter-39 codex HIGH: input is fractional VMR, "
             "NOT ppmv — passing ``8`` is an unphysical value and "
             "is rejected.  For 8 ppmv, use ``8e-6``.")
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
        "--days", type=float, default=None,
        help="Override per-case integration length (in days).  Takes "
             "precedence over both --quick and the matrix defaults.  "
             "Useful for cross-grid spin-up convergence studies "
             "(e.g. ``--only hydro --test held_suarez --days 200`` to "
             "run the canonical Held-Suarez spin-up).")
    p.add_argument(
        "--list", action="store_true",
        help="List all test cases and exit")
    p.add_argument(
        "--list-category-scripts", action="store_true",
        help="List canonical per-category runner scripts and exit")
    p.add_argument(
        "--no-cross-grid-plots", action="store_true",
        help="Skip the post-run cross-grid comparison plots and summary "
             "(useful for single-grid runs or quick iteration)")
    p.add_argument(
        "--cross-grid-plots-only", action="store_true",
        help="Skip running tests; only generate cross-grid comparison plots "
             "from an existing output tree.")
    return p


def filter_tests(tests: list[TestCase], args) -> list[TestCase]:
    filtered = tests
    eq_map = {"sw": "shallow_water", "hydro": "hydrostatic",
              "nh": "nonhydrostatic"}
    if args.only != "all":
        eq_set = eq_map[args.only]
        filtered = [t for t in filtered if t.equation_set == eq_set]
    if getattr(args, "family", "all") != "all":
        wanted = args.family
        filtered = [
            t for t in filtered
            if wanted in _CASE_FAMILIES.get(t.case, frozenset())
        ]
    if args.grid != "all":
        filtered = [t for t in filtered if t.grid_type == args.grid]
    if args.test:
        filtered = [t for t in filtered if args.test in t.case]
    return filtered


def main():
    parser = build_parser()
    args = parser.parse_args()

    # iter-16 codex review LOW: validate ``--days`` is positive.
    # Zero or negative values silently produce nonsensical runs.
    if args.days is not None and args.days <= 0:
        parser.error("--days must be positive")

    # iter-31: thread per-run GHG overrides through to
    # _make_rrtmgp_physics.  iter-32 codex MEDIUM: zero is a valid
    # sensitivity-test value (RRTMGP gas_optics has explicit
    # zero-abundance fallback at gas_optics.py:172-179) — only
    # reject NEGATIVE / non-finite values.
    import math as _math
    for fname, fval in [
        ("--co2-ppmv", args.co2_ppmv),
        ("--ch4-ppbv", args.ch4_ppbv),
        ("--n2o-ppbv", args.n2o_ppbv),
    ]:
        if fval is not None and (fval < 0 or not _math.isfinite(fval)):
            parser.error(f"{fname} must be a non-negative finite number")
    _RUNTIME_RRTMGP_OVERRIDES["co2_ppmv"] = args.co2_ppmv
    _RUNTIME_RRTMGP_OVERRIDES["ch4_ppbv"] = args.ch4_ppbv
    _RUNTIME_RRTMGP_OVERRIDES["n2o_ppbv"] = args.n2o_ppbv
    _RUNTIME_RRTMGP_OVERRIDES["cloud_scheme"] = args.cloud_scheme

    # iter-39: validate and stash ozone-profile knobs.  iter-39
    # codex review:
    #
    # HIGH — ``--ozone-max-vmr`` is fractional VMR, NOT ppmv;
    #   reject any value > 1 (unphysical) so ``8`` does not
    #   silently mean VMR=8 instead of 8 ppmv.
    # MEDIUM — ``--ozone-peak-hpa`` / ``--ozone-max-vmr`` only
    #   affect the analytical Gaussian path.  Reject the explicit
    #   contradiction case (``--ozone-source standard
    #   --ozone-peak-hpa 50``) at parse time; the implicit
    #   "no source given" case is auto-promoted to analytical
    #   inside ``_make_rrtmgp_physics``.
    if args.ozone_peak_hpa is not None and (
        args.ozone_peak_hpa <= 0 or not _math.isfinite(args.ozone_peak_hpa)
    ):
        parser.error("--ozone-peak-hpa must be a positive finite number")
    if args.ozone_max_vmr is not None:
        if args.ozone_max_vmr <= 0 or not _math.isfinite(args.ozone_max_vmr):
            parser.error("--ozone-max-vmr must be a positive finite number")
        if args.ozone_max_vmr > 1.0:
            parser.error(
                "--ozone-max-vmr is fractional volume mixing ratio "
                "(0 < vmr <= 1); for 8 ppmv use 8e-6.  Got "
                f"{args.ozone_max_vmr}"
            )
    if args.ozone_source in ("standard", "none") and (
        args.ozone_peak_hpa is not None or args.ozone_max_vmr is not None
    ):
        parser.error(
            "--ozone-peak-hpa / --ozone-max-vmr only affect the "
            "analytical Gaussian path; cannot be combined with "
            f"--ozone-source {args.ozone_source}.  Use "
            "--ozone-source analytical (or omit it for implicit "
            "promotion)."
        )
    _RUNTIME_RRTMGP_OVERRIDES["ozone_source"] = args.ozone_source
    _RUNTIME_RRTMGP_OVERRIDES["ozone_peak_hpa"] = args.ozone_peak_hpa
    _RUNTIME_RRTMGP_OVERRIDES["ozone_max_vmr"] = args.ozone_max_vmr

    # iter-7 codex review LOW: reset the multi-resolution warning
    # dedup set so a single ``main()`` invocation produces at most one
    # warning per grid_dir.  Persists across invocations only when the
    # interpreter calls ``main()`` repeatedly (rare), but resetting on
    # entry keeps the warning behaviour invocation-local.
    _RES_DIR_WARNED.clear()

    tests = filter_tests(TEST_MATRIX, args)

    if args.list_category_scripts:
        print("Canonical category runner scripts:")
        for cat, path in CATEGORY_RUNNER_HINTS.items():
            print(f"  - {cat:<14} {path}")
        return

    if args.list:
        print(f"{'#':>3}  {'Equation Set':<16}  {'Case':<22}  "
              f"{'Grid':<14}  {'Resolution':<10}  {'Vert':<7}  "
              f"{'Family':<8}  {'Days':>8}  {'Quick':>8}")
        print("-" * 113)
        for i, tc in enumerate(tests, 1):
            print(f"{i:3d}  {tc.equation_set:<16}  {tc.case:<22}  "
                  f"{tc.grid_type:<14}  {tc.resolution:<10}  "
                  f"{tc.vertical_coord:<7}  {tc.family:<8}  "
                  f"{tc.duration_days:8.4f}  "
                  f"{tc.quick_days:8.4f}")
        print(f"\nTotal: {len(tests)} test cases "
              f"(of {len(TEST_MATRIX)} in full matrix)")
        return
    output_base = Path(args.output)

    # --cross-grid-plots-only: skip the test loop entirely, just regenerate
    # comparison artifacts from existing per-grid output trees.
    # iter-5 codex review L2: this path runs BEFORE the empty-tests
    # check so a filter that matches no test cases doesn't suppress
    # the regeneration.  Filters DO restrict which case directories
    # get re-plotted (matched against tc.case names from the filtered
    # TEST_MATRIX).
    if args.cross_grid_plots_only:
        print("=" * 78)
        print("  Cross-grid comparison plots — generating from existing output")
        print(f"  Output base: {output_base}")
        if args.test or args.only != "all" or args.grid != "all":
            allowed_cases = {t.case for t in tests}
            # iter-73: when ``--test <name>`` is explicitly passed
            # but ``<name>`` is not in TEST_MATRIX (e.g., "rce" /
            # "omip" — these are run by external scripts
            # ``run_rce.py`` / ``run_omip.py`` and have no matrix
            # runner), fall back to ``{args.test}`` so the filter
            # still matches a per-case directory of that name.
            # Without this, ``run_rce_cross_grid.sh`` /
            # ``run_omip_cross_grid.sh`` invocations of
            # ``--cross-grid-plots-only --test rce`` would silently
            # find 0 cases.
            if args.test and not allowed_cases:
                allowed_cases = {args.test}
            allowed_grids = {t.grid_type for t in tests} if args.grid != "all" else None
            print(f"  Filters: case={args.test or '*'} only={args.only} grid={args.grid}")
            print(f"  Matching {len(allowed_cases)} case name(s): "
                  f"{', '.join(sorted(allowed_cases)) or '(none)'}")
            if allowed_grids is not None:
                print(f"  Restricting to grids: {sorted(allowed_grids)}")
        else:
            allowed_cases = None  # no filter
            allowed_grids = None
        print("=" * 78)
        cases = _walk_atmosphere_test_cases(output_base)
        if not cases:
            print(f"  No per-case output directories under {output_base}")
            return
        n_combos = 0
        for case_dir in cases:
            if allowed_cases is not None and case_dir.name not in allowed_cases:
                continue
            for vc in _vertical_coords_for_case(
                case_dir, allowed_grids=allowed_grids,
            ):
                try:
                    _create_cross_grid_comparisons_atmosphere(
                        case_dir,
                        vertical_coord=vc,
                        allowed_grids=allowed_grids,
                    )
                    n_combos += 1
                except Exception as e:  # noqa: BLE001
                    label = case_dir.name + (f"/{vc}" if vc else "")
                    print(f"  [WARN] {label}: {e}")
        print(f"  Done.  {n_combos} (case, vert) combo(s) processed.")
        return

    if not tests:
        print("No tests match the given filters.")
        return

    if args.resolution:
        # iter-95 fix: ``--resolution N`` (integer) was previously
        # applied verbatim to every grid type, breaking 3 of 4
        # parsers.  Specifically:
        #   cubed_sphere ``int(res[1:])``: "16" → 6 (silent wrong N)
        #   latlon ``res.split("x")``: "16" → unpack error
        #   icosahedral ``res.replace("ico","")``: "16" → level=16
        #     (4.29e+10 cells, ValueError)
        #   spectral ``res.replace("T","")``: "16" → 16 (correct)
        #
        # Now: if ``--resolution`` is a bare integer N, dispatch
        # per-grid:
        #   cubed_sphere → f"C{N}"
        #   latlon → f"{N}x{2*N}"
        #   icosahedral → "ico{level}" where 4^level ≈ N²/10
        #     (matches cell count to ≈ N×2N latlon coverage)
        #   spectral → f"T{N}"
        # Per-grid strings (e.g. "C36", "ico5") are still passed
        # through unchanged.
        # iter-115 (codex iter-114-followup): use the shared
        # ``validate_cli_resolution`` + ``expand_cli_resolution``
        # helpers so the atmosphere/ocean/OMIP CLIs all share
        # one implementation.  iter-115 also adds the per-grid
        # format whitelist, rejecting strings like ``2j`` /
        # ``hello`` that previously slipped past float()
        # parsing only to crash later in the per-grid parser.
        from legoesm.driver.cli_resolution import (
            validate_cli_resolution as _validate,
            expand_cli_resolution as _expand_shared,
        )
        cli_res = args.resolution
        N = _validate(
            cli_res,
            additional_examples="'C36', 'ico5', '72x144', 'T21'",
        )
        is_bare_int = N is not None

        def _expand_cli_res(grid_type: str) -> str:
            if not is_bare_int:
                return cli_res
            return _expand_shared(N, grid_type)

        tests = [TestCase(
            t.equation_set, t.case, t.grid_type,
            _expand_cli_res(t.grid_type),
            t.vertical_coord, t.duration_days, t.quick_days, t.run_kwargs)
            for t in tests]

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
        if args.days is not None:
            days = args.days
        elif args.quick:
            days = tc.quick_days
        else:
            days = tc.duration_days
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

    # --- Cross-grid comparison plots ---
    if not args.no_cross_grid_plots:
        print("\n" + "=" * 78)
        print("  CROSS-GRID COMPARISON PLOTS")
        print("=" * 78)
        cases = _walk_atmosphere_test_cases(output_base)
        for case_dir in cases:
            for vc in _vertical_coords_for_case(case_dir):
                try:
                    _create_cross_grid_comparisons_atmosphere(
                        case_dir, vertical_coord=vc,
                    )
                except Exception as e:  # noqa: BLE001
                    # Don't fail the whole run on plotting errors.
                    label = case_dir.name + (f"/{vc}" if vc else "")
                    print(f"  [WARN] comparison plots failed for {label}: {e}")
        print("=" * 78)

    if n_fail > 0 or n_error > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
