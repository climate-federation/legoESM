#!/usr/bin/env python
"""Dry baroclinic wave benchmark replicating CliMA Figure 3 (Yatunin et al. 2026).

Runs a 10-day Jablonowski-Williamson (2006) baroclinic instability test and
generates:

1. **12-panel figure** (baroclinic_wave_benchmark.png):
   - Top row: Surface pressure perturbation at 4 evenly-spaced snapshot days
   - Middle row: 850 hPa temperature at those same days
   - Bottom row: 850 hPa relative vorticity at those same days
   Snapshots are taken every 2 days.  For a 10-day run the figure shows
   days 2, 4, 8, and 10 (similar to Fig. 5 of Jablonowski & Williamson 2006).
   Northern Hemisphere only (0-90N), PlateCarree projection.

2. **Conservation timeseries** (baroclinic_wave_conservation.png):
   - Relative dry air mass deviation
   - Relative total energy deviation
   - Min surface pressure vs time
   - Max wind speed vs time

3. **NPZ diagnostics** (baroclinic_wave_diagnostics.npz)

Supported grids:
  spectral      -- Gaussian grid + spectral PE dycore (default)
  cubed-sphere  -- Cubed-sphere C-D grid + FV3 PE dycore
  icosahedral   -- MPAS Voronoi mesh + TRiSK PE dycore

Usage:
    JAX_ENABLE_X64=1 python scripts/run_baroclinic_wave_benchmark.py
    JAX_ENABLE_X64=1 python scripts/run_baroclinic_wave_benchmark.py --grid cubed-sphere --resolution C48
    JAX_ENABLE_X64=1 python scripts/run_baroclinic_wave_benchmark.py --grid spectral --resolution T42 --dt 600 --days 10
    JAX_ENABLE_X64=1 python scripts/run_baroclinic_wave_benchmark.py --grid icosahedral --resolution 5 --dt 150 --days 10

References:
    Jablonowski & Williamson (2006), QJRMS 132, 2943-2975.
    Ullrich et al. (2016), DCMIP2016 Test Case Document.
    Yatunin et al. (2026), JAMES, Figure 3.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# legoESM imports
# ---------------------------------------------------------------------------
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    SigmaCoordinate,
    create_sigma_coordinate,
    pressure_from_sigma,
)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, FV3HydrostaticState
from legoesm.core.operators import global_integral
from tests.test_cases.baroclinic_wave import P0
from legoesm.core.cfl import (
    adaptive_hyperdiff_coeff,
    estimate_min_dx_cubed_sphere,
    estimate_min_dx_gaussian,
    cfl_check_and_adjust,
)
from legoesm import constants

GRID_CHOICES = ("spectral", "cubed-sphere", "icosahedral")

# iter-219 / iter-212: per-grid recommended ``--scan-steps`` value
# resolved when the user passes ``--scan-steps auto``.  Module-level so
# tests can import it (regression-pin for the dispatch table) and so
# any future retune lands in exactly one place.  See scaling.md §13
# for the measurement rationale.
AUTO_SCAN_STEPS = {
    "spectral":     24,   # iter-212: +29..45 % gain (Python-dispatch matters for SH)
    "cubed-sphere": 24,   # iter-212: +20 % gain
    "icosahedral":  1,    # iter-212: scan=24 → -9 % at I5 (MPAS dycore well-fused)
}


def chunk_crosses_boundary(prev_step: int, current_step: int, period: int) -> bool:
    """Return True if the half-open interval ``(prev_step, current_step]``
    crosses a multiple of ``period``.

    iter-208: the older ``current_step % period == 0`` predicate was
    silently broken under ``--scan-steps`` > ``period`` — the multi-step
    chunk could leap over the boundary entirely so no step was ever
    exactly divisible.  This formulation is robust to any chunk size
    and reduces to the modulo predicate when ``current_step =
    prev_step + 1`` (the ``scan_steps == 1`` regime).
    """
    if period <= 0:
        return False
    return (current_step // period) > (prev_step // period)


def resolve_scan_steps(grid: str, raw_scan_steps, *, verbose: bool = True) -> int:
    """Resolve a user-supplied ``--scan-steps`` value into the integer
    actually used by the benchmark.  Centralises the iter-219 ``auto``
    dispatch and the iter-216 RuntimeWarning so both paths can be
    regression-pinned in a unit test instead of via subprocess.

    Parameters
    ----------
    grid : str
        One of ``GRID_CHOICES``.
    raw_scan_steps : str | int
        Either the string ``"auto"`` (case-insensitive) or anything
        ``int(...)`` accepts.
    verbose : bool, default True
        If True, prints the iter-219 dispatch banner on ``auto`` (matches
        the original ``main()`` behaviour).  Tests pass ``False`` to
        keep stdout clean.

    Returns
    -------
    int
        The resolved scan-steps integer (always ≥ 1 by construction in
        AUTO_SCAN_STEPS; users may still pass 0 explicitly, in which
        case downstream ``max(1, scan_steps)`` clamps).
    """
    if isinstance(raw_scan_steps, str) and raw_scan_steps.lower() == "auto":
        resolved = AUTO_SCAN_STEPS.get(grid, 1)
        if verbose:
            print(
                f"[--scan-steps auto] grid={grid} → "
                f"scan-steps={resolved} (iter-212 honest recommendation)"
            )
    else:
        resolved = int(raw_scan_steps)

    if resolved > 1 and grid == "icosahedral":
        warnings.warn(
            f"--scan-steps={resolved} on icosahedral grid: "
            f"iter-212 measured -9 % throughput at I5 (MPAS dycore is "
            f"already well-fused).  Consider --scan-steps=1 or auto.",
            RuntimeWarning, stacklevel=2,
        )
    return resolved


# ---------------------------------------------------------------------------
# Cubed-sphere to lat-lon regridding
# ---------------------------------------------------------------------------

def regrid_cubed_sphere_to_latlon(
    field_cs: np.ndarray,
    lon_cs: np.ndarray,
    lat_cs: np.ndarray,
    lon_ll: np.ndarray,
    lat_ll: np.ndarray,
) -> np.ndarray:
    """Regrid a cubed-sphere field to a regular lat-lon grid.

    Uses inverse-distance-weighted interpolation from the nearest
    cubed-sphere points for each lat-lon target point. This is fast
    and sufficient for visualization.

    Parameters
    ----------
    field_cs : np.ndarray, shape (6, n, n) or (6*n*n,)
        Scalar field on cubed-sphere.
    lon_cs, lat_cs : np.ndarray, shape (6, n, n)
        Cell-center coordinates in radians.
    lon_ll : np.ndarray, shape (n_lon,)
        Target longitudes in radians.
    lat_ll : np.ndarray, shape (n_lat,)
        Target latitudes in radians.

    Returns
    -------
    np.ndarray, shape (n_lat, n_lon)
    """
    from scipy.spatial import cKDTree

    # Flatten cubed-sphere coordinates to Cartesian (for k-d tree on sphere)
    lon_flat = lon_cs.ravel()
    lat_flat = lat_cs.ravel()
    field_flat = field_cs.ravel()

    # Convert to 3D Cartesian
    cos_lat = np.cos(lat_flat)
    x_cs = cos_lat * np.cos(lon_flat)
    y_cs = cos_lat * np.sin(lon_flat)
    z_cs = np.sin(lat_flat)

    tree = cKDTree(np.column_stack([x_cs, y_cs, z_cs]))

    # Target grid
    lon2d, lat2d = np.meshgrid(lon_ll, lat_ll)
    cos_lat_ll = np.cos(lat2d.ravel())
    x_ll = cos_lat_ll * np.cos(lon2d.ravel())
    y_ll = cos_lat_ll * np.sin(lon2d.ravel())
    z_ll = np.sin(lat2d.ravel())

    target_xyz = np.column_stack([x_ll, y_ll, z_ll])

    # Query k nearest neighbors and do inverse-distance weighting
    k = 4
    dist, idx = tree.query(target_xyz, k=k)

    # Handle exact matches (dist=0)
    dist = np.maximum(dist, 1e-15)
    weights = 1.0 / dist
    weights /= weights.sum(axis=1, keepdims=True)

    result = np.sum(weights * field_flat[idx], axis=1)
    return result.reshape(len(lat_ll), len(lon_ll))


# ---------------------------------------------------------------------------
# Energy diagnostics
# ---------------------------------------------------------------------------

def compute_total_energy(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> float:
    """Compute global total energy (kinetic + internal).

    E = integral over (KE + c_v * T) * dp/g * dA

    For the dry baroclinic wave (no moisture, no topography):
        KE = 0.5 * (u^2 + v^2)
        Internal = c_vd * T
    """
    g = constants.g
    c_vd = constants.c_vd
    area = grid.area  # (6, n, n)

    u = state.u.data
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data
    dsigma = sigma_coord.dsigma  # (nlev,)

    # dp = dsigma * p_s for each column
    dp = p_s[..., None] * dsigma  # (6, n, n, nlev)

    KE = 0.5 * (u ** 2 + v ** 2)
    energy_density = (KE + c_vd * T) * dp / g  # (6, n, n, nlev)

    # Sum over levels, then area-weighted global sum
    column_energy = jnp.sum(energy_density, axis=-1)  # (6, n, n)
    total = jnp.sum(column_energy * area)
    return float(total)


def compute_dry_mass(
    state: HydrostaticState,
    grid: CubedSphereGrid,
) -> float:
    """Compute global dry air mass = integral(p_s / g * dA)."""
    return float(jnp.sum(state.p_s.data * grid.area) / constants.g)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def _parse_resolution(res_str: str, grid_type: str) -> int:
    """Parse a resolution string like 'C48', 'T42', 'I5', or plain '48'."""
    s = res_str.strip().upper()
    if grid_type == "spectral":
        return int(s.lstrip("T"))
    elif grid_type == "icosahedral":
        return int(s.lstrip("I"))
    else:
        return int(s.lstrip("C"))


def _default_resolution(grid_type: str) -> str:
    """Return a sensible default resolution string for each grid type."""
    if grid_type == "spectral":
        return "T42"
    elif grid_type == "icosahedral":
        return "5"  # subdivision level 5 → 10242 cells (~120 km)
    return "C48"


def _default_dt(grid_type: str) -> float:
    if grid_type == "spectral":
        return 600.0
    elif grid_type == "icosahedral":
        return 150.0
    return 450.0


def _estimate_min_dx_icosahedral(level: int, radius: float = constants.R_earth) -> float:
    """Estimate minimum grid spacing on an icosahedral Voronoi mesh.

    Parameters
    ----------
    level : int
        Subdivision level.  nCells = 10 * 4^level + 2.
    radius : float
        Sphere radius [m].

    Returns
    -------
    dx_min : float
        Approximate minimum grid spacing [m].
    """
    n_cells = 10 * 4 ** level + 2
    dx_avg = radius * np.sqrt(4.0 * np.pi / n_cells)
    # SCVT meshes are very uniform; min dx ≈ 0.9 * mean dx
    return float(0.9 * dx_avg)


def main():
    parser = argparse.ArgumentParser(
        description="Dry baroclinic wave benchmark (CliMA Figure 3 reproduction)"
    )
    parser.add_argument(
        "--grid", type=str, choices=GRID_CHOICES, default="spectral",
        help="Grid / dycore type (default: spectral)"
    )
    parser.add_argument(
        "--resolution", type=str, default=None,
        help="Resolution: T<n> for spectral (e.g. T42), C<n> for cubed-sphere "
             "(e.g. C48), subdivision level for icosahedral (e.g. 5).  "
             "Defaults: T42 / C48 / 5."
    )
    parser.add_argument(
        "--nlev", type=int, default=26,
        help="Number of sigma levels (default: 26)"
    )
    parser.add_argument(
        "--dt", type=float, default=None,
        help="Time step in seconds (default: 600 spectral, 450 cubed-sphere, 150 icosahedral)"
    )
    parser.add_argument(
        "--days", type=int, default=10,
        help="Total simulation days (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Tag appended to output filenames (default: auto-generated from "
             "grid, resolution, and timestamp, e.g. "
             "'spectral_T42_20260327_143022'). "
             "Pass --tag '' to use plain filenames (old behavior)."
    )
    parser.add_argument(
        "--scan-steps", default="1",
        help="Number of consecutive ``model.step`` calls to fuse inside a "
             "single ``jax.lax.scan`` body (default: 1 = no scanning).  "
             "Pass ``auto`` to use the iter-212 honest per-grid recommendation: "
             "spectral=24, cubed-sphere=24, icosahedral=1.  "
             "Amortises Python+JAX-dispatch overhead per step.  Iter-212 "
             "measurements per grid (1-day BCW, GPU): "
             "**spectral**: scan=24 → +29-45 % (recommended); "
             "**cubed-sphere**: scan=24 → +20 % (recommended); "
             "**icosahedral**: scan=24 → -9 % at I5 (NOT recommended — "
             "MPAS dycore is already well-fused, scan adds compile cost "
             "with no offsetting Python-overhead gain).  Diagnostic "
             "samples still fire on every chunk-boundary that crosses "
             "``diag_interval_steps`` (=3600/dt), but pick a value "
             "smaller than ``diag_interval_steps`` to keep hourly "
             "cadence intact."
    )
    args = parser.parse_args()

    # iter-219 ``auto`` dispatch + iter-216 icosahedral warning live
    # in the module-level ``resolve_scan_steps`` helper so both paths
    # can be regression-pinned without subprocess overhead.
    args.scan_steps = resolve_scan_steps(args.grid, args.scan_steps)

    grid_type = args.grid
    res_str = args.resolution or _default_resolution(grid_type)
    N_GRID = _parse_resolution(res_str, grid_type)
    N_LEV = args.nlev
    DT = args.dt if args.dt is not None else _default_dt(grid_type)
    N_DAYS = args.days
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build output filename tag
    if args.tag is None:
        # Auto-generate: {grid}_{resolution}_{YYYYMMDD_HHMMSS}
        from datetime import datetime
        _res = res_str
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _file_tag = f"_{grid_type}_{_res}_{_ts}"
    elif args.tag == "":
        _file_tag = ""
    else:
        _file_tag = f"_{args.tag}"

    def _out(basename: str, ext: str) -> Path:
        """Build output path: OUTPUT_DIR / {basename}{_file_tag}.{ext}"""
        return OUTPUT_DIR / f"{basename}{_file_tag}.{ext}"

    USE_SPECTRAL = grid_type == "spectral"
    USE_ICOSAHEDRAL = grid_type == "icosahedral"

    # Grid spacing estimate.
    # For icosahedral, create the mesh early so we can compute dx_min
    # from actual mesh metrics.  The TRiSK vector Laplacian uses both
    # dcEdge (cell-cell) and dvEdge (vertex-vertex), and dvEdge can be
    # ~0.5× dcEdge on SCVT meshes.  Using the formula-based estimate
    # would over-estimate dx_min and produce an unstable diffusion coeff.
    _ico_mesh = None
    if USE_SPECTRAL:
        dx_min = estimate_min_dx_gaussian(N_GRID)
    elif USE_ICOSAHEDRAL:
        from legoesm.grids.voronoi import create_voronoi_mesh
        _ico_mesh = create_voronoi_mesh(subdivision_level=N_GRID)
        dx_min = float(jnp.minimum(jnp.min(_ico_mesh.dcEdge),
                                    jnp.min(_ico_mesh.dvEdge)))
    else:
        dx_min = estimate_min_dx_cubed_sphere(N_GRID)

    # CFL check (manual for icosahedral since cfl_check_and_adjust expects
    # cubed-sphere resolution parameter)
    DT_requested = DT
    if USE_ICOSAHEDRAL:
        from legoesm.core.cfl import cfl_max_dt
        dt_max = cfl_max_dt(dx_min, 60.0 + 300.0, cfl_number=0.8, ndim=2)
        if DT > dt_max:
            nice_values = [600, 450, 300, 240, 200, 180, 150, 120, 100, 90, 60, 45, 30]
            DT = dt_max
            for nv in nice_values:
                if nv <= dt_max:
                    DT = float(nv)
                    break
            print(f"  WARNING: --dt {DT_requested:.0f}s exceeds CFL limit "
                  f"(dt_max={dt_max:.0f}s at dx_min={dx_min/1000:.1f}km). "
                  f"Reducing to dt={DT:.0f}s.")
    else:
        DT = cfl_check_and_adjust(
            DT, N_GRID, model_type="primitive_eq",
            max_wind=60.0, gravity_wave_speed=300.0,
            grid_type="gaussian" if USE_SPECTRAL else "cubed_sphere",
        )

    # Hyperdiffusion
    nu4 = adaptive_hyperdiff_coeff(dx_min, DT, order=4, safety=0.5)

    if USE_SPECTRAL:
        res_label = f"T{N_GRID}"
    elif USE_ICOSAHEDRAL:
        n_cells = 10 * 4 ** N_GRID + 2
        res_label = f"I{N_GRID}({n_cells})"
    else:
        res_label = f"C{N_GRID}"
    print("=" * 72)
    print("  Dry Baroclinic Wave Benchmark (CliMA Figure 3)")
    print("=" * 72)
    print(f"  Grid:            {grid_type}")
    print(f"  Resolution:      {res_label}")
    print(f"  Vertical levels: {N_LEV}")
    print(f"  Time step:       {DT:.0f} s")
    print(f"  Duration:        {N_DAYS} days")
    print(f"  dx_min:          {dx_min / 1000:.1f} km")
    print(f"  Hyperdiffusion:  {nu4:.3e} m^4/s")
    print(f"  Output:          {OUTPUT_DIR}/")
    print()

    # -----------------------------------------------------------------------
    # Grid and initial conditions
    # -----------------------------------------------------------------------
    print("Creating grid...")
    sigma = create_sigma_coordinate(N_LEV)

    if USE_SPECTRAL:
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPrimitiveEquationModel,
            SpectralPEConfig,
            spectral_pe_to_grid,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral

        grid = create_gaussian_grid(N_GRID)
        cdgrid = None

        print("Initializing Jablonowski-Williamson baroclinic wave (spectral)...")
        state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)

        # Diagnostics from grid-point fields
        gp = spectral_pe_to_grid(state, grid, sigma)
        ps_init = np.array(gp['p_s'])
        print(f"  Initial max |u|: {float(jnp.max(jnp.abs(gp['u']))):.1f} m/s")
        print(f"  Initial mean T:  {float(jnp.mean(gp['T'])):.1f} K")
        print(f"  Initial p_s:     {float(jnp.mean(gp['p_s'])) / 100:.1f} hPa")

        config = SpectralPEConfig(
            hyperdiff_coeff=nu4,
            hyperdiff_order=4,
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, config)

    elif USE_ICOSAHEDRAL:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
            MPASPrimitiveEquationConfig,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity

        grid = _ico_mesh  # already created during dx_min computation
        cdgrid = None

        print("Initializing Jablonowski-Williamson baroclinic wave (icosahedral)...")
        state = baroclinic_wave_init_mpas(grid, sigma, perturbed=True)
        ps_init = np.array(state.p_s.data)
        u_e, v_n = reconstruct_cell_velocity(state.u.data, grid)
        print(f"  Initial max |u|: {float(jnp.max(jnp.sqrt(u_e**2 + v_n**2))):.1f} m/s")
        print(f"  Initial mean T:  {float(jnp.mean(state.T.data)):.1f} K")
        print(f"  Initial p_s:     {float(jnp.mean(state.p_s.data)) / 100:.1f} hPa")

        config = MPASPrimitiveEquationConfig(
            nu_del4=nu4,
            nu_del4_ps=nu4,
            fix_mass=True,
            pv_scheme="energy",
            time_integrator="ssp_rk3",
        )
        model = MPASPrimitiveEquationModel(grid, sigma, config)

    else:
        from legoesm.grids.cubed_sphere import (
            CubedSphereGrid,
            create_cubed_sphere,
            rotate_winds_grid_to_geo,
        )
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.operators_cdgrid import dgrid_vorticity, dgrid_to_center_vector
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
            CDGridPrimitiveEquationConfig,
            hydrostatic_to_fv3,
            fv3_to_hydrostatic,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        grid = create_cubed_sphere(N_GRID)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        print("Initializing Jablonowski-Williamson baroclinic wave...")
        state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
        ps_init = np.array(state_cc.p_s.data)
        print(f"  Initial max |u|: {float(jnp.max(jnp.abs(state_cc.u.data))):.1f} m/s")
        print(f"  Initial mean T:  {float(jnp.mean(state_cc.T.data)):.1f} K")
        print(f"  Initial p_s:     {float(jnp.mean(state_cc.p_s.data)) / 100:.1f} hPa")

        state = hydrostatic_to_fv3(state_cc, cdgrid)

        config = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=nu4,
            hyperdiff_ps_coeff=nu4,
            use_conservation_fixer=True,
            fix_mass=True,
            anchor_mass_to_initial=True,
            time_integrator="ssp_rk3",
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, config)

    # -----------------------------------------------------------------------
    # Helper closures for grid-dependent diagnostics
    # -----------------------------------------------------------------------
    def _get_grid_fields(st):
        """Return (p_s, u, v, T) as numpy arrays from the current state."""
        if USE_SPECTRAL:
            gp = spectral_pe_to_grid(st, grid, sigma)
            return gp['p_s'], gp['u'], gp['v'], gp['T']
        elif USE_ICOSAHEDRAL:
            u_e, v_n = reconstruct_cell_velocity(st.u.data, grid)
            return st.p_s.data, u_e, v_n, st.T.data
        else:
            cc = fv3_to_hydrostatic(st, cdgrid)
            u_c, v_c = dgrid_to_center_vector(st.u_d.data, st.v_d.data)
            return cc.p_s.data, u_c, v_c, cc.T.data

    def _blowup_check(st):
        ps, _, _, _ = _get_grid_fields(st)
        ps_max = float(jnp.max(ps))
        return jnp.all(jnp.isfinite(ps)) and ps_max < 2e5

    # -----------------------------------------------------------------------
    # Time integration with diagnostic collection
    # -----------------------------------------------------------------------
    n_steps_total = int(N_DAYS * 86400 / DT)
    diag_interval_steps = max(1, int(3600 / DT))  # every hour

    # Snapshot days for the benchmark figure — every 2 days, always including
    # the final day so long runs (--days 20, 30, …) get late-time snapshots.
    snapshot_days = list(range(2, N_DAYS + 1, 2))
    if N_DAYS not in snapshot_days:
        snapshot_days.append(N_DAYS)
    snapshot_steps = {int(d * 86400 / DT): d for d in snapshot_days}

    # Diagnostic storage
    diag_times = []
    diag_dry_mass = []
    diag_total_energy = []
    diag_ps_min = []
    diag_max_wind = []

    # Initial diagnostics
    ps0, u0, v0, T0 = _get_grid_fields(state)
    if USE_SPECTRAL:
        area = grid.grid_area  # (n_lat, n_lon)
        dsigma = sigma.dsigma
        mass_init = float(jnp.sum(ps0 * area) / constants.g)
        dp0 = ps0[..., None] * dsigma
        KE0 = 0.5 * (u0 ** 2 + v0 ** 2)
        energy_init = float(jnp.sum(
            jnp.sum((KE0 + constants.c_vd * T0) * dp0 / constants.g, axis=-1)
            * area
        ))
    elif USE_ICOSAHEDRAL:
        area = grid.areaCell  # (nCells,)
        dsigma = sigma.dsigma
        mass_init = float(jnp.sum(ps0 * area) / constants.g)
        dp0 = ps0[..., None] * dsigma  # (nCells, nlev)
        KE0 = 0.5 * (u0 ** 2 + v0 ** 2)
        energy_init = float(jnp.sum(
            jnp.sum((KE0 + constants.c_vd * T0) * dp0 / constants.g, axis=-1)
            * area
        ))
    else:
        state_cc_init = fv3_to_hydrostatic(state, cdgrid)
        mass_init = compute_dry_mass(state_cc_init, grid)
        energy_init = compute_total_energy(state_cc_init, grid, sigma)

    diag_times.append(0.0)
    diag_dry_mass.append(mass_init)
    diag_total_energy.append(energy_init)
    diag_ps_min.append(float(jnp.min(ps0)))
    diag_max_wind.append(float(jnp.max(jnp.sqrt(u0 ** 2 + v0 ** 2))))

    # Snapshot storage: day -> dict with 'p_s', 'u', 'v', 'T' as numpy
    snapshots = {}

    print(f"\nIntegrating for {n_steps_total} steps...")

    # JIT warmup — compile both the per-step path and (if requested)
    # the scanned multi-step path so the timed loop is on the warm path.
    SCAN_STEPS = max(1, int(args.scan_steps))

    print("  JIT compiling (first step)...", end=" ", flush=True)
    t_jit = time.time()
    state = model.step(state, DT)
    jax.block_until_ready(jax.tree.leaves(state))
    print(f"done ({time.time() - t_jit:.1f}s)")

    if SCAN_STEPS > 1:
        # Build a JIT-compiled function that runs ``SCAN_STEPS`` steps
        # via ``jax.lax.scan``.  Each call replaces ``SCAN_STEPS`` Python
        # iterations + JAX dispatches with one.  ``model.step`` is
        # already JAX-pure (it takes a pytree state, returns a pytree
        # state, with ``dt`` static) so it slots straight into a scan
        # body.  The scan accumulates nothing (output ``None``) — we
        # only care about the final carry.
        def _scan_body(s, _):
            return model.step(s, DT), None

        @jax.jit
        def _scan_chunk(s):
            s_new, _ = jax.lax.scan(_scan_body, s, None, length=SCAN_STEPS)
            return s_new

        print(f"  JIT compiling ({SCAN_STEPS}-step scan)...", end=" ", flush=True)
        t_jit2 = time.time()
        state = _scan_chunk(state)
        jax.block_until_ready(jax.tree.leaves(state))
        print(f"done ({time.time() - t_jit2:.1f}s)")
        # The scan-chunk warmup advanced the state by SCAN_STEPS extra
        # steps; account for that in the post-warmup loop bookkeeping.
        steps_consumed_in_warmup = 1 + SCAN_STEPS
    else:
        steps_consumed_in_warmup = 1

    t_start = time.time()
    last_print = t_start

    step = steps_consumed_in_warmup - 1
    prev_current_step = step + 1   # current_step value at the *start* of the next iteration
    while step + 1 < n_steps_total:
        if SCAN_STEPS > 1 and (step + 1 + SCAN_STEPS) <= n_steps_total:
            state = _scan_chunk(state)
            step += SCAN_STEPS
        else:
            state = model.step(state, DT)
            step += 1

        current_step = step + 1  # 1-indexed (we already did step 0 warmup)
        day = current_step * DT / 86400.0

        # Blowup check whenever the chunk crosses a 100-step boundary.
        if chunk_crosses_boundary(prev_current_step, current_step, 100):
            if not _blowup_check(state):
                print(f"\n  *** BLOWUP at day {day:.2f}, step {current_step} ***")
                sys.exit(1)

        # Hourly diagnostics — fire if the just-completed chunk crossed
        # at least one ``diag_interval_steps`` boundary (iter-208 fix).
        if chunk_crosses_boundary(prev_current_step, current_step, diag_interval_steps):
            ps_now, u_now, v_now, T_now = _get_grid_fields(state)
            if USE_SPECTRAL or USE_ICOSAHEDRAL:
                mass_now = float(jnp.sum(ps_now * area) / constants.g)
                dp_now = ps_now[..., None] * dsigma
                KE_now = 0.5 * (u_now ** 2 + v_now ** 2)
                energy_now = float(jnp.sum(
                    jnp.sum((KE_now + constants.c_vd * T_now) * dp_now / constants.g, axis=-1)
                    * area
                ))
            else:
                cc_now = fv3_to_hydrostatic(state, cdgrid)
                mass_now = compute_dry_mass(cc_now, grid)
                energy_now = compute_total_energy(cc_now, grid, sigma)

            diag_times.append(day)
            diag_dry_mass.append(mass_now)
            diag_total_energy.append(energy_now)
            diag_ps_min.append(float(jnp.min(ps_now)))
            diag_max_wind.append(float(jnp.max(jnp.sqrt(u_now ** 2 + v_now ** 2))))

        # Save snapshots
        if current_step in snapshot_steps:
            snap_day = snapshot_steps[current_step]
            ps_s, u_s, v_s, T_s = _get_grid_fields(state)
            snapshots[snap_day] = {
                'p_s': np.array(ps_s),
                'u': np.array(u_s),
                'v': np.array(v_s),
                'T': np.array(T_s),
            }
            print(f"  Snapshot saved at day {snap_day}")

        # Progress reporting every 30 seconds
        now = time.time()
        if now - last_print > 30:
            elapsed = now - t_start
            steps_done = step
            steps_per_sec = steps_done / elapsed
            eta = (n_steps_total - current_step) / steps_per_sec
            ps_now, _, _, _ = _get_grid_fields(state)
            ps_min_cur = float(jnp.min(ps_now))
            print(
                f"  Day {day:6.2f}/{N_DAYS} | "
                f"ps_min={ps_min_cur / 100:.1f} hPa | "
                f"{steps_per_sec:.1f} steps/s | "
                f"ETA {eta / 60:.0f} min"
            )
            last_print = now

        # Roll forward the chunk-boundary cursor for the next iteration.
        prev_current_step = current_step

    elapsed_total = time.time() - t_start
    # iter-212: 3-decimal seconds + 2-decimal sps so short runs don't
    # report integer-rounded sps (the iter-204/205 ".../1s (479.4 sps)"
    # numbers were inflated upper bounds — see scaling.md §10.b).
    print(f"\nIntegration complete: {elapsed_total:.3f}s "
          f"({n_steps_total / max(elapsed_total, 1e-9):.2f} steps/s)")

    # Convert arrays
    diag_times = np.array(diag_times)
    diag_dry_mass = np.array(diag_dry_mass)
    diag_total_energy = np.array(diag_total_energy)
    diag_ps_min = np.array(diag_ps_min)
    diag_max_wind = np.array(diag_max_wind)

    # -----------------------------------------------------------------------
    # Save NPZ diagnostics
    # -----------------------------------------------------------------------
    npz_path = _out("baroclinic_wave_diagnostics", "npz")
    # Throughput stats land in the npz so post-hoc parsers don't need
    # to re-grep stdout (used by ``scripts/bench/parse_strong_sweep.py``).
    npz_data = dict(
        times_days=diag_times,
        dry_mass=diag_dry_mass,
        total_energy=diag_total_energy,
        ps_min=diag_ps_min,
        max_wind=diag_max_wind,
        ps_init=ps_init,
        resolution=N_GRID,
        nlev=N_LEV,
        dt=DT,
        wall_time_s=elapsed_total,
        steps_per_sec=n_steps_total / max(elapsed_total, 1e-9),
        n_steps_total=n_steps_total,
        backend=jax.default_backend(),
        snapshot_days=np.array(sorted(snapshots.keys())),
    )
    # Save all snapshot fields so long runs preserve intermediate data.
    for snap_day in sorted(snapshots.keys()):
        for var, arr in snapshots[snap_day].items():
            npz_data[f"snapshot_day{snap_day:03d}_{var}"] = arr
    np.savez(npz_path, **npz_data)
    print(f"  Diagnostics saved to {npz_path}")

    # -----------------------------------------------------------------------
    # Regridding setup for plotting
    # -----------------------------------------------------------------------
    print("\nPreparing snapshots for plotting...")

    # Find sigma level closest to 850 hPa (sigma = 0.85)
    sigma_full = np.array(sigma.sigma_full)
    k_850 = int(np.argmin(np.abs(sigma_full - 0.85)))
    p_850_actual = sigma_full[k_850] * 1000  # hPa
    print(f"  850 hPa level: k={k_850}, actual p={p_850_actual:.0f} hPa")

    if USE_SPECTRAL:
        # Spectral: data is already on a lat-lon (Gaussian) grid
        lon_ll = np.array(grid.lon)   # (n_lon,), radians
        lat_ll = np.array(grid.lat)   # (n_lat,), radians

        def _field_to_latlon(field_2d):
            """Identity — spectral data is already (n_lat, n_lon)."""
            return np.array(field_2d)
    elif USE_ICOSAHEDRAL:
        # Icosahedral: regrid unstructured cell-center data to regular lat-lon
        lon_cell = np.array(grid.lonCell)   # (nCells,), radians
        lat_cell = np.array(grid.latCell)   # (nCells,), radians

        n_lon_plot = 360
        n_lat_plot = 180
        lon_ll = np.linspace(0, 2 * np.pi, n_lon_plot, endpoint=False)
        lat_ll = np.linspace(-np.pi / 2, np.pi / 2, n_lat_plot)

        # Build k-d tree once for reuse across all fields
        from scipy.spatial import cKDTree
        cos_lat_c = np.cos(lat_cell)
        _ico_tree = cKDTree(np.column_stack([
            cos_lat_c * np.cos(lon_cell),
            cos_lat_c * np.sin(lon_cell),
            np.sin(lat_cell),
        ]))
        lon2d_t, lat2d_t = np.meshgrid(lon_ll, lat_ll)
        cos_lat_t = np.cos(lat2d_t.ravel())
        _ico_target_xyz = np.column_stack([
            cos_lat_t * np.cos(lon2d_t.ravel()),
            cos_lat_t * np.sin(lon2d_t.ravel()),
            np.sin(lat2d_t.ravel()),
        ])
        _ico_dist, _ico_idx = _ico_tree.query(_ico_target_xyz, k=4)
        _ico_dist = np.maximum(_ico_dist, 1e-15)
        _ico_w = 1.0 / _ico_dist
        _ico_w /= _ico_w.sum(axis=1, keepdims=True)

        def _field_to_latlon(field_cells):
            """IDW interpolation from unstructured cells to lat-lon."""
            flat = np.asarray(field_cells).ravel()
            return np.sum(_ico_w * flat[_ico_idx], axis=1).reshape(
                len(lat_ll), len(lon_ll)
            )
    else:
        from legoesm.grids.cubed_sphere import rotate_winds_grid_to_geo
        lon_cs = np.array(grid.lon)   # (6, n, n), radians
        lat_cs = np.array(grid.lat)   # (6, n, n), radians

        n_lon_plot = 360
        n_lat_plot = 180
        lon_ll = np.linspace(0, 2 * np.pi, n_lon_plot, endpoint=False)
        lat_ll = np.linspace(-np.pi / 2, np.pi / 2, n_lat_plot)

        def _field_to_latlon(field_cs):
            return regrid_cubed_sphere_to_latlon(
                field_cs, lon_cs, lat_cs, lon_ll, lat_ll
            )

    # -----------------------------------------------------------------------
    # Figure 1: 12-panel benchmark figure (JW2006 Figure 5 style)
    # -----------------------------------------------------------------------
    N_PLOT_COLS = 4
    available_days = sorted(snapshots.keys())

    # Pick N_PLOT_COLS evenly-spaced snapshots from the available set.
    if len(available_days) >= N_PLOT_COLS:
        indices = np.linspace(0, len(available_days) - 1, N_PLOT_COLS).round().astype(int)
        plot_days = [available_days[i] for i in indices]
    else:
        plot_days = available_days

    print(f"Generating {3 * len(plot_days)}-panel benchmark figure "
          f"(days {', '.join(str(d) for d in plot_days)})...")

    if len(plot_days) >= 2:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            has_cartopy = True
        except ImportError:
            print("  WARNING: cartopy not installed, using basic projection")
            has_cartopy = False

        n_cols = len(plot_days)
        fig, axes = plt.subplots(
            3, n_cols,
            figsize=(5.5 * n_cols, 10),
            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)} if has_cartopy else {},
        )
        # Ensure axes is 2D even when n_cols == 1
        if axes.ndim == 1:
            axes = axes[:, None]

        lon_deg = np.degrees(lon_ll)
        lat_deg = np.degrees(lat_ll)
        nh_mask = lat_deg >= 0
        lat_nh = lat_deg[nh_mask]

        for col, day in enumerate(plot_days):
            snap = snapshots[day]

            # --- Surface pressure perturbation ---
            ps_pert = snap['p_s'] - ps_init
            ps_pert_ll = _field_to_latlon(ps_pert)
            ps_pert_nh = ps_pert_ll[nh_mask, :] / 100  # hPa

            ax = axes[0, col]
            if has_cartopy:
                vmax_ps = max(np.abs(ps_pert_nh).max(), 1.0)
                cf = ax.contourf(
                    lon_deg, lat_nh, ps_pert_nh,
                    levels=np.linspace(-vmax_ps, vmax_ps, 21),
                    cmap="RdBu_r",
                    transform=ccrs.PlateCarree(),
                    extend="both",
                )
                ax.coastlines(linewidth=0.5, color="gray")
                ax.set_extent([0, 360, 0, 90], crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.05, shrink=0.8)
                cb.set_label("hPa")
            else:
                vmax_ps = max(np.abs(ps_pert_nh).max(), 1.0)
                cf = ax.contourf(
                    lon_deg, lat_nh, ps_pert_nh,
                    levels=np.linspace(-vmax_ps, vmax_ps, 21),
                    cmap="RdBu_r", extend="both",
                )
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.08, shrink=0.8)
                cb.set_label("hPa")
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 90)
            ax.set_title(f"Surface pressure perturbation, day {day}", fontsize=11)

            # --- 850 hPa temperature ---
            T_850 = snap['T'][..., k_850]
            T_850_ll = _field_to_latlon(T_850)
            T_850_nh = T_850_ll[nh_mask, :]

            ax = axes[1, col]
            if has_cartopy:
                cf = ax.contourf(
                    lon_deg, lat_nh, T_850_nh,
                    levels=20, cmap="RdYlBu_r",
                    transform=ccrs.PlateCarree(),
                )
                ax.coastlines(linewidth=0.5, color="gray")
                ax.set_extent([0, 360, 0, 90], crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.05, shrink=0.8)
                cb.set_label("K")
            else:
                cf = ax.contourf(lon_deg, lat_nh, T_850_nh,
                                 levels=20, cmap="RdYlBu_r")
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.08, shrink=0.8)
                cb.set_label("K")
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 90)
            ax.set_title(f"850 hPa temperature, day {day}", fontsize=11)

            # --- 850 hPa relative vorticity ---
            u_snap = snap['u']
            v_snap = snap['v']

            if USE_SPECTRAL or USE_ICOSAHEDRAL:
                # Winds are already geographic (east, north);
                # for icosahedral, Perot reconstruction gives (u_east, v_north)
                u_850 = u_snap[..., k_850]
                v_850 = v_snap[..., k_850]
                if USE_ICOSAHEDRAL:
                    u_east_850_ll = _field_to_latlon(u_850)
                    v_north_850_ll = _field_to_latlon(v_850)
                else:
                    u_east_850_ll = u_850
                    v_north_850_ll = v_850
            else:
                # Rotate grid-aligned winds to geographic
                u_east_3d = np.zeros_like(u_snap)
                v_north_3d = np.zeros_like(v_snap)
                for k in range(N_LEV):
                    u_e, v_n = rotate_winds_grid_to_geo(
                        jnp.array(u_snap[..., k]),
                        jnp.array(v_snap[..., k]),
                        grid.angle,
                    )
                    u_east_3d[..., k] = np.array(u_e)
                    v_north_3d[..., k] = np.array(v_n)

                u_east_850_ll = _field_to_latlon(u_east_3d[..., k_850])
                v_north_850_ll = _field_to_latlon(v_north_3d[..., k_850])

            a = float(constants.R_earth)
            dlon = lon_ll[1] - lon_ll[0]
            dlat = lat_ll[1] - lat_ll[0]
            cos_lat_2d = np.cos(lat_ll)[:, None]

            dvdlon = np.gradient(v_north_850_ll, dlon, axis=1)
            u_cos = u_east_850_ll * cos_lat_2d
            du_cos_dlat = np.gradient(u_cos, dlat, axis=0)

            cos_lat_safe = np.maximum(cos_lat_2d, 1e-6)
            vort_ll = (1.0 / (a * cos_lat_safe)) * dvdlon - (1.0 / a) * du_cos_dlat / cos_lat_safe
            vort_nh = vort_ll[nh_mask, :]

            ax = axes[2, col]
            vmax_vort = 2e-4
            if has_cartopy:
                cf = ax.contourf(
                    lon_deg, lat_nh, vort_nh,
                    levels=np.linspace(-vmax_vort, vmax_vort, 21),
                    cmap="RdBu_r",
                    transform=ccrs.PlateCarree(),
                    extend="both",
                )
                ax.coastlines(linewidth=0.5, color="gray")
                ax.set_extent([0, 360, 0, 90], crs=ccrs.PlateCarree())
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.05, shrink=0.8)
                cb.set_label("s$^{-1}$")
            else:
                cf = ax.contourf(
                    lon_deg, lat_nh, vort_nh,
                    levels=np.linspace(-vmax_vort, vmax_vort, 21),
                    cmap="RdBu_r", extend="both",
                )
                cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                                  pad=0.08, shrink=0.8)
                cb.set_label("s$^{-1}$")
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 90)
            ax.set_title(f"850 hPa relative vorticity, day {day}", fontsize=11)

        if USE_SPECTRAL:
            dycore_label = "spectral PE"
        elif USE_ICOSAHEDRAL:
            dycore_label = "MPAS TRiSK PE"
        else:
            dycore_label = "C-D grid PE"
        plt.suptitle(
            f"Dry Baroclinic Wave Benchmark  |  {res_label} L{N_LEV}  |  "
            f"dt={DT:.0f}s  |  legoESM {dycore_label}",
            fontsize=13, y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = _out("baroclinic_wave_benchmark", "png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fig_path}")
    else:
        print("  Skipping benchmark figure (not enough snapshots)")

    # -----------------------------------------------------------------------
    # Figure 2: Conservation timeseries (CliMA Figure 9 style)
    # -----------------------------------------------------------------------
    print("Generating conservation timeseries figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Relative dry mass deviation
    mass_rel = (diag_dry_mass - mass_init) / mass_init
    axes[0, 0].plot(diag_times, mass_rel, "b-", linewidth=1.2)
    axes[0, 0].set_ylabel("Relative deviation")
    axes[0, 0].set_title("(a) Dry air mass conservation")
    axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel("Time [days]")

    # (b) Relative total energy deviation
    energy_rel = (diag_total_energy - energy_init) / abs(energy_init)
    axes[0, 1].plot(diag_times, energy_rel, "r-", linewidth=1.2)
    axes[0, 1].set_ylabel("Relative deviation")
    axes[0, 1].set_title("(b) Total energy conservation")
    axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel("Time [days]")

    # (c) Minimum surface pressure
    axes[1, 0].plot(diag_times, diag_ps_min / 100, "k-", linewidth=1.2)
    axes[1, 0].set_ylabel("Min p$_s$ [hPa]")
    axes[1, 0].set_title("(c) Minimum surface pressure")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Time [days]")

    # (d) Maximum wind speed
    axes[1, 1].plot(diag_times, diag_max_wind, "g-", linewidth=1.2)
    axes[1, 1].set_ylabel("Max |v| [m/s]")
    axes[1, 1].set_title("(d) Maximum wind speed")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("Time [days]")

    plt.suptitle(
        f"Baroclinic Wave Conservation  |  {res_label} L{N_LEV}  |  "
        f"dt={DT:.0f}s",
        fontsize=13,
    )
    plt.tight_layout()
    cons_path = _out("baroclinic_wave_conservation", "png")
    plt.savefig(cons_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {cons_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  Resolution:      {res_label} L{N_LEV}")
    print(f"  Duration:        {N_DAYS} days ({n_steps_total} steps)")
    print(f"  Wall time:       {elapsed_total:.3f}s "
          f"({n_steps_total / max(elapsed_total, 1e-9):.2f} steps/s)")
    print(f"  Mass drift:      {mass_rel[-1]:+.3e} (relative)")
    print(f"  Energy drift:    {energy_rel[-1]:+.3e} (relative)")
    print(f"  Min p_s (final): {diag_ps_min[-1] / 100:.1f} hPa")
    print(f"  Max wind (final):{diag_max_wind[-1]:.1f} m/s")
    print(f"\n  Outputs:")
    print(f"    {_out('baroclinic_wave_benchmark', 'png')}")
    print(f"    {_out('baroclinic_wave_conservation', 'png')}")
    print(f"    {_out('baroclinic_wave_diagnostics', 'npz')}")
    print()


if __name__ == "__main__":
    main()
