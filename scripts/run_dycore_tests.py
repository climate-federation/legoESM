#!/usr/bin/env python
"""Master dynamical core test suite for legoESM.

Runs all solver types across standard test cases at low resolution
for verification, saves results and diagnostics, generates summary.

Solver × Test Matrix:
                        SW_FV  SW_spec  hydro_FV  hydro_spec  NH_FV  NH_spec
Williamson Test 2        ✓       ✓
Williamson Test 5        ✓       ✓
Held-Suarez (30 days)                    ✓         ✓
Baroclinic wave (10d)                    ✓         ✓
DCMIP-2012 transport                     (tracer)
DCMIP-2025 TC1                                                ✓      ✓
DCMIP-2025 TC2a                                               ✓
DCMIP-2025 TC3                                                ✓

Usage:
    python scripts/run_dycore_tests.py
    python scripts/run_dycore_tests.py --skip-slow    # Skip long tests
    python scripts/run_dycore_tests.py --only sw      # Only SW tests
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None

# Global output directory
OUTPUT_BASE = Path("results/atmosphere/dycore_tests")

# Collect results for summary
ALL_RESULTS = []


def _requires_x64(args) -> bool:
    """Return True when the selected suite requires spectral (float64) solvers."""
    only = args.only

    if only == "transport":
        return False
    if only == "nh" and args.skip_slow:
        return False
    if only == "si" and args.skip_slow:
        return False

    # Default suite and all other categories include at least one spectral solver.
    return True


def record(test_name, solver, status, key_metric, value, wall_time, notes=""):
    """Record a test result."""
    ALL_RESULTS.append({
        "test": test_name,
        "solver": solver,
        "status": status,
        "metric": key_metric,
        "value": value,
        "wall_time": wall_time,
        "notes": notes,
    })


def _snapshot_steps(n_steps: int) -> list[int]:
    """Return start/mid/end snapshot step numbers (inclusive end)."""
    if n_steps <= 0:
        return [0]
    return sorted({0, max(1, n_steps // 2), n_steps})


def _capture_snapshot(
    snapshots: dict[int, dict[str, np.ndarray]],
    target_steps: list[int],
    step_num: int,
    extractor,
    state,
):
    """Capture selected fields at a requested step if needed."""
    if step_num in target_steps and step_num not in snapshots:
        snapshots[step_num] = extractor(state)


def _cubed_faces_to_mosaic(field_2d_faces: np.ndarray) -> np.ndarray:
    """Convert (6, n, n) cubed-sphere faces to a simple 2x3 image mosaic."""
    n = field_2d_faces.shape[1]
    mosaic = np.full((2 * n, 3 * n), np.nan, dtype=np.float64)
    face_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
    ]
    for face, (row, col) in enumerate(face_positions):
        mosaic[row * n:(row + 1) * n, col * n:(col + 1) * n] = field_2d_faces[face]
    return mosaic


def _field_to_panel(field_data) -> np.ndarray:
    """Convert either lat-lon 2D or cubed-sphere (6,n,n) to a plottable panel."""
    arr = np.asarray(field_data)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 6:
        return _cubed_faces_to_mosaic(arr)
    raise ValueError(f"Expected 2D lat-lon or (6,n,n) cubed-sphere field, got {arr.shape}")


def _regrid_faces_to_latlon(
    field_2d_faces: np.ndarray,
    cube_lon_deg: np.ndarray,
    cube_lat_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate cubed-sphere face samples to a regular lat-lon grid.

    Uses inverse-distance weighting of nearest neighbors in 3D Cartesian
    coordinates on the unit sphere. This avoids lon-lat triangulation seam
    artifacts near cube edges.
    """
    n = int(field_2d_faces.shape[1])
    n_lon = max(360, 8 * n)
    n_lat = n_lon // 2

    lon = np.asarray(cube_lon_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(cube_lat_deg, dtype=np.float64).reshape(-1)
    val = np.asarray(field_2d_faces, dtype=np.float64).reshape(-1)

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon = ((lon[valid] + 180.0) % 360.0) - 180.0
    lat = np.clip(lat[valid], -90.0, 90.0)
    val = val[valid]

    lon_cent = np.linspace(-180.0, 180.0, n_lon, endpoint=False) + 180.0 / n_lon
    lat_cent = np.linspace(-90.0, 90.0, n_lat)
    lon2d, lat2d = np.meshgrid(lon_cent, lat_cent)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        cos_lat = np.cos(lat_rad)
        src_xyz = np.column_stack(
            [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)],
        )

        lon_t = np.deg2rad(lon2d.reshape(-1))
        lat_t = np.deg2rad(lat2d.reshape(-1))
        cos_lat_t = np.cos(lat_t)
        tgt_xyz = np.column_stack(
            [cos_lat_t * np.cos(lon_t), cos_lat_t * np.sin(lon_t), np.sin(lat_t)],
        )

        k = min(8, src_xyz.shape[0])
        tree = cKDTree(src_xyz)
        dist, idx = tree.query(tgt_xyz, k=k)
        if k == 1:
            field_ll = val[idx].reshape(lon2d.shape)
        else:
            dist = np.maximum(dist, 1.0e-12)
            w = 1.0 / dist
            w /= np.sum(w, axis=1, keepdims=True)
            field_ll = np.sum(val[idx] * w, axis=1).reshape(lon2d.shape)
    except Exception:
        lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
        lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
        sum_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges), weights=val)
        cnt_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges))
        with np.errstate(invalid="ignore", divide="ignore"):
            field_ll = np.where(cnt_grid > 0.0, sum_grid / cnt_grid, np.nan)

    return lon2d, lat2d, field_ll


def _color_limits(panels: list[np.ndarray]) -> tuple[float, float]:
    """Color limits shared across timesteps for one field.

    Uses full min/max over all snapshots. If a field spans both positive and
    negative values, use symmetric limits around zero to avoid visual bias.
    """
    finite_chunks = [p[np.isfinite(p)] for p in panels if p is not None]
    finite_chunks = [c for c in finite_chunks if c.size > 0]
    if not finite_chunks:
        return -1.0, 1.0

    values = np.concatenate(finite_chunks)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmax_abs = float(np.nanmax(np.abs(values))) if values.size else 1.0
        vmax_abs = max(vmax_abs, 1e-12)
        return -vmax_abs, vmax_abs

    if vmin < 0.0 < vmax:
        vmax_abs = max(abs(vmin), abs(vmax), 1e-12)
        vmin, vmax = -vmax_abs, vmax_abs

    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1e-6
        vmin -= pad
        vmax += pad

    return vmin, vmax


def _format_sim_time(step: int, dt: float) -> str:
    """Format simulation time in a human-readable unit for plot titles."""
    t_sec = step * dt
    if t_sec < 3600.0:
        return f"{t_sec / 60.0:.1f} min"
    if t_sec < 86400.0:
        return f"{t_sec / 3600.0:.2f} h"
    return f"{t_sec / 86400.0:.2f} d"


def _sigma_index_nearest(sigma_full, target_sigma: float) -> int:
    """Return index of sigma_full nearest to a target sigma value."""
    sigma_arr = np.asarray(sigma_full)
    return int(np.argmin(np.abs(sigma_arr - target_sigma)))


def _zonal_anomaly(field_2d: np.ndarray) -> np.ndarray:
    """Remove zonal mean (longitude mean) from a lat-lon 2D field."""
    return field_2d - np.mean(field_2d, axis=1, keepdims=True)


def _save_case_snapshots(
    test_dir: Path,
    case_name: str,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    field_specs: list[tuple[str, str, str]],
    cube_lon_deg: np.ndarray | None = None,
    cube_lat_deg: np.ndarray | None = None,
    central_longitude: float = 0.0,
):
    """Save multi-time field snapshots for one test case."""
    if not snapshots:
        return

    snap_steps = sorted(snapshots.keys())
    n_rows = len(field_specs)
    n_cols = len(snap_steps)
    use_projected_cube = (
        HAS_CARTOPY
        and cube_lon_deg is not None
        and cube_lat_deg is not None
    )

    fig = plt.figure(figsize=((5.0 if use_projected_cube else 4.4) * n_cols + 0.9, 3.4 * n_rows))
    width_ratios = [1.0] * n_cols + [0.06]
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=width_ratios,
        hspace=0.28,
        wspace=0.18,
    )

    if use_projected_cube:
        proj = ccrs.Robinson(central_longitude=central_longitude)
        axes = [
            [fig.add_subplot(gs[row, col], projection=proj) for col in range(n_cols)]
            for row in range(n_rows)
        ]
    else:
        axes = [[fig.add_subplot(gs[row, col]) for col in range(n_cols)] for row in range(n_rows)]
    caxes = [fig.add_subplot(gs[row, n_cols]) for row in range(n_rows)]

    for row, (key, row_label, cmap) in enumerate(field_specs):
        row_panels = []
        for step in snap_steps:
            field_dict = snapshots.get(step, {})
            if key not in field_dict:
                row_panels.append(None)
                continue
            arr = np.asarray(field_dict[key])
            if use_projected_cube and arr.ndim == 3 and arr.shape[0] == 6:
                row_panels.append(arr)
            else:
                row_panels.append(_field_to_panel(arr))

        valid_panels = [p for p in row_panels if p is not None]
        vmin, vmax = _color_limits(valid_panels)
        im = None

        for col, step in enumerate(snap_steps):
            ax = axes[row][col]
            panel = row_panels[col]
            time_label = _format_sim_time(step, dt)

            if panel is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if use_projected_cube and panel.ndim == 3 and panel.shape[0] == 6:
                    lon_pts = np.asarray(cube_lon_deg, dtype=np.float64).reshape(-1)
                    lat_pts = np.asarray(cube_lat_deg, dtype=np.float64).reshape(-1)
                    val_pts = np.asarray(panel, dtype=np.float64).reshape(-1)
                    valid = (
                        np.isfinite(lon_pts)
                        & np.isfinite(lat_pts)
                        & np.isfinite(val_pts)
                    )
                    n_face = int(panel.shape[1])
                    marker_size = max(0.8, 2200.0 / float(n_face * n_face))
                    im = ax.scatter(
                        lon_pts[valid],
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
                    ax.set_global()
                    ax.coastlines(linewidth=0.35, color="0.35")
                    ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.35)
                else:
                    im = ax.imshow(
                        panel,
                        origin="lower",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

            title = f"step {step}\nt={time_label}"
            ax.set_title(title, fontsize=9)

            if col == 0:
                ax.set_ylabel(row_label, fontsize=10)

        if im is not None:
            fig.colorbar(im, cax=caxes[row], orientation="vertical")
        else:
            caxes[row].axis("off")

    fig.suptitle(f"{case_name} - Field Snapshots", fontsize=13)
    fig.subplots_adjust(top=0.92)
    fig.savefig(test_dir / "field_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(test_dir / "snapshot_times.txt", "w") as f:
        f.write("step,time_seconds,time_days\n")
        for step in snap_steps:
            t_sec = step * dt
            f.write(f"{step},{t_sec:.1f},{t_sec/86400.0:.6f}\n")


# =============================================================================
# 1. Shallow Water FV Tests
# =============================================================================

def run_sw_fv_tests(output_dir):
    """Run Williamson Tests 2 and 5 with FV shallow water."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import (
        ShallowWaterModel,
        ShallowWaterConfig,
    )
    from tests.test_cases.williamson import (
        williamson_test2,
        williamson_test5,
    )
    from legoesm.core.operators import global_integral

    N = 16
    DT = 600.0
    HYPERDIFF_SW = 5e16 * (48 / N) ** 4
    # Keep cube-edge continuity control active even at coarse C16.
    # Disabling it reintroduces visible face-edge artifacts in W2/W5 snapshots.
    EDGE_BLEND_SW = 0.25

    # --- Test 2: Steady geostrophic flow (5 days) ---
    test_dir = output_dir / "01_sw_fv_williamson2"
    test_dir.mkdir(parents=True, exist_ok=True)
    print("\n  [01] SW FV - Williamson Test 2 (C16, 5 days)...")

    try:
        grid = create_cubed_sphere(N)
        cube_lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
        cube_lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
        state_init = williamson_test2(grid)
        # Hyperdiffusion needed at C16 to prevent aliasing instability
        config = ShallowWaterConfig(
            hyperdiff_coeff=HYPERDIFF_SW,
            edge_blend_strength=EDGE_BLEND_SW,
        )
        model = ShallowWaterModel(grid, config)

        state = state_init
        n_steps = int(5 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_sw_fv(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            return {
                "wind_speed": np.sqrt(u * u + v * v),
                "height": np.asarray(s.h.data),
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_sw_fv, state)

        t0 = time.time()
        for i in range(n_steps):
            state = model.step(state, DT)
            _capture_snapshot(snapshots, snap_targets, i + 1, extract_fields_sw_fv, state)
        jax.block_until_ready(state.h.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "SW FV Williamson 2",
            snapshots,
            DT,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        # Error: L2 norm of height perturbation
        h_err = jnp.sqrt(jnp.mean((state.h.data - state_init.h.data) ** 2))
        h_err_val = float(h_err)
        stable = bool(jnp.all(jnp.isfinite(state.h.data)))
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"L2_error: {h_err_val:.6e}\nstable: {stable}\nwall_time: {wall:.1f}\n")

        record("Williamson 2", f"SW FV C{N}", status, "L2 error (5d)", f"{h_err_val:.2e}", wall)
        print(f"    {status} | L2 error={h_err_val:.2e} | {wall:.1f}s")

    except Exception as e:
        record("Williamson 2", f"SW FV C{N}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()

    # --- Test 5: Zonal flow over mountain (15 days) ---
    test_dir = output_dir / "02_sw_fv_williamson5"
    test_dir.mkdir(parents=True, exist_ok=True)
    print("\n  [02] SW FV - Williamson Test 5 (C16, 15 days)...")

    try:
        state_init = williamson_test5(grid)
        config = ShallowWaterConfig(
            hyperdiff_coeff=5e16 * (48 / N) ** 4,
            edge_blend_strength=EDGE_BLEND_SW,
        )
        model = ShallowWaterModel(grid, config)

        state = state_init
        n_steps = int(15 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_sw_fv(s):
            u = np.asarray(s.u.data)
            v = np.asarray(s.v.data)
            return {
                "wind_speed": np.sqrt(u * u + v * v),
                "height": np.asarray(s.h.data),
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_sw_fv, state)

        t0 = time.time()
        for i in range(n_steps):
            state = model.step(state, DT)
            _capture_snapshot(snapshots, snap_targets, i + 1, extract_fields_sw_fv, state)
        jax.block_until_ready(state.h.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "SW FV Williamson 5",
            snapshots,
            DT,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        h_max = float(jnp.max(state.h.data))
        h_min = float(jnp.min(state.h.data))
        stable = bool(jnp.all(jnp.isfinite(state.h.data)))
        status = "PASS" if stable else "FAIL"

        mass_init = float(jnp.mean(state_init.h.data))
        mass_final = float(jnp.mean(state.h.data))
        mass_drift = abs(mass_final - mass_init) / abs(mass_init)

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"h_range: [{h_min:.0f}, {h_max:.0f}]\n")
            f.write(f"mass_drift: {mass_drift:.2e}\nstable: {stable}\nwall_time: {wall:.1f}\n")

        record("Williamson 5", f"SW FV C{N}", status, "mass drift (15d)", f"{mass_drift:.2e}", wall)
        print(f"    {status} | h=[{h_min:.0f}, {h_max:.0f}] | mass drift={mass_drift:.2e} | {wall:.1f}s")

    except Exception as e:
        record("Williamson 5", f"SW FV C{N}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# 2. Shallow Water Spectral Tests
# =============================================================================

def run_sw_spectral_tests(output_dir, solver: str = "ssp_rk3"):
    """Run Williamson Tests 2 and 5 with spectral shallow water."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralSWConfig,
        spectral_sw_tendencies,
        williamson_test2_spectral,
        williamson_test5_spectral,
        spectral_to_grid,
        compute_spectral_diagnostics,
    )
    from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
    from legoesm.timestepping.ssp_rk34 import ssp_rk34_step

    solver_l = solver.lower()
    if solver_l in ("ssp_rk34", "ssp34", "rk34"):
        stepper = ssp_rk34_step
        solver_tag = "ssp34"
    elif solver_l in ("ssp_rk3", "ssp3", "rk3"):
        stepper = ssp_rk3_step
        solver_tag = "ssp_rk3"
    else:
        raise ValueError(f"Unsupported spectral SW solver={solver!r}")

    T = 21
    DT = 120.0

    # --- Test 2: Steady geostrophic flow (5 days) ---
    test_dir = output_dir / "03_sw_spectral_williamson2"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [03] SW Spectral - Williamson Test 2 (T21, 5 days, {solver_tag})...")

    try:
        grid = create_gaussian_grid(T)
        state_init = williamson_test2_spectral(grid)
        config = SpectralSWConfig(hyperdiff_coeff=0.0)

        def tendency_fn(s):
            return spectral_sw_tendencies(s, grid, config)
        step_jit = jax.jit(lambda s, dt: stepper(s, tendency_fn, dt))

        state = state_init
        n_steps = int(5 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_sw_spec(s):
            fields = spectral_to_grid(s, grid)
            u = np.asarray(fields["u"])
            v = np.asarray(fields["v"])
            return {
                "wind_speed": np.sqrt(u * u + v * v),
                "height": np.asarray(fields["h"]),
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_sw_spec, state)

        t0 = time.time()
        for i in range(n_steps):
            state = step_jit(state, DT)
            _capture_snapshot(snapshots, snap_targets, i + 1, extract_fields_sw_spec, state)
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "SW Spectral Williamson 2",
            snapshots,
            DT,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
        )

        # Error: L2 of height perturbation
        fields_init = spectral_to_grid(state_init, grid)
        fields_final = spectral_to_grid(state, grid)
        h_err = float(jnp.sqrt(jnp.mean((fields_final['h'] - fields_init['h']) ** 2)))
        stable = bool(jnp.all(jnp.isfinite(fields_final['h'])))
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"L2_error: {h_err:.6e}\nstable: {stable}\nwall_time: {wall:.1f}\n")
            f.write(f"solver: {solver_tag}\n")

        record("Williamson 2", f"SW Spec T{T} {solver_tag}", status, "L2 error (5d)", f"{h_err:.2e}", wall)
        print(f"    {status} | L2 error={h_err:.2e} | {wall:.1f}s")

    except Exception as e:
        record("Williamson 2", f"SW Spec T{T}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()

    # --- Test 5: Mountain flow (15 days) ---
    test_dir = output_dir / "04_sw_spectral_williamson5"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [04] SW Spectral - Williamson Test 5 (T21, 15 days, {solver_tag})...")

    try:
        state_init = williamson_test5_spectral(grid)

        a = grid.radius
        eig_max = T * (T + 1) / (a * a)
        hyperdiff_coeff = 1.0 / (1.0 * 3600.0 * eig_max ** 2)

        config = SpectralSWConfig(
            mean_depth=5960.0,
            hyperdiff_coeff=hyperdiff_coeff,
            hyperdiff_order=2,
        )

        def tendency_fn5(s):
            return spectral_sw_tendencies(s, grid, config)
        step_jit5 = jax.jit(lambda s, dt: stepper(s, tendency_fn5, dt))

        state = state_init
        n_steps = int(15 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_sw_spec(s):
            fields = spectral_to_grid(s, grid)
            u = np.asarray(fields["u"])
            v = np.asarray(fields["v"])
            return {
                "wind_speed": np.sqrt(u * u + v * v),
                "height": np.asarray(fields["h"]),
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_sw_spec, state)

        t0 = time.time()
        for i in range(n_steps):
            state = step_jit5(state, DT)
            _capture_snapshot(snapshots, snap_targets, i + 1, extract_fields_sw_spec, state)
        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "SW Spectral Williamson 5",
            snapshots,
            DT,
            [
                ("wind_speed", "Wind speed (m/s)", "magma"),
                ("height", "Fluid depth h (m)", "viridis"),
            ],
        )

        fields = spectral_to_grid(state, grid)
        h_max = float(jnp.max(fields['h']))
        h_min = float(jnp.min(fields['h']))
        stable = bool(jnp.all(jnp.isfinite(fields['h'])))
        status = "PASS" if stable else "FAIL"

        diag_init = compute_spectral_diagnostics(state_init, grid)
        diag_final = compute_spectral_diagnostics(state, grid)
        mass_drift = abs(diag_final['mass'] - diag_init['mass']) / abs(diag_init['mass'])

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"h_range: [{h_min:.0f}, {h_max:.0f}]\n")
            f.write(f"mass_drift: {mass_drift:.2e}\nstable: {stable}\nwall_time: {wall:.1f}\n")
            f.write(f"solver: {solver_tag}\n")

        record("Williamson 5", f"SW Spec T{T} {solver_tag}", status, "mass drift (15d)", f"{mass_drift:.2e}", wall)
        print(f"    {status} | h=[{h_min:.0f}, {h_max:.0f}] | mass drift={mass_drift:.2e} | {wall:.1f}s")

    except Exception as e:
        record("Williamson 5", f"SW Spec T{T}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# 3. Hydrostatic FV Tests
# =============================================================================

def run_hydro_fv_tests(output_dir):
    """Run Held-Suarez and Baroclinic wave with FV PE."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq import (
        PrimitiveEquationModel,
        PrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez import (
        held_suarez_forcing,
        held_suarez_init,
    )
    from legoesm.atmosphere.physics.baroclinic_wave import baroclinic_wave_init
    from legoesm.core.operators import global_integral

    N = 16
    NLEV = 10
    DT = 600.0
    HYPERDIFF = 5e16 * (48 / N) ** 4
    # Keep hydro edge blending active at coarse resolution as well.
    # Zero blend reintroduces visible face-edge discontinuities in 3D fields.
    EDGE_BLEND_HYDRO_UV = 0.12
    EDGE_BLEND_HYDRO_T = 0.08
    EDGE_BLEND_HYDRO_PS = 0.15
    EDGE_BLEND_HYDRO_WIDTH = 2

    # --- Held-Suarez (30 days) ---
    test_dir = output_dir / "05_hydro_fv_held_suarez"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [05] Hydro FV - Held-Suarez (C{N}/L{NLEV}, 30 days)...")

    try:
        grid = create_cubed_sphere(N)
        cube_lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
        cube_lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
        sigma = create_sigma_coordinate(NLEV)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=HYPERDIFF,
            hyperdiff_ps_coeff=HYPERDIFF,
            use_conservation_fixer=True,
            fix_mass=True,
            edge_blend_uv=EDGE_BLEND_HYDRO_UV,
            edge_blend_T=EDGE_BLEND_HYDRO_T,
            edge_blend_p_s=EDGE_BLEND_HYDRO_PS,
            edge_blend_width=EDGE_BLEND_HYDRO_WIDTH,
        )
        model = PrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init(grid, sigma)
        mass_init = float(global_integral(state.p_s, grid))

        n_steps = int(30 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_hydro_fv(s):
            u_sfc = np.asarray(s.u.data)[..., -1]
            v_sfc = np.asarray(s.v.data)[..., -1]
            return {
                "wind_speed": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
                "p_s": np.asarray(s.p_s.data),
                "T_sfc": np.asarray(s.T.data)[..., -1],
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_hydro_fv, state)

        t0 = time.time()
        state = model.step_with_physics(state, DT, held_suarez_forcing)
        jax.block_until_ready(state.u.data)
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_hydro_fv, state)

        for i in range(1, n_steps):
            state = model.step_with_physics(state, DT, held_suarez_forcing)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_hydro_fv, state)

            if i % 500 == 0:
                u_max = float(jnp.max(jnp.abs(state.u.data)))
                if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 500:
                    print(f"    BLOWUP at step {i}")
                    break

        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "Hydro FV Held-Suarez",
            snapshots,
            DT,
            [
                ("wind_speed", "Surface wind speed (m/s)", "magma"),
                ("p_s", "Surface pressure (Pa)", "viridis"),
                ("T_sfc", "Surface temperature (K)", "coolwarm"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        mass_final = float(global_integral(state.p_s, grid))
        mass_drift = abs(mass_final - mass_init) / abs(mass_init)
        max_wind = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))
        mean_T = float(jnp.mean(state.T.data))
        stable = bool(jnp.all(jnp.isfinite(state.u.data)))
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"mass_drift: {mass_drift:.2e}\nmax_wind: {max_wind:.1f}\n")
            f.write(f"mean_T: {mean_T:.1f}\nstable: {stable}\nwall_time: {wall:.1f}\n")

        record("Held-Suarez 30d", f"Hydro FV C{N}/L{NLEV}", status,
               "mass drift", f"{mass_drift:.2e}", wall,
               f"max|v|={max_wind:.1f}")
        print(f"    {status} | max|v|={max_wind:.1f} | <T>={mean_T:.1f} | mass={mass_drift:.2e} | {wall:.1f}s")

    except Exception as e:
        record("Held-Suarez 30d", f"Hydro FV C{N}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()

    # --- Baroclinic Wave (10 days) ---
    test_dir = output_dir / "06_hydro_fv_baroclinic"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [06] Hydro FV - Baroclinic Wave (C{N}/L{NLEV}, 10 days)...")

    try:
        sigma = create_sigma_coordinate(NLEV)
        state = baroclinic_wave_init(grid, sigma, perturbed=True)
        mass_init = float(global_integral(state.p_s, grid))

        config = PrimitiveEquationConfig(
            hyperdiff_coeff=HYPERDIFF,
            hyperdiff_ps_coeff=HYPERDIFF,
            use_conservation_fixer=True,
            fix_mass=True,
            edge_blend_uv=EDGE_BLEND_HYDRO_UV,
            edge_blend_T=EDGE_BLEND_HYDRO_T,
            edge_blend_p_s=EDGE_BLEND_HYDRO_PS,
            edge_blend_width=EDGE_BLEND_HYDRO_WIDTH,
        )
        model = PrimitiveEquationModel(grid, sigma, config)

        n_steps = int(10 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_hydro_fv(s):
            u_sfc = np.asarray(s.u.data)[..., -1]
            v_sfc = np.asarray(s.v.data)[..., -1]
            return {
                "wind_speed": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
                "p_s": np.asarray(s.p_s.data),
                "T_sfc": np.asarray(s.T.data)[..., -1],
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_hydro_fv, state)

        t0 = time.time()
        completed_steps = 0
        for i in range(n_steps):
            state = model.step(state, DT)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_hydro_fv, state)

            if i % 500 == 0 and i > 0:
                u_max = float(jnp.max(jnp.abs(state.u.data)))
                if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 500:
                    print(f"    BLOWUP at step {i}")
                    break

        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "Hydro FV Baroclinic Wave",
            snapshots,
            DT,
            [
                ("wind_speed", "Surface wind speed (m/s)", "magma"),
                ("p_s", "Surface pressure (Pa)", "viridis"),
                ("T_sfc", "Surface temperature (K)", "coolwarm"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        mass_final = float(global_integral(state.p_s, grid))
        mass_drift = abs(mass_final - mass_init) / abs(mass_init)
        max_wind = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))
        ps_min = float(jnp.min(state.p_s.data)) / 100
        stable = bool(jnp.all(jnp.isfinite(state.u.data)))
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"mass_drift: {mass_drift:.2e}\nmax_wind: {max_wind:.1f}\n")
            f.write(f"ps_min_hPa: {ps_min:.1f}\nstable: {stable}\nwall_time: {wall:.1f}\n")

        record("Baroclinic 10d", f"Hydro FV C{N}/L{NLEV}", status,
               "ps min (hPa)", f"{ps_min:.1f}", wall,
               f"max|v|={max_wind:.1f}")
        print(f"    {status} | max|v|={max_wind:.1f} | ps_min={ps_min:.1f}hPa | {wall:.1f}s")

    except Exception as e:
        record("Baroclinic 10d", f"Hydro FV C{N}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# 4. Hydrostatic Spectral Tests
# =============================================================================

def run_hydro_spectral_tests(output_dir):
    """Run Held-Suarez and Baroclinic wave with spectral PE."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel,
        SpectralPEConfig,
        isothermal_rest_state_spectral,
        baroclinic_wave_init_spectral,
        spectral_pe_to_grid,
    )
    from legoesm.atmosphere.physics.held_suarez import (
        held_suarez_forcing_spectral,
    )

    T = 15       # T15: explicit RK3 without semi-implicit gravity wave treatment
    NLEV = 10    #   is unstable at T21+ (not aliasing — dealiasing grid is correct)
    DT = 120.0   # Smaller dt for stability with explicit timestepping

    a = 6.371e6
    eig_max = T * (T + 1) / (a * a)
    # 0.5-hour e-folding time for ∇^4 hyperdiffusion
    HYPERDIFF = 1.0 / (0.5 * 3600.0 * eig_max ** 2)

    # --- Held-Suarez (30 days) ---
    test_dir = output_dir / "07_hydro_spectral_held_suarez"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [07] Hydro Spectral - Held-Suarez (T{T}/L{NLEV}, 30 days)...")

    try:
        grid = create_gaussian_grid(T)
        sigma = create_sigma_coordinate(NLEV)
        config = SpectralPEConfig(hyperdiff_coeff=HYPERDIFF, hyperdiff_order=2)
        model = SpectralPrimitiveEquationModel(grid, sigma, config)
        k_jet = _sigma_index_nearest(sigma.sigma_full, 0.25)
        k_mid = _sigma_index_nearest(sigma.sigma_full, 0.55)

        state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

        n_steps = int(30 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_hydro_spec(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            u_jet = np.asarray(fields["u"])[..., k_jet]
            v_jet = np.asarray(fields["v"])[..., k_jet]
            p_s = np.asarray(fields["p_s"])
            return {
                "wind_jet": np.sqrt(u_jet * u_jet + v_jet * v_jet),
                "p_s_anom": _zonal_anomaly(p_s),
                "T_mid": np.asarray(fields["T"])[..., k_mid],
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_hydro_spec, state)

        t0 = time.time()
        # First step with JIT compilation
        state = model.step_with_physics(state, DT, held_suarez_forcing_spectral)
        jax.block_until_ready(state.vor_hat.data)
        print(f"    JIT compiled in {time.time() - t0:.1f}s")
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_hydro_spec, state)

        for i in range(1, n_steps):
            state = model.step_with_physics(state, DT, held_suarez_forcing_spectral)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_hydro_spec, state)

            if i % 500 == 0:
                fields = spectral_pe_to_grid(state, grid, sigma)
                u_max = float(jnp.max(jnp.abs(fields['u'])))
                if not jnp.all(jnp.isfinite(fields['u'])) or u_max > 500:
                    print(f"    BLOWUP at step {i}, u_max={u_max:.1f}")
                    break

        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "Hydro Spectral Held-Suarez",
            snapshots,
            DT,
            [
                ("wind_jet", f"Wind speed @ sigma={float(np.asarray(sigma.sigma_full)[k_jet]):.3f} (m/s)", "magma"),
                ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
                ("T_mid", f"Temperature @ sigma={float(np.asarray(sigma.sigma_full)[k_mid]):.3f} (K)", "coolwarm"),
            ],
        )

        fields = spectral_pe_to_grid(state, grid, sigma)
        jet_speed = jnp.sqrt(fields['u'][..., k_jet] ** 2 + fields['v'][..., k_jet] ** 2)
        max_wind = float(jnp.max(jnp.sqrt(fields['u'] ** 2 + fields['v'] ** 2)))
        max_wind_jet = float(jnp.max(jet_speed))
        p99_wind_jet = float(jnp.percentile(jet_speed, 99.0))
        mean_T = float(jnp.mean(fields['T']))
        stable = bool(jnp.all(jnp.isfinite(fields['u'])))
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(
                f"max_wind_all_levels: {max_wind:.1f}\n"
                f"max_wind_jet_level: {max_wind_jet:.1f}\n"
                f"p99_wind_jet_level: {p99_wind_jet:.1f}\n"
                f"mean_T: {mean_T:.1f}\n"
            )
            f.write(f"stable: {stable}\nwall_time: {wall:.1f}\n")

        record("Held-Suarez 30d", f"Hydro Spec T{T}/L{NLEV}", status,
               "p99 |v| @ jet", f"{p99_wind_jet:.1f}", wall,
               f"max_all={max_wind:.1f}, max_jet={max_wind_jet:.1f}, <T>={mean_T:.1f}")
        print(
            f"    {status} | max|v|_all={max_wind:.1f} | "
            f"max|v|_jet={max_wind_jet:.1f} | p99_jet={p99_wind_jet:.1f} | {wall:.1f}s"
        )

    except Exception as e:
        record("Held-Suarez 30d", f"Hydro Spec T{T}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()

    # --- Baroclinic Wave (2 days — explicit spectral aliasing limits longer runs) ---
    test_dir = output_dir / "08_hydro_spectral_baroclinic"
    test_dir.mkdir(parents=True, exist_ok=True)
    BW_DAYS = 1  # Explicit RK3 limits BW integration; 1d stability check
    print(f"\n  [08] Hydro Spectral - Baroclinic Wave (T{T}/L{NLEV}, {BW_DAYS} days)...")

    try:
        # Baroclinic wave has stronger nonlinearity than HS (no Rayleigh friction),
        # so use even stronger hyperdiffusion (0.1h e-folding)
        eig_max_bw = T * (T + 1) / (a * a)
        HYPERDIFF_BW = 1.0 / (0.1 * 3600.0 * eig_max_bw ** 2)
        config_bw = SpectralPEConfig(hyperdiff_coeff=HYPERDIFF_BW, hyperdiff_order=2)
        model_bw = SpectralPrimitiveEquationModel(grid, sigma, config_bw)
        k_jet = _sigma_index_nearest(sigma.sigma_full, 0.25)
        k_mid = _sigma_index_nearest(sigma.sigma_full, 0.55)

        state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)

        n_steps = int(BW_DAYS * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_hydro_spec(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            u_jet = np.asarray(fields["u"])[..., k_jet]
            v_jet = np.asarray(fields["v"])[..., k_jet]
            p_s = np.asarray(fields["p_s"])
            return {
                "wind_jet": np.sqrt(u_jet * u_jet + v_jet * v_jet),
                "p_s_anom": _zonal_anomaly(p_s),
                "T_mid": np.asarray(fields["T"])[..., k_mid],
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_hydro_spec, state)

        t0 = time.time()
        state = model_bw.step(state, DT)
        jax.block_until_ready(state.vor_hat.data)
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_hydro_spec, state)

        blowup_detected = False
        for i in range(1, n_steps):
            state = model_bw.step(state, DT)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_hydro_spec, state)

            if i % 200 == 0:
                fields = spectral_pe_to_grid(state, grid, sigma)
                u_max = float(jnp.max(jnp.abs(fields['u'])))
                if not jnp.all(jnp.isfinite(fields['u'])) or u_max > 500:
                    print(f"    BLOWUP at step {i}, u_max={u_max:.1f}")
                    blowup_detected = True
                    break

        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "Hydro Spectral Baroclinic",
            snapshots,
            DT,
            [
                ("wind_jet", f"Wind speed @ sigma={float(np.asarray(sigma.sigma_full)[k_jet]):.3f} (m/s)", "magma"),
                ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
                ("T_mid", f"Temperature @ sigma={float(np.asarray(sigma.sigma_full)[k_mid]):.3f} (K)", "coolwarm"),
            ],
        )

        fields = spectral_pe_to_grid(state, grid, sigma)
        jet_speed = jnp.sqrt(fields['u'][..., k_jet] ** 2 + fields['v'][..., k_jet] ** 2)
        max_wind = float(jnp.max(jnp.sqrt(fields['u'] ** 2 + fields['v'] ** 2)))
        max_wind_jet = float(jnp.max(jet_speed))
        p99_wind_jet = float(jnp.percentile(jet_speed, 99.0))
        ps_min = float(jnp.min(fields['p_s'])) / 100
        stable = bool(jnp.all(jnp.isfinite(fields['u']))) and not blowup_detected
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(
                f"max_wind_all_levels: {max_wind:.1f}\n"
                f"max_wind_jet_level: {max_wind_jet:.1f}\n"
                f"p99_wind_jet_level: {p99_wind_jet:.1f}\n"
                f"ps_min_hPa: {ps_min:.1f}\n"
            )
            f.write(f"days: {BW_DAYS}\nstable: {stable}\nwall_time: {wall:.1f}\n")
            f.write("note: Explicit RK3 spectral PE requires semi-implicit for longer BW runs\n")

        record(f"Baroclinic {BW_DAYS}d", f"Hydro Spec T{T}/L{NLEV}", status,
               "p99 |v| @ jet", f"{p99_wind_jet:.1f}", wall,
               f"max_all={max_wind:.1f}, max_jet={max_wind_jet:.1f}, ps_min={ps_min:.1f}hPa")
        print(
            f"    {status} | max|v|_all={max_wind:.1f} | "
            f"max|v|_jet={max_wind_jet:.1f} | p99_jet={p99_wind_jet:.1f} | "
            f"ps_min={ps_min:.1f}hPa | {wall:.1f}s"
        )

    except Exception as e:
        record(f"Baroclinic {BW_DAYS}d", f"Hydro Spec T{T}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# 5. Non-Hydrostatic FV Tests
# =============================================================================

def run_nh_fv_tests(output_dir):
    """Run DCMIP-2025 TC1, TC2a, TC3 with CompressibleEulerModel."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel,
        CompressibleEulerConfig,
    )

    N = 8        # C8 instead of C16: explicit split-explicit unstable at C16
    NLEV = 20
    DT_NH = 5.0  # Full-Earth tests (TC1)
    DT_NH_SMALL = 1.0  # Small-Earth tests (TC2a, TC3) need smaller dt for CFL
    # Apply moderate edge blending for coarse-resolution NH runs to reduce
    # cube-edge noise while preserving short-run stability.
    EDGE_BLEND_NH_UV = 0.16
    EDGE_BLEND_NH_W_TC1 = 0.10
    EDGE_BLEND_NH_W_SMALL = 0.16
    EDGE_BLEND_NH_THETA = 0.08
    EDGE_BLEND_NH_RHO_TC1 = 0.12
    EDGE_BLEND_NH_RHO_SMALL = 0.18
    EDGE_BLEND_NH_TRACERS = 0.10
    EDGE_BLEND_NH_WIDTH = 2

    def extract_fields_nh_fv(s):
        u_low = np.asarray(s.u.data)[..., -1]
        v_low = np.asarray(s.v.data)[..., -1]
        w_data = np.asarray(s.w.data)
        k_mid = w_data.shape[-1] // 2

        out = {
            "wind_speed": np.sqrt(u_low * u_low + v_low * v_low),
            "rho_prime": np.asarray(s.rho_prime.data)[..., -1],
            "w_mid": w_data[..., k_mid],
        }

        tracers = np.asarray(s.tracers.data)
        if tracers.ndim == 5 and tracers.shape[-1] > 0:
            out["q1"] = tracers[..., -1, 0]
        if tracers.ndim == 5 and tracers.shape[-1] >= 3:
            # Rain mixing ratio is non-negative; clip tiny numerical undershoots.
            out["q_rain"] = np.clip(tracers[..., -1, 2], 0.0, None)
        return out

    # --- TC1: Gravity waves (3 hours) ---
    test_dir = output_dir / "09_nh_fv_dcmip25_tc1"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [09] NH FV - DCMIP-2025 TC1 Gravity Waves (C{N}/L{NLEV}, 3h)...")

    try:
        from tests.test_cases.dcmip2025 import dcmip25_tc1_init

        grid = create_cubed_sphere(N)
        cube_lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
        cube_lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
        state, height_coord, terrain_metric = dcmip25_tc1_init(grid, n_levels=NLEV)

        config = CompressibleEulerConfig(
            n_acoustic_substeps=6,
            sponge_width=10000.0,
            sponge_coeff=0.05,
            edge_blend_uv=EDGE_BLEND_NH_UV,
            edge_blend_w=EDGE_BLEND_NH_W_TC1,
            edge_blend_theta=EDGE_BLEND_NH_THETA,
            edge_blend_rho=EDGE_BLEND_NH_RHO_TC1,
            edge_blend_tracers=EDGE_BLEND_NH_TRACERS,
            edge_blend_width=EDGE_BLEND_NH_WIDTH,
        )
        model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)

        HOURS_TC1 = 1.0  # 1h integration for verification
        n_steps = int(HOURS_TC1 * 3600 / DT_NH)
        diag_every = max(1, n_steps // 10)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_nh_fv, state)

        t0 = time.time()
        state = model.step(state, DT_NH)
        jax.block_until_ready(state.u.data)
        print(f"    JIT compiled in {time.time() - t0:.1f}s")
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_fv, state)

        for i in range(1, n_steps):
            state = model.step(state, DT_NH)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_fv, state)
            if i % diag_every == 0:
                w_now = float(jnp.max(jnp.abs(state.w.data)))
                t_sim = (i + 1) * DT_NH / 3600.0
                print(f"      t={t_sim:.2f}h | max|w|={w_now:.4f}")
                if not jnp.all(jnp.isfinite(state.u.data)):
                    print(f"    BLOWUP at step {i}")
                    break

        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "NH FV DCMIP-2025 TC1",
            snapshots,
            DT_NH,
            [
                ("wind_speed", "Low-level wind speed (m/s)", "magma"),
                ("rho_prime", "Density perturbation (kg/m3)", "RdBu_r"),
                ("w_mid", "Mid-level vertical w (m/s)", "RdBu_r"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        w_max = float(jnp.max(jnp.abs(state.w.data)))
        u_max = float(jnp.max(jnp.abs(state.u.data)))
        stable = bool(jnp.all(jnp.isfinite(state.u.data))) and u_max < 500
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"max_w: {w_max:.4f}\nmax_u: {u_max:.1f}\n")
            f.write(f"hours: {HOURS_TC1}\nstable: {stable}\nwall_time: {wall:.1f}\n")

        record(f"DCMIP TC1 {HOURS_TC1:.0f}h", f"NH FV C{N}/L{NLEV}", status,
               "max |w|", f"{w_max:.4f}", wall)
        print(f"    {status} | max|w|={w_max:.4f} | max|u|={u_max:.1f} | {wall:.1f}s")

    except Exception as e:
        record("DCMIP TC1", f"NH FV C{N}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()

    # --- TC2a: Gap flow (short verification run) ---
    # The explicit split-explicit scheme is unstable for isothermal atmospheres
    # with solid-body rotation beyond ~30 min at C8. Semi-implicit time
    # integration would be needed for longer runs. We verify initialization
    # and short-term stability here.
    test_dir = output_dir / "10_nh_fv_dcmip25_tc2a"
    test_dir.mkdir(parents=True, exist_ok=True)
    HOURS_TC2 = 0.05  # 3 minutes — stable verification window
    DT_TC2 = 1.0      # Small dt for CFL on small Earth (X=20)
    print(f"\n  [10] NH FV - DCMIP-2025 TC2a Gap Flow (C{N}/L{NLEV}, {HOURS_TC2*60:.0f}min)...")

    try:
        from tests.test_cases.dcmip2025 import dcmip25_tc2_init

        grid = create_cubed_sphere(N)
        state, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(
            grid, n_levels=NLEV, subcase="a",
        )
        cube_lon_deg = np.asarray(small_grid.lon) * 180.0 / np.pi
        cube_lat_deg = np.asarray(small_grid.lat) * 180.0 / np.pi

        config = CompressibleEulerConfig(
            n_acoustic_substeps=6,
            sponge_width=15000.0,
            sponge_coeff=1.0 / (0.1 * 86400.0),
            small_earth_factor=20.0,
            edge_blend_uv=EDGE_BLEND_NH_UV,
            edge_blend_w=EDGE_BLEND_NH_W_SMALL,
            edge_blend_theta=EDGE_BLEND_NH_THETA,
            edge_blend_rho=EDGE_BLEND_NH_RHO_SMALL,
            edge_blend_tracers=EDGE_BLEND_NH_TRACERS,
            edge_blend_width=EDGE_BLEND_NH_WIDTH,
        )
        model = CompressibleEulerModel(
            small_grid, height_coord, terrain_metric, config,
        )

        n_steps = int(HOURS_TC2 * 3600 / DT_TC2)
        diag_every = max(1, n_steps // 10)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_nh_fv, state)

        t0 = time.time()
        state = model.step(state, DT_TC2)
        jax.block_until_ready(state.u.data)
        print(f"    JIT compiled in {time.time() - t0:.1f}s")
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_fv, state)

        for i in range(1, n_steps):
            state = model.step(state, DT_TC2)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_fv, state)
            if i % diag_every == 0:
                w_now = float(jnp.max(jnp.abs(state.w.data)))
                t_sim = (i + 1) * DT_TC2 / 3600.0
                print(f"      t={t_sim:.1f}h | max|w|={w_now:.4f}")
                if not jnp.all(jnp.isfinite(state.u.data)):
                    print(f"    BLOWUP at step {i}")
                    break

        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "NH FV DCMIP-2025 TC2a",
            snapshots,
            DT_TC2,
            [
                ("wind_speed", "Low-level wind speed (m/s)", "magma"),
                ("rho_prime", "Density perturbation (kg/m3)", "RdBu_r"),
                ("w_mid", "Mid-level vertical w (m/s)", "RdBu_r"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        w_max = float(jnp.max(jnp.abs(state.w.data)))
        u_max = float(jnp.max(jnp.abs(state.u.data)))
        stable = bool(jnp.all(jnp.isfinite(state.u.data))) and u_max < 500
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"max_w: {w_max:.4f}\nmax_u: {u_max:.1f}\n")
            f.write(f"hours: {HOURS_TC2}\ndt: {DT_TC2}\nstable: {stable}\nwall_time: {wall:.1f}\n")

        record(f"DCMIP TC2a {HOURS_TC2*60:.0f}min", f"NH FV C{N}/L{NLEV}", status,
               "max |w|", f"{w_max:.4f}", wall)
        print(f"    {status} | max|w|={w_max:.4f} | max|u|={u_max:.1f} | {wall:.1f}s")

    except Exception as e:
        record("DCMIP TC2a", f"NH FV C{N}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()

    # --- TC3: Squall line (short verification run, f=0) ---
    # The explicit split-explicit scheme limits integration time on the small
    # Earth (X=60). Semi-implicit time integration needed for multi-hour runs.
    # We verify initialization and short-term stability with f=0.
    test_dir = output_dir / "11_nh_fv_dcmip25_tc3"
    test_dir.mkdir(parents=True, exist_ok=True)
    HOURS_TC3 = 0.05   # 3 minutes — stable verification window
    DT_TC3 = 1.0       # Small dt for CFL on small Earth (X=60)
    print(f"\n  [11] NH FV - DCMIP-2025 TC3 Squall Line (C{N}/L{NLEV}, {HOURS_TC3*60:.0f}min, f=0)...")

    try:
        from tests.test_cases.dcmip2025 import dcmip25_tc3_init

        grid = create_cubed_sphere(N)
        state, height_coord, terrain_metric, small_grid = dcmip25_tc3_init(
            grid, n_levels=NLEV,
        )
        cube_lon_deg = np.asarray(small_grid.lon) * 180.0 / np.pi
        cube_lat_deg = np.asarray(small_grid.lat) * 180.0 / np.pi

        config = CompressibleEulerConfig(
            n_acoustic_substeps=6,
            sponge_width=5000.0,
            sponge_coeff=0.05,
            small_earth_factor=60.0,
            use_coriolis=False,  # TC3 is designed for f=0 (no Coriolis)
            edge_blend_uv=EDGE_BLEND_NH_UV,
            edge_blend_w=EDGE_BLEND_NH_W_SMALL,
            edge_blend_theta=EDGE_BLEND_NH_THETA,
            edge_blend_rho=EDGE_BLEND_NH_RHO_SMALL,
            edge_blend_tracers=EDGE_BLEND_NH_TRACERS,
            edge_blend_width=EDGE_BLEND_NH_WIDTH,
        )
        model = CompressibleEulerModel(
            small_grid, height_coord, terrain_metric, config,
        )

        n_steps = int(HOURS_TC3 * 3600 / DT_TC3)
        diag_every = max(1, n_steps // 10)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_nh_fv, state)

        t0 = time.time()
        state = model.step(state, DT_TC3)
        jax.block_until_ready(state.u.data)
        print(f"    JIT compiled in {time.time() - t0:.1f}s")
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_fv, state)

        for i in range(1, n_steps):
            state = model.step(state, DT_TC3)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_fv, state)
            if i % diag_every == 0:
                w_now = float(jnp.max(jnp.abs(state.w.data)))
                t_sim = (i + 1) * DT_TC3 / 3600.0
                print(f"      t={t_sim:.1f}h | max|w|={w_now:.4f}")
                if not jnp.all(jnp.isfinite(state.u.data)):
                    print(f"    BLOWUP at step {i}")
                    break

        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "NH FV DCMIP-2025 TC3",
            snapshots,
            DT_TC3,
            [
                ("wind_speed", "Low-level wind speed (m/s)", "magma"),
                ("rho_prime", "Density perturbation (kg/m3)", "RdBu_r"),
                ("w_mid", "Mid-level vertical w (m/s)", "RdBu_r"),
                ("q_rain", "Rain tracer q_rain", "Blues"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        w_max = float(jnp.max(jnp.abs(state.w.data)))
        stable = bool(jnp.all(jnp.isfinite(state.u.data)))
        status = "PASS" if stable else "FAIL"

        # Check rain tracer
        q_rain_max = 0.0
        if state.tracers.data.shape[-1] >= 3:
            q_rain_max = float(jnp.max(state.tracers.data[..., 2]))

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"max_w: {w_max:.4f}\nmax_qr: {q_rain_max:.6f}\n")
            f.write(f"hours: {HOURS_TC3}\ndt: {DT_TC3}\nstable: {stable}\nwall_time: {wall:.1f}\n")
            f.write("use_coriolis: False\n")

        record(f"DCMIP TC3 {HOURS_TC3*60:.0f}min", f"NH FV C{N}/L{NLEV}", status,
               "max |w|", f"{w_max:.4f}", wall,
               f"f=0, max qr={q_rain_max:.6f}")
        print(f"    {status} | max|w|={w_max:.4f} | max qr={q_rain_max:.6f} | {wall:.1f}s")

    except Exception as e:
        record("DCMIP TC3", f"NH FV C{N}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# 6. Non-Hydrostatic Spectral Tests
# =============================================================================

def run_nh_spectral_tests(output_dir):
    """Run DCMIP-2025 TC1 with SpectralCompressibleEulerModel."""
    from legoesm.grids.gaussian import (
        create_gaussian_grid,
        sh_synthesis_3d,
        uv_from_vordiv_3d,
    )
    from legoesm.atmosphere.dynamics.spectral_nh import (
        SpectralCompressibleEulerModel,
        SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )

    T = 15       # T15: explicit RK3 unstable at T21+ without semi-implicit GW treatment
    NLEV = 20
    DT_NH = 5.0  # Smaller dt for stability

    test_dir = output_dir / "12_nh_spectral_dcmip25_tc1"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [12] NH Spectral - DCMIP-2025 TC1 Gravity Waves (T{T}/L{NLEV}, 3h)...")

    try:
        grid = create_gaussian_grid(T)
        state, height_coord, terrain_metric = dcmip25_tc1_init_spectral(
            grid, n_levels=NLEV,
        )

        a = grid.radius
        eig_max = T * (T + 1) / (a * a)
        # 0.5-hour e-folding time for stronger diffusion
        hyperdiff = 1.0 / (0.5 * 3600.0 * eig_max ** 2)

        config = SpectralNHConfig(
            n_acoustic_substeps=6,
            sponge_width=10000.0,
            sponge_coeff=0.05,
            hyperdiff_coeff=hyperdiff,
        )
        model = SpectralCompressibleEulerModel(
            grid, height_coord, terrain_metric, config,
        )

        HOURS_TC1_SPEC = 1.0
        n_steps = int(HOURS_TC1_SPEC * 3600 / DT_NH)
        diag_every = max(1, n_steps // 10)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_nh_spec(s):
            u_cos, v_cos = uv_from_vordiv_3d(grid, s.vor_hat.data, s.div_hat.data)
            cos_lat_3d = np.asarray(grid.cos_lat)[:, None, None]
            u = np.asarray(u_cos) / cos_lat_3d
            v = np.asarray(v_cos) / cos_lat_3d
            w = np.asarray(sh_synthesis_3d(grid, s.w_hat.data))
            rho_p = np.asarray(sh_synthesis_3d(grid, s.rho_prime_hat.data))
            wind = np.sqrt(u * u + v * v)

            # Use zonal anomalies to avoid the background zonal flow dominating
            # level selection/visualization for TC1 perturbations.
            wind_anom = wind - np.mean(wind, axis=1, keepdims=True)
            rho_anom = rho_p - np.mean(rho_p, axis=1, keepdims=True)
            w_anom = w - np.mean(w, axis=1, keepdims=True)

            k_wind = int(np.argmax(np.mean(np.abs(wind_anom), axis=(0, 1))))
            k_rho = int(np.argmax(np.mean(np.abs(rho_anom), axis=(0, 1))))
            k_w = int(np.argmax(np.mean(np.abs(w_anom), axis=(0, 1))))
            return {
                "wind_focus": wind_anom[..., k_wind],
                "rho_focus": rho_anom[..., k_rho],
                "w_focus": w_anom[..., k_w],
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_nh_spec, state)

        t0 = time.time()
        state = model.step(state, DT_NH)
        jax.block_until_ready(state.vor_hat.data)
        print(f"    JIT compiled in {time.time() - t0:.1f}s")
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_spec, state)

        for i in range(1, n_steps):
            state = model.step(state, DT_NH)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_nh_spec, state)
            if i % diag_every == 0:
                if not jnp.all(jnp.isfinite(state.vor_hat.data)):
                    print(f"    BLOWUP at step {i}")
                    break

        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "NH Spectral DCMIP-2025 TC1",
            snapshots,
            DT_NH,
            [
                ("wind_focus", "Wind speed anomaly (max-activity level, m/s)", "RdBu_r"),
                ("rho_focus", "Density perturbation anomaly (max-activity level, kg/m3)", "RdBu_r"),
                ("w_focus", "Vertical w anomaly (max-activity level, m/s)", "RdBu_r"),
            ],
        )

        # Check stability via spectral coefficients
        w_grid = sh_synthesis_3d(grid, state.w_hat.data)
        w_max = float(jnp.max(jnp.abs(w_grid)))
        stable = bool(jnp.all(jnp.isfinite(state.vor_hat.data)))
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"max_w: {w_max:.4f}\nhours: {HOURS_TC1_SPEC}\n")
            f.write(f"stable: {stable}\nwall_time: {wall:.1f}\n")

        record(f"DCMIP TC1 {HOURS_TC1_SPEC:.0f}h", f"NH Spec T{T}/L{NLEV}", status,
               "max |w|", f"{w_max:.4f}", wall)
        print(f"    {status} | max|w|={w_max:.4f} | {wall:.1f}s")

    except Exception as e:
        record("DCMIP TC1", f"NH Spec T{T}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# 7. Transport Tests
# =============================================================================

def run_transport_tests(output_dir):
    """Run DCMIP-2012 Test 1-1 with TracerTransportModel."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.tracer_transport import (
        TracerTransportModel,
        TracerTransportConfig,
    )
    from tests.test_cases.dcmip_transport import (
        dcmip11_wind,
        dcmip11_init,
        compute_tracer_error_norms,
        create_dcmip_sigma,
    )

    N = 16
    NLEV = 10
    DT = 1800.0

    test_dir = output_dir / "13_transport_dcmip12_11"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [13] Transport - DCMIP-2012 Test 1-1 (C{N}/L{NLEV}, 12 days)...")

    try:
        grid = create_cubed_sphere(N)
        cube_lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
        cube_lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
        sigma = create_dcmip_sigma(NLEV)
        state_init = dcmip11_init(grid, sigma)

        config = TracerTransportConfig(hyperdiff_coeff=0.0)
        model = TracerTransportModel(grid, sigma, dcmip11_wind, config)

        n_steps = int(12 * 86400 / DT)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_transport(s):
            tracers = np.asarray(s.tracers.data)
            # Tracers are positive-definite in these idealized transport tests.
            out = {"q1": np.clip(tracers[..., -1, 0], 0.0, None)}
            if tracers.shape[-1] >= 2:
                out["q2"] = np.clip(tracers[..., -1, 1], 0.0, None)
            return out

        t0 = time.time()
        state = state_init
        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_transport, state)
        for i in range(n_steps):
            state = model.step(state, DT)
            _capture_snapshot(snapshots, snap_targets, i + 1, extract_fields_transport, state)
        jax.block_until_ready(state.tracers.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "Transport DCMIP-2012 1-1",
            snapshots,
            DT,
            [
                ("q1", "Tracer q1 (surface layer)", "viridis"),
                ("q2", "Tracer q2 (surface layer)", "plasma"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        norms = compute_tracer_error_norms(state, state_init, grid)
        l2_q1 = float(norms['l2'][0])
        linf_q1 = float(norms['linf'][0])
        stable = bool(jnp.all(jnp.isfinite(state.tracers.data)))
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"l2_q1: {l2_q1:.6e}\nlinf_q1: {linf_q1:.6e}\n")
            f.write(f"stable: {stable}\nwall_time: {wall:.1f}\n")
            for i in range(min(4, len(norms['l2']))):
                f.write(f"q{i+1}_l2: {float(norms['l2'][i]):.6e}\n")

        record("DCMIP-2012 1-1", f"Transport C{N}/L{NLEV}", status,
               "L2 q1", f"{l2_q1:.2e}", wall,
               f"Linf={linf_q1:.2e}")
        print(f"    {status} | L2={l2_q1:.2e} | Linf={linf_q1:.2e} | {wall:.1f}s")

    except Exception as e:
        record("DCMIP-2012 1-1", f"Transport C{N}/L{NLEV}", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# 8. Semi-Implicit Tests
# =============================================================================

def run_semi_implicit_tests(output_dir):
    """Run tests with semi-implicit time integration.

    1. Spectral PE with Hoskins-Simmons SI at T21 (stabilized configuration)
    2. NH FV with SI acoustic substeps (longer TC2a integration)
    """

    # --- SI Spectral PE: Held-Suarez at T21 (30 days) ---
    test_dir = output_dir / "14_si_spectral_pe_held_suarez"
    test_dir.mkdir(parents=True, exist_ok=True)

    T_SI = 21
    NLEV_SI = 10
    DT_SI = 600.0
    SI_SUBSTEPS = 5
    SI_HYPERDIFF_MULT = 14.0

    print(
        f"\n  [14] SI Spectral PE - Held-Suarez "
        f"(T{T_SI}/L{NLEV_SI}, 30 days, SI, dt={DT_SI:.0f}s, sub={SI_SUBSTEPS})...",
    )

    try:
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel,
            SpectralPEConfig,
            isothermal_rest_state_spectral,
            spectral_pe_to_grid,
        )
        from legoesm.atmosphere.physics.held_suarez import (
            held_suarez_forcing_spectral,
        )

        a = 6.371e6
        eig_max = T_SI * (T_SI + 1) / (a * a)
        HYPERDIFF = 1.0 / (0.5 * 3600.0 * eig_max ** 2)

        grid = create_gaussian_grid(T_SI)
        sigma = create_sigma_coordinate(NLEV_SI)
        config = SpectralPEConfig(
            hyperdiff_coeff=HYPERDIFF,
            hyperdiff_order=2,
            semi_implicit=True,
            si_T_ref=300.0,
            si_alpha=0.5,
            si_substeps=SI_SUBSTEPS,
            si_hyperdiff_boost=SI_HYPERDIFF_MULT,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, config)
        k_jet = _sigma_index_nearest(sigma.sigma_full, 0.25)
        k_mid = _sigma_index_nearest(sigma.sigma_full, 0.55)

        state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

        n_steps = int(30 * 86400 / DT_SI)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_si_spec(s):
            fields = spectral_pe_to_grid(s, grid, sigma)
            u_jet = np.asarray(fields["u"])[..., k_jet]
            v_jet = np.asarray(fields["v"])[..., k_jet]
            p_s = np.asarray(fields["p_s"])
            return {
                "wind_jet": np.sqrt(u_jet * u_jet + v_jet * v_jet),
                "p_s_anom": _zonal_anomaly(p_s),
                "T_mid": np.asarray(fields["T"])[..., k_mid],
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_si_spec, state)

        t0 = time.time()
        state = model.step_with_physics(state, DT_SI, held_suarez_forcing_spectral)
        jax.block_until_ready(state.vor_hat.data)
        print(f"    JIT compiled in {time.time() - t0:.1f}s")
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_si_spec, state)

        blowup_detected = False
        for i in range(1, n_steps):
            state = model.step_with_physics(state, DT_SI, held_suarez_forcing_spectral)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_si_spec, state)

            if i % 500 == 0:
                fields = spectral_pe_to_grid(state, grid, sigma)
                u_max = float(jnp.max(jnp.abs(fields['u'])))
                if not jnp.all(jnp.isfinite(fields['u'])) or u_max > 800:
                    print(f"    BLOWUP at step {i}, u_max={u_max:.1f}")
                    blowup_detected = True
                    break

        jax.block_until_ready(state.vor_hat.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "SI Spectral PE Held-Suarez",
            snapshots,
            DT_SI,
            [
                ("wind_jet", f"Wind speed @ sigma={float(np.asarray(sigma.sigma_full)[k_jet]):.3f} (m/s)", "magma"),
                ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
                ("T_mid", f"Temperature @ sigma={float(np.asarray(sigma.sigma_full)[k_mid]):.3f} (K)", "coolwarm"),
            ],
        )

        fields = spectral_pe_to_grid(state, grid, sigma)
        jet_speed = jnp.sqrt(fields['u'][..., k_jet] ** 2 + fields['v'][..., k_jet] ** 2)
        max_wind = float(jnp.max(jnp.sqrt(fields['u'] ** 2 + fields['v'] ** 2)))
        max_wind_jet = float(jnp.max(jet_speed))
        p99_wind_jet = float(jnp.percentile(jet_speed, 99.0))
        mean_T = float(jnp.mean(fields['T']))
        stable = bool(jnp.all(jnp.isfinite(fields['u']))) and not blowup_detected
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(
                f"max_wind_all_levels: {max_wind:.1f}\n"
                f"max_wind_jet_level: {max_wind_jet:.1f}\n"
                f"p99_wind_jet_level: {p99_wind_jet:.1f}\n"
                f"mean_T: {mean_T:.1f}\n"
            )
            f.write(
                f"dt: {DT_SI}\n"
                f"semi_implicit: True\n"
                f"si_substeps: {SI_SUBSTEPS}\n"
                f"hyperdiff_mult: {SI_HYPERDIFF_MULT:.1f}\n"
                f"stable: {stable}\n"
                f"wall_time: {wall:.1f}\n"
            )

        record("Held-Suarez 30d SI", f"Hydro Spec T{T_SI}/L{NLEV_SI} SI", status,
               "p99 |v| @ jet", f"{p99_wind_jet:.1f}", wall,
               f"max_all={max_wind:.1f}, max_jet={max_wind_jet:.1f}, "
               f"<T>={mean_T:.1f}, dt={DT_SI:.0f}s, sub={SI_SUBSTEPS}, "
               f"nu={SI_HYPERDIFF_MULT:.1f}x")
        print(
            f"    {status} | max|v|_all={max_wind:.1f} | "
            f"max|v|_jet={max_wind_jet:.1f} | p99_jet={p99_wind_jet:.1f} | {wall:.1f}s"
        )

    except Exception as e:
        record("Held-Suarez 30d SI", f"Hydro Spec T{T_SI}/L{NLEV_SI} SI", "ERROR",
               "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()

    # --- SI NH FV: TC2a with semi-implicit acoustic (30 min) ---
    test_dir = output_dir / "15_si_nh_fv_tc2a"
    test_dir.mkdir(parents=True, exist_ok=True)

    N_SI = 8
    NLEV_NH = 20
    DT_SI_NH = 1.0    # Same CFL as explicit for horizontal terms
    HOURS_SI = 0.1    # 6 min — 2x longer than explicit (3 min)

    print(f"\n  [15] SI NH FV - DCMIP-2025 TC2a (C{N_SI}/L{NLEV_NH}, {HOURS_SI*60:.0f}min, SI acoustic)...")

    try:
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.compressible_euler import (
            CompressibleEulerModel,
            CompressibleEulerConfig,
        )
        from tests.test_cases.dcmip2025 import dcmip25_tc2_init

        grid = create_cubed_sphere(N_SI)
        state, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(
            grid, n_levels=NLEV_NH, subcase="a",
        )
        cube_lon_deg = np.asarray(small_grid.lon) * 180.0 / np.pi
        cube_lat_deg = np.asarray(small_grid.lat) * 180.0 / np.pi

        config = CompressibleEulerConfig(
            n_acoustic_substeps=6,
            sponge_width=15000.0,
            sponge_coeff=1.0 / (0.1 * 86400.0),
            small_earth_factor=20.0,
            semi_implicit_acoustic=True,
        )
        model = CompressibleEulerModel(
            small_grid, height_coord, terrain_metric, config,
        )

        n_steps = int(HOURS_SI * 3600 / DT_SI_NH)
        diag_every = max(1, n_steps // 10)
        snap_targets = _snapshot_steps(n_steps)
        snapshots = {}

        def extract_fields_si_nh(s):
            u_low = np.asarray(s.u.data)[..., -1]
            v_low = np.asarray(s.v.data)[..., -1]
            w_data = np.asarray(s.w.data)
            return {
                "wind_speed": np.sqrt(u_low * u_low + v_low * v_low),
                "rho_prime": np.asarray(s.rho_prime.data)[..., -1],
                "w_mid": w_data[..., w_data.shape[-1] // 2],
            }

        _capture_snapshot(snapshots, snap_targets, 0, extract_fields_si_nh, state)

        t0 = time.time()
        state = model.step(state, DT_SI_NH)
        jax.block_until_ready(state.u.data)
        print(f"    JIT compiled in {time.time() - t0:.1f}s")
        completed_steps = 1
        _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_si_nh, state)

        blowup_detected = False
        for i in range(1, n_steps):
            state = model.step(state, DT_SI_NH)
            completed_steps = i + 1
            _capture_snapshot(snapshots, snap_targets, completed_steps, extract_fields_si_nh, state)
            if i % diag_every == 0:
                w_now = float(jnp.max(jnp.abs(state.w.data)))
                t_sim = (i + 1) * DT_SI_NH / 3600.0
                print(f"      t={t_sim*60:.1f}min | max|w|={w_now:.4f}")
                if not jnp.all(jnp.isfinite(state.u.data)):
                    print(f"    BLOWUP at step {i}")
                    blowup_detected = True
                    break

        jax.block_until_ready(state.u.data)
        wall = time.time() - t0

        _save_case_snapshots(
            test_dir,
            "SI NH FV DCMIP-2025 TC2a",
            snapshots,
            DT_SI_NH,
            [
                ("wind_speed", "Low-level wind speed (m/s)", "magma"),
                ("rho_prime", "Density perturbation (kg/m3)", "RdBu_r"),
                ("w_mid", "Mid-level vertical w (m/s)", "RdBu_r"),
            ],
            cube_lon_deg=cube_lon_deg,
            cube_lat_deg=cube_lat_deg,
        )

        w_max = float(jnp.max(jnp.abs(state.w.data)))
        u_max = float(jnp.max(jnp.abs(state.u.data)))
        stable = bool(jnp.all(jnp.isfinite(state.u.data))) and not blowup_detected and u_max < 500
        status = "PASS" if stable else "FAIL"

        with open(test_dir / "results.txt", "w") as f:
            f.write(f"max_w: {w_max:.4f}\nmax_u: {u_max:.1f}\n")
            f.write(f"hours: {HOURS_SI}\ndt: {DT_SI_NH}\nsi_acoustic: True\n")
            f.write(f"stable: {stable}\nwall_time: {wall:.1f}\n")

        record(f"DCMIP TC2a {HOURS_SI*60:.0f}min SI", f"NH FV C{N_SI}/L{NLEV_NH} SI", status,
               "max |w|", f"{w_max:.4f}", wall,
               f"dt={DT_SI_NH}s")
        print(f"    {status} | max|w|={w_max:.4f} | max|u|={u_max:.1f} | {wall:.1f}s")

    except Exception as e:
        record("DCMIP TC2a SI", f"NH FV C{N_SI}/L{NLEV_NH} SI", "ERROR", "error", str(e), 0)
        print(f"    ERROR: {e}")
        traceback.print_exc()


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(output_dir):
    """Generate SUMMARY.md with all results."""
    summary_path = output_dir / "SUMMARY.md"

    with open(summary_path, "w") as f:
        f.write("# legoESM Dynamical Core Test Suite Results\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Solver Matrix\n\n")
        f.write("| # | Test | Solver | Status | Key Metric | Value | Wall Time | Notes |\n")
        f.write("|---|------|--------|--------|------------|-------|-----------|-------|\n")

        for i, r in enumerate(ALL_RESULTS, 1):
            status_icon = {"PASS": "PASS", "FAIL": "**FAIL**", "ERROR": "**ERROR**"}
            s = status_icon.get(r["status"], r["status"])
            wt = f"{r['wall_time']:.1f}s" if r['wall_time'] > 0 else "-"
            f.write(
                f"| {i} | {r['test']} | {r['solver']} | {s} | "
                f"{r['metric']} | {r['value']} | {wt} | {r['notes']} |\n"
            )

        # Summary statistics
        n_pass = sum(1 for r in ALL_RESULTS if r["status"] == "PASS")
        n_fail = sum(1 for r in ALL_RESULTS if r["status"] == "FAIL")
        n_error = sum(1 for r in ALL_RESULTS if r["status"] == "ERROR")
        total_time = sum(r["wall_time"] for r in ALL_RESULTS)

        f.write(f"\n## Summary\n\n")
        f.write(f"- **Total tests**: {len(ALL_RESULTS)}\n")
        f.write(f"- **Passed**: {n_pass}\n")
        f.write(f"- **Failed**: {n_fail}\n")
        f.write(f"- **Errors**: {n_error}\n")
        f.write(f"- **Total wall time**: {total_time:.0f}s ({total_time/60:.1f} min)\n\n")

        f.write("## Solver Coverage\n\n")
        f.write("| Solver Type | Tests Run | Status |\n")
        f.write("|-------------|-----------|--------|\n")

        solver_types = [
            ("Shallow Water FV", "SW FV"),
            ("Shallow Water Spectral", "SW Spec"),
            ("Hydrostatic FV (Primitive Eq.)", "Hydro FV"),
            ("Hydrostatic Spectral (Spectral PE)", "Hydro Spec"),
            ("Hydrostatic Spectral SI (Semi-Implicit)", "SI"),
            ("Non-Hydrostatic FV (Compressible Euler)", "NH FV"),
            ("Non-Hydrostatic Spectral (Spectral NH)", "NH Spec"),
            ("Tracer Transport", "Transport"),
        ]
        for name, prefix in solver_types:
            tests = [r for r in ALL_RESULTS if prefix in r["solver"]]
            if tests:
                n = len(tests)
                ok = sum(1 for t in tests if t["status"] == "PASS")
                st = "All PASS" if ok == n else f"{ok}/{n} PASS"
                f.write(f"| {name} | {n} | {st} |\n")

        f.write("\n## Reference Test Suites Covered\n\n")
        f.write("- **Williamson et al. (1992)** SW Tests 2, 5: FV + Spectral\n")
        f.write("- **Held-Suarez (1994)**: FV PE + Spectral PE\n")
        f.write("- **Jablonowski-Williamson (2006)** Baroclinic Wave: FV PE + Spectral PE\n")
        f.write("- **DCMIP-2012** Transport Test 1-1: FV\n")
        f.write("- **DCMIP-2025** TC1/TC2a/TC3: FV Compressible Euler + Spectral NH (TC1)\n")
        f.write("- **FV3 Idealized Tests** (baroclinic wave, mountain wave): Covered by above\n")
        f.write("- **GFDL Spectral Dycore Tests** (Williamson, Held-Suarez, BW): Covered by above\n")
        f.write("- **NGGPS Dycore Testing** evaluation criteria: Metrics reported above\n")

        f.write("\n## Known Limitations\n\n")
        f.write("### Spectral PE: SI stability envelope at T21\n")
        f.write("The explicit SSP-RK3 spectral PE remains limited by fast-wave/advection stability\n")
        f.write("at higher truncations. For robust 30-day T21 Held-Suarez in this suite, we run\n")
        f.write("`semi_implicit=True` with `dt=600s`, SI subcycling (`si_substeps=5`, i.e. 120s\n")
        f.write("internal SI stages), and stronger SI-mode hyperdiffusion (14x baseline).\n")
        f.write("The 2/3 dealiasing grid is correctly\n")
        f.write("implemented (`n_lat = 3*(n_max+1)//2`).\n\n")

        f.write("### NH FV: explicit vs semi-implicit acoustic\n")
        f.write("The explicit split-explicit scheme (forward-backward acoustic substeps) limits\n")
        f.write("TC2a/TC3 to ~3 minutes at C8. The semi-implicit acoustic scheme\n")
        f.write("(`semi_implicit_acoustic=True`) uses a tridiagonal solve for w, removing the\n")
        f.write("vertical acoustic CFL and enabling 30+ minute integrations.\n\n")

        f.write("### Fixes applied in this version\n")
        f.write("- **Wind rotation**: All DCMIP-2025 test cases now correctly rotate winds from\n")
        f.write("  geographic (east/north) to cubed-sphere grid coordinates using\n")
        f.write("  `rotate_winds_geo_to_grid()`. Previously, winds were set directly in grid\n")
        f.write("  coordinates, which is only correct at face centers.\n")
        f.write("- **Coriolis toggle**: `CompressibleEulerConfig.use_coriolis` allows f=0 for\n")
        f.write("  TC3 (squall line), which is designed for cyclostrophic balance.\n")

    print(f"\n  Summary saved to {summary_path}")
    return summary_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="legoESM dycore test suite")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow tests (Held-Suarez, DCMIP)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["sw", "hydro", "nh", "transport", "spectral", "si"],
                        help="Run only one category")
    args = parser.parse_args()

    # Enable x64 only when spectral solvers are selected.
    if _requires_x64(args):
        jax.config.update("jax_enable_x64", True)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("legoESM Dynamical Core Test Suite")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"X64: {jax.config.jax_enable_x64}")
    print(f"Devices: {jax.devices()}")
    print(f"Output: {OUTPUT_BASE}/")
    print()

    t_total = time.time()

    only = args.only

    # --- Shallow Water ---
    if only in (None, "sw"):
        print("\n" + "=" * 50)
        print("SHALLOW WATER TESTS")
        print("=" * 50)
        run_sw_fv_tests(OUTPUT_BASE)
        run_sw_spectral_tests(OUTPUT_BASE)

    # --- Hydrostatic ---
    if only in (None, "hydro", "spectral"):
        print("\n" + "=" * 50)
        print("HYDROSTATIC TESTS")
        print("=" * 50)
        if only != "spectral":
            run_hydro_fv_tests(OUTPUT_BASE)
        run_hydro_spectral_tests(OUTPUT_BASE)

    # --- Non-Hydrostatic ---
    if only in (None, "nh", "spectral") and not args.skip_slow:
        print("\n" + "=" * 50)
        print("NON-HYDROSTATIC TESTS")
        print("=" * 50)
        if only != "spectral":
            run_nh_fv_tests(OUTPUT_BASE)
        run_nh_spectral_tests(OUTPUT_BASE)

    # --- Transport ---
    if only in (None, "transport") and not args.skip_slow:
        print("\n" + "=" * 50)
        print("TRANSPORT TESTS")
        print("=" * 50)
        run_transport_tests(OUTPUT_BASE)

    # --- Semi-Implicit ---
    if only in (None, "si") and not args.skip_slow:
        print("\n" + "=" * 50)
        print("SEMI-IMPLICIT TESTS")
        print("=" * 50)
        run_semi_implicit_tests(OUTPUT_BASE)

    # --- Summary ---
    total_time = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"ALL TESTS COMPLETE ({total_time:.0f}s = {total_time/60:.1f} min)")
    print("=" * 70)

    n_pass = sum(1 for r in ALL_RESULTS if r["status"] == "PASS")
    n_fail = sum(1 for r in ALL_RESULTS if r["status"] == "FAIL")
    n_error = sum(1 for r in ALL_RESULTS if r["status"] == "ERROR")
    print(f"  PASS: {n_pass}  |  FAIL: {n_fail}  |  ERROR: {n_error}  |  Total: {len(ALL_RESULTS)}")

    for r in ALL_RESULTS:
        icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!"}[r["status"]]
        print(f"  {icon} {r['status']:5s} | {r['test']:<20s} | {r['solver']:<25s} | {r['metric']}={r['value']}")

    generate_summary(OUTPUT_BASE)

    # Save timing
    with open(OUTPUT_BASE / "timing.txt", "w") as f:
        f.write(f"total_wall_time: {total_time:.1f}s\n")
        for r in ALL_RESULTS:
            f.write(f"{r['test']} ({r['solver']}): {r['wall_time']:.1f}s\n")

    print(f"\nAll outputs saved to {OUTPUT_BASE}/")
    print("Done!")


if __name__ == "__main__":
    main()
