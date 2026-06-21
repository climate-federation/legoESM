#!/usr/bin/env python
"""Diagnose barotropic-mode grid-scale noise in MPAS overturning restarts.

Implements the diagnostic plan from ``docs/dev-notes/issues/barotropic_mode_noise.md``
on the MPAS Voronoi C-grid.  The lat-lon C-grid suffers from a Coriolis-
averaging null mode that contaminates time-mean V_baro with grid-scale
±5 cm/s noise; the analogous TRiSK rotational null branch on hexagonal
C-grids (Thuburn 2008; Ringler et al. 2010) is hypothesised to affect
the MPAS port.  This script computes from each restart snapshot:

  * |u_bar|_max, |u_bar|_rms                — overall velocity scale
  * sigma(vector_laplacian_del2(u_bar))     — grid-scale noise scale
  * grid-scale noise / signal ratio         — proxy for Crit 1
  * |⟨v_t(u_bar)⟩|_max off polar caps       — Crit 1.2 analog
  * cross-restart time-mean |u_bar|         — noise should average down,
                                              residual = real signal +
                                              non-zero-mean noise mode

Snapshots only (not time-averaged Hu fields), so this is necessary-but-
not-sufficient.  Strong grid-scale noise here ⇒ M2 (u_bar viscosity)
and/or M3 (implicit CN) likely needed.  Weak noise ⇒ either the
hypothesis was wrong, or the noise has cancelled at this resolution.

Usage:
    python scripts/run/global_overturning/diagnose_mpas_baro_noise.py \
        [results/ocean/global_overturning_mpas_baseline]

Default scans all restart_day*.npz in the baseline output directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Suppress JAX device warnings for this CPU-only diagnostic
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.core.operators_voronoi import (
    edge_thickness,
    vector_laplacian_del2,
    tangential_velocity,
)


def _u_bar_from_state(u_3d: np.ndarray, eta: np.ndarray, H_bathy: np.ndarray,
                      mesh, z_coord, min_water_col: float = 0.5) -> np.ndarray:
    """Depth-mean velocity at edges from a snapshot."""
    u_3d_j = jnp.asarray(u_3d, dtype=jnp.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    H_j = jnp.asarray(H_bathy, dtype=jnp.float64)

    h_k = compute_layer_thickness(eta_j, H_j, z_coord, min_water_column_m=min_water_col)
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    h_e_k = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
    Hu = jnp.sum(u_3d_j * h_e_k, axis=1)
    H_total = jnp.maximum(eta_j + H_j, min_water_col)
    H_e = edge_thickness(H_total, mesh)
    u_bar = Hu / jnp.maximum(H_e, 1e-10)
    return np.asarray(u_bar)


def _polar_edge_mask(mesh, polar_lat_deg: float = 70.0) -> np.ndarray:
    """True if either cell touching this edge is poleward of the cap."""
    lat_cell = np.asarray(mesh.latCell) * 180.0 / np.pi
    polar_cell = np.abs(lat_cell) > polar_lat_deg
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    return polar_cell[c1] | polar_cell[c2]


def _ocean_edge_mask(mesh, land_mask: np.ndarray) -> np.ndarray:
    """True if both cells touching this edge are wet."""
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    return (land_mask[c1] > 0.5) & (land_mask[c2] > 0.5)


def _diagnose_one(npz_path: Path, mesh, z_coord, min_water_col: float = 0.5):
    """Return a metrics dict for a single restart NPZ."""
    d = np.load(npz_path, allow_pickle=False)
    day = float(d["time_days"])

    u_bar = _u_bar_from_state(
        d["u"], d["eta"], d["H_bathy"], mesh, z_coord, min_water_col,
    )

    # Tangential reconstruction: gives meridional component when the
    # edge is roughly zonal (and vice-versa).  Mostly diagnostic of the
    # rotational null branch which lives in the curl_vertex kernel.
    u_bar_j = jnp.asarray(u_bar, dtype=jnp.float64)
    v_t = np.asarray(tangential_velocity(u_bar_j, mesh))

    # Grid-scale Laplacian of u_bar — proxy for the TRiSK null branch.
    lap = np.asarray(vector_laplacian_del2(u_bar_j, mesh))
    # The Laplacian has units 1/m² · (m/s); rescale by mean dvEdge² so
    # the metric has units of m/s and is comparable to u_bar itself.
    dv = np.asarray(mesh.dvEdge)
    lap_scaled = lap * np.mean(dv) ** 2

    ocean = _ocean_edge_mask(mesh, np.asarray(d["land_mask"]))
    polar = _polar_edge_mask(mesh, polar_lat_deg=70.0)
    interior = ocean & ~polar

    if not interior.any():
        return None

    u_bar_int = u_bar[interior]
    v_t_int = v_t[interior]
    lap_int = lap_scaled[interior]

    rms_u = float(np.sqrt(np.mean(u_bar_int**2)))
    max_u = float(np.max(np.abs(u_bar_int)))
    sigma_grid = float(np.std(lap_int))
    max_vt = float(np.max(np.abs(v_t_int)))
    grid_signal_ratio = sigma_grid / max(rms_u, 1.0e-30)

    return {
        "day": day,
        "rms_u_bar": rms_u,
        "max_abs_u_bar": max_u,
        "sigma_grid_scale": sigma_grid,
        "grid_to_signal_ratio": grid_signal_ratio,
        "max_abs_v_t_off_polar": max_vt,
        "u_bar": u_bar,
        "ocean_edge_mask": ocean,
    }


def _interpret(metrics: list[dict]) -> str:
    """Heuristic interpretation against Crit 1.3 thresholds."""
    if not metrics:
        return "No metrics"
    last = metrics[-1]
    ratio = last["grid_to_signal_ratio"]
    sig_grid = last["sigma_grid_scale"]
    msg = []
    if ratio > 0.30:
        msg.append(
            f"  ⚠ Grid-scale-to-signal ratio = {ratio:.2f} is HIGH — "
            f"strong evidence of TRiSK rotational null-branch noise.")
    elif ratio > 0.10:
        msg.append(
            f"  ⚠ Grid-scale-to-signal ratio = {ratio:.2f} is MODERATE — "
            f"some grid-scale noise present.")
    else:
        msg.append(
            f"  ✓ Grid-scale-to-signal ratio = {ratio:.2f} is LOW — "
            f"u_bar field looks clean.")

    if sig_grid > 1.0e-2:
        msg.append(
            f"  ⚠ σ(grid-scale u_bar) = {sig_grid:.2e} m/s exceeds the "
            f"lat-lon Crit 1.3 threshold of 1e-2 m/s.")
    else:
        msg.append(
            f"  ✓ σ(grid-scale u_bar) = {sig_grid:.2e} m/s within "
            f"Crit 1.3 threshold (1e-2 m/s).")
    return "\n".join(msg)


def _time_mean_metrics(metrics: list[dict], mesh) -> dict:
    """Time-mean over restarts (skipping day-0 rest state)."""
    runs = [m for m in metrics if m["day"] > 0]
    if not runs:
        return None
    # Element-wise time mean across snapshots — noise should partially
    # cancel (random sign), residual is real signal + DC component of
    # the null mode.
    u_bar_stack = np.stack([m["u_bar"] for m in runs], axis=0)
    u_bar_mean = u_bar_stack.mean(axis=0)
    ocean = runs[0]["ocean_edge_mask"]
    polar = _polar_edge_mask(mesh, polar_lat_deg=70.0)
    interior = ocean & ~polar

    u_bar_j = jnp.asarray(u_bar_mean, dtype=jnp.float64)
    lap = np.asarray(vector_laplacian_del2(u_bar_j, mesh))
    dv = np.asarray(mesh.dvEdge)
    lap_scaled = lap * np.mean(dv) ** 2
    v_t = np.asarray(tangential_velocity(u_bar_j, mesh))

    rms_u = float(np.sqrt(np.mean(u_bar_mean[interior]**2)))
    sigma_grid = float(np.std(lap_scaled[interior]))
    max_vt = float(np.max(np.abs(v_t[interior])))
    return {
        "n_snapshots": len(runs),
        "rms_u_bar_timemean": rms_u,
        "sigma_grid_timemean": sigma_grid,
        "max_abs_v_t_timemean": max_vt,
        "grid_to_signal_ratio_timemean": sigma_grid / max(rms_u, 1.0e-30),
    }


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = Path(
            "results/ocean/global_overturning_mpas_baseline")

    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    restarts = sorted(run_dir.glob("restart_day*.npz"))
    if not restarts:
        print(f"No restart_day*.npz in {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Reconstruct the mesh + z_coord from the run config.  These match
    # ``run_global_overturning_mpas_baseline.py``: ico4 mesh, 20 levels,
    # 5500 m max depth.
    sample = np.load(restarts[0], allow_pickle=False)
    sub_level = int(sample["mpas_subdivision_level"])
    nlev = int(sample["u"].shape[1])
    H_max = float(np.max(sample["H_bathy"]))

    print(f"Loading mesh: ico{sub_level} ({sample['u'].shape[0]} edges, "
          f"{sample['eta'].shape[0]} cells)")
    print(f"Vertical: {nlev} levels, H_max={H_max:.0f} m")
    print()

    mesh = create_voronoi_mesh(subdivision_level=sub_level)
    # The defaults match the GlobalOverturningConfig used in the run
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=H_max, dz_surface=10.0, dz_deep=300.0,
    )

    metrics = []
    print(f"{'day':>6} {'year':>6} {'rms u_bar':>11} "
          f"{'σ_grid':>11} {'σ/rms':>8} {'|v_t|max':>10}")
    print("-" * 60)
    for r in restarts:
        m = _diagnose_one(r, mesh, z_coord)
        if m is None:
            continue
        metrics.append(m)
        print(f"{int(m['day']):>6d} {m['day']/365.25:>6.2f} "
              f"{m['rms_u_bar']:>11.3e} {m['sigma_grid_scale']:>11.3e} "
              f"{m['grid_to_signal_ratio']:>8.3f} "
              f"{m['max_abs_v_t_off_polar']:>10.3e}")

    print()
    print("Time-mean across all post-spinup restarts:")
    tm = _time_mean_metrics(metrics, mesh)
    if tm is not None:
        print(f"  n_snapshots: {tm['n_snapshots']}")
        print(f"  rms(u_bar)_t: {tm['rms_u_bar_timemean']:.3e} m/s")
        print(f"  σ(grid u_bar)_t: {tm['sigma_grid_timemean']:.3e} m/s")
        print(f"  max|v_t(u_bar)|_t off polar: "
              f"{tm['max_abs_v_t_timemean']:.3e} m/s")
        print(f"  σ/rms ratio (time-mean): "
              f"{tm['grid_to_signal_ratio_timemean']:.3f}")

    print()
    print("Interpretation (heuristic vs lat-lon Crit 1.3 thresholds):")
    print(_interpret(metrics))
    print()
    print("Notes:")
    print("  * σ_grid is the standard deviation of the (rescaled) "
          "vector Laplacian of u_bar")
    print("    on interior edges (off polar caps).  A clean field has "
          "σ_grid ≪ rms(u_bar);")
    print("    a noisy null-branch contaminated field has σ_grid ~ rms.")
    print("  * v_t = tangential reconstruction; gives meridional "
          "component for zonal edges.")
    print("  * Snapshots only — Drake-band closure metric needs "
          "time-averaged Hu fields, not")
    print("    snapshots, and would need a 1-yr re-run with in-loop "
          "averaging to compute.")


if __name__ == "__main__":
    main()
