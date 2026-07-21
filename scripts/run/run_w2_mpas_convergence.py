#!/usr/bin/env python3
"""Williamson Test Case 2 convergence benchmark on MPAS Voronoi meshes.

Runs TC2 (steady geostrophic flow) for 1 day at mesh levels 3, 4, 5
(642, 2562, 10242 cells) and prints L2 / Linf error norms and
convergence rates.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python3.14 scripts/run_w2_mpas_convergence.py
"""
from __future__ import annotations

import sys
import os
import time
import math

# Ensure project root is on sys.path so tests can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
    MPASShallowWaterModel,
    MPASShallowWaterConfig,
)
from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
    williamson_test2_mpas,
    compute_error_norms_mpas,
)


def run_tc2(level: int, duration_days: float = 1.0, lloyd_iters: int = 50):
    """Run TC2 at a given mesh level and return error norms."""
    print(f"\n{'='*60}")
    print(f"Level {level}: generating mesh (lloyd_iterations={lloyd_iters}) ...")
    t0 = time.time()
    mesh = create_voronoi_mesh(level, lloyd_iterations=lloyd_iters)
    print(f"  nCells={mesh.nCells}, nEdges={mesh.nEdges}, nVertices={mesh.nVertices}")
    print(f"  Mesh generation: {time.time()-t0:.1f}s")

    # Compute characteristic cell spacing
    dx_min = float(jnp.min(mesh.dcEdge))
    dx_mean = float(jnp.mean(mesh.dcEdge))
    print(f"  dx_min={dx_min/1e3:.1f}km, dx_mean={dx_mean/1e3:.1f}km")

    # Configuration: no diffusion, energy-conserving PV
    config = MPASShallowWaterConfig(
        pv_scheme="energy",
        fix_mass=True,
        fix_energy=False,
        time_integrator="rk4",
    )
    model = MPASShallowWaterModel(mesh, config)

    # Initial condition (also the analytic solution for TC2)
    state = williamson_test2_mpas(mesh)
    h_ref = state.h.data.copy()

    # Time stepping: CFL-based dt
    u_max = 40.0  # ~38.6 m/s
    dt = 0.5 * dx_min / u_max
    dt = min(dt, 1200.0)  # cap at 20 min
    nsteps = int(duration_days * 86400.0 / dt)
    dt = duration_days * 86400.0 / nsteps  # adjust for exact duration
    print(f"  dt={dt:.1f}s, nsteps={nsteps}")

    # Integrate
    t0 = time.time()
    s = state
    # JIT-compile first step
    s = model.step(s, dt)
    t_jit = time.time() - t0
    print(f"  JIT compile: {t_jit:.1f}s")

    t0 = time.time()
    for _ in range(nsteps - 1):
        s = model.step(s, dt)
    t_run = time.time() - t0
    print(f"  Integration: {t_run:.1f}s ({(t_run)/(nsteps-1)*1e3:.1f}ms/step)")

    # Error norms
    norms = compute_error_norms_mpas(s.h.data, h_ref, mesh)
    h_mean = float(jnp.mean(h_ref))
    print(f"  L2  = {norms['l2']:.6e}")
    print(f"  Linf= {norms['linf']:.6e}  ({norms['linf']*h_mean:.2f}m)")

    return {
        "level": level,
        "nCells": mesh.nCells,
        "dx_mean": dx_mean,
        "l2": norms["l2"],
        "linf": norms["linf"],
    }


def main():
    levels = [3, 4, 5]
    results = []

    for level in levels:
        res = run_tc2(level, duration_days=1.0)
        results.append(res)

    # Print convergence table
    print(f"\n{'='*60}")
    print("W2 Convergence Summary (1-day integration, no diffusion)")
    print(f"{'='*60}")
    print(f"{'Level':>6} {'nCells':>8} {'dx(km)':>8} {'L2':>12} {'Linf':>12} {'L2 rate':>8} {'Linf rate':>10}")
    print("-" * 74)

    for i, r in enumerate(results):
        dx_km = r["dx_mean"] / 1e3
        l2_rate = ""
        linf_rate = ""
        if i > 0:
            prev = results[i - 1]
            dx_ratio = prev["dx_mean"] / r["dx_mean"]
            l2_r = math.log(prev["l2"] / r["l2"]) / math.log(dx_ratio)
            linf_r = math.log(prev["linf"] / r["linf"]) / math.log(dx_ratio)
            l2_rate = f"{l2_r:.2f}"
            linf_rate = f"{linf_r:.2f}"

        print(f"{r['level']:>6} {r['nCells']:>8} {dx_km:>8.1f} "
              f"{r['l2']:>12.4e} {r['linf']:>12.4e} "
              f"{l2_rate:>8} {linf_rate:>10}")

    print(f"{'='*60}")
    print("Expected 2nd-order convergence rate: ~2.0")


if __name__ == "__main__":
    main()
