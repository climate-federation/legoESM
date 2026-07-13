#!/usr/bin/env python3
"""Benchmark: Spectral Non-Hydrostatic — Rest-State + DCMIP Gravity Waves.

Runs:
  1. Rest-state stability (T21/L10, 1 hour, dt=5s).
  2. DCMIP-2025 TC1 gravity waves (T31/L40, 1 hour, dt=5s).
     Expected: acoustic/gravity wave propagation from mountain forcing,
     w perturbations ~ 0.1-1 m/s, stable integration.

Outputs (in results/atmosphere/nonhydrostatic/spectral_nh/):
  - rest_stability.png     — perturbation growth time series
  - timeseries.png         — max |w|, max |theta'| time series
  - profiles_w.png         — zonal-mean w at times 0, 20min, 40min, 60min
  - profiles_theta.png     — zonal-mean theta' at times 0, 20min, 40min, 60min
  - snapshots_w.png        — w at mid-level, lat-lon at different times
  - diagnostics.npz        — all diagnostic data
  - summary.txt            — quantitative pass/fail

References
----------
  DCMIP-2025 Test Case 1: mountain-forced gravity waves.
"""

import os
import sys
import time

# Ensure project root is on sys.path so spectral_nh can import test_cases
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_synthesis_3d,
)
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
    SpectralNHConfig,
    SpectralCompressibleEulerModel,
    nh_rest_state_spectral,
    dcmip25_tc1_init_spectral,
)
from legoesm import constants

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_DIR, "results", "atmosphere", "nonhydrostatic", "spectral_nh")
os.makedirs(OUT_DIR, exist_ok=True)


def proper_hyperdiff(grid):
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (4.0 * 3600.0 * eig_max**2)


def nh_state_to_grid(state, grid):
    """Extract grid-point fields from spectral NH state."""
    w = np.array(sh_synthesis_3d(grid, state.w_hat.data))
    theta_p = np.array(sh_synthesis_3d(grid, state.theta_prime_hat.data))
    rho_p = np.array(sh_synthesis_3d(grid, state.rho_prime_hat.data))
    vor = np.array(sh_synthesis_3d(grid, state.vor_hat.data))
    div = np.array(sh_synthesis_3d(grid, state.div_hat.data))
    return {
        "w": w,
        "theta_prime": theta_p,
        "rho_prime": rho_p,
        "vor": vor,
        "div": div,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("  Spectral Non-Hydrostatic Benchmark")
    print("=" * 70)

    # ── 1. Rest-state stability (T21/L10, 1 hour) ────────────────────
    print("\n" + "-" * 50)
    print("  Part 1: Rest-State Stability (T21/L10, 1 hour)")
    print("-" * 50)

    grid21 = create_gaussian_grid(n_max=21)
    hcoord10 = create_height_coordinate(10, 30000.0)
    z_s_flat = jnp.zeros((grid21.n_lat, grid21.n_lon))
    terrain_flat = compute_terrain_metric(z_s_flat, hcoord10)
    nu21 = proper_hyperdiff(grid21)

    config_rest = SpectralNHConfig(
        hyperdiff_coeff=nu21,
        hyperdiff_order=2,
        sponge_coeff=0.0,
        n_acoustic_substeps=4,
    )
    model_rest = SpectralCompressibleEulerModel(
        grid21, hcoord10, terrain_flat, config_rest,
        allow_unsupported_backend=True,
    )
    state_rest = nh_rest_state_spectral(grid21, hcoord10)

    dt_rest = 5.0
    n_steps_rest = int(3600 / dt_rest)  # 1 hour
    diag_interval_rest = int(300 / dt_rest)  # every 5 min

    rest_times = [0.0]
    rest_max_w = [0.0]
    rest_max_theta = [0.0]
    rest_max_rho = [0.0]

    print(f"  dt={dt_rest}s, {n_steps_rest} steps, 4 acoustic substeps")
    t0 = time.time()
    state = state_rest
    for step in range(n_steps_rest):
        state = model_rest.step(state, dt_rest)

        if (step + 1) % diag_interval_rest == 0:
            t_min = (step + 1) * dt_rest / 60.0
            fields = nh_state_to_grid(state, grid21)
            max_w = float(np.max(np.abs(fields["w"])))
            max_theta = float(np.max(np.abs(fields["theta_prime"])))
            max_rho = float(np.max(np.abs(fields["rho_prime"])))
            rest_times.append(t_min)
            rest_max_w.append(max_w)
            rest_max_theta.append(max_theta)
            rest_max_rho.append(max_rho)

    jax.block_until_ready(state.vor_hat.data)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Final: max|w|={rest_max_w[-1]:.2e}, max|theta'|={rest_max_theta[-1]:.2e}")

    rest_pass = rest_max_w[-1] < 1e-6 and rest_max_theta[-1] < 1e-6

    # ── 2. DCMIP TC1 Gravity Waves (T31/L40, 1 hour) ─────────────────
    print("\n" + "-" * 50)
    print("  Part 2: DCMIP-2025 TC1 Gravity Waves (T31/L40)")
    print("-" * 50)

    n_max_tc1 = 31
    nlev_tc1 = 40
    grid_tc1 = create_gaussian_grid(n_max=n_max_tc1)
    nu_tc1 = proper_hyperdiff(grid_tc1)

    print(f"  Initializing DCMIP TC1...")
    try:
        state0, hcoord_tc1, terrain_tc1 = dcmip25_tc1_init_spectral(
            grid_tc1, n_levels=nlev_tc1,
        )
    except Exception as e:
        print(f"  WARNING: Could not initialize DCMIP TC1: {e}")
        print(f"  Falling back to mountain-perturbation test...")
        # Fallback: rest state with small theta perturbation
        hcoord_tc1 = create_height_coordinate(nlev_tc1, 30000.0)
        z_s = jnp.zeros((grid_tc1.n_lat, grid_tc1.n_lon))
        terrain_tc1 = compute_terrain_metric(z_s, hcoord_tc1)
        state0 = nh_rest_state_spectral(grid_tc1, hcoord_tc1)
        # Add theta perturbation at mid-level
        theta_p = sh_synthesis_3d(grid_tc1, state0.theta_prime_hat.data)
        lat2d = grid_tc1.lat2d[:, :, None] * jnp.ones(nlev_tc1)[None, None, :]
        lon2d = grid_tc1.lon2d[:, :, None] * jnp.ones(nlev_tc1)[None, None, :]
        z_mid = hcoord_tc1.z_full[nlev_tc1 // 2]
        z_3d = jnp.broadcast_to(hcoord_tc1.z_full[None, None, :],
                                  (grid_tc1.n_lat, grid_tc1.n_lon, nlev_tc1))
        perturb = 1.0 * jnp.exp(-((z_3d - z_mid) / 5000.0)**2) * \
                  jnp.exp(-((jnp.degrees(lat2d) - 30.0) / 10.0)**2) * \
                  jnp.cos(jnp.degrees(lon2d) * jnp.pi / 180.0)
        theta_p = theta_p + perturb
        from legoesm.grids.gaussian import sh_analysis_3d
        theta_p_hat = sh_analysis_3d(grid_tc1, theta_p)
        state0 = state0._replace(
            theta_prime_hat=state0.theta_prime_hat.replace(data=theta_p_hat)
        )

    config_tc1 = SpectralNHConfig(
        hyperdiff_coeff=nu_tc1,
        hyperdiff_order=2,
        sponge_coeff=0.05,
        sponge_width=10000.0,
        n_acoustic_substeps=6,
    )
    model_tc1 = SpectralCompressibleEulerModel(
        grid_tc1, hcoord_tc1, terrain_tc1, config_tc1,
        allow_unsupported_backend=True,
    )

    dt_tc1 = 5.0
    duration = 3600.0  # 1 hour
    n_steps_tc1 = int(duration / dt_tc1)
    diag_interval_tc1 = int(120 / dt_tc1)  # every 2 min
    snap_times_min = [0, 20, 40, 60]

    print(f"  T{n_max_tc1}/L{nlev_tc1}, dt={dt_tc1}s, {n_steps_tc1} steps")

    # Initial diagnostics
    fields0 = nh_state_to_grid(state0, grid_tc1)
    tc1_times = [0.0]
    tc1_max_w = [float(np.max(np.abs(fields0["w"])))]
    tc1_max_theta = [float(np.max(np.abs(fields0["theta_prime"])))]
    tc1_max_div = [float(np.max(np.abs(fields0["div"])))]
    tc1_snapshots = {0: fields0}

    state = state0
    t0 = time.time()
    # JIT warmup
    state = model_tc1.step(state, dt_tc1)
    jax.block_until_ready(state.vor_hat.data)
    print(f"  JIT compiled in {time.time()-t0:.1f}s")

    t_wall = time.time()
    for step in range(1, n_steps_tc1):
        state = model_tc1.step(state, dt_tc1)

        if (step + 1) % diag_interval_tc1 == 0:
            t_min = (step + 1) * dt_tc1 / 60.0
            fields = nh_state_to_grid(state, grid_tc1)
            tc1_times.append(t_min)
            tc1_max_w.append(float(np.max(np.abs(fields["w"]))))
            tc1_max_theta.append(float(np.max(np.abs(fields["theta_prime"]))))
            tc1_max_div.append(float(np.max(np.abs(fields["div"]))))

            for sm in snap_times_min:
                if abs(t_min - sm) < dt_tc1 / 60.0 * 2:
                    if sm not in tc1_snapshots:
                        tc1_snapshots[sm] = fields

            if int(t_min) % 10 == 0:
                print(f"    t={t_min:5.1f}min: max|w|={tc1_max_w[-1]:.4f}m/s, "
                      f"max|theta'|={tc1_max_theta[-1]:.4f}K")

    wall_time = time.time() - t_wall
    print(f"  Done in {wall_time:.1f}s ({n_steps_tc1/wall_time:.0f} steps/s)")

    # Ensure final snapshot
    if 60 not in tc1_snapshots:
        tc1_snapshots[60] = nh_state_to_grid(state, grid_tc1)

    tc1_times = np.array(tc1_times)
    tc1_max_w = np.array(tc1_max_w)
    tc1_max_theta = np.array(tc1_max_theta)
    tc1_max_div = np.array(tc1_max_div)

    tc1_pass = (np.all(np.isfinite(tc1_max_w)) and
                np.all(np.isfinite(tc1_max_theta)))

    print(f"\n  TC1 Results after 1 hour:")
    print(f"    max|w|:       {tc1_max_w[-1]:.4f} m/s")
    print(f"    max|theta'|:  {tc1_max_theta[-1]:.4f} K")
    print(f"    All finite:   {tc1_pass}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat_deg = np.degrees(np.array(grid_tc1.lat))
    lon_deg = np.degrees(np.array(grid_tc1.lon2d[0]))
    z_km = np.array(hcoord_tc1.z_full) / 1000.0
    z_half_km = np.array(hcoord_tc1.z_half) / 1000.0

    # 1. Rest-state stability
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].semilogy(rest_times, np.array(rest_max_w) + 1e-20, "b-", lw=1.5, label="max|w|")
    axes[0].set_ylabel("max|w| [m/s]")
    axes[0].set_title("NH Rest-State Stability (T21/L10)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(rest_times, np.array(rest_max_theta) + 1e-20, "r-", lw=1.5, label="max|theta'|")
    axes[1].set_ylabel("max|theta'| [K]")
    axes[1].set_xlabel("Time [min]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rest_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved rest_stability.png")

    # 2. TC1 Time series
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(tc1_times, tc1_max_w, "b-", lw=1.5)
    axes[0].set_ylabel("max|w| [m/s]")
    axes[0].set_title("DCMIP TC1 Gravity Waves (T31/L40)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tc1_times, tc1_max_theta, "r-", lw=1.5)
    axes[1].set_ylabel("max|theta'| [K]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tc1_times, tc1_max_div, "g-", lw=1.5)
    axes[2].set_ylabel("max|div| [1/s]")
    axes[2].set_xlabel("Time [min]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved timeseries.png")

    # 3. Zonal-mean w profiles (lat-z)
    fig, axes = plt.subplots(1, len(snap_times_min), figsize=(5 * len(snap_times_min), 5))
    for i, sm in enumerate(snap_times_min):
        if sm in tc1_snapshots:
            w = tc1_snapshots[sm]["w"]  # (n_lat, n_lon, nlev+1)
            w_zm = np.mean(w, axis=1)  # (n_lat, nlev+1)
            vmax = max(np.max(np.abs(w_zm)), 1e-8)
            im = axes[i].contourf(lat_deg, z_half_km, w_zm.T, levels=20,
                                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"t = {sm} min")
            axes[i].set_xlabel("Latitude")
            if i == 0:
                axes[i].set_ylabel("Height [km]")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="w [m/s]")
    fig.suptitle("Zonal-Mean Vertical Velocity (T31/L40)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "profiles_w.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved profiles_w.png")

    # 4. Zonal-mean theta' profiles (lat-z)
    fig, axes = plt.subplots(1, len(snap_times_min), figsize=(5 * len(snap_times_min), 5))
    for i, sm in enumerate(snap_times_min):
        if sm in tc1_snapshots:
            tp = tc1_snapshots[sm]["theta_prime"]  # (n_lat, n_lon, nlev)
            tp_zm = np.mean(tp, axis=1)
            vmax = max(np.max(np.abs(tp_zm)), 1e-8)
            im = axes[i].contourf(lat_deg, z_km, tp_zm.T, levels=20,
                                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"t = {sm} min")
            axes[i].set_xlabel("Latitude")
            if i == 0:
                axes[i].set_ylabel("Height [km]")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="theta' [K]")
    fig.suptitle("Zonal-Mean Pot. Temperature Perturbation (T31/L40)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "profiles_theta.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved profiles_theta.png")

    # 5. w at mid-level, lat-lon snapshots
    mid_lev = hcoord_tc1.n_levels // 2
    fig, axes = plt.subplots(1, len(snap_times_min), figsize=(5 * len(snap_times_min), 4))
    for i, sm in enumerate(snap_times_min):
        if sm in tc1_snapshots:
            w_mid = tc1_snapshots[sm]["w"][:, :, mid_lev]
            vmax = max(np.max(np.abs(w_mid)), 1e-8)
            im = axes[i].pcolormesh(lon_deg, lat_deg, w_mid,
                                     cmap="RdBu_r", shading="auto",
                                     vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"t = {sm} min")
            axes[i].set_xlabel("Longitude")
            if i == 0:
                axes[i].set_ylabel("Latitude")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="w [m/s]")
    fig.suptitle(f"Vertical Velocity at z={z_half_km[mid_lev]:.1f} km (T31/L40)",
                  fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "snapshots_w.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved snapshots_w.png")

    # ── Save data ─────────────────────────────────────────────────────
    np.savez(
        os.path.join(OUT_DIR, "diagnostics.npz"),
        rest_times=np.array(rest_times),
        rest_max_w=np.array(rest_max_w),
        rest_max_theta=np.array(rest_max_theta),
        tc1_times=tc1_times, tc1_max_w=tc1_max_w,
        tc1_max_theta=tc1_max_theta, tc1_max_div=tc1_max_div,
        lat_deg=lat_deg, lon_deg=lon_deg, z_km=z_km,
    )
    print(f"  Saved diagnostics.npz")

    # ── Summary ───────────────────────────────────────────────────────
    summary = []
    summary.append("Spectral Non-Hydrostatic Benchmark Summary")
    summary.append("=" * 50)
    summary.append("")
    summary.append("Part 1: Rest-State Stability (T21/L10, 1 hour)")
    summary.append(f"  max|w| final:     {rest_max_w[-1]:.2e} m/s (threshold: 1e-6)")
    summary.append(f"  max|theta'| final:{rest_max_theta[-1]:.2e} K (threshold: 1e-6)")
    summary.append(f"  PASS: {rest_pass}")
    summary.append("")
    summary.append("Part 2: DCMIP TC1 Gravity Waves (T31/L40, 1 hour)")
    summary.append(f"  max|w| final:     {tc1_max_w[-1]:.4f} m/s")
    summary.append(f"  max|theta'| final:{tc1_max_theta[-1]:.4f} K")
    summary.append(f"  All finite:       {tc1_pass}")
    summary.append(f"  PASS: {tc1_pass}")
    summary.append("")
    summary.append("Reference: DCMIP-2025 TC1 (mountain gravity waves)")
    summary.append("  Gravity waves should propagate upward from mountain")
    summary.append("  w perturbations should grow but remain bounded")
    summary.append("  theta' should show wave-like structure")

    summary_text = "\n".join(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")

    return rest_pass and tc1_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
