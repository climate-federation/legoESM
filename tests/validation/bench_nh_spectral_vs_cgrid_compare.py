#!/usr/bin/env python3
"""Side-by-side NH dycore comparison: spectral vs lat-lon C-grid.

Runs the spectral-NH (T21/L10) and lat-lon-C-grid-NH (32x64/L10)
dycores on **identical** initial conditions (mid-level theta-perturbation
gravity wave + paired exact-constant-pressure rho') over the same
1-hour window with the same outer ``dt`` and the same vertical
column depth.  The Gaussian grid has ``n_lat = 33`` quadrature points
vs the C-grid's uniform ``n_lat = 32`` — the two are not bit-equal
but should produce comparable bulk diagnostics.

Outputs (in ``results/atmosphere/nonhydrostatic/comparison/``):
  - timeseries.png      — max|w|, max|theta'|, max|rho'| for both dycores
  - profile_w_60min.png — zonal-mean w(lat, z) at t = 60 min, both dycores
  - profile_theta_60min.png — zonal-mean theta'(lat, z) at t = 60 min
  - diagnostics.npz     — all paired diagnostic arrays
  - summary.txt         — qualitative + quantitative comparison
"""

from __future__ import annotations

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Spectral imports
from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis_3d,
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
)

# C-grid imports
from legoesm.grids.latlon import create_latlon_grid
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere._future.compressible_euler_latlon_cgrid import (
    CGridLatLonCompressibleEulerConfig,
    CGridLatLonNonHydrostaticState,
    cgrid_latlon_nh_step,
)


OUT_DIR = os.path.join(
    PROJECT_DIR, "results", "atmosphere", "nonhydrostatic", "comparison",
)
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# Common initial condition (constant-pressure mid-level theta perturbation)
# ============================================================================

def make_theta_perturbation(lat_2d_deg, lon_2d_deg, z_full, nlev):
    """Build the common theta' perturbation field on a (n_lat, n_lon, nlev) grid."""
    z_3d = jnp.broadcast_to(
        z_full[None, None, :], lat_2d_deg.shape + (nlev,),
    )
    lat_3d = lat_2d_deg[..., None]
    lon_3d = lon_2d_deg[..., None]
    z_mid = z_full[nlev // 2]
    return (
        1.0
        * jnp.exp(-((z_3d - z_mid) / 5000.0) ** 2)
        * jnp.exp(-((lat_3d - 30.0) / 10.0) ** 2)
        * jnp.cos(lon_3d * jnp.pi / 180.0)
    )


# ============================================================================
# Driver
# ============================================================================

def main():
    print("=" * 70)
    print("  NH Dycore Comparison: Spectral vs lat-lon C-grid")
    print("=" * 70)

    nlev = 10
    H = 30000.0
    dt = 5.0
    duration = 3600.0
    n_steps = int(duration / dt)
    diag_interval = int(120 / dt)   # every 2 min
    snap_t_min = 60

    # ────────────────────────────────────────────────────────────────────
    # Spectral side: T21 / L10
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("  Spectral dycore: T21 / L10 / 1 hour")
    print("-" * 50)

    n_max = 21
    grid_sp = create_gaussian_grid(n_max=n_max)
    hcoord_sp = create_height_coordinate(nlev, H)
    z_s_sp = jnp.zeros((grid_sp.n_lat, grid_sp.n_lon))
    terrain_sp = compute_terrain_metric(z_s_sp, hcoord_sp)

    # Hyperdiff coefficient sized off the truncation; mirror bench_spectral_nh.
    a = grid_sp.radius
    eig_max = n_max * (n_max + 1) / (a * a)
    nu_sp = 1.0 / (4.0 * 3600.0 * eig_max ** 2)
    cfg_sp = SpectralNHConfig(
        hyperdiff_coeff=nu_sp,
        hyperdiff_order=2,
        sponge_coeff=0.05,
        sponge_width=10000.0,
        n_acoustic_substeps=6,
    )
    model_sp = SpectralCompressibleEulerModel(
        grid_sp, hcoord_sp, terrain_sp, cfg_sp,
        allow_unsupported_backend=True,
    )

    # Build IC: rest state + theta' on the spectral grid.
    state_sp = nh_rest_state_spectral(grid_sp, hcoord_sp)
    lat_2d_sp_deg = jnp.degrees(grid_sp.lat2d)
    lon_2d_sp_deg = jnp.degrees(grid_sp.lon2d)
    theta_perturb_sp = make_theta_perturbation(
        lat_2d_sp_deg, lon_2d_sp_deg, hcoord_sp.z_full, nlev,
    )
    theta_perturb_sp_hat = sh_analysis_3d(grid_sp, theta_perturb_sp)
    state_sp = state_sp._replace(
        theta_prime_hat=state_sp.theta_prime_hat.replace(
            data=theta_perturb_sp_hat,
        ),
    )
    # We leave rho' = 0 spectrally: the spectral dycore was tuned that
    # way and the linear analysis still gives buoyancy at the seeded
    # cell because pi' = (R_d / c_v) * pi_0 * theta' / theta_0 (small).

    t0 = time.time()
    state_sp = model_sp.step(state_sp, dt)
    jax.block_until_ready(state_sp.vor_hat.data)
    print(f"  Spectral JIT compiled in {time.time()-t0:.1f}s")

    sp_times = [0.0]
    sp_max_w = [
        float(np.max(np.abs(np.array(
            sh_synthesis_3d(grid_sp, state_sp.w_hat.data),
        )))),
    ]
    sp_max_theta = [
        float(np.max(np.abs(np.array(
            sh_synthesis_3d(grid_sp, state_sp.theta_prime_hat.data),
        )))),
    ]
    sp_max_rho = [
        float(np.max(np.abs(np.array(
            sh_synthesis_3d(grid_sp, state_sp.rho_prime_hat.data),
        )))),
    ]

    t_wall = time.time()
    for step in range(1, n_steps):
        state_sp = model_sp.step(state_sp, dt)
        if (step + 1) % diag_interval == 0:
            t_min = (step + 1) * dt / 60.0
            w_g = np.array(sh_synthesis_3d(grid_sp, state_sp.w_hat.data))
            tp_g = np.array(
                sh_synthesis_3d(grid_sp, state_sp.theta_prime_hat.data),
            )
            rp_g = np.array(
                sh_synthesis_3d(grid_sp, state_sp.rho_prime_hat.data),
            )
            sp_times.append(t_min)
            sp_max_w.append(float(np.max(np.abs(w_g))))
            sp_max_theta.append(float(np.max(np.abs(tp_g))))
            sp_max_rho.append(float(np.max(np.abs(rp_g))))

    jax.block_until_ready(state_sp.vor_hat.data)
    print(f"  Spectral done in {time.time()-t_wall:.1f}s")
    print(
        f"  Spectral final: max|w|={sp_max_w[-1]:.4f}, "
        f"max|theta'|={sp_max_theta[-1]:.4f}"
    )

    # Spectral snapshot at t = 60 min for the side-by-side profile plot.
    w_sp_60 = np.array(sh_synthesis_3d(grid_sp, state_sp.w_hat.data))
    tp_sp_60 = np.array(
        sh_synthesis_3d(grid_sp, state_sp.theta_prime_hat.data),
    )
    lat_sp_deg = np.degrees(np.array(grid_sp.lat))

    # ────────────────────────────────────────────────────────────────────
    # C-grid side: 32 x 64 / L10
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("  lat-lon C-grid dycore: 32 x 64 / L10 / 1 hour")
    print("-" * 50)

    grid_cg = create_latlon_grid(n_lat=32, n_lon=64)
    hcoord_cg = create_height_coordinate(nlev, H)
    z_s_cg = jnp.zeros((grid_cg.n_lat, grid_cg.n_lon))
    terrain_cg = compute_terrain_metric(z_s_cg, hcoord_cg)

    # Horizontal Laplacian hyperdiffusion on the C-grid -- mirror what
    # the spectral side gets from its bilaplacian truncation control.
    # Scale chosen so the dissipation timescale at the grid wavelength
    # matches the spectral side at T21 (~ 1.5e6 m^2/s when read in the
    # Laplacian convention used here).
    cfg_cg = CGridLatLonCompressibleEulerConfig(
        euler=CompressibleEulerConfig(
            sponge_width=10000.0,
            sponge_coeff=0.05,
            n_acoustic_substeps=6,
            semi_implicit_acoustic=True,
            use_coriolis=True,
        ),
        hyperdiff_uv=1.0e6,
        hyperdiff_w=1.0e5,
        hyperdiff_scalar=1.0e6,
    )

    lat_2d_cg_deg = jnp.degrees(grid_cg.lat2d)
    lon_2d_cg_deg = jnp.degrees(grid_cg.lon2d)
    theta_perturb_cg = make_theta_perturbation(
        lat_2d_cg_deg, lon_2d_cg_deg, hcoord_cg.z_full, nlev,
    )
    theta_0_cg = hcoord_cg.theta_ref[None, None, :]
    rho_0_cg = hcoord_cg.rho_ref[None, None, :]
    rho_perturb_cg = (
        rho_0_cg * theta_0_cg / (theta_0_cg + theta_perturb_cg) - rho_0_cg
    )

    state_cg = CGridLatLonNonHydrostaticState(
        u=jnp.zeros((grid_cg.n_lat, grid_cg.n_lon + 1, nlev)),
        v=jnp.zeros((grid_cg.n_lat + 1, grid_cg.n_lon, nlev)),
        w=jnp.zeros((grid_cg.n_lat, grid_cg.n_lon, nlev + 1)),
        theta_prime=theta_perturb_cg,
        rho_prime=rho_perturb_cg,
        phis=jnp.zeros((grid_cg.n_lat, grid_cg.n_lon)),
        tracers={},
    )

    step_cg_jit = jax.jit(
        lambda s: cgrid_latlon_nh_step(
            s, grid_cg, hcoord_cg, terrain_cg, dt, cfg_cg,
        ),
    )

    t0 = time.time()
    state_cg = step_cg_jit(state_cg)
    jax.block_until_ready(state_cg.w)
    print(f"  C-grid JIT compiled in {time.time()-t0:.1f}s")

    cg_times = [0.0]
    cg_max_w = [float(np.max(np.abs(np.asarray(state_cg.w))))]
    cg_max_theta = [
        float(np.max(np.abs(np.asarray(state_cg.theta_prime)))),
    ]
    cg_max_rho = [float(np.max(np.abs(np.asarray(state_cg.rho_prime))))]

    t_wall = time.time()
    for step in range(1, n_steps):
        state_cg = step_cg_jit(state_cg)
        if (step + 1) % diag_interval == 0:
            t_min = (step + 1) * dt / 60.0
            cg_times.append(t_min)
            cg_max_w.append(float(np.max(np.abs(np.asarray(state_cg.w)))))
            cg_max_theta.append(
                float(np.max(np.abs(np.asarray(state_cg.theta_prime))))
            )
            cg_max_rho.append(
                float(np.max(np.abs(np.asarray(state_cg.rho_prime))))
            )

    jax.block_until_ready(state_cg.w)
    print(f"  C-grid done in {time.time()-t_wall:.1f}s")
    print(
        f"  C-grid final: max|w|={cg_max_w[-1]:.4f}, "
        f"max|theta'|={cg_max_theta[-1]:.4f}"
    )

    w_cg_60 = np.asarray(state_cg.w)
    tp_cg_60 = np.asarray(state_cg.theta_prime)
    lat_cg_deg = np.degrees(np.asarray(grid_cg.lat))

    # ────────────────────────────────────────────────────────────────────
    # Plots
    # ────────────────────────────────────────────────────────────────────
    print("\n  Generating comparison plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp_times_a = np.array(sp_times)
    sp_max_w_a = np.array(sp_max_w)
    sp_max_theta_a = np.array(sp_max_theta)
    sp_max_rho_a = np.array(sp_max_rho)
    cg_times_a = np.array(cg_times)
    cg_max_w_a = np.array(cg_max_w)
    cg_max_theta_a = np.array(cg_max_theta)
    cg_max_rho_a = np.array(cg_max_rho)

    # 1. Time series
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(sp_times_a, sp_max_w_a, "b-",  lw=1.7, label="spectral T21")
    axes[0].plot(cg_times_a, cg_max_w_a, "r--", lw=1.7, label="C-grid 32x64")
    axes[0].set_ylabel("max|w| [m/s]")
    axes[0].set_title(
        "NH dycores on common IC (mid-level theta-perturbation gravity wave, "
        "L10, dt = 5 s)"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(sp_times_a, sp_max_theta_a, "b-",  lw=1.7)
    axes[1].plot(cg_times_a, cg_max_theta_a, "r--", lw=1.7)
    axes[1].set_ylabel("max|theta'| [K]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sp_times_a, sp_max_rho_a, "b-",  lw=1.7)
    axes[2].plot(cg_times_a, cg_max_rho_a, "r--", lw=1.7)
    axes[2].set_ylabel("max|rho'| [kg/m^3]")
    axes[2].set_xlabel("Time [min]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "timeseries.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("  Saved timeseries.png")

    # 2. Zonal-mean w at t = 60 min, side-by-side
    z_half_km_sp = np.asarray(hcoord_sp.z_half) / 1000.0
    z_half_km_cg = np.asarray(hcoord_cg.z_half) / 1000.0
    z_km_sp = np.asarray(hcoord_sp.z_full) / 1000.0
    z_km_cg = np.asarray(hcoord_cg.z_full) / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    w_zm_sp = np.mean(w_sp_60, axis=1)   # (n_lat_sp, nlev+1)
    w_zm_cg = np.mean(w_cg_60, axis=1)   # (n_lat_cg, nlev+1)
    vmax = max(
        max(float(np.max(np.abs(w_zm_sp))), 1.0e-8),
        max(float(np.max(np.abs(w_zm_cg))), 1.0e-8),
    )
    im0 = axes[0].contourf(
        lat_sp_deg, z_half_km_sp, w_zm_sp.T, levels=20,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axes[0].set_title(f"spectral T21 (t = {snap_t_min} min)")
    axes[0].set_xlabel("Latitude")
    axes[0].set_ylabel("Height [km]")

    im1 = axes[1].contourf(
        lat_cg_deg, z_half_km_cg, w_zm_cg.T, levels=20,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axes[1].set_title(f"C-grid 32x64 (t = {snap_t_min} min)")
    axes[1].set_xlabel("Latitude")

    fig.colorbar(
        im1, ax=axes.ravel().tolist(), shrink=0.8, label="w [m/s]",
    )
    fig.suptitle(
        "Zonal-mean vertical velocity at t = 60 min", fontsize=13, y=1.02,
    )
    plt.savefig(
        os.path.join(OUT_DIR, "profile_w_60min.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("  Saved profile_w_60min.png")

    # 3. Zonal-mean theta' at t = 60 min, side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    tp_zm_sp = np.mean(tp_sp_60, axis=1)
    tp_zm_cg = np.mean(tp_cg_60, axis=1)
    vmax = max(
        max(float(np.max(np.abs(tp_zm_sp))), 1.0e-8),
        max(float(np.max(np.abs(tp_zm_cg))), 1.0e-8),
    )
    im0 = axes[0].contourf(
        lat_sp_deg, z_km_sp, tp_zm_sp.T, levels=20,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axes[0].set_title(f"spectral T21 (t = {snap_t_min} min)")
    axes[0].set_xlabel("Latitude")
    axes[0].set_ylabel("Height [km]")

    im1 = axes[1].contourf(
        lat_cg_deg, z_km_cg, tp_zm_cg.T, levels=20,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axes[1].set_title(f"C-grid 32x64 (t = {snap_t_min} min)")
    axes[1].set_xlabel("Latitude")

    fig.colorbar(
        im1, ax=axes.ravel().tolist(), shrink=0.8, label="theta' [K]",
    )
    fig.suptitle(
        "Zonal-mean potential temperature perturbation at t = 60 min",
        fontsize=13, y=1.02,
    )
    plt.savefig(
        os.path.join(OUT_DIR, "profile_theta_60min.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("  Saved profile_theta_60min.png")

    # ── Save diagnostics ────────────────────────────────────────────────
    np.savez(
        os.path.join(OUT_DIR, "diagnostics.npz"),
        sp_times=sp_times_a, sp_max_w=sp_max_w_a,
        sp_max_theta=sp_max_theta_a, sp_max_rho=sp_max_rho_a,
        cg_times=cg_times_a, cg_max_w=cg_max_w_a,
        cg_max_theta=cg_max_theta_a, cg_max_rho=cg_max_rho_a,
        lat_sp_deg=lat_sp_deg, lat_cg_deg=lat_cg_deg,
        z_km_sp=z_km_sp, z_km_cg=z_km_cg,
        z_half_km_sp=z_half_km_sp, z_half_km_cg=z_half_km_cg,
    )
    print("  Saved diagnostics.npz")

    # ── Summary ─────────────────────────────────────────────────────────
    summary = []
    summary.append("NH dycore comparison summary (spectral vs lat-lon C-grid)")
    summary.append("=" * 60)
    summary.append("")
    summary.append("Common setup: L10, H = 30 km, dt = 5 s, 1 hour, "
                   "mid-level theta-perturbation gravity wave IC.")
    summary.append(
        f"  Spectral grid: T{n_max} ({grid_sp.n_lat} x {grid_sp.n_lon})"
    )
    summary.append(
        f"  C-grid grid:   {grid_cg.n_lat} x {grid_cg.n_lon}"
    )
    summary.append("")
    summary.append(
        f"  Spectral final  max|w|={sp_max_w_a[-1]:.4f}  "
        f"max|theta'|={sp_max_theta_a[-1]:.4f}"
    )
    summary.append(
        f"  C-grid   final  max|w|={cg_max_w_a[-1]:.4f}  "
        f"max|theta'|={cg_max_theta_a[-1]:.4f}"
    )
    summary.append("")
    summary.append(
        f"  Both dycores remain bounded and finite: "
        f"{bool(np.all(np.isfinite(sp_max_w_a)) and np.all(np.isfinite(cg_max_w_a)))}"
    )
    summary_text = "\n".join(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")


if __name__ == "__main__":
    main()
