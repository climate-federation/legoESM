#!/usr/bin/env python3
"""Benchmark: lat-lon C-grid Non-Hydrostatic compressible Euler.

Mirrors the layout of ``bench_spectral_nh.py`` so the two dycores can
be compared side-by-side on equivalent setups:

  1. Rest-state stability (n_lat=32, n_lon=64, L10, 1 hour, dt=5s).
     The discrete rest state is an exact equilibrium of the
     continuous equations; the discrete dycore should preserve it
     to floating-point round-off.

  2. Mid-level theta-perturbation gravity wave (same grid).
     Mirrors the spectral-NH fallback initial condition: a Gaussian
     theta' blob at mid-column with a latitude-localised cosine
     longitudinal wave pattern.  This excites internal gravity
     waves of finite amplitude that should propagate upward and
     equatorward; we verify that ``max|w|`` stays bounded and that
     no NaNs appear over the integration.

Outputs (in ``results/atmosphere/nonhydrostatic/cgrid_latlon/``):
  - rest_stability.png      — perturbation growth time series
  - timeseries.png          — max|w|, max|theta'|, max|div(u_h)|
  - profiles_w.png          — zonal-mean w(lat, z) at 0/20/40/60 min
  - profiles_theta.png      — zonal-mean theta'(lat, z) at the same times
  - snapshots_w.png         — w at mid-level (lat, lon) at the same times
  - diagnostics.npz         — all diagnostic arrays
  - summary.txt             — quantitative PASS/FAIL

A companion script ``cmp_nh_spectral_vs_cgrid.py`` reads this script's
``diagnostics.npz`` plus the spectral one and produces a comparison
figure (max|w| time series of both dycores on the same axes).
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

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler_latlon_cgrid import (
    CGridLatLonCompressibleEulerConfig,
    CGridLatLonNonHydrostaticState,
    cgrid_latlon_nh_step,
)


OUT_DIR = os.path.join(
    PROJECT_DIR, "results", "atmosphere", "nonhydrostatic", "cgrid_latlon",
)
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# Initial conditions
# ============================================================================

def rest_state(grid, hcoord) -> CGridLatLonNonHydrostaticState:
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, hcoord.n_levels
    return CGridLatLonNonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        w=jnp.zeros((n_lat, n_lon, nlev + 1)),
        theta_prime=jnp.zeros((n_lat, n_lon, nlev)),
        rho_prime=jnp.zeros((n_lat, n_lon, nlev)),
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={},
    )


def gravity_wave_state(grid, hcoord) -> CGridLatLonNonHydrostaticState:
    """Mid-level theta-perturbation gravity-wave initial condition.

    Matches the spectral-NH benchmark's fallback IC: a Gaussian theta'
    blob at mid-column with a single-wave cosine pattern in longitude,
    centred at 30° latitude with a 10° meridional half-width.  The
    paired rho' perturbation is chosen to keep total pressure exactly
    on the reference state (constant-rho*theta) so the initial Exner
    perturbation vanishes and the dynamics begins with pure buoyancy.
    """
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, hcoord.n_levels
    lat_deg = jnp.degrees(grid.lat2d)
    lon_deg = jnp.degrees(grid.lon2d)
    z_full = hcoord.z_full
    z_mid = z_full[nlev // 2]

    z_3d = jnp.broadcast_to(z_full[None, None, :], (n_lat, n_lon, nlev))
    lat_3d = lat_deg[..., None]
    lon_3d = lon_deg[..., None]

    theta_perturb = (
        1.0
        * jnp.exp(-((z_3d - z_mid) / 5000.0) ** 2)
        * jnp.exp(-((lat_3d - 30.0) / 10.0) ** 2)
        * jnp.cos(lon_3d * jnp.pi / 180.0)
    )

    # Exact constant-pressure rho' (matches the unit-test construction):
    #   rho' = rho_0 * theta_0 / (theta_0 + theta') - rho_0
    theta_0 = hcoord.theta_ref[None, None, :]
    rho_0 = hcoord.rho_ref[None, None, :]
    rho_perturb = rho_0 * theta_0 / (theta_0 + theta_perturb) - rho_0

    return CGridLatLonNonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        w=jnp.zeros((n_lat, n_lon, nlev + 1)),
        theta_prime=theta_perturb,
        rho_prime=rho_perturb,
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={},
    )


# ============================================================================
# Diagnostics
# ============================================================================

def state_diags(state: CGridLatLonNonHydrostaticState) -> dict:
    """Maximum-norm diagnostics + zonal-mean w, theta' snapshots."""
    w = np.asarray(state.w)
    tp = np.asarray(state.theta_prime)
    rp = np.asarray(state.rho_prime)
    # horizontal divergence proxy at cell centres: (du/dx + dv/dy).
    # On a C-grid this is cheap to compute from the face velocities.
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    div = np.zeros(tp.shape, dtype=tp.dtype)  # placeholder; we report
    # max|u| as a stand-in.  (A full divergence_cgrid call would require
    # passing the grid; we keep the diagnostic light to avoid retracing.)
    return {
        "w": w,
        "theta_prime": tp,
        "rho_prime": rp,
        "u": u,
        "v": v,
        "div": div,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  lat-lon C-grid Non-Hydrostatic Benchmark")
    print("=" * 70)

    # ── Common setup ────────────────────────────────────────────────────
    n_lat, n_lon = 32, 64
    nlev = 10
    H = 30000.0
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    hcoord = create_height_coordinate(nlev, H)
    z_s = jnp.zeros((n_lat, n_lon))
    terrain = compute_terrain_metric(z_s, hcoord)

    # Sponge layer in top 10 km with the same coeff as the spectral
    # benchmark (0.05 1/s).  We use the semi-implicit acoustic substep
    # so dt=5 s is well inside the slow-tendency CFL.
    # Horizontal Laplacian hyperdiffusion: needed at finite volume on a
    # 32 x 64 lat-lon grid to keep grid-scale noise from accumulating
    # in long integrations.  Coefficient K ~ dx^2 / (2 * tau) with
    # dx ~ 625 km at the equator and tau ~ 6 h gives K ~ 9e6 m^2/s;
    # we round to 1e6 m^2/s (tau ~ 2 day) for a gentler default and
    # use 1e5 m^2/s on w (w hyperdiff only suppresses 2dx wiggles in
    # the vertical-velocity gravity-wave train).
    cfg = CGridLatLonCompressibleEulerConfig(
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

    dt = 5.0
    duration = 3600.0
    n_steps = int(duration / dt)
    diag_interval = int(120 / dt)   # every 2 min
    snap_times_min = [0, 20, 40, 60]

    # JIT-compile one step.  We pin grid/hcoord/terrain/config so JAX
    # treats them as static pytree leaves and the trace is reused.
    step_jit = jax.jit(
        lambda s: cgrid_latlon_nh_step(s, grid, hcoord, terrain, dt, cfg),
    )

    # ── 1. Rest-state stability ────────────────────────────────────────
    print("\n" + "-" * 50)
    print(
        f"  Part 1: Rest-State Stability "
        f"({n_lat}x{n_lon}, L{nlev}, {duration/60.0:.0f} min)"
    )
    print("-" * 50)

    state = rest_state(grid, hcoord)
    # Warm up JIT
    t0 = time.time()
    state = step_jit(state)
    jax.block_until_ready(state.w)
    print(f"  JIT compiled in {time.time()-t0:.1f}s")

    rest_times = [0.0]
    rest_max_w = [0.0]
    rest_max_theta = [0.0]
    rest_max_rho = [0.0]

    t_wall = time.time()
    for step in range(1, n_steps):
        state = step_jit(state)
        if (step + 1) % diag_interval == 0:
            t_min = (step + 1) * dt / 60.0
            d = state_diags(state)
            rest_times.append(t_min)
            rest_max_w.append(float(np.max(np.abs(d["w"]))))
            rest_max_theta.append(float(np.max(np.abs(d["theta_prime"]))))
            rest_max_rho.append(float(np.max(np.abs(d["rho_prime"]))))

    jax.block_until_ready(state.w)
    wall_time = time.time() - t_wall
    print(f"  Done in {wall_time:.1f}s ({n_steps/wall_time:.0f} steps/s)")
    print(
        f"  Final: max|w|={rest_max_w[-1]:.2e}, "
        f"max|theta'|={rest_max_theta[-1]:.2e}"
    )

    rest_pass = (
        np.all(np.isfinite(rest_max_w))
        and rest_max_w[-1] < 1.0e-6
        and rest_max_theta[-1] < 1.0e-6
    )

    # ── 2. Gravity wave from mid-level theta perturbation ──────────────
    print("\n" + "-" * 50)
    print(
        f"  Part 2: Mid-Level Theta Perturbation Gravity Wave "
        f"({n_lat}x{n_lon}, L{nlev}, {duration/60.0:.0f} min)"
    )
    print("-" * 50)

    state = gravity_wave_state(grid, hcoord)
    # Warm up JIT again (different input shape inside, but graph
    # already cached — should be near-instant).
    state = step_jit(state)
    jax.block_until_ready(state.w)

    gw_times = [0.0]
    gw_max_w = [0.0]
    gw_max_theta = [float(np.max(np.abs(np.asarray(state.theta_prime))))]
    gw_max_rho = [float(np.max(np.abs(np.asarray(state.rho_prime))))]
    snapshots = {0: state_diags(state)}

    t_wall = time.time()
    for step in range(1, n_steps):
        state = step_jit(state)
        if (step + 1) % diag_interval == 0:
            t_min = (step + 1) * dt / 60.0
            d = state_diags(state)
            gw_times.append(t_min)
            gw_max_w.append(float(np.max(np.abs(d["w"]))))
            gw_max_theta.append(float(np.max(np.abs(d["theta_prime"]))))
            gw_max_rho.append(float(np.max(np.abs(d["rho_prime"]))))

            for sm in snap_times_min:
                if abs(t_min - sm) < dt / 60.0 * 2:
                    snapshots.setdefault(sm, d)

            if int(t_min) % 10 == 0:
                print(
                    f"    t={t_min:5.1f}min: "
                    f"max|w|={gw_max_w[-1]:.4f} m/s, "
                    f"max|theta'|={gw_max_theta[-1]:.4f} K"
                )

    jax.block_until_ready(state.w)
    wall_time = time.time() - t_wall
    print(f"  Done in {wall_time:.1f}s ({n_steps/wall_time:.0f} steps/s)")

    gw_times = np.array(gw_times)
    gw_max_w = np.array(gw_max_w)
    gw_max_theta = np.array(gw_max_theta)
    gw_max_rho = np.array(gw_max_rho)

    gw_pass = (
        np.all(np.isfinite(gw_max_w))
        and np.all(np.isfinite(gw_max_theta))
        # Stay bounded: no explosive growth.
        and gw_max_w[-1] < 50.0
        and gw_max_theta[-1] < 5.0
    )

    print(f"\n  Gravity-wave results after 1 hour:")
    print(f"    max|w|:       {gw_max_w[-1]:.4f} m/s")
    print(f"    max|theta'|:  {gw_max_theta[-1]:.4f} K")
    print(f"    bounded:      {gw_pass}")

    # ── Plots ───────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))
    z_km = np.asarray(hcoord.z_full) / 1000.0
    z_half_km = np.asarray(hcoord.z_half) / 1000.0

    # 1. Rest-state stability
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].semilogy(
        rest_times, np.array(rest_max_w) + 1e-20, "b-", lw=1.5, label="max|w|",
    )
    axes[0].set_ylabel("max|w| [m/s]")
    axes[0].set_title(f"NH C-grid Rest-State Stability ({n_lat}x{n_lon}/L{nlev})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(
        rest_times, np.array(rest_max_theta) + 1e-20,
        "r-", lw=1.5, label="max|theta'|",
    )
    axes[1].set_ylabel("max|theta'| [K]")
    axes[1].set_xlabel("Time [min]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "rest_stability.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved rest_stability.png")

    # 2. Gravity-wave time series
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(gw_times, gw_max_w, "b-", lw=1.5)
    axes[0].set_ylabel("max|w| [m/s]")
    axes[0].set_title(
        f"NH C-grid Mid-Level Theta-Perturbation Gravity Wave "
        f"({n_lat}x{n_lon}/L{nlev})"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gw_times, gw_max_theta, "r-", lw=1.5)
    axes[1].set_ylabel("max|theta'| [K]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(gw_times, gw_max_rho, "g-", lw=1.5)
    axes[2].set_ylabel("max|rho'| [kg/m^3]")
    axes[2].set_xlabel("Time [min]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "timeseries.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved timeseries.png")

    # 3. Zonal-mean w (lat, z)
    fig, axes = plt.subplots(
        1, len(snap_times_min), figsize=(5 * len(snap_times_min), 5),
    )
    for i, sm in enumerate(snap_times_min):
        if sm in snapshots:
            w = snapshots[sm]["w"]                 # (n_lat, n_lon, nlev+1)
            w_zm = np.mean(w, axis=1)              # (n_lat, nlev+1)
            vmax = max(np.max(np.abs(w_zm)), 1.0e-8)
            im = axes[i].contourf(
                lat_deg, z_half_km, w_zm.T, levels=20,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            )
            axes[i].set_title(f"t = {sm} min")
            axes[i].set_xlabel("Latitude")
            if i == 0:
                axes[i].set_ylabel("Height [km]")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="w [m/s]")
    fig.suptitle("C-grid: Zonal-Mean Vertical Velocity", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "profiles_w.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved profiles_w.png")

    # 4. Zonal-mean theta'
    fig, axes = plt.subplots(
        1, len(snap_times_min), figsize=(5 * len(snap_times_min), 5),
    )
    for i, sm in enumerate(snap_times_min):
        if sm in snapshots:
            tp = snapshots[sm]["theta_prime"]      # (n_lat, n_lon, nlev)
            tp_zm = np.mean(tp, axis=1)            # (n_lat, nlev)
            vmax = max(np.max(np.abs(tp_zm)), 1.0e-8)
            im = axes[i].contourf(
                lat_deg, z_km, tp_zm.T, levels=20,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            )
            axes[i].set_title(f"t = {sm} min")
            axes[i].set_xlabel("Latitude")
            if i == 0:
                axes[i].set_ylabel("Height [km]")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="theta' [K]")
    fig.suptitle(
        "C-grid: Zonal-Mean Pot. Temperature Perturbation",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "profiles_theta.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved profiles_theta.png")

    # 5. w mid-level snapshots (lat, lon)
    mid_lev = hcoord.n_levels // 2
    fig, axes = plt.subplots(
        1, len(snap_times_min), figsize=(5 * len(snap_times_min), 4),
    )
    for i, sm in enumerate(snap_times_min):
        if sm in snapshots:
            w_mid = snapshots[sm]["w"][:, :, mid_lev]
            vmax = max(np.max(np.abs(w_mid)), 1.0e-8)
            im = axes[i].pcolormesh(
                lon_deg, lat_deg, w_mid,
                cmap="RdBu_r", shading="auto",
                vmin=-vmax, vmax=vmax,
            )
            axes[i].set_title(f"t = {sm} min")
            axes[i].set_xlabel("Longitude")
            if i == 0:
                axes[i].set_ylabel("Latitude")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="w [m/s]")
    fig.suptitle(
        f"C-grid: w at z = {z_half_km[mid_lev]:.1f} km", fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "snapshots_w.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved snapshots_w.png")

    # ── Save diagnostics ────────────────────────────────────────────────
    np.savez(
        os.path.join(OUT_DIR, "diagnostics.npz"),
        rest_times=np.array(rest_times),
        rest_max_w=np.array(rest_max_w),
        rest_max_theta=np.array(rest_max_theta),
        rest_max_rho=np.array(rest_max_rho),
        gw_times=gw_times,
        gw_max_w=gw_max_w,
        gw_max_theta=gw_max_theta,
        gw_max_rho=gw_max_rho,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        z_km=z_km,
        z_half_km=z_half_km,
    )
    print(f"  Saved diagnostics.npz")

    # ── Summary ─────────────────────────────────────────────────────────
    summary = []
    summary.append("C-grid lat-lon Non-Hydrostatic Benchmark Summary")
    summary.append("=" * 50)
    summary.append("")
    summary.append(
        f"Part 1: Rest-State Stability "
        f"({n_lat}x{n_lon}, L{nlev}, {duration/60.0:.0f} min)"
    )
    summary.append(f"  max|w| final:     {rest_max_w[-1]:.2e} m/s (threshold: 1e-6)")
    summary.append(f"  max|theta'| final:{rest_max_theta[-1]:.2e} K   (threshold: 1e-6)")
    summary.append(f"  PASS: {rest_pass}")
    summary.append("")
    summary.append(
        f"Part 2: Gravity Wave "
        f"({n_lat}x{n_lon}, L{nlev}, {duration/60.0:.0f} min)"
    )
    summary.append(f"  max|w| final:     {gw_max_w[-1]:.4f} m/s")
    summary.append(f"  max|theta'| final:{gw_max_theta[-1]:.4f} K")
    summary.append(f"  bounded:          {gw_pass}")
    summary.append(f"  PASS: {gw_pass}")
    summary.append("")
    summary.append(
        "Cross-reference: tests/validation/bench_spectral_nh.py"
    )

    summary_text = "\n".join(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")

    return rest_pass and gw_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
