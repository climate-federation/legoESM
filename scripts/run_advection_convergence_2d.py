"""Level 2: 2D prescribed-flow advection on ocean lat-lon C-grid.

Tests the full horizontal advection operator (u + v combined) with
divergence, grid metrics, and dimensional splitting.

Test: Swirling deformation flow (Nair & Lauritzen 2010)
  - Non-divergent 2D flow that deforms a cosine bell into a filament
  - Flow reverses at t = T/2, tracer returns to IC at t = T
  - Error at t = T measures combined spatial + temporal accuracy

Usage
-----
Quick:
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence_2d.py --quick

Full:
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence_2d.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ALL_SCHEMES = ["upwind", "tvd", "dst3", "weno5", "weno7"]
# SOM excluded initially — needs separate 3-sweep integration path

RESOLUTIONS = [(32, 64), (64, 128), (128, 256)]
RESOLUTIONS_QUICK = [(32, 64), (64, 128)]

# Deformation period: 5 days (arbitrary, just needs enough steps)
T_PERIOD = 5.0 * 86400.0


# ---------------------------------------------------------------------------
# Prescribed flow: swirling deformation (Nair & Lauritzen 2010)
# ---------------------------------------------------------------------------

def _swirl_velocity(lon_u, lat_u, lon_v, lat_v, t, T=T_PERIOD):
    """Compute swirling deformation velocity at u-faces and v-faces.

    Non-divergent 2D flow that deforms and reverses:
      u = -A * sin²(lon) * sin(2*lat) * cos(π*t/T)
      v =  A * sin(2*lon) * cos²(lat) * cos(π*t/T)

    where A scales to give CFL ~ 0.5 at the reference resolution.

    Parameters
    ----------
    lon_u, lat_u : arrays in radians, at u-face positions
    lon_v, lat_v : arrays in radians, at v-face positions
    t : float, current time [s]
    T : float, reversal period [s]

    Returns
    -------
    u_east : array at u-faces [m/s]
    v_north : array at v-faces [m/s]
    """
    R = 6371.0e3  # Earth radius
    # Stream function ψ = A*sin²(λ)*cos²(φ) gives non-divergent flow:
    #   u = -(1/R)∂ψ/∂φ = +(A/R)*sin²(λ)*sin(2φ)
    #   v = +(1/(R*cosφ))∂ψ/∂λ = +(A/R)*sin(2λ)*cos(φ)
    # Verified: div = (1/(Rcosφ))[∂u/∂λ + ∂(v*cosφ)/∂φ] = 0 analytically.
    A = 20.0  # m/s

    cos_phase = jnp.cos(jnp.pi * t / T)

    u = A * jnp.sin(lon_u) ** 2 * jnp.sin(2.0 * lat_u) * cos_phase
    v = A * jnp.sin(2.0 * lon_v) * jnp.cos(lat_v) * cos_phase

    return u, v


def _cosine_bell(lon, lat, lon0, lat0, R_bell):
    """Cosine bell centered at (lon0, lat0) with radius R_bell [radians].

    Returns values in [0, 1]. Zero outside the bell.
    """
    # Great-circle distance
    r = jnp.arccos(jnp.clip(
        jnp.sin(lat) * jnp.sin(lat0)
        + jnp.cos(lat) * jnp.cos(lat0) * jnp.cos(lon - lon0),
        -1.0, 1.0))
    return jnp.where(r < R_bell, 0.5 * (1.0 + jnp.cos(jnp.pi * r / R_bell)), 0.0)


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

def _create_grid(n_lat, n_lon):
    """Create a global periodic lat-lon grid (no walls)."""
    from legoesm.grids.latlon import create_regional_latlon_grid
    # Wide channel with bell centered far from walls
    grid, _ = create_regional_latlon_grid(
        n_lat, n_lon,
        lat_south=-10.0, lat_north=80.0,
        lon_west=0.0, lon_east=360.0,
        periodic_x=True)
    return grid


def _get_face_coordinates(grid):
    """Get lon/lat in radians at u-faces and v-faces.

    For lat-lon C-grid:
      u-faces: (n_lat, n_lon+1) at cell longitude interfaces
      v-faces: (n_lat+1, n_lon) at cell latitude interfaces
    """
    lon_cell = np.asarray(grid.lon)  # (n_lon,) radians
    lat_cell = np.asarray(grid.lat)  # (n_lat_grid,) radians
    dlon = float(grid.dlon)
    dlat = float(grid.dlat)

    n_lat_grid = lat_cell.shape[0]
    n_lon = lon_cell.shape[0]

    # u-face longitudes: face j is at lon_cell[j] - dlon/2
    lon_u_1d = np.concatenate([lon_cell - dlon / 2,
                                [lon_cell[-1] + dlon / 2]])  # (n_lon+1,)
    # u-face latitudes: same as cell centers
    lat_u_1d = lat_cell  # (n_lat_grid,)

    # 2D arrays for u-faces: (n_lat_grid, n_lon+1)
    lon_u = np.broadcast_to(lon_u_1d[np.newaxis, :], (n_lat_grid, n_lon + 1))
    lat_u = np.broadcast_to(lat_u_1d[:, np.newaxis], (n_lat_grid, n_lon + 1))

    # v-face latitudes: face i is at lat_cell[i] - dlat/2
    lat_v_1d = np.concatenate([lat_cell - dlat / 2,
                                [lat_cell[-1] + dlat / 2]])  # (n_lat_grid+1,)
    # v-face longitudes: same as cell centers
    lon_v = np.broadcast_to(lon_cell[np.newaxis, :], (n_lat_grid + 1, n_lon))
    lat_v = np.broadcast_to(lat_v_1d[:, np.newaxis], (n_lat_grid + 1, n_lon))

    return (jnp.array(lon_u), jnp.array(lat_u),
            jnp.array(lon_v), jnp.array(lat_v))


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------

def _flux_divergence_2d(scheme, tracer, mass_flux_u, mass_flux_v,
                        h_u, h_v, grid, dt):
    """Compute div(h*u*T_face) for the given scheme."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid

    if scheme == "upwind":
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _upwind_to_u_points, _upwind_to_v_points)
        tr_u = _upwind_to_u_points(tracer, mass_flux_u)
        tr_v = _upwind_to_v_points(tracer, mass_flux_v)
    elif scheme == "tvd":
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _tvd_to_u_points, _tvd_to_v_points)
        tr_u = _tvd_to_u_points(tracer, mass_flux_u)
        tr_v = _tvd_to_v_points(tracer, mass_flux_v)
    elif scheme == "dst3":
        from legoesm.ocean.advection import dst3_to_u_points, dst3_to_v_points
        tr_u = dst3_to_u_points(tracer, mass_flux_u, h_u, grid, dt)
        tr_v = dst3_to_v_points(tracer, mass_flux_v, h_v, grid, dt)
    elif scheme == "weno5":
        from legoesm.ocean.advection import weno5_to_u_points, weno5_to_v_points
        tr_u = weno5_to_u_points(tracer, mass_flux_u)
        tr_v = weno5_to_v_points(tracer, mass_flux_v)
    elif scheme == "weno7":
        from legoesm.ocean.advection import weno7_to_u_points, weno7_to_v_points
        tr_u = weno7_to_u_points(tracer, mass_flux_u)
        tr_v = weno7_to_v_points(tracer, mass_flux_v)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return divergence_cgrid(mass_flux_u * tr_u, mass_flux_v * tr_v, grid)


def _run_single_2d(scheme, n_lat, n_lon, output_dir, time_integrator="euler",
                   ab2_eps=0.1):
    """Run Level 2 deformational flow test for one scheme+integrator+resolution."""

    nlev = 1
    h_uniform = 100.0

    grid = _create_grid(n_lat, n_lon)
    n_lat_grid, n_lon_actual = grid.area.shape
    lon_u, lat_u, lon_v, lat_v = _get_face_coordinates(grid)

    # Initial condition: cosine bell at (lon0=π, lat0=π/4) = (180°E, 45°N)
    lon_cell = jnp.array(grid.lon)  # (n_lon,) radians
    lat_cell = jnp.array(grid.lat)  # (n_lat_grid,) radians
    lon_2d = jnp.broadcast_to(lon_cell[jnp.newaxis, :], (n_lat_grid, n_lon_actual))
    lat_2d = jnp.broadcast_to(lat_cell[:, jnp.newaxis], (n_lat_grid, n_lon_actual))

    R_bell = jnp.pi / 6.0  # 30 degrees radius
    T_init = _cosine_bell(lon_2d, lat_2d, jnp.pi, jnp.pi / 4.0, R_bell)
    tracer = T_init[..., jnp.newaxis]  # (n_lat_grid, n_lon, 1)

    # Time stepping: full period T (deform + reverse)
    R = float(grid.radius)
    cos_mid = float(jnp.cos(jnp.radians(45.0)))
    dx_min = R * float(grid.dlon) * cos_mid
    # dt chosen for CFL ~ 0.3 with max velocity ~ 20 m/s
    dt = 0.3 * dx_min / 20.0
    n_steps = int(np.ceil(T_PERIOD / dt))
    dt = T_PERIOD / n_steps  # exact period coverage

    print(f"    {scheme}+{time_integrator} @ {n_lat}x{n_lon}: n_steps={n_steps}, "
          f"dt={dt:.1f}s, dx_min={dx_min/1e3:.0f}km")

    h_cell = jnp.full((n_lat_grid, n_lon_actual, nlev), h_uniform)
    h_u = jnp.full((n_lat_grid, n_lon_actual + 1, nlev), h_uniform)
    h_v = jnp.full((n_lat_grid + 1, n_lon_actual, nlev), h_uniform)

    def _mass_flux_at(t_eval):
        """Compute mass fluxes at a given time."""
        u_vel, v_vel = _swirl_velocity(
            lon_u[..., jnp.newaxis], lat_u[..., jnp.newaxis],
            lon_v[..., jnp.newaxis], lat_v[..., jnp.newaxis],
            t_eval)
        return h_uniform * u_vel, h_uniform * v_vel

    # Save snapshots at t=0, T/4, T/2, 3T/4, T
    snap_steps = {0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps}
    snapshots = {0: np.asarray(tracer[..., 0])}

    blew_up = False
    div_prev = None  # for AB2

    t0 = time.time()
    for step in range(n_steps):
        t_current = step * dt

        if time_integrator == "euler":
            mfu, mfv = _mass_flux_at(t_current)
            div_now = _flux_divergence_2d(
                scheme, tracer, mfu, mfv, h_u, h_v, grid, dt)
            tracer = (h_cell * tracer - dt * div_now) / h_cell

        elif time_integrator == "ab2":
            # Use velocity at start of step (consistent with model's AB2 path)
            mfu, mfv = _mass_flux_at(t_current)
            div_now = _flux_divergence_2d(
                scheme, tracer, mfu, mfv, h_u, h_v, grid, dt)
            if div_prev is None:
                # First step → Euler fallback (matches model's eager step())
                tracer = (h_cell * tracer - dt * div_now) / h_cell
            else:
                effective = (1.5 + ab2_eps) * div_now - (0.5 + ab2_eps) * div_prev
                tracer = (h_cell * tracer - dt * effective) / h_cell
            div_prev = div_now

        elif time_integrator == "rk3":
            # SSP-RK3 (Shu-Osher form); time-dependent flow → use t at each stage
            mfu0, mfv0 = _mass_flux_at(t_current)
            div0 = _flux_divergence_2d(
                scheme, tracer, mfu0, mfv0, h_u, h_v, grid, dt)
            k1 = (h_cell * tracer - dt * div0) / h_cell
            # Stage 2 advances internally to t+dt
            mfu1, mfv1 = _mass_flux_at(t_current + dt)
            div1 = _flux_divergence_2d(
                scheme, k1, mfu1, mfv1, h_u, h_v, grid, dt)
            k2 = 0.75 * tracer + 0.25 * ((h_cell * k1 - dt * div1) / h_cell)
            # Stage 3 advances internally to t+dt/2 (k2 at intermediate state)
            mfu2, mfv2 = _mass_flux_at(t_current + 0.5 * dt)
            div2 = _flux_divergence_2d(
                scheme, k2, mfu2, mfv2, h_u, h_v, grid, dt)
            tracer = (1.0 / 3.0) * tracer + (2.0 / 3.0) * (
                (h_cell * k2 - dt * div2) / h_cell)
        else:
            raise ValueError(f"Unknown time_integrator: {time_integrator}")

        if (step + 1) in snap_steps:
            snapshots[step + 1] = np.asarray(tracer[..., 0])

        # Detect NaN blowup
        if (step + 1) % max(1, n_steps // 10) == 0:
            if not bool(jnp.all(jnp.isfinite(tracer))):
                print(f"      BLOWUP at step {step+1}/{n_steps}")
                blew_up = True
                break

    jax.block_until_ready(tracer)
    wall = time.time() - t0

    # Error: compare final tracer to initial (should be identical for
    # non-divergent reversible flow)
    tracer_final = np.asarray(tracer[..., 0])
    tracer_init = np.asarray(T_init)
    error = tracer_final - tracer_init
    area = np.asarray(grid.area)

    l1 = float(np.sum(np.abs(error) * area) / np.sum(np.abs(tracer_init) * area))
    l2 = float(np.sqrt(np.sum(error**2 * area) / np.sum(tracer_init**2 * area)))
    linf = float(np.max(np.abs(error)) / np.max(np.abs(tracer_init)))

    mass_init = float(np.sum(tracer_init * area))
    mass_final = float(np.sum(tracer_final * area))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)

    result = {
        "scheme": scheme, "n_lat": n_lat, "n_lon": n_lon,
        "time_integrator": time_integrator,
        "blew_up": blew_up,
        "l1": l1, "l2": l2, "linf": linf,
        "mass_drift": mass_drift, "wall": wall,
        "dt": dt, "n_steps": n_steps,
    }

    # Save for plotting
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / f"{scheme}_{time_integrator}_{n_lat}x{n_lon}.npz",
                 tracer_init=tracer_init,
                 tracer_final=tracer_final,
                 error=error,
                 snapshots={str(k): v for k, v in snapshots.items()},
                 lon_deg=np.asarray(grid.lon) * 180 / np.pi,
                 lat_deg=np.asarray(grid.lat) * 180 / np.pi,
                 area=area)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_snapshots(data_file, output_dir):
    """Plot time evolution of tracer through deformation cycle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(data_file, allow_pickle=True)
    name = data_file.stem
    lon = d["lon_deg"]
    lat = d["lat_deg"]

    snapshots = d["snapshots"].item()
    steps = sorted([int(k) for k in snapshots.keys()])

    n_snaps = len(steps)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4))
    if n_snaps == 1:
        axes = [axes]

    for i, step in enumerate(steps):
        field = snapshots[str(step)]
        ax = axes[i]
        im = ax.pcolormesh(lon, lat, field, cmap="RdYlBu_r",
                          vmin=-0.1, vmax=1.1)
        frac = step / max(steps) if max(steps) > 0 else 0
        ax.set_title(f"t/T = {frac:.2f}")
        ax.set_xlabel("Lon (deg)")
        if i == 0:
            ax.set_ylabel("Lat (deg)")
        ax.set_aspect("equal")

    plt.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(name.replace("_", " "), fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_snapshots.png", dpi=150)
    plt.close(fig)


def _plot_final_compare(results, data_dir, output_dir):
    """Plot all schemes at final time side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use finest resolution
    max_res = max(r["n_lon"] for r in results)
    schemes = sorted(set(r["scheme"] for r in results
                         if r["n_lon"] == max_res))

    n_schemes = len(schemes) + 1  # +1 for exact
    fig, axes = plt.subplots(1, n_schemes, figsize=(4 * n_schemes, 4))

    first = True
    for i, scheme in enumerate(["exact"] + schemes):
        ax = axes[i]
        r = [r for r in results
             if r["scheme"] == (schemes[0] if scheme == "exact" else scheme)
             and r["n_lon"] == max_res]
        if not r:
            continue
        d = np.load(data_dir / f"{r[0]['scheme']}_{r[0]['n_lat']}x{max_res}.npz",
                    allow_pickle=True)
        lon = d["lon_deg"]
        lat = d["lat_deg"]
        if scheme == "exact":
            field = d["tracer_init"]
            title = "Exact"
        else:
            field = d["tracer_final"]
            l2 = [x for x in results if x["scheme"] == scheme
                  and x["n_lon"] == max_res][0]["l2"]
            title = f"{scheme}\nL2={l2:.2e}"

        im = ax.pcolormesh(lon, lat, field, cmap="RdYlBu_r",
                          vmin=-0.1, vmax=1.1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lon (deg)")
        if first:
            ax.set_ylabel("Lat (deg)")
            first = False
        ax.set_aspect("equal")

    plt.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.suptitle(f"Final state comparison (n_lon={max_res})", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "final_compare.png", dpi=150)
    plt.close(fig)


def _plot_convergence(results, output_dir):
    """Plot convergence rates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    schemes = sorted(set(r["scheme"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(schemes), 2)))

    for i, scheme in enumerate(schemes):
        sr = sorted([r for r in results if r["scheme"] == scheme],
                    key=lambda r: r["n_lon"])
        if len(sr) < 2:
            continue
        n_lons = [r["n_lon"] for r in sr]
        l2s = [r["l2"] for r in sr]
        rate = np.log(l2s[0] / l2s[-1]) / np.log(n_lons[-1] / n_lons[0])
        ax.loglog(n_lons, l2s, 'o-', color=colors[i],
                  label=f"{scheme} (rate={rate:.1f})", linewidth=2, markersize=8)

    for order, ls in [(1, ':'), (2, '--')]:
        n_ref = np.array([64, 256])
        y_ref = 1.0 * (n_ref[0] / n_ref) ** order
        ax.loglog(n_ref, y_ref, ls, color='gray', alpha=0.4,
                  label=f"O(dx^{order})")

    ax.set_xlabel("n_lon")
    ax.set_ylabel("L2 error norm")
    ax.set_title("Convergence — Level 2 (2D deformational flow)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_rates.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Level 2: 2D prescribed-flow advection convergence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--schemes", type=str, default=",".join(ALL_SCHEMES),
                   help="Comma-separated schemes")
    p.add_argument("--time-integrators", type=str, default="euler,ab2,rk3",
                   help="Comma-separated time integrators")
    p.add_argument("--output", type=str,
                   default="results/advection_convergence/level2_2d",
                   help="Output directory")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: fewer resolutions")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    from legoesm.core.precision import set_policy, PrecisionPolicy
    set_policy(PrecisionPolicy.fp64())

    schemes = [s.strip() for s in args.schemes.split(",")]
    time_integrators = [t.strip() for t in args.time_integrators.split(",")]
    resolutions = RESOLUTIONS_QUICK if args.quick else RESOLUTIONS
    output_dir = Path(args.output)

    print("=" * 70)
    print("  Level 2: 2D Deformational Flow Advection Test")
    print("=" * 70)
    print(f"  Schemes:     {', '.join(schemes)}")
    print(f"  Integrators: {', '.join(time_integrators)}")
    print(f"  Resolutions: {[f'{r[0]}x{r[1]}' for r in resolutions]}")
    print(f"  Period:      {T_PERIOD/86400:.0f} days (deform + reverse)")
    print(f"  Output:      {output_dir}")
    print("=" * 70)

    all_results = []
    t0 = time.time()

    for ti in time_integrators:
        for scheme in schemes:
            for n_lat, n_lon in resolutions:
                result = _run_single_2d(scheme, n_lat, n_lon, output_dir,
                                       time_integrator=ti)
                all_results.append(result)
                status = "BLEW UP" if result["blew_up"] else "ok"
                print(f"      L2={result['l2']:.2e}  Linf={result['linf']:.2e}  "
                      f"mass_drift={result['mass_drift']:.2e}  "
                      f"wall={result['wall']:.1f}s  [{status}]")

    total_wall = time.time() - t0

    # Summary
    print(f"\n{'='*100}")
    print(f"  LEVEL 2 SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Scheme':<10} {'Integrator':<10} {'Resolution':>12} {'L1':>10} {'L2':>10} "
          f"{'Linf':>10} {'mass_drift':>12} {'status':>10}")
    print("-" * 100)
    for r in all_results:
        status = "BLEW UP" if r["blew_up"] else "ok"
        print(f"  {r['scheme']:<10} {r['time_integrator']:<10} "
              f"{r['n_lat']}x{r['n_lon']:>4} "
              f"{r['l1']:>10.2e} {r['l2']:>10.2e} "
              f"{r['linf']:>10.2e} {r['mass_drift']:>12.2e} {status:>10}")

    # Convergence rates per (scheme, integrator)
    print(f"\n  Convergence rates (L2):")
    for ti in time_integrators:
        for scheme in schemes:
            sr = sorted(
                [r for r in all_results
                 if r["scheme"] == scheme and r["time_integrator"] == ti
                 and not r["blew_up"]],
                key=lambda r: r["n_lon"])
            if len(sr) >= 2:
                rate = (np.log(sr[0]["l2"] / sr[-1]["l2"])
                        / np.log(sr[-1]["n_lon"] / sr[0]["n_lon"]))
                print(f"    {scheme:<10}+{ti:<6}: rate={rate:>5.2f}  "
                      f"L2(coarse)={sr[0]['l2']:.2e}  L2(fine)={sr[-1]['l2']:.2e}")
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    # Plots — only the 4-panel snapshots per scheme/integrator
    try:
        for r in all_results:
            npz = output_dir / (
                f"{r['scheme']}_{r['time_integrator']}_{r['n_lat']}x{r['n_lon']}.npz")
            if npz.exists():
                _plot_snapshots(npz, output_dir)
        print(f"\n  Plots saved to: {output_dir}")
    except (ImportError, TypeError, KeyError):
        print("  (some plots skipped — comparison plots need integrator support)")


if __name__ == "__main__":
    main()
