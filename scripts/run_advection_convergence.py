"""Ocean advection scheme convergence testing.

Isolated tests with known analytical solutions to verify each tracer
advection scheme works correctly, independent of pressure, barotropic
solver, or physics parameterizations.

Level 1: Pure 1D zonal advection
  Advect a Gaussian with uniform zonal velocity on a periodic lat-lon
  channel. After one revolution, compare to initial condition. Verify
  convergence rate matches expected formal order.

Usage
-----
Quick (low-res only):
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence.py --quick

Full convergence study:
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence.py

Single scheme:
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence.py --schemes tvd
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


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

ALL_SCHEMES = ["upwind", "tvd", "dst3", "weno5", "weno7", "som"]
# ppm and ppm_fct excluded: ppm is testing-only, ppm_fct has known limiter bug

EXPECTED_ORDERS = {
    "upwind": 1,
    "tvd": 2,
    "dst3": 3,
    "weno5": 5,
    "weno7": 7,
    "som": 2,   # 2nd-order moments
}

# Resolutions: (n_lat, n_lon) — only n_lon matters for zonal advection
# n_lat is kept small and fixed since we only advect zonally
RESOLUTIONS_QUICK = [(8, 32), (8, 64)]
RESOLUTIONS_FULL = [(8, 32), (8, 64), (8, 128), (8, 256)]


# ---------------------------------------------------------------------------
# Level 1: Pure 1D zonal advection
# ---------------------------------------------------------------------------

def _create_grid(n_lat, n_lon):
    """Create a periodic lat-lon channel grid."""
    from legoesm.grids.latlon import create_regional_latlon_grid
    grid, _ = create_regional_latlon_grid(
        n_lat, n_lon,
        lat_south=20.0, lat_north=30.0,
        lon_west=0.0, lon_east=360.0,
        periodic_x=True,
    )
    return grid


SIGMA_LON = 60.0  # Gaussian half-width in degrees (wide enough for good convergence)
CENTER_LON = 180.0  # Initial center


def _gaussian_tracer(grid, center_lon=CENTER_LON, nlev=1):
    """Create a Gaussian tracer blob centered at given longitude.

    Returns array of shape (n_lat_grid, n_lon, nlev).
    Grid lon/lat are 1D: lon (n_lon,), lat (n_lat_grid,).
    """
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi  # (n_lon,)
    lon_centered = lon_deg - center_lon
    lon_centered = np.where(lon_centered > 180, lon_centered - 360, lon_centered)
    lon_centered = np.where(lon_centered < -180, lon_centered + 360, lon_centered)
    tracer_1d = np.exp(-0.5 * (lon_centered / SIGMA_LON) ** 2)  # (n_lon,)

    n_lat_grid = grid.area.shape[0]
    n_lon = grid.area.shape[1]
    tracer_2d = np.broadcast_to(tracer_1d[np.newaxis, :], (n_lat_grid, n_lon))
    return jnp.array(tracer_2d[..., np.newaxis] * np.ones(nlev))


def _uniform_zonal_mass_flux(grid, nlev, velocity_mps, h_uniform):
    """Create uniform eastward mass flux at u-faces.

    mass_flux_u = h * u, shape (n_lat_grid, n_lon+1, nlev).
    """
    n_lat_grid, n_lon = grid.area.shape
    mf = jnp.full((n_lat_grid, n_lon + 1, nlev), h_uniform * velocity_mps)
    return mf


def _uniform_layer_thickness(grid, nlev, h_uniform):
    """Uniform layer thickness at cell centers and u/v faces."""
    n_lat_grid, n_lon = grid.area.shape
    h_cell = jnp.full((n_lat_grid, n_lon, nlev), h_uniform)
    h_u = jnp.full((n_lat_grid, n_lon + 1, nlev), h_uniform)
    h_v = jnp.full((n_lat_grid + 1, n_lon, nlev), h_uniform)
    return h_cell, h_u, h_v


def _advect_one_step_horizontal(scheme, tracer, mass_flux_u, mass_flux_v,
                                h_cell, h_u, h_v, grid, dt):
    """Apply one horizontal advection step for a given scheme.

    Returns updated tracer field.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid

    n_lon = tracer.shape[1]

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

    elif scheme == "som":
        # SOM uses its own full operator — handled separately
        raise ValueError("SOM uses full 3-sweep operator, not this path")

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    tracer_flux_u = mass_flux_u * tr_u
    tracer_flux_v = mass_flux_v * tr_v
    div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)

    # Flux-form update: h_new * T_new = h_old * T_old - dt * div(h*u*T)
    # With uniform h and incompressible flow, h_new = h_old
    hT_new = h_cell * tracer - dt * div_hut
    return hT_new / h_cell


def _advect_som_one_step(tracer, som_moments, mass_flux_u, mass_flux_v,
                         w, h_cell, land_mask, grid, dt):
    """Apply one SOM advection step."""
    from legoesm.ocean.advection_som import som_advect_tracers
    # SOM needs h_k_new — for incompressible uniform flow, h_k_new = h_k_old
    T_new, T_som_new = som_advect_tracers(
        tracer, som_moments, mass_flux_u, mass_flux_v, w,
        h_cell, h_cell, grid, dt, land_mask)
    return T_new, T_som_new


def _run_level1_single(scheme, n_lat, n_lon, output_dir):
    """Run Level 1 test for a single scheme at a single resolution.

    Returns dict with error norms and metadata.
    """
    nlev = 1
    h_uniform = 100.0  # meters
    n_test_steps = 20   # Fixed number of steps — avoids error saturation

    grid = _create_grid(n_lat, n_lon)
    n_lat_grid, n_lon_actual = grid.area.shape

    # Tracer: Gaussian blob centered at 180 deg
    tracer_init = _gaussian_tracer(grid, nlev=nlev)

    # Velocity: uniform eastward flow at CFL ~ 0.5
    R = float(grid.radius)
    cos_lat_center = np.cos(np.radians(25.0))
    dx = R * float(grid.dlon) * cos_lat_center
    velocity = 42.0  # m/s — arbitrary, CFL set by dt
    dt = 0.5 * dx / velocity  # CFL = 0.5
    n_steps = n_test_steps

    actual_cfl = velocity * dt / dx
    # How far does the tracer move in degrees?
    shift_deg = n_steps * velocity * dt / (R * cos_lat_center) * 180.0 / np.pi

    print(f"    {scheme} @ {n_lat}x{n_lon}: n_steps={n_steps}, "
          f"dt={dt:.1f}s, CFL={actual_cfl:.3f}, shift={shift_deg:.1f}deg")

    # Exact solution: Gaussian shifted eastward by shift_deg
    tracer_exact = _gaussian_tracer(grid, center_lon=CENTER_LON + shift_deg, nlev=nlev)

    # Create fields
    h_cell, h_u, h_v = _uniform_layer_thickness(grid, nlev, h_uniform)
    mass_flux_u = _uniform_zonal_mass_flux(grid, nlev, velocity, h_uniform)
    mass_flux_v = jnp.zeros((n_lat_grid + 1, n_lon_actual, nlev))
    w = jnp.zeros((n_lat_grid, n_lon_actual, nlev + 1))

    # Time integration
    tracer = tracer_init
    t0 = time.time()

    if scheme == "som":
        # SOM moments: 9 higher-order polynomial coefficients per cell, init to zero
        som_moments = jnp.zeros(tracer.shape + (9,))
        # Land mask: all ocean (1.0) for this test
        land_mask = jnp.ones((n_lat_grid, n_lon_actual))
        for step in range(n_steps):
            tracer, som_moments = _advect_som_one_step(
                tracer, som_moments, mass_flux_u, mass_flux_v,
                w, h_cell, land_mask, grid, dt)
    else:
        for step in range(n_steps):
            tracer = _advect_one_step_horizontal(
                scheme, tracer, mass_flux_u, mass_flux_v,
                h_cell, h_u, h_v, grid, dt)

    jax.block_until_ready(tracer)
    wall = time.time() - t0

    # Error norms vs shifted exact solution (area-weighted)
    error = np.asarray(tracer[..., 0] - tracer_exact[..., 0])
    exact = np.asarray(tracer_exact[..., 0])
    area = np.asarray(grid.area)

    l1 = float(np.sum(np.abs(error) * area) / np.sum(np.abs(exact) * area))
    l2 = float(np.sqrt(np.sum(error**2 * area) / np.sum(exact**2 * area)))
    linf = float(np.max(np.abs(error)) / np.max(np.abs(exact)))

    # Conservation
    mass_init = float(np.sum(np.asarray(tracer_init[..., 0]) * area))
    mass_final = float(np.sum(np.asarray(tracer[..., 0]) * area))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)

    result = {
        "scheme": scheme, "n_lon": n_lon, "n_lat": n_lat,
        "l1": l1, "l2": l2, "linf": linf,
        "mass_drift": mass_drift, "wall": wall,
        "dt": dt, "cfl": actual_cfl, "n_steps": n_steps,
    }

    # Save fields for plotting
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / f"{scheme}_{n_lon}.npz",
                 tracer_init=np.asarray(tracer_init[..., 0]),
                 tracer_final=np.asarray(tracer[..., 0]),
                 tracer_exact=np.asarray(tracer_exact[..., 0]),
                 error=error,
                 lon_deg=np.asarray(grid.lon) * 180 / np.pi,
                 lat_deg=np.asarray(grid.lat) * 180 / np.pi,
                 area=area)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_4panel(data_file, output_dir):
    """Plot init / final / exact / error for a single scheme+resolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(data_file)
    tracer_init = d["tracer_init"]
    tracer_final = d["tracer_final"]
    tracer_exact = d.get("tracer_exact", tracer_init)  # fallback to init
    error = d["error"]
    lon = d["lon_deg"]

    # Take a cross-section at the middle latitude
    mid_lat = tracer_init.shape[0] // 2
    lon_1d = lon if lon.ndim == 1 else lon[mid_lat, :]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    name = data_file.stem  # e.g. "tvd_64"

    axes[0, 0].plot(lon_1d, tracer_init[mid_lat, :], 'k--', label="Initial")
    axes[0, 0].plot(lon_1d, tracer_exact[mid_lat, :], 'g-', label="Exact (shifted)")
    axes[0, 0].legend()
    axes[0, 0].set_title("Initial + Exact")
    axes[0, 0].set_ylabel("Tracer")

    axes[0, 1].plot(lon_1d, tracer_final[mid_lat, :])
    axes[0, 1].set_title("Computed (after advection)")

    axes[1, 0].plot(lon_1d, tracer_exact[mid_lat, :], 'k--', label="Exact")
    axes[1, 0].plot(lon_1d, tracer_final[mid_lat, :], 'b-', label="Computed")
    axes[1, 0].legend()
    axes[1, 0].set_title("Exact vs Computed")
    axes[1, 0].set_xlabel("Longitude (deg)")
    axes[1, 0].set_ylabel("Tracer")

    axes[1, 1].plot(lon_1d, error[mid_lat, :])
    axes[1, 1].set_title("Error")
    axes[1, 1].set_xlabel("Longitude (deg)")

    fig.suptitle(name.replace("_", " "), fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_4panel.png", dpi=150)
    plt.close(fig)


def _plot_cross_section_compare(results, data_dir, output_dir):
    """Plot all schemes overlaid on one axis at the highest resolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find highest resolution
    max_nlon = max(r["n_lon"] for r in results)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot exact (shifted) first
    first_scheme = results[0]["scheme"]
    d = np.load(data_dir / f"{first_scheme}_{max_nlon}.npz")
    mid_lat = d["tracer_init"].shape[0] // 2
    lon_raw = d["lon_deg"]
    lon_1d = lon_raw if lon_raw.ndim == 1 else lon_raw[mid_lat, :]
    exact = d.get("tracer_exact", d["tracer_init"])
    ax.plot(lon_1d, exact[mid_lat, :], 'k--', linewidth=2,
            label="Exact", zorder=10)

    # Plot each scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALL_SCHEMES)))
    schemes_plotted = set()
    for r in results:
        if r["n_lon"] != max_nlon:
            continue
        if r["scheme"] in schemes_plotted:
            continue
        schemes_plotted.add(r["scheme"])
        d = np.load(data_dir / f"{r['scheme']}_{max_nlon}.npz")
        idx = ALL_SCHEMES.index(r["scheme"]) if r["scheme"] in ALL_SCHEMES else 0
        ax.plot(lon_1d, d["tracer_final"][mid_lat, :],
                color=colors[idx], label=f"{r['scheme']} (L2={r['l2']:.2e})")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Tracer")
    ax.set_title(f"Cross-scheme comparison (n_lon={max_nlon})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cross_section_compare.png", dpi=150)
    plt.close(fig)


def _plot_convergence(results, output_dir):
    """Plot log(error) vs log(dx) with reference slopes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    schemes = sorted(set(r["scheme"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(schemes), 2)))

    for i, scheme in enumerate(schemes):
        scheme_results = sorted(
            [r for r in results if r["scheme"] == scheme],
            key=lambda r: r["n_lon"])
        if len(scheme_results) < 2:
            continue
        n_lons = [r["n_lon"] for r in scheme_results]
        l2s = [r["l2"] for r in scheme_results]

        # Compute empirical convergence rate
        if len(n_lons) >= 2 and l2s[-1] > 0 and l2s[0] > 0:
            rate = np.log(l2s[0] / l2s[-1]) / np.log(n_lons[-1] / n_lons[0])
        else:
            rate = 0

        expected = EXPECTED_ORDERS.get(scheme, "?")
        ax.loglog(n_lons, l2s, 'o-', color=colors[i],
                  label=f"{scheme} (rate={rate:.1f}, expected={expected})",
                  linewidth=2, markersize=8)

    # Reference slopes
    n_ref = np.array([32, 256])
    for order, ls in [(1, ':'), (2, '--'), (3, '-.'), (5, '-'), (7, '-')]:
        y_ref = 0.5 * (n_ref[0] / n_ref) ** order
        ax.loglog(n_ref, y_ref, ls, color='gray', alpha=0.4,
                  label=f"O(dx^{order})" if order <= 3 else None)

    ax.set_xlabel("n_lon (grid points)")
    ax.set_ylabel("L2 error norm")
    ax.set_title("Convergence rates — Level 1 (1D zonal advection)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()  # finer resolution on the right
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_rates.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Ocean advection scheme convergence testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--schemes", type=str, default=",".join(ALL_SCHEMES),
                   help="Comma-separated schemes to test")
    p.add_argument("--output", type=str, default="results/advection_convergence",
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
    resolutions = RESOLUTIONS_QUICK if args.quick else RESOLUTIONS_FULL
    output_base = Path(args.output)
    level1_dir = output_base / "level1_1d"

    print("=" * 70)
    print("  Ocean Advection Convergence Testing")
    print("=" * 70)
    print(f"  Schemes:     {', '.join(schemes)}")
    print(f"  Resolutions: {[f'{r[0]}x{r[1]}' for r in resolutions]}")
    print(f"  Output:      {output_base}")
    print("=" * 70)

    # --- Level 1: 1D zonal advection ---
    print("\n--- Level 1: Pure 1D zonal advection ---\n")

    all_results = []
    t0 = time.time()

    for scheme in schemes:
        for n_lat, n_lon in resolutions:
            result = _run_level1_single(scheme, n_lat, n_lon, level1_dir)
            all_results.append(result)
            print(f"      L1={result['l1']:.2e}  L2={result['l2']:.2e}  "
                  f"Linf={result['linf']:.2e}  mass_drift={result['mass_drift']:.2e}  "
                  f"wall={result['wall']:.2f}s")

    total_wall = time.time() - t0

    # Summary table
    print(f"\n{'='*90}")
    print(f"  LEVEL 1 SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Scheme':<10} {'n_lon':>6} {'L1':>10} {'L2':>10} "
          f"{'Linf':>10} {'mass_drift':>12} {'CFL':>6}")
    print("-" * 90)
    for r in all_results:
        print(f"  {r['scheme']:<10} {r['n_lon']:>6} {r['l1']:>10.2e} "
              f"{r['l2']:>10.2e} {r['linf']:>10.2e} "
              f"{r['mass_drift']:>12.2e} {r['cfl']:>6.3f}")

    # Convergence rates
    print(f"\n  Convergence rates (L2):")
    for scheme in schemes:
        sr = sorted([r for r in all_results if r["scheme"] == scheme],
                    key=lambda r: r["n_lon"])
        if len(sr) >= 2 and sr[-1]["l2"] > 0 and sr[0]["l2"] > 0:
            rate = np.log(sr[0]["l2"] / sr[-1]["l2"]) / np.log(sr[-1]["n_lon"] / sr[0]["n_lon"])
            expected = EXPECTED_ORDERS.get(scheme, "?")
            status = "OK" if rate >= expected * 0.8 else "LOW"
            print(f"    {scheme:<10}: {rate:.2f} (expected {expected}) {status}")
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # Save summary
    level1_dir.mkdir(parents=True, exist_ok=True)
    with open(level1_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    # Plots
    try:
        for r in all_results:
            npz = level1_dir / f"{r['scheme']}_{r['n_lon']}.npz"
            if npz.exists():
                _plot_4panel(npz, level1_dir)

        _plot_cross_section_compare(all_results, level1_dir, level1_dir)
        _plot_convergence(all_results, level1_dir)
        print(f"\n  Plots saved to: {level1_dir}")
    except ImportError:
        print("  (matplotlib not available, skipping plots)")

    # Exit code
    any_bad = False
    for scheme in schemes:
        sr = sorted([r for r in all_results if r["scheme"] == scheme],
                    key=lambda r: r["n_lon"])
        if len(sr) >= 2 and sr[-1]["l2"] > 0 and sr[0]["l2"] > 0:
            rate = np.log(sr[0]["l2"] / sr[-1]["l2"]) / np.log(sr[-1]["n_lon"] / sr[0]["n_lon"])
            expected = EXPECTED_ORDERS.get(scheme, 1)
            if rate < expected * 0.5:
                print(f"\n  WARNING: {scheme} convergence rate {rate:.2f} "
                      f"is below 50% of expected {expected}")
                any_bad = True

    if any_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
