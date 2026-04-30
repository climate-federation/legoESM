"""Ocean advection scheme convergence testing.

Isolated tests with known analytical solutions to verify each tracer
advection scheme works correctly, independent of pressure, barotropic
solver, or physics parameterizations.

Level 1: Pure 1D zonal advection
  Advect a Gaussian zonally with uniform velocity on a periodic lat-lon
  channel. **Fixed total advection distance** (default 30°), so total
  simulation time stays constant under refinement and only spatial
  truncation decreases. CFL is held at 0.5; n_steps grows with
  resolution.

Test design caveats (as of 2026-04-29 audit):
  - The test mirrors the model's eager step() AB2 path (first step Euler
    when div_prev is None). It does NOT match the integrate_scan/training
    path which pre-initializes the previous flux to zero.
  - The test's RK3 uses the Shu-Osher SSP convex-combination form. The
    model's RK3 uses the Butcher weighted-flux form. They are equivalent
    for linear schemes + constant h, but differ for TVD/WENO with
    nonlinear limiters.
  - AB2 of TVD-limited fluxes is NOT TVD-monotone (the model's docstring
    notes this). Expect overshoots in TVD+AB2 results.

Usage
-----
Quick (low-res only):
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence.py --quick

Full convergence study (all schemes × {euler, ab2, rk3}):
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
SINE_WAVENUMBER = 2  # k for sin(k*lon) test mode

# Module-level switch — set by main() based on --initial-condition
_IC_KIND = "gaussian"


def _smooth_tracer(grid, shift_deg=0.0, nlev=1):
    """Create the test tracer field on the grid.

    Two modes (selected by global _IC_KIND):
      "gaussian" — Gaussian centered at CENTER_LON + shift_deg (sigma=60°).
                   Smooth but with localized shoulders that may stress
                   high-order schemes' smoothness indicators.
      "sine"    — sin(k*(lon - shift_deg_rad)) for k=SINE_WAVENUMBER.
                   Periodic, infinitely smooth, no localized features.
                   Should expose true convergence rates.

    Returns shape (n_lat_grid, n_lon, nlev). Field is uniform in latitude.
    """
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi  # (n_lon,)

    if _IC_KIND == "sine":
        k = SINE_WAVENUMBER
        # sin(k * (lon_deg - shift_deg) * pi/180)
        tracer_1d = np.sin(np.radians(k * (lon_deg - shift_deg)))
    else:  # gaussian
        center_lon = CENTER_LON + shift_deg
        lon_centered = lon_deg - center_lon
        lon_centered = np.where(lon_centered > 180, lon_centered - 360, lon_centered)
        lon_centered = np.where(lon_centered < -180, lon_centered + 360, lon_centered)
        tracer_1d = np.exp(-0.5 * (lon_centered / SIGMA_LON) ** 2)

    n_lat_grid = grid.area.shape[0]
    n_lon = grid.area.shape[1]
    tracer_2d = np.broadcast_to(tracer_1d[np.newaxis, :], (n_lat_grid, n_lon))
    return jnp.array(tracer_2d[..., np.newaxis] * np.ones(nlev))


def _uniform_zonal_mass_flux(grid, nlev, velocity_mps, h_uniform):
    """Create eastward mass flux at u-faces with uniform angular speed.

    To get uniform angular rotation (so all latitudes shift the same amount
    over time t), the linear velocity must scale as cos(lat):
        dlon/dt = u / (R*cos(lat)) = u0 / R   if u = u0 * cos(lat)

    This way the analytical "shifted Gaussian" works at every latitude.

    mass_flux_u = h * u, shape (n_lat_grid, n_lon+1, nlev).
    """
    n_lat_grid, n_lon = grid.area.shape
    cos_lat = jnp.array(grid.cos_lat)  # (n_lat_grid,) at u-face latitudes
    u_face = velocity_mps * cos_lat[:, None, None]  # (n_lat_grid, 1, 1)
    mf = jnp.broadcast_to(h_uniform * u_face,
                          (n_lat_grid, n_lon + 1, nlev))
    return mf


def _uniform_layer_thickness(grid, nlev, h_uniform):
    """Uniform layer thickness at cell centers and u/v faces."""
    n_lat_grid, n_lon = grid.area.shape
    h_cell = jnp.full((n_lat_grid, n_lon, nlev), h_uniform)
    h_u = jnp.full((n_lat_grid, n_lon + 1, nlev), h_uniform)
    h_v = jnp.full((n_lat_grid + 1, n_lon, nlev), h_uniform)
    return h_cell, h_u, h_v


def _flux_divergence(scheme, tracer, mass_flux_u, mass_flux_v,
                     h_u, h_v, grid, dt):
    """Compute div(h*u*T) for a given advection scheme.

    Returns the flux divergence; the time integrator handles the update.
    """
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
    elif scheme == "som":
        raise ValueError("SOM uses full 3-sweep operator, not this path")
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return divergence_cgrid(mass_flux_u * tr_u, mass_flux_v * tr_v, grid)


def _advect_one_step_horizontal(scheme, tracer, mass_flux_u, mass_flux_v,
                                h_cell, h_u, h_v, grid, dt,
                                time_integrator="euler",
                                div_prev=None, ab2_eps=0.1):
    """Advance tracer one step using the specified time integrator.

    Returns
    -------
    tracer_new : array
    div_current : array
        Flux divergence at the start of this step (for AB2 carry).
    """
    div_now = _flux_divergence(scheme, tracer, mass_flux_u, mass_flux_v,
                                h_u, h_v, grid, dt)

    if time_integrator == "euler":
        tracer_new = tracer - (dt / h_cell) * div_now
    elif time_integrator == "ab2":
        if div_prev is None:
            # First step: fall back to Euler
            tracer_new = tracer - (dt / h_cell) * div_now
        else:
            tracer_new = tracer - (dt / h_cell) * (
                (1.5 + ab2_eps) * div_now - (0.5 + ab2_eps) * div_prev)
    elif time_integrator == "rk3":
        # SSP-RK3: k1 = T + dt*F(T)
        # k2 = 3/4 T + 1/4 (k1 + dt F(k1))
        # T^{n+1} = 1/3 T + 2/3 (k2 + dt F(k2))
        k1 = tracer - (dt / h_cell) * div_now
        div1 = _flux_divergence(scheme, k1, mass_flux_u, mass_flux_v,
                                h_u, h_v, grid, dt)
        k2 = 0.75 * tracer + 0.25 * (k1 - (dt / h_cell) * div1)
        div2 = _flux_divergence(scheme, k2, mass_flux_u, mass_flux_v,
                                h_u, h_v, grid, dt)
        tracer_new = (1.0 / 3.0) * tracer + (2.0 / 3.0) * (
            k2 - (dt / h_cell) * div2)
    else:
        raise ValueError(f"Unknown time_integrator: {time_integrator}")

    return tracer_new, div_now


def _advect_som_one_step(tracer, som_moments, mass_flux_u, mass_flux_v,
                         w, h_cell, land_mask, grid, dt):
    """Apply one SOM advection step."""
    from legoesm.ocean.advection_som import som_advect_tracers
    # SOM needs h_k_new — for incompressible uniform flow, h_k_new = h_k_old
    T_new, T_som_new = som_advect_tracers(
        tracer, som_moments, mass_flux_u, mass_flux_v, w,
        h_cell, h_cell, grid, dt, land_mask)
    return T_new, T_som_new


def _run_level1_single(scheme, n_lat, n_lon, output_dir, dt_override=None,
                       time_integrator="euler"):
    """Run Level 1 test for a single scheme at a single resolution.

    Parameters
    ----------
    dt_override : float or None
        If given, use this dt instead of CFL-based dt. This fixes the
        temporal error so spatial convergence can be isolated.
    time_integrator : {"euler", "ab2", "rk3"}
        Time integration method.

    Returns dict with error norms and metadata.
    """
    nlev = 1
    h_uniform = 100.0  # meters
    # Fixed total advection distance (degrees), so total simulation
    # time stays constant under refinement and only spatial truncation
    # decreases. Audit found that fixing n_steps was the wrong design —
    # it made total advection distance shrink linearly with dx.
    target_shift_deg = 30.0

    grid = _create_grid(n_lat, n_lon)
    n_lat_grid, n_lon_actual = grid.area.shape

    # Tracer: smooth field (Gaussian or sine, set by --initial-condition)
    tracer_init = _smooth_tracer(grid, shift_deg=0.0, nlev=nlev)

    # Velocity: u = u0 * cos(lat) so angular rotation is uniform
    # (all latitudes shift by the same number of degrees per unit time).
    # CFL is computed at the equator-equivalent (where u=u0 and dx=R*dlon).
    R = float(grid.radius)
    velocity = 42.0  # m/s — equatorial equivalent (max linear speed)
    dx_eq = R * float(grid.dlon)  # equatorial dx (max dx)
    if dt_override is not None:
        dt = dt_override
    else:
        dt = 0.5 * dx_eq / velocity  # CFL = 0.5 at equator-equivalent

    # Fixed total time: angular speed = u0/R, so to shift target_shift_deg
    # we need t_total = target_shift_rad * R / u0
    t_total = (np.radians(target_shift_deg)) * R / velocity
    n_steps = int(np.ceil(t_total / dt))
    # Adjust dt so n_steps * dt == t_total exactly → exact analytical shift
    dt = t_total / n_steps
    actual_cfl = velocity * dt / dx_eq
    shift_deg = target_shift_deg

    print(f"    {scheme}+{time_integrator} @ {n_lat}x{n_lon}: n_steps={n_steps}, "
          f"dt={dt:.1f}s, CFL={actual_cfl:.3f}, shift={shift_deg:.1f}deg")

    # Exact solution: smooth field shifted eastward by shift_deg.
    #
    # IMPORTANT: The FV evolution `T_new = T - dt*(F_E-F_W)/dx` updates
    # T as a cell average (the discrete divergence is the exact cell-
    # average of dF/dx). But the model's WENO assumes input T is point
    # values and converts internally. So evolved T has a mixed
    # interpretation — but for linear advection with smooth fields,
    # comparing the evolved T against the **cell-average** of the exact
    # solution is what reveals the true scheme order.
    #
    # See dycore expert audit (2026-04-29): without this, all WENO+RK3
    # tests cap at rate 2.00 from the point-value/cell-avg comparison
    # mismatch (for sin(2x), this floor = dx^2/6).
    from legoesm.core.weno import point_to_cellavg_periodic
    tracer_exact_pt = _smooth_tracer(grid, shift_deg=shift_deg, nlev=nlev)
    tracer_exact = point_to_cellavg_periodic(tracer_exact_pt, axis=1, order=6)

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
        div_prev = None
        for step in range(n_steps):
            tracer, div_prev = _advect_one_step_horizontal(
                scheme, tracer, mass_flux_u, mass_flux_v,
                h_cell, h_u, h_v, grid, dt,
                time_integrator=time_integrator,
                div_prev=div_prev)

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
        "time_integrator": time_integrator,
        "l1": l1, "l2": l2, "linf": linf,
        "mass_drift": mass_drift, "wall": wall,
        "dt": dt, "cfl": actual_cfl, "n_steps": n_steps,
    }

    # Save fields for plotting
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / f"{scheme}_{time_integrator}_{n_lon}.npz",
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


def _plot_cross_section_compare(results, data_dir, output_dir, tag=""):
    """Plot all schemes overlaid on one axis at the highest resolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return
    max_nlon = max(r["n_lon"] for r in results)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot exact (shifted) first
    first = results[0]
    ti = first["time_integrator"]
    d = np.load(data_dir / f"{first['scheme']}_{ti}_{max_nlon}.npz")
    mid_lat = d["tracer_init"].shape[0] // 2
    lon_raw = d["lon_deg"]
    lon_1d = lon_raw if lon_raw.ndim == 1 else lon_raw[mid_lat, :]
    exact = d.get("tracer_exact", d["tracer_init"])
    ax.plot(lon_1d, exact[mid_lat, :], 'k--', linewidth=2,
            label="Exact", zorder=10)

    colors = plt.cm.tab10(np.linspace(0, 1, len(ALL_SCHEMES)))
    schemes_plotted = set()
    for r in results:
        if r["n_lon"] != max_nlon or r["scheme"] in schemes_plotted:
            continue
        schemes_plotted.add(r["scheme"])
        d = np.load(data_dir / f"{r['scheme']}_{r['time_integrator']}_{max_nlon}.npz")
        idx = ALL_SCHEMES.index(r["scheme"]) if r["scheme"] in ALL_SCHEMES else 0
        ax.plot(lon_1d, d["tracer_final"][mid_lat, :],
                color=colors[idx], label=f"{r['scheme']} (L2={r['l2']:.2e})")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Tracer")
    title_suffix = f" — {tag}" if tag else ""
    ax.set_title(f"Cross-scheme comparison (n_lon={max_nlon}){title_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"cross_section_compare{('_' + tag) if tag else ''}.png"
    plt.savefig(output_dir / fname, dpi=150)
    plt.close(fig)


def _plot_convergence(results, output_dir, tag=""):
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
    title_suffix = f" — {tag}" if tag else ""
    ax.set_title(f"Convergence rates — Level 1 (1D zonal advection){title_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()  # finer resolution on the right
    plt.tight_layout()
    fname = f"convergence_rates{('_' + tag) if tag else ''}.png"
    plt.savefig(output_dir / fname, dpi=150)
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
    p.add_argument("--time-integrators", type=str, default="euler,ab2,rk3",
                   help="Comma-separated time integrators (euler, ab2, rk3)")
    p.add_argument("--initial-condition", type=str, default="gaussian",
                   choices=["gaussian", "sine"],
                   help="Initial tracer field. 'sine' is infinitely smooth "
                        "and exposes true convergence orders.")
    p.add_argument("--output", type=str, default="results/advection_convergence",
                   help="Output directory")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: fewer resolutions")
    p.add_argument("--fixed-dt", action="store_true",
                   help="Fix dt across resolutions to isolate spatial order "
                        "(uses dt from finest grid, so CFL decreases with coarsening)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    from legoesm.core.precision import set_policy, PrecisionPolicy
    set_policy(PrecisionPolicy.fp64())

    schemes = [s.strip() for s in args.schemes.split(",")]
    time_integrators = [t.strip() for t in args.time_integrators.split(",")]
    resolutions = RESOLUTIONS_QUICK if args.quick else RESOLUTIONS_FULL
    output_base = Path(args.output)
    level1_dir = output_base / "level1_1d"

    # Set the global initial-condition kind (read by _smooth_tracer)
    global _IC_KIND
    _IC_KIND = args.initial_condition

    # Compute fixed dt from finest grid if requested
    dt_fixed = None
    if args.fixed_dt:
        finest_nlon = max(r[1] for r in resolutions)
        finest_grid = _create_grid(resolutions[0][0], finest_nlon)
        R = float(finest_grid.radius)
        cos25 = np.cos(np.radians(25.0))
        dx_finest = R * float(finest_grid.dlon) * cos25
        dt_fixed = 0.5 * dx_finest / 42.0  # CFL=0.5 at finest grid

    mode = "fixed-dt" if args.fixed_dt else "fixed-CFL"
    print("=" * 70)
    print("  Ocean Advection Convergence Testing")
    print("=" * 70)
    print(f"  Initial:     {_IC_KIND}")
    print(f"  Schemes:     {', '.join(schemes)}")
    print(f"  Integrators: {', '.join(time_integrators)}")
    print(f"  Resolutions: {[f'{r[0]}x{r[1]}' for r in resolutions]}")
    print(f"  Mode:        {mode}" + (f" (dt={dt_fixed:.1f}s)" if dt_fixed else ""))
    print(f"  Output:      {output_base}")
    print("=" * 70)

    # --- Level 1: 1D zonal advection ---
    print(f"\n--- Level 1: Pure 1D zonal advection ({mode}) ---\n")

    all_results = []
    t0 = time.time()

    for ti in time_integrators:
        for scheme in schemes:
            if scheme == "som" and ti != "euler":
                # SOM advection uses its own internal sweep — skip non-Euler
                continue
            for n_lat, n_lon in resolutions:
                result = _run_level1_single(scheme, n_lat, n_lon, level1_dir,
                                           dt_override=dt_fixed,
                                           time_integrator=ti)
                all_results.append(result)
                print(f"      L1={result['l1']:.2e}  L2={result['l2']:.2e}  "
                      f"Linf={result['linf']:.2e}  mass_drift={result['mass_drift']:.2e}  "
                      f"wall={result['wall']:.2f}s")

    total_wall = time.time() - t0

    # Summary table
    print(f"\n{'='*100}")
    print(f"  LEVEL 1 SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Scheme':<10} {'Integrator':<10} {'n_lon':>6} {'L1':>10} {'L2':>10} "
          f"{'Linf':>10} {'mass_drift':>12} {'CFL':>6}")
    print("-" * 100)
    for r in all_results:
        print(f"  {r['scheme']:<10} {r['time_integrator']:<10} {r['n_lon']:>6} "
              f"{r['l1']:>10.2e} {r['l2']:>10.2e} {r['linf']:>10.2e} "
              f"{r['mass_drift']:>12.2e} {r['cfl']:>6.3f}")

    # Convergence rates per (scheme, integrator) — pairwise + endpoint
    print(f"\n  Convergence rates (L2):")
    print(f"    {'Scheme+TI':<20}  {'pairwise rates':<30}  {'endpoint':>10}")
    for ti in time_integrators:
        for scheme in schemes:
            sr = sorted([r for r in all_results
                         if r["scheme"] == scheme and r["time_integrator"] == ti],
                        key=lambda r: r["n_lon"])
            if len(sr) < 2 or sr[-1]["l2"] <= 0 or sr[0]["l2"] <= 0:
                continue
            # Pairwise rates
            pw = []
            for i in range(1, len(sr)):
                if sr[i]["l2"] > 0 and sr[i-1]["l2"] > 0:
                    r_pw = (np.log(sr[i-1]["l2"] / sr[i]["l2"])
                            / np.log(sr[i]["n_lon"] / sr[i-1]["n_lon"]))
                    pw.append(r_pw)
            endpoint = (np.log(sr[0]["l2"] / sr[-1]["l2"])
                        / np.log(sr[-1]["n_lon"] / sr[0]["n_lon"]))
            pw_str = ", ".join(f"{r:.2f}" for r in pw)
            label = f"{scheme}+{ti}"
            print(f"    {label:<20}  {pw_str:<30}  {endpoint:>10.2f}")
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # Save summary
    level1_dir.mkdir(parents=True, exist_ok=True)
    with open(level1_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    # Plots — only for the first integrator (simpler 4-panel layout)
    try:
        for r in all_results:
            npz = level1_dir / f"{r['scheme']}_{r['time_integrator']}_{r['n_lon']}.npz"
            if npz.exists():
                _plot_4panel(npz, level1_dir)
        # Cross-section compare and convergence are per-integrator
        for ti in time_integrators:
            sub = [r for r in all_results if r["time_integrator"] == ti]
            if sub:
                _plot_cross_section_compare(sub, level1_dir, level1_dir,
                                            tag=ti)
                _plot_convergence(sub, level1_dir, tag=ti)
        print(f"\n  Plots saved to: {level1_dir}")
    except (ImportError, TypeError):
        # TypeError if plot helpers don't accept tag kwarg yet
        print("  (some plots may be skipped — helpers need tag support)")


if __name__ == "__main__":
    main()
