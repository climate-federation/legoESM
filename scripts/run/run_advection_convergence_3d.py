"""Level 3: Vertical advection in a 2D zonal-vertical overturning cell.

Tests the full 3D tracer update (horizontal u + vertical w + flux divergence)
with prescribed flow that is non-divergent and time-reversing — the bell
deforms during the first half period, then the flow reverses, and the bell
should return to its initial position at t = T.

Stream function approach:
  ψ(x, z, t) = ψ₀ * sin(2π*x/Lx) * sin(π*z/H) * cos(π*t/T)

Velocity:
  u = -∂ψ/∂z = -ψ₀ * (π/H) * sin(2π*x/Lx) * cos(π*z/H) * cos(π*t/T)
  w = +∂ψ/∂x = ψ₀ * (2π/Lx) * cos(2π*x/Lx) * sin(π*z/H) * cos(π*t/T)

Properties:
  - ∂u/∂x + ∂w/∂z = 0 exactly (mass-conserving with constant h)
  - w = 0 at z=0 (top) and z=H (bottom): closed vertical walls
  - u and w periodic in x

Flow forms two counter-rotating cells in the x-z plane that reverse at
t = T/2 and undo their action at t = T.

Tracer: 2D Gaussian blob in (x, z) at center (Lx/4, H/2). The bell
travels through both cells over the period.

Usage
-----
Quick:
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence_3d.py --quick

Full:
    JAX_ENABLE_X64=1 python scripts/run_advection_convergence_3d.py
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

from legoesm import constants

jax.config.update("jax_enable_x64", True)

ALL_SCHEMES = ["upwind", "tvd", "dst3", "weno5", "weno7"]
ALL_INTEGRATORS = ["euler", "ab2", "rk3"]

# Resolutions: (n_lat, n_lon, n_lev). Keep n_lat small (just enough for the
# C-grid structure); the test is effectively 2D (x, z).
RESOLUTIONS = [(4, 32, 16), (4, 64, 32), (4, 128, 64)]
RESOLUTIONS_QUICK = [(4, 32, 16), (4, 64, 32)]

T_PERIOD = 5.0 * 86400.0  # 5 days, deform + reverse


# ---------------------------------------------------------------------------
# Prescribed 2D zonal-vertical flow (overturning cell)
# ---------------------------------------------------------------------------

def _overturning_velocity(lon_u, lon_v, lon_w, z_u, z_w, t, T=T_PERIOD,
                          Lx_rad=2.0 * jnp.pi, H=5500.0, psi0=3.5e4):
    """Compute u (m/s) at u-faces and w (m/s) at vertical interfaces.

    Stream function: ψ = ψ₀ sin(2πx/Lx) sin(πz/H) cos(πt/T)

    Parameters
    ----------
    lon_u, z_u : arrays at u-face positions (lon, z) in (rad, meters)
    lon_w, z_w : arrays at w-face positions (lon, z) in (rad, meters)
    Lx_rad : zonal extent in radians (default 2π for global)
    H : vertical extent in meters
    psi0 : stream function amplitude (m²/s) — sets the velocity scale

    Returns
    -------
    u_east : array at u-faces [m/s]
    w_vert : array at vertical interfaces [m/s]
    """
    R = constants.R_earth
    cos_phase = jnp.cos(jnp.pi * t / T)
    kx = 2.0 * jnp.pi / Lx_rad  # one full cycle in lon
    kz = jnp.pi / H              # half cycle in z (peak at H/2)

    # u = -∂ψ/∂z
    # In spherical coords, ∂/∂x = (1/(R*cos(lat))) * ∂/∂lon, but for our
    # zonal-vertical test we ignore the cos(lat) factor (use mid-channel).
    u = -psi0 * kz * jnp.sin(kx * lon_u) * jnp.cos(kz * z_u) * cos_phase

    # w = +∂ψ/∂x. Note: x = R*cos(lat)*lon, so ∂ψ/∂x = (1/(R*cos(lat))) * ∂ψ/∂lon.
    # Ignoring cos(lat), use R as effective radius.
    w = (psi0 * kx / R) * jnp.cos(kx * lon_w) * jnp.sin(kz * z_w) * cos_phase

    return u, w


# ---------------------------------------------------------------------------
# Tracer initial condition: 2D Gaussian in (x, z)
# ---------------------------------------------------------------------------

def _gaussian_blob_xz(lon_2d, z_2d, lon0, z0, sigma_lon, sigma_z):
    """2D Gaussian centered at (lon0, z0) in (radians, meters).

    Returns array same shape as lon_2d/z_2d.
    """
    dlon = lon_2d - lon0
    dlon = jnp.where(dlon > jnp.pi, dlon - 2 * jnp.pi, dlon)
    dlon = jnp.where(dlon < -jnp.pi, dlon + 2 * jnp.pi, dlon)
    return jnp.exp(-0.5 * ((dlon / sigma_lon) ** 2 + ((z_2d - z0) / sigma_z) ** 2))


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

def _create_grid(n_lat, n_lon, n_lev, H_max=5500.0):
    """Create periodic lat-lon grid + uniform z-coord."""
    from legoesm.grids.latlon import create_regional_latlon_grid
    grid, _ = create_regional_latlon_grid(
        n_lat, n_lon,
        lat_south=20.0, lat_north=30.0,
        lon_west=0.0, lon_east=360.0,
        periodic_x=True)

    # Uniform vertical coord: simple 1D arrays of cell-center and interface z
    dz = H_max / n_lev
    z_centers = np.array([(k + 0.5) * dz for k in range(n_lev)])  # +0.5 dz from top
    z_interfaces = np.array([k * dz for k in range(n_lev + 1)])
    return grid, z_centers, z_interfaces, dz, H_max


# ---------------------------------------------------------------------------
# 3D advection step (horizontal + vertical)
# ---------------------------------------------------------------------------

def _flux_divergence_3d(scheme, tracer, mass_flux_u, mass_flux_v,
                        w_half, h_u, h_v, h_cell, grid, dt):
    """Compute total (horizontal + vertical) flux divergence per unit volume.

    Returns
    -------
    rhs : array — the dt * rhs term for the FV update T_new = T - rhs/h.
        Specifically, returns div(h*u*T) + div_z(w*T) (shape n_lat, n_lon, n_lev).
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid

    # Horizontal scheme dispatch
    if scheme == "upwind":
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            upwind_to_u_points, upwind_to_v_points)
        from legoesm.ocean.vertical import flux_form_vertical_tracer_advection
        tr_u = upwind_to_u_points(tracer, mass_flux_u)
        tr_v = upwind_to_v_points(tracer, mass_flux_v)
        vert_flux_div = flux_form_vertical_tracer_advection(tracer, w_half)
    elif scheme == "tvd":
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            tvd_to_u_points, tvd_to_v_points)
        from legoesm.ocean.vertical import flux_form_vertical_tracer_advection_tvd
        tr_u = tvd_to_u_points(tracer, mass_flux_u)
        tr_v = tvd_to_v_points(tracer, mass_flux_v)
        vert_flux_div = flux_form_vertical_tracer_advection_tvd(
            tracer, w_half, h_cell, dt)
    elif scheme == "dst3":
        from legoesm.ocean.advection import (
            dst3_to_u_points, dst3_to_v_points,
            flux_form_vertical_tracer_advection_dst3)
        tr_u = dst3_to_u_points(tracer, mass_flux_u, h_u, grid, dt)
        tr_v = dst3_to_v_points(tracer, mass_flux_v, h_v, grid, dt)
        vert_flux_div = flux_form_vertical_tracer_advection_dst3(
            tracer, w_half, h_cell, dt)
    elif scheme == "weno5":
        from legoesm.ocean.advection import (
            weno5_to_u_points, weno5_to_v_points,
            flux_form_vertical_tracer_advection_weno5)
        tr_u = weno5_to_u_points(tracer, mass_flux_u)
        tr_v = weno5_to_v_points(tracer, mass_flux_v)
        vert_flux_div = flux_form_vertical_tracer_advection_weno5(
            tracer, w_half, h_cell, dt)
    elif scheme == "weno7":
        from legoesm.ocean.advection import (
            weno7_to_u_points, weno7_to_v_points,
            flux_form_vertical_tracer_advection_weno7)
        tr_u = weno7_to_u_points(tracer, mass_flux_u)
        tr_v = weno7_to_v_points(tracer, mass_flux_v)
        vert_flux_div = flux_form_vertical_tracer_advection_weno7(
            tracer, w_half, h_cell, dt)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    div_hut = divergence_cgrid(mass_flux_u * tr_u, mass_flux_v * tr_v, grid)
    return div_hut + vert_flux_div


def _run_single_3d(scheme, n_lat, n_lon, n_lev, output_dir,
                   time_integrator="euler", ab2_eps=0.1):
    """Run Level 3 overturning cell test for one scheme+integrator+resolution."""
    h_uniform = 100.0  # cell layer thickness — but we override via z_coord

    grid, z_centers, z_interfaces, dz, H_max = _create_grid(n_lat, n_lon, n_lev)
    n_lat_grid, n_lon_actual = grid.area.shape

    # Cell layer thickness (uniform = dz)
    h_cell = jnp.full((n_lat_grid, n_lon_actual, n_lev), float(dz))
    h_u = jnp.full((n_lat_grid, n_lon_actual + 1, n_lev), float(dz))
    h_v = jnp.full((n_lat_grid + 1, n_lon_actual, n_lev), float(dz))

    # 2D arrays: cell-center lon and z (the test field is uniform in lat)
    lon_cell = jnp.array(grid.lon)  # (n_lon,) radians
    # cell-center grids
    lon_2d = jnp.broadcast_to(
        lon_cell[jnp.newaxis, :, jnp.newaxis],
        (n_lat_grid, n_lon_actual, n_lev))
    z_centers_jnp = jnp.array(z_centers)
    z_2d = jnp.broadcast_to(
        z_centers_jnp[jnp.newaxis, jnp.newaxis, :],
        (n_lat_grid, n_lon_actual, n_lev))

    # Initial condition: Gaussian blob at (lon=π/2, z=H/2)
    # Sigma chosen so the blob spans ~25% of the domain in each direction.
    sigma_lon = jnp.pi / 4.0   # 45° in longitude
    sigma_z = H_max / 6.0      # ~17% of depth
    # Place bell at lon=0 where vertical flow is at max magnitude
    # (cos(0)=1 in the w field). At z=H/2, w=w_max here.
    bell_lon0 = 0.0
    bell_z0 = H_max / 2.0
    T_init_3d = _gaussian_blob_xz(lon_2d, z_2d, bell_lon0, bell_z0,
                                  sigma_lon, sigma_z)

    tracer = T_init_3d  # already (n_lat, n_lon, n_lev)

    # Mass flux face coordinates
    dlon = float(grid.dlon)
    lon_u_1d = jnp.concatenate([lon_cell - dlon / 2,
                                jnp.array([lon_cell[-1] + dlon / 2])])  # (n_lon+1,)
    # u-face: shape (n_lat_grid, n_lon+1, n_lev)
    lon_u_3d = jnp.broadcast_to(
        lon_u_1d[jnp.newaxis, :, jnp.newaxis],
        (n_lat_grid, n_lon_actual + 1, n_lev))
    z_u_3d = jnp.broadcast_to(
        z_centers_jnp[jnp.newaxis, jnp.newaxis, :],
        (n_lat_grid, n_lon_actual + 1, n_lev))

    # w-face: shape (n_lat, n_lon, n_lev+1) at vertical interfaces
    z_interfaces_jnp = jnp.array(z_interfaces)
    lon_w_3d = jnp.broadcast_to(
        lon_cell[jnp.newaxis, :, jnp.newaxis],
        (n_lat_grid, n_lon_actual, n_lev + 1))
    z_w_3d = jnp.broadcast_to(
        z_interfaces_jnp[jnp.newaxis, jnp.newaxis, :],
        (n_lat_grid, n_lon_actual, n_lev + 1))

    # Estimate dt from CFL constraint at the maximum velocity
    R = float(grid.radius)
    psi0 = 3.5e4  # gives u_max ≈ 20 m/s, w_max ≈ 5.5e-3 m/s ≈ 475 m/day
    Lx_rad = 2.0 * float(jnp.pi)
    kx = 2 * float(jnp.pi) / Lx_rad
    kz = float(jnp.pi) / H_max
    u_max = psi0 * kz       # max |u|
    w_max = psi0 * kx / R   # max |w|

    cos_mid = float(jnp.cos(jnp.radians(25.0)))
    dx_min = R * dlon * cos_mid
    cfl_u = u_max / dx_min
    cfl_w = w_max / dz
    dt_cfl = 0.3 / max(cfl_u, cfl_w)
    n_steps = int(np.ceil(T_PERIOD / dt_cfl))
    dt = T_PERIOD / n_steps

    print(f"    {scheme}+{time_integrator} @ {n_lat}x{n_lon}x{n_lev}: "
          f"n_steps={n_steps}, dt={dt:.0f}s, "
          f"u_max={u_max:.2f}m/s, w_max={w_max*86400:.2f}m/d")

    # Snapshot times
    snap_steps = {0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps}
    snapshots = {0: np.asarray(tracer)}

    def _mass_flux_at(t_eval):
        u_vel, w_vel = _overturning_velocity(
            lon_u_3d, None, lon_w_3d, z_u_3d, z_w_3d,
            t_eval, T=T_PERIOD, Lx_rad=Lx_rad, H=H_max, psi0=psi0)
        # mass_flux_u = h * u
        mf_u = h_u * u_vel
        # mass_flux_v = 0 (no meridional flow in this test)
        mf_v = jnp.zeros((n_lat_grid + 1, n_lon_actual, n_lev))
        # w at vertical interfaces (n_lat, n_lon, n_lev+1)
        return mf_u, mf_v, w_vel

    blew_up = False
    rhs_prev = None  # for AB2

    t0 = time.time()
    for step in range(n_steps):
        t_current = step * dt

        if time_integrator == "euler":
            mfu, mfv, w = _mass_flux_at(t_current)
            rhs = _flux_divergence_3d(scheme, tracer, mfu, mfv, w,
                                      h_u, h_v, h_cell, grid, dt)
            tracer = tracer - (dt / h_cell) * rhs

        elif time_integrator == "ab2":
            mfu, mfv, w = _mass_flux_at(t_current)
            rhs = _flux_divergence_3d(scheme, tracer, mfu, mfv, w,
                                      h_u, h_v, h_cell, grid, dt)
            if rhs_prev is None:
                tracer = tracer - (dt / h_cell) * rhs
            else:
                eff = (1.5 + ab2_eps) * rhs - (0.5 + ab2_eps) * rhs_prev
                tracer = tracer - (dt / h_cell) * eff
            rhs_prev = rhs

        elif time_integrator == "rk3":
            mfu0, mfv0, w0 = _mass_flux_at(t_current)
            rhs0 = _flux_divergence_3d(
                scheme, tracer, mfu0, mfv0, w0, h_u, h_v, h_cell, grid, dt)
            k1 = tracer - (dt / h_cell) * rhs0

            mfu1, mfv1, w1 = _mass_flux_at(t_current + dt)
            rhs1 = _flux_divergence_3d(
                scheme, k1, mfu1, mfv1, w1, h_u, h_v, h_cell, grid, dt)
            k2 = 0.75 * tracer + 0.25 * (k1 - (dt / h_cell) * rhs1)

            mfu2, mfv2, w2 = _mass_flux_at(t_current + 0.5 * dt)
            rhs2 = _flux_divergence_3d(
                scheme, k2, mfu2, mfv2, w2, h_u, h_v, h_cell, grid, dt)
            tracer = (1.0 / 3.0) * tracer + (2.0 / 3.0) * (
                k2 - (dt / h_cell) * rhs2)
        else:
            raise ValueError(f"Unknown time_integrator: {time_integrator}")

        if (step + 1) in snap_steps:
            snapshots[step + 1] = np.asarray(tracer)

        if (step + 1) % max(1, n_steps // 10) == 0:
            if not bool(jnp.all(jnp.isfinite(tracer))):
                print(f"      BLOWUP at step {step+1}/{n_steps}")
                blew_up = True
                break

    jax.block_until_ready(tracer)
    wall = time.time() - t0

    # Error: compare final to initial (deformational test → returns to IC at t=T)
    tracer_final = np.asarray(tracer)
    tracer_init_np = np.asarray(T_init_3d)
    error = tracer_final - tracer_init_np
    area = np.asarray(grid.area)  # (n_lat, n_lon)
    # Volume weights: area * dz
    vol = area[..., np.newaxis] * dz  # (n_lat, n_lon, n_lev)

    l1 = float(np.sum(np.abs(error) * vol) /
               np.sum(np.abs(tracer_init_np) * vol))
    l2 = float(np.sqrt(np.sum(error**2 * vol) /
                       np.sum(tracer_init_np**2 * vol)))
    linf = float(np.max(np.abs(error)) / np.max(np.abs(tracer_init_np)))

    # Conservation drift via the centralized helper (iter-159
    # — same migration as iter-157 cosine_bell).  The inline
    # ``abs(mass_final - mass_init) / abs(mass_init)`` pattern
    # is the iter-78 pathology that returns NaN-as-False on
    # blown-up runs.  The helper is NaN-aware and uses a
    # ``DEFAULT_MIN_BASELINE = 1.0`` floor.
    mass_init = float(np.sum(tracer_init_np * vol))
    mass_final = float(np.sum(tracer_final * vol))
    from legoesm.diagnostics import compute_relative_drift
    mass_drift = compute_relative_drift([mass_init, mass_final])

    result = {
        "scheme": scheme, "time_integrator": time_integrator,
        "n_lat": n_lat, "n_lon": n_lon, "n_lev": n_lev,
        "blew_up": blew_up,
        "l1": l1, "l2": l2, "linf": linf,
        "mass_drift": mass_drift, "wall": wall,
        "dt": dt, "n_steps": n_steps,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir
                 / f"{scheme}_{time_integrator}_{n_lon}x{n_lev}.npz",
                 tracer_init=tracer_init_np,
                 tracer_final=tracer_final,
                 error=error,
                 lon_deg=np.asarray(grid.lon) * 180 / np.pi,
                 z_centers=z_centers,
                 snapshots={str(k): v for k, v in snapshots.items()})

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_xz_snapshots(data_file, output_dir):
    """Plot x-z slice through the middle latitude for each snapshot time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(data_file, allow_pickle=True)
    name = data_file.stem
    lon = d["lon_deg"]
    z = d["z_centers"]
    snaps = d["snapshots"].item()
    steps = sorted([int(k) for k in snaps.keys()])

    n_snaps = len(steps)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 3.5),
                             squeeze=False)
    axes = axes[0]

    for i, step in enumerate(steps):
        field = snaps[str(step)]
        # take middle latitude slice → (n_lon, n_lev)
        mid_lat = field.shape[0] // 2
        slc = field[mid_lat, :, :].T  # (n_lev, n_lon) for plotting
        ax = axes[i]
        im = ax.pcolormesh(lon, -z, slc, cmap="RdYlBu_r",
                          vmin=-0.1, vmax=1.1)
        frac = step / max(steps) if max(steps) > 0 else 0
        ax.set_title(f"t/T = {frac:.2f}")
        ax.set_xlabel("Lon (deg)")
        if i == 0:
            ax.set_ylabel("Depth (m)")
    plt.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.suptitle(name.replace("_", " "), fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_snapshots.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Level 3: 3D vertical advection convergence test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--schemes", type=str, default=",".join(ALL_SCHEMES))
    p.add_argument("--time-integrators", type=str, default=",".join(ALL_INTEGRATORS))
    p.add_argument("--output", type=str,
                   default="results/advection_convergence/level3_3d")
    p.add_argument("--quick", action="store_true")
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
    print("  Level 3: 3D Vertical Advection Test (overturning cell)")
    print("=" * 70)
    print(f"  Schemes:     {', '.join(schemes)}")
    print(f"  Integrators: {', '.join(time_integrators)}")
    print(f"  Resolutions: {[f'{r[1]}x{r[2]}' for r in resolutions]}")
    print(f"  Period:      {T_PERIOD/86400:.0f} days (deform + reverse)")
    print(f"  Output:      {output_dir}")
    print("=" * 70)

    all_results = []
    t0 = time.time()

    for ti in time_integrators:
        for scheme in schemes:
            for n_lat, n_lon, n_lev in resolutions:
                result = _run_single_3d(scheme, n_lat, n_lon, n_lev, output_dir,
                                       time_integrator=ti)
                all_results.append(result)
                status = "BLEW UP" if result["blew_up"] else "ok"
                print(f"      L2={result['l2']:.2e}  Linf={result['linf']:.2e}  "
                      f"mass_drift={result['mass_drift']:.2e}  "
                      f"wall={result['wall']:.1f}s  [{status}]")

    total_wall = time.time() - t0

    # Summary
    print(f"\n{'='*110}")
    print(f"  LEVEL 3 SUMMARY")
    print(f"{'='*110}")
    print(f"  {'Scheme':<10} {'Integrator':<10} {'Resolution':>12} {'L1':>10} "
          f"{'L2':>10} {'Linf':>10} {'mass_drift':>12} {'status':>10}")
    print("-" * 110)
    for r in all_results:
        status = "BLEW UP" if r["blew_up"] else "ok"
        print(f"  {r['scheme']:<10} {r['time_integrator']:<10} "
              f"{r['n_lon']}x{r['n_lev']:>4} "
              f"{r['l1']:>10.2e} {r['l2']:>10.2e} "
              f"{r['linf']:>10.2e} {r['mass_drift']:>12.2e} {status:>10}")

    print(f"\n  Convergence rates (L2):")
    print(f"    {'Scheme+TI':<20}  {'pairwise rates':<30}  {'endpoint':>10}")
    for ti in time_integrators:
        for scheme in schemes:
            sr = sorted(
                [r for r in all_results
                 if r["scheme"] == scheme and r["time_integrator"] == ti
                 and not r["blew_up"]],
                key=lambda r: r["n_lon"])
            if len(sr) < 2 or sr[-1]["l2"] <= 0 or sr[0]["l2"] <= 0:
                continue
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

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    try:
        for r in all_results:
            npz = output_dir / (
                f"{r['scheme']}_{r['time_integrator']}_{r['n_lon']}x{r['n_lev']}.npz")
            if npz.exists():
                _plot_xz_snapshots(npz, output_dir)
        print(f"\n  Plots saved to: {output_dir}")
    except (ImportError, KeyError):
        pass


if __name__ == "__main__":
    main()
