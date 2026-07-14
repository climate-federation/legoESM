#!/usr/bin/env python3
"""Benchmark: Spectral Hydrostatic PE — Jablonowski-Williamson Baroclinic Wave.

Runs:
  1. Isothermal rest-state stability check (10 days, T21/L20).
  2. JW06 baroclinic wave (10 days, T42/L20).
     Expected: wave breaks around day 7-9 with ~30 m/s surface winds,
     surface pressure anomaly ~40 hPa, ~10 K T anomaly at 850 hPa.

Outputs (in results/atmosphere/hydrostatic/spectral_pe/):
  - conservation.png      — mass (dry), energy time series
  - timeseries.png        — global-mean T, max wind, min p_s
  - profiles_T.png        — zonal-mean T at days 0, 4, 7, 10
  - profiles_u.png        — zonal-mean u at days 0, 4, 7, 10
  - snapshots_T850.png    — T at 850 hPa (lat-lon) at days 0, 4, 7, 10
  - snapshots_ps.png      — surface pressure (lat-lon) at days 0, 4, 7, 10
  - snapshots_vor.png     — 500 hPa relative vorticity at days 0, 4, 7, 10
  - rest_stability.png    — rest-state T drift
  - diagnostics.npz       — all diagnostic data
  - summary.txt           — quantitative pass/fail

References
----------
  Jablonowski, C. & Williamson, D. L. (2006). A baroclinic instability
  test case for atmospheric model dynamical cores. Quart. J. Royal
  Meteor. Soc., 132, 2943-2975.
"""

import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_synthesis,
    sh_synthesis_3d,
    uv_from_vordiv_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral
from legoesm import constants

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_DIR, "results", "atmosphere", "hydrostatic", "spectral_pe")
os.makedirs(OUT_DIR, exist_ok=True)


def proper_hyperdiff(grid, tau_hours=4.0):
    """∇⁴ hyperdiffusion with given e-folding time at truncation wavenumber."""
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (tau_hours * 3600.0 * eig_max**2)


def state_to_grid(state, grid, sigma_coord):
    """Extract grid-point fields from spectral PE state."""
    d = spectral_pe_to_grid(state, grid, sigma_coord)
    return d


def compute_pe_diagnostics(state, grid, sigma_coord):
    """Compute scalar diagnostics: mass, energy, mean T, max wind, min p_s."""
    d = state_to_grid(state, grid, sigma_coord)
    T = d["T"]         # (n_lat, n_lon, nlev)
    u = d["u"]
    v = d["v"]
    p_s = d["p_s"]     # (n_lat, n_lon)

    # Area weights (Gaussian quadrature + uniform longitude)
    w = grid.weights  # (n_lat,)
    n_lon = grid.n_lon
    dlon = 2.0 * np.pi / n_lon
    area_w = w[:, None] * dlon  # (n_lat, 1) broadcasts with (n_lat, n_lon)
    area_total = float(jnp.sum(w)) * n_lon * dlon  # = 4*pi

    # Dry mass = integral(p_s * area) / g
    dry_mass = float(jnp.sum(p_s * area_w)) / constants.g

    # Global mean T (mass-weighted)
    dsigma = sigma_coord.dsigma  # (nlev,)
    T_mean = float(jnp.sum(
        T * area_w[:, :, None] * dsigma[None, None, :]
    ) / (area_total * jnp.sum(dsigma)))

    # Max wind speed
    wind = jnp.sqrt(u**2 + v**2)
    max_wind = float(jnp.max(wind))

    # Min surface pressure
    min_ps = float(jnp.min(p_s))

    # Kinetic energy
    KE = float(jnp.sum(
        0.5 * (u**2 + v**2) * p_s[:, :, None] * dsigma[None, None, :] *
        area_w[:, :, None]
    ) / constants.g)

    return {
        "dry_mass": dry_mass,
        "T_mean": T_mean,
        "max_wind": max_wind,
        "min_ps": min_ps,
        "KE": KE,
    }


def zonal_mean(field, grid):
    """Compute zonal mean (lon average) of a lat-lon[-lev] field."""
    return np.mean(np.array(field), axis=1)


def field_at_sigma(field_3d, sigma_coord, target_sigma):
    """Extract field at nearest sigma level."""
    sigma = np.array(sigma_coord.sigma_full)
    k = np.argmin(np.abs(sigma - target_sigma))
    return np.array(field_3d[:, :, k])


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("  Spectral Hydrostatic PE Benchmark")
    print("=" * 70)

    # ── 1. Rest-state stability (T21/L20, 10 days) ───────────────────
    print("\n" + "-" * 50)
    print("  Part 1: Isothermal Rest-State Stability (T21/L20)")
    print("-" * 50)

    grid21 = create_gaussian_grid(n_max=21, dealiasing="cubic")
    sigma20 = create_sigma_coordinate(20)
    nu21 = proper_hyperdiff(grid21)
    config21 = SpectralPEConfig(hyperdiff_coeff=nu21, hyperdiff_order=2)
    model21 = SpectralPrimitiveEquationModel(grid21, sigma20, config21,
                                              allow_unsupported_backend=True)
    state_rest = isothermal_rest_state_spectral(grid21, sigma20, T_init=300.0)

    dt_rest = 600.0
    n_steps_rest = int(10 * 86400 / dt_rest)
    diag_interval_rest = int(86400 / dt_rest)  # daily

    rest_times = [0.0]
    rest_T_mean = [300.0]
    rest_T_drift = [0.0]

    print(f"  Integrating 10 days, dt={dt_rest:.0f}s")
    t0 = time.time()
    state = state_rest
    for step in range(n_steps_rest):
        state = model21.step(state, dt_rest)
        if (step + 1) % diag_interval_rest == 0:
            day = (step + 1) * dt_rest / 86400.0
            d = state_to_grid(state, grid21, sigma20)
            T_mean = float(jnp.mean(d["T"]))
            rest_times.append(day)
            rest_T_mean.append(T_mean)
            rest_T_drift.append(T_mean - 300.0)
            print(f"    Day {day:5.1f}: <T> = {T_mean:.6f} K, drift = {T_mean-300:.2e} K")

    jax.block_until_ready(state.vor_hat.data)
    print(f"  Done in {time.time()-t0:.1f}s")

    rest_pass = abs(rest_T_drift[-1]) < 0.01  # < 10 mK drift in 10 days

    # ── 2. JW06 Baroclinic Wave (T42/L20, 10 days) ───────────────────
    print("\n" + "-" * 50)
    print("  Part 2: Jablonowski-Williamson Baroclinic Wave (T42/L20)")
    print("-" * 50)

    n_max = 42
    nlev = 20
    grid = create_gaussian_grid(n_max=n_max, dealiasing="cubic")
    sigma = create_sigma_coordinate(nlev)
    nu = proper_hyperdiff(grid)
    # Leapfrog + SI with spectral filter and sponge layer.
    # - Leapfrog is neutral for oscillatory gravity-wave modes
    # - SI treats gravity waves implicitly (Hoskins & Simmons 1975)
    # - Spectral filter prevents truncation-wavenumber energy buildup
    # - Sponge layer damps upper-level artifacts from 1/p amplification
    # - Level-dependent diffusion (pscale) strengthens damping at low-p
    config = SpectralPEConfig(
        hyperdiff_coeff=nu,
        hyperdiff_order=2,
        time_integrator="leapfrog",
        semi_implicit=True,
        si_T_ref=300.0,
        si_alpha=0.5,
        robert_asselin_coeff=0.05,
        sponge_sigma=0.1,
        sponge_tau=2.0 * 3600.0,       # 2-hour e-folding at model top
        hyperdiff_pscale=0.5,           # level-dependent diffusion
        spectral_filter_order=8,
        spectral_filter_strength=0.01,  # 99% removal at n_max
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config,
                                            allow_unsupported_backend=True)
    print(f"  Transform grid: {grid.n_lat}x{grid.n_lon} (cubic dealiasing)")
    print(f"  Time integrator: leapfrog + SI (alpha={config.si_alpha})")
    print(f"  Robert-Asselin: gamma={config.robert_asselin_coeff}")
    print(f"  Spectral filter: order {config.spectral_filter_order}, "
          f"strength {config.spectral_filter_strength}")

    state0 = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)
    dt = 300.0  # 5 min
    n_days = 10
    n_steps = int(n_days * 86400 / dt)

    # Diagnostic cadence: every 6 hours
    diag_interval = int(6 * 3600 / dt)
    snap_days = [0, 4, 7, 10]

    print(f"  Grid: T{n_max}, nlev={nlev}, dt={dt:.0f}s, {n_steps} steps")

    # Initial fields
    fields0 = state_to_grid(state0, grid, sigma)
    diag0 = compute_pe_diagnostics(state0, grid, sigma)

    times = [0.0]
    mass_ts = [diag0["dry_mass"]]
    T_mean_ts = [diag0["T_mean"]]
    max_wind_ts = [diag0["max_wind"]]
    min_ps_ts = [diag0["min_ps"]]
    KE_ts = [diag0["KE"]]

    snapshots = {0: fields0}

    state = state0
    t0 = time.time()
    # JIT warmup
    state = model.step(state, dt)
    jax.block_until_ready(state.vor_hat.data)
    print(f"  JIT compiled in {time.time()-t0:.1f}s")

    t_wall = time.time()
    for step in range(1, n_steps):
        state = model.step(state, dt)

        if (step + 1) % diag_interval == 0:
            day = (step + 1) * dt / 86400.0
            diag = compute_pe_diagnostics(state, grid, sigma)
            times.append(day)
            mass_ts.append(diag["dry_mass"])
            T_mean_ts.append(diag["T_mean"])
            max_wind_ts.append(diag["max_wind"])
            min_ps_ts.append(diag["min_ps"])
            KE_ts.append(diag["KE"])

            # Snapshot?
            for sd in snap_days:
                if abs(day - sd) < dt / 86400.0 * 2:
                    if sd not in snapshots:
                        snapshots[sd] = state_to_grid(state, grid, sigma)

            if abs(day - round(day)) < 0.1:
                print(f"    Day {day:5.1f}: <T>={diag['T_mean']:.2f}K, "
                      f"max|v|={diag['max_wind']:.1f}m/s, "
                      f"min(ps)={diag['min_ps']/100:.1f}hPa")

    wall_time = time.time() - t_wall
    print(f"  Done in {wall_time:.1f}s ({n_steps/wall_time:.0f} steps/s)")

    # Ensure final snapshot
    if n_days not in snapshots:
        snapshots[n_days] = state_to_grid(state, grid, sigma)

    # ── Quantitative checks ───────────────────────────────────────────
    times = np.array(times)
    mass_ts = np.array(mass_ts)
    T_mean_ts = np.array(T_mean_ts)
    max_wind_ts = np.array(max_wind_ts)
    min_ps_ts = np.array(min_ps_ts)
    KE_ts = np.array(KE_ts)

    # iter-92 audit followup: previously
    # ``abs(mass_ts[-1] - mass_ts[0]) / abs(mass_ts[0])`` would
    # NaN if ``mass_ts[0] = 0`` (rest state).  Migrate to the
    # shared helper for the iter-78/80 floor convention.
    from legoesm.diagnostics.conservation_drift import compute_relative_drift
    mass_drift = compute_relative_drift(mass_ts)
    final_max_wind = max_wind_ts[-1]
    final_min_ps = min_ps_ts[-1] / 100.0  # hPa

    print(f"\n  JW06 Results after {n_days} days:")
    print(f"    Mass drift:   {mass_drift:.2e}  (expect < 1e-6)")
    print(f"    Max wind:     {final_max_wind:.1f} m/s  (expect 25-40 m/s)")
    print(f"    Min p_s:      {final_min_ps:.1f} hPa  (expect 960-980 hPa)")
    print(f"    <T> drift:    {T_mean_ts[-1] - T_mean_ts[0]:.3f} K")

    jw_pass = (np.all(np.isfinite(max_wind_ts)) and
               final_max_wind > 15.0 and final_max_wind < 60.0 and
               mass_drift < 1e-4)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat_deg = np.degrees(np.array(grid.lat))
    lon_deg = np.degrees(np.array(grid.lon2d[0]))
    sigma_full = np.array(sigma.sigma_full)

    # 1. Conservation
    # iter-92 audit followup: previously inlined
    # ``(KE_ts - KE_ts[0]) / max(abs(KE_ts[0]), 1e-30)`` (the
    # iter-78/80 pathology pattern) and
    # ``(mass_ts - mass_ts[0]) / abs(mass_ts[0])`` (NaN if
    # mass_ts[0] = 0).  Both replaced by the shared
    # ``relative_drift_series`` helper which uses a 1.0 floor and
    # returns absolute drift in natural units when the baseline is
    # near zero.  For JW06 atmosphere (mass_ts[0] ~ 5e+19 Pa·m²,
    # KE_ts[0] > 0) the floor never bites; this is a
    # single-source-of-truth unification.
    from legoesm.diagnostics.conservation_drift import relative_drift_series
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(times, relative_drift_series(mass_ts), "b-", lw=1.5)
    axes[0].set_ylabel("Rel. mass error")
    axes[0].set_title("Spectral PE — JW06 Baroclinic Wave: Conservation")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, relative_drift_series(KE_ts), "r-", lw=1.5)
    axes[1].set_ylabel("Rel. KE change")
    axes[1].set_xlabel("Time [days]")
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "conservation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved conservation.png")

    # 2. Time series
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, T_mean_ts, "k-", lw=1.5)
    axes[0].set_ylabel("Global <T> [K]")
    axes[0].set_title("Spectral PE — JW06: Mean Diagnostics")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, max_wind_ts, "b-", lw=1.5)
    axes[1].set_ylabel("Max |wind| [m/s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, min_ps_ts / 100.0, "r-", lw=1.5)
    axes[2].set_ylabel("Min p_s [hPa]")
    axes[2].set_xlabel("Time [days]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved timeseries.png")

    # 3. Zonal-mean T profiles at different times
    fig, axes = plt.subplots(1, len(snap_days), figsize=(5 * len(snap_days), 5))
    for i, sd in enumerate(snap_days):
        if sd in snapshots:
            T_zm = zonal_mean(snapshots[sd]["T"], grid)  # (n_lat, nlev)
            im = axes[i].contourf(lat_deg, sigma_full, T_zm.T, levels=20, cmap="RdYlBu_r")
            axes[i].invert_yaxis()
            axes[i].set_title(f"Day {sd}")
            axes[i].set_xlabel("Latitude")
            if i == 0:
                axes[i].set_ylabel("Sigma")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="T [K]")
    fig.suptitle("Zonal-Mean Temperature (T42/L20)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "profiles_T.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved profiles_T.png")

    # 4. Zonal-mean u profiles
    fig, axes = plt.subplots(1, len(snap_days), figsize=(5 * len(snap_days), 5))
    for i, sd in enumerate(snap_days):
        if sd in snapshots:
            u_zm = zonal_mean(snapshots[sd]["u"], grid)
            im = axes[i].contourf(lat_deg, sigma_full, u_zm.T, levels=20, cmap="RdBu_r")
            axes[i].invert_yaxis()
            axes[i].set_title(f"Day {sd}")
            axes[i].set_xlabel("Latitude")
            if i == 0:
                axes[i].set_ylabel("Sigma")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="u [m/s]")
    fig.suptitle("Zonal-Mean Zonal Wind (T42/L20)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "profiles_u.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved profiles_u.png")

    # 5. T at ~850 hPa (sigma ~ 0.85) lat-lon snapshots
    fig, axes = plt.subplots(1, len(snap_days), figsize=(5 * len(snap_days), 4))
    for i, sd in enumerate(snap_days):
        if sd in snapshots:
            T_850 = field_at_sigma(snapshots[sd]["T"], sigma, 0.85)
            im = axes[i].pcolormesh(lon_deg, lat_deg, T_850, cmap="RdYlBu_r", shading="auto")
            axes[i].set_title(f"Day {sd}")
            axes[i].set_xlabel("Longitude")
            if i == 0:
                axes[i].set_ylabel("Latitude")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="T [K]")
    fig.suptitle("Temperature at 850 hPa (T42/L20)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "snapshots_T850.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved snapshots_T850.png")

    # 6. Surface pressure snapshots
    fig, axes = plt.subplots(1, len(snap_days), figsize=(5 * len(snap_days), 4))
    for i, sd in enumerate(snap_days):
        if sd in snapshots:
            ps = np.array(snapshots[sd]["p_s"]) / 100.0  # hPa
            im = axes[i].pcolormesh(lon_deg, lat_deg, ps, cmap="RdBu_r", shading="auto")
            axes[i].set_title(f"Day {sd}")
            axes[i].set_xlabel("Longitude")
            if i == 0:
                axes[i].set_ylabel("Latitude")
            plt.colorbar(im, ax=axes[i], shrink=0.8, label="p_s [hPa]")
    fig.suptitle("Surface Pressure (T42/L20)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "snapshots_ps.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved snapshots_ps.png")

    # 7. 500 hPa vorticity snapshots
    fig, axes = plt.subplots(1, len(snap_days), figsize=(5 * len(snap_days), 4))
    for i, sd in enumerate(snap_days):
        if sd in snapshots:
            vor_500 = field_at_sigma(snapshots[sd]["vor"], sigma, 0.5) if "vor" in snapshots[sd] else None
            if vor_500 is not None:
                vmax = max(abs(np.min(vor_500)), abs(np.max(vor_500)))
                vmax = max(vmax, 1e-6)
                im = axes[i].pcolormesh(lon_deg, lat_deg, vor_500 * 1e5,
                                         cmap="RdBu_r", shading="auto")
                axes[i].set_title(f"Day {sd}")
                axes[i].set_xlabel("Longitude")
                if i == 0:
                    axes[i].set_ylabel("Latitude")
                plt.colorbar(im, ax=axes[i], shrink=0.8, label="vor [1e-5/s]")
    fig.suptitle("Relative Vorticity at 500 hPa (T42/L20)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "snapshots_vor.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved snapshots_vor.png")

    # 8. Rest-state stability plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(rest_times, rest_T_drift, "k-o", ms=4, lw=1.5)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("T drift [K]")
    ax.set_title("Rest-State Temperature Drift (T21/L20)")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rest_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved rest_stability.png")

    # ── Save data ─────────────────────────────────────────────────────
    np.savez(
        os.path.join(OUT_DIR, "diagnostics.npz"),
        times=times, mass=mass_ts, T_mean=T_mean_ts,
        max_wind=max_wind_ts, min_ps=min_ps_ts, KE=KE_ts,
        lat_deg=lat_deg, lon_deg=lon_deg, sigma=sigma_full,
        rest_times=np.array(rest_times), rest_T_drift=np.array(rest_T_drift),
    )
    print(f"  Saved diagnostics.npz")

    # ── Summary ───────────────────────────────────────────────────────
    summary = []
    summary.append("Spectral Hydrostatic PE Benchmark Summary")
    summary.append("=" * 50)
    summary.append("")
    summary.append("Part 1: Isothermal Rest State (T21/L20, 10 days)")
    summary.append(f"  Final T drift:   {rest_T_drift[-1]:.2e} K  (threshold: 0.01 K)")
    summary.append(f"  PASS: {rest_pass}")
    summary.append("")
    summary.append("Part 2: JW06 Baroclinic Wave (T42/L20, 10 days)")
    summary.append(f"  Mass drift:      {mass_drift:.2e}  (expect < 1e-4)")
    summary.append(f"  Max wind (d10):  {final_max_wind:.1f} m/s  (expect 25-40 m/s)")
    summary.append(f"  Min p_s (d10):   {final_min_ps:.1f} hPa  (expect 960-980 hPa)")
    summary.append(f"  <T> drift:       {T_mean_ts[-1] - T_mean_ts[0]:.3f} K")
    summary.append(f"  All finite:      {bool(np.all(np.isfinite(max_wind_ts)))}")
    summary.append(f"  PASS: {jw_pass}")
    summary.append("")
    summary.append("Literature reference: Jablonowski & Williamson (2006)")
    summary.append("  JW06 Fig. 7: day-9 surface pressure anomaly ~40 hPa")
    summary.append("  JW06 Fig. 8: day-9 T anomaly at 850 hPa ~10 K")
    summary.append("  JW06 Table 4: max wind at day 9 ~ 30 m/s (T42)")
    summary.append("  Wave breaks around day 7-9 in NH midlatitudes")

    summary_text = "\n".join(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")

    return rest_pass and jw_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
