#!/usr/bin/env python
"""Focused spectral Held-Suarez runner with harmonized diagnostics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm import constants
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import compute_geopotential, create_sigma_coordinate


def _snapshot_steps(n_steps: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    return sorted({0, max(1, n_steps // 2), n_steps})


def _fmt_time(step: int, dt: float) -> str:
    t_sec = step * dt
    if t_sec < 3600.0:
        return f"{t_sec/60.0:.1f} min"
    if t_sec < 86400.0:
        return f"{t_sec/3600.0:.2f} h"
    return f"{t_sec/86400.0:.2f} d"


def _weighted_mean_2d(field: np.ndarray, area2d: np.ndarray) -> float:
    return float(np.sum(field * area2d) / np.sum(area2d))


def _horizontal_profile_3d(field: np.ndarray, area2d: np.ndarray) -> np.ndarray:
    num = np.sum(field * area2d[..., None], axis=(0, 1))
    den = np.sum(area2d)
    return np.asarray(num / den, dtype=np.float64)


def _compute_mass_energy(fields: dict[str, np.ndarray], sigma, area2d: np.ndarray) -> tuple[float, float]:
    u = np.asarray(fields["u"], dtype=np.float64)
    v = np.asarray(fields["v"], dtype=np.float64)
    T = np.asarray(fields["T"], dtype=np.float64)
    p_s = np.asarray(fields["p_s"], dtype=np.float64)
    phis = np.asarray(fields["phis"], dtype=np.float64)
    dsigma = np.asarray(sigma.dsigma, dtype=np.float64)

    mass = float(np.sum(p_s * area2d))

    p_s_j = jnp.asarray(p_s)
    T_j = jnp.asarray(T)
    phis_j = jnp.asarray(phis)
    Phi = np.asarray(compute_geopotential(T_j, p_s_j, sigma, phis_j), dtype=np.float64)
    ke = 0.5 * (u * u + v * v) * p_s[..., None] * dsigma[None, None, :]
    ie = constants.c_vd * T * p_s[..., None] * dsigma[None, None, :]
    pe = Phi * p_s[..., None] * dsigma[None, None, :]
    energy = float(np.sum(np.sum(ke + ie + pe, axis=-1) * area2d) / constants.g)

    return mass, energy


def _save_snapshot_times(out_dir: Path, steps: list[int], dt: float) -> None:
    with open(out_dir / "snapshot_times.txt", "w") as f:
        f.write("step,time_seconds,time_days\n")
        for st in steps:
            t = st * dt
            f.write(f"{st},{t:.6f},{t/86400.0:.8f}\n")


def _save_snapshots(
    out_dir: Path,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    lon_deg_2d: np.ndarray,
    lat_deg_2d: np.ndarray,
) -> None:
    steps = sorted(snapshots.keys())
    fields = [
        ("wind_sfc", "Surface wind (m/s)", "magma"),
        ("p_s_anom", "Surface pressure anomaly (Pa)", "RdBu_r"),
        ("T_mid", "Mid-level temperature (K)", "coolwarm"),
    ]
    fig, axes = plt.subplots(len(fields), len(steps), figsize=(5.0 * len(steps), 3.8 * len(fields)))
    if len(fields) == 1:
        axes = np.array([axes])
    if len(steps) == 1:
        axes = np.array([[ax] for ax in axes[:, None].flatten()])

    for r, (key, label, cmap) in enumerate(fields):
        row_vals = [np.asarray(snapshots[s][key], dtype=np.float64) for s in steps]
        vmin = float(np.min([np.nanmin(v) for v in row_vals]))
        vmax = float(np.max([np.nanmax(v) for v in row_vals]))
        if vmin < 0 < vmax:
            m = max(abs(vmin), abs(vmax), 1.0e-12)
            vmin, vmax = -m, m
        for c, st in enumerate(steps):
            ax = axes[r, c]
            im = ax.pcolormesh(
                lon_deg_2d,
                lat_deg_2d,
                np.asarray(snapshots[st][key], dtype=np.float64),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
            ax.set_xlim(0.0, 360.0)
            ax.set_ylim(-90.0, 90.0)
            ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
            ax.set_yticks([-60, -30, 0, 30, 60])
            ax.grid(True, alpha=0.15)
            ax.set_title(f"{label}\nstep {st}, t={_fmt_time(st, dt)}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label)
            fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.03)

    fig.suptitle("Hydro Spectral Held-Suarez - Field Snapshots", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "field_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_snapshot_times(out_dir, steps, dt)


def _save_conservation(out_dir: Path, times_days, mass, energy):
    mass = np.asarray(mass, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    t = np.asarray(times_days, dtype=np.float64)
    mass_rel = (mass - mass[0]) / max(abs(mass[0]), 1.0e-30)
    energy_rel = (energy - energy[0]) / max(abs(energy[0]), 1.0e-30)

    with open(out_dir / "conservation_timeseries.csv", "w") as f:
        f.write("time_days,mass,energy,mass_drift_rel,energy_drift_rel\n")
        for i in range(t.size):
            f.write(f"{t[i]:.8f},{mass[i]:.12e},{energy[i]:.12e},{mass_rel[i]:.12e},{energy_rel[i]:.12e}\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, mass_rel, lw=1.8, color="tab:blue")
    axes[0].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[0].set_ylabel("Mass drift (rel.)")
    axes[0].set_title("Mass conservation")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, energy_rel, lw=1.8, color="tab:red")
    axes[1].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[1].set_ylabel("Energy drift (rel.)")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title("Energy conservation")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return float(mass_rel[-1]), float(energy_rel[-1])


def _save_slab(out_dir: Path, times_days, mean_ps, mean_T, max_wind):
    t = np.asarray(times_days, dtype=np.float64)
    ps = np.asarray(mean_ps, dtype=np.float64)
    T = np.asarray(mean_T, dtype=np.float64)
    w = np.asarray(max_wind, dtype=np.float64)
    with open(out_dir / "slab_timeseries.csv", "w") as f:
        f.write("time_days,mean_p_s,mean_T,max_wind\n")
        for i in range(t.size):
            f.write(f"{t[i]:.8f},{ps[i]:.12e},{T[i]:.12e},{w[i]:.12e}\n")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, ps, lw=1.8)
    axes[0].set_ylabel("Mean p_s [Pa]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Global mean surface pressure")
    axes[1].plot(t, T, lw=1.8, color="tab:red")
    axes[1].set_ylabel("Mean T [K]")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Global mean temperature")
    axes[2].plot(t, w, lw=1.8, color="tab:blue")
    axes[2].set_ylabel("Max wind [m/s]")
    axes[2].set_xlabel("Time (days)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Maximum wind speed")
    fig.tight_layout()
    fig.savefig(out_dir / "slab_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_profiles(out_dir: Path, profile_steps: list[int], dt: float, sigma_full: np.ndarray, profiles: dict[int, dict[str, np.ndarray]]):
    keys = ["wind_profile", "T_profile"]
    with open(out_dir / "vertical_profiles.csv", "w") as f:
        f.write("step,time_days,sigma,wind_profile,T_profile\n")
        for st in profile_steps:
            td = st * dt / 86400.0
            for k in range(sigma_full.size):
                wp = float(profiles[st]["wind_profile"][k])
                tp = float(profiles[st]["T_profile"][k])
                f.write(f"{st},{td:.8f},{sigma_full[k]:.8f},{wp:.12e},{tp:.12e}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for st in profile_steps:
        td = st * dt / 86400.0
        axes[0].plot(profiles[st]["wind_profile"], sigma_full, lw=1.8, label=f"{td:.2f} d")
        axes[1].plot(profiles[st]["T_profile"], sigma_full, lw=1.8, label=f"{td:.2f} d")
    for ax, title, xlabel in [
        (axes[0], "Wind profile", "Wind [m/s]"),
        (axes[1], "Temperature profile", "T [K]"),
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
        ax.invert_yaxis()
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Sigma")
    fig.tight_layout()
    fig.savefig(out_dir / "vertical_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run hydro spectral Held-Suarez with diagnostics.")
    parser.add_argument("--truncation", type=int, default=42)
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--dt", type=float, default=180.0)
    parser.add_argument("--days", type=float, default=5.0)
    parser.add_argument("--mean-every", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/atmosphere/hydrostatic/held_suarez_spectral"))
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Hydro Spectral Held-Suarez")
    print("=" * 70)
    print(f"T{args.truncation}, L{args.levels}, dt={args.dt}s, days={args.days}")
    print(f"Output: {out}")

    grid = create_gaussian_grid(args.truncation)
    sigma = create_sigma_coordinate(args.levels)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0, p_s_init=1e5)

    a = float(grid.radius)
    eig_max = args.truncation * (args.truncation + 1) / (a * a)
    config = SpectralPEConfig(
        hyperdiff_coeff=1.0 / (0.5 * 3600.0 * eig_max ** 2),
        hyperdiff_order=2,
        semi_implicit=False,
        time_integrator="ssp_rk54",
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config)

    n_steps = int(args.days * 86400.0 / args.dt)
    snap_steps = _snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {}

    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))
    lon2d, lat2d = np.meshgrid(lon_deg, lat_deg)
    area2d = np.broadcast_to(np.asarray(grid.weights, dtype=np.float64)[:, None] / float(grid.n_lon), lon2d.shape)

    k_mid = int(np.argmin(np.abs(np.asarray(sigma.sigma_full, dtype=np.float64) - 0.55)))

    times_days = []
    mass_series = []
    energy_series = []
    mean_ps_series = []
    mean_T_series = []
    max_wind_series = []

    def _extract_diag_and_optional(step: int, capture: bool):
        f = spectral_pe_to_grid(state, grid, sigma)
        u = np.asarray(f["u"], dtype=np.float64)
        v = np.asarray(f["v"], dtype=np.float64)
        T = np.asarray(f["T"], dtype=np.float64)
        p_s = np.asarray(f["p_s"], dtype=np.float64)
        wind = np.sqrt(u * u + v * v)
        mass, energy = _compute_mass_energy(f, sigma, area2d)
        T_prof = _horizontal_profile_3d(T, area2d)
        times_days.append(step * args.dt / 86400.0)
        mass_series.append(mass)
        energy_series.append(energy)
        mean_ps_series.append(_weighted_mean_2d(p_s, area2d))
        mean_T_series.append(float(np.mean(T_prof)))
        max_wind_series.append(float(np.max(wind)))
        if capture:
            snapshots[step] = {
                "wind_sfc": wind[..., -1],
                "p_s_anom": p_s - np.mean(p_s, axis=1, keepdims=True),
                "T_mid": T[..., k_mid],
            }
            profiles[step] = {
                "wind_profile": _horizontal_profile_3d(wind, area2d),
                "T_profile": _horizontal_profile_3d(T, area2d),
            }

    _extract_diag_and_optional(0, capture=True)

    print(f"Integrating {n_steps} steps...")
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    stable = True

    # Warmup compile
    state = model.step_with_physics(state, args.dt, held_suarez_forcing_spectral)
    jax.block_until_ready(state.vor_hat.data)

    for i in range(1, n_steps + 1):
        if i > 1:
            state = model.step_with_physics(state, args.dt, held_suarez_forcing_spectral)
        if i in snap_steps:
            _extract_diag_and_optional(i, capture=True)
        elif i % args.mean_every == 0 or i == n_steps:
            _extract_diag_and_optional(i, capture=False)

        if i % progress_every == 0:
            print(f"  progress: {i}/{n_steps}")
        if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
            stable = False
            print(f"  non-finite at step {i}")
            break

    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0
    print(f"Done in {wall:.1f}s")

    _save_snapshots(out, snapshots, args.dt, lon2d, lat2d)
    mass_rel_end, energy_rel_end = _save_conservation(out, times_days, mass_series, energy_series)
    _save_slab(out, times_days, mean_ps_series, mean_T_series, max_wind_series)
    _save_profiles(out, sorted(profiles.keys()), args.dt, np.asarray(sigma.sigma_full, dtype=np.float64), profiles)

    with open(out / "results.txt", "w") as f:
        f.write(f"truncation: T{args.truncation}\n")
        f.write(f"levels: {args.levels}\n")
        f.write(f"dt_s: {args.dt}\n")
        f.write(f"days: {args.days}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"mass_drift_rel: {mass_rel_end:.12e}\n")
        f.write(f"energy_drift_rel: {energy_rel_end:.12e}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
