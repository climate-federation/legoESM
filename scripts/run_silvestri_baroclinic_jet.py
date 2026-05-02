#!/usr/bin/env python3
"""Silvestri et al. (2024) baroclinic jet: WENO vs. centered comparison.

Configures a baroclinic jet on a spherical sector (60S-40S, 20deg wide)
matching the setup of Silvestri et al. (2024, JAMES, Section 5).  Runs
multiple momentum advection schemes and saves snapshots + diagnostics
for comparison.

Usage
-----
Quick smoke test (CPU, ~2 min):
    python scripts/run_silvestri_baroclinic_jet.py \\
        --resolution 20x20 --days 10 --schemes centered

Development run (CPU, ~20 min):
    python scripts/run_silvestri_baroclinic_jet.py \\
        --resolution 40x40 --days 100 --schemes centered,weno5

GPU production run:
    JAX_ENABLE_X64=1 python scripts/run_silvestri_baroclinic_jet.py \\
        --resolution 160x160 --days 1000 --schemes centered,leith,weno5

References
----------
Silvestri et al. (2024), "A new WENO-based momentum advection scheme
for simulations of ocean mesoscale turbulence", JAMES.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.grids.latlon import create_regional_latlon_grid  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    curl_vertex_cgrid,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.eos import LinearEOSConfig  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    rest_state_latlon_cgrid_ocean,
)
from legoesm.ocean.sponge import (  # noqa: E402
    SpongeForcing,
    compute_sponge_gamma_latlon,
)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402

g = constants.g
Omega = constants.Omega


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class SilvestriConfig:
    """Physical and numerical parameters for the baroclinic jet."""

    # --- Domain ---
    lat_south: float = -60.0
    lat_north: float = -40.0
    lon_west: float = 0.0
    lon_east: float = 20.0
    H_max: float = 1000.0
    nlev: int = 50               # 50 levels x 20 m = 1000 m

    # --- Stratification ---
    N2: float = 4.0e-6           # uniform Brunt-Vaisala [s^-2]

    # --- Meridional front ---
    front_lat_center: float = -50.0  # phi_0 [degrees]
    front_width_deg: float = 2.0     # Delta_phi [degrees]
    delta_b: float = 5.0e-3          # buoyancy jump [m/s^2]

    # --- Linear EOS ---
    alpha_T: float = 2.0e-4
    rho_0: float = 1025.0
    T_ref: float = 10.0
    S_uniform: float = 35.0

    # --- Mixing ---
    A_v: float = 1.0e-4          # vertical viscosity [m^2/s]
    K_v: float = 1.0e-5          # vertical diffusivity [m^2/s]

    # --- Sponge ---
    sponge_width_deg: float = 3.0
    sponge_timescale_days: float = 50.0

    # --- Wall taper ---
    wall_taper_deg: float = 3.0

    # --- Perturbation ---
    noise_amplitude: float = 1.0e-3  # T noise [K]
    noise_seed: int = 42

    # --- Barotropic ---
    n_barotropic_substeps: int = 30
    barotropic_diffusion_alpha: float = 0.05
    barotropic_div_damp: float = 0.05
    bebt: float = 0.2

    # --- Bottom drag ---
    bottom_drag_r: float = 1.1e-3  # [m/s]

    @property
    def dTdz(self) -> float:
        """Vertical T gradient from uniform N^2 [K/m]."""
        return self.N2 / (g * self.alpha_T)

    @property
    def delta_T_front(self) -> float:
        """Temperature jump across the front [K]."""
        return self.delta_b / (g * self.alpha_T)


# Scheme-specific config overrides
SCHEME_CONFIGS: dict[str, dict] = {
    "centered": {
        "momentum_advection": "vector_invariant",
        "tracer_advection": "tvd",
        "C_smag": 0.25,
        "C_leith": 0.0,
        "A_h": 0.0,
    },
    "leith": {
        "momentum_advection": "vector_invariant",
        "tracer_advection": "tvd",
        "C_smag": 0.0,
        "C_leith": 1.5,
        "A_h": 0.0,
    },
    "weno5": {
        "momentum_advection": "weno5",
        "tracer_advection": "weno7",
        "C_smag": 0.0,
        "C_leith": 0.0,
        "A_h": 0.0,
    },
    "weno5_leith": {
        "momentum_advection": "weno5",
        "tracer_advection": "weno7",
        "C_smag": 0.0,
        "C_leith": 1.0,
        "A_h": 0.0,
    },
}


# =====================================================================
# Grid and coordinate helpers
# =====================================================================

def _meridional_taper(lat_deg: np.ndarray, cfg: SilvestriConfig) -> np.ndarray:
    """Half-cosine taper: 0 at N/S walls, 1 in interior."""
    w = cfg.wall_taper_deg
    d_south = lat_deg - cfg.lat_south
    d_north = cfg.lat_north - lat_deg
    taper_s = np.where(
        d_south < w,
        0.5 * (1.0 - np.cos(np.pi * np.clip(d_south / w, 0, 1))),
        1.0)
    taper_n = np.where(
        d_north < w,
        0.5 * (1.0 - np.cos(np.pi * np.clip(d_north / w, 0, 1))),
        1.0)
    return taper_s * taper_n


def create_grid_and_zcoord(n_lat: int, n_lon: int, cfg: SilvestriConfig):
    """Create regional grid (periodic-x) and uniform vertical coordinate."""
    grid, wall_mask = create_regional_latlon_grid(
        n_lat, n_lon,
        lat_south=cfg.lat_south, lat_north=cfg.lat_north,
        lon_west=cfg.lon_west, lon_east=cfg.lon_east,
        periodic_x=True,
    )
    # Uniform dz = H_max / nlev
    dz_uniform = cfg.H_max / cfg.nlev
    z_coord = create_ocean_z_star(
        n_levels=cfg.nlev, H_max=cfg.H_max,
        dz_surface=dz_uniform, dz_deep=dz_uniform,
    )
    return grid, wall_mask, z_coord


# =====================================================================
# Initial conditions
# =====================================================================

def create_initial_state(grid, z_coord, wall_mask, cfg: SilvestriConfig):
    """Build IC: uniform N^2 + tanh front + thermal wind + noise.

    Steps:
    1. Rest state with flat bottom at H_max
    2. T(lat, z) = T_ref + dTdz*z + delta_T/2 * tanh((lat-lat0)/w) * taper
    3. Thermal-wind-balanced u from analytical dT/dy
    4. Remove depth-mean u (purely baroclinic, eta=0)
    5. Small random perturbation in T
    """
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_surface=cfg.T_ref, T_deep=cfg.T_ref,
        S_uniform=cfg.S_uniform,
        H_max=cfg.H_max,
        land_mask_override=jnp.array(wall_mask),
    )

    # --- Temperature ---
    lat_rad = np.asarray(grid.lat)           # (n_lat,)
    lat_deg = np.degrees(lat_rad)
    z_full = np.asarray(z_coord.z_full_ref)  # (nlev,) negative values
    mask = np.asarray(state.land_mask.data)   # (n_lat, n_lon)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels

    # Background: T_ref + dTdz * z
    T_bg = cfg.T_ref + cfg.dTdz * z_full     # (nlev,)

    # Tanh front * wall taper (depth-independent for uniform N^2)
    lat0_rad = np.radians(cfg.front_lat_center)
    w_rad = np.radians(cfg.front_width_deg)
    taper = _meridional_taper(lat_deg, cfg)   # (n_lat,)
    front_1d = (cfg.delta_T_front / 2.0
                * np.tanh((lat_rad - lat0_rad) / w_rad)
                * taper)                       # (n_lat,)

    T_data = np.zeros((n_lat, n_lon, nlev), dtype=np.float64)
    for k in range(nlev):
        T_data[:, :, k] = (T_bg[k] + front_1d[:, np.newaxis]) * mask

    # White noise perturbation
    rng = np.random.default_rng(seed=cfg.noise_seed)
    T_data += (cfg.noise_amplitude
               * rng.standard_normal(T_data.shape)
               * mask[:, :, np.newaxis])

    state = state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units))

    # --- Thermal wind velocity ---
    state = _set_thermal_wind(state, grid, z_coord, cfg)

    return state


def _set_thermal_wind(state, grid, z_coord, cfg: SilvestriConfig):
    """Compute thermal-wind-balanced u from analytical dT/dy.

    f0 * du/dz = -g * alpha_T * dT/dy

    Uses analytical dT/dy = (delta_T / 2*w*R) * sech^2 * taper
    (avoids finite-difference noise). Integrates bottom-up with u=0
    at bottom. Removes depth-mean for purely baroclinic IC.
    """
    lat_rad = np.asarray(grid.lat)      # (n_lat,)
    lat_deg = np.degrees(lat_rad)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    dz = np.asarray(z_coord.dz_ref)     # (nlev,)
    z_full = np.asarray(z_coord.z_full_ref)
    R = float(grid.radius)

    lat0_rad = np.radians(cfg.front_lat_center)
    w_rad = np.radians(cfg.front_width_deg)
    f0 = 2.0 * Omega * np.sin(lat0_rad)  # negative in S. hemisphere

    # Analytical dT/dy at cell-center latitudes
    taper = _meridional_taper(lat_deg, cfg)  # (n_lat,)
    sech2 = 1.0 / np.cosh((lat_rad - lat0_rad) / w_rad) ** 2
    dTdy = cfg.delta_T_front / (2.0 * w_rad * R) * sech2 * taper  # (n_lat,)

    # Thermal wind: du/dz = -(g * alpha_T / f0) * dT/dy
    # dT/dy is depth-independent => du/dz is depth-independent
    # => u(z) = coeff * dTdy * (z - z_bot)
    coeff = -g * cfg.alpha_T / f0
    z_bot = z_full[-1]
    U_2d = coeff * dTdy[:, np.newaxis] * (z_full[np.newaxis, :] - z_bot)
    # U_2d: (n_lat, nlev)

    # Remove depth mean for purely baroclinic IC
    H_col = np.sum(dz)
    U_bar = np.sum(U_2d * dz[np.newaxis, :], axis=1) / H_col  # (n_lat,)
    U_2d = U_2d - U_bar[:, np.newaxis]

    # Broadcast to u-face array (n_lat, n_lon+1, nlev)
    u_data = np.zeros((n_lat, n_lon + 1, nlev), dtype=np.float64)
    for j in range(n_lon + 1):
        u_data[:, j, :] = U_2d
    u_mask = np.asarray(state.u_mask.data)
    u_data *= u_mask[:, :, np.newaxis]

    # eta = 0 (depth-mean removed → no geostrophic SSH needed)
    return state._replace(
        u=Field(jnp.array(u_data), name="u",
                dims=state.u.dims, units=state.u.units))


# =====================================================================
# Sponge forcing
# =====================================================================

def create_sponge(grid, z_coord, state, cfg: SilvestriConfig):
    """Create sponge restoring at both N/S boundaries."""
    gamma = compute_sponge_gamma_latlon(
        grid, cfg.lat_south, cfg.lat_north,
        width_deg=cfg.sponge_width_deg,
        timescale_days=cfg.sponge_timescale_days,
    )
    return SpongeForcing(
        gamma=jnp.array(gamma),
        T_ref=state.T.data,
        S_ref=state.S.data,
        u_ref=state.u.data,
        v_ref=state.v.data,
    )


# =====================================================================
# Model configuration
# =====================================================================

def make_model_config(cfg: SilvestriConfig, scheme: str):
    """Build LatLonCGridOceanConfig for a given scheme."""
    sc = SCHEME_CONFIGS[scheme]
    return LatLonCGridOceanConfig(
        g=g,
        rho_0=cfg.rho_0,
        A_h=sc.get("A_h", 0.0),
        B_h=0.0,
        C_smag=sc["C_smag"],
        C_leith=sc["C_leith"],
        K_h=0.0,
        K_bih=0.0,
        A_v=cfg.A_v,
        K_v=cfg.K_v,
        bottom_drag_r=cfg.bottom_drag_r,
        n_barotropic_substeps=cfg.n_barotropic_substeps,
        barotropic_diffusion_alpha=cfg.barotropic_diffusion_alpha,
        barotropic_div_damp=cfg.barotropic_div_damp,
        bebt=cfg.bebt,
        eos="linear",
        eos_linear=LinearEOSConfig(
            alpha_T=cfg.alpha_T, rho_ref=cfg.rho_0,
            T_ref=cfg.T_ref, S_ref=cfg.S_uniform),
        tracer_advection=sc["tracer_advection"],
        momentum_advection=sc["momentum_advection"],
    )


# =====================================================================
# Diagnostics
# =====================================================================

def compute_diagnostics(state, grid, z_coord):
    """Compute scalar diagnostics from the current state."""
    u = np.asarray(state.u.data)     # (n_lat, n_lon+1, nlev)
    v = np.asarray(state.v.data)     # (n_lat+1, n_lon, nlev)
    T = np.asarray(state.T.data)     # (n_lat, n_lon, nlev)
    eta = np.asarray(state.eta.data)
    mask = np.asarray(state.land_mask.data)
    dz = np.asarray(z_coord.dz_ref)
    area = np.asarray(grid.area)     # (n_lat, n_lon)

    # Speed at cell centers (approximate)
    u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    speed = np.sqrt(u_cc**2 + v_cc**2) * mask[:, :, np.newaxis]

    # Volume-weighted KE = 0.5 * integral(u^2 + v^2, dV)
    vol = area[:, :, np.newaxis] * dz[np.newaxis, np.newaxis, :] * mask[:, :, np.newaxis]
    KE_field = 0.5 * (u_cc**2 + v_cc**2)
    total_KE = float(np.sum(KE_field * vol))

    # Eddy KE: deviation from zonal mean
    u_bar = np.mean(u_cc * mask[:, :, np.newaxis], axis=1, keepdims=True)
    v_bar = np.mean(v_cc * mask[:, :, np.newaxis], axis=1, keepdims=True)
    n_ocean = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0)
    u_bar = u_bar / n_ocean[:, :, np.newaxis] * mask.shape[1]
    v_bar = v_bar / n_ocean[:, :, np.newaxis] * mask.shape[1]
    # Simpler: just take mean along axis=1 (all ocean for channel)
    u_bar = np.mean(u_cc, axis=1, keepdims=True)
    v_bar = np.mean(v_cc, axis=1, keepdims=True)
    u_prime = u_cc - u_bar
    v_prime = v_cc - v_bar
    EKE_field = 0.5 * (u_prime**2 + v_prime**2)
    eddy_KE = float(np.sum(EKE_field * vol))

    return {
        "total_KE": total_KE,
        "eddy_KE": eddy_KE,
        "max_speed": float(np.max(speed)),
        "mean_T": float(np.sum(T * vol) / np.maximum(np.sum(vol), 1e-30)),
        "max_abs_eta": float(np.max(np.abs(eta))),
    }


def extract_snapshot(state, grid):
    """Extract key 2D/3D fields for saving."""
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    # Surface vorticity via curl_vertex_cgrid (2D, level 0)
    zeta_3d = np.asarray(
        curl_vertex_cgrid(state.u.data, state.v.data, grid))
    return {
        "T": np.asarray(state.T.data),
        "u": u,
        "v": v,
        "eta": np.asarray(state.eta.data),
        "zeta_surface": zeta_3d[:, :, 0],
    }


# =====================================================================
# Time loop
# =====================================================================

def run_one_scheme(
    scheme: str,
    n_lat: int,
    n_lon: int,
    days: float,
    dt: float,
    cfg: SilvestriConfig,
    output_dir: Path,
    n_snaps: int = 20,
) -> dict:
    """Run a single scheme and save results."""
    print(f"\n{'='*60}")
    print(f"Scheme: {scheme}")
    print(f"  Resolution: {n_lat}x{n_lon}, {cfg.nlev} levels")
    print(f"  Duration: {days} days, dt={dt}s")
    sc = SCHEME_CONFIGS[scheme]
    print(f"  momentum_advection={sc['momentum_advection']}, "
          f"tracer_advection={sc['tracer_advection']}")
    print(f"  C_smag={sc['C_smag']}, C_leith={sc['C_leith']}")
    print(f"{'='*60}")

    # Setup
    grid, wall_mask, z_coord = create_grid_and_zcoord(n_lat, n_lon, cfg)
    state0 = create_initial_state(grid, z_coord, wall_mask, cfg)
    sponge = create_sponge(grid, z_coord, state0, cfg)
    model_cfg = make_model_config(cfg, scheme)
    model = LatLonCGridOceanModel(grid, z_coord, model_cfg)

    # CFL check
    try:
        model.check_barotropic_cfl(dt)
    except Exception as e:
        print(f"  CFL warning: {e}")

    # Print IC diagnostics
    diag0 = compute_diagnostics(state0, grid, z_coord)
    print(f"  IC: total_KE={diag0['total_KE']:.4e}, "
          f"max_speed={diag0['max_speed']:.4f} m/s")

    # Time loop
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 200)
    snap_every = max(1, n_steps // n_snaps)

    diagnostics = {k: [] for k in
                   ["time_days", "total_KE", "eddy_KE",
                    "max_speed", "mean_T", "max_abs_eta"]}
    snapshots = {}

    state = state0
    t0 = time.time()
    ok = True

    for i in range(n_steps):
        state = model.step(state, dt, sponge=sponge)
        step = i + 1

        if step % diag_every == 0 or step == n_steps:
            day = step * dt / 86400.0
            d = compute_diagnostics(state, grid, z_coord)
            for k, v in d.items():
                diagnostics[k].append(v)
            diagnostics["time_days"].append(day)

            if step % (diag_every * 10) == 0 or step == n_steps:
                elapsed = time.time() - t0
                print(f"  Day {day:7.1f} | KE={d['total_KE']:.4e} "
                      f"EKE={d['eddy_KE']:.4e} "
                      f"max_spd={d['max_speed']:.4f} | "
                      f"{elapsed:.0f}s")

            if not np.isfinite(d["max_speed"]) or d["max_speed"] > 50.0:
                print(f"  *** BLOWUP at day {day:.1f} ***")
                ok = False
                break

        if step % snap_every == 0 or step == n_steps:
            snapshots[step] = extract_snapshot(state, grid)

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s "
          f"({'OK' if ok else 'BLOWUP'})")

    # Save diagnostics
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{scheme}_diagnostics.npz",
        **{k: np.array(v) for k, v in diagnostics.items()})

    # Save final snapshot
    if snapshots:
        last_key = max(snapshots.keys())
        np.savez_compressed(
            output_dir / f"{scheme}_final_snapshot.npz",
            **snapshots[last_key])

    # Save all snapshot times
    snap_times = {f"snap_{k}_day": k * dt / 86400.0 for k in snapshots}
    for k, snap in snapshots.items():
        for field_name, data in snap.items():
            snap_times[f"snap_{k}_{field_name}"] = data
    np.savez_compressed(
        output_dir / f"{scheme}_snapshots.npz", **snap_times)

    return {"scheme": scheme, "ok": ok, "diagnostics": diagnostics}


# =====================================================================
# Plotting (optional)
# =====================================================================

def plot_comparison(results: list[dict], output_dir: Path):
    """Plot KE time series and final surface vorticity for all schemes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total KE time series
    ax = axes[0]
    for r in results:
        d = r["diagnostics"]
        ax.semilogy(d["time_days"], d["total_KE"],
                     label=r["scheme"])
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Total KE [m^5/s^2]")
    ax.set_title("Total Kinetic Energy")
    ax.legend()

    # Eddy KE time series
    ax = axes[1]
    for r in results:
        d = r["diagnostics"]
        ax.semilogy(d["time_days"], d["eddy_KE"],
                     label=r["scheme"])
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Eddy KE [m^5/s^2]")
    ax.set_title("Eddy Kinetic Energy")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "ke_timeseries.png", dpi=150)
    plt.close()
    print(f"  Saved KE time series to {output_dir / 'ke_timeseries.png'}")

    # Surface vorticity from final snapshots — shared color bar
    n_schemes = len(results)
    if n_schemes == 0:
        return

    # First pass: find global vmax across all schemes
    all_zeta = {}
    global_vmax = 1e-8
    for r in results:
        snap_file = output_dir / f"{r['scheme']}_final_snapshot.npz"
        if snap_file.exists():
            data = np.load(snap_file)
            zeta = data["zeta_surface"]
            all_zeta[r["scheme"]] = zeta
            vmax = max(abs(float(np.nanpercentile(zeta, 2))),
                       abs(float(np.nanpercentile(zeta, 98))))
            global_vmax = max(global_vmax, vmax)

    # Second pass: plot with shared color scale
    fig, axes = plt.subplots(1, n_schemes, figsize=(5 * n_schemes, 4),
                             squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[0, idx]
        if r["scheme"] in all_zeta:
            im = ax.imshow(all_zeta[r["scheme"]], origin="lower",
                           cmap="RdBu_r", vmin=-global_vmax,
                           vmax=global_vmax, aspect="auto")
            ax.set_title(r["scheme"])
        else:
            ax.set_title(f"{r['scheme']} (no snapshot)")
    fig.colorbar(im, ax=axes[0, :].tolist(), label="Relative vorticity [1/s]",
                 shrink=0.8)
    plt.suptitle("Surface Relative Vorticity (final)")
    plt.tight_layout()
    plt.savefig(output_dir / "surface_vorticity.png", dpi=150)
    plt.close()
    print(f"  Saved surface vorticity to "
          f"{output_dir / 'surface_vorticity.png'}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Silvestri et al. (2024) baroclinic jet comparison")
    parser.add_argument(
        "--schemes", default="centered,weno5",
        help="Comma-separated scheme names: "
             + ", ".join(SCHEME_CONFIGS.keys()))
    parser.add_argument(
        "--resolution", default="20x20",
        help="n_lat x n_lon (e.g. 20x20, 80x80, 160x160)")
    parser.add_argument("--days", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--nlev", type=int, default=50)
    parser.add_argument("--n-snaps", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/ocean/silvestri_jet"))
    parser.add_argument(
        "--front-width", type=float, default=2.0,
        help="Front width Delta_phi [degrees]")
    parser.add_argument(
        "--sponge-tau", type=float, default=50.0,
        help="Sponge restoring timescale [days]")
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comparison plots")
    args = parser.parse_args()

    n_lat, n_lon = [int(x) for x in args.resolution.split("x")]
    schemes = [s.strip() for s in args.schemes.split(",")]
    for s in schemes:
        if s not in SCHEME_CONFIGS:
            parser.error(f"Unknown scheme '{s}'. "
                         f"Available: {list(SCHEME_CONFIGS.keys())}")

    cfg = SilvestriConfig(
        nlev=args.nlev,
        front_width_deg=args.front_width,
        sponge_timescale_days=args.sponge_tau,
    )

    print(f"Silvestri Baroclinic Jet Comparison")
    print(f"  Schemes: {schemes}")
    print(f"  Grid: {n_lat}x{n_lon}x{cfg.nlev}")
    print(f"  Duration: {args.days} days, dt={args.dt}s")
    print(f"  Front: lat0={cfg.front_lat_center}, "
          f"width={cfg.front_width_deg} deg, "
          f"delta_T={cfg.delta_T_front:.2f} K")
    print(f"  dT/dz = {cfg.dTdz:.5f} K/m (N^2={cfg.N2:.1e})")
    print(f"  Output: {args.output_dir}")

    results = []
    for scheme in schemes:
        r = run_one_scheme(
            scheme, n_lat, n_lon,
            days=args.days, dt=args.dt, cfg=cfg,
            output_dir=args.output_dir, n_snaps=args.n_snaps)
        results.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Scheme':<15} {'Status':<8} {'Final KE':>12} {'Final EKE':>12} "
          f"{'Max Speed':>10}")
    print(f"{'-'*60}")
    for r in results:
        d = r["diagnostics"]
        ke = d["total_KE"][-1] if d["total_KE"] else 0
        eke = d["eddy_KE"][-1] if d["eddy_KE"] else 0
        spd = d["max_speed"][-1] if d["max_speed"] else 0
        status = "OK" if r["ok"] else "BLOWUP"
        print(f"{r['scheme']:<15} {status:<8} {ke:>12.4e} {eke:>12.4e} "
              f"{spd:>10.4f}")
    print(f"{'='*60}")

    if args.plot:
        plot_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
