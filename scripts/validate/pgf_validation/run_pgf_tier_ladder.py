#!/usr/bin/env python
"""PGF tiered test ladder for ocean models.

Runs a systematic progression of PGF stress tests from cheapest/most
diagnostic (Tier 1: idealized rest-state) to most realistic (Tier 5:
JRA-forced realistic bathymetry).  Each tier adds one source of
complexity, so failures can be attributed cleanly.

See docs/ocean_experiments/pgf_test_plan.md for the full rationale.

Tiers
-----
  1  τ=0, idealized bathymetry, uniform stratification
  2  τ=0, idealized bathymetry, realistic stratification (WOA-like)
  3  τ=0, realistic bathymetry (ETOPO), realistic stratification
  4  Wind-driven, idealized bathymetry  (future)
  5  Wind-driven, realistic bathymetry  (future)

Usage
-----
    JAX_ENABLE_X64=1 python scripts/validate/pgf_validation/run_pgf_tier_ladder.py
    JAX_ENABLE_X64=1 python scripts/validate/pgf_validation/run_pgf_tier_ladder.py --tier 1
    JAX_ENABLE_X64=1 python scripts/validate/pgf_validation/run_pgf_tier_ladder.py --tier 2 --pgf-scheme smc03
    JAX_ENABLE_X64=1 python scripts/validate/pgf_validation/run_pgf_tier_ladder.py --tier 3 --days 90
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
    compute_layer_thickness,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH, rho_0
from legoesm import constants


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TierConfig:
    """Configuration for a single PGF tier run."""
    tier: int = 1
    # Grid
    n_lat: int = 36
    n_lon: int = 72
    n_levels: int = 20
    H_max: float = 4000.0
    dz_surface: float = 10.0
    dz_deep: float = 500.0
    # PGF scheme
    pgf_scheme: str = "adcroft"
    coord: str = "partial"  # "zstar" or "partial"
    # Bathymetry
    bathymetry_type: str = "seamount"  # "flat", "step", "seamount", "etopo"
    seamount_height_m: float = 3800.0
    seamount_sigma_deg: float = 10.0
    smoothing_passes: int = 5
    H_min: float = 50.0  # ETOPO: minimum ocean depth before masking as land
    # Stratification
    stratification: str = "uniform"  # "uniform", "exponential", "woa"
    T_water_init_C: float = 20.0
    T_deep: float = 2.0
    S_uniform: float = 35.0
    # Model config
    barotropic_solver: str = "implicit_cn"
    momentum_advection: str = "vector_invariant"
    bottom_drag_r: float = 1.0e-3
    A_h: float = 1.0e4
    # Integration
    days: float = 30.0
    dt: float = 600.0
    record_every_days: float = 1.0
    # Output
    output_dir: str = ""


# ---------------------------------------------------------------------------
# WOA-like stratification profiles (equatorial Pacific representative)
# ---------------------------------------------------------------------------

def woa_equatorial_pacific_profiles(z_levels: jnp.ndarray):
    """Representative T(z), S(z) from WOA climatology, equatorial Pacific.

    These are simplified analytic fits to the World Ocean Atlas annual-mean
    profiles at (0°N, 140°W), capturing the key features: warm mixed layer,
    sharp thermocline, cold abyss with realistic density structure.

    Parameters
    ----------
    z_levels : array, shape (nlev,)
        Cell-centre depths [m], negative (below sea level).

    Returns
    -------
    T_profile, S_profile : arrays, shape (nlev,)
        Temperature [°C] and salinity [PSU] profiles.
    """
    depth = jnp.abs(z_levels)  # positive downward [m]

    # Temperature: warm mixed layer (28°C), thermocline drop, cold abyss (1.5°C)
    # Two-exponential fit approximating WOA equatorial Pacific
    T_profile = 1.5 + 16.0 * jnp.exp(-depth / 300.0) + 10.5 * jnp.exp(-depth / 50.0)

    # Salinity: surface fresh (~34.5), subsurface max (~35.2 at ~150m), deep (~34.7)
    # Gaussian bump for the salinity maximum
    S_profile = (
        34.7
        + 0.5 * jnp.exp(-((depth - 150.0) ** 2) / (100.0**2))
        - 0.2 * jnp.exp(-depth / 30.0)
    )

    return T_profile, S_profile


# ---------------------------------------------------------------------------
# Bathymetry builders
# ---------------------------------------------------------------------------

def build_bathymetry(grid, config: TierConfig):
    """Build bathymetry array and land mask for the given config.

    Returns
    -------
    H_bathy : array (n_lat, n_lon), positive [m]
    land_mask : array (n_lat, n_lon), 1=ocean 0=land
    """
    lat_deg = np.asarray(grid.lat2d) * 180.0 / np.pi
    lon_deg = np.asarray(grid.lon2d) * 180.0 / np.pi

    if config.bathymetry_type == "flat":
        H_bathy = np.full((grid.n_lat, grid.n_lon), config.H_max)
        land_mask = np.where(np.abs(lat_deg) < 80.0, 1.0, 0.0)

    elif config.bathymetry_type == "step":
        # Northern half shallow, southern half deep
        H_deep = config.H_max
        H_shallow = config.H_max * 0.2  # 20% of max depth
        H_bathy = np.where(lat_deg > 0, H_shallow, H_deep)
        land_mask = np.where(np.abs(lat_deg) < 80.0, 1.0, 0.0)

    elif config.bathymetry_type == "seamount":
        # Beckmann-Haidvogel Gaussian seamount
        dlon = lon_deg - 180.0
        dlon = np.where(dlon > 180.0, dlon - 360.0, dlon)
        dlon = np.where(dlon < -180.0, dlon + 360.0, dlon)
        dlat = lat_deg - 0.0
        r2 = dlat**2 + dlon**2
        seamount = config.seamount_height_m * np.exp(
            -r2 / (2 * config.seamount_sigma_deg**2)
        )
        H_bathy = config.H_max - seamount
        H_bathy = np.maximum(H_bathy, 10.0)  # floor
        land_mask = np.where(np.abs(lat_deg) < 80.0, 1.0, 0.0)

    elif config.bathymetry_type == "ridge":
        # Mid-ocean ridge: sinusoidal ridge across the equator
        ridge_height = config.H_max * 0.6
        ridge_width_deg = 5.0
        ridge = ridge_height * np.exp(-(lat_deg**2) / (2 * ridge_width_deg**2))
        H_bathy = config.H_max - ridge
        H_bathy = np.maximum(H_bathy, 100.0)
        land_mask = np.where(np.abs(lat_deg) < 80.0, 1.0, 0.0)

    elif config.bathymetry_type == "etopo":
        H_bathy, land_mask = _load_etopo_bathymetry(grid, config)

    else:
        raise ValueError(f"Unknown bathymetry_type: {config.bathymetry_type!r}")

    # Apply smoothing
    if config.smoothing_passes > 0 and config.bathymetry_type != "flat":
        from legoesm.ocean.bathymetry import _laplacian_smooth_2d
        H_bathy = _laplacian_smooth_2d(H_bathy, config.smoothing_passes,
                                        is_cubed=False)

    return jnp.asarray(H_bathy), jnp.asarray(land_mask)


def _load_etopo_bathymetry(grid, config: TierConfig):
    """Load and regrid ETOPO bathymetry."""
    import subprocess
    data_dir = Path("data/bathymetry")
    etopo_file = data_dir / "etopo_1deg.nc"
    etopo_url = (
        "https://upwell.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
        "altitude%5B(-90):60:(90)%5D%5B(-180):60:(180)%5D"
    )
    if not etopo_file.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading ETOPO -> {etopo_file}")
        subprocess.run(
            ["curl", "-fsS", "-o", str(etopo_file), etopo_url], check=True,
        )

    import xarray as xr
    ds = xr.open_dataset(etopo_file)
    alt = ds["altitude"].values  # (lat, lon), positive = above sea level
    etopo_lat = ds["latitude"].values
    etopo_lon = ds["longitude"].values

    from scipy.interpolate import RegularGridInterpolator
    # Handle endpoint deduplication for periodic longitude
    if np.isclose(etopo_lon[0] % 360, etopo_lon[-1] % 360):
        etopo_lon = etopo_lon[:-1]
        alt = alt[:, :-1]

    interp = RegularGridInterpolator(
        (etopo_lat, etopo_lon), alt, method="linear",
        bounds_error=False, fill_value=None,
    )
    lat_deg = np.asarray(grid.lat2d) * 180.0 / np.pi
    lon_deg = np.asarray(grid.lon2d) * 180.0 / np.pi
    # Wrap to ETOPO longitude range
    lon_query = lon_deg % 360.0
    if etopo_lon.min() < 0:
        lon_query = np.where(lon_query > 180.0, lon_query - 360.0, lon_query)

    pts = np.stack([lat_deg.ravel(), lon_query.ravel()], axis=-1)
    altitude = interp(pts).reshape(grid.n_lat, grid.n_lon)

    # Convert: altitude > 0 → land; altitude < 0 → ocean depth
    H_bathy = np.maximum(-altitude, 0.0)
    # Enforce H_min
    land_mask = np.where(H_bathy >= config.H_min, 1.0, 0.0)
    H_bathy = np.where(land_mask > 0, np.maximum(H_bathy, config.H_min), 0.0)
    # Cap at H_max
    H_bathy = np.minimum(H_bathy, config.H_max)

    return H_bathy, land_mask


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------

def build_initial_state(grid, z_coord, config: TierConfig):
    """Build initial state with the specified stratification and bathymetry.

    Returns
    -------
    state : LatLonCGridOceanState
    coord : OceanZStarCoordinate or OceanPartialCellCoordinate
    """
    H_bathy, land_mask = build_bathymetry(grid, config)

    # Build state with flat-bottom init, then override T/S and bathymetry
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=config.T_water_init_C,
        T_deep=config.T_deep,
        S_uniform=config.S_uniform,
        H_max=config.H_max,
        H_bathy_override=H_bathy,
        land_mask_override=land_mask,
    )

    # Build coordinate (z-star or partial cells)
    if config.coord == "partial":
        coord = create_partial_cell_coordinate(z_coord, H_bathy)
    else:
        coord = z_coord

    # Override T/S based on stratification type
    if config.stratification == "uniform":
        # Uniform T and S — no horizontal or vertical density gradients
        T_3d = jnp.full_like(state.T.data, config.T_deep)
        S_3d = jnp.full_like(state.S.data, config.S_uniform)
        state = state._replace(
            T=state.T.replace(data=T_3d),
            S=state.S.replace(data=S_3d),
        )

    elif config.stratification == "exponential":
        # Exponential T(z), uniform S — the default from rest_state init
        # If using partial cells, use centroid-aware initialization
        if config.coord == "partial":
            centroid = compute_centroid_depth(
                jnp.zeros_like(H_bathy), H_bathy, coord,
            )
            T_per_cell = config.T_deep + (config.T_water_init_C - config.T_deep) * jnp.exp(
                -centroid / _SCALE_DEPTH
            )
            T_per_cell = jnp.where(coord.is_active, T_per_cell, config.T_deep)
            state = state._replace(T=state.T.replace(data=T_per_cell))
        # else: the rest_state_latlon_cgrid_ocean already set exponential T(z)

    elif config.stratification == "woa":
        # WOA-like realistic profiles
        T_profile, S_profile = woa_equatorial_pacific_profiles(z_coord.z_full_ref)
        if config.coord == "partial":
            # Centroid-aware: evaluate profiles at actual cell-centre depths
            centroid = compute_centroid_depth(
                jnp.zeros_like(H_bathy), H_bathy, coord,
            )
            # Interpolate profiles to centroid depths
            ref_depths = jnp.abs(z_coord.z_full_ref)
            T_3d = jnp.interp(centroid, ref_depths, T_profile)
            S_3d = jnp.interp(centroid, ref_depths, S_profile)
            T_3d = jnp.where(coord.is_active, T_3d, T_profile[-1])
            S_3d = jnp.where(coord.is_active, S_3d, S_profile[-1])
        else:
            n_lat, n_lon = grid.n_lat, grid.n_lon
            T_3d = jnp.broadcast_to(
                T_profile[jnp.newaxis, jnp.newaxis, :],
                (n_lat, n_lon, z_coord.n_levels),
            )
            S_3d = jnp.broadcast_to(
                S_profile[jnp.newaxis, jnp.newaxis, :],
                (n_lat, n_lon, z_coord.n_levels),
            )
        state = state._replace(
            T=state.T.replace(data=T_3d),
            S=state.S.replace(data=S_3d),
        )

    else:
        raise ValueError(f"Unknown stratification: {config.stratification!r}")

    # Mask land cells in T/S
    mask_3d = state.land_mask.data[:, :, jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=state.T.data * mask_3d),
        S=state.S.replace(data=state.S.data * mask_3d),
    )

    return state, coord


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class PGFDiagnostics:
    """Collected diagnostics from a PGF tier run."""
    # Time series (one per record step)
    time_days: list = field(default_factory=list)
    u_max: list = field(default_factory=list)
    v_max: list = field(default_factory=list)
    speed_max: list = field(default_factory=list)
    eta_max: list = field(default_factory=list)
    mean_T: list = field(default_factory=list)
    mean_S: list = field(default_factory=list)
    # KE diagnostics
    KE_total: list = field(default_factory=list)
    # Equatorial diagnostics (|lat| < 5°)
    KE_equatorial: list = field(default_factory=list)
    speed_max_equatorial: list = field(default_factory=list)
    # Depth-binned KE (surface, mid, deep)
    KE_surface: list = field(default_factory=list)  # top 200m
    KE_mid: list = field(default_factory=list)       # 200-1000m
    KE_deep: list = field(default_factory=list)      # below 1000m

    def is_stable(self) -> bool:
        """Check if the run completed without NaN."""
        return len(self.speed_max) > 0 and all(
            np.isfinite(s) for s in self.speed_max
        )

    def max_speed(self) -> float:
        """Maximum speed reached during the run [m/s]."""
        if not self.speed_max:
            return float("nan")
        return max(self.speed_max)

    def ke_trend(self) -> float:
        """Linear trend in log(KE) [1/day]. Positive = growing."""
        if len(self.KE_total) < 3:
            return float("nan")
        ke = np.array(self.KE_total)
        t = np.array(self.time_days)
        # Filter out zeros/nans for log
        valid = (ke > 0) & np.isfinite(ke)
        if valid.sum() < 3:
            return float("nan")
        coeffs = np.polyfit(t[valid], np.log(ke[valid]), 1)
        return float(coeffs[0])

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


def collect_diagnostics(
    state,
    grid,
    z_coord,
    coord,
    day: float,
    diag: PGFDiagnostics,
):
    """Collect PGF-relevant diagnostics from the current state."""
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    eta = np.asarray(state.eta.data)
    mask = np.asarray(state.land_mask.data)

    # Basic diagnostics
    diag.time_days.append(day)
    diag.u_max.append(float(np.nanmax(np.abs(u))))
    diag.v_max.append(float(np.nanmax(np.abs(v))))

    # Speed at cell centres (average u from faces)
    # u is (n_lat, n_lon+1, nlev), v is (n_lat+1, n_lon, nlev)
    u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_c = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    speed = np.sqrt(u_c**2 + v_c**2)
    speed_ocean = np.where(mask[:, :, np.newaxis] > 0, speed, 0.0)
    diag.speed_max.append(float(np.nanmax(speed_ocean)))

    diag.eta_max.append(float(np.nanmax(np.abs(eta))))

    ocean_cells = mask.sum()
    if ocean_cells > 0:
        diag.mean_T.append(float((T * mask[:, :, np.newaxis]).sum() /
                                  (ocean_cells * T.shape[-1])))
        diag.mean_S.append(float((S * mask[:, :, np.newaxis]).sum() /
                                  (ocean_cells * S.shape[-1])))
    else:
        diag.mean_T.append(float("nan"))
        diag.mean_S.append(float("nan"))

    # KE = 0.5 * rho_0 * (u^2 + v^2) * h * area, summed
    # Simplified: compute volume-mean KE
    h_k = np.asarray(compute_layer_thickness(
        jnp.asarray(eta), state.H_bathy.data, coord,
    ))
    ke_density = 0.5 * (u_c**2 + v_c**2)  # m²/s²
    ke_weighted = ke_density * h_k * mask[:, :, np.newaxis]
    diag.KE_total.append(float(ke_weighted.sum()))

    # Equatorial band (|lat| < 5°)
    lat_deg = np.abs(np.asarray(grid.lat2d)) * 180.0 / np.pi
    eq_mask = (lat_deg < 5.0) & (mask > 0)
    eq_mask_3d = eq_mask[:, :, np.newaxis]
    speed_eq = np.where(eq_mask_3d, speed, 0.0)
    diag.speed_max_equatorial.append(float(np.nanmax(speed_eq)))
    diag.KE_equatorial.append(float((ke_weighted * eq_mask_3d).sum()))

    # Depth-binned KE
    z_centres = np.abs(np.asarray(z_coord.z_full_ref))  # positive [m]
    surface_mask = z_centres < 200.0
    mid_mask = (z_centres >= 200.0) & (z_centres < 1000.0)
    deep_mask = z_centres >= 1000.0
    diag.KE_surface.append(float(ke_weighted[:, :, surface_mask].sum()))
    diag.KE_mid.append(float(ke_weighted[:, :, mid_mask].sum()))
    diag.KE_deep.append(float(ke_weighted[:, :, deep_mask].sum()))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diagnostics(diag: PGFDiagnostics, config: TierConfig, out_dir: Path):
    """Generate standard diagnostic plots."""
    t = np.array(diag.time_days)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"PGF Tier {config.tier} — {config.pgf_scheme} / {config.coord} / "
        f"{config.bathymetry_type} / {config.stratification}",
        fontsize=13,
    )

    # 1. Max speed vs time
    ax = axes[0, 0]
    ax.semilogy(t, np.array(diag.speed_max) * 1e3, "b-", label="global")
    ax.semilogy(t, np.array(diag.speed_max_equatorial) * 1e3, "r--", label="|lat|<5°")
    ax.set_ylabel("max |speed| [mm/s]")
    ax.set_xlabel("time [days]")
    ax.legend(fontsize=8)
    ax.set_title("Max speed")
    ax.grid(True, alpha=0.3)

    # 2. Total KE vs time
    ax = axes[0, 1]
    ke = np.array(diag.KE_total)
    if np.any(ke > 0):
        ax.semilogy(t, ke, "k-")
    ax.set_ylabel("total KE [m³·m²/s²]")
    ax.set_xlabel("time [days]")
    ax.set_title("Total KE")
    ax.grid(True, alpha=0.3)

    # 3. Depth-binned KE
    ax = axes[0, 2]
    ax.semilogy(t, np.array(diag.KE_surface), "r-", label="0-200m")
    ax.semilogy(t, np.array(diag.KE_mid), "g-", label="200-1000m")
    ax.semilogy(t, np.array(diag.KE_deep), "b-", label=">1000m")
    ax.set_ylabel("KE [m³·m²/s²]")
    ax.set_xlabel("time [days]")
    ax.legend(fontsize=8)
    ax.set_title("KE by depth")
    ax.grid(True, alpha=0.3)

    # 4. Max |eta|
    ax = axes[1, 0]
    ax.plot(t, np.array(diag.eta_max) * 100, "k-")
    ax.set_ylabel("max |η| [cm]")
    ax.set_xlabel("time [days]")
    ax.set_title("SSH drift")
    ax.grid(True, alpha=0.3)

    # 5. Mean T drift
    ax = axes[1, 1]
    mean_T = np.array(diag.mean_T)
    ax.plot(t, mean_T - mean_T[0], "r-")
    ax.set_ylabel("ΔT [°C]")
    ax.set_xlabel("time [days]")
    ax.set_title("Mean T drift")
    ax.grid(True, alpha=0.3)

    # 6. Max |u| and |v| separately
    ax = axes[1, 2]
    ax.semilogy(t, np.array(diag.u_max) * 1e3, "b-", label="|u|max")
    ax.semilogy(t, np.array(diag.v_max) * 1e3, "r-", label="|v|max")
    ax.set_ylabel("max velocity [mm/s]")
    ax.set_xlabel("time [days]")
    ax.legend(fontsize=8)
    ax.set_title("Component velocities")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "diagnostics.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def make_step_block(model, dt):
    """Build a jax.lax.scan block for N model steps."""
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def run_tier(config: TierConfig):
    """Run a single PGF tier test and return diagnostics."""
    print(f"\n{'='*70}")
    print(f"PGF Tier {config.tier}: pgf={config.pgf_scheme}, coord={config.coord}, "
          f"bathy={config.bathymetry_type}, strat={config.stratification}")
    print(f"Grid: {config.n_lat}x{config.n_lon}, {config.n_levels} levels, "
          f"H_max={config.H_max}m, dt={config.dt}s, {config.days} days")
    print(f"{'='*70}")

    # Build grid and vertical coordinate
    grid = create_latlon_grid(n_lat=config.n_lat, n_lon=config.n_lon)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels,
        H_max=config.H_max,
        dz_surface=config.dz_surface,
        dz_deep=config.dz_deep,
    )

    # Build initial state
    print("Building initial state...")
    state, coord = build_initial_state(grid, z_coord, config)

    # Build model
    ocean_config = LatLonCGridOceanConfig(
        pgf_scheme=config.pgf_scheme,
        barotropic_solver=config.barotropic_solver,
        momentum_advection=config.momentum_advection,
        bottom_drag_r=config.bottom_drag_r,
        A_h=config.A_h,
    )
    model = LatLonCGridOceanModel(grid, coord, config=ocean_config)
    step_block = make_step_block(model, config.dt)

    # Integration loop
    steps_per_day = int(86400.0 / config.dt)
    record_interval = max(1, int(config.record_every_days))
    total_days = int(config.days)

    diag = PGFDiagnostics()

    # Record initial state
    collect_diagnostics(state, grid, z_coord, coord, 0.0, diag)
    print(f"  Day {0:6.1f}: speed_max={diag.speed_max[-1]*1e3:8.2f} mm/s, "
          f"KE={diag.KE_total[-1]:.2e}")

    t0 = time.time()
    for day in range(1, total_days + 1):
        try:
            state = step_block(state, n_inner=steps_per_day)
        except Exception as e:
            print(f"  Day {day}: CRASHED — {e}")
            break

        # Check for NaN
        u_data = np.asarray(state.u.data)
        if not np.all(np.isfinite(u_data)):
            print(f"  Day {day}: NaN detected — stopping")
            diag.speed_max.append(float("nan"))
            diag.time_days.append(float(day))
            break

        if day % record_interval == 0 or day == total_days:
            collect_diagnostics(state, grid, z_coord, coord, float(day), diag)
            elapsed = time.time() - t0
            print(f"  Day {day:6.1f}: speed_max={diag.speed_max[-1]*1e3:8.2f} mm/s, "
                  f"KE={diag.KE_total[-1]:.2e}, "
                  f"eq_speed={diag.speed_max_equatorial[-1]*1e3:.2f} mm/s "
                  f"[{elapsed:.0f}s]")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Stable: {diag.is_stable()}")
    print(f"  Max speed: {diag.max_speed()*1e3:.2f} mm/s")
    print(f"  KE trend: {diag.ke_trend():.4f} /day")

    return diag


# ---------------------------------------------------------------------------
# Tier presets
# ---------------------------------------------------------------------------

TIER_PRESETS = {
    1: dict(
        bathymetry_type="seamount",
        stratification="uniform",
        days=90,
        smoothing_passes=5,
    ),
    2: dict(
        bathymetry_type="seamount",
        stratification="woa",
        days=90,
        smoothing_passes=5,
    ),
    3: dict(
        bathymetry_type="etopo",
        stratification="woa",
        days=90,
        smoothing_passes=5,
        n_lat=90,
        n_lon=180,
        H_max=5000.0,
        dz_surface=20.0,
        dz_deep=500.0,
    ),
}


def tier_config_from_args(args) -> TierConfig:
    """Build a TierConfig from CLI args with tier-specific defaults."""
    # Start with tier preset
    preset = TIER_PRESETS.get(args.tier, {})

    # CLI args override preset
    cfg = TierConfig(
        tier=args.tier,
        n_lat=args.n_lat or preset.get("n_lat", 36),
        n_lon=args.n_lon or preset.get("n_lon", 72),
        n_levels=args.n_levels or preset.get("n_levels", 20),
        H_max=args.H_max or preset.get("H_max", 4000.0),
        dz_surface=args.dz_surface or preset.get("dz_surface", 10.0),
        dz_deep=args.dz_deep or preset.get("dz_deep", 500.0),
        pgf_scheme=args.pgf_scheme,
        coord=args.coord,
        bathymetry_type=args.bathymetry_type or preset.get("bathymetry_type", "seamount"),
        stratification=args.stratification or preset.get("stratification", "uniform"),
        smoothing_passes=args.smoothing_passes if args.smoothing_passes is not None
            else preset.get("smoothing_passes", 5),
        barotropic_solver=args.barotropic_solver,
        momentum_advection=args.momentum_advection,
        bottom_drag_r=args.bottom_drag_r,
        A_h=args.A_h,
        days=args.days or preset.get("days", 30.0),
        dt=args.dt,
        record_every_days=args.record_every_days,
        H_min=args.H_min,
        T_water_init_C=preset.get("T_water_init_C", 20.0),
        T_deep=preset.get("T_deep", 2.0),
        S_uniform=preset.get("S_uniform", 35.0),
    )
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PGF tiered test ladder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tier", type=int, default=None,
                        help="Run a specific tier (1-3). Default: run all.")
    # Grid
    parser.add_argument("--n-lat", type=int, default=None)
    parser.add_argument("--n-lon", type=int, default=None)
    parser.add_argument("--n-levels", type=int, default=None)
    parser.add_argument("--H-max", type=float, default=None)
    parser.add_argument("--dz-surface", type=float, default=None)
    parser.add_argument("--dz-deep", type=float, default=None)
    # PGF
    parser.add_argument("--pgf-scheme", type=str, default="adcroft",
                        choices=["adcroft", "smc03"])
    parser.add_argument("--coord", type=str, default="partial",
                        choices=["zstar", "partial"])
    # Bathymetry / stratification overrides
    parser.add_argument("--bathymetry-type", type=str, default=None,
                        choices=["flat", "step", "seamount", "ridge", "etopo"])
    parser.add_argument("--stratification", type=str, default=None,
                        choices=["uniform", "exponential", "woa"])
    parser.add_argument("--smoothing-passes", type=int, default=None)
    parser.add_argument("--H-min", type=float, default=50.0)
    # Model
    parser.add_argument("--barotropic-solver", type=str, default="implicit_cn",
                        choices=["explicit_substep", "implicit_cn"])
    parser.add_argument("--momentum-advection", type=str, default="vector_invariant",
                        choices=["vector_invariant", "weno5", "weno7"])
    parser.add_argument("--bottom-drag-r", type=float, default=1.0e-3)
    parser.add_argument("--A-h", type=float, default=1.0e4)
    # Integration
    parser.add_argument("--days", type=float, default=None)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--record-every-days", type=float, default=1.0)
    # Sweep mode
    parser.add_argument("--sweep-pgf", action="store_true",
                        help="Run both adcroft and smc03 for comparison")

    args = parser.parse_args()

    # Determine which tiers to run
    if args.tier is not None:
        tiers = [args.tier]
    else:
        tiers = [1, 2, 3]

    # Determine PGF schemes to test
    if args.sweep_pgf:
        pgf_schemes = ["adcroft", "smc03"]
    else:
        pgf_schemes = [args.pgf_scheme]

    all_results = {}

    for tier in tiers:
        for pgf in pgf_schemes:
            args.pgf_scheme = pgf
            args.tier = tier
            cfg = tier_config_from_args(args)

            # Output directory
            tag = f"tier{tier}_{pgf}_{cfg.coord}_{cfg.bathymetry_type}_{cfg.stratification}"
            out_dir = Path(f"results/pgf_validation/{tag}")
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg.output_dir = str(out_dir)

            # Run
            diag = run_tier(cfg)

            # Save diagnostics
            with open(out_dir / "diagnostics.json", "w") as f:
                json.dump(diag.to_dict(), f, indent=2, default=str)
            with open(out_dir / "config.json", "w") as f:
                json.dump(asdict(cfg), f, indent=2)

            # Plot
            try:
                plot_diagnostics(diag, cfg, out_dir)
                print(f"  Plots saved to {out_dir}/")
            except Exception as e:
                print(f"  Plot failed: {e}")

            # Store result
            key = f"tier{tier}_{pgf}"
            all_results[key] = {
                "stable": diag.is_stable(),
                "max_speed_mm_s": diag.max_speed() * 1e3,
                "ke_trend": diag.ke_trend(),
                "output_dir": str(out_dir),
            }

    # Print comparison table
    print(f"\n{'='*70}")
    print("PGF TIER LADDER — SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':<35} {'Stable':>7} {'MaxSpeed':>12} {'KE trend':>12}")
    print(f"{'':35} {'':>7} {'[mm/s]':>12} {'[1/day]':>12}")
    print(f"{'-'*70}")
    for key, res in all_results.items():
        stable = "YES" if res["stable"] else "NO"
        speed = f"{res['max_speed_mm_s']:.2f}"
        trend = f"{res['ke_trend']:.4f}" if np.isfinite(res["ke_trend"]) else "N/A"
        print(f"{key:<35} {stable:>7} {speed:>12} {trend:>12}")

    # Save summary
    summary_path = Path("results/pgf_validation/summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
