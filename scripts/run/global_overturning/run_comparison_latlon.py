#!/usr/bin/env python
"""Lat-lon side of the MPAS-vs-LatLon comparison experiment.

Matched configuration with run_comparison_mpas.py — see
docs/ocean/experiments/mpas_vs_latlon_comparison_plan.md for full details.

Key settings (shared with MPAS):
  - ETOPO bathymetry, H_max=5500, H_min=10, smooth=2, MEO r=0.2, snap=30%
  - North cap at 80°N
  - dt=1200s, 30-day initial test
  - A_h=1e4, C_smag_lap=0.33
  - Implicit vertical mixing, KPP (K_conv=1.0), enhanced diffusion convection
  - TVD tracer advection, GM/Redi κ=600
  - Quadratic bottom drag (r=1e-3, BBL=100m, u_bg=0.1)
  - Adcroft PGF, implicit-CN barotropic
  - Per-timestep scalar diagnostics, daily 3D snapshots

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_comparison_latlon.py
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_comparison_latlon.py --days 365 --restart results/ocean/comparison_mpas_v_latlon/latlon/restarts/restart_day000030.npz
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

# Set fp64 precision BEFORE any model imports.
# Matches MPAS script; eliminates float32 rounding errors in implicit
# solver at thin partial cells.
from legoesm.core.precision import set_policy, PrecisionPolicy
set_policy(PrecisionPolicy.fp64())

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.latlon import create_latlon_grid, create_mercator_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, KPPConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    LateralMixingConfig, GMRediConfig, VisbeckConfig,
)
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import (
    OceanConvectionConfig, EnhancedDiffusionConfig,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate, compute_centroid_depth,
)


# ============================================================================
# Shared configuration constants (must match run_comparison_mpas.py)
# ============================================================================

N_LAT = 180
N_LON = 360
N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
DT = 1200.0               # seconds
SNAP_FRAC = 0.30
NORTH_CAP_LAT = 80.0

# Viscosity
A_H = 1.0e4               # constant Laplacian floor [m²/s]
C_SMAG_LAP = 0.33         # Smagorinsky Laplacian coefficient

# Vertical mixing
A_V = 1.0e-4
K_V = 1.0e-5

# Bottom drag
BOTTOM_DRAG_R = 1.0e-3
BOTTOM_DRAG_BBL = 100.0
BOTTOM_DRAG_BG_VEL = 0.1

# GM/Redi
KAPPA_GM = 600.0
KAPPA_REDI = 600.0
S_MAX = 0.005

# Forcing
TAU_MAX = 0.1
TROPICAL_WIND_SCALE = 0.5
TROPICAL_WIND_LAT_DEG = 15.0
TAU_T = 2592000.0         # 30-day restoring
TAU_S = 2592000.0
T_STAR_EQ = 25.0
T_STAR_POLE = 0.0
S_STAR = 35.0

OUTPUT_DIR = Path("results/ocean/comparison_mpas_v_latlon/latlon")


# ============================================================================
# Helpers
# ============================================================================

def snap_partial_cells_2d(H_bathy, z_coord, min_frac=SNAP_FRAC):
    """Snap thin partial cells to nearest interface (2D lat-lon version).

    Same algorithm as the MPAS 1D version but operates on (n_lat, n_lon).
    """
    abs_z_half = jnp.abs(z_coord.z_half_ref)
    nlev = z_coord.n_levels
    # Flatten for vectorised computation, then reshape.
    shape = H_bathy.shape
    H_flat = H_bathy.ravel()
    n_above = jnp.sum(abs_z_half[None, :] < H_flat[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    abs_z_at_bottom = abs_z_half[bottom_level]
    dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_flat - abs_z_at_bottom
    frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]
    z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_flat - z_upper < z_lower - H_flat,
                           z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_flat > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_flat)
    H_new = jnp.where(H_new <= 0, 0.0, H_new)
    return H_new.reshape(shape)


def save_restart(state, day, output_dir):
    """Save state as restart_dayXXXXXX.npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": int(round(day * 86400 / DT)), "time_days": float(day),
               "grid_type": "latlon", "n_lat": N_LAT, "n_lon": N_LON}
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        payload[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    np.savez_compressed(fname, **payload)
    return fname


def load_restart(restart_path, template_state):
    """Load restart npz into template state."""
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    replacements = {}
    for f in template_state._fields:
        if f not in data:
            continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = Field(data=arr, name=obj.name, dims=obj.dims,
                                units=obj.units)
    return template_state._replace(**replacements), restart_day


def save_snapshot(state, grid, ocean_mask, day, output_dir):
    """Save 6-panel diagnostic PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_np = np.asarray(ocean_mask) > 0.5
    lon = np.degrees(np.asarray(grid.lon))
    lat = np.degrees(np.asarray(grid.lat))
    LON, LAT = np.meshgrid(lon, lat)

    u_np = np.asarray(state.u.data[:, :, 0])  # surface u
    v_np = np.asarray(state.v.data[:, :, 0])  # surface v
    # Speed at cell centers (average face velocities)
    u_cell = 0.5 * (u_np[:, :-1] + u_np[:, 1:])
    v_cell = 0.5 * (v_np[:-1, :] + v_np[1:, :])
    speed_sfc = np.sqrt(u_cell**2 + v_cell**2)
    speed_sfc = np.where(mask_np, speed_sfc, np.nan)

    eta = np.asarray(state.eta.data)
    sst = np.asarray(state.T.data[:, :, 0])

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    vmax = max(0.01, np.nanpercentile(speed_sfc[mask_np], 99))
    pc = axes[0, 0].pcolormesh(LON, LAT, speed_sfc, cmap="magma",
                                vmin=0, vmax=vmax, shading="auto")
    plt.colorbar(pc, ax=axes[0, 0], label="m/s")
    axes[0, 0].set_title("Surface speed")

    eta_p = np.where(mask_np, eta, np.nan)
    vm = max(0.01, np.nanmax(np.abs(eta_p)))
    pc = axes[0, 1].pcolormesh(LON, LAT, eta_p, cmap="RdBu_r",
                                vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(pc, ax=axes[0, 1], label="m")
    axes[0, 1].set_title("SSH")

    sst_p = np.where(mask_np, sst, np.nan)
    pc = axes[0, 2].pcolormesh(LON, LAT, sst_p, cmap="RdYlBu_r",
                                shading="auto")
    plt.colorbar(pc, ax=axes[0, 2], label="°C")
    axes[0, 2].set_title("SST")

    # Max-depth speed
    u_all = np.asarray(state.u.data)
    v_all = np.asarray(state.v.data)
    u_cell_3d = 0.5 * (u_all[:, :-1, :] + u_all[:, 1:, :])
    v_cell_3d = 0.5 * (v_all[:-1, :, :] + v_all[1:, :, :])
    speed_3d = np.sqrt(u_cell_3d**2 + v_cell_3d**2)
    spd_max = np.where(mask_np, np.max(speed_3d, axis=2), np.nan)
    pc = axes[1, 0].pcolormesh(LON, LAT, spd_max, cmap="magma",
                                vmin=0,
                                vmax=max(0.01, np.nanpercentile(
                                    spd_max[mask_np], 99)),
                                shading="auto")
    plt.colorbar(pc, ax=axes[1, 0], label="m/s")
    axes[1, 0].set_title("Max-depth speed")

    sss = np.asarray(state.S.data[:, :, 0])
    sss_p = np.where(mask_np, sss, np.nan)
    pc = axes[1, 1].pcolormesh(LON, LAT, sss_p, cmap="YlGnBu",
                                shading="auto")
    plt.colorbar(pc, ax=axes[1, 1], label="PSU")
    axes[1, 1].set_title("SSS")

    deep_lev = min(15, state.T.data.shape[2] - 1)
    T_deep = np.asarray(state.T.data[:, :, deep_lev])
    T_deep_p = np.where(mask_np, T_deep, np.nan)
    pc = axes[1, 2].pcolormesh(LON, LAT, T_deep_p, cmap="RdYlBu_r",
                                shading="auto")
    plt.colorbar(pc, ax=axes[1, 2], label="°C")
    axes[1, 2].set_title(f"T at level {deep_lev}")

    for ax in axes.flat:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

    fig.suptitle(f"Lat-lon comparison — day {day:.1f} "
                 f"(year {day/365.25:.2f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"snapshot_day{int(round(day)):06d}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def _apply_from_config(args, p):
    """Load a config.json and use it to fill in unset CLI args.

    Explicit CLI flags always win. config.json values are used only
    for args that were not provided on the command line.
    """
    import json
    if not args.from_config:
        return args
    with open(args.from_config) as f:
        cfg = json.load(f)
    # Map config.json keys → argparse dest names
    mapping = {
        "dt": "dt", "days": "days",
        "A_h": "a_h", "B_h": "b_h", "C_smag_lap": "c_smag_lap",
        "C_smag": "c_smag", "K_h": "k_h",
        "A_h_floor": "a_h_floor", "A_h_merid": "a_h_merid",
        "A_h_eq_boost": "a_h_eq_boost",
        "A_h_eq_sigma_deg": "a_h_eq_sigma_deg",
        "B_h_barotropic": "b_h_barotropic",
        "kappa_GM": "kappa_gm", "kappa_Redi": "kappa_redi",
        "S_max": "s_max",
        "ke_gradient_scheme": "ke_gradient_scheme",
        "slope_foot_alpha": "slope_foot_alpha",
        "momentum_advection": "momentum_advection",
        "save_every_days": "save_every_days",
        "n_lon": "n_lon", "lat_max": "lat_max",
    }
    # Boolean flags
    bool_mapping = {
        "flat_bottom": "flat_bottom",
        "uniform_T": "uniform_T",
        "mercator": "mercator",
    }
    # Detect which args were explicitly set on the command line
    # (argparse doesn't track this, so we compare against defaults)
    defaults = vars(p.parse_args([]))
    current = vars(args)
    for cfg_key, arg_key in mapping.items():
        if cfg_key in cfg and current.get(arg_key) == defaults.get(arg_key):
            setattr(args, arg_key, cfg[cfg_key])
    for cfg_key, arg_key in bool_mapping.items():
        if cfg_key in cfg and not current.get(arg_key):
            setattr(args, arg_key, cfg[cfg_key])
    print(f"  Loaded config from {args.from_config}")
    # Print which values came from config vs CLI
    for cfg_key, arg_key in {**mapping, **bool_mapping}.items():
        if cfg_key in cfg:
            src = "config" if current.get(arg_key) == defaults.get(arg_key) else "CLI"
    return args


def main():
    p = argparse.ArgumentParser(
        description="Lat-lon side of MPAS-vs-LatLon comparison.")
    p.add_argument("--from-config", default=None,
                   help="Load parameters from a previous run's config.json. "
                        "Explicit CLI flags override config.json values.")
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--restart", default=None)
    p.add_argument("--tag", default=None,
                   help="Experiment tag (e.g., 'e1'). Output goes to "
                        "results/.../latlon_{tag}/. If omitted, uses 'latlon/'.")
    p.add_argument("--save-every-days", type=int, default=1,
                   help="Save restart and snapshot every N days (default 1).")
    p.add_argument("--b-h", type=float, default=None,
                   help="Override B_h biharmonic viscosity [m⁴/s] (default: 0).")
    p.add_argument("--a-h", type=float, default=None,
                   help="Override A_h Laplacian viscosity [m²/s] (default: 1e4).")
    p.add_argument("--c-smag-lap", type=float, default=None,
                   help="Override C_smag_lap (default: 0.33).")
    p.add_argument("--a-h-lat-scaling", action="store_true", default=False,
                   help="Enable cos(lat) scaling on A_h.")
    p.add_argument("--a-h-floor", type=float, default=None,
                   help="Minimum A_h after lat scaling [m²/s].")
    p.add_argument("--flat-bottom", action="store_true",
                   help="Use flat bottom (H=H_MAX everywhere) with "
                        "same coastlines from ETOPO.")
    p.add_argument("--b-h-barotropic", type=float, default=None,
                   help="Override B_h_barotropic [m⁴/s] (default: 0).")
    p.add_argument("--momentum-advection", default=None,
                   help="Override momentum_advection scheme "
                        "(default: vector_invariant). Options: "
                        "vector_invariant, weno5, weno7.")
    p.add_argument("--c-smag", type=float, default=None,
                   help="Override C_smag biharmonic Smagorinsky (default: 0).")
    p.add_argument("--kappa-gm", type=float, default=None,
                   help="Override kappa_GM [m²/s] (default: 600).")
    p.add_argument("--kappa-redi", type=float, default=None,
                   help="Override kappa_Redi [m²/s] (default: 600).")
    p.add_argument("--k-h", type=float, default=None,
                   help="Override K_h horizontal tracer diffusivity [m²/s] (default: 0).")
    p.add_argument("--s-max", type=float, default=None,
                   help="Override GM/Redi S_max slope limit (default: 0.005).")
    p.add_argument("--dt", type=float, default=None,
                   help="Override timestep [s] (default: 1200).")
    p.add_argument("--no-bh-lat-scaling", action="store_true",
                   help="Disable cos⁴(lat) scaling on B_h.")
    p.add_argument("--a-h-merid", type=float, default=None,
                   help="Meridional-only Laplacian viscosity [m²/s] (default: 0).")
    p.add_argument("--uniform-T", action="store_true",
                   help="Initialize with uniform T=10°C (barotropic test).")
    p.add_argument("--mercator", action="store_true",
                   help="Use Mercator grid (isotropic cells) instead of "
                        "regular lat-lon.")
    p.add_argument("--n-lon", type=int, default=None,
                   help="Override N_LON (default: 360). For quick tests "
                        "use 180 (2° resolution).")
    p.add_argument("--lat-max", type=float, default=None,
                   help="Override NORTH_CAP_LAT for Mercator grid [deg]. "
                        "Lower values give larger polar cells (default: 80).")
    p.add_argument("--ke-gradient-scheme", default=None,
                   help="KE gradient scheme: 'centered' (default) or "
                        "'hollingsworth' (NEMO nkeg_HW; required for "
                        "stratified ocean over sloping bathymetry, "
                        "fixes Hollingsworth-Kallberg instability).")
    p.add_argument("--a-h-eq-boost", type=float, default=None,
                   help="Equatorial A_h boost factor (DINO uses 3.0). "
                        "Stabilizes weak-Coriolis equatorial region.")
    p.add_argument("--a-h-eq-sigma-deg", type=float, default=None,
                   help="Equatorial boost Gaussian half-width [deg] "
                        "(DINO uses 5.0).")
    p.add_argument("--slope-foot-alpha", type=float, default=None,
                   help="Adcroft PGF slope foot alpha (DINO production: 3.0).")
    p.add_argument("--etopo",
                   default=os.environ.get("LEGOESM_ETOPO_PATH", "data/bathymetry/etopo_1deg.nc"))
    args = p.parse_args()
    args = _apply_from_config(args, p)

    if args.tag:
        outdir = OUTPUT_DIR.parent / f"latlon_{args.tag}"
    else:
        outdir = OUTPUT_DIR
    restart_dir = outdir / "restarts"
    snapshot_dir = outdir / "snapshots"
    restart_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- Grid ---
    n_lon = args.n_lon if args.n_lon else N_LON
    n_lat = N_LAT * n_lon // N_LON  # scale proportionally
    lat_max = args.lat_max if args.lat_max else NORTH_CAP_LAT
    if args.mercator:
        grid = create_mercator_grid(n_lon=n_lon, lat_max_deg=lat_max)
        print(f"=== Mercator comparison run: {grid.n_lat}x{grid.n_lon}, "
              f"{args.days} days ===")
        print("  *** MERCATOR grid (isotropic cells) ***")
    else:
        grid = create_latlon_grid(n_lat, n_lon)
        print(f"=== Lat-lon comparison run: {n_lat}x{n_lon}, "
              f"{args.days} days ===")
    z_coord_base = create_ocean_z_star(n_levels=N_LEVELS, H_max=H_MAX,
                                       dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP)

    # --- Bathymetry (matched with MPAS) ---
    bathy_cfg = BathymetryConfig(
        source="file", path=args.etopo,
        H_max=H_MAX, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True,
        north_cap_lat=lat_max,
        south_cap_lat=None,     # full Southern Ocean
    )
    H_bathy_raw, ocean_mask = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy_raw = jnp.asarray(H_bathy_raw, dtype=jnp.float64)
    ocean_mask = jnp.asarray(ocean_mask, dtype=jnp.float64)

    # Flat bottom option: keep coastlines, set all ocean to H_MAX
    if args.flat_bottom:
        H_bathy_raw = jnp.where(ocean_mask > 0.5, H_MAX, 0.0)
        print("  *** FLAT BOTTOM mode: H = H_MAX everywhere ***")

    # Snap partial cells (same 30% as MPAS)
    H_snapped = snap_partial_cells_2d(H_bathy_raw, z_coord_base)
    ocean_mask = jnp.where(H_snapped > 0, ocean_mask, 0.0)

    z_coord = create_partial_cell_coordinate(z_coord_base, H_snapped)

    n_ocean = int(jnp.sum(ocean_mask > 0.5))
    n_total = int(ocean_mask.size)
    print(f"  Grid: {grid.n_lat}x{grid.n_lon}, {N_LEVELS} levels")
    print(f"  Ocean cells: {n_ocean}/{n_total} "
          f"({100.0*n_ocean/n_total:.1f}%)")
    print(f"  Vertical: dz_sfc={DZ_SURFACE}m, dz_deep={DZ_DEEP}m")

    # --- Initial condition ---
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_water_init_C=20.0, T_deep=2.0,
        S_uniform=S_STAR, H_max=H_MAX,
        land_mask_override=ocean_mask,
        H_bathy_override=H_snapped,
    )
    # Temperature initialization
    if args.uniform_T:
        T_init = jnp.where(ocean_mask[..., jnp.newaxis] > 0.5,
                            10.0 * jnp.ones_like(state.T.data), 0.0)
        print("  *** UNIFORM T = 10°C (barotropic test) ***")
    else:
        # Centroid-aware exponential T(z) — same as MPAS
        centroid = compute_centroid_depth(
            jnp.zeros_like(H_snapped), H_snapped, z_coord,
        )
        T_init = 2.0 + 18.0 * jnp.exp(-centroid / _SCALE_DEPTH)
        T_init = jnp.where(z_coord.is_active, T_init, 0.0)
        T_init = T_init * ocean_mask[..., jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=T_init.astype(state.T.data.dtype)),
    )

    # --- Physics ---
    # For barotropic tests (--uniform-T): wind only, no T/S restoring,
    # no KPP, no convection — purely barotropic dynamics.
    if args.uniform_T:
        physics = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(
                scheme="prescribed",
                prescribed=PrescribedForcingConfig(
                    wind_profile="global_wind", tau_max=TAU_MAX,
                    tropical_wind_scale=TROPICAL_WIND_SCALE,
                    tropical_wind_lat_deg=TROPICAL_WIND_LAT_DEG,
                ),
            ),
            vertical_mixing=VerticalMixingConfig(scheme="none"),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(scheme="none"),
            shortwave_penetration=None,
        )
        print("  *** BAROTROPIC PHYSICS: wind only, no T/S restoring ***")
    else:
        physics = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(
                scheme="combined",
                prescribed=PrescribedForcingConfig(
                    wind_profile="global_wind", tau_max=TAU_MAX,
                    tropical_wind_scale=TROPICAL_WIND_SCALE,
                    tropical_wind_lat_deg=TROPICAL_WIND_LAT_DEG,
                ),
                restoring=RestoringConfig(
                    tau_T=TAU_T, tau_S=TAU_S,
                    T_star_eq=T_STAR_EQ, T_star_pole=T_STAR_POLE,
                    S_star=S_STAR, T_profile="cosine",
                ),
            ),
            vertical_mixing=VerticalMixingConfig(
                scheme="kpp",
                kpp=KPPConfig(K_conv=1.0),
            ),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
            ),
            shortwave_penetration=None,
        )

    # --- Model config ---
    ocean_config = LatLonCGridOceanConfig.from_flat(
        A_h=args.a_h if args.a_h is not None else A_H,
        A_h_lat_scaling=args.a_h_lat_scaling,
        A_h_floor=args.a_h_floor if args.a_h_floor is not None else 0.0,
        B_h=args.b_h if args.b_h is not None else 0.0,
        B_h_lat_scaling=not args.no_bh_lat_scaling,
        B_h_barotropic=args.b_h_barotropic if args.b_h_barotropic is not None else 0.0,
        C_smag_lap=args.c_smag_lap if args.c_smag_lap is not None else C_SMAG_LAP,
        C_smag=args.c_smag if args.c_smag is not None else 0.0,
        A_v=A_V,
        K_v=K_V,
        K_h=args.k_h if args.k_h is not None else 0.0,
        A_h_merid=args.a_h_merid if args.a_h_merid is not None else 0.0,
        bottom_drag_r=BOTTOM_DRAG_R,
        bottom_drag_bbl_thickness=BOTTOM_DRAG_BBL,
        bottom_drag_bg_velocity=BOTTOM_DRAG_BG_VEL,
        pgf_scheme="adcroft",
        barotropic_solver="implicit_cn",
        momentum_advection=args.momentum_advection if args.momentum_advection is not None else "vector_invariant",
        ke_gradient_scheme=args.ke_gradient_scheme if args.ke_gradient_scheme is not None else "centered",
        A_h_eq_boost=args.a_h_eq_boost if args.a_h_eq_boost is not None else 1.0,
        A_h_eq_sigma_deg=args.a_h_eq_sigma_deg if args.a_h_eq_sigma_deg is not None else 5.0,
        slope_foot_alpha=args.slope_foot_alpha if args.slope_foot_alpha is not None else 0.0,
        implicit_vertical_mixing=True,
        tracer_advection="tvd",
        eos="wright",
        freshwater_closure="virtual_salt_flux",
        S_ref=S_STAR,
        gm_redi=GMRediConfig(
            kappa_GM=args.kappa_gm if args.kappa_gm is not None else KAPPA_GM,
            kappa_Redi=args.kappa_redi if args.kappa_redi is not None else KAPPA_REDI,
            S_max=args.s_max if args.s_max is not None else S_MAX,
            visbeck=VisbeckConfig(enabled=False),
            slope_scheme="centered",
        ),
        physics=physics,
    )

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    # --- Restart? ---
    start_day = 0.0
    if args.restart:
        state, start_day = load_restart(args.restart, state)
        print(f"  Resumed from {args.restart} at day {start_day:.0f}")

    # --- Save and print actual config ---
    import json
    dt = args.dt if args.dt is not None else DT
    run_config = {
        "tag": args.tag or "default",
        "grid_type": "mercator" if args.mercator else "latlon",
        "n_lat": int(grid.n_lat),
        "n_lon": int(grid.n_lon),
        "lat_max": float(lat_max),
        "n_levels": N_LEVELS,
        "H_max": H_MAX,
        "dz_surface": DZ_SURFACE,
        "dz_deep": DZ_DEEP,
        "flat_bottom": args.flat_bottom,
        "uniform_T": args.uniform_T,
        "dt": float(dt),
        "days": float(args.days),
        "save_every_days": float(args.save_every_days),
        "A_h": float(ocean_config.lateral_viscosity.A_h),
        "A_h_lat_scaling": float(ocean_config.lateral_viscosity.A_h_lat_scaling),
        "A_h_floor": float(ocean_config.lateral_viscosity.A_h_floor),
        "A_h_merid": float(ocean_config.lateral_viscosity.A_h_merid),
        "A_h_eq_boost": float(ocean_config.lateral_viscosity.A_h_eq_boost),
        "A_h_eq_sigma_deg": float(ocean_config.lateral_viscosity.A_h_eq_sigma_deg),
        "ke_gradient_scheme": ocean_config.ke_gradient_scheme,
        "slope_foot_alpha": float(ocean_config.slope_foot_alpha),
        "B_h": float(ocean_config.lateral_viscosity.B_h),
        "B_h_lat_scaling": ocean_config.lateral_viscosity.B_h_lat_scaling,
        "B_h_barotropic": float(ocean_config.lateral_viscosity.B_h_barotropic),
        "C_smag_lap": float(ocean_config.lateral_viscosity.C_smag_lap),
        "C_smag": float(ocean_config.lateral_viscosity.C_smag),
        "A_v": float(ocean_config.A_v),
        "K_v": float(ocean_config.K_v),
        "K_h": float(ocean_config.K_h),
        "bottom_drag_r": float(ocean_config.bottom_drag.bottom_drag_r),
        "bottom_drag_bbl_thickness": float(ocean_config.bottom_drag.bottom_drag_bbl_thickness),
        "bottom_drag_bg_velocity": float(ocean_config.bottom_drag.bottom_drag_bg_velocity),
        "pgf_scheme": ocean_config.pgf_scheme,
        "barotropic_solver": ocean_config.barotropic.barotropic_solver,
        "momentum_advection": ocean_config.momentum_advection,
        "tracer_advection": ocean_config.tracer_advection,
        "eos": ocean_config.eos,
        "freshwater_closure": ocean_config.freshwater_closure,
        "S_ref": float(ocean_config.S_ref),
        "implicit_vertical_mixing": ocean_config.implicit_vertical_mixing,
        "kappa_GM": float(ocean_config.gm_redi.kappa_GM),
        "kappa_Redi": float(ocean_config.gm_redi.kappa_Redi),
        "S_max": float(ocean_config.gm_redi.S_max),
        "slope_scheme": ocean_config.gm_redi.slope_scheme,
        "surface_complement": ocean_config.gm_redi.surface_complement,
        "visbeck_enabled": ocean_config.gm_redi.visbeck.enabled,
        "wind_profile": "global_wind",
        "tau_max": TAU_MAX,
        "tropical_wind_scale": TROPICAL_WIND_SCALE,
        "tropical_wind_lat_deg": TROPICAL_WIND_LAT_DEG,
        "T_star_eq": T_STAR_EQ,
        "T_star_pole": T_STAR_POLE,
        "S_star": S_STAR,
        "tau_T_days": TAU_T / 86400,
        "tau_S_days": TAU_S / 86400,
        "precision": "fp64",
        "restart_from": str(args.restart) if args.restart else None,
        "command": " ".join(sys.argv),
    }
    config_path = outdir / "config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"\n  Config saved to {config_path}")

    # Print key parameters (from actual config, not constants)
    print(f"  Config:")
    print(f"    A_h={ocean_config.lateral_viscosity.A_h:.0e}, "
          f"C_smag_lap={ocean_config.lateral_viscosity.C_smag_lap}")
    print(f"    A_v={ocean_config.A_v:.0e}, K_v={ocean_config.K_v:.0e}")
    print(f"    GM/Redi: κ_GM={ocean_config.gm_redi.kappa_GM}, "
          f"κ_Redi={ocean_config.gm_redi.kappa_Redi}, "
          f"S_max={ocean_config.gm_redi.S_max}")
    print(f"    dt={dt}s, tracer_advection={ocean_config.tracer_advection}")
    print(f"    Wind: τ_max={TAU_MAX}, tropical_scale={TROPICAL_WIND_SCALE}")

    # --- Time loop ---
    total_days = args.days
    dt = args.dt if args.dt is not None else DT
    n_steps = int(total_days * 86400 / dt)
    diag_every_day = args.save_every_days
    diag_steps = int(diag_every_day * 86400 / dt)

    # Per-timestep CSV
    csv_path = outdir / "timeseries.csv"
    csv_exists = csv_path.exists() and args.restart
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "step", "day", "max_u", "max_eta", "mean_SST",
            "global_KE", "max_CFL_h", "wall_s",
        ])

    print(f"\n  Running {total_days:.0f} days ({n_steps} steps)")
    print(f"  Per-timestep scalars → {csv_path}")
    print(f"  Daily snapshots → {snapshot_dir}\n")

    # Precompute for diagnostics — use OCEAN-only cells for dx_min
    R = constants.R_earth
    dlat = grid.lat[1] - grid.lat[0]   # uniform spacing [rad]
    dlon = grid.dlon                     # scalar [rad]
    cos_lat = grid.cos_lat               # (n_lat,)
    # Mask to ocean latitudes only (any ocean cell in that row)
    ocean_row = jnp.any(ocean_mask > 0.5, axis=1)  # (n_lat,)
    cos_ocean = jnp.where(ocean_row, cos_lat, 1.0)  # 1.0 for land rows
    dx_min = float(R * dlon * jnp.min(cos_ocean))
    dy = float(R * dlat)

    t0 = time.time()
    for k in range(n_steps):
        state = model.step(state, dt=dt)
        day = start_day + (k + 1) * dt / 86400.0
        step = int(round(day * 86400 / dt))

        # --- Per-timestep scalar diagnostics ---
        u_data = state.u.data
        v_data = state.v.data
        eta_data = state.eta.data
        T_data_cur = state.T.data

        max_u_comp = float(jnp.maximum(
            jnp.max(jnp.abs(u_data)), jnp.max(jnp.abs(v_data))))
        max_eta = float(jnp.max(jnp.abs(eta_data)))
        mean_sst = float(
            jnp.where(ocean_mask > 0.5, T_data_cur[:, :, 0], 0.0).sum()
            / jnp.maximum(jnp.sum(ocean_mask > 0.5), 1)
        )

        # Global KE (approximate: u² + v² at cell centers, weighted by area)
        u_cell = 0.5 * (u_data[:, :-1, :] + u_data[:, 1:, :])
        v_cell = 0.5 * (v_data[:-1, :, :] + v_data[1:, :, :])
        speed_sq = u_cell**2 + v_cell**2
        area = grid.area  # (n_lat, n_lon)
        global_KE = float(0.5 * jnp.sum(
            speed_sq * area[..., jnp.newaxis] * ocean_mask[..., jnp.newaxis]))

        # Horizontal CFL: max(|u| * dt / dx) — use minimum dx (near poles)
        max_CFL_h = float(max_u_comp * dt / min(dx_min, dy))

        wall = time.time() - t0
        csv_writer.writerow([
            step, f"{day:.4f}", f"{max_u_comp:.6e}", f"{max_eta:.6e}",
            f"{mean_sst:.4f}", f"{global_KE:.6e}", f"{max_CFL_h:.4f}",
            f"{wall:.1f}",
        ])
        csv_file.flush()

        # --- Console output ---
        if (k + 1) % diag_steps == 0 or k == 0 or k == n_steps - 1:
            year = day / 365.25
            print(f"  day {day:6.1f}  max|u|={max_u_comp:.4f}  "
                  f"max|eta|={max_eta:.4f}  <SST>={mean_sst:.2f}  "
                  f"CFL={max_CFL_h:.3f}  wall={wall:.0f}s")

        # --- Blowup check ---
        if not np.isfinite(max_u_comp) or max_u_comp > 20:
            print(f"  *** BLOWUP at day {day:.1f} ***")
            save_restart(state, day, restart_dir)
            save_snapshot(state, grid, ocean_mask, day, snapshot_dir)
            break

        # --- Daily snapshot ---
        if (k + 1) % diag_steps == 0 or k == n_steps - 1:
            save_restart(state, day, restart_dir)
            save_snapshot(state, grid, ocean_mask, day, snapshot_dir)

    csv_file.close()
    wall_total = time.time() - t0
    print(f"\nDone: {day:.0f} days in {wall_total:.0f}s "
          f"({wall_total/60:.1f} min)")


if __name__ == "__main__":
    main()
