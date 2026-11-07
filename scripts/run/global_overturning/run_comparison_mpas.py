#!/usr/bin/env python
"""MPAS side of the MPAS-vs-LatLon comparison experiment.

Matched configuration with run_comparison_latlon.py — see
docs/ocean/experiments/mpas_vs_latlon_comparison_plan.md for full details.

Key settings (shared with lat-lon):
  - ETOPO bathymetry, H_max=5500, H_min=10, smooth=2, MEO r=0.2, snap=30%
  - North cap at 80°N (for fairness with lat-lon)
  - dt=1200s, 30-day initial test
  - A_h=1e4, C_smag_lap=0.33
  - Implicit vertical mixing, KPP (K_conv=1.0), enhanced diffusion convection
  - TVD tracer advection, GM/Redi κ=600
  - Quadratic bottom drag (r=1e-3, BBL=100m, u_bg=0.1)
  - Adcroft PGF, implicit-CN barotropic
  - Per-timestep scalar diagnostics, daily 3D snapshots

MPAS-only: K_zeta_bih (TRiSK null mode damping), derived from the mesh spacing.

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_comparison_mpas.py
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_comparison_mpas.py --days 365 --restart results/ocean/comparison_mpas_v_latlon/mpas/restarts/restart_day000030.npz
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
# float32 implicit solver accumulates rounding errors at thin partial
# cells, causing ~0.001 PSU/10-day S drift.  float64 eliminates this.
from legoesm.core.precision import set_policy, PrecisionPolicy
set_policy(PrecisionPolicy.fp64())

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_mpas import rest_state_mpas_ocean, reconstruct_cell_velocity
from legoesm.ocean.mpas_config import MPASOceanConfig
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
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate


# ============================================================================
# Shared configuration constants (must match run_comparison_latlon.py)
# ============================================================================

SUBDIVISION = 5           # ico5 ~ 120 km ~ 1°
N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
DT = 1200.0               # seconds
SNAP_FRAC = 0.30
NORTH_CAP_LAT = 80.0      # cap Arctic for parity with lat-lon

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

OUTPUT_DIR = Path("results/ocean/comparison_mpas_v_latlon/mpas")


# ============================================================================
# Helpers
# ============================================================================

def snap_partial_cells(H_bathy, z_coord, min_frac=SNAP_FRAC):
    """Round H_bathy to nearest interface when bottom partial cell < min_frac."""
    abs_z_half = jnp.abs(z_coord.z_half_ref)
    nlev = z_coord.n_levels
    n_above = jnp.sum(abs_z_half[None, :] < H_bathy[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    abs_z_at_bottom = abs_z_half[bottom_level]
    dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_bathy - abs_z_at_bottom
    frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]
    z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_bathy - z_upper < z_lower - H_bathy,
                           z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_bathy > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_bathy)
    return jnp.where(H_new <= 0, 0.0, H_new)


def save_restart(state, day, output_dir):
    """Save state as restart_dayXXXXXX.npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": int(round(day * 86400 / DT)), "time_days": float(day),
               "grid_type": "mpas", "subdivision": SUBDIVISION}
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


def save_snapshot(state, mesh, ocean_mask, day, output_dir):
    """Save 6-panel diagnostic PNG using tripcolor + cartopy."""
    import matplotlib.tri as mtri
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _has_cartopy = True
    except ImportError:
        _has_cartopy = False

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_np = np.asarray(ocean_mask) > 0.5
    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))

    u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
    speed = np.asarray(jnp.sqrt(u_east**2 + v_north**2))
    eta = np.asarray(state.eta.data)
    sst = np.asarray(state.T.data[:, 0])

    # Shift lon to [-180, 180] for cartopy
    lon_shifted = np.where(lon > 180, lon - 360, lon)

    # Build Delaunay triangulation; mask land triangles
    tri = mtri.Triangulation(lon_shifted, lat)
    mask_tri = np.all(mask_np[tri.triangles], axis=1)
    tri.set_mask(~mask_tri)

    def ocean_field(f):
        return np.where(mask_np, f, np.nan)

    if _has_cartopy:
        proj = ccrs.Robinson(central_longitude=200)
        fig, axes = plt.subplots(2, 3, figsize=(24, 12),
                                 subplot_kw={"projection": proj})
    else:
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    def plot_panel(ax, field, title, cmap, vmin=None, vmax=None,
                   symmetric=False):
        f = ocean_field(field)
        if vmin is None:
            vmin = np.nanpercentile(f, 1)
        if vmax is None:
            vmax = np.nanpercentile(f, 99)
        if symmetric:
            vm = max(abs(vmin), abs(vmax))
            vmin, vmax = -vm, vm
        if _has_cartopy:
            tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree(), rasterized=True)
            ax.add_feature(cfeature.LAND, facecolor="0.85",
                           edgecolor="0.5", linewidth=0.3)
            ax.coastlines(linewidth=0.3, color="0.4")
            ax.set_global()
        else:
            tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                              rasterized=True)
        ax.set_title(title, fontsize=12)
        plt.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)

    # Surface speed
    spd_sfc = speed[:, 0]
    plot_panel(axes[0, 0], spd_sfc, "Surface speed [m/s]", "magma",
               vmin=0, vmax=max(0.1, np.nanpercentile(ocean_field(spd_sfc), 99)))
    # SSH
    plot_panel(axes[0, 1], eta, "SSH [m]", "RdBu_r", symmetric=True)
    # SST
    plot_panel(axes[0, 2], sst, "SST [°C]", "RdYlBu_r")
    # Max-depth speed
    spd_max = np.max(speed, axis=1)
    plot_panel(axes[1, 0], spd_max, "Max-depth speed [m/s]", "magma",
               vmin=0, vmax=max(0.1, np.nanpercentile(ocean_field(spd_max), 99)))
    # SSS
    sss = np.asarray(state.S.data[:, 0])
    plot_panel(axes[1, 1], sss, "SSS [PSU]", "YlGnBu")
    # Deep T
    deep_lev = min(15, state.T.data.shape[1] - 1)
    T_deep = np.asarray(state.T.data[:, deep_lev])
    plot_panel(axes[1, 2], T_deep, f"T at level {deep_lev} [°C]", "RdYlBu_r")

    fig.suptitle(f"MPAS comparison — day {day:.1f} "
                 f"(year {day/365.25:.2f})", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / f"snapshot_day{int(round(day)):06d}.png",
                dpi=150, bbox_inches="tight")
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
    mapping = {
        "dt": "dt", "days": "days",
        "A_h": "a_h", "B_h": "b_h",
        "K_zeta_bih": "k_zeta_bih", "apvm_dt": "apvm_dt",
        "kappa_GM": "kappa_gm", "kappa_Redi": "kappa_redi",
        "S_max": "s_max",
        "save_every_days": "save_every_days",
        "C_smag_lap": "c_smag_lap",
    }
    bool_mapping = {
        "flat_bottom": "flat_bottom",
        "uniform_T": "uniform_T",
    }
    defaults = vars(p.parse_args([]))
    current = vars(args)
    for cfg_key, arg_key in mapping.items():
        if cfg_key in cfg and current.get(arg_key) == defaults.get(arg_key):
            setattr(args, arg_key, cfg[cfg_key])
    for cfg_key, arg_key in bool_mapping.items():
        if cfg_key in cfg and not current.get(arg_key):
            setattr(args, arg_key, cfg[cfg_key])
    print(f"  Loaded config from {args.from_config}")
    return args


def main():
    p = argparse.ArgumentParser(
        description="MPAS side of MPAS-vs-LatLon comparison.")
    p.add_argument("--from-config", default=None,
                   help="Load parameters from a previous run's config.json. "
                        "Explicit CLI flags override config.json values.")
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--restart", default=None)
    p.add_argument("--tag", default=None,
                   help="Experiment tag (e.g., 'e0b'). Output goes to "
                        "results/.../mpas_{tag}/. If omitted, uses 'mpas/'.")
    p.add_argument("--save-every-days", type=int, default=1,
                   help="Save restart and snapshot every N days (default 1).")
    p.add_argument("--k-zeta-bih", type=float, default=None,
                   help="Override K_zeta_bih (default: derive it from the "
                        "mesh spacing as K_ref*(dx/dx_ref)^3, anchored on the "
                        "ico6 mesh where 1e14 was tuned).")
    p.add_argument("--apvm-dt", type=float, default=None,
                   help="Override apvm_dt [s] (default: 0 = disabled).")
    p.add_argument("--b-h", type=float, default=None,
                   help="Override B_h biharmonic viscosity [m⁴/s] (default: 0).")
    p.add_argument("--a-h", type=float, default=None,
                   help="Override A_h Laplacian viscosity [m²/s] (default: 1e4).")
    p.add_argument("--c-smag-lap", type=float, default=None,
                   help="Override C_smag_lap (default: 0.33). Set 0 to disable.")
    p.add_argument("--flat-bottom", action="store_true",
                   help="Use flat bottom (H=H_MAX everywhere) with same coastlines.")
    p.add_argument("--uniform-T", action="store_true",
                   help="Initialize with uniform T=10°C (barotropic test).")
    p.add_argument("--kappa-gm", type=float, default=None,
                   help="Override kappa_GM [m²/s] (default: 600).")
    p.add_argument("--kappa-redi", type=float, default=None,
                   help="Override kappa_Redi [m²/s] (default: 600).")
    p.add_argument("--s-max", type=float, default=None,
                   help="Override GM/Redi S_max (default: 0.005).")
    p.add_argument("--dt", type=float, default=None,
                   help="Override timestep [s] (default: 1200).")
    p.add_argument("--etopo",
                   default=os.environ.get("LEGOESM_ETOPO_PATH", "data/bathymetry/etopo_1deg.nc"))
    args = p.parse_args()
    args = _apply_from_config(args, p)

    if args.tag:
        outdir = OUTPUT_DIR.parent / f"mpas_{args.tag}"
    else:
        outdir = OUTPUT_DIR
    restart_dir = outdir / "restarts"
    snapshot_dir = outdir / "snapshots"
    restart_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- Grid ---
    print(f"=== MPAS comparison run: ico{SUBDIVISION}, {args.days} days ===")
    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
    z_coord = create_ocean_z_star(n_levels=N_LEVELS, H_max=H_MAX,
                                  dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP)

    # --- Bathymetry (matched with lat-lon) ---
    bathy_cfg = BathymetryConfig(
        source="file", path=args.etopo,
        H_max=H_MAX, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True,
    )
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)

    # Apply north cap at 80°N for parity with lat-lon
    lat_cell = np.degrees(np.asarray(mesh.latCell))
    north_cap_mask = jnp.asarray(lat_cell <= NORTH_CAP_LAT, dtype=H_bathy_raw.dtype)
    H_bathy_raw = H_bathy_raw * north_cap_mask
    ocean_mask = ocean_mask * north_cap_mask

    # Flat bottom option: keep coastlines, set all ocean to H_MAX
    if args.flat_bottom:
        H_bathy_raw = jnp.where(ocean_mask > 0.5, H_MAX, 0.0)
        print("  *** FLAT BOTTOM mode: H = H_MAX everywhere ***")

    H_snapped = snap_partial_cells(H_bathy_raw, z_coord)
    ocean_mask = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    n_ocean = int(jnp.sum(ocean_mask > 0.5))
    print(f"  Mesh: nCells={mesh.nCells}, nEdges={mesh.nEdges}")
    print(f"  Ocean cells: {n_ocean}/{mesh.nCells} "
          f"(after 80°N cap)")
    print(f"  Vertical: {N_LEVELS} levels, dz_sfc={DZ_SURFACE}m, "
          f"dz_deep={DZ_DEEP}m")

    # --- Physics ---
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
    config = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        A_h=args.a_h if args.a_h is not None else A_H,
        A_v=A_V,
        C_smag_lap=args.c_smag_lap if args.c_smag_lap is not None else C_SMAG_LAP,
        K_v=K_V,
        bottom_drag_r=BOTTOM_DRAG_R,
        bottom_drag_bbl_thickness=BOTTOM_DRAG_BBL,
        bottom_drag_bg_velocity=BOTTOM_DRAG_BG_VEL,
        # None => derived from the mesh spacing (dx^3, ico6-anchored).
        K_zeta_bih=args.k_zeta_bih,
        B_h=args.b_h if args.b_h is not None else 0.0,
        apvm_dt=args.apvm_dt if args.apvm_dt is not None else 0.0,
        equatorial_visc_boost=0.0,
        pgf_scheme="adcroft",             # now works: use_h_actual_pgf=True by default
        implicit_vertical_mixing=True,
        tracer_advection="tvd",
        gm_redi=GMRediConfig(
            kappa_GM=args.kappa_gm if args.kappa_gm is not None else KAPPA_GM,
            kappa_Redi=args.kappa_redi if args.kappa_redi is not None else KAPPA_REDI,
            S_max=args.s_max if args.s_max is not None else S_MAX,
            visbeck=VisbeckConfig(enabled=False),
            slope_scheme="centered",
        ),
        physics=physics,
    )

    model = MPASOceanModel(mesh, pc_coord, config)

    # --- Initial condition ---
    if args.uniform_T:
        T_data = jnp.where(pc_coord.is_active,
                            10.0 * jnp.ones((mesh.nCells, N_LEVELS)), 0.0)
        print("  *** UNIFORM T = 10°C (barotropic test) ***")
    else:
        T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
        T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, N_LEVELS))
        T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(pc_coord.is_active,
                        jnp.full_like(T_data, S_STAR), 0.0)

    state = rest_state_mpas_ocean(
        mesh, z_coord, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=S_STAR, H_max=H_MAX, land_lat_threshold=90.0,
    )
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(data=H_snapped.astype(dtype), name="H_bathy",
                      dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask.astype(dtype)),
        T=Field(data=T_data.astype(dtype), name="T",
                dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data.astype(dtype), name="S",
                dims=("nCells", "nlev"), units="PSU"),
    )

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
        "grid_type": "mpas",
        "subdivision": SUBDIVISION,
        "n_cells": int(mesh.nCells),
        "n_edges": int(mesh.nEdges),
        "n_levels": N_LEVELS,
        "H_max": H_MAX,
        "dz_surface": DZ_SURFACE,
        "dz_deep": DZ_DEEP,
        "flat_bottom": args.flat_bottom,
        "uniform_T": args.uniform_T,
        "dt": float(dt),
        "days": float(args.days),
        "save_every_days": float(args.save_every_days),
        "A_h": float(config.A_h),
        "A_v": float(config.A_v),
        "K_v": float(config.K_v),
        "C_smag_lap": float(config.C_smag_lap),
        "B_h": float(config.B_h),
        "K_zeta_bih": float(model.config.K_zeta_bih),
        "apvm_dt": float(config.apvm_dt),
        "equatorial_visc_boost": float(config.equatorial_visc_boost),
        "bottom_drag_r": float(config.bottom_drag_r),
        "bottom_drag_bbl_thickness": float(config.bottom_drag_bbl_thickness),
        "bottom_drag_bg_velocity": float(config.bottom_drag_bg_velocity),
        "pgf_scheme": config.pgf_scheme,
        "barotropic_solver": config.barotropic_solver,
        "momentum_advection": getattr(config, "momentum_advection", "vector_invariant"),
        "tracer_advection": config.tracer_advection,
        "implicit_vertical_mixing": config.implicit_vertical_mixing,
        "kappa_GM": float(config.gm_redi.kappa_GM),
        "kappa_Redi": float(config.gm_redi.kappa_Redi),
        "S_max": float(config.gm_redi.S_max),
        "slope_scheme": config.gm_redi.slope_scheme,
        "visbeck_enabled": config.gm_redi.visbeck.enabled,
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

    # Print key parameters (from actual config object)
    print(f"  Config:")
    print(f"    A_h={config.A_h:.0e}, C_smag_lap={config.C_smag_lap}")
    print(f"    A_v={config.A_v:.0e}, K_v={config.K_v:.0e}")
    print(f"    K_zeta_bih={model.config.K_zeta_bih:.0e} (MPAS-only)")
    print(f"    GM/Redi: κ_GM={config.gm_redi.kappa_GM}, "
          f"κ_Redi={config.gm_redi.kappa_Redi}, "
          f"S_max={config.gm_redi.S_max}")
    print(f"    dt={dt}s, tracer_advection={config.tracer_advection}")
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

    # Precompute cell areas for KE
    area_cell = jnp.asarray(mesh.areaCell)

    t0 = time.time()
    for k in range(n_steps):
        state = model.step(state, dt=dt)
        day = start_day + (k + 1) * dt / 86400.0
        step = int(round(day * 86400 / dt))

        # --- Per-timestep scalar diagnostics ---
        u_data = state.u.data
        eta_data = state.eta.data
        T_data_cur = state.T.data

        max_u = float(jnp.max(jnp.abs(u_data)))
        max_eta = float(jnp.max(jnp.abs(eta_data)))
        mean_sst = float(
            jnp.where(ocean_mask > 0.5, T_data_cur[:, 0], 0.0).sum()
            / jnp.maximum(jnp.sum(ocean_mask > 0.5), 1)
        )

        # Global KE: 0.5 * rho_0 * sum(u_edge² * dcEdge * dvEdge * h_e)
        # Approximate: use edge lengths as proxy for area
        edge_area = mesh.dcEdge * mesh.dvEdge  # (nEdges,)
        h_e = pc_coord.dz_ref[None, :]  # approximate
        KE_edge = 0.5 * jnp.sum(u_data**2 * edge_area[:, None])
        global_KE = float(KE_edge)

        # Horizontal CFL: max(|u| * dt / dcEdge)
        max_CFL_h = float(jnp.max(
            jnp.max(jnp.abs(u_data), axis=1) * dt / mesh.dcEdge
        ))

        wall = time.time() - t0
        csv_writer.writerow([
            step, f"{day:.4f}", f"{max_u:.6e}", f"{max_eta:.6e}",
            f"{mean_sst:.4f}", f"{global_KE:.6e}", f"{max_CFL_h:.4f}",
            f"{wall:.1f}",
        ])
        csv_file.flush()

        # --- Console output ---
        if (k + 1) % diag_steps == 0 or k == 0 or k == n_steps - 1:
            year = day / 365.25
            print(f"  day {day:6.1f}  max|u|={max_u:.4f}  "
                  f"max|eta|={max_eta:.4f}  <SST>={mean_sst:.2f}  "
                  f"CFL={max_CFL_h:.3f}  wall={wall:.0f}s")

        # --- Blowup check ---
        if not np.isfinite(max_u) or max_u > 20:
            print(f"  *** BLOWUP at day {day:.1f} ***")
            save_restart(state, day, restart_dir)
            save_snapshot(state, mesh, ocean_mask, day, snapshot_dir)
            break

        # --- Daily snapshot ---
        if (k + 1) % diag_steps == 0 or k == n_steps - 1:
            save_restart(state, day, restart_dir)
            save_snapshot(state, mesh, ocean_mask, day, snapshot_dir)

    csv_file.close()
    wall_total = time.time() - t0
    print(f"\nDone: {day:.0f} days in {wall_total:.0f}s "
          f"({wall_total/60:.1f} min)")


if __name__ == "__main__":
    main()
