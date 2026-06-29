#!/usr/bin/env python
"""Momentum balance diagnostics for OMIP restart files.

Reconstructs the OMIP model configuration, loads a restart file, and
computes per-term momentum tendencies via ``diagnose_momentum=True``.

**Important**: the PE tendency function does NOT include Coriolis
(f × u).  Coriolis is applied in the forward-backward step function
and is never returned as a diagnostic.  The ``Vort+Cor`` field from
``MomentumTendencyDiagnostics`` is actually the *relative* vorticity
advection (ζ × F / h), not the planetary Coriolis.  This script
computes the Coriolis term explicitly from the restart velocity and
adds it to the budget for a complete picture.

Produces:
  1. Budget closure check (Σ components == Total, excluding Coriolis)
  2. Term magnitudes (RMS, all ocean vs deep interior)
  3. Geostrophic balance check (PGF vs f×v, by depth)
  4. Signed budget by latitude band and depth
  5. Plots: depth profiles, zonal means, balance pair map

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/tmp/diagnose_omip_momentum.py \\
        --run-dir results/omip_uniform_profile --restart-day 365 \\
        --bathymetry data/bathymetry/etopo_1deg.nc

    # With JRA55 forcing (adds wind stress to the diagnostic):
    JAX_ENABLE_X64=1 .venv/bin/python scripts/tmp/diagnose_omip_momentum.py \\
        --run-dir results/omip_uniform_profile --restart-day 365 \\
        --bathymetry data/bathymetry/etopo_1deg.nc \\
        --jra55-cache data/jra55_ryf_cache/jra55_do_v14_omip2_1deg_noleap.zarr

    # Load parameters from run_config.json automatically:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/tmp/diagnose_omip_momentum.py \\
        --run-dir results/omip_uniform_50yr
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())


def parse_args():
    p = argparse.ArgumentParser(description="OMIP momentum balance diagnostics")
    p.add_argument("--run-dir", type=str, required=True,
                   help="Path to OMIP results directory (contains latlon/180x360/)")
    p.add_argument("--restart-day", type=int, default=None,
                   help="Which restart day to use (default: latest)")
    p.add_argument("--resolution", type=str, default="180x360")
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--H-max", type=float, default=5500.0)
    # Physics parameters — overridden by run_config.json if present
    p.add_argument("--A-h", type=float, default=3e4)
    p.add_argument("--B-h", type=float, default=5e11)
    p.add_argument("--K-h", type=float, default=0.0)
    p.add_argument("--A-h-eq-boost", type=float, default=2.0)
    p.add_argument("--A-h-eq-sigma", type=float, default=5.0)
    p.add_argument("--A-h-floor", type=float, default=2000.0)
    p.add_argument("--C-smag-lap", type=float, default=0.15)
    p.add_argument("--C-smag", type=float, default=0.2)
    p.add_argument("--slope-foot-alpha", type=float, default=3.0)
    # Bathymetry
    p.add_argument("--bathymetry", type=str, default=None,
                   help="ETOPO file (if the run used realistic bathymetry)")
    # JRA55 forcing (optional — adds wind stress to the diagnostic)
    p.add_argument("--jra55-cache", type=str, default=None,
                   help="JRA55-do Zarr cache path. When provided, "
                        "reconstructs surface forcing for the diagnostic "
                        "so that the Wind/KPP term includes actual wind stress.")
    return p.parse_args()


def _load_config_from_json(run_dir, resolution, args):
    """Try to load run_config.json and override CLI defaults."""
    config_path = Path(run_dir) / "latlon" / resolution / "run_config.json"
    if not config_path.exists():
        config_path = Path(run_dir) / "run_config.json"
    if not config_path.exists():
        return
    print(f"  Loading config from {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    cli = cfg.get("cli_args", {})
    # Map config keys to argparse attribute names
    key_map = {
        "A_h": "A_h", "B_h": "B_h", "K_h": "K_h",
        "A_h_eq_boost": "A_h_eq_boost", "A_h_eq_sigma": "A_h_eq_sigma",
        "A_h_floor": "A_h_floor", "C_smag_lap": "C_smag_lap",
        "C_smag": "C_smag", "slope_foot_alpha": "slope_foot_alpha",
        "bathymetry": "bathymetry",
    }
    for json_key, attr in key_map.items():
        if json_key in cli and cli[json_key] is not None:
            setattr(args, attr, cli[json_key])
            print(f"    {attr} = {cli[json_key]}")


def _build_config(args):
    """Build LatLonCGridOceanConfig matching the OMIP run."""
    from legoesm.ocean.state import LatLonCGridOceanConfig

    use_bathymetry = args.bathymetry is not None
    if not use_bathymetry:
        return LatLonCGridOceanConfig.from_flat(
            A_h=args.A_h, A_h_lat_scaling=True,
            A_h_floor=args.A_h_floor,
            A_h_eq_boost=args.A_h_eq_boost,
            A_h_eq_sigma_deg=args.A_h_eq_sigma,
            K_h=args.K_h, A_v=1e-3, K_v=1e-4,
            B_h=args.B_h,
            C_smag=args.C_smag, C_smag_lap=args.C_smag_lap,
            slope_foot_alpha=args.slope_foot_alpha,
            B_h_barotropic=1e14,
            bottom_drag_r=2.5e-3,
            bottom_drag_bbl_thickness=100.0,
            bottom_drag_bg_velocity=0.1,
            n_barotropic_substeps=30,
            use_conservation_fixer=True,
            barotropic_solver="implicit_cn",
            pgf_scheme="smc03",
            maxvel_barotropic=0.0,
        )

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import (
        OceanConvectionConfig, EnhancedDiffusionConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig, VisbeckConfig, LateralMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, KPPConfig,
    )
    bathy_physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp", kpp=KPPConfig()),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        shortwave_penetration=None,
    )
    bathy_gm_redi = GMRediConfig(
        kappa_GM=800.0, kappa_Redi=800.0, S_max=0.005,
        visbeck=VisbeckConfig(
            enabled=True, alpha=0.015, kappa_min=200.0, kappa_max=2000.0,
        ),
    )
    return LatLonCGridOceanConfig.from_flat(
        A_h=args.A_h, A_h_lat_scaling=True,
        A_h_floor=args.A_h_floor,
        A_h_eq_boost=args.A_h_eq_boost,
        A_h_eq_sigma_deg=args.A_h_eq_sigma,
        K_h=args.K_h, A_v=1e-3, K_v=1e-4,
        B_h=args.B_h,
        C_smag=args.C_smag, C_smag_lap=args.C_smag_lap,
        slope_foot_alpha=args.slope_foot_alpha,
        B_h_barotropic=1e14,
        bottom_drag_r=2.5e-3,
        bottom_drag_bbl_thickness=100.0,
        bottom_drag_bg_velocity=0.1,
        n_barotropic_substeps=30,
        use_conservation_fixer=True,
        physics=bathy_physics,
        gm_redi=bathy_gm_redi,
        barotropic_solver="implicit_cn",
        pgf_scheme="smc03",
        maxvel_barotropic=0.0,
    )


def _reconstruct_jra55_forcing(args, grid, state, day):
    """Reconstruct JRA55 surface forcing for a specific day."""
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.forcing.jra55_do import jra55_to_atm_surface, load_jra55_slice
    from legoesm.coupler.config import CouplerConfig
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm import constants as _consts
    import xarray as xr

    cache_path = Path(args.jra55_cache)
    ds = xr.open_zarr(str(cache_path), decode_times=False)
    ref_year = int(ds.attrs.get("ref_year", 1958))
    ds.close()

    n_lat = grid.lat.shape[0]
    n_lon = grid.lon.shape[0]
    lat_2d = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :], (n_lat, n_lon))

    slc = load_jra55_slice(str(cache_path), float(day),
                           ref_year=ref_year, cycle=True)
    atm = jra55_to_atm_surface(slc, lat_2d, lon_2d, float(day),
                                ref_year=ref_year)

    sst_K = state.T.data[..., 0] + _consts.T_freeze
    u_o = jnp.zeros_like(sst_K)
    v_o = jnp.zeros_like(sst_K)
    tile_resp = ocean_tile_response(atm, sst_K, u_o, v_o, CouplerConfig())

    sw_net = atm.sw_down * (1.0 - tile_resp.albedo)
    q_net = (sw_net + atm.lw_down - tile_resp.lw_up
             - tile_resp.shflx - tile_resp.lhflx)

    sf = OceanSurfaceForcing(
        sw_down=atm.sw_down, q_net=q_net,
        tau_x=tile_resp.tau_x, tau_y=tile_resp.tau_y,
        freshwater=None,
    )
    print(f"  tau_x: [{float(jnp.min(sf.tau_x)):.3f}, "
          f"{float(jnp.max(sf.tau_x)):.3f}] Pa")
    return sf


def _compute_coriolis(grid, data):
    """Compute Coriolis f×v at u-faces from restart velocity.

    Returns coriolis_u interpolated to cell centers: (n_lat, n_lon, nlev).
    """
    from legoesm import constants
    lat = np.asarray(grid.lat)  # radians
    f_cell = 2 * constants.Omega * np.sin(lat)  # (n_lat,)
    f_u = f_cell[:, None]  # broadcasts to (n_lat, n_lon+1)

    u = data["u"]  # (n_lat, n_lon+1, nlev)
    v = data["v"]  # (n_lat+1, n_lon, nlev)

    # Average v to u-points (4-point)
    v_at_u_core = 0.25 * (
        v[:-1, :, :] + v[1:, :, :]
        + np.roll(v[:-1, :, :], 1, axis=1)
        + np.roll(v[1:, :, :], 1, axis=1)
    )
    v_at_u = np.concatenate(
        [v_at_u_core, v_at_u_core[:, 0:1, :]], axis=1,
    )  # (n_lat, n_lon+1, nlev)

    coriolis_u = f_u[..., None] * v_at_u  # (n_lat, n_lon+1, nlev)

    # Interpolate to cell centers
    return 0.5 * (coriolis_u[:, :-1] + coriolis_u[:, 1:])


def _uface_to_cell(arr):
    """Interpolate u-face field (n_lat, n_lon+1, ...) to cell centers."""
    return 0.5 * (arr[:, :-1] + arr[:, 1:])


def main():
    args = parse_args()

    # --- Locate restart file ---
    res_dir = Path(args.run_dir) / "latlon" / args.resolution
    if not res_dir.exists():
        res_dir = Path(args.run_dir)

    # Try loading config from run_config.json
    _load_config_from_json(args.run_dir, args.resolution, args)

    if args.restart_day is not None:
        restart_path = res_dir / f"restart_day{args.restart_day:06d}.npz"
    else:
        restarts = sorted(res_dir.glob("restart_day*.npz"))
        if not restarts:
            print(f"No restart files found in {res_dir}")
            sys.exit(1)
        restart_path = restarts[-1]

    day = int(restart_path.stem.removeprefix("restart_day"))
    yr = day / 365.0
    print(f"Using {restart_path.name} (day {day}, year {yr:.1f})")

    out_dir = res_dir / "balance_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Reconstruct model ---
    n_lat, n_lon = [int(x) for x in args.resolution.split("x")]

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import make_ocean_physics

    grid = create_latlon_grid(n_lat, n_lon)
    use_bathymetry = args.bathymetry is not None
    if use_bathymetry:
        z_coord = create_ocean_z_star(
            n_levels=args.nlev, H_max=args.H_max,
            dz_surface=20.0, dz_deep=500.0,
        )
    else:
        z_coord = create_ocean_z_star(n_levels=args.nlev, H_max=args.H_max)

    config = _build_config(args)

    # Load restart
    data = np.load(restart_path, allow_pickle=False)
    H_bathy_np = data.get("H_bathy", None)
    land_mask_np = data.get("land_mask", None)

    if H_bathy_np is not None and land_mask_np is not None:
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=args.H_max,
            H_bathy_override=jnp.asarray(H_bathy_np, dtype=jnp.float64),
            land_mask_override=jnp.asarray(land_mask_np, dtype=jnp.float64),
        )
    else:
        state = rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=args.H_max)

    for f in state._fields:
        if f not in data:
            continue
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        state = state._replace(**{f: obj.replace(
            data=jnp.asarray(data[f], dtype=obj.data.dtype))})

    print(f"State loaded: u={state.u.data.shape}, T={state.T.data.shape}")
    print(f"  max|u| = {float(jnp.max(jnp.abs(state.u.data))):.3f} m/s")
    print(f"  max|v| = {float(jnp.max(jnp.abs(state.v.data))):.3f} m/s")

    # --- Reconstruct surface forcing if JRA55 cache provided ---
    surface_forcing = None
    if args.jra55_cache is not None:
        print(f"Reconstructing JRA55 forcing for day {day}...")
        surface_forcing = _reconstruct_jra55_forcing(args, grid, state, day)

    # --- Compute momentum tendencies ---
    physics_fn = make_ocean_physics(config.physics) if config.physics else None
    print("Computing momentum tendencies (diagnose_momentum=True)...")
    tendencies, mom_diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, config,
        physics_fn=physics_fn,
        surface_forcing=surface_forcing,
        diagnose_momentum=True,
    )
    jax.block_until_ready(tendencies.du_dt.data)
    print("  Done.")

    # --- Compute actual Coriolis f×v (NOT in the tendency diagnostic) ---
    print("Computing Coriolis f×v from restart velocities...")
    coriolis_u_cell = _compute_coriolis(grid, data)

    # --- Extract numpy arrays (cell-centered for regional analysis) ---
    lat_deg = np.asarray(grid.lat) * 180 / np.pi
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    mask = np.asarray(state.land_mask.data)
    H_np = np.asarray(data["H_bathy"]) if "H_bathy" in data else None
    ocean = mask > 0.5
    area = np.asarray(grid.area)
    z_full = -np.abs(np.asarray(z_coord.z_full_ref))
    nlev = z_coord.n_levels

    # All tendency terms interpolated to cell centers
    terms = {
        "KE+PGF": _uface_to_cell(np.asarray(mom_diag.KE_PGF_u.data)),
        "Rel.vort": _uface_to_cell(np.asarray(mom_diag.vortcor_u.data)),
        "Vert adv": _uface_to_cell(np.asarray(mom_diag.vertadv_u.data)),
        "KPP+Wind": _uface_to_cell(np.asarray(mom_diag.phys_u.data)),
        "A_h": _uface_to_cell(np.asarray(mom_diag.Ah_lap_u.data)),
        "B_h": _uface_to_cell(np.asarray(mom_diag.Bh_bilap_u.data)),
        "Smag": _uface_to_cell(np.asarray(mom_diag.Cs_smag_u.data)),
        "Bot drag": _uface_to_cell(np.asarray(mom_diag.botdrag_u.data)),
        "A_v(bg)": _uface_to_cell(np.asarray(mom_diag.Av_vert_u.data)),
        "Sponge": _uface_to_cell(np.asarray(mom_diag.sponge_u.data)),
        "PE Total": _uface_to_cell(np.asarray(mom_diag.total_u.data)),
    }
    # Add Coriolis (computed from restart, not from tendency function)
    terms["Coriolis"] = coriolis_u_cell

    # Deep interior mask
    if H_np is not None:
        deep_interior = binary_erosion(
            ocean & (H_np > 3000.0), iterations=3, border_value=0)
    else:
        deep_interior = ocean
    n_deep = int(np.sum(deep_interior))
    n_ocean = int(np.sum(ocean))
    print(f"Deep interior: {n_deep} / {n_ocean} ocean cells "
          f"({100*n_deep/max(n_ocean,1):.0f}%)")

    # ====================================================================
    # Check 1: PE budget closure (excludes Coriolis — that's expected)
    # ====================================================================
    pe_terms = [k for k in terms if k not in ("PE Total", "Coriolis")]
    sum_pe = sum(terms[k] for k in pe_terms)
    residual = terms["PE Total"] - sum_pe
    print(f"\n{'='*60}")
    print("PE budget closure (Coriolis is NOT in PE tendencies)")
    print(f"{'='*60}")
    print(f"  max|residual| = {np.max(np.abs(residual)):.2e}")

    # ====================================================================
    # Check 2: Term magnitudes (deep interior)
    # ====================================================================
    print(f"\n{'='*60}")
    print("Term magnitudes (RMS, deep interior, surface)")
    print(f"{'='*60}")
    print(f"  {'Term':<12} {'Surface':>12} {'200m':>12} {'1000m':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    k_200 = min(range(nlev), key=lambda k: abs(z_full[k] + 200))
    k_1000 = min(range(nlev), key=lambda k: abs(z_full[k] + 1000))
    for name, arr in terms.items():
        rms_sfc = np.sqrt(np.mean(arr[..., 0][deep_interior]**2))
        rms_200 = np.sqrt(np.mean(arr[..., k_200][deep_interior]**2))
        rms_1000 = np.sqrt(np.mean(arr[..., k_1000][deep_interior]**2))
        print(f"  {name:<12} {rms_sfc:12.4e} {rms_200:12.4e} {rms_1000:12.4e}")

    # ====================================================================
    # Check 3: Geostrophic balance (PGF vs f×v by depth)
    # ====================================================================
    print(f"\n{'='*60}")
    print("Geostrophic balance: PGF vs Coriolis (deep interior)")
    print(f"{'='*60}")
    print(f"  {'Depth':>8} {'RMS(PGF)':>12} {'RMS(f*v)':>12} {'PGF/f*v':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for k in range(nlev):
        pgf = np.sqrt(np.mean(terms["KE+PGF"][..., k][deep_interior]**2))
        cor = np.sqrt(np.mean(terms["Coriolis"][..., k][deep_interior]**2))
        ratio = pgf / max(cor, 1e-30)
        marker = " <-- geostrophic" if 0.8 < ratio < 1.2 else ""
        print(f"  {z_full[k]:>8.0f} {pgf:>12.4e} {cor:>12.4e} {ratio:>10.2f}{marker}")

    # ====================================================================
    # Check 4: Signed budget by latitude band
    # ====================================================================
    bands = [
        ("S.Ocean 60-40S", -60, -40),
        ("Subtrop S", -40, -15),
        ("Equatorial", -15, 15),
        ("Subtrop N", 15, 40),
        ("Subpolar N", 40, 65),
    ]
    budget_names = ["KE+PGF", "Coriolis", "KPP+Wind", "A_v(bg)", "A_h",
                    "PE Total"]

    for k_level, depth_label in [(0, "Surface"), (k_200, f"~{-z_full[k_200]:.0f}m"),
                                  (k_1000, f"~{-z_full[k_1000]:.0f}m")]:
        print(f"\n{'='*60}")
        print(f"Signed budget (deep interior, {depth_label})")
        print(f"{'='*60}")
        header = f"  {'Region':<16}"
        for n in budget_names:
            header += f" {n:>10}"
        print(header)
        print(f"  {'-'*16}" + f" {'-'*10}" * len(budget_names))
        for label, lat_lo, lat_hi in bands:
            rows = (lat_deg >= lat_lo) & (lat_deg < lat_hi)
            region = deep_interior & rows[:, None]
            if not np.any(region):
                continue
            region_area = area[region]
            total_area = np.sum(region_area)
            line = f"  {label:<16}"
            for n in budget_names:
                val = np.sum(terms[n][..., k_level][region] * region_area) / total_area
                line += f" {val:>10.2e}"
            print(line)

    # ====================================================================
    # Check 5: Volume conservation
    # ====================================================================
    print(f"\n{'='*60}")
    print("Volume conservation (mean η)")
    print(f"{'='*60}")
    all_restarts = sorted(res_dir.glob("restart_day*.npz"))
    for rp in all_restarts[-10:]:  # last 10 only
        d = np.load(rp, allow_pickle=False)
        eta = d["eta"]
        mean_eta = float(np.sum(eta * mask * area) / np.sum(mask * area))
        rday = int(rp.stem.removeprefix("restart_day"))
        print(f"  day {rday:>6} (yr {rday/365:5.1f}): mean η = {mean_eta:.6e} m")

    # ====================================================================
    # PLOTS
    # ====================================================================
    plot_terms = ["KE+PGF", "Coriolis", "KPP+Wind", "A_v(bg)", "A_h",
                  "Rel.vort", "Vert adv"]
    plot_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
                   "#a65628", "#f781bf"]

    # ---- Plot 1: Depth profile RMS (deep interior) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, color in zip(plot_terms, plot_colors):
        rms_prof = [np.sqrt(np.mean(terms[name][..., k][deep_interior]**2))
                    for k in range(nlev)]
        axes[0].plot(rms_prof, z_full, label=name, color=color, linewidth=1.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("RMS tendency [m/s²]")
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_title("RMS magnitude (deep interior)")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim([z_full[-1], 0])

    # Geostrophic ratio
    pgf_prof = [np.sqrt(np.mean(terms["KE+PGF"][..., k][deep_interior]**2))
                for k in range(nlev)]
    cor_prof = [np.sqrt(np.mean(terms["Coriolis"][..., k][deep_interior]**2))
                for k in range(nlev)]
    ratios = [p / max(c, 1e-30) for p, c in zip(pgf_prof, cor_prof)]
    axes[1].plot(ratios, z_full, "k-", linewidth=2)
    axes[1].axvline(1.0, color="grey", linestyle="--", label="Geostrophy")
    axes[1].set_xlabel("|PGF| / |f×v|")
    axes[1].set_title("Geostrophic balance ratio")
    axes[1].legend(fontsize=10)
    axes[1].set_ylim([z_full[-1], 0])

    plt.suptitle(f"u-momentum budget (day {day}, deep interior)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "depth_profile.png", dpi=150)
    plt.close()
    print(f"\nSaved: {out_dir / 'depth_profile.png'}")

    # ---- Plot 2: Signed zonal-mean at 3 depths ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_i, k_level in enumerate([0, k_200, k_1000]):
        for name, color in zip(plot_terms, plot_colors):
            field = terms[name][..., k_level]
            masked = np.where(deep_interior, field, np.nan)
            area_masked = np.where(deep_interior, area, np.nan)
            zonal_sum = np.nansum(masked * area, axis=1)
            zonal_area = np.nansum(area_masked, axis=1)
            zonal_mean = np.where(zonal_area > 0,
                                  zonal_sum / zonal_area, np.nan)
            axes[ax_i].plot(lat_deg, zonal_mean, label=name, color=color,
                           linewidth=1.5)
        # Also plot PE Total
        field = terms["PE Total"][..., k_level]
        masked = np.where(deep_interior, field, np.nan)
        zonal_sum = np.nansum(masked * area, axis=1)
        zonal_mean = np.where(zonal_area > 0, zonal_sum / zonal_area, np.nan)
        axes[ax_i].plot(lat_deg, zonal_mean, "k--", linewidth=2,
                       label="PE Total")
        axes[ax_i].axhline(0, color="grey", lw=0.5)
        axes[ax_i].set_xlabel("Latitude (°)")
        axes[ax_i].set_title(f"z = {z_full[k_level]:.0f} m")
        axes[ax_i].set_xlim([-75, 75])
        axes[ax_i].legend(fontsize=6)
    axes[0].set_ylabel("u-tendency [m/s²]")
    plt.suptitle(f"Signed zonal-mean budget (deep interior, day {day})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "signed_zonal_mean.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'signed_zonal_mean.png'}")

    # ---- Plot 3: Balance pair map (surface, deep interior) ----
    pair_terms = ["KE+PGF", "Coriolis", "KPP+Wind", "A_v(bg)", "A_h"]
    pair_labels = [
        "KPP+Wind vs A_v", "KPP+Wind vs Coriolis", "PGF vs Coriolis",
        "PGF vs A_h", "A_h vs KPP+Wind", "Other",
    ]
    pair_colors_list = [
        "#4daf4a", "#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#999999",
    ]
    pair_map = np.full(ocean.shape, np.nan)
    for j in range(n_lat):
        for i in range(n_lon):
            if not deep_interior[j, i]:
                continue
            mags_pt = {k: abs(terms[k][j, i, 0]) for k in pair_terms}
            sorted_t = sorted(mags_pt.items(), key=lambda x: -x[1])
            t1, t2 = sorted_t[0][0], sorted_t[1][0]
            pair = tuple(sorted([t1, t2]))
            if pair == ("A_v(bg)", "KPP+Wind"):
                pair_map[j, i] = 0
            elif pair == ("Coriolis", "KPP+Wind"):
                pair_map[j, i] = 1
            elif pair == ("Coriolis", "KE+PGF"):
                pair_map[j, i] = 2
            elif pair == ("A_h", "KE+PGF"):
                pair_map[j, i] = 3
            elif pair == ("A_h", "KPP+Wind"):
                pair_map[j, i] = 4
            else:
                pair_map[j, i] = 5

    cmap = ListedColormap(pair_colors_list)
    bounds = np.arange(-0.5, len(pair_labels) + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(lon_deg, lat_deg, pair_map, cmap=cmap, norm=norm,
                       shading="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025,
                        ticks=range(len(pair_labels)))
    cbar.ax.set_yticklabels(pair_labels)
    ax.contour(lon_deg, lat_deg, deep_interior.astype(float), levels=[0.5],
               colors="k", linewidths=0.7, linestyles="--")
    ax.set_title(f"Dominant balance pair (surface, deep interior, day {day})")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlabel("Longitude (°)")
    plt.tight_layout()
    plt.savefig(out_dir / "balance_pairs.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'balance_pairs.png'}")

    print(f"\nAll diagnostics saved to: {out_dir}")


if __name__ == "__main__":
    main()
