#!/usr/bin/env python
"""Momentum balance diagnostics for idealized Wolfe-Cessi runs.

Reconstructs the model from a restart file, calls the tendency function
with diagnose_momentum=True, and produces:

1. Depth-integrated momentum balance maps (each term, surface layer)
2. Budget closure residual (should be machine-precision zero)
3. Dominant balance identification (geostrophic, Ekman, frictional)
4. Volume conservation check (mean eta drift across restarts)

Usage:
    python scripts/run/global_overturning/diagnose_momentum_balance.py [--res 5|1]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions,
    create_forcings, create_eos_config,
)
from legoesm.ocean.physics.combined import make_ocean_physics


def _load_state_from_restart(restart_path, grid, z_coord, config):
    """Load state from restart npz, reconstructing the NamedTuple."""
    d = np.load(restart_path, allow_pickle=False)
    # Create a template state to get the structure
    template = create_initial_conditions("latlon", grid, z_coord, config)
    # Replace data from restart
    replacements = {}
    for f in template._fields:
        if f in d and hasattr(getattr(template, f), "replace"):
            replacements[f] = getattr(template, f).replace(
                data=jnp.asarray(d[f]))
    return template._replace(**replacements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str, default="1",
                        choices=["1", "5"], help="Resolution: 1 or 5 degree")
    args = parser.parse_args()

    if args.res == "5":
        n_lat, n_lon = 36, 72
        run_dir = Path("results/ocean/global_overturning_idealized_5deg")
    else:
        n_lat, n_lon = 180, 360
        run_dir = Path("results/ocean/global_overturning_idealized_1deg")

    out_dir = run_dir / "balance_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find last restart
    restarts = sorted(run_dir.glob("restart_day*.npz"))
    if not restarts:
        print(f"No restarts in {run_dir}")
        return
    last = restarts[-1]
    day = int(last.stem.removeprefix("restart_day"))
    yr = day / 365.0
    print(f"Using {last.name} (year {yr:.1f})")

    # Reconstruct model
    config = GlobalOverturningConfig()
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)

    ocean_config = LatLonCGridOceanConfig.from_flat(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=config.A_h,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_config,
        barotropic_solver="implicit_cn",
    )

    state = _load_state_from_restart(last, grid, z_coord, config)
    print(f"State loaded: u={state.u.data.shape}, T={state.T.data.shape}")

    # ---- Compute momentum tendencies ----
    physics_fn = make_ocean_physics(ocean_config.physics) if ocean_config.physics else None
    print("Computing momentum tendencies (diagnose_momentum=True)...")
    tendencies, mom_diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, ocean_config,
        physics_fn=physics_fn,
        diagnose_momentum=True,
    )
    jax.block_until_ready(tendencies.du_dt.data)
    print("  Done.")

    # Extract numpy arrays for plotting
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi
    mask_u = np.asarray(state.u_mask.data)
    mask_v = np.asarray(state.v_mask.data)

    # NOTE: "Rel.vort" is the relative vorticity advection ζ×F/h, NOT
    # the planetary Coriolis f×u.  Coriolis is applied in the
    # forward-backward step function and is not captured here.
    # The "PE Total" therefore excludes Coriolis — it is the tendency
    # returned by the PE function, not the full du/dt.
    terms_u = {
        "KE+PGF": np.asarray(mom_diag.KE_PGF_u.data),
        "Rel.vort": np.asarray(mom_diag.vortcor_u.data),
        "D-term": np.asarray(mom_diag.Dterm_u.data),
        "Vert adv": np.asarray(mom_diag.vertadv_u.data),
        "KPP+Wind": np.asarray(mom_diag.phys_u.data),
        "A_h": np.asarray(mom_diag.Ah_lap_u.data),
        "B_h": np.asarray(mom_diag.Bh_bilap_u.data),
        "Smag": np.asarray(mom_diag.Cs_smag_u.data),
        "Leith": np.asarray(mom_diag.Cl_leith_u.data),
        "Bot drag": np.asarray(mom_diag.botdrag_u.data),
        "A_v(bg)": np.asarray(mom_diag.Av_vert_u.data),
        "Sponge": np.asarray(mom_diag.sponge_u.data),
        "PE Total": np.asarray(mom_diag.total_u.data),
    }
    terms_v = {
        "KE+PGF": np.asarray(mom_diag.KE_PGF_v.data),
        "Rel.vort": np.asarray(mom_diag.vortcor_v.data),
        "D-term": np.asarray(mom_diag.Dterm_v.data),
        "Vert adv": np.asarray(mom_diag.vertadv_v.data),
        "KPP+Wind": np.asarray(mom_diag.phys_v.data),
        "A_h": np.asarray(mom_diag.Ah_lap_v.data),
        "B_h": np.asarray(mom_diag.Bh_bilap_v.data),
        "Smag": np.asarray(mom_diag.Cs_smag_v.data),
        "Leith": np.asarray(mom_diag.Cl_leith_v.data),
        "Bot drag": np.asarray(mom_diag.botdrag_v.data),
        "A_v(bg)": np.asarray(mom_diag.Av_vert_v.data),
        "Sponge": np.asarray(mom_diag.sponge_v.data),
        "PE Total": np.asarray(mom_diag.total_v.data),
    }

    # ---- Check 1: Budget closure (PE terms only, excludes Coriolis) ----
    sum_components_u = sum(v for k, v in terms_u.items() if k != "PE Total")
    sum_components_v = sum(v for k, v in terms_v.items() if k != "PE Total")
    residual_u = terms_u["PE Total"] - sum_components_u
    residual_v = terms_v["PE Total"] - sum_components_v
    max_res_u = float(np.max(np.abs(residual_u)))
    max_res_v = float(np.max(np.abs(residual_v)))
    print(f"\n=== Budget closure ===")
    print(f"  max|residual_u| = {max_res_u:.2e}")
    print(f"  max|residual_v| = {max_res_v:.2e}")
    print(f"  (should be ~machine epsilon × max tendency)")

    # ---- Check 2: Term magnitudes (RMS over ocean, surface layer) ----
    print(f"\n=== Term magnitudes (RMS, surface layer, m/s²) ===")
    print(f"  {'Term':<12} {'u-mom':>12} {'v-mom':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for name in terms_u:
        rms_u = float(np.sqrt(np.mean(terms_u[name][:, :, 0]**2)))
        rms_v = float(np.sqrt(np.mean(terms_v[name][:, :, 0]**2)))
        print(f"  {name:<12} {rms_u:12.4e} {rms_v:12.4e}")

    # ---- Check 3: Geostrophic balance note ----
    # NOTE: the "Rel.vort" term is ζ×F/h (relative vorticity advection),
    # NOT the planetary Coriolis f×u.  Coriolis is applied in the
    # forward-backward step function and is not in these diagnostics.
    # To check geostrophy, compute f×v from the restart velocities
    # and compare with KE+PGF.  Use diagnose_omip_momentum.py for
    # a complete budget including Coriolis.
    print(f"\n=== Geostrophic balance note ===")
    print(f"  'Rel.vort' is relative vorticity advection (ζ×F/h),")
    print(f"  NOT Coriolis (f×u).  Coriolis is applied in the step")
    print(f"  function and is not captured in these diagnostics.")
    print(f"  Use diagnose_omip_momentum.py for full budget with Coriolis.")

    # ---- Check 4: Volume conservation ----
    print(f"\n=== Volume conservation (mean η across restarts) ===")
    mask = np.asarray(state.land_mask.data)
    area = np.asarray(grid.area)
    ocean = mask > 0.5
    for rp in restarts:
        d = np.load(rp, allow_pickle=False)
        eta = d["eta"]
        mean_eta = float(np.sum(eta * mask * area) / np.sum(mask * area))
        rday = int(rp.stem.removeprefix("restart_day"))
        print(f"  day {rday:>6} (yr {rday/365:5.1f}): mean η = {mean_eta:.6e} m")

    # ---- Plot 1: Surface momentum balance maps (u-component) ----
    plot_terms = [k for k in terms_u if k != "PE Total"]
    n_terms = len(plot_terms)
    fig, axes = plt.subplots(2, (n_terms + 1) // 2, figsize=(5 * ((n_terms + 1) // 2), 8))
    axes = axes.flatten()
    lon_u = np.concatenate([lon, [lon[0] + 360]])  # u-face longitudes
    for i, name in enumerate(plot_terms):
        field = terms_u[name][:, :, 0]  # surface layer
        field = np.where(mask_u[:, :] > 0.5, field, np.nan)
        vmax = float(np.nanpercentile(np.abs(field), 98))
        vmax = max(vmax, 1e-12)
        im = axes[i].pcolormesh(lon_u, lat, field, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
        axes[i].set_title(f"{name}")
        axes[i].set_ylabel("Lat (°)")
    for i in range(n_terms, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(f"u-momentum tendency terms (surface, yr {yr:.0f}) [m/s²]",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "momentum_balance_u_surface.png", dpi=130)
    plt.close()

    # ---- Plot 2: Surface momentum balance maps (v-component) ----
    fig, axes = plt.subplots(2, (n_terms + 1) // 2, figsize=(5 * ((n_terms + 1) // 2), 8))
    axes = axes.flatten()
    lat_v = np.linspace(-90, 90, n_lat + 1)
    for i, name in enumerate(plot_terms):
        field = terms_v[name][:, :, 0]
        field = np.where(mask_v[:, :] > 0.5, field, np.nan)
        vmax = float(np.nanpercentile(np.abs(field), 98))
        vmax = max(vmax, 1e-12)
        im = axes[i].pcolormesh(lon, lat_v, field, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
        axes[i].set_title(f"{name}")
        axes[i].set_ylabel("Lat (°)")
    for i in range(n_terms, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(f"v-momentum tendency terms (surface, yr {yr:.0f}) [m/s²]",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "momentum_balance_v_surface.png", dpi=130)
    plt.close()

    # ---- Plot 3: Dominant balance map ----
    # For each surface ocean point, which two terms are largest?
    all_terms_u = np.stack([terms_u[k][:, :, 0] for k in plot_terms], axis=-1)
    magnitudes = np.abs(all_terms_u)
    top2 = np.argsort(magnitudes, axis=-1)[..., -2:]  # indices of 2 largest
    # Color by the dominant pair
    fig, ax = plt.subplots(figsize=(12, 5))
    dominant = top2[..., -1]  # single largest term
    dominant = np.where(mask_u[:, :] > 0.5, dominant, np.nan)
    im = ax.pcolormesh(lon_u, lat, dominant, cmap="tab10",
                       vmin=-0.5, vmax=len(plot_terms) - 0.5, shading="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025,
                        ticks=range(len(plot_terms)))
    cbar.ax.set_yticklabels(plot_terms)
    ax.set_title(f"Dominant u-momentum term (surface, yr {yr:.0f})")
    ax.set_xlabel("Lon (°)"); ax.set_ylabel("Lat (°)")
    plt.tight_layout()
    plt.savefig(out_dir / "dominant_balance_u.png", dpi=130)
    plt.close()

    # ---- Plot 4: Depth profile of term magnitudes (zonal mean) ----
    z_full = -np.abs(np.asarray(z_coord.z_full_ref))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax_i, (terms, comp_label) in enumerate(
        [(terms_u, "u-momentum"), (terms_v, "v-momentum")]
    ):
        for name in plot_terms:
            # Zonal-mean RMS profile
            field = terms[name]
            rms_profile = np.sqrt(np.nanmean(field**2, axis=(0, 1)))
            axes[ax_i].plot(rms_profile, z_full, label=name)
        axes[ax_i].set_xlabel("RMS tendency (m/s²)")
        axes[ax_i].set_title(comp_label)
        axes[ax_i].set_xscale("log")
        axes[ax_i].legend(fontsize=7)
        axes[ax_i].grid(alpha=0.3)
    axes[0].set_ylabel("Depth (m)")
    plt.suptitle(f"Momentum tendency depth profiles (yr {yr:.0f})")
    plt.tight_layout()
    plt.savefig(out_dir / "tendency_depth_profiles.png", dpi=130)
    plt.close()

    print(f"\nPlots saved to {out_dir}/")
    print(f"  momentum_balance_u_surface.png")
    print(f"  momentum_balance_v_surface.png")
    print(f"  dominant_balance_u.png")
    print(f"  tendency_depth_profiles.png")


if __name__ == "__main__":
    main()
