#!/usr/bin/env python
"""Locate the first NaN/Inf in the ETOPO blowup.

Loads the day-10 restart, steps forward with the same physics
config used in run_omip.py --bathymetry, and on each step:
  1. Checks T, S, u, v, eta, w for non-finite values
  2. When found, reports (field, k, j, i, lat, lon, H_bathy)
     for the first non-finite cell, plus context.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())


def _scan_for_nan(arr_np, name):
    """Return list of (k_or_None, j, i) for non-finite cells in arr."""
    bad = ~np.isfinite(arr_np)
    if not bad.any():
        return []
    if arr_np.ndim == 3:
        # (j, i, k) layout — convert to (k, j, i)
        idx = np.argwhere(bad)
        return [(int(k), int(j), int(i)) for j, i, k in idx]
    elif arr_np.ndim == 2:
        idx = np.argwhere(bad)
        return [(None, int(j), int(i)) for j, i in idx]
    else:
        return []


def main():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.core.field import Field
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
    from legoesm import constants as _consts

    # --- Setup: must match run_omip.py --bathymetry path ---
    n_lat, n_lon, nlev = 180, 360, 20
    H_max = 5000.0
    dt = 300.0

    print("Building grid + bathymetry…")
    grid = create_latlon_grid(n_lat, n_lon)
    z_base = create_ocean_z_star(
        n_levels=nlev, H_max=H_max, dz_surface=20.0, dz_deep=500.0,
    )
    bathy_cfg = BathymetryConfig(
        source="file", path="data/bathymetry/etopo_1deg.nc",
        H_max=H_max, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True, r_factor_max=0.2,
        north_cap_lat=80.0, south_cap_lat=-80.0,
    )
    H_bathy_init, land_mask_init = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy_init = jnp.asarray(H_bathy_init, dtype=jnp.float64)
    land_mask_init = jnp.asarray(land_mask_init, dtype=jnp.float64)
    z_coord = create_partial_cell_coordinate(z_base, H_bathy_init)

    bathy_physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="kpp", kpp=KPPConfig(),
        ),
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
            enabled=True, alpha=0.015,
            kappa_min=200.0, kappa_max=2000.0,
        ),
    )
    config = LatLonCGridOceanConfig(
        A_h=2.0e5, A_h_lat_scaling=True,
        K_h=1e3, A_v=1e-3, K_v=1e-4,
        B_h=5.0e9,
        bottom_drag_r=2.5e-3, bottom_drag_bbl_thickness=100.0,
        n_barotropic_substeps=30,
        use_conservation_fixer=True,
        physics=bathy_physics,
        gm_redi=bathy_gm_redi,
        barotropic_solver="implicit_cn",
        pgf_scheme="smc03",
    )
    model = LatLonCGridOceanModel(grid, z_coord, config)

    # --- Load day-10 restart (manual: just an npz of field arrays) ---
    restart_file = "results/etopo_1month/latlon/180x360/restart_day000010.npz"
    print(f"Loading restart: {restart_file}")
    npz = np.load(restart_file)
    print(f"  keys: {list(npz.keys())}")

    # Build a rest state with the right shape, then overwrite each field
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        H_bathy_override=H_bathy_init,
        land_mask_override=land_mask_init,
    )
    replacements = {}
    for f in state._fields:
        if f in npz.files:
            obj = getattr(state, f)
            if obj is not None and hasattr(obj, "replace"):
                replacements[f] = obj.replace(
                    data=jnp.asarray(npz[f], dtype=jnp.float64),
                )
    state = state._replace(**replacements)
    print(f"  state T dtype: {state.T.data.dtype}, shape: {state.T.data.shape}")
    print(f"  state max|T|: {float(jnp.max(jnp.abs(state.T.data))):.3f}")
    print(f"  state max|S|: {float(jnp.max(jnp.abs(state.S.data))):.3f}")
    print(f"  state max|u|: {float(jnp.max(jnp.abs(state.u.data))):.4f}")

    # --- Set up JRA55 forcing ---
    print("Setting up JRA55-do forcing…")
    from scripts.run.run_omip import _setup_jra55_forcing_state, _jra55_step
    import argparse
    args = argparse.Namespace(
        jra55_cache="data/jra55_ryf_cache/jra55_do_v14_omip2_1deg_noleap.zarr",
        jra55_cycle=True,
        jra55_co2_ppmv=400.0,
        jra55_ref_year=1900,
        jra55_no_sponge=True,
        jra55_no_sss_restoring=True,
        jra55_no_freeze_cap=True,
        T_ramp_days=10.0,
        sponge_lat_min=-60.0,
        sponge_lat_max=60.0,
        sponge_width_deg=5.0,
        sponge_tau_inner_days=5.0,
        sponge_tau_outer_days=30.0,
        sss_piston_velocity=5.0e-7,
        T_freeze_ocean=_consts.T_freeze_ocean,
        H_max=H_max,
    )
    jra55_state = _setup_jra55_forcing_state(args, grid, "latlon", z_coord=z_coord)

    # --- Step forward, detecting first NaN ---
    # day 10 = step 2880 at dt=300. Run up to step 3744 (where we expect blowup).
    # Skip ahead to step 3625 (just before the velocity jump observed earlier).
    start_step = 2880  # day 10
    fast_forward_to = 3620  # step right before the instability
    max_steps_to_run = 60  # tight window after fast-forward

    print(f"\nFast-forwarding silently from day 10 (step {start_step}) "
          f"to step {fast_forward_to}…")
    for fi in range(fast_forward_to - start_step):
        state = _jra55_step(state, start_step + fi, dt, model, jra55_state)
    state.T.data.block_until_ready()
    print(f"  reached step {fast_forward_to}, day "
          f"{fast_forward_to * dt / 86400.0:.3f}")
    print(f"  max|u|={float(jnp.max(jnp.abs(state.u.data))):.4f}, "
          f"max|eta|={float(jnp.max(jnp.abs(state.eta.data))):.4f}")

    print(f"\nNow stepping ONE step at a time, reporting location of "
          f"max|u| each step…")
    print(f"{'step':>6} {'day':>7} {'maxT':>8} {'max|u|':>9} "
          f"{'k_u':>4} {'j_u':>4} {'i_u':>4} "
          f"{'lat_u':>8} {'lon_u':>8} {'H_u':>8} "
          f"{'max|eta|':>10} {'j_eta':>5} {'i_eta':>5} "
          f"{'NaN':>5}")
    start_step = fast_forward_to

    lat = np.asarray(grid.lat) if hasattr(grid, 'lat') else None
    lon = np.asarray(grid.lon) if hasattr(grid, 'lon') else None
    H_bathy_np = np.asarray(state.H_bathy.data)
    z_full = np.asarray(z_coord.z_full_ref)

    for i in range(max_steps_to_run):
        step_idx = start_step + i
        state = _jra55_step(state, step_idx, dt, model, jra55_state)

        # Per-step finite check
        T = np.asarray(state.T.data)
        S = np.asarray(state.S.data)
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        eta = np.asarray(state.eta.data)
        w = np.asarray(state.w.data)

        check_fields = [("T", T), ("S", S), ("u", u), ("v", v),
                         ("eta", eta), ("w", w)]
        bad_field = None
        bad_locs = []
        for name, arr in check_fields:
            locs = _scan_for_nan(arr, name)
            if locs:
                bad_field = name
                bad_locs = locs
                break

        # Per-step reporting with location of max|u| and max|eta|
        day = (step_idx + 1) * dt / 86400.0
        u_finite = u[np.isfinite(u)]
        eta_finite = eta[np.isfinite(eta)]
        T_finite = T[np.isfinite(T)]
        if u_finite.size and eta_finite.size:
            # location of max|u|
            absu = np.where(np.isfinite(u), np.abs(u), 0.0)
            ku, ju, iu = np.unravel_index(np.argmax(absu), u.shape)
            # u has shape (n_lat, n_lon+1, nlev) in (j, i, k) order
            ju_idx, iu_idx, ku_idx = int(ku), int(ju), int(iu)  # placeholder
            # Actual axis order: u shape is (n_lat, n_lon+1, nlev)
            # numpy returns indices in (j, i, k) order
            j_u, i_u, k_u = ku, ju, iu
            lat_u = float(lat[j_u]) if lat is not None and j_u < len(lat) else float('nan')
            # u-points: i_u in [0, n_lon], same lon as cell-center i (with wrap)
            lon_u = float(lon[i_u % n_lon]) if lon is not None else float('nan')
            H_u = float(H_bathy_np[j_u, i_u % n_lon])

            abseta = np.where(np.isfinite(eta), np.abs(eta), 0.0)
            jeta, ieta = np.unravel_index(np.argmax(abseta), eta.shape)

            maxT = float(np.max(np.abs(T_finite))) if T_finite.size else float('nan')
            maxU = float(np.max(np.abs(u_finite)))
            maxEta = float(np.max(np.abs(eta_finite)))
            print(f"{step_idx+1:>6} {day:>7.3f} {maxT:>8.3f} {maxU:>9.4f} "
                  f"{k_u:>4d} {j_u:>4d} {i_u:>4d} "
                  f"{lat_u:>8.2f} {lon_u:>8.2f} {H_u:>8.1f} "
                  f"{maxEta:>10.5f} {int(jeta):>5d} {int(ieta):>5d} "
                  f"{(bad_field or '-'):>5}")
        else:
            print(f"{step_idx+1:>6} {day:>7.3f} (already non-finite) "
                  f"{(bad_field or '-')}")

        if bad_field is not None:
            print(f"\n  >> First NaN/Inf in field {bad_field!r} at step "
                  f"{step_idx+1}, day {day:.3f}")
            print(f"  >> Total non-finite cells: {len(bad_locs)}")
            print(f"  >> First 10 non-finite cells "
                  f"(k=level idx surface→bottom, j=lat idx, i=lon idx):")
            print(f"  {'k':>4} {'j':>4} {'i':>4} "
                  f"{'lat':>8} {'lon':>8} {'depth':>8} "
                  f"{'H_bathy':>9} {'on land?':>9}")
            for k, j, i_idx in bad_locs[:10]:
                lat_deg = float(lat[j]) if lat is not None else -999.0
                lon_deg = float(lon[i_idx]) if lon is not None else -999.0
                depth = float(z_full[k]) if k is not None else 0.0
                H = float(H_bathy_np[j, i_idx])
                on_land = bool(land_mask_init[j, i_idx] < 0.5)
                k_str = "-" if k is None else str(k)
                print(f"  {k_str:>4} {j:>4} {i_idx:>4} "
                      f"{lat_deg:>8.2f} {lon_deg:>8.2f} {depth:>8.1f} "
                      f"{H:>9.1f} {str(on_land):>9}")

            # Step back: rerun the previous step and find where it
            # was already getting "bad" but not yet NaN.
            print(f"\n  >> Pre-blowup state diagnostics (current step):")
            for name, arr in check_fields:
                # Mask non-finite, find biggest finite values
                fin = arr[np.isfinite(arr)]
                if fin.size:
                    print(f"     max(|{name}|) = {float(np.max(np.abs(fin))):.4f}, "
                          f"finite cells: {fin.size}/{arr.size}")
                else:
                    print(f"     {name}: all non-finite")

            # Find location of the worst (largest |value|) BEFORE NaN
            # appeared — this is essentially the cell that diverged.
            # We can also report which depth-level concentrations are largest.
            print(f"\n  >> Per-level max|u|, max|v|, max|w|, max|T-T_init|:")
            print(f"  {'k':>4} {'depth':>8} {'maxU':>10} {'maxV':>10} "
                  f"{'maxW':>12} {'maxT':>8}")
            for k in range(nlev):
                u_k = u[:, :, k]
                v_k = v[:, :, k]
                w_k = w[:, :, k]
                T_k = T[:, :, k]
                u_fin = u_k[np.isfinite(u_k)]
                v_fin = v_k[np.isfinite(v_k)]
                w_fin = w_k[np.isfinite(w_k)]
                T_fin = T_k[np.isfinite(T_k)]
                mU = float(np.max(np.abs(u_fin))) if u_fin.size else float('nan')
                mV = float(np.max(np.abs(v_fin))) if v_fin.size else float('nan')
                mW = float(np.max(np.abs(w_fin))) if w_fin.size else float('nan')
                mT = float(np.max(np.abs(T_fin))) if T_fin.size else float('nan')
                print(f"  {k:>4} {float(z_full[k]):>8.1f} "
                      f"{mU:>10.4f} {mV:>10.4f} {mW:>12.4e} {mT:>8.3f}")
            return 0

    print("\nNo NaN found in the test window.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
