#!/usr/bin/env python
"""Phase 4 of the realistic-geometry plan: Wolfe-Cessi-style global
overturning spinup on **real ETOPO bathymetry** with partial cells.

Builds on the existing ``run_global_overturning_implicit_spinup.py``
(idealised-continent baseline) but swaps in:

- Real ETOPO bathymetry via ``init_ocean_bathymetry`` + the lat-lon
  dispatch landed in this PR.
- ``OceanPartialCellCoordinate`` to represent the bathymetric H_bathy
  on the z* reference grid.
- ``pgf_scheme="adcroft"`` (centered + Adcroft-Campin 2004 face
  correction with h_actual integration — the canonical MITgcm/MOM6/
  NEMO ``ln_hpg_zps`` recipe for z*+partial cells).  Verified
  bit-equivalent to the previous SMC03 default on lat-lon 5° ETOPO
  to 4 sig figs at 30 days, and stable for 1-yr (max|u| converging
  to ~1.4 m/s, max|eta| ~0.27 m, no NaN).  See
  ``docs/ocean_experiments/density_jacobian_pgf_mpas.md`` §8d.
- ``momentum_advection="vector_invariant"`` which now uses the
  Arakawa-Lamb 1981 12-point triad PV flux (NEMO ``dyn_vor_een``,
  via ``pv_flux_al81_partial_cell``) — required for stability on
  partial-cell topography.
- Production-typical closures: drag = 2.5e-3, B_h = 5e9 m^4/s,
  GM/Redi K_GM = K_Redi = 800 m^2/s.

Initial condition: rest state with **centroid-aware** exponential
T(z) stratification (so the rest-state PGF residual is small) on
the ETOPO partial-cell coord.  Surface forcing is the standard
Wolfe-Cessi-style two-belt wind + cosine-latitude SST restoring
applied to the actual ETOPO land mask (not the idealised-continent
mask) — the wind/restoring profiles are functions of latitude and
work transparently on any geometry.

Spinup: 50 sim-years at 5° resolution (36×72 grid, 20 levels), dt
= 600 s.  Restarts every 5 sim-years.  The aim is qualitative:

- Does the model integrate 50 years NaN-free?
- Does an AMOC-like meridional overturning develop?
- Do gyre patterns spin up?
- Does T/S stratification stay sensible (no runaway warming, no
  cold pools below T_freeze)?

If all four are yes, the realistic-geometry stack is validated for
production cold-start spinup work.  If not, we have specific
failure modes to investigate before the fuller Phase 4-5
deliverable.

Outputs (results/ocean/global_overturning_realistic_geometry/):
  - restart_dayNNNNNN.npz at every 5-year mark + final
  - run.log
  - umax_timeseries.png

Usage:
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_realistic_geometry.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_forcings,
    create_eos_config,
    create_gm_redi_config,
    global_overturning_model_config,
)
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH


OUTPUT_DIR = Path("results/ocean/global_overturning_realistic_geometry")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")
ETOPO_URL = (
    "https://upwell.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
    "altitude%5B(-90):60:(90)%5D%5B(-180):60:(180)%5D"
)


def _ensure_etopo():
    if ETOPO_FILE.exists():
        return
    ETOPO_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ETOPO → {ETOPO_FILE}")
    subprocess.run(
        ["curl", "-fsS", "-o", str(ETOPO_FILE), ETOPO_URL], check=True,
    )


def _save_restart(state, day, output_dir):
    npz = {
        "step": int(round(day * 86400 / 600)),
        "time_days": float(day),
        "grid_type": "latlon",
    }
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        npz[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    np.savez_compressed(fname, **npz)
    print(f"    Restart saved: {fname.name}")


def _make_step_block(model, dt):
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_etopo()

    # ---- Configuration ----
    total_years = float(os.environ.get("GO_RG_YEARS", "50.0"))
    dt = 600.0
    n_steps = int(total_years * 365.0 * 86400 / dt)
    block_size = 1000
    restart_every_years = 5.0
    n_steps_per_restart = int(restart_every_years * 365.0 * 86400 / dt)

    config = GlobalOverturningConfig(
        use_gm_redi=True,
        # Production-typical closure values for partial-cell ETOPO at
        # coarse resolution (per the closure-sweep findings on the
        # realistic-geometry instability — see partial_cells_results.md).
        bottom_drag_coeff=2.5e-3,
        # Match the Phase 6 closure-sweep config that ran 30-day stable
        # on real ETOPO: A_h=1e4 (vs the GO default 2e5 which is for
        # idealised geometry; on real ETOPO with partial cells the
        # higher A_h interacts unhelpfully at coastal partial-cell
        # vertices), H_max=5000 (covers the deepest ETOPO), dz_surface
        # =20 (avoids degenerate sub-2dz_surface partial cells at the
        # surface).
        A_h=1.0e4,
        H_max=5000.0,
        dz_surface=20.0,
        kappa_GM=800.0,
        kappa_Redi=800.0,
        # Stratification config.
        T_water_init_C=20.0,
        T_deep_C=2.0,
        T_scale_depth=1000.0,
    )

    grid = create_latlon_grid(36, 72)
    z_coord_base = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )

    # ETOPO bathymetry: realistic geometry instead of the idealized
    # rectangular-block continent.
    # MEO r-factor cap: bound the bathymetry-induced PGF error per
    # Mellor-Ezer-Oey 1994.  Typical production target: r_max ≈ 0.2
    # (the Beckmann-Haidvogel 1993 stability boundary).  Set to None
    # to disable.  Realistic-geometry plan tested r=0.03 without
    # partial cells (32% volume loss); with partial cells we expect
    # r=0.2 to be sufficient because the partial-cell PGF correction
    # already handles the residual O(centroid_offset) error.
    r_factor_max = float(os.environ.get("GO_RG_R_MAX", "0.2"))
    bathy_cfg = BathymetryConfig(
        source="file",
        path=str(ETOPO_FILE),
        H_max=config.H_max,
        H_min=50.0,
        smoothing_passes=5,
        enforce_straits=True,
        fill_isolated_basins=True,
        depth_is_negative=True,
        r_factor_max=r_factor_max if r_factor_max > 0 else None,
    )
    H_bathy, ocean_mask = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy.astype(jnp.float32)
    ocean_mask = ocean_mask.astype(jnp.float32)

    # Partial-cell coordinate built from the real ETOPO H_bathy.
    z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)

    n_ocean = int(np.sum(ocean_mask))
    n_total = int(ocean_mask.size)

    # Initial state: rest with centroid-aware stratified T.
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_water_init_C=config.T_water_init_C, T_deep=config.T_water_init_C,
        S_uniform=config.S_uniform,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )
    centroid = compute_centroid_depth(
        jnp.zeros_like(H_bathy), H_bathy, z_coord,
    )
    T_per_cell = config.T_deep_C + (config.T_water_init_C - config.T_deep_C) * jnp.exp(
        -centroid / config.T_scale_depth,
    )
    T_per_cell = jnp.where(z_coord.is_active, T_per_cell, config.T_deep_C)
    T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
    )

    # Surface forcing + EOS + GM/Redi: reuse the existing
    # Wolfe-Cessi-style helpers from the global_overturning experiment.
    # Two-belt wind + cosine SST restoring are functions of latitude
    # so they work on any land mask.  Set ``GO_RG_PHYSICS=none`` to
    # disable forcing for diagnostic comparison with the rest-state
    # ETOPO 30-day test.
    if os.environ.get("GO_RG_PHYSICS", "on") == "none":
        physics = None
    else:
        physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)

    ocean_config = global_overturning_model_config(
        config, physics=physics, eos_config=eos_config, gm_redi_cfg=gm_redi_cfg,
        # Production partial-cell dycore + biharmonic on top of the shared base
        B_h=5.0e9,                           # production biharmonic
        bottom_drag_bbl_thickness=100.0,
        pgf_scheme="adcroft",   # AC + h_actual; previously "smc03"
                                # (bit-equivalent on lat-lon, simpler)
        momentum_advection="vector_invariant",   # → AL81 for partial cells
        barotropic_solver="implicit_cn",         # n_barotropic_substeps unused
    )

    print(f"=== Phase 4: Wolfe-Cessi spinup on real ETOPO bathymetry ===")
    print(f"  Output:           {OUTPUT_DIR}")
    print(f"  Grid:             36×72 (5°), 20 levels, H_max={config.H_max} m")
    print(f"  Coord:            partial cells (z* + h_partial)")
    print(f"  PGF scheme:       {ocean_config.pgf_scheme}")
    print(f"  Momentum adv:     {ocean_config.momentum_advection} (AL81)")
    print(f"  Barotropic solver: {ocean_config.barotropic_solver}")
    print(f"  Bottom drag:      r = {ocean_config.bottom_drag_r:.1e} 1/s")
    print(f"  B_h biharmonic:   {ocean_config.B_h:.1e} m⁴/s")
    print(f"  GM/Redi:          K_GM = {gm_redi_cfg.kappa_GM:.0f} m²/s, "
          f"K_Redi = {gm_redi_cfg.kappa_Redi:.0f} m²/s")
    print(f"  Wet cells:        {n_ocean}/{n_total} "
          f"({100.0*n_ocean/n_total:.1f}%)")
    print(f"  H_bathy range:    [{float(H_bathy[ocean_mask>0].min()):.0f}, "
          f"{float(H_bathy.max()):.0f}] m")
    print(f"  MEO r-factor cap: {r_factor_max if r_factor_max > 0 else 'OFF'}")
    print(f"  dt = {dt} s, n_steps = {n_steps:,} ({total_years} sim-yr)")
    print(f"  Block size: {block_size} steps  ({n_steps // block_size} blocks)")
    print(f"  Restart cadence: every {restart_every_years} yr")
    print()
    print(f"Initial state: rest with centroid-aware stratified T")
    _wet = state.land_mask.data[..., jnp.newaxis] > 0.5
    _T_min = float(jnp.min(jnp.where(_wet, state.T.data, jnp.inf)))
    _T_max = float(jnp.max(jnp.where(_wet, state.T.data, -jnp.inf)))
    print(f"  T range: [{_T_min:.2f}, {_T_max:.2f}] °C")
    print(f"  S uniform: {float(jnp.mean(state.S.data)):.2f} PSU")
    print(f"  η: {float(jnp.max(jnp.abs(state.eta.data))):.3e} m  (rest)")
    print()

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    block_fn = _make_step_block(model, dt)

    _save_restart(state, 0.0, OUTPUT_DIR)

    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size
    print(f"Starting integration ({n_blocks} blocks × {block_size} steps "
          f"+ {n_remainder} remainder)")

    t0 = time.time()
    last_print = t0

    times_yr = [0.0]
    umax_history = [float(jnp.max(jnp.abs(state.u.data)))]
    eta_max_history = [float(jnp.max(jnp.abs(state.eta.data)))]

    steps_done = 0
    last_restart_step = 0
    progress_every = max(1, n_blocks // 50)

    s = state
    blew_up = False
    for b in range(n_blocks):
        s = block_fn(s, block_size)
        steps_done += block_size

        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(s.eta.data)
            day = steps_done * dt / 86400.0
            _save_restart(s, day, OUTPUT_DIR)
            last_restart_step = steps_done

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(s.eta.data)
                yr = steps_done * dt / (365.0 * 86400.0)
                u_max = float(jnp.max(jnp.abs(s.u.data)))
                eta_max = float(jnp.max(jnp.abs(s.eta.data)))
                _wet = s.land_mask.data[..., jnp.newaxis] > 0.5
                T_min = float(jnp.min(jnp.where(_wet, s.T.data, jnp.inf)))
                T_max = float(jnp.max(jnp.where(_wet, s.T.data, -jnp.inf)))
                finite = bool(
                    jnp.all(jnp.isfinite(s.u.data))
                    and jnp.all(jnp.isfinite(s.T.data))
                )
                eta_remaining = (now - t0) / max(yr, 1e-3) * max(total_years - yr, 0.0) / 60.0
                print(
                    f"  Yr {yr:5.2f}/{total_years:.0f} | "
                    f"|η|max={eta_max:.2e} | "
                    f"T∈[{T_min:.1f},{T_max:.1f}] | "
                    f"|u|max={u_max:.3f} | "
                    f"finite={finite} | ETA {eta_remaining:.1f} min"
                )
                times_yr.append(yr)
                umax_history.append(u_max)
                eta_max_history.append(eta_max)
                if not finite:
                    print("  BLEW UP — aborting")
                    blew_up = True
                    break
                last_print = now

    # Final remainder
    if not blew_up and n_remainder > 0:
        s = block_fn(s, n_remainder)
        jax.block_until_ready(s.eta.data)

    wall = time.time() - t0
    print(f"\nSpinup complete in {wall:.0f}s ({wall/60:.1f} min)")

    final_yr = steps_done * dt / (365.0 * 86400.0)
    print(f"  Final state at sim year {final_yr:.2f}")
    if not blew_up:
        u_max = float(jnp.max(jnp.abs(s.u.data)))
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        T_min = float(jnp.min(s.T.data[s.land_mask.data[..., None] > 0]))
        T_max = float(jnp.max(s.T.data))
        print(f"  |η|max = {eta_max:.4e} m")
        print(f"  T range: [{T_min:.2f}, {T_max:.2f}] °C")
        print(f"  |u|max = {u_max:.4e} m/s")
        _save_restart(s, final_yr * 365.0, OUTPUT_DIR)

    # |u|max time series plot
    times_yr_a = np.array(times_yr)
    umax_a = np.array(umax_history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_yr_a, umax_a, "o-", color="C0", ms=3)
    ax.set_xlabel("Sim year")
    ax.set_ylabel("max|u| (m/s)")
    ax.set_title(
        f"Realistic-geometry GO spinup (5°, ETOPO partial cells, "
        f"SMC03+AL81 + drag=2.5e-3 + GM/Redi + B_h)\n"
        f"Final max|u| = {umax_a[-1]:.3f} m/s after {final_yr:.1f} yr  "
        f"({'PASS' if not blew_up else 'NaN'})"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umax_timeseries.png", dpi=130)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'umax_timeseries.png'}")

    # run.log
    with open(OUTPUT_DIR / "run.log", "w") as f:
        f.write(f"Realistic-geometry global overturning spinup\n")
        f.write(f"grid = 36x72, n_levels = {config.n_levels}, H_max = {config.H_max}\n")
        f.write(f"coord = partial cells (ETOPO bathymetry)\n")
        f.write(f"pgf_scheme = smc03\n")
        f.write(f"momentum_advection = vector_invariant (AL81)\n")
        f.write(f"barotropic_solver = implicit_cn\n")
        f.write(f"bottom_drag_r = {ocean_config.bottom_drag_r}\n")
        f.write(f"B_h = {ocean_config.B_h}\n")
        f.write(f"K_GM = {gm_redi_cfg.kappa_GM}, K_Redi = {gm_redi_cfg.kappa_Redi}\n")
        f.write(f"total_years = {total_years}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"final_year = {final_yr}\n")
        f.write(f"blew_up = {blew_up}\n")
        f.write(f"wall_time_s = {wall:.1f}\n")
        f.write(f"\nYear-by-year:\n")
        for yr, u, eta in zip(times_yr, umax_history, eta_max_history):
            f.write(f"  yr={yr:5.2f}  |u|max={u:.4f}  |eta|max={eta:.3e}\n")
    print(f"Saved {OUTPUT_DIR / 'run.log'}")


if __name__ == "__main__":
    main()
