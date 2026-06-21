"""Phase 6 of the SMC03 plan: ETOPO 30-day rest-state stress test.

Loads real ETOPO bathymetry, regrids to a lat-lon grid, applies
modest Laplacian smoothing, and integrates the rest-state for 30
days under implicit-CN barotropic + linear bottom drag, with the
SMC03 density-Jacobian PGF.

This is the headline real-bathymetry check.  Pass criterion (per
``docs/ocean/experiments/density_jacobian_pgf_plan.md`` §3 Phase 6):

- Model integrates 30 sim-days without NaN.
- ``|u|max < 50 mm/s``.
- Spurious flow not concentrated at any particular bathymetric
  feature (visual check on the saved snapshot).

Defaults: 2° resolution (90×180), 20 levels, 30 sim-days.  The
plan's spec called for 1° but 2° captures the same partial-cell
diversity and runs in ~10× less wall time.  Use ``--n-lat 180
--n-lon 360 --n-levels 30`` to run the full plan-spec config.

Usage:

    JAX_ENABLE_X64=1 python scripts/validate/realistic_geometry/run_phase6_etopo.py
    JAX_ENABLE_X64=1 python scripts/validate/realistic_geometry/run_phase6_etopo.py --pgf-scheme adcroft   # baseline
"""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH


DATA_DIR = Path("data/bathymetry")
ETOPO_FILE = DATA_DIR / "etopo_1deg.nc"
ETOPO_URL = (
    "https://upwell.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
    "altitude%5B(-90):60:(90)%5D%5B(-180):60:(180)%5D"
)


def _ensure_etopo():
    if ETOPO_FILE.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ETOPO → {ETOPO_FILE}")
    subprocess.run(
        ["curl", "-fsS", "-o", str(ETOPO_FILE), ETOPO_URL], check=True,
    )


def _make_step_block(model, dt, frozen_T=None, frozen_S=None):
    """Build a scan-based block of N model steps.

    Parameters
    ----------
    frozen_T, frozen_S : array or None
        If provided, after each ``model.step`` call the T and S fields
        are restored to these values.  Lets the dynamics see the
        density gradient but prevents tracer evolution — used to
        distinguish kinematic from thermodynamic instability modes.
    """
    do_freeze = frozen_T is not None and frozen_S is not None

    def scan_body(state, _):
        new_state = model.step(state, dt)
        if do_freeze:
            new_state = new_state._replace(
                T=new_state.T.replace(data=frozen_T),
                S=new_state.S.replace(data=frozen_S),
            )
        return new_state, None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-lat", type=int, default=90)
    parser.add_argument("--n-lon", type=int, default=180)
    parser.add_argument("--n-levels", type=int, default=20)
    parser.add_argument("--H-max", type=float, default=5000.0)
    parser.add_argument("--dz-surface", type=float, default=20.0)
    parser.add_argument("--dz-deep", type=float, default=500.0)
    parser.add_argument("--smoothing-passes", type=int, default=5)
    parser.add_argument("--H-min", type=float, default=50.0,
                        help="Minimum ocean depth [m]; cells shallower "
                        "than this become land.  Default 50 m: avoids "
                        "degenerate partial cells thinner than ~2x "
                        "dz_surface that destabilise the implicit-CN "
                        "barotropic on steep continental shelves.")
    parser.add_argument("--bottom-drag-r", type=float, default=1.0e-3)
    parser.add_argument("--bbl-thickness", type=float, default=100.0)
    parser.add_argument("--momentum-advection", type=str,
                        default="vector_invariant",
                        choices=["vector_invariant", "weno5", "weno7"],
                        help="Momentum advection scheme.  Vector-invariant "
                        "(default) uses q·flux with h_vtx — the form that "
                        "exercises any PV-thickness inconsistency at "
                        "step vertices.  WENO5/7 uses flux-form WENO "
                        "upwind reconstruction — bypasses the q "
                        "stencil and isolates whether vector-invariant "
                        "at lateral steps is the bug.")
    parser.add_argument("--A-h", type=float, default=1.0e4,
                        help="Horizontal Laplacian viscosity [m²/s].  "
                        "Default 1e4 (LatLonCGridOceanConfig default); "
                        "ETOPO at 2° benefits from 5e4 to suppress "
                        "coastal computational modes.")
    parser.add_argument("--B-h", type=float, default=0.0,
                        help="Biharmonic momentum viscosity [m⁴/s].  "
                        "Production-typical at 3°: 5e9.  Targets the "
                        "topographic computational mode in stratified "
                        "flow (frozen-T diagnostic confirmed).")
    parser.add_argument("--gm-redi", action="store_true",
                        help="Enable GM/Redi isopycnal mixing "
                        "(Gent-McWilliams 1990 + Redi 1982).  "
                        "Parameterizes mesoscale eddy effects on the "
                        "tracer field at coarse resolution.  Reduces "
                        "horizontal density gradients via isopycnal "
                        "slumping — physical, not stability tuning.")
    parser.add_argument("--K-GM", type=float, default=800.0,
                        help="GM bolus diffusivity [m²/s] (when "
                        "--gm-redi).  Production-typical: 800.")
    parser.add_argument("--K-Redi", type=float, default=800.0,
                        help="Redi isopycnal diffusivity [m²/s] (when "
                        "--gm-redi).  Production-typical: 800; matching "
                        "K_GM cancels horizontal off-diagonal terms.")
    parser.add_argument("--S-max", type=float, default=0.01,
                        help="GM/Redi slope-clipping threshold "
                        "(Danabasoglu-McWilliams).  Standard: 0.01.")
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--record-every-days", type=float, default=1.0)
    parser.add_argument("--pgf-scheme", type=str, default="smc03",
                        choices=["adcroft", "smc03"])
    parser.add_argument("--coord", type=str, default="partial",
                        choices=["zstar", "partial"])
    # Diagnostic experiments E1/E2 for the ETOPO 30-day instability.
    # E1: --homogeneous-ts removes stratification (uniform T, S);
    # tests whether the bug needs stratification × coastline coupling.
    # E2: --flat-bottom replaces ETOPO with a global flat-bottom basin
    # at H=H_max; tests whether the bug needs irregular bathymetry /
    # coastlines at all.  Combine the two to bisect.
    parser.add_argument("--homogeneous-ts", action="store_true",
                        help="E1: override the exponential thermocline "
                        "with uniform T=10°C, S=35 PSU.  Removes "
                        "stratification — tests if the instability "
                        "needs stratification × coastline coupling.")
    parser.add_argument("--flat-bottom", action="store_true",
                        help="E2: skip ETOPO and use a global flat-bottom "
                        "basin at H=H_max with no land mask.  Removes "
                        "coastlines and topography — tests if the "
                        "instability needs irregular geometry at all.")
    parser.add_argument("--meridional-T-gradient", action="store_true",
                        help="E3 (use with --flat-bottom): override the "
                        "uniform-by-depth T(z) with a sin²(lat)-modulated "
                        "T(y, z) so the rest state has a thermal-wind "
                        "shear (warm equator, cold poles).  Distinguishes "
                        "'any horizontal gradient → instability' (E3 fails) "
                        "from 'gradient × topography' (E3 stable, prior "
                        "ETOPO run still fails).")
    parser.add_argument("--T-eq-surface", type=float, default=25.0,
                        help="E3: equatorial surface temperature [°C].")
    parser.add_argument("--T-pol-surface", type=float, default=2.0,
                        help="E3: polar surface temperature [°C].")
    parser.add_argument("--zonly-T-init", action="store_true",
                        help="E4: replace centroid-aware T with "
                        "T(k) = exp(-|z_full_ref[k]|/scale_depth), so "
                        "every column's cell-mean T at level k is "
                        "identical (no horizontal density gradient at "
                        "partial-bottom faces).  Tests whether the "
                        "centroid-aware T-init is what seeds the runaway.")
    parser.add_argument("--linear-T-z", action="store_true",
                        help="Diagnostic: replace exponential T(z) with "
                        "a linear T(z) = T_surf - (T_surf-T_deep)*z/H_max. "
                        "Strips d²ρ/dz² (curvature) from the in-cell "
                        "integral.  Tests the dycore-expert's hypothesis "
                        "that the bottom-cell σ residual scales with "
                        "stratification curvature × partial-thickness "
                        "mismatch.  Used with --frozen-ts.")
    parser.add_argument("--frozen-ts", action="store_true",
                        help="Diagnostic: restore T,S to their initial "
                        "values after every step.  PGF still sees the "
                        "horizontal density gradient but baroclinic "
                        "instability cannot release APE.  Distinguishes "
                        "kinematic (PGF-driven) from thermodynamic "
                        "(advection-of-T-driven) failure modes.")
    args = parser.parse_args()

    _ensure_etopo()

    pgf_tag = "" if args.pgf_scheme == "adcroft" else f"_pgf-{args.pgf_scheme}"
    diag_tag = ""
    if args.flat_bottom:
        diag_tag += "_E2flatbottom"
    if args.homogeneous_ts:
        diag_tag += "_E1homog"
    if args.meridional_T_gradient:
        diag_tag += "_E3meridT"
    if args.zonly_T_init:
        diag_tag += "_E4zonlyT"
    if args.linear_T_z:
        diag_tag += "_linearTz"
    if args.frozen_ts:
        diag_tag += "_frozenTS"
    if args.B_h > 0:
        diag_tag += f"_Bh{args.B_h:.0e}"
    if args.gm_redi:
        diag_tag += f"_GM{int(args.K_GM)}Redi{int(args.K_Redi)}"
    if args.momentum_advection != "vector_invariant":
        diag_tag += f"_mom-{args.momentum_advection}"
    output_dir = Path(
        f"results/realistic_geometry_validation/"
        f"phase6_etopo_{args.coord}_{args.n_lat}x{args.n_lon}_{args.n_levels}lev"
        f"_smooth{args.smoothing_passes}_drag{args.bottom_drag_r:.0e}{pgf_tag}{diag_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_latlon_grid(args.n_lat, args.n_lon)
    z_coord_base = create_ocean_z_star(
        n_levels=args.n_levels, H_max=args.H_max,
        dz_surface=args.dz_surface, dz_deep=args.dz_deep,
    )

    if args.flat_bottom:
        # E2: global flat-bottom basin at H_max — no land, no
        # topography.  Tests whether the ETOPO instability needs
        # irregular geometry to manifest.
        H_bathy = jnp.full(
            (args.n_lat, args.n_lon), args.H_max, dtype=jnp.float32,
        )
        ocean_mask = jnp.ones(
            (args.n_lat, args.n_lon), dtype=jnp.float32,
        )
    else:
        bathy_cfg = BathymetryConfig(
            source="file",
            path=str(ETOPO_FILE),
            H_max=args.H_max,
            H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            enforce_straits=True,
            strait_width_factor=1.0,
            fill_isolated_basins=True,
            depth_is_negative=True,
        )
        H_bathy, ocean_mask = init_ocean_bathymetry(grid, bathy_cfg)
        # Match the rest_state working dtype (float32 by default) to avoid
        # dtype-mismatch carry errors inside the scan-based step block.
        H_bathy = H_bathy.astype(jnp.float32)
        ocean_mask = ocean_mask.astype(jnp.float32)

    if args.coord == "partial":
        z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)
    else:
        z_coord = z_coord_base

    n_ocean = int(np.sum(ocean_mask))
    n_total = int(ocean_mask.size)
    print(f"=== Phase 6 ETOPO 30-day rest-state ===")
    print(f"  Grid:            {args.n_lat}x{args.n_lon} "
          f"(~{180.0/args.n_lat:.2f}° lat, {360.0/args.n_lon:.2f}° lon)")
    print(f"  Levels:          {args.n_levels}, H_max={args.H_max} m, "
          f"dz=[{args.dz_surface}, {args.dz_deep}]")
    print(f"  Coord:           {args.coord}")
    print(f"  PGF scheme:      {args.pgf_scheme}")
    print(f"  Smoothing:       {args.smoothing_passes} Laplacian passes")
    print(f"  Bottom drag:     r={args.bottom_drag_r:.1e} 1/s, "
          f"BBL={args.bbl_thickness} m")
    print(f"  Momentum visc:   A_h={args.A_h:.1e} m²/s, "
          f"B_h={args.B_h:.1e} m⁴/s")
    if args.gm_redi:
        print(f"  GM/Redi:         K_GM={args.K_GM:.0f}, "
              f"K_Redi={args.K_Redi:.0f}, S_max={args.S_max:.3f}")
    else:
        print(f"  GM/Redi:         off")
    print(f"  Wet cells:       {n_ocean}/{n_total} "
          f"({100.0*n_ocean/n_total:.1f}%)")
    print(f"  H_bathy range:   [{float(H_bathy[ocean_mask>0].min()):.0f}, "
          f"{float(H_bathy.max()):.0f}] m")
    if args.flat_bottom:
        print(f"  E2: flat-bottom mode (no ETOPO, no land mask)")
    if args.homogeneous_ts:
        print(f"  E1: homogeneous T=10°C, S=35 PSU (no stratification)")
    if args.meridional_T_gradient:
        print(f"  E3: meridional T gradient T_eq={args.T_eq_surface}°C, "
              f"T_pol={args.T_pol_surface}°C")
    if args.zonly_T_init:
        print(f"  E4: z-only T-init (T(k) from z_full_ref, no centroid offset)")
    if args.frozen_ts:
        print(f"  Diag: frozen T,S (restored to initial after each step)")
    print(f"  dt = {args.dt} s, total = {args.days} sim-days")
    print(f"  Output: {output_dir}")
    print()

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )

    if args.meridional_T_gradient:
        # E3: T(y, z) = T_deep + (T_surf(lat) - T_deep) * exp(-z/scale)
        # with T_surf(lat) = T_eq + (T_pol - T_eq) * sin²(lat).
        # Provides an APE-bearing thermal-wind shear without any
        # topography — distinguishes "horizontal gradient" causes
        # from "topography × gradient" causes.
        if args.coord == "partial":
            centroid = compute_centroid_depth(
                jnp.zeros_like(H_bathy), H_bathy, z_coord,
            )
            is_active = z_coord.is_active
        else:
            # Pure z*: centroid_depth = |z_full_ref| · J = |z_full_ref|
            # at η=0 with H=H_max, so the abs(z_full_ref) field
            # broadcast to (n_lat, n_lon, nlev) suffices.
            centroid = jnp.broadcast_to(
                jnp.abs(z_coord.z_full_ref),
                (args.n_lat, args.n_lon, args.n_levels),
            )
            is_active = jnp.ones_like(centroid, dtype=jnp.bool_)
        # sin²(lat).  ``grid.lat`` is radians, shape (n_lat,).
        sin2_lat = jnp.sin(grid.lat) ** 2                 # (n_lat,)
        T_deep = 2.0
        T_surf = (
            args.T_eq_surface
            + (args.T_pol_surface - args.T_eq_surface) * sin2_lat
        )                                                  # (n_lat,)
        T_surf_3d = T_surf[:, jnp.newaxis, jnp.newaxis]    # (n_lat, 1, 1)
        T_per_cell = T_deep + (T_surf_3d - T_deep) * jnp.exp(
            -centroid / _SCALE_DEPTH,
        )
        T_per_cell = jnp.where(is_active, T_per_cell, T_deep)
        T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(
            T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
        )
    elif args.homogeneous_ts:
        # E1: override the exponential thermocline with a uniform
        # T = 10°C, S = 35 PSU.  The ``rest_state`` ctor already set
        # S = 35 uniformly; we only need to flatten T.
        T_uniform = jnp.full_like(state.T.data, 10.0)
        T_uniform = T_uniform * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(T=state.T.replace(data=T_uniform))
    elif args.linear_T_z:
        # Diagnostic: T(z) linear in depth.
        # T(z) = T_surf + (T_deep - T_surf) * z / H_max with z positive
        # downward, so T_surf at z=0, T_deep at z=H_max.  Centroid-
        # aware (still uses each cell's actual centroid).
        T_deep = 2.0
        T_surf = 20.0
        if args.coord == "partial":
            centroid = compute_centroid_depth(
                jnp.zeros_like(H_bathy), H_bathy, z_coord,
            )
            is_active = z_coord.is_active
        else:
            centroid = jnp.broadcast_to(
                jnp.abs(z_coord.z_full_ref),
                (args.n_lat, args.n_lon, args.n_levels),
            )
            is_active = jnp.ones_like(centroid, dtype=jnp.bool_)
        T_per_cell = T_surf + (T_deep - T_surf) * centroid / args.H_max
        T_per_cell = jnp.where(is_active, T_per_cell, T_deep)
        T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(
            T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
        )
    elif args.zonly_T_init:
        # E4: depth-only T(k) using z_full_ref (NOT centroid-aware).
        # Every column's cell-mean T at level k is identical, so the
        # horizontal density gradient at partial-bottom faces vanishes
        # (under linear EOS at least; with a pressure-dependent EOS a
        # tiny residual remains from compressibility × Δh_partial,
        # negligible for diagnostic purposes).  Trade-off: at partial-
        # bottom cells the cell-mean T no longer matches T(actual
        # centroid), so the rest-state PGF residual is slightly
        # larger than centroid-aware — a small spurious initial flow
        # seeds, but the seed is *not* a real horizontal gradient.
        T_deep = 2.0
        T_surf = 20.0
        z_abs = jnp.abs(z_coord_base.z_full_ref).astype(state.T.data.dtype)
        T_1d = T_deep + (T_surf - T_deep) * jnp.exp(-z_abs / _SCALE_DEPTH)
        T_per_cell = jnp.broadcast_to(
            T_1d[jnp.newaxis, jnp.newaxis, :],
            (args.n_lat, args.n_lon, args.n_levels),
        )
        if args.coord == "partial":
            T_per_cell = jnp.where(z_coord.is_active, T_per_cell, T_deep)
        T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(
            T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
        )
    elif args.coord == "partial":
        # Centroid-aware T initialization to keep the rest-state PGF clean.
        centroid = compute_centroid_depth(
            jnp.zeros_like(H_bathy), H_bathy, z_coord,
        )
        T_per_cell = 2.0 + (20.0 - 2.0) * jnp.exp(-centroid / _SCALE_DEPTH)
        T_per_cell = jnp.where(z_coord.is_active, T_per_cell, 2.0)
        T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(
            T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
        )

    gm_redi_cfg = None
    if args.gm_redi:
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        gm_redi_cfg = GMRediConfig(
            kappa_GM=args.K_GM,
            kappa_Redi=args.K_Redi,
            S_max=args.S_max,
        )
    cfg = LatLonCGridOceanConfig(
        barotropic_solver="implicit_cn",
        physics=None,
        A_h=args.A_h,
        B_h=args.B_h,
        bottom_drag_r=args.bottom_drag_r,
        bottom_drag_bbl_thickness=args.bbl_thickness,
        pgf_scheme=args.pgf_scheme,
        gm_redi=gm_redi_cfg,
        momentum_advection=args.momentum_advection,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    if args.frozen_ts:
        block_fn = _make_step_block(
            model, args.dt,
            frozen_T=state.T.data, frozen_S=state.S.data,
        )
    else:
        block_fn = _make_step_block(model, args.dt)

    n_total_steps = int(args.days * 86400.0 / args.dt)
    steps_per_record = int(args.record_every_days * 86400.0 / args.dt)
    n_records = max(1, int(args.days / args.record_every_days))

    print(f"Starting integration ({n_total_steps} steps, "
          f"{steps_per_record} per record)")
    t0 = time.time()

    times_days = [0.0]
    umax_history = [float(jnp.max(jnp.abs(state.u.data)))]
    eta_max_history = [float(jnp.max(jnp.abs(state.eta.data)))]

    s = state
    blew_up = False
    for record_idx in range(n_records):
        s = block_fn(s, steps_per_record)
        jax.block_until_ready(s.eta.data)
        day = (record_idx + 1) * args.record_every_days
        u_max = float(jnp.max(jnp.abs(s.u.data)))
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        T_max = float(jnp.max(s.T.data))
        T_min = float(jnp.min(s.T.data))
        times_days.append(day)
        umax_history.append(u_max)
        eta_max_history.append(eta_max)
        elapsed = time.time() - t0
        eta_remaining = elapsed / day * (args.days - day) if day > 0 else 0
        finite = bool(jnp.all(jnp.isfinite(s.u.data)) and
                      jnp.all(jnp.isfinite(s.T.data)))
        print(f"  Day {day:5.1f}/{args.days:.0f}  "
              f"|u|max={u_max*1000:8.3f} mm/s  "
              f"|eta|max={eta_max:.3e} m  "
              f"T=[{T_min:.2f},{T_max:.2f}]  "
              f"finite={finite}  ETA {eta_remaining/60:.1f} min")
        if not finite:
            print("  BLEW UP — aborting integration")
            blew_up = True
            break

    wall = time.time() - t0
    print(f"\nIntegration complete in {wall:.0f}s ({wall/60:.1f} min)")

    times_days = np.array(times_days)
    umax_history = np.array(umax_history)
    eta_max_history = np.array(eta_max_history)
    final_umax_mm = umax_history[-1] * 1000

    pass_threshold_mm = 50.0
    finite_final = (not blew_up) and bool(jnp.all(jnp.isfinite(s.u.data)))
    passed = finite_final and (final_umax_mm <= pass_threshold_mm)
    status = "PASS" if passed else "FAIL"
    print(f"\n--- Phase 6 result ---")
    print(f"Coord:         {args.coord}")
    print(f"PGF scheme:    {args.pgf_scheme}")
    print(f"Final max|u|:  {final_umax_mm:.3f} mm/s")
    print(f"Threshold:     {pass_threshold_mm:.1f} mm/s")
    print(f"Status:        {status}")

    # |u|max time series
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_days, umax_history * 1000, "o-", color="C0", ms=4)
    ax.axhline(pass_threshold_mm, color="C3", ls="--",
               label=f"Pass threshold ({pass_threshold_mm:.0f} mm/s)")
    ax.set_xlabel("Sim day")
    ax.set_ylabel("max|u| (mm/s)")
    ax.set_yscale("log")
    ax.set_title(
        f"Phase 6 ETOPO {args.n_lat}×{args.n_lon} "
        f"(coord={args.coord}, pgf={args.pgf_scheme}, smooth={args.smoothing_passes})\n"
        f"Final max|u|={final_umax_mm:.3f} mm/s — {status}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "umax_timeseries.png", dpi=130)
    plt.close()
    print(f"Saved {output_dir / 'umax_timeseries.png'}")

    # Surface speed snapshot
    if finite_final:
        u_c = 0.5 * (np.asarray(s.u.data[:, :-1, 0]) + np.asarray(s.u.data[:, 1:, 0]))
        v_c = 0.5 * (np.asarray(s.v.data[:-1, :, 0]) + np.asarray(s.v.data[1:, :, 0]))
        speed_sfc = np.sqrt(u_c ** 2 + v_c ** 2)
        lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
        lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
        speed_plot = np.where(np.isfinite(speed_sfc), speed_sfc, 0.0)
        # Mask land cells for clarity.
        land = (np.asarray(ocean_mask) < 0.5)
        speed_plot = np.ma.masked_where(land, speed_plot)
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.pcolormesh(lon_deg, lat_deg, speed_plot * 1000,
                           cmap="hot_r", shading="auto")
        plt.colorbar(im, ax=ax, label="Surface |U| (mm/s)", fraction=0.025)
        H_np = np.asarray(H_bathy)
        H_plot = np.where(land, np.nan, H_np)
        cs = ax.contour(lon_deg, lat_deg, H_plot / 1000.0,
                        levels=[1, 2, 3, 4], colors="white",
                        linewidths=0.6, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%g km")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(
            f"Phase 6 ETOPO surface |U| at day {args.days:.0f}  "
            f"(coord={args.coord}, pgf={args.pgf_scheme})\n"
            f"White contours: bathymetry (km depth)"
        )
        plt.tight_layout()
        plt.savefig(output_dir / "velocity_snapshot.png", dpi=130)
        plt.close()
        print(f"Saved {output_dir / 'velocity_snapshot.png'}")

    log_path = output_dir / "run.log"
    with open(log_path, "w") as f:
        f.write(f"Phase 6 ETOPO 30-day rest-state\n")
        f.write(f"coord = {args.coord}\n")
        f.write(f"pgf_scheme = {args.pgf_scheme}\n")
        f.write(f"grid = {args.n_lat}x{args.n_lon}\n")
        f.write(f"n_levels = {args.n_levels}\n")
        f.write(f"smoothing_passes = {args.smoothing_passes}\n")
        f.write(f"bottom_drag_r = {args.bottom_drag_r}\n")
        f.write(f"bbl_thickness = {args.bbl_thickness}\n")
        f.write(f"days = {args.days}\n")
        f.write(f"final_max_u_mm_per_s = {final_umax_mm:.6f}\n")
        f.write(f"pass_threshold_mm_per_s = {pass_threshold_mm:.1f}\n")
        f.write(f"status = {status}\n")
        f.write(f"finite_final = {finite_final}\n")
        f.write(f"wall_time_s = {wall:.1f}\n")
        f.write(f"\nDay-by-day:\n")
        for d, u, e in zip(times_days, umax_history, eta_max_history):
            f.write(f"  day={d:5.1f}  |u|max={u*1000:.4f} mm/s  "
                    f"|eta|max={e:.3e} m\n")
    print(f"Saved {log_path}")


if __name__ == "__main__":
    main()
