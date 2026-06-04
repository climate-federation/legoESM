"""Per-term momentum-tendency instrumentation of the OMIP WOA cold-start blowup.

The realistic-WOA cold-start blows up by ~day 0.5 on BOTH the tripole and
latlon-bathy grids, independent of PGF scheme / viscosity / dt / nlev /
KE-gradient scheme (job 8106193). This script localises WHICH momentum term
grows first and WHERE, by calling ``model.tendencies_with_diagnostics`` (whose
components sum to du_dt to machine precision; see
tests/ocean/unit/test_momentum_diagnostics_closure.py) at every step and
logging the max|term| + its latitude.

Run with NO surface forcing (surface_forcing=None) so the pure cold-start
geostrophic adjustment is isolated from the CORE-II applicator. A second arm
adds the forcing to confirm it is not the trigger.

Usage (GPU, sbatch):
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python scripts/tmp/_diag_omip_per_term_tendency.py \
      --grid tripole --steps 16 --dt 600 [--forcing] [--ke-gradient-scheme ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import run_omip_core2 as roc  # noqa: E402

_MESH = "data/grids/eORCA1.2_mesh_mask.nc"
_SEC_PER_DAY = 86400.0


def _field_max_loc(field, lat_T_deg, lon_T_deg):
    """Return (max_abs, lat, lon, level) for a 3D C-grid Field (nan-aware)."""
    arr = np.asarray(field.data)
    a = np.abs(arr)
    if not np.isfinite(a).any():
        return float("nan"), float("nan"), float("nan"), -1
    flat = int(np.nanargmax(a))
    j, i, k = np.unravel_index(flat, a.shape)
    nlat, nlon = lat_T_deg.shape
    jj, ii = min(j, nlat - 1), min(i, nlon - 1)
    lat = round(float(lat_T_deg[jj, ii]), 1)
    lon = round(float(lon_T_deg[jj, ii]), 1)
    return float(a.flat[flat]), lat, lon, int(k)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="tripole", choices=["tripole", "latlon_bathy"])
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--mesh", default=_MESH)
    p.add_argument("--latlon-res", default="180x360")
    p.add_argument("--woa-t", default="data/woa18/woa18_decav_t00_01.nc")
    p.add_argument("--woa-s", default="data/woa18/woa18_decav_s00_01.nc")
    p.add_argument("--ke-gradient-scheme", default=None,
                   choices=["centered", "hollingsworth"])
    p.add_argument("--pgf-scheme", default=None, choices=["adcroft", "smc03"],
                   help="Pressure-gradient scheme. smc03 = density-Jacobian "
                        "(documented partial-cell-over-steep-topo mitigation).")
    p.add_argument("--partial-cell", action="store_true",
                   help="Use OceanPartialCellCoordinate (z-level + partial steps, "
                        "NEMO-faithful) instead of the plain sigma-like z* coord. "
                        "Activates the Adcroft/SMC03 PGF correction.")
    p.add_argument("--flat-bottom", action="store_true",
                   help="Replace bathymetry with a flat bottom (H_max) over ocean "
                        "cells -- isolates the PGF-over-steep-topography seed from "
                        "the core baroclinic response to realistic stratification.")
    p.add_argument("--forcing", action="store_true",
                   help="Apply CORE-II surface forcing (default: none, isolate dynamics).")
    p.add_argument("--uniform-strat", action="store_true",
                   help="Replace T,S with their horizontal ocean-mean profile so the "
                        "true baroclinic PGF is zero -- step-1 KE_PGF then isolates the "
                        "pure partial-cell PGF discretisation error on the real topography "
                        "(BH rest test on real geometry).")
    p.add_argument("--adaptive-implicit-vertadv", action="store_true",
                   help="Enable adaptive-implicit vertical momentum advection "
                        "(Shchepetkin 2015 / NEMO ln_zad_Aimp). NOTE: when on, the "
                        "vertadv term moves to the step level so the tendency-diag "
                        "vertadv_{u,v} read ~0; watch max|u| per step + the residual "
                        "non-vertadv driver instead.")
    p.add_argument("--barotropic-solver", default=None, choices=[None,"explicit_substep","implicit_cn"])
    p.add_argument("--barotropic-diffusion-alpha", type=float, default=None)
    p.add_argument("--n-barotropic-substeps", type=int, default=None)
    p.add_argument("--B-h", type=float, default=None,
                   help="Biharmonic momentum viscosity [m4/s] -- scale-selective "
                        "damping of grid-scale (2dx) modes. tripole OMIP default is 0 (OFF).")
    p.add_argument("--momentum-rk3", action="store_true",
                   help="Use SSP-RK3 outer momentum integrator (NEMO key_RK3 mirror).")
    p.add_argument("--top-n", type=int, default=4,
                   help="Print the N largest terms per step.")
    args = p.parse_args()

    print(f"[setup] building {args.grid} WOA cold-start "
          f"(ke_gradient_scheme={args.ke_gradient_scheme}, forcing={args.forcing}) ...")
    if args.grid == "tripole":
        grid, z_coord, model, state, H_bathy = roc.build_tripole(
            args.nlev, args.H_max, args.mesh, woa_init=True,
            woa_t=args.woa_t, woa_s=args.woa_s,
            ke_gradient_scheme=args.ke_gradient_scheme,
            pgf_scheme=args.pgf_scheme,
            partial_cell=args.partial_cell,
            flat_bottom=args.flat_bottom,
            B_h=args.B_h,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            barotropic_solver=args.barotropic_solver,
            barotropic_diffusion_alpha=args.barotropic_diffusion_alpha,
            n_barotropic_substeps=args.n_barotropic_substeps,
        )
    else:
        nlat, nlon = (int(x) for x in args.latlon_res.split("x"))
        grid, z_coord, model, state, H_bathy = roc.build_latlon_bathy(
            args.nlev, args.H_max, args.mesh, n_lat=nlat, n_lon=nlon,
            woa_init=True, woa_t=args.woa_t, woa_s=args.woa_s,
            ke_gradient_scheme=args.ke_gradient_scheme,
            pgf_scheme=args.pgf_scheme,
            partial_cell=args.partial_cell,
            flat_bottom=args.flat_bottom,
            B_h=args.B_h,
            adaptive_implicit_vertadv=(True if args.adaptive_implicit_vertadv else None),
            momentum_time_integrator=("rk3" if args.momentum_rk3 else None),
            barotropic_solver=args.barotropic_solver,
            barotropic_diffusion_alpha=args.barotropic_diffusion_alpha,
            n_barotropic_substeps=args.n_barotropic_substeps,
        )

    lat_T_deg = np.rad2deg(np.asarray(grid.lat_T))
    lon_T_deg = np.rad2deg(np.asarray(grid.lon_T))
    dt = float(args.dt)

    if args.uniform_strat:
        # Replace T,S with their ocean-area-mean profile at each level so the
        # field is HORIZONTALLY UNIFORM: the TRUE baroclinic PGF is then zero
        # everywhere, and any step-1 KE_PGF (with u=0) is PURE partial-cell PGF
        # discretisation error on the real eORCA1 topography -- the
        # Beckmann-Haidvogel "rest test" extended to the realistic geometry.
        # Distinguishes a PGF-over-steep-topo discretisation bug (seed stays
        # ~1e-2) from the real cold-start baroclinic adjustment (seed ~1e-5).
        T = np.array(state.T.data, dtype=np.float64)   # writable copy
        S = np.array(state.S.data, dtype=np.float64)
        ocean = np.asarray(state.land_mask.data) > 0.5
        nlev = T.shape[-1]
        for k in range(nlev):
            tk = T[..., k]; sk = S[..., k]
            wet = ocean & np.isfinite(tk) & (np.abs(tk) > 0)
            if wet.any():
                T[..., k] = np.where(ocean, float(tk[wet].mean()), tk)
                S[..., k] = np.where(ocean, float(sk[wet].mean()), sk)
        state = state._replace(
            T=state.T.replace(data=jnp.asarray(T)),
            S=state.S.replace(data=jnp.asarray(S)),
        )
        print("[setup] HORIZONTALLY-UNIFORM stratification (true baroclinic "
              "PGF = 0; step-1 PGF = pure discretisation error on real topo)")

    # Optional CORE-II surface forcing (built once; perpetual record 0 is fine
    # for a few-step probe — the IC imbalance, not the forcing phase, is the
    # hypothesised trigger).
    surface_forcing = None
    if args.forcing:
        from legoesm.ocean.coupler import compute_omip2_surface_forcing
        from legoesm.ocean.forcing import load_core2_nyf
        forcing = load_core2_nyf()
        app_grid = "tripole" if args.grid == "tripole" else "latlon"
        surface_forcing = compute_omip2_surface_forcing(
            state, forcing=forcing, idx_t=0, grid=grid, grid_type=app_grid,
        )
        print("[setup] CORE-II surface forcing applied (record 0).")

    from legoesm.ocean.vertical import compute_layer_thickness
    _area_T = np.asarray(grid.area_T)                 # (n_lat, n_lon)
    _ocean = np.asarray(state.land_mask.data) > 0.5

    def total_ke(st):
        # Volume-integrated kinetic energy (per unit rho_0):
        #   KE = sum_ocean 0.5*(u_c^2+v_c^2) * h * area .
        # If KE grows super-exponentially while the (wind) energy INPUT is
        # tiny, the discretisation is injecting energy (non-conservative) ->
        # the EEN/f-split + KE-pairing is implicated.  If KE stays bounded /
        # tracks the forcing work, the cascade is NOT a global energy source.
        u = np.asarray(st.u.data); v = np.asarray(st.v.data)
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            return float("nan")
        h = np.asarray(compute_layer_thickness(
            st.eta.data, st.H_bathy.data, z_coord,
            min_water_column_m=model.config.min_water_column_m))
        u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])      # (n_lat, n_lon, nlev)
        v_c = 0.5 * (v[:-1, :, :] + v[1:, :, :])
        ke = 0.5 * (u_c ** 2 + v_c ** 2) * h * _area_T[..., None] * _ocean[..., None]
        return float(np.nansum(ke))

    def diag_state(st):
        u = np.asarray(st.u.data); v = np.asarray(st.v.data)
        umax, ulat = (float(np.nanmax(np.abs(u))), float("nan")) if u.size else (0.0, 0.0)
        if np.isfinite(np.abs(u)).any():
            j = int(np.unravel_index(np.nanargmax(np.abs(u)), u.shape)[0])
            ulat = round(float(lat_T_deg[min(j, lat_T_deg.shape[0] - 1), 0]), 1)
        vmax = float(np.nanmax(np.abs(v))) if v.size else 0.0
        fin = bool(np.isfinite(u).all() and np.isfinite(v).all()
                   and np.isfinite(st.T.data).all())
        return umax, ulat, vmax, fin

    umax0, ulat0, vmax0, fin0 = diag_state(state)
    print(f"[step 0] max|u|={umax0:.3e} @ {ulat0}N  max|v|={vmax0:.3e}  "
          f"KE={total_ke(state):.4e}  finite={fin0}")

    for step in range(1, args.steps + 1):
        # Per-term tendency breakdown at the START of this step.
        tend, diag = model.tendencies_with_diagnostics(
            state, surface_forcing=surface_forcing, dt=dt,
        )
        rows = []
        for name in diag._fields:
            f = getattr(diag, name)
            if not hasattr(f, "data"):
                continue
            m, lat, lon, k = _field_max_loc(f, lat_T_deg, lon_T_deg)
            rows.append((name, m, lat, lon, k))
        rows.sort(key=lambda r: (-(r[1] if np.isfinite(r[1]) else -1)))
        top = rows[: args.top_n]
        top_str = "  ".join(
            f"{n}={m:.2e}@(lat{lat},lon{lon},k{k})" for n, m, lat, lon, k in top
        )

        # Advance one step.
        state = model.step(state, dt, surface_forcing=surface_forcing)
        state = jax.block_until_ready(state)
        umax, ulat, vmax, fin = diag_state(state)
        print(f"[step {step}] max|u|={umax:.3e}@{ulat}N max|v|={vmax:.3e} "
              f"KE={total_ke(state):.4e} finite={fin} | top terms: {top_str}")
        if not fin:
            print(f"[BLOWUP] non-finite at step {step}")
            return 1

    print("[done] no blowup within probe window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
