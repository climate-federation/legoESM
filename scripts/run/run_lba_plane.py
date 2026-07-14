"""Run the LBA land-diurnal deep-convection case on the legoESM plane CRM.

LBA (Rondonia, Amazon) is a LAND case: convection is driven by the PRESCRIBED
diurnal surface sensible + latent heat fluxes (``SFC_FLX_FXD=.true.``), NOT
large-scale forcing (``dolargescale=.false.``). Per ``CASES/LBA/prm``:
``LAND=.true.``, ``docoriolis=.false.``, ``donudging_uv`` (τ=7200 s).

This driver assembles LBA from SAM's ``CASES/LBA`` (sounding + the H/LE diurnal
flux series) via :func:`build_lba_setup` and runs the plane dycore + physics
(radiation + double-moment Morrison + Smagorinsky + wind nudging), applying the
TIME-VARYING prescribed surface fluxes each step via
:func:`apply_prescribed_surface_fluxes_plane` (host-side flux interpolation, so
no JIT recompilation). The afternoon convective spin-up needs many hours of sim
(HPC); this driver validates stable + mass-conserving integration with the
diurnally-heated surface.

Example:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/run_lba_plane.py \
        --nx 8 --ny 8 --nlev 30 --H 20000 --dx 1000 --dt 5 --steps 30 --day0 0.2
"""

from __future__ import annotations

import argparse
import os as _os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64",
                  _os.environ.get("JAX_ENABLE_X64", "1") == "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rcemip_plane as rcp  # noqa: E402

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (  # noqa: E402
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
)
from legoesm.atmosphere.dynamics.crm.sam_case_setup import (  # noqa: E402
    build_lba_setup,
    apply_prescribed_surface_fluxes_plane,
    apply_prescribed_radiative_cooling_plane,
)
from legoesm.atmosphere.forcing.sam_case_forcing import (  # noqa: E402
    surface_at_day, interp_rad_to_levels,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402


# Default case dir: external LEGOESM_GSAM_ROOT if set, else the repo-local
# cache (scripts/data/fetch_les_forcing.py); --case-dir overrides. See
# resolve_sam_case_dir.
from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir  # noqa: E402
_GSAM_LBA = resolve_sam_case_dir("LBA")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", default=_GSAM_LBA)
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--nlev", type=int, default=64,
                   help="≈SAM grd level count (stretched dz_sfc=50 m, VGRID).")
    p.add_argument("--H", type=float, default=20000.0)
    p.add_argument("--dx", type=float, default=1000.0)
    p.add_argument("--dt", type=float, default=2.0,
                   help="dz_sfc=50 m near-surface CFL needs dt≲2 s (VGRID).")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--day0", type=float, default=0.2,
                   help="start time [days]; LBA flux peaks ≈0.22 (local noon).")
    p.add_argument("--microphysics", default="morrison")
    p.add_argument("--radiation", default="none",
                   help="LBA uses PRESCRIBED radiative cooling (doradforcing); "
                        "'none' applies the rad-file profile (default). Use "
                        "'gray'/'rrtmgp' to override with interactive radiation.")
    p.add_argument("--hyperdiff", type=float, default=None,
                   help="biharmonic coeff; default None auto-scales 1e8·(dx/"
                        "1000)⁴ (a fixed value over-damps at finer dx).")
    p.add_argument("--smag-cs", type=float, default=0.19,
                   help="Smagorinsky Cs; SAM default 0.19 (GATE/LBA prm).")
    p.add_argument("--cd-land", type=float, default=1.5e-3,
                   help="bulk surface momentum drag coefficient (land).")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--emit-profiles", action="store_true",
                   help="at run end, compute + print + save (lba_profiles.npz) "
                        "the convective magnitudes/profiles bundle (w'^2(z), "
                        "T/q_v, cloud frac, condensate, CWV, max|w|, precip) for "
                        "the magnitudes-and-profiles comparison vs SAM-LBA.")
    p.add_argument("--output", type=Path, default=Path("results/lba_plane"))
    return p.parse_args()


def run_lba(case_dir, *, nx=8, ny=8, nlev=64, H=20000.0, dx=1000.0, dt=2.0,
            steps=30, day0=0.2, microphysics="morrison", radiation="none",
            hyperdiff=None, smag_cs=0.19, cd_land=1.5e-3, print_every=5,
            verbose=True, use_sam_grd=False, emit_profiles=False,
            output_dir=Path("results/lba_plane"), dtype=jnp.float64):
    """Assemble + run LBA with prescribed diurnal surface fluxes; return diags.

    ``use_sam_grd=True`` uses the EXACT SAM ``grd`` levels (nlev derived from the
    grd; faithful but CPU-heavy); default uses the geometric stretch.
    ``hyperdiff=None`` auto-scales 1e8·(dx/1000)⁴ (dx-appropriate 2Δx filter;
    SAM has no explicit hyperdiff)."""
    if not _os.path.isdir(case_dir):
        raise FileNotFoundError(f"LBA case dir not found: {case_dir}")
    if hyperdiff is None:
        hyperdiff = rcp.dx_aware_hyperdiff(dx)
    if use_sam_grd:
        from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_grd
        nlev = int(read_sam_grd(_os.path.join(case_dir, "grd")).z_full_bottom_up.shape[0])
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             coriolis_mode="none", dtype=dtype)
    n_tracers = 11 if microphysics == "morrison" else 10
    setup = build_lba_setup(case_dir, grid, nlev=nlev, H=H,
                            n_tracers=n_tracers, use_sam_grd=use_sam_grd,
                            dtype=dtype)
    hc = setup.height_coord
    tm = make_flat_plane_terrain_metric(grid, hc)
    # codex iter-52 F + iter-54 D: acoustic-CFL guard (horizontal substep + fine
    # vertical grid).
    rcp.cfl_guard(dt, dx, float(jnp.min(hc.dz)), label="LBA")
    if verbose:
        sfc0 = surface_at_day(setup.sfc, day0)
        print(f"  p_sfc={setup.p_sfc_pa:.0f} Pa, lat={setup.latitude}, "
              f"day0={day0}: H={sfc0['shf']:.1f} LE={sfc0['lhf']:.1f} W/m²")
    cfg = CompressibleEulerConfig(
        hyperdiff_coeff=hyperdiff, hyperdiff_rho_coeff=hyperdiff,
        hyperdiff_w_coeff=hyperdiff,
        semi_implicit_acoustic=True, substep_horizontal_acoustic=True,
        use_coriolis=False, fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=smag_cs, smagorinsky_prandtl=1.0,
        smagorinsky_wall_damping=False,   # SAM dosmagor: smix=grd, no wall cap
        sgs_vertical_diffusion=True,      # SGS-VERT #81: full 3D SGS like SAM
                                          # (mass-weighted vertical Smag-K flux)
        sponge_width=0.4 * float(hc.H),   # 0.4·ACTUAL top (hc.H≠param H with grd)
        sponge_w_only=True,               # SAM damping.f90: damp w only
        sponge_profile_shape="sam_rational",  # SAM zzz/(1+zzz) taper
        horizontal_advection_scheme="van_leer",  # 2nd-order TVD monotone —
                                          # match RCE + SAM monotone higher-order;
                                          # config default upwind1 (1st-order) is
                                          # too diffusive for convective anomalies.
        vertical_tracer_advection="van_leer",
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    # LBA: doradforcing=.true. ⇒ prescribed radiative cooling, NOT interactive
    # RRTM. radiation is MUTUALLY EXCLUSIVE (no double-counting, codex iter-42 B):
    # "none" (default, faithful) ⇒ radiation_config=None (dycore radiation OFF)
    # AND the prescribed dT/dt|_rad profile is applied each step below; any other
    # value ⇒ interactive radiation AND the prescribed cooling is skipped.
    radiation_config = (None if radiation == "none" else
                        rcp._build_radiation_config(
                            radiation, update_interval_steps=1, clouds=True,
                            insolation="rcemip", t_sfc=300.0))
    microphysics_config = rcp._build_microphysics_config(microphysics)
    z_full = np.asarray(hc.z_full)
    # NO interactive surface flux (LBA prescribes them) — only (optional)
    # radiation + microphysics + the wind-nudging "forcing".
    physics_fn = rcp.make_rcemip_physics(
        grid, hc, tm, radiation_config, microphysics_config, dt,
        surface_flux=False, ls_forcing_physics=setup.ls_forcing_physics)

    state = setup.initial_state
    mass0 = float(compute_dry_mass_plane(state, grid, hc, tm))
    if verbose:
        print("\nstep    t [s]   day      H[W/m2]  LE[W/m2]  max|w|    "
              "max(q_v)   d(mass)")
    finite, rel = True, 0.0
    # The LBA afternoon flux DECLINES toward sunset, so the END state (day~0.44)
    # sits on the falling limb / cold-pool phase, NOT the convective peak (codex
    # iter-80 [HIGH]). Track the PEAK-convection state over the run (highest
    # max|w| at the sampled steps) and emit ITS profiles — the faithful
    # afternoon magnitude, not a fragile end snapshot.
    peak_w = -1.0
    peak_state = state
    for i in range(steps):
        day = day0 + (i + 1) * dt / 86400.0
        sfc_now = surface_at_day(setup.sfc, day)
        # Prescribed (time-varying) surface H/LE + interactive land momentum
        # drag (SFC_TAU_FXD=.false.), then the PRESCRIBED radiative cooling
        # (doradforcing), then the dycore step.
        state = apply_prescribed_surface_fluxes_plane(
            state, sfc_now["shf"], sfc_now["lhf"], hc, dt, cd_momentum=cd_land)
        if radiation == "none":
            dTdt_rad = interp_rad_to_levels(setup.rad, z_full, day)
            state = apply_prescribed_radiative_cooling_plane(
                state, dTdt_rad, hc, dt)
        state = model.step(state, dt=dt, physics_fn=physics_fn)
        finite = bool(jnp.all(jnp.isfinite(state.w.data))
                      and jnp.all(jnp.isfinite(state.theta_prime.data)))
        rel = abs(float(compute_dry_mass_plane(state, grid, hc, tm)) - mass0) \
            / abs(mass0)
        if (i + 1) % print_every == 0 or i == 0:
            mw = float(jnp.max(jnp.abs(state.w.data)))
            if mw > peak_w:
                peak_w = mw
                peak_state = state
            if verbose:
                print(f"{i+1:5d}  {(i+1)*dt:6.0f}  {day:6.3f}  "
                      f"{sfc_now['shf']:8.1f}  {sfc_now['lhf']:8.1f}  "
                      f"{mw:8.3e}  "
                      f"{float(jnp.max(state.tracers.data[..., 0])):9.3e}  "
                      f"{rel:8.2e}", flush=True)
        if not finite:
            break
    profiles = None
    if emit_profiles and finite:
        if verbose:
            print(f"\n[PEAK-over-afternoon: max|w|={peak_w:.3f} m/s — emitting "
                  f"the PEAK state's profiles, not the falling-limb end state]")
        profiles = rcp.emit_crm_profiles(
            peak_state, hc, output_dir, label="LBA",
            npz_name="lba_profiles.npz", verbose=verbose)
    return {"finite": finite, "rel_mass": rel, "max_w": peak_w,
            "state": state, "setup": setup, "profiles": profiles}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"LBA plane: nx={args.nx} ny={args.ny} nlev={args.nlev} H={args.H} "
          f"dx={args.dx} dt={args.dt} steps={args.steps} day0={args.day0}")
    out = run_lba(
        args.case_dir, nx=args.nx, ny=args.ny, nlev=args.nlev, H=args.H,
        dx=args.dx, dt=args.dt, steps=args.steps, day0=args.day0,
        microphysics=args.microphysics, radiation=args.radiation,
        hyperdiff=args.hyperdiff, smag_cs=args.smag_cs, cd_land=args.cd_land,
        emit_profiles=args.emit_profiles, output_dir=args.output,
        print_every=args.print_every)
    if not out["finite"]:
        print("\nNON-FINITE STATE — aborting.")
        raise SystemExit(1)
    print(f"\nOK: LBA ran {args.steps} steps, stable + mass-conserving "
          f"(rel d(mass)={out['rel_mass']:.2e}). Output: {args.output}")


if __name__ == "__main__":
    main()
