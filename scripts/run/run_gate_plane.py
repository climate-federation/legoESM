"""Run the GATE_IDEAL deep-convection case on the legoESM plane CRM.

Assembles the case from SAM's ``CASES/GATE_IDEAL`` files (sounding + large-scale
forcing + surface) via :func:`build_gate_ideal_setup` and runs the SAME plane
dycore + physics stack as ``run_rcemip_plane.py`` (oceflx surface fluxes over the
fixed SST, RRTMGP/gray radiation, double-moment Morrison microphysics,
Smagorinsky), driven by the SAM large-scale forcing (subsidence + advective
tendencies + domain-mean wind nudging).

Per ``CASES/GATE_IDEAL/prm``: Coriolis OFF, interactive surface fluxes over a
fixed SST = 299.88 K, steady large-scale forcing. GATE convects over many hours,
which is infeasible to spin up on CPU — this driver validates that the assembled
case integrates stably + mass-conservingly through the full physics stack and
emits sensible column diagnostics; the multi-day equilibrium run is for HPC.

Faithfulness notes (codex iter-40):
* The large-scale forcing is NOT double-counted: SAM's ``forcing.f90`` applies
  the ``lsf`` ADVECTIVE tendencies (``tls``=dT/dt|_ls [ABSOLUTE T, added to the
  static-energy ``t``≈tabs], ``qls``=dq_v/dt|_ls — the HORIZONTAL large-scale
  advection) AND ``subsidence()`` (vertical ``−w_ls·∂φ/∂z``) as SEPARATE terms;
  the D3 operator mirrors that exactly (advective tendencies + the upwind
  subsidence operator), so they sum, not double-count. FORCING-T (iter-55):
  ``tls`` is converted T→θ via ``/exner_ref`` at the SAM seam (it is an
  absolute-T tendency, not a θ tendency).
* ``fix_mass`` anchors DRY-AIR mass (``compute_dry_mass_plane``), so the
  d(mass)~1e-15 diagnostic is DRY mass; the water tracers (q_v, hydrometeors)
  change freely via surface evaporation + precipitation — fix_mass does not
  hide moisture changes.
* PRODUCTION gaps for a quantitative profile/anomaly comparison (vs this
  stability smoke): use dx ≲ 500 m (1 km under-resolves GATE updrafts/cold
  pools), ``--radiation rrtmgp`` with the GATE latitude (8.5°N) for the SW solar
  geometry (the gray default + RCEMIP insolation are smoke-only), and the
  multi-day spin-up to convective equilibrium. Smagorinsky here is SAM-LIKE
  (Cs=0.2), not bit-faithful SAM SGS.

Example:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/run_gate_plane.py \
        --nx 8 --ny 8 --nlev 30 --H 20000 --dx 1000 --dt 5 --steps 30 \
        --microphysics morrison --radiation gray
"""

from __future__ import annotations

import argparse
import os as _os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64",
                  _os.environ.get("JAX_ENABLE_X64", "1") == "1")

# Reuse the validated RCEMIP plane machinery (physics composition + run loop
# helpers) rather than duplicating it.
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
    build_gate_ideal_setup,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402


# Default case dir: external LEGOESM_GSAM_ROOT if set, else the repo-local
# cache (scripts/data/fetch_les_forcing.py); --case-dir overrides. See
# resolve_sam_case_dir.
from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir  # noqa: E402
_GSAM_GATE = resolve_sam_case_dir("GATE_IDEAL")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", default=_GSAM_GATE,
                   help="SAM CASES/GATE_IDEAL directory (snd/lsf/sfc).")
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--nlev", type=int, default=64,
                   help="≈SAM grd level count; the stretched dz_sfc=50 m grid "
                        "needs ~64 levels for a gentle stretch (VGRID).")
    p.add_argument("--H", type=float, default=20000.0, help="model top [m]")
    p.add_argument("--dx", type=float, default=1000.0, help="dx=dy [m]")
    p.add_argument("--dt", type=float, default=2.0,
                   help="dz_sfc=50 m near-surface CFL needs dt≲2 s (VGRID).")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--microphysics", default="morrison")
    p.add_argument("--radiation", default="rrtmgp",
                   help="GATE prm: dolongwave+doshortwave (interactive RRTM) ⇒ "
                        "'rrtmgp' is the SAM-faithful default; 'gray' is a "
                        "CPU-cheap SMOKE-ONLY approximation (codex iter-56).")
    p.add_argument("--hyperdiff", type=float, default=None,
                   help="biharmonic coeff; default None auto-scales 1e8·(dx/"
                        "1000)⁴ (a fixed value over-damps at finer dx).")
    p.add_argument("--smag-cs", type=float, default=0.19,
                   help="Smagorinsky Cs; SAM default 0.19 (GATE/LBA prm).")
    p.add_argument("--radiation-interval", type=int, default=None,
                   help="recompute radiation every N steps; cached θ tendency is "
                        "forward-Euler applied each step. Default None ⇒ derive "
                        "from SAM's 90 s GATE radiation PERIOD (nrad=45 * dt=2): "
                        "N=round(90/dt) so the cadence matches SAM at any dt. "
                        "1 = every step (costlier, over-frequent vs SAM).")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--emit-profiles", action="store_true",
                   help="at run end, compute + print + save (gate_profiles.npz) "
                        "the full convective magnitudes/profiles bundle "
                        "(w'^2(z), T/q_v(z), cloud frac, condensate, CWV, precip) "
                        "for the magnitudes-and-profiles comparison vs SAM-GATE.")
    p.add_argument("--output", type=Path, default=Path("results/gate_plane"))
    return p.parse_args()


def run_gate_ideal(case_dir, *, nx=8, ny=8, nlev=64, H=20000.0, dx=1000.0,
                   dt=2.0, steps=20, microphysics="morrison", radiation="rrtmgp",
                   hyperdiff=None, smag_cs=0.19, print_every=5, verbose=True,
                   vertical_tracer_advection="van_leer", use_sam_grd=False,
                   radiation_interval=None, emit_profiles=False, seed_amp=0.1,
                   output_dir=Path("results/gate_plane"), dtype=jnp.float64):
    """Assemble + run GATE_IDEAL; return the final diagnostics dict.

    Returns ``{"finite", "rel_mass", "max_w", "state", "setup"}``. Raises if the
    case dir is missing. ``use_sam_grd=True`` uses the EXACT SAM ``grd`` vertical
    levels (266; faithful but CPU-heavy) — the grid nlev is derived from the
    grd, overriding ``nlev``/``H``; default uses the geometric stretch.

    ``hyperdiff=None`` (default) auto-scales the biharmonic coefficient as
    ``1e8·(dx/1000)⁴`` — the biharmonic CFL + the 2Δx damping rate both go as
    ``K/dx⁴``, so a FIXED coefficient over-damps at finer dx (16× too strong at
    dx=500 m). SAM uses NO explicit hyperdiffusion (its monotone advection + SGS
    do the grid-scale dissipation); legoESM's compressible dycore needs a weak
    2Δx filter, kept dx-appropriate here.
    """
    if not _os.path.isdir(case_dir):
        raise FileNotFoundError(f"GATE case dir not found: {case_dir}")
    if hyperdiff is None:
        hyperdiff = rcp.dx_aware_hyperdiff(dx)
    if radiation_interval is None:
        # SAM GATE_IDEAL radiation PERIOD = nrad*dt_SAM = 45*2 s = 90 s. Derive
        # the step count for THIS dt so the PHYSICAL cadence matches SAM at any
        # dt (codex [S2]: nrad is in steps, faithfulness is the 90 s period, not
        # the step count). At the SAM dt=2 this is 45 steps; at dt=1 it is 90.
        radiation_interval = max(1, round(90.0 / dt))
    if radiation_interval < 1:
        raise ValueError(
            f"radiation_interval must be >= 1, got {radiation_interval}")
    if use_sam_grd:
        # the grid nlev MUST match the grd's level count (else shape mismatch)
        from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_grd
        nlev = int(read_sam_grd(_os.path.join(case_dir, "grd")).z_full_bottom_up.shape[0])
    # GATE_IDEAL prm: docoriolis=.false. ⇒ no Coriolis on the grid.
    # lat0=8.5°N (GATE_IDEAL prm latitude0) so the radiation insolation sees
    # the GATE latitude, NOT the equator (RAD-8).
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             coriolis_mode="none", lat0=8.5, dtype=dtype)
    n_tracers = 11 if microphysics == "morrison" else 10
    setup = build_gate_ideal_setup(case_dir, grid, nlev=nlev, H=H,
                                   n_tracers=n_tracers, seed_amp=seed_amp,
                                   use_sam_grd=use_sam_grd, dtype=dtype)
    hc = setup.height_coord
    tm = make_flat_plane_terrain_metric(grid, hc)
    # codex iter-52 F + iter-54 D: warn on acoustic-CFL violations (horizontal
    # substep at fine dx, OR the fine vertical grid) — fail-fast for fine configs.
    rcp.cfl_guard(dt, dx, float(jnp.min(hc.dz)), label="GATE")
    if verbose:
        print(f"  SST={setup.sst:.2f} K, p_sfc={setup.p_sfc_pa:.0f} Pa, "
              f"Coriolis={'on' if setup.coriolis else 'off'}, "
              f"radiation every {radiation_interval} steps "
              f"({radiation_interval * dt:.0f} s period; SAM GATE 90 s)")
    cfg = CompressibleEulerConfig(
        hyperdiff_coeff=hyperdiff, hyperdiff_rho_coeff=hyperdiff,
        hyperdiff_w_coeff=hyperdiff,
        semi_implicit_acoustic=True, substep_horizontal_acoustic=True,
        use_coriolis=setup.coriolis,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=smag_cs, smagorinsky_prandtl=1.0,
        smagorinsky_wall_damping=False,   # SAM dosmagor: smix=grd, no wall cap
        sgs_vertical_diffusion=True,      # SGS-VERT #81: full 3D SGS like SAM
                                          # (mass-weighted vertical Smag-K flux on
                                          # u/v/θ'/tracers/w; SAM diffuse_*_z is 3D)
        # 0.4·(ACTUAL top) — hc.H, not the param H (they differ with use_sam_grd:
        # the grd top ≈30 km ≠ the geometric H). SAM nub=0.6 ⇒ base at 0.6·H.
        sponge_width=0.4 * float(hc.H),
        sponge_w_only=True,               # SAM damping.f90: damp w only
        sponge_profile_shape="sam_rational",  # SAM zzz/(1+zzz) taper
        horizontal_advection_scheme="van_leer",  # 2nd-order TVD monotone —
                                          # match RCE driver + SAM's monotone
                                          # higher-order advection; the config
                                          # default upwind1 (1st-order) is far
                                          # too diffusive for the convective
                                          # anomalies (over-smooths profiles).
        vertical_tracer_advection=vertical_tracer_advection,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    # RAD-8: GATE_IDEAL uses interactive RRTM at 8.5°N (NOT the RCEMIP-protocol
    # insolation, which is a different case). insolation="off" = the latitude
    # daily-mean at lat0=8.5° (≈428.5 W/m², perpetual-equinox; ≈437.7 at the
    # actual day 244.6). The old "rcemip" preset gave S_0·cosθ = 551.58·cos42°
    # ≈ 409.5 W/m² — the RCEMIP value, ~5% low + case-inappropriate for GATE.
    radiation_config = rcp._build_radiation_config(
        radiation, update_interval_steps=1, clouds=True,
        insolation="off", t_sfc=setup.sst)
    microphysics_config = rcp._build_microphysics_config(microphysics)
    # Split physics so RRTM radiation runs only every ``radiation_interval``
    # steps (SAM GATE_IDEAL `prm`: nrad=45) — recomputed, then HELD and applied
    # as a forward-Euler θ increment each step. This is SAM's exact radiation
    # cadence (`nrad`); every-step radiation is both 45× costlier AND less
    # SAM-faithful. Cheap physics (surface fluxes + M2005 + GATE large-scale
    # forcing) runs every dycore outer step.
    non_rad_fn, rad_fn = rcp.split_rad_from_other_physics(
        grid, hc, tm, radiation_config, microphysics_config, dt,
        surface_flux=True, T_sfc=setup.sst, p_sfc=setup.p_sfc_pa,
        ls_forcing_physics=setup.ls_forcing_physics)

    state = setup.initial_state
    mass0 = float(compute_dry_mass_plane(state, grid, hc, tm))
    if verbose:
        print("\nstep    t [s]    max|w|     min(theta')  max(theta')   "
              "max(q_v)   d(mass)")
    finite, rel = True, 0.0
    cached_rad_tend = None
    for i in range(steps):
        # SAM nrad cadence: recompute radiation every ``radiation_interval``
        # steps; hold + forward-Euler apply the cached θ tendency each step.
        if rad_fn is not None and i % radiation_interval == 0:
            cached_rad_tend = rad_fn(state, grid, hc, tm)
        state = model.step(state, dt=dt, physics_fn=non_rad_fn)
        if cached_rad_tend is not None:
            state = rcp.apply_radiation_forward_euler(state, cached_rad_tend, dt)
        finite = bool(jnp.all(jnp.isfinite(state.w.data))
                      and jnp.all(jnp.isfinite(state.theta_prime.data)))
        rel = abs(float(compute_dry_mass_plane(state, grid, hc, tm)) - mass0) \
            / abs(mass0)
        if verbose and ((i + 1) % print_every == 0 or i == 0):
            print(f"{i+1:5d}  {(i+1)*dt:7.1f}  "
                  f"{float(jnp.max(jnp.abs(state.w.data))):9.3e}  "
                  f"{float(jnp.min(state.theta_prime.data)):11.4e}  "
                  f"{float(jnp.max(state.theta_prime.data)):11.4e}  "
                  f"{float(jnp.max(state.tracers.data[..., 0])):9.3e}  "
                  f"{rel:8.2e}", flush=True)
        if not finite:
            break
    profiles = None
    if emit_profiles and finite:
        profiles = rcp.emit_crm_profiles(
            state, hc, output_dir, label="GATE",
            npz_name="gate_profiles.npz", verbose=verbose)
    return {"finite": finite, "rel_mass": rel,
            "max_w": float(jnp.max(jnp.abs(state.w.data))),
            "state": state, "setup": setup, "profiles": profiles}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"GATE_IDEAL plane: nx={args.nx} ny={args.ny} nlev={args.nlev} "
          f"H={args.H} dx={args.dx} dt={args.dt} steps={args.steps}")
    out = run_gate_ideal(
        args.case_dir, nx=args.nx, ny=args.ny, nlev=args.nlev, H=args.H,
        dx=args.dx, dt=args.dt, steps=args.steps,
        microphysics=args.microphysics, radiation=args.radiation,
        hyperdiff=args.hyperdiff, smag_cs=args.smag_cs,
        radiation_interval=args.radiation_interval,
        emit_profiles=args.emit_profiles, output_dir=args.output,
        print_every=args.print_every)
    if not out["finite"]:
        print("\nNON-FINITE STATE — aborting.")
        raise SystemExit(1)
    print(f"\nOK: GATE_IDEAL ran {args.steps} steps, stable + mass-conserving "
          f"(rel d(mass)={out['rel_mass']:.2e}). Output: {args.output}")


if __name__ == "__main__":
    main()
