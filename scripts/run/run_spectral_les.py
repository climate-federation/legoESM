"""Neutral ABL on the pseudo-spectral INCOMPRESSIBLE plane-LES core.

Pressure-gradient-driven neutral channel (constant body force ``f_x = u_*²/Lz``
⇒ a target surface friction velocity and a Monin–Obukhov log-law equilibrium in
a few eddy turnovers). This is the clean validation case against the jax-alfa
oracle's physical target — the log law, ``φ_m→1`` and the resolved-variance
similarity that any faithful ABL LES must reproduce. Unlike the compressible
plane dycore, this core SUSTAINS the resolved turbulence
(``les_plane_turbulence_notes.md``).

Saves ``final_profiles.npz`` (mean U, resolved uu/vv/ww/tke, uw/vw flux, u_*) in
the SAME format as ``run_les_plane.py`` so
``scripts/validate/validate_les_vs_oracle.py`` compares it to MOST directly.

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_spectral_les.py \\
       --nx 96 --ny 96 --nz 96 --Lx 2000 --Ly 2000 --Lz 1000 \\
       --ustar 0.45 --dt 0.4 --hours 1.5 --f32
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402

if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.timestepping.split_explicit import select_dt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import les_record  # noqa: E402

_KAPPA = constants.kappa_von_karman


def build(args, dtype):
    # --sgs-model (smagorinsky|vreman) selects the STATIC sub-closure and is only
    # consulted when the dynamic LASD closure is OFF (i.e. --static). Now that
    # dynamic is the default, a non-default static sub-closure requested WITHOUT
    # --static would be silently ignored (LASD runs instead). Make that an
    # explicit error rather than a silent no-op (codex review; dispatch-hardening).
    if not args.static and args.sgs_model != "smagorinsky":
        raise ValueError(
            f"--sgs-model {args.sgs_model!r} is a static SGS sub-closure but the "
            "dynamic LASD closure is active (the default); pass --static to use it.")
    cfg = sl.SpectralLESConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        z0=args.z0, c_s=args.cs, wall_damping=True, dealias=True,
        smagorinsky_dynamic=not args.static, nu_floor=args.nu_floor,
        sgs_model=args.sgs_model, time_scheme=args.time_scheme)
    g = sl.make_grid(cfg, dtype=dtype)
    z = g.z_c
    key = jax.random.PRNGKey(0)
    if args.ekman:
        # Neutral ROTATING Ekman layer: uniform geostrophic mean wind (Ug, 0)
        # spun up under Coriolis f_cor (sl.rhs adds f·(u-u_geo)). The turbulent
        # Ekman spiral + cross-isobar veering + ~0.3·u*/f depth emerge from that
        # balance. Reference: laminar Ekman spiral + Coleman (1990) / Andren
        # (1994) neutral truly-rotating ABL-LES intercomparison.
        u_tar = jnp.full_like(z, args.Ug)
        amp = args.ic_amp * args.Ug                # IC perturbation (fraction of Ug)
    else:
        # Log-law mean IC (shear from t=0) + divergence-free small perturbations.
        u_tar = args.ustar / _KAPPA * jnp.log(jnp.clip(z, args.z0, None) / args.z0)
        amp = args.ic_amp * args.ustar / _KAPPA   # IC perturbation (fraction of bulk)
    u = jnp.broadcast_to(u_tar, (args.ny, args.nx, args.nz)).astype(dtype) + (
        amp * jax.random.normal(key, (args.ny, args.nx, args.nz), dtype=dtype))
    v = amp * jax.random.normal(jax.random.PRNGKey(1),
                                (args.ny, args.nx, args.nz), dtype=dtype)
    w = jnp.zeros((args.ny, args.nx, args.nz + 1), dtype=dtype)
    w = w.at[..., 1:args.nz].set(
        0.5 * amp * jax.random.normal(jax.random.PRNGKey(2),
                                      (args.ny, args.nx, args.nz - 1), dtype=dtype))
    u, v, w = sl.project(u, v, w, dt=args.dt, g=g)          # divergence-free IC
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u),
                             rhs_v_prev=jnp.zeros_like(v), rhs_w_prev=jnp.zeros_like(w))
    # Ekman: geostrophic balance (Coriolis vs PG) drives the wind, no body force.
    # Neutral channel: constant PG body force ⇒ target u_*.
    force = (0.0, 0.0) if args.ekman else (args.ustar ** 2 / args.Lz, 0.0)
    return g, st, force


def profiles(st, g):
    u = np.asarray(st.u); v = np.asarray(st.v)
    wc = np.asarray(sl.f2c(st.w))
    um = u.mean((0, 1)); vm = v.mean((0, 1)); wm = wc.mean((0, 1))
    up, vp, wp = u - um, v - vm, wc - wm
    uw = (up * wp).mean((0, 1)); vw = (vp * wp).mean((0, 1))
    uu = (up * up).mean((0, 1)); vv = (vp * vp).mean((0, 1)); ww = (wp * wp).mean((0, 1))
    return um, vm, uu, vv, ww, uw, vw


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--Lx", type=float, default=2000.0)
    p.add_argument("--Ly", type=float, default=2000.0)
    p.add_argument("--Lz", type=float, default=1000.0)
    p.add_argument("--z0", type=float, default=0.1)
    p.add_argument("--cs", type=float, default=0.25,
                   help="Smagorinsky C_s. ~0.25-0.30 + 3/2 dealiasing reproduces MOST; the Bou-Zeid LASD dynamic coefficient is the proper fix.")
    p.add_argument("--ustar", type=float, default=0.45, help="target u_* [m/s]")
    p.add_argument("--dt", type=float, default=0.4)
    p.add_argument("--hours", type=float, default=1.5)
    p.add_argument("--f32", action="store_true")
    # SGS closure: dynamic LASD is the DEFAULT (consistent with the sibling
    # run_spectral_sbl.py / run_spectral_cbl.py drivers). The static
    # constant-coefficient Smagorinsky closure cannot self-transition the
    # pure-shear neutral/ekman cases from small IC perturbations — it applies
    # mean-shear-based eddy viscosity to the laminar field and damps the
    # instabilities that trigger turbulence (the constant-Smagorinsky
    # deficiency that motivated dynamic models; Germano 1991). Verified: static
    # SUSTAINS a developed neutral field (warm-start u*_res≈0.13, wvar≈0.10) but
    # relaminarises it from rest, whereas dynamic LASD self-transitions and
    # sustains (u*_res≈0.26). The buoyancy-forced gabls1/wangara cases transition
    # fine under static; only the cold-start pure-shear cases need dynamic.
    sgs = p.add_mutually_exclusive_group()
    sgs.add_argument("--dynamic", action="store_true",
                     help="Bou-Zeid LASD scale-dependent dynamic C_s(x,y,z) — the "
                          "DEFAULT. Kept for back-compat/explicitness; dynamic is on "
                          "unless --static is given.")
    sgs.add_argument("--static", action="store_true",
                     help="use the static constant-coefficient Smagorinsky closure "
                          "instead of dynamic LASD. NOTE: static relaminarises the "
                          "pure-shear neutral/ekman cases from a cold start (it sustains "
                          "a developed field but cannot self-transition) — use it for "
                          "warm-started/developed runs, the buoyancy-forced cases, or "
                          "anisotropic dx<dz grids where LASD destabilises.")
    p.add_argument("--ekman", action="store_true",
                   help="neutral ROTATING Ekman layer: geostrophic wind Ug + "
                        "Coriolis fcor instead of the non-rotating PG channel")
    p.add_argument("--Ug", type=float, default=10.0,
                   help="geostrophic wind speed [m/s] (Ekman mode)")
    p.add_argument("--fcor", type=float, default=1.0e-4,
                   help="Coriolis parameter [1/s] (Ekman mode)")
    p.add_argument("--ic-amp", type=float, default=0.05, help="IC perturbation as fraction of bulk wind")
    p.add_argument("--nu-floor", type=float, default=0.0,
                   help="background eddy-viscosity floor [m²/s] — damps residual "
                        "high-k energy in the quiescent layer above the BL where "
                        "the Smagorinsky ν_t vanishes.")
    p.add_argument("--tau-bulk", type=float, default=100.0, help="bulk-relax timescale [s]")
    p.add_argument("--sgs-model", choices=["smagorinsky", "vreman"],
                   default="smagorinsky", help="static SGS closure (ignored if --dynamic)")
    p.add_argument("--time-scheme", choices=["rk3", "ab2"], default="rk3")
    p.add_argument("--adaptive-dt", action="store_true",
                   help="CFL-adaptive dt (recomputed every --dt-recompute steps)")
    p.add_argument("--cfl", type=float, default=1.0, help="advective CFL target (RK3 stable ~1.4)")
    p.add_argument("--dt-max", type=float, default=2.0, help="cap on the adaptive dt [s]")
    p.add_argument("--max-wind", type=float, default=20.0,
                   help="conservative max resolved speed [m/s] for the static-CFL dt")
    p.add_argument("--dt-recompute", type=int, default=25,
                   help="steps between adaptive-dt recomputations")
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--record-frames", type=int, default=20,
                   help="evenly-spaced frames (snapshots + profiles) for the "
                        "publication diagnostics; 0 disables.")
    p.add_argument("--case-label", type=str, default="ekman",
                   help="case name stored in the frames (plot title).")
    p.add_argument("--output", type=Path, default=Path("results/spectral_neutral"))
    return p


def main():
    args = make_parser().parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)

    g, st, force = build(args, dtype)
    # CONSTANT-MASS-FLUX forcing: pin the bulk ⟨u⟩ to the log-law target each step
    # so the surface drag cannot collapse the mean wind (standard channel-LES
    # forcing). u_* then emerges from the resolved+SGS surface stress at that
    # bulk velocity and the BL equilibrates to Monin–Obukhov in a few turnovers,
    # rather than the ~14-turnover spin-up of a fixed body force.
    z = g.z_c
    u_bulk_target = float(jnp.mean(
        args.ustar / _KAPPA * jnp.log(jnp.clip(z, args.z0, None) / args.z0)))

    # Rayleigh sponge in the top 25%: relax the resolved fluctuations toward
    # their horizontal mean and damp w, so turbulent momentum/energy transported
    # to the stress-free rigid lid is absorbed rather than accumulating there
    # (standard ABL-LES; without it the half-channel piles momentum at the lid).
    zc = g.z_c
    z_sp = 0.75 * args.Lz
    spc = jnp.where(zc > z_sp,
                    0.5 * (1.0 - jnp.cos(jnp.pi * (zc - z_sp) / (args.Lz - z_sp))),
                    0.0).astype(dtype)               # (nz,) 0→1 ramp
    zf = g.z_f
    spf = jnp.where(zf > z_sp,
                    0.5 * (1.0 - jnp.cos(jnp.pi * (zf - z_sp) / (args.Lz - z_sp))),
                    0.0).astype(dtype)
    tau_sp = 50.0                                    # sponge timescale [s]
    rc = (args.dt / tau_sp) * spc
    rf = (args.dt / tau_sp) * spf

    # INTEGRAL-CONTROLLED body force: a uniform fx(t) adapts SLOWLY to hold the
    # bulk ⟨u⟩ at target. Unlike the exact per-step velocity re-pin (which
    # re-injected the turbulent ⟨u⟩-fluctuation energy each step → wvar ~5× too
    # high), the force integrates the bulk error on a slow timescale tau_bulk, so
    # it tracks only the MEAN drag, not the fast fluctuations. fx is carried
    # across steps as an explicit state.
    fx0 = jnp.asarray(0.0 if args.ekman else args.ustar ** 2 / args.Lz, dtype=dtype)
    gain = 1.0 / args.tau_bulk

    # Ekman: geostrophic forcing (u_geo, f_cor) replaces the channel's PG body
    # force + bulk integral control (the wind is set by the f vs PG balance, not
    # pinned to a log-law target). `args.ekman` is a static Python bool ⇒ the
    # branch is resolved at trace time, no per-step host control flow.
    u_geo = (args.Ug, 0.0) if args.ekman else (0.0, 0.0)
    f_cor = args.fcor if args.ekman else 0.0

    @partial(jax.jit, static_argnames=("first",))
    def step(state, fx, dt, first=False):
        force = (0.0, 0.0) if args.ekman else (fx, 0.0)
        state, us = sl.step(state, g=g, dt=dt, u_geo=u_geo,
                            f_cor=f_cor, first=first, force=force)
        rc = (dt / tau_sp) * spc                            # sponge from the live dt
        rf = (dt / tau_sp) * spf
        u, v, w = state.u, state.v, state.w
        u = u - rc * (u - u.mean((0, 1), keepdims=True))   # sponge: damp fluctuations
        v = v - rc * (v - v.mean((0, 1), keepdims=True))
        w = w - rf * w                                       # sponge: damp w toward 0
        # The z-varying sponge damping does NOT preserve ∇·u=0; re-project so the
        # next step starts incompressible (codex physics review 2026-06-09).
        u, v, w = sl.project(u, v, w, dt=dt, g=g)
        if not args.ekman:
            fx = fx + gain * (u_bulk_target - jnp.mean(u))  # slow integral control
        return state._replace(u=u, v=v, w=w), fx, us
    tau = args.Lz / args.ustar                              # eddy turnover [s]
    print(f"[spectral-LES neutral] {args.nx}x{args.ny}x{args.nz} "
          f"L=({args.Lx},{args.Ly},{args.Lz}) m  u*_tar={args.ustar}  "
          f"{args.time_scheme} sgs={'LASD' if not args.static else args.sgs_model}  "
          f"dt={'adaptive cfl='+str(args.cfl) if args.adaptive_dt else args.dt}  "
          f"turnover~{tau:.0f}s  dtype={dtype.__name__}")
    rec = args.record_frames > 0
    zc_np = np.asarray(g.z_c)
    theta_const = np.full((args.ny, args.nx, args.nz), 290.0, np.float32)
    if rec:
        h_idx, h_z = les_record.select_heights(zc_np, args.Lz)
        frame = 0
        _last_rec_h = [None]

        def _save(t_hours):
            nonlocal frame
            les_record.record_frame(
                args.output, frame, t_hours, args.case_label, zc_np,
                np.asarray(st.u), np.asarray(st.v), np.asarray(sl.f2c(st.w)),
                theta_const, args.Lx, args.Ly, h_idx, h_z, args.z0)
            frame += 1
            _last_rec_h[0] = t_hours
        print(f"  recording {args.record_frames} frames; heights[m]={np.round(h_z,1)}")

    fx = fx0
    # STATIC trace-time CFL dt (reuses timestepping.split_explicit.select_dt): a
    # compile-time-constant float from a conservative max wind, NOT a per-step
    # re-pin — keeps the step scan-friendly / reverse-differentiable (see
    # select_n_outer_split's --adaptive-dt discussion).
    dt0 = (select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                     dt_cap=args.dt_max) if args.adaptive_dt else float(args.dt))
    dt = jnp.asarray(dt0, dtype)
    T = args.hours * 3600.0
    print(f"  dt={dt0:.3f}s ({'static-CFL' if args.adaptive_dt else 'fixed'})")
    st, fx, us = step(st, fx, dt, first=True)
    if rec:
        _save(0.0)
    t = float(dt); i = 1
    next_rec = T / args.record_frames if rec else jnp.inf
    blk = max(1, args.dt_recompute)
    t0 = time.time()
    while t < T:
        dth = float(dt)
        for _ in range(blk):
            st, fx, us = step(st, fx, dt, first=False)
        t += blk * dth; i += blk
        mw = float(jnp.max(jnp.abs(st.w)))
        if not np.isfinite(mw) or mw > 1e3:
            print(f"[BLOWUP] step {i} t={t:.0f}s max|w|={mw}"); return 1
        if rec and t >= next_rec and frame < args.record_frames:
            _save(t / 3600.0); next_rec += T / args.record_frames
        if (i // blk) % max(1, args.print_every // blk) == 0:
            wc = sl.f2c(st.w); wv = float(jnp.mean(wc ** 2))
            spd = float(jnp.mean(jnp.sqrt(st.u ** 2 + st.v ** 2)))
            print(f"{i:7d} {t:8.0f}s dt={dth:.3f} max|w|={mw:6.3f} wvar={wv:7.4f} "
                  f"u*={float(us):.3f} <spd>={spd:5.2f}")
    wall = time.time() - t0
    # ALWAYS record the terminal state. The initial-state frame consumes one
    # slot of --record-frames, so the in-loop cadence stops one step short of
    # t = T and the old `frame < record_frames` guard was already exhausted
    # here -- every reference silently ended one cadence step early (measured:
    # --record-frames 3 over 0.25 h gave t = 0, 0.083, 0.167 h, never 0.25).
    # Guarded on the time, not the count, so it cannot emit a duplicate frame.
    if rec and (_last_rec_h[0] is None or t / 3600.0 > _last_rec_h[0] + 1.0e-9):
        _save(t / 3600.0)
    print(f"[DONE] wall={wall:.1f}s  {i/wall:.1f} steps/s  ({i} steps, t={t:.0f}s)")
    um, vm, uu, vv, ww, uw, vw = profiles(st, g)
    u_star_res = float((uw[0] ** 2 + vw[0] ** 2) ** 0.25)   # RESOLVED stress only
    u_star = float(us)   # TOTAL surface stress (wall model = resolved+SGS) — the
    #                      physically-correct MOST u_* (the resolved part alone is
    #                      low because the near-wall surface layer is under-resolved
    #                      on a uniform grid and the SGS carries the rest).
    z = np.asarray(g.z_c)
    np.savez(args.output / "final_profiles.npz",
             z=z, theta=np.full_like(z, 290.0), u=um, v=vm,
             wvar=ww, uu=uu, vv=vv, ww=ww, tke=0.5 * (uu + vv + ww),
             uw=uw, vw=vw, u_star=u_star, u_star_resolved=u_star_res,
             z0=args.z0, case="neutral_spectral")
    print(f"  profiles -> {args.output}/final_profiles.npz  (u*_total≈{u_star:.3f}"
          f", u*_resolved≈{u_star_res:.3f}, target {args.ustar})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
