"""GABLS3 (Cabauw, 2 July 2006, 00-09 UTC) on the spectral plane LES.

Stable nocturnal boundary layer with a low-level jet through the morning
transition. Forcing from ``legoesm.atmosphere.forcing.gabls3_les`` (Basu 2008):
prescribed θ and q at 0.25 m through Monin–Obukhov similarity, time-varying
geostrophic wind, advective θ / q / momentum tendencies, a 0.0029 K/m lid
inversion. Moisture acts only through θ_v buoyancy (no condensation).

The case is HELD OUT of tuning. While the catalog lists unapproved values
(``GABLS3_PENDING``) only ``--smoke`` runs are allowed.

Example (CPU smoke):
    JAX_PLATFORMS=cpu python scripts/run/run_gabls3_les.py --smoke --hours 0.01 \\
        --nx 32 --ny 32 --nz 32 --emit-suite-artifact /tmp/gabls3_smoke.npz
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

from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.dynamics.les.spectral_les_moist import (  # noqa: E402
    make_anelastic_reference,
)
from legoesm.atmosphere.forcing import gabls3_les as gb  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig  # noqa: E402
from legoesm.atmosphere.les_suite.bridge import save_artifact  # noqa: E402
from legoesm.atmosphere.les_suite.catalog import (  # noqa: E402
    GABLS3_APPROVED,
    GABLS3_PENDING,
    GABLS3_SPONGE_TAU_S,
    MOIST_CASES,
    case_microphysics,
)
from legoesm.atmosphere.les_suite.checkpoint import (  # noqa: E402
    EXIT_CHECKPOINTED,
    add_checkpoint_args,
    checkpoint_fingerprint,
    load_checkpoint,
    restore_state,
    save_checkpoint,
)
from legoesm.atmosphere.les_suite.frame_stats import (  # noqa: E402
    MoistRunEmitter,
    column_contents,
    moist_step_budget,
)
from legoesm.atmosphere.les_suite.scm_runner import microphysics_record  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_les_suite as suite  # noqa: E402  (the suite's SGS closure selector)

_CASE = next(c for c in MOIST_CASES if c.name == "gabls3")
_THETA_REF_K = gb.SURFACE_THETA[0][1]      # pending: θ(0.25 m) at 00 UTC


def parse_args(argv=None):
    gr = _CASE.grid
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=gr.nx)
    p.add_argument("--ny", type=int, default=gr.ny)
    p.add_argument("--nz", type=int, default=gr.nz)
    p.add_argument("--dt", type=float, default=gr.dt_s)
    p.add_argument("--hours", type=float, default=None,
                   help="only with --smoke; production runs the case's 9 h")
    p.add_argument("--smoke", action="store_true",
                   help="short test run; allowed while values are pending")
    p.add_argument("--sgs", choices=["lasd", "smagorinsky", "vreman"],
                   default=_CASE.sgs_variants[0])
    p.add_argument("--f32", action="store_true")
    p.add_argument("--record-frames", type=int, default=54)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--case-label", default=_CASE.name)
    p.add_argument("--emit-suite-artifact", type=Path, required=True)
    add_checkpoint_args(p)
    return p.parse_args(argv)


def build(args, dtype):
    """Grid + reference column + initial state."""
    gr = _CASE.grid
    cfg = sl.SpectralLESConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, Lx=gr.Lx_m, Ly=gr.Ly_m, Lz=gr.Lz_m,
        z0=gb.Z0_M, dealias=True, **suite._sgs_les_config(args.sgs),
        c_s=suite._SBL_STATIC_CS, time_scheme="rk3", buoyancy=True,
        theta_ref0=_THETA_REF_K, pr_sgs=suite._PR_SGS, nu_floor=suite._NU_FLOOR_M2_S,
        sgs_buoyancy=False, moist=True, n_tracers=1, monotone_scalars=False,
        sfc_flux_drag="most", sfc_ref_height=gb.Z_REF_M, most_virtual=True,
        theta_top_gradient=gb.THETA_TOP_GRADIENT_K_M)
    g = sl.make_grid(cfg, dtype=dtype)
    z_c = np.asarray(g.z_c)
    th, q, u, v = gb.initial_profiles(z_c)
    ref = make_anelastic_reference(z_c, np.asarray(g.z_f), gb.P_SFC_PA, th, q,
                                   dtype=dtype)
    sig_uv, sig_th = (jnp.asarray(s, dtype) for s in gb.perturbation_std(z_c))
    shape = (args.ny, args.nx, args.nz)

    def noise(seed):
        return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype)

    th3 = jnp.broadcast_to(jnp.asarray(th, dtype), shape) + sig_th * noise(0)
    u3 = jnp.broadcast_to(jnp.asarray(u, dtype), shape) + sig_uv * noise(1)
    v3 = jnp.broadcast_to(jnp.asarray(v, dtype), shape) + sig_uv * noise(2)
    w3 = jnp.zeros((args.ny, args.nx, args.nz + 1), dtype)
    u3, v3, w3 = sl.project(u3, v3, w3, dt=args.dt, g=g)
    tr = jnp.broadcast_to(jnp.asarray(q, dtype)[None, None, :, None],
                          shape + (1,))
    st = sl.SpectralLESState(
        u=u3, v=v3, w=w3, rhs_u_prev=jnp.zeros_like(u3), rhs_v_prev=jnp.zeros_like(v3),
        rhs_w_prev=jnp.zeros_like(w3), theta=th3, rhs_theta_prev=jnp.zeros_like(th3),
        tracers=tr, rhs_tracers_prev=jnp.zeros_like(tr))
    return g, st, ref


def forcing_at(z_c, f_cor, t0, t1):
    """Everything the step over ``[t0, t1]`` applies (interval means): effective
    geostrophic wind (momentum advection folded in), θ and q advection profiles,
    and the 0.25 m θ, q."""
    ug, vg = gb.geostrophic_mean(z_c, t0, t1)
    adv = gb.advection_mean(z_c, t0, t1)
    th_s, q_s = gb.surface_mean(t0, t1)
    return dict(ug=ug + adv["v"] / f_cor, vg=vg - adv["u"] / f_cor,
                th_adv=adv["theta"], q_adv=adv["q"], th_s=th_s, q_s=q_s)


def main(argv=None):
    args = parse_args(argv)
    # This driver has no condensation; it refuses a catalog that records otherwise.
    recorded = case_microphysics("gabls3")
    if recorded != "none":
        raise RuntimeError(f"gabls3: the catalog records microphysics {recorded!r}, "
                           "this driver runs none")
    if GABLS3_PENDING and not args.smoke:
        raise SystemExit("gabls3: values the user has not approved — production is "
                         "refused (run --smoke to test):\n  " + "\n  ".join(GABLS3_PENDING))
    if args.hours is not None and not args.smoke:
        raise SystemExit("only a --smoke run may set --hours")
    if args.smoke and args.hours is None:
        raise SystemExit("--smoke needs --hours")
    if args.record_frames < 1:
        raise SystemExit("--record-frames must be >= 1")
    if args.wall_limit_s is not None and args.checkpoint is None:
        raise SystemExit("--wall-limit-s needs --checkpoint")
    hours = args.hours if args.hours is not None else _CASE.duration_hours
    dtype = jnp.float32 if args.f32 else jnp.float64
    g, st, ref = build(args, dtype)
    f_cor = gb.coriolis_parameter()
    z_c = np.asarray(g.z_c)
    Lz = g.cfg.Lz
    z_sp = _CASE.sponge_base_m()
    spc = jnp.where(g.z_c > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (g.z_c - z_sp) / (Lz - z_sp))), 0.0).astype(dtype)
    spf = jnp.where(g.z_f > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (g.z_f - z_sp) / (Lz - z_sp))), 0.0).astype(dtype)

    def _cols(state):
        return column_contents(state, g, ref)

    @partial(jax.jit, static_argnames=("first",))
    def step(state, dt, ug, vg, th_adv, q_adv, th_s, q_s, first=False):
        state, us, b = sl.step(state, g=g, dt=dt, u_geo=(ug, vg), f_cor=f_cor,
                               first=first, force=(0.0, 0.0), t_sfc=th_s, q_sfc=q_s,
                               return_budget=True)
        c0 = _cols(state)
        state = state._replace(theta=state.theta + dt * th_adv,
                               tracers=state.tracers.at[..., 0].add(dt * q_adv))
        c1 = _cols(state)
        rc, rf = (dt / GABLS3_SPONGE_TAU_S) * spc, (dt / GABLS3_SPONGE_TAU_S) * spf
        u = state.u - rc * (state.u - sl.planar_mean(state.u, g, keepdims=True))
        v = state.v - rc * (state.v - sl.planar_mean(state.v, g, keepdims=True))
        th = state.theta - rc * (state.theta
                                 - sl.planar_mean(state.theta, g, keepdims=True))
        w = state.w - rf * state.w
        u, v, w = sl.project(u, v, w, dt=dt, g=g)
        state = state._replace(u=u, v=v, w=w, theta=th)
        budget = moist_step_budget(b, ls=c1 - c0, sponge=_cols(state) - c1)
        return state, us, budget

    dt0 = float(args.dt)
    T = hours * 3600.0
    n_steps = int(round(T / dt0))
    nz = z_c.shape[0]
    print(f"[GABLS3 LES] {args.nx}x{args.ny}x{args.nz} dx={g.dx:.2f} dz={g.dz:.2f} "
          f"dt={dt0}s sgs={args.sgs} f={f_cor:.4e} {n_steps} steps "
          f"{'SMOKE' if args.smoke else 'PRODUCTION'} dtype={dtype.__name__}")

    def _frc(t):
        return forcing_at(z_c, f_cor, t, t + dt0)

    f0 = _frc(0.0)
    emitter = MoistRunEmitter(g, ref, (jnp.asarray(f0["ug"], dtype),
                                       jnp.asarray(f0["vg"], dtype)), f_cor)
    cum = {k: np.zeros(nz) for k in ("theta_adv", "qv_adv", "u_geo", "v_geo")}
    frame = 0

    def _save(t):
        nonlocal frame
        fr = _frc(t)                   # what the next step applies at this state
        emitter.record(t, st, 0.0, 0.0, 0.0,
                       series={f"{k}_cum": v.copy() for k, v in cum.items()},
                       t_sfc=jnp.asarray(fr["th_s"], dtype),
                       q_sfc=jnp.asarray(fr["q_s"], dtype),
                       u_geo=(jnp.asarray(fr["ug"], dtype), jnp.asarray(fr["vg"], dtype)))
        frame += 1

    fp = (checkpoint_fingerprint(args, (st, ref), __file__, gb.__file__)
          if args.checkpoint is not None else None)
    resumed = (load_checkpoint(args.checkpoint, fp)
               if fp is not None and args.checkpoint.exists() else None)
    t, i = 0.0, 0
    next_rec = T / args.record_frames
    if resumed is None:
        emitter.start(st)
        _save(0.0)
    else:
        st = restore_state(resumed["st"])
        t, i = float(resumed["t"]), int(resumed["i"])
        next_rec, frame = float(resumed["next_rec"]), int(resumed["frame"])
        cum = {k: np.asarray(v) for k, v in resumed["cum"].items()}
        emitter.load_state_dict(resumed["emitter"])
        print(f"[RESUME] {args.checkpoint} at step {i}/{n_steps}")

    def _checkpoint():
        save_checkpoint(args.checkpoint, dict(
            st=st, t=t, i=i, next_rec=next_rec, frame=frame, cum=cum, emitter=emitter.state_dict()), fp)

    dt = jnp.asarray(dt0, dtype)
    t0_wall = time.time()
    while i < n_steps:
        fr = _frc(t)
        st, us, budget = step(st, dt, *(jnp.asarray(fr[k], dtype) for k in
                                        ("ug", "vg", "th_adv", "q_adv", "th_s", "q_s")),
                              first=(i == 0))
        emitter.add(budget)
        cum["theta_adv"] += dt0 * fr["th_adv"]
        cum["qv_adv"] += dt0 * fr["q_adv"]
        cum["u_geo"] += dt0 * fr["ug"]
        cum["v_geo"] += dt0 * fr["vg"]
        i += 1
        t = i * dt0
        if i % args.print_every == 0:
            mw = float(jnp.max(jnp.abs(st.w)))
            if not np.isfinite(mw) or mw > 1e3:
                print(f"[BLOWUP] step {i} max|w|={mw}")
                return 1
            print(f"{i:7d} {t / 3600.0:5.2f}h max|w|={mw:5.2f} u*={float(us):.3f} "
                  f"min q={float(jnp.min(st.tracers[..., 0])):.3e}", flush=True)
        if t >= next_rec - 1e-9 and i < n_steps:
            _save(t)
            next_rec += T / args.record_frames
            if args.checkpoint is not None:
                _checkpoint()
        if (args.wall_limit_s is not None and i < n_steps
                and time.time() - t0_wall >= args.wall_limit_s):
            _checkpoint()
            print(f"[CHECKPOINT] wall limit at step {i}/{n_steps}; exit "
                  f"{EXIT_CHECKPOINTED}")
            return EXIT_CHECKPOINTED
    wall = time.time() - t0_wall
    print(f"[DONE] wall={wall:.0f}s {i / max(wall, 1e-9):.2f} steps/s")
    _save(t)
    emitter.close_budgets(st)
    art = emitter.artifact(
        case_name=args.case_label, sgs=args.sgs, f_c=float(f_cor),
        les_setup={
            "grid": f"{args.nx}x{args.ny}x{args.nz}",
            "domain_m": [g.cfg.Lx, g.cfg.Ly, g.cfg.Lz], "dt_s": dt0, "hours": hours,
            "sgs": args.sgs, "sgs_buoyancy": False, "nu_floor_m2_s": g.cfg.nu_floor,
            "pr_sgs": g.cfg.pr_sgs, "scalar_advection": "spectral",
            "z0_m": gb.Z0_M, "surface_ref_height_m": gb.Z_REF_M,
            "theta_ref_K": _THETA_REF_K, "most_virtual": True,
            "theta_top_gradient_K_m": gb.THETA_TOP_GRADIENT_K_M,
            "sponge": [z_sp, GABLS3_SPONGE_TAU_S],
            "microphysics": microphysics_record(
                MicrophysicsConfig(scheme=case_microphysics("gabls3"))),
            "held_out": True, "smoke": bool(args.smoke),
            "approved": list(GABLS3_APPROVED), "pending": list(GABLS3_PENDING),
            "surface": "prescribed θ, q at 0.25 m through MOST; the realised flux "
                       "series is recorded",
            "forcing": "u_geo/v_geo_cum = effective geostrophic wind (momentum "
                       "advection folded in); theta_adv/qv_adv_cum = advection",
        })
    save_artifact(art, args.emit_suite_artifact)
    print(f"[emit-suite-artifact] {art.case_name}: {art.nt} frames -> "
          f"{args.emit_suite_artifact}")
    if args.checkpoint is not None and args.checkpoint.exists():
        args.checkpoint.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
