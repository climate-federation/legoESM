"""Eady baroclinic instability — AUDITED/REBUILT eddy-resolving stack.

Runs the corrected ``build_eady_uniform_setup`` (implicit-CN barotropic, WENO5
tracer+momentum, RK3+AB2 integrators, smc03 PGF, scale-aware biharmonic
Smagorinsky, KPP & GM/Redi off) forward and reports, vs the analytical Eady rate:

  * STABILITY — finite u, no grid-scale spectral spike, bounded max|u|.
  * GROWTH RATE — fit λ from log(EKE) over the linear phase; EKE ~ exp(2·σ_Eady·t).

Compare against the stale-stack baseline (``run_eady_advection_comparison.py``) and
the Veros reference (``scripts/tmp/_eady_veros_growth.py``).

Usage::

    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/run/run_eady_rebuilt.py --resolution 30x30 --days 30 --dt 600
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np

_SPD = 86400.0


def _eke(state) -> float:
    """Eddy KE = <0.5 (u - <u>_zonal)^2> over wet cells (u on C-grid faces)."""
    u = np.asarray(state.u.data)                 # (n_lat, n_lon+1, nlev)
    u = np.nan_to_num(u, nan=0.0)
    u_zm = u.mean(axis=1, keepdims=True)         # zonal (lon) mean
    return float(np.mean(0.5 * (u - u_zm) ** 2))


def _zonal_spectrum(state):
    """Zonal (lon) wavenumber EKE spectrum of the eddy u, averaged over lat+depth.

    Returns (P[k], gridscale_frac). ``gridscale_frac`` = fraction of spectral
    energy in the top quartile of wavenumbers (the dissipation/grid-scale range):
    a clean eddy-resolving run rolls off there (small frac); an under-dissipated
    run piles grid-scale energy up (large frac). This is the effective-resolution /
    over-vs-under-dissipation proxy the min-dissipation search optimizes.
    """
    u = np.nan_to_num(np.asarray(state.u.data), nan=0.0)   # (lat, lon+1, lev)
    u = u[:, :-1, :]                                        # drop the wrap face -> (lat,lon,lev)
    u_eddy = u - u.mean(axis=1, keepdims=True)              # remove zonal mean
    n_lon = u_eddy.shape[1]
    uh = np.fft.rfft(u_eddy, axis=1)                        # zonal FFT
    P = np.mean(np.abs(uh) ** 2, axis=(0, 2)) / n_lon       # P[k], avg lat+depth
    P[0] = 0.0                                              # drop the mean
    total = P.sum()
    if total <= 0:
        return P, 0.0
    k_hi = len(P) - max(1, len(P) // 4)                     # top quartile of k
    return P, float(P[k_hi:].sum() / total)


def run(resolution: str, days: float, dt: float, out: str, no_sponge: bool = False,
        c_smag=None, c_leith=0.0, c_smag_lap=0.0, b_h=0.0, a_h=0.0, tag="",
        u_surface=None, smag_cfl_safety=0.0, momentum_advection="weno5",
        ke_gradient_scheme="centered", tracer_advection="weno5",
        barotropic_solver="implicit_cn"):
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.eady_uniform import (
        build_eady_uniform_setup, EadyUniformConfig, compute_sponge_mask,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    parts = resolution.split("x")
    n_lat = int(parts[0]); n_lon = int(parts[1]) if len(parts) > 1 else int(parts[0])
    cfg_e = (EadyUniformConfig() if u_surface is None
             else EadyUniformConfig(U_surface=u_surface))
    sigma = 0.31 * cfg_e.f0 * cfg_e.Lambda / cfg_e.N      # analytical Eady [1/s]

    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        recipe = build_eady_uniform_setup(
            n_lat=n_lat, n_lon=n_lon, config=cfg_e, c_smag=c_smag,
            c_leith=c_leith, c_smag_lap=c_smag_lap, b_h=b_h,
            smag_cfl_safety=smag_cfl_safety, a_h=a_h, momentum_advection=momentum_advection,
            ke_gradient_scheme=ke_gradient_scheme, tracer_advection=tracer_advection,
            barotropic_solver=barotropic_solver)
        grid, z, cfg, phys, state = (recipe.grid, recipe.z_coord,
                                     recipe.model_config, recipe.physics_config,
                                     recipe.initial_state)
        model = LatLonCGridOceanModel(grid, z, cfg)

        # AB2 outer integrator: seed prior-increment carries to zero (constant
        # scan pytree). implicit_cn barotropic needs no rigid-lid/psi carry.
        if cfg.outer_integrator == "ab2" and state.T_incr_prev is None:
            def _z(d):
                return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                             dims=d.dims, units=d.units)
            state = state._replace(
                T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
                u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))

        steps_per_block = max(1, int(round(_SPD / dt)))

        # Wall sponge (the old working config had one "to prevent nonlinear
        # steepening at the solid N/S walls"; dropping it lets eddies blow up
        # there, worse at finer res). Relax T->IC and damp u,v within
        # sponge_width_deg of each wall. gamma(y) [1/s], quadratic ramp.
        sponge = not no_sponge
        if sponge:
            gamma = np.asarray(compute_sponge_mask(grid, cfg_e))   # (n_lat, n_lon)
            g_lat = gamma[:, 0]                                     # constant in lon
            decay_T = jnp.asarray(np.exp(-gamma * dt))             # (n_lat, n_lon)
            decay_u = jnp.asarray(np.exp(-g_lat * dt))[:, None, None]   # (n_lat,1,1)
            g_v = np.concatenate([[g_lat[0]], 0.5 * (g_lat[:-1] + g_lat[1:]),
                                  [g_lat[-1]]])                    # v-grid lat (n_lat+1)
            decay_v = jnp.asarray(np.exp(-g_v * dt))[:, None, None]
            T_init = state.T.data

        def _body(s, _):
            s = model.step(s, dt)
            if sponge:
                dT = decay_T[:, :, None]
                # .replace preserves Field metadata (grid location cell/edge) — a
                # fresh Field(...) would default v's 'edge' to 'cell' and break the
                # scan carry pytree.
                s = s._replace(
                    T=s.T.replace(data=s.T.data * dT + T_init * (1.0 - dT)),
                    u=s.u.replace(data=s.u.data * decay_u),
                    v=s.v.replace(data=s.v.data * decay_v))
            return s, None

        @partial(jax.jit, static_argnames=("n",))
        def _block(s, n):
            s, _ = jax.lax.scan(_body, s, None, length=n)
            return s

        _mom = (cfg.momentum_advection if cfg.momentum_advection != "vector_invariant"
                else f"vec_inv+{cfg.ke_gradient_scheme}")
        print(f"== REBUILT Eady: {n_lat}x{n_lon}, dt={dt:.0f}s, {days:.0f}d, fp64 | "
              f"{cfg.barotropic.barotropic_solver}/trac={cfg.tracer_advection}/mom={_mom}/"
              f"rk3+ab2/smc03 | U={cfg_e.U_surface} sponge={not no_sponge} | "
              f"Csmag={cfg.lateral_viscosity.C_smag} Cleith={cfg.lateral_viscosity.C_leith} cap={smag_cfl_safety} "
              f"B_h={cfg.B_h} ==")
        print(f"   sigma_Eady={sigma:.3e}/s, tau={1/sigma/_SPD:.2f}d, "
              f"EKE rate 2sigma={2*sigma:.3e}/s; Ld={cfg_e.Ld_km:.0f}km, "
              f"dx_lat~{(cfg_e.lat_north-cfg_e.lat_south)*111/n_lat:.0f}km", flush=True)

        ts, ekes, umax_series = [], [], []
        ts.append(0.0); ekes.append(_eke(state)); umax_series.append(
            float(jnp.max(jnp.abs(state.u.data))))
        t0 = time.time()
        n_days = int(round(days))
        blew = False
        for d in range(1, n_days + 1):
            state = _block(state, steps_per_block)
            jax.block_until_ready(state.u.data)
            umax = float(jnp.max(jnp.abs(state.u.data)))
            e = _eke(state)
            ts.append(d * _SPD); ekes.append(e); umax_series.append(umax)
            finite = bool(jnp.all(jnp.isfinite(state.u.data)))
            if not finite or umax > 50.0:
                print(f"   *** BLEW UP at day {d}: max|u|={umax:.3e} finite={finite}")
                blew = True
                break
            if d % 5 == 0 or d == n_days:
                print(f"   day {d:3d}: EKE={e:.3e} m2/s2  max|u|={umax:.4f} m/s  "
                      f"{time.time()-t0:5.0f}s", flush=True)

        ts, ekes = np.array(ts), np.array(ekes)
        # Fit growth rate over the linear phase: from when EKE starts rising to
        # before it saturates (heuristic: EKE within [10x seed, 0.3x final-max]).
        e0 = ekes[0]
        emax = ekes.max()
        lin = (ekes > 10 * e0) & (ekes < 0.3 * emax) & (ts > 3 * _SPD)
        verdict = "BLEW UP" if blew else "STABLE"
        if lin.sum() >= 3:
            lam = np.polyfit(ts[lin], np.log(ekes[lin]), 1)[0]
            ratio = lam / (2 * sigma)
            print(f"\n   {verdict}. Fitted EKE growth lambda={lam:.3e}/s over "
                  f"{lin.sum()} pts (analytical 2sigma={2*sigma:.3e}/s, "
                  f"ratio {ratio:.2f})")
            print(f"   -> growth tau={2/lam/_SPD:.2f}d (Eady tau={1/sigma/_SPD:.2f}d); "
                  f"max|u| peak={max(umax_series):.3f} m/s")
        else:
            print(f"\n   {verdict}. Not enough linear-phase points to fit growth "
                  f"(EKE range {e0:.2e}..{emax:.2e}).")

        # --- Saturation + effective-resolution diagnostics (the min-dissipation
        # search optimizes these) ---
        # Saturated iff EKE is stationary over the last third: std/mean < 0.25 AND
        # not still climbing (last-third mean within 2x of the global max).
        sat = False
        eke_sat = float("nan")
        if not blew and len(ekes) >= 9:
            tail = ekes[-max(3, len(ekes) // 3):]
            eke_sat = float(tail.mean())
            cv = float(tail.std() / tail.mean()) if tail.mean() > 0 else 9.9
            sat = (cv < 0.25) and (eke_sat > 0.3 * emax)
        gs_frac = float("nan")
        if not blew:
            _, gs_frac = _zonal_spectrum(state)
        umax_final = umax_series[-1]
        print(f"   SATURATION: saturated={sat} EKE_sat={eke_sat:.3e} "
              f"max|u|_final={umax_final:.3f} | gridscale_frac={gs_frac:.4f} "
              f"(low=clean/high-eff-res, high=under-damped grid pileup)")
        # Single parseable line for the ralph min-dissipation loop:
        diss = f"A_h={a_h:g},B_h={b_h:g},C_smag={cfg.lateral_viscosity.C_smag:g},C_leith={c_leith:g},cap={smag_cfl_safety:g}"
        print(f"VERDICT res={n_lat}x{n_lon} U={cfg_e.U_surface} dt={dt:.0f} sponge={not no_sponge} "
              f"mom={cfg.momentum_advection} | stable={not blew} saturated={sat} "
              f"max_u={umax_final:.3f} EKE_sat={eke_sat:.3e} gridscale_frac={gs_frac:.4f} "
              f"| dissipation[{diss}]", flush=True)

        np.savez_compressed(f"{out}/eady_rebuilt_{n_lat}x{n_lon}{tag}.npz",
                            t=ts, eke=ekes, umax=np.array(umax_series),
                            sigma_eady=sigma, blew=blew, gridscale_frac=gs_frac,
                            saturated=sat, eke_sat=eke_sat)
        return verdict
    finally:
        set_policy(_prev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", default="30x30", help="n_lat x n_lon (e.g. 30x30).")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--out", default="results/eady_rebuilt")
    ap.add_argument("--no-sponge", action="store_true",
                    help="Disable the wall sponge (isolates whether the blowup is "
                         "wall steepening vs interior dissipation).")
    ap.add_argument("--c-smag", type=float, default=None,
                    help="Biharmonic Smagorinsky coeff (default config 0.2).")
    ap.add_argument("--c-leith", type=float, default=0.0,
                    help="Leith viscosity coeff (enstrophy-cascade-aware).")
    ap.add_argument("--c-smag-lap", type=float, default=0.0,
                    help="Laplacian Smagorinsky coeff.")
    ap.add_argument("--b-h", type=float, default=0.0, help="Fixed biharmonic [m^4/s].")
    ap.add_argument("--a-h", type=float, default=0.0, help="Harmonic Laplacian viscosity [m^2/s] (Veros uses 5e3).")
    ap.add_argument("--tag", default="", help="Output filename suffix.")
    ap.add_argument("--u-surface", type=float, default=None,
                    help="Surface jet speed [m/s]: 0.8 strong (tau~5d), 0.2 weak/"
                         "balanced (tau~20d, the physical eddy-resolving regime).")
    ap.add_argument("--smag-cfl-safety", type=float, default=0.0,
                    help="Cap Smagorinsky viscosity at this fraction of the viscous "
                         "CFL limit (the audit fix; prevents biharmonic-CFL blowup "
                         "at fine res).")
    ap.add_argument("--momentum-advection", default="weno5",
                    choices=("weno5", "weno7", "vector_invariant"),
                    help="weno5 = flux-form track; vector_invariant = NEMO-like "
                         "track (pair with --ke-gradient hollingsworth).")
    ap.add_argument("--ke-gradient", default="centered",
                    choices=("centered", "hollingsworth"),
                    help="KE-gradient scheme for vector_invariant momentum; "
                         "hollingsworth = NEMO nn_dynkeg=1 (anti-Hollingsworth-instab).")
    ap.add_argument("--tracer-advection", default="weno5")
    ap.add_argument("--barotropic-solver", default="implicit_cn",
                    choices=("implicit_cn", "explicit_substep", "rigid_lid"))
    args = ap.parse_args()
    import os
    os.makedirs(args.out, exist_ok=True)
    import jax
    jax.config.update("jax_enable_x64", True)
    run(args.resolution, args.days, args.dt, args.out, no_sponge=args.no_sponge,
        c_smag=args.c_smag, c_leith=args.c_leith, c_smag_lap=args.c_smag_lap,
        b_h=args.b_h, a_h=args.a_h, tag=args.tag, u_surface=args.u_surface,
        smag_cfl_safety=args.smag_cfl_safety,
        momentum_advection=args.momentum_advection,
        ke_gradient_scheme=args.ke_gradient, tracer_advection=args.tracer_advection,
        barotropic_solver=args.barotropic.barotropic_solver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
