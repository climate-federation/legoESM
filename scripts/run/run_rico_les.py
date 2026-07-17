"""RICO precipitating trade-wind cumulus on the spectral TRUE-LES core +
swappable microphysics (Morrison/M2005 default; warm rain ACTIVE).

RICO (van Zanten et al. 2011, JAMES 3; gSAM ``CASES/RICO``) — the GCSS/GEWEX
precipitating shallow-cumulus intercomparison:

* IC: gSAM ``snd`` (θ 297.9 K mixed layer, q_v 16 g/kg, trades u −9.9 → v −3.8).
* lsf: subsidence (0 → −0.5 cm/s at 2260 m, constant above), combined
  radiative+advective cooling tls = −2.5 K/day, moistening/drying qls.
* Surface: INTERACTIVE bulk fluxes over fixed SST = 299.8 K
  (``SFC_FLX_FXD=.false.``): C_h·|U₁|·(θ_s−θ₁), C_q·|U₁|·(q_s−q₁) with the
  van Zanten exchange coefficients; q_s from the shared saturation formula
  (no re-derived Clausius–Clapeyron).
* Coriolis f = 0.451e-4 (18°N) toward the lsf geostrophic wind.
* PRECIPITATING: Morrison warm rain on; surface rain rate is the key
  literature diagnostic (~0.3 mm/day domain mean by hour 24).

Reference targets (van Zanten 2011, hours 20–24): cloud cover ~10–20 %,
LWP ~10–20 g/m² (mean), cumulus tops ~2.5 km, surface precip ~0.3 mm/day
(ensemble range 0–0.8).

Reuses the BOMEX driver's forcing/profile machinery (same spectral core +
microphysics adapter; precedent: run_gate_plane imports run_rcemip_plane).

Example (GPU):
    JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_rico_les.py \\
        --nx 128 --ny 128 --nz 100 --hours 24 --f32
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
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.dynamics.les.spectral_les_moist import (  # noqa: E402
    make_anelastic_reference,
    make_les_microphysics_fn,
)
from legoesm.atmosphere.physics.microphysics.config import (  # noqa: E402
    MicrophysicsConfig,
    MorrisonConfig,
)
from legoesm.atmosphere.forcing.sam_case_forcing import (  # noqa: E402
    read_sam_lsf,
    read_sam_snd,
    read_sam_sfc,
    surface_at_day,
)
from legoesm.timestepping.split_explicit import select_dt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import les_record  # noqa: E402
import run_bomex_les as bx  # noqa: E402  (shared forcing/profiles machinery)

from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir  # noqa: E402

# Default case dir: external LEGOESM_GSAM_ROOT if set, else the repo-local
# cache (scripts/data/fetch_les_forcing.py); --case-dir overrides. See
# resolve_sam_case_dir.
_DEFAULT_CASE = resolve_sam_case_dir("RICO")
_FCOR = 0.451e-4                      # CASES/RICO/prm fcor (18°N)
# van Zanten et al. 2011 bulk exchange coefficients (at the 20 m level).
_C_H = 0.001094
_C_Q = 0.001133


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", default=_DEFAULT_CASE)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=100,
                   help="RICO protocol: 100 × dz=40 m (Lz=4000 m).")
    p.add_argument("--Lx", type=float, default=12800.0)
    p.add_argument("--Ly", type=float, default=12800.0)
    p.add_argument("--Lz", type=float, default=4000.0)
    p.add_argument("--z0", type=float, default=1.0e-4)
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--microphysics", default="morrison")
    p.add_argument("--n-tracers", type=int, default=9)
    p.add_argument("--scalar-advection", choices=["van_leer","weno5","weno5_hv"], default="van_leer",
                   help="monotone scalar reconstruction (weno5=less diffusive).")
    p.add_argument("--w-hyperdiff", type=float, default=0.0,
                   help="OPT-IN horizontal w-hyperdiffusion nu4 [m^4/s].")
    p.add_argument("--theta-hyperdiff", type=float, default=0.0,
                   help="OPT-IN scale-selective k4 hyperdiff on theta [m^4/s].")
    p.add_argument("--div-damping", type=float, default=0.0,
                   help="OPT-IN momentum divergence damping alpha [m^2/s].")
    # Shared with bx.build (the moist-stability switches; defaults match BOMEX).
    p.add_argument("--filter-monotone-scalars", action="store_true")
    p.add_argument("--filter-monotone-qv", action="store_true")
    p.add_argument("--micro-order", choices=["pre", "post"], default="pre")
    p.add_argument("--dynamic", action="store_true", default=True)
    p.add_argument("--static", dest="dynamic", action="store_false")
    p.add_argument("--sgs-model", choices=["smagorinsky", "vreman"],
                   default="vreman")
    p.add_argument("--cs", type=float, default=0.18)
    p.add_argument("--sgs-buoyancy", dest="sgs_buoyancy", action="store_true",
                   default=True, help="Lilly stable-stratification SGS suppression "
                   "(damps entrainment above cloud; on by default).")
    p.add_argument("--no-sgs-buoyancy", dest="sgs_buoyancy", action="store_false",
                   help="disable Lilly SGS suppression (strain-only ν_t).")
    p.add_argument("--nu-floor", type=float, default=0.0)
    p.add_argument("--time-scheme", choices=["rk3", "ab2"], default="rk3")
    p.add_argument("--micro-every", type=int, default=1)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--record-frames", type=int, default=24)
    p.add_argument("--case-label", type=str, default="rico")
    p.add_argument("--output", type=Path, default=Path("results/les_rico"))
    return p.parse_args()


def main():
    args = parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)
    # The BOMEX build covers everything deck-driven (snd IC + lsf forcing +
    # reference column); only the surface coupling differs (interactive bulk).
    g, st, ref, forc, _th = bx.build(args, dtype)
    sfc0 = forc["sfc"]
    sst = sfc0["sst"]
    p_sfc = float(read_sam_snd(Path(args.case_dir) / "snd").pres0) * 100.0
    micro_cfg = (MicrophysicsConfig(
        scheme="morrison", morrison=MorrisonConfig(morrison_flavor="sam"))
        if args.microphysics == "morrison"
        else MicrophysicsConfig(scheme=args.microphysics))
    dt0 = float(args.dt)
    micro = make_les_microphysics_fn(micro_cfg, ref, g.dz,
                                     dt0 * args.micro_every)
    forcing = bx.make_forcing_fn(g, ref, forc, dtype)

    # Interactive bulk surface fluxes over the fixed SST (van Zanten):
    #   θ-flux = C_h·|U₁|·(θ_sfc − θ₁)   [K·m/s]
    #   q-flux = C_q·|U₁|·(q_sat(SST,p_sfc) − q₁)   [kg/kg·m/s]
    # Planar-mean bulk (the wall model is planar-mean too); recomputed inside
    # the jitted step from the live state — no recompilation.
    exn_sfc = float(ref.exner_c[0])
    th_sfc = sst / exn_sfc                       # SST → surface θ
    q_sfc = float(saturation_mixing_ratio(jnp.asarray(sst),
                                          jnp.asarray(p_sfc)))

    zc = g.z_c; z_sp = 0.75 * args.Lz; tau_sp = 60.0
    spc = jnp.where(zc > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zc - z_sp) / (args.Lz - z_sp))), 0.0).astype(dtype)
    zf = g.z_f
    spf = jnp.where(zf > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zf - z_sp) / (args.Lz - z_sp))), 0.0).astype(dtype)

    @partial(jax.jit, static_argnames=("first", "do_micro"))
    def step(state, dt, first=False, do_micro=True):
        u1, v1 = state.u[..., 0], state.v[..., 0]
        spd1 = jnp.mean(jnp.sqrt(u1 ** 2 + v1 ** 2 + 1.0))   # gustiness floor 1
        th_flux = _C_H * spd1 * (th_sfc - jnp.mean(state.theta[..., 0]))
        qv_flux = _C_Q * spd1 * (q_sfc - jnp.mean(state.tracers[..., 0, 0]))
        state, us = sl.step(state, g=g, dt=dt,
                            u_geo=(forc["ug"], forc["vg"]), f_cor=_FCOR,
                            first=first, force=(0.0, 0.0),
                            sfc_theta_flux=th_flux, sfc_qv_flux=qv_flux)
        state = forcing(state, dt)
        if do_micro:
            dth, dtr, pr = micro(state.theta, state.tracers)
            tr = state.tracers + dt * args.micro_every * dtr
            wat = tr[..., :6]
            created = jnp.sum(jnp.clip(-wat, 0.0, None))
            tr = jnp.concatenate([jnp.clip(wat, 0.0, None),
                                  jnp.clip(tr[..., 6:], 0.0, None)], axis=-1)
            state = state._replace(
                theta=state.theta + dt * args.micro_every * dth, tracers=tr)
            precip = jnp.mean(pr)                 # domain-mean [kg/m²/s]
        else:
            created = jnp.asarray(0.0, state.theta.dtype)
            precip = jnp.asarray(0.0, state.theta.dtype)
        rc = (dt / tau_sp) * spc
        rf = (dt / tau_sp) * spf
        u = state.u - rc * (state.u - state.u.mean((0, 1), keepdims=True))
        v = state.v - rc * (state.v - state.v.mean((0, 1), keepdims=True))
        w = state.w - rf * state.w
        u, v, w = sl.project(u, v, w, dt=dt, g=g)
        return state._replace(u=u, v=v, w=w), us, created, precip

    T = args.hours * 3600.0
    n_steps = int(round(T / dt0))
    print(f"[RICO LES] {args.nx}x{args.ny}x{args.nz} dx={g.dx:.0f} dz={g.dz:.0f}"
          f" dt={dt0:.2f}s {args.time_scheme} "
          f"sgs={'LASD' if args.dynamic else args.sgs_model} "
          f"micro={micro.scheme_name} f={_FCOR:.2e} dtype={dtype.__name__}")
    print(f"  sfc: INTERACTIVE bulk over SST={sst:.1f} K "
          f"(θ_s={th_sfc:.2f} K, q_s={q_sfc * 1e3:.2f} g/kg), {n_steps} steps")

    rec = args.record_frames > 0
    zc_np = np.asarray(g.z_c)
    if rec:
        h_idx, h_z = les_record.select_heights(zc_np, args.Lz)
        frame = 0

        def _save(t_hours):
            nonlocal frame
            les_record.record_frame(
                args.output, frame, t_hours, args.case_label, zc_np,
                np.asarray(st.u), np.asarray(st.v), np.asarray(sl.f2c(st.w)),
                np.asarray(st.theta), args.Lx, args.Ly, h_idx, h_z, args.z0,
                qc3=np.asarray(st.tracers[..., 1]),
                rho_z=np.asarray(ref.rho_c))
            frame += 1

    dt = jnp.asarray(dt0, dtype)
    if rec:
        _save(0.0)
    t = 0.0; i = 0
    created_tot, precip_accum = 0.0, 0.0          # accum [kg/m² = mm]
    # Trailing time-mean of cloud metrics over the quasi-steady 2nd half
    # (t > 0.5·T): RICO cumulus pulse, so instantaneous cc/LWP swing frame-to-
    # frame; GCSS reports time-means. Sampled at the print cadence.
    cc_sum = lwp_sum = 0.0; n_cavg = 0
    t_cavg0 = 0.5 * T
    next_rec = T / args.record_frames if rec else np.inf
    t0 = time.time()
    while t < T:
        st, us, created, pr = step(st, dt, first=(i == 0),
                                   do_micro=((i + 1) % args.micro_every == 0))
        created_tot += float(created)
        precip_accum += float(pr) * dt0 * args.micro_every
        t += dt0; i += 1
        if i % args.print_every == 0:
            mw = float(jnp.max(jnp.abs(st.w)))
            if not np.isfinite(mw) or mw > 1e3:
                print(f"[BLOWUP] step {i} max|w|={mw}"); return 1
            d = bx.moist_profiles(st, g, ref)
            rate = float(pr) * 86400.0            # kg/m²/s → mm/day
            print(f"{i:7d} {t/3600.0:5.2f}h max|w|={mw:5.2f} "
                  f"cc={d['cloud_cover']:.3f} LWP={d['lwp']:6.2f} g/m² "
                  f"qr_max={float(jnp.max(st.tracers[..., 2])):.2e} "
                  f"P={rate:.3f} mm/d acc={precip_accum:.3f} mm "
                  f"u*={float(us):.3f}", flush=True)
            if t >= t_cavg0:
                cc_sum += float(d["cloud_cover"]); lwp_sum += float(d["lwp"])
                n_cavg += 1
        if rec and t >= next_rec and frame < args.record_frames:
            _save(t / 3600.0); next_rec += T / args.record_frames
    wall = time.time() - t0
    print(f"[DONE] wall={wall:.0f}s  {i/wall:.1f} steps/s")
    if rec and frame < args.record_frames:
        _save(t / 3600.0)
    d = bx.moist_profiles(st, g, ref)
    cc_avg = cc_sum / n_cavg if n_cavg else float(d["cloud_cover"])
    lwp_avg = lwp_sum / n_cavg if n_cavg else float(d["lwp"])
    np.savez(args.output / "rico_les_final.npz", z=zc_np,
             **{k: v for k, v in d.items() if isinstance(v, np.ndarray)},
             cloud_cover=d["cloud_cover"], lwp=d["lwp"],
             cloud_cover_timemean=cc_avg, lwp_timemean=lwp_avg,
             precip_accum_mm=precip_accum)
    print(f"  FINAL: cloud cover={d['cloud_cover']:.3f} "
          f"(2nd-half mean {cc_avg:.3f}) (ref 0.10-0.20), "
          f"LWP={d['lwp']:.2f} (2nd-half mean {lwp_avg:.2f}) g/m² "
          f"(ref ~10-20), accum precip={precip_accum:.3f} mm "
          f"(ref ~0.3 mm/day · {args.hours:.0f} h)")
    print(f"  profiles -> {args.output}/rico_les_final.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
