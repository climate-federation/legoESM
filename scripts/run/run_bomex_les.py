"""BOMEX shallow-cumulus on the pseudo-spectral incompressible TRUE-LES core,
coupled to the swappable legoESM microphysics (Morrison/M2005 default).

BOMEX (Siebesma et al. 2003, JAS 60; gSAM ``CASES/BOMEX``) — the canonical GCSS
trade-wind shallow-cumulus case:

* IC: the gSAM ``snd`` sounding (θ, q_v, u, v; well-mixed to 520 m, cloud layer
  to ~1500 m, inversion above; trade wind u ≈ −8.75 m/s).
* Large-scale forcing (``lsf``): subsidence w_ls (0 → −0.65 cm/s at 1500 m → 0
  at 2100 m) on θ and q_v; the COMBINED radiative+advective cooling
  tls = −2 K/day below 1500 m; drying qls = −1.2e-8 kg/kg/s below 300 m.
* Surface: FIXED kinematic fluxes (SHF 9.46 W/m², LHF 153.4 W/m²) + the MOST
  wall model with z0 tuned to the BOMEX u* ≈ 0.28 m/s.
* Coriolis f = 0.376e-4 s⁻¹ toward the HEIGHT-DEPENDENT geostrophic wind
  u_g = −10 + 1.8e-3·z m/s, v_g = 0 (the lsf u_ls profile).
* Microphysics: swappable via --microphysics (default morrison, SAM flavor).
  BOMEX is non-precipitating by construction (autoconversion is negligible at
  its LWC) — doprecip=.false. in SAM is matched by physics, not by a switch.

Reference targets (hours 3–6): cloud cover 10–15 %, LWP ~5–10 g/m², cloud base
~500 m, cloud top ~1500–2000 m, well-mixed subcloud layer, no drizzle.

Outputs (every run): frame snapshots (x-y cross-sections of w, θ', |U|, q_c +
LWP map) + planar-mean profiles (θ, q_v, q_c, cloud fraction, fluxes) as npz +
publication PNGs (plot_les_diagnostics.py, moist-aware).

Example (GPU):
    JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_bomex_les.py \\
        --nx 64 --ny 64 --nz 75 --hours 6 --f32
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
from legoesm.atmosphere.dynamics import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.dynamics.spectral_les_moist import (  # noqa: E402
    conserving_positive,
    make_anelastic_reference,
    make_les_microphysics_fn,
)
from legoesm.atmosphere.physics.microphysics.config import (  # noqa: E402
    MicrophysicsConfig,
    MorrisonConfig,
)
from legoesm.atmosphere.sam_case_forcing import (  # noqa: E402
    read_sam_lsf,
    read_sam_snd,
    read_sam_sfc,
    surface_at_day,
)
from legoesm.timestepping.split_explicit import select_dt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import les_record  # noqa: E402

import os as _os  # noqa: E402

_GSAM_ROOT = _os.environ.get(
    "LEGOESM_GSAM_ROOT", "/home/gentine/Documents/Code/gSAM/gsam1.8.7/gSAM1.8.7"
)
_DEFAULT_CASE = f"{_GSAM_ROOT}/CASES/BOMEX"
_FCOR = 0.376e-4                       # CASES/BOMEX/prm fcor [1/s]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", default=_DEFAULT_CASE)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=75,
                   help="BOMEX protocol: 75 × dz=40 m (Lz=3000 m).")
    p.add_argument("--Lx", type=float, default=6400.0)
    p.add_argument("--Ly", type=float, default=6400.0)
    p.add_argument("--Lz", type=float, default=3000.0)
    p.add_argument("--z0", type=float, default=1.0e-4,
                   help="roughness; 1e-4 m gives the BOMEX u*≈0.28 m/s at the "
                        "8.75 m/s trade wind through the neutral wall model.")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--adaptive-dt", action="store_true",
                   help="static-CFL dt from --max-wind (trace-time constant).")
    p.add_argument("--cfl", type=float, default=0.8)
    p.add_argument("--dt-max", type=float, default=2.0)
    p.add_argument("--max-wind", type=float, default=12.0)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--microphysics", default="morrison",
                   help="any MicrophysicsConfig scheme (swappable; morrison "
                        "default, SAM flavor).")
    p.add_argument("--n-tracers", type=int, default=9,
                   help="standard slot layout; 9 covers the double-moment "
                        "schemes (morrison/thompson/sb).")
    p.add_argument("--dynamic", action="store_true", default=True,
                   help="LASD dynamic SGS (default on; near-iso dx/dz).")
    p.add_argument("--static", dest="dynamic", action="store_false")
    p.add_argument("--sgs-model", choices=["smagorinsky", "vreman"],
                   default="vreman")
    p.add_argument("--cs", type=float, default=0.18)
    p.add_argument("--nu-floor", type=float, default=0.0)
    p.add_argument("--time-scheme", choices=["rk3", "ab2"], default="rk3")
    p.add_argument("--micro-every", type=int, default=1,
                   help="apply microphysics every N dycore steps (1 = each).")
    p.add_argument("--micro-order", choices=["pre", "post"], default="pre",
                   help="'pre' = forcing/microphysics before dycore so buoyancy "
                        "sees the fresh latent heating in the same step "
                        "(default); 'post' reproduces the old split order.")
    p.add_argument("--micro-substeps", type=int, default=1,
                   help="subcycle microphysics within each micro-every interval.")
    p.add_argument("--micro-relax-factor", type=float, default=1.0,
                   help="experimental: multiply the dt passed into the column "
                        "microphysics, reducing saturation-adjustment stiffness "
                        "while retaining the actual dycore update interval.")
    p.add_argument("--positive-mode", choices=["conserving", "clip"],
                   default="conserving",
                   help="water-tracer non-negativity after microphysics.")
    p.add_argument("--filter-monotone-scalars", action="store_true",
                   help="reproduce the old behavior: apply the sharp spectral "
                   "cutoff to monotone theta/tracers after each dycore step.")
    p.add_argument("--filter-monotone-qv", action="store_true",
                   help="apply the sharp spectral cutoff only to monotone q_v; "
                        "hydrometeor tracers remain unfiltered.")
    p.add_argument("--print-every", type=int, default=500)
    p.add_argument("--record-frames", type=int, default=12)
    p.add_argument("--case-label", type=str, default="bomex")
    p.add_argument("--output", type=Path, default=Path("results/les_bomex"))
    return p.parse_args()


def build(args, dtype):
    """Grid + reference column + IC + forcing profiles from the gSAM deck."""
    cfg = sl.SpectralLESConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        z0=args.z0, dealias=True, c_s=args.cs,
        smagorinsky_dynamic=args.dynamic, sgs_model=args.sgs_model,
        time_scheme=args.time_scheme, nu_floor=args.nu_floor,
        buoyancy=True, theta_ref0=300.0, pr_sgs=1.0,
        moist=True, n_tracers=args.n_tracers, monotone_scalars=True,
        filter_monotone_qv=args.filter_monotone_qv,
        filter_monotone_scalars=args.filter_monotone_scalars)
    g = sl.make_grid(cfg, dtype=dtype)
    case = Path(args.case_dir)
    snd = read_sam_snd(case / "snd")
    lsf = read_sam_lsf(case / "lsf")
    sfc0 = surface_at_day(read_sam_sfc(case / "sfc"), day=0.0)

    z_c = np.asarray(g.z_c)
    z_snd = np.asarray(snd.z)
    th_prof = np.interp(z_c, z_snd, np.asarray(snd.theta))
    qv_prof = np.interp(z_c, z_snd, np.asarray(snd.q_v))   # already kg/kg
    u_prof = np.interp(z_c, z_snd, np.asarray(snd.u))
    v_prof = np.interp(z_c, z_snd, np.asarray(snd.v))
    # lsf (day-0 block): geostrophic wind + subsidence + tendencies on z_c.
    zl = np.asarray(lsf.z[0]); order = np.argsort(zl)
    interp = lambda f: np.interp(z_c, zl[order],            # noqa: E731
                                 np.asarray(f[0])[order], left=np.nan,
                                 right=0.0)
    # np.interp clamps ug to the lsf top value (−5.51 at 2500 m) above the
    # forcing column — matching SAM's own clamped lsf interpolation; the
    # analytic Siebesma ug=−10+1.8e-3·z would differ by ≤0.9 m/s only inside
    # the sponge layer (codex (e) note — deck-faithful behaviour kept).
    ug = np.interp(z_c, zl[order], np.asarray(lsf.u_ls[0])[order])
    vg = np.interp(z_c, zl[order], np.asarray(lsf.v_ls[0])[order])
    w_ls = np.nan_to_num(interp(lsf.w_ls))
    tls = np.nan_to_num(interp(lsf.T_ls))                   # dT/dt [K/s]
    qls = np.nan_to_num(interp(lsf.qv_ls))                  # dq_v/dt [1/s]

    p_sfc = float(snd.pres0) * 100.0
    ref = make_anelastic_reference(z_c, np.asarray(g.z_f), p_sfc,
                                   th_prof, qv_prof, dtype=dtype)

    # Initial state: sounding profiles + small near-surface θ noise to seed
    # convection (the SBL/CBL drivers' standard seeding).
    ny, nx, nz = args.ny, args.nx, args.nz
    key = jax.random.PRNGKey(0)
    seed = (jnp.asarray(z_c) < 600.0).astype(dtype)
    th3 = (jnp.broadcast_to(jnp.asarray(th_prof, dtype), (ny, nx, nz))
           + 0.1 * jax.random.normal(key, (ny, nx, nz), dtype) * seed)
    u3 = (jnp.broadcast_to(jnp.asarray(u_prof, dtype), (ny, nx, nz))
          + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (ny, nx, nz), dtype)
          * seed)
    v3 = (jnp.broadcast_to(jnp.asarray(v_prof, dtype), (ny, nx, nz))
          + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (ny, nx, nz), dtype)
          * seed)
    w3 = jnp.zeros((ny, nx, nz + 1), dtype)
    u3, v3, w3 = sl.project(u3, v3, w3, dt=args.dt, g=g)
    tr = jnp.zeros((ny, nx, nz, args.n_tracers), dtype)
    tr = tr.at[..., 0].set(jnp.asarray(qv_prof, dtype)[None, None, :])
    st = sl.SpectralLESState(
        u=u3, v=v3, w=w3, rhs_u_prev=jnp.zeros_like(u3),
        rhs_v_prev=jnp.zeros_like(v3), rhs_w_prev=jnp.zeros_like(w3),
        theta=th3, rhs_theta_prev=jnp.zeros_like(th3),
        tracers=tr, rhs_tracers_prev=jnp.zeros_like(tr))

    # Kinematic surface fluxes from the sfc deck (constant; SFC_FLX_FXD):
    #   θ-flux = SHF/(ρ_sfc·c_pd·Π_sfc),  q_v-flux = LHF/(ρ_sfc·L_v).
    rho_sfc = float(ref.rho_c[0]); exn_sfc = float(ref.exner_c[0])
    th_flux = sfc0["shf"] / (rho_sfc * constants.c_pd * exn_sfc)
    qv_flux = sfc0["lhf"] / (rho_sfc * constants.L_v)
    forc = dict(ug=jnp.asarray(ug, dtype), vg=jnp.asarray(vg, dtype),
                w_ls=jnp.asarray(w_ls, dtype),
                tls=jnp.asarray(tls, dtype), qls=jnp.asarray(qls, dtype),
                th_flux=th_flux, qv_flux=qv_flux, sfc=sfc0)
    return g, st, ref, forc, th_prof


def make_forcing_fn(g, ref, forc, dtype):
    """Jitted large-scale forcing increment (subsidence + tls + qls + sponge).

    Subsidence: φ ← φ − dt·w_ls·∂φ/∂z on θ and q_v (GCSS BOMEX spec applies it
    to the thermodynamic scalars; w_ls < 0 ⇒ downward). One-sided derivative
    against the subsiding flow (upwind for w_ls<0 = forward difference in
    bottom-up indexing), matching SAM's upwind subsidence operator.
    tls is an ABSOLUTE-T tendency (SAM forcing.f90) → θ via the reference Exner.
    """
    dz = g.dz
    exner = jnp.asarray(ref.exner_c, dtype)                  # (nz,)
    w_ls, tls, qls = forc["w_ls"], forc["tls"], forc["qls"]

    def ddz_up(f):
        # forward difference (f[k+1]−f[k])/dz — upwind for subsidence (w<0);
        # top row clamps to 0 gradient (w_ls=0 there anyway).
        d = (f[..., 1:] - f[..., :-1]) / dz
        return jnp.concatenate([d, jnp.zeros_like(d[..., :1])], axis=-1)

    dth_dt_T = tls / exner                                   # θ tendency

    def forcing(state, dt):
        th, tr = state.theta, state.tracers
        th = th + dt * (-w_ls * ddz_up(th) + dth_dt_T)
        qv = tr[..., 0] + dt * (-w_ls * ddz_up(tr[..., 0]) + qls)
        tr = tr.at[..., 0].set(qv)
        return state._replace(theta=th, tracers=tr)

    return forcing


def moist_profiles(st, g, ref):
    """Planar-mean moist profiles + cloud diagnostics."""
    th = np.asarray(st.theta); tr = np.asarray(st.tracers)
    qv, qc, qr = tr[..., 0], tr[..., 1], tr[..., 2]
    rho = np.asarray(ref.rho_c)
    cf_z = (qc > 1.0e-5).mean(axis=(0, 1))                  # cloud frac (z)
    cc_proj = float(np.any(qc > 1.0e-5, axis=-1).mean())    # projected cover
    lwp = float((qc * rho[None, None, :]).sum(-1).mean() * g.dz) * 1e3  # g/m²
    return dict(theta=th.mean((0, 1)), qv=qv.mean((0, 1)), qc=qc.mean((0, 1)),
                qr=qr.mean((0, 1)), cloud_frac=cf_z, cloud_cover=cc_proj,
                lwp=lwp)


def main():
    args = parse_args()
    if args.micro_substeps < 1:
        raise ValueError("--micro-substeps must be >= 1")
    if args.micro_relax_factor <= 0.0:
        raise ValueError("--micro-relax-factor must be > 0")
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)
    g, st, ref, forc, th_prof = build(args, dtype)
    micro_cfg = (MicrophysicsConfig(
        scheme="morrison", morrison=MorrisonConfig(morrison_flavor="sam"))
        if args.microphysics == "morrison"
        else MicrophysicsConfig(scheme=args.microphysics))
    dt0 = (select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                     dt_cap=args.dt_max) if args.adaptive_dt
           else float(args.dt))
    micro_dt = (
        dt0 * args.micro_every / args.micro_substeps
        * args.micro_relax_factor)
    micro = make_les_microphysics_fn(micro_cfg, ref, g.dz, micro_dt)
    forcing = make_forcing_fn(g, ref, forc, dtype)
    rho_c = jnp.asarray(ref.rho_c, dtype)            # for conserving positivity

    # Rayleigh sponge (top 25%, w + fluctuations) — BOMEX dodamping: absorb
    # gravity waves at the inversion-capped lid (same pattern as the dry
    # drivers; re-project after).
    zc = g.z_c; z_sp = 0.75 * args.Lz; tau_sp = 60.0
    spc = jnp.where(zc > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zc - z_sp) / (args.Lz - z_sp))), 0.0).astype(dtype)
    zf = g.z_f
    spf = jnp.where(zf > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zf - z_sp) / (args.Lz - z_sp))), 0.0).astype(dtype)

    def _positive(tr):
        if args.positive_mode == "clip":
            q = jnp.clip(tr[..., :6], 0.0, None)
            numbers = jnp.clip(tr[..., 6:], 0.0, None)
            created = jnp.sum(jnp.clip(-tr[..., :6], 0.0, None)
                              * rho_c[None, None, :, None] * g.dz)
            return jnp.concatenate([q, numbers], axis=-1), created
        return conserving_positive(tr, rho_c, g.dz)

    def _apply_micro(state, dt):
        created_tot = jnp.asarray(0.0, state.theta.dtype)
        dt_sub = dt * args.micro_every / args.micro_substeps
        for _ in range(args.micro_substeps):
            dth, dtr, _pr = micro(state.theta, state.tracers)
            tr = state.tracers + dt_sub * dtr
            # MASS-CONSERVING positivity (NOT a plain clip): the non-monotone
            # spectral scalar transport undershoots at the moisture inversion;
            # a plain clip would convert those undershoots into spurious water
            # and drive runaway whole-column condensation. Borrow instead.
            tr, created = _positive(tr)
            state = state._replace(
                theta=state.theta + dt_sub * dth, tracers=tr)
            created_tot = created_tot + created
        return state, created_tot

    @partial(jax.jit, static_argnames=("first", "do_micro"))
    def step(state, dt, first=False, do_micro=True):
        created = jnp.asarray(0.0, state.theta.dtype)
        if args.micro_order == "pre":
            state = forcing(state, dt)
            if do_micro:
                state, created = _apply_micro(state, dt)
        state, us = sl.step(state, g=g, dt=dt,
                            u_geo=(forc["ug"], forc["vg"]), f_cor=_FCOR,
                            first=first, force=(0.0, 0.0),
                            sfc_theta_flux=forc["th_flux"],
                            sfc_qv_flux=forc["qv_flux"])
        if args.micro_order == "post":
            state = forcing(state, dt)
            if do_micro:
                state, created = _apply_micro(state, dt)
        rc = (dt / tau_sp) * spc
        rf = (dt / tau_sp) * spf
        u = state.u - rc * (state.u - state.u.mean((0, 1), keepdims=True))
        v = state.v - rc * (state.v - state.v.mean((0, 1), keepdims=True))
        w = state.w - rf * state.w
        u, v, w = sl.project(u, v, w, dt=dt, g=g)
        return state._replace(u=u, v=v, w=w), us, created

    T = args.hours * 3600.0
    n_steps = int(round(T / dt0))
    print(f"[BOMEX LES] {args.nx}x{args.ny}x{args.nz} dx={g.dx:.0f} dz={g.dz:.0f}"
          f" dt={dt0:.2f}s {args.time_scheme} "
          f"sgs={'LASD' if args.dynamic else args.sgs_model} "
          f"micro={micro.scheme_name} f={_FCOR:.2e} dtype={dtype.__name__}")
    print(f"  sfc: SHF={forc['sfc']['shf']:.1f} LHF={forc['sfc']['lhf']:.1f} "
          f"W/m² (θ-flux={forc['th_flux']:.5f} K·m/s, "
          f"q-flux={forc['qv_flux']:.2e} m/s), {n_steps} steps")

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
        _save(0.0)                       # the INITIAL state (codex (f))
    t = 0.0; i = 0
    created_tot = 0.0
    next_rec = T / args.record_frames if rec else np.inf
    t0 = time.time()
    while t < T:
        # micro fires on every micro_every-th step, never on a partial first
        # increment (codex (f): uniform cadence from step 1).
        st, us, created = step(st, dt, first=(i == 0),
                               do_micro=((i + 1) % args.micro_every == 0))
        created_tot += float(created)
        t += dt0; i += 1
        if i % args.print_every == 0:
            mw = float(jnp.max(jnp.abs(st.w)))
            if not np.isfinite(mw) or mw > 1e3:
                print(f"[BLOWUP] step {i} max|w|={mw}"); return 1
            d = moist_profiles(st, g, ref)
            print(f"{i:7d} {t/3600.0:5.2f}h max|w|={mw:5.2f} "
                  f"cc={d['cloud_cover']:.3f} LWP={d['lwp']:6.2f} g/m² "
                  f"qc_max={float(jnp.max(st.tracers[..., 1])):.2e} "
                  f"u*={float(us):.3f} clip_q={created_tot:.2e}", flush=True)
        if rec and t >= next_rec and frame < args.record_frames:
            _save(t / 3600.0); next_rec += T / args.record_frames
    wall = time.time() - t0
    print(f"[DONE] wall={wall:.0f}s  {i/wall:.1f} steps/s")
    if rec and frame < args.record_frames:
        _save(t / 3600.0)
    d = moist_profiles(st, g, ref)
    np.savez(args.output / "bomex_les_final.npz", z=zc_np, **{
        k: v for k, v in d.items() if isinstance(v, np.ndarray)},
        cloud_cover=d["cloud_cover"], lwp=d["lwp"])
    print(f"  FINAL: cloud cover={d['cloud_cover']:.3f} (ref 0.10-0.15), "
          f"LWP={d['lwp']:.2f} g/m² (ref ~5-10), "
          f"qc_max={d['qc'].max():.2e}")
    print(f"  profiles -> {args.output}/bomex_les_final.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
