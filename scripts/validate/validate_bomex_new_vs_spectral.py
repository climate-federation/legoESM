"""BOMEX shallow-cumulus: NEW pseudo-incompressible dycore vs the FORMER (pseudo-spectral)
LES core, SAME gSAM BOMEX forcing + SAME microphysics. Compares the GCSS moist diagnostics
(cloud cover, LWP, cloud base/top, mean θ/q_v/q_c profiles).

BOMEX (Siebesma 2003 / gSAM CASES/BOMEX): trade-wind shallow cu; fixed surface SHF/LHF,
height-dependent geostrophic wind, subsidence + radiative-cooling + drying large-scale
forcing, Coriolis f=0.376e-4. Reference (hrs 3-6): cloud cover ~10-15%, LWP ~5-10 g/m².

REDUCED by default for affordable compute (this is a cross-dycore CONSISTENCY + physical-
realism check, not a full 6 h GCSS submission). Set --hours/--nx/--nz for a fuller run.

Run (GPU):
  LEGOESM_GSAM_ROOT=/home/gentine/Documents/Code/gSAM/gsam1.8.7/gSAM1.8.7 \
  JAX_PLATFORMS=cuda .venv/bin/python scripts/validate/validate_bomex_new_vs_spectral.py --f32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402
if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_plane as pin  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.dynamics.les.spectral_les_moist import (  # noqa: E402
    make_anelastic_reference, make_les_microphysics_fn)
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig  # noqa: E402
from legoesm.atmosphere.sam_case_forcing import (  # noqa: E402
    read_sam_snd, read_sam_lsf, read_sam_sfc, surface_at_day)

_FCOR = 0.376e-4


def _bomex_inputs(args, dtype, z_c, z_f):
    root = os.environ.get("LEGOESM_GSAM_ROOT", "")
    case = f"{root}/CASES/BOMEX"
    snd = read_sam_snd(f"{case}/snd")
    lsf = read_sam_lsf(f"{case}/lsf")
    sfc0 = surface_at_day(read_sam_sfc(f"{case}/sfc"), day=0.0)
    zc = np.asarray(z_c)
    zs = np.asarray(snd.z)
    th = np.interp(zc, zs, np.asarray(snd.theta))
    qv = np.interp(zc, zs, np.asarray(snd.q_v))
    u = np.interp(zc, zs, np.asarray(snd.u))
    v = np.interp(zc, zs, np.asarray(snd.v))
    blk = lambda a: (np.asarray(a)[0] if np.asarray(a).ndim == 2 else np.asarray(a))
    zl = blk(lsf.z)                                         # day-0 forcing block
    interp = lambda a: np.nan_to_num(np.interp(zc, zl, blk(a)))  # noqa: E731
    ug = interp(lsf.u_ls); vg = interp(lsf.v_ls)
    w_ls = interp(lsf.w_ls); T_ls = interp(lsf.T_ls); qls = interp(lsf.qv_ls)
    ref = make_anelastic_reference(z_c, z_f, float(snd.pres0) * 100.0, th, qv, dtype=dtype)
    exner = np.asarray(ref.exner_c)
    dth_dt = T_ls / exner                                   # absolute-T tendency → θ
    rho_sfc = float(ref.rho_c[0]); exn_sfc = float(ref.exner_c[0])
    th_flux = float(sfc0["shf"]) / (rho_sfc * constants.c_pd * exn_sfc)
    qv_flux = float(sfc0["lhf"]) / (rho_sfc * constants.L_v)
    return dict(th=th, qv=qv, u=u, v=v, ug=ug, vg=vg, w_ls=w_ls, dth_dt=dth_dt,
                qls=qls, th_flux=th_flux, qv_flux=qv_flux, ref=ref,
                p_sfc=float(snd.pres0) * 100.0)


def _seed(shape, key, z_c, dtype):
    seed = (jnp.asarray(z_c) < 600.0).astype(dtype)
    return 0.1 * jax.random.normal(key, shape, dtype) * seed


def _profiles(theta, tracers, rho_c, dz):
    th = np.asarray(theta); qv = np.asarray(tracers[..., 0]); qc = np.asarray(tracers[..., 1])
    cf = (qc > 1e-6).mean((0, 1))
    cc = float((qc > 1e-6).any(-1).mean())
    lwp = float((qc * np.asarray(rho_c)[None, None, :]).sum(-1).mean() * dz) * 1e3
    return dict(theta=th.mean((0, 1)), qv=qv.mean((0, 1)), qc=qc.mean((0, 1)),
                cloud_frac=cf, cloud_cover=cc, lwp=lwp)


def run_new(args, dtype, inp):
    nx, ny, nz = args.nx, args.ny, args.nz
    cfg = pin.PseudoIncompressibleConfig(
        nx=nx, ny=ny, nz=nz, Lx=args.Lx, Ly=args.Lx, Lz=args.Lz, theta_ref0=float(inp["th"][0]),
        scheme="weno5", moist=True, n_tracers=3, sgs="vreman", surface="flux",
        sfc_theta_flux=inp["th_flux"], sfc_qv_flux=inp["qv_flux"], f_cor=_FCOR,
        nu_floor=args.nu_floor, hyperdiff_coeff=args.hyperdiff, poisson_maxiter=args.maxiter)
    g = pin.make_grid(cfg, dtype=dtype)
    sh = (ny, nx, nz)
    th3 = jnp.broadcast_to(jnp.asarray(inp["th"], dtype), sh) + _seed(sh, jax.random.PRNGKey(0), g.z_c, dtype)
    u3 = jnp.broadcast_to(jnp.asarray(inp["u"], dtype), sh) + _seed(sh, jax.random.PRNGKey(1), g.z_c, dtype)
    v3 = jnp.broadcast_to(jnp.asarray(inp["v"], dtype), sh) + _seed(sh, jax.random.PRNGKey(2), g.z_c, dtype)
    w3 = jnp.zeros((ny, nx, nz + 1), dtype)
    tr = jnp.zeros(sh + (3,), dtype).at[..., 0].set(jnp.asarray(inp["qv"], dtype)[None, None, :])
    u3, v3, w3, _ = pin.project(u3, v3, w3, th3, tr, jnp.zeros(sh, dtype), args.dt, g)
    st = pin.PseudoIncompressibleState(u=u3, v=v3, w=w3, theta=th3, pi_prev=jnp.zeros(sh, dtype), tracers=tr)
    forcing = pin.PseudoIncompressibleForcing(
        subsidence_w=jnp.asarray(inp["w_ls"], dtype), dtheta_dt_ls=jnp.asarray(inp["dth_dt"], dtype),
        dqv_dt_ls=jnp.asarray(inp["qls"], dtype), ug_prof=jnp.asarray(inp["ug"], dtype),
        vg_prof=jnp.asarray(inp["vg"], dtype))
    micro = pin.make_microphysics(g, MicrophysicsConfig(scheme="kessler"), args.dt,
                                  p_sfc=inp["p_sfc"], qv_prof=jnp.asarray(inp["qv"], dtype))

    @jax.jit
    def one(state):
        s2 = pin.step(state, g, jnp.asarray(args.dt, dtype), forcing)
        dth, dtr, _ = micro(s2.theta, s2.tracers)
        tr2 = jnp.maximum(s2.tracers + args.dt * dtr, 0.0)
        return s2._replace(theta=s2.theta + args.dt * dth, tracers=tr2)

    nsteps = int(args.hours * 3600 / args.dt)
    for _ in range(nsteps):
        st = one(st)
    jax.block_until_ready(st.u)
    return _profiles(st.theta, st.tracers, inp["ref"].rho_c, g.dz)


def run_spectral(args, dtype, inp):
    nx, ny, nz = args.nx, args.ny, args.nz
    cfg = sl.SpectralLESConfig(
        nx=nx, ny=ny, nz=nz, Lx=args.Lx, Ly=args.Lx, Lz=args.Lz, z0=0.1, dealias=True,
        c_s=0.18, smagorinsky_dynamic=False, sgs_model="vreman", time_scheme="rk3",
        buoyancy=True, theta_ref0=float(inp["th"][0]), pr_sgs=1.0, nu_floor=args.nu_floor,
        moist=True, monotone_scalars=True, n_tracers=3, scalar_advection="weno5")
    g = sl.make_grid(cfg, dtype=dtype)
    sh = (ny, nx, nz)
    th3 = jnp.broadcast_to(jnp.asarray(inp["th"], dtype), sh) + _seed(sh, jax.random.PRNGKey(0), g.z_c, dtype)
    u3 = jnp.broadcast_to(jnp.asarray(inp["u"], dtype), sh) + _seed(sh, jax.random.PRNGKey(1), g.z_c, dtype)
    v3 = jnp.broadcast_to(jnp.asarray(inp["v"], dtype), sh) + _seed(sh, jax.random.PRNGKey(2), g.z_c, dtype)
    w3 = jnp.zeros((ny, nx, nz + 1), dtype)
    u3, v3, w3 = sl.project(u3, v3, w3, dt=args.dt, g=g)
    tr = jnp.zeros(sh + (3,), dtype).at[..., 0].set(jnp.asarray(inp["qv"], dtype)[None, None, :])
    st = sl.SpectralLESState(u=u3, v=v3, w=w3, rhs_u_prev=jnp.zeros_like(u3),
                             rhs_v_prev=jnp.zeros_like(v3), rhs_w_prev=jnp.zeros_like(w3),
                             theta=th3, rhs_theta_prev=jnp.zeros_like(th3),
                             tracers=tr, rhs_tracers_prev=jnp.zeros_like(tr))
    ref = inp["ref"]
    micro = make_les_microphysics_fn(MicrophysicsConfig(scheme="kessler"), ref, g.dz, args.dt)
    exner = jnp.asarray(ref.exner_c, dtype)
    w_ls = jnp.asarray(inp["w_ls"], dtype); dth = jnp.asarray(inp["dth_dt"], dtype)
    qls = jnp.asarray(inp["qls"], dtype)
    ug = jnp.asarray(inp["ug"], dtype); vg = jnp.asarray(inp["vg"], dtype)

    def ddz_up(f):
        d = (f[..., 1:] - f[..., :-1]) / g.dz
        return jnp.concatenate([d, jnp.zeros_like(d[..., :1])], axis=-1)

    @jax.jit
    def one(state):
        # spectral step: geostrophic via u_geo profile mean (spectral uses scalar u_geo)
        s2, _us = sl.step(state, g, args.dt, (ug, vg), _FCOR,
                          sfc_theta_flux=inp["th_flux"], sfc_qv_flux=inp["qv_flux"])
        th_n = s2.theta - args.dt * w_ls[None, None, :] * ddz_up(s2.theta) + args.dt * dth[None, None, :]
        qv_n = s2.tracers[..., 0] - args.dt * w_ls[None, None, :] * ddz_up(s2.tracers[..., 0]) + args.dt * qls[None, None, :]
        tr_f = s2.tracers.at[..., 0].set(qv_n)
        dthm, dtrm, _ = micro(th_n, tr_f)
        tr2 = jnp.maximum(tr_f + args.dt * dtrm, 0.0)
        return s2._replace(theta=th_n + args.dt * dthm, tracers=tr2)

    nsteps = int(args.hours * 3600 / args.dt)
    for _ in range(nsteps):
        st = one(st)
    jax.block_until_ready(st.u)
    return _profiles(st.theta, st.tracers, ref.rho_c, g.dz)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=48)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--nz", type=int, default=75)
    p.add_argument("--Lx", type=float, default=4800.0)
    p.add_argument("--Lz", type=float, default=3000.0)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--nu-floor", type=float, default=0.1)
    p.add_argument("--hyperdiff", type=float, default=0.0,
                   help="new-dycore horizontal biharmonic de-noiser coeff [m⁴/s], 0=off")
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--f32", action="store_true")
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    # shared grid heights (same Lz/nz for both dycores)
    dz = args.Lz / args.nz
    z_c = (jnp.arange(args.nz, dtype=dtype) + 0.5) * dz
    z_f = jnp.arange(args.nz + 1, dtype=dtype) * dz
    inp = _bomex_inputs(args, dtype, z_c, z_f)
    print(f"BOMEX new-vs-spectral: nx={args.nx} nz={args.nz} hours={args.hours} "
          f"dt={args.dt} dtype={'f32' if args.f32 else 'f64'} "
          f"(th_flux={inp['th_flux']:.4f} qv_flux={inp['qv_flux']:.2e})")
    new = run_new(args, dtype, inp)
    print(f"  NEW:      cloud_cover={new['cloud_cover']:.3f}  LWP={new['lwp']:.2f} g/m²  "
          f"qc_max={1e3*new['qc'].max():.3f} g/kg")
    spec = run_spectral(args, dtype, inp)
    print(f"  SPECTRAL: cloud_cover={spec['cloud_cover']:.3f}  LWP={spec['lwp']:.2f} g/m²  "
          f"qc_max={1e3*spec['qc'].max():.3f} g/kg")
    # cross-dycore profile consistency
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))  # noqa: E731
    print(f"  profile rel-diff: theta={rel(new['theta'],spec['theta']):.3f} "
          f"qv={rel(new['qv'],spec['qv']):.3f} qc={rel(new['qc'],spec['qc']):.3f}")
    print(f"  reference targets (hrs 3-6): cloud cover ~0.10-0.15, LWP ~5-10 g/m²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
