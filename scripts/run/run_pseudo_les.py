"""Dry-ABL LES driver for the pseudo-incompressible plane dycore.

The third plane LES core (`pseudo_incompressible_plane`) had unit tests + the
cross-dycore validator `scripts/validate/validate_bl_new_vs_spectral.py` but no
`scripts/run` driver. This is that driver — parity with `run_spectral_les.py`
(spectral core) and `run_les_plane.py` (compressible core) so the same dry-ABL
suite runs on all three "grid types".

Cases (dry): neutral / ekman (geostrophic neutral BL), gabls1 (stable, surface
cooling), wangara (convective, +ve surface flux). Case setup mirrors the
validated `validate_bl_new_vs_spectral.py::run_new`; the pseudo core has NO
constant-pressure-gradient body-force hook, so the neutral channel is driven by
Coriolis + geostrophic wind (i.e. neutral == geostrophic Ekman here — a genuine
limitation vs the spectral core's `f_x=u*^2/Lz` channel).

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_pseudo_les.py \
       --case ekman --nx 64 --ny 64 --nz 64 --hours 1.0 --f32 \
       --output results/pseudo_ekman

Saves ``final_profiles.npz`` (z, mean U/V/theta, resolved uu/vv/ww/tke, uw/vw
flux, u_*) so it drops into the same downstream tooling as the other cores.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402

if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_plane as pip  # noqa: E402
from legoesm.atmosphere.dynamics.les.spectral_les_plane import most_surface_flux  # noqa: E402

# --- dry-ABL cases -----------------------------------------------------------
# theta0 [K], Ug [m/s], fcor [1/s], z0 [m], Lz [m], cooling_rate [K/hr],
# Q0 [K m/s] kinematic surface heat flux, zi/gamma sounding (inversion), surface.
# gabls1/wangara mirror validate_bl_new_vs_spectral.py::CASES.
_CASES = {
    "neutral": dict(theta0=300.0, Ug=10.0, fcor=1.0e-4, z0=0.1, Lz=1000.0,
                    cooling_rate=0.0, Q0=0.0, zi=600.0, gamma=0.003,
                    surface="flux", nu_floor=0.0, dt=0.4, hours=1.0),
    "ekman": dict(theta0=300.0, Ug=10.0, fcor=1.0e-4, z0=0.1, Lz=1500.0,
                  cooling_rate=0.0, Q0=0.0, zi=1500.0, gamma=0.0,
                  surface="flux", nu_floor=0.0, dt=0.4, hours=1.0),
    "gabls1": dict(theta0=265.0, Ug=8.0, fcor=1.39e-4, z0=0.1, Lz=400.0,
                   cooling_rate=0.25, Q0=0.0, zi=100.0, gamma=0.01,
                   surface="most_cooling", nu_floor=0.05, dt=0.2, hours=1.0),
    "wangara": dict(theta0=290.0, Ug=5.0, fcor=1.0e-4, z0=0.1, Lz=1500.0,
                    cooling_rate=0.0, Q0=0.06, zi=200.0, gamma=0.003,
                    surface="flux", nu_floor=0.0, dt=0.4, hours=1.0),
}


def _sounding(z, c, dtype):
    """Stable theta sounding: theta0 below zi, +gamma above (smoothed).

    Same smoothed-inversion form as validate_bl_new_vs_spectral._sounding
    (gamma=0 -> uniform theta = neutral)."""
    zi, di, gam = c["zi"], 25.0, c["gamma"]
    return (c["theta0"] + 0.5 * gam * ((z - zi) + di * jnp.log(jnp.cosh((z - zi) / di))
                                       + di * jnp.log(2.0))).astype(dtype)


def build(case, args, dtype):
    c = dict(_CASES[case])                     # per-run copy — never mutate the global
    if getattr(args, "dt", None) is not None:
        c["dt"] = args.dt
    nx, ny, nz = args.nx, args.ny, args.nz
    cfg = pip.PseudoIncompressibleConfig(
        nx=nx, ny=ny, nz=nz, Lx=args.Lx, Ly=args.Lx, Lz=c["Lz"],
        theta_ref0=c["theta0"], scheme=getattr(args, "scheme", "weno5"),
        momentum_scheme=getattr(args, "momentum_scheme", None),
        sgs=args.sgs, c_vreman=0.07, c_s=0.16,
        nu_floor=(args.nu_floor if args.nu_floor is not None else c["nu_floor"]),
        hyperdiff_coeff=getattr(args, "hyperdiff", 0.0),
        shapiro_coeff=getattr(args, "shapiro", 0.0),
        shapiro_order=getattr(args, "shapiro_order", 1),
        momentum_shapiro_coeff=getattr(args, "momentum_shapiro", 0.0),
        momentum_shapiro_order=getattr(args, "momentum_shapiro_order", 1),
        f_cor=c["fcor"], ug=c["Ug"], vg=0.0,
        surface=c["surface"], z0=c["z0"], sfc_theta_flux=c["Q0"],
        poisson_maxiter=args.maxiter)
    g = pip.make_grid(cfg, dtype=dtype)
    z = g.z_c
    sh = (ny, nx, nz)
    seed = (z < 0.5 * c["Lz"]).astype(dtype)          # perturb only lower half
    th3 = jnp.broadcast_to(_sounding(z, c, dtype), sh).astype(dtype) + (
        args.ic_amp * jax.random.normal(jax.random.PRNGKey(0), sh, dtype) * seed)
    u = jnp.full(sh, c["Ug"], dtype) + (
        args.ic_amp * jax.random.normal(jax.random.PRNGKey(1), sh, dtype) * seed)
    v = args.ic_amp * jax.random.normal(jax.random.PRNGKey(2), sh, dtype) * seed
    w = jnp.zeros((ny, nx, nz + 1), dtype)
    # make the IC divergence-free (one projection) before the time loop
    u, v, w, _pi0 = pip.project(u, v, w, th3, None, jnp.zeros(sh, dtype), c["dt"], g)
    st = pip.PseudoIncompressibleState(u=u, v=v, w=w, theta=th3,
                                       pi_prev=jnp.zeros(sh, dtype))
    return cfg, g, st, c


def diagnose(st, g, cfg, t_sfc_final=None):
    """Planar-mean + resolved-moment profiles + wall u_* (numpy).

    Velocities are collocated to cell centres first (C-grid u/v faces + w faces
    -> centres) so variances and uw/vw fluxes come from collocated fields.
    """
    u = np.asarray(st.u); v = np.asarray(st.v); w = np.asarray(st.w)
    th = np.asarray(st.theta); z = np.asarray(g.z_c)
    uc = 0.5 * (u + np.roll(u, 1, axis=1))            # x-faces -> centres
    vc = 0.5 * (v + np.roll(v, 1, axis=0))            # y-faces -> centres
    wc = 0.5 * (w[..., :-1] + w[..., 1:])             # w-faces -> centres
    um, vm, thm = uc.mean((0, 1)), vc.mean((0, 1)), th.mean((0, 1))
    up, vp, wp = uc - um, vc - vm, wc - wc.mean((0, 1))
    uu, vv, ww = (up * up).mean((0, 1)), (vp * vp).mean((0, 1)), (wp * wp).mean((0, 1))
    uw, vw = (up * wp).mean((0, 1)), (vp * wp).mean((0, 1))
    tke = 0.5 * (uu + vv + ww)
    spd = np.sqrt(um ** 2 + vm ** 2)
    spd1 = float(np.mean(np.sqrt(uc[..., 0] ** 2 + vc[..., 0] ** 2)))
    if cfg.surface == "most_cooling" and t_sfc_final is not None:
        # faithful wall u_* = the SAME stability-corrected MOST solve the core uses
        # (neutral drag over-reads under the stable GABLS1 surface layer).
        ustar_wall = float(most_surface_flux(
            spd1, float(thm[0]), float(t_sfc_final), float(z[0]), cfg.z0,
            cfg.theta_ref0)[0])
    else:
        # neutral drag law Cd=(kappa/log(z_c0/z0))^2 (neutral/flux cases)
        cd = (constants.kappa_von_karman / np.log(z[0] / cfg.z0)) ** 2
        ustar_wall = float(np.sqrt(cd) * spd1)
    ustar_res = float((uw[0] ** 2 + vw[0] ** 2) ** 0.25)   # resolved-flux proxy
    return dict(
        z=z, um=um, vm=vm, thm=thm, spd=spd, uu=uu, vv=vv, ww=ww, tke=tke,
        uw=uw, vw=vw, ustar_wall=ustar_wall, ustar_res=ustar_res,
        max_w=float(np.abs(wc).max()), wvar_max=float(ww.max()),
        jet=float(spd.max()), jetz=float(z[np.argmax(spd)]),
        th_sfc=float(thm[0]), dthdz_low=float((thm[2] - thm[0]) / (z[2] - z[0])),
        finite=bool(np.all(np.isfinite(th)) and np.all(np.isfinite(uc))
                    and np.all(np.isfinite(w))))


def _record_frame(out, frame, t_h, label, st, g, c, Lx, h_idx, h_z, les_record):
    """Centre the C-grid velocities and hand one frame to the shared recorder."""
    u = np.asarray(st.u); v = np.asarray(st.v); w = np.asarray(st.w)
    uc = 0.5 * (u + np.roll(u, 1, axis=1))
    vc = 0.5 * (v + np.roll(v, 1, axis=0))
    wc = 0.5 * (w[..., :-1] + w[..., 1:])
    les_record.record_frame(out, frame, t_h, label, np.asarray(g.z_c),
                            uc, vc, wc, np.asarray(st.theta),
                            Lx, Lx, h_idx, h_z, c["z0"])


def run(case, args, dtype):
    cfg, g, st, c = build(case, args, dtype)
    dt = jnp.asarray(c["dt"], dtype)
    nsteps = int(args.hours * 3600.0 / c["dt"])
    t_sfc0 = c["theta0"]
    rate_s = c["cooling_rate"] / 3600.0

    @jax.jit
    def scan_body(state, n):
        if c["cooling_rate"] > 0:
            t_sfc = jnp.asarray(t_sfc0 - rate_s * (n.astype(dtype) * c["dt"]), dtype)
            forcing = pip.PseudoIncompressibleForcing(t_sfc=t_sfc)
        else:
            forcing = None
        return pip.step(state, g, dt, forcing), None

    _warm = jax.lax.scan(scan_body, st, jnp.arange(1))[0]   # trigger compile only
    _warm.u.block_until_ready()                            # (does NOT advance st)
    rec = getattr(args, "record_frames", 0)
    t0 = time.time()
    if rec > 0:
        import les_record                                  # scripts/run on sys.path[0]
        label = getattr(args, "case_label", None) or case
        h_idx, h_z = les_record.select_heights(np.asarray(g.z_c), c["Lz"])
        _record_frame(args.output, 0, 0.0, label, st, g, c, args.Lx, h_idx, h_z, les_record)
        chunk, done, frame = max(1, nsteps // rec), 0, 1
        while done < nsteps:                               # chunked scan → per-frame state
            n = min(chunk, nsteps - done)
            st, _ = jax.lax.scan(scan_body, st, jnp.arange(done, done + n))
            done += n
            _record_frame(args.output, frame, done * c["dt"] / 3600.0, label,
                          st, g, c, args.Lx, h_idx, h_z, les_record)
            frame += 1
        st.u.block_until_ready()
    else:
        st, _ = jax.lax.scan(scan_body, st, jnp.arange(nsteps))  # nsteps from the IC
        st.u.block_until_ready()
    wall = time.time() - t0
    t_sfc_final = (t_sfc0 - rate_s * max(nsteps - 1, 0) * c["dt"]
                   if c["cooling_rate"] > 0 else None)
    d = diagnose(st, g, cfg, t_sfc_final=t_sfc_final)
    d["steps_per_s"] = nsteps / wall if wall > 0 else float("nan")
    d["nsteps"] = nsteps
    d["wall_s"] = wall
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=list(_CASES), required=True)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--Lx", type=float, default=2000.0)
    p.add_argument("--dt", type=float, default=None, help="override case dt [s]")
    p.add_argument("--hours", type=float, default=None, help="override case sim hours")
    p.add_argument("--sgs", choices=["none", "smagorinsky", "vreman", "lasd"],
                   default="vreman")
    p.add_argument("--scheme", choices=["weno5", "van_leer", "upwind"],
                   default="weno5", help="SCALAR advection (weno5 needs "
                   "f64 at sharp inversions; van_leer is f32-robust)")
    p.add_argument("--momentum-scheme",
                   choices=["weno5", "weno7", "weno9", "van_leer", "upwind", "central"],
                   default=None, help="MOMENTUM advection (default: follow --scheme); "
                   "weno7/weno9 are "
                   "sharper (less upwind diffusion) than weno5 and stay stable; "
                   "'central' is non-dissipative (sharpest) but needs SGS 2Δ control")
    p.add_argument("--nu-floor", type=float, default=None,
                   help="background eddy-viscosity floor [m2/s] (default: per-case)")
    p.add_argument("--hyperdiff", type=float, default=0.0,
                   help="horizontal biharmonic de-noiser coeff [m4/s] — suppresses "
                   "the fine-res 2dx grid-noise that NaNs the buoyant cases (try 1e4-1e6)")
    p.add_argument("--shapiro", type=float, default=0.0,
                   help="per-step [1,2,1] low-pass strength s in [0,1] on theta+tracers "
                   "— CFL-unlimited de-noiser; unlocks f32 gabls1 (try 0.05-0.2)")
    p.add_argument("--shapiro-order", type=int, default=1,
                   help="Shapiro order for the scalar θ de-noiser (flat passband); "
                   "8-16 lets a stronger --shapiro suppress fine-res 2Δ θ-noise in f32")
    p.add_argument("--momentum-shapiro", type=float, default=0.0,
                   help="per-step Shapiro low-pass strength s in [0,1] on VELOCITY — "
                   "CFL-unlimited 2Δ de-noiser that makes --momentum-scheme central clean "
                   "(div-free preserved, no re-projection); try 0.2-0.5 with a high order")
    p.add_argument("--momentum-shapiro-order", type=int, default=8,
                   help="Shapiro order for the velocity de-noiser (flat passband); "
                   "8-16 avoids over-damping the resolved eddies (order 1 = [1,2,1])")
    p.add_argument("--ic-amp", type=float, default=0.1,
                   help="IC perturbation amplitude [m/s and K]")
    p.add_argument("--maxiter", type=int, default=200, help="BiCGSTAB max iters")
    p.add_argument("--f32", action="store_true")
    p.add_argument("--record-frames", type=int, default=0,
                   help="evenly-spaced frames (snapshots + profile npz) for "
                   "scripts/plot/plot_les_diagnostics.py. 0 = off (final profile only).")
    p.add_argument("--case-label", type=str, default=None,
                   help="label stored in the frames (default: the case name)")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    c = dict(_CASES[args.case])                # local copy; build() honors args.dt itself
    if args.dt is not None:
        c["dt"] = args.dt
    if args.hours is None:
        args.hours = c["hours"]
    out = args.output or Path(f"results/pseudo_{args.case}")
    out.mkdir(parents=True, exist_ok=True)

    # The buoyant cases (surface cooling / heating) are not numerically robust in
    # float32 on this core — they NaN at production-ish grids where the neutral
    # cases are fine (see docs/physics-notes/les_crossgrid_regression_2026-07.md).
    # f64 runs them finite + turbulent, matching validate_bl_new_vs_spectral.py.
    if args.f32 and (c["cooling_rate"] > 0 or c["Q0"] != 0.0) and args.shapiro == 0.0:
        print(f"  WARNING: case '{args.case}' has active buoyancy forcing; the "
              f"pseudo core's stable-BL 2Δ θ-mode is f32-fragile (NaN-prone). Add "
              f"--shapiro 0.2-0.3 (CFL-unlimited θ de-noiser), or use f64 (omit --f32).",
              file=sys.stderr)

    print(f"[pseudo-LES] case={args.case} grid={args.nx}x{args.ny}x{args.nz} "
          f"Lz={c['Lz']} dt={c['dt']} hours={args.hours} sgs={args.sgs} "
          f"dtype={'f32' if args.f32 else 'f64'} surface={c['surface']}")
    d = run(args.case, args, dtype)
    np.savez(out / "final_profiles.npz",
             z=d["z"], um=d["um"], vm=d["vm"], thm=d["thm"], spd=d["spd"],
             uu=d["uu"], vv=d["vv"], ww=d["ww"], tke=d["tke"], uw=d["uw"], vw=d["vw"],
             ustar_wall=d["ustar_wall"], ustar_res=d["ustar_res"])
    print(f"  {d['nsteps']} steps  wall={d['wall_s']:.1f}s  {d['steps_per_s']:.1f} steps/s")
    print(f"  finite={d['finite']}  max|w|={d['max_w']:.3f}  wvar_max={d['wvar_max']:.4f}")
    print(f"  u*_wall={d['ustar_wall']:.3f}  u*_res={d['ustar_res']:.3f}  "
          f"jet={d['jet']:.2f}@{d['jetz']:.0f}m  θ_sfc={d['th_sfc']:.2f}  "
          f"dθ/dz_low={d['dthdz_low']:.4f}")
    print(f"  profiles -> {out/'final_profiles.npz'}")
    return 0 if d["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
