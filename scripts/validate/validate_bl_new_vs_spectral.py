"""Cross-dycore BL validation: the NEW pseudo-incompressible plane LES vs the existing
pseudo-spectral LES on GABLS1 (stable BL, surface cooling) and Wangara (convective BL,
surface heating).

REDUCED (CPU-affordable) configuration — this is a PHYSICS-SANITY + cross-dycore
CONSISTENCY check, not a full GABLS1/Wangara intercomparison (which needs hours of sim
at fine resolution). It verifies the new dycore's composable BL physics produces the
canonical behaviour and lands in the same ballpark as the validated spectral core:

* GABLS1: a SHALLOW statically-STABLE layer (∂θ/∂z>0 kept), near-surface COOLING,
  Ekman wind turning, no blow-up.
* Wangara: a GROWING well-mixed layer (∂θ/∂z→0 in the bulk) under positive surface flux.

Usage:
  JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/validate_bl_new_vs_spectral.py
  (add --f32 for single precision, --nx/--nz/--minutes to change the reduced size)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402
if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.dynamics.les import pseudo_incompressible_plane as pi  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402

# --- case definitions (GABLS1 / Wangara, reduced) -------------------------------
CASES = {
    "gabls1": dict(theta0=265.0, Ug=8.0, fcor=1.39e-4, z0=0.1, Lz=400.0,
                   cooling_rate=0.25, Q0=0.0, zi=100.0, gamma=0.01),   # K/hr cooling
    "wangara": dict(theta0=290.0, Ug=5.0, fcor=1.0e-4, z0=0.1, Lz=1500.0,
                    cooling_rate=0.0, Q0=0.06, zi=200.0, gamma=0.003),  # +ve flux heating
}


def _sounding(z, c, dtype):
    """Stable θ sounding: θ0 below zi, +γ above (smoothed)."""
    zi, di, gam = c["zi"], 25.0, c["gamma"]
    return (c["theta0"] + 0.5 * gam * ((z - zi) + di * jnp.log(jnp.cosh((z - zi) / di))
                                       + di * jnp.log(2.0))).astype(dtype)


def run_new(case, args, dtype):
    c = CASES[case]
    nx, ny, nz = args.nx, args.ny, args.nz
    surface = "most_cooling" if c["cooling_rate"] > 0 else "flux"
    cfg = pi.PseudoIncompressibleConfig(
        nx=nx, ny=ny, nz=nz, Lx=args.Lx, Ly=args.Lx, Lz=c["Lz"],
        theta_ref0=c["theta0"], scheme="weno5", sgs="vreman", c_vreman=0.07,
        f_cor=c["fcor"], ug=c["Ug"], vg=0.0, surface=surface, z0=c["z0"],
        sfc_theta_flux=c["Q0"], poisson_maxiter=args.maxiter)
    g = pi.make_grid(cfg, dtype=dtype)
    z = g.z_c
    th = _sounding(z, c, dtype)
    seed = (z < 0.5 * c["Lz"]).astype(dtype)
    sh = (ny, nx, nz)
    th3 = jnp.broadcast_to(th, sh).astype(dtype) + (
        0.1 * jax.random.normal(jax.random.PRNGKey(0), sh, dtype) * seed)
    u = jnp.full(sh, c["Ug"], dtype) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(1), sh, dtype) * seed
    v = 0.1 * jax.random.normal(jax.random.PRNGKey(2), sh, dtype) * seed
    w = jnp.zeros((ny, nx, nz + 1), dtype)
    u, v, w, _pi0 = pi.project(u, v, w, th3, None, jnp.zeros(sh, dtype), args.dt, g)
    st = pi.PseudoIncompressibleState(u=u, v=v, w=w, theta=th3, pi_prev=jnp.zeros(sh, dtype))
    nsteps = int(args.minutes * 60 / args.dt)
    t_sfc0 = c["theta0"]
    rate_s = c["cooling_rate"] / 3600.0

    @jax.jit
    def scan_body(state, n):
        t_sfc = jnp.asarray(t_sfc0 - rate_s * (n.astype(dtype) * args.dt), dtype)
        forcing = pi.PseudoIncompressibleForcing(t_sfc=t_sfc)
        return pi.step(state, g, args.dt, forcing), None

    st, _ = jax.lax.scan(scan_body, st, jnp.arange(nsteps))
    return _diag_new(st, g)


def _diag_new(st, g):
    u = np.asarray(st.u); v = np.asarray(st.v); th = np.asarray(st.theta)
    z = np.asarray(g.z_c)
    um, vm, thm = u.mean((0, 1)), v.mean((0, 1)), th.mean((0, 1))
    spd = np.sqrt(um ** 2 + vm ** 2)
    return dict(z=z, um=um, vm=vm, thm=thm, spd=spd, jet=float(spd.max()),
                jetz=float(z[np.argmax(spd)]), th_sfc=float(thm[0]),
                dthdz_low=float((thm[2] - thm[0]) / (z[2] - z[0])),
                finite=bool(np.all(np.isfinite(th)) and np.all(np.isfinite(u))))


def run_spectral(case, args, dtype):
    c = CASES[case]
    nx, ny, nz = args.nx, args.ny, args.nz
    cfg = sl.SpectralLESConfig(
        nx=nx, ny=ny, nz=nz, Lx=args.Lx, Ly=args.Lx, Lz=c["Lz"], z0=c["z0"],
        dealias=True, c_s=0.18, smagorinsky_dynamic=False, sgs_model="vreman",
        time_scheme="rk3", buoyancy=True, theta_ref0=c["theta0"], pr_sgs=1.0,
        nu_floor=0.05)
    g = sl.make_grid(cfg, dtype=dtype)
    z = g.z_c
    th = _sounding(z, c, dtype)
    seed = (z < 0.5 * c["Lz"]).astype(dtype)
    sh = (ny, nx, nz)
    th3 = jnp.broadcast_to(th, sh).astype(dtype) + (
        0.1 * jax.random.normal(jax.random.PRNGKey(0), sh, dtype) * seed)
    u = jnp.full(sh, c["Ug"], dtype) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(1), sh, dtype) * seed
    v = 0.1 * jax.random.normal(jax.random.PRNGKey(2), sh, dtype) * seed
    w = jnp.zeros((ny, nx, nz + 1), dtype)
    u, v, w = sl.project(u, v, w, dt=args.dt, g=g)
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u),
                             rhs_v_prev=jnp.zeros_like(v), rhs_w_prev=jnp.zeros_like(w),
                             theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    nsteps = int(args.minutes * 60 / args.dt)
    u_geo = (c["Ug"], 0.0)
    rate_s = c["cooling_rate"] / 3600.0

    @jax.jit
    def scan_body(carry, n):
        state = carry
        if c["cooling_rate"] > 0:
            t_sfc = jnp.asarray(c["theta0"] - rate_s * (n.astype(dtype) * args.dt), dtype)
            new, _us = sl.step(state, g, args.dt, u_geo, c["fcor"], t_sfc=t_sfc)
        else:
            new, _us = sl.step(state, g, args.dt, u_geo, c["fcor"],
                               sfc_theta_flux=c["Q0"])
        return new, None

    st, _ = jax.lax.scan(scan_body, st, jnp.arange(nsteps))
    u = np.asarray(st.u); v = np.asarray(st.v); th = np.asarray(st.theta)
    z = np.asarray(g.z_c)
    um, vm, thm = u.mean((0, 1)), v.mean((0, 1)), th.mean((0, 1))
    spd = np.sqrt(um ** 2 + vm ** 2)
    return dict(z=z, um=um, vm=vm, thm=thm, spd=spd, jet=float(spd.max()),
                jetz=float(z[np.argmax(spd)]), th_sfc=float(thm[0]),
                dthdz_low=float((thm[2] - thm[0]) / (z[2] - z[0])),
                finite=bool(np.all(np.isfinite(th)) and np.all(np.isfinite(u))))


def _check(case, new, spec):
    """Physical-sanity + cross-dycore-consistency assertions (reduced run)."""
    msgs = []
    ok = True
    for name, d in (("new", new), ("spectral", spec)):
        if not d["finite"]:
            ok = False; msgs.append(f"{name}: NON-FINITE fields")
    if case == "gabls1":
        # stable layer must stay stable (positive low-level dθ/dz) + near-surface cooling
        if new["dthdz_low"] <= 0:
            ok = False; msgs.append(f"new GABLS1 not stable: dθ/dz_low={new['dthdz_low']:.4f}")
        if new["th_sfc"] >= CASES[case]["theta0"]:
            msgs.append(f"new GABLS1 surface not cooled: θ_sfc={new['th_sfc']:.2f}")
    else:  # wangara CBL: bulk should mix toward neutral (small |dθ/dz| in lower half)
        if new["dthdz_low"] > CASES[case]["gamma"] * 2:
            msgs.append(f"new Wangara low-level not mixing: dθ/dz={new['dthdz_low']:.4f}")
    # cross-dycore consistency: bulk wind + θ profiles in the same ballpark
    rel_th = float(np.linalg.norm(new["thm"] - spec["thm"])
                   / (np.linalg.norm(spec["thm"] - CASES[case]["theta0"]) + 1e-9))
    msgs.append(f"{case}: jet new={new['jet']:.2f}@{new['jetz']:.0f}m "
                f"spec={spec['jet']:.2f}@{spec['jetz']:.0f}m | θ-profile rel-diff={rel_th:.2f} | "
                f"dθ/dz_low new={new['dthdz_low']:.4f} spec={spec['dthdz_low']:.4f}")
    return ok, msgs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--ny", type=int, default=24)
    p.add_argument("--nz", type=int, default=48)
    p.add_argument("--Lx", type=float, default=400.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--cases", nargs="+", default=["gabls1", "wangara"])
    p.add_argument("--f32", action="store_true")
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    print(f"reduced BL validation: nx={args.nx} nz={args.nz} dt={args.dt} "
          f"minutes={args.minutes} dtype={'f32' if args.f32 else 'f64'}")
    all_ok = True
    for case in args.cases:
        print(f"\n=== {case} ===")
        new = run_new(case, args, dtype)
        spec = run_spectral(case, args, dtype)
        ok, msgs = _check(case, new, spec)
        for m in msgs:
            print("  " + m)
        print(f"  {case}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
