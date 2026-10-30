"""Generate the SDM oracle fixture by RUNNING ERF 26.09's super-droplet physics.

Oracle: erf-model/ERF tag ``26.09`` (commit
6d09aeb120eb4eac388106004bfaaa7f795a2e26) — ``Source/Particles/
ERF_SuperDropletPCMassChange.H`` (dRsqdt, NewtonSolver, TI be/rk4/cn/dirk212/
rk3bs), the ``MassChange_LV`` per-particle body, ``ERF_SuperDropletPCCoalescence
.H/.cpp`` (CollisionKernel, coalescence_rate, coal_update_attribs),
``MaterialProperties/ERF_TerminalVelocity.H`` and
``ERF_MaterialProperties.H`` (water coefficients), ``Utils/
ERF_MicrophysicsUtils.H`` (erf_esatw) — compiled by ``build.sh`` (g++, double,
-O0 -ffp-contract=off) into ``sdm_driver``.

Usage (CPU node)::

    bash scripts/validate/oracle/sdm/build.sh <ERF @ 6d09aeb1> <build dir>
    python scripts/validate/oracle/sdm/make_fixture.py --driver <build dir>/sdm_driver \\
        --erf <ERF @ 6d09aeb1> \\
        --out tests/atmosphere/microphysics/oracle_fixtures/sdm_erf.npz

Inputs are synthetic but span the regimes the scheme sees: haze to drizzle
radii (30 nm - 6 mm), sub/supersaturation 0.95-1.01, 240-300 K, NaCl solute
0-1e-16 kg, dt 1-60 s, and every ERF integrator at ERF's default solver
parameters (cfl 1000, Newton 1e-6/1e-99/1e-12/10) plus the explicit ones at
cfl 1 (many sub-steps).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ORACLE = dict(repo="erf-model/ERF", tag="26.09",
              sha="6d09aeb120eb4eac388106004bfaaa7f795a2e26")
STATES = np.array([[1.15, 1.0e5, 300.0], [0.80, 7.0e4, 275.0],
                   [0.45, 3.5e4, 240.0]])             # rho [kg/m3], p [Pa], T [K]
METHODS = {"be": 0, "rk4_adaptive": 1, "cn": 2, "dirk2": 3, "rk3bs": 4}
ERF_SOLVER = dict(cfl=1000.0, rtol=1.0e-6, atol=1.0e-99, stol=1.0e-12, maxits=10)
FOUR_THIRDS_PI_RHO = 4.0 / 3.0 * math.pi * 1000.0


def _fmt(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


def _run(driver, text):
    r = subprocess.run([driver], input=text, capture_output=True, text=True, check=True)
    return np.array([float(x) for x in r.stdout.split()])


def vterm(driver, radii, rho, p, T):
    txt = "VTERM\n%d\n" % radii.size + "\n".join(
        _fmt([r, a, b, c]) for r, a, b, c in zip(radii, rho, p, T))
    return _run(driver, txt).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--erf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sha = subprocess.run(["git", "-C", a.erf, "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert sha == ORACLE["sha"], sha
    d = a.driver
    fx = {}
    rng = np.random.default_rng(20261024)

    # --- terminal velocities: 60 radii x 3 states --------------------------
    radii = np.geomspace(1.0e-7, 6.0e-3, 60)
    R_, S_ = np.meshgrid(radii, np.arange(3), indexing="ij")
    st = STATES[S_.ravel()]
    o = vterm(d, R_.ravel(), st[:, 0], st[:, 1], st[:, 2])
    fx.update(vt_r=R_.ravel(), vt_rho=st[:, 0], vt_p=st[:, 1], vt_T=st[:, 2],
              vt_rogers_yau=o[:, 0], vt_atlas_ulbrich=o[:, 1], vt_shima=o[:, 2])

    # --- collision kernels: 20x20 radius pairs x 2 states --------------------
    rr = np.geomspace(2.0e-7, 3.0e-3, 20)
    R1, R2, K = np.meshgrid(rr, rr, np.arange(2), indexing="ij")
    R1, R2, K = R1.ravel(), R2.ravel(), K.ravel()
    st = STATES[K]
    v1 = vterm(d, R1, st[:, 0], st[:, 1], st[:, 2])[:, 2]
    v2 = vterm(d, R2, st[:, 0], st[:, 1], st[:, 2])[:, 2]
    m1 = FOUR_THIRDS_PI_RHO * R1 ** 3
    m2 = FOUR_THIRDS_PI_RHO * R2 ** 3
    txt = "KERNEL\n%d\n" % R1.size + "\n".join(
        _fmt(x) for x in zip(R1, R2, v1, v2, m1, m2, st[:, 1], st[:, 2]))
    o = _run(d, txt).reshape(-1, 5)
    fx.update(k_r1=R1, k_r2=R2, k_v1=v1, k_v2=v2, k_m1=m1, k_m2=m2, k_p=st[:, 1],
              k_T=st[:, 2], k_golovin=o[:, 0], k_long=o[:, 1], k_hall=o[:, 2],
              k_sedimentation=o[:, 3], k_brownian=o[:, 4])

    # --- erf_esatw over both branches ----------------------------------------
    T = np.concatenate([np.linspace(150.0, 360.0, 211),
                        273.16 + np.array([-70.0, -69.999999, 69.999999, 70.0])])
    fx.update(esat_T=T, esat=_run(d, "ESATW\n%d\n" % T.size + "\n".join(repr(float(t)) for t in T)))

    # --- growth-ODE right-hand side and Jacobian ------------------------------
    r = np.geomspace(5.0e-8, 2.0e-3, 30)
    cases = [(S, T, p, ms) for S in (0.98, 1.0, 1.002, 1.01)
             for (T, p) in ((300.0, 1.0e5), (275.0, 7.0e4), (250.0, 5.0e4))
             for ms in (0.0, 1.0e-18)]
    rows = [(ri * ri, S, T, p, ms) for (S, T, p, ms) in cases for ri in r]
    rows = np.array(rows)
    o = _run(d, "DRSQ\n%d\n" % len(rows) + "\n".join(_fmt(x) for x in rows)).reshape(-1, 2)
    fx.update(drsq_u=rows[:, 0], drsq_S=rows[:, 1], drsq_T=rows[:, 2], drsq_p=rows[:, 3],
              drsq_msol=rows[:, 4], drsq_rhs=o[:, 0], drsq_jac=o[:, 1])

    # --- MassChange_LV: every integrator, ERF defaults + explicit at cfl 1 ----
    r0 = np.array([3.0e-8, 1.0e-7, 5.0e-7, 1.0e-6, 2.0e-6, 8.0e-6, 2.0e-5, 6.0e-5,
                   2.0e-4, 1.0e-3])
    msol = np.array([1.0e-19, 1.0e-18, 1.0e-18, 0.0, 1.0e-17, 0.0, 1.0e-16, 0.0,
                     1.0e-18, 0.0])
    groups = []
    for dt, S, T, p in ((10.0, 1.003, 290.0, 9.0e4), (10.0, 0.95, 285.0, 8.5e4),
                        (60.0, 1.0005, 275.0, 7.0e4), (1.0, 0.999, 300.0, 1.0e5),
                        (60.0, 0.90, 260.0, 5.0e4), (5.0, 1.01, 250.0, 4.5e4)):
        Sp = S + 1.0e-4 * rng.standard_normal(r0.size)        # per-particle S, T
        Tp = T + 0.5 * rng.standard_normal(r0.size)
        groups.append((dt, Sp, Tp, np.full(r0.size, p)))
    # The explicit TI methods at ERF's default cfl 1000 run far outside their
    # stability region and amplify round-off to O(1): ERF -O0 vs ERF -O2/FMA
    # differ by 0.60 (rk4) / 0.96 (rk3bs) relative on these groups, so they
    # are not a reproducible oracle output.  They are run at cfl 1 instead.
    runs = [(m, ERF_SOLVER["cfl"]) for m in ("be", "cn", "dirk2")] + [
        ("rk4_adaptive", 1.0), ("rk3bs", 1.0)]
    mc = {}
    for gi, (dt, Sp, Tp, pp) in enumerate(groups):
        mc[f"mc{gi}_dt"] = np.array(dt)
        mc[f"mc{gi}_S"], mc[f"mc{gi}_T"], mc[f"mc{gi}_p"] = Sp, Tp, pp
        for meth, cfl in runs:
            hdr = "MASSCHANGE\n%d %s %d\n%d\n" % (
                METHODS[meth], _fmt([cfl, ERF_SOLVER["rtol"], ERF_SOLVER["atol"],
                                     ERF_SOLVER["stol"]]), ERF_SOLVER["maxits"], r0.size)
            o = _run(d, hdr + "\n".join(_fmt(x) for x in zip(r0, msol, Sp, Tp, pp,
                                                             np.full(r0.size, dt))))
            o = o.reshape(-1, 2)
            tag = f"mc{gi}_{meth}_cfl{int(cfl)}"
            mc[tag + "_r"], mc[tag + "_ok"] = o[:, 0], o[:, 1]
    fx.update(mc_r0=r0, mc_msol=msol, mc_ngroups=np.array(len(groups)),
              mc_runs=np.array([f"{m}_cfl{int(c)}" for m, c in runs]), **mc)

    # --- coalescence: injected pairs + uniforms, 4 kernels x Brownian on/off -
    n = 64
    rad = np.exp(rng.normal(np.log(12e-6), 0.6, n))
    rad[:6] = [80e-6, 150e-6, 300e-6, 600e-6, 1.2e-3, 2.4e-3]
    rad[6:9] = [0.05e-6, 0.1e-6, 0.3e-6]                   # haze for the Brownian term
    mult = np.floor(10 ** rng.uniform(3, 9, n))
    mult[10] = mult[11] = 4.0e6                    # equal multiplicities -> split branch
    mult[12] = 2 * mult[13]                        # exact multiple -> zero remainder
    sol = np.where(np.arange(n) % 3 == 0, 1.0e-18, 0.0)
    perm = rng.permutation(n)
    ia, ib = perm[0::2], perm[1::2]
    U = rng.uniform(size=n // 2)
    rho, p, T = STATES[1]
    V, dtc = 1.0e-6, 1.0
    vt = vterm(d, rad, np.full(n, rho), np.full(n, p), np.full(n, T))[:, 2]
    mass = FOUR_THIRDS_PI_RHO * rad ** 3
    fx.update(c_mult=mult, c_rad=rad, c_sol=sol, c_ia=ia, c_ib=ib, c_U=U,
              c_state=np.array([rho, p, T, V, dtc]))
    for kname, kc in (("golovin", 0), ("hall", 1), ("sedimentation", 2), ("long", 3)):
        for brown in (0, 1):
            txt = "COAL\n%d %d %s %d\n" % (kc, brown, _fmt([V, dtc, p, T]), n)
            txt += "\n".join(_fmt(x) for x in zip(mult, rad, mass, vt, sol))
            txt += "\n%d\n" % ia.size + "\n".join(
                "%d %d %r" % (x, y, float(u)) for x, y, u in zip(ia, ib, U))
            o = _run(d, txt).reshape(n, 4)
            tag = f"c_{kname}_b{brown}"
            fx[tag + "_mult"], fx[tag + "_rad"], fx[tag + "_sol"] = o[:, 0], o[:, 1], o[:, 3]

    meta = dict(ORACLE, generator="scripts/validate/oracle/sdm/make_fixture.py",
                driver="scripts/validate/oracle/sdm/sdm_driver.cpp",
                build="g++ -std=c++17 -O0 -ffp-contract=off (double)",
                compiler=subprocess.run(["g++", "--version"], capture_output=True,
                                        text=True).stdout.splitlines()[0],
                erf_solver=ERF_SOLVER)
    fx["meta"] = np.array(json.dumps(meta))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
