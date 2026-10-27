"""Generate the fast-SBM oracle fixture by RUNNING the WRF v4.8.0 FAST_SBM code.

Builds inputs (synthetic cloud/rain/ice spectra on the port's mass-doubling
grid, a realistic collection kernel and fall speeds from the WRF ``run/``
tables), feeds them to the compiled oracle driver (``build.sh``, every real
promoted to double) and stores inputs + oracle outputs in one small ``.npz``.
The port is never called here.

Usage:
    python make_fixture.py --driver <build>/sbm_driver --wrf <WRF checkout> \
        --out tests/atmosphere/microphysics/oracle_fixtures/fast_sbm_wrf.npz
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import (
    COL,
    DLNR,
    mass_doubling_grid_np,
)

SHA = "06d4240ae989cc3e50af412bb472df3d9048783c"
NKR = 33
DT = 60.0
F_SI2CGS = 1.0e-9        # m^-3 kg^-1 -> cm^-3 g^-1
AA1_CGS, BB1, AA2_CGS, BB2 = 2.53e12, 5.42e3, 3.41e13, 6.13e3   # FAST_SBM DATA

# (T [K], p [Pa], q_c, N_c [1/m3], q_r, N_r [1/m3], target oracle S)
CASES = {
    "trop_deep_warm":    (293.0, 9.0e4, 1.5e-3, 80e6, 1.0e-3, 2.0e3, +3.0e-3),
    "sc_stratocu":       (285.0, 9.5e4, 3.5e-4, 60e6, 2.0e-5, 5.0e3, +1.0e-3),
    "midlat_cont_evap":  (278.0, 8.0e4, 1.0e-3, 400e6, 2.0e-4, 1.0e3, -2.0e-2),
    "polar_supercooled": (258.0, 8.5e4, 5.0e-5, 50e6, 1.0e-5, 5.0e2, +5.0e-4),
    "rain_shaft_subsat": (290.0, 9.5e4, 0.0, 0.0, 5.0e-4, 1.0e3, -1.0e-1),
    "edge_empty":        (290.0, 9.0e4, 0.0, 0.0, 0.0, 0.0, +2.0e-3),
}


def fmt(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


def run(driver, text):
    r = subprocess.run([driver], input=text, capture_output=True, text=True)
    if r.returncode != 0 or "WRF_ERROR_FATAL" in r.stdout:
        raise RuntimeError(f"oracle failed rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def floats(out):
    return np.array([float(x) for x in out.split()])


def lognormal_f(m, q_mass, n_num, sigma):
    """f [m^-3 kg^-1] of a lognormal-in-radius mode carrying q_mass [kg/m3]."""
    if q_mass <= 0.0 or n_num <= 0.0:
        return np.zeros_like(m)
    r = np.cbrt(m / (4.0 / 3.0 * math.pi * constants.rho_water))
    r_med = np.cbrt(q_mass / n_num / (4.0 / 3.0 * math.pi * constants.rho_water)) \
        * math.exp(-1.5 * math.log(sigma) ** 2)
    n_lnr = np.exp(-np.log(r / r_med) ** 2 / (2.0 * math.log(sigma) ** 2))
    f = n_lnr / (3.0 * m)
    return f * q_mass / np.sum(f * m * 3.0 * COL * m)


def exponential_f(m, q_mass, n_num):
    """f of an exponential-in-diameter mode carrying q_mass [kg/m3]."""
    if q_mass <= 0.0 or n_num <= 0.0:
        return np.zeros_like(m)
    d = 2.0 * np.cbrt(m / (4.0 / 3.0 * math.pi * constants.rho_water))
    lam = (math.pi * constants.rho_water * n_num / q_mass) ** (1.0 / 3.0)
    f = np.exp(-lam * d) * d / (3.0 * m)
    return f * q_mass / np.sum(f * m * 3.0 * COL * m)


def liquid_f(m, case):
    T, p, qc, nc, qr, nr, _ = case
    rho = p / (constants.R_d * T)
    return lognormal_f(m, qc * rho, nc, 1.4) + exponential_f(m, qr * rho, nr), rho


def q_for_oracle_s(T, p, s):
    """q such that the FAST_SBM supersaturation (l.4594-4597) equals s."""
    e = (1.0 + s) * AA1_CGS * 0.1 * math.exp(-BB1 / T)   # satcurve-ok: oracle POLYSVP
    return constants.epsilon_fast_sbm * e / (p - constants.one_minus_epsilon_fast_sbm * e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--wrf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    head = subprocess.run(["git", "-C", a.wrf, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == SHA, head
    drv = a.driver
    m = mass_doubling_grid_np()                      # kg
    xl_g = m * 1.0e3
    run_dir = f"{a.wrf}/run"
    kz = np.array(open(f"{run_dir}/kernels_z.asc").read().split(), float)
    ywll = kz[:NKR * NKR].reshape(NKR, NKR, order="F") * 1.0e-6          # m^3/s
    ck = ywll * DT * DLNR                                                 # m^3 (Kernals_KS)
    vt_tab = np.array(open(f"{run_dir}/termvels.asc").read().split(), float)[:NKR] / 100.0
    out = {"masses": m, "col": np.float64(COL), "dt": np.float64(DT), "ck": ck,
           "vt_table": vt_tab}

    # courant_bott_KS tables
    o = run(drv, "TABLES\n%d\n%s\n%s\n%s\n" % (NKR, fmt(xl_g), fmt(ck * 1e6), fmt(np.zeros(NKR))))
    rows = np.array([[float(x) for x in l.split()] for l in o.strip().splitlines()])
    out["tab_ima"] = rows[:, 2].reshape(NKR, NKR).astype(np.int32)
    out["tab_chucm"] = rows[:, 3].reshape(NKR, NKR)

    # POLYSVP (type 0)
    T_es = np.linspace(233.15, 313.15, 17)
    out["es_T"] = T_es
    out["es_cgs"] = np.array([floats(run(drv, f"POLYSVP\n{float(t)!r}\n"))[0] for t in T_es])

    # coll_xxx_lwf / coll_xyx_lwf
    names = list(CASES)
    out["case_names"] = np.array(names)
    rng = np.random.default_rng(0)
    for n in names:
        f, rho = liquid_f(m, CASES[n])
        g = 3.0 * m ** 2 * f
        coll = lambda gg: floats(run(drv, "COLL\n%d\n%s\n%s\n%s\n" % (
            NKR, fmt(xl_g), fmt(ck * 1e6), fmt(gg))))
        out[f"coll_{n}_g"] = g
        out[f"coll_{n}_out"] = coll(g)
        # The oracle's OWN round-off sensitivity: max relative change of its
        # output when every nonzero input bin moves by exactly one ulp, random
        # sign (5 draws).
        spread = 0.0
        for _ in range(5):
            gp = np.where(g != 0.0, np.nextafter(
                g, np.where(rng.random(NKR) < 0.5, -np.inf, np.inf)), g)
            o, b = coll(gp), out[f"coll_{n}_out"]
            d = np.maximum(np.abs(o), np.abs(b))
            spread = max(spread, float(np.max(np.where(d > 0, np.abs(o - b) / np.where(d > 0, d, 1.0), 0.0))))
        out[f"coll_{n}_ulp_spread"] = np.float64(spread)
    for n in ("polar_supercooled", "trop_deep_warm", "sc_stratocu"):
        f, rho = liquid_f(m, CASES[n])
        gl = 3.0 * m ** 2 * f
        gi = 3.0 * m ** 2 * lognormal_f(m, 2.0e-4 * rho, 2.0e4, 2.0)
        v = floats(run(drv, "RIME\n%d\n%s\n%s\n%s\n%s\n" % (
            NKR, fmt(xl_g), fmt(ck * 1e6), fmt(gi), fmt(gl))))
        out[f"rime_{n}_gi"], out[f"rime_{n}_gl"] = gi, gl
        out[f"rime_{n}_gi_out"], out[f"rime_{n}_gl_out"] = v[:NKR], v[NKR:]

    # ONECOND1 (one call: FAST_SBM loop with NCOND=1), still air and table velocities
    r_cm = np.cbrt(m / (4.0 / 3.0 * math.pi * constants.rho_water)) * 100.0
    for n in names:
        T, p = CASES[n][0], CASES[n][1]
        f, rho = liquid_f(m, CASES[n])
        q = q_for_oracle_s(T, p, CASES[n][6])
        out[f"cond_{n}_in"] = np.array([T, q, p, rho])
        out[f"cond_{n}_f"] = f
        for tag, v in (("v0", np.zeros(NKR)), ("vtab", vt_tab)):
            txt = "COND\n%d %d\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n" % (
                NKR, 1, fmt([DT, T, q, p * 10.0, rho * 1e-3, COL]),
                fmt([AA1_CGS, BB1, AA2_CGS, BB2]), fmt(xl_g), fmt(r_cm),
                fmt(np.full(NKR, constants.rho_water * 1e-3)), fmt(v * 100.0),
                fmt(f * F_SI2CGS))
            r = floats(run(drv, txt))
            out[f"cond_{n}_{tag}_T"] = r[0]
            out[f"cond_{n}_{tag}_q"] = r[1]
            out[f"cond_{n}_{tag}_f"] = r[3:] / F_SI2CGS

    # FREEZ
    for n in ("polar_supercooled", "trop_deep_warm"):
        for T in (268.0, 258.0, 243.0):
            p = CASES[n][1]
            rho = p / (constants.R_d * T)
            f, _ = liquid_f(m, (T,) + CASES[n][1:])
            r = floats(run(drv, "FREEZ\n%d\n%s\n%s\n%s\n" % (
                NKR, fmt([DT, T, rho * 1e-3, COL]), fmt(xl_g), fmt(f * F_SI2CGS))))
            k = f"frz_{n}_{int(T)}"
            out[k + "_in"] = np.array([T, rho])
            out[k + "_f"] = f
            out[k + "_T"] = r[0]
            out[k + "_liq"], out[k + "_cryst"], out[k + "_hail"] = (
                r[1:34] / F_SI2CGS, r[34:67] / F_SI2CGS, r[67:100] / F_SI2CGS)

    # J_W_MELT: one category populated at a time (habit1..3, snow, graupel, hail)
    cats = ["habit1", "habit2", "habit3", "snow", "graupel", "hail"]
    out["melt_cats"] = np.array(cats)
    for dt_m in (DT, 150.0):              # 150 s: mid-rate fraction 1.5 (oracle unclamped)
        for T in (274.0, 278.0):
            p = 8.5e4
            rho = p / (constants.R_d * T)
            fi = lognormal_f(m, 5.0e-4 * rho, 1.0e4, 2.5)
            for ci, c in enumerate(cats):
                spec = [np.zeros(NKR) for _ in cats]
                spec[ci] = fi
                r = floats(run(drv, "MELT\n%d\n%s\n%s\n%s\n%s\n%s\n%s\n" % (
                    NKR, fmt([dt_m, T, rho * 1e-3, COL]), fmt(xl_g),
                    fmt(np.concatenate(spec[:3]) * F_SI2CGS), fmt(spec[3] * F_SI2CGS),
                    fmt(spec[4] * F_SI2CGS), fmt(spec[5] * F_SI2CGS))))
                k = f"melt_{int(dt_m)}_{int(T)}_{c}"
                out[k + "_T"] = r[0]
                out[k + "_liq"] = r[1:34] / F_SI2CGS
                out[k + "_ice"] = r[34 + 33 * ci:67 + 33 * ci] / F_SI2CGS
            out[f"melt_{int(dt_m)}_{int(T)}_in"] = np.array([dt_m, T, rho])
            out[f"melt_{int(dt_m)}_{int(T)}_f"] = fi

    # FALFLUXHUCM_Z: 40 levels, NONUNIFORM thickness (level 0 = top in the fixture)
    nz = 40
    dz_bu = 150.0 + 10.0 * np.arange(nz)                    # bottom-up thickness [m]
    zc = np.cumsum(dz_bu) - 0.5 * dz_bu
    T = 295.0 - 6.5e-3 * zc
    p = 1.0e5 * np.exp(-zc / 8000.0)
    rho = p / (constants.R_d * T)
    qr = np.where((zc > 2000) & (zc < 6000), 8e-4 * np.sin(np.pi * (zc - 2000) / 4000) ** 2, 0.0)
    fz = np.stack([lognormal_f(m, (1e-4 if 2500 < z < 5500 else 0.0) * r, 100e6, 1.4)
                   + exponential_f(m, q * r, 2.0e3) for z, q, r in zip(zc, qr, rho)])
    vz = vt_tab[None, :] * np.sqrt(1.2 / rho)[:, None]      # m/s (nz, nkr)
    chem_si = fz / rho[:, None]
    # IFALL skip (chem >= 1e-20 CGS = 1e-14 SI at some level): one bin tiny but
    # nonzero everywhere (skipped, must not move) and one bin empty.
    chem_si[:, -1] = 1.0e-16
    chem_si[:, -2] = 0.0
    zcgs = np.zeros(nz)
    z_full = 0.0
    for k in range(nz):                                     # FAST_SBM l.4271-4275
        zcgs[k] = z_full + 0.5 * dz_bu[k] * 100.0
        z_full += dz_bu[k] * 100.0
    o = run(drv, "FALL\n%d %d\n%s\n%s\n%s\n%s\n%s\n%s\n" % (
        nz, NKR, repr(DT), fmt(rho * 1e-3), fmt(p * 10.0), fmt(zcgs),
        fmt(vz * 100.0), fmt(chem_si * 1e-6)))
    toks = o.split()
    out["fall_nsub"] = np.array([int(x) for x in toks[:NKR]], np.int32)
    chem_o = np.array([float(x) for x in toks[NKR:]]).reshape(nz, NKR) * 1e6
    out["fall_chem_in"] = chem_si[::-1].copy()
    out["fall_v"] = vz[::-1].copy()
    out["fall_rho"] = rho[::-1].copy()
    out["fall_dz"] = dz_bu[::-1].copy()
    out["fall_chem_out"] = chem_o[::-1].copy()

    out["meta"] = np.array(json.dumps({
        "repo": "wrf-model/WRF", "file": "phys/module_mp_fast_sbm.F", "sha": SHA,
        "precision": "fp64 (R4SIZE->8, -fdefault-real-8)", "dt": DT,
        "generator": "scripts/validate/oracle/fast_sbm/make_fixture.py"}))
    np.savez_compressed(a.out, **out)
    print("FIXTURE_OK", a.out, len(out))


if __name__ == "__main__":
    main()
