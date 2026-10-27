"""Generate tests/atmosphere/microphysics/oracle_fixtures/p3.npz by RUNNING WRF P3.

Oracle: WRF v4.8.0 (https://github.com/wrf-model/WRF, tag v4.8.0, SHA
06d4240ae989cc3e50af412bb472df3d9048783c) phys/module_mp_p3.F (P3 v4.5.2) with
run/p3_lookupTable_1.dat-v5.4_2momI, built by build.sh (fp64, legoESM
saturation + constants, process-rate dumps).  Configuration: WRF
mp_physics = 50 (P3_1CATEGORY) as mp_p3_wrapper_wrf runs it -- one 2-moment ice
category, specified droplet number (nccnst = 200 /cm^3, p3_init), KK2000 warm
rain (iparam = 3), Cooper (1986) nucleation, max_total_Ni = 2000 /L, no
sub-grid cloud fraction, clbfact_dep = clbfact_sub = 1.

Usage (inside a SLURM job, after ``bash build.sh <WRF> <build>``)::

    PYTHONPATH=<legoESM packages> python make_fixture.py --build <build> \
        --cases <dir holding RCE/snd_300K, DYCOMS_RF01/snd, MC3E/snd> --out p3.npz

Harness glue (the tier-2 p3_microphysics audit's): level order is top-down
here and bottom-up in the driver (model='WRF'); N_c, N_r are per m^3 here and
per kg in the oracle (/ rho, the input air density); th = T*(1e5/p)**(R_d/c_pd)
with the patched module's rd/cp, th_old = th, qv_old = qv (no dynamical
forcing), uzpl = 0, it = 2.

Input columns: the tier-2 audit's five (trop_deep, sc_dycoms, midlat_strat from
the gSAM 1.8.8 case soundings that audit used -- sounding files only, read as
data, sha256 recorded in the fixture; arctic_mixed and edge analytic), the
``coverage`` column (one process family per level), and 226 single-level
nucleation points (one column each).  Every case runs at dt = 20 s
(run_scm_rce_campaign: 600 s / 30 substeps); the columns also at dt = 300 s
(sedimentation sub-steps and the conservation limiters bind).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.thermo import saturation_specific_humidity as _qsl  # noqa: E402
from legoesm.thermo import saturation_specific_humidity_ice as _qsi  # noqa: E402

KAP = C.R_d / C.c_pd
NC_SPEC = 2.0e8          # WRF P3 specified nccnst = 200 /cm^3 (p3_init); input N_c is not read
DTS = (20.0, 300.0)
IN_FIELDS = ("p", "rho", "dz", "T", "q_v", "q_c", "N_c", "q_r", "N_r", "q_i",
             "q_rim", "N_i", "B_rim")
OVARS = ["T", "qv", "qc", "nc", "qr", "nr", "qi", "qirim", "ni", "birim"]
DIAG = ["qccon", "qcevp", "qrcon", "qrevp", "qcaut", "qcacc", "ncautr", "nrslf", "qccol",
        "qrcol", "nrcol", "qidep", "qisub", "qinuc", "ninuc", "nislf", "qimlt", "nimlt",
        "qcheti", "qrheti", "qcshd", "qwgrth", "rhorime_c", "nccol", "nrevp", "qcnuc",
        "ncheti", "qchomi", "qrhomi", "nrheti", "ncautc", "ncacc", "nisub", "epsi_tot",
        "nrshdr", "ncshdc"]
OUT_FIELDS = OVARS + DIAG
WRF_TAG, WRF_SHA = "v4.8.0", "06d4240ae989cc3e50af412bb472df3d9048783c"
TABLE = "p3_lookupTable_1.dat-v5.4_2momI"
CASE_FILES = ("RCE/snd_300K", "DYCOMS_RF01/snd", "MC3E/snd")


def qsl(T, p):
    return np.asarray(_qsl(np.asarray(T, float), np.asarray(p, float)))


def qsi(T, p):
    return np.asarray(_qsi(np.asarray(T, float), np.asarray(p, float)))


def _rho(p, T, qv):
    return p / (C.R_d * T * (1.0 + (C.R_v / C.R_d - 1.0) * qv))


def _nr_for_D(q, rho, D):
    """rain number per m^3 for mass-mean diameter D [m]."""
    return np.where(q > 0, q * rho / (np.pi / 6 * C.rho_water * D ** 3), 0.0)


def _blank(n):
    z = np.zeros(n)
    return dict(q_c=z.copy(), q_r=z.copy(), q_i=z.copy(), q_rim=z.copy(), B_rim=z.copy(),
                N_c=np.full(n, NC_SPEC), N_r=z.copy(), N_i=z.copy())


def _finish(name, z, p, T, qv, h):
    """bottom-up inputs -> top-down dict; dz from interface midpoints."""
    zi = np.concatenate([[max(0.0, z[0] - (z[1] - z[0]) / 2)], 0.5 * (z[1:] + z[:-1]),
                         [z[-1] + (z[-1] - z[-2]) / 2]])
    dz = np.diff(zi)
    rho = _rho(p, T, qv)
    d = dict(name=name, z=z, p=p, T=T, q_v=qv, rho=rho, dz=dz, **h)
    return {k: (v[::-1].copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ---------------------------------------------------------------- audit columns
def trop_deep(orc):
    """RCE/snd_300K: deep-convective hydrometeor profile."""
    nl = int(open(os.path.join(orc, "RCE", "snd_300K")).read().split("\n")[1].split()[1])
    a = np.loadtxt(os.path.join(orc, "RCE", "snd_300K"), skiprows=2, max_rows=nl)
    z, pmb, th, q = a[:, 0], a[:, 1], a[:, 2], a[:, 3] * 1e-3
    sel = (pmb > 100.0)
    z, p, th, q = z[sel], pmb[sel] * 100.0, th[sel], q[sel]
    idx = np.unique(np.linspace(0, len(z) - 1, 40).round().astype(int))
    z, p, th, q = z[idx], p[idx], th[idx], q[idx]
    T = th * (p / 1e5) ** KAP
    n = len(z); h = _blank(n); qv = q.copy()
    for k in range(n):
        Tk, pk, zk = T[k], p[k], z[k]
        if zk < 1000.0:
            h["q_r"][k] = 4e-4
        elif zk < 5000.0:
            qv[k] = qsl(Tk, pk); h["q_c"][k] = 8e-4; h["q_r"][k] = 6e-4
        elif Tk > 250.0:
            qv[k] = qsl(Tk, pk); h["q_c"][k] = 3e-4
            h["q_r"][k] = 2e-4 if Tk > 263.0 else 0.0
            h["q_i"][k] = 1e-3; h["q_rim"][k] = 6e-4; h["B_rim"][k] = 6e-4 / 600.0
            h["N_i"][k] = 5e3
        elif Tk > 225.0:
            # 1.06, not 1.05: off the 0.05 nucleation gate (an input ON a hard
            # threshold measures saturation rounding, not P3)
            qv[k] = 1.06 * qsi(Tk, pk); h["q_i"][k] = 3e-4; h["N_i"][k] = 1e5
            h["q_rim"][k] = 3e-5; h["B_rim"][k] = 3e-5 / 300.0
            if 233.0 < Tk < 236.0:
                h["q_c"][k] = 2e-5
        elif Tk > 200.0:
            if k % 2 == 0:
                qv[k] = 1.2 * qsi(Tk, pk); h["q_i"][k] = 2e-5; h["N_i"][k] = 2e5
            else:
                qv[k] = 1.25 * qsi(Tk, pk)
    rho = _rho(p, T, qv)
    h["N_r"] = _nr_for_D(h["q_r"], rho, 1.0e-3)
    return _finish("trop_deep", z, p, T, qv, h)


def sc_dycoms(orc):
    """DYCOMS_RF01/snd: saturated stratocumulus 600-840 m with drizzle."""
    nl = int(open(os.path.join(orc, "DYCOMS_RF01", "snd")).read().split("\n")[1].split(",")[1])
    a = np.loadtxt(os.path.join(orc, "DYCOMS_RF01", "snd"), skiprows=2, max_rows=nl)
    z, thl, qt = a[:, 0], a[:, 2], a[:, 3] * 1e-3
    sel = z <= 1500.0
    z, thl, qt = z[sel], thl[sel], qt[sel]
    idx = np.unique(np.linspace(0, len(z) - 1, 30).round().astype(int))
    z, thl, qt = z[idx], thl[idx], qt[idx]
    z = np.maximum(z, 2.5)
    n = len(z); p = np.empty(n); T = np.empty(n); qv = np.empty(n); qc = np.zeros(n)
    pk, zprev = 101780.0, 0.0
    for k in range(n):
        Tg = thl[k] * (pk / 1e5) ** KAP
        pk = pk * np.exp(-C.g * (z[k] - zprev) / (C.R_d * Tg)); zprev = z[k]
        T0 = thl[k] * (pk / 1e5) ** KAP
        lo, hi = T0, T0 + C.L_v / C.c_pd * qt[k]  # latent-ok: gSAM DYCOMS sounding thl (constant fac_cond)
        for _ in range(200):
            Tk = 0.5 * (lo + hi)
            f = Tk - T0 - C.L_v / C.c_pd * max(qt[k] - float(qsl(Tk, pk)), 0.0)  # latent-ok: gSAM DYCOMS sounding thl
            lo, hi = (Tk, hi) if f < 0 else (lo, Tk)
        qck = max(qt[k] - float(qsl(Tk, pk)), 0.0)
        p[k], T[k], qv[k], qc[k] = pk, Tk, qt[k] - qck, qck
    h = _blank(n); h["q_c"] = qc
    h["q_r"] = np.where((z < 840.0), 2e-5, 0.0)
    rho = _rho(p, T, qv)
    h["N_r"] = _nr_for_D(h["q_r"], rho, 1.0e-4)
    return _finish("sc_dycoms", z, p, T, qv, h)


def midlat_strat(orc):
    """MC3E/snd first block: stratiform snow, melting layer, evaporating rain."""
    lines = open(os.path.join(orc, "MC3E", "snd")).read().split("\n")
    hdr = lines[1].split(); nl = int(hdr[1]); psfc = float(hdr[2]) * 100.0
    a = np.array([[float(x) for x in l.split()] for l in lines[2:2 + nl]])
    pmb, th, q = a[:, 1], a[:, 2], a[:, 3] * 1e-3
    sel = (pmb * 100.0 <= psfc) & (pmb >= 150.0)
    p, th, q = pmb[sel] * 100.0, th[sel], q[sel]
    T = th * (p / 1e5) ** KAP
    rho = _rho(p, T, q)
    z = np.concatenate([[0.0], np.cumsum(-np.diff(p) / (rho[:-1] * C.g))]) + 112.0
    n = len(z); h = _blank(n); qv = q.copy()
    T0 = C.T_freeze
    for k in range(n):
        Tk, pk = T[k], p[k]
        if Tk < T0 - 25.0:
            continue
        if T0 - 25.0 <= Tk < T0 - 12.0:
            qv[k] = 1.02 * qsi(Tk, pk); h["q_i"][k] = 4e-4; h["N_i"][k] = 2e4
        elif T0 - 12.0 <= Tk < T0:
            qv[k] = qsl(Tk, pk); h["q_c"][k] = 1e-4
            h["q_i"][k] = 5e-4; h["N_i"][k] = 1e4; h["q_rim"][k] = 1e-4
            h["B_rim"][k] = 1e-4 / 300.0
        elif T0 <= Tk < T0 + 4.0:
            qv[k] = 0.98 * qsl(Tk, pk); h["q_i"][k] = 2e-4; h["N_i"][k] = 5e3
            h["q_rim"][k] = 4e-5; h["B_rim"][k] = 4e-5 / 300.0; h["q_r"][k] = 3e-4
        else:
            qv[k] = min(qv[k], 0.8 * float(qsl(Tk, pk))); h["q_r"][k] = 3e-4
    rho = _rho(p, T, qv)
    h["N_r"] = _nr_for_D(h["q_r"], rho, 1.2e-3)
    return _finish("midlat_strat", z, p, T, qv, h)


def arctic_mixed(orc):
    """ANALYTIC Arctic mixed-phase stratus (M-PACE-like)."""
    z = np.concatenate([np.linspace(10.0, 1500.0, 16), np.linspace(1800.0, 9000.0, 14)])
    T = np.where(z <= 1200.0, 258.0 - 6.5e-3 * z,
                 258.0 - 6.5e-3 * 1200.0 + 4.0 - 6.5e-3 * (z - 1200.0))
    T = np.maximum(T, 210.0)
    p = np.empty_like(z); pk, zprev = 101000.0, 0.0
    for k in range(len(z)):
        pk = pk * np.exp(-C.g * (z[k] - zprev) / (C.R_d * T[k])); zprev = z[k]; p[k] = pk
    n = len(z); h = _blank(n); qv = np.empty(n)
    for k in range(n):
        Tk, pk = T[k], p[k]
        if 600.0 <= z[k] <= 1200.0:
            qv[k] = qsl(Tk, pk); h["q_c"][k] = 2e-4
            h["q_i"][k] = 1e-5; h["N_i"][k] = 1e3; h["q_rim"][k] = 2e-6
            h["B_rim"][k] = 2e-6 / 400.0
        elif z[k] < 600.0:
            qv[k] = qsi(Tk, pk); h["q_i"][k] = 2e-5; h["N_i"][k] = 1e3
        elif z[k] > 4000.0 and k % 3 == 0:
            qv[k] = 1.15 * qsi(Tk, pk)
        else:
            qv[k] = 0.6 * qsi(Tk, pk)
    return _finish("arctic_mixed", z, p, T, qv, h)


def edge(orc):
    """Dry column, sub-qsmall cloud/ice, warm residual ice, stratospheric ice-
    supersaturated levels with rho < 0.1."""
    z = np.linspace(50.0, 21000.0, 14)
    T = np.maximum(295.0 - 6.5e-3 * z, 205.0)
    p = np.empty_like(z); pk, zprev = 101300.0, 0.0
    for k in range(len(z)):
        pk = pk * np.exp(-C.g * (z[k] - zprev) / (C.R_d * T[k])); zprev = z[k]; p[k] = pk
    qv = 0.4 * qsl(T, p)
    h = _blank(len(z))
    h["q_c"][1] = 1e-15; h["q_i"][2] = 1e-15; h["N_i"][2] = 1.0
    h["q_i"][0] = 5e-9; h["N_i"][0] = 10.0
    for k in (12, 13):
        qv[k] = 1.3 * float(qsi(T[k], p[k]))
    return _finish("edge", z, p, T, qv, h)


def coverage(orc):
    """Built here (bottom-up, one process family per level) for the processes
    the audit columns leave at zero; the top two levels put warm rain above a
    level colder than -40 C so sedimented rain freezes homogeneously."""
    lv = [  # (p, T, qv factor, qv curve, q_c, q_r, D_r, q_i, N_i, q_rim, rho_rim)
        (95000., 290.0, 0.90, "l", 0.0, 5e-4, 1.0e-3, 0.0, 0.0, 0.0, 0.0),     # evaporating rain
        (92500., 288.0, 1.005, "l", 5e-4, 3e-4, 1.0e-3, 0.0, 0.0, 0.0, 0.0),  # condensation on cloud + rain
        (90000., 286.0, 1.01, "l", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),        # activation (no condensate)
        (87500., 283.0, 1.05, "l", 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),       # saturation-adjustment limit
        (85000., 275.0, 1.00, "l", 3e-4, 2e-4, 1.0e-3, 3e-4, 2e3, 2e-4, 500.),  # melting + shedding
        (82500., 274.0, 0.85, "l", 0.0, 0.0, 0.0, 5e-4, 3e3, 1e-4, 300.),     # melting, subsaturated
        (80000., 271.5, 1.00, "l", 1.5e-3, 2e-3, 1.5e-3, 2e-3, 5e2, 1.8e-3, 850.),  # wet growth
        (75000., 267.0, 1.00, "l", 2e-4, 1e-4, 1.0e-3, 0.0, 0.0, 0.0, 0.0),   # immersion freezing
        (70000., 262.0, 0.75, "i", 0.0, 0.0, 0.0, 5e-5, 1e4, 0.0, 0.0),       # sublimation
        (65000., 258.5, 1.003, "l", 2e-4, 0.0, 0.0, 1e-4, 1e4, 0.0, 0.0),     # mixed phase, supersaturated
        (60000., 255.0, 0.95, "i", 0.0, 0.0, 0.0, 1e-13, 10.0, 0.0, 0.0),     # instant sublimation
        (50000., 245.0, 0.95, "l", 5e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),     # instant cloud evap
        (45000., 233.0, 1.00, "l", 5e-5, 3e-5, 5.0e-4, 0.0, 0.0, 0.0, 0.0),  # homogeneous freezing
        (40000., 234.0, 1.00, "i", 2e-5, 1e-5, 5.0e-4, 1e-5, 1e4, 0.0, 0.0),  # supercooled, ice-saturated
        (30000., 225.0, 1.30, "i", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),       # nucleation at the cap
        (20000., 215.0, 1.10, "i", 0.0, 0.0, 0.0, 1e-6, 5e4, 0.0, 0.0),       # deposition
        (15000., 210.0, 1.00, "i", 0.0, 0.0, 0.0, 2e-6, 3e7, 0.0, 0.0),       # total-Ni cap
        (14000., 232.0, 0.90, "i", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),       # receives rain -> homogeneous freezing
        (13000., 272.0, 0.95, "l", 0.0, 5e-4, 1.0e-3, 0.0, 0.0, 0.0, 0.0),   # warm rain above (no immersion freezing)
    ]
    a = np.array([[r[0], r[1], r[2], 0.0 if r[3] == "l" else 1.0] + list(r[4:]) for r in lv])
    p, T, fq, ice = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    qv = fq * np.where(ice > 0, qsi(T, p), qsl(T, p))
    rho = _rho(p, T, qv)
    z = np.concatenate([[100.0], 100.0 + np.cumsum(-np.diff(p) / (0.5 * (rho[1:] + rho[:-1]) * C.g))])
    h = _blank(len(p))
    h["q_c"], h["q_r"], h["q_i"], h["N_i"], h["q_rim"] = a[:, 4], a[:, 5], a[:, 7], a[:, 8], a[:, 9]
    h["B_rim"] = np.where(a[:, 10] > 0, a[:, 9] / np.where(a[:, 10] > 0, a[:, 10], 1.0), 0.0)
    h["N_r"] = np.where(a[:, 5] > 0, _nr_for_D(a[:, 5], rho, np.where(a[:, 6] > 0, a[:, 6], 1.0)), 0.0)
    return _finish("coverage", z, p, T, qv, h)


def nucleation_points(orc):
    """226 single-level points: no condensate, liquid-subsaturated, T 200-262 K,
    ice supersaturation 0.02-0.45 (both sides of the 0.05 gate)."""
    pts = []
    for p in (30000.0, 60000.0):
        for T in np.arange(200.0, 263.0, 3.5):
            for s in (0.02, 0.049, 0.051, 0.08, 0.15, 0.3, 0.45):
                qv = (1.0 + s) * float(qsi(T, p))
                if qv / float(qsl(T, p)) >= 0.999:
                    continue
                pts.append((p, T, qv))
    arr = np.array(pts)
    n = len(arr)
    h = _blank(n)
    return dict(name="nuc", p=arr[:, 0], T=arr[:, 1], q_v=arr[:, 2],
                rho=_rho(arr[:, 0], arr[:, 1], arr[:, 2]), dz=np.full(n, 200.0), **h)


COLUMNS = (trop_deep, sc_dycoms, midlat_strat, arctic_mixed, edge, coverage)


# ---------------------------------------------------------------- oracle driver
def run_oracle(build, inp, dt, tag):
    """inp: dict of (ncol, nlev) top-down arrays.  Returns dict of (ncol, nlev)
    top-down outputs + prt_liq/prt_sol (ncol,)."""
    ncol, nlev = inp["T"].shape
    fin = os.path.join(build, f"in_{tag}.txt")
    fout = os.path.join(build, f"out_{tag}.txt")
    look = os.path.join(build, "lookup")
    with open(fin, "w") as f:
        # ncol nlev dt it kappa
        f.write(f"{ncol} {nlev} {dt!r} 2 {KAP!r}\n{look}\n")
        for c in range(ncol):
            for k in range(nlev - 1, -1, -1):
                rho = inp["rho"][c, k]
                vals = [inp["p"][c, k], inp["dz"][c, k], inp["T"][c, k], inp["q_v"][c, k],
                        inp["q_c"][c, k], inp["q_r"][c, k], inp["N_r"][c, k] / rho,
                        inp["q_i"][c, k], inp["q_rim"][c, k], inp["N_i"][c, k],
                        inp["B_rim"][c, k]]
                f.write(" ".join(f"{v:.17e}" for v in vals) + "\n")
    r = subprocess.run([os.path.join(build, "p3_driver"), fin, fout],
                       capture_output=True, text=True, cwd=build)
    assert "P3_DRIVER_DONE" in r.stdout, r.stdout + r.stderr
    cols, cur = [], None
    for line in open(fout):
        if line.startswith("COL"):
            parts = line.split()
            cur = dict(prt_liq=float(parts[2]), prt_sol=float(parts[3]), rows=[])
            cols.append(cur)
        else:
            cur["rows"].append([float(x) for x in line.split()])
    out = {n: np.stack([np.array(c["rows"])[::-1, i] for c in cols]) for i, n in enumerate(OUT_FIELDS)}
    out["prt_liq"] = np.array([c["prt_liq"] for c in cols])
    out["prt_sol"] = np.array([c["prt_sol"] for c in cols])
    return out


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_cases(orc):
    cases = {}
    for f in COLUMNS:
        c = f(orc)
        cases[c["name"]] = {k: np.asarray(c[k], float)[None, :] for k in IN_FIELDS}
    nuc = nucleation_points(orc)
    cases["nuc"] = {k: np.asarray(nuc[k], float)[:, None] for k in IN_FIELDS}
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--cases", required=True,
                    help="directory holding RCE/snd_300K, DYCOMS_RF01/snd, MC3E/snd")
    ap.add_argument("--wrf", required=True, help="WRF v4.8.0 checkout (provenance only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build = os.path.abspath(a.build)
    orc = os.path.abspath(a.cases)
    head = subprocess.run(["git", "-C", a.wrf, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == WRF_SHA, head
    # instrument check: the lego_sat stub vs legoesm.thermo
    TP = np.array([(T, p) for T in np.arange(185.0, 315.0, 2.5) for p in (5e3, 2e4, 5e4, 8e4, 1.02e5)])
    np.savetxt(os.path.join(build, "qsat_pts.txt"), TP, fmt="%.17e")
    r = subprocess.run([os.path.join(build, "qsat_check"), os.path.join(build, "qsat_pts.txt")],
                       capture_output=True, text=True, check=True)
    qs = np.array([[float(x) for x in l.split()] for l in r.stdout.strip().split("\n")])
    rel_l = float(np.max(np.abs(qs[:, 2] - qsl(TP[:, 0], TP[:, 1])) / qsl(TP[:, 0], TP[:, 1])))
    rel_i = float(np.max(np.abs(qs[:, 3] - qsi(TP[:, 0], TP[:, 1])) / qsi(TP[:, 0], TP[:, 1])))
    assert rel_l < 1e-14 and rel_i < 1e-14, (rel_l, rel_i)

    cases = build_cases(orc)
    fx = {}
    for name, inp in cases.items():
        for k, v in inp.items():
            fx[f"in_{name}_{k}"] = v
        for dt in (DTS if name != "nuc" else DTS[:1]):
            o = run_oracle(build, inp, dt, f"{name}_dt{dt:g}")
            for k, v in o.items():
                fx[f"out_{name}_dt{dt:g}_{k}"] = v
    meta = dict(
        oracle="WRF v4.8.0 phys/module_mp_p3.F (P3 v4.5.2), mp_physics=50 (P3_1CATEGORY)",
        repo="https://github.com/wrf-model/WRF", tag=WRF_TAG, sha=WRF_SHA,
        module_mp_p3_sha256=sha256(os.path.join(a.wrf, "phys", "module_mp_p3.F")),
        lookup_table_1_sha256=sha256(os.path.join(build, "lookup", TABLE)),
        patched_module_sha256=sha256(os.path.join(build, "module_mp_p3_lego.F")),
        case_soundings_sha256={c: sha256(os.path.join(orc, c)) for c in CASE_FILES},
        case_soundings_source="gSAM 1.8.8 CASES (tier-2 p3_microphysics audit inputs)",
        config=dict(nCat=1, trplMomI=False, log_predictNc=False, nccnst_per_m3=2.0e8,
                    iparam=3, max_total_Ni_per_m3=2.0e6, scpf_on=False, clbfact_dep=1.0,
                    clbfact_sub=1.0, it=2, model="WRF", typeDiags_ON=False),
        glue="legoESM constants (cp, g, rd, xxlv) and saturation (qv_sat) patched into the "
             "oracle; th=T*(1e5/p)**(R_d/c_pd); th_old=th; qv_old=qv; uzpl=0; ssat=0; "
             "N_r per kg = per m3 / rho_in",
        compiler=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        flags="-O0 -fdefault-real-8 -fdefault-double-8 -ffree-form -cpp",
        qsat_stub_vs_thermo_max_rel=dict(liquid=rel_l, ice=rel_i),
        cases=sorted(cases), dts=list(DTS), nuc_dts=[DTS[0]],
        out_fields=OUT_FIELDS,
    )
    fx["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
