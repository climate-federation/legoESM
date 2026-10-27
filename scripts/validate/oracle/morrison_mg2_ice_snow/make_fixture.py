"""Generate the CAM6 MG2 two-moment snow oracle fixture by RUNNING the oracle.

Oracle: ESCOMP/CAM ``cam_cesm2_1_rel_60`` (commit
a03b84b7c4e34f965b115686f22a043b85739e56), ``src/physics/cam/micro_mg2_0.F90``
(full ``micro_mg_tend``), ``micro_mg_utils.F90`` and ``wv_sat_methods.F90``
(CAM's Goff-Gratch saturation) compiled UNMODIFIED.  Settings
are CAM6 + CLUBB (``mg2_snow_driver.F90``), micro_mg_dcs = 500 um.

Two parts, written to one ``.npz``:
  * ``col_*``: five ice-free columns carrying snow + snow number (CAM L32,
    sounding shapes) run through the whole ``micro_mg_tend`` at relvar 2,
    dt = 600 s.  Built so that only processes this port carries are active:
    no cloud ice (MG2 naai = 0, legoESM's nucleation gates closed: RH_ice <
    1.08 everywhere, RH_liq < 0.999 below 265.15 K), no rain below the melting
    point (no rain-snow collection / Bigg), no riming inside the 265-270 K
    Hallett-Mossop window, no cloud water below 233 K;
  * ``pt_*``: a point grid through the snow kernels, Bigg freezing and the
    ice->snow autoconversion (``mg2_snow_procs.F90``).

Usage (CPU node; needs gfortran and the CAM checkout)::

    python scripts/validate/oracle/morrison_mg2_ice_snow/make_fixture.py \\
        --cam /burg/glab/users/pg2328/oracles/CAM --build /tmp/mg2s_bld \\
        --out tests/atmosphere/microphysics/oracle_fixtures/mg2_ice_snow.npz
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT = 600.0          # CAM6 macmic sub-step: 1800 s / cld_macmic_num_steps = 3
RELVAR = 2.0
DCS = 500.0e-6      # CAM6 MG2 micro_mg_dcs (namelist_defaults_cam.xml:1244)
RHOSN = 250.0       # MG2 rhosn [kg/m^3]
COL_FIELDS = ("T", "q_v", "q_c", "cf", "q_r", "N_r", "q_s", "N_s", "N_c",
              "p_full", "p_half", "rho")
ORACLE_COLS = ("tlat", "qvlat", "qctend", "qrtend", "nrtend", "qstend",
               "nstend", "qssedten", "psacwstot", "bergstot", "evapsnow",
               "meltsdttot", "ums", "qrsedten", "sflx", "rflx", "msacwitot",
               "pracstot", "mnuccrtot")
PT_IN = ("T", "p", "rho", "qfrac", "qcic", "ncic", "qs", "ns", "lcldm",
         "precip_frac", "qiic", "qr", "nr", "qi", "ni")
PT_OUT = ("qsic", "nsic", "lams", "n0s", "ums", "uns", "nsagg", "pgam", "lamc",
          "ncic_adj", "psacws", "npsacws", "bergs", "prds", "qvl", "qvi",
          "qric", "nric", "lamr", "mnuccr", "nnuccr", "lami", "n0i", "niic",
          "prci", "nprci", "pre", "am")


def _qsat(T, p):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.thermo import (saturation_specific_humidity,
                                saturation_specific_humidity_ice)
    return (np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p))),
            np.asarray(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(p))))


def _ns(lam, q):
    """Per-mass snow number for slope ``lam`` [1/m] at mass ``q`` [kg/kg]."""
    return lam ** 3 * q / (math.pi * RHOSN)


def build_columns():
    """Five snow columns (level 0 = model top)."""
    from legoesm import constants as C
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0
    hyai = np.array(CAM6_L32_HYAI)
    hybi = np.array(CAM6_L32_HYBI)
    nlev = len(hyai) - 1
    T0 = C.T_freeze
    cols = []

    def base(ps, Ts, gam, Ttrop):
        ph = hyai * CAM6_L32_P0 + hybi * ps
        pf = 0.5 * (ph[1:] + ph[:-1])
        T = np.maximum(Ts * (pf / ps) ** (C.R_d * gam / C.g), Ttrop)
        return ph, pf, T

    def finish(name, ph, pf, T, rhl, qc, cf, qr, lamr, qs, lams, Nc):
        rho = pf / (C.R_d * T)
        ql, qi = _qsat(T, pf)
        # every legoESM nucleation gate stays closed (MG2 runs naai = 0)
        rhl = np.where(T <= 265.15, np.minimum(rhl, 0.99), rhl)
        qv = np.minimum(rhl * ql, 1.05 * qi)
        Nr = np.where(qr > 0, lamr ** 3 * rho * qr / (math.pi * C.rho_water), 0.0)
        cols.append(dict(name=name, p_half=ph, p_full=pf, T=T, q_v=qv, q_c=qc,
                         cf=cf, q_r=qr, N_r=Nr, q_s=qs,
                         N_s=np.where(qs > 0, _ns(lams, qs), 0.0),
                         N_c=np.full(nlev, Nc), rho=rho))

    # 1. midlatitude stratiform snow melting into rain; supercooled liquid
    #    layer at 270.5-273 K (riming + Bergeron), cf = 1 at the top so the
    #    clear snow layers above the cloud carry precip_frac = 1.
    ph, pf, T = base(100000.0, 283.0, 6.5e-3, 215.0)
    P = pf / 100.0
    cl = (T > 270.5) & (T < 273.0)
    cf = np.where(cl, 0.6, 0.0)
    cf[0] = 1.0
    qc = np.where(cl, 0.6 * 3.0e-4, 0.0)
    rhl = np.where(cl, 1.0, np.where(T > T0, 0.8, 0.85))
    qs = np.where((P >= 400) & (P <= 950),
                  np.interp(P, [400, 700, 950], [3e-5, 4e-4, 1e-4]), 0.0)
    qr = np.where(T > T0, 1.5e-4, 0.0)
    finish("strat_melt", ph, pf, T, rhl, qc, cf, qr, np.full(nlev, 3000.0), qs,
           np.full(nlev, 2500.0), 100.0e6)

    # 2. Arctic mixed-phase: liquid below 265 K at RH_liq 0.95, snow to the
    #    surface (no melt), surface snow flux.
    ph, pf, T = base(101000.0, 262.0, 5.0e-3, 215.0)
    P = pf / 100.0
    cl = (P >= 850) & (P <= 930)
    cf = np.where(cl, 0.3, 0.0)
    cf[0] = 1.0
    qc = np.where(cl, 0.3 * 2.0e-4, 0.0)
    rhl = np.where(cl, 0.95, 0.85)
    qs = np.where(P >= 600, np.interp(P, [600, 900, 1100], [2e-5, 8e-5, 6e-5]), 0.0)
    finish("arctic_mixed", ph, pf, T, rhl, qc, cf, np.zeros(nlev),
           np.full(nlev, 3000.0), qs, np.full(nlev, 5000.0), 50.0e6)

    # 3. cold snow aloft sublimating into dry air below (no liquid anywhere).
    ph, pf, T = base(100000.0, 250.0, 6.0e-3, 205.0)
    P = pf / 100.0
    cf = np.zeros(nlev)
    cf[0] = 1.0
    rhl = np.where(P < 650, 0.9, 0.5)
    qs = np.where((P >= 300) & (P <= 800),
                  np.interp(P, [300, 550, 800], [1e-5, 1.5e-4, 2e-5]), 0.0)
    finish("cold_snow", ph, pf, T, rhl, np.zeros(nlev), cf, np.zeros(nlev),
           np.full(nlev, 3000.0), qs, np.full(nlev, 1500.0), 50.0e6)

    # 4. sub-grid cloud aloft (cf 0.3, liquid at 271-272.5 K); the clear
    #    layers below inherit precip_frac 0.3 and the snow sublimates only in
    #    their precipitating clear air.
    ph, pf, T = base(100000.0, 284.0, 6.5e-3, 215.0)
    cl = (T > 271.0) & (T < 272.5)
    cf = np.where(cl, 0.3, 0.0)
    cf[0] = 1.0
    qc = np.where(cl, 0.3 * 5.0e-4, 0.0)
    rhl = np.where(cl, 1.0, np.where(T > T0, 0.6, 0.7))
    qs = np.where(T < 273.0, 0.0, 1e-4)
    qs = np.where(cl, 3e-4, qs)
    qr = np.where(T > T0, 5e-5, 0.0)
    finish("pf_inherit", ph, pf, T, rhl, qc, cf, qr, np.full(nlev, 4000.0), qs,
           np.full(nlev, 3500.0), 80.0e6)

    # 5. edge column: in-precip cap, slope bounds both sides, orphan number,
    #    partial (heat-limited) melt, q_c and sublimation and aggregation
    #    limiters binding.  Cold base so the snow-only levels stay frozen.
    ph, pf, T = base(100000.0, 268.0, 6.5e-3, 215.0)
    T = T.copy()
    nl = nlev
    cf = np.zeros(nl)
    cf[0] = 1.0
    qc = np.zeros(nl)
    qs = np.zeros(nl)
    lams = np.full(nl, 2000.0)
    qr = np.zeros(nl)
    rhl = np.full(nl, 0.85)
    k = nl - 1
    T[k] = 276.0                       # partial pre-melt: L_f/c_p q_s > T - 275.15
    qs[k] = 4.0e-3
    qr[k] = 2e-4
    T[k - 2] = 272.0                   # q_c limiter: tiny liquid, big snow
    qc[k - 2] = 1.0e-9
    cf[k - 2] = 0.3
    qs[k - 2] = 1.0e-3
    rhl[k - 2] = 1.0
    T[k - 4] = 271.0                   # in-precip cap: qs/precip_frac > 0.01
    qc[k - 4] = 1.0e-5
    cf[k - 4] = 0.1
    qs[k - 4] = 2.0e-3
    rhl[k - 4] = 1.0
    qs[k - 6] = 1.0e-5                 # slope above lams_max: number reset
    lams[k - 6] = 2.0e5
    qs[k - 8] = 2.0e-4                 # slope below lams_min
    lams[k - 8] = 200.0
    qs[k - 10] = 1.0e-9                # sublimation limiter (very dry)
    rhl[k - 10] = 0.2
    qs[k - 12] = 5.0e-3                # aggregation limiter (big, few flakes)
    lams[k - 12] = 520.0
    rhl[k - 12] = 0.85
    finish("edge", ph, pf, T, rhl, qc, cf, qr, np.full(nl, 3000.0), qs, lams,
           100.0e6)
    c = cols[-1]
    c["q_s"][k - 14] = 1.0e-19         # below qsmall with an orphan number
    c["N_s"][k - 14] = 1.0e3
    return cols


def build_points():
    """Point grid through the snow kernels, Bigg freezing and ice autoconv."""
    from legoesm import constants as C
    pts = []
    for T in (240.0, 262.0, 268.0, 269.0, 271.0, 273.0, 274.0):
        p = 70000.0
        rho = p / (C.R_d * T)
        for qfrac in (0.6, 0.97):
            for qcic, ncic in ((0.0, 0.0), (2e-4, 50e6 / rho), (1.5e-3, 300e6 / rho)):
                for qs_ip, lam_s in ((0.0, 2000.0), (1e-6, 8000.0), (3e-4, 2000.0),
                                     (5e-3, 600.0), (1e-4, 3.0e5), (1e-4, 300.0)):
                    for lcldm, pf in ((1.0, 1.0), (0.3, 0.3), (0.3, 0.8),
                                      (1.0e-4, 0.5)):
                        qr_ip = 2e-4 if T < C.T_freeze else 0.0
                        lam_r = 3000.0
                        qi, ni = ((5e-6, (1.0 / 60e-6) ** 3 * 5e-6 / (math.pi * 500.0))
                                  if T < 270.0 else (0.0, 0.0))
                        ns_ip = _ns(lam_s, qs_ip)
                        nr_ip = lam_r ** 3 * qr_ip / (math.pi * C.rho_water)
                        pts.append((T, p, rho, qfrac, qcic, ncic, qs_ip * pf,
                                    ns_ip * pf, lcldm, pf, 0.0, qr_ip * pf,
                                    nr_ip * pf, qi, ni))
    return np.array(pts)


def write_column_inputs(cols, path):
    nlev = len(cols[0]["T"])
    with open(path, "w") as f:
        f.write(f"{len(cols)} {nlev} {DT!r}\n")
        for c in cols:
            rho = c["rho"]
            ncnst = float(c["N_c"][0])
            f.write(f"{ncnst:.17e}\n")
            pdel = np.diff(c["p_half"])
            lcl = np.maximum(c["cf"], 1e-4)
            for k in range(nlev):
                row = [c["T"][k], c["q_v"][k], c["q_c"][k], 0.0,
                       ncnst / rho[k] * lcl[k] if c["q_c"][k] > 0 else 0.0, 0.0,
                       c["q_r"][k], c["q_s"][k], c["N_r"][k] / rho[k], c["N_s"][k],
                       RELVAR, 1.0, c["p_full"][k], pdel[k],
                       c["cf"][k], c["cf"][k], c["cf"][k], 1.0]
                f.write(" ".join(f"{x:.17e}" for x in row) + "\n")


def load_column_outputs(path, ncol, nlev):
    lines = Path(path).read_text().split("\n")
    prect = np.array([float(line.split()[1]) for line in lines[:ncol]])
    body = np.array([[float(x) for x in line.split()] for line in lines[ncol:]
                     if line.strip()]).reshape(ncol, nlev, -1)
    out = {k: body[:, :, i + 2] for i, k in enumerate(ORACLE_COLS)}
    out["prect"] = prect
    return out


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.cam), str(a.build)],
                   check=True)
    cols = build_columns()
    ncol, nlev = len(cols), len(cols[0]["T"])
    fx = {f"col_{k}": np.stack([c[k] for c in cols]) for k in COL_FIELDS}
    fx["col_names"] = np.array([c["name"] for c in cols])
    fx["dt"] = np.array(DT)
    fx["relvar"] = np.array(RELVAR)
    fx["dcs"] = np.array(DCS)
    fin = a.build / "col_in.txt"
    fout = a.build / "col_out.txt"
    write_column_inputs(cols, fin)
    subprocess.run([str(a.build / "mg2_snow_driver"), str(fin), str(fout), "F"],
                   check=True)
    o = load_column_outputs(fout, ncol, nlev)
    for k in ORACLE_COLS:
        if k in ("sflx", "rflx"):
            fx[f"ora_{k}_sfc"] = o[k][:, -1]
        else:
            fx[f"ora_{k}"] = o[k]
    pts = build_points()
    pin = a.build / "pt_in.txt"
    pout = a.build / "pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_snow_procs"), str(pin), str(pout), repr(DCS)],
                   check=True)
    po = np.loadtxt(pout)
    for i, k in enumerate(PT_IN):
        fx[f"pt_in_{k}"] = pts[:, i]
    for i, k in enumerate(PT_OUT):
        fx[f"pt_out_{k}"] = po[:, i]
    meta = dict(
        oracle="ESCOMP/CAM", tag="cam_cesm2_1_rel_60", sha=_git(a.cam, "rev-parse", "HEAD"),
        files=["src/physics/cam/micro_mg2_0.F90", "src/physics/cam/micro_mg_utils.F90",
               "src/physics/cam/wv_sat_methods.F90"],
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        legoesm_sha=_git(HERE, "rev-parse", "HEAD"),
        generator=("python scripts/validate/oracle/morrison_mg2_ice_snow/"
                   f"make_fixture.py --cam <CAM> --build <dir> --out {a.out.name}"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


if __name__ == "__main__":
    main()
