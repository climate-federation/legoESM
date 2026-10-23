"""Generate the CAM6 MG2 warm-rain oracle fixture by RUNNING the oracle.

Oracle: ESCOMP/CAM ``cam_cesm2_1_rel_60`` (commit
a03b84b7c4e34f965b115686f22a043b85739e56), ``src/physics/cam/micro_mg2_0.F90``
(full ``micro_mg_tend``), ``micro_mg_utils.F90`` and ``wv_sat_methods.F90``
(CAM's Goff-Gratch saturation) compiled UNMODIFIED.  Settings are CAM6 + CLUBB: precip_frac_method
'in_cloud', allow_sed_supersat=.false., microp_uniform=.false., nccons with
the column droplet number, legoESM's constants (``mg2_driver.F90``).

Two parts, written to one ``.npz``:
  * ``col_*``: six warm columns (CAM L32, published-sounding shapes; built
    by :func:`build_columns`, reused from the 2026-10-18 tier-1 audit) run
    through the whole ``micro_mg_tend`` at relvar 1, 2, 10, dt = 600 s;
  * ``pt_*``: a point grid through the warm-rain kernels
    (``mg2_procs_driver.F90``).

Usage (CPU node; needs gfortran and the CAM checkout)::

    python scripts/validate/oracle/morrison_mg2_warm_rain/make_fixture.py \\
        --cam /burg/glab/users/pg2328/oracles/CAM --build /tmp/mg2_bld \\
        --out tests/atmosphere/microphysics/oracle_fixtures/mg2_warm_rain.npz
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
RELVARS = (1.0, 2.0, 10.0)
COL_FIELDS = ("T", "q_v", "q_c", "cf", "q_r", "N_r", "N_c", "p_full",
              "p_half", "rho")
ORACLE_COLS = ("tlat", "qvlat", "qctend", "qrtend", "nctend", "nrtend",
               "nevapr", "pratot", "prctot", "qrsedten", "qcsedten", "umr",
               "qcrat", "rflx_bot", "lflx_bot")
PT_IN = ("qcic", "ncic", "rho", "relvar", "qr", "nr", "T", "p", "qfrac",
         "lcldm", "precip_frac", "qiic")
PT_OUT = ("prc", "nprc", "nprc1", "ncic_adj", "pra", "npra", "nragg", "lamr",
          "n0r", "nric", "qric", "pre", "qvl")


def build_columns():
    """The six tier-1 audit columns (level 0 = model top)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm import constants as C
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0
    from legoesm.thermo import saturation_specific_humidity
    hyai = np.array(CAM6_L32_HYAI)
    hybi = np.array(CAM6_L32_HYBI)
    nlev = len(hyai) - 1

    def qsat(T, p):
        return np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))

    cols = []

    def base(ps, Ts, gam, Ttrop):
        ph = hyai * CAM6_L32_P0 + hybi * ps
        pf = 0.5 * (ph[1:] + ph[:-1])
        T = np.maximum(Ts * (pf / ps) ** (C.R_d * gam / C.g), Ttrop)
        return ph, pf, T

    def lam_to_nr(lam, rho, qr):  # exponential PSD: N_vol = lam^3 rho q/(pi rho_w)
        return lam ** 3 * rho * qr / (math.pi * C.rho_water)

    def finish(name, ph, pf, T, rh, qc, cf, qr, lam, Nc):
        rho = pf / (C.R_d * T)
        rh = np.where(T < 253.0, np.minimum(rh, 0.3), rh)   # ice paths idle
        qv = rh * qsat(T, pf)
        warm = T > 275.0
        qc = np.where(warm, qc, 0.0)
        qr = np.where(warm, qr, 0.0)
        cf = np.where(qc > 0, cf, 0.0)
        Nr = np.where(qr > 0, lam_to_nr(lam, rho, qr), 0.0)
        cols.append(dict(name=name, p_half=ph, p_full=pf, T=T, q_v=qv, q_c=qc,
                         cf=cf, q_r=qr, N_r=Nr, N_c=np.full(nlev, Nc), rho=rho))

    # tropical deep convective (warm-rain part below the freezing level)
    ph, pf, T = base(101000.0, 300.0, 6.2e-3, 195.0)
    P = pf / 100.0
    rh = np.where(P > 950, 0.80, np.where(P > 900, 0.85, 0.5))
    cl = (P <= 900) & (P >= 600)
    rh = np.where(cl, 1.0 - 0.05 * (1 - 0.35), rh)
    rh = np.where(P < 400, 0.3, rh)
    qc = np.where(cl, 0.35 * 1.0e-3, 0.0)
    qr = np.where(P >= 600, np.interp(P, [600, 900, 1100], [2e-4, 6e-4, 5e-4]), 0.0)
    finish("trop_deep", ph, pf, T, rh, qc, 0.35, qr, 2500.0, 50.0e6)

    # subtropical stratocumulus (DYCOMS-II RF02-shaped)
    ps = 101780.0
    ph = hyai * CAM6_L32_P0 + hybi * ps
    pf = 0.5 * (ph[1:] + ph[:-1])
    P = pf / 100.0
    kap = C.R_d / C.c_pd
    th_l = np.where(P >= 925, 288.3, 297.0 + 0.04 * (925 - P))
    qt = np.where(P >= 925, 9.45e-3, 1.5e-3)
    T = th_l * (pf / 1.0e5) ** kap
    for _ in range(60):
        qs = qsat(T, pf)
        ql = np.maximum(qt - qs, 0.0)
        Tn = th_l * (pf / 1.0e5) ** kap + C.L_v / C.c_pd * ql  # latent-ok: tier-1 audit column recipe (fixture inputs)
        T = 0.5 * T + 0.5 * Tn
    T = np.maximum(T, 200.0)
    qs = qsat(T, pf)
    ql = np.maximum(qt - qs, 0.0)
    cfc = 0.95
    rh = np.where(ql > 0, 1.0 - 0.05 * (1 - cfc), np.minimum(qt / qs, 0.9))
    qr = np.where(P >= 925, np.where(ql > 0, 4e-5, 1.5e-5), 0.0)
    finish("subtrop_sc", ph, pf, T, rh, ql, cfc, qr, 15000.0, 55.0e6)

    # midlatitude continental warm stratiform rain
    ph, pf, T = base(100000.0, 295.0, 6.5e-3, 210.0)
    P = pf / 100.0
    cl = (P <= 820) & (P >= 680)
    rh = np.where(cl, 1.0 - 0.05 * (1 - 0.7), np.where(P > 820, 0.75, 0.5))
    qc = np.where(cl, 0.7 * 4.0e-4, 0.0)
    qr = np.where(P >= 680, np.interp(P, [680, 820, 1100], [1.5e-4, 3e-4, 2e-4]), 0.0)
    finish("midlat_cont", ph, pf, T, rh, qc, 0.7, qr, 3000.0, 200.0e6)

    # trade cumulus (RICO-shaped)
    ph, pf, T = base(101540.0, 299.2, 6.0e-3, 200.0)
    P = pf / 100.0
    cl = (P <= 950) & (P >= 800)
    rh = np.where(cl, 0.92, np.where(P > 950, 0.80, 0.6))
    qc = np.where(cl, 0.06 * 1.5e-3, 0.0)
    qr = np.where(P >= 800, np.where(cl, 3e-5, 2e-5), 0.0)
    finish("trade_cu", ph, pf, T, rh, qc, 0.06, qr, 4000.0, 70.0e6)

    # edge / threshold column
    ph, pf, T = base(100000.0, 290.0, 6.5e-3, 210.0)
    rho = pf / (C.R_d * T)
    nl = nlev
    rh = np.full(nl, 0.3)
    qc = np.zeros(nl)
    cf = np.zeros(nl)
    qr = np.zeros(nl)
    lam = np.full(nl, 5000.0)
    k = nl - 1
    qc[k] = 5e-9; cf[k] = 0.1; rh[k] = 0.97
    qc[k - 1] = 5e-10; cf[k - 1] = 0.1; rh[k - 1] = 0.97
    qc[k - 2] = 2e-6; cf[k - 2] = 0.0; rh[k - 2] = 0.97
    qr[k - 2] = 1e-4
    qr[k - 3] = 1e-16; lam[k - 3] = 5000.0
    qr[k - 4] = 5e-4; lam[k - 4] = 800.0
    qc[k - 5] = 3e-3; cf[k - 5] = 0.15; qr[k - 5] = 2.5e-3; lam[k - 5] = 2500.0
    rh[k - 5] = 0.99
    qc[k - 6] = 4e-4; cf[k - 6] = 1.0; rh[k - 6] = 1.02; qr[k - 6] = 1e-5
    Nr = np.where(qr > 0, lam ** 3 * rho * qr / (math.pi * C.rho_water), 0.0)
    qv = rh * qsat(T, pf)
    cols.append(dict(name="edge", p_half=ph, p_full=pf, T=T, q_v=qv, q_c=qc,
                     cf=np.where(qc > 0, cf, 0.0), q_r=qr, N_r=Nr,
                     N_c=np.full(nl, 100.0e6), rho=rho))

    # polar dry
    ph, pf, T = base(100500.0, 245.0, 5.0e-3, 200.0)
    finish("polar_dry", ph, pf, T, np.full(nlev, 0.6), np.zeros(nlev), 0.0,
           np.zeros(nlev), 3000.0, 30.0e6)
    return cols


def build_points():
    """Point grid through the warm-rain kernels (columns of PT_IN)."""
    from legoesm import constants as C
    T, p = 285.0, 85000.0
    rho = p / (C.R_d * T)
    fracs = ((1.0, 1.0, 0.0), (0.3, 0.3, 0.0), (0.3, 0.8, 0.0),
             (1.0e-4, 0.5, 0.0), (0.3, 0.8, 2.0e-6))   # lcldm, precip_frac, qiic
    pts = []
    for qcic in (0.0, 1e-9, 2e-8, 5e-7, 1e-4, 5e-3):
        for Nc in (30e6, 300e6):
            for qr_ip in (0.0, 1e-17, 1e-6, 1e-4, 8e-3, 2e-2):
                for lam in (900.0, 3000.0, 1.0e5):
                    for relvar in (1.0, 10.0):
                        for lcldm, pf, qiic in fracs:
                            for qfrac in (0.7, 1.05):
                                nr_ip = lam ** 3 * qr_ip / (math.pi * C.rho_water)
                                pts.append((qcic, Nc / rho, rho, relvar, qr_ip * pf,
                                            nr_ip * pf, T, p, qfrac, lcldm, pf, qiic))
    return np.array(pts)


def write_column_inputs(cols, path, relvar):
    """``relvar``: a scalar, or a per-level ``(nlev,)`` profile."""
    nlev = len(cols[0]["T"])
    relvar = np.broadcast_to(np.asarray(relvar, dtype=np.float64), (nlev,))
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
                       c["q_r"][k], 0.0, c["N_r"][k] / rho[k], 0.0,
                       relvar[k], 1.0, c["p_full"][k], pdel[k],
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
    fx["relvars"] = np.array(RELVARS)
    for rv in RELVARS:
        fin = a.build / f"col_in_rv{rv:g}.txt"
        fout = a.build / f"col_out_rv{rv:g}.txt"
        write_column_inputs(cols, fin, rv)
        subprocess.run([str(a.build / "mg2_driver"), str(fin), str(fout), "F"],
                       check=True)
        o = load_column_outputs(fout, ncol, nlev)
        for k in ("prctot", "pratot", "nevapr", "qrtend", "nrtend", "qrsedten",
                  "qcrat"):
            fx[f"ora_rv{rv:g}_{k}"] = o[k]
        fx[f"ora_rv{rv:g}_rain_sfc_flux"] = o["rflx_bot"][:, -1]
    pts = build_points()
    pin = a.build / "pt_in.txt"
    pout = a.build / "pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_procs"), str(pin), str(pout)], check=True)
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
        generator=("python scripts/validate/oracle/morrison_mg2_warm_rain/"
                   f"make_fixture.py --cam <CAM> --build <dir> --out {a.out.name}"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


if __name__ == "__main__":
    main()
