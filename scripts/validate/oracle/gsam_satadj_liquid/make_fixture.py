"""Generate the gSAM liquid saturation-adjustment oracle fixture by RUNNING
the oracle.

Oracle: gSAM 1.8.8 ``satadj_liquid`` (SRC/MICRO_M2005/microphysics.f90
:1638-1720), body verbatim, stock tolerance (thresh_mult = 1, maxiter = 20),
legoESM saturation curve fed in; built by
``scripts/validate/oracle/sb2001_warm_rain/build.sh`` (same gSAM driver).  No
git tag/SHA exists; the pin is the tarball
http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz (sha256
0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c).
The driver maps (T, q_v, q_c) to micro_proc's inputs q_t = q_v + q_c and
T_l = T - (L_v/c_pd) q_c, as micro_proc does.

Whole call: five warm columns (the audit's, from gSAM's CASES soundings)
through gSAM's micro_proc order -- satadj_liquid, then M2005MICRO_GRAUPEL
(KK2000, IRAIN=0, the M2005 default; Nc0 = 100 cm^-3; dt = 15 s); the
post-step T and q_v are stored.

Points: the 2026-10-19 tier-2 audit grid (T 230-310 K, p 40-1000 hPa,
supersaturated cloud-free points S = 1.0005-1.5 and sub/near-saturated cloudy
points), including p = 40 hPa where gSAM does not adjust.

Usage (CPU node; needs gfortran and the gSAM source)::

    python scripts/validate/oracle/gsam_satadj_liquid/make_fixture.py \\
        --gsam <dir containing SRC/MICRO_M2005> --build <dir> \\
        --out tests/atmosphere/microphysics/oracle_fixtures/gsam_satadj_liquid.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE.parent / "sb2001_warm_rain"     # shared gSAM build + provenance
BUILD_SH = BUILD_DIR / "build.sh"


def _shared():
    spec = importlib.util.spec_from_file_location(
        "sb2001_make_fixture", BUILD_DIR / "make_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
TEMPS = (230.0, 255.0, 273.0, 285.0, 300.0, 310.0)
PRESS = (4000.0, 20000.0, 50000.0, 85000.0, 100000.0)
SUPERSAT = (1.0005, 1.01, 1.05, 1.2, 1.5)
CLOUDY = ((0.8, 1e-5), (0.8, 1e-3), (0.95, 1e-4), (0.99, 1e-3), (0.999, 1e-5),
          (1.0, 1e-4))


def build_points():
    """Rows (T, q_v, q_c, p)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.thermo import saturation_specific_humidity
    rows = []
    for T in TEMPS:
        for p in PRESS:
            qs = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
            for S in SUPERSAT:
                rows.append([T, S * qs, 0.0, p])
            for S, qc in CLOUDY:
                rows.append([T, S * qs, qc, p])
    return np.array(rows, dtype=np.float64)


# --- whole-call columns (the 2026-10-19 tier-2 audit columns) --------------
# Built from gSAM's own case soundings (CASES/<case>/snd) with condensate added
# here; level 0 = surface (gSAM order).  Morrison's sam flavour is compared on
# the post-step T and q_v, which droplet settling and rain sedimentation (the
# scheme's remaining documented departures) do not touch within one call.
DT_COL = 15.0
NC0_COL = 100.0
SOUNDINGS = (("BOMEX", "snd"), ("RICO", "snd"), ("DYCOMS_RF01", "snd"),
             ("RCE", "snd_300K"))


def read_snd(gsam, case, fname="snd"):
    lines = open(Path(gsam) / "CASES" / case / fname).read().splitlines()
    hdr = lines[1].replace(",", " ").split()
    nlev, p0 = int(float(hdr[1])), float(hdr[2])
    z, th, q = [], [], []
    for ln in lines[2:2 + nlev]:
        v = ln.split()
        z.append(float(v[0])); th.append(float(v[2])); q.append(float(v[3]) * 1e-3)
    return np.array(z), np.array(th), np.array(q), p0 * 100.0


def hydro(th, dzv, p0):
    """Bottom-up hydrostatic integration with legoESM g/R_d/c_pd; T from theta."""
    from legoesm import constants as C
    kap = C.R_d / C.c_pd
    n = len(th)
    ph = np.zeros(n + 1); ph[0] = p0
    pf = np.zeros(n); T = np.zeros(n)
    for k in range(n):
        p_top = ph[k]
        for it in range(30):
            pm = ph[k] if it == 0 else 0.5 * (ph[k] + p_top)
            Tk = th[k] * (pm / 1.0e5) ** kap
            p_top = ph[k] * np.exp(-C.g * dzv / (C.R_d * Tk))
        ph[k + 1] = p_top
        pf[k] = 0.5 * (ph[k] + ph[k + 1])
        T[k] = th[k] * (pf[k] / 1.0e5) ** kap
    rho = (ph[:-1] - ph[1:]) / (C.g * dzv)
    return pf, ph, T, rho


def _qsat(T, p):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.thermo import saturation_specific_humidity
    return np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))


def _nr(qr, D):
    from legoesm import constants as C
    return qr / (np.pi * C.rho_water * D ** 3)


def make_col(gsam, case, ztop, dzv, cloud, rain, fname="snd", extra=None):
    z0, th0, q0, p0 = read_snd(gsam, case, fname)
    zc = (np.arange(int(round(ztop / dzv))) + 0.5) * dzv
    th = np.interp(zc, z0, th0); qt = np.interp(zc, z0, q0)
    pf, ph, T, rho = hydro(th, dzv, p0)
    qs = _qsat(T, pf)
    qc = np.array([cloud(z) for z in zc])
    qv = np.where(qc > 0, qs, np.minimum(qt, 0.98 * qs))
    qr = np.array([rain(z)[0] for z in zc])
    Dr = np.array([rain(z)[1] for z in zc])
    col = dict(z=zc, dz=np.full_like(zc, dzv), p_full=pf, p_half=ph, T=T, rho=rho,
               q_v=qv, q_c=qc, q_r=qr, n_r_kg=np.where(qr > 0, _nr(qr, Dr), 0.0))
    if extra is not None:
        extra(col)
    return col


def build_columns(gsam):
    def lin(z, a, b, top):
        return top * (z - a) / (b - a) if a <= z <= b else 0.0

    def ss(col):
        sel = (col["z"] >= 600) & (col["z"] <= 1500)
        col["q_v"] = np.where(sel, 1.01 * _qsat(col["T"], col["p_full"]), col["q_v"])

    def edge(col):
        col["q_c"][3] = 1e-9; col["q_r"][3] = 1e-12; col["n_r_kg"][3] = _nr(1e-12, 1e-4)
        col["q_c"][6] = 5e-7; col["q_r"][8] = 5e-9; col["n_r_kg"][8] = _nr(5e-9, 2e-4)
        col["q_c"][10] = 1e-4                       # cloud in RH 0.9 air
        col["q_v"][10] = 0.9 * _qsat(col["T"][10], col["p_full"][10])

    cols = {
        "bomex_cu": make_col(gsam, "BOMEX", 3000., 100., lambda z: lin(z, 600, 1500, 3e-4),
                             lambda z: (1e-6, 100e-6) if 600 <= z <= 1500 else (0., 1e-4)),
        "rico_rain": make_col(gsam, "RICO", 4000., 4000. / 30, lambda z: lin(z, 600, 2600, 1e-3),
                              lambda z: (5e-4 * min(1.0, (2600 - z) / 2000.), 0.6e-3)
                              if z <= 2600 else (0., 1e-3)),
        "dycoms_sc": make_col(gsam, "DYCOMS_RF01", 1200., 40., lambda z: lin(z, 600, 840, 4.5e-4),
                              lambda z: (1e-5 * min(1.0, z / 840.), 80e-6)
                              if z <= 840 else (0., 1e-4)),
        "bomex_supersat": make_col(gsam, "BOMEX", 3000., 100., lambda z: 0.0,
                                   lambda z: (0., 1e-4), extra=ss),
        "edge_dry": make_col(gsam, "BOMEX", 3000., 100., lambda z: 0.0,
                             lambda z: (0., 1e-4), extra=edge),
    }
    assert len({len(c["T"]) for c in cols.values()}) == 1
    return cols


def run_columns(exe, cols, work):
    names = list(cols)
    nlev = len(cols[names[0]]["T"])
    fin, fout = work / "cols_in.txt", work / "cols_out.txt"
    with open(fin, "w") as f:
        f.write(f"{len(names)} {nlev}\n")
        for n in names:
            c = cols[n]
            for k in range(nlev):
                f.write(" ".join(f"{v:.17e}" for v in (
                    c["p_full"][k], c["rho"][k], c["dz"][k], 0.0, c["T"][k], c["q_v"][k],
                    c["q_c"][k], c["q_r"][k], c["n_r_kg"][k], 0, 0, 0, 0, 0, 0)) + "\n")
    # m2005 <irain=0 (KK2000)> <fixpgam=0> <do_satadj=1> <Nc0> <dt> <icemicro=1>
    log = subprocess.run([str(exe), "m2005", "0", "0", "1", f"{NC0_COL}", f"{DT_COL}",
                          "1", str(fin), str(fout)], check=True, capture_output=True,
                         text=True).stdout
    assert "M2005_OK" in log, log
    tok = np.array(open(fout).read().split(), dtype=float).reshape(len(names), nlev * 72 + 2)
    lev = tok[:, :nlev * 72].reshape(len(names), nlev, 72)
    return names, dict(T=lev[:, :, 0], q_v=lev[:, :, 1], precip=tok[:, -2] / DT_COL), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsam", required=True, help="dir containing SRC/MICRO_M2005")
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bld = Path(a.build)
    shared = _shared()
    prov = shared.provenance(a.gsam)
    prov["generator_sha256"]["gsam_satadj_liquid/make_fixture.py"] = shared._sha(
        Path(__file__))
    subprocess.run(["bash", str(BUILD_SH), a.gsam, str(bld)], check=True)
    pts = build_points()
    fin, fout = bld / "satadj_in.txt", bld / "satadj_out.txt"
    with open(fin, "w") as f:
        f.write(f"{len(pts)}\n")
        for r in pts:
            f.write(" ".join(f"{v:.17e}" for v in r) + "\n")
    log = subprocess.run([str(bld / "gsam_driver"), "satadj", "lego", "1", "20",
                          str(fin), str(fout)], check=True, capture_output=True,
                         text=True).stdout
    assert "SATADJ_OK" in log, log
    o = np.loadtxt(fout)
    assert o.shape == (len(pts), 4), o.shape
    meta = dict(
        oracle="gSAM 1.8.8 SRC/MICRO_M2005/microphysics.f90 satadj_liquid",
        tarball="http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz",
        sha256="0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c",
        settings="stock tolerance thresh_mult=1, maxiter=20; legoESM q_sat curve",
        **prov, driver_log=log.strip(),
        generator="python scripts/validate/oracle/gsam_satadj_liquid/make_fixture.py "
                  "--gsam <gSAM1.8.8> --build <dir> --out gsam_satadj_liquid.npz")
    cols = build_columns(a.gsam)
    names, oc, clog = run_columns(bld / "gsam_driver", cols, bld)
    meta["column_driver_log"] = clog.strip()
    meta["sounding_sha256"] = {f"{c}/{f}": shared._sha(Path(a.gsam) / "CASES" / c / f)
                               for c, f in SOUNDINGS}
    col_arrs = {f"col_{k}": np.stack([cols[n][k] for n in names])
                for k in ("p_full", "p_half", "T", "rho", "dz", "q_v", "q_c", "q_r", "n_r_kg")}
    np.savez_compressed(
        a.out, meta=json.dumps(meta), T=pts[:, 0], q_v=pts[:, 1], q_c=pts[:, 2],
        p=pts[:, 3], ora_T=o[:, 0], ora_q_v=o[:, 1], ora_q_c=o[:, 2],
        ora_maxiter_hit=o[:, 3], col_names=np.array(names), col_dt=DT_COL,
        col_Nc0_cm3=NC0_COL, ora_col_T=oc["T"], ora_col_q_v=oc["q_v"],
        ora_col_precip=oc["precip"], **col_arrs)
    print("FIXTURE_OK", a.out, len(pts))


if __name__ == "__main__":
    main()
