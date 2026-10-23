"""Generate the stand-alone Seifert-Beheng (gSAM M2005, liquid only, SB2001)
oracle fixture by RUNNING the oracle.

Oracle: gSAM 1.8.8 ``micro_proc`` order -- ``satadj_liquid`` then
``M2005MICRO_GRAUPEL`` (SRC/MICRO_M2005/{microphysics.f90,module_mp_graupel.f90})
with ``doicemicro=.false.`` (ILIQ=1), ``dosb_warm_rain=.true.`` (IRAIN=1),
``dopredictNc=.false.`` (INUM=1) and ``dofix_pgam=.false.``, legoESM
saturation curve and constants fed in; built by
``scripts/validate/oracle/sb2001_warm_rain/build.sh`` (the shared gSAM driver,
``gsam_driver m2005 1 0 1 <Nc0> <dt> 0``).  No git tag/SHA exists; the pin is
the tarball http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz (sha256
0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c).

Columns (level 0 = surface, gSAM order), from gSAM's own case soundings with
condensate added here:

* ``warm15``: the five warm columns of the 2026-10-19 tier-2 audit
  (``gsam_satadj_liquid/make_fixture.py:build_columns``) at dt = 15 s,
  Nc0 = 100 cm^-3;
* ``warm60``: the same columns at dt = 60 s (more sedimentation sub-steps,
  conservation limiters);
* ``warm15_nc1000``: the same columns at Nc0 = 1000 cm^-3 (droplet shape at
  its lower bound);
* ``cold``: RCE 300 K sounding to 9 km (top levels supercooled, ILIQ=1 keeps
  them liquid) with rain whose mean size sits outside both slope bounds,
  sub-cloud rain in dry air, and threshold-size amounts; dt = 60 s,
  Nc0 = 300 cm^-3;
* ``stress``: the same sounding at the GCM time step dt = 1200 s and
  Nc0 = 10 cm^-3 (droplet shape at its upper bound): heavy cloud + rain (the
  q_c conservation limiter binds), light rain in RH-0.3 air (the q_r limiter
  binds), surface fog (cloud fallout at the surface), a dry column (no
  hydrometeor anywhere), and a just-saturated thin cloud (droplet slope at its
  upper bound);
* ``edge_rh`` / ``edge_qsmall``: two-level columns from the codex review
  (2026-10-20) -- the small-amount removal's RH test is evaluated once, before
  the rain removal moistens the air; rain exactly at QSMALL evaporates.

Usage (CPU node; needs gfortran and the gSAM source)::

    python scripts/validate/oracle/gsam_sb_liquid/make_fixture.py \\
        --gsam <dir containing SRC/MICRO_M2005 and CASES> --build <dir> \\
        --out tests/atmosphere/microphysics/oracle_fixtures/gsam_sb_liquid.npz
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
SATADJ_GEN = HERE.parent / "gsam_satadj_liquid" / "make_fixture.py"
FIELDS = ("p_full", "p_half", "T", "rho", "dz", "q_v", "q_c", "q_r", "n_r_kg")
GROUPS = (("warm15", 15.0, 100.0), ("warm60", 60.0, 100.0), ("cold", 60.0, 300.0),
          ("warm15_nc1000", 15.0, 1000.0),
          ("stress", 1200.0, 10.0), ("edge_rh", 0.01, 100.0),
          ("edge_qsmall", 600.0, 100.0))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cold_columns(gsam, sat):
    """Supercooled / clamped-slope / dry-sub-cloud columns (30 levels, 9 km)."""
    def cloud(z):
        return 6e-4 if 2000 <= z <= 7500 else 0.0

    def rain_big(z):          # mean diameter 3.5 mm: slope below LAMMINR
        return (2e-3, 3.5e-3) if z <= 6000 else (0.0, 1e-4)

    def rain_small(z):        # mean diameter 12 um: slope above LAMMAXR
        return (2e-4, 12e-6) if 2000 <= z <= 8500 else (0.0, 1e-4)

    def dry_below(col):       # sub-cloud air at RH 0.6 under the rain
        sel = col["z"] < 2000
        col["q_v"] = np.where(sel, 0.6 * sat._qsat(col["T"], col["p_full"]),
                              col["q_v"])
        col["q_r"][1] = 5e-9                        # threshold-size rain, RH 0.6
        col["n_r_kg"][1] = sat._nr(5e-9, 2e-4)

    mk = sat.make_col
    return {
        "cold_bigdrops": mk(gsam, "RCE", 9000., 300., cloud, rain_big,
                            fname="snd_300K", extra=dry_below),
        "cold_smalldrops": mk(gsam, "RCE", 9000., 300., cloud, rain_small,
                              fname="snd_300K"),
        "cold_cloudonly": mk(gsam, "RCE", 9000., 300., cloud,
                             lambda z: (0.0, 1e-4), fname="snd_300K"),
    }


def stress_columns(gsam, sat):
    """GCM-step columns that bind the limiters (30 levels, 9 km)."""
    def none(z):
        return (0.0, 1e-4)

    def rh(frac, lo, hi):
        def f(col):
            sel = (col["z"] >= lo) & (col["z"] <= hi)
            col["q_v"] = np.where(sel, frac * sat._qsat(col["T"], col["p_full"]),
                                  col["q_v"])
        return f

    def thin(col):
        sel = (col["z"] >= 1000) & (col["z"] <= 3000)
        col["q_v"] = np.where(sel, sat._qsat(col["T"], col["p_full"]), col["q_v"])
        col["q_c"] = np.where(sel, 1e-8, col["q_c"])

    mk = sat.make_col
    return {
        "stress_heavy": mk(gsam, "RCE", 9000., 300.,
                           lambda z: 2e-3 if 1000 <= z <= 5000 else 0.0,
                           lambda z: (5e-3, 1e-3) if z <= 5000 else (0.0, 1e-4),
                           fname="snd_300K"),
        "stress_evap": mk(gsam, "RCE", 9000., 300., lambda z: 0.0,
                          lambda z: (3e-7, 5e-4) if 2000 <= z <= 6000 else (0.0, 1e-4),
                          fname="snd_300K", extra=rh(0.3, 0.0, 9000.0)),
        "stress_fog": mk(gsam, "RCE", 9000., 300.,
                         lambda z: 5e-4 if z < 600 else 0.0, none, fname="snd_300K"),
        "stress_dry": mk(gsam, "RCE", 9000., 300., lambda z: 0.0, none,
                         fname="snd_300K", extra=rh(0.5, 0.0, 9000.0)),
        "stress_thincloud": mk(gsam, "RCE", 9000., 300., lambda z: 0.0, none,
                               fname="snd_300K", extra=thin),
    }


def edge_columns(sat, tag):
    """Two-level columns (level 0 = surface) at 1000 hPa, rho = 1, dz = 100 m."""
    from legoesm import constants as C
    T = 280.0 if tag == "edge_rh" else 200.0
    p = np.full(2, 1.0e5)
    qs = sat._qsat(np.full(2, T), p)
    col = dict(z=np.array([50.0, 150.0]), dz=np.full(2, 100.0), p_full=p,
               p_half=np.array([1.0e5 + 500.0, 1.0e5, 1.0e5 - 500.0]),
               T=np.full(2, T), rho=np.ones(2))
    if tag == "edge_rh":
        col.update(q_v=np.array([0.9 * qs[0] - 1e-12, qs[1]]),
                   q_c=np.array([0.0, 1e-4]), q_r=np.array([0.0, 1e-5]),
                   n_r_kg=np.array([0.0, 10.0]))
    else:
        qr = 1.0e-14
        col.update(q_v=0.95 * qs, q_c=np.zeros(2), q_r=np.array([0.0, qr]),
                   n_r_kg=np.array([0.0, qr * 50000.0 ** 3 / (np.pi * C.rho_water)]))
    return {tag: col}


def run_group(exe, cols, dt, nc0, work, tag):
    names = list(cols)
    nlev = len(cols[names[0]]["T"])
    assert all(len(cols[n]["T"]) == nlev for n in names)
    fin, fout = work / f"sbl_{tag}_in.txt", work / f"sbl_{tag}_out.txt"
    with open(fin, "w") as f:
        f.write(f"{len(names)} {nlev}\n")
        for n in names:
            c = cols[n]
            for k in range(nlev):
                f.write(" ".join(f"{v:.17e}" for v in (
                    c["p_full"][k], c["rho"][k], c["dz"][k], 0.0, c["T"][k],
                    c["q_v"][k], c["q_c"][k], c["q_r"][k], c["n_r_kg"][k],
                    0, 0, 0, 0, 0, 0)) + "\n")
    # m2005 <irain=1 SB2001> <fixpgam=0> <do_satadj=1> <Nc0> <dt> <icemicro=0>
    log = subprocess.run([str(exe), "m2005", "1", "0", "1", f"{nc0}", f"{dt}",
                          "0", str(fin), str(fout)], check=True,
                         capture_output=True, text=True).stdout
    assert "M2005_OK" in log and "nhit_maxiter= 0" in " ".join(log.split()), log
    tok = np.array(open(fout).read().split(), dtype=float).reshape(
        len(names), nlev * 72 + 2)
    lev = tok[:, :nlev * 72].reshape(len(names), nlev, 72)
    out = dict(T=lev[:, :, 0], q_v=lev[:, :, 1], q_c=lev[:, :, 2],
               q_r=lev[:, :, 3], n_r_kg=lev[:, :, 4], q_i=lev[:, :, 5],
               precip=tok[:, -2] / dt, snow=tok[:, -1] / dt,
               # ODIAG(j) = lev[..., 11 + j]: process rates before (1, 7) and
               # after (21, 27) the conservation limiters -- coverage only.
               prc_raw=lev[:, :, 12], prc_fin=lev[:, :, 32],
               pre_raw=lev[:, :, 18], pre_fin=lev[:, :, 38])
    assert np.all(out["q_i"] == 0.0) and np.all(out["snow"] == 0.0)
    return names, out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsam", required=True,
                    help="dir containing SRC/MICRO_M2005 and CASES")
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bld = Path(a.build)
    shared = _load(BUILD_DIR / "make_fixture.py", "sb2001_make_fixture")
    sat = _load(SATADJ_GEN, "satadj_make_fixture")
    prov = shared.provenance(a.gsam)
    prov["generator_sha256"]["gsam_sb_liquid/make_fixture.py"] = shared._sha(
        Path(__file__))
    subprocess.run(["bash", str(BUILD_SH), a.gsam, str(bld)], check=True)
    warm = sat.build_columns(a.gsam)
    cold = cold_columns(a.gsam, sat)
    stress = stress_columns(a.gsam, sat)
    arrs, logs = {}, {}
    for tag, dt, nc0 in GROUPS:
        cols = ({"cold": cold, "stress": stress}.get(tag, warm)
                if not tag.startswith("edge") else edge_columns(sat, tag))
        names, out, log = run_group(bld / "gsam_driver", cols, dt, nc0, bld, tag)
        logs[tag] = log.strip()
        arrs[f"{tag}_names"] = np.array(names)
        arrs[f"{tag}_dt"] = dt
        arrs[f"{tag}_Nc0_cm3"] = nc0
        for k in FIELDS:
            arrs[f"{tag}_{k}"] = np.stack([cols[n][k] for n in names])
        for k in ("T", "q_v", "q_c", "q_r", "n_r_kg", "precip", "prc_raw",
                  "prc_fin", "pre_raw", "pre_fin"):
            arrs[f"{tag}_ora_{k}"] = out[k]
    meta = dict(
        oracle="gSAM 1.8.8 micro_proc: satadj_liquid + M2005MICRO_GRAUPEL",
        tarball="http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz",
        sha256="0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c",
        settings=("doicemicro=F (ILIQ=1), dosb_warm_rain=T (IRAIN=1), "
                  "dopredictNc=F (INUM=1), dofix_pgam=F; satadj stock "
                  "tolerance thresh_mult=1, maxiter=20; legoESM q_sat curve "
                  "and constants"),
        groups=[list(g) for g in GROUPS], **prov, driver_logs=logs,
        sounding_sha256={f"{c}/{f}": shared._sha(Path(a.gsam) / "CASES" / c / f)
                         for c, f in (("BOMEX", "snd"), ("RICO", "snd"),
                                      ("DYCOMS_RF01", "snd"), ("RCE", "snd_300K"))},
        generator=("python scripts/validate/oracle/gsam_sb_liquid/make_fixture.py "
                   "--gsam <gSAM1.8.8> --build <dir> --out gsam_sb_liquid.npz"))
    np.savez_compressed(a.out, meta=json.dumps(meta), **arrs)
    print("FIXTURE_OK", a.out)


if __name__ == "__main__":
    main()
