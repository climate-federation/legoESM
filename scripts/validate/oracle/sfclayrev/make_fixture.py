"""Generate tests/atmosphere/oracle_fixtures/sfclayrev.npz by RUNNING the WRF
revised MM5 surface-layer oracle (WRF v4.8.0 phys/module_sf_sfclayrev.F ->
phys/physics_mmm/sf_sfclayrev.F90, namelist defaults isfflx=1, isftcflx=0,
iz0tlnd=0, shalwater_z0=0, kim_tofd=.false., lakemask=0).

usage:
  bash build.sh <WRF> <build>
  python make_fixture.py --driver <build>/sfclayrev_driver \
      --out tests/atmosphere/oracle_fixtures/sfclayrev.npz

Columns: named regime columns (land/water x stable/neutral/unstable, calm
wind, the ust < 0.001 branch, |br| > 250 on both signs, z/L past the 10.0
look-up table, xland exactly 1.5, theta_1 == theta_g, qsfc <= 0) plus 40
seeded random columns.  Each column runs NSTEP = 3 consecutive calls with the
inout state carried as WRF carries it between steps, for scm_force_flux = 0
and 1.  Constants are legoESM's (``legoesm.constants``) plus WRF's SVP1-3.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants as C

ORACLE = dict(repo="wrf-model/WRF", tag="v4.8.0",
              sha="06d4240ae989cc3e50af412bb472df3d9048783c",
              physics_mmm_sha="550b5b494758ad2756394e93cb08b5709b8feb83",
              files=["phys/module_sf_sfclayrev.F", "phys/physics_mmm/sf_sfclayrev.F90",
                     "phys/ccpp_kind_types.F"])
IN = ("u1", "v1", "t1", "qv1", "p1", "dz1", "psfc", "tsk", "xland", "mavail", "pblh", "dx",
      "ust", "mol", "znt", "hfx", "qfx", "qsfc", "ustm")
OUT = ("ust", "mol", "znt", "hfx", "qfx", "lh", "qsfc", "br", "fm", "fh", "psim", "psih",
       "wspd", "u10", "v10", "th2", "t2", "q2", "zol", "rmol", "regime", "gz1oz0", "chs",
       "chs2", "cqs2", "cpm", "flhc", "flqc", "qgh", "ustm", "ck", "cka", "cd", "cda")
NSTEP = 3
SVP = (0.6112, 17.67, 29.65)   # WRF module_model_constants SVP1-3


def constants_vector():
    # cp g rovcp r xlv svp1 svp2 svp3 svpt0 ep1 ep2 karman p1000mb (driver order)
    return [C.c_pd, C.g, C.R_d / C.c_pd, C.R_d, C.L_v, *SVP, C.T_freeze,  # latent-ok: WRF sfclayrev oracle constant
            C.R_v / C.R_d - 1.0, C.R_d / C.R_v, C.kappa_von_karman, C.p_ref]


def base(**kw):
    d = dict(u1=5.0, v1=2.0, t1=290.0, qv1=0.008, p1=99000.0, dz1=50.0, psfc=99600.0,
             tsk=290.0, xland=1.0, mavail=1.0, pblh=800.0, dx=1.0e5, ust=0.3, mol=0.0,
             znt=0.1, hfx=50.0, qfx=5.0e-5, qsfc=0.009, ustm=0.3)
    d.update(kw)
    return d


def named_columns():
    W = dict(xland=2.0, znt=1.0e-4)
    cols = {
        "land_unstable": base(tsk=300.0, hfx=200.0, qfx=1.0e-4, pblh=1200.0),
        "land_stable": base(tsk=282.0, hfx=-20.0, qfx=0.0, pblh=150.0),
        "land_mol_neg_to_neutral": base(tsk=284.0, mol=-0.1),     # br>0 clamped to 0
        "water_unstable": base(tsk=296.0, **W),
        "water_stable": base(tsk=286.0, **W),
        "water_calm": base(u1=0.0, v1=0.0, tsk=291.0, dx=0.0, **W),
        "land_calm_floor": base(u1=0.0, v1=0.0, pblh=0.0, hfx=0.0, qfx=0.0, dx=1000.0,
                                tsk=289.5),
        "land_ust_tiny": base(tsk=299.0, ust=5.0e-4),
        "land_br_above_250": base(u1=0.0, v1=0.0, pblh=0.0, hfx=0.0, qfx=0.0, dx=0.0,
                                  tsk=275.0),
        "land_br_below_m250": base(u1=0.0, v1=0.0, pblh=0.0, hfx=0.0, qfx=0.0, dx=0.0,
                                   tsk=305.0),
        "land_thin_layer_stable": base(dz1=4.0, tsk=280.0, u1=2.0, v1=0.0),
        "land_thin_layer_unstable": base(dz1=4.0, tsk=303.0, u1=1.0, v1=0.5, znt=0.5),
        "xland_exactly_1p5": base(xland=1.5, znt=1.0e-3, tsk=294.0),
        "land_qsfc_nonpos": base(qsfc=0.0, tsk=293.0, mavail=0.4),
        "water_strong_wind": base(u1=25.0, v1=-10.0, tsk=300.0, ust=0.9, **W),
        "water_charnock_cap": base(u1=45.0, v1=5.0, tsk=292.0, ust=1.6, **W),
        "water_ust_tiny": base(tsk=293.0, ust=5.0e-4, **W),
    }
    # theta_1 == theta_g to within 1e-5 K: flhc = 0 branch
    d = base(tsk=0.0, qv1=0.0, qsfc=0.0)
    rovcp = C.R_d / C.c_pd
    thx = d["t1"] * (C.p_ref * 0.001 / (d["p1"] / 1000.0)) ** rovcp
    d["tsk"] = thx / (C.p_ref / d["psfc"]) ** rovcp
    cols["land_dtheta_zero"] = d
    return cols


def random_columns(n, seed=20261103):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        water = i % 2 == 1
        t1 = rng.uniform(250.0, 305.0)
        d = base(u1=rng.uniform(-15, 15), v1=rng.uniform(-15, 15), t1=t1,
                 qv1=rng.uniform(0.0, 0.018), p1=rng.uniform(85000, 101000),
                 dz1=rng.uniform(10.0, 120.0), tsk=t1 + rng.uniform(-12, 12),
                 xland=2.0 if water else 1.0, mavail=rng.uniform(0.05, 1.0),
                 pblh=rng.uniform(0, 3000), dx=rng.choice([0.0, 3000.0, 2.5e4, 2.0e5]),
                 ust=rng.uniform(0.01, 0.8), mol=rng.uniform(-1, 1),
                 znt=rng.uniform(1e-5, 2e-3) if water else rng.uniform(0.01, 1.5),
                 hfx=rng.uniform(-50, 400), qfx=rng.uniform(-1e-5, 3e-4),
                 qsfc=rng.uniform(0.0, 0.02))
        d["psfc"] = d["p1"] + rng.uniform(300.0, 1500.0)
        out[f"random_{i:02d}"] = d
    return out


def write_inputs(path, cols):
    with open(path, "w") as f:
        f.write(f"{len(cols)}\n")
        f.write(" ".join(repr(float(x)) for x in constants_vector()) + "\n")
        for d in cols.values():
            f.write(" ".join(repr(float(d[k])) for k in IN) + "\n")


def parse_outputs(path, ncol):
    lines = open(path).read().split("\n")
    assert lines[0].startswith("SFCLAYREV"), lines[0]
    out = np.zeros((NSTEP, ncol, len(OUT)))
    i = 1
    for s in range(NSTEP):
        assert lines[i].startswith("STEP") and "sf_sfclayrev_run OK" in lines[i], lines[i]
        i += 1
        for c in range(ncol):
            out[s, c] = np.array(lines[i].split(), float); i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = {**named_columns(), **random_columns(40)}
    ncol = len(cols)
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        write_inputs(fin, cols)
        for sff in (0, 1):
            r = subprocess.run([a.driver, str(fin), str(fout), str(NSTEP), str(sff)],
                               capture_output=True, text=True)
            print(r.stdout, r.stderr)
            assert r.returncode == 0, r
            res[sff] = parse_outputs(fout, ncol)
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.split("\n")[0]
    meta = dict(oracle=ORACLE, compiler=gfortran + " -O2 -DDOUBLE_PRECISION -fdefault-real-8"
                " -cpp -DEM_CORE=1", constants=dict(zip(
                    ("cp", "g", "rovcp", "r", "xlv", "svp1", "svp2", "svp3", "svpt0", "ep1",
                     "ep2", "karman", "p1000mb"), constants_vector())),
                names=list(cols), inputs=list(IN), outputs=list(OUT), nstep=NSTEP,
                options="isfflx=1 isftcflx=0 iz0tlnd=0 shalwater_z0=0 kim_tofd=F lakemask=0",
                layout="ora_sff{0,1}[step, column, output]; lh is unassigned by the oracle "
                       "when scm_force_flux=1 (driver initialises it to 0)")
    np.savez_compressed(a.out, meta=json.dumps(meta),
                        **{f"in_{k}": np.array([d[k] for d in cols.values()]) for k in IN},
                        ora_sff0=res[0], ora_sff1=res[1])
    print("WROTE", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
