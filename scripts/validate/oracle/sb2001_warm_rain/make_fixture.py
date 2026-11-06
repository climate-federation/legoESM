"""Generate the gSAM M2005 Seifert-Beheng (2001) warm-rain oracle fixture by
RUNNING the oracle.

Oracle: gSAM 1.8.8 ``SRC/MICRO_M2005/module_mp_graupel.f90`` (M2005MICRO_GRAUPEL,
GRAUPEL_INIT) with ``dosb_warm_rain=.true.`` (IRAIN=1), the gSAM DEFAULT
``dofix_pgam=.false.`` (droplet shape diagnosed from N_c) and
``dopredictNc=.false.`` (INUM=1, N_c = Nc0).  No git tag/SHA exists; the pin
is the tarball http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz (sha256
0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c), built by
``build.sh`` (legoESM saturation curve and constants fed in, ODIAG dumps).

Point grid: one-level columns through the kernel (dz = 1e9 m so sedimentation
does nothing, dt = 1 s), both temperature branches (263 K and 293 K), two
pressures, a q_c x q_r grid that includes the exact gate values 1e-6 and 1e-8,
and specified droplet numbers that reach both pgam bounds (2 and 10) and both
droplet-slope bounds.  Dumped: the RAW rates (before the conservation limiters)
PRC, NPRC1, PRA, and the droplet PSD the rates used (NC3D after the slope
clamp, PGAM, NU).

Usage (CPU node; needs gfortran and the gSAM source)::

    python scripts/validate/oracle/sb2001_warm_rain/make_fixture.py \\
        --gsam <dir containing gSAM1.8.8/SRC> --build <dir> \\
        --out tests/atmosphere/microphysics/oracle_fixtures/sb2001_warm_rain.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NC0_CM3 = (10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0)
TEMPS = (263.0, 293.0)
PRESS = (95000.0, 70000.0)
QC = (0.0, 1.0e-12, 5.0e-7, 1.0e-6, 2.0e-6, 1.0e-5, 1.0e-4, 5.0e-4, 2.0e-3, 5.0e-3)
QR = (0.0, 5.0e-9, 1.0e-8, 1.0e-7, 1.0e-5, 1.0e-4, 1.0e-3)
RAIN_D = 300.0e-6     # mean rain diameter for N_r (PRC/PRA do not read N_r)
DZ, DT = 1.0e9, 1.0
# sha256 of the gSAM 1.8.8 sources the build reads (SRC/MICRO_M2005/), checked
# before building so a fixture can only come from the pinned oracle.
SOURCE_SHA256 = {
    "micro_params.f90": "306923acc9cf3e80b4c8d090b2109a3553514ab865a8ac5a8734efe0d238584f",
    "microphysics.f90": "c77dce88aaefccd9ec3bf3f7635d748e1ce36cfc40f63169409d63b1e1dcca28",
    "module_mp_graupel.f90": "0b89c7582e76ebc88e636e2456f429943d8c5fe80102391537404abc8854cbb6",
    "wrf_debug.f90": "261efbc8969939de94e0f61c9e470a7886d04884b6da6975961f4b2075f45bca",
}
GENERATOR_FILES = ("build.sh", "patch_gsam.py", "gsam_driver.F90", "gsam_stubs.F90",
                   "satadj_mod.F90", "make_fixture.py")
# ODIAG columns of the driver output row (12 state values precede ODIAG(1)).
_ODIAG = dict(prc=1, nprc1=3, pra=4, nc3d=44, pgam=45, nu=46)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(gsam):
    """Verify the oracle sources against SOURCE_SHA256 and hash the generator."""
    src = Path(gsam) / "SRC" / "MICRO_M2005"
    got = {f: _sha(src / f) for f in SOURCE_SHA256}
    assert got == SOURCE_SHA256, got
    gfort = subprocess.run(["gfortran", "--version"], capture_output=True,
                           text=True).stdout.splitlines()[0]
    return dict(source_sha256=got, gfortran=gfort,
                generator_sha256={f: _sha(HERE / f) for f in GENERATOR_FILES})


def build_points():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm import constants as C
    from legoesm.thermo import saturation_specific_humidity
    rows = []
    for T in TEMPS:
        for p in PRESS:
            rho = p / (C.R_d * T)
            qs = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
            for qc in QC:
                for qr in QR:
                    nr = qr / (np.pi * C.rho_water * RAIN_D ** 3) if qr > 0 else 0.0
                    rows.append([p, rho, DZ, 0.0, T, qs, qc, qr, nr,
                                 0, 0, 0, 0, 0, 0])
    return np.array(rows, dtype=np.float64)


def run_oracle(exe, pts, nc0, work):
    fin, fout = work / "pts_in.txt", work / f"pts_out_nc{int(nc0)}.txt"
    with open(fin, "w") as f:
        f.write(f"{len(pts)} 1\n")
        for r in pts:
            f.write(" ".join(f"{v:.17e}" for v in r) + "\n")
    # m2005 <irain> <fixpgam> <do_satadj> <Nc0_cm3> <dt> <icemicro>
    log = subprocess.run([str(exe), "m2005", "1", "0", "0", f"{nc0}", f"{DT}", "1",
                          str(fin), str(fout)], check=True, capture_output=True,
                         text=True).stdout
    assert "M2005_OK" in log, log
    rows = [list(map(float, ln.split())) for ln in open(fout)]
    lev = np.array([r for r in rows if len(r) == 72])
    assert lev.shape == (len(pts), 72), lev.shape
    return {k: lev[:, 11 + j] for k, j in _ODIAG.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsam", required=True, help="dir containing SRC/MICRO_M2005")
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bld = Path(a.build)
    prov = provenance(a.gsam)
    subprocess.run(["bash", str(HERE / "build.sh"), a.gsam, str(bld)], check=True)
    exe = bld / "gsam_driver"
    pts = build_points()
    out = {"q_c": [], "q_r": [], "N_c": [], "rho": [], "T": [], "p": []}
    ora = {k: [] for k in _ODIAG}
    for nc0 in NC0_CM3:
        o = run_oracle(exe, pts, nc0, bld)
        out["q_c"].append(pts[:, 6]); out["q_r"].append(pts[:, 7])
        out["rho"].append(pts[:, 1]); out["T"].append(pts[:, 4]); out["p"].append(pts[:, 0])
        out["N_c"].append(np.full(len(pts), nc0 * 1.0e6))      # per m^3
        for k in _ODIAG:
            ora[k].append(o[k])
    meta = dict(
        oracle="gSAM 1.8.8 SRC/MICRO_M2005 (M2005MICRO_GRAUPEL)",
        tarball="http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz",
        sha256="0e9ea4a37f4daca5051cc4018cc95bbc5583159c7c27d7d05d1104fae4ed270c",
        settings="IRAIN=1 (dosb_warm_rain), dofix_pgam=.false., INUM=1 (Nc0), "
                 "doicemicro=.true., raw rates before conservation limiters",
        **prov,
        generator="python scripts/validate/oracle/sb2001_warm_rain/make_fixture.py "
                  "--gsam <gSAM1.8.8> --build <dir> --out sb2001_warm_rain.npz")
    arrs = {k: np.concatenate(v) for k, v in out.items()}
    arrs.update({f"ora_{k}": np.concatenate(v) for k, v in ora.items()})
    np.savez_compressed(a.out, meta=json.dumps(meta), **arrs)
    print("FIXTURE_OK", a.out, len(arrs["q_c"]))


if __name__ == "__main__":
    main()
