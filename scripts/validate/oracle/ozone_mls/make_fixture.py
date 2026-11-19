"""Generate tests/atmosphere/oracle_fixtures/ozone_mls.npz.

Oracle: gSAM 1.8.8 ``SRC/RAD_RRTM/rad.f90`` ``tracesini`` (lines 1219-1431, compiled
verbatim with -fdefault-real-8; build.sh) reading ``RUNDATA/rrtmg_lw.nc``.  It returns
the layer-mean O3 on the nzm model layers plus the extra layer above the model top.

Grids (the tier-2 audit set, interface pressures handed to both sides as the same
doubles: the oracle gets ``p_half/100`` [hPa] printed to 17 digits):
  rcemip74    : gSAM CASES/RCEMIP1/snd levels; interfaces = log-p midpoints, bottom =
                the snd surface pressure, top = one half-step above the top level in log p.
  caml32      : CAM6 L32 hybrid interfaces at p_s = 1000 hPa (legoesm.grids.vertical).
  ifs137      : ecRad 1.7.2 test/ifs/ecrad_meridian.nc column 16 half levels, the 0 Pa
                top replaced by half the top full-level pressure.
  ifs137_top0 : the same column with its 0 Pa top kept (gSAM's extra layer is then
                degenerate; only the model layers are compared).

usage (compute node, JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu, PYTHONPATH incl. packages):
    bash build.sh <gSAM1.8.8> <conda env> <build dir>
    python make_fixture.py <build dir>/oracle_driver <gSAM1.8.8> <ecrad checkout> <out.npz>
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import netCDF4
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def grids(gsam, ecrad):
    """name -> p_half [Pa], index 0 = model top."""
    out = {}
    lines = open(os.path.join(gsam, "CASES/RCEMIP1/snd")).read().split("\n")
    hdr = lines[1].split()
    nl, psfc = int(hdr[1]), float(hdr[2])
    p = np.array([float(ln.split()[1]) for ln in lines[2:2 + nl]])        # hPa, surface first
    lp = np.log(p)
    lpi = np.concatenate([[np.log(psfc)], 0.5 * (lp[1:] + lp[:-1]),
                          [lp[-1] + 0.5 * (lp[-1] - lp[-2])]])
    out["rcemip74"] = (np.exp(lpi) * 100.0)[::-1]
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0
    out["caml32"] = np.asarray(CAM6_L32_HYAI) * CAM6_L32_P0 + np.asarray(CAM6_L32_HYBI) * 1.0e5
    with netCDF4.Dataset(os.path.join(ecrad, "test/ifs/ecrad_meridian.nc")) as ds:
        phl = np.asarray(ds["pressure_hl"][16, :], np.float64)              # Pa, top first
    out["ifs137_top0"] = phl.copy()
    pf = 0.5 * (phl[1:] + phl[:-1])
    phl[0] = 0.5 * pf[0]
    out["ifs137"] = phl
    return out


def run_oracle(oracle, gsam, p_half):
    presi = p_half[::-1] / 100.0                     # hPa, surface first (as the port)
    pres = 0.5 * (presi[1:] + presi[:-1])            # printed only by the oracle
    with tempfile.TemporaryDirectory() as d:
        fin, fout = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
        with open(fin, "w") as f:
            f.write(f"{len(pres)}\n")
            f.writelines(f"{x!r}\n" for x in pres.tolist())
            f.writelines(f"{x!r}\n" for x in presi.tolist())
        run = subprocess.run([oracle, os.path.join(gsam, "RUNDATA"), fin, fout],
                             capture_output=True, text=True, check=True)
        assert "ORACLE_DONE" in run.stdout, run.stdout
        o3 = np.loadtxt(fout)[:, 1]
    assert o3.size == len(pres) + 1
    return o3[:-1][::-1], o3[-1]                     # model layers top first, extra layer


def main():
    ORACLE, GSAM, ECRAD, OUT = sys.argv[1:5]
    out = {}
    for name, ph in grids(GSAM, ECRAD).items():
        assert np.all(np.diff(ph) > 0), name
        lay, above = run_oracle(ORACLE, GSAM, ph)
        out[f"{name}/p_half"] = ph
        out[f"{name}/o3_oracle"] = lay
        out[f"{name}/o3_above_oracle"] = np.asarray(above)
        print(name, len(lay), "layers; above-top", above)
    sha = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],  # noqa: E731
                                   capture_output=True, text=True).stdout.strip()
    sha256 = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()  # noqa: E731
    meta = dict(
        oracle="gSAM 1.8.8 SRC/RAD_RRTM/rad.f90 tracesini (lines 1219-1431, verbatim), r8 build",
        rad_f90_sha256=sha256(os.path.join(GSAM, "SRC/RAD_RRTM/rad.f90")),
        rrtmg_lw_nc_sha256=sha256(os.path.join(GSAM, "RUNDATA/rrtmg_lw.nc")),
        ecrad=sha(ECRAD), legoesm=sha(REPO),
        generator="scripts/validate/oracle/ozone_mls/{build.sh,oracle_driver.F90,make_fixture.py}")
    np.savez_compressed(OUT, meta=json.dumps(meta), **out)
    print("FIXTURE_DONE", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
