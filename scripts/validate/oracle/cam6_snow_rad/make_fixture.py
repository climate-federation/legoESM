"""Fixture for the CAM6 snow-for-radiation oracle (tests/atmosphere/oracle_fixtures/cam6_snow_rad.npz).

usage: python make_fixture.py <oracle build dir> <out.npz>

240 points that reach every branch of micro_mg_cam.F90 L2309-2321 and
L2576-2583: cloud fraction on both sides of 1e-4, deep convective fraction on
both sides of 1e-4, liquid on both sides of 1e-10, snow on both sides of
1e-6 and 1e-7 kg/kg, and realistic T, p, dp.  Snow number is positive (CAM's
MG2 keeps it so); the zero-number guard is tested separately.
"""
import os
import subprocess
import sys

import numpy as np

from legoesm import constants


def main(build_dir, out):
    r = np.random.default_rng(5)
    n = 240
    pick = lambda vals: r.choice(vals, n)
    cld = pick([0.0, 5e-5, 1e-4, 2e-4, 0.05, 0.4, 1.0]) * r.uniform(0.5, 1.0, n)
    cld[r.random(n) < 0.15] = 1e-4
    deepcu = pick([0.0, 5e-5, 1e-4, 0.02, 0.3, 0.6])
    qliq = pick([0.0, 5e-11, 1e-10, 2e-10, 1e-5, 3e-4])
    qs = pick([0.0, 5e-8, 1e-7, 5e-7, 1e-6, 2e-6, 1e-4, 1e-3])
    ns = 10 ** r.uniform(2.0, 6.0, n)
    T = r.uniform(200.0, 290.0, n)
    p = r.uniform(1.0e4, 9.5e4, n)
    rho = p / (constants.R_d * T)
    pdel = r.uniform(500.0, 4000.0, n)
    with open(os.path.join(build_dir, "snow_in.bin"), "wb") as fh:
        np.array([n], dtype="<i4").tofile(fh)
        for a in (cld, deepcu, qliq, qs, ns, rho, pdel):
            np.ascontiguousarray(a, dtype="<f8").tofile(fh)
    subprocess.run(["./snow_oracle", "snow_in.bin", "snow_out.bin"], cwd=build_dir, check=True)
    o = np.fromfile(os.path.join(build_dir, "snow_out.bin"), dtype="<f8").reshape(3, n)
    np.savez(out, cld=cld, deepcu=deepcu, qliq=qliq, qs=qs, ns=ns, T=T, p=p, pdel=pdel,
             cldfsnow=o[0], icswp=o[1], des=o[2], gravit=constants.g)
    print(f"wrote {out}: {n} points; cldfsnow 0.25 at {np.sum(o[0] == 0.25)}, "
          f"zeroed at {np.sum((o[0] == 0) & (cld > 1e-4))}, des>0 at {np.sum(o[2] > 0)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
