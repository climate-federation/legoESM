"""Generate tests/atmosphere/oracle_fixtures/amd.npz by RUNNING AMR-Wind's amd_muvel.

Oracle: AMR-Wind v4.2.0 (Exawind/amr-wind, 0ee09c22b9afeca06231e4910276c0ba05fb52dd)
src/turbulence/LES/AMD.H::amd_muvel, compiled verbatim (g++ -O0, amrex type stubs) by
build.sh, with the thermal gradient zero and beta = 0 (pure-shear AMD, as legoESM's
amd_nu_t; AMR-Wind's buoyancy numerator is not part of legoESM's scheme).

Inputs: the SAME point set as the Vreman fixture (jax-alfa LES oracle CBL/NBL/SBL
velocity-gradient tensors, every 17th point, plus 246 synthetic edge tensors), built by
``scripts/validate/oracle/vreman/make_fixture.py::build_points``.  C = 0.3 (legoESM's
CompressibleEulerConfig.amd_c default; AMR-Wind's own default C_poincare is 1/3).

usage (compute node):
    bash build.sh <amr-wind@v4.2.0> <build dir>
    python make_fixture.py <build dir>/amd_oracle <dir with oracle_lasd_<case>.npz> <out.npz>
"""
import importlib.util
import json
import os
import sys

import numpy as np

C_AMD = 0.3
_VREMAN_GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vreman",
                           "make_fixture.py")


def load_points_module():
    spec = importlib.util.spec_from_file_location("vreman_make_fixture", _VREMAN_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    exe, les_dir, out = sys.argv[1:4]
    pts = load_points_module()
    A, H, ids = pts.build_points(les_dir)
    nu, log = pts.run_oracle(exe, A, H, C_AMD)
    meta = dict(oracle="AMR-Wind v4.2.0 src/turbulence/LES/AMD.H amd_muvel (beta=0)",
                oracle_sha="0ee09c22b9afeca06231e4910276c0ba05fb52dd",
                les_source="jax-alfa v0.0.1 93fe46587228fb075e53dd4a981b2df33de04443 "
                           "(spin-up HEAD aa7baebf99282711416dadbde7dfdfcba8940524)",
                cases=list(pts.CASES) + ["edge"], stride=pts.STRIDE, c_amd=C_AMD,
                seed=pts.SEED, oracle_log=log)
    np.savez(out, a=A, h=H, case_id=ids, nu=nu, meta=json.dumps(meta))
    floor = 10.0 * np.finfo(np.float64).eps
    print("FIXTURE_DONE", out, A.shape[1], "floored points", int(np.sum(nu == floor)),
          f"{os.path.getsize(out)} bytes")


if __name__ == "__main__":
    main()
