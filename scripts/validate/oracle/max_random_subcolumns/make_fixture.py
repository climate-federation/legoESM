"""Generate tests/atmosphere/oracle_fixtures/max_random_subcolumns.npz.

Oracle: CAM cam_cesm2_1_rel_60 (a03b84b7) ``mcica_subcol_sw`` ->
``generate_stochastic_clouds_sw`` case(2) (maximum-random), compiled with ONE
inserted line that replaces CAM's KISS uniforms by harness-supplied ones just
before the recursion (build.sh), so CAM's own recursion, ``CDF >= 1 - cldf``
cloudy test and in-cloud path assignment run on the same uniforms the test
hands legoESM.

Columns: ecRad 1.7.2 ``test/ifs/ecrad_meridian.nc`` (32 IFS columns, 137
levels; the tier-2 audit inputs), plus one synthetic tie column whose cloud
fractions put ``1 - cf`` exactly on the stratified uniforms ``(i+0.5)/8``.
Cases (8 subcolumns, the CloudConfig default):
  * ``rnd``  : meridian columns, NumPy uniforms (seed below), different per column;
  * ``table``: meridian + tie column, legoESM's stratified table for 8 subcolumns
               (what ``generate_subcolumns(cf, 8)`` uses with no shift) -- dyadic
               uniforms, so it exercises the exact ties.
The oracle needs ngptsw = 112 subcolumns; the 8 uniforms are tiled over them
and only the first 8 are stored.

usage (compute node, JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu, PYTHONPATH incl. packages):
    bash build.sh <CAM@cam_cesm2_1_rel_60> <cime@cime5.6.47> <build dir>
    python make_fixture.py <build dir>/oracle_driver <CAM> <ecrad checkout> <out.npz>
"""
import json
import os
import subprocess
import sys
import tempfile

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.clouds import subcolumns  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
N_SUB = 8
NGPTSW = 112
SEED = 20261025


def run_oracle(oracle, cf, lic, iic, play, u):
    ncol, nlay = cf.shape
    cdf = np.stack([u[s % N_SUB] for s in range(NGPTSW)])        # (112, ncol, nlay)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "dims.txt"), "w") as f:
            f.write(f"{ncol} {nlay}\n")
        with open(os.path.join(d, "in.bin"), "wb") as f:
            for a in (play, cf, lic, iic, cdf):
                np.asarray(a, "<f8").ravel(order="F").tofile(f)
        run = subprocess.run([oracle, d], capture_output=True, text=True, check=True)
        assert "ORACLE_DONE" in run.stdout, run.stdout
        raw = np.fromfile(os.path.join(d, "out.bin"), dtype="<f8")
    n = NGPTSW * ncol * nlay
    assert raw.size == 3 * n
    cldf, clwp, ciwp = (raw[i * n:(i + 1) * n].reshape(NGPTSW, ncol, nlay, order="F")
                        for i in range(3))
    assert set(np.unique(cldf)) <= {0.0, 1.0}
    return cldf[:N_SUB] > 0.5, clwp[:N_SUB], ciwp[:N_SUB]


def main():
    ORACLE, CAM, ECRAD, OUT = sys.argv[1:5]
    nc = os.path.join(ECRAD, "test/ifs/ecrad_meridian.nc")
    with netCDF4.Dataset(nc) as ds:
        ph = np.asarray(ds["pressure_hl"][:], np.float64)
        cf = np.asarray(ds["cloud_fraction"][:], np.float64)
        ql = np.asarray(ds["q_liquid"][:], np.float64)
        qi = np.asarray(ds["q_ice"][:], np.float64)
    ncol, nlay = cf.shape
    dp = ph[:, 1:] - ph[:, :-1]
    play = 0.5 * (ph[:, 1:] + ph[:, :-1]) / 100.0                 # hPa, top first
    lwp = ql * dp / constants.g
    iwp = qi * dp / constants.g

    tie = np.zeros((1, nlay))
    vals = 1.0 - (np.arange(N_SUB) + 0.5) / N_SUB
    tie[0, 10:10 + 3 * N_SUB:3] = vals        # isolated layers
    tie[0, 60:60 + N_SUB] = vals              # one contiguous block
    tie_lwp = tie * 1.0e-2

    cases = {
        "rnd": (cf, lwp, iwp, play,
                np.random.default_rng(SEED).random((N_SUB, ncol, nlay))),
        "table": (np.concatenate([cf, tie]), np.concatenate([lwp, tie_lwp]),
                  np.concatenate([iwp, tie_lwp]), np.concatenate([play, play[:1]]),
                  None),
    }
    out = {}
    for name, (c, lw, iw, pl, u) in cases.items():
        nc_ = c.shape[0]
        if u is None:
            tab = subcolumns._stratified_table(N_SUB, nlay)          # (8, nlay)
            u = np.broadcast_to(tab[:, None, :], (N_SUB, nc_, nlay))
        lic, iic = (np.asarray(a) for a in subcolumns.in_cloud_paths(
            jnp.asarray(c), jnp.asarray(lw), jnp.asarray(iw)))
        mask, clwp, ciwp = run_oracle(ORACLE, c, lic, iic, pl, u)
        out.update({f"{name}/cf": c, f"{name}/lwp": lw, f"{name}/iwp": iw,
                    f"{name}/u": np.ascontiguousarray(u if name == "rnd" else u[:, :1]),
                    f"{name}/mask_oracle": mask, f"{name}/clwp_oracle": clwp,
                    f"{name}/ciwp_oracle": ciwp})
        print(name, "cells", mask.size, "cloudy", int(mask.sum()))

    sha = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],  # noqa: E731
                                   capture_output=True, text=True).stdout.strip()
    meta = dict(
        oracle=f"CAM cam_cesm2_1_rel_60 {sha(CAM)} rrtmg/aer_src/mcica_subcol_gen_sw.f90 case(2)",
        oracle_note="one inserted line injects the uniforms before the case(2) recursion",
        inputs=f"ecRad {sha(ECRAD)} test/ifs/ecrad_meridian.nc + synthetic tie column",
        n_sub=N_SUB, seed=SEED, legoesm=sha(REPO),
        generator="scripts/validate/oracle/max_random_subcolumns/"
                  "{build.sh,stubs.F90,oracle_driver.F90,make_fixture.py}")
    np.savez_compressed(OUT, meta=json.dumps(meta), **out)
    print("FIXTURE_DONE", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
