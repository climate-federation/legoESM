"""Generate tests/atmosphere/oracle_fixtures/mc3d_kernels.npz.

Oracle: microhh rte-rrtmgp-cpp v0.3 (fc291af01820674ff6dcd05889d7eb191b3daca4) device
functions ``henyey``, ``rayleigh``, ``sample_tau``, ``mie_sample_angle`` and the
``create_knull_grid`` kernel, compiled for CPU in double precision (build.sh).

The uniform deviates are NumPy draws (plus u = 0 and u = 1e-12); the test hands the
same numbers to legoESM's kernels in place of their ``jax.random.uniform`` draw.  The
Mie table slice fed to the
oracle (v0.3 ships no Mie table) is legoESM's committed LUT, band 9, which is microhh
head's data/mie_lut_broadband.nc (provenance test in tests/unit/test_mc3d_mie.py).
Inputs:
  henyey : |g| in [1e-3, 0.999999] (legoESM's isotropic branch below 1e-3 is not the
           oracle's formula and is excluded), a uniform g sweep plus edge values, plus
           for 64 g values the u where mu ~ 0 (cancellation-sensitive);
  rayleigh : the N uniforms plus u = 0 and u = 1e-12;
  free path: max(Float_epsilon, sample_tau(u)/k) (the kernel's step) for the same
           uniforms with k lognormal around 1 (sample_tau's epsilon and the step
           floor only show near u = 0);
  mie    : r_eff uniform in [1.5, 21.5] um (below 2.5 um exercises the fmod branch);
  knull  : a 16x16x12 lognormal extinction field with one cloud, 4x4x3 coarse blocks.

usage (compute node, JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu, PYTHONPATH incl. packages):
    bash build.sh <rte-rrtmgp-cpp@v0.3> <build dir>
    python make_fixture.py <build dir>/oracle_driver <rte-rrtmgp-cpp@v0.3> <out.npz>
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

from legoesm.atmosphere.physics.radiation.mc3d import mie  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
N = 4000
BAND = 9


def wbin(path, ints, arrs):
    with open(path, "wb") as f:
        np.asarray(ints, "<i4").tofile(f)
        for a in arrs:
            np.asarray(a, "<f8").ravel().tofile(f)


def main():
    ORACLE, RTE, OUT = sys.argv[1:4]
    rng = np.random.default_rng(3)
    u = rng.random(N)
    g = rng.uniform(-0.95, 0.99, N)
    g[:8] = [1e-3, -1e-3, 2e-3, -2e-3, 0.999999, 0.9999, -0.9999, 0.5]
    g = np.where(np.abs(g) < 1e-3, 1e-3, g)
    g0 = np.concatenate([np.linspace(-0.95, -0.01, 32), np.linspace(0.01, 0.99, 32)])
    u0 = ((1.0 - g0 * g0) / np.sqrt(1.0 + g0 * g0) - 1.0 + g0) / (2.0 * g0)   # mu(u0) = 0
    g = np.concatenate([g, g0])
    u_hg = np.concatenate([u, u0])
    u_edge = np.concatenate([u, [0.0, 1e-12]])
    k_fp = rng.lognormal(0.0, 2.0, u_edge.size)
    k_fp[-2:] = 1.0
    lut = mie.load_mie_sampling_lut(dtype=jnp.float64)
    cdf = np.asarray(lut.phase_cdf[BAND], np.float64)
    ang = np.asarray(lut.phase_cdf_angle[BAND], np.float64)
    re = rng.uniform(1.5, 21.5, N)
    re[:4] = [2.5, 2.0, 1.75, 21.5]
    nx, ny, nz, kx, ky, kz = 16, 16, 12, 4, 4, 3
    kext = rng.lognormal(-6.0, 2.0, (nz, ny, nx))       # oracle layout: x fastest
    kext[:, 5:9, 5:9] += 0.05
    with tempfile.TemporaryDirectory() as d:
        wbin(f"{d}/henyey_in.bin", [g.size], [g, u_hg])
        wbin(f"{d}/rayl_in.bin", [u_edge.size], [u_edge])
        wbin(f"{d}/tau_in.bin", [u_edge.size], [u_edge, k_fp])
        wbin(f"{d}/mie_in.bin", [cdf.size, ang.shape[0], N], [cdf, ang, u, re])
        wbin(f"{d}/knull_in.bin", [nx, ny, nz, kx, ky, kz], [kext])
        run = subprocess.run([ORACLE, d], capture_output=True, text=True, check=True)
        assert "ORACLE_DONE" in run.stdout, run.stdout
        rb = lambda name, n: np.fromfile(f"{d}/{name}", dtype="<f8", count=n)  # noqa: E731
        out = dict(henyey=rb("henyey_out.bin", g.size), rayleigh=rb("rayl_out.bin", u_edge.size),
                   tau=rb("tau_out.bin", u_edge.size), mie_angle=rb("mie_out.bin", N),
                   knull=rb("knull_out.bin", kx * ky * kz))
    for k, v in out.items():
        assert np.all(np.isfinite(v)), k
    sha = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],  # noqa: E731
                                   capture_output=True, text=True).stdout.strip()
    meta = dict(oracle=f"microhh rte-rrtmgp-cpp v0.3 {sha(RTE)} device functions, double precision",
                n=N, seed=3, mie_band=BAND,
                knull_dims=[nx, ny, nz, kx, ky, kz], knull_kmin=1e-3, legoesm=sha(REPO),
                generator="scripts/validate/oracle/mc3d_kernels/{build.sh,oracle_driver.cpp,make_fixture.py}")
    np.savez_compressed(OUT, meta=json.dumps(meta), u=u, u_hg=u_hg, u_edge=u_edge, k_fp=k_fp,
                        g=g, re=re, kext=kext,
                        **{f"{k}_oracle": v for k, v in out.items()})
    print("FIXTURE_DONE", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
