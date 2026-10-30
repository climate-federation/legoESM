"""Generate tests/atmosphere/oracle_fixtures/vreman.npz by RUNNING Vreman's reference Fortran.

Oracle: A.W. Vreman's own implementation, http://www.vremanresearch.nl/Vreman_Subgridmodel_Fortran.txt
(unversioned, sha256 8a7efd8c..., downloaded 2026-10-19), anisotropic form d_i = h_i**2 as
the author's comment prescribes; built by build.sh (verbatim check) from vreman_oracle.f.

Inputs (tier-2 audit les_sgs_closures): velocity-gradient tensors from the jax-alfa LES
oracle's OWN 32^3 fields (CBL_N91, NBL_A94, SBL_GABLS1 decks, jax-alfa v0.0.1 =
93fe4658...; files ``oracle_lasd_<case>.npz`` written by the audit's oracle_lasd.py),
interpolated to u-v-p nodes with jax-alfa's own convention (StrainRates.
StrainsUVPnodes_Dealias) and dimensionalised (gradients / z_scale, widths * z_scale).
Every 17th grid point is kept (17 is coprime with nz=32, so every level is sampled) to
keep the fixture under 1 MB, plus 246 synthetic edge tensors (seed 20261019: rest, rank 1,
rank 2, normal strains, 20 quiescent 1e-7, 20 weak 1e-5, 200 random on an anisotropic grid).
c = 0.07 (legoESM's c_vreman default).

usage (compute node):
    bash build.sh <Vreman_Subgridmodel_Fortran.txt> <build dir>
    python make_fixture.py <build dir>/vreman_oracle <dir with oracle_lasd_<case>.npz> <out.npz>
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

CASES = ("cbl", "nbl", "sbl")
STRIDE = 17
C_VREMAN = 0.07
SEED = 20261019


def uvp_tensor(G):
    """a_cd = du_c/dx_d at u-v-p nodes; w-node gradients moved to centres with the
    oracle's convention (jax-alfa StrainRates.StrainsUVPnodes_Dealias)."""
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = G
    nz = dudx.shape[2]

    def w2c(f, bottom_half):
        o = np.zeros_like(f)
        o[:, :, 1:nz - 1] = 0.5 * (f[:, :, 1:nz - 1] + f[:, :, 2:nz])
        o[:, :, 0] = 0.5 * f[:, :, 1] if bottom_half else f[:, :, 0]
        o[:, :, nz - 1] = f[:, :, nz - 1]
        return o
    return [dudx, dudy, w2c(dudz, False), dvdx, dvdy, w2c(dvdz, False),
            w2c(dwdx, True), w2c(dwdy, True), dwdz]


def edge_tensors():
    rng = np.random.default_rng(SEED)
    E, H = [], []

    def add(a, h=(100.0, 100.0, 20.0)):
        E.append(np.asarray(a, float).reshape(3, 3)); H.append(h)
    add(np.zeros((3, 3)))                                    # rest
    add([[0, 0, 1e-2], [0, 0, 0], [0, 0, 0]])                # pure du/dz (rank 1)
    uo, go = rng.normal(size=3), rng.normal(size=3)
    add(1e-2 * np.outer(uo, go))                             # general rank 1
    add(np.diag([1e-3, -1e-3, 0.0]))                         # rank-2 2-D strain
    add(np.diag([-1e-3, 0.5e-3, 0.5e-3]))                    # compressive normal
    add(np.diag([1e-3, -0.5e-3, -0.5e-3]))                   # expanding normal
    for _ in range(20):
        add(1e-7 * rng.normal(size=(3, 3)))                  # quiescent
    for _ in range(20):
        add(1e-5 * rng.normal(size=(3, 3)))                  # weak turbulence
    for _ in range(200):
        add(1e-2 * rng.normal(size=(3, 3)), (50.0, 50.0, 10.0))  # anisotropic grid
    E = np.stack(E)
    return (np.stack([E[:, c, d] for c in range(3) for d in range(3)]),
            np.asarray(H, float).T)


def build_points(les_dir):
    """Return (A (9,N), H (3,N), case_id (N,)) in the fixture order."""
    As, Hs, ids = [], [], []
    for ic, case in enumerate(CASES):
        o = np.load(os.path.join(les_dir, f"oracle_lasd_{case}.npz"))
        zs = float(o["z_scale"])
        A = np.stack([a / zs for a in uvp_tensor(list(o["grads"]))]).reshape(9, -1)
        keep = np.arange(0, A.shape[1], STRIDE)
        h = np.array([float(o["dx"]), float(o["dy"]), float(o["dz"])]) * zs
        As.append(A[:, keep]); Hs.append(np.repeat(h[:, None], keep.size, axis=1))
        ids.append(np.full(keep.size, ic, np.int8))
    Ae, He = edge_tensors()
    As.append(Ae); Hs.append(He); ids.append(np.full(Ae.shape[1], len(CASES), np.int8))
    return (np.ascontiguousarray(np.concatenate(As, 1)),
            np.ascontiguousarray(np.concatenate(Hs, 1)), np.concatenate(ids))


def run_oracle(exe, A, H, coef):
    """Stream-binary I/O of vreman_oracle.f / the AMD driver: int32 n, a11..a33, h1..h3
    (n float64 each), float64 coef -> n float64."""
    n = A.shape[1]
    with tempfile.TemporaryDirectory() as d:
        fi, fo = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(fi, "wb") as f:
            np.array([n], np.int32).tofile(f)
            for row in np.concatenate([A, H]):
                row.astype(np.float64).tofile(f)
            np.array([coef], np.float64).tofile(f)
        log = subprocess.run([exe, fi, fo], check=True, capture_output=True, text=True).stdout
        out = np.fromfile(fo, np.float64)
    assert out.size == n and np.all(np.isfinite(out)), log
    return out, log.strip()


def main():
    exe, les_dir, out = sys.argv[1:4]
    A, H, ids = build_points(les_dir)
    nu, log = run_oracle(exe, A, H, C_VREMAN)
    meta = dict(oracle="Vreman reference Fortran (vremanresearch.nl, 2005)",
                oracle_sha256="8a7efd8c64f63602746917e81c978ff71f442863cfdbcf4c2f378c0004598feb",
                les_source="jax-alfa v0.0.1 93fe46587228fb075e53dd4a981b2df33de04443 "
                           "(spin-up HEAD aa7baebf99282711416dadbde7dfdfcba8940524)",
                cases=list(CASES) + ["edge"], stride=STRIDE, c_vreman=C_VREMAN,
                seed=SEED, oracle_log=log)
    np.savez(out, a=A, h=H, case_id=ids, nu=nu, meta=json.dumps(meta))
    print("FIXTURE_DONE", out, A.shape[1], "zero-guard points", int(np.sum(nu == 0.0)),
          f"{os.path.getsize(out)} bytes")


if __name__ == "__main__":
    main()
