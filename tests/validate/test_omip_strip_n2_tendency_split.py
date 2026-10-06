"""interface_split: on a column evolved by an exact backward-Euler diffusion step the residual is zero;
a non-diffusive source shows up in the residual only."""
import numpy as np


def _be_step(x, K, dz, dzh, dt):
    n = x.size
    A = np.zeros((n, n))
    for i in range(n - 1):
        c = dt * K[i] / dzh[i]
        A[i, i] += c / dz[i]; A[i, i + 1] -= c / dz[i]
        A[i + 1, i + 1] += c / dz[i + 1]; A[i + 1, i] -= c / dz[i + 1]
    return np.linalg.solve(np.eye(n) + A, x)


def test_residual_vanishes_for_pure_implicit_diffusion_and_catches_a_source():
    from scripts.validate.ocean_fidelity.omip_strip_n2_tendency_split import interface_split
    rng = np.random.default_rng(0)
    nz, nt, dt = 12, 6, 150.0
    dz = rng.uniform(1.0, 3.0, nz); zc = np.cumsum(dz) - dz / 2; dzh = np.diff(zc)
    K = rng.uniform(1e-4, 5e-2, (nt, nz - 1))
    X = np.empty((nt, nz)); X[0] = np.linspace(28.0, 20.0, nz)
    for n in range(1, nt):
        X[n] = _be_step(X[n - 1], K[n], dz, dzh, dt)
    tot, dif = interface_split(X[:, None], K[:, None], dz[None, None], dzh[None, None], dt, 5)
    assert np.allclose(tot, dif, rtol=1e-9, atol=1e-14)
    Xs = X.copy(); Xs[3:, 5] += 1e-3                              # a source at level 5 from step 3
    tot2, dif2 = interface_split(Xs[:, None], K[:, None], dz[None, None], dzh[None, None], dt, 5)
    assert abs((tot2 - dif2)[2, 0]) > 1e-7
