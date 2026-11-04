"""The Legendre derivative table must be exact at the truncation wavenumber.

div(grad f) == lap f exercises the meridional-derivative table ``Hnm`` in both
the gradient synthesis and the divergence analysis. The table previously
dropped the P_{n_max+1} term at n = n_max, giving O(1) relative error there
(measured 6.3 at T31).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids import gaussian as gs

jax.config.update("jax_enable_x64", True)

# 10x the measured worst case (3.9e-13 at T31, float64).
_TOL = 4e-12


def _rel_div_grad_error(grid, f):
    lap = np.asarray(grid.lap)
    gx, gy = gs.spectral_gradient_3d(grid, jnp.asarray(f))
    _, div = gs.vordiv_from_uv_3d(grid, gx, gy)
    ref = lap * f[:, 0]
    return np.linalg.norm(np.asarray(div)[:, 0] - ref) / np.linalg.norm(ref)


@pytest.mark.parametrize(
    "n,m", [(5, 1), (5, 2), (5, 3), (20, 7), (31, 0), (31, 5), (30, 1), (31, 31)]
)
def test_div_grad_equals_laplacian_per_mode(n, m):
    grid = gs.create_gaussian_grid(31, radius=constants.R_earth)
    i = int(np.nonzero((np.asarray(grid.ls) == n) & (np.asarray(grid.ms) == m))[0][0])
    f = np.zeros((grid.lap.size, 1), complex)
    f[i] = 1.0
    err = _rel_div_grad_error(grid, f)
    assert err < _TOL, f"(n={n}, m={m}) rel err {err:.3e}"


@pytest.mark.parametrize("n_max", [21, 31])
def test_div_grad_equals_laplacian_random_spectrum(n_max):
    grid = gs.create_gaussian_grid(n_max, radius=constants.R_earth)
    rng = np.random.default_rng(0)
    k = grid.lap.size
    f = (rng.standard_normal(k) + 1j * rng.standard_normal(k))[:, None]
    f[np.asarray(grid.ms) == 0] = f[np.asarray(grid.ms) == 0].real
    f[np.asarray(grid.ls) == 0] = 0.0
    err = _rel_div_grad_error(grid, f)
    assert err < _TOL, f"T{n_max} rel err {err:.3e}"
