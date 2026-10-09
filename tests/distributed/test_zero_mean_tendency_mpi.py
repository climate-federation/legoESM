"""``zero_mean_tendency`` under lat-band MPI must equal the single-process result.

The correction divides the global area integral of the tendency by the global
area.  Under MPI the numerator was allreduced while the denominator was the
rank's own band area, so the removed mean was too large by ``A_global / A_band``
(about ``n_ranks``).  Forward value AND gradient are checked against the serial
reference on the same global field.

Run: ``mpirun -np 2 python -m pytest tests/distributed/test_zero_mean_tendency_mpi.py``
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm import constants
from legoesm.core.conservation import zero_mean_tendency
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid
from legoesm.parallel.comm import build_comm_topology
from legoesm.parallel.latlon_mpi import (
    make_latlon_band_layout,
    slice_latlon_grid_to_band,
)

_NLEV = 3


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _global_grid():
    return create_latlon_grid(
        n_lat=16, radius=constants.R_earth, omega=constants.Omega,
    )


def _field(grid, nlev=None):
    lat = np.asarray(grid.lat2d, dtype=np.float64)
    lon = np.asarray(grid.lon2d, dtype=np.float64)
    f = 1.0 + np.sin(lat) ** 2 + 0.3 * np.cos(2.0 * lon) * np.cos(lat)
    if nlev is None:
        return jnp.asarray(f)
    return jnp.asarray(np.stack([f * (k + 1) for k in range(nlev)], axis=-1))


def _weights(f):
    return jnp.asarray(np.linspace(0.5, 2.0, f.size).reshape(f.shape))


@pytest.fixture()
def band():
    """Serial references first (local backend), then arm MPI and slice."""
    g = _global_grid()
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
    if n_ranks < 2:
        pytest.skip("needs >= 2 MPI ranks")
    refs = {}
    for nlev in (None, _NLEV):
        f = _field(g, nlev)
        w = _weights(f)
        refs[nlev] = (
            f, w, zero_mean_tendency(f, g),
            jax.grad(lambda x: jnp.sum(w * zero_mean_tendency(x, g)))(f),
        )
    set_halo_backend("mpi", build_comm_topology(rank, n_ranks))
    layout = make_latlon_band_layout(rank, n_ranks, g.n_lat, g.n_lon)
    s, e = layout.lat_start, layout.lat_end
    yield slice_latlon_grid_to_band(g, layout), s, e, refs, g
    set_halo_backend("local")


@pytest.mark.parametrize("nlev", [None, _NLEV])
def test_mpi_value_matches_serial(band, nlev):
    bgrid, s, e, refs, _ = band
    f, _, ref, _ = refs[nlev]
    out = zero_mean_tendency(f[s:e], bgrid)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref[s:e]),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nlev", [None, _NLEV])
def test_mpi_gradient_matches_serial(band, nlev):
    """Rank r differentiates its own band loss; the shared correction's
    cotangent must be summed over ranks to give the serial gradient."""
    bgrid, s, e, refs, _ = band
    f, w, _, gref = refs[nlev]
    wb = w[s:e]
    g = jax.grad(lambda x: jnp.sum(wb * zero_mean_tendency(x, bgrid)))(f[s:e])
    assert np.isfinite(np.asarray(g)).all()
    np.testing.assert_allclose(np.asarray(g), np.asarray(gref[s:e]),
                               rtol=1e-12, atol=1e-12)


def _cube():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    return create_cubed_sphere(8)


@pytest.mark.parametrize("grid_kind", ["latlon", "cube"])
@pytest.mark.parametrize("nlev", [None, _NLEV])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_replicated_full_grid_matches_serial(band, grid_kind, nlev, dtype):
    """Every rank holds the whole (identical) field: the reduced numerator and
    area both scale by n_ranks, so value AND gradient must stay the serial
    ones.  Serial references are computed on the local backend."""
    g = band[4] if grid_kind == "latlon" else _cube()
    shape = g.area.shape + (() if nlev is None else (nlev,))
    rng = np.random.default_rng(1)
    f = jnp.asarray((1.0 + rng.random(shape)).astype(dtype))
    w = jnp.asarray(rng.random(shape).astype(dtype))

    def run():
        val = jax.jit(lambda x: zero_mean_tendency(x, g))(f)
        grad = jax.grad(lambda x: jnp.sum(w * zero_mean_tendency(x, g)))(f)
        return np.asarray(val), np.asarray(grad)

    val, grad = run()
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    topo = get_mpi_topology()
    set_halo_backend("local")
    try:
        ref_val, ref_grad = run()
    finally:
        set_halo_backend("mpi", topo)
    assert get_halo_backend() == "mpi"
    tol = 1e-12 if dtype == np.float64 else 1e-6
    np.testing.assert_allclose(val, ref_val, rtol=tol, atol=tol)
    np.testing.assert_allclose(grad, ref_grad, rtol=tol, atol=tol)
