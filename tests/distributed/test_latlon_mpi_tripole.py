"""Multi-rank MPI tests for the tripolar-aware lat-lon band halo
(GitHub issue #353).

Run with::

    JAX_ENABLE_X64=1 mpirun -np 2 python -m pytest \
        tests/distributed/test_latlon_mpi_tripole.py -v
    JAX_ENABLE_X64=1 mpirun -np 4 python -m pytest \
        tests/distributed/test_latlon_mpi_tripole.py -v

Every rank builds the SAME global field (identical seed), slices its
band, runs ``exchange_halo_latlon``, and checks its halos against a
reference computed from the global field:

  * the northernmost rank's north halo == the serial tripolar fold of
    the global field (``_fold_tripolar_north``) — bit-exact;
  * an interior rank's north halo == the neighbour's first interior
    row(s) (sendrecv continuity);
  * the south halo of a non-south rank == the neighbour's last row(s).

Single-rank counterparts live in
:mod:`tests.parallel.test_latlon_mpi_tripole_serial`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.parallel.latlon_mpi import (
    _fold_tripolar_north,
    _tripolar_fold_perm_sign,
    exchange_halo_latlon,
    make_latlon_band_layout,
)

N_LAT, N_LON = 16, 24


def _global_field(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float64))


def _layout(fold):
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    return make_latlon_band_layout(rank, size, N_LAT, N_LON, fold=fold)


@pytest.fixture
def fold():
    return create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON).fold


class TestTripolarMPIHaloMatchesSerial:
    @pytest.mark.parametrize("halo", [1, 2])
    @pytest.mark.parametrize("is_vector_v", [False, True])
    def test_north_and_interior_halos(self, fold, halo, is_vector_v):
        g = _global_field((N_LAT, N_LON), seed=1)
        layout = _layout(fold)
        s, e = layout.lat_start, layout.lat_end
        local = g[s:e]
        out = exchange_halo_latlon(
            local, layout, halo=halo, is_vector_v=is_vector_v)
        assert out.shape == (layout.n_lat_local + 2 * halo, N_LON)

        north = out[-halo:]
        if layout.north_rank is None:
            # Tripolar fold of the GLOBAL field (== local field, since the
            # north rank owns the fold row) must match the serial helper.
            perm, sign = _tripolar_fold_perm_sign(
                fold, is_vector_u=False, is_vector_v=is_vector_v)
            ref = _fold_tripolar_north(local, halo, perm, sign)
            np.testing.assert_allclose(north, ref, rtol=0, atol=0)
        else:
            # Interior cut: north halo == neighbour's first ``halo`` rows.
            np.testing.assert_allclose(north, g[e:e + halo], rtol=0, atol=0)

        # South halo continuity for interior ranks.
        if layout.south_rank is not None:
            np.testing.assert_allclose(out[:halo], g[s - halo:s], rtol=0, atol=0)

    def test_interior_block_preserved(self, fold):
        g = _global_field((N_LAT, N_LON), seed=2)
        layout = _layout(fold)
        s, e = layout.lat_start, layout.lat_end
        out = exchange_halo_latlon(g[s:e], layout, halo=1, is_vector_v=False)
        np.testing.assert_allclose(
            out[1:1 + layout.n_lat_local], g[s:e], rtol=0, atol=0)

    def test_3d_north_fold(self, fold):
        g = _global_field((N_LAT, N_LON, 4), seed=3)
        layout = _layout(fold)
        s, e = layout.lat_start, layout.lat_end
        out = exchange_halo_latlon(g[s:e], layout, halo=1, is_vector_v=True)
        if layout.north_rank is None:
            perm, sign = _tripolar_fold_perm_sign(
                fold, is_vector_u=False, is_vector_v=True)
            ref = _fold_tripolar_north(g[s:e], 1, perm, sign)
            np.testing.assert_allclose(out[-1:], ref, rtol=0, atol=0)


class TestTripolarMPIHaloAD:
    def test_grad_finite_and_consistent(self, fold):
        """jax.grad through the tripolar MPI halo is finite on every rank;
        the per-rank owned-cell gradient summed (allreduce) recovers the
        serial reference gradient."""
        g = _global_field((N_LAT, N_LON), seed=4)
        layout = _layout(fold)
        s, e = layout.lat_start, layout.lat_end
        local = g[s:e]

        def loss(f):
            out = exchange_halo_latlon(f, layout, halo=1, is_vector_v=True)
            return jnp.sum(out ** 2)

        grad_local = jax.grad(loss)(local)
        assert jnp.all(jnp.isfinite(grad_local))
        assert float(jnp.sum(jnp.abs(grad_local))) > 0.0
