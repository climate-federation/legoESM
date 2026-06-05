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
from legoesm.grids.halo import set_halo_backend
from legoesm.parallel.latlon_mpi import (
    _fold_tripolar_north,
    _tripolar_fold_perm_sign,
    exchange_halo_latlon,
    make_latlon_band_layout,
    slice_cgrid_geometry_to_band,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    is_tripolar,
    pad_ns_scalar,
    _fold_is_local,
    gradient_y_cgrid,
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


class TestIssue356Bug1IsTripolarConsistent:
    """Bug 1: is_tripolar() must return the same value on all ranks.

    Before the fix, slice_cgrid_geometry_to_band set fold.is_active=False
    on non-northernmost ranks, causing is_tripolar() to disagree →
    different MPI call counts → MPI_ERR_TRUNCATE.
    """

    def test_is_tripolar_consistent_across_ranks(self, fold):
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        local_tp = int(is_tripolar(band))
        all_tp = MPI.COMM_WORLD.allgather(local_tp)
        assert len(set(all_tp)) == 1, (
            f"is_tripolar() disagrees across ranks: {all_tp}")

    def test_fold_is_local_only_on_northernmost(self, fold):
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        local_fil = int(_fold_is_local(band))
        all_fil = MPI.COMM_WORLD.allgather(local_fil)
        n_north = sum(all_fil)
        assert n_north == 1, (
            f"Exactly one rank should have fold_is_local, got {n_north}: {all_fil}")
        assert all_fil[rank] == int(layout.north_rank is None)


class TestIssue356Bug2InterpToVPoints:
    """Bug 2: _interp_to_v_points must match serial at partition boundaries.

    Before the fix, "average-then-pad" used only rank-local data, missing
    the neighbor's first row at partition cuts.
    """

    def test_interp_matches_serial_at_partition_boundaries(self, fold):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _interp_to_v_points,
        )
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rng = np.random.default_rng(42)
        f_global = jnp.asarray(
            rng.standard_normal((N_LAT, N_LON), dtype=np.float64))
        f_v_serial = _interp_to_v_points(f_global, grid=geom)

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end

        try:
            set_halo_backend("mpi", layout)
            f_v_local = _interp_to_v_points(f_global[s:e], grid=band)
        finally:
            set_halo_backend("local")

        max_err = float(jnp.max(jnp.abs(f_v_local - f_v_serial[s:e + 1])))
        all_errs = MPI.COMM_WORLD.allgather(max_err)
        assert max(all_errs) < 1e-10, (
            f"_interp_to_v_points max error across ranks: {all_errs}")

    def test_pad_ns_scalar_consistent_mpi_calls(self, fold):
        """pad_ns_scalar must not cause MPI call-count mismatch (Bug 1)."""
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rng = np.random.default_rng(99)
        f_global = jnp.asarray(
            rng.standard_normal((N_LAT, N_LON), dtype=np.float64))

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end

        f_interior = 0.5 * (f_global[s:e][:-1] + f_global[s:e][1:])
        try:
            set_halo_backend("mpi", layout)
            result = pad_ns_scalar(f_interior, band)
        finally:
            set_halo_backend("local")
        assert jnp.all(jnp.isfinite(result)), (
            f"pad_ns_scalar produced non-finite values on rank {rank}")


class TestIssue356GradientYPartitionCut:
    """gradient_y_cgrid must match serial at partition boundaries.

    Before the fix, the tripolar code path used rank-local data only
    and zeroed the south v-face unconditionally — both wrong at
    partition cuts under MPI.
    """

    def test_gradient_y_matches_serial_at_partition_boundaries(self, fold):
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rng = np.random.default_rng(77)
        f_global = jnp.asarray(
            rng.standard_normal((N_LAT, N_LON), dtype=np.float64))
        df_serial = gradient_y_cgrid(f_global, geom)

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end

        try:
            set_halo_backend("mpi", layout)
            df_local = gradient_y_cgrid(f_global[s:e], band)
        finally:
            set_halo_backend("local")

        max_err = float(jnp.max(jnp.abs(df_local - df_serial[s:e + 1])))
        all_errs = MPI.COMM_WORLD.allgather(max_err)
        assert max(all_errs) < 1e-10, (
            f"gradient_y_cgrid max error across ranks: {all_errs}")

    def test_gradient_y_3d_matches_serial(self, fold):
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rng = np.random.default_rng(88)
        f_global = jnp.asarray(
            rng.standard_normal((N_LAT, N_LON, 3), dtype=np.float64))
        df_serial = gradient_y_cgrid(f_global, geom)

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end

        try:
            set_halo_backend("mpi", layout)
            df_local = gradient_y_cgrid(f_global[s:e], band)
        finally:
            set_halo_backend("local")

        max_err = float(jnp.max(jnp.abs(df_local - df_serial[s:e + 1])))
        all_errs = MPI.COMM_WORLD.allgather(max_err)
        assert max(all_errs) < 1e-10, (
            f"gradient_y_cgrid 3D max error across ranks: {all_errs}")


class TestPgfYMeridionalFoldMPI:
    """Meridional v-face PGF operators must match serial at partition cuts.

    Follow-up to #357 review: the y-face PGF operators originally built the
    padded v-face array from already-computed interior faces, which cannot
    reconstruct the cross-partition v-face (it needs the neighbour rank's
    adjacent CELL column).  After the cell-pad-first fix, the rank-local
    result must equal the serial reference slice ``[s:e+1]`` on every rank.
    """

    NLEV = 4

    def _cell_fields(self, seed):
        rng = np.random.default_rng(seed)
        # Positive partial-cell thicknesses and a density field.
        h = jnp.asarray(
            0.1 + np.abs(rng.standard_normal(
                (N_LAT, N_LON, self.NLEV), dtype=np.float64)))
        rho = jnp.asarray(
            1025.0 + rng.standard_normal(
                (N_LAT, N_LON, self.NLEV), dtype=np.float64))
        is_active = jnp.ones((N_LAT, N_LON, self.NLEV), dtype=bool)
        return h, rho, is_active

    def test_density_jacobian_pgf_y_matches_serial(self, fold):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            density_jacobian_pgf_smc03_y,
        )
        from legoesm import constants
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        h, rho, is_active = self._cell_fields(seed=7)
        ref = density_jacobian_pgf_smc03_y(rho, h, is_active, geom, constants.g)

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end
        try:
            set_halo_backend("mpi", layout)
            local = density_jacobian_pgf_smc03_y(
                rho[s:e], h[s:e], is_active[s:e], band, constants.g)
        finally:
            set_halo_backend("local")

        max_err = float(jnp.max(jnp.abs(local - ref[s:e + 1])))
        all_errs = MPI.COMM_WORLD.allgather(max_err)
        assert max(all_errs) < 1e-9, (
            f"density_jacobian_pgf_smc03_y max error across ranks: {all_errs}")

    def test_partial_cell_pgf_correction_y_matches_serial(self, fold):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            partial_cell_pgf_correction_y,
        )
        from legoesm import constants
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rng = np.random.default_rng(11)
        centroid = jnp.asarray(
            np.abs(rng.standard_normal((N_LAT, N_LON, self.NLEV))))
        rho_prime = jnp.asarray(
            rng.standard_normal((N_LAT, N_LON, self.NLEV)))
        ref = partial_cell_pgf_correction_y(
            centroid, rho_prime, geom, constants.g)

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end
        try:
            set_halo_backend("mpi", layout)
            local = partial_cell_pgf_correction_y(
                centroid[s:e], rho_prime[s:e], band, constants.g)
        finally:
            set_halo_backend("local")

        max_err = float(jnp.max(jnp.abs(local - ref[s:e + 1])))
        all_errs = MPI.COMM_WORLD.allgather(max_err)
        assert max(all_errs) < 1e-9, (
            f"partial_cell_pgf_correction_y max error across ranks: {all_errs}")


class TestCurlVertexFoldMPI:
    """curl_vertex_cgrid must match serial at partition cuts.

    Follow-up to #357 review: with the fold active on every rank,
    is_tripolar() routes all ranks through the tripolar curl path, which
    halo-exchanged ``u`` but plain-jnp.pad-zeroed the ``dx_T`` metric at
    partition cuts.  After switching dx_T to backend-aware lat padding, the
    rank-local vorticity must equal the serial reference slice ``[s:e+1]``.
    """

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_curl_matches_serial(self, fold, ndim):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            curl_vertex_cgrid,
        )
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rng = np.random.default_rng(23)
        nlev = 3
        u_shape = (N_LAT, N_LON + 1) + ((nlev,) if ndim == 3 else ())
        v_shape = (N_LAT + 1, N_LON) + ((nlev,) if ndim == 3 else ())
        u = jnp.asarray(rng.standard_normal(u_shape, dtype=np.float64))
        v = jnp.asarray(rng.standard_normal(v_shape, dtype=np.float64))
        zeta_serial = curl_vertex_cgrid(u, v, geom)

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end
        try:
            set_halo_backend("mpi", layout)
            zeta_local = curl_vertex_cgrid(u[s:e], v[s:e + 1], band)
        finally:
            set_halo_backend("local")

        max_err = float(jnp.max(jnp.abs(zeta_local - zeta_serial[s:e + 1])))
        all_errs = MPI.COMM_WORLD.allgather(max_err)
        assert max(all_errs) < 1e-9, (
            f"curl_vertex_cgrid max error across ranks: {all_errs}")


class TestMinCellToVfaceFoldMPI:
    """min_cell_to_vface must match serial at partition cuts.

    Follow-up to #357 review: the v-face min-rule originally hardcoded a
    zero south row and used rank-local cells only, so the partition-cut
    face thickness was wrong (and the south cut was forced to a wall).
    After the cell-pad-first fix the rank-local result must equal the
    serial reference slice ``[s:e+1]``.
    """

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_min_cell_to_vface_matches_serial(self, fold, ndim):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            min_cell_to_vface,
        )
        geom = create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)
        rng = np.random.default_rng(31)
        shape = (N_LAT, N_LON) + ((3,) if ndim == 3 else ())
        # Positive thickness field (min-rule operand).
        f = jnp.asarray(0.1 + np.abs(rng.standard_normal(shape)))
        ref = min_cell_to_vface(f, geom)

        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        layout = make_latlon_band_layout(
            rank, n_ranks, N_LAT, N_LON, fold=geom.fold)
        band = slice_cgrid_geometry_to_band(geom, layout)
        s, e = layout.lat_start, layout.lat_end
        try:
            set_halo_backend("mpi", layout)
            local = min_cell_to_vface(f[s:e], band)
        finally:
            set_halo_backend("local")

        max_err = float(jnp.max(jnp.abs(local - ref[s:e + 1])))
        all_errs = MPI.COMM_WORLD.allgather(max_err)
        assert max(all_errs) < 1e-9, (
            f"min_cell_to_vface max error across ranks: {all_errs}")
