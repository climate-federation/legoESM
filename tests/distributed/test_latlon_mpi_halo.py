"""Multi-rank MPI halo-exchange tests for the lat-lon band decomposition.

Run with::

    mpirun -np 2 python -m pytest tests/distributed/test_latlon_mpi_halo.py -v
    mpirun -np 4 python -m pytest tests/distributed/test_latlon_mpi_halo.py -v

These tests verify that the MPI band halo produces the same padded
field as the canonical single-rank
:func:`legoesm.grids.halo_latlon.pad_halo_latlon*` helpers, restricted
to the interior longitude columns.  Match is required to bit-level
(``atol=0, rtol=0``) — pole-fold is exact integer indexing + sign flip,
no floating-point round-off.

Single-rank counterparts live in
:mod:`tests.parallel.test_latlon_mpi_halo_serial`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_halo_latlon_3d,
    pad_halo_latlon_vector,
    pad_halo_latlon_vector_3d,
)
from legoesm.parallel.latlon_mpi import (
    exchange_halo_latlon,
    make_latlon_band_layout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _global_test_field_2d(n_lat: int, n_lon: int, seed: int = 0) -> jnp.ndarray:
    """A spatially non-trivial 2D global field, identical on every rank.

    Identical seeding is critical: each rank constructs the same global
    field, then slices its own band.  Gather + reconcile recovers the
    full padded field for comparison against the serial reference.
    """
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((n_lat, n_lon), dtype=np.float64))


def _global_test_field_3d(n_lat: int, n_lon: int, nlev: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((n_lat, n_lon, nlev), dtype=np.float64)
    )


def _layout_for_this_rank(n_lat: int, n_lon: int):
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
    return make_latlon_band_layout(rank, n_ranks, n_lat, n_lon)


def _gather_lat(local_array: jnp.ndarray) -> jnp.ndarray | None:
    """Gather rank-local lat band onto rank 0.

    Each rank contributes its local block; the result on rank 0 is the
    concatenation along axis 0 in rank order.  Other ranks receive
    ``None``.
    """
    comm = MPI.COMM_WORLD
    gathered = comm.gather(np.asarray(local_array), root=0)
    if comm.Get_rank() == 0:
        return jnp.asarray(np.concatenate(gathered, axis=0))
    return None


def _reconcile_padded_bands(local_padded, layout, halo):
    """Stitch together the padded local bands into the global padded field.

    Each local band is shape ``(n_lat_local + 2*halo, n_lon, ...)``.
    Inter-rank halos overlap between neighbours, but interior data is
    unique to one rank, so we strip the *interior-facing* halo from
    every rank and keep the pole-side halo only on the boundary ranks.

    Returns
    -------
    On rank 0: the reconciled global padded field of shape
    ``(n_lat_global + 2*halo, n_lon, ...)``.  Other ranks: ``None``.
    """
    is_south_pole = layout.south_rank is None
    is_north_pole = layout.north_rank is None
    s_strip = 0 if is_south_pole else halo
    e_strip = None if is_north_pole else -halo
    keep = local_padded[s_strip:e_strip]
    return _gather_lat(keep)


# ---------------------------------------------------------------------------
# Bit-exact match: MPI exchange ≡ serial pad_halo (interior lon columns)
# ---------------------------------------------------------------------------


class TestMPIHaloMatchesSerial:

    @pytest.mark.parametrize("halo", [1, 2])
    def test_scalar_2d(self, halo):
        n_lat, n_lon = 16, 24
        global_field = _global_test_field_2d(n_lat, n_lon, seed=42)
        layout = _layout_for_this_rank(n_lat, n_lon)
        local = global_field[layout.lat_start:layout.lat_end]
        local_padded = exchange_halo_latlon(
            local, layout, halo=halo, is_vector_v=False,
        )
        # Local-band correctness: shape must be (n_lat_local + 2*halo, n_lon)
        assert local_padded.shape == (layout.n_lat_local + 2 * halo, n_lon)
        # Reconcile to global, compare against serial reference (interior
        # lon columns only — MPI never lon-pads).
        global_padded = _reconcile_padded_bands(local_padded, layout, halo)
        if MPI.COMM_WORLD.Get_rank() == 0:
            ref = pad_halo_latlon(global_field, halo=halo)[:, halo:-halo]
            np.testing.assert_allclose(global_padded, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("halo", [1, 2])
    def test_scalar_3d(self, halo):
        n_lat, n_lon, nlev = 16, 24, 4
        global_field = _global_test_field_3d(n_lat, n_lon, nlev, seed=7)
        layout = _layout_for_this_rank(n_lat, n_lon)
        local = global_field[layout.lat_start:layout.lat_end]
        local_padded = exchange_halo_latlon(
            local, layout, halo=halo, is_vector_v=False,
        )
        assert local_padded.shape == (
            layout.n_lat_local + 2 * halo, n_lon, nlev,
        )
        global_padded = _reconcile_padded_bands(local_padded, layout, halo)
        if MPI.COMM_WORLD.Get_rank() == 0:
            ref = pad_halo_latlon_3d(global_field, halo=halo)[:, halo:-halo, :]
            np.testing.assert_allclose(global_padded, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("halo", [1, 2])
    def test_vector_v_2d(self, halo):
        n_lat, n_lon = 16, 24
        global_field = _global_test_field_2d(n_lat, n_lon, seed=3)
        layout = _layout_for_this_rank(n_lat, n_lon)
        local = global_field[layout.lat_start:layout.lat_end]
        local_padded = exchange_halo_latlon(
            local, layout, halo=halo, is_vector_v=True,
        )
        global_padded = _reconcile_padded_bands(local_padded, layout, halo)
        if MPI.COMM_WORLD.Get_rank() == 0:
            ref = pad_halo_latlon_vector(global_field, halo=halo)[:, halo:-halo]
            np.testing.assert_allclose(global_padded, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("halo", [1, 2])
    def test_vector_v_3d(self, halo):
        n_lat, n_lon, nlev = 16, 24, 4
        global_field = _global_test_field_3d(n_lat, n_lon, nlev, seed=11)
        layout = _layout_for_this_rank(n_lat, n_lon)
        local = global_field[layout.lat_start:layout.lat_end]
        local_padded = exchange_halo_latlon(
            local, layout, halo=halo, is_vector_v=True,
        )
        global_padded = _reconcile_padded_bands(local_padded, layout, halo)
        if MPI.COMM_WORLD.Get_rank() == 0:
            ref = pad_halo_latlon_vector_3d(
                global_field, halo=halo,
            )[:, halo:-halo, :]
            np.testing.assert_allclose(global_padded, ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Inter-rank consistency: neighbouring ranks see *each other's* boundary
# rows in their halo.
# ---------------------------------------------------------------------------


class TestInterRankConsistency:

    def test_south_halo_is_neighbour_top_row(self):
        """Interior rank's south[halo-1] row must equal the southern
        neighbour's interior[-1] row (the one nearest the partition cut)."""
        n_lat, n_lon = 16, 24
        global_field = _global_test_field_2d(n_lat, n_lon, seed=5)
        layout = _layout_for_this_rank(n_lat, n_lon)
        local = global_field[layout.lat_start:layout.lat_end]
        halo = 1
        local_padded = exchange_halo_latlon(
            local, layout, halo=halo, is_vector_v=False,
        )
        if layout.south_rank is not None:
            # south halo row at index (halo-1) should be the neighbour's
            # interior[-1] (which globally is row layout.lat_start - 1).
            expected = global_field[layout.lat_start - 1]
            np.testing.assert_allclose(
                local_padded[halo - 1], expected, rtol=0, atol=0,
            )
        if layout.north_rank is not None:
            # north halo row at index (-(halo)) should be neighbour's
            # interior[0] (globally row layout.lat_end).
            expected = global_field[layout.lat_end]
            np.testing.assert_allclose(
                local_padded[-halo], expected, rtol=0, atol=0,
            )

    def test_halo_width_2_inter_rank(self):
        """With halo=2 between interior ranks, both halo rows must
        match the neighbour's two boundary-adjacent interior rows in
        the right order."""
        n_lat, n_lon = 32, 24
        global_field = _global_test_field_2d(n_lat, n_lon, seed=2)
        layout = _layout_for_this_rank(n_lat, n_lon)
        local = global_field[layout.lat_start:layout.lat_end]
        halo = 2
        local_padded = exchange_halo_latlon(
            local, layout, halo=halo, is_vector_v=False,
        )
        if layout.south_rank is not None:
            # south_halo (rows 0..halo) must equal neighbour's
            # interior[-halo:] in the same row order.
            expected = global_field[
                layout.lat_start - halo:layout.lat_start
            ]
            np.testing.assert_allclose(
                local_padded[:halo], expected, rtol=0, atol=0,
            )
        if layout.north_rank is not None:
            expected = global_field[
                layout.lat_end:layout.lat_end + halo
            ]
            np.testing.assert_allclose(
                local_padded[-halo:], expected, rtol=0, atol=0,
            )


# ---------------------------------------------------------------------------
# AD safety: jax.grad through MPI sendrecv returns a finite gradient
# ---------------------------------------------------------------------------


class TestADUnderMPI:

    def test_grad_through_mpi_halo_is_finite(self):
        """Reverse-mode AD through the AD-safe ``_sendrecv_vjp`` halo
        must produce finite gradients on every rank.  We just check
        finiteness + shape — exact value vs. serial reference is hard
        to compute under MPI without re-implementing the backward in
        the test (the AD-correctness of the wrapper itself is covered
        in tests/distributed/test_mpi_differentiability.py)."""
        n_lat, n_lon = 16, 24
        global_field = _global_test_field_2d(n_lat, n_lon, seed=13)
        layout = _layout_for_this_rank(n_lat, n_lon)
        local = global_field[layout.lat_start:layout.lat_end]

        def loss(field):
            padded = exchange_halo_latlon(
                field, layout, halo=2, is_vector_v=False,
            )
            return jnp.sum(padded ** 2)

        g = jax.grad(loss)(local)
        assert g.shape == local.shape
        assert bool(jnp.all(jnp.isfinite(g)))
