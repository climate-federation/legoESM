"""Single-rank correctness tests for the tripolar-aware lat-lon band MPI
halo machinery (GitHub issue #353).

These tests do NOT require mpi4jax / mpi4py — they exercise the
``n_ranks=1`` branch of the tripolar-fold path and the pure-Python
scatter / gather / geometry-slice helpers.  The multi-rank counterparts
live in :mod:`tests.distributed.test_latlon_mpi_tripole`.

The bit-exactness target is the SERIAL ocean fold convention
(:func:`legoesm.ocean.dynamics.latlon_cgrid_operators._fold_row` /
``pad_ns_scalar`` / ``pad_ns_vector_u`` / ``pad_ns_vector_v``): the MPI
northernmost-rank north halo must equal the serial permutation fold
bit-for-bit (``rtol=0, atol=0`` — pure integer indexing + sign flip).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.dynamics.latlon_cgrid_operators import _fold_row
from legoesm.parallel.latlon_mpi import (
    _fold_tripolar_north,
    _is_tripolar_layout,
    _tripolar_fold_perm_sign,
    exchange_halo_latlon,
    make_latlon_band_layout,
    scatter_state_latlon_cgrid_ocean,
    gather_state_latlon_cgrid_ocean,
    slice_cgrid_geometry_to_band,
    slice_zcoord_to_band,
)

N_LAT, N_LON = 12, 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tripole_geom():
    return create_synthetic_tripole(n_lat=N_LAT, n_lon=N_LON)


@pytest.fixture
def single_rank_tripole_layout(tripole_geom):
    return make_latlon_band_layout(
        rank=0, n_ranks=1, n_lat=N_LAT, n_lon=N_LON, fold=tripole_geom.fold,
    )


def _rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float64))


# ---------------------------------------------------------------------------
# Part 1: fold descriptor travels on the layout
# ---------------------------------------------------------------------------


class TestLayoutFold:
    def test_fold_carried_on_layout(self, single_rank_tripole_layout, tripole_geom):
        assert single_rank_tripole_layout.fold is tripole_geom.fold
        assert _is_tripolar_layout(single_rank_tripole_layout)

    def test_default_layout_has_no_fold(self):
        layout = make_latlon_band_layout(0, 1, N_LAT, N_LON)
        assert layout.fold is None
        assert not _is_tripolar_layout(layout)

    def test_perm_sign_selection(self, tripole_geom):
        fold = tripole_geom.fold
        perm, sign = _tripolar_fold_perm_sign(
            fold, is_vector_u=False, is_vector_v=False)
        np.testing.assert_array_equal(perm, fold.perm_T)
        assert sign == 1.0
        perm, sign = _tripolar_fold_perm_sign(
            fold, is_vector_u=True, is_vector_v=False)
        np.testing.assert_array_equal(perm, fold.perm_T)
        assert sign == fold.vector_sign_u
        perm, sign = _tripolar_fold_perm_sign(
            fold, is_vector_u=False, is_vector_v=True)
        np.testing.assert_array_equal(perm, fold.perm_v)
        assert sign == fold.vector_sign_v


# ---------------------------------------------------------------------------
# Part 2: _fold_tripolar_north bit-exact vs serial _fold_row (halo=1)
# ---------------------------------------------------------------------------


class TestFoldTripolarNorthMatchesSerial:
    """The keystone: the MPI fold helper must equal the serial ocean
    ``_fold_row`` north convention bit-for-bit."""

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_scalar_north(self, tripole_geom, ndim):
        fold = tripole_geom.fold
        shape = (N_LAT, N_LON) + ((4,) if ndim == 3 else ())
        field = _rand(shape, seed=1)
        out = _fold_tripolar_north(field, 1, fold.perm_T, 1.0)
        ref = _fold_row(field[-1:], fold.perm_T, 1.0, N_LON)
        np.testing.assert_allclose(out, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_u_face_north_with_wrap_column(self, tripole_geom, ndim):
        """u-face fields carry n_lon+1 columns (periodic wrap)."""
        fold = tripole_geom.fold
        shape = (N_LAT, N_LON + 1) + ((4,) if ndim == 3 else ())
        field = _rand(shape, seed=2)
        out = _fold_tripolar_north(field, 1, fold.perm_T, fold.vector_sign_u)
        ref = _fold_row(field[-1:], fold.perm_T, fold.vector_sign_u, N_LON)
        np.testing.assert_allclose(out, ref, rtol=0, atol=0)
        # The wrap column must equal column 0 of the folded core.
        np.testing.assert_allclose(out[:, -1], out[:, 0], rtol=0, atol=0)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_v_face_north(self, tripole_geom, ndim):
        fold = tripole_geom.fold
        shape = (N_LAT, N_LON) + ((4,) if ndim == 3 else ())
        field = _rand(shape, seed=3)
        out = _fold_tripolar_north(field, 1, fold.perm_v, fold.vector_sign_v)
        ref = _fold_row(field[-1:], fold.perm_v, fold.vector_sign_v, N_LON)
        np.testing.assert_allclose(out, ref, rtol=0, atol=0)

    def test_halo2_multi_row_fold(self, tripole_geom):
        """For halo=2 the two ghost rows are the fold partners of the two
        boundary rows, nearest-boundary first."""
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON), seed=4)
        out = _fold_tripolar_north(field, 2, fold.perm_T, 1.0)
        # ghost row nearest boundary (out[-2]) = fold partner of field[-1]
        np.testing.assert_allclose(out[-2], field[-1][fold.perm_T], rtol=0, atol=0)
        # ghost row farthest (out[-1]) = fold partner of field[-2]
        np.testing.assert_allclose(out[-1], field[-2][fold.perm_T], rtol=0, atol=0)

    def test_bad_column_count_raises(self, tripole_geom):
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON + 3), seed=5)
        with pytest.raises(ValueError, match="columns"):
            _fold_tripolar_north(field, 1, fold.perm_T, 1.0)


# ---------------------------------------------------------------------------
# exchange_halo_latlon: single-rank tripolar north == serial fold
# ---------------------------------------------------------------------------


class TestSingleRankExchangeTripolar:
    @pytest.mark.parametrize("halo", [1, 2])
    def test_scalar_north_halo(self, single_rank_tripole_layout, tripole_geom, halo):
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON), seed=6)
        out = exchange_halo_latlon(
            field, single_rank_tripole_layout, halo=halo, is_vector_v=False)
        assert out.shape == (N_LAT + 2 * halo, N_LON)
        north = out[-halo:]
        ref = _fold_tripolar_north(field, halo, fold.perm_T, 1.0)
        np.testing.assert_allclose(north, ref, rtol=0, atol=0)
        # interior block is preserved untouched
        np.testing.assert_allclose(out[halo:halo + N_LAT], field, rtol=0, atol=0)

    @pytest.mark.parametrize("halo", [1, 2])
    def test_v_vector_north_halo(self, single_rank_tripole_layout, tripole_geom, halo):
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON, 3), seed=7)
        out = exchange_halo_latlon(
            field, single_rank_tripole_layout, halo=halo, is_vector_v=True)
        north = out[-halo:]
        ref = _fold_tripolar_north(field, halo, fold.perm_v, fold.vector_sign_v)
        np.testing.assert_allclose(north, ref, rtol=0, atol=0)

    def test_u_vector_north_halo_with_wrap(self, single_rank_tripole_layout, tripole_geom):
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON + 1), seed=8)
        out = exchange_halo_latlon(
            field, single_rank_tripole_layout, halo=1, is_vector_u=True)
        north = out[-1:]
        ref = _fold_row(field[-1:], fold.perm_T, fold.vector_sign_u, N_LON)
        np.testing.assert_allclose(north, ref, rtol=0, atol=0)

    def test_is_vector_uv_mutually_exclusive(self, single_rank_tripole_layout):
        field = _rand((N_LAT, N_LON), seed=99)
        with pytest.raises(ValueError, match="mutually exclusive"):
            exchange_halo_latlon(
                field, single_rank_tripole_layout, halo=1,
                is_vector_u=True, is_vector_v=True)

    def test_non_tripole_layout_unchanged(self):
        """Without a fold, the north halo is the geographic pole-fold —
        backward compatibility."""
        from legoesm.parallel.latlon_mpi import _pole_fold_north
        layout = make_latlon_band_layout(0, 1, N_LAT, N_LON)  # no fold
        field = _rand((N_LAT, N_LON), seed=9)
        out = exchange_halo_latlon(field, layout, halo=1, is_vector_v=False)
        np.testing.assert_allclose(
            out[-1:], _pole_fold_north(field, 1, negate=False), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Fix 3: backend-dispatched pad_halo_latlon also tripolar-aware at north
# ---------------------------------------------------------------------------


class TestPadHaloLatlonTripolar:
    """``pad_halo_latlon`` / ``pad_halo_latlon_vector`` (the lon-wrap +
    lat-fold backend path) must apply the ORCA permutation fold at the
    north on an active tripolar layout, not the atmospheric 180°-roll
    (codex #353 finding 3)."""

    def _run(self, layout, field, *, vector):
        from legoesm.grids.halo import set_halo_backend, get_halo_backend
        from legoesm.grids.halo_latlon import (
            pad_halo_latlon, pad_halo_latlon_vector,
        )
        prev = get_halo_backend()
        try:
            set_halo_backend("mpi", layout)
            fn = pad_halo_latlon_vector if vector else pad_halo_latlon
            return fn(field, halo=1)
        finally:
            set_halo_backend(prev)

    def test_scalar_north_fold(self, single_rank_tripole_layout, tripole_geom):
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON), seed=12)
        out = self._run(single_rank_tripole_layout, field, vector=False)
        # (N_LAT+2, N_LON+2): strip the lon halo, take the north row.
        north = out[-1, 1:1 + N_LON]
        ref = _fold_tripolar_north(field, 1, fold.perm_T, 1.0)[0]
        np.testing.assert_allclose(north, ref, rtol=0, atol=0)

    def test_vector_v_north_fold(self, single_rank_tripole_layout, tripole_geom):
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON), seed=13)
        out = self._run(single_rank_tripole_layout, field, vector=True)
        north = out[-1, 1:1 + N_LON]
        ref = _fold_tripolar_north(field, 1, fold.perm_v, fold.vector_sign_v)[0]
        np.testing.assert_allclose(north, ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Part 6: pad_with_pole_bc_lat — tripolar north = fold (not wall)
# ---------------------------------------------------------------------------


class TestPadWithPoleBcTripolar:
    """The tripolar north fold in pad_with_pole_bc_lat is OPT-IN via
    ``north_fold``: default keeps the WALL (backward compatible); only
    ``north_fold=True`` produces the fold seam.  South always walls.
    Exercised via the MPI backend with a single-rank tripolar layout (no
    mpi4jax needed)."""

    def _run(self, layout, field, *, north_fold):
        from legoesm.grids.halo import set_halo_backend, get_halo_backend
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        prev = get_halo_backend()
        try:
            set_halo_backend("mpi", layout)
            return pad_with_pole_bc_lat(
                field, halo=1, south_value=0.0, north_value=0.0,
                is_vector_v=False, north_fold=north_fold)
        finally:
            set_halo_backend(prev)

    def test_default_keeps_north_wall(self, single_rank_tripole_layout):
        """Backward-compat: without north_fold the north stays a zero wall
        even on a tripolar layout (codex #353 finding 2)."""
        field = _rand((N_LAT, N_LON), seed=10)
        out = self._run(single_rank_tripole_layout, field, north_fold=False)
        np.testing.assert_allclose(out[0], jnp.zeros((N_LON,)), rtol=0, atol=0)
        np.testing.assert_allclose(out[-1], jnp.zeros((N_LON,)), rtol=0, atol=0)

    def test_north_fold_opt_in(self, single_rank_tripole_layout, tripole_geom):
        fold = tripole_geom.fold
        field = _rand((N_LAT, N_LON), seed=10)
        out = self._run(single_rank_tripole_layout, field, north_fold=True)
        # South wall: zero row.
        np.testing.assert_allclose(out[0], jnp.zeros((N_LON,)), rtol=0, atol=0)
        # North seam: fold partner of the last interior row, NOT zero.
        ref_north = _fold_row(field[-1:], fold.perm_T, 1.0, N_LON)[0]
        np.testing.assert_allclose(out[-1], ref_north, rtol=0, atol=0)
        assert float(jnp.max(jnp.abs(out[-1]))) > 0.0


# ---------------------------------------------------------------------------
# AD safety
# ---------------------------------------------------------------------------


class TestADSafe:
    def test_grad_through_tripolar_exchange_finite(
        self, single_rank_tripole_layout):
        field = _rand((N_LAT, N_LON), seed=11)

        def loss(f):
            out = exchange_halo_latlon(
                f, single_rank_tripole_layout, halo=2, is_vector_v=True)
            return jnp.sum(out ** 2)

        g = jax.grad(loss)(field)
        assert g.shape == field.shape
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.sum(jnp.abs(g))) > 0.0


# ---------------------------------------------------------------------------
# Part 4: ocean state scatter / gather round-trip (single rank == identity)
# ---------------------------------------------------------------------------


def _ocean_state(n_lat, n_lon, nlev, *, with_optional=False, seed=20):
    from legoesm.core.field import Field
    from legoesm.ocean.state import LatLonCGridOceanState
    rng = np.random.default_rng(seed)

    def F(shape):
        return Field(data=jnp.asarray(rng.standard_normal(shape)), name="x")

    kw = {}
    if with_optional:
        kw = dict(
            T_som=F((n_lat, n_lon, nlev, 9)),
            S_som=F((n_lat, n_lon, nlev, 9)),
            T_flux_div_prev=F((n_lat, n_lon, nlev)),
            S_flux_div_prev=F((n_lat, n_lon, nlev)),
        )
    return LatLonCGridOceanState(
        u=F((n_lat, n_lon + 1, nlev)), v=F((n_lat + 1, n_lon, nlev)),
        T=F((n_lat, n_lon, nlev)), S=F((n_lat, n_lon, nlev)),
        eta=F((n_lat, n_lon)), H_bathy=F((n_lat, n_lon)),
        land_mask=F((n_lat, n_lon)), u_mask=F((n_lat, n_lon + 1)),
        v_mask=F((n_lat + 1, n_lon)), w=F((n_lat, n_lon, nlev)),
        **kw,
    )


class TestOceanScatterGather:
    @pytest.mark.parametrize("with_optional", [False, True])
    def test_single_rank_roundtrip_identity(self, with_optional):
        nlev = 3
        state = _ocean_state(N_LAT, N_LON, nlev, with_optional=with_optional)
        layout = make_latlon_band_layout(0, 1, N_LAT, N_LON)
        band = scatter_state_latlon_cgrid_ocean(state, layout)
        # Single rank owns the whole grid: scatter is a no-op slice.
        assert band.u.shape == (N_LAT, N_LON + 1, nlev)
        assert band.v.shape == (N_LAT + 1, N_LON, nlev)
        recovered = gather_state_latlon_cgrid_ocean(band, layout)
        for name in ("u", "v", "T", "S", "eta", "H_bathy",
                     "land_mask", "u_mask", "v_mask", "w"):
            np.testing.assert_allclose(
                getattr(recovered, name).data, getattr(state, name).data,
                rtol=0, atol=0, err_msg=f"{name} round-trip mismatch")
        if with_optional:
            for name in ("T_som", "S_som", "T_flux_div_prev", "S_flux_div_prev"):
                np.testing.assert_allclose(
                    getattr(recovered, name).data, getattr(state, name).data,
                    rtol=0, atol=0)
        else:
            assert recovered.T_som is None and recovered.S_som is None

    def test_band_slicing_stagger_shapes(self):
        """A non-trivial band: v-face fields get the +1 shared row."""
        nlev = 2
        state = _ocean_state(N_LAT, N_LON, nlev)
        # Pretend 3 ranks; check rank 1's band shapes.
        layout = make_latlon_band_layout(1, 3, N_LAT, N_LON)
        n_loc = layout.n_lat_local
        band = scatter_state_latlon_cgrid_ocean(state, layout)
        assert band.T.shape == (n_loc, N_LON, nlev)
        assert band.u.shape == (n_loc, N_LON + 1, nlev)
        assert band.u_mask.shape == (n_loc, N_LON + 1)
        # v-face: one extra (shared boundary) row.
        assert band.v.shape == (n_loc + 1, N_LON, nlev)
        assert band.v_mask.shape == (n_loc + 1, N_LON)


# ---------------------------------------------------------------------------
# Part 5: geometry / zcoord band-slicing
# ---------------------------------------------------------------------------


class TestGeometrySlicing:
    def test_single_rank_slice_identity(self, tripole_geom):
        layout = make_latlon_band_layout(0, 1, N_LAT, N_LON, fold=tripole_geom.fold)
        band = slice_cgrid_geometry_to_band(tripole_geom, layout)
        assert band.n_lat == N_LAT
        for name in ("lat_T", "dx_T", "area_T", "f_T",
                     "dx_u", "f_u", "dx_v", "f_v", "area_q",
                     "cos_alpha_u", "cos_alpha_v"):
            np.testing.assert_allclose(
                getattr(band, name), getattr(tripole_geom, name),
                rtol=0, atol=0, err_msg=f"{name}")
        # fold carried through; perm unchanged (lon-only).
        np.testing.assert_array_equal(band.fold.perm_T, tripole_geom.fold.perm_T)

    def test_multiband_concat_recovers_global_tpoints(self, tripole_geom):
        """T-point metrics concatenated across bands == global."""
        n_ranks = 3
        layouts = [
            make_latlon_band_layout(r, n_ranks, N_LAT, N_LON, fold=tripole_geom.fold)
            for r in range(n_ranks)
        ]
        bands = [slice_cgrid_geometry_to_band(tripole_geom, ly) for ly in layouts]
        for name in ("dx_T", "area_T", "f_T", "lat_T", "f_u"):
            recon = jnp.concatenate([getattr(b, name) for b in bands], axis=0)
            np.testing.assert_allclose(
                recon, getattr(tripole_geom, name), rtol=0, atol=0,
                err_msg=f"{name} concat mismatch")

    def test_total_area_is_global_on_every_band(self, tripole_geom):
        """total_area is the GLOBAL denominator on every rank (codex #353
        finding 5), not a band-local sum."""
        for r in range(3):
            layout = make_latlon_band_layout(
                r, 3, N_LAT, N_LON, fold=tripole_geom.fold)
            band = slice_cgrid_geometry_to_band(tripole_geom, layout)
            np.testing.assert_allclose(
                band.total_area, tripole_geom.total_area, rtol=0, atol=0)

    def test_fold_active_on_all_ranks_local_only_on_north(self, tripole_geom):
        """is_tripolar() must be consistent across all ranks (issue #356).

        Under MPI, divergent is_tripolar() results cause different code
        paths and MPI call counts → MPI_ERR_TRUNCATE.  The fix: fold is
        ACTIVE on all ranks, with fold_j=-1 as a sentinel on non-
        northernmost ranks meaning "fold exists but is not local."
        The _fold_is_local() helper checks fold_j >= 0.
        """
        n_ranks = 3
        for r in range(n_ranks):
            layout = make_latlon_band_layout(
                r, n_ranks, N_LAT, N_LON, fold=tripole_geom.fold)
            band = slice_cgrid_geometry_to_band(tripole_geom, layout)
            assert band.fold.is_active, (
                f"rank {r}: fold must be active on ALL ranks (issue #356)")
            if layout.north_rank is None:
                assert band.fold.fold_j >= 0, (
                    f"rank {r} owns north → fold_j must be >= 0")
                assert band.fold.cap_j >= 0, (
                    f"rank {r} owns north → cap_j must be >= 0")
            else:
                assert band.fold.fold_j == -1, (
                    f"rank {r} is interior → fold_j must be -1 sentinel")
                assert band.fold.cap_j == -1, (
                    f"rank {r} is interior → cap_j must be -1 sentinel")


class TestZcoordSlicing:
    def test_partial_cell_per_column_arrays_sliced(self):
        """Per-cell arrays sliced; reference profiles + scalars passthrough."""
        from legoesm.ocean.vertical import (
            create_ocean_z_star, create_partial_cell_coordinate,
        )
        nlev = 5
        base = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
        H_bathy = jnp.asarray(
            np.random.default_rng(0).uniform(500.0, 4000.0, (N_LAT, N_LON)))
        zc = create_partial_cell_coordinate(base, H_bathy)
        layout = make_latlon_band_layout(1, 3, N_LAT, N_LON)
        s, e = layout.lat_start, layout.lat_end
        band = slice_zcoord_to_band(zc, layout)
        # h_partial is a per-cell (n_lat, n_lon, nlev) field → sliced.
        assert band.h_partial.shape == (layout.n_lat_local, N_LON, nlev)
        np.testing.assert_allclose(
            band.h_partial, zc.h_partial[s:e], rtol=0, atol=0)
        # reference profile unchanged.
        np.testing.assert_allclose(band.dz_ref, zc.dz_ref, rtol=0, atol=0)
        assert int(band.n_levels) == nlev
