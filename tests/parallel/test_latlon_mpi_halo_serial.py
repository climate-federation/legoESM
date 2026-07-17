"""Single-rank correctness tests for the lat-lon band-MPI halo machinery.

These tests do NOT require mpi4jax / mpi4py — they exercise the
``n_ranks=1`` branch of
:func:`legoesm.parallel.latlon_mpi.exchange_halo_latlon` and friends,
which has to reproduce the serial pole-fold convention.

Multi-rank counterparts live in
:mod:`tests.distributed.test_latlon_mpi_halo`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_halo_latlon_3d,
    pad_halo_latlon_vector,
    pad_halo_latlon_vector_3d,
)
from legoesm.parallel.latlon_mpi import (
    LatLonBandLayout,
    build_padded_grid,
    exchange_halo_latlon,
    make_latlon_band_layout,
    pad_state_halos,
    scatter_state_latlon,
    strip_halos,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_rank_layout():
    """A 1-rank layout that owns the whole lat dimension."""
    n_lat, n_lon = 12, 16
    return make_latlon_band_layout(rank=0, n_ranks=1, n_lat=n_lat, n_lon=n_lon)


@pytest.fixture
def scalar_field_2d(single_rank_layout):
    """A spatially non-trivial 2D scalar field on the layout's lat × lon grid."""
    n_lat, n_lon = (
        single_rank_layout.n_lat_global,
        single_rank_layout.n_lon_global,
    )
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat)
    lon = jnp.linspace(0.0, 2.0 * jnp.pi, n_lon, endpoint=False)
    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")
    return jnp.sin(2.0 * lon2d) * jnp.cos(lat2d) + 0.3 * jnp.sin(3.0 * lat2d)


@pytest.fixture
def scalar_field_3d(scalar_field_2d):
    nlev = 5
    return jnp.stack([scalar_field_2d * (k + 1.0) for k in range(nlev)], axis=-1)


# ---------------------------------------------------------------------------
# exchange_halo_latlon: single-rank reproduces serial pole-fold
# ---------------------------------------------------------------------------


class TestSingleRankPoleFold:
    """At n_ranks=1, exchange_halo_latlon must reproduce the lat halo of
    the canonical serial ``pad_halo_latlon*`` helpers, restricted to the
    interior longitude columns (MPI keeps full lon on every rank, so we
    do NOT lon-pad)."""

    @pytest.mark.parametrize("halo", [1, 2, 3])
    def test_scalar_2d(self, scalar_field_2d, single_rank_layout, halo):
        out = exchange_halo_latlon(
            scalar_field_2d, single_rank_layout, halo=halo, is_vector_v=False,
        )
        ref = pad_halo_latlon(scalar_field_2d, halo=halo)
        # serial pads in lon too; strip the lon pad before comparing
        ref_lat_only = ref[:, halo:-halo]
        np.testing.assert_allclose(out, ref_lat_only, rtol=0, atol=0)

    @pytest.mark.parametrize("halo", [1, 2])
    def test_scalar_3d(self, scalar_field_3d, single_rank_layout, halo):
        out = exchange_halo_latlon(
            scalar_field_3d, single_rank_layout, halo=halo, is_vector_v=False,
        )
        ref = pad_halo_latlon_3d(scalar_field_3d, halo=halo)
        ref_lat_only = ref[:, halo:-halo, :]
        np.testing.assert_allclose(out, ref_lat_only, rtol=0, atol=0)

    @pytest.mark.parametrize("halo", [1, 2])
    def test_vector_v_2d(self, scalar_field_2d, single_rank_layout, halo):
        """v-field gets sign flip at the pole; verify by reusing the
        ``pad_halo_latlon_vector`` reference."""
        out = exchange_halo_latlon(
            scalar_field_2d, single_rank_layout, halo=halo, is_vector_v=True,
        )
        ref = pad_halo_latlon_vector(scalar_field_2d, halo=halo)
        ref_lat_only = ref[:, halo:-halo]
        np.testing.assert_allclose(out, ref_lat_only, rtol=0, atol=0)

    @pytest.mark.parametrize("halo", [1, 2])
    def test_vector_v_3d(self, scalar_field_3d, single_rank_layout, halo):
        out = exchange_halo_latlon(
            scalar_field_3d, single_rank_layout, halo=halo, is_vector_v=True,
        )
        ref = pad_halo_latlon_vector_3d(scalar_field_3d, halo=halo)
        ref_lat_only = ref[:, halo:-halo, :]
        np.testing.assert_allclose(out, ref_lat_only, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Layout sanity
# ---------------------------------------------------------------------------


class TestLayout:
    def test_layout_single_rank_owns_all(self):
        layout = make_latlon_band_layout(0, 1, n_lat=24, n_lon=48)
        assert layout.lat_start == 0
        assert layout.lat_end == 24
        assert layout.n_lat_local == 24
        assert layout.south_rank is None
        assert layout.north_rank is None

    def test_layout_balanced_4ranks(self):
        layouts = [
            make_latlon_band_layout(r, 4, n_lat=20, n_lon=40)
            for r in range(4)
        ]
        # Even division: 20 / 4 = 5 rows each.
        for lay in layouts:
            assert lay.n_lat_local == 5
        assert layouts[0].lat_start == 0 and layouts[0].lat_end == 5
        assert layouts[1].lat_start == 5 and layouts[1].lat_end == 10
        assert layouts[2].lat_start == 10 and layouts[2].lat_end == 15
        assert layouts[3].lat_start == 15 and layouts[3].lat_end == 20

    def test_layout_unbalanced_remainder(self):
        # 22 lat / 4 ranks ⇒ first 2 ranks get 6 rows, last 2 get 5.
        layouts = [
            make_latlon_band_layout(r, 4, n_lat=22, n_lon=40)
            for r in range(4)
        ]
        assert [lay.n_lat_local for lay in layouts] == [6, 6, 5, 5]
        # Coverage is exact and non-overlapping.
        ends = [lay.lat_end for lay in layouts]
        starts = [lay.lat_start for lay in layouts]
        assert starts == [0, 6, 12, 17]
        assert ends == [6, 12, 17, 22]

    def test_layout_pole_neighbors(self):
        layouts = [
            make_latlon_band_layout(r, 4, n_lat=16, n_lon=32)
            for r in range(4)
        ]
        assert layouts[0].south_rank is None and layouts[0].north_rank == 1
        assert layouts[-1].north_rank is None and layouts[-1].south_rank == 2

    def test_layout_rejects_oversubscription(self):
        # Cannot give every rank at least 1 lat row.
        with pytest.raises(ValueError, match="Cannot decompose"):
            make_latlon_band_layout(0, 4, n_lat=3, n_lon=8)


# ---------------------------------------------------------------------------
# pad_state_halos / strip_halos round-trip
# ---------------------------------------------------------------------------


class _FakeState:
    """Lightweight state mock shaped like CGridLatLonHydrostaticState.

    Lets us test pad/strip round-trip without depending on the dycore
    config or sigma coords.  Field shapes mirror the real state:

      u : (n_lat,   n_lon, nlev)  — represented here as scalar (we
                                     ignore the n_lon+1 face-count
                                     subtlety in this serial test;
                                     MPI tests cover the v-face row
                                     duplication explicitly)
      v : (n_lat+1, n_lon, nlev)
      T, p_s, phis : standard shapes
    """

    __slots__ = ("u", "v", "T", "p_s", "phis", "tracers")

    def __init__(self, u, v, T, p_s, phis, tracers):
        self.u, self.v, self.T = u, v, T
        self.p_s, self.phis = p_s, phis
        self.tracers = tracers

    def _replace(self, **kwargs):
        merged = {
            "u": self.u, "v": self.v, "T": self.T,
            "p_s": self.p_s, "phis": self.phis,
            "tracers": self.tracers,
        }
        merged.update(kwargs)
        return _FakeState(**merged)


def _make_state(n_lat: int, n_lon: int, nlev: int, with_tracers: bool):
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon, nlev)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)))
    T = jnp.asarray(rng.standard_normal((n_lat, n_lon, nlev)))
    p_s = jnp.asarray(rng.standard_normal((n_lat, n_lon)))
    phis = jnp.asarray(rng.standard_normal((n_lat, n_lon)))
    tracers = {}
    if with_tracers:
        tracers["q_v"] = jnp.asarray(rng.standard_normal((n_lat, n_lon, nlev)))
        tracers["q_c"] = jnp.asarray(rng.standard_normal((n_lat, n_lon, nlev)))
    return _FakeState(u, v, T, p_s, phis, tracers)


class TestPadStripRoundTrip:
    @pytest.mark.parametrize("halo", [1, 2])
    @pytest.mark.parametrize("with_tracers", [False, True])
    def test_pad_strip_identity(self, halo, with_tracers, single_rank_layout):
        state = _make_state(
            n_lat=single_rank_layout.n_lat_global,
            n_lon=single_rank_layout.n_lon_global,
            nlev=4,
            with_tracers=with_tracers,
        )
        padded = pad_state_halos(state, single_rank_layout, halo=halo)
        recovered = strip_halos(padded, single_rank_layout, halo=halo)
        for field in ("u", "v", "T", "p_s", "phis"):
            np.testing.assert_allclose(
                getattr(recovered, field), getattr(state, field),
                rtol=0, atol=0,
            )
        if with_tracers:
            for name in state.tracers:
                np.testing.assert_allclose(
                    recovered.tracers[name], state.tracers[name],
                    rtol=0, atol=0,
                )

    def test_strip_zero_halo_noop(self, single_rank_layout):
        state = _make_state(
            n_lat=single_rank_layout.n_lat_global,
            n_lon=single_rank_layout.n_lon_global,
            nlev=4, with_tracers=False,
        )
        assert strip_halos(state, single_rank_layout, halo=0) is state


# ---------------------------------------------------------------------------
# build_padded_grid
# ---------------------------------------------------------------------------


@pytest.fixture
def latlon_grid():
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=24)


class TestBuildPaddedGrid:
    """Padded-grid construction must (a) be bit-exact to the serial grid
    at interior cells, and (b) keep ``lat`` strictly monotonic across
    halos so vertex-area formulas don't divide by zero.  An earlier
    implementation used ``mode='edge'`` to pad ``lat``, producing
    duplicate values at the halo and ``A_vertex = 0`` inside
    ``curl_vertex_cgrid`` (diagnosed by
    ``scripts/tmp/_diag_mpi_step_nans.py``).  The fix uses linear
    extrapolation by ``dlat``.
    """

    @pytest.mark.parametrize("halo", [1, 2])
    def test_interior_matches_serial_within_storage_eps(
        self, latlon_grid, halo,
    ):
        """Interior cells of the padded grid must equal the original
        grid to within the storage dtype's epsilon.

        Exact bit-equality fails because the original grid stores
        ``cos_lat`` etc. at the storage dtype (fp32 by default) — the
        serial path computes ``cos`` in fp64 then casts, whereas this
        helper takes the fp32-stored ``lat`` and recomputes ``cos``.
        The metric difference is at the storage-eps level (~1e-6 in
        fp32) and irrelevant for the operator stencils — the dycore-
        level bit-exactness contract is enforced by
        ``test_state_after_one_step_matches_serial`` in
        ``test_latlon_mpi_step_serial.py`` instead.
        """
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1,
            n_lat=latlon_grid.n_lat, n_lon=latlon_grid.n_lon,
        )
        padded = build_padded_grid(latlon_grid, layout, halo=halo)
        assert padded.n_lat == latlon_grid.n_lat + 2 * halo
        # ``lat`` itself is just a slice of the original (no recompute),
        # so it remains bit-exact.
        np.testing.assert_allclose(
            padded.lat[halo:-halo], latlon_grid.lat,
            rtol=0, atol=0,
            err_msg="interior lat must remain a bit-exact slice",
        )
        # ``dy`` is uniform; recompute matches up to storage eps.
        np.testing.assert_allclose(
            padded.dy[halo:-halo], latlon_grid.dy,
            rtol=1e-6, atol=1e-6,
        )
        for field in ("cos_lat", "sin_lat"):
            sliced = getattr(padded, field)[halo:-halo]
            np.testing.assert_allclose(
                sliced, getattr(latlon_grid, field),
                rtol=1e-6, atol=1e-6,
                err_msg=f"interior {field} diverges from serial beyond eps",
            )
        for field in ("lat2d", "lon2d", "f", "dx", "area"):
            sliced = getattr(padded, field)[halo:-halo, :]
            ref = getattr(latlon_grid, field)
            # ``area`` near the pole has relative eps amplified by the
            # tiny cos(lat) factor — absolute tolerance is what matters.
            np.testing.assert_allclose(
                sliced, ref, rtol=1e-6, atol=1e-6 * float(jnp.max(jnp.abs(ref))),
                err_msg=f"interior {field} diverges from serial beyond eps",
            )

    @pytest.mark.parametrize("halo", [1, 2])
    def test_halo_lat_steps_from_the_pole(self, latlon_grid, halo):
        """Halo ``lat`` must step away from the pole boundary, not from
        the interior cell centers.

        Cell centers sit at ``-π/2 + dlat/2`` (south-most) and
        ``π/2 - dlat/2`` (north-most).  Extrapolating from these by
        whole ``dlat`` would place halo cells at lat values
        *symmetric* across the pole to interior rows, and ``sin`` is
        symmetric about ``-π/2`` — so the operator's
        ``A_vertex = R² dlon |sin Δ|`` collapses to 0.  Instead the
        halo cell at offset ``k`` (1-indexed from the pole) sits at
        ``∓π/2 ∓ k·dlat``.  This keeps ``sin`` strictly monotonic in
        the halo and the operator finite.
        """
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1,
            n_lat=latlon_grid.n_lat, n_lon=latlon_grid.n_lon,
        )
        padded = build_padded_grid(latlon_grid, layout, halo=halo)
        dlat = float(latlon_grid.dlat)
        for k in range(halo):
            expected_south = -float(np.pi) / 2.0 - (halo - k) * dlat
            np.testing.assert_allclose(
                padded.lat[k], expected_south, rtol=0, atol=1e-6,
            )
            expected_north = float(np.pi) / 2.0 + (k + 1) * dlat
            np.testing.assert_allclose(
                padded.lat[-halo + k], expected_north,
                rtol=0, atol=1e-6,
            )

    @pytest.mark.parametrize("halo", [1, 2])
    def test_vertex_areas_strictly_positive_at_halo(
        self, latlon_grid, halo,
    ):
        """The root cause of the Stage-2 NaN smoke failure was
        ``A_vertex = 0`` in the halo region.  Pin that down: with
        linear extrapolation, every consecutive pair of padded
        latitudes must produce a positive ``|sin Δ|``."""
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1,
            n_lat=latlon_grid.n_lat, n_lon=latlon_grid.n_lon,
        )
        padded = build_padded_grid(latlon_grid, layout, halo=halo)
        sin_lat = np.sin(np.asarray(padded.lat, dtype=np.float64))
        diffs = np.abs(np.diff(sin_lat))
        assert np.all(diffs > 0), (
            f"build_padded_grid produced duplicate / zero-Δ sin(lat) at "
            f"indices {np.where(diffs == 0)[0].tolist()} — this would "
            f"NaN the Coriolis term in curl_vertex_cgrid."
        )

    def test_zero_halo_passthrough(self, latlon_grid):
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1,
            n_lat=latlon_grid.n_lat, n_lon=latlon_grid.n_lon,
        )
        padded = build_padded_grid(latlon_grid, layout, halo=0)
        assert padded.n_lat == latlon_grid.n_lat
        np.testing.assert_allclose(padded.lat, latlon_grid.lat, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Stage-1 guardrails (full equivalence test lives in
# tests/parallel/test_latlon_mpi_step_serial.py, which requires the
# dycore stack)
# ---------------------------------------------------------------------------


class TestStage12Guardrails:
    """Stage-2 supports dry PE + tracers.  Physics (``physics_fn``)
    remains unsupported and must raise loudly.  Polar filter is now
    supported (Stage 3-E commit cd3e662d+) — its guardrail test was
    inverted to verify the lift, not the legacy NotImplementedError.
    """

    def test_polar_filter_accepted(self, single_rank_layout):
        """``config.use_polar_filter=True`` → factory builds the step
        function without raising.  Previously raised
        ``NotImplementedError`` (Stage-3 placeholder); Stage 3-E now
        ships the filter under MPI so the wrapper accepts the flag
        and the slice-equivariance tests in
        ``tests/distributed/test_latlon_mpi_polar_filter.py`` pin
        the correctness of the rank-local mask + 2-D + 3-D filter.
        """
        from legoesm.parallel.latlon_mpi import make_latlon_mpi_step
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel,
            CGridLatLonPrimitiveEquationConfig,
        )
        grid = create_latlon_grid(n_lat=8)
        sigma = create_sigma_coordinate(n_levels=4)
        cfg = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, use_polar_filter=True,
        )
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
        step_fn = make_latlon_mpi_step(model, single_rank_layout, halo=2)
        assert callable(step_fn), (
            "make_latlon_mpi_step must return a callable when "
            "use_polar_filter=True; got "
            f"{type(step_fn).__name__}"
        )
        # Additionally pin the mask shapes — a regression that
        # returns a step_fn whose internal model was built with the
        # wrong dt (and therefore the wrong-size or zero mask) would
        # silently fail every step under load.  Exposed via the
        # closed-over ``mpi_model`` on the step_fn closure.
        # Key on the free-variable name, not index position — Codex
        # round-4 non-blocking suggestion: a future refactor that
        # reorders closure variables would silently make
        # ``__closure__[0]`` point at the wrong object and the rest
        # of this test would pass against an unrelated object.
        _closure = dict(zip(
            step_fn.__code__.co_freevars,
            step_fn.__closure__,
        ))
        mpi_model = _closure["mpi_model"].cell_contents
        # ``mpi_model`` is the rebuilt CGridLatLonPrimitiveEquationModel
        # inside make_latlon_mpi_step; its ``_polar_mask`` and
        # ``_polar_mask_v`` must be non-None and properly shaped.
        assert mpi_model._polar_mask is not None, (
            "MPI model rebuilt by make_latlon_mpi_step is missing the "
            "cell-centered polar filter mask."
        )
        assert mpi_model._polar_mask_v is not None, (
            "MPI model rebuilt by make_latlon_mpi_step is missing the "
            "v-face polar filter mask."
        )
        # Mask shapes match the rank-local band on the single-rank
        # layout (which is the entire grid).
        assert mpi_model._polar_mask.shape == (grid.n_lat, grid.n_lon // 2 + 1)
        assert mpi_model._polar_mask_v.shape == (grid.n_lat + 1, grid.n_lon // 2 + 1)


# ---------------------------------------------------------------------------
# AD-VJP: jax.grad through the single-rank halo path is finite
# ---------------------------------------------------------------------------


class TestADSafe:
    def test_grad_through_serial_halo(
        self, scalar_field_2d, single_rank_layout,
    ):
        """The single-rank branch routes through pure JAX (no mpi4jax
        custom_vjp), so jax.grad must produce a finite, sensible
        gradient.  The multi-rank AD path is covered separately under
        tests/distributed/."""

        def loss(field):
            padded = exchange_halo_latlon(
                field, single_rank_layout, halo=2, is_vector_v=False,
            )
            return jnp.sum(padded ** 2)

        g = jax.grad(loss)(scalar_field_2d)
        assert g.shape == scalar_field_2d.shape
        assert jnp.all(jnp.isfinite(g))
        # Loss is sum(padded**2), so gradient at any *interior* point
        # is at least 2 * field there (the pole-fold reuses the same
        # values, so pole rows accumulate extra contributions but
        # interior points only contribute once → exactly 2 * x).
        interior = g[2:-2]
        expected = 2.0 * scalar_field_2d[2:-2]
        np.testing.assert_allclose(interior, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# interp_cell_to_vface_halo: serial bit-identity guarantees
# ---------------------------------------------------------------------------
#
# The backend-aware cell→v-face interp used by the PE dycore (pressure-
# gradient T_v, continuity dp_v, vertical-advection sd_v, physics-
# tendency dv coupling) promises BIT-IDENTICAL serial behaviour: under
# the local backend, under an armed non-band MPI topology, and on a
# single-rank band layout it must reproduce the legacy
# ``interp_cell_to_vface`` (pole edge-copy convention) exactly.  The
# interior-cut averaging branch needs real ranks and is pinned by
# tests/distributed/test_latlon_mpi_step.py at np>1.


class TestInterpCellToVfaceHaloSerial:

    @pytest.fixture
    def cell_field_3d(self, single_rank_layout):
        """Non-trivial (n_lat, n_lon, nlev) cell-centred field."""
        n_lat, n_lon = (
            single_rank_layout.n_lat_global,
            single_rank_layout.n_lon_global,
        )
        rng = np.random.default_rng(20260610)
        return jnp.asarray(rng.standard_normal((n_lat, n_lon, 5)))

    def test_local_backend_bit_identical(self, cell_field_3d):
        """No backend armed → literal delegation to the legacy interp."""
        from legoesm.grids.halo import set_halo_backend
        from legoesm.grids.operators_latlon_cgrid import (
            interp_cell_to_vface,
            interp_cell_to_vface_halo,
        )
        set_halo_backend("local")
        ref = interp_cell_to_vface(cell_field_3d)
        out = interp_cell_to_vface_halo(cell_field_3d)
        assert out.shape == (
            cell_field_3d.shape[0] + 1,
            *cell_field_3d.shape[1:],
        )
        assert out.dtype == ref.dtype
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))

    def test_single_rank_band_backend_bit_identical(
        self, cell_field_3d, single_rank_layout,
    ):
        """Armed single-rank band layout (south/north both poles) →
        still the legacy pole edge-copy convention, bit-for-bit."""
        from legoesm.grids.halo import set_halo_backend
        from legoesm.grids.operators_latlon_cgrid import (
            interp_cell_to_vface,
            interp_cell_to_vface_halo,
        )
        set_halo_backend("mpi", single_rank_layout)
        try:
            out = interp_cell_to_vface_halo(cell_field_3d)
        finally:
            set_halo_backend("local")
        ref = interp_cell_to_vface(cell_field_3d)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
        # Pole faces carry the legacy edge copy.
        np.testing.assert_array_equal(
            np.asarray(out[0]), np.asarray(cell_field_3d[0]))
        np.testing.assert_array_equal(
            np.asarray(out[-1]), np.asarray(cell_field_3d[-1]))

    def test_non_band_mpi_topology_falls_back(self, cell_field_3d):
        """Armed MPI backend with a NON-band topology (e.g. the
        cubed-sphere CommTopology) → local fallback, not a crash."""
        from legoesm.grids.halo import set_halo_backend
        from legoesm.grids.operators_latlon_cgrid import (
            interp_cell_to_vface,
            interp_cell_to_vface_halo,
        )

        class _NotABandLayout:
            pass

        set_halo_backend("mpi", _NotABandLayout())
        try:
            out = interp_cell_to_vface_halo(cell_field_3d)
        finally:
            set_halo_backend("local")
        ref = interp_cell_to_vface(cell_field_3d)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
