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
    @pytest.mark.parametrize("halo", [1, 2])
    def test_single_rank_grid_matches_serial_pad(self, latlon_grid, halo):
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1,
            n_lat=latlon_grid.n_lat, n_lon=latlon_grid.n_lon,
        )
        padded = build_padded_grid(latlon_grid, layout, halo=halo)
        # n_lat axis grew by 2*halo
        assert padded.n_lat == latlon_grid.n_lat + 2 * halo
        # Interior of metric slices matches the original
        np.testing.assert_allclose(
            padded.lat[halo:-halo], latlon_grid.lat,
            rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            padded.area[halo:-halo, :], latlon_grid.area,
            rtol=0, atol=0,
        )
        # Pole-side halo metrics were edge-padded (since both pole rows
        # are owned at n_ranks=1, the halo rows fall outside the grid)
        np.testing.assert_allclose(
            padded.lat[:halo], latlon_grid.lat[0], rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            padded.lat[-halo:], latlon_grid.lat[-1], rtol=0, atol=0,
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
    and polar filter remain unsupported and must raise loudly."""

    def test_polar_filter_raises(self, single_rank_layout):
        """``config.use_polar_filter=True`` → factory NotImplementedError."""
        from legoesm.parallel.latlon_mpi import make_latlon_mpi_step
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel,
            CGridLatLonPrimitiveEquationConfig,
        )
        grid = create_latlon_grid(n_lat=8)
        sigma = create_sigma_coordinate(n_levels=4)
        cfg = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, use_polar_filter=True,
        )
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
        with pytest.raises(NotImplementedError, match="polar filter"):
            make_latlon_mpi_step(model, single_rank_layout, halo=2)


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
