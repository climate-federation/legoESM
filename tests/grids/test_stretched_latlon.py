"""Tests for ``create_stretched_latlon_grid`` (EXT-F1).

A lat-lon C-grid builder taking an ARBITRARY per-row meridional spacing
array ``dy_deg`` (degrees) — the missing piece for the Veros
``global_flexible`` setup, whose ``dyt`` is Vinokur-stretched.  The
operators are already variable-dy safe (Mercator/DINO precedent); this
file validates the BUILDER:

1. EXACTLY-uniform ``dy_deg`` reproduces ``create_regional_latlon_grid``
   BIT-exactly (the builder delegates to it), and a near-uniform
   ``dy_deg`` (cumsum + u-centred path) stays within ~ULP of it — the
   delegation branch is continuous.
2. Mercator-spaced ``dy_deg`` reproduces ``create_mercator_grid``'s
   faces / dy / areas to round-off.  CENTRES differ BY CONVENTION:
   Mercator places centres analytically at ``arcsin(tanh(Δλ(k+½)))``,
   the stretched builder uses the pyOM/Veros u-centred recursion
   (faces bisect centres) — documented, and the metric consistency is
   tested instead.
3. Metric self-consistency on a stretched array.
4. The Veros ``global_flexible`` ``set_grid`` Vinokur construction
   (transcribed below) yields interior centres matching Veros ``yt``
   and faces matching Veros ``yu`` (``u_centered_grid`` + the
   ``yu[2] = y_origin`` shift from ``veros/core/numerics.py``).
5. ``ensure_geometry`` consistency (the u-centred placement makes the
   geometry's centre-midpoint face reconstruction EXACT).
6. A 5-step PE smoke on a small stretched channel: finite fields +
   rigid-lid tracer conservation (the operators-are-safe integration
   check).

Run in fp64: placement comparisons are at the 1e-12 level.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import (  # noqa: E402
    create_mercator_grid,
    create_regional_latlon_grid,
    create_stretched_latlon_grid,
)


# =====================================================================
# Transcribed Veros references (veros/tools/setup.py +
# veros/core/numerics.py) — used to construct the global_flexible dyt
# and the oracle yt/yu placement.  Transcribed, not imported: the test
# suite must not depend on a Veros installation.
# =====================================================================


def _veros_sinhc_inverse(y: float) -> float:
    """Approximate inverse of sinh(y)/y (veros.tools.setup)."""
    if y < 2.7829681:
        ybar = y - 1.0
        return np.sqrt(6 * ybar) * (
            1
            - 0.15 * ybar
            + 0.057321429 * ybar**2
            - 0.024907295 * ybar**3
            + 0.0077424461 * ybar**4
            - 0.0010794123 * ybar**5
        )
    v = np.log(y)
    w = 1.0 / y - 0.028527431
    return (
        v
        + (1 + 1.0 / v) * np.log(2 * v)
        - 0.02041793
        + 0.24902722 * w
        + 1.9496443 * w**2
        - 2.6294547 * w**3
        + 8.56795911 * w**4
    )


def _veros_vinokur_two_sided(
    n_cells: int, total_length: float, lower_stepsize: float
) -> np.ndarray:
    """``veros.tools.get_vinokur_grid_steps(..., two_sided_grid=True)``
    for the global_flexible case (no ``upper_stepsize``, default
    ``refine_towards='upper'``).  Symmetric about the centre, finest at
    the centre (equator), coarsest at the ends (poles)."""
    assert n_cells % 2 == 0
    n = n_cells // 2 + 1
    target_sum = total_length * 0.5
    s0 = float(target_sum) / float(lower_stepsize * n)
    assert s0 > 1.0  # global_flexible regime → sinhc branch
    f = _veros_sinhc_inverse(s0) * 0.5
    stretched = 1 + np.tanh(f * np.linspace(0.0, 1.0, n)) / np.tanh(f)
    steps = np.diff(stretched * target_sum)
    steps = steps[::-1]  # refine_towards="upper"
    return np.concatenate([steps[::-1], steps])


def _veros_u_centered_and_shift(dyt_interior: np.ndarray, y_origin: float):
    """Veros ``calc_grid`` meridional placement: ghost-row edge
    extension, ``u_centered_grid``, and the ``yu[2] = y_origin`` shift.
    Returns the FULL (ny+4,) ``yt``/``yu`` including ghost rows."""
    dyt = np.concatenate(
        [dyt_interior[:1], dyt_interior[:1], dyt_interior,
         dyt_interior[-1:], dyt_interior[-1:]]
    )
    n = dyt.size
    yu = np.zeros(n)
    yt = np.zeros(n)
    yu[1:] = np.cumsum(dyt[1:])
    yt[0] = yu[0] - dyt[0] * 0.5
    yt[1:] = 2.0 * yu[:-1]
    alt = np.ones(n)
    alt[::2] = -1.0
    yt = alt * np.cumsum(alt * yt)
    yt = yt + y_origin - yu[2]
    yu = yu + y_origin - yu[2]
    return yt, yu


class TestVinokurTranscription:
    def test_matches_veros_docstring_example(self):
        """Anchor the transcription on the values printed in the Veros
        ``get_vinokur_grid_steps`` docstring:
        ``get_vinokur_grid_steps(14, 180, 5, two_sided_grid=True)``."""
        ref = np.array([
            18.2451554, 17.23915939, 15.43744632, 13.17358802,
            10.78720589, 8.53852027, 6.57892471, 6.57892471,
            8.53852027, 10.78720589, 13.17358802, 15.43744632,
            17.23915939, 18.2451554,
        ])
        steps = _veros_vinokur_two_sided(14, 180.0, 5.0)
        np.testing.assert_allclose(steps, ref, atol=1e-7)
        assert abs(steps.sum() - 180.0) < 1e-10


# =====================================================================
# (a) Uniform dy_deg reproduces create_regional_latlon_grid
# =====================================================================


_GRID_ARRAY_FIELDS = (
    "lat", "lon", "lat2d", "lon2d", "cos_lat", "sin_lat",
    "lat_v", "cos_lat_v", "f", "dx", "dy", "area",
)


@pytest.mark.parametrize("periodic_x", [True, False])
class TestUniformEquivalence:
    """Uniform ``dy_deg`` == the uniform regional builder.

    EXACTLY-uniform input delegates to ``create_regional_latlon_grid``
    → every field is BIT-exact.  A near-uniform input (one entry
    perturbed at the 1e-9-degree level) exercises the cumsum +
    u-centred path, which must stay within float64 round-off of the
    uniform builder — the delegation branch is continuous."""

    def _grids(self, periodic_x, d):
        n_lat, n_lon = 21, 16
        assert d.size == n_lat
        gs, ms = create_stretched_latlon_grid(
            d, n_lon=n_lon, lat_south=-40.0,
            lon_west=10.0, lon_east=74.0,
            periodic_x=periodic_x, dtype=jnp.float64)
        gr, mr = create_regional_latlon_grid(
            n_lat=n_lat, n_lon=n_lon, lat_south=-40.0, lat_north=44.0,
            lon_west=10.0, lon_east=74.0,
            periodic_x=periodic_x, dtype=jnp.float64)
        return gs, ms, gr, mr

    def test_exactly_uniform_bit_exact(self, periodic_x):
        gs, ms, gr, mr = self._grids(periodic_x, np.full(21, 4.0))
        assert (gs.n_lat, gs.n_lon) == (gr.n_lat, gr.n_lon)
        assert gs.radius == gr.radius
        assert gs.dlon == gr.dlon
        assert gs.dlat == gr.dlat
        for name in _GRID_ARRAY_FIELDS:
            np.testing.assert_array_equal(
                np.asarray(getattr(gs, name)),
                np.asarray(getattr(gr, name)), err_msg=name)
        assert float(gs.total_area) == float(gr.total_area)
        np.testing.assert_array_equal(np.asarray(ms), np.asarray(mr))

    def test_near_uniform_cumsum_path_at_roundoff(self, periodic_x):
        d = np.full(21, 4.0)
        d[10] += 1e-9          # breaks exact equality → cumsum path
        gs, ms, gr, mr = self._grids(periodic_x, d)
        np.testing.assert_array_equal(np.asarray(ms), np.asarray(mr))
        np.testing.assert_array_equal(np.asarray(gs.lon), np.asarray(gr.lon))
        # placement tolerances dominated by the 1e-9-deg perturbation
        # (~2e-11 rad); metric fields relative
        np.testing.assert_allclose(
            np.asarray(gs.lat), np.asarray(gr.lat), rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(gs.lat_v), np.asarray(gr.lat_v), rtol=0, atol=1e-10)
        for name in ("cos_lat", "sin_lat", "cos_lat_v"):
            np.testing.assert_allclose(
                np.asarray(getattr(gs, name)),
                np.asarray(getattr(gr, name)), rtol=0, atol=1e-10)
        for name in ("dy", "dx", "area", "f"):
            np.testing.assert_allclose(
                np.asarray(getattr(gs, name)),
                np.asarray(getattr(gr, name)), rtol=1e-9, atol=0)
        # the perturbation genuinely enlarges the span by 1e-9 deg →
        # total_area differs by ~8e-12 RELATIVE (geometric, not noise)
        np.testing.assert_allclose(
            float(gs.total_area), float(gr.total_area), rtol=1e-10)
        assert abs(gs.dlat - gr.dlat) < 1e-10


# =====================================================================
# (b) Mercator-spaced dy_deg vs create_mercator_grid
# =====================================================================


class TestMercatorEquivalence:
    """Feed the Mercator FACE spacings in: faces / dy / areas agree to
    round-off.  CENTRES cannot agree bit-exactly — Mercator's analytic
    centre map ``arcsin(tanh(Δλ(k+½)))`` is a different placement
    convention from the u-centred recursion (documented in the builder
    docstring) — so the centre check bounds the convention delta and
    the metric consistency is asserted instead.  Shapes also differ by
    construction: Mercator has NO wall rows, the stretched builder adds
    one at each end → compare against the stretched INTERIOR ``[1:-1]``.
    """

    @pytest.fixture(scope="class")
    def pair(self):
        gm = create_mercator_grid(
            n_lon=120, lat_max_deg=60.0, dtype=jnp.float64)
        face_deg = np.degrees(np.asarray(gm.lat_v, dtype=np.float64))
        d = np.diff(face_deg)                     # (n_lat,) cell heights
        gs, _ = create_stretched_latlon_grid(
            d, n_lon=120, lat_south=float(face_deg[0]),
            lon_west=0.0, lon_east=360.0,
            periodic_x=True, dtype=jnp.float64)
        return gm, gs

    def test_shapes(self, pair):
        gm, gs = pair
        assert gs.n_lat == gm.n_lat + 2          # wall rows added
        assert gs.n_lon == gm.n_lon

    def test_faces_dy_area_match_interior(self, pair):
        gm, gs = pair
        np.testing.assert_allclose(
            np.asarray(gs.lat_v)[1:-1], np.asarray(gm.lat_v),
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(gs.dy)[1:-1], np.asarray(gm.dy), rtol=1e-11)
        np.testing.assert_allclose(
            np.asarray(gs.area)[1:-1], np.asarray(gm.area), rtol=1e-11)
        assert gs.dlon == pytest.approx(gm.dlon, rel=1e-14)

    def test_centres_differ_only_by_convention_delta(self, pair):
        """The u-centred centres sit within their cells and within a
        small fraction of the row height of the Mercator analytic
        centres (the placements agree at O(Δφ²))."""
        gm, gs = pair
        c_s = np.asarray(gs.lat)[1:-1]
        c_m = np.asarray(gm.lat)
        row_h = np.diff(np.asarray(gm.lat_v))
        delta = np.abs(c_s - c_m)
        assert delta.max() > 0.0                   # genuinely different
        assert np.all(delta < 0.05 * row_h), (
            "centre-convention delta should be a small fraction of the "
            f"row height, got max ratio {(delta / row_h).max():.3e}")

    def test_dlat_scalar_is_min_row(self, pair):
        gm, gs = pair
        # Both builders document dlat = smallest (most CFL-stringent)
        # row.  The stretched grid's wall rows are edge-extended, so the
        # min over rows is the same as Mercator's.
        assert gs.dlat == pytest.approx(gm.dlat, rel=1e-12)


# =====================================================================
# (c) Metric self-consistency on a stretched array
# =====================================================================


def _smooth_stretched_d(n=24, span=60.0):
    """Smooth asymmetric stretching (geometric, ~6% row growth)."""
    raw = 1.06 ** np.arange(n)
    return span * raw / raw.sum()


class TestStretchedConsistency:
    @pytest.fixture(scope="class")
    def grid(self):
        d = _smooth_stretched_d()
        g, mask = create_stretched_latlon_grid(
            d, n_lon=12, lat_south=-20.0, lon_west=0.0, lon_east=60.0,
            periodic_x=False, dtype=jnp.float64)
        return g, mask, d

    def test_faces_and_centres_interleave_monotone(self, grid):
        g, _, _ = grid
        lat_v = np.asarray(g.lat_v)
        lat = np.asarray(g.lat)
        assert np.all(np.diff(lat_v) > 0)
        assert np.all(lat_v[:-1] < lat) and np.all(lat < lat_v[1:])
        # combined face/centre sequence strictly monotone
        merged = np.empty(2 * lat.size + 1)
        merged[::2] = lat_v
        merged[1::2] = lat
        assert np.all(np.diff(merged) > 0)

    def test_dy_sum_equals_lat_span(self, grid):
        g, _, d = grid
        # single-cell heights = dy/2; interior sum == sum(dy_deg)
        heights_deg = np.degrees(np.asarray(g.dy) / 2.0 / g.radius)
        np.testing.assert_allclose(heights_deg[1:-1].sum(), d.sum(),
                                   rtol=1e-13)
        # wall rows edge-extended
        assert heights_deg[0] == pytest.approx(d[0], rel=1e-13)
        assert heights_deg[-1] == pytest.approx(d[-1], rel=1e-13)
        # dy consistent with faces (the defining identity)
        np.testing.assert_allclose(
            np.asarray(g.dy) / 2.0 / g.radius, np.diff(np.asarray(g.lat_v)),
            rtol=0, atol=1e-16)

    def test_first_interior_south_face_at_lat_south(self, grid):
        g, _, _ = grid
        assert np.degrees(float(np.asarray(g.lat_v)[1])) == pytest.approx(
            -20.0, abs=1e-12)

    def test_areas_positive_and_telescope(self, grid):
        g, _, _ = grid
        area = np.asarray(g.area)
        assert np.all(area > 0)
        lat_v = np.asarray(g.lat_v)
        band = g.radius**2 * g.dlon * g.n_lon * (
            np.sin(lat_v[-1]) - np.sin(lat_v[0]))
        assert float(g.total_area) == pytest.approx(band, rel=1e-13)

    def test_cos_f_dx_at_centres(self, grid):
        g, _, _ = grid
        lat = np.asarray(g.lat)
        np.testing.assert_allclose(
            np.asarray(g.cos_lat),
            np.maximum(np.abs(np.cos(lat)), 1e-10), rtol=1e-15)
        np.testing.assert_allclose(
            np.asarray(g.sin_lat), np.sin(lat), rtol=0, atol=1e-15)
        from legoesm import constants
        np.testing.assert_allclose(
            np.asarray(g.f)[:, 0],
            2.0 * constants.Omega * np.sin(lat), rtol=1e-14)
        np.testing.assert_allclose(
            np.asarray(g.dx)[:, 0],
            g.radius * 2.0 * g.dlon * np.asarray(g.cos_lat), rtol=1e-14)

    def test_dlat_scalar_is_min_row(self, grid):
        g, _, d = grid
        assert g.dlat == pytest.approx(np.deg2rad(d.min()), rel=1e-12)

    def test_wall_mask_closed_basin(self, grid):
        g, mask, _ = grid
        m = np.asarray(mask)
        assert m.shape == (g.n_lat, g.n_lon)
        assert np.all(m[0] == 0) and np.all(m[-1] == 0)
        assert np.all(m[:, 0] == 0) and np.all(m[:, -1] == 0)
        assert np.all(m[1:-1, 1:-1] == 1)

    # --- input validation ---

    def test_rejects_nonpositive_dy(self):
        with pytest.raises(ValueError, match="positive and finite"):
            create_stretched_latlon_grid(
                np.array([1.0, -1.0, 1.0]), n_lon=4, lat_south=0.0)

    def test_rejects_non1d(self):
        with pytest.raises(ValueError, match="1-D"):
            create_stretched_latlon_grid(
                np.ones((3, 2)), n_lon=4, lat_south=0.0)

    def test_rejects_span_past_pole(self):
        with pytest.raises(ValueError, match="north pole"):
            create_stretched_latlon_grid(
                np.full(10, 10.0), n_lon=4, lat_south=0.0)

    def test_rejects_too_rapid_variation(self):
        """The u-centred recursion degenerates (centre escapes its
        cell) for rough dy arrays — must fail LOUDLY, not corrupt the
        metrics. 50% row-to-row jumps are enough to trip it."""
        d = np.array([3.0, 2.0, 1.5, 1.0, 1.0, 1.5, 2.0, 3.0])
        with pytest.raises(ValueError, match="varies too rapidly"):
            create_stretched_latlon_grid(d, n_lon=4, lat_south=-7.5)

    def test_rejects_bad_lon_range_closed(self):
        with pytest.raises(ValueError, match="lon_west"):
            create_stretched_latlon_grid(
                np.full(4, 1.0), n_lon=4, lat_south=0.0,
                lon_west=30.0, lon_east=10.0, periodic_x=False)


# =====================================================================
# (d) Veros global_flexible set_grid construction
# =====================================================================


class TestVerosGlobalFlexiblePlacement:
    """Build the global_flexible dyt (Vinokur two-sided, equatorial
    spacing factor 0.5, total 160°, y_origin=-80) at a reduced-class
    resolution and check the builder's interior centres/faces land on
    Veros's ``yt``/``yu`` (the documented mapping:
    ``lat_south = y_origin - dyt[0]``; interior rows ``[1:-1]`` ↔ Veros
    ``[2:-2]``)."""

    @pytest.fixture(scope="class")
    def setup(self):
        ny = 40                                     # rescalable class
        eq_spacing = 0.5 * 160.0 / ny               # the setup's formula
        dyt_int = _veros_vinokur_two_sided(ny, 160.0, eq_spacing)
        y_origin = -80.0
        yt, yu = _veros_u_centered_and_shift(dyt_int, y_origin)
        g, mask = create_stretched_latlon_grid(
            dyt_int, n_lon=18, lat_south=y_origin - dyt_int[0],
            lon_west=90.0, lon_east=450.0,          # x_origin=90, cyclic
            periodic_x=True, dtype=jnp.float64)
        return dyt_int, yt, yu, g, mask

    def test_dyt_is_equator_refined_symmetric(self, setup):
        dyt_int, _, _, _, _ = setup
        assert abs(dyt_int.sum() - 160.0) < 1e-9
        np.testing.assert_allclose(dyt_int, dyt_int[::-1], atol=1e-12)
        assert dyt_int.min() == dyt_int[len(dyt_int) // 2]
        assert dyt_int.max() == dyt_int[0] > 2.0 * dyt_int.min() / 1.5

    def test_interior_centres_match_veros_yt(self, setup):
        _, yt, _, g, _ = setup
        centres_deg = np.degrees(np.asarray(g.lat, dtype=np.float64))
        np.testing.assert_allclose(
            centres_deg[1:-1], yt[2:-2], rtol=0, atol=1e-12)

    def test_faces_match_veros_yu(self, setup):
        _, _, yu, g, _ = setup
        faces_deg = np.degrees(np.asarray(g.lat_v, dtype=np.float64))
        # interior faces = north faces of Veros cells 1..ny+1
        np.testing.assert_allclose(
            faces_deg[1:-1], yu[1:-2], rtol=0, atol=1e-12)

    def test_wall_rows_match_veros_ghost_centres(self, setup):
        """The single wall row each side coincides with Veros's first
        ghost row (same edge-extended dyt + same recursion)."""
        _, yt, _, g, _ = setup
        centres_deg = np.degrees(np.asarray(g.lat, dtype=np.float64))
        assert centres_deg[0] == pytest.approx(yt[1], abs=1e-12)
        assert centres_deg[-1] == pytest.approx(yt[-2], abs=1e-12)


# =====================================================================
# Factory + pytree/JIT
# =====================================================================


class TestFactoryAndPytree:
    def test_factory_dispatch(self):
        from legoesm.grids.factory import (
            REGIONAL_GRID_TYPES, create_regional_grid)
        assert "latlon_stretched" in REGIONAL_GRID_TYPES
        d = np.full(6, 2.0)
        kw = dict(dy_deg=d, n_lon=8, lat_south=-6.0,
                  lon_west=0.0, lon_east=16.0, periodic_x=True,
                  dtype=jnp.float64)
        g_f, m_f = create_regional_grid("latlon_stretched", **kw)
        g_d, m_d = create_stretched_latlon_grid(**kw)
        np.testing.assert_array_equal(np.asarray(g_f.lat), np.asarray(g_d.lat))
        np.testing.assert_array_equal(np.asarray(g_f.area), np.asarray(g_d.area))
        np.testing.assert_array_equal(np.asarray(m_f), np.asarray(m_d))

    def test_grid_is_pytree_and_traces_through_jit(self):
        d = _smooth_stretched_d(12, 30.0)
        g, _ = create_stretched_latlon_grid(
            d, n_lon=8, lat_south=-10.0, lon_west=0.0, lon_east=40.0,
            periodic_x=True, dtype=jnp.float64)
        leaves = jax.tree_util.tree_leaves(g)
        assert len(leaves) > 0

        @jax.jit
        def total_area_of(grid):
            return jnp.sum(grid.area)

        assert float(total_area_of(g)) == pytest.approx(
            float(g.total_area), rel=1e-14)


# =====================================================================
# (e) ensure_geometry consistency
# =====================================================================


class TestGeometryConsistency:
    """``ensure_geometry`` now routes the grid's EXACT face latitudes
    (``grid.lat_v``) into ``create_latlon_geometry``'s variable-dlat
    branch (previously it reconstructed faces as centre midpoints —
    exact for the u-centred interior but ~4% off at the stretched
    grid's north wall row, and only O(Δφ²)-approximate on Mercator).
    All derived per-row metrics must agree with the grid's, INCLUDING
    the wall rows."""

    @pytest.fixture(scope="class")
    def pair(self):
        from legoesm.core.precision import (
            PrecisionPolicy, get_policy, set_policy)
        from legoesm.grids.latlon import ensure_geometry
        d = _smooth_stretched_d(20, 50.0)
        g, _ = create_stretched_latlon_grid(
            d, n_lon=10, lat_south=-25.0, lon_west=0.0, lon_east=50.0,
            periodic_x=True, dtype=jnp.float64)
        prev = get_policy()
        set_policy(PrecisionPolicy.fp64())
        try:
            geom = ensure_geometry(g)
        finally:
            set_policy(prev)
        return g, geom

    def test_per_row_cell_heights_exact_incl_walls(self, pair):
        g, geom = pair
        np.testing.assert_allclose(
            np.asarray(geom.dy_T)[:, 0], np.asarray(g.dy) / 2.0,
            rtol=1e-13)

    def test_areas_match(self, pair):
        g, geom = pair
        np.testing.assert_allclose(
            np.asarray(geom.area_T), np.asarray(g.area), rtol=1e-13)
        assert float(geom.total_area) == pytest.approx(
            float(g.total_area), rel=1e-13)

    def test_dy_v_positive_and_centre_spacings(self, pair):
        g, geom = pair
        dy_v = np.asarray(geom.dy_v)[:, 0]
        assert np.all(dy_v > 0)

    def test_two_point_probe_footgun_detected(self):
        """An array uniform at the boundary AND middle probes but
        stretched elsewhere must still get variable-dlat metrics
        (regression for the centre-based two-point detection)."""
        from legoesm.core.precision import (
            PrecisionPolicy, get_policy, set_policy)
        from legoesm.grids.latlon import ensure_geometry
        # rows 0-1 (→ boundary probe) and the middle pair uniform at
        # 2°; a gentle 2.4° bump in between
        d = np.array([2.0, 2.0, 2.2, 2.4, 2.2, 2.0, 2.0, 2.0,
                      2.0, 2.0, 2.2, 2.4, 2.2, 2.0, 2.0, 2.0])
        g, _ = create_stretched_latlon_grid(
            d, n_lon=6, lat_south=-17.0, lon_west=0.0, lon_east=30.0,
            periodic_x=True, dtype=jnp.float64)
        prev = get_policy()
        set_policy(PrecisionPolicy.fp64())
        try:
            geom = ensure_geometry(g)
        finally:
            set_policy(prev)
        np.testing.assert_allclose(
            np.asarray(geom.dy_T)[:, 0], np.asarray(g.dy) / 2.0,
            rtol=1e-13)

    def test_mercator_geometry_now_exact(self):
        """With faces routed through, the Mercator geometry's per-row
        heights match the grid's analytic faces exactly (previously
        centre-midpoint approximate)."""
        from legoesm.core.precision import (
            PrecisionPolicy, get_policy, set_policy)
        from legoesm.grids.latlon import ensure_geometry
        gm = create_mercator_grid(
            n_lon=90, lat_max_deg=60.0, dtype=jnp.float64)
        prev = get_policy()
        set_policy(PrecisionPolicy.fp64())
        try:
            geom = ensure_geometry(gm)
        finally:
            set_policy(prev)
        np.testing.assert_allclose(
            np.asarray(geom.dy_T)[:, 0], np.asarray(gm.dy) / 2.0,
            rtol=1e-13)
        np.testing.assert_allclose(
            np.asarray(geom.area_T), np.asarray(gm.area), rtol=1e-13)


# =====================================================================
# (f) 5-step PE smoke on a small stretched channel
# =====================================================================


@pytest.fixture()
def _fp64_policy():
    from legoesm.core.precision import (
        PrecisionPolicy, get_policy, set_policy)
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _stretched_channel(**cfg_kw):
    """Closed-wall (N/S) periodic-x channel on a Vinokur-stretched grid
    with a meridional T gradient driving a geostrophic adjustment —
    the ``test_dt_mom_async`` basin pattern on a stretched grid."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    d = _veros_vinokur_two_sided(16, 40.0, 1.0)   # 16 rows over 40°,
    grid, wall_mask = create_stretched_latlon_grid(  # ~1.2° eq, ~4.5° ends
        d, n_lon=8, lat_south=-20.0 - d[0] / 2.0,
        lon_west=0.0, lon_east=40.0, periodic_x=True)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    Hb = jnp.full((grid.n_lat, grid.n_lon), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(wall_mask),
        H_bathy_override=Hb)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        enable_runtime_checks=False, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _tracer_content(model, state, field):
    """Σ_ijk area·h_k·X·land_mask — the flux-form conserved total."""
    from legoesm.ocean.vertical import compute_layer_thickness
    h_k = np.asarray(compute_layer_thickness(
        state.eta.data, state.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m))
    area = np.asarray(model.grid.area)[:, :, None]
    lm = np.asarray(state.land_mask.data)[:, :, None]
    return float(np.sum(np.asarray(field) * h_k * area * lm,
                        dtype=np.float64))


class TestStretchedPESmoke:
    def test_free_surface_5_steps_finite(self, _fp64_policy):
        state, model = _stretched_channel()
        final, _ = model.integrate_scan(state, n_steps=5, dt=1800.0)
        for f in (final.T.data, final.S.data, final.u.data,
                  final.v.data, final.eta.data):
            assert np.all(np.isfinite(np.asarray(f)))
        # the T front actually drives flow (test is not vacuously static)
        assert float(np.max(np.abs(np.asarray(final.u.data)))) > 0.0

    def test_rigid_lid_5_steps_conserves_tracer(self, _fp64_policy):
        """The integration check for the scoping's 'operators are
        variable-dy safe' claim: rigid lid + closed walls + no forcing
        ⇒ heat/salt totals conserved to round-off on the STRETCHED
        grid.  A uniform-dlat assumption anywhere in the advection /
        continuity / mixing stack would break this."""
        state, model = _stretched_channel(
            barotropic_solver="rigid_lid", outer_integrator="ab2")
        heat0 = _tracer_content(model, state, state.T.data)
        salt0 = _tracer_content(model, state, state.S.data)
        final, _ = model.integrate_scan(state, n_steps=5, dt=1800.0)
        assert np.all(np.isfinite(np.asarray(final.T.data)))
        assert float(np.max(np.abs(np.asarray(final.eta.data)))) == 0.0
        heat1 = _tracer_content(model, final, final.T.data)
        salt1 = _tracer_content(model, final, final.S.data)
        rel_heat = abs(heat1 - heat0) / abs(heat0)
        rel_salt = abs(salt1 - salt0) / max(abs(salt0), 1e-30)
        assert rel_heat < 1e-11, (
            f"stretched-grid heat not conserved: rel drift {rel_heat:.3e}")
        assert rel_salt < 1e-11, (
            f"stretched-grid salt not conserved: rel drift {rel_salt:.3e}")
