"""Dry-to-wet river-runoff routing.

The property that matters is MASS: ``sum(runoff * area)`` must be unchanged by
routing, on a grid whose cell areas vary by two orders of magnitude between
the equator and the pole. A test that only checked "the dry cell is now zero"
would pass while silently rescaling the discharge, so every test here budgets
the integral.

The other failure mode is a routing plan that looks right and moves water to
the wrong place: across the dateline, over the pole, or beyond the radius. So
the geometry is checked on cases with a known answer rather than on norms.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.ocean.forcing.runoff_mapper import (  # noqa: E402
    VALID_RUNOFF_SCHEMES,
    apply_runoff_map,
    build_runoff_map,
)


def _grid(n_lat=6, n_lon=12):
    """Cell centres + areas from the model's OWN lat-lon grid.

    Using ``grid.area`` rather than a re-derived formula keeps the test
    budgeting the same areas the driver routes with."""
    g = create_latlon_grid(n_lat, n_lon=n_lon)
    return (np.asarray(g.lat2d), np.asarray(g.lon2d), np.asarray(g.area))


def _discharge(runoff, area):
    return float(np.sum(np.asarray(runoff) * area))


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_unknown_scheme_raises_and_lists_the_valid_ones():
    lat, lon, area = _grid()
    mask = np.ones(area.shape, dtype=bool)
    with pytest.raises(ValueError, match="Unknown runoff routing scheme"):
        build_runoff_map(mask, area, lat, lon, scheme="nearset")


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_nonpositive_or_nonfinite_radius_raises(bad):
    lat, lon, area = _grid()
    mask = np.ones(area.shape, dtype=bool)
    with pytest.raises(ValueError, match="radius_m"):
        build_runoff_map(mask, area, lat, lon, radius_m=bad)


def test_all_land_mask_raises():
    lat, lon, area = _grid()
    mask = np.zeros(area.shape, dtype=bool)
    with pytest.raises(ValueError, match="no wet cells"):
        build_runoff_map(mask, area, lat, lon)


def test_valid_schemes_tuple_is_the_single_source_of_truth():
    assert VALID_RUNOFF_SCHEMES == ("nearest", "spread")


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", VALID_RUNOFF_SCHEMES)
def test_routing_conserves_total_discharge(scheme):
    """The whole point: moving a flux DENSITY between cells of different area
    must preserve the integral, not the density."""
    lat, lon, area = _grid()
    mask = np.ones(area.shape, dtype=bool)
    mask[:, 3:6] = False                      # a land block
    runoff = np.zeros(area.shape)
    runoff[2, 4] = 3.0e-4                     # a river inside the land block
    runoff[1, 4] = 1.0e-4
    runoff[0, 0] = 5.0e-5                     # already wet: must be untouched

    rmap = build_runoff_map(mask, area, lat, lon, scheme=scheme,
                            radius_m=1.2e7, reference_runoff=runoff)
    assert rmap.unrouted_fraction == pytest.approx(0.0, abs=1e-12)
    out = np.asarray(apply_runoff_map(runoff, rmap))

    assert _discharge(out, area) == pytest.approx(_discharge(runoff, area),
                                                  rel=1e-12)
    assert np.all(out[~mask] == 0.0), "dry cells must be emptied"
    assert np.all(out >= 0.0)


def test_nearest_applies_the_exact_area_ratio_not_a_copy():
    """Force the ONLY recipient into a different latitude row and check the
    delivered density is exactly ``F_src * A_src / A_dst``.

    A no-op copy (``F_dst = F_src``) conserves mass only when the two cells
    have equal area, which is why the recipient must not be a same-row
    neighbour: on a lat-lon grid those areas are identical and the bug would
    hide. Here the areas differ by construction.
    """
    lat, lon, area = _grid(n_lat=6, n_lon=12)
    mask = np.zeros(area.shape, dtype=bool)
    mask[3, 6] = True                          # the ONLY wet cell, mid-lat
    src = (0, 6)                               # donor in the polar row
    runoff = np.zeros(area.shape)
    runoff[src] = 1.0e-3

    a_src, a_dst = area[src], area[3, 6]
    assert a_dst / a_src > 2.0, "test premise: the two areas must differ"

    rmap = build_runoff_map(mask, area, lat, lon, scheme="nearest",
                            radius_m=2.0e7)
    out = np.asarray(apply_runoff_map(runoff, rmap))
    assert out[3, 6] == pytest.approx(runoff[src] * a_src / a_dst, rel=1e-12)
    assert out[3, 6] != pytest.approx(runoff[src], rel=1e-6)
    assert float(np.sum(out * area)) == pytest.approx(runoff[src] * a_src,
                                                      rel=1e-12)


@pytest.mark.parametrize("scheme", VALID_RUNOFF_SCHEMES)
def test_moved_discharge_equals_the_donors_whatever_the_recipients(scheme):
    lat, lon, area = _grid(n_lat=4, n_lon=4)
    mask = np.ones(area.shape, dtype=bool)
    mask[0, 0] = False                        # donor: polar row, small area
    runoff = np.zeros(area.shape)
    runoff[0, 0] = 1.0e-3

    rmap = build_runoff_map(mask, area, lat, lon, scheme=scheme,
                            radius_m=1.2e7)
    out = np.asarray(apply_runoff_map(runoff, rmap))
    assert (out > 0).any()
    assert float(np.sum(out * area)) == pytest.approx(
        runoff[0, 0] * area[0, 0], rel=1e-12)
    assert out[0, 0] == 0.0


def test_wet_cell_runoff_is_preserved_exactly():
    lat, lon, area = _grid()
    mask = np.ones(area.shape, dtype=bool)
    mask[3, 3] = False
    runoff = np.full(area.shape, 2.0e-5)
    rmap = build_runoff_map(mask, area, lat, lon, radius_m=1.2e7)
    out = np.asarray(apply_runoff_map(runoff, rmap))
    # A wet cell that receives nothing keeps its own value bit-for-bit.
    far = (0, 9)
    assert mask[far]
    assert out[far] == runoff[far]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_nearest_recipient_is_chosen_across_the_dateline():
    """A donor at lon ~0 must reach a wet cell at lon ~360, not travel the
    long way round. Chord distance on the unit sphere makes this automatic —
    the test pins that it really is being used."""
    lat, lon, area = _grid(n_lat=3, n_lon=8)
    mask = np.zeros(area.shape, dtype=bool)
    mask[1, 7] = True                          # the ONLY wet cell, just west
    runoff = np.zeros(area.shape)
    runoff[1, 0] = 1.0e-3                      # donor at the seam
    rmap = build_runoff_map(mask, area, lat, lon, radius_m=1.0e7)
    out = np.asarray(apply_runoff_map(runoff, rmap))
    assert out[1, 7] > 0.0
    assert float(np.sum(out * area)) == pytest.approx(
        runoff[1, 0] * area[1, 0], rel=1e-12)


def test_radius_limits_routing_and_the_shortfall_is_reported():
    """Discharge with no wet cell in range must NOT be teleported, and the
    builder must say how much it failed to place rather than hide it.

    The donor/recipient pair is deliberately NOT antipodal: an exactly
    opposite pair sits at chord 2.0, the boundary of the whole-sphere search,
    where inclusion depends on the last bit of the grid metrics (float32 grid
    coordinates put it outside, float64 inside). See the separate
    whole-sphere test below for that case.
    """
    lat, lon, area = _grid(n_lat=6, n_lon=12)
    mask = np.zeros(area.shape, dtype=bool)
    mask[0, 0] = True                          # one wet cell at the far south
    runoff = np.zeros(area.shape)
    runoff[5, 3] = 1.0e-3                      # donor far away but NOT opposite

    tight = build_runoff_map(mask, area, lat, lon, radius_m=1.0e5,
                             reference_runoff=runoff)
    assert tight.unrouted_fraction == pytest.approx(1.0, rel=1e-12)
    out = np.asarray(apply_runoff_map(runoff, tight))
    assert float(np.sum(out * area)) == pytest.approx(0.0, abs=1e-30)

    wide = build_runoff_map(mask, area, lat, lon, radius_m=4.0e7,
                            reference_runoff=runoff)
    assert wide.unrouted_fraction == pytest.approx(0.0, abs=1e-12)
    out_w = np.asarray(apply_runoff_map(runoff, wide))
    assert float(np.sum(out_w * area)) == pytest.approx(
        runoff[5, 3] * area[5, 3], rel=1e-6)


def test_whole_sphere_radius_reaches_even_the_antipode():
    """A radius at or past the antipode means "everywhere", and must say so
    regardless of the grid metrics' precision -- an exactly opposite cell is
    at chord 2.0, the boundary of a bounded search."""
    lat, lon, area = _grid(n_lat=6, n_lon=12)
    mask = np.zeros(area.shape, dtype=bool)
    mask[0, 0] = True
    runoff = np.zeros(area.shape)
    runoff[5, 6] = 1.0e-3                      # the antipodal cell
    rmap = build_runoff_map(mask, area, lat, lon, radius_m=4.0e7,
                            reference_runoff=runoff)
    assert rmap.unrouted_fraction == pytest.approx(0.0, abs=1e-12)
    out = np.asarray(apply_runoff_map(runoff, rmap))
    assert float(np.sum(out * area)) == pytest.approx(
        runoff[5, 6] * area[5, 6], rel=1e-6)


def test_spread_reaches_more_recipients_than_nearest():
    lat, lon, area = _grid(n_lat=6, n_lon=12)
    mask = np.ones(area.shape, dtype=bool)
    mask[3, 6] = False
    runoff = np.zeros(area.shape)
    runoff[3, 6] = 1.0e-3
    r_near = build_runoff_map(mask, area, lat, lon, scheme="nearest",
                              radius_m=6.0e6)
    r_spread = build_runoff_map(mask, area, lat, lon, scheme="spread",
                                radius_m=6.0e6)
    n_near = int(np.sum(np.asarray(apply_runoff_map(runoff, r_near)) > 0))
    n_spread = int(np.sum(np.asarray(apply_runoff_map(runoff, r_spread)) > 0))
    assert n_near == 1
    assert n_spread > 1


def test_radius_default_matches_fesom_runoff_radius():
    """FESOM2's namelist.forcing ships runoff_radius = 500 km."""
    import inspect

    sig = inspect.signature(build_runoff_map)
    assert sig.parameters["radius_m"].default == 500.0e3
    assert sig.parameters["scheme"].default == "nearest"


# ---------------------------------------------------------------------------
# jit / AD / shape contract
# ---------------------------------------------------------------------------

def test_apply_is_jit_and_grad_safe():
    lat, lon, area = _grid()
    mask = np.ones(area.shape, dtype=bool)
    mask[2, 2] = False
    rmap = build_runoff_map(mask, area, lat, lon, radius_m=1.2e7)
    runoff = np.full(area.shape, 1.0e-5)

    eager = np.asarray(apply_runoff_map(runoff, rmap))
    jitted = np.asarray(jax.jit(apply_runoff_map, static_argnums=())(
        jax.numpy.asarray(runoff), rmap))
    np.testing.assert_allclose(jitted, eager, rtol=0.0, atol=0.0)

    def loss(r):
        return jax.numpy.sum(apply_runoff_map(r, rmap) ** 2)

    g = np.asarray(jax.grad(loss)(jax.numpy.asarray(runoff)))
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


def test_shape_mismatch_raises_rather_than_broadcasting():
    lat, lon, area = _grid(n_lat=4, n_lon=4)
    mask = np.ones(area.shape, dtype=bool)
    mask[0, 0] = False
    rmap = build_runoff_map(mask, area, lat, lon, radius_m=1.2e7)
    with pytest.raises(ValueError, match="Rebuild the map"):
        apply_runoff_map(np.zeros((6, 12)), rmap)


def test_unstructured_1d_grid_is_supported():
    """The flat-index design must serve MPAS ``(nCells,)`` too."""
    n = 40
    rng = np.random.default_rng(0)
    lat = np.arcsin(rng.uniform(-1.0, 1.0, n))
    lon = rng.uniform(0.0, 2.0 * np.pi, n)
    area = np.full(n, 4.0 * np.pi * constants.R_earth ** 2 / n)
    mask = np.ones(n, dtype=bool)
    mask[:5] = False
    runoff = np.zeros(n)
    runoff[:5] = 1.0e-4
    rmap = build_runoff_map(mask, area, lat, lon, radius_m=2.0e7)
    out = np.asarray(apply_runoff_map(runoff, rmap))
    assert out.shape == (n,)
    assert np.all(out[:5] == 0.0)
    assert float(np.sum(out * area)) == pytest.approx(
        float(np.sum(runoff * area)), rel=1e-12)
