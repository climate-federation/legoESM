"""Unit tests for ``scripts/validate/ocean_fidelity/fesom_pseudo_arm.py``.

``main`` does live I/O (a FESOM snapshot and a NEMO grid_T) and is not
exercised here.  What is exercised is ``idw_to_points``, because the whole
probe rests on one claim about it: that it applies the SAME weighting as
``compare_omip_nemo.regrid_curv_to_latlon`` and therefore may be used for the
leg that function cannot express, interpolation onto a point cloud.  A second
implementation of an interpolation is exactly the kind of thing that drifts
from its sibling silently and produces a confident wrong number, so the
equivalence is pinned here as well as gated at run time.

``test_k_changes_the_answer`` is the non-vacuity partner of the equivalence
test: it shows the weighting is live on this fixture, so agreement between the
two routines is a real coincidence of arithmetic rather than both collapsing
onto the nearest-neighbour value.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_VALIDATE = Path(__file__).resolve().parents[2] / "scripts" / "validate"
_MOD = _VALIDATE / "ocean_fidelity" / "fesom_pseudo_arm.py"


def _load():
    sys.path.insert(0, str(_VALIDATE))
    spec = importlib.util.spec_from_file_location("fesom_pseudo_arm", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()


def _source():
    """A coarse global source grid carrying a smooth, non-degenerate field."""
    slat = np.linspace(-80.0, 80.0, 33)
    slon = np.linspace(0.0, 350.0, 36)
    lon2d, lat2d = np.meshgrid(slon, slat)
    field = (np.sin(np.deg2rad(lat2d)) * 3.0
             + np.cos(np.deg2rad(lon2d)) * 2.0 + 34.0)
    return lat2d, lon2d, field, np.ones_like(field)


def test_matches_the_grid_regridder_on_its_own_targets():
    from compare_omip_nemo import regrid_curv_to_latlon

    lat2d, lon2d, field, mask = _source()
    tgt_lat = np.linspace(-70.0, 70.0, 15)
    tgt_lon = np.linspace(5.0, 355.0, 24)
    ref, ref_cov = regrid_curv_to_latlon(field, lat2d, lon2d, mask,
                                         tgt_lat, tgt_lon)
    tlon2d, tlat2d = np.meshgrid(tgt_lon, tgt_lat)
    got, got_cov = m.idw_to_points(field, lat2d, lon2d, mask,
                                   tlat2d.ravel(), tlon2d.ravel())
    assert np.max(np.abs(got.reshape(ref.shape) - ref)) < 1e-12
    assert np.array_equal(got_cov.reshape(ref_cov.shape) > 0.5, ref_cov > 0.5)


def test_k_changes_the_answer():
    """Non-vacuity: the stencil is live, so the equivalence above is real."""
    lat2d, lon2d, field, mask = _source()
    tlat = np.array([12.0, -35.0, 61.0])
    tlon = np.array([77.0, 200.0, 311.0])
    a, _ = m.idw_to_points(field, lat2d, lon2d, mask, tlat, tlon, k=1)
    b, _ = m.idw_to_points(field, lat2d, lon2d, mask, tlat, tlon, k=4)
    assert np.max(np.abs(a - b)) > 1e-6


def test_constant_field_is_reproduced_exactly():
    """Any convex weighting must return the constant; a normalisation bug
    would not."""
    lat2d, lon2d, _, mask = _source()
    const = np.full(lat2d.shape, 34.7)
    got, _ = m.idw_to_points(const, lat2d, lon2d, mask,
                             np.array([0.0, 44.0]), np.array([17.0, 260.0]))
    assert np.allclose(got, 34.7, atol=1e-12)


def test_coverage_flag_marks_points_far_from_any_wet_cell():
    lat2d, lon2d, field, mask = _source()
    wet = mask.copy()
    wet[:, :] = 0.0
    wet[0, 0] = 1.0                      # a single wet source cell
    near_lat, near_lon = lat2d[0, 0], lon2d[0, 0]
    _, cov = m.idw_to_points(field, lat2d, lon2d, wet,
                             np.array([near_lat, 0.0]),
                             np.array([near_lon, 180.0]), k=1)
    assert cov[0] > 0.5 and cov[1] < 0.5


def test_unpaired_target_arrays_raise():
    lat2d, lon2d, field, mask = _source()
    with pytest.raises(ValueError, match="paired"):
        m.idw_to_points(field, lat2d, lon2d, mask,
                        np.array([0.0, 1.0]), np.array([0.0]))


def test_dry_source_raises():
    lat2d, lon2d, field, _ = _source()
    with pytest.raises(ValueError, match="no ocean source"):
        m.idw_to_points(field, lat2d, lon2d, np.zeros_like(field),
                        np.array([0.0]), np.array([0.0]))


def test_wrms_is_area_weighted_not_a_cell_count():
    """The retraction this campaign earned (3558d23d2) came from weighting a
    sum of squares by counts; pin that this helper does not."""
    err = np.array([[1.0, 3.0]])
    area = np.array([[9.0, 1.0]])
    mask = np.ones_like(err, dtype=bool)
    assert m.wrms(err, area, mask) == pytest.approx(np.sqrt(1.8))
    assert m.wrms(err, np.ones_like(area), mask) == pytest.approx(np.sqrt(5.0))
