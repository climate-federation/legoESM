"""Unit tests for the per-archetype observed-SOC loader (Stage-B B1).

Pure NumPy (no JAX / model step); the loader is a fixed calibration target.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from legoesm.land.carbon.soc_observations import (
    column_soc,
    per_archetype_cover_weight,
    per_archetype_observed_soc,
)


def test_column_soc_known_integral():
    # 1 cell, 3 layers: organic [2,4,1] kgC/m3, dz [0.1,0.2,0.5] m.
    # col = 2*0.1 + 4*0.2 + 1*0.5 = 0.2 + 0.8 + 0.5 = 1.5 kgC/m2.
    organic = np.array([[2.0, 4.0, 1.0]])
    dz = np.array([0.1, 0.2, 0.5])
    npt.assert_allclose(column_soc(organic, dz), [1.5], rtol=1e-12)


def test_column_soc_supports_per_cell_dz():
    # dz may be (ncell, n_layer): distinct thickness per cell.
    organic = np.array([[2.0], [2.0]])
    dz = np.array([[0.5], [1.5]])
    npt.assert_allclose(column_soc(organic, dz), [1.0, 3.0], rtol=1e-12)


def test_column_soc_layer_axis_mismatch_raises():
    with pytest.raises(ValueError):
        column_soc(np.ones((1, 3)), np.ones(2))


def test_per_archetype_cover_weighted_mean():
    # 2 cells, 2 PFTs, 2 layers.
    #   cell0 column SOC = [1,2].[0.5,0.5] = 0.5 + 1.0 = 1.5 kgC/m2
    #   cell1 column SOC = [3,4].[0.5,0.5] = 1.5 + 2.0 = 3.5 kgC/m2
    organic = np.array([[1.0, 2.0], [3.0, 4.0]])
    dz = np.array([0.5, 0.5])
    # Membership: archetype 0 = {(cell0,pft0) w=0.75, (cell1,pft1) w=0.25};
    #             archetype 1 = {(cell1,pft0) w=1.0}.
    cid = np.array([[0, -1], [1, 0]])
    cw = np.array([[0.75, 0.0], [1.0, 0.25]])
    out = per_archetype_observed_soc(organic, dz, cid, cw, n_arch=2)
    # arch0 = (0.75*1.5 + 0.25*3.5) / (0.75 + 0.25) = (1.125 + 0.875)/1.0 = 2.0
    # arch1 = 3.5 (single member, full cover).
    npt.assert_allclose(out, [2.0, 3.5], rtol=1e-12)


def test_per_archetype_infers_n_arch():
    organic = np.array([[10.0]])
    dz = np.array([1.0])
    cid = np.array([[0, 2]])           # highest id = 2 -> n_arch = 3
    cw = np.array([[0.5, 0.5]])
    out = per_archetype_observed_soc(organic, dz, cid, cw)
    assert out.shape == (3,)
    npt.assert_allclose(out[0], 10.0, rtol=1e-12)   # arch 0 <- cell0 (soc 10)
    assert np.isnan(out[1])                          # arch 1 has no members
    npt.assert_allclose(out[2], 10.0, rtol=1e-12)   # arch 2 <- cell0 (soc 10)


def test_zero_cover_archetype_is_nan_not_zero():
    organic = np.array([[1.0]])
    dz = np.array([1.0])
    cid = np.array([[0, -1]])
    cw = np.array([[1.0, 0.0]])
    out = per_archetype_observed_soc(organic, dz, cid, cw, n_arch=2)
    npt.assert_allclose(out[0], 1.0, rtol=1e-12)
    assert np.isnan(out[1])            # archetype 1 unassigned -> NaN, never 0


def test_cover_weight_sums_assigned_cover():
    # Same membership as test_per_archetype_cover_weighted_mean.
    #   archetype 0 <- (cell0,pft0) 0.75 + (cell1,pft1) 0.25 = 1.0
    #   archetype 1 <- (cell1,pft0) 1.0                       = 1.0
    cid = np.array([[0, -1], [1, 0]])
    cw = np.array([[0.75, 0.0], [1.0, 0.25]])
    out = per_archetype_cover_weight(cid, cw, n_arch=2)
    npt.assert_allclose(out, [1.0, 1.0], rtol=1e-12)


def test_cover_weight_unassigned_is_zero():
    cid = np.array([[0, -1]])
    cw = np.array([[1.0, 0.0]])
    out = per_archetype_cover_weight(cid, cw, n_arch=3)
    npt.assert_allclose(out, [1.0, 0.0, 0.0], rtol=1e-12)


def test_cover_weight_matches_observed_denominator():
    # per_archetype_observed_soc uses per_archetype_cover_weight as its
    # denominator: an archetype with zero cover weight -> NaN observed SOC.
    organic = np.array([[1.0], [3.0]])
    dz = np.array([1.0])
    cid = np.array([[0, -1], [1, -1]])
    cw = np.array([[2.0, 0.0], [0.0, 0.0]])
    weight = per_archetype_cover_weight(cid, cw, n_arch=2)
    soc = per_archetype_observed_soc(organic, dz, cid, cw, n_arch=2)
    npt.assert_allclose(weight, [2.0, 0.0], rtol=1e-12)
    npt.assert_allclose(soc[0], 1.0, rtol=1e-12)   # cover>0 -> finite
    assert np.isnan(soc[1])                         # cover==0 -> NaN


def test_shape_mismatch_raises():
    organic = np.array([[1.0, 2.0]])
    dz = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        per_archetype_observed_soc(
            organic, dz, np.array([[0]]), np.array([[1.0, 0.0]]))
