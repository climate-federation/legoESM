"""FV3_3D iter 902: construction-time nord validation in PE and NH models.

Iter 890 added validate_corner_div_damp_nord(); iter 902 wires it into
both CDGridPrimitiveEquationModel.__init__ and
CDGridCompressibleEulerModel.__init__ so misuse is rejected at model
construction rather than at first dynamics step.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    compute_terrain_metric,
    create_height_coordinate,
    standard_hybrid_levels,
)


_N = 8
_NLEV = 5


def _pe_args():
    grid = create_cubed_sphere(_N)
    coord = standard_hybrid_levels(_NLEV)
    return grid, coord


def _nh_args():
    grid = create_cubed_sphere(_N)
    hc = create_height_coordinate(_NLEV, 30000.0)
    tm = compute_terrain_metric(jnp.zeros((6, _N, _N)), hc)
    return grid, hc, tm


def test_pe_model_accepts_canonical_nord3():
    """PE: nord=3 (FV3 canonical upper) accepted at __init__."""
    grid, coord = _pe_args()
    cfg = CDGridPrimitiveEquationConfig(corner_div_damp_nord=3)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    assert model.config.corner_div_damp_nord == 3


def test_pe_model_rejects_nord4():
    """PE: nord=4 (outside FV3 namelist range) rejected at __init__."""
    grid, coord = _pe_args()
    cfg = CDGridPrimitiveEquationConfig(corner_div_damp_nord=4)
    with pytest.raises(ValueError, match=r"outside FV3 namelist range"):
        CDGridPrimitiveEquationModel(grid, coord, cfg)


def test_nh_model_accepts_canonical_nord3():
    """NH: nord=3 accepted at __init__."""
    grid, hc, tm = _nh_args()
    cfg = CDGridCompressibleEulerConfig(corner_div_damp_nord=3)
    model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    assert model.config.corner_div_damp_nord == 3


def test_nh_model_rejects_nord4():
    """NH: nord=4 rejected at __init__."""
    grid, hc, tm = _nh_args()
    cfg = CDGridCompressibleEulerConfig(corner_div_damp_nord=4)
    with pytest.raises(ValueError, match=r"outside FV3 namelist range"):
        CDGridCompressibleEulerModel(grid, hc, tm, cfg)


def test_pe_model_rejects_negative_nord():
    """PE: negative nord rejected at __init__."""
    grid, coord = _pe_args()
    cfg = CDGridPrimitiveEquationConfig(corner_div_damp_nord=-1)
    with pytest.raises(ValueError, match=r"outside FV3 namelist range"):
        CDGridPrimitiveEquationModel(grid, coord, cfg)


def test_nh_model_rejects_negative_nord():
    """NH: negative nord rejected at __init__."""
    grid, hc, tm = _nh_args()
    cfg = CDGridCompressibleEulerConfig(corner_div_damp_nord=-1)
    with pytest.raises(ValueError, match=r"outside FV3 namelist range"):
        CDGridCompressibleEulerModel(grid, hc, tm, cfg)
