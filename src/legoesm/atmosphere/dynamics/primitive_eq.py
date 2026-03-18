"""Backward-compatible alias for primitive equation hydrostatic model.

This module re-exports classes from primitive_eq_cdgrid for backward
compatibility with code that imports from the old module name.
"""

from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel as PrimitiveEquationModel,
    CDGridPrimitiveEquationConfig as PrimitiveEquationConfig,
    cdgrid_hydrostatic_tendencies as hydrostatic_tendencies,
)
from legoesm.core.operators_3d import (
    vorticity_3d as _vorticity_3d,
    divergence_3d as _divergence_3d,
    gradient_x_3d as _gradient_x_3d,
)

__all__ = [
    "PrimitiveEquationModel",
    "PrimitiveEquationConfig",
    "hydrostatic_tendencies",
    "_vorticity_3d",
    "_divergence_3d",
    "_gradient_x_3d",
]
