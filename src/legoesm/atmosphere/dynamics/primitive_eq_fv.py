"""Backward-compatible alias for FV (C-D grid) primitive equation model.

This module re-exports classes from primitive_eq_cdgrid for backward
compatibility with code that imports from the old FV module name.

The C-D grid discretization (primitive_eq_cdgrid) is the FV implementation
for the cubed-sphere.
"""

from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel as FVPrimitiveEquationModel,
    CDGridPrimitiveEquationConfig as FVPrimitiveEquationConfig,
)

__all__ = [
    "FVPrimitiveEquationModel",
    "FVPrimitiveEquationConfig",
]
