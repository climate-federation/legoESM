"""Backward-compatible alias for FV (C-D grid) ocean primitive equation model.

This module re-exports functions from ocean_pe_cdgrid for backward
compatibility with code that imports from the old FV module name.

The C-D grid discretization (ocean_pe_cdgrid) is the FV implementation
for the cubed-sphere ocean model.
"""

from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
    ocean_baroclinic_tendencies_cdgrid as ocean_baroclinic_tendencies_fv,
)

__all__ = [
    "ocean_baroclinic_tendencies_fv",
]
