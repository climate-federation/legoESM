"""Backward-compatible alias for ocean primitive equation model.

This module re-exports classes and functions from ocean_pe_cdgrid for backward
compatibility with code that imports from the old module name.
"""

from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
    ocean_baroclinic_tendencies_cdgrid as ocean_baroclinic_tendencies,
    _vertical_advection_ocean,
)

__all__ = [
    "ocean_baroclinic_tendencies",
    "_vertical_advection_ocean",
]
