"""Sea ice model state container."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.core.field import Field


class SeaIceState(NamedTuple):
    """Thermodynamic slab sea ice state.

    All fields have shape (6, n, n).
    """
    h_ice: Field               # Ice thickness [m]
    T_ice: Field               # Ice surface temperature [K]
    concentration: Field       # Ice areal fraction [0-1]
