"""Land model state container."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.core.field import Field


class LandState(NamedTuple):
    """Slab land state.

    All fields have shape (6, n, n).
    """
    T_soil: Field          # Soil slab temperature [K]
    W_bucket: Field        # Bucket soil moisture [kg/m2]
