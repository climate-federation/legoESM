"""Lake model state container."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.core.field import Field


class LakeState(NamedTuple):
    """Two-layer lake state.

    All fields have shape (6, n, n).
    """
    T_epi: Field         # Epilimnion temperature [K]
    T_hypo: Field        # Hypolimnion temperature [K]
