"""Configuration switch for the (removed) physics-level ocean bottom drag."""

from __future__ import annotations

from typing import NamedTuple


class BottomDragConfig(NamedTuple):
    """Physics-level bottom drag: only ``"none"`` is valid.

    Bottom drag is owned by the dynamics (``DynBottomDragConfig``); any other
    scheme is rejected by the physics builders so drag is never double-counted.
    The former linear/quadratic sub-configs were read by nothing and are gone.
    """
    scheme: str = "none"
