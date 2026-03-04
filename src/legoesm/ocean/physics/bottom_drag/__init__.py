"""Bottom drag parameterizations for the ocean."""

from legoesm.ocean.physics.bottom_drag.config import (
    BottomDragConfig,
    LinearDragConfig,
    QuadraticDragConfig,
)
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput
from legoesm.ocean.physics.bottom_drag.integration import (
    make_bottom_drag_physics,
)

__all__ = [
    "BottomDragConfig",
    "LinearDragConfig",
    "QuadraticDragConfig",
    "BottomDragOutput",
    "make_bottom_drag_physics",
]
