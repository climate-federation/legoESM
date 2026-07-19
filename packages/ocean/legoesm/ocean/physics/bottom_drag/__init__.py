"""Bottom-drag CONFIG for the ocean.

The physics-level bottom-drag implementation (``make_bottom_drag_physics`` and
its linear/quadratic kernels) was DELETED: it was dead, and wiring it would have
been a bug.

Bottom drag has a single owner, the DYNAMICS level: the 3D PE solvers apply the
full ``-r*u_bot/h_bot`` (with BBL / partial-cell handling) and carry its
depth-mean into the barotropic mode via ``F_slow_u``/``F_slow_v`` at every
substep (``ocean_tendency_common.implicit_bottom_drag_factor``, the MOM6
convention). ``mpas_physics`` already RAISED on any physics-level scheme other
than "none", every in-tree experiment set ``BottomDragConfig(scheme="none")``
with a "use model-level drag" comment, and no production caller ever reached the
factory -- so its only possible effect was to DOUBLE-COUNT drag against that
single-owner invariant.

``BottomDragConfig`` stays: it is how a config states "physics-level drag off",
and it is what the deprecation guard inspects to reject anything else loudly.
"""

from legoesm.ocean.physics.bottom_drag.config import (
    BottomDragConfig,
    LinearDragConfig,
    QuadraticDragConfig,
)

__all__ = [
    "BottomDragConfig",
    "LinearDragConfig",
    "QuadraticDragConfig",
]
