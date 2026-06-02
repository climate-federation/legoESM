"""The grid operator interface — the Stage-B2 ``step(grid, state)`` target (A2.3).

Design L2: operators *dispatch on grid type*.  One dynamical core, specialised
per grid, so the core becomes ``step(grid, state)`` with zero per-dycore
branching.  This module is the *contract* for that — the operator methods every
grid will expose (cubed-sphere -> FC-Gram + Duo-Grid halo; lat-lon -> FV;
Voronoi/MPAS -> TRiSK), expressed as ``methods-on-pytree``.

Existing grids are ``NamedTuple`` pytrees and operators are currently *free
functions* (``divergence_3d(u, v, grid)``); Stage B2 routes those through these
grid methods (adapter-wrapping the concrete grids), at which point the dispatch
tables collapse to discovery.  So no concrete grid satisfies ``GridOperators``
yet — this is the forward target, validated here with a toy grid.

**Static/traced contract (D5 — the single most important discipline).**  A grid's
*structure* — per-axis topology ``(Periodic|Bounded|Flat)``, shape ``(Nx,Ny,Nz)``,
halo widths, stagger convention, the architecture handle, sharding spec — is
**static** (hashable ``aux_data``, no arrays).  Its *numerics* — node coords,
spacings, metric terms/Jacobians, area/volume weights — are **traced** (jnp-array
children).  Get this wrong and ``jit`` recompiles forever, silently degrading
agility; a CI test must assert zero recompiles across changes that should be
dynamic.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GridOperators(Protocol):
    """The grid-dispatched differential + remap operators (design L2).

    Each grid provides the implementation; the dynamical core calls these without
    knowing which grid it holds.  Signatures mirror the current free-function
    operators so Stage B2 can wrap them: ``divergence(u, v)``, ``gradient(s)``,
    ``vorticity(u, v)``, ``interpolate(field, src, dst)``, ``halo_fill(field)``.
    """

    def divergence(self, u: Any, v: Any) -> Any:
        """Horizontal divergence of a vector field ``(u, v)`` -> scalar."""
        ...

    def gradient(self, scalar: Any) -> Any:
        """Horizontal gradient of a scalar -> vector ``(d/dx, d/dy)``."""
        ...

    def vorticity(self, u: Any, v: Any) -> Any:
        """Relative vorticity (curl) of ``(u, v)`` -> scalar."""
        ...

    def interpolate(self, field: Any, src_location: Any, dst_location: Any) -> Any:
        """Stagger interpolation/averaging between grid locations (Center/Face)."""
        ...

    def halo_fill(self, field: Any) -> Any:
        """Fill the halo/ghost cells of a CELL-CENTRE field (cubed-sphere
        cross-panel; lat-lon pole/wrap; ...).

        ``halo_fill`` is the cell-centre operator-contract halo: a single field
        argument cannot disambiguate stagger, so the *staggered* (face / corner /
        edge) halos a dycore needs are filled by stagger-specific routines inside
        the core, not through this generic seam.  An adapter therefore documents
        ``halo_fill`` as cell-centre-only.
        """
        ...


_REQUIRED_OPERATORS: tuple[tuple[str, int], ...] = (
    ("divergence", 2),
    ("gradient", 1),
    ("vorticity", 2),
    ("interpolate", 3),
    ("halo_fill", 1),
)


def validate_grid_operators(grid: Any) -> None:
    """Raise ``TypeError`` unless *grid* implements every operator with its arity.

    Stronger than ``isinstance(grid, GridOperators)`` — a ``runtime_checkable``
    Protocol only proves attribute presence (it would pass ``divergence = 1``).
    Stage B2's grid registry uses this to validate an adapter-wrapped grid.
    """
    for method_name, n_args in _REQUIRED_OPERATORS:
        fn = getattr(grid, method_name, None)
        if not callable(fn):
            raise TypeError(
                f"grid {grid!r} does not expose a callable {method_name!r} operator"
            )
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"grid.{method_name} signature could not be inspected ({exc})"
            ) from exc
        try:
            sig.bind(*([object()] * n_args))
        except TypeError as exc:
            raise TypeError(
                f"grid.{method_name} must accept {n_args} positional argument(s) "
                f"(signature {sig}): {exc}"
            ) from exc
