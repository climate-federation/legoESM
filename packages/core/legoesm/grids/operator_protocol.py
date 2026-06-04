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


def _validate_operators(grid: Any, required: tuple[tuple[str, int], ...]) -> None:
    """Raise ``TypeError`` unless *grid* exposes every ``(name, arity)`` operator."""
    for method_name, n_args in required:
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


def validate_grid_operators(grid: Any) -> None:
    """Raise ``TypeError`` unless *grid* implements every operator with its arity.

    Stronger than ``isinstance(grid, GridOperators)`` — a ``runtime_checkable``
    Protocol only proves attribute presence (it would pass ``divergence = 1``).
    Stage B2's grid registry uses this to validate an adapter-wrapped grid.
    """
    _validate_operators(grid, _REQUIRED_OPERATORS)


# --- Edge-normal (TRiSK) operator interface -------------------------------
#
# MPAS/Voronoi meshes carry a SINGLE edge-normal velocity ``u_edge`` rather than
# a component pair ``(u, v)``, so the component-velocity ``GridOperators``
# contract (``divergence(u, v)``) does not fit them.  ``EdgeOperators`` is the
# sibling contract for edge-normal grids: the differential operators
# ``divergence(u_edge)`` -> cell, ``gradient(phi_cell)`` -> edge,
# ``vorticity(u_edge)`` -> vertex, plus the TRiSK ``tangential`` reconstruction and
# ``cell_to_edge`` remap (grouped here as ``GridOperators`` groups ``interpolate``
# alongside its derivatives).  Spectral and SFNO grids operate through global
# transforms / neural maps, not grid-local stencils, so they expose NEITHER
# contract and share their grid through the dynamical-core modules directly.


@runtime_checkable
class EdgeOperators(Protocol):
    """Grid-dispatched TRiSK operators for an edge-normal velocity ``u_edge``."""

    def divergence(self, u_edge: Any) -> Any:
        """Divergence of edge-normal velocity -> cell-centre scalar."""
        ...

    def gradient(self, phi_cell: Any) -> Any:
        """Gradient of a cell scalar -> edge-normal component."""
        ...

    def vorticity(self, u_edge: Any) -> Any:
        """Relative vorticity (curl) of ``u_edge`` -> vertex scalar."""
        ...

    def tangential(self, u_edge: Any) -> Any:
        """Reconstruct the edge-tangential velocity from ``u_edge`` (TRiSK)."""
        ...

    def cell_to_edge(self, phi_cell: Any) -> Any:
        """Average a cell-centre scalar onto edges."""
        ...


_REQUIRED_EDGE_OPERATORS: tuple[tuple[str, int], ...] = (
    ("divergence", 1),
    ("gradient", 1),
    ("vorticity", 1),
    ("tangential", 1),
    ("cell_to_edge", 1),
)


def validate_edge_operators(grid: Any) -> None:
    """Raise ``TypeError`` unless *grid* implements every edge-normal operator."""
    _validate_operators(grid, _REQUIRED_EDGE_OPERATORS)
