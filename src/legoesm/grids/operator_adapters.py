"""Concrete :class:`~legoesm.grids.operator_protocol.GridOperators` adapters (B2).

The grids are ``NamedTuple`` pytrees and the differential/remap operators are
free functions (``divergence_cgrid(u, v, grid)``).  Stage B2 routes those through
the grid-dispatched ``methods-on-pytree`` interface by ADAPTER-WRAPPING a concrete
grid: the adapter holds the grid and delegates each operator method to the shared
free function, so a real grid satisfies :class:`GridOperators` (validated by
:func:`validate_grid_operators`) with NO change to the grids or operators and
byte-identical numerics.

This makes the operator contract load-bearing: a dynamical core can call
``ops.divergence(u, v)`` without knowing which grid it holds (design L2).  This
module supplies the lat-lon C-grid adapter (over the shared operators in
:mod:`legoesm.grids.operators_latlon_cgrid`); the cubed-sphere (FC-Gram / Duo-Grid)
and MPAS (TRiSK) adapters follow the same pattern.
"""

from __future__ import annotations

from typing import Any

from legoesm.grids.operators_latlon_cgrid import (
    curl_vertex_cgrid,
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
    pad_ns_scalar,
)


class LatLonCGridOperators:
    """:class:`GridOperators` over a lat-lon C-grid, delegating to the shared
    free-function operators (a thin method wrapper — identical numerics)."""

    def __init__(self, grid: Any) -> None:
        self._grid = grid

    @property
    def grid(self) -> Any:
        return self._grid

    def divergence(self, u: Any, v: Any) -> Any:
        """Horizontal C-grid flux divergence of ``(u, v)`` -> cell-centre scalar."""
        return divergence_cgrid(u, v, self._grid)

    def gradient(self, scalar: Any) -> Any:
        """Horizontal gradient of a cell-centre scalar -> ``(d/dx, d/dy)`` on faces."""
        return (
            gradient_x_cgrid(scalar, self._grid),
            gradient_y_cgrid(scalar, self._grid),
        )

    def vorticity(self, u: Any, v: Any) -> Any:
        """Relative vorticity (curl) of ``(u, v)`` at cell vertices."""
        return curl_vertex_cgrid(u, v, self._grid)

    def interpolate(self, field: Any, src_location: Any, dst_location: Any) -> Any:
        """Stagger interpolation of a cell-centre *field* to a face location.

        Only the cell-centre -> face transforms exist on this adapter, so BOTH
        ends of the stagger pair are validated: ``src_location`` must be
        ``"center"`` and ``dst_location`` ``"uface"`` (x-faces) or ``"vface"``
        (y-faces).  Any other pair raises ``ValueError`` rather than silently
        applying a cell-centre transform to a mislabelled source field.
        """
        if (src_location, dst_location) == ("center", "uface"):
            return interp_cell_to_uface(field)
        if (src_location, dst_location) == ("center", "vface"):
            return interp_cell_to_vface(field, self._grid)
        raise ValueError(
            f"unsupported interpolation {src_location!r}->{dst_location!r}; "
            "the lat-lon C-grid adapter supports only 'center'->'uface'/'vface'"
        )

    def halo_fill(self, field: Any) -> Any:
        """Fill the N/S halo rows of a cell-centre scalar (lat-lon pole/wall BC)."""
        return pad_ns_scalar(field, self._grid)


def latlon_cgrid_operators(grid: Any) -> LatLonCGridOperators:
    """Wrap a lat-lon C-grid in its :class:`GridOperators` adapter."""
    return LatLonCGridOperators(grid)
