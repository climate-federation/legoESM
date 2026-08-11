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
:mod:`legoesm.grids.operators_latlon_cgrid`), the cubed-sphere C-D adapter (over
:mod:`legoesm.core.operators_cdgrid`), and the MPAS edge-normal adapter (the
:class:`~legoesm.grids.operator_protocol.EdgeOperators` sibling, over the TRiSK
kernels in :mod:`legoesm.core.operators_voronoi`).

**Load-bearing status.**  Two production dycores actually *route through* their
adapter (byte-identical, since the adapter delegates to the same free functions):
the lat-lon C-grid shallow-water core (``ops.divergence``/``gradient``/``vorticity``)
and the MPAS shallow-water core (``ops.divergence``/``gradient``).  The cubed-sphere
SW core is FV3-faithful with *bespoke* operators (FB-mode, 2-stage, duogrid corner
work — ``dgrid_to_cgrid``, ``d_sw5_corner_divergence``, ``fv3_del6_vorticity_damping``),
NOT the generic ``cgrid_*`` functions the cube adapter wraps, so the cube adapter is
a validated *forward contract*, not a drop-in route (routing it would change FV3
numerics).  Spectral / SFNO operate through global transforms / neural maps and
expose neither differential contract.
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

    def vorticity(self, u: Any, v: Any, u_ext: Any = None,
                  lat_pad_pole: Any = None, cos_lat_pad: Any = None) -> Any:
        """Relative vorticity (curl) of ``(u, v)`` at cell vertices.

        ``u_ext``: optional pre-padded ``u`` (one lat ghost row per
        side, ``pad_with_pole_bc_lat(u, halo=1, 0, 0)``) so a caller
        that already fused that exchange can share it — see
        ``curl_vertex_cgrid``.  ``lat_pad_pole`` / ``cos_lat_pad``:
        optional precomputed stage-invariant geometry pads, forwarded
        verbatim (see ``curl_vertex_cgrid``).
        """
        return curl_vertex_cgrid(u, v, self._grid, u_ext=u_ext,
                                 lat_pad_pole=lat_pad_pole,
                                 cos_lat_pad=cos_lat_pad)

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
        """Fill the N/S halo rows of a CELL-CENTRE scalar (lat-lon pole/wall BC).

        Cell-centre only (the GridOperators ``halo_fill`` contract); the
        staggered face-velocity halos are filled inside the dycore.
        """
        return pad_ns_scalar(field, self._grid)


def latlon_cgrid_operators(grid: Any) -> LatLonCGridOperators:
    """Wrap a lat-lon C-grid in its :class:`GridOperators` adapter."""
    return LatLonCGridOperators(grid)


class CubedSphereCDGridOperators:
    """:class:`GridOperators` over a cubed-sphere C-D grid, delegating to the
    shared ``core.operators_cdgrid`` free functions (identical numerics).

    Wraps a ``CubedSphereCDGrid``.  Cube velocity is staggered: divergence acts
    on C-grid winds ``(u_c, v_c)``; vorticity on D-grid (corner) winds
    ``(u_d, v_d)`` — the same convention the cube dycores use.
    """

    def __init__(self, cdgrid: Any) -> None:
        self._cdgrid = cdgrid

    @property
    def cdgrid(self) -> Any:
        return self._cdgrid

    def divergence(self, u: Any, v: Any) -> Any:
        """C-grid flux divergence of ``(u_c, v_c)`` -> cell-centre scalar."""
        from legoesm.core.operators_cdgrid import cgrid_divergence

        return cgrid_divergence(u, v, self._cdgrid)

    def gradient(self, scalar: Any) -> Any:
        """C-grid gradient of a cell-centre scalar -> face-normal components."""
        from legoesm.core.operators_cdgrid import cgrid_gradient_2d

        return cgrid_gradient_2d(scalar, self._cdgrid)

    def vorticity(self, u: Any, v: Any) -> Any:
        """Relative vorticity of D-grid (corner) winds ``(u_d, v_d)``."""
        from legoesm.core.operators_cdgrid import dgrid_vorticity

        return dgrid_vorticity(u, v, self._cdgrid)

    def interpolate(self, field: Any, src_location: Any, dst_location: Any) -> Any:
        """Cell-centre <-> corner stagger interpolation (the cube's scalar stagger)."""
        # Authorized use of the cubed-sphere interp helpers.
        from legoesm.core.operators_cdgrid import (
            interp_center_to_corner,
            interp_corner_to_center,
        )

        if (src_location, dst_location) == ("center", "corner"):
            return interp_center_to_corner(field, self._cdgrid)
        if (src_location, dst_location) == ("corner", "center"):
            return interp_corner_to_center(field)
        raise ValueError(
            f"unsupported interpolation {src_location!r}->{dst_location!r}; "
            "the cubed-sphere adapter supports 'center'<->'corner'"
        )

    def halo_fill(self, field: Any) -> Any:
        """Fill the cross-panel halo of a CELL-CENTRE cubed-sphere field.

        Cell-centre only (the GridOperators ``halo_fill`` contract; ``pad_halo_auto``
        auto-selects 2-D/3-D, NOT stagger).  The C-grid face and D-grid corner
        halos are filled by stagger-specific routines inside the cube dycore.
        """
        from legoesm.core.operators_cdgrid import pad_halo_auto

        return pad_halo_auto(field, self._cdgrid)


def cubed_sphere_cdgrid_operators(cdgrid: Any) -> CubedSphereCDGridOperators:
    """Wrap a cubed-sphere C-D grid in its :class:`GridOperators` adapter."""
    return CubedSphereCDGridOperators(cdgrid)


def _rank_route(field: Any, slice_fn: Any, column_fn: Any) -> Any:
    """Pick the 2-D-slice vs column TRiSK kernel by field rank, rejecting rank>2.

    Rank 1 (``(n*,)``) -> single horizontal slice (shallow water); rank 2
    (``(n*, nlev)``) -> column (primitive-equation / ocean).  The ``*_3d`` kernels
    broadcast with ``[:, None]`` and expect exactly rank 2, so a rank-3 bundle
    (e.g. MPAS tracers ``(nCells, nlev, n_tracers)``) must be flattened by the
    caller first — silently feeding it to a column kernel would mis-broadcast.
    """
    nd = field.ndim
    if nd == 1:
        return slice_fn
    if nd == 2:
        return column_fn
    raise ValueError(
        f"MPAS edge operator received a rank-{nd} field; supported ranks are 1 "
        "(horizontal slice) and 2 (column '(n*, nlev)'). Flatten any trailing "
        "channel/tracer axis to '(n*, ncol)' before calling."
    )


class MPASEdgeOperators:
    """:class:`~legoesm.grids.operator_protocol.EdgeOperators` over an MPAS/Voronoi
    mesh, delegating to the shared TRiSK free functions in
    :mod:`legoesm.core.operators_voronoi` (identical numerics).

    MPAS carries a single edge-normal velocity ``u_edge`` (not a ``(u, v)`` pair),
    so it implements the *edge-normal* sibling contract rather than the
    component-velocity ``GridOperators``.  ``divergence``/``vorticity`` are
    differential operators; ``tangential`` (edge-tangential reconstruction) and
    ``cell_to_edge`` (remap) are the TRiSK reconstruction/averaging counterparts
    — mirroring how ``GridOperators`` groups ``interpolate`` alongside the
    derivatives.

    Each method **dispatches on field rank** so the SAME adapter backs every MPAS
    dycore: a single horizontal slice (``(nEdges,)`` / ``(nCells,)``) routes to the
    2-D TRiSK kernel for the shallow-water core; a column field
    (``(nEdges, nlev)`` / ``(nCells, nlev)``) routes to the ``*_3d`` kernel for the
    primitive-equation atmosphere (``primitive_eq_mpas``) and ocean
    (``ocean_model_mpas``).  Rank is static at trace time, so the Python ``if`` is
    JIT-safe.

    The TRiSK kernels are imported at function scope to break the ``grids``<->
    ``core`` import cycle (``core.operators_cdgrid`` imports ``grids.cubed_sphere_cdgrid``;
    a module-scope ``grids.operator_adapters`` -> ``core`` edge would re-enter the
    ``grids`` package mid-init).
    """

    def __init__(self, mesh: Any) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Any:
        return self._mesh

    def divergence(self, u_edge: Any) -> Any:
        """Divergence of edge-normal velocity -> cell-centre scalar (2-D or column)."""
        from legoesm.core.operators_voronoi import (
            divergence_cell,
            divergence_cell_3d,
        )

        fn = _rank_route(u_edge, divergence_cell, divergence_cell_3d)
        return fn(u_edge, self._mesh)

    def gradient(self, phi_cell: Any) -> Any:
        """Gradient of a cell scalar -> edge-normal component (2-D or column)."""
        from legoesm.core.operators_voronoi import gradient_edge, gradient_edge_3d

        fn = _rank_route(phi_cell, gradient_edge, gradient_edge_3d)
        return fn(phi_cell, self._mesh)

    def vorticity(self, u_edge: Any) -> Any:
        """Relative vorticity (curl) of ``u_edge`` -> vertex scalar (2-D or column)."""
        from legoesm.core.operators_voronoi import curl_vertex, curl_vertex_3d

        fn = _rank_route(u_edge, curl_vertex, curl_vertex_3d)
        return fn(u_edge, self._mesh)

    def tangential(self, u_edge: Any) -> Any:
        """Reconstruct the edge-tangential velocity from ``u_edge`` (TRiSK; 2-D/column)."""
        from legoesm.core.operators_voronoi import (
            tangential_velocity,
            tangential_velocity_3d,
        )

        fn = _rank_route(u_edge, tangential_velocity, tangential_velocity_3d)
        return fn(u_edge, self._mesh)

    def cell_to_edge(self, phi_cell: Any) -> Any:
        """Average a cell-centre scalar onto edges (2-D or column)."""
        from legoesm.core.operators_voronoi import (
            cell_to_edge_avg,
            cell_to_edge_avg_3d,
        )

        fn = _rank_route(phi_cell, cell_to_edge_avg, cell_to_edge_avg_3d)
        return fn(phi_cell, self._mesh)


def mpas_edge_operators(mesh: Any) -> MPASEdgeOperators:
    """Wrap an MPAS/Voronoi mesh in its :class:`EdgeOperators` adapter."""
    return MPASEdgeOperators(mesh)
