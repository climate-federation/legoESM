"""Capability matrix + validated factory for grid x extent x operators x nesting.

legoESM can solve on several horizontal grids (lat-lon, cubed-sphere, doubly-
periodic Cartesian plane, icosahedral/MPAS Voronoi, Gaussian/spectral, tripole),
at several spatial extents (global, regional, doubly-periodic box), with several
operator families (component ``(u, v)`` C-grid operators, edge-normal TRiSK
operators, Cartesian plane operators, or global spectral transforms).  **Not
every combination exists** — e.g. you cannot build a regional Gaussian/spectral
grid, edge operators do not apply to a lat-lon C-grid, and regional *nesting* is
not implemented for any grid yet.

This module is the single validated entry point: :func:`instantiate` checks the
requested ``(grid_type, extent, operators, nesting)`` against the capability
matrix and either builds it (delegating to the existing
:func:`legoesm.grids.factory.create_grid` /
:func:`~legoesm.grids.factory.create_regional_grid` /
:func:`legoesm.grids.plane.create_plane_grid` + the operator adapters) or raises
a precise error naming what is and is not available — never silently falling back
to a different grid/extent/operator.

The grid-type sets are DERIVED from the factory's ``GLOBAL_GRID_TYPES`` /
``REGIONAL_GRID_TYPES`` (single source of truth); only the operator-family and
nesting columns are owned here.
"""

from __future__ import annotations

from typing import Any

from legoesm.grids.factory import (
    GLOBAL_GRID_TYPES,
    REGIONAL_GRID_TYPES,
    create_grid,
    create_regional_grid,
)

#: The three spatial extents.
EXTENTS: tuple[str, ...] = ("global", "regional", "double_periodic")

#: User-facing grid aliases -> canonical factory grid_type.  The MPAS Voronoi mesh
#: is built on an icosahedral base (``grids.voronoi._icosahedral_base``), so
#: ``"icosahedral"`` is the same family as ``"mpas"``; ``"plane"`` / ``"xy"`` name
#: the doubly-periodic Cartesian box.
_ALIASES: dict[str, str] = {
    "icosahedral": "mpas",
    "voronoi": "mpas",
    "xy": "plane",
    "cartesian": "plane",
    "doubly_periodic": "plane",
}

#: Operator family each grid solves with — i.e. how you take divergence/gradient/
#: vorticity on it.  ``None`` (spectral) means there is no grid-local stencil
#: operator: the dynamics run through global transforms, so requesting
#: ``operators=True`` on such a grid is an explicit error.
_OPERATOR_FAMILY: dict[str, str | None] = {
    "latlon": "grid_operators",        # component (u, v) lat-lon C-grid
    "cubed_sphere": "grid_operators",  # component (u, v) cubed-sphere C-D grid
    "tripole": "grid_operators",       # curvilinear lat-lon C-grid (ocean)
    "mercator": "grid_operators",      # regional lat-lon C-grid family
    "mpas": "edge_operators",          # edge-normal TRiSK (icosahedral Voronoi)
    "plane": "plane_operators",        # Cartesian doubly-periodic operators
    "gaussian": None,                  # spectral transforms — no grid-local ops
}

#: Doubly-periodic Cartesian box grid(s) — the third extent.
_DOUBLE_PERIODIC: frozenset[str] = frozenset({"plane"})

#: Regional NESTING (a child grid refined inside a parent) — not implemented for
#: any grid yet.  ``grid_type -> frozenset of extents at which nesting exists``;
#: empty everywhere today, so any ``nesting=True`` request is flagged.
_NESTING_SUPPORT: dict[str, frozenset[str]] = {}


def _canonical(grid_type: str) -> str:
    return _ALIASES.get(grid_type, grid_type)


def _all_grid_types() -> set[str]:
    return (
        set(GLOBAL_GRID_TYPES)
        | set(REGIONAL_GRID_TYPES)
        | set(_DOUBLE_PERIODIC)
        | set(_OPERATOR_FAMILY)
    )


def supported_extents(grid_type: str) -> frozenset[str]:
    """The extents a grid supports (derived from the factory's grid-type sets)."""
    g = _canonical(grid_type)
    extents: set[str] = set()
    if g in GLOBAL_GRID_TYPES:
        extents.add("global")
    if g in REGIONAL_GRID_TYPES:
        extents.add("regional")
    if g in _DOUBLE_PERIODIC:
        extents.add("double_periodic")
    return frozenset(extents)


def operator_family(grid_type: str) -> str | None:
    """The operator family for a grid (``None`` = spectral / no grid-local ops)."""
    return _OPERATOR_FAMILY.get(_canonical(grid_type))


def capability_matrix() -> dict[str, dict[str, Any]]:
    """The full matrix: ``grid_type -> {extents, operator_family, nesting}``."""
    matrix: dict[str, dict[str, Any]] = {}
    for g in sorted(_all_grid_types()):
        matrix[g] = {
            "extents": sorted(supported_extents(g)),
            "operator_family": operator_family(g),
            "nesting_extents": sorted(_NESTING_SUPPORT.get(g, frozenset())),
        }
    return matrix


def _build_operators(grid_type: str, grid: Any) -> Any:
    """Wrap *grid* in its operator adapter, or raise if it has no grid-local ops."""
    family = operator_family(grid_type)
    g = _canonical(grid_type)
    if family is None:
        raise ValueError(
            f"grid {grid_type!r} ({g}) has no grid-local differential operators: "
            f"it is a spectral grid whose dynamics run through global transforms, "
            f"not divergence/gradient/vorticity stencils.  Do not request "
            f"operators=True for it."
        )
    if family == "grid_operators" and g in ("latlon", "mercator", "tripole"):
        from legoesm.grids.operator_adapters import latlon_cgrid_operators

        return latlon_cgrid_operators(grid)
    if family == "grid_operators" and g == "cubed_sphere":
        from legoesm.grids.operator_adapters import cubed_sphere_cdgrid_operators

        return cubed_sphere_cdgrid_operators(grid)
    if family == "edge_operators":
        from legoesm.grids.operator_adapters import mpas_edge_operators

        return mpas_edge_operators(grid)
    if family == "plane_operators":
        # The plane grid's operators are the module-level Cartesian operators in
        # the atmosphere component (legoesm.atmosphere.dynamics.plane_operators);
        # the substrate cannot import a component, so the plane carries its own
        # operators with it — the caller uses them directly, not via an adapter.
        raise ValueError(
            f"plane (doubly-periodic) operators are Cartesian and live with the "
            f"plane dynamics, not in a substrate adapter; build the grid here and "
            f"use legoesm.atmosphere.dynamics.plane_operators directly."
        )
    raise ValueError(  # pragma: no cover — every family handled above
        f"no operator adapter wired for grid {grid_type!r} (family {family!r})."
    )


def instantiate(
    grid_type: str,
    *,
    extent: str = "global",
    operators: bool = False,
    nesting: bool = False,
    **kwargs: Any,
):
    """Validate and build a ``(grid [, operators])`` for the requested combination.

    Parameters
    ----------
    grid_type
        ``latlon`` | ``cubed_sphere`` | ``plane`` (alias ``xy``) | ``mpas``
        (alias ``icosahedral``) | ``gaussian`` | ``tripole`` | ``mercator``.
    extent
        ``"global"`` | ``"regional"`` | ``"double_periodic"``.
    operators
        If true, also build and return the grid's operator adapter; the result is
        a ``(grid, operators)`` tuple.  Raises if the grid has no grid-local
        operators (spectral) — see :func:`operator_family`.
    nesting
        If true, request a *nested* (refined-child) grid.  Not implemented for any
        grid yet, so this currently always raises ``NotImplementedError`` — the
        hook is here so callers get a clear message instead of a wrong grid.

    Raises
    ------
    ValueError
        Unknown grid type, unknown/unsupported extent for the grid, or operators
        requested on a grid that has none.
    NotImplementedError
        Nesting requested where it is not available.
    """
    g = _canonical(grid_type)
    if g not in _all_grid_types():
        raise ValueError(
            f"Unknown grid_type {grid_type!r}.  Available: "
            f"{', '.join(sorted(_all_grid_types()))} "
            f"(aliases: {', '.join(sorted(_ALIASES))})."
        )
    if extent not in EXTENTS:
        raise ValueError(
            f"Unknown extent {extent!r}; expected one of {EXTENTS}."
        )

    allowed = supported_extents(g)
    if extent not in allowed:
        raise ValueError(
            f"grid {grid_type!r} ({g}) does not support extent {extent!r}.  "
            f"It is available at: {sorted(allowed) or 'no extents'}.  "
            f"(global grids: {sorted(GLOBAL_GRID_TYPES)}; regional grids: "
            f"{sorted(REGIONAL_GRID_TYPES)}; doubly-periodic: "
            f"{sorted(_DOUBLE_PERIODIC)}.)"
        )

    if nesting:
        nest_extents = _NESTING_SUPPORT.get(g, frozenset())
        if extent not in nest_extents:
            raise NotImplementedError(
                f"regional nesting is not available for grid {grid_type!r} ({g}) "
                f"at extent {extent!r}; grid nesting is not yet implemented for "
                f"any grid in legoESM."
            )

    # --- build the grid via the existing single-source factories ---
    if extent == "global":
        grid = create_grid(g, **kwargs)
    elif extent == "regional":
        grid = create_regional_grid(g, **kwargs)
    else:  # double_periodic
        from legoesm.grids.plane import create_plane_grid

        grid = create_plane_grid(**kwargs)

    if not operators:
        return grid

    # ``create_regional_grid('latlon', ...)`` returns ``(grid, wall_mask)``; wrap
    # the grid object, not the tuple.
    grid_obj = grid[0] if isinstance(grid, tuple) else grid
    ops = _build_operators(g, grid_obj)
    return grid, ops
