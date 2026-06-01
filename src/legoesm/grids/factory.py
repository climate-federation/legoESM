"""Uniform grid instantiation — one entry, any global grid, any component.

The L1 principle: the grid is the foundational value object, and a model is
constructed *on* a grid.  :func:`create_grid` is the single, component-agnostic
entry to that — the **same** call instantiates the grid an atmosphere or an ocean
(or land/ice) is then built on, so no component owns its own grid-construction
dispatch.

Two construction modes, both behind the one entry:

* **resolution-based** global grids take a single ``resolution`` int:

  ================  =====================================================
  grid_type         resolution means                  constructor
  ================  =====================================================
  ``cubed_sphere``  cube panel size N (C\\ *N*)        ``create_cubed_sphere``
  ``gaussian``      spectral truncation ``n_max``     ``create_gaussian_grid``
  ``latlon``        number of latitudes ``n_lat``     ``create_latlon_grid``
  ``mpas``          SCVT subdivision level            ``create_voronoi_mesh``
  ================  =====================================================

* **file-backed** global grid (ocean OMIP): ``tripole`` loads a NEMO mesh from
  ``grid_file=<mesh_mask/domcfg .nc>`` (``resolution`` is ignored) ->
  ``create_tripole_grid``.

Per-grid options (``radius``, ``lloyd_iterations``, ``n_lon`` ...) pass through as
keyword arguments.  The doubly-periodic ``plane`` box (LES / SCM) is NOT here: it
takes ``(nx, ny, nlev, dx, dy)`` rather than a single resolution or a file — call
``legoesm.grids.plane.create_plane_grid`` directly.

Expects a **canonical** ``grid_type`` (legacy aliases like ``voronoi`` /
``icosahedral`` are normalised to ``mpas`` at the config boundary via
``driver.config.normalize_grid_type``).
"""

from __future__ import annotations

from typing import Any

#: Global grids ocean and atmosphere share, instantiable via :func:`create_grid`.
GLOBAL_GRID_TYPES: tuple[str, ...] = (
    "cubed_sphere",
    "gaussian",
    "latlon",
    "mpas",
    "tripole",
)

# The resolution-based subset (a single int sizes them).
_RESOLUTION_GRIDS: frozenset[str] = frozenset(
    {"cubed_sphere", "gaussian", "latlon", "mpas"}
)


def create_grid(grid_type: str, resolution: int | None = None, **kwargs: Any):
    """Instantiate any global horizontal grid by canonical *grid_type*.

    Usable by every component (atmosphere/ocean/land/ice) — the grid is the same
    foundational value regardless of who builds on it.  Resolution-based grids
    need ``resolution``; the file-backed ``tripole`` ocean grid needs
    ``grid_file=``.  Raises ``ValueError`` for an unknown grid type (with the
    available list), never silently defaulting.
    """
    if grid_type in _RESOLUTION_GRIDS:
        if resolution is None:
            raise ValueError(f"grid_type {grid_type!r} requires a resolution.")
        if grid_type == "cubed_sphere":
            from legoesm.grids.cubed_sphere import create_cubed_sphere

            return create_cubed_sphere(resolution, **kwargs)
        if grid_type == "gaussian":
            from legoesm.grids.gaussian import create_gaussian_grid

            return create_gaussian_grid(resolution, **kwargs)
        if grid_type == "latlon":
            from legoesm.grids.latlon import create_latlon_grid

            return create_latlon_grid(resolution, **kwargs)
        # mpas
        from legoesm.grids.voronoi import create_voronoi_mesh

        return create_voronoi_mesh(resolution, **kwargs)

    if grid_type == "tripole":
        grid_file = kwargs.pop("grid_file", None)
        if grid_file is None:
            raise ValueError(
                "tripole is a file-backed ocean grid: pass "
                "grid_file=<NEMO mesh_mask/domcfg .nc>."
            )
        from legoesm.grids.tripole import create_tripole_grid

        return create_tripole_grid(grid_file, **kwargs)

    raise ValueError(
        f"Unknown grid_type {grid_type!r}. Available global grids: "
        f"{', '.join(GLOBAL_GRID_TYPES)} (for a doubly-periodic box/SCM use "
        f"legoesm.grids.plane.create_plane_grid)."
    )
