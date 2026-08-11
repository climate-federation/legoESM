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
    "fesom",
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

    if grid_type == "fesom":
        # FESOM2 unstructured triangular mesh (the fesom_jax dycore).  Like
        # tripole it is mesh-file-backed, so it is NOT in _RESOLUTION_GRIDS.
        if resolution is not None:
            raise ValueError(
                "fesom is a mesh-file-backed ocean grid: pass "
                "mesh_dir=<FESOM mesh directory>, not a resolution."
            )
        mesh_dir = kwargs.pop("mesh_dir", None)  # None => packaged pi mesh
        H_max = kwargs.pop("H_max", None)
        nlev = kwargs.pop("nlev", None)
        land_lat_threshold = kwargs.pop("land_lat_threshold", 90.0)
        # Explicit interface depths, so FESOM can be built on the SAME vertical
        # grid as the other dycores. Silently dropping this (the pre-2026-08-08
        # behaviour) left FESOM on uniform 1 m layers while the caller's
        # diagnostic weighted it with a stretched z-star dz.
        zbar = kwargs.pop("zbar", None)
        if H_max is None or nlev is None:
            raise ValueError(
                "fesom requires H_max and nlev (a FESOM mesh without a "
                "vertical specification is meaningless here)."
            )
        # Function-scope imports: the core grids package must not import the
        # ocean package or fesom_jax at module scope.
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            FesomOceanGrid,
            _require_fesom_jax,
            build_flat_bottom_mesh,
        )

        # Raise a clear ImportError naming the package + install command
        # rather than a raw ModuleNotFoundError from the line below.
        _require_fesom_jax()
        from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh

        mesh = load_mesh(DEFAULT_PI_MESH_DIR if mesh_dir is None else mesh_dir)
        mesh = build_flat_bottom_mesh(
            mesh,
            H_max=H_max,
            nlev=nlev,
            land_lat_threshold=land_lat_threshold,
            zbar=zbar,
        )
        if kwargs:
            raise ValueError(
                f"create_grid('fesom', ...) got unexpected keyword(s) "
                f"{sorted(kwargs)}. Refusing to silently drop them."
            )
        return FesomOceanGrid(mesh)

    raise ValueError(
        f"Unknown grid_type {grid_type!r}. Available global grids: "
        f"{', '.join(GLOBAL_GRID_TYPES)} (for a regional grid use "
        f"create_regional_grid; for a doubly-periodic box/SCM use "
        f"legoesm.grids.plane.create_plane_grid)."
    )


#: Regional / limited-area grids instantiable via :func:`create_regional_grid`.
REGIONAL_GRID_TYPES: tuple[str, ...] = (
    "latlon", "latlon_stretched", "mercator", "mpas", "cubed_sphere",
)


def create_regional_grid(grid_type: str, **kwargs: Any):
    """Instantiate a REGIONAL / limited-area horizontal grid by canonical grid_type.

    The regional counterpart of :func:`create_grid` (which builds the *global*
    grids), giving the atmosphere the same uniform regional entry the ocean
    already uses.  Regional grids are NOT sized by a single ``resolution`` — each
    takes its own domain bounds / cell size — so per-type kwargs pass straight
    through to the underlying constructor, which enforces its own required
    arguments:

    ============  ==============================================  =================
    grid_type     constructor (key kwargs)                        returns
    ============  ==============================================  =================
    ``latlon``    ``create_regional_latlon_grid`` (n_lat, n_lon,  ``(grid, mask)``
                  lat_south, lat_north, lon_west=, lon_east=,
                  periodic_x=)
    ``latlon_stretched`` ``create_stretched_latlon_grid`` (dy_deg, ``(grid, mask)``
                  n_lon, lat_south, lon_west=, lon_east=,
                  periodic_x=) — arbitrary per-row meridional
                  spacing (e.g. Vinokur)
    ``mercator``  ``create_mercator_grid`` (n_lon, lat_max_deg,   ``grid``
                  lon_west_deg=, lon_east_deg=)
    ``mpas``      ``create_regional_voronoi_mesh`` (lon_range,     ``grid``
                  lat_range, resolution_km=, periodic_x=)
    ``cubed_sphere`` ``create_cubed_sphere_panel`` (n, face_id=,   ``grid``
                  return_cdgrid=)
    ============  ==============================================  =================

    Note ``latlon`` / ``latlon_stretched`` return a ``(grid, wall_mask)`` tuple
    (the others return a bare grid object — they carry their boundary masking differently).  Raises
    ``ValueError`` for an unknown *grid_type*.  Doubly-periodic *idealized* boxes
    (LES / SCM / RCE) are a distinct extent — use
    ``legoesm.grids.plane.create_plane_grid``.
    """
    if grid_type == "latlon":
        from legoesm.grids.latlon import create_regional_latlon_grid

        return create_regional_latlon_grid(**kwargs)
    if grid_type == "latlon_stretched":
        from legoesm.grids.latlon import create_stretched_latlon_grid

        return create_stretched_latlon_grid(**kwargs)
    if grid_type == "mercator":
        from legoesm.grids.latlon import create_mercator_grid

        return create_mercator_grid(**kwargs)
    if grid_type == "mpas":
        from legoesm.grids.voronoi import create_regional_voronoi_mesh

        return create_regional_voronoi_mesh(**kwargs)
    if grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere_panel

        return create_cubed_sphere_panel(**kwargs)
    raise ValueError(
        f"Unknown regional grid_type {grid_type!r}. Available: "
        f"{', '.join(REGIONAL_GRID_TYPES)} (global grids: create_grid; "
        f"idealized box: legoesm.grids.plane.create_plane_grid)."
    )
