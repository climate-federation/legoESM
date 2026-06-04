"""Grid-agnostic ocean column physics.

Ocean parameterizations that act on the vertical — convective
adjustment, vertical mixing, EOS-derived static stability — are
intrinsically **column** operations: they depend only on the
``(T, S, rho, vertical-coordinate, jacobian)`` profiles and *not* on the
horizontal grid topology.  The scheme implementations in
:mod:`legoesm.ocean.physics.convection` and
:mod:`legoesm.ocean.physics.vertical_mixing` already operate on raw
``(..., nlev)`` arrays; this module is the thin, explicit grid-agnostic
surface so the same physics runs on every grid:

    cubed-sphere   ``(6, n, n, nlev)``
    lat-lon C-grid ``(n_lat, n_lon, nlev)``  (tripole / eORCA included)
    MPAS           ``(nCells, nlev)``
    spectral       ``(..., nlev)`` after a grid transform

The leading horizontal dimensions are arbitrary and untouched — every
routine here broadcasts over them, so a profile evaluated on one grid is
bit-identical to the same profile on any other grid.

Pre-impl search (CLAUDE.md): the convective-adjustment maths already
lives in ``convection/enhanced_diffusion.py`` (``enhanced_diffusion_
convection``) and the K-combination in
``vertical_mixing/k_profiles.py`` (``compute_vertical_K_profiles``);
this module re-exports them through a grid-agnostic API rather than
re-deriving N²/EOS/diffusion.
"""

from __future__ import annotations


from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import (
    enhanced_diffusion_convection,
)


def convective_adjustment_K(T, S, rho, z_coord, jacobian, cfg=None):
    """Convective-adjustment vertical diffusivity at interior interfaces.

    Oceananigans-style ``ConvectiveAdjustmentVerticalDiffusivity``:
    return a LARGE diffusivity where the column is statically unstable
    (``N² < 0``) and the background value where it is stable.  This is
    the diffusivity that, fed into a backward-Euler vertical-diffusion
    solve, rapidly homogenises a gravitationally unstable column.

    Grid-agnostic: the leading horizontal dimensions of the inputs are
    arbitrary and untouched (works for cubed-sphere ``(6, n, n, nlev)``,
    lat-lon / tripole ``(n_lat, n_lon, nlev)``, MPAS ``(nCells, nlev)``).

    Apply the result by ADDING ``K_conv`` into the vertical-diffusion K
    profile of the grid's implicit solve — exactly what
    :func:`legoesm.ocean.physics.vertical_mixing.k_profiles.
    compute_vertical_K_profiles` already does when
    ``convection.scheme == "enhanced_diffusion"``.  This function is the
    standalone grid-agnostic entry point for callers that build the K
    profile themselves.

    Parameters
    ----------
    T, S, rho : array, shape ``(..., nlev)``
        Cell-centered potential temperature [°C], salinity [PSU], and
        in-situ density [kg/m³].  Only ``rho`` enters the static-stability
        criterion; ``T``/``S`` are accepted for a uniform call signature
        (and used only on the explicit-diffusion path, which is disabled
        here).
    z_coord : OceanZStarCoordinate | OceanPartialCellCoordinate
        Vertical coordinate (supplies ``dz_ref`` for ``N²``).
    jacobian : array, shape ``(...)``
        z-star layer-stretching factor ``(eta + H) / H``.
    cfg : EnhancedDiffusionConfig, optional
        Convective-adjustment parameters (``K_conv``, ``K_bg``, smooth
        vs. hard transition).  Defaults to :class:`EnhancedDiffusionConfig`.

    Returns
    -------
    K_conv : array, shape ``(..., nlev-1)``
        Convective diffusivity at interior interfaces [m²/s].
    """
    if cfg is None:
        cfg = EnhancedDiffusionConfig()
    out = enhanced_diffusion_convection(
        T, S, rho, z_coord, jacobian, cfg, apply_diffusion=False,
    )
    return out.K_v


def extract_cell_center_velocity(u, v, grid_type: str):
    """Interpolate staggered velocity to cell centers — grid-agnostic.

    Column physics that need momentum (e.g. KPP shear) want ``u``/``v`` at
    T-points.  Factoring the per-stagger interpolation here keeps the
    physics layer free of grid-type branches.

    - ``"latlon"`` / ``"latlon_regional"`` / ``"tripole"`` (Arakawa
      C-grid): ``u`` lives on lon faces ``(n_lat, n_lon+1, nlev)`` and
      ``v`` on lat faces ``(n_lat+1, n_lon, nlev)``; average the two
      adjacent faces to the cell center ``(n_lat, n_lon, nlev)``.
    - ``"cubed_sphere"`` / ``"a_grid"`` / ``"latlon_a"`` (A-grid):
      already cell-centered → returned unchanged.
    - ``"mpas"``: edge-normal velocity needs the mesh (cellsOnEdge /
      TRiSK reconstruction); raise so the caller supplies a
      mesh-reconstructed cell velocity instead of guessing.

    Returns
    -------
    (u_cell, v_cell) : each ``(..., nlev)`` at T-points.
    """
    if grid_type in ("latlon", "latlon_regional", "tripole"):
        # u: average adjacent lon faces (axis -2, length n_lon+1).
        u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
        # v: average adjacent lat faces (axis 0, length n_lat+1).
        v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])
        return u_cell, v_cell
    if grid_type in ("cubed_sphere", "a_grid", "latlon_a"):
        return u, v
    if grid_type == "mpas":
        raise NotImplementedError(
            "extract_cell_center_velocity: MPAS edge->cell reconstruction "
            "needs the VoronoiMesh (cellsOnEdge / TRiSK); pass the "
            "mesh-reconstructed cell velocity directly."
        )
    raise ValueError(
        f"extract_cell_center_velocity: unknown grid_type {grid_type!r}"
    )


__all__ = ["convective_adjustment_K", "extract_cell_center_velocity"]
