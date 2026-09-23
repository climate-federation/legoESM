"""Frontogenesis function/angle producer for the frontal GW source (FRONTGF).

Oracle: E3SM ``components/eam/src/dynamics/se/gravity_waves_sources.F90::
compute_frontogenesis`` (Mark Taylor 2011; vendored e3sm-source @ v3.0.1)::

    theta   = T * (p0 / p)^kappa          # potential temperature
    gradth  = grad_sphere(theta)          # horizontal gradient [K/m]
    C       = (gradth . grad) U           # via HOMME ugradv_sphere
    frontgf = -(gradth . C)               # [K^2 m^-2 s^-1]
    frontga = atan2(gradth_y, gradth_x + 1e-10)   # gradient angle [rad]

HOMME's ``ugradv_sphere`` (derivative_mod_base.F90:688) is the COVARIANT
directional derivative computed by the Cartesian-embedding trick: embed the
horizontal wind in R^3 (``vec_sphere2cart``), apply ``(a . grad)`` to each
Cartesian component as a SCALAR, then project back onto the local tangent
basis (dropping the radial part = Levi-Civita connection on the sphere).
This module reproduces exactly that recipe.  Mechanical consequence used as
a test canary: for solid-body rotation ``(a . grad)U = Omega x a``, so
``frontgf = -gradth . (Omega x gradth) == 0`` for ANY theta — a naive
component-wise (non-covariant) implementation fails this identically-zero
identity at high latitude.

Faithfulness scope / departures:
* The E3SM producer runs on the SE (GLL) grid and applies the SE DSS
  (mass-matrix boundary exchange) plus an optional GLL->physgrid remap;
  both are numerical details of the spectral-element discretization, not
  part of the mathematical definition.  Our per-grid gradients (exact
  spherical-harmonic gradients on the Gaussian grid; centered finite
  differences on the lat-lon grid) replace them.
* ``theta`` reuses the shared Exner helper (``theta = T / exner_function(p)``
  with ``constants.p_ref = 1e5 Pa``), the same form as E3SM's
  ``T*(psurf_ref/p)^kappa`` (HOMME ``p0 = 1e5 Pa``).
* Supported grid families: single-column (zeros — no horizontal
  gradients), spectral Gaussian, UNIFORM GLOBAL lat-lon (array-checked:
  uniform spacing + periodic longitude), and the MPAS Voronoi mesh (cell
  scalar gradient = Ringler edge gradient ``gradient_edge_3d`` reconstructed
  to cell (east, north) components by the Perot ``reconstruct_cell_velocity``
  — the mesh's own second-order gradient operator).  Cubed-sphere and the
  regional/stretched lat-lon builders return
  ``frontogenesis_supported(grid) == False`` and the coupled pipeline keeps
  its loud frontal-source rejection there (no silent no-op, no wrong
  stencil).

Pre-impl search: mirrors the ``moisture_convergence_supported`` /
``compute_moisture_convergence`` grid-dispatch diagnostic in
``atmosphere/physics/_shared.py`` (same predicate+compute pattern, same
grid-family detection); no existing lat-lon scalar-gradient operator was
found (``operators_3d`` gradients are cubed-sphere), so the small centered-
difference gradient below is local to this module.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import exner_function

__physics_contract__ = {
    "summary": (
        "Frontogenesis function F = -grad(theta) . [(grad(theta) . grad) U] "
        "and gradient angle (E3SM FRONTGF/FRONTGA), the trigger field for "
        "the frontal (CM) gravity-wave source."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "p_full": "Pa",
    },
    "outputs": {
        "frontgf": "K^2 m^-2 s^-1", "frontga": "rad",
    },
    "sign_convention": (
        "Diagnostic only (no tendency). frontgf > 0 where the flow SHARPENS "
        "the horizontal theta gradient (frontogenesis; confluent deformation "
        "aligned with grad theta), < 0 for frontolysis; the frontal GW "
        "source launches where frontgf exceeds frontgfc. (grad(theta).grad) "
        "is the COVARIANT directional derivative (Cartesian embedding + "
        "tangential projection, HOMME ugradv_sphere), so rigid solid-body "
        "rotation gives frontgf == 0 identically."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "E3SM v3.0.1 components/eam/src/dynamics/se/"
        "gravity_waves_sources.F90::compute_frontogenesis (M. Taylor 2011); "
        "HOMME derivative_mod_base.F90::ugradv_sphere"
    ),
    "idealized_test": (
        "tests/unit/test_frontogenesis_diagnostic.py: solid-body rotation "
        "-> F == 0 for any theta (covariant canary); equatorial confluence "
        "u=-alpha*x with theta=beta*x -> F = +alpha*beta^2; spectral-exact "
        "gradient pins; spectral-vs-latlon cross-implementation agreement."
    ),
}

# E3SM gravity_waves_sources.F90:178 — regulariser on the x-gradient inside
# atan2 (keeps the angle defined where grad(theta) -> 0); oracle literal.
_FRONTGA_EPS = 1.0e-10


def _is_uniform_global_latlon(grid) -> bool:
    """True for a UNIFORM, GLOBALLY-PERIODIC lat-lon grid.

    The centered-difference stencil below assumes uniform latitude spacing
    (``grid.dlat`` is only a REPRESENTATIVE spacing on the stretched
    builder) and a periodic longitude seam (regional grids add wall
    columns).  Enabling it on ``create_regional_latlon_grid`` /
    ``create_stretched_latlon_grid`` output would silently produce a wrong
    frontal GWD trigger (codex R1 finding 1) — so the family gate checks
    the ARRAYS, not just the attribute names.  Build-time only (concrete
    numpy on grid metadata; never traced).
    """
    if not (hasattr(grid, "dlat") and hasattr(grid, "dlon")
            and hasattr(grid, "lat") and hasattr(grid, "lon")):
        return False
    import numpy as np

    lat = np.asarray(grid.lat)
    lon = np.asarray(grid.lon)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 3 or lon.size < 3:
        return False
    # Dtype-aware tolerance at the ROUNDOFF scale of the stored axis values
    # (default grids store fp32 axes: spacing differences carry
    # ~2*eps32*|lat| ~ 4e-7 rad of storage noise).  1e2*eps*max(1,|lat|)
    # (~1.2e-5 rad in fp32) accepts that noise while rejecting even a MILD
    # stretched-builder step change (0.01 deg ~ 1.7e-4 rad) — codex R2
    # found the earlier 1e4*eps band wide enough to admit percent-scale
    # metric error.
    eps = float(np.finfo(lat.dtype).eps)
    tol = 1.0e2 * eps * max(1.0, float(np.max(np.abs(lat))))
    dlat = float(grid.dlat)
    dlon = float(grid.dlon)
    uniform_lat = bool(np.max(np.abs(np.diff(lat) - dlat)) < tol)
    uniform_lon = bool(np.max(np.abs(np.diff(lon) - dlon)) < tol)
    global_lon = bool(abs(lon.size * dlon - 2.0 * np.pi) < 1.0e-6)
    # Latitude must span the GLOBE too (cell centers of a uniform global
    # grid: +/-(pi/2 - dlat/2)).  Without this a periodic regional CHANNEL
    # (full longitude, walled latitude band) or an SPMD latitude-band slice
    # would pass, and the one-sided edge differences would then sit at
    # physical walls / internal partition edges (decomposition-dependent
    # FRONTGF) — codex R2 finding 1.
    span_tol = max(tol, 1.0e-6)
    global_lat = bool(
        abs(float(lat[0]) + (np.pi / 2.0 - 0.5 * dlat)) < span_tol
        and abs(float(lat[-1]) - (np.pi / 2.0 - 0.5 * dlat)) < span_tol
    )
    return uniform_lat and uniform_lon and global_lon and global_lat


def frontogenesis_supported(grid) -> bool:
    """True when :func:`compute_frontogenesis` has an operator for *grid*.

    Mirrors the dispatch exactly (single-column, spectral Gaussian,
    uniform GLOBAL lat-lon).  Cubed-sphere, MPAS, and regional/stretched
    lat-lon variants are NOT supported — the coupled pipeline uses this
    predicate to keep rejecting a frontal GWD source there loudly instead
    of launching nothing (or launching from a wrong stencil).
    """
    return (
        getattr(grid, "grid_n_columns", None) == 1
        or (hasattr(grid, "n_max") and hasattr(grid, "Pnm"))
        or _is_uniform_global_latlon(grid)
        or _is_voronoi(grid)
    )


def _is_voronoi(grid) -> bool:
    return all(hasattr(grid, a) for a in
               ("cellsOnEdge", "dcEdge", "edgesOnCell", "latCell", "lonCell"))


def _voronoi_gradient(mesh, f):
    """Cell-centred (east, north) gradient of ``f`` (nCells, 1, K) [1/m]."""
    from legoesm.core.operators_voronoi import gradient_edge_3d
    from legoesm.grids.voronoi import reconstruct_cell_velocity

    g_edge = gradient_edge_3d(f[:, 0, :], mesh)           # (nEdges, K)
    gx, gy = reconstruct_cell_velocity(g_edge, mesh)      # (nCells, K) each
    return gx[:, None, :], gy[:, None, :]


# ---------------------------------------------------------------------------
# Per-grid scalar gradients (batched over a trailing level axis)
# ---------------------------------------------------------------------------

def _spectral_gradient(grid, f):
    """Exact SH gradient of ``f`` (n_lat, n_lon, K) -> (d/dx, d/dy) [1/m].

    Zonal:      df/dx = synth(i m f_hat) / (a cos(lat))
    Meridional: df/dy = -synth_H(f_hat) / (a cos(lat))

    where ``synth_H`` uses the derivative Legendre ``Hnm = -(1-mu^2) dP/dmu``
    so ``synth_H(f_hat) = cos(lat) * d f / d(colat) = -cos(lat) * df/dlat``
    (the same convention ``uv_from_vordiv`` builds ``u*cos``/``v*cos`` from).
    Gaussian latitudes exclude the poles, so the ``1/cos`` is finite.
    """
    from legoesm.grids.gaussian import (
        sh_analysis_3d,
        sh_synthesis_3d,
        sh_synthesis_H_3d,
    )

    f_hat = sh_analysis_3d(grid, f)                       # (n_sh, K)
    im = 1j * grid.ms.astype(jnp.float64)
    df_dlon = sh_synthesis_3d(grid, im[:, None] * f_hat)  # d/d(lon)
    coslat_h = sh_synthesis_H_3d(grid, f_hat)             # cos*d/d(colat)
    a_cos = grid.radius * grid.cos_lat[:, None, None]
    return df_dlon / a_cos, -coslat_h / a_cos


def _latlon_gradient(grid, f):
    """Centered-difference gradient of ``f`` (n_lat, n_lon, K) [1/m].

    Longitude is periodic (roll); latitude uses centered interior and
    one-sided edge differences.  Metric: dx = a cos(lat) dlon, dy = a dlat.
    """
    a = float(getattr(grid, "radius", None) or _earth_radius())
    dlon = grid.dlon
    dlat = grid.dlat
    coslat = jnp.cos(grid.lat2d)[:, :, None]
    df_dlon = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dlon)
    df_dx = df_dlon / (a * coslat)
    interior = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dlat)
    south = (f[1:2, :, :] - f[0:1, :, :]) / dlat
    north = (f[-1:, :, :] - f[-2:-1, :, :]) / dlat
    df_dy = jnp.concatenate([south, interior, north], axis=0) / a
    return df_dx, df_dy


def _earth_radius() -> float:
    from legoesm import constants

    return constants.R_earth


# ---------------------------------------------------------------------------
# Covariant core (E3SM ugradv_sphere Cartesian-embedding recipe)
# ---------------------------------------------------------------------------

def _frontogenesis_core(u, v, theta, lat2d, lon2d, grad_fn):
    """F = -gradth . [(gradth . grad) U] with the covariant derivative.

    All fields (n_lat, n_lon, K).  ``grad_fn(f) -> (df/dx, df/dy)`` is the
    grid family's scalar gradient.  Mirrors HOMME ``ugradv_sphere``:
    embed (u, v) in R^3 with the local (east, north) tangent basis, take
    ``(gradth . grad)`` of each Cartesian component as a scalar, project
    back onto the tangent basis (the radial part drops — Levi-Civita
    connection), then contract with gradth.
    """
    gx, gy = grad_fn(theta)                               # [K/m]

    sinl = jnp.sin(lon2d)[:, :, None]
    cosl = jnp.cos(lon2d)[:, :, None]
    sinp = jnp.sin(lat2d)[:, :, None]
    cosp = jnp.cos(lat2d)[:, :, None]
    # vec_sphere2cart: east and north unit vectors in R^3.
    e_lon = (-sinl, cosl, jnp.zeros_like(sinl))
    e_lat = (-sinp * cosl, -sinp * sinl, cosp)

    c_lon = jnp.zeros_like(gx)
    c_lat = jnp.zeros_like(gx)
    for el, ep in zip(e_lon, e_lat):
        w = u * el + v * ep                               # Cartesian component
        wx, wy = grad_fn(w)
        c_comp = gx * wx + gy * wy                        # (gradth . grad) w
        c_lon = c_lon + c_comp * el                       # tangential projection
        c_lat = c_lat + c_comp * ep
    frontgf = -(gx * c_lon + gy * c_lat)                  # [K^2 m^-2 s^-1]
    frontga = jnp.arctan2(gy, gx + _FRONTGA_EPS)          # [rad]
    return frontgf, frontga


# ---------------------------------------------------------------------------
# Public producer
# ---------------------------------------------------------------------------

def compute_frontogenesis(
    u_grid: jnp.ndarray,
    v_grid: jnp.ndarray,
    T_grid: jnp.ndarray,  # noqa: N803 - physics symbol (T), matches e3sm_cam_gwd
    p_full_grid: jnp.ndarray,
    grid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Frontogenesis function + angle on the native grid (E3SM FRONTGF).

    Parameters
    ----------
    u_grid, v_grid, T_grid, p_full_grid : jax.Array
        Grid-shaped fields:

        * spectral Gaussian / lat-lon — ``(n_lat, n_lon, nlev)``
        * single column — any shape reducible to ``(ncol, nlev)``
    grid
        Grid metadata (dispatched by family; see
        :func:`frontogenesis_supported`).

    Returns
    -------
    (frontgf, frontga) : jax.Array, each shape (ncol, nlev)
        Column-flattened frontogenesis function [K^2 m^-2 s^-1] and
        gradient angle [rad], the layout ``e3sm_cam_gwd`` expects for
        ``frontgf_col``.
    """
    # Single-column: no horizontal gradients -> frontgf = 0 everywhere
    # (mirrors compute_moisture_convergence; the frontal trigger
    # frontgf > frontgfc then never fires, which is the physical answer
    # for a column with no resolved front).
    if getattr(grid, "grid_n_columns", None) == 1:
        ncol_flat = T_grid.reshape(-1, T_grid.shape[-1]).shape[0]
        z = jnp.zeros((ncol_flat, T_grid.shape[-1]), dtype=T_grid.dtype)
        return z, z

    theta = T_grid / exner_function(p_full_grid)          # T*(p0/p)^kappa

    if hasattr(grid, "n_max") and hasattr(grid, "Pnm"):
        grad_fn = lambda f: _spectral_gradient(grid, f)   # noqa: E731
        lat2d, lon2d = grid.lat2d, grid.lon2d
    elif _is_uniform_global_latlon(grid):
        grad_fn = lambda f: _latlon_gradient(grid, f)     # noqa: E731
        lat2d, lon2d = grid.lat2d, grid.lon2d
    elif _is_voronoi(grid):
        # Column fields (nCells, K) -> the core's (n_lat, n_lon, K) layout
        # with n_lon == 1; cell lat/lon play the 2-D coordinate arrays.
        grad_fn = lambda f: _voronoi_gradient(grid, f)    # noqa: E731
        lat2d = jnp.asarray(grid.latCell)[:, None]
        lon2d = jnp.asarray(grid.lonCell)[:, None]
        u_grid, v_grid = u_grid[:, None, :], v_grid[:, None, :]
        theta = theta[:, None, :]
    else:
        raise TypeError(
            f"compute_frontogenesis: unsupported grid type {type(grid)!r}; "
            "supported families are single-column, spectral Gaussian, MPAS "
            "Voronoi, and UNIFORM GLOBAL lat-lon (regional/stretched lat-lon variants "
            "would get a wrong stencil; guard with "
            "frontogenesis_supported(grid))."
        )

    frontgf, frontga = _frontogenesis_core(
        u_grid, v_grid, theta, lat2d, lon2d, grad_fn,
    )
    n_lat, n_lon, nlev = frontgf.shape
    return (
        frontgf.reshape(n_lat * n_lon, nlev),
        frontga.reshape(n_lat * n_lon, nlev),
    )
