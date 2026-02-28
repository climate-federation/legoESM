"""DCMIP-2012 transport test cases (Tests 1-1, 1-2, 1-3).

Prescribed-wind 3D tracer transport problems from the Dynamical Core
Model Intercomparison Project (DCMIP-2012).  All tests use an isothermal
atmosphere (T = 300 K) with analytically prescribed winds.

Test 1-1: 3D Deformational Flow (Nair & Lauritzen 2010, extended to 3D)
Test 1-2: Hadley-like Meridional Circulation
Test 1-3: Horizontal Advection of Thin Clouds over Orography

References
----------
- Kent, Ullrich, Jablonowski (2014), QJRMS 140:1279-1293.
- Ullrich et al. (2012), DCMIP Test Case Document v1.7.
- Nair & Lauritzen (2010), JCP 229:8868-8887.
- Lauritzen et al. (2012), GMD 5:887-901.

The Fortran reference implementation is:
  dcmip_initial_conditions_test_1_2_3_v5.f90
  Authors: James Kent, Paul Ullrich, Christiane Jablonowski (U. Michigan)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import TracerState
from legoesm.core.operators import global_integral
from legoesm.grids.cubed_sphere import CubedSphereGrid, rotate_winds_geo_to_grid
from legoesm.grids.vertical import SigmaCoordinate, create_sigma_coordinate
from legoesm import constants

# ============================================================================
# Common constants (matching the DCMIP Fortran v5 reference)
# ============================================================================
_a = constants.R_earth        # 6371220.0 m
_Rd = 287.0                   # J/(kg*K) — DCMIP uses 287.0, not 287.05
_g = constants.g              # 9.80616 m/s^2
_cp = 1004.5                  # DCMIP value
_p0 = 1.0e5                  # Reference surface pressure (Pa)
_T0 = 300.0                  # Isothermal temperature (K)
_H = _Rd * _T0 / _g          # Scale height (m) ~ 8781.4
_ztop = 12000.0               # Model top (m)
_ptop = _p0 * jnp.exp(-_ztop / _H)


def create_dcmip_sigma(n_levels: int) -> SigmaCoordinate:
    """Create a sigma coordinate matching the DCMIP model domain.

    The DCMIP transport tests assume z_top = 12000 m, which for an
    isothermal atmosphere (T = 300 K) gives:

        sigma_top = exp(-z_top / H) ≈ 0.2553

    where H = R_d * T0 / g ≈ 8781.4 m.

    Using the default sigma_top=0.01 places the model top at ~40 km,
    far above the DCMIP domain, causing the prescribed wind fields to
    blow up.

    Parameters
    ----------
    n_levels : int
        Number of vertical levels.

    Returns
    -------
    SigmaCoordinate
    """
    sigma_top = float(jnp.exp(-_ztop / _H))  # ~0.2553
    return create_sigma_coordinate(n_levels, sigma_top=sigma_top)


def _height_from_sigma(sigma, ps=_p0):
    """Convert sigma coordinate to approximate height z = H ln(p0/p)."""
    p = sigma * ps
    return _H * jnp.log(_p0 / jnp.clip(p, 1e-30, None))


def _height_3d(sigma_coord: SigmaCoordinate, grid: CubedSphereGrid):
    """Height at full levels. Shape (nlev,) — no horizontal variation for flat surface."""
    return _height_from_sigma(sigma_coord.sigma_full)


# ============================================================================
# Test 1-1: 3D Deformational Flow
# ============================================================================

_tau11 = 12.0 * 86400.0       # Period = 12 days [s]
_u0_11 = 2.0 * jnp.pi * _a / _tau11
_k0 = 10.0 * _a / _tau11      # Deformation velocity magnitude
_omega0 = 23000.0 * jnp.pi / _tau11
_bs = 0.2                     # Shape function parameter (v5)

# Tracer centers
_lambda0 = 5.0 * jnp.pi / 6.0   # 150 deg
_lambda1 = 7.0 * jnp.pi / 6.0   # 210 deg
_phi0 = 0.0
_phi1 = 0.0
_rr = 0.5                     # Horizontal half-width / a (radians)
_zz = 1000.0                  # Vertical half-width (m)
_z0 = 5000.0                  # Vertical center (m)


def dcmip11_wind(
    t: float | jax.Array,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Prescribed wind for DCMIP Test 1-1 (3D deformational flow).

    Returns grid-aligned (u, v) and sigma_dot at interfaces.

    Parameters
    ----------
    t : float or scalar array
        Current time [s].
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate

    Returns
    -------
    u_grid, v_grid : jax.Array, shape (6, n, n, nlev)
    sigma_dot : jax.Array, shape (6, n, n, nlev+1)
    """
    lon = grid.lon  # (6, n, n)
    lat = grid.lat

    # Height at full levels: (nlev,) broadcast to (6, n, n, nlev)
    z_full = _height_from_sigma(sigma_coord.sigma_full)  # (nlev,)
    z = z_full[None, None, None, :]  # (1,1,1,nlev)

    # Pressure at full levels: p = sigma * p0 (flat surface, ps = p0)
    p = sigma_coord.sigma_full[None, None, None, :] * _p0  # (1,1,1,nlev)

    # Translating longitude
    lonp = lon[..., None] - 2.0 * jnp.pi * t / _tau11  # (6,n,n,1)

    # Shape function (v5)
    s = (1.0 + jnp.exp((_ptop - _p0) / (_bs * _ptop))
         - jnp.exp((p - _p0) / (_bs * _ptop))
         - jnp.exp((_ptop - p) / (_bs * _ptop)))

    # Divergent part of u (v5)
    ud = ((_omega0 * _a) / (_bs * _ptop)
          * jnp.cos(lonp) * jnp.cos(lat[..., None])**2
          * jnp.cos(2.0 * jnp.pi * t / _tau11)
          * (-jnp.exp((p - _p0) / (_bs * _ptop))
             + jnp.exp((_ptop - p) / (_bs * _ptop))))

    # Zonal wind (geographic east)
    u_east = (_k0 * jnp.sin(lonp)**2 * jnp.sin(2.0 * lat[..., None])
              * jnp.cos(jnp.pi * t / _tau11)
              + _u0_11 * jnp.cos(lat[..., None])
              + ud)

    # Meridional wind (geographic north)
    v_north = (_k0 * jnp.sin(2.0 * lonp) * jnp.cos(lat[..., None])
               * jnp.cos(jnp.pi * t / _tau11))

    # Vertical velocity w in z-coordinates (v5)
    w = ((-(_Rd * _T0) / (_g * p)) * _omega0
         * jnp.sin(lonp) * jnp.cos(lat[..., None])
         * jnp.cos(2.0 * jnp.pi * t / _tau11) * s)

    # Rotate geographic winds to grid-aligned
    # Need (6, n, n, nlev) angle: broadcast angle (6,n,n) to (6,n,n,nlev)
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle[..., None]
    )

    # Convert w to sigma_dot at interfaces
    # omega = -(g * p / (Rd * T0)) * w
    # sigma_dot = omega / ps = -(g / (Rd * T0)) * sigma * w_at_interfaces
    # But we need sigma_dot at half levels. Interpolate w to half levels first.
    # w is at full levels. Interpolate to half levels:
    w_half = _interpolate_to_half_levels(w, sigma_coord)  # (6,n,n,nlev+1)

    # sigma_dot = omega / p_s = -g * p_half / (Rd * T0 * p_s) * w_half
    # Since p_s = p0 and p_half = sigma_half * p0:
    # sigma_dot = -g * sigma_half * w_half / (Rd * T0)
    sigma_half = sigma_coord.sigma_half[None, None, None, :]  # (1,1,1,nlev+1)
    sigma_dot = -_g * sigma_half * w_half / (_Rd * _T0)

    # Enforce boundary conditions
    sigma_dot = sigma_dot.at[..., 0].set(0.0)
    sigma_dot = sigma_dot.at[..., -1].set(0.0)

    return u_grid, v_grid, sigma_dot


def _interpolate_to_half_levels(field_full, sigma_coord):
    """Linearly interpolate from full levels to half levels.

    field_full: shape (..., nlev)
    returns: shape (..., nlev+1)
    """
    nlev = sigma_coord.n_levels
    # Interior half-level k+1/2 = average of full levels k and k+1
    interior = 0.5 * (field_full[..., :-1] + field_full[..., 1:])  # (..., nlev-1)

    # Top boundary: extrapolate from first two full levels
    top = field_full[..., :1]  # (..., 1) — use top full level value

    # Bottom boundary: extrapolate from last two full levels
    bottom = field_full[..., -1:]  # (..., 1)

    return jnp.concatenate([top, interior, bottom], axis=-1)  # (..., nlev+1)


def dcmip11_init(
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> TracerState:
    """Initialize DCMIP Test 1-1 (4 tracers).

    Tracers:
      q1: 3D cosine bells
      q2: Correlated cosine bell (q2 = 0.9 - 0.8*q1^2)
      q3: Slotted ellipse
      q4: 1 - 0.3*(q1 + q2 + q3) (conservation check)

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate

    Returns
    -------
    TracerState with tracers shape (6, n, n, nlev, 4) and time = 0.
    """
    lon = grid.lon  # (6, n, n)
    lat = grid.lat
    n = lon.shape[1]
    nlev = sigma_coord.n_levels

    # Height at full levels
    z = _height_from_sigma(sigma_coord.sigma_full)  # (nlev,)
    z = z[None, None, None, :]  # (1,1,1,nlev)

    # Great-circle distances (in radians, i.e. without factor a)
    r1 = jnp.arccos(jnp.clip(
        jnp.sin(lat[..., None]) * jnp.sin(_phi0)
        + jnp.cos(lat[..., None]) * jnp.cos(_phi0) * jnp.cos(lon[..., None] - _lambda0),
        -1.0, 1.0
    ))
    r2 = jnp.arccos(jnp.clip(
        jnp.sin(lat[..., None]) * jnp.sin(_phi1)
        + jnp.cos(lat[..., None]) * jnp.cos(_phi1) * jnp.cos(lon[..., None] - _lambda1),
        -1.0, 1.0
    ))

    # Combined distance (horizontal + vertical) — Fortran: d = min(1, (r/rr)^2 + ((z-z0)/zz)^2)
    d1 = jnp.minimum(1.0, (r1 / _rr)**2 + ((z - _z0) / _zz)**2)
    d2 = jnp.minimum(1.0, (r2 / _rr)**2 + ((z - _z0) / _zz)**2)

    # q1: 3D cosine bells
    q1 = 0.5 * (1.0 + jnp.cos(jnp.pi * d1)) + 0.5 * (1.0 + jnp.cos(jnp.pi * d2))

    # q2: Correlated cosine bell
    q2 = 0.9 - 0.8 * q1**2

    # q3: Slotted ellipse
    # Inside the ellipse: q3 = 1, outside: q3 = 0.1
    q3 = jnp.where(d1 <= _rr, 1.0, jnp.where(d2 <= _rr, 1.0, 0.1))
    # Slot: remove where z > z0 and |lat| < 0.125 rad
    q3 = jnp.where(
        (z > _z0) & (jnp.abs(lat[..., None]) < 0.125),
        0.1, q3
    )

    # q4: conservation tracer
    q4 = 1.0 - 0.3 * (q1 + q2 + q3)

    # Stack tracers: (6, n, n, nlev, 4)
    tracers_data = jnp.stack([q1, q2, q3, q4], axis=-1)

    tracers = Field(
        data=tracers_data,
        name="tracers",
        dims=("face", "x", "y", "level", "tracer"),
        units="kg/kg",
    )
    time = Field(data=jnp.array(0.0), name="time", dims=(), units="s")

    return TracerState(tracers=tracers, time=time)


# ============================================================================
# Test 1-2: Hadley-like Meridional Circulation
# ============================================================================

_tau12 = 1.0 * 86400.0        # Period = 1 day [s]
_u0_12 = 40.0                 # Zonal wind amplitude [m/s]
_w0_12 = 0.15                 # Vertical velocity amplitude [m/s] (v5)
_K12 = 5.0                    # Number of Hadley cells
_z1_12 = 2000.0               # Lower tracer bound [m] (v5)
_z2_12 = 5000.0               # Upper tracer bound [m] (v5)
_z0_12 = 0.5 * (_z1_12 + _z2_12)  # Midpoint
_ztop12 = 12000.0


def dcmip12_wind(
    t: float | jax.Array,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Prescribed wind for DCMIP Test 1-2 (Hadley-like circulation).

    Parameters
    ----------
    t : float or scalar array
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate

    Returns
    -------
    u_grid, v_grid : (6, n, n, nlev)
    sigma_dot : (6, n, n, nlev+1)
    """
    lon = grid.lon
    lat = grid.lat

    z_full = _height_from_sigma(sigma_coord.sigma_full)  # (nlev,)
    z = z_full[None, None, None, :]
    p = sigma_coord.sigma_full[None, None, None, :] * _p0

    # Reference density ratio: rho0/rho = p0/(p) since T is constant
    # rho = p / (Rd * T0), rho0 = p0 / (Rd * T0)
    rho_ratio = _p0 / p  # (1,1,1,nlev)

    cos_t = jnp.cos(jnp.pi * t / _tau12)

    # Zonal wind (geographic)
    u_east = _u0_12 * jnp.cos(lat[..., None]) * jnp.ones_like(z)

    # Meridional wind (v5: multiply by rho0/rho)
    v_north = (-rho_ratio * (_a * _w0_12 * jnp.pi) / (_K12 * _ztop12)
               * jnp.cos(lat[..., None])
               * jnp.sin(_K12 * lat[..., None])
               * jnp.cos(jnp.pi * z / _ztop12)
               * cos_t)

    # Vertical velocity w (v5: multiply by rho0/rho)
    w = (rho_ratio * (_w0_12 / _K12)
         * (-2.0 * jnp.sin(_K12 * lat[..., None]) * jnp.sin(lat[..., None])
            + _K12 * jnp.cos(lat[..., None]) * jnp.cos(_K12 * lat[..., None]))
         * jnp.sin(jnp.pi * z / _ztop12)
         * cos_t)

    # Rotate to grid-aligned
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle[..., None]
    )

    # Convert w to sigma_dot
    w_half = _interpolate_to_half_levels(w, sigma_coord)
    sigma_half = sigma_coord.sigma_half[None, None, None, :]
    sigma_dot = -_g * sigma_half * w_half / (_Rd * _T0)
    sigma_dot = sigma_dot.at[..., 0].set(0.0)
    sigma_dot = sigma_dot.at[..., -1].set(0.0)

    return u_grid, v_grid, sigma_dot


def dcmip12_init(
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> TracerState:
    """Initialize DCMIP Test 1-2 (1 tracer: vertical layer).

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate

    Returns
    -------
    TracerState with tracers shape (6, n, n, nlev, 1) and time = 0.
    """
    z_full = _height_from_sigma(sigma_coord.sigma_full)
    z = z_full[None, None, None, :]

    n = grid.lon.shape[1]
    nlev = sigma_coord.n_levels

    # Tracer: cosine-bell layer between z1 and z2
    q1 = jnp.where(
        (z > _z1_12) & (z < _z2_12),
        0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * (z - _z0_12) / (_z2_12 - _z1_12))),
        0.0,
    )
    # Broadcast to full shape
    q1 = jnp.broadcast_to(q1, (6, n, n, nlev))

    tracers_data = q1[..., None]  # (6, n, n, nlev, 1)

    tracers = Field(
        data=tracers_data,
        name="tracers",
        dims=("face", "x", "y", "level", "tracer"),
        units="kg/kg",
    )
    time = Field(data=jnp.array(0.0), name="time", dims=(), units="s")

    return TracerState(tracers=tracers, time=time)


# ============================================================================
# Test 1-3: Horizontal Advection over Orography
# ============================================================================

_tau13 = 12.0 * 86400.0       # Period = 12 days [s]
_u0_13 = 2.0 * jnp.pi * _a / _tau13
_alpha13 = jnp.pi / 6.0       # Rotation angle = 30 deg
_lambdam = 3.0 * jnp.pi / 2.0  # Mountain center longitude (270 deg)
_phim = 0.0                    # Mountain center latitude (equator)
_h0_mt = 2000.0                # Peak mountain height (m)
_rm = 3.0 * jnp.pi / 4.0      # Mountain radius (radians)
_zetam = jnp.pi / 16.0         # Mountain oscillation half-width

# Cloud tracer parameters
_lambdap = jnp.pi / 2.0       # Cloud center longitude (90 deg)
_phip = 0.0                    # Cloud center latitude
_rp = jnp.pi / 4.0            # Cloud radius (radians)
_zp1 = 3050.0                 # Midpoint of tracer 1 (m)
_zp2 = 5050.0                 # Midpoint of tracer 2 (m)
_zp3 = 8200.0                 # Midpoint of tracer 3 (m)
_dzp1 = 1000.0                # Thickness of tracer 1 (m)
_dzp2 = 1000.0                # Thickness of tracer 2 (m)
_dzp3 = 400.0                 # Thickness of tracer 3 (m)


def _mountain_height(lon, lat):
    """Compute mountain surface elevation z_s (Schaer-type with oscillation).

    Parameters
    ----------
    lon, lat : jax.Array, shape (6, n, n)

    Returns
    -------
    zs : jax.Array, shape (6, n, n)
    """
    r = jnp.arccos(jnp.clip(
        jnp.sin(_phim) * jnp.sin(lat)
        + jnp.cos(_phim) * jnp.cos(lat) * jnp.cos(lon - _lambdam),
        -1.0, 1.0
    ))
    zs = jnp.where(
        r < _rm,
        (_h0_mt / 2.0) * (1.0 + jnp.cos(jnp.pi * r / _rm)) * jnp.cos(jnp.pi * r / _zetam)**2,
        0.0,
    )
    return zs


def dcmip13_wind(
    t: float | jax.Array,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Prescribed wind for DCMIP Test 1-3 (solid-body rotation over mountain).

    The horizontal wind is time-independent solid-body rotation with an
    inclination angle alpha. The physical vertical velocity w = 0, but the
    terrain-following coordinate generates a non-zero sigma_dot as flow
    crosses the mountain.

    Note: Since we use pure sigma coordinates (not hybrid-eta), the
    terrain-following sigma_dot is computed during advection via the
    continuity equation. Here we set sigma_dot = 0 because w = 0
    everywhere — the effective vertical advection in sigma coordinates
    is implicitly handled through the sigma-coordinate metric terms.

    For a simple sigma-coordinate model, sigma_dot from orographic flow
    is diagnosed from the horizontal divergence via compute_sigma_dot,
    which is already called by the transport model. The divergence-free
    flow + orography produces the correct sigma_dot.

    Parameters
    ----------
    t : float or scalar array
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate

    Returns
    -------
    u_grid, v_grid : (6, n, n, nlev)
    sigma_dot : (6, n, n, nlev+1) — all zeros for w=0 in physical space
    """
    lon = grid.lon
    lat = grid.lat
    nlev = sigma_coord.n_levels

    # Solid-body rotation (geographic)
    u_east = _u0_13 * (jnp.cos(lat) * jnp.cos(_alpha13)
                       + jnp.sin(lat) * jnp.cos(lon) * jnp.sin(_alpha13))
    v_north = -_u0_13 * jnp.sin(lon) * jnp.sin(_alpha13)

    # Expand to 3D: (6, n, n) -> (6, n, n, nlev)
    u_east_3d = u_east[..., None] * jnp.ones(nlev)
    v_north_3d = v_north[..., None] * jnp.ones(nlev)

    # Rotate to grid-aligned
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east_3d, v_north_3d, grid.angle[..., None]
    )

    # sigma_dot = 0 (w = 0 in physical space; sigma_dot from orographic
    # flow should be computed by the model's continuity equation)
    sigma_dot = jnp.zeros((*lon.shape, nlev + 1))

    return u_grid, v_grid, sigma_dot


def dcmip13_init(
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
) -> TracerState:
    """Initialize DCMIP Test 1-3 (4 tracers: cloud layers over orography).

    Tracers:
      q1: Cloud at z = 3050 m, thickness 1000 m
      q2: Cloud at z = 5050 m, thickness 1000 m
      q3: Thin cloud at z = 8200 m, thickness 400 m (step function)
      q4: q1 + q2 + q3

    Note: This test uses orography, so height depends on horizontal
    position. For sigma coordinates, z = z_s + sigma * (z_top - z_s)
    where z_s is the surface elevation.

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate

    Returns
    -------
    TracerState with tracers shape (6, n, n, nlev, 4) and time = 0.
    """
    lon = grid.lon
    lat = grid.lat
    n = lon.shape[1]
    nlev = sigma_coord.n_levels

    # Surface elevation
    zs = _mountain_height(lon, lat)  # (6, n, n)

    # Height at full levels, accounting for orography:
    # In pressure-based isothermal: p = ps * sigma, ps = p0 * exp(-zs/H)
    # z = H * ln(p0/p) = H * ln(p0 / (sigma * ps)) = H * ln(p0/(sigma * p0 * exp(-zs/H)))
    #   = H * (ln(1/sigma) + zs/H) = -H * ln(sigma) + zs
    # But actually: for isothermal atmosphere p = p0 * exp(-z/H)
    # so z = H * ln(p0/p). With p = sigma * ps and ps = p0 * exp(-zs/H):
    # z = H * ln(p0 / (sigma * p0 * exp(-zs/H))) = H * (-ln(sigma) + zs/H) = -H*ln(sigma) + zs
    sigma_full = sigma_coord.sigma_full  # (nlev,)
    z = -_H * jnp.log(sigma_full)[None, None, None, :] + zs[..., None]  # (6,n,n,nlev)

    # Great-circle distance from cloud center
    r = jnp.arccos(jnp.clip(
        jnp.sin(_phip) * jnp.sin(lat)
        + jnp.cos(_phip) * jnp.cos(lat) * jnp.cos(lon - _lambdap),
        -1.0, 1.0
    ))
    r = r[..., None]  # (6, n, n, 1)

    # q1: Cloud at zp1
    rz1 = jnp.abs(z - _zp1)
    q1 = jnp.where(
        (rz1 < 0.5 * _dzp1) & (r < _rp),
        0.25 * (1.0 + jnp.cos(2.0 * jnp.pi * rz1 / _dzp1)) * (1.0 + jnp.cos(jnp.pi * r / _rp)),
        0.0,
    )

    # q2: Cloud at zp2
    rz2 = jnp.abs(z - _zp2)
    q2 = jnp.where(
        (rz2 < 0.5 * _dzp2) & (r < _rp),
        0.25 * (1.0 + jnp.cos(2.0 * jnp.pi * rz2 / _dzp2)) * (1.0 + jnp.cos(jnp.pi * r / _rp)),
        0.0,
    )

    # q3: Thin cloud at zp3 (step function)
    rz3 = jnp.abs(z - _zp3)
    q3 = jnp.where(
        (rz3 < 0.5 * _dzp3) & (r < _rp),
        1.0,
        0.0,
    )

    # q4: Sum of all clouds
    q4 = q1 + q2 + q3

    tracers_data = jnp.stack([q1, q2, q3, q4], axis=-1)

    tracers = Field(
        data=tracers_data,
        name="tracers",
        dims=("face", "x", "y", "level", "tracer"),
        units="kg/kg",
    )
    time = Field(data=jnp.array(0.0), name="time", dims=(), units="s")

    return TracerState(tracers=tracers, time=time)


# ============================================================================
# Error norms
# ============================================================================

def compute_tracer_error_norms(
    state_final: TracerState,
    state_initial: TracerState,
    grid: CubedSphereGrid,
) -> dict[str, jax.Array]:
    """Compute normalized error norms for each tracer.

    For flow-reversal tests (1-1, 1-2), the exact solution at t = T
    is the initial condition.

    Returns area-weighted l1, l2, linf norms for each tracer.

    Parameters
    ----------
    state_final : TracerState
    state_initial : TracerState
    grid : CubedSphereGrid

    Returns
    -------
    dict with keys 'l1', 'l2', 'linf', each of shape (n_tracers,).
    """
    q_final = state_final.tracers.data     # (6, n, n, nlev, n_tracers)
    q_exact = state_initial.tracers.data

    error = q_final - q_exact  # (6, n, n, nlev, n_tracers)
    n_tracers = error.shape[-1]

    # Area weights: grid.area is (6, n, n)
    area = grid.area  # (6, n, n)

    # For each tracer, compute area-weighted norms (sum over faces, x, y, levels)
    l1_norms = []
    l2_norms = []
    linf_norms = []

    for i in range(n_tracers):
        err_i = error[..., i]    # (6, n, n, nlev)
        exact_i = q_exact[..., i]

        # Sum over levels, then area-weight horizontal
        # l1 = sum(|err| * area) / sum(|exact| * area)
        err_area = jnp.sum(jnp.abs(err_i), axis=-1) * area  # (6, n, n)
        exact_area = jnp.sum(jnp.abs(exact_i), axis=-1) * area

        l1 = jnp.sum(err_area) / jnp.maximum(jnp.sum(exact_area), 1e-30)

        # l2 = sqrt(sum(err^2 * area) / sum(exact^2 * area))
        err2_area = jnp.sum(err_i**2, axis=-1) * area
        exact2_area = jnp.sum(exact_i**2, axis=-1) * area
        l2 = jnp.sqrt(jnp.sum(err2_area) / jnp.maximum(jnp.sum(exact2_area), 1e-30))

        # linf = max|err| / max|exact|
        linf = jnp.max(jnp.abs(err_i)) / jnp.maximum(jnp.max(jnp.abs(exact_i)), 1e-30)

        l1_norms.append(l1)
        l2_norms.append(l2)
        linf_norms.append(linf)

    return {
        'l1': jnp.stack(l1_norms),
        'l2': jnp.stack(l2_norms),
        'linf': jnp.stack(linf_norms),
    }
