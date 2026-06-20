"""Colliding modons — nonlinear shallow-water test (issue #521).

Twin-vortex "soliton" test (GFDL FV3 ``test_cases.F90`` case-8; the JAMES
"Colliding Modons" nonlinear dynamical-core test, doi:10.1002/2017MS000965).
Two localized zonal-wind Gaussians of opposite sign sit on the equator a
half-circumference apart on a NON-ROTATING planet (f=0) with a flat free
surface.  Under nonlinear shallow-water dynamics each adjusts into a modon
(dipole) that propagates; the westerly and easterly pairs collide near a cube
face boundary, exchange vortices, depart in opposite directions, and return to
their initial positions after ~100 days — a stringent test of cube-seam
fidelity (no spurious vortex generation / asymmetry at the corners).

Faithful to FV3 case-8 (``tools/test_cases.F90`` L1377-1452):
  f0 = fC = 0,  phis = 0,  delp = gh0 = 5e3*g  (uniform free surface, h=5000 m)
  ubar = soliton_Umax = 50 m/s,  r0 = soliton_size = 750 km,  Nsolitons = 2
  westerly centre p0 = (pi/2, 0); easterly centre p0 = (pi/2+pi, 0)
  u = ubar*exp(-(r/r0)^2) projected onto the D-grid edge (zonal wind, v=0).

The FV3 inner-product projection of the geographic east vector onto the D-grid
edge direction is exactly the ``cos_angle_edge_*`` / ``sin_angle_edge_*`` metric
rotation already used by ``cosine_bell_cubesphere`` / the Williamson-2 cube IC,
so it is reused here (no geometry re-derivation).
"""
from __future__ import annotations

import jax.numpy as jnp

# --- FV3 case-8 namelist constants (tools/test_cases.F90 L111-112, L1382-1385) ---
_UBAR = 50.0            # soliton_Umax — peak zonal wind [m/s]
_R0 = 750.0e3           # soliton_size — Gaussian e-folding radius [m]
_H0 = 5.0e3            # gh0/g — uniform free-surface height [m]
# centres on the equator, a half-circumference apart; sign = jet direction.
_CENTRES = ((jnp.pi * 0.5, 0.0, +1.0),       # #1 westerly
            (jnp.pi * 0.5 + jnp.pi, 0.0, -1.0))  # #2 easterly (subtracted)


def _gc_dist(lon, lat, lon0, lat0, radius):
    """Great-circle distance [m] from (lon0,lat0) to each (lon,lat) (spherical
    law of cosines; same form as ``cosine_bell._cosine_bell``)."""
    cosd = (jnp.sin(lat0) * jnp.sin(lat)
            + jnp.cos(lat0) * jnp.cos(lat) * jnp.cos(lon - lon0))
    return radius * jnp.arccos(jnp.clip(cosd, -1.0, 1.0))


def modon_zonal_wind(lon, lat, radius):
    """Geographic zonal wind u_east [m/s] of the twin modons (v_north = 0).

    u_east = sum_centres sign * ubar * exp(-(r/r0)^2),  r = great-circle dist.
    """
    u_east = jnp.zeros_like(lon)
    for lon0, lat0, sign in _CENTRES:
        r = _gc_dist(lon, lat, lon0, lat0, radius)
        u_east = u_east + sign * _UBAR * jnp.exp(-(r / _R0) ** 2)
    return u_east


def colliding_modons_cubesphere(grid, cdgrid):
    """Colliding-modon initial condition on the FV3 edge-midpoint C-D grid.

    Returns ``(state, cdgrid_nonrot)`` where ``cdgrid_nonrot`` has the Coriolis
    field zeroed (f=0, FV3 case-8 ``f0=fC=0``); assign it to ``model.cdgrid``
    before stepping so the run is non-rotating.
    """
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState)

    R = grid.radius
    h = jnp.full_like(grid.lon, _H0)
    h_s = jnp.zeros_like(h)

    # zonal wind at the two D-grid edge-midpoint staggers, rotated to grid frame
    u_e_x = modon_zonal_wind(cdgrid.lon_edge_x, cdgrid.lat_edge_x, R)
    u_d = cdgrid.cos_angle_edge_x * u_e_x          # v_north = 0
    u_e_y = modon_zonal_wind(cdgrid.lon_edge_y, cdgrid.lat_edge_y, R)
    v_d = -cdgrid.sin_angle_edge_y * u_e_y

    state = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
    cdgrid_nonrot = cdgrid._replace(f_corner=jnp.zeros_like(cdgrid.f_corner))
    return state, cdgrid_nonrot
