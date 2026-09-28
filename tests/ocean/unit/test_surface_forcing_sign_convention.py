"""Pin the (intentional, documented) wind-stress sign-convention divergence
between the ``prescribed`` and ``external`` surface-forcing schemes.

``prescribed_surface_forcing`` applies +tau as an ON-OCEAN stress
(``du/dt = +tau_x/(rho_0 dz_0)``); ``external_surface_forcing`` applies the
ATMOSPHERE convention (ocean reaction ``= -tau``,
``du/dt = -tau_x/(rho_0 dz_0)``). For the SAME input wind stress the two
therefore accelerate the ocean in OPPOSITE directions — see the explicit note
in ``external.py`` ("NOTE this is the OPPOSITE tau sign of the ``prescribed``
scheme").

This is by design (``external`` is the coupler / atmosphere-convention path),
but it is a documented bug-magnet: feeding an on-ocean stress to ``external``
(or an atmosphere-convention stress to ``prescribed``) silently drives the
ocean backward, and no runtime guard can detect the mismatch from the data
alone (the sign of tau is not self-describing). This test locks BOTH signs and
their exact opposition so an accidental flip in either scheme — or an
"accidental unification" that breaks the coupler — fails loudly.

Heat is deliberately NOT contrasted here: both schemes deposit ``q_net > 0``
INTO the ocean (same sign); only the momentum convention diverges.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import surface_stress_faces
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.external import (
    external_surface_forcing,
)
from legoesm.ocean.physics.surface_forcing.prescribed import (
    prescribed_surface_forcing,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.ocean.vertical import create_ocean_z_star

N_LAT, N_LON, NLEV = 4, 5, 3
TAU_X = 0.1  # Pa, eastward (positive)


def _fields():
    """A-grid-shaped fields (u/v share T's 2-D shape → no C-grid interpolation,
    so the surface du/dt sign is read directly), all-ocean (J=1)."""
    z = create_ocean_z_star(n_levels=NLEV, H_max=1000.0)
    shape3 = (N_LAT, N_LON, NLEV)
    u = jnp.zeros(shape3)
    v = jnp.zeros(shape3)
    T = jnp.zeros(shape3)
    S = 35.0 * jnp.ones(shape3)
    jacobian = jnp.ones((N_LAT, N_LON))
    return z, u, v, T, S, jacobian


def test_prescribed_applies_positive_tau_as_on_ocean_stress():
    """prescribed: eastward stress (+tau_x) → eastward acceleration (du/dt>0),
    and ONLY the surface layer is forced."""
    z, u, v, T, S, jac = _fields()
    grid = SimpleNamespace(grid_lat=jnp.zeros((N_LAT, N_LON)))
    cfg = PrescribedForcingConfig(tau_x=TAU_X, wind_profile="constant")
    out = prescribed_surface_forcing(u, v, T, S, z, jac, grid, cfg)
    assert float(out.du_dt[..., 0].mean()) > 0.0
    assert np.allclose(np.asarray(out.du_dt[..., 1:]), 0.0)


def test_external_applies_negative_tau_atmosphere_convention():
    """external: the SAME eastward stress (+tau_x) → westward acceleration
    (du/dt<0), the atmosphere/coupler convention."""
    z, u, v, T, S, jac = _fields()
    dz_0 = z.dz_ref[0] * jac
    sf = OceanSurfaceForcing(tau_x=TAU_X * jnp.ones((N_LAT, N_LON)))
    out = external_surface_forcing(u, v, T, S, dz_0, sf)
    assert float(out.du_dt[..., 0].mean()) < 0.0
    assert np.allclose(np.asarray(out.du_dt[..., 1:]), 0.0)


def test_prescribed_and_external_momentum_are_exactly_opposite():
    """The footgun, pinned: identical wind stress → equal-magnitude,
    opposite-sign surface acceleration. We feed ``external`` the EXACT tau that
    ``prescribed`` computed, so the opposition is exact regardless of any
    internal wind-profile scaling."""
    z, u, v, T, S, jac = _fields()
    grid = SimpleNamespace(grid_lat=jnp.zeros((N_LAT, N_LON)))
    cfg = PrescribedForcingConfig(tau_x=TAU_X, wind_profile="constant")
    dz_0 = z.dz_ref[0] * jac

    out_p = prescribed_surface_forcing(u, v, T, S, z, jac, grid, cfg)
    # Identical stress field into the external (atmosphere-convention) path.
    sf = OceanSurfaceForcing(tau_x=jnp.broadcast_to(
        out_p.tau_x, (N_LAT, N_LON)))
    out_e = external_surface_forcing(u, v, T, S, dz_0, sf)

    du_p = np.asarray(out_p.du_dt[..., 0])
    du_e = np.asarray(out_e.du_dt[..., 0])
    assert float(np.abs(du_p).max()) > 0.0, "vacuous: prescribed accel is zero"
    assert np.allclose(du_p, -du_e), (
        "prescribed (+tau, on-ocean) and external (-tau, atmosphere) must be "
        "EXACT opposites for the same stress; a sign change in either scheme "
        "(or an accidental unification) breaks coupled forcing")


def test_native_ocean_stress_bypasses_lossy_geographic_round_trip():
    """A source-native pair is the exact operand used for face interpolation."""
    z, _, _, _, _, jac = _fields()
    tau_i = jnp.asarray([
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.3, 0.5, 0.7, 0.9, 1.1],
    ], dtype=jnp.float64)
    tau_j = -tau_i
    geographic_poison = jnp.full_like(tau_i, 99.0)
    forcing = OceanSurfaceForcing(
        tau_x=geographic_poison,
        tau_y=geographic_poison,
        tau_i_native=tau_i,
        tau_j_native=tau_j,
    )
    grid = SimpleNamespace(fold=None)
    tau_u, tau_v, _, _ = surface_stress_faces(
        forcing, jnp.float64, z, jac, grid)

    padded = np.pad(np.asarray(tau_i), ((0, 0), (1, 1)), mode="wrap")
    expected_u = 0.5 * (padded[:, :-1] + padded[:, 1:])
    np.testing.assert_array_equal(np.asarray(tau_u), expected_u)
    np.testing.assert_array_equal(
        np.asarray(tau_v)[1:-1],
        0.5 * (np.asarray(tau_j)[:-1] + np.asarray(tau_j)[1:]),
    )
    assert not np.any(np.asarray(tau_u) == -99.0)


def test_native_ocean_stress_pair_is_fail_closed():
    z, _, _, _, _, jac = _fields()
    forcing = OceanSurfaceForcing(tau_i_native=jnp.ones((N_LAT, N_LON)))
    with np.testing.assert_raises_regex(ValueError, "must be supplied together"):
        surface_stress_faces(
            forcing, jnp.float64, z, jac, SimpleNamespace(fold=None))


def test_native_ocean_stress_applies_nemo_coastal_unmasking_literally():
    """sbcmod.F90:543-546 doubles a wet/dry coastal face after averaging."""
    tmask = np.asarray([
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    active = jnp.asarray(tmask[..., None] > 0.5)
    z = SimpleNamespace(is_active=active, dz_ref=jnp.ones((1,)))
    grid = SimpleNamespace(fold=None, seam_wall_rows=None)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.ones(tmask.shape),
        tau_j_native=jnp.ones(tmask.shape),
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d

    umask, vmask = compute_face_masks_3d(active, grid)
    tau_u, tau_v, _, _ = surface_stress_faces(
        forcing, jnp.float64, z, jnp.ones(tmask.shape), grid,
        cell_mask=jnp.asarray(tmask), u_mask=umask[..., 0],
        v_mask=vmask[..., 0],
    )
    lon = np.pad(tmask, ((0, 0), (1, 1)), mode="wrap")
    tmax_u = np.maximum(lon[:, :-1], lon[:, 1:])
    lat = np.pad(tmask, ((1, 1), (0, 0)), mode="constant")
    tmax_v = np.maximum(lat[:-1], lat[1:])
    expected_u = (2.0 - np.asarray(umask[..., 0])) * tmax_u
    expected_v = (2.0 - np.asarray(vmask[..., 0])) * tmax_v
    # The shared C-grid interpolation applies the physical meridional wall
    # before sbcmod's factor; multiplying cannot revive those boundary faces.
    expected_v[[0, -1], :] = 0.0
    np.testing.assert_array_equal(np.asarray(tau_u), expected_u)
    np.testing.assert_array_equal(np.asarray(tau_v), expected_v)
    assert np.any(expected_u == 2.0), "control has no doubled coastal U face"
    assert np.any(expected_v == 2.0), "control has no doubled coastal V face"
