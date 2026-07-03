"""Energy-conserving C-grid Coriolis (vertex-f Sadourny form).

Pins the property that motivates it: on a β-plane the vertex-f Coriolis does NO
spurious work (Σ u·cor_u + Σ v·cor_v == 0 to machine precision), unlike the
default face-f ``coriolis_cgrid`` which leaks ~1e-6·f·KE because f_u != f_v.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    coriolis_cgrid,
    coriolis_cgrid_energy_conserving,
)

NY = NX = 24
DX = 20.0e3


def _geom(beta):
    return create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX, dy_m=DX, f0=1e-4, beta=beta, y_origin_m=-20e3
    )


def _masks():
    mask = np.ones((NY, NX))
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0.0
    um = np.zeros((NY, NX + 1))
    vm = np.zeros((NY + 1, NX))
    um[:, 1:-1] = mask[:, :-1] * mask[:, 1:]
    vm[1:-1, :] = mask[:-1, :] * mask[1:, :]
    return jnp.asarray(um), jnp.asarray(vm)


def _power(u, v, cu, cv, um, vm):
    return float(jnp.sum(u * cu * um) + jnp.sum(v * cv * vm))


def test_energy_conserving_on_beta_plane():
    """Vertex-f Coriolis does no net work on random β-plane fields."""
    g = _geom(1e-11)
    um, vm = _masks()
    rng = np.random.default_rng(0)
    fscale = 1e-4
    for _ in range(5):
        u = jnp.asarray(rng.standard_normal((NY, NX + 1))) * um
        v = jnp.asarray(rng.standard_normal((NY + 1, NX))) * vm
        cu, cv = coriolis_cgrid_energy_conserving(u, v, g, u_mask=um, v_mask=vm)
        ke = float(jnp.sum(u * u * um) + jnp.sum(v * v * vm))
        assert abs(_power(u, v, cu, cv, um, vm)) / (fscale * ke) < 1e-12


def test_face_f_form_leaks_on_beta_plane():
    """Non-vacuous companion: the default face-f form DOES leak on a β-plane
    (so the energy-conserving variant is solving a real defect)."""
    g = _geom(1e-11)
    um, vm = _masks()
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.standard_normal((NY, NX + 1))) * um
    v = jnp.asarray(rng.standard_normal((NY + 1, NX))) * vm
    cu, cv = coriolis_cgrid(u, v, g, u_mask=um, v_mask=vm)
    ke = float(jnp.sum(u * u * um) + jnp.sum(v * v * vm))
    assert abs(_power(u, v, cu, cv, um, vm)) / (1e-4 * ke) > 1e-9


def test_reduces_to_face_f_on_f_plane():
    """On an f-plane (β=0) the vertex-f and face-f forms agree (same f)."""
    g = _geom(0.0)
    um, vm = _masks()
    rng = np.random.default_rng(2)
    u = jnp.asarray(rng.standard_normal((NY, NX + 1))) * um
    v = jnp.asarray(rng.standard_normal((NY + 1, NX))) * vm
    cu_e, cv_e = coriolis_cgrid_energy_conserving(u, v, g, u_mask=um, v_mask=vm)
    cu_f, cv_f = coriolis_cgrid(u, v, g, u_mask=um, v_mask=vm)
    np.testing.assert_allclose(np.asarray(cu_e), np.asarray(cu_f), atol=1e-12)
    np.testing.assert_allclose(np.asarray(cv_e), np.asarray(cv_f), atol=1e-12)


def test_uniform_flow_gives_local_f():
    """Interior cor_u for uniform v=1,u=0 is ~f (local Coriolis), sign +f·v."""
    g = _geom(1e-11)
    um, vm = _masks()
    u = jnp.zeros((NY, NX + 1)) * um
    v = jnp.ones((NY + 1, NX)) * vm
    cu, _ = coriolis_cgrid_energy_conserving(u, v, g, u_mask=um, v_mask=vm)
    fu = np.asarray(g.f_u)
    cu = np.asarray(cu)
    # deep interior (away from walls) cor_u ~ f_u * v(=1)
    sl = (slice(4, NY - 4), slice(4, NX - 3))
    np.testing.assert_allclose(cu[sl], fu[sl], rtol=0.02)


def test_3d_matches_2d_per_level():
    """3D input (broadcast over levels) equals the 2D op per level."""
    g = _geom(1e-11)
    um, vm = _masks()
    rng = np.random.default_rng(3)
    u2 = jnp.asarray(rng.standard_normal((NY, NX + 1))) * um
    v2 = jnp.asarray(rng.standard_normal((NY + 1, NX))) * vm
    u3 = jnp.stack([u2, 2.0 * u2], axis=-1)
    v3 = jnp.stack([v2, 2.0 * v2], axis=-1)
    cu3, cv3 = coriolis_cgrid_energy_conserving(u3, v3, g, u_mask=um, v_mask=vm)
    cu2, cv2 = coriolis_cgrid_energy_conserving(u2, v2, g, u_mask=um, v_mask=vm)
    np.testing.assert_allclose(np.asarray(cu3[..., 0]), np.asarray(cu2), atol=1e-12)
    np.testing.assert_allclose(np.asarray(cv3[..., 1]), 2.0 * np.asarray(cv2), atol=1e-12)
