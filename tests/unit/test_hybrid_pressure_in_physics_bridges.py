"""On the CAM6 L32 hybrid grid the turbulence and microphysics bridges must
hand their schemes ``coord.pressure_at_full(p_s)``, not ``p_s * sigma_full``
(the hybrid's ``sigma_full`` is the ``A_full + B_full`` compatibility view).

Observed at the seam: the bridges compute density via the shared
``compute_rho(T, p_full, ...)`` and heights via ``compute_heights_from_sigma /
compute_layer_dz(T, p_half, ...)``; both are monkeypatched to record the
pressure they receive.  With the fix reverted the recorded pressure is
``p_s * (A_full + B_full)`` and the assertion fails wherever ``p_s != p_ref``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.microphysics import integration as micro_int
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.turbulence import integration as turb_int
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.grids.vertical import make_cam6_l32_levels

P_S = 9.0e4  # != p_ref = 1e5, so A*p_ref + B*p_s differs from (A+B)*p_s


@pytest.fixture(scope="module")
def setup():
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    coord = make_cam6_l32_levels()
    state = held_suarez_init_mpas(mesh, coord, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    state = state._replace(
        p_s=state.p_s.replace(data=jnp.full((ncell,), P_S, state.p_s.data.dtype)),
        tracers={"q_v": jnp.full((ncell, nlev), 0.004, state.T.data.dtype),
                 "q_c": jnp.zeros((ncell, nlev), state.T.data.dtype),
                 "q_r": jnp.zeros((ncell, nlev), state.T.data.dtype)})
    return mesh, coord, state


def _spy(monkeypatch, module, name, seen):
    orig = getattr(module, name)

    def rec(T, p, *a, **kw):
        seen.append(np.asarray(p))
        return orig(T, p, *a, **kw)
    monkeypatch.setattr(module, name, rec)


def _check(seen_full, seen_half, coord, p_s, ncell, nlev):
    assert seen_full and seen_half, "bridge did not reach the shared helpers"
    want_full = np.asarray(coord.pressure_at_full(p_s)).reshape(ncell, nlev)
    want_half = np.asarray(coord.pressure_at_half(p_s)).reshape(ncell, nlev + 1)
    wrong_full = np.asarray(p_s)[:, None] * np.asarray(coord.sigma_full)[None, :]
    assert np.abs(want_full - wrong_full).max() > 100.0, "fixture cannot discriminate"
    np.testing.assert_allclose(seen_full[0], want_full, rtol=0, atol=1e-6)
    np.testing.assert_allclose(seen_half[0], want_half, rtol=0, atol=1e-6)


def test_mpas_turbulence_bridge_uses_hybrid_pressure(setup, monkeypatch):
    mesh, coord, state = setup
    seen_full, seen_half = [], []
    _spy(monkeypatch, turb_int, "_compute_rho", seen_full)
    _spy(monkeypatch, turb_int, "_compute_heights_from_sigma", seen_half)
    fn = turb_int._make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0)
    ncell, nlev = state.T.data.shape
    fn(state, mesh, coord, forcing={"T_sfc": jnp.full((ncell,), 292.0)})
    _check(seen_full, seen_half, coord, state.p_s.data, ncell, nlev)


def test_microphysics_bridge_uses_hybrid_pressure(setup, monkeypatch):
    mesh, coord, state = setup
    seen_full, seen_half = [], []
    _spy(monkeypatch, micro_int, "_compute_rho", seen_full)
    _spy(monkeypatch, micro_int, "_compute_heights_from_sigma", seen_half)
    fn = micro_int.make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler"), model_type="mpas", dt=300.0)
    ncell, nlev = state.T.data.shape
    fn(state, mesh, coord)
    _check(seen_full, seen_half, coord, state.p_s.data, ncell, nlev)
