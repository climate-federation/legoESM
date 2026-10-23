"""No radiative thlp2 source reaches prognostic CLUBB on the MPAS lane.

CAM6 adds CLUBB's radiative thlp2 source only under
``clubb_cloudtop_cooling`` (``do_cldcool``), default ``.false.``
(namelist_defaults_cam.xml cesm2_1_rel :1200; clubb_intr.F90 :1935; user
decision 2026-10-19).  The MPAS turbulence physics_fn must therefore not hand
the cached ``PhysicsState.rad_heating`` to the kernel, even when it is set.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig, init_clubb_moments, pack_clubb_moments)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.integration import (  # noqa: E402
    _make_mpas_turbulence)


@pytest.fixture(scope="module")
def mpas_setup():
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas)
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    sigma = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    q_v = jnp.full((ncell, nlev), 0.008, dtype=state.T.data.dtype)
    return mesh, sigma, state._replace(tracers={"q_v": q_v})


def test_mpas_lane_does_not_route_rad_heating(mpas_setup, monkeypatch):
    mesh, sigma, state = mpas_setup
    ncell, nlev = state.T.data.shape
    cfg = CLUBBConfig(prognostic=True)
    seen = []
    real = clubb_mod.clubb_turbulence_prognostic

    def spy(*a, **k):
        seen.append(set(k))
        return real(*a, **k)

    monkeypatch.setattr(clubb_mod, "clubb_turbulence_prognostic", spy)
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme="clubb", clubb=cfg), 300.0)
    phys_state = types.SimpleNamespace(
        clubb_moments=pack_clubb_moments(init_clubb_moments(ncell, nlev, cfg)),
        rad_heating=jnp.full((ncell, nlev), 1.0e-4))
    fn(state, mesh, sigma, phys_state=phys_state,
       forcing={"T_sfc": jnp.full((ncell,), 295.0)})
    assert seen, "the prognostic CLUBB kernel was not called"
    assert all("rad_dT_dt" not in k for k in seen), seen
