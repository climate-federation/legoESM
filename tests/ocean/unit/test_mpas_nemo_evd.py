"""NEMO zdfevd on the MPAS lane (ocean_model_mpas -> apply_nemo_evd).

A warm layer planted under a cold one in half the cells must be mixed away by
rn_evd within one step, and a stable state must step bitwise as if EVD were
off (the trigger never fires there).  The legacy additive density-jump
convection on MPAS is untouched (not selected by this card).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig,
    OceanConvectionConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.vertical import create_ocean_z_star

EVD = EnhancedDiffusionConfig(
    K_conv=100.0, K_bg=0.0, nu_conv=0.0, nu_bg=0.0, n2_mode="nemo_bn2",
    n2_eos_form="teos10", smooth_transition=False, n2_threshold=-1e-12,
    evd_composition="nemo_replace")


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture()
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture()
def z_coord():
    return create_ocean_z_star(n_levels=8, H_max=4000.0)


def _state(mesh, z_coord, inversion: bool):
    state = rest_state_mpas_ocean(mesh, z_coord, H_max=4000.0)
    z = jnp.asarray(z_coord.z_full_ref)
    T = jnp.broadcast_to((20.0 + 15.0 * z / 4000.0)[None, :], state.T.data.shape)
    north = (jnp.asarray(mesh.latCell) > 0.0) & (state.land_mask.data > 0.5)
    if inversion:
        T = T.at[:, 2].set(jnp.where(north, T[:, 1] + 3.0, T[:, 2]))
    return state._replace(T=state.T.replace(data=T)), north


def _model(mesh, z_coord, evd: bool):
    conv = (OceanConvectionConfig(scheme="enhanced_diffusion",
                                  enhanced_diffusion=EVD)
            if evd else OceanConvectionConfig(scheme="none"))
    cfg = MPASOceanConfig(
        barotropic_solver="implicit_cn", implicit_vertical_mixing=True,
        A_h=1.0e3, A_v=1.4e-6, K_v=1.0e-10, bottom_drag_r=1.1e-3,
        barotropic_implicit_pcg_tol=1.0e-10,
        barotropic_implicit_pcg_maxiter=300,
        physics=OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="tke", tke=TKEConfig()),
            convection=conv))
    return MPASOceanModel(mesh, z_coord, cfg)


def _step(model, state):
    s = model.seed_tke(state) if hasattr(model, "seed_tke") else state
    return model.step(s, dt=300.0)


def test_evd_mixes_the_planted_inversion(mesh, z_coord):
    state, north = _state(mesh, z_coord, inversion=True)
    on = _step(_model(mesh, z_coord, True), state).T.data
    off = _step(_model(mesh, z_coord, False), state).T.data
    jump_on = on[:, 2] - on[:, 1]
    jump_off = off[:, 2] - off[:, 1]
    # rn_evd over one 300 s step: the inversion is mostly removed in the north
    assert float(jnp.max(jnp.where(north, jump_on, -jnp.inf))) < 0.5 * float(
        jnp.min(jnp.where(north, jump_off, jnp.inf)))
    assert bool(jnp.all(jnp.isfinite(on)))
    dry = state.land_mask.data <= 0.5
    assert float(jnp.max(jnp.where(dry[:, None], jnp.abs(on - off), 0.0))) == 0.0


def test_stable_state_steps_as_if_evd_were_off(mesh, z_coord):
    state, _ = _state(mesh, z_coord, inversion=False)
    on = _step(_model(mesh, z_coord, True), state).T.data
    off = _step(_model(mesh, z_coord, False), state).T.data
    assert float(jnp.max(jnp.abs(on - off))) == 0.0


def test_the_step_routes_through_the_shared_nemo_helper(mesh, z_coord, monkeypatch):
    """Both MPAS convection branches would mix the planted inversion, so the
    step test alone cannot tell NEMO's replace path from the legacy additive
    one.  Pin that the nemo_replace card calls apply_nemo_evd."""
    import legoesm.ocean.dynamics.ocean_model_mpas as om
    calls = []
    real = om.apply_nemo_evd

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(om, "apply_nemo_evd", spy)
    state, _ = _state(mesh, z_coord, inversion=True)
    _step(_model(mesh, z_coord, True), state)
    assert calls, "nemo_replace card did not reach apply_nemo_evd"
