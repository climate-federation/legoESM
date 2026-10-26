"""zdfiwm on the MPAS Voronoi lane (2026-09-01 three-grid unification A2).

Covers the three pieces that wired ``--iwm`` onto MPAS:
  * the model-level ADDITIVE splice (NEMO zdfphy order: closure first,
    zdf_iwm adds onto avt/avm) actually changes a stepped state;
  * the constructor both-or-neither / implicit-only guards (mirrors the
    lat-lon C-grid model);
  * the forcing maps SURVIVE the config-replace model rebuilds the OMIP
    driver performs (codex 2026-09-01: a rebuild that drops them silently
    reverts --iwm).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
    IWMConfig,
    uniform_iwm_forcing,
)
from legoesm.ocean.vertical import create_ocean_z_star


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


def _stratified_state(mesh, z_coord):
    state = rest_state_mpas_ocean(mesh, z_coord, H_max=4000.0)
    # Stable stratification so N^2 > 0 and the wave-driven K is nonzero.
    z = jnp.asarray(z_coord.z_full_ref)  # negative downward
    T_prof = 20.0 + 15.0 * z / 4000.0    # warm surface, cold bottom
    T = jnp.broadcast_to(T_prof[None, :], state.T.data.shape)
    return state._replace(T=state.T.replace(data=T))


def _cfg(iwm_enabled: bool) -> MPASOceanConfig:
    # Uniform-fallback powers default to ~1e-10 W/m^2 (near-zero placeholder);
    # use a production-scale power so the additive K is measurable in one step.
    vm = VerticalMixingConfig(
        scheme="tke", tke=TKEConfig(),
        iwm=IWMConfig(enabled=iwm_enabled, power_nsq_wm2=5.0e-3,
                      power_bot_wm2=5.0e-3))
    return MPASOceanConfig(
        barotropic_solver="implicit_cn",
        implicit_vertical_mixing=True,
        A_h=1.0e3, A_v=1.0e-3, K_v=1.0e-4,
        bottom_drag_r=1.1e-3,
        barotropic_implicit_pcg_tol=1.0e-10,
        barotropic_implicit_pcg_maxiter=300,
        physics=OceanPhysicsConfig(vertical_mixing=vm),
    )


def test_iwm_splice_changes_step(mesh, z_coord):
    """One step with iwm.enabled (uniform-power fallback maps) must differ
    from the identical step without it — the additive K reaches the tracer
    solve.  Non-vacuity: the disabled arm equals a second disabled arm."""
    state = _stratified_state(mesh, z_coord)

    m_on = MPASOceanModel(mesh, z_coord, _cfg(True))
    m_off = MPASOceanModel(mesh, z_coord, _cfg(False))
    s_on = m_on.seed_tke(state) if hasattr(m_on, "seed_tke") else state
    s_off = m_off.seed_tke(state) if hasattr(m_off, "seed_tke") else state

    out_on = m_on.step(s_on, dt=300.0)
    out_off = m_off.step(s_off, dt=300.0)
    out_off2 = MPASOceanModel(mesh, z_coord, _cfg(False)).step(s_off, dt=300.0)

    assert bool(jnp.all(jnp.isfinite(out_on.T.data)))
    d_on = float(jnp.max(jnp.abs(out_on.T.data - out_off.T.data)))
    d_ctrl = float(jnp.max(jnp.abs(out_off2.T.data - out_off.T.data)))
    assert d_ctrl == 0.0
    assert d_on > 0.0, "iwm splice had NO effect on the stepped state"


def test_ctor_guards(mesh, z_coord):
    maps = uniform_iwm_forcing(IWMConfig(enabled=True), (mesh.nCells,))
    # maps without the config switch: loud reject (silent-ignore forbidden)
    with pytest.raises(ValueError, match="silently ignored"):
        MPASOceanModel(mesh, z_coord, _cfg(False), iwm_forcing=maps)
    # iwm.enabled without implicit vertical mixing: loud reject
    bad = _cfg(True)._replace(implicit_vertical_mixing=False)
    with pytest.raises(ValueError, match="implicit"):
        MPASOceanModel(mesh, z_coord, bad)


def test_maps_survive_config_replace_rebuild(mesh, z_coord):
    maps = uniform_iwm_forcing(IWMConfig(enabled=True), (mesh.nCells,))
    model = MPASOceanModel(mesh, z_coord, _cfg(True), iwm_forcing=maps)
    # The driver's rebuild idiom: config._replace + explicit carry-over.
    rebuilt = MPASOceanModel(
        mesh, z_coord, model.config._replace(A_h=2.0e3),
        iwm_forcing=getattr(model, "_iwm_forcing", None))
    assert rebuilt._iwm_forcing is maps
