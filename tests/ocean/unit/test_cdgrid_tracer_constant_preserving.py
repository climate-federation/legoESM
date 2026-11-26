"""A uniform tracer stays uniform on the cubed-sphere ``OceanModel``.

The horizontal tracer operator is the FCT flux form ``-div_h(u q)`` on the
C-grid velocity, while the vertical one is advective ``-w dq/dz``.  Without the
``+ q div_h(u)`` term a uniform salinity picks up ``-q div_h(u)`` wherever the
flow is divergent: 35 psu drifted ~3e-2 psu in 10 steps of 300 s on this setup.
``OceanConfig.tracer_advection_constant_preserving`` (default True) adds the
term; False keeps the old operator.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

S0 = 35.0
N_STEPS = 10
DT = 300.0


def _run(constant_preserving, staggering):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import OceanConfig
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.experiments.inertia_gravity_wave import (
        InertiaGravityWaveConfig, create_initial_conditions)

    grid = create_cubed_sphere(8)
    z = create_ocean_z_star(n_levels=2, H_max=1000.0)
    s = create_initial_conditions("cubed_sphere", grid, z, InertiaGravityWaveConfig())
    wet = s.land_mask.data[..., None]
    s = s._replace(S=s.S.replace(data=jnp.full_like(s.S.data, S0) * wet))
    cfg = OceanConfig(A_h=1e2, K_h=0.0, A_v=1e-4, K_v=1e-5,
                      n_barotropic_substeps=10, barotropic_staggering=staggering,
                      tracer_advection_constant_preserving=constant_preserving)
    m = OceanModel(grid, z, cfg)
    for _ in range(N_STEPS):
        s = m.step(s, DT)
    return float(jnp.max(jnp.abs(jnp.where(wet > 0.5, s.S.data - S0, 0.0))))


def test_default_on():
    from legoesm.ocean.state import OceanConfig
    assert OceanConfig().tracer_advection_constant_preserving is True


@pytest.mark.parametrize("staggering", ["a_grid", "c_grid"])
def test_uniform_salinity_stays_uniform(staggering):
    assert _run(True, staggering) < 1e-12


def test_old_operator_still_drifts():
    assert _run(False, "a_grid") > 1e-2
