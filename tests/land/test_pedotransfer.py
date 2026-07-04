"""Unit tests for the Cosby (1984) texture -> Clapp-Hornberger pedotransfer."""

import jax.numpy as jnp
import numpy as np

from legoesm.land.pedotransfer import (
    cosby_hydraulic_params,
    soil_hydraulics_config_from_texture,
)


def test_loam_values_physical():
    # loam ~ 40% sand, 20% clay
    p = cosby_hydraulic_params(jnp.array(40.0), jnp.array(20.0))
    assert np.isclose(float(p.theta_sat), 0.489 - 0.00126 * 40.0)     # ~0.439
    assert np.isclose(float(p.b_ch), 2.91 + 0.159 * 20.0)            # ~6.09
    assert float(p.psi_sat) < 0.0                                     # matric potential negative
    assert 1e-7 < float(p.K_sat) < 1e-4                               # plausible loam K_sat [m/s]


def test_texture_monotonicity():
    sandy = cosby_hydraulic_params(jnp.array(80.0), jnp.array(5.0))
    clayey = cosby_hydraulic_params(jnp.array(20.0), jnp.array(60.0))
    # more sand -> lower porosity, higher K_sat; more clay -> larger b
    assert float(sandy.theta_sat) < float(clayey.theta_sat)
    assert float(sandy.K_sat) > float(clayey.K_sat)
    assert float(clayey.b_ch) > float(sandy.b_ch)


def test_config_builder_switches_curve():
    cfg = soil_hydraulics_config_from_texture(40.0, 20.0)
    assert cfg.retention_curve == "clapp_hornberger"
    assert np.isclose(cfg.theta_sat, 0.489 - 0.00126 * 40.0)
    assert cfg.b_ch > 0 and cfg.psi_sat < 0 and cfg.K_sat > 0


def test_differentiable_in_texture():
    import jax
    b_of_clay = lambda clay: cosby_hydraulic_params(jnp.array(30.0), clay).b_ch
    g = jax.grad(b_of_clay)(jnp.array(25.0))
    assert np.isclose(float(g), 0.159)        # d b / d clay = 0.159
