"""Tests for FC-Gram ocean PE models."""

import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.eos import wright_eos
from legoesm.ocean.dynamics.ocean_pe_fc import ocean_baroclinic_tendencies_fc
from legoesm.core.operators_fc import build_fc_config


@pytest.fixture
def ocean_grid():
    return create_cubed_sphere(8)


@pytest.fixture
def ocean_z_coord():
    return create_ocean_z_star(n_levels=5, H_max=4000.0)


@pytest.fixture
def ocean_state(ocean_grid, ocean_z_coord):
    return rest_state_ocean(
        ocean_grid, ocean_z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )


def test_ocean_fc_tracer_advection(ocean_grid, ocean_z_coord, ocean_state):
    """FC ocean tracer tendencies have correct shapes and are finite."""
    fc_config = build_fc_config(d=2, C=4, degree=5)
    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
    tend = ocean_baroclinic_tendencies_fc(
        ocean_state, ocean_grid, ocean_z_coord, fc_config, config,
    )
    assert tend.dT_dt.data.shape == (6, 8, 8, 5)
    assert tend.dS_dt.data.shape == (6, 8, 8, 5)
    assert jnp.all(jnp.isfinite(tend.dT_dt.data))
    assert jnp.all(jnp.isfinite(tend.dS_dt.data))


def test_ocean_fc_pressure_gradient(ocean_grid, ocean_z_coord, ocean_state):
    """FC ocean momentum tendencies have correct shapes and are finite."""
    fc_config = build_fc_config(d=2, C=4, degree=5)
    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
    tend = ocean_baroclinic_tendencies_fc(
        ocean_state, ocean_grid, ocean_z_coord, fc_config, config,
    )
    assert tend.du_dt.data.shape == (6, 8, 8, 5)
    assert tend.dv_dt.data.shape == (6, 8, 8, 5)
    assert jnp.all(jnp.isfinite(tend.du_dt.data))
    assert jnp.all(jnp.isfinite(tend.dv_dt.data))


def test_ocean_fc_land_masking(ocean_grid, ocean_z_coord, ocean_state):
    """FC ocean tendencies are zero where land_mask is zero."""
    fc_config = build_fc_config(d=2, C=4, degree=5)
    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
    tend = ocean_baroclinic_tendencies_fc(
        ocean_state, ocean_grid, ocean_z_coord, fc_config, config,
    )
    mask = ocean_state.land_mask.data
    mask_3d = mask[..., jnp.newaxis]
    land = (mask_3d == 0.0)
    assert float(jnp.max(jnp.abs(tend.du_dt.data * (1 - mask_3d)))) == 0.0
    assert float(jnp.max(jnp.abs(tend.dT_dt.data * (1 - mask_3d)))) == 0.0


def test_ocean_fc_cgrid_div_damping(ocean_grid, ocean_z_coord, ocean_state):
    """FC + div-damping ocean tendencies have correct shapes."""
    fc_config = build_fc_config(d=2, C=4, degree=5, div_damp_2=1e5)
    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
    tend = ocean_baroclinic_tendencies_fc(
        ocean_state, ocean_grid, ocean_z_coord, fc_config, config,
    )
    assert tend.du_dt.data.shape == (6, 8, 8, 5)
    assert jnp.all(jnp.isfinite(tend.du_dt.data))
    assert tend.deta_dt.data.shape == (6, 8, 8)
    assert jnp.all(jnp.isfinite(tend.deta_dt.data))


def test_ocean_fc_barotropic_unchanged():
    """Barotropic substeps are reused from ocean_pe (not modified)."""
    from legoesm.ocean.dynamics.barotropic import barotropic_substeps
    assert callable(barotropic_substeps)


def test_ocean_fc_eos_unchanged():
    """EOS is reused from ocean_pe (not modified by FC)."""
    T = jnp.array([10.0])
    S = jnp.array([35.0])
    p = jnp.array([1e7])
    rho = wright_eos(T, S, p)
    assert jnp.all(jnp.isfinite(rho))
    assert float(rho[0]) > 1000.0


# ==============================================================================
# OceanModel discretization integration tests
# ==============================================================================

def test_ocean_model_fc_gram_step(ocean_grid, ocean_z_coord, ocean_state):
    """OceanModel with fc_gram discretization runs a step."""
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    config = OceanConfig(
        use_conservation_fixer=False,
        A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
    )
    model = OceanModel(
        ocean_grid, ocean_z_coord, config=config,
        discretization="fc_gram",
    )
    state_new = model.step(ocean_state, 60.0)
    assert jnp.all(jnp.isfinite(state_new.eta.data))
    assert jnp.all(jnp.isfinite(state_new.T.data))


def test_ocean_model_fc_gram_cgrid_step(ocean_grid, ocean_z_coord, ocean_state):
    """OceanModel with fc_gram_cgrid discretization runs a step."""
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.core.operators_fc import build_fc_config
    fc_config = build_fc_config(d=2, C=4, degree=5, div_damp_2=1e5)
    config = OceanConfig(
        use_conservation_fixer=False,
        A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
    )
    model = OceanModel(
        ocean_grid, ocean_z_coord, config=config,
        discretization="fc_gram_cgrid", fc_config=fc_config,
    )
    state_new = model.step(ocean_state, 60.0)
    assert jnp.all(jnp.isfinite(state_new.eta.data))
    assert jnp.all(jnp.isfinite(state_new.T.data))


def test_ocean_model_centered_step(ocean_grid, ocean_z_coord, ocean_state):
    """OceanModel with default centered discretization runs a step."""
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    config = OceanConfig(use_conservation_fixer=False)
    model = OceanModel(ocean_grid, ocean_z_coord, config=config)
    state_new = model.step(ocean_state, 60.0)
    assert jnp.all(jnp.isfinite(state_new.eta.data))
    assert jnp.all(jnp.isfinite(state_new.T.data))


def test_ocean_model_invalid_discretization(ocean_grid, ocean_z_coord):
    """OceanModel rejects invalid discretization."""
    import pytest as _pytest
    with _pytest.raises(ValueError, match="Unknown ocean discretization"):
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        OceanModel(ocean_grid, ocean_z_coord, discretization="invalid")
