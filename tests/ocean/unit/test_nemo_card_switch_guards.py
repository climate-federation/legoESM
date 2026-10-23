"""Card-selected NEMO switches must fail loudly instead of reading as off."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
    surface_stress_faces,
)


def _geographic():
    return SimpleNamespace(tau_x=jnp.zeros((2, 2)), tau_y=jnp.zeros((2, 2)))


def test_fold_vector_sign_refuses_geographic_only_stress():
    with pytest.raises(ValueError, match="tau_i_native"):
        surface_stress_faces(_geographic(), jnp.float64, None, None, None,
                             nemo_fold_vector_sign=True)


def test_fold_vector_sign_refuses_bypassed_native_stress():
    forcing = _geographic()
    forcing.tau_i_native = jnp.zeros((2, 2))
    forcing.tau_j_native = jnp.zeros((2, 2))
    with pytest.raises(ValueError, match="tau_i_native"):
        surface_stress_faces(forcing, jnp.float64, None, None, None,
                             nemo_fold_vector_sign=True, use_native=False)


def test_fold_vector_sign_refuses_half_geographic_stress():
    with pytest.raises(ValueError, match="tau_i_native"):
        surface_stress_faces(SimpleNamespace(tau_y=jnp.zeros((2, 2))),
                             jnp.float64, None, None, None,
                             nemo_fold_vector_sign=True)


def test_fold_vector_sign_refuses_bypassed_native_only_stress():
    forcing = SimpleNamespace(tau_i_native=jnp.zeros((2, 2)),
                              tau_j_native=jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="tau_i_native"):
        surface_stress_faces(forcing, jnp.float64, None, None, None,
                             nemo_fold_vector_sign=True, use_native=False)


def test_half_native_stress_is_refused():
    forcing = SimpleNamespace(tau_i_native=jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="supplied together"):
        surface_stress_faces(forcing, jnp.float64, None, None, None,
                             nemo_fold_vector_sign=True)


def test_fold_vector_sign_without_any_stress_is_still_none():
    assert surface_stress_faces(SimpleNamespace(), jnp.float64, None, None,
                                None, nemo_fold_vector_sign=True) is None


@pytest.mark.parametrize("over", [
    {},                                        # default al81 vector form
    {"vorticity_scheme": "een_planetary"},
    {"vorticity_scheme": "een_total", "momentum_advection": "flux_form"},
    {"vorticity_scheme": "een_total", "momentum_advection": "weno5"},
])
def test_een_relative_vorticity_refused_where_it_would_be_ignored(over):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0)
    config = LatLonCGridOceanConfig.from_flat(
        A_h=0.0, A_v=0.0, K_h=0.0, K_v=0.0)._replace(
            een_nemo_relative_vorticity=True, een_metric_weighting="nemo",
            **over)
    with pytest.raises(ValueError, match="requires vector-form"):
        latlon_cgrid_ocean_baroclinic_tendencies(state, grid, z, config)


def _small_case():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0)
    config = LatLonCGridOceanConfig.from_flat(A_h=0.0, A_v=0.0, K_h=0.0, K_v=0.0)
    return state, grid, z, config


def test_fold_vector_sign_refused_on_the_explicit_stress_path():
    state, grid, z, config = _small_case()
    config = config._replace(surface_stress_nemo_fold_vector_sign=True,
                             surface_stress_implicit=False)
    with pytest.raises(ValueError, match="requires surface_stress_implicit"):
        latlon_cgrid_ocean_baroclinic_tendencies(state, grid, z, config)
