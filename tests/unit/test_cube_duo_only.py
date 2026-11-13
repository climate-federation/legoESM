"""The retired atmosphere cube selectors cannot reach shared ocean C/D cores."""
from pathlib import Path

import pytest
import yaml

from legoesm.atmosphere.dynamics import (
    create_model, get_solver_class, resolve_solver_name,
)
from legoesm.driver.component_factory import create_atmosphere_dycore, driver_solver_name
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig


@pytest.mark.parametrize('dynamics', ['shallow_water', 'hydrostatic', 'nonhydrostatic'])
@pytest.mark.parametrize('discretization', ['cdgrid', 'centered', 'finite_volume', 'cgrid', 'fv'])
def test_every_legacy_cube_atmosphere_selection_raises(dynamics, discretization):
    if dynamics == 'hydrostatic' and discretization == 'cdgrid':
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        grid = create_cubed_sphere(2)
        for name in ('spectral_primitive_equations', 'sfno_primitive_equations',
                     'ucast_primitive_equations'):
            with pytest.raises(ValueError, match='fv3_duo'):
                create_model(name, grid=grid)
        for column_lane in (False, True):
            for integrator in ('leapfrog_si', 'semi_implicit'):
                unsupported = ExperimentConfig(
                    grid=GridConfig(resolution=2, nlev=5),
                    dycore=DycoreConfig(column_lane=column_lane,
                                        energy_fixer=False,
                                        time_integrator=integrator))
                with pytest.raises(ValueError, match='fv3_duo.*semi-implicit'):
                    create_atmosphere_dycore(unsupported, grid, None)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type='cubed_sphere'),
        dycore=DycoreConfig(model_type=dynamics, discretization=discretization))
    for call in (lambda: driver_solver_name(cfg),
                 lambda: create_atmosphere_dycore(cfg, None, None),
                 cfg.validate_strict,
                 lambda: resolve_solver_name(dynamics=dynamics, discretization=discretization)):
        with pytest.raises(ValueError, match='fv3_duo'):
            call()
    suffix = {'shallow_water': 'shallow_water', 'hydrostatic': 'primitive_equations',
              'nonhydrostatic': 'compressible_euler'}[dynamics]
    for prefix in ('cdgrid_', 'cgrid_', 'fv_', ''):
        name = prefix + suffix
        for call in (lambda: create_model(name), lambda: get_solver_class(name),
                     lambda: resolve_solver_name(equations=name)):
            with pytest.raises(ValueError, match='fv3_duo'):
                call()


def test_cube_defaults_resolve_to_fv3_duo():
    from legoesm.config import DEFAULT_CONFIG
    from scripts.run.run_amip import _default_discretization_for_grid
    from scripts.run.run_coupled import _COUPLED_DISCRETIZATION

    assert DycoreConfig().discretization == 'fv3_duo'
    assert DEFAULT_CONFIG['atmosphere']['discretization'] == 'fv3_duo'
    assert _default_discretization_for_grid('cubed_sphere') == 'fv3_duo'
    assert _COUPLED_DISCRETIZATION['cubed_sphere'] == 'fv3_duo'
    cfg = ExperimentConfig()
    assert driver_solver_name(cfg) == 'fv3_duo_primitive_equations'
    assert resolve_solver_name(dynamics='hydrostatic', discretization='fv3_duo') == 'fv3_duo_primitive_equations'
    root = Path(__file__).resolve().parents[2]
    for deck in ('amip_sota.yaml', 'amip_realism_c12.yaml'):
        data = yaml.safe_load((root / 'config' / 'amip' / deck).read_text())
        assert data['discretization'] == 'fv3_duo'
    # Non-cube driver defaults and ambiguous lat-lon selections stay intact.
    assert _default_discretization_for_grid('latlon') == 'centered'
    assert _default_discretization_for_grid('gaussian') == 'centered'
    assert _default_discretization_for_grid('mpas') == 'mpas'
    assert _COUPLED_DISCRETIZATION['latlon'] == 'latlon_cgrid'
    for disc in ('centered', 'finite_volume'):
        assert driver_solver_name(cfg._replace(grid=GridConfig(grid_type='latlon'),
                                          dycore=DycoreConfig(discretization=disc))) == 'latlon_cgrid_primitive_equations'
