"""CAM-style top diffusion sponge for the MPAS dycore: a per-level multiplier
on the del2 viscosity, geometric from `factor` at the model top to 1 below the
top n layers; off state (0 / 1.0) is exactly ones so production is byte-identical."""
from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig, sponge_del2_profile)
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "scripts", "run"))
from run_amip import build_arg_parser, build_config_from_args  # noqa: E402


def test_profile_off_state_is_exact_ones_and_ramp_is_geometric():
    assert np.array_equal(np.asarray(sponge_del2_profile(30, 0, 1.0)), np.ones(30))
    assert np.array_equal(np.asarray(sponge_del2_profile(30, 3, 1.0)), np.ones(30))
    prof = np.asarray(sponge_del2_profile(30, 3, 8.0))
    np.testing.assert_allclose(prof[:4], [8.0, 4.0, 2.0, 1.0], rtol=1e-12)
    assert np.array_equal(prof[3:], np.ones(27))
    assert MPASPrimitiveEquationConfig().sponge_del2_top_layers == 0
    assert MPASPrimitiveEquationConfig().sponge_del2_top_factor == 1.0
    jit_prof = jax.jit(lambda f: sponge_del2_profile(30, 3, f))(8.0)
    np.testing.assert_allclose(np.asarray(jit_prof), prof, rtol=1e-14)
    g = jax.grad(lambda f: sponge_del2_profile(30, 3, f).sum())(8.0)
    assert np.isfinite(g) and g != 0.0


def test_config_defaults_gates_and_wiring():
    assert DycoreConfig().mpas_sponge_del2_top_layers == 0
    assert DycoreConfig().mpas_sponge_del2_top_factor == 1.0
    mpas = dict(discretization="mpas")
    grid = GridConfig(grid_type="mpas", vertical_coord="sigma", nlev=36, resolution=6)
    ExperimentConfig(dycore=DycoreConfig(mpas_sponge_del2_top_layers=3,
                                         mpas_sponge_del2_top_factor=8.0, **mpas),
                     grid=grid).validate_strict()
    for bad in (dict(mpas_sponge_del2_top_layers=-1), dict(mpas_sponge_del2_top_layers=36),
                dict(mpas_sponge_del2_top_factor=0.5),
                dict(mpas_sponge_del2_top_factor=float("inf")),
                dict(mpas_sponge_del2_top_layers=2, model_type="nonhydrostatic"),
                dict(mpas_sponge_del2_top_layers=2, a_h_scale=0.0)):
        with pytest.raises(ValueError, match="mpas_sponge_del2"):
            ExperimentConfig(dycore=DycoreConfig(**mpas, **bad), grid=grid).validate_strict()
    ExperimentConfig(dycore=DycoreConfig(mpas_sponge_del2_top_layers=0,
                                         a_h_scale=0.0, **mpas),
                     grid=grid).validate_strict()
    with pytest.raises(ValueError, match="mpas_sponge_del2"):
        ExperimentConfig(dycore=DycoreConfig(discretization="cdgrid",
                                             mpas_sponge_del2_top_layers=2)).validate_strict()
    args = build_arg_parser().parse_args(["--mpas-sponge-del2-top-layers", "3",
                                          "--mpas-sponge-del2-top-factor", "8"])
    cfg = build_config_from_args(args)
    assert cfg.dycore.mpas_sponge_del2_top_layers == 3
    assert cfg.dycore.mpas_sponge_del2_top_factor == 8.0


@pytest.mark.parametrize("nu_del4", [0.0, 1.0e17])
def test_tendency_applies_the_profile_only_to_the_top_layers(nu_del4):
    """The enabled sponge changes the del2 momentum tendency by exactly the
    per-level factor in the top layers and nowhere else, independently of the
    del4 term."""
    import importlib.util
    import pathlib
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import mpas_hydrostatic_tendencies
    from legoesm.grids.vertical import create_sigma_coordinate
    helpers_path = pathlib.Path(__file__).with_name("test_mpas_atmosphere.py")
    spec = importlib.util.spec_from_file_location("_mpas_atm_helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    nlev = 8
    mesh = helpers._make_mesh(level=2)
    sig = create_sigma_coordinate(nlev, dtype=jnp.float64)
    state = helpers._add_perturbation_hydro(helpers._make_hydrostatic_state(mesh, nlev), mesh, nlev)
    on = MPASPrimitiveEquationConfig(nu_del2=1.0e5, nu_del4=nu_del4,
                                     sponge_del2_top_layers=2,
                                     sponge_del2_top_factor=4.0)
    off = MPASPrimitiveEquationConfig(nu_del2=1.0e5, nu_del4=nu_del4)

    def du(cfg):
        return np.asarray(mpas_hydrostatic_tendencies(state, mesh, sig, cfg).du_dt.data)

    du_on, du_off = du(on), du(off)
    increment = du_on - du_off
    if nu_del4 == 0.0:
        zero = du(off._replace(nu_del2=0.0))
        visc_on, visc_off = du_on - zero, du_off - zero        # isolate the del2 term
        assert np.abs(visc_off).max() > 0.0                      # non-vacuous
        np.testing.assert_allclose(visc_on[:, 0], 4.0 * visc_off[:, 0], rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(visc_on[:, 1], 2.0 * visc_off[:, 1], rtol=1e-9, atol=1e-14)
        assert np.array_equal(visc_on[:, 2:], visc_off[:, 2:])
    else:
        du_off_del2_only = du(MPASPrimitiveEquationConfig(nu_del2=1.0e5, nu_del4=0.0))
        du_on_del2_only = du(MPASPrimitiveEquationConfig(nu_del2=1.0e5, nu_del4=0.0,
                                                         sponge_del2_top_layers=2,
                                                         sponge_del2_top_factor=4.0))
        np.testing.assert_allclose(increment, du_on_del2_only - du_off_del2_only,
                                   rtol=1e-9, atol=1e-14)
        assert not np.array_equal(du_off, du_off_del2_only)      # del4 term is active
    assert np.abs(increment).max() > 0.0
