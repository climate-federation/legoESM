"""Virtual temperature in the MPAS hydrostatic dynamics, every vertical form
(``MPASPrimitiveEquationConfig.virtual_temperature`` /
``ExperimentConfig.mpas_virtual_temperature``).

With the switch on, the geopotential and the pressure-gradient correction use
``T_v = T (1 + (R_v/R_d - 1) q_v)``; the temperature equation keeps the dry T.
So the momentum tendency of (T, q_v) with the switch on must equal the momentum
tendency of T_v with the switch off, while the temperature and surface-pressure
tendencies stay those of the dry T.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
from legoesm.core.field import Field
from legoesm.grids.vertical import create_sigma_coordinate, make_cam6_l32_levels
from legoesm.thermo import virtual_temperature

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                                os.pardir, "scripts", "run"))
from run_amip import build_arg_parser, build_config_from_args  # noqa: E402

NLEV = 32


def mpas_hydrostatic_tendencies(state, mesh, crd, cfg, form="legacy"):
    return mpas_pe_tendencies(state, mesh, crd, cfg, vertical_form=form)


def _helpers():
    path = pathlib.Path(__file__).parents[3] / "unit" / "test_mpas_atmosphere.py"
    spec = importlib.util.spec_from_file_location("_mpas_atm_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def case():
    h = _helpers()
    mesh = h._make_mesh(level=2)
    state = h._add_perturbation_hydro(h._make_hydrostatic_state(mesh, NLEV), mesh, NLEV)
    lat = np.asarray(mesh.latCell)
    lon = np.asarray(mesh.lonCell)
    prof = np.linspace(210.0, 295.0, NLEV)                                  # top -> bottom
    T = prof[None, :] + 4.0 * np.cos(lat)[:, None] + 1.5 * np.sin(2 * lon)[:, None]
    # moist low levels, varying sideways (the buoyancy the switch adds)
    shape = np.exp(-np.arange(NLEV)[::-1] / 6.0)[None, :]
    q = 0.018 * (0.4 + 0.6 * np.cos(lat) ** 2)[:, None] * (1 + 0.3 * np.sin(3 * lon))[:, None] * shape
    ps = 1.0e5 + 800.0 * np.sin(lat)
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)),
                           p_s=state.p_s.replace(data=jnp.asarray(ps)))
    qf = Field(data=jnp.asarray(q), name="q_v", dims=("nCells", "nlev"), units="kg/kg")
    return mesh, state, state._replace(tracers={"q_v": qf}), jnp.asarray(q)


@pytest.mark.parametrize("coord,form", [("hybrid", "legacy"), ("hybrid", "as83"), ("hybrid", "as83_logmean"),
                                        ("hybrid", "sb81_theta"), ("sigma", "legacy"), ("sigma", "sb81_theta")])
def test_on_equals_dry_dynamics_driven_by_tv_and_keeps_dry_thermodynamics(case, coord, form):
    mesh, dry, moist, q = case
    crd = make_cam6_l32_levels() if coord == "hybrid" else create_sigma_coordinate(NLEV, dtype=jnp.float64)
    base = MPASPrimitiveEquationConfig(nu_del2=1.0e5)
    on = mpas_hydrostatic_tendencies(moist, mesh, crd, base._replace(virtual_temperature=True), form)
    off = mpas_hydrostatic_tendencies(moist, mesh, crd, base, form)
    tv_state = moist._replace(T=moist.T.replace(data=virtual_temperature(moist.T.data, q)))
    off_tv = mpas_hydrostatic_tendencies(tv_state, mesh, crd, base, form)
    du_on, du_off, du_tv = (np.asarray(t.du_dt.data) for t in (on, off, off_tv))
    np.testing.assert_allclose(du_on, du_tv, rtol=1e-10, atol=1e-14 * np.abs(du_tv).max())
    assert np.abs(du_on - du_off).max() > 1e-3 * np.abs(du_off).max()       # non-vacuous
    np.testing.assert_array_equal(np.asarray(on.dT_dt.data), np.asarray(off.dT_dt.data))
    np.testing.assert_array_equal(np.asarray(on.dp_s_dt.data), np.asarray(off.dp_s_dt.data))


@pytest.mark.parametrize("form", ["legacy", "as83"])
def test_default_off_and_dry_air_unchanged(case, form):
    mesh, dry, moist, q = case
    crd = make_cam6_l32_levels()
    base = MPASPrimitiveEquationConfig(nu_del2=1.0e5)
    assert MPASPrimitiveEquationConfig().virtual_temperature is False
    off_dry = mpas_hydrostatic_tendencies(dry, mesh, crd, base, form)
    off_moist = mpas_hydrostatic_tendencies(moist, mesh, crd, base, form)
    np.testing.assert_array_equal(np.asarray(off_moist.du_dt.data), np.asarray(off_dry.du_dt.data))
    zero_q = moist._replace(tracers={"q_v": moist.tracers["q_v"].replace(data=jnp.zeros_like(q))})
    on_zero = mpas_hydrostatic_tendencies(zero_q, mesh, crd, base._replace(virtual_temperature=True), form)
    np.testing.assert_allclose(np.asarray(on_zero.du_dt.data), np.asarray(off_dry.du_dt.data),
                               rtol=1e-12, atol=1e-14 * np.abs(np.asarray(off_dry.du_dt.data)).max())
    with pytest.raises(ValueError, match="q_v"):
        mpas_hydrostatic_tendencies(dry, mesh, crd, base._replace(virtual_temperature=True), form)


def test_coefficient_is_the_model_constants():
    T = jnp.asarray(300.0)
    q = jnp.asarray(0.018)
    np.testing.assert_allclose(float(virtual_temperature(T, q)),
                               300.0 * (1.0 + (constants.R_v / constants.R_d - 1.0) * 0.018), rtol=1e-14)


@pytest.mark.parametrize("form", ["legacy", "as83"])
def test_gradient_through_the_switch_is_finite_and_live(case, form):
    mesh, dry, moist, q = case
    crd = make_cam6_l32_levels()
    cfg = MPASPrimitiveEquationConfig(nu_del2=1.0e5, virtual_temperature=True)

    def f(qq):
        s = moist._replace(tracers={"q_v": moist.tracers["q_v"].replace(data=qq)})
        return jnp.sum(mpas_hydrostatic_tendencies(s, mesh, crd, cfg, form).du_dt.data ** 2)

    g = np.asarray(jax.grad(f)(q))
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0


def test_flag_reaches_the_dycore_and_is_refused_off_the_mpas_core():
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.run_config_yaml import load_yaml_config
    deck = pathlib.Path(__file__).parents[4] / "config" / "amip" / "amip_production.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(deck), parser))
    on = build_config_from_args(parser.parse_args(["--mpas-virtual-temperature"]))
    off = build_config_from_args(parser.parse_args([]))
    assert on.mpas_virtual_temperature is True and off.mpas_virtual_temperature is False
    mesh = _helpers()._make_mesh(level=2)
    assert create_atmosphere_dycore(on, mesh, make_cam6_l32_levels()).config.virtual_temperature is True
    assert create_atmosphere_dycore(off, mesh, make_cam6_l32_levels()).config.virtual_temperature is False
    assert ExperimentConfig().mpas_virtual_temperature is False
    bad = ExperimentConfig(mpas_virtual_temperature=True)
    with pytest.raises(ValueError, match="mpas_virtual_temperature"):
        bad.validate_strict()
