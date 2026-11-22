"""Separate rate for the MPAS vertical del4 filter on tracers.

``nu_vert4_tracer`` sets the filter rate for every tracer the dycore filters;
the temperature filter keeps ``nu_vert4_T``.  None (the default) follows
``nu_vert4_T``, so existing runs are bit-identical; 0.0 leaves the tracers
unfiltered while T keeps its filter.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel,
    mpas_legacy_pe_tendencies)
from legoesm.core.field import Field
from tests.atmosphere.hydrostatic.unit.test_mpas_vert4_flux_form import NU, _tracer_state

# The three per-tracer filter forms the dycore can select for q_v.
FORMS = {
    "projected": {},
    "log": {"vert4_log_form_tracers": ("q_v",)},
    "flux": {"vert4_flux_form_tracers": ("q_v",)},
}


def _tend(state, mesh, coord, cfg):
    td = mpas_legacy_pe_tendencies(state, mesh, coord, cfg, None, 0.0)
    return (np.asarray(td.dT_dt.data),
            {k: np.asarray(v.data) for k, v in td.tracer_tendencies.items()})


CASES = [(f, False) for f in FORMS]


def _setup(form, pas=False):
    mesh, coord, cfg, state = _tracer_state()
    cfg = cfg._replace(**FORMS[form])
    return mesh, coord, cfg, state


def test_default_is_none():
    assert MPASPrimitiveEquationConfig().nu_vert4_tracer is None


@pytest.mark.parametrize("form,pas", CASES)
def test_none_follows_the_temperature_rate_bitwise(form, pas):
    """None gives the same tendencies and the same one-step state as an
    explicit tracer rate equal to nu_vert4_T."""
    mesh, coord, cfg, state = _setup(form, pas)
    assert cfg.nu_vert4_tracer is None
    same = cfg._replace(nu_vert4_tracer=cfg.nu_vert4_T)
    dT0, dq0 = _tend(state, mesh, coord, cfg)
    dT1, dq1 = _tend(state, mesh, coord, same)
    np.testing.assert_array_equal(dT0, dT1)
    assert set(dq0) == {"q_v", "q_c"}
    for k in dq0:
        np.testing.assert_array_equal(dq0[k], dq1[k])
    s0 = MPASPrimitiveEquationModel(mesh, coord, cfg).step(state, 300.0)
    s1 = MPASPrimitiveEquationModel(mesh, coord, same).step(state, 300.0)
    np.testing.assert_array_equal(np.asarray(s0.T.data), np.asarray(s1.T.data))
    for k in s0.tracers:
        np.testing.assert_array_equal(np.asarray(s0.tracers[k].data),
                                      np.asarray(s1.tracers[k].data))


@pytest.mark.parametrize("form,pas", CASES)
def test_zero_tracer_rate_filters_T_only(form, pas):
    """nu_vert4_T = 2e-6, nu_vert4_tracer = 0: T keeps its filter tendency
    bitwise, every tracer gets none, and the filter
    leaves column tracer mass unchanged."""
    mesh, coord, cfg, state = _setup(form, pas)
    assert cfg.nu_vert4_T == NU == 2.0e-6
    on_T, on_q = _tend(state, mesh, coord, cfg)                       # both filtered
    sp_T, sp_q = _tend(state, mesh, coord, cfg._replace(nu_vert4_tracer=0.0))
    off_T, off_q = _tend(state, mesh, coord, cfg._replace(nu_vert4_T=0.0))
    np.testing.assert_array_equal(sp_T, on_T)
    assert np.max(np.abs(on_T - off_T)) > 0.0                         # control: T filter acts
    dp = np.asarray(state.p_s.data)[:, None] * np.asarray(coord.dsigma)[None, :]
    for k in on_q:
        assert np.max(np.abs(on_q[k] - off_q[k])) > 0.0               # control: filter acts on k
        filt = sp_q[k] - off_q[k]
        np.testing.assert_array_equal(filt, np.zeros_like(filt))
        np.testing.assert_array_equal((filt * dp).sum(-1), np.zeros(dp.shape[0]))


@pytest.mark.parametrize("form,pas", CASES)
def test_tracer_rate_gates_on_its_own(form, pas):
    """nu_vert4_T = 0 with nu_vert4_tracer = 2e-6: T unfiltered, tracers get
    exactly the filter they get today at nu_vert4_T = 2e-6."""
    mesh, coord, cfg, state = _setup(form, pas)
    on_T, on_q = _tend(state, mesh, coord, cfg)
    off_T, _ = _tend(state, mesh, coord, cfg._replace(nu_vert4_T=0.0))
    tr_T, tr_q = _tend(state, mesh, coord,
                       cfg._replace(nu_vert4_T=0.0, nu_vert4_tracer=NU))
    np.testing.assert_array_equal(tr_T, off_T)
    for k in on_q:
        np.testing.assert_array_equal(tr_q[k], on_q[k])


@pytest.mark.parametrize("flag,expect", [((), None), (("--mpas-nu-vert4-tracer", "0"), 0.0)])
def test_factory_threads_the_tracer_rate(flag, expect):
    """Production deck plus CLI flag -> config builder -> the dycore object the
    run steps: the tracer rate arrives, and the T rate is not touched."""
    import pathlib
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.driver.run_config_yaml import load_yaml_config
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.grids.voronoi import create_voronoi_mesh
    from scripts.run.run_amip import build_arg_parser, build_config_from_args
    deck = pathlib.Path(__file__).parents[4] / "config" / "amip" / "amip_production.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(deck), parser))
    cfg = build_config_from_args(parser.parse_args(list(flag)))
    model = create_atmosphere_dycore(cfg, create_voronoi_mesh(2), make_cam6_l32_levels())
    assert model.config.nu_vert4_tracer == expect
    assert model.config.nu_vert4_T == cfg.dycore.mpas_nu_vert4_T > 0.0
