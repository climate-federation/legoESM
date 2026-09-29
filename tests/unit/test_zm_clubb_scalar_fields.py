"""ZM / CLUBB tunables as flat ExperimentConfig scalars reach the scheme configs
the MPAS step is built from (convection_config_for / turbulence_config_for),
through the CLI, the production deck and --params; unset keeps the defaults."""
from __future__ import annotations

import pathlib

import pytest

from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBParams
from legoesm.atmosphere.physics.turbulence.integration import (
    materialize_sub_config,
)
from legoesm.driver.config import (
    CLUBB_PROGNOSTIC_ONLY,
    CLUBB_SCALAR_FIELDS,
    ZM_SCALAR_FIELDS,
    ExperimentConfig,
)
from legoesm.driver.physics_pipeline import (
    convection_config_for,
    turbulence_config_for,
)

_DECK = (pathlib.Path(__file__).resolve().parents[2]
         / "config" / "amip" / "amip_production.yaml")
# Inside each parameter's __param_spec__ bounds, and != its default.
_ZM = {"zm_c0_lnd": 0.015, "zm_c0_ocn": 0.045, "zm_ke": 1.0e-5,
       "zm_dmpdz": -2.0e-3, "zm_tau": 5400.0, "zm_capelmt": 100.0}
_CLUBB = {"clubb_c14": 3.0, "clubb_c8": 5.0, "clubb_c11": 0.5,
          "clubb_c11b": 0.5, "clubb_gamma_coef": 0.25,
          "clubb_gamma_coefb": 0.25, "clubb_beta": 2.0, "clubb_c_k10": 1.0}


def _parse(argv):
    from scripts.run import run_amip
    return run_amip.build_config_from_args(
        run_amip.build_arg_parser().parse_args(argv))


def _leaves(cfg):
    zm = convection_config_for(cfg).zhang_mcfarlane
    cp = materialize_sub_config(turbulence_config_for(cfg)).clubb.params
    return ({f: getattr(zm, leaf) for f, leaf in ZM_SCALAR_FIELDS.items()},
            {f: getattr(cp, leaf) for f, leaf in CLUBB_SCALAR_FIELDS.items()})


def test_tables_are_real_fields_appended_at_the_tuple_end():
    assert set(_ZM) == set(ZM_SCALAR_FIELDS)
    assert set(_CLUBB) == set(CLUBB_SCALAR_FIELDS)
    assert CLUBB_PROGNOSTIC_ONLY <= set(CLUBB_SCALAR_FIELDS)
    tail = list(ZM_SCALAR_FIELDS) + list(CLUBB_SCALAR_FIELDS)
    assert list(ExperimentConfig._fields[-len(tail):]) == tail


def test_cli_values_reach_the_mpas_scheme_configs():
    argv = ["--convection", "zhang_mcfarlane", "--turbulence", "clubb",
            "--clubb-prognostic"]
    for f, v in {**_ZM, **_CLUBB}.items():
        argv += ["--" + f.replace("_", "-"), str(v)]
    cfg = _parse(argv)
    cfg.validate_strict()
    zm, cp = _leaves(cfg)
    assert zm == _ZM
    assert cp == _CLUBB


def test_unset_keeps_scheme_defaults():
    cfg = ExperimentConfig(convection="zhang_mcfarlane", turbulence="clubb",
                           clubb_prognostic=True)
    zm, cp = _leaves(cfg)
    assert zm == {f: getattr(ZhangMcFarlaneConfig(), leaf)
                  for f, leaf in ZM_SCALAR_FIELDS.items()}
    assert cp == {f: getattr(CLUBBParams(), leaf)
                  for f, leaf in CLUBB_SCALAR_FIELDS.items()}


def test_production_deck_writes_exactly_the_defaults():
    """The explicit deck rows must not change a production run."""
    import yaml
    deck = yaml.safe_load(_DECK.read_text())
    assert deck["convection"] == "zhang_mcfarlane"
    assert deck["turbulence"] == "clubb"
    for f, leaf in ZM_SCALAR_FIELDS.items():
        assert deck[f] == getattr(ZhangMcFarlaneConfig(), leaf), f
    for f, leaf in CLUBB_SCALAR_FIELDS.items():
        assert deck[f] == getattr(CLUBBParams(), leaf), f


def test_production_deck_loads_through_the_cli():
    from scripts.run import run_amip
    from legoesm.driver.run_config_yaml import load_yaml_config
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(_DECK, p))
    cfg = run_amip.build_config_from_args(p.parse_args([]))
    # The deck alone lacks the forcing paths its launcher supplies
    # (amip_production.sh), so validate_strict reports those; none of its
    # errors may concern the ZM/CLUBB rows.
    try:
        cfg.validate_strict()
    except ValueError as exc:
        assert not any(k in str(exc) for k in (*ZM_SCALAR_FIELDS,
                                                *CLUBB_SCALAR_FIELDS)), exc
    zm, cp = _leaves(cfg)
    assert zm == {f: getattr(ZhangMcFarlaneConfig(), leaf)
                  for f, leaf in ZM_SCALAR_FIELDS.items()}
    assert cp == {f: getattr(CLUBBParams(), leaf)
                  for f, leaf in CLUBB_SCALAR_FIELDS.items()}


def test_params_route_reaches_the_scheme_configs():
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        build_atm_scalar_param_map,
    )
    smap = build_atm_scalar_param_map()
    inv = {v: k for k, v in smap.items()}
    params = {inv[f]: v for f, v in {**_ZM, **_CLUBB}.items()}
    cfg = apply_params_to_config(
        ExperimentConfig(convection="zhang_mcfarlane", turbulence="clubb",
                         clubb_prognostic=True),
        params, driver="run_amip", scalar_param_map=smap)
    zm, cp = _leaves(cfg)
    assert zm == _ZM
    assert cp == _CLUBB


def test_zm_refused_without_zm():
    for conv in ("bechtold", "none"):
        cfg = ExperimentConfig(convection=conv, zm_dmpdz=-2.0e-3)
        with pytest.raises(ValueError, match="zm_dmpdz"):
            cfg.validate_strict()
        with pytest.raises(ValueError, match="zm_dmpdz"):
            convection_config_for(cfg)


def test_clubb_refused_without_clubb():
    cfg = ExperimentConfig(turbulence="louis", clubb_beta=2.0)
    with pytest.raises(ValueError, match="clubb_beta"):
        cfg.validate_strict()
    with pytest.raises(ValueError, match="clubb_beta"):
        turbulence_config_for(cfg)


def test_prognostic_only_leaves_refused_on_diagnostic_clubb():
    cfg = ExperimentConfig(turbulence="clubb", clubb_c14=3.0)
    with pytest.raises(ValueError, match="clubb_c14"):
        cfg.validate_strict()
    with pytest.raises(ValueError, match="clubb_c14"):
        turbulence_config_for(cfg)
    # gamma_coef and beta are read by the diagnostic path too.
    ok = ExperimentConfig(turbulence="clubb", clubb_beta=2.0)
    assert turbulence_config_for(ok).clubb.params.beta == 2.0


def test_clubb_refused_under_a_turbulence_override():
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    cfg = ExperimentConfig(
        turbulence="clubb", clubb_prognostic=True, clubb_beta=2.0,
        turbulence_override=materialize_sub_config(
            TurbulenceConfig(scheme="clubb"))._replace(
                clubb=materialize_sub_config(TurbulenceConfig(
                    scheme="clubb")).clubb._replace(prognostic=True)))
    with pytest.raises(ValueError, match="clubb_beta"):
        turbulence_config_for(cfg)


@pytest.mark.parametrize("field,value", [
    ("zm_dmpdz", -1.0e-2),      # spec bounds (-0.003, -0.00033)
    ("zm_c0_ocn", 0.1),         # spec bounds (0.002, 0.06)
    ("clubb_c14", 10.0),        # spec bounds (0.5, 6.0)
])
def test_out_of_spec_bounds_refused(field, value):
    cfg = ExperimentConfig(convection="zhang_mcfarlane", turbulence="clubb",
                           **{field: value})
    with pytest.raises(ValueError, match=field):
        cfg.validate_strict()
