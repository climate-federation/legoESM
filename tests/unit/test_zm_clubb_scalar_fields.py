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
    ZM_WIDTH_FIELDS,
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
    # The MPAS column loop (the pipeline refuses ZM CMT + prognostic CLUBB).
    cfg = cfg._replace(dycore=cfg.dycore._replace(discretization="mpas"))
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


# --- ZM detrainment smoothing widths (ZM_WIDTH_FIELDS) ----------------------
# Read only by the MPAS macro/micro sub-cycle with ZM + CLUBB; != defaults.
_ZW = {"zm_detrain_phase_width_K": 0.2,
       "zm_detrain_number_width_kgkg_s": 2.0e-12}
_ZW_HOST = dict(convection="zhang_mcfarlane", turbulence="clubb",
                cld_macmic_num_steps=3)


def _mpas(cfg):
    return cfg._replace(grid=cfg.grid._replace(grid_type="mpas"),
                        dycore=cfg.dycore._replace(discretization="mpas"))


def _zw_leaves(cfg):
    zm = convection_config_for(cfg).zhang_mcfarlane
    return {f: getattr(zm, leaf) for f, leaf in ZM_WIDTH_FIELDS.items()}


def test_width_cli_values_reach_the_zm_config():
    argv = ["--convection", "zhang_mcfarlane", "--turbulence", "clubb",
            "--cld-macmic-num-steps", "3"]
    for f, v in _ZW.items():
        argv += ["--" + f.replace("_", "-"), str(v)]
    cfg = _mpas(_parse(argv))
    assert {f: getattr(cfg, f) for f in _ZW} == _ZW
    cfg.validate_strict()
    assert _zw_leaves(cfg) == _ZW


def test_width_unset_keeps_scheme_defaults():
    cfg = _mpas(_parse(["--convection", "zhang_mcfarlane"]))
    assert all(getattr(cfg, f) is None for f in ZM_WIDTH_FIELDS)
    assert _zw_leaves(cfg) == {f: getattr(ZhangMcFarlaneConfig(), leaf)
                               for f, leaf in ZM_WIDTH_FIELDS.items()}


def test_production_deck_names_the_widths_at_their_defaults():
    import yaml
    deck = yaml.safe_load(_DECK.read_text())
    for f, leaf in ZM_WIDTH_FIELDS.items():
        assert deck[f] == getattr(ZhangMcFarlaneConfig(), leaf), f
    from scripts.run import run_amip
    from legoesm.driver.run_config_yaml import load_yaml_config
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(_DECK, p))
    cfg = run_amip.build_config_from_args(p.parse_args([]))
    assert {f: getattr(cfg, f) for f in ZM_WIDTH_FIELDS} == {
        f: deck[f] for f in ZM_WIDTH_FIELDS}
    try:
        cfg.validate_strict()
    except ValueError as exc:
        assert not any(k in str(exc) for k in ZM_WIDTH_FIELDS), exc


@pytest.mark.parametrize("host", [
    dict(_ZW_HOST, convection="bechtold"),
    dict(_ZW_HOST, turbulence="louis"),
    dict(_ZW_HOST, cld_macmic_num_steps=1),
])
def test_width_refused_where_nothing_reads_it(host):
    cfg = _mpas(ExperimentConfig(**host, zm_detrain_phase_width_K=0.2))
    with pytest.raises(ValueError, match="zm_detrain_phase_width_K"):
        cfg.validate_strict()


def test_width_refused_off_the_mpas_lane():
    """The booking runs only in the MPAS macmic loop (combined.py _defer_dlf);
    elsewhere cld_macmic_num_steps>=2 itself is refused."""
    cfg = ExperimentConfig(**_ZW_HOST, zm_detrain_phase_width_K=0.2)
    assert not cfg.column_loop_lane
    with pytest.raises(ValueError, match="cld_macmic_num_steps=3 is wired"):
        cfg.validate_strict()


@pytest.mark.parametrize("field", sorted(ZM_WIDTH_FIELDS))
@pytest.mark.parametrize("value", [-1.0e-3, float("nan"), float("inf"), 1.0e-320,
                                   1.0e-39, 1.0e-20, 1.0e19, 1.0e300])
def test_width_out_of_range_refused(field, value):
    cfg = _mpas(ExperimentConfig(**_ZW_HOST, **{field: value}))
    with pytest.raises(ValueError, match=field):
        cfg.validate_strict()
    from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
        validate_detrain_widths,
    )
    zm = ZhangMcFarlaneConfig()._replace(**{ZM_WIDTH_FIELDS[field]: value})
    with pytest.raises(ValueError, match="finite and >= 0"):
        validate_detrain_widths(zm.detrain_phase_width_K,
                                zm.detrain_number_width_kgkg_s)


@pytest.mark.parametrize("field", sorted(ZM_WIDTH_FIELDS))
@pytest.mark.parametrize("value", [False, True])
def test_width_bool_refused(field, value):
    """A YAML ``false`` reaches the config as a bool (the deck loader coerces
    strings only); it must not read as 0 = the hard switch."""
    cfg = _mpas(ExperimentConfig(**_ZW_HOST, **{field: value}))
    with pytest.raises(ValueError, match=field):
        cfg.validate_strict()


def test_width_zero_hard_switch_accepted():
    _mpas(ExperimentConfig(**_ZW_HOST, zm_detrain_phase_width_K=0.0,
                           zm_detrain_number_width_kgkg_s=0.0)).validate_strict()
    _mpas(ExperimentConfig(**_ZW_HOST, zm_detrain_phase_width_K=1.1e-19,
                           zm_detrain_number_width_kgkg_s=1.1e-19)).validate_strict()
    _mpas(ExperimentConfig(**_ZW_HOST, zm_detrain_phase_width_K=9.0e18,
                           zm_detrain_number_width_kgkg_s=9.0e18)).validate_strict()


def test_widths_have_no_params_route():
    """Excluded from __param_spec__ (tier 0): --params must not reach them."""
    from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
    assert not set(ZM_WIDTH_FIELDS) & set(build_atm_scalar_param_map().values())
