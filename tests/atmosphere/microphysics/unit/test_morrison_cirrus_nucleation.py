"""MorrisonConfig.cirrus_nucleation="cam6_liu_penner": MG2's use of CAM6's activated ice number naai
(micro_mg2_0.F90:1117-1129) in the homogeneous-nucleation block, its guards, and the process-budget term.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import apply_microphysics_experiment_flags  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402
from legoesm.thermo import saturation_vapor_pressure_ice_goff_gratch  # noqa: E402
from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison  # noqa: E402


def _cfg(mode):
    return _production_morrison()._replace(homogeneous_ice_nucleation=True, cirrus_nucleation=mode,
                                           publish_process_budget=True)


def _col(rhi=1.3):
    """Two cold columns, three levels near 200 hPa, ice-supersaturated, no cloud ice yet."""
    T = jnp.asarray([[212.0, 218.0, 224.0], [214.0, 220.0, 226.0]])
    p_half = jnp.asarray([[1.6e4, 1.9e4, 2.2e4, 2.5e4]] * 2)
    p = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    rho = p / (C.R_d * T)
    esi = saturation_vapor_pressure_ice_goff_gratch(T)
    q_v = rhi * C.epsilon * esi / (p - (1 - C.epsilon) * esi)
    z = jnp.zeros_like(T)
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z, N_c=z + 1e8, N_r=z, N_i=z, N_s=z)
    dz = (p_half[:, 1:] - p_half[:, :-1]) / (C.g * rho)
    return T, q_v, hyd, p, p_half, rho, dz, jnp.full_like(T, 0.4)


def _run(cfg, naai=None, rhi=1.3):
    T, q_v, hyd, p, ph, rho, dz, cf = _col(rhi)
    kw = {} if naai is None else {"naai": naai}
    return morrison_microphysics(T, q_v, hyd, p, ph, rho, dz, 600.0, cfg, cloud_fraction=cf,
                                 relvar=jnp.ones_like(T), **kw)


def test_lp_number_follows_naai():
    T = _col()[0]
    small, big = (_run(_cfg("cam6_liu_penner"), jnp.full_like(T, n)) for n in (2.0e4, 2.0e5))
    # MG2: target naai*icldm, nothing present before: number source scales with naai
    ratio = np.asarray(big.process_budget["cirrus_nucleation"]) / np.asarray(
        small.process_budget["cirrus_nucleation"])
    assert np.all(np.asarray(small.process_budget["cirrus_nucleation"]) > 0.0)
    assert np.all(ratio > 5.0)


def test_lp_off_where_naai_zero_or_subsaturated():
    T = _col()[0]
    zero = _run(_cfg("cam6_liu_penner"), jnp.zeros_like(T))
    assert np.all(np.asarray(zero.process_budget["cirrus_nucleation"]) == 0.0)
    dry = _run(_cfg("cam6_liu_penner"), jnp.full_like(T, 2.0e5), rhi=1.02)   # below MG2's 1.05
    assert np.all(np.asarray(dry.process_budget["cirrus_nucleation"]) == 0.0)


def test_fixed_mode_publishes_zero_and_refuses_naai():
    T = _col()[0]
    out = _run(_cfg("fixed"))
    assert np.all(np.asarray(out.process_budget["cirrus_nucleation"]) == 0.0)
    with pytest.raises(ValueError, match="naai= is read only"):
        _run(_cfg("fixed"), jnp.zeros_like(T))


def test_lp_guards():
    with pytest.raises(ValueError, match="cam6_liu_penner"):
        _run(_cfg("cam6_liu_penner"))
    with pytest.raises(ValueError, match="homogeneous_ice_nucleation"):
        _run(_cfg("cam6_liu_penner")._replace(homogeneous_ice_nucleation=False),
             jnp.zeros_like(_col()[0]))
    with pytest.raises(ValueError, match="cirrus_nucleation"):
        _run(_cfg("bogus"))


def test_run_amip_cli_round_trip():
    from scripts.run.run_amip import build_arg_parser, build_config_from_args
    args = build_arg_parser().parse_args(["--morrison-cirrus-nucleation", "cam6_liu_penner",
                                          "--cirrus-aerosol-file", "x.npz"])
    cfg = build_config_from_args(args)
    assert cfg.morrison_cirrus_nucleation == "cam6_liu_penner"
    assert cfg.cirrus_aerosol_file == "x.npz"
    d = build_config_from_args(build_arg_parser().parse_args([]))
    assert d.morrison_cirrus_nucleation == "fixed" and d.cirrus_aerosol_file is None


def test_applier_threads_and_validates():
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    mc = MicrophysicsConfig(scheme="morrison")
    out = apply_microphysics_experiment_flags(mc.morrison, "morrison",
                                              morrison_cirrus_nucleation="cam6_liu_penner")
    assert out.cirrus_nucleation == "cam6_liu_penner"
    with pytest.raises(ValueError):
        apply_microphysics_experiment_flags(mc.morrison, "morrison", morrison_cirrus_nucleation="x")
