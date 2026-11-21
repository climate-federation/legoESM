"""MorrisonConfig.cirrus_nucleation="cam6_liu_penner": MG2's use of CAM6's activated ice number naai
(micro_mg2_0.F90:1117-1129) in the homogeneous-nucleation block and its guards, read on the ice number
tendency (the columns are below Cooper's 235.15 K cutoff and hold no ice, so every new crystal is cirrus);
RULE 5 tests of the MG2 gate.
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
    return _production_morrison()._replace(homogeneous_ice_nucleation=True, cirrus_nucleation=mode)


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


def _run(cfg, naai=None, rhi=1.3, **kw):
    T, q_v, hyd, p, ph, rho, dz, cf = _col(rhi)
    if naai is not None:
        kw["naai"] = naai
    return morrison_microphysics(T, q_v, hyd, p, ph, rho, dz, 600.0, cfg, cloud_fraction=cf,
                                 relvar=jnp.ones_like(T), **kw)


def test_lp_number_follows_naai():
    T = _col()[0]
    small, big = (_run(_cfg("cam6_liu_penner"), jnp.full_like(T, n)) for n in (2.0e4, 2.0e5))
    # MG2: target naai*icldm, nothing present before: number source scales with naai
    ratio = np.asarray(big.dN_i_dt) / np.asarray(small.dN_i_dt)
    assert np.all(np.asarray(small.dN_i_dt) > 0.0)
    assert np.all(ratio > 5.0)


def test_lp_off_where_naai_zero_or_subsaturated():
    T = _col()[0]
    zero = _run(_cfg("cam6_liu_penner"), jnp.zeros_like(T))
    assert np.all(np.asarray(zero.dN_i_dt) == 0.0)
    # below MG2's 1.05 and below the default ramp (1.05 (1 - 0.01))
    dry = _run(_cfg("cam6_liu_penner"), jnp.full_like(T, 2.0e5), rhi=1.02)
    assert np.all(np.asarray(dry.dN_i_dt) == 0.0)


def test_fixed_mode_refuses_naai():
    T = _col()[0]
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


def test_naai_gate_reads_the_scaled_qvl():
    """micro_mg2_0.F90:1107-1118: with sub-grid clouds relhum is already on the
    qsfm-scaled qvl, so air at rhi 1.02 (gate closed with qsfm = 1) nucleates
    once qsfm = 0.9 (1.02/0.9 > 1.05)."""
    T = _col()[0]
    naai = jnp.full_like(T, 2.0e5)
    off = _run(_cfg("cam6_liu_penner"), naai, rhi=1.02)
    on = _run(_cfg("cam6_liu_penner"), naai, rhi=1.02, qsatfac_ramp=jnp.full_like(T, 0.9))
    assert (np.asarray(off.dN_i_dt) == 0.0).all() and (np.asarray(on.dN_i_dt) > 0.0).all()


# RULE 5, MG2 gate on the humidity (relhum esl/esi > 1.05): cases rhi below,
# inside and above the default ramp [1.05 (1 - 0.01), 1.05].
GATE_RHI = (1.035, 1.0425, 1.047, 1.06)


def _gate_dn(rel_width):
    T = _col()[0]
    cfg = _cfg("cam6_liu_penner")._replace(cirrus_rel_width=rel_width, cirrus_T_width_K=0.0)
    return [np.asarray(_run(cfg, jnp.full_like(T, 2.0e5), rhi=r).dN_i_dt) for r in GATE_RHI]


def test_gate_hard_limit_is_mg2():
    hard = _gate_dn(0.0)
    assert (hard[0] == 0).all() and (hard[1] == 0).all() and (hard[2] == 0).all()
    assert (hard[3] > 0).all()


# per case, max |default - hard| / max |hard| over the columns, measured 2026-11-21
GATE_GAPS = (0.0, 0.1956610971067675, 0.7985398426202741, 0.0)


def test_gate_default_gap_per_case():
    hard, soft = _gate_dn(0.0), _gate_dn(_cfg("fixed").cirrus_rel_width)
    scale = float(np.abs(hard[3]).max())
    gaps = [float(np.abs(s - h).max()) / scale for s, h in zip(soft, hard)]
    print("cirrus gate gaps", repr(tuple(gaps)))
    np.testing.assert_allclose(gaps, GATE_GAPS, rtol=1e-6, atol=1e-12)


def test_gate_gradient_at_default():
    """Inside the default humidity ramp: AD of the ice number source along
    q_v (1 + s) equals the central difference, finite and non-zero."""
    T, q_v, hyd, p, ph, rho, dz, cf = _col(1.047)
    cfg = _cfg("cam6_liu_penner")

    def f(s):
        return jnp.sum(morrison_microphysics(T, q_v * (1.0 + s), hyd, p, ph, rho, dz, 600.0, cfg,
                                             cloud_fraction=cf, relvar=jnp.ones_like(T),
                                             naai=jnp.full_like(T, 2.0e5)).dN_i_dt)

    h = 1.0e-5
    g = float(jax.grad(f)(0.0))
    fd = float((f(h) - f(-h)) / (2 * h))
    print("gate grad", g, fd)
    assert np.isfinite(g) and g != 0.0
    assert abs(g - fd) <= 1e-3 * abs(fd)
