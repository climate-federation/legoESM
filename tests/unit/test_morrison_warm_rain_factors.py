"""CAM6 MG2-style warm-rain multipliers on the Morrison kk2000 branch.

``morrison_autocon_fact`` scales the KK2000 autoconversion rate (before the
number caps, so rain-number source and cloud-number sink follow it);
``morrison_accre_enhan_fact`` scales KK2000 accretion (MG2 ``accre_enhan``).
Defaults 1.0 are bit-identical to the unscaled law.
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"


def _col():
    rng = np.random.default_rng(3)
    n = 16
    q_c = jnp.asarray(rng.uniform(1e-4, 8e-4, n))
    q_r = jnp.asarray(rng.uniform(1e-6, 2e-4, n))
    N_c = jnp.asarray(rng.uniform(3e7, 3e8, n))
    rho = jnp.asarray(rng.uniform(0.8, 1.2, n))
    return q_c, q_r, N_c, rho


def test_default_is_bitwise_the_unscaled_law():
    from legoesm.atmosphere.physics.microphysics import _warm_rain as wr
    q_c, q_r, N_c, rho = _col()
    prc, _, _ = wr.autoconversion_kk2000(q_c, N_c, rho, 60.0)
    old = (wr._KK2000_AUTOCONV_PREFACTOR
           * wr.safe_pow(q_c, wr._KK2000_AUTOCONV_QC_EXPONENT)
           * wr.safe_pow(N_c / 1.0e6, wr._KK2000_AUTOCONV_NC_EXPONENT))
    np.testing.assert_array_equal(np.asarray(prc).view(np.uint64),
                                  np.asarray(old).view(np.uint64))
    pra = wr.accretion_kk2000(q_c, q_r)
    old = wr._KK2000_ACCRETION_PREFACTOR * wr.safe_pow(
        q_c * q_r, wr._KK2000_ACCRETION_EXPONENT)
    np.testing.assert_array_equal(np.asarray(pra).view(np.uint64),
                                  np.asarray(old).view(np.uint64))


def test_kernel_factors_scale_mass_and_number():
    from legoesm.atmosphere.physics.microphysics import _warm_rain as wr
    q_c, q_r, N_c, rho = _col()
    p1, n1, x1 = wr.autoconversion_kk2000(q_c, N_c, rho, 60.0)
    p3, n3, x3 = wr.autoconversion_kk2000(q_c, N_c, rho, 60.0, fact=0.3)
    np.testing.assert_allclose(p3, 0.3 * p1, rtol=1e-14)
    np.testing.assert_allclose(n3, 0.3 * n1, rtol=1e-14)  # below both caps here
    np.testing.assert_array_equal(x3, x1)
    np.testing.assert_allclose(wr.accretion_kk2000(q_c, q_r, fact=4.0),
                               4.0 * wr.accretion_kk2000(q_c, q_r), rtol=1e-14)


def _morrison_budget(cfg):
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    q_c, q_r, _, rho = (a[None, :] for a in _col())   # one column, 16 levels
    shape = q_c.shape
    z = jnp.zeros(shape)
    fields = dict(q_c=q_c, q_r=q_r, N_r=jnp.full(shape, 1e5))
    hyd = HydrometeorState(**{f: fields.get(f, z)
                              for f in HydrometeorState._fields})
    out = morrison_microphysics(
        jnp.full(shape, 288.0), jnp.full(shape, 1.0e-2), hyd,
        jnp.full(shape, 9.0e4), jnp.full((1, shape[1] + 1), 9.0e4), rho,
        jnp.full(shape, 200.0), 1.0, cfg._replace(publish_qc_budget=True))
    return out.qc_budget


def test_factor_reaches_the_threaded_scheme_config():
    """The flat ExperimentConfig field reaches the MorrisonConfig the MPAS and
    FV lanes both build via thread_morrison_scalars, and scales the applied
    autoconversion / accretion terms of the real kernel by exactly the factor."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    base = MorrisonConfig()
    cfg = thread_morrison_scalars(
        ExperimentConfig(microphysics="morrison", morrison_autocon_fact=0.25,
                         morrison_accre_enhan_fact=2.0), "morrison", base)
    assert (cfg.autocon_fact, cfg.accre_enhan_fact) == (0.25, 2.0)
    b1, b2 = _morrison_budget(base), _morrison_budget(cfg)
    assert float(jnp.min(-b1["autoconversion"])) > 0.0
    # APPLIED terms carry the donor clamp, whose scale moves with the total
    # sink where it binds, so the ratio is the factor to ~1e-3 here (exact at
    # the kernel, see test_kernel_factors_scale_mass_and_number).
    np.testing.assert_allclose(b2["autoconversion"], 0.25 * b1["autoconversion"],
                               rtol=2e-3)
    np.testing.assert_allclose(b2["accretion"], 2.0 * b1["accretion"], rtol=2e-3)
    # untouched config: the leaf object passes through unchanged
    assert thread_morrison_scalars(ExperimentConfig(microphysics="morrison"),
                                   "morrison", base) is base
    sb = thread_morrison_scalars(
        ExperimentConfig(microphysics="morrison",
                         morrison_warm_rain_scheme="seifert_beheng"),
        "morrison", base)
    assert sb.warm_rain_scheme == "seifert_beheng"


def test_validate_strict_refuses_inert_or_foreign_use():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    ExperimentConfig(microphysics="morrison",
                     morrison_autocon_fact=0.2).validate_strict()
    with pytest.raises(ValueError, match="kk2000"):
        ExperimentConfig(microphysics="morrison", morrison_autocon_fact=0.2,
                         morrison_warm_rain_scheme="seifert_beheng"
                         ).validate_strict()
    with pytest.raises(ValueError, match="require microphysics='morrison'"):
        ExperimentConfig(microphysics="thompson",
                         morrison_accre_enhan_fact=2.0).validate_strict()
    with pytest.raises(ValueError, match="outside MorrisonConfig spec"):
        ExperimentConfig(microphysics="morrison",
                         morrison_autocon_fact=50.0).validate_strict()
    with pytest.raises(ValueError, match="unknown"):
        ExperimentConfig(microphysics="morrison",
                         morrison_warm_rain_scheme="kk2001").validate_strict()
    d = ExperimentConfig._field_defaults
    m = MorrisonConfig()
    assert d["morrison_autocon_fact"] == m.autocon_fact == 1.0
    assert d["morrison_accre_enhan_fact"] == m.accre_enhan_fact == 1.0
    assert d["morrison_warm_rain_scheme"] == m.warm_rain_scheme == "kk2000"


def test_cli_round_trip_and_production_deck_changes_nothing():
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--morrison-autocon-fact", "0.3", "--morrison-accre-enhan-fact", "1.5",
        "--morrison-warm-rain-scheme", "kk2000"]), parser))
    assert (cfg.morrison_autocon_fact, cfg.morrison_accre_enhan_fact,
            cfg.morrison_warm_rain_scheme) == (0.3, 1.5, "kk2000")
    keys = load_yaml_config(_AMIP / "amip_production.yaml", build_arg_parser())
    assert keys["morrison_warm_rain_scheme"] == "kk2000"
    assert keys["morrison_autocon_fact"] == 1.0
    assert keys["morrison_accre_enhan_fact"] == 1.0
    # Naming the defaults changes nothing: the deck's values thread to the
    # SAME MorrisonConfig object the untouched config yields.
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    base = MorrisonConfig()
    named = {k: keys[k] for k in ("morrison_warm_rain_scheme",
                                  "morrison_autocon_fact",
                                  "morrison_accre_enhan_fact")}
    assert thread_morrison_scalars(
        ExperimentConfig(microphysics="morrison", **named), "morrison",
        base) is base


def test_params_route_maps_the_leaves():
    from legoesm.driver.run_config_yaml import _ATM_SCALAR_PARAM_MAP
    assert _ATM_SCALAR_PARAM_MAP["atm.micro.MorrisonConfig.autocon_fact"] == (
        "morrison_autocon_fact")
    assert _ATM_SCALAR_PARAM_MAP["atm.micro.MorrisonConfig.accre_enhan_fact"] == (
        "morrison_accre_enhan_fact")
