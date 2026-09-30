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


def test_caps_still_bind_after_a_large_factor():
    """The factor scales PRC BEFORE the SAM number caps: at fact=20 and a
    long step the rain-number source is still <= N_c/dt and <= NPRC."""
    from legoesm.atmosphere.physics.microphysics import _warm_rain as wr
    q_c, _, _, rho = _col()
    N_c = jnp.full(q_c.shape, 1.0e6)    # 1 cm^-3: a large drop source
    dt = 3600.0
    prc, nr, _ = wr.autoconversion_kk2000(q_c, N_c, rho, dt, fact=20.0)
    cap = N_c / dt
    assert bool(jnp.all(nr <= cap * (1 + 1e-12)))
    assert bool(jnp.any(jnp.isclose(nr, cap, rtol=1e-12)))   # the cap binds
    assert bool(jnp.all(nr <= prc * rho / wr._KK2000_CONS29 * (1 + 1e-12)))


def _morrison_budget(cfg, dt=1.0):
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
        jnp.full(shape, 200.0), dt, cfg._replace(publish_qc_budget=True))
    return out.qc_budget, q_c, dt


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
    (b1, _, _), (b2, _, _) = _morrison_budget(base), _morrison_budget(cfg)
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


def test_donor_clamp_bounds_large_factors():
    """At the validated upper bounds and a long step the applied cloud-water
    sinks never exceed the available q_c (the existing donor clamp binds)."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    cfg = MorrisonConfig(autocon_fact=20.0, accre_enhan_fact=10.0)
    b, q_c, dt = _morrison_budget(cfg, dt=1800.0)
    removed = -(b["autoconversion"] + b["accretion"]) * dt
    assert bool(jnp.all(removed <= q_c * (1 + 1e-9)))


def test_params_route_reaches_the_threaded_leaf():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from legoesm.driver.run_config_yaml import (
        _ATM_SCALAR_PARAM_MAP,
        apply_params_to_config,
    )
    cfg = apply_params_to_config(
        ExperimentConfig(microphysics="morrison"),
        {"atm.micro.MorrisonConfig.autocon_fact": 0.4,
         "atm.micro.MorrisonConfig.accre_enhan_fact": 3.0},
        driver="run_amip", scalar_param_map=_ATM_SCALAR_PARAM_MAP)
    leaf = thread_morrison_scalars(cfg, "morrison", MorrisonConfig())
    assert (leaf.autocon_fact, leaf.accre_enhan_fact) == (0.4, 3.0)


def test_spectral_lane_threads_the_same_scalars():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import _spectral_micro_config
    mc = _spectral_micro_config(ExperimentConfig(
        microphysics="morrison", morrison_autocon_fact=0.5))
    assert mc.morrison.autocon_fact == 0.5
    assert _spectral_micro_config(ExperimentConfig(microphysics="kessler")
                                  ).scheme == "kessler"


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


# --- warm_rain_scheme="kk2000_cam6": CAM6 MG2 kk2000_liq_autoconversion ------

def _cam6_fortran_prc(qc, nc_per_kg, rho, relvar):
    """micro_mg_utils.F90:689-736 transcribed with math.gamma (independent of
    the JAX kernel): prc_coef = gamma(relvar+2.47)/gamma(relvar)/relvar**2.47;
    prc = prc_coef*0.01*1350*qc**2.47*(nc*1e-6*rho)**(-1.1) if qc >= 1e-8."""
    import math
    if qc < 1.0e-8:
        return 0.0
    coef = math.gamma(relvar + 2.47) / math.gamma(relvar) / relvar ** 2.47
    return coef * 0.01 * 1350.0 * qc ** 2.47 * (nc_per_kg * 1.0e-6 * rho) ** (-1.1)


@pytest.mark.parametrize("qc,nc_cm3,rho,relvar", [
    (2.0e-4, 50.0, 1.1, 10.0),
    (8.0e-4, 200.0, 0.9, 2.0),
    (5.0e-5, 20.0, 1.2, 0.5),
])
def test_cam6_rate_reproduces_the_fortran_formula(qc, nc_cm3, rho, relvar):
    from legoesm.atmosphere.physics.microphysics import _warm_rain as wr
    n_per_m3 = nc_cm3 * 1.0e6
    prc, nr, _ = wr.autoconversion_kk2000_cam6(
        jnp.asarray([qc]), jnp.asarray([n_per_m3]), jnp.asarray([rho]), relvar)
    want = _cam6_fortran_prc(qc, n_per_m3 / rho, rho, relvar)
    np.testing.assert_allclose(float(prc[0]), want, rtol=1e-12)
    # nprc = prc/droplet_mass_25um per kg (rhow=1000) -> per volume x rho
    m25 = 4.0 / 3.0 * np.pi * 1000.0 * (25.0e-6) ** 3
    np.testing.assert_allclose(float(nr[0]), want * rho / m25, rtol=1e-12)


def test_cam6_guards_and_factor():
    from legoesm.atmosphere.physics.microphysics import _warm_rain as wr
    rho, n = jnp.asarray([1.0, 1.0]), jnp.asarray([1e8, 1e8])
    prc, nr, _ = wr.autoconversion_kk2000_cam6(
        jnp.asarray([5.0e-9, 1.0e-2]), n, rho, 10.0)
    assert float(prc[0]) == 0.0 and float(nr[0]) == 0.0          # icsmall gate
    capped = _cam6_fortran_prc(5.0e-3, 1e8, 1.0, 10.0)            # 5e-3 cap
    np.testing.assert_allclose(float(prc[1]), capped, rtol=1e-12)
    p1, _, _ = wr.autoconversion_kk2000_cam6(jnp.asarray([3e-4]), n[:1], rho[:1], 10.0)
    p2, _, _ = wr.autoconversion_kk2000_cam6(jnp.asarray([3e-4]), n[:1], rho[:1],
                                            10.0, fact=0.2)
    np.testing.assert_allclose(p2, 0.2 * p1, rtol=1e-14)
    # relvar clipped to CAM6's [0.001, 10] at BOTH ends (relvar is CAM6's
    # inverse relative variance: small = strong enhancement)
    p_lo, _, _ = wr.autoconversion_kk2000_cam6(jnp.asarray([3e-4]), n[:1], rho[:1], 1e-6)
    p_lo_ref, _, _ = wr.autoconversion_kk2000_cam6(jnp.asarray([3e-4]), n[:1], rho[:1], 1e-3)
    np.testing.assert_allclose(p_lo, p_lo_ref, rtol=1e-12)
    assert float(p_lo[0]) > float(p1[0])
    p_hi, _, _ = wr.autoconversion_kk2000_cam6(jnp.asarray([3e-4]), n[:1], rho[:1], 50.0)
    np.testing.assert_allclose(p_hi, p1, rtol=1e-14)
    # cloud-number sink from the CAPPED water: prc*rho/x_c == prc*ncic/qcic
    prc, _, x_c = wr.autoconversion_kk2000_cam6(jnp.asarray([1.0e-2]), n[:1],
                                               rho[:1], 10.0)
    ncic_per_kg = 1e8 / 1.0
    np.testing.assert_allclose(float(prc[0] * rho[0] / x_c[0]),
                               float(prc[0]) * ncic_per_kg / 5.0e-3, rtol=1e-12)


def test_cam6_option_reaches_the_kernel_through_the_threading():
    """The flat selector threads onto the leaf MPAS/FV use, and the Morrison
    kernel then produces the CAM6 rate (differs from the SAM kk2000 one)."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    exp = ExperimentConfig(microphysics="morrison",
                           morrison_warm_rain_scheme="kk2000_cam6",
                           morrison_autocon_fact=0.5)
    exp.validate_strict()
    leaf = thread_morrison_scalars(exp, "morrison", MorrisonConfig())
    assert leaf.warm_rain_scheme == "kk2000_cam6" and leaf.autocon_fact == 0.5
    b_sam, _, _ = _morrison_budget(MorrisonConfig())
    b_cam, _, _ = _morrison_budget(leaf)
    ratio = np.asarray(b_cam["autoconversion"] / b_sam["autoconversion"])
    assert np.all(np.isfinite(ratio)) and not np.allclose(ratio, 1.0, rtol=1e-3)


def test_cam6_accretion_uses_the_capped_water():
    """In the kk2000_cam6 branch accretion sees the 5e-3 in-cloud cap: the
    applied accretion at q_c = 1e-2 equals that at q_c = 5e-3 (dt = 1 s, the
    donor clamp does not bind); the SAM kk2000 branch has no such cap."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

    def acc(cfg, qc):
        shape = (1, 2)
        z = jnp.zeros(shape)
        f = dict(q_c=jnp.full(shape, qc), q_r=jnp.full(shape, 1e-4),
                 N_r=jnp.full(shape, 1e5))
        hyd = HydrometeorState(**{k: f.get(k, z) for k in HydrometeorState._fields})
        out = morrison_microphysics(
            jnp.full(shape, 288.0), jnp.full(shape, 1.0e-2), hyd,
            jnp.full(shape, 9.0e4), jnp.full((1, 3), 9.0e4), jnp.full(shape, 1.0),
            jnp.full(shape, 200.0), 1.0, cfg._replace(publish_qc_budget=True))
        return np.asarray(out.qc_budget["accretion"])

    cam = MorrisonConfig(warm_rain_scheme="kk2000_cam6")
    np.testing.assert_allclose(acc(cam, 1.0e-2), acc(cam, 5.0e-3), rtol=1e-10)
    sam = MorrisonConfig()
    assert not np.allclose(acc(sam, 1.0e-2), acc(sam, 5.0e-3), rtol=1e-3)


def test_kk2000_path_unchanged_by_the_cam6_option():
    """Adding the option leaves the default kk2000 budget bit-identical to the
    same leaf with the (unread) CAM6 relvar changed."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    a, _, _ = _morrison_budget(MorrisonConfig())
    b, _, _ = _morrison_budget(MorrisonConfig(kk2000_cam6_relvar=0.3))
    for k in ("autoconversion", "accretion"):
        np.testing.assert_array_equal(np.asarray(a[k]).view(np.uint64),
                                      np.asarray(b[k]).view(np.uint64))
