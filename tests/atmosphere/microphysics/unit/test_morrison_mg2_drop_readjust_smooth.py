"""RULE 5 tests of the smoothed CAM6 MG2 final droplet mean-size readjust
(micro_mg2_0.F90:2743-2835; in-cloud mean diameter held in [2, 50] um,
ncic <= qcic/1e-20), prognostic droplet number only (#2216, option 2).

``MorrisonConfig.mg2_drop_readjust_log_width`` is the LogSumExp width (ln
space) of both limits, default 0.01 = the cloud-optics width
(``CloudConfig.mg2_drop_*_log_width``); 0 = MG2's hard clip (in the
oracle tests' HARD_LIMIT).  Gap: per fixture column, the default's departure
from that hard limit in the whole-call droplet tendency, relative to the
column's largest MG2 droplet tendency.  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import HARD_LIMIT, TOL, _rel  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FXD = ROOT / "tests/atmosphere/microphysics/oracle_fixtures"
WIDTH = 0.01
# measured 2026-11-01; re-measure on any change of the readjust or its width.
# ARM_GAP: whole-call droplet tendency at the arm's resolved defaults vs MG2.
# Re-measured 2026-11-01 when the arm's EARLY droplet size reset became smooth
# by default (#2216, test_morrison_mg2_drop_early_reset_smooth.py): frz
# column 3 moved 3.7e-8 -> 1.5e-6 and icelim column 6 by 1e-3 relative; every
# other entry moved in the 7th digit or less, or below the 1e-12 floor.
# Re-measured 2026-11-01 when the arm's SEDIMENTATION fall-speed size reset
# became smooth by default (#2216, test_morrison_mg2_sed_reset_smooth.py):
# act/sed column 8 moved 1.6939e-3 -> 1.7009e-3, icelim column 4 by 1e-3 and
# column 5 by 4e-4 relative, frz column 2 by 3.8e-4 and column 0 by 2.9e-6;
# every other entry moved by less than 1e-6 relative, or below the 1e-12 floor.
# (The melt and icelim columns 8-9 gaps come from the other approved
# smoothings, not the readjust: its GAP there is <= 7e-6.)
ARM_GAP = {
    "mg2_prognostic_nc": [2.837937896417557e-16, 6.502802590835931e-14, 1.546676106662368e-16,
                         2.877001770591271e-16, 1.1553350560029991e-10, 0.0],
    "mg2_prognostic_nc_act": [1.418956462149667e-16, 5.281464033166238e-15,
                             1.546676106662368e-16, 7.192474161047813e-16,
                             2.6139676646330366e-10, 0.0, 1.922456754670477e-16,
                             3.954884734458624e-14, 0.0017009230792667833],
    "mg2_prognostic_nc_sed": [1.418956462149667e-16, 5.281464033166238e-15,
                             1.546676106662368e-16, 7.192474161047813e-16,
                             2.6139676646330366e-10, 0.0, 1.922456754670477e-16,
                             3.954884734458624e-14, 0.0017009230792667833, 0.0016965082452150327,
                             1.2833761767092315e-16],
    "mg2_prognostic_nc_melt": [1.1029928257250563, 0.1733159638831987],
    "mg2_prognostic_nc_icelim": [0.015747026523859654, 2.200600984794053e-06,
                                5.437369304914632e-07, 7.014162717927606e-07,
                                0.0004168671711452365, 2.6768813002128014e-08,
                                3.259730304140352e-08, 4.331625567139087e-08,
                                0.45653134092247827, 0.35822587760631697, 0.0, 0.0],
    "mg2_prognostic_nc_frz": [0.0008640828714004535, 0.0033183731458311513,
                             0.00040413815305175943, 1.4759385063413541e-06,
                             5.860854300625975e-05, 5.755777299422316e-05],
}
# GAP: the readjust alone (every other switch at its hard limit), default vs hard.
GAP = {
    "mg2_prognostic_nc": [6.607589070726086e-26, 6.370765990006775e-14, 3.4886004364331466e-26,
                          8.976045932137181e-24, 1.1530107845990739e-10, 0.0],
    "mg2_prognostic_nc_act": [2.4358002012914664e-21, 7.46006794684731e-15, 4.220081173104613e-28,
                              1.223550485686393e-21, 2.608704944694982e-10, 0.0,
                              1.922456754670477e-16, 1.654242103333483e-22, 0.0016938689792927084],
    "mg2_prognostic_nc_sed": [2.4358002012914664e-21, 7.46006794684731e-15, 4.220081173104613e-28,
                              1.223550485686393e-21, 2.608704944694982e-10, 0.0,
                              1.922456754670477e-16, 1.654242103333483e-22, 0.0016938689792927084,
                              0.0016965082452157407, 0.0],
    "mg2_prognostic_nc_melt": [4.31261087578904e-15, 1.022983489417948e-14],
    "mg2_prognostic_nc_icelim": [0.00965331904124295, 2.2006009849294657e-06, 5.881128078786946e-12,
                                 7.014162698487991e-07, 0.00041646053027353885,
                                 2.675838499926385e-08, 2.210631004355063e-10,
                                 4.3316256796771135e-08, 6.797747339418225e-06,
                                 1.3183745953741097e-07, 0.0, 0.0],
    "mg2_prognostic_nc_frz": [0.0008640803562600605, 0.0033183731458318773, 0.00040398555129459286,
                              3.658033878895597e-08, 1.3316515416457637e-14, 5.7557773001394866e-05],
}


def _load(name):
    fx = dict(np.load(FXD / f"{name}.npz"))
    fx.setdefault("col_npccn", np.zeros_like(fx["col_T"]))
    return fx


def _run(fx, width):
    hyd = {k: jnp.asarray(fx[f"col_{v}"]) for k, v in
           (("q_i", "q_i"), ("N_i", "ni_kg"), ("q_s", "q_s"), ("N_s", "ns_kg")) if f"col_{v}" in fx}
    cfg = _cfg(do_snow_to_graupel=False)._replace(mg2_drop_readjust_log_width=width)
    return A._run(fx, cfg=cfg, hyd_kw=hyd)


def _readjust_inputs(fx, monkeypatch):
    """The arguments of the real Morrison readjust call (hard limit)."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    seen = []
    real = M.mg2_droplet_number_readjust

    def spy(*a, **k):
        seen.append(a)
        return real(*a, **k)

    monkeypatch.setattr(M, "mg2_droplet_number_readjust", spy)
    out = _run(fx, 0.0)
    monkeypatch.setattr(M, "mg2_droplet_number_readjust", real)
    return out, seen[-1]


def _two_um_bound_binds(args):
    """Cells where MG2's 2 um minimum mean diameter lowers the number: the
    hard port's readjusted mean diameter (pgam+1)/lambda sits on 2 um."""
    q_post, N_post, _, cf, rho, _, _ = args
    q_ic = jnp.minimum(jnp.maximum(q_post, 0.0) / cf, W.KK2000_CAM6_QC_MAX)
    n_ic = jnp.maximum(N_post, 0.0) / cf
    active, pgam, lam, nc_kg = W.mg2_droplet_psd(q_ic, n_ic, rho)
    d_mean = np.asarray((pgam + 1.0) / lam)
    return (np.asarray(active) & np.isclose(d_mean, 2.0e-6, rtol=1e-12, atol=0.0)
            & (np.asarray(nc_kg * rho) < np.asarray(n_ic)))


def test_defaults_and_hard_limit():
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    c = CloudConfig()
    assert MorrisonConfig().mg2_drop_readjust_log_width == WIDTH \
        == c.mg2_drop_number_cap_log_width == c.mg2_drop_slope_log_width
    assert ExperimentConfig._field_defaults["morrison_mg2_drop_readjust_log_width"] == WIDTH
    assert HARD_LIMIT["mg2_drop_readjust_log_width"] == 0.0


@pytest.mark.parametrize("name", ["mg2_prognostic_nc_act", "mg2_prognostic_nc_sed"])
def test_hard_limit_matches_mg2_where_the_2um_bound_binds(name, monkeypatch):
    """Whole-call droplet tendency vs the RUNNING MG2 at the hard limit, on
    fixtures whose readjust has cells the 2 um bound moves; the default
    width does not match (the comparison can fail: cells near a bound move,
    cells far past it keep MG2's value to exp(-distance/width))."""
    fx = _load(name)
    out, args = _readjust_inputs(fx, monkeypatch)
    binds = _two_um_bound_binds(args)
    assert binds.sum() >= 5
    rho = fx["col_rho"]
    err = _rel(np.asarray(out.dN_c_dt) / rho, fx["ora_nctend"])
    assert float(err.max()) <= TOL
    soft = _rel(np.asarray(_run(fx, WIDTH).dN_c_dt) / rho, fx["ora_nctend"])
    assert float(soft.max()) > 1e3 * TOL


@pytest.mark.parametrize("name", sorted(GAP))
def test_default_gap_per_column_is_the_recorded_one(name):
    fx = _load(name)
    hard, soft = _run(fx, 0.0), _run(fx, WIDTH)
    d = np.abs(np.asarray(soft.dN_c_dt) - np.asarray(hard.dN_c_dt))
    scale = np.abs(fx["ora_nctend"] * fx["col_rho"])
    gap = d.max(axis=1) / np.maximum(scale.max(axis=1), 1e-300)
    print(name, "readjust gap per column", [float(x) for x in gap])
    np.testing.assert_allclose(gap, GAP[name], rtol=1e-6, atol=1e-12)
    # a gap below the 1e-12 floor must not vanish (or appear) unnoticed
    assert np.array_equal(gap > 0.0, np.asarray(GAP[name]) > 0.0)


def _arm_defaults():
    """The prognostic arm's resolved Morrison smoothing (every HARD_LIMIT field
    at the production resolution), the readjust width at its default."""
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison
    prod = _production_morrison()
    assert prod.mg2_drop_readjust_log_width == WIDTH
    return _cfg(do_snow_to_graupel=False)._replace(**{k: getattr(prod, k) for k in HARD_LIMIT})


@pytest.mark.parametrize("name", sorted(GAP))
def test_arm_default_gap_to_mg2_per_column_is_the_recorded_one(name):
    """The whole-call droplet tendency at the arm's defaults vs the RUNNING MG2,
    per fixture column, relative to the column's largest MG2 value (every
    approved smoothing contributes, not only the readjust)."""
    fx = _load(name)
    hyd = {k: jnp.asarray(fx[f"col_{v}"]) for k, v in
           (("q_i", "q_i"), ("N_i", "ni_kg"), ("q_s", "q_s"), ("N_s", "ns_kg")) if f"col_{v}" in fx}
    out = A._run(fx, cfg=_arm_defaults(), hyd_kw=hyd)
    ora = fx["ora_nctend"] * fx["col_rho"]
    d = np.abs(np.asarray(out.dN_c_dt) - ora)
    gap = d.max(axis=1) / np.maximum(np.abs(ora).max(axis=1), 1e-300)
    print(name, "arm-default gap to MG2 per column", [float(x) for x in gap])
    np.testing.assert_allclose(gap, ARM_GAP[name], rtol=1e-6, atol=1e-12)
    assert np.array_equal(gap > 0.0, np.asarray(ARM_GAP[name]) > 0.0)


def test_qsmall_boundary_keeps_mg2_semantics():
    """At qcic == qsmall exactly MG2's size routine leaves the number (a negative
    one survives); the smooth path keeps that."""
    one = jnp.ones(2)
    args = (W.MG2_QSMALL * one, -1.0e3 * one, 2.0e3 * one, one, one, 600.0, -5.0 * one)
    for w in (0.0, WIDTH):
        got = W.mg2_droplet_number_readjust(*args, presence_top=W.MG2_QSMALL, log_width=w)
        np.testing.assert_array_equal(np.asarray(got), -5.0 * np.ones(2))


@pytest.mark.parametrize("width", [WIDTH, 1.0])
def test_mean_diameter_bounds_as_documented(width):
    """Few droplets (50 um side) and many (2 um side, and past the number cap):
    D >= 2 um always; D <= 50 um x exp(w ln(1 + 25^(-1/w))) (= 50 um to
    round-off at the default)."""
    q = jnp.full(4, 1.0e-4)
    n = jnp.asarray([1.0e-9, 1.0, 1.0e12, 1.0e30])
    rho = jnp.ones(4)
    _, pgam, lam, nc = W.mg2_droplet_psd(q, n, rho, number_cap_log_width=width,
                                         slope_log_width=width)
    d = np.asarray((pgam + 1.0) / lam)
    over = width * np.log1p(25.0 ** (-1.0 / width))
    assert np.all(d >= 2.0e-6 * (1 - 1e-12))
    assert np.all(d <= 50.0e-6 * np.exp(over) * (1 + 1e-12))
    assert np.asarray(nc).max() <= 1.0e-4 / 1.0e-20
    if width == 1.0:
        assert d[0] > 51.9e-6        # the documented excursion is reached
    else:
        assert d.max() <= 50.0e-6 * (1 + 1e-12)


def _near_2um_bound(u):
    """Readjust inputs whose mean diameter sits ``3*u*WIDTH`` in ln N from the
    2 um bound (u > 0: too many droplets, the side MG2 clips)."""
    q = 1.0e-11 * jnp.ones_like(u)
    one = jnp.ones_like(u)
    # number at the 2 um bound (pgam is flat to ~1e-4 between 2e3 and 5e3 m^-3)
    n_b = W.mg2_size_bounded_nc(q, 5.0e3 * one, one)
    n_post = n_b * jnp.exp(3.0 * WIDTH * u)
    n0 = 2.0 * n_b
    dt = 600.0
    return q, n_post, n0, one, one, dt, (n_post - n0) / dt


def test_gradients_through_the_readjust():
    u = jnp.asarray([-2.0, -0.5, 0.5, 2.0])

    def f(x, width):
        return W.mg2_droplet_number_readjust(*_near_2um_bound(x), presence_top=W.MG2_QSMALL,
                                             log_width=width)

    check_grads(lambda x: f(x, WIDTH), (u,), order=2, modes=["rev"], eps=1e-3, atol=1e-6,
                rtol=1e-3)
    g = np.asarray(jax.grad(lambda x: jnp.sum(f(x, WIDTH)))(u))
    hard = np.asarray(jax.grad(lambda x: jnp.sum(f(x, 0.0)))(u))
    assert np.all(np.isfinite(g)) and np.all(g != 0.0)
    # MG2's clip has no derivative on its clipped side; the rounded one does
    assert np.all(np.abs(hard[u > 0]) < 1e-2 * np.abs(g[u > 0]))


def test_whole_call_gradient_finite_at_the_default(monkeypatch):
    fx = _load("mg2_prognostic_nc_frz")
    _, args = _readjust_inputs(fx, monkeypatch)
    binds = jnp.asarray(_two_um_bound_binds(args))
    rho = jnp.asarray(fx["col_rho"])
    cfg = _cfg(do_snow_to_graupel=False)
    hyd = {k: jnp.asarray(fx[f"col_{v}"]) for k, v in
           (("q_i", "q_i"), ("N_i", "ni_kg"), ("q_s", "q_s"), ("N_s", "ns_kg")) if f"col_{v}" in fx}

    def f(nc):
        out = A._run(fx, cfg=cfg._replace(mg2_drop_readjust_log_width=WIDTH), N_c=nc, hyd_kw=hyd)
        return jnp.sum(jnp.where(binds, out.dN_c_dt, 0.0))

    g = np.asarray(jax.grad(f)(jnp.asarray(fx["col_nc_kg"]) * rho))
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0


def test_flat_field_validation_cli_and_deck():
    from scripts.run.run_amip import _postprocess_args, build_arg_parser, build_config_from_args

    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from legoesm.driver.run_config_yaml import load_yaml_config
    kw = dict(microphysics="morrison", morrison_prognostic_droplet_number=True,
              cam6_aerosol_dir="/x")
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="finite float in"):
            ExperimentConfig(**kw, morrison_mg2_drop_readjust_log_width=bad).validate_strict()
    with pytest.raises(ValueError, match="silently inert"):
        ExperimentConfig(microphysics="morrison",
                         morrison_mg2_drop_readjust_log_width=0.0).validate_strict()
    base = MorrisonConfig()
    assert thread_morrison_scalars(ExperimentConfig(morrison_mg2_drop_readjust_log_width=0.0),
                                   "morrison", base) == base._replace(
        mg2_drop_readjust_log_width=0.0)
    assert thread_morrison_scalars(ExperimentConfig(), "morrison", base) is base
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--morrison-mg2-drop-readjust-log-width", "0.0"]), parser))
    assert cfg.morrison_mg2_drop_readjust_log_width == 0.0
    amip = ROOT / "config/amip"
    arm = load_yaml_config(amip / "amip_production_prognostic_nc.yaml", build_arg_parser())
    assert arm["morrison_mg2_drop_readjust_log_width"] == WIDTH        # named explicitly
    assert arm["morrison_prognostic_droplet_number"] is True
    prod = load_yaml_config(amip / "amip_production.yaml", build_arg_parser())
    assert "morrison_mg2_drop_readjust_log_width" not in prod
    assert prod["morrison_prognostic_droplet_number"] is False
