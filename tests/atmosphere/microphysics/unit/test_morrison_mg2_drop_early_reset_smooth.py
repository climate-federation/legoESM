"""RULE 5 tests of the smoothed CAM6 MG2 EARLY droplet size reset
(size_dist_param_liq before autoconversion, micro_mg2_0.F90:1296; in-cloud
mean diameter held in [2, 50] um, ncic <= qcic/1e-20), prognostic droplet
number only (#2216).  Its number feeds every droplet sink, its (pgam, lamc)
snow riming.

``MorrisonConfig.mg2_drop_early_reset_log_width`` is the LogSumExp width (ln
space) of both limits, default 0.01 = the final readjust's and the
cloud-optics width; 0 = MG2's hard clip (in the oracle tests' HARD_LIMIT).
Gap: per fixture column, the default's departure from that hard limit in the
whole-call droplet tendency (every other switch at its hard limit), relative
to the column's largest MG2 droplet tendency.  Same format as
test_morrison_mg2_drop_readjust_smooth.py.  x64.
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
# GAP: the early reset alone (every other switch at its hard limit), default
# vs hard.  Measured 2026-11-01; re-measure on any change of the reset or its width.
GAP = {
    "mg2_prognostic_nc": [1.4189689482087786e-16, 2.442677115339385e-15, 0.0,
                          1.4385008852956354e-16, 4.4403003651462774e-23, 0.0],
    "mg2_prognostic_nc_act": [0.0, 7.724141148505623e-15, 0.0, 1.15079586576765e-15,
                              3.3101195937839705e-16, 0.0, 0.0, 0.0, 1.8904448190317247e-05],
    "mg2_prognostic_nc_sed": [0.0, 7.724141148505623e-15, 0.0, 1.15079586576765e-15,
                              3.3101195937839705e-16, 0.0, 0.0, 0.0, 1.8904448190317247e-05,
                              1.0014780483961052e-12, 0.0],
    "mg2_prognostic_nc_melt": [0.0, 2.1536494514062063e-16],
    "mg2_prognostic_nc_icelim": [1.225183576030362e-08, 3.310093747954233e-16,
                                 2.0805266594234733e-15, 2.0568525968385814e-09,
                                 2.410809332404832e-16, 1.2501383963215426e-14,
                                 3.227000734825026e-11, 4.3049578560881357e-16,
                                 2.1178623062142157e-09, 6.784044268192166e-06, 0.0, 0.0],
    "mg2_prognostic_nc_frz": [4.1525187284142895e-06, 2.2850797281078893e-10,
                              4.482433066897439e-08, 1.4759385066427562e-06,
                              5.775705707537222e-08, 5.886106341891886e-08],
}


def _load(name):
    fx = dict(np.load(FXD / f"{name}.npz"))
    fx.setdefault("col_npccn", np.zeros_like(fx["col_T"]))
    return fx


def _hyd(fx):
    return {k: jnp.asarray(fx[f"col_{v}"]) for k, v in
            (("q_i", "q_i"), ("N_i", "ni_kg"), ("q_s", "q_s"), ("N_s", "ns_kg")) if f"col_{v}" in fx}


def _run(fx, width):
    cfg = _cfg(do_snow_to_graupel=False)._replace(mg2_drop_early_reset_log_width=width)
    return A._run(fx, cfg=cfg, hyd_kw=_hyd(fx))


def _reset_inputs(fx, monkeypatch):
    """The arguments of the real Morrison early-reset call (hard limit)."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    seen = []
    real = M.mg2_size_dist_liq

    def spy(*a, **k):
        seen.append(a)
        return real(*a, **k)

    monkeypatch.setattr(M, "mg2_size_dist_liq", spy)
    out = _run(fx, 0.0)
    monkeypatch.setattr(M, "mg2_size_dist_liq", real)
    assert len(seen) == 1
    return out, seen[0]


def _bounds_bind(args):
    """Cells where MG2's reset moves the number, split by which bound: the
    hard port's mean diameter (pgam+1)/lambda sits on 2 um / 50 um."""
    q, n, rho = args
    active, pgam, lam, nc_kg = W.mg2_droplet_psd(q, n, rho)
    d_mean = np.asarray((pgam + 1.0) / lam)
    moved = np.asarray(active) & ~np.isclose(np.asarray(nc_kg * rho), np.asarray(n),
                                              rtol=1e-12, atol=0.0)
    return (moved & np.isclose(d_mean, 2.0e-6, rtol=1e-12, atol=0.0),
            moved & np.isclose(d_mean, 50.0e-6, rtol=1e-12, atol=0.0))


def test_defaults_and_hard_limit():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    m = MorrisonConfig()
    assert m.mg2_drop_early_reset_log_width == WIDTH == m.mg2_drop_readjust_log_width
    assert ExperimentConfig._field_defaults["morrison_mg2_drop_early_reset_log_width"] == WIDTH
    assert HARD_LIMIT["mg2_drop_early_reset_log_width"] == 0.0


@pytest.mark.parametrize("name", ["mg2_prognostic_nc_act", "mg2_prognostic_nc_sed",
                                  "mg2_prognostic_nc_frz"])
def test_hard_limit_matches_mg2_where_the_bounds_bind(name, monkeypatch):
    """Whole-call droplet tendency vs the RUNNING MG2 at the hard limit, on
    fixtures whose early reset has cells a bound moves (2 um on all three,
    50 um on act/sed); the default width does not match (the comparison can
    fail)."""
    fx = _load(name)
    out, args = _reset_inputs(fx, monkeypatch)
    lo, hi = _bounds_bind(args)
    assert lo.sum() >= 5
    if name != "mg2_prognostic_nc_frz":
        assert hi.sum() >= 5
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
    print(name, "early-reset gap per column", [float(x) for x in gap])
    np.testing.assert_allclose(gap, GAP[name], rtol=1e-6, atol=1e-12)
    # a gap below the 1e-12 floor must not vanish (or appear) unnoticed
    assert np.array_equal(gap > 0.0, np.asarray(GAP[name]) > 0.0)


def _near_2um_bound(u):
    """Reset inputs whose mean diameter sits ``3*u*WIDTH`` in ln N from the
    2 um bound (u > 0: too many droplets, the side MG2 clips)."""
    q = 1.0e-11 * jnp.ones_like(u)
    one = jnp.ones_like(u)
    n_b = W.mg2_size_bounded_nc(q, 5.0e3 * one, one)
    return q, n_b * jnp.exp(3.0 * WIDTH * u), one


def test_gradients_through_the_reset():
    u = jnp.asarray([-2.0, -0.5, 0.5, 2.0])
    w = dict(number_cap_log_width=WIDTH, slope_log_width=WIDTH)

    def f(x, **k):   # number, shape and slope: all three reach the sinks / riming
        n, pgam, lam = W.mg2_size_dist_liq(*_near_2um_bound(x), **k)
        return jnp.stack([jnp.log(n), pgam, jnp.log(lam)])

    check_grads(lambda x: f(x, **w), (u,), order=2, modes=["rev"], eps=1e-3, atol=1e-6,
                rtol=1e-3)
    g = np.asarray(jax.grad(lambda x: jnp.sum(f(x, **w)[0]))(u))
    hard = np.asarray(jax.grad(lambda x: jnp.sum(f(x)[0]))(u))
    assert np.all(np.isfinite(g)) and np.all(g != 0.0)
    # MG2's clip has no derivative of the number on its clipped side; the rounded one does
    assert np.all(np.abs(hard[u > 0]) < 1e-2 * np.abs(g[u > 0]))


def test_whole_call_gradient_finite_at_the_default(monkeypatch):
    fx = _load("mg2_prognostic_nc_frz")
    _, args = _reset_inputs(fx, monkeypatch)
    binds = jnp.asarray(_bounds_bind(args)[0])
    rho = jnp.asarray(fx["col_rho"])
    cfg = _cfg(do_snow_to_graupel=False)._replace(mg2_drop_early_reset_log_width=WIDTH)

    def f(nc):
        out = A._run(fx, cfg=cfg, N_c=nc, hyd_kw=_hyd(fx))
        return jnp.sum(jnp.where(binds, out.dN_c_dt, 0.0))

    g = np.asarray(jax.grad(f)(jnp.asarray(fx["col_nc_kg"]) * rho))
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0


def test_flat_field_validation_cli_and_deck():
    from scripts.run.run_amip import _postprocess_args, build_arg_parser, build_config_from_args

    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from legoesm.driver.run_config_yaml import load_yaml_config
    f = "morrison_mg2_drop_early_reset_log_width"
    kw = dict(microphysics="morrison", morrison_prognostic_droplet_number=True,
              cam6_aerosol_dir="/x")
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=f"{f} must be a finite float in"):
            ExperimentConfig(**kw, **{f: bad}).validate_strict()
    with pytest.raises(ValueError, match=f"{f}=0.0 is read only .* silently inert"):
        ExperimentConfig(microphysics="morrison", **{f: 0.0}).validate_strict()
    base = MorrisonConfig()
    assert thread_morrison_scalars(ExperimentConfig(**{f: 0.0}), "morrison", base) == \
        base._replace(mg2_drop_early_reset_log_width=0.0)
    assert thread_morrison_scalars(ExperimentConfig(), "morrison", base) is base
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--morrison-mg2-drop-early-reset-log-width", "0.0"]), parser))
    assert cfg.morrison_mg2_drop_early_reset_log_width == 0.0
    amip = ROOT / "config/amip"
    arm = load_yaml_config(amip / "amip_production_prognostic_nc.yaml", build_arg_parser())
    assert arm[f] == WIDTH        # named explicitly
    assert arm["morrison_prognostic_droplet_number"] is True
    prod = load_yaml_config(amip / "amip_production.yaml", build_arg_parser())
    assert f not in prod
    assert prod["morrison_prognostic_droplet_number"] is False
