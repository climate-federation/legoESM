"""RULE 5 tests of the smoothed MG2 frozen-rain destination (user decision
2026-10-22, 15a): cloud ice where the mean drop 1/lamr < Dcs, snow otherwise,
ramped over ``mg2_frozen_rain_ice_ramp_frac`` x Dcs below Dcs.

Fixture ``mg2_prognostic_nc_frz.npz`` (``make_fixture.py --prognostic-nc-frz``):
the RUNNING MG2 on supercooled rain with mean drops of 300-700 um; MG2's
frozen rain to snow (mnuccr) and to cloud ice (mnuccri) after its rain mass
limiter (tap_ilim2), which scales both by the same ratio, so the routed share
per cell is MG2's own; MG2's rain slope lamr before the split (tap_frz).  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics import morrison as M  # noqa: E402

from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FX = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_frz.npz"
DCS = 500.0e-6
FRAC = 0.1
# Per fixture column: max over freezing cells of |our ice share - MG2's|, at
# the default ramp 0.1 (measured 2026-10-22; re-measure on any change).
GAP = np.array([0.9554092099831757, 0.7493973243222857, 0.9998947057616662,
                0.9986772983229906, 0.9989738264511224, 0.9999185208914257])


def _fx():
    return dict(np.load(FX))


def _shares(fx, frac):
    lamr = jnp.asarray(fx["ora_frz_lamr"])     # MG2's own rain slope (tap_frz)
    m, mi = fx["ora_ilim2_mnuccr"], fx["ora_ilim2_mnuccri"]
    on = (m + mi) > 0
    return lamr, np.asarray(W.mg2_frozen_rain_ice_fraction(lamr, DCS, frac)), (mi > 0) * 1.0, on


def test_fixture_covers_both_destinations_and_the_ramp():
    fx = _fx()
    lamr, soft, cam, on = _shares(fx, FRAC)
    assert on.sum() >= 100 and (cam[on] == 1).sum() >= 30 and (cam[on] == 0).sum() >= 30
    assert ((soft > 0) & (soft < 1) & on).sum() >= 30


def test_hard_limit_matches_mg2():
    fx = _fx()
    _, hard, cam, on = _shares(fx, 0.0)
    assert np.array_equal(hard[on], cam[on])


def test_default_gap_per_column_is_the_recorded_one():
    fx = _fx()
    _, soft, cam, on = _shares(fx, _cfg()._field_defaults["mg2_frozen_rain_ice_ramp_frac"])
    gap = np.max(np.abs(soft - cam) * on, axis=1)
    print("frozen-rain split gap per column", [float(x) for x in gap])
    np.testing.assert_allclose(gap, GAP, rtol=1e-6, atol=1e-12)


def test_gradient_through_the_ramp():
    lam = jnp.asarray([1.0 / 520e-6, 1.0 / 495e-6, 1.0 / 470e-6, 1.0 / 452e-6, 1.0 / 400e-6])

    def f(x):
        return W.mg2_frozen_rain_ice_fraction(x * 1.0e3, DCS, FRAC)

    check_grads(f, (lam / 1.0e3,), order=2, modes=["rev"], eps=1e-4, atol=1e-6, rtol=1e-4)
    g = jax.grad(lambda x: jnp.sum(f(x)))(lam / 1.0e3)
    assert np.all(np.asarray(g[1:4]) != 0.0) and float(g[0]) == 0.0 and float(g[4]) == 0.0


def test_scheme_routes_the_share(monkeypatch):
    """With do_graupel=False (MG2 routing) the scheme sends share x frozen rain
    to cloud ice and the rest to snow: against the hard split, the smooth one
    moves frozen rain out of cloud ice in every freezing cell where the share
    drops (post-source snow/ice sedimentation moves it on, so the cell sum is
    not conserved)."""
    fx = _fx()
    seen = []
    real = M.mg2_frozen_rain_ice_fraction

    def spy(*a):
        seen.append(np.asarray(real(*a)))
        return real(*a)

    monkeypatch.setattr(M, "mg2_frozen_rain_ice_fraction", spy)
    hyd = dict(q_i=jnp.asarray(fx["col_q_i"]), N_i=jnp.asarray(fx["col_ni_kg"]),
               q_s=jnp.asarray(fx["col_q_s"]), N_s=jnp.asarray(fx["col_ns_kg"]))
    outs = [A._run(fx, cfg=_cfg(do_graupel=False)._replace(mg2_frozen_rain_ice_ramp_frac=fr),
                   hyd_kw=hyd) for fr in (0.0, FRAC)]
    df = seen[1] - seen[0]
    di = np.asarray(outs[1].dq_i_dt) - np.asarray(outs[0].dq_i_dt)
    on = (fx["ora_ilim2_mnuccr"] + fx["ora_ilim2_mnuccri"]) > 0
    moved = (np.abs(df) > 1e-6) & on
    assert moved.sum() >= 30
    assert np.all(np.sign(di[moved]) == np.sign(df[moved]))


def test_port_slope_classifies_like_mg2():
    """The scheme's own rain slope (not MG2's) sends the same cells to ice."""
    fx = _fx()
    cfg = M.resolve_morrison_flavor(_cfg())
    rho, cf = jnp.asarray(fx["col_rho"]), jnp.asarray(fx["col_cf"])
    _, _, lamr, _ = W.mg2_inprecip_rain(jnp.asarray(fx["col_q_r"]), jnp.asarray(fx["col_N_r"]),
                                        rho, cf, cfg.lamr_min, cfg.lamr_max)
    _, _, cam, on = _shares(fx, 0.0)
    hard = np.asarray(W.mg2_frozen_rain_ice_fraction(lamr, DCS, 0.0))
    assert np.array_equal(hard[on], cam[on])


def test_scheme_hard_routing_is_mg2s(monkeypatch):
    fx = _fx()
    seen = []
    real = M.mg2_frozen_rain_ice_fraction

    def spy(*a):
        seen.append(np.asarray(real(*a)))
        return real(*a)

    monkeypatch.setattr(M, "mg2_frozen_rain_ice_fraction", spy)
    hyd = dict(q_i=jnp.asarray(fx["col_q_i"]), N_i=jnp.asarray(fx["col_ni_kg"]),
               q_s=jnp.asarray(fx["col_q_s"]), N_s=jnp.asarray(fx["col_ns_kg"]))
    A._run(fx, cfg=_cfg(do_graupel=False), hyd_kw=hyd)
    _, _, cam, on = _shares(fx, 0.0)
    assert np.array_equal(seen[0][on], cam[on])
