"""RULE 5 for the EKE local source step (``eke_apply_local_source``) and the 3-D floor.

Switches: the positive part of E, the l_min floor of the dissipation length, the
one-sided clamp of a supplied production, the positive/negative split of a
signed source (and of a signed override), and the 3-D storage floor
``max(E, e_min)`` (``eke_floor``).

* Hard limit (every width 0): Veros d657322b ``integrate_eke`` (alpha_eke = 0,
  horizontally uniform E, so no vertical or lateral diffusion) to 1e-12 on cells
  with forcing >= 0; elsewhere the documented implicit negative-part fold, written
  out below, to 1e-12.
* Defaults: the per-cell gap from Veros is pinned (1e-6 rel, 1e-12 abs).
* Gradients through each transition are finite and non-zero at the defaults, and
  each smooth switch departs from the hard one only inside its band.

Fixture ``oracle_fixtures/veros_d657322b_eke_source.npz``; probe and votes in
``oracle_audit/batch_frazil_edmf_kf``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.ocean.physics.lateral_mixing import eke as E

jax.config.update("jax_enable_x64", True)

FIX = np.load(Path(__file__).parent / "oracle_fixtures" / "veros_d657322b_eke_source.npz")
C_K, C_EPS, L_MIN, CROSS, CRHIN, K_MAX, DT = FIX["settings"]
BASE = dict(c_k=C_K, c_eps=C_EPS, l_min=L_MIN, eke_cross=CROSS, eke_crhin=CRHIN, kappa_gm_max=K_MAX)
HARD = E.EKEConfig(**BASE, e_pos_width=0.0, length_switch_width=0.0, kappa_cap_width=0.0,
                   source_split_width=0.0)
SMOOTH = E.EKEConfig(**BASE)
W_E, W_L, W_S = SMOOTH.e_pos_width, SMOOTH.length_switch_width, SMOOTH.source_split_width


def _step(cfg, eke=None, forc=None):
    eke = jnp.asarray(FIX["eke"] if eke is None else eke)
    forc = jnp.asarray(FIX["forc"] if forc is None else forc)
    return np.asarray(E.eke_apply_local_source(
        eke, jnp.zeros_like(eke), jnp.asarray(FIX["eke_len"]), cfg, DT, signed_source=forc))


def _fold(eke, L, dt, prod, signed):
    """The hard form: Veros's implicit dissipation, legoESM's implicit negative part."""
    e = np.maximum(eke, 0.0)
    pos, neg = np.maximum(signed, 0.0), np.maximum(-signed, 0.0)
    rate = C_EPS * np.sqrt(e) / np.maximum(L, L_MIN) + neg / np.maximum(e, 1e-30)
    return (e + dt * (prod + pos)) / (1.0 + dt * rate)


def test_hard_limit_matches_veros_where_forcing_nonnegative():
    got, ref = _step(HARD), FIX["eke_new"]
    m = FIX["forc"] >= 0.0
    assert m.sum() > 20
    assert (np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30))[m].max() <= 1e-12


def test_hard_limit_is_the_implicit_fold_elsewhere():
    rng = np.random.default_rng(3)
    e = rng.choice([-1e-4, 0.0, 1e-9, 1e-3, 3.0], 400) * rng.uniform(0.5, 1.5, 400)
    L = rng.uniform(10.0, 2e4, 400)
    x = rng.choice([-1, 0, 1], 400) * 10.0 ** rng.uniform(-14, -6, 400)
    p = rng.choice([-1, 1], 400) * 10.0 ** rng.uniform(-14, -6, 400)
    ref = _fold(e, L, DT, np.maximum(p, 0.0), x)
    got = np.asarray(E.eke_apply_local_source(jnp.asarray(e), jnp.zeros(400), jnp.asarray(L), HARD, DT,
                                              production_override=jnp.asarray(p), signed_source=jnp.asarray(x)))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)
    ref = _fold(e, L, DT, 0.0, p + x)
    got = np.asarray(E.eke_apply_local_source(jnp.asarray(e), jnp.zeros(400), jnp.asarray(L), HARD, DT,
                                              production_override=jnp.asarray(p), clamp_production=False,
                                              signed_source=jnp.asarray(x)))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_smooth_default_gap_pinned():
    np.testing.assert_allclose(_step(SMOOTH) - FIX["eke_new"], FIX["pin_eke_new"], rtol=1e-6, atol=1e-12)


def test_floor_hard_limit_and_band():
    e = jnp.asarray([0.0, SMOOTH.e_min, SMOOTH.e_min + W_E / 4, 1.0])
    np.testing.assert_array_equal(np.asarray(E.eke_floor(e, HARD)), np.maximum(np.asarray(e), SMOOTH.e_min))
    d = np.asarray(E.eke_floor(e, SMOOTH)) - np.asarray(E.eke_floor(e, HARD))
    np.testing.assert_array_equal(d[[0, 3]], 0.0)
    assert 0.0 < d[1] <= W_E / 8 and 0.0 < d[2] <= W_E / 8


_E0, _L0 = 1.0e-3, 1.0e3


def _src(cfg, **kw):
    e = kw.pop("e", jnp.asarray(_E0))
    L = kw.pop("L", jnp.asarray(_L0))
    return E.eke_apply_local_source(e, jnp.asarray(0.0), L, cfg, DT, **kw)


_SWITCHES = {
    # name: (scalar function of x at the defaults, x at the transition, FD step)
    "positive_part_E": (lambda x: _src(SMOOTH, e=x), 0.0, W_E / 64),
    "dissipation_lmin": (lambda x: _src(SMOOTH, L=x), L_MIN, W_L / 64),
    "signed_split": (lambda x: _src(SMOOTH, signed_source=x), 0.0, W_S / 64),
    "signed_override_split": (lambda x: _src(SMOOTH, production_override=x, clamp_production=False), 0.0, W_S / 64),
    "production_clamp": (lambda x: _src(SMOOTH, production_override=x), W_S / 2, W_S / 64),
    "floor_e_min": (lambda x: E.eke_floor(x, SMOOTH), SMOOTH.e_min, W_E / 64),
}


@pytest.mark.parametrize("name", list(_SWITCHES))
def test_gradient_through_switch(name):
    f, x0, eps = _SWITCHES[name]
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    check_grads(f, (x0,), order=1, modes=["fwd", "rev"], eps=eps)
    g = jax.grad(f)(x0)
    assert np.isfinite(g) and g != 0.0


_BAND = {
    # name: (fn of cfg, bound on |smooth - hard|)
    "positive_part_E": (lambda c: _src(c, e=jnp.asarray(W_E / 4)), None),
    "dissipation_lmin": (lambda c: _src(c, L=jnp.asarray(L_MIN + W_L / 4)), None),
    "signed_split": (lambda c: _src(c, signed_source=jnp.asarray(W_S / 4)), None),
    "production_clamp": (lambda c: _src(c, production_override=jnp.asarray(W_S / 4)), DT * 4.0 * W_S / 27.0),
}


@pytest.mark.parametrize("name", list(_BAND))
def test_smooth_switch_active_inside_its_band(name):
    f, bound = _BAND[name]
    d = abs(float(f(SMOOTH)) - float(f(HARD)))
    assert d > 0.0
    if bound is not None:
        assert d <= bound


@pytest.mark.parametrize("name", ["positive_part_E", "dissipation_lmin", "signed_split", "production_clamp"])
def test_smooth_switch_exact_outside_its_band(name):
    far = {"positive_part_E": dict(e=jnp.asarray(10 * W_E)), "dissipation_lmin": dict(L=jnp.asarray(L_MIN + 10 * W_L)),
           "signed_split": dict(signed_source=jnp.asarray(10 * W_S)),
           "production_clamp": dict(production_override=jnp.asarray(10 * W_S))}[name]
    assert float(_src(SMOOTH, **far)) == float(_src(HARD, **far))


def test_zero_production_creates_no_energy():
    """The one-sided clamp is exactly 0 for P <= 0 (a smooth max would add w/8 every step)."""
    hard = float(_src(HARD, production_override=jnp.asarray(0.0)))
    for p in (0.0, -1e-9):
        assert float(_src(SMOOTH, production_override=jnp.asarray(p))) == hard


def test_split_sums_to_the_source_and_stays_nonnegative():
    e = jnp.asarray([0.0, 1e-12, 1e-6, 1.0])
    for x in (-W_S, -W_S / 4, 0.0, W_S / 4):
        out = np.asarray(E.eke_apply_local_source(e, jnp.zeros(4), jnp.full(4, _L0), SMOOTH, DT,
                                                  signed_source=jnp.full(4, x)))
        assert np.all(out >= 0.0) and np.all(np.isfinite(out))


def test_driver_uses_eke_floor():
    import inspect
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    src = inspect.getsource(LatLonCGridOceanModel._eke_3d_step)
    assert "eke_floor(E, eke_cfg)" in src and "jnp.maximum(E, eke_cfg.e_min)" not in src
