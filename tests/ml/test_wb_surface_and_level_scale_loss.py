"""Near-surface WB2 loss terms and the per-level 6 h residual scaling."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.losses import (
    LossConfig,
    carry_mse_terms,
    surface_field_terms,
)

N_LAT, N_LON, NLEV = 6, 8, 4
LAT_W = jnp.linspace(0.5, 1.0, N_LAT)


class _C:
    def __init__(self, **kw):
        z3 = jnp.zeros((N_LAT, N_LON, NLEV))
        self.T = kw.get("T", z3)
        self.u = kw.get("u", z3)
        self.v = kw.get("v", z3)
        self.q_v = kw.get("q_v", z3)
        self.p_s = kw.get("p_s", jnp.zeros((N_LAT, N_LON)))


def _sf():
    return jnp.linspace(0.2, 0.95, NLEV)


def test_per_level_scale_divides_each_level_by_its_own_scale():
    """An error of 1 K on the LOWEST level only, with that level's scale 2 K:
    the T terms are those of a 0.5 K error at unit scale."""
    base = dict(residual_normalize=True, w_bias_T=3.0, w_crps_T=0.7,
                level_weighting="uniform",
                u_resid_scale_lev=(1.0,) * NLEV, v_resid_scale_lev=(1.0,) * NLEV,
                q_resid_scale_lev=(1.0,) * NLEV)
    dT = jnp.zeros((N_LAT, N_LON, NLEV)).at[..., -1].set(1.0)
    half = dT * 0.5
    a = LossConfig(**base, T_resid_scale_lev=(1.0, 1.0, 1.0, 2.0))
    b = LossConfig(**base, T_resid_scale_lev=(1.0,) * NLEV)
    _, ta = carry_mse_terms(_C(T=dT), _C(), _sf(), LAT_W, a)
    _, tb = carry_mse_terms(_C(T=half), _C(), _sf(), LAT_W, b)
    for k in ("mse_T", "bias_T", "crps_T"):
        np.testing.assert_allclose(float(ta[k]), float(tb[k]), rtol=1e-12, err_msg=k)
    # and it is NOT what the scalar 1 K scale gives
    _, tc = carry_mse_terms(_C(T=dT), _C(), _sf(), LAT_W, b)
    assert float(tc["mse_T"]) == pytest.approx(4.0 * float(ta["mse_T"]))


def test_uniform_per_level_scale_matches_scalar_scale():
    rng = np.random.default_rng(0)
    pred = _C(T=jnp.asarray(rng.normal(size=(N_LAT, N_LON, NLEV))),
              u=jnp.asarray(rng.normal(size=(N_LAT, N_LON, NLEV))),
              v=jnp.asarray(rng.normal(size=(N_LAT, N_LON, NLEV))),
              q_v=jnp.asarray(1e-4 * rng.normal(size=(N_LAT, N_LON, NLEV))),
              p_s=jnp.asarray(50.0 * rng.normal(size=(N_LAT, N_LON))))
    common = dict(residual_normalize=True, w_bias_T=2.0, w_bias_u=1.0,
                  w_crps_T=0.5, w_crps_q=0.3, w_crps_ps=0.2)
    scalar = LossConfig(**common, T_resid_scale=1.5, wind_resid_scale=3.0,
                        q_resid_scale=5e-4, ps_resid_scale=200.0)
    lev = LossConfig(**common, ps_resid_scale=200.0,
                     T_resid_scale_lev=(1.5,) * NLEV, u_resid_scale_lev=(3.0,) * NLEV,
                     v_resid_scale_lev=(3.0,) * NLEV, q_resid_scale_lev=(5e-4,) * NLEV)
    la, ta = carry_mse_terms(pred, _C(), _sf(), LAT_W, scalar)
    lb, tb = carry_mse_terms(pred, _C(), _sf(), LAT_W, lev)
    assert set(ta) == set(tb)
    np.testing.assert_allclose(float(la), float(lb), rtol=1e-10)


def test_per_level_scales_all_or_nothing_and_sized():
    with pytest.raises(ValueError, match="all of T, u, v, q"):
        carry_mse_terms(_C(), _C(), _sf(), LAT_W, LossConfig(
            residual_normalize=True, T_resid_scale_lev=(1.0,) * NLEV))
    bad = LossConfig(residual_normalize=True, T_resid_scale_lev=(1.0,) * 3,
                     u_resid_scale_lev=(1.0,) * NLEV, v_resid_scale_lev=(1.0,) * NLEV,
                     q_resid_scale_lev=(1.0,) * NLEV)
    with pytest.raises(ValueError, match="levels"):
        carry_mse_terms(_C(), _C(), _sf(), LAT_W, bad)


def _fields(c):
    return {k: jnp.full((N_LAT, N_LON), c) for k in ("t2m", "u10", "v10", "mslp")}


def test_surface_terms_analytic_and_masked():
    """Error c everywhere, scale s: mse = w (c/s)^2, bias = wb (c/s)^2,
    crps = wc |c|/s; masked points never count."""
    cfg = LossConfig(w_t2m=1.0, w_bias_t2m=10.0, w_crps_t2m=0.5,
                     t2m_resid_scale=2.0, w_mslp=0.1, mslp_resid_scale=100.0)
    pred, tgt = _fields(0.0), _fields(0.0)
    pred["t2m"] = jnp.full((N_LAT, N_LON), 3.0).at[0, 0].set(1e6)
    pred["mslp"] = jnp.full((N_LAT, N_LON), 50.0)
    valid = {k: jnp.ones((N_LAT, N_LON), bool) for k in pred}
    valid["t2m"] = valid["t2m"].at[0, 0].set(False)
    loss, t = surface_field_terms(pred, tgt, valid, LAT_W, cfg)
    assert set(t) == {"mse_t2m", "bias_t2m", "crps_t2m", "mse_mslp"}
    np.testing.assert_allclose(float(t["mse_t2m"]), 1.5 ** 2, rtol=1e-12)
    np.testing.assert_allclose(float(t["bias_t2m"]), 10.0 * 1.5 ** 2, rtol=1e-12)
    np.testing.assert_allclose(float(t["crps_t2m"]), 0.5 * 1.5, rtol=1e-12)
    np.testing.assert_allclose(float(t["mse_mslp"]), 0.1 * 0.25, rtol=1e-12)
    np.testing.assert_allclose(float(loss), sum(float(v) for v in t.values()), rtol=1e-12)


def test_surface_term_without_scale_is_an_error():
    with pytest.raises(ValueError, match="u10_resid_scale"):
        surface_field_terms(_fields(1.0), _fields(0.0),
                            {k: jnp.ones((N_LAT, N_LON), bool) for k in _fields(0)},
                            LAT_W, LossConfig(w_u10=0.1))
    loss, t = surface_field_terms(_fields(1.0), _fields(0.0), {}, LAT_W, LossConfig())
    assert t == {} and float(loss) == 0.0
