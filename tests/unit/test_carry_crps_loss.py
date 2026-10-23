"""Grid-space CRPS (deterministic M=1 limit = |error|) in ``carry_mse``.

The WeatherBench objective is ``combined_loss`` -> ``carry_mse``; decks set
``w_crps_*`` and the terms used to be ignored.  Each term is the area mean of
``|pred - target|`` x level weight, divided by the variable scale (not
scale^2), mirroring the spectral lane's deterministic CRPS.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.losses import LossConfig, carry_mse, level_weights


class _Carry(NamedTuple):
    T: object
    u: object
    v: object
    q_v: object
    p_s: object


NLAT, NLON, NLEV = 4, 8, 3
SIGMA = jnp.linspace(0.1, 0.9, NLEV)
_OFF = dict(w_T=0.0, w_u=0.0, w_v=0.0, w_q=0.0, w_ps=0.0)


def _carry(e3=0.0, e2=0.0):
    z3 = jnp.zeros((NLAT, NLON, NLEV))
    z2 = jnp.zeros((NLAT, NLON))
    return _Carry(T=z3 + e3, u=z3 + e3, v=z3 + e3, q_v=z3 + e3, p_s=z2 + e2)


@pytest.mark.parametrize("var,scale_attr", [
    ("T", "T_scale"), ("u", "wind_scale"), ("v", "wind_scale"),
    ("q", "q_scale"), ("ps", "ps_scale"),
])
def test_crps_term_is_level_weighted_mae_over_scale(var, scale_attr):
    err = jnp.array([-1.0, 2.0, -4.0])          # sign must not cancel
    cfg = LossConfig(**_OFF, **{f"w_crps_{var}": 0.7},
                     level_weighting="pressure")
    pred = _carry(e3=err, e2=-3.0)
    loss = float(carry_mse(pred, _carry(), SIGMA, config=cfg))
    lev_w = np.asarray(level_weights(SIGMA, config=cfg))
    mae = 3.0 if var == "ps" else float(np.mean(np.abs(np.asarray(err)) * lev_w))
    expected = 0.7 * mae / getattr(cfg, scale_attr)
    assert expected > 0
    np.testing.assert_allclose(loss, expected, rtol=1e-6)


def test_crps_uses_residual_scales_when_residual_normalize():
    cfg = LossConfig(**_OFF, w_crps_T=1.0, residual_normalize=True,
                     level_weighting="uniform")
    loss = float(carry_mse(_carry(e3=2.0), _carry(), SIGMA, config=cfg))
    lev_w = np.asarray(level_weights(SIGMA, config=cfg))
    np.testing.assert_allclose(loss, 2.0 * np.mean(lev_w) / cfg.T_resid_scale,
                               rtol=1e-6)


@pytest.mark.parametrize("setting", [
    {"persistence_normalize": True},
    {"ensemble_size": 4},
    {"w_afcrps": 0.5},
])
def test_wb_loss_config_refuses_unimplemented_settings(setting):
    from legoesm.training.scale_build import make_loss_config

    with pytest.raises(ValueError, match="does not implement"):
        make_loss_config(None, {"loss": setting})
    make_loss_config(None, {"loss": {"w_crps_T": 0.5}})   # supported: no raise


def test_crps_zero_residual_scale_uses_the_mse_floor():
    """A zero residual scale is floored exactly like the MSE term
    (scale^2 >= 1e-30), so the loss stays finite."""
    cfg = LossConfig(**_OFF, w_crps_T=1.0, residual_normalize=True,
                     T_resid_scale=0.0, level_weighting="uniform")
    loss = float(carry_mse(_carry(e3=2.0), _carry(), SIGMA, config=cfg))
    lev_w = np.asarray(level_weights(SIGMA, config=cfg))
    assert np.isfinite(loss)
    np.testing.assert_allclose(loss, 2.0 * np.mean(lev_w) / 1e-15, rtol=1e-6)
