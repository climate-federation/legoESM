"""Direct test for ACE2-style residual normalization in ``carry_mse``.

When ``residual_normalize=True`` each variable's MSE is divided by the
residual scale² (std of the 6 h field change) instead of the full-field
scale². For a fixed temperature error the loss therefore scales up by
(T_scale / T_resid_scale)² — verified here on a synthetic carry.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.training.losses import carry_mse, LossConfig


class _Carry(NamedTuple):
    T: object
    u: object
    v: object
    q_v: object
    p_s: object
    held_sw_up_toa: object
    held_lw_up_toa: object
    held_sw_net_sfc: object
    held_lw_net_sfc: object


def _carry(nlat=4, nlon=8, nlev=3, dT=0.0):
    z3 = jnp.zeros((nlat, nlon, nlev))
    z2 = jnp.zeros((nlat, nlon))
    return _Carry(T=z3 + dT, u=z3, v=z3, q_v=z3, p_s=z2,
                  held_sw_up_toa=z2, held_lw_up_toa=z2,
                  held_sw_net_sfc=z2, held_lw_net_sfc=z2)


def test_residual_norm_amplifies_T_by_scale_ratio():
    nlev = 3
    sigma = jnp.linspace(0.1, 0.9, nlev)
    tgt = _carry(nlev=nlev, dT=0.0)
    pred = _carry(nlev=nlev, dT=2.0)            # uniform +2 K error
    only_T = dict(w_T=1.0, w_u=0.0, w_v=0.0, w_q=0.0, w_ps=0.0,
                  w_bias_T=0.0, level_weighting="uniform")
    cfg_full = LossConfig(normalize_by_scale=True, **only_T)        # T_scale=30
    cfg_resid = LossConfig(residual_normalize=True, **only_T)       # T_resid_scale=1.5
    Lf = float(carry_mse(pred, tgt, sigma, config=cfg_full))
    Lr = float(carry_mse(pred, tgt, sigma, config=cfg_resid))
    assert Lf > 0 and Lr > 0
    ratio = Lr / Lf
    expected = (cfg_full.T_scale / cfg_full.T_resid_scale) ** 2     # (30/1.5)^2 = 400
    assert np.isclose(ratio, expected, rtol=1e-4), (ratio, expected)


def test_residual_default_off_matches_full_field():
    """residual_normalize defaults False -> identical to the legacy path."""
    sigma = jnp.linspace(0.1, 0.9, 3)
    tgt = _carry(dT=0.0); pred = _carry(dT=1.5)
    cfg = LossConfig(w_T=1.0, w_u=0.0, w_v=0.0, w_q=0.0, w_ps=0.0,
                     level_weighting="uniform")
    assert cfg.residual_normalize is False
    L = float(carry_mse(pred, tgt, sigma, config=cfg))
    assert np.isclose(L, 1.5 ** 2 / cfg.T_scale ** 2, rtol=1e-4)


if __name__ == "__main__":
    test_residual_norm_amplifies_T_by_scale_ratio()
    test_residual_default_off_matches_full_field()
    print("residual_norm loss: self-checks passed")
