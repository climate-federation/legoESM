"""CAM6 host treatment of the Zhang-McFarlane detrained condensate
(``clubb_intr.F90:2345-2376``, non-``zmconv_microp``, ``dlf2 = 0``) vs a
literal transcription of the Fortran (``tests/atmosphere/_cam6_detrain_oracle``).

Tolerance 1e-12 relative per element, no absolute slack (exact zeros above
the warm end / below the cold end must be exact).  Test temperatures sit on a
1 K grid that avoids the ramp ends (268.15 / 238.15 K), where legoESM's
``T_freeze - 5`` differs from CAM's literal in the last bit.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (  # noqa: E402
    cam6_detrained_condensate_partition,
)
from tests.atmosphere._cam6_detrain_oracle import (  # noqa: E402
    CPAIR,
    LATICE,
    cam6_clubb_detrain_oracle,
)

RTOL = 1e-12


def _column():
    """Three columns spanning 230-280 K; dlf with a zero and a (non-physical)
    negative entry so the oracle's max(0, .) in the number sources is hit."""
    t = np.linspace(230.0, 280.0, 51)
    t = np.stack([t, t[::-1], t])
    dlf = np.stack([np.full(51, 1.0e-7), np.geomspace(1e-9, 3e-6, 51),
                    np.where(np.arange(51) % 7 == 0, 0.0, 5.0e-8)])
    dlf[2, 3] = -2.0e-9
    return dlf, t


def test_oracle_constants_are_legoesm_constants():
    assert constants.L_f == LATICE
    assert constants.c_pd == CPAIR
    cfg = ZhangMcFarlaneConfig()
    assert abs(cfg.detrain_ice_t_warm_K - 268.15) < 1e-12
    assert abs(cfg.detrain_ice_t_cold_K - 238.15) < 1e-12


def test_partition_matches_cam6_fortran():
    dlf, t = _column()
    ours = cam6_detrained_condensate_partition(jnp.asarray(dlf), jnp.asarray(t))
    ref = cam6_clubb_detrain_oracle(dlf, t)
    for name, a, b in zip(ours._fields, ours, ref):
        assert np.any(b != 0.0), name   # non-vacuous
        np.testing.assert_allclose(np.asarray(a), b, rtol=RTOL, atol=0.0, err_msg=name)
    # the ramp is exercised on all three branches
    assert np.any(t > 268.15) and np.any(t < 238.15) and np.any((t > 238.15) & (t < 268.15))


def test_partition_conserves_water_and_frozen_enthalpy():
    dlf, t = _column()
    p = cam6_detrained_condensate_partition(jnp.asarray(dlf), jnp.asarray(t))
    np.testing.assert_allclose(np.asarray(p.dq_c + p.dq_i), dlf, rtol=1e-15, atol=1e-30)
    dp = np.linspace(500.0, 4000.0, 51)
    col = lambda x: np.sum(np.asarray(x) * dp, axis=-1) / constants.g  # noqa: E731
    np.testing.assert_allclose(col(p.dq_c) + col(p.dq_i), col(dlf), rtol=1e-14, atol=1e-30)
    # c_pd dT - L_f dq_i = 0: the fusion heating is the ice part's latent heat
    np.testing.assert_allclose(constants.c_pd * np.asarray(p.dT_dt),
                               constants.L_f * np.asarray(p.dq_i), rtol=1e-14, atol=1e-30)


def test_partition_gradient_through_the_ramp():
    """Finite, non-zero derivative through the ramp (T inside 238-268 K)."""
    # dlf = a * 1e-7 with a = O(1): finite-difference steps on a stay clear of
    # the oracle's max(0, dlf) kink at dlf = 0 (raw dlf ~ 1e-7 < the FD step).
    a = jnp.asarray([1.0, 2.0, 0.4])
    t = jnp.asarray([240.0, 252.5, 266.0])

    def f(t_, a_):
        p = cam6_detrained_condensate_partition(a_ * 1.0e-7, t_)
        return jnp.stack([p.dq_c * 1e7, p.dq_i * 1e7, p.dN_c * 1e-5,
                          p.dN_i * 1e-3, p.dT_dt * 1e4])

    check_grads(f, (t, a), order=2, modes=("fwd", "rev"))
    g = jax.jacfwd(lambda t_: f(t_, a)[1])(t)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.all(np.asarray(jnp.diagonal(g)) < 0.0)   # warmer -> less ice
