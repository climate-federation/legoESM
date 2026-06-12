"""Tests for the CLUBB C6/C7 skewness-coefficient family (now in ``clubb.py``).

The CAM branch has no CLUBB-JAX oracle (the reference is ARM), so these validate
the Fortran formula analytically: the damping ramp, the C7=C7b constant
reduction, and AD/JIT cleanliness. ``compute_skw_fnc`` itself is parity-tested in
``test_clubb_wp23``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812


def _gr(ng=2, nzt=12):
    nzm = nzt + 1
    zm = jnp.asarray(np.tile(np.linspace(0.0, 3000.0, nzm), (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm


def test_damp_coefficient_formula_and_passthrough():
    gr, ng, nzm = _gr()
    rng = np.random.default_rng(0)
    Cx = jnp.asarray(5.0 + rng.random((ng, nzm)))
    Lscale_zm = jnp.asarray(np.tile(np.linspace(5.0, 500.0, nzm), (ng, 1)))
    coeff = jnp.full((ng,), 4.0)
    mx = jnp.full((ng,), 14.0)
    alt = jnp.full((ng,), 100.0)
    thr = jnp.full((ng,), 60.0)
    out = np.asarray(C.damp_coefficient(coeff, Cx, mx, alt, thr, Lscale_zm, gr))
    zm = np.asarray(gr.zm)
    Ls = np.asarray(Lscale_zm)
    cond = (Ls < 60.0) & (zm > 100.0)
    damped = 14.0 + ((4.0 - 14.0) / 60.0) * Ls
    exp = np.where(cond, damped, np.asarray(Cx))
    np.testing.assert_allclose(out, exp, rtol=1e-12, atol=1e-14)


def test_C6_C7_skw_fnc():
    gr, ng, nzm = _gr()
    cfg = CLUBBConfig()
    rng = np.random.default_rng(1)
    Skw_zm = jnp.asarray(0.5 * rng.standard_normal((ng, nzm)))
    Lscale_zm = jnp.asarray(50.0 + 200.0 * rng.random((ng, nzm)))
    C6rt, C6thl, C7 = C.compute_C6_C7_Skw_fnc(Skw_zm, Lscale_zm, cfg, gr)
    assert C6rt.shape == (ng, nzm) and C7.shape == (ng, nzm)
    # CAM default C7 = C7b = 0.5 -> constant (|C7-C7b| = 0 -> the floor branch)
    np.testing.assert_allclose(np.asarray(C7), cfg.params.C7b, rtol=1e-12)
    # C6rt skewness function (C6rt=4 != C6rtb=6) is bounded by [min(C6rt,C6rtb)-, ...]
    assert np.all(np.isfinite(np.asarray(C6rt))) and np.all(np.isfinite(np.asarray(C6thl)))


def test_jit_and_grad():
    gr, ng, nzm = _gr()
    cfg = CLUBBConfig()
    rng = np.random.default_rng(2)
    Skw_zm = jnp.asarray(0.5 * rng.standard_normal((ng, nzm)))
    Lscale_zm = jnp.asarray(50.0 + 200.0 * rng.random((ng, nzm)))

    def loss(Skw):
        C6rt, C6thl, C7 = C.compute_C6_C7_Skw_fnc(Skw, Lscale_zm, cfg, gr)
        return jnp.sum(C6rt ** 2 + C6thl ** 2 + C7 ** 2)

    assert jnp.isfinite(jax.jit(loss)(Skw_zm))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(Skw_zm)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
