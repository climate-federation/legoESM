"""Kessler at its default smoothing creates no water in subsaturated air.

The saturation adjustment may only evaporate cloud that exists: after the step
no cloud is negative (so no downstream positivity floor can create water), and
a rain-free column keeps its total water exactly.  Tolerance (GLM/Claude vote
2026-11-04): column water change within 1e-12 of the column water, fp64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

FIXTURE = (Path(__file__).resolve().parents[4]
           / "tests/atmosphere/oracle_fixtures/kessler.npz")


def _columns(rh_scale):
    fx = np.load(FIXTURE)
    T, qv, pf, rho, dz = (jnp.asarray(fx[k]) for k in
                          ("col_T", "col_q_v", "col_p_full", "col_rho", "col_dz"))
    z = jnp.zeros_like(T)
    return T, rh_scale * qv, z, z, pf, rho, dz


@pytest.mark.parametrize("dt", [1200.0, 300.0])
def test_dry_columns_create_no_water(dt):
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_step_states
    T, qv, qc, qr, pf, rho, dz = _columns(0.3)
    T1, qv1, qc1, qr1, pr, _ = kessler_step_states(
        T, qv, qc, qr, pf, rho, dz, dt, KesslerConfig())
    assert float(jnp.min(qc1)) >= 0.0, float(jnp.min(qc1))
    assert float(jnp.min(qr1)) >= 0.0
    w0 = jnp.sum((qv + qc + qr) * rho * dz, axis=-1)
    w1 = jnp.sum((qv1 + qc1 + qr1) * rho * dz, axis=-1) + pr * dt
    rel = np.asarray(jnp.abs(w1 - w0) / w0)
    assert rel.max() <= 1.0e-12, rel.max()


@pytest.mark.parametrize("qv0", [0.0, 1.0e-10])
def test_condensation_never_exceeds_the_water_present(qv0):
    """Hot, (nearly) vapour-free columns: the smoothed switch may not condense
    more than exists (the vapour floor would create the difference)."""
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_step_states
    T, qv, qc, qr, pf, rho, dz = _columns(1.0)
    T = jnp.full_like(T, 320.0)
    qv = jnp.full_like(qv, qv0)
    out = kessler_step_states(T, qv, qc, qr, pf, rho, dz, 300.0, KesslerConfig())
    w0 = jnp.sum((qv + qc + qr) * rho * dz, axis=-1)
    w1 = jnp.sum((out[1] + out[2] + out[3]) * rho * dz, axis=-1) + out[4] * 300.0
    if qv0 == 0.0:
        # roundoff of the smoothed clip only (measured 2.5e-16 kg m-2); the
        # switch without the upper bound condensed ~3e-6 kg m-2 at 320 K
        assert float(jnp.max(jnp.abs(w1))) <= 1.0e-12, float(jnp.max(jnp.abs(w1)))
    else:
        rel = np.asarray(jnp.abs(w1 - w0) / w0)
        assert rel.max() <= 1.0e-12, rel.max()


def test_gradient_across_the_saturation_switch():
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_step_states
    T, qv, qc, qr, pf, rho, dz = _columns(1.0)
    cfg = KesslerConfig()

    def cloud_after(scale):
        out = kessler_step_states(T, scale * qv, qc, qr, pf, rho, dz, 300.0, cfg)
        return jnp.sum(out[2])

    for scale in (0.95, 1.0, 1.05):
        g = jax.grad(cloud_after)(jnp.asarray(scale))
        assert np.isfinite(float(g)) and float(g) != 0.0, (scale, float(g))
        check_grads(cloud_after, (jnp.asarray(scale),), order=1,
                    modes=("rev",), eps=1e-6)
