"""The scheme runs MG2's start-of-step rain freezing below -40 C
(micro_mg2_0.F90:1182-1212) on the INPUT temperature, before the processes,
then the post-sedimentation freezing (:2571-2610) on the updated one.  The
helper itself is checked against MG2 in test_mg2_thin_air_oracle; this test
fails when the scheme stops calling it at the start of the step."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402


def test_scheme_freezes_rain_at_the_start_of_the_step(monkeypatch):
    from legoesm.atmosphere.physics.microphysics import morrison as M
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (
        HARD_LIMIT, _production_morrison)
    T = jnp.array([[constants.T_freeze - 45.0, constants.T_freeze - 44.0]])
    p = jnp.array([[30000.0, 32000.0]])
    ph = jnp.array([[29000.0, 31000.0, 33000.0]])
    rho = p / (constants.R_d * T)
    dz = (ph[:, 1:] - ph[:, :-1]) / (constants.g * rho)
    q_r = jnp.array([[1.0e-4, 1.0e-4]])
    z = jnp.zeros_like(T)
    hyd = HydrometeorState(q_c=z, q_r=q_r, q_i=z, q_s=z, q_g=z, N_c=z,
                           N_r=1.0e4 * rho, N_i=z, N_s=z)
    seen = []
    real = M.mg2_instant_rain_freeze

    def spy(T_, *a, **k):
        seen.append(np.asarray(T_))
        return real(T_, *a, **k)
    monkeypatch.setattr(M, "mg2_instant_rain_freeze", spy)
    cfg = _production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6")
    M.morrison_microphysics(T, 1.0e-6 + z, hyd, p, ph, rho, dz, 600.0, cfg,
                            cloud_fraction=0.5 + z)
    assert len(seen) >= 1
    np.testing.assert_array_equal(seen[0], np.asarray(T))
    # implementation pin (eager run): start of step, post-sedimentation mass, number
    assert len(seen) == 3, len(seen)


def test_rain_falling_below_minus_40_freezes_after_sedimentation():
    """Inversion column: rain at -38 C falls into a rain-free level at -45 C;
    MG2's post-sedimentation freezing (:2571-2610) freezes what arrives
    (small drops, so to cloud ice; heat-limit far from binding), so no rain
    is left there.  Measured: 1e-25 kg/kg left, 1e-9 without the freezing."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (
        HARD_LIMIT, _production_morrison)
    T = jnp.array([[constants.T_freeze - 38.0, constants.T_freeze - 45.0]])
    p = jnp.array([[30000.0, 32000.0]])
    ph = jnp.array([[29000.0, 31000.0, 33000.0]])
    rho = p / (constants.R_d * T)
    dz = (ph[:, 1:] - ph[:, :-1]) / (constants.g * rho)
    z = jnp.zeros_like(T)
    q_r = jnp.array([[1.0e-4, 0.0]])
    hyd = HydrometeorState(q_c=z, q_r=q_r, q_i=z, q_s=z, q_g=z, N_c=z,
                           N_r=jnp.array([[1.0e4, 0.0]]) * rho, N_i=z, N_s=z)
    cfg = _production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6")
    dt = 600.0
    out = M.morrison_microphysics(T, 1.0e-6 + z, hyd, p, ph, rho, dz, dt, cfg,
                                  cloud_fraction=0.5 + z)
    r_bot = float(q_r[0, 1] + out.dq_r_dt[0, 1] * dt)
    print("bottom rain", r_bot)
    assert abs(r_bot) <= 1.0e-12 * float(q_r[0, 0])
