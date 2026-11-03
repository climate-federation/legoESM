"""Seeding of WRF Tiedtke's previous-step turbulence carries (qvpblten, qfx)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.driver.compiled_segments import tiedtke_lag_seeds
from legoesm.training.era5_to_state import prognostic_carry_seeds


def test_other_schemes_leave_carry_unchanged():
    q, ps = jnp.ones((2, 3, 4)), jnp.ones((2, 3))
    assert tiedtke_lag_seeds("zhang_mcfarlane", q, ps) == {}
    assert prognostic_carry_seeds("kessler", "louis", (2, 3, 4)) == {}


def test_cold_start_zeros_shaped_like_q_v_and_p_s():
    q, ps = jnp.ones((2, 3, 4)), jnp.ones((2, 3))
    s = tiedtke_lag_seeds("tiedtke", q, ps)
    assert s["turb_dq_v_prev"].shape == q.shape
    assert s["evap_sfc_prev"].shape == ps.shape
    assert not np.any(np.asarray(s["turb_dq_v_prev"]))
    s2 = prognostic_carry_seeds("kessler", "louis", (2, 3, 4), "tiedtke")
    assert s2["turb_dq_v_prev"].shape == (2, 3, 4)
    assert s2["evap_sfc_prev"].shape == (2, 3)


def test_restored_pair_is_used():
    """A carried / checkpoint-restored pair wins over the zero seed (WRF
    restarts RQVBLTEN and QFX); a missing pair (None, None) cold-starts."""
    q, ps = jnp.ones((2, 3, 4)), jnp.ones((2, 3))
    prev = (jnp.full(q.shape, 2.0), jnp.full(ps.shape, 3.0))
    s = tiedtke_lag_seeds("tiedtke", q, ps, prev=prev)
    assert s["turb_dq_v_prev"] is prev[0] and s["evap_sfc_prev"] is prev[1]
    s0 = tiedtke_lag_seeds("tiedtke", q, ps, prev=(None, None))
    assert not np.any(np.asarray(s0["evap_sfc_prev"]))


def test_physics_factory_refuses_tiedtke_without_turbulence():
    """Vote V21a: the lags come from the turbulence scheme; with none they
    would stay at their zero seed, so make_physics refuses (direct
    PhysicsConfig / SCM callers bypass ExperimentConfig.validate_strict)."""
    import pytest
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    cfg = PhysicsConfig(convection=ConvectionConfig(scheme="tiedtke"),
                        turbulence=TurbulenceConfig(scheme="none"))
    with pytest.raises(ValueError, match="turbulence='none'"):
        make_physics(cfg)
