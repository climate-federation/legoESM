"""Tests for gradient checkpointing through the compiled dycore.

Validates:
- Gradients flow through multi-step rollouts with jax.checkpoint
- Checkpointed forward pass matches non-checkpointed
- Hierarchical checkpointing (per-step inside segment + per-segment in rollout)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import (
    SegmentCarry,
    build_segment_fn,
    pack_carry,
    pack_forcing,
)
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.grids.cubed_sphere import create_cubed_sphere

N = 4
NLEV = 3
_GRID = create_cubed_sphere(N)
_OPTIONAL_3D_OUTPUT_FIELDS = (
    "du_dt",
    "dv_dt",
    "dq_i_dt",
    "dq_s_dt",
    "dq_g_dt",
    "dN_c_dt",
    "dN_r_dt",
    "dN_i_dt",
)


def _zero_physics_output(T, p_s):
    kwargs = dict(
        dT_dt=jnp.zeros(T.shape),
        dq_v_dt=jnp.zeros(T.shape),
        dq_c_dt=jnp.zeros(T.shape),
        dq_r_dt=jnp.zeros(T.shape),
        precip=jnp.zeros(p_s.shape),
        sw_net_sfc=jnp.zeros(p_s.shape),
        lw_net_sfc=jnp.zeros(p_s.shape),
        sw_up_toa=jnp.zeros(p_s.shape),
        lw_up_toa=jnp.zeros(p_s.shape),
        sw_down_toa=jnp.zeros(p_s.shape),
    )
    for field_name in _OPTIONAL_3D_OUTPUT_FIELDS:
        if field_name in PhysicsOutput._fields:
            kwargs[field_name] = jnp.zeros(T.shape)
    if "conv_prog" in PhysicsOutput._fields:
        kwargs["conv_prog"] = jnp.asarray(0.0, dtype=T.dtype)
    return kwargs


class _MockModel:
    """Dynamics mock: T += increment * dt / 86400."""
    _state_type = HydrostaticState
    def __init__(self, increment=1.0):
        self._inc = increment
    def step(self, state, dt):
        return state._replace(T=state.T.replace(data=state.T.data + self._inc * dt / 86400.0))


def _mock_step_unified(need_rad, T, p_s, q_v, q_c, q_r, u, v,
                       sst, sic, lat, lon, doy, sod, dt, sw, s0, o3, aer,
                       h_dT, h_sw_sfc, h_lw_sfc, h_sw_toa, h_lw_toa, h_sw_dtoa,
                       **kw):
    tau = kw.get("tau_equator", jnp.float32(7.2))
    kwargs = _zero_physics_output(T, p_s)
    kwargs["dT_dt"] = jnp.full(T.shape, 1e-5) * tau / 7.2
    phys = PhysicsOutput(**kwargs)
    held = (h_dT, h_sw_sfc, h_lw_sfc, h_sw_toa, h_lw_toa, h_sw_dtoa)
    return phys, held


def _make_carry(T_val=280.0):
    s3 = (6, N, N, NLEV)
    s2 = (6, N, N)
    state = HydrostaticState(
        u=Field(jnp.zeros(s3), name="u", dims=("f","x","y","l"), units="m/s"),
        v=Field(jnp.zeros(s3), name="v", dims=("f","x","y","l"), units="m/s"),
        T=Field(jnp.full(s3, T_val), name="T", dims=("f","x","y","l"), units="K"),
        p_s=Field(jnp.full(s2, 101325.0), name="p_s", dims=("f","x","y"), units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=("f","x","y"), units="m2/s2"),
    )
    return pack_carry(
        state, q_v=jnp.ones(s3)*0.01, q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
        held_dT_rad=jnp.zeros(s3), held_sw_net_sfc=jnp.zeros(s2),
        held_lw_net_sfc=jnp.zeros(s2), held_sw_up_toa=jnp.zeros(s2),
        held_lw_up_toa=jnp.zeros(s2), held_sw_down_toa=jnp.zeros(s2),
        step_index=0,
    )


_FORCING = pack_forcing(
    sst=jnp.full((6,N,N), 300.0), sic=jnp.zeros((6,N,N)),
    day_of_year=1.0, seconds_of_day=0.0,
    solar_weights=jnp.ones(14), s_0=1361.0,
    o3_vmr=jnp.zeros((6,N,N,NLEV)), aerosol_od=jnp.zeros((6,N,N)),
)


def _build(gradient_checkpoint, tau_equator=7.2):
    return build_segment_fn(
        model=_MockModel(), step_unified=_mock_step_unified, grid=_GRID,
        sigma_full=jnp.linspace(0.1, 1.0, NLEV),
        dsigma=jnp.full(NLEV, 1.0/NLEV), dt=600.0, rad_update_steps=1,
        microphysics="none", fix_moisture=False, fix_mass=False,
        fric_decay=jnp.ones(NLEV), qv_smooth_coeff=0.0,
        lat=_GRID.lat, lon=_GRID.lon, start_day=0.0,
        gradient_checkpoint=gradient_checkpoint,
        tau_equator=tau_equator,
    )


class TestCheckpointedForwardPass:
    """Checkpointed and non-checkpointed forward passes match."""

    def test_results_match_5_steps(self):
        run_ckpt = _build(gradient_checkpoint=True)
        run_nockpt = _build(gradient_checkpoint=False)

        ic = _make_carry()
        res_ckpt = run_ckpt.raw(ic, 5, _FORCING)
        res_nockpt = run_nockpt.raw(ic, 5, _FORCING)

        np.testing.assert_allclose(
            np.asarray(res_ckpt.T), np.asarray(res_nockpt.T), atol=1e-12,
        )

    def test_results_match_50_steps(self):
        run_ckpt = _build(gradient_checkpoint=True)
        run_nockpt = _build(gradient_checkpoint=False)

        ic = _make_carry()
        res_ckpt = run_ckpt.raw(ic, 50, _FORCING)
        res_nockpt = run_nockpt.raw(ic, 50, _FORCING)

        np.testing.assert_allclose(
            np.asarray(res_ckpt.T), np.asarray(res_nockpt.T), atol=1e-10,
        )


class TestGradientThroughCheckpoint:
    """Gradients flow through checkpointed rollouts."""

    def test_grad_5_steps(self):
        """5-step gradient is finite and nonzero."""
        run_seg = _build(gradient_checkpoint=True, tau_equator=7.2)

        def loss(tau):
            seg = _build(gradient_checkpoint=True, tau_equator=tau)
            ic = _make_carry()
            target = _make_carry(282.0)
            pred = seg.raw(ic, 5, _FORCING)
            return jnp.mean((pred.T - target.T) ** 2)

        grad = jax.grad(loss)(jnp.float32(7.2))
        assert jnp.isfinite(grad)
        assert float(jnp.abs(grad)) > 0

    def test_grad_100_steps(self):
        """100-step gradient through checkpointed scan is finite."""
        def loss(tau):
            seg = _build(gradient_checkpoint=True, tau_equator=tau)
            ic = _make_carry()
            target = _make_carry(282.0)
            pred = seg.raw(ic, 100, _FORCING)
            return jnp.mean((pred.T - target.T) ** 2)

        grad = jax.grad(loss)(jnp.float32(7.2))
        assert jnp.isfinite(grad), f"Gradient is {grad}"
        assert float(jnp.abs(grad)) > 0

    def test_grad_checkpointed_matches_uncheckpointed(self):
        """Checkpointed and uncheckpointed gradients agree."""
        def loss(tau, ckpt):
            seg = _build(gradient_checkpoint=ckpt, tau_equator=tau)
            ic = _make_carry()
            target = _make_carry(282.0)
            pred = seg.raw(ic, 10, _FORCING)
            return jnp.mean((pred.T - target.T) ** 2)

        tau = jnp.float32(7.2)
        grad_ckpt = jax.grad(loss)(tau, True)
        grad_nockpt = jax.grad(loss)(tau, False)

        np.testing.assert_allclose(
            float(grad_ckpt), float(grad_nockpt), rtol=1e-4,
        )


class TestHierarchicalCheckpointing:
    """Multi-segment rollout with two levels of checkpointing."""

    def test_multi_segment_gradient(self):
        """Gradient through 3 segments of 10 steps each (30 steps total)."""
        def loss(tau):
            seg = _build(gradient_checkpoint=True, tau_equator=tau)
            ic = _make_carry()
            target = _make_carry(282.0)

            # 3 segments of 10 steps, with outer checkpointing
            def _one_seg(carry, _):
                return seg.raw(carry, 10, _FORCING), None

            ckpt_seg = jax.checkpoint(_one_seg, prevent_cse=False)
            final, _ = jax.lax.scan(ckpt_seg, ic, None, length=3)

            return jnp.mean((final.T - target.T) ** 2)

        grad = jax.grad(loss)(jnp.float32(7.2))
        assert jnp.isfinite(grad)
        assert float(jnp.abs(grad)) > 0
        print(f"Hierarchical checkpoint gradient: {float(grad):.6e}")
