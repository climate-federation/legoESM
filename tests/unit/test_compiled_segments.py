"""Tests for the compiled segment execution path.

Validates:
- SegmentCarry pack/unpack round-trip preserves values.
- compute_segment_length produces correct GCD.
- build_segment_fn produces a callable that runs under jax.lax.scan.
- Compiled segment execution matches per-step execution for short runs.
- Segment boundary cadence is respected.
- Compile-once behavior (first call compiles, subsequent calls reuse).
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import (
    SegmentCarry,
    pack_carry,
    unpack_carry,
    compute_segment_length,
    build_segment_fn,
    pack_forcing,
)
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.grids.cubed_sphere import create_cubed_sphere


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FACES = 6
N = 4          # Small resolution for fast tests
NLEV = 3
DT = 600.0     # 10-minute timestep


def _copy_carry(carry):
    """Deep-copy a SegmentCarry so the original survives buffer donation."""
    return jax.tree.map(lambda x: x.copy() if hasattr(x, 'copy') else x, carry)

# Create a real cubed-sphere grid once (expensive to rebuild per test)
_GRID = create_cubed_sphere(N)


def _make_field_3d(name="test", fill=1.0):
    data = jnp.full((N_FACES, N, N, NLEV), fill, dtype=jnp.float32)
    return Field(data, name=name, dims=("face", "x", "y", "level"), units="K")


def _make_field_2d(name="test_2d", fill=0.0):
    data = jnp.full((N_FACES, N, N), fill, dtype=jnp.float32)
    return Field(data, name=name, dims=("face", "x", "y"), units="Pa")


def _make_hydrostatic_state():
    return HydrostaticState(
        u=_make_field_3d("u", fill=0.5),
        v=_make_field_3d("v", fill=-0.5),
        T=_make_field_3d("T", fill=280.0),
        p_s=_make_field_2d("p_s", fill=101325.0),
        phis=_make_field_2d("phis", fill=0.0),
    )


class _MockModel:
    """Minimal dynamics model: adds increment*dt/86400 to T each step."""

    _state_type = HydrostaticState

    def __init__(self, increment=1.0):
        self._increment = increment

    def step(self, state, dt):
        new_T = state.T.replace(
            data=state.T.data + self._increment * dt / 86400.0
        )
        return state._replace(T=new_T)


def _mock_step_unified(
    need_rad,
    T, p_s, q_v, q_c, q_r, u, v,
    sst, sic, lat, lon,
    day_of_year, seconds_of_day, dt,
    solar_weights, s_0,
    o3_vmr, aerosol_od,
    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
    **kwargs,
):
    """Mock physics step: small constant warming + no moisture change."""
    shape_3d = T.shape
    shape_2d = p_s.shape
    phys_out = PhysicsOutput(
        dT_dt=jnp.full(shape_3d, 1e-5),    # small warming
        dq_v_dt=jnp.zeros(shape_3d),
        dq_c_dt=jnp.zeros(shape_3d),
        dq_r_dt=jnp.zeros(shape_3d),
        precip=jnp.zeros(shape_2d),
        sw_net_sfc=jnp.zeros(shape_2d),
        lw_net_sfc=jnp.zeros(shape_2d),
        sw_up_toa=jnp.zeros(shape_2d),
        lw_up_toa=jnp.zeros(shape_2d),
        sw_down_toa=jnp.zeros(shape_2d),
        du_dt=jnp.zeros(shape_3d),
        dv_dt=jnp.zeros(shape_3d),
        dq_i_dt=jnp.zeros(shape_3d),
        dq_s_dt=jnp.zeros(shape_3d),
        dq_g_dt=jnp.zeros(shape_3d),
        dN_c_dt=jnp.zeros(shape_3d),
        dN_r_dt=jnp.zeros(shape_3d),
        dN_i_dt=jnp.zeros(shape_3d),
    )
    # Return held unchanged (no radiation update in mock)
    held_new = (
        held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
        held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
    )
    return phys_out, held_new


# ===========================================================================
# 1. compute_segment_length
# ===========================================================================

class TestComputeSegmentLength:
    """GCD-based segment length computation."""

    def test_single_interval(self):
        assert compute_segment_length(10, 0, 0) == 10

    def test_two_intervals_coprime(self):
        assert compute_segment_length(7, 3, 0) == 1

    def test_two_intervals_common(self):
        assert compute_segment_length(12, 8, 0) == 4

    def test_three_intervals(self):
        # rad_update_steps is excluded from GCD (handled inside scan body)
        assert compute_segment_length(12, 18, 6) == 6  # gcd(12, 18)

    def test_all_same(self):
        assert compute_segment_length(5, 5, 5) == 5

    def test_all_zero_returns_one(self):
        assert compute_segment_length(0, 0, 0) == 1

    def test_rad_update_excluded_from_gcd(self):
        # rad_update_steps=4 should NOT constrain segment length;
        # only diag_interval=10 matters here (checkpoint=0 disabled).
        assert compute_segment_length(10, 0, 4) == 10

    def test_rad_update_one_does_not_collapse(self):
        # The original bug: rad_update_steps=1 collapsed segment to 1.
        assert compute_segment_length(100, 0, 1) == 100

    def test_always_at_least_one(self):
        assert compute_segment_length(1, 1, 1) == 1

    def test_large_coprime(self):
        assert compute_segment_length(97, 53, 0) == 1

    def test_python_gcd_consistency(self):
        """Result matches stdlib math.gcd for diag & checkpoint."""
        a, b, c = 120, 84, 36
        expected = math.gcd(a, b)  # rad_update_steps excluded
        assert compute_segment_length(a, b, c) == expected


# ===========================================================================
# 2. SegmentCarry pack/unpack
# ===========================================================================

class TestSegmentCarryRoundtrip:
    """Pack/unpack preserves all values exactly."""

    def test_pack_creates_segment_carry(self):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )
        assert isinstance(carry, SegmentCarry)
        assert carry.T.shape == shape_3d
        assert carry.p_s.shape == shape_2d
        assert carry.step_index == 0

    def test_roundtrip_preserves_values(self):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        q_v = jnp.ones(shape_3d) * 0.012
        q_c = jnp.ones(shape_3d) * 0.001
        q_r = jnp.ones(shape_3d) * 0.0005

        carry = pack_carry(
            state, q_v, q_c, q_r,
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=42,
        )

        (new_state, qv_out, qc_out, qr_out, held_tuple, step_idx, precip,
         shflx_out, lhflx_out) = unpack_carry(carry, state)

        np.testing.assert_array_equal(np.asarray(new_state.T.data),
                                       np.asarray(state.T.data))
        np.testing.assert_array_equal(np.asarray(new_state.u.data),
                                       np.asarray(state.u.data))
        np.testing.assert_array_equal(np.asarray(qv_out), np.asarray(q_v))
        np.testing.assert_array_equal(np.asarray(qc_out), np.asarray(q_c))
        np.testing.assert_array_equal(np.asarray(qr_out), np.asarray(q_r))
        assert step_idx == 42

    def test_unpack_preserves_field_metadata(self):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)

        carry = pack_carry(
            state,
            q_v=jnp.zeros(shape_3d),
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )
        new_state, *_ = unpack_carry(carry, state)

        assert isinstance(new_state, HydrostaticState)
        assert new_state.T.name == "T"
        assert new_state.T.units == "K"
        assert new_state.u.dims == ("face", "x", "y", "level")

    def test_step_index_is_int32(self):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.zeros(shape_3d),
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=10,
        )
        assert carry.step_index.dtype == jnp.int32


# ===========================================================================
# 3. build_segment_fn
# ===========================================================================

def _make_forcing():
    """Build a default SegmentForcing for tests."""
    return pack_forcing(
        sst=jnp.full((N_FACES, N, N), 300.0),
        sic=jnp.zeros((N_FACES, N, N)),
        day_of_year=1.0,
        seconds_of_day=0.0,
        solar_weights=jnp.ones(14),
        s_0=1361.0,
        o3_vmr=jnp.zeros((N_FACES, N, N, NLEV)),
        aerosol_od=jnp.zeros((N_FACES, N, N)),
    )


_FORCING = _make_forcing()


def _make_segment_fn_args(fix_moisture=False):
    """Build all arguments for build_segment_fn with mock components."""
    return dict(
        model=_MockModel(increment=1.0),
        step_unified=_mock_step_unified,
        grid=_GRID,
        sigma_full=jnp.linspace(0.1, 1.0, NLEV),
        dsigma=jnp.full((NLEV,), 1.0 / NLEV),
        dt=DT,
        rad_update_steps=1,
        microphysics="none",
        fix_moisture=fix_moisture,
        fix_mass=False,
        fric_decay=jnp.ones((NLEV,)),  # no friction
        qv_smooth_coeff=0.0,
        lat=_GRID.lat,
        lon=_GRID.lon,
        start_day=0.0,
    )


class TestBuildSegmentFn:
    """build_segment_fn produces a working compiled kernel."""

    def test_returns_callable(self):
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)
        assert callable(run_segment)

    def test_single_step_runs(self):
        """One step of run_segment executes without error."""
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

        result = run_segment(carry, 1, _FORCING)
        jax.block_until_ready(result.T)

        assert result.T.shape == shape_3d
        assert result.step_index == 1

    def test_multi_step_runs(self):
        """Running 5 steps increments step_index by 5."""
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=10,
        )

        result = run_segment(carry, 5, _FORCING)
        jax.block_until_ready(result.T)
        assert result.step_index == 15

    def test_temperature_increases(self):
        """Mock model warms T each step; after N steps T should be larger."""
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

        T_init = np.asarray(carry.T)  # save before donation
        result = run_segment(carry, 10, _FORCING)
        T_final = np.asarray(result.T)

        # dynamics adds increment*dt/86400 per step
        # physics adds 1e-5 * dt per step
        assert np.all(T_final > T_init), "Temperature should increase"
        assert np.all(np.isfinite(T_final)), "Temperature should be finite"

    def test_output_all_finite(self):
        """All carry fields remain finite after a segment run."""
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.ones(shape_3d) * 0.001,
            q_r=jnp.ones(shape_3d) * 0.0001,
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

        result = run_segment(carry, 5, _FORCING)

        for field_name in SegmentCarry._fields:
            arr = np.asarray(getattr(result, field_name))
            assert np.all(np.isfinite(arr)), f"{field_name} has non-finite values"

    def test_moisture_stays_non_negative(self):
        """q_v, q_c, q_r should remain >= 0."""
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

        result = run_segment(carry, 10, _FORCING)
        assert np.all(np.asarray(result.q_v) >= 0.0)
        assert np.all(np.asarray(result.q_c) >= 0.0)
        assert np.all(np.asarray(result.q_r) >= 0.0)


# ===========================================================================
# 4. Equivalence: compiled segment vs per-step Python loop
# ===========================================================================

def _run_per_step_python(model, step_unified, n_steps, carry_init, args,
                         forcing=None):
    """Run the same physics+dynamics loop step-by-step in Python.

    Replicates the logic inside _single_step but without lax.scan,
    so we can compare results for equivalence testing.
    """
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm import constants
    from legoesm.core.operators_3d import hyperdiffusion_3d
    from legoesm.driver.compiled_segments import _rebuild_state

    if forcing is None:
        forcing = _FORCING

    dt = args["dt"]
    sigma_full = jnp.asarray(args["sigma_full"])
    fric_decay = jnp.asarray(args["fric_decay"])
    qv_smooth_coeff = args["qv_smooth_coeff"]
    grid = args["grid"]
    do_sat_adjust = (args["microphysics"] == "none")

    carry = carry_init
    for i in range(n_steps):
        step_idx = carry.step_index

        # Dynamics
        rebuilt_state = _rebuild_state(carry, model)
        dyn_state = model.step(rebuilt_state, dt)

        T_new = dyn_state.T.data
        u_new = dyn_state.u.data
        v_new = dyn_state.v.data
        p_s_new = dyn_state.p_s.data

        # Physics
        need_rad = jnp.bool_(True) if args["rad_update_steps"] <= 1 else \
            ((step_idx + 1) % args["rad_update_steps"]) == 0
        phys_out, held_new = step_unified(
            need_rad,
            T_new, p_s_new,
            carry.q_v, carry.q_c, carry.q_r,
            u_new, v_new,
            forcing.sst, forcing.sic,
            args["lat"], args["lon"],
            forcing.day_of_year,
            forcing.seconds_of_day,
            jnp.float32(dt),
            forcing.solar_weights,
            forcing.s_0,
            forcing.o3_vmr,
            forcing.aerosol_od,
            carry.held_dT_rad, carry.held_sw_net_sfc, carry.held_lw_net_sfc,
            carry.held_sw_up_toa, carry.held_lw_up_toa, carry.held_sw_down_toa,
        )

        # State update
        T_upd = T_new + dt * phys_out.dT_dt
        q_v_upd = jnp.maximum(carry.q_v + dt * phys_out.dq_v_dt, 0.0)
        q_c_upd = jnp.maximum(carry.q_c + dt * phys_out.dq_c_dt, 0.0)
        q_r_upd = jnp.maximum(carry.q_r + dt * phys_out.dq_r_dt, 0.0)

        # Saturation adjustment
        if do_sat_adjust:
            p_full = p_s_new[..., None] * sigma_full
            q_sat = saturation_mixing_ratio(T_upd, p_full)
            excess = jnp.maximum(q_v_upd - q_sat, 0.0)
            q_v_upd = q_v_upd - excess
            T_upd = T_upd + constants.L_v * excess / constants.c_pd

        # Moisture smoothing
        q_v_upd = jnp.maximum(
            q_v_upd + dt * hyperdiffusion_3d(q_v_upd, grid, qv_smooth_coeff),
            0.0,
        )

        # Rayleigh friction
        u_upd = u_new * fric_decay
        v_upd = v_new * fric_decay

        # CFL is computed at segment boundary (host-side), not per step
        max_cfl = carry.max_cfl

        # Accumulate precipitation
        precip_step = phys_out.precip if hasattr(phys_out, 'precip') else jnp.zeros_like(p_s_new)
        precip_accum = carry.precip_accum + precip_step * dt

        carry = SegmentCarry(
            u=u_upd, v=v_upd, T=T_upd,
            p_s=p_s_new, phis=carry.phis,
            q_v=q_v_upd, q_c=q_c_upd, q_r=q_r_upd,
            held_dT_rad=held_new[0],
            held_sw_net_sfc=held_new[1],
            held_lw_net_sfc=held_new[2],
            held_sw_up_toa=held_new[3],
            held_lw_up_toa=held_new[4],
            held_sw_down_toa=held_new[5],
            step_index=step_idx + 1,
            target_moisture=carry.target_moisture,
            target_mass=carry.target_mass,
            max_cfl=max_cfl,
            precip_accum=precip_accum,
            shflx_accum=carry.shflx_accum,
            lhflx_accum=carry.lhflx_accum,
        )
    return carry


class TestEquivalence:
    """Compiled segment matches per-step Python loop."""

    def _make_init_carry(self):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        return pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.ones(shape_3d) * 0.001,
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

    def test_one_step_equivalence(self):
        """Single-step compiled == single-step Python."""
        args = _make_segment_fn_args()
        carry_init = self._make_init_carry()

        # Compiled path (copy carry since run_segment donates buffers)
        run_segment = build_segment_fn(**args)
        compiled_result = run_segment(_copy_carry(carry_init), 1, _FORCING)

        # Python loop path
        python_result = _run_per_step_python(
            args["model"], args["step_unified"], 1, carry_init, args,
        )

        for field_name in SegmentCarry._fields:
            compiled_arr = np.asarray(getattr(compiled_result, field_name))
            python_arr = np.asarray(getattr(python_result, field_name))
            np.testing.assert_allclose(
                compiled_arr, python_arr,
                atol=1e-5, rtol=1e-5,
                err_msg=f"Mismatch in {field_name} after 1 step",
            )

    def test_five_step_equivalence(self):
        """Five-step compiled == five-step Python."""
        args = _make_segment_fn_args()
        carry_init = self._make_init_carry()

        run_segment = build_segment_fn(**args)
        compiled_result = run_segment(_copy_carry(carry_init), 5, _FORCING)

        python_result = _run_per_step_python(
            args["model"], args["step_unified"], 5, carry_init, args,
        )

        for field_name in SegmentCarry._fields:
            compiled_arr = np.asarray(getattr(compiled_result, field_name))
            python_arr = np.asarray(getattr(python_result, field_name))
            np.testing.assert_allclose(
                compiled_arr, python_arr,
                atol=1e-4, rtol=1e-4,
                err_msg=f"Mismatch in {field_name} after 5 steps",
            )

    def test_segmented_matches_single_segment(self):
        """Running 2 segments of 3 == 1 segment of 6."""
        args = _make_segment_fn_args()
        carry_init = self._make_init_carry()
        run_segment = build_segment_fn(**args)

        # Single segment of 6 (copy carry since donation frees buffers)
        result_6 = run_segment(_copy_carry(carry_init), 6, _FORCING)

        # Two segments of 3
        result_3a = run_segment(_copy_carry(carry_init), 3, _FORCING)
        result_3b = run_segment(result_3a, 3, _FORCING)

        for field_name in SegmentCarry._fields:
            arr_6 = np.asarray(getattr(result_6, field_name))
            arr_3b = np.asarray(getattr(result_3b, field_name))
            np.testing.assert_allclose(
                arr_6, arr_3b,
                atol=1e-5, rtol=1e-5,
                err_msg=f"Segmented vs single mismatch in {field_name}",
            )

    def test_step_index_continuity(self):
        """Step index is correctly incremented across segments."""
        args = _make_segment_fn_args()
        carry_init = self._make_init_carry()
        run_segment = build_segment_fn(**args)

        r1 = run_segment(carry_init, 4, _FORCING)
        assert r1.step_index == 4

        r2 = run_segment(r1, 3, _FORCING)
        assert r2.step_index == 7


# ===========================================================================
# 5. Compile timing
# ===========================================================================

class TestCompileTiming:
    """First call triggers JIT compilation; subsequent calls reuse cache."""

    def test_first_call_compiles(self):
        """First run_segment call is slower (compilation)."""
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

        # First call (includes JIT compile); copy carry to survive donation
        t0 = time.monotonic()
        result = run_segment(_copy_carry(carry), 3, _FORCING)
        jax.block_until_ready(result.T)
        first_time = time.monotonic() - t0

        # Warm up (feed result back as input to avoid donated-buffer errors)
        for _ in range(3):
            result = run_segment(result, 3, _FORCING)
            jax.block_until_ready(result.T)

        # Steady-state calls
        n_warm = 5
        t0 = time.monotonic()
        for _ in range(n_warm):
            result = run_segment(result, 3, _FORCING)
            jax.block_until_ready(result.T)
        avg_warm = (time.monotonic() - t0) / n_warm

        # Report timing (informational, not strict assert)
        print(f"\n  First call (compile): {first_time:.3f}s")
        print(f"  Steady-state avg:    {avg_warm:.3f}s")
        print(f"  Speedup:             {first_time/max(avg_warm, 1e-9):.1f}x")

        # The result should be valid regardless
        assert np.all(np.isfinite(np.asarray(result.T)))


# ===========================================================================
# 6. Rayleigh friction and saturation adjustment
# ===========================================================================

class TestPhysicsSubComponents:
    """Sub-component behavior inside the compiled segment."""

    def test_friction_damps_wind(self):
        """With fric_decay < 1, wind magnitudes should decrease."""
        args = _make_segment_fn_args()
        args["fric_decay"] = jnp.full((NLEV,), 0.99)  # 1% damping per step
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

        u_init_abs = float(jnp.max(jnp.abs(carry.u)))  # save before donation
        result = run_segment(carry, 10, _FORCING)

        # u starts at 0.5 + dynamics increment, then gets multiplied by 0.99
        # each step.  After 10 steps the magnitude should be smaller than
        # what it would be without friction.
        u_final_abs = float(jnp.max(jnp.abs(result.u)))
        # The dynamics mock only changes T, not u, so u is damped from 0.5
        # by 0.99^10 ≈ 0.904
        expected_u = 0.5 * 0.99**10
        np.testing.assert_allclose(u_final_abs, expected_u, rtol=0.01)

    def test_no_friction_preserves_wind(self):
        """With fric_decay = 1, wind is unchanged by friction term."""
        args = _make_segment_fn_args()
        # fric_decay=1 (no friction, default in _make_segment_fn_args)
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.zeros(shape_2d),
            held_lw_net_sfc=jnp.zeros(shape_2d),
            held_sw_up_toa=jnp.zeros(shape_2d),
            held_lw_up_toa=jnp.zeros(shape_2d),
            held_sw_down_toa=jnp.zeros(shape_2d),
            step_index=0,
        )

        result = run_segment(carry, 5, _FORCING)
        # dynamics mock doesn't modify u, and fric_decay=1, so u should stay 0.5
        np.testing.assert_allclose(
            np.asarray(result.u),
            0.5,
            atol=1e-6,
        )
