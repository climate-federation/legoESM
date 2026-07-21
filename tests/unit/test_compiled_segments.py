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

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import (
    SegmentCarry,
    pack_carry,
    unpack_carry,
    compute_segment_length,
    build_segment_fn,
    pack_forcing,
    _sqrt_checkpointed_scan,
    _NESTED_CKPT_STEPS,
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
    T, p_s, q_v, q_c, q_r, conv_prog, u, v,
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
        conv_prog=conv_prog,
    )
    # Return held unchanged (no radiation update in mock)
    held_new = (
        held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
        held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
    )
    # Slab-land temperature passes through unchanged (mock has no land).
    return phys_out, held_new, kwargs.get("T_land")


# Optional carry fields that are None in the legacy warm-rain/diagnostic
# carries — skipped by the field-by-field equivalence comparisons below.
_OPTIONAL_CARRY_FIELDS = (
    "q_i", "q_s", "q_g", "N_c", "N_r", "N_i",
    "tke", "qke", "gwd_spectrum", "cloud_fraction", "land_ml", "w_land", "snow",
)


def _assert_carries_match(compiled_result, python_result, atol, rtol,
                          context=""):
    """Field-by-field carry comparison; optional fields compared only
    when present (non-None) on both sides."""
    for field_name in SegmentCarry._fields:
        c_val = getattr(compiled_result, field_name)
        p_val = getattr(python_result, field_name)
        if field_name in _OPTIONAL_CARRY_FIELDS and (
                c_val is None or p_val is None):
            assert c_val is None and p_val is None, (
                f"{field_name} present on one side only {context}"
            )
            continue
        np.testing.assert_allclose(
            np.asarray(c_val), np.asarray(p_val),
            atol=atol, rtol=rtol,
            err_msg=f"Mismatch in {field_name} {context}",
        )


def _mock_step_unified_stateful(
    need_rad,
    T, p_s, q_v, q_c, q_r, conv_prog, u, v,
    sst, sic, lat, lon,
    day_of_year, seconds_of_day, dt,
    solar_weights, s_0,
    o3_vmr, aerosol_od,
    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
    **kwargs,
):
    """Stateful mock physics (issue #413): advances a prognostic tke
    carry by the AR1-like map ``tke_new = 0.9*tke + 1e-3`` so memory of
    the input carry is observable across steps; everything else matches
    the stateless mock."""
    phys_out, held_new, _T_land = _mock_step_unified(
        need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
        sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
        solar_weights, s_0, o3_vmr, aerosol_od,
        held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
        held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
        **kwargs,
    )
    tke = kwargs.get("tke")
    if tke is not None:
        phys_out = phys_out._replace(
            tke=(0.9 * tke + 1e-3).astype(tke.dtype),
        )
    cloud_fraction = kwargs.get("cloud_fraction")
    if cloud_fraction is not None:
        # Diagnostic CLUBB cf carry (one-step lag): advance so its memory across
        # steps is observable in the compiled-vs-python equivalence check.
        phys_out = phys_out._replace(
            cloud_fraction=(0.5 * cloud_fraction + 0.1).astype(cloud_fraction.dtype),
        )
    gwd_spectrum = kwargs.get("gwd_spectrum")
    if gwd_spectrum is not None:
        phys_out = phys_out._replace(
            gwd_spectrum=(0.5 * gwd_spectrum).astype(gwd_spectrum.dtype),
        )
    return phys_out, held_new, _T_land


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

    def test_no_host_cadence_uses_fallback(self):
        # diag + checkpoint disabled (e.g. spmd milestone-1): without a
        # fallback the segment collapses to 1 and EVERY step pays a host
        # boundary — the production-SPMD anti-scaling (job 8471423; the
        # driver passes one model day of steps).
        assert compute_segment_length(0, 0, 0, fallback_interval=144) == 144

    def test_fallback_ignored_when_cadence_exists(self):
        assert compute_segment_length(10, 0, 0, fallback_interval=144) == 10

    def test_fallback_zero_or_negative_keeps_one(self):
        # DT > 86400 makes the driver's int(days*86400/DT) collapse to 0
        # — must stay the legacy 1-step segment, never 0.
        assert compute_segment_length(0, 0, 0, fallback_interval=0) == 1
        assert compute_segment_length(0, 0, 0, fallback_interval=-5) == 1

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

        (new_state, qv_out, qc_out, qr_out, conv_prog_out, held_tuple, step_idx, precip,
         shflx_out, lhflx_out) = unpack_carry(carry, state)

        np.testing.assert_array_equal(np.asarray(new_state.T.data),
                                       np.asarray(state.T.data))
        np.testing.assert_array_equal(np.asarray(new_state.u.data),
                                       np.asarray(state.u.data))
        np.testing.assert_array_equal(np.asarray(qv_out), np.asarray(q_v))
        np.testing.assert_array_equal(np.asarray(qc_out), np.asarray(q_c))
        np.testing.assert_array_equal(np.asarray(qr_out), np.asarray(q_r))
        np.testing.assert_array_equal(np.asarray(conv_prog_out), np.zeros((shape_2d[0] * shape_2d[1] * shape_2d[2],)))
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
        s_0=constants.S_0,
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

    def test_energy_consistent_moisture_clip_wiring(self):
        """Issue #323: the opt-in energy-consistent q_v floor compiles and
        runs in the REAL compiled segment, keeps q_v >= 0 and stays finite,
        and is BITWISE-IDENTICAL to the legacy path when off (default ==
        explicit False) — the bit-identity guarantee for the gate."""
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)

        def _fresh_carry():
            # fresh each call: run_segment donates its input buffers
            return pack_carry(
                _make_hydrostatic_state(),
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

        # flag ON: compiles, runs, finite, non-negative q_v
        args_on = _make_segment_fn_args()
        args_on["energy_consistent_moisture_clip"] = True
        res_on = build_segment_fn(**args_on)(_fresh_carry(), 3, _FORCING)
        jax.block_until_ready(res_on.q_v)
        assert np.all(np.isfinite(np.asarray(res_on.q_v)))
        assert np.all(np.isfinite(np.asarray(res_on.T)))
        assert np.all(np.asarray(res_on.q_v) >= 0.0)

        # default (flag absent) vs explicit False: bitwise identical
        r_default = build_segment_fn(**_make_segment_fn_args())(
            _fresh_carry(), 3, _FORCING)
        args_off = _make_segment_fn_args()
        args_off["energy_consistent_moisture_clip"] = False
        r_off = build_segment_fn(**args_off)(_fresh_carry(), 3, _FORCING)
        np.testing.assert_array_equal(
            np.asarray(r_default.T), np.asarray(r_off.T))
        np.testing.assert_array_equal(
            np.asarray(r_default.q_v), np.asarray(r_off.q_v))

    def test_energy_consistent_moisture_clip_fires_in_segment(self):
        """Issue #323: when the vapour sink drives q_v < 0 inside the REAL
        compiled segment, the energy-consistent floor (flag ON) cools T by
        exactly ``(L_v/c_pd)*deficit`` relative to the legacy floor (flag
        OFF), while both floor q_v to zero.  This exercises the correction
        firing in the segment, not just the helper in isolation."""
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        q_v0 = 0.01

        def _fresh_carry():
            return pack_carry(
                _make_hydrostatic_state(),
                q_v=jnp.ones(shape_3d) * q_v0,
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

        def _drying_step(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                         sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                         solar_weights, s_0, o3_vmr, aerosol_od,
                         held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                         held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                         **kwargs):
            base, held_new, T_land = _mock_step_unified(
                need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                solar_weights, s_0, o3_vmr, aerosol_od,
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa, **kwargs)
            # Sink twice the available vapour => q_v + dt*dq_v = -q_v < 0
            # (clip fires), with condensation warming matching the FULL sink.
            dq_v = -2.0 * q_v / dt
            dT = -(constants.L_v / constants.c_pd) * dq_v
            return base._replace(dq_v_dt=dq_v, dT_dt=dT), held_new, T_land

        base_args = _make_segment_fn_args()
        base_args["step_unified"] = _drying_step
        r_on = build_segment_fn(
            **{**base_args, "energy_consistent_moisture_clip": True}
        )(_fresh_carry(), 1, _FORCING)
        r_off = build_segment_fn(
            **{**base_args, "energy_consistent_moisture_clip": False}
        )(_fresh_carry(), 1, _FORCING)
        jax.block_until_ready(r_on.T)

        # both floor q_v to zero (q_v_raw = -q_v0 < 0)
        np.testing.assert_allclose(np.asarray(r_on.q_v), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(r_off.q_v), 0.0, atol=1e-6)
        # flag ON is cooler by (L_v/c_pd) * deficit, deficit = q_v0
        expected = (constants.L_v / constants.c_pd) * q_v0
        dT_diff = np.asarray(r_off.T) - np.asarray(r_on.T)
        np.testing.assert_allclose(dT_diff, expected, rtol=2e-3)

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
            val = getattr(result, field_name)
            if field_name in _OPTIONAL_CARRY_FIELDS and val is None:
                continue  # optional fields: None in legacy carries
            arr = np.asarray(val)
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
        # Per-step solar time — mirror _single_step's diurnal-cycle fix
        # (advance the wall clock from the absolute step index).
        from legoesm.forcing.time_utils import day_to_calendar
        _abs_day = args["start_day"] + (step_idx + 1) * dt / 86400.0
        _doy_step, _sod_step = day_to_calendar(_abs_day)
        # Stateful-physics carries (issue #413): pass active (non-None)
        # carries by keyword, mirroring _single_step.
        _phys_carry_in = {
            k: getattr(carry, k)
            for k in ("tke", "qke", "gwd_spectrum")
            if getattr(carry, k) is not None
        }
        phys_out, held_new, _T_land_ref = step_unified(
            need_rad,
            T_new, p_s_new,
            carry.q_v, carry.q_c, carry.q_r, carry.conv_prog,
            u_new, v_new,
            forcing.sst, forcing.sic,
            args["lat"], args["lon"],
            _doy_step,
            _sod_step,
            jnp.float32(dt),
            forcing.solar_weights,
            forcing.s_0,
            forcing.o3_vmr,
            forcing.aerosol_od,
            carry.held_dT_rad, carry.held_sw_net_sfc, carry.held_lw_net_sfc,
            carry.held_sw_up_toa, carry.held_lw_up_toa, carry.held_sw_down_toa,
            **_phys_carry_in,
        )

        # State update
        T_upd = T_new + dt * phys_out.dT_dt
        q_v_upd = jnp.maximum(carry.q_v + dt * phys_out.dq_v_dt, 0.0)
        q_c_upd = jnp.maximum(carry.q_c + dt * phys_out.dq_c_dt, 0.0)
        q_r_upd = jnp.maximum(carry.q_r + dt * phys_out.dq_r_dt, 0.0)

        # Time-integrate radiative fluxes + lowest-level T (uses the
        # PRE-sat-adjust T_upd, matching _single_step's accumulation point).
        sw_net_sfc_accum = carry.sw_net_sfc_accum + held_new[1] * dt
        lw_net_sfc_accum = carry.lw_net_sfc_accum + held_new[2] * dt
        sw_up_toa_accum = carry.sw_up_toa_accum + held_new[3] * dt
        lw_up_toa_accum = carry.lw_up_toa_accum + held_new[4] * dt
        sw_down_toa_accum = carry.sw_down_toa_accum + held_new[5] * dt
        t_low_accum = carry.t_low_accum + T_upd[..., -1] * dt
        # Clear-sky TOA accumulation (#843): the mock has no pipeline, so the
        # compiled path runs in "hold" mode — the held clear-sky (zeros)
        # passes through and integrates unchanged.  Mirror that here.
        sw_up_toa_clr_accum = (
            carry.sw_up_toa_clr_accum + carry.held_sw_up_toa_clr * dt)
        lw_up_toa_clr_accum = (
            carry.lw_up_toa_clr_accum + carry.held_lw_up_toa_clr * dt)

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
            conv_prog=phys_out.conv_prog,
            held_dT_rad=held_new[0],
            held_sw_net_sfc=held_new[1],
            held_lw_net_sfc=held_new[2],
            held_sw_up_toa=held_new[3],
            held_lw_up_toa=held_new[4],
            held_sw_up_toa_clr=carry.held_sw_up_toa_clr,
            held_lw_up_toa_clr=carry.held_lw_up_toa_clr,
            held_sw_down_toa=held_new[5],
            step_index=step_idx + 1,
            target_moisture=carry.target_moisture,
            target_mass=carry.target_mass,
            max_cfl=max_cfl,
            precip_accum=precip_accum,
            shflx_accum=carry.shflx_accum,
            lhflx_accum=carry.lhflx_accum,
            sw_up_toa_accum=sw_up_toa_accum,
            lw_up_toa_accum=lw_up_toa_accum,
            sw_up_toa_clr_accum=sw_up_toa_clr_accum,
            lw_up_toa_clr_accum=lw_up_toa_clr_accum,
            sw_down_toa_accum=sw_down_toa_accum,
            sw_net_sfc_accum=sw_net_sfc_accum,
            lw_net_sfc_accum=lw_net_sfc_accum,
            t_low_accum=t_low_accum,
            T_land=carry.T_land,
            # Stateful-physics carries (#413): replaced by the updated
            # values riding PhysicsOutput (None falls back to the input,
            # mirroring _single_step).
            tke=(None if carry.tke is None
                 else (phys_out.tke if phys_out.tke is not None
                       else carry.tke)),
            qke=(None if carry.qke is None
                 else (phys_out.qke if phys_out.qke is not None
                       else carry.qke)),
            gwd_spectrum=(None if carry.gwd_spectrum is None
                          else (phys_out.gwd_spectrum
                                if phys_out.gwd_spectrum is not None
                                else carry.gwd_spectrum)),
            # Lagged total precip for the convective cloud (mirrors the
            # compiled _single_step rebuild).
            conv_precip_prev=precip_step,
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

        _assert_carries_match(compiled_result, python_result,
                              atol=1e-5, rtol=1e-5, context="after 1 step")

    def test_five_step_equivalence(self):
        """Five-step compiled == five-step Python."""
        args = _make_segment_fn_args()
        carry_init = self._make_init_carry()

        run_segment = build_segment_fn(**args)
        compiled_result = run_segment(_copy_carry(carry_init), 5, _FORCING)

        python_result = _run_per_step_python(
            args["model"], args["step_unified"], 5, carry_init, args,
        )

        _assert_carries_match(compiled_result, python_result,
                              atol=1e-4, rtol=1e-4, context="after 5 steps")

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

        _assert_carries_match(result_6, result_3b, atol=1e-5, rtol=1e-5,
                              context="segmented vs single")

    def test_step_index_continuity(self):
        """Step index is correctly incremented across segments."""
        args = _make_segment_fn_args()
        carry_init = self._make_init_carry()
        run_segment = build_segment_fn(**args)

        r1 = run_segment(carry_init, 4, _FORCING)
        assert r1.step_index == 4

        r2 = run_segment(r1, 3, _FORCING)
        assert r2.step_index == 7


class TestSegmentMeanAccumulators:
    """Radiation / T_low segment accumulators (CMOR diurnal-alias fix).

    The mock physics passes the held fluxes through unchanged, so after
    ``n`` steps each flux accumulator must equal EXACTLY
    ``n * dt * held_seed`` — an analytic expectation independent of the
    Python reference loop (which the equivalence tests already cover).
    A wrong sign, a missed step, or accumulating the wrong field all
    break the equality.
    """

    _SW_UP, _LW_UP, _SW_DN = 100.0, 240.0, 340.0
    _SW_SFC, _LW_SFC = 160.0, -60.0

    def _make_carry_with_fluxes(self):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        return pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * 0.01,
            q_c=jnp.zeros(shape_3d),
            q_r=jnp.zeros(shape_3d),
            held_dT_rad=jnp.zeros(shape_3d),
            held_sw_net_sfc=jnp.full(shape_2d, self._SW_SFC),
            held_lw_net_sfc=jnp.full(shape_2d, self._LW_SFC),
            held_sw_up_toa=jnp.full(shape_2d, self._SW_UP),
            held_lw_up_toa=jnp.full(shape_2d, self._LW_UP),
            held_sw_down_toa=jnp.full(shape_2d, self._SW_DN),
            step_index=0,
        )

    def test_flux_accumulators_exact(self):
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)
        n_steps, dt = 5, args["dt"]

        result = run_segment(self._make_carry_with_fluxes(), n_steps, _FORCING)

        for field, held in (
            ("sw_up_toa_accum", self._SW_UP),
            ("lw_up_toa_accum", self._LW_UP),
            ("sw_down_toa_accum", self._SW_DN),
            ("sw_net_sfc_accum", self._SW_SFC),
            ("lw_net_sfc_accum", self._LW_SFC),
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(result, field)),
                np.full((N_FACES, N, N), n_steps * dt * held),
                rtol=1e-6,
                err_msg=f"{field}: expected n*dt*held (held constant in mock)",
            )
        # Segment-mean recovery: accum / duration == the held flux.
        np.testing.assert_allclose(
            np.asarray(result.sw_up_toa_accum) / (n_steps * dt),
            self._SW_UP, rtol=1e-6,
        )

    def test_t_low_accumulator_is_temperature_like(self):
        """t_low_accum / duration must sit near the actual lowest-level T
        (catches a forgotten accumulation -> 0, or the wrong field/level)."""
        args = _make_segment_fn_args()
        run_segment = build_segment_fn(**args)
        n_steps, dt = 5, args["dt"]

        carry0 = self._make_carry_with_fluxes()
        t_low_0 = np.asarray(carry0.T[..., -1])
        result = run_segment(carry0, n_steps, _FORCING)

        t_low_mean = np.asarray(result.t_low_accum) / (n_steps * dt)
        # The toy dycore setup is far from equilibrium and moves T_low by a
        # few K over 5 steps; 10 K still cleanly separates a real T_low mean
        # (~t_low_0) from the failure modes this guards against: a forgotten
        # accumulation (0 K) or accumulating one of the flux fields
        # (100-340 "K").  Exact per-step tracking is covered by the
        # equivalence tests.
        np.testing.assert_allclose(t_low_mean, t_low_0, atol=10.0)

    def test_accumulators_reset_semantics(self):
        """pack_carry seeds zero accumulators (segment-start reset)."""
        carry = self._make_carry_with_fluxes()
        for field in ("sw_up_toa_accum", "lw_up_toa_accum",
                      "sw_down_toa_accum", "sw_net_sfc_accum",
                      "lw_net_sfc_accum", "t_low_accum"):
            assert float(np.abs(np.asarray(getattr(carry, field))).max()) == 0.0


class TestPerStepSolarTime:
    """Diurnal-cycle fix: the compiled scan advances the solar wall clock
    every step instead of freezing it at the segment time.

    A probe mock writes each step's ``seconds_of_day`` into
    ``held_sw_down_toa``; with rad_update_steps=1 the accumulator then holds
    ``sum_i seconds_of_day(step i) * dt``.  A FROZEN clock (the bug) would
    make every step identical, so the accumulator would equal
    ``n * dt * seconds_of_day(step 0)`` — the test asserts it matches the
    ADVANCING sum instead, which differs.
    """

    def _probe_args(self):
        args = _make_segment_fn_args()

        def _probe_step(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                        sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                        *rest, **kwargs):
            # Emit this step's seconds_of_day as the down-TOA "flux" so the
            # accumulator records the solar clock actually seen each step.
            shape_2d = p_s.shape
            sod = jnp.broadcast_to(jnp.asarray(seconds_of_day, p_s.dtype), shape_2d)
            phys_out = PhysicsOutput(
                dT_dt=jnp.zeros(T.shape), dq_v_dt=jnp.zeros(T.shape),
                dq_c_dt=jnp.zeros(T.shape), dq_r_dt=jnp.zeros(T.shape),
                precip=jnp.zeros(shape_2d),
                sw_net_sfc=jnp.zeros(shape_2d), lw_net_sfc=jnp.zeros(shape_2d),
                sw_up_toa=jnp.zeros(shape_2d), lw_up_toa=jnp.zeros(shape_2d),
                sw_down_toa=sod,
                du_dt=jnp.zeros(T.shape), dv_dt=jnp.zeros(T.shape),
                dq_i_dt=jnp.zeros(T.shape), dq_s_dt=jnp.zeros(T.shape),
                dq_g_dt=jnp.zeros(T.shape), dN_c_dt=jnp.zeros(T.shape),
                dN_r_dt=jnp.zeros(T.shape), dN_i_dt=jnp.zeros(T.shape),
                conv_prog=conv_prog,
            )
            held_new = (jnp.zeros(T.shape), jnp.zeros(shape_2d),
                        jnp.zeros(shape_2d), jnp.zeros(shape_2d),
                        jnp.zeros(shape_2d), sod)   # held_sw_down_toa = sod
            # T_land passes through unchanged (mirrors _mock_step_unified) so
            # the carry pytree structure is preserved across the scan.
            return phys_out, held_new, kwargs.get("T_land")

        args["step_unified"] = _probe_step
        args["start_day"] = 0.0
        return args

    def test_seconds_of_day_advances_within_segment(self):
        args = self._probe_args()
        run_segment = build_segment_fn(**args)
        n = 5
        carry = pack_carry(
            _make_hydrostatic_state(),
            q_v=jnp.ones((N_FACES, N, N, NLEV)) * 0.01,
            q_c=jnp.zeros((N_FACES, N, N, NLEV)),
            q_r=jnp.zeros((N_FACES, N, N, NLEV)),
            held_dT_rad=jnp.zeros((N_FACES, N, N, NLEV)),
            held_sw_net_sfc=jnp.zeros((N_FACES, N, N)),
            held_lw_net_sfc=jnp.zeros((N_FACES, N, N)),
            held_sw_up_toa=jnp.zeros((N_FACES, N, N)),
            held_lw_up_toa=jnp.zeros((N_FACES, N, N)),
            held_sw_down_toa=jnp.zeros((N_FACES, N, N)),
            step_index=0,
        )
        result = run_segment(carry, n, _FORCING)

        # Expected: sum over steps of seconds_of_day(step i) * dt, with
        # seconds_of_day(step i) = ((i+1)*DT) % 86400 (start_day=0).
        expect = sum((((i + 1) * DT) % 86400.0) * DT for i in range(n))
        got = float(np.asarray(result.sw_down_toa_accum).flat[0])
        np.testing.assert_allclose(got, expect, rtol=1e-6)

        # And it must NOT equal the frozen-clock value (the bug).
        frozen = n * DT * ((n * DT) % 86400.0)   # all steps at segment-end time
        assert abs(got - frozen) > 1.0, "solar clock appears frozen per segment"


class TestStatefulPhysicsCarry:
    """Issue #413: tke / qke / gwd_spectrum ride the SegmentCarry.

    The stateful mock advances ``tke_new = 0.9*tke + 1e-3`` (and halves
    the GWD spectrum), so a dropped or reseeded carry is observable:
    after n steps tke must equal the n-fold composition of that map
    applied to the SEED — which is only possible if every step consumed
    the previous step's output.
    """

    def _make_stateful_carry(self, tke_seed=1.0e-2):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        ncol = N_FACES * N * N
        return pack_carry(
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
            tke=jnp.full((ncol, NLEV), tke_seed, dtype=jnp.float32),
            gwd_spectrum=jnp.full((ncol, 2, 3), 1.0, dtype=jnp.float32),
        )

    def _stateful_args(self):
        args = _make_segment_fn_args()
        args["step_unified"] = _mock_step_unified_stateful
        return args

    @staticmethod
    def _tke_after(seed, n):
        x = seed
        for _ in range(n):
            x = 0.9 * x + 1e-3
        return x

    def test_carry_advances_across_scan_steps(self):
        """n compiled steps == n-fold map of the SEED (memory, no reseed)."""
        run_segment = build_segment_fn(**self._stateful_args())
        carry = self._make_stateful_carry(tke_seed=1.0e-2)
        out = run_segment(_copy_carry(carry), 5, _FORCING)
        np.testing.assert_allclose(
            np.asarray(out.tke),
            self._tke_after(1.0e-2, 5),
            rtol=1e-6,
            err_msg="tke after 5 scan steps is not the 5-fold map of the "
                    "seed — the carry was dropped or reseeded inside scan",
        )
        np.testing.assert_allclose(
            np.asarray(out.gwd_spectrum), 1.0 * 0.5 ** 5, rtol=1e-6,
        )

    def test_carry_memory_two_seeds_differ(self):
        """Same dynamics, two different input carries ⇒ outputs differ."""
        run_segment = build_segment_fn(**self._stateful_args())
        out_a = run_segment(self._make_stateful_carry(1.0e-2), 3, _FORCING)
        out_b = run_segment(self._make_stateful_carry(5.0e-2), 3, _FORCING)
        assert not np.array_equal(np.asarray(out_a.tke),
                                  np.asarray(out_b.tke)), (
            "Output carry is independent of the input carry (issue #405)"
        )

    def test_segmented_matches_single_segment_stateful(self):
        """2 segments of 3 == 1 segment of 6 with the carry active."""
        run_segment = build_segment_fn(**self._stateful_args())
        r6 = run_segment(self._make_stateful_carry(), 6, _FORCING)
        r3a = run_segment(self._make_stateful_carry(), 3, _FORCING)
        r3b = run_segment(r3a, 3, _FORCING)
        _assert_carries_match(r6, r3b, atol=1e-6, rtol=1e-6,
                              context="stateful segmented vs single")

    def test_compiled_matches_python_reference_stateful(self):
        """Compiled scan == per-step Python reference with the carry."""
        args = self._stateful_args()
        carry_init = self._make_stateful_carry()
        run_segment = build_segment_fn(**args)
        compiled_result = run_segment(_copy_carry(carry_init), 4, _FORCING)
        python_result = _run_per_step_python(
            args["model"], args["step_unified"], 4, carry_init, args,
        )
        _assert_carries_match(compiled_result, python_result,
                              atol=1e-5, rtol=1e-5,
                              context="stateful after 4 steps")


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


# ===========================================================================
# 7. Issue #316: radiation subcycling — cond-free outer/inner scan
# ===========================================================================

class TestRadiationSubcycle:
    """Issue #316 regression tests.

    The legacy single-scan body used ``jax.lax.cond`` to gate radiation
    inside the inner loop.  With a large RRTMGP branch and scan length
    >> 1, XLA JIT compile time grew to hours.  The fix replaces the
    cond with a Python-static outer/inner scan pair: outer body
    computes radiation once, inner body subcycles
    ``rad_update_steps - 1`` cheap "held radiation" steps before the
    final rad step.  These tests verify both the scheduling math
    (compute_segment_length, scan dispatch) and the
    behavioural equivalence to the legacy cond body.
    """

    def _make_init_carry(self):
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        return pack_carry(
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

    def test_segment_length_unchanged_when_already_multiple(self):
        # diag=4320 = 360 * 12, so rad=12 divides cleanly — no snap needed.
        assert compute_segment_length(4320, 0, 12) == 4320

    def test_segment_length_unchanged_when_rad_does_not_divide_gcd(self):
        # Snapping down would break the "segment_length divides every
        # cadence interval" invariant — leave it alone and let the
        # legacy cond-based scan handle the segment.
        assert compute_segment_length(4320, 0, 7) == 4320

    def test_segment_length_unchanged_when_rad_one(self):
        # rad_update_steps <= 1 means subcycling is impossible by
        # definition (no held-radiation steps to amortise over).
        assert compute_segment_length(4320, 0, 1) == 4320

    def test_segment_length_unchanged_when_seg_below_rad(self):
        # GCD is finer than radiation cadence — degenerate case, fall
        # through to legacy scan.
        assert compute_segment_length(100, 0, 200) == 100

    def test_subcycle_matches_legacy_cond_body(self):
        """Subcycled outer/inner scan output == legacy cond scan output.

        Mock physics is identical whether ``need_rad`` is True or
        False (the mock ignores rad), so the two paths must agree to
        machine precision.  This is the contract that guarantees the
        fix introduces no numerical change.
        """
        args = _make_segment_fn_args()
        args["rad_update_steps"] = 4

        # Legacy: data-dependent cond (no step_unified_no_rad passed)
        run_legacy = build_segment_fn(**args)

        # Subcycled: cond-elided outer/inner scan
        run_subcycle = build_segment_fn(
            **args, step_unified_no_rad=_mock_step_unified,
        )

        carry_init = self._make_init_carry()
        # 12 steps = 3 outer iters × 4 inner steps each — exact multiple.
        legacy_out = run_legacy(_copy_carry(carry_init), 12, _FORCING)
        subcycle_out = run_subcycle(_copy_carry(carry_init), 12, _FORCING)
        jax.block_until_ready(legacy_out.T)
        jax.block_until_ready(subcycle_out.T)

        for field_name in SegmentCarry._fields:
            if field_name in _OPTIONAL_CARRY_FIELDS:
                continue  # optional stateful/DM fields: None in legacy carries
            a = np.asarray(getattr(legacy_out, field_name))
            b = np.asarray(getattr(subcycle_out, field_name))
            np.testing.assert_allclose(
                a, b, atol=1e-6, rtol=1e-6,
                err_msg=f"Subcycled vs legacy mismatch in {field_name}",
            )

    def test_subcycle_falls_back_when_n_not_multiple_of_rad(self):
        """``n_steps % rad_update_steps != 0`` → legacy scan path.

        With ``step_unified_no_rad`` provided AND rad>1 AND n%rad==0,
        :func:`build_segment_fn` uses the subcycled scan.  When
        n%rad!=0, it must fall back to the legacy cond body so the
        radiation cadence within the segment stays correct.  We test
        by running 7 steps with rad=4: 7 is not a multiple of 4, so
        the subcycled-aware build still routes through the legacy
        body and produces the same result as the legacy build alone.
        """
        args = _make_segment_fn_args()
        args["rad_update_steps"] = 4

        run_legacy = build_segment_fn(**args)
        run_subcycle = build_segment_fn(
            **args, step_unified_no_rad=_mock_step_unified,
        )

        carry_init = self._make_init_carry()
        legacy_out = run_legacy(_copy_carry(carry_init), 7, _FORCING)
        subcycle_out = run_subcycle(_copy_carry(carry_init), 7, _FORCING)

        for field_name in SegmentCarry._fields:
            if field_name in _OPTIONAL_CARRY_FIELDS:
                continue  # optional stateful/DM fields: None in legacy carries
            np.testing.assert_allclose(
                np.asarray(getattr(legacy_out, field_name)),
                np.asarray(getattr(subcycle_out, field_name)),
                atol=1e-6, rtol=1e-6,
                err_msg=f"Fallback mismatch in {field_name}",
            )

    def test_subcycle_step_index_matches_legacy(self):
        """``step_index`` increments by ``n_steps`` regardless of path."""
        args = _make_segment_fn_args()
        args["rad_update_steps"] = 3

        run_subcycle = build_segment_fn(
            **args, step_unified_no_rad=_mock_step_unified,
        )
        carry = self._make_init_carry()
        result = run_subcycle(carry, 9, _FORCING)  # 3 outer × 3 inner
        assert int(result.step_index) == 9

    def test_subcycle_disabled_when_rad_one(self):
        """``rad_update_steps == 1`` → no subcycle, even with no_rad variant.

        Subcycling needs at least one held-rad step per outer iter
        (i.e. rad_update_steps - 1 >= 1).  When rad=1 build_segment_fn
        must route through the legacy single-scan path.
        """
        args = _make_segment_fn_args()
        args["rad_update_steps"] = 1

        run_subcycle = build_segment_fn(
            **args, step_unified_no_rad=_mock_step_unified,
        )
        run_legacy = build_segment_fn(**args)

        carry_init = self._make_init_carry()
        a = run_subcycle(_copy_carry(carry_init), 5, _FORCING)
        b = run_legacy(_copy_carry(carry_init), 5, _FORCING)
        for field_name in SegmentCarry._fields:
            if field_name in _OPTIONAL_CARRY_FIELDS:
                continue  # optional stateful/DM fields: None in legacy carries
            np.testing.assert_allclose(
                np.asarray(getattr(a, field_name)),
                np.asarray(getattr(b, field_name)),
                atol=1e-6, rtol=1e-6,
            )

    def test_build_step_unified_static_need_rad_api(self):
        """``static_need_rad`` API accepts True / False / None.

        The numerical equivalence between the static variants and the
        data-dependent ``lax.cond`` body is verified end-to-end by
        :meth:`test_subcycle_matches_legacy_cond_body` — at the
        segment level mock physics ignores ``need_rad``, so both
        scan paths must yield bit-equivalent state.  Here we just pin
        the API surface: the three call forms must produce distinct
        callables (so callers can build both variants up-front and
        dispatch Python-side).
        """
        from legoesm.driver.physics_pipeline import PhysicsPipeline
        import inspect
        sig = inspect.signature(PhysicsPipeline.build_step_unified)
        assert "static_need_rad" in sig.parameters
        param = sig.parameters["static_need_rad"]
        assert param.default is None  # default preserves legacy cond

    def test_unaligned_start_falls_back_to_legacy(self):
        """Codex review #1/#2/#3: alignment guard.

        Legacy cond body fires fresh radiation when ``(step_idx + 1)
        % rad_update_steps == 0`` — i.e., the last inner step of each
        radiation cycle.  The subcycled outer body's
        ``(k-1)`` no-rad + ``1`` rad pattern only matches that when
        the segment starts on an aligned step.  If
        ``carry.step_index % rad_update_steps != 0`` (checkpoint
        restart mid-cycle), the subcycled path would fire rad on the
        wrong absolute step — so :func:`build_segment_fn` must fall
        back to the legacy scan and produce identical output to the
        legacy-only build.
        """
        args = _make_segment_fn_args()
        args["rad_update_steps"] = 4

        run_legacy = build_segment_fn(**args)
        run_subcycle = build_segment_fn(
            **args, step_unified_no_rad=_mock_step_unified,
        )

        # Start mid-cycle: step_index=2 with rad=4.  Legacy would fire
        # rad on inner-step idx 1 (absolute idx 3 → (3+1)%4==0).  A
        # naive subcycled path would fire rad on inner-step idx 3.
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry_unaligned = pack_carry(
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
            step_index=2,  # NOT a multiple of rad_update_steps=4
        )

        out_legacy = run_legacy(_copy_carry(carry_unaligned), 8, _FORCING)
        out_subcycle = run_subcycle(_copy_carry(carry_unaligned), 8, _FORCING)
        jax.block_until_ready(out_legacy.T)
        jax.block_until_ready(out_subcycle.T)

        # When the alignment guard is correct, build_segment_fn falls
        # back to the legacy body and the two outputs are identical.
        for field_name in SegmentCarry._fields:
            if field_name in _OPTIONAL_CARRY_FIELDS:
                continue  # optional stateful/DM fields: None in legacy carries
            np.testing.assert_allclose(
                np.asarray(getattr(out_legacy, field_name)),
                np.asarray(getattr(out_subcycle, field_name)),
                atol=1e-6, rtol=1e-6,
                err_msg=(
                    f"Unaligned start: subcycle path silently used "
                    f"despite step_index=2, rad=4 — mismatch in "
                    f"{field_name}.  Codex iter alignment guard FAIL."
                ),
            )

    def test_need_rad_timing_recorded(self):
        """Codex review #9: timing-recording mock catches off-by-one.

        Use a stateful mock that records the ``need_rad`` value seen
        per scan call.  In the legacy cond body the value flows
        through ``lax.cond``; in the subcycled body the rad-only mock
        is called on the last inner step and the no-rad-only mock on
        the earlier ``k-1`` steps.  We can't easily inspect the
        in-trace bool, but we *can* compare the held-radiation arrays
        post-scan: the legacy path overwrites held with the rad
        output on the last step of every cycle.  If we make the
        rad-mock write a distinct sentinel into ``dT_dt_rad`` and the
        no-rad-mock leave it alone, the post-segment held value
        unambiguously identifies whether rad fired correctly.
        """
        sentinel_dT_rad = 1.234e-3

        def rad_mock(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                     sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                     solar_weights, s_0, o3_vmr, aerosol_od,
                     held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                     held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                     **kwargs):
            phys_out, _, _T_land_new = _mock_step_unified(
                need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                solar_weights, s_0, o3_vmr, aerosol_od,
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                **kwargs,
            )
            # New held with sentinel — represents "fresh radiation"
            held_new = (
                jnp.full_like(held_dT_rad, sentinel_dT_rad),
                held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
            )
            # step_unified now returns a 3-tuple incl. the slab-land
            # skin temperature (#325); pass it through unchanged.
            return phys_out, held_new, _T_land_new

        def no_rad_mock(*args, **kwargs):
            # Identical to _mock_step_unified — passes held through unchanged.
            return _mock_step_unified(*args, **kwargs)

        args = _make_segment_fn_args()
        args["step_unified"] = rad_mock
        args["rad_update_steps"] = 3

        # Aligned start: subcycle path must fire rad on inner-step
        # idx 2 (absolute idx (3*i)+2 → (idx+1)%3==0).
        run_subcycle = build_segment_fn(
            **args, step_unified_no_rad=no_rad_mock,
        )
        run_legacy = build_segment_fn(**args)

        carry = self._make_init_carry()  # step_index=0
        # 9 steps = 3 outer × 3 inner → rad fires on absolute steps 2, 5, 8.
        out_sub = run_subcycle(_copy_carry(carry), 9, _FORCING)
        out_leg = run_legacy(_copy_carry(carry), 9, _FORCING)

        # Both paths should end with held_dT_rad == sentinel — because
        # the rad mock fires (in subcycle: on inner-step 2 of every
        # outer iter; in legacy: when (idx+1)%3==0).
        np.testing.assert_allclose(
            np.asarray(out_sub.held_dT_rad), sentinel_dT_rad,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(out_leg.held_dT_rad), sentinel_dT_rad,
            atol=1e-9,
        )

    def test_subcycle_dispatches_no_rad_body_and_accepts_2tuple(self):
        """Regression for #325 codex review (two coupled hazards):

        1. The subcycled held-radiation steps must dispatch to the
           ``step_unified_no_rad`` body via ``step_fn``.  A prior
           refactor called the outer ``step_unified`` directly, so the
           expensive radiation branch (and the slab-land update) ran on
           every step regardless of ``rad_update_steps``.
        2. A ``step_unified`` implementation may return a legacy 2-tuple
           ``(phys_out, held_new)`` — neural / SFNO training wrappers do
           — and the compiled segment path must accept it (land inert),
           not crash unpacking a 3rd value.

        Both mocks tag ``held_dT_rad`` with a distinct sentinel; running a
        single step with ``rad_update_steps=2`` (step_idx 0 ->
        ``(0+1)%2 != 0`` -> no-rad) must leave the *no-rad* sentinel.
        """
        # ``_run_subcycled`` runs ``rad_update_steps - 1`` no-rad steps
        # then one rad step per outer cycle, and is only used when
        # ``n_steps`` divides evenly by the cadence.  Cadence 2 over 2
        # steps => one outer cycle = 1 no-rad step + 1 rad step.  Both
        # bodies are traced; the no-rad body sets a (trace-time) flag, so
        # if the dispatch is wrong (outer ``step_unified`` used for the
        # no-rad slot) the flag stays False.
        called = {"no_rad": False}

        def rad_mock(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                     sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                     solar_weights, s_0, o3_vmr, aerosol_od,
                     held_dT_rad, *held_rest, **kwargs):
            phys_out, held, _T_land = _mock_step_unified(
                need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                solar_weights, s_0, o3_vmr, aerosol_od,
                held_dT_rad, *held_rest, **kwargs)
            return phys_out, held, _T_land  # 3-tuple (PhysicsPipeline form)

        def norad_mock_2tuple(need_rad, T, p_s, q_v, q_c, q_r, conv_prog,
                              u, v, sst, sic, lat, lon, day_of_year,
                              seconds_of_day, dt, solar_weights, s_0,
                              o3_vmr, aerosol_od, held_dT_rad, *held_rest,
                              **kwargs):
            called["no_rad"] = True  # set at trace time when dispatched
            phys_out, held, _ = _mock_step_unified(
                need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                solar_weights, s_0, o3_vmr, aerosol_od,
                held_dT_rad, *held_rest, **kwargs)
            return phys_out, held  # LEGACY 2-tuple (neural / SFNO form)

        args = _make_segment_fn_args()
        args["step_unified"] = rad_mock
        args["rad_update_steps"] = 2
        run = build_segment_fn(**args, step_unified_no_rad=norad_mock_2tuple)

        carry = self._make_init_carry()  # step_index=0
        out = run(_copy_carry(carry), 2, _FORCING)  # 1 outer cycle

        # (1) The no-rad body must have been dispatched (and not crash on
        #     its 2-tuple return).
        assert called["no_rad"], (
            "subcycle held-step did not dispatch to step_unified_no_rad — "
            "outer step_unified used instead (#325 dispatch regression)"
        )
        # (2) The run completes with finite prognostics (2-tuple accepted).
        assert bool(jnp.all(jnp.isfinite(out.T)))
        assert bool(jnp.all(jnp.isfinite(out.held_dT_rad)))

    def test_raw_remains_traceable_under_grad(self):
        """Codex review axis D: ``.raw`` must work under ``jax.grad``.

        The JIT path is fine to do a host sync on ``carry.step_index``
        (one scalar transfer per Python segment, outside the hot
        loop).  The ``.raw`` path is used inside
        ``eqx.filter_value_and_grad`` / ``jax.grad`` — where
        ``carry.step_index`` becomes a JAX tracer and ``int(tracer)``
        raises ``TracerIntegerConversionError``.  We assert that
        ``run_segment.raw`` traces cleanly under ``jax.grad`` (i.e.,
        the dispatcher does not inspect ``step_index``).
        """
        args = _make_segment_fn_args()
        args["rad_update_steps"] = 4
        run_segment_jit = build_segment_fn(
            **args, step_unified_no_rad=_mock_step_unified,
        )
        run_raw = run_segment_jit.raw

        carry_init = self._make_init_carry()

        def loss_fn(T0):
            # Replace carry.T with traced T0 so the gradient flows back.
            c = carry_init._replace(T=T0)
            out = run_raw(c, 4, _FORCING)
            return jnp.sum(out.T)

        T0 = carry_init.T.astype(jnp.float32)
        # If .raw secretly calls int(carry.step_index), jax.grad will
        # raise on the inner Tracer.  We don't care about the gradient
        # value here — only that tracing succeeds.
        grad_T = jax.grad(loss_fn)(T0)
        assert grad_T.shape == T0.shape

    def test_compiled_segments_subcycle_dispatch_matches_compute_segment_length(self):
        """End-to-end: typical AMIP cadence routes through subcycled path.

        Issue #316 worked example: dt=20 s, diag_days=1 →
        diag_interval=4320.  At rad_update_steps=12 (4-min radiation
        cadence) compute_segment_length keeps the segment at 4320
        (clean multiple of 12) and build_segment_fn's run_segment
        must dispatch to the subcycled path — i.e. produce the same
        output as the legacy path on this exact (n_steps,
        rad_update_steps) pair.  We don't measure HLO size here (CPU
        backend optimises both paths fine); instead we lock in the
        invariant that the subcycled and legacy paths agree.
        """
        seg = compute_segment_length(4320, 0, 12)
        assert seg == 4320
        assert seg % 12 == 0

        args = _make_segment_fn_args()
        args["rad_update_steps"] = 12

        # Use a 24-step segment (still a multiple of rad=12) so the
        # test runs in a few seconds on CPU — the full 4320 would be
        # right for GPU compile-time benchmarking but overkill here.
        n_steps = 24

        run_legacy = build_segment_fn(**args)
        run_subcycle = build_segment_fn(
            **args, step_unified_no_rad=_mock_step_unified,
        )

        carry_init = self._make_init_carry()
        out_legacy = run_legacy(_copy_carry(carry_init), n_steps, _FORCING)
        out_subcycle = run_subcycle(_copy_carry(carry_init), n_steps, _FORCING)
        for field_name in SegmentCarry._fields:
            if field_name in _OPTIONAL_CARRY_FIELDS:
                continue  # optional stateful/DM fields: None in legacy carries
            np.testing.assert_allclose(
                np.asarray(getattr(out_legacy, field_name)),
                np.asarray(getattr(out_subcycle, field_name)),
                atol=1e-5, rtol=1e-5,
                err_msg=(
                    f"Subcycled vs legacy AMIP-cadence mismatch in "
                    f"{field_name} (rad_update_steps=12, n_steps={n_steps})"
                ),
            )


# ---------------------------------------------------------------------------
# Double-moment carry (q_i / N_c / N_i) — radiation r_eff coupling
# ---------------------------------------------------------------------------


class TestDoubleMomentCarry:
    """The optional q_i/N_c/N_i carry fields thread through build_segment_fn:
    populated from a double-moment microphysics, they are (a) passed to the
    per-step physics by keyword (so radiation gets number-aware r_eff) and
    (b) evolved each step from PhysicsOutput.dq_i/dN_c/dN_i — while warm-rain
    runs (fields None) are byte-unchanged (covered by TestEquivalence)."""

    def _dm_mock(self, seen):
        """step_unified mock recording the double-moment kwargs it receives and
        emitting nonzero hydrometeor/number tendencies so the carry evolves."""
        def _step(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                  sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                  solar_weights, s_0, o3_vmr, aerosol_od,
                  held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                  held_sw_up_toa, held_lw_up_toa, held_sw_down_toa, **kwargs):
            for _k in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                seen[_k] = kwargs.get(_k)
            s3, s2 = T.shape, p_s.shape
            phys_out = PhysicsOutput(
                dT_dt=jnp.zeros(s3), dq_v_dt=jnp.zeros(s3),
                dq_c_dt=jnp.zeros(s3), dq_r_dt=jnp.zeros(s3),
                precip=jnp.zeros(s2), sw_net_sfc=jnp.zeros(s2),
                lw_net_sfc=jnp.zeros(s2), sw_up_toa=jnp.zeros(s2),
                lw_up_toa=jnp.zeros(s2), sw_down_toa=jnp.zeros(s2),
                du_dt=jnp.zeros(s3), dv_dt=jnp.zeros(s3),
                dq_i_dt=jnp.full(s3, 1.0e-7),
                dq_s_dt=jnp.full(s3, 2.0e-7),
                dq_g_dt=jnp.full(s3, 3.0e-7),
                dN_c_dt=jnp.full(s3, 2.0),
                dN_r_dt=jnp.full(s3, 4.0),
                dN_i_dt=jnp.full(s3, 3.0),
                conv_prog=conv_prog,
            )
            held_new = (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                        held_sw_up_toa, held_lw_up_toa, held_sw_down_toa)
            return phys_out, held_new, kwargs.get("T_land")
        return _step

    def test_carry_evolves_and_passes_number_to_physics(self):
        seen = {}
        args = _make_segment_fn_args()
        args["step_unified"] = self._dm_mock(seen)
        run_segment = build_segment_fn(**args)

        state = _make_hydrostatic_state()
        s3 = (N_FACES, N, N, NLEV)
        s2 = (N_FACES, N, N)
        nc0, nr0, ni0 = 1.0e8, 1.0e6, 5.0e3
        qi0, qs0, qg0 = 1.0e-4, 2.0e-4, 3.0e-4
        carry = pack_carry(
            state, q_v=jnp.ones(s3) * 0.01, q_c=jnp.ones(s3) * 1e-3,
            q_r=jnp.zeros(s3),
            held_dT_rad=jnp.zeros(s3), held_sw_net_sfc=jnp.zeros(s2),
            held_lw_net_sfc=jnp.zeros(s2), held_sw_up_toa=jnp.zeros(s2),
            held_lw_up_toa=jnp.zeros(s2), held_sw_down_toa=jnp.zeros(s2),
            step_index=0,
            q_i=jnp.full(s3, qi0), q_s=jnp.full(s3, qs0), q_g=jnp.full(s3, qg0),
            N_c=jnp.full(s3, nc0), N_r=jnp.full(s3, nr0), N_i=jnp.full(s3, ni0),
        )
        # The double-moment fields are real arrays in the packed carry.
        assert carry.N_c is not None and carry.q_i is not None

        n_steps = 2
        result = run_segment(carry, n_steps, _FORCING)
        jax.block_until_ready(result.N_c)

        # (a) physics received ALL hydrometeor/number columns by keyword.
        for _k in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
            assert seen[_k] is not None, _k
        # (b) every field evolved by n_steps * dt * its tendency (clipped >= 0).
        nd = n_steps * DT
        np.testing.assert_allclose(np.asarray(result.q_i), qi0 + nd * 1.0e-7, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(result.q_s), qs0 + nd * 2.0e-7, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(result.q_g), qg0 + nd * 3.0e-7, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(result.N_c), nc0 + nd * 2.0, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(result.N_r), nr0 + nd * 4.0, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(result.N_i), ni0 + nd * 3.0, rtol=1e-5)

    def test_warm_rain_carry_keeps_number_fields_none(self):
        """Without q_i/N_c/N_i in pack_carry, the carry fields stay None
        (legacy warm-rain behaviour, no extra leaves)."""
        state = _make_hydrostatic_state()
        s3 = (N_FACES, N, N, NLEV)
        s2 = (N_FACES, N, N)
        carry = pack_carry(
            state, q_v=jnp.ones(s3) * 0.01, q_c=jnp.zeros(s3),
            q_r=jnp.zeros(s3),
            held_dT_rad=jnp.zeros(s3), held_sw_net_sfc=jnp.zeros(s2),
            held_lw_net_sfc=jnp.zeros(s2), held_sw_up_toa=jnp.zeros(s2),
            held_lw_up_toa=jnp.zeros(s2), held_sw_down_toa=jnp.zeros(s2),
            step_index=0,
        )
        assert all(getattr(carry, k) is None for k in
                   ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"))
        # None fields contribute no pytree leaves.
        leaves = jax.tree.leaves(carry)
        assert all(isinstance(x, jax.Array) for x in leaves)


class TestQvSmoothingGate:
    """Direct leaf tests for the fix-5 moisture-smoothing static gate
    (``_apply_qv_smoothing``), the only operator-split-tail hot-loop change in
    PR #798 (#797). The gate skips the cube-only ∇⁴ operator when
    ``qv_smooth_coeff == 0`` so the lat-lon training rollout does not crash on
    ``grid.halo_interp_offsets``, while the nonzero path stays bit-identical to
    the former nested ``max(q + dt·∇⁴, 0)``."""

    def test_zero_coeff_skips_operator_and_only_floors(self):
        from types import SimpleNamespace

        from legoesm.driver.compiled_segments import _apply_qv_smoothing

        def _boom(*args, **kwargs):
            raise AssertionError(
                "hyperdiffusion_3d must NOT be called when qv_smooth_coeff == 0 "
                "(that call is what crashes lat-lon on grid.halo_interp_offsets)")

        q = jnp.array([[-1.0, 2.0, 0.0, 0.5]])
        # grid deliberately has NO halo_interp_offsets — mimics a lat-lon grid.
        statics = SimpleNamespace(
            qv_smooth_coeff=0.0, dt=100.0, grid=object(), hyperdiffusion_3d=_boom)
        out = _apply_qv_smoothing(q, statics)
        # operator skipped; only the positivity floor applied.
        assert jnp.array_equal(out, jnp.maximum(q, 0.0))

    def test_nonzero_coeff_is_bit_identical_to_former_nested_max(self):
        from types import SimpleNamespace

        from legoesm.driver.compiled_segments import _apply_qv_smoothing

        seen = {}

        def _hd(field, grid, coeff):
            seen["coeff"] = coeff
            seen["grid"] = grid
            return jnp.full_like(field, 1.0e-3)  # arbitrary ∇⁴ tendency

        q = jnp.array([[0.01, -0.002, 0.5, 3.0e-4]])
        dt, coeff, grid = 90.0, 0.25, object()
        statics = SimpleNamespace(
            qv_smooth_coeff=coeff, dt=dt, grid=grid, hyperdiffusion_3d=_hd)
        out = _apply_qv_smoothing(q, statics)
        # Former inline form: max(q + dt*hyperdiffusion_3d(q, grid, coeff), 0).
        expected = jnp.maximum(q + dt * jnp.full_like(q, 1.0e-3), 0.0)
        assert jnp.array_equal(out, expected)
        # The real operator was invoked with the configured coeff + grid.
        assert seen["coeff"] == coeff and seen["grid"] is grid


class TestNestedCheckpointedScan:
    """#841: nested (Griewank / sqrt-N) checkpointing of the rollout scan is a
    reverse-mode MEMORY SCHEDULE only — the forward is bit-identical and the
    gradient is EXACT vs a plain ``lax.scan``.  This is what lets the WB scale
    trainer's ~6000-step 0.7-deg rollout fit (O(sqrt N) carries instead of the
    O(N) ~1 TB trajectory that OOMs)."""

    @staticmethod
    def _step(c, _):
        # A coupled nonlinear recurrence so the adjoint is sensitive to every
        # step (a decoupled map could pass even with a dropped-step bug).
        x = jnp.tanh(1.03 * c + 0.1) + 0.02 * jnp.roll(c, 1)
        return x, None

    @pytest.mark.parametrize("n_steps", [289, 300, _NESTED_CKPT_STEPS])
    def test_forward_and_gradient_match_plain_scan(self, n_steps):
        x0 = jnp.linspace(-1.0, 1.0, 8)

        def plain(x):
            c, _ = jax.lax.scan(self._step, x, None, length=n_steps)
            return jnp.sum(c ** 2)

        def nested(x):
            c = _sqrt_checkpointed_scan(self._step, x, n_steps)
            return jnp.sum(c ** 2)

        # Forward: same op sequence (chunk boundaries do not reorder steps).
        np.testing.assert_allclose(
            float(nested(x0)), float(plain(x0)), rtol=1e-6,
            err_msg="nested checkpointed scan changed the forward value")
        # Gradient: EXACT — checkpointing recomputes the same math.  A broken
        # chunking (dropped/duplicated steps, wrong boundary carry) gives an
        # O(1) relative error, far above this tolerance.
        g_nested = jax.grad(nested)(x0)
        g_plain = jax.grad(plain)(x0)
        np.testing.assert_allclose(
            np.asarray(g_nested), np.asarray(g_plain), rtol=1e-5, atol=1e-8,
            err_msg="nested checkpointed scan gradient != plain scan gradient")

    def test_remainder_steps_are_not_dropped(self):
        """A step count that is NOT a multiple of the chunk size must still run
        EXACTLY n_steps (the trailing remainder scan) — a count regression would
        change the forward."""
        # 290 = 17*17 + 1: inner=17, n_outer=17 (289 steps) + 1 remainder.
        n_steps = 290
        x0 = jnp.linspace(0.0, 1.0, 6)
        ref, _ = jax.lax.scan(self._step, x0, None, length=n_steps)
        got = _sqrt_checkpointed_scan(self._step, x0, n_steps)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6,
                                   err_msg="remainder steps dropped/miscounted")


class TestTiledStepFnRouting:
    """P4 increment 1b: ``tiled_step_fn`` replaces ONLY the dynamics core
    of the scan body — same cc HydrostaticState contract; everything else
    (physics mock, fixers, carry plumbing) untouched."""

    def _run(self, tiled_step_fn, explicit_none=False, q_v_fill=0.01):
        args = _make_segment_fn_args()
        if tiled_step_fn is not None or explicit_none:
            args["tiled_step_fn"] = tiled_step_fn
        run_segment = build_segment_fn(**args)
        state = _make_hydrostatic_state()
        shape_3d = (N_FACES, N, N, NLEV)
        shape_2d = (N_FACES, N, N)
        carry = pack_carry(
            state,
            q_v=jnp.ones(shape_3d) * q_v_fill,
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
        out = run_segment(carry, 2, _FORCING)
        jax.block_until_ready(out.T)
        return out

    def test_tiled_step_fn_routes_dynamics(self):
        """A marker tiled step (T += 7 K/step) must drive the trajectory
        instead of the mock model's +dt/86400 K/step increment."""
        def _marker_step(state):
            return state._replace(
                T=state.T.replace(data=state.T.data + 7.0))

        # DRY column (q_v=0): microphysics="none" activates the segment's
        # T-DEPENDENT saturation adjustment (do_sat_adjust), which at
        # q_v=0.01 releases ~9 K more latent heat on the cooler base
        # trajectory than on the +7 K/step marker one at the lowest level
        # (measured: level sigma=1.0 delta 4.73 vs 13.99) — zero vapor
        # makes it inert so the dynamics delta is exactly pinnable.
        out = self._run(_marker_step, q_v_fill=0.0)
        base = self._run(None, q_v_fill=0.0)
        # ONLY the dynamics core differs: tiled = 2*7 K, mock dynamics =
        # 2*DT/86400 K, physics identical in both branches.  Pinning the
        # elementwise difference to that exact value catches (a) the mock
        # dynamics still running in the tiled branch and (b) the physics
        # increment (2 * 1e-5 K/s * DT) being dropped from either branch —
        # a >threshold check alone would not (codex round-14 Low).
        expected = 14.0 - 2.0 * DT / 86400.0
        np.testing.assert_allclose(
            np.asarray(out.T - base.T), expected, atol=1e-3)

    def test_default_none_is_untouched(self):
        """Passing tiled_step_fn=None EXPLICITLY is byte-identical to
        omitting the kwarg (the legacy body)."""
        a = self._run(None)
        b = self._run(None, explicit_none=True)
        assert float(jnp.max(jnp.abs(a.T - b.T))) == 0.0
        assert float(jnp.max(jnp.abs(a.p_s - b.p_s))) == 0.0
