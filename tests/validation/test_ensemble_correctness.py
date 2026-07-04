"""Ensemble correctness: vmap over ensemble members == independent runs.

Validates that ``jax.vmap`` over a batched ``SegmentCarry`` produces
bit-identical results to running each ensemble member independently
through the same compiled segment function.  This is a critical
invariant for differentiable ensemble data assimilation and ensemble
forecasting workflows.

Setup
-----
- C4/L3 cubed-sphere grid (fast).
- 2 ensemble members with different initial temperature (280 K, 281 K).
- 5-step integration using the compiled segment infrastructure.
- Mock dynamics (constant warming) and mock physics (small tendency).

The test compares the vmapped path against looped independent runs
and requires exact (atol=0) agreement, since both code paths execute
identical XLA operations on identical inputs.
"""

from __future__ import annotations

import os

# Ensure float64 is available for scientific correctness.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import (
    SegmentCarry,
    SegmentForcing,
    build_segment_fn,
    pack_carry,
    pack_forcing,
)
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm import constants


# ---------------------------------------------------------------------------
# Grid and shape constants
# ---------------------------------------------------------------------------

N_FACES = 6
N = 4          # C4 resolution
NLEV = 3       # 3 vertical levels
DT = 600.0     # 10-minute timestep
N_STEPS = 5    # number of integration steps
N_ENSEMBLE = 2 # number of ensemble members

SHAPE_3D = (N_FACES, N, N, NLEV)
SHAPE_2D = (N_FACES, N, N)
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

# Build the cubed-sphere grid once (expensive to recreate per test).
_GRID = create_cubed_sphere(N)


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


# ---------------------------------------------------------------------------
# Mock model and physics (replicates test_compiled_segments.py patterns)
# ---------------------------------------------------------------------------

class _MockModel:
    """Minimal dynamics model: adds increment * dt / 86400 to T each step.

    Does not modify u, v, p_s, or phis, so the only prognostic change
    from dynamics is a controlled temperature increment.
    """

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
    """Mock physics: small constant warming tendency, no moisture change."""
    shape_3d = T.shape
    kwargs = _zero_physics_output(T, p_s)
    kwargs["dT_dt"] = jnp.full(shape_3d, 1e-5)
    # Pass the prognostic convective state through unchanged so the scan
    # carry stays shape/type-consistent (production carries
    # ``phys_out.conv_prog`` straight into the next step). A scalar zero
    # would break the ``(ncol,)`` carry contract.
    if "conv_prog" in PhysicsOutput._fields:
        kwargs["conv_prog"] = conv_prog
    phys_out = PhysicsOutput(**kwargs)
    held_new = (
        held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
        held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
    )
    return phys_out, held_new


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forcing() -> SegmentForcing:
    """Build a default SegmentForcing for tests."""
    return pack_forcing(
        sst=jnp.full(SHAPE_2D, 300.0),
        sic=jnp.zeros(SHAPE_2D),
        day_of_year=1.0,
        seconds_of_day=0.0,
        solar_weights=jnp.ones(14),
        s_0=constants.S_0,
        o3_vmr=jnp.zeros(SHAPE_3D),
        aerosol_od=jnp.zeros(SHAPE_2D),
    )


def _make_segment_fn_args() -> dict:
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
        fix_moisture=False,
                    fix_mass=False,
        fric_decay=jnp.ones((NLEV,)),
        qv_smooth_coeff=0.0,
        lat=_GRID.lat,
        lon=_GRID.lon,
        start_day=0.0,
    )


def _make_carry(T_fill: float) -> SegmentCarry:
    """Build a SegmentCarry with the given uniform temperature."""
    state = HydrostaticState(
        u=Field(jnp.full(SHAPE_3D, 0.5), name="u",
                dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(jnp.full(SHAPE_3D, -0.5), name="v",
                dims=("face", "x", "y", "level"), units="m/s"),
        T=Field(jnp.full(SHAPE_3D, T_fill), name="T",
                dims=("face", "x", "y", "level"), units="K"),
        p_s=Field(jnp.full(SHAPE_2D, 101325.0), name="p_s",
                  dims=("face", "x", "y"), units="Pa"),
        phis=Field(jnp.zeros(SHAPE_2D), name="phis",
                   dims=("face", "x", "y"), units="m2/s2"),
    )
    return pack_carry(
        state,
        q_v=jnp.ones(SHAPE_3D) * 0.01,
        q_c=jnp.zeros(SHAPE_3D),
        q_r=jnp.zeros(SHAPE_3D),
        held_dT_rad=jnp.zeros(SHAPE_3D),
        held_sw_net_sfc=jnp.zeros(SHAPE_2D),
        held_lw_net_sfc=jnp.zeros(SHAPE_2D),
        held_sw_up_toa=jnp.zeros(SHAPE_2D),
        held_lw_up_toa=jnp.zeros(SHAPE_2D),
        held_sw_down_toa=jnp.zeros(SHAPE_2D),
        step_index=0,
    )


def _stack_carries(*carries: SegmentCarry) -> SegmentCarry:
    """Stack N SegmentCarry instances into a single batched carry.

    Each field gets a leading ensemble dimension via jnp.stack.
    """
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *carries)


def _copy_carry(carry: SegmentCarry) -> SegmentCarry:
    """Deep-copy a SegmentCarry so the original survives buffer donation."""
    return jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry)


def _extract_member(batched_carry: SegmentCarry, idx: int) -> SegmentCarry:
    """Extract ensemble member ``idx`` from a batched carry."""
    return jax.tree.map(lambda x: x[idx], batched_carry)


# ---------------------------------------------------------------------------
# Build a non-JIT scan function suitable for jax.vmap
# ---------------------------------------------------------------------------

def _build_raw_segment_fn(args: dict):
    """Build a raw (non-JIT, non-donating) scan-based segment function.

    This is identical to the kernel produced by ``build_segment_fn`` but
    without the ``@jax.jit`` wrapper and ``donate_argnums``, so that
    ``jax.vmap`` can trace through it cleanly.

    We also build the standard JIT version via ``build_segment_fn`` for
    the independent-run path, ensuring both code paths use the same
    underlying scan body.
    """
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm import constants
    from legoesm.core.operators_3d import hyperdiffusion_3d
    from legoesm.core.conservation import fix_ps_mass_target
    from legoesm.core.cfl import cfl_number_from_state, estimate_min_dx_cubed_sphere
    from legoesm.driver.compiled_segments import _rebuild_state

    model = args["model"]
    step_unified = args["step_unified"]
    grid = args["grid"]
    sigma_full = jnp.asarray(args["sigma_full"])
    dt = args["dt"]
    rad_update_steps = args["rad_update_steps"]
    microphysics = args["microphysics"]
    fric_decay = jnp.asarray(args["fric_decay"])
    qv_smooth_coeff = args["qv_smooth_coeff"]
    lat = args["lat"]
    lon = args["lon"]
    do_sat_adjust = (microphysics == "none")
    _dt = jnp.asarray(dt)
    _fric_decay = jnp.asarray(fric_decay)

    # Precompute minimum grid spacing for CFL monitoring (mirrors build_segment_fn)
    if hasattr(grid, "n"):
        _dx_min = jnp.asarray(estimate_min_dx_cubed_sphere(grid.n))
    else:
        _dx_min = jnp.asarray(1e6)

    def run_segment_raw(carry: SegmentCarry, n_steps: int,
                        forcing: SegmentForcing) -> SegmentCarry:
        """Run n_steps via jax.lax.scan without JIT or buffer donation."""

        def _single_step(carry: SegmentCarry, _unused):
            step_idx = carry.step_index

            # Dynamics
            dyn_state = model.step(_rebuild_state(carry, model), _dt)
            T_new = dyn_state.T.data
            u_new = dyn_state.u.data
            v_new = dyn_state.v.data
            p_s_new = dyn_state.p_s.data

            # Dry mass fixer (target-anchored, only active when target_mass > 0)
            p_s_new = jnp.where(
                carry.target_mass > 0.0,
                fix_ps_mass_target(p_s_new, carry.target_mass, grid),
                p_s_new,
            )

            # Physics
            need_rad = jnp.where(
                rad_update_steps <= 1,
                jnp.bool_(True),
                ((step_idx + 1) % rad_update_steps) == 0,
            )
            phys_out, held_new = step_unified(
                need_rad,
                T_new, p_s_new,
                carry.q_v, carry.q_c, carry.q_r, carry.conv_prog,
                u_new, v_new,
                forcing.sst, forcing.sic, lat, lon,
                forcing.day_of_year, forcing.seconds_of_day, _dt,
                forcing.solar_weights, forcing.s_0,
                forcing.o3_vmr, forcing.aerosol_od,
                carry.held_dT_rad, carry.held_sw_net_sfc, carry.held_lw_net_sfc,
                carry.held_sw_up_toa, carry.held_lw_up_toa, carry.held_sw_down_toa,
            )

            # State update
            T_upd = T_new + _dt * phys_out.dT_dt
            q_v_upd = jnp.maximum(carry.q_v + _dt * phys_out.dq_v_dt, 0.0)
            q_c_upd = jnp.maximum(carry.q_c + _dt * phys_out.dq_c_dt, 0.0)
            q_r_upd = jnp.maximum(carry.q_r + _dt * phys_out.dq_r_dt, 0.0)

            # Saturation adjustment
            if do_sat_adjust:
                p_full = p_s_new[..., None] * sigma_full
                q_sat = saturation_mixing_ratio(T_upd, p_full)
                excess = jnp.maximum(q_v_upd - q_sat, 0.0)
                q_v_upd = q_v_upd - excess
                T_upd = T_upd + constants.L_v * excess / constants.c_pd

            # Moisture smoothing
            q_v_upd = jnp.maximum(
                q_v_upd + _dt * hyperdiffusion_3d(q_v_upd, grid, qv_smooth_coeff),
                0.0,
            )

            # Rayleigh friction
            u_upd = u_new * _fric_decay
            v_upd = v_new * _fric_decay

            # CFL computed at segment boundary (host-side), not per step
            max_cfl = carry.max_cfl

            # Precipitation accumulation
            precip_step = (phys_out.precip
                           if hasattr(phys_out, "precip")
                           else jnp.zeros_like(p_s_new))
            precip_accum = carry.precip_accum + precip_step * _dt

            # Match dtypes to prevent promotion issues in scan
            def _match_dtype(new_val, ref_val):
                if hasattr(ref_val, "dtype") and hasattr(new_val, "dtype"):
                    if new_val.dtype != ref_val.dtype:
                        return new_val.astype(ref_val.dtype)
                return new_val

            # Use _replace to update only changed fields; this is robust
            # against future additions to SegmentCarry.
            new_carry = carry._replace(
                u=_match_dtype(u_upd, carry.u),
                v=_match_dtype(v_upd, carry.v),
                T=_match_dtype(T_upd, carry.T),
                p_s=_match_dtype(p_s_new, carry.p_s),
                q_v=_match_dtype(q_v_upd, carry.q_v),
                q_c=_match_dtype(q_c_upd, carry.q_c),
                q_r=_match_dtype(q_r_upd, carry.q_r),
                held_dT_rad=_match_dtype(held_new[0], carry.held_dT_rad),
                held_sw_net_sfc=_match_dtype(held_new[1], carry.held_sw_net_sfc),
                held_lw_net_sfc=_match_dtype(held_new[2], carry.held_lw_net_sfc),
                held_sw_up_toa=_match_dtype(held_new[3], carry.held_sw_up_toa),
                held_lw_up_toa=_match_dtype(held_new[4], carry.held_lw_up_toa),
                held_sw_down_toa=_match_dtype(held_new[5], carry.held_sw_down_toa),
                step_index=step_idx + 1,
                max_cfl=max_cfl,
                precip_accum=_match_dtype(precip_accum, carry.precip_accum),
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(_single_step, carry, None, length=n_steps)
        return final_carry

    return run_segment_raw


# ===========================================================================
# Tests
# ===========================================================================

class TestEnsembleCorrectness:
    """Validate that vmapping over ensemble members produces bit-identical
    results to running each member independently.

    This is the key invariant for differentiable ensemble workflows:
    batched execution must not introduce numerical differences compared
    to serial member-by-member execution.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Build segment functions and ensemble carries once per test."""
        self.args = _make_segment_fn_args()
        self.forcing = _make_forcing()

        # Two ensemble members: different initial T
        self.carry_0 = _make_carry(T_fill=280.0)
        self.carry_1 = _make_carry(T_fill=281.0)

        # Batched carry: leading dim = ensemble
        self.batched_carry = _stack_carries(self.carry_0, self.carry_1)

        # Build the JIT-compiled segment fn for independent runs
        self.run_segment = build_segment_fn(**self.args)

        # Build the raw (non-JIT) segment fn for vmapping
        self.run_segment_raw = _build_raw_segment_fn(self.args)

    def test_vmap_matches_independent_runs(self):
        """Vmapped ensemble execution is bit-identical to independent runs.

        Runs 5 steps for 2 ensemble members via:
        1. jax.vmap(run_segment_raw)(batched_carry, n_steps, forcing)
        2. run_segment(carry_0, n_steps, forcing) and
           run_segment(carry_1, n_steps, forcing) independently.

        Compares all prognostic SegmentCarry fields with atol=0 (exact
        match).  Diagnostic accumulators (max_cfl) that involve
        floating-point reductions are compared with a tight but
        non-zero tolerance since operation ordering may differ.
        """
        # Diagnostic accumulators where ULP-level differences from
        # reduction ordering are expected and harmless.
        _DIAGNOSTIC_FIELDS = {"max_cfl"}

        # --- Vmapped path ---
        vmapped_fn = jax.vmap(
            self.run_segment_raw,
            in_axes=(0, None, None),
        )
        vmapped_result = vmapped_fn(self.batched_carry, N_STEPS, self.forcing)
        jax.block_until_ready(vmapped_result)

        # --- Independent paths ---
        indep_result_0 = self.run_segment(
            _copy_carry(self.carry_0), N_STEPS, self.forcing
        )
        indep_result_1 = self.run_segment(
            _copy_carry(self.carry_1), N_STEPS, self.forcing
        )
        jax.block_until_ready(indep_result_0)
        jax.block_until_ready(indep_result_1)

        # --- Compare member 0 ---
        vmap_member_0 = _extract_member(vmapped_result, 0)
        for field_name in SegmentCarry._fields:
            vmap_val = getattr(vmap_member_0, field_name)
            indep_val = getattr(indep_result_0, field_name)
            # Optional carry slots (land tile, extra microphysics species)
            # are None on both paths under this mock setup — nothing to
            # compare numerically. Assert structural agreement, then skip.
            if vmap_val is None or indep_val is None:
                assert vmap_val is None and indep_val is None, field_name
                continue
            vmap_arr = np.asarray(vmap_val)
            indep_arr = np.asarray(indep_val)
            if field_name in _DIAGNOSTIC_FIELDS:
                np.testing.assert_allclose(
                    vmap_arr, indep_arr, atol=1e-14, rtol=1e-14,
                    err_msg=(
                        f"Member 0, diagnostic '{field_name}': "
                        f"vmap result differs from independent run"
                    ),
                )
            else:
                np.testing.assert_allclose(
                    vmap_arr, indep_arr, atol=0, rtol=0,
                    err_msg=(
                        f"Member 0, field '{field_name}': "
                        f"vmap result differs from independent run"
                    ),
                )

        # --- Compare member 1 ---
        vmap_member_1 = _extract_member(vmapped_result, 1)
        for field_name in SegmentCarry._fields:
            vmap_val = getattr(vmap_member_1, field_name)
            indep_val = getattr(indep_result_1, field_name)
            # Optional carry slots are None on both paths — skip (see above).
            if vmap_val is None or indep_val is None:
                assert vmap_val is None and indep_val is None, field_name
                continue
            vmap_arr = np.asarray(vmap_val)
            indep_arr = np.asarray(indep_val)
            if field_name in _DIAGNOSTIC_FIELDS:
                np.testing.assert_allclose(
                    vmap_arr, indep_arr, atol=1e-14, rtol=1e-14,
                    err_msg=(
                        f"Member 1, diagnostic '{field_name}': "
                        f"vmap result differs from independent run"
                    ),
                )
            else:
                np.testing.assert_allclose(
                    vmap_arr, indep_arr, atol=0, rtol=0,
                    err_msg=(
                        f"Member 1, field '{field_name}': "
                        f"vmap result differs from independent run"
                    ),
                )

    def test_ensemble_members_differ(self):
        """Sanity check: the two ensemble members produce different results.

        If both members returned identical output, the test above would
        pass vacuously.  This confirms the 1 K initial T difference
        propagates through integration.
        """
        vmapped_fn = jax.vmap(
            self.run_segment_raw,
            in_axes=(0, None, None),
        )
        vmapped_result = vmapped_fn(self.batched_carry, N_STEPS, self.forcing)
        jax.block_until_ready(vmapped_result)

        T_member_0 = np.asarray(_extract_member(vmapped_result, 0).T)
        T_member_1 = np.asarray(_extract_member(vmapped_result, 1).T)

        # The 1 K initial difference should persist (mock dynamics adds
        # the same increment to both, mock physics is state-independent
        # except through saturation adjustment which depends on T).
        assert not np.allclose(T_member_0, T_member_1, atol=0), (
            "Ensemble members should differ in temperature"
        )
        # The difference should be O(1 K), not catastrophically large.
        max_diff = np.max(np.abs(T_member_0 - T_member_1))
        assert 0.5 < max_diff < 5.0, (
            f"Temperature difference between members is {max_diff:.4f} K, "
            f"expected approximately 1 K"
        )

    def test_vmap_preserves_step_index(self):
        """Step index is correctly incremented for all ensemble members."""
        vmapped_fn = jax.vmap(
            self.run_segment_raw,
            in_axes=(0, None, None),
        )
        vmapped_result = vmapped_fn(self.batched_carry, N_STEPS, self.forcing)
        jax.block_until_ready(vmapped_result)

        for m in range(N_ENSEMBLE):
            member = _extract_member(vmapped_result, m)
            assert int(member.step_index) == N_STEPS, (
                f"Member {m}: step_index is {int(member.step_index)}, "
                f"expected {N_STEPS}"
            )

    def test_vmap_all_fields_finite(self):
        """All carry fields remain finite after vmapped ensemble integration."""
        vmapped_fn = jax.vmap(
            self.run_segment_raw,
            in_axes=(0, None, None),
        )
        vmapped_result = vmapped_fn(self.batched_carry, N_STEPS, self.forcing)
        jax.block_until_ready(vmapped_result)

        for m in range(N_ENSEMBLE):
            member = _extract_member(vmapped_result, m)
            for field_name in SegmentCarry._fields:
                val = getattr(member, field_name)
                if val is None:
                    # Optional carry slot not populated by the mock setup.
                    continue
                arr = np.asarray(val)
                assert np.all(np.isfinite(arr)), (
                    f"Member {m}, field '{field_name}' has non-finite values"
                )

    def test_raw_matches_compiled_single_member(self):
        """Sanity check: raw segment fn matches compiled segment fn.

        Ensures the raw (non-JIT) function used for vmapping produces
        the same results as the JIT-compiled ``build_segment_fn`` version,
        so the vmap comparison is valid.
        """
        # Diagnostic accumulators where ULP-level differences from
        # reduction ordering are expected and harmless.
        _DIAGNOSTIC_FIELDS = {"max_cfl"}

        # Raw path (same function used in vmap)
        raw_result = self.run_segment_raw(
            _copy_carry(self.carry_0), N_STEPS, self.forcing
        )
        jax.block_until_ready(raw_result)

        # Compiled path
        compiled_result = self.run_segment(
            _copy_carry(self.carry_0), N_STEPS, self.forcing
        )
        jax.block_until_ready(compiled_result)

        for field_name in SegmentCarry._fields:
            raw_val = getattr(raw_result, field_name)
            compiled_val = getattr(compiled_result, field_name)
            # Optional carry slots are None on both paths — skip (see above).
            if raw_val is None or compiled_val is None:
                assert raw_val is None and compiled_val is None, field_name
                continue
            raw_arr = np.asarray(raw_val)
            compiled_arr = np.asarray(compiled_val)
            if field_name in _DIAGNOSTIC_FIELDS:
                np.testing.assert_allclose(
                    raw_arr, compiled_arr, atol=1e-14, rtol=1e-14,
                    err_msg=(
                        f"Diagnostic '{field_name}': raw segment fn "
                        f"differs from compiled segment fn"
                    ),
                )
            else:
                np.testing.assert_allclose(
                    raw_arr, compiled_arr, atol=0, rtol=0,
                    err_msg=(
                        f"Field '{field_name}': raw segment fn differs "
                        f"from compiled segment fn"
                    ),
                )
