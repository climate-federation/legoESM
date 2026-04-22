"""Scaling readiness validation.

Verifies that the operationalized infrastructure is compatible with
multi-device execution:
1. ParallelRuntime detects available devices
2. SegmentCarry is a valid pytree for sharding
3. SegmentForcing is a valid pytree for sharding
4. Compiled segment function compiles without error at C16
5. SYPD estimation (wall time per simulated day)

Does not require multiple GPUs — tests the code paths that would be
used for multi-GPU, verifying they work on a single device.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.compiled_segments import (
    SegmentCarry, SegmentForcing,
    pack_carry, pack_forcing, build_segment_fn,
)
from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig, OutputConfig
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.grids.vertical import create_sigma_coordinate


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
        kwargs["conv_prog"] = jnp.zeros((p_s.size,), dtype=T.dtype)
    return kwargs


class TestScalingReadiness:

    def test_segment_carry_is_valid_pytree(self):
        """SegmentCarry flattens and unflattens correctly."""
        s3 = (6, 8, 8, 10)
        s2 = (6, 8, 8)
        carry = SegmentCarry(
            u=jnp.zeros(s3), v=jnp.zeros(s3), T=jnp.zeros(s3),
            p_s=jnp.zeros(s2), phis=jnp.zeros(s2),
            q_v=jnp.zeros(s3), q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
            conv_prog=jnp.zeros((6 * 8 * 8,)),
            held_dT_rad=jnp.zeros(s3),
            held_sw_net_sfc=jnp.zeros(s2), held_lw_net_sfc=jnp.zeros(s2),
            held_sw_up_toa=jnp.zeros(s2), held_lw_up_toa=jnp.zeros(s2),
            held_sw_down_toa=jnp.zeros(s2),
            step_index=jnp.int32(0),
            target_moisture=jnp.float32(0.0),
            target_mass=jnp.float32(0.0),
            max_cfl=jnp.float32(0.0),
            precip_accum=jnp.zeros(s2),
            shflx_accum=jnp.zeros(s2),
            lhflx_accum=jnp.zeros(s2),
        )
        leaves, treedef = jax.tree.flatten(carry)
        reconstructed = treedef.unflatten(leaves)
        assert isinstance(reconstructed, SegmentCarry)
        assert len(leaves) == len(SegmentCarry._fields)

    def test_segment_forcing_is_valid_pytree(self):
        """SegmentForcing flattens and unflattens correctly."""
        s2 = (6, 8, 8)
        s3 = (6, 8, 8, 10)
        forcing = pack_forcing(
            sst=jnp.zeros(s2), sic=jnp.zeros(s2),
            day_of_year=1.0, seconds_of_day=0.0,
            solar_weights=jnp.ones(14), s_0=1361.0,
            o3_vmr=jnp.zeros(s3), aerosol_od=jnp.zeros(s2),
        )
        leaves, treedef = jax.tree.flatten(forcing)
        reconstructed = treedef.unflatten(leaves)
        assert isinstance(reconstructed, SegmentForcing)

    def test_parallel_runtime_detects_devices(self):
        """ParallelRuntime detects at least 1 device."""
        from legoesm.parallel.runtime import ParallelRuntime
        rt = ParallelRuntime.create(grid_type="cubed_sphere", n_devices=1)
        assert rt.mode in ("serial", "multi_device", "mpi", "hybrid")
        assert rt.local_device_count >= 1

    def test_c16_compile_and_run(self, tmp_path):
        """C16/L10 compile + 1-day run completes and reports timing."""
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=16, nlev=10),
            dycore=DycoreConfig(dt=600.0),
            output=OutputConfig(diag_days=1),
            days=1,
            dataset="analytical",
        )
        driver = ModelDriver(cfg, output_dir=tmp_path)
        driver.setup()

        t0 = time.monotonic()
        status = driver.run()
        wall = time.monotonic() - t0

        assert status == "COMPLETED"
        sypd = 1.0 / (wall / 86400.0)  # simulated years per day
        print(f"\n  C16/L10: {wall:.1f}s wall time, ~{sypd:.0f} SYPD")

    def test_all_carry_leaves_are_arrays(self):
        """Every leaf in SegmentCarry is a JAX array (no Python objects)."""
        s3 = (6, 4, 4, 3)
        s2 = (6, 4, 4)
        carry = SegmentCarry(
            u=jnp.zeros(s3), v=jnp.zeros(s3), T=jnp.zeros(s3),
            p_s=jnp.zeros(s2), phis=jnp.zeros(s2),
            q_v=jnp.zeros(s3), q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
            conv_prog=jnp.zeros((6 * 8 * 8,)),
            held_dT_rad=jnp.zeros(s3),
            held_sw_net_sfc=jnp.zeros(s2), held_lw_net_sfc=jnp.zeros(s2),
            held_sw_up_toa=jnp.zeros(s2), held_lw_up_toa=jnp.zeros(s2),
            held_sw_down_toa=jnp.zeros(s2),
            step_index=jnp.int32(0),
            target_moisture=jnp.float32(0.0),
            target_mass=jnp.float32(0.0),
            max_cfl=jnp.float32(0.0),
            precip_accum=jnp.zeros(s2),
            shflx_accum=jnp.zeros(s2),
            lhflx_accum=jnp.zeros(s2),
        )
        for field_name in SegmentCarry._fields:
            val = getattr(carry, field_name)
            assert isinstance(val, jax.Array), f"{field_name} is {type(val)}"

    def test_raw_segment_fn_available(self):
        """build_segment_fn returns function with .raw attribute for training."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.core.state import HydrostaticState

        grid = create_cubed_sphere(4)

        class M:
            _state_type = HydrostaticState
            def step(self, s, dt):
                return s

        def mock(need_rad, T, p_s, *a, **kw):
            p = PhysicsOutput(**_zero_physics_output(T, p_s))
            return p, (a[16], a[17], a[18], a[19], a[20], a[21])

        sigma = create_sigma_coordinate(3)
        fn = build_segment_fn(
            model=M(), step_unified=mock, grid=grid,
            sigma_coord=sigma,
            sigma_full=jnp.asarray(sigma.sigma_full),
            dsigma=jnp.asarray(sigma.dsigma), dt=600.0, rad_update_steps=1,
            microphysics="none", fix_moisture=False, fix_mass=False,
            fric_decay=jnp.ones(3), qv_smooth_coeff=0.0,
            lat=grid.lat, lon=grid.lon, start_day=0.0,
            sat_adjust_without_microphysics=False,
        )
        assert callable(fn)
        assert hasattr(fn, 'raw'), "Missing .raw attribute for training"
        assert callable(fn.raw)
