"""Stratosphere/tropopause fidelity tests for RRTMGP.

These tests grew with the ralph-loop iter-1 → iter-30 work to pin
down behaviour the audit found drifting from upstream rte-rrtmgp.

| Test class | Iter | Pins |
|---|---|---|
| ``TestClipToTableRange`` | 1+2 | ``_clip_to_table_range`` math for T outside [160, 355]K |
| ``TestRelativeAbundanceSafeDiv`` | 1+2/5 | safe-div for combined_vmr ∈ (0, eps] band |
| ``TestOutOfRangeTemperature`` | 1+2 | T-extrap doesn't catastrophically break planck/major/minor OD |
| ``TestMixedPrecision`` | 1+2/27 | fp32 inputs work end-to-end through fp64 tables |
| ``TestOptimalLwSecant`` | 2/4/7/8/11/12/21/25/43 | upstream ``compute_optimal_angles`` formula faithfulness + scan/loop/shard equivalence + extreme-tau limits |
| ``TestStandardO3Profile`` | 3/3.5/6 | skewed Gaussian + 20 ppb baseline + 200-400 DU column total |
| ``TestTropopauseBoundary`` | 26 | dead-branch AD safety at all-stratosphere / all-troposphere extreme columns |
| ``TestADSafetyAtStratosphereTau`` | 20 | iter-14/iter-19 max(d,eps) regression guard for stratospheric tau_tot |
| ``TestCloudKwargsHelper`` | 17 | iter-15/iter-16 cf²-double-discount regression guard |
| ``TestHeatingRateSign`` | 13 | iter-13 sign-fix regression guard (commit 0be22f0f) |

Companion file ``test_radiation.py`` adds cache-key tests
(iter-32/36/37/39/41/42) and sharded-equivalence tests
(iter-28/29) under the ``TestRRTMGP`` and
``TestColumnShardedRadiation`` classes.

The properties are required for differentiable training across the
full stratospheric column and for mixed-precision execution where
fp32 halo extrapolations can push temperatures out of the
fp64-table range.

A separate ``TestColumnShardedRadiation::test_rrtmgp_*`` pair in
``test_radiation.py`` (iter-28/29) pins MPI/GPU column-shard
invariance for both the default and ``use_optimal_angle=True``
paths.

Full chronology in ``docs/specs/rrtmgp.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# RRTMGP module accesses experimental Metal kernels at import; force CPU.
# Setting via env BEFORE import is the canonical approach but pytest may
# import jax first.  Use config to be safe.
jax.config.update("jax_platform_name", "cpu")

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
    gas_optics,
    optics_utils,
)


# ---------------------------------------------------------------------------
# _clip_to_table_range
# ---------------------------------------------------------------------------


class TestClipToTableRange:
    """Direct tests for the new safety helper."""

    def test_clips_below_min(self):
        t_ref = jnp.array([160.0, 200.0, 355.0])
        t = jnp.array([100.0, 150.0, 160.0, 200.0])
        out = gas_optics._clip_to_table_range(t, t_ref)
        np.testing.assert_array_equal(np.asarray(out), [160.0, 160.0, 160.0, 200.0])

    def test_clips_above_max(self):
        t_ref = jnp.array([160.0, 200.0, 355.0])
        t = jnp.array([300.0, 355.0, 400.0, 500.0])
        out = gas_optics._clip_to_table_range(t, t_ref)
        np.testing.assert_array_equal(np.asarray(out), [300.0, 355.0, 355.0, 355.0])

    def test_in_range_passthrough(self):
        t_ref = jnp.array([160.0, 200.0, 355.0])
        t = jnp.array([180.0, 250.0, 300.0])
        out = gas_optics._clip_to_table_range(t, t_ref)
        np.testing.assert_array_equal(np.asarray(out), [180.0, 250.0, 300.0])

    def test_clip_is_differentiable(self):
        """In-range T should have gradient 1; out-of-range T should have 0."""
        t_ref = jnp.array([160.0, 355.0])

        def loss(t):
            return jnp.sum(gas_optics._clip_to_table_range(t, t_ref))

        grad_fn = jax.grad(loss)
        # In range: gradient is 1.
        g_in = grad_fn(jnp.array([200.0]))
        np.testing.assert_allclose(np.asarray(g_in), [1.0])
        # Below min: gradient is 0 (clipped).
        g_low = grad_fn(jnp.array([100.0]))
        np.testing.assert_allclose(np.asarray(g_low), [0.0])
        # Above max: gradient is 0 (clipped).
        g_high = grad_fn(jnp.array([400.0]))
        np.testing.assert_allclose(np.asarray(g_high), [0.0])


# ---------------------------------------------------------------------------
# Safe-divide for binary-species fraction
# ---------------------------------------------------------------------------


def _toy_lookup_and_vmr():
    """Build a minimal lookup-table fixture by loading the LW gas-optics file.

    This keeps the test self-contained and avoids depending on the full
    radiation pipeline.  The LW file is bundled with the source tree.
    """
    from pathlib import Path
    from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
        lookup_gas_optics_longwave,
        lookup_volume_mixing_ratio,
        constants as optics_constants,
    )

    nc_path = (
        Path(__file__).resolve().parents[4]
        / "src/legoesm/atmosphere/physics/radiation/rrtmgp/optics"
        / "rrtmgp_data/rrtmgp-gas-lw-g128.nc"
    )
    lookup = lookup_gas_optics_longwave.from_data_file(str(nc_path))
    global_means = {
        optics_constants.DRY_AIR_KEY: optics_constants.DRY_AIR_VMR,
        "co2": 415e-6,
        "ch4": 1900e-9,
        "n2o": 332e-9,
        "o2": 0.20948,
        "n2": 0.78084,
        "co": 1.5e-7,
        "ccl4": 7.5e-11,
        "cfc11": 2.2e-10,
        "cfc12": 5.0e-10,
        "cfc22": 2.4e-10,
        "cf4": 8.5e-11,
        "no2": 3.0e-10,
    }
    vmr_lib = lookup_volume_mixing_ratio.LookupVolumeMixingRatio(
        global_means=global_means, profiles=None
    )
    return lookup, vmr_lib


@pytest.fixture(scope="module")
def lookup_vmr():
    return _toy_lookup_and_vmr()


class TestRelativeAbundanceSafeDiv:
    """The combined-VMR division must not NaN even when both species are 0."""

    def test_finite_when_combined_vmr_zero(self, lookup_vmr):
        lookup, vmr_lib = lookup_vmr
        # Use vmr_fields that make both major species exactly zero for the
        # selected band.  We force h2o and o3 to zero (the most common LW
        # major species in band 0/1).
        ncol, nlev = 2, 5
        shape = (ncol, 1, nlev)
        zero = jnp.zeros(shape)
        vmr_fields = {lookup.idx_h2o: zero, lookup.idx_o3: zero}
        # Temperatures and pressures inside the table.
        T = jnp.full(shape, 250.0)
        # Force lower atmosphere (p > p_ref_trop ≈ 1e4 Pa).
        p = jnp.full(shape, 5e4)
        molecules = jnp.full(shape, 1e22)
        tau = gas_optics.compute_major_optical_depth(
            lookup, vmr_lib, molecules, T, p, igpt=jnp.array(0),
            vmr_fields=vmr_fields,
        )
        assert jnp.all(jnp.isfinite(tau)), "major OD must be finite when combined VMR = 0"

    def test_finite_grad_when_combined_vmr_zero(self, lookup_vmr):
        lookup, vmr_lib = lookup_vmr
        shape = (1, 1, 1)
        zero = jnp.zeros(shape)
        vmr_fields = {lookup.idx_h2o: zero, lookup.idx_o3: zero}
        T = jnp.full(shape, 250.0)
        p = jnp.full(shape, 5e4)
        molecules = jnp.full(shape, 1e22)

        def loss(T_in):
            tau = gas_optics.compute_major_optical_depth(
                lookup, vmr_lib, molecules, T_in, p, igpt=jnp.array(0),
                vmr_fields=vmr_fields,
            )
            return jnp.sum(tau)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), "∂tau/∂T must be finite when combined VMR = 0"

    def test_bounded_grad_when_combined_vmr_subepsilon(self, lookup_vmr):
        """Iter-5: gradient must stay bounded when combined_vmr is in
        the (0, eps] band where the old ``> 0`` where would pick the
        division branch and emit ``1/eps ~ 1e30`` gradients."""
        lookup, vmr_lib = lookup_vmr
        shape = (1, 1, 1)
        # Set h2o + o3 to a tiny value to hit the (0, eps] regime.
        tiny_vmr = jnp.full(shape, 1.0e-35)
        vmr_fields = {lookup.idx_h2o: tiny_vmr, lookup.idx_o3: tiny_vmr}
        T = jnp.full(shape, 250.0)
        p = jnp.full(shape, 5e4)
        molecules = jnp.full(shape, 1e22)

        def loss(vmr):
            v = {lookup.idx_h2o: vmr, lookup.idx_o3: vmr}
            tau = gas_optics.compute_major_optical_depth(
                lookup, vmr_lib, molecules, T, p, igpt=jnp.array(0),
                vmr_fields=v,
            )
            return jnp.sum(tau)

        g = jax.grad(loss)(tiny_vmr)
        assert jnp.all(jnp.isfinite(g))
        # Bounded — must be << 1e20 to confirm the safety fix.
        assert jnp.abs(g).max() < 1e20, (
            f"∂tau/∂vmr must stay bounded for combined_vmr in (0, eps]; "
            f"got max |grad| = {float(jnp.abs(g).max()):.3e}"
        )


# ---------------------------------------------------------------------------
# Out-of-range temperature handling
# ---------------------------------------------------------------------------


class TestOutOfRangeTemperature:
    """Cold halo / mesospheric / hot tropical T must produce finite tau and B."""

    @pytest.mark.parametrize("T_aloft", [120.0, 150.0])
    def test_cold_mesospheric_planck_finite(self, lookup_vmr, T_aloft):
        lookup, vmr_lib = lookup_vmr
        shape = (1, 1, 4)
        T = jnp.array([300.0, 250.0, 220.0, T_aloft]).reshape(shape)
        pfrac = jnp.full(shape, 0.01)
        src = gas_optics.compute_planck_sources(
            lookup, pfrac, T, igpt=jnp.array(0)
        )
        assert jnp.all(jnp.isfinite(src)), f"Planck source must be finite for T={T_aloft}K"
        # Source at the cold cap should be <= source at the bottom of the
        # table range (T=160K), not negative or absurd.
        bottom_src = gas_optics.compute_planck_sources(
            lookup, pfrac, jnp.full(shape, lookup.t_planck[0]),
            igpt=jnp.array(0),
        )
        np.testing.assert_allclose(
            np.asarray(src[0, 0, 3]), np.asarray(bottom_src[0, 0, 3]),
            rtol=1e-6,
            err_msg=(
                "Out-of-range cold T should saturate at the table minimum, "
                "not extrapolate to negative or huge values."
            ),
        )

    @pytest.mark.parametrize("T_hot", [360.0, 500.0])
    def test_super_hot_planck_finite(self, lookup_vmr, T_hot):
        lookup, vmr_lib = lookup_vmr
        shape = (1, 1, 1)
        T = jnp.full(shape, T_hot)
        pfrac = jnp.full(shape, 0.01)
        src = gas_optics.compute_planck_sources(
            lookup, pfrac, T, igpt=jnp.array(0)
        )
        assert jnp.all(jnp.isfinite(src))
        top_src = gas_optics.compute_planck_sources(
            lookup, pfrac, jnp.full(shape, lookup.t_planck[-1]),
            igpt=jnp.array(0),
        )
        np.testing.assert_allclose(
            np.asarray(src), np.asarray(top_src), rtol=1e-6
        )

    def test_cold_planck_grad_finite(self, lookup_vmr):
        """∂planck_src/∂T should be 0 outside the table, finite inside."""
        lookup, _ = lookup_vmr
        shape = (1, 1, 1)
        pfrac = jnp.full(shape, 0.01)

        def src_sum(T):
            return jnp.sum(
                gas_optics.compute_planck_sources(
                    lookup, pfrac, T, igpt=jnp.array(0)
                )
            )

        # In-range: nonzero finite gradient.
        T_in = jnp.full(shape, 250.0)
        g_in = jax.grad(src_sum)(T_in)
        assert jnp.all(jnp.isfinite(g_in))
        assert jnp.abs(g_in).max() > 0.0
        # Below table range: clipped, zero gradient.
        T_low = jnp.full(shape, 100.0)
        g_low = jax.grad(src_sum)(T_low)
        assert jnp.all(jnp.isfinite(g_low))
        np.testing.assert_allclose(np.asarray(g_low), 0.0, atol=1e-12)
        # Above table range: clipped, zero gradient.
        T_high = jnp.full(shape, 400.0)
        g_high = jax.grad(src_sum)(T_high)
        assert jnp.all(jnp.isfinite(g_high))
        np.testing.assert_allclose(np.asarray(g_high), 0.0, atol=1e-12)

    def test_cold_major_optical_depth_finite(self, lookup_vmr):
        lookup, vmr_lib = lookup_vmr
        shape = (1, 1, 1)
        T = jnp.full(shape, 130.0)
        p = jnp.full(shape, 5e4)
        molecules = jnp.full(shape, 1e22)
        tau = gas_optics.compute_major_optical_depth(
            lookup, vmr_lib, molecules, T, p, igpt=jnp.array(0)
        )
        assert jnp.all(jnp.isfinite(tau))

    def test_cold_minor_optical_depth_finite(self, lookup_vmr):
        lookup, vmr_lib = lookup_vmr
        shape = (1, 1, 1)
        T = jnp.full(shape, 130.0)
        # Pressure on both sides of the tropopause to exercise both branches.
        for p_value in [5e4, 5e3]:
            p = jnp.full(shape, p_value)
            molecules = jnp.full(shape, 1e22)
            tau = gas_optics.compute_minor_optical_depth(
                lookup, vmr_lib, molecules, T, p, igpt=jnp.array(0)
            )
            assert jnp.all(jnp.isfinite(tau)), f"minor OD must be finite at p={p_value}"

    def test_out_of_range_T_saturates_major_OD(self, lookup_vmr):
        """Major OD must saturate at the table boundary for out-of-range T.

        ``compute_major_optical_depth`` uses temperature ONLY for the
        kmajor/vmr_ref table interpolation; with ``_clip_to_table_range``
        in place, T outside [t_ref[0], t_ref[-1]] must give the exact
        boundary lookup.  Closes codex iter-2 LOW coverage gap.

        Note: ``compute_minor_optical_depth`` does NOT fully saturate
        because the density-scaling factor ``p / T`` in
        ``scale_with_density_fn`` uses the *physical* temperature (not
        the clipped table-lookup T).  That is the correct physics — the
        Lorentz line-shape density scaling is meaningful outside the
        table range — so minor OD only needs to be finite, which is
        already covered by ``test_cold_minor_optical_depth_finite``.
        """
        lookup, vmr_lib = lookup_vmr
        shape = (1, 1, 1)
        T_cold = jnp.full(shape, 130.0)
        T_ref_min = jnp.full(shape, float(lookup.t_ref[0]))
        T_hot = jnp.full(shape, 500.0)
        T_ref_max = jnp.full(shape, float(lookup.t_ref[-1]))
        p = jnp.full(shape, 5e4)
        molecules = jnp.full(shape, 1e22)

        tau_cold = gas_optics.compute_major_optical_depth(
            lookup, vmr_lib, molecules, T_cold, p, igpt=jnp.array(0)
        )
        tau_ref_min = gas_optics.compute_major_optical_depth(
            lookup, vmr_lib, molecules, T_ref_min, p, igpt=jnp.array(0)
        )
        np.testing.assert_allclose(
            np.asarray(tau_cold), np.asarray(tau_ref_min), rtol=1e-12,
            err_msg="major OD must saturate at lower table boundary T",
        )

        tau_hot = gas_optics.compute_major_optical_depth(
            lookup, vmr_lib, molecules, T_hot, p, igpt=jnp.array(0)
        )
        tau_ref_max = gas_optics.compute_major_optical_depth(
            lookup, vmr_lib, molecules, T_ref_max, p, igpt=jnp.array(0)
        )
        np.testing.assert_allclose(
            np.asarray(tau_hot), np.asarray(tau_ref_max), rtol=1e-12,
            err_msg="major OD must saturate at upper table boundary T",
        )

    def test_out_of_range_T_saturates_rayleigh_OD(self, lookup_vmr):
        """Rayleigh OD must also saturate at table boundary T (same logic
        as major; no density scaling)."""
        # Need shortwave lookup for Rayleigh.
        from pathlib import Path
        from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
            lookup_gas_optics_shortwave,
        )
        _, vmr_lib = lookup_vmr
        sw_path = (
            Path(__file__).resolve().parents[4]
            / "src/legoesm/atmosphere/physics/radiation/rrtmgp/optics"
            / "rrtmgp_data/rrtmgp-gas-sw-g112.nc"
        )
        lookup_sw = lookup_gas_optics_shortwave.from_data_file(str(sw_path))

        shape = (1, 1, 1)
        T_cold = jnp.full(shape, 130.0)
        T_ref_min = jnp.full(shape, float(lookup_sw.t_ref[0]))
        p = jnp.full(shape, 5e4)
        molecules = jnp.full(shape, 1e22)

        tau_cold = gas_optics.compute_rayleigh_optical_depth(
            lookup_sw, vmr_lib, molecules, T_cold, p, igpt=jnp.array(0)
        )
        tau_ref = gas_optics.compute_rayleigh_optical_depth(
            lookup_sw, vmr_lib, molecules, T_ref_min, p, igpt=jnp.array(0)
        )
        np.testing.assert_allclose(
            np.asarray(tau_cold), np.asarray(tau_ref), rtol=1e-12,
            err_msg="Rayleigh OD must saturate at lower table boundary T",
        )


# ---------------------------------------------------------------------------
# Mixed-precision consistency
# ---------------------------------------------------------------------------


class TestMixedPrecision:
    """Same input should yield consistent fluxes across x32 / x64.

    The tables themselves are loaded at the active JAX precision; this
    test verifies that the *control flow* is invariant to dtype and that
    the float32 result is within an acceptable tolerance of float64.
    """

    def test_planck_source_finite_float32(self, lookup_vmr):
        # Tables are already loaded at the active precision (x64 in test
        # env via conftest).  Just verify that casting inputs to float32
        # does not crash the interpolation path.
        lookup, _ = lookup_vmr
        shape = (1, 1, 1)
        T = jnp.full(shape, 250.0, dtype=jnp.float32)
        pfrac = jnp.full(shape, 0.01, dtype=jnp.float32)
        src = gas_optics.compute_planck_sources(
            lookup, pfrac, T, igpt=jnp.array(0)
        )
        assert jnp.all(jnp.isfinite(src))

    def test_solve_columns_finite_float32_inputs(self):
        """Iter-27: end-to-end ``solve_columns`` with float32 inputs.

        Tables are loaded at x64 in the test env (per conftest), but
        ``solve_columns`` line 717 casts every input to the table
        dtype as the first step.  Verify that this cast doesn't crash
        the JIT compile path or produce non-finite values for a
        physically reasonable column.

        Specifically covers the mixed-precision use case where a
        downstream callsite (e.g. a fp32 training loop) feeds the
        RRTMGP solver fp32 atmosphere state.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 8
        # All inputs explicitly float32.
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1, dtype=jnp.float32)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 290.0, nlev, dtype=jnp.float32)[None, :]
        sfc_T = jnp.array([295.0], dtype=jnp.float32)
        q_v = jnp.full((ncol, nlev), 5e-3, dtype=jnp.float32)
        cos_z = jnp.array([0.5], dtype=jnp.float32)

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        # Internal cast promotes everything to table dtype (float64).
        for name in ("lw_flux_up", "sw_flux_up", "heating_rate"):
            val = getattr(out, name)
            assert jnp.all(jnp.isfinite(val)), (
                f"{name} contains non-finite values for fp32 inputs"
            )

    def test_solve_columns_fp32_matches_fp64_inputs(self):
        """Iter-63: ``solve_columns`` fp32 vs fp64 input must give
        numerically-equivalent fluxes.

        ``solve_columns`` casts inputs to the table dtype (fp64) as
        first step.  This test pins that the cast is dtype-only (not
        precision-degrading): with identical physical column, calling
        with fp32 inputs vs fp64 inputs must produce relative errors
        no larger than ~fp32 round-off (5e-6 rtol, 1e-3 atol on
        W/m²-scale fluxes).

        Catches regressions where a future refactor accidentally
        skips the cast or uses fp32 intermediates inside the
        correlated-k inner loop.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 8

        def _build_inputs(dtype):
            p_half = jnp.broadcast_to(
                jnp.linspace(100.0, 1.0e5, nlev + 1, dtype=dtype)[None, :],
                (ncol, nlev + 1),
            )
            p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
            T = jnp.linspace(220.0, 290.0, nlev, dtype=dtype)[None, :]
            sfc_T = jnp.array([295.0], dtype=dtype)
            q_v = jnp.full((ncol, nlev), 5e-3, dtype=dtype)
            cos_z = jnp.array([0.5], dtype=dtype)
            return p_full, p_half, T, sfc_T, q_v, cos_z

        p_full_64, p_half_64, T_64, sfc_64, q_64, cos_64 = (
            _build_inputs(jnp.float64)
        )
        out64 = solver.solve_columns(
            T=T_64, p_full=p_full_64, p_half=p_half_64,
            sfc_temperature=sfc_64, q_v=q_64, cos_zenith=cos_64,
        )

        p_full_32, p_half_32, T_32, sfc_32, q_32, cos_32 = (
            _build_inputs(jnp.float32)
        )
        out32 = solver.solve_columns(
            T=T_32, p_full=p_full_32, p_half=p_half_32,
            sfc_temperature=sfc_32, q_v=q_32, cos_zenith=cos_32,
        )

        # fp32 inputs round-trip through table (fp64) interp.  Tolerance
        # is set by fp32 input round-off (~1e-7 rel) plus a small slack
        # for table-interp non-linearity.
        for name in ("lw_flux_up", "lw_flux_down", "sw_flux_up",
                     "sw_flux_down", "heating_rate"):
            v32 = np.asarray(getattr(out32, name))
            v64 = np.asarray(getattr(out64, name))
            np.testing.assert_allclose(
                v32, v64, rtol=5e-6, atol=1e-3,
                err_msg=(
                    f"{name}: fp32-input vs fp64-input divergence "
                    f"exceeds fp32-round-off tolerance.  Likely a "
                    f"missing dtype cast in solve_columns."
                ),
            )

    def test_solve_columns_bfloat16_inputs_finite(self):
        """Iter-63: ``solve_columns`` must accept bfloat16 inputs
        without crashing (covers ML/training paths that pack state in
        bf16).  Table-dtype cast is the safety net; this test pins
        the cast covers bf16 too.

        We only assert finiteness — bfloat16 has 8-bit mantissa
        (~2e-2 relative precision) so a fp64-comparison would be
        meaningless.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 8
        bf16 = jnp.bfloat16
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1, dtype=bf16)[None, :],
            (ncol, nlev + 1),
        )
        p_full = (0.5 * (p_half[:, :-1].astype(jnp.float32)
                         + p_half[:, 1:].astype(jnp.float32))
                  ).astype(bf16)
        T = jnp.linspace(220.0, 290.0, nlev, dtype=bf16)[None, :]
        sfc_T = jnp.array([295.0], dtype=bf16)
        q_v = jnp.full((ncol, nlev), 5e-3, dtype=bf16)
        cos_z = jnp.array([0.5], dtype=bf16)

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        for name in ("lw_flux_up", "sw_flux_up", "heating_rate"):
            val = getattr(out, name)
            assert jnp.all(jnp.isfinite(val)), (
                f"{name} contains non-finite values for bfloat16 inputs"
            )


# ---------------------------------------------------------------------------
# Optimal LW diffusivity angle (iter-2)
# ---------------------------------------------------------------------------


class TestOptimalLwSecant:
    """Iter-2 ``_compute_optimal_lw_secant`` matches upstream formula."""

    def test_optimal_angle_fit_loaded(self, lookup_vmr):
        lookup, _ = lookup_vmr
        assert lookup.optimal_angle_fit is not None, (
            "rrtmgp-gas-lw-g128.nc should ship optimal_angle_fit; if absent "
            "the user is running a pre-2023 RRTMGP data file."
        )
        assert lookup.optimal_angle_fit.shape == (lookup.n_bnd, 2)

    def test_secant_formula_matches_upstream(self, lookup_vmr):
        """secant = c0 * exp(-tau_total) + c1 (per band, per col)."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
            two_stream,
        )
        lookup, _ = lookup_vmr
        # Synthetic tau with halo width 1 and 3 interior layers.
        # interior tau values sum to 2.0 → trans_total = exp(-2) ≈ 0.1353.
        tau = jnp.array(
            [[[0.0, 0.5, 1.0, 0.5, 0.0]]],  # halos at ends
            dtype=jnp.float64,
        )
        band_idx = jnp.array(0)
        c0, c1 = float(lookup.optimal_angle_fit[0, 0]), float(
            lookup.optimal_angle_fit[0, 1]
        )
        expected = c0 * np.exp(-2.0) + c1
        secant = two_stream._compute_optimal_lw_secant(
            tau, band_idx, lookup.optimal_angle_fit, halo_width=1
        )
        np.testing.assert_allclose(np.asarray(secant)[0, 0, 0], expected, rtol=1e-12)

    def test_secant_extreme_tau_limits(self, lookup_vmr):
        """iter-43: pin numerical limits of ``_compute_optimal_lw_secant``.

        Verifies the asymptotic behaviour required by the upstream
        formula ``c0 * exp(-Σtau) + c1``:

        - ``tau → +∞``: ``trans → 0``, so ``secant → c1`` (regardless
          of c0).
        - ``tau = inf``: same limit; no overflow / NaN.
        - ``tau = NaN`` (pathological input): NaN propagates (no
          silent masking that would hide an upstream bug).

        Catches a regression in the ``jnp.maximum(tau, 0)`` clamp
        or in the ``exp`` underflow behaviour that would shift the
        asymptote.
        """
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
            two_stream,
        )
        lookup, _ = lookup_vmr
        # Use band 5 (c0=0.20, c1=1.50 per iter-2 inspection) so the
        # tau dependence is non-trivial in the finite range.
        band_idx = jnp.array(5)
        c1 = float(lookup.optimal_angle_fit[5, 1])

        # tau = 1000 (large but finite).  exp(-3000) underflows to 0.
        tau_huge = jnp.full((1, 1, 5), 1000.0, dtype=jnp.float64)
        secant = two_stream._compute_optimal_lw_secant(
            tau_huge, band_idx, lookup.optimal_angle_fit, halo_width=1
        )
        np.testing.assert_allclose(np.asarray(secant)[0, 0, 0], c1, rtol=1e-12)

        # tau = +inf.
        tau_inf = jnp.full((1, 1, 5), jnp.inf, dtype=jnp.float64)
        secant_inf = two_stream._compute_optimal_lw_secant(
            tau_inf, band_idx, lookup.optimal_angle_fit, halo_width=1
        )
        np.testing.assert_allclose(np.asarray(secant_inf)[0, 0, 0], c1, rtol=1e-12)
        assert jnp.isfinite(secant_inf).all(), (
            "tau=inf must produce finite secant=c1, not nan/inf"
        )

        # tau = NaN: NaN must propagate (no silent masking).
        tau_nan = jnp.full((1, 1, 5), jnp.nan, dtype=jnp.float64)
        secant_nan = two_stream._compute_optimal_lw_secant(
            tau_nan, band_idx, lookup.optimal_angle_fit, halo_width=1
        )
        assert jnp.isnan(secant_nan).all(), (
            "tau=NaN must propagate to secant=NaN; silent masking would "
            "hide upstream optical-depth NaN bugs"
        )

    def test_secant_constant_when_c0_is_zero(self, lookup_vmr):
        """Iter-25: when ``optimal_angle_fit[band, 0] == 0`` (most bands
        in the shipped data; per iter-2 inspection: bands 0-4, 9, 10,
        13, 14, 15), the secant must be exactly ``c1`` regardless of
        the column transmissivity.  Catches a regression where a
        future edit might inadvertently break the linear-in-trans
        formula or sample the wrong axis."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
            two_stream,
        )
        lookup, _ = lookup_vmr
        zero_c0_bands = np.where(np.asarray(lookup.optimal_angle_fit[:, 0]) == 0.0)[0]
        assert zero_c0_bands.size > 0, (
            "expected at least one band with c0=0 in the shipped data"
        )
        band_idx = jnp.array(int(zero_c0_bands[0]))
        c1 = float(lookup.optimal_angle_fit[band_idx, 1])
        # Vary tau across a wide range; secant should be invariant.
        for tau_val in [0.01, 0.1, 1.0, 10.0]:
            tau = jnp.array([[[0.0, tau_val, 0.0]]], dtype=jnp.float64)
            secant = two_stream._compute_optimal_lw_secant(
                tau, band_idx, lookup.optimal_angle_fit, halo_width=1
            )
            np.testing.assert_allclose(
                np.asarray(secant)[0, 0, 0], c1, rtol=1e-12,
                err_msg=(
                    f"band {band_idx} has c0=0; secant must equal c1={c1} "
                    f"regardless of tau, got secant={float(secant[0,0,0]):.6f} "
                    f"at tau={tau_val}"
                ),
            )

    def test_secant_physically_reasonable_range(self, lookup_vmr):
        """Iter-21: optimal LW secant must lie in [1.0, 2.0] for all
        bands across the full tau range [0, +inf].  The Fu-Liou 1.66
        is the canonical default; upstream RFMIP reports band-mean
        secants of 1.5-1.9.  Anything outside [1.0, 2.0] would imply
        an unphysical diffusivity (<1 means no angle widening, >2
        means more than ~60° tilt) and signals a misconfigured
        ``optimal_angle_fit`` (e.g. via a corrupted data file or a
        manually-constructed lookup with wrong coefficient ordering).
        """
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
            two_stream,
        )
        lookup, _ = lookup_vmr
        fit = lookup.optimal_angle_fit
        # For each band, eval secant at tau=0 (trans=1) and tau=inf
        # (trans=0).  These bracket the achievable secant range.
        c0 = np.asarray(fit[:, 0])
        c1 = np.asarray(fit[:, 1])
        secant_tau0 = c0 + c1     # trans=1: tau=0
        secant_tauinf = c1        # trans=0: tau=inf
        all_secants = np.concatenate([secant_tau0, secant_tauinf])
        assert (all_secants >= 1.0).all() and (all_secants <= 2.0).all(), (
            f"optimal_angle_fit secants outside [1.0, 2.0] range: "
            f"min={all_secants.min():.3f}, max={all_secants.max():.3f}; "
            f"per-band tau=0: {secant_tau0}, tau=inf: {secant_tauinf}"
        )

    def test_secant_halo_width_zero(self, lookup_vmr):
        """Iter-12: ``_compute_optimal_lw_secant(halo_width=0)`` sums
        over ALL z cells (no halo strip).  Guards against future
        refactors that hard-code ``hw=1`` and silently drop layers in
        callers that pass un-halo'd optical depth (e.g. direct unit
        tests on the helper)."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
            two_stream,
        )
        lookup, _ = lookup_vmr
        # 5 interior cells, no halos.
        tau = jnp.array([[[0.5, 1.0, 0.5, 0.5, 0.5]]], dtype=jnp.float64)
        band_idx = jnp.array(0)
        c0, c1 = float(lookup.optimal_angle_fit[0, 0]), float(
            lookup.optimal_angle_fit[0, 1]
        )
        # Sum over all 5 cells = 3.0, trans = exp(-3) ≈ 0.04979.
        expected = c0 * np.exp(-3.0) + c1
        secant = two_stream._compute_optimal_lw_secant(
            tau, band_idx, lookup.optimal_angle_fit, halo_width=0
        )
        np.testing.assert_allclose(np.asarray(secant)[0, 0, 0], expected, rtol=1e-12)

    def test_secant_grad_finite(self, lookup_vmr):
        """Gradient of secant w.r.t. tau must be finite (no NaN)."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
            two_stream,
        )
        lookup, _ = lookup_vmr
        tau = jnp.array([[[0.0, 0.3, 0.7, 0.3, 0.0]]], dtype=jnp.float64)

        def loss(t):
            s = two_stream._compute_optimal_lw_secant(
                t, jnp.array(0), lookup.optimal_angle_fit, halo_width=1
            )
            return jnp.sum(s)

        g = jax.grad(loss)(tau)
        assert jnp.all(jnp.isfinite(g))

    def test_solve_lw_optimal_angle_smoke(self, lookup_vmr):
        """Solve LW with ``use_optimal_angle=True`` does not crash and
        differs from the fixed-1.66 result (sanity check that the kwarg
        actually plumbs through to the two-stream solver)."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        cfg_baseline = RRTMGPConfig(use_optimal_angle=False)
        cfg_optimal = RRTMGPConfig(use_optimal_angle=True)
        # Fresh instances so the solver pulls the new config.
        solver_b = RRTMGP.from_legoesm_config(cfg_baseline)
        solver_o = RRTMGP.from_legoesm_config(cfg_optimal)

        ncol, nlev = 2, 8
        # Decreasing pressure TOA -> surface, range 100 Pa -> 1e5 Pa.
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(
            jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev)
        )
        sfc_T = jnp.full((ncol,), 295.0)
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.full((ncol,), 0.5)

        out_b = solver_b.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        out_o = solver_o.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        assert jnp.all(jnp.isfinite(out_b.lw_flux_up))
        assert jnp.all(jnp.isfinite(out_o.lw_flux_up))
        # Optimal-angle path should change the result somewhere (band-
        # dependent secants ≠ 1.66) — guard against silently-no-op plumbing.
        max_diff = float(jnp.max(jnp.abs(out_b.lw_flux_up - out_o.lw_flux_up)))
        assert max_diff > 1e-3, (
            f"use_optimal_angle=True should change LW flux; got max_diff={max_diff}"
        )

    def test_raises_when_optimal_angle_fit_missing(self, lookup_vmr):
        """``solve_lw(use_optimal_angle=True)`` must error when the
        gas-optics file lacks ``optimal_angle_fit`` rather than silently
        degrading to fixed 1.66 (codex iter-2 MEDIUM)."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream
        from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
            atmospheric_state, lookup_volume_mixing_ratio,
            constants as optics_constants,
        )
        import dataclasses

        lookup, vmr_lib = lookup_vmr
        # Build a sibling lookup with optimal_angle_fit=None to simulate
        # an old data file.
        lookup_no_fit = dataclasses.replace(lookup, optimal_angle_fit=None)

        # Fake an optics_lib with .gas_optics_lw exposing the stripped lookup.
        class _FakeOpticsLib:
            gas_optics_lw = lookup_no_fit
        atmos_state = atmospheric_state.AtmosphericState(
            sfc_emis=0.98, sfc_alb=0.06, zenith=0.0,
            irrad=1360.0, vmr=vmr_lib, toa_flux_lw=0.0,
        )
        # Minimal inputs (shape match expected).
        T = jnp.full((1, 1, 3), 250.0)
        p = jnp.full((1, 1, 3), 5e4)
        mol = jnp.full((1, 1, 3), 1e22)
        with pytest.raises(ValueError, match="optimal_angle_fit"):
            two_stream.solve_lw(
                p, T, mol, _FakeOpticsLib(), atmos_state,
                sfc_temperature=jnp.array([[260.0]]),
                use_optimal_angle=True,
            )

    def test_optimal_angle_per_column_secants_differ(self, lookup_vmr):
        """Verify multi-column path: columns with different total tau
        produce different secants, exercising the broadcasting against
        the ``(ncol, 1, nlev+2)`` optical-depth tensor."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
            two_stream,
        )
        lookup, _ = lookup_vmr
        # Pick a band where c0 != 0 so the secant actually varies with tau.
        nonzero = np.where(np.asarray(lookup.optimal_angle_fit[:, 0]) != 0.0)[0]
        assert nonzero.size > 0, "expected at least one band with nonzero c0"
        band_idx = jnp.array(int(nonzero[0]))
        # 3 cols: tau totals 0.1, 1.0, 5.0 → trans 0.90, 0.37, 0.0067.
        # halo width 1; interior has 3 cells per column.
        tau = jnp.array(
            [
                [[0.0, 0.05, 0.0, 0.05, 0.0]],
                [[0.0, 0.5,  0.0, 0.5,  0.0]],
                [[0.0, 2.5,  0.0, 2.5,  0.0]],
            ],
            dtype=jnp.float64,
        )
        secant = two_stream._compute_optimal_lw_secant(
            tau, band_idx, lookup.optimal_angle_fit, halo_width=1
        )
        assert secant.shape == (3, 1, 1)
        # Different total tau ⇒ different secants (band c0 != 0).
        vals = np.asarray(secant).reshape(-1)
        assert len(set(vals.tolist())) == 3, (
            f"expected 3 distinct secants, got {vals}"
        )

    def test_solve_columns_is_column_permutation_invariant(self, lookup_vmr):
        """Iter-11: ``solve_columns`` output for column ``k`` must
        depend only on that column's inputs.  Permuting the column
        axis must permute the output identically.  Catches per-column
        state leaks (e.g. a stale ``cumulative_flux`` carry that
        accidentally accumulates across columns instead of along the
        vertical axis), which would break sharded MPI/GPU runs where
        each rank holds a different column subset.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 4, 8
        # Each column gets a slightly different T profile so the
        # outputs are distinguishable per column.
        T = jnp.stack(
            [jnp.linspace(220.0 + 5.0 * k, 290.0 + 5.0 * k, nlev) for k in range(ncol)]
        )
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        sfc_T = jnp.array([295.0 + 5.0 * k for k in range(ncol)])
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.array([0.4 + 0.05 * k for k in range(ncol)])

        out_orig = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        # Reverse the column axis on every input.
        perm = jnp.array([3, 2, 1, 0])
        out_perm = solver.solve_columns(
            T=T[perm],
            p_full=p_full[perm],
            p_half=p_half[perm],
            sfc_temperature=sfc_T[perm],
            q_v=q_v[perm],
            cos_zenith=cos_z[perm],
        )
        for name in ("lw_flux_up", "sw_flux_up", "heating_rate"):
            a = getattr(out_orig, name)
            b = getattr(out_perm, name)
            np.testing.assert_allclose(
                np.asarray(a)[perm], np.asarray(b),
                rtol=1e-12, atol=1e-12,
                err_msg=(
                    f"{name}: permuting the column axis must permute "
                    "the output identically (column-local independence)."
                ),
            )

    def test_optimal_angle_flux_impact_bounded(self, lookup_vmr):
        """Iter-8: optimal-angle LW flux must be within ~10% of the
        fixed-1.66 result.  Establishes a sanity bound — if a future
        change perturbs the optimal-angle math so the flux drifts by
        50% or 200%, this would flag it.  Upstream comparisons (RFMIP)
        report band-mean secants of 1.5-1.9 so the integrated LW flux
        should differ from the 1.66 result by single-digit percent."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        ncol, nlev = 1, 10
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        # US Std-ish T profile: 220K aloft → 290K surface.
        T = jnp.broadcast_to(
            jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev)
        )
        sfc_T = jnp.array([295.0])
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.array([0.5])

        out_b = RRTMGP.from_legoesm_config(RRTMGPConfig()).solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        out_o = RRTMGP.from_legoesm_config(
            RRTMGPConfig(use_optimal_angle=True)
        ).solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )

        # Pick the surface (highest signal) upward LW flux for comparison.
        b_sfc = float(jnp.abs(out_b.lw_flux_up[0, 0]))
        o_sfc = float(jnp.abs(out_o.lw_flux_up[0, 0]))
        rel = abs(o_sfc - b_sfc) / max(b_sfc, 1e-12)
        assert rel < 0.10, (
            f"|optimal − fixed-1.66| / fixed = {rel:.3%} at surface — "
            f"expected < 10% (band-mean secants are 1.5-1.9); "
            f"fixed={b_sfc:.2f} W/m², optimal={o_sfc:.2f} W/m²"
        )
        # Also assert it's NOT essentially zero (would mean both paths
        # collapsed to the same constant — sign of broken plumbing).
        assert rel > 1e-6, (
            f"Optimal-angle path produced indistinguishable flux from "
            f"fixed-1.66; check use_optimal_angle plumbing.  rel={rel:.3e}"
        )

    def test_optimal_angle_use_scan_equivalence(self, lookup_vmr):
        """Iter-7: ``use_optimal_angle=True`` must give bit-equivalent
        results with ``use_scan=True`` and ``use_scan=False``.  This
        guards against the optimal-angle path silently breaking the
        GPU/CPU equivalence the existing ``test_rrtmgp_use_scan_equivalence``
        verifies for the fixed-1.66 path."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        ncol, nlev = 2, 8
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(
            jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev)
        )
        sfc_T = jnp.full((ncol,), 295.0)
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.full((ncol,), 0.5)

        cfg_loop = RRTMGPConfig(use_optimal_angle=True, use_scan=False)
        cfg_scan = RRTMGPConfig(use_optimal_angle=True, use_scan=True)
        out_loop = RRTMGP.from_legoesm_config(cfg_loop).solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        out_scan = RRTMGP.from_legoesm_config(cfg_scan).solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        for name, a, b in (
            ("lw_flux_up",   out_loop.lw_flux_up,   out_scan.lw_flux_up),
            ("lw_flux_down", out_loop.lw_flux_down, out_scan.lw_flux_down),
            ("heating_rate", out_loop.heating_rate, out_scan.heating_rate),
        ):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-10,
                err_msg=(
                    f"{name}: use_optimal_angle=True must match between "
                    "scan and unrolled column recurrence."
                ),
            )

    def test_solve_columns_use_optimal_angle_differentiable(self, lookup_vmr):
        """End-to-end ``jax.grad`` through ``solve_columns(..., use_optimal_angle=True)``
        must produce finite gradients w.r.t. temperature."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        cfg = RRTMGPConfig(use_optimal_angle=True)
        solver = RRTMGP.from_legoesm_config(cfg)
        ncol, nlev = 1, 5
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 290.0, nlev)[None, :]
        sfc_T = jnp.array([295.0])
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.array([0.5])

        def loss(T):
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half,
                sfc_temperature=sfc_T, q_v=q_v, cos_zenith=cos_z,
            )
            return jnp.sum(out.lw_flux_up)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), (
            "∂(LW flux)/∂T must be finite through use_optimal_angle=True path"
        )
        # Non-trivial gradient — guard against constant-fold collapse.
        assert jnp.abs(g).max() > 1e-9


class TestStandardO3Profile:
    """Iter-3 skewed-Gaussian climatology fits US Std Atm 1976 within a
    factor of ~2 at the canonical levels."""

    def test_peak_at_10_hPa(self):
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            _standard_o3_profile,
        )
        # Fine grid across the stratosphere.
        p_hPa = jnp.logspace(jnp.log10(0.1), jnp.log10(1000.0), 200)
        o3 = _standard_o3_profile(p_hPa * 100.0)
        peak_idx = int(jnp.argmax(o3))
        peak_p = float(p_hPa[peak_idx])
        # The skewed-Gaussian peak sits at 10 hPa.  ``argmax`` on a fine
        # log grid resolves to within a few percent.
        assert 9.0 < peak_p < 11.0, f"peak at {peak_p} hPa, expected 10"

    def test_canonical_levels_within_factor_of_two(self):
        """Compare new profile to US Std Atm 1976 reference at key
        stratospheric levels (peak ± 2 dex in log-pressure)."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            _standard_o3_profile,
        )
        # Reference O3 from US Std Atm 1976 (ppm, mid-latitude annual mean):
        ref = {
            100.0: 0.25,   # tropopause:    ~250 ppb
            30.0:  5.0,    # mid-strat:     ~5 ppm
            10.0:  8.0,    # peak:          ~7-9 ppm (we target 9)
            1.0:   3.0,    # upper strat:   ~3 ppm
        }
        for p_hPa_val, expected_ppm in ref.items():
            p = jnp.array([p_hPa_val * 100.0])
            o3_ppm = float(_standard_o3_profile(p)[0]) * 1e6
            ratio = o3_ppm / expected_ppm
            assert 0.5 < ratio < 2.0, (
                f"O3 at {p_hPa_val} hPa: got {o3_ppm:.3f} ppm, "
                f"expected ~{expected_ppm} (ratio {ratio:.2f}, "
                f"target [0.5, 2.0])"
            )

    def test_extended_coverage_troposphere_and_mesosphere(self):
        """Beyond the stratospheric peak ± 2 dex, the analytic fit is
        order-of-magnitude only.  Closes codex iter-3 coverage gap on
        the 200-500 hPa UT/LS transition and the mesosphere ~0.1 hPa.
        Tolerance widened to ``[0.2, 5.0]`` to reflect what a single
        skewed-Gaussian + background can deliver."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            _standard_o3_profile,
        )
        ref = {
            1000.0: 0.025,  # surface/BL:   ~25 ppb (we cap at 20 ppb)
            500.0:  0.050,  # mid-trop:     ~50 ppb (we have 20 ppb)
            200.0:  0.100,  # near tropopause: ~100 ppb
            0.1:    0.080,  # mesosphere:   ~80 ppb
        }
        for p_hPa_val, expected_ppm in ref.items():
            p = jnp.array([p_hPa_val * 100.0])
            o3_ppm = float(_standard_o3_profile(p)[0]) * 1e6
            ratio = o3_ppm / expected_ppm
            assert 0.2 < ratio < 5.0, (
                f"O3 at {p_hPa_val} hPa: got {o3_ppm:.4f} ppm, "
                f"expected ~{expected_ppm} (ratio {ratio:.2f}, "
                f"target [0.2, 5.0] for non-peak levels)"
            )

    def test_tropospheric_background_floor(self):
        """Below ~250 hPa the profile is capped at 20 ppb to prevent
        the Gaussian's deep skirt from collapsing to ~ppt values."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            _standard_o3_profile,
        )
        # Deep troposphere; Gaussian alone would give << 1 ppb.
        p = jnp.array([5e4, 8e4, 1e5])  # 500, 800, 1000 hPa
        o3 = _standard_o3_profile(p)
        # All three should be at the 20 ppb background.
        np.testing.assert_allclose(
            np.asarray(o3), 2.0e-8 * np.ones_like(np.asarray(o3)),
            rtol=1e-12,
            err_msg="tropospheric O3 must equal 20 ppb background",
        )

    def test_grad_through_o3_profile(self):
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            _standard_o3_profile,
        )

        def loss(p):
            return jnp.sum(_standard_o3_profile(p))

        # Use real pressure values away from the σ-discontinuity at 10 hPa.
        p = jnp.array([5e4, 1e3, 50.0])  # 500, 10, 0.5 hPa
        g = jax.grad(loss)(p)
        assert jnp.all(jnp.isfinite(g))

    def test_integrated_o3_column_in_dobson_units(self):
        """Iter-6: integrated O3 column [DU] must be in a physically
        reasonable range (200-400 DU spans tropics to high latitudes
        through the annual cycle; mid-lat annual mean ≈ 300 DU).
        Verifies the iter-3.5 skewed-Gaussian + tropospheric background
        gives a realistic column total, not just realistic point values.
        """
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            _standard_o3_profile,
        )
        # Integration ``DU = 7891 * ∫ vmr_o3 dp[Pa]`` derived from
        # ``DU = (N_A / (g * M_air)) / 2.687e20 * ∫ vmr * dp``,
        # using N_A=6.022e23, g=9.81 m/s², M_air=0.02897 kg/mol,
        # 1 DU = 2.687e20 molecules/m².
        p = jnp.logspace(0.0, jnp.log10(1.0e5), 5000)  # 1 Pa → 1e5 Pa
        vmr = _standard_o3_profile(p)
        integral = float(jnp.trapezoid(vmr, p))
        du = 7891.0 * integral
        assert 200.0 < du < 400.0, (
            f"Std O3 column = {du:.1f} DU, expected [200, 400] "
            f"(typical mid-lat range 250-350)."
        )


class TestTropopauseBoundary:
    """Iter-26: exercise solve_columns when every layer is on ONE
    side of the tropopause (p_ref_trop ≈ 100 hPa).  Smoke-tests that
    the lower/upper-atm dual-branch in compute_minor_optical_depth
    and the +itropo-1 pressure shift in compute_major_optical_depth
    don't crash when one branch contributes nothing.  Also pins the
    differentiability across the boundary."""

    def test_all_stratospheric_column(self):
        """Column with every layer above the tropopause (p ≤ 100 hPa).
        Upper-atm branch dominates; lower-atm branch should compute
        zero contribution and the dead-branch gradient should still
        be finite."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 12
        # All in stratosphere: 1 Pa → 8000 Pa (= 80 hPa).
        p_half = jnp.broadcast_to(
            jnp.linspace(1.0, 8000.0, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.full((ncol, nlev), 240.0)
        sfc_T = jnp.array([245.0])
        q_v = jnp.full((ncol, nlev), 1e-6)
        cos_z = jnp.array([0.5])

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        assert jnp.all(jnp.isfinite(out.lw_flux_up))
        assert jnp.all(jnp.isfinite(out.sw_flux_up))
        assert jnp.all(jnp.isfinite(out.heating_rate))

        # AD path must be finite — the lower-branch
        # ``_compute_minor_optical_depth`` is evaluated everywhere via
        # the jnp.where, so an unmasked NaN gradient from the dead
        # branch would surface here.
        def loss(T_in):
            o = solver.solve_columns(
                T=T_in, p_full=p_full, p_half=p_half,
                sfc_temperature=sfc_T, q_v=q_v, cos_zenith=cos_z,
            )
            return jnp.sum(o.heating_rate)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), (
            "all-stratosphere column gradient must be finite — "
            "the dead lower-atm branch in compute_minor_optical_depth "
            "must not propagate NaN cotangents."
        )

    def test_all_tropospheric_column(self):
        """Column with every layer below the tropopause (p > 100 hPa).
        Symmetric to the stratosphere test — lower-atm branch dominates,
        upper-atm dead branch must not break AD."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 8
        # All in troposphere: 200 hPa → 1000 hPa (= 2e4 → 1e5 Pa).
        p_half = jnp.broadcast_to(
            jnp.linspace(2.0e4, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 290.0, nlev)[None, :]
        sfc_T = jnp.array([295.0])
        q_v = jnp.linspace(1e-5, 1e-2, nlev)[None, :]
        cos_z = jnp.array([0.5])

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        assert jnp.all(jnp.isfinite(out.lw_flux_up))
        assert jnp.all(jnp.isfinite(out.heating_rate))

        def loss(T_in):
            o = solver.solve_columns(
                T=T_in, p_full=p_full, p_half=p_half,
                sfc_temperature=sfc_T, q_v=q_v, cos_zenith=cos_z,
            )
            return jnp.sum(o.heating_rate)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), (
            "all-troposphere column gradient must be finite — "
            "the dead upper-atm branch must not propagate NaN "
            "cotangents."
        )


class TestADSafetyAtStratosphereTau:
    """Iter-20: end-to-end ``jax.grad`` through ``solve_columns`` for
    a stratosphere-only column whose SW optical depth is small enough
    that the iter-14 (commit 59407953) AD-unsafe maximum-floor pattern
    would have triggered.  Pins the AD chain.

    Pre-iter-14 symptom: ``tau_tot`` reaches the 1e-12 floor in cloud-
    free, low-water-vapor stratospheric layers; the legacy
    ``num / jnp.maximum(tau_tot, 1e-12)`` divide's VJP overflows to
    NaN.  Codex iter-18 review found 4 more sites with the same
    pattern (iter-19 fixed) — this test pins the full chain.
    """

    def test_jax_grad_finite_for_pure_stratosphere_column(self):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 12
        # Pure stratosphere: pressure 1 to 200 hPa, T 220K throughout.
        # No clouds, near-zero water vapor.
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 2.0e4, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.full((ncol, nlev), 220.0)
        sfc_T = jnp.array([230.0])
        # 0.1 ppm water vapor — typical mid-stratosphere.
        q_v = jnp.full((ncol, nlev), 1e-7)
        cos_z = jnp.array([0.4])

        def loss(T_in):
            out = solver.solve_columns(
                T=T_in, p_full=p_full, p_half=p_half,
                sfc_temperature=sfc_T, q_v=q_v, cos_zenith=cos_z,
            )
            return jnp.sum(out.heating_rate)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), (
            f"∂(heating_rate)/∂T through stratosphere column must be "
            f"finite; got {g}.  Sign of iter-14/iter-19 AD-floor "
            f"regression — see commits 59407953 and codex iter-18 "
            f"survivor finding."
        )
        # Non-trivial gradient — guard against constant-fold collapse.
        assert jnp.abs(g).max() > 0.0


class TestCloudKwargsHelper:
    """Iter-17: ``CloudProperties.to_rrtmg_kwargs`` must NOT include
    cloud_fraction.  Pinning this prevents the
    cf²-double-discount bug (commit 4c9591bb, lost in AIMIP-#312
    merge, restored iter-15/16) from resurfacing.

    Reference impact (from the original fix commit message):
    - cf = 0.6: −59 W/m² OSR (radiative bias)
    - cf = 0.3: −113 W/m² OSR
    """

    def test_kwargs_excludes_cloud_fraction(self):
        from legoesm.atmosphere.physics.clouds.cloud_fraction import (
            CloudProperties,
        )
        shape = (4, 10)
        props = CloudProperties(
            cloud_fraction=jnp.full(shape, 0.5),
            lwp=jnp.full(shape, 1e-3),
            iwp=jnp.full(shape, 0.5e-3),
            r_eff_liq=jnp.full(shape, 1e-5),
            r_eff_ice=jnp.full(shape, 2e-5),
        )
        kwargs = props.to_rrtmg_kwargs()
        # cloud_fraction MUST NOT be in the kwargs.  RRTMG already
        # discounts by cf via the grid-mean LWP that comes through
        # cloud_path_liq.
        assert "cloud_fraction" not in kwargs, (
            "to_rrtmg_kwargs must NOT include cloud_fraction — passing "
            "it together with grid-mean LWP double-counts the cf "
            "discount (commit 4c9591bb; -59..-113 W/m² OSR bias)."
        )
        # The legitimate kwargs are present.
        for key in ("cloud_path_liq", "cloud_path_ice",
                    "cloud_r_eff_liq", "cloud_r_eff_ice"):
            assert key in kwargs, f"missing {key} in to_rrtmg_kwargs output"


class TestHeatingRateSign:
    """Iter-13: regression guard for the sign-inverted heating rate bug.

    Commit 0be22f0f (2026-05-22) fixed a sign error in
    ``compute_heating_rate`` that was driving thermal runaway in long
    AMIP integrations (T̄ 261 → 293 K over 120 days, NaN blowup at
    day 125).  The fix was lost when the AIMIP-#312 merge reverted
    ``two_stream.py``; iter-13 restores it.

    Pin the sign here so any future regression is caught immediately
    instead of after weeks of unstable runs.
    """

    def test_free_tropospheric_LW_cools(self):
        """Free troposphere (300-800 hPa) must show LW *cooling*
        (negative heating rate) for a US-Std-like warm surface column.
        Pre-fix bug: this region showed +2..+5 K/day heating instead."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 20
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 290.0, nlev)[None, :]  # TOA-first
        sfc_T = jnp.array([300.0])
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.array([0.5])

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        p_hPa = np.asarray(p_full)[0] / 100.0
        hr_K_per_day = np.asarray(out.lw_heating_rate)[0] * 86400.0
        # Free-tropospheric mask: 300-800 hPa.
        mask = (p_hPa > 300.0) & (p_hPa < 800.0)
        assert mask.any(), "test setup error: no levels in 300-800 hPa"
        # All free-tropospheric levels must show LW cooling.
        assert (hr_K_per_day[mask] < 0).all(), (
            f"Free-tropospheric LW heating rates must all be negative "
            f"(LW cooling); got {hr_K_per_day[mask]} K/day at "
            f"p={p_hPa[mask]} hPa.  Sign-inverted heating-rate bug "
            f"regression — see commit 0be22f0f."
        )

    def test_lw_cooling_finite_across_q_v_range(self):
        """iter-49: ``solve_columns`` must produce finite-and-sensible
        column-mean LW cooling across a wide ``q_v`` range (1e-5 to
        1e-2 kg/kg).  The exact magnitude / sign per layer depends on
        the optical-depth regime (thin → direct cooling to space,
        thick → mostly local emission-absorption balance), so we only
        pin: every column-mean LW heating rate must be negative
        (cooling).

        Pre-iter-13 sign fix this would have shown positive column-
        mean across the entire q_v range — a louder regression than
        the original single-profile check (test_free_tropospheric_LW_cools).
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 20
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 290.0, nlev)[None, :]
        sfc_T = jnp.array([300.0])
        cos_z = jnp.array([0.5])

        for q_v_value in [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]:
            q_v = jnp.full((ncol, nlev), q_v_value)
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z,
            )
            mean_hr = float(jnp.mean(out.lw_heating_rate))
            assert mean_hr < 0, (
                f"column-mean LW heating must be negative (cooling) at "
                f"q_v={q_v_value}; got {mean_hr:.4e} K/s.  Pre-iter-13 "
                f"sign fix this would be positive across the range."
            )

    def test_clear_sky_SW_heats(self):
        """Clear-sky daytime SW must heat the atmosphere (non-negative
        heating rate everywhere).  Pre-fix bug had SW *cooling* layers."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 20
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 290.0, nlev)[None, :]
        sfc_T = jnp.array([300.0])
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.array([0.5])  # sun overhead-ish

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        sw_hr = np.asarray(out.sw_heating_rate)[0]
        assert (sw_hr >= 0).all(), (
            f"Clear-sky SW heating must be non-negative; got {sw_hr}.  "
            f"Sign-inverted heating-rate bug regression — see commit "
            f"0be22f0f."
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
