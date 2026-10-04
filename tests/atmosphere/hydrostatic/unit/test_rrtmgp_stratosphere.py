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

Full chronology in ``docs/science/specs/rrtmgp.md``.
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
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
    monochromatic_two_stream,
    rte_utils,
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
    from tests.legoesm_paths import legoesm_source_path
    from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
        lookup_gas_optics_longwave,
        lookup_volume_mixing_ratio,
        constants as optics_constants,
    )

    nc_path = legoesm_source_path(
        "atmosphere/physics/radiation/rrtmgp/optics"
        "/rrtmgp_data/rrtmgp-gas-lw-g128.nc"
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
        from tests.legoesm_paths import legoesm_source_path
        from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
            lookup_gas_optics_shortwave,
        )
        _, vmr_lib = lookup_vmr
        sw_path = legoesm_source_path(
            "atmosphere/physics/radiation/rrtmgp/optics"
            "/rrtmgp_data/rrtmgp-gas-sw-g112.nc"
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

        Iter-69 codex review fix: skip when ``jax_enable_x64`` is off,
        because JAX silently downcasts ``jnp.float64`` to fp32 in
        that mode and the test would compare fp32-against-fp32
        (trivial pass that would mask a real missing-cast bug).
        """
        if not jax.config.read("jax_enable_x64"):
            pytest.skip(
                "fp32-vs-fp64 cast test requires JAX_ENABLE_X64=1; "
                "without x64 enabled jnp.float64 silently downcasts to "
                "fp32, making the comparison trivial."
            )
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

    def test_solve_columns_subset_matches_full_columns_slice(self, lookup_vmr):
        """Iter-64: MPI column-shard correctness condition.

        ``solve_columns`` called on a column subset must produce
        bit-identical results to the corresponding slice of a full-
        batch call.  This is the formal MPI-correctness invariant:
        rank ``k`` of an N-rank MPI run holds columns ``[k*M : (k+1)*M]``
        of the global batch (where M = ncol_global / N), and its
        ``solve_columns(local_cols)`` must equal
        ``solve_columns(global_cols)[k*M : (k+1)*M]``.

        Stronger than iter-11's permutation-invariance because it
        rules out global-axis-dependent normalisations (e.g. an
        accidental ``mean(over_columns)`` term that drops with ncol).
        Combined with iter-11 permutation-invariance, this pins
        RRTMGP as embarrassingly parallel over the column axis on
        MPI / GPU / TPU.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        nlev = 8
        # Full batch of 6 columns, each with distinguishable inputs.
        ncol_full = 6
        T_full = jnp.stack([
            jnp.linspace(220.0 + 4.0 * k, 290.0 + 4.0 * k, nlev)
            for k in range(ncol_full)
        ])
        p_half_full = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
            (ncol_full, nlev + 1),
        )
        p_full_full = 0.5 * (p_half_full[:, :-1] + p_half_full[:, 1:])
        sfc_T_full = jnp.array([293.0 + 2.0 * k for k in range(ncol_full)])
        q_v_full = jnp.full((ncol_full, nlev), 5e-3)
        cos_z_full = jnp.array([0.4 + 0.05 * k for k in range(ncol_full)])

        out_full = solver.solve_columns(
            T=T_full, p_full=p_full_full, p_half=p_half_full,
            sfc_temperature=sfc_T_full, q_v=q_v_full, cos_zenith=cos_z_full,
        )

        # Now call again on the FIRST HALF (cols 0..2) — simulates
        # rank 0 of a 2-rank MPI run.
        sl = slice(0, 3)
        out_subset = solver.solve_columns(
            T=T_full[sl], p_full=p_full_full[sl], p_half=p_half_full[sl],
            sfc_temperature=sfc_T_full[sl], q_v=q_v_full[sl],
            cos_zenith=cos_z_full[sl],
        )

        # Subset call must match slice of full call bit-for-bit.
        for name in ("lw_flux_up", "lw_flux_down", "sw_flux_up",
                     "sw_flux_down", "heating_rate"):
            a = np.asarray(getattr(out_full, name))[sl]
            b = np.asarray(getattr(out_subset, name))
            np.testing.assert_allclose(
                a, b, rtol=1e-12, atol=1e-12,
                err_msg=(
                    f"{name}: subset call differs from slice of full "
                    "call.  RRTMGP is leaking column-global state — "
                    "MPI ranks would compute inconsistent fluxes."
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
        # scan vs unrolled differ only by data movement, not arithmetic, so
        # agreement is round-off: tighten under x64, relax to a float32 quantum
        # under the default fp32 policy.  A genuine scan/loop divergence is
        # O(flux), caught by either bound.
        if jax.config.read("jax_enable_x64"):
            rtol = atol = 1e-10
        else:
            rtol, atol = 1e-5, 1e-6
        for name, a, b in (
            ("lw_flux_up",   out_loop.lw_flux_up,   out_scan.lw_flux_up),
            ("lw_flux_down", out_loop.lw_flux_down, out_scan.lw_flux_down),
            ("heating_rate", out_loop.heating_rate, out_scan.heating_rate),
        ):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=rtol, atol=atol,
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

    def test_jax_grad_finite_at_mesosphere_pressure_boundary(self):
        """Iter-65: stratosphere/mesosphere boundary AD safety.

        ``p_ref`` (gas-optics table) spans ``[1.005 Pa, 109663 Pa]``
        (≈ 0.01 hPa to 1100 hPa, ≈ surface to ~65 km).  Above 65 km
        we cross the lowest table entry and ``_pressure_interpolant``
        extrapolates via its log-space linear interpolant.  Because
        the column top is the dominant LW-cooling contributor for
        thin upper atmospheres, an AD-unsafe path here would leak
        NaN gradients into any training loop that uses an extended
        vertical extent (e.g. CRMs, mesospheric chemistry models).

        Test exercises a column whose top sits at 0.5 Pa — ~7 layers
        deep into the "below p_ref[-1] = 1.005 Pa" extrapolation
        zone — and pins ``jax.grad`` finite through the full forward
        pass.  Pre-iter-65 there was no test guarding this regime.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 10
        # Column extends from 0.5 Pa (mesosphere, ~75 km) down to
        # 1000 Pa (lower stratosphere, ~30 km).  Top half is below
        # ``p_ref[-1]`` and exercises the extrapolation path.
        p_half = jnp.broadcast_to(
            jnp.geomspace(0.5, 1.0e3, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = jnp.sqrt(p_half[:, :-1] * p_half[:, 1:])
        # Mesopause is ~190K, mid-stratosphere ~220K → linear range.
        T = jnp.linspace(190.0, 230.0, nlev)[None, :]
        sfc_T = jnp.array([230.0])
        # Mesospheric H2O is ~5 ppm (Brasseur–Solomon).
        q_v = jnp.full((ncol, nlev), 5e-6)
        cos_z = jnp.array([0.4])

        def loss(T_in):
            out = solver.solve_columns(
                T=T_in, p_full=p_full, p_half=p_half,
                sfc_temperature=sfc_T, q_v=q_v, cos_zenith=cos_z,
            )
            return jnp.sum(out.heating_rate)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), (
            f"∂(heating_rate)/∂T at mesospheric column top must be "
            f"finite; got {g}.  Likely an AD-unsafe extrapolation "
            f"in ``_pressure_interpolant`` for p < p_ref[-1] (= 1.005 Pa)."
        )
        # Non-trivial gradient.
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


class TestEnergyConservation:
    """Iter-67: column energy conservation invariant for RRTMGP.

    For any radiation scheme, the column-integrated heating rate
    (converted to a flux divergence with the layer's
    ``dp * c_p / g`` mass-times-specific-heat factor) must match
    the net flux convergence:

        F_net(sfc) - F_net(TOA) = ∫ heating_rate * c_p * dp / g

    Equivalently (dimensionally cleaner):

        ∑(hr * dp) = (g / c_p) · (F_net_TOA - F_net_sfc)

    This is the discrete form of ``hr = g/c_p · ∂F_net/∂p``.

    Pre-iter-67 only ``gray_radiation`` had this pin in
    ``test_radiation.py::test_energy_conservation``; the
    sign-fix iter-13 + iter-17 cf² fix make this invariant a
    high-value RRTMGP regression guard because any future change
    to ``compute_heating_rate`` that breaks the relation would
    silently leak energy.
    """

    def test_column_flux_divergence_matches_heating_rate(self):
        """Combined LW+SW: column flux convergence == column
        integrated heating × c_p × dp / g.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
        from legoesm import constants as const

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 2, 16
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(
            jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev)
        )
        sfc_T = jnp.array([298.0, 295.0])
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.array([0.6, 0.4])

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )

        # Convention: index 0 = TOA, index -1 = surface (per
        # ``solve_columns`` docstring).  Output flux arrays are
        # interface-valued shape (ncol, nlev+1).
        # Net flux is positive downward.
        F_net_toa = (
            out.lw_flux_down[:, 0] - out.lw_flux_up[:, 0]
            + out.sw_flux_down[:, 0] - out.sw_flux_up[:, 0]
        )
        F_net_sfc = (
            out.lw_flux_down[:, -1] - out.lw_flux_up[:, -1]
            + out.sw_flux_down[:, -1] - out.sw_flux_up[:, -1]
        )

        # ``heating_rate`` is K/s, shape (ncol, nlev).  Use SAME
        # ordering convention (TOA-first) as the flux arrays.  In
        # TOA-first ordering the layer-thickness ``dp`` increases
        # with index (top thin, bottom thick).
        dp = p_half[:, 1:] - p_half[:, :-1]  # positive, TOA-first

        # ∑(hr * dp) should equal (g / c_p) · (F_net_TOA - F_net_sfc).
        lhs = jnp.sum(out.heating_rate * dp, axis=1)
        rhs = (const.g / const.c_pd) * (F_net_toa - F_net_sfc)

        np.testing.assert_allclose(
            np.asarray(lhs), np.asarray(rhs),
            rtol=5.0e-5, atol=1.0e-6,
            err_msg=(
                f"Column flux divergence does NOT match column-"
                f"integrated heating-rate × dp.  This is a hard "
                f"energy-conservation invariant; failure indicates "
                f"a regression in ``compute_heating_rate`` (iter-13 "
                f"sign-fix territory) or in the flux-net interface "
                f"convention."
            ),
        )

    def test_lw_only_flux_divergence_matches_lw_heating(self):
        """LW-only: same invariant for the LW heating-rate branch."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
        from legoesm import constants as const

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        ncol, nlev = 1, 12
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 290.0, nlev)[None, :]
        sfc_T = jnp.array([298.0])
        q_v = jnp.full((ncol, nlev), 5e-3)
        # Nighttime → SW path zero, isolating LW.
        cos_z = jnp.array([-0.5])

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )

        F_lw_net_toa = out.lw_flux_down[:, 0] - out.lw_flux_up[:, 0]
        F_lw_net_sfc = out.lw_flux_down[:, -1] - out.lw_flux_up[:, -1]
        dp = p_half[:, 1:] - p_half[:, :-1]
        lhs = jnp.sum(out.lw_heating_rate * dp, axis=1)
        rhs = (const.g / const.c_pd) * (F_lw_net_toa - F_lw_net_sfc)

        np.testing.assert_allclose(
            np.asarray(lhs), np.asarray(rhs),
            rtol=5.0e-5, atol=1.0e-6,
            err_msg=(
                "LW column flux divergence does NOT match LW "
                "column-integrated heating rate.  Sign-fix iter-13 "
                "regression."
            ),
        )


class TestCloudPath:
    """Iter-68: physical-sign + AD pins for the cloud path.

    Pre-iter-68 cloud coverage was:
      - ``TestCloudKwargsHelper`` (iter-17): structural — to_rrtmg_kwargs
        does NOT include cloud_fraction (cf²-double-discount guard).
      - ``test_iter37_include_clouds_flag_changes_flux``: on-vs-off
        produces different flux.
      - ``test_iter41_clear_sky_optics_raises_on_direct_cloud_call``:
        include_clouds=False + cloud kwarg → ValueError.

    No test pinned the **sign** of the cloud effect or AD safety
    through the cloud-path inputs.  A future refactor that flipped
    sign-of-cloud-emission would pass all 3 existing tests yet
    produce climates dominated by cloud-greenhouse cooling.

    | test | invariant |
    |---|---|
    | ``test_cloud_path_zero_matches_no_cloud`` | LWP=zeros ≡ LWP=None bit-for-bit |
    | ``test_low_cloud_reduces_surface_sw`` | sign: low cloud → reduced surface SW |
    | ``test_low_cloud_increases_surface_lw_down`` | sign: low cloud → boosted surface LW down (greenhouse) |
    | ``test_cloud_path_liq_differentiable`` | ``jax.grad`` w.r.t. cloud_path_liq finite + sign-correct |
    """

    @staticmethod
    def _base_inputs(ncol=1, nlev=16):
        """Daylit tropical column ready to host a low cloud."""
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 295.0, nlev)[None, :]
        sfc_T = jnp.array([298.0] * ncol)
        q_v = jnp.full((ncol, nlev), 8e-3)
        cos_z = jnp.array([0.7] * ncol)
        return T, p_full, p_half, sfc_T, q_v, cos_z

    @staticmethod
    def _low_cloud_path(T_shape, nlev):
        """Liquid water path 0.05 kg/m² in the lowest 3 layers
        (700-1000 hPa).  Realistic stratocumulus-deck LWP."""
        path = jnp.zeros(T_shape)
        path = path.at[:, -3:].set(0.05)
        return path

    def test_cloud_path_zero_matches_no_cloud(self):
        """LWP=zeros must produce bit-identical fluxes to LWP=None.
        Catches an unintended bias in the include_clouds branch
        (e.g. a non-trivial cloud overhead even at zero LWP).

        Requires ``JAX_ENABLE_X64=1``: the ``rtol=1e-10`` "bit-identical"
        claim is an fp64 property.  In the default fp32 policy the
        zero-cloud branch reorders the gas+cloud optics combine vs the
        no-cloud path, so the two agree only to ~1e-6 (a float32 quantum) —
        a false failure, NOT a real branch leak (verified: passes under x64).
        Mirror the iter-69 skip-guard used by the fp32-vs-fp64 cast test
        above."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip(
                "zero-cloud bit-identity check requires JAX_ENABLE_X64=1; "
                "the 1e-10 tolerance is unachievable under the fp32 policy "
                "where the cloud-combine reorders float ops (~1e-6 noise)."
            )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=True)
        )
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()

        out_none = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
            cloud_path_liq=None, cloud_path_ice=None,
        )
        zero_path = jnp.zeros(T.shape)
        out_zero = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
            cloud_path_liq=zero_path,
            cloud_r_eff_liq=jnp.full(T.shape, 1.0e-5),
            cloud_path_ice=zero_path,
            cloud_r_eff_ice=jnp.full(T.shape, 2.0e-5),
        )
        for name in ("sw_flux_up", "sw_flux_down", "lw_flux_up",
                     "lw_flux_down", "heating_rate"):
            a = np.asarray(getattr(out_none, name))
            b = np.asarray(getattr(out_zero, name))
            np.testing.assert_allclose(
                a, b, rtol=1.0e-10, atol=1.0e-9,
                err_msg=(
                    f"{name}: LWP=None and LWP=zeros must give bit-"
                    f"identical fluxes (zero-cloud branch leak)."
                ),
            )

    def test_low_cloud_reduces_surface_sw(self):
        """Low warm cloud (LWP=0.05 kg/m² in lowest 3 layers) must
        REDUCE the surface downwelling SW flux relative to a
        clear-sky column.  This is the elementary cloud-albedo
        effect.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=True)
        )
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()
        cloud_path_liq = self._low_cloud_path(T.shape, T.shape[1])
        r_eff_liq = jnp.full(T.shape, 1.0e-5)

        out_clear = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        out_cloud = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
            cloud_path_liq=cloud_path_liq, cloud_r_eff_liq=r_eff_liq,
        )

        # Surface = layer index -1 (TOA-first convention).
        sw_down_clear = float(out_clear.sw_flux_down[0, -1])
        sw_down_cloud = float(out_cloud.sw_flux_down[0, -1])
        assert sw_down_cloud < sw_down_clear, (
            f"Low warm cloud must reduce surface SW down; got "
            f"clear={sw_down_clear:.1f}, cloud={sw_down_cloud:.1f} W/m²."
        )
        reduction = sw_down_clear - sw_down_cloud
        assert 10.0 < reduction < 800.0, (
            f"Surface SW reduction {reduction:.1f} W/m² is outside "
            f"the plausible [10, 800] W/m² band for stratocumulus-"
            f"deck LWP=0.05 kg/m².  Indicates broken cloud-SW optics."
        )

    def test_low_cloud_increases_surface_lw_down(self):
        """Low warm cloud must INCREASE the surface downwelling LW
        flux (the cloud emits Planck radiation at its temperature ≈
        warm-cloud-base T).  This is the elementary cloud-greenhouse
        effect.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=True)
        )
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()
        cloud_path_liq = self._low_cloud_path(T.shape, T.shape[1])
        r_eff_liq = jnp.full(T.shape, 1.0e-5)

        out_clear = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        out_cloud = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
            cloud_path_liq=cloud_path_liq, cloud_r_eff_liq=r_eff_liq,
        )

        lw_down_clear = float(out_clear.lw_flux_down[0, -1])
        lw_down_cloud = float(out_cloud.lw_flux_down[0, -1])
        assert lw_down_cloud > lw_down_clear, (
            f"Low warm cloud must increase surface LW down "
            f"(greenhouse); got clear={lw_down_clear:.1f}, "
            f"cloud={lw_down_cloud:.1f} W/m²."
        )
        boost = lw_down_cloud - lw_down_clear
        assert 5.0 < boost < 200.0, (
            f"Surface LW-down boost {boost:.1f} W/m² is outside the "
            f"plausible [5, 200] W/m² band for warm stratocumulus.  "
            f"Indicates broken cloud-LW optics (sign flip, missing "
            f"Planck source, cf²-double-discount regression)."
        )

    def test_cloud_path_liq_differentiable(self):
        """``jax.grad`` w.r.t. ``cloud_path_liq`` must be finite +
        sign-correct (negative for surface SW, positive for surface
        LW down).  Catches AD-unsafe-floor regressions in cloud-
        optics mixing inside ``cloud_optics.compute_lw/sw_optical_
        properties``.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=True)
        )
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()
        r_eff_liq = jnp.full(T.shape, 1.0e-5)

        def sfc_sw_loss(lwp):
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z,
                cloud_path_liq=lwp, cloud_r_eff_liq=r_eff_liq,
            )
            return jnp.sum(out.sw_flux_down[:, -1])

        def sfc_lw_loss(lwp):
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z,
                cloud_path_liq=lwp, cloud_r_eff_liq=r_eff_liq,
            )
            return jnp.sum(out.lw_flux_down[:, -1])

        lwp0 = self._low_cloud_path(T.shape, T.shape[1])
        g_sw = jax.grad(sfc_sw_loss)(lwp0)
        g_lw = jax.grad(sfc_lw_loss)(lwp0)
        assert jnp.all(jnp.isfinite(g_sw)), (
            f"∂(surface SW)/∂(cloud_path_liq) must be finite; got {g_sw}"
        )
        assert jnp.all(jnp.isfinite(g_lw)), (
            f"∂(surface LW down)/∂(cloud_path_liq) must be finite; got {g_lw}"
        )
        # Active cloud layers (last 3) must have non-zero gradient.
        assert jnp.abs(g_sw[:, -3:]).max() > 0.0, (
            "Active-cloud-layer ∂(SW)/∂(LWP) is identically zero — "
            "broken AD plumbing through cloud-SW optics."
        )
        assert jnp.abs(g_lw[:, -3:]).max() > 0.0, (
            "Active-cloud-layer ∂(LW down)/∂(LWP) is identically zero — "
            "broken AD plumbing through cloud-LW optics."
        )

    def test_high_ice_cloud_increases_outgoing_lw(self):
        """Iter-70: cirrus-deck IWP=0.005 kg/m² in the upper troposphere
        must REDUCE outgoing LW (the cold cirrus emits less than the
        warm surface below; classic ice-cloud greenhouse).  Catches
        sign-flip regressions in the cloud_path_ice / r_eff_ice branch
        of ``cloud_optics.compute_lw_optical_properties``.

        Codex iter-69 Q5 follow-up: ``test_low_cloud_*`` only exercise
        cloud_path_liq.  This test extends the sign-pin to
        cloud_path_ice.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=True)
        )
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()
        # Cirrus IWP=0.005 kg/m² in the top 3 layers (upper trop).
        iwp = jnp.zeros(T.shape).at[:, :3].set(5.0e-3)
        r_eff_ice = jnp.full(T.shape, 2.5e-5)

        out_clear = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        out_ice = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
            cloud_path_ice=iwp, cloud_r_eff_ice=r_eff_ice,
        )

        # TOA upward LW: index 0 (TOA-first convention).
        olr_clear = float(out_clear.lw_flux_up[0, 0])
        olr_ice = float(out_ice.lw_flux_up[0, 0])
        assert olr_ice < olr_clear, (
            f"High cold cirrus must reduce outgoing LW at TOA "
            f"(emits at cold T_top instead of warm surface); got "
            f"clear={olr_clear:.1f}, ice={olr_ice:.1f} W/m²."
        )
        reduction = olr_clear - olr_ice
        assert 2.0 < reduction < 100.0, (
            f"TOA OLR reduction {reduction:.1f} W/m² is outside "
            f"the plausible [2, 100] W/m² band for cirrus "
            f"IWP=0.005 kg/m².  Indicates broken cloud-LW ice "
            f"optics."
        )

    def test_cloud_path_ice_differentiable(self):
        """Iter-70: ``jax.grad`` w.r.t. ``cloud_path_ice`` finite +
        sign-correct (negative for TOA outgoing LW since ice clouds
        reduce OLR).  Codex iter-69 Q5 follow-up.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=True)
        )
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()
        r_eff_ice = jnp.full(T.shape, 2.5e-5)

        def toa_olr_loss(iwp):
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z,
                cloud_path_ice=iwp, cloud_r_eff_ice=r_eff_ice,
            )
            return jnp.sum(out.lw_flux_up[:, 0])

        iwp0 = jnp.zeros(T.shape).at[:, :3].set(5.0e-3)
        g = jax.grad(toa_olr_loss)(iwp0)
        assert jnp.all(jnp.isfinite(g)), (
            f"∂(TOA OLR)/∂(cloud_path_ice) must be finite; got {g}"
        )
        # Per-layer gradients can be mixed-sign (multi-layer cloud
        # systems redistribute optical depth — adding IWP at one
        # interior layer can boost the effective emission of layers
        # above/below).  Physically the invariant is the
        # **column-summed** gradient: increasing IWP throughout the
        # active cloud must reduce TOA OLR (cold cloud > warm
        # surface in Planck-source magnitude).
        col_grad = float(jnp.sum(g[:, :3]))
        assert col_grad < 0.0, (
            f"∑(∂(TOA OLR)/∂(cloud_path_ice)) over active-cloud "
            f"layers must be negative; got {col_grad:.1f} W·m²/kg.  "
            f"Per-layer breakdown {g[:, :3]} — see test docstring "
            f"for why per-layer signs can be mixed."
        )
        # Magnitude check: gradient must be non-trivial (guards
        # against an AD constant-fold collapse to all-zeros).
        assert jnp.abs(g[:, :3]).max() > 1.0, (
            "Active-cloud-layer ∂(TOA OLR)/∂(IWP) is essentially "
            "zero — broken AD plumbing through cloud-LW ice optics."
        )


class TestAerosolPath:
    """Iter-66: regression guards for the SW aerosol path in ``solve_sw``.

    ``solve_sw`` accepts an optional ``aerosol_optical_depth`` array that
    is combined band-uniformly with the gas+cloud optical depth via the
    AD-safe SW optical-property mixing (restored from commit 59407953,
    iter-14).  Pre-iter-66 the only aerosol coverage was a cache-key
    test in ``test_radiation.py`` — no value, sign, or differentiability
    pin.

    | test | invariant |
    |---|---|
    | ``test_aerosol_zero_matches_no_aerosol`` | AOD=zeros ≡ AOD=None bit-for-bit |
    | ``test_aerosol_reduces_toa_sw_down`` | sign: scattering aerosol decreases surface SW |
    | ``test_aerosol_path_differentiable`` | ``jax.grad`` w.r.t. AOD is finite |
    """

    @staticmethod
    def _base_inputs(ncol=1, nlev=12):
        """Build a daylit tropical-like column."""
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.linspace(220.0, 295.0, nlev)[None, :]
        sfc_T = jnp.array([298.0] * ncol)
        q_v = jnp.full((ncol, nlev), 8e-3)
        cos_z = jnp.array([0.7] * ncol)  # ~45° solar elevation
        return T, p_full, p_half, sfc_T, q_v, cos_z

    def test_aerosol_zero_matches_no_aerosol(self):
        """``aerosol_optical_depth = zeros`` must produce bit-identical
        fluxes to ``aerosol_optical_depth = None``.  The two paths
        branch on a Python ``if aerosol_optical_depth is not None``
        check; if the zero-AOD branch picks up a non-trivial code
        path (e.g. an unintended bias from ``maximum(tau, 1e-12)``
        applied to gas tau only in one branch), this test catches it.

        Requires ``JAX_ENABLE_X64=1`` (mirrors the zero-cloud bit-identity
        twin): the leak this guards against is O(1e-12), but under the fp32
        policy the AOD=zeros branch reorders the gas+aerosol optics combine
        vs the AOD=None path, so the two agree only to ~1e-6 (a float32
        quantum) — relaxing the bound would make the 1e-12 leak check vacuous.
        """
        if not jax.config.read("jax_enable_x64"):
            pytest.skip(
                "zero-aerosol bit-identity check requires JAX_ENABLE_X64=1; "
                "the 1e-12 tolerance is unachievable under the fp32 policy "
                "where the aerosol-combine reorders float ops (~1e-6 noise)."
            )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()

        out_none = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z, aerosol_optical_depth=None,
        )
        out_zero = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
            aerosol_optical_depth=jnp.zeros_like(T),
        )
        for name in ("sw_flux_up", "sw_flux_down", "lw_flux_up",
                     "heating_rate"):
            a = np.asarray(getattr(out_none, name))
            b = np.asarray(getattr(out_zero, name))
            np.testing.assert_allclose(
                a, b, rtol=1e-12, atol=1e-12,
                err_msg=(
                    f"{name}: AOD=None and AOD=zeros must give "
                    f"bit-identical fluxes (zero-aerosol branch leak)."
                ),
            )

    def test_aerosol_reduces_toa_sw_down(self):
        """Adding a non-trivial AOD (τ=0.5 throughout column, SSA<1)
        must reduce the surface downwelling SW flux relative to a
        clear-aerosol-free column.  This is the elementary physical
        sign of scattering+absorbing aerosol.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()

        # Per-layer AOD totalling τ_aer_col ≈ 0.5 (mid-range thick haze).
        aod = jnp.full(T.shape, 0.5 / T.shape[1])

        out_clear = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z, aerosol_optical_depth=None,
        )
        out_aero = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z, aerosol_optical_depth=aod,
        )

        # Surface = interface index -1.  Flux arrays are
        # interface-dimensioned (ncol, nlev+1) in TOA-first convention;
        # see ``solve_columns`` docstring.  Iter-69 codex review fix:
        # previously used ``argmax(p_full)`` which is a full-level
        # index in [0, nlev-1] and indexes the LEVEL ABOVE SURFACE
        # rather than the surface interface itself.
        sw_down_clear = float(out_clear.sw_flux_down[0, -1])
        sw_down_aero = float(out_aero.sw_flux_down[0, -1])

        assert sw_down_aero < sw_down_clear, (
            f"Aerosol (τ=0.5, SSA=0.93) should reduce surface SW "
            f"downward flux; got clear={sw_down_clear:.2f}, "
            f"aero={sw_down_aero:.2f} W/m²."
        )
        # Plausibility: a τ=0.5 absorbing column should reduce surface
        # SW by ~10-50 W/m² (depending on solar angle).  Sanity check.
        reduction = sw_down_clear - sw_down_aero
        assert 1.0 < reduction < 200.0, (
            f"Surface SW reduction {reduction:.1f} W/m² is outside "
            f"the plausible [1, 200] W/m² band for τ=0.5 mid-range "
            f"haze.  Indicates broken aerosol+gas optical-property "
            f"mixing in solve_sw."
        )

    def test_aerosol_path_differentiable(self):
        """``jax.grad`` w.r.t. ``aerosol_optical_depth`` must be finite.

        The aerosol path inside ``solve_sw`` (lines 462-497) uses
        ``safe_divide(w_num, tau_tot)`` and ``safe_divide(g_num,
        g_denom)`` with the AD-safe ``safe_divide`` helper.  This
        test pins the chain — if a future refactor reverts to
        ``a / jnp.maximum(b, eps)``, the ``-a/b**2`` VJP overflows
        for tau_tot near the 1e-12 floor and gradients leak NaN.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
        T, p_full, p_half, sfc_T, q_v, cos_z = self._base_inputs()

        def loss(aod):
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z, aerosol_optical_depth=aod,
            )
            return jnp.sum(out.sw_flux_down)

        aod0 = jnp.full(T.shape, 0.1 / T.shape[1])
        g = jax.grad(loss)(aod0)
        assert jnp.all(jnp.isfinite(g)), (
            f"∂(sum sw_flux_down)/∂(aerosol_optical_depth) must be "
            f"finite; got {g}.  Likely an AD-unsafe ``maximum(tau, "
            f"eps)`` divide in solve_sw aerosol-mixing block."
        )
        # Adding AOD reduces sw_flux_down, so the gradient is negative.
        assert jnp.sum(g) < 0.0, (
            f"d(sum sw_flux_down)/d(AOD) should be negative for a "
            f"scattering aerosol; got {float(jnp.sum(g))}."
        )


class TestRteRecurrenceScanEquivalence:
    """Iter-71: pin the GPU(scan) vs CPU(for-loop) equivalence of the
    vertical-recurrence engine that drives the monochromatic two-stream
    solver.

    ``rte_utils.recurrent_op_with_halos`` dispatches to two distinct code
    paths: ``lax.scan`` (auto-selected on GPU/TPU, where the unrolled
    ``dynamic_update_slice`` chain of the for-loop is slow and the scan
    lowers to a single fused kernel) and a Python ``for`` loop
    (auto-selected on CPU/Metal).  All four Shonk-Hogan recurrences of the
    monochromatic LW/SW solver (direct-beam, albedo, upward-emission,
    downward-flux) flow through it, exposed by the ``rrtmgp_use_scan``
    config knob.

    ``test_production_blockers`` already pins that the knob *routes*
    correctly; what was untested is that the two paths return the **same
    numbers**.  If they diverge, a GPU run silently produces different
    fluxes (and different gradients) than the CPU reference — a
    cross-backend break (CLAUDE.md: "Cross-backend: dtype, x64,
    unsupported kernels, comm semantics") that no flux norm or
    integration test would localise.  The paths run identical sequential
    arithmetic in identical order (differing only by a ``moveaxis``
    transpose, no reductions), so they must agree to round-off.  Forced
    ``use_scan=`` overrides the platform auto-pick so the scan path is
    exercised on this CPU test host.
    """

    @staticmethod
    def _tol():
        # The paths differ only by data movement, not arithmetic, so
        # agreement is essentially round-off — tighten under x64, relax
        # for an x32 conftest.  A genuine divergence is O(flux), caught
        # by either bound.
        if jax.config.read("jax_enable_x64"):
            return dict(rtol=1e-11, atol=1e-12)
        return dict(rtol=1e-5, atol=1e-6)

    @staticmethod
    def _albedo_like_op(carry, r_diff, t_diff):
        """A Shonk-Hogan-style geometric-series recurrence (mirrors the
        real ``albedo_op`` in ``monochromatic_two_stream``).  Non-linear
        in the carry, so a path that mishandled carry threading or scan
        direction would diverge immediately."""
        denom = 1.0 - r_diff * carry
        out = r_diff + t_diff ** 2 * carry / denom
        return out, out

    @staticmethod
    def _halo_inputs(seed, d0=2, d1=3, nz=14):
        """3D inputs carrying one halo layer at each z end — the
        convention ``recurrent_op_with_halos`` strips then re-pads.
        Ranges keep ``1 - r_diff*carry`` well away from zero."""
        rng = np.random.default_rng(seed)
        return (
            {
                "r_diff": jnp.asarray(rng.uniform(0.05, 0.40, (d0, d1, nz))),
                "t_diff": jnp.asarray(rng.uniform(0.30, 0.55, (d0, d1, nz))),
            },
            jnp.asarray(rng.uniform(0.0, 0.9, (d0, d1))),
        )

    def test_recurrent_op_with_halos_scan_equals_forloop_forward(self):
        inputs, init = self._halo_inputs(71)
        carry_s, out_s = rte_utils.recurrent_op_with_halos(
            self._albedo_like_op, init, inputs, forward=True, use_scan=True,
        )
        carry_f, out_f = rte_utils.recurrent_op_with_halos(
            self._albedo_like_op, init, inputs, forward=True, use_scan=False,
        )
        np.testing.assert_allclose(
            np.asarray(out_s), np.asarray(out_f), **self._tol(),
            err_msg="forward recurrence: scan vs for-loop output diverged",
        )
        np.testing.assert_allclose(
            np.asarray(carry_s), np.asarray(carry_f), **self._tol(),
            err_msg="forward recurrence: scan vs for-loop carry diverged",
        )

    def test_recurrent_op_with_halos_scan_equals_forloop_reverse(self):
        inputs, init = self._halo_inputs(72)
        carry_s, out_s = rte_utils.recurrent_op_with_halos(
            self._albedo_like_op, init, inputs, forward=False, use_scan=True,
        )
        carry_f, out_f = rte_utils.recurrent_op_with_halos(
            self._albedo_like_op, init, inputs, forward=False, use_scan=False,
        )
        np.testing.assert_allclose(
            np.asarray(out_s), np.asarray(out_f), **self._tol(),
            err_msg="reverse recurrence: scan vs for-loop output diverged",
        )
        np.testing.assert_allclose(
            np.asarray(carry_s), np.asarray(carry_f), **self._tol(),
            err_msg="reverse recurrence: scan vs for-loop carry diverged",
        )

    def test_recurrent_op_with_halos_grad_scan_equals_forloop(self):
        """Reverse-mode AD must agree between the two paths: training on
        GPU (scan) must see the same gradients as the CPU reference, else
        a differentiable run optimises against a backend-dependent
        objective."""
        inputs, init = self._halo_inputs(73)

        def loss(r_diff, use_scan):
            _, out = rte_utils.recurrent_op_with_halos(
                self._albedo_like_op, init,
                {"r_diff": r_diff, "t_diff": inputs["t_diff"]},
                forward=True, use_scan=use_scan,
            )
            return jnp.sum(out ** 2)

        g_s = jax.grad(lambda r: loss(r, True))(inputs["r_diff"])
        g_f = jax.grad(lambda r: loss(r, False))(inputs["r_diff"])
        assert jnp.all(jnp.isfinite(g_s)), f"scan-path grad not finite: {g_s}"
        assert jnp.all(jnp.isfinite(g_f)), f"loop-path grad not finite: {g_f}"
        np.testing.assert_allclose(
            np.asarray(g_s), np.asarray(g_f), **self._tol(),
            err_msg="reverse-mode AD: scan vs for-loop gradient diverged",
        )

    @staticmethod
    def _lw_inputs(seed, d0=2, d1=3, nz=14):
        rng = np.random.default_rng(seed)
        sh3, sh2 = (d0, d1, nz), (d0, d1)
        return dict(
            t_diff=jnp.asarray(rng.uniform(0.30, 0.60, sh3)),
            r_diff=jnp.asarray(rng.uniform(0.05, 0.30, sh3)),
            src_up=jnp.asarray(rng.uniform(0.5, 2.0, sh3)),
            src_down=jnp.asarray(rng.uniform(0.5, 2.0, sh3)),
            toa_flux_down=jnp.asarray(rng.uniform(0.0, 1.0, sh2)),
            sfc_src=jnp.asarray(rng.uniform(1.0, 3.0, sh2)),
            sfc_emissivity=jnp.asarray(rng.uniform(0.80, 0.99, sh2)),
        )

    def test_lw_transport_scan_equals_forloop(self):
        """End-to-end LW: all three of ``_solve_rte_2stream``'s
        recurrences (albedo, emission, downward-flux) must give identical
        ``flux_up`` / ``flux_down`` / ``flux_net`` on both paths."""
        kw = self._lw_inputs(74)
        f_scan = monochromatic_two_stream.lw_transport(**kw, use_scan=True)
        f_loop = monochromatic_two_stream.lw_transport(**kw, use_scan=False)
        for key in ("flux_up", "flux_down", "flux_net"):
            np.testing.assert_allclose(
                np.asarray(f_scan[key]), np.asarray(f_loop[key]), **self._tol(),
                err_msg=f"lw_transport scan vs for-loop diverged in {key!r}",
            )

    @staticmethod
    def _sw_inputs(seed, d0=2, d1=3, nz=14):
        rng = np.random.default_rng(seed)
        sh3, sh2 = (d0, d1, nz), (d0, d1)
        return dict(
            t_diff=jnp.asarray(rng.uniform(0.30, 0.60, sh3)),
            r_diff=jnp.asarray(rng.uniform(0.05, 0.30, sh3)),
            src_up=jnp.asarray(rng.uniform(0.0, 1.0, sh3)),
            src_down=jnp.asarray(rng.uniform(0.0, 1.0, sh3)),
            sfc_src=jnp.asarray(rng.uniform(0.0, 1.0, sh2)),
            sfc_albedo=jnp.asarray(rng.uniform(0.05, 0.30, sh2)),
            flux_down_dir=jnp.asarray(rng.uniform(0.0, 10.0, sh3)),
        )

    def test_sw_transport_scan_equals_forloop(self):
        """End-to-end SW (including the direct-beam ``flux_down_dir`` add)
        must be path-independent."""
        kw = self._sw_inputs(75)
        f_scan = monochromatic_two_stream.sw_transport(**kw, use_scan=True)
        f_loop = monochromatic_two_stream.sw_transport(**kw, use_scan=False)
        for key in ("flux_up", "flux_down", "flux_net"):
            np.testing.assert_allclose(
                np.asarray(f_scan[key]), np.asarray(f_loop[key]), **self._tol(),
                err_msg=f"sw_transport scan vs for-loop diverged in {key!r}",
            )


class TestSolverCompilationStability:
    """Iter-72: pin that ``RRTMGP.solve_columns`` is XLA-compilation
    stable — recompiling only on genuine shape changes, never on
    value-only changes.  This is the single most important property for
    GPU throughput.

    The physics pipeline runs the solver under ``jax.jit``
    (``integration.py`` / ``physics_pipeline.py``).  A function that
    re-traces whenever input *values* change — rather than only when
    *shapes* change — recompiles on every step; on a GPU that compile
    dwarfs the kernel runtime and destroys scaling.  Value-dependent
    retraces come from Python control flow on traced values, non-static
    config leaking into the trace, or unstable shapes (CLAUDE.md: "Perf
    regression: retrace, host callbacks, scatters, sharding, Python
    loops").

    Mechanism: a list cell incremented inside the jitted body counts
    *traces* — JAX runs the Python body once per compile, and a cache hit
    does not re-run it.  Contract pinned here: two same-shape calls with
    different data compile exactly once; only an ``nlev`` change compiles
    again.  A future edit that introduces value-dependent retracing trips
    the first assertion.
    """

    @staticmethod
    def _inputs(ncol=2, nlev=16, *, seed=0):
        rng = np.random.default_rng(seed)
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        dtype = p_full.dtype
        # Perturb T per-call so successive same-shape calls carry genuinely
        # different data — a constant-folded no-op would otherwise mask a
        # retrace.  Range stays inside the table-supported band [220, 295].
        T = jnp.asarray(
            220.0 + 75.0 * rng.uniform(0.0, 1.0, (ncol, nlev)), dtype=dtype
        )
        sfc_T = jnp.full((ncol,), 298.0, dtype=dtype)
        q_v = jnp.full((ncol, nlev), 8e-3, dtype=dtype)
        cos_z = jnp.full((ncol,), 0.7, dtype=dtype)
        return T, p_full, p_half, sfc_T, q_v, cos_z

    def test_solve_columns_compiles_once_per_shape(self):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=False))
        n_traces = [0]

        @jax.jit
        def run(T, p_full, p_half, sfc_T, q_v, cos_z):
            n_traces[0] += 1  # once per trace (compile), not per call
            return solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z,
            )

        # Two same-shape calls with DIFFERENT data must share one compile.
        out_a = run(*self._inputs(seed=1))
        out_b = run(*self._inputs(seed=2))
        jax.block_until_ready(out_a.lw_flux_up)
        jax.block_until_ready(out_b.lw_flux_up)
        assert n_traces[0] == 1, (
            f"solve_columns retraced on a value-only change "
            f"({n_traces[0]} traces for two same-shape calls) — a "
            f"value-dependent retrace recompiles on every GPU step and "
            f"destroys scaling."
        )
        assert jnp.all(jnp.isfinite(out_a.lw_flux_up))
        # Outputs must genuinely differ, else the 2nd call was a
        # constant-folded no-op that could hide a retrace.
        assert not bool(jnp.allclose(out_a.lw_flux_up, out_b.lw_flux_up)), (
            "two different-T columns produced identical LW flux — inputs "
            "were not actually distinct, so the no-retrace check is vacuous."
        )

        # A genuine shape change (different nlev) is the ONLY thing that
        # should trigger a second compile.
        out_c = run(*self._inputs(nlev=20, seed=3))
        jax.block_until_ready(out_c.lw_flux_up)
        assert n_traces[0] == 2, (
            f"expected exactly one additional compile on an nlev change; "
            f"got {n_traces[0]} total traces."
        )


class TestGasVmrOverride:
    """Iter-73: pin that runtime gas-VMR overrides (``ghg_vmr_override``
    for CO2/CH4/N2O/CFCs and the ``o3_vmr`` field) actually reach the
    gas-optics kernel and change fluxes in the physically correct
    direction.

    Audit (iter-73): ``solve_columns`` writes each override into
    ``vmr_fields``; ``gas_optics.get_vmr`` overwrites the table global
    mean with it via ``jnp.where(species_idx == gas_idx, field, gm)`` —
    and this is consulted for the **major** OD path, the **minor** OD
    path (``_compute_minor_optical_depth`` L452 for the absorber's own
    VMR, L418 for the scaling gas), and the H2O dry-factor.  So a minor
    absorber's override is honoured too.  Nothing tested this: a refactor
    that dropped ``vmr_fields`` from the minor path (or had ``get_vmr``
    prefer the baked global mean) would silently make a CO2-doubling or
    CH4 experiment a no-op — exactly the class of regression AIMIP-#312
    introduced elsewhere.  The CH4 test is the load-bearing guard: CH4 is
    a *minor* absorber, so a nonzero OLR response proves the minor path
    reads ``vmr_fields``.

    Signs/magnitudes below were measured against this idealized tropical
    column (ncol=2, nlev=16); the bands are wide enough to tolerate
    round-off but reject a broken (Δ≈0) override.
    """

    @staticmethod
    def _cols(ncol=2, nlev=16):
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(jnp.linspace(220.0, 295.0, nlev)[None, :], (ncol, nlev))
        sfc_T = jnp.full((ncol,), 298.0)
        q_v = jnp.full((ncol, nlev), 8e-3)
        cos_z = jnp.full((ncol,), 0.7)
        return T, p_full, p_half, sfc_T, q_v, cos_z

    def _solver(self):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        return RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=False))

    def _olr(self, solver, **kw):
        T, p_full, p_half, sfc_T, q_v, cos_z = self._cols()
        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z, **kw,
        )
        return float(out.lw_flux_up[0, 0])  # TOA-first → OLR

    def test_co2_doubling_reduces_olr(self):
        solver = self._solver()
        co2_gm = solver._vmr_lib.global_means["co2"]
        base = self._olr(solver)
        doubled = self._olr(solver, ghg_vmr_override={"co2": 2.0 * co2_gm})
        drop = base - doubled
        assert drop > 0.2, (
            f"Doubling CO2 must reduce TOA OLR (greenhouse); got "
            f"base={base:.3f}, 2xCO2={doubled:.3f} (Δ={drop:+.3f} W/m²). "
            f"Δ≈0 ⇒ ghg_vmr_override silently dropped before gas optics."
        )
        assert drop < 10.0, (
            f"2xCO2 OLR reduction {drop:.3f} W/m² implausibly large for "
            f"this column — suspect a units/precedence error."
        )

    def test_minor_gas_ch4_override_changes_olr(self):
        """Load-bearing guard for the *minor*-gas override path: CH4 is a
        minor absorber, so a 5x override changing OLR proves
        ``_compute_minor_optical_depth`` honours ``vmr_fields``."""
        solver = self._solver()
        ch4_gm = solver._vmr_lib.global_means["ch4"]
        base = self._olr(solver)
        enhanced = self._olr(solver, ghg_vmr_override={"ch4": 5.0 * ch4_gm})
        drop = base - enhanced
        assert drop > 0.1, (
            f"5x CH4 (a MINOR absorber) must reduce OLR; got Δ={drop:+.3f} "
            f"W/m². Δ≈0 ⇒ the minor-gas optical-depth path ignores "
            f"vmr_fields (override silently dropped for minor species)."
        )
        assert drop < 8.0, f"5x CH4 OLR drop {drop:.3f} W/m² implausibly large."

    def test_o3_override_changes_sw_and_lw(self):
        """The ``o3_vmr`` field (separate plumbing from ghg_vmr_override)
        must reach optics: high stratospheric O3 absorbs SW (less reaches
        the surface) and adds LW greenhouse (lower OLR)."""
        solver = self._solver()
        T, p_full, p_half, sfc_T, q_v, cos_z = self._cols()
        ncol, nlev = T.shape

        def run(o3_val):
            return solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z,
                o3_vmr=jnp.full((ncol, nlev), o3_val),
            )

        lo = run(1.0e-9)   # essentially no ozone
        hi = run(5.0e-6)   # ~5 ppmv stratospheric ozone
        # Surface SW down: interface index -1 = surface (TOA-first).
        sw_drop = float(lo.sw_flux_down[0, -1] - hi.sw_flux_down[0, -1])
        olr_drop = float(lo.lw_flux_up[0, 0] - hi.lw_flux_up[0, 0])
        assert sw_drop > 10.0, (
            f"Raising O3 must reduce surface SW (O3 absorbs shortwave); "
            f"got Δ(sfc SW)={sw_drop:+.3f} W/m². Δ≈0 ⇒ o3_vmr ignored."
        )
        assert olr_drop > 2.0, (
            f"Raising O3 must reduce OLR (9.6 µm greenhouse); got "
            f"Δ(OLR)={olr_drop:+.3f} W/m²."
        )

    def test_co2_override_differentiable(self):
        """``jax.grad`` of OLR w.r.t. the CO2 override scalar is finite and
        negative — pins AD through the override path (GHG-forcing
        sensitivity / training)."""
        solver = self._solver()
        T, p_full, p_half, sfc_T, q_v, cos_z = self._cols()
        co2_gm = solver._vmr_lib.global_means["co2"]

        def sum_olr(co2):
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
                q_v=q_v, cos_zenith=cos_z, ghg_vmr_override={"co2": co2},
            )
            return jnp.sum(out.lw_flux_up[:, 0])

        g = jax.grad(sum_olr)(jnp.asarray(co2_gm, dtype=p_full.dtype))
        assert jnp.isfinite(g), f"∂(OLR)/∂(CO2) must be finite; got {g}"
        assert float(g) < 0.0, (
            f"∂(OLR)/∂(CO2) must be negative (more CO2 → less OLR); "
            f"got {float(g):.3e}"
        )
        assert abs(float(g)) > 1.0, (
            "∂(OLR)/∂(CO2) ≈ 0 — AD likely constant-folded the override "
            "to a non-traced value (broken differentiable GHG forcing)."
        )


class TestCloudFractionCoverage:
    """Iter-74: pin the ``cloud_fraction`` partial-coverage treatment.

    Audit (iter-74): ``optics.RRTMOptics._combine_cloud_optics`` scales
    the cloud optical depth by ``cloud_fraction`` exactly once
    (``optical_depth * cloud_fraction``), keeping cloud ssa / asymmetry,
    then combines with the gas optics.  This is a documented linear-OD
    ("gray") cloud-fraction simplification — upstream rte-rrtmgp leaves
    fractional/overlap handling to the host — so the contract is:
    ``cf=0`` ≡ clear sky, ``cf=1`` ≡ the unfractioned (``cf=None``) full
    cloud, and intermediate ``cf`` interpolates monotonically.  This is
    the path the iter-15/16 ``cf²`` double-discount bug corrupted (cf
    applied twice); ``TestCloudKwargsHelper`` guards the helper side, and
    this class guards the optics side end-to-end.

    Equivalences are exact (the scaling is ``×0`` / ``×1``); intermediate
    behaviour is checked for strict monotonicity + correct sign rather
    than a magnitude (linear-OD scaling is intentionally non-McICA).
    """

    @staticmethod
    def _tol():
        if jax.config.read("jax_enable_x64"):
            return dict(rtol=1e-11, atol=1e-10)
        return dict(rtol=1e-5, atol=1e-4)

    def _setup(self, ncol=2, nlev=16):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=True))
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(jnp.linspace(220.0, 295.0, nlev)[None, :], (ncol, nlev))
        base = dict(
            T=T, p_full=p_full, p_half=p_half,
            sfc_temperature=jnp.full((ncol,), 298.0),
            q_v=jnp.full((ncol, nlev), 8e-3),
            cos_zenith=jnp.full((ncol,), 0.7),
        )
        lwp = jnp.zeros((ncol, nlev)).at[:, 10:13].set(0.05)
        reff = jnp.full((ncol, nlev), 1.0e-5)
        cloud = dict(cloud_path_liq=lwp, cloud_r_eff_liq=reff)
        return solver, base, cloud, ncol, nlev

    def _cf(self, ncol, nlev, value):
        return jnp.full((ncol, nlev), value)

    def test_cf_zero_equals_clear_sky(self):
        """cloud_fraction=0 scales cloud OD to zero ⇒ identical to a
        cloud-free column (LW and SW)."""
        solver, base, cloud, ncol, nlev = self._setup()
        clear = solver.solve_columns(**base)
        cf0 = solver.solve_columns(
            **base, **cloud, cloud_fraction=self._cf(ncol, nlev, 0.0)
        )
        for field in ("lw_flux_up", "lw_flux_down", "sw_flux_up", "sw_flux_down"):
            np.testing.assert_allclose(
                np.asarray(getattr(cf0, field)), np.asarray(getattr(clear, field)),
                **self._tol(),
                err_msg=f"cf=0 must equal clear sky in {field!r} (cloud OD→0)",
            )

    def test_cf_one_equals_unfractioned(self):
        """cloud_fraction=1 is the identity scaling ⇒ identical to passing
        no cloud_fraction at all (full cloud)."""
        solver, base, cloud, ncol, nlev = self._setup()
        full = solver.solve_columns(**base, **cloud)  # cf=None
        cf1 = solver.solve_columns(
            **base, **cloud, cloud_fraction=self._cf(ncol, nlev, 1.0)
        )
        for field in ("lw_flux_up", "lw_flux_down", "sw_flux_up", "sw_flux_down"):
            np.testing.assert_allclose(
                np.asarray(getattr(cf1, field)), np.asarray(getattr(full, field)),
                **self._tol(),
                err_msg=f"cf=1 must equal the unfractioned cloud in {field!r}",
            )

    def test_cf_partial_is_monotonic_between_clear_and_full(self):
        """A half-covered cell sits strictly between clear and full cloud:
        OLR and surface SW both decrease monotonically as cf rises (the
        single-application, correct-direction guard against cf² or a
        flipped scaling)."""
        solver, base, cloud, ncol, nlev = self._setup()
        clear = solver.solve_columns(**base)
        cf05 = solver.solve_columns(
            **base, **cloud, cloud_fraction=self._cf(ncol, nlev, 0.5)
        )
        full = solver.solve_columns(**base, **cloud)

        olr_clear = float(clear.lw_flux_up[0, 0])
        olr_half = float(cf05.lw_flux_up[0, 0])
        olr_full = float(full.lw_flux_up[0, 0])
        assert olr_full < olr_half < olr_clear, (
            f"OLR must be monotone clear>half>full; got clear={olr_clear:.2f}, "
            f"half={olr_half:.2f}, full={olr_full:.2f}."
        )

        sw_clear = float(clear.sw_flux_down[0, -1])
        sw_half = float(cf05.sw_flux_down[0, -1])
        sw_full = float(full.sw_flux_down[0, -1])
        assert sw_full < sw_half < sw_clear, (
            f"Surface SW must be monotone clear>half>full; got "
            f"clear={sw_clear:.2f}, half={sw_half:.2f}, full={sw_full:.2f}."
        )

    def test_cf_differentiable(self):
        """∂(flux)/∂(cloud_fraction) finite, with column-summed signs
        matching the physics: raising cf lowers OLR (greenhouse) and
        lowers surface SW (albedo).  Pins AD through the cf scaling."""
        solver, base, cloud, ncol, nlev = self._setup()
        cf0 = self._cf(ncol, nlev, 0.5)

        def olr(cf):
            out = solver.solve_columns(**base, **cloud, cloud_fraction=cf)
            return jnp.sum(out.lw_flux_up[:, 0])

        def sfc_sw(cf):
            out = solver.solve_columns(**base, **cloud, cloud_fraction=cf)
            return jnp.sum(out.sw_flux_down[:, -1])

        g_olr = jax.grad(olr)(cf0)
        g_sw = jax.grad(sfc_sw)(cf0)
        assert jnp.all(jnp.isfinite(g_olr)) and jnp.all(jnp.isfinite(g_sw)), (
            "∂(flux)/∂(cloud_fraction) must be finite"
        )
        # Gradients are nonzero only in the cloudy layers (10:13).
        assert float(jnp.sum(g_olr[:, 10:13])) < 0.0, (
            "raising cloud fraction must lower OLR (greenhouse); got "
            f"∑∂OLR/∂cf={float(jnp.sum(g_olr[:, 10:13])):.3e}"
        )
        assert float(jnp.sum(g_sw[:, 10:13])) < 0.0, (
            "raising cloud fraction must lower surface SW (albedo); got "
            f"∑∂SW/∂cf={float(jnp.sum(g_sw[:, 10:13])):.3e}"
        )


class TestSolarSpectralFraction:
    """Iter-75: pin the ``solar_spectral_fraction`` per-g-point solar
    source weighting.

    Audit (iter-75): ``solve_columns`` clips the weights, normalises them
    to sum 1, and validates ``len == n_gpt_sw``; ``two_stream.solve_sw``
    then uses ``solar_flux = irrad * weight[igpt]`` where ``weight`` is
    the external array if given **else** the table
    ``optics_lib.solar_fraction_by_gpt[igpt]`` (a clean if/else — the
    external array *replaces* the table, no double-application).  The
    table's own weights already sum to 1.

    ``test_amip_rrtmg`` pins that custom weights are *consumed*; these
    pins add the faithfulness contracts: feeding back the table weights
    reproduces the default path exactly (external ≡ default), the
    internal sum-normalisation makes the absolute scale irrelevant, the
    length guard fires, and the weighting is differentiable.

    (The raw TOA-insolation budget is deliberately not pinned here: the
    stripped output's top interface sits below the top halo layer, so its
    down-flux is already g-point-dependently attenuated — not a clean
    S₀·μ₀ probe.  The ``table ≡ default`` equivalence is the consistency
    guarantee instead.)
    """

    @staticmethod
    def _tol():
        if jax.config.read("jax_enable_x64"):
            return dict(rtol=1e-10, atol=1e-9)
        return dict(rtol=1e-4, atol=1e-3)

    def _setup(self, ncol=2, nlev=16):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=False))
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(jnp.linspace(220.0, 295.0, nlev)[None, :], (ncol, nlev))
        base = dict(
            T=T, p_full=p_full, p_half=p_half,
            sfc_temperature=jnp.full((ncol,), 298.0),
            q_v=jnp.full((ncol, nlev), 8e-3),
            cos_zenith=jnp.full((ncol,), 0.7),
        )
        return solver, base

    def test_table_weights_reproduce_default(self):
        """Feeding the table's own ``solar_fraction_by_gpt`` as the
        external weights must reproduce the ``None`` (default) path
        exactly — the external override and the built-in path share one
        formula (``irrad * weight[igpt]``)."""
        solver, base = self._setup()
        default = solver.solve_columns(**base)
        table_weights = solver.optics_lib.solar_fraction_by_gpt
        explicit = solver.solve_columns(**base, solar_spectral_fraction=table_weights)
        for field in ("sw_flux_down", "sw_flux_up", "sw_heating_rate"):
            np.testing.assert_allclose(
                np.asarray(getattr(explicit, field)),
                np.asarray(getattr(default, field)),
                **self._tol(),
                err_msg=(
                    f"solar_spectral_fraction=table_weights must match the "
                    f"default (None) path in {field!r}; divergence means the "
                    f"external and built-in solar-source paths disagree."
                ),
            )

    def test_normalization_invariance(self):
        """The solver normalises the weights to sum 1, so multiplying every
        weight by a constant is a no-op — the field is a *fraction*, not an
        absolute flux."""
        solver, base = self._setup()
        ng = solver.optics_lib.n_gpt_sw
        w = jnp.linspace(0.1, 1.0, ng)  # arbitrary positive distribution
        out_1x = solver.solve_columns(**base, solar_spectral_fraction=w)
        out_7x = solver.solve_columns(**base, solar_spectral_fraction=7.0 * w)
        for field in ("sw_flux_down", "sw_flux_up"):
            np.testing.assert_allclose(
                np.asarray(getattr(out_7x, field)),
                np.asarray(getattr(out_1x, field)),
                **self._tol(),
                err_msg=(
                    f"scaling all solar weights by 7 changed {field!r} — the "
                    f"internal sum-normalisation is missing or broken."
                ),
            )

    def test_spectral_redistribution_changes_surface_sw(self):
        """Concentrating the solar source in different g-points changes the
        surface SW (g-points have different gas absorption) — the weighting
        is spectrally meaningful, not a global scalar."""
        solver, base = self._setup()
        ng = solver.optics_lib.n_gpt_sw
        first = jnp.zeros(ng).at[:5].set(1.0)   # energy in the first 5 g-points
        last = jnp.zeros(ng).at[-5:].set(1.0)   # energy in the last 5 g-points
        sw_first = solver.solve_columns(**base, solar_spectral_fraction=first)
        sw_last = solver.solve_columns(**base, solar_spectral_fraction=last)
        diff = abs(
            float(sw_first.sw_flux_down[0, -1]) - float(sw_last.sw_flux_down[0, -1])
        )
        assert diff > 1.0, (
            f"two different solar spectral distributions gave near-identical "
            f"surface SW (Δ={diff:.3e} W/m²) — weights not reaching the "
            f"per-g-point solar source."
        )

    def test_wrong_length_raises(self):
        solver, base = self._setup()
        ng = solver.optics_lib.n_gpt_sw
        with pytest.raises(ValueError):
            solver.solve_columns(
                **base, solar_spectral_fraction=jnp.ones(ng - 1)
            )

    def test_differentiable(self):
        """``jax.grad`` of surface SW w.r.t. the spectral weights is finite
        and non-trivial (AD through the per-g-point solar source)."""
        solver, base = self._setup()
        ng = solver.optics_lib.n_gpt_sw
        w0 = jnp.ones(ng)

        def sfc_sw(w):
            out = solver.solve_columns(**base, solar_spectral_fraction=w)
            return jnp.sum(out.sw_flux_down[:, -1])

        g = jax.grad(sfc_sw)(w0)
        assert jnp.all(jnp.isfinite(g)), f"∂(sfc SW)/∂(weights) must be finite"
        assert float(jnp.max(jnp.abs(g))) > 1e-6, (
            "∂(sfc SW)/∂(weights) ≈ 0 — solar weighting constant-folded out "
            "of the differentiable path."
        )


class TestLwAerosolPath:
    """Iter-76: pin the new longwave aerosol path (feature, was SW-only).

    Implemented (iter-76) as a prescribed pure-absorbing aerosol
    (single-scattering albedo 0 — the dominant LW aerosol effect, matching
    upstream rte-rrtmgp's LW aerosol support).  In ``two_stream.solve_lw``
    the aerosol optical depth is added to the gas+cloud optical depth and
    the combined ssa diluted, injected into ``precomputed_props`` *before*
    the optimal-angle secant so both the diffusivity fit and the
    source/properties solve see the aerosol-inclusive transmissivity.
    Exposed via ``solve_columns(aerosol_absorption_optical_depth_lw=...)``; ``None``
    (default) leaves the longwave solution byte-identical (zero
    regression for every existing caller — the mixing block is skipped).

    Physics pinned: an elevated absorbing aerosol layer (colder than the
    surface) reduces OLR (it replaces warm-surface emission with cold
    aerosol emission, like a thin cloud) and raises surface downwelling LW
    (extra atmospheric emission), and is AD-differentiable through both the
    fixed-secant and optimal-angle paths.
    """

    @staticmethod
    def _cols(ncol=2, nlev=16):
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(jnp.linspace(220.0, 295.0, nlev)[None, :], (ncol, nlev))
        base = dict(
            T=T, p_full=p_full, p_half=p_half,
            sfc_temperature=jnp.full((ncol,), 298.0),
            q_v=jnp.full((ncol, nlev), 8e-3),
            cos_zenith=jnp.full((ncol,), 0.7),
        )
        return base, ncol, nlev

    def _solver(self, use_optimal_angle=False, gpoint_batch_size=None):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        cfg = dict(include_clouds=False, use_optimal_angle=use_optimal_angle)
        # gpoint_batch_size=0 forces the sequential per-g-point lax.scan (vs the
        # default 16-wide vmap block); a test that captures per-g-point values
        # via jax.debug.callback needs that deterministic g-point ORDER, because
        # callbacks inside a vmapped block fire in an undefined order.
        if gpoint_batch_size is not None:
            cfg["gpoint_batch_size"] = gpoint_batch_size
        return RRTMGP.from_legoesm_config(RRTMGPConfig(**cfg))

    @staticmethod
    def _elevated_aod(ncol, nlev, value=0.3):
        # Upper-troposphere layers (TOA-first input convention: low index
        # = high altitude / cold), where absorbing aerosol most reduces OLR.
        return jnp.zeros((ncol, nlev)).at[:, 2:5].set(value)

    def test_zero_lw_aod_equals_no_aerosol(self):
        """aerosol_absorption_optical_depth_lw=0 ≡ omitting it (the pure-absorbing
        mixing is inert at zero optical depth)."""
        solver = self._solver()
        base, ncol, nlev = self._cols()
        none = solver.solve_columns(**base)
        zero = solver.solve_columns(
            **base, aerosol_absorption_optical_depth_lw=jnp.zeros((ncol, nlev))
        )
        for field in ("lw_flux_up", "lw_flux_down", "lw_heating_rate"):
            np.testing.assert_allclose(
                np.asarray(getattr(zero, field)), np.asarray(getattr(none, field)),
                rtol=1e-9, atol=1e-9,
                err_msg=f"zero LW AOD must match no-aerosol in {field!r}",
            )

    def test_elevated_lw_aerosol_reduces_olr(self):
        solver = self._solver()
        base, ncol, nlev = self._cols()
        none = solver.solve_columns(**base)
        aer = solver.solve_columns(
            **base, aerosol_absorption_optical_depth_lw=self._elevated_aod(ncol, nlev)
        )
        drop = float(none.lw_flux_up[0, 0]) - float(aer.lw_flux_up[0, 0])
        assert drop > 1.0, (
            f"An elevated (cold) absorbing aerosol layer must reduce OLR; "
            f"got Δ(OLR)={-drop:+.3f} W/m². Δ≈0 ⇒ aerosol_absorption_optical_depth_lw "
            f"not reaching the LW optical depth."
        )
        assert drop < 80.0, f"OLR drop {drop:.3f} W/m² implausibly large."

    def test_lw_aerosol_increases_surface_downwelling(self):
        solver = self._solver()
        base, ncol, nlev = self._cols()
        none = solver.solve_columns(**base)
        aer = solver.solve_columns(
            **base, aerosol_absorption_optical_depth_lw=self._elevated_aod(ncol, nlev)
        )
        rise = float(aer.lw_flux_down[0, -1]) - float(none.lw_flux_down[0, -1])
        assert rise > 0.5, (
            f"Absorbing aerosol adds atmospheric emission ⇒ surface "
            f"downwelling LW must rise; got Δ={rise:+.3f} W/m²."
        )

    def test_lw_aerosol_differentiable(self):
        solver = self._solver()
        base, ncol, nlev = self._cols()
        aod0 = self._elevated_aod(ncol, nlev)

        def olr(a):
            return jnp.sum(solver.solve_columns(
                **base, aerosol_absorption_optical_depth_lw=a).lw_flux_up[:, 0])

        g = jax.grad(olr)(aod0)
        assert jnp.all(jnp.isfinite(g)), f"∂(OLR)/∂(LW AOD) must be finite; {g}"
        assert float(jnp.sum(g[:, 2:5])) < 0.0, (
            f"∑∂(OLR)/∂(LW AOD) over the aerosol layers must be negative; "
            f"got {float(jnp.sum(g[:, 2:5])):.3f}"
        )

    def test_lw_aerosol_enters_optimal_angle_secant(self, monkeypatch):
        """White-box guard that aerosol τ reaches the optimal-angle
        diffusivity secant (``_compute_optimal_lw_secant``, fit on the
        column transmissivity exp(-Στ)) — not only the later
        source/transport solve.

        Codex iter-79→80 review: a flux differential cannot isolate this.
        The optimal-secant and fixed-1.66 paths run *different* nonlinear
        LW solves, so the optimal−fixed correction can shift when aerosol
        is added even if the secant saw gas-only τ; the source-side effect
        does not cancel.  So capture the exact optical depth handed to
        ``_compute_optimal_lw_secant`` and assert its column sum grows by
        exactly the injected aerosol OD for every g-point.  If aerosol were
        injected *after* the secant, the captured sums would be identical
        (Δ = 0).

        Requires ``JAX_ENABLE_X64=1``: the wiring property is an fp64-exact
        "Δ = injected OD per g-point" identity.  Under the default fp32 policy
        the gas+aerosol optics combine reorders float ops, so a g-point's
        captured Δ can drift by O(1) (a float32 quantum on the O(10-100)
        column-summed optical depth) — a false failure, not a wiring leak
        (verified: exact under x64). Mirror the zero-cloud/zero-aerosol guards."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip(
                "aerosol-enters-secant wiring identity requires "
                "JAX_ENABLE_X64=1; the exact per-g-point Δ is an fp64 property "
                "(fp32 reorders the optics combine, O(1) quantum noise)."
            )
        from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream

        seen: list[float] = []
        orig = two_stream._compute_optimal_lw_secant

        def spy(optical_depth, *args, **kwargs):
            # Defer to execution time (the scan body is traced, not run, at
            # call time) so the concrete summed τ is captured per g-point.
            # ordered=True: append in program (g-point) order — an UNORDERED
            # callback may reorder/coalesce effects even in a sequential scan, so
            # the two runs' arrays could still misalign by g-point.
            jax.debug.callback(
                lambda v: seen.append(float(v)), jnp.sum(optical_depth),
                ordered=True,
            )
            return orig(optical_depth, *args, **kwargs)

        monkeypatch.setattr(two_stream, "_compute_optimal_lw_secant", spy)
        # Sequential per-g-point scan (gpoint_batch_size=0) so the spy's
        # jax.debug.callback fires in deterministic g-point order in BOTH the
        # gas-only and gas+aerosol runs; under the default 16-wide vmap block the
        # callback order within a block is undefined, so the two captured arrays
        # would misalign by g-point (each gas τ differs) and the per-g-point Δ
        # comparison would see arbitrary differences, not the injected OD.
        solver = self._solver(use_optimal_angle=True, gpoint_batch_size=0)
        base, ncol, nlev = self._cols()
        aod_value = 1.0
        aod = jnp.zeros((ncol, nlev)).at[:, 2:5].set(aod_value)

        # block_until_ready before reading/clearing ``seen``: the ordered
        # callback fires in-order WITHIN a run, but on an async backend the
        # effects can still be in flight when solve_columns returns — forcing
        # the outputs guarantees every g-point's append has landed first.
        seen.clear()
        out_gas = solver.solve_columns(**base)  # gas only
        jax.block_until_ready(out_gas.lw_flux_up)
        gas = np.array(seen)
        seen.clear()
        out_aer = solver.solve_columns(
            **base, aerosol_absorption_optical_depth_lw=aod)
        jax.block_until_ready(out_aer.lw_flux_up)
        aer = np.array(seen)

        assert gas.size > 0 and gas.size == aer.size, (
            "the optimal-angle secant was not invoked once per g-point"
        )
        # Aerosol adds aod_value to 3 interior layers in each of ncol
        # columns; the secant operates on the full optical-depth array, so
        # its column sum must rise by exactly aod_value*3*ncol for EVERY
        # g-point (Δ = 0 would mean aerosol bypasses the secant).
        expected = aod_value * 3 * ncol
        np.testing.assert_allclose(
            aer - gas, np.full_like(aer, expected), rtol=0.0, atol=1e-6,
            err_msg=(
                "optical depth handed to _compute_optimal_lw_secant did not "
                "grow by the injected aerosol OD — aerosol is NOT entering τ "
                "before the secant fit."
            ),
        )

    def test_lw_aerosol_reachable_through_public_shim(self):
        """Iter-81 (codex review): the LW aerosol kwarg must be reachable +
        effective through the public ``rrtmgp_radiation`` compatibility
        entry point, not only the direct ``RRTMGP.solve_columns`` API —
        otherwise the feature is silently absent from production callers
        going through the shim."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        base, ncol, nlev = self._cols()
        cfg = RRTMGPConfig(include_clouds=False)
        none = rrtmgp_radiation(config=cfg, **base)
        aer = rrtmgp_radiation(
            config=cfg, **base,
            aerosol_absorption_optical_depth_lw=self._elevated_aod(ncol, nlev),
        )
        drop = float(none.lw_flux_up[0, 0]) - float(aer.lw_flux_up[0, 0])
        assert drop > 1.0, (
            f"LW aerosol passed through rrtmgp_radiation() did not change OLR "
            f"(Δ={drop:+.3f} W/m²) — the shim is not forwarding "
            f"aerosol_absorption_optical_depth_lw to solve_columns."
        )


class TestDeltaScaling:
    """Iter-77: pin the shortwave cloud delta-scaling transform.

    ``optics.RRTMOptics._apply_delta_scaling_for_cloud`` removes the
    forward-scattering peak (delta-Eddington, forward fraction f = g²)
    from the cloud optics before the two-stream solve.  It is applied to
    every cloudy shortwave column (``_combine_cloud_optics`` for
    ``is_lw=False``) yet had no direct test — a sign/index slip (e.g.
    ``f = g`` instead of ``g²``, or swapping the ssa/asymmetry divides)
    would silently bias every cloudy SW flux.

    The transform is exactly (audit iter-77):
        f  = ω g²
        τ' = (1 − f) τ
        ω' = ω(1 − g²) / (1 − f)        [ = (ω − f)/(1 − f) ]
        g' = (g − g²)/(1 − g²) = g/(1+g)
    """

    def _optics(self):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        return RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=True)
        ).optics_lib

    def test_matches_delta_eddington_formula(self):
        optics = self._optics()
        tau = jnp.array([10.0, 5.0, 0.7])
        ssa = jnp.array([0.99, 0.80, 0.95])
        g = jnp.array([0.85, 0.60, 0.70])
        out = optics._apply_delta_scaling_for_cloud(
            {"optical_depth": tau, "ssa": ssa, "asymmetry_factor": g}
        )
        f = np.asarray(ssa) * np.asarray(g) ** 2
        tau_exp = (1 - f) * np.asarray(tau)
        ssa_exp = np.asarray(ssa) * (1 - np.asarray(g) ** 2) / (1 - f)
        g_exp = np.asarray(g) / (1 + np.asarray(g))
        # The implementation computes ω' as (ω − f)/(1 − f); the expected here
        # uses the algebraically-equal ω(1 − g²)/(1 − f).  Equal in exact
        # arithmetic, so tighten to fp64 round-off under x64; relax to fp32
        # round-off when x64 is off (the default test env downcasts to float32,
        # where the differing operation order separates the two by ~1e-7).
        if jax.config.read("jax_enable_x64"):
            rtol = atol = 1e-12
        else:
            rtol = atol = 1e-6
        np.testing.assert_allclose(
            np.asarray(out["optical_depth"]), tau_exp, rtol=rtol, atol=atol,
            err_msg="delta-scaled τ' != (1 − ω g²) τ",
        )
        np.testing.assert_allclose(
            np.asarray(out["ssa"]), ssa_exp, rtol=rtol, atol=atol,
            err_msg="delta-scaled ω' != ω(1 − g²)/(1 − ω g²)",
        )
        np.testing.assert_allclose(
            np.asarray(out["asymmetry_factor"]), g_exp, rtol=rtol, atol=atol,
            err_msg="delta-scaled g' != g/(1 + g)",
        )

    def test_conservative_scattering_preserved(self):
        """ω = 1 must stay ω' = 1 — delta-scaling conserves a
        non-absorbing cloud (else it would spuriously create absorption)."""
        optics = self._optics()
        tau = jnp.array([8.0, 3.0])
        ssa = jnp.array([1.0, 1.0])
        g = jnp.array([0.85, 0.50])
        out = optics._apply_delta_scaling_for_cloud(
            {"optical_depth": tau, "ssa": ssa, "asymmetry_factor": g}
        )
        np.testing.assert_allclose(
            np.asarray(out["ssa"]), np.ones(2), rtol=1e-12, atol=1e-12,
            err_msg="delta-scaling must preserve conservative scattering ω=1",
        )

    def test_isotropic_is_identity(self):
        """g = 0 (isotropic): f = 0, so τ, ω, g are unchanged."""
        optics = self._optics()
        tau = jnp.array([4.0, 1.5])
        ssa = jnp.array([0.9, 0.5])
        g = jnp.array([0.0, 0.0])
        out = optics._apply_delta_scaling_for_cloud(
            {"optical_depth": tau, "ssa": ssa, "asymmetry_factor": g}
        )
        np.testing.assert_allclose(np.asarray(out["optical_depth"]), np.asarray(tau),
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(out["ssa"]), np.asarray(ssa),
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(out["asymmetry_factor"]), np.zeros(2),
                                   atol=1e-12)

    def test_reduces_tau_and_asymmetry(self):
        """Removing the forward peak shrinks both the optical depth and the
        asymmetry (the remaining phase function is less forward-peaked)."""
        optics = self._optics()
        tau = jnp.array([10.0, 5.0])
        ssa = jnp.array([0.99, 0.9])
        g = jnp.array([0.85, 0.7])
        out = optics._apply_delta_scaling_for_cloud(
            {"optical_depth": tau, "ssa": ssa, "asymmetry_factor": g}
        )
        assert np.all(np.asarray(out["optical_depth"]) < np.asarray(tau)), (
            "delta-scaling must reduce optical depth (forward peak removed)"
        )
        assert np.all(np.asarray(out["asymmetry_factor"]) < np.asarray(g)), (
            "delta-scaling must reduce asymmetry g → g/(1+g)"
        )

    def test_differentiable(self):
        optics = self._optics()
        tau = jnp.array([10.0]); ssa = jnp.array([0.95])

        def tau_out(g):
            o = optics._apply_delta_scaling_for_cloud(
                {"optical_depth": tau, "ssa": ssa, "asymmetry_factor": g}
            )
            return jnp.sum(o["optical_depth"] + o["ssa"] + o["asymmetry_factor"])

        grad = jax.grad(tau_out)(jnp.array([0.7]))
        assert jnp.all(jnp.isfinite(grad)), f"delta-scaling grad non-finite: {grad}"


class TestRayleighScattering:
    """Iter-78: pin the shortwave Rayleigh (molecular) scattering optical
    depth (``gas_optics.compute_rayleigh_optical_depth``).

    Rayleigh scattering sets the clear-sky SW backscatter / planetary
    albedo floor.  The iter-1 audit marked it faithful and
    ``TestOutOfRangeTemperature.test_out_of_range_T_saturates_rayleigh_OD``
    guards the table-T clip, but the physical behaviour was unpinned.  The
    Rayleigh OD must be a non-negative scattering optical depth, scale
    linearly with the air-column amount (∝ ``molecules`` — a fixed
    cross-section times the number of scatterers), be spectrally resolved
    (∝ 1/λ⁴, far stronger in the blue / high-energy g-points), and be
    differentiable.
    """

    @staticmethod
    def _lookup_sw():
        from tests.legoesm_paths import legoesm_source_path
        from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
            lookup_gas_optics_shortwave,
        )
        sw_path = legoesm_source_path(
            "atmosphere/physics/radiation/rrtmgp/optics"
            "/rrtmgp_data/rrtmgp-gas-sw-g112.nc"
        )
        return lookup_gas_optics_shortwave.from_data_file(str(sw_path))

    @staticmethod
    def _state(shape=(1, 1, 1)):
        return (
            jnp.full(shape, 288.0),   # T
            jnp.full(shape, 5.0e4),   # p
            jnp.full(shape, 1.0e22),  # molecules
        )

    def test_rayleigh_od_nonnegative_all_gpoints(self, lookup_vmr):
        _, vmr_lib = lookup_vmr
        lkp = self._lookup_sw()
        T, p, mol = self._state()
        ods = jnp.array([
            gas_optics.compute_rayleigh_optical_depth(
                lkp, vmr_lib, mol, T, p, igpt=jnp.array(i))[0, 0, 0]
            for i in range(lkp.n_gpt)
        ])
        assert jnp.all(ods >= 0.0), "Rayleigh scattering OD must be >= 0"
        assert jnp.any(ods > 0.0), "Rayleigh OD identically zero — not computed"

    def test_rayleigh_od_linear_in_molecules(self, lookup_vmr):
        _, vmr_lib = lookup_vmr
        lkp = self._lookup_sw()
        T, p, mol = self._state()
        igpt = jnp.array(50)
        od1 = gas_optics.compute_rayleigh_optical_depth(lkp, vmr_lib, mol, T, p, igpt)
        od2 = gas_optics.compute_rayleigh_optical_depth(
            lkp, vmr_lib, 2.0 * mol, T, p, igpt)
        # Rayleigh OD = cross-section · air amount → exactly linear.
        np.testing.assert_allclose(
            np.asarray(od2), 2.0 * np.asarray(od1), rtol=1e-10, atol=0.0,
            err_msg="Rayleigh OD must scale linearly with the air-column amount",
        )

    def test_rayleigh_od_spectrally_resolved(self, lookup_vmr):
        _, vmr_lib = lookup_vmr
        lkp = self._lookup_sw()
        T, p, mol = self._state()
        ods = np.array([
            float(gas_optics.compute_rayleigh_optical_depth(
                lkp, vmr_lib, mol, T, p, igpt=jnp.array(i))[0, 0, 0])
            for i in range(lkp.n_gpt)
        ])
        pos = ods[ods > 0]
        # 1/λ⁴ dependence ⇒ orders-of-magnitude spread across g-points, not
        # a flat per-band constant.
        assert pos.max() / pos.min() > 10.0, (
            f"Rayleigh OD nearly flat across g-points (max/min="
            f"{pos.max() / pos.min():.2f}) — spectral 1/λ⁴ dependence lost"
        )

    def test_rayleigh_od_differentiable(self, lookup_vmr):
        _, vmr_lib = lookup_vmr
        lkp = self._lookup_sw()
        T, p, mol = self._state()

        def total_od(m):
            return gas_optics.compute_rayleigh_optical_depth(
                lkp, vmr_lib, m, T, p, igpt=jnp.array(50)).sum()

        g = jax.grad(total_od)(mol)
        assert jnp.all(jnp.isfinite(g)), "Rayleigh OD gradient must be finite"
        assert float(g[0, 0, 0]) > 0.0, (
            "more air ⇒ more Rayleigh OD (∂(OD)/∂(molecules) > 0)"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Minor-gas loop over the g-point's own band only
# ---------------------------------------------------------------------------


class TestMinorGasBandLoop:
    """The minor-gas loop runs only over the intervals of the g-point's band
    (``max_minor_per_bnd_*`` trips); it must equal the loop over every
    interval in the table, bit for bit, for every g-point of both tables."""

    @pytest.mark.parametrize("spectrum", ["lw", "sw"])
    def test_band_loop_equals_full_table_loop(self, lookup_vmr, spectrum):
        import dataclasses
        from tests.legoesm_paths import legoesm_source_path
        from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
            lookup_gas_optics_shortwave,
        )
        lookup, vmr_lib = lookup_vmr
        if spectrum == "sw":
            lookup = lookup_gas_optics_shortwave.from_data_file(str(
                legoesm_source_path(
                    "atmosphere/physics/radiation/rrtmgp/optics"
                    "/rrtmgp_data/rrtmgp-gas-sw-g112.nc")))
        full = dataclasses.replace(
            lookup,
            max_minor_per_bnd_lower=lookup.n_minor_absrb_lower,
            max_minor_per_bnd_upper=lookup.n_minor_absrb_upper)
        assert lookup.max_minor_per_bnd_lower < lookup.n_minor_absrb_lower
        assert lookup.max_minor_per_bnd_upper < lookup.n_minor_absrb_upper
        # troposphere to stratosphere, warm and cold, moist and dry
        p = jnp.asarray(np.geomspace(50.0, 1.0e5, 24))
        T = jnp.asarray(np.linspace(190.0, 300.0, 24))
        molecules = jnp.asarray(np.geomspace(1.0e21, 2.0e25, 24))
        # every gas present, so each minor interval contributes (a gas left at
        # zero would hide a dropped interval)
        vmr_fields = {i: jnp.asarray(np.geomspace(1e-9, 1e-6, 24))
                      for i in set(lookup.idx_gases.values())}
        vmr_fields[lookup.idx_h2o] = jnp.asarray(np.geomspace(3e-6, 3e-2, 24))

        def tau(lk):
            return jax.jit(jax.vmap(lambda g: gas_optics.compute_minor_optical_depth(
                lk, vmr_lib, molecules, T, p, g, vmr_fields)))(
                    jnp.arange(lk.n_gpt))

        got, want = tau(lookup), tau(full)
        assert np.any(np.asarray(want) > 0.0)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
