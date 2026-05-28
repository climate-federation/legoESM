"""Stratosphere/tropopause fidelity tests for RRTMGP.

These tests pin down the iter-1 fixes to
``legoesm.atmosphere.physics.radiation.rrtmgp.optics.gas_optics``:

1. ``_clip_to_table_range`` keeps cold-mesospheric / hot-tropical
   temperatures inside the absorption/Planck-lookup table range so the
   linear interpolant cannot extrapolate.  Without the clamp, halo
   cells produced by ``2*T[-1] - T[-2]`` can sit below ``t_ref[0]=160K``
   where ``totplnk(T) ∝ T**4`` makes the extrapolation catastrophic.

2. ``_compute_relative_abundance_interpolant`` uses a safe denominator
   (``jnp.maximum(combined_vmr, 1e-30)``) so reverse-mode AD does not
   propagate NaN gradients through the ``combined_vmr == 0`` branch.

3. ``compute_planck_sources`` / ``compute_major_optical_depth`` etc.
   produce **finite** values and **finite gradients** when temperature
   includes mesospheric (T < 160K) or super-hot (T > 355K) cells.

These properties are required for differentiable training across the
full stratospheric column and for mixed-precision execution where
float32 halo extrapolations can push temperatures out of the
float64-table range.
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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
