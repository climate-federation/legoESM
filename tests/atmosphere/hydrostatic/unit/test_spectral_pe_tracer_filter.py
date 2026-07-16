"""Tests for spectral PE tracer hyperdiffusion + spectral filter (PR3).

Pins:

* The tracer filter combines the spectral exponential filter and the
  implicit hyperdiffusion factor into a single ``(n_sh,)`` multiplier
  applied via one SH round-trip per tracer per step.
* When neither knob is active (``spectral_filter_strength==0`` AND
  ``hyperdiff_coeff==0``) the tracer filter is a no-op (no transform
  cost; tracers untouched).
* When only the spectral filter is active, tracers are damped at the
  same per-mode rate as ``vor / div / T``.
* When only the hyperdiffusion is active, tracers are damped via the
  ``exp(-nu · eig · dt_eff)`` factor with ``dt_eff = dt`` for SSP-RK
  and ``dt_eff = 2·dt`` for leapfrog (matching the helper used for
  ``vor / div / T``).
* The tracer filter preserves the tracer container type (``Field`` vs
  raw JAX array).
* Long-roll stability: with hyperdiffusion ON, a high-wavenumber tracer
  perturbation is bounded after many steps (no spectral blow-up).
* Low-wavenumber tracer modes (n=0 — the global mean) are essentially
  preserved by the filter so global mass is not artificially drained
  every step.
* AD: ``jax.grad`` flows through one filtered step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    SpectralHydrostaticState,
    isothermal_rest_state_spectral,
    apply_filter_to_tracers,
    compute_spectral_filter,
)
from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis_3d,
    sh_synthesis_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate


jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def sigma_coord():
    return create_sigma_coordinate(10)


@pytest.fixture(scope="module")
def rest_state(grid, sigma_coord):
    return isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.0,
    )


def _make_field(data, name="q"):
    return Field(
        data=data, name=name,
        dims=("lat", "lon", "level"), units="kg/kg",
    )


def _proper_hyperdiff(grid):
    """Hyperdiff coeff that gives ~ 4-hour e-folding at the cutoff wavenumber."""
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (4.0 * 3600.0 * eig_max ** 2)


# ---------------------------------------------------------------------------
# Pure helper unit tests
# ---------------------------------------------------------------------------

class TestApplyFilterToTracers:
    def test_none_passthrough(self, grid, sigma_coord):
        # None tracers, None filter — both are no-ops.
        sf = compute_spectral_filter(grid.ls, grid.n_max)
        assert apply_filter_to_tracers(None, sf, grid) is None
        assert apply_filter_to_tracers({}, None, grid) == {}

    def test_unit_filter_is_roundtrip(self, grid, sigma_coord):
        """Filter ≡ 1 at every mode → output equals input (band-limited
        only; arbitrary grid functions will lose energy in modes beyond
        n_max).  Build the test tracer from a known SH coefficient so
        the round-trip is exact at this resolution."""
        nlev = sigma_coord.n_levels
        # Build a tracer from a single complex SH coefficient at
        # (m=3, n=5) — a sub-truncation mode that round-trips exactly.
        target_n = 5
        target_m = 3
        # Find the index of (n=5, m=3) in the (n_sh,) layout.
        idx = int(
            jnp.argmin(jnp.abs(grid.ls - target_n) + jnp.abs(grid.ms - target_m))
        )
        q_hat = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)
        q_hat = q_hat.at[idx, :].set(0.01)
        q_grid = sh_synthesis_3d(grid, q_hat)

        unit_filter = jnp.ones(grid.n_sh, dtype=jnp.float64)
        out = apply_filter_to_tracers(
            {"q_v": _make_field(q_grid)}, unit_filter, grid,
        )
        assert "q_v" in out
        new_q = out["q_v"].data
        # Round-trip is exact for band-limited input.
        max_err = float(jnp.max(jnp.abs(new_q - q_grid)))
        assert max_err < 1e-10, (
            f"Unit filter should round-trip a band-limited tracer "
            f"exactly; max err {max_err}"
        )

    def test_zero_filter_zeros_tracers(self, grid, sigma_coord):
        """Filter ≡ 0 at every mode → output is exactly zero."""
        nlev = sigma_coord.n_levels
        q = 0.01 * jnp.cos(grid.lon[None, :, None]) * jnp.ones(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        zero_filter = jnp.zeros(grid.n_sh, dtype=jnp.float64)
        out = apply_filter_to_tracers(
            {"q_v": _make_field(q)}, zero_filter, grid,
        )
        new_q = out["q_v"].data
        assert float(jnp.max(jnp.abs(new_q))) < 1e-12

    def test_preserves_field_container(self, grid, sigma_coord):
        """Field-wrapped tracer stays Field-wrapped through the filter."""
        nlev = sigma_coord.n_levels
        q = 0.005 * jnp.ones(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        unit_filter = jnp.ones(grid.n_sh, dtype=jnp.float64)
        out = apply_filter_to_tracers(
            {"q_v": _make_field(q, "q_v")}, unit_filter, grid,
        )
        assert hasattr(out["q_v"], "data")
        assert hasattr(out["q_v"], "replace")
        # Metadata preserved.
        assert out["q_v"].name == "q_v"

    def test_preserves_raw_container(self, grid, sigma_coord):
        """Raw-array tracer stays raw through the filter (no Field promotion)."""
        nlev = sigma_coord.n_levels
        q = 0.01 * jnp.ones(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        unit_filter = jnp.ones(grid.n_sh, dtype=jnp.float64)
        out = apply_filter_to_tracers(
            {"q_v": q}, unit_filter, grid,
        )
        assert not hasattr(out["q_v"], "data"), (
            "Raw-array input should produce raw-array output, not Field"
        )

    def test_high_wavenumber_damped_more(self, grid, sigma_coord):
        """An exponential filter damps high-n modes much more than low-n.

        Use an aggressive filter (``cutoff_fraction=0.01``) and check
        that wavenumber-1 retention is essentially 100 % while a mode
        very close to ``n_max`` (n=20 with n_max=21) is heavily damped.
        """
        nlev = sigma_coord.n_levels
        # Build a band-limited tracer with energy at n=1 (m=1) and
        # n=20 (m=20) — pure SH modes for an exact-round-trip baseline.
        idx_low = int(
            jnp.argmin(jnp.abs(grid.ls - 1) + jnp.abs(grid.ms - 1))
        )
        idx_high = int(
            jnp.argmin(jnp.abs(grid.ls - 20) + jnp.abs(grid.ms - 20))
        )
        q_hat = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)
        q_hat = q_hat.at[idx_low, :].set(0.01)
        q_hat = q_hat.at[idx_high, :].set(0.01)
        q = sh_synthesis_3d(grid, q_hat)

        # Aggressive filter: 1 % retention at n_max → strong damping
        # at the truncation edge.
        sf = compute_spectral_filter(
            grid.ls, grid.n_max, order=8, cutoff_fraction=0.01,
        )
        out = apply_filter_to_tracers(
            {"q_v": _make_field(q)}, sf, grid,
        )
        new_q = out["q_v"].data
        q_hat_in = sh_analysis_3d(grid, q)
        q_hat_out = sh_analysis_3d(grid, new_q)
        amp_low_in = float(jnp.abs(q_hat_in[idx_low, 0]))
        amp_low_out = float(jnp.abs(q_hat_out[idx_low, 0]))
        amp_high_in = float(jnp.abs(q_hat_in[idx_high, 0]))
        amp_high_out = float(jnp.abs(q_hat_out[idx_high, 0]))
        # n=1 retention should be > 99 % (filter ≈ exp(-α (1/21)^8) ≈ 1).
        assert amp_low_out / amp_low_in > 0.99, (
            f"n=1 preservation ratio {amp_low_out / amp_low_in:.4f}"
        )
        # n=20 should be heavily damped (filter ≈ exp(-α (20/21)^8) ≈
        # exp(-4.6 * 0.68) ≈ 0.044).
        assert amp_high_out / amp_high_in < 0.20, (
            f"n=20 retention ratio {amp_high_out / amp_high_in:.4f}"
        )


# ---------------------------------------------------------------------------
# Model-level: filter applied via _ensure_tracer_filter / _apply_tracer_filter
# ---------------------------------------------------------------------------

class TestTracerFilterPrecomputation:
    def test_no_knobs_no_filter(self, grid, sigma_coord):
        """Both knobs off → no precomputed filter (skip the SH round-trip)."""
        config = SpectralPEConfig(
            hyperdiff_coeff=0.0,
            spectral_filter_strength=0.0,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        model._ensure_tracer_filter(300.0)
        assert model._tracer_filter is None

    def test_only_spectral_filter(self, grid, sigma_coord):
        """spectral_filter on → tracer_filter equals spectral_filter."""
        config = SpectralPEConfig(
            hyperdiff_coeff=0.0,
            spectral_filter_strength=0.5,
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        model._ensure_tracer_filter(300.0)
        assert model._tracer_filter is not None
        assert bool(jnp.allclose(
            model._tracer_filter, model._spectral_filter,
        ))

    def test_only_hyperdiff_filter(self, grid, sigma_coord):
        """hyperdiff_coeff on → tracer_filter equals exp(-nu*eig*dt)."""
        nu = _proper_hyperdiff(grid)
        config = SpectralPEConfig(
            hyperdiff_coeff=nu,
            hyperdiff_order=2,
            spectral_filter_strength=0.0,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        model._ensure_tracer_filter(300.0)
        assert model._tracer_filter is not None
        # Construct the expected filter manually for SSP-RK3 (dt_eff=dt).
        eig = (grid.ls * (grid.ls + 1) / grid.radius ** 2) ** config.hyperdiff_order
        expected = jnp.exp(-nu * eig * 300.0)
        assert bool(jnp.allclose(model._tracer_filter, expected))

    def test_combined_filter(self, grid, sigma_coord):
        """Both knobs on → tracer_filter is the product."""
        nu = _proper_hyperdiff(grid)
        config = SpectralPEConfig(
            hyperdiff_coeff=nu,
            hyperdiff_order=2,
            spectral_filter_strength=0.5,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        model._ensure_tracer_filter(300.0)
        assert model._tracer_filter is not None
        # Reconstruct the components.
        eig = (grid.ls * (grid.ls + 1) / grid.radius ** 2) ** 2
        hyper = jnp.exp(-nu * eig * 300.0)
        sf = compute_spectral_filter(
            grid.ls, grid.n_max, order=8, cutoff_fraction=0.5,
        )
        expected = sf * hyper
        assert bool(jnp.allclose(model._tracer_filter, expected))

    def test_leapfrog_uses_2dt(self, grid, sigma_coord):
        """Leapfrog → dt_eff = 2*dt in the hyperdiff factor."""
        nu = _proper_hyperdiff(grid)
        config = SpectralPEConfig(
            hyperdiff_coeff=nu,
            hyperdiff_order=2,
            spectral_filter_strength=0.0,
            time_integrator="leapfrog_si",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        model._ensure_tracer_filter(300.0)
        # Expected: exp(-nu * eig * 600.0) (dt_eff = 2*dt).
        eig = (grid.ls * (grid.ls + 1) / grid.radius ** 2) ** 2
        expected = jnp.exp(-nu * eig * 600.0)
        assert bool(jnp.allclose(model._tracer_filter, expected))


# ---------------------------------------------------------------------------
# End-to-end: full step with tracer filter
# ---------------------------------------------------------------------------

class TestSpectralPEStepWithTracerFilter:
    def test_uniform_tracer_essentially_unchanged(
        self, grid, sigma_coord, rest_state,
    ):
        """A uniform tracer field (only the n=0 mode populated) survives
        a full step with the spectral filter on — global mass is
        preserved (or at most damped by exp(-0)=1 for n=0)."""
        nlev = sigma_coord.n_levels
        qv = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 0.013, dtype=jnp.float64,
        )
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            spectral_filter_strength=0.5,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        state = rest_state._replace(
            tracers={"q_v": _make_field(qv, "q_v")},
        )
        new_state = model.step(state, dt=300.0)
        new_qv = new_state.tracers["q_v"].data
        # Global mean (the n=0 mode) should be preserved to within
        # SH round-trip precision and integrator noise.
        mean_diff = float(jnp.abs(jnp.mean(new_qv) - jnp.mean(qv)))
        assert mean_diff < 1e-9, (
            f"Uniform tracer global mean drift {mean_diff} exceeded 1e-9"
        )

    def test_high_wavenumber_perturbation_damped(
        self, grid, sigma_coord, rest_state,
    ):
        """A purely high-wavenumber tracer perturbation must be DAMPED
        after one step when the filter is on, and PRESERVED (modulo
        integrator noise) when both knobs are off."""
        nlev = sigma_coord.n_levels
        # Wave-15 zonal perturbation (close to n_max=21).
        lon = grid.lon[None, :, None]
        amp_in = 0.001
        q_pert = amp_in * jnp.sin(15 * lon) * jnp.ones(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        # Add a uniform background so we're not testing pure noise.
        qv = 0.013 + q_pert

        # --- Filter ON ---
        config_on = SpectralPEConfig(
            hyperdiff_coeff=0.0,
            spectral_filter_strength=0.1,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model_on = SpectralPrimitiveEquationModel(grid, sigma_coord, config_on)
        state = rest_state._replace(
            tracers={"q_v": _make_field(qv, "q_v")},
        )
        s_on = model_on.step(state, dt=300.0)
        # Inspect the wave-15 amplitude on the post-step tracer.
        new_q_on = s_on.tracers["q_v"].data
        new_q_hat_on = sh_analysis_3d(grid, new_q_on)
        # Original amplitude in spectral space.
        q_hat = sh_analysis_3d(grid, qv)
        n15_mask = (grid.ls == 15)
        amp_in_15 = float(jnp.max(jnp.abs(q_hat[n15_mask])))
        amp_out_on_15 = float(jnp.max(jnp.abs(new_q_hat_on[n15_mask])))

        # --- Filter OFF ---
        config_off = SpectralPEConfig(
            hyperdiff_coeff=0.0,
            spectral_filter_strength=0.0,
            time_integrator="ssp_rk3",
        )
        model_off = SpectralPrimitiveEquationModel(grid, sigma_coord, config_off)
        s_off = model_off.step(state, dt=300.0)
        new_q_off = s_off.tracers["q_v"].data
        new_q_hat_off = sh_analysis_3d(grid, new_q_off)
        amp_out_off_15 = float(jnp.max(jnp.abs(new_q_hat_off[n15_mask])))
        # With filter ON the high-wavenumber amplitude must be SMALLER
        # than the unfiltered counterpart.  We don't pin the exact
        # ratio because integrator noise is also present in both runs.
        assert amp_out_on_15 < amp_out_off_15, (
            f"Filter must damp wave-15 vs unfiltered: "
            f"on={amp_out_on_15}, off={amp_out_off_15}"
        )
        # And the filtered amplitude is below the input.
        assert amp_out_on_15 < amp_in_15, (
            f"Filter must reduce wave-15 amplitude relative to input: "
            f"in={amp_in_15}, out={amp_out_on_15}"
        )

    def test_long_roll_high_wave_does_not_blow_up(
        self, grid, sigma_coord, rest_state,
    ):
        """50 steps with hyperdiffusion must keep a high-wavenumber
        tracer perturbation BOUNDED (no exponential growth)."""
        nlev = sigma_coord.n_levels
        lon = grid.lon[None, :, None]
        amp_in = 0.001
        # Wave-18 perturbation — very close to truncation.
        q_pert = amp_in * jnp.sin(18 * lon) * jnp.ones(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        qv = 0.013 + q_pert
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid) * 100.0,  # aggressive
            spectral_filter_strength=0.5,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        state = rest_state._replace(
            tracers={"q_v": _make_field(qv, "q_v")},
        )
        s = state
        for _ in range(50):
            s = model.step(s, dt=300.0)
        new_qv = s.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(new_qv)))
        # After 50 steps with damping, the wave-18 amplitude must be
        # smaller than (or equal to) the input amplitude.
        new_q_hat = sh_analysis_3d(grid, new_qv)
        n18_mask = (grid.ls == 18)
        amp_out = float(jnp.max(jnp.abs(new_q_hat[n18_mask])))
        # The input amplitude in spectral space.
        q_hat_in = sh_analysis_3d(grid, qv)
        amp_in_n18 = float(jnp.max(jnp.abs(q_hat_in[n18_mask])))
        assert amp_out < amp_in_n18, (
            f"50 steps of hyperdiff should reduce wave-18; "
            f"in={amp_in_n18}, out={amp_out}"
        )

    def test_filter_skipped_when_no_tracers(
        self, grid, sigma_coord, rest_state,
    ):
        """When state.tracers is None the apply pathway is a no-op even
        when the precomputed filter is non-None."""
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            spectral_filter_strength=0.5,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        # No tracers in state.
        new_state = model.step(rest_state, dt=300.0)
        assert new_state.tracers is None

    def test_grad_through_step_with_filter(
        self, grid, sigma_coord, rest_state,
    ):
        """``jax.grad`` flows through one filtered step with tracers."""
        nlev = sigma_coord.n_levels
        qv_base = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 0.01, dtype=jnp.float64,
        )
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            spectral_filter_strength=0.5,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        def loss(scale):
            qv = scale * qv_base
            s = rest_state._replace(
                tracers={"q_v": _make_field(qv, "q_v")},
            )
            new_s = model.step(s, dt=300.0)
            return jnp.sum(new_s.tracers["q_v"].data ** 2)

        g = jax.grad(loss)(jnp.array(1.0))
        assert bool(jnp.isfinite(g))
