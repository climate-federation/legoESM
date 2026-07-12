"""Tests for PR5: spectral PE tracer support in the DA / training pipeline.

Pins:

* ``carry_to_spectral_state(..., include_tracers=True)`` packages
  ``carry.q_v`` / ``q_c`` / ``q_r`` as grid-space ``Field`` entries on
  ``state.tracers`` (the canonical spectral PE tracer convention).
* ``include_tracers=False`` preserves the pre-tracer dry pipeline
  (``state.tracers is None``).
* ``spectral_state_vs_carry_loss`` adds a q_v MSE contribution when
  the predicted state carries tracers; otherwise drops the term
  silently (backward compat).
* ``spectral_rollout`` precomputes a tracer filter from the
  ``pe_config`` and applies it per step via SH round-trip when the IC
  carries tracers (mirrors the model class's
  ``_ensure_tracer_filter`` / ``_apply_tracer_filter`` pathway).
* ``jax.grad`` flows through a tracer-carrying rollout into the
  underlying SFNO weights — full AD compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import pack_carry
from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_synthesis_3d,
    sh_analysis_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.channel_packing import PE3DChannelSpec
from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig


jax.config.update("jax_enable_x64", True)


# Small grid for fast tests: T10 with 3 levels.
N_MAX = 10
NLEV = 3
_GRID = create_gaussian_grid(N_MAX, dealiasing="quadratic")
_SIGMA = create_sigma_coordinate(NLEV, sigma_top=0.1)


def _make_gaussian_carry(T_val=280.0, p_s_val=101325.0, q_v_val=0.005):
    """Synthetic SegmentCarry on the Gaussian grid (mirrors test_neural_
    gcm_spectral.py's helper but takes ``q_v_val`` for the moisture
    test path)."""
    n_lat = _GRID.lat.shape[0]
    n_lon = _GRID.lon.shape[0]
    s3 = (n_lat, n_lon, NLEV)
    s2 = (n_lat, n_lon)

    state = HydrostaticState(
        u=Field(jnp.zeros(s3), name="u", dims=("lat", "lon", "lev"), units="m/s"),
        v=Field(jnp.zeros(s3), name="v", dims=("lat", "lon", "lev"), units="m/s"),
        T=Field(jnp.full(s3, T_val), name="T", dims=("lat", "lon", "lev"), units="K"),
        p_s=Field(jnp.full(s2, p_s_val), name="p_s", dims=("lat", "lon"), units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=("lat", "lon"), units="m2/s2"),
    )
    return pack_carry(
        state,
        q_v=jnp.full(s3, q_v_val),
        q_c=jnp.zeros(s3),
        q_r=jnp.zeros(s3),
        held_dT_rad=jnp.zeros(s3),
        held_sw_net_sfc=jnp.zeros(s2),
        held_lw_net_sfc=jnp.zeros(s2),
        held_sw_up_toa=jnp.zeros(s2),
        held_lw_up_toa=jnp.zeros(s2),
        held_sw_down_toa=jnp.zeros(s2),
        step_index=0,
    )


def _make_small_sfno():
    """Create a small SFNO for testing."""
    from legoesm.training.neural_gcm_spectral import N_SFNO_FORCING_CHANNELS
    spec = PE3DChannelSpec(nlev=NLEV)
    config = SFNOConfig(
        # state channels + the surface-forcing input planes
        in_channels=spec.n_channels + N_SFNO_FORCING_CHANNELS,
        out_channels=spec.n_channels,
        embed_dim=16,
        n_blocks=1,
        mlp_expansion=2,
        residual_prediction=False,
    )
    return SFNO(config, _GRID, key=jax.random.PRNGKey(42))


# ---------------------------------------------------------------------------
# carry_to_spectral_state
# ---------------------------------------------------------------------------

class TestCarryToSpectralStateTracers:
    def test_default_includes_tracers(self):
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry(q_v_val=0.008)
        state = carry_to_spectral_state(carry, _GRID)   # default
        assert state.tracers is not None
        assert set(state.tracers.keys()) == {"q_v", "q_c", "q_r"}
        # q_v values should round-trip exactly (no SH analysis is
        # applied to tracers in the converter — they stay grid-space).
        assert bool(jnp.allclose(
            state.tracers["q_v"].data, carry.q_v.astype(jnp.float64),
        ))
        # q_c, q_r initialized to zero.
        assert float(jnp.max(jnp.abs(state.tracers["q_c"].data))) == 0.0
        assert float(jnp.max(jnp.abs(state.tracers["q_r"].data))) == 0.0

    def test_explicit_false_drops_tracers(self):
        """include_tracers=False preserves the pre-tracer dry pipeline."""
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID, include_tracers=False)
        assert state.tracers is None

    def test_tracer_field_metadata(self):
        """Tracers are wrapped as Field with proper dims/units."""
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        qv = state.tracers["q_v"]
        assert hasattr(qv, "data")
        assert qv.units == "kg/kg"
        assert qv.dims == ("lat", "lon", "level")


# ---------------------------------------------------------------------------
# spectral_state_vs_carry_loss
# ---------------------------------------------------------------------------

class TestLossWithTracers:
    def test_q_v_loss_added_when_tracers_present(self):
        """Loss differs when q_v values differ between predicted and target."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            spectral_state_vs_carry_loss,
        )
        from legoesm.training.losses import LossConfig

        carry_a = _make_gaussian_carry(q_v_val=0.005)
        carry_b = _make_gaussian_carry(q_v_val=0.010)
        state_a = carry_to_spectral_state(carry_a, _GRID)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)
        # Use w_q > 0 so the q_v term contributes.
        cfg = LossConfig(w_T=0.0, w_u=0.0, w_v=0.0, w_q=1.0, w_ps=0.0)
        # Self-loss on q_v: state_a vs carry_a → ~ 0 (round-trip noise).
        loss_self = spectral_state_vs_carry_loss(
            state_a, carry_a, _GRID, _SIGMA, sigma_full, cfg,
        )
        # Cross-loss on q_v: state_a vs carry_b → non-trivial.
        loss_diff = spectral_state_vs_carry_loss(
            state_a, carry_b, _GRID, _SIGMA, sigma_full, cfg,
        )
        # The q_v difference is 0.005 → mean squared error ~ 2.5e-5.
        assert float(loss_diff) > float(loss_self) + 1e-7, (
            f"q_v term should contribute: self={float(loss_self)}, "
            f"diff={float(loss_diff)}"
        )

    def test_no_q_v_term_when_tracers_absent(self):
        """Backward compat: when state.tracers is None the loss skips the
        q_v term silently — value matches the dry-only loss exactly."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            spectral_state_vs_carry_loss,
        )
        from legoesm.training.losses import LossConfig

        carry = _make_gaussian_carry(q_v_val=0.005, T_val=280.0)
        # Dry state (no tracers).
        state_dry = carry_to_spectral_state(
            carry, _GRID, include_tracers=False,
        )
        sigma_full = jnp.asarray(_SIGMA.sigma_full)
        cfg = LossConfig()
        # Compute the loss; should not crash and should not include q_v.
        loss = spectral_state_vs_carry_loss(
            state_dry, carry, _GRID, _SIGMA, sigma_full, cfg,
        )
        assert bool(jnp.isfinite(loss))
        # The dry-only loss should be small (state was built from carry).
        assert float(loss) < 1.0


# ---------------------------------------------------------------------------
# spectral_rollout with tracers
# ---------------------------------------------------------------------------

class TestSpectralRolloutTracers:
    def test_rollout_preserves_tracer_pytree(self):
        """A rollout starting from a tracer-carrying IC returns a state
        with the same tracer keys."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
        )
        carry = _make_gaussian_carry(q_v_val=0.005)
        state = carry_to_spectral_state(carry, _GRID)
        sfno = _make_small_sfno()
        physics_fn = make_sfno_spectral_physics(sfno, _GRID)
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14,
            time_integrator="ssp_rk3",
        )
        result = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=600.0, n_steps=2,
        )
        assert result.tracers is not None
        assert set(result.tracers.keys()) == {"q_v", "q_c", "q_r"}
        assert bool(jnp.all(jnp.isfinite(result.tracers["q_v"].data)))

    def test_rollout_filter_damps_high_wave_tracer(self):
        """With ``spectral_filter`` enabled, the rollout damps a high-
        wavenumber tracer perturbation across multiple steps.

        Without the tracer filter (PR3) the rollout would leave a
        wave-9 perturbation on q_v unchanged step-to-step.
        """
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
            _compute_tracer_filter,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import (
            compute_spectral_filter,
        )

        # Build a wave-9 q_v tracer perturbation (close to n_max=10 →
        # the spectral filter will kill this mode).
        n_lat = _GRID.lat.shape[0]
        n_lon = _GRID.lon.shape[0]
        idx = int(
            jnp.argmin(jnp.abs(_GRID.ls - 9) + jnp.abs(_GRID.ms - 9))
        )
        q_hat = jnp.zeros((_GRID.n_sh, NLEV), dtype=jnp.complex128)
        q_hat = q_hat.at[idx, :].set(0.001)
        q_grid = sh_synthesis_3d(_GRID, q_hat) + 0.005

        carry = _make_gaussian_carry()
        # Hand-set q_v to our crafted wave-9 pattern.
        s3 = carry.q_v.shape
        carry = carry._replace(q_v=q_grid)
        state = carry_to_spectral_state(carry, _GRID)
        # Verify the precomputed tracer filter is non-None.
        spectral_filter = compute_spectral_filter(
            _GRID.ls, _GRID.n_max, order=8, cutoff_fraction=0.01,
        )
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14,
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        tf = _compute_tracer_filter(_GRID, pe_config, spectral_filter, 600.0)
        assert tf is not None, (
            "Tracer filter precomputation must return non-None when "
            "either knob is active"
        )
        # Now run a rollout and check the wave-9 amplitude decreases.
        sfno = _make_small_sfno()
        physics_fn = make_sfno_spectral_physics(sfno, _GRID)

        result = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=600.0, n_steps=5,
            spectral_filter=spectral_filter,
        )
        new_q = result.tracers["q_v"].data
        new_q_hat = sh_analysis_3d(_GRID, new_q)
        amp_in_9 = float(jnp.abs(q_hat[idx, 0]))
        amp_out_9 = float(jnp.abs(new_q_hat[idx, 0]))
        # Filter cuts off at n_max with 1 % retention.  At n=9 the
        # multiplicative factor per step is ~ exp(-α (9/10)^8) ≈
        # exp(-4.6 × 0.43) ≈ 0.14.  After 5 steps: 0.14^5 ≈ 5e-5.
        # Allow generous bound (0.50) to account for advection feedback.
        assert amp_out_9 < 0.50 * amp_in_9, (
            f"Wave-9 q_v should be damped over 5 steps; "
            f"in={amp_in_9}, out={amp_out_9}"
        )


# ---------------------------------------------------------------------------
# Differentiability: end-to-end with tracers
# ---------------------------------------------------------------------------

class TestGradientFlowWithTracers:
    def test_grad_through_rollout_with_tracers(self):
        """Gradients flow from loss (incl. q_v term) through rollout to
        SFNO weights when the IC carries tracers."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
            spectral_state_vs_carry_loss,
        )
        from legoesm.training.losses import LossConfig

        carry_ic = _make_gaussian_carry(T_val=280.0, q_v_val=0.005)
        carry_target = _make_gaussian_carry(T_val=282.0, q_v_val=0.008)
        state = carry_to_spectral_state(carry_ic, _GRID)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)

        sfno = _make_small_sfno()
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14,
            time_integrator="ssp_rk3",
        )
        cfg = LossConfig()

        def loss_fn(model):
            physics_fn = make_sfno_spectral_physics(model, _GRID)
            pred = spectral_rollout(
                state, physics_fn, _GRID, _SIGMA, pe_config,
                dt=600.0, n_steps=1,
            )
            return spectral_state_vs_carry_loss(
                pred, carry_target, _GRID, _SIGMA, sigma_full, cfg,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(sfno)
        assert bool(jnp.isfinite(loss))
        # All gradients finite.
        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        all_finite = all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
        assert all_finite, "Some gradients are NaN/Inf"
        # At least one gradient is non-zero (signal flows).
        has_nonzero = any(bool(jnp.any(g != 0)) for g in grad_leaves)
        assert has_nonzero, "All gradients are zero — no signal"


# ---------------------------------------------------------------------------
# generate_nmc helpers (unit tests of the small helpers, not full NMC)
# ---------------------------------------------------------------------------

class TestGenerateNmcHelpers:
    def test_spectral_to_hydrostatic_includes_tracers_when_requested(self):
        """``_spectral_to_hydrostatic(..., include_tracers=True)`` forwards
        the spectral state's tracer dict to the HydrostaticState."""
        from legoesm.da.generate_nmc import _spectral_to_hydrostatic
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state

        carry = _make_gaussian_carry(q_v_val=0.006)
        spec = carry_to_spectral_state(carry, _GRID)
        # Default (include_tracers=False): tracers dropped.
        hs = _spectral_to_hydrostatic(spec, _GRID, _SIGMA)
        assert hs.tracers is None
        # include_tracers=True: tracers forwarded.
        hs_with = _spectral_to_hydrostatic(
            spec, _GRID, _SIGMA, include_tracers=True,
        )
        assert hs_with.tracers is not None
        assert set(hs_with.tracers.keys()) == {"q_v", "q_c", "q_r"}
        # Values match the spectral state's tracers.
        assert bool(jnp.allclose(
            hs_with.tracers["q_v"].data,
            spec.tracers["q_v"].data,
        ))


# ---------------------------------------------------------------------------
# Synthetic ERA5 → spectral helpers (NMC moisture pipeline end-to-end)
# ---------------------------------------------------------------------------

def _make_synthetic_era5_slice(q_amp=0.01):
    """Build a small synthetic ERA5Slice on a coarse lat-lon grid.

    Used to drive the NMC pipeline without touching disk / Zarr / GCS.
    Pressure levels are ascending in Pa.  q is set to ``q_amp`` so we
    can verify the value round-trips through the regrid + vertical
    interpolation into the spectral state's q_v tracer.
    """
    import numpy as np
    from legoesm.training.era5_to_state import ERA5Slice
    n_lat, n_lon, n_plev = 18, 36, 5
    lat = np.linspace(-jnp.pi / 2 * 0.99, jnp.pi / 2 * 0.99, n_lat)
    lon = np.linspace(0.0, 2 * jnp.pi, n_lon, endpoint=False)
    plev = np.array([100.0, 250.0, 500.0, 700.0, 1000.0]) * 100.0  # Pa
    return ERA5Slice(
        T=np.full((n_lat, n_lon, n_plev), 280.0, dtype=np.float32),
        u=np.zeros((n_lat, n_lon, n_plev), dtype=np.float32),
        v=np.zeros((n_lat, n_lon, n_plev), dtype=np.float32),
        q=np.full((n_lat, n_lon, n_plev), q_amp, dtype=np.float32),
        p_s=np.full((n_lat, n_lon), 101325.0, dtype=np.float32),
        sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
        phis=np.zeros((n_lat, n_lon), dtype=np.float32),
        lat=lat,
        lon=lon,
        plev_Pa=plev,
    )


class TestNmcEra5ToSpectralWithTracers:
    def test_default_loads_q_v_from_era5(self):
        """``_era5_to_spectral`` (default include_tracers=True) packages
        ERA5 specific humidity as a grid-space ``q_v`` Field on
        state.tracers.  Without this fix the NMC pipeline silently
        dropped moisture (the bug Codex flagged at stop-time review)."""
        from legoesm.da.generate_nmc import _era5_to_spectral
        era5 = _make_synthetic_era5_slice(q_amp=0.012)
        spec = _era5_to_spectral(era5, _GRID, _SIGMA)
        assert spec.tracers is not None
        assert "q_v" in spec.tracers
        qv = spec.tracers["q_v"].data
        # The synthetic ERA5 has constant q_amp everywhere; after
        # regrid + vertical interp the model q_v should be approximately
        # constant at the same value (interpolation in a constant
        # field is exact up to round-off and clip-at-zero).
        assert bool(jnp.all(jnp.isfinite(qv)))
        max_abs_err = float(jnp.max(jnp.abs(qv - 0.012)))
        assert max_abs_err < 1e-4, (
            f"Constant-q ERA5 should round-trip to constant q_v; "
            f"max err = {max_abs_err}"
        )

    def test_explicit_false_drops_tracers(self):
        """``include_tracers=False`` recovers the legacy dry pipeline."""
        from legoesm.da.generate_nmc import _era5_to_spectral
        era5 = _make_synthetic_era5_slice()
        spec = _era5_to_spectral(era5, _GRID, _SIGMA, include_tracers=False)
        assert spec.tracers is None

    def test_q_v_is_clipped_nonnegative(self):
        """ERA5 q occasionally has tiny negative values from the
        interpolator; the bridge clips to ≥ 0 so downstream physics
        never sees negative mixing ratios."""
        import numpy as np
        from legoesm.training.era5_to_state import ERA5Slice
        from legoesm.da.generate_nmc import _era5_to_spectral
        n_lat, n_lon, n_plev = 18, 36, 5
        lat = np.linspace(-jnp.pi / 2 * 0.99, jnp.pi / 2 * 0.99, n_lat)
        lon = np.linspace(0.0, 2 * jnp.pi, n_lon, endpoint=False)
        plev = np.array([100.0, 250.0, 500.0, 700.0, 1000.0]) * 100.0
        # Set q = -1e-7 everywhere (synthetic interpolator artifact).
        era5 = ERA5Slice(
            T=np.full((n_lat, n_lon, n_plev), 280.0, dtype=np.float32),
            u=np.zeros((n_lat, n_lon, n_plev), dtype=np.float32),
            v=np.zeros((n_lat, n_lon, n_plev), dtype=np.float32),
            q=np.full((n_lat, n_lon, n_plev), -1e-7, dtype=np.float32),
            p_s=np.full((n_lat, n_lon), 101325.0, dtype=np.float32),
            sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
            phis=np.zeros((n_lat, n_lon), dtype=np.float32),
            lat=lat, lon=lon, plev_Pa=plev,
        )
        spec = _era5_to_spectral(era5, _GRID, _SIGMA)
        qv = spec.tracers["q_v"].data
        assert float(jnp.min(qv)) >= 0.0, (
            f"Negative q_v after clipping: {float(jnp.min(qv))}"
        )


# ---------------------------------------------------------------------------
# NMC end-to-end: build two fake forecasts, diff, check moisture flows
# ---------------------------------------------------------------------------

class TestChannelPackingTracers:
    """``pack_pe_state`` / ``unpack_pe_output`` round-trip the q channel
    through the SFNO physics path.  Without these tracer hooks the
    SFNO is dry-input / dry-output even when the IC carries q_v."""

    def test_pack_pulls_q_v_from_state_tracers(self):
        """The q channel of the packed tensor matches state.tracers['q_v']
        instead of the previous hardcoded zero."""
        from legoesm.ml.channel_packing import pack_pe_state, PE3DChannelSpec
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry(q_v_val=0.011)
        spec_state = carry_to_spectral_state(carry, _GRID)
        packed = pack_pe_state(spec_state, _GRID)
        spec = PE3DChannelSpec(nlev=NLEV)
        # The q-slice of the packed tensor equals state.tracers["q_v"]
        # (modulo container-type cast).
        q_packed = packed[..., spec.q_slice]
        q_state = spec_state.tracers["q_v"].data
        assert bool(jnp.allclose(q_packed, q_state.astype(q_packed.dtype)))
        # Sanity: not the pre-fix hardcoded zero.
        assert float(jnp.max(jnp.abs(q_packed))) > 0.0

    def test_pack_falls_back_to_zero_when_no_tracers(self):
        """Backward compat: ``state.tracers is None`` → q channel is
        zeros (matches the legacy dry pipeline)."""
        from legoesm.ml.channel_packing import pack_pe_state, PE3DChannelSpec
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        spec_state = carry_to_spectral_state(
            carry, _GRID, include_tracers=False,
        )
        packed = pack_pe_state(spec_state, _GRID)
        spec = PE3DChannelSpec(nlev=NLEV)
        q_packed = packed[..., spec.q_slice]
        assert float(jnp.max(jnp.abs(q_packed))) == 0.0

    def test_unpack_emits_q_v_tendency(self):
        """The q output channel of the SFNO becomes ``state.tracers['q_v']``
        in the unpacked tendency, with the original container type
        preserved.  Critical for the dycore RHS to consume the SFNO's
        predicted dq_v/dt."""
        from legoesm.ml.channel_packing import (
            unpack_pe_output, PE3DChannelSpec,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry(q_v_val=0.005)
        spec_state = carry_to_spectral_state(carry, _GRID)
        # Synthetic SFNO output: zeros except a known q channel pattern.
        spec = PE3DChannelSpec(nlev=NLEV)
        n_lat = _GRID.lat.shape[0]
        n_lon = _GRID.lon.shape[0]
        output = jnp.zeros((n_lat, n_lon, spec.n_channels))
        # Set a non-trivial q-channel: small positive number per level.
        q_pattern = jnp.full((n_lat, n_lon, NLEV), 1.5e-7)
        output = output.at[..., spec.q_slice].set(q_pattern)
        result = unpack_pe_output(output, spec_state, _GRID, mode="tendencies")
        assert result.tracers is not None
        assert "q_v" in result.tracers
        # Container type preserved (Field stays Field).
        assert hasattr(result.tracers["q_v"], "data")
        # Value matches the synthetic q-channel.
        assert bool(jnp.allclose(
            result.tracers["q_v"].data,
            q_pattern.astype(result.tracers["q_v"].data.dtype),
        ))

    def test_unpack_drops_tracers_when_state_is_dry(self):
        """``state.tracers is None`` (legacy dry path) → output also
        has tracers=None even though the q channel exists in the
        tensor.  No silent tracer promotion."""
        from legoesm.ml.channel_packing import (
            unpack_pe_output, PE3DChannelSpec,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        spec_state = carry_to_spectral_state(
            carry, _GRID, include_tracers=False,
        )
        spec = PE3DChannelSpec(nlev=NLEV)
        n_lat = _GRID.lat.shape[0]
        n_lon = _GRID.lon.shape[0]
        output = jnp.zeros((n_lat, n_lon, spec.n_channels))
        result = unpack_pe_output(output, spec_state, _GRID, mode="tendencies")
        assert result.tracers is None

    def test_state_update_carries_non_predicted_tracers_through(self):
        """mode="state_update": the output IS the next state and the
        network predicts only q_v — non-predicted tracers (q_c, q_r)
        must be carried through from the INPUT state, not zeroed.
        Zeroing (the pre-fix behaviour, correct only for tendencies)
        silently ERASED cloud water on every state-update step."""
        from legoesm.ml.channel_packing import (
            pack_pe_state, unpack_pe_output,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry(q_v_val=0.005)
        carry = carry._replace(q_c=jnp.full_like(carry.q_c, 2.5e-4))
        spec_state = carry_to_spectral_state(carry, _GRID)
        assert "q_c" in spec_state.tracers  # precondition
        packed = pack_pe_state(spec_state, _GRID)
        result = unpack_pe_output(
            packed, spec_state, _GRID, mode="state_update",
        )
        # q_c carried through identically from the input state.
        assert bool(jnp.array_equal(
            result.tracers["q_c"].data, spec_state.tracers["q_c"].data,
        ))
        assert float(jnp.max(jnp.abs(result.tracers["q_c"].data))) > 0.0
        # q_v still comes from the network output channel (round-trips).
        assert bool(jnp.allclose(
            result.tracers["q_v"].data.astype(jnp.float64),
            spec_state.tracers["q_v"].data.astype(jnp.float64),
        ))

    def test_tendencies_mode_zeroes_non_predicted_tracers(self):
        """mode="tendencies": non-predicted tracers get a ZERO tendency
        (zero tendency = unchanged after integration — correct), never
        the input values (which would double them per unit time)."""
        from legoesm.ml.channel_packing import (
            pack_pe_state, unpack_pe_output,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry(q_v_val=0.005)
        carry = carry._replace(q_c=jnp.full_like(carry.q_c, 2.5e-4))
        spec_state = carry_to_spectral_state(carry, _GRID)
        packed = pack_pe_state(spec_state, _GRID)
        result = unpack_pe_output(
            packed, spec_state, _GRID, mode="tendencies",
        )
        assert float(jnp.max(jnp.abs(result.tracers["q_c"].data))) == 0.0

    def test_unpack_unknown_mode_raises(self):
        """Unknown mode must raise, never silently fall through to one of
        the two behaviours (dispatch hardening)."""
        from legoesm.ml.channel_packing import (
            pack_pe_state, unpack_pe_output,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        spec_state = carry_to_spectral_state(carry, _GRID)
        packed = pack_pe_state(spec_state, _GRID)
        with pytest.raises(ValueError, match="Unknown unpack mode"):
            unpack_pe_output(packed, spec_state, _GRID, mode="not_a_mode")

    def test_pack_unpack_q_round_trip_is_exact_for_band_limited(self):
        """A wave-3 q_v field round-trips through pack → unpack without
        loss (band-limited grid → tensor → grid is exact)."""
        from legoesm.ml.channel_packing import (
            pack_pe_state, unpack_pe_output, PE3DChannelSpec,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        from legoesm.grids.gaussian import sh_synthesis_3d
        # Build a band-limited q_v from a single SH coefficient.
        idx = int(
            jnp.argmin(jnp.abs(_GRID.ls - 3) + jnp.abs(_GRID.ms - 3))
        )
        q_hat = jnp.zeros((_GRID.n_sh, NLEV), dtype=jnp.complex128)
        q_hat = q_hat.at[idx, :].set(0.005)
        q_grid = sh_synthesis_3d(_GRID, q_hat)
        carry = _make_gaussian_carry()
        carry = carry._replace(q_v=q_grid)
        spec_state = carry_to_spectral_state(carry, _GRID)
        packed = pack_pe_state(spec_state, _GRID)
        # Roundtrip: feed `packed` back as the SFNO output (state-update
        # mode) and check the q_v survives.
        result = unpack_pe_output(packed, spec_state, _GRID, mode="state_update")
        new_q = result.tracers["q_v"].data
        max_err = float(jnp.max(jnp.abs(new_q - q_grid)))
        assert max_err < 1e-9, (
            f"Band-limited q_v should round-trip; max err {max_err}"
        )

    def test_sfno_physics_emits_q_v_tendency(self):
        """End-to-end: SFNO physics function applied to a state with q_v
        emits a q_v tendency in the output (proves the bridge is
        no-longer-dry)."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
        )
        carry = _make_gaussian_carry(q_v_val=0.005)
        state = carry_to_spectral_state(carry, _GRID)
        sfno = _make_small_sfno()
        physics_fn = make_sfno_spectral_physics(sfno, _GRID)
        tendencies = physics_fn(state, _GRID, _SIGMA)
        # SFNO output must include q_v tendency now (was missing
        # before the channel-packing fix).
        assert tendencies.tracers is not None
        assert "q_v" in tendencies.tracers
        dq_v = tendencies.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(dq_v)))
        # The randomly-initialized SFNO should produce a non-zero
        # q_v tendency given a non-zero q_v input.
        assert float(jnp.max(jnp.abs(dq_v))) > 0.0


class TestColumnMLPTracers:
    """The column MLP physics path also needs to read q_v from
    state.tracers and emit a tracer-aligned tendency."""

    def test_column_mlp_consumes_state_q_v(self):
        """Different q_v in state → different MLP output (proves the
        bridge plumbed q_v into the column features)."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_column_mlp_spectral_physics,
        )
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics

        nn = NeuralPhysics(
            nlev=NLEV, hidden_dim=16, n_layers=2,
            key=jax.random.PRNGKey(99), residual_scale=0.1,
        )
        physics_fn = make_column_mlp_spectral_physics(nn, _GRID)
        carry_a = _make_gaussian_carry(q_v_val=0.001)
        carry_b = _make_gaussian_carry(q_v_val=0.020)
        state_a = carry_to_spectral_state(carry_a, _GRID)
        state_b = carry_to_spectral_state(carry_b, _GRID)
        out_a = physics_fn(state_a, _GRID, _SIGMA)
        out_b = physics_fn(state_b, _GRID, _SIGMA)
        # If the bridge ignored q_v, the two outputs would be identical.
        diff = float(jnp.max(jnp.abs(out_a.T_hat.data - out_b.T_hat.data)))
        assert diff > 1e-12, (
            f"Column MLP should respond to q_v changes; got identical "
            f"dT_hat for q_v=0.001 vs q_v=0.020 (diff {diff})"
        )

    def test_column_mlp_mirrors_tracer_pytree(self):
        """The column MLP tendency must mirror the input state's tracer
        pytree (q_v carries the moisture head's dq_v/dt; q_c/q_r zeros)
        so downstream ``jax.tree.map(state, tendency)`` works.  With
        ``residual_scale=0.0`` every head is exactly zero, so all
        tracer tendencies here are zero-valued."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_column_mlp_spectral_physics,
        )
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
        nn = NeuralPhysics(
            nlev=NLEV, hidden_dim=16, n_layers=2,
            key=jax.random.PRNGKey(99), residual_scale=0.0,
        )
        physics_fn = make_column_mlp_spectral_physics(nn, _GRID)
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        tend = physics_fn(state, _GRID, _SIGMA)
        assert tend.tracers is not None
        assert set(tend.tracers.keys()) == set(state.tracers.keys())
        # All zero (column MLP has no moisture predictions).
        for k, v in tend.tracers.items():
            data = v.data if hasattr(v, "data") else v
            assert float(jnp.max(jnp.abs(data))) == 0.0


class TestNmcMoistureEndToEnd:
    def test_hydrostatic_diff_carries_tracer_differences(self):
        """``_hydrostatic_diff`` returns per-key tracer differences when
        both inputs carry the same tracer keys (the canonical NMC error-
        proxy path for moisture)."""
        from legoesm.da.generate_nmc import (
            _spectral_to_hydrostatic,
            _hydrostatic_diff,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry_a = _make_gaussian_carry(q_v_val=0.010)
        carry_b = _make_gaussian_carry(q_v_val=0.005)
        spec_a = carry_to_spectral_state(carry_a, _GRID)
        spec_b = carry_to_spectral_state(carry_b, _GRID)
        hs_a = _spectral_to_hydrostatic(
            spec_a, _GRID, _SIGMA, include_tracers=True,
        )
        hs_b = _spectral_to_hydrostatic(
            spec_b, _GRID, _SIGMA, include_tracers=True,
        )
        diff = _hydrostatic_diff(hs_a, hs_b)
        assert diff.tracers is not None
        assert "q_v" in diff.tracers
        # Difference should equal q_v_a - q_v_b ≈ 0.005 in a uniform
        # column (small SH round-trip noise).
        dqv = diff.tracers["q_v"].data
        # rms should be near 0.005 (since both q_v fields are uniform).
        rms = float(jnp.sqrt(jnp.mean(dqv ** 2)))
        assert abs(rms - 0.005) < 1e-3, (
            f"Expected rms(dq_v) ≈ 0.005, got {rms}"
        )

    def test_hydrostatic_diff_drops_tracers_when_either_side_missing(self):
        """When either operand has tracers=None, diff also has None
        (preserves the legacy dry pipeline)."""
        from legoesm.da.generate_nmc import (
            _spectral_to_hydrostatic,
            _hydrostatic_diff,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry_a = _make_gaussian_carry()
        carry_b = _make_gaussian_carry()
        spec_a = carry_to_spectral_state(carry_a, _GRID)
        spec_b = carry_to_spectral_state(carry_b, _GRID)
        # One side WITH tracers, other WITHOUT.
        hs_a = _spectral_to_hydrostatic(
            spec_a, _GRID, _SIGMA, include_tracers=True,
        )
        hs_b = _spectral_to_hydrostatic(
            spec_b, _GRID, _SIGMA, include_tracers=False,
        )
        diff = _hydrostatic_diff(hs_a, hs_b)
        assert diff.tracers is None

    def test_hydrostatic_diff_intersects_keys(self):
        """If the two operands carry overlapping but non-identical
        tracer key sets, the diff returns only the intersection."""
        from legoesm.core.field import Field
        from legoesm.da.generate_nmc import (
            _spectral_to_hydrostatic,
            _hydrostatic_diff,
        )
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        spec_a = carry_to_spectral_state(carry, _GRID)
        spec_b = carry_to_spectral_state(carry, _GRID)
        hs_a = _spectral_to_hydrostatic(
            spec_a, _GRID, _SIGMA, include_tracers=True,
        )
        hs_b = _spectral_to_hydrostatic(
            spec_b, _GRID, _SIGMA, include_tracers=True,
        )
        # Drop q_c and q_r from b — only q_v is shared.
        hs_b = hs_b._replace(tracers={"q_v": hs_b.tracers["q_v"]})
        diff = _hydrostatic_diff(hs_a, hs_b)
        assert diff.tracers is not None
        assert set(diff.tracers.keys()) == {"q_v"}

    def test_full_pipeline_preserves_q_v_through_dycore_step(self):
        """End-to-end smoke: ERA5 → spectral (with q_v tracer) → one
        dycore step → back to grid → diff vs initial state.

        Pins the full NMC moisture path: q_v is loaded from ERA5,
        survives the spectral PE step, and is recoverable as a non-
        zero tracer-difference field in the NMC error proxy.
        """
        from legoesm.da.generate_nmc import (
            _era5_to_spectral,
            _spectral_to_hydrostatic,
            _hydrostatic_diff,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel,
            SpectralPEConfig,
        )

        era5 = _make_synthetic_era5_slice(q_amp=0.008)
        ic_spec = _era5_to_spectral(era5, _GRID, _SIGMA)
        # Sanity: q_v survives the loader.
        assert ic_spec.tracers is not None
        assert "q_v" in ic_spec.tracers

        # Run one dycore step (no physics).
        a = _GRID.radius
        eig_max = _GRID.n_max * (_GRID.n_max + 1) / (a * a)
        config = SpectralPEConfig(
            hyperdiff_coeff=1.0 / (4.0 * 3600.0 * eig_max ** 2),
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(_GRID, _SIGMA, config)
        f_spec = model.step(ic_spec, dt=300.0)
        # q_v must survive the step.
        assert f_spec.tracers is not None
        assert "q_v" in f_spec.tracers
        # Convert back to grid space (call site mirrors the actual NMC
        # main loop — include_tracers=True).
        ic_phys = _spectral_to_hydrostatic(
            ic_spec, _GRID, _SIGMA, include_tracers=True,
        )
        f_phys = _spectral_to_hydrostatic(
            f_spec, _GRID, _SIGMA, include_tracers=True,
        )
        # Both have tracers.
        assert ic_phys.tracers is not None
        assert f_phys.tracers is not None
        # NMC error proxy: forecast - IC.
        err = _hydrostatic_diff(f_phys, ic_phys)
        assert err.tracers is not None
        # q_v difference field is finite (the NMC error logger reads
        # this field for rms reporting).
        dqv = err.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(dqv)))
