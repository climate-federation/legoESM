"""Tests for NeuralGCM spectral training (legoesm.training.neural_gcm_spectral).

Validates:
- State conversion: SegmentCarry <-> SpectralHydrostaticState
- SFNO spectral physics wrapper produces valid tendencies
- Spectral rollout runs for a few steps without NaN
- Loss function returns a finite scalar
- Gradient flows through the full pipeline (SFNO weights)

Uses synthetic data only — no GCS/ERA5 access required.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import pack_carry
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.channel_packing import PE3DChannelSpec

# Small grid for fast tests: T10 with 3 levels
N_MAX = 10
NLEV = 3
_GRID = create_gaussian_grid(N_MAX, dealiasing="quadratic")
_SIGMA = create_sigma_coordinate(NLEV, sigma_top=0.1)


def _make_gaussian_carry(T_val=280.0, p_s_val=101325.0, q_v_val=0.005):
    """Create a synthetic SegmentCarry on the Gaussian grid."""
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
        q_v=jnp.ones(s3) * q_v_val,
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


def _assert_state_finite(pytree, label="state"):
    """Iterate over a SpectralHydrostaticState (NamedTuple) and assert all
    leaves are finite.  Handles both ``Field`` slots and the
    ``tracers`` slot (a ``dict[str, Field | jax.Array] | None``)."""
    for field in pytree:
        if field is None:
            continue
        if isinstance(field, dict):
            for v in field.values():
                data = v.data if hasattr(v, "data") else v
                assert bool(jnp.all(jnp.isfinite(data))), (
                    f"NaN in {label} tracer {getattr(v, 'name', '?')}"
                )
        else:
            assert bool(jnp.all(jnp.isfinite(field.data))), (
                f"NaN in {label} {field.name}"
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
# 1. carry_to_spectral_state
# ---------------------------------------------------------------------------

class TestCarryToSpectralState:

    def test_produces_spectral_state(self):
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        assert state.vor_hat.data.shape == (_GRID.n_sh, NLEV)
        assert state.T_hat.data.shape == (_GRID.n_sh, NLEV)
        assert state.lnps_hat.data.shape == (_GRID.n_sh,)
        assert state.phis_hat.data.shape == (_GRID.n_sh,)

    def test_dtypes_are_complex128(self):
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        assert state.T_hat.data.dtype == jnp.complex128
        assert state.vor_hat.data.dtype == jnp.complex128

    def test_finite_values(self):
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        _assert_state_finite(state, label="carry-to-spectral state")


# ---------------------------------------------------------------------------
# 2. make_sfno_spectral_physics
# ---------------------------------------------------------------------------

class TestSFNOSpectralPhysics:

    def test_returns_spectral_tendencies(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        sfno = _make_small_sfno()

        physics_fn = make_sfno_spectral_physics(sfno, _GRID)
        tendencies = physics_fn(state, _GRID, _SIGMA)

        # Should return SpectralHydrostaticState
        assert hasattr(tendencies, 'vor_hat')
        assert tendencies.T_hat.data.shape == state.T_hat.data.shape
        assert tendencies.phis_hat.data.shape == state.phis_hat.data.shape

    def test_tendencies_finite(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        sfno = _make_small_sfno()

        physics_fn = make_sfno_spectral_physics(sfno, _GRID)
        tendencies = physics_fn(state, _GRID, _SIGMA)

        _assert_state_finite(tendencies, label="tendency")

    def test_channel_mismatch_raises(self):
        """An SFNO without the forcing input planes must be rejected —
        silently reading nlev off the wrong channel count would misalign
        every packed field."""
        from legoesm.training.neural_gcm_spectral import (
            make_sfno_spectral_physics,
        )
        from legoesm.ml.channel_packing import PE3DChannelSpec
        from legoesm.ml.sfno import SFNO, SFNOConfig
        spec = PE3DChannelSpec(nlev=NLEV)
        legacy = SFNO(
            SFNOConfig(
                in_channels=spec.n_channels,      # no forcing planes
                out_channels=spec.n_channels,
                embed_dim=16, n_blocks=1, mlp_expansion=2,
                residual_prediction=False,
            ),
            _GRID, key=jax.random.PRNGKey(0),
        )
        with pytest.raises(ValueError, match="forcing"):
            make_sfno_spectral_physics(legacy, _GRID)

    def test_responds_to_prescribed_sst(self):
        """Same state, different prescribed T_sfc → different SFNO
        tendencies (the AMIP / interannual-variability pathway)."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        sfno = _make_small_sfno()
        physics_fn = make_sfno_spectral_physics(sfno, _GRID)
        ncol = len(_GRID.lat) * len(_GRID.lon)

        def _forcing(t_val):
            return {
                "T_sfc": jnp.full((ncol,), t_val, dtype=jnp.float64),
                "sic": jnp.zeros((ncol,), dtype=jnp.float64),
                "day_of_year": jnp.asarray(180.0),
                "seconds_of_day": jnp.asarray(43200.0),
            }

        out_a = physics_fn(state, _GRID, _SIGMA, forcing=_forcing(285.0))
        out_b = physics_fn(state, _GRID, _SIGMA, forcing=_forcing(300.0))
        diff = float(jnp.max(jnp.abs(out_a.T_hat.data - out_b.T_hat.data)))
        assert diff > 1e-12, (
            f"SFNO tendencies must respond to prescribed T_sfc (diff={diff})"
        )

    def test_unforced_call_stays_finite(self):
        """forcing=None (idealized/legacy path) uses internal proxies and
        stays finite."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        physics_fn = make_sfno_spectral_physics(_make_small_sfno(), _GRID)
        _assert_state_finite(
            physics_fn(state, _GRID, _SIGMA), label="unforced tendency",
        )


# ---------------------------------------------------------------------------
# 3. spectral_rollout
# ---------------------------------------------------------------------------

class TestSpectralRollout:

    def test_runs_without_nan(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        sfno = _make_small_sfno()
        physics_fn = make_sfno_spectral_physics(sfno, _GRID)

        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14,
            time_integrator="ssp_rk3",
        )

        result = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=1800.0, n_steps=2,
        )

        _assert_state_finite(result, label="rollout result")

    def test_output_same_shape(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        sfno = _make_small_sfno()
        physics_fn = make_sfno_spectral_physics(sfno, _GRID)

        pe_config = SpectralPEConfig(time_integrator="ssp_rk3")

        result = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=1800.0, n_steps=1,
        )

        assert result.T_hat.data.shape == state.T_hat.data.shape
        assert result.lnps_hat.data.shape == state.lnps_hat.data.shape

    def _forcing_base(self):
        ncol = len(_GRID.lat) * len(_GRID.lon)
        return {
            "T_sfc": jnp.full((ncol,), 290.0, dtype=jnp.float64),
            "sic": jnp.zeros((ncol,), dtype=jnp.float64),
            "day_of_year": jnp.asarray(32.0),
            "seconds_of_day": jnp.asarray(21600.0),
        }

    def test_forced_rollout_finite_and_sst_sensitive(self):
        """forcing_base threads through the scan into the physics fn:
        the forced rollout runs finite and a different prescribed SST
        yields a different final state."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        physics_fn = make_sfno_spectral_physics(_make_small_sfno(), _GRID)
        pe_config = SpectralPEConfig(time_integrator="ssp_rk3")

        fb = self._forcing_base()
        out_a = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=1800.0, n_steps=2, forcing_base=fb,
        )
        _assert_state_finite(out_a, label="forced rollout")
        fb_warm = dict(fb, T_sfc=fb["T_sfc"] + 10.0)
        out_b = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=1800.0, n_steps=2, forcing_base=fb_warm,
        )
        diff = float(jnp.max(jnp.abs(out_a.T_hat.data - out_b.T_hat.data)))
        assert diff > 1e-14, "prescribed SST did not reach the rollout physics"

    def test_forcing_with_rad_gating_raises(self):
        """forcing_base + rad-gating is the classical AMIP combination and
        must be rejected here (spectral_amip_rollout owns it)."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        physics_fn = make_sfno_spectral_physics(_make_small_sfno(), _GRID)
        pe_config = SpectralPEConfig(time_integrator="ssp_rk3")
        with pytest.raises(ValueError, match="forcing_base"):
            spectral_rollout(
                state, physics_fn, _GRID, _SIGMA, pe_config,
                dt=1800.0, n_steps=1,
                rad_physics_fn=physics_fn, rad_update_interval=4,
                forcing_base=self._forcing_base(),
            )


# ---------------------------------------------------------------------------
# 4. Loss function
# ---------------------------------------------------------------------------

class TestLoss:

    def test_finite_scalar(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            spectral_state_vs_carry_loss,
        )
        carry_ic = _make_gaussian_carry(T_val=280.0)
        carry_target = _make_gaussian_carry(T_val=282.0)
        state = carry_to_spectral_state(carry_ic, _GRID)

        sigma_full = jnp.asarray(_SIGMA.sigma_full)
        loss = spectral_state_vs_carry_loss(
            state, carry_target, _GRID, _SIGMA, sigma_full,
        )

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert float(loss) > 0.0

    def test_zero_loss_for_identical(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            spectral_state_vs_carry_loss,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)

        loss = spectral_state_vs_carry_loss(
            state, carry, _GRID, _SIGMA, sigma_full,
        )
        # Not exactly zero due to SH analysis/synthesis roundtrip, but small
        assert float(loss) < 1.0


# ---------------------------------------------------------------------------
# 5. Gradient flow through full pipeline
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_grad_through_rollout(self):
        """Verify gradients flow from loss through rollout to SFNO weights."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
            spectral_state_vs_carry_loss,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry_ic = _make_gaussian_carry(T_val=280.0)
        carry_target = _make_gaussian_carry(T_val=282.0)
        state = carry_to_spectral_state(carry_ic, _GRID)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)

        sfno = _make_small_sfno()
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14,
            time_integrator="ssp_rk3",
        )

        def loss_fn(model):
            physics_fn = make_sfno_spectral_physics(model, _GRID)
            pred = spectral_rollout(
                state, physics_fn, _GRID, _SIGMA, pe_config,
                dt=1800.0, n_steps=1,
            )
            return spectral_state_vs_carry_loss(
                pred, carry_target, _GRID, _SIGMA, sigma_full,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(sfno)

        # Loss should be finite
        assert jnp.isfinite(loss)

        # Grads should be finite and non-zero for at least some params
        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0
        all_finite = all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
        assert all_finite, "Some gradients are NaN/Inf"

        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves)
        assert has_nonzero, "All gradients are zero — no signal flows"


# ---------------------------------------------------------------------------
# 6. Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_config(self):
        from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
        cfg = NeuralGCMSpectralConfig()
        assert cfg.n_max == 42
        assert cfg.n_levels == 10
        assert cfg.dt == 600.0

    def test_channel_count(self):
        from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
        cfg = NeuralGCMSpectralConfig()
        spec = PE3DChannelSpec(nlev=cfg.n_levels)
        assert spec.n_channels == 4 * 10 + 2  # 42 channels


# ---------------------------------------------------------------------------
# 7. Column MLP spectral physics (Rasp 2018 style)
# ---------------------------------------------------------------------------

def _make_small_column_mlp():
    """Create a small NeuralPhysics for testing."""
    from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
    return NeuralPhysics(
        nlev=NLEV, hidden_dim=16, n_layers=2,
        key=jax.random.PRNGKey(99), residual_scale=0.01,
    )


class TestColumnMLPSpectralPhysics:

    def test_returns_spectral_tendencies(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_column_mlp_spectral_physics,
        )
        import equinox as eqx

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        nn = _make_small_column_mlp()
        # NeuralPhysics ZERO-inits its final layer (epoch-0 stability
        # contract: untrained net = exactly-zero tendencies), so exercise
        # the MLP->T_hat WIRING with an explicitly perturbed final bias.
        nn = eqx.tree_at(
            lambda m: m.layers[-1].bias, nn,
            jnp.full_like(nn.layers[-1].bias, 0.1))

        physics_fn = make_column_mlp_spectral_physics(nn, _GRID)
        tend = physics_fn(state, _GRID, _SIGMA)

        assert tend.T_hat.data.shape == state.T_hat.data.shape
        # Column physics only affects T — vor/div/lnps should be zero
        assert jnp.allclose(tend.vor_hat.data, 0.0)
        assert jnp.allclose(tend.div_hat.data, 0.0)
        assert jnp.allclose(tend.lnps_hat.data, 0.0)
        # T tendency should be non-zero (perturbed final layer)
        assert jnp.any(tend.T_hat.data != 0.0)

    def test_tendencies_finite(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_column_mlp_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        nn = _make_small_column_mlp()

        physics_fn = make_column_mlp_spectral_physics(nn, _GRID)
        tend = physics_fn(state, _GRID, _SIGMA)

        _assert_state_finite(tend, label="column-MLP tendency")

    def test_rollout_with_column_mlp(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_column_mlp_spectral_physics,
            spectral_rollout,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        nn = _make_small_column_mlp()
        physics_fn = make_column_mlp_spectral_physics(nn, _GRID)

        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14, time_integrator="ssp_rk3",
        )
        result = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=1800.0, n_steps=2,
        )
        _assert_state_finite(result, label="column-MLP rollout")

    def test_grad_through_column_mlp(self):
        """Gradients flow from loss through dycore to column MLP weights."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_column_mlp_spectral_physics,
            spectral_rollout,
            spectral_state_vs_carry_loss,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry_ic = _make_gaussian_carry(T_val=280.0)
        carry_target = _make_gaussian_carry(T_val=282.0)
        state = carry_to_spectral_state(carry_ic, _GRID)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)

        nn = _make_small_column_mlp()
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14, time_integrator="ssp_rk3",
        )

        def loss_fn(model):
            physics_fn = make_column_mlp_spectral_physics(model, _GRID)
            pred = spectral_rollout(
                state, physics_fn, _GRID, _SIGMA, pe_config,
                dt=1800.0, n_steps=1,
            )
            return spectral_state_vs_carry_loss(
                pred, carry_target, _GRID, _SIGMA, sigma_full,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(nn)

        assert jnp.isfinite(loss)
        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
        assert any(jnp.any(g != 0) for g in grad_leaves)


# ---------------------------------------------------------------------------
# 8. Physics-based params + spectral PE
# ---------------------------------------------------------------------------

class TestPhysicsParamsSpectral:

    def test_make_physics_fn(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_physics_params_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        params = TrainablePhysicsParams.from_defaults()

        physics_fn = make_physics_params_spectral_physics(params, _GRID, dt=600.0)
        tend = physics_fn(state, _GRID, _SIGMA)

        assert tend.T_hat.data.shape == state.T_hat.data.shape
        _assert_state_finite(tend, label="physics-params tendency")

    def test_grad_through_physics_params(self):
        """Gradients flow through physics + dycore to trainable params."""
        from legoesm.training.trainable_params import TrainablePhysicsParams
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_physics_params_spectral_physics,
            spectral_rollout,
            spectral_state_vs_carry_loss,
        )
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

        carry_ic = _make_gaussian_carry(T_val=280.0)
        carry_target = _make_gaussian_carry(T_val=282.0)
        state = carry_to_spectral_state(carry_ic, _GRID)
        sigma_full = jnp.asarray(_SIGMA.sigma_full)

        params = TrainablePhysicsParams.from_defaults()
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=1e14, time_integrator="ssp_rk3",
        )

        def loss_fn(p):
            physics_fn = make_physics_params_spectral_physics(p, _GRID, dt=1800.0)
            pred = spectral_rollout(
                state, physics_fn, _GRID, _SIGMA, pe_config,
                dt=1800.0, n_steps=1,
            )
            return spectral_state_vs_carry_loss(
                pred, carry_target, _GRID, _SIGMA, sigma_full,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        # Grads should be finite for all raw_values
        for name, g in grads.raw_values.items():
            assert jnp.isfinite(g), f"Non-finite grad for {name}"
        # At least some grads should be non-zero
        has_nonzero = any(g != 0 for g in grads.raw_values.values())
        assert has_nonzero, "All physics param gradients are zero"


class TestAreaWeightedLoss:
    """PR B #6: the spectral-state MSE/CRPS must be Gaussian-latitude area
    weighted, not a plain jnp.mean that over-weights the poles (which would
    contradict the area-weighted bias term in the same function)."""

    def test_mse_downweights_polar_error(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state, spectral_state_vs_carry_loss,
        )
        from legoesm.training.losses import LossConfig

        carry = _make_gaussian_carry()
        pred_state = carry_to_spectral_state(carry, _GRID)

        # Two equal-magnitude, equal-extent temperature errors: one placed on
        # the two most-polar Gaussian rows (small quadrature weight), one on
        # the two most-equatorial rows (large weight). Area weighting must make
        # the polar-error loss strictly smaller.
        lat = jnp.asarray(_GRID.lat)
        order = jnp.argsort(jnp.abs(lat))
        eq_rows = order[:2]
        polar_rows = order[-2:]
        dT = 10.0

        def _bump(rows):
            return carry._replace(T=carry.T.at[rows, :, :].add(dT))

        cfg = LossConfig(w_T=1.0, w_u=0.0, w_v=0.0, w_q=0.0, w_ps=0.0)
        loss_polar = float(spectral_state_vs_carry_loss(
            pred_state, _bump(polar_rows), _GRID, _SIGMA, _SIGMA.sigma_full, cfg))
        loss_equator = float(spectral_state_vs_carry_loss(
            pred_state, _bump(eq_rows), _GRID, _SIGMA, _SIGMA.sigma_full, cfg))
        assert loss_polar < loss_equator, (
            f"area-weighting should down-weight polar error: polar={loss_polar:.6e} "
            f">= equator={loss_equator:.6e} (plain jnp.mean would make them equal)"
        )


# ---------------------------------------------------------------------------
# ACE2-gap fixes: budget constraints, curriculum, chunked streaming
# ---------------------------------------------------------------------------

class TestBudgetConstraints:

    def test_sfno_tendency_conserves_dry_mass(self):
        """The (l=0,m=0) spectral coefficient of the SFNO's dlnps/dt — the
        global-mean surface-pressure (mass) tendency — must be projected
        to zero (ACE2-style dry-mass fixer)."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state, make_sfno_spectral_physics,
        )
        import numpy as np
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        physics_fn = make_sfno_spectral_physics(_make_small_sfno(), _GRID)
        tend = physics_fn(state, _GRID, _SIGMA)
        mean_idx = np.where(
            (np.asarray(_GRID.ls) == 0) & (np.asarray(_GRID.ms) == 0)
        )[0]
        assert mean_idx.size == 1
        coeff = tend.lnps_hat.data[mean_idx[0]]
        assert float(jnp.abs(coeff)) == 0.0, (
            f"global-mean lnps tendency not projected out ({coeff})"
        )

    def test_forced_rollout_keeps_qv_nonnegative(self):
        """Moisture positivity on forced learned-physics runs: q_v must be
        clipped >= 0 after every step (negative water fed the *1e3 NN
        feature and drove the runaway class).

        Deterministic (non-vacuous): the physics fn applies a constant
        negative q_v tendency that would drive q_v to ~-7e-5 over the
        rollout absent the clip, so removing the clip fails the assert.
        """
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state, spectral_rollout,
        )
        carry = _make_gaussian_carry(q_v_val=1.0e-6)  # near-zero: clip bites
        state = carry_to_spectral_state(carry, _GRID)

        def _neg_qv_physics(st, grid_, sigma_, **_kw):  # forced path passes forcing=
            # Zero every tendency except a constant drying of q_v strong
            # enough to cross zero on step 1 (1e-8/s * 1800s >> 1e-6).
            tend = jax.tree.map(jnp.zeros_like, st)
            qv = st.tracers["q_v"]
            drying = jnp.full_like(qv.data, -1.0e-8)
            return tend._replace(
                tracers=dict(tend.tracers, q_v=qv.replace(data=drying)),
            )

        ncol = len(_GRID.lat) * len(_GRID.lon)
        fb = {
            "T_sfc": jnp.full((ncol,), 290.0, dtype=jnp.float64),
            "sic": jnp.zeros((ncol,), dtype=jnp.float64),
            "day_of_year": jnp.asarray(1.0),
            "seconds_of_day": jnp.asarray(0.0),
        }
        out = spectral_rollout(
            state, _neg_qv_physics, _GRID, _SIGMA,
            SpectralPEConfig(time_integrator="ssp_rk3"),
            dt=1800.0, n_steps=4, forcing_base=fb,
        )
        qv = out.tracers["q_v"]
        qv = qv.data if hasattr(qv, "data") else qv
        assert float(jnp.min(qv)) >= 0.0, "q_v went negative on a forced run"
        # The drying really was applied: without the clip the mean would
        # be ~-7e-5; with it the column must sit essentially at zero,
        # far below the 1e-6 initial value.
        assert float(jnp.mean(qv)) < 5.0e-7


class TestRolloutCurriculum:

    def _cfg(self, curriculum, leads):
        from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
        from legoesm.training.losses import LossConfig
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
        return NeuralGCMSpectralConfig(
            n_max=N_MAX, n_levels=NLEV, dt=1800.0,
            pe_config=SpectralPEConfig(time_integrator="ssp_rk3"),
            n_epochs=1, lr=1e-4, warmup_steps=0,  # total_steps=1: keep decay_steps>0
            rollout_curriculum=curriculum,
            loss_config=LossConfig(
                multi_step_hours=leads,
                multi_step_weights=(1.0,) * len(leads) if leads else None,
            ),
            log_every=1, checkpoint_dir="/tmp/_curr_test_ckpt",
        )

    def test_missing_lead_target_raises(self):
        """A curriculum lead without a loaded target must be a hard error."""
        from legoesm.training.neural_gcm_spectral import (
            _train_spectral_loop, carry_to_spectral_state,
            make_sfno_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        cfg = self._cfg(((24, 1),), (1,))  # 24h lead, only 1h target loaded
        with pytest.raises(ValueError, match="no loaded target"):
            _train_spectral_loop(
                _make_small_sfno(), make_sfno_spectral_physics,
                _GRID, _SIGMA, [state], [(carry,)], cfg,
            )

    def test_single_phase_curriculum_trains(self):
        """A 1-epoch curriculum phase runs end-to-end: one rollout to the
        phase lead, scored against that lead's target, finite loss."""
        from legoesm.training.neural_gcm_spectral import (
            _train_spectral_loop, carry_to_spectral_state,
            make_sfno_spectral_physics,
        )
        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        cfg = self._cfg(((1, 1),), (1,))  # one 1h-lead epoch (2 steps)
        model, hist = _train_spectral_loop(
            _make_small_sfno(), make_sfno_spectral_physics,
            _GRID, _SIGMA, [state], [(carry,)], cfg,
        )
        assert len(hist) == 1 and jnp.isfinite(hist[0])


class TestChunkLoader:

    def test_partitioning_and_total(self, monkeypatch):
        """_make_chunk_loader partitions windows into <=chunk_windows groups
        and reports samples/epoch = sum(n_days)*4 (6-hourly cadence)."""
        import legoesm.training.neural_gcm_spectral as mod
        calls = []

        def _fake_load(config, grid, sigma, cache_dir, windows=None):
            calls.append(tuple(windows))
            n = sum(w[2] for w in windows) * 4
            return ["ic"] * n, ["tgt"] * n, [None] * n

        monkeypatch.setattr(mod, "load_training_data", _fake_load)
        cfg = mod.NeuralGCMSpectralConfig(
            windows=tuple((1979 + i, 0, 1) for i in range(5)),
            chunk_windows=2,
        )
        loader, n_total = mod._make_chunk_loader(
            cfg, _GRID, _SIGMA, "unused", None, None,
        )
        assert n_total == 5 * 4
        chunks = list(loader())
        assert len(chunks) == 3           # 2 + 2 + 1 windows
        assert len(calls) == 3
        assert sum(len(c[0]) for c in chunks) == n_total

    def test_requires_windows(self):
        import legoesm.training.neural_gcm_spectral as mod
        cfg = mod.NeuralGCMSpectralConfig(windows=None, chunk_windows=4)
        with pytest.raises(ValueError, match="requires config.windows"):
            mod._make_chunk_loader(cfg, _GRID, _SIGMA, "unused", None, None)


# ---------------------------------------------------------------------------
# #817. Semi-implicit training core for the WB lane
# ---------------------------------------------------------------------------

class TestSemiImplicitTrainingCore:
    """The semi-implicit SSP-RK3-SI training core (#817): pe_config.semi_implicit
    routes spectral_rollout through the Hoskins-Simmons implicit gravity-wave
    step, damping the explicit core's tangent-linear (adjoint) growth that NaN'd
    WB value_and_grad past ~6 h."""

    @staticmethod
    def _configs():
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
        explicit = SpectralPEConfig(
            time_integrator="ssp_rk3", semi_implicit=False, hyperdiff_coeff=0.0)
        si = SpectralPEConfig(
            time_integrator="ssp_rk3", semi_implicit=True,
            si_alpha=0.6, hyperdiff_coeff=0.0)  # alpha>0.5 actively damps
        return explicit, si

    @staticmethod
    def _zero_physics(s, grid, sigma, **kwargs):
        # Zero tendency -> the rollout is the PURE spectral dycore (epoch-0, when
        # the learned physics is zero-init: the exact WB adjoint-blowup scenario).
        return jax.tree_util.tree_map(
            lambda a: jnp.zeros_like(a) if eqx.is_inexact_array(a) else a, s)

    def test_semi_implicit_rollout_runs_finite_and_matches_shapes(self):
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state, spectral_rollout)
        state = carry_to_spectral_state(_make_gaussian_carry(), _GRID)
        _explicit, si = self._configs()
        out = spectral_rollout(
            state, self._zero_physics, _GRID, _SIGMA, si, dt=1800.0, n_steps=3)
        _assert_state_finite(out, label="SI rollout")
        assert out.T_hat.data.shape == state.T_hat.data.shape
        assert out.div_hat.data.shape == state.div_hat.data.shape

    def test_semi_implicit_honours_substeps(self):
        """si_substeps > 1 sub-steps ssp_rk3_step_si at dt/si_substeps and stays
        finite (the model's _do_step sub-stepping, made scan-safe)."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state, spectral_rollout)
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
        state = carry_to_spectral_state(_make_gaussian_carry(), _GRID)
        cfg = SpectralPEConfig(
            time_integrator="ssp_rk3", semi_implicit=True, si_substeps=3,
            si_alpha=0.6, hyperdiff_coeff=0.0)
        out = spectral_rollout(
            state, self._zero_physics, _GRID, _SIGMA, cfg, dt=3600.0, n_steps=2)
        _assert_state_finite(out, label="SI substep rollout")

    def test_semi_implicit_rollout_is_differentiable(self):
        """The SI rollout is reverse-mode DIFFERENTIABLE (finite, non-trivial
        gradient) at a moderate dt — the property the WB lane needs.  The SI
        matrices are precomputed constants w.r.t. the perturbation and
        ``ssp_rk3_step_si``'s per-mode linear solve is AD-safe, so ``jax.grad``
        flows cleanly through the implicit correction.

        NOTE on the adjoint-DAMPING benefit: it is realised only where the
        explicit gravity-wave CFL BINDS (fine resolution).  At this T10 smoke
        grid the CFL is ~13 ks so the explicit core is already adjoint-stable and
        the SI machinery gives no benefit (and is actually less stable at very
        large dt) — the damping demonstration belongs to the at-scale WB run
        (#817 follow-up), not a coarse unit grid."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state, spectral_rollout)
        state0 = carry_to_spectral_state(_make_gaussian_carry(), _GRID)
        _explicit, si = self._configs()

        def loss(pert):
            s = state0._replace(
                T_hat=state0.T_hat.replace(data=state0.T_hat.data + pert))
            out = spectral_rollout(
                s, self._zero_physics, _GRID, _SIGMA, si, dt=1800.0, n_steps=6)
            return jnp.real(jnp.vdot(out.T_hat.data, out.T_hat.data))

        g = jax.grad(loss)(jnp.zeros_like(state0.T_hat.data))
        assert bool(jnp.all(jnp.isfinite(g))), "SI rollout gradient not finite"
