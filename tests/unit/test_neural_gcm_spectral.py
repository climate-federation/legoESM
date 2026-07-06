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


def _make_gaussian_carry(T_val=280.0, p_s_val=101325.0):
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
        q_v=jnp.ones(s3) * 0.005,
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
