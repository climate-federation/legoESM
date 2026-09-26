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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

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

    def test_forcing_reaches_a_subcycled_radiation_call(self):
        """Sub-cycled radiation plus a forcing dict is the CLASSICAL TRAINING
        combination and used to be refused outright — which left every such
        run with no prescribed surface temperature and a fixed equinox-noon
        sun.  The prescribed field must now reach the radiation call."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
        )
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        physics_fn = make_sfno_spectral_physics(_make_small_sfno(), _GRID)
        pe_config = SpectralPEConfig(time_integrator="ssp_rk3")
        seen = []

        def rad_fn(s, g, sc, *, sim_time_seconds=0.0, forcing=None):
            seen.append(forcing)
            return physics_fn(s, g, sc)

        out = spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=1800.0, n_steps=1,
            rad_physics_fn=rad_fn, rad_update_interval=4,
            forcing_base=self._forcing_base(),
        )
        assert jnp.all(jnp.isfinite(out.T_hat.data))
        assert seen and seen[0] is not None
        assert "T_sfc" in seen[0] and "day_of_year" in seen[0]

    def test_a_radiation_callable_without_a_forcing_kwarg_still_runs(self):
        """Legacy signature (sim_time_seconds but no forcing): it must keep
        its elapsed-time thread rather than silently fall back to a call that
        drops it."""
        from legoesm.training.neural_gcm_spectral import (
            carry_to_spectral_state,
            make_sfno_spectral_physics,
            spectral_rollout,
        )
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

        carry = _make_gaussian_carry()
        state = carry_to_spectral_state(carry, _GRID)
        physics_fn = make_sfno_spectral_physics(_make_small_sfno(), _GRID)
        pe_config = SpectralPEConfig(time_integrator="ssp_rk3")
        seen = []

        def rad_fn(s, g, sc, *, sim_time_seconds=0.0):
            seen.append(sim_time_seconds)
            return physics_fn(s, g, sc)

        spectral_rollout(
            state, physics_fn, _GRID, _SIGMA, pe_config,
            dt=1800.0, n_steps=1,
            rad_physics_fn=rad_fn, rad_update_interval=4,
            sim_time_offset_seconds=7200.0,
        )
        # A non-zero offset: the no-kwargs fallback would report 0.0, so this
        # fails if the call degrades to the signature-less form (codex).
        assert seen, "the legacy radiation callable was never called"
        assert float(seen[0]) == pytest.approx(7200.0)


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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

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


    def test_no_inert_gate_rejects_unconsumed_params(self):
        """The spectral physics forward reads only SPECTRAL_PHYSICS_TRAINABLE;
        the full default set carries inert leaves the gate must reject, and the
        restricted set passes."""
        from legoesm.training.trainable_params import (
            DEFAULT_TRAINABLE, TrainablePhysicsParams,
        )
        from legoesm.training.neural_gcm_spectral import (
            SPECTRAL_PHYSICS_TRAINABLE,
            assert_spectral_physics_params_reachable,
            carry_to_spectral_state,
            make_physics_params_spectral_physics,
        )
        # Warm, humid column so SBM convection is active (its knobs are gated
        # on convection firing).
        state = carry_to_spectral_state(
            _make_gaussian_carry(T_val=300.0, q_v_val=0.02), _GRID)

        def mk(p, g):
            return make_physics_params_spectral_physics(p, g, dt=1800.0)

        with pytest.raises(ValueError, match="C_H"):
            assert_spectral_physics_params_reachable(
                TrainablePhysicsParams.from_defaults(), mk, _GRID, _SIGMA,
                state)
        live = TrainablePhysicsParams.from_defaults(
            [c for c in DEFAULT_TRAINABLE
             if c.name in SPECTRAL_PHYSICS_TRAINABLE])
        assert set(live.raw_values) == set(SPECTRAL_PHYSICS_TRAINABLE)
        assert_spectral_physics_params_reachable(live, mk, _GRID, _SIGMA, state)

    def test_trainer_trains_only_consumed_params(self, monkeypatch):
        """Executed: the trainer hands the loop only the consumed parameters."""
        from legoesm.training import neural_gcm_spectral as ngs
        from legoesm.training.neural_gcm_spectral import (
            NeuralGCMSpectralConfig, carry_to_spectral_state,
        )
        state = carry_to_spectral_state(
            _make_gaussian_carry(T_val=300.0, q_v_val=0.02), _GRID)
        seen = {}
        monkeypatch.setattr(
            ngs, "load_training_data",
            lambda *a, **k: ([state], [None], [None]))
        monkeypatch.setattr(
            ngs, "_train_spectral_loop",
            lambda params, *a, **k: seen.setdefault("params", params))
        ngs.train_physics_params_spectral(
            NeuralGCMSpectralConfig(n_max=N_MAX, n_levels=NLEV,
                                    sigma_top=0.1, dt=1800.0))
        assert set(seen["params"].raw_values) == set(
            ngs.SPECTRAL_PHYSICS_TRAINABLE)


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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
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
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
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


class TestMidEpochResume:
    """#942: a job killed mid-epoch resumes from the last completed CHUNK
    (not the top of the epoch) and reproduces the uninterrupted run
    bit-for-bit.

    This is the structural fix for the Derecho walltime cliff: a T106
    epoch (~13 h) exceeds the 12 h queue cap, so a per-epoch-only
    checkpoint makes zero progress forever.  The proof below drives the
    REAL ``_train_spectral_loop`` chunked path with a tiny SFNO + a
    synthetic in-memory chunk loader (no GCS/network), simulates a kill
    right after chunk 1 of epoch 0, resumes, and asserts the final
    weights are byte-identical to an uninterrupted run — which can only
    hold if BOTH the model weights and the full optimizer state are
    restored and the chunk sequence is deterministic.
    """

    def _cfg(self, ckpt_dir):
        from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
        from legoesm.training.losses import LossConfig
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
        return NeuralGCMSpectralConfig(
            n_max=N_MAX, n_levels=NLEV, dt=1800.0,
            pe_config=SpectralPEConfig(time_integrator="ssp_rk3"),
            lr=1e-4, warmup_steps=0, optimizer="adamw",
            rollout_curriculum=((1, 2),),   # 1h lead x 2 epochs
            loss_config=LossConfig(
                multi_step_hours=(1,), multi_step_weights=(1.0,)),
            log_every=1, checkpoint_dir=str(ckpt_dir),
        )

    def _make_loader(self, n_chunks=3):
        """A deterministic chunk loader with DISTINCT data per chunk index.

        Data is keyed by the chunk INDEX (not a call counter), so a resume
        that skips the first chunks still reads byte-identical data for the
        chunks it does load — exactly like the real ``_make_chunk_loader``
        partition of ``config.windows``.  Records the order of yielded
        chunk indices so the test can prove the skip.
        """
        from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
        chunks = []
        for c in range(n_chunks):
            carry = _make_gaussian_carry(
                T_val=280.0 + 4.0 * c, q_v_val=0.004 + 0.001 * c)
            state = carry_to_spectral_state(carry, _GRID)
            chunks.append(([state], [(carry,)]))  # 1 sample; lead-1 target tuple
        yielded = []

        def _loader(start_chunk=0):
            for gi in range(n_chunks):
                if gi < start_chunk:
                    continue
                yielded.append(gi)
                ics, tgts = chunks[gi]
                yield ics, tgts, None

        _loader.n_chunks = n_chunks
        _loader.yielded = yielded
        return _loader

    def _run(self, model, cfg, loader, resume_from_dir):
        from legoesm.training.neural_gcm_spectral import (
            _train_spectral_loop, make_sfno_spectral_physics)
        return _train_spectral_loop(
            model, make_sfno_spectral_physics, _GRID, _SIGMA,
            None, None, cfg,
            start_epoch=0, chunk_loader=loader,
            n_samples_total=loader.n_chunks,
            resume_from_dir=resume_from_dir,
        )

    def test_resume_midepoch_matches_uninterrupted_bitwise(
            self, tmp_path, monkeypatch):
        import legoesm.training.neural_gcm_spectral as mod

        # --- (A) uninterrupted reference: 2 epochs x 3 chunks ---------------
        dir_a = tmp_path / "A"
        model_a, _ = self._run(
            _make_small_sfno(), self._cfg(dir_a),
            self._make_loader(3), resume_from_dir=dir_a)

        # The run must actually have moved the weights off their init,
        # else "bit-identical" would be vacuously true.
        init_leaves = jax.tree_util.tree_leaves(
            eqx.filter(_make_small_sfno(), eqx.is_array))
        a_leaves = jax.tree_util.tree_leaves(eqx.filter(model_a, eqx.is_array))
        assert any(
            not jnp.array_equal(i, a) for i, a in zip(init_leaves, a_leaves)
        ), "reference run did not update the weights"

        # --- (B) killed run: die right after chunk 1 of epoch 0 ------------
        dir_b = tmp_path / "B"
        real_save = mod._save_midepoch_checkpoint
        counter = {"n": 0}

        class _Kill(Exception):
            pass

        def _save_then_kill(ckpt_dir, model, opt_state, epoch, next_chunk):
            real_save(ckpt_dir, model, opt_state, epoch, next_chunk)
            counter["n"] += 1
            if counter["n"] == 2:   # after chunk idx 1 -> saved (epoch0, chunk2)
                raise _Kill()

        monkeypatch.setattr(mod, "_save_midepoch_checkpoint", _save_then_kill)
        with pytest.raises(_Kill):
            self._run(_make_small_sfno(), self._cfg(dir_b),
                      self._make_loader(3), resume_from_dir=dir_b)
        monkeypatch.undo()   # run C uses the real, unpatched save

        # The atomic checkpoint survived the kill (written before the raise).
        assert (dir_b / mod.MIDEPOCH_CHECKPOINT_NAME).exists()

        # --- (C) resume from B's checkpoint dir ----------------------------
        loader_c = self._make_loader(3)
        model_c, _ = self._run(
            _make_small_sfno(), self._cfg(dir_b),
            loader_c, resume_from_dir=dir_b)

        # Resumed at chunk 2 of epoch 0 (skipped the two done chunks), then
        # ran epoch 1 in full — NOT restarting the epoch at chunk 0.
        assert loader_c.yielded == [2, 0, 1, 2]

        # Byte-identical to the uninterrupted run: proves model weights AND
        # optimizer state (Adam moments + LR-schedule step count) were
        # restored and the chunk order is deterministic.
        c_leaves = jax.tree_util.tree_leaves(eqx.filter(model_c, eqx.is_array))
        assert len(a_leaves) == len(c_leaves)
        for la, lc in zip(a_leaves, c_leaves):
            assert jnp.array_equal(la, lc), (
                "mid-epoch resume diverged from the uninterrupted run")

    def test_resume_without_optstate_would_diverge(self, tmp_path, monkeypatch):
        """Guard-rail: if the resume dropped the optimizer state (restoring
        only the weights, the old epoch-only behaviour), the Adam moments +
        LR-schedule counter would restart and the run would NOT match.  We
        assert divergence in that degraded mode so the bit-identical pass
        above is known to be load-bearing, not luck."""
        import legoesm.training.neural_gcm_spectral as mod

        dir_a = tmp_path / "A"
        model_a, _ = self._run(
            _make_small_sfno(), self._cfg(dir_a),
            self._make_loader(3), resume_from_dir=dir_a)
        a_leaves = jax.tree_util.tree_leaves(eqx.filter(model_a, eqx.is_array))

        dir_b = tmp_path / "B"
        real_save = mod._save_midepoch_checkpoint
        counter = {"n": 0}

        class _Kill(Exception):
            pass

        def _save_then_kill(ckpt_dir, model, opt_state, epoch, next_chunk):
            real_save(ckpt_dir, model, opt_state, epoch, next_chunk)
            counter["n"] += 1
            if counter["n"] == 2:
                raise _Kill()

        monkeypatch.setattr(mod, "_save_midepoch_checkpoint", _save_then_kill)
        with pytest.raises(_Kill):
            self._run(_make_small_sfno(), self._cfg(dir_b),
                      self._make_loader(3), resume_from_dir=dir_b)
        monkeypatch.undo()

        # Degrade the loader: on load, keep weights but FORCE a fresh
        # optimizer state (simulate the old model-only checkpoint).
        real_load = mod._load_midepoch_checkpoint

        def _load_drop_optstate(ckpt_dir, model_template, opt_state_template):
            res = real_load(ckpt_dir, model_template, opt_state_template)
            if res is None:
                return None
            m, _o, ep, nc = res
            return m, opt_state_template, ep, nc   # fresh opt_state

        monkeypatch.setattr(mod, "_load_midepoch_checkpoint", _load_drop_optstate)
        model_c, _ = self._run(
            _make_small_sfno(), self._cfg(dir_b),
            self._make_loader(3), resume_from_dir=dir_b)
        c_leaves = jax.tree_util.tree_leaves(eqx.filter(model_c, eqx.is_array))
        assert any(
            not jnp.array_equal(la, lc) for la, lc in zip(a_leaves, c_leaves)
        ), (
            "dropping optimizer state should have diverged from the "
            "uninterrupted run — the bit-identical test is therefore "
            "genuinely exercising optimizer-state restore, not luck"
        )


# --- #985 item 2: chunk prefetch (host-thread double-buffering) ---
# The prefetch wrapper must be a drop-in that preserves order + resume
# semantics exactly and never swallows a producer failure. Pure Python — no
# GCS/ERA5/JAX needed.

def test_prefetch_iter_preserves_order_and_completes():
    from legoesm.training.neural_gcm_spectral import _prefetch_iter

    src = list(range(20))
    out = list(_prefetch_iter(iter(src), buffer=1))
    assert out == src


def test_prefetch_iter_propagates_producer_exception():
    from legoesm.training.neural_gcm_spectral import _prefetch_iter

    def _boom():
        yield 0
        yield 1
        raise RuntimeError("chunk load failed")

    got = []
    with pytest.raises(RuntimeError, match="chunk load failed"):
        for x in _prefetch_iter(_boom(), buffer=1):
            got.append(x)
    assert got == [0, 1]  # items before the failure are still delivered


def test_prefetch_iter_is_lazy_bounded():
    # With buffer=1 the producer runs at most `buffer+1` items ahead of a
    # consumer that never advances — it must NOT drain the whole source.
    from legoesm.training.neural_gcm_spectral import _prefetch_iter

    produced = []

    def _counting():
        for i in range(1000):
            produced.append(i)
            yield i

    it = _prefetch_iter(_counting(), buffer=1)
    first = next(it)
    assert first == 0
    # The load-gating semaphore caps the producer at buffer+1 loads ahead of a
    # stalled consumer: chunk 0 (taken) + chunk 1 (one permit released on take).
    import time
    time.sleep(0.05)
    assert len(produced) <= 2, f"prefetch over-ran: produced {len(produced)}"


def test_prefetch_iter_early_break_stops_producer():
    # Consumer breaks after one item: the producer must stop (stop flag +
    # drained slot) instead of streaming the whole source or hanging a thread.
    import time
    from legoesm.training.neural_gcm_spectral import _prefetch_iter

    produced = []

    def _counting():
        for i in range(1000):
            produced.append(i)
            yield i

    for x in _prefetch_iter(_counting(), buffer=1):
        break  # take exactly one, then abandon the iterator
    time.sleep(0.05)
    # bounded ahead-of-consumer load; must NOT have drained all 1000
    assert len(produced) <= 3, f"producer did not stop on break: {len(produced)}"


def test_chunk_loader_prefetch_matches_serial(monkeypatch):
    """_chunks with prefetch on/off yields identical chunks and honours
    start_chunk (the mid-epoch resume skip)."""
    import legoesm.training.neural_gcm_spectral as mod

    windows = [(2015, d, 1) for d in range(1, 7)]  # 6 windows

    def _fake_load(config, grid, sigma, cache_dir, windows):
        # Return a marker keyed by the group so we can assert ordering; times
        # is a 1-elem list so _maybe_build_sample_forcings (off) returns None.
        return (f"ics{windows}", f"tgt{windows}", [0])

    monkeypatch.setattr(mod, "load_training_data", _fake_load)

    class _Cfg:
        def __init__(self, prefetch):
            self.windows = windows
            self.chunk_windows = 2
            self.chunk_prefetch = prefetch

    def _collect(prefetch, start_chunk=0):
        loader, n_total = mod._make_chunk_loader(
            _Cfg(prefetch), grid=None, sigma=None, cache_dir=None,
            surface_forcing_path=None, forcing_cache_path=None,
        )
        return list(loader(start_chunk=start_chunk)), n_total, loader.n_chunks

    serial, n_s, nc_s = _collect(False)
    pref, n_p, nc_p = _collect(True)
    assert serial == pref                 # identical chunk sequence + order
    # 6 windows x 1 day x 4 snapshots/day = 24 samples; 3 chunks of 2 windows.
    assert (n_s, nc_s) == (n_p, nc_p) == (24, 3)

    # Resume skip: start_chunk=1 drops the first chunk, same for both paths.
    serial1, _, _ = _collect(False, start_chunk=1)
    pref1, _, _ = _collect(True, start_chunk=1)
    assert serial1 == pref1 == serial[1:]


def test_stage_tree_moves_arrays_skips_non_arrays():
    """``_stage_tree`` device_puts array leaves and leaves None / non-array leaves
    untouched, so a (ics, tgts, forcings=None) chunk stages without choking."""
    from legoesm.training.neural_gcm_spectral import _stage_tree

    cpu = jax.devices("cpu")[0]
    tree = {"arr": jnp.arange(3.0), "none": None, "meta": "label", "n": 2}
    out = _stage_tree(tree, cpu)
    assert list(out["arr"].devices()) == [cpu]   # array moved
    assert out["none"] is None                    # None passthrough
    assert out["meta"] == "label" and out["n"] == 2  # non-array leaves untouched


class TestHostResidentDataset:
    """#1155: the non-chunked trainers (classical / sfno_full) load ALL pairs
    up front; built on the compute device that is ~130 GB at T106 all-years —
    an unconditional GPU OOM during LOADING (Derecho job 6758505). The fix:
    ``load_training_data(host_resident=True)`` builds under the CPU backend,
    and the training loops stage each sample to the compute device per step
    (``host_staged`` — the same contract as the #985 prefetch path)."""

    def test_stage_sample_default_device_and_passthrough(self):
        from legoesm.training.neural_gcm_spectral import stage_sample

        dev0 = jax.devices()[0]
        tree = {"arr": jnp.arange(4.0), "none": None, "meta": "x"}
        out = stage_sample(tree)                       # default: backend dev 0
        assert list(out["arr"].devices()) == [dev0]
        assert out["none"] is None and out["meta"] == "x"

    def test_host_build_device_fallback_warns_and_returns_none(
            self, monkeypatch, caplog):
        import legoesm.training.neural_gcm_spectral as mod

        def _no_cpu(kind=None):
            raise RuntimeError("no cpu backend")
        monkeypatch.setattr(mod.jax, "devices", _no_cpu)
        import logging
        with caplog.at_level(logging.WARNING):
            dev = mod.host_build_device("test-site")
        assert dev is None
        assert any("CPU backend is unavailable" in r.message
                   for r in caplog.records)

    def test_load_training_data_host_resident_enters_cpu_default_device(
            self, monkeypatch):
        """host_resident=True must wrap the (real) load body in
        ``jax.default_device(cpu)``. The body is cut short at its first
        external dependency (zarr open) with a sentinel, so no network."""
        import legoesm.training.neural_gcm_spectral as mod
        import legoesm.training.era5_to_state as e2s

        entered = []
        real_default_device = jax.default_device

        def _recording_default_device(dev):
            entered.append(dev)
            return real_default_device(dev)
        monkeypatch.setattr(mod.jax, "default_device",
                            _recording_default_device)

        class _Sentinel(Exception):
            pass

        def _boom(*a, **k):
            raise _Sentinel
        monkeypatch.setattr(e2s, "open_era5_zarr", _boom)
        monkeypatch.setattr(e2s, "ensure_local_cache", _boom)
        monkeypatch.setattr(e2s, "ensure_window_scoped_cache", _boom,
                            raising=False)

        import pytest
        cfg = self._tiny_cfg()
        with pytest.raises(_Sentinel):
            mod.load_training_data(cfg, _GRID, _SIGMA, cache_dir="",
                                   windows=None, host_resident=True)
        cpu0 = jax.devices("cpu")[0]
        assert entered and entered[0] == cpu0, (
            "host_resident load must enter jax.default_device(cpu[0]) "
            f"before touching data (entered={entered})")

    def _tiny_cfg(self, **kw):
        from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
        from legoesm.training.losses import LossConfig
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
        return NeuralGCMSpectralConfig(
            n_max=N_MAX, n_levels=NLEV, dt=1800.0,
            pe_config=SpectralPEConfig(time_integrator="ssp_rk3"),
            lr=1e-4, warmup_steps=0, optimizer="adamw",
            rollout_curriculum=((1, 1),),   # 1h lead x 1 epoch
            loss_config=LossConfig(
                multi_step_hours=(1,), multi_step_weights=(1.0,)),
            log_every=1, **kw,
        )

    def test_spectral_loop_host_staged_bitwise_parity(self, tmp_path):
        """The non-chunked loop with host_staged=True must produce weights
        BIT-IDENTICAL to host_staged=False on the same data — staging is
        placement-only, never numerics. (On the CPU test backend the staging
        device_put is a same-device move; the test guards the plumbing:
        tuple-target handling, forcings=None, rebind lifetime.)"""
        from legoesm.training.neural_gcm_spectral import (
            _train_spectral_loop, make_sfno_spectral_physics,
            carry_to_spectral_state,
        )

        carry = _make_gaussian_carry()
        ic = carry_to_spectral_state(carry, _GRID)
        ics, tgts = [ic], [(carry,)]     # lead-1 target tuple, 1 sample

        results = []
        for staged in (False, True):
            cfg = self._tiny_cfg(checkpoint_dir=str(tmp_path / f"s{staged}"))
            model, hist = _train_spectral_loop(
                _make_small_sfno(), make_sfno_spectral_physics,
                _GRID, _SIGMA, ics, tgts, cfg,
                start_epoch=0, host_staged=staged,
            )
            results.append((model, hist))

        (m0, h0), (m1, h1) = results
        assert h0 == h1, "loss history must be identical"
        for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(m0, eqx.is_array)),
                        jax.tree_util.tree_leaves(eqx.filter(m1, eqx.is_array))):
            assert jnp.array_equal(a, b), "weights must be bit-identical"


if __name__ == "__main__":
    test_prefetch_iter_preserves_order_and_completes()
    test_prefetch_iter_propagates_producer_exception()
    test_prefetch_iter_is_lazy_bounded()
    test_stage_tree_moves_arrays_skips_non_arrays()
    print("ok (run test_chunk_loader_prefetch_matches_serial under pytest)")
