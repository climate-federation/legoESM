"""Unit tests for the U-Cast primitive-equation emulator.

Exercises :class:`~legoesm.atmosphere.dynamics.neural.ucast_pe.UCastPrimitiveEquationModel`
directly: both operating modes (``state_update`` / ``hybrid_tendencies``), the
moisture-tracer round-trip, post-hoc dry-air-mass conservation, physics
coupling, the MC-Dropout ensemble step (the probabilistic-forecast headline),
two-axis solver dispatch, and end-to-end differentiability.

A tiny U-Cast (``model_channels=8``, 2 levels) on a T8 Gaussian grid with 3
sigma levels is the smallest configuration that still wires together the
U-Net, channel packing, the vorticity/divergence <-> u,v transforms, and the
conservation path.  Mirrors ``tests/unit/test_sfno_pe.py`` so the two
emulators are held to the same contract.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    isothermal_rest_state_spectral,
)
from legoesm.atmosphere.dynamics.neural.ucast_pe import (
    UCastPrimitiveEquationModel,
    UCastPrimitiveEquationConfig,
)
from legoesm.ml.ucast import UCast, UCastConfig, UNetBlock
from legoesm.ml.channel_packing import PE3DChannelSpec
from legoesm.atmosphere.physics._shared import zero_like_tracers
from legoesm.core.field import Field

# Spectral transforms require float64.
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def grid_t8():
    """Small T8 Gaussian grid (14 x 28 grid)."""
    return create_gaussian_grid(n_max=8)


@pytest.fixture(scope="module")
def sigma_coord():
    return create_sigma_coordinate(n_levels=3)


@pytest.fixture(scope="module")
def pe_state(grid_t8, sigma_coord):
    """Isothermal rest state with a uniform moisture (q_v) tracer."""
    nlev = sigma_coord.n_levels
    qv = jnp.full((grid_t8.n_lat, grid_t8.n_lon, nlev), 5.0e-3, dtype=jnp.float64)
    tracers = {
        "q_v": Field(data=qv, name="q_v", dims=("lat", "lon", "level"),
                     units="kg/kg"),
    }
    return isothermal_rest_state_spectral(grid_t8, sigma_coord, tracers=tracers)


@pytest.fixture(scope="module")
def ucast_config(sigma_coord):
    """Tiny U-Cast config matched to the PE channel count (4*nlev + 2)."""
    n_channels = PE3DChannelSpec(nlev=sigma_coord.n_levels).n_channels
    return UCastConfig(
        in_channels=n_channels, out_channels=n_channels,
        model_channels=8, channel_mult=(1, 2), num_blocks=1, attn_levels=(),
        dropout=0.2,
    )


@pytest.fixture(scope="module")
def ucast_config_tendency(ucast_config):
    """Tendency-mode network: hybrid_tendencies requires non-residual output."""
    return ucast_config._replace(residual_prediction=False)


def _global_mean_ps(grid, lnps_hat_data):
    p_s = jnp.exp(sh_synthesis(grid, lnps_hat_data))
    w = grid.weights[:, None]
    return jnp.sum(p_s * w) / (jnp.sum(w) * grid.n_lon)


def _model(grid, sigma_coord, cfg, **kw):
    return UCastPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma_coord, config=cfg, **kw,
    )


# =============================================================================
# Channel-count contract
# =============================================================================

class TestChannelContract:

    def test_channel_count_matches_state(self, pe_state, sigma_coord):
        nlev = pe_state.T_hat.data.shape[-1]
        assert nlev == sigma_coord.n_levels
        assert PE3DChannelSpec(nlev=nlev).n_channels == 4 * nlev + 2


# =============================================================================
# state_update mode
# =============================================================================

class TestStateUpdateMode:

    def test_runs_and_shapes_preserved(self, grid_t8, sigma_coord, pe_state,
                                       ucast_config):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))
        new_state = model.step(pe_state, dt=3600.0)
        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
            old = getattr(pe_state, name).data
            new = getattr(new_state, name).data
            assert new.shape == old.shape
            assert new.dtype == old.dtype
            assert jnp.all(jnp.isfinite(new)), f"{name} has NaN/Inf"

    def test_q_v_tracer_round_trips(self, grid_t8, sigma_coord, pe_state,
                                    ucast_config):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))
        new_state = model.step(pe_state, dt=3600.0)
        assert new_state.tracers is not None and "q_v" in new_state.tracers
        q_out = new_state.tracers["q_v"].data
        assert q_out.shape == (grid_t8.n_lat, grid_t8.n_lon, sigma_coord.n_levels)
        assert jnp.all(jnp.isfinite(q_out))

    def test_phis_static(self, grid_t8, sigma_coord, pe_state, ucast_config):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))
        new_state = model.step(pe_state, dt=3600.0)
        np.testing.assert_array_equal(
            new_state.phis_hat.data, pe_state.phis_hat.data,
        )


# =============================================================================
# hybrid_tendencies mode
# =============================================================================

class TestHybridMode:

    def test_runs_and_shapes_preserved(self, grid_t8, sigma_coord, pe_state,
                                       ucast_config_tendency):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config_tendency, mode="hybrid_tendencies",
            correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))
        new_state = model.step(pe_state, dt=600.0)
        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
            new = getattr(new_state, name).data
            assert new.shape == getattr(pe_state, name).data.shape
            assert jnp.all(jnp.isfinite(new)), f"{name} has NaN/Inf"

    def test_unknown_mode_raises(self, grid_t8, sigma_coord, pe_state,
                                 ucast_config):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="not_a_mode", correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="Unknown mode"):
            model.step(pe_state, dt=600.0)


# =============================================================================
# Documented semantics / config-guard contracts
# =============================================================================

class TestContracts:

    def test_state_update_is_dt_invariant(self, grid_t8, sigma_coord, pe_state,
                                           ucast_config):
        """state_update IS the time-stepping operator (one trained macro-step);
        ``dt`` is ignored in this mode, so two different ``dt`` give the same
        next state.  Documents the intended (SFNO-parity) semantics."""
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=True,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))
        a = model.step(pe_state, dt=3600.0).T_hat.data
        b = model.step(pe_state, dt=21600.0).T_hat.data
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_unwired_moisture_flags_raise(self, grid_t8, sigma_coord,
                                          ucast_config):
        for flag in ("correct_moisture_budget", "clip_q"):
            cfg = UCastPrimitiveEquationConfig(
                ucast_config=ucast_config, mode="state_update", **{flag: True},
            )
            with pytest.raises(NotImplementedError):
                _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))

    def test_hybrid_requires_non_residual(self, grid_t8, sigma_coord,
                                          ucast_config):
        """hybrid_tendencies + residual_prediction=True must be rejected: a
        residual net emits state+delta, not a tendency."""
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config,  # residual_prediction=True (default)
            mode="hybrid_tendencies", correct_mass=False,
        )
        with pytest.raises(ValueError, match="residual_prediction=False"):
            _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))

    def test_normalization_rejected_in_hybrid(self, grid_t8, sigma_coord,
                                              ucast_config_tendency):
        """use_normalization=True in hybrid_tendencies must be rejected:
        state-level stats cannot denormalize a tendency."""
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config_tendency, mode="hybrid_tendencies",
            correct_mass=False, use_normalization=True,
        )
        with pytest.raises(NotImplementedError, match="hybrid_tendencies"):
            _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))

    def test_channel_count_mismatch_rejected(self, grid_t8, sigma_coord):
        """ucast_config channels must equal the PE channel count 4*nlev+2;
        a mismatch must fail fast (else unpack_pe_output reads clamped/wrong
        channels and corrupts lnps)."""
        nch = PE3DChannelSpec(nlev=sigma_coord.n_levels).n_channels
        uc = UCastConfig(in_channels=nch - 1, out_channels=nch - 1,
                         model_channels=8, channel_mult=(1, 2), num_blocks=1,
                         attn_levels=())
        cfg = UCastPrimitiveEquationConfig(ucast_config=uc, mode="state_update")
        with pytest.raises(ValueError, match="PE channel count"):
            _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))

    def test_mismatched_supplied_model_rejected(self, grid_t8, sigma_coord,
                                                ucast_config):
        """A supplied ucast_model whose config differs from config.ucast_config
        must be rejected (else the residual/channel/conditioning guards, which
        read config.ucast_config, would not apply to the real network)."""
        # Residual network, but wrapper config says non-residual + hybrid.
        residual_net = UCast(config=ucast_config, key=jax.random.PRNGKey(0))
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config._replace(residual_prediction=False),
            mode="hybrid_tendencies", correct_mass=False,
        )
        with pytest.raises(ValueError, match="does not match"):
            UCastPrimitiveEquationModel(
                grid=grid_t8, sigma_coord=sigma_coord, config=cfg,
                ucast_model=residual_net)

    def test_normalization_requires_stats(self, grid_t8, sigma_coord,
                                          ucast_config):
        """use_normalization=True without norm_stats must fail fast (else a
        normalized checkpoint silently runs on raw channels)."""
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update",
            use_normalization=True,
        )
        with pytest.raises(ValueError, match="requires norm_stats"):
            _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))

    def test_conditional_channels_rejected(self, grid_t8, sigma_coord):
        nch = PE3DChannelSpec(nlev=sigma_coord.n_levels).n_channels
        uc = UCastConfig(in_channels=nch, out_channels=nch, model_channels=8,
                         channel_mult=(1, 2), num_blocks=1, attn_levels=(),
                         num_conditional_channels=2)
        cfg = UCastPrimitiveEquationConfig(ucast_config=uc, mode="state_update")
        with pytest.raises(NotImplementedError, match="conditioning"):
            _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))


# =============================================================================
# Conservation
# =============================================================================

class TestMassConservation:

    @staticmethod
    def _identity_model(grid, sigma_coord, ucast_config, *, correct_mass):
        """U-Cast with a zero-init head is the identity under residual
        prediction, so the dry-air-mass corrector must be a near no-op."""
        assert ucast_config.residual_prediction
        ucast = UCast(config=ucast_config, key=jax.random.PRNGKey(0))
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update",
            correct_mass=correct_mass,
        )
        return UCastPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma_coord, config=cfg, ucast_model=ucast,
        )

    def test_correction_is_noop_on_conserving_prediction(
        self, grid_t8, sigma_coord, pe_state, ucast_config,
    ):
        model = self._identity_model(
            grid_t8, sigma_coord, ucast_config, correct_mass=True,
        )
        new_state = model.step(pe_state, dt=3600.0)
        m_old = _global_mean_ps(grid_t8, pe_state.lnps_hat.data)
        m_new = _global_mean_ps(grid_t8, new_state.lnps_hat.data)
        rel_err = jnp.abs(m_new - m_old) / m_old
        assert jnp.all(jnp.isfinite(new_state.lnps_hat.data))
        assert float(rel_err) < 1e-4

    def test_correction_path_is_active(self, grid_t8, sigma_coord, pe_state,
                                       ucast_config):
        """A non-trivial (woken) prediction: ``correct_mass=True`` changes lnps
        vs ``correct_mass=False`` (the conservation branch is wired in)."""
        shared = UCast(config=ucast_config, key=jax.random.PRNGKey(3))
        # Wake the zero-init head so the prediction is not the identity.
        shared = eqx.tree_at(
            lambda m: m.out_conv.weight, shared,
            jax.random.normal(jax.random.PRNGKey(4),
                              shared.out_conv.weight.shape) * 0.05,
        )
        cfg_off = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=False,
        )
        cfg_on = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=True,
        )
        m_off = UCastPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=cfg_off, ucast_model=shared)
        m_on = UCastPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=cfg_on, ucast_model=shared)
        lnps_off = m_off.step(pe_state, dt=3600.0).lnps_hat.data
        lnps_on = m_on.step(pe_state, dt=3600.0).lnps_hat.data
        assert jnp.all(jnp.isfinite(lnps_on))
        assert not jnp.allclose(lnps_off, lnps_on)


# =============================================================================
# MC-Dropout ensemble (probabilistic forecast)
# =============================================================================

class TestEnsemble:

    @staticmethod
    def _woken_model(grid, sigma_coord, ucast_config):
        ucast = UCast(config=ucast_config, key=jax.random.PRNGKey(0))
        # Wake both the head and every block's zero-init conv1 so the
        # MC-Dropout signal (in the block main branch, gated by conv1) can
        # actually propagate to the output and create ensemble spread.
        key = jax.random.PRNGKey(1)
        key, k = jax.random.split(key)
        ucast = eqx.tree_at(
            lambda m: m.out_conv.weight, ucast,
            jax.random.normal(k, ucast.out_conv.weight.shape) * 0.1,
        )

        def wake(block, bkey):
            if isinstance(block, UNetBlock):
                return eqx.tree_at(
                    lambda b: b.conv1.weight, block,
                    jax.random.normal(bkey, block.conv1.weight.shape) * 0.1,
                )
            return block

        keys = jax.random.split(key, len(ucast.enc) + len(ucast.dec))
        new_enc = [wake(b, keys[i]) for i, b in enumerate(ucast.enc)]
        new_dec = [wake(b, keys[len(ucast.enc) + i]) for i, b in enumerate(ucast.dec)]
        ucast = eqx.tree_at(lambda m: (m.enc, m.dec), ucast, (new_enc, new_dec))
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=True,
        )
        return UCastPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma_coord, config=cfg, ucast_model=ucast)

    def test_ensemble_shapes(self, grid_t8, sigma_coord, pe_state, ucast_config):
        model = self._woken_model(grid_t8, sigma_coord, ucast_config)
        ens = model.ensemble_step(pe_state, jax.random.PRNGKey(7), n_members=4)
        # Each leaf gains a leading n_members axis.
        assert ens.T_hat.data.shape == (4,) + pe_state.T_hat.data.shape
        assert ens.lnps_hat.data.shape == (4,) + pe_state.lnps_hat.data.shape
        assert jnp.all(jnp.isfinite(ens.T_hat.data))

    def test_ensemble_members_differ(self, grid_t8, sigma_coord, pe_state,
                                     ucast_config):
        """MC-Dropout produces a non-degenerate spread across members."""
        model = self._woken_model(grid_t8, sigma_coord, ucast_config)
        ens = model.ensemble_step(pe_state, jax.random.PRNGKey(7), n_members=4)
        spread = jnp.max(jnp.abs(ens.T_hat.data[0] - ens.T_hat.data[1]))
        assert float(spread) > 0.0

    def test_ensemble_reproducible(self, grid_t8, sigma_coord, pe_state,
                                   ucast_config):
        model = self._woken_model(grid_t8, sigma_coord, ucast_config)
        a = model.ensemble_step(pe_state, jax.random.PRNGKey(7), n_members=3)
        b = model.ensemble_step(pe_state, jax.random.PRNGKey(7), n_members=3)
        np.testing.assert_array_equal(
            np.asarray(a.T_hat.data), np.asarray(b.T_hat.data),
        )

    def test_ensemble_requires_state_update(self, grid_t8, sigma_coord,
                                            ucast_config_tendency, pe_state):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config_tendency, mode="hybrid_tendencies",
            correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="state_update"):
            model.ensemble_step(pe_state, jax.random.PRNGKey(0), n_members=2)


# =============================================================================
# Physics coupling
# =============================================================================

def _zero_physics_with_T_heating(state, grid, sigma_coord):
    def z(field):
        return field.replace(data=jnp.zeros_like(field.data))
    dT = state.T_hat.replace(data=jnp.full_like(state.T_hat.data, 1.0e-4 + 0.0j))
    return SpectralHydrostaticState(
        vor_hat=z(state.vor_hat), div_hat=z(state.div_hat), T_hat=dT,
        lnps_hat=z(state.lnps_hat), phis_hat=z(state.phis_hat),
        tracers=zero_like_tracers(state.tracers),
    )


class TestPhysicsCoupling:

    def test_hybrid(self, grid_t8, sigma_coord, pe_state, ucast_config_tendency):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config_tendency, mode="hybrid_tendencies",
            correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(2))
        new_state = model.step_with_physics(
            pe_state, dt=600.0, physics_fn=_zero_physics_with_T_heating)
        assert new_state.T_hat.data.shape == pe_state.T_hat.data.shape
        assert jnp.all(jnp.isfinite(new_state.T_hat.data))
        assert jnp.all(jnp.isfinite(new_state.vor_hat.data))

    def test_state_update(self, grid_t8, sigma_coord, pe_state, ucast_config):
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config, mode="state_update", correct_mass=False,
        )
        model = _model(grid_t8, sigma_coord, cfg, key=jax.random.PRNGKey(2))
        new_state = model.step_with_physics(
            pe_state, dt=600.0, physics_fn=_zero_physics_with_T_heating)
        assert new_state.T_hat.data.shape == pe_state.T_hat.data.shape
        assert jnp.all(jnp.isfinite(new_state.T_hat.data))


# =============================================================================
# Two-axis dispatch
# =============================================================================

class TestDispatch:

    def test_resolve_solver_name(self):
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(dynamics="hydrostatic", discretization="u_cast")
        assert name == "ucast_primitive_equations"

    def test_get_solver_class(self):
        from legoesm.atmosphere.dynamics import get_solver_class
        cls = get_solver_class("ucast_primitive_equations")
        assert cls is UCastPrimitiveEquationModel

    def test_in_discretization_options(self):
        from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS
        assert "u_cast" in DISCRETIZATION_OPTIONS


# =============================================================================
# Differentiability
# =============================================================================

class TestDifferentiability:

    def test_grad_wrt_input_state(self, grid_t8, sigma_coord, pe_state,
                                  ucast_config_tendency):
        ucast = UCast(config=ucast_config_tendency, key=jax.random.PRNGKey(4))
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config_tendency, mode="hybrid_tendencies",
            correct_mass=False)
        model = UCastPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=cfg, ucast_model=ucast)

        def loss(T_hat_data):
            state = pe_state._replace(T_hat=pe_state.T_hat.replace(data=T_hat_data))
            return jnp.mean(jnp.abs(model.step(state, dt=600.0).T_hat.data) ** 2)

        g = jax.grad(loss)(pe_state.T_hat.data)
        assert g.shape == pe_state.T_hat.data.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_network_params(self, grid_t8, sigma_coord, pe_state,
                                     ucast_config_tendency):
        ucast = UCast(config=ucast_config_tendency, key=jax.random.PRNGKey(5))
        # Wake the head so the gradient through the residual is non-trivial.
        ucast = eqx.tree_at(
            lambda m: m.out_conv.weight, ucast,
            jax.random.normal(jax.random.PRNGKey(6),
                              ucast.out_conv.weight.shape) * 0.1)
        cfg = UCastPrimitiveEquationConfig(
            ucast_config=ucast_config_tendency, mode="hybrid_tendencies",
            correct_mass=False)

        def loss(net):
            model = UCastPrimitiveEquationModel(
                grid=grid_t8, sigma_coord=sigma_coord, config=cfg, ucast_model=net)
            return jnp.mean(jnp.abs(model.step(pe_state, dt=600.0).T_hat.data) ** 2)

        grads = eqx.filter_grad(loss)(ucast)
        assert jnp.all(jnp.isfinite(grads.out_conv.weight))
        assert float(jnp.max(jnp.abs(grads.out_conv.weight))) > 0.0
