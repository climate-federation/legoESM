"""Unit tests for the SFNO primitive-equation dynamical core.

Exercises :class:`~legoesm.atmosphere.dynamics.sfno_pe.SFNOPrimitiveEquationModel`
directly: instantiation with a tiny randomly-initialised SFNO, both
operating modes (``state_update`` and ``hybrid_tendencies``), the
post-hoc dry-air-mass conservation correction, physics coupling, and
end-to-end differentiability.

No pretrained weights are required — every test builds a small random
network (``embed_dim=12``, ``n_blocks=2``) on a T8 Gaussian grid with
3 sigma levels, which is the smallest configuration that still wires
together the encoder, spectral blocks, decoder, channel packing, the
vorticity/divergence <-> u,v transforms and the conservation path.

Backend note: the Gaussian/spectral transforms need float64, so x64 is
enabled below.  On the Apple GPU (mps) backend, which lacks float64, the
spectral grid/transforms are routed to CPU automatically by
``place_spectral_grid``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    isothermal_rest_state_spectral,
)
from legoesm.atmosphere.dynamics.sfno_pe import (
    SFNOPrimitiveEquationModel,
    SFNOPrimitiveEquationConfig,
)
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.channel_packing import PE3DChannelSpec
from legoesm.atmosphere.physics._shared import zero_like_tracers
from legoesm.core.field import Field

# Spectral transforms require float64.
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Fixtures: smallest valid grid / vertical coordinate / network
# =============================================================================

@pytest.fixture(scope="module")
def grid_t8():
    """Small T8 Gaussian grid (14 x 28 grid, 45 SH coeffs)."""
    return create_gaussian_grid(n_max=8)


@pytest.fixture(scope="module")
def sigma_coord():
    """Three-level sigma coordinate."""
    return create_sigma_coordinate(n_levels=3)


@pytest.fixture(scope="module")
def pe_state(grid_t8, sigma_coord):
    """Isothermal rest state with a uniform moisture (q_v) tracer.

    Including ``q_v`` exercises the tracer pack/unpack channel of the
    PE packing, which a dry state would skip.
    """
    nlev = sigma_coord.n_levels
    qv = jnp.full((grid_t8.n_lat, grid_t8.n_lon, nlev), 5.0e-3, dtype=jnp.float64)
    tracers = {
        "q_v": Field(data=qv, name="q_v", dims=("lat", "lon", "level"),
                     units="kg/kg"),
    }
    return isothermal_rest_state_spectral(grid_t8, sigma_coord, tracers=tracers)


@pytest.fixture(scope="module")
def sfno_config(sigma_coord):
    """Tiny SFNO config matched to the PE channel count (4*nlev + 2)."""
    n_channels = PE3DChannelSpec(nlev=sigma_coord.n_levels).n_channels
    return SFNOConfig(
        in_channels=n_channels,
        out_channels=n_channels,
        embed_dim=12,
        n_blocks=2,
        mlp_expansion=2,
    )


def _global_mean_ps(grid, lnps_hat_data):
    """Area-weighted (Gaussian-quadrature) global mean of surface pressure.

    ``p_s = exp(lnps)`` synthesised to the grid, then averaged with the
    Gaussian latitude weights (uniform in longitude).  This is the dry-
    air-mass proxy that ``correct_dry_air_mass`` targets.
    """
    p_s = jnp.exp(sh_synthesis(grid, lnps_hat_data))
    w = grid.weights[:, None]
    return jnp.sum(p_s * w) / (jnp.sum(w) * grid.n_lon)


# =============================================================================
# Channel-count sanity (the model's contract with channel_packing)
# =============================================================================

class TestChannelContract:

    def test_channel_count_matches_state(self, pe_state, sigma_coord):
        """PE channel count must be 4*nlev + 2 (u, v, T, q + lnps, phis)."""
        nlev = pe_state.T_hat.data.shape[-1]
        assert nlev == sigma_coord.n_levels
        assert PE3DChannelSpec(nlev=nlev).n_channels == 4 * nlev + 2


# =============================================================================
# state_update mode
# =============================================================================

class TestStateUpdateMode:

    def test_runs_and_shapes_preserved(self, grid_t8, sigma_coord, pe_state,
                                       sfno_config):
        """state_update step returns a state with identical field shapes/dtypes
        and finite spectral coefficients."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(pe_state, dt=21600.0)

        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
            old = getattr(pe_state, name).data
            new = getattr(new_state, name).data
            assert new.shape == old.shape
            assert new.dtype == old.dtype
            assert jnp.all(jnp.isfinite(new)), f"{name} has NaN/Inf"

    def test_q_v_tracer_round_trips(self, grid_t8, sigma_coord, pe_state,
                                    sfno_config):
        """The moisture (q_v) tracer survives a state_update step with the
        correct grid-space shape and stays finite."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(pe_state, dt=21600.0)
        assert new_state.tracers is not None
        assert "q_v" in new_state.tracers
        q_out = new_state.tracers["q_v"].data
        assert q_out.shape == (grid_t8.n_lat, grid_t8.n_lon, sigma_coord.n_levels)
        assert jnp.all(jnp.isfinite(q_out))

    def test_phis_static_in_state_update(self, grid_t8, sigma_coord, pe_state,
                                         sfno_config):
        """Surface geopotential (static) is carried through unchanged."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(pe_state, dt=21600.0)
        np.testing.assert_array_equal(
            new_state.phis_hat.data, pe_state.phis_hat.data,
        )


# =============================================================================
# hybrid_tendencies mode
# =============================================================================

class TestHybridMode:

    def test_runs_and_shapes_preserved(self, grid_t8, sigma_coord, pe_state,
                                       sfno_config):
        """hybrid_tendencies step (SFNO tendencies + SSP-RK3) returns a
        correctly-shaped, finite state."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="hybrid_tendencies",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(pe_state, dt=600.0)

        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
            old = getattr(pe_state, name).data
            new = getattr(new_state, name).data
            assert new.shape == old.shape
            assert jnp.all(jnp.isfinite(new)), f"{name} has NaN/Inf"

    def test_unknown_mode_raises(self, grid_t8, sigma_coord, pe_state,
                                 sfno_config):
        """An unrecognised mode must raise ValueError (no silent default)."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="not_a_mode",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        with pytest.raises(ValueError, match="Unknown mode"):
            model.step(pe_state, dt=600.0)


# =============================================================================
# Conservation correction (headline property)
# =============================================================================

class TestMassConservation:

    @staticmethod
    def _identity_model(grid, sigma_coord, sfno_config, *, correct_mass):
        """Build a model whose SFNO is the identity map.

        ``residual_prediction=True`` means ``output = input + decoder(...)``.
        Zeroing the decoder weight+bias makes ``decoder(...) == 0``, so the
        SFNO reproduces its input exactly and the state_update prediction
        equals the input state.  On such a (trivially) mass-conserving
        prediction the dry-air-mass corrector must be a no-op.
        """
        assert sfno_config.residual_prediction
        sfno = SFNO(config=sfno_config, grid=grid, key=jax.random.PRNGKey(0))
        sfno = eqx.tree_at(
            lambda m: (m.decoder.weight, m.decoder.bias),
            sfno,
            (jnp.zeros_like(sfno.decoder.weight),
             jnp.zeros_like(sfno.decoder.bias)),
        )
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=correct_mass, correct_moisture_budget=False,
        )
        return SFNOPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma_coord, config=config, sfno_model=sfno,
        )

    def test_correction_is_noop_on_conserving_prediction(
        self, grid_t8, sigma_coord, pe_state, sfno_config,
    ):
        """HEADLINE: with an identity (mass-conserving) prediction, enabling
        the dry-air-mass correction preserves the global-mean surface
        pressure (dry air mass proxy) to spectral-roundtrip tolerance.

        This exercises ``_apply_conservation`` -> ``correct_dry_air_mass``
        and the lnps log/analysis/synthesis round-trip, and asserts the
        right limiting behaviour: a correct corrector leaves a conserving
        prediction essentially untouched (the only residual is the
        truncation error of representing ``log(p_s)`` in the T8 spectrum).
        """
        model = self._identity_model(
            grid_t8, sigma_coord, sfno_config, correct_mass=True,
        )
        new_state = model.step(pe_state, dt=21600.0)

        m_old = _global_mean_ps(grid_t8, pe_state.lnps_hat.data)
        m_new = _global_mean_ps(grid_t8, new_state.lnps_hat.data)
        rel_err = jnp.abs(m_new - m_old) / m_old
        assert jnp.all(jnp.isfinite(new_state.lnps_hat.data))
        assert float(rel_err) < 1e-4, (
            f"global-mean p_s drifted by {float(rel_err):.2e} under the "
            f"mass corrector on a conserving prediction"
        )

    def test_correction_path_is_active(self, grid_t8, sigma_coord, pe_state,
                                       sfno_config):
        """The corrector must actually fire: with a non-trivial (random,
        untrained) prediction, ``correct_mass=True`` changes lnps relative
        to ``correct_mass=False`` (i.e. the conservation branch is wired in,
        not a silent no-op), while keeping the state finite."""
        shared = SFNO(config=sfno_config, grid=grid_t8,
                      key=jax.random.PRNGKey(3))

        cfg_off = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=False, correct_moisture_budget=False,
        )
        cfg_on = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=True, correct_moisture_budget=False,
        )
        m_off = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=cfg_off,
            sfno_model=shared,
        )
        m_on = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=cfg_on,
            sfno_model=shared,
        )
        lnps_off = m_off.step(pe_state, dt=21600.0).lnps_hat.data
        lnps_on = m_on.step(pe_state, dt=21600.0).lnps_hat.data

        assert jnp.all(jnp.isfinite(lnps_on))
        # The mean (l=0,m=0) coefficient is what the uniform shift touches.
        assert not jnp.allclose(lnps_off, lnps_on)


# =============================================================================
# Physics coupling
# =============================================================================

def _zero_physics_with_T_heating(state, grid, sigma_coord):
    """A trivial physics_fn: uniform small temperature tendency, zeros else.

    Signature matches what ``step_with_physics`` calls:
    ``(state, grid, sigma_coord) -> tendencies`` (a SpectralHydrostaticState).

    The tendency pytree must be structurally identical to ``state`` so the
    model's ``jax.tree.map`` over (state, tendency) does not trip on a
    dict-vs-None tracer mismatch — hence ``zero_like_tracers`` (the same
    helper the production spectral-PE physics orchestrator uses).
    """
    def z(field):
        return field.replace(data=jnp.zeros_like(field.data))
    dT = state.T_hat.replace(
        data=jnp.full_like(state.T_hat.data, 1.0e-4 + 0.0j),
    )
    return SpectralHydrostaticState(
        vor_hat=z(state.vor_hat),
        div_hat=z(state.div_hat),
        T_hat=dT,
        lnps_hat=z(state.lnps_hat),
        phis_hat=z(state.phis_hat),
        tracers=zero_like_tracers(state.tracers),
    )


class TestPhysicsCoupling:

    def test_step_with_physics_hybrid(self, grid_t8, sigma_coord, pe_state,
                                      sfno_config):
        """Physics tendencies added to SFNO tendencies before RK integration."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="hybrid_tendencies",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(2),
        )
        new_state = model.step_with_physics(
            pe_state, dt=600.0, physics_fn=_zero_physics_with_T_heating,
        )
        assert new_state.T_hat.data.shape == pe_state.T_hat.data.shape
        assert jnp.all(jnp.isfinite(new_state.T_hat.data))
        assert jnp.all(jnp.isfinite(new_state.vor_hat.data))

    def test_step_with_physics_state_update(self, grid_t8, sigma_coord,
                                            pe_state, sfno_config):
        """In state_update mode physics tendencies are applied additively
        after the SFNO prediction.  dt must equal dt_sfno (the network's
        macro step) — mismatches raise (see TestStateUpdateDtContract)."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(2),
        )
        new_state = model.step_with_physics(
            pe_state, dt=21600.0, physics_fn=_zero_physics_with_T_heating,
        )
        assert new_state.T_hat.data.shape == pe_state.T_hat.data.shape
        assert jnp.all(jnp.isfinite(new_state.T_hat.data))


# =============================================================================
# Differentiability (core legoESM contract)
# =============================================================================

class TestDifferentiability:

    @staticmethod
    def _model(grid, sigma_coord, sfno_config, sfno):
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="hybrid_tendencies",
            correct_mass=False, correct_moisture_budget=False,
        )
        return SFNOPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma_coord, config=config, sfno_model=sfno,
        )

    def test_grad_wrt_input_state(self, grid_t8, sigma_coord, pe_state,
                                  sfno_config):
        """jax.grad of a scalar loss on the stepped state w.r.t. the input
        temperature coefficients is finite and non-trivial."""
        sfno = SFNO(config=sfno_config, grid=grid_t8,
                    key=jax.random.PRNGKey(4))
        model = self._model(grid_t8, sigma_coord, sfno_config, sfno)

        def loss(T_hat_data):
            state = pe_state._replace(
                T_hat=pe_state.T_hat.replace(data=T_hat_data),
            )
            out = model.step(state, dt=600.0)
            # Real, mean-based scalar over complex spectral T.
            return jnp.mean(jnp.abs(out.T_hat.data) ** 2)

        g = jax.grad(loss)(pe_state.T_hat.data)
        assert g.shape == pe_state.T_hat.data.shape
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_grad_wrt_network_params(self, grid_t8, sigma_coord, pe_state,
                                     sfno_config):
        """eqx.filter_grad of a scalar loss w.r.t. the SFNO weights is
        finite and non-trivial for encoder and decoder."""
        sfno = SFNO(config=sfno_config, grid=grid_t8,
                    key=jax.random.PRNGKey(5))

        def loss(net):
            model = self._model(grid_t8, sigma_coord, sfno_config, net)
            out = model.step(pe_state, dt=600.0)
            return jnp.mean(jnp.abs(out.T_hat.data) ** 2)

        grads = eqx.filter_grad(loss)(sfno)
        assert jnp.all(jnp.isfinite(grads.encoder.weight))
        assert jnp.all(jnp.isfinite(grads.decoder.weight))
        assert float(jnp.max(jnp.abs(grads.encoder.weight))) > 0.0
        assert float(jnp.max(jnp.abs(grads.decoder.weight))) > 0.0


# =============================================================================
# Normalization config-guard contracts
# =============================================================================

class TestNormalizationGuards:
    """Fail-fast guards on the normalization config (mirrors the U-Cast
    emulator).  Without these, two latent traps fire silently:
    (1) denormalizing a *tendency* with *state* stats adds the state mean to
        d/dt; (2) ``use_normalization`` without stats silently skips
        (de)normalization, so a normalized checkpoint runs on raw channels."""

    def test_normalization_rejected_in_hybrid(self, grid_t8, sigma_coord,
                                              sfno_config):
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="hybrid_tendencies",
            use_normalization=True, correct_mass=False,
        )
        with pytest.raises(NotImplementedError, match="hybrid_tendencies"):
            SFNOPrimitiveEquationModel(
                grid=grid_t8, sigma_coord=sigma_coord, config=config,
                key=jax.random.PRNGKey(0),
            )

    def test_normalization_requires_stats(self, grid_t8, sigma_coord,
                                          sfno_config):
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            use_normalization=True, correct_mass=False,
        )
        with pytest.raises(ValueError, match="requires norm_stats"):
            SFNOPrimitiveEquationModel(
                grid=grid_t8, sigma_coord=sigma_coord, config=config,
                key=jax.random.PRNGKey(0),
            )


# =============================================================================
# state_update dt contract
# =============================================================================

class TestStateUpdateDtContract:
    """state_update advances the state by exactly ``dt_sfno`` per call; a
    caller ``dt`` that disagrees must raise instead of silently
    desynchronising the caller's clock from the model state (previously
    ``config.dt_sfno`` was never consumed and ANY dt was accepted while
    the network jumped its trained macro step)."""

    @staticmethod
    def _model(grid, sigma_coord, sfno_config):
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="state_update",
            correct_mass=False, correct_moisture_budget=False,
        )
        return SFNOPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )

    def test_wrong_dt_raises(self, grid_t8, sigma_coord, pe_state,
                             sfno_config):
        model = self._model(grid_t8, sigma_coord, sfno_config)
        with pytest.raises(ValueError, match="dt_sfno"):
            model.step(pe_state, dt=3600.0)

    def test_wrong_dt_raises_in_step_with_physics(self, grid_t8, sigma_coord,
                                                  pe_state, sfno_config):
        model = self._model(grid_t8, sigma_coord, sfno_config)
        with pytest.raises(ValueError, match="dt_sfno"):
            model.step_with_physics(
                pe_state, dt=600.0, physics_fn=_zero_physics_with_T_heating,
            )

    def test_matching_dt_accepted(self, grid_t8, sigma_coord, pe_state,
                                  sfno_config):
        model = self._model(grid_t8, sigma_coord, sfno_config)
        out = model.step(pe_state, dt=21600.0)
        assert jnp.all(jnp.isfinite(out.T_hat.data))

    def test_unknown_mode_raises_in_step_with_physics(self, grid_t8,
                                                      sigma_coord, pe_state,
                                                      sfno_config):
        """step_with_physics must mirror step(): unknown mode raises, never
        silently runs the state_update branch (dispatch hardening)."""
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="not_a_mode",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        with pytest.raises(ValueError, match="Unknown mode"):
            model.step_with_physics(
                pe_state, dt=21600.0,
                physics_fn=_zero_physics_with_T_heating,
            )


# =============================================================================
# hybrid_tendencies residual-net semantics
# =============================================================================

class TestHybridResidualSemantics:
    """hybrid_tendencies reads the (residual) SFNO as a NEXT-STATE predictor
    and forms the finite-difference tendency ``(output - input)/dt_sfno``.

    Pins the fix for the mode misread: previously the residual net's
    ~next-state output (``input + delta``) was integrated directly as a
    per-second tendency — an O(state)-magnitude d/dt, wrong by a factor
    ~dt_sfno."""

    def test_identity_net_gives_zero_tendency(self, grid_t8, sigma_coord,
                                              pe_state, sfno_config):
        """HEADLINE: a residual net with a zeroed decoder is the identity
        map (output == input), so the hybrid tendency must be EXACTLY zero
        and one RK step must return the state unchanged.  Under the old
        (buggy) reading the identity net's output ~= the state itself was
        integrated as a tendency, changing the state massively."""
        assert sfno_config.residual_prediction
        sfno = SFNO(config=sfno_config, grid=grid_t8,
                    key=jax.random.PRNGKey(0))
        sfno = eqx.tree_at(
            lambda m: (m.decoder.weight, m.decoder.bias),
            sfno,
            (jnp.zeros_like(sfno.decoder.weight),
             jnp.zeros_like(sfno.decoder.bias)),
        )
        config = SFNOPrimitiveEquationConfig(
            sfno_config=sfno_config, mode="hybrid_tendencies",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            sfno_model=sfno,
        )
        out = model.step(pe_state, dt=600.0)
        for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
            np.testing.assert_allclose(
                np.asarray(getattr(out, name).data),
                np.asarray(getattr(pe_state, name).data),
                rtol=1e-12, atol=1e-12,
                err_msg=f"{name} changed under a zero-tendency identity net",
            )
        np.testing.assert_allclose(
            np.asarray(out.tracers["q_v"].data),
            np.asarray(pe_state.tracers["q_v"].data),
            rtol=1e-12, atol=1e-12,
        )

    def test_hybrid_rejects_non_residual_net(self, grid_t8, sigma_coord,
                                             sfno_config):
        """With residual_prediction=False the raw decoder output has no
        defined scale, so (output - input)/dt_sfno is ill-defined →
        constructing the hybrid model must raise."""
        cfg_nonres = sfno_config._replace(residual_prediction=False)
        config = SFNOPrimitiveEquationConfig(
            sfno_config=cfg_nonres, mode="hybrid_tendencies",
            correct_mass=False, correct_moisture_budget=False,
        )
        with pytest.raises(ValueError, match="residual_prediction"):
            SFNOPrimitiveEquationModel(
                grid=grid_t8, sigma_coord=sigma_coord, config=config,
                key=jax.random.PRNGKey(0),
            )

    def test_state_update_accepts_non_residual_net(self, grid_t8, sigma_coord,
                                                   pe_state, sfno_config):
        """residual_prediction=False + state_update stays valid (the
        run_aimip sfno_full eval and the neural-GCM full-emulator training
        both use exactly this combination)."""
        cfg_nonres = sfno_config._replace(residual_prediction=False)
        config = SFNOPrimitiveEquationConfig(
            sfno_config=cfg_nonres, mode="state_update",
            correct_mass=False, correct_moisture_budget=False,
        )
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        out = model.step(pe_state, dt=21600.0)
        assert jnp.all(jnp.isfinite(out.T_hat.data))
