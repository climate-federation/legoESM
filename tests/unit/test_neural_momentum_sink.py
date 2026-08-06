"""#1464: the learned column arm must be able to carry a momentum sink.

`make_column_physics_fn` returned ZERO vor/div tendencies, i.e. a model with no
surface drag of any kind, while the classical arm it was scored against
inherits `TurbulenceConfig.scheme = "smagorinsky"` whenever a `PhysicsConfig`
is built naming only radiation and convection. The two arms therefore differed
by the presence of a momentum sink, and the T106 campaign's degradation ranked
exactly as that predicts (wind_speed_10m 3.83x, u10 3.02x at 240 h, monotone in
height, z500 the only field that improved).

These tests pin: the confound is real and reachable in code; the new
`momentum_physics_fn` hook actually changes vor/div; the default stays
bit-identical; and the network still owns the thermodynamics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.neural_physics import (  # noqa: E402
    build_column_physics,
    make_column_physics_fn,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.training.neural_gcm_spectral import (  # noqa: E402
    carry_to_spectral_state,
    make_turbulence_only_spectral_physics,
)
from tests.unit.test_learned_column import _mini_spectral_state  # noqa: E402

NLEV = 4


def _setup(seed=0):
    grid = create_gaussian_grid(n_max=10)
    model = build_column_physics(nlev=NLEV, hidden_dim=8, n_layers=2,
                                 residual_scale=0.1,
                                 key=jax.random.PRNGKey(seed))
    carry, sigma = _mini_spectral_state(grid, NLEV)
    return grid, model, carry_to_spectral_state(carry, grid), sigma


def test_the_confound_is_real_default_arm_has_no_momentum_tendency():
    """Documents the defect: with no momentum source, vor/div are EXACTLY 0."""
    grid, model, state, sigma = _setup()
    out = make_column_physics_fn(model, grid)(state, grid, sigma)
    assert float(jnp.max(jnp.abs(out.vor_hat.data))) == 0.0
    assert float(jnp.max(jnp.abs(out.div_hat.data))) == 0.0


def test_the_classical_arm_inherits_a_momentum_sink():
    """The other half of the confound: naming only rad+conv keeps smagorinsky."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    p = PhysicsConfig()
    assert p.turbulence.scheme == "smagorinsky"
    assert PhysicsConfig(radiation=p.radiation,
                         convection=p.convection).turbulence.scheme == \
        "smagorinsky"


def test_momentum_physics_fn_actually_reaches_vor_div():
    grid, model, state, sigma = _setup()
    marker_vor = 3.5e-6
    marker_div = -1.25e-6

    def _mom(st, g, sc):
        return st._replace(
            vor_hat=st.vor_hat.replace(
                data=jnp.full_like(st.vor_hat.data, marker_vor)),
            div_hat=st.div_hat.replace(
                data=jnp.full_like(st.div_hat.data, marker_div)),
            # A source that ALSO emits lnps: must be ignored (see below).
            lnps_hat=st.lnps_hat.replace(
                data=jnp.full_like(st.lnps_hat.data, 7.0e-9)),
        )

    out = make_column_physics_fn(model, grid, momentum_physics_fn=_mom)(
        state, grid, sigma)
    np.testing.assert_allclose(np.asarray(out.vor_hat.data), marker_vor)
    np.testing.assert_allclose(np.asarray(out.div_hat.data), marker_div)
    assert float(np.max(np.abs(np.asarray(out.lnps_hat.data)))) == 0.0, (
        "lnps was taken from the momentum source. Surface stress has no "
        "surface-pressure tendency, and the PE ADDS this to its own "
        "continuity-derived dlnps/dt, so a mean value here breaks dry-mass "
        "conservation with anchoring off (codex, #1464)")


def test_thermodynamics_still_come_from_the_network():
    """The hook must not let the momentum source touch T or the tracers."""
    grid, model, state, sigma = _setup(seed=3)

    def _mom(st, g, sc):
        return st._replace(
            vor_hat=st.vor_hat.replace(
                data=jnp.full_like(st.vor_hat.data, 1e-5)),
            div_hat=st.div_hat.replace(
                data=jnp.full_like(st.div_hat.data, 1e-5)),
            T_hat=st.T_hat.replace(
                data=jnp.full_like(st.T_hat.data, 999.0)),   # must be IGNORED
        )

    base = make_column_physics_fn(model, grid)(state, grid, sigma)
    with_mom = make_column_physics_fn(model, grid, momentum_physics_fn=_mom)(
        state, grid, sigma)
    np.testing.assert_allclose(np.asarray(with_mom.T_hat.data),
                               np.asarray(base.T_hat.data), rtol=0, atol=0)


def test_default_is_bit_identical_to_the_historical_behaviour():
    grid, model, state, sigma = _setup(seed=5)
    a = make_column_physics_fn(model, grid)(state, grid, sigma)
    b = make_column_physics_fn(model, grid, momentum_physics_fn=None)(
        state, grid, sigma)
    for f in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
        np.testing.assert_array_equal(np.asarray(getattr(a, f).data),
                                      np.asarray(getattr(b, f).data))


def _sheared(state, seed=0):
    """A state with real vor/div — smagorinsky is a SHEAR closure, so on the
    quiescent mini-state it returns exactly zero and proves nothing."""
    rng = np.random.default_rng(seed)
    return state._replace(
        vor_hat=state.vor_hat.replace(data=jnp.asarray(
            rng.normal(scale=1e-5, size=state.vor_hat.data.shape))),
        div_hat=state.div_hat.replace(data=jnp.asarray(
            rng.normal(scale=1e-6, size=state.div_hat.data.shape))))


def test_turbulence_only_builder_produces_a_real_momentum_tendency():
    grid, model, state, sigma = _setup(seed=7)
    out = make_turbulence_only_spectral_physics(dt=1200.0)(
        _sheared(state), grid, sigma)
    assert float(jnp.max(jnp.abs(out.vor_hat.data))) > 0.0, (
        "no vorticity tendency — the paired builder supplies no sink, so the "
        "learned arm would still have no drag")
    assert float(jnp.max(jnp.abs(out.div_hat.data))) > 0.0


def test_quiescent_state_gives_zero_so_the_shear_test_is_not_vacuous():
    """Guards the test above: on an unsheared state the same call is 0."""
    grid, model, state, sigma = _setup(seed=7)
    out = make_turbulence_only_spectral_physics(dt=1200.0)(state, grid, sigma)
    assert float(jnp.max(jnp.abs(out.vor_hat.data))) == 0.0


def test_the_sources_own_heat_diffusion_is_discarded_by_the_adapter():
    """smagorinsky ALSO diffuses temperature (measured 1.64e-5 on a sheared
    state). The adapter must take only vor/div/lnps, so the network keeps sole
    ownership of the thermodynamics."""
    grid, model, state, sigma = _setup(seed=9)
    sheared = _sheared(state, seed=1)
    src = make_turbulence_only_spectral_physics(dt=1200.0)
    src_out = src(sheared, grid, sigma)
    assert float(jnp.max(jnp.abs(src_out.T_hat.data))) > 0.0, (
        "the source has no heat tendency here — this test cannot show it is "
        "discarded")
    base = make_column_physics_fn(model, grid)(sheared, grid, sigma)
    with_mom = make_column_physics_fn(
        model, grid, momentum_physics_fn=src)(sheared, grid, sigma)
    np.testing.assert_array_equal(np.asarray(with_mom.T_hat.data),
                                  np.asarray(base.T_hat.data))
    assert float(jnp.max(jnp.abs(with_mom.vor_hat.data))) > 0.0


def test_turbulence_only_builder_rejects_an_unknown_scheme():
    with pytest.raises(Exception):
        make_turbulence_only_spectral_physics(dt=1200.0,
                                              turbulence_scheme="not_a_scheme")


def test_stateful_turbulence_schemes_are_refused():
    """The builder threads no PhysicsState, so a prognostic scheme would
    re-seed its carry from the floor every step — refuse, don't run it wrong."""
    for scheme in ("tke", "mynn25", "clubb"):
        with pytest.raises(ValueError, match="PROGNOSTIC"):
            make_turbulence_only_spectral_physics(dt=1200.0,
                                                  turbulence_scheme=scheme)


def test_campaign_builder_wires_the_drag_when_asked():
    """END-TO-END on the route the T106 campaign actually takes.

    The first version of this fix was DEAD CODE: scale_build built the neural
    arm without the hook, so nothing changed for the benchmark (codex
    Critical). This test fails if that regresses.
    """
    from legoesm.training import scale_build as sb
    import inspect
    src = inspect.getsource(sb.build_scale_case
                            if hasattr(sb, "build_scale_case") else sb)
    assert "momentum_physics_fn=" in src, (
        "scale_build no longer passes momentum_physics_fn — the neural arm "
        "is back to running with no surface drag (#1464)")
    assert "surface_drag" in src, (
        "the opt-in flag is gone; the campaign cannot request the drag")
