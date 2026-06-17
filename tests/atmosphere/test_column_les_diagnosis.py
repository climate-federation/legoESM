"""Unit tests for :mod:`legoesm.atmosphere.dynamics.column_les_diagnosis`.

Stage-6 composition: turn a finished column-LES state into a closure
coefficient.  Verifies the top-down→ascending reversal + flux/gradient
co-location with an analytic eddy-diffusivity case, the entrainment structure,
and the method dispatch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.column_les_diagnosis import (
    EddyDiffusivityProfile,
    diagnose_column_coefficient,
    diagnose_eddy_diffusivity,
    diagnose_entrainment,
)
from legoesm.atmosphere.dynamics.les_closure_diagnosis import EntrainmentDiagnosis
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)

_W, _A = 2.0, 0.5
_GAMMA = 0.01  # |dθ/dz| [K/m]; θ DECREASES upward (unstable) so K>0


def _checkerboard_sign(ny=4, nx=4):
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    return np.where((ii + jj) % 2 == 0, 1.0, -1.0)


def _les_state_with_known_K():
    grid = create_plane_grid(nx=4, ny=4, nlev=8, dx=2_000.0, dy=2_000.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(8, H=10_000.0)
    # theta_ref DECREASING upward (unstable): θ = θ0 − γ·z, so ∂⟨θ⟩/∂z = −γ.
    z = jnp.asarray(hc.z_full)
    theta_ref = 300.0 - _GAMMA * z
    hc = hc._replace(theta_ref=theta_ref)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)

    s2d = _checkerboard_sign()
    s_full = jnp.asarray(s2d)[:, :, None] * jnp.ones((4, 4, 8))
    s_half = jnp.asarray(s2d)[:, :, None] * jnp.ones((4, 4, 9))
    tracers = jnp.zeros((4, 4, 8, 3), dtype=jnp.float64).at[..., 0].set(0.01)
    state = state._replace(
        w=state.w.replace(data=_W * s_half),
        theta_prime=state.theta_prime.replace(data=_A * s_full),  # zero mean
        tracers=state.tracers.replace(data=tracers),
    )
    return state, hc


def test_eddy_diffusivity_analytic_and_ascending():
    state, hc = _les_state_with_known_K()
    out = diagnose_eddy_diffusivity(state, hc)
    assert isinstance(out, EddyDiffusivityProfile)
    assert out.K.shape == (7,)  # nlev-1
    # z ascending.
    assert bool(jnp.all(jnp.diff(out.z_m) > 0))
    # K = -w'θ'/(∂⟨θ⟩/∂z) = -(W·A)/(-γ) = W·A/γ at every valid interface.
    expected_K = _W * _A / _GAMMA
    assert bool(jnp.all(out.valid))
    np.testing.assert_allclose(np.asarray(out.K), expected_K, rtol=1e-9)


def test_eddy_diffusivity_stable_is_countergradient_invalid():
    """θ INCREASING upward (stable) + upward heat flux ⇒ counter-gradient ⇒
    K<0 ⇒ flagged invalid (no down-gradient closure)."""
    state, hc = _les_state_with_known_K()
    z = jnp.asarray(hc.z_full)
    hc = hc._replace(theta_ref=300.0 + _GAMMA * z)  # stable now
    out = diagnose_eddy_diffusivity(state, hc)
    assert not bool(jnp.any(out.valid))


def test_entrainment_returns_diagnosis():
    state, hc = _les_state_with_known_K()
    out = diagnose_entrainment(state, hc)
    assert isinstance(out, EntrainmentDiagnosis)
    assert jnp.isfinite(out.entrainment_buoyancy_flux)
    # valid is a boolean scalar.
    assert out.valid.dtype == jnp.bool_


def test_entrainment_matches_manual_reversed_composition():
    """diagnose_entrainment must equal an independent reversed call (validates
    the theta_v build + top-down->ascending reversal wiring)."""
    from legoesm import constants
    from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
        entrainment_velocity_from_buoyancy_flux,
    )
    from legoesm.atmosphere.dynamics.rce_diagnostics import (
        resolved_turbulent_fluxes_plane,
    )

    state, hc = _les_state_with_known_K()
    fluxes = resolved_turbulent_fluxes_plane(state, hc)
    theta_total = hc.theta_ref + state.theta_prime.data
    q_v = state.tracers.data[..., 0]
    coeff = 1.0 / constants.epsilon - 1.0
    thetav_mean = jnp.mean(theta_total * (1.0 + coeff * q_v), axis=(0, 1))
    manual = entrainment_velocity_from_buoyancy_flux(
        fluxes.w_thetav[::-1], thetav_mean[::-1], fluxes.z_half_interior[::-1]
    )
    out = diagnose_entrainment(state, hc)
    assert int(out.inversion_index) == int(manual.inversion_index)
    np.testing.assert_allclose(
        np.asarray(out.w_entrainment), np.asarray(manual.w_entrainment), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(out.delta_thetav), np.asarray(manual.delta_thetav), rtol=1e-12)


def test_dispatch_eddy_diffusivity():
    state, hc = _les_state_with_known_K()
    out = diagnose_column_coefficient(state, hc, method="eddy_diffusivity")
    assert isinstance(out, EddyDiffusivityProfile)


def test_dispatch_entrainment():
    state, hc = _les_state_with_known_K()
    out = diagnose_column_coefficient(state, hc, method="entrainment")
    assert isinstance(out, EntrainmentDiagnosis)


def test_dispatch_unknown_method_raises():
    state, hc = _les_state_with_known_K()
    with pytest.raises(ValueError, match="Unknown column-LES diagnosis method"):
        diagnose_column_coefficient(state, hc, method="mixing_length")


def test_diagnosis_jit():
    state, hc = _les_state_with_known_K()
    out = jax.jit(
        lambda s: diagnose_eddy_diffusivity(s, hc).K
    )(state)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_entrainment_jit():
    state, hc = _les_state_with_known_K()
    out = jax.jit(
        lambda s: diagnose_entrainment(s, hc).w_entrainment
    )(state)
    assert jnp.isfinite(out)
