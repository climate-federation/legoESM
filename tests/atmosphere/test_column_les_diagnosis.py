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
    CEpsProfile,
    ClubbCoefficientProfile,
    EddyDiffusivityProfile,
    PrandtlProfile,
    diagnose_c_eps_coefficient,
    diagnose_clubb_coefficient,
    diagnose_column_coefficient,
    diagnose_eddy_diffusivity,
    diagnose_entrainment,
    diagnose_prandtl_number,
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


def _les_state_with_shear():
    """LES state with a mean-wind shear (for K_m) + checkerboard w (for wp2)."""
    state, hc = _les_state_with_known_K()
    z = jnp.asarray(hc.z_full)
    s2d = _checkerboard_sign()
    u_mean = 0.01 * z                                  # constant shear in u
    u = u_mean[None, None, :] + 0.3 * jnp.asarray(s2d)[:, :, None]
    return state._replace(u=state.u.replace(data=u)), hc


def test_clubb_coefficient_matches_manual_composition():
    """diagnose_clubb_coefficient must equal an independent manual composition of
    the leaf functions — validates the top-down→ascending reversal of ALL arrays,
    the wp2 interior co-location, and the mixing_length wiring."""
    from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
        clubb_coefficient_from_diffusivity,
        momentum_diffusivity_from_fluxes,
    )
    from legoesm.atmosphere.dynamics.rce_diagnostics import (
        resolved_turbulent_fluxes_plane,
        vertical_velocity_variance_plane,
    )
    from legoesm.atmosphere.physics._shared import mixing_length

    state, hc = _les_state_with_shear()
    l_mix_max = 100.0
    out = diagnose_clubb_coefficient(state, hc, l_mix_max=l_mix_max)
    assert isinstance(out, ClubbCoefficientProfile)
    assert out.C_K.shape == (7,) and bool(jnp.all(jnp.diff(out.z_m) > 0))

    fluxes = resolved_turbulent_fluxes_plane(state, hc)
    wp2_half = vertical_velocity_variance_plane(state, hc)
    u_mean = jnp.mean(state.u.data, axis=(0, 1))
    v_mean = jnp.mean(state.v.data, axis=(0, 1))
    z_full = jnp.asarray(hc.z_full)
    Km, km_v = momentum_diffusivity_from_fluxes(
        fluxes.w_u[::-1], fluxes.w_v[::-1], u_mean[::-1], v_mean[::-1], z_full[::-1])
    z_m = fluxes.z_half_interior[::-1]
    CK, valid = clubb_coefficient_from_diffusivity(
        Km, km_v, mixing_length(z_m, l_mix_max), wp2_half[1:-1][::-1])
    np.testing.assert_allclose(np.asarray(out.C_K), np.asarray(CK), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(out.valid), np.asarray(valid))


def test_clubb_coefficient_min_valid_levels_invalidates_column():
    # Requiring more valid levels than exist flags the WHOLE column invalid.
    state, hc = _les_state_with_shear()
    out = diagnose_clubb_coefficient(state, hc, l_mix_max=100.0, min_valid_levels=999)
    assert not bool(jnp.any(out.valid))


def test_dispatch_clubb_coefficient_requires_l_mix_max():
    state, hc = _les_state_with_shear()
    out = diagnose_column_coefficient(
        state, hc, method="clubb_coefficient", l_mix_max=100.0)
    assert isinstance(out, ClubbCoefficientProfile)
    with pytest.raises(ValueError, match="requires l_mix_max"):
        diagnose_column_coefficient(state, hc, method="clubb_coefficient")


def test_prandtl_number_matches_manual_composition():
    """diagnose_prandtl_number = K_m/K_h from the momentum + heat inversions,
    co-located — validates the wiring (heat K reused, momentum reversed)."""
    from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
        momentum_diffusivity_from_fluxes,
        prandtl_number_from_diffusivities,
    )
    from legoesm.atmosphere.dynamics.rce_diagnostics import (
        resolved_turbulent_fluxes_plane,
    )

    state, hc = _les_state_with_shear()
    out = diagnose_prandtl_number(state, hc)
    assert isinstance(out, PrandtlProfile)
    assert out.Pr_t.shape == (7,) and bool(jnp.all(jnp.diff(out.z_m) > 0))

    kh = diagnose_eddy_diffusivity(state, hc)
    fluxes = resolved_turbulent_fluxes_plane(state, hc)
    u_mean = jnp.mean(state.u.data, axis=(0, 1))
    v_mean = jnp.mean(state.v.data, axis=(0, 1))
    z_full = jnp.asarray(hc.z_full)
    Km, km_v = momentum_diffusivity_from_fluxes(
        fluxes.w_u[::-1], fluxes.w_v[::-1], u_mean[::-1], v_mean[::-1], z_full[::-1])
    Pr, valid = prandtl_number_from_diffusivities(Km, km_v, kh.K, kh.valid)
    np.testing.assert_allclose(np.asarray(out.Pr_t), np.asarray(Pr), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(out.valid), np.asarray(valid))


def test_dispatch_prandtl_number():
    state, hc = _les_state_with_shear()
    out = diagnose_column_coefficient(state, hc, method="prandtl_number")
    assert isinstance(out, PrandtlProfile)


def test_c_eps_matches_manual_composition():
    """diagnose_c_eps_coefficient = c_eps_from_budget(K_m, K_h, S2, N2, l, wp2) —
    validates the co-located reversal + the theta_v/N2 wiring."""
    from legoesm import constants
    from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
        c_eps_from_budget,
        mean_gradient_at_interfaces,
        momentum_diffusivity_from_fluxes,
    )
    from legoesm.atmosphere.dynamics.rce_diagnostics import (
        resolved_turbulent_fluxes_plane,
        vertical_velocity_variance_plane,
    )
    from legoesm.atmosphere.physics._shared import mixing_length

    state, hc = _les_state_with_shear()
    l_mix_max = 100.0
    out = diagnose_c_eps_coefficient(state, hc, l_mix_max=l_mix_max)
    assert isinstance(out, CEpsProfile)
    assert out.C_eps.shape == (7,) and bool(jnp.all(jnp.diff(out.z_m) > 0))

    kh = diagnose_eddy_diffusivity(state, hc)
    fluxes = resolved_turbulent_fluxes_plane(state, hc)
    wp2_half = vertical_velocity_variance_plane(state, hc)
    u_mean = jnp.mean(state.u.data, axis=(0, 1))[::-1]
    v_mean = jnp.mean(state.v.data, axis=(0, 1))[::-1]
    z_asc = jnp.asarray(hc.z_full)[::-1]
    theta_total = hc.theta_ref + state.theta_prime.data
    q_v = state.tracers.data[..., 0]
    coeff = 1.0 / constants.epsilon - 1.0
    thetav = jnp.mean(theta_total * (1.0 + coeff * q_v), axis=(0, 1))[::-1]
    Km, kmv = momentum_diffusivity_from_fluxes(
        fluxes.w_u[::-1], fluxes.w_v[::-1], u_mean, v_mean, z_asc)
    shear_sq = (mean_gradient_at_interfaces(u_mean, z_asc) ** 2
                + mean_gradient_at_interfaces(v_mean, z_asc) ** 2)
    N_sq = (constants.g * mean_gradient_at_interfaces(thetav, z_asc)
            / jnp.maximum(0.5 * (thetav[1:] + thetav[:-1]), 1.0))
    z_m = fluxes.z_half_interior[::-1]
    ce, valid = c_eps_from_budget(
        Km, kmv, kh.K, kh.valid, shear_sq, N_sq,
        mixing_length(z_m, l_mix_max), wp2_half[1:-1][::-1])
    np.testing.assert_allclose(np.asarray(out.C_eps), np.asarray(ce), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(out.valid), np.asarray(valid))


def test_dispatch_c_eps_requires_l_mix_max():
    state, hc = _les_state_with_shear()
    out = diagnose_column_coefficient(state, hc, method="c_eps", l_mix_max=100.0)
    assert isinstance(out, CEpsProfile)
    with pytest.raises(ValueError, match="requires l_mix_max"):
        diagnose_column_coefficient(state, hc, method="c_eps")


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
