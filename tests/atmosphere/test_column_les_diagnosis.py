"""Unit tests for :mod:`legoesm.atmosphere.dynamics.les.column_les_diagnosis`.

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

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
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
from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import EntrainmentDiagnosis
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
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        entrainment_velocity_from_buoyancy_flux,
    )
    from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
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
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        clubb_coefficient_from_diffusivity,
        momentum_diffusivity_from_fluxes,
    )
    from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
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


def test_clubb_coefficient_excludes_boundary_wp2():
    """The interior C_K must use ONLY the interior wp2 half-levels (``wp2_half[1:-1]``);
    the two BOUNDARY half-levels (top + surface) must not leak in.  Scaling ONLY the
    boundary-interface w (→ huge boundary wp2, interior wp2 + interior fluxes
    unchanged) must leave EVERY interior C_K identical — a slice that kept a boundary
    (``[:-1]`` or ``[1:]``) would put the huge wp2 at an edge interface and change C_K
    there.  Locks the wp2 co-location PHYSICALLY: ``test_clubb_coefficient_matches_
    manual_composition`` uses the SAME slice (and level-constant wp2), so it cannot."""
    state, hc = _les_state_with_shear()
    w = state.w.data                                   # (4,4,9) half-level w
    # Scale ONLY the top (k=0) and surface (k=-1) half-level interfaces by 100×.
    w_sentinel = w.at[:, :, 0].multiply(100.0).at[:, :, -1].multiply(100.0)
    state_b = state._replace(w=state.w.replace(data=w_sentinel))
    out_a = diagnose_clubb_coefficient(state, hc, l_mix_max=100.0)
    out_b = diagnose_clubb_coefficient(state_b, hc, l_mix_max=100.0)
    np.testing.assert_array_equal(np.asarray(out_a.C_K), np.asarray(out_b.C_K))
    np.testing.assert_array_equal(np.asarray(out_a.valid), np.asarray(out_b.valid))


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
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        momentum_diffusivity_from_fluxes,
        prandtl_number_from_diffusivities,
    )
    from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
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
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        c_eps_from_budget,
        mean_gradient_at_interfaces,
        momentum_diffusivity_from_fluxes,
    )
    from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
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


def test_column_les_realism_turbulent_vs_dead():
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import column_les_realism
    # The sheared/turbulent fixture (checkerboard w) → realistic.
    state, hc = _les_state_with_shear()
    assert bool(column_les_realism(state, hc))
    # A rest state (w≡0 ⇒ wp2≈0) → NOT realistic (no turbulence developed).
    dead = state._replace(w=state.w.replace(data=jnp.zeros_like(state.w.data)))
    assert not bool(column_les_realism(dead, hc))


def test_column_les_realism_nonfinite_is_unrealistic():
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import column_les_realism
    state, hc = _les_state_with_shear()
    blown = state._replace(
        theta_prime=state.theta_prime.replace(
            data=state.theta_prime.data.at[0, 0, 0].set(jnp.nan)))
    assert not bool(column_les_realism(blown, hc))


def test_column_les_realism_rejects_thermo_drift():
    """iter 66: a turbulent + finite LES whose mean θ drifted off the GCM column
    reference is rejected (the §9 temperature/MSE-drift gap)."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import column_les_realism
    # The shear fixture's θ' is a zero-MEAN checkerboard (pure fluctuation) — the
    # mean state sits on the reference, so it passes (turbulent + finite + consistent).
    state, hc = _les_state_with_shear()
    assert bool(column_les_realism(state, hc))
    # Add a uniform +6 K offset to θ' → the mean state drifted 6 K (> the 3 K
    # default) — rejected, even though it stays turbulent + finite. Turbulent
    # FLUCTUATIONS are unchanged (the offset is horizontally uniform), so only the
    # mean-drift term fires.
    drifted = state._replace(theta_prime=state.theta_prime.replace(
        data=state.theta_prime.data + 6.0))
    assert not bool(column_les_realism(drifted, hc))
    # The drift is the ONLY failure: a looser configured threshold accepts it.
    assert bool(column_les_realism(drifted, hc, theta_drift_rms_max_K=20.0))


def test_column_les_realism_rejects_moisture_blowup():
    """iter 67: a turbulent + finite + θ-consistent LES whose q_v ran away (a
    finite-but-unphysical moisture blow-up the finite check misses) is rejected."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import column_les_realism
    state, hc = _les_state_with_shear()
    assert bool(column_les_realism(state, hc))   # q_v ≈ 0.01 kg/kg → physical
    # Drive q_v to 0.1 kg/kg (100 g/kg, ~2.5x any physical value) → rejected, even
    # though it stays turbulent + finite + θ-consistent.
    wet = state._replace(tracers=state.tracers.replace(
        data=state.tracers.data.at[..., 0].set(0.1)))
    assert not bool(column_les_realism(wet, hc))
    # The moisture cap is the ONLY failure: a looser configured cap accepts it.
    assert bool(column_les_realism(wet, hc, q_v_max=0.2))
    # A large-NEGATIVE q_v (centered advection can overshoot; q>=0 is NOT guaranteed)
    # is also a runaway → rejected (a tiny undershoot would pass the -1e-3 tolerance).
    neg = state._replace(tracers=state.tracers.replace(
        data=state.tracers.data.at[..., 0].set(-0.05)))
    assert not bool(column_les_realism(neg, hc))


def test_column_les_realism_optin_rh_cap():
    """iter 68: the OPT-IN supersaturation cap (rh_max). The shared mock uses an
    UNPHYSICAL uniform q (= 0.01 kg/kg), grossly supersaturated aloft where q_sat→0
    — so it passes the default gate (rh_max off) but is REJECTED when the cap is
    enabled, which is exactly why the cap is off by default. A PHYSICAL q (sub-
    saturated everywhere) passes even with the cap on."""
    from legoesm import constants
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import column_les_realism
    from legoesm.atmosphere.dynamics.crm.rce_diagnostics import temperature_3d_plane
    from legoesm.thermo import saturation_mixing_ratio

    state, hc = _les_state_with_shear()
    assert bool(column_les_realism(state, hc))                  # rh_max off → passes
    assert not bool(column_les_realism(state, hc, rh_max=1.5))  # uniform q supersat aloft

    # A physical q = 0.5·q_sat (RH ≈ 0.5 at every level) passes the enabled cap.
    temp = temperature_3d_plane(state, hc)
    p = constants.p_ref * jnp.asarray(hc.exner_ref) ** (1.0 / constants.kappa)
    q_phys = 0.5 * saturation_mixing_ratio(temp, p)
    physical = state._replace(tracers=state.tracers.replace(
        data=state.tracers.data.at[..., 0].set(q_phys)))
    assert bool(column_les_realism(physical, hc, rh_max=1.5))


def test_gate_diagnosis_realism_invalidates():
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import gate_diagnosis_realism
    prof = ClubbCoefficientProfile(
        z_m=jnp.array([1.0, 2.0, 3.0]), C_K=jnp.array([0.4, 0.5, 0.6]),
        valid=jnp.array([True, True, True]))
    assert bool(jnp.all(gate_diagnosis_realism(prof, jnp.asarray(True)).valid))
    assert not bool(jnp.any(gate_diagnosis_realism(prof, jnp.asarray(False)).valid))


def test_realism_gate_nan_diagnosis_contained_to_finite_field():
    """Codex top-trap: a blown-up LES gives NaN diagnosis values; with the realism
    gate firing (valid all-False) the ASSEMBLED field is FINITE (background), for
    BOTH a profile (masked-sum) and the scalar entrainment (assemble masks invalid)."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import gate_diagnosis_realism
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import EntrainmentDiagnosis
    from legoesm.training.feedback_assembly import assemble_feedback_field

    class _Rec:
        flat_index = 0

    nan = jnp.asarray(jnp.nan)
    prof = ClubbCoefficientProfile(
        z_m=jnp.array([1.0, 2.0]), C_K=jnp.array([nan, nan]),
        valid=jnp.array([True, True]))
    gated_prof = gate_diagnosis_realism(prof, jnp.asarray(False))
    f1 = assemble_feedback_field([_Rec()], [gated_prof], (2, 2),
                                 method="clubb_coefficient", background=0.4)
    assert bool(jnp.all(jnp.isfinite(f1))) and bool(jnp.allclose(f1, 0.4))

    ent = EntrainmentDiagnosis(
        inversion_index=jnp.asarray(0), z_inversion=nan,
        entrainment_buoyancy_flux=nan, delta_thetav=nan,
        w_entrainment=nan, valid=jnp.asarray(True))
    gated_ent = gate_diagnosis_realism(ent, jnp.asarray(False))
    f2 = assemble_feedback_field([_Rec()], [gated_ent], (2, 2),
                                 method="entrainment", background=0.5)
    assert bool(jnp.all(jnp.isfinite(f2))) and bool(jnp.allclose(f2, 0.5))


# ---------------------------------------------------------------------------
# Forward/inverse round-trip: the LES diagnosis MUST invert the integrator's
# ACTUAL forward closure (else OSSE parameter recovery is impossible, no matter
# how good the LES). This is non-circular: it imports clubb_eddy_diffusivity
# (the SAME forward clubb_lite_turbulence integrates) and inverts it with
# clubb_coefficient_from_diffusivity (the SAME inverse diagnose_clubb_coefficient
# uses) — so a change to EITHER form that de-syncs them fails here.
# ---------------------------------------------------------------------------

def _roundtrip_setup():
    from legoesm.atmosphere.physics._shared import mixing_length

    l_mix_max = 250.0
    z = jnp.linspace(20.0, 3000.0, 12, dtype=jnp.float64)   # ascending interior heights [m]
    l_mix = mixing_length(z, l_mix_max)                     # the GCM's Blackadar length
    wp2 = jnp.linspace(0.5, 3.0, 12, dtype=jnp.float64)     # resolved w'^2 [m^2/s^2] > floor
    return l_mix, wp2


def test_clubb_forward_inverse_round_trip_scalar():
    """Km = clubb_eddy_diffusivity(C_K, l, sqrt_wp2) then C_K = Km/(l*sqrt_wp2) recovers the
    EXACT scalar C_K — the algebraic guarantee underlying perfect-model recovery (clause 6)."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        clubb_coefficient_from_diffusivity,
    )
    from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_eddy_diffusivity

    l_mix, wp2 = _roundtrip_setup()
    for c_k_true in (0.1, 0.3, 0.55, 1.0):
        km = clubb_eddy_diffusivity(c_k_true, l_mix, jnp.sqrt(wp2))   # integrator forward
        c_k, valid = clubb_coefficient_from_diffusivity(             # diagnosis inverse
            km, jnp.ones_like(km, dtype=bool), l_mix, wp2)
        assert bool(jnp.all(valid))
        np.testing.assert_allclose(np.asarray(c_k), c_k_true, rtol=1e-12)


def test_clubb_forward_inverse_round_trip_per_column():
    """The per-column C_K field path (the LES-informed correction) round-trips too: a
    (ncol,) C_K broadcast through the forward is recovered at EVERY level by the inverse."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        clubb_coefficient_from_diffusivity,
    )
    from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_eddy_diffusivity

    l_mix_1d, wp2_1d = _roundtrip_setup()
    ncol = 3
    l_mix = jnp.broadcast_to(l_mix_1d, (ncol, l_mix_1d.shape[0]))
    wp2 = jnp.broadcast_to(wp2_1d, (ncol, wp2_1d.shape[0]))
    c_k_col = jnp.asarray([0.2, 0.35, 0.5], dtype=jnp.float64)         # per-column field
    km = clubb_eddy_diffusivity(c_k_col, l_mix, jnp.sqrt(wp2))
    c_k, valid = clubb_coefficient_from_diffusivity(
        km, jnp.ones_like(km, dtype=bool), l_mix, wp2)
    assert bool(jnp.all(valid))
    # each column's recovered C_K is its true field value at every level
    expected = np.broadcast_to(np.asarray(c_k_col)[:, None], np.asarray(c_k).shape)
    np.testing.assert_allclose(np.asarray(c_k), expected, rtol=1e-12)


def test_round_trip_self_test_is_non_vacuous():
    """Non-vacuity (Domain-Architect tripwire doctrine): if the forward used a DIFFERENT
    form (l^2 instead of l), the genuine inverse would NOT recover C_K — proving the
    round-trip above genuinely pins the form, not a tautology."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        clubb_coefficient_from_diffusivity,
    )

    l_mix, wp2 = _roundtrip_setup()
    c_k_true = 0.3
    km_wrong = c_k_true * (l_mix ** 2) * jnp.sqrt(wp2)               # a DESYNCED forward
    c_k, _ = clubb_coefficient_from_diffusivity(
        km_wrong, jnp.ones_like(km_wrong, dtype=bool), l_mix, wp2)
    assert not np.allclose(np.asarray(c_k), c_k_true, rtol=1e-6)     # must NOT recover


def test_prandtl_forward_inverse_round_trip():
    """K_h = clubb_heat_diffusivity(K_m, Pr_t, l) then Pr_t = K_m/K_h recovers the EXACT
    Pr_t — pins the integrator's heat-diffusivity form against the diagnosis inverse (the
    second of the three multi-coefficient closures, after C_K)."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        prandtl_number_from_diffusivities,
    )
    from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_heat_diffusivity

    l_mix, wp2 = _roundtrip_setup()
    k_m = 0.4 * l_mix * jnp.sqrt(wp2)                         # any positive K_m profile
    ok = jnp.ones_like(k_m, dtype=bool)
    for pr_t_true in (0.5, 1.0, 1.6):
        k_h = clubb_heat_diffusivity(k_m, pr_t_true, l_mix)  # integrator forward
        pr_t, valid = prandtl_number_from_diffusivities(k_m, ok, k_h, ok)  # diagnosis inverse
        assert bool(jnp.all(valid))
        np.testing.assert_allclose(np.asarray(pr_t), pr_t_true, rtol=1e-12)


def test_prandtl_round_trip_non_vacuous():
    """Non-vacuity: a desynced forward (K_h = K_m * Pr_t instead of / Pr_t) would NOT be
    recovered by the genuine inverse — the round-trip genuinely pins the form."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        prandtl_number_from_diffusivities,
    )

    l_mix, wp2 = _roundtrip_setup()
    k_m = 0.4 * l_mix * jnp.sqrt(wp2)
    ok = jnp.ones_like(k_m, dtype=bool)
    pr_t_true = 1.6
    k_h_wrong = k_m * pr_t_true                              # DESYNCED forward (× not ÷)
    pr_t, _ = prandtl_number_from_diffusivities(k_m, ok, k_h_wrong, ok)
    assert not np.allclose(np.asarray(pr_t), pr_t_true, rtol=1e-6)


def test_c_eps_forward_inverse_round_trip():
    """The wp2-budget inversion round-trips: build the steady-state production that
    clubb_wp2_dissipation_rate(C_eps) sustains, realise it through clubb_wp2_production
    (exercising BOTH the shear AND buoyancy terms), and recover the EXACT C_eps. This pins
    the most subtle diagnosis (a budget inversion, not a ratio) against the integrator's
    actual production + dissipation forms."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import c_eps_from_budget
    from legoesm.atmosphere.physics.turbulence.clubb_lite import (
        clubb_wp2_dissipation_rate,
        clubb_wp2_production,
    )

    l_mix, wp2 = _roundtrip_setup()                          # l_mix > 1 m so l_safe == l_mix
    sqrt_wp2 = jnp.sqrt(wp2)
    ok = jnp.ones_like(wp2, dtype=bool)
    for c_eps_true in (0.06, 0.2, 0.6):
        # forward: the net production that sustains wp2 at this C_eps (rate * wp2)
        p_target = clubb_wp2_dissipation_rate(c_eps_true, sqrt_wp2, l_mix) * wp2
        # realise P through BOTH production terms: buoyancy = 10% of P, shear = the rest
        n2 = -0.1 * p_target                                 # unstable (−K_h·N² > 0)
        k_h = jnp.ones_like(wp2)
        s2 = jnp.ones_like(wp2)
        k_m = 0.9 * p_target                                 # K_m·S² = 0.9·P
        # the integrator's production form must reproduce the target (shared helper)
        np.testing.assert_allclose(
            np.asarray(clubb_wp2_production(k_m, k_h, s2, n2)), np.asarray(p_target),
            rtol=1e-12)
        c_eps, valid = c_eps_from_budget(k_m, ok, k_h, ok, s2, n2, l_mix, wp2)  # inverse
        assert bool(jnp.all(valid))
        np.testing.assert_allclose(np.asarray(c_eps), c_eps_true, rtol=1e-12)


def test_c_eps_round_trip_non_vacuous():
    """Non-vacuity: a desynced forward dissipation (C_eps·wp2/ℓ — power 1 not 3/2) gives a
    production the genuine wp2^{3/2} inverse does NOT map back to C_eps."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import c_eps_from_budget

    l_mix, wp2 = _roundtrip_setup()
    ok = jnp.ones_like(wp2, dtype=bool)
    c_eps_true = 0.3
    p_wrong = c_eps_true * wp2 / l_mix                       # DESYNCED (wp2, not wp2^{3/2})
    s2 = jnp.ones_like(wp2)
    n2 = jnp.zeros_like(wp2)
    c_eps, _ = c_eps_from_budget(p_wrong, ok, jnp.ones_like(wp2), ok, s2, n2, l_mix, wp2)
    assert not np.allclose(np.asarray(c_eps), c_eps_true, rtol=1e-6)


def test_realism_breakdown_each_flag_is_load_bearing():
    """LESRealismBreakdown surfaces WHICH realism term failed (campaign observability, iter
    508). A single-failure state must set EXACTLY its own flag False (the others True) and
    overall False — so the operator can tell laminar from blown-up from drifted from wet."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
        column_les_realism,
        column_les_realism_breakdown,
    )

    state, hc = _les_state_with_shear()

    healthy = column_les_realism_breakdown(state, hc)
    assert all(bool(f) for f in (healthy.turbulent, healthy.finite,
                                 healthy.thermo_consistent, healthy.moisture_physical,
                                 healthy.rh_ok, healthy.overall))

    # dead (w≡0) → ONLY turbulent fails
    dead = column_les_realism_breakdown(
        state._replace(w=state.w.replace(data=jnp.zeros_like(state.w.data))), hc)
    assert not bool(dead.turbulent) and not bool(dead.overall)
    assert bool(dead.finite) and bool(dead.thermo_consistent) and bool(dead.moisture_physical)

    # blown up (a NaN) → ONLY finite fails
    blown = column_les_realism_breakdown(state._replace(
        theta_prime=state.theta_prime.replace(
            data=state.theta_prime.data.at[0, 0, 0].set(jnp.nan))), hc)
    assert not bool(blown.finite) and not bool(blown.overall)
    assert bool(blown.moisture_physical)

    # θ-drifted (+6 K uniform) → ONLY thermo_consistent fails (fluctuations unchanged)
    drifted = column_les_realism_breakdown(state._replace(
        theta_prime=state.theta_prime.replace(data=state.theta_prime.data + 6.0)), hc)
    assert not bool(drifted.thermo_consistent) and not bool(drifted.overall)
    assert bool(drifted.turbulent) and bool(drifted.finite) and bool(drifted.moisture_physical)

    # moisture runaway (q_v=0.1) → ONLY moisture_physical fails
    wet = column_les_realism_breakdown(state._replace(
        tracers=state.tracers.replace(data=state.tracers.data.at[..., 0].set(0.1))), hc)
    assert not bool(wet.moisture_physical) and not bool(wet.overall)
    assert bool(wet.turbulent) and bool(wet.finite) and bool(wet.thermo_consistent)

    # overall is bit-identical to the public column_les_realism bool (the bool API delegates)
    assert bool(column_les_realism(state, hc)) == bool(healthy.overall)
    assert bool(column_les_realism(
        state._replace(w=state.w.replace(data=jnp.zeros_like(state.w.data))), hc)) is False


def test_summarize_realism_breakdowns_counts_each_mode():
    """summarize_realism_breakdowns aggregates a STACKED breakdown into per-mode rejection
    counts (iter 509) — the campaign-report 'WHY are columns invalid' line. Failure counts
    are per-criterion (a column can fail several), independent of n_rejected."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
        LESRealismBreakdown,
        summarize_realism_breakdowns,
    )

    bd = LESRealismBreakdown(
        turbulent=jnp.asarray([True, False, True, True]),
        finite=jnp.asarray([True, True, False, True]),
        thermo_consistent=jnp.asarray([True, True, True, True]),
        moisture_physical=jnp.asarray([True, True, True, False]),
        rh_ok=jnp.asarray([True, True, True, True]),
        overall=jnp.asarray([True, False, False, False]),   # AND of the columns above
    )
    s = summarize_realism_breakdowns(bd)
    assert (s.n_total, s.n_realistic, s.n_rejected) == (4, 1, 3)
    assert s.n_not_turbulent == 1
    assert s.n_not_finite == 1
    assert s.n_moisture_runaway == 1
    assert s.n_thermo_drift == 0
    assert s.n_supersaturated == 0
    assert all(isinstance(v, int) for v in s)               # host-side ints, for a log line


def test_summarize_over_vmapped_breakdowns_end_to_end():
    """The intended operator pattern composes: vmap column_les_realism_breakdown over a STACK
    of worst-column LES states, then summarize. A healthy + a dead (w≡0) column → 1 realistic,
    1 rejected for being laminar."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
        column_les_realism_breakdown,
        summarize_realism_breakdowns,
    )

    state, hc = _les_state_with_shear()
    dead = state._replace(w=state.w.replace(data=jnp.zeros_like(state.w.data)))
    stacked = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), state, dead)

    bd = jax.vmap(lambda s: column_les_realism_breakdown(s, hc))(stacked)
    summary = summarize_realism_breakdowns(bd)
    assert summary.n_total == 2
    assert summary.n_realistic == 1            # the healthy column
    assert summary.n_rejected == 1
    assert summary.n_not_turbulent == 1        # the dead column, for being laminar
    assert summary.n_not_finite == 0


def test_mask_diagnosis_above_excludes_sponge_levels():
    """mask_diagnosis_above (iter 513, 'keep the diagnosis below the sponge') invalidates
    profile levels at/above z_max; a no-op for z_max=None and for a scalar diagnosis without
    a z_m profile (entrainment)."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
        ClubbCoefficientProfile,
        mask_diagnosis_above,
    )

    z_m = jnp.asarray([100.0, 500.0, 1000.0, 1800.0])     # ascending interface heights [m]
    prof = ClubbCoefficientProfile(
        z_m=z_m, C_K=jnp.asarray([0.3, 0.3, 0.3, 0.3]),
        valid=jnp.asarray([True, True, True, True]))

    # domain_top 2000, relax 0.25 → sponge base 1500 m: only the 1800 m level is in the sponge.
    masked = mask_diagnosis_above(prof, 2000.0 * (1.0 - 0.25))
    np.testing.assert_array_equal(np.asarray(masked.valid), [True, True, True, False])
    np.testing.assert_array_equal(np.asarray(masked.C_K), np.asarray(prof.C_K))  # value kept

    # z_max=None → unchanged (byte-identical default path)
    assert mask_diagnosis_above(prof, None) is prof

    # a scalar diagnosis (no z_m) → unchanged (entrainment is a single inversion level)
    from types import SimpleNamespace
    scalar = SimpleNamespace(w_entrainment=jnp.asarray(0.1), valid=jnp.asarray(True))
    assert mask_diagnosis_above(scalar, 1500.0) is scalar
