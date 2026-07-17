"""RCE diagnostics tests (plane NH CRM)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
    cloud_fraction_profile_plane,
    column_moist_static_energy_plane, column_total_water_plane,
    column_water_vapor_plane, condensate_profile_plane,
    crm_comparison_profiles_plane, domain_mean_profiles_plane,
    precipitation_rate_proxy_plane,
    pseudo_equivalent_potential_temperature,
    resolved_turbulent_fluxes_plane,
    updraft_mass_flux_plane, vertical_velocity_variance_plane,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup():
    grid = create_plane_grid(
        nx=4, ny=4, nlev=8, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(8, H=10_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((4, 4, 8, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)
    tracers = tracers.at[..., 1].set(0.001)
    tracers = tracers.at[..., 2].set(0.0005)
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )
    return state, grid, hc, tm


def test_column_water_vapor_positive_and_finite():
    state, _, hc, _ = _setup()
    cwv = column_water_vapor_plane(state, hc)
    assert cwv.shape == (4, 4)
    assert bool(jnp.all(cwv > 0.0))


def test_column_total_water_includes_all_slots():
    state, _, hc, _ = _setup()
    cwv_only = column_water_vapor_plane(state, hc)
    ctw = column_total_water_plane(state, hc, water_slot_indices=(0, 1, 2))
    assert bool(jnp.all(ctw > cwv_only))


def test_mse_finite_and_per_cell():
    state, _, hc, _ = _setup()
    mse = column_moist_static_energy_plane(state, hc)
    assert mse.shape == (4, 4)
    assert bool(jnp.all(jnp.isfinite(mse)))


def test_mse_increases_with_q_v():
    state, _, hc, _ = _setup()
    mse_initial = float(column_moist_static_energy_plane(state, hc)[0, 0])
    state_wet = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 0].add(0.005),
        ),
    )
    mse_wet = float(column_moist_static_energy_plane(state_wet, hc)[0, 0])
    assert mse_wet > mse_initial


def test_theta_e_proxy_greater_than_theta_when_moist():
    """θ_e_proxy > θ for any positive q_v."""
    state, _, hc, _ = _setup()
    theta_e = pseudo_equivalent_potential_temperature(state, hc)
    theta_total = hc.theta_ref + state.theta_prime.data
    assert bool(jnp.all(theta_e >= theta_total))


def test_theta_e_proxy_returns_nan_for_nonpositive_T():
    """Codex iter-3: T <= 0 → NaN (not silent inf via clipped denom)."""
    state, _, hc, _ = _setup()
    # Drive T to zero by pushing theta_total = 0 via large negative theta_prime.
    theta_kill = -hc.theta_ref[None, None, :] * jnp.ones((4, 4, 8))
    state_bad = state._replace(
        theta_prime=state.theta_prime.replace(data=theta_kill),
    )
    theta_e = pseudo_equivalent_potential_temperature(state_bad, hc)
    assert bool(jnp.all(jnp.isnan(theta_e)))


def test_domain_mean_profiles_shape():
    state, _, hc, _ = _setup()
    profiles = domain_mean_profiles_plane(state, hc)
    assert profiles.T.shape == (8,)
    assert profiles.q_v.shape == (8,)
    assert profiles.theta_e_proxy.shape == (8,)
    assert profiles.T_std.shape == (8,)


def test_domain_mean_std_is_zero_for_uniform_state():
    state, _, hc, _ = _setup()
    profiles = domain_mean_profiles_plane(state, hc)
    np.testing.assert_allclose(np.asarray(profiles.T_std), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(profiles.q_v_std), 0.0, atol=1.0e-10)


def test_cloud_fraction_is_one_when_qc_above_threshold():
    state, _, hc, _ = _setup()
    cf = cloud_fraction_profile_plane(state, hc)
    np.testing.assert_allclose(np.asarray(cf), np.ones(8), rtol=0.0, atol=0.0)


def test_cloud_fraction_zero_at_zero_qc():
    state, _, hc, _ = _setup()
    state = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 1].set(0.0),
        ),
    )
    cf = cloud_fraction_profile_plane(state, hc)
    np.testing.assert_allclose(np.asarray(cf), np.zeros(8), rtol=0.0, atol=0.0)


def test_cloud_fraction_counts_cloud_ice():
    """Cloud fraction must count cloud ICE (slot 3 = anvil), not just q_c — a
    q_c-only count misses the deep-convective anvil (RCEMIP masks q_c+q_i)."""
    state, _, hc, _ = _setup()
    # 4-slot tracer state (qv, qc, qr, qi) with q_c=0 but cloud ice > threshold.
    tr = jnp.zeros((4, 4, 8, 4), dtype=jnp.float64)
    tr = tr.at[..., 3].set(1.0e-3)            # cloud ice (slot 3) above threshold
    state = state._replace(tracers=state.tracers.replace(data=tr))
    cf = cloud_fraction_profile_plane(state, hc)            # default counts ice
    np.testing.assert_allclose(np.asarray(cf), np.ones(8), rtol=0.0, atol=0.0)
    cf_qc_only = cloud_fraction_profile_plane(state, hc, qi_slot=None)
    np.testing.assert_allclose(np.asarray(cf_qc_only), np.zeros(8), rtol=0.0, atol=0.0)


def test_precipitation_proxy_scales_with_q_r():
    state, _, hc, _ = _setup()
    p_1 = precipitation_rate_proxy_plane(state, hc)
    state2 = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 2].multiply(2.0),
        ),
    )
    p_2 = precipitation_rate_proxy_plane(state2, hc)
    np.testing.assert_allclose(np.asarray(p_2 / p_1), 2.0, rtol=1.0e-12)


def test_precipitation_proxy_zero_at_zero_q_r():
    state, _, hc, _ = _setup()
    state = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[..., 2].set(0.0),
        ),
    )
    p = precipitation_rate_proxy_plane(state, hc)
    np.testing.assert_array_equal(np.asarray(p), np.zeros((4, 4)))


def test_diagnostics_jit_compilable():
    state, _, hc, _ = _setup()
    fn = jax.jit(
        lambda s: (
            column_water_vapor_plane(s, hc),
            column_moist_static_energy_plane(s, hc),
            pseudo_equivalent_potential_temperature(s, hc),
            domain_mean_profiles_plane(s, hc),
            cloud_fraction_profile_plane(s, hc),
            precipitation_rate_proxy_plane(s, hc),
        )
    )
    out = fn(state)
    for arr in jax.tree_util.tree_leaves(out):
        assert bool(jnp.all(jnp.isfinite(arr)))


def test_diagnostics_support_jax_grad():
    state, _, hc, _ = _setup()

    def loss_fn(qv_data):
        s = state._replace(
            tracers=state.tracers.replace(
                data=state.tracers.data.at[..., 0].set(qv_data),
            ),
        )
        return jnp.sum(column_moist_static_energy_plane(s, hc) ** 2)

    g = jax.grad(loss_fn)(state.tracers.data[..., 0])
    assert g.shape == state.tracers.data[..., 0].shape
    assert bool(jnp.all(jnp.isfinite(g)))


# Codex iter-2: tracer-slot validation
def test_cwv_rejects_negative_qv_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        column_water_vapor_plane(state, hc, qv_slot=-1)


def test_cwv_rejects_oob_qv_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        column_water_vapor_plane(state, hc, qv_slot=99)


def test_total_water_rejects_duplicate_slots():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="duplicates"):
        column_total_water_plane(state, hc, water_slot_indices=(0, 0))


def test_total_water_rejects_empty():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="at least one slot"):
        column_total_water_plane(state, hc, water_slot_indices=())


def test_cloud_fraction_rejects_negative_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        cloud_fraction_profile_plane(state, hc, qc_slot=-1)


def test_precip_rejects_oob_slot():
    state, _, hc, _ = _setup()
    with pytest.raises(ValueError, match="out of range"):
        precipitation_rate_proxy_plane(state, hc, qr_slot=99)


# Codex iter-2: plane-shape contract
def test_diagnostics_reject_non_plane_layout():
    """4D rho_prime (cubed-sphere shape) must raise."""
    state, _, hc, _ = _setup()
    state_4d = state._replace(
        rho_prime=state.rho_prime.replace(
            data=jnp.zeros((6, 4, 4, 8), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="plane-only"):
        column_water_vapor_plane(state_4d, hc)


def test_diagnostics_reject_mismatched_theta_shape():
    state, _, hc, _ = _setup()
    state_bad = state._replace(
        theta_prime=state.theta_prime.replace(
            data=jnp.zeros((2, 2, 8), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="theta_prime"):
        column_moist_static_energy_plane(state_bad, hc)


def test_diagnostics_reject_mismatched_tracer_shape():
    state, _, hc, _ = _setup()
    state_bad = state._replace(
        tracers=state.tracers.replace(
            data=jnp.zeros((2, 2, 8, 3), dtype=jnp.float64),
        ),
    )
    with pytest.raises(ValueError, match="tracers shape"):
        column_water_vapor_plane(state_bad, hc)


# ---------------------------------------------------------------------------
# Convective-intensity diagnostics (iter-43): w'², updraft mass flux,
# condensate profiles, full comparison bundle.
# ---------------------------------------------------------------------------

_W_CHECKER = 2.0  # m/s checkerboard updraft/downdraft amplitude


def _setup_convective(n_tracers: int = 6):
    """Rest state + a (i+j)-checkerboard w field (=±_W_CHECKER) and a full
    6-slot condensate set. Checkerboard ⇒ analytic w'²=_W_CHECKER², and
    exactly half the cells are updrafts ⇒ M_up = 0.5·rho·w."""
    grid = create_plane_grid(
        nx=4, ny=4, nlev=8, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(8, H=10_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((4, 4, 8, n_tracers), dtype=jnp.float64)
    vals = [0.01, 1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5]
    for s in range(n_tracers):
        tracers = tracers.at[..., s].set(vals[s])
    ii, jj = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
    sign = np.where((ii + jj) % 2 == 0, 1.0, -1.0)            # (4,4), 8 +/8 −
    w = _W_CHECKER * jnp.asarray(sign)[:, :, None] * jnp.ones((4, 4, 9))
    state = state._replace(
        w=state.w.replace(data=w),
        tracers=state.tracers.replace(data=tracers),
    )
    return state, grid, hc, tm


def test_w_variance_checkerboard_is_amplitude_squared():
    """var(±W checkerboard) = W² at every level; shape on the w grid."""
    state, _, hc, _ = _setup_convective()
    w2 = vertical_velocity_variance_plane(state, hc)
    assert w2.shape == (9,)                          # nlev+1, w grid
    np.testing.assert_allclose(
        np.asarray(w2), _W_CHECKER ** 2, rtol=1.0e-12)


def test_w_variance_zero_for_uniform_w():
    state, _, hc, _ = _setup_convective()
    state = state._replace(
        w=state.w.replace(data=jnp.full((4, 4, 9), 3.0)))
    w2 = vertical_velocity_variance_plane(state, hc)
    np.testing.assert_allclose(np.asarray(w2), 0.0, atol=1.0e-12)


def test_updraft_mass_flux_half_area():
    """M_up = <rhow·w·H(w>0)> = 0.5·rho_ref_half·W on the w grid (half the
    cells are updrafts); SAM rhow = base-state ρ at w levels."""
    state, _, hc, _ = _setup_convective()
    m_up = updraft_mass_flux_plane(state, hc)
    assert m_up.shape == (9,)                        # nlev+1, w grid
    expected = 0.5 * np.asarray(hc.rho_ref_half) * _W_CHECKER
    np.testing.assert_allclose(np.asarray(m_up), expected, rtol=1.0e-12)


def test_updraft_mass_flux_threshold_excludes_all():
    """Threshold above the updraft amplitude ⇒ no updraft cells ⇒ zero."""
    state, _, hc, _ = _setup_convective()
    m_up = updraft_mass_flux_plane(state, hc, w_threshold=2.0 * _W_CHECKER)
    np.testing.assert_array_equal(np.asarray(m_up), np.zeros(9))


def test_condensate_profile_full_morrison_slots():
    """cloud=q_c+q_i (1,3); precip=q_r+q_s+q_g (2,4,5)."""
    state, _, hc, _ = _setup_convective(n_tracers=6)
    cond = condensate_profile_plane(state, hc)
    np.testing.assert_allclose(np.asarray(cond.q_cloud), 1.0e-3 + 2.0e-4)
    np.testing.assert_allclose(
        np.asarray(cond.q_precip), 5.0e-4 + 1.0e-4 + 5.0e-5)
    np.testing.assert_allclose(
        np.asarray(cond.q_total), 1.2e-3 + 6.5e-4, rtol=1.0e-12)


def test_condensate_profile_drops_missing_warm_rain_slots():
    """Warm-rain (3 tracers): ice/snow/graupel slots dropped ⇒ cloud=q_c,
    precip=q_r only."""
    state, _, hc, _ = _setup_convective(n_tracers=3)
    cond = condensate_profile_plane(state, hc)
    np.testing.assert_allclose(np.asarray(cond.q_cloud), 1.0e-3)   # q_c only
    np.testing.assert_allclose(np.asarray(cond.q_precip), 5.0e-4)  # q_r only


def test_condensate_rejects_negative_slot():
    """Codex iter-43 C: negative slot raises (catch a mis-specified layout)."""
    state, _, hc, _ = _setup_convective()
    with pytest.raises(ValueError, match="non-negative"):
        condensate_profile_plane(state, hc, cloud_slots=(1, -3))


def test_crm_comparison_bundle_shapes_and_consistency():
    """Assembler matches the leaf helpers + correct grid lengths."""
    state, _, hc, _ = _setup_convective()
    b = crm_comparison_profiles_plane(state, hc)
    for arr in (b.w_var_half, b.updraft_mass_flux_half, b.z_half):
        assert arr.shape == (9,)            # w / interface grid
    for arr in (b.T, b.q_v, b.q_cloud, b.cloud_fraction, b.z_full):
        assert arr.shape == (8,)            # full levels
    # consistency with the leaf helpers
    np.testing.assert_allclose(
        np.asarray(b.w_var_half),
        np.asarray(vertical_velocity_variance_plane(state, hc)))
    np.testing.assert_allclose(
        np.asarray(b.updraft_mass_flux_half),
        np.asarray(updraft_mass_flux_plane(state, hc)))
    assert float(b.max_w) == pytest.approx(_W_CHECKER)
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate(
        [b.T, b.q_v, b.q_cloud, b.q_precip, b.updraft_mass_flux_half]))))


def test_crm_comparison_bundle_jit_and_grad():
    """Bundle is JIT-able + differentiable (AD-safe diagnostics)."""
    state, _, hc, _ = _setup_convective()
    out = jax.jit(lambda s: crm_comparison_profiles_plane(s, hc))(state)
    for arr in jax.tree_util.tree_leaves(out):
        assert bool(jnp.all(jnp.isfinite(arr)))

    def loss(w_data):
        s = state._replace(w=state.w.replace(data=w_data))
        return jnp.sum(crm_comparison_profiles_plane(s, hc).w_var_half)

    g = jax.grad(loss)(state.w.data)
    assert g.shape == state.w.data.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# Resolved turbulent fluxes (stage 5 of docs/COMPARE_REANALYSIS.md)
# ---------------------------------------------------------------------------

def _checkerboard_sign(ny=4, nx=4):
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    return np.where((ii + jj) % 2 == 0, 1.0, -1.0)  # (ny,nx), zero mean


def _correlated_flux_state(W=2.0, A=0.5, B=1e-3, C=3.0):
    """Checkerboard w=W·s, theta_prime=A·s, q_v=q0+B·s, u=C·s, v=0.

    Perfectly (anti)correlated ⇒ analytic <w'φ'> = W·A_phi at every interior
    interface (s²=1, zero horizontal means)."""
    state, grid, hc, tm = _setup_convective()
    s2d = _checkerboard_sign()
    s_full = jnp.asarray(s2d)[:, :, None] * jnp.ones((4, 4, 8))
    s_half = jnp.asarray(s2d)[:, :, None] * jnp.ones((4, 4, 9))
    tracers = state.tracers.data.at[..., 0].set(0.01 + B * np.asarray(s_full))
    state = state._replace(
        w=state.w.replace(data=W * s_half),
        theta_prime=state.theta_prime.replace(data=A * s_full),
        u=state.u.replace(data=C * s_full),
        v=state.v.replace(data=jnp.zeros((4, 4, 8))),
        tracers=state.tracers.replace(data=tracers),
    )
    return state, hc, dict(W=W, A=A, B=B, C=C)


def test_resolved_fluxes_shapes_and_zhalf():
    state, hc, _ = _correlated_flux_state()
    f = resolved_turbulent_fluxes_plane(state, hc)
    for arr in (f.z_half_interior, f.w_theta, f.w_qv, f.w_u, f.w_v, f.w_thetav):
        assert arr.shape == (7,)  # nlev-1 interior interfaces
    np.testing.assert_allclose(
        np.asarray(f.z_half_interior), np.asarray(hc.z_half)[1:-1])


def test_resolved_fluxes_analytic_covariance():
    state, hc, p = _correlated_flux_state()
    f = resolved_turbulent_fluxes_plane(state, hc)
    # <w'θ'> = W·A, <w'q'> = W·B, <w'u'> = W·C, <w'v'> = 0, all interfaces.
    np.testing.assert_allclose(np.asarray(f.w_theta), p["W"] * p["A"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f.w_qv), p["W"] * p["B"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f.w_u), p["W"] * p["C"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f.w_v), 0.0, atol=1e-12)


def test_resolved_buoyancy_flux_exact_cross_term():
    state, hc, p = _correlated_flux_state()
    f = resolved_turbulent_fluxes_plane(state, hc)
    # Exact nonlinear-product result for w=W·s, θ=θ_ref+A·s, q=q0+B·s, <s>=0:
    #   <w'θ_v'> = W·[ A·(1 + c·q0) + c·θ0·B ],  θ0 = θ_ref at the interface.
    coeff = 1.0 / constants.epsilon - 1.0
    q0, W, A, B = 0.01, p["W"], p["A"], p["B"]
    theta_ref = np.asarray(hc.theta_ref)
    theta_ref_iface = 0.5 * (theta_ref[:-1] + theta_ref[1:])  # (nlev-1,)
    expected = W * (A * (1.0 + coeff * q0) + coeff * theta_ref_iface * B)
    np.testing.assert_allclose(np.asarray(f.w_thetav), expected, rtol=1e-12)
    assert bool(jnp.all(f.w_thetav > 0.0))


def test_resolved_flux_bad_uv_shape_raises():
    state, hc, _ = _correlated_flux_state()
    state = state._replace(u=state.u.replace(data=jnp.zeros((4, 4, 9))))
    with pytest.raises(ValueError, match="full-level layout"):
        resolved_turbulent_fluxes_plane(state, hc)


def test_resolved_flux_zero_for_uniform_w():
    state, hc, _ = _correlated_flux_state()
    state = state._replace(w=state.w.replace(data=jnp.full((4, 4, 9), 1.5)))
    f = resolved_turbulent_fluxes_plane(state, hc)
    for arr in (f.w_theta, f.w_qv, f.w_u, f.w_thetav):
        np.testing.assert_allclose(np.asarray(arr), 0.0, atol=1e-12)


def test_resolved_flux_zero_for_uniform_scalar():
    state, hc, _ = _correlated_flux_state()
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=jnp.zeros((4, 4, 8))))
    f = resolved_turbulent_fluxes_plane(state, hc)
    np.testing.assert_allclose(np.asarray(f.w_theta), 0.0, atol=1e-12)


def test_resolved_flux_bad_w_shape_raises():
    state, hc, _ = _correlated_flux_state()
    # w on full levels (nlev) instead of half (nlev+1) must be rejected.
    state = state._replace(w=state.w.replace(data=jnp.zeros((4, 4, 8))))
    with pytest.raises(ValueError, match="half levels"):
        resolved_turbulent_fluxes_plane(state, hc)


def test_resolved_flux_bad_qv_slot_raises():
    state, hc, _ = _correlated_flux_state()
    with pytest.raises(ValueError):
        resolved_turbulent_fluxes_plane(state, hc, qv_slot=99)


def test_resolved_flux_jit_and_grad():
    state, hc, _ = _correlated_flux_state()
    out = jax.jit(lambda s: resolved_turbulent_fluxes_plane(s, hc))(state)
    for arr in jax.tree_util.tree_leaves(out):
        assert bool(jnp.all(jnp.isfinite(arr)))

    def loss(w_data):
        s = state._replace(w=state.w.replace(data=w_data))
        return jnp.sum(resolved_turbulent_fluxes_plane(s, hc).w_theta)

    g = jax.grad(loss)(state.w.data)
    assert g.shape == state.w.data.shape
    assert bool(jnp.all(jnp.isfinite(g)))
