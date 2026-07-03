"""Direct unit tests for the GEOMETRIC EKE closure (Torres et al. 2025,
JAMES, doi:10.1029/2025MS005394) — ``EKEConfig.closure="geometric"``.

Covers: paper-constant defaults, dispatch/validation hardening, the pure
formulas (kappa_gm Eq. 6, kappa_n Eq. 7, R_d App. D, the Eq.-4 dissipation
fold), the budget terms on manufactured states (B_C against a hand-computed
∫kappa·M⁴/N² dz; B_T against a hand-assembled ∫kappa_u|∇u|² dz), positivity
by construction, transport conservation with the GEOMETRIC shim,
differentiability through two closure steps with alpha/C_eps/kappa_u as
TRACED values (calibration readiness), and the model-step integration
(default-closure bit-identity + a geometric channel spin-up).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.physics.lateral_mixing.eke import (
    EKEConfig,
    GeometricConfig,
    eke_apply_local_source,
    geometric_dissipation_length,
    geometric_kappa_gm,
    geometric_kappa_n,
    geometric_rossby_radius,
    validate_eke_config,
    validate_geometric_config,
)


# ---------------------------------------------------------------------------
# Config defaults + validation/dispatch hardening
# ---------------------------------------------------------------------------


def test_geometric_defaults_match_torres_2025():
    """The hand-tuned constants of Torres et al. (2025): alpha=0.04 (Eq. 6),
    C_eps=0.022 (Eq. 4), kappa_u=1500 (Eq. 3), kappa_E=500 (Eq. 5),
    Gamma=0.35 + 40 km L_mix cap (Eq. 7), R_d=0.4·∫Ndz/f in [2,40] km
    (App. D), ∫M²/N floor 1e-10 (Eq. 6), kappa floors 10 (Table 1), and the
    1e-6·H cold start (App. E)."""
    g = GeometricConfig()
    assert g.alpha == 0.04
    assert g.c_eps_geometric == 0.022
    assert g.kappa_u == 1500.0
    assert g.kappa_e == 500.0
    assert g.gamma_n == 0.35
    assert g.l_mix_max == 4.0e4
    assert g.rossby_factor == 0.4
    assert g.r_d_min == 2.0e3 and g.r_d_max == 4.0e4
    assert g.mn_floor == 1.0e-10
    # Caps: the paper states NO upper bound (the DEFAULT 1000 cap is
    # relaxed); 1.5e4 is the legoESM forward-stability safety ABOVE the
    # paper's realized maxima (12,716 / 10,604, Table 1) — see the
    # GeometricConfig provenance comments.
    assert g.kappa_gm_min == 10.0 and g.kappa_gm_max == 1.5e4
    assert g.kappa_n_min == 10.0 and g.kappa_n_max == 1.5e4
    assert g.e0_per_depth == 1.0e-6
    assert g.kappa_n_coupling is False


def test_default_closure_is_eden_greatbatch():
    cfg = EKEConfig()
    assert cfg.closure == "eden_greatbatch"
    assert cfg.geometric is None
    validate_eke_config(cfg)  # default stays valid


def test_validate_rejects_unknown_closure():
    with pytest.raises(ValueError, match="closure"):
        validate_eke_config(EKEConfig(closure="geometricc"))


def test_validate_rejects_geometric_without_subconfig():
    with pytest.raises(ValueError, match="GeometricConfig"):
        validate_eke_config(EKEConfig(closure="geometric"))


def test_validate_rejects_unused_geometric_subconfig():
    with pytest.raises(ValueError, match="silently ignored"):
        validate_eke_config(EKEConfig(geometric=GeometricConfig()))


def test_validate_rejects_geometric_with_eke_3d():
    with pytest.raises(ValueError, match="eke_3d"):
        validate_eke_config(EKEConfig(
            closure="geometric", geometric=GeometricConfig(), eke_3d=True))


def test_validate_rejects_geometric_with_isopycnal_diffusion():
    with pytest.raises(ValueError, match="kappa_n_coupling"):
        validate_eke_config(EKEConfig(
            closure="geometric", geometric=GeometricConfig(),
            isopycnal_diffusion=True))


def test_validate_rejects_geometric_with_eg_source_flags():
    with pytest.raises(ValueError, match="source"):
        validate_eke_config(EKEConfig(
            closure="geometric", geometric=GeometricConfig(),
            source_kdiss_h=True, kdiss_h_flux_form=True))
    with pytest.raises(ValueError, match="source"):
        validate_eke_config(EKEConfig(
            closure="geometric", geometric=GeometricConfig(),
            gm_source_mode="realized"))


def test_validate_geometric_config_bad_params():
    for bad in (
        GeometricConfig(alpha=0.0),
        GeometricConfig(c_eps_geometric=-1.0),
        GeometricConfig(kappa_u=-1.0),
        GeometricConfig(kappa_e=-1.0),
        GeometricConfig(rossby_factor=0.0),
        GeometricConfig(r_d_min=5.0e4),          # > r_d_max
        GeometricConfig(mn_floor=0.0),
        GeometricConfig(kappa_gm_min=-1.0),
        GeometricConfig(gamma_n=0.0),
        GeometricConfig(l_mix_max=0.0),
        GeometricConfig(kappa_n_min=1.0e9),      # > kappa_n_max (1.5e4)
        GeometricConfig(e0_per_depth=-1.0),
    ):
        with pytest.raises(ValueError):
            validate_geometric_config(bad)


# ---------------------------------------------------------------------------
# Pure formulas (Eq. 6, Eq. 7, App. D R_d, Eq. 4 fold)
# ---------------------------------------------------------------------------


def test_kappa_gm_matches_eq6_and_bounds():
    geom = GeometricConfig()
    int_E = jnp.array([1.0e3, 0.0, 5.0e4])          # ∫EKE dz [m³/s²]
    int_MN = jnp.array([0.5, 0.0, 2.0])             # ∫M²/N dz [m/s]
    kappa = geometric_kappa_gm(int_E, int_MN, geom)
    # Eq. 6 closed form away from the bounds.
    np.testing.assert_allclose(float(kappa[0]), 0.04 * 1.0e3 / 0.5, rtol=1e-12)
    np.testing.assert_allclose(float(kappa[2]), 0.04 * 5.0e4 / 2.0, rtol=1e-12)
    # Zero energy → kappa floored at kappa_gm_min (Table 1 floor).
    assert float(kappa[1]) == geom.kappa_gm_min
    # Denominator floor 1e-10: flat isopycnals do not divide by zero — and
    # the kappa_gm_max safety cap binds there (the floored denominator would
    # otherwise give alpha·E/1e-10 ~ 1e8, the probe-verified CFL hazard).
    k_flat = geometric_kappa_gm(jnp.array([1.0]), jnp.array([0.0]), geom)
    assert float(k_flat[0]) == geom.kappa_gm_max
    # With the cap lifted, the floored-denominator closed form is exposed.
    geom_uncapped = GeometricConfig(kappa_gm_max=float("inf"))
    k_unc = geometric_kappa_gm(jnp.array([1.0]), jnp.array([0.0]),
                               geom_uncapped)
    np.testing.assert_allclose(float(k_unc[0]), 0.04 * 1.0 / 1.0e-10,
                               rtol=1e-12)
    # A finite kappa_gm_max (legoESM safety option) caps.
    geom_cap = GeometricConfig(kappa_gm_max=100.0)
    k_cap = geometric_kappa_gm(jnp.array([1.0e6]), jnp.array([0.5]), geom_cap)
    assert float(k_cap[0]) == 100.0


def test_rossby_radius_matches_appendix_d_and_bounds():
    geom = GeometricConfig()
    f = jnp.array([1.0e-4, 1.0e-4, 1.0e-12, -1.0e-4])
    int_N = jnp.array([5.0, 1.0e-9, 5.0, 5.0])
    r_d = geometric_rossby_radius(int_N, f, geom)
    # Closed form 0.4·∫N dz/|f| in the unbounded range.
    np.testing.assert_allclose(float(r_d[0]), 0.4 * 5.0 / 1.0e-4, rtol=1e-12)
    # Lower bound 2 km (weak stratification).
    assert float(r_d[1]) == geom.r_d_min
    # Upper bound 40 km (equator: f→0).
    assert float(r_d[2]) == geom.r_d_max
    # |f|: sign-symmetric.
    assert float(r_d[3]) == float(r_d[0])


def test_kappa_n_matches_eq7_and_caps():
    geom = GeometricConfig()
    H = jnp.array([4000.0, 4000.0])
    int_E = jnp.array([2.0e2, 2.0e2])               # EKE_0 = 0.05 m²/s²
    r_d = jnp.array([3.0e4, 4.0e4])                 # cap binds at l_mix_max
    kappa = geometric_kappa_n(int_E, H, r_d, geom)
    eke0 = 2.0e2 / 4000.0
    np.testing.assert_allclose(
        float(kappa[0]), 0.35 * 3.0e4 * np.sqrt(2.0 * eke0), rtol=1e-10)
    np.testing.assert_allclose(
        float(kappa[1]), 0.35 * 4.0e4 * np.sqrt(2.0 * eke0), rtol=1e-10)
    # L_mix = min(R_d, 40 km): an R_d above the cap is capped.
    geom_low_cap = GeometricConfig(l_mix_max=1.0e4)
    k_cap = geometric_kappa_n(int_E[:1], H[:1], jnp.array([3.0e4]),
                              geom_low_cap)
    np.testing.assert_allclose(
        float(k_cap[0]), 0.35 * 1.0e4 * np.sqrt(2.0 * eke0), rtol=1e-10)
    # Zero energy → floor (Table 1: min 10 m²/s).
    k0 = geometric_kappa_n(jnp.zeros(1), H[:1], r_d[:1], geom)
    assert float(k0[0]) == geom.kappa_n_min


def test_dissipation_fold_reproduces_eq4_rate():
    """eke_apply_local_source with (c_eps=c_eps_geometric, L=R_d·√H)
    reproduces the Eq.-4 implicit update
    E_{n+1} = (E_n + dt·P)/(1 + dt·C_eps·√(E_n/H)/R_d) exactly."""
    geom = GeometricConfig()
    dt = 86400.0
    E = jnp.array([1.0e3, 0.0, 5.0e4])              # ∫EKE dz [m³/s²]
    H = jnp.array([4000.0, 2000.0, 5000.0])
    r_d = jnp.array([3.0e4, 2.0e3, 4.0e4])
    P = jnp.array([1.0e-2, 0.0, 5.0e-1])            # B_C + B_T [m³/s³]
    L_eff = geometric_dissipation_length(r_d, H)
    shim = EKEConfig()._replace(c_eps=geom.c_eps_geometric)
    E_new = eke_apply_local_source(
        E, jnp.zeros_like(E), L_eff, shim, dt, production_override=P)
    rate = geom.c_eps_geometric * np.sqrt(
        (np.asarray(E) + 1e-30) / np.asarray(H)) / np.asarray(r_d)
    expected = (np.asarray(E) + dt * np.asarray(P)) / (1.0 + dt * rate)
    np.testing.assert_allclose(np.asarray(E_new), expected, rtol=1e-10)
    assert np.all(np.asarray(E_new) >= 0.0)


def test_positivity_by_construction_extreme_params():
    """E ≥ 0 ALWAYS, for arbitrary (even huge) C_eps and production, over
    iterated updates — the semi-implicit fold guarantee."""
    rng = np.random.default_rng(7)
    E = jnp.asarray(np.abs(rng.standard_normal(64)) * 1.0e4)
    H = jnp.asarray(rng.uniform(10.0, 6000.0, 64))
    r_d = jnp.asarray(rng.uniform(2.0e3, 4.0e4, 64))
    P = jnp.asarray(np.abs(rng.standard_normal(64)) * 10.0)
    geom = GeometricConfig(c_eps_geometric=50.0)    # absurdly strong sink
    shim = EKEConfig()._replace(c_eps=geom.c_eps_geometric)
    L_eff = geometric_dissipation_length(r_d, H)
    for _ in range(20):
        E = eke_apply_local_source(
            E, jnp.zeros_like(E), L_eff, shim, 86400.0,
            production_override=P)
        assert bool(jnp.all(E >= 0.0)), "E went negative"
        assert bool(jnp.all(jnp.isfinite(E)))


# ---------------------------------------------------------------------------
# B_C on a manufactured column: ∫M⁴/N² dz and ∫M²/N dz hand-computed
# ---------------------------------------------------------------------------


def _manufactured_column(n_lat=4, n_lon=6, nlev=5, N2c=1.0e-5, S0=5.0e-4):
    """Uniform stratification (N² = N2c everywhere) + uniform slope S0:
    σ = N·S0 at every interface, so the column integrals are analytic."""
    from legoesm.ocean.vertical import create_ocean_z_star

    z = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    dz = np.asarray(z.dz_ref)
    jac = jnp.ones((n_lat, n_lon))
    rho_ref = 1024.0
    # rho[k+1] - rho[k] = rho_ref·N2c·dz_half/g  (k=0 is the SHALLOW level;
    # N² = -(g/ρ0)(rho[k]-rho[k+1])/dz_half — eos.compute_buoyancy_frequency)
    dz_half = 0.5 * (dz[:-1] + dz[1:])
    rho_prof = rho_ref * np.concatenate(
        [[0.0], np.cumsum(N2c * dz_half / constants.g)]) + rho_ref
    rho = jnp.asarray(np.broadcast_to(rho_prof, (n_lat, n_lon, nlev)))
    S_x = jnp.full((n_lat, n_lon, nlev - 1), S0)
    S_y = jnp.zeros((n_lat, n_lon, nlev - 1))
    return rho, S_x, S_y, z, jac, dz_half, rho_ref


def test_column_integrals_match_hand_computation():
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_geometric_column_integrals,
    )
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig

    N2c, S0 = 1.0e-5, 5.0e-4
    rho, S_x, S_y, z, jac, dz_half, rho_ref = _manufactured_column(
        N2c=N2c, S0=S0)
    f = jnp.full((4, 6), 1.0e-4)
    int_s2, int_s, int_N, H_col, wet = compute_geometric_column_integrals(
        rho, S_x, S_y, z, jac, f, VisbeckConfig(), rho_ref=rho_ref)
    W = float(np.sum(dz_half))                       # Σ dz_half
    N = np.sqrt(N2c)
    # ∫M⁴/N² dz = ∫(N S)² dz = N²·S0²·W ; ∫M²/N dz = N·S0·W ; ∫N dz = N·W.
    # rtol 1e-6: the shared machinery's sqrt regularisers (|S|²+1e-30,
    # N² floor 1e-30) shift the integrals at the ~1e-7 relative level.
    np.testing.assert_allclose(np.asarray(int_s2), N2c * S0 ** 2 * W,
                               rtol=1e-6)
    np.testing.assert_allclose(np.asarray(int_s), N * S0 * W, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(int_N), N * W, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(H_col), 4000.0, rtol=1e-6)
    assert bool(jnp.all(wet))
    # B_C = kappa_gm·∫M⁴/N² dz against the full hand chain (Eq. 2 + Eq. 6).
    geom = GeometricConfig()
    int_E = jnp.full((4, 6), 1.0e3)
    kappa = geometric_kappa_gm(int_E, int_s, geom)
    b_c = np.asarray(kappa * int_s2)
    kappa_hand = np.clip(0.04 * 1.0e3 / max(N * S0 * W, geom.mn_floor),
                         geom.kappa_gm_min, None)
    np.testing.assert_allclose(b_c, kappa_hand * N2c * S0 ** 2 * W, rtol=1e-6)


def test_column_integrals_zero_on_dry_columns():
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_geometric_column_integrals,
    )
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig

    rho, S_x, S_y, z, jac, _, rho_ref = _manufactured_column()
    jac = jac.at[0, 0].set(0.0)                      # dry column
    f = jnp.full((4, 6), 1.0e-4)
    int_s2, int_s, int_N, H_col, wet = compute_geometric_column_integrals(
        rho, S_x, S_y, z, jac, f, VisbeckConfig(), rho_ref=rho_ref)
    assert not bool(wet[0, 0])
    assert float(int_s2[0, 0]) == 0.0
    assert float(int_s[0, 0]) == 0.0
    assert float(int_N[0, 0]) == 0.0
    assert float(H_col[0, 0]) == 0.0
    for a in (int_s2, int_s, int_N, H_col):
        assert bool(jnp.all(jnp.isfinite(a)))


# ---------------------------------------------------------------------------
# B_T = ∫kappa_u|∇h u_h|² dz (Eq. 3) — reuses the K_diss_h flux-form operator
# ---------------------------------------------------------------------------


def _bt_inputs(seed=3, n_lat=8, n_lon=16, nlev=3):
    from legoesm.grids.latlon import create_latlon_grid

    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.1 * rng.standard_normal((n_lat, n_lon + 1, nlev)))
    u = u.at[:, -1].set(u[:, 0])                     # periodic wrap
    v = jnp.asarray(0.1 * rng.standard_normal((n_lat + 1, n_lon, nlev)))
    v = v.at[0].set(0.0).at[-1].set(0.0)             # polar walls
    dz = jnp.asarray(np.broadcast_to(
        np.array([100.0, 400.0, 3500.0]), (n_lat, n_lon, nlev)))
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.ones((n_lat + 1, n_lon))
    v_mask = v_mask.at[0].set(0.0).at[-1].set(0.0)
    return grid, u, v, dz, mask, u_mask, v_mask


def test_bt_zero_for_uniform_flow_and_nonnegative():
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        geometric_barotropic_production,
    )
    grid, u, v, dz, mask, u_mask, v_mask = _bt_inputs()
    # Uniform zonal flow, no meridional flow: |∇u|² = 0 → B_T = 0.
    # (u constant; v = 0 — the only strictly gradient-free C-grid flow.)
    u0 = jnp.full_like(u, 0.2)
    v0 = jnp.zeros_like(v)
    bt0 = geometric_barotropic_production(
        u0, v0, grid, 1500.0, dz, mask, u_mask, v_mask)
    np.testing.assert_allclose(np.asarray(bt0), 0.0, atol=1e-18)
    # Random flow: ≥ 0 everywhere (positive-definite flux form, no clamp).
    bt = geometric_barotropic_production(
        u, v, grid, 1500.0, dz, mask, u_mask, v_mask)
    assert bool(jnp.all(bt >= 0.0))
    assert bool(jnp.all(jnp.isfinite(bt)))


def test_bt_matches_hand_assembled_depth_integral_and_kappa_linearity():
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        flux_divergence_viscosity_cgrid,
    )
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        geometric_barotropic_production,
    )
    grid, u, v, dz, mask, u_mask, v_mask = _bt_inputs()
    kappa_u = 1500.0
    bt = geometric_barotropic_production(
        u, v, grid, kappa_u, dz, mask, u_mask, v_mask)
    # Hand-assembled: the SAME validated dissipation density, depth-summed.
    _vu, _vv, diss = flux_divergence_viscosity_cgrid(
        u, v, grid, kappa_u, cos_power=0,
        mask=mask, u_mask=u_mask, v_mask=v_mask, want_dissipation=True)
    np.testing.assert_allclose(
        np.asarray(bt), np.asarray(jnp.sum(diss * dz, axis=-1)), rtol=1e-13)
    # Linear in kappa_u (Eq. 3): doubling kappa_u doubles B_T.
    bt2 = geometric_barotropic_production(
        u, v, grid, 2.0 * kappa_u, dz, mask, u_mask, v_mask)
    np.testing.assert_allclose(np.asarray(bt2), 2.0 * np.asarray(bt),
                               rtol=1e-12)


def test_bt_no_spurious_staircase_shear_on_partial_cells():
    """At a topographic STEP a face closed at depth must contribute ZERO B_T
    (free-slip, per-level 3-D face masks) — the same bug class the K_diss_h
    flux-form source fixed (+37% spurious EKE credit on global_4deg).  With
    a partial-cell z_coord the sub-seafloor 'wall shear' credit vanishes."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate,
    )
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        geometric_barotropic_production,
    )
    n_lat, n_lon, nlev = 8, 12, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    z = create_ocean_z_star(n_levels=nlev, H_max=3000.0)
    H = np.full((n_lat, n_lon), 3000.0)
    H[:, : n_lon // 2] = float(-np.asarray(z.z_half_ref)[3])  # shallow half
    zp = create_partial_cell_coordinate(z, jnp.asarray(H))
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.zeros((n_lat + 1, n_lon)).at[1:-1, :].set(1.0)
    dz = jnp.asarray(np.broadcast_to(
        np.asarray(z.dz_ref), (n_lat, n_lon, nlev)))
    # Uniform flow in the WET cells only (zero below the shallow seafloor):
    # the only |∇u|² a 2-D-masked operator sees is the spurious staircase
    # wall shear — the 3-D-masked B_T must be exactly zero.
    act = np.asarray(zp.is_active)
    u3 = np.zeros((n_lat, n_lon + 1, nlev))
    act_u = np.minimum(act, np.concatenate([act[:, -1:], act], axis=1)[:, :-1])
    u3[:, :-1, :] = 0.2 * act_u
    u3[:, -1, :] = u3[:, 0, :]
    u = jnp.asarray(u3)
    v = jnp.zeros((n_lat + 1, n_lon, nlev))
    bt_3d = geometric_barotropic_production(
        u, v, grid, 1500.0, dz, mask, u_mask, v_mask, z_coord=zp)
    bt_2d = geometric_barotropic_production(
        u, v, grid, 1500.0, dz, mask, u_mask, v_mask, z_coord=None)
    assert float(jnp.max(jnp.abs(bt_3d))) < 1e-15, \
        "3-D-masked B_T credits spurious staircase wall shear"
    # Control scale: kappa_u·(Δu/Δx)²·dz ~ 1500·(0.2/4e5)²·600 ≈ 1e-7 m³/s³.
    assert float(jnp.max(bt_2d)) > 1e-9, \
        "2-D-masked control should expose the staircase shear (fixture dead?)"


# ---------------------------------------------------------------------------
# Transport conservation with the GEOMETRIC shim (kappa_e diffusion)
# ---------------------------------------------------------------------------


def test_geometric_transport_conserves_integral_E():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_horizontal_transport,
    )
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    nlat, nlon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(11)
    E = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon))) * 1.0e3)
    U = jnp.asarray(0.1 * rng.standard_normal((nlat, nlon + 1)))
    U = U.at[:, -1].set(U[:, 0])
    V = jnp.asarray(0.1 * rng.standard_normal((nlat + 1, nlon)))
    V = V.at[0].set(0.0).at[-1].set(0.0)
    ones2 = jnp.ones((nlat, nlon))
    geom = GeometricConfig()
    shim = EKEConfig()._replace(c_eps=geom.c_eps_geometric,
                                k_iso=geom.kappa_e)
    tend = eke_horizontal_transport(
        E, U, V, grid, shim, ones2, jnp.ones((nlat, nlon + 1)),
        jnp.ones((nlat + 1, nlon)))
    area = np.asarray(grid.area)
    resid = abs(float(np.sum(np.asarray(tend) * area))) / (
        float(np.sum(np.abs(np.asarray(tend)) * area)) + 1e-300)
    assert resid < 1e-12, "GEOMETRIC E-transport not conservative"


# ---------------------------------------------------------------------------
# Differentiability / calibration readiness: alpha, C_eps, kappa_u traced
# ---------------------------------------------------------------------------


def test_grad_through_two_geometric_steps_wrt_traced_params():
    """Two chained GEOMETRIC budget updates (kappa_gm → B_C (+B_T) → implicit
    D_e fold), differentiated wrt (E0, alpha, c_eps, kappa_u) passed as
    TRACED values — finite + nonzero gradients (the later differentiable-
    calibration entry point)."""
    int_s2 = jnp.full((6,), 2.0e-11)     # ∫M⁴/N² dz [m/s²]
    int_s = jnp.full((6,), 2.0e-3)       # ∫M²/N dz [m/s]
    H = jnp.full((6,), 4000.0)
    r_d = jnp.full((6,), 3.0e4)
    bt_per_kappa = jnp.full((6,), 1.0e-5)  # ∫|∇u|²dz (B_T / kappa_u)
    dt = 86400.0

    def two_steps(E0, alpha, c_eps, kappa_u):
        geom = GeometricConfig(alpha=alpha, c_eps_geometric=c_eps,
                               kappa_u=kappa_u)
        shim = EKEConfig()._replace(c_eps=c_eps)
        L_eff = geometric_dissipation_length(r_d, H)
        E = E0
        for _ in range(2):
            kappa = geometric_kappa_gm(E, int_s, geom)
            P = kappa * int_s2 + kappa_u * bt_per_kappa
            E = eke_apply_local_source(
                E, jnp.zeros_like(E), L_eff, shim, dt,
                production_override=P)
        return jnp.sum(E ** 2) + jnp.sum(
            geometric_kappa_n(E, H, r_d, geom) ** 2)

    # E0 chosen so kappa_gm = alpha·E/∫M²/N = 0.04·500/2e-3 = 1e4 stays
    # INTERIOR to [kappa_gm_min, kappa_gm_max] — a clipped kappa would
    # (correctly) zero the alpha gradient.
    E0 = jnp.full((6,), 5.0e2)
    grads = jax.grad(two_steps, argnums=(0, 1, 2, 3))(
        E0, 0.04, 0.022, 1500.0)
    for name, g in zip(("E0", "alpha", "c_eps", "kappa_u"), grads):
        assert bool(jnp.all(jnp.isfinite(g))), f"non-finite grad wrt {name}"
        assert float(jnp.max(jnp.abs(g))) > 0.0, f"zero grad wrt {name}"


# ---------------------------------------------------------------------------
# Model-step integration: dispatch, default bit-identity, geometric channel
# ---------------------------------------------------------------------------


def _channel(eke_cfg, n_steps=10, dt=1800.0, n_lat=12, n_lon=24, nlev=4):
    """Coarse channel with a meridional T front (same fixture family as
    test_eke.py). Returns (state, model, gm_cfg)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig, Field
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=85.0)
    T = np.asarray(state.T.data)
    lat = np.degrees(np.asarray(grid.lat))
    T = T + 2.0 * np.tanh(lat / 20.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    gm = GMRediConfig(kappa_GM=0.0, kappa_Redi=1.0e3, eke=eke_cfg)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        n_barotropic_substeps=8, enable_runtime_checks=False, gm_redi=gm)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    if eke_cfg is not None:
        if eke_cfg.closure == "geometric":
            units, e0 = "m^3/s^2", eke_cfg.geometric.e0_per_depth * 4000.0
        else:
            units, e0 = "m^2/s^2", eke_cfg.e_min
        state = state._replace(
            eke=Field(data=jnp.full((grid.n_lat, grid.n_lon), e0),
                      name="eke", dims=("lat", "lon"), units=units))
    for _ in range(n_steps):
        state = model.step(state, dt=dt)
    return state, model, gm


def test_model_construction_rejects_unknown_closure():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(8, 16)
    z = create_ocean_z_star(n_levels=3, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(gm_redi=GMRediConfig(
        eke=EKEConfig(closure="geometricc")))
    with pytest.raises(ValueError, match="closure"):
        LatLonCGridOceanModel(grid, z, cfg)


def test_default_closure_bit_identical_to_explicit():
    """EKEConfig() and EKEConfig(closure='eden_greatbatch') step
    bit-identically (the closure literal adds no traced difference)."""
    s1, _m1, _g1 = _channel(EKEConfig(), n_steps=5)
    s2, _m2, _g2 = _channel(EKEConfig(closure="eden_greatbatch"), n_steps=5)
    for name in ("u", "v", "T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(s1, name).data),
            np.asarray(getattr(s2, name).data))
    np.testing.assert_array_equal(
        np.asarray(s1.eke.data), np.asarray(s2.eke.data))


def test_geometric_channel_runs_positive_and_kappa_in_bounds():
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_geometric_step_kappa,
    )
    geom = GeometricConfig(kappa_n_coupling=True)
    eke_cfg = EKEConfig(closure="geometric", geometric=geom)
    state, model, gm = _channel(eke_cfg, n_steps=10)
    E = np.asarray(state.eke.data)
    lm = np.asarray(state.land_mask.data) > 0
    assert np.all(E >= 0.0), "∫EKE dz went negative"
    assert np.all(np.isfinite(E))
    assert np.all(np.isfinite(np.asarray(state.T.data)))
    assert np.all(np.isfinite(np.asarray(state.u.data)))
    assert state.eke.units == "m^3/s^2"
    # The field departed the uniform cold-start (sources/transport active).
    assert float(np.std(E[lm])) > 0.0
    # kappa_gm / kappa_n within the configured bounds on wet columns.
    _E, kgm, kn, _bc, _L, _dz = compute_geometric_step_kappa(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        state.eke.data, model.grid, model.z_coord, gm,
        eos=model.config.eos, eos_linear=model.config.eos_linear,
        mask=state.land_mask.data,
        rho_0=model.config.constants.rho_0, g=model.config.constants.g)
    kgm = np.asarray(kgm); kn = np.asarray(kn)
    assert np.all(kgm[lm] >= geom.kappa_gm_min - 1e-12)
    assert np.all(np.isfinite(kgm))
    assert kn is not None
    assert np.all(kn[lm] >= geom.kappa_n_min - 1e-12)
    assert np.all(np.isfinite(kn))
    # Dry columns contribute exactly zero diffusivity.
    if (~lm).any():
        assert np.all(kgm[~lm] == 0.0) and np.all(kn[~lm] == 0.0)


def test_geometric_cold_start_without_eke_field():
    """state.eke=None cold start: the step builds the paper's e0·H initial
    condition internally and threads a 2-D ∫EKE dz field."""
    geom = GeometricConfig()
    eke_cfg = EKEConfig(closure="geometric", geometric=geom)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(8, 16)
    z = create_ocean_z_star(n_levels=3, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, H_max=4000.0, land_lat_threshold=85.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        gm_redi=GMRediConfig(kappa_GM=0.0, kappa_Redi=1.0e3, eke=eke_cfg))
    model = LatLonCGridOceanModel(grid, z, cfg)
    state = model.step(state, dt=1800.0)
    assert state.eke is not None
    E = np.asarray(state.eke.data)
    assert E.shape == (grid.n_lat, grid.n_lon)
    assert np.all(E >= 0.0) and np.all(np.isfinite(E))
