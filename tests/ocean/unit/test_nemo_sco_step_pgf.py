"""NEMO ``hpg_sco`` staircase PGF (``pgf_scheme="nemo_sco"``) — issue #1226.

The DINO momentum-budget instrumentation localized the ACC deficit to the
discrete pressure force at topography-adjacent u-faces (channel wall pgf
NEMO +0.586e9 vs legoESM −0.816e9 m^4/s^2 — a sign flip — while interior pgf
matched to 1.5%).  Root cause: the legacy ``adcroft`` PGF on the masked-zco
staircase is the eta=0 along-level gradient ONLY; NEMO ``hpg_sco``
(dynhpg.F90 5.0.1:340-390, the DINO ``namdyn_hpg`` selection) additionally
carries the qco ``(1+r3t)`` thickness stretch of the hydrostatic integral
and the ``gdept_z0`` slope-correction term (zuap) — the eta-proportional
terms that transmit the discrete topographic form stress at steps.

Gates here (cheap, analytic — the two-column "form stress" case from the
issue work order):

* **Two-column step exactness**: uniform rho' + tilted eta over a staircase
  of ANY unequal depths must produce the exact baroclinic face force
  ``-g*(rho'/rho0)*grad(eta)`` at EVERY wet face — NEMO's stencil telescopes
  to this because ``e3w(1)=2*gdept(1)``, ``e3w(k)=gdept(k)-gdept(k-1)``
  (zgr_lib.F90 depth_to_e3) make ``e3w(1)/2 + sum e3w = gdept(k)``.  The
  legacy adcroft scheme gives EXACTLY ZERO there (the missing form stress —
  the non-vacuous self-test).
* **eta=0 reduction**: at eta=0 both new terms vanish identically, so
  nemo_sco is BIT-IDENTICAL to adcroft on the staircase (the certified
  rest-state behaviour is untouched).
* **Staircase rest**: homogeneous T,S at rest on the staircase -> exactly
  zero PGF tendency.
* **Dispatch hardening**: nemo_sco with the midpoint ``cell_integral``
  quadrature or a pure z* coordinate fails LOUDLY (fn-entry + model-level
  ``_validate_config``), never a silent fall-through.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import LinearEOSConfig, make_eos_fn
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_full_step_coordinate,
    create_ocean_z_star,
)

NX, NY, NZ = 8, 6, 8
H_MAX = 2000.0
DX_M = 5.0e4
RHO_0 = 1000.0
ALPHA_T = 2.0e-4
T_REF_C = 10.0
DT_C = -5.0            # T = T_REF + DT_C everywhere -> rho' = +RHO_0*ALPHA*5
LATC = -45.0
F0 = 2.0 * constants.Omega * np.sin(np.radians(LATC))


def _grid():
    return create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX_M, dy_m=DX_M, f0=F0, beta=0.0,
        y_origin_m=-NY * DX_M / 2, x_origin_m=-NX * DX_M / 2,
        cartesian_pseudo_lat=True)


def _staircase(bottom_level_2d):
    """Full-step staircase coordinate (NEMO ln_zco masked z-levels)."""
    z_ref = create_ocean_z_star(n_levels=NZ, H_max=H_MAX)
    return create_full_step_coordinate(z_ref, jnp.asarray(bottom_level_2d))


def _cfg(pgf_scheme="nemo_sco", pgf_quadrature="nemo_trapezoid"):
    return LatLonCGridOceanConfig.from_flat(
        eos="linear",
        eos_linear=LinearEOSConfig(
            rho_ref=RHO_0, alpha_T=ALPHA_T, beta_S=0.0, T_ref=T_REF_C),
        rho_0=RHO_0, g=constants.g,
        A_h=0.0, K_h=0.0, bottom_drag_r=0.0,
        pgf_scheme=pgf_scheme, pgf_quadrature=pgf_quadrature,
        n_barotropic_substeps=2, enable_runtime_checks=False,
    )


def _state(grid, coord, eta_2d, T_uniform_C=T_REF_C + DT_C):
    """All-ocean rest state on the staircase with prescribed eta and
    UNIFORM T, S (flat isopycnals: rho' constant everywhere).

    T, S, eta are cast to float64: the rest-state builder follows the
    finite-volume f32 precision policy, and an f32 EOS rounds
    ``1 - alpha*(T-T_ref)`` at ~6e-5 relative — 5 orders above the f64
    telescoping residual this analytic gate pins."""
    H_bathy = jnp.sum(coord.h_partial, axis=-1)
    wall = jnp.ones((NY, NX), dtype=jnp.float64)
    st = rest_state_latlon_cgrid_ocean(
        grid, coord, land_mask_override=wall, H_bathy_override=H_bathy)
    return st._replace(
        T=st.T.replace(data=jnp.full(st.T.data.shape, T_uniform_C,
                                     dtype=jnp.float64)),
        S=st.S.replace(data=jnp.full(st.S.data.shape, 35.0,
                                     dtype=jnp.float64)),
        eta=st.eta.replace(data=jnp.asarray(eta_2d, dtype=jnp.float64)),
    )


def _pgf_diag(coord, eta_2d, cfg, T_uniform_C=T_REF_C + DT_C):
    grid = _grid()
    state = _state(grid, coord, eta_2d, T_uniform_C=T_uniform_C)
    _, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, coord, cfg, dt=300.0, diagnose_momentum=True)
    return grid, np.asarray(diag.KE_PGF_u.data), np.asarray(diag.KE_PGF_v.data)


def _rho_hat():
    """The constant density anomaly rho(T,S) - rho_0 the EOS produces."""
    eos = make_eos_fn("linear", LinearEOSConfig(
        rho_ref=RHO_0, alpha_T=ALPHA_T, beta_S=0.0, T_ref=T_REF_C))
    rho = float(eos(jnp.asarray(T_REF_C + DT_C), jnp.asarray(35.0),
                    jnp.asarray(0.0)))
    return rho - RHO_0


def _x_staircase(shallow_level):
    """Deep west half, shallow east half (steps at the mid and wrap faces)."""
    bl = np.full((NY, NX), NZ - 1, dtype=np.int32)
    bl[:, NX // 2:] = shallow_level
    return bl


def _y_staircase(shallow_level):
    bl = np.full((NY, NX), NZ - 1, dtype=np.int32)
    bl[NY // 2:, :] = shallow_level
    return bl


def test_two_column_step_form_stress_x():
    """Uniform rho' + zonally tilted eta over an x-staircase: the baroclinic
    face force is EXACTLY -g*(rho'/rho0)*deta/dx at every wet u-face —
    including the step-adjacent (wall) faces of unequal-depth column pairs —
    and is independent of the step depths.  The legacy adcroft scheme gives
    exactly zero (the missing discrete form stress, #1226)."""
    grid = _grid()
    eta = 0.5 * np.sin(2.0 * np.pi * (np.arange(NX) + 0.5) / NX)[None, :]
    eta = np.broadcast_to(eta, (NY, NX)).copy()

    coord = _staircase(_x_staircase(3))
    _, pgf_u, _ = _pgf_diag(coord, eta, _cfg())
    u_mask, _ = compute_face_masks_3d(coord.is_active, grid)
    u_mask = np.asarray(u_mask) > 0.5

    deta_dx = np.asarray(gradient_x_cgrid(jnp.asarray(eta), grid))
    expected = (-(_rho_hat() / RHO_0) * constants.g
                * deta_dx[:, :, None] * np.ones((1, 1, NZ)))
    np.testing.assert_allclose(
        pgf_u[u_mask], expected[u_mask], rtol=1e-10, atol=1e-16)
    # masked (dry / below-step) faces carry no tendency
    assert np.all(pgf_u[~u_mask] == 0.0)

    # Depth-independence: a different shallow depth gives the SAME wet-face
    # force on the common wet faces (the telescoping is exact for any step).
    coord5 = _staircase(_x_staircase(5))
    _, pgf_u5, _ = _pgf_diag(coord5, eta, _cfg())
    np.testing.assert_allclose(
        pgf_u5[u_mask], expected[u_mask], rtol=1e-10, atol=1e-16)

    # Non-vacuous: the legacy adcroft scheme transmits NONE of this force
    # (uniform rho' -> eta=0 p' is horizontally uniform -> gradient 0).
    _, pgf_u_leg, _ = _pgf_diag(coord, eta, _cfg(pgf_scheme="adcroft"))
    assert np.all(pgf_u_leg == 0.0)


def test_two_column_step_form_stress_y():
    """Meridional mirror: y-staircase + eta(y) tilt -> exact
    -g*(rho'/rho0)*deta/dy at every wet v-face (the zvap/zhpj branch)."""
    grid = _grid()
    eta = 0.4 * np.sin(np.pi * (np.arange(NY) + 0.5) / NY)[:, None]
    eta = np.broadcast_to(eta, (NY, NX)).copy()

    coord = _staircase(_y_staircase(3))
    _, _, pgf_v = _pgf_diag(coord, eta, _cfg())
    _, v_mask = compute_face_masks_3d(coord.is_active, grid)
    v_mask = np.asarray(v_mask) > 0.5

    deta_dy = np.asarray(gradient_y_cgrid(jnp.asarray(eta), grid))
    expected = (-(_rho_hat() / RHO_0) * constants.g
                * deta_dy[:, :, None] * np.ones((1, 1, NZ)))
    np.testing.assert_allclose(
        pgf_v[v_mask], expected[v_mask], rtol=1e-10, atol=1e-16)
    assert np.all(pgf_v[~v_mask] == 0.0)

    _, _, pgf_v_leg = _pgf_diag(coord, eta, _cfg(pgf_scheme="adcroft"))
    assert np.all(pgf_v_leg == 0.0)


def test_f90_recurrence_oracle_nonuniform_rho():
    """Direct dynhpg oracle with FULLY NONUNIFORM rho' (varying in lat, lon
    AND k), unequal step depths and nonzero eta (codex r3 MED): the uniform-
    rho' telescope tests cannot catch a vertical-weight/pairing error that
    preserves the free-surface identity.  Here zhpi+zuap is transliterated
    from dynhpg.F90 5.0.1:340-390 + the qco macros — surface formula,
    (rhd(k)+rhd(k-1)) pairing, per-column (1+r3t) e3w stretch, gdept_z0
    slope term, below-seafloor rhd mask, and the metric division — and every
    wet u/v face/level must match the model's KE_PGF to roundoff."""
    grid_ = _grid()
    bl = np.full((NY, NX), NZ - 1, dtype=np.int32)
    bl[:, NX // 2:] = 3                     # x-steps
    bl[NY // 2:, : NX // 4] = 5             # + y-steps in the deep half
    coord = _staircase(bl)

    rng = np.random.default_rng(11)
    eta = (0.4 * np.sin(2.0 * np.pi * (np.arange(NX) + 0.5) / NX)[None, :]
           + 0.2 * np.cos(np.pi * (np.arange(NY) + 0.5) / NY)[:, None])
    T3 = (T_REF_C - 4.0
          + 2.0 * rng.standard_normal((NY, NX, NZ)))     # varies in ALL dims
    H_bathy = jnp.sum(coord.h_partial, axis=-1)
    wall = jnp.ones((NY, NX), dtype=jnp.float64)
    st = rest_state_latlon_cgrid_ocean(
        grid_, coord, land_mask_override=wall, H_bathy_override=H_bathy)
    st = st._replace(
        T=st.T.replace(data=jnp.asarray(T3, dtype=jnp.float64)),
        S=st.S.replace(data=jnp.full(st.S.data.shape, 35.0,
                                     dtype=jnp.float64)),
        eta=st.eta.replace(data=jnp.asarray(eta, dtype=jnp.float64)),
    )
    _, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        st, grid_, coord, _cfg(), dt=300.0, diagnose_momentum=True)
    pgf_u = np.asarray(diag.KE_PGF_u.data)
    pgf_v = np.asarray(diag.KE_PGF_v.data)

    # ---- oracle: literal dynhpg recurrence in numpy ----
    g = constants.g
    rho = RHO_0 * (1.0 - ALPHA_T * (T3 - T_REF_C))       # linear EOS, beta_S=0
    rho_p = rho - RHO_0
    active = np.asarray(coord.is_active)
    rho_m = np.where(active, rho_p, 0.0)                 # NEMO masked rhd*rho0
    # gdept ladder: cast the (policy-f32) coordinate values to f64 BEFORE
    # differencing — the model's quadrature does exactly that (hi_precision
    # t_q = asarray(ladder, float64)); an f32-native subtraction re-rounds
    # e3w at ~2e-5 m (=> ~2e-4 Pa in P, 7e-8 relative — the probe signal).
    t = np.abs(np.asarray(coord.z_full_ref)).astype(np.float64)
    ht0 = np.asarray(jnp.sum(coord.h_partial, axis=-1))
    r3t = np.where(ht0 > 0.0, eta / np.maximum(ht0, 1.0), 0.0)
    # per-column hydrostatic recurrence P(k) [Pa]: surface e3w(1)=2*gdept(1),
    # interior e3w(k)=gdept(k)-gdept(k-1), pair (rhd(k)+rhd(k-1)); the qco
    # (1+r3t) e3w factor is column-constant -> applied as one stretch.
    P = np.zeros_like(rho_m)
    P[..., 0] = 0.5 * g * (2.0 * t[0]) * rho_m[..., 0]
    for k in range(1, NZ):
        P[..., k] = (P[..., k - 1]
                     + 0.5 * g * (t[k] - t[k - 1])
                     * (rho_m[..., k] + rho_m[..., k - 1]))
    P = P * (1.0 + r3t)[..., None]                       # qco e3w stretch
    Z = t[None, None, :] * (1.0 + r3t)[..., None] - eta[..., None]  # gdept_z0

    # Face assembly through the SHARED horizontal operators (gradient_*_cgrid
    # / interp_cell_to_*face): the vertical structure under test — surface
    # weight, (rhd(k)+rhd(k-1)) pairing, ladder increments, (1+r3t) stretch,
    # gdept_z0, below-seafloor mask — is all hand-built above, while the
    # trivial 2-point delta/metric plumbing is common (a hand-copied metric
    # formula only re-tests the grid's dx representation, at ~1e-5).
    P_j = jnp.asarray(P)
    Z_j = jnp.asarray(Z)
    R_j = jnp.asarray(rho_m)
    expected_u = np.asarray(
        -(gradient_x_cgrid(P_j, grid_)
          - constants.g * interp_cell_to_uface(R_j)
          * gradient_x_cgrid(Z_j, grid_)) / RHO_0)
    expected_v = np.asarray(
        -(gradient_y_cgrid(P_j, grid_)
          - constants.g * interp_cell_to_vface(R_j, grid_)
          * gradient_y_cgrid(Z_j, grid_)) / RHO_0)
    u_mask, v_mask = compute_face_masks_3d(coord.is_active, grid_)
    u_mask = np.asarray(u_mask) > 0.5
    v_mask = np.asarray(v_mask) > 0.5
    np.testing.assert_allclose(
        pgf_u[u_mask], expected_u[u_mask], rtol=1e-12, atol=1e-19)
    np.testing.assert_allclose(
        pgf_v[v_mask], expected_v[v_mask], rtol=1e-12, atol=1e-19)


def test_eta_zero_reduces_to_adcroft_bitwise():
    """At eta=0 the stretch is *1.0 and the gdept_z0 slope difference is a
    1-D-ladder difference == 0, so nemo_sco must be BIT-IDENTICAL to the
    legacy adcroft path on a stratified staircase state (the certified
    rest-state behaviour is untouched)."""
    grid = _grid()
    coord = _staircase(_x_staircase(3))
    eta0 = np.zeros((NY, NX))
    H_bathy = jnp.sum(coord.h_partial, axis=-1)
    wall = jnp.ones((NY, NX), dtype=jnp.float64)
    # stratified (non-uniform) T from the rest-state builder
    state = rest_state_latlon_cgrid_ocean(
        grid, coord, land_mask_override=wall, H_bathy_override=H_bathy)
    state = state._replace(eta=state.eta.replace(data=jnp.asarray(eta0)))
    out = {}
    for scheme in ("nemo_sco", "adcroft"):
        _, diag = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, coord, _cfg(pgf_scheme=scheme),
            dt=300.0, diagnose_momentum=True)
        out[scheme] = (np.asarray(diag.KE_PGF_u.data),
                       np.asarray(diag.KE_PGF_v.data))
    np.testing.assert_array_equal(out["nemo_sco"][0], out["adcroft"][0])
    np.testing.assert_array_equal(out["nemo_sco"][1], out["adcroft"][1])


def test_staircase_rest_stays_at_rest():
    """Homogeneous T,S at rest (eta=0) on the staircase: the PGF tendency is
    exactly zero (issue #1226 gate b)."""
    coord = _staircase(_x_staircase(3))
    eta0 = np.zeros((NY, NX))
    _, pgf_u, pgf_v = _pgf_diag(coord, eta0, _cfg())
    assert np.all(pgf_u == 0.0)
    assert np.all(pgf_v == 0.0)


def test_nemo_sco_requires_trapezoid_quadrature():
    """nemo_sco + the midpoint cell_integral p' would break the telescoping
    (spurious rest-eta PGF at steps) -> loud fn-entry error, and the same
    pairing is rejected at model construction."""
    coord = _staircase(_x_staircase(3))
    eta = np.zeros((NY, NX))
    with pytest.raises(ValueError, match="nemo_trapezoid"):
        _pgf_diag(coord, eta, _cfg(pgf_quadrature="cell_integral"))
    with pytest.raises(ValueError, match="nemo_trapezoid"):
        LatLonCGridOceanModel._validate_config(
            _cfg(pgf_quadrature="cell_integral"))
    # the valid pairing passes model validation
    LatLonCGridOceanModel._validate_config(_cfg())


def test_nemo_sco_requires_partial_coord():
    """nemo_sco on a pure z* coordinate is undefined (r3t/gdept_z0 are
    staircase identities) -> loud error, never a silent fall-through to the
    raw terrain-following gradient."""
    grid = _grid()
    z_star = create_ocean_z_star(n_levels=NZ, H_max=H_MAX)
    wall = jnp.ones((NY, NX), dtype=jnp.float64)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_star, land_mask_override=wall,
        H_bathy_override=jnp.full((NY, NX), H_MAX))
    with pytest.raises(ValueError, match="OceanPartialCellCoordinate"):
        latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_star, _cfg(), dt=300.0, diagnose_momentum=True)
