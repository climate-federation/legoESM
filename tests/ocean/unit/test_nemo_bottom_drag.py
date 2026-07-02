"""Tests for the NEMO zdfdrg non-linear bottom drag (np_non_lin / np_loglayer).

Test surface:
1. **Log-layer Cd formula** — Cd = clip((κ/ln(½h/z0))², cd0, cdmax) matches
   the zdfdrg.F90 expression; BOTH clip ends engage (thin cell → cdmax,
   very thick cell → cd0 floor); the sub-roughness guard stays finite.
2. **Effective r** — r = Cd·√(ū² + v̄² + ke0): full-speed quadrature with
   the background KE (NOT a max-floor), positive, F90-matching.
3. **Dispatch hardening** — unknown ``bottom_drag_scheme`` raises at the
   shared validator AND through the lat-lon tendencies entry.
4. **Lat-lon integration** — with a uniform bottom flow the NEMO branch
   produces the analytic bottom-cell drag tendency at interior faces
   (t-point construction + 2-point face average collapse for uniform u);
   the legacy scheme is bit-identical to the pre-change formula.
5. **AD safety** — gradients through the NEMO branch are finite.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.dynamics.ocean_tendency_common import (
    BOTTOM_DRAG_SCHEMES,
    nemo_drag_r_from_speed_sq,
    nemo_effective_bottom_drag_r,
    nemo_loglayer_cd,
    validate_bottom_drag_scheme,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


K_VK = constants.kappa_von_karman
# ORCA1 namdrg_bot oracle values
CD0, CDMAX, Z0, KE0 = 1.0e-3, 0.1, 3.0e-3, 2.5e-3


class TestLoglayerCd:
    def test_matches_f90_formula_midrange(self):
        for h in [10.0, 50.0, 100.0, 500.0, 4000.0]:
            cd = float(nemo_loglayer_cd(
                jnp.asarray(h), z0=Z0, cd_min=CD0, cd_max=CDMAX,
                von_karman=K_VK))
            ref = (K_VK / np.log(0.5 * h / Z0)) ** 2
            ref = min(max(CD0, ref), CDMAX)
            assert cd == pytest.approx(ref, rel=1e-12), h

    def test_thin_cell_hits_cdmax(self):
        # ½h barely above z0: raw Cd = (κ/ln(½h/z0))² >> cdmax → clip.
        cd = float(nemo_loglayer_cd(
            jnp.asarray(0.0070), z0=Z0, cd_min=CD0, cd_max=CDMAX,
            von_karman=K_VK))
        assert cd == pytest.approx(CDMAX)

    def test_thick_cell_hits_cd0_floor(self):
        # raw Cd < cd0 needs ln(½h/z0) > κ/√cd0 = 12.6 → h > 2·z0·e^12.6.
        h = 2.0 * Z0 * math.exp(K_VK / math.sqrt(CD0)) * 1.5
        cd = float(nemo_loglayer_cd(
            jnp.asarray(h), z0=Z0, cd_min=CD0, cd_max=CDMAX,
            von_karman=K_VK))
        assert cd == pytest.approx(CD0)

    def test_subroughness_thickness_finite_and_capped(self):
        # h/2 <= z0 would blow up ln(); the e·z0 floor keeps it finite and
        # the physically-correct fully-rough cdmax applies.
        for h in [0.0, 1.0e-4, 2.0 * Z0]:
            cd = float(nemo_loglayer_cd(
                jnp.asarray(h), z0=Z0, cd_min=CD0, cd_max=CDMAX,
                von_karman=K_VK))
            assert np.isfinite(cd)
            assert cd == pytest.approx(CDMAX)


class TestEffectiveR:
    def test_quadratic_full_speed_with_ke0(self):
        u, v = 0.3, -0.4
        r = float(nemo_effective_bottom_drag_r(
            jnp.asarray(u), jnp.asarray(v), jnp.asarray(1000.0),
            scheme="nemo_quadratic", cd0=CD0, cd_max=CDMAX, z0=Z0, ke0=KE0,
            von_karman=K_VK))
        assert r == pytest.approx(CD0 * np.sqrt(u * u + v * v + KE0),
                                  rel=1e-12)
        # ke0 acts in quadrature, not as a max floor: at rest the residual
        # speed is √ke0, not 0.
        r0 = float(nemo_effective_bottom_drag_r(
            jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1000.0),
            scheme="nemo_quadratic", cd0=CD0, cd_max=CDMAX, z0=Z0, ke0=KE0,
            von_karman=K_VK))
        assert r0 == pytest.approx(CD0 * np.sqrt(KE0), rel=1e-12)

    def test_loglayer_couples_cd_of_h(self):
        h = 42.0
        r = float(nemo_effective_bottom_drag_r(
            jnp.asarray(0.2), jnp.asarray(0.1), jnp.asarray(h),
            scheme="nemo_loglayer", cd0=CD0, cd_max=CDMAX, z0=Z0, ke0=KE0,
            von_karman=K_VK))
        cd = (K_VK / np.log(0.5 * h / Z0)) ** 2
        assert r == pytest.approx(cd * np.sqrt(0.05 + KE0), rel=1e-12)

    def test_speed_sq_form_matches_component_form(self):
        r1 = nemo_effective_bottom_drag_r(
            jnp.asarray(0.3), jnp.asarray(0.4), jnp.asarray(100.0),
            scheme="nemo_loglayer", cd0=CD0, cd_max=CDMAX, z0=Z0, ke0=KE0,
            von_karman=K_VK)
        r2 = nemo_drag_r_from_speed_sq(
            jnp.asarray(0.25), jnp.asarray(100.0),
            scheme="nemo_loglayer", cd0=CD0, cd_max=CDMAX, z0=Z0, ke0=KE0,
            von_karman=K_VK)
        assert float(r1) == pytest.approx(float(r2), rel=1e-12)

    def test_grad_finite(self):
        def loss(uv):
            r = nemo_effective_bottom_drag_r(
                uv[0], uv[1], jnp.asarray(30.0),
                scheme="nemo_loglayer", cd0=CD0, cd_max=CDMAX, z0=Z0,
                ke0=KE0, von_karman=K_VK)
            return jnp.sum(r)
        g = jax.grad(loss)(jnp.asarray([0.2, -0.1]))
        assert bool(jnp.all(jnp.isfinite(g)))
        # ... including exactly at rest (ke0 keeps √ smooth at |U| = 0).
        g0 = jax.grad(loss)(jnp.asarray([0.0, 0.0]))
        assert bool(jnp.all(jnp.isfinite(g0)))


class TestDispatch:
    def test_validator_raises_on_unknown(self):
        with pytest.raises(ValueError, match="bottom_drag_scheme"):
            validate_bottom_drag_scheme("nemo_logloyer")  # typo

    def test_known_schemes_pass(self):
        for s in BOTTOM_DRAG_SCHEMES:
            assert validate_bottom_drag_scheme(s) == s

    def test_shared_core_rejects_legacy(self):
        with pytest.raises(ValueError, match="legacy"):
            nemo_drag_r_from_speed_sq(
                jnp.asarray(0.1), jnp.asarray(10.0), scheme="legacy",
                cd0=CD0, cd_max=CDMAX, z0=Z0, ke0=KE0, von_karman=K_VK)


# -----------------------------------------------------------------------------
# Lat-lon C-grid integration through the baroclinic tendencies
# -----------------------------------------------------------------------------

def _tendencies_botdrag(scheme, u0=0.25, v0=-0.15, **drag_over):
    """Run the lat-lon PE tendencies with a uniform (u0, v0) flow and
    return the ISOLATED bottom-drag momentum diagnostics
    (``diag.botdrag_u/.botdrag_v``)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0)
    state = state._replace(
        u=state.u.replace(data=jnp.full_like(state.u.data, u0)),
        v=state.v.replace(data=jnp.full_like(state.v.data, v0)),
    )
    flat = dict(
        A_h=0.0, A_v=0.0, K_h=0.0, K_v=0.0,
        bottom_drag_scheme=scheme,
    )
    flat.update(drag_over)
    config = LatLonCGridOceanConfig.from_flat(**flat)
    _, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z, config, diagnose_momentum=True)
    return state, diag.botdrag_u.data, diag.botdrag_v.data


def _bottom_dz(state, z):
    """Actual bottom-cell thickness dz_ref[-1]·J (create_ocean_z_star builds
    STRETCHED levels and the rest state's default H_bathy=5500 puts the
    z-star Jacobian at H_bathy/H_max)."""
    J = float(np.asarray(state.H_bathy.data).ravel()[0]) / float(
        -np.asarray(z.z_half_ref)[-1])
    return float(np.asarray(z.dz_ref)[-1]) * J


@pytest.mark.parametrize("scheme", ["nemo_quadratic", "nemo_loglayer"])
def test_latlon_bottom_cell_drag_matches_analytic(scheme):
    u0, v0 = 0.25, -0.15
    state, du, dv = _tendencies_botdrag(scheme, u0, v0)
    from legoesm.ocean.vertical import create_ocean_z_star
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    dz_bot = _bottom_dz(state, z)
    if scheme == "nemo_loglayer":
        cd = (K_VK / np.log(0.5 * dz_bot / Z0)) ** 2
        cd = min(max(CD0, cd), CDMAX)
    else:
        cd = CD0
    r = cd * np.sqrt(u0 * u0 + v0 * v0 + KE0)
    expect_u = -r * u0 / dz_bot
    expect_v = -r * v0 / dz_bot
    # Uniform flow → the t-point speed and the face averages are uniform;
    # every INTERIOR face carries the same analytic bottom-cell tendency.
    # State storage is float32 ⇒ f32-appropriate tolerances.
    du_bot = np.asarray(du[..., -1])
    dv_bot = np.asarray(dv[..., -1])
    np.testing.assert_allclose(du_bot[1:-1, 1:-1], expect_u, rtol=1e-5)
    np.testing.assert_allclose(dv_bot[2:-2, 1:-1], expect_v, rtol=1e-5)
    # Nothing above the bottom level.
    assert float(jnp.max(jnp.abs(du[..., :-1]))) == 0.0
    # Drag opposes the flow (sign convention).
    assert np.all(du_bot[1:-1, 1:-1] < 0.0)   # u0 > 0
    assert np.all(dv_bot[2:-2, 1:-1] > 0.0)   # v0 < 0


def test_latlon_legacy_scheme_bit_identical_formula():
    # legacy + u_bg > 0 keeps the historical per-component MOM6 form.
    u0, v0 = 0.2, -0.3
    r_lin, u_bg = 2.5e-4, 0.1
    state, du, dv = _tendencies_botdrag(
        "legacy", u0, v0, bottom_drag_r=r_lin, bottom_drag_bg_velocity=u_bg)
    from legoesm.ocean.vertical import create_ocean_z_star
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    dz_bot = _bottom_dz(state, z)
    cd_eq = r_lin / u_bg
    expect_u = -(cd_eq * np.sqrt(u0 * u0 + u_bg * u_bg)) * u0 / dz_bot
    expect_v = -(cd_eq * np.sqrt(v0 * v0 + u_bg * u_bg)) * v0 / dz_bot
    np.testing.assert_allclose(
        np.asarray(du[..., -1])[1:-1, 1:-1], expect_u, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(dv[..., -1])[2:-2, 1:-1], expect_v, rtol=1e-5)


def test_latlon_unknown_scheme_raises():
    with pytest.raises(ValueError, match="bottom_drag_scheme"):
        _tendencies_botdrag("nemo_quadratc")  # typo


def test_config_defaults_are_legacy_everywhere():
    from legoesm.ocean.state import (
        DynBottomDragConfig, LatLonCGridOceanConfig, OceanConfig,
    )
    from legoesm.ocean.mpas_config import MPASOceanConfig
    assert DynBottomDragConfig().bottom_drag_scheme == "legacy"
    assert (LatLonCGridOceanConfig.from_flat()
            .bottom_drag.bottom_drag_scheme == "legacy")
    assert OceanConfig().bottom_drag_scheme == "legacy"
    assert MPASOceanConfig().bottom_drag_scheme == "legacy"
    # ORCA1 namdrg_bot parameter defaults
    d = DynBottomDragConfig()
    assert d.bottom_drag_cd0 == pytest.approx(1.0e-3)
    assert d.bottom_drag_cdmax == pytest.approx(0.1)
    assert d.bottom_drag_z0 == pytest.approx(3.0e-3)
    assert d.bottom_drag_ke0 == pytest.approx(2.5e-3)
