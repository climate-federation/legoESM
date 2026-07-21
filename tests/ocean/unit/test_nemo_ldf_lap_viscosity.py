"""Node 14 — NEMO ``dyn_ldf_lev_lap`` faithful lateral viscosity.

Truth-tier tests for the embedded div-curl viscosity operator
(``nemo_ldf_lap_viscosity_cgrid``) and its coefficient
(``nemo_lateral_viscosity_coefficients``):

* the coefficient is exactly NEMO ``ldf_c2d`` ``ahmt/ahmf = ½·rn_Uv·MAX(e1,e2)``
  at T/F points, and on a UNIFORM-Δφ grid it does NOT shrink with cos(φ) at high
  latitude (unlike ``A_h·cos φ``);
* with a CONSTANT coefficient the embedded operator reduces EXACTLY to
  ``A_h · vector_laplacian_cgrid`` (so it is a strict generalisation, backward-
  compatible in the constant-coefficient limit).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    nemo_lateral_viscosity_coefficients,
    nemo_ldf_lap_viscosity_cgrid,
    vector_laplacian_cgrid,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_horizontal_viscosity
from legoesm.ocean.state import LatLonCGridOceanConfig


def _geo(n_lat=40, n_lon=80):
    return ensure_geometry(create_latlon_grid(n_lat, n_lon))


def test_coefficient_is_half_UM_max_e1_e2():
    """ahmt/ahmf = ½·rn_Uv·MAX(e1,e2) at T- and F-points (NEMO ldf_c2d L138-139)."""
    geo = _geo()
    half_UM = 0.135  # ½·rn_Uv with rn_Uv = 0.27
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geo, half_UM)

    e1t = np.asarray(geo.dx_u[:, 0]); e2t = np.asarray(geo.dy_u[:, 0])
    e1f = np.asarray(geo.dx_v[:, 0]); e2f = np.asarray(geo.dy_v[:, 0])
    np.testing.assert_allclose(
        np.asarray(ahmt), half_UM * np.maximum(e1t, e2t), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ahmf), half_UM * np.maximum(e1f, e2f), rtol=1e-12)
    assert ahmt.shape == (40,)
    assert ahmf.shape == (41,)


def test_high_lat_does_not_shrink_with_cos_on_uniform_grid():
    """On a uniform-Δφ grid e2=R·Δφ is constant, so MAX(e1,e2)=e2 at high lat and
    ahmt does NOT collapse with cos(φ) — the node-14 point vs ``A_h·cos φ``."""
    geo = _geo(n_lat=60, n_lon=120)   # spans ~[-87,87]°
    half_UM = 0.135
    ahmt, _ = nemo_lateral_viscosity_coefficients(geo, half_UM)
    ahmt = np.asarray(ahmt)
    lat = np.degrees(np.asarray(geo.lat))

    j_eq = int(np.argmin(np.abs(lat)))
    j_hi = int(np.argmin(np.abs(lat - 80.0)))
    cos_hi = np.cos(np.deg2rad(lat[j_hi]))

    # A_h·cos φ would drop to cos(80°)≈0.17× of the equator value; the NEMO
    # coefficient stays O(1) because e2=R·Δφ (constant) wins the MAX.
    ratio = ahmt[j_hi] / ahmt[j_eq]
    assert ratio > 0.5, f"ahmt collapsed at high lat: ratio={ratio}"
    assert ratio > 3.0 * cos_hi, (
        f"ahmt should stay well above the cos φ form: ratio={ratio}, cos={cos_hi}")


def test_reduces_to_A_h_vector_laplacian_for_constant_coeff():
    """Constant ahmt=ahmf=A_h ⇒ embedded operator == A_h·vector_laplacian_cgrid.

    Proves the embedded form is a strict generalisation (backward-compatible in
    the constant-coefficient limit) and that the div/curl wiring / signs match
    the verified vector Laplacian to machine precision."""
    geo = _geo()
    rng = np.random.default_rng(0)
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    nlev = 3
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, nlev)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)))
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1))
    v_mask = jnp.ones((n_lat + 1, n_lon))

    A_h = 1.234e4
    ahmt = jnp.full((n_lat,), A_h)
    ahmf = jnp.full((n_lat + 1,), A_h)

    nu, nv = nemo_ldf_lap_viscosity_cgrid(
        u, v, geo, ahmt, ahmf, mask=mask, u_mask=u_mask, v_mask=v_mask)
    vu, vv = vector_laplacian_cgrid(
        u, v, geo, mask=mask, u_mask=u_mask, v_mask=v_mask)

    np.testing.assert_allclose(np.asarray(nu), A_h * np.asarray(vu),
                               rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(np.asarray(nv), A_h * np.asarray(vv),
                               rtol=1e-11, atol=1e-9)


def test_constant_coeff_equals_vector_laplacian_with_land_mask():
    """Constant-coefficient equivalence also holds through the land / face /
    vertex masking paths (a wet-basin mask), so the embedded operator inherits
    the verified free-slip wall handling of ``vector_laplacian_cgrid``."""
    geo = _geo()
    rng = np.random.default_rng(1)
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))
    m = np.ones((n_lat, n_lon))
    m[: n_lat // 4, :] = 0.0        # a southern land band
    m[:, : n_lon // 5] = 0.0        # a western wall
    mask = jnp.asarray(m)
    u_mask = jnp.asarray(np.minimum(m, np.roll(m, -1, axis=1)))
    u_mask = jnp.concatenate([u_mask, u_mask[:, :1]], axis=1)
    v_mask = jnp.asarray(np.pad(np.minimum(m[:-1], m[1:]), ((1, 1), (0, 0))))

    A_h = 5.0e3
    ahmt = jnp.full((n_lat,), A_h)
    ahmf = jnp.full((n_lat + 1,), A_h)
    nu, nv = nemo_ldf_lap_viscosity_cgrid(
        u, v, geo, ahmt, ahmf, mask=mask, u_mask=u_mask, v_mask=v_mask)
    vu, vv = vector_laplacian_cgrid(
        u, v, geo, mask=mask, u_mask=u_mask, v_mask=v_mask)
    np.testing.assert_allclose(np.asarray(nu), A_h * np.asarray(vu),
                               rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(np.asarray(nv), A_h * np.asarray(vv),
                               rtol=1e-10, atol=1e-9)


def test_variable_coefficient_is_dissipative():
    """With the REAL latitude-varying ahmt/ahmf (the placement cross-terms are
    NONZERO, unlike the constant-coeff reduction test) the operator REMOVES kinetic
    energy on a periodic domain: Σ(u·visc_u·A_u + v·visc_v·A_v) < 0.  A sign flip or
    gross mis-placement of the embedded coefficient would make this positive."""
    # Anisotropic grid (Δλ=9°≫Δφ=1.5°): e1=R·Δλ·cosφ wins the MAX at low lat and
    # e2=R·Δφ at high lat, so ahmt varies ~6× → the placement cross-terms are large.
    geo = _geo(n_lat=120, n_lon=40)
    rng = np.random.default_rng(3)
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geo, 0.135)
    assert float(ahmt.max() / ahmt.min()) > 2.0
    nu, nv = nemo_ldf_lap_viscosity_cgrid(u, v, geo, ahmt, ahmf)   # no walls (periodic)
    A_u = np.asarray(geo.dx_u * geo.dy_u)       # u-face control area
    A_v = np.asarray(geo.dx_v * geo.dy_v)       # v-face control area
    prod = float(np.sum(np.asarray(u) * np.asarray(nu) * A_u)
                 + np.sum(np.asarray(v) * np.asarray(nv) * A_v))
    assert prod < 0.0, f"variable-coeff viscosity is not dissipative: KE production={prod}"


# --- Dispatch / reject / recipe wiring (CLAUDE.md dispatch-hardening: same-PR) ---

def _visc_config(operator, **lv):
    return LatLonCGridOceanConfig.from_flat(
        lateral_viscosity_operator=operator, A_h=1.5e4, A_h_lat_scaling=True, **lv)


def _call_visc(geo, config):
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))
    # 3-D fields (the RHS runs on (…, nlev)); nlev=1 keeps it cheap.
    u3 = u[..., None]; v3 = v[..., None]
    du3 = jnp.zeros_like(u3); dv3 = jnp.zeros_like(v3)
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1)); v_mask = jnp.ones((n_lat + 1, n_lon))
    out = _bc_horizontal_viscosity(
        du3, dv3, u3, v3, geo, mask, u_mask, v_mask, config,
        None, None, 1.0)
    return out[0]   # du_dt


def test_dispatch_routes_to_nemo_operator():
    """Selecting ``nemo_div_curl`` in the config ROUTES the RHS to the embedded
    operator (the du_dt tendency equals the direct operator call)."""
    geo = _geo()
    cfg = _visc_config("nemo_div_curl")
    du = np.asarray(_call_visc(geo, cfg))[..., 0]
    half_UM = cfg.lateral_viscosity.A_h / (geo.radius * geo.dlon)
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geo, half_UM)
    n_lat, n_lon = geo.lat.shape[0], geo.lon.shape[0]
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1)))
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))
    mask = jnp.ones((n_lat, n_lon))
    u_mask = jnp.ones((n_lat, n_lon + 1)); v_mask = jnp.ones((n_lat + 1, n_lon))
    ref, _ = nemo_ldf_lap_viscosity_cgrid(
        u, v, geo, ahmt, ahmf, mask=mask, u_mask=u_mask, v_mask=v_mask)
    assert float(np.max(np.abs(du))) > 0.0
    np.testing.assert_allclose(du, np.asarray(ref), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("bad", [
    {"A_h_eq_boost": 2.0}, {"A_h_cap_boost": 2.0}, {"A_h_floor": 1000.0},
    {"B_h": 1.0e10},
])
def test_nemo_operator_rejects_incompatible_options(bad):
    """The embedded operator RAISES (never silently ignores) on eq/cap/floor boosts
    and B_h — they would double-scale or be silently dropped."""
    geo = _geo()
    cfg = _visc_config("nemo_div_curl", **bad)
    with pytest.raises(ValueError):
        _call_visc(geo, cfg)


def test_unknown_operator_raises():
    geo = _geo()
    cfg = _visc_config("typo_operator")
    with pytest.raises(ValueError):
        _call_visc(geo, cfg)


def test_dino_recipe_wires_nemo_div_curl_and_propagates():
    """The DINO nemo_dino_kamm / _mlf cards select nemo_div_curl, and the builder
    propagates it into the model config with boosts/floor/B_h at their defaults."""
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, dino_config_for_recipe, dino_lat_lon_model_config,
    )
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        assert DINO_RECIPES[recipe]["lateral_viscosity_operator"] == "nemo_div_curl"
    geo = _geo(n_lat=20, n_lon=40)
    cfg = dino_config_for_recipe("nemo_dino_kamm")
    mc, _ = dino_lat_lon_model_config(geo, cfg)
    assert mc.lateral_viscosity_operator == "nemo_div_curl"
    assert mc.lateral_viscosity.A_h_eq_boost == 1.0
    assert mc.lateral_viscosity.A_h_floor == 0.0
    assert mc.lateral_viscosity.B_h == 0.0
