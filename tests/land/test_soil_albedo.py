"""Unit tests for CLM soil-colour background albedo (Table 3.3 / eq. 3.52)."""

import jax.numpy as jnp
import numpy as np

from legoesm.land.soil_albedo import (
    soil_albedo,
    soil_albedo_broadband,
    SOIL_COLOR_ALBEDO,
    N_SOIL_COLOR,
)


def test_dry_soil_reaches_dry_albedo():
    # theta_top = 0 -> Delta = 0.11 -> alpha = min(sat+0.11, dry) = dry (by table design)
    for cls in (1, 10, 20):
        vis, nir = soil_albedo(jnp.array(cls), jnp.array(0.0))
        dv, dn = SOIL_COLOR_ALBEDO[cls - 1][0], SOIL_COLOR_ALBEDO[cls - 1][1]
        assert np.isclose(float(vis), dv, atol=1e-6)
        assert np.isclose(float(nir), dn, atol=1e-6)


def test_wet_soil_reaches_saturated_albedo():
    # theta_top >= 0.11/0.40 = 0.275 -> Delta = 0 -> alpha = saturated
    vis, nir = soil_albedo(jnp.array(1), jnp.array(0.30))
    assert np.isclose(float(vis), 0.25, atol=1e-6)   # class 1 sat_vis
    assert np.isclose(float(nir), 0.50, atol=1e-6)   # class 1 sat_nir


def test_intermediate_moisture_between_sat_and_dry():
    vis, nir = soil_albedo(jnp.array(5), jnp.array(0.15))   # Delta = 0.11-0.06 = 0.05
    sv = SOIL_COLOR_ALBEDO[4][2]
    assert np.isclose(float(vis), min(sv + 0.05, SOIL_COLOR_ALBEDO[4][0]), atol=1e-6)
    assert SOIL_COLOR_ALBEDO[4][2] < float(vis) < SOIL_COLOR_ALBEDO[4][0]


def test_darker_classes_have_lower_albedo():
    cls = jnp.arange(1, N_SOIL_COLOR + 1)
    vis, nir = soil_albedo(cls, jnp.zeros(N_SOIL_COLOR))
    assert np.all(np.diff(np.asarray(vis)) <= 1e-9)   # monotonically non-increasing
    assert np.all(np.diff(np.asarray(nir)) <= 1e-9)


def test_class_clipping_and_broadband():
    # out-of-range classes clip into [1,20]; broadband is the VIS/NIR blend
    vis, nir = soil_albedo(jnp.array(0), jnp.array(0.0))
    assert np.isclose(float(vis), SOIL_COLOR_ALBEDO[0][0])     # class 0 -> class 1
    bb = soil_albedo_broadband(jnp.array(10), jnp.array(0.2))
    v, n = soil_albedo(jnp.array(10), jnp.array(0.2))
    assert np.isclose(float(bb), 0.5 * float(v) + 0.5 * float(n))


def test_differentiable_in_moisture():
    import jax
    g = jax.grad(lambda th: soil_albedo_broadband(jnp.array(3), th))(jnp.array(0.1))
    assert np.isfinite(float(g)) and float(g) < 0.0   # wetter -> darker


# ---------------------------------------------------------------------------
# Integration into surface_albedo.land_albedo (soil + veg + snow blend)
# ---------------------------------------------------------------------------
from legoesm.surface_albedo import land_albedo, land_vegetation_albedo, LandAlbedoConfig


def test_land_albedo_backward_compatible():
    # No soil/LAI -> unchanged latitude-band vegetation albedo (snow-free).
    lat = jnp.array(0.2)
    a = land_albedo(lat, jnp.array(0.0), jnp.array(0.0))
    assert np.isclose(float(a), float(land_vegetation_albedo(lat)))


def test_land_albedo_soil_veg_blend_by_lai():
    lat = jnp.array(0.2); cfg = LandAlbedoConfig()
    veg = float(land_vegetation_albedo(lat, cfg))
    soil = 0.30
    # LAI=0 -> all soil; large LAI -> all veg
    a_bare = land_albedo(lat, jnp.array(0.0), jnp.array(0.0), cfg,
                         soil_albedo=jnp.array(soil), lai=jnp.array(0.0))
    a_dense = land_albedo(lat, jnp.array(0.0), jnp.array(0.0), cfg,
                          soil_albedo=jnp.array(soil), lai=jnp.array(20.0))
    assert np.isclose(float(a_bare), soil, atol=1e-3)
    assert np.isclose(float(a_dense), veg, atol=1e-3)
    # intermediate LAI lies between
    a_mid = land_albedo(lat, jnp.array(0.0), jnp.array(0.0), cfg,
                        soil_albedo=jnp.array(soil), lai=jnp.array(1.0))
    assert min(soil, veg) <= float(a_mid) <= max(soil, veg)


def test_land_albedo_snow_overrides():
    # Full snow cover -> snow albedo regardless of soil/veg.
    cfg = LandAlbedoConfig()
    a = land_albedo(jnp.array(0.2), jnp.array(1e6), jnp.array(0.0), cfg,
                    soil_albedo=jnp.array(0.1), lai=jnp.array(0.0))
    assert np.isclose(float(a), cfg.alpha_snow_max, atol=1e-3)
