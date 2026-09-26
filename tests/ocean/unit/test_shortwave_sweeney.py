"""Direct tests for the FESOM2 / Sweeney-2005 shortwave penetration scheme."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.ocean.eos import rho_0, c_sw  # noqa: E402
from legoesm.ocean.physics.shortwave_penetration import (  # noqa: E402
    JERLOV_PENETRATING_FRACTION,
    RGB_PENETRATING_FRACTION,
    SWEENEY_VISIBLE_FRACTION,
    ShortwavePenetrationConfig,
    apply_shortwave_penetration,
    penetrating_fraction,
    shortwave_penetration_rgb_tendency,
    shortwave_penetration_sweeney_tendency,
)

NLEV = 20
DZ = 10.0
SW_COLUMNS = (150.0, 200.0)
CHL_COLUMNS = (0.1, 2.0)


def _sweeney_absorbed_fraction(chl, dz, swap_bands=False):
    """Independent numpy evaluation of FESOM2's oce_shortwave_pene profile."""
    dz = np.asarray(dz, dtype=np.float64)
    z = np.concatenate(([0.0], -np.cumsum(dz)))
    c = np.log10(max(float(chl), 0.02))
    v1 = 0.321 + (0.008 * c + 0.132 * c**2 + 0.038 * c**3 - 0.017 * c**4 - 0.007 * c**5)
    v2 = 1.0 - v1
    if swap_bands:
        v1, v2 = v2, v1
    sc1 = 1.54 - 0.197 * c + 0.166 * c**2 - 0.252 * c**3 - 0.055 * c**4 + 0.042 * c**5
    sc2 = 7.925 - 6.644 * c + 3.662 * c**2 - 1.815 * c**3 - 0.218 * c**4 + 0.502 * c**5
    intensity = v1 * np.exp(z / sc1) + v2 * np.exp(z / sc2)
    fraction = intensity[:-1] - intensity[1:]
    fraction[-1] = intensity[-2]  # deepest wet cell absorbs the remainder
    return fraction


def _dz_column(nlev=NLEV):
    return jnp.full((nlev,), DZ, dtype=jnp.float64)


def _wet_column(nlev=NLEV):
    return jnp.ones((nlev,), dtype=jnp.float64)


def test_sweeney_profile_matches_reference_and_rejects_swapped_bands():
    sw_visible = jnp.array(SW_COLUMNS, dtype=jnp.float64)
    chl = jnp.array(CHL_COLUMNS, dtype=jnp.float64)
    dz = jnp.full((len(CHL_COLUMNS), NLEV), DZ, dtype=jnp.float64)
    wet = jnp.ones((len(CHL_COLUMNS), NLEV), dtype=jnp.float64)
    d_tendency = np.asarray(
        shortwave_penetration_sweeney_tendency(sw_visible, chl, dz, wet, rho_0, c_sw))
    assert d_tendency.shape == (len(CHL_COLUMNS), NLEV)
    heating = rho_0 * c_sw * DZ * d_tendency
    for column, (sw_col, chl_col) in enumerate(zip(SW_COLUMNS, CHL_COLUMNS)):
        dz_column = np.full(NLEV, DZ)
        np.testing.assert_allclose(
            heating[column], sw_col * _sweeney_absorbed_fraction(chl_col, dz_column),
            rtol=0.0, atol=1e-12)
        swapped = _sweeney_absorbed_fraction(chl_col, dz_column, swap_bands=True)
        assert float(np.max(np.abs(heating[column] - sw_col * swapped))) > 1e-3 * sw_col


def test_sweeney_conserves_column_integral_and_zero_on_dry_cells():
    sw_visible = 180.0
    chl = 0.5
    d_full = np.asarray(shortwave_penetration_sweeney_tendency(
        sw_visible, chl, _dz_column(), _wet_column(), rho_0, c_sw))
    assert float(np.sum(rho_0 * c_sw * DZ * d_full)) == pytest.approx(sw_visible, rel=1e-12)
    dz_truncated = _dz_column().at[5:].set(0.0)
    wet_truncated = _wet_column().at[5:].set(0.0)
    d_truncated = np.asarray(shortwave_penetration_sweeney_tendency(
        sw_visible, chl, dz_truncated, wet_truncated, rho_0, c_sw))
    assert float(np.sum(rho_0 * c_sw * DZ * d_truncated[:5])) == pytest.approx(sw_visible, rel=1e-12)
    assert np.all(d_truncated[5:] == 0.0)
    np.testing.assert_allclose(
        rho_0 * c_sw * DZ * d_truncated[:5],
        sw_visible * _sweeney_absorbed_fraction(chl, np.full(5, DZ)), rtol=0.0, atol=1e-12)


def test_apply_shortwave_penetration_sweeney_equals_kernel():
    config = ShortwavePenetrationConfig(scheme="sweeney_2band")
    sw_down = jnp.array(SW_COLUMNS, dtype=jnp.float64)
    chl = jnp.array(CHL_COLUMNS, dtype=jnp.float64)
    dz = jnp.full((len(CHL_COLUMNS), NLEV), DZ, dtype=jnp.float64)
    wet = jnp.ones((len(CHL_COLUMNS), NLEV), dtype=jnp.float64)
    applied = np.asarray(apply_shortwave_penetration(
        config, sw_down, chl=chl, dz_live=dz, wet_cell=wet, rho_0=rho_0, c_sw=c_sw))
    expected = np.asarray(shortwave_penetration_sweeney_tendency(sw_down, chl, dz, wet, rho_0, c_sw))
    np.testing.assert_allclose(applied, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("missing_kwarg", ["chl", "dz_live"])
def test_apply_shortwave_penetration_requires_chl_and_dz_live(missing_kwarg):
    config = ShortwavePenetrationConfig(scheme="sweeney_2band")
    kwargs = {"chl": 0.5, "dz_live": _dz_column(), "wet_cell": _wet_column(),
              "rho_0": rho_0, "c_sw": c_sw}
    del kwargs[missing_kwarg]
    with pytest.raises(ValueError):
        apply_shortwave_penetration(config, 180.0, **kwargs)


@pytest.mark.parametrize(("scheme", "expected"), [
    ("jerlov_2band", 0.94), ("rgb_chl", 1.0), ("sweeney_2band", 0.54)])
def test_penetrating_fraction_per_scheme(scheme, expected):
    assert penetrating_fraction(ShortwavePenetrationConfig(scheme=scheme)) == pytest.approx(expected)


def test_penetrating_fraction_constants_and_unknown_scheme():
    assert (JERLOV_PENETRATING_FRACTION, RGB_PENETRATING_FRACTION, SWEENEY_VISIBLE_FRACTION) == (0.94, 1.0, 0.54)
    with pytest.raises(ValueError):
        penetrating_fraction(ShortwavePenetrationConfig(scheme="bogus"))
    with pytest.raises(ValueError):
        apply_shortwave_penetration(ShortwavePenetrationConfig(scheme="bogus"), 1.0)


def test_rgb_kernel_conserves_column_integral():
    sw_down = jnp.asarray(150.0)
    chl = jnp.asarray(0.1)
    config = ShortwavePenetrationConfig(scheme="rgb_chl")
    d_tendency = np.asarray(shortwave_penetration_rgb_tendency(
        sw_down, chl, _dz_column(), _wet_column(), config, rho_0, c_sw))
    assert d_tendency.shape == (NLEV,)
    assert np.all(np.isfinite(d_tendency))
    assert float(np.sum(rho_0 * c_sw * DZ * d_tendency)) == pytest.approx(150.0, rel=1e-12)
