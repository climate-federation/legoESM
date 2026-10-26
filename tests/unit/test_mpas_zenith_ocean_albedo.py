"""The production lane's open-ocean albedo must follow the sun angle.

A flat 0.06 is roughly right under an overhead sun and badly wrong where the
sun never rises far, which is where this model carries a -20.3 W/m^2 clear-sky
shortwave deficit.  These tests pin the SHAPE of the dependence (bright at
grazing incidence, dark overhead, seasonally reversed between hemispheres) so
the parameterisation cannot be silently replaced by something monotone in
latitude or, worse, by the constant it exists to replace.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.driver.model_driver import _mpas_zenith_ocean_albedo

MARCH_EQUINOX_DAY = 79.0        # day-of-year 80
JUNE_SOLSTICE_DAY = 172.0       # day-of-year 173
FLAT_MODEL_VALUE = 0.06


def _alb(lat_deg, day):
    return np.asarray(_mpas_zenith_ocean_albedo(
        jnp.deg2rad(jnp.asarray(np.atleast_1d(lat_deg), dtype=jnp.float64)), day))


def test_rises_monotonically_toward_the_pole_at_equinox():
    lats = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    a = _alb(lats, MARCH_EQUINOX_DAY)
    assert np.all(np.diff(a) > 0.0), a
    assert a[0] < FLAT_MODEL_VALUE          # overhead sun: darker than the constant
    assert a[-1] > 3.0 * FLAT_MODEL_VALUE   # grazing sun: several times brighter


def test_the_flat_value_is_only_right_in_the_subtropics():
    """The constant 0.06 is not wrong everywhere -- it is a mid-latitude value,
    which is exactly why a global constant hides a large polar error."""
    a30 = float(_alb(30.0, MARCH_EQUINOX_DAY)[0])
    assert a30 == pytest.approx(FLAT_MODEL_VALUE, abs=0.01)


def test_season_reverses_between_hemispheres():
    """In June the Arctic ocean darkens (high sun) while the Southern Ocean
    brightens (winter).  A latitude-only fit would get this backwards."""
    n_mar, n_jun = float(_alb(70.0, MARCH_EQUINOX_DAY)[0]), float(_alb(70.0, JUNE_SOLSTICE_DAY)[0])
    s_mar, s_jun = float(_alb(-70.0, MARCH_EQUINOX_DAY)[0]), float(_alb(-70.0, JUNE_SOLSTICE_DAY)[0])
    assert n_jun < n_mar
    assert s_jun > s_mar


def test_stays_in_physical_range_including_polar_night():
    for day in (MARCH_EQUINOX_DAY, JUNE_SOLSTICE_DAY):
        a = _alb(np.linspace(-89.0, 89.0, 60), day)
        assert np.all(np.isfinite(a))
        assert np.all(a >= 0.03) and np.all(a <= 0.40)


def test_is_differentiable():
    """The model is differentiable end to end; a surface boundary condition
    that breaks the gradient would be a silent regression."""
    g = jax.grad(lambda x: jnp.sum(_mpas_zenith_ocean_albedo(x, MARCH_EQUINOX_DAY)))
    d = g(jnp.deg2rad(jnp.asarray([10.0, 50.0, 80.0])))
    assert np.all(np.isfinite(np.asarray(d)))
    assert np.any(np.abs(np.asarray(d)) > 0.0)


def test_the_launcher_accepts_the_flag_on_mpas_and_still_refuses_spectral():
    """The guard is narrowed, not deleted: MPAS now implements the curve, the
    spectral standalone path still has nowhere to put it and a silently ignored
    flag is exactly what the guard exists to prevent."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts" / "run"))
    from run_amip import build_arg_parser, _postprocess_args, build_config_from_args

    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--grid-type", "voronoi",
        "--discretization", "mpas", "--dynamic-albedo"]), parser))
    assert cfg.dynamic_albedo is True

    with pytest.raises(SystemExit):
        _postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--discretization", "spectral",
            "--dynamic-albedo"]), parser)
