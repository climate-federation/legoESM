"""Oracle match: orbital geometry and open-ocean albedo (CESM2.1).

Oracle: cime5.6.47 (d8e9b894d7eb96fe96f8f8a6943b421147f94606), RUN as compiled
Fortran -- share/util/shr_orb_mod.F90 (shr_orb_params, shr_orb_decl,
shr_orb_cosz; unmodified, vernal equinox ``ve = 80.5``) and the coupler ocean
albedo drivers/mct/main/seq_flux_mct.F90 L881-894 (direct albedo at the
instantaneous cosine, diffuse ``seq_flux_mct_albdif = 0.06``).  Fixture
``oracle_fixtures/rad_boundary.npz`` (3024 points: 7 columns x 9 days x 48
half-hours, calendar day from the production ``day_to_calendar`` map) generated
by scripts/validate/oracle/rad_boundary/{build.sh,make_fixture.py}.

Ours = the production functions on the MPAS ``_run_column`` lane:
``solar_declination`` / ``earth_sun_distance_factor`` (orbit on),
``integration._compute_insolation`` (diurnal cos zenith), and
``integration.cesm_ocean_albedo_direct``, which ``_call_radiation_backend``
applies to the open-ocean fraction of the direct beam at each radiation call.

Scope: pointwise relative error on quantities that do not cross zero.  The
cosine itself crosses zero at the terminator, where the oracle's own round-off
(cos of a ~2*pi*calday argument, ~1e-13 absolute) has no relative meaning; it is
checked here through the direct albedo it produces (pointwise, both beams'
day/night split included).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation import integration
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, RadiationConfig, RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.solar import (
    OrbitalParameters, earth_sun_distance_factor, solar_declination,
)

_FIX = Path(__file__).parent / "oracle_fixtures" / "rad_boundary.npz"
_TOL = 1e-10  # user-approved oracle-match tolerance (max relative error, fp64)


@pytest.fixture(scope="module")
def case():
    z = np.load(_FIX)
    meta = json.loads(str(z["meta"]))
    orbit = OrbitalParameters(**meta["orbit"])
    s0 = constants.S_0
    rcfg = RadiationConfig(
        scheme="rrtmgp", gray=GrayRadiationConfig(S_0=s0),
        rrtmgp=RRTMGPConfig(S_0=s0), diurnal_cycle=True, orbit=orbit)
    return z, orbit, rcfg


def _rel(ours, ref):
    ours, ref = np.asarray(ours), np.asarray(ref)
    assert ours.shape == ref.shape and np.all(ref != 0.0)
    return float(np.max(np.abs(ours - ref) / np.abs(ref)))


def _cos_zenith(z, rcfg):
    def one(lat, lon, doy, sod):
        return integration._compute_insolation(
            lat[None], rcfg, lon=lon[None], day_of_year=doy, seconds_of_day=sod)[1][0]
    return jax.vmap(one)(*(jnp.asarray(z[k]) for k in
                           ("lat", "lon", "day_of_year", "seconds_of_day")))


def test_declination_and_distance_factor_match_shr_orb(case):
    z, orbit, _ = case
    doy = jnp.asarray(z["day_of_year"])
    dec = jax.vmap(lambda d: solar_declination(d, orbit=orbit))(doy)
    eccf = jax.vmap(lambda d: earth_sun_distance_factor(d, orbit))(doy)
    assert _rel(dec, z["delta"]) <= _TOL
    assert _rel(eccf, z["eccf"]) <= _TOL


def test_ocean_albedo_matches_cesm_coupler(case):
    z, _, rcfg = case
    # the formula at the oracle's own cosine (dark side included)
    a_at_oracle_cos = integration.cesm_ocean_albedo_direct(jnp.asarray(z["cosz"]))
    assert _rel(a_at_oracle_cos, z["albedo_dir"]) <= _TOL
    # end to end: at the cosine our production insolation path computes
    a_ours = integration.cesm_ocean_albedo_direct(_cos_zenith(z, rcfg))
    assert _rel(a_ours, z["albedo_dir"]) <= _TOL
    day = z["cosz"] > 0.0
    assert day.sum() > 1000 and (~day).sum() > 1000
    np.testing.assert_array_equal(
        z["albedo_dif"][day], integration.OCEAN_ALBEDO_DIFFUSE_CESM)


def test_backend_direct_albedo_swaps_only_the_open_ocean_term():
    """``_call_radiation_backend`` hands the solver the diffuse blend unchanged
    and a direct albedo whose open-ocean part is the CESM curve at the call's
    cosine (the blend is linear in the open-ocean fraction)."""
    seen = {}

    class _Stop(Exception):
        pass

    class _Solver:
        def solve_columns(self, **kw):
            seen.update(kw)
            raise _Stop

    cos = jnp.asarray([0.9, 0.3, 0.004, 0.0])
    alb_dif = jnp.asarray([0.06, 0.40, 0.25, 0.06])
    frac = jnp.asarray([1.0, 0.0, 0.5, 1.0])
    T = jnp.full((4, 3), 250.0)
    p_half = jnp.broadcast_to(jnp.asarray([1.0e4, 4.0e4, 7.0e4, 1.0e5]), (4, 4))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    rcfg = RadiationConfig(scheme="rrtmgp", diurnal_cycle=True)
    with pytest.raises(_Stop):
        integration._call_radiation_backend(
            rcfg, T, p_full, p_half, jnp.full((4,), 290.0), jnp.zeros(4),
            jnp.zeros((4, 3)), constants.S_0 * cos, cos_sza=cos,
            sfc_albedo_override=alb_dif, open_ocean_frac=frac,
            rrtmgp_solver=_Solver())
    np.testing.assert_array_equal(np.asarray(seen["sfc_albedo"]), np.asarray(alb_dif))
    want = alb_dif + frac * (integration.cesm_ocean_albedo_direct(cos)
                             - integration.OCEAN_ALBEDO_DIFFUSE_CESM)
    np.testing.assert_array_equal(np.asarray(seen["sfc_albedo_direct"]),
                                  np.asarray(want))
    assert float(seen["sfc_albedo_direct"][1]) == 0.40   # no ocean: unchanged
    with pytest.raises(ValueError, match="rrtmgp solver only"):
        integration._call_radiation_backend(
            RadiationConfig(scheme="gray", diurnal_cycle=True), T, p_full,
            p_half, jnp.full((4,), 290.0), jnp.zeros(4), jnp.zeros((4, 3)),
            constants.S_0 * cos, cos_sza=cos, sfc_albedo_override=alb_dif,
            open_ocean_frac=frac)


def test_gradients_finite_on_fixture_columns(case):
    z, _, rcfg = case
    lat, lon = jnp.asarray(z["lat"]), jnp.asarray(z["lon"])
    doy, sod = jnp.asarray(z["day_of_year"]), jnp.asarray(z["seconds_of_day"])

    def loss(lat_):
        def one(la, lo, d, s):
            ins, cz, _, _ = integration._compute_insolation(
                la[None], rcfg, lon=lo[None], day_of_year=d, seconds_of_day=s)
            return ins[0] * (1.0 - integration.cesm_ocean_albedo_direct(cz)[0])
        return jnp.sum(jax.vmap(one)(lat_, lon, doy, sod))

    g = jax.grad(loss)(lat)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(g) != 0.0)
    ga = jax.grad(lambda c: jnp.sum(integration.cesm_ocean_albedo_direct(c)))(
        jnp.asarray(z["cosz"]))
    assert np.all(np.isfinite(np.asarray(ga)))
