"""Oracle match: CESM2.1 direct/diffuse open-ocean albedo on the FV pipeline lane.

Oracle and fixture are the merged tier-1 ones (see test_rad_boundary_oracle.py):
cime5.6.47 (d8e9b894d7eb96fe96f8f8a6943b421147f94606) seq_flux_mct.F90
L881-894 run as compiled Fortran, fixture ``oracle_fixtures/rad_boundary.npz``
(3024 points), same metric (max relative error, fp64, 1e-10).

Ours = the FV / lat-lon / cube lane (``physics_pipeline``): with
``dynamic_albedo`` and rrtmgp, ``compute_radiation_core`` builds the DIFFUSE
blend (open ocean 0.06) and the rrtmgp ``radiation_fn`` hands the solver a
DIRECT albedo whose open-ocean part is the CESM curve at the call's own cosine.
The RRTMGP solver is replaced by a stub that echoes the two surface albedos it
receives (lw_flux_up top = direct, lw_flux_up surface = diffuse), so the test
reads exactly what the solver would see.
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
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as rrtmgp_mod
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle, earth_orbit, earth_sun_distance_factor,
)
from legoesm.driver import physics_pipeline as pp
from legoesm.driver.config import ExperimentConfig
from legoesm.grids.latlon import create_latlon_grid

_FIX = Path(__file__).parent / "oracle_fixtures" / "rad_boundary.npz"
_TOL = 1e-10  # same oracle-match bound as the merged tier-1 test
NLEV = 3


def _rel(ours, ref):
    ours, ref = np.asarray(ours), np.asarray(ref)
    assert ours.shape == ref.shape and np.all(ref != 0.0)
    return float(np.max(np.abs(ours - ref) / np.abs(ref)))


class _EchoSolver:
    """Stands in for RRTMGP: echoes the surface albedos it is handed."""

    def solve_columns(self, **kw):
        T = kw["T"]
        ncol, nlev = T.shape
        dif = jnp.broadcast_to(kw["sfc_albedo"], (ncol,))
        dirr = jnp.broadcast_to(kw["sfc_albedo_direct"], (ncol,))
        lw_up = jnp.zeros((ncol, nlev + 1)).at[:, 0].set(dirr).at[:, -1].set(dif)
        sw_up = jnp.zeros((ncol, nlev + 1)).at[:, 0].set(1.0).at[:, -1].set(dirr)
        z = jnp.zeros((ncol, nlev))
        return RadiationOutput(
            lw_flux_up=lw_up, lw_flux_down=jnp.zeros((ncol, nlev + 1)),
            sw_flux_up=sw_up, sw_flux_down=jnp.ones((ncol, nlev + 1)),
            heating_rate=z, lw_heating_rate=z, sw_heating_rate=z)


@pytest.fixture
def echo_solver(monkeypatch):
    monkeypatch.setattr(rrtmgp_mod.RRTMGP, "from_legoesm_config",
                        staticmethod(lambda cfg: _EchoSolver()))


def _cfg(**kw):
    base = dict(radiation="rrtmgp", microphysics="none", diurnal_cycle=True,
                orbital_insolation=True, dynamic_albedo=True)
    base.update(kw)
    return ExperimentConfig(**base)


def test_pipeline_direct_albedo_matches_cesm_coupler(echo_solver):
    z = np.load(_FIX)
    orbit = json.loads(str(z["meta"]))["orbit"]
    eo = earth_orbit()
    assert (eo.eccentricity, eo.obliquity_deg, eo.long_perihelion_deg) == (
        orbit["eccentricity"], orbit["obliquity_deg"], orbit["long_perihelion_deg"])
    rad_fn, _ = pp._build_rrtmgp_radiation_fn(_cfg())
    T = jnp.full((1, NLEV), 250.0)
    p_half = jnp.asarray([[1.0e4, 4.0e4, 7.0e4, 1.0e5]])
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])

    def one(lat, lon, doy, sod):
        out = rad_fn(T, p_full, p_half, jnp.full((1, NLEV), 1.0e-3),
                     jnp.full((1,), 290.0), lat[None], lon[None], doy, sod,
                     jnp.full((1,), integration.OCEAN_ALBEDO_DIFFUSE_CESM),
                     jnp.ones(1), None, None, jnp.zeros(0),
                     open_ocean_frac_col=jnp.ones(1))
        return out.lw_flux_up[0, 0], out.lw_flux_up[0, -1]

    a_dir, a_dif = jax.vmap(one)(*(jnp.asarray(z[k]) for k in
                                   ("lat", "lon", "day_of_year", "seconds_of_day")))
    assert _rel(a_dir, z["albedo_dir"]) <= _TOL
    day = z["cosz"] > 0.0
    assert day.sum() > 1000 and (~day).sum() > 1000
    np.testing.assert_array_equal(np.asarray(a_dif)[day], z["albedo_dif"][day])


def _core(pipe, **kw):
    ad = pipe.adapter
    s2 = ad.shape_2d
    s3 = (*s2, NLEV)
    lat = jnp.broadcast_to(jnp.linspace(-1.2, 1.2, s2[0])[:, None], s2)
    lon = jnp.broadcast_to(jnp.linspace(0.0, 6.0, s2[1])[None, :], s2)
    sic = jnp.broadcast_to(jnp.linspace(0.0, 1.0, s2[1])[None, :], s2)
    doy, sod = 172.0, 43200.0
    out = pipe.compute_radiation_core(
        jnp.full(s3, 270.0), jnp.full(s2, 1.0e5), jnp.full(s3, 0.003),
        jnp.full(s2, 295.0), sic, lat, lon, doy, sod, jnp.zeros(0),
        constants.S_0, None, None, u=jnp.zeros(s3), v=jnp.zeros(s3), dt=600.0,
        **kw)
    cos = jnp.maximum(cos_zenith_angle(lat, lon, doy, sod / 3600.0,
                                       orbit=pipe.orbit), 0.0)
    return out, sic, cos, earth_sun_distance_factor(doy, pipe.orbit)


def test_compute_radiation_core_splits_open_ocean_albedo(echo_solver):
    pipe = pp.build_physics_pipeline(create_latlon_grid(6, 8),
                                     _sigma(), _cfg())
    assert pipe.cesm_ocean_albedo is True
    out, sic, cos, eccf = _core(pipe)
    dif = sic * pipe.albedo_ice + (1.0 - sic) * integration.OCEAN_ALBEDO_DIFFUSE_CESM
    want_dir = dif + (1.0 - sic) * (integration.cesm_ocean_albedo_direct(cos)
                                     - integration.OCEAN_ALBEDO_DIFFUSE_CESM)
    np.testing.assert_allclose(np.asarray(-out[2]), np.asarray(dif), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(out[4]), np.asarray(want_dir), rtol=1e-12)
    # surface net SW is the solver's own down - up (two surface albedos)
    np.testing.assert_allclose(np.asarray(out[1]),
                               np.asarray(eccf * (1.0 - want_dir)), rtol=1e-12)
    with pytest.raises(ValueError, match="direct-beam albedo"):
        _core(pipe, sfc_albedo_override=jnp.full(pipe.adapter.shape_2d, 0.2))


def test_gray_keeps_single_albedo_path():
    pipe = pp.build_physics_pipeline(create_latlon_grid(6, 8), _sigma(),
                                     _cfg(radiation="gray"))
    assert pipe.dynamic_albedo is True and pipe.cesm_ocean_albedo is False


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_direct_albedo_gradient_finite(dtype):
    """d(direct albedo)/d(cos z, open-ocean fraction) on all fixture points
    (night included), fp64 and fp32."""
    z = np.load(_FIX)
    cos = jnp.asarray(z["cosz"], dtype=dtype)
    frac = jnp.full_like(cos, 0.7)
    g_c, g_f = jax.grad(lambda c, f: jnp.sum(integration.cesm_direct_surface_albedo(
        jnp.full_like(c, 0.3), f, c)), argnums=(0, 1))(cos, frac)
    for g in (g_c, g_f):
        assert g.dtype == dtype
        assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0.0)


def _sigma(nlev=NLEV):
    class _S:
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
        dsigma = jnp.diff(sigma_half)

        def pressure_at_full(self, p_s):
            return p_s[..., None] * self.sigma_full

        def pressure_at_half(self, p_s):
            return p_s[..., None] * self.sigma_half

        def layer_thickness_dp(self, p_s):
            return p_s[..., None] * self.dsigma
    return _S()
