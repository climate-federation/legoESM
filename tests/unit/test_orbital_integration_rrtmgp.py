"""Orbital insolation in the shared radiation `integration` path (MPAS/spectral).

Codex adversarial review (round 3, HIGH): when RadiationConfig.orbit is set,
_compute_insolation returned eccentricity-scaled insolation, but the RRTMGP
consumer (_call_radiation_backend) used cos_sza for the solver and folded the
distance factor into the wrong quantity — so the +/-3.4% orbital change hit
diagnostics but NOT the actual RRTMGP shortwave fluxes.

These tests prove the fix on the SW FLUX (not only toa_insolation):
_compute_insolation now returns eccf separately, and _call_radiation_backend
keeps cos_zenith geometric and applies eccf as a SW-flux scale.  A fake solver
isolates the rescale arithmetic from the (heavy) gas-optics tables.

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
import legoesm.atmosphere.physics.radiation.integration as rad_int
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    _call_radiation_backend,
    _compute_insolation,
    make_radiation_physics,
)
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.physics.radiation.solar import (
    earth_orbit,
    earth_sun_distance_factor,
)

S0 = float(constants.S_0)
NCOL, NLEV = 2, 2


class _FakeSolver:
    """A solver stub returning unit SW/LW fluxes so the eccf rescale on the
    SW fields is directly observable (the real gas-optics tables are not
    needed to test the flux-scaling arithmetic)."""

    def solve_columns(self, *, T, **kw):
        ncol, nlev = T.shape
        flux = jnp.ones((ncol, nlev + 1))
        heat = jnp.ones((ncol, nlev))
        return RadiationOutput(
            lw_flux_up=flux, lw_flux_down=flux,
            sw_flux_up=flux, sw_flux_down=flux,
            heating_rate=heat, lw_heating_rate=heat, sw_heating_rate=heat,
        )


def _inputs():
    T = jnp.full((NCOL, NLEV), 260.0)
    p_full = jnp.broadcast_to(jnp.array([3e4, 7e4]), (NCOL, NLEV))
    p_half = jnp.broadcast_to(jnp.array([1e4, 5e4, 1e5]), (NCOL, NLEV + 1))
    T_sfc = jnp.full((NCOL,), 290.0)
    q_v = jnp.full((NCOL, NLEV), 1e-3)
    o3 = jnp.zeros((NCOL, NLEV))  # override -> skip the ozone computation
    return T, p_full, p_half, T_sfc, q_v, o3


def _run_backend(cfg, cos_sza, insol, eccf, f_day=None):
    T, p_full, p_half, T_sfc, q_v, o3 = _inputs()
    return _call_radiation_backend(
        radiation_config=cfg,
        eccf=eccf,
        T=T, p_full=p_full, p_half=p_half,
        sfc_temperature=T_sfc, lat=jnp.zeros((NCOL,)), q_v=q_v,
        insolation=insol, cos_sza=cos_sza, f_day=f_day,
        o3_vmr_override=o3, rrtmgp_solver=_FakeSolver(),
    )


def _cfg(orbit):
    return RadiationConfig(scheme="rrtmgp", rrtmgp=RRTMGPConfig(S_0=S0),
                           cloud_scheme="none", orbit=orbit)


def test_diurnal_rrtmgp_sw_flux_scaled_by_eccf():
    """Diurnal: cos_zenith stays geometric; the SW fluxes are scaled by the
    (a/r)^2 distance factor (LW untouched)."""
    cos = jnp.array([0.8, 0.5])
    day = 3.0  # perihelion
    eccf = float(earth_sun_distance_factor(day, earth_orbit()))
    off = _run_backend(_cfg(None), cos, S0 * cos, eccf=1.0)
    on = _run_backend(_cfg(earth_orbit()), cos, S0 * eccf * cos, eccf=eccf)
    # SW fluxes scaled by eccf...
    r = np.asarray(on.sw_flux_down) / np.asarray(off.sw_flux_down)
    assert np.allclose(r, eccf, rtol=1e-6)
    assert np.allclose(np.asarray(on.sw_heating_rate)
                       / np.asarray(off.sw_heating_rate), eccf, rtol=1e-6)
    # ...LW untouched.
    assert np.allclose(np.asarray(on.lw_flux_down),
                       np.asarray(off.lw_flux_down))


def test_daily_mean_rrtmgp_keeps_cos_zenith_geometric():
    """Daily-mean: eccf must NOT inflate cos_zenith (the optical path stays
    <= the circular-orbit geometric value); it is applied to the SW flux via
    _sw_scale = f_day * eccf instead."""
    f_day = jnp.array([0.5, 0.5])
    day = 3.0
    eccf = float(earth_sun_distance_factor(day, earth_orbit()))
    # daily-mean insolation already carries eccf (the _compute_insolation
    # contract); the backend divides it out for cos_zenith.
    q_geom = jnp.array([300.0, 200.0])  # geometric daily-mean
    off = _run_backend(_cfg(None), None, q_geom, eccf=1.0, f_day=f_day)
    on = _run_backend(_cfg(earth_orbit()), None, q_geom * eccf, eccf=eccf,
                      f_day=f_day)
    # SW flux scaled by eccf (f_day cancels in the ratio).
    r = np.asarray(on.sw_flux_down) / np.asarray(off.sw_flux_down)
    assert np.allclose(r, eccf, rtol=1e-6)


def test_compute_insolation_returns_eccf_per_branch():
    """_compute_insolation's 4th return value is the (a/r)^2 factor: the real
    distance factor for diurnal / daily-mean, exactly 1.0 for the idealized
    RCE / perpetual-equinox branches."""
    orbit = earth_orbit()
    lat = jnp.array([0.0, 0.5])
    lon = jnp.zeros_like(lat)
    day = 3.0
    eccf_ref = float(earth_sun_distance_factor(day, orbit))

    # diurnal
    cfg = RadiationConfig(scheme="rrtmgp", diurnal_cycle=True, orbit=orbit)
    *_, eccf = _compute_insolation(lat, cfg, lon=lon, day_of_year=day,
                                   seconds_of_day=43200.0)
    assert float(eccf) == pytest.approx(eccf_ref, rel=1e-9)

    # daily-mean
    cfg_dm = RadiationConfig(scheme="rrtmgp", diurnal_cycle=False, orbit=orbit)
    cfg_dm = cfg_dm._replace(gray=cfg_dm.gray._replace(perpetual_equinox=False))
    *_, eccf_dm = _compute_insolation(lat, cfg_dm, day_of_year=day)
    assert float(eccf_dm) == pytest.approx(eccf_ref, rel=1e-9)

    # circular orbit -> 1.0
    cfg_circ = RadiationConfig(scheme="rrtmgp", diurnal_cycle=True, orbit=None)
    *_, eccf_circ = _compute_insolation(lat, cfg_circ, lon=lon,
                                        day_of_year=day, seconds_of_day=43200.0)
    assert float(eccf_circ) == 1.0

    # RCE fixed-zenith idealization -> 1.0 (orbit does not apply)
    cfg_rce = RadiationConfig(scheme="rrtmgp", rce_fixed_cos_zenith=0.62,
                              orbit=orbit)
    *_, eccf_rce = _compute_insolation(lat, cfg_rce, day_of_year=day)
    assert float(eccf_rce) == 1.0


class _StopAfterBackend(Exception):
    """Sentinel: stop the radiation_fn right after the backend call so we test
    the eccf wiring without needing a real SH transform / spectral state."""


def _spectral_pe_ducks(n_lat=4, n_lon=6, nlev=3):
    grid = SimpleNamespace(lat=jnp.linspace(-1.2, 1.2, n_lat),
                           lon=jnp.linspace(0.0, 6.0, n_lon))
    p_s = jnp.full((n_lat, n_lon), 1.0e5)

    class _Sigma:
        n_levels = nlev

        def pressure_at_full(self, ps):
            return jnp.broadcast_to(jnp.linspace(3e4, 9e4, nlev),
                                    (n_lat, n_lon, nlev))

        def pressure_at_half(self, ps):
            return jnp.broadcast_to(jnp.linspace(1e4, 1e5, nlev + 1),
                                    (n_lat, n_lon, nlev + 1))

    grid_fields = {"T": jnp.full((n_lat, n_lon, nlev), 260.0), "p_s": p_s}
    ncol = n_lat * n_lon
    state = SimpleNamespace(
        tracers={"q_v": jnp.full((ncol, nlev), 0.01)},
        T=SimpleNamespace(data=jnp.zeros((ncol, nlev))),
    )
    return state, grid, _Sigma(), grid_fields


@pytest.mark.parametrize("orbital,expect_factor", [(True, None), (False, 1.0)])
def test_spectral_pe_nondiurnal_forwards_eccf(monkeypatch, orbital, expect_factor):
    """Regression for the codex round-4 HIGH: the spectral-PE NON-diurnal
    radiation branch must bind eccf (it shares the ``eccf=eccf`` backend
    kwarg with the diurnal branch).  Before the fix this raised
    UnboundLocalError on every non-diurnal spectral run."""
    from legoesm.atmosphere.physics.radiation.solar import (
        earth_orbit, earth_sun_distance_factor,
    )

    captured: dict = {}

    def _spy(**kw):
        captured["eccf"] = kw.get("eccf")
        raise _StopAfterBackend

    monkeypatch.setattr(rad_int, "_call_radiation_backend", _spy)

    orbit = earth_orbit() if orbital else None
    cfg = RadiationConfig(
        scheme="gray", diurnal_cycle=False, orbit=orbit,
        # perpetual_equinox=False forces the daily-mean (orbital) branch.
        gray=GrayRadiationConfig(perpetual_equinox=False),
    )
    fn = make_radiation_physics(cfg, model_type="spectral_pe")
    state, grid, sigma, grid_fields = _spectral_pe_ducks()

    day = 3.0  # perihelion
    with pytest.raises(_StopAfterBackend):
        fn(state, grid, sigma, grid_fields=grid_fields,
           forcing={"day_of_year": day})

    eccf = float(captured["eccf"])
    ref = (float(earth_sun_distance_factor(day, earth_orbit()))
           if orbital else expect_factor)
    assert eccf == pytest.approx(ref, rel=1e-9)


class _SpySolver(_FakeSolver):
    """Records the optical-path cosine the solver was handed."""

    def solve_columns(self, *, T, cos_zenith, **kw):
        self.cos = np.asarray(cos_zenith)
        return super().solve_columns(T=T, **kw)


def _run_tsi(cos_sza, insol, tsi, f_day=None):
    T, p_full, p_half, T_sfc, q_v, o3 = _inputs()
    spy = _SpySolver()
    out = _call_radiation_backend(
        radiation_config=_cfg(None), tsi=tsi,
        T=T, p_full=p_full, p_half=p_half,
        sfc_temperature=T_sfc, lat=jnp.zeros((NCOL,)), q_v=q_v,
        insolation=insol, cos_sza=cos_sza, f_day=f_day,
        o3_vmr_override=o3, rrtmgp_solver=spy,
    )
    return out, spy.cos


@pytest.mark.parametrize("diurnal", [True, False])
def test_transient_tsi_scales_rrtmgp_sw_flux_not_cos_zenith(diurnal):
    """Transient TSI (2026-10-06, codex r2 P1): the wrappers pass insolation
    already scaled by tsi/S_0, but the RRTMGP solver runs at its configured
    S_0.  Diurnal: TSI used to be dropped (geometric cos_sza, no flux scale).
    Daily-mean: TSI used to inflate cos_zenith instead of the flux.  Both
    branches must now scale the SW flux by tsi/S_0 with an unchanged cosine;
    tsi=None stays the static-S_0 path."""
    ratio = 1.01
    cos = jnp.array([0.8, 0.5]) if diurnal else None
    f_day = None if diurnal else jnp.array([0.5, 0.5])
    q = jnp.array([300.0, 200.0]) if cos is None else S0 * cos
    off, cos_off = _run_tsi(cos, q, None, f_day)
    on, cos_on = _run_tsi(cos, q * ratio, jnp.asarray(S0 * ratio), f_day)
    np.testing.assert_allclose(cos_on, cos_off, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(on.sw_flux_down)
                               / np.asarray(off.sw_flux_down), ratio, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(on.sw_heating_rate)
                               / np.asarray(off.sw_heating_rate), ratio, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(on.lw_flux_down),
                                  np.asarray(off.lw_flux_down))
