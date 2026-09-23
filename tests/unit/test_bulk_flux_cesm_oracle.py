"""Oracle pin: ``bulk_scheme="large_yeager_cesm"`` reproduces CESM/CIME's
``shr_flux_atmOcn`` (cime5.6.47 ``shr_flux_mod.F90``, the CAM6/CESM2 coupler
air-sea flux law).

``_shr_flux_atmocn_ref`` below is a line-by-line NumPy transcription of the
Fortran (column loop, ``do while`` iteration with ``flux_con_tol = 0`` /
``flux_con_max_iter = 2``, ``gust_fac = 0``, cold-air-outbreak off) with
``ssq`` supplied by the caller (the model's own saturation curve + salinity
factor; CESM's private ``qsat`` fit is deliberately not re-derived).  CESM's
fluxes are positive DOWNWARD and its stress is the wind-direction stress on
the ocean; legoESM's are positive UPWARD with the stress opposing the wind,
so the pin compares ``-sen``, ``-lat``, ``-taux``, ``-tauy``.

Run with ``JAX_ENABLE_X64=1`` (the pin is at rel 1e-10).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.bulk_flux import (
    _CESM_MINWIND,
    compute_sam_oceflx_fluxes,
    validate_bulk_scheme,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_KARMAN = constants.kappa_vk
_G = constants.g
_ZVIR = constants.R_v / constants.R_d - 1.0
_CPVIR = constants.c_pv / constants.c_pd - 1.0


def _cdn(u):
    return 0.0027 / u + 0.000142 + 0.0000764 * u


def _psimhu(x):
    return math.log((1.0 + x * (2.0 + x)) * (1.0 + x * x) / 8.0) - 2.0 * math.atan(x) + 1.571


def _psixhu(x):
    return 2.0 * math.log((1.0 + x * x) / 2.0)


def _fsign(a, b):
    """Fortran SIGN(a, b): |a| with the sign of b (b == 0 -> positive)."""
    return abs(a) if b >= 0.0 else -abs(a)


_CLAMPED: list = []


def _shr_flux_atmocn_ref(zbot, ubot, vbot, thbot, qbot, rbot, ts, us, vs, ssq,
                         minwind=_CESM_MINWIND, zref=10.0, ztref=2.0):
    """Transcription of shr_flux_atmOcn (lines 309-444) for ONE column."""
    flux_con_tol = 0.0
    flux_con_max_iter = 2
    vmag_old = max(minwind, math.sqrt((ubot - us) ** 2 + (vbot - vs) ** 2))
    vmag = vmag_old                                   # gust_fac = 0
    delt = thbot - ts
    delq = qbot - ssq
    alz = math.log(zbot / zref)
    cp = constants.c_pd * (1.0 + _CPVIR * ssq)
    stable = 0.5 + _fsign(0.5, delt)
    rdn = math.sqrt(_cdn(vmag))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = 0.0346
    ustar = rdn * vmag
    tstar = rhn * delt
    qstar = ren * delq
    ustar_prev = ustar * 2.0
    it = 0
    while abs((ustar - ustar_prev) / ustar) > flux_con_tol and it < flux_con_max_iter:
        it += 1
        ustar_prev = ustar
        hol = _KARMAN * _G * zbot * (tstar / thbot + qstar / (1.0 / _ZVIR + qbot)) / ustar ** 2
        hol = _fsign(min(abs(hol), 10.0), hol)
        stable = 0.5 + _fsign(0.5, hol)
        xsq = max(math.sqrt(abs(1.0 - 16.0 * hol)), 1.0)
        xqq = math.sqrt(xsq)
        psimh = -5.0 * hol * stable + (1.0 - stable) * _psimhu(xqq)
        psixh = -5.0 * hol * stable + (1.0 - stable) * _psixhu(xqq)
        rd = rdn / (1.0 + rdn / _KARMAN * (alz - psimh))
        u10n = vmag * rd / rdn
        rdn = math.sqrt(_cdn(u10n))
        ren = 0.0346
        rhn = (1.0 - stable) * 0.0327 + stable * 0.018
        rd = rdn / (1.0 + rdn / _KARMAN * (alz - psimh))
        rh = rhn / (1.0 + rhn / _KARMAN * (alz - psixh))
        re = ren / (1.0 + ren / _KARMAN * (alz - psixh))
        ustar = rd * vmag
        tstar = rh * delt
        qstar = re * delq
    assert it >= 1
    _hol_raw = _KARMAN * _G * zbot * (tstar / thbot + qstar / (1.0 / _ZVIR + qbot)) / ustar ** 2
    _CLAMPED.append(abs(_hol_raw) > 10.0)
    tau = rbot * ustar * ustar
    taux = tau * (ubot - us) / vmag_old
    tauy = tau * (vbot - vs) / vmag_old
    sen = cp * tau * tstar / ustar
    lat = constants.L_v * tau * qstar / ustar
    # 2 m reference temperature diagnostic
    al2 = math.log(zref / ztref)
    hol = hol * ztref / zbot
    xsq = max(1.0, math.sqrt(abs(1.0 - 16.0 * hol)))
    xqq = math.sqrt(xsq)
    psix2 = -5.0 * hol * stable + (1.0 - stable) * _psixhu(xqq)
    fac = (rh / _KARMAN) * (alz + al2 - psixh + psix2)
    tref = thbot - delt * fac
    tref = tref - 0.01 * ztref
    return dict(sen=sen, lat=lat, taux=taux, tauy=tauy, ustar=ustar, tref=tref)


def _states(n=20, seed=7):
    """Synthetic air-sea states spanning stable / unstable / calm / windy."""
    rng = np.random.default_rng(seed)
    zbot = rng.uniform(20.0, 120.0, n)
    ubot = rng.uniform(-15.0, 15.0, n)
    vbot = rng.uniform(-10.0, 10.0, n)
    us = rng.uniform(-0.5, 0.5, n)
    vs = rng.uniform(-0.5, 0.5, n)
    ts = rng.uniform(271.5, 304.0, n)
    thbot = ts + rng.uniform(-6.0, 4.0, n)           # both signs of delt
    rbot = rng.uniform(1.05, 1.3, n)
    from legoesm.thermo import saturation_mixing_ratio
    ssq = 0.98 * np.asarray(saturation_mixing_ratio(
        jnp.asarray(ts), jnp.asarray(rbot * constants.R_d * ts)))
    qbot = ssq * rng.uniform(0.5, 1.05, n)
    # calm column below the CESM minimum wind, an exactly-neutral one, and a
    # strongly unstable calm deep-layer one that drives |hol| onto its +-10 clamp
    ubot[0], vbot[0], us[0], vs[0] = 0.2, 0.1, 0.0, 0.0
    thbot[1] = ts[1]
    ubot[2], vbot[2], zbot[2], thbot[2] = 0.6, 0.0, 120.0, ts[2] - 9.0
    return dict(zbot=zbot, ubot=ubot, vbot=vbot, thbot=thbot, qbot=qbot,
                rbot=rbot, ts=ts, us=us, vs=vs, ssq=ssq)


def _model(st, **kw):
    j = {k: jnp.asarray(v) for k, v in st.items()}
    return compute_sam_oceflx_fluxes(
        j["ubot"] - j["us"], j["vbot"] - j["vs"], j["thbot"], j["qbot"],
        j["ts"], j["ssq"], j["rbot"], z_bot=j["zbot"], variant="cesm",
        return_2m=True, **kw,
    )


def test_cesm_variant_pins_shr_flux_atmocn():
    st = _states()
    tau_x, tau_y, shflx, lhflx, ustar, tref = _model(st)
    for i in range(len(st["zbot"])):
        ref = _shr_flux_atmocn_ref(**{k: float(v[i]) for k, v in st.items()})
        np.testing.assert_allclose(float(tau_x[i]), -ref["taux"], rtol=1e-10)
        np.testing.assert_allclose(float(tau_y[i]), -ref["tauy"], rtol=1e-10)
        np.testing.assert_allclose(float(shflx[i]), -ref["sen"], rtol=1e-10)
        np.testing.assert_allclose(float(lhflx[i]), -ref["lat"], rtol=1e-10)
        np.testing.assert_allclose(float(ustar[i]), ref["ustar"], rtol=1e-10)
        np.testing.assert_allclose(float(tref[i]), ref["tref"], rtol=1e-10)


def test_states_cover_both_stability_branches_the_wind_floor_and_the_hol_clamp():
    st = _states()
    delt = st["thbot"] - st["ts"]
    assert (delt > 0).sum() >= 3 and (delt < 0).sum() >= 3
    assert math.hypot(st["ubot"][0] - st["us"][0], st["vbot"][0] - st["vs"][0]) < _CESM_MINWIND
    _CLAMPED.clear()
    for i in range(len(st["zbot"])):
        _shr_flux_atmocn_ref(**{k: float(v[i]) for k, v in st.items()})
    assert any(_CLAMPED), "no state reaches the |hol| = 10 clamp"


def test_most_solver_refuses_the_cesm_name():
    from legoesm.core.bulk_flux import compute_most_fluxes
    st = _states(n=3)
    j = {k: jnp.asarray(v) for k, v in st.items()}
    with pytest.raises(ValueError, match="large_yeager_cesm"):
        compute_most_fluxes(j["ubot"], j["vbot"], j["thbot"], j["qbot"], j["ts"],
                            j["ssq"], j["rbot"], scheme="large_yeager_cesm")


def test_sam_variant_is_not_the_cesm_law():
    """Non-vacuity: the SAM oracle differs (1 m/s floor, floored cdn, dry cp)."""
    st = _states()
    j = {k: jnp.asarray(v) for k, v in st.items()}
    sam = compute_sam_oceflx_fluxes(
        j["ubot"] - j["us"], j["vbot"] - j["vs"], j["thbot"], j["qbot"],
        j["ts"], j["ssq"], j["rbot"], z_bot=j["zbot"], variant="sam")
    cesm = _model(st)
    assert not np.allclose(np.asarray(sam[2]), np.asarray(cesm[2]), rtol=1e-6)
    assert not np.allclose(np.asarray(sam[0]), np.asarray(cesm[0]), rtol=1e-6)


def test_minwind_is_honoured():
    st = _states()
    a = _model(st)
    b = _model(st, minwind=2.0)
    ref = _shr_flux_atmocn_ref(**{k: float(v[0]) for k, v in st.items()}, minwind=2.0)
    np.testing.assert_allclose(float(b[4][0]), ref["ustar"], rtol=1e-10)
    assert not np.isclose(float(a[4][0]), float(b[4][0]))


def test_unknown_variant_raises():
    st = _states(n=3)
    j = {k: jnp.asarray(v) for k, v in st.items()}
    with pytest.raises(ValueError, match="variant"):
        compute_sam_oceflx_fluxes(
            j["ubot"], j["vbot"], j["thbot"], j["qbot"], j["ts"], j["ssq"],
            j["rbot"], z_bot=j["zbot"], variant="bogus")


def test_cesm_variant_is_differentiable():
    st = _states(n=3)
    j = {k: jnp.asarray(v) for k, v in st.items()}

    def loss(u):
        out = compute_sam_oceflx_fluxes(
            u, j["vbot"], j["thbot"], j["qbot"], j["ts"], j["ssq"], j["rbot"],
            z_bot=j["zbot"], variant="cesm")
        return jnp.sum(out[2] + out[3] + out[0])

    g = jax.grad(loss)(j["ubot"])
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0.0)


# --- dispatch: the scheme name reaches every ocean flux site -----------------

def test_scheme_name_is_valid_and_reaches_surface_layer():
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )
    validate_bulk_scheme("large_yeager_cesm")
    st = _states(n=6)
    j = {k: jnp.asarray(v) for k, v in st.items()}
    cfg = SurfaceLayerConfig(bulk_scheme="large_yeager_cesm")
    got = compute_surface_fluxes(
        j["ubot"], j["vbot"], j["thbot"], j["qbot"], j["ts"], j["ssq"],
        j["rbot"], cfg, z_ref=j["zbot"])
    want = compute_sam_oceflx_fluxes(
        j["ubot"], j["vbot"], j["thbot"], j["qbot"], j["ts"], j["ssq"],
        j["rbot"], z_bot=j["zbot"], variant="cesm")
    for a, b in zip(got, want):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)
    # and it is NOT the constant-coefficient fall-through
    const = compute_surface_fluxes(
        j["ubot"], j["vbot"], j["thbot"], j["qbot"], j["ts"], j["ssq"],
        j["rbot"], SurfaceLayerConfig(bulk_scheme="constant"))
    assert not np.allclose(np.asarray(got[2]), np.asarray(const[2]), rtol=1e-3)


def test_scheme_reaches_coupled_ocean_tile():
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.coupler import ocean_tile_response
    shape = (3,)
    forcing = AtmToSurface(
        sw_down=jnp.full(shape, 200.0), lw_down=jnp.full(shape, 350.0),
        precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
        T_lowest=jnp.array([295.0, 280.0, 301.0]),
        q_lowest=jnp.array([0.012, 0.005, 0.02]),
        u_lowest=jnp.array([8.0, 0.1, -12.0]), v_lowest=jnp.zeros(shape),
        p_lowest=jnp.full(shape, 100000.0), p_surface=jnp.full(shape, 101325.0),
        rho_lowest=jnp.full(shape, 1.2), cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )
    sst = jnp.full(shape, 300.0)
    resp = ocean_tile_response(
        forcing, sst, jnp.zeros(shape), jnp.zeros(shape),
        CouplerConfig(bulk_scheme="large_yeager_cesm"))
    want = compute_sam_oceflx_fluxes(
        forcing.u_lowest, forcing.v_lowest, forcing.T_lowest, forcing.q_lowest,
        sst, resp.q_surface, forcing.rho_lowest, z_bot=10.0, variant="cesm")
    np.testing.assert_allclose(np.asarray(resp.shflx), np.asarray(want[2]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(resp.lhflx), np.asarray(want[3]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(resp.tau_x), np.asarray(want[0]), rtol=1e-12)


def test_scheme_selectable_from_experiment_config_and_amip_cli():
    from legoesm.driver.config import VALID_SURFACE_BULK, ExperimentConfig
    assert "large_yeager_cesm" in VALID_SURFACE_BULK
    ExperimentConfig(surface_bulk_scheme="large_yeager_cesm",
                     turbulence="louis").validate_strict()
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        _STABILITY_SCHEMES,
    )
    assert "large_yeager_cesm" in _STABILITY_SCHEMES
    from scripts.run.run_amip import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args(["--surface-bulk-scheme", "large_yeager_cesm"])
    assert args.surface_bulk_scheme == "large_yeager_cesm"
