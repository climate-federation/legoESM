"""Layered-pack snow density against line-by-line transliterations of CTSM
clm5.0 ``SnowHydrologyMod`` (``SnowCompaction`` with Vionnet 2012 overburden and
``WindDriftCompaction``; ``NewSnowBulkDensity`` with Slater 2017 + wind), CLM5
parameters (clm50_params.c250311), ``frac_sno = 1``.  Run with
``JAX_ENABLE_X64=1``."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.snow_column import (SnowColumnState, _remap_equal_mass,
                                      clm5_compaction_rate, new_snow_bulk_density,
                                      snow_add_mass, snow_compact)

jax.config.update("jax_enable_x64", True)

TF = constants.T_freeze


def _ctsm_wind_drift(bi, forc_wind, dz, zpseudo, mobile):
    """CTSM WindDriftCompaction (oracle)."""
    rho_min, rho_max, drift_gs, drift_sph, tau_ref = 50.0, 350.0, 0.35e-3, 1.0, 48.0 * 3600.0
    if mobile:
        Frho = 1.25 - 0.0042 * (max(rho_min, bi) - rho_min)
        MO = 0.34 * (-0.583 * drift_gs - 0.833 * drift_sph + 0.833) + 0.66 * Frho
        SI = -2.868 * math.exp(-0.085 * forc_wind) + 1.0 + MO
        if SI > 0.0:
            SI = min(SI, 3.25)
            zpseudo = zpseudo + 0.5 * dz * (3.25 - SI)
            gamma_drift = SI * math.exp(-zpseudo / 0.1)
            tau_inverse = gamma_drift / tau_ref
            rate = -max(0.0, rho_max - bi) * tau_inverse
            zpseudo = zpseudo + 0.5 * dz * (3.25 - SI)
        else:
            mobile = False
            rate = 0.0
    else:
        rate = 0.0
    return rate, zpseudo, mobile


def _ctsm_snow_compaction(ice, liq, T, dz, dtime, forc_wind, wind_drift=True):
    """CTSM clm5.0 SnowCompaction for one column, top layer first, frac_sno = 1,
    imelt = 0.  Plain Python on purpose (the oracle)."""
    denice, denh2o, tfrz = constants.rho_ice, constants.rho_water, TF
    c3, c4, c5 = 2.777e-6, 0.04, 2.0
    upplim_destruct_metamorph = 175.0
    ceta, aeta, beta, eta0 = 450.0, 0.1, 0.023, 7.62237e6
    dz = list(dz)
    burden, zpseudo, mobile = 0.0, 0.0, True
    for j in range(len(ice)):
        wx = ice[j] + liq[j]
        void = 1.0 - (ice[j] / denice + liq[j] / denh2o) / dz[j]
        if void > 0.001 and ice[j] > 0.1:
            bi = ice[j] / dz[j]
            td = tfrz - T[j]
            ddz1 = -c3 * math.exp(-c4 * td)
            if bi > upplim_destruct_metamorph:
                ddz1 = ddz1 * math.exp(-46.0e-3 * (bi - upplim_destruct_metamorph))
            if liq[j] > 0.01 * dz[j]:
                ddz1 = ddz1 * c5
            f1 = 1.0 / (1.0 + 60.0 * liq[j] / (denh2o * dz[j]))
            eta = f1 * 4.0 * (bi / ceta) * math.exp(aeta * td + beta * bi) * eta0
            ddz2 = -(burden + wx / 2.0) / eta
            if wind_drift:
                ddz4, zpseudo, mobile = _ctsm_wind_drift(bi, forc_wind, dz[j], zpseudo, mobile)
            else:
                ddz4 = 0.0
            pdzdtc = ddz1 + ddz2 + ddz4
            dz[j] = max(dz[j] * (1.0 + pdzdtc * dtime), ice[j] / denice + liq[j] / denh2o)
        else:
            mobile = False
        burden = burden + wx
    return dz


def _ctsm_bifall(forc_t, forc_wind):
    """CTSM NewSnowBulkDensity, Slater2017, wind_dependent_snow_density (oracle)."""
    tfrz = TF
    if forc_t > tfrz + 2.0:
        bifall = 50.0 + 1.7 * (17.0) ** 1.5
    elif forc_t > tfrz - 15.0:
        bifall = 50.0 + 1.7 * (forc_t - tfrz + 15.0) ** 1.5
    else:
        t_for_bifall_degC = (forc_t - tfrz) if forc_t > tfrz - 57.55 else -57.55
        bifall = (-(50.0 / 15.0 + 0.0333 * 15) * t_for_bifall_degC
                  - 0.0333 * t_for_bifall_degC ** 2)
    if forc_wind > 0.1:
        bifall = bifall + 266.861 * ((1.0 + math.tanh(forc_wind / 5.0)) / 2.0) ** 8.8
    return bifall


# canonical columns: (ice, liq, T, density), top first
_COLUMNS = {
    "fresh_cold": ([6.0] * 5, [0.0] * 5, [TF - 12, TF - 10, TF - 8, TF - 6, TF - 4],
                   [100.0, 120.0, 150.0, 180.0, 220.0]),
    "dense_wet": ([40.0] * 5, [1.5, 1.0, 0.5, 0.0, 0.0], [TF, TF, TF - 0.5, TF - 1, TF - 2],
                  [300.0, 320.0, 350.0, 380.0, 400.0]),
    # trace top layer (no compaction, stops the drift below it) + saturated base
    "trace_and_saturated": ([0.05, 2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.1],
                            [TF - 3] * 5, [100.0, 150.0, 200.0, 250.0, 905.0]),
}
_WINDS = (0.0, 4.0, 15.0)


def _state(name):
    return SnowColumnState(*(jnp.asarray(v, dtype=float)[None] for v in _COLUMNS[name]))


@pytest.mark.parametrize("wind", _WINDS)
@pytest.mark.parametrize("name", sorted(_COLUMNS))
def test_compaction_matches_ctsm_snowcompaction(name, wind):
    ice, liq, T, rho = (np.asarray(v, dtype=float) for v in _COLUMNS[name])
    dt = 1800.0
    st = _state(name)
    dz0 = (ice + liq) / rho
    want = np.asarray(_ctsm_snow_compaction(ice, liq, T, dz0, dt, wind))
    got_rho = np.asarray(snow_compact(st, dt, jnp.asarray([wind])).density[0])
    np.testing.assert_allclose((ice + liq) / got_rho, want, rtol=1e-12)
    rate = np.asarray(clm5_compaction_rate(st, jnp.asarray([wind]))[0])
    implied = (np.asarray(_ctsm_snow_compaction(ice, liq, T, dz0, 1.0, wind)) - dz0) / dz0
    np.testing.assert_allclose(rate, implied, rtol=1e-7, atol=1e-16)


def test_wind_drift_is_active_and_stops_below_an_undriftable_layer():
    """Windy fresh snow: drift dominates the top layer and decays with pseudo
    depth; a trace top layer (not compactable) makes every layer below immobile."""
    st = _state("fresh_cold")
    calm = np.asarray(clm5_compaction_rate(st, jnp.asarray([0.0]))[0])
    windy = np.asarray(clm5_compaction_rate(st, jnp.asarray([15.0]))[0])
    assert windy[0] < 3.0 * calm[0]                      # drift adds compaction on top
    gain = windy - calm
    assert np.all(gain < 0.0) and abs(gain[0]) > abs(gain[-1])
    tr = _state("trace_and_saturated")
    np.testing.assert_allclose(clm5_compaction_rate(tr, jnp.asarray([15.0])),
                               clm5_compaction_rate(tr, jnp.asarray([0.0])), rtol=1e-14)


def test_thirty_day_density_trajectory_matches_ctsm():
    """30 days at a 30-min step, fixed temperatures and 6 m/s wind, compaction
    only: every layer's density tracks the oracle to 1e-9."""
    ice, liq, T, rho = (np.asarray(v, dtype=float) for v in _COLUMNS["fresh_cold"])
    dt, n, wind = 1800.0, 30 * 48, 6.0
    st = _state("fresh_cold")
    step = jax.jit(lambda s: snow_compact(s, dt, jnp.asarray([wind])))
    dz = list((ice + liq) / rho)
    for _ in range(n):
        st = step(st)
        dz = _ctsm_snow_compaction(ice, liq, T, dz, dt, wind)
    np.testing.assert_allclose(np.asarray(st.density[0]), (ice + liq) / np.asarray(dz),
                               rtol=1e-9)
    assert np.all(np.asarray(st.density[0]) > rho)
    # No 450 kg/m3 target any more: a calm month at -12..-4 C stays well below it.
    st_c = _state("fresh_cold")
    calm = jax.jit(lambda s: snow_compact(s, dt, jnp.asarray([0.0])))
    for _ in range(n):
        st_c = calm(st_c)
    assert np.all(np.asarray(st_c.density[0]) < 300.0), st_c.density


@pytest.mark.parametrize("T_air", [TF + 5.0, TF + 1.0, TF - 5.0, TF - 14.9, TF - 15.1,
                                   TF - 30.0, TF - 70.0])
@pytest.mark.parametrize("wind", [0.0, 0.1, 0.11, 3.0, 12.0])
def test_new_snow_density_matches_ctsm(T_air, wind):
    got = float(new_snow_bulk_density(jnp.asarray(T_air), jnp.asarray(wind)))
    assert got == pytest.approx(_ctsm_bifall(T_air, wind), rel=1e-12)


def test_snowfall_adds_its_own_thickness():
    st = _state("fresh_cold")
    dz0 = float(st.swe_ice[0, 0] / st.density[0, 0])
    out = snow_add_mass(st, jnp.asarray([3.0]), jnp.asarray([TF - 5.0]),
                        rho_fresh=jnp.asarray([80.0]))
    dz1 = float((out.swe_ice[0, 0] + out.swe_liq[0, 0]) / out.density[0, 0])
    assert dz1 == pytest.approx(dz0 + 3.0 / 80.0, rel=1e-12)
    with pytest.raises(ValueError, match="rho_fresh"):
        snow_add_mass(st, jnp.asarray([3.0]), jnp.asarray([TF - 5.0]))


def test_remap_conserves_thickness():
    """Merging layers of different density keeps the total thickness (CLM's
    layer combination conserves dz), so the remap itself never densifies."""
    st = _state("fresh_cold")._replace(swe_ice=jnp.asarray([[1.0, 9.0, 3.0, 4.0, 13.0]]))
    ice, liq, T, rho = _remap_equal_mass(*st)
    thick0 = float(jnp.sum((st.swe_ice + st.swe_liq) / st.density))
    thick1 = float(jnp.sum((ice + liq) / rho))
    assert thick1 == pytest.approx(thick0, rel=1e-12)


def test_compaction_conserves_water_and_enthalpy_and_is_differentiable():
    st = _state("dense_wet")
    out = snow_compact(st, 1800.0, jnp.asarray([4.0]))
    np.testing.assert_array_equal(out.swe_ice, st.swe_ice)
    np.testing.assert_array_equal(out.swe_liq, st.swe_liq)
    np.testing.assert_array_equal(out.T, st.T)
    g = jax.grad(lambda t: jnp.sum(snow_compact(st._replace(T=st.T + t), 1800.0,
                                                jnp.asarray([4.0])).density))(0.0)
    assert np.isfinite(float(g)) and float(g) > 0.0      # warmer snow compacts faster
    # drift needs SI > 0, i.e. wind above ~6.3 m/s for this snow
    gw = jax.grad(lambda w: jnp.sum(snow_compact(_state("fresh_cold"), 1800.0,
                                                 jnp.asarray([w])).density))(8.0)
    assert np.isfinite(float(gw)) and float(gw) > 0.0    # windier snow compacts faster
