"""Tests for the NEMO sbcdcy diurnal-shortwave modulation (ln_dm2dc).

1. **F90 transliteration cross-check** — the vectorised jnp.where port
   matches a scalar per-point numpy transliteration of sbcdcy.F90 (with
   its literal IF branches) over lat × lon × day-of-year × window.
2. **Daily-mean preservation** — the window-weighted sum of the factor
   over a full day is exactly 1 wherever there is daylight (0 in polar
   night) — the defining property of sbc_dcy.
3. **Shape sanity** — night factor is 0, the local-noon factor exceeds
   the daily mean, negative first-window lo (NEMO's zlo = -dt/2 at
   midnight) stays finite/correct.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from legoesm.ocean.forcing.diurnal_cycle import diurnal_sw_factor


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _fint(t1, t2, aa, bb, cc):
    tp = 2.0 * math.pi
    return (aa * t2 + bb / tp * math.sin(cc + tp * t2)
            - aa * t1 - bb / tp * math.sin(cc + tp * t1))


def _f90_sbc_dcy_reference(lon_deg, lat_deg, day_of_year, year_len,
                           zlo, zup):
    """Scalar transliteration of sbc_dcy + sbc_dcy_param (sbcdcy.F90)."""
    rad = math.pi / 180.0
    tp = 2.0 * math.pi
    rcc = rad * lon_deg - math.pi
    rtmd = ((0.5 - lon_deg / 360.0) + 1.0) % 1.0
    zdsws = 11.0 + day_of_year
    zdec = (-23.5 * rad) * math.cos(zdsws * tp / year_len)
    raa = math.sin(rad * lat_deg) * math.sin(zdec)
    rbb = math.cos(rad * lat_deg) * math.cos(zdec)
    rab = -raa / rbb if rbb != 0.0 else math.inf * (-np.sign(raa) or 1.0)
    if abs(rab) < 1.0:
        ztx = 1.0 / tp * (math.acos(rab) - rcc)
        ztest = -rbb * math.sin(rcc + tp * ztx)
        if ztest > 0.0:
            rdawn, rdusk = ztx, rtmd + (rtmd - ztx)
        else:
            rdusk, rdawn = ztx, rtmd - (ztx - rtmd)
        rdawn = (rdawn + 1.0) % 1.0
        rdusk = (rdusk + 1.0) % 1.0
        rscal = 0.0
        if rdawn < rdusk:
            if (rdusk - rdawn) >= 0.001:
                rscal = 1.0 / _fint(rdawn, rdusk, raa, rbb, rcc)
        else:
            if (rdusk + (1.0 - rdawn)) >= 0.001:
                rscal = 1.0 / (_fint(0.0, rdusk, raa, rbb, rcc)
                               + _fint(rdawn, 1.0, raa, rbb, rcc))
        if rdawn < rdusk:
            zlousd = min(max(zlo, rdawn), zup)
            zupusd = max(min(zup, rdusk), zlo)
            ztmp = _fint(zlousd, zupusd, raa, rbb, rcc)
        else:
            ztmp = (_fint(min(zlo, rdusk), min(zup, rdusk), raa, rbb, rcc)
                    + _fint(max(zlo, rdawn), max(zup, rdawn), raa, rbb, rcc))
    else:
        if raa > rbb:                       # 24h day
            rscal = 1.0 / _fint(0.0, 1.0, raa, rbb, rcc)
            ztmp = _fint(zlo, zup, raa, rbb, rcc)
        else:                               # polar night
            rscal = 0.0
            ztmp = 0.0
    return max(ztmp * rscal / (zup - zlo), 0.0)


@pytest.mark.parametrize("day_of_year", [1, 80, 172, 264, 355])
def test_matches_f90_transliteration(day_of_year):
    lats = np.array([-85.0, -60.0, -30.0, 0.0, 23.5, 45.0, 70.0, 89.0])
    lons = np.array([-179.0, -90.0, 0.0, 33.7, 120.0, 179.0])
    dt_frac = 3600.0 / 86400.0
    for w in range(0, 24, 5):               # five windows across the day
        zlo = w / 24.0 - 0.5 * dt_frac
        zup = zlo + dt_frac
        ref = np.array([[_f90_sbc_dcy_reference(lo, la, day_of_year, 365.0,
                                                zlo, zup)
                         for lo in lons] for la in lats])
        got = np.asarray(diurnal_sw_factor(
            lons[None, :], lats[:, None],
            day_of_year=day_of_year, year_len_days=365.0,
            t_frac_lo=zlo, t_frac_up=zup))
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12,
                                   err_msg=f"window {w}h day {day_of_year}")


@pytest.mark.parametrize("day_of_year", [15, 172, 355])
def test_daily_mean_preserved(day_of_year):
    lats = np.array([-88.0, -45.0, 0.0, 45.0, 66.6, 88.0])
    lons = np.array([-120.0, 0.0, 77.0])
    n = 48                                   # dt = 30 min
    dtf = 1.0 / n
    acc = np.zeros((lats.size, lons.size))
    any_light = np.zeros_like(acc)
    for k in range(n):
        zlo = k * dtf - 0.5 * dtf
        f = np.asarray(diurnal_sw_factor(
            lons[None, :], lats[:, None],
            day_of_year=day_of_year, year_len_days=365.0,
            t_frac_lo=zlo, t_frac_up=zlo + dtf))
        acc += f * dtf
        any_light = np.maximum(any_light, f)
    # daylight columns integrate back to exactly 1; polar night to 0.
    lit = any_light > 0.0
    np.testing.assert_allclose(acc[lit], 1.0, rtol=1e-9)
    np.testing.assert_allclose(acc[~lit], 0.0, atol=1e-15)
    # both regimes must actually occur in this sweep (incl. polar night)
    assert lit.any() and (~lit).any()


def test_equator_noon_amplification_and_night_zero():
    dtf = 3600.0 / 86400.0
    # local noon at lon=0 is t=0.5; local midnight t=0.
    noon = float(diurnal_sw_factor(
        0.0, 0.0, day_of_year=80, year_len_days=365.0,
        t_frac_lo=0.5 - 0.5 * dtf, t_frac_up=0.5 + 0.5 * dtf))
    midnight = float(diurnal_sw_factor(
        0.0, 0.0, day_of_year=80, year_len_days=365.0,
        t_frac_lo=-0.5 * dtf, t_frac_up=0.5 * dtf))
    assert noon > 2.0            # clear-sky diurnal peak ~pi at the equator
    assert midnight == pytest.approx(0.0, abs=1e-12)


def test_first_window_negative_lo_finite():
    # NEMO's first step of the day has zlo = -dt/2 < 0.
    f = np.asarray(diurnal_sw_factor(
        np.array([0.0, 90.0]), np.array([10.0, 50.0]),
        day_of_year=200, year_len_days=365.0,
        t_frac_lo=-0.5 / 24.0, t_frac_up=0.5 / 24.0))
    assert np.all(np.isfinite(f)) and np.all(f >= 0.0)


def test_dm2dc_sw_factor_mpas_paired_cell_coords():
    """The applicator's shared ``dm2dc_sw_factor`` handles the MPAS Voronoi
    mesh: 1-D PAIRED ``latCell``/``lonCell`` (radians) each carrying its own
    (lat, lon), returning a per-cell ``(nCells,)`` factor -- not a meshgrid.

    Two contracts:
      * mean-preserving -- the window-weighted sum of the factor over a full
        day is ~1 at every daylit cell (defining property of sbc_dcy);
      * NON-VACUOUS -- a single local-noon window factor is finite, >= 0 and
        NOT identically 1 (proving the diurnal shape is actually applied on
        the MPAS branch, not silently passed through).
    """
    from types import SimpleNamespace
    from legoesm.ocean.coupler.omip2_applicator import dm2dc_sw_factor

    # Paired 1-D cell centres spanning latitudes/longitudes (radians), the
    # SAME attribute names the run driver reads for the MPAS grid.
    lat_deg = np.array([-55.0, -30.0, -5.0, 5.0, 20.0, 45.0, 60.0, 0.0])
    lon_deg = np.array([0.0, 40.0, 90.0, 140.0, 200.0, 250.0, 300.0, 350.0])
    grid = SimpleNamespace(
        latCell=np.deg2rad(lat_deg), lonCell=np.deg2rad(lon_deg))
    nCells = lat_deg.shape[0]

    day, year_len = 80, 365.0
    n_win = 24
    dt_frac = 1.0 / n_win

    # Full-day window-weighted mean of the factor (mean-preservation).
    acc = np.zeros(nCells)
    for k in range(n_win):
        t_lo = k * dt_frac
        fac = dm2dc_sw_factor(grid, (day, year_len, t_lo, t_lo + dt_frac))
        assert np.asarray(fac).shape == (nCells,)
        assert np.all(np.isfinite(fac)) and np.all(np.asarray(fac) >= 0.0)
        acc += np.asarray(fac) * dt_frac

    # Daylit cells (day 80 is post-equinox: all these latitudes see the sun).
    np.testing.assert_allclose(acc, 1.0, atol=1e-9)

    # Non-vacuity: a single local-noon window must NOT be identically 1.
    t_noon_lo = 0.5 - 0.5 * dt_frac
    fac_noon = np.asarray(dm2dc_sw_factor(
        grid, (day, year_len, t_noon_lo, t_noon_lo + dt_frac)))
    assert fac_noon.shape == (nCells,)
    assert np.all(np.isfinite(fac_noon))
    assert np.max(np.abs(fac_noon - 1.0)) > 0.5
