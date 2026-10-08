"""Tests for the sub-grid elevation-band snow scheme (``legoesm.land.snow_bands``).

Covers:
- band construction (equal-area Gaussian anomalies, zero mean, dispatch guard)
- per-band precipitation phase (warm cell / cold high bands; mass conservation)
- banded budget step (water-budget closure, snow capping -> permanent ice,
  flat-cell equivalence with the cell-mean ``update_snow``)
- integration in ``step_multilayer_land`` (state wiring, cold-band snow on a warm
  cell brightens the albedo, disabled scheme keeps the legacy pytree)
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land,
)
from legoesm.land.snow_bands import (
    ElevationSnowBandConfig,
    band_albedo,
    band_elevation_anomalies,
    band_net_radiation,
    band_precip_snow,
    band_snow_albedo,
    band_snow_cover,
    step_snow_bands,
)
from legoesm.land.snow_budget import update_snow
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.surface_albedo import LandAlbedoConfig, snow_albedo, snow_cover_fraction


def _band_cfg(std_elev):
    return ElevationSnowBandConfig(
        band_dz=band_elevation_anomalies(jnp.asarray(std_elev)),
    )


def _make_forcing(ncol, T_air=290.0, precip=1e-4, precip_snow=0.0):
    return AtmToSurface(
        sw_down=jnp.full(ncol, 300.0),
        lw_down=jnp.full(ncol, 350.0),
        precip_total=jnp.full(ncol, precip),
        precip_snow=jnp.full(ncol, precip_snow),
        T_lowest=jnp.full(ncol, T_air),
        q_lowest=jnp.full(ncol, 0.008),
        u_lowest=jnp.full(ncol, 5.0),
        v_lowest=jnp.full(ncol, 2.0),
        p_lowest=jnp.full(ncol, 95000.0),
        p_surface=jnp.full(ncol, 100000.0),
        rho_lowest=jnp.full(ncol, 1.2),
        cos_zenith=jnp.full(ncol, 0.5),
        co2_ppmv=jnp.full(ncol, 400.0),
        has_radiation=jnp.ones(ncol),
        has_precipitation=jnp.ones(ncol),
    )


class TestBandConstruction(unittest.TestCase):

    def test_equal_area_anomalies_zero_mean(self):
        dz = band_elevation_anomalies(jnp.array([0.0, 500.0, 2000.0]))
        self.assertEqual(dz.shape, (3, 5))
        # Equal-area bands: the area-weighted mean anomaly is 0 (cell mean preserved)
        npt.assert_allclose(dz.mean(axis=-1), 0.0, atol=1e-10)
        # Flat cell -> all bands at the cell mean
        npt.assert_allclose(dz[0], 0.0)
        # Monotone increasing with band index; scales linearly with sigma
        self.assertTrue(bool(jnp.all(jnp.diff(dz[2]) > 0)))
        npt.assert_allclose(dz[2], 4.0 * dz[1], rtol=1e-12)

    def test_unsupported_band_count_raises(self):
        with self.assertRaises(ValueError):
            band_elevation_anomalies(jnp.array([100.0]), n_bands=4)


class TestBandPhase(unittest.TestCase):

    def test_warm_cell_cold_peaks_snow(self):
        """+2 C rain cell: flat cell all rain; sigma=800 m puts high bands below
        freezing -> snow on the peaks, and band rain+snow == total everywhere."""
        cfg = _band_cfg([0.0, 800.0])
        T_air = jnp.full(2, constants.T_freeze + 2.0)
        precip = jnp.full(2, 1e-4)
        precip_snow = jnp.zeros(2)                            # warm cell: forcing = rain
        snowfall = band_precip_snow(T_air, precip, precip_snow, cfg)
        npt.assert_allclose(snowfall[0], 0.0)                 # flat: all rain
        self.assertGreater(float(snowfall[1, -1]), 0.0)       # top band snows
        npt.assert_allclose(snowfall[1, :3], 0.0)             # low/mid bands rain
        # Phase repartition conserves mass per band by construction
        self.assertTrue(bool(jnp.all(snowfall <= precip[:, None] + 1e-30)))

    def test_flat_cell_preserves_forcing_snow_fraction(self):
        """No relief -> the band snowfall EXACTLY reproduces the forcing precip_snow,
        whatever convention set it (mixed-phase, smooth ramp) -- NOT a re-derived hard
        threshold.  Codex finding: a hard-threshold band phase would corrupt flat cells
        in the coupled model where the atmosphere pre-partitions phase."""
        cfg = _band_cfg([0.0, 0.0, 0.0])
        precip = jnp.array([1e-4, 1e-4, 1e-4])
        # (warm cell but 30% snow), (cold cell but 40% rain), (all snow) -- none of
        # these equals precip_total * 1[T<T_freeze], so a threshold rule would differ.
        T_air = jnp.array([constants.T_freeze + 3.0, constants.T_freeze - 3.0,
                           constants.T_freeze - 1.0])
        precip_snow = jnp.array([0.3e-4, 0.6e-4, 1e-4])
        snowfall = band_precip_snow(T_air, precip, precip_snow, cfg)
        # Every band reproduces the cell forcing snow rate exactly (identity at dz=0)
        npt.assert_allclose(snowfall, jnp.broadcast_to(precip_snow[:, None], snowfall.shape),
                            rtol=1e-12)

    def test_relief_shift_conserves_total_and_reduces_to_forcing(self):
        """Relief shifts the snow fraction by the band-vs-cell freezing indicator and
        conserves total precip per band (rain = total - snow)."""
        cfg = _band_cfg([1000.0])
        T_air = jnp.array([constants.T_freeze + 1.0])         # warm cell, 20% snow
        precip = jnp.array([2e-4]); precip_snow = jnp.array([0.4e-4])
        snowfall = band_precip_snow(T_air, precip, precip_snow, cfg)
        self.assertGreaterEqual(float(snowfall[0, -1]), float(snowfall[0, 0]))  # colder->more snow
        self.assertTrue(bool(jnp.all(snowfall >= 0.0)))
        self.assertTrue(bool(jnp.all(snowfall <= precip[:, None] + 1e-30)))
        # cold top band (T-1.4K below cell, below freezing) -> converts rain to snow
        npt.assert_allclose(snowfall[0, -1], precip[0])


class TestBandStep(unittest.TestCase):

    def test_water_budget_closes(self):
        """Per-cell frozen-store closure: d(SWE + ice) = snowfall - snow_melt
        - ice_melt - ice_runoff (in - out - d storage = 0)."""
        cfg = _band_cfg([0.0, 600.0, 1500.0])
        ncol, dt = 3, 1800.0
        swe0 = jnp.array([[0.0] * 5, [30.0] * 5, [900.0, 950.0, 1200.0, 1500.0, 2000.0]])
        ice0 = jnp.array([[0.0] * 5, [0.0] * 5, [50.0, 0.0, 500.0, 8000.0, 10000.0]])
        age0 = jnp.full((ncol, 5), 86400.0)
        T_sfc = jnp.array([constants.T_freeze + 1.0] * ncol)
        snowfall = band_precip_snow(
            jnp.array([constants.T_freeze - 1.0] * ncol), jnp.full(ncol, 5e-4),
            jnp.full(ncol, 5e-4), cfg,
        )
        out = step_snow_bands(
            swe0, age0, ice0, T_sfc, snowfall, dt,
            Q_net=jnp.full(ncol, 50.0), cfg=cfg, snow_age_activation_K=0.0
        )
        d_store = (out.swe_total + out.ice_total) - (swe0.mean(-1) + ice0.mean(-1))
        source = (snowfall.mean(-1) * dt - out.snow_melt - out.ice_melt
                  - out.ice_runoff * dt)
        npt.assert_allclose(d_store, source, atol=1e-8)
        self.assertTrue(bool(jnp.all(out.swe_bands >= 0.0)))
        self.assertTrue(bool(jnp.all(out.ice_bands >= 0.0)))
        self.assertTrue(bool(jnp.all(out.ice_bands <= cfg.swe_cap + 1e-6)))

    def test_snow_firnifies_to_ice_not_instant_runoff(self):
        """Seasonal snow above the snow cap compacts into the ice reservoir (delayed
        glacier storage), NOT same-step runoff -- codex [high]: winter accumulation must
        build ice, not discharge instantly."""
        cfg = _band_cfg([1000.0])
        swe0 = jnp.full((1, 5), cfg.swe_snow_cap)               # snow at the cap
        ice0 = jnp.zeros((1, 5))
        snowfall = jnp.full((1, 5), 1e-3)                       # 3.6 kg over the step
        dt = 3600.0
        out = step_snow_bands(
            swe0, jnp.zeros((1, 5)), ice0, jnp.array([constants.T_freeze - 20.0]),
            snowfall, dt, Q_net=jnp.zeros(1), cfg=cfg, snow_age_activation_K=0.0
        )
        # Seasonal snow stays capped; the excess went to ICE, not runoff (a decadal
        # ice/tau fraction ~3.6e-6 discharges this step, hence the loose rtol).
        npt.assert_allclose(out.swe_bands, cfg.swe_snow_cap, rtol=1e-9)
        npt.assert_allclose(out.ice_bands, 1e-3 * dt, rtol=1e-4)  # ~all excess -> ice
        # Discharge this step is only ice/tau (tiny, ~decadal) -- essentially no runoff
        self.assertLess(float(out.ice_runoff[0]) * dt, 1e-3)     # << the 3.6 kg input
        self.assertEqual(float(out.ice_melt[0]), 0.0)            # no melt at -20 C

    def test_ice_ceiling_sheds_frozen_runoff(self):
        """Ice above the reservoir ceiling (swe_cap) is shed as frozen runoff, bounding
        perennial storage."""
        cfg = _band_cfg([1000.0])
        ice0 = jnp.full((1, 5), cfg.swe_cap)                    # already at the ceiling
        swe0 = jnp.full((1, 5), 2.0 * cfg.swe_snow_cap)         # snow that will firnify
        dt = 3600.0
        out = step_snow_bands(
            swe0, jnp.zeros((1, 5)), ice0, jnp.array([constants.T_freeze - 20.0]),
            jnp.zeros((1, 5)), dt, Q_net=jnp.zeros(1), cfg=cfg, snow_age_activation_K=0.0
        )
        # Ice pinned at (just below) the ceiling; the firnified snow + discharge leaves
        self.assertLessEqual(float(out.ice_bands.max()), cfg.swe_cap + 1e-6)
        self.assertGreater(float(out.ice_runoff[0]), 0.0)       # frozen discharge > 0

    def test_ablation_melts_exposed_ice(self):
        """When the seasonal snow is gone, leftover melt energy ablates exposed ice
        (ice_melt > 0) which leaves as runoff -- the ablation zone."""
        cfg = _band_cfg([0.0])
        swe0 = jnp.zeros((1, 5))                                # no seasonal snow
        ice0 = jnp.full((1, 5), 500.0)                          # bare glacier ice
        dt = 3600.0
        out = step_snow_bands(
            swe0, jnp.zeros((1, 5)), ice0, jnp.array([constants.T_freeze + 5.0]),
            jnp.zeros((1, 5)), dt, Q_net=jnp.full(1, 150.0), cfg=cfg, snow_age_activation_K=0.0
        )
        self.assertEqual(float(out.snow_melt[0]), 0.0)          # no snow to melt
        self.assertGreater(float(out.ice_melt[0]), 0.0)         # ice ablates
        # Energy-limited: ice_melt == Q_net*dt/L_f (well below the 500 kg available)
        npt.assert_allclose(out.ice_melt, 150.0 * dt / constants.L_f, rtol=1e-9)
        self.assertLess(float(out.ice_bands.max()), 500.0)      # reservoir shrank

    def test_fresh_snow_brightens_glaciated_band(self):
        """Fresh snow on a glaciated band strongly rejuvenates that band's snow age
        (bright fresh snow) even though the deep ice persists -- codex finding 2:
        capped/perennial ice must still brighten when it snows (the separate ice
        reservoir carries the perennial darkening, not a frozen age clock).  With
        mass-weighted grain-age mixing, a heavy fresh dump onto a thin seasonal layer
        drops the age far below the aged firn value (does not need an exact-0 reset)."""
        cfg = _band_cfg([1000.0])
        age0_val = 60.0 * 86400.0
        swe0 = jnp.full((1, 5), 2.0)                            # thin seasonal snow on ice
        ice0 = jnp.full((1, 5), 5000.0)                         # thick perennial ice
        age0 = jnp.full((1, 5), age0_val)                      # old firn
        snowfall = jnp.full((1, 5), 5e-3)                      # 18 kg/m2 fresh over the step
        out = step_snow_bands(
            swe0, age0, ice0, jnp.array([constants.T_freeze - 20.0]), snowfall, 3600.0,
            Q_net=jnp.zeros(1), cfg=cfg, snow_age_activation_K=0.0
        )
        # Heavy fresh (18 kg/m2) >> old (2 kg/m2) -> age drops to ~0.1*(age+dt).
        self.assertLess(float(out.snow_age_bands.max()), 0.2 * age0_val)  # snowed -> bright
        self.assertGreater(float(out.ice_bands.min()), 5000.0 - 1.0)  # ice persists

    def test_cold_bands_hold_snow_warm_bands_melt(self):
        """Energy-limited melt gates on the band-downscaled T_sfc: bands above the
        freezing height keep their snow while low bands melt."""
        cfg = _band_cfg([1200.0])
        swe0 = jnp.full((1, 5), 20.0)
        # Cell-mean surface +3 C: with sigma=1200 m the top two bands (+0.53, +1.40
        # sigma) sit 3.8/10 K colder -> below freezing -> no melt there.
        out = step_snow_bands(
            swe0, jnp.zeros((1, 5)), jnp.zeros((1, 5)), jnp.array([constants.T_freeze + 3.0]),
            jnp.zeros((1, 5)), 1800.0, Q_net=jnp.full(1, 200.0), cfg=cfg, snow_age_activation_K=0.0
        )
        self.assertLess(float(out.swe_bands[0, 0]), 20.0)      # lowest band melted
        npt.assert_allclose(out.swe_bands[0, 3:], 20.0)        # cold bands intact

    def test_flat_cell_equivalence_with_update_snow(self):
        """sigma=0, no capping -> the banded budget collapses to the cell-mean
        ``update_snow`` (ice reservoir stays empty)."""
        cfg = _band_cfg([0.0, 0.0])
        ncol, dt = 2, 900.0
        swe0_cell = jnp.array([12.0, 0.5])
        swe0 = jnp.broadcast_to(swe0_cell[:, None], (ncol, 5))
        age0 = jnp.array([3600.0, 0.0])
        T_sfc = jnp.array([constants.T_freeze + 2.0, constants.T_freeze - 5.0])
        precip_snow = jnp.array([0.0, 3e-4])
        Q_net = jnp.array([120.0, -30.0])
        out = step_snow_bands(
            swe0, jnp.broadcast_to(age0[:, None], (ncol, 5)), jnp.zeros((ncol, 5)),
            T_sfc, jnp.broadcast_to(precip_snow[:, None], (ncol, 5)),
            dt, Q_net=Q_net, cfg=cfg, snow_age_activation_K=0.0
        )
        ref_snow, ref_age, ref_melt = update_snow(
            swe0_cell, age0, T_sfc, precip_snow, dt, Q_net=Q_net, snow_age_activation_K=0.0
        )
        npt.assert_allclose(out.swe_total, ref_snow, rtol=1e-12)
        npt.assert_allclose(out.ice_total, 0.0, atol=1e-12)    # no firnification
        # SWE-weighted aggregate age == cell age when bands are uniform
        npt.assert_allclose(out.snow_age, ref_age, rtol=1e-12)
        npt.assert_allclose(out.snow_melt, ref_melt, rtol=1e-12)
        # Per-band age is returned and matches the cell reference band-wise
        npt.assert_allclose(out.snow_age_bands, jnp.broadcast_to(ref_age[:, None], (ncol, 5)),
                            rtol=1e-12)

    def test_float64_band_dz_does_not_promote_step(self):
        """step_snow_bands with a float64 band_dz and a float32 state keeps the returned
        SWE/ice/age float32 (coupled float32 carry-dtype hazard, codex #3)."""
        cfg = ElevationSnowBandConfig(
            band_dz=band_elevation_anomalies(jnp.asarray([1000.0], dtype=jnp.float64)))
        f32 = jnp.float32
        out = step_snow_bands(
            jnp.zeros((1, 5), f32), jnp.zeros((1, 5), f32), jnp.zeros((1, 5), f32),
            jnp.array([270.0], f32), jnp.full((1, 5), 1e-4, f32), f32(600.0),
            Q_net=jnp.full((1, 5), 20.0, f32), cfg=cfg, snow_age_activation_K=0.0
        )
        self.assertEqual(out.swe_bands.dtype, f32)
        self.assertEqual(out.ice_bands.dtype, f32)
        self.assertEqual(out.snow_age_bands.dtype, f32)

    def test_rain_on_snow_refreezes(self):
        """Gap 6 cold content: rain on a sub-freezing snow band refreezes into SWE
        (releasing L_f), not running off; a warm/snow-free band does not refreeze."""
        cfg = _band_cfg([0.0])
        swe0 = jnp.full((1, 5), 20.0)
        rain = jnp.full((1, 5), 1e-4)                          # 0.36 kg/m2 over the step
        out = step_snow_bands(
            swe0, jnp.zeros((1, 5)), jnp.zeros((1, 5)),
            jnp.array([constants.T_freeze - 5.0]), jnp.zeros((1, 5)), 3600.0,
            Q_net=jnp.zeros(1), cfg=cfg, precip_rain_bands=rain, snow_age_activation_K=0.0)
        npt.assert_allclose(out.refreeze, 1e-4 * 3600.0, rtol=1e-6)   # all rain refroze
        self.assertGreater(float(out.swe_total[0]), 20.0)            # SWE grew by the rain
        out_warm = step_snow_bands(
            jnp.zeros((1, 5)), jnp.zeros((1, 5)), jnp.zeros((1, 5)),
            jnp.array([constants.T_freeze + 5.0]), jnp.zeros((1, 5)), 3600.0,
            Q_net=jnp.zeros(1), cfg=cfg, precip_rain_bands=rain, snow_age_activation_K=0.0)
        self.assertEqual(float(out_warm.refreeze[0]), 0.0)          # no snow -> no refreeze

    def test_refreeze_bounded_by_cold_content(self):
        """A thin, barely sub-freezing band cannot refreeze more rain than its cold
        content c_pi*swe*(Tf - T_band)/L_f allows; the residual rain stays liquid
        (runs off/infiltrates) instead of releasing unbounded latent heat."""
        cfg = _band_cfg([0.0])
        swe0 = jnp.full((1, 5), 0.5)                        # thin cold band
        dT = 1.0                                            # 1 K below freezing
        heavy_rain = jnp.full((1, 5), 1e-2)                # 36 kg/m2 over the step
        out = step_snow_bands(
            swe0, jnp.zeros((1, 5)), jnp.zeros((1, 5)),
            jnp.array([constants.T_freeze - dT]), jnp.zeros((1, 5)), 3600.0,
            Q_net=jnp.zeros(1), cfg=cfg, precip_rain_bands=heavy_rain, snow_age_activation_K=0.0)
        # Cold-content cap [kg/m2] = c_pi*swe*dT/L_f -- far below the 36 kg/m2 rain.
        cap = float(constants.c_pi * 0.5 * dT / constants.L_f)
        rain_mass = float(heavy_rain[0, 0]) * 3600.0
        self.assertLess(cap, rain_mass)                    # cap actually binds
        npt.assert_allclose(out.refreeze, cap, rtol=1e-6)  # capped at cold content
        # SWE grew by exactly the (capped) refrozen mass, not by the whole rain flux.
        npt.assert_allclose(out.swe_total, 0.5 + cap, rtol=1e-6)

    def test_blowing_snow_sublimation(self):
        """Gap 5: wind above the mobilisation threshold sublimes SWE (opt-in rate)."""
        cfg = ElevationSnowBandConfig(
            band_dz=band_elevation_anomalies(jnp.asarray([0.0])),
            blow_snow_subl_rate=1e-6, blow_snow_wind_thresh_ms=5.0)
        swe0 = jnp.full((1, 5), 100.0)
        out = step_snow_bands(
            swe0, jnp.zeros((1, 5)), jnp.zeros((1, 5)),
            jnp.array([constants.T_freeze - 10.0]), jnp.zeros((1, 5)), 3600.0,
            Q_net=jnp.zeros(1), cfg=cfg, wind=jnp.array([15.0]), snow_age_activation_K=0.0)      # 10 m/s over thresh
        self.assertGreater(float(out.blow_subl[0]), 0.0)
        self.assertLess(float(out.swe_total[0]), 100.0)             # SWE reduced
        out_calm = step_snow_bands(
            swe0, jnp.zeros((1, 5)), jnp.zeros((1, 5)),
            jnp.array([constants.T_freeze - 10.0]), jnp.zeros((1, 5)), 3600.0,
            Q_net=jnp.zeros(1), cfg=cfg, wind=jnp.array([3.0]), snow_age_activation_K=0.0)       # below threshold
        self.assertEqual(float(out_calm.blow_subl[0]), 0.0)

    def test_per_band_age_fresh_and_perennial_coexist(self):
        """Fresh snowfall on the high bands REJUVENATES only those bands' age while the
        snow-free bands (no fresh snow) keep aging -- perennial firn and fresh snow
        coexist radiatively (codex finding: one cell clock cannot represent this).  With
        mass-weighted grain-age mixing the fresh bands are younger than the aged clock
        and strictly younger than the no-snow bands (which age up by dt)."""
        cfg = _band_cfg([1500.0])
        age0_val = 30.0 * 86400.0
        swe0 = jnp.full((1, 5), 10.0)
        age0 = jnp.full((1, 5), age0_val)                     # 30-day-old pack
        # Cold cell: the high (colder) bands stay below freezing and get fresh snow; the
        # low warm bands get rain (no fresh snow).  18 kg/m2 fresh over the step.
        snowfall = band_precip_snow(
            jnp.array([constants.T_freeze - 0.5]), jnp.array([5e-3]),
            jnp.array([5e-3]), cfg,
        )
        out = step_snow_bands(
            swe0, age0, jnp.zeros((1, 5)), jnp.array([constants.T_freeze - 5.0]),
            snowfall, 3600.0, Q_net=jnp.zeros(1), cfg=cfg, snow_age_activation_K=0.0
        )
        got_snow = snowfall[0] > 1e-10
        self.assertTrue(bool(jnp.any(got_snow)) and bool(jnp.any(~got_snow)))
        # Bands that received fresh snow are rejuvenated (younger than the aged clock);
        # bands that did not keep aging past the initial age.
        self.assertTrue(bool(jnp.all(out.snow_age_bands[0][got_snow] < age0_val)))
        self.assertTrue(bool(jnp.all(out.snow_age_bands[0][~got_snow] > age0_val)))
        # Fresh bands are strictly younger than the still-aging snow-free bands.
        self.assertLess(float(out.snow_age_bands[0][got_snow].max()),
                        float(out.snow_age_bands[0][~got_snow].min()))


class TestBandAlbedo(unittest.TestCase):

    def _acfg(self):
        return LandAlbedoConfig()

    def test_mixed_age_contribution_between_fresh_and_old(self):
        """band_snow_albedo blends each band's OWN age-decayed snow albedo: a cell with
        one fresh band and one perennial-firn band lands BETWEEN all-fresh and all-old,
        and differs from collapsing to a single averaged age (codex finding 2)."""
        acfg = self._acfg()
        cf = lambda s: snow_cover_fraction(s, acfg)
        af = lambda a: snow_albedo(a, acfg)
        swe = jnp.full((1, 2), 200.0)                        # both bands fully covered
        fresh, old = 0.0, 60.0 * 86400.0
        _, c_fresh = band_snow_albedo(swe, jnp.full((1, 2), fresh), cf, af)
        _, c_old = band_snow_albedo(swe, jnp.full((1, 2), old), cf, af)
        _, c_mix = band_snow_albedo(swe, jnp.array([[fresh, old]]), cf, af)
        self.assertLess(float(c_old[0]), float(c_mix[0]))    # firn band darkens the cell
        self.assertLess(float(c_mix[0]), float(c_fresh[0]))  # fresh band brightens it
        # A single averaged age differs from the true per-band blend because the snow
        # albedo is nonlinear (exp decay) in age: mean-age contribution != mixed
        _, c_meanage = band_snow_albedo(swe, jnp.full((1, 2), 0.5 * old), cf, af)
        self.assertNotAlmostEqual(float(c_meanage[0]), float(c_mix[0]), places=4)

    def test_uniform_bands_reduce_to_cell_mean(self):
        """Uniform bands (flat cell) -> snow_contrib == alpha_snow(age) * f_snow."""
        acfg = self._acfg()
        cf = lambda s: snow_cover_fraction(s, acfg)
        af = lambda a: snow_albedo(a, acfg)
        swe = jnp.full((1, 5), 15.0); age = jnp.full((1, 5), 5.0 * 86400.0)
        f, contrib = band_snow_albedo(swe, age, cf, af)
        npt.assert_allclose(f, cf(swe[:, 0]), rtol=1e-12)
        npt.assert_allclose(contrib, af(age[:, 0]) * cf(swe[:, 0]), rtol=1e-12)

    def test_band_albedo_mean_matches_aggregate_blend(self):
        """band_albedo (per band) averaged over bands equals the aggregate cell blend
        base*(1 - f_snow) + snow_contrib from band_snow_albedo (same physics, per band)."""
        acfg = self._acfg()
        cf = lambda s: snow_cover_fraction(s, acfg)
        af = lambda a: snow_albedo(a, acfg)
        base = jnp.array([0.22])
        swe = jnp.array([[0.0, 5.0, 40.0, 200.0, 1000.0]])
        age = jnp.array([[0.0, 1.0, 5.0, 30.0, 120.0]]) * 86400.0
        ab = band_albedo(swe, age, base, cf, af)                # (1, 5)
        f, contrib = band_snow_albedo(swe, age, cf, af)
        npt.assert_allclose(jnp.mean(ab, -1), base * (1.0 - f) + contrib, rtol=1e-12)

    def test_snow_albedo_zenith_brightens_at_low_sun(self):
        """BATS/Dickinson zenith brightening (gap 3): snow albedo rises as the sun drops
        toward the horizon; overhead sun (mu=1) reduces to the legacy diffuse albedo."""
        acfg = self._acfg()
        age = jnp.array([2.0 * 86400.0])
        a_overhead = snow_albedo(age, acfg, cos_zenith=jnp.array([1.0]))
        a_grazing = snow_albedo(age, acfg, cos_zenith=jnp.array([0.1]))
        a_legacy = snow_albedo(age, acfg)                       # no zenith -> diffuse
        npt.assert_allclose(a_overhead, a_legacy, rtol=1e-12)   # mu=1 -> f=0 -> diffuse
        self.assertGreater(float(a_grazing[0]), float(a_overhead[0]))  # brighter low sun
        self.assertLessEqual(float(a_grazing[0]), 1.0)          # bounded

    def test_exposed_glacier_ice_darkens_snow_free_band(self):
        """A snow-free band over the firn/ice reservoir shows the dark ablation-ice
        albedo (gap 4), not the bright soil/veg base; a snow-covered glaciated band
        keeps its bright snow albedo (accumulation zone)."""
        acfg = self._acfg()
        cf = lambda s: snow_cover_fraction(s, acfg)
        af = lambda a: snow_albedo(a, acfg)
        cfg = ElevationSnowBandConfig(band_dz=band_elevation_anomalies(jnp.asarray([0.0])))
        base = jnp.array([0.25])                                # bright bare soil/veg
        # snow-free (swe=0) glaciated (ice above the exposure threshold) -> dark ice
        a_ice = band_albedo(jnp.zeros((1, 1)), jnp.zeros((1, 1)), base, cf, af,
                            ice_bands=jnp.full((1, 1), 500.0), cfg=cfg)
        npt.assert_allclose(a_ice, cfg.alpha_glacier_ice, atol=1e-6)
        # snow-free, ice-free -> the soil/veg base (no darkening)
        a_soil = band_albedo(jnp.zeros((1, 1)), jnp.zeros((1, 1)), base, cf, af,
                             ice_bands=jnp.zeros((1, 1)), cfg=cfg)
        npt.assert_allclose(a_soil, 0.25, atol=1e-6)
        # snow-covered glacier (deep fresh snow) -> bright snow, ice hidden
        a_snow = band_albedo(jnp.full((1, 1), 500.0), jnp.zeros((1, 1)), base, cf, af,
                             ice_bands=jnp.full((1, 1), 500.0), cfg=cfg)
        self.assertGreater(float(a_snow[0, 0]), 0.6)            # bright fresh snow


class TestBandRadiation(unittest.TestCase):
    """Banded surface radiation balance (gap 1+2: per-band SEB + elevation lapse)."""

    def _cfg(self, std_elev, **kw):
        return ElevationSnowBandConfig(
            band_dz=band_elevation_anomalies(jnp.asarray([std_elev])), **kw)

    def test_downflux_conserves_cell_mean(self):
        """Zero-mean band anomalies -> the elevation-lapsed SW/LW DOWN aggregate to
        the cell mean exactly (pure sub-grid redistribution, no spurious energy)."""
        cfg = self._cfg(1200.0)                                  # strong relief
        T = jnp.array([275.0]); a = jnp.full((1, 5), 0.4)
        sw, lw, emis = jnp.array([300.0]), jnp.array([250.0]), 0.97
        dz = cfg.band_dz
        sw_b = sw[:, None] * (1.0 + cfg.sw_elev_grad_per_m * dz)
        lw_b = lw[:, None] - cfg.lw_elev_lapse_W_m2_per_m * dz
        npt.assert_allclose(jnp.mean(sw_b, -1), sw, rtol=1e-12)
        npt.assert_allclose(jnp.mean(lw_b, -1), lw, rtol=1e-12)
        # And band_net_radiation's SW-flux-weighted albedo * incident recovers reflected
        r = band_net_radiation(T, a, sw, lw, emis, cfg)
        npt.assert_allclose(r.alpha_eff, 0.4, rtol=1e-12)        # uniform albedo -> mean

    def test_flat_bands_match_surface_radiation_fluxes(self):
        """At zero relief the banded balance reduces to the shared cell-mean formula."""
        cfg = self._cfg(0.0)
        T = jnp.array([268.0]); a = jnp.full((1, 5), 0.6)
        sw, lw, emis = jnp.array([180.0]), jnp.array([220.0]), 0.98
        r = band_net_radiation(T, a, sw, lw, emis, cfg)
        sw_net, lw_net, _ = surface_radiation_fluxes(sw, lw, T, jnp.array([0.6]), emis)
        npt.assert_allclose(r.sw_net_agg, sw_net, rtol=1e-12)
        npt.assert_allclose(r.lw_net_agg, lw_net, rtol=1e-12)
        npt.assert_allclose(r.alpha_eff, 0.6, rtol=1e-12)

    def test_elevation_covariance_cools_surface(self):
        """Bright high bands intercepting the enhanced high-altitude sun absorb LESS
        SW than a single cell-mean albedo would (the covariance the cell mean loses),
        and the SW-flux-weighted albedo exceeds the plain band mean."""
        cfg = self._cfg(1500.0)
        T = jnp.array([265.0])
        # albedo rises with elevation (bright fresh snow on the cold high bands)
        a = jnp.array([[0.20, 0.35, 0.55, 0.72, 0.82]])
        sw, lw, emis = jnp.array([320.0]), jnp.array([230.0]), 0.97
        r = band_net_radiation(T, a, sw, lw, emis, cfg)
        naive_absorb = sw * (1.0 - jnp.mean(a))                  # cell-mean albedo
        self.assertLess(float(r.sw_net_agg[0]), float(naive_absorb[0]))
        self.assertGreater(float(r.alpha_eff[0]), float(jnp.mean(a)))

    def test_shortwave_conserves_reflected_plus_absorbed(self):
        """SW-flux-weighted alpha_eff makes reflected + absorbed == incident (energy
        conserving) even with strong relief + the elevation SW lapse — the un-clamped
        zero-mean lapse must not create or destroy shortwave (codex #2)."""
        cfg = self._cfg(1500.0)
        T = jnp.array([265.0]); a = jnp.array([[0.20, 0.35, 0.55, 0.72, 0.82]])
        sw, lw, emis = jnp.array([320.0]), jnp.array([230.0]), 0.97
        r = band_net_radiation(T, a, sw, lw, emis, cfg)
        npt.assert_allclose(r.alpha_eff * sw + r.sw_net_agg, sw, rtol=1e-10)

    def test_lapsed_downflux_mean_unchanged_by_relief(self):
        """The un-clamped elevation lapse conserves the cell-mean SW/LW DOWN exactly at
        large relief (zero-mean band anomalies), so the aggregate forcing is preserved."""
        cfg = self._cfg(2500.0)                                  # extreme relief
        dz = cfg.band_dz
        sw, lw = jnp.array([250.0]), jnp.array([180.0])
        sw_b = sw[:, None] * (1.0 + cfg.sw_elev_grad_per_m * dz)
        lw_b = lw[:, None] - cfg.lw_elev_lapse_W_m2_per_m * dz
        npt.assert_allclose(jnp.mean(sw_b, -1), sw, rtol=1e-12)
        npt.assert_allclose(jnp.mean(lw_b, -1), lw, rtol=1e-12)

    def test_high_band_colder_emits_less_lw(self):
        """A high band (positive dz) is lapse-cooled -> lower skin T -> less up-welling
        LW; the aggregate up-welling LW is below the cell-skin-T Stefan-Boltzmann value
        offset by the convex T^4 average (banded emission != emission at the mean T)."""
        cfg = self._cfg(1000.0)
        T = jnp.array([270.0]); a = jnp.full((1, 5), 0.5)
        sw, lw, emis = jnp.array([0.0]), jnp.array([240.0]), 1.0
        r = band_net_radiation(T, a, sw, lw, emis, cfg)
        T_skin = T[:, None] - cfg.lapse_rate_K_m * cfg.band_dz
        npt.assert_allclose(
            r.lw_up_agg, jnp.mean(emis * constants.sigma_sb * T_skin ** 4, -1), rtol=1e-12)
        # night (sw=0): alpha_eff falls back to the plain band mean, not 0/0
        npt.assert_allclose(r.alpha_eff, 0.5, rtol=1e-12)

    def test_extreme_relief_no_negative_downflux(self):
        """Extreme relief + max tuned lapse must NOT drive any band's SW/LW DOWN negative
        (the per-column slope is capped) while the cell-mean is still conserved exactly
        (codex #1: the un-clamped lapse could otherwise flip the flux sign)."""
        cfg = ElevationSnowBandConfig(
            band_dz=band_elevation_anomalies(jnp.asarray([2500.0])),  # ~3500 m top band
            lw_elev_lapse_W_m2_per_m=6.0e-2, sw_elev_grad_per_m=1.2e-4)  # bound maxima
        dz = cfg.band_dz
        sw, lw = jnp.array([250.0]), jnp.array([160.0])                 # low LW down
        dz_hi = jnp.max(dz, -1, keepdims=True); dz_lo = jnp.min(dz, -1, keepdims=True)
        lw_slope = jnp.minimum(cfg.lw_elev_lapse_W_m2_per_m,
                               jnp.maximum(lw[:, None], 0.0) / jnp.maximum(dz_hi, 1e-6))
        sw_slope = jnp.minimum(cfg.sw_elev_grad_per_m, 1.0 / jnp.maximum(-dz_lo, 1e-6))
        sw_b = sw[:, None] * (1.0 + sw_slope * dz); lw_b = lw[:, None] - lw_slope * dz
        self.assertTrue(bool(jnp.all(lw_b >= -1e-9)))          # no negative LW down
        self.assertTrue(bool(jnp.all(sw_b >= -1e-9)))          # no negative SW down
        npt.assert_allclose(jnp.mean(sw_b, -1), sw, rtol=1e-12)  # mean still conserved
        npt.assert_allclose(jnp.mean(lw_b, -1), lw, rtol=1e-12)
        # And band_net_radiation itself stays finite / conservative at these extremes
        r = band_net_radiation(jnp.array([255.0]), jnp.full((1, 5), 0.6), sw, lw, 0.97, cfg)
        self.assertTrue(bool(jnp.all(jnp.isfinite(r.Rn_bands))))
        npt.assert_allclose(r.alpha_eff * sw + r.sw_net_agg, sw, rtol=1e-10)

    def test_sky_view_reduces_valley_lw_loss(self):
        """Gap 5: sky_view_min < 1 scales down the NET LW of the shielded (valley) bands
        (terrain shielding -> less radiative cooling); the ridge band is unchanged."""
        cfg_open = ElevationSnowBandConfig(
            band_dz=band_elevation_anomalies(jnp.asarray([1000.0])))
        cfg_shield = cfg_open._replace(sky_view_min=0.5)
        T = jnp.array([260.0]); a = jnp.full((1, 5), 0.6)
        sw, lw, emis = jnp.array([0.0]), jnp.array([200.0]), 0.97   # night: LW only
        r0 = band_net_radiation(T, a, sw, lw, emis, cfg_open)
        r1 = band_net_radiation(T, a, sw, lw, emis, cfg_shield)
        # valley band (lowest dz, index 0): shielded -> less negative net radiation
        self.assertGreater(float(r1.Rn_bands[0, 0]), float(r0.Rn_bands[0, 0]))
        # ridge band (highest dz, index -1): V == 1, unchanged
        npt.assert_allclose(r1.Rn_bands[0, -1], r0.Rn_bands[0, -1], rtol=1e-9)

    def test_float64_band_dz_does_not_promote_output(self):
        """A float64 CLM band_dz with a float32 state must NOT promote band_net_radiation
        outputs to float64 (coupled float32 lax.scan carry-dtype hazard, codex #3)."""
        cfg = ElevationSnowBandConfig(
            band_dz=band_elevation_anomalies(jnp.asarray([1000.0], dtype=jnp.float64)))
        T = jnp.array([268.0], jnp.float32); a = jnp.full((1, 5), 0.6, jnp.float32)
        r = band_net_radiation(T, a, jnp.array([200.], jnp.float32),
                               jnp.array([250.], jnp.float32), jnp.float32(0.97), cfg)
        self.assertEqual(r.Rn_bands.dtype, jnp.float32)
        self.assertEqual(r.sw_net_agg.dtype, jnp.float32)
        self.assertEqual(r.alpha_eff.dtype, jnp.float32)
        self.assertEqual(r.lw_up_agg.dtype, jnp.float32)


class TestMultilayerIntegration(unittest.TestCase):

    def _config(self, std_elev):
        return MultiLayerLandConfig(
            snow_albedo_feedback=True,
            elev_bands=(_band_cfg(std_elev) if std_elev is not None else None),
        )

    def test_disabled_keeps_legacy_pytree(self):
        config = self._config(None)
        state = init_multilayer_land_state(2, config)
        self.assertIsNone(state.snow_bands)
        self.assertIsNone(state.snow_age_bands)
        new_state, _, _ = step_multilayer_land(
            state, _make_forcing(2), config, U_min=1.0, dt=600.0,
            lat=jnp.full(2, 0.7),
        )
        self.assertIsNone(new_state.snow_bands)
        self.assertIsNone(new_state.snow_age_bands)

    def test_missing_band_state_raises(self):
        config = self._config([0.0, 500.0])
        legacy = init_multilayer_land_state(2, self._config(None))
        with self.assertRaises(ValueError):
            step_multilayer_land(
                legacy, _make_forcing(2), config, U_min=1.0, dt=600.0,
                lat=jnp.full(2, 0.7),
            )

    def test_elev_bands_with_canopy_raises(self):
        """Elevation bands + the two-leaf canopy surface scheme are rejected: the banded
        radiation would override the canopy's radiative closure (codex #5)."""
        from legoesm.land.surface_scheme import TwoLeafCanopyConfig
        config = self._config([500.0])._replace(surface_scheme=TwoLeafCanopyConfig())
        state = init_multilayer_land_state(1, config)
        with self.assertRaises(ValueError):
            step_multilayer_land(state, _make_forcing(1), config, U_min=1.0, dt=600.0,
                                 lat=jnp.full(1, 0.7))

    def test_warm_cell_cold_peaks_accumulates_and_brightens(self):
        """+2 C rain-only cell: the banded mountain cell accumulates snow on its
        cold high bands and ends up brighter than the flat cell."""
        std = [0.0, 1500.0]
        config = self._config(std)
        state = init_multilayer_land_state(2, config)
        forcing = _make_forcing(2, T_air=constants.T_freeze + 2.0, precip=5e-4)
        lat = jnp.full(2, 0.7)
        resp = None
        for _ in range(6):
            state, resp, _ = step_multilayer_land(
                state, forcing, config, U_min=1.0, dt=1800.0, lat=lat,
            )
        self.assertEqual(state.snow_bands.shape, (2, 5))
        npt.assert_allclose(state.snow_bands[0], 0.0, atol=1e-12)  # flat: no snow
        self.assertGreater(float(state.snow_bands[1, -1]), 0.0)   # peaks hold snow
        # Aggregate snow_depth mirrors the band mean
        npt.assert_allclose(
            state.snow_depth, state.snow_bands.mean(-1), rtol=1e-12,
        )
        self.assertGreater(float(resp.albedo[1]), float(resp.albedo[0]))

    def test_flat_bands_match_disabled_scheme(self):
        """sigma=0 bands + forcing whose precip_snow equals the T-threshold phase
        reproduce the disabled-scheme trajectory (same snow, same albedo)."""
        T_air = constants.T_freeze - 4.0
        precip = 4e-4
        # Cell-mean threshold phase: below freezing -> all snow
        forcing = _make_forcing(1, T_air=T_air, precip=precip, precip_snow=precip)
        lat = jnp.full(1, 0.9)
        cfg_off = self._config(None)
        cfg_on = self._config([0.0])
        s_off = init_multilayer_land_state(1, cfg_off)
        s_on = init_multilayer_land_state(1, cfg_on)
        for _ in range(4):
            s_off, r_off, _ = step_multilayer_land(
                s_off, forcing, cfg_off, U_min=1.0, dt=1800.0, lat=lat,
            )
            s_on, r_on, _ = step_multilayer_land(
                s_on, forcing, cfg_on, U_min=1.0, dt=1800.0, lat=lat,
            )
        npt.assert_allclose(s_on.snow_depth, s_off.snow_depth, rtol=1e-12)
        npt.assert_allclose(r_on.albedo, r_off.albedo, rtol=1e-12)
        npt.assert_allclose(s_on.T_soil, s_off.T_soil, rtol=1e-12)

    def test_freshwater_budget_ice_discharge(self):
        """End-to-end (gap 4): a glaciated band whose ice reservoir sits at the ceiling
        sheds frozen discharge to runoff WITHOUT infiltrating, and the coupling identity
        freshwater == surface_runoff + subsurface_runoff holds; a thin-ice band sheds
        far less (delayed discharge, not instant cap runoff)."""
        std = [1000.0]
        config = self._config(std)
        cap = config.elev_bands.swe_cap
        snow_cap = config.elev_bands.swe_snow_cap
        base = init_multilayer_land_state(1, config)
        # Cold cell, all-snow forcing so the pack only grows (no melt/ET confound); the
        # snow at the snow cap firnifies into ice each step.
        forcing = _make_forcing(1, T_air=constants.T_freeze - 15.0, precip=6e-4,
                                precip_snow=6e-4)
        lat = jnp.full(1, 1.2)
        glac = base._replace(ice_bands=jnp.full((1, 5), cap),
                             snow_bands=jnp.full((1, 5), snow_cap),
                             snow_depth=jnp.full(1, snow_cap))
        thin = base._replace(ice_bands=jnp.full((1, 5), 100.0),
                             snow_bands=jnp.full((1, 5), snow_cap),
                             snow_depth=jnp.full(1, snow_cap))
        s_glac, r_glac, _ = step_multilayer_land(glac, forcing, config, U_min=1.0,
                                                 dt=1800.0, lat=lat)
        _, r_thin, _ = step_multilayer_land(thin, forcing, config, U_min=1.0,
                                            dt=1800.0, lat=lat)
        # Coupling identity: freshwater to ocean == surface + subsurface runoff
        npt.assert_allclose(
            r_glac.freshwater_flux,
            s_glac.runoff_surface + s_glac.runoff_subsurface, rtol=1e-9,
        )
        # The at-ceiling glacier sheds more freshwater than the thin-ice band
        self.assertGreater(float(r_glac.freshwater_flux[0]),
                           float(r_thin.freshwater_flux[0]))
        # Ice stays bounded by the ceiling (permanent-storage bound)
        self.assertLessEqual(float(s_glac.ice_bands.max()), cap + 1e-6)

    def test_bands_feedback_off_and_lat_none_no_crash(self):
        """Bands with snow_albedo_feedback=False AND lat=None must run (uniform snow-free
        base albedo, no snow blend) instead of crashing in land_vegetation_albedo, and
        keep the band-aggregate consistency (codex #6)."""
        config = self._config([1000.0])._replace(snow_albedo_feedback=False)
        state = init_multilayer_land_state(1, config)
        ns, resp, _ = step_multilayer_land(
            state, _make_forcing(1, T_air=constants.T_freeze - 5.0, precip=4e-4,
                                 precip_snow=4e-4),
            config, U_min=1.0, dt=600.0, lat=None,
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(resp.albedo))))
        npt.assert_allclose(ns.snow_depth, ns.snow_bands.mean(-1), rtol=1e-12)

    def test_deposition_onto_melted_band_conserves_mass(self):
        """Frost/dew deposition that regrows an emptied banded pack must land in the
        band store (mean_k snow_bands == snow_depth), not vanish (codex #4).  Force a
        cold, saturated, still surface with no sun so latent flux is downward (deposition)
        onto a tiny pack that melts out; the band aggregate must stay consistent."""
        config = self._config([800.0])
        state = init_multilayer_land_state(1, config)
        # tiny initial pack on all bands; cold saturated air, no radiation -> deposition
        state = state._replace(snow_bands=jnp.full((1, 5), 0.05),
                               snow_depth=jnp.full(1, 0.05))
        forcing = _make_forcing(1, T_air=constants.T_freeze - 2.0, precip=0.0)._replace(
            sw_down=jnp.zeros(1), q_lowest=jnp.full(1, 0.02), cos_zenith=jnp.zeros(1))
        ns, _, _ = step_multilayer_land(state, forcing, config, U_min=1.0, dt=1800.0,
                                        lat=jnp.full(1, 0.7))
        # Band-aggregate consistency holds through the sublimation/deposition path
        npt.assert_allclose(ns.snow_depth, ns.snow_bands.mean(-1), rtol=1e-9)
        self.assertTrue(bool(jnp.all(ns.snow_bands >= 0.0)))

    def test_differentiable_through_bands(self):
        """grad of a snow/albedo-dependent scalar w.r.t. lapse rate is finite."""
        std = [900.0]

        def loss(lapse):
            cfg = MultiLayerLandConfig(
                snow_albedo_feedback=True,
                elev_bands=ElevationSnowBandConfig(
                    band_dz=band_elevation_anomalies(jnp.asarray(std)),
                    lapse_rate_K_m=lapse,
                ),
            )
            state = init_multilayer_land_state(1, cfg)
            forcing = _make_forcing(1, T_air=constants.T_freeze + 1.0, precip=5e-4)
            for _ in range(3):
                state, resp, _ = step_multilayer_land(
                    state, forcing, cfg, U_min=1.0, dt=1800.0,
                    lat=jnp.full(1, 0.7),
                )
            return jnp.sum(resp.albedo) + jnp.sum(state.snow_depth)

        g = jax.grad(loss)(6.0e-3)
        self.assertTrue(bool(jnp.isfinite(g)))


if __name__ == "__main__":
    unittest.main()
