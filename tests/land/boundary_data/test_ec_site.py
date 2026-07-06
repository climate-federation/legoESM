"""Unit tests for the DifferBESS-v2 EC-site driver adapter.

Exercises the variable mapping (DifferBESS driver -> legoESM AtmToSurface +
CanopyLandParams), the production-faithful LAI handling, and the gap-filling on a
small synthetic v2-schema NetCDF, plus the pure derivation helpers.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.land.boundary_data.ec_site import (
    read_ec_site_driver,
    _specific_humidity_from_vpd,
    _gapfill,
)


def _synthetic_v2(tmp_path, name="SYN", n=8, dt_min=30, igbp=7, climate=1, c4=0.0,
                  ta_c=20.0, vpd_hpa=10.0, swc_pct=30.0, lai=2.0, nan_idx=None,
                  sza_deg=30.0, vcmax_cov=1.0, p_nan=False, fdiff_cov=1.0,
                  sw_in=400.0, omit=(), times=None, obs_sentinel=False, emiss_nan=False,
                  vcmax_vals=None):
    """Build a minimal DifferBESS-v2 driver NetCDF the adapter can read."""
    if times is not None:
        time = np.asarray(times, dtype="datetime64[ns]"); n = len(time)
    else:
        time = np.datetime64("2005-06-01T00:15") + np.arange(n) * np.timedelta64(dt_min, "m")
    ones = np.ones(n)
    ta = ta_c * ones
    if nan_idx is not None:        # punch a gap into TA
        ta = ta.copy(); ta[nan_idx] = np.nan
    sza = (sza_deg * ones) if np.isscalar(sza_deg) else np.asarray(sza_deg, float)
    sw = (sw_in * ones) if np.isscalar(sw_in) else np.asarray(sw_in, float)
    if vcmax_vals is not None:     # explicit Vcmax series (e.g. with finite zeros)
        vcmax = np.asarray(vcmax_vals, dtype=float)
    else:                          # vcmax_cov = fraction of finite Vcmax samples
        vcmax = 60.0 * ones.copy()
        vcmax[int(round(vcmax_cov * n)):] = np.nan
    fdiff = 0.4 * ones.copy()      # fdiff_cov = fraction of finite diffuse-PAR ratio
    fdiff[int(round(fdiff_cov * n)):] = np.nan
    precip = np.full(n, np.nan) if p_nan else 0.0 * ones
    dv = dict(
        SW_IN=("time", sw), LW_IN=("time", 350.0 * ones),
        TA=("time", ta), VPD=("time", vpd_hpa * ones), PA=("time", 100.0 * ones),
        WS=("time", 3.0 * ones), P=("time", precip), CO2=("time", 400.0 * ones),
        SWC=("time", swc_pct * ones), TS=("time", (ta_c - 2.0) * ones),
        SZA=("time", sza), LAI=("time", lai * ones), CI=("time", 0.7 * ones),
        T_GROWTH=("time", 22.0 * ones),
        EMISSIVITY=("time", np.full(n, np.nan) if emiss_nan else 0.98 * ones),
        Vcmax25_C3Leaf=("time", vcmax),
        BESS_PAR_DIFF_PAR_RATIO=("time", fdiff),
        Albedo_BSA_vis=("time", 0.08 * ones), Albedo_WSA_vis=("time", 0.10 * ones),
        Albedo_BSA_nir=("time", 0.25 * ones), Albedo_WSA_nir=("time", 0.28 * ones),
        IGBP=("time", float(igbp) * ones), CLIMATE=("time", float(climate) * ones),
        C4=("time", c4 * ones), CANOPY_HEIGHT=("time", 10.0 * ones),
        LAT=("time", 38.0 * ones), LONG=("time", -120.0 * ones),
        ET=("time", 2.0 * ones), GPP_DT=("time", 8.0 * ones),
        H=("time", 50.0 * ones), NEE=("time", -5.0 * ones),
    )
    if obs_sentinel:               # FLUXNET -9999 fill values in the obs fluxes
        for k in ("ET", "GPP_DT", "H", "NEE"):
            dv[k] = ("time", np.full(n, -9999.0))
    for k in omit:                 # simulate schema omission of a variable
        dv.pop(k, None)
    ds = xr.Dataset({k: v for k, v in dv.items()}, coords={"time": time})
    p = str(tmp_path / f"{name}_driver_v2.nc")
    ds.to_netcdf(p)
    return p


# --- pure helpers ----------------------------------------------------------
def test_specific_humidity_from_vpd_zero_at_saturation():
    T = np.array([293.15]); p = np.array([1.0e5])
    # VPD=0 -> e=es -> q at saturation (>0); huge VPD -> e clamped to 0 -> q=0.
    q_sat = _specific_humidity_from_vpd(T, np.array([0.0]), p)
    q_dry = _specific_humidity_from_vpd(T, np.array([1.0e5]), p)
    assert q_sat[0] > 0.01
    assert q_dry[0] == 0.0


def test_specific_humidity_known_value():
    # 20 C, VPD 1000 Pa, p 1e5 Pa: es~2339, e~1339, q = eps e/(p-(1-eps)e).
    T = np.array([293.15]); q = _specific_humidity_from_vpd(
        T, np.array([1000.0]), np.array([1.0e5]))
    assert 0.007 < float(q[0]) < 0.009


def test_gapfill_interpolates_and_flags():
    a = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    filled, observed = _gapfill(a)
    assert np.all(np.isfinite(filled))
    assert np.allclose(filled, [1, 2, 3, 4, 5, 6])              # linear interp
    assert observed.tolist() == [True, False, True, False, False, True]


def test_gapfill_all_nan_is_finite_and_invalid():
    filled, observed = _gapfill(np.full(4, np.nan))
    assert np.all(np.isfinite(filled)) and not observed.any()


# --- end-to-end mapping ----------------------------------------------------
def test_read_maps_units_and_shapes(tmp_path):
    d = read_ec_site_driver(_synthetic_v2(tmp_path, n=8, dt_min=30))
    f, cp = d.forcing, d.canopy_params
    assert f.T_lowest.shape == (8, 1) and cp.LAI.shape == (8, 1)
    assert d.dt_s == 1800.0
    # air temperature mapped degC -> K via constants.T_freeze
    assert jnp.allclose(f.T_lowest, 20.0 + constants.T_freeze)
    # p_surface = PA(kPa)*1000
    assert jnp.allclose(f.p_surface, 1.0e5)
    # cos_zenith = cos(30 deg)
    assert jnp.allclose(f.cos_zenith, np.cos(np.deg2rad(30.0)))
    # rho = p/(Rd Tv), all finite & physical
    assert jnp.all(jnp.isfinite(f.rho_lowest)) and float(f.rho_lowest[0, 0]) > 1.0
    # prescribed soil skin T mapped from TS (degC -> K)
    assert jnp.allclose(d.T_soil_top, 18.0 + constants.T_freeze)


def test_lai_production_faithful_kn_fnonveg(tmp_path):
    # LAI clipped to [0,7]; kn = -0.62 ln(mean LAI)+0.98; FNonVeg = exp(-0.5 CI LAI).
    d = read_ec_site_driver(_synthetic_v2(tmp_path, lai=2.0))
    cp = d.canopy_params
    assert jnp.allclose(cp.LAI, 2.0)
    assert jnp.allclose(cp.kn, -0.62 * np.log(2.0) + 0.98)
    assert jnp.allclose(cp.FNonVeg, np.exp(-0.5 * 0.7 * 2.0))


def test_pft_and_aero_from_igbp(tmp_path):
    # IGBP 7 (WSA) -> SAV (rz0m 0.12); IGBP 3 (DBF) -> DBF (rz0m 0.055).
    d_sav = read_ec_site_driver(_synthetic_v2(tmp_path, name="sav", igbp=7))
    d_dbf = read_ec_site_driver(_synthetic_v2(tmp_path, name="dbf", igbp=3))
    assert d_sav.pft == "SAV" and jnp.allclose(d_sav.canopy_params.rz0m, 0.12)
    assert d_dbf.pft == "DBF" and jnp.allclose(d_dbf.canopy_params.rz0m, 0.055)


def test_gap_marks_invalid_but_fills_forcing(tmp_path):
    # A NaN in TA -> that step is gap-filled (finite forcing) but flagged invalid.
    d = read_ec_site_driver(_synthetic_v2(tmp_path, n=6, ta_c=20.0, nan_idx=2))
    assert jnp.all(jnp.isfinite(d.forcing.T_lowest))     # filled
    assert not d.valid[2] and d.valid[0]                 # step 2 flagged


def test_dry_vs_wet_moisture_stress(tmp_path):
    # SWC 12% (dry) -> low beta; SWC 35% (wet) -> beta saturates at 1.
    dry = read_ec_site_driver(_synthetic_v2(tmp_path, name="dry", swc_pct=12.0))
    wet = read_ec_site_driver(_synthetic_v2(tmp_path, name="wet", swc_pct=35.0))
    assert float(dry.w_frac_rz[0, 0]) < float(wet.w_frac_rz[0, 0])
    assert jnp.allclose(wet.w_frac_rz, 1.0)


# --- adversarial regressions (codex Stage-A findings) ----------------------
def test_nighttime_sza_above_90_gives_zero_cos_and_is_observed(tmp_path):
    # SZA > 90 (night) must stay observed (not NaN-masked) and yield cos_zenith=0,
    # not an interpolated daytime angle.
    sza = np.array([150.0, 120.0, 30.0, 45.0])     # 2 night, 2 day
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="nt", n=4, sza_deg=sza))
    cz = np.asarray(d.forcing.cos_zenith).ravel()
    assert cz[0] == 0.0 and cz[1] == 0.0           # night -> 0
    assert cz[2] > 0.0 and cz[3] > 0.0             # day   -> positive
    assert d.valid.all()                            # SZA observed everywhere


def test_all_missing_vcmax_falls_back_and_stays_valid(tmp_path):
    # An entirely-missing Vcmax25_C3Leaf uses the PFT/climate lookup (NOT 0) and
    # those steps STAY valid (the lookup is the intended production forcing).
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="vc", igbp=7, vcmax_cov=0.0))
    vc3 = np.asarray(d.canopy_params.Vcmax25_C3_leaf)
    assert np.all(vc3 > 1.0)                        # not the gap-filled zero
    assert d.valid.all()                            # lookup fallback -> still valid


def test_cropland_forces_vcmax_lookup(tmp_path):
    # IGBP 10/12 (cropland) uses the PFT/climate lookup, NOT the driver value,
    # even with full driver coverage (DifferBESS io.py:427-428), and stays valid.
    from legoesm.land.canopy.config import lookup_vcmax25
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="cro", igbp=10, vcmax_cov=1.0))
    expect = lookup_vcmax25(d.pft, d.climate, c4=False)
    assert d.pft == "CRO"
    assert jnp.allclose(d.canopy_params.Vcmax25_C3_leaf, expect)   # lookup, not 60
    assert d.valid.all()                            # lookup-backed -> valid


def test_sparse_vcmax_uses_lookup_dense_gates_validity(tmp_path):
    # Coverage <= 50% -> lookup (valid everywhere); > 50% -> driver, whose
    # per-step gaps then invalidate those steps (DifferBESS io.py:429).
    from legoesm.land.canopy.config import lookup_vcmax25
    sparse = read_ec_site_driver(_synthetic_v2(tmp_path, name="sp", igbp=7, vcmax_cov=0.3))
    dense = read_ec_site_driver(_synthetic_v2(tmp_path, name="dn", igbp=7, vcmax_cov=0.9))
    expect = lookup_vcmax25(sparse.pft, sparse.climate, c4=False)
    assert jnp.allclose(sparse.canopy_params.Vcmax25_C3_leaf, expect)   # sparse -> lookup
    assert sparse.valid.all()                                          # lookup -> valid
    assert jnp.allclose(dense.canopy_params.Vcmax25_C3_leaf, 60.0)      # dense -> driver
    assert dense.valid[0] and not dense.valid[-1]   # driver gap -> that step invalid


def test_zero_vcmax_samples_are_observed(tmp_path):
    # Finite zero Vcmax (dormant season) is a valid observation, not missing:
    # those steps stay valid and use Vcmax=0 (production counts every non-NaN).
    vals = [60.0, 0.0, 60.0, 0.0, 60.0, 0.0]    # all finite, includes zeros
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="vz", n=6, igbp=7, vcmax_vals=vals))
    vc3 = np.asarray(d.canopy_params.Vcmax25_C3_leaf).ravel()
    assert d.valid.all()                         # zeros are observed -> valid
    assert np.any(vc3 == 0.0)                    # zero samples used unchanged


def test_missing_emissivity_uses_fallback_and_stays_valid(tmp_path):
    # Missing EMISSIVITY uses the site-mean/0.97 production fallback and does NOT
    # invalidate the step (excluded from _REQUIRED), matching DifferBESS.
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="em", emiss_nan=True))
    assert jnp.allclose(d.canopy_params.emissivity, 0.97)   # EMISS_VEG fallback
    assert d.valid.all()


def test_diffuse_par_gap_invalidates_steps(tmp_path):
    # BESS_PAR_DIFF_PAR_RATIO drives the albedo blend; a gap there must invalidate
    # the step (partial), and an all-missing series must invalidate every step
    # while still producing a finite (0.5-default) albedo.
    part = read_ec_site_driver(_synthetic_v2(tmp_path, name="fdp", n=6, fdiff_cov=0.5))
    assert part.valid[0] and not part.valid[-1]            # gap at the tail -> invalid
    allmiss = read_ec_site_driver(_synthetic_v2(tmp_path, name="fdm", fdiff_cov=0.0))
    assert not allmiss.valid.any()                          # entirely missing -> invalid
    assert jnp.all(jnp.isfinite(allmiss.canopy_params.ALB_VIS))   # 0.5-default blend


def test_missing_c4_metadata_invalidates(tmp_path):
    # An absent/NaN C4 fraction must invalidate every step (no silent pure-C3),
    # while fC4 defaults to 0.0 to keep arrays finite.
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="c4n", c4=np.nan))
    assert not d.valid.any()
    assert jnp.allclose(d.canopy_params.fC4, 0.0)
    assert jnp.all(jnp.isfinite(d.canopy_params.fC4))


def test_swc_above_75_invalidates(tmp_path):
    # SWC > 75% is rejected by the production physical-range guard -> invalid.
    ok = read_ec_site_driver(_synthetic_v2(tmp_path, name="swcok", swc_pct=60.0))
    bad = read_ec_site_driver(_synthetic_v2(tmp_path, name="swcbad", swc_pct=80.0))
    assert ok.valid.any()
    assert not bad.valid.any()


def test_d_leaf_is_pft_specific(tmp_path):
    # PFT-specific leaf width from PFT_LEAF_WIDTH (ENF 0.01, EBF 0.04, GRA 0.02,
    # DBF 0.025) drives leaf boundary-layer resistance — not a universal default.
    widths = {0: ("ENF", 0.01), 1: ("EBF", 0.04), 9: ("GRA", 0.02), 3: ("DBF", 0.025)}
    for igbp, (pft, w) in widths.items():
        d = read_ec_site_driver(_synthetic_v2(tmp_path, name=f"dl{igbp}", igbp=igbp))
        assert d.pft == pft
        assert d.canopy_params.d_leaf is not None
        assert jnp.allclose(d.canopy_params.d_leaf, w)


def test_all_missing_precip_marks_all_invalid(tmp_path):
    # An all-missing P series -> zero precip forcing but every step invalid.
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="p", p_nan=True))
    assert jnp.all(jnp.isfinite(d.forcing.precip_total))
    assert not d.valid.any()


def test_lai_above_7_is_masked_invalid(tmp_path):
    # Production DifferBESS masks ds.LAI outside [0,7] -> missing (io.py:315),
    # NOT clip-and-pass.  LAI=8 must invalidate every step, while LAI stays
    # finite (gap-filled) for the NaN-intolerant canopy.
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="lai8", lai=8.0))
    assert not d.valid.any()
    assert jnp.all(jnp.isfinite(d.canopy_params.LAI))


def test_missing_metadata_variable_does_not_crash(tmp_path):
    # Schema OMISSION of any required site scalar (not just NaN) must yield a
    # finite, all-invalid record (no KeyError / no int(round(NaN)) crash).
    for var in ("C4", "CANOPY_HEIGHT", "LAT", "LONG", "IGBP", "CLIMATE"):
        d = read_ec_site_driver(_synthetic_v2(tmp_path, name=f"om{var}", omit=(var,)))
        assert not d.valid.any()
        assert jnp.all(jnp.isfinite(d.forcing.T_lowest))
        assert jnp.all(jnp.isfinite(d.canopy_params.hc))
        assert d.pft in ("ENF", "EBF", "DNF", "DBF", "MF", "SHR", "SAV", "GRA", "CRO")
        # Returned geometry is finite even when LAT/LONG are omitted (gated invalid).
        assert np.isfinite(d.lat_rad) and np.isfinite(d.lon_deg)


def test_observed_flux_sentinels_become_nan(tmp_path):
    # FLUXNET -9999 fill values in the observed fluxes must map to NaN (not be
    # compared / not convert into a huge negative LE).
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="sent", obs_sentinel=True))
    for k in ("gpp_umol", "le_wm2", "h_wm2", "nee_umol"):
        assert np.all(np.isnan(d.obs[k])), k


def test_malformed_time_axis_invalidates(tmp_path):
    # Duplicate, reversed, or irregular timestamps must invalidate every step
    # (zero/negative/irregular dt -> NaN/Inf precip otherwise) while keeping
    # precip finite via a safe fallback dt.
    base = np.datetime64("2005-06-01T00:00")
    mk = lambda mins: [base + np.timedelta64(m, "m") for m in mins]
    for label, mins in [("dup", [0, 0, 30, 60]),        # duplicate
                        ("rev", [90, 60, 30, 0]),        # reversed
                        ("irr", [0, 30, 90, 120])]:      # irregular spacing
        d = read_ec_site_driver(_synthetic_v2(tmp_path, name=f"t{label}", times=mk(mins)))
        assert not d.valid.any(), label
        assert jnp.all(jnp.isfinite(d.forcing.precip_total)), label


def test_unsupported_igbp_climate_invalidates(tmp_path):
    # An unknown finite IGBP/CLIMATE code must NOT silently pass as a valid DBF/
    # temperate run; it falls back to finite params but invalidates every step.
    bad_igbp = read_ec_site_driver(_synthetic_v2(tmp_path, name="bi", igbp=99))
    bad_clim = read_ec_site_driver(_synthetic_v2(tmp_path, name="bc", climate=99))
    assert not bad_igbp.valid.any() and not bad_clim.valid.any()
    assert jnp.all(jnp.isfinite(bad_igbp.canopy_params.rz0m))   # finite fallback


def test_negative_shortwave_clamped_or_masked(tmp_path):
    # SW_IN in [-20,0) -> clamped to 0 but kept observed (valid); SW_IN < -20 ->
    # masked NaN -> gap-filled but invalid (DifferBESS noise handling).
    sw = np.array([400.0, -5.0, -50.0, 300.0])     # ok, noise, sentinel, ok
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="sw", n=4, sw_in=sw))
    swd = np.asarray(d.forcing.sw_down).ravel()
    assert np.all(swd >= 0.0)                       # no negative SW into the canopy
    assert swd[1] == 0.0                            # [-20,0) clamped to 0
    assert d.valid[0] and d.valid[1]                # noise clamp kept observed
    assert not d.valid[2]                           # < -20 masked -> invalid


def test_rho_matches_canonical_virtual_temperature(tmp_path):
    # rho = p/(R_d * T_v), T_v = T(1 + (1/eps - 1) q) — no hardcoded 0.61.
    d = read_ec_site_driver(_synthetic_v2(tmp_path, name="rho"))
    f = d.forcing
    T = np.asarray(f.T_lowest); q = np.asarray(f.q_lowest); p = np.asarray(f.p_surface)
    T_v = T * (1.0 + (1.0 / constants.epsilon - 1.0) * q)
    assert jnp.allclose(f.rho_lowest, p / (constants.R_d * T_v))
