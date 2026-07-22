"""Smoke test for the offline EC-site diagnostic driver (scripts/run/run_ec_site.py).

Builds a small synthetic DifferBESS-style v2 driver NetCDF (one diurnal cycle),
runs ``run_site`` end to end (read -> vmap canopy -> obs comparison -> NetCDF),
and asserts the pipeline produces finite fluxes, sensible skill metrics, daytime
GPP > 0, and a well-formed output file.  The variable-mapping correctness of the
reader itself is covered by tests/land/boundary_data/test_ec_site.py.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import xarray as xr
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_DRIVER_PY = _REPO / "scripts" / "run" / "run_ec_site.py"


def _load_driver_module():
    spec = importlib.util.spec_from_file_location("run_ec_site", _DRIVER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_driver(path: str, n: int = 96) -> None:
    """A minimal valid v2 driver: n half-hourly steps over a diurnal cycle."""
    rng = np.arange(n)
    hour = (rng * 0.5) % 24.0
    day = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)   # 0 at night
    sw = 850.0 * day                                               # W/m2
    sza = 90.0 - 65.0 * day                                        # deg (<90 day)
    ones = np.ones(n)

    def var(values):
        return ("time", np.asarray(values, dtype="f8"))

    ds = xr.Dataset(
        {
            "SW_IN": var(sw), "LW_IN": var(330.0 * ones), "TA": var(20.0 + 8.0 * day),
            "VPD": var(8.0 + 10.0 * day), "PA": var(99.0 * ones), "WS": var(2.5 * ones),
            "P": var(0.0 * ones), "CO2": var(410.0 * ones), "SZA": var(sza),
            "SWC": var(28.0 * ones), "TS": var(19.0 + 6.0 * day),
            "LAI": var(2.4 * ones), "CI": var(0.7 * ones), "T_GROWTH": var(21.0 * ones),
            "EMISSIVITY": var(0.97 * ones), "Vcmax25_C3Leaf": var(45.0 * ones),
            "BESS_PAR_DIFF_PAR_RATIO": var(0.3 + 0.4 * (1.0 - day)),
            "Albedo_BSA_vis": var(0.08 * ones), "Albedo_WSA_vis": var(0.09 * ones),
            "Albedo_BSA_nir": var(0.30 * ones), "Albedo_WSA_nir": var(0.32 * ones),
            "IGBP": var(3.0 * ones),       # DBF
            "CLIMATE": var(2.0 * ones), "C4": var(0.0 * ones),
            "CANOPY_HEIGHT": var(20.0 * ones), "LAT": var(39.0 * ones),
            "LONG": var(-86.0 * ones), "ELEVATION": var(275.0 * ones),
            # observed fluxes (only daytime "measured" for GPP/H; ET in mm/day)
            "GPP_DT": var(np.where(day > 0.1, 12.0 * day, 0.0)),
            "NEE": var(-8.0 * day + 2.0), "ET": var(2.0 * day), "H": var(60.0 * day),
            "USTAR": var(0.15 + 0.35 * day),   # friction velocity [m/s]
        },
        coords={"time": (np.datetime64("2015-06-01T00:00")
                         + np.arange(n) * np.timedelta64(30, "m"))},
        attrs={"site": "SYN-Test"},
    )
    ds.to_netcdf(path)


def test_run_ec_site_diagnostic_smoke(tmp_path):
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc)
    mod = _load_driver_module()

    out_dir = str(tmp_path / "out")
    metrics = mod.run_site(driver_nc, "diagnostic", out_dir, chunk=96)

    # metrics for all three fluxes, with a finite RMSE on the valid steps
    assert set(metrics) == {"GPP", "LE", "H"}
    for flux in ("GPP", "LE", "H"):
        assert metrics[flux]["n"] > 0
        assert np.isfinite(metrics[flux]["rmse"])

    # output NetCDF is well-formed and the modelled fluxes are finite + physical
    out_nc = pathlib.Path(out_dir) / "SYN-Test_ec_diagnostic.nc"
    assert out_nc.exists()
    ds = xr.open_dataset(out_nc)
    v = ds.valid.values.astype(bool)
    assert v.any()
    gpp = ds.gpp_mod.values
    le = ds.le_mod.values
    assert np.all(np.isfinite(gpp[v]))
    assert np.all(np.isfinite(le[v]))
    # daytime photosynthesis is positive somewhere
    assert np.nanmax(gpp[v]) > 0.0
    # latent heat stays physical (no runaway): below ~ peak SW + slack
    assert np.nanmax(le[v]) < 1000.0


def test_run_ec_site_prognostic_smoke(tmp_path):
    """Prognostic mode integrates the multilayer soil forward (lax.scan) and
    reports the NaN-revert count; on clean synthetic forcing nothing reverts."""
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc)
    mod = _load_driver_module()

    out_dir = str(tmp_path / "out_prog")
    metrics = mod.run_site(driver_nc, "prognostic", out_dir, chunk=96)

    # prognostic mode also evaluates the soil STATE (top ~5 cm T / moisture) vs the
    # driver's shallowest observed TS / SWC.
    assert set(metrics) == {"GPP", "LE", "H", "TS", "SWC", "USTAR"}
    for flux in ("GPP", "LE", "H", "TS", "SWC", "USTAR"):
        assert metrics[flux]["n"] > 0
        assert np.isfinite(metrics[flux]["rmse"])
        # the NaN guard must keep every modelled-on-valid step finite
        assert metrics[flux]["model_nan"] == 0

    out_nc = pathlib.Path(out_dir) / "SYN-Test_ec_prognostic.nc"
    assert out_nc.exists()
    ds = xr.open_dataset(out_nc)
    assert ds.attrs["mode"] == "prognostic"
    v = ds.valid.values.astype(bool)
    assert np.all(np.isfinite(ds.gpp_mod.values[v]))
    assert np.all(np.isfinite(ds.le_mod.values[v]))
    assert np.all(np.isfinite(ds.h_mod.values[v]))
    # the exact scoring mask is persisted so metric counts are reproducible
    assert "score_valid" in ds and "reverted" in ds
    sv = ds.score_valid.values.astype(bool)
    n_recomputed = int((sv & np.isfinite(ds.gpp_mod.values)
                        & np.isfinite(ds.gpp_obs.values)).sum())
    assert n_recomputed == metrics["GPP"]["n"]


def test_reader_exposes_volumetric_theta_for_prognostic_ic(tmp_path):
    """The prognostic IC must initialise soil moisture from the observed
    volumetric water content (SWC/100), not by inverting w_frac_rz with
    mismatched thresholds.  Lock the reader contract that backs that fix."""
    from legoesm.land.boundary_data.ec_site import read_ec_site_driver

    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc)                      # SWC = 28.0 % everywhere
    d = read_ec_site_driver(driver_nc)

    theta = np.asarray(d.theta_soil).ravel()
    assert np.all(np.isfinite(theta))
    assert np.allclose(theta, 0.28, atol=1e-9)   # 28 % -> 0.28 m3/m3


def test_run_ec_site_rejects_unknown_mode(tmp_path):
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc, n=8)
    mod = _load_driver_module()
    with pytest.raises(ValueError, match="mode"):
        mod.run_site(driver_nc, "bogus-mode", str(tmp_path / "o"), chunk=8)


def test_texture_config_builds_van_genuchten_hydraulics():
    """Fix 1: ``_build_land_config(texture=(sand, clay))`` maps the USDA class to
    per-site van-Genuchten hydraulics (Carsel-Parrish) + the matching
    theta_fc/theta_wp, overriding the loam-for-all preset.  Locks the wiring and
    the (theta_wp < theta_fc <= theta_sat) ordering the SWC-bias fix depends on
    (a swapped fc/wp unpack would silently invert the water-holding band)."""
    from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
    mod = _load_driver_module()
    cfg = mod._build_land_config(
        mod.TwoLeafCanopyConfig(), soil="auto", bottom_bc="free_drainage",
        depth_m=0.0, texture=(79.4, 9.4))          # loamy sand (US-SRM)
    assert isinstance(cfg.hydraulics, SoilHydraulicsConfig)
    assert cfg.hydraulics.retention_curve == "van_genuchten"
    assert 1e-5 < float(cfg.hydraulics.K_sat) < 1e-3
    assert 0.30 < float(cfg.hydraulics.theta_sat) < 0.50
    assert float(cfg.theta_wp) < float(cfg.theta_fc) <= float(cfg.hydraulics.theta_sat)
    # a clay texture must hold MORE water and drain SLOWER than the loamy sand.
    clay = mod._build_land_config(
        mod.TwoLeafCanopyConfig(), soil="auto", bottom_bc="free_drainage",
        depth_m=0.0, texture=(20.0, 60.0))
    assert float(clay.theta_fc) > float(cfg.theta_fc)
    assert float(clay.hydraulics.K_sat) < float(cfg.hydraulics.K_sat)


def test_texture_lookup_reads_site_table():
    """``_texture_lookup`` resolves a site to its (sand, clay) from the committed
    table, and returns None for an unknown site (loam-default fallback path)."""
    mod = _load_driver_module()
    csv = str(_REPO / mod._DEFAULT_TEXTURE_CSV)
    tex = mod._texture_lookup("US-SRM", csv)
    assert tex is not None
    sand, clay = tex
    assert sand > 70.0 and clay < 15.0            # loamy sand (Santa Rita, BIF)
    assert mod._texture_lookup("ZZ-Nowhere", csv) is None


def test_ec_site_physics_table_and_defaults():
    """``ec_site_physics`` returns the per-site tower height + root/column depth
    for a table site (BADM-anchored), and the generic fallbacks for an unknown
    site — the consolidated single source of truth that replaced the ad-hoc
    per-site environment overrides.  A phreatophyte site (US-Ton) carries a
    deeper root e-folding depth and column than the 1 m / model-default fallback.
    """
    mod = _load_driver_module()
    ton = mod.ec_site_physics("US-Ton")
    assert ton["z_ref"] == 23.5                    # FLUXNET BADM Reference_height_v
    assert ton["root_depth"] == 5.0                # deep-rooted blue oaks (phreatophyte)
    assert ton["soil_depth_m"] > 0.0               # deepened column
    mms = mod.ec_site_physics("US-MMS")
    assert mms["z_ref"] == 46.0 and mms["root_depth"] == 2.0   # tall tower, deep loam
    var = mod.ec_site_physics("US-Var")
    assert var["z_ref"] == 2.0 and var["root_depth"] == 1.0    # short tower, shallow grass
    unknown = mod.ec_site_physics("ZZ-Nowhere")
    assert unknown == {"z_ref": 10.0, "root_depth": 1.0, "soil_depth_m": 0.0}


def test_build_land_config_threads_z_ref():
    """The consolidated tower height reaches the surface-layer config: ``z_ref``
    passed to ``_build_land_config`` lands on ``MultiLayerLandConfig.z_ref`` so
    the Monin-Obukhov profile / u* is anchored at the real measurement height
    (a silently-dropped z_ref would leave every site at the generic 10 m)."""
    mod = _load_driver_module()
    cfg = mod._build_land_config(
        mod.TwoLeafCanopyConfig(), soil="default", bottom_bc="free_drainage",
        depth_m=0.0, z_ref=46.0)
    assert float(cfg.z_ref) == 46.0


def test_stress_b0_flag_reaches_canopy_config():
    """The ``stress_b0`` selector threads onto the canopy config: the offline
    default (False) leaves the Ball-Berry cuticular intercept unstressed, and
    the legacy opt-in (True) restores the both-slope-and-intercept stress."""
    mod = _load_driver_module()
    assert mod.TwoLeafCanopyConfig(stress_b0=False).stress_b0 is False
    assert mod.TwoLeafCanopyConfig(stress_b0=True).stress_b0 is True


def test_run_site_rejects_unknown_canopy(tmp_path):
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc, n=8)
    mod = _load_driver_module()
    with pytest.raises(ValueError, match="canopy"):
        mod.run_site(driver_nc, "prognostic", str(tmp_path / "o"), chunk=8,
                     canopy="bogus")


def test_clmml_requires_prognostic_mode(tmp_path):
    """CLM-ML cannot be vmapped over timesteps -> diagnostic mode is rejected."""
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc, n=8)
    mod = _load_driver_module()
    with pytest.raises(ValueError, match="clmml requires .*prognostic"):
        mod.run_site(driver_nc, "diagnostic", str(tmp_path / "o"), chunk=8,
                     canopy="clmml")


def test_vcmax_scale_raises_gpp(tmp_path):
    """Scaling Vcmax25 up raises GPP — the photosynthetic-capacity knob."""
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc)
    mod = _load_driver_module()
    base = mod.run_site(driver_nc, "prognostic", str(tmp_path / "b"), chunk=96)
    hi = mod.run_site(driver_nc, "prognostic", str(tmp_path / "h"), chunk=96,
                      vcmax_scale=1.5)
    assert hi["GPP"]["bias"] > base["GPP"]["bias"]


def test_clmml_turbulence_rejects_unknown(tmp_path):
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc, n=8)
    mod = _load_driver_module()
    with pytest.raises(ValueError):
        mod.run_site(driver_nc, "prognostic", str(tmp_path / "o"), chunk=8,
                     canopy="clmml", clmml_turbulence="bogus")


def test_canopy_interception_changes_water_partition(tmp_path):
    """Enabling the shared interception scheme carries a canopy-water store and
    re-routes rain (interception loss), so the fluxes differ from the no-
    interception run and the run stays finite (water-conserving path)."""
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc)
    # the base synthetic driver is rain-free (P=0); interception only acts on
    # rain, so inject a wet spell (2 mm/step over the first 24 h).
    d = xr.open_dataset(driver_nc)
    P = np.zeros(d.sizes["time"]); P[:48] = 2.0
    d = d.assign(P=("time", P))
    d.to_netcdf(driver_nc + ".rain"); import os; os.replace(driver_nc + ".rain", driver_nc)
    mod = _load_driver_module()
    off = mod.run_site(driver_nc, "prognostic", str(tmp_path / "off"), chunk=96,
                       canopy="two_leaf")
    on = mod.run_site(driver_nc, "prognostic", str(tmp_path / "on"), chunk=96,
                      canopy="two_leaf", interception=True)
    for flux in ("GPP", "LE", "H"):
        assert np.isfinite(on[flux]["rmse"])
    # the synthetic driver has rain, so interception must move at least one flux
    assert (off["LE"]["bias"] != on["LE"]["bias"]
            or off["H"]["bias"] != on["H"]["bias"])


def test_spinup_steps_trims_to_eval_window(tmp_path):
    """--spinup-steps runs extra steps before the window but scores only the
    evaluation window: the output length and metric count match a no-spinup run
    of the same window, and the spin-up soil state carries into it."""
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc, n=96)
    mod = _load_driver_module()
    base = mod.run_site(driver_nc, "prognostic", str(tmp_path / "b"), chunk=96,
                        start_step=48, max_steps=48)
    spun = mod.run_site(driver_nc, "prognostic", str(tmp_path / "s"), chunk=96,
                        start_step=48, max_steps=48, spinup_steps=48)
    # same evaluation window => same scored sample count
    assert base["GPP"]["n"] == spun["GPP"]["n"]
    out = xr.open_dataset(pathlib.Path(tmp_path / "s") / "SYN-Test_ec_prognostic.nc")
    assert out.sizes["time"] == 48          # scored window only, spin-up trimmed
    # the spun-up soil state differs from the cold start => fluxes are not identical
    assert base["LE"]["bias"] != spun["LE"]["bias"]


def test_de_hai_has_reference_height():
    """DE-Hai (canopy ~34 m) must have a tower z_ref above its canopy, else the
    CLM-ML within-canopy wind profile hits a math-domain error."""
    mod = _load_driver_module()
    phys = mod.ec_site_physics("DE-Hai")
    assert phys["z_ref"] > 34.0


def test_stomatal_m_scale_raises_transpiration(tmp_path):
    """Scaling the Ball-Berry slope up moves latent heat up / sensible heat down
    (the Bowen-ratio lever) at fixed forcing; m_scale=1 is a no-op."""
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc)
    mod = _load_driver_module()
    base = mod.run_site(driver_nc, "prognostic", str(tmp_path / "b"), chunk=96)
    hi = mod.run_site(driver_nc, "prognostic", str(tmp_path / "h"), chunk=96,
                      stomatal_m_scale=2.0)
    # more stomatal opening => more latent, less sensible heat
    assert hi["LE"]["bias"] > base["LE"]["bias"]
    assert hi["H"]["bias"] < base["H"]["bias"]


def test_plant_wilting_point_separate_from_soil(tmp_path):
    """A PLANT wilting point below the soil moisture lets transpiration (and its
    GPP) continue where a single soil wilting point would shut it off — the
    separate soil/plant wilting knobs (phreatophyte deep extraction)."""
    driver_nc = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(driver_nc)
    # dry the synthetic soil below the default wilting point
    d = xr.open_dataset(driver_nc)
    d = d.assign(SWC=("time", np.full(d.sizes["time"], 8.0)))   # 0.08 vol
    d.to_netcdf(driver_nc + ".dry"); import os; os.replace(driver_nc + ".dry", driver_nc)
    mod = _load_driver_module()
    hi_wp = mod.run_site(driver_nc, "prognostic", str(tmp_path / "hi"), chunk=96,
                         canopy="two_leaf")                       # soil wp = plant wp
    lo_wp = mod.run_site(driver_nc, "prognostic", str(tmp_path / "lo"), chunk=96,
                         canopy="two_leaf", plant_wilting_point=0.04)
    # lower plant wilting point => more root-zone availability => more GPP
    assert lo_wp["GPP"]["bias"] > hi_wp["GPP"]["bias"]


def test_config_separates_soil_and_plant_wilting():
    from legoesm.land.config import MultiLayerLandConfig
    c = MultiLayerLandConfig(theta_wp=0.15, theta_wp_plant=0.06)
    assert c.theta_wp == 0.15 and c.theta_wp_plant == 0.06
    assert MultiLayerLandConfig(theta_wp=0.15).theta_wp_plant is None   # default
