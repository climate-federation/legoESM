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


def _make_driver(path: str, n: int = 96, igbp: float = 3.0,
                 fc4: float = 0.0) -> None:
    """A minimal valid v2 driver: n half-hourly steps over a diurnal cycle.

    ``igbp``/``fc4`` default to a pure-C3 DBF; pass ``igbp=8, fc4=0.5`` for a
    woody-savanna (SAV) fixture with a C3-tree + C4-grass canopy mix.
    """
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
            "IGBP": var(igbp * ones),
            "CLIMATE": var(2.0 * ones), "C4": var(fc4 * ones),
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


def test_savanna_c3_c4_vcmax_split(tmp_path):
    """On a savanna (fC4>0) fixture, the C3 (tree) and C4 (grass) Vcmax knobs each
    move GPP independently, and the per-pathway flag overrides --vcmax-scale."""
    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5)          # woody savanna, 50% C4
    mod = _load_driver_module()
    base = mod.run_site(driver_nc, "prognostic", str(tmp_path / "b"), chunk=96)
    c4 = mod.run_site(driver_nc, "prognostic", str(tmp_path / "c4"), chunk=96,
                      vcmax_c4_scale=1.6)               # grass only
    c3 = mod.run_site(driver_nc, "prognostic", str(tmp_path / "c3"), chunk=96,
                      vcmax_c3_scale=1.6)               # trees only
    assert c4["GPP"]["bias"] > base["GPP"]["bias"]      # grass tuning raises GPP
    assert c3["GPP"]["bias"] > base["GPP"]["bias"]      # tree  tuning raises GPP
    # Per-pathway flag wins over the combined knob: c4-only != both-scaled.
    both = mod.run_site(driver_nc, "prognostic", str(tmp_path / "bo"), chunk=96,
                        vcmax_scale=1.6)
    assert both["GPP"]["bias"] > c4["GPP"]["bias"]      # scaling both > grass alone


def test_grass_tree_knob_reaches_diagnostic_mode(tmp_path):
    """The C3/C4 knobs apply in diagnostic mode too (not prognostic-only)."""
    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5)
    mod = _load_driver_module()
    base = mod.run_site(driver_nc, "diagnostic", str(tmp_path / "b"), chunk=96)
    hi = mod.run_site(driver_nc, "diagnostic", str(tmp_path / "h"), chunk=96,
                      vcmax_c4_scale=1.6)
    assert hi["GPP"]["bias"] != base["GPP"]["bias"]


def _single_step_canopy_inputs(driver_nc):
    """Read a driver and return the (T_soil, forcing, params, w_frac, wind) tuple
    for one daytime timestep (ncol=1), mirroring run_ec_site._diagnostic_fluxes."""
    import jax
    import jax.numpy as jnp
    from legoesm.land.boundary_data.ec_site import read_ec_site_driver
    d = read_ec_site_driver(driver_nc)
    lai = np.asarray(d.canopy_params.LAI).ravel()
    sw = np.asarray(d.forcing.sw_down).ravel()
    t = int(np.argmax((sw > 100.0) & (lai > 0.1)))     # a lit, leafy step
    tree = lambda x: jax.tree_util.tree_map(lambda a: a[t], x)
    f, p = tree(d.forcing), tree(d.canopy_params)
    wind = jnp.sqrt(f.u_lowest ** 2 + f.v_lowest ** 2 + 1.0)
    return d.T_soil_top[t], f, p, d.w_frac_rz[t], wind, d.dt_s


def test_mosaic_single_patch_identical_to_canopy(tmp_path):
    """N=1 mosaic (frac=1, unit scales) is bit-identical to the direct canopy."""
    import jax.numpy as jnp
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)
    from legoesm.land.surface_scheme.patch_mosaic import (
        PatchMosaicConfig, PatchSpec, compute_mosaic_canopy_fluxes)

    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5)
    Ts, f, p, wf, wind, dt = _single_step_canopy_inputs(driver_nc)
    cc, lc = TwoLeafCanopyConfig(max_iters=30), MultiLayerLandConfig()
    kw = dict(T_soil_top=Ts, forcing=f, canopy_config=cc, land_config=lc,
              canopy_params=p, w_frac_rz=wf, wind_speed=wind,
              wind_dir_x=jnp.ones_like(wind), wind_dir_y=jnp.zeros_like(wind),
              soil_thermal_fn=lambda G, dt_: Ts, dt=dt,
              LAI_override=p.LAI, TgC_override=p.TgC)
    direct = compute_two_leaf_canopy_fluxes(**kw)
    mono = compute_mosaic_canopy_fluxes(
        mosaic=PatchMosaicConfig(patches=(PatchSpec(frac=1.0),)).validate(), **kw)
    for fld in ("gpp", "lhflx", "shflx", "T_surface"):
        a, b = getattr(direct, fld), getattr(mono, fld)
        np.testing.assert_allclose(np.asarray(a).ravel(), np.asarray(b).ravel(),
                                   rtol=1e-12, atol=0.0, err_msg=fld)


def test_mosaic_two_identical_patches_equal_single(tmp_path):
    """Two identical patches (any area split) reduce to the single-canopy value."""
    import jax.numpy as jnp
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.patch_mosaic import (
        PatchMosaicConfig, PatchSpec, compute_mosaic_canopy_fluxes)

    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5)
    Ts, f, p, wf, wind, dt = _single_step_canopy_inputs(driver_nc)
    cc, lc = TwoLeafCanopyConfig(max_iters=30), MultiLayerLandConfig()
    kw = dict(T_soil_top=Ts, forcing=f, canopy_config=cc, land_config=lc,
              canopy_params=p, w_frac_rz=wf, wind_speed=wind,
              wind_dir_x=jnp.ones_like(wind), wind_dir_y=jnp.zeros_like(wind),
              soil_thermal_fn=lambda G, dt_: Ts, dt=dt,
              LAI_override=p.LAI, TgC_override=p.TgC)
    one = compute_mosaic_canopy_fluxes(
        mosaic=PatchMosaicConfig(patches=(PatchSpec(frac=1.0),)).validate(), **kw)
    two = compute_mosaic_canopy_fluxes(
        mosaic=PatchMosaicConfig(patches=(PatchSpec(frac=0.3),
                                          PatchSpec(frac=0.7))).validate(), **kw)
    np.testing.assert_allclose(np.asarray(two.gpp).ravel(),
                               np.asarray(one.gpp).ravel(), rtol=1e-9, atol=0.0)


def test_mosaic_c3_tree_vs_c4_grass_patches_differ(tmp_path):
    """A C3-tree + C4-grass 2-patch mosaic gives a different GPP than the blended
    single canopy — the two-source split does something."""
    import jax.numpy as jnp
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.patch_mosaic import (
        PatchMosaicConfig, PatchSpec, compute_mosaic_canopy_fluxes, savanna_two_patch)

    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5)
    Ts, f, p, wf, wind, dt = _single_step_canopy_inputs(driver_nc)
    cc, lc = TwoLeafCanopyConfig(max_iters=30), MultiLayerLandConfig()
    kw = dict(T_soil_top=Ts, forcing=f, canopy_config=cc, land_config=lc,
              canopy_params=p, w_frac_rz=wf, wind_speed=wind,
              wind_dir_x=jnp.ones_like(wind), wind_dir_y=jnp.zeros_like(wind),
              soil_thermal_fn=lambda G, dt_: Ts, dt=dt,
              LAI_override=p.LAI, TgC_override=p.TgC)
    blended = compute_mosaic_canopy_fluxes(
        mosaic=PatchMosaicConfig(patches=(PatchSpec(frac=1.0),)).validate(), **kw)
    savanna = compute_mosaic_canopy_fluxes(mosaic=savanna_two_patch(tree_frac=0.4), **kw)
    assert np.isfinite(np.asarray(savanna.gpp)).all()
    assert float(savanna.gpp[0]) != float(blended.gpp[0])


def test_mosaic_compute_api_validates_at_entry(tmp_path):
    """compute_mosaic_canopy_fluxes rejects a bad mosaic (fractions sum != 1)
    before running the canopy, not silently returning corrupted fluxes."""
    import jax.numpy as jnp
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.patch_mosaic import (
        PatchMosaicConfig, PatchSpec, compute_mosaic_canopy_fluxes)

    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5, n=8)
    Ts, f, p, wf, wind, dt = _single_step_canopy_inputs(driver_nc)
    bad = PatchMosaicConfig(patches=(PatchSpec(frac=0.4), PatchSpec(frac=0.4)))
    with pytest.raises(ValueError):
        compute_mosaic_canopy_fluxes(
            mosaic=bad, T_soil_top=Ts, forcing=f,
            canopy_config=TwoLeafCanopyConfig(max_iters=30),
            land_config=MultiLayerLandConfig(), canopy_params=p, w_frac_rz=wf,
            wind_speed=wind, wind_dir_x=jnp.ones_like(wind),
            wind_dir_y=jnp.zeros_like(wind), soil_thermal_fn=lambda G, dt_: Ts,
            dt=dt, LAI_override=p.LAI, TgC_override=p.TgC)


def test_clmml_mosaic_orchestration_area_weights(monkeypatch):
    """The CLM-ML outer-loop mosaic runs one prognostic column PER TILE with the
    tile's PFT + root depth, and area-weights the flux series.  The (pre-existing,
    synthetic-driver) CLM-ML solve is stubbed so this checks MY orchestration
    deterministically: per-tile config wiring + area-weighting."""
    import numpy as np
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.surface_scheme.patch_mosaic import savanna_clmml_two_patch
    mod = _load_driver_module()

    seen = []
    def _stub(d, cc, lc, u_min, nudge_tau_days=0.0, clmml_sai=0.5):
        assert isinstance(cc, CLMMLCanopyConfig)
        seen.append((int(cc.pft_clm), float(lc.root_depth)))
        k = len(seen)                                   # 1st tile -> 1s, 2nd -> 2s
        base = np.full(4, float(k))
        rev = np.array([0, k - 1, 0, 0])                # tile2 reverts step 1
        # (gpp, le, h, T_surface, reverted, ts_soil, swc_soil, ustar)
        return base, base * 10, base * 100, base + 290, rev, base + 280, base / 10, base / 20

    monkeypatch.setattr(mod, "_prognostic_fluxes", _stub)
    m = savanna_clmml_two_patch(tree_frac=0.4, tree_pft=7, grass_pft=15,
                                tree_root_m=5.0, grass_root_m=0.5)
    gpp, le, h, ts, reverted, ts_soil, swc, ustar = mod._clmml_mosaic_prognostic(
        None, m, soil="default", bottom_bc="free_drainage", soil_depth_m=0.0,
        k_sat_decay_m=0.0, soil_evap_resistance_exp=2.0, z_ref=10.0, texture=None,
        interception=False, plant_wilting_point=None, clmml_turbulence="most",
        clmml_stomatal="wue", u_min=1.0, nudge_tau_days=0.0, clmml_sai=0.5)
    # per-tile config wiring: tree (pft 7, root 5) then grass (pft 15, root 0.5)
    assert seen == [(7, 5.0), (15, 0.5)]
    # area-weight: 0.4*tile1 + 0.6*tile2 (tile1 val=1, tile2 val=2)
    np.testing.assert_allclose(gpp, np.full(4, 0.4 * 1.0 + 0.6 * 2.0))
    np.testing.assert_allclose(le, np.full(4, 0.4 * 10.0 + 0.6 * 20.0))
    np.testing.assert_allclose(h, np.full(4, 0.4 * 100.0 + 0.6 * 200.0))
    # soil-state / ustar diagnostics area-weighted (schema preserved, not dropped)
    np.testing.assert_allclose(ts_soil, np.full(4, 0.4 * 281.0 + 0.6 * 282.0))
    np.testing.assert_allclose(swc, np.full(4, 0.4 * 0.1 + 0.6 * 0.2))
    np.testing.assert_allclose(ustar, np.full(4, 0.4 * 0.05 + 0.6 * 0.1))
    # reverted = per-step any-tile max (tile2 reverted step 1)
    np.testing.assert_array_equal(reverted, np.array([0, 1, 0, 0]))


def test_two_leaf_mosaic_prognostic_orchestration(monkeypatch, tmp_path):
    """Two-leaf outer-loop mosaic runs one prognostic column per tile with the
    tile's ROOT DEPTH (deep tree vs shallow grass) and area-weights.  Stubs the
    solve to check per-tile root-depth threading + area-weighting deterministically."""
    import numpy as np
    from legoesm.land.surface_scheme.patch_mosaic import savanna_two_patch
    mod = _load_driver_module()
    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5, n=8)
    from legoesm.land.boundary_data.ec_site import read_ec_site_driver
    d = read_ec_site_driver(driver_nc)
    seen = []
    def _stub(dd, cc, lc, u_min, nudge_tau_days=0.0, clmml_sai=0.5):
        seen.append(round(float(lc.root_depth), 3))
        k = len(seen)
        base = np.full(4, float(k))
        return base, base * 10, base * 100, base + 290, np.zeros(4), base, base, base
    monkeypatch.setattr(mod, "_prognostic_fluxes", _stub)
    m = savanna_two_patch(tree_frac=0.4, grass_fc4=0.0, tree_root_m=5.0,
                          grass_root_m=0.5)
    gpp, le, h, ts, rev, tsoil, swc, ustar = mod._two_leaf_mosaic_prognostic(
        d, m, soil="default", bottom_bc="free_drainage", soil_depth_m=0.0,
        k_sat_decay_m=0.0, soil_evap_resistance_exp=2.0, z_ref=10.0, texture=None,
        interception=False, plant_wilting_point=None, stress_b0=False,
        root_depth=2.0, u_min=1.0, nudge_tau_days=0.0)
    assert seen == [5.0, 0.5]                       # tree deep, grass shallow
    np.testing.assert_allclose(gpp, np.full(4, 0.4 * 1.0 + 0.6 * 2.0))
    np.testing.assert_allclose(h, np.full(4, 0.4 * 100.0 + 0.6 * 200.0))


def test_run_site_savanna_mosaic_diagnostic(tmp_path):
    """--mosaic savanna runs end-to-end in diagnostic mode with finite skill."""
    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5)
    mod = _load_driver_module()
    res = mod.run_site(driver_nc, "diagnostic", str(tmp_path / "m"), chunk=96,
                       mosaic="savanna", tree_frac=0.4)
    assert np.isfinite(res["GPP"]["bias"])


def test_run_site_mosaic_rejects_bad_combos(tmp_path):
    driver_nc = str(tmp_path / "SAV-Test_driver_v2.nc")
    _make_driver(driver_nc, igbp=8.0, fc4=0.5, n=8)
    mod = _load_driver_module()
    with pytest.raises(ValueError):                      # unknown mosaic
        mod.run_site(driver_nc, "diagnostic", str(tmp_path / "a"), chunk=8,
                     mosaic="bogus")
    with pytest.raises(ValueError):                      # clmml + diagnostic
        mod.run_site(driver_nc, "diagnostic", str(tmp_path / "c"), chunk=8,
                     mosaic="savanna", canopy="clmml")
    # (two_leaf + prognostic + savanna is now VALID — the outer-loop mosaic.)


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
