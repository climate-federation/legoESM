"""CLM surfdata → PFT + soil land-params (synthetic file, no network)."""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.clm_surface_map import (
    load_clm_surface, CLMSurfaceParamProvider, _N_PFT)
from legoesm.land.surface_params import CLM5_PFT_NAMES, LandSurfaceParams


def _write_synthetic_surfdata(path, nlat=6, nlon=8, n_nat=15, nlev=10):
    import xarray as xr
    lat = np.linspace(85, -85, nlat); lon = np.linspace(0, 315, nlon)
    LO, LA = np.meshgrid(lon, lat)
    rng = np.random.default_rng(0)
    nat = rng.uniform(0, 1, (n_nat, nlat, nlon)); nat /= nat.sum(0, keepdims=True); nat *= 100
    ds = xr.Dataset(
        {"LATIXY": (("lsmlat", "lsmlon"), LA),
         "LONGXY": (("lsmlat", "lsmlon"), LO),
         "PCT_NAT_PFT": (("natpft", "lsmlat", "lsmlon"), nat),
         "PCT_NATVEG": (("lsmlat", "lsmlon"), np.full((nlat, nlon), 80.0)),
         "PCT_CROP": (("lsmlat", "lsmlon"), np.full((nlat, nlon), 10.0)),
         "PCT_GLACIER": (("lsmlat", "lsmlon"),
                         np.where(LA > 70, 100.0, 0.0)),   # ice sheet poleward of 70

         "PCT_SAND": (("nlevsoi", "lsmlat", "lsmlon"),
                      rng.uniform(20, 80, (nlev, nlat, nlon))),
         "PCT_CLAY": (("nlevsoi", "lsmlat", "lsmlon"),
                      rng.uniform(5, 40, (nlev, nlat, nlon)))})
    ds.to_netcdf(path)


def test_load_and_provider(tmp_path):
    f = str(tmp_path / "surf.nc")
    _write_synthetic_surfdata(f)
    # target grid (coarser, shifted) -> exercises nearest-neighbour regrid
    lat = np.linspace(80, -80, 5); lon = np.linspace(10, 300, 6)
    LO, LA = np.meshgrid(lon, lat)
    m = load_clm_surface(f, LA.ravel(), LO.ravel())
    ncol = 30

    fr = np.asarray(m["pft_fractions"])
    assert fr.shape == (ncol, _N_PFT)
    assert np.allclose(fr.sum(1), 1.0, atol=1e-6)        # normalised per column
    assert np.all(fr >= 0.0)
    assert np.all(np.asarray(m["theta_fc"]) > np.asarray(m["theta_wp"]))

    prov = CLMSurfaceParamProvider(m["pft_fractions"], m["theta_wp"], m["theta_fc"],
                                   m["glacier_frac"])
    p = prov()
    assert isinstance(p, LandSurfaceParams)
    assert p.albedo_veg.shape == (ncol,)
    # PFT-weighted params stay within the CLM5 table envelope, EXCEPT glacier cells
    # whose albedo is raised toward ice (Greenland fix). Non-glacier in [0.05,0.40].
    nonglac = np.asarray(m["glacier_frac"]) < 0.5
    assert float(jnp.min(p.albedo_veg)) >= 0.05
    assert np.all(np.asarray(p.albedo_veg)[nonglac] <= 0.40 + 1e-9)
    assert float(jnp.min(p.Vc_max25)) >= 0.0
    # soil props came from the reference map, not the PFT table
    assert np.allclose(np.asarray(p.theta_wp), np.asarray(m["theta_wp"]))
    assert np.allclose(np.asarray(p.theta_fc), np.asarray(m["theta_fc"]))


def test_crop_fraction_goes_to_crop_slot(tmp_path):
    f = str(tmp_path / "s2.nc")
    _write_synthetic_surfdata(f)
    lat = np.linspace(80, -80, 4); lon = np.linspace(0, 270, 4)
    LO, LA = np.meshgrid(lon, lat)
    m = load_clm_surface(f, LA.ravel(), LO.ravel())
    fr = np.asarray(m["pft_fractions"])
    i_crop = CLM5_PFT_NAMES.index("crop_c3")
    assert np.all(fr[:, i_crop] > 0.0)                    # 10% crop everywhere


def test_glacier_raises_albedo(tmp_path):
    """Glacier (ice-sheet) cells get a bright ice base albedo (Greenland fix),
    well above any vegetated PFT albedo; non-glacier cells unchanged."""
    from legoesm.land.clm_surface_map import (
        load_clm_surface, CLMSurfaceParamProvider, _GLACIER_ALBEDO)
    f = str(tmp_path / "g.nc")
    _write_synthetic_surfdata(f)
    lat = np.linspace(85, -85, 8); lon = np.linspace(0, 315, 8)
    LO, LA = np.meshgrid(lon, lat)
    m = load_clm_surface(f, LA.ravel(), LO.ravel())
    p = CLMSurfaceParamProvider(m["pft_fractions"], m["theta_wp"], m["theta_fc"],
                                m["glacier_frac"])()
    fg = np.asarray(m["glacier_frac"]); alb = np.asarray(p.albedo_veg)
    full_ice = fg > 0.99
    assert full_ice.any()
    assert np.allclose(alb[full_ice], _GLACIER_ALBEDO, atol=1e-6)   # 100% glacier -> ice albedo
    assert np.all(alb[fg < 0.01] < _GLACIER_ALBEDO)                 # non-glacier darker


def test_tuned_variant_selection(tmp_path):
    """variant='multilayer' bakes its own per-PFT albedo + glacier ice base; an
    unknown variant raises (dispatch hardening)."""
    from legoesm.land.clm_surface_map import (
        load_clm_surface, CLMSurfaceParamProvider, _VARIANT_TUNED,
        TUNED_GLACIER_ALBEDO, TUNED_GLACIER_ALBEDO_MULTILAYER)
    f = str(tmp_path / "v.nc")
    _write_synthetic_surfdata(f)
    lat = np.linspace(85, -85, 8); lon = np.linspace(0, 315, 8)
    LO, LA = np.meshgrid(lon, lat)
    m = load_clm_surface(f, LA.ravel(), LO.ravel())
    args = (m["pft_fractions"], m["theta_wp"], m["theta_fc"], m["glacier_frac"])
    slab = CLMSurfaceParamProvider(*args, variant="slab")()
    mult = CLMSurfaceParamProvider(*args, variant="multilayer")()
    # the two calibrations differ on the vegetated albedo
    nonglac = np.asarray(m["glacier_frac"]) < 0.01
    assert not np.allclose(np.asarray(slab.albedo_veg)[nonglac],
                           np.asarray(mult.albedo_veg)[nonglac])
    # multilayer stays physical
    assert float(jnp.min(mult.albedo_veg)) >= 0.05
    assert np.all(np.asarray(mult.albedo_veg)[nonglac] <= 0.45)
    # glacier base albedo follows the variant
    full_ice = np.asarray(m["glacier_frac"]) > 0.99
    assert np.allclose(np.asarray(slab.albedo_veg)[full_ice], TUNED_GLACIER_ALBEDO, atol=1e-6)
    assert np.allclose(np.asarray(mult.albedo_veg)[full_ice],
                       TUNED_GLACIER_ALBEDO_MULTILAYER, atol=1e-6)
    assert set(_VARIANT_TUNED) == {"slab", "multilayer"}
    with pytest.raises(ValueError, match="unknown tuned variant"):
        CLMSurfaceParamProvider(*args, variant="bogus")


def test_multilayer_extended_bake(tmp_path):
    """The multilayer variant overrides theta_wp/fc with the per-PFT PLANT btran
    thresholds (slab keeps the soil map), and the per-cell Ch + thermal helpers are
    PFT-weighted from the calibrated tables."""
    from legoesm.land.clm_surface_map import (
        load_clm_surface, CLMSurfaceParamProvider, clm_multilayer_ch,
        clm_multilayer_thermal_config, _TUNED_PFT_CH_MULTILAYER)
    f = str(tmp_path / "e.nc")
    _write_synthetic_surfdata(f)
    lat = np.linspace(80, -80, 5); lon = np.linspace(10, 300, 6)
    LO, LA = np.meshgrid(lon, lat)
    m = load_clm_surface(f, LA.ravel(), LO.ravel())
    args = (m["pft_fractions"], m["theta_wp"], m["theta_fc"], m["glacier_frac"])
    slab = CLMSurfaceParamProvider(*args, variant="slab")()
    mult = CLMSurfaceParamProvider(*args, variant="multilayer")()
    # slab theta_wp == soil map; multilayer == PFT-weighted plant thresholds (differ)
    assert np.allclose(np.asarray(slab.theta_wp), np.asarray(m["theta_wp"]))
    assert not np.allclose(np.asarray(mult.theta_wp), np.asarray(m["theta_wp"]))
    assert np.all(np.asarray(mult.theta_fc) > np.asarray(mult.theta_wp))   # range valid
    # per-cell Ch + thermal: PFT-weighted, physical
    ch = np.asarray(clm_multilayer_ch(m))
    assert ch.shape == (30,) and np.all((ch >= 2e-3 - 1e-9) & (ch <= 6e-3 + 1e-9))
    th = clm_multilayer_thermal_config(m)
    assert np.asarray(th.C_soil).shape == (30, 1)
    assert np.all(np.asarray(th.C_soil) > 0) and np.all(np.asarray(th.k_solid) > 0)


def test_clm_multilayer_setup_builds_params_and_config(tmp_path):
    """clm_multilayer_setup composes the faithful CLM default multilayer land:
    per-column LandSurfaceParams + a MultiLayerLandConfig whose hydraulics/thermal
    carry the spatial maps while the non-spatial sub-configs come from base."""
    from legoesm.land.clm_surface_map import load_clm_surface, clm_multilayer_setup
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.surface_params import LandSurfaceParams

    f = str(tmp_path / "s.nc")
    _write_synthetic_surfdata(f)
    lat = np.linspace(70, -70, 4); lon = np.linspace(20, 300, 5)
    LO, LA = np.meshgrid(lon, lat)
    m = load_clm_surface(f, LA.ravel(), LO.ravel())
    ncol = LA.size

    base = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=6, total_depth=2.5))
    params, cfg = clm_multilayer_setup(m, base_config=base)

    # params: per-column LandSurfaceParams, physical albedo/emissivity ranges
    assert isinstance(params, LandSurfaceParams)
    assert np.asarray(params.albedo_veg).shape == (ncol,)
    assert np.all((np.asarray(params.albedo_veg) >= 0.0)
                  & (np.asarray(params.albedo_veg) <= 1.0))
    assert np.all((np.asarray(params.emissivity) > 0.8)
                  & (np.asarray(params.emissivity) <= 1.0))
    # config: base's soil grid survives; hydraulics/thermal are the spatial maps
    assert isinstance(cfg, MultiLayerLandConfig)
    assert cfg.soil_grid.n_layers == 6 and cfg.soil_grid.total_depth == 2.5
    assert np.asarray(cfg.thermal.C_soil).shape == (ncol, 1)
    assert np.all(np.asarray(cfg.thermal.C_soil) > 0)
    assert np.all(np.asarray(cfg.thermal.k_solid) > 0)
    assert np.asarray(cfg.hydraulics.theta_sat).shape == (ncol, 1)


def test_clm_multilayer_setup_preserves_surface_snow_stomata(tmp_path):
    """clm_multilayer_setup must PRESERVE the base's surface/snow/stomata features
    (it may overwrite ONLY hydraulics/thermal).  This is the contract the AMIP
    driver's _setup_multilayer_land relies on to thread surface_bulk_scheme +
    snow_albedo_feedback + land_stomatal_beta through, so use_multilayer_land is a
    strict upgrade of the slab rather than a constant-bulk/no-snow/no-stomata
    partial regression.  A silent reset here would re-introduce that regression."""
    from legoesm.land.clm_surface_map import load_clm_surface, clm_multilayer_setup
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.carbon.stomata import StomataConfig
    from legoesm.land.soil_grid import SoilGridConfig

    f = str(tmp_path / "s.nc")
    _write_synthetic_surfdata(f)
    lat = np.linspace(70, -70, 4); lon = np.linspace(20, 300, 5)
    LO, LA = np.meshgrid(lon, lat)
    m = load_clm_surface(f, LA.ravel(), LO.ravel())

    # exactly what _setup_multilayer_land builds: land MOST for the soil SEB
    # (matching the atmospheric land tile) + snow-albedo feedback + Jarvis stomata.
    base = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=6, total_depth=2.5),
        bulk_scheme="most",
        snow_albedo_feedback=True,
        stomata=StomataConfig(enabled=True),
    )
    _params, cfg = clm_multilayer_setup(m, base_config=base)

    assert cfg.bulk_scheme == "most", "surface bulk scheme was reset (regression)"
    assert cfg.snow_albedo_feedback is True, "snow-albedo feedback was reset"
    assert cfg.stomata.enabled is True, "stomata was reset (regression)"


def test_clm_provider_pft_weights_spatial_lai():
    """CLMSurfaceParamProvider maps per-PFT MONTHLY_LAI to a PFT-weighted per-cell
    LAI so the two-leaf canopy gets a spatial climatology, NOT a uniform scalar:
    a bare cell -> LAI ~ 0 (no spurious canopy), a forest cell -> the forest LAI."""
    fr = np.zeros((2, _N_PFT)); fr[0, 0] = 1.0; fr[1, 4] = 1.0   # cell0 bare, cell1 PFT-4
    lai_pft = np.zeros((_N_PFT, 2)); lai_pft[4, :] = 5.0          # PFT-4 -> LAI 5
    prov = CLMSurfaceParamProvider(
        jnp.asarray(fr), jnp.full(2, 0.1), jnp.full(2, 0.3), jnp.zeros(2),
        variant="multilayer", lai_pft=jnp.asarray(lai_pft))
    lai = np.asarray(prov().LAI)
    assert lai.shape == (2,)
    assert lai[0] < 0.01,      f"bare cell should have LAI~0, got {lai[0]}"
    assert 4.5 < lai[1] < 5.5, f"forest cell should have LAI~5, got {lai[1]}"


def test_clm_provider_lai_none_without_climatology():
    """No lai_pft (surfdata lacks MONTHLY_LAI) -> LAI stays None; the two-leaf
    canopy's None-safe read then falls back to its scalar default."""
    prov = CLMSurfaceParamProvider(
        jnp.full((2, _N_PFT), 1.0 / _N_PFT), jnp.full(2, 0.1), jnp.full(2, 0.3),
        jnp.zeros(2), variant="multilayer")
    assert prov().LAI is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
