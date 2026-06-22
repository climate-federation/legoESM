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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
