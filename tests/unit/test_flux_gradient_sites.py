"""Known-answer checks for the multi-site flux-gradient analysis."""
import pathlib
import sys

import numpy as np
import pytest
import xarray as xr

from legoesm import constants

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import flux_gradient_sites as fs  # noqa: E402

KAPPA = constants.kappa_von_karman


def site(canopy=0.0, d=0.0, offset=0.0, h=-40.0):
    n, ustar = 200, 0.4
    z = np.tile(np.array([d + 2.0, d + 8.0, d + 32.0]), (n, 1))
    wind = ustar / KAPPA * np.log((z - d) / 0.01)
    t = 280.0
    rho_cp = 1e5 / (constants.R_d * t) * constants.c_pd
    theta_star = -(h / rho_cp) / ustar
    theta = t + theta_star / KAPPA * np.log((z - d) / 0.01)
    temp = theta - constants.g / constants.c_pd * z
    temp[:, 1] += offset
    f = lambda v: (("time", "lev_f"), np.full((n, 1), v))
    return xr.Dataset({"z_u": (("time", "lev_u"), z), "wind": (("time", "lev_u"), wind), "z_t": (("time", "lev_t"), z),
                       "temp": (("time", "lev_t"), temp), "z_f": f(3.0), "ustar": f(ustar), "h": f(h), "le": f(np.nan),
                       "t_flux": f(t), "pressure": ("time", np.full(n, 1e5))},
                      attrs={"canopy_height_m": canopy, "displacement_m": d})


def test_log_profile_gives_phi_one_for_both_kinds():
    s = fs.site_samples(site(), 0, 0.1, 0.2, False, False)
    for key, (_, zeta, phi) in s.items():
        assert np.nanmedian(phi) == pytest.approx(1.0, rel=1e-6), key
        assert np.nanmedian(zeta) > 0


def test_displacement_is_used_in_height_and_canopy_levels_dropped():
    s = fs.site_samples(site(canopy=12.0, d=10.0), 0, 0.1, 0.2, False, False)
    assert s and all(np.nanmedian(phi) == pytest.approx(1.0, rel=1e-6) for _, _, phi in s.values())
    assert list(s) == ["m:18-42", "h:18-42"], "the pair starting at 12 m is above; the 12 m level itself is at the top"


def test_spike_and_weak_wind_are_filtered():
    ds = site()
    ds.h.values[:5] = 44000.0
    ds.ustar.values[5:10] = 0.05
    _, zeta, _ = fs.site_samples(ds, 0, 0.1, 0.2, False, False)["m:2-8"]
    assert np.isnan(zeta[:10]).all() and np.isfinite(zeta[10:]).all()


def test_small_temperature_difference_is_dropped():
    s = fs.site_samples(site(h=-0.2), 0, 0.0, 0.2, False, False)
    assert np.isnan(s["h:2-8"][2]).all()


def test_neutral_offset_uses_only_near_zero_heat_flux():
    diff = np.r_[np.full(50, 0.3), np.full(50, 9.0)]
    h = np.r_[np.zeros(50), np.full(50, -40.0)]
    assert fs.neutral_offset(diff, h, np.full(100, 0.5)) == pytest.approx(0.3)
    assert np.isnan(fs.neutral_offset(diff[:5], h[:5], np.full(5, 0.5)))


def test_offset_correction_recovers_phi_through_the_pipeline():
    def both(offset):
        ds = xr.concat([site(h=-40.0, offset=offset), site(h=-0.5, offset=offset)], dim="time")
        return fs.site_samples(ds, 0, 0.1, 0.2, False, True)["h:2-8"][2][:200]

    assert np.nanmedian(both(0.5)) == pytest.approx(1.0, rel=0.03)
    assert np.nanmedian(both(0.0)) == pytest.approx(1.0, rel=0.03), "an unbiased profile keeps its gradient"


def test_arms_keep_the_same_sample_and_differ_only_by_the_correction():
    ds = xr.concat([site(h=-25.0, offset=0.1), site(h=-40.0, offset=0.1), site(h=-0.5, offset=0.1)], dim="time")
    raw, corrected = (fs.site_samples(ds, 0, 0.1, 0.2, False, c)["h:2-8"][2] for c in (False, True))
    assert np.isfinite(raw).any()
    assert (np.isfinite(raw) == np.isfinite(corrected)).all()


def test_site_heat_curve_leaves_out_near_neutral_but_pairs_keep_it():
    zeta = np.r_[np.full(50, 0.05), np.full(50, 0.3)]
    out = fs.summarise({"h:2-8": (4.0, zeta, np.ones(100)), "m:2-8": (4.0, zeta, np.ones(100))}, 30, 50)
    assert out["site"]["h"]["n"] == 50 and out["site"]["m"]["n"] == 100
    assert out["pairs"]["h:2-8"]["n"] == 100


def test_no_bin_mixes_stable_and_unstable():
    assert not ((fs.EDGES[:-1] < 0) & (fs.EDGES[1:] > 0)).any()


def test_excluded_sites_are_not_analysed():
    assert {"fino1", "ibox_kolsass", "ibox_flag1"} <= set(fs.EXCLUDED)


def test_nearest_flux_arm_has_no_offset_correction():
    ds = xr.concat([site(h=-40.0, offset=0.5), site(h=-0.5, offset=0.5)], dim="time")
    phi = fs.nearest_samples(ds, 0.1, 0.2)["h:2-8"][2][:200]
    assert abs(np.nanmedian(phi) - 1.0) > 0.1


def test_moisture_raises_buoyancy_flux():
    assert fs.buoyancy_flux(100.0, 100.0, 290.0, 1e5) > fs.buoyancy_flux(100.0, np.nan, 290.0, 1e5)


def test_nearest_and_lowest_flux_levels():
    ds = xr.Dataset({"z_f": (("time", "lev_f"), np.tile([60.0, 4.0, 25.0], (3, 1)))})
    assert fs.lowest_flux_level(ds) == 1 and fs.nearest_flux_level(ds, 30.0) == 2


def test_lowest_flux_level_is_above_the_canopy():
    ds = xr.Dataset({"z_f": (("time", "lev_f"), np.tile([1.5, 11.0, 29.0], (3, 1)))}, attrs={"canopy_height_m": 10.0})
    assert fs.lowest_flux_level(ds) == 1


def test_missing_displacement_uses_the_assumed_fraction_and_bare_site_is_zero():
    assert fs.displacement(xr.Dataset(attrs={"displacement_m": np.nan, "canopy_height_m": 18.0})) == pytest.approx(12.0)
    assert fs.displacement(xr.Dataset(attrs={"displacement_m": np.nan, "canopy_height_m": np.nan})) == 0.0
