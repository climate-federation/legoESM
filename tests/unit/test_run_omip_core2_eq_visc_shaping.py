"""Equatorial A_h shaping through the CORE-II driver: ``--A-h-eq-boost`` now
accepts REDUCTIONS (< 1, the NEMO ORCA1 eddy_viscosity_3D shape: 20000 m2/s
midlatitude -> 1000 at the equator) and ``--A-h-eq-sigma-deg`` sets the ramp
width.  Covers the argparse round-trip and the ``replace_flat`` threading into
the nested LateralViscosityConfig — a flag that parses but never reaches the
config is the silently-ignored-flag footgun.
"""

from __future__ import annotations

from scripts.run.run_omip_core2 import _build_arg_parser
from legoesm.ocean.state import LatLonCGridOceanConfig


def test_cli_round_trip():
    p = _build_arg_parser()
    a = p.parse_args([])
    assert a.A_h_eq_boost is None and a.A_h_eq_sigma_deg is None
    a = p.parse_args(["--A-h-eq-boost", "0.05", "--A-h-eq-sigma-deg", "7.0"])
    assert a.A_h_eq_boost == 0.05
    assert a.A_h_eq_sigma_deg == 7.0


def test_replace_flat_routes_reduction_into_lateral_viscosity():
    cfg = LatLonCGridOceanConfig()
    out = cfg.replace_flat(A_h=20000.0, A_h_eq_boost=0.05,
                           A_h_eq_sigma_deg=7.0)
    assert out.lateral_viscosity.A_h == 20000.0
    assert out.lateral_viscosity.A_h_eq_boost == 0.05
    assert out.lateral_viscosity.A_h_eq_sigma_deg == 7.0


def test_tke_lc_etau_round_trip_and_choices():
    """fesom-mimic card knobs (2026-08-18): parse, and reject junk."""
    import pytest
    p = _build_arg_parser()
    a = p.parse_args([])
    assert a.tke_lc is None and a.tke_etau is None
    a = p.parse_args(["--tke-lc", "off", "--tke-etau", "none"])
    assert a.tke_lc == "off" and a.tke_etau == "none"
    with pytest.raises(SystemExit):
        p.parse_args(["--tke-lc", "maybe"])
    with pytest.raises(SystemExit):
        p.parse_args(["--tke-etau", "surface"])


def test_tke_lc_etau_reach_the_card():
    """The overrides must land in the TKEConfig (silently-ignored-flag guard)."""
    from scripts.run.run_omip_core2 import orca1_zdftke_config
    base = orca1_zdftke_config()          # returns the TKEConfig itself
    assert base.lc is True and base.etau_mode == "below_ml"
    off = orca1_zdftke_config(lc=False, etau_mode="none")
    assert off.lc is False and off.etau_mode == "none"
    import pytest
    with pytest.raises(ValueError):
        orca1_zdftke_config(etau_mode="surface")


def test_shear_now2_variant_accepted_and_reaches_card():
    """nemo_face_native_now2 (the key_RK3-oracle spatial variant) must parse,
    validate, and land in TKEConfig; junk still raises."""
    import pytest
    p = _build_arg_parser()
    a = p.parse_args(["--tke-shear-production", "nemo_face_native_now2"])
    assert a.tke_shear_production == "nemo_face_native_now2"
    from scripts.run.run_omip_core2 import orca1_zdftke_config
    cfg = orca1_zdftke_config(shear_production="nemo_face_native_now2")
    assert cfg.tke_shear_production == "nemo_face_native_now2"
    with pytest.raises(ValueError):
        orca1_zdftke_config(shear_production="face_native")


def test_ah_profile_file_round_trip_and_helper(tmp_path):
    """--A-h-profile-file parses; the helper reproduces the file's shape as
    ratios (equator ~1000/20000, midlat ~1); replace_flat routes the tuple."""
    import numpy as np
    import netCDF4 as nc4
    p = _build_arg_parser()
    assert p.parse_args([]).A_h_profile_file is None
    a = p.parse_args(["--A-h-profile-file", "/x/eddy.nc"])
    assert a.A_h_profile_file == "/x/eddy.nc"

    # synthetic file: 20000 everywhere, a 1000 plateau within 4 deg of the
    # equator. Rows are 4 deg apart, so a NARROWER plateau would put the test
    # grid's +-1 deg centres on the interp ramp between 1000 and 20000
    # (ratio 0.29) -- the first version asserted <0.2 against exactly that
    # ramp value; the helper was right, the fixture was not.
    ny, nx = 41, 8
    lat = np.linspace(-80, 80, ny)[:, None] * np.ones((1, nx))
    ahm = np.full((1, 3, ny, nx), 20000.0)
    ahm[:, :, np.abs(lat[:, 0]) <= 4.0, :] = 1000.0
    f = tmp_path / "eddy.nc"
    ds = nc4.Dataset(f, "w")
    ds.createDimension("t", 1); ds.createDimension("z", 3)
    ds.createDimension("y", ny); ds.createDimension("x", nx)
    ds.createVariable("ahmf_3d", "f8", ("t", "z", "y", "x"))[:] = ahm
    ds.createVariable("nav_lat", "f8", ("y", "x"))[:] = lat
    ds.close()

    from scripts.run.run_omip_core2 import ah_profile_from_file
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(n_lat=90, n_lon=180)
    prof = ah_profile_from_file(g, str(f), 20000.0)
    prof = np.asarray(prof)
    lat_deg = np.degrees(np.asarray(g.lat))
    assert prof[np.argmin(np.abs(lat_deg))] < 0.1       # 0.05 on the plateau
    assert abs(prof[np.argmin(np.abs(lat_deg - 45))] - 1.0) < 0.05

    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig().replace_flat(
        A_h=20000.0, A_h_lat_profile=tuple(float(x) for x in prof))
    assert cfg.lateral_viscosity.A_h_lat_profile is not None
    assert len(cfg.lateral_viscosity.A_h_lat_profile) == 90
