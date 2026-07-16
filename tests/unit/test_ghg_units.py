"""Regression: external GHG annual-file units must reach radiation correctly.

Bug (2026-06-11): the CMIP6/input4MIPs GHG file stores DIMENSIONLESS mole
fractions (units="1", CO2 ≈ 3.37e-4 mol/mol).  ``get_ghg_at_time`` returned
the raw mole fraction labelled "co2_ppmv", and ``ghg_concentrations_to_vmr``
then multiplied by 1e-6, so radiation saw CO2 ≈ 3.4e-10 vmr — effectively
zero CO2 (a ~1 K/day spurious longwave cooling in AMIP runs).
"""

import numpy as np
import pytest
import xarray as xr

from legoesm.forcing.external import (
    GHGConfig,
    get_ghg_at_time,
    ghg_concentrations_to_vmr,
    _load_ghg_annual_file,
)


def _write_ghg(path, *, units, co2, ch4, n2o, cfc11, cfc12):
    ds = xr.Dataset(
        {
            "CO2": (("time", "lat", "lon"), np.full((2, 1, 1), co2)),
            "CH4": (("time", "lat", "lon"), np.full((2, 1, 1), ch4)),
            "N2O": (("time", "lat", "lon"), np.full((2, 1, 1), n2o)),
            "CFC_11": (("time", "lat", "lon"), np.full((2, 1, 1), cfc11)),
            "CFC_12": (("time", "lat", "lon"), np.full((2, 1, 1), cfc12)),
        },
        coords={"time": [1979.0, 1980.0], "lat": [0.0], "lon": [0.0]},
    )
    for v in ("CO2", "CH4", "N2O", "CFC_11", "CFC_12"):
        ds[v].attrs["units"] = units
    ds.to_netcdf(path)


def test_mole_fraction_file_gives_physical_vmr(tmp_path):
    # input4MIPs convention: units="1", mole fractions.
    p = tmp_path / "ghg_molefrac.nc"
    _write_ghg(p, units="1", co2=336.8e-6, ch4=1550e-9, n2o=301e-9,
               cfc11=170e-12, cfc12=300e-12)
    cfg = GHGConfig(source="annual_file", path=str(p), start_year=1979)
    ghg = get_ghg_at_time(cfg, 0.0)
    assert ghg["co2_ppmv"] == pytest.approx(336.8, rel=1e-3)
    assert ghg["ch4_ppbv"] == pytest.approx(1550.0, rel=1e-3)
    vmr = ghg_concentrations_to_vmr(ghg)
    # CO2 must reach radiation as ~3.37e-4, NOT 3.4e-10.
    assert vmr["co2"] == pytest.approx(3.368e-4, rel=1e-3)
    assert vmr["ch4"] == pytest.approx(1.55e-6, rel=1e-3)
    assert vmr["n2o"] == pytest.approx(3.01e-7, rel=1e-3)


def test_all_species_molefraction_roundtrip(tmp_path):
    # Every species (incl. CFC-11/12 pptv) must reach radiation at its
    # physical VMR from a units="1" mole-fraction file.
    p = tmp_path / "ghg_all.nc"
    _write_ghg(p, units="1", co2=400e-6, ch4=1800e-9, n2o=330e-9,
               cfc11=240e-12, cfc12=530e-12)
    cfg = GHGConfig(source="annual_file", path=str(p), start_year=1979)
    ghg = get_ghg_at_time(cfg, 0.0)
    assert ghg["co2_ppmv"] == pytest.approx(400.0, rel=1e-3)
    assert ghg["ch4_ppbv"] == pytest.approx(1800.0, rel=1e-3)
    assert ghg["n2o_ppbv"] == pytest.approx(330.0, rel=1e-3)
    assert ghg["cfc11_pptv"] == pytest.approx(240.0, rel=1e-3)
    assert ghg["cfc12_pptv"] == pytest.approx(530.0, rel=1e-3)
    vmr = ghg_concentrations_to_vmr(ghg)
    assert vmr["cfc11"] == pytest.approx(240e-12, rel=1e-3)
    assert vmr["cfc12"] == pytest.approx(530e-12, rel=1e-3)


def test_nmol_per_mol_units(tmp_path):
    # CH4 in nmol/mol (~1800) must NOT trip the magnitude heuristic.
    p = tmp_path / "ghg_nmol.nc"
    _write_ghg(p, units="nmol/mol", co2=400e3, ch4=1800.0, n2o=330.0,
               cfc11=0.24, cfc12=0.53)
    # CO2 in nmol/mol is unusual but the per-variable units attr drives it.
    years, data = _load_ghg_annual_file(str(p))
    assert float(data["CH4"][0]) == pytest.approx(1800e-9, rel=1e-3)


def test_lru_cache_arrays_readonly(tmp_path):
    p = tmp_path / "ghg_ro.nc"
    _write_ghg(p, units="1", co2=336.8e-6, ch4=1550e-9, n2o=301e-9,
               cfc11=170e-12, cfc12=300e-12)
    _, data = _load_ghg_annual_file(str(p))
    with pytest.raises((ValueError, RuntimeError)):
        data["CO2"][0] = 999.0  # cached array must be read-only


def test_empty_units_uses_heuristic_not_molefraction(tmp_path):
    # A no-units file with CO2 ~ 400 (ppmv-valued) must be heuristic-
    # scaled to mole fraction, NOT assumed already mol/mol.
    p = tmp_path / "ghg_nounits.nc"
    ds = xr.Dataset(
        {
            "CO2": (("time", "lat", "lon"), np.full((2, 1, 1), 400.0)),
            "CH4": (("time", "lat", "lon"), np.full((2, 1, 1), 1800.0)),
            "N2O": (("time", "lat", "lon"), np.full((2, 1, 1), 330.0)),
            "CFC_11": (("time", "lat", "lon"), np.full((2, 1, 1), 240.0)),
            "CFC_12": (("time", "lat", "lon"), np.full((2, 1, 1), 530.0)),
        },
        coords={"time": [1979.0, 1980.0], "lat": [0.0], "lon": [0.0]},
    )
    ds.to_netcdf(p)  # no units attribute
    years, data = _load_ghg_annual_file(str(p))
    # 400 ppmv-valued → heuristic ×1e-6 → 4e-4 mole fraction.
    assert float(data["CO2"][0]) == pytest.approx(4.0e-4, rel=1e-3)


def test_ppmv_file_passes_through(tmp_path):
    # Pre-scaled file: units="1e-6"/"ppmv" etc. — loader normalizes to
    # mole fraction, getter converts back; same physical VMR.
    p = tmp_path / "ghg_ppmv.nc"
    _write_ghg(p, units="1e-6", co2=336.8, ch4=1550e3, n2o=301e3,
               cfc11=170e6, cfc12=300e6)
    # NOTE: CH4/N2O/CFC in "1e-6"-file would need their own scale; here we
    # only assert CO2 round-trips, since real files use consistent per-var
    # units handled by the attribute table.
    years, data = _load_ghg_annual_file(str(p))
    # Loader returns mole fraction: 336.8 * 1e-6 = 3.368e-4.
    assert float(data["CO2"][0]) == pytest.approx(3.368e-4, rel=1e-3)


def test_real_synthetic_deck_file():
    # The committed synthetic deck file must yield physical CO2.
    import os
    f = "data/forcing_amip/ghg_amip_1979-1981.nc"
    if not os.path.exists(f):
        pytest.skip("synthetic deck not generated")
    cfg = GHGConfig(source="annual_file", path=f, start_year=1979)
    vmr = ghg_concentrations_to_vmr(get_ghg_at_time(cfg, 0.0))
    assert 3.0e-4 < vmr["co2"] < 3.5e-4, f"CO2 vmr {vmr['co2']:.2e} unphysical"
