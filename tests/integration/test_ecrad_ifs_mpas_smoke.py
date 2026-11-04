"""radiation='ecrad_ifs' on the MPAS column lane (smallest test mesh), with
IFS CLOUDSC + CUMASTRN as in the IFS deck, fed by the input4MIPs TSI, GHG and
ozone files the other lanes read (user 2026-11-04, votes M1-M10).

Pins: without those sources the build refuses (vote M7a); with them, at
rad_update_steps=2, a 1-step run uses only the full-call variant, a 3-step run
(full, RADHEATN-only, full) also traces the RADHEATN-only variant and ends on
a REFRESHED carry; both stay finite and change the state, and the values the
lane receives are the files' (RII0 ~ TSI x distance factor, 1979 CO2, ozone).
The real-feed tests skip when the staged files are absent."""
from __future__ import annotations

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

DT = 300.0
NLEV = 8
_FORCING = pathlib.Path(os.environ.get(
    "LEGOESM_CMIP6_FORCING_DIR",
    "/burg-archive/glab/users/pg2328/legoESM/data/cmip6_forcing"))
# the production feeds (config/amip/amip_production.ginsburg.sh)
_FEEDS = dict(
    ozone_forcing="external", ozone_file=str(_FORCING / (
        "vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_zonalmean_185001-201412.nc")),
    ghg_forcing="external", ghg_file=str(
        _FORCING / "ghg_cmip6_UoM-CMIP-1-2-0_halogens_0000-2014.nc"),
    solar_source="spectral_file", solar_file=str(
        _FORCING / "solar_cmip6_SOLARIS-HEPPA-3-2_mon.nc"),
    solar_tsi_var="TSI", solar_spectral_var="SSI_frac",
    solar_spectral_band_order="rrtmg_sw", start_year=1979,
    orbital_insolation=True)          # the production deck row
_needs_feeds = pytest.mark.skipif(
    not all(pathlib.Path(_FEEDS[k]).exists()
            for k in ("ozone_file", "ghg_file", "solar_file")),
    reason="input4MIPs forcing files not staged")


def _driver(tmp, n_steps, **feeds):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=NLEV,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=DT),
        output=OutputConfig(output_dir=str(tmp), diag_days=0, checkpoint_days=0),
        days=n_steps * DT / 86400.0, dataset="analytical", fix_moisture=True,
        precision="fp64",                # the IFS deck row (port is fp64)
        distributed=False, n_devices=1, turbulence="louis", radiation="ecrad_ifs",
        rad_update_steps=2, diurnal_cycle=True, convection="bechtold",
        bechtold_use_ifs_ascent=True, microphysics="ifs_cloudsc",
        gravity_wave_drag="none", cloud_scheme="none", cld_macmic_num_steps=1,
        **feeds)
    d = ModelDriver(cfg, output_dir=str(tmp))
    d.setup()
    return d


def test_refuses_without_input4mips_sources(tmp_path):
    with pytest.raises(ValueError, match="input4MIPs"):
        _driver(tmp_path, 1)


@_needs_feeds
def test_refuses_empty_ozone_file(tmp_path):
    """No built-in reference ozone in place of the input4MIPs file (codex r1)."""
    with pytest.raises(ValueError, match="ozone_file is empty"):
        _driver(tmp_path, 1, **{**_FEEDS, "ozone_file": ""})


def _run(tmp, n_steps, monkeypatch):
    from legoesm.atmosphere.physics.radiation.ecrad_ifs import lane
    d = _driver(tmp, n_steps, **_FEEDS)
    from legoesm.atmosphere.physics.radiation import integration as rint
    calls, seen, raw = [], [], []
    orig, orig_src = lane.step, rint._ecrad_ifs_sources

    def spy(cfg, sources, **k):       # records each traced variant + inputs
        calls.append(k["full_call"])
        if k["full_call"]:
            jax.debug.callback(
                lambda r, c, h, o: seen.append(tuple(map(np.asarray, (r, c, h, o)))),
                sources.solar_irradiance, sources.gas_mmr["co2"],
                sources.gas_mmr["hcfc22"], sources.o3_mmr)
        return orig(cfg, sources, **k)

    def spy_src(full_call, tsi, eccf, o3, ghg, dtype):   # the forcing it read
        out = orig_src(full_call, tsi, eccf, o3, ghg, dtype)
        jax.debug.callback(
            lambda t, e, r: raw.append((full_call, *map(float, (t, e, r)))),
            tsi, eccf, out.solar_irradiance)
        return out

    monkeypatch.setattr(rint, "_ecrad_ifs_sources", spy_src)
    monkeypatch.setattr(lane, "step", spy)
    T0 = np.asarray(d.state.T.data).copy()
    assert d.run(compiled=True, segment_callback=lambda drv, day, dt: None) == "COMPLETED"
    s = d.state
    assert np.isfinite(np.asarray(s.T.data)).all()
    for name, f in s.tracers.items():
        assert np.isfinite(np.asarray(f.data)).all(), name
    assert np.max(np.abs(np.asarray(s.T.data) - T0)) > 0.0
    ps = getattr(d, "_mpas_phys_state", None)
    assert ps is not None and ps.ecrad_carry is not None
    c = np.asarray(ps.ecrad_carry)
    assert np.isfinite(c).all() and np.max(np.abs(c)) > 0.0
    # exact loader values (M1/M4): the file's TSI x the orbit's distance
    # factor on full AND held calls; the file's 1979 CO2 and HCFC-22 (cfc22)
    from legoesm.atmosphere.physics.radiation.solar import (
        earth_orbit, earth_sun_distance_factor)
    from legoesm.forcing.external import (
        get_ghg_at_time, get_solar_forcing_at_time, ghg_concentrations_to_vmr)
    day0 = float(d.config.start_day)
    tsi = float(get_solar_forcing_at_time(d._solar_config, day0)["tsi"])
    eccf = float(earth_sun_distance_factor(
        d._calendar_for_radiation(day0)[0], earth_orbit()))
    vmr = ghg_concentrations_to_vmr(get_ghg_at_time(d._ghg_config, day0))
    assert {full for full, *_ in raw} == set(calls)
    for _full, t, e, r in raw:
        assert t == tsi and e != 1.0
        # 15 min from 1 Jan drift ~3.5e-7 (near perihelion); NOT valid for
        # a start near quadrature, where 15 min drifts ~6e-6
        assert e == pytest.approx(eccf, rel=1e-6, abs=0.0)
        assert r == pytest.approx(t * e, rel=1e-15)
    assert seen
    for rii0, co2, hcfc22, o3 in seen:
        assert float(co2) == pytest.approx(vmr["co2"] * 44.011 / 28.970, rel=1e-12,
                                           abs=0.0)
        assert float(hcfc22) == pytest.approx(vmr["cfc22"] * 86.469 / 28.970,
                                              rel=1e-12, abs=0.0)
        print(f"ecrad_ifs feeds: RII0={float(rii0):.3f} W/m2 co2_mmr={float(co2):.4e} "
              f"hcfc22_mmr={float(hcfc22):.4e} o3_mmr min/max="
              f"{o3.min():.3e}/{o3.max():.3e}")
        assert 1300.0 < float(rii0) < 1450.0          # TSI x Earth-Sun factor
        assert 4.9e-4 < float(co2) < 5.3e-4           # 1979 ~337 ppm x 44.011/28.97
        assert 1.0e-11 < float(hcfc22) < 1.0e-9       # 1979 HCFC-22 ~45-60 ppt
        assert o3.min() >= 0.0 and 1.0e-7 < o3.max() < 3.0e-5
        assert o3[:, 0].mean() > o3[:, -1].mean()     # top level (index 0) richer
    return c, set(calls)


@_needs_feeds
@pytest.mark.slow
def test_ecrad_lane_runs_full_and_held_steps(tmp_path, monkeypatch):
    c1, v1 = _run(tmp_path / "one", 1, monkeypatch)
    c3, v3 = _run(tmp_path / "three", 3, monkeypatch)
    assert v1 == {True}                  # step 0: the full call only
    assert v3 == {True, False}           # step 1: the RADHEATN-only variant
    # step 2 is a new full call: the carry is refreshed, not held
    assert np.max(np.abs(c3 - c1)) > 0.0
