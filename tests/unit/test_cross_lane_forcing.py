"""One external-forcing set, every lane (gridaudit 2026-10-09).

Each lane is set up for real (tiny grid, rrtmg) with the forcing readers
monkeypatched to a synthetic latitude- and day-dependent set, then stopped at
its first hand-off to physics: the per-step ``forcing`` dict given to
``model.step`` (MPAS; the fv3_duo column lane runs this same ``_run_column``),
the spectral ``forcing_data`` dict, or the ``pack_forcing`` kwargs that become
the compiled cube / lat-lon SegmentForcing.  Every lane must hand its
radiation the same TSI, spectrum, ozone, aerosol column, CCN AOD, volcanic LW
aerosol, GHG set (halogens included) and insolation calendar; the radiation
config each lane builds is spied for the constant-solar TSI and the CMIP
experiment CO2.  Matrix: fv3_duo_gaps/fu/gridaudit_matrix.md.
"""
from __future__ import annotations

import math
import re
import shlex
import time
from pathlib import Path

import numpy as np
import pytest

import legoesm.forcing.external as ext
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver
from legoesm.forcing.time_utils import day_to_calendar

REPO = Path(__file__).resolve().parents[2]
NGPT = 112
SPECTRUM = np.linspace(1.0, 2.0, NGPT) / np.linspace(1.0, 2.0, NGPT).sum()
GHG = {"co2_ppmv": 400.0, "ch4_ppbv": 1800.0, "n2o_ppbv": 320.0,
       "cfc11_pptv": 250.0, "cfc12_pptv": 530.0, "cfc22_pptv": 100.0,
       "ccl4_pptv": 90.0, "cf4_pptv": 75.0}
GHG_VMR = {"co2": 400e-6, "ch4": 1800e-9, "n2o": 320e-9, "cfc11": 250e-12,
           "cfc12": 530e-12, "cfc22": 100e-12, "ccl4": 90e-12, "cf4": 75e-12}

LANES = {
    "mpas": (dict(grid_type="mpas", resolution=3, nlev=8, vertical_coord="hybrid"),
             dict(discretization="mpas", dt=300.0)),
    "spectral": (dict(grid_type="gaussian", resolution=8, nlev=8),
                 dict(model_type="hydrostatic", discretization="spectral", dt=600.0)),
    "cube": (dict(grid_type="cubed_sphere", resolution=4, nlev=5), dict(dt=600.0)),
    "latlon": (dict(grid_type="latlon", resolution=8, nlev=5),
               dict(model_type="hydrostatic", discretization="latlon_cgrid", dt=300.0)),
}


def _o3(lat, day):
    return 1e-6 * (2.0 + np.cos(lat)) + 1e-9 * day


def _aod(lat, day, ccn):
    # CCN: fine-mode file (path "ccn.nc"), visible band.
    return (0.05 if ccn else 0.1) * (1.0 + 0.5 * np.sin(lat)) + 1e-4 * day + (0.01 if ccn else 0.0)


class _Stop(Exception):
    pass


def _patch_readers(monkeypatch):
    monkeypatch.setattr(ext, "get_solar_forcing_at_time", lambda c, day: {
        "tsi": 1360.5 + 1e-3 * float(day), "solar_fraction_by_gpt": SPECTRUM})
    monkeypatch.setattr(ext, "get_ozone_at_time", lambda c, day, lat_grid=None, p_grid=None: (
        _o3(np.asarray(lat_grid), day)[:, None] * np.ones(np.shape(p_grid))))
    monkeypatch.setattr(ext, "get_aerosol_at_time", lambda c, day, lat_grid=None, band="gray": (
        _aod(np.asarray(lat_grid), day, "ccn" in (c.path or ""))))
    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_at_time", lambda c, day, lat_grid=None: None)
    monkeypatch.setattr(ext, "get_aerosol_lw_at_time", lambda c, day, lat_grid=None: (
        np.full((np.size(lat_grid), 2), 0.01), np.array([2000.0, 3000.0, 4000.0])))
    monkeypatch.setattr(ext, "get_ghg_at_time", lambda c, day: dict(GHG))


def _spy_lane(monkeypatch, d, lane):
    """Run the lane up to its first physics hand-off; return what it handed
    over plus the radiation config it built."""
    import legoesm.atmosphere.physics.combined as combined
    import legoesm.driver.compiled_segments as cs
    rec = {}
    # what run() sets before dispatching to a lane
    d._segment_callback = d._checkpoint_callback = None
    d._run_wallclock_start = time.time()
    if lane in ("mpas", "spectral"):
        orig = combined.make_physics

        def _mk(cfg, *a, **k):
            rec["rad"] = cfg.radiation
            return orig(cfg, *a, **k)
        monkeypatch.setattr(combined, "make_physics", _mk)

        def _step(self, *a, **k):
            rec["forcing"] = dict(k.get("forcing") or k.get("forcing_data"))
            raise _Stop
        monkeypatch.setattr(type(d.model), "step", _step)
        with pytest.raises(_Stop):
            (d._run_column if lane == "mpas" else d._run_spectral)(0, None)
        f, rad = rec["forcing"], rec["rad"]
        s0 = rad.rrtmgp.S_0 if rad.scheme == "rrtmgp" else rad.gray.S_0
        return dict(
            tsi=float(f.get("tsi", s0)), spectrum=f.get("solar_spectral_fraction"),
            o3=f["o3_vmr"], aer=f["aerosol_od"], ccn=f.get("aerosol_ccn_aod"),
            lw=f.get("aerosol_lw_od"), ghg=f.get("ghg_vmr"),
            co2_cfg=rad.rrtmgp.co2_ppmv, doy=float(f["day_of_year"]))
    orig_build = cs.build_segment_fn

    def _build(*a, **k):
        rec["start_day"] = k["start_day"]
        return orig_build(*a, **k)

    def _pack(**k):
        rec["pack"] = k
        raise _Stop
    monkeypatch.setattr(cs, "build_segment_fn", _build)
    monkeypatch.setattr(cs, "pack_forcing", _pack)
    with pytest.raises(_Stop):
        d._run_compiled(0, None)
    k = rec["pack"]
    w = np.asarray(k["solar_weights"])
    return dict(
        tsi=float(k["s_0"]), spectrum=w if w.size else None,
        o3=k["o3_vmr"], aer=k["aerosol_od"], ccn=k["aerosol_ccn_aod"],
        lw=k["aerosol_lw_od"], ghg=k["ghg_vmr"], co2_cfg=rec.get("co2_cfg"),
        # the compiled scan's per-step solar calendar (first step end)
        doy=float(day_to_calendar(rec["start_day"] + d.config.dycore.dt / 86400.0)[0]))


def _build(tmp_path, lane, **over):
    g, dy = LANES[lane]
    cfg = ExperimentConfig(
        grid=GridConfig(**g), dycore=DycoreConfig(**dy),
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=0),
        days=1.0, dataset="analytical", radiation="rrtmg", convection="none",
        turbulence="none", microphysics="none", precision="fp64",
        distributed=False, **over)
    d = ModelDriver(cfg, output_dir=str(tmp_path / lane))
    d.setup()
    return d


def _lat_col(d, ncol):
    lat = np.asarray(d._grid_lat)
    if lat.size != ncol:  # spectral: 1-D Gaussian latitudes
        lat = np.broadcast_to(lat[:, None], (lat.size, ncol // lat.size))
    return lat.reshape(-1)


@pytest.mark.parametrize("lane", sorted(LANES))
def test_every_lane_hands_radiation_the_same_external_forcing(monkeypatch, tmp_path, lane):
    _patch_readers(monkeypatch)
    d = _build(tmp_path, lane, ozone_forcing="external", ozone_file="o3.nc",
               solar_source="spectral_file", solar_file="sol.nc",
               ghg_forcing="external", ghg_file="ghg.nc",
               aerosol_forcing="external", aerosol_file="aer.nc",
               aerosol_ccn_file="ccn.nc", volcanic_aerosol_file="volc.nc",
               volcanic_aerosol_lw=True, insolation_start_doy=91.0)
    r = _spy_lane(monkeypatch, d, lane)
    day = 0.0  # every lane's first forcing sample is the start day
    o3, aer = np.asarray(r["o3"]), np.asarray(r["aer"])
    lat = _lat_col(d, o3.shape[0])
    assert r["tsi"] == pytest.approx(1360.5 + 1e-3 * day, rel=1e-14)
    np.testing.assert_allclose(r["spectrum"], SPECTRUM, rtol=1e-14)
    np.testing.assert_allclose(o3, np.broadcast_to(_o3(lat, day)[:, None], o3.shape), rtol=1e-12)
    np.testing.assert_allclose(aer.sum(-1), _aod(lat, day, False), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(r["ccn"]).reshape(-1), _aod(lat, day, True), rtol=1e-12)
    lw = np.asarray(r["lw"])
    assert lw.shape == o3.shape and lw.sum() > 0.0
    assert {k: float(v) for k, v in r["ghg"].items()} == pytest.approx(GHG_VMR, rel=1e-14)
    # insolation_start_doy=91 shifts the radiation calendar on every lane
    assert math.floor(r["doy"]) == math.floor(d._calendar_for_radiation(0.0)[0]) == 91


@pytest.mark.parametrize("lane", sorted(LANES))
def test_constant_sun_and_experiment_co2_reach_every_lane(monkeypatch, tmp_path, lane):
    """--solar-s0 and a fixed CMIP experiment's CO2 (abrupt-4xCO2: 1137.2
    ppmv) must reach every lane's radiation.  Before the fix MPAS/spectral
    ran the radiation-module default S_0, and the compiled pipeline (built
    before the experiment override) ran the default 415 ppmv."""
    import legoesm.driver.physics_pipeline as pp
    seen = {}
    orig = pp._RADIATION_BUILDERS["rrtmg"]

    def _builder(config):
        seen["co2"] = config.co2_ppmv
        return orig(config)
    monkeypatch.setitem(pp._RADIATION_BUILDERS, "rrtmg", _builder)
    d = _build(tmp_path, lane, S_0=1350.0, experiment="abrupt-4xCO2")
    r = _spy_lane(monkeypatch, d, lane)
    assert r["tsi"] == pytest.approx(1350.0, rel=1e-14)
    co2 = r["co2_cfg"] if lane in ("mpas", "spectral") else seen["co2"]
    assert co2 == pytest.approx(4.0 * 284.3, rel=1e-12)


@pytest.mark.parametrize("lane", sorted(LANES) + ["spectral-dry"])
def test_every_lane_samples_the_same_forcing_days(monkeypatch, tmp_path, lane):
    """Two-day run, solar file on gray radiation: every lane samples the
    daily forcing at the day it is IN (0 then 1).  The compiled lanes used to
    sample the segment-END day (0 then 2: one day ahead of MPAS)."""
    days = []
    monkeypatch.setattr(ext, "get_solar_forcing_at_time", lambda c, day: (
        days.append(float(day)) or {"tsi": 1361.0, "solar_fraction_by_gpt": None}))
    g, dy = LANES[lane.split("-")[0]]
    cfg = ExperimentConfig(
        grid=GridConfig(**g), dycore=DycoreConfig(**dy),
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=0),
        days=2.0, dataset="analytical", radiation="gray", convection="none",
        # louis: spectral full-physics (forcing dict); spectral-dry: legacy gray
        turbulence="louis" if lane == "spectral" else "none",
        precision="fp64", distributed=False,
        solar_source="file", solar_file="sol.nc")
    d = ModelDriver(cfg, output_dir=str(tmp_path / lane))
    d.setup()
    days.clear()  # setup's start-day sample
    assert d.run() == "COMPLETED"
    assert sorted(set(days)) == [0.0, 1.0], days


def _deck(name):
    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    return read_yaml_with_includes(str(REPO / "config" / "amip" / name))


_SWITCHES = ("ozone_forcing", "solar_source", "solar_tsi_var", "solar_spectral_var",
             "solar_spectral_band_order", "ghg_forcing", "aerosol_forcing",
             "orbital_insolation", "diurnal_cycle")
_DECKS = ("amip_production.yaml", "amip_production_fv3duo_c24.yaml",
          "amip_sundqvist_l36.yaml", "amip_sundqvist_latlon24.yaml",
          "amip_production_l36.yaml")


def _shell_array(text, name):
    """Tokens of every ``name=(...)`` / ``name+=(...)`` block (comments cut)."""
    toks = []
    for body in re.findall(rf"^\s*{name}\+?=\(\s*\n(.*?)^\s*\)", text, re.S | re.M):
        for line in body.splitlines():
            toks += shlex.split(line, comments=True)
    for body in re.findall(rf"{name}\+?=\(([^\n]*?)\)\s*$", text, re.M):  # one-liners
        toks += shlex.split(body, comments=True)
    return toks


def test_production_decks_and_coupled_launcher_select_the_same_forcing():
    want = {k: _deck(_DECKS[0]).get(k) for k in _SWITCHES}
    for name in _DECKS[1:]:
        assert {k: _deck(name).get(k) for k in _SWITCHES} == want, name
    from scripts.run.run_coupled import build_parser
    sh = (REPO / "config" / "amip" / "amip_production.ginsburg.sh").read_text()
    coupled = _shell_array(sh, "CMIP6_FORCING_FLAGS")
    args = build_parser().parse_args(coupled)
    for k in _SWITCHES:
        assert getattr(args, k) == want[k], k
    # same files on both sides: each file flag names the same shell variable
    amip = _shell_array(sh, "AMIP_PATH_FLAGS")
    for flag in ("--ozone-file", "--solar-file", "--ghg-file", "--aerosol-file",
                 "--aerosol-ccn-file", "--volcanic-aerosol-file"):
        assert coupled[coupled.index(flag) + 1] == amip[amip.index(flag) + 1], flag
    assert ("--volcanic-aerosol-lw" in coupled) == ("--volcanic-aerosol-lw" in amip)
