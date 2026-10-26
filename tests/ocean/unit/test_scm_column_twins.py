"""Direct tests for scripts/validate/ocean_fidelity/run_scm_column_twins.py.

Pure pieces (nearest-wet-column selection + distance guard, IC extraction,
record-index convention, dz derivation, sign flip, CLI round-trip) are tested
with numpy/netCDF4 only; the SCM-construction tests (T0 really has f_c=0 and
u = v = 0 exactly; forcing closures finite over the run window) use a tiny
5-level column with constant vertical mixing on CPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "run_scm_column_twins",
    _ROOT / "scripts" / "validate" / "ocean_fidelity" / "run_scm_column_twins.py",
)
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)


# ---------------------------------------------------------------------------
# point-spec parsing / CLI
# ---------------------------------------------------------------------------


def test_parse_point_spec_roundtrip():
    assert tw.parse_point_spec("barents=73,40") == ("barents", 73.0, 40.0)
    name, lat, lon = tw.parse_point_spec("nh_midlat=35,-40")
    assert (name, lat, lon) == ("nh_midlat", 35.0, -40.0)


@pytest.mark.parametrize(
    "bad", ["nolatlon", "x=1", "x=1,2,3", "=1,2", "x=99,0", "x=-95,10"],
)
def test_parse_point_spec_rejects_malformed(bad):
    with pytest.raises(ValueError):
        tw.parse_point_spec(bad)


def test_cli_defaults_roundtrip():
    p = tw.build_arg_parser()
    args = p.parse_args([])
    assert args.days == 90
    assert args.dt == 3600.0
    assert args.start_month == 1
    # default trimmed to kpp-only after job 8889618 (constant stays
    # available via --vmix kpp,constant)
    assert args.vmix == "kpp"
    assert args.tiers == "T0,T1"
    assert args.sw_mode == "penetrate"
    assert args.convection == "none"
    assert args.max_point_distance_deg == 2.0
    assert tw._resolve_points(args) == tw.DEFAULT_POINTS

    args2 = p.parse_args(
        ["--point", "a=10,20", "--point", "b=-5,-60", "--days", "30",
         "--vmix", "constant", "--tiers", "T0"]
    )
    assert tw._resolve_points(args2) == {"a": (10.0, 20.0), "b": (-5.0, -60.0)}
    assert args2.days == 30


def test_default_points_are_the_four_diagnostics():
    assert tw.DEFAULT_POINTS == {
        "siberian_shelf": (76.0, 125.0),
        "barents": (73.0, 40.0),
        "nh_midlat": (35.0, -40.0),
        "subtrop_pac": (20.0, -150.0),
    }


# ---------------------------------------------------------------------------
# nearest wet column + distance guard
# ---------------------------------------------------------------------------


def _mesh(lat0=70.0, lon0=120.0, ny=5, nx=7, dlat=1.0, dlon=1.0):
    lat1 = lat0 + dlat * np.arange(ny)
    lon1 = lon0 + dlon * np.arange(nx)
    lon2d, lat2d = np.meshgrid(lon1, lat1)
    return lat2d, lon2d


def test_nearest_wet_column_selects_nearest_wet():
    lat2d, lon2d = _mesh()
    wet = np.ones(lat2d.shape, dtype=bool)
    wet[2, 3] = False   # the exactly-requested cell is LAND
    j, i, d = tw.nearest_wet_column(lat2d, lon2d, wet, 72.0, 123.0)
    assert (j, i) != (2, 3)
    assert wet[j, i]
    # a neighbouring cell, within ~1.2 deg
    assert d < 1.2
    assert abs(j - 2) <= 1 and abs(i - 3) <= 1


def test_nearest_wet_column_exact_hit_zero_distance():
    lat2d, lon2d = _mesh()
    wet = np.ones(lat2d.shape, dtype=bool)
    j, i, d = tw.nearest_wet_column(lat2d, lon2d, wet, 72.0, 123.0)
    assert (j, i) == (2, 3)
    # An EXACT hit is not exactly 0: great_circle_deg goes through
    # arccos(clip(cos_ang)), and arccos loses ~sqrt(eps) precision near 1
    # — an exact hit returns O(1e-6) deg, not 0 (job 8889618 measured
    # 8.5e-7).  1e-4 deg ~ 11 m, far below any ORCA grid spacing, so the
    # bound still pins "this really is the requested cell".
    assert d < 1e-4


def test_nearest_wet_column_too_far_raises():
    lat2d, lon2d = _mesh()
    wet = np.zeros(lat2d.shape, dtype=bool)
    wet[0, 0] = True    # only wet cell is ~ (70N, 120E)
    with pytest.raises(tw.PointTooFarError):
        tw.nearest_wet_column(lat2d, lon2d, wet, 74.0, 126.0,
                              max_dist_deg=2.0)
    with pytest.raises(tw.PointTooFarError):
        tw.nearest_wet_column(lat2d, lon2d, np.zeros_like(wet), 70.0, 120.0)


def test_great_circle_handles_lon_wrap():
    # -150E and 210E are the same meridian.  Same arccos-near-1 roundoff
    # bound as the exact-hit test (O(1e-6) deg, platform-dependent).
    d1 = tw.great_circle_deg(20.0, -150.0, 20.0, 210.0)
    assert float(d1) < 1e-4


# ---------------------------------------------------------------------------
# record index (mirror of run_omip_core2._idx_t) + calendar helpers
# ---------------------------------------------------------------------------


def test_record_index_floor_convention_6hourly():
    n = 1460  # 6-hourly NYF
    assert tw.record_index(0.0, n) == 0
    assert tw.record_index(21599.0, n) == 0     # floor, NOT round
    assert tw.record_index(21600.0, n) == 1
    assert tw.record_index(10800.0, n) == 0     # record centre
    # perpetual-year wrap
    assert tw.record_index(tw._YEAR_S, n) == 0
    assert tw.record_index(tw._YEAR_S + 21600.0, n) == 1
    assert tw.record_index(tw._YEAR_S - 1.0, n) == n - 1


def test_record_index_daily():
    assert tw.record_index(86399.0, 365) == 0
    assert tw.record_index(86400.0, 365) == 1
    with pytest.raises(ValueError):
        tw.record_index(0.0, 0)


def test_host_loop_end_of_step_record_convention():
    # The 3-D host loop is 1-based and samples at END-of-step time
    # (_idx_t(step, dt) = floor(step*dt / 6h)); the SCM driver mirrors it by
    # updating fluxes at t_start + dt.  The step integrating [18000, 21600]
    # must therefore use record 1, not record 0 (codex finding #2).
    dt = 3600.0
    t_start = 5 * dt                                    # 18000 s
    assert tw.record_index(t_start, 1460) == 0          # start-of-step: rec 0
    assert tw.record_index(t_start + dt, 1460) == 1     # driver samples rec 1


def test_month_start_seconds_noleap():
    assert tw.month_start_seconds(1) == 0.0
    assert tw.month_start_seconds(2) == 31 * 86400.0
    assert tw.month_start_seconds(3) == 59 * 86400.0
    assert tw.month_start_seconds(12) == 334 * 86400.0
    for bad in (0, 13):
        with pytest.raises(ValueError):
            tw.month_start_seconds(bad)


def test_month_sequence_wraps():
    assert tw.month_sequence(1, 3) == [1, 2, 3]
    assert tw.month_sequence(11, 3) == [11, 12, 1]


def test_atm_to_ocean_stress_sign_flip():
    # air_sea_fluxes: ATMOSPHERIC convention (-rho Cd |U| u, opposes wind).
    # Westerly wind u10>0 -> tau_atm<0 -> on-ocean stress must be POSITIVE
    # (eastward surface acceleration in the SCM's OCEAN_DIRECT convention).
    assert tw.atm_to_ocean_stress(-0.1) == pytest.approx(0.1)
    assert tw.atm_to_ocean_stress(0.25) == pytest.approx(-0.25)
    assert tw.atm_to_ocean_stress(0.0) == 0.0


# ---------------------------------------------------------------------------
# dz derivation
# ---------------------------------------------------------------------------


def test_dz_from_center_depths_monotone_positive():
    zc = np.array([0.5, 1.6, 3.0, 5.0, 8.0])
    dz = tw.dz_from_center_depths(zc)
    assert dz.shape == zc.shape
    assert np.all(dz > 0.0)
    # interfaces nest the centres: cumulative interface depths bracket zc
    zi = np.concatenate([[0.0], np.cumsum(dz)])
    assert np.all(zi[:-1] < zc) and np.all(zc < zi[1:])
    # uniform grid is exact
    zu = np.arange(0.5, 10.0, 1.0)
    assert np.allclose(tw.dz_from_center_depths(zu), 1.0)


def test_dz_from_center_depths_rejects_nonmonotone():
    with pytest.raises(ValueError):
        tw.dz_from_center_depths(np.array([1.0, 0.5, 2.0]))
    with pytest.raises(ValueError):
        tw.dz_from_center_depths(np.array([1.0]))


# ---------------------------------------------------------------------------
# IC extraction from synthetic NEMO-layout netCDF files
# ---------------------------------------------------------------------------

_FILL = -1.0e34


def _write_ic_pair(tmp_path: Path, n_wet_col=3, land_row0=False):
    """Tiny (time=2, z=5, y=4, x=6) IC pair in the NEMO woce layout.

    ``land_row0=True`` lands the ENTIRE j=0 row, so a point requested on
    that row selects a nearest-wet cell one row north (offset in latitude).
    """
    import netCDF4

    nz, ny, nx = 5, 4, 6
    lat2d = 70.0 + np.arange(ny)[:, None] + np.zeros((ny, nx))
    lon2d = 20.0 + np.arange(nx)[None, :] + np.zeros((ny, nx))
    nav_lev = np.array([0.5, 1.6, 3.0, 5.0, 8.0])

    def _make(path, var, units, base):
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("z", nz)
            ds.createDimension("y", ny)
            ds.createDimension("x", nx)
            v = ds.createVariable(var, "f8", ("time", "z", "y", "x"),
                                  fill_value=_FILL)
            v.units = units
            la = ds.createVariable("nav_lat", "f8", ("y", "x"))
            lo = ds.createVariable("nav_lon", "f8", ("y", "x"))
            le = ds.createVariable("nav_lev", "f8", ("z",))
            le.units = "m"
            tv = ds.createVariable("time", "f8", ("time",))
            la[:], lo[:], le[:] = lat2d, lon2d, nav_lev
            tv[:] = [0.0, 31.0]
            data = np.full((2, nz, ny, nx), _FILL)
            for m in range(2):
                for k in range(nz):
                    data[m, k] = base + m - 0.5 * k
            # column (1, 2): wet only down to n_wet_col levels
            data[:, n_wet_col:, 1, 2] = _FILL
            # column (0, 0): land everywhere
            data[:, :, 0, 0] = _FILL
            if land_row0:
                data[:, :, 0, :] = _FILL
            v[:] = data
    tpath = tmp_path / "woce_temp.nc"
    spath = tmp_path / "woce_salt.nc"
    _make(tpath, "contemp", "Celsius degrees", base=10.0)
    _make(spath, "presalt", "g/kg", base=34.0)
    return tpath, spath


def test_ic_extraction_shape_finite_monotone(tmp_path):
    tpath, spath = _write_ic_pair(tmp_path, n_wet_col=3)
    ic = tw.extract_ic_profile(tpath, spath, 0, 71.0, 22.0, max_dist_deg=2.0)
    # requested cell (71N, 22E) == grid (j=1, i=2), truncated at 3 wet levels
    assert (ic["j"], ic["i"]) == (1, 2)
    assert ic["n_wet"] == 3
    assert ic["T"].shape == (3,) and ic["S"].shape == (3,)
    assert np.all(np.isfinite(ic["T"])) and np.all(np.isfinite(ic["S"]))
    assert np.all(np.diff(ic["nav_lev"]) > 0)          # depth-monotone
    assert ic["temp_var"] == "contemp"
    assert ic["salt_var"] == "presalt"
    assert ic["dist_deg"] < 1e-6
    # month index selects the right record (month 1 differs by +1)
    ic2 = tw.extract_ic_profile(tpath, spath, 1, 71.0, 22.0)
    assert np.allclose(ic2["T"], ic["T"] + 1.0)


def test_ic_extraction_full_depth_column(tmp_path):
    tpath, spath = _write_ic_pair(tmp_path)
    ic = tw.extract_ic_profile(tpath, spath, 0, 73.0, 25.0)
    assert ic["n_wet"] == 5
    assert np.allclose(ic["S"], 34.0 - 0.5 * np.arange(5))


def test_ic_extraction_land_point_moves_to_wet_neighbor(tmp_path):
    tpath, spath = _write_ic_pair(tmp_path)
    ic = tw.extract_ic_profile(tpath, spath, 0, 70.0, 20.0)  # (0,0) is land
    assert (ic["j"], ic["i"]) != (0, 0)
    assert ic["dist_deg"] <= 1.5


def test_ic_extraction_too_far_raises(tmp_path):
    tpath, spath = _write_ic_pair(tmp_path)
    with pytest.raises(tw.PointTooFarError):
        tw.extract_ic_profile(tpath, spath, 0, 40.0, -60.0, max_dist_deg=2.0)


# ---------------------------------------------------------------------------
# referee (grid_T) extraction: per-point e3t thickness, data-less trailing
# records, per-variable time-length guards (codex 2026-07 findings #3/#4)
# ---------------------------------------------------------------------------

_FILL_GRIDT = 1.0e20        # NEMO/XIOS _FillValue convention of grid_T files
_GRIDT_DZ = np.array([1.0, 2.0, 4.0, 8.0])       # deptht_bounds thicknesses


def _write_gridt(tmp_path: Path, n_time=3, garbage_last=False, with_e3t=True,
                 so_time_dim=None):
    """Synthetic NEMO grid_T referee (time, z=4, y=3, x=4), CF noleap time.

    Record ``k`` is calendar month ``(k % 12) + 1`` starting January; data
    value is ``base + k + 0.1*kz`` everywhere.  ``e3t`` equals the global
    ``deptht_bounds`` thickness except at the probe point (j=1, i=2), where
    the column is partial-cell (``0.5*dz + 0.001*k``, record-varying so
    averaging is observable).  ``garbage_last=True`` writes the LAST record
    of to/so/e3t as all-fill while its ``time_counter`` entry EXISTS — the
    killed-run signature of the real RUN_REF file (40 time entries, 39
    written payloads).  ``so_time_dim`` puts ``so`` on its own SHORTER time
    dim to exercise the per-variable length guard.
    """
    import netCDF4

    nz, ny, nx = 4, 3, 4
    path = tmp_path / "gridt.nc"
    bounds = np.stack(
        [np.concatenate([[0.0], np.cumsum(_GRIDT_DZ)[:-1]]),
         np.cumsum(_GRIDT_DZ)], axis=1,
    )
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time_counter", n_time)
        ds.createDimension("deptht", nz)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        ds.createDimension("axis_nbounds", 2)
        tc = ds.createVariable("time_counter", "f8", ("time_counter",))
        tc.units = "seconds since 1900-01-01 00:00:00"
        tc.calendar = "noleap"
        starts = np.concatenate(
            [[0.0], np.cumsum(np.asarray(tw._NOLEAP_MONTH_DAYS, float))]
        )
        tc[:] = [
            ((k // 12) * 365.0 + starts[k % 12] + 15.0) * 86400.0
            for k in range(n_time)
        ]
        la = ds.createVariable("nav_lat", "f8", ("y", "x"))
        lo = ds.createVariable("nav_lon", "f8", ("y", "x"))
        la[:] = 70.0 + np.arange(ny)[:, None] + np.zeros((ny, nx))
        lo[:] = 20.0 + np.arange(nx)[None, :] + np.zeros((ny, nx))
        dv = ds.createVariable("deptht", "f8", ("deptht",))
        dv[:] = bounds.mean(axis=1)
        db = ds.createVariable(
            "deptht_bounds", "f8", ("deptht", "axis_nbounds"))
        db[:] = bounds

        def _mk(name, dims, base):
            v = ds.createVariable(name, "f8", dims, fill_value=_FILL_GRIDT)
            v.missing_value = _FILL_GRIDT
            nt_v = len(ds.dimensions[dims[0]])
            data = np.empty((nt_v, nz, ny, nx))
            for k in range(nt_v):
                for kz in range(nz):
                    data[k, kz] = base + k + 0.1 * kz
            if garbage_last:
                data[-1] = _FILL_GRIDT
            v[:] = data

        _mk("to", ("time_counter", "deptht", "y", "x"), 10.0)
        if so_time_dim:
            ds.createDimension(so_time_dim, n_time - 1)
            _mk("so", (so_time_dim, "deptht", "y", "x"), 30.0)
        else:
            _mk("so", ("time_counter", "deptht", "y", "x"), 30.0)
        if with_e3t:
            v = ds.createVariable(
                "e3t", "f8", ("time_counter", "deptht", "y", "x"),
                fill_value=_FILL_GRIDT,
            )
            v.missing_value = _FILL_GRIDT
            e = np.empty((n_time, nz, ny, nx))
            for k in range(n_time):
                e[k] = np.broadcast_to(
                    _GRIDT_DZ[:, None, None], (nz, ny, nx)).copy()
                e[k, :, 1, 2] = 0.5 * _GRIDT_DZ + 0.001 * k   # probe column
            if garbage_last:
                e[-1] = _FILL_GRIDT
            v[:] = e
    return path


def test_referee_uses_per_point_e3t_thickness(tmp_path):
    # finding #3: the probe column is partial-cell — dz must be the
    # per-point e3t on the selected records, NOT the global deptht_bounds.
    path = _write_gridt(tmp_path, n_time=3, with_e3t=True)
    nemo = tw.extract_nemo_reference(path, 1, 2, [1, 2], clim=False)
    assert nemo["dz_source"] == "e3t"
    # clim=False: first occurrence of months 1, 2 = records [0], [1];
    # across-month mean of e3t(1,2) = 0.5*dz + 0.001*(0+1)/2
    assert np.allclose(nemo["dz"], 0.5 * _GRIDT_DZ + 0.001 * 0.5)
    assert not np.allclose(nemo["dz"], _GRIDT_DZ)
    assert nemo["records_used"] == [[0], [1]]
    assert nemo["dropped_records"] == []


def test_referee_falls_back_to_deptht_bounds_without_e3t(tmp_path):
    path = _write_gridt(tmp_path, n_time=3, with_e3t=False)
    nemo = tw.extract_nemo_reference(path, 1, 2, [1], clim=False)
    assert nemo["dz_source"] == "deptht_bounds"
    assert np.allclose(nemo["dz"], _GRIDT_DZ)


def test_referee_clim_e3t_averages_all_month_occurrences(tmp_path):
    path = _write_gridt(tmp_path, n_time=14, with_e3t=True)  # months 1..12,1,2
    nemo = tw.extract_nemo_reference(path, 1, 2, [1], clim=True)
    # month 1 occurs at records 0 and 12 -> e3t mean over {0, 12}
    assert nemo["records_used"] == [[0, 12]]
    assert np.allclose(nemo["dz"], 0.5 * _GRIDT_DZ + 0.001 * 6.0)


def test_referee_excludes_dataless_trailing_record(tmp_path):
    # finding #4 (real-file mechanism): the time axis has one MORE entry
    # than the written to/so/e3t payloads; --nemo-clim must not average the
    # all-fill record into the referee.
    path = _write_gridt(tmp_path, n_time=14, garbage_last=True)
    nemo = tw.extract_nemo_reference(path, 1, 2, [2], clim=True)
    # month 2 candidates {1, 13}; record 13 is data-less -> only record 1
    assert nemo["records_used"] == [[1]]
    assert nemo["dropped_records"] == [13]
    assert np.all(np.isfinite(nemo["T"]))
    assert np.allclose(nemo["T"][0], 10.0 + 1 + 0.1 * np.arange(4))
    assert np.allclose(nemo["S"][0], 30.0 + 1 + 0.1 * np.arange(4))
    assert np.all(np.isfinite(nemo["dz"]))        # e3t skipped rec 13 too


def test_referee_month_only_on_dataless_record_raises(tmp_path):
    # months are [1, 2, 3] and record 2 (month 3) is data-less: selecting
    # month 3 must raise, never return a fill/NaN referee.
    path = _write_gridt(tmp_path, n_time=3, garbage_last=True)
    with pytest.raises(ValueError, match="data-less"):
        tw.extract_nemo_reference(path, 1, 2, [3], clim=False)


def test_referee_so_time_length_mismatch_raises(tmp_path):
    # per-variable time-length guard: so on a different/shorter time dim
    # than to must be refused, not silently mispaired.
    path = _write_gridt(tmp_path, n_time=3, so_time_dim="time_counter2")
    with pytest.raises(ValueError, match="time length"):
        tw.extract_nemo_reference(path, 1, 2, [1], clim=False)


# ---------------------------------------------------------------------------
# physics-config dispatch hardening
# ---------------------------------------------------------------------------


def test_build_physics_config_unknown_vmix_raises():
    with pytest.raises(ValueError):
        tw.build_physics_config("nope", "top", "none")
    with pytest.raises(ValueError):
        tw.build_physics_config("kpp", "top", "mystery_convection")


def test_core2pointfluxes_rejects_bad_sw_mode():
    class _F:  # minimal forcing stub (time axis only)
        time_s = np.array([0.0, 21600.0])

    with pytest.raises(ValueError):
        tw.Core2PointFluxes(_F(), 35.0, -40.0, sw_mode="sideways")


def test_core2pointfluxes_rejects_nonuniform_records():
    class _F:
        time_s = np.array([0.0, 21600.0, 90000.0])

    with pytest.raises(ValueError):
        tw.Core2PointFluxes(_F(), 35.0, -40.0)


# ---------------------------------------------------------------------------
# tiny-column SCM runs (CPU; 5 levels, constant mixing)
# ---------------------------------------------------------------------------


def _tiny_run_inputs():
    from legoesm.ocean.forcing.jra55_do import synthetic_ocean_forcing

    forcing = synthetic_ocean_forcing(0, n_time=12, nlon=36, nlat=18)
    dz = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    T = np.array([15.0, 14.0, 12.0, 9.0, 5.0])
    S = np.array([35.0, 35.1, 35.2, 35.3, 35.4])
    return forcing, dz, T, S


def test_t0_has_no_coriolis_no_wind_and_uv_stay_exactly_zero():
    forcing, dz, T, S = _tiny_run_inputs()
    scm, fluxes = tw.build_scm(
        T_profile=T, S_profile=S, dz=dz, lat_deg=35.0, lon_deg=-40.0,
        dt=600.0, tier="T0", vmix="constant", sw_mode="top",
        convection="none", forcing=forcing,
    )
    # construction-level: T0 really is dynamics-free
    assert scm.forcing.f_c == 0.0
    assert scm.forcing.tau_x is None and scm.forcing.tau_y is None
    assert scm.forcing.q_net is not None
    T_ic_surface = float(np.asarray(scm.state.T.data)[0, 0, 0, 0])
    for _ in range(4):
        fluxes.update(scm.state, scm.t_seconds + scm.dt)
        scm.step()
    assert np.all(np.asarray(scm.state.u.data) == 0.0)
    assert np.all(np.asarray(scm.state.v.data) == 0.0)
    # buoyancy physics really ACTED (not a silent no-op): the bulk q_net is
    # nonzero and the surface temperature moved away from the IC.
    assert fluxes._cache["q_net_total"] != 0.0
    T_end_surface = float(np.asarray(scm.state.T.data)[0, 0, 0, 0])
    assert abs(T_end_surface - T_ic_surface) > 1.0e-9
    assert np.all(np.isfinite(np.asarray(scm.state.T.data)))


def test_t1_has_coriolis_and_wind_produces_motion():
    from legoesm import constants

    forcing, dz, T, S = _tiny_run_inputs()
    scm, fluxes = tw.build_scm(
        T_profile=T, S_profile=S, dz=dz, lat_deg=35.0, lon_deg=-40.0,
        dt=600.0, tier="T1", vmix="constant", sw_mode="top",
        convection="none", forcing=forcing,
    )
    f_expected = 2.0 * constants.Omega * np.sin(np.deg2rad(35.0))
    assert scm.forcing.f_c == pytest.approx(f_expected, rel=1e-12)
    assert callable(scm.forcing.tau_x) and callable(scm.forcing.tau_y)
    for _ in range(4):
        fluxes.update(scm.state, scm.t_seconds + scm.dt)
        scm.step()
    # DIRECTION lock (not just |u|>0, which a flipped sign would also pass):
    # synthetic midlat WESTERLIES (u10 = 10 sin(2 lat) > 0 at 35N) must give
    # an EASTWARD on-ocean stress after the atmospheric->OCEAN_DIRECT flip,
    # and hence an eastward surface current (4 steps of CN rotation turn it
    # by only ~11 deg, far from reversing the sign).
    assert fluxes._cache["tau_x"] > 0.0
    u_surface = float(np.asarray(scm.state.u.data)[0, 0, 0, 0])
    assert u_surface > 0.0
    assert np.all(np.isfinite(np.asarray(scm.state.u.data)))


def test_forcing_closures_finite_over_run_window():
    forcing, dz, T, S = _tiny_run_inputs()
    scm, fluxes = tw.build_scm(
        T_profile=T, S_profile=S, dz=dz, lat_deg=35.0, lon_deg=-40.0,
        dt=600.0, tier="T1", vmix="constant", sw_mode="penetrate",
        convection="none", forcing=forcing,
    )
    year_s = tw._YEAR_S
    for t in (0.0, 0.25 * year_s, 0.5 * year_s, 0.99 * year_s):
        fluxes.update(scm.state, t)
        f = scm.forcing
        vals = [f.tau_x(t), f.tau_y(t), f.q_net(t), f.e_minus_p(t),
                f.sw_down(t)]
        assert all(np.isfinite(v) for v in vals), vals
        # penetrate mode mirrors the 3-D core's two-band split: the closure
        # pair must close the surface heat budget EXACTLY —
        # top(q_net - 0.94 sw) + penetrating(0.94 sw) == q_net_total.
        assert f.sw_down(t) >= 0.0
        assert f.sw_down(t) == pytest.approx(
            tw._SW_PENETRATION_FRACTION * fluxes._cache["sw_net"]
        )
        assert f.q_net(t) + f.sw_down(t) == pytest.approx(
            fluxes._cache["q_net_total"]
        )
        assert fluxes.last_idx == tw.record_index(t, fluxes.n_rec)


def test_build_scm_rejects_unknown_tier():
    forcing, dz, T, S = _tiny_run_inputs()
    with pytest.raises(ValueError):
        tw.build_scm(
            T_profile=T, S_profile=S, dz=dz, lat_deg=0.0, lon_deg=0.0,
            dt=600.0, tier="T9", vmix="constant", sw_mode="top",
            convection="none", forcing=forcing,
        )


def test_run_point_tier_daily_history_and_meta_shapes():
    forcing, dz, T, S = _tiny_run_inputs()
    # run_point_tier reads the SELECTED cell coords from ic (finding #1)
    ic = {"T": T, "S": S, "cell_lat": 20.0, "cell_lon": -150.0}
    r = tw.run_point_tier(
        ic=ic, dz=dz, forcing=forcing,
        tier="T0", vmix="constant", sw_mode="top", convection="none",
        days=1, dt=7200.0, t0_seconds=0.0,
    )
    assert r["time_days"].shape == (1,)
    assert r["T"].shape == (1, 5) and r["S"].shape == (1, 5)
    assert r["u"].shape == (1, 5) and r["K_v"].shape[0] == 1
    assert r["z_center"].shape == (5,)
    assert np.all(np.diff(r["z_center"]) > 0)   # positive-down monotone
    assert np.all(r["u"] == 0.0) and np.all(r["v"] == 0.0)
    assert np.all(np.isfinite(r["T"]))
    assert r["cell_lat"] == 20.0 and r["cell_lon"] == -150.0


def test_run_point_tier_forcing_and_coriolis_at_selected_cell(
    tmp_path, monkeypatch,
):
    # codex 2026-07 finding #1: a requested point whose nearest WET cell is
    # one row north (synthetic land row) must sample the CORE-II forcing AND
    # compute f_c at the SELECTED cell coords (71N), never at the requested
    # point (70N) — the IC/referee column lives at the cell.
    from legoesm import constants

    tpath, spath = _write_ic_pair(tmp_path, land_row0=True)
    ic = tw.extract_ic_profile(tpath, spath, 0, 70.0, 22.0, max_dist_deg=2.0)
    assert ic["j"] == 1                       # moved off the land row
    assert ic["cell_lat"] == pytest.approx(71.0)
    assert ic["cell_lon"] == pytest.approx(22.0)
    assert ic["cell_lat"] != 70.0             # really offset from the request

    forcing, _, _, _ = _tiny_run_inputs()
    seen: list[tuple[float, float]] = []
    real_extract = tw.extract_point_forcing_series

    def _recorder(forcing_, lat_deg, lon_deg):
        seen.append((float(lat_deg), float(lon_deg)))
        return real_extract(forcing_, lat_deg, lon_deg)

    monkeypatch.setattr(tw, "extract_point_forcing_series", _recorder)
    dz = tw.dz_from_center_depths(ic["nav_lev"])[: ic["n_wet"]]
    r = tw.run_point_tier(
        ic=ic, dz=dz, forcing=forcing, tier="T1", vmix="constant",
        sw_mode="top", convection="none", days=1, dt=7200.0, t0_seconds=0.0,
    )
    # forcing series extracted at the CELL, not the requested point
    assert len(seen) == 1
    assert seen[0][0] == pytest.approx(71.0)
    assert seen[0][1] == pytest.approx(22.0)
    # Coriolis from the CELL latitude
    f_cell = 2.0 * constants.Omega * np.sin(np.deg2rad(71.0))
    f_req = 2.0 * constants.Omega * np.sin(np.deg2rad(70.0))
    assert r["f_c"] == pytest.approx(f_cell, rel=1e-12)
    assert abs(r["f_c"] - f_req) > 1e-7
    assert r["cell_lat"] == pytest.approx(71.0)


# ---------------------------------------------------------------------------
# jitted fast path (job 8889618 timeout fix): series extraction + parity
# ---------------------------------------------------------------------------


def test_extract_point_forcing_series_channels_and_values():
    forcing, _, _, _ = _tiny_run_inputs()
    series = tw.extract_point_forcing_series(forcing, 35.0, -40.0)
    assert set(series) == set(tw._FORCING_CHANNELS)
    n_rec = int(np.asarray(forcing.time_s).size)
    for name, arr in series.items():
        assert arr.shape == (n_rec, 1, 1), name
        assert np.all(np.isfinite(arr)), name
    # The column matches the SHARED 3-D bilinear convention (a90bd1935).
    from legoesm.ocean.coupler.omip2_applicator import _bilinear_point_maps

    i4, j4, w4 = _bilinear_point_maps(
        np.asarray(forcing.lat), np.asarray(forcing.lon),
        np.asarray([35.0]), np.asarray([-40.0]),
    )
    expected_u10 = (
        np.asarray(forcing.u10)[:, i4[:, 0], j4[:, 0]] * w4[:, 0]
    ).sum(axis=1)
    assert np.allclose(
        series["u10"][:, 0, 0],
        expected_u10,
    )
    # Optional NEMO-parity channels: fallbacks identical to the host
    # producers' internal defaults (snow=0, slp=standard atmosphere).
    from legoesm import constants

    if getattr(forcing, "snow", None) is None:
        assert np.all(series["snow"] == 0.0)
    if getattr(forcing, "slp", None) is None:
        assert np.allclose(series["slp"], float(constants.p_atm_std))


def test_build_jitted_step_rejects_bad_tier_sw_mode_and_channels():
    forcing, dz, T, S = _tiny_run_inputs()
    scm, _ = tw.build_scm(
        T_profile=T, S_profile=S, dz=dz, lat_deg=35.0, lon_deg=-40.0,
        dt=600.0, tier="T1", vmix="constant", sw_mode="top",
        convection="none", forcing=forcing,
    )
    series = tw.extract_point_forcing_series(forcing, 35.0, -40.0)
    with pytest.raises(ValueError):
        tw.build_jitted_step(scm, series, tier="T9", sw_mode="top")
    with pytest.raises(ValueError):
        tw.build_jitted_step(scm, series, tier="T1", sw_mode="sideways")
    with pytest.raises(ValueError):
        tw.build_jitted_step(
            scm, {"u10": series["u10"]}, tier="T1", sw_mode="top",
        )


def _run_closure_reference(tier, sw_mode, nsteps, dt):
    """Reference trajectory: Core2PointFluxes closures + eager scm.step()."""
    forcing, dz, T, S = _tiny_run_inputs()
    scm, fluxes = tw.build_scm(
        T_profile=T, S_profile=S, dz=dz, lat_deg=35.0, lon_deg=-40.0,
        dt=dt, tier=tier, vmix="constant", sw_mode=sw_mode,
        convection="none", forcing=forcing,
    )
    for _ in range(nsteps):
        fluxes.update(scm.state, scm.t_seconds + dt)
        scm.step()
    return scm.state


def _run_jitted(tier, sw_mode, nsteps, dt):
    """Same trajectory through the ONE-jit hot path."""
    import jax.numpy as jnp

    forcing, dz, T, S = _tiny_run_inputs()
    scm, fluxes = tw.build_scm(
        T_profile=T, S_profile=S, dz=dz, lat_deg=35.0, lon_deg=-40.0,
        dt=dt, tier=tier, vmix="constant", sw_mode=sw_mode,
        convection="none", forcing=forcing,
    )
    series = tw.extract_point_forcing_series(forcing, 35.0, -40.0)
    step_fn = tw.build_jitted_step(scm, series, tier=tier, sw_mode=sw_mode)
    state = scm.state
    for k in range(nsteps):
        idx = tw.record_index((k + 1) * dt, fluxes.n_rec)
        state, _k_v = step_fn(state, jnp.asarray(idx, dtype=jnp.int32))
    return state


@pytest.mark.parametrize(
    "tier,sw_mode", [("T1", "penetrate"), ("T0", "top")],
)
def test_jitted_step_matches_closure_reference(tier, sw_mode):
    # The production hot path (jit built ONCE, record index traced) must
    # reproduce the validated closure reference step-for-step: same NCAR
    # bulk fluxes at the same END-of-step record, same SW split, same
    # atm->OCEAN_DIRECT sign flip, same 3-D S_ref virtual-salt closure
    # (jit: shared virtual_salt_flux_from_net leaf; reference: the
    # pre-compensated e_minus_p channel through prescribed's local-S
    # formula), same implicit solve + CN rotation.  A record off-by-one or
    # a sign flip moves T by O(1e-3) degC over 3 steps — far above tol.
    import jax

    nsteps, dt = 3, 600.0
    ref = _run_closure_reference(tier, sw_mode, nsteps, dt)
    got = _run_jitted(tier, sw_mode, nsteps, dt)
    # x64: both paths call the same jnp formulas on the same f64 inputs
    # (only jit fusion/FMA reassociation differs).  x32: the closure path
    # rounds host f64 floats into f32 arrays while the jit path computes
    # in f32 throughout, so allow a looser (still sign/record-detecting)
    # bound.
    tol = 1e-9 if jax.config.read("jax_enable_x64") else 1e-4
    for fname in ("T", "S", "u", "v"):
        a = np.asarray(getattr(ref, fname).data, dtype=np.float64)
        b = np.asarray(getattr(got, fname).data, dtype=np.float64)
        assert np.allclose(a, b, rtol=0.0, atol=tol), (
            fname, float(np.abs(a - b).max()),
        )
    if tier == "T0":
        # dynamics-free tier: exactly zero momentum on BOTH paths
        assert np.all(np.asarray(got.u.data) == 0.0)
        assert np.all(np.asarray(got.v.data) == 0.0)
    else:
        # physics really acted — the parity above is not trivially-zero
        assert np.any(np.asarray(got.u.data) != 0.0)
        assert np.any(
            np.asarray(got.T.data) != np.asarray(_tiny_run_inputs()[2]).reshape(1, 1, 1, -1)
        )


def test_jitted_step_salt_is_3d_virtual_salt_closure():
    # codex 2026-07 finding #2: salt must follow the 3-D OMIP closure
    # dS/dt = -S_ref*F_fw/(rho_0*dz_0) (freshwater.virtual_salt_flux with
    # the config-default S_ref), NOT prescribed's local-S form
    # S_top*(E-P)/dz_0.  With a uniform S=20 column (far from S_ref=35) the
    # two differ by a factor ~S_ref*rho_w/(rho_0*S_top) ~ 1.7.  The lock is
    # the column-integrated salt budget: vertical diffusion redistributes
    # within the column, so sum(dS*dz) ~ dt * S_ref * (E-P) * rho_w / rho_0
    # (T0 => no other salt source; constant mixing ignores the freshwater
    # buoyancy struct).
    import jax
    import jax.numpy as jnp
    from legoesm import constants

    forcing, dz, T, _ = _tiny_run_inputs()
    S20 = np.full(5, 20.0)
    dt = 600.0
    scm, fluxes = tw.build_scm(
        T_profile=T, S_profile=S20, dz=dz, lat_deg=35.0, lon_deg=-40.0,
        dt=dt, tier="T0", vmix="constant", sw_mode="top",
        convection="none", forcing=forcing,
    )
    series = tw.extract_point_forcing_series(forcing, 35.0, -40.0)
    step_fn = tw.build_jitted_step(scm, series, tier="T0", sw_mode="top")
    state0 = scm.state
    idx = tw.record_index(dt, fluxes.n_rec)   # END-of-step record
    state1, _kv = step_fn(state0, jnp.asarray(idx, dtype=jnp.int32))

    # raw E-P from the SAME producers/record (validated reference cache);
    # the closure evaluates at the identical state + end-of-step time.
    fluxes.update(state0, dt)
    e_minus_p = fluxes._cache["e_minus_p"]
    assert e_minus_p != 0.0                    # non-vacuous
    S_ref, rho_0 = tw._virtual_salt_ref_constants()
    assert S_ref == pytest.approx(35.0)
    assert rho_0 == pytest.approx(constants.rho_ocean)

    dS = (
        np.asarray(state1.S.data, dtype=np.float64)
        - np.asarray(state0.S.data, dtype=np.float64)
    )[0, 0, 0]
    dS_col_got = float(np.sum(dS * dz))
    expected_3d = (
        dt * S_ref * e_minus_p * float(constants.rho_water) / rho_0
    )
    local_s_form = dt * 20.0 * e_minus_p       # the OLD (wrong-parity) form
    # loose enough for any small non-conservation of the implicit solve,
    # far tighter than the 43% local-S discrepancy
    tol = 5e-3 if jax.config.read("jax_enable_x64") else 2e-2
    assert dS_col_got == pytest.approx(expected_3d, rel=tol)
    assert dS_col_got != pytest.approx(local_s_form, rel=0.2)
