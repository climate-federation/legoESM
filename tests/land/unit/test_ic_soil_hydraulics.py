"""Soil-hydraulics stamp on land restarts, and the water-conserving IC conversion.

A land state carries matric potential; potential and water content are tied by
the hydraulics.  A state evolved on Clapp-Hornberger soil and read on van
Genuchten soil pairs each potential with the wrong water content, and the
Richards step turns the mismatch into lost water (the regridded LMIP state's
deep root zone dried from 0.29 to 0.15 in one day).
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.restart import (
    HYDRAULICS_SOURCE_CLM_MAP, HYDRAULICS_SOURCE_SURFDATA_COSBY,
    conform_soil_water, convert_ic_soil_water, load_land_restart,
    save_land_restart, soil_hydraulics_column_signature, soil_hydraulics_stamp,
    soil_hydraulics_stamps_match)
from legoesm.land.richards import RichardsConfig, psi_dry_floor, solve_richards
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig, psi_from_theta, theta_from_psi)
from legoesm.land.state import MultiLayerLandState

_NCOL, _NLAY = 6, 10
_GRID = make_soil_grid(SoilGridConfig(n_layers=_NLAY, total_depth=3.0,
                                      growth_factor=2.0))
_DZ = np.asarray(_GRID.dz, dtype=np.float64)


def _vg(dtype=jnp.float64):
    col = lambda v: jnp.full((_NCOL, 1), v, dtype=dtype)
    return SoilHydraulicsConfig(theta_r=col(0.078), theta_sat=col(0.43),
                                alpha_vg=col(3.6), n_vg=col(1.56),
                                K_sat=col(2.9e-6))


def _ch():
    return SoilHydraulicsConfig(retention_curve="clapp_hornberger",
                                theta_sat=0.45, psi_sat=-0.2, b_ch=5.0,
                                K_sat=1.0e-5, theta_r=0.0)


def _state(theta, psi):
    z = jnp.zeros(_NCOL)
    return MultiLayerLandState(
        T_soil=jnp.full((_NCOL, _NLAY), 280.0), psi_soil=jnp.asarray(psi),
        theta_soil=jnp.asarray(theta), runoff_surface=z, runoff_subsurface=z,
        snow_depth=z, snow_age=z, TgC=None, surface_water=z)


def _param_file(tmp_path, name="params.nc", body=b"hydraulic parameters"):
    p = tmp_path / name
    p.write_bytes(body)
    return p


def _bands(h):
    lo = np.broadcast_to(np.asarray(theta_from_psi(psi_dry_floor(h), h)),
                         (_NCOL, _NLAY))
    tr = np.asarray(h.theta_r)
    hi = np.broadcast_to(tr + (1.0 - 1.0e-4) * (np.asarray(h.theta_sat) - tr),
                         (_NCOL, _NLAY))
    return lo, hi


# --- stamp ------------------------------------------------------------------

def test_stamp_round_trips_and_absent_reads_none(tmp_path):
    st = _state(np.full((_NCOL, _NLAY), 0.3), np.full((_NCOL, _NLAY), -1.0))
    stamp = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                  _param_file(tmp_path))
    save_land_restart(tmp_path / "a.npz", st, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0, soil_hydraulics=stamp)
    save_land_restart(tmp_path / "b.npz", st, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0)
    _, meta = load_land_restart(tmp_path / "a.npz",
                                expected_land_mode="multilayer",
                                expected_ncol=_NCOL)
    assert meta["soil_hydraulics"] == stamp
    _, meta = load_land_restart(tmp_path / "b.npz",
                                expected_land_mode="multilayer",
                                expected_ncol=_NCOL)
    assert meta["soil_hydraulics"] is None


def test_stamp_identity_is_content_not_name(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                              _param_file(tmp_path / "a", "x.nc", b"one"))
    b = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                              _param_file(tmp_path / "b", "x.nc", b"two"))
    assert a["parameter_file"] == b["parameter_file"]
    assert not soil_hydraulics_stamps_match(a, b)
    # Provenance keys never enter the comparison.
    assert soil_hydraulics_stamps_match(a, dict(a, attested=True,
                                                attested_source_md5="f00"))
    with pytest.raises(ValueError, match="missing"):
        soil_hydraulics_stamps_match(a, {"version": 1})
    with pytest.raises(ValueError, match="unknown soil-hydraulics source"):
        soil_hydraulics_stamp("van_genuchten", "somewhere", _param_file(tmp_path))
    with pytest.raises(ValueError, match="retention curve"):
        soil_hydraulics_stamp("van_genuchtn", HYDRAULICS_SOURCE_CLM_MAP,
                              _param_file(tmp_path))


# --- conform ----------------------------------------------------------------

def test_conform_conserves_each_column_and_lands_in_the_band():
    h = _vg()
    lo, hi = _bands(h)
    theta = np.full((_NCOL, _NLAY), 0.25)
    theta[0, :4] = 0.01            # below theta_r: dry lift from the column
    theta[1, :3] = 0.47            # above theta_sat: fits in the column's room
    theta[2, :] = 0.46             # whole column over capacity: pond
    theta[3, 5] = 0.01             # outside the land mask: untouched
    theta[4, :] = 0.43             # exactly at theta_sat: capped just below, to pond
    land = np.array([True, True, True, False, True, True])
    new, pond, rep = conform_soil_water(theta, _DZ, h, land_mask=land)

    before = (theta * _DZ).sum(1)
    after = (new * _DZ).sum(1) + pond
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-15)
    assert np.all(new[land] >= lo[land] - 1e-15)
    assert np.all(new[land] <= hi[land] + 1e-15)
    np.testing.assert_array_equal(new[3], theta[3])
    np.testing.assert_array_equal(new[5], theta[5])
    assert pond[2] > 0 and pond[4] > 0 and np.all(pond[[0, 1, 3, 5]] == 0)
    np.testing.assert_allclose(new[2], hi[2])
    np.testing.assert_allclose(new[4], hi[4])
    assert np.all(new[land] < np.asarray(h.theta_sat)[land])
    assert rep["dry_columns"] == 1 and rep["pond_columns"] == 2
    assert rep["wet_columns"] == 3


def test_conform_accepts_a_column_on_the_floor_to_float32_rounding():
    """A column resting on the dry floor, stored in float32, reads a hair below
    it; that is rounding, not a column to refuse."""
    h = _vg()
    lo, _ = _bands(h)
    theta = np.full((_NCOL, _NLAY), 0.25)
    theta[0] = lo[0].astype(np.float32).astype(np.float64)
    assert np.any(theta[0] < lo[0])
    new, pond, _ = conform_soil_water(theta, _DZ, h)
    assert np.all(new[0] >= lo[0] - 1e-15)
    lifted = ((new[0] - theta[0]) * _DZ).sum()
    assert 0.0 <= lifted <= np.finfo(np.float32).eps * (lo[0] * _DZ).sum()


def test_conform_refuses_a_column_it_cannot_lift_without_creating_water():
    theta = np.full((_NCOL, _NLAY), 0.25)
    theta[0, :] = 0.01
    with pytest.raises(ValueError, match="dry floor"):
        conform_soil_water(theta, _DZ, _vg())


# --- conversion ---------------------------------------------------------------

def _ch_world_state():
    """A state evolved on Clapp-Hornberger soil, some layers below VG theta_r."""
    theta = np.tile(np.linspace(0.20, 0.32, _NLAY), (_NCOL, 1))
    theta[0, :2] = 0.04
    psi = np.asarray(psi_from_theta(jnp.asarray(theta), _ch()))
    return _state(theta, psi)


def _meta(stamp, regridded):
    return {"soil_hydraulics": stamp,
            "metadata": {"regridded_from": "src.npz"} if regridded else {}}


def test_unstamped_state_is_refused(tmp_path):
    with pytest.raises(ValueError, match="soil-hydraulics stamp"):
        convert_ic_soil_water(_ch_world_state(), {"metadata": {}}, _vg(),
                              soil_hydraulics_stamp(
                                  "van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                  _param_file(tmp_path)), _DZ)


def test_matching_stamp_without_column_signature_is_converted(tmp_path):
    """Equal file stamps alone do not prove equal parameters (the builder or
    its tables can change under the same file); without a per-column signature
    every column is converted."""
    stamp = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                  _param_file(tmp_path))
    st = _ch_world_state()
    out, rep = convert_ic_soil_water(st, _meta(stamp, False), _vg(), stamp, _DZ)
    assert rep is not None and rep["converted_columns"] == _NCOL


@pytest.mark.parametrize("same_stamp", [True, False])
def test_regridded_or_foreign_state_gets_this_runs_potential(tmp_path, same_stamp):
    """Equal stamps + regridded isolates the regridded lookup (it lives in the
    file's metadata dict, not at the top of the loader's meta)."""
    run = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                _param_file(tmp_path))
    src = run if same_stamp else soil_hydraulics_stamp(
        "clapp_hornberger", HYDRAULICS_SOURCE_SURFDATA_COSBY,
        _param_file(tmp_path, "cosby.nc", b"cosby"))
    h = _vg()
    out, rep = convert_ic_soil_water(_ch_world_state(), _meta(src, same_stamp),
                                     h, run, _DZ)
    assert rep is not None and rep["regridded"] is same_stamp
    psi = np.asarray(out.psi_soil)
    assert np.all(np.isfinite(psi.astype(np.float32)))
    assert np.all(psi >= np.asarray(psi_dry_floor(h)))
    np.testing.assert_allclose(np.asarray(theta_from_psi(out.psi_soil, h)),
                               np.asarray(out.theta_soil), rtol=1e-9, atol=1e-9)


def _one_step(st, h, dtype):
    cast = lambda a: jnp.asarray(a, dtype=dtype)
    z = jnp.zeros(_NCOL, dtype=dtype)
    grid = _GRID._replace(**{f: cast(getattr(_GRID, f)) for f in _GRID._fields})
    o = solve_richards(cast(st.psi_soil), cast(st.theta_soil), grid, h,
                       RichardsConfig(), z, jnp.zeros((_NCOL, _NLAY), dtype),
                       300.0, surface_water=cast(st.surface_water))
    assert o.theta_new.dtype == dtype
    return o


def _wet_ch_world_state():
    """Two layers above the VG theta_sat, room below: the wet rule moves the
    excess down without filling any layer to saturation."""
    theta = np.tile(np.linspace(0.20, 0.30, _NLAY), (_NCOL, 1))
    theta[:, :2] = 0.44
    psi = np.asarray(psi_from_theta(jnp.asarray(theta), _ch()))
    return _state(theta, psi)


_UNCONVERGED_PICARD = pytest.mark.xfail(strict=True, reason=(
    "pre-existing Richards defect: on a wet-over-dry front the fixed-count "
    "Picard iteration does not converge (it cycles between two states), and an "
    "unconverged iterate is not mass-conservative (about 0.5 mm here, 117 mm "
    "for a real column filled to exact saturation). Separate fix."))


@pytest.mark.parametrize("wet", [False, pytest.param(True, marks=_UNCONVERGED_PICARD)])
@pytest.mark.parametrize("dtype,atol_mm", [(jnp.float64, 1e-9), (jnp.float32, 1e-2)])
def test_first_step_after_conversion_closes_the_water_budget(
        tmp_path, dtype, atol_mm, wet):
    """Per column, through the solver: storage change + runoff == 0 with no
    input.  The carried Clapp-Hornberger potential fails the same check."""
    run = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                _param_file(tmp_path))
    src = soil_hydraulics_stamp("clapp_hornberger", HYDRAULICS_SOURCE_SURFDATA_COSBY,
                                _param_file(tmp_path, "cosby.nc", b"cosby"))
    st0 = _wet_ch_world_state() if wet else _ch_world_state()
    conv, rep = convert_ic_soil_water(st0, _meta(src, True), _vg(), run, _DZ)
    assert (rep["wet_columns"] > 0) is wet
    h = _vg(dtype)

    def residual_mm(st):
        o = _one_step(st, h, dtype)
        w0 = (np.asarray(st.theta_soil, np.float64) * _DZ).sum(1) \
            + np.asarray(st.surface_water, np.float64)
        w1 = (np.asarray(o.theta_new, np.float64) * _DZ).sum(1) \
            + np.asarray(o.surface_water, np.float64)
        out = (np.asarray(o.runoff_surface, np.float64)
               + np.asarray(o.runoff_subsurface, np.float64)) * 300.0 / 1e3
        return 1e3 * np.abs(w1 - w0 + out)

    assert residual_mm(conv).max() < atol_mm
    assert residual_mm(st0).max() > 1.0


def test_one_day_keeps_the_deep_water_that_the_carried_potential_loses(tmp_path):
    run = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                _param_file(tmp_path))
    src = soil_hydraulics_stamp("clapp_hornberger", HYDRAULICS_SOURCE_SURFDATA_COSBY,
                                _param_file(tmp_path, "cosby.nc", b"cosby"))
    st0 = _ch_world_state()
    conv, _ = convert_ic_soil_water(st0, _meta(src, True), _vg(), run, _DZ)
    h = _vg()

    z = jnp.zeros(_NCOL)
    sink = jnp.zeros((_NCOL, _NLAY))

    @jax.jit
    def day(psi, theta, sw):
        def body(c, _):
            o = solve_richards(*c[:2], _GRID, h, RichardsConfig(), z, sink,
                               300.0, surface_water=c[2])
            return (o.psi_new, o.theta_new, o.surface_water), None
        return jax.lax.scan(body, (psi, theta, sw), None, length=288)[0]

    def deep_after_one_day(st):
        _, theta, _ = day(st.psi_soil, st.theta_soil, st.surface_water)
        return float(np.asarray(theta)[:, -1].mean())

    deep0 = float(np.asarray(st0.theta_soil)[:, -1].mean())
    assert abs(deep_after_one_day(conv) - deep0) < 0.005
    assert deep0 - deep_after_one_day(st0) > 0.05


@_UNCONVERGED_PICARD
def test_overflowing_column_closes_the_budget_on_its_first_step(tmp_path):
    """A column holding more than it can store: saturated throughout, the rest
    in the surface pond, which the first step keeps up to pond_max and runs
    off beyond it."""
    run = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                _param_file(tmp_path))
    src = soil_hydraulics_stamp("clapp_hornberger", HYDRAULICS_SOURCE_SURFDATA_COSBY,
                                _param_file(tmp_path, "cosby.nc", b"cosby"))
    theta = np.full((_NCOL, _NLAY), 0.45)
    st0 = _state(theta, np.asarray(psi_from_theta(jnp.asarray(theta), _ch())))
    conv, rep = convert_ic_soil_water(st0, _meta(src, True), _vg(), run, _DZ)
    assert rep["pond_columns"] == _NCOL
    o = _one_step(conv, _vg(), jnp.float64)
    w0 = (np.asarray(conv.theta_soil) * _DZ).sum(1) + np.asarray(conv.surface_water)
    w1 = (np.asarray(o.theta_new) * _DZ).sum(1) + np.asarray(o.surface_water)
    out = (np.asarray(o.runoff_surface) + np.asarray(o.runoff_subsurface)) * 300.0 / 1e3
    assert 1e3 * np.abs(w1 - w0 + out).max() < 1e-6


def test_column_signature_converts_only_the_columns_whose_soil_changed(tmp_path):
    """Equal stamps, not regridded: only columns whose parameters differ from
    the writer's get this run's potential; the rest are carried verbatim."""
    h = _vg()
    stamp = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                  _param_file(tmp_path))
    st = _ch_world_state()
    sig = soil_hydraulics_column_signature(h, _NCOL, _NLAY)
    out, rep = convert_ic_soil_water(st, _meta(stamp, False), h, stamp, _DZ,
                                     file_column_sig=sig)
    assert out is st and rep is None
    changed = h._replace(theta_sat=h.theta_sat.at[2, 0].set(0.41))
    out, rep = convert_ic_soil_water(st, _meta(stamp, False), changed, stamp,
                                     _DZ, file_column_sig=sig)
    assert rep["converted_columns"] == 1
    moved = np.any(np.asarray(out.psi_soil) != np.asarray(st.psi_soil), axis=1)
    np.testing.assert_array_equal(moved, np.arange(_NCOL) == 2)
    # The writers' Clapp-Hornberger config signs too, and differently.
    ch = soil_hydraulics_column_signature(_ch(), _NCOL, _NLAY)
    assert ch.shape == (_NCOL,) and not np.any(ch == sig)
    # float32 and float64 builds of the same parameters sign alike.
    np.testing.assert_array_equal(
        sig, soil_hydraulics_column_signature(_vg(jnp.float32), _NCOL, _NLAY))


def test_signature_round_trips_through_the_restart(tmp_path):
    h = _vg()
    st = _state(np.full((_NCOL, _NLAY), 0.3), np.full((_NCOL, _NLAY), -1.0))
    save_land_restart(tmp_path / "s.npz", st, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0, hydraulics=h,
                      soil_hydraulics=soil_hydraulics_stamp(
                          "van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                          _param_file(tmp_path)))
    _, meta = load_land_restart(tmp_path / "s.npz",
                                expected_land_mode="multilayer",
                                expected_ncol=_NCOL)
    np.testing.assert_array_equal(meta["soil_hydraulics_column_sig"],
                                  soil_hydraulics_column_signature(h, _NCOL, _NLAY))


def test_overflow_without_a_surface_water_store_is_refused(tmp_path):
    run = soil_hydraulics_stamp("van_genuchten", HYDRAULICS_SOURCE_CLM_MAP,
                                _param_file(tmp_path))
    theta = np.full((_NCOL, _NLAY), 0.45)
    st = _state(theta, np.full((_NCOL, _NLAY), -0.1))._replace(surface_water=None)
    with pytest.raises(ValueError, match="surface-water store"):
        convert_ic_soil_water(st, _meta(run, True), _vg(), run, _DZ)
