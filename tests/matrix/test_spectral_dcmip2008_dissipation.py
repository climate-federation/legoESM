"""Spectral (T21) lane of ``run_baroclinic``: the DCMIP-2008 dry rows run the
reference EUL dissipation of their own document (DCMIP 2008 Table 4 del4 K4,
no exponential filter), and so do the baroclinic-wave rows (baroclinic,
baroclinic_steady, rotated_baroclinic, rotated_steady: Table 4 borrowed via
the DCMIP 2008 Jablonowski & Williamson 2006 wave tests); every other row
keeps the lane default.

The lane's per-step exp filter (order 8, 0.01 retention, per 600 s step)
e-folds n = 14 in ~55 min at T21 and took ~25 % off the 5-0-0 / 6-3-0
response vs the cube / lat-lon / ico rows.  Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")


@pytest.fixture(scope="module")
def mm():
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    spec = importlib.util.spec_from_file_location("_atmmatrix_specdiss", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


class _Stop(Exception):
    pass


def _dispatch(mm, monkeypatch, tmp_path, tc):
    """Run ``tc`` through its registered matrix runner up to the time loop;
    return the spectral-PE config (if any model was built), the IC,
    step_fn and dt it would integrate with."""
    import legoesm.atmosphere.dynamics.gcm.spectral_pe as spe
    seen = {}
    real = spe.SpectralPrimitiveEquationModel

    class Spy(real):
        def __init__(self, grid, sigma, config, *a, **k):
            seen["config"] = config
            super().__init__(grid, sigma, config, *a, **k)

    def fake(step_fn, state, dt, n_steps, *a, **k):
        seen.update(step_fn=step_fn, state=state, dt=dt)
        raise _Stop

    with monkeypatch.context() as m:      # fresh spy per dispatch
        m.setattr(spe, "SpectralPrimitiveEquationModel", Spy)
        m.setattr(mm, "_run_timeloop", fake)
        with pytest.raises(_Stop):
            mm.RUNNERS[tc.case](tc, tmp_path, tc.quick_days)
    return seen


def _lane(mm, monkeypatch, tmp_path, case, kwargs=None, vert="hybrid"):
    tc = mm.TestCase("hydrostatic", case, "spectral", "T21", vert,
                     2.0, 0.5, dict(kwargs or {}))
    return _dispatch(mm, monkeypatch, tmp_path, tc)


def _spectral_rows(mm):
    return [t for t in mm._build_test_matrix()
            if t.grid_type == "spectral" and t.equation_set == "hydrostatic"]


def test_every_spectral_hydro_row_on_its_voted_dissipation(mm, monkeypatch,
                                                           tmp_path):
    """Dispatch EVERY spectral hydrostatic matrix row (actual matrix
    TestCases, registered runners) and pin the constructed config:
    R = DCMIP 2008 Table 4 K4 2.0e16, no exp filter (DCMIP-2008 dry rows
    + the baroclinic-wave rows, vote 2026-11-26); F = lane default
    2.338e15 + order-8 0.01 filter (gw31: K4 / 125^3); transport rows
    build no spectral-PE model at all."""
    R = {"inertio_gravity_3_2", "mountain_rossby_5_0", "rossby_haurwitz_6_0",
         "baroclinic", "baroclinic_steady", "rotated_baroclinic",
         "rotated_steady"}
    rows = _spectral_rows(mm)
    seen_cases = set()
    for tc in rows:
        assert tc.resolution == "T21", tc
        seen_cases.add(tc.case)
        cfg = _dispatch(mm, monkeypatch, tmp_path / f"{tc.case}_"
                        f"{tc.vertical_coord}", tc).get("config")
        if tc.case.startswith("dcmip_transport"):
            assert cfg is None, tc.case
            continue
        assert cfg is not None, tc.case
        assert cfg.spectral_filter_order == 8, tc.case
        assert cfg.hyperdiff_order == 2, tc.case
        assert cfg.dealiasing_fraction == 0.667, tc.case
        assert cfg.T_damping_pressure_surfaces == (
            tc.case == "rest_state_topo"), tc.case
        if tc.case in R:
            assert (cfg.hyperdiff_coeff, cfg.spectral_filter_strength) == (
                2.0e16, 0.0), tc.case
        else:
            hd = 2.338e15 / (125.0 ** 3 if tc.case == "gravity_wave_3_1"
                             else 1.0)
            assert cfg.spectral_filter_strength == 0.01, tc.case
            assert cfg.hyperdiff_coeff == pytest.approx(hd, rel=1e-15), tc.case
    assert R <= seen_cases
    assert {"held_suarez", "held_suarez_topo", "amip", "gravity_wave_3_1",
            "rest_state_topo", "dcmip_transport_11", "dcmip_transport_12",
            "dcmip_transport_13"} <= seen_cases


@pytest.mark.parametrize("case", sorted(
    ["inertio_gravity_3_2", "mountain_rossby_5_0", "rossby_haurwitz_6_0"]))
def test_dcmip2008_rows_run_table4_eul_dissipation(mm, monkeypatch, tmp_path,
                                                  case):
    cfg = _lane(mm, monkeypatch, tmp_path, case)["config"]
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.hyperdiff_coeff == 2.0e16          # Table 4, T21
    assert cfg.hyperdiff_order == 2               # del4
    assert cfg.dealiasing_fraction == 0.667       # 2/3 band untouched


# Every spectral row run_baroclinic serves, with the matrix kwargs.
_BAROCLINIC_ROWS = [
    ("baroclinic", "hybrid", {}),
    ("baroclinic", "sigma", {}),
    ("baroclinic_steady", "sigma", {"perturbed": False}),
    ("rotated_baroclinic", "hybrid",
     {"alpha": 0.7853981633974483, "perturbed": True}),
    ("rotated_steady", "hybrid",
     {"alpha": 0.7853981633974483, "perturbed": False}),
]


@pytest.mark.parametrize("case,vert,kwargs", _BAROCLINIC_ROWS)
def test_baroclinic_rows_run_table4_eul_dissipation(mm, monkeypatch, tmp_path,
                                                  case, vert, kwargs):
    """Vote 2026-11-26 (2:1 switch): the baroclinic-wave rows drop the per-step
    exp filter + del4 2.338e15 for DCMIP 2008 Table 4 K4 (no filter)."""
    cfg = _lane(mm, monkeypatch, tmp_path, case, kwargs, vert)["config"]
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.hyperdiff_coeff == 2.0e16          # Table 4, T21
    assert cfg.hyperdiff_order == 2               # del4
    assert cfg.dealiasing_fraction == 0.667       # 2/3 band untouched
    assert not cfg.T_damping_pressure_surfaces


@pytest.mark.parametrize("case", sorted(
    ["baroclinic", "baroclinic_steady", "rotated_baroclinic",
     "rotated_steady"]))
def test_baroclinic_note_says_borrowed_not_dcmip2016(mm, case):
    hd, filt, note = mm._spectral_pe_dissipation(case, 21)
    assert (hd, filt) == (2.0e16, 0.0)
    assert "not a DCMIP-2016" in note and "K2=2.5e5 NOT applied" in note


@pytest.mark.parametrize("case,kwargs,hd", [
    ("rest_state_topo", {"h_0": 2000.0}, 2.338e15),
    ("gravity_wave_3_1", {}, 2.338e15 / 125.0 ** 3),
])
def test_other_rows_keep_the_lane_default(mm, monkeypatch, tmp_path,
                                          case, kwargs, hd):
    cfg = _lane(mm, monkeypatch, tmp_path, case, kwargs)["config"]
    assert cfg.spectral_filter_strength == 0.01
    assert cfg.spectral_filter_order == 8
    assert cfg.hyperdiff_coeff == pytest.approx(hd, rel=1e-15)


@pytest.mark.parametrize("case", [
    "held_suarez", "held_suarez_topo", "amip", "gravity_wave_3_1",
    "rest_state_topo", "dcmip_transport_11", "dcmip_transport_12",
    "dcmip_transport_13"])
def test_f_rows_helper_returns_lane_default(mm, case):
    """Rows that stay on the lane default (F).  HS / AMIP build their own
    SpectralPEConfig with the same literals (not via the helper)."""
    hd, filt, note = mm._spectral_pe_dissipation(case, 21)
    assert (hd, filt, note) == (2.338e15, 0.01, "")
    assert case not in mm.SPECTRAL_TABLE4_CASES


def test_table4_case_set_is_exactly_the_voted_rows(mm):
    assert mm.SPECTRAL_TABLE4_CASES == frozenset({
        "inertio_gravity_3_2", "mountain_rossby_5_0", "rossby_haurwitz_6_0",
        "baroclinic", "baroclinic_steady", "rotated_baroclinic",
        "rotated_steady"})


@pytest.mark.parametrize("n_max,k4", [
    (21, 2.0e16), (42, 1.0e16), (85, 1.0e15), (106, 0.5e15),
    (170, 1.5e14), (340, 1.5e13)])
def test_table4_lookup_every_truncation(mm, n_max, k4):
    """DCMIP 2008 Sect. 2.1 Table 4, all six EUL rows."""
    hd, filt, note = mm._spectral_pe_dissipation("mountain_rossby_5_0", n_max)
    assert (hd, filt) == (k4, 0.0)
    assert "Table 4" in note


def test_table4_refuses_an_unlisted_truncation(mm):
    with pytest.raises(ValueError, match="Table 4"):
        mm._spectral_pe_dissipation("mountain_rossby_5_0", 63)
    with pytest.raises(ValueError, match="Table 4"):
        mm._spectral_pe_dissipation("baroclinic", 63)
    assert mm._spectral_pe_dissipation("held_suarez", 63)[1] == 0.01


def test_ig_6_3_0_response_reaches_the_other_grids(mm, monkeypatch, tmp_path):
    """6-3-0 at T21 through the matrix lane: interior (below the top level)
    max |V| at 0.25 d.  Lane default (exp filter) gives 4.4 m/s; the
    reference dissipation 6.0; cube / lat-lon / ico peak 6.4 / 6.4 / 6.5."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
    s = _lane(mm, monkeypatch, tmp_path, "inertio_gravity_3_2")
    assert s["dt"] == 600.0
    model_state = s["state"]
    for _ in range(36):                           # 0.25 d
        model_state = s["step_fn"](model_state, s["dt"])
    from legoesm.grids.gaussian import create_gaussian_grid
    grid = create_gaussian_grid(21)
    sigma = mm._create_vertical(mm.DEFAULT_NLEV, "hybrid")
    f = spectral_pe_to_grid(model_state, grid, sigma)
    speed = np.sqrt(np.asarray(f["u"]) ** 2 + np.asarray(f["v"]) ** 2)
    interior = float(speed[..., 1:].max())
    assert 5.5 < interior < 6.8, interior
