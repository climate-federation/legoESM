"""Spectral (T21) PE lane of the atmosphere matrix: EVERY spectral-PE row
runs the CAM-EUL reference (user decision 2026-11-27, "option 1"): DCMIP
2008 Sect. 2.1 Table 4 del4 K4, no exponential spectral filter, full band
n <= N (dealiasing_fraction 1.0).  The old lane default (2.338e15 (21/n)^4
+ order-8 0.01 per-step filter, 2/3 state truncation) left the T21 JW wave
essentially absent (day-9 min p_s 999.45 vs 979.96 hPa).  Needs
JAX_ENABLE_X64=1.
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
    TestCases, registered runners) and pin the constructed config: Table 4
    K4 2.0e16 (gw31: / 125^3), no exp filter, full band; transport rows
    build no spectral-PE model at all."""
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
        assert cfg.hyperdiff_order == 2, tc.case
        assert cfg.dealiasing_fraction == 1.0, tc.case
        assert cfg.spectral_filter_strength == 0.0, tc.case
        assert cfg.T_damping_pressure_surfaces == (
            tc.case == "rest_state_topo"), tc.case
        hd = 2.0e16 / (125.0 ** 3 if tc.case == "gravity_wave_3_1" else 1.0)
        assert cfg.hyperdiff_coeff == pytest.approx(hd, rel=1e-15), tc.case
    assert {"held_suarez", "held_suarez_topo", "amip", "gravity_wave_3_1",
            "rest_state_topo", "dcmip_transport_11", "dcmip_transport_12",
            "dcmip_transport_13"} | mm.SPECTRAL_TABLE4_CASES <= seen_cases


@pytest.mark.parametrize("case", sorted(
    ["inertio_gravity_3_2", "mountain_rossby_5_0", "rossby_haurwitz_6_0"]))
def test_dcmip2008_rows_run_table4_eul_dissipation(mm, monkeypatch, tmp_path,
                                                  case):
    cfg = _lane(mm, monkeypatch, tmp_path, case)["config"]
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.hyperdiff_coeff == 2.0e16          # Table 4, T21
    assert cfg.hyperdiff_order == 2               # del4
    assert cfg.dealiasing_fraction == 1.0         # full band, as CAM-EUL


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
    cfg = _lane(mm, monkeypatch, tmp_path, case, kwargs, vert)["config"]
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.hyperdiff_coeff == 2.0e16          # Table 4, T21
    assert cfg.hyperdiff_order == 2               # del4
    assert cfg.dealiasing_fraction == 1.0
    assert not cfg.T_damping_pressure_surfaces


@pytest.mark.parametrize("case", sorted(
    ["baroclinic", "baroclinic_steady", "rotated_baroclinic",
     "rotated_steady"]))
def test_baroclinic_note_says_borrowed_not_dcmip2016(mm, case):
    hd, filt, note = mm._spectral_pe_dissipation(case, 21)
    assert (hd, filt) == (2.0e16, 0.0)
    assert "not a DCMIP-2016" in note and "K2=2.5e5 NOT applied" in note


@pytest.mark.parametrize("case,kwargs,hd", [
    ("rest_state_topo", {"h_0": 2000.0}, 2.0e16),
    ("gravity_wave_3_1", {}, 2.0e16 / 125.0 ** 3),
])
def test_other_baroclinic_lane_rows_on_cam_eul(mm, monkeypatch, tmp_path,
                                               case, kwargs, hd):
    cfg = _lane(mm, monkeypatch, tmp_path, case, kwargs)["config"]
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.dealiasing_fraction == 1.0
    assert cfg.hyperdiff_coeff == pytest.approx(hd, rel=1e-15)


@pytest.mark.parametrize("case", ["held_suarez", "held_suarez_topo", "amip",
                                  "gravity_wave_3_1", "rest_state_topo"])
def test_other_rows_helper_returns_table4(mm, case):
    hd, filt, note = mm._spectral_pe_dissipation(case, 21)
    assert (hd, filt) == (2.0e16, 0.0)
    assert "Table 4" in note and "full band" in note


def test_matrix_table_is_the_library_table(mm):
    from legoesm.atmosphere.dynamics.gcm import spectral_pe as spe
    assert mm.DCMIP2008_EUL_K4 is spe.CAM_EUL_DCMIP2008_K4


@pytest.mark.parametrize("n_max,k4", [(21, 2.0e16), (42, 1.0e16)])
def test_table4_lookup_matrix_truncations(mm, n_max, k4):
    """DCMIP 2008 Sect. 2.1 Table 4 rows the matrix lanes accept."""
    hd, filt, note = mm._spectral_pe_dissipation("mountain_rossby_5_0", n_max)
    assert (hd, filt) == (k4, 0.0)
    assert "Table 4" in note


@pytest.mark.parametrize("n_max", [85, 106, 170, 340])
def test_matrix_refuses_table4_rows_beyond_its_checked_step(mm, n_max):
    """Vote 2026-11-27: the fixed 600 s step is unchecked above T42."""
    for case in ("held_suarez", "amip", "baroclinic", "mountain_rossby_5_0"):
        with pytest.raises(ValueError, match="fixed explicit step"):
            mm._spectral_pe_dissipation(case, n_max)


@pytest.mark.parametrize("case", ["held_suarez", "amip"])
def test_matrix_runner_refuses_t85_before_building_a_grid(mm, monkeypatch,
                                                          tmp_path, case):
    import legoesm.grids.gaussian as gg

    def _no_grid(*a, **k):
        raise AssertionError("grid built before the refusal")
    monkeypatch.setattr(gg, "create_gaussian_grid", _no_grid)
    tc = mm.TestCase("hydrostatic", case, "spectral", "T85",
                     "hybrid", 2.0, 0.5, {})
    with pytest.raises(ValueError, match="fixed explicit step"):
        _dispatch(mm, monkeypatch, tmp_path, tc)


@pytest.mark.parametrize("case", ["mountain_rossby_5_0", "baroclinic",
                                  "held_suarez", "amip"])
def test_table4_refuses_an_unlisted_truncation(mm, case):
    with pytest.raises(ValueError, match="Table 4"):
        mm._spectral_pe_dissipation(case, 63)


def test_ig_6_3_0_response_reaches_the_other_grids(mm, monkeypatch, tmp_path):
    """6-3-0 at T21 through the matrix lane: interior (below the top level)
    max |V| at 0.25 d.  Old lane default (exp filter, 2/3 cut) gave 4.4 m/s;
    Table 4 with the 2/3 cut 6.0; option 1 (full band) 6.49; cube / lat-lon /
    ico peak 6.4 / 6.4 / 6.5."""
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
