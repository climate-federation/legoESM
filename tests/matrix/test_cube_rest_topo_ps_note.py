"""The cube rest_state_topo row records why its surface-pressure drift is
larger than the other grids' (FV3-duogrid corner halo imbalance + Lin PGF
truncation on unfiltered DCMIP 2012 2-0-0 terrain at C36).  Vote 2026-11-27
(codex + author): accept and document, keep the terrain unfiltered.  The note
must reach the cube branch of the runner that the rest_state_topo dispatch
actually calls.  The cube held_suarez_topo row, and ONLY that row, runs the
Lin (1997) sec. 3 log-p hydrostatic PGF option (user decision 2026-11-27)
against the p**kappa truncation on its isothermal start, and says so in its
notes."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")


@pytest.fixture(scope="module")
def matrix_mod():
    spec = importlib.util.spec_from_file_location("_atmmatrix_rest", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:                                    # pragma: no cover
        sys.modules.pop(spec.name, None)
        raise
    yield mod
    sys.modules.pop(spec.name, None)


def test_note_names_the_diagnosed_mechanism(matrix_mod):
    note = matrix_mod.CUBE_REST_TOPO_PS_NOTE
    for word in ("duogrid", "ext_scalar", "geopk", "Lin PGF", "unfiltered",
                 "h0**2", "discretely balanced"):
        assert word in note, word


class _Stop(Exception):
    pass


def _cube_row_notes(mm, monkeypatch, tmp_path, case):
    """Run each registered cube row of ``case`` up to its results.txt write,
    with the time loop stubbed, and return the list of notes strings, one per row."""
    tcs = [t for t in mm._build_test_matrix()
           if t.case == case and t.grid_type == "cubed_sphere"]
    assert tcs, case
    return [_one_row_notes(mm, monkeypatch, tmp_path, tc) for tc in tcs]


def _one_row_notes(mm, monkeypatch, tmp_path, tc):
    seen = {}

    def fake_loop(step_fn, state, dt, n_steps, *a, **k):
        return state, [], {"mass": [1.0, 1.0]}, 0.0, True

    def fake_write(output_dir, row, *a, **k):
        seen.update(row)
        raise _Stop

    with monkeypatch.context() as m:
        m.setattr(mm, "_run_timeloop", fake_loop)
        m.setattr(mm, "_write_results_txt", fake_write)
        with pytest.raises(_Stop):
            mm.RUNNERS[tc.case](tc, tmp_path, tc.quick_days)
    return seen["notes"]


def test_cube_rest_row_records_the_note(matrix_mod, monkeypatch, tmp_path):
    notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path,
                            "rest_state_topo")
    assert all(matrix_mod.CUBE_REST_TOPO_PS_NOTE in n for n in notes)


def test_other_cube_rows_do_not_carry_it(matrix_mod, monkeypatch, tmp_path):
    notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path,
                            "mountain_rossby_5_0")
    assert all(matrix_mod.CUBE_REST_TOPO_PS_NOTE not in n for n in notes)


def test_hs_topo_note_names_the_diagnosed_mechanism(matrix_mod):
    note = matrix_mod.CUBE_HS_TOPO_WIND_NOTE
    for word in ("Lin 1997 sec. 3 log-p", "OPTION", "departs from FV3",
                 "p**kappa", "constant theta", "isothermal", "unfiltered",
                 "every other cube row keeps p**kappa",
                 "adiabatic panel-edge wind", "flat HS agrees",
                 "PLAUSIBLE", "accepted"):
        assert word in note, word


def test_cube_hs_topo_row_records_its_note(matrix_mod, monkeypatch,
                                           tmp_path):
    notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path,
                            "held_suarez_topo")
    for n in notes:
        assert matrix_mod.CUBE_HS_TOPO_WIND_NOTE in n
        assert matrix_mod.CUBE_REST_TOPO_PS_NOTE not in n


def test_flat_cube_hs_row_does_not_carry_it(matrix_mod, monkeypatch,
                                            tmp_path):
    notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path, "held_suarez")
    assert all(matrix_mod.CUBE_HS_TOPO_WIND_NOTE not in n for n in notes)


# The adapter's own record of the flag (written iff hydrostatic_logp=True).
_LOGP_MARK = "hydrostatic PGF: Lin 1997 sec. 3 log-p contour OPTION"


def test_only_the_hs_topo_cube_row_runs_logp(matrix_mod, monkeypatch,
                                             tmp_path):
    """held_suarez_topo builds its duo with hydrostatic_logp=True; the other
    cube duo hydrostatic rows keep FV3's p**kappa default."""
    import scripts.matrix._cube_duo_adapters as A
    seen = []
    real = A.build_duo_hydro

    def spy(*a, **k):
        duo = real(*a, **k)
        seen.append(bool(duo.model.config.hydrostatic_logp))
        return duo

    monkeypatch.setattr(A, "build_duo_hydro", spy)
    for case, want in (("held_suarez_topo", True), ("rest_state_topo", False),
                       ("held_suarez", False)):
        seen.clear()
        notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path, case)
        assert seen and all(f is want for f in seen), (case, seen)
        assert all((_LOGP_MARK in n) is want for n in notes), case
