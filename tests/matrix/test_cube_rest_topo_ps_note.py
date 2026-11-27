"""The cube rest_state_topo row records why its surface-pressure drift is
larger than the other grids' (FV3-duogrid corner halo imbalance + Lin PGF
truncation on unfiltered DCMIP 2012 2-0-0 terrain at C36).  The cube
held_suarez_topo row likewise records why its early wind transient is larger
(Lin 1997 hydrostatic PGF truncation on the isothermal HS IC).  Vote 2026-11-27
(codex + author): accept and document, keep the terrain unfiltered.  The note
must reach the cube branch of the runner that the rest_state_topo dispatch
actually calls."""
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
    with the time loop stubbed, and return the notes strings joined."""
    tcs = [t for t in mm._build_test_matrix()
           if t.case == case and t.grid_type == "cubed_sphere"]
    assert tcs, case
    return "\n".join(_one_row_notes(mm, monkeypatch, tmp_path, tc)
                     for tc in tcs)


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
    assert matrix_mod.CUBE_REST_TOPO_PS_NOTE in notes


def test_other_cube_rows_do_not_carry_it(matrix_mod, monkeypatch, tmp_path):
    notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path,
                            "mountain_rossby_5_0")
    assert matrix_mod.CUBE_REST_TOPO_PS_NOTE not in notes


def test_hs_topo_note_names_the_diagnosed_mechanism(matrix_mod):
    note = matrix_mod.CUBE_HS_TOPO_WIND_NOTE
    for word in ("Lin 1997 PGF", "p**kappa", "constant theta", "isothermal",
                 "unfiltered", "h0**2", "use_logp is nonhydrostatic-only"):
        assert word in note, word


def test_cube_hs_topo_row_records_its_note(matrix_mod, monkeypatch,
                                           tmp_path):
    notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path,
                            "held_suarez_topo")
    assert matrix_mod.CUBE_HS_TOPO_WIND_NOTE in notes
    assert matrix_mod.CUBE_REST_TOPO_PS_NOTE not in notes


def test_flat_cube_hs_row_does_not_carry_it(matrix_mod, monkeypatch,
                                            tmp_path):
    notes = _cube_row_notes(matrix_mod, monkeypatch, tmp_path, "held_suarez")
    assert matrix_mod.CUBE_HS_TOPO_WIND_NOTE not in notes
