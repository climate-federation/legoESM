"""Direct test for ``scripts/validate/mpas_ledger_report.py`` (#1354)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "mpas_ledger_report.py"

_PROCS = ("turbulence", "convection", "microphysics", "radiation",
          "other_physics", "clips", "dynamics")


def _load():
    spec = importlib.util.spec_from_file_location("_ledrep", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ledrep"] = mod
    spec.loader.exec_module(mod)
    return mod


def _artifact(tmp_path, rates, area=None, name="budget_ledger_columns.npz",
              fingerprint=True, n_steps=768, cell_partitioned=None):
    """Write an artifact.  ``area=None`` reproduces a PRE-#1354 file, which
    carried no weights at all.  ``fingerprint=False`` reproduces the first
    draft of the writer, which shipped weights but nothing to prove they were
    the ones the energy tracker used."""
    from legoesm.parallel.geometry_consistency import content_hash48
    path = tmp_path / name
    kw = {}
    if area is not None:
        w = np.asarray(area, dtype=np.float64).ravel()
        kw["area_cell"] = w
        if fingerprint:
            kw["area_hash48"] = np.asarray(content_hash48(w))
    if cell_partitioned is not None:
        kw["cell_partitioned"] = bool(cell_partitioned)
    np.savez(path, ledger_rates=rates, processes=np.asarray(_PROCS),
             columns=np.asarray(("water_kg_m2_s", "energy_W_m2")),
             n_steps=n_steps, day=3.0, **kw)
    return path


def _equal_area(ncol):
    """Weights that make the weighted reduction agree with the unweighted one,
    so a test about something else is not silently also testing weighting."""
    return np.full(ncol, 4.0)


def test_global_mean_reproduces_a_planted_row(tmp_path, capsys):
    """A planted 18 W/m2 in one row must come out as exactly that row's mean.

    Two columns with unequal values, so the reduction is exercised rather than
    passed through.
    """
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1] = 12.0   # convection energy, column 0
    rates[1, 1, 1] = 24.0   # convection energy, column 1 -> mean 18
    mod = _load()
    art = _artifact(tmp_path, rates, _equal_area(2))
    assert mod.main([str(art)]) == 0
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if l.strip().startswith("convection"))
    assert "18.000" in row, row
    total = next(l for l in out.splitlines() if "TOTAL" in l)
    assert "18.000" in total, total


def test_water_column_is_converted_to_mm_per_day(tmp_path, capsys):
    """1 kg/m2/s planted -> 86400 mm/day printed; the unit label is load-bearing."""
    rates = np.zeros((1, 7, 2))
    rates[0, 0, 0] = 1.0
    mod = _load()
    mod.main([str(_artifact(tmp_path, rates, _equal_area(1)))])
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if l.strip().startswith("turbulence"))
    assert "86400.0000" in row, row


# --- the reduction is part of the answer (#1354) ----------------------------
# These exist because an unweighted row was differenced against an
# area-weighted dE/dt and produced a published attribution that had to be
# retracted.  The tool must not be able to make that mistake silently again.


def test_refuses_when_the_reduction_is_undecidable(tmp_path):
    """A pre-#1354 artifact has no weights; printing anyway is the bug."""
    import pytest
    mod = _load()
    art = _artifact(tmp_path, np.zeros((3, 7, 2)))
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art)])
    assert "REFUSING" in str(exc.value)


def test_zero_length_area_cell_is_treated_as_absent(tmp_path):
    """The writer's "this run had no weights" sentinel must also refuse."""
    import pytest
    mod = _load()
    art = _artifact(tmp_path, np.zeros((3, 7, 2)), np.zeros(0))
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art)])
    assert "REFUSING" in str(exc.value)


def test_area_weighting_changes_the_answer_and_matches_by_hand(tmp_path, capsys):
    """On unequal cells the weighted row differs from the unweighted one.

    Non-vacuity: the planted values and areas are chosen so the two reductions
    disagree by 4 W/m2, so swapping the weighted reduction back for a plain
    mean fails this test rather than passing it by luck.
    """
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1] = 10.0
    rates[1, 1, 1] = 30.0                       # unweighted mean = 20
    area = np.array([3.0, 1.0])                 # weighted = (30+30)/4 = 15
    mod = _load()
    assert mod.main([str(_artifact(tmp_path, rates, area))]) == 0
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines()
               if l.strip().startswith("convection"))
    assert "15.000" in row, row
    assert "20.000" not in row, row
    assert "area-weighted (weights from the artifact)" in out


def test_areacell_file_overrides_and_is_reported(tmp_path, capsys):
    """Historical artifacts stay readable by supplying the mesh separately."""
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1] = 10.0
    rates[1, 1, 1] = 30.0
    wpath = tmp_path / "areaCell.npy"
    np.save(wpath, np.array([3.0, 1.0]))
    mod = _load()
    art = _artifact(tmp_path, rates)             # no weights inside
    assert mod.main([str(art), "--areacell", str(wpath)]) == 0
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines()
               if l.strip().startswith("convection"))
    assert "15.000" in row, row
    assert "areaCell.npy" in out


def test_wrong_mesh_size_weights_are_rejected(tmp_path):
    """Weights from a different mesh would silently mis-weight every row."""
    import pytest
    mod = _load()
    art = _artifact(tmp_path, np.zeros((3, 7, 2)), np.ones(5))
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art)])
    assert "wrong mesh" in str(exc.value)


def test_allow_unweighted_prints_but_stamps_the_provenance(tmp_path, capsys):
    """The escape hatch must be visible in the output, not just in the flags."""
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1] = 10.0
    rates[1, 1, 1] = 30.0
    mod = _load()
    art = _artifact(tmp_path, rates)
    assert mod.main([str(art), "--allow-unweighted"]) == 3
    out = capsys.readouterr().out
    assert "UNWEIGHTED" in out
    row = next(l for l in out.splitlines()
               if l.strip().startswith("convection"))
    assert "20.000" in row, row


# --- findings from the dual review of this very script ----------------------
# codex rated the unverified --areacell override HIGH; GLM rated it MED and
# supplied the remedy (ship a positional fingerprint).  Both are pinned here.


def test_override_that_is_not_this_runs_mesh_is_rejected(tmp_path):
    """Same cell COUNT, different weights -> refuse, not a quiet wrong mean."""
    import pytest
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1], rates[1, 1, 1] = 10.0, 30.0
    art = _artifact(tmp_path, rates, np.array([3.0, 1.0]))
    wpath = tmp_path / "other_mesh.npy"
    np.save(wpath, np.array([1.0, 3.0]))      # a PERMUTATION: length agrees
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art), "--areacell", str(wpath)])
    assert "does NOT match" in str(exc.value)


def test_override_equal_to_the_shipped_weights_is_accepted(tmp_path, capsys):
    """The fingerprint must ACCEPT the right array, or it is just a blocker.

    Non-vacuity partner of the test above: same code path, opposite verdict.
    """
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1], rates[1, 1, 1] = 10.0, 30.0
    art = _artifact(tmp_path, rates, np.array([3.0, 1.0]))
    wpath = tmp_path / "same_mesh.npy"
    np.save(wpath, np.array([3.0, 1.0]))
    mod = _load()
    assert mod.main([str(art), "--areacell", str(wpath)]) == 0
    out = capsys.readouterr().out
    assert "fingerprint matches the run" in out
    row = next(l for l in out.splitlines()
               if l.strip().startswith("convection"))
    assert "15.000" in row, row


def test_override_on_a_pre_fingerprint_artifact_is_marked_unverified(
        tmp_path, capsys):
    """Old artifacts stay usable, but the output must not claim more than it
    can prove."""
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1], rates[1, 1, 1] = 10.0, 30.0
    art = _artifact(tmp_path, rates, np.array([3.0, 1.0]), fingerprint=False)
    wpath = tmp_path / "mesh.npy"
    np.save(wpath, np.array([3.0, 1.0]))
    mod = _load()
    assert mod.main([str(art), "--areacell", str(wpath)]) == 0
    assert "UNVERIFIED" in capsys.readouterr().out


def test_an_archive_passed_as_areacell_is_rejected(tmp_path):
    """np.load on a .npz returns an NpzFile, which np.asarray silently
    mangles into a 0-d object array (GLM)."""
    import pytest
    art = _artifact(tmp_path, np.zeros((2, 7, 2)))
    bad = tmp_path / "not_an_array.npz"
    np.savez(bad, areaCell=np.array([3.0, 1.0]))
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art), "--areacell", str(bad)])
    assert "not a" in str(exc.value) and "plain array" in str(exc.value)


def test_rank_local_artifact_is_refused(tmp_path):
    """Under cell partitioning the rows are ONE rank's columns; no weighting
    makes that a global budget.

    No such artifact exists TODAY -- driver setup already refuses the ledger
    whenever the world size exceeds one or a Voronoi layout is present. This
    pins the reader against the day that refusal is lifted, which its own
    message calls "serial-only for now".
    """
    import pytest
    art = _artifact(tmp_path, np.zeros((2, 7, 2)), np.array([3.0, 1.0]),
                    cell_partitioned=True)
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art)])
    assert "ONE RANK" in str(exc.value)


def test_empty_interval_is_refused(tmp_path):
    """n_steps == 0 divides by the max(...,1) floor and yields zeros that read
    as a measured budget."""
    import pytest
    art = _artifact(tmp_path, np.zeros((2, 7, 2)), np.array([3.0, 1.0]),
                    n_steps=0)
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art)])
    assert "n_steps is 0" in str(exc.value)


def test_non_finite_rates_are_refused(tmp_path):
    """One NaN turns a weighted row into NaN; skipping it would reduce over a
    different column set per row."""
    import pytest
    rates = np.zeros((2, 7, 2))
    rates[1, 3, 1] = np.nan
    art = _artifact(tmp_path, rates, np.array([3.0, 1.0]))
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art)])
    assert "non-finite" in str(exc.value)


def test_negative_area_weights_are_refused(tmp_path):
    """A sign-flipped or fill-valued weight would silently invert a row."""
    import pytest
    art = _artifact(tmp_path, np.zeros((2, 7, 2)), np.array([3.0, -1.0]))
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main([str(art)])
    assert "finite and positive" in str(exc.value)
