"""Regression tests for ``scripts/matrix/run_atmosphere_test_matrix.py`` family filter.

The filter previously combined an alias-to-equation-set table with a
single ``family`` field on ``TestCase``, which meant ``--family
dcmip2008`` matched any case tagged ``hydro`` (including the canonical
J-W and DCMIP 2012 rest-state-topo).  The replacement uses an explicit
per-case multi-tag table; these tests guard against regressing back to
the alias behaviour.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATRIX_PATH = _REPO_ROOT / "scripts" / "matrix" / "run_atmosphere_test_matrix.py"


@pytest.fixture(scope="module")
def matrix_module():
    """Load ``scripts/run_atmosphere_test_matrix`` as an importable module.

    The script lives outside any package; load by file path instead of
    relying on ``sys.path`` mutation.
    """
    spec = importlib.util.spec_from_file_location(
        "_atmosphere_matrix_for_tests", _MATRIX_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_atmosphere_matrix_for_tests"] = mod
    spec.loader.exec_module(mod)
    return mod


def _names_for(matrix_module, *, family, grid="all"):
    """Return the set of case names that pass the filter."""
    args = SimpleNamespace(
        only="all", family=family, grid=grid, test=None,
    )
    return {tc.case for tc in matrix_module.filter_tests(
        matrix_module.TEST_MATRIX, args)}


# ---------------------------------------------------------------------------
# Vintage filters must NOT bleed equation-set tags
# ---------------------------------------------------------------------------


def test_dcmip2008_excludes_canonical_jw(matrix_module):
    """``--family dcmip2008`` must not include the canonical J-W test
    (``baroclinic``) — that one is *not* a DCMIP 2008 test."""
    cases = _names_for(matrix_module, family="dcmip2008")
    assert "baroclinic" not in cases
    assert "rest_state_topo" not in cases  # DCMIP 2012, not 2008
    # M1.a + M1.b DCMIP 2008 cases.
    assert cases == {
        "rotated_baroclinic", "rotated_steady",
        "gravity_wave_3_1", "inertio_gravity_3_2",
        "mountain_rossby_5_0", "rossby_haurwitz_6_0",
    }


def test_dcmip2008_excludes_rest_state_topo(matrix_module):
    """rest_state_topo is DCMIP 2012, not DCMIP 2008."""
    cases = _names_for(matrix_module, family="dcmip2008")
    assert "rest_state_topo" not in cases


def test_dcmip2012_includes_only_2012_cases(matrix_module):
    cases = _names_for(matrix_module, family="dcmip2012")
    assert cases == {
        "rest_state_topo",
        "dcmip_transport_11",
        "dcmip_transport_12",
        "dcmip_transport_13",
    }


def test_dcmip2016_currently_empty(matrix_module):
    """No DCMIP 2016 cases land until M3."""
    cases = _names_for(matrix_module, family="dcmip2016")
    assert cases == set()


# ---------------------------------------------------------------------------
# Equation-set filters
# ---------------------------------------------------------------------------


def test_sw_includes_all_williamson(matrix_module):
    cases = _names_for(matrix_module, family="sw")
    # All four wired Williamson tests must appear at least once.
    for w in ("williamson2", "williamson5", "williamson6", "cosine_bell"):
        assert w in cases


def test_climate_excludes_amip_only_filter(matrix_module):
    """AMIP is not a Hughes idealized test — it must show under
    ``climate`` but not under ``hughes``."""
    climate = _names_for(matrix_module, family="climate")
    hughes = _names_for(matrix_module, family="hughes")
    assert "amip" in climate
    assert "amip" not in hughes


def test_hughes_includes_full_canonical_set(matrix_module):
    cases = _names_for(matrix_module, family="hughes")
    expected = {
        "williamson2", "williamson5", "williamson6", "cosine_bell",
        "baroclinic", "rotated_baroclinic", "rotated_steady",
        "rest_state_topo", "held_suarez", "held_suarez_topo",
        "gravity_wave_3_1", "inertio_gravity_3_2",
        "mountain_rossby_5_0", "rossby_haurwitz_6_0",
        "dcmip_transport_11", "dcmip_transport_12", "dcmip_transport_13",
        "dcmip_tc1", "dcmip_tc2", "dcmip_tc3",
    }
    assert cases == expected


def test_all_returns_full_matrix(matrix_module):
    args = SimpleNamespace(only="all", family="all", grid="all", test=None)
    filtered = matrix_module.filter_tests(matrix_module.TEST_MATRIX, args)
    assert len(filtered) == len(matrix_module.TEST_MATRIX)


# ---------------------------------------------------------------------------
# Multi-tag membership
# ---------------------------------------------------------------------------


def test_rotated_baroclinic_has_both_hydro_and_dcmip2008(matrix_module):
    """Membership must be multi-valued: rotated_baroclinic is both a
    hydrostatic case and a DCMIP-2008 case."""
    fams = matrix_module._CASE_FAMILIES["rotated_baroclinic"]
    assert "hydro" in fams
    assert "dcmip2008" in fams
    assert "hughes" in fams


def test_dcmip_transport_has_both_tracer_and_dcmip2012(matrix_module):
    fams = matrix_module._CASE_FAMILIES["dcmip_transport_11"]
    assert "tracer" in fams
    assert "dcmip2012" in fams
    assert "hughes" in fams


def test_amip_not_hughes(matrix_module):
    fams = matrix_module._CASE_FAMILIES["amip"]
    assert "hughes" not in fams


# ---------------------------------------------------------------------------
# CLI must accept every documented family (including reserved ones)
# ---------------------------------------------------------------------------


_DOCUMENTED_FAMILIES = (
    "sw", "hydro", "nh", "moist", "climate", "tracer",
    "dcmip2008", "dcmip2012", "dcmip2016",
    "hughes", "all",
)


@pytest.mark.parametrize("family", _DOCUMENTED_FAMILIES)
def test_cli_accepts_documented_family(matrix_module, family):
    """argparse must accept every family advertised in the catalog —
    including reserved values whose case-set is empty today."""
    parser = matrix_module.build_parser()
    # parse_args raises SystemExit on a rejected --family choice
    args = parser.parse_args(["--list", "--family", family])
    assert args.family == family


def test_dcmip2016_is_a_reserved_choice(matrix_module):
    """dcmip2016 has no cases in M1.a but must remain a valid CLI
    choice (catalog documents it as the M3 family)."""
    assert "dcmip2016" in matrix_module._FAMILY_CHOICES
    assert "dcmip2016" in matrix_module._RESERVED_FAMILIES


def test_iter118_save_timeseries_csv_skips_blowup_info(matrix_module, tmp_path):
    """new_test_dycores iter-121 functional test for iter-118 fix.

    `_save_timeseries_csv` must skip underscore-prefixed metadata keys
    (e.g., `_blowup_info` from `_run_timeloop` on FAIL) which are
    dicts not lists.  Pre-iter-118, a FAIL would crash the writer
    silently with `KeyError: 0` after writing the header — leaving an
    empty header-only csv that blocked post-mortem investigation of
    the iter-102 TC2 cube full-mode BLOWUP.

    Parallel to the existing ocean-side test
    `test_save_timeseries_csv_skips_blowup_info_metadata` in
    `test_ocean_cross_grid_plots.py` (ocean-side fix landed earlier
    as ocean-iter-125).
    """
    diag = {
        "steps": [0, 100, 200],
        "times": [0.0, 0.5, 1.0],
        "max_abs_w": [0.0, 0.1, 0.2],
        "_blowup_info": {  # iter-105 metadata; dict, not list.
            "step": 200, "day": 1.0, "metric": 1234.5,
            "is_finite": False, "threshold": 1000.0,
            "reason": "test",
        },
    }
    # Pre-iter-118 this would crash silently with KeyError: 0 after
    # writing the header row.
    matrix_module._save_timeseries_csv(tmp_path, diag, dt=300.0)
    csv_text = (tmp_path / "mean_timeseries.csv").read_text()
    assert "_blowup_info" not in csv_text, (
        "iter-118 regression: ``_blowup_info`` metadata column "
        "appearing in mean_timeseries.csv — the private-key filter "
        "was dropped.  See iter-118 fix in matrix runner."
    )
    assert "max_abs_w" in csv_text, (
        "iter-118 regression: legitimate timeseries column missing "
        "from csv.  Writer is not iterating the keys correctly."
    )
    # Verify data rows are present (not just header).
    lines = csv_text.strip().splitlines()
    assert len(lines) == 4, (  # header + 3 data rows
        f"iter-118 regression: expected 4 lines in csv (header + "
        f"3 data rows from diag); got {len(lines)}.  This was the "
        "pre-iter-118 failure mode — empty header-only csv on FAIL."
    )
