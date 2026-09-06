"""Guards for the round-31 walk into ``dyn_zdf``.

Two things are guarded here, and neither needs a GYRE model run.

First the ALIGNMENT ARITHMETIC.  The walk's whole premise is that legoESM's
pre-implicit exposure lines up with ``dynzdf.F90:121-122`` and not with the
record's ``uu_Kaa_pre``, because legoESM performs NEMO's barotropic removal
(``dynzdf.F90:149-150``) and its barotropic bottom-stress addition
(``dynzdf.F90:156-159``) inside its own solver.  That premise is only usable
if the three statements, evaluated on the record's own operands, reproduce the
record's own pre-solve vector -- so the calibration is exercised on synthetic
operands with a KNOWN answer, and each of the three statements is shown to
matter by deleting it.

Second the COLUMN-UNIFORMITY MEASURE, which is what names an owner.  It has to
say "uniform" for the shift the barotropic correction actually applies and
"not uniform" for a bottom-cell spike, and both are asserted on constructed
fields rather than on the run.

Scoring the model needs the oracle records and a full GYRE step, so it lives
in the gate; the numbers are in the round-31 receipt with their SHA-256.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round31_zdf_walk",
    TESTCASES / "nemo_testcase_l2_gyre_round31_zdf_walk.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

NX, NY, NZ = 5, 4, 6
JPKM1 = NZ - 1


def _rec(*, rng=None, bottom=JPKM1) -> dict:
    """A synthetic zdf-matrix record with every operand the walk consumes."""
    rng = rng or np.random.default_rng(31)
    shape = (NX, NY, NZ)
    arrays = {
        "uu_Kbb_in": rng.normal(size=shape),
        "vv_Kbb_in": rng.normal(size=shape),
        "uu_Krhs_in": rng.normal(size=shape) * 1e-5,
        "vv_Krhs_in": rng.normal(size=shape) * 1e-5,
        "uu_b_Kaa": rng.normal(size=(NX, NY)) * 1e-3,
        "vv_b_Kaa": rng.normal(size=(NX, NY)) * 1e-3,
        "umask": np.ones(shape),
        "vmask": np.ones(shape),
        "e3u_Kaa": 10.0 + rng.random(shape),
        "e3v_Kaa": 10.0 + rng.random(shape),
        "rCdU_bot": -rng.random((NX, NY)) * 1e-4,
        "mbku": np.full((NX, NY), float(bottom)),
        "mbkv": np.full((NX, NY), float(bottom)),
        "rDt": 14400.0,
    }
    return {"arrays": arrays,
            "header": {"jpi": NX, "jpj": NY, "jpk": NZ, "jpkm1": JPKM1,
                       "ntsi": 1, "ntei": NX, "ntsj": 1, "ntej": NY}}


def test_the_explicit_update_is_nemos_own_association():
    """``dynzdf.F90:121-122``: (Kbb + rDt*Krhs)*umask, in that order."""
    rec = _rec()
    a = rec["arrays"]
    want = (a["uu_Kbb_in"] + a["rDt"] * a["uu_Krhs_in"]) * a["umask"]
    assert np.array_equal(gate.nemo_explicit_update(rec, "u"), want)
    # a masked face must be exactly zero, whatever the RHS says
    a["umask"][0, 0, 0] = 0.0
    assert gate.nemo_explicit_update(rec, "u")[0, 0, 0] == 0.0


def test_the_pre_solve_vector_walks_all_three_statements():
    rec = _rec()
    a = rec["arrays"]
    built = gate.nemo_pre_solve_vector(rec, "u")
    expl = gate.nemo_explicit_update(rec, "u")
    # :149-150 -- the barotropic removal, levels 1..jpkm1 only
    interior = (expl[..., :JPKM1] - a["uu_b_Kaa"][..., None]) * a["umask"][..., :JPKM1]
    assert np.array_equal(built[..., JPKM1:], expl[..., JPKM1:]), (
        "the level above jpkm1 must not be touched")
    # :156-159 -- the bottom stress lands on mbku only
    k = int(a["mbku"][0, 0]) - 1
    others = [j for j in range(JPKM1) if j != k]
    assert np.allclose(built[..., others], interior[..., others], atol=0, rtol=0)
    term = (rec["arrays"]["rDt"] * 0.5
            * (np.roll(a["rCdU_bot"], -1, axis=0) + a["rCdU_bot"])
            * a["uu_b_Kaa"] / a["e3u_Kaa"][..., k])
    assert np.allclose(built[..., k], interior[..., k] + term, atol=0, rtol=0)


def test_each_of_the_three_statements_changes_the_answer():
    """Non-vacuity: drop one operand at a time and the vector must move."""
    rec = _rec()
    base = gate.nemo_pre_solve_vector(rec, "u").copy()
    for key in ("uu_Krhs_in", "uu_b_Kaa", "rCdU_bot"):
        moved = _rec()
        moved["arrays"][key] = np.zeros_like(np.asarray(moved["arrays"][key]))
        assert not np.array_equal(gate.nemo_pre_solve_vector(moved, "u"), base), (
            f"zeroing {key} left the pre-solve vector unchanged; the walk is "
            "not reading it")


def test_the_bottom_stress_follows_mbku_not_the_last_level():
    """A shallower column must take its stress at ITS own deepest wet level."""
    shallow = _rec(bottom=2)
    built = gate.nemo_pre_solve_vector(shallow, "u")
    expl = gate.nemo_explicit_update(shallow, "u")
    a = shallow["arrays"]
    interior = (expl[..., :JPKM1] - a["uu_b_Kaa"][..., None]) * a["umask"][..., :JPKM1]
    assert not np.allclose(built[..., 1], interior[..., 1], atol=0, rtol=0)
    assert np.allclose(built[..., JPKM1 - 1], interior[..., JPKM1 - 1],
                       atol=0, rtol=0)


def test_column_uniformity_calls_a_uniform_shift_uniform():
    """The shape ``stprk3_stg.F90:444-445`` produces must read as uniform."""
    mask = np.ones((NX, NY, NZ), dtype=bool)
    shift = np.broadcast_to(np.linspace(1e-7, 5e-7, NX * NY).reshape(NX, NY, 1),
                            (NX, NY, NZ))
    out = gate._column_uniformity(np.array(shift), mask)
    assert out["fraction_column_uniform"] == 1.0
    assert out["median_spread_over_peak"] == 0.0


def test_column_uniformity_calls_a_bottom_spike_structured():
    """And the shape a bottom-cell error produces must NOT."""
    mask = np.ones((NX, NY, NZ), dtype=bool)
    err = np.full((NX, NY, NZ), 7.1e-08)
    err[..., -1] = 9.5e-07
    out = gate._column_uniformity(err, mask)
    assert out["fraction_column_uniform"] == 0.0
    assert out["median_spread_over_peak"] > gate.UNIFORM_TOL


def test_column_uniformity_ignores_single_level_columns():
    mask = np.zeros((NX, NY, NZ), dtype=bool)
    mask[..., 0] = True
    out = gate._column_uniformity(np.ones((NX, NY, NZ)), mask)
    assert out["columns_with_two_or_more_wet_levels"] == 0
    assert out["fraction_column_uniform"] is None


def test_the_verdict_thresholds_are_the_preregistered_ones():
    assert gate.UNIFORM_TOL == 1.0e-2
    src = (TESTCASES / "nemo_testcase_l2_gyre_round31_zdf_walk.py").read_text()
    assert ">= 0.9" in src and "< 0.10" in src, (
        "the preregistered 90 per cent / 10 per cent verdict bands are gone")


@pytest.mark.parametrize("mode", ["calibrate", "pre_solve", "depth_profile"])
def test_every_mode_is_reachable_and_stamps_its_tree(mode):
    assert mode in gate.MODES
    src = (TESTCASES / "nemo_testcase_l2_gyre_round31_zdf_walk.py").read_text()
    assert src.count('"worktree": worktree_stamp()') == len(gate.MODES)
