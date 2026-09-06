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


def _masked(shape) -> np.ndarray:
    m = np.ones(shape)
    m[..., JPKM1:] = 0.0
    return m


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
        # NEMO's own records have the face mask zero above jpkm1, because the
        # explicit update never runs there; the fixture must too or it is not
        # a record this walk could ever be handed.
        "umask": _masked(shape),
        "vmask": _masked(shape),
        "e3u_Kaa": 10.0 + rng.random(shape),
        "e3v_Kaa": 10.0 + rng.random(shape),
        "rCdU_bot": -rng.random((NX, NY)) * 1e-4,
        "mbku": np.full((NX, NY), float(bottom)),
        "mbkv": np.full((NX, NY), float(bottom)),
        "rDt": 14400.0,
    }
    return {"arrays": arrays,
            # The writer's tile must end BEFORE the array does, as NEMO's
            # does (ntei 34 of jpi 36), or the +1 neighbour shift would wrap.
            "header": {"jpi": NX, "jpj": NY, "jpk": NZ, "jpkm1": JPKM1,
                       "ntsi": 1, "ntei": NX - 1, "ntsj": 1, "ntej": NY - 1}}


def test_the_explicit_update_is_nemos_own_association():
    """``dynzdf.F90:121-122``: (Kbb + rDt*Krhs)*umask, in that order."""
    rec = _rec()
    a = rec["arrays"]
    want = (a["uu_Kbb_in"] + a["rDt"] * a["uu_Krhs_in"]) * a["umask"]
    assert np.array_equal(gate.nemo_explicit_update(rec, "u"), want)
    # a masked face must be exactly zero, whatever the RHS says
    a["umask"][0, 0, 0] = 0.0
    assert gate.nemo_explicit_update(rec, "u")[0, 0, 0] == 0.0


def test_the_walk_refuses_a_record_whose_tile_reaches_the_edge():
    """np.roll wraps where NEMO reads a real neighbour."""
    rec = _rec()
    rec["header"]["ntei"] = rec["header"]["jpi"]
    with pytest.raises(Exception, match="wraps instead of reading"):
        gate.nemo_pre_solve_vector(rec, "u")


def test_the_walk_refuses_a_mask_that_is_wet_above_jpkm1():
    """NEMO's explicit update never runs there, so a value would be invented."""
    rec = _rec()
    rec["arrays"]["umask"][..., JPKM1:] = 1.0
    with pytest.raises(Exception, match="above jpkm1"):
        gate.nemo_explicit_update(rec, "u")


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


def test_the_verdict_ladder_is_the_preregistered_one():
    """Exercised, not grepped: the ladder is a function and this calls it."""
    assert gate.UNIFORM_TOL == 1.0e-2
    assert gate.verdict_for([0.95, 0.99]) == "BAROTROPIC-CORRECTION CANDIDATE"
    assert gate.verdict_for([0.0, 0.05]) == "IMPLICIT-SOLVE CANDIDATE"
    assert gate.verdict_for([0.5, 0.5]) == "NO OWNER NAMED"
    # one face over the band and one under names NOTHING, in both orders
    assert gate.verdict_for([0.95, 0.05]) == "NO OWNER NAMED"
    assert gate.verdict_for([0.05, 0.95]) == "NO OWNER NAMED"
    # the boundaries themselves
    assert gate.verdict_for([0.9, 0.9]) == "BAROTROPIC-CORRECTION CANDIDATE"
    assert gate.verdict_for([0.1, 0.1]) == "NO OWNER NAMED"
    # an unscorable face can never produce an owner
    assert gate.verdict_for([None, 0.0]) == "NO OWNER NAMED"
    assert gate.verdict_for([]) == "NO OWNER NAMED"


def test_an_exactly_correct_column_votes_for_nothing():
    """A zero-error column has span 0 and peak 0; it must not read uniform.

    Counting it uniform makes every already-correct column vote for the
    barotropic owner, which on a nearly exact field is every column.
    """
    mask = np.ones((NX, NY, NZ), dtype=bool)
    out = gate._column_uniformity(np.zeros((NX, NY, NZ)), mask)
    assert out["columns_exact"] == NX * NY
    assert out["columns_scored"] == 0
    assert out["fraction_column_uniform"] is None
    assert gate.verdict_for([out["fraction_column_uniform"]]) == "NO OWNER NAMED"
    # and a field that is exact everywhere EXCEPT one structured column must
    # be judged on that column alone, not diluted by the exact ones
    err = np.zeros((NX, NY, NZ))
    err[0, 0, -1] = 9.5e-07
    err[0, 0, :-1] = -7.1e-08
    out = gate._column_uniformity(err, mask)
    assert out["columns_scored"] == 1
    assert out["fraction_column_uniform"] == 0.0


def test_a_non_finite_error_is_fatal_not_uniform():
    """A NaN column used to read as uniform while nanmax hid it."""
    mask = np.ones((NX, NY, NZ), dtype=bool)
    err = np.zeros((NX, NY, NZ))
    err[1, 1, :] = np.nan
    with pytest.raises(Exception, match="non-finite"):
        gate._column_uniformity(err, mask)


@pytest.mark.parametrize("mode", ["calibrate", "pre_solve", "depth_profile",
                                  "ordering_size", "ordering_regression"])
def test_every_mode_is_reachable_and_stamps_its_tree(mode):
    assert mode in gate.MODES
    src = (TESTCASES / "nemo_testcase_l2_gyre_round31_zdf_walk.py").read_text()
    assert src.count('"worktree": worktree_stamp()') == len(gate.MODES)


def test_the_ordering_constant_is_mean_A_minus_uu_b():
    """P4g's operand, on operands whose answer is known by construction."""
    rec = _rec()
    a = rec["arrays"]
    a["uu_Kbb_in"][:] = 1.0e-3        # a UNIFORM column, so its weighted
    a["uu_Krhs_in"][:] = 0.0          # depth mean is 1e-3 for any thickness
    a["uu_b_Kaa"][:] = 4.0e-4
    interior = np.ones((NX, NY), dtype=bool)
    delta, _drag, wet = gate.ordering_constant(rec, "u", interior)
    assert wet.all()
    assert np.allclose(delta[wet], 6.0e-4, atol=1e-18, rtol=0)
    # and it is the DIFFERENCE, not either side: move uu_b, delta must follow
    a["uu_b_Kaa"][:] = 9.0e-4
    delta2, _d2, _w2 = gate.ordering_constant(rec, "u", interior)
    assert np.allclose(delta2[wet], 1.0e-4, atol=1e-18, rtol=0)


def test_the_ordering_constant_is_thickness_weighted():
    """A non-uniform column must weight by e3, not by level count."""
    rec = _rec()
    a = rec["arrays"]
    a["uu_Krhs_in"][:] = 0.0
    a["uu_b_Kaa"][:] = 0.0
    a["uu_Kbb_in"][:] = 0.0
    a["uu_Kbb_in"][..., 0] = 1.0      # only the top cell moves
    a["e3u_Kaa"][:] = 1.0
    a["e3u_Kaa"][..., 0] = 99.0       # and it is 99 of the column's 104
    interior = np.ones((NX, NY), dtype=bool)
    delta, _drag, wet = gate.ordering_constant(rec, "u", interior)
    # levels 1..jpkm1 = 5 of them: one of thickness 99 and four of thickness 1
    assert np.allclose(delta[wet], 99.0 / 103.0, atol=1e-15, rtol=0)


def test_the_bottom_drag_factor_is_nemos_own_diagonal_term():
    """``dynzdf.F90:296``: zDt_2*(rCdU_bot(i+1,j)+rCdU_bot(i,j))/e3u(iku,Kaa)."""
    rec = _rec()
    a = rec["arrays"]
    a["rCdU_bot"][:] = -1.0e-4
    a["e3u_Kaa"][:] = 300.0
    interior = np.ones((NX, NY), dtype=bool)
    _delta, drag, wet = gate.ordering_constant(rec, "u", interior)
    want = 14400.0 * 0.5 * abs(-1.0e-4 + -1.0e-4) / 300.0
    assert np.allclose(drag[wet], want, atol=0, rtol=1e-15)
    # non-vacuity: it is a SUM of two neighbours, not one of them doubled
    a["rCdU_bot"][:] = 0.0
    a["rCdU_bot"][2, 2] = -1.0e-4
    _d, drag2, _w = gate.ordering_constant(rec, "u", interior)
    assert drag2[2, 2] > 0 and drag2[1, 2] > 0 and drag2[3, 2] == 0.0
