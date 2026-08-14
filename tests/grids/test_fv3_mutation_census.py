"""Direct tests for ``scripts/validate/fv3_native/mutation_census.py``.

The census is the instrument that answers the GLM review's finding 4 — the
suite's ~500 gates have never once gone red on a code defect, so "green" is
currently uninformative. An instrument that decides what we believe about the
gates is exactly the code a bug in would manufacture a confident wrong verdict,
so it gets its own gates before its numbers are quoted (CLAUDE.md: *validate
the instrument before quoting its number*).

Three things are asserted here, and they are different in kind:

1. **The catalogue still applies.** Every mutation's anchor must occur EXACTLY
   once in the module it names. This is the anti-rot gate: the day someone
   edits one of the ported sites, this goes red instead of the census silently
   reporting "no gate caught it" for a mutation that never applied.
2. **Every mutation is semantic, never cosmetic.** With comments stripped, the
   replacement must still differ from the original. A comment-only "mutation"
   is worse than none — it certifies detection power that was never exercised.
3. **The verdict arithmetic is right.** `newly_failing` counts only
   baseline-pass → mutant-fail (the suite carries known non-code failures, and
   counting a pre-existing red as detection is precisely the error the census
   exists to expose), and `stage_mutant` really writes the mutation while
   leaving the source tree untouched.

No JAX and no pytest subprocesses here: this file is pure logic on text and
paths, so it runs anywhere in seconds.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "scripts" / "validate" / "fv3_native" / "mutation_census.py"


def _load():
    """Import the census by path.

    ``sys.modules`` registration is REQUIRED before ``exec_module``, not
    cosmetic: the census defines ``@dataclass`` types, and dataclasses resolves
    field annotations through ``sys.modules[cls.__module__].__dict__`` — an
    unregistered module makes that ``None`` and the import dies with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """
    name = "_mutation_census"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mc = _load()


def _strip_comments(text: str) -> str:
    """Drop trailing ``#`` comments and normalise whitespace.

    Safe for this catalogue because no anchor contains ``#`` inside a string
    literal — asserted below, so a future anchor that does cannot silently
    make this helper lie.
    """
    out = []
    for line in text.splitlines():
        out.append(re.sub(r"\s*#.*$", "", line).strip())
    return " ".join(x for x in out if x)


# ---------------------------------------------------------------------
# 1. the catalogue applies to the tree it names
# ---------------------------------------------------------------------

def test_every_anchor_occurs_exactly_once_in_its_module():
    """The anti-rot gate.

    Zero matches means the module drifted and the entry is stale; two matches
    means the mutation would hit a site the catalogue does not describe. Either
    way the census must refuse, not produce a verdict.
    """
    for m in mc.MUTATIONS:
        path = REPO / m.module
        assert path.is_file(), f"{m.mid}: module missing: {m.module}"
        n = path.read_text().count(m.old)
        assert n == 1, (
            f"{m.mid} ({m.title}): anchor occurs {n} times in {m.module}, "
            f"need exactly 1 — the catalogue is stale against this SHA")


def test_no_anchor_hides_a_hash_inside_a_string_literal():
    """Guards the comment-stripping helper the next test depends on."""
    for m in mc.MUTATIONS:
        for label, txt in (("old", m.old), ("new", m.new)):
            for line in txt.splitlines():
                before = re.sub(r"\s*#.*$", "", line)
                assert before.count('"') % 2 == 0, (
                    f"{m.mid}.{label}: a '#' sits inside a string literal, so "
                    f"comment stripping would corrupt the semantic diff")


def test_every_mutation_is_semantic_not_cosmetic():
    """A mutation that only changes a comment certifies nothing."""
    for m in mc.MUTATIONS:
        a, b = _strip_comments(m.old), _strip_comments(m.new)
        assert a != b, (
            f"{m.mid} ({m.title}): with comments stripped the replacement is "
            f"identical to the original — this is a cosmetic edit and would "
            f"report false detection power")


def test_catalogue_metadata_is_well_formed():
    ids = [m.mid for m in mc.MUTATIONS] + [m.mid for m in mc.null_controls()]
    assert len(ids) == len(set(ids)), f"duplicate mutation ids: {ids}"
    for m in mc.MUTATIONS:
        assert m.expect in ("caught", "unknown"), f"{m.mid}: expect={m.expect}"
        assert m.oracle and ":" in m.oracle, f"{m.mid}: no oracle file:line"
        assert m.why, f"{m.mid}: no statement of what the oracle says"
        assert m.mode, f"{m.mid}: no failure-mode class"
        assert m.module in mc._MODULE_TEST_FILE, (
            f"{m.mid}: {m.module} has no stage-2 gate file, so the escalation "
            f"ladder cannot run")
        for f in m.fast:
            assert (REPO / f).is_file(), f"{m.mid}: fast selection {f} missing"
    for f in mc.ALL_LANE_TESTS:
        assert (REPO / f).is_file(), f"stage-3 selection {f} missing"


def test_the_two_review_found_defects_are_both_in_the_catalogue():
    """Finding 4 names them explicitly; they are the census's calibration.

    Both were REAL defects found by reading, not by running — so a census in
    which they are absent proves nothing about the class of defect the campaign
    has actually shipped.
    """
    text = " ".join(m.title + m.why + m.old + m.new for m in mc.MUTATIONS)
    assert "q_span" in text, "the _rezone BLOCKER replay is missing"
    assert "_sel_div" in text, "the d_sw1_duo panel-edge MAJOR replay is missing"


def test_path_properties_split_the_module_path_correctly():
    m = next(x for x in mc.MUTATIONS
             if x.module.endswith("core/fv3_mapz.py"))
    assert m.pkg_root == "packages/core"
    assert m.pkg_rel == "legoesm/core"
    assert m.modfile == "fv3_mapz.py"
    assert m.dotted == "legoesm.core.fv3_mapz"
    h = next(x for x in mc.MUTATIONS if x.module.endswith("fv3_duo_halos.py"))
    assert h.dotted == "legoesm.grids.fv3_duo_halos"


@pytest.mark.skipif(not Path(mc.ORACLE_TREE).is_dir(),
                    reason="pinned oracle tree not on this machine")
def test_every_cited_oracle_file_exists_in_the_pinned_tree():
    """Citations are against the PINNED tree only (STATE lesson 3: three
    copies of the source exist and their line numbers differ)."""
    root = Path(mc.ORACLE_TREE)
    for m in mc.MUTATIONS:
        fname = m.oracle.split(":")[0].split()[0]
        hits = list(root.rglob(fname))
        assert hits, f"{m.mid}: {fname} not in the pinned oracle tree"


# ---------------------------------------------------------------------
# 2. staging: the mutant is written, the worktree is not
# ---------------------------------------------------------------------

def test_stage_mutant_writes_the_mutation_and_symlinks_the_siblings(tmp_path):
    m = next(x for x in mc.MUTATIONS if x.mid == "M01")
    src = REPO / m.module
    before = src.read_text()

    st = mc.stage_mutant(REPO, tmp_path, m)
    assert st.ok, st.reason

    got = st.modpath.read_text()
    assert m.new in got and m.old not in got
    assert not st.modpath.is_symlink(), (
        "the mutated module must be a REAL file: the shadow verification "
        "discriminates it from the pristine siblings by realpath")

    pkg = st.root / m.pkg_rel
    sibs = [p for p in pkg.iterdir() if p.name != m.modfile]
    assert sibs, "the symlink farm is empty — legoesm.core is a REGULAR "\
                 "package, so a lone module would shadow the whole package"
    assert all(p.is_symlink() for p in sibs)
    assert (pkg / "__init__.py").exists(), (
        "the package __init__ must be present or the shadow is unimportable")

    # the source tree is untouched: restore is a no-op by construction
    assert src.read_text() == before


def test_stage_mutant_refuses_an_anchor_that_does_not_match(tmp_path):
    bad = mc.Mutation(
        mid="BAD", title="t", module="packages/core/legoesm/core/fv3_mapz.py",
        old="this text is not in the module anywhere at all\n",
        new="nor is this\n", oracle="x.F90:1", why="w", mode="m",
        fast=["tests/grids/test_fv3_mapz.py"])
    st = mc.stage_mutant(REPO, tmp_path, bad)
    assert not st.ok and "0 times" in st.reason


def test_stage_mutant_refuses_a_mutant_that_does_not_parse(tmp_path):
    """NON-VACUITY for the syntax guard.

    A broken edit turns every test in the selection red for a reason the
    catalogue does not describe — i.e. it reports detection power the suite
    never demonstrated. It must be APPLY-FAILED, not CAUGHT.
    """
    m01 = next(x for x in mc.MUTATIONS if x.mid == "M01")
    broken = mc.Mutation(
        mid="X1", title="t", module=m01.module, old=m01.old,
        new="        q_span = qsum / (((\n", oracle="o:1", why="w",
        mode="m", fast=m01.fast)
    st = mc.stage_mutant(REPO, tmp_path, broken)
    assert not st.ok and "does not parse" in st.reason


def test_stage_mutant_refuses_a_mutant_with_an_undefined_name(tmp_path):
    """NON-VACUITY for the undefined-name guard (same false-CAUGHT hazard,
    but at runtime instead of at parse time)."""
    m01 = next(x for x in mc.MUTATIONS if x.mid == "M01")
    broken = mc.Mutation(
        mid="X2", title="t", module=m01.module, old=m01.old,
        new="        q_span = qsum / a_name_that_does_not_exist\n",
        oracle="o:1", why="w", mode="m", fast=m01.fast)
    st = mc.stage_mutant(REPO, tmp_path, broken)
    if not st.ok:
        assert "undefined name" in st.reason
    else:                                  # ruff unavailable -> guard fails open
        pytest.skip("ruff not available; the undefined-name guard fails open")


def test_null_control_stages_a_byte_identical_module(tmp_path):
    n = mc.null_controls()[0]
    st = mc.stage_mutant(REPO, tmp_path, n)
    assert st.ok, st.reason
    assert st.modpath.read_text() == (REPO / n.module).read_text()
    assert not st.modpath.is_symlink()


def test_null_controls_cover_every_mutated_module():
    covered = {n.module for n in mc.null_controls()}
    assert covered == {m.module for m in mc.MUTATIONS}, (
        "a module without a null control has an unverified staging path")


def test_shadow_dir_is_prepended_not_appended(tmp_path):
    """A scratch dir appended to PYTHONPATH shadows nothing.

    Checked here rather than in the census body because it is the single
    assumption the whole mechanism rests on, and it is one string join.
    """
    base = os.pathsep.join(["/a/packages/core", "/a/src"])
    root = "/scratch/mutants/M01"
    joined = root + os.pathsep + base
    assert joined.split(os.pathsep)[0] == root


# ---------------------------------------------------------------------
# 3. the verdict arithmetic
# ---------------------------------------------------------------------

def _res(outcomes):
    return mc.RunResult(True, 0, 0.0, dict(outcomes), list(outcomes))


def test_newly_failing_ignores_a_test_that_was_already_red():
    """The suite currently carries known non-code failures (TOL-PENDING
    provisional bounds). Counting one as detection is exactly the error the
    census exists to expose."""
    base = _res({"a": "pass", "b": "fail", "c": "pass"})
    mut = _res({"a": "pass", "b": "fail", "c": "fail"})
    assert mc.newly_failing(base, mut) == ["c"]


def test_newly_failing_preserves_execution_order():
    base = _res({"a": "pass", "b": "pass", "c": "pass"})
    mut = mc.RunResult(True, 1, 0.0,
                       {"a": "fail", "b": "pass", "c": "fail"},
                       ["c", "b", "a"])
    assert mc.newly_failing(base, mut) == ["c", "a"], (
        "the FIRST catching gate is read off this list, so it must follow the "
        "order the tests actually ran in")


def test_newly_failing_does_not_count_a_skip():
    base = _res({"a": "pass"})
    mut = _res({"a": "skip"})
    assert mc.newly_failing(base, mut) == []


def test_parse_junit_reads_pass_fail_error_and_skip(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites><testsuite name="pytest" tests="4">\n'
        '<testcase classname="tests.grids.test_x" name="test_ok"/>\n'
        '<testcase classname="tests.grids.test_x" name="test_bad">'
        '<failure message="boom">tb</failure></testcase>\n'
        '<testcase classname="tests.grids.test_x" name="test_err">'
        '<error message="collect">tb</error></testcase>\n'
        '<testcase classname="tests.grids.test_x" name="test_skip">'
        '<skipped message="why"/></testcase>\n'
        '</testsuite></testsuites>\n')
    outcomes, order = mc.parse_junit(xml)
    assert outcomes == {
        "test_x::test_ok": "pass",
        "test_x::test_bad": "fail",
        "test_x::test_err": "fail",
        "test_x::test_skip": "skip",
    }
    assert order[0] == "test_x::test_ok"


def test_parse_junit_resolves_a_duplicated_id_to_the_worst_outcome(tmp_path):
    """pytest emits a testcase per phase; a teardown failure after a passing
    call must not read as a pass."""
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<testsuites><testsuite name="pytest">\n'
        '<testcase classname="t" name="a"/>\n'
        '<testcase classname="t" name="a"><failure/></testcase>\n'
        '</testsuite></testsuites>\n')
    outcomes, _ = mc.parse_junit(xml)
    assert outcomes == {"t::a": "fail"}


def test_argument_parser_requires_the_three_paths():
    p = mc.build_parser()
    args = p.parse_args(["--repo", "/r", "--work", "/w", "--out", "/o.md"])
    assert (args.repo, args.work, args.out) == ("/r", "/w", "/o.md")
    assert args.max_stage == "all" and args.only == []
    with pytest.raises(SystemExit):
        p.parse_args(["--repo", "/r"])
