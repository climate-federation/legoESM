"""The full-matrix sbatch's case list must match the registry.

scripts/cluster/ocean_full_matrix.sbatch hardcodes the 27 case names as an
array, so `--array=0-26` maps to a reproducible set of cases across runs.
A literal list drifts silently: a case added to _build_test_matrix() would
simply never be swept, and one removed would send an array task to an
undefined index. This test is what makes the literal safe.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SBATCH = _REPO / "scripts" / "cluster" / "ocean_full_matrix.sbatch"


def _registry_case_names():
    """Case names in _build_test_matrix() order, de-duplicated."""
    if "_rm_full" not in sys.modules:
        p = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"
        sys.path.insert(0, str(p.parent))
        spec = importlib.util.spec_from_file_location("_rm_full", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_rm_full"] = mod
        spec.loader.exec_module(mod)
    rm = sys.modules["_rm_full"]
    seen = []
    for tc in rm._build_test_matrix():
        if tc.case not in seen:
            seen.append(tc.case)
    return seen


def _sbatch_case_names():
    """The CASES=( ... ) literal, with line continuations folded."""
    text = _SBATCH.read_text()
    m = re.search(r"^CASES=\((.*?)\)$", text, re.S | re.M)
    assert m, "CASES=( ... ) not found in the sbatch"
    return m.group(1).replace("\\\n", " ").split()


def test_the_sbatch_list_is_exactly_the_registry():
    """Same names, same ORDER -- the array index is the contract."""
    assert _sbatch_case_names() == _registry_case_names()


def test_the_array_range_in_the_docstring_matches_the_list():
    """`--array=0-26` and 27 names have to agree, or the last case is
    skipped or a task indexes past the end."""
    names = _sbatch_case_names()
    text = _SBATCH.read_text()
    ranges = set(re.findall(r"--array=0-(\d+)", text))
    assert ranges, "no --array=0-N example in the sbatch"
    assert ranges == {str(len(names) - 1)}, (
        f"sbatch documents --array=0-{sorted(ranges)} for {len(names)} cases")


def test_prefix_collisions_exist_so_the_exact_selector_is_required():
    """NON-VACUITY for the '=' selector the sbatch uses.

    If no case name were a prefix of another, a plain substring --only
    would be harmless and the exactness comment would be cargo. Pin that
    the collisions are real, so the reason survives.
    """
    names = _registry_case_names()
    collisions = {a for a in names for b in names
                  if a != b and b.startswith(a)}
    assert collisions, "no prefix collisions: the '=' selector needs a reason"
