"""Unit tests for ``scripts/experiment/validate_templates.py``.

Covers the metadata validator and the end-to-end template sweep: every shipped
template under ``config/templates/`` must load + resolve + pass
``validate_strict`` and carry a well-formed ``experiment:`` block, and a
deliberately malformed template must be reported as FAIL (not crash the sweep).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP_DIR = _REPO_ROOT / "scripts" / "experiment"


@pytest.fixture(scope="module")
def vt():
    if str(_EXP_DIR) not in sys.path:
        sys.path.insert(0, str(_EXP_DIR))
    return importlib.import_module("validate_templates")


def test_meta_errors_accepts_well_formed(vt):
    meta = {"tier": "tier1", "complexity": "shallow_water", "extent": "global",
            "maturity": "run_tested", "description": "x", "data": []}
    assert vt._meta_errors(meta) == []


def test_meta_errors_flags_bad_tier_and_maturity(vt):
    errs = vt._meta_errors({"tier": "tier9", "complexity": "sw",
                            "maturity": "someday", "description": "x", "data": []})
    assert any("tier" in e for e in errs)
    assert any("maturity" in e for e in errs)


def test_meta_errors_flags_missing_fields_and_bad_data(vt):
    errs = vt._meta_errors({"tier": "tier1", "maturity": "run_tested",
                            "data": "not-a-list"})
    assert any("complexity" in e for e in errs)
    assert any("description" in e for e in errs)
    assert any("data" in e for e in errs)


def test_all_shipped_templates_validate(vt):
    paths = vt.collect_templates(_REPO_ROOT / "config" / "templates")
    assert paths, "no templates found to validate"
    failures = [(vt.validate_template(p)) for p in paths]
    bad = [r for r in failures if not r.ok]
    assert not bad, "templates failed validation: " + "; ".join(
        f"{r.rel_path}: {r.error}" for r in bad
    )


def test_malformed_template_reported_not_crash(vt, tmp_path):
    # a template missing the experiment block + with a nonsense grid must be
    # reported FAIL, not raise.
    bad = tmp_path / "config" / "templates" / "bad" / "broken.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_text("model:\n  type: atmosphere_only\ngrid:\n  type: not_a_grid\n")
    # point the module's templates-parent at tmp so rel_path resolves
    monkey = vt._DEFAULT_TEMPLATES
    try:
        vt._DEFAULT_TEMPLATES = tmp_path / "config" / "templates"
        r = vt.validate_template(bad)
    finally:
        vt._DEFAULT_TEMPLATES = monkey
    assert r.ok is False and r.error
