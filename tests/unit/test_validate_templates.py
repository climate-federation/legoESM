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


def _write_tmpl(vt, tmp_path, body: str):
    """Write a template under a tmp templates root and point the module at it."""
    root = tmp_path / "config" / "templates"
    (root / "x").mkdir(parents=True, exist_ok=True)
    f = root / "x" / "t.yaml"
    f.write_text(body)
    vt._DEFAULT_TEMPLATES = root
    return f


def test_complexity_must_match_resolved_model_type(vt, tmp_path):
    # codex HIGH-1: declares hydrostatic but leaves dynamics at the shallow_water
    # default -> must be reported FAIL (silent wrong-model otherwise).
    saved = vt._DEFAULT_TEMPLATES
    try:
        f = _write_tmpl(vt, tmp_path, (
            "experiment:\n  tier: tier2\n  complexity: hydrostatic\n  extent: global\n"
            "  maturity: run_tested\n  description: mismatch\n  data: []\n"
            "model:\n  type: atmosphere_only\n"
            "grid:\n  type: cubed_sphere\n  resolution: 36\n  n_levels: 26\n"
            "  vertical_coord: hybrid\n"
            "atmosphere:\n  dynamics: shallow_water\n  dt_seconds: 600\n"
            "time:\n  duration_hours: 24\n"
        ))
        r = vt.validate_template(f)
        assert r.ok is False and "model_type" in r.error
    finally:
        vt._DEFAULT_TEMPLATES = saved


def test_unknown_data_id_rejected(vt, tmp_path):
    # codex MEDIUM-2: experiment.data id absent from data_catalog.yaml -> FAIL.
    saved = vt._DEFAULT_TEMPLATES
    try:
        f = _write_tmpl(vt, tmp_path, (
            "experiment:\n  tier: tier3\n  complexity: shallow_water\n  extent: global\n"
            "  maturity: init_only\n  description: bad data id\n"
            "  data: [no_such_dataset_xyz]\n"
            "model:\n  type: atmosphere_only\n"
            "grid:\n  type: cubed_sphere\n  resolution: 48\n  n_levels: 1\n"
            "  vertical_coord: none\n"
            "atmosphere:\n  dynamics: shallow_water\n  dt_seconds: 600\n"
            "time:\n  duration_hours: 24\n"
        ))
        r = vt.validate_template(f)
        assert r.ok is False and "data_catalog" in r.error
    finally:
        vt._DEFAULT_TEMPLATES = saved


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
