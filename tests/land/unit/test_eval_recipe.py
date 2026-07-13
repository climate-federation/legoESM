"""Unit tests for :mod:`legoesm.land.evaluation.recipe` and references."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm.land.evaluation import fluxio, references
from legoesm.land.evaluation.recipe import (
    parse_recipe,
    run_case,
    _resolve_variable,
)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def test_reference_registry_guard():
    assert references.get_reference_spec("fortran_v2").fmt == "clm_ml_out"
    with pytest.raises(ValueError, match="Unknown reference"):
        references.get_reference_spec("no_such_ref")


def test_resolve_reference_explicit_path(tmp_path):
    d = tmp_path / "myref"
    d.mkdir()
    got = references.resolve_reference_dir(
        "fortran_v2", tmp_path, explicit_path=str(d)
    )
    assert got == d


def test_resolve_reference_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(references.DATA_ROOT_ENV, raising=False)
    # zenodo_obs has no in-tree default -> must fail loudly.
    with pytest.raises(FileNotFoundError, match="Could not locate reference"):
        references.resolve_reference_dir("zenodo_obs", tmp_path)


def test_resolve_reference_bundle_env(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "clm_ml_v2_fortran").mkdir(parents=True)
    monkeypatch.setenv(references.DATA_ROOT_ENV, str(bundle))
    got = references.resolve_reference_dir("fortran_v2", tmp_path)
    assert got == bundle / "clm_ml_v2_fortran"


# ---------------------------------------------------------------------------
# Recipe parsing / validation
# ---------------------------------------------------------------------------


def _minimal_recipe():
    return {
        "name": "r",
        "cases": [{
            "name": "c",
            "model": {"label": "m", "path": "results/x", "tag": "TAG"},
            "references": ["fortran_v2"],
            "variables": ["flux:shflx"],
            "metrics": ["rmse", "bias_score"],
        }],
    }


def test_parse_recipe_ok():
    rec = parse_recipe(_minimal_recipe())
    assert rec.name == "r"
    assert rec.cases[0].model.tag == "TAG"


def test_parse_recipe_unknown_metric():
    data = _minimal_recipe()
    data["cases"][0]["metrics"] = ["rmse", "bogus_metric"]
    with pytest.raises(ValueError, match="unknown metric"):
        parse_recipe(data)


def test_parse_recipe_unknown_reference():
    data = _minimal_recipe()
    data["cases"][0]["references"] = ["not_a_ref"]
    with pytest.raises(ValueError, match="Unknown reference"):
        parse_recipe(data)


def test_parse_recipe_missing_model_fields():
    data = _minimal_recipe()
    data["cases"][0]["model"] = {"label": "m"}  # no path/tag
    with pytest.raises(ValueError, match="needs 'path' and 'tag'"):
        parse_recipe(data)


def test_resolve_variable_forms():
    assert _resolve_variable("flux:shflx") == ("flux", "shflx")
    # 'shflx' is unique to flux -> resolvable bare.
    assert _resolve_variable("shflx") == ("flux", "shflx")


def test_resolve_variable_ambiguous_raises():
    # 'gpp' lives in both flux and fsun schemas.
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_variable("gpp")


def test_resolve_variable_unknown_raises():
    with pytest.raises(ValueError, match="unknown key"):
        _resolve_variable("not_a_var")


# ---------------------------------------------------------------------------
# End-to-end run_case on synthetic .out directories
# ---------------------------------------------------------------------------


def _write_out(dir_path: Path, tag: str, schema_tag: str, cols: dict):
    """Write a <tag>_<schema>.out with the given named columns."""
    specs = fluxio.SCHEMAS[schema_tag]
    n = len(next(iter(cols.values())))
    keys = [s[0] for s in specs]
    lines = []
    for i in range(n):
        row = [cols.get(k, np.zeros(n))[i] for k in keys]
        lines.append(" ".join(f"{float(x):.6f}" for x in row))
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{tag}_{schema_tag}.out").write_text("\n".join(lines) + "\n")


def test_run_case_scores_perfect_and_offset(tmp_path):
    repo = tmp_path
    model_dir = repo / "results" / "model"
    ref_dir = repo / "ref"
    n = 100
    rng = np.random.default_rng(1)
    shflx = rng.normal(50, 20, n)
    lhflx = rng.normal(80, 30, n)

    # Model == reference for shflx (perfect); lhflx has a +10 bias.
    _write_out(model_dir, "TAG", "flux", {"shflx": shflx, "lhflx": lhflx + 10})
    _write_out(ref_dir, "TAG", "flux", {"shflx": shflx, "lhflx": lhflx})

    data = {
        "name": "r",
        "cases": [{
            "name": "c",
            "model": {"label": "model", "path": "results/model", "tag": "TAG"},
            "references": [{"id": "fortran_v2", "path": str(ref_dir),
                            "label": "REF"}],
            "variables": ["flux:shflx", "flux:lhflx"],
            "metrics": ["bias", "rmse", "bias_score", "rmse_score",
                        "taylor_score"],
        }],
    }
    rec = parse_recipe(data)
    cards = run_case(rec.cases[0], repo)
    assert len(cards) == 1
    card = cards[0]
    by_key = {v.key: v for v in card.variables}

    # shflx perfect.
    assert by_key["flux:shflx"].score == pytest.approx(1.0, abs=1e-9)
    assert by_key["flux:shflx"].metrics["bias"] == pytest.approx(0.0, abs=1e-9)
    # lhflx: +10 bias, but shape identical -> rmse_score high, bias_score < 1.
    assert by_key["flux:lhflx"].metrics["bias"] == pytest.approx(10.0)
    assert by_key["flux:lhflx"].metrics["bias_score"] < 1.0
    assert by_key["flux:lhflx"].metrics["rmse_score"] == pytest.approx(
        1.0, abs=1e-6
    )
    assert 0.0 < card.overall_score() <= 1.0


def test_run_case_obs_reference_raises(tmp_path):
    repo = tmp_path
    _write_out(repo / "results" / "m", "TAG", "flux",
               {"shflx": np.ones(5)})
    data = {
        "name": "r",
        "cases": [{
            "name": "c",
            "model": {"label": "m", "path": "results/m", "tag": "TAG"},
            "references": ["fluxnet_obs"],
            "variables": ["flux:shflx"],
            "metrics": ["rmse"],
        }],
    }
    rec = parse_recipe(data)
    with pytest.raises(ValueError, match="row-aligned recipe scorer"):
        run_case(rec.cases[0], repo)


def test_run_case_missing_schema_file_records_n0(tmp_path):
    repo = tmp_path
    model_dir = repo / "results" / "m"
    ref_dir = repo / "ref"
    # Only flux.out written; asking for an fsun: variable must record n=0
    # (no fsun.out) instead of crashing or fabricating a score.
    _write_out(model_dir, "TAG", "flux", {"shflx": np.ones(10)})
    _write_out(ref_dir, "TAG", "flux", {"shflx": np.ones(10)})
    data = {
        "name": "r",
        "cases": [{
            "name": "c",
            "model": {"label": "m", "path": "results/m", "tag": "TAG"},
            "references": [{"id": "fortran_v2", "path": str(ref_dir)}],
            "variables": ["flux:shflx", "fsun:tl_sun"],
            "metrics": ["rmse", "bias_score"],
        }],
    }
    rec = parse_recipe(data)
    cards = run_case(rec.cases[0], repo)
    by_key = {v.key: v for v in cards[0].variables}
    assert by_key["flux:shflx"].n == 10
    assert by_key["fsun:tl_sun"].n == 0
    assert np.isnan(by_key["fsun:tl_sun"].score)
