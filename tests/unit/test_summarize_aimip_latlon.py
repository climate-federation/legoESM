"""Direct test for the AIMIP lat-lon summary-table aggregator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "_summarize_aimip", REPO / "scripts" / "plot" / "summarize_aimip_latlon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(rmse_scale):
    """A scorecard variant block with all metrics = rmse_scale."""
    keys = ["T", "u", "v", "q", "ps", "rsut", "olr", "sfc_net_sw", "sfc_net_lw"]
    return {"eval_metrics": {k: {"rmse": rmse_scale, "bias": 0.0} for k in keys},
            "schemes": {"convection": "sbm", "turbulence": "louis",
                        "gravity_wave_drag": "hines", "microphysics": "kessler",
                        "radiation": "rrtmgp"}}


def test_family_table_picks_lowest_composite(tmp_path):
    mod = _load_mod()
    sc = {"classical": _block(1.0), "column_nn": _block(2.0), "sfno": _block(3.0)}
    p = tmp_path / "aimip_latlon_scorecard.json"
    p.write_text(json.dumps(sc))
    md = mod.summarize_run(p)
    assert "Best family: `classical`" in md
    assert "`classical`" in md and "★" in md
    # column order present
    for k in ("rsut", "olr", "sfc_net_sw", "sfc_net_lw", "composite"):
        assert k in md


def test_failed_variant_does_not_crash(tmp_path):
    mod = _load_mod()
    sc = {"classical": _block(1.0), "sfno": {"variant": "sfno", "error": "boom"}}
    p = tmp_path / "aimip_latlon_scorecard.json"
    p.write_text(json.dumps(sc))
    md = mod.summarize_run(p)
    assert "`sfno` (FAILED)" in md
    assert "Best family: `classical`" in md


def test_sweep_picks_best_combo(tmp_path):
    mod = _load_mod()
    for name, scale in [("combo_baseline", 2.0), ("combo_conv_kuo", 1.0),
                        ("combo_gwd_none", 3.0)]:
        d = tmp_path / name
        d.mkdir()
        (d / "aimip_latlon_scorecard.json").write_text(
            json.dumps({"classical": _block(scale)}))
    md = mod.summarize_sweep(tmp_path)
    assert "Best combination: `combo_conv_kuo`" in md
    assert "convection=`sbm`" in md  # schemes echoed


def test_sweep_empty_root(tmp_path):
    mod = _load_mod()
    md = mod.summarize_sweep(tmp_path)
    assert "No combo scorecards found" in md


def test_all_failed_is_not_rankable(tmp_path):
    mod = _load_mod()
    sc = {"classical": {"variant": "classical", "error": "x"},
          "sfno": {"variant": "sfno", "error": "y"}}
    p = tmp_path / "aimip_latlon_scorecard.json"
    p.write_text(json.dumps(sc))
    md = mod.summarize_run(p)
    assert "No rankable family" in md
    assert "**★**" not in md  # no row marked best (legend has a bare ★)


def test_nan_metric_makes_entry_unrankable(tmp_path):
    mod = _load_mod()
    import math
    nan_block = _block(1.0)
    nan_block["eval_metrics"]["T"]["rmse"] = float("nan")   # missing a state key
    nan_block["eval_metrics"]["olr"]["rmse"] = float("nan")  # missing a flux key
    sc = {"good": _block(5.0), "nanmodel": nan_block}
    scores = mod._composite_scores(sc)
    # NaN -> metric absent -> group incomplete -> composite inf (never NaN),
    # and the finite-scored complete entry is best.
    assert all(not math.isnan(s) for s in scores.values())
    assert math.isinf(scores["nanmodel"])
    assert mod._best(scores) == "good"


def test_partial_entry_cannot_beat_complete(tmp_path):
    """Codex round-2: a partial entry (few below-median metrics) must not
    out-rank a complete one via median-normalization."""
    mod = _load_mod()
    import math
    complete = _block(10.0)
    partial = {"eval_metrics": {"T": {"rmse": 1.0, "bias": 0.0},
                                "rsut": {"rmse": 1.0, "bias": 0.0}}}
    scores = mod._composite_scores({"complete": complete, "partial": partial})
    assert math.isinf(scores["partial"])      # missing most metrics
    assert math.isfinite(scores["complete"])
    assert mod._best(scores) == "complete"


def test_missing_flux_group_not_rankable(tmp_path):
    mod = _load_mod()
    no_flux = _block(1.0)
    for k in ("rsut", "olr", "sfc_net_sw", "sfc_net_lw"):
        del no_flux["eval_metrics"][k]
    sc = {"stateonly": no_flux}
    p = tmp_path / "aimip_latlon_scorecard.json"
    p.write_text(json.dumps(sc))
    scores = mod._composite_scores({"stateonly": no_flux})
    import math
    assert math.isinf(scores["stateonly"])  # both groups required
    assert "No rankable family" in mod.summarize_run(p)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
