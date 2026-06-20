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
    assert "sfno (FAILED)" in md
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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
