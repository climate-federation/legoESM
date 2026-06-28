"""Direct test for the AIMIP sweep scorecard aggregator (ranking logic).

json-only (matplotlib import is deferred inside _heatmap), so it imports
cheaply. Builds a synthetic sweep dir of combo scorecards + a manifest and
checks collect() ranks by score = T@500 RMSE + |T bias|, lowest first.
"""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_aimip_sweep_sc", REPO / "scripts" / "validate" / "aimip_sweep_scorecard.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _write_combo(d: Path, name: str, T, T_sfc, u, v, ps, T_bias):
    cdir = d / name
    cdir.mkdir(parents=True)
    sc = {"variants": {"classical": {"eval_metrics": {
        "rmse": {"T": {"mean": T}, "T_sfc": {"mean": T_sfc},
                 "u": {"mean": u}, "v": {"mean": v}, "p_s": {"mean": ps}},
        "bias": {"T": {"mean": T_bias}}}}}}
    (cdir / "aimip_scorecard.json").write_text(json.dumps(sc))


def test_collect_ranks_by_score(tmp_path):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    # baseline T=1.5,bias=0.1 -> score 1.6 ; good T=1.2,bias=0.0 -> 1.2 ;
    # bad T=2.0,bias=-0.3 -> 2.3
    _write_combo(sweep, "combo_baseline", 1.5, 1.9, 3.0, 3.0, 100.0, 0.1)
    _write_combo(sweep, "combo_conv_x", 1.2, 1.8, 2.8, 2.8, 95.0, 0.0)
    _write_combo(sweep, "combo_turb_y", 2.0, 3.0, 3.2, 3.1, 110.0, -0.3)
    manifest = {"combos": [
        {"name": "combo_baseline", "overrides": {"aimip_convection": "tiedtke"}},
        {"name": "combo_conv_x", "overrides": {"aimip_convection": "x"}},
        {"name": "combo_turb_y", "overrides": {"aimip_turbulence": "y"}},
    ]}
    rows = mod.collect(sweep, manifest)
    assert [r["name"] for r in rows] == ["combo_conv_x", "combo_baseline", "combo_turb_y"]
    assert abs(rows[0]["score"] - 1.2) < 1e-9
    assert abs(rows[1]["score"] - 1.6) < 1e-9     # 1.5 + |0.1|
    assert rows[0]["schemes"] == {"convection": "x"}
    # text table renders + names the best
    txt = mod._text_table(rows)
    assert "BEST: combo_conv_x" in txt


def test_failed_combo_skipped(tmp_path):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    _write_combo(sweep, "combo_ok", 1.3, 1.8, 2.8, 2.8, 95.0, 0.0)
    # an errored combo: no eval_metrics
    (sweep / "combo_bad").mkdir()
    (sweep / "combo_bad" / "aimip_scorecard.json").write_text(
        json.dumps({"variants": {"classical": {"error": "NaN"}}}))
    rows = mod.collect(sweep, {"combos": []})
    assert [r["name"] for r in rows] == ["combo_ok"]


def test_stale_combo_excluded_by_manifest(tmp_path):
    """A combo dir not in the current manifest (e.g. a leftover *_none from a
    prior sweep) is excluded so it can't corrupt the ranking."""
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    _write_combo(sweep, "combo_conv_x", 1.2, 1.8, 2.8, 2.8, 95.0, 0.0)
    _write_combo(sweep, "combo_turb_none", 0.5, 0.5, 1.0, 1.0, 50.0, 0.0)  # stale, best-looking
    manifest = {"combos": [{"name": "combo_conv_x",
                            "overrides": {"aimip_convection": "x"}}]}
    rows = mod.collect(sweep, manifest)
    assert [r["name"] for r in rows] == ["combo_conv_x"]  # stale excluded despite lower score


if __name__ == "__main__":
    import tempfile
    for fn in (test_collect_ranks_by_score, test_failed_combo_skipped,
               test_stale_combo_excluded_by_manifest):
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
        print(f"  ok  {fn.__name__}")
    print("aimip_sweep_scorecard: self-checks passed")
