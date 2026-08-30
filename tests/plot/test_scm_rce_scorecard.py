"""The SCM-RCE scorecard generator must render all three figures (PDF+PNG).

Uses fabricated arm checkpoints + per-SST JSONs so it has no dependence on the
real campaign artifacts, matching the PR #1697 smoke-test pattern.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _gen():
    p = (Path(__file__).resolve().parents[2] / "scripts" / "plot"
         / "plot_scm_rce_scorecard.py")
    spec = importlib.util.spec_from_file_location("scm_rce_scorecard", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _fabricate(tmp_path):
    arm_root = tmp_path / "arms"
    schemes = ["emanuel", "tiedtke", "kuo"]
    for seed in ("1", "2"):
        d = arm_root / f"arm_thermo_physical_f0joint_seed{seed}"
        d.mkdir(parents=True)
        for i, s in enumerate(schemes):
            (d / f"scheme_{s}.json").write_text(json.dumps({
                "scheme": s,
                "prior": {"thermo_score": 8.0 + i},
                "tuned": {"thermo_score": 2.0 + i + 0.1 * int(seed)}}))
    per = tmp_path / "persst"
    per.mkdir()
    z = np.linspace(30000.0, 0.0, 30).tolist()
    for i, s in enumerate(schemes):
        blk = {}
        for sst in (295, 300, 305):
            blk[str(sst)] = {
                "thermo": (np.inf if (s == "kuo" and sst == 295)
                           else 2.0 + i + 0.01 * sst),
                "status": "failed" if (s == "kuo" and sst == 295) else "ok",
                "T_profile": (np.linspace(300, 200, 30) + i).tolist(),
                "qv_profile": (np.linspace(0.018, 0.0, 30)).tolist(),
                "z_m": z,
                "T_ref": np.linspace(300, 195, 30).tolist(),
                "qv_ref": np.linspace(0.019, 0.0, 30).tolist()}
        (per / f"per_sst_{s}.json").write_text(json.dumps(
            {"scheme": s, "n_preset_params": i, "per_sst": blk}))
    return arm_root, per


def test_all_three_figures_render(tmp_path):
    gen = _gen()
    arm_root, per = _fabricate(tmp_path)
    out = tmp_path / "fig" / "scorecard"
    gen.main(["--arm-root", str(arm_root), "--per-sst-dir", str(per),
              "--seeds", "1", "2", "--out-prefix", str(out)])
    for name in ("scores", "percase", "profiles"):
        for ext in ("pdf", "png"):
            f = tmp_path / "fig" / f"scorecard_{name}.{ext}"
            assert f.exists() and f.stat().st_size > 0, f"missing {f}"


def test_fig1_ranks_best_first(tmp_path):
    gen = _gen()
    arm_root, per = _fabricate(tmp_path)
    order = gen.fig1_scores(arm_root, ["1", "2"],
                            str(tmp_path / "fig" / "sc"))
    # emanuel (tuned ~2.1) < tiedtke (~3.1) < kuo (~4.1).
    assert order == ["emanuel", "tiedtke", "kuo"]
