"""Smoke test for the LES-tuning publication figure script.

Renders all three figures from SYNTHETIC scores + profiles into a temp dir and
asserts the PDF+PNG pair for each lands on disk. matplotlib only; no model
import, no dependency on the real campaign outputs.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts/plot/plot_les_tuning_publication.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("pubfig", _SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["pubfig"] = m
    spec.loader.exec_module(m)
    return m


def _synthetic(tmp, mod):
    # scores.json in the full-key shape the campaign writes
    rows = []
    for i, s in enumerate(["ysu", "louis", "clubb", "tke", "smagorinsky",
                           "mynn25", "edmf", "clubb_lite", "holtslag_boville"]):
        pc = {c: 0.2 + 0.1 * ((i + j) % 5) for j, c in enumerate(mod.CASES)}
        rows.append({"scheme": s, "n_seeds": 5, "default": 1.0 - 0.03 * i,
                     "tuned": 0.9 - 0.03 * i, "tuned_mean": 0.9 - 0.03 * i,
                     "tuned_std": 0.02,
                     "pc_default": {c: v + 0.1 for c, v in pc.items()},
                     "pc_tuned": pc})
    (tmp / "scores.json").write_text(json.dumps(rows))

    # profile npz: one per case with the keys fig_profiles reads
    nz = 12
    z = np.linspace(0.0, 1500.0, nz)
    mask = np.ones(nz, bool)
    prof = tmp / "prof"
    prof.mkdir()
    clubb = tmp / "clubb"
    clubb.mkdir()
    schemes = ["ysu", "smagorinsky", "louis", "mynn25", "edmf"]
    for case in mod.CASES:
        d = {"z_scm": z, "mask": mask,
             "les_scmlev_theta": 290 + 0.004 * z,
             "les_scmlev_qv": 1e-2 - 5e-6 * z}
        for s in schemes:
            d[f"scm_{s}_tuned_theta"] = 290 + 0.004 * z + np.random.default_rng(1).normal(0, .1, nz)
            d[f"scm_{s}_tuned_qv"] = 1e-2 - 5e-6 * z
        np.savez(prof / f"profiles_{case}.npz", **d)
        # clubb comes from a separate dir
        np.savez(clubb / f"profiles_{case}.npz",
                 **{"z_scm": z, "mask": mask,
                    "scm_clubb_tuned_theta": 290 + 0.004 * z,
                    "scm_clubb_tuned_qv": 1e-2 - 5e-6 * z})
    return prof, clubb


def test_all_three_figures_render(mod, tmp_path):
    prof, clubb = _synthetic(tmp_path, mod)
    out = tmp_path / "out"
    out.mkdir()
    rows = mod.load_scores(str(tmp_path / "scores.json"))
    mod.fig_scores(rows, str(out))
    mod.fig_percase(rows, str(out))
    mod.fig_profiles(str(prof), str(clubb), str(out))
    for stem in ("fig1_scores", "fig2_percase", "fig3_profiles"):
        for ext in ("pdf", "png"):
            f = out / f"{stem}.{ext}"
            assert f.exists() and f.stat().st_size > 0, f


def test_scores_sort_best_first(mod, tmp_path):
    _synthetic(tmp_path, mod)
    rows = mod.load_scores(str(tmp_path / "scores.json"))
    tuned = [mod._g(r, "t", "tuned") for r in rows]
    assert tuned == sorted(tuned), "ranking not ascending (best first)"
