"""End-to-end test for the combined EC-site (flux-tower) figure.

Runs ``scripts/plot/plot_ec_site_combined.py`` on the checked-in model+obs outputs
(``scripts/validate/ec_site_example_data``) and pins the pooled daily skill it
prints to the values quoted in ``docs/land/ec_site_evaluation_runbook.md`` and the
figure caption, so a model change that moves the published numbers fails here
instead of silently diverging from the docs.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PLOTTER = _ROOT / "scripts" / "plot" / "plot_ec_site_combined.py"
_DATA = _ROOT / "scripts" / "validate" / "ec_site_example_data"

# Documented pooled daily NSE vs raw obs and vs closure-corrected obs.
_DOCUMENTED = {"LE": (0.605, 0.687), "H": (0.575, 0.676), "GPP": (0.726, None)}
_TOL = 0.005


@pytest.mark.skipif(not any(_DATA.glob("*.nc")), reason="example outputs not present")
def test_combined_figure_reproduces_documented_skill(tmp_path):
    out = tmp_path / "ecsite"
    proc = subprocess.run(
        [sys.executable, str(_PLOTTER), str(_DATA), str(out)],
        capture_output=True, text=True, cwd=_ROOT, env={**os.environ}, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    png = pathlib.Path(f"{out}_combined.png")
    assert png.exists() and png.stat().st_size > 100_000

    got = {}
    for key, raw, corr in re.findall(
            r"pooled (\w+)\s*: NSE_raw=([-\d.na]+) NSE_corr=([-\d.na]+)", proc.stdout):
        got[key] = (float(raw), float(corr))
    assert set(got) == set(_DOCUMENTED), proc.stdout
    for key, (raw_doc, corr_doc) in _DOCUMENTED.items():
        raw, corr = got[key]
        assert abs(raw - raw_doc) <= _TOL, (key, raw, raw_doc)
        if corr_doc is not None:
            assert abs(corr - corr_doc) <= _TOL, (key, corr, corr_doc)
