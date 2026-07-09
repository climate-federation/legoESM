"""Every shipped LMIP template under templates/land/biophysics/ must validate.

A template that fails ``validate_config`` is a broken experiment definition that
would only surface at ``init_experiment.py`` / job-submission time (on the
cluster, after queueing) — this catches it in CI instead.  Globs the directory
so a NEW template is covered automatically.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from legoesm.land.lmip_config import validate_config

_TEMPLATE_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "templates" / "land" / "biophysics"
)
_TEMPLATES = sorted(_TEMPLATE_DIR.glob("*.yaml"))


def test_biophysics_templates_present():
    # Guard against a bad path silently parametrizing zero cases (a vacuous pass).
    assert _TEMPLATES, f"no templates found under {_TEMPLATE_DIR}"


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_biophysics_template_validates(path):
    cfg = validate_config(yaml.safe_load(path.read_text()))
    # Sanity: the resolved config exposes the required top-level sections.
    assert cfg.physics["land_mode"] in ("multilayer", "slab")
    assert cfg.time["n_steps"] > 0


def test_canopy_4deg_smoke_defaults():
    """The 4° smoke ships with the merged-code updates enabled — the whole point
    of the run (freeze/thaw stabilisation + per-cell canopy albedo)."""
    path = _TEMPLATE_DIR / "lmip_canopy_4deg_smoke.yaml"
    cfg = validate_config(yaml.safe_load(path.read_text()))
    assert cfg.physics["surface_scheme"] == "two_leaf_canopy"
    assert cfg.physics["enable_freeze_thaw"] is True
    assert cfg.physics["snow_albedo_feedback"] is True
    assert cfg.grid["resolution"] == 45


def test_lulcc_template_declares_transient_cover_and_eluc():
    """The LULCC option template declares a transient reconstruction dataset + the
    E_LUC bookkeeping block (the schema accepts it; the driver validates the
    surfdata is actually transient at run time)."""
    path = _TEMPLATE_DIR / "lmip_canopy_lulcc.yaml"
    cfg = validate_config(yaml.safe_load(path.read_text()))
    assert cfg.surfdata["land_cover_dataset"] != "clm5"        # a real reconstruction
    assert cfg.raw.get("land_use_change", {}).get("scheme") == "bookkeeping"
