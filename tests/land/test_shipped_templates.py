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


def test_smoke_4deg_defaults():
    """The 4° smoke ships the production physics — canopy + freeze/thaw — so a
    clean smoke is evidence about the configuration the 2° template runs."""
    path = _TEMPLATE_DIR / "smoke_4deg.yaml"
    cfg = validate_config(yaml.safe_load(path.read_text()))
    assert cfg.physics["surface_scheme"] == "two_leaf_canopy"
    assert cfg.physics["enable_freeze_thaw"] is True
    assert cfg.physics["snow_albedo_feedback"] is True
    assert cfg.grid["resolution"] == 45


def test_lmip_biophys_2deg_ships_calibration_on():
    """The 2° template IS the calibrated configuration (the one the published
    soil-state IC was spun up with): all three calibration selectors on, cold
    start by default, and the corrected c260716 surfdata.  A silent revert to
    pre-calibration defaults would spin up the wrong state and only surface
    weeks later as a confusing climatology — pin it here."""
    path = _TEMPLATE_DIR / "lmip_biophys_2deg.yaml"
    cfg = validate_config(yaml.safe_load(path.read_text()))
    assert cfg.physics["albedo_calibration"] == "amip_multilayer"
    assert cfg.physics["root_calibration"] == "amip_multilayer"
    assert (cfg.physics["soil_n_layers"], cfg.physics["soil_depth_m"]) == (10, 3.0)
    assert cfg.physics["snow_scheme"] == "single"
    assert cfg.restart["from"] == ""                       # cold start = spin-up
    assert cfg.surfdata["path"].endswith("c260716.nc")     # not the superseded c250617


def test_smoke_matches_production_physics():
    """The 4° smoke and the 2° template agree on every physics key, so a green
    smoke certifies the production configuration and not a diverged cousin."""
    smoke = validate_config(yaml.safe_load(
        (_TEMPLATE_DIR / "smoke_4deg.yaml").read_text()))
    prod = validate_config(yaml.safe_load(
        (_TEMPLATE_DIR / "lmip_biophys_2deg.yaml").read_text()))
    assert smoke.physics == prod.physics


def test_lulcc_template_declares_transient_cover_and_eluc():
    """The LULCC option template declares a transient reconstruction dataset + the
    E_LUC bookkeeping block (the schema accepts it; the driver validates the
    surfdata is actually transient at run time)."""
    path = _TEMPLATE_DIR / "lmip_canopy_lulcc.yaml"
    cfg = validate_config(yaml.safe_load(path.read_text()))
    assert cfg.surfdata["land_cover_dataset"] != "clm5"        # a real reconstruction
    assert cfg.raw.get("land_use_change", {}).get("scheme") == "bookkeeping"
