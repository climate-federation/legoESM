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
    soil-state IC was spun up with): calibrated albedo + glacier + per-PFT root
    values present, AMIP soil grid, cold start by default, and the corrected
    c260716 surfdata.  A silent revert to pre-calibration defaults would spin
    up the wrong state and only surface weeks later as a confusing
    climatology — pin it here."""
    path = _TEMPLATE_DIR / "lmip_biophys_2deg.yaml"
    cfg = validate_config(yaml.safe_load(path.read_text()))
    assert cfg.physics["albedo"], "calibrated albedo block missing"
    assert cfg.physics["glacier_albedo_vis"] is not None
    assert cfg.physics["root_depth_per_pft"] is not None
    assert (cfg.physics["soil_n_layers"], cfg.physics["soil_depth_m"]) == (10, 3.0)
    assert cfg.physics["snow_scheme"] == "single"
    assert cfg.restart["from"] == ""                       # cold start = spin-up
    assert cfg.surfdata["path"].endswith("c260716.nc")     # not the superseded c250617


# The 2026-07 calibration SNAPSHOT the published 2° run / soil-state IC used
# (clm_surface_map at the run commit 14461553 on lmip-calibrated-config).  The
# coupled land's clm_surface_map has since been RETUNED (e.g. glacier broadband
# 0.7981 vs this 0.7178), so the live constants can no longer serve as the
# reference — the template deliberately freezes the as-run values, and this
# snapshot is what protects them from a fat-fingered YAML edit.
_ASRUN_ALBEDO = {"alpha_snow_max": 0.8077, "alpha_snow_min": 0.5207,
                 "snow_depth_crit": 15.4229, "tau_snow_decay": 3.6739 * 86400.0,
                 "soil_dry_albedo_boost": 0.1458}
_ASRUN_GLACIER_BROADBAND = 0.7178
_ASRUN_ROOT_DEPTH = (0.088, 1.706, 1.469, 1.414, 1.631, 1.675, 1.527, 1.408,
                     1.088, 0.716, 0.622, 0.723, 0.457, 0.507, 0.504, 0.488,  # const-ok: per-PFT root DEPTH [m] snapshot; 0.622 is a rooting depth, not epsilon
                     0.477)


def test_template_values_match_the_asrun_calibration_snapshot():
    """The template's numbers ARE the reproduction contract for the published
    run and its soil-state IC — pin them to the as-run snapshot so an edit is a
    deliberate, named act.  (An equality pin against the LIVE clm_surface_map
    constants would be wrong: the coupled land has been retuned since, and
    following it would silently break reproduction.)"""
    path = _TEMPLATE_DIR / "lmip_biophys_2deg.yaml"
    p = validate_config(yaml.safe_load(path.read_text())).physics
    for k, v in _ASRUN_ALBEDO.items():
        assert p["albedo"][k] == pytest.approx(v), k
    # Glacier pair: integrates to the as-run broadband under the 0.5/0.5
    # weights, preserving the uncalibrated pair's 0.20 vis-NIR contrast.
    vis, nir = p["glacier_albedo_vis"], p["glacier_albedo_nir"]
    assert 0.5 * (vis + nir) == pytest.approx(_ASRUN_GLACIER_BROADBAND)
    assert vis - nir == pytest.approx(0.20)
    assert tuple(p["root_depth_per_pft"]) == pytest.approx(_ASRUN_ROOT_DEPTH)
    # wp < fc elementwise — a swapped pair makes beta_root degenerate.
    assert all(w < f for w, f in zip(p["theta_wp_per_pft"],
                                     p["theta_fc_per_pft"]))


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
