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


def test_calibrated_spinup_template_is_the_calibrated_config():
    """The calibrated spin-up template must actually ship all three calibration
    selectors ON and cold-start.  A template that silently reverted to the
    pre-calibration defaults would spin up the WRONG model state and only show
    up as a confusing climatology weeks later."""
    path = _TEMPLATE_DIR / "lmip_calibrated_spinup.yaml"
    cfg = validate_config(yaml.safe_load(path.read_text()))
    p = cfg.physics
    assert p["albedo_calibration"] == "amip_multilayer"
    assert p["root_calibration"] == "amip_multilayer"
    assert (p["soil_n_layers"], p["soil_depth_m"]) == (10, 3.0)
    assert p["enable_freeze_thaw"] is True          # LMIP is ahead of AMIP here
    assert p["surface_scheme"] == "two_leaf_canopy"
    assert p["bulk_scheme"] == "most"
    assert cfg.restart["from"] == ""                # cold start
    assert cfg.time["dt"] == 3600.0
    assert cfg.time["n_steps"] == 87600             # 10 yr hourly, noleap


def test_calibrated_spinup_and_production_share_physics():
    """The spin-up and the production template it chains into must run the SAME
    physics — a mismatch would put a discontinuity at the restart boundary."""
    spin = validate_config(yaml.safe_load(
        (_TEMPLATE_DIR / "lmip_calibrated_spinup.yaml").read_text())).physics
    prod = validate_config(yaml.safe_load(
        (_TEMPLATE_DIR / "lmip_canopy_10yr.yaml").read_text())).physics
    for key in ("albedo_calibration", "root_calibration", "soil_n_layers",
                "soil_depth_m", "soil_growth_factor", "enable_freeze_thaw",
                "surface_scheme", "bulk_scheme", "snow_scheme",
                "stomatal_model", "stomata_enabled"):
        assert spin[key] == prod[key], (
            f"physics.{key} differs between the spin-up and production "
            f"templates ({spin[key]!r} vs {prod[key]!r}) — the chained run "
            "would jump physics at the restart boundary")


# --------------------------------------------------------------------------
# Template surfdata must match what the downloader actually stages
# --------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DOWNLOADER = _REPO_ROOT / "scripts" / "data" / "download_lmip_data.sh"

# Templates that intentionally use a DIFFERENT surfdata (documented, not drift).
_SURFDATA_EXEMPT = {
    "smoke_test": "",                                  # synthetic, injected by tests
    "lmip_canopy_lulcc": "legoesm_surfdata_hyde_1920-1929.nc",   # transient LULCC
}


def _downloader_surfdata_name() -> str:
    """The single dated surfdata build ``download_lmip_data.sh`` stages."""
    import shlex
    for line in _DOWNLOADER.read_text().splitlines():
        if line.strip().startswith("SURFDATA_NAME="):
            # shlex handles the quoting AND drops the trailing shell comment.
            rhs = shlex.split(line.split("=", 1)[1], comments=True)
            assert rhs, f"could not parse SURFDATA_NAME from: {line!r}"
            name = rhs[0]
            assert name.endswith(".nc"), f"parsed {name!r}, expected a .nc filename"
            return name
    raise AssertionError(f"SURFDATA_NAME not found in {_DOWNLOADER}")


def test_downloader_surfdata_name_parses():
    """Guard the parser itself — a silently mis-parsed name would make the
    drift check below vacuous (it would compare against a garbage string)."""
    name = _downloader_surfdata_name()
    assert name.startswith("legoesm_surfdata_") and name.endswith(".nc")
    assert "#" not in name and '"' not in name


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_template_surfdata_matches_the_downloader(path):
    """A template pinning an OLD surfdata build is a silent data bug.

    ``download_lmip_data.sh`` stages exactly one dated build; a template asking
    for a different name either fails at job start (file absent) or — worse —
    picks up a stale leftover on a shared filesystem and runs the WRONG boundary
    data.  That is not hypothetical: every template pinned c250617 while the
    downloader fetched c260716, and c250617 over-states global land area by ~26%
    (187.65 vs 146.94 x10^12 m2 at 2 deg), inflating every global total.
    """
    stem = path.stem
    got = pathlib.PurePath(
        yaml.safe_load(path.read_text())["surfdata"]["path"]).name
    if stem in _SURFDATA_EXEMPT:
        assert got == _SURFDATA_EXEMPT[stem], (
            f"{stem} is surfdata-exempt but its path changed to {got!r}; "
            "update _SURFDATA_EXEMPT deliberately or point it at the downloader build")
        return
    assert got == _downloader_surfdata_name(), (
        f"{stem} pins surfdata {got!r} but download_lmip_data.sh stages "
        f"{_downloader_surfdata_name()!r} — the run would use the wrong boundary data")
