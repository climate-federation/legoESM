"""Golden characterization of the YAML -> ExperimentConfig translation.

This pins the *exact current output* of ``config.Config.to_experiment_config``
so the Stage-A1 keystone refactor (collapsing the hand-written translator onto
the canonical ``experiment_config_from_dict``) can be proven bit-identical
instead of merely "looks right".

It is a **characterization** test: it asserts invariance, not correctness.  Any
*intentional* behaviour change (e.g. finally honouring the legacy ``equations``
key, or accepting ``radiation`` as a bare string) must regenerate the goldens /
update the explicit asserts in a dedicated diff so the change is visible in
review — exactly the bit-repro discipline the master plan mandates.

Scope: the standalone ``config/*.yaml`` files that ``legoesm run`` feeds through
``to_experiment_config`` (today, the Williamson cases).  The ``config/aimip/``
configs are consumed by ``scripts/run/run_aimip.py``'s separate AMIP loader, not by
this function (several even use an incompatible ``radiation:`` string schema);
they are folded into this golden surface at Stage C1 when the loaders unify.

Regenerate after an intended change::

    LEGOESM_REGEN_GOLDEN=1 .venv/bin/python -m pytest \
        tests/unit/test_yaml_to_experiment_config_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from legoesm.config import Config
from legoesm.driver.config import experiment_config_to_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "golden" / "yaml_to_experiment"
_REGEN = os.environ.get("LEGOESM_REGEN_GOLDEN") == "1"


# A run config carries at least one of these top-level sections; the loader
# feeds it through ``to_experiment_config``.  ``config/`` also holds non-run
# YAML — e.g. ``data_catalog.yaml`` (a top-level ``datasets:`` catalog consumed
# by a different loader) — which does NOT translate to an ExperimentConfig and
# would otherwise serialize to a meaningless all-defaults golden.
_RUN_CONFIG_SECTIONS = ("grid", "dycore", "model", "atmosphere", "time")


def _config_files() -> list[Path]:
    """Standalone RUN configs the basic loader consumes via ``legoesm run``.

    Filters ``config/*.yaml`` down to genuine run configs (those carrying a
    run-config section), excluding dataset catalogs / other non-run YAML.
    """
    out: list[Path] = []
    for path in sorted((REPO_ROOT / "config").glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            # Surface a malformed run config loudly (with its path) rather
            # than silently dropping it from golden coverage.
            raise AssertionError(f"malformed config YAML {path}: {exc}") from exc
        if isinstance(doc, dict) and any(s in doc for s in _RUN_CONFIG_SECTIONS):
            out.append(path)
    return out


def _golden_path(config_path: Path) -> Path:
    return GOLDEN_DIR / (config_path.stem + ".json")


def _canonical_from_config(cfg: Config) -> dict:
    """Canonical dict of the translated ExperimentConfig, JSON-normalised.

    Round-tripping through JSON makes floats/tuples normalise identically on both
    sides of the comparison (no spurious repr drift).
    """
    ec = cfg.to_experiment_config()
    d = experiment_config_to_dict(ec)
    return json.loads(json.dumps(d, sort_keys=True, default=str))


@pytest.mark.parametrize(
    "config_path", _config_files(), ids=lambda p: p.name
)
def test_yaml_to_experiment_config_matches_golden(config_path: Path) -> None:
    produced = _canonical_from_config(Config.from_yaml(str(config_path)))
    golden = _golden_path(config_path)
    if _REGEN:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(json.dumps(produced, indent=2, sort_keys=True) + "\n")
    assert golden.exists(), (
        f"missing golden {golden.relative_to(REPO_ROOT)}; regenerate with "
        f"LEGOESM_REGEN_GOLDEN=1"
    )
    expected = json.loads(golden.read_text())
    assert produced == expected, (
        f"{config_path.relative_to(REPO_ROOT)} no longer translates to its pinned "
        f"ExperimentConfig. If intentional, regenerate goldens "
        f"(LEGOESM_REGEN_GOLDEN=1) in a dedicated diff."
    )


def test_known_configs_are_covered() -> None:
    """Guard against the glob silently going empty (e.g. a moved config dir)."""
    names = {p.name for p in _config_files()}
    assert {"williamson_test2.yaml", "williamson_test5.yaml"} <= names


# ----------------------------------------------------------------------
# Explicit semantic pin of the YAML-field -> ExperimentConfig-field mapping.
# Files alone only exercise the cubed-sphere / shallow-water corner; these
# in-memory configs cover the renames and unit conversions the translator
# performs, so the refactor cannot quietly drop or rewire one of them.
# ----------------------------------------------------------------------
def test_translation_field_mapping_is_pinned() -> None:
    cfg = Config.from_dict(
        {
            "grid": {
                "type": "latlon",
                "resolution": 96,
                "n_levels": 60,
                "vertical_coord": "sigma",
                "p_top_Pa": 100.0,
                "stretching": 1.5,
            },
            "atmosphere": {
                "dynamics": "hydrostatic",
                "discretization": "spectral",
                "dt_seconds": 300,
                "hyperdiff_scale": 2.0,
            },
            "conservation": {"fix_mass": False},
            "time": {
                "duration_hours": 480,          # -> days = 20
                "output_interval_hours": 36,    # -> diag_days = 36/24 = 1.5
                "start_day": 3.0,
            },
            "output": {
                "path": "out/run/",
                "checkpoint_days": 30,
                "monthly_means": True,
                "checkpoint_format": "zarr",
            },
            "forcing": {"dataset": "era5", "path": "/data/era5.zarr"},
            "radiation": {"scheme": "rrtmgp"},
            "surface": {"T_init": 288.0, "RH_init": 0.8},
            "hardware": {"parallelism": {"distributed": True}},
            "seed": 9,
        }
    )
    ec = cfg.to_experiment_config()

    # Grid renames.
    assert ec.grid.grid_type == "latlon"
    assert ec.grid.resolution == 96
    assert ec.grid.nlev == 60                     # n_levels -> nlev
    assert ec.grid.vertical_coord == "sigma"
    assert ec.grid.p_top_Pa == 100.0
    assert ec.grid.stretching == 1.5
    # Dycore renames + legacy normalization.
    assert ec.dycore.model_type == "hydrostatic"  # dynamics -> model_type
    assert ec.dycore.discretization == "spectral"
    assert ec.dycore.dt == 300.0                  # dt_seconds -> dt
    assert ec.dycore.hyperdiff_scale == 2.0
    assert ec.dycore.fix_mass is False
    assert ec.dycore.conservation_fixer is False  # both driven by conservation.fix_mass
    # Time / integration unit conversions.
    assert ec.days == 20                          # duration_hours // 24
    assert ec.start_day == 3.0
    assert ec.output.diag_days == 1.5             # output_interval_hours / 24, exact
    # Output passthrough.
    assert ec.output.output_dir == "out/run/"
    assert ec.output.checkpoint_days == 30
    assert ec.output.monthly_means is True
    assert ec.output.checkpoint_format == "zarr"
    # Forcing / radiation / surface / hardware.
    assert ec.dataset == "era5"
    assert ec.forcing_path == "/data/era5.zarr"
    assert ec.radiation == "rrtmgp"               # radiation.scheme
    assert ec.T_init == 288.0
    assert ec.rh_init == 0.8
    assert ec.distributed is True
    assert ec.seed == 9                           # master RNG seed (reproducibility)


def test_hyperdiff_scale_defaults_to_canonical_one() -> None:
    """A YAML that omits the key gets the ExperimentConfig default scale."""
    from legoesm.driver.config import DycoreConfig
    ec = Config.from_dict({}).to_experiment_config()
    assert ec.dycore.hyperdiff_scale == DycoreConfig._field_defaults["hyperdiff_scale"] == 1.0


def test_retired_hyperdiffusion_coeff_key_names_its_replacement() -> None:
    with pytest.raises(ValueError, match="hyperdiff_scale"):
        Config.from_dict(
            {"atmosphere": {"hyperdiffusion_coeff": 0.0}}).to_experiment_config()


def test_unread_yaml_keys_are_not_declared() -> None:
    """conservation.fix_energy and output.format reached no ExperimentConfig
    field; the schema must not advertise them."""
    from legoesm.config import DEFAULT_CONFIG
    assert "fix_energy" not in DEFAULT_CONFIG["conservation"]
    assert "format" not in DEFAULT_CONFIG["output"]


def test_sub_daily_output_interval_is_exact() -> None:
    """A 6-hour output interval is a quarter-day cadence, not truncated to 1 day."""
    cfg = Config.from_dict({"time": {"output_interval_hours": 6}})
    assert cfg.to_experiment_config().output.diag_days == 0.25


@pytest.mark.parametrize("hours", [0, 0.0, -6, float("nan")])
def test_non_positive_output_interval_raises(hours) -> None:
    cfg = Config.from_dict({"time": {"output_interval_hours": hours}})
    with pytest.raises(ValueError, match="output_interval_hours must be > 0"):
        cfg.to_experiment_config()


@pytest.mark.parametrize("blk,key,val", [("conservation", "fix_energy", True),
                                         ("output", "format", "zarr")])
def test_removed_yaml_keys_raise(blk, key, val):
    with pytest.raises(ValueError, match=f"{blk}.{key} was removed and did nothing"):
        Config.from_dict({blk: {key: val}}).to_experiment_config()
