"""CLI + config wiring for run_coupled's --carbon-ic (the finidat->run carbon-IC
loader).  The flag flows to CoupledConfig.carbon_ic_path via the preset override
layer, and the coupled default is a cold-start ("" path).

Pure config-level tests (no JAX step / no driver setup): the seeding + phi
threading + grid-match physics are covered by tests/land/unit/test_global_init.py
(the loader) and tests/unit/test_coupler.py (init_surface_state override + the
end-to-end coupled integration)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_mod", REPO / "scripts" / "run" / "run_coupled.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_carbon_ic_flag_defaults_to_cold_start():
    """Unset --carbon-ic => "" => cold-start carbon (byte-identical legacy)."""
    args = mod.build_parser().parse_args([])
    assert args.carbon_ic == ""


def test_carbon_ic_flag_roundtrip():
    args = mod.build_parser().parse_args(["--carbon-ic", "results/gcic/global_carbon_ic.npz"])
    assert args.carbon_ic == "results/gcic/global_carbon_ic.npz"


def test_coupled_config_default_carbon_ic_path_is_empty():
    from legoesm.driver.coupled_config import CoupledConfig
    assert CoupledConfig().carbon_ic_path == ""


def test_preset_override_flows_carbon_ic_path():
    # The CLI adds overrides["carbon_ic_path"] and every preset does
    # defaults.update(overrides) -> CoupledConfig(**defaults), so the finidat path
    # reaches the CoupledConfig the driver reads.
    from legoesm.driver.coupled_config import preset_slab_carbon
    cfg = preset_slab_carbon(carbon_ic_path="p.npz")
    assert cfg.carbon_ic_path == "p.npz"


def test_carbon_ic_override_only_set_when_nonempty():
    """The run_coupled main() adds overrides["carbon_ic_path"] ONLY when
    --carbon-ic is non-empty, so an unset flag never overrides a preset's default
    (mirrors the `if getattr(args, "carbon_ic", "")` guard)."""
    overrides = {}
    args = mod.build_parser().parse_args([])
    if getattr(args, "carbon_ic", ""):
        overrides["carbon_ic_path"] = args.carbon_ic
    assert "carbon_ic_path" not in overrides
    args2 = mod.build_parser().parse_args(["--carbon-ic", "x.npz"])
    if getattr(args2, "carbon_ic", ""):
        overrides["carbon_ic_path"] = args2.carbon_ic
    assert overrides["carbon_ic_path"] == "x.npz"
