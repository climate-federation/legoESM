"""run_coupled --cmip-drs-tree / --cmip-institution-id / --cmip-source-id /
--cmip-variant-label reach OutputConfig (same fields as run_amip)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from legoesm.driver.config import OutputConfig

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_cmip_writer_mod", REPO / "scripts" / "run" / "run_coupled.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_FIELDS = ("cmip_drs_tree", "cmip_institution_id", "cmip_source_id",
           "cmip_variant_label")


def test_defaults_are_outputconfig_defaults():
    kw = mod.cmip_writer_output_kwargs(mod.build_parser().parse_args([]))
    out0 = OutputConfig()
    assert kw == {f: getattr(out0, f) for f in _FIELDS}


def test_flags_round_trip_to_outputconfig():
    args = mod.build_parser().parse_args([
        "--cmip-drs-tree", "--cmip-institution-id", "INST-X",
        "--cmip-source-id", "SRC-1-0", "--cmip-variant-label", "r2i1p1f1"])
    out = OutputConfig(**mod.cmip_writer_output_kwargs(args))
    assert (out.cmip_drs_tree, out.cmip_institution_id, out.cmip_source_id,
            out.cmip_variant_label) == (True, "INST-X", "SRC-1-0", "r2i1p1f1")


def test_built_driver_carries_the_flags(tmp_path):
    """main(build_only=True) on the committed slab deck: the flags reach the
    driver's OutputConfig, not just the parser."""
    driver = mod.main([
        "--config", str(REPO / "config" / "cmip" / "cmip_ocean_slab.yaml"),
        "--ozone-file", "ozone.nc", "--solar-file", "solar.nc",
        "--ghg-file", "ghg.nc", "--aerosol-file", "aerosol.nc",
        "--woa-t-path", "woa_t.nc", "--woa-s-path", "woa_s.nc",
        "--output", str(tmp_path),
        "--cmip-drs-tree", "--cmip-institution-id", "INST-X",
        "--cmip-source-id", "SRC-1-0", "--cmip-variant-label", "r2i1p1f1",
    ], build_only=True)
    out = driver.atm_config.output
    assert (out.cmip_drs_tree, out.cmip_institution_id, out.cmip_source_id,
            out.cmip_variant_label) == (True, "INST-X", "SRC-1-0", "r2i1p1f1")
