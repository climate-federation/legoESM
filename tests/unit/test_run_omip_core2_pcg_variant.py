"""CORE-II OMIP ``--barotropic-pcg-variant`` production wiring (codex scaling
lever 10): the validated single-reduce (Chronopoulos-Gear) PCG body exists and
is dispatch-guarded in ``barotropic_common.solve_helmholtz_implicit``; this
covers the DRIVER exposure — argparse round-trip, the flat-override threading
into the implicit-CN builders (tripole/latlon_bathy via ``replace_flat`` into
the nested BarotropicConfig; mpas via the top-level config field), and the
loud cube refusal (no PCG on the split-explicit subcycle — a silently-ignored
flag is the dispatch footgun CLAUDE.md forbids).
"""

from __future__ import annotations

import pytest

from scripts.run.run_omip_core2 import (
    _build_arg_parser,
    _validate_pcg_variant_grid,
)


# ------------------------------------------------------------------ argparse

def test_cli_round_trip_default_and_values():
    p = _build_arg_parser()
    assert p.parse_args([]).barotropic_pcg_variant is None
    assert p.parse_args(
        ["--barotropic-pcg-variant", "single_reduce"]
    ).barotropic_pcg_variant == "single_reduce"
    assert p.parse_args(
        ["--barotropic-pcg-variant", "standard"]
    ).barotropic_pcg_variant == "standard"


def test_cli_rejects_unknown_variant():
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(
            ["--barotropic-pcg-variant", "chronops"])


# ------------------------------------------------------------ grid guard

def test_guard_rejects_cube_and_allows_pcg_grids():
    with pytest.raises(SystemExit):
        _validate_pcg_variant_grid("cubed_sphere", "single_reduce")
    for grid in ("tripole", "latlon_bathy", "mpas"):
        _validate_pcg_variant_grid(grid, "single_reduce")     # live -> ok
    for grid in ("tripole", "latlon_bathy", "cubed_sphere", "mpas"):
        _validate_pcg_variant_grid(grid, None)                # no flag -> ok


def test_guard_rejects_explicit_solver_combo():
    """codex r1 #4: an explicit-substep override runs no PCG, so a PCG
    variant alongside it would silently measure nothing."""
    for grid in ("tripole", "latlon_bathy", "mpas"):
        with pytest.raises(SystemExit):
            _validate_pcg_variant_grid(grid, "single_reduce",
                                       "explicit_substep")
        _validate_pcg_variant_grid(grid, "single_reduce", "implicit_cn")
        _validate_pcg_variant_grid(grid, "single_reduce", None)
        _validate_pcg_variant_grid(grid, None, "explicit_substep")


# ------------------------------------------------- config-field existence

def test_config_fields_exist_with_measured_defaults():
    """The flat-override key must be a REAL field on both config surfaces
    (a typo'd _ovr key would only fail at run time) with the production
    default 'standard' (single_reduce stays opt-in — regime-dependent)."""
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig

    assert BarotropicConfig().barotropic_implicit_pcg_variant == "standard"
    assert (LatLonCGridOceanConfig().flat_get(
        "barotropic_implicit_pcg_variant") == "standard")

    from legoesm.ocean.mpas_config import MPASOceanConfig

    # MPAS default single_reduce (gpoly x 15, owner decision 2026-10-02;
    # single_reduce_deep is opt-in, evidence in mpas_config.py); the lat-lon
    # C-grid default above is a different operator and stays standard.
    assert MPASOceanConfig().barotropic_implicit_pcg_variant == "single_reduce"


# ------------------------------------------------- builder threading

class _Stop(Exception):
    pass


def test_build_tripole_threads_variant_into_barotropic_config(monkeypatch):
    """build_tripole must route the flag through ``replace_flat`` into
    ``config.barotropic.barotropic_implicit_pcg_variant`` — captured at the
    model rebuild (the first consumer of the overridden config).
    ``_create_setup`` is stubbed with a default real config, so no mesh /
    forcing data is touched."""
    import scripts.run.run_omip_core2 as core2
    from scripts.run import run_omip
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as oml
    from legoesm.ocean.state import LatLonCGridOceanConfig

    mesh_path = run_omip._parse_resolution("tripole", "eorca1")["mesh_path"]

    def _fake_setup(*a, **k):
        return (object(), object(), LatLonCGridOceanConfig(), object(), None)

    captured = {}

    def _capture_model(grid, z_coord, config, *a, **k):
        captured["cfg"] = config
        raise _Stop

    monkeypatch.setattr(run_omip, "_create_setup", _fake_setup)
    # build_tripole re-imports the class from the module inside the override
    # branch, so patching the module attribute intercepts the rebuild.
    monkeypatch.setattr(oml, "LatLonCGridOceanModel", _capture_model)
    with pytest.raises(_Stop):
        core2.build_tripole(5, 6000.0, mesh_path,
                            barotropic_pcg_variant="single_reduce")
    assert (captured["cfg"].barotropic.barotropic_implicit_pcg_variant
            == "single_reduce")


def test_build_mpas_ocean_threads_variant(monkeypatch, capsys):
    """build_mpas_ocean must apply the flag via ``config._replace`` (top-level
    MPAS field).  _create_setup is stubbed with a minimal real MPAS config;
    the run stops at the mesh read, AFTER the override was applied+printed
    (a typo'd key would raise at ``_replace`` instead)."""
    import scripts.run.run_omip_core2 as core2
    from scripts.run import run_omip
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig

    cfg0 = MPASOceanConfig()._replace(physics=OceanPhysicsConfig())

    def _fake_setup(*a, **k):
        return (object(), object(), cfg0, object(), None)

    def _stop_mesh_read(path, **_kw):
        raise _Stop

    monkeypatch.setattr(run_omip, "_create_setup", _fake_setup)
    monkeypatch.setattr(core2, "read_mesh_mask_bathy", _stop_mesh_read)
    with pytest.raises(_Stop):
        core2.build_mpas_ocean(5, 6000.0, "dummy_mesh.nc", level=5,
                               barotropic_solver="explicit_substep",
                               barotropic_pcg_variant="standard")
    out = capsys.readouterr().out
    # the NON-default value, so the print proves the flag was threaded
    assert "barotropic_implicit_pcg_variant" in out
    assert "'standard'" in out
    # codex r1 #1: the MAIN call site forwards --barotropic-solver to the
    # MPAS builder too — lock the builder-side threading for BOTH keys.
    assert "'barotropic_solver': 'explicit_substep'" in out
