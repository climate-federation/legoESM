"""``--tke-surface-bc-level`` and where the surface TKE value is held.

NEMO holds ``en(1)`` at the z=0 W-point and SOLVES the tridiagonal from
``jk=2`` (``zdftke.F90:264,403-410``), so its first interior interface is a
genuinely solved row that receives downward TKE transport. legoESM's default
``interior_pinned`` pins that row ITSELF, one w-level deeper, and a pinned row
can receive no transport at all -- the measured signature is a turbulent layer
ending at 25 m against the oracle's 73 m.

The option was reverted from the ORCA1 card on 2026-09-08 because the
virtual-surface face distance was then taken from ``dz_surface``, the top
cell's midpoint depth and half of ``e3t(1)``, which doubled the coupling.
#1690 gave that face its own metric (``dz_face_surface = dz_ref[0]*jacobian``,
derived inside ``tke_vertical_mixing``), so the revert's reason no longer
holds. This module pins the flag that re-exposes it, and the two preconditions
that must raise rather than be silently ignored.
"""
from __future__ import annotations

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_card_default_is_unchanged_without_the_flag():
    """No existing run moves; the flag reports the gap, it does not hide it."""
    assert _core2().orca1_zdftke_config().tke_surface_bc_level == "interior_pinned"


def test_flag_reaches_the_closure_config():
    cfg = _core2().orca1_zdftke_config(surface_bc_level="nemo_z0")
    assert cfg.tke_surface_bc_level == "nemo_z0"
    # the card already satisfies both preconditions
    assert cfg.surface_bc == "nemo_dirichlet"
    assert cfg.tke_mxl_choice in (3, 4)


def test_explicit_default_value_is_also_accepted():
    cfg = _core2().orca1_zdftke_config(surface_bc_level="interior_pinned")
    assert cfg.tke_surface_bc_level == "interior_pinned"


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="surface_bc_level"):
        _core2().orca1_zdftke_config(surface_bc_level="z0")


def test_nemo_z0_requires_the_dirichlet_surface_value():
    """The virtual surface row IS the held value, so a flux BC makes nemo_z0
    meaningless rather than merely different."""
    with pytest.raises(ValueError, match="nemo_dirichlet"):
        _core2().orca1_zdftke_config(surface_bc_level="nemo_z0",
                                     surface_bc="veros_flux")


def test_nemo_z0_requires_the_mxl0_anchor():
    """The surface viscosity comes from the ln_mxl0 anchor, which exists only
    for tke_mxl_choice 3 or 4; without it the closure writes None and the
    carried-coefficient seeding goes partial (codex 9693003)."""
    with pytest.raises(ValueError, match="tke_mxl_choice"):
        _core2().orca1_zdftke_config(surface_bc_level="nemo_z0", mxl_choice=2)


def test_it_rides_the_tripole_vmix_builder():
    vm = _core2().build_tripole_vmix_config(
        "tke", tke_surface_bc_level="nemo_z0")
    assert vm.tke.tke_surface_bc_level == "nemo_z0"


def test_builder_refuses_it_without_the_tke_closure():
    with pytest.raises(ValueError, match="--tke-surface-bc-level"):
        _core2().build_tripole_vmix_config(
            "kpp", tke_surface_bc_level="nemo_z0")


def test_parser_exposes_both_choices_and_defaults_to_none():
    p = _core2()._build_arg_parser()
    assert p.parse_args([]).tke_surface_bc_level is None
    assert p.parse_args(
        ["--tke-surface-bc-level", "nemo_z0"]).tke_surface_bc_level == "nemo_z0"


def test_parser_rejects_an_unknown_choice():
    with pytest.raises(SystemExit):
        _core2()._build_arg_parser().parse_args(
            ["--tke-surface-bc-level", "z0"])


# --- forwarding hops, the gap codex found on the previous flag --------------

def _fwd(func_name, callee, kwarg):
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(_core2()))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == func_name)
    found = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = node.func
            name = (target.attr if isinstance(target, ast.Attribute)
                    else getattr(target, "id", None))
            if name == callee:
                found.append([k for k in node.keywords if k.arg == kwarg])
    assert found, f"{func_name} never calls {callee}"
    return found


def test_build_tripole_forwards_the_flag():
    for kws in _fwd("build_tripole", "build_tripole_vmix_config",
                    "tke_surface_bc_level"):
        assert kws


def test_main_forwards_the_parsed_flag():
    import ast
    for kws in _fwd("main", "build_tripole", "tke_surface_bc_level"):
        assert kws, ("main calls build_tripole without forwarding the flag; "
                     "it would parse and then be silently discarded")
        assert isinstance(kws[0].value, ast.Attribute)
        assert kws[0].value.attr == "tke_surface_bc_level"


def test_the_forwarding_probe_can_fail():
    """Non-vacuity: the probe must report a missing keyword as missing."""
    for kws in _fwd("build_tripole", "build_tripole_vmix_config",
                    "a_keyword_no_caller_passes"):
        assert not kws


# --- codex 9698860 [HIGH]: validator scope, previously uncovered ------------

@pytest.mark.parametrize("grid,kw", [
    ("mpas", {"mpas_vmix": "kpp"}),
    ("latlon_bathy", {}),
])
def test_validator_refuses_the_flag_where_no_tke_closure_runs(grid, kw):
    """The flag was accepted on any grid and then silently dropped -- the
    dispatch footgun this validator exists to prevent."""
    with pytest.raises(SystemExit, match="--tke-surface-bc-level"):
        _core2()._validate_tke_card_grid(
            grid, tke_surface_bc_level="nemo_z0", **kw)


def test_validator_accepts_the_tripole_tke_configuration():
    """Non-vacuity for the refusals above: the supported case must pass, or a
    blanket rejection would satisfy them."""
    _core2()._validate_tke_card_grid(
        "tripole", tripole_vmix="tke", tke_surface_bc_level="nemo_z0")


def test_preconditions_read_the_resolved_card_not_its_defaults():
    """ORDERING regression: the guard ran before surface_bc/tke_mxl_choice
    were applied, so it validated the card defaults instead of the requested
    values and let an illegal combination through."""
    # mxl_choice=2 is requested here; the card default is legal, so a guard
    # reading the default would NOT raise.
    with pytest.raises(ValueError, match="tke_mxl_choice"):
        _core2().orca1_zdftke_config(surface_bc_level="nemo_z0", mxl_choice=2)
    # same shape for the surface BC
    with pytest.raises(ValueError, match="nemo_dirichlet"):
        _core2().orca1_zdftke_config(surface_bc_level="nemo_z0",
                                     surface_bc="veros_flux")
    # and the legal combination still resolves
    cfg = _core2().orca1_zdftke_config(surface_bc_level="nemo_z0",
                                       mxl_choice=4)
    assert cfg.tke_surface_bc_level == "nemo_z0" and cfg.tke_mxl_choice == 4


def test_main_forwards_the_flag_to_every_direct_vmix_build():
    """The FESOM lane builds its TKE card in ``main`` directly; dropping the
    flag there makes ``--tke-surface-bc-level`` a silent no-op on FESOM."""
    for kws in _fwd("main", "build_tripole_vmix_config",
                    "tke_surface_bc_level"):
        assert kws, ("main builds a vmix config without forwarding "
                     "tke_surface_bc_level; the flag would be discarded")


def test_fesom_stage_gate_admits_the_flag():
    """The FESOM stage-B4 allowlist refuses any dest it does not list, so a
    card that names the placement died at launch even though main forwards it."""
    assert "tke_surface_bc_level" in _core2()._FESOM_WIRED_DESTS
