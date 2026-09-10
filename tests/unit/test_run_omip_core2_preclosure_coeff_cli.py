"""``--tke-preclosure-coeff-source`` and NEMO's carried ``avm_k/avt_k``.

NEMO's ``zdf_tke`` consumes the SAVE'd ``avm_k/avt_k`` pair in ``zdf_sh2``, the
Prandtl ratio, the TKE matrix and the RHS, and overwrites that pair only AFTER
the solve (``zdftke.F90``); it never evaluates shear production against a
coefficient produced inside the same call.  legoESM implements that lifetime as
``TKEConfig.tke_preclosure_coeff_source="carried_previous_step"`` and the DINO
NEMO-oracle preset selects it, but ``orca1_zdftke_config`` never did -- so every
OMIP/ORCA1 run evaluated production against a freshly recomputed coefficient.

This module pins the flag that closes that gap: the card default is unchanged
(so no run moves without the flag), the flag reaches the config, an unknown
value raises rather than silently no-opping, and the flag is refused on the two
grids whose ocean state has no coefficient memory to carry.
"""
from __future__ import annotations

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_card_default_is_unchanged_without_the_flag():
    """No existing run moves. The gap is REPORTED by the flag, not by a
    silent default change (CLAUDE.md: a fix whose default keeps the bug)."""
    cfg = _core2().orca1_zdftke_config()
    assert cfg.tke_preclosure_coeff_source == "current_subiteration"


def test_flag_reaches_the_closure_config():
    cfg = _core2().orca1_zdftke_config(
        preclosure_coeff_source="carried_previous_step")
    assert cfg.tke_preclosure_coeff_source == "carried_previous_step"


def test_explicit_default_value_is_also_accepted():
    """The A arm of the controlled pair names the value explicitly rather
    than relying on the default (CLAUDE.md: a default is not a record)."""
    cfg = _core2().orca1_zdftke_config(
        preclosure_coeff_source="current_subiteration")
    assert cfg.tke_preclosure_coeff_source == "current_subiteration"


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="preclosure_coeff_source"):
        _core2().orca1_zdftke_config(preclosure_coeff_source="carried")


def test_it_rides_the_tripole_vmix_builder():
    vm = _core2().build_tripole_vmix_config(
        "tke", tke_preclosure_coeff_source="carried_previous_step")
    assert vm.scheme == "tke"
    assert vm.tke.tke_preclosure_coeff_source == "carried_previous_step"


def test_builder_refuses_it_without_the_tke_closure():
    with pytest.raises(ValueError, match="--tke-preclosure-coeff-source"):
        _core2().build_tripole_vmix_config(
            "kpp", tke_preclosure_coeff_source="carried_previous_step")


@pytest.mark.parametrize("grid,vmix_kw", [
    ("mpas", {"mpas_vmix": "tke"}),
    ("fesom", {"fesom_vmix": "legoesm_tke"}),
])
def test_grids_without_coefficient_memory_are_refused(grid, vmix_kw):
    """MPAS and FESOM run the same closure but their ocean states carry no
    tke_avm/tke_avt slots, so the flag would be accepted and do nothing.  The
    looser card-knob loop admits them, which is why this needs its own guard."""
    with pytest.raises(SystemExit, match="carried avm_k"):
        _core2()._validate_tke_card_grid(
            grid, tke_preclosure_coeff_source="carried_previous_step",
            **vmix_kw)


def test_the_tripole_closure_is_accepted_by_the_same_guard():
    """Non-vacuity for the guard above: the one supported configuration must
    pass, or the parametrized test would be satisfied by a blanket refusal."""
    _core2()._validate_tke_card_grid(
        "tripole", tripole_vmix="tke",
        tke_preclosure_coeff_source="carried_previous_step")


def test_parser_exposes_both_choices_and_defaults_to_none():
    p = _core2()._build_arg_parser()
    args = p.parse_args([])
    assert args.tke_preclosure_coeff_source is None
    args = p.parse_args(["--tke-preclosure-coeff-source",
                         "carried_previous_step"])
    assert args.tke_preclosure_coeff_source == "carried_previous_step"


def test_parser_rejects_an_unknown_choice():
    p = _core2()._build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--tke-preclosure-coeff-source", "carried"])
