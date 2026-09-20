"""Flags that the MPAS lane cannot honour must REFUSE, not go quiet.

A 2026-09-20 manifest audit found the MPAS OMIP run had been executing library
defaults for the entire lateral-mixing and GM/Redi stack while its card looked
harmonised with the tripole's. The cause was not a wrong value -- it was
silence: ``build_mpas_ocean`` does not accept those parameters, so passing them
changed nothing and the run reported success. Every cross-grid "MPAS confirms"
statement in that campaign rests on the silence.

The driver already refuses ``--A-h-profile-file`` and the GM operator flags off
the tripole lane, and its own comment names the principle: a knob accepted by
validation and then dropped at the call site is the silent-discard footgun the
guard exists to prevent. These tests extend that to the six flags the audit
found unguarded.

Each test also pins the other half of the contract: the guard must fire only
when the flag was actually SET, so a plain MPAS run still works.
"""
from __future__ import annotations

import pytest


def _parse(argv):
    from scripts.run import run_omip_core2
    return run_omip_core2._build_arg_parser().parse_args(argv)


_MPAS_BASE = ["--grid", "mpas", "--mpas-level", "7"]
_TRIPOLE_BASE = ["--grid", "tripole"]


@pytest.mark.parametrize("flag,value", [
    ("--redi-coefficient", "nemo21"),
    ("--redi-aht0", "900"),
])
def test_redi_knobs_are_refused_off_tripole(flag, value):
    """Both live only in build_tripole."""
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod
    src = Path(mod.__file__).read_text()
    assert "_tripole_only" in src, "the Redi guard is missing entirely"
    block = src.split("_tripole_only = [", 1)[1][:900]
    assert flag in block, f"{flag} is not covered by the tripole-only guard"
    assert "raise SystemExit" in block
    assert 'args.grid != "tripole"' in block


@pytest.mark.parametrize("flag", [
    "--momentum-rk3",
    "--adaptive-implicit-vertadv",
    "--min-levels",
    "--smag-cfl-safety",
    "--gateway-transports",
])
def test_flags_the_mpas_builder_cannot_accept_are_refused(flag):
    from pathlib import Path
    import scripts.run.run_omip_core2 as mod
    src = Path(mod.__file__).read_text()
    assert "_not_on_mpas" in src, "the MPAS guard is missing entirely"
    block = src.split("_not_on_mpas = [", 1)[1][:1100]
    assert flag in block, f"{flag} is not covered by the MPAS guard"
    assert "raise SystemExit" in block
    assert 'args.grid == "mpas"' in block


def test_the_mpas_builder_really_does_not_accept_them():
    """The guard's PREMISE, checked against the signature rather than assumed.

    If someone later wires one of these through build_mpas_ocean, this test
    goes red and the guard should be narrowed rather than left to refuse a
    flag that would now work.
    """
    import inspect
    from scripts.run.run_omip_core2 import build_mpas_ocean

    params = set(inspect.signature(build_mpas_ocean).parameters)
    for name in ("redi_coefficient", "redi_aht0", "momentum_time_integrator",
                 "adaptive_implicit_vertadv", "min_levels",
                 "smag_cfl_safety", "store_mass_flux"):
        assert name not in params, (
            f"build_mpas_ocean now accepts {name!r}; wire it at the call site "
            f"and narrow the guard instead of refusing a supported flag")


def test_the_flags_the_mpas_builder_does_accept_are_not_guarded():
    """Non-vacuity in the other direction: the guard must not over-refuse.

    A_h, C_smag_lap, gm_treguier and gm_kappa_min ARE threaded into
    build_mpas_ocean, so a harmonisation run can still set them.
    """
    import inspect
    from scripts.run.run_omip_core2 import build_mpas_ocean

    params = set(inspect.signature(build_mpas_ocean).parameters)
    for name in ("A_h", "C_smag_lap", "gm_treguier", "gm_kappa_min"):
        assert name in params

    from pathlib import Path
    import scripts.run.run_omip_core2 as mod
    src = Path(mod.__file__).read_text()
    guard = src.split("_not_on_mpas = [", 1)[1][:1100]
    for flag in ("--A-h", "--C-smag-lap", "--gm-treguier", "--gm-kappa-min"):
        assert flag not in guard, (
            f"{flag} IS supported on MPAS and must not be refused")


def test_a_plain_mpas_run_sets_none_of_the_guarded_flags():
    """The guards key off explicit use, so defaults must not trip them."""
    a = _parse(_MPAS_BASE)
    assert a.redi_coefficient is None
    assert a.redi_aht0 is None
    assert a.momentum_rk3 is False
    assert a.adaptive_implicit_vertadv is False
    assert a.min_levels == 1
    assert a.smag_cfl_safety is None
