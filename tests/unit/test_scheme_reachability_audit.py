"""SCHEME reachability audit — implemented must imply selectable.

The repo already gates the REVERSE direction, thoroughly:
  * ``test_dispatch_hardening``      — a factory must RAISE on an unknown scheme
  * ``test_validate_strict_coverage``— a config must REJECT a typo'd scheme
Both answer "can a bad name get in?". Neither answers "can a GOOD name get in?"
-- so a scheme could be implemented, validated, oracle-pinned, and still be
unselectable from the driver a user actually runs. That is the gap this closes.

It bit every axis at once, because each was pinned by a point-fix test in
exactly ONE driver and so drifted in the other: ``run_coupled`` blocked
bechtold/tiedtke/emanuel/kain_fritsch/zhang_mcfarlane (tiedtke is run_amip's
convection DEFAULT and bechtold is what config/amip/amip_production.yaml pins --
a coupled run could select NEITHER of the two the atmosphere is actually run
with) and blocked clubb_lite/ysu; ``run_amip`` blocked mynn25.

The audit computes {schemes validate_strict ACCEPTS} - {schemes the driver
OFFERS} and asserts it EQUALS the shrink-only baseline. Both directions are
enforced: an unreachable scheme goes red until it is wired or consciously
excluded WITH A REASON, and a newly-wired scheme goes red until the baseline
shrinks. Deliberate exclusions stay listed rather than silently tolerated.

NOT everything narrower is drift -- see the baseline's docstring. A driver may
decline a scheme that does not FUNCTION in its configuration, and this file
pins those exclusions as intentional (``test_baseline_entries_are_still_excluded``)
so they cannot be quietly "fixed" into a broken offer.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest
from legoesm.driver.config import (
    VALID_CLOUD_SCHEMES,
    VALID_CONVECTION_SCHEMES,
    VALID_GWD,
    VALID_MICROPHYSICS,
    VALID_RADIATION,
    VALID_SURFACE_BULK,
    VALID_TURBULENCE,
    ExperimentConfig,
)

from tests.unit._scheme_reachability_baseline import UNREACHABLE_SCHEMES

# axis -> (canonical tuple, argparse dest)
_AXES = {
    "convection": (VALID_CONVECTION_SCHEMES, "convection"),
    "microphysics": (VALID_MICROPHYSICS, "microphysics"),
    "turbulence": (VALID_TURBULENCE, "turbulence"),
    "clouds": (VALID_CLOUD_SCHEMES, "clouds"),
    "radiation": (VALID_RADIATION, "radiation"),
    "gravity_wave_drag": (VALID_GWD, "gravity_wave_drag"),
    "surface_bulk_scheme": (VALID_SURFACE_BULK, "surface_bulk_scheme"),
}

_DRIVERS = {
    "run_amip": ("scripts/run/run_amip.py", "build_arg_parser"),
    "run_coupled": ("scripts/run/run_coupled.py", "build_parser"),
}


def _actions(driver: str) -> dict:
    path, builder = _DRIVERS[driver]
    name = f"_audit_{driver}"
    sys.argv = [name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {a.dest: a for a in getattr(mod, builder)()._actions}


@pytest.fixture(scope="module")
def parsers():
    return {d: _actions(d) for d in _DRIVERS}


def _compute_unreachable(parsers) -> dict:
    """{(driver, axis, scheme): 'offered?'} for every scheme a driver has a flag
    for but does not offer. A driver with NO flag on an axis is not audited on
    it -- absence of a flag is a different gap (task: missing CLI flags), and
    conflating them would make this audit unable to distinguish "flag exists but
    omits a scheme" (drift) from "no flag at all" (unimplemented surface)."""
    out = {}
    for driver, actions in parsers.items():
        for axis, (canonical, dest) in _AXES.items():
            action = actions.get(dest)
            if action is None or not getattr(action, "choices", None):
                continue
            for scheme in canonical:
                if scheme not in action.choices:
                    out[(driver, axis, scheme)] = None
    return out


def test_scheme_reachability_matches_baseline(parsers):
    """THE RATCHET. Computed unreachable set must EQUAL the baseline."""
    computed = set(_compute_unreachable(parsers))
    baseline = set(UNREACHABLE_SCHEMES)

    newly_unreachable = computed - baseline
    newly_reachable = baseline - computed

    assert not newly_unreachable, (
        "Scheme(s) accepted by ExperimentConfig.validate_strict but NOT "
        "offered by the driver's CLI -- implemented but unselectable:\n  "
        + "\n  ".join(f"{d}: --{a} {s}" for d, a, s in sorted(newly_unreachable))
        + "\n\nWire the scheme into the driver's choices= (derive it from the "
          "canonical VALID_* tuple in driver/config.py), or -- if the scheme "
          "genuinely does not FUNCTION there -- add it to "
          "tests/unit/_scheme_reachability_baseline.py WITH A REASON."
    )
    assert not newly_reachable, (
        "Baseline lists scheme(s) as unreachable that ARE now offered:\n  "
        + "\n  ".join(f"{d}: --{a} {s}" for d, a, s in sorted(newly_reachable))
        + "\nThe baseline is SHRINK-ONLY: delete these entries."
    )


#: (driver, axis) pairs that MUST expose a choices= flag. GROW-ONLY.
#:
#: _compute_unreachable deliberately SKIPS an axis with no flag (absence of a
#: flag is a different gap from a flag that omits a scheme). That skip is a
#: vacuity hole on its own: DELETING an entire --flag would silently drop the
#: axis from the audit and pass (codex). This pins the surface so a deletion
#: goes red and has to be argued for.
_REQUIRED_FLAGS = {
    ("run_amip", "convection"), ("run_amip", "microphysics"),
    ("run_amip", "turbulence"), ("run_amip", "clouds"),
    ("run_amip", "radiation"), ("run_amip", "surface_bulk_scheme"),
    ("run_coupled", "convection"), ("run_coupled", "microphysics"),
    ("run_coupled", "turbulence"), ("run_coupled", "clouds"),
    ("run_coupled", "radiation"), ("run_coupled", "surface_bulk_scheme"),
}


@pytest.mark.parametrize("key", sorted(_REQUIRED_FLAGS))
def test_the_audited_surface_still_exists(parsers, key):
    """Closes the audit's other vacuity hole: a deleted flag must not pass.

    Without this, removing --convection from run_coupled entirely would make
    the ratchet skip the axis and stay green -- an unselectable scheme is the
    very thing this file exists to catch, and deleting the flag makes EVERY
    scheme on that axis unselectable at once.
    """
    driver, axis = key
    action = parsers[driver].get(_AXES[axis][1])
    assert action is not None, (
        f"{driver} no longer has a --{axis} flag; the reachability audit would "
        f"silently SKIP this axis. If the removal is intentional, remove the "
        f"entry from _REQUIRED_FLAGS and say why."
    )
    assert getattr(action, "choices", None), (
        f"{driver} --{axis} lost its choices=; the audit skips choice-less "
        f"actions, so every scheme on this axis would go unaudited. (GWD is "
        f"the one legitimate exception -- it has its own pins below.)"
    )


def test_the_audit_would_catch_a_deleted_flag(parsers):
    """Non-vacuity for the guard above (codex: the existing regression test
    only removes ONE choice, so a whole-flag deletion slipped through)."""
    mutated = {d: dict(a) for d, a in parsers.items()}
    del mutated["run_coupled"]["convection"]
    # the ratchet itself stays green -- that IS the hole...
    assert not any(k[0] == "run_coupled" and k[1] == "convection"
                   for k in _compute_unreachable(mutated)), (
        "precondition: _compute_unreachable skips a missing flag")
    # ...so _REQUIRED_FLAGS is what must catch it.
    assert mutated["run_coupled"].get("convection") is None
    assert ("run_coupled", "convection") in _REQUIRED_FLAGS, (
        "the deleted flag is not pinned by _REQUIRED_FLAGS -- nothing would "
        "notice its removal"
    )


def test_every_baseline_entry_has_a_reason():
    """An exclusion without a reason is indistinguishable from an oversight."""
    for key, reason in UNREACHABLE_SCHEMES.items():
        assert isinstance(reason, str) and len(reason) > 40, (
            f"{key} needs a real reason explaining why the scheme does not "
            f"function in that driver, got {reason!r}"
        )


@pytest.mark.parametrize("key", sorted(UNREACHABLE_SCHEMES))
def test_baseline_entries_are_still_excluded(parsers, key):
    """Self-test: the baseline must be about REAL exclusions.

    Guards the audit against vacuity from the other side -- if a baseline entry
    named a scheme the driver actually offers, or an axis/driver that does not
    exist, the ratchet would silently be checking nothing.
    """
    driver, axis, scheme = key
    assert driver in _DRIVERS, f"unknown driver {driver!r}"
    assert axis in _AXES, f"unknown axis {axis!r}"
    action = parsers[driver].get(_AXES[axis][1])
    assert action is not None, f"{driver} has no --{axis} flag"
    assert scheme not in action.choices, (
        f"{driver} --{axis} NOW offers {scheme!r}; the baseline is shrink-only "
        f"-- delete this entry."
    )


def test_the_audit_would_catch_a_regression(parsers):
    """Non-vacuity: prove the computation flags a real gap.

    Simulates the historical bug (run_coupled omitting bechtold) by removing it
    from the parsed choices, and asserts the audit notices. Without this, a
    broken _compute_unreachable would pass silently forever.
    """
    import copy
    mutated = {d: dict(a) for d, a in parsers.items()}
    action = copy.copy(mutated["run_coupled"]["convection"])
    action.choices = [c for c in action.choices if c != "bechtold"]
    mutated["run_coupled"]["convection"] = action

    computed = _compute_unreachable(mutated)
    assert ("run_coupled", "convection", "bechtold") in computed, (
        "the audit failed to detect a scheme removed from a driver's choices"
    )


# --- GWD: the one axis `choices=` cannot express -------------------------

@pytest.mark.parametrize("spec", ["hines", "none", "hines+mcfarlane",
                                  "mcfarlane+prognostic_spectral"])
@pytest.mark.parametrize("driver", sorted(_DRIVERS))
def test_gwd_composites_are_selectable_from_every_driver(parsers, driver, spec):
    """GWD accepts a '+'-joined COMPOSITION whose source tendencies are summed
    (#834) -- orographic and non-orographic drag parameterize distinct wave
    populations and run together in CMIP-class GCMs.

    argparse `choices=` cannot express that, which split the drivers in OPPOSITE
    ways: run_coupled kept choices= and so REJECTED every composite (making #834
    unreachable there), while run_amip dropped choices= and so had NO cli typo
    rejection at all. Both now share driver.config.parse_gwd_spec.

    This axis is invisible to the choices=-based audit above (an action with no
    choices is skipped), so it needs its own pin.
    """
    _, builder = _DRIVERS[driver]
    _reparse(driver, builder, ["--gravity-wave-drag", spec])
    ExperimentConfig(gravity_wave_drag=spec).validate_strict()


@pytest.mark.parametrize("bad", [
    "garbage", "hines+garbage",
    # composites that are membership-valid but SEMANTICALLY invalid; a
    # membership-only CLI check accepted these while validate_strict rejected
    # them, so the flag advertised a spec the config then refused (codex).
    "none+hines", "e3sm_cam+hines", "ml_emulator+hines", "hines+hines",
])
@pytest.mark.parametrize("driver", sorted(_DRIVERS))
def test_gwd_typos_are_rejected_at_the_cli(parsers, driver, bad):
    """The other half: dropping choices= must not cost typo rejection."""
    _, builder = _DRIVERS[driver]
    with pytest.raises(SystemExit):
        _reparse(driver, builder, ["--gravity-wave-drag", bad])


@pytest.mark.parametrize("spec", [
    "hines", "none", "hines+mcfarlane", "mcfarlane+prognostic_spectral",
    "garbage", "hines+garbage", "none+hines", "e3sm_cam+hines",
    "ml_emulator+hines", "hines+hines",
])
def test_gwd_cli_and_validate_strict_never_disagree(spec):
    """The CLI validator must be the config validator, not a copy of it.

    parse_gwd_spec delegates to ExperimentConfig.validate_strict for the
    SEMANTICS (it only pre-checks membership to give a better message), so the
    two cannot drift -- the same reasoning as resolving the effective surface
    config through the production path in driver/air_sea_consistency.py.
    """
    import argparse

    from legoesm.driver.config import parse_gwd_spec

    try:
        parse_gwd_spec(spec)
        cli_ok = True
    except argparse.ArgumentTypeError:
        cli_ok = False
    try:
        ExperimentConfig(gravity_wave_drag=spec).validate_strict()
        cfg_ok = True
    except ValueError:
        cfg_ok = False
    assert cli_ok == cfg_ok, (
        f"--gravity-wave-drag {spec!r}: CLI {'accepts' if cli_ok else 'rejects'} "
        f"but validate_strict {'accepts' if cfg_ok else 'rejects'}"
    )


def _reparse(driver: str, builder: str, argv: list):
    path = _DRIVERS[driver][0]
    name = f"_gwd_{driver}"
    sys.argv = [name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, builder)().parse_args(argv)


# --- no phantoms: everything offered must actually be accepted --------------

@pytest.mark.parametrize("driver", sorted(_DRIVERS))
def test_every_offered_scheme_passes_validate_strict(parsers, driver):
    """The inverse error: offering a value the config REJECTS.

    That is what ``--surface-bulk-scheme most`` was -- it parsed fine and then
    died at driver construction, contradicting the flag's own NOTE.
    """
    actions = parsers[driver]
    for axis, (_canonical, dest) in _AXES.items():
        action = actions.get(dest)
        if action is None or not getattr(action, "choices", None):
            continue
        for scheme in action.choices:
            kwargs = {axis if axis != "clouds" else "cloud_scheme": scheme}
            # MOST schemes carry a documented CROSS-FIELD contract (they need
            # turbulence != "none" so the atmosphere runs the same bulk-flux
            # algorithm as the ocean tile), so satisfy it rather than trip it.
            if axis == "surface_bulk_scheme" and scheme != "constant":
                kwargs["turbulence"] = "louis"
            try:
                ExperimentConfig(**kwargs).validate_strict()
            except ValueError as e:  # pragma: no cover - failure path
                pytest.fail(
                    f"{driver} offers --{dest} {scheme!r} but validate_strict "
                    f"rejects it (a phantom choice): {e}"
                )
