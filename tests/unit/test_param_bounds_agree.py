"""One parameter, one range. Two disagreeing ranges mean neither is the range.

A tunable declares bounds twice: in its scheme's ``__param_spec__``, which the
registry-qualified ``--params`` path range-checks, and in the driver's
``validate_strict``, which the YAML and CLI path range-checks. When they
disagree, a value can be legal through one door and illegal through the other —
and the door a campaign actually uses is whichever the launcher happens to
take.

FOUND BY: a proposal to run ``cloud_q_c_diagnostic`` at 5e-6, described as
"an order of magnitude below its own declared lower bound". It is below the
SPEC's lower bound of 5e-5 and comfortably inside the DRIVER's, which starts at
1e-6. So the bound quoted in the discussion was not the bound that would have
guarded the run.

This is the hidden-choice class in CLAUDE.md: a scientific limit that exists in
two places and is enforced in one.

SHRINK-ONLY. The list below is debt, not permission. Adding an entry means
shipping a parameter whose legal range depends on which door you came through.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _ROOT / "packages" / "coupler" / "legoesm" / "driver" / "config.py"
_PACKAGES = _ROOT / "packages"

# name -> (driver range, spec range) as they stand today. Each needs the two
# declarations reconciled; until then the value that is legal depends on the
# entry point.  SHRINK-ONLY.
_DISAGREE: dict[str, str] = {
    "bechtold_delta_deep":
        "driver (2.475e-05, 2.25e-04) vs spec (3.3e-05, 3.0e-04)",
    "bechtold_downdraft_alpha":
        "driver (0, 0.9) vs spec (0.00495, 0.045) — the driver admits twenty "
        "times the spec's upper bound",
    "bechtold_epsilon_deep":
        "driver (5.775e-04, 3.5e-03) vs spec (7.0e-04, 4.2e-03)",
}

_TOL = 1e-12


def _driver_bounds() -> dict[str, tuple[float, float]]:
    """``("name", lo, hi)`` triples from the driver's range checks."""
    out = {}
    for m in re.finditer(
            r'\(\s*"([a-z_0-9]+)"\s*,\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*\)',
            _DRIVER.read_text()):
        try:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        except ValueError:                       # not a numeric triple
            continue
    return out


def _spec_bounds() -> dict[str, list[tuple[tuple[float, float], str]]]:
    """``name -> [((lo, hi), file), ...]`` from EVERY ``__param_spec__``, in
    sorted file order.  All declarations are kept: a name two specs bound
    differently is a disagreement whichever file is read last."""
    out: dict[str, list] = {}
    for path in sorted(_PACKAGES.rglob("*.py")):
        text = path.read_text(errors="replace")
        if "__param_spec__" not in text:
            continue
        for m in re.finditer(
                r'"([a-z_0-9]+)"\s*:\s*\{[^}]*"bounds"\s*:\s*'
                r'\(\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*\)', text):
            try:
                b = (float(m.group(2)), float(m.group(3)))
            except ValueError:
                continue
            out.setdefault(m.group(1), []).append(
                (b, str(path.relative_to(_ROOT))))
    return out


def _disagreements(driver=None, spec=None) -> dict[str, str]:
    driver = _driver_bounds() if driver is None else driver
    spec = _spec_bounds() if spec is None else spec
    bad = {}
    for name, (lo, hi) in driver.items():
        for sname, decls in spec.items():
            # the driver prefixes the scheme onto the spec's field name
            if not (name == sname or name.endswith("_" + sname)
                    or name == "cloud_" + sname):
                continue
            for (slo, shi), where in decls:
                if (abs(lo - slo) > _TOL * max(1.0, abs(slo))
                        or abs(hi - shi) > _TOL * max(1.0, abs(shi))):
                    bad[name] = (f"driver ({lo:g}, {hi:g}) vs spec "
                                 f"({slo:g}, {shi:g}) in {where}")
    return bad


def test_a_second_spec_declaring_another_range_is_caught():
    """Non-vacuity: one agreeing and one diverging declaration of the same
    name is a disagreement, in either order."""
    driver = {"bechtold_dnoprc": (7.5e-5, 1.2e-3)}
    good, bad = ((7.5e-5, 1.2e-3), "a.py"), ((1.0e-4, 1.0e-3), "b.py")
    for decls in ([good, bad], [bad, good]):
        assert "bechtold_dnoprc" in _disagreements(driver, {"dnoprc": decls})
    assert not _disagreements(driver, {"dnoprc": [good, good]})


def test_the_comparison_finds_both_declarations():
    """Non-vacuity: if either side parses to nothing, everything 'agrees'."""
    driver, spec = _driver_bounds(), _spec_bounds()
    assert len(driver) >= 20, f"only {len(driver)} driver ranges parsed"
    assert len(spec) >= 100, f"only {len(spec)} spec ranges parsed"


def test_no_new_parameter_declares_two_different_ranges():
    new = sorted(set(_disagreements()) - set(_DISAGREE))
    assert not new, (
        f"{new} declare different legal ranges in the driver and in their "
        f"scheme's parameter spec, so whether a value is accepted depends on "
        f"which entry point set it. Reconcile the two, or add an entry to "
        f"_DISAGREE naming the reconciliation owed.")


def test_the_disagreement_list_only_shrinks():
    fixed = sorted(set(_DISAGREE) - set(_disagreements()))
    assert not fixed, (
        f"{fixed} now agree — delete their _DISAGREE entries. A stale entry "
        f"is a waiver covering nothing.")


@pytest.mark.parametrize("name", sorted(_DISAGREE))
def test_each_known_disagreement_is_still_real(name):
    """An entry that stops describing the code is a lie in a list."""
    assert name in _disagreements(), (
        f"{name} is listed as disagreeing but does not; remove it")
