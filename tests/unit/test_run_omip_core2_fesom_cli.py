"""``--grid fesom`` stage gate: ALLOWLIST semantics against the REAL parser.

GLM review 2026-09-01: a blocklist is fail-open as the driver grows — every
USER-SET selector not on the wired allowlist must be rejected, including
flags added to the driver AFTER this test was written (that is the point of
the inversion, and why these tests drive the real parser, not a stub).
"""
from __future__ import annotations

import pytest

from scripts.run.run_omip_core2 import (
    _FESOM_WIRED_DESTS,
    _build_arg_parser,
    validate_fesom_stage,
)

BASE = ["--grid", "fesom", "--fesom-mesh-dir", "/some/mesh",
        "--output", "/some/out", "--no-emp"]


def _parse(extra=()):
    p = _build_arg_parser()
    return p.parse_args(BASE + list(extra)), p


def test_mesh_dir_required():
    p = _build_arg_parser()
    args = p.parse_args(["--grid", "fesom", "--output", "/o", "--no-emp"])
    with pytest.raises(SystemExit, match="fesom-mesh-dir"):
        validate_fesom_stage(args, p)


def test_clean_args_pass():
    args, p = _parse()
    validate_fesom_stage(args, p)  # no raise


@pytest.mark.parametrize("extra", [
    ["--dm2dc"], ["--iwm"], ["--runoff"], ["--mle"], ["--isf"],
    ["--bbl-adv"], ["--prognostic-sea-ice"], ["--woa-init"],
    ["--momentum-rk3"], ["--geothermal"], ["--sw-rgb-chl"],
    ["--sss-restore"], ["--gateway-transports"], ["--partial-cell"],
])
def test_physics_selectors_rejected(extra):
    args, p = _parse(extra)
    with pytest.raises(SystemExit, match="silently dropped"):
        validate_fesom_stage(args, p)


def test_arbitrary_user_set_value_rejected():
    """The allowlist catches ANY non-default user setting, not only the
    flags a blocklist happened to enumerate (the fail-open GLM closed)."""
    args, p = _parse(["--sss-restore-tau-days", "30"])
    with pytest.raises(SystemExit, match="sss-restore-tau-days"):
        validate_fesom_stage(args, p)


def test_no_emp_optout_allowed():
    """Turning OFF a default-on lever the lane cannot run is exactly what
    the gate demands — must not be rejected."""
    args, p = _parse()
    assert args.emp_freshwater is False
    validate_fesom_stage(args, p)


def test_allowlist_is_tight():
    """Every allowlisted dest exists on the parser (a typo here would
    silently reopen the gate for the misspelled flag)."""
    p = _build_arg_parser()
    defaults = {a.dest for a in p._actions}
    missing = _FESOM_WIRED_DESTS - defaults
    assert not missing, f"allowlisted dests not on the parser: {missing}"


def test_grid_choice_registered():
    p = _build_arg_parser()
    args = p.parse_args(BASE)
    assert args.grid == "fesom"
