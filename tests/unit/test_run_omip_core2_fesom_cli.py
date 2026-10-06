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
    ["--mle"], ["--isf"],
    ["--bbl-adv"],
    ["--momentum-rk3"], ["--geothermal"],
    ["--gateway-transports"], ["--partial-cell"],
    ["--ice-categories", "2"], ["--ice-ridging"],
])
def test_physics_selectors_rejected(extra):
    args, p = _parse(extra)
    with pytest.raises(SystemExit, match="silently dropped"):
        validate_fesom_stage(args, p)


@pytest.mark.parametrize("extra", [
    ["--dm2dc"], ["--sw-rgb-chl"], ["--sw-rgb-chl", "--chl-file", "/c.nc"],
    ["--fesom-unforced"], ["--smoke"],
])
def test_b2b3_forcing_selectors_allowed(extra):
    """Stage B2+B3 wired the CORE-II forcing selectors through the fesom
    forced loop — the gate must now let them through."""
    args, p = _parse(extra)
    validate_fesom_stage(args, p)  # no raise


@pytest.mark.parametrize("extra", [
    # B4 — prognostic ice (free-drift, 1 category)
    ["--prognostic-sea-ice"],
    ["--prognostic-sea-ice", "--ice-init", "/ice.nc"],
    ["--prognostic-sea-ice", "--ice-ocean-heat-coeff", "0.005"],
    ["--prognostic-sea-ice", "--prognostic-ice-dynamics", "free_drift"],
    # B4 — runoff (node-adjacency coastal spread)
    ["--runoff"],
    ["--runoff", "--runoff-spread-passes", "4"],
    # B4 — SSS restoring, WATER-FLUX channel with a target
    ["--sss-restore", "--sss-restore-channel", "water_flux",
     "--sss-restore-file", "/sss.nc"],
    ["--sss-restore", "--sss-restore-channel", "water_flux",
     "--sss-restore-file", "/sss.nc", "--sss-restore-tau-days", "45.5",
     "--sss-restore-bound-mmday", "4.0",
     "--sss-restore-normalization", "live_s", "--sss-ice-gate-nemo"],
    ["--sss-restore", "--sss-restore-channel", "water_flux", "--woa-init"],
    # B4 — NEMO monthly / WOA initial condition
    ["--woa-init"],
    ["--woa-init", "--woa-t", "/t.nc", "--woa-s", "/s.nc"],
    ["--nemo-monthly-init", "/t.nc", "/s.nc"],
    ["--nemo-monthly-init", "/t.nc", "/s.nc", "--nemo-init-month", "7"],
])
def test_b4_selectors_allowed(extra):
    """Stage B4 wired prognostic ice + SSS restore (water_flux) + runoff +
    the NEMO-monthly/WOA IC — the gate must now let them through."""
    args, p = _parse(extra)
    validate_fesom_stage(args, p)  # no raise


def test_sss_restore_tracer_channel_accepted():
    """The tracer channel is wired on the fesom lane.

    It used to be refused, on the stated grounds that ``FesomOceanState.S``
    is a read-only facade.  ``.S`` is a read-only property, but the facade is
    a frozen dataclass and ``with_surface_salinity`` writes the inner state,
    so the refusal rested on a false premise.
    """
    args, p = _parse(["--sss-restore", "--sss-restore-file", "/sss.nc",
                      "--sss-restore-channel", "tracer"])
    validate_fesom_stage(args, p)  # no raise


def test_sss_restore_requires_an_explicit_channel():
    """An UNSET channel is refused rather than defaulted.

    The flag's global default is None, which means 'tracer'.  Accepting it
    here would hand a fesom card the non-NEMO virtual-salt form by omission,
    on a lane built for ORCA1 parity -- a scientific choice nobody made.  The
    lane refuses to pick.
    """
    args, p = _parse(["--sss-restore", "--sss-restore-file", "/sss.nc"])
    assert args.sss_restore_channel is None
    with pytest.raises(SystemExit, match="explicit"):
        validate_fesom_stage(args, p)


def test_sss_restore_needs_a_target():
    args, p = _parse(["--sss-restore", "--sss-restore-channel",
                      "water_flux"])
    with pytest.raises(SystemExit, match="target"):
        validate_fesom_stage(args, p)


@pytest.mark.parametrize("extra", [
    ["--prognostic-sea-ice"], ["--runoff"],
    ["--sss-restore", "--sss-restore-channel", "water_flux",
     "--sss-restore-file", "/sss.nc"],
])
def test_b4_selectors_rejected_under_unforced_smoke(extra):
    """--fesom-unforced consumes no forcing/coupling — accepting a B4
    coupling selector there would silently drop it."""
    args, p = _parse(["--fesom-unforced"] + extra)
    with pytest.raises(SystemExit, match="silently dropped"):
        validate_fesom_stage(args, p)


def test_ic_selectors_allowed_under_unforced_smoke():
    """IC selectors legitimately configure the B1 smoke's initial state —
    they must NOT be caught by the unforced-drop check."""
    args, p = _parse(["--fesom-unforced",
                      "--nemo-monthly-init", "/t.nc", "/s.nc"])
    validate_fesom_stage(args, p)  # no raise


def test_arbitrary_user_set_value_rejected():
    """The allowlist catches ANY non-default user setting, not only the
    flags a blocklist happened to enumerate (the fail-open GLM closed)."""
    args, p = _parse(["--bbl-adv"])
    with pytest.raises(SystemExit, match="bbl-adv"):
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


def test_iwm_accepted_on_fesom():
    """NEMO ORCA1 runs ln_zdfiwm=T; the FESOM closure bridge now splices
    zdfiwm additively (same single-owner kernel as the other two lanes), so
    the stage gate must accept it instead of rejecting it as unwired."""
    args, p = _parse(["--iwm", "--iwm-forcing-file", "/iwm.nc"])
    validate_fesom_stage(args, p)  # no raise
    assert args.iwm and args.iwm_forcing_file == "/iwm.nc"


def test_river_mouth_gate_accepted_on_fesom():
    """The river-mouth restoring gate is wired on the fesom lane (same
    contract as the host loop: no SSS restoring where the runoff map is
    wet) — it must pass the stage gate like the other lanes' cards."""
    args, p = _parse(["--runoff", "--river-mouth-restoring-gate"])
    validate_fesom_stage(args, p)  # no raise
    assert args.river_mouth_restoring_gate


def test_window_mean_flags_allowed():
    args, p = _parse(["--state-accumulate", "--mld-accumulate"])
    validate_fesom_stage(args, p)  # no raise


def test_vertical_momentum_nemo_advective_reaches_fesom_config():
    import inspect
    import scripts.run.run_omip_core2 as core2
    args, p = _parse(["--vertical-momentum-scheme", "nemo_advective"])
    validate_fesom_stage(args, p)  # allowlisted: no raise
    assert args.vertical_momentum_scheme == "nemo_advective"
    src = inspect.getsource(core2.main)
    i = src.index("build_fesom_ocean(")
    call = src[i:src.index("vmix_config=", i)]
    assert "vertical_momentum_scheme=args.vertical_momentum_scheme" in call
    assert 'args.grid == "fesom" and args.vertical_momentum_scheme == "nemo_advective"' in src
    b = inspect.getsource(core2.build_fesom_ocean)
    assert '"vertical_momentum_scheme": vertical_momentum_scheme' in b


def test_fesom_config_field_default_and_forwarding():
    import inspect
    from legoesm.ocean.dynamics.ocean_model_fesom import FesomOceanConfig, FesomOceanModel
    assert FesomOceanConfig().vertical_momentum_scheme == "fesom_flux"
    assert FesomOceanConfig(vertical_momentum_scheme="nemo_advective").vertical_momentum_scheme == "nemo_advective"
    assert "vertical_momentum=self.config.vertical_momentum_scheme" in inspect.getsource(FesomOceanModel.step)
