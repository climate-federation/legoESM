"""Gap 11: the sea-ice exchange coefficients are selectable as a named set.

Our SeaIceConfig and ORCA1 disagree on all four coefficients. The oracle
values are read from the RUN'S namelist_cfg, not namelist_ref -- the reference
file carries 1.4e-3 for the three air-ice coefficients and the cfg overrides
them to 1.0e-3, so quoting the reference would have been wrong by 40%.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_runner_ice", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cfg():
    from legoesm.ice.config import SeaIceConfig
    return SeaIceConfig()


def test_unset_changes_nothing():
    """The default must be byte-identical to the current production config."""
    m, c = _mod(), _cfg()
    assert m.apply_ice_exchange_set(c, None) is c


def test_nemo_si3_matches_the_oracle_namelist_exactly():
    """These four numbers are the whole point; pin them to the oracle."""
    out = _mod().apply_ice_exchange_set(_cfg(), "nemo_si3")
    assert out.Cd_ice == pytest.approx(1.0e-3)      # rn_Cd_ia
    assert out.Ch_ice == pytest.approx(1.0e-3)      # rn_Ch_ia
    assert out.drag_atm == pytest.approx(1.0e-3)    # rn_Cd_ia
    assert out.drag_ocean == pytest.approx(5.0e-3)  # rn_Cd_io


def test_it_actually_moves_every_coefficient():
    """Guards against a set that silently matches our defaults: if any field
    were left alone the 'faithful' arm would differ from the oracle in that
    field while claiming to match."""
    before, after = _cfg(), _mod().apply_ice_exchange_set(_cfg(), "nemo_si3")
    for f in ("Cd_ice", "Ch_ice", "drag_atm", "drag_ocean"):
        assert getattr(before, f) != getattr(after, f), f


def test_nothing_else_is_touched():
    """A named set must change ONLY its four coefficients -- one variable."""
    before, after = _cfg(), _mod().apply_ice_exchange_set(_cfg(), "nemo_si3")
    changed = {f for f in before._fields
               if getattr(before, f) is not getattr(after, f)
               and getattr(before, f) != getattr(after, f)}
    assert changed == {"Cd_ice", "Ch_ice", "drag_atm", "drag_ocean"}, changed


def test_unknown_set_raises():
    with pytest.raises(SystemExit, match="unknown --ice-exchange"):
        _mod().apply_ice_exchange_set(_cfg(), "nemo_si4")


def test_flag_round_trips_and_defaults_to_none():
    p = _mod()._build_arg_parser()
    assert p.parse_args([]).ice_exchange is None
    assert p.parse_args(["--ice-exchange", "nemo_si3"]).ice_exchange == "nemo_si3"


def test_the_set_is_applied_on_BOTH_ice_lanes():
    """The runner builds SeaIceConfig twice (the FESOM lane and the host lane).
    A flag wired into only one of them is the silent-discard failure this repo
    has hit before.

    Counted over the AST, not the raw text: codex's review found that a
    substring count passes even when BOTH calls are commented out, since a
    comment still contains the string. That check accepted inert wiring.
    """
    calls = [n for n in ast.walk(ast.parse(RUNNER.read_text()))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "apply_ice_exchange_set"]
    assert len(calls) == 2, f"expected both ice lanes wired, found {len(calls)}"
    for c in calls:
        assert any(getattr(a, "attr", None) == "ice_exchange" for a in c.args), \
            "a lane calls the helper without passing args.ice_exchange"


def test_sublimation_coefficient_is_covered_through_the_shared_Ch():
    """ORCA1 sets rn_Ce_ia=1.0e-3 and we have no dedicated latent field, but
    our bulk formula drives the latent flux with the SAME Ch as the sensible
    one, and the oracle sets both to the same value. Pin the mechanism: if
    bulk_flux ever gives latent its own coefficient, this preset stops
    covering rn_Ce_ia and this test should be the thing that notices.
    """
    bulk = (pathlib.Path(__file__).resolve().parents[2] / "packages" / "core"
            / "legoesm" / "core" / "bulk_flux.py").read_text()
    assert "lhflx = rho * _L * Ch * wind_speed" in bulk, (
        "latent flux no longer uses the shared Ch; rn_Ce_ia coverage is stale")
    assert "rn_Ce_ia" in RUNNER.read_text()


def test_helper_is_reachable_as_a_module_level_function():
    tree = ast.parse(RUNNER.read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "apply_ice_exchange_set" in names
