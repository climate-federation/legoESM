"""Gap 10: the dynamics equation of state is selectable, and TEOS-10 is refused.

Before this change the OMIP runner set no EOS at all, so every lane silently
ran the ``LatLonCGridOceanConfig`` default ``"wright"`` while the SAME runner
already selected a TEOS-10 form for the TKE N2 (ORCA1 runs ln_teos10=.true.).

The refusal of ``nemo_teos10`` is the substantive assertion here. NEMO's
TEOS-10 branch consumes Conservative Temperature and ABSOLUTE salinity
(eosbn2.F90:1924 sets l_useCT=.TRUE.; :218 documents "TEOS10: SA ... g/kg")
and converts nothing -- its input files are pre-converted offline. Our tracers
and WOA initial condition are potential temperature and PRACTICAL salinity, so
accepting the flag would feed SP into an SA polynomial.
"""
from __future__ import annotations

import argparse
import ast
import pathlib

import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_runner_eos", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- the refusal ------------------------------------------------------------

def test_teos10_is_refused_by_name_not_as_unknown():
    """It must refuse with the UNIT reason, not fall through to 'unknown'.

    A bare "unknown value" message would be indistinguishable from a typo and
    would invite someone to add it to the tuple without doing the conversion.
    """
    m = _mod()
    with pytest.raises(SystemExit) as e:
        m._validate_omip_eos("nemo_teos10")
    msg = str(e.value)
    assert "ABSOLUTE" in msg and "PRACTICAL" in msg, msg
    assert "unknown" not in msg.lower(), msg


def test_unesco80_is_refused_because_it_is_an_in_situ_polynomial():
    """Both reviewers flagged this and a check value CONFIRMED it.

    UNESCO 1980 takes IN-SITU temperature; our tracers are potential
    temperature. Our unesco80_eos reproduces the published in-situ check
    value rho(35, 25, 10000 dbar) = 1062.538 to 2e-4 kg/m3 (job 9715403), so
    it is not a theta refit and selecting it would be a unit error of the
    same class as TEOS-10.
    """
    m = _mod()
    with pytest.raises(SystemExit) as e:
        m._validate_omip_eos("unesco80")
    msg = str(e.value)
    assert "IN-SITU" in msg and "POTENTIAL" in msg, msg
    assert "unknown" not in msg.lower(), msg


def test_refused_forms_are_absent_from_the_accepted_tuple():
    """A name cannot be both accepted and refused -- that would make the
    refusal unreachable via the CLI while still passing a direct call."""
    m = _mod()
    for name in m._OMIP_EOS_REFUSED:
        assert name not in m._OMIP_EOS_FORMS, name
    assert "wright" in m._OMIP_EOS_FORMS


def test_every_refusal_explains_a_unit_reason():
    """Guards against a future name being refused with an empty rationale."""
    for name, reason in _mod()._OMIP_EOS_REFUSED.items():
        assert len(reason) > 80, f"{name}: refusal reason too thin"
        assert "salinity" in reason.lower() or "temperature" in reason.lower()


def test_unknown_eos_raises():
    m = _mod()
    with pytest.raises(SystemExit, match="unknown"):
        m._validate_omip_eos("not_an_eos")


def test_none_is_accepted_and_means_leave_the_default_alone():
    """The whole point of the default: an unset flag must change nothing."""
    assert _mod()._validate_omip_eos(None) is None


@pytest.mark.parametrize("name", ("wright", "nemo_eos80", "nemo_seos"))
def test_supported_forms_pass(name):
    """These three take potential temperature and practical salinity, which is
    what our tracers carry. NEMO's own EOS-80 check value is quoted for
    POTENTIAL temperature (eosbn2.F90:226), which is why it stays accepted."""
    assert _mod()._validate_omip_eos(name) is None


# --- the CLI ----------------------------------------------------------------

def test_flag_round_trips_and_defaults_to_none():
    p = _mod()._build_arg_parser()
    assert p.parse_args([]).eos is None
    assert p.parse_args(["--eos", "nemo_eos80"]).eos == "nemo_eos80"


@pytest.mark.parametrize("name", ("nemo_teos10", "unesco80"))
def test_argparse_rejects_every_refused_form(name):
    """choices= must not advertise a value the validator will refuse."""
    p = _mod()._build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--eos", name])


# --- the wiring -------------------------------------------------------------
# A flag that parses but is never forwarded is inert. These read the source
# rather than running the model, because build_tripole needs a mesh.

def _runner_tree():
    return ast.parse(RUNNER.read_text())


def test_build_tripole_accepts_eos():
    for node in ast.walk(_runner_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == "build_tripole":
            names = [a.arg for a in node.args.args + node.args.kwonlyargs]
            assert "eos" in names, "build_tripole lost its eos parameter"
            return
    pytest.fail("build_tripole not found")


def test_the_flag_is_forwarded_to_build_tripole():
    """eos=args.eos must appear in a build_tripole(...) call."""
    found = False
    for node in ast.walk(_runner_tree()):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if getattr(fn, "id", None) != "build_tripole":
            continue
        for kw in node.keywords:
            if (kw.arg == "eos" and isinstance(kw.value, ast.Attribute)
                    and kw.value.attr == "eos"):
                found = True
    assert found, "build_tripole is never called with eos=args.eos"


def test_build_tripole_actually_applies_it_to_the_override_dict():
    """Forwarding is not enough: the value must reach the config override.

    Guards against a parameter that is accepted and then dropped -- the exact
    failure mode that left the three GM operator flags inert on other grids.
    """
    src = RUNNER.read_text()
    assert '_ovr["eos"] = eos' in src, "eos never reaches the config override"
    assert "_validate_omip_eos(eos)" in src, "build_tripole skips validation"
