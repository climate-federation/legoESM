"""`--freshwater-closure` is registered on the real parser and reaches config.

Repo rule: a new user-tunable field needs a same-PR CLI flag AND a round-trip
test.  `run_omip_core2` builds its parser inside `main()` (no module-level
parse_args), so registration is proven the way the sibling SPMD CLI test does
it -- drive the REAL entry point and assert we get the DOWNSTREAM error, not
argparse's "unrecognized arguments" (which would mean the flag is not
registered at all).

Why this flag matters enough to pin: `real_freshwater` removes the virtual salt
flux that was adding +10.109 psu.m of spurious Arctic salt over 60 days
(docs/dev-notes/ocean_real_freshwater_design.md).  An unregistered or
mis-wired flag would silently leave the defect in place while the run log
looked correct.
"""
from __future__ import annotations

import sys

import pytest


def _argparse_rejected(err: str) -> bool:
    """True if argparse itself rejected the flag (exit code 2 / usage text)."""
    return "unrecognized arguments" in err or err == "2"


def test_flag_is_registered_on_the_real_parser(monkeypatch, capsys):
    """Prove registration via --help, which argparse serves and exits(0) on
    WITHOUT running anything.

    Do NOT drive main() with a VALID choice to test this: the flag has no early
    validator, so main() proceeds into a real simulation and the test hangs
    (that mistake timed out a 30-minute node slot).  --help is both correct and
    instant.
    """
    import scripts.run.run_omip_core2 as R

    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--help"])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--freshwater-closure" in out, (
        "--freshwater-closure is not registered on the parser")
    for choice in ("none", "virtual_salt_flux", "real_freshwater"):
        assert choice in out, f"choice {choice!r} missing from --help"


def test_invalid_choice_is_rejected_by_argparse(monkeypatch):
    # Dispatch hardening at the CLI layer: a typo must not fall through to a
    # default closure.  argparse `choices` should reject it outright.
    import scripts.run.run_omip_core2 as R

    monkeypatch.setattr(sys, "argv", [
        "run_omip_core2.py", "--freshwater-closure", "real_freshwter"])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert str(e.value.code) == "2", (
        "an invalid --freshwater-closure must be rejected by argparse "
        f"(exit 2), got {e.value.code!r}")


def test_unset_flag_leaves_config_default_untouched():
    """The default is None -> "do not override", so every pre-existing run
    stays bit-identical.  This is the property that protects the entire
    existing OMIP corpus from the new mode."""
    import inspect

    import scripts.run.run_omip_core2 as R

    src = inspect.getsource(R)
    assert 'if args.freshwater_closure is not None:' in src, (
        "the flag must only override the config when EXPLICITLY set; an "
        "unconditional assignment would change every existing run")
    i = src.index('p.add_argument("--freshwater-closure"')
    assert "default=None" in src[i:i + 300], (
        "--freshwater-closure must default to None (leave config alone)")
