"""Direct import/CLI-surface tests for the committed SI3 NaN-fix proofs."""

import argparse
import importlib.util
from pathlib import Path

import pytest


_SCRIPTS = (
    Path(__file__).parents[3]
    / "scripts"
    / "validate"
    / "ocean_fidelity"
    / "testcases"
    / "si3_nanfix"
)


def test_verify_proof1_active_bits_import_and_parser(monkeypatch):
    path = _SCRIPTS / "verify_proof1_active_bits.py"
    spec = importlib.util.spec_from_file_location("si3_nanfix_proof1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ParserReached(Exception):
        pass

    def stop_after_parser(parser, args=None, namespace=None):
        del args, namespace
        case = next(action for action in parser._actions if action.dest == "case")
        assert tuple(case.choices) == ("o12_5cat", "d194", "probe_tile")
        assert case.required
        raise ParserReached

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_after_parser)
    with pytest.raises(ParserReached):
        module.main()


def test_verify_proof2_orca2_step1_import_only():
    path = _SCRIPTS / "verify_proof2_orca2_step1.py"
    spec = importlib.util.spec_from_file_location("si3_nanfix_proof2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)
    assert len(module.NINE_FIELDS) == 9
