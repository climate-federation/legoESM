"""``run_spectral_cbl.py`` must stamp its frames with the case it ran.

The driver integrates two different cases -- Nieuwstadt CBL_N91 and Wangara
Day 33 -- and writes the label into every recorded frame. That label is the
ONLY thing ``legoesm.training.les_reference`` has to confirm a reference
belongs to the case it is being built for, and the two are deliberately NOT
cross-aliased (accepting the alias would let a genuine Wangara directory
become the CBL reference).

``--case-label`` used to default to the literal ``"cbl"`` for BOTH cases. The
production Wangara run omitted the flag, so eight hours of a genuine Wangara
integration (theta0 = 277 K, diurnal flux, rotating) were stamped ``"cbl"``,
the reference loader refused them, and the eight-case turbulence-tuning
campaign died six seconds in. The default now follows ``--case``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = (Path(__file__).resolve().parents[2]
         / "scripts" / "run" / "run_spectral_cbl.py")


def _load():
    # les_record lives beside the driver and is imported by bare name.
    sys.path.insert(0, str(_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("run_spectral_cbl", _PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_the_label_map_covers_every_selectable_case(mod):
    """A new --case choice without a label entry would KeyError at run time,
    after the LES has been queued -- which is the worst place to find out."""
    import argparse
    p = argparse.ArgumentParser()
    # Read the real choices off the driver's own parser rather than repeating
    # them here, so adding a case makes this test fail instead of passing.
    src = _PATH.read_text()
    i = src.index('p.add_argument("--case", choices=')
    choices_literal = src[i + len('p.add_argument("--case", choices='):]
    choices_literal = choices_literal[:choices_literal.index("]") + 1]
    choices = eval(choices_literal)  # noqa: S307 - a list literal from our own source
    del p, argparse
    assert set(choices) == set(mod._CASE_LABELS), (
        f"--case offers {sorted(choices)} but _CASE_LABELS covers "
        f"{sorted(mod._CASE_LABELS)}")


def test_wangara_is_not_labelled_cbl(mod):
    assert mod._CASE_LABELS["wangara"] == "wangara"
    assert mod._CASE_LABELS["nieuwstadt"] == "cbl"


def test_the_labels_are_what_the_reference_loader_accepts(mod):
    """Non-vacuous: the strings must satisfy the CONSUMER, not just each other.

    A label that looks right but is not in the loader's alias table produces
    exactly the failure this test exists to prevent, one LES run later.
    """
    from legoesm.training.les_reference import _case_matches
    for case, label in mod._CASE_LABELS.items():
        expect = "cbl" if case == "nieuwstadt" else case
        assert _case_matches(label, expect), (
            f"the loader rejects label {label!r} for case {expect!r}")
    # And the two must NOT be interchangeable, or the guard is decoration.
    assert not _case_matches("cbl", "wangara")
    assert not _case_matches("wangara", "cbl")
