"""The round-2 ablation probe's source edit applies exactly once or refuses."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite"))
from cosci_ablation import ablated_source  # noqa: E402


def test_edit_applies_once():
    assert ablated_source("a = f(x)\nb = 2\n", ["a = f(x)|||a = 0.0 * f(x)"]) == "a = 0.0 * f(x)\nb = 2\n"


@pytest.mark.parametrize("src", ["b = 2\n", "a = f(x)\na = f(x)\n"])
def test_edit_refuses_missing_or_ambiguous_target(src):
    with pytest.raises(SystemExit):
        ablated_source(src, ["a = f(x)|||a = 0"])
