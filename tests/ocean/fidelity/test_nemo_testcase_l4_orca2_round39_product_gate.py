"""Binding tests for the ORCA2 round-39 product wrapper."""

from types import SimpleNamespace

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round39_product_gate as gate,
)


def _parent(*, e3_max=0.027675060932892848):
    return {
        "first_non_bit_rank1_disputed_source": {
            "boundary": "e3", "differing_cells": 1754,
            "absolute_max": e3_max,
        },
        "one_variable_depth_arms": {
            "parent_depth_vs_record": {"absolute_max": 7.356587026022005e-18},
            "recorded_e3_only_vs_record": {"absolute_max": 9.423577925168289e-11},
            "recorded_rhs_only_vs_record": {"absolute_max": 4.235164736271502e-22},
        },
        "compiled_product_walk": {"per_level_product": {"bit_exact": False}},
        "scientific_plant": {"requested": False, "fires": None},
        "citations": {},
    }


def test_wrapper_requires_exact_round38_reproduction(monkeypatch):
    monkeypatch.setattr(gate.round38, "run", lambda *args, **kwargs: _parent())
    result = gate.run(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), None)
    assert result["round38_reproduced"] is True


def test_wrapper_rejects_a_moved_parent(monkeypatch):
    monkeypatch.setattr(
        gate.round38, "run", lambda *args, **kwargs: _parent(e3_max=0.0))
    with pytest.raises(gate.round38.GateError, match="did not reproduce"):
        gate.run(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), None)
