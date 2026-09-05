"""Focused controls for the round-17 coupled momentum owner."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_c1d_omip_l3_slab_ocean_card,
)
from scripts.validate.ocean_fidelity.testcases.nemo_rung36_round17_momentum_gate import (
    GateError,
    _records,
    evaluate,
)

ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/"
    "c1d_omip_l3_coupled10m_r17_oracle_b")


def test_slab_card_selects_resolved_quadratic_bottom_drag() -> None:
    bottom = build_c1d_omip_l3_slab_ocean_card().recipe.model_config.bottom_drag
    assert bottom.bottom_drag_scheme == "nemo_quadratic"
    assert bottom.bottom_drag_cd0 == 1.0e-3
    assert bottom.bottom_drag_ke0 == 2.5e-3


def test_round17_owner_controls_are_private() -> None:
    hooks = _NEMOWSRK3TestHooks()
    assert hooks.barotropic_kmm_seed_weight == 1.0
    assert hooks.linearize_quadratic_bottom_drag is False
    public_fields = build_c1d_omip_l3_slab_ocean_card().recipe.model_config._fields
    assert "barotropic_kmm_seed_weight" not in public_fields
    assert "linearize_quadratic_bottom_drag" not in public_fields


def test_round17_schema_rejects_false_derived_count(tmp_path: Path) -> None:
    path = tmp_path / "bad_spg_statement.bin"
    path.write_bytes(
        b"NEMO_L3SPGS_001 "
        + struct.pack("=9i", 2, 3, 3, 1, 1, 1, 946, 20, 64)
        + np.zeros(21, dtype=np.float64).tobytes())
    with pytest.raises(ValueError, match="untrue header"):
        _records(path, b"NEMO_L3SPGS_001 ", "9i", 21)


def test_round17_missing_stream_raises_named_gate_error(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="required stream"):
        evaluate(tmp_path, nstep=2)


@pytest.mark.skipif(not ROOT.is_dir(), reason="pinned round-17 oracle unavailable")
def test_round17_kt3_owner_and_controls_bind() -> None:
    report = evaluate(ROOT, nstep=2)
    assert report["verdict"] == "AT_BAR"
    assert report["first_over_bar"] is None
    assert all(stat["bit_identical"] == 2
               for stat in report["trajectory_aggregates"].values())
    arms = report["one_variable_arms"]
    assert arms["seed_weight_0"]["first_over_bar"] is not None
    assert arms["seed_weight_0p5"]["first_over_bar"] is not None
    assert arms["seed_weight_1"]["first_over_bar"] is None
    assert arms["linear_drag"]["first_over_bar"] is not None


@pytest.mark.skipif(not ROOT.is_dir(), reason="pinned round-17 oracle unavailable")
def test_round17_row_level_plant_binds() -> None:
    report = evaluate(ROOT, nstep=2, plant=True)
    assert report["verdict"] == "STOP_FIRST_OVER_BAR"
    assert report["plant_binding"] == {
        "target": "kt1-2.SSH_SUBSTEP.ua_statement[kt2,jn1]",
        "red": True,
    }
