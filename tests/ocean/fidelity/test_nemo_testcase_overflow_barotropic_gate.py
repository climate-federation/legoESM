"""Direct fail-closed controls for the OVERFLOW 19-frame gate."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_overflow_barotropic_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_overflow_barotropic_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write_zero_trace(path: Path) -> None:
    nx, ny, ncycle, nfields, bits = gate.EXPECTED
    full = np.zeros(nx * ny, dtype=np.float64).tobytes()
    a2d = np.zeros((nx - 4) * (ny - 4), dtype=np.float64).tobytes()
    with path.open("wb") as handle:
        handle.write(b"NEMO_L1_OVBT_1  ")
        handle.write(struct.pack("=11i", 1, 1, 1, 1, 1, 3, ncycle, nx, ny, nfields, bits))
        for jn in range(1, ncycle + 1):
            handle.write(struct.pack("=i", jn))
            for name in gate.FIELDS:
                handle.write(a2d if name in ("slow_u", "slow_v") else full)


def test_oracle_reader_inventory_is_exact_and_rejects_file_side_extra(tmp_path):
    path = tmp_path / "trace.bin"
    _write_zero_trace(path)
    report = gate.read_oracle_trace(path)
    assert report["header"]["nfields"] == 19
    assert len(report["substeps"]) == 4
    with path.open("ab") as handle:
        handle.write(b"unaccounted")
    with pytest.raises(gate.GateError, match="trailing unregistered bytes"):
        gate.read_oracle_trace(path)


def test_oracle_reader_rejects_truncated_payload(tmp_path):
    path = tmp_path / "trace.bin"
    _write_zero_trace(path)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(gate.GateError, match="truncated record"):
        gate.read_oracle_trace(path)


@pytest.mark.parametrize("name", ("eta_entry", "u_exit"))
def test_planted_frame_controls_make_the_gate_row_red(name):
    oracle = np.zeros((3, 4), dtype=np.float64)
    row = gate.score_frame(
        name, oracle, oracle.copy(), np.ones_like(oracle, dtype=bool), plant=True
    )
    assert row["status"] == "DEBT"
    assert row["normalized_max_abs"] == 1.0


def test_frame_registry_is_complete_and_nonduplicated():
    assert len(gate.FIELDS) == 19
    assert len(set(gate.FIELDS)) == 19
    assert set(gate.STAGGER.values()) == {"T", "U", "V"}
    gate.validate_frame_registry()


def test_frame_registry_fails_closed_on_an_unregistered_frame():
    incomplete = dict(gate.FRAME_REGISTRY)
    incomplete.pop("eta_pgf")
    with pytest.raises(gate.GateError, match="missing=.*eta_pgf"):
        gate.validate_frame_registry(incomplete)


ORACLE_PRESENT = all(path.is_file() for path in (
    gate.DEFAULT_ORACLE, gate.DEFAULT_NEW_ENTRY, gate.DEFAULT_CERTIFIED_ENTRY,
    gate.DEFAULT_TRAJECTORY))
needs_oracle = pytest.mark.skipif(not ORACLE_PRESENT, reason="OVERFLOW oracle dumps absent")


def test_production_predicate_is_the_card_resolution():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import nemo_flux_form_update_active
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    cfg = build_nemo_testcase_card(gate.CASE).recipe.model_config
    assert nemo_flux_form_update_active(cfg) is True
    assert nemo_flux_form_update_active(cfg._replace(momentum_advection="vector_invariant")) is False
    assert nemo_flux_form_update_active(cfg._replace(momentum_time_integrator="rk3")) is False


@needs_oracle
def test_planted_entry_control_exits_nonzero_end_to_end(tmp_path):
    """The REAL gate (both arms, real oracle record) with the entry plant:
    exit 1 (DEBT), the planted frame is the first DEBT of BOTH arms, and the
    top-level status is computed from the production arm's rows."""
    out = tmp_path / "planted.json"
    code = gate.main(["--plant-entry", "--allow-dirty", "--output", str(out)])
    assert code == 1
    report = json.loads(out.read_text())
    assert report["controls"] == {"plant_entry": True, "plant_exit": False}
    for arm in report["arms"].values():
        assert arm["first_over_bar"]["substep"] == 1
        assert arm["first_over_bar"]["frame"] == "eta_entry"
        assert arm["first_over_bar"]["normalized_max_abs"] >= 0.5
    production_rows = [row for substep in report["arms"]["nemo_flux_form_update"]["substeps"]
                       for row in substep["rows"]]
    assert report["status"] == (
        "DEBT" if any(row["status"] == "DEBT" for row in production_rows) else "AT-BAR")
    assert report["production_flux_form_update_active"] is True
    assert report["legoesm_git_sha"].split("-")[0] and len(report["legoesm_git_sha"].split("-")[0]) == 40


@needs_oracle
def test_undetected_plant_exits_two(monkeypatch):
    """If the scorer stops seeing plants, the planted run must exit 2, not 1."""
    real = gate.score_frame

    def blind(name, oracle, candidate, mask, *, plant=False):
        return real(name, oracle, candidate, mask, plant=False)

    monkeypatch.setattr(gate, "score_frame", blind)
    assert gate.main(["--plant-entry", "--allow-dirty"]) == 2


def test_dirty_tree_is_refused_unless_allowed(monkeypatch):
    import legoesm.ocean.fidelity.provenance as provenance

    def dirty(*, allow_dirty=False, repo=None):
        if allow_dirty:
            return "f" * 40 + "-dirty"
        raise RuntimeError("refusing to stamp")

    monkeypatch.setattr(provenance, "git_sha", dirty)
    with pytest.raises(gate.GateError, match="refusing to stamp"):
        gate.git_sha()
    assert gate.git_sha(allow_dirty=True).endswith("-dirty")
