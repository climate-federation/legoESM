"""Controls for the round-333 full-array traldf_iso_lap acquisition."""
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round333_traldf_gate as gate,
)

ROOT = Path(__file__).parents[3]
ACQ = ROOT / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition"


def test_schema_covers_both_tracers_fluxes_operands_and_increment() -> None:
    assert len(gate.FIELDS) == 53
    assert {"zfu", "zfv", "zfw_lower", "zfw_upper", "rhs_increment"} <= set(gate.FIELDS)
    assert {"A11", "A13", "A31", "hmsku", "vmsku", "r1_e1e2t"} <= set(gate.FIELDS)
    assert {"wet", "signed-zero", "checksum", "truncation", "time-level"} <= set(gate.PLANTS)


def test_patches_are_additions_only_and_mark_every_observer() -> None:
    for name in ("traldf_iso_round333.patch", "traldf_iso_scheme_round333.patch"):
        text = (ACQ / name).read_text()
        removed = [line for line in text.splitlines()
                   if line.startswith("-") and not line.startswith("---")]
        assert not removed
        assert "R333_LDF_OBSERVER" in text


def test_writer_uses_full_explicit_extents_checksums_and_rank_names() -> None:
    text = (ACQ / "round333_traldf_writer.F90").read_text()
    assert "jpi,jpj,jpk,jpts" in text
    assert "checksum4(value)" in text
    assert '"_r",I4.4' in text
    assert "narea-1" in text
    assert "STORAGE_SIZE(1._wp)" in text


def test_driver_retains_parent_and_required_controls() -> None:
    text = (ACQ / "run.sh").read_text()
    assert "nemo_testcase_l4_orca2_round328_dynzdf_acquisition" in text
    assert "-fcheck=bounds" in text
    assert "R333_OBSERVER" in text
    assert "--admit-existing" in text
    assert "retention_armed=1" in text
