from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as gate,
)


def _write_frame(path, *, kt: int, stage: int | None, values: np.ndarray) -> None:
    if stage is None:
        magic = "NEMO_L1_ENTRY_1".ljust(16).encode("ascii")
        level = 1 if kt % 2 else 3
        header = struct.pack("=8i", 1, kt, level, gate.NX, gate.NY,
                             gate.NZ, gate.NTR, 64)
    else:
        magic = "NEMO_L1_STAGE_1".ljust(16).encode("ascii")
        level = {1: 3 if kt % 2 else 1, 2: 2,
                 3: 3 if kt % 2 else 1}[stage]
        header = struct.pack("=9i", 1, kt, stage, level, gate.NX,
                             gate.NY, gate.NZ, gate.NTR, 64)
    path.write_bytes(magic + header + values.astype(np.float64).tobytes())


def test_read_state_frame_strips_halos_and_dummy_level(tmp_path):
    n3 = gate.NX * gate.NY * gate.NZ
    n2 = gate.NX * gate.NY
    values = np.arange(4 * n3 + n2, dtype=np.float64)
    path = tmp_path / "entry.bin"
    _write_frame(path, kt=1, stage=None, values=values)

    fields = gate.read_state_frame(path, kt=1, stage=None)

    assert fields["T"].shape == (148, 90, 30)
    assert fields["S"].shape == (148, 90, 30)
    assert fields["u"].shape == (148, 90, 30)
    assert fields["v"].shape == (148, 90, 30)
    assert fields["ssh"].shape == (148, 90)
    raw_t = values[:n3].reshape((gate.NX, gate.NY, gate.NZ), order="F")
    assert fields["T"][0, 0, 0] == raw_t[2, 2, 0]
    assert fields["T"][-1, -1, -1] == raw_t[-3, -3, 29]


def test_read_state_frame_rejects_wrong_stage_level(tmp_path):
    n3 = gate.NX * gate.NY * gate.NZ
    n2 = gate.NX * gate.NY
    values = np.zeros(4 * n3 + n2, dtype=np.float64)
    path = tmp_path / "stage.bin"
    _write_frame(path, kt=2, stage=1, values=values)
    raw = bytearray(path.read_bytes())
    # Header field 4 is the time level; offset is magic + three prior ints.
    raw[16 + 3 * 4:16 + 4 * 4] = struct.pack("=i", 99)
    path.write_bytes(raw)

    with pytest.raises(gate.GateError, match="bad header"):
        gate.read_state_frame(path, kt=2, stage=1)


def test_compare_fields_is_exact_and_ranks_max_abs():
    baseline = {
        "T": np.zeros((2,)),
        "S": np.zeros((2,)),
        "u": np.zeros((2,)),
        "v": np.zeros((2,)),
        "ssh": np.zeros((2,)),
    }
    actual = {name: value.copy() for name, value in baseline.items()}
    actual["S"][1] = 1.0
    actual["ssh"][0] = 2.0

    result = gate.compare_fields(actual, baseline)

    assert result["first_non_bit_field"] == "S"
    assert [row["field"] for row in result["ranked_non_bit_by_max_abs"]] == [
        "ssh", "S"
    ]
    assert result["rows"]["T"]["bit_identical"] is True
    assert result["rows"]["S"]["first_unequal_index"] == [1]


def test_compare_fields_detects_signed_zero_bit_difference():
    baseline = {
        name: np.zeros((1,), dtype=np.float64) for name in gate.FIELD_ORDER
    }
    actual = {name: value.copy() for name, value in baseline.items()}
    actual["T"][0] = np.float64(-0.0)

    result = gate.compare_fields(actual, baseline)

    assert result["first_non_bit_field"] == "T"
    assert result["rows"]["T"]["unequal"] == 1
    assert result["rows"]["T"]["max_abs"] == 0.0


def test_surface_support_names_every_missing_step(tmp_path):
    result = gate.surface_support(tmp_path)
    assert result["trajectory_supported"] is False
    assert result["required"] == 20
    assert result["missing"] == (
        [f"oracle_ocean_surface_input_kt{kt:08d}.bin" for kt in range(1, 11)]
        + [
            f"oracle_ocean_surface_input_rank0001_kt{kt:08d}.bin"
            for kt in range(1, 11)
        ]
    )


def test_entry_support_names_every_missing_slab(tmp_path):
    result = gate.entry_support(tmp_path)
    assert result["trajectory_supported"] is False
    assert result["required"] == 20
    assert result["missing"] == (
        [f"oracle_step_entry_kt{kt:08d}.bin" for kt in range(1, 11)]
        + [
            f"oracle_step_entry_rank0001_kt{kt:08d}.bin"
            for kt in range(1, 11)
        ]
    )


def test_compiled_source_anchors_are_line_rigid(tmp_path):
    lines = [f"line {number}" for number in range(1, 461)]
    lines[441] = "snwice_mass  (:,:) = tmask(:,:,1) * SUM( mass, dim=3 )"
    lines[452] = "zsshadj = glob_2Dsum( 'iceistate', mass ) / area"
    lines[457] = "ssh(:,:,Kmm) = ssh(:,:,Kmm) - zsshadj"
    lines[458] = "ssh(:,:,Kbb) = ssh(:,:,Kbb) - zsshadj"
    path = tmp_path / "iceistate.f90"
    path.write_text("\n".join(lines) + "\n")

    result = gate.validate_compiled_source(path)
    assert result["line_start"] == 442
    assert result["line_end"] == 459

    lines.insert(441, "planted rigid shift")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(gate.GateError, match="source anchor missing"):
        gate.validate_compiled_source(path)


def test_report_provenance_refuses_an_unstampable_tree(monkeypatch):
    sentinel = {"commit": "a" * 40, "clean": True}
    monkeypatch.setattr(gate, "worktree_stamp", lambda: sentinel)
    assert gate.provenance_stamp() is sentinel

    def refuse():
        raise RuntimeError("dirty planted tree")

    monkeypatch.setattr(gate, "worktree_stamp", refuse)
    with pytest.raises(gate.GateError, match="dirty planted tree"):
        gate.provenance_stamp()


def test_acquisition_report_binds_tree_binary_and_compiled_source():
    run_sh = (
        Path(gate.__file__).parent
        / "nemo_testcase_l4_orca2_round1_surface_acquisition/run.sh"
    ).read_text()
    assert '"worktree": worktree_stamp()' in run_sh
    assert 'for name in ("nemo", "compiled_iceistate.f90", "compiled_stprk3.f90")' in run_sh
    assert 'raise SystemExit(f"twin producer differs: {name}")' in run_sh
    assert 'readonly MODE=${1:---run}' in run_sh
    assert 'readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY' in run_sh
    assert 'phase3/orca2_rounds/round5/acquisition' in run_sh
    assert 'readonly EXPECTED_INHERITED_STREAMS=107' in run_sh
    assert 'readonly EXPECTED_TARGET_STREAMS=136' in run_sh
    assert 'oracle_si3_bulk_operands.bin' in run_sh
    assert 'oracle_tke_walk_kt00000002.bin' in run_sh
    assert 'oracle_ocean_surface_input_rank0001_kt%08d.bin' in run_sh
    assert 'oracle_step_entry_rank0001_kt%08d.bin' in run_sh
    assert '"compiled_stprk3.f90"' in run_sh


def test_round6_independent_trajectory_uses_production_step_and_only_bridges_ssh():
    source = Path(gate.__file__).read_text()
    assert "state = state._replace(" in source
    assert "eta=state.eta.replace(data=" in source
    assert "_NEMOWSRK3TestHooks(expose_live_stage_operands=True)" in source
    assert "trace = model.step(" in source
    assert "state = trace.state_after" in source
    assert '"claim_label": claim_label' in source
    assert '"given_nemo_entry_eligibility": eligibility' in source
    assert '"rnf_tsc"' in source


def test_round66_independent_mode_skips_the_ssh_bridge_and_labels_its_owner():
    source = Path(gate.__file__).read_text()
    body = source.split("def candidate_trajectory")[1].split("\ndef ")[0]
    assert "if bridge_ssh:" in body
    assert '"INDEPENDENT_WITH_DECISION52_SSH" if bridge_ssh else "INDEPENDENT"' in body
    assert '"initial_ssh" if first_field == "ssh" else "initial_ts"' in body
    assert "decision52_bridge\": executed_entry if bridge_ssh else None" in body


def test_eos80_coefficient_audit_matches_the_compiled_block_and_can_fail(
        tmp_path):
    """126 numbers re-derived from the compiled file, not trusted by eye."""
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs"
        "/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90")
    if not source.is_file():
        pytest.skip("the round-5 ORCA2 compiled build is not on this host")

    result = gate.validate_eos80_coefficients(source)
    assert result["coefficients_compared"] == 126
    assert (result["density_terms"], result["alpha_terms"],
            result["beta_terms"]) == (52, 35, 35)
    assert result["normalization"]["rdeltaS"] == 20.0
    assert result["normalization"]["r1_S0"] == 1.0 / 40.0

    with pytest.raises(gate.GateError, match="differ from the compiled"):
        gate.validate_eos80_coefficients(source, plant=True)


def test_shared_eos_branch_check_reads_the_compiled_routines():
    """The EOS-80 statement is a coefficient selection, not a second formula."""
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs"
        "/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90")
    if not source.is_file():
        pytest.skip("the round-5 ORCA2 compiled build is not on this host")

    result = gate.validate_shared_eos_branch(source)
    assert result["rab_case_shared"] is True
    assert result["bn2_has_no_eos_branch"] is True


def test_fortran_real_refuses_a_non_literal():
    assert gate._fortran_real("1._wp/40._wp") == 1.0 / 40.0
    assert gate._fortran_real("-2.2856621162e+01_wp") == -22.856621162
    with pytest.raises(gate.GateError, match="Fortran real expression"):
        gate._fortran_real("rn_alpha")


def test_an_unbuilt_statement_is_recorded_not_swallowed():
    """A registered refusal stops the ladder WITHOUT a magnitude.

    Round 8 transcribed the vertex thickness on the fold row, so the refusal
    this used to name no longer exists and the magnitude key is now built from
    whichever registered stop fired.  The invariants it guards are unchanged:
    only a deliberate refusal is caught, every stop is UNMEASURED, and the
    gate exits non-zero.
    """
    source = Path(gate.__file__).read_text()
    assert "except (NotImplementedError, ValueError) as exc:" in source
    assert 'f"UNMEASURED_{blocker[\'status\']}"' in source
    # the stop list is the registry itself, so a stop can never exit zero
    assert "*UNBUILT_STATEMENTS,\n    ):" in source
    # and a refusal is only ever caught at the ONE call site that classifies
    # it -- scope the search to this function's own body, not the rest of the
    # module, or main()'s own handler is counted too.
    body = source.split("def candidate_trajectory")[1].split("\ndef ")[0]
    assert body.count("except ") == 1
    assert "unbuilt_statement_blocker(exc, kt)" in body


def test_only_a_registered_refusal_may_be_labelled_as_an_unbuilt_statement():
    """An unbuilt statement from ANOTHER routine must not borrow a citation.

    Found by the round-7 independent review: the first version labelled every
    refusal as the vertex-thickness gap and cited it to the compiled routine
    that builds that field.  Rounds 8 and 9 each transcribed the registered
    statement and registered the next one, so this exercises the registry
    rather than one name.
    """
    (only,) = gate.UNBUILT_STATEMENTS
    labelled = gate.unbuilt_statement_blocker(
        ValueError(gate.UNBUILT_STATEMENTS[only]["refusal"] +
                   "; call apply_shortwave_penetration(...) to dispatch the "
                   "rgb_chl scheme (needs chl/dz_live/wet_cell)."), kt=1)
    assert labelled["status"] == only == "STOP_PRODUCTION_QSR_RGB_PIPELINE_GAP"
    assert labelled["kt"] == 1
    assert "traqsr.f90:213,258-468" in labelled["nemo_source_citation"]
    # the resolved setting that makes this a gap, not a kernel choice
    assert "ln_qsr_rgb = .true." in labelled["resolved_setting"]

    # Every statement an earlier round transcribed is out of the registry, so
    # its old refusal text cannot be labelled either -- the registry shrinks
    # with each transcription.
    for stale in (
        "nemo_avg4 is not defined for a tripolar fold",
        "lateral_viscosity_operator='nemo_div_curl' needs a lat-lon grid "
        "with a scalar dlon",
        "some other operator is not built",
    ):
        with pytest.raises(gate.GateError, match="unregistered"):
            gate.unbuilt_statement_blocker(NotImplementedError(stale), kt=3)


def test_entry_eligibility_refuses_to_certify_a_non_bit_entry_field():
    """Also from the review: the stop arm certified the twin unconditionally."""
    def bridge(**bits):
        rows = {name: {"bit_identical": bits.get(name, True)}
                for name in gate.FIELD_ORDER}
        return {"rows": rows}

    assert gate.entry_eligibility(bridge()) == (True, "DECISION52_SSH_ONLY")
    # sea-surface height is the ONE field Decision 52 may supply
    assert gate.entry_eligibility(bridge(ssh=False)) == (
        True, "DECISION52_SSH_ONLY")
    for field in ("T", "S", "u", "v"):
        assert gate.entry_eligibility(bridge(**{field: False})) == (
            False, "STOP_INITIAL_T_S_TRANSCRIPTION"), field


def test_round7_ablation_rebuilds_the_unaltered_initial_state():
    """The gate's ablation must go through the card's own helper."""
    source = Path(gate.__file__).read_text()
    assert "apply_hand_alterations=False" in source
    assert "the hand-alteration ablation is vacuous" in source
    assert '"hand_alteration_ablation": ablation' in source
